//! Pseudo-observations of the state occupation probabilities, the time spent
//! in each state and the cumulative transition hazards of a multi-state
//! process, as `survival::pseudo` makes them on a multi-state `survfit`.
//!
//! # Algorithm
//!
//! The process is that of [`multistate`](crate::multistate): intervals
//! `(start, stop]` in a state, ending in a state or in censoring, with an id for
//! each subject. For the occupation probability `p_j(t)` of state `j`, the time
//! in state `j` up to `t`, `A_j(t)`, the integral of `p_j` from the origin, and
//! the cumulative hazard `Lambda_jk(t)` of the transition from `j` to `k`, with
//! `theta_hat` the estimate and `U_g` the influence of cluster `g`, the
//! derivative of the estimate with respect to the case weights of the cluster's
//! rows times the weights, the pseudo-observation is
//!
//! ```text
//! theta_hat + n U_g,        n the number of clusters,
//! ```
//!
//! as in [`pseudo`](crate::pseudo) \[1\]\[2\]. The cluster is the cluster column, else
//! the id column. A stratum column gives one curve for each stratum, a cluster
//! being in one stratum, and `n` is the number of clusters of the stratum.
//!
//! The influence of the probabilities and of the integrals comes from
//! [`multistate::variance`](crate::multistate::variance), whose recursion keeps
//! the influence of the initial distribution when it is estimated from the
//! opening rows, as `survfit` does without `p0`. The integral is formed from the
//! influence at every event time, a chunk of times at a time as in
//! [`PseudoOptions::max_influence_values`](super::PseudoOptions); the influence
//! before the first event time is that of the initial distribution, so the
//! integral starts at the origin of the curves, which the caller gives, since
//! `survfit` takes it from the data in ways that depend on the initial states.
//!
//! The hazard of a transition has the closed form of [`hazard`](super::hazard)
//! with the weight at risk taken in the state the transition leaves: a row
//! in state `j` over `(s, u]` that ends in `k` has
//!
//! ```text
//! dLambda_jk(t)/dw = 1(ends in k at u <= t) / Y_j(u)
//!                    - sum over event times l in (s, min(u, t)] of n_jk(l) / Y_j(l)^2,
//! ```
//!
//! and a row in another state has no influence on it. A move to the state the
//! row is in counts as a transition, as in [`multistate::variance`](crate::multistate::variance).
//!
//! # References
//!
//! 1. Andersen P. K., Klein J. P., Rosthoj S. (2003) Generalised linear models
//!    for correlated pseudo-observations, with applications to multi-state
//!    models, Biometrika 90: 15-27.
//! 2. Therneau T. M. (2024) Methods of the survfit function, vignette `methods`
//!    of the survival package, on residuals of multi-state curves.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "state, row and cluster counts are checked to fit in u32 by the data type"
)]

use epistat_core::InputError;
use rayon::prelude::*;

use crate::check::check_ascending;
use crate::error::{Error, Result};
use crate::multistate::{self, Options, VarianceOptions};
use crate::order::{dense_index, dense_of_present};
use crate::surv::{SurvBuf, SurvData};
use crate::survcheck::{Policy, survcheck};
use crate::timefix::TimeFix;

use super::steps::{Steps, read};

/// What is estimated for each state or transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateTarget {
    /// The probability of being in each state, `pstate` of `survival::pseudo`.
    Occupation,
    /// The expected time in each state from the origin to the time, `auc` of
    /// `survival::pseudo`.
    TimeInState,
    /// The cumulative hazard of each transition that occurs, `cumhaz` of
    /// `survival::pseudo`.
    CumulativeHazard,
}

/// What [`pseudo_states`] does besides the data and the times.
#[derive(Debug, Clone, PartialEq)]
pub struct StateOptions {
    /// What is estimated. It has no default.
    pub target: StateTarget,
    /// The starting time of the curves, `fit$t0` of `survfit`. It has no
    /// default, see [`multistate::VarianceOptions::origin`]. The first exit time
    /// must not be before it, and a time of a time-in-state target must not be.
    pub origin: f64,
    /// The distribution over the states before the first time, as in
    /// [`multistate::Options::initial`]. `None` forms it from the opening rows.
    pub initial: Option<Vec<f64>>,
    /// How near-equal times are treated, `survival`'s rule by default.
    pub timefix: TimeFix,
    /// The most values of influence held at once for the time in state, see
    /// [`PseudoOptions::max_influence_values`](super::PseudoOptions). `None`
    /// sets no bound.
    pub max_influence_values: Option<usize>,
}

impl StateOptions {
    /// Options for a target and the starting time, with the initial
    /// distribution from the data, `survival`'s tie rule and no bound.
    #[must_use]
    pub fn new(target: StateTarget, origin: f64) -> Self {
        Self {
            target,
            origin,
            initial: None,
            timefix: TimeFix::default(),
            max_influence_values: None,
        }
    }
}

/// The estimate of one stratum at the times.
#[derive(Debug, Clone, PartialEq)]
pub struct StateEstimate {
    /// The caller's stratum code, 0 without a stratum column.
    pub group: u32,
    /// The number of clusters.
    pub n_subjects: usize,
    /// The estimate, row-major by time and then by state or transition.
    pub estimate: Vec<f64>,
    /// Whether the time is after the last exit time of the stratum.
    pub beyond: Vec<bool>,
}

/// Pseudo-observations of a multi-state process.
#[derive(Debug, Clone, PartialEq)]
pub struct StatePseudo {
    /// The times, ascending.
    pub times: Vec<f64>,
    /// The number of values for each subject and time: the states, or the
    /// transitions for the cumulative hazard.
    pub width: usize,
    /// The transitions as `(from, to)` state ranks, which are the status codes
    /// less 1, ordered by `to` and then `from`, for the cumulative hazard and
    /// empty otherwise. The columns are in this order. With a stratum column
    /// they are those that occur in any stratum, and a transition that does not
    /// occur in a stratum has the value 0 there.
    pub transitions: Vec<(u32, u32)>,
    /// The stratum of each subject, as in [`StateEstimate::group`].
    pub group: Vec<u32>,
    /// The caller's label of each subject, the cluster column or else the id
    /// column, in order of stratum and then label.
    pub subject: Vec<u32>,
    /// The pseudo-observations, row-major by subject, time and then state or
    /// transition.
    pub values: Vec<f64>,
    /// The estimate of each stratum.
    pub groups: Vec<StateEstimate>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight, which are the rows used.
    pub n_used: usize,
}

/// The pseudo-observations of the subjects of every stratum at `times`.
///
/// See [multi-state processes](crate::pseudo#multi-state-processes) for the
/// targets and the influence they come from.
///
/// # Errors
///
/// An input error when `times` is empty, not finite or not strictly
/// ascending, the origin is not finite, a time in state is asked for before the
/// origin, the first exit time is before the origin, the data are not valid
/// multi-state data (see [`multistate::estimate`]), a cluster has rows in more
/// than one stratum, or the bound on influence values is below the influence at
/// one time.
///
/// # Panics
///
/// Only when a [`SurvData`] invariant is broken: a stop time of a row used that is
/// not among the exit times.
pub fn pseudo_states(
    data: &SurvData<'_>,
    times: &[f64],
    opts: &StateOptions,
) -> Result<StatePseudo> {
    if times.is_empty() {
        return Err(Error::from(InputError::Empty { what: "times" }));
    }
    check_ascending(times, "times", true)?;
    if !opts.origin.is_finite() {
        return Err(Error::invalid("origin must be finite"));
    }
    if opts.target == StateTarget::TimeInState && times[0] < opts.origin {
        return Err(Error::invalid(format!(
            "the time {} of a time in state is before the origin, {}",
            times[0], opts.origin
        )));
    }
    let data = data.clone().with_timefix(opts.timefix)?;
    let n_states = data.followup()?.n_codes() as usize;
    let mut out = StatePseudo {
        times: times.to_vec(),
        width: 0,
        transitions: Vec::new(),
        group: Vec::new(),
        subject: Vec::new(),
        values: Vec::new(),
        groups: Vec::new(),
        n_rows: data.n_rows(),
        n_used: data.n_used(),
    };
    let width_of = |transitions: &[(u32, u32)]| match opts.target {
        StateTarget::CumulativeHazard => transitions.len(),
        _ => n_states,
    };

    // One fit for each stratum, on the rows of that stratum.
    let parts: Vec<(u32, Option<SurvBuf>)> = match data.stratum() {
        None => vec![(0, None)],
        Some(codes) => {
            let levels = data
                .stratum_levels()
                .ok_or_else(|| Error::invalid("a stratum column has no levels"))?;
            let cluster_codes = data.cluster().or_else(|| data.id());
            if let Some(subjects) = cluster_codes {
                let mut owner = vec![u32::MAX; data.n_ids().max(data.n_clusters())];
                for row in data.used_rows() {
                    let slot = &mut owner[subjects[row as usize] as usize];
                    if *slot == u32::MAX {
                        *slot = codes[row as usize];
                    } else if *slot != codes[row as usize] {
                        return Err(Error::invalid(
                            "a cluster has rows in more than one stratum, so its \
                             pseudo-values cannot be collapsed",
                        ));
                    }
                }
            }
            (0..levels.len())
                .filter_map(|level| {
                    let rows: Vec<u32> = (0..data.n_rows() as u32)
                        .filter(|&r| codes[r as usize] as usize == level)
                        .collect();
                    // A stratum with no row of positive weight has no curve.
                    let used = data
                        .weight()
                        .is_none_or(|w| rows.iter().any(|&r| w[r as usize] > 0.0));
                    used.then(|| (levels.code(level).unwrap_or(0), Some((level, rows))))
                })
                .map(|(code, part)| {
                    Ok((
                        code,
                        part.map(|(_, rows)| subset(&data, &rows)).transpose()?,
                    ))
                })
                .collect::<Result<_>>()?
        }
    };
    if parts.is_empty() {
        return Err(Error::invalid("no row has positive weight"));
    }

    let mut results = Vec::new();
    for (code, buffer) in &parts {
        let view = buffer.as_ref().map(SurvBuf::as_data);
        let part = view.as_ref().unwrap_or(&data);
        results.push((*code, one_stratum(part, times, opts)?));
    }
    // The transitions of the strata, in the order of `to` and then `from`.
    if opts.target == StateTarget::CumulativeHazard {
        let mut all: Vec<(u32, u32)> = results
            .iter()
            .flat_map(|(_, r)| r.transitions.iter().copied())
            .collect();
        all.sort_by_key(|&(from, to)| (to, from));
        all.dedup();
        out.transitions = all;
    }
    out.width = width_of(&out.transitions);
    let k = times.len();
    for (code, r) in results {
        let columns: Vec<usize> = if opts.target == StateTarget::CumulativeHazard {
            r.transitions
                .iter()
                .map(|t| out.transitions.iter().position(|a| a == t).expect("listed"))
                .collect()
        } else {
            (0..n_states).collect()
        };
        let place = |source: &[f64], rows: usize| -> Vec<f64> {
            let mut wide = vec![0.0f64; rows * out.width];
            for row in 0..rows {
                for (c, &dest) in columns.iter().enumerate() {
                    wide[row * out.width + dest] = source[row * columns.len() + c];
                }
            }
            wide
        };
        let n = r.labels.len();
        out.values.extend(place(&r.values, n * k));
        out.group.extend(std::iter::repeat_n(code, n));
        out.subject.extend_from_slice(&r.labels);
        out.groups.push(StateEstimate {
            group: code,
            n_subjects: n,
            estimate: place(&r.estimate, k),
            beyond: r.beyond,
        });
    }
    Ok(out)
}

/// The rows of `rows` as a data set of their own, with the times already fixed
/// and the grouping columns given by the caller's codes.
fn subset(data: &SurvData<'_>, rows: &[u32]) -> Result<SurvBuf> {
    let surv = data.followup()?;
    let pick = |v: &[f64]| rows.iter().map(|&r| v[r as usize]).collect::<Vec<f64>>();
    let codes = |dense: Option<&[u32]>, levels: Option<crate::surv::Levels<'_>>| {
        dense.map(|d| {
            rows.iter()
                .map(|&r| {
                    let c = d[r as usize];
                    levels.map_or(c, |l| l.code(c as usize).unwrap_or(c))
                })
                .collect::<Vec<u32>>()
        })
    };
    let mut builder = SurvBuf::builder(
        pick(surv.stop()),
        rows.iter().map(|&r| surv.status()[r as usize]).collect(),
    )
    .n_codes(surv.n_codes());
    if let Some(start) = surv.start() {
        builder = builder.start(pick(start));
    }
    if let Some(w) = data.weight() {
        builder = builder.weight(pick(w));
    }
    if let Some(c) = codes(data.cluster(), data.cluster_levels()) {
        builder = builder.cluster(c);
    }
    if let Some(c) = codes(data.id(), data.id_levels()) {
        builder = builder.id(c);
    }
    if let Some(s) = data.istate() {
        builder = builder.istate(rows.iter().map(|&r| s[r as usize]).collect());
    }
    builder.build()
}

/// What one stratum gives.
struct Stratum {
    labels: Vec<u32>,
    transitions: Vec<(u32, u32)>,
    estimate: Vec<f64>,
    values: Vec<f64>,
    beyond: Vec<bool>,
}

/// The distinct stop times of the rows used, ascending.
fn exit_times(data: &SurvData<'_>) -> Result<Vec<f64>> {
    let stop = data.followup()?.stop();
    let mut times: Vec<f64> = data.used_rows().map(|r| stop[r as usize]).collect();
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| a.total_cmp(b).is_eq());
    Ok(times)
}

fn one_stratum(data: &SurvData<'_>, at: &[f64], opts: &StateOptions) -> Result<Stratum> {
    let k = at.len();
    let grid = exit_times(data)?;
    let Some(&last) = grid.last() else {
        return Err(Error::invalid("no row has positive weight"));
    };
    if grid[0] < opts.origin {
        return Err(Error::invalid(format!(
            "the first exit time {} is before the origin {}",
            grid[0], opts.origin
        )));
    }
    let beyond: Vec<bool> = at.iter().map(|&t| t > last).collect();
    let ms = Options {
        timefix: TimeFix::Exact,
        rate_tolerance: None,
        initial: opts.initial.clone(),
    };
    if opts.target == StateTarget::CumulativeHazard {
        return hazards(data, &grid, at, beyond);
    }
    let estimate = multistate::estimate(data, &grid, &ms)?;
    let states = estimate.n_states;
    let probability = |position: usize| -> &[f64] {
        if position == 0 {
            &estimate.initial
        } else {
            &estimate.probabilities[(position - 1) * states..position * states]
        }
    };

    // The cluster labels, from a first call that asks for nothing.
    let labels = multistate::variance(
        data,
        &grid,
        &ms,
        &VarianceOptions {
            origin: opts.origin,
            horizons: Vec::new(),
            influence_times: Vec::new(),
        },
    )?
    .clusters;
    let n = labels.len();
    let integrate = opts.target == StateTarget::TimeInState;
    let mut knots: Vec<f64> = Vec::new();
    if integrate {
        if opts.origin < grid[0] {
            knots.push(opts.origin);
        }
        knots.extend(grid.iter().copied().filter(|&t| t <= at[k - 1]));
    } else {
        knots.extend_from_slice(at);
    }
    let per_pass = match opts.max_influence_values {
        None => knots.len().max(1),
        Some(limit) => {
            let per_time = n.saturating_mul(states);
            let fit = limit.checked_div(per_time).unwrap_or(0);
            if fit == 0 {
                return Err(Error::invalid(format!(
                    "the influence of {n} clusters in {states} states at one time is \
                     {per_time} values, more than the {limit} allowed"
                )));
            }
            fit
        }
    };
    let influence = read(
        &Steps {
            knots: &knots,
            at,
            integrate,
        },
        &[n],
        states,
        per_pass,
        |chunk| {
            let v = multistate::variance(
                data,
                &grid,
                &ms,
                &VarianceOptions {
                    origin: opts.origin,
                    horizons: Vec::new(),
                    influence_times: chunk.to_vec(),
                },
            )?;
            Ok(vec![v.influence])
        },
    )?;
    let influence = &influence[0];

    // The estimate at each time.
    let mut theta = vec![0.0f64; k * states];
    for (slot, &t) in at.iter().enumerate() {
        let position = grid.partition_point(|&x| x <= t);
        if integrate {
            // From the origin, each piece at its own level up to the time.
            let mut area = vec![0.0f64; states];
            let mut from = opts.origin;
            for p in 0..=position {
                let to = if p < grid.len() { grid[p].min(t) } else { t };
                let to = if p == position { t } else { to };
                if to > from {
                    for (a, &v) in area.iter_mut().zip(probability(p)) {
                        *a += v * (to - from);
                    }
                }
                from = from.max(to);
                if p == position {
                    break;
                }
            }
            theta[slot * states..(slot + 1) * states].copy_from_slice(&area);
        } else {
            theta[slot * states..(slot + 1) * states].copy_from_slice(probability(position));
        }
    }
    let n_f = n as f64;
    let mut values = Vec::with_capacity(n * k * states);
    for c in 0..n {
        for slot in 0..k {
            for j in 0..states {
                values
                    .push(theta[slot * states + j] + n_f * influence[(c * k + slot) * states + j]);
            }
        }
    }
    Ok(Stratum {
        labels,
        transitions: Vec::new(),
        estimate: theta,
        values,
        beyond,
    })
}

const OUTSIDE: u32 = u32::MAX;

/// A used row among the exit times of its stratum.
struct Piece {
    first: usize,
    last: usize,
    from: usize,
    to: u32,
    weight: f64,
    at_risk: bool,
    cluster: usize,
}

/// The cumulative hazards of the transitions that occur, and their influence.
#[expect(clippy::too_many_lines, reason = "one pass of the algorithm")]
fn hazards(data: &SurvData<'_>, grid: &[f64], at: &[f64], beyond: Vec<bool>) -> Result<Stratum> {
    let check = survcheck(data)?;
    check.enforce(Policy::for_data(data))?;
    let surv = data.followup()?;
    let start = surv
        .start()
        .ok_or_else(|| Error::invalid("multi-state estimates need the start of each interval"))?;
    let ids = data
        .id()
        .ok_or_else(|| Error::invalid("multi-state estimates need an id column"))?;
    let states = surv.n_codes() as usize;
    let k = at.len();
    let m = grid.len();
    let rows: Vec<u32> = data.used_rows().collect();
    let (codes, levels) = match (data.cluster(), data.cluster_levels()) {
        (Some(c), l) => (c, l),
        (None, _) => (ids, data.id_levels()),
    };
    let present = dense_of_present(rows.iter().map(|&r| codes[r as usize]));
    let n = present.len();
    let labels: Vec<u32> = present
        .iter()
        .map(|&c| levels.map_or(c, |l| l.code(c as usize).unwrap_or(c)))
        .collect();

    let mut pieces = Vec::with_capacity(rows.len());
    let mut risk_diff = vec![0.0f64; states * (m + 1)];
    let mut moves = std::collections::BTreeMap::<(u32, u32), Vec<f64>>::new();
    for &r in &rows {
        let r = r as usize;
        let from = check.istate[r];
        if from == 0 {
            return Err(Error::invalid(format!(
                "row {r} has no starting state: give istate, or drop the row"
            )));
        }
        let from = (from - 1) as usize;
        let to = match surv.status()[r] {
            0 => OUTSIDE,
            code => u32::try_from(code - 1).unwrap_or(OUTSIDE),
        };
        let weight = data.weight().map_or(1.0, |w| w[r]);
        let first = grid.partition_point(|&t| t <= start[r]);
        let last = grid
            .binary_search_by(|t| t.total_cmp(&surv.stop()[r]))
            .expect("the stop time of a used row is an exit time");
        let at_risk = first <= last;
        if at_risk {
            risk_diff[from * (m + 1) + first] += weight;
            risk_diff[from * (m + 1) + last + 1] -= weight;
            if (to as usize) < states {
                moves
                    .entry((from as u32, to))
                    .or_insert_with(|| vec![0.0; m])[last] += weight;
            }
        }
        pieces.push(Piece {
            first,
            last,
            from,
            to,
            weight,
            at_risk,
            cluster: dense_index(&present, codes[r]),
        });
    }
    let mut risk = vec![0.0f64; states * m];
    for j in 0..states {
        let mut run = 0.0f64;
        for l in 0..m {
            run += risk_diff[j * (m + 1) + l];
            risk[j * m + l] = run;
        }
    }
    let mut transitions: Vec<(u32, u32)> = moves.keys().copied().collect();
    transitions.sort_by_key(|&(from, to)| (to, from));
    let width = transitions.len();
    // The hazard of each transition at each exit time, and the prefix sums of
    // `n / Y^2`.
    let mut increment = vec![0.0f64; width * m];
    let mut prefix = vec![0.0f64; width * (m + 1)];
    for (t, &(from, to)) in transitions.iter().enumerate() {
        let counted = &moves[&(from, to)];
        for l in 0..m {
            let y = risk[from as usize * m + l];
            if counted[l] > 0.0 {
                increment[t * m + l] = counted[l] / y;
                prefix[t * (m + 1) + l + 1] = prefix[t * (m + 1) + l] + counted[l] / (y * y);
            } else {
                prefix[t * (m + 1) + l + 1] = prefix[t * (m + 1) + l];
            }
        }
    }
    let through: Vec<usize> = at
        .iter()
        .map(|&t| grid.partition_point(|&x| x <= t))
        .collect();
    let mut theta = vec![0.0f64; k * width];
    for t in 0..width {
        for (slot, &th) in through.iter().enumerate() {
            let mut h = 0.0f64;
            for l in 0..th {
                h += increment[t * m + l];
            }
            theta[slot * width + t] = h;
        }
    }

    // Rows by cluster, in row order.
    let mut offsets = vec![0usize; n + 1];
    for p in &pieces {
        offsets[p.cluster + 1] += 1;
    }
    for c in 0..n {
        offsets[c + 1] += offsets[c];
    }
    let mut fill = offsets.clone();
    let mut by_cluster = vec![0usize; pieces.len()];
    for (i, p) in pieces.iter().enumerate() {
        by_cluster[fill[p.cluster]] = i;
        fill[p.cluster] += 1;
    }
    let by_from: Vec<Vec<usize>> = (0..states)
        .map(|j| {
            transitions
                .iter()
                .enumerate()
                .filter(|&(_, &(from, _))| from as usize == j)
                .map(|(t, _)| t)
                .collect()
        })
        .collect();
    let n_f = n as f64;
    let values: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|c| {
            let mut cell = vec![0.0f64; k * width];
            for &i in &by_cluster[offsets[c]..offsets[c + 1]] {
                let p = &pieces[i];
                if !p.at_risk {
                    continue;
                }
                for &t in &by_from[p.from] {
                    let (_, to) = transitions[t];
                    for (slot, &th) in through.iter().enumerate() {
                        let mut v = -(prefix[t * (m + 1) + (p.last + 1).min(th)]
                            - prefix[t * (m + 1) + p.first.min(th)]);
                        if p.to == to && p.last < th {
                            v += 1.0 / risk[p.from * m + p.last];
                        }
                        cell[slot * width + t] += p.weight * v;
                    }
                }
            }
            for slot in 0..k {
                for t in 0..width {
                    cell[slot * width + t] = theta[slot * width + t] + n_f * cell[slot * width + t];
                }
            }
            cell
        })
        .collect();
    Ok(Stratum {
        labels,
        transitions,
        estimate: theta,
        values,
        beyond,
    })
}
