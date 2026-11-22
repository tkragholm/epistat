//! State occupation probabilities of a multi-state process, by the Aalen-Johansen
//! product integral, their infinitesimal jackknife standard errors, and a
//! clustered bootstrap of them.
//!
//! # Algorithm
//!
//! The process has `S` states. Each row is an interval `(start, stop]` spent in
//! a state `from` and ending in a state `to`, with a weight. For an event time
//! `t` the weight at risk in state `i` and the weight moving from `i` to `j` are
//!
//! ```text
//! Y_i(t)  = sum of weight over intervals with start < t <= stop in state i
//! n_ij(t) = sum of weight over intervals with stop == t, start < t,
//!           moving from i to j (j != i)
//! ```
//!
//! The transition matrix at `t` has off-diagonal entries `A_ij = n_ij / Y_i`
//! and diagonal `A_ii = max(0, 1 - sum over j != i of A_ij)`. A state with
//! nobody at risk has an identity row. The occupation probabilities `p(t)` start
//! at the distribution `p(0)` over the opening states of the individuals and
//! advance by the Aalen-Johansen product integral \[1\]\[2\]
//!
//! ```text
//! p_j(t) = sum over i of p_i(t-) A_ij(t).
//! ```
//!
//! The `start < t` clause on the transitions matters. An interval with
//! `start == stop == t` satisfies `stop == t` but not `start < t <= stop`, so
//! without the clause it would be counted as leaving a state it was never
//! counted as occupying, the diagonal of `A` would go negative and the running
//! product would oscillate with growing amplitude.
//!
//! A state rank at or above `S` means outside the state space. It is used for a
//! censored interval, or a state deliberately left out, and it is counted
//! into neither the risk set nor the transitions.
//!
//! # Input
//!
//! [`estimate`] and [`bootstrap`] read a [`SurvData`]. Its states are the
//! codes `1..=n_codes`, so `S` is `n_codes` and state rank `r` is code `r + 1`.
//! The state a row starts in is its `istate`, as resolved by
//! [`survcheck`]: the given `istate`, or the state
//! the earlier rows of the subject led to when there is none. The state it
//! ends in is its `status`. A status of 0 is censoring, and it is the one
//! way to say "outside the state space": a move to a state that is left out of
//! the analysis is recorded as status 0, which counts the row at risk until
//! its stop and as a move to nowhere. A row that starts in a left-out state is
//! left out by the caller, since a state without a code has no rank. A row
//! whose starting state is unknown (0) is an error, because it has no state to
//! be at risk in.
//!
//! The data must have an id column and a start column, and [`bootstrap`] a
//! cluster column. A subject's rows are checked with
//! [`survcheck`] and the policy for multi-state
//! data, so overlapping rows, jumps and teleports are errors, and gaps in the
//! follow-up of a subject are allowed. The times first go through the tie rule
//! of [`TimeFix`], `survival::aeqSurv` by default, as
//! `survfit` applies it. Rows of weight 0 are not used. The result reports
//! `n_rows`, all rows, and `n_used`, the rows of positive weight. The opening
//! row of a subject is its first row in time order, by stop and then start,
//! whatever its weight, so a subject whose first row has weight 0 adds nothing
//! to the initial distribution.
//!
//! The restricted expected time in each state up to a horizon `h` is the integral
//! of `p` from an origin `o` to `h`. Since `p` is a step function, the integral
//! is a sum of rectangles, and the segment from the origin to the first event
//! time after it uses the starting distribution.
//!
//! # Agreement with R
//!
//! The probabilities and the weight at risk agree with `survival::survfit` on
//! `Surv(start, stop, state)` with `id` and `istate`, given the initial
//! distribution `survfit` reports as `p0`. [`integrate`] agrees with the
//! restricted mean time in state of `summary(fit, rmean = tau)`, which sums the
//! probability at each event time times the time to the next, from the starting
//! time to `tau`. A time exactly at `tau` adds nothing, and a `tau` past the last
//! event time holds the last probabilities. `survfit` starts at time 0 when
//! everyone begins in one state or `start.time` is given, and otherwise at the
//! first transition time, so `origin` must be the starting time `survfit` used
//! (`fit$t0`) to get the same totals.
//!
//! # Summation order
//!
//! The accumulations of `Y` and `n` run over the intervals in input order, one
//! interval at a time, and the vector-matrix product and the total at risk use
//! compensated sums. A difference array over the sorted times would turn the
//! `O(T rows)` accumulation into `O(rows log T)`, but it sums partial sums
//! instead of the weights in range and differs in the last digits, by about
//! `rows * eps`. The accumulation here keeps the order of a direct
//! implementation, in parallel over times, and so gives the same bits. When the
//! times are ascending a precomputed plan finds, by binary search, the range of
//! times each interval is at risk at and the time it leaves at, and adds
//! the weights in the same order, so the sums are the same numbers. The plan
//! depends on the intervals and the times only, so a bootstrap builds it once.
//!
//! # The variance
//!
//! [`variance`] gives the standard errors that `survfit` reports for a
//! multi-state fit: of the state probabilities (`std.err`), of the cumulative
//! transition hazards (`std.chaz`), of the restricted time in each state from
//! the origin (`std.auc`), and of the initial distribution when it is estimated
//! (`se0`), by the infinitesimal jackknife \[4\]\[5\]. The influence of a row is the
//! derivative of the estimate with respect to its case weight times the weight,
//! the rows of a cluster, the cluster column or else the id column, are summed,
//! and the variance is the sum of squares over clusters. `survfit` fixes the
//! origin of the restricted times from the data in a way that depends on the
//! initial states, so it is an argument here, as it is for [`integrate`].
//! The standard error of the restricted time to a horizon between event times is
//! exact, the one `summary(fit, rmean =)` gives for a fit made with
//! `influence = TRUE`, and not the interpolation it makes without. The recursion
//! and how its memory is bounded are described in `influence`: clusters are
//! processed in blocks, so the influence of every cluster at every time is never
//! held. The influence at times chosen by the caller is returned.
//!
//! # The bootstrap
//!
//! [`bootstrap`] reweights the intervals by random cluster
//! weights, Dirichlet(1, ..., 1) scaled to mean 1 \[3\], one weight per cluster
//! drawn from a seed per replicate, and repeats the estimate. The point estimate
//! and every replicate go through one function, so an interval is
//! around what was reported. Replicates run in parallel and each depends on its
//! seed only, so the result does not depend on the number of threads. The
//! ranking of rows, the grid of event times and the opening state of each
//! individual are the same in every replicate, and the plan is built once.
//!
//! # Edge cases
//!
//! The estimate fails with an error when more weight leaves a state than was at
//! risk in it, beyond a tolerance. That cannot happen for well-formed data,
//! because every interval counted as moving was counted as at risk with the
//! same weight, and the check is there because the failure is otherwise silent:
//! the probabilities still sum to one and only their range gives it away. A
//! distribution at the origin with no weight is all zeros.
//!
//! # References
//!
//! 1. Aalen OO, Johansen S. An empirical transition matrix for non-homogeneous
//!    Markov chains based on censored observations. Scand J Stat
//!    1978;5(3):141-50.
//! 2. Andersen PK, Borgan O, Gill RD, Keiding N. Statistical models based on
//!    counting processes. New York: Springer; 1993.
//! 3. Rubin DB. The Bayesian bootstrap. Ann Stat 1981;9(1):130-4.
//!    doi:10.1214/aos/1176345338
//! 4. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000.
//! 5. Therneau TM. Methods of the survfit function, vignette `methods` of the
//!    survival package, on the infinitesimal jackknife for the Aalen-Johansen
//!    estimate.

#![expect(
    clippy::cast_possible_truncation,
    reason = "a plan is built only when the row and table counts fit in u32"
)]

use std::borrow::Cow;

use epistat_rng::dirichlet_weights;
use epistat_stats::sum::compensated;
use rayon::prelude::*;

use crate::check::{check_ascending, check_lengths};
use crate::error::{Error, Result};
use crate::order::{dense_index, dense_of_present};
use crate::surv::SurvData;
use crate::survcheck::{Policy, survcheck};
use crate::timefix::TimeFix;

pub(crate) mod influence;

/// The intervals as four slices of the same length, with the state ranks
/// of the data. A rank at or above the number of states is outside the space.
#[derive(Debug, Clone, Copy)]
struct Rows<'a> {
    starts: &'a [f64],
    stops: &'a [f64],
    from_states: &'a [u32],
    to_states: &'a [u32],
}

impl Rows<'_> {
    const fn len(&self) -> usize {
        self.starts.len()
    }
}

/// The rank that stands for a state outside the space.
const OUTSIDE: u32 = u32::MAX;

/// The occupation probabilities at every event time, and the weight at risk.
struct Occupation {
    /// `time * state`, row-major: the probability of each state after each time.
    probabilities: Vec<f64>,
    /// The total weight at risk at each time.
    at_risk: Vec<f64>,
}

/// Which cells of the risk and transition tables each row adds its weight to.
///
/// With the times ascending, the times a row is at risk at, and the time it
/// leaves at, are ranges found by binary search, and they depend on the
/// intervals and the times only, so a bootstrap finds them once for all its
/// replicates. Each list is in row order. Walking it adds the weights to every
/// cell in the order of the two loops over the rows in the direct path, so each
/// sum is the same f64.
struct Plan {
    risk: Vec<RiskEntry>,
    moves: Vec<MoveEntry>,
}

struct RiskEntry {
    row: u32,
    origin: u32,
    first: u32,
    last: u32,
}

struct MoveEntry {
    row: u32,
    /// `origin * states + destination` within a time's block.
    slot: u32,
    first: u32,
    last: u32,
}

impl Plan {
    /// `None` when the times are not ascending and free of NaN, or a table
    /// would not be indexed by `u32`; the sums then run over every time.
    fn new(intervals: Rows<'_>, times: &[f64], states: usize) -> Option<Self> {
        if !times.windows(2).all(|pair| pair[0] <= pair[1])
            || u32::try_from(intervals.starts.len()).is_err()
            || u32::try_from(times.len() * states * states).is_err()
        {
            return None;
        }
        let mut risk = Vec::new();
        let mut moves = Vec::new();
        for row in 0..intervals.starts.len() {
            let (start, stop) = (intervals.starts[row], intervals.stops[row]);
            let origin = intervals.from_states[row] as usize;
            if origin >= states || start.is_nan() || stop.is_nan() {
                continue;
            }
            // `start < time <= stop`.
            let first = times.partition_point(|&time| time <= start);
            let last = times.partition_point(|&time| time <= stop);
            if first < last {
                risk.push(RiskEntry {
                    row: row as u32,
                    origin: origin as u32,
                    first: first as u32,
                    last: last as u32,
                });
            }
            let destination = intervals.to_states[row] as usize;
            // `time == stop` and `start < time`: all the times equal to `stop`
            // qualify when `start < stop`, and none otherwise.
            if destination < states && origin != destination && start < stop {
                let first = times.partition_point(|&time| time < stop);
                if first < last {
                    moves.push(MoveEntry {
                        row: row as u32,
                        slot: (origin * states + destination) as u32,
                        first: first as u32,
                        last: last as u32,
                    });
                }
            }
        }
        Some(Self { risk, moves })
    }
}

/// Weight at risk in each state, and weight moving between them, at every time.
///
/// A state rank of `states` or above means "outside the space", and is counted
/// into neither the risk set nor the transitions.
///
/// Returns `at_risk` as `time * state` and `moves` as `time * state * state`,
/// both row-major.
///
/// The `start < time` clause on the transitions keeps a zero-length interval
/// from being credited as leaving a state it was never at risk in.
// `stops[index] == time` is exact on purpose: the times are the intervals' own
// stop times, and a row leaves at its stop or it does not.
#[expect(clippy::float_cmp, reason = "a row leaves at its own stop time")]
fn accumulate(
    starts: &[f64],
    stops: &[f64],
    from_states: &[u32],
    to_states: &[u32],
    weights: &[f64],
    times: &[f64],
    states: usize,
    plan: Option<&Plan>,
) -> (Vec<f64>, Vec<f64>) {
    let rows = starts.len();

    let mut at_risk = vec![0.0f64; times.len() * states];
    let mut moves = vec![0.0f64; times.len() * states * states];

    if let Some(plan) = plan {
        for entry in &plan.risk {
            let weight = weights[entry.row as usize];
            for position in entry.first as usize..entry.last as usize {
                at_risk[position * states + entry.origin as usize] += weight;
            }
        }
        for entry in &plan.moves {
            let weight = weights[entry.row as usize];
            for position in entry.first as usize..entry.last as usize {
                moves[position * states * states + entry.slot as usize] += weight;
            }
        }
        return (at_risk, moves);
    }

    // Parallel over times. Each time owns one risk row and one transition
    // block and reads the intervals without writing them, so there is
    // nothing to reconcile between threads, and each accumulation runs over
    // the intervals in input order.
    at_risk
        .par_chunks_mut(states)
        .zip(moves.par_chunks_mut(states * states))
        .zip(times.par_iter())
        .for_each(|((risk, block), &time)| {
            for index in 0..rows {
                let origin = from_states[index] as usize;
                if starts[index] < time && time <= stops[index] && origin < states {
                    risk[origin] += weights[index];
                }
            }
            // A second pass rather than one fused with the first, so each
            // accumulation runs over the intervals in the same order.
            for index in 0..rows {
                let origin = from_states[index] as usize;
                let destination = to_states[index] as usize;
                if stops[index] == time
                    && starts[index] < time
                    && origin < states
                    && destination < states
                    && origin != destination
                {
                    block[origin * states + destination] += weights[index];
                }
            }
        });
    (at_risk, moves)
}

/// The occupation probabilities at every time of `times`, and the weight at
/// risk at each.
///
/// `weights` is one weight per interval, `times` the event times, usually
/// ascending, `initial` the distribution over the `states` states before the
/// first time, and `rate_tolerance` how far the summed leaving rate of a state
/// may exceed 1 before the data is refused. The accumulations and the product
/// into transition matrices happen in one call.
///
/// The arithmetic follows the textbook order throughout: a state with nobody
/// at risk gets an identity row rather than a division, the diagonal is
/// `1 - sum(rates)` clamped at zero, and the running product applies the matrix
/// on the right, so `p_j = sum_i p_i A_ij`. Reassociating any of those moves the
/// last digits.
///
/// # Errors
///
/// [`Error::Input`] when `weights` does not hold one weight per interval,
/// `initial` does not hold one entry per state, or more weight leaves a state
/// than was at risk in it.
fn occupation(
    intervals: Rows<'_>,
    weights: &[f64],
    times: &[f64],
    initial: &[f64],
    states: usize,
    rate_tolerance: f64,
) -> Result<Occupation> {
    check_lengths(&[("starts", intervals.len()), ("weights", weights.len())])?;
    if initial.len() != states {
        return Err(Error::input(
            "the initial distribution must have one entry per state",
        ));
    }

    let plan = Plan::new(intervals, times, states);
    let (probabilities, at_risk) = product_integral(
        intervals,
        weights,
        times,
        initial,
        states,
        rate_tolerance,
        true,
        plan.as_ref(),
    )?;
    Ok(Occupation {
        probabilities,
        at_risk,
    })
}

/// The distribution at the origin, for one weighting.
///
/// The total is a compensated sum. It is a normalising denominator, so a
/// last-place difference here would move every probability of every curve.
///
/// A total weight of zero, which leaves no distribution to normalise, is an
/// [`Error::Input`].
fn initial_distribution(
    first_states: &[i32],
    first_positions: &[u32],
    weights: &[f64],
    states: usize,
) -> Result<Vec<f64>> {
    let mut occupancy = vec![0.0f64; states];
    for (&opening, &position) in first_states.iter().zip(first_positions.iter()) {
        // A negative rank is a state outside the declared space.
        if let Ok(state) = usize::try_from(opening) {
            occupancy[state] += weights[position as usize];
        }
    }
    let total = compensated(occupancy.iter().copied());
    if total <= 0.0 {
        return Err(Error::input(
            "the opening intervals carry no weight in the state space, so there is no initial distribution",
        ));
    }
    Ok(occupancy.iter().map(|value| value / total).collect())
}

/// The product integral over `times`: `states` probabilities per time, and with
/// `with_risk` the total weight at risk at each time, otherwise an empty
/// vector. The point estimate and every bootstrap replicate come through here.
fn product_integral(
    intervals: Rows<'_>,
    weights: &[f64],
    times: &[f64],
    initial: &[f64],
    states: usize,
    rate_tolerance: f64,
    with_risk: bool,
    plan: Option<&Plan>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    // The accumulations for every time at once.
    let (at_risk, moves) = accumulate(
        intervals.starts,
        intervals.stops,
        intervals.from_states,
        intervals.to_states,
        weights,
        times,
        states,
        plan,
    );

    // The product integral is sequential in time by construction, so no rayon
    // here. It is `T * S^2` and `T` is bounded; the parallelism that pays is
    // over replicates, one level up.
    let mut probabilities = initial.to_vec();
    let mut next = vec![0.0f64; states];
    let mut transition = vec![0.0f64; states * states];
    let mut out = Vec::with_capacity(times.len() * states);
    let mut totals = Vec::with_capacity(if with_risk { times.len() } else { 0 });

    for (position, &time) in times.iter().enumerate() {
        let risk = &at_risk[position * states..(position + 1) * states];
        let block = &moves[position * states * states..(position + 1) * states * states];
        transition.fill(0.0);
        for i in 0..states {
            if risk[i] <= 0.0 {
                transition[i * states + i] = 1.0;
                continue;
            }
            let mut leaving = 0.0f64;
            for j in 0..states {
                if i == j {
                    continue;
                }
                let rate = block[i * states + j] / risk[i];
                transition[i * states + j] = rate;
                leaving += rate;
            }
            if leaving > 1.0 + rate_tolerance {
                return Err(Error::input(format!(
                    "more weight left state {i} at time {time} than was at risk \
                     in it ({leaving:.4}); the interval data is malformed"
                )));
            }
            transition[i * states + i] = (1.0 - leaving).max(0.0);
        }
        for j in 0..states {
            next[j] =
                compensated((0..states).map(|i| probabilities[i] * transition[i * states + j]));
        }
        probabilities.copy_from_slice(&next);
        out.extend_from_slice(&probabilities);
        if with_risk {
            totals.push(compensated(risk.iter().copied()));
        }
    }
    Ok((out, totals))
}

/// The restricted mean time in each state up to `horizon`, from one replicate's
/// curve.
///
/// The integral of the occupation probability, a sum of rectangles because the
/// curve is a step function. The segment from the origin to the first event
/// time uses the starting distribution.
///
/// `curve` holds one row of `states` probabilities for each of `times`, row
/// major, the row at a time being the distribution from that time on. Rows at
/// or before `origin` only set the current distribution. The result has one
/// total per state, all zero when `horizon` is not after `origin`.
#[must_use]
pub fn integrate(
    curve: &[f64],
    times: &[f64],
    initial: &[f64],
    states: usize,
    horizon: f64,
    origin: f64,
) -> Vec<f64> {
    let mut totals = vec![0.0f64; states];
    if horizon <= origin {
        return totals;
    }
    let mut occupancy = initial.to_vec();
    let mut previous = origin;
    for (position, &time) in times.iter().enumerate() {
        let row = &curve[position * states..(position + 1) * states];
        if time <= origin {
            occupancy.copy_from_slice(row);
            continue;
        }
        let boundary = time.min(horizon);
        let width = boundary - previous;
        if width > 0.0 {
            for state in 0..states {
                totals[state] += occupancy[state] * width;
            }
        }
        previous = boundary;
        if previous >= horizon {
            return totals;
        }
        occupancy.copy_from_slice(row);
    }
    let width = horizon - previous;
    if width > 0.0 {
        for state in 0..states {
            totals[state] += occupancy[state] * width;
        }
    }
    totals
}

/// The clustered bootstrap of the occupation probabilities, replicates and all.
///
/// Replicate `r` draws one weight per cluster from `seeds[r]` and multiplies
/// the base weight `base[i]` of each interval by the weight of its cluster
/// `cluster_codes[i]`. The opening distribution is formed from the weights of
/// the intervals at `first_positions`, which open the follow-up of each
/// individual, in the states `first_states`, where a negative rank means a
/// state outside the space.
///
/// `horizons` selects what comes back. Empty returns the occupation
/// probabilities, `replicate * time * state`. Non-empty returns the restricted
/// expected times from `origin`, `replicate * horizon * state`. Returns the
/// replicates concatenated in seed order.
///
/// # Errors
///
/// [`Error::Input`] for an empty state space, a slice of the wrong length, a
/// cluster code not below `n_clusters`, a position outside the intervals, an
/// opening state not below `states`, a replicate in which the opening
/// intervals carry no weight inside the state space (no initial distribution
/// exists), or when more weight leaves a state than was at risk in it in some
/// replicate.
fn bootstrap_rows(
    intervals: Rows<'_>,
    base: &[f64],
    cluster_codes: &[u32],
    first_positions: &[u32],
    first_states: &[i32],
    times: &[f64],
    horizons: &[f64],
    origin: f64,
    seeds: &[u64],
    n_clusters: usize,
    states: usize,
    rate_tolerance: f64,
) -> Result<Vec<f64>> {
    if states == 0 {
        return Err(Error::input(
            "an empty state space has no occupation probabilities to estimate",
        ));
    }
    check_lengths(&[
        ("starts", intervals.len()),
        ("base", base.len()),
        ("cluster_codes", cluster_codes.len()),
    ])?;
    check_lengths(&[
        ("first_positions", first_positions.len()),
        ("first_states", first_states.len()),
    ])?;
    if cluster_codes
        .iter()
        .any(|&code| code as usize >= n_clusters)
    {
        return Err(Error::input("cluster_codes must be below n_clusters"));
    }
    if first_positions
        .iter()
        .any(|&position| position as usize >= intervals.len())
    {
        return Err(Error::input("first_positions index past the intervals"));
    }
    if first_states
        .iter()
        .any(|&state| usize::try_from(state).is_ok_and(|rank| rank >= states))
    {
        return Err(Error::input("first_states must be below the state count"));
    }

    let plan = Plan::new(intervals, times, states);
    let per_replicate: Vec<Vec<f64>> = seeds
        .par_iter()
        .map_init(
            || (Vec::<f64>::new(), Vec::<f64>::new()),
            |(draw, weights), &seed| {
                dirichlet_weights(n_clusters, seed, draw);
                // The replicate's row weights: the base weight times its
                // cluster's draw.
                weights.clear();
                weights.extend(
                    base.iter()
                        .zip(cluster_codes.iter())
                        .map(|(&weight, &code)| weight * draw[code as usize]),
                );
                let weights: &[f64] = weights;
                let initial = initial_distribution(first_states, first_positions, weights, states)?;
                let (curve, _) = product_integral(
                    intervals,
                    weights,
                    times,
                    &initial,
                    states,
                    rate_tolerance,
                    false,
                    plan.as_ref(),
                )?;
                if horizons.is_empty() {
                    return Ok(curve);
                }
                let mut out = Vec::with_capacity(horizons.len() * states);
                for &horizon in horizons {
                    out.extend_from_slice(&integrate(
                        &curve, times, &initial, states, horizon, origin,
                    ));
                }
                Ok(out)
            },
        )
        .collect::<Result<_>>()?;

    Ok(per_replicate.concat())
}

/// Settings of [`estimate`] and [`bootstrap`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Options {
    /// How times that differ by rounding error are treated before the
    /// estimate. `survival::aeqSurv` by default, as `survfit` applies it.
    pub timefix: TimeFix,
    /// How far the summed leaving rate of a state may exceed 1 before the
    /// data is refused as malformed. It bounds a check, not a result. `None`,
    /// the default, derives it from the rounding of well-formed data: the
    /// weight at risk and the weight moving are plain sums over at most `m`
    /// rows, each within `m` units of rounding of its exact value, and the
    /// leaving rate adds one division per state, so it exceeds 1 by at most
    /// `(2 m + S) ε` for `S` states. `m` is taken as the number of rows used.
    /// A given value must be finite and not negative.
    pub rate_tolerance: Option<f64>,
    /// The distribution over the states before the first time, `n_codes`
    /// entries that are not negative and sum to 1. `None`, the default, takes
    /// it from the opening rows of the subjects, weighted. [`bootstrap`]
    /// forms it in each replicate from the replicate's weights, so it refuses
    /// a distribution given here.
    pub initial: Option<Vec<f64>>,
}

/// The tolerance of the leaving-rate check: the one given, checked, or the
/// rounding bound of [`Options::rate_tolerance`] for `rows` rows and `states`
/// states.
fn rate_tolerance(opts: &Options, rows: usize, states: usize) -> Result<f64> {
    match opts.rate_tolerance {
        Some(tolerance) if tolerance.is_finite() && tolerance >= 0.0 => Ok(tolerance),
        Some(tolerance) => Err(Error::input(format!(
            "rate_tolerance must be finite and not negative, got {tolerance}"
        ))),
        None => Ok((2 * rows + states) as f64 * f64::EPSILON),
    }
}

/// The occupation probabilities at every time, with the rows behind them.
#[derive(Debug, Clone, PartialEq)]
pub struct Estimate {
    /// `time * state`, row-major: the probability of each state after each
    /// time. State `k` is status code `k + 1`.
    pub probabilities: Vec<f64>,
    /// The total weight at risk at each time.
    pub at_risk: Vec<f64>,
    /// The distribution over the states before the first time, as given or as
    /// formed from the opening rows.
    pub initial: Vec<f64>,
    /// The number of states, `n_codes` of the data.
    pub n_states: usize,
    /// The number of rows of the data, including rows of weight 0.
    pub n_rows: usize,
    /// The number of rows of positive weight, which are the rows used.
    pub n_used: usize,
}

/// The replicates of a clustered bootstrap, with the rows behind them.
#[derive(Debug, Clone, PartialEq)]
pub struct Bootstrap {
    /// The replicates concatenated in seed order. Without horizons each is
    /// `time * state`, the occupation probabilities. With horizons each is
    /// `horizon * state`, the restricted expected time in each state from the
    /// origin to the horizon.
    pub values: Vec<f64>,
    /// The number of replicates, one per seed.
    pub n_replicates: usize,
    /// The number of states, `n_codes` of the data.
    pub n_states: usize,
    /// The number of rows of the data, including rows of weight 0.
    pub n_rows: usize,
    /// The number of rows of positive weight, which are the rows used.
    pub n_used: usize,
}

/// What the data give the shared code, once checked: the ranks of the states,
/// the opening row of each subject and the clusters, for the rows used.
struct Prepared<'a> {
    data: SurvData<'a>,
    /// The rows used, or `None` when every row is used.
    index: Option<Vec<u32>>,
    from_states: Vec<u32>,
    to_states: Vec<u32>,
    /// Positions among the rows used, one per subject whose opening row is
    /// used, in the order of the subjects' dense ids.
    first_positions: Vec<u32>,
    first_states: Vec<i32>,
    clusters: Option<Vec<u32>>,
    n_states: usize,
}

/// The column `values` over the rows used.
fn gather<'a>(values: &'a [f64], index: Option<&[u32]>) -> Cow<'a, [f64]> {
    index.map_or(Cow::Borrowed(values), |rows| {
        Cow::Owned(rows.iter().map(|&row| values[row as usize]).collect())
    })
}

impl<'a> Prepared<'a> {
    /// Apply the tie rule, check the follow-up of each subject and read the
    /// states of the rows used.
    fn new(data: &SurvData<'a>, timefix: TimeFix, clustered: bool) -> Result<Self> {
        let data = data.clone().with_timefix(timefix)?;
        let check = survcheck(&data)?;
        check.enforce(Policy::for_data(&data))?;
        let surv = data.followup()?;
        let stop = surv.stop();
        let start = surv.start().ok_or_else(|| {
            Error::input("multi-state estimates need the start of each interval, the start column")
        })?;
        if data.n_used() == 0 {
            return Err(Error::input(
                "no row has positive weight, so there is nothing to estimate",
            ));
        }
        let ids = data.id().ok_or_else(|| {
            Error::input(
                "multi-state estimates need an id column to know which rows belong together",
            )
        })?;
        let n_states = surv.n_codes() as usize;
        let index = data.used_index();
        let every: Vec<u32>;
        let rows: &[u32] = if let Some(used) = &index {
            used
        } else {
            every = (0..data.n_rows() as u32).collect();
            &every
        };

        let mut from_states = Vec::with_capacity(rows.len());
        let mut to_states = Vec::with_capacity(rows.len());
        for &row in rows {
            let state = check.istate[row as usize];
            if state == 0 {
                return Err(Error::input(format!(
                    "row {row} has no starting state: give istate, or drop the row"
                )));
            }
            from_states.push(state - 1);
            to_states.push(match surv.status()[row as usize] {
                0 => OUTSIDE,
                code => u32::try_from(code - 1).expect("status is at least 1 here"),
            });
        }

        // The opening row of a subject: the first by stop, then start, then
        // row number, the order survcheck reads the rows in.
        let mut opening = vec![u32::MAX; data.n_ids()];
        for row in 0..data.n_rows() {
            let slot = &mut opening[ids[row] as usize];
            let earlier = *slot == u32::MAX
                || (stop[row], start[row])
                    .partial_cmp(&(stop[*slot as usize], start[*slot as usize]))
                    .is_some_and(std::cmp::Ordering::is_lt);
            if earlier {
                *slot = row as u32;
            }
        }
        let mut position = vec![u32::MAX; if index.is_some() { data.n_rows() } else { 0 }];
        if index.is_some() {
            for (at, &row) in rows.iter().enumerate() {
                position[row as usize] = at as u32;
            }
        }
        let mut first_positions = Vec::new();
        let mut first_states = Vec::new();
        for &row in &opening {
            let at = if index.is_some() {
                position[row as usize]
            } else {
                row
            };
            if at != u32::MAX {
                first_positions.push(at);
                let rank = from_states[at as usize];
                first_states.push(i32::try_from(rank).map_err(|_| {
                    Error::input("a state code above the largest i32 is not supported")
                })?);
            }
        }

        let clusters = if clustered {
            let codes = data.cluster().ok_or_else(|| {
                Error::input("the bootstrap resamples clusters, so the data need a cluster column")
            })?;
            Some(rows.iter().map(|&row| codes[row as usize]).collect())
        } else {
            None
        };
        Ok(Self {
            data,
            index,
            from_states,
            to_states,
            first_positions,
            first_states,
            clusters,
            n_states,
        })
    }

    /// The start, stop and weight of the rows used.
    fn columns(&self) -> Result<Columns<'_>> {
        let surv = self.data.followup()?;
        let start = surv
            .start()
            .ok_or_else(|| Error::input("multi-state estimates need a start column"))?;
        let index = self.index.as_deref();
        Ok(Columns {
            starts: gather(start, index),
            stops: gather(surv.stop(), index),
            weights: match self.data.weight() {
                Some(weight) => gather(weight, index),
                None => Cow::Owned(vec![1.0; self.data.n_used()]),
            },
        })
    }
}

/// The columns of the rows used.
struct Columns<'a> {
    starts: Cow<'a, [f64]>,
    stops: Cow<'a, [f64]>,
    weights: Cow<'a, [f64]>,
}

/// The occupation probabilities and the weight at risk at each of `times`, for
/// follow-up data with several states.
///
/// The states are the status codes `1..=n_codes`, entered at the ends of the
/// rows. See the module documentation for what the data supply, which rows
/// count, and the checks that run first. `times` are the event times, usually
/// ascending.
///
/// # Errors
///
/// [`Error::Input`] when the data have no id or start column, a row has no
/// starting state, no row has positive weight, the rows of a subject overlap or
/// jump or teleport between states, the tie rule fails, `opts.initial` has the
/// wrong length, a negative or non-finite entry or a sum that is not 1, no
/// weight is in the opening states when the distribution is formed from the
/// data, or more weight leaves a state than was at risk in it.
pub fn estimate(data: &SurvData<'_>, times: &[f64], opts: &Options) -> Result<Estimate> {
    let prepared = Prepared::new(data, opts.timefix, false)?;
    let Columns {
        starts,
        stops,
        weights,
    } = prepared.columns()?;
    let rows = Rows {
        starts: &starts,
        stops: &stops,
        from_states: &prepared.from_states,
        to_states: &prepared.to_states,
    };
    let states = prepared.n_states;
    let initial = match &opts.initial {
        Some(given) => {
            check_distribution(given, states)?;
            given.clone()
        }
        None => initial_distribution(
            &prepared.first_states,
            &prepared.first_positions,
            &weights,
            states,
        )?,
    };
    let tolerance = rate_tolerance(opts, rows.len(), states)?;
    let occupation = occupation(rows, &weights, times, &initial, states, tolerance)?;
    Ok(Estimate {
        probabilities: occupation.probabilities,
        at_risk: occupation.at_risk,
        initial,
        n_states: states,
        n_rows: prepared.data.n_rows(),
        n_used: prepared.data.n_used(),
    })
}

/// A given initial distribution: one entry per state, none negative or not a
/// number, summing to 1 to within the rounding of its entries.
fn check_distribution(initial: &[f64], states: usize) -> Result<()> {
    if initial.len() != states {
        return Err(Error::input(
            "the initial distribution must have one entry per state",
        ));
    }
    if initial.iter().any(|&p| !(p >= 0.0 && p.is_finite())) {
        return Err(Error::input(
            "the initial distribution must not hold a negative or non-finite entry",
        ));
    }
    // Each entry is rounded to a double, which is off by at most half an
    // epsilon of itself, so the total is off by at most half an epsilon per
    // state. The sum itself is compensated.
    let total = compensated(initial.iter().copied());
    if (total - 1.0).abs() > states as f64 * f64::EPSILON {
        return Err(Error::input(format!(
            "the initial distribution must sum to 1, and sums to {total}"
        )));
    }
    Ok(())
}

/// The clustered bootstrap of the occupation probabilities or of the restricted
/// expected times, for follow-up data with several states.
///
/// Replicate `r` draws one weight per cluster from `seeds[r]`, multiplies the
/// weight of each row used by the weight of its cluster, forms the opening
/// distribution from the opening rows with those weights and repeats the
/// estimate of [`estimate`]. The cluster column is required, and `n_clusters`
/// of the data is the number of weights drawn, so a cluster whose rows all have
/// weight 0 draws a weight that is not used. The estimate depends on the
/// weights through ratios only, so this changes no result in distribution.
///
/// `horizons` selects what comes back. Empty returns the occupation
/// probabilities at `times`. Non-empty returns the restricted expected times
/// from `origin` to each horizon. See [`Bootstrap::values`] for the layout.
///
/// # Errors
///
/// As [`estimate`], and [`Error::Input`] when the data have no cluster column,
/// when `opts.initial` is given, or when in some replicate the opening rows
/// carry no weight.
pub fn bootstrap(
    data: &SurvData<'_>,
    times: &[f64],
    horizons: &[f64],
    origin: f64,
    seeds: &[u64],
    opts: &Options,
) -> Result<Bootstrap> {
    if opts.initial.is_some() {
        return Err(Error::input(
            "the bootstrap forms the initial distribution in each replicate, so none can be given",
        ));
    }
    let prepared = Prepared::new(data, opts.timefix, true)?;
    let Columns {
        starts,
        stops,
        weights,
    } = prepared.columns()?;
    let rows = Rows {
        starts: &starts,
        stops: &stops,
        from_states: &prepared.from_states,
        to_states: &prepared.to_states,
    };
    let clusters = prepared
        .clusters
        .as_deref()
        .ok_or_else(|| Error::input("the bootstrap needs a cluster column"))?;
    let values = bootstrap_rows(
        rows,
        &weights,
        clusters,
        &prepared.first_positions,
        &prepared.first_states,
        times,
        horizons,
        origin,
        seeds,
        prepared.data.n_clusters(),
        prepared.n_states,
        rate_tolerance(opts, rows.len(), prepared.n_states)?,
    )?;
    Ok(Bootstrap {
        values,
        n_replicates: seeds.len(),
        n_states: prepared.n_states,
        n_rows: prepared.data.n_rows(),
        n_used: prepared.data.n_used(),
    })
}

/// What [`variance`] reports besides the standard errors at each time.
#[derive(Debug, Clone, PartialEq)]
pub struct VarianceOptions {
    /// The starting time of the curve, `fit$t0` of `survfit`: where the
    /// restricted time in state begins. It has no default, since `survfit` takes
    /// it from the data in ways that depend on the initial states. No time of
    /// `times` may be before it.
    pub origin: f64,
    /// Horizons at which the standard error of the restricted time in each
    /// state, from `origin`, is wanted. Ascending. A horizon not after `origin`
    /// has a standard error of 0.
    pub horizons: Vec<f64>,
    /// Times at which the influence of every cluster is returned. Ascending. The
    /// influence at a time is that after the last of `times` not after it, and
    /// that of the initial distribution before the first. The influence of all
    /// clusters at all times is not kept, see [`Variance::influence`].
    pub influence_times: Vec<f64>,
}

impl VarianceOptions {
    /// Options with the given starting time and nothing else requested.
    #[must_use]
    pub const fn new(origin: f64) -> Self {
        Self {
            origin,
            horizons: Vec::new(),
            influence_times: Vec::new(),
        }
    }
}

/// The standard errors of the occupation probabilities and the cumulative
/// hazards by the infinitesimal jackknife, with what is needed to read them.
#[derive(Debug, Clone, PartialEq)]
pub struct Variance {
    /// `time * state`, row-major: the probability of each state after each
    /// time, from the recursion the standard errors come from. It equals
    /// [`Estimate::probabilities`] up to rounding, and is the estimate the
    /// errors belong to.
    pub probabilities: Vec<f64>,
    /// `time * state`, row-major: the standard error of the probability of each
    /// state, `fit$std.err` of `survfit`.
    pub std_err: Vec<f64>,
    /// The transitions that occur among the rows used, as `(from, to)` ranks of
    /// states, ordered by `to` and then `from`, the order of the columns of
    /// `fit$cumhaz`. A move to the state the row is in is a transition. `survfit`
    /// also has a column for a transition seen only in rows of weight 0, which
    /// is not used here.
    pub transitions: Vec<(u32, u32)>,
    /// `time * transition`: the cumulative hazard of each transition.
    pub cumhaz: Vec<f64>,
    /// `time * transition`: its standard error, `fit$std.chaz`.
    pub std_cumhaz: Vec<f64>,
    /// `time * state`: the standard error of the restricted time in each state
    /// from `origin` to each time, `fit$std.auc`. It is exact when `times`
    /// holds every event time.
    pub std_time: Vec<f64>,
    /// The standard error of each state's initial probability, 0 when the
    /// initial distribution is given.
    pub std_initial: Vec<f64>,
    /// The horizons of [`VarianceOptions::horizons`].
    pub horizons: Vec<f64>,
    /// `horizon * state`: the standard error of the restricted time in each
    /// state from the origin to the horizon, the `se(rmean)` that
    /// `summary(fit, rmean = horizon)` gives for a `fit` made with
    /// `influence = TRUE`.
    pub std_rmean: Vec<f64>,
    /// The number of states.
    pub n_states: usize,
    /// The number of clusters, the cluster column or else the ids, among the
    /// rows used.
    pub n_clusters: usize,
    /// The caller's code of each cluster, ascending.
    pub clusters: Vec<u32>,
    /// The times of [`VarianceOptions::influence_times`].
    pub influence_times: Vec<f64>,
    /// `cluster * influence time * state`, row-major: the influence of each
    /// cluster on the probability of each state, the sum over its rows of the
    /// weight times the derivative of the probability with respect to it. The
    /// squares of a column sum to the variance. Only the times asked for are
    /// kept, since all of them would hold `clusters * times * states` values.
    pub influence: Vec<f64>,
    /// The number of rows of the data, including rows of weight 0.
    pub n_rows: usize,
    /// The number of rows of positive weight, which are the rows used.
    pub n_used: usize,
}

/// For each of `at`, the index of the last of `times` not after it, or `None`
/// when it is before the first.
pub(crate) fn snapshot_positions(times: &[f64], at: &[f64]) -> Vec<Option<usize>> {
    at.iter()
        .map(|&t| times.partition_point(|&x| x <= t).checked_sub(1))
        .collect()
}

/// The standard errors of the occupation probabilities, the cumulative hazards
/// and the restricted time in each state, by the infinitesimal jackknife that
/// `survfit` reports.
///
/// `times` are the event times, strictly ascending and not before
/// `v.origin`, as for [`estimate`], and every event time must be among them,
/// since an event at another time is not counted. The influence of each cluster
/// is collapsed over its rows, the cluster column or else the id column, and the
/// standard error at a time is the square root of the sum of squares over the
/// clusters. Weights enter as case weights: the influence of a row is the weight
/// times the derivative of the estimate with respect to it. The initial
/// distribution is given in `opts.initial`, and then it carries no variance, or
/// formed from the opening rows, and then its estimation adds to the variance,
/// as in `survfit` without `p0`. See the module `influence` for the recursion
/// and for how memory is bounded: the influence of every cluster at every time is
/// never held, only that at [`VarianceOptions::influence_times`].
///
/// # Errors
///
/// As [`estimate`], and [`Error::Input`] when `times`, `v.horizons` or
/// `v.influence_times` is not ascending or has a value that is not finite, or a
/// time is before `v.origin`.
pub fn variance(
    data: &SurvData<'_>,
    times: &[f64],
    opts: &Options,
    v: &VarianceOptions,
) -> Result<Variance> {
    check_ascending(times, "times", true)?;
    check_ascending(&v.horizons, "horizons", false)?;
    check_ascending(&v.influence_times, "influence_times", false)?;
    if !v.origin.is_finite() {
        return Err(Error::input("origin must be finite"));
    }
    if times.first().is_some_and(|&t| t < v.origin) {
        return Err(Error::input("no time may be before the origin"));
    }
    let prepared = Prepared::new(data, opts.timefix, false)?;
    let Columns {
        starts,
        stops,
        weights,
    } = prepared.columns()?;
    let states = prepared.n_states;
    let initial = match &opts.initial {
        Some(given) => {
            check_distribution(given, states)?;
            given.clone()
        }
        None => initial_distribution(
            &prepared.first_states,
            &prepared.first_positions,
            &weights,
            states,
        )?,
    };

    // Clusters: the cluster column, else the ids.
    let (codes, levels) = match (prepared.data.cluster(), prepared.data.id()) {
        (Some(codes), _) => (codes, prepared.data.cluster_levels()),
        (None, Some(codes)) => (codes, prepared.data.id_levels()),
        (None, None) => unreachable!("Prepared::new needs an id column"),
    };
    let rows_used: Cow<'_, [u32]> = prepared.index.as_deref().map_or_else(
        || Cow::Owned((0..prepared.data.n_rows() as u32).collect()),
        Cow::Borrowed,
    );
    let row_cluster: Vec<u32> = rows_used.iter().map(|&r| codes[r as usize]).collect();
    let present = dense_of_present(row_cluster.iter().copied());
    let local: Vec<u32> = row_cluster
        .iter()
        .map(|&c| dense_index(&present, c) as u32)
        .collect();
    let n_clusters = present.len();

    // The influence of the initial distribution, when it is estimated.
    let u0 = if opts.initial.is_none() {
        let total = compensated(
            prepared
                .first_positions
                .iter()
                .map(|&position| weights[position as usize]),
        );
        let mut u0 = vec![0.0f64; n_clusters * states];
        for (&position, &opening) in prepared
            .first_positions
            .iter()
            .zip(prepared.first_states.iter())
        {
            let cluster = local[position as usize] as usize;
            let weight = weights[position as usize];
            for (j, &p0) in initial.iter().enumerate() {
                let indicator = if usize::try_from(opening) == Ok(j) {
                    1.0
                } else {
                    0.0
                };
                u0[cluster * states + j] += weight * (indicator - p0) / total;
            }
        }
        Some(u0)
    } else {
        None
    };

    let snapshots = snapshot_positions(times, &v.influence_times);
    let out = influence::run(
        &influence::Process {
            start: &starts,
            stop: &stops,
            from: &prepared.from_states,
            to: &prepared.to_states,
            weight: &weights,
            cluster: &local,
        },
        &influence::Request {
            states,
            n_clusters,
            times,
            p0: &initial,
            u0: u0.as_deref(),
            origin: v.origin,
            horizons: &v.horizons,
            snapshots: &snapshots,
        },
    );
    Ok(Variance {
        probabilities: out.p,
        std_err: out.std_p,
        transitions: out.transitions,
        cumhaz: out.cumhaz,
        std_cumhaz: out.std_c,
        std_time: out.std_a,
        std_initial: out.std_initial,
        horizons: v.horizons.clone(),
        std_rmean: out.std_h,
        n_states: states,
        n_clusters,
        clusters: present
            .iter()
            .map(|&c| levels.map_or(c, |l| l.code(c as usize).unwrap_or(c)))
            .collect(),
        influence_times: v.influence_times.clone(),
        influence: out.influence,
        n_rows: prepared.data.n_rows(),
        n_used: prepared.data.n_used(),
    })
}

#[cfg(test)]
mod tests {
    use super::{Plan, Rows, accumulate, bootstrap_rows, occupation};

    // Person A moves from state 0 to 1 at 2 and stays until 5. Person B is
    // censored at 3, a state outside the space. Person C moves at 4 and stays
    // until 6.
    const STARTS: [f64; 5] = [0.0, 2.0, 0.0, 0.0, 4.0];
    const STOPS: [f64; 5] = [2.0, 5.0, 3.0, 4.0, 6.0];
    const FROM: [u32; 5] = [0, 1, 0, 0, 1];
    const TO: [u32; 5] = [1, 1, 2, 1, 1];

    fn intervals<'a>(
        starts: &'a [f64],
        stops: &'a [f64],
        from_states: &'a [u32],
        to_states: &'a [u32],
    ) -> Rows<'a> {
        Rows {
            starts,
            stops,
            from_states,
            to_states,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-14
    }

    /// Worked by hand. At 2 three are at risk in state 0 and one moves, so the
    /// state 0 probability is 2/3. At 4 one is at risk in each state and the
    /// one in state 0 moves, so everyone is in state 1.
    #[test]
    fn the_product_integral_matches_a_hand_computed_curve() {
        let intervals = intervals(&STARTS, &STOPS, &FROM, &TO);
        let out = occupation(intervals, &[1.0; 5], &[2.0, 4.0], &[1.0, 0.0], 2, 1e-9).unwrap();
        for (got, want) in out
            .probabilities
            .iter()
            .zip([2.0 / 3.0, 1.0 / 3.0, 0.0, 1.0])
        {
            assert!(close(*got, want), "{:?}", out.probabilities);
        }
        assert_eq!(out.at_risk, vec![3.0, 2.0]);
    }

    /// The precomputed plan and the direct loops add the weights in the same
    /// order, so the tables agree to the bit.
    #[test]
    fn the_plan_and_the_direct_sums_agree_to_the_bit() {
        let weights = [0.1, 0.7, 0.3, 1.9, 0.2];
        let times = [2.0, 3.0, 4.0, 6.0];
        let intervals = intervals(&STARTS, &STOPS, &FROM, &TO);
        let plan = Plan::new(intervals, &times, 2).expect("ascending times");
        let direct = accumulate(&STARTS, &STOPS, &FROM, &TO, &weights, &times, 2, None);
        let planned = accumulate(
            &STARTS,
            &STOPS,
            &FROM,
            &TO,
            &weights,
            &times,
            2,
            Some(&plan),
        );
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&direct.0), bits(&planned.0));
        assert_eq!(bits(&direct.1), bits(&planned.1));
    }

    #[test]
    fn a_zero_length_interval_is_not_a_transition() {
        let intervals = intervals(&[0.0, 1.0], &[3.0, 1.0], &[0, 0], &[0, 1]);
        let out = occupation(intervals, &[1.0; 2], &[1.0], &[1.0, 0.0], 2, 1e-9).unwrap();
        assert_eq!(out.probabilities, vec![1.0, 0.0]);
    }

    #[test]
    fn the_bootstrap_is_reproducible_and_has_the_documented_shape() {
        let intervals = intervals(&STARTS, &STOPS, &FROM, &TO);
        let run = |horizons: &[f64]| {
            bootstrap_rows(
                intervals,
                &[1.0; 5],
                &[0, 1, 0, 2, 2],
                &[0, 2, 3],
                &[0, 0, 0],
                &[2.0, 4.0],
                horizons,
                0.0,
                &[1, 2, 3],
                3,
                2,
                1e-9,
            )
            .unwrap()
        };
        let curves = run(&[]);
        assert_eq!(curves.len(), 3 * 2 * 2);
        let again = run(&[]);
        assert_eq!(
            curves.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            again.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let times_in_state = run(&[6.0]);
        assert_eq!(times_in_state.len(), 3 * 2);
        for pair in times_in_state.chunks(2) {
            assert!(close(pair[0] + pair[1], 6.0), "{pair:?}");
        }
    }

    #[test]
    fn malformed_input_is_refused() {
        let intervals = intervals(&STARTS, &STOPS, &FROM, &TO);
        assert!(
            occupation(intervals, &[1.0; 5], &[2.0], &[1.0], 2, 1e-9).is_err(),
            "initial distribution of the wrong size"
        );
        let refused = |states: usize, codes: &[u32]| {
            bootstrap_rows(
                intervals,
                &[1.0; 5],
                codes,
                &[0],
                &[0],
                &[2.0],
                &[],
                0.0,
                &[1],
                2,
                states,
                1e-9,
            )
            .is_err()
        };
        assert!(refused(0, &[0; 5]), "empty state space");
        assert!(refused(2, &[0, 0, 0, 0, 2]), "cluster code out of range");
        assert!(!refused(2, &[0, 0, 0, 0, 1]));
    }

    #[test]
    fn opening_intervals_without_weight_have_no_initial_distribution() {
        let intervals = intervals(&STARTS, &STOPS, &FROM, &TO);
        let run = |base: &[f64]| {
            bootstrap_rows(
                intervals,
                base,
                &[0, 0, 0, 0, 0],
                &[0, 2, 3],
                &[0, 0, 0],
                &[2.0],
                &[],
                0.0,
                &[1],
                1,
                2,
                1e-9,
            )
        };
        assert!(run(&[1.0; 5]).is_ok());
        assert!(run(&[0.0; 5]).is_err());
    }
}
