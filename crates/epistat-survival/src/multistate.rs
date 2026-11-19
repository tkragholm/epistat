//! State occupation probabilities of a multi-state process, by the Aalen-Johansen
//! product integral, and a clustered bootstrap of them.
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
//! The restricted expected time in each state up to a horizon `h` is the integral
//! of `p` from an origin `o` to `h`. Since `p` is a step function, the integral
//! is a sum of rectangles, and the segment from the origin to the first event
//! time after it uses the starting distribution.
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
//! # The bootstrap
//!
//! [`bootstrap_state_occupation`] reweights the intervals by random cluster
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

#![expect(
    clippy::cast_possible_truncation,
    reason = "a plan is built only when the row and table counts fit in u32"
)]

use epistat_rng::dirichlet_weights;
use epistat_stats::sum::compensated;
use rayon::prelude::*;

use crate::check::check_lengths;
use crate::error::{Error, Result};

/// The intervals of a multi-state process, one entry per row.
///
/// States are ranks into the caller's own state space; a rank at or above the
/// number of states means "outside the space". Rows are read positionally, so
/// the four slices are the same length.
#[derive(Debug, Clone, Copy)]
pub struct Intervals<'a> {
    starts: &'a [f64],
    stops: &'a [f64],
    from_states: &'a [u32],
    to_states: &'a [u32],
}

impl<'a> Intervals<'a> {
    /// The interval `(starts[i], stops[i]]` spent in state `from_states[i]`,
    /// ending in `to_states[i]`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the slices differ in length.
    pub fn new(
        starts: &'a [f64],
        stops: &'a [f64],
        from_states: &'a [u32],
        to_states: &'a [u32],
    ) -> Result<Self> {
        check_lengths(&[
            ("starts", starts.len()),
            ("stops", stops.len()),
            ("from_states", from_states.len()),
            ("to_states", to_states.len()),
        ])?;
        Ok(Self {
            starts,
            stops,
            from_states,
            to_states,
        })
    }

    /// The number of intervals.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.starts.len()
    }

    /// Whether there are no intervals.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }
}

/// The occupation probabilities at every event time, and the weight at risk.
#[derive(Debug, Clone, PartialEq)]
pub struct Occupation {
    /// `time * state`, row-major: the probability of each state after each time.
    pub probabilities: Vec<f64>,
    /// The total weight at risk at each time.
    pub at_risk: Vec<f64>,
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
    fn new(intervals: Intervals<'_>, times: &[f64], states: usize) -> Option<Self> {
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
pub fn state_occupation_curve(
    intervals: Intervals<'_>,
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
    intervals: Intervals<'_>,
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
pub fn bootstrap_state_occupation(
    intervals: Intervals<'_>,
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

#[cfg(test)]
mod tests {
    use super::{Intervals, Plan, accumulate, bootstrap_state_occupation, state_occupation_curve};

    // Person A moves from state 0 to 1 at 2 and stays until 5. Person B is
    // censored at 3, a state outside the space. Person C moves at 4 and stays
    // until 6.
    const STARTS: [f64; 5] = [0.0, 2.0, 0.0, 0.0, 4.0];
    const STOPS: [f64; 5] = [2.0, 5.0, 3.0, 4.0, 6.0];
    const FROM: [u32; 5] = [0, 1, 0, 0, 1];
    const TO: [u32; 5] = [1, 1, 2, 1, 1];

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-14
    }

    /// Worked by hand. At 2 three are at risk in state 0 and one moves, so the
    /// state 0 probability is 2/3. At 4 one is at risk in each state and the
    /// one in state 0 moves, so everyone is in state 1.
    #[test]
    fn the_product_integral_matches_a_hand_computed_curve() {
        let intervals = Intervals::new(&STARTS, &STOPS, &FROM, &TO).unwrap();
        let out = state_occupation_curve(intervals, &[1.0; 5], &[2.0, 4.0], &[1.0, 0.0], 2, 1e-9)
            .unwrap();
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
        let intervals = Intervals::new(&STARTS, &STOPS, &FROM, &TO).unwrap();
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
        let intervals = Intervals::new(&[0.0, 1.0], &[3.0, 1.0], &[0, 0], &[0, 1]).unwrap();
        let out =
            state_occupation_curve(intervals, &[1.0; 2], &[1.0], &[1.0, 0.0], 2, 1e-9).unwrap();
        assert_eq!(out.probabilities, vec![1.0, 0.0]);
    }

    #[test]
    fn the_bootstrap_is_reproducible_and_has_the_documented_shape() {
        let intervals = Intervals::new(&STARTS, &STOPS, &FROM, &TO).unwrap();
        let run = |horizons: &[f64]| {
            bootstrap_state_occupation(
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
        assert!(Intervals::new(&[0.0], &[1.0, 2.0], &[0], &[0]).is_err());
        let intervals = Intervals::new(&STARTS, &STOPS, &FROM, &TO).unwrap();
        assert!(
            state_occupation_curve(intervals, &[1.0; 5], &[2.0], &[1.0], 2, 1e-9).is_err(),
            "initial distribution of the wrong size"
        );
        let refused = |states: usize, codes: &[u32]| {
            bootstrap_state_occupation(
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
        let intervals = Intervals::new(&STARTS, &STOPS, &FROM, &TO).unwrap();
        let run = |base: &[f64]| {
            bootstrap_state_occupation(
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
