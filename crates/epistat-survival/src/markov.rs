//! The Titman-Putter landmark test of the Markov property for one transition of
//! a multi-state process.
//!
//! # Algorithm
//!
//! Let the origin of the transition be qualifying state 0, and let the other
//! qualifying states be the states a person may have occupied at a landmark. The
//! stays in the origin are the episodes at risk. Episode `i` has an entry `a_i`,
//! an exit `b_i` and an indicator `d_i` that it ends in the target state. A
//! stay covers a landmark `s` when `start < s <= stop`.
//!
//! The risk-set sums are over `a_i < t <= b_i`. Let `Y_i` be the number of
//! episodes at risk at `b_i`. The Nelson-Aalen estimate of the cumulative
//! intensity of the transition is the right-continuous step function `L` with
//! increments `d_i / Y_i` summed at each distinct exit time. For a landmark
//! `s` and a qualifying state `j`, let `I_ij(s)` be 1 when the person of episode
//! `i` occupied `j` at `s` and let `A_i(s) = 1(b_i > s)`. The martingale
//! residual and the trace are
//!
//! ```text
//! M_i(s) = d_i A_i(s) - [ L(max(s, b_i)) - L(max(s, a_i)) ]
//! Z_j(s) = sum over i of M_i(s) I_ij(s) A_i(s).
//! ```
//!
//! The variance estimate is
//!
//! ```text
//! V_j(s) = sum over i of (d_i / Y_i^2) A_i(s) [ B2_i - 2 m_i B_i + m_i^2 Y_i ]
//! ```
//!
//! with `B_i` the sum of `I_ij(s)` over the risk set at `b_i`, `B2_i` the sum of
//! its squares, which is `B_i` since an indicator equals its square, and
//! `m_i = B_i / Y_i`. The normalised trace is `z_j(s) = Z_j(s) / sqrt(V_j(s))`,
//! or 0 when `V_j(s)` is not positive. The statistic of state `j` is the mean
//! over the `K` landmarks of `|z_j(s)|`.
//!
//! The p-value comes from a wild bootstrap \[1\]. Each replicate draws one
//! standard normal multiplier `G_i` per episode, shared by every landmark and
//! state, because the trace at two landmarks is two readings of one sample. The
//! replicate score is `sum over i of d_i (I_ij - m_i) A_i G_i`, divided by
//! `sqrt(V_j)`, and its statistic is the mean of the absolute values over the
//! landmarks. With `B` replicates, the p-value is `(1 + #{replicates at or above
//! the observed statistic}) / (B + 1)`.
//!
//! # Arithmetic
//!
//! The sums of traces, variances and scores are compensated. The cumulative
//! intensity groups the increments at each instant before it accumulates them,
//! walking the rows in input order, and then adds one group total per sorted
//! distinct time. Adding the increments one at a time in sorted order is a
//! different sequence of roundings, which moves the traces by hundreds of
//! units in the last place.
//!
//! The multipliers are standard normals by the Box-Muller transform \[2\] over
//! `epistat_rng::Stream` (`Stream::normal`: `sqrt(-2 ln u1) cos(2 pi u2)` for
//! two successive uniforms). The stream is exact, but `ln` and `cos` come from the platform's
//! maths library and may differ in the last place between platforms. The
//! p-value is a Monte Carlo quantity, so a difference of that size is
//! immaterial. Replicates and slots run in parallel, each depending on its seed
//! or its slot only, so the result does not depend on the number of threads.
//!
//! # Edge cases
//!
//! With no replicates the p-values are `None`. A slot whose variance is not
//! positive has a normalised trace of 0 and contributes 0 to every replicate.
//! That amounts to dropping the slot from the sum, while the mean is still
//! taken over every landmark.
//! The test needs at least one qualifying state and one landmark.
//!
//! # References
//!
//! 1. Titman AC, Putter H. General tests of the Markov property in multi-state
//!    models. Biostatistics 2022;23(2):380-96. doi:10.1093/biostatistics/kxaa030
//! 2. Box GEP, Muller ME. A note on the generation of random normal deviates.
//!    Ann Math Stat 1958;29(2):610-1. doi:10.1214/aoms/1177706645
//! 3. de Wreede LC, Fiocco M, Putter H. mstate: an R package for the analysis of
//!    competing risks and multi-state models. J Stat Softw 2011;38(7):1-30.
//!    doi:10.18637/jss.v038.i07

#![expect(
    clippy::cast_possible_truncation,
    reason = "the constructor of Stays checks that the row count fits in u32"
)]

use std::collections::HashMap;

use epistat_rng::Stream;
use epistat_stats::sum::compensated;
use rayon::prelude::*;

use crate::check::{check_lengths, check_row_count, refuse_non_finite};
use crate::error::{Error, Result};

/// For each query time, the sum of `values` over the risk set `entry < t <= exit`.
///
/// Two suffix sums rather than a scan per query: the naive form is quadratic and
/// this runs once per landmark per qualifying state. Since `entry < exit` always, `entry >= t`
/// implies `exit > t`, so the risk set is the episodes still open minus those
/// not yet begun.
#[cfg(test)]
fn risk_set_sums(values: &[f64], entry: &[f64], exit: &[f64], at: &[f64]) -> Vec<f64> {
    RiskIndex::new(entry, exit, at).sums(values)
}

/// The part of [`risk_set_sums`] that depends on the episodes and the query
/// times but not on `values`: the sort order of the entries and exits, and
/// where each query time falls among them. A test runs the sums for many
/// `values` over the same episodes, so this is built once.
struct RiskIndex {
    exit_order: Vec<u32>,
    entry_order: Vec<u32>,
    exit_position: Vec<u32>,
    entry_position: Vec<u32>,
}

impl RiskIndex {
    fn new(entry: &[f64], exit: &[f64], at: &[f64]) -> Self {
        let sorted = |keys: &[f64]| -> (Vec<u32>, Vec<f64>) {
            let mut order: Vec<u32> = (0..keys.len() as u32).collect();
            order.sort_by(|&a, &b| keys[a as usize].total_cmp(&keys[b as usize]));
            let sorted_keys: Vec<f64> = order.iter().map(|&i| keys[i as usize]).collect();
            (order, sorted_keys)
        };
        let lower_bound = |keys: &[f64], target: f64| -> u32 {
            let (mut low, mut high) = (0usize, keys.len());
            while low < high {
                let mid = usize::midpoint(low, high);
                if keys[mid] < target {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low as u32
        };
        let (exit_order, exit_keys) = sorted(exit);
        let (entry_order, entry_keys) = sorted(entry);
        Self {
            exit_position: at
                .iter()
                .map(|&time| lower_bound(&exit_keys, time))
                .collect(),
            entry_position: at
                .iter()
                .map(|&time| lower_bound(&entry_keys, time))
                .collect(),
            exit_order,
            entry_order,
        }
    }

    fn sums(&self, values: &[f64]) -> Vec<f64> {
        let suffix = |order: &[u32]| -> Vec<f64> {
            let mut totals = vec![0.0f64; order.len() + 1];
            for position in (0..order.len()).rev() {
                totals[position] = totals[position + 1] + values[order[position] as usize];
            }
            totals
        };
        let exit_suffix = suffix(&self.exit_order);
        let entry_suffix = suffix(&self.entry_order);
        self.exit_position
            .iter()
            .zip(&self.entry_position)
            .map(|(&exit, &entry)| exit_suffix[exit as usize] - entry_suffix[entry as usize])
            .collect()
    }
}

/// The Nelson-Aalen cumulative intensity as a right-continuous step function.
///
/// Increments at one instant are summed before the running total: two
/// transitions recorded at the same time are one jump, and evaluating between
/// them would return a value the estimator never takes.
struct StepCumulative {
    times: Vec<f64>,
    totals: Vec<f64>,
}

impl StepCumulative {
    /// Group first, then accumulate. The increments at each instant are summed
    /// walking the rows in input order, and then one group total at a time is
    /// added in sorted time order. Adding the increments one by one in sorted
    /// order is a different sequence of roundings.
    fn build(times: &[f64], increments: &[f64]) -> Self {
        let mut grouped: HashMap<u64, f64> = HashMap::new();
        for (&time, &increment) in times.iter().zip(increments.iter()) {
            *grouped.entry(time.to_bits()).or_insert(0.0) += increment;
        }
        let mut distinct: Vec<f64> = grouped.keys().map(|&bits| f64::from_bits(bits)).collect();
        distinct.sort_by(f64::total_cmp);

        let mut totals = Vec::with_capacity(distinct.len());
        let mut running = 0.0f64;
        for time in &distinct {
            running += grouped[&time.to_bits()];
            totals.push(running);
        }
        Self {
            times: distinct,
            totals,
        }
    }

    fn at(&self, when: f64) -> f64 {
        let (mut low, mut high) = (0usize, self.times.len());
        while low < high {
            let mid = usize::midpoint(low, high);
            if self.times[mid] <= when {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if low == 0 { 0.0 } else { self.totals[low - 1] }
    }
}

/// Standard normal multipliers from the stream, one `Stream::normal` each.
fn standard_normals(count: usize, seed: u64) -> Vec<f64> {
    let mut stream = Stream::seeded(seed);
    (0..count).map(|_| stream.normal()).collect()
}

/// The code of a stay in no qualifying state, and of an individual whose stays
/// cover no landmark.
const NOT_QUALIFYING: u32 = u32::MAX;

/// The interval data the test reads: every stay of every individual, in input
/// order.
#[derive(Debug, Clone, Copy)]
pub struct Stays<'a> {
    person: &'a [u32],
    start: &'a [f64],
    stop: &'a [f64],
    state: &'a [u32],
    event: &'a [f64],
}

impl<'a> Stays<'a> {
    /// The stays, one entry per stay. `person` is the dense code of the
    /// individual. `state` is the index of the state occupied during the stay
    /// among the qualifying states, or any value at or above their count for
    /// another state. Qualifying state 0 is the origin of the transition, so its
    /// stays are the ones at risk. `event` is 1 when the stay ends in the target
    /// state of the transition.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the columns differ in length, or when a
    /// start or stop time is NaN or infinite.
    pub fn new(
        person: &'a [u32],
        start: &'a [f64],
        stop: &'a [f64],
        state: &'a [u32],
        event: &'a [f64],
    ) -> Result<Self> {
        check_lengths(&[
            ("person", person.len()),
            ("start", start.len()),
            ("stop", stop.len()),
            ("state", state.len()),
            ("event", event.len()),
        ])?;
        check_row_count(person.len())?;
        refuse_non_finite(start, "start")?;
        refuse_non_finite(stop, "stop")?;
        Ok(Self {
            person,
            start,
            stop,
            state,
            event,
        })
    }
}

/// One byte per (stay at risk, landmark, qualifying state): whether the stay's
/// individual occupied that state at that landmark.
///
/// A stay covers landmark `s` when `start < s <= stop`, the convention of
/// `mstate`. Every stay is scanned, not only those at risk, because the
/// indicator records where the individual was before entering the origin. When
/// several stays of one individual cover a landmark, the last in input order
/// counts.
fn occupancy(stays: &Stays<'_>, at_risk: &[usize], grid: &[f64], qualifying: usize) -> Vec<u8> {
    let n_people = stays
        .person
        .iter()
        .copied()
        .max()
        .map_or(0, |most| most as usize + 1);
    let mut occupied = vec![NOT_QUALIFYING; n_people];
    let mut indicators = vec![0u8; at_risk.len() * grid.len() * qualifying];
    for (landmark_index, &landmark) in grid.iter().enumerate() {
        occupied.fill(NOT_QUALIFYING);
        for row in 0..stays.person.len() {
            if stays.start[row] < landmark && stays.stop[row] >= landmark {
                occupied[stays.person[row] as usize] = stays.state[row];
            }
        }
        for (i, &row) in at_risk.iter().enumerate() {
            let state = occupied[stays.person[row] as usize] as usize;
            if state < qualifying {
                indicators[(i * grid.len() + landmark_index) * qualifying + state] = 1;
            }
        }
    }
    indicators
}

/// The result of the test. Per-landmark-and-state values are flattened
/// `landmark * qualifying + state`.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkovTest {
    /// The trace `Z_j(s)`, `grid * qualifying` values.
    pub trace: Vec<f64>,
    /// The variance estimate `V_j(s)`, `grid * qualifying` values.
    pub variance: Vec<f64>,
    /// Stays at risk whose individual occupied the state at the landmark,
    /// `grid * qualifying` values.
    pub n_occupied: Vec<f64>,
    /// Transitions under test after each landmark, `grid` values.
    pub n_events_after: Vec<f64>,
    /// The statistic of each qualifying state, `qualifying` values.
    pub statistic: Vec<f64>,
    /// The wild-bootstrap p-value of each qualifying state, `qualifying` values,
    /// `None` without replicates.
    pub p_value: Vec<Option<f64>>,
}

/// The observed traces, their variance, the summary statistic and the p-value,
/// with one wild-bootstrap replicate per seed.
fn titman_putter(stays: &Stays<'_>, grid: &[f64], qualifying: usize, seeds: &[u64]) -> MarkovTest {
    let at_risk: Vec<usize> = (0..stays.state.len())
        .filter(|&row| stays.state[row] == 0)
        .collect();
    let indicators = occupancy(stays, &at_risk, grid, qualifying);
    let t0: Vec<f64> = at_risk.iter().map(|&row| stays.start[row]).collect();
    let t1: Vec<f64> = at_risk.iter().map(|&row| stays.stop[row]).collect();
    let d: Vec<f64> = at_risk.iter().map(|&row| stays.event[row]).collect();
    let n = t0.len();
    let n_bootstrap = seeds.len();

    let ones = vec![1.0f64; n];
    let risk_index = RiskIndex::new(&t0, &t1, &t1);
    let at_risk_n = risk_index.sums(&ones);
    let increment: Vec<f64> = d
        .iter()
        .zip(at_risk_n.iter())
        .map(|(&event, &size)| if size > 0.0 { event / size } else { 0.0 })
        .collect();
    let cumulative = StepCumulative::build(&t1, &increment);

    let slots = grid.len() * qualifying;
    let mut centred = vec![0.0f64; slots * n];

    // Per landmark: who is still under observation after it, the events among
    // them, and the martingale residual. All of it is read-only below.
    let landmarks: Vec<(Vec<f64>, f64, Vec<f64>)> = grid
        .par_iter()
        .map(|&landmark| {
            let after: Vec<f64> = t1.iter().map(|&stop| f64::from(stop > landmark)).collect();
            let events_after: f64 = (0..n).map(|i| d[i] * after[i]).sum();
            let residual: Vec<f64> = (0..n)
                .map(|i| {
                    d[i] * after[i]
                        - (cumulative.at(landmark.max(t1[i])) - cumulative.at(landmark.max(t0[i])))
                })
                .collect();
            (after, events_after, residual)
        })
        .collect();
    let n_events_after: Vec<f64> = landmarks.iter().map(|(_, events, _)| *events).collect();

    // One (landmark, state) slot is independent of every other and writes its
    // own row of `centred`.
    let one_slot = |slot: usize, centred_row: &mut [f64]| -> (f64, f64, f64) {
        let (landmark_index, state) = (slot / qualifying, slot % qualifying);
        let (after, _, residual) = &landmarks[landmark_index];
        let indicator: Vec<f64> = (0..n)
            .map(|i| f64::from(indicators[(i * grid.len() + landmark_index) * qualifying + state]))
            .collect();

        let trace = compensated((0..n).map(|i| residual[i] * indicator[i] * after[i]));

        let bar = risk_index.sums(&indicator);
        let mean: Vec<f64> = bar
            .iter()
            .zip(at_risk_n.iter())
            .map(|(&total, &size)| if size > 0.0 { total / size } else { 0.0 })
            .collect();
        // An indicator is 0 or 1, so it equals its square and the risk-set sum
        // of the squares is `bar` bit for bit.
        let squared = &bar;

        let variance = compensated((0..n).map(|i| {
            increment[i]
                * after[i]
                * (squared[i] - 2.0 * mean[i] * bar[i] + mean[i] * mean[i] * at_risk_n[i])
        }));
        let occupied: f64 = indicator.iter().sum();
        for i in 0..n {
            centred_row[i] = d[i] * (indicator[i] - mean[i]) * after[i];
        }
        (trace, variance, occupied)
    };
    let per_slot: Vec<(f64, f64, f64)> = if n == 0 {
        (0..slots).map(|slot| one_slot(slot, &mut [])).collect()
    } else {
        centred
            .par_chunks_mut(n)
            .enumerate()
            .map(|(slot, row)| one_slot(slot, row))
            .collect()
    };
    let traces: Vec<f64> = per_slot.iter().map(|slot| slot.0).collect();
    let variances: Vec<f64> = per_slot.iter().map(|slot| slot.1).collect();
    let n_occupied: Vec<f64> = per_slot.iter().map(|slot| slot.2).collect();

    let normalised: Vec<f64> = (0..grid.len() * qualifying)
        .map(|slot| {
            if variances[slot] > 0.0 {
                traces[slot] / variances[slot].sqrt()
            } else {
                0.0
            }
        })
        .collect();
    let statistic: Vec<f64> = (0..qualifying)
        .map(|state| {
            let total = compensated(
                (0..grid.len()).map(|landmark| normalised[landmark * qualifying + state].abs()),
            );
            total / grid.len() as f64
        })
        .collect();

    // The replicates. One multiplier per episode, reused across the whole grid
    // and every qualifying state: the trace at two landmarks is two readings of
    // one sample, and drawing independently per landmark would break the
    // correlation the summary averages over.
    let exceeded: Vec<usize> = seeds
        .par_iter()
        .map(|&seed| {
            let multipliers = standard_normals(n, seed);
            (0..qualifying)
                .map(|state| {
                    let total: f64 = (0..grid.len())
                        .map(|landmark| {
                            let slot = landmark * qualifying + state;
                            if variances[slot] <= 0.0 {
                                return 0.0;
                            }
                            let base = slot * n;
                            let score =
                                compensated((0..n).map(|i| centred[base + i] * multipliers[i]));
                            (score / variances[slot].sqrt()).abs()
                        })
                        .sum();
                    usize::from(total / grid.len() as f64 >= statistic[state])
                })
                .collect::<Vec<usize>>()
        })
        .reduce(
            || vec![0usize; qualifying],
            |mut left, right| {
                for (slot, value) in left.iter_mut().zip(right) {
                    *slot += value;
                }
                left
            },
        );
    let p_value = exceeded
        .iter()
        .map(|&count| (n_bootstrap > 0).then(|| (count + 1) as f64 / (n_bootstrap + 1) as f64))
        .collect();

    MarkovTest {
        trace: traces,
        variance: variances,
        n_occupied,
        n_events_after,
        statistic,
        p_value,
    }
}

/// The Titman-Putter test of one transition, from the interval data.
///
/// `grid` holds the landmarks and `qualifying` the number of qualifying states
/// of `stays`. There is one wild-bootstrap replicate per entry of `seeds`, and
/// the multipliers of a replicate come from its seed. See [`MarkovTest`] for
/// what comes back and the module documentation for the method.
///
/// A landmark and state whose variance is not positive adds 0 to the statistic
/// and to every replicate, which drops the slot from the sum. The mean is still
/// taken over all landmarks.
///
/// # Errors
///
/// [`Error::Input`] when there is no qualifying state or no landmark.
pub fn markov_test(
    stays: &Stays<'_>,
    grid: &[f64],
    qualifying: usize,
    seeds: &[u64],
) -> Result<MarkovTest> {
    if qualifying == 0 || grid.is_empty() {
        return Err(Error::input(
            "the test needs at least one qualifying state and one landmark",
        ));
    }
    Ok(titman_putter(stays, grid, qualifying, seeds))
}

#[cfg(test)]
mod tests {
    use super::{Stays, markov_test, risk_set_sums, titman_putter};

    /// A three-state fixture with `a` as state 0, `b` as 1 and everything else
    /// as 2. Rows are
    /// `(person, start, stop, state, ends in b)`.
    const FIXTURE: [(u32, f64, f64, u32, f64); 21] = [
        (1, 0.0, 2.0, 0, 1.0),
        (1, 2.0, 5.5, 1, 0.0),
        (1, 5.5, 9.0, 0, 0.0),
        (2, 0.0, 3.5, 0, 1.0),
        (2, 3.5, 8.0, 1, 0.0),
        (3, 0.0, 4.0, 0, 1.0),
        (3, 4.0, 7.5, 1, 0.0),
        (4, 0.0, 6.0, 0, 0.0),
        (5, 0.0, 1.5, 0, 1.0),
        (5, 1.5, 3.0, 1, 0.0),
        (5, 3.0, 6.5, 0, 1.0),
        (5, 6.5, 10.0, 1, 0.0),
        (6, 0.0, 8.5, 0, 0.0),
        (7, 0.0, 2.5, 0, 1.0),
        (7, 2.5, 9.5, 1, 0.0),
        (8, 0.0, 5.0, 0, 0.0),
        (9, 0.0, 1.0, 0, 1.0),
        (9, 1.0, 4.5, 1, 0.0),
        (9, 4.5, 10.0, 0, 0.0),
        (10, 0.0, 7.0, 0, 1.0),
        (10, 7.0, 9.0, 1, 0.0),
    ];

    /// `mstate::MarkovTest(..., transition = 1, grid = c(1, 2, 3), dist =
    /// "normal")`, R 4.6.1, mstate 0.3.3: `zbar`, `est_cov[, 1, 1]`,
    /// `orig_stat` and `nobs_grid`.
    #[test]
    fn matches_mstate() {
        let column = |f: fn(&(u32, f64, f64, u32, f64)) -> f64| -> Vec<f64> {
            FIXTURE.iter().map(f).collect()
        };
        let person: Vec<u32> = FIXTURE.iter().map(|row| row.0).collect();
        let state: Vec<u32> = FIXTURE.iter().map(|row| row.3).collect();
        let (start, stop, event) = (column(|r| r.1), column(|r| r.2), column(|r| r.4));
        let stays = Stays::new(&person, &start, &stop, &state, &event).unwrap();
        let out = titman_putter(&stays, &[1.0, 2.0, 3.0], 2, &[]);

        let zbar = [
            [-1.110_223_024_625_16e-16, 0.0],
            [-4.876_869_381_030_53e-2, 4.876_869_381_030_54e-2],
            [4.724_561_616_186_37e-1, -4.724_561_616_186_37e-1],
        ];
        let variance = [0.0, 0.688_837_868_480_726, 0.751_337_868_480_726];
        for (landmark, (expected, &expected_variance)) in zbar.iter().zip(&variance).enumerate() {
            for (state, &z) in expected.iter().enumerate() {
                let slot = landmark * 2 + state;
                let v = out.variance[slot];
                let normalised = if v > 0.0 {
                    out.trace[slot] / v.sqrt()
                } else {
                    0.0
                };
                assert!((normalised - z).abs() < 1e-12);
                assert!((v - expected_variance).abs() < 1e-12);
            }
        }
        for statistic in &out.statistic {
            assert!((statistic - 0.173_741_618_476_314).abs() < 1e-12);
        }
        assert_eq!(out.n_events_after, vec![7.0, 5.0, 4.0]);
        assert!(out.p_value.iter().all(Option::is_none));
    }

    /// The indicator reads the stay covering the landmark, over every stay of
    /// the individual, including stays that are not at risk.
    #[test]
    fn occupancy_reads_the_stay_covering_the_landmark() {
        // Person 0 is in state 1 on (0, 2] and in the origin on (2, 5].
        let stays = Stays::new(&[0, 0], &[0.0, 2.0], &[2.0, 5.0], &[1, 0], &[0.0, 1.0]).unwrap();
        let indicators = super::occupancy(&stays, &[1], &[2.0, 3.0], 2);
        // At 2 the first stay covers it (start < 2 <= stop); at 3 the second.
        assert_eq!(indicators, vec![0, 1, 1, 0]);
    }

    /// The risk set at `t` is the stays with `entry < t <= exit`. The stays
    /// carry 1, 2, 4 and 8 on (0, 2], (1, 3], (2, 4] and (0, 2], so the last
    /// ties the first on both keys.
    #[test]
    fn risk_set_sums_count_entry_exclusive_and_exit_inclusive() {
        let values = [1.0, 2.0, 4.0, 8.0];
        let entry = [0.0, 1.0, 2.0, 0.0];
        let exit = [2.0, 3.0, 4.0, 2.0];
        let at = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0];
        let sums = risk_set_sums(&values, &entry, &exit, &at);
        assert_eq!(sums, vec![0.0, 9.0, 9.0, 11.0, 6.0, 6.0, 4.0, 0.0]);
    }

    #[test]
    fn a_non_finite_time_is_refused_with_its_column() {
        let person = [0u32, 0];
        let state = [0u32, 1];
        let event = [1.0, 0.0];
        let finite = [0.0, 2.0];
        for (bad, column) in [(f64::NAN, "start"), (f64::INFINITY, "stop")] {
            let broken = [1.0, bad];
            let (start, stop) = if column == "start" {
                (&broken, &[2.0, 3.0])
            } else {
                (&finite, &broken)
            };
            let Err(error) = Stays::new(&person, start, stop, &state, &event) else {
                panic!("a non-finite {column} must be refused");
            };
            assert!(
                error
                    .to_string()
                    .starts_with(&format!("{column}: 1 non-finite"))
            );
        }
        assert!(Stays::new(&person, &finite, &[2.0, 3.0], &state, &event).is_ok());
    }

    #[test]
    fn the_bootstrap_p_value_is_reproducible_and_in_range() {
        let person: Vec<u32> = FIXTURE.iter().map(|row| row.0).collect();
        let state: Vec<u32> = FIXTURE.iter().map(|row| row.3).collect();
        let start: Vec<f64> = FIXTURE.iter().map(|row| row.1).collect();
        let stop: Vec<f64> = FIXTURE.iter().map(|row| row.2).collect();
        let event: Vec<f64> = FIXTURE.iter().map(|row| row.4).collect();
        let stays = Stays::new(&person, &start, &stop, &state, &event).unwrap();
        let seeds: Vec<u64> = (0..50).collect();
        let first = markov_test(&stays, &[1.0, 2.0, 3.0], 2, &seeds).unwrap();
        let second = markov_test(&stays, &[1.0, 2.0, 3.0], 2, &seeds).unwrap();
        assert_eq!(
            first
                .p_value
                .iter()
                .map(|p| p.unwrap().to_bits())
                .collect::<Vec<_>>(),
            second
                .p_value
                .iter()
                .map(|p| p.unwrap().to_bits())
                .collect::<Vec<_>>()
        );
        assert!(
            first
                .p_value
                .iter()
                .all(|&p| (1.0 / 51.0..=1.0).contains(&p.unwrap()))
        );
    }

    #[test]
    fn a_test_without_states_or_landmarks_is_refused() {
        let stays = Stays::new(&[0], &[0.0], &[1.0], &[0], &[1.0]).unwrap();
        assert!(markov_test(&stays, &[1.0], 0, &[]).is_err());
        assert!(markov_test(&stays, &[], 1, &[]).is_err());
    }
}
