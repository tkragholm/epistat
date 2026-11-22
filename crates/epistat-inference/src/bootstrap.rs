//! Bootstrap replicates whose draws depend only on the seed and the
//! replicate number.
//!
//! Replicate `r` draws from the stream seeded with
//! [`replicate_seed`]`(seed, r)`, so a replicate is the same whichever thread
//! computes it and in whatever order. The output is the same on any number
//! of threads.
//!
//! # Method
//!
//! [`multiplier`] is the multiplier bootstrap of an asymptotically linear
//! estimator \[1\], here with Rademacher multipliers \[2\]. Given the estimate
//! and its influence function `psi`, each replicate is
//!
//! ```text
//! estimate + sum_i s_i psi_i / n,      s_i = +1 or -1 with probability 1/2
//! ```
//!
//! The signs come from the bits of `SplitMix64` outputs, 64 per output, lowest
//! bit first, a set bit meaning `+1`. The values are not centred.
//!
//! [`cluster_totals`] is the pairs cluster bootstrap \[3\]\[4\], reduced to what
//! an estimator that is linear in per-cluster totals needs. Each replicate
//! draws `G` of the `G` clusters with replacement. If cluster `g` is drawn
//! `k_g` times, then for each column `v` the replicate total is
//!
//! ```text
//! T = sum_g k_g v_g
//! ```
//!
//! The caller builds a ratio or other statistic from the totals. The counts
//! `k_g` come from `epistat_rng::resample_counts`, which draws each cluster
//! index with the multiply-and-shift mapping of `Stream::below`, so the
//! draws carry its negligible bias. The output holds `replicates * columns`
//! totals, replicate by replicate. All columns use the same counts within a
//! replicate.
//!
//! # Determinism
//!
//! Replicates are computed in parallel, each from its own seed, and written to
//! its own slot. Inside a replicate the sums run in cluster order. The result
//! is therefore identical, to the bit, for any number of threads.
//!
//! # Edge cases
//!
//! - An empty influence function is [`InferenceError::Empty`].
//! - No columns is [`InferenceError::Empty`] and columns of different lengths
//!   are [`InferenceError::Length`]. Columns of length zero give an empty
//!   result.
//! - Zero replicates give an empty result.
//!
//! # References
//!
//! 1. Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time
//!    periods. J Econom. 2021;225(2):200-30.
//! 2. Liu RY. Bootstrap procedures under some non-i.i.d. models. Ann Stat.
//!    1988;16(4):1696-1708.
//! 3. Efron B. Bootstrap methods: another look at the jackknife. Ann Stat.
//!    1979;7(1):1-26.
//! 4. Efron B, Tibshirani RJ. An introduction to the bootstrap. New York:
//!    Chapman & Hall; 1993.

use epistat_rng::{Stream, rademacher_sum, rademacher_sums, replicate_seed, resample_counts};
use rayon::prelude::*;

use crate::{InferenceError, check_length};

/// Replicates of a linear estimator under the Rademacher multiplier
/// bootstrap: `estimate + sum(s_i psi_i) / n` with independent signs `s_i`.
/// Replicate `r` draws its signs from `replicate_seed(seed, r)`.
///
/// # Errors
///
/// [`InferenceError::Empty`] for an empty influence function.
pub fn multiplier(
    estimate: f64,
    psi: &[f64],
    replicates: usize,
    seed: u64,
) -> Result<Vec<f64>, InferenceError> {
    if psi.is_empty() {
        return Err(InferenceError::Empty);
    }
    let n = psi.len() as f64;
    let mut out = vec![0.0; replicates];
    out.par_chunks_mut(LANES)
        .enumerate()
        .for_each(|(block, chunk)| {
            let first = (block * LANES) as u64;
            if chunk.len() == LANES {
                let mut streams: [Stream; LANES] =
                    std::array::from_fn(|l| Stream::seeded(replicate_seed(seed, first + l as u64)));
                let sums = rademacher_sums(psi, &mut streams);
                for (slot, sum) in chunk.iter_mut().zip(sums) {
                    *slot = estimate + sum / n;
                }
            } else {
                for (l, slot) in chunk.iter_mut().enumerate() {
                    let mut stream = Stream::seeded(replicate_seed(seed, first + l as u64));
                    *slot = estimate + rademacher_sum(psi, &mut stream) / n;
                }
            }
        });
    Ok(out)
}

/// Replicates drawn side by side in [`multiplier`].
const LANES: usize = 8;

/// Totals of per-cluster columns under the pairs cluster bootstrap: each
/// replicate draws `G` of the `G` clusters with replacement, and a cluster
/// drawn `k` times contributes `k` times its values. Replicate `r` draws from
/// `replicate_seed(seed, r)`.
///
/// Returns `replicates * columns.len()` totals, replicate by replicate.
///
/// # Errors
///
/// [`InferenceError::Empty`] for no columns and [`InferenceError::Length`]
/// when they differ in length.
pub fn cluster_totals(
    columns: &[&[f64]],
    replicates: usize,
    seed: u64,
) -> Result<Vec<f64>, InferenceError> {
    let clusters = columns.first().ok_or(InferenceError::Empty)?.len();
    for column in columns {
        check_length("per-cluster column", clusters, column.len())?;
    }
    if clusters == 0 {
        return Ok(Vec::new());
    }
    let width = columns.len();
    let mut totals = vec![0.0; replicates * width];
    totals
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(r, row)| {
            let counts = resample_counts(clusters, replicate_seed(seed, r as u64));
            for (slot, column) in row.iter_mut().zip(columns) {
                *slot = counts
                    .iter()
                    .zip(*column)
                    .map(|(&k, &v)| f64::from(k) * v)
                    .fold(0.0, |a, b| a + b);
            }
        });
    Ok(totals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_testkit::assert_same_on_threads;

    #[test]
    fn replicates_do_not_depend_on_threads() {
        let psi: Vec<f64> = (0..5_000).map(|i| f64::from(i % 13) - 6.0).collect();
        let columns: Vec<f64> = (0..300).map(|i| f64::from(i % 5)).collect();
        assert_same_on_threads(|| {
            let mut out = multiplier(1.0, &psi, 500, 42).unwrap();
            out.extend(cluster_totals(&[&columns, &columns], 200, 7).unwrap());
            out
        });
    }

    #[test]
    fn a_cluster_total_matches_its_resample_counts() {
        let values = [1.0, 10.0, 100.0];
        let totals = cluster_totals(&[&values], 3, 5).unwrap();
        for (r, &total) in totals.iter().enumerate() {
            let counts = resample_counts(3, replicate_seed(5, r as u64));
            let expected: f64 = counts
                .iter()
                .zip(values)
                .map(|(&k, v)| f64::from(k) * v)
                .sum();
            assert_eq!(total.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn multiplier_replicates_centre_on_the_estimate() {
        let psi = vec![1.0; 64];
        let draws = multiplier(3.0, &psi, 2_000, 1).unwrap();
        let mean = draws.iter().sum::<f64>() / 2_000.0;
        assert!((mean - 3.0).abs() < 0.01);
    }

    #[test]
    fn multiplier_equals_one_replicate_at_a_time() {
        for n in [1usize, 5, 64, 65, 1_000] {
            let psi: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 11 == 3 {
                        -0.0
                    } else {
                        (i as f64).cos() * 1e-2
                    }
                })
                .collect();
            for replicates in [0usize, 1, 7, 8, 9, 16, 17, 100] {
                let want: Vec<f64> = (0..replicates as u64)
                    .map(|r| {
                        let mut stream = Stream::seeded(replicate_seed(3, r));
                        0.25 + rademacher_sum(&psi, &mut stream) / n as f64
                    })
                    .collect();
                assert_same_on_threads(|| multiplier(0.25, &psi, replicates, 3).unwrap());
                let got = multiplier(0.25, &psi, replicates, 3).unwrap();
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "n = {n}, replicates = {replicates}"
                );
            }
        }
    }
}
