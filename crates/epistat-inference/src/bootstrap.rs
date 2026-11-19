//! Bootstrap replicates whose draws depend only on the seed and the
//! replicate number, so they are the same on any number of threads.

use epistat_rng::{Stream, rademacher_sum, replicate_seed, resample_counts};
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
    Ok((0..replicates as u64)
        .into_par_iter()
        .map(|r| estimate + rademacher_sum(psi, &mut Stream::seeded(replicate_seed(seed, r))) / n)
        .collect())
}

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
}
