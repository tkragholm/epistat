//! The between-within split of columns into cluster means and deviations.

use epistat_core::check;
use epistat_inference::Clusters;
use epistat_linalg::{Mat, MatRef, ROW_BLOCK};
use rayon::prelude::*;

use crate::Error;

/// The clusters handled by one task.
const CLUSTERS_PER_TASK: usize = 4096;

/// The family means of the columns of `x` (one row per family, in the order of
/// the clusters) and the deviations from them (one row per row of `x`), for a
/// between-within model.
///
/// Entering both as covariates in a model separates the within-family effect,
/// the coefficient of the deviation, from the between-family contextual
/// effect, that of the mean (Neuhaus JM, Kalbfleisch JD. Between- and
/// within-cluster covariate effects in the analysis of clustered data.
/// Biometrics. 1998;54(2):638-45).
///
/// Each family is summed by one task over its rows in row order, so the result
/// does not depend on the number of threads. A family whose values of a
/// column are all equal, as doubles, has that value as its mean and deviation
/// exactly 0 in it, not a deviation of the order of rounding.
///
/// # Errors
///
/// [`InputError::Length`](epistat_core::InputError::Length) when `clusters`
/// covers another number of rows.
pub fn between_within(
    x: MatRef<'_, f64>,
    clusters: &Clusters,
) -> Result<(Mat<f64>, Mat<f64>), Error> {
    check::same_length("clusters", x.nrows(), clusters.len())?;
    let (n, g, columns) = (x.nrows(), clusters.count(), x.ncols());
    let of = clusters.of();
    // The rows of each cluster: the ranges `bounds[c]..bounds[c + 1]` of
    // `order`, or of the rows themselves when the cluster numbers do not
    // decrease along the rows. Rows keep their order within a cluster.
    let mut bounds = vec![0usize; g + 1];
    for &c in of {
        bounds[c as usize + 1] += 1;
    }
    for c in 0..g {
        bounds[c + 1] += bounds[c];
    }
    let sorted = of.windows(2).all(|pair| pair[0] <= pair[1]);
    let order = (!sorted).then(|| {
        let mut next = bounds.clone();
        let mut order = vec![0usize; n];
        for (i, &c) in of.iter().enumerate() {
            let c = c as usize;
            order[next[c]] = i;
            next[c] += 1;
        }
        order
    });
    let row = |k: usize| order.as_ref().map_or(k, |order| order[k]);
    let mut means = vec![0.0; g * columns];
    let mut constant = vec![true; g * columns];
    let width = (CLUSTERS_PER_TASK * columns).max(1);
    means
        .par_chunks_mut(width)
        .zip(constant.par_chunks_mut(width))
        .enumerate()
        .for_each(|(block, (means, constant))| {
            let first_cluster = block * CLUSTERS_PER_TASK;
            let mut first = vec![0.0; columns];
            let mut sums = vec![0.0; columns];
            for c in 0..means.len() / columns.max(1) {
                let (lo, hi) = (bounds[first_cluster + c], bounds[first_cluster + c + 1]);
                let flags = &mut constant[c * columns..(c + 1) * columns];
                sums.fill(0.0);
                let (mut count, mut total) = (0usize, 0.0);
                for k in lo..hi {
                    let i = row(k);
                    for j in 0..columns {
                        let v = x[(i, j)];
                        if count == 0 {
                            first[j] = v;
                        } else if v.to_bits() != first[j].to_bits() {
                            flags[j] = false;
                        }
                        sums[j] += v;
                    }
                    total += 1.0;
                    count += 1;
                }
                for j in 0..columns {
                    means[c * columns + j] = if count == 0 {
                        0.0
                    } else if flags[j] {
                        first[j]
                    } else {
                        sums[j] / total
                    };
                }
            }
        });
    let mut deviations = Mat::<f64>::zeros(n, columns);
    for j in 0..columns {
        deviations
            .col_as_slice_mut(j)
            .par_chunks_mut(ROW_BLOCK)
            .enumerate()
            .for_each(|(block, chunk)| {
                for (k, out) in chunk.iter_mut().enumerate() {
                    let i = block * ROW_BLOCK + k;
                    let c = of[i] as usize;
                    if !constant[c * columns + j] {
                        *out = x[(i, j)] - means[c * columns + j];
                    }
                }
            });
    }
    let means = Mat::from_fn(g, columns, |c, j| means[c * columns + j]);
    Ok((means, deviations))
}
