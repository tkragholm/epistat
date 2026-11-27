//! The k-medoids fit shared by the position-wise panel and the
//! optimal-matching triangle, and the checks on its inputs.
//!
//! A distance supplies two things: the nearest medoid of a sequence, and the
//! most central member of each cluster. Everything else is common: k-means++
//! seeding, the alternating loop, the final assignment and the objective, and
//! the choice between restarts.

use epistat_rng::Stream;
use rayon::prelude::*;

use crate::error::{below, length};
use crate::{Attempt, Error, Restarts, best_attempt};

/// What a fit needs to know about a distance. Positions are the caller's
/// order of the sequences, the order the seeding walks and a tie between two
/// equally central members resolves in.
pub(crate) trait FitDistance: Sync {
    /// Rank of the closest medoid of the sequence at `index` and the distance
    /// to it. Strictly less, so the earliest medoid attaining the minimum
    /// wins.
    fn nearest(&self, index: usize, medoids: &[u32]) -> (usize, f64);

    /// The most central member of each cluster of `labels`, as positions, the
    /// earliest of two equally central members. A cluster with no member is
    /// `None`. Clusters beyond the highest label may be left out.
    fn medoids_from(&self, weights: &[f64], labels: &[u32], k: usize) -> Vec<Option<u32>>;
}

/// That there is one weight per sequence, and `count` the number of sequences.
pub(crate) fn check_weights(count: usize, weights: usize) -> Result<(), Error> {
    length("weights", count, weights)
}

/// That `k` types can be drawn from `count` sequences.
pub(crate) fn check_types(count: usize, k: usize) -> Result<(), Error> {
    if k == 0 || k > count {
        return Err(Error::invalid(format!(
            "asked for {k} types from {count} distinct sequences"
        )));
    }
    Ok(())
}

/// That `weights` and `labels` both have one entry per sequence.
pub(crate) fn check_lengths(count: usize, weights: usize, labels: usize) -> Result<(), Error> {
    length("weights", count, weights)?;
    length("labels", count, labels)
}

/// That a partition of `count` sequences into `k` clusters is well formed:
/// one weight and one label per sequence, and every label below `k`.
pub(crate) fn check_partition(
    count: usize,
    weights: usize,
    labels: &[u32],
    k: usize,
) -> Result<(), Error> {
    check_lengths(count, weights, labels.len())?;
    for &label in labels {
        below("labels", label as usize, k, "below the number of clusters")?;
    }
    Ok(())
}

/// Every restart of the fit, in parallel, and the best of them. Restart `i`
/// seeds from `Stream::seeded(restarts.seeds()[i])`.
pub(crate) fn fit<D: FitDistance>(
    distance: &D,
    count: usize,
    weights: &[f64],
    k: usize,
    restarts: &Restarts,
    max_iterations: usize,
) -> Result<Attempt, Error> {
    // The restarts are independent, and each owns a stream seeded from the
    // caller's, so the set is one deterministic fit.
    let attempts: Vec<Attempt> = epistat_optim::multistart::run(restarts.len(), |attempt| {
        let mut stream = Stream::seeded(restarts.seeds()[attempt]);
        one_fit(distance, count, weights, k, max_iterations, &mut stream)
    });
    best_attempt(attempts).ok_or_else(|| Error::invalid("a fit needs at least one restart"))
}

/// k-means++ seeding: `k` starting medoids, drawn proportional to `d^2 * w`.
///
/// The first is the head of the panel, so it depends on no draw. Each of the
/// rest costs exactly one uniform, and only when the remaining scores are not
/// all zero. A degenerate step pads deterministically and takes no draw.
///
/// The running total is accumulated left to right and compared with `>=`, and
/// the total is summed left to right too.
// `usize as u32` on sequence indices, which the callers bound by `u32::MAX`.
#[expect(clippy::cast_possible_truncation, reason = "indices are below 2^32")]
fn seed_medoids<D: FitDistance>(
    distance: &D,
    count: usize,
    weights: &[f64],
    k: usize,
    stream: &mut Stream,
) -> Vec<u32> {
    let mut medoids: Vec<u32> = vec![0];
    let mut chosen = vec![false; count];
    chosen[0] = true;

    while medoids.len() < k {
        let scores: Vec<f64> = (0..count)
            .into_par_iter()
            .map(|index| {
                let nearest = distance.nearest(index, &medoids).1;
                nearest * nearest * weights[index]
            })
            .collect();
        let total: f64 = scores.iter().sum();

        if total <= 0.0 {
            // Every remaining sequence is already a medoid, or is at distance
            // zero from one. Padded in index order, and no draw is taken.
            match chosen.iter().position(|taken| !taken) {
                Some(next) => {
                    medoids.push(next as u32);
                    chosen[next] = true;
                }
                None => break,
            }
            continue;
        }

        let target = stream.uniform() * total;
        let mut running = 0.0;
        for (index, contribution) in scores.iter().enumerate() {
            running += contribution;
            if running >= target {
                medoids.push(index as u32);
                chosen[index] = true;
                break;
            }
        }
    }
    medoids
}

/// One seeding and the alternating loop that follows it.
#[expect(clippy::cast_possible_truncation, reason = "ranks are below 2^32")]
fn one_fit<D: FitDistance>(
    distance: &D,
    count: usize,
    weights: &[f64],
    k: usize,
    max_iterations: usize,
    stream: &mut Stream,
) -> Attempt {
    let mut medoids = seed_medoids(distance, count, weights, k, stream);
    let mut iterations = 0usize;
    let mut converged = false;

    for iteration in 1..=max_iterations {
        iterations = iteration;
        let labels: Vec<u32> = (0..count)
            .into_par_iter()
            .map(|index| distance.nearest(index, &medoids).0 as u32)
            .collect();

        let mut moved = false;
        for (cluster, choice) in distance
            .medoids_from(weights, &labels, k)
            .iter()
            .enumerate()
        {
            if let Some(best) = choice.filter(|&best| best != medoids[cluster]) {
                medoids[cluster] = best;
                moved = true;
            }
        }
        if !moved {
            converged = true;
            break;
        }
    }

    // Assigned once more against the medoids the loop settled on, because a
    // run that stopped at `max_iterations` moved its medoids after the last
    // assignment. The distance to the chosen medoid is kept for the
    // objective, summed in index order.
    let assigned: Vec<(u32, f64)> = (0..count)
        .into_par_iter()
        .map(|index| {
            let (rank, nearest) = distance.nearest(index, &medoids);
            (rank as u32, nearest)
        })
        .collect();
    let objective = assigned
        .iter()
        .zip(weights.iter())
        .fold(0.0f64, |total, ((_, nearest), weight)| {
            total + nearest * weight
        });
    let labels: Vec<u32> = assigned.into_iter().map(|(rank, _)| rank).collect();
    Attempt {
        medoids,
        labels,
        iterations,
        converged,
        objective,
    }
}
