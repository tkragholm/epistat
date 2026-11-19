//! Hamming and dynamic Hamming distance: the distances that factorise, and
//! k-medoids and the silhouette over them without a distance matrix.
//!
//! Both compare position `t` of one sequence with position `t` of the other and
//! never shift a sequence in time, so they are defined on sequences of one
//! length. That is what lets the medoid update and the silhouette reduce to
//! per-position symbol-weight tables, so nothing here is superlinear in the
//! number of sequences.
//!
//! # Algorithm
//!
//! With a substitution cost `c_t(a, b)` at position `t`, the distance of
//! sequences `x` and `y` of length `T` is
//!
//! ```text
//! d(x, y) = sum_{t : x_t != y_t} c_t(x_t, y_t).
//! ```
//!
//! Hamming distance \[1\] is `c_t = 1`, the number of mismatched positions.
//! Dynamic Hamming distance \[2\] lets the cost depend on the position. The
//! costs are a dense table of `T * alphabet * alphabet` values, indexed
//! `(t * alphabet + a) * alphabet + b`, which
//! [`transition_costs`](crate::transition_costs) derives from the data. The
//! terms are added in position order.
//!
//! # Factorisation
//!
//! For a cluster `C` with weights `w`, let `f_{C,t}(a)` be the summed weight of
//! members of `C` that carry symbol `a` at position `t`. Then the weighted
//! distance sum from a sequence `i` to the members of `C` is
//!
//! ```text
//! sum_{j in C} w_j d(i, j) = sum_t sum_{a != i_t} f_{C,t}(a) * c_t(i_t, a).
//! ```
//!
//! Under Hamming distance this is `sum_t (W_C - f_{C,t}(i_t))`, with `W_C` the
//! total weight of `C`. The two forms are equal in exact arithmetic and differ
//! in the last bit, so the code follows the shortcut when there is no cost
//! table and the full inner sum when there is one. In the full sum the symbols
//! of a position are added in the order they were first met scanning the members
//! in ascending index. A different order is a different float, and the medoid
//! is chosen by comparing these sums. Nothing is fused.
//!
//! The medoid update takes, for each cluster, the member with the least such
//! sum, the first in ascending index on a tie. The clusters are updated in
//! parallel and each is scanned serially, so the tie-break does not depend on
//! the thread count. The distance of every sequence to every cluster costs one
//! pass over the sequences and the tables, so the silhouette is linear too.
//!
//! # Edge cases
//!
//! - A panel needs a positive length and alphabet, a symbol count that divides
//!   into whole sequences, and no more than `2^32` sequences.
//! - A cost table that is not exactly `length * alphabet * alphabet` is
//!   refused, so a pair of symbols cannot take another pair's cost.
//! - A cluster with no member keeps its medoid.
//! - Distances to the nearest medoid are strict minima, so the earliest medoid
//!   wins a tie.
//!
//! # References
//!
//! 1. Hamming RW. Error detecting and error correcting codes. Bell Syst Tech
//!    J. 1950;29(2):147-60. doi:10.1002/j.1538-7305.1950.tb00463.x
//! 2. Lesnard L. Setting cost in optimal matching to uncover contemporaneous
//!    socio-temporal patterns. Sociol Methods Res. 2010;38(3):389-419.
//!    doi:10.1177/0049124110362526
//! 3. Kaufman L, Rousseeuw PJ. Finding groups in data: an introduction to
//!    cluster analysis. New York: Wiley; 1990. doi:10.1002/9780470316801

#![expect(
    clippy::cast_possible_truncation,
    reason = "sequence indices are bounded by the u32 limit checked when a panel or triangle is built, symbol ranks by the alphabet, and cluster ranks by the number of labels"
)]

use epistat_rng::Stream;
use rayon::prelude::*;

use crate::silhouette::{rank_labels, weighted_average};
use crate::{Attempt, Restarts, SequenceError, best_attempt};

/// One panel of equal-length rank-encoded sequences and the costs over them.
pub struct Panel<'a> {
    encoded: &'a [u8],
    /// The number of sequences.
    pub count: usize,
    /// Positions per sequence.
    pub length: usize,
    /// The number of states a symbol rank can take.
    pub alphabet: usize,
    /// Dense `length * alphabet * alphabet` substitution costs, or `None` for
    /// plain Hamming.
    ///
    /// `f64`, unlike the optimal-matching triangle: there are only
    /// `length * alphabet^2` of them, so the size is irrelevant, and they are
    /// usually transition rates derived in double precision.
    costs: Option<&'a [f64]>,
}

impl<'a> Panel<'a> {
    /// A panel over `encoded`, which holds sequences of `length` symbols one
    /// after another, each a rank below `alphabet`.
    ///
    /// # Errors
    ///
    /// [`SequenceError::Input`] when the length or alphabet is zero, the symbols
    /// do not divide into sequences of `length`, a symbol is outside the
    /// alphabet, the cost table is not dense, or there are more than `2^32`
    /// sequences.
    pub fn new(
        encoded: &'a [u8],
        length: usize,
        alphabet: usize,
        costs: Option<&'a [f64]>,
    ) -> Result<Self, SequenceError> {
        if length == 0 || alphabet == 0 {
            return Err(SequenceError::input(
                "a position-wise panel needs a positive length and alphabet",
            ));
        }
        if encoded.len() % length != 0 {
            return Err(SequenceError::input(format!(
                "{} encoded symbols do not divide into sequences of {length}; a \
                 position-wise distance is only defined on a common time axis",
                encoded.len()
            )));
        }
        if let Some(bad) = encoded.iter().find(|&&rank| rank as usize >= alphabet) {
            return Err(SequenceError::input(format!(
                "symbol rank {bad} is outside an alphabet of {alphabet}; the caller \
                 encoded against a different alphabet from the one it sent"
            )));
        }
        if let Some(table) = costs.filter(|table| table.len() != length * alphabet * alphabet) {
            return Err(SequenceError::input(format!(
                "cost table has {} entries for {length} positions over an alphabet \
                 of {alphabet}; it must be dense, or a symbol pair would silently \
                 take another pair's cost",
                table.len()
            )));
        }
        // Sequence indices are carried as `u32`, because a fit holds several
        // arrays of them. This is the only place a panel is built, so it is the
        // only place the bound is established.
        if encoded.len() / length > u32::MAX as usize {
            return Err(SequenceError::input(
                "more than 2^32 sequences; the fit indexes them with 32-bit integers",
            ));
        }
        Ok(Self {
            encoded,
            count: encoded.len() / length,
            length,
            alphabet,
            costs,
        })
    }

    /// The sequence at `index`.
    #[must_use]
    pub fn at(&self, index: usize) -> &[u8] {
        &self.encoded[index * self.length..(index + 1) * self.length]
    }

    /// `c_p(from, to)`: one for every mismatch under plain Hamming.
    fn cost(&self, position: usize, from: u8, to: u8) -> f64 {
        self.costs.map_or(1.0, |table| {
            table[(position * self.alphabet + from as usize) * self.alphabet + to as usize]
        })
    }

    /// Accumulated over positions in order, so a plain Hamming distance is a sum
    /// of exact ones and lands on the integer count.
    fn distance(&self, left: &[u8], right: &[u8]) -> f64 {
        let mut total = 0.0;
        for position in 0..self.length {
            let (from, to) = (left[position], right[position]);
            if from != to {
                total += self.cost(position, from, to);
            }
        }
        total
    }

    /// Rank of the closest medoid and the distance to it. Strictly less, so the
    /// earliest medoid attaining the minimum wins.
    fn nearest(&self, index: usize, medoids: &[u32]) -> (usize, f64) {
        let sequence = self.at(index);
        let mut best = (0usize, f64::INFINITY);
        for (rank, &medoid) in medoids.iter().enumerate() {
            let distance = self.distance(sequence, self.at(medoid as usize));
            if distance < best.1 {
                best = (rank, distance);
            }
        }
        best
    }
}

/// Per-position symbol weights for one set of members: the `f_{C,p}` above.
struct Frequencies {
    alphabet: usize,
    /// `length * alphabet`, indexed by position then symbol rank.
    weight: Vec<f64>,
    /// The symbols present at each position, in the order they were first seen
    /// scanning members in ascending index. The medoid cost adds its terms in
    /// this order.
    order: Vec<Vec<u8>>,
    total: f64,
}

impl Frequencies {
    fn build(panel: &Panel, members: &[u32], weights: &[f64]) -> Self {
        let mut weight = vec![0.0f64; panel.length * panel.alphabet];
        let mut seen = vec![false; panel.length * panel.alphabet];
        let mut order: Vec<Vec<u8>> = (0..panel.length).map(|_| Vec::new()).collect();
        let mut total = 0.0;
        for &member in members {
            let index = member as usize;
            total += weights[index];
            for (position, &symbol) in panel.at(index).iter().enumerate() {
                let slot = position * panel.alphabet + symbol as usize;
                if !seen[slot] {
                    seen[slot] = true;
                    order[position].push(symbol);
                }
                weight[slot] += weights[index];
            }
        }
        Self {
            alphabet: panel.alphabet,
            weight,
            order,
            total,
        }
    }

    /// Total weighted distance from `candidate` to every member.
    ///
    /// Two formulas. Under Hamming it is `W - f_p(c_p)` per position; under a
    /// general cost it is the full inner sum. They agree in exact arithmetic
    /// and not in the last bit, so the choice follows whether a cost table is
    /// present.
    // Not `mul_add`: the terms are a product then a sum, rounded twice.
    fn medoid_cost(&self, panel: &Panel, candidate: &[u8]) -> f64 {
        let mut cost = 0.0;
        for (position, &symbol) in candidate.iter().enumerate() {
            if panel.costs.is_none() {
                cost += self.total - self.weight[position * self.alphabet + symbol as usize];
            } else {
                for &other in &self.order[position] {
                    if other != symbol {
                        cost += self.weight[position * self.alphabet + other as usize]
                            * panel.cost(position, symbol, other);
                    }
                }
            }
        }
        cost
    }
}

/// Members of each cluster, in ascending index.
// `usize as u32` on a sequence index. `Panel::new` refuses a panel of more than
// `u32::MAX` sequences, so it cannot truncate.
fn members_by_cluster(labels: &[u32], k: usize) -> Vec<Vec<u32>> {
    let mut members: Vec<Vec<u32>> = (0..k).map(|_| Vec::new()).collect();
    for (index, &label) in labels.iter().enumerate() {
        members[label as usize].push(index as u32);
    }
    members
}

/// k-means++ seeding: `k` starting medoids, drawn proportional to `d^2 * w`.
///
/// The first is the head of the panel, so it depends on no draw. Each of the
/// rest costs exactly one uniform, and only when the remaining scores are not
/// all zero. A degenerate step pads deterministically and takes no draw.
///
/// The running total is accumulated left to right and compared with `>=`, and
/// the total is summed left to right too.
// `usize as u32` on sequence indices, bounded by `Panel::new`.
fn seed_medoids(panel: &Panel, weights: &[f64], k: usize, stream: &mut Stream) -> Vec<u32> {
    let mut medoids: Vec<u32> = vec![0];
    let mut chosen = vec![false; panel.count];
    chosen[0] = true;

    while medoids.len() < k {
        let scores: Vec<f64> = (0..panel.count)
            .into_par_iter()
            .map(|index| {
                let distance = panel.nearest(index, &medoids).1;
                distance * distance * weights[index]
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

/// Every restart of the k-medoids fit, and the best of them.
///
/// Restart `i` seeds by weighted k-means++ from `Stream::seeded(restarts.seeds()[i])`
/// and then alternates assignment and medoid update, to convergence or
/// `max_iterations`. The restarts run in parallel,
/// each with its own stream.
///
/// The [`Attempt`] holds the final medoids, the labels, the iterations run,
/// whether it converged, and the total weighted distance to the chosen
/// medoids, which is what the restarts are chosen on.
///
/// # Errors
///
/// [`SequenceError::Input`] when the weights do not match the panel one to
/// one, or `k` is zero or more than the sequences.
pub fn kmedoids(
    panel: &Panel<'_>,
    weights: &[f64],
    k: usize,
    restarts: &Restarts,
    max_iterations: usize,
) -> Result<Attempt, SequenceError> {
    if weights.len() != panel.count {
        return Err(SequenceError::input(format!(
            "{} sequences but {} weights; they must correspond one to one",
            panel.count,
            weights.len()
        )));
    }
    if k == 0 || k > panel.count {
        return Err(SequenceError::input(format!(
            "asked for {k} types from {} distinct sequences",
            panel.count
        )));
    }

    // The restarts are independent, and each owns a stream seeded from the
    // caller's, so the set is one deterministic fit.
    let attempts: Vec<Attempt> = epistat_optim::multistart::run(restarts.len(), |attempt| {
        let mut stream = Stream::seeded(restarts.seeds()[attempt]);
        one_fit(panel, weights, k, max_iterations, &mut stream)
    });
    best_attempt(attempts).ok_or_else(|| SequenceError::input("a fit needs at least one restart"))
}

/// One seeding and the alternating loop that follows it.
// `usize as u32` on sequence indices, bounded by `Panel::new`.
fn one_fit(
    panel: &Panel,
    weights: &[f64],
    k: usize,
    max_iterations: usize,
    stream: &mut Stream,
) -> Attempt {
    let mut medoids = seed_medoids(panel, weights, k, stream);
    let mut iterations = 0usize;
    let mut converged = false;

    {
        for iteration in 1..=max_iterations {
            iterations = iteration;
            let labels: Vec<u32> = (0..panel.count)
                .into_par_iter()
                .map(|index| panel.nearest(index, &medoids).0 as u32)
                .collect();

            // Parallel over clusters, serial within one. The candidate scan
            // takes the first member attaining the minimum, so splitting a
            // cluster across threads would need that tie-break reassembled.
            let chosen: Vec<Option<u32>> = members_by_cluster(&labels, k)
                .par_iter()
                .map(|members| {
                    let first = *members.first()?;
                    let frequencies = Frequencies::build(panel, members, weights);
                    let mut best = first;
                    let mut best_cost: Option<f64> = None;
                    for &candidate in members {
                        let cost = frequencies.medoid_cost(panel, panel.at(candidate as usize));
                        if best_cost.is_none_or(|previous| cost < previous) {
                            best = candidate;
                            best_cost = Some(cost);
                        }
                    }
                    Some(best)
                })
                .collect();

            let mut moved = false;
            for (cluster, choice) in chosen.iter().enumerate() {
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
        let assigned: Vec<(u32, f64)> = (0..panel.count)
            .into_par_iter()
            .map(|index| {
                let (rank, distance) = panel.nearest(index, &medoids);
                (rank as u32, distance)
            })
            .collect();
        let objective = assigned
            .iter()
            .zip(weights.iter())
            .fold(0.0f64, |total, ((_, distance), weight)| {
                total + distance * weight
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
}

/// `sum_{j in c} w_j d(i, j)` for every sequence `i` and cluster `c`.
///
/// Row-major by sequence, `k` columns. One symbol-weight table per cluster,
/// then one pass over the sequences, so the cost is linear in them.
fn cluster_sums(panel: &Panel, weights: &[f64], labels: &[u32], k: usize) -> Vec<f64> {
    let tables: Vec<Frequencies> = members_by_cluster(labels, k)
        .iter()
        .map(|members| Frequencies::build(panel, members, weights))
        .collect();

    let mut flat = vec![0.0f64; panel.count * k];
    flat.par_chunks_mut(k).enumerate().for_each(|(index, row)| {
        let sequence = panel.at(index);
        for (cluster, slot) in row.iter_mut().enumerate() {
            *slot = tables[cluster].medoid_cost(panel, sequence);
        }
    });
    flat
}

/// `sum_{j in c} w_j d(i, j)` for every sequence and cluster, checked.
///
/// The position-wise counterpart of
/// [`CondensedMatrix::cluster_distance_sums`](crate::CondensedMatrix::cluster_distance_sums),
/// with the same contract: `n * k` sums row-major by sequence, so the caller
/// picks the denominator. Linear rather than quadratic because the distance
/// factorises.
///
/// # Errors
///
/// [`SequenceError::Input`] when the weights or labels do not match the panel
/// one to one, or a label is not below `k`.
pub fn cluster_distance_sums(
    panel: &Panel<'_>,
    weights: &[f64],
    labels: &[u32],
    k: usize,
) -> Result<Vec<f64>, SequenceError> {
    if weights.len() != panel.count || labels.len() != panel.count {
        return Err(SequenceError::input(format!(
            "{} sequences but {} weights and {} labels; all three describe the same \
             sequences",
            panel.count,
            weights.len(),
            labels.len()
        )));
    }
    if let Some(bad) = labels.iter().find(|&&label| label as usize >= k) {
        return Err(SequenceError::input(format!(
            "cluster label {bad} is outside the {k} clusters declared"
        )));
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    Ok(cluster_sums(panel, weights, labels, k))
}

/// The weighted average silhouette width of a partition, under a position-wise
/// distance.
///
/// `labels` are the caller's cluster numbers, any integers; they are ranked
/// in ascending order. See [`crate::silhouette`] for the weighting. Returns 0.0
/// for fewer than two distinct labels or when no sequence can be scored.
///
/// # Errors
///
/// [`SequenceError::Input`] when the weights or labels do not match the panel
/// one to one.
pub fn silhouette(
    panel: &Panel<'_>,
    weights: &[f64],
    labels: &[i64],
) -> Result<f64, SequenceError> {
    if weights.len() != panel.count || labels.len() != panel.count {
        return Err(SequenceError::input(format!(
            "{} sequences but {} weights and {} labels; all three describe the same \
             sequences",
            panel.count,
            weights.len(),
            labels.len()
        )));
    }
    let (ranks, k) = rank_labels(labels);
    if k < 2 {
        return Ok(0.0);
    }
    let sums = cluster_sums(panel, weights, &ranks, k);
    Ok(weighted_average(&sums, k, |index| index, weights, &ranks))
}

#[cfg(test)]
mod tests {
    use super::{Panel, kmedoids, silhouette};
    use crate::Restarts;

    fn one() -> Restarts {
        Restarts::from_seed(1, 1).unwrap()
    }

    /// Twelve sequences with weights, and the partition `wcKMedoids` of
    /// WeightedCluster returned for them.
    const SEQUENCES: [&str; 12] = [
        "GGGEEEEE", "GGVVEEEE", "GGGGTTTE", "NNNNNNNN", "GGGNNNNN", "GGEEEEEE", "GGVVVEEE",
        "CGGGEEEE", "GNNNEEEE", "GGGGGTTT", "NNNGGGGG", "EEEEEEEE",
    ];
    const WEIGHTS: [f64; 12] = [
        4000.0, 3500.0, 40.0, 30.0, 8.0, 6.0, 5.0, 3.0, 2.0, 1.0, 1.0, 1.0,
    ];
    const PARTITION: [i64; 12] = [1, 2, 1, 4, 4, 1, 2, 1, 1, 1, 4, 1];

    /// Rank-encoded over the sorted alphabet.
    // `usize as u8` on a rank in an alphabet of five.
    fn encoded() -> (Vec<u8>, usize) {
        let mut alphabet: Vec<char> = SEQUENCES.iter().flat_map(|s| s.chars()).collect();
        alphabet.sort_unstable();
        alphabet.dedup();
        let bytes = SEQUENCES
            .iter()
            .flat_map(|s| s.chars())
            .map(|c| alphabet.iter().position(|&a| a == c).unwrap_or(0) as u8)
            .collect();
        (bytes, alphabet.len())
    }

    #[test]
    fn the_silhouette_is_weightedclusters_asww() {
        // `wcClusterQuality(d, cl, weights = w)$stats[["ASWw"]]`, WeightedCluster
        // 2.0, TraMineR 2.2.13, R 4.6.1, on Hamming distances.
        let (bytes, alphabet) = encoded();
        let panel = Panel::new(&bytes, 8, alphabet, None).expect("a valid panel");
        let value = silhouette(&panel, &WEIGHTS, &PARTITION).unwrap();
        assert!((value - 0.981_074_279_353).abs() < 1e-9, "{value}");
    }

    #[test]
    fn fewer_than_two_labels_score_zero() {
        let (bytes, alphabet) = encoded();
        let panel = Panel::new(&bytes, 8, alphabet, None).expect("a valid panel");
        assert!(silhouette(&panel, &WEIGHTS, &[3; 12]).unwrap().abs() < f64::EPSILON);
    }

    #[test]
    fn unit_costs_are_hamming() {
        // A dense cost table of ones is plain Hamming, through the general
        // branch of the medoid cost rather than the `W - f_p` shortcut.
        let (bytes, alphabet) = encoded();
        let ones = vec![1.0; 8 * alphabet * alphabet];
        let plain = Panel::new(&bytes, 8, alphabet, None).expect("a valid panel");
        let costed = Panel::new(&bytes, 8, alphabet, Some(&ones)).expect("a valid panel");
        let left = silhouette(&plain, &WEIGHTS, &PARTITION).unwrap();
        let right = silhouette(&costed, &WEIGHTS, &PARTITION).unwrap();
        assert!((left - right).abs() < 1e-12, "{left} against {right}");
    }

    #[test]
    fn two_clean_groups_are_found() {
        // Six sequences of 0 with one stray symbol each, six of 1, in a panel
        // of length 5, so the heaviest of each group is its medoid.
        let mut encoded = Vec::new();
        for group in 0..2u8 {
            for stray in 0..6usize {
                let mut sequence = [group; 5];
                sequence[stray % 5] = 2;
                encoded.extend_from_slice(&sequence);
            }
        }
        let panel = Panel::new(&encoded, 5, 3, None).unwrap();
        let weights: Vec<f64> = (0..12).map(|i| 1.0 + f64::from(i % 6)).collect();
        let four = Restarts::from_seed(3, 4).unwrap();
        let fit = kmedoids(&panel, &weights, 2, &four, 20).unwrap();
        assert!(fit.converged);
        assert_eq!(fit.labels.len(), 12);
        let first = fit.labels[0];
        assert!(fit.labels[..6].iter().all(|&label| label == first));
        assert!(fit.labels[6..].iter().all(|&label| label != first));
        let again = kmedoids(&panel, &weights, 2, &four, 20).unwrap();
        assert_eq!(fit, again);
        let ranked: Vec<i64> = fit.labels.iter().map(|&label| i64::from(label)).collect();
        assert!(silhouette(&panel, &weights, &ranked).unwrap() > 0.5);
    }

    #[test]
    fn explicit_seeds_keep_the_best_single_restart_fit() {
        let encoded: Vec<u8> = (0..12u8)
            .flat_map(|i| [i % 3, (i / 2) % 3, i % 2])
            .collect();
        let panel = Panel::new(&encoded, 3, 3, None).unwrap();
        let weights: Vec<f64> = (0..12).map(|i| 1.0 + f64::from(i % 4)).collect();
        let seeds = vec![5, 5 + 7919, 5 + 2 * 7919, 5 + 3 * 7919];
        let all = kmedoids(
            &panel,
            &weights,
            3,
            &Restarts::from_seeds(seeds.clone()).unwrap(),
            20,
        )
        .unwrap();
        let best = seeds
            .iter()
            .map(|&seed| {
                kmedoids(
                    &panel,
                    &weights,
                    3,
                    &Restarts::from_seeds(vec![seed]).unwrap(),
                    20,
                )
                .unwrap()
            })
            .reduce(|best, next| {
                if next.objective < best.objective {
                    next
                } else {
                    best
                }
            })
            .unwrap();
        assert_eq!(all, best);
    }

    #[test]
    fn invalid_panels_and_fits_are_refused() {
        assert!(Panel::new(&[0, 1, 0], 0, 2, None).is_err());
        assert!(Panel::new(&[0, 1, 0], 2, 2, None).is_err());
        assert!(Panel::new(&[0, 3], 2, 2, None).is_err());
        assert!(Panel::new(&[0, 1], 2, 2, Some(&[1.0; 7])).is_err());
        let panel = Panel::new(&[0, 1, 1, 0], 2, 2, None).unwrap();
        assert!(kmedoids(&panel, &[1.0, 1.0], 0, &one(), 5).is_err());
        assert!(kmedoids(&panel, &[1.0, 1.0], 3, &one(), 5).is_err());
        assert!(kmedoids(&panel, &[1.0], 1, &one(), 5).is_err());
    }
}
