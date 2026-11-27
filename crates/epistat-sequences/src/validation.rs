//! Validation of a typology: agreement between partitions, the costs derived
//! from data, and the replicate loops that compare a fit with null data and
//! with refits.
//!
//! [`fit_typology`] fits one typology: it derives the distance's costs from
//! the weights, fits k-medoids with restarts and scores the fit by its
//! silhouette. [`null_silhouettes`] runs one (k, null model) batch of a parametric
//! bootstrap: draw null data, collapse it to distinct sequences, derive the
//! distance's costs from the null's own sequences, fit k-medoids and score the
//! fit by its silhouette. A typology whose silhouette is not above those of its
//! nulls has not found structure. [`refit_agreements`] runs one k of a
//! stability check: derive costs from each replicate's weights, refit, and
//! compare the refit with a reference partition by [`partition_agreement`].
//!
//! # Algorithm
//!
//! Partition agreement. With a contingency table of weights `n_ab` between the
//! labels `a` of the first partition and `b` of the second, row totals `n_a.`,
//! column totals `n_.b` and total `n`, and `P(x) = x (x - 1) / 2` extended to
//! real `x`, the weighted adjusted Rand index \[1\] is
//!
//! ```text
//! index    = sum_ab P(n_ab)
//! expected = sum_a P(n_a.) * sum_b P(n_.b) / P(n)
//! ceiling  = (sum_a P(n_a.) + sum_b P(n_.b)) / 2
//! ARI      = (index - expected) / (ceiling - expected)
//! ```
//!
//! The share agreeing matches types greedily: the cells of the table are taken
//! heaviest first, ties by the pair of labels, and a cell is kept when neither
//! of its labels is matched yet. The share is the weight of the kept cells over
//! `n`.
//!
//! Transition costs for dynamic Hamming distance \[2\]. With `p_s(a, b)` the
//! weighted rate of moving from `a` to `b` between positions `s` and `s + 1`,
//! zero where `a` is never left, the cost of the pair `a != b` at position `t`
//! is `2` minus the mean, over the steps into and out of the position, of
//! `p_s(a, b) + p_s(b, a)`. Position 0 has only the step out of it, and the
//! last position only the step into it. The pooled substitution costs for
//! optimal matching are `2 - (p(a, b) + p(b, a))` with rates pooled over every
//! step, as in the `TRATE` option of TraMineR \[3\]. The indel cost is half the
//! largest off-diagonal substitution cost, or 1 when there is none.
//!
//! Each replicate is independent, and the position-wise replicates run in
//! parallel. Optimal-matching replicates run one at a time, because each holds
//! a quadratic triangle and its construction and the fit over it are already
//! parallel. Replicate `r` draws and seeds its fit with
//! [`replicate_seed`]`(seed, r)`, so the result does not depend on the thread
//! count.
//!
//! # Summation order
//!
//! Keyed totals, such as the contingency cells and the row totals of a
//! transition, are accumulated in the order their keys were first seen, and
//! the sums over them are taken in that order. Those sums use
//! [`epistat_stats::sum::compensated`], Neumaier's compensated sum, whose order
//! of operations is fixed.
//!
//! # Edge cases
//!
//! - Empty input, or a total weight that is not positive, gives NaN for both
//!   agreement measures. The ARI is also NaN when the ceiling equals the
//!   expected value, as when both partitions are a single cluster.
//! - A null with fewer than `k` distinct sequences, or a refit with fewer than
//!   `k` sequences of positive weight, gives `None` for that replicate. So does
//!   a null whose fit has no silhouette, with fewer than two types.
//! - Negative and NaN weights are refused, as are weights, symbols and cost
//!   tables that do not match the panel.
//!
//! # References
//!
//! 1. Hubert L, Arabie P. Comparing partitions. J Classif. 1985;2(1):193-218.
//!    doi:10.1007/BF01908075
//! 2. Lesnard L. Setting cost in optimal matching to uncover contemporaneous
//!    socio-temporal patterns. Sociol Methods Res. 2010;38(3):389-419.
//!    doi:10.1177/0049124110362526
//! 3. Gabadinho A, Ritschard G, Müller NS, Studer M. Analyzing and visualizing
//!    state sequences in R with TraMineR. J Stat Softw. 2011;40(4):1-37.
//!    doi:10.18637/jss.v040.i04
//! 4. Studer M. WeightedCluster library manual: a practical guide to creating
//!    typologies of trajectories in the social sciences with R. LIVES Working
//!    Papers 24. Lausanne: NCCR LIVES; 2013.

#![expect(
    clippy::cast_possible_truncation,
    reason = "sequence indices are bounded by the u32 limit checked when a panel or triangle is built, symbol ranks by the alphabet, cluster ranks by the number of labels, and costs are narrowed to the single precision of the triangle"
)]

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use epistat_rng::replicate_seed;
use epistat_stats::sum::compensated;

use crate::error::{check_symbols, length, weights as check_weight_values};
use crate::nulls::{self, NullModel};
use crate::optimal_matching::{self, CondensedMatrix};
use crate::positionwise::{self, Panel};
use crate::{Attempt, Error, Restarts};
use epistat_core::InputError;

/// The distances a typology can be fitted with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distance {
    /// One per mismatched position.
    Hamming,
    /// Hamming with the position-specific costs of [`transition_costs`].
    DynamicHamming,
    /// Optimal matching under [`substitution_costs`] and [`indel_for`].
    OptimalMatching,
}

impl Distance {
    /// The distance named `name`: `"hamming"`, `"dhd"` (dynamic Hamming) or
    /// `"om"` (optimal matching).
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for any other name.
    pub fn named(name: &str) -> Result<Self, Error> {
        match name {
            "hamming" => Ok(Self::Hamming),
            "dhd" => Ok(Self::DynamicHamming),
            "om" => Ok(Self::OptimalMatching),
            other => Err(Error::invalid(format!(
                "unknown distance {other:?}; choose from \"dhd\", \"hamming\" and \"om\""
            ))),
        }
    }
}

/// The settings every fit in a batch shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FitSettings {
    /// The distance the fits use.
    pub distance: Distance,
    /// Number of types.
    pub k: usize,
    /// The number of restarts of a fit, at least one. The functions of this
    /// module fit once per null replicate or per refit, and each fit draws
    /// its restart seeds with [`Restarts::from_seed`]`(seed, restarts)` from
    /// its own seed: [`replicate_seed`]`(seed, r)` for null replicate `r` of
    /// [`null_silhouettes`], which is also the seed of that replicate's null
    /// draw, and `seed` itself for every refit of [`refit_agreements`] and for
    /// [`fit_typology`]. A caller that needs other restart seeds calls
    /// [`positionwise::kmedoids`] or [`CondensedMatrix::kmedoids`] with
    /// [`Restarts::from_seeds`].
    pub restarts: usize,
    /// Cap on the iterations of each seeding's alternating loop.
    pub max_iterations: usize,
}

/// A fitted typology: the best of the restarts and its silhouette.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Typology {
    /// The best attempt: its medoids, the cluster of each sequence, and the
    /// objective.
    pub attempt: Attempt,
    /// The weighted average silhouette width, or [`None`] when it is undefined:
    /// fewer than two types, or no sequence that can be scored.
    pub silhouette: Option<f64>,
}

/// Agreement between two partitions of the same weighted sequences.
#[derive(Clone, Debug, PartialEq)]
pub struct Agreement {
    /// Weighted adjusted Rand index, NaN when undefined.
    pub adjusted_rand: f64,
    /// Weighted share of sequences whose types correspond after the greedy
    /// relabelling.
    pub share_agreeing: f64,
    /// `(type in first, type in second)` in the order the greedy relabelling
    /// matched them.
    pub correspondence: Vec<(u32, u32)>,
    /// Distinct types in the first partition.
    pub k_first: usize,
    /// Distinct types in the second partition.
    pub k_second: usize,
}

/// Per-key totals that remember the order keys were first seen in.
///
/// The sums over them add in that order, so it is part of the result.
struct Ordered<K> {
    index: HashMap<K, usize>,
    keys: Vec<K>,
    totals: Vec<f64>,
}

impl<K: std::hash::Hash + Eq + Copy> Ordered<K> {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
            keys: Vec::new(),
            totals: Vec::new(),
        }
    }

    fn add(&mut self, key: K, value: f64) {
        if let Some(&at) = self.index.get(&key) {
            self.totals[at] += value;
        } else {
            self.index.insert(key, self.keys.len());
            self.keys.push(key);
            self.totals.push(0.0 + value);
        }
    }
}

/// `x (x - 1) / 2`, the pair count of `x` items extended to real `x`.
fn pairs(value: f64) -> f64 {
    value * (value - 1.0) / 2.0
}

/// Compare two labellings of the same weighted sequences.
///
/// The weighted adjusted Rand index and the greedy share agreeing, as defined in
/// the module documentation. NaN for both when the inputs are empty or the
/// total weight is not positive.
///
/// # Errors
///
/// [`Error::Input`] when the three inputs differ in length.
#[expect(
    clippy::float_cmp,
    reason = "a zero pair count and a ceiling equal to the expected value are the undefined cases"
)]
pub fn partition_agreement(
    first: &[u32],
    second: &[u32],
    weights: &[f64],
) -> Result<Agreement, Error> {
    length("second", first.len(), second.len())?;
    length("weights", first.len(), weights.len())?;
    if first.is_empty() {
        return Ok(Agreement {
            adjusted_rand: f64::NAN,
            share_agreeing: f64::NAN,
            correspondence: Vec::new(),
            k_first: 0,
            k_second: 0,
        });
    }

    let mut contingency: Ordered<(u32, u32)> = Ordered::new();
    let mut rows: Ordered<u32> = Ordered::new();
    let mut columns: Ordered<u32> = Ordered::new();
    let mut total = 0.0;
    for ((&left, &right), &weight) in first.iter().zip(second).zip(weights) {
        contingency.add((left, right), weight);
        rows.add(left, weight);
        columns.add(right, weight);
        total += weight;
    }
    let (k_first, k_second) = (rows.keys.len(), columns.keys.len());
    if total <= 0.0 {
        return Ok(Agreement {
            adjusted_rand: f64::NAN,
            share_agreeing: f64::NAN,
            correspondence: Vec::new(),
            k_first,
            k_second,
        });
    }

    let agreements = compensated(contingency.totals.iter().map(|&v| pairs(v)));
    let row_pairs = compensated(rows.totals.iter().map(|&v| pairs(v)));
    let column_pairs = compensated(columns.totals.iter().map(|&v| pairs(v)));
    let all_pairs = pairs(total);
    let expected = if all_pairs == 0.0 {
        0.0
    } else {
        row_pairs * column_pairs / all_pairs
    };
    #[expect(
        clippy::manual_midpoint,
        reason = "the sum is halved in plain arithmetic, which `f64::midpoint` does not promise"
    )]
    let ceiling = (row_pairs + column_pairs) / 2.0;
    let adjusted_rand = if ceiling == expected {
        f64::NAN
    } else {
        (agreements - expected) / (ceiling - expected)
    };

    // Heaviest cell first, ties by the pair of types, and a cell is kept when
    // neither of its types is matched yet.
    let mut cells: Vec<((u32, u32), f64)> = contingency
        .keys
        .iter()
        .copied()
        .zip(contingency.totals.iter().copied())
        .collect();
    cells.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut correspondence: Vec<(u32, u32)> = Vec::new();
    let mut matched_first: HashSet<u32> = HashSet::new();
    let mut matched_second: HashSet<u32> = HashSet::new();
    let mut matched = 0.0;
    for ((left, right), weight) in cells {
        if matched_first.contains(&left) || matched_second.contains(&right) {
            continue;
        }
        matched_first.insert(left);
        matched_second.insert(right);
        correspondence.push((left, right));
        matched += weight;
    }

    Ok(Agreement {
        adjusted_rand,
        share_agreeing: matched / total,
        correspondence,
        k_first,
        k_second,
    })
}

/// Weighted counts of moves between states, per group of steps, and the rates
/// they give.
///
/// A group is a step between two positions for [`transition_costs`] and the
/// whole sequence for [`substitution_costs`]. For each (group, state) the
/// targets are remembered in the order they were first seen, which is the order
/// the row total adds in.
struct Tally {
    alphabet: usize,
    counts: Vec<f64>,
    seen: Vec<bool>,
    order: Vec<Vec<u8>>,
}

impl Tally {
    fn new(groups: usize, alphabet: usize) -> Self {
        Self {
            alphabet,
            counts: vec![0.0; groups * alphabet * alphabet],
            seen: vec![false; groups * alphabet * alphabet],
            order: vec![Vec::new(); groups * alphabet],
        }
    }

    fn cell(&self, group: usize, from: usize, to: usize) -> usize {
        (group * self.alphabet + from) * self.alphabet + to
    }

    fn add(&mut self, group: usize, from: u8, to: u8, weight: f64) {
        let at = self.cell(group, usize::from(from), usize::from(to));
        if !self.seen[at] {
            self.seen[at] = true;
            self.order[group * self.alphabet + usize::from(from)].push(to);
        }
        self.counts[at] += weight;
    }

    /// The rate of each move, 0 where the state is never left in the group.
    fn rates(&self, groups: usize) -> Vec<f64> {
        let mut rates = vec![0.0f64; self.counts.len()];
        for group in 0..groups {
            for from in 0..self.alphabet {
                let targets = &self.order[group * self.alphabet + from];
                let total = compensated(
                    targets
                        .iter()
                        .map(|&to| self.counts[self.cell(group, from, usize::from(to))]),
                );
                if total > 0.0 {
                    for &to in targets {
                        let at = self.cell(group, from, usize::from(to));
                        rates[at] = self.counts[at] / total;
                    }
                }
            }
        }
        rates
    }
}

/// Position-specific substitution costs for dynamic Hamming distance.
///
/// Returns the dense `length * alphabet * alphabet` table the position-wise
/// fits read, `(position * alphabet + first) * alphabet + second`, with 0
/// on the diagonal. Every rank below `alphabet` is taken to occur. The cost of
/// a pair is 2 minus the mean, over the steps into and out of the position, of
/// the weighted transition rates in the two directions, as in the module
/// documentation.
///
/// # Errors
///
/// [`Error::Input`] when there is not one weight per sequence of the panel.
pub fn transition_costs(panel: &Panel<'_>, weights: &[f64]) -> Result<Vec<f64>, Error> {
    length("weights", panel.count, weights.len())?;
    let (length, alphabet) = (panel.length, panel.alphabet);
    let steps = length.saturating_sub(1);

    let mut tally = Tally::new(steps, alphabet);
    for (index, &weight) in weights.iter().enumerate() {
        let sequence = panel.at(index);
        for step in 0..steps {
            tally.add(step, sequence[step], sequence[step + 1], weight);
        }
    }
    let rates = tally.rates(steps);
    let cell = |step: usize, from: usize, to: usize| tally.cell(step, from, to);

    let mut costs = vec![0.0f64; length * alphabet * alphabet];
    for position in 0..length {
        let sides: Vec<usize> = [position.checked_sub(1), Some(position)]
            .into_iter()
            .flatten()
            .filter(|&step| step < steps)
            .collect();
        for first in 0..alphabet {
            for second in 0..alphabet {
                if first == second {
                    continue;
                }
                let mean = if sides.is_empty() {
                    0.0
                } else {
                    let observed = sides.iter().map(|&step| {
                        rates[cell(step, first, second)] + rates[cell(step, second, first)]
                    });
                    compensated(observed) / sides.len() as f64
                };
                costs[(position * alphabet + first) * alphabet + second] = 2.0 - mean;
            }
        }
    }
    Ok(costs)
}

/// The optimal-matching substitution matrix from transition rates pooled over
/// every step (the `"TRATE"` option of TraMineR).
///
/// `sequences` yields each rank-encoded sequence; they need not share a
/// length. Returns the dense `alphabet * alphabet` matrix, row-major, with 0
/// on the diagonal.
///
/// # Errors
///
/// [`Error::Input`] when there is not one weight per sequence or a symbol is
/// not below `alphabet`.
pub fn substitution_costs<'a>(
    sequences: impl Iterator<Item = &'a [u8]>,
    weights: &[f64],
    alphabet: usize,
) -> Result<Vec<f64>, Error> {
    let mut tally = Tally::new(1, alphabet);
    let mut sequences = sequences;
    let mut seen = 0_usize;
    while let Some(sequence) = sequences.next() {
        let Some(&weight) = weights.get(seen) else {
            return Err(InputError::Length {
                what: "weights",
                expected: seen + 1 + sequences.count(),
                found: weights.len(),
            }
            .into());
        };
        check_symbols("sequences", sequence, alphabet)?;
        for pair in sequence.windows(2) {
            tally.add(0, pair[0], pair[1], weight);
        }
        seen += 1;
    }
    length("weights", seen, weights.len())?;
    let rates = tally.rates(1);
    let mut costs = vec![0.0f64; alphabet * alphabet];
    for first in 0..alphabet {
        for second in 0..alphabet {
            if first != second {
                costs[first * alphabet + second] =
                    2.0 - (rates[first * alphabet + second] + rates[second * alphabet + first]);
            }
        }
    }
    Ok(costs)
}

/// Half the largest off-diagonal substitution cost, or 1.0 without one.
///
/// # Errors
///
/// [`Error::Input`] when `substitution` is not a dense `alphabet * alphabet`
/// matrix.
pub fn indel_for(substitution: &[f64], alphabet: usize) -> Result<f64, Error> {
    length("substitution", alphabet * alphabet, substitution.len())?;
    let mut largest: Option<f64> = None;
    for first in 0..alphabet {
        for second in 0..alphabet {
            if first != second {
                let cost = substitution[first * alphabet + second];
                // Keeps the first of equal values and replaces only on a
                // strictly greater one.
                if largest.is_none_or(|current| cost > current) {
                    largest = Some(cost);
                }
            }
        }
    }
    Ok(largest.map_or(1.0, |cost| cost / 2.0))
}

/// Duplicate sequences merged with summed weights, heaviest first, ties by the
/// sequence; then re-ranked onto the symbols that occur.
///
/// Returns the merged sequences rank-encoded and concatenated, their weights,
/// and the size of the reduced alphabet. The reduction matters: optimal
/// matching's indel cost is half the largest substitution cost over the
/// alphabet present.
fn collapse(
    drawn: &[u8],
    length: usize,
    weights: &[f64],
    alphabet: usize,
) -> (Vec<u8>, Vec<f64>, usize) {
    let mut totals: Ordered<&[u8]> = Ordered::new();
    for (sequence, &weight) in drawn.chunks_exact(length).zip(weights) {
        totals.add(sequence, weight);
    }
    let mut merged: Vec<(&[u8], f64)> = totals.keys.into_iter().zip(totals.totals).collect();
    merged.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(b.0))
    });

    let mut present = vec![false; alphabet];
    for &(sequence, _) in &merged {
        for &symbol in sequence {
            present[symbol as usize] = true;
        }
    }
    let mut rank = vec![0u8; alphabet];
    let mut size = 0usize;
    for (symbol, &occurs) in present.iter().enumerate() {
        if occurs {
            rank[symbol] = size as u8;
            size += 1;
        }
    }
    let encoded = merged
        .iter()
        .flat_map(|&(sequence, _)| sequence.iter().map(|&symbol| rank[symbol as usize]))
        .collect();
    let weights = merged.iter().map(|&(_, weight)| weight).collect();
    (encoded, weights, size)
}

/// Every sequence at `order`, gathered into one buffer in that order.
fn gather(panel: &Panel<'_>, order: &[u32]) -> Vec<u8> {
    order
        .iter()
        .flat_map(|&index| panel.at(index as usize).iter().copied())
        .collect()
}

/// One fit, with its costs derived from
/// `weights` over the panel in the panel's order.
///
/// `order` is the order the fit walks, a permutation of the panel, and
/// `fit_weights` the weights in that order. Returns the best attempt, its
/// labels in `order`'s order, and, when asked, the silhouette of the fit.
fn fit(
    panel: &Panel<'_>,
    weights: &[f64],
    order: &[u32],
    fit_weights: &[f64],
    settings: FitSettings,
    seed: u64,
    score: bool,
) -> Result<Typology, Error> {
    let FitSettings {
        distance,
        k,
        restarts,
        max_iterations,
    } = settings;
    let restarts = Restarts::from_seed(seed, restarts)?;
    if distance == Distance::OptimalMatching {
        let substitution = substitution_costs(
            (0..panel.count).map(|index| panel.at(index)),
            weights,
            panel.alphabet,
        )?;
        let indel = indel_for(&substitution, panel.alphabet)?;
        // The triangle holds single-precision values, so the costs are narrowed.
        let narrowed: Vec<f32> = substitution.iter().map(|&cost| cost as f32).collect();
        let indel = indel as f32;
        let lengths = vec![panel.length as u32; panel.count];
        let encoded = gather(panel, &(0..panel.count as u32).collect::<Vec<_>>());
        let matrix: CondensedMatrix =
            optimal_matching::condensed(&encoded, &lengths, &narrowed, panel.alphabet, indel)?;
        let attempt = matrix.kmedoids(order, fit_weights, k, &restarts, max_iterations)?;
        let silhouette = if score {
            matrix.silhouette(order, fit_weights, &attempt.labels)?
        } else {
            None
        };
        return Ok(Typology {
            attempt,
            silhouette,
        });
    }

    let costs = if distance == Distance::DynamicHamming {
        Some(transition_costs(panel, weights)?)
    } else {
        None
    };
    let encoded = gather(panel, order);
    let ordered = Panel::new(&encoded, panel.length, panel.alphabet, costs.as_deref())?;
    let attempt = positionwise::kmedoids(&ordered, fit_weights, k, &restarts, max_iterations)?;
    let silhouette = if score {
        positionwise::silhouette(&ordered, fit_weights, &attempt.labels)?
    } else {
        None
    };
    Ok(Typology {
        attempt,
        silhouette,
    })
}

/// A typology of the panel: costs derived from `weights`, a k-medoids fit with
/// `settings.restarts` restarts, and the silhouette of the best.
///
/// The sequences are fitted in the panel's order, which the seeding walks: the
/// first sequence is the first medoid, so a caller who wants the heaviest
/// sequence to seed the fit sorts the panel heaviest first. The costs of
/// [`Distance::DynamicHamming`] and [`Distance::OptimalMatching`] are derived
/// from `weights` over the same sequences ([`transition_costs`],
/// [`substitution_costs`] and [`indel_for`]). The restarts use the seeds of
/// [`Restarts::from_seed`]`(seed, settings.restarts)`.
///
/// # Errors
///
/// [`Error::Input`] when the weights do not match the panel or a weight is
/// negative or NaN, and [`Error::Invalid`] when `settings.restarts` is zero
/// or `settings.k` is zero or more than the sequences.
pub fn fit_typology(
    panel: &Panel<'_>,
    weights: &[f64],
    settings: FitSettings,
    seed: u64,
) -> Result<Typology, Error> {
    length("weights", panel.count, weights.len())?;
    check_weight_values("weights", weights)?;
    let order: Vec<u32> = (0..panel.count as u32).collect();
    fit(panel, weights, &order, weights, settings, seed, true)
}

/// That a fit has at least one restart, before any replicate is drawn.
fn check_restarts(settings: FitSettings) -> Result<(), Error> {
    if settings.restarts == 0 {
        return Err(Error::invalid("a fit needs at least one restart"));
    }
    Ok(())
}

/// Runs `replicate` for `0..replicates`, in parallel unless the distance holds
/// a quadratic triangle per replicate, and returns the results in order.
fn each_replicate<T: Send>(
    replicates: usize,
    distance: Distance,
    replicate: impl Fn(usize) -> Result<T, Error> + Sync + Send,
) -> Result<Vec<T>, Error> {
    if distance == Distance::OptimalMatching {
        (0..replicates).map(replicate).collect()
    } else {
        (0..replicates).into_par_iter().map(replicate).collect()
    }
}

/// The silhouettes of k-medoids fits to `replicates` null datasets.
///
/// Replicate `r` draws null sequences from `panel` and `weights` under `model`
/// (see [`nulls::null_sequences`]) with seed [`replicate_seed`]`(seed, r)`, collapses them to
/// distinct sequences with summed weights, derives the distance's costs from
/// them, and fits `settings.k` types with the same seed. Returns one entry per
/// replicate, in order: its silhouette, or `None` when the null had fewer than
/// k distinct sequences.
///
/// # Errors
///
/// [`Error::Input`] when the weights do not match the panel, a weight is
/// negative or NaN, there is no positive weight, or `k` is zero.
pub fn null_silhouettes(
    panel: &Panel<'_>,
    weights: &[f64],
    model: NullModel,
    settings: FitSettings,
    seed: u64,
    replicates: usize,
) -> Result<Vec<Option<f64>>, Error> {
    length("weights", panel.count, weights.len())?;
    check_weight_values("weights", weights)?;
    if settings.k == 0 {
        return Err(Error::invalid(
            "asked for 0 types; a typology has at least one",
        ));
    }
    check_restarts(settings)?;
    each_replicate(replicates, settings.distance, |replicate| {
        let seed = replicate_seed(seed, replicate as u64);
        let drawn = nulls::null_sequences(panel, weights, model, seed)?;
        let (encoded, merged, alphabet) = collapse(&drawn, panel.length, weights, panel.alphabet);
        if merged.len() < settings.k {
            return Ok(None);
        }
        let null = Panel::new(&encoded, panel.length, alphabet, None)?;
        let order: Vec<u32> = (0..null.count as u32).collect();
        let typology = fit(&null, &merged, &order, &merged, settings, seed, true)?;
        Ok(typology.silhouette)
    })
}

/// Agreement of refits under replicate weights with a reference partition.
///
/// `replicate_weights` holds one weight per sequence of `panel` for each
/// replicate, replicate by replicate. Each replicate with at least `k`
/// sequences of positive weight derives its costs from its weights, refits
/// with `seed` in the order heaviest first, ties by the sequence, and is
/// compared with `reference` under the full-data `weights`. Returns one entry
/// per replicate, in order: the adjusted Rand index and the share agreeing, or
/// `None` when the replicate had too few sequences of positive weight.
///
/// # Errors
///
/// [`Error::Input`] when the weights, the reference or the replicate
/// weights do not match the panel, a replicate weight is negative or NaN, or
/// `k` is zero or more than the sequences.
pub fn refit_agreements(
    panel: &Panel<'_>,
    weights: &[f64],
    replicate_weights: &[f64],
    reference: &[u32],
    settings: FitSettings,
    seed: u64,
) -> Result<Vec<Option<(f64, f64)>>, Error> {
    let count = panel.count;
    length("weights", count, weights.len())?;
    length("reference", count, reference.len())?;
    if count == 0 || !replicate_weights.len().is_multiple_of(count) {
        return Err(Error::invalid(format!(
            "{} replicate weights do not divide into replicates of {count} \
             sequences",
            replicate_weights.len()
        )));
    }
    check_weight_values("replicate_weights", replicate_weights)?;
    if settings.k == 0 || settings.k > count {
        return Err(Error::invalid(format!(
            "asked for {} types from {count} distinct sequences",
            settings.k
        )));
    }
    check_restarts(settings)?;
    let replicates = replicate_weights.len() / count;
    each_replicate(replicates, settings.distance, |replicate| {
        let drawn = &replicate_weights[replicate * count..(replicate + 1) * count];
        if drawn.iter().filter(|&&weight| weight > 0.0).count() < settings.k {
            return Ok(None);
        }
        let mut order: Vec<u32> = (0..count as u32).collect();
        order.sort_by(|&a, &b| {
            drawn[b as usize]
                .partial_cmp(&drawn[a as usize])
                .unwrap_or(Ordering::Equal)
                .then_with(|| panel.at(a as usize).cmp(panel.at(b as usize)))
        });
        let fit_weights: Vec<f64> = order.iter().map(|&index| drawn[index as usize]).collect();
        let labels = fit(panel, drawn, &order, &fit_weights, settings, seed, false)?
            .attempt
            .labels;
        let mut by_sequence = vec![0_u32; count];
        for (&index, &label) in order.iter().zip(&labels) {
            by_sequence[index as usize] = label;
        }
        let agreement = partition_agreement(reference, &by_sequence, weights)?;
        Ok(Some((agreement.adjusted_rand, agreement.share_agreeing)))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Distance, FitSettings, NullModel, Panel, Restarts, collapse, fit_typology, indel_for,
        null_silhouettes, partition_agreement, refit_agreements, substitution_costs,
        transition_costs,
    };
    use crate::{nulls, positionwise};
    use epistat_rng::replicate_seed;

    #[test]
    fn identical_partitions_agree_fully() {
        let agreement =
            partition_agreement(&[0, 0, 1, 1], &[5, 5, 7, 7], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((agreement.adjusted_rand - 1.0).abs() < 1e-12);
        assert!((agreement.share_agreeing - 1.0).abs() < 1e-12);
        assert_eq!(agreement.correspondence, vec![(1, 7), (0, 5)]);
        assert_eq!((agreement.k_first, agreement.k_second), (2, 2));
    }

    #[test]
    fn empty_and_weightless_partitions_are_undefined() {
        let empty = partition_agreement(&[], &[], &[]).unwrap();
        assert!(empty.adjusted_rand.is_nan() && empty.share_agreeing.is_nan());
        let weightless = partition_agreement(&[0, 1], &[1, 0], &[0.0, 0.0]).unwrap();
        assert!(weightless.adjusted_rand.is_nan());
        assert_eq!((weightless.k_first, weightless.k_second), (2, 2));
        assert!(partition_agreement(&[0], &[0, 1], &[1.0]).is_err());
    }

    #[test]
    fn states_that_always_follow_one_another_are_cheap() {
        // A always becomes B, so the pair costs 2 - 1 = 1 at every position,
        // and C never meets either, so its pairs cost 2.
        let encoded = [0u8, 1, 0, 1, 2, 2];
        let panel = Panel::new(&encoded, 2, 3, None).unwrap();
        let costs = transition_costs(&panel, &[1.0, 2.0, 1.0]).unwrap();
        let at = |position: usize, first: usize, second: usize| {
            costs[(position * 3 + first) * 3 + second]
        };
        for position in 0..2 {
            assert!((at(position, 0, 1) - 1.0).abs() < 1e-15);
            assert!((at(position, 1, 0) - 1.0).abs() < 1e-15);
            assert!((at(position, 0, 2) - 2.0).abs() < 1e-15);
            assert!(at(position, 2, 2).abs() < 1e-15);
        }
        let pooled = substitution_costs(encoded.chunks(2), &[1.0, 2.0, 1.0], 3).unwrap();
        assert!((pooled[1] - 1.0).abs() < 1e-15);
        assert!((pooled[2] - 2.0).abs() < 1e-15);
    }

    fn settings(distance: Distance, k: usize) -> FitSettings {
        FitSettings {
            distance,
            k,
            restarts: 2,
            max_iterations: 10,
        }
    }

    #[test]
    fn distances_are_named_and_unknown_names_refused() {
        assert_eq!(Distance::named("hamming").unwrap(), Distance::Hamming);
        assert_eq!(Distance::named("dhd").unwrap(), Distance::DynamicHamming);
        assert_eq!(Distance::named("om").unwrap(), Distance::OptimalMatching);
        assert!(Distance::named("lcs").is_err());
    }

    #[test]
    fn every_distance_gives_one_silhouette_per_null_replicate() {
        let encoded: Vec<u8> = (0..40u8).map(|i| (i / 3 + i % 2) % 3).collect();
        let panel = Panel::new(&encoded, 4, 3, None).unwrap();
        let weights: Vec<f64> = (0..10).map(|i| 1.0 + f64::from(i)).collect();
        for distance in [
            Distance::Hamming,
            Distance::DynamicHamming,
            Distance::OptimalMatching,
        ] {
            let silhouettes = null_silhouettes(
                &panel,
                &weights,
                NullModel::FirstOrderMarkov,
                settings(distance, 2),
                7,
                3,
            )
            .unwrap();
            assert_eq!(silhouettes.len(), 3);
            let again = null_silhouettes(
                &panel,
                &weights,
                NullModel::FirstOrderMarkov,
                settings(distance, 2),
                7,
                3,
            )
            .unwrap();
            let bits = |values: &[Option<f64>]| -> Vec<Option<u64>> {
                values.iter().map(|value| value.map(f64::to_bits)).collect()
            };
            assert_eq!(bits(&silhouettes), bits(&again));
        }
        assert!(
            null_silhouettes(
                &panel,
                &weights,
                NullModel::StateIndependence,
                settings(Distance::Hamming, 0),
                1,
                1
            )
            .is_err()
        );
        let mut negative = weights.clone();
        negative[3] = -1.0;
        assert!(
            null_silhouettes(
                &panel,
                &negative,
                NullModel::StateIndependence,
                settings(Distance::Hamming, 2),
                1,
                1
            )
            .is_err()
        );
    }

    #[test]
    fn a_refit_of_the_same_weights_agrees_with_a_matching_reference() {
        // Two groups of five identical sequences, so any refit with k = 2
        // recovers them and the reference partition is the groups.
        let mut encoded = Vec::new();
        for group in 0..2u8 {
            for _ in 0..5 {
                encoded.extend_from_slice(&[group; 4]);
            }
        }
        let panel = Panel::new(&encoded, 4, 2, None).unwrap();
        let weights = vec![1.0; 10];
        let reference: Vec<u32> = (0..10).map(|i| u32::from(i >= 5)).collect();
        let replicate_weights = vec![2.0; 20];
        let result = refit_agreements(
            &panel,
            &weights,
            &replicate_weights,
            &reference,
            settings(Distance::Hamming, 2),
            3,
        )
        .unwrap();
        assert_eq!(result.len(), 2);
        for entry in result {
            let (rand, share) = entry.expect("enough sequences");
            assert!((rand - 1.0).abs() < 1e-12);
            assert!((share - 1.0).abs() < 1e-12);
        }
        // Fewer positive weights than types gives no entry.
        let sparse = [vec![1.0], vec![0.0; 9]].concat();
        let none = refit_agreements(
            &panel,
            &weights,
            &sparse,
            &reference,
            settings(Distance::Hamming, 2),
            3,
        )
        .unwrap();
        assert_eq!(none, vec![None]);
    }

    /// The restart seeds of a null replicate are `Restarts::from_seed(seed, n)`
    /// of that replicate's seed, so a loop written out by hand over the same
    /// replicates gives the same silhouettes.
    #[test]
    fn the_null_loop_equals_a_hand_rolled_loop() {
        let encoded: Vec<u8> = (0..40u8).map(|i| (i / 3 + i % 2) % 3).collect();
        let panel = Panel::new(&encoded, 4, 3, None).unwrap();
        let weights: Vec<f64> = (0..10).map(|i| 1.0 + f64::from(i)).collect();
        let model = NullModel::FirstOrderMarkov;
        let given = FitSettings {
            distance: Distance::Hamming,
            k: 2,
            restarts: 3,
            max_iterations: 10,
        };
        let got = null_silhouettes(&panel, &weights, model, given, 7, 4).unwrap();
        let by_hand: Vec<Option<f64>> = (0..4u64)
            .map(|r| {
                let seed = replicate_seed(7, r);
                let drawn = nulls::null_sequences(&panel, &weights, model, seed).unwrap();
                let (encoded, merged, alphabet) = collapse(&drawn, 4, &weights, 3);
                if merged.len() < 2 {
                    return None;
                }
                let null = Panel::new(&encoded, 4, alphabet, None).unwrap();
                let restarts = Restarts::from_seed(seed, 3).unwrap();
                let fit = positionwise::kmedoids(&null, &merged, 2, &restarts, 10).unwrap();
                positionwise::silhouette(&null, &merged, &fit.labels).unwrap()
            })
            .collect();
        let bits = |v: &[Option<f64>]| -> Vec<Option<u64>> {
            v.iter().map(|x| x.map(f64::to_bits)).collect()
        };
        assert_eq!(bits(&got), bits(&by_hand));
    }

    #[test]
    fn a_typology_is_the_chain_of_its_parts() {
        let encoded: Vec<u8> = (0..40u8).map(|i| (i / 3 + i % 2) % 3).collect();
        let panel = Panel::new(&encoded, 4, 3, None).unwrap();
        let weights: Vec<f64> = (0..10).map(|i| 10.0 - f64::from(i)).collect();
        let costs = transition_costs(&panel, &weights).unwrap();
        let costed = Panel::new(&encoded, 4, 3, Some(&costs)).unwrap();
        let restarts = Restarts::from_seed(5, 3).unwrap();
        let attempt = positionwise::kmedoids(&costed, &weights, 2, &restarts, 10).unwrap();
        let silhouette = positionwise::silhouette(&costed, &weights, &attempt.labels).unwrap();
        let given = FitSettings {
            distance: Distance::DynamicHamming,
            k: 2,
            restarts: 3,
            max_iterations: 10,
        };
        let typology = fit_typology(&panel, &weights, given, 5).unwrap();
        assert_eq!(typology.attempt, attempt);
        assert_eq!(
            typology.silhouette.map(f64::to_bits),
            silhouette.map(f64::to_bits)
        );
        for distance in [Distance::Hamming, Distance::OptimalMatching] {
            let settings = FitSettings { distance, ..given };
            assert_eq!(
                fit_typology(&panel, &weights, settings, 5)
                    .unwrap()
                    .attempt
                    .labels
                    .len(),
                10
            );
        }
    }

    #[test]
    fn a_fit_with_one_type_has_no_silhouette() {
        let encoded: Vec<u8> = (0..40u8).map(|i| (i / 3 + i % 2) % 3).collect();
        let panel = Panel::new(&encoded, 4, 3, None).unwrap();
        let weights = vec![1.0; 10];
        let typology = fit_typology(&panel, &weights, settings(Distance::Hamming, 1), 5).unwrap();
        assert_eq!(typology.silhouette, None);
    }

    #[test]
    fn cost_helpers_refuse_what_does_not_fit() {
        let encoded = [0u8, 1, 0, 1, 2, 2];
        let panel = Panel::new(&encoded, 2, 3, None).unwrap();
        assert!(transition_costs(&panel, &[1.0, 1.0]).is_err());
        assert!(transition_costs(&panel, &[1.0; 4]).is_err());
        assert!(substitution_costs(encoded.chunks(2), &[1.0, 1.0], 3).is_err());
        assert!(substitution_costs(encoded.chunks(2), &[1.0; 4], 3).is_err());
        assert!(substitution_costs(encoded.chunks(2), &[1.0; 3], 2).is_err());
        assert!(indel_for(&[0.0, 1.0, 1.0], 2).is_err());
        assert_eq!(indel_for(&[0.0, 1.0, 1.0, 0.0], 2).unwrap(), 0.5);
        assert_eq!(indel_for(&[0.0], 1).unwrap(), 1.0);
    }
}
