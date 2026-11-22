//! Match weights from matched pairs, and the effective sample size.
//!
//! # Weights
//!
//! For the ATT every matched anchor weighs 1 and a candidate weighs the sum,
//! over the pairs it is in, of `1 / m` with `m` the number of candidates of the
//! pair's anchor. For the ATC the roles are reversed. A candidate used by two
//! anchors that each have two candidates weighs `1/2 + 1/2 = 1`. For the ATE
//! and the ATM every pair adds 1 to both its units. These are the raw weights
//! of `matchit(normalize = FALSE)`; [`UnitWeightSet::normalized`] divides each
//! group's positive weights by their mean, as `matchit()` does by default.
//! Sums run in the order of the unit ids, so they do not depend on the hash
//! order.
//!
//! # Effective sample size
//!
//! Kish's effective sample size of weights `w_i` is
//!
//! ```text
//! n_eff = (sum_i w_i)^2 / sum_i w_i^2
//! ```
//!
//! It does not depend on the scale of the weights. A weight of 0 is a unit that
//! is not in the sample. When the squares overflow or lose precision the
//! weights are divided by the largest first.
//!
//! # References
//!
//! 1. Kish L. Survey Sampling. Wiley, 1965, p. 427.
//! 2. Stuart EA. Matching methods for causal inference: a review and a look
//!    forward. Statistical Science 2010;25:1-21.

use crate::types::{Estimand, MatchOutcome, MatchedPair};
use itertools::Itertools;
use rapidhash::RapidHashMap;
use serde::{Deserialize, Serialize};

/// Match-weight extraction method for matched pairs.
///
/// The weights are raw: no group is rescaled. `matchit()` rescales by default
/// (`normalize = TRUE`); [`UnitWeightSet::normalized`] does the same here.
/// Raw weights equal `matchit(normalize = FALSE)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchWeightMethod {
    /// Every pair contributes +1 to both anchor and candidate units.
    ///
    /// `matchit()` has no such weighting. Its weights for `estimand = "ATE"`
    /// are stratum weights, which equal these only when every anchor has the
    /// same number of controls and both are normalized.
    PairCount,
    /// Every matched anchor has weight 1, candidate pairs are divided by anchor match count.
    ///
    /// This is `matchit()`'s weighting for the estimand whose focal group is
    /// the anchors, with or without replacement.
    AnchorUnitCandidateFractional,
    /// Every matched candidate has weight 1, anchor pairs are divided by candidate reuse count.
    ///
    /// This is `matchit()`'s weighting for the estimand whose focal group is
    /// the candidates, applied to pairs that were formed from the anchors'
    /// side. `matchit()` itself would form the pairs from the candidates'
    /// side.
    CandidateUnitAnchorFractional,
}

impl MatchWeightMethod {
    /// Resolve a default weighting method from an estimand.
    #[must_use]
    pub const fn for_estimand(estimand: Estimand) -> Self {
        match estimand {
            Estimand::Att => Self::AnchorUnitCandidateFractional,
            Estimand::Atc => Self::CandidateUnitAnchorFractional,
            Estimand::Ate | Estimand::Atm => Self::PairCount,
        }
    }
}

/// Pair-level weight entry aligned to one matched pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairWeight {
    /// Matched pair identifier tuple.
    pub pair: MatchedPair,
    /// Pair-level weight.
    pub weight: f64,
}

/// Pair-level weight output.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PairWeightSet {
    /// Pair-level entries in the same order as source pairs.
    pub pairs: Vec<PairWeight>,
}

impl PairWeightSet {
    /// Total pair-level weight.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.pairs.iter().map(|entry| entry.weight).sum()
    }

    /// Convert pair-level weights to a flat table shape.
    #[must_use]
    pub fn to_table(&self) -> PairWeightTable {
        PairWeightTable {
            rows: self
                .pairs
                .iter()
                .map(|entry| PairWeightRow {
                    anchor_id: entry.pair.anchor_id().to_string(),
                    candidate_id: entry.pair.comparator_id().to_string(),
                    pair_weight: entry.weight,
                })
                .collect(),
        }
    }
}

/// Flat row for pair-level table outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairWeightRow {
    /// Anchor-side unit identifier.
    pub anchor_id: String,
    /// Candidate-side unit identifier.
    pub candidate_id: String,
    /// Pair-level weight.
    pub pair_weight: f64,
}

/// Pair-weight rows for dataframe/table ingestion.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PairWeightTable {
    /// Flat pair-level rows.
    pub rows: Vec<PairWeightRow>,
}

impl PairWeightTable {
    /// Total pair-level weight.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.rows.iter().map(|row| row.pair_weight).sum()
    }
}

/// Unit role used by flat unit-weight tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum UnitRole {
    /// Anchor-side unit.
    Anchor,
    /// Candidate-side unit.
    Candidate,
}

/// Flat row for unit-weight table outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnitWeightRow {
    /// Unit identifier.
    pub unit_id: String,
    /// Anchor/candidate role.
    pub role: UnitRole,
    /// Pure matching-derived unit weight.
    pub match_weight: f64,
    /// Optional sampling weight used during composition.
    pub sampling_weight: Option<f64>,
    /// Product weight (`match_weight * sampling_weight` if supplied).
    pub analysis_weight: f64,
}

/// Unit-weight rows for dataframe/table ingestion.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UnitWeightTable {
    /// Flat unit-level rows.
    pub rows: Vec<UnitWeightRow>,
}

impl UnitWeightTable {
    /// Total analysis weight across all rows.
    #[must_use]
    pub fn total_analysis_weight(&self) -> f64 {
        self.rows.iter().map(|row| row.analysis_weight).sum()
    }
}

/// Unit-level weights split by anchor and candidate groups.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UnitWeightSet {
    /// Weights keyed by anchor identifier.
    pub anchor: RapidHashMap<String, f64>,
    /// Weights keyed by candidate identifier.
    pub candidate: RapidHashMap<String, f64>,
}

impl UnitWeightSet {
    /// Compose match weights with external sampling weights (`analysis = match * sampling`).
    ///
    /// Missing sampling weights default to `1.0`.
    #[must_use]
    pub fn composed_with_sampling(&self, sampling_weights: &RapidHashMap<String, f64>) -> Self {
        Self {
            anchor: compose_with_sampling(&self.anchor, sampling_weights),
            candidate: compose_with_sampling(&self.candidate, sampling_weights),
        }
    }

    /// Rescale the weights of each group so that their mean is 1.
    ///
    /// Within the anchor group and within the candidate group separately,
    /// every positive weight is divided by the mean of the positive weights of
    /// that group, so that the weights of a group sum to the number of its
    /// units that have one. This is the rule of `matchit()` with
    /// `normalize = TRUE`. It changes the scale of the weights and not their
    /// ratios, so weighted means are unchanged. A group with no positive
    /// weight is left as it is.
    ///
    /// The mean is formed in the order of the unit identifiers, so the result
    /// does not depend on the order of the hash map.
    #[must_use]
    pub fn normalized(&self) -> Self {
        Self {
            anchor: normalize_group(&self.anchor),
            candidate: normalize_group(&self.candidate),
        }
    }

    /// Effective sample size for anchor weights, summed in the order of the
    /// unit identifiers. See [`effective_sample_size`].
    ///
    /// # Errors
    ///
    /// [`InvalidWeightError`] for a weight that is negative or not finite.
    pub fn anchor_effective_sample_size(&self) -> Result<Option<f64>, InvalidWeightError> {
        effective_sample_size(in_id_order(&self.anchor))
    }

    /// Effective sample size for candidate weights, summed in the order of the
    /// unit identifiers. See [`effective_sample_size`].
    ///
    /// # Errors
    ///
    /// [`InvalidWeightError`] for a weight that is negative or not finite.
    pub fn candidate_effective_sample_size(&self) -> Result<Option<f64>, InvalidWeightError> {
        effective_sample_size(in_id_order(&self.candidate))
    }

    /// Sum of all anchor-side unit weights, in the order of the unit
    /// identifiers.
    #[must_use]
    pub fn anchor_total_weight(&self) -> f64 {
        in_id_order(&self.anchor).into_iter().sum()
    }

    /// Sum of all candidate-side unit weights, in the order of the unit
    /// identifiers.
    #[must_use]
    pub fn candidate_total_weight(&self) -> f64 {
        in_id_order(&self.candidate).into_iter().sum()
    }

    /// Combined anchor + candidate total weight.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.anchor_total_weight() + self.candidate_total_weight()
    }

    /// Convert to a flat table with match weights only.
    #[must_use]
    pub fn to_table(&self) -> UnitWeightTable {
        unit_weight_table_from_set(self, None)
    }

    /// Convert to a flat table with composed analysis weights.
    #[must_use]
    pub fn to_table_with_sampling(
        &self,
        sampling_weights: &RapidHashMap<String, f64>,
    ) -> UnitWeightTable {
        unit_weight_table_from_set(self, Some(sampling_weights))
    }
}

impl MatchOutcome {
    /// Extract unit-level match weights from this outcome using the supplied method.
    #[must_use]
    pub fn match_weights(&self, method: MatchWeightMethod) -> UnitWeightSet {
        match_weights_from_pairs(&self.pairs, method)
    }

    /// Extract unit-level match weights for an explicit estimand preset.
    #[must_use]
    pub fn match_weights_for_estimand(&self, estimand: Estimand) -> UnitWeightSet {
        self.match_weights(MatchWeightMethod::for_estimand(estimand))
    }

    /// Extract unit-level match weights using the realized estimand in diagnostics.
    #[must_use]
    pub fn match_weights_for_realized_estimand(&self) -> UnitWeightSet {
        self.match_weights_for_estimand(self.diagnostics.realized_estimand)
    }

    /// Extract composed analysis weights (`match_weight * sampling_weight`).
    #[must_use]
    pub fn analysis_weights(
        &self,
        method: MatchWeightMethod,
        sampling_weights: &RapidHashMap<String, f64>,
    ) -> UnitWeightSet {
        self.match_weights(method)
            .composed_with_sampling(sampling_weights)
    }

    /// Extract composed analysis weights using realized-estimand defaults.
    #[must_use]
    pub fn analysis_weights_for_realized_estimand(
        &self,
        sampling_weights: &RapidHashMap<String, f64>,
    ) -> UnitWeightSet {
        self.match_weights_for_realized_estimand()
            .composed_with_sampling(sampling_weights)
    }

    /// Extract unit-weight table rows using the supplied method.
    #[must_use]
    pub fn unit_weight_table(&self, method: MatchWeightMethod) -> UnitWeightTable {
        self.match_weights(method).to_table()
    }

    /// Extract unit-weight table rows with composed analysis weights.
    #[must_use]
    pub fn unit_weight_table_with_sampling(
        &self,
        method: MatchWeightMethod,
        sampling_weights: &RapidHashMap<String, f64>,
    ) -> UnitWeightTable {
        self.match_weights(method)
            .to_table_with_sampling(sampling_weights)
    }

    /// Extract unit-weight table rows using realized-estimand defaults.
    #[must_use]
    pub fn unit_weight_table_for_realized_estimand(&self) -> UnitWeightTable {
        self.match_weights_for_realized_estimand().to_table()
    }

    /// Extract realized-estimand unit-weight table with composed analysis weights.
    #[must_use]
    pub fn unit_weight_table_for_realized_estimand_with_sampling(
        &self,
        sampling_weights: &RapidHashMap<String, f64>,
    ) -> UnitWeightTable {
        self.match_weights_for_realized_estimand()
            .to_table_with_sampling(sampling_weights)
    }

    /// Extract pair-level match weights from this outcome using the supplied method.
    #[must_use]
    pub fn pair_weights(&self, method: MatchWeightMethod) -> PairWeightSet {
        pair_weights_from_pairs(&self.pairs, method)
    }

    /// Extract pair-level match weights for an explicit estimand preset.
    #[must_use]
    pub fn pair_weights_for_estimand(&self, estimand: Estimand) -> PairWeightSet {
        self.pair_weights(MatchWeightMethod::for_estimand(estimand))
    }

    /// Extract pair-level match weights using the realized estimand in diagnostics.
    #[must_use]
    pub fn pair_weights_for_realized_estimand(&self) -> PairWeightSet {
        self.pair_weights_for_estimand(self.diagnostics.realized_estimand)
    }

    /// Extract pair-weight table rows using the supplied method.
    #[must_use]
    pub fn pair_weight_table(&self, method: MatchWeightMethod) -> PairWeightTable {
        self.pair_weights(method).to_table()
    }

    /// Extract pair-weight table rows using realized-estimand defaults.
    #[must_use]
    pub fn pair_weight_table_for_realized_estimand(&self) -> PairWeightTable {
        self.pair_weights_for_realized_estimand().to_table()
    }
}

/// Extract unit-level match weights from matched pairs.
#[must_use]
pub fn match_weights_from_pairs(pairs: &[MatchedPair], method: MatchWeightMethod) -> UnitWeightSet {
    match method {
        MatchWeightMethod::PairCount => pair_count_weights(pairs),
        MatchWeightMethod::AnchorUnitCandidateFractional => {
            anchor_unit_candidate_fractional_weights(pairs)
        }
        MatchWeightMethod::CandidateUnitAnchorFractional => {
            candidate_unit_anchor_fractional_weights(pairs)
        }
    }
}

/// Extract pair-level match weights from matched pairs.
#[must_use]
pub fn pair_weights_from_pairs(pairs: &[MatchedPair], method: MatchWeightMethod) -> PairWeightSet {
    match method {
        MatchWeightMethod::PairCount => PairWeightSet {
            pairs: pairs
                .iter()
                .cloned()
                .map(|pair| PairWeight { pair, weight: 1.0 })
                .collect(),
        },
        MatchWeightMethod::AnchorUnitCandidateFractional => {
            let anchor_counts = pairs.iter().map(MatchedPair::anchor_id).counts();
            PairWeightSet {
                pairs: pairs
                    .iter()
                    .cloned()
                    .map(|pair| PairWeight {
                        weight: 1.0
                            / to_f64(anchor_counts.get(pair.anchor_id()).copied().unwrap_or(1)),
                        pair,
                    })
                    .collect(),
            }
        }
        MatchWeightMethod::CandidateUnitAnchorFractional => {
            let candidate_counts = pairs.iter().map(MatchedPair::comparator_id).counts();
            PairWeightSet {
                pairs: pairs
                    .iter()
                    .cloned()
                    .map(|pair| PairWeight {
                        weight: 1.0
                            / to_f64(
                                candidate_counts
                                    .get(pair.comparator_id())
                                    .copied()
                                    .unwrap_or(1),
                            ),
                        pair,
                    })
                    .collect(),
            }
        }
    }
}

/// A weight that is negative or not finite, with its position in the input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidWeightError {
    /// Position of the weight, counting from 0.
    pub position: usize,
    /// The weight.
    pub weight: f64,
}

impl std::fmt::Display for InvalidWeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the weight at position {} is {}, which is negative or not finite",
            self.position, self.weight
        )
    }
}

impl std::error::Error for InvalidWeightError {}

/// Kish's effective sample size, `(sum w)^2 / sum w^2`, of the weights.
///
/// The value does not depend on the scale of the weights: equal weights of any
/// size give their number. A zero weight is a unit that is not in the sample
/// and counts for nothing. The result is `None` when no weight is positive,
/// since there is no sample to have a size.
///
/// The sums are taken in the order given. When the weights are so large or
/// so small that their squares overflow or lose precision, they are divided
/// by the largest first.
///
/// # Errors
///
/// [`InvalidWeightError`] for the first weight that is negative or not
/// finite.
pub fn effective_sample_size<I>(weights: I) -> Result<Option<f64>, InvalidWeightError>
where
    I: IntoIterator<Item = f64>,
{
    let weights: Vec<f64> = weights.into_iter().collect();
    let mut largest = 0.0_f64;
    for (position, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(InvalidWeightError { position, weight });
        }
        largest = largest.max(weight);
    }
    if largest == 0.0 {
        return Ok(None);
    }
    let sums = |scale: f64| {
        weights
            .iter()
            .fold((0.0_f64, 0.0_f64), |(sum, sum_sq), &weight| {
                let weight = weight / scale;
                (sum + weight, sum_sq + weight * weight)
            })
    };
    let (mut sum, mut sum_sq) = sums(1.0);
    // The squares are accurate while the largest one is a normal number, and
    // the sums are finite while they have not overflowed.
    let plain = largest * largest >= f64::MIN_POSITIVE
        && sum.is_finite()
        && sum_sq.is_finite()
        && (sum * sum).is_finite();
    if !plain {
        (sum, sum_sq) = sums(largest);
    }
    Ok(Some((sum * sum) / sum_sq))
}

fn pair_count_weights(pairs: &[MatchedPair]) -> UnitWeightSet {
    let mut weights = UnitWeightSet::default();
    for pair in pairs {
        *weights
            .anchor
            .entry(pair.anchor_id().to_string())
            .or_insert(0.0) += 1.0;
        *weights
            .candidate
            .entry(pair.comparator_id().to_string())
            .or_insert(0.0) += 1.0;
    }
    weights
}

fn anchor_unit_candidate_fractional_weights(pairs: &[MatchedPair]) -> UnitWeightSet {
    let anchor_counts = pairs.iter().map(MatchedPair::anchor_id).counts();

    let anchor = anchor_counts
        .keys()
        .map(|id| ((*id).to_string(), 1.0_f64))
        .collect::<RapidHashMap<_, _>>();
    let mut candidate = RapidHashMap::default();

    for pair in pairs {
        let count = anchor_counts.get(pair.anchor_id()).copied().unwrap_or(1);
        *candidate
            .entry(pair.comparator_id().to_string())
            .or_insert(0.0) += 1.0 / to_f64(count);
    }

    UnitWeightSet { anchor, candidate }
}

fn candidate_unit_anchor_fractional_weights(pairs: &[MatchedPair]) -> UnitWeightSet {
    let candidate_counts = pairs.iter().map(MatchedPair::comparator_id).counts();

    let candidate = candidate_counts
        .keys()
        .map(|id| ((*id).to_string(), 1.0_f64))
        .collect::<RapidHashMap<_, _>>();
    let mut anchor = RapidHashMap::default();

    for pair in pairs {
        let count = candidate_counts
            .get(pair.comparator_id())
            .copied()
            .unwrap_or(1);
        *anchor.entry(pair.anchor_id().to_string()).or_insert(0.0) += 1.0 / to_f64(count);
    }

    UnitWeightSet { anchor, candidate }
}

/// The weights of a group in the order of their unit identifiers. Each
/// `RapidHashMap` is seeded at random, so a sum over its iteration order
/// would differ from run to run in the last bits.
fn in_id_order(group: &RapidHashMap<String, f64>) -> Vec<f64> {
    let mut entries: Vec<(&String, f64)> = group.iter().map(|(id, w)| (id, *w)).collect();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    entries.into_iter().map(|(_, weight)| weight).collect()
}

fn normalize_group(group: &RapidHashMap<String, f64>) -> RapidHashMap<String, f64> {
    let mut positive: Vec<(&String, f64)> = group
        .iter()
        .filter(|(_, weight)| **weight > 0.0)
        .map(|(id, weight)| (id, *weight))
        .collect();
    positive.sort_unstable_by(|left, right| left.0.cmp(right.0));
    let sum = positive
        .iter()
        .fold(0.0_f64, |acc, (_, weight)| acc + weight);
    if positive.is_empty() || !sum.is_finite() {
        return group.clone();
    }
    let mean = sum / to_f64(positive.len());
    group
        .iter()
        .map(|(id, weight)| {
            let scaled = if *weight > 0.0 {
                *weight / mean
            } else {
                *weight
            };
            (id.clone(), scaled)
        })
        .collect()
}

fn compose_with_sampling(
    match_weights: &RapidHashMap<String, f64>,
    sampling_weights: &RapidHashMap<String, f64>,
) -> RapidHashMap<String, f64> {
    match_weights
        .iter()
        .map(|(id, weight)| {
            let sampling = sampling_weights.get(id).copied().unwrap_or(1.0);
            (id.clone(), *weight * sampling)
        })
        .collect()
}

fn unit_weight_rows_for_group(
    group: &RapidHashMap<String, f64>,
    role: UnitRole,
    sampling_weights: Option<&RapidHashMap<String, f64>>,
) -> Vec<UnitWeightRow> {
    group
        .iter()
        .map(|(id, match_weight)| {
            let sampling_weight = sampling_weights.and_then(|weights| weights.get(id).copied());
            let analysis_weight = *match_weight * sampling_weight.unwrap_or(1.0);

            UnitWeightRow {
                unit_id: id.clone(),
                role,
                match_weight: *match_weight,
                sampling_weight,
                analysis_weight,
            }
        })
        .collect()
}

fn unit_weight_table_from_set(
    weights: &UnitWeightSet,
    sampling_weights: Option<&RapidHashMap<String, f64>>,
) -> UnitWeightTable {
    let mut rows = unit_weight_rows_for_group(&weights.anchor, UnitRole::Anchor, sampling_weights);
    rows.extend(unit_weight_rows_for_group(
        &weights.candidate,
        UnitRole::Candidate,
        sampling_weights,
    ));
    rows.sort_unstable_by(|left, right| {
        left.unit_id
            .cmp(&right.unit_id)
            .then(left.role.cmp(&right.role))
    });

    UnitWeightTable { rows }
}

use super::to_f64;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MatchDiagnostics;

    fn outcome_with_pairs(pairs: Vec<MatchedPair>, realized_estimand: Estimand) -> MatchOutcome {
        MatchOutcome {
            unmatched_cases: 0,
            used_controls: 0,
            matched_cases: 0,
            avg_controls_per_case: None,
            diagnostics: MatchDiagnostics {
                realized_estimand,
                ..MatchDiagnostics::default()
            },
            pairs,
        }
    }

    #[test]
    fn pair_count_weights_are_raw_pair_frequencies() {
        let outcome = outcome_with_pairs(
            vec![
                MatchedPair::new("a1", "c1"),
                MatchedPair::new("a1", "c2"),
                MatchedPair::new("a2", "c2"),
            ],
            Estimand::Att,
        );

        let weights = outcome.match_weights(MatchWeightMethod::PairCount);
        assert_eq!(weights.anchor.get("a1").copied(), Some(2.0));
        assert_eq!(weights.anchor.get("a2").copied(), Some(1.0));
        assert_eq!(weights.candidate.get("c1").copied(), Some(1.0));
        assert_eq!(weights.candidate.get("c2").copied(), Some(2.0));
    }

    #[test]
    fn att_style_weights_normalize_candidate_side_by_anchor_ratio() {
        let outcome = outcome_with_pairs(
            vec![
                MatchedPair::new("a1", "c1"),
                MatchedPair::new("a1", "c2"),
                MatchedPair::new("a2", "c2"),
            ],
            Estimand::Att,
        );

        let weights = outcome.match_weights(MatchWeightMethod::AnchorUnitCandidateFractional);
        assert_eq!(weights.anchor.get("a1").copied(), Some(1.0));
        assert_eq!(weights.anchor.get("a2").copied(), Some(1.0));
        assert_eq!(weights.candidate.get("c1").copied(), Some(0.5));
        assert_eq!(weights.candidate.get("c2").copied(), Some(1.5));

        let total_anchor: f64 = weights.anchor.values().sum();
        let total_candidate: f64 = weights.candidate.values().sum();
        assert!((total_anchor - total_candidate).abs() < 1e-12);
    }

    #[test]
    fn normalized_weights_have_mean_one_in_each_group_and_keep_zeros() {
        let mut weights = UnitWeightSet::default();
        weights.anchor.insert("a1".into(), 1.0);
        weights.anchor.insert("a2".into(), 1.0);
        weights.candidate.insert("c1".into(), 0.5);
        weights.candidate.insert("c2".into(), 1.5);
        weights.candidate.insert("c3".into(), 0.0);
        weights.candidate.insert("c4".into(), 1.0);

        let normalized = weights.normalized();
        assert_eq!(normalized.anchor.get("a1").copied(), Some(1.0));
        // The positive candidate weights have mean 1, so they are unchanged.
        assert_eq!(normalized.candidate.get("c1").copied(), Some(0.5));
        assert_eq!(normalized.candidate.get("c2").copied(), Some(1.5));
        assert_eq!(normalized.candidate.get("c3").copied(), Some(0.0));

        let mut scaled = UnitWeightSet::default();
        scaled.candidate.insert("c1".into(), 1.0 / 3.0);
        scaled.candidate.insert("c2".into(), 1.0 / 3.0);
        let scaled = scaled.normalized();
        assert!((scaled.candidate["c1"] - 1.0).abs() < 1e-15);
        assert!((scaled.candidate_total_weight() - 2.0).abs() < 1e-15);
        assert!(UnitWeightSet::default().normalized().anchor.is_empty());
    }

    #[test]
    fn realized_estimand_method_resolves_expected_scheme() {
        let outcome = outcome_with_pairs(vec![MatchedPair::new("a1", "c1")], Estimand::Atc);
        let weights = outcome.match_weights_for_realized_estimand();

        assert_eq!(weights.candidate.get("c1").copied(), Some(1.0));
        assert_eq!(weights.anchor.get("a1").copied(), Some(1.0));
    }

    #[test]
    fn sampling_composition_and_ess_work() {
        let weights = UnitWeightSet {
            anchor: [("a1".to_string(), 1.0_f64), ("a2".to_string(), 2.0)]
                .into_iter()
                .collect(),
            candidate: std::iter::once(("c1".to_string(), 0.5_f64)).collect(),
        };
        let sampling: RapidHashMap<String, f64> =
            [("a2".to_string(), 3.0), ("c1".to_string(), 4.0)]
                .into_iter()
                .collect();

        let composed = weights.composed_with_sampling(&sampling);
        assert_eq!(composed.anchor.get("a1").copied(), Some(1.0));
        assert_eq!(composed.anchor.get("a2").copied(), Some(6.0));
        assert_eq!(composed.candidate.get("c1").copied(), Some(2.0));

        let ess = effective_sample_size([1.0, 2.0, 3.0]).unwrap().unwrap();
        assert!((ess - (36.0 / 14.0)).abs() < 1e-12);
    }

    #[test]
    fn pair_weights_support_pair_level_extraction_and_realized_estimand_resolution() {
        let outcome = outcome_with_pairs(
            vec![
                MatchedPair::new("a1", "c1"),
                MatchedPair::new("a1", "c2"),
                MatchedPair::new("a2", "c2"),
            ],
            Estimand::Att,
        );

        let pair_count = outcome.pair_weights(MatchWeightMethod::PairCount);
        assert_eq!(pair_count.pairs.len(), 3);
        assert!((pair_count.total_weight() - 3.0).abs() < 1e-12);

        let realized = outcome.pair_weights_for_realized_estimand();
        assert_eq!(realized.pairs.len(), 3);
        assert!((realized.total_weight() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn unit_weight_total_helpers_return_expected_sums() {
        let weights = UnitWeightSet {
            anchor: [("a1".to_string(), 1.0_f64), ("a2".to_string(), 2.0)]
                .into_iter()
                .collect(),
            candidate: [("c1".to_string(), 0.5_f64), ("c2".to_string(), 1.5)]
                .into_iter()
                .collect(),
        };

        assert!((weights.anchor_total_weight() - 3.0).abs() < 1e-12);
        assert!((weights.candidate_total_weight() - 2.0).abs() < 1e-12);
        assert!((weights.total_weight() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn explicit_estimand_presets_cover_ate_and_atm() {
        let outcome = outcome_with_pairs(
            vec![
                MatchedPair::new("a1", "c1"),
                MatchedPair::new("a1", "c2"),
                MatchedPair::new("a2", "c3"),
            ],
            Estimand::Att,
        );

        let ate = outcome.match_weights_for_estimand(Estimand::Ate);
        let atm = outcome.match_weights_for_estimand(Estimand::Atm);

        assert_eq!(ate, atm);
        assert_eq!(ate.anchor.get("a1").copied(), Some(2.0));
        assert_eq!(ate.candidate.get("c2").copied(), Some(1.0));
    }

    #[test]
    fn dataframe_tables_include_sampling_composition() {
        let outcome = outcome_with_pairs(
            vec![
                MatchedPair::new("a1", "c1"),
                MatchedPair::new("a1", "c2"),
                MatchedPair::new("a2", "c2"),
            ],
            Estimand::Att,
        );
        let sampling: RapidHashMap<String, f64> = [
            ("a1".to_string(), 2.0),
            ("a2".to_string(), 1.0),
            ("c1".to_string(), 4.0),
            ("c2".to_string(), 0.5),
        ]
        .into_iter()
        .collect();

        let table = outcome.unit_weight_table_with_sampling(
            MatchWeightMethod::AnchorUnitCandidateFractional,
            &sampling,
        );

        assert_eq!(table.rows.len(), 4);
        assert!(table.rows.iter().any(|row| row.unit_id == "a1"
            && row.role == UnitRole::Anchor
            && (row.analysis_weight - 2.0).abs() < 1e-12));
        assert!(table.rows.iter().any(|row| row.unit_id == "c2"
            && row.role == UnitRole::Candidate
            && (row.analysis_weight - 0.75).abs() < 1e-12));
    }

    #[test]
    fn pair_weight_table_flattens_pairs() {
        let outcome = outcome_with_pairs(vec![MatchedPair::new("a1", "c1")], Estimand::Ate);

        let table = outcome.pair_weight_table_for_realized_estimand();
        assert_eq!(table.rows.len(), 1);
        assert_eq!(table.rows[0].anchor_id, "a1");
        assert_eq!(table.rows[0].candidate_id, "c1");
        assert!((table.rows[0].pair_weight - 1.0).abs() < 1e-12);
    }
}
