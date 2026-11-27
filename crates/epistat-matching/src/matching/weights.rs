//! Match weights from matched pairs, and the effective sample size.
//!
//! # Weights
//!
//! For the ATT every matched case weighs 1 and a control weighs the sum,
//! over the pairs it is in, of `1 / m` with `m` the number of controls of the
//! pair's case. For the ATC the roles are reversed. A control used by two
//! cases that each have two controls weighs `1/2 + 1/2 = 1`. For the ATE
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

use crate::Error;
use crate::types::{Estimand, MatchOutcome, MatchedPair};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Match-weight extraction method for matched pairs.
///
/// The weights are raw: no group is rescaled. `matchit()` rescales by default
/// (`normalize = TRUE`); [`UnitWeightSet::normalized`] does the same here.
/// Raw weights equal `matchit(normalize = FALSE)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchWeightMethod {
    /// Every pair contributes +1 to both its case and its control.
    ///
    /// `matchit()` has no such weighting. Its weights for `estimand = "ATE"`
    /// are stratum weights, which equal these only when every case has the
    /// same number of controls and both are normalized.
    PairCount,
    /// Every matched case has weight 1, and a control has `1 / m` from each
    /// case with `m` controls it was drawn for.
    ///
    /// This is `matchit()`'s weighting for the estimand whose focal group is
    /// the cases, with or without replacement.
    CaseUnitControlFractional,
    /// Every matched control has weight 1, and a case has `1 / m` from each
    /// control drawn `m` times.
    ///
    /// This is `matchit()`'s weighting for the estimand whose focal group is
    /// the controls, applied to pairs that were formed from the cases'
    /// side. `matchit()` itself would form the pairs from the controls'
    /// side.
    ControlUnitCaseFractional,
}

impl MatchWeightMethod {
    /// [`Self::CaseUnitControlFractional`] under its name in 0.2.
    #[deprecated(
        since = "0.3.0",
        note = "use MatchWeightMethod::CaseUnitControlFractional"
    )]
    #[expect(
        non_upper_case_globals,
        reason = "the variant name of 0.2, kept for one release"
    )]
    pub const AnchorUnitCandidateFractional: Self = Self::CaseUnitControlFractional;
    /// [`Self::ControlUnitCaseFractional`] under its name in 0.2.
    #[deprecated(
        since = "0.3.0",
        note = "use MatchWeightMethod::ControlUnitCaseFractional"
    )]
    #[expect(
        non_upper_case_globals,
        reason = "the variant name of 0.2, kept for one release"
    )]
    pub const CandidateUnitAnchorFractional: Self = Self::ControlUnitCaseFractional;

    /// Resolve a default weighting method from an estimand.
    #[must_use]
    pub const fn for_estimand(estimand: Estimand) -> Self {
        match estimand {
            Estimand::Att => Self::CaseUnitControlFractional,
            Estimand::Atc => Self::ControlUnitCaseFractional,
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
                    case_id: entry.pair.case_id.as_str().to_string(),
                    control_id: entry.pair.control_id.as_str().to_string(),
                    pair_weight: entry.weight,
                })
                .collect(),
        }
    }
}

/// Flat row for pair-level table outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairWeightRow {
    /// Case-side unit identifier.
    pub case_id: String,
    /// Control-side unit identifier.
    pub control_id: String,
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
#[non_exhaustive]
pub enum UnitRole {
    /// A case.
    Case,
    /// A control.
    Control,
}

impl UnitRole {
    /// The case role under its name in 0.2.
    #[deprecated(since = "0.3.0", note = "use UnitRole::Case")]
    #[expect(
        non_upper_case_globals,
        reason = "the variant name of 0.2, kept for one release"
    )]
    pub const Anchor: Self = Self::Case;
    /// The control role under its name in 0.2.
    #[deprecated(since = "0.3.0", note = "use UnitRole::Control")]
    #[expect(
        non_upper_case_globals,
        reason = "the variant name of 0.2, kept for one release"
    )]
    pub const Candidate: Self = Self::Control;
}

/// Flat row for unit-weight table outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnitWeightRow {
    /// Unit identifier.
    pub unit_id: String,
    /// Case/control role.
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

/// The weight of each unit, the cases and the controls apart, each kept in
/// the order of the unit ids.
///
/// Every sum, mean and table formed from the set runs over the ids in sorted
/// order, so the results do not depend on how the set was built.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UnitWeightSet {
    pub(crate) case: BTreeMap<String, f64>,
    pub(crate) control: BTreeMap<String, f64>,
}

impl UnitWeightSet {
    /// The set of the weights of `cases` and of `controls`, each by unit id.
    /// A later weight of an id replaces an earlier one.
    #[must_use]
    pub fn new(
        cases: impl IntoIterator<Item = (String, f64)>,
        controls: impl IntoIterator<Item = (String, f64)>,
    ) -> Self {
        Self {
            case: cases.into_iter().collect(),
            control: controls.into_iter().collect(),
        }
    }

    /// The cases and their weights, in the order of their ids.
    pub fn cases(&self) -> impl Iterator<Item = (&str, f64)> {
        self.case.iter().map(|(id, weight)| (id.as_str(), *weight))
    }

    /// The controls and their weights, in the order of their ids.
    pub fn controls(&self) -> impl Iterator<Item = (&str, f64)> {
        self.control
            .iter()
            .map(|(id, weight)| (id.as_str(), *weight))
    }

    /// The weight of the case `id`, `None` for a unit that is not a case.
    #[must_use]
    pub fn case_weight(&self, id: &str) -> Option<f64> {
        self.case.get(id).copied()
    }

    /// The weight of the control `id`, `None` for a unit that is not a
    /// control.
    #[must_use]
    pub fn control_weight(&self, id: &str) -> Option<f64> {
        self.control.get(id).copied()
    }

    /// Compose match weights with external sampling weights (`analysis = match * sampling`).
    ///
    /// Missing sampling weights default to `1.0`.
    #[must_use]
    pub fn composed_with_sampling(&self, sampling_weights: &HashMap<String, f64>) -> Self {
        Self {
            case: compose_with_sampling(&self.case, sampling_weights),
            control: compose_with_sampling(&self.control, sampling_weights),
        }
    }

    /// Rescale the weights of each group so that their mean is 1.
    ///
    /// Within the case group and within the control group separately,
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
            case: normalize_group(&self.case),
            control: normalize_group(&self.control),
        }
    }

    /// Effective sample size for case weights, summed in the order of the
    /// unit identifiers. See [`effective_sample_size`].
    ///
    /// # Errors
    ///
    /// [`Error`] for a weight that is negative or not finite.
    pub fn case_effective_sample_size(&self) -> Result<Option<f64>, Error> {
        kish(in_id_order(&self.case))
    }

    /// Effective sample size for control weights, summed in the order of the
    /// unit identifiers. See [`effective_sample_size`].
    ///
    /// # Errors
    ///
    /// [`Error`] for a weight that is negative or not finite.
    pub fn control_effective_sample_size(&self) -> Result<Option<f64>, Error> {
        kish(in_id_order(&self.control))
    }

    /// Sum of all case-side unit weights, in the order of the unit
    /// identifiers.
    #[must_use]
    pub fn case_total_weight(&self) -> f64 {
        in_id_order(&self.case).into_iter().sum()
    }

    /// Sum of all control-side unit weights, in the order of the unit
    /// identifiers.
    #[must_use]
    pub fn control_total_weight(&self) -> f64 {
        in_id_order(&self.control).into_iter().sum()
    }

    /// Combined case + control total weight.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.case_total_weight() + self.control_total_weight()
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
        sampling_weights: &HashMap<String, f64>,
    ) -> UnitWeightTable {
        unit_weight_table_from_set(self, Some(sampling_weights))
    }
}

impl MatchOutcome {
    /// The unit weights of the pairs by `method`.
    #[must_use]
    pub fn match_weights(&self, method: MatchWeightMethod) -> UnitWeightSet {
        match_weights_from_pairs(&self.pairs, method)
    }

    /// The pair weights of the pairs by `method`.
    #[must_use]
    pub fn pair_weights(&self, method: MatchWeightMethod) -> PairWeightSet {
        pair_weights_from_pairs(&self.pairs, method)
    }
}

/// Extract unit-level match weights from matched pairs.
#[must_use]
pub fn match_weights_from_pairs(pairs: &[MatchedPair], method: MatchWeightMethod) -> UnitWeightSet {
    match method {
        MatchWeightMethod::PairCount => pair_count_weights(pairs),
        MatchWeightMethod::CaseUnitControlFractional => {
            anchor_unit_candidate_fractional_weights(pairs)
        }
        MatchWeightMethod::ControlUnitCaseFractional => {
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
        MatchWeightMethod::CaseUnitControlFractional => {
            let anchor_counts = pairs
                .iter()
                .map(|pair: &MatchedPair| pair.case_id.as_str())
                .counts();
            PairWeightSet {
                pairs: pairs
                    .iter()
                    .cloned()
                    .map(|pair| PairWeight {
                        weight: 1.0
                            / to_f64(
                                anchor_counts
                                    .get(pair.case_id.as_str())
                                    .copied()
                                    .unwrap_or(1),
                            ),
                        pair,
                    })
                    .collect(),
            }
        }
        MatchWeightMethod::ControlUnitCaseFractional => {
            let candidate_counts = pairs
                .iter()
                .map(|pair: &MatchedPair| pair.control_id.as_str())
                .counts();
            PairWeightSet {
                pairs: pairs
                    .iter()
                    .cloned()
                    .map(|pair| PairWeight {
                        weight: 1.0
                            / to_f64(
                                candidate_counts
                                    .get(pair.control_id.as_str())
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

/// Kish's effective sample size, `(sum w)^2 / sum w^2`, of the weights.
///
/// This is [`epistat_stats::weighted::effective_sample_size`], which takes a
/// slice; the weights are collected first.
///
/// # Errors
///
/// [`Error::Input`] for the first weight that is negative or not finite.
#[deprecated(
    since = "0.3.0",
    note = "use epistat_stats::weighted::effective_sample_size on a slice"
)]
pub fn effective_sample_size<I>(weights: I) -> Result<Option<f64>, Error>
where
    I: IntoIterator<Item = f64>,
{
    kish(weights)
}

/// Kish's effective sample size of the weights, in the order given.
pub(crate) fn kish(weights: impl IntoIterator<Item = f64>) -> Result<Option<f64>, Error> {
    let weights: Vec<f64> = weights.into_iter().collect();
    epistat_stats::weighted::effective_sample_size(&weights).map_err(Error::from)
}

fn pair_count_weights(pairs: &[MatchedPair]) -> UnitWeightSet {
    let mut weights = UnitWeightSet::default();
    for pair in pairs {
        *weights
            .case
            .entry(pair.case_id.as_str().to_string())
            .or_insert(0.0) += 1.0;
        *weights
            .control
            .entry(pair.control_id.as_str().to_string())
            .or_insert(0.0) += 1.0;
    }
    weights
}

fn anchor_unit_candidate_fractional_weights(pairs: &[MatchedPair]) -> UnitWeightSet {
    let anchor_counts = pairs
        .iter()
        .map(|pair: &MatchedPair| pair.case_id.as_str())
        .counts();

    let anchor = anchor_counts
        .keys()
        .map(|id| ((*id).to_string(), 1.0_f64))
        .collect::<BTreeMap<_, _>>();
    let mut candidate = BTreeMap::new();

    for pair in pairs {
        let count = anchor_counts
            .get(pair.case_id.as_str())
            .copied()
            .unwrap_or(1);
        *candidate
            .entry(pair.control_id.as_str().to_string())
            .or_insert(0.0) += 1.0 / to_f64(count);
    }

    UnitWeightSet {
        case: anchor,
        control: candidate,
    }
}

fn candidate_unit_anchor_fractional_weights(pairs: &[MatchedPair]) -> UnitWeightSet {
    let candidate_counts = pairs
        .iter()
        .map(|pair: &MatchedPair| pair.control_id.as_str())
        .counts();

    let candidate = candidate_counts
        .keys()
        .map(|id| ((*id).to_string(), 1.0_f64))
        .collect::<BTreeMap<_, _>>();
    let mut anchor = BTreeMap::new();

    for pair in pairs {
        let count = candidate_counts
            .get(pair.control_id.as_str())
            .copied()
            .unwrap_or(1);
        *anchor
            .entry(pair.case_id.as_str().to_string())
            .or_insert(0.0) += 1.0 / to_f64(count);
    }

    UnitWeightSet {
        case: anchor,
        control: candidate,
    }
}

/// The weights of a group in the order of their unit identifiers. Each
/// `RapidHashMap` is seeded at random, so a sum over its iteration order
/// would differ from run to run in the last bits.
fn in_id_order(group: &BTreeMap<String, f64>) -> Vec<f64> {
    group.values().copied().collect()
}

fn normalize_group(group: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    let positive: Vec<(&String, f64)> = group
        .iter()
        .filter(|(_, weight)| **weight > 0.0)
        .map(|(id, weight)| (id, *weight))
        .collect();
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
    match_weights: &BTreeMap<String, f64>,
    sampling_weights: &HashMap<String, f64>,
) -> BTreeMap<String, f64> {
    match_weights
        .iter()
        .map(|(id, weight)| {
            let sampling = sampling_weights.get(id).copied().unwrap_or(1.0);
            (id.clone(), *weight * sampling)
        })
        .collect()
}

fn unit_weight_rows_for_group(
    group: &BTreeMap<String, f64>,
    role: UnitRole,
    sampling_weights: Option<&HashMap<String, f64>>,
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
    sampling_weights: Option<&HashMap<String, f64>>,
) -> UnitWeightTable {
    let mut rows = unit_weight_rows_for_group(&weights.case, UnitRole::Case, sampling_weights);
    rows.extend(unit_weight_rows_for_group(
        &weights.control,
        UnitRole::Control,
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
        assert_eq!(weights.case.get("a1").copied(), Some(2.0));
        assert_eq!(weights.case.get("a2").copied(), Some(1.0));
        assert_eq!(weights.control.get("c1").copied(), Some(1.0));
        assert_eq!(weights.control.get("c2").copied(), Some(2.0));
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

        let weights = outcome.match_weights(MatchWeightMethod::CaseUnitControlFractional);
        assert_eq!(weights.case.get("a1").copied(), Some(1.0));
        assert_eq!(weights.case.get("a2").copied(), Some(1.0));
        assert_eq!(weights.control.get("c1").copied(), Some(0.5));
        assert_eq!(weights.control.get("c2").copied(), Some(1.5));

        let total_anchor: f64 = weights.case.values().sum();
        let total_candidate: f64 = weights.control.values().sum();
        assert!((total_anchor - total_candidate).abs() < 1e-12);
    }

    #[test]
    fn normalized_weights_have_mean_one_in_each_group_and_keep_zeros() {
        let mut weights = UnitWeightSet::default();
        weights.case.insert("a1".into(), 1.0);
        weights.case.insert("a2".into(), 1.0);
        weights.control.insert("c1".into(), 0.5);
        weights.control.insert("c2".into(), 1.5);
        weights.control.insert("c3".into(), 0.0);
        weights.control.insert("c4".into(), 1.0);

        let normalized = weights.normalized();
        assert_eq!(normalized.case.get("a1").copied(), Some(1.0));
        // The positive candidate weights have mean 1, so they are unchanged.
        assert_eq!(normalized.control.get("c1").copied(), Some(0.5));
        assert_eq!(normalized.control.get("c2").copied(), Some(1.5));
        assert_eq!(normalized.control.get("c3").copied(), Some(0.0));

        let mut scaled = UnitWeightSet::default();
        scaled.control.insert("c1".into(), 1.0 / 3.0);
        scaled.control.insert("c2".into(), 1.0 / 3.0);
        let scaled = scaled.normalized();
        assert!((scaled.control["c1"] - 1.0).abs() < 1e-15);
        assert!((scaled.control_total_weight() - 2.0).abs() < 1e-15);
        assert!(UnitWeightSet::default().normalized().case.is_empty());
    }

    #[test]
    fn realized_estimand_method_resolves_expected_scheme() {
        let outcome = outcome_with_pairs(vec![MatchedPair::new("a1", "c1")], Estimand::Atc);
        let weights = outcome.match_weights(MatchWeightMethod::for_estimand(
            outcome.diagnostics.realized_estimand,
        ));

        assert_eq!(weights.control.get("c1").copied(), Some(1.0));
        assert_eq!(weights.case.get("a1").copied(), Some(1.0));
    }

    #[test]
    fn sampling_composition_and_ess_work() {
        let weights = UnitWeightSet {
            case: [("a1".to_string(), 1.0_f64), ("a2".to_string(), 2.0)]
                .into_iter()
                .collect(),
            control: std::iter::once(("c1".to_string(), 0.5_f64)).collect(),
        };
        let sampling: std::collections::HashMap<String, f64> =
            [("a2".to_string(), 3.0), ("c1".to_string(), 4.0)]
                .into_iter()
                .collect();

        let composed = weights.composed_with_sampling(&sampling);
        assert_eq!(composed.case.get("a1").copied(), Some(1.0));
        assert_eq!(composed.case.get("a2").copied(), Some(6.0));
        assert_eq!(composed.control.get("c1").copied(), Some(2.0));

        let ess = kish([1.0, 2.0, 3.0]).unwrap().unwrap();
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

        let realized = outcome.pair_weights(MatchWeightMethod::for_estimand(
            outcome.diagnostics.realized_estimand,
        ));
        assert_eq!(realized.pairs.len(), 3);
        assert!((realized.total_weight() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn unit_weight_total_helpers_return_expected_sums() {
        let weights = UnitWeightSet {
            case: [("a1".to_string(), 1.0_f64), ("a2".to_string(), 2.0)]
                .into_iter()
                .collect(),
            control: [("c1".to_string(), 0.5_f64), ("c2".to_string(), 1.5)]
                .into_iter()
                .collect(),
        };

        assert!((weights.case_total_weight() - 3.0).abs() < 1e-12);
        assert!((weights.control_total_weight() - 2.0).abs() < 1e-12);
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

        let ate = outcome.match_weights(MatchWeightMethod::for_estimand(Estimand::Ate));
        let atm = outcome.match_weights(MatchWeightMethod::for_estimand(Estimand::Atm));

        assert_eq!(ate, atm);
        assert_eq!(ate.case.get("a1").copied(), Some(2.0));
        assert_eq!(ate.control.get("c2").copied(), Some(1.0));
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
        let sampling: std::collections::HashMap<String, f64> = [
            ("a1".to_string(), 2.0),
            ("a2".to_string(), 1.0),
            ("c1".to_string(), 4.0),
            ("c2".to_string(), 0.5),
        ]
        .into_iter()
        .collect();

        let table = outcome
            .match_weights(MatchWeightMethod::CaseUnitControlFractional)
            .to_table_with_sampling(&sampling);

        assert_eq!(table.rows.len(), 4);
        assert!(table.rows.iter().any(|row| row.unit_id == "a1"
            && row.role == UnitRole::Case
            && (row.analysis_weight - 2.0).abs() < 1e-12));
        assert!(table.rows.iter().any(|row| row.unit_id == "c2"
            && row.role == UnitRole::Control
            && (row.analysis_weight - 0.75).abs() < 1e-12));
    }

    #[test]
    fn pair_weight_table_flattens_pairs() {
        let outcome = outcome_with_pairs(vec![MatchedPair::new("a1", "c1")], Estimand::Ate);

        let table = outcome
            .pair_weights(MatchWeightMethod::for_estimand(
                outcome.diagnostics.realized_estimand,
            ))
            .to_table();
        assert_eq!(table.rows.len(), 1);
        assert_eq!(table.rows[0].case_id, "a1");
        assert_eq!(table.rows[0].control_id, "c1");
        assert!((table.rows[0].pair_weight - 1.0).abs() < 1e-12);
    }
}
