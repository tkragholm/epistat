use super::records::MatchingRecord;
use crate::types::Estimand;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::BuildHasher;

/// Reference population used to derive subclass cutpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubclassReferenceGroup {
    /// Derive quantiles from anchor scores.
    Anchors,
    /// Derive quantiles from candidate scores.
    Candidates,
    /// Derive quantiles from pooled anchor + candidate scores.
    Pooled,
}

/// Configuration for propensity-score subclassification.
///
/// The defaults are MatchIt's: six subclasses, cutpoints at equally spaced
/// quantiles of the reference group's scores, and at least one unit of each
/// group in every subclass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubclassificationConfig {
    /// Requested number of subclasses (must be at least 1).
    pub subclasses: usize,
    /// Optional reference group used for cutpoints.
    ///
    /// Defaults by estimand:
    /// - ATT -> `Anchors`
    /// - ATC -> `Candidates`
    /// - ATE/ATM -> `Pooled`
    pub reference_group: Option<SubclassReferenceGroup>,
    /// Smallest number of units of each group a subclass may hold.
    ///
    /// When the quantile cutpoints leave a subclass with fewer, units from the
    /// nearest other subclasses are reassigned to it: of the units of the
    /// deficient group, the one whose score is closest to the deficient
    /// subclass. With 0 nothing is reassigned, and subclasses without both
    /// groups are dropped.
    ///
    /// When some group has fewer than `min_n` units per subclass in total, the
    /// request cannot be met and every unit is dropped.
    pub min_n: usize,
}

impl Default for SubclassificationConfig {
    fn default() -> Self {
        Self {
            subclasses: 6,
            reference_group: None,
            min_n: 1,
        }
    }
}

/// Per-subclass summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubclassSummary {
    /// Subclass index.
    pub subclass: usize,
    /// Number of anchor units in subclass.
    pub anchor_count: usize,
    /// Number of candidate units in subclass.
    pub candidate_count: usize,
    /// Anchor share in subclass (`anchor_count / total`) when total > 0.
    pub subclass_propensity: Option<f64>,
}

/// Result of propensity-score subclassification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubclassificationOutcome {
    /// Requested estimand for weighting.
    pub estimand: Estimand,
    /// Requested subclass count.
    pub requested_subclasses: usize,
    /// Realized subclass count after non-overlap filtering.
    pub realized_subclasses: usize,
    /// The requested count plus one quantiles of the reference scores, from the
    /// smallest to the largest. A unit belongs to the subclass whose interval
    /// starts at the largest quantile that is at most its score, and the
    /// largest interval also takes the units above the last quantile. With
    /// `min_n` above 0 the assignments can depart from these intervals.
    pub quantile_cuts: Vec<f64>,
    /// Subclass assignment per retained unit id. Subclasses are numbered from
    /// 0 in order of score, skipping any that no unit falls in.
    pub assignments: HashMap<String, usize>,
    /// Subclassification weights per retained unit id.
    pub weights: HashMap<String, f64>,
    /// Unit ids excluded due to missing/non-finite score or non-overlap subclass.
    pub dropped_ids: Vec<String>,
    /// Subclass summaries.
    pub subclasses: Vec<SubclassSummary>,
}

impl SubclassificationOutcome {
    /// Number of retained units.
    #[must_use]
    pub fn retained_units(&self) -> usize {
        self.assignments.len()
    }
}

/// One unit with a usable score.
struct ScoredUnit {
    id: String,
    is_anchor: bool,
    score: f64,
    subclass: usize,
}

/// Build propensity-score subclasses and estimand-specific subclass weights.
///
/// The cutpoints are the quantiles of the reference group's scores that R's
/// `quantile()` gives by default (linear interpolation between order
/// statistics), and the subclasses are the intervals between them, closed on
/// the left. This is MatchIt's `method = "subclass"`. The weights are the
/// subclass weights of MatchIt before it rescales each group to sum to its
/// number of retained units.
#[must_use]
pub fn subclassify_by_propensity_score_map<R: MatchingRecord, H: BuildHasher>(
    anchors: &[R],
    candidates: &[R],
    propensity_scores: &HashMap<String, f64, H>,
    estimand: Estimand,
    config: &SubclassificationConfig,
) -> SubclassificationOutcome {
    let requested_subclasses = config.subclasses.max(1);
    let reference_group = resolve_reference_group(config.reference_group, estimand);
    let reference_scores =
        sorted_reference_scores(anchors, candidates, propensity_scores, reference_group);
    if reference_scores.is_empty() {
        return dropped_only_outcome(anchors, candidates, estimand, requested_subclasses);
    }

    let quantile_cuts = quantile_cuts(&reference_scores, requested_subclasses);
    let interior = &quantile_cuts[1..quantile_cuts.len() - 1];

    let mut dropped_ids = Vec::new();
    let mut units = Vec::with_capacity(anchors.len() + candidates.len());
    for (rows, is_anchor) in [(anchors, true), (candidates, false)] {
        for row in rows {
            match propensity_scores
                .get(row.id())
                .copied()
                .filter(|score| score.is_finite())
            {
                Some(score) => units.push(ScoredUnit {
                    id: row.id().to_string(),
                    is_anchor,
                    score,
                    subclass: interior.partition_point(|cut| *cut <= score),
                }),
                None => dropped_ids.push(row.id().to_string()),
            }
        }
    }
    // Intervals that no unit falls in, which tied quantiles produce, are not
    // subclasses.
    let subclass_count = renumber_present_subclasses(&mut units);

    if config.min_n > 0 && !fill_deficient_subclasses(&mut units, subclass_count, config.min_n) {
        let mut outcome = dropped_only_outcome(anchors, candidates, estimand, requested_subclasses);
        outcome.quantile_cuts = quantile_cuts;
        return outcome;
    }

    let mut anchor_counts = vec![0usize; subclass_count];
    let mut candidate_counts = vec![0usize; subclass_count];
    for unit in &units {
        if unit.is_anchor {
            anchor_counts[unit.subclass] += 1;
        } else {
            candidate_counts[unit.subclass] += 1;
        }
    }
    let valid_subclasses = valid_subclasses(&anchor_counts, &candidate_counts);
    let subclasses = build_subclass_summaries(&anchor_counts, &candidate_counts);
    let realized_subclasses = valid_subclasses
        .iter()
        .filter(|is_valid| **is_valid)
        .count();

    let mut assignments = HashMap::new();
    let mut weights = HashMap::new();
    for unit in units {
        let propensity = subclasses[unit.subclass].subclass_propensity;
        match propensity.filter(|_| valid_subclasses[unit.subclass]) {
            Some(propensity) => {
                weights.insert(
                    unit.id.clone(),
                    subclass_weight_for_estimand(estimand, propensity, unit.is_anchor),
                );
                assignments.insert(unit.id, unit.subclass);
            }
            None => dropped_ids.push(unit.id),
        }
    }

    SubclassificationOutcome {
        estimand,
        requested_subclasses,
        realized_subclasses,
        quantile_cuts,
        assignments,
        weights,
        dropped_ids,
        subclasses,
    }
}

fn resolve_reference_group(
    reference_group: Option<SubclassReferenceGroup>,
    estimand: Estimand,
) -> SubclassReferenceGroup {
    reference_group.unwrap_or(match estimand {
        Estimand::Att => SubclassReferenceGroup::Anchors,
        Estimand::Atc => SubclassReferenceGroup::Candidates,
        Estimand::Ate | Estimand::Atm => SubclassReferenceGroup::Pooled,
    })
}

fn sorted_reference_scores<R: MatchingRecord, H: BuildHasher>(
    anchors: &[R],
    candidates: &[R],
    propensity_scores: &HashMap<String, f64, H>,
    reference_group: SubclassReferenceGroup,
) -> Vec<f64> {
    let mut scores = match reference_group {
        SubclassReferenceGroup::Anchors => collect_finite_scores(anchors, propensity_scores),
        SubclassReferenceGroup::Candidates => collect_finite_scores(candidates, propensity_scores),
        SubclassReferenceGroup::Pooled => {
            let mut pooled = collect_finite_scores(anchors, propensity_scores);
            pooled.extend(collect_finite_scores(candidates, propensity_scores));
            pooled
        }
    };
    scores.sort_unstable_by(f64::total_cmp);
    scores
}

fn collect_finite_scores<R: MatchingRecord, H: BuildHasher>(
    rows: &[R],
    propensity_scores: &HashMap<String, f64, H>,
) -> Vec<f64> {
    rows.iter()
        .filter_map(|row| propensity_scores.get(row.id()).copied())
        .filter(|score| score.is_finite())
        .collect()
}

/// The `subclasses + 1` quantiles at probabilities `0, 1/k, ..., 1` of the
/// sorted scores, by linear interpolation between order statistics: the
/// quantile at `p` lies at position `1 + (n - 1) p` of the sorted values (R's
/// type 7, evaluated in the same arithmetic). The probabilities are multiples
/// of `1/k` as `seq(0, 1, length.out = k + 1)` builds them, so a quantile that
/// falls on an order statistic is that statistic exactly. `sorted_scores` must
/// not be empty.
fn quantile_cuts(sorted_scores: &[f64], subclasses: usize) -> Vec<f64> {
    let count = sorted_scores.len();
    let step = 1.0 / to_f64(subclasses);
    (0..=subclasses)
        .map(|bucket| {
            let probability = if bucket == subclasses {
                1.0
            } else {
                to_f64(bucket) * step
            };
            let position = 1.0 + to_f64(count - 1) * probability;
            let lower_rank = floor_rank(position, count);
            let lower = sorted_scores[lower_rank - 1];
            let fraction = position - to_f64(lower_rank);
            let upper = sorted_scores[lower_rank.min(count - 1)];
            if fraction > 0.0 && upper.total_cmp(&lower).is_ne() {
                (1.0 - fraction) * lower + fraction * upper
            } else {
                lower
            }
        })
        .collect()
}

/// The largest rank in `1..=count` that is at most `position`, which is at
/// least 1.
fn floor_rank(position: f64, count: usize) -> usize {
    let (mut low, mut high) = (1, count);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if to_f64(middle) <= position {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

/// Renumber the subclasses that some unit occupies as `0..count`, keeping
/// their order, and return `count`.
fn renumber_present_subclasses(units: &mut [ScoredUnit]) -> usize {
    let mut present: Vec<usize> = units.iter().map(|unit| unit.subclass).collect();
    present.sort_unstable();
    present.dedup();
    for unit in units.iter_mut() {
        unit.subclass = present.partition_point(|value| *value < unit.subclass);
    }
    present.len()
}

/// Reassign units so that every subclass holds at least `min_n` units of each
/// group, and return `false` when the groups are too small for that.
///
/// Each group is treated on its own, in the order of its units. For
/// `level = 0..min_n`, while some subclass holds no unit of the group beyond
/// the `level` it already supplied, the lowest such subclass takes one unit
/// from the nearest subclass on one side that has a unit to give. It looks
/// towards the lowest or highest subclass when it is the highest or lowest, and
/// otherwise towards the side with the larger sum, over the subclasses that
/// hold more than one unit, of the surplus divided by the distance in subclass
/// numbers, going left when the sums are equal. From that subclass it takes the
/// unit whose score is closest to the deficient one, the latest unit among
/// equal scores.
fn fill_deficient_subclasses(
    units: &mut [ScoredUnit],
    subclass_count: usize,
    min_n: usize,
) -> bool {
    for is_anchor in [true, false] {
        let members: Vec<usize> = (0..units.len())
            .filter(|idx| units[*idx].is_anchor == is_anchor)
            .collect();
        if members.len() < subclass_count * min_n {
            return false;
        }
        let mut counts = vec![0usize; subclass_count];
        for idx in &members {
            counts[units[*idx].subclass] += 1;
        }
        for _ in 0..min_n {
            while let Some(deficient) = counts.iter().position(|count| *count == 0) {
                let from_left = look_left(&counts, deficient);
                let Some(donor) = donor_subclass(&counts, deficient, from_left) else {
                    return false;
                };
                let mut chosen: Option<usize> = None;
                for idx in members
                    .iter()
                    .copied()
                    .filter(|idx| units[*idx].subclass == donor)
                {
                    let better = chosen.is_none_or(|best| {
                        if from_left {
                            units[idx].score >= units[best].score
                        } else {
                            units[idx].score <= units[best].score
                        }
                    });
                    if better {
                        chosen = Some(idx);
                    }
                }
                let Some(chosen) = chosen else {
                    return false;
                };
                units[chosen].subclass = deficient;
                counts[deficient] += 1;
                counts[donor] -= 1;
            }
            for count in &mut counts {
                *count = count.saturating_sub(1);
            }
        }
    }
    true
}

/// Whether the subclass `deficient` takes a unit from below it.
fn look_left(counts: &[usize], deficient: usize) -> bool {
    if deficient + 1 == counts.len() {
        return true;
    }
    if deficient == 0 {
        return false;
    }
    let pull: f64 = counts
        .iter()
        .enumerate()
        .filter(|(other, count)| *other != deficient && **count > 1)
        .map(|(other, count)| (to_f64(*count) - 1.0) / (to_f64(other) - to_f64(deficient)))
        .sum();
    pull <= 0.0
}

/// The nearest subclass on the chosen side with a unit to give, or on the other
/// side when that side has none.
fn donor_subclass(counts: &[usize], deficient: usize, from_left: bool) -> Option<usize> {
    let below = (0..deficient).rev().find(|other| counts[*other] > 0);
    let above = (deficient + 1..counts.len()).find(|other| counts[*other] > 0);
    if from_left {
        below.or(above)
    } else {
        above.or(below)
    }
}

fn valid_subclasses(anchor_counts: &[usize], candidate_counts: &[usize]) -> Vec<bool> {
    anchor_counts
        .iter()
        .zip(candidate_counts)
        .map(|(anchor_count, candidate_count)| *anchor_count > 0 && *candidate_count > 0)
        .collect()
}

fn build_subclass_summaries(
    anchor_counts: &[usize],
    candidate_counts: &[usize],
) -> Vec<SubclassSummary> {
    anchor_counts
        .iter()
        .zip(candidate_counts)
        .enumerate()
        .map(|(subclass, (anchor_count, candidate_count))| {
            let total = *anchor_count + *candidate_count;
            let subclass_propensity = if total == 0 {
                None
            } else {
                Some(to_f64(*anchor_count) / to_f64(total))
            };

            SubclassSummary {
                subclass,
                anchor_count: *anchor_count,
                candidate_count: *candidate_count,
                subclass_propensity,
            }
        })
        .collect()
}

fn subclass_weight_for_estimand(estimand: Estimand, propensity: f64, is_anchor_unit: bool) -> f64 {
    match estimand {
        Estimand::Att => {
            if is_anchor_unit {
                1.0
            } else {
                propensity / (1.0 - propensity)
            }
        }
        Estimand::Atc => {
            if is_anchor_unit {
                (1.0 - propensity) / propensity
            } else {
                1.0
            }
        }
        Estimand::Ate | Estimand::Atm => {
            if is_anchor_unit {
                1.0 / propensity
            } else {
                1.0 / (1.0 - propensity)
            }
        }
    }
}

fn dropped_only_outcome<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    estimand: Estimand,
    requested_subclasses: usize,
) -> SubclassificationOutcome {
    let mut dropped_ids = anchors
        .iter()
        .map(|row| row.id().to_string())
        .collect::<Vec<_>>();
    dropped_ids.extend(candidates.iter().map(|row| row.id().to_string()));

    SubclassificationOutcome {
        estimand,
        requested_subclasses,
        dropped_ids,
        ..SubclassificationOutcome::default()
    }
}

use super::to_f64;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    fn record(id: &str) -> BaseRecord {
        BaseRecord::new(
            id,
            NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid test date"),
        )
    }

    fn assert_close(value: f64, expected: f64) {
        assert!((value - expected).abs() < 1e-12);
    }

    #[test]
    fn pooled_subclassification_preserves_overlap_strata() {
        let anchors = vec![record("a1"), record("a2")];
        let candidates = vec![record("c1"), record("c2")];
        let scores = HashMap::from([
            (String::from("a1"), 0.10),
            (String::from("a2"), 0.30),
            (String::from("c1"), 0.20),
            (String::from("c2"), 0.40),
        ]);
        let config = SubclassificationConfig {
            subclasses: 2,
            reference_group: Some(SubclassReferenceGroup::Pooled),
            min_n: 0,
        };

        let outcome = subclassify_by_propensity_score_map(
            &anchors,
            &candidates,
            &scores,
            Estimand::Att,
            &config,
        );

        assert_eq!(outcome.realized_subclasses, 2);
        assert_eq!(outcome.retained_units(), 4);
        assert!(outcome.dropped_ids.is_empty(), "{:?}", outcome.dropped_ids);
        assert_close(*outcome.weights.get("a1").expect("anchor weight"), 1.0);
        assert_close(*outcome.weights.get("c1").expect("candidate weight"), 1.0);
    }

    #[test]
    fn non_overlap_subclasses_are_dropped() {
        let anchors = vec![record("a1"), record("a2")];
        let candidates = vec![record("c1"), record("c2")];
        let scores = HashMap::from([
            (String::from("a1"), 0.10),
            (String::from("a2"), 0.20),
            (String::from("c1"), 0.80),
            (String::from("c2"), 0.90),
        ]);
        let config = SubclassificationConfig {
            subclasses: 2,
            reference_group: Some(SubclassReferenceGroup::Pooled),
            min_n: 0,
        };

        let outcome = subclassify_by_propensity_score_map(
            &anchors,
            &candidates,
            &scores,
            Estimand::Ate,
            &config,
        );

        assert_eq!(outcome.realized_subclasses, 0);
        assert!(outcome.assignments.is_empty(), "{:?}", outcome.assignments);
        assert_eq!(outcome.dropped_ids.len(), 4);
    }

    #[test]
    fn ate_weights_follow_subclass_propensities() {
        let anchors = vec![record("a1"), record("a2"), record("a3")];
        let candidates = vec![record("c1"), record("c2"), record("c3")];
        let scores = HashMap::from([
            (String::from("a1"), 0.10),
            (String::from("a2"), 0.11),
            (String::from("a3"), 0.90),
            (String::from("c1"), 0.12),
            (String::from("c2"), 0.92),
            (String::from("c3"), 0.13),
        ]);
        let config = SubclassificationConfig {
            subclasses: 2,
            reference_group: Some(SubclassReferenceGroup::Pooled),
            min_n: 0,
        };

        let outcome = subclassify_by_propensity_score_map(
            &anchors,
            &candidates,
            &scores,
            Estimand::Ate,
            &config,
        );

        assert_close(*outcome.weights.get("a1").expect("a1 weight"), 1.5);
        assert_close(*outcome.weights.get("a2").expect("a2 weight"), 1.5);
        assert_close(*outcome.weights.get("c1").expect("c1 weight"), 3.0);
        assert_close(*outcome.weights.get("a3").expect("a3 weight"), 3.0);
        assert_close(*outcome.weights.get("c2").expect("c2 weight"), 1.5);
        assert_close(*outcome.weights.get("c3").expect("c3 weight"), 1.5);
    }

    #[test]
    fn a_score_at_a_cutpoint_goes_to_the_upper_subclass() {
        let anchors = vec![record("a1"), record("a2"), record("a3")];
        let candidates = vec![record("c1"), record("c2"), record("c3")];
        let scores = HashMap::from([
            (String::from("a1"), 0.1),
            (String::from("c1"), 0.2),
            (String::from("a2"), 0.3),
            (String::from("c2"), 0.3),
            (String::from("c3"), 0.4),
            (String::from("a3"), 0.5),
        ]);
        let config = SubclassificationConfig {
            subclasses: 2,
            reference_group: Some(SubclassReferenceGroup::Pooled),
            min_n: 0,
        };
        let outcome = subclassify_by_propensity_score_map(
            &anchors,
            &candidates,
            &scores,
            Estimand::Ate,
            &config,
        );
        // The pooled median is 0.3, and both units scored 0.3 are in the upper
        // subclass.
        assert_close(outcome.quantile_cuts[1], 0.3);
        assert_eq!(outcome.assignments["a1"], 0);
        assert_eq!(outcome.assignments["c1"], 0);
        assert_eq!(outcome.assignments["a2"], 1);
        assert_eq!(outcome.assignments["c2"], 1);
    }

    #[test]
    fn an_empty_subclass_takes_the_nearest_unit_of_the_missing_group() {
        let anchors = vec![record("a1"), record("a2"), record("a3")];
        let candidates = vec![record("c1"), record("c2"), record("c3"), record("c4")];
        let scores = HashMap::from([
            (String::from("a1"), 0.1),
            (String::from("a2"), 0.5),
            (String::from("a3"), 0.9),
            (String::from("c1"), 0.12),
            (String::from("c2"), 0.15),
            (String::from("c3"), 0.8),
            (String::from("c4"), 0.85),
        ]);
        let config = SubclassificationConfig {
            subclasses: 3,
            reference_group: Some(SubclassReferenceGroup::Anchors),
            min_n: 1,
        };
        let outcome = subclassify_by_propensity_score_map(
            &anchors,
            &candidates,
            &scores,
            Estimand::Att,
            &config,
        );
        // The middle subclass holds a2 and no candidate. Candidates sit below
        // (two) and above (two) it, so it looks left and takes c2, the lower
        // candidate closest to it.
        assert_eq!(outcome.assignments["a2"], 1);
        assert_eq!(outcome.assignments["c2"], 1);
        assert_eq!(outcome.assignments["c1"], 0);
        assert_eq!(outcome.assignments["c3"], 2);
        assert_eq!(outcome.realized_subclasses, 3);
        assert!(outcome.dropped_ids.is_empty(), "{:?}", outcome.dropped_ids);
    }

    #[test]
    fn too_few_units_for_min_n_drops_everything() {
        let anchors = vec![record("a1"), record("a2")];
        let candidates = vec![record("c1")];
        let scores = HashMap::from([
            (String::from("a1"), 0.1),
            (String::from("a2"), 0.5),
            (String::from("c1"), 0.3),
        ]);
        let config = SubclassificationConfig {
            subclasses: 2,
            reference_group: Some(SubclassReferenceGroup::Pooled),
            min_n: 1,
        };
        let outcome = subclassify_by_propensity_score_map(
            &anchors,
            &candidates,
            &scores,
            Estimand::Att,
            &config,
        );
        assert!(outcome.assignments.is_empty(), "{:?}", outcome.assignments);
        assert_eq!(outcome.dropped_ids.len(), 3);
    }
}
