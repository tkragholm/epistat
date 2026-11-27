//! Subclassification on the propensity score.
//!
//! # Algorithm
//!
//! The cutpoints are the quantiles of the reference group's scores at
//! probabilities `0, 1/k, ..., 1` for `k` subclasses, with R's type 7 rule:
//! the quantile at probability `p` of `n` sorted scores `x_(1) <= ... <= x_(n)`
//! is `x_(j) + g (x_(j+1) - x_(j))` with `j + g = 1 + (n - 1) p`. The
//! subclasses are the intervals between consecutive cutpoints, closed on the
//! left, and the last is closed on both sides. Intervals without a unit are
//! left out and the others renumbered in order.
//!
//! A subclass with fewer than `min_n` units of a group takes units of that
//! group from the nearest subclass that has some to give: the unit whose score
//! is closest to the deficient subclass. A subclass without both groups is
//! dropped, and so are all its units.
//!
//! # Weights
//!
//! With `p` the share of anchors among the units of a subclass, an anchor
//! weighs
//!
//! ```text
//! ATT: 1        ATC: (1 - p) / p        ATE: 1 / p
//! ```
//!
//! and a candidate weighs
//!
//! ```text
//! ATT: p / (1 - p)        ATC: 1        ATE: 1 / (1 - p)
//! ```
//!
//! The weights are not rescaled to sum to the number of kept units in each
//! group.
//!
//! # References
//!
//! 1. Rosenbaum PR, Rubin DB. Reducing bias in observational studies using
//!    subclassification on the propensity score. Journal of the American
//!    Statistical Association 1984;79:516-524.
//! 2. Hyndman RJ, Fan Y. Sample quantiles in statistical packages. The American
//!    Statistician 1996;50:361-365.

use super::exact::unit_ids;
use super::records::MatchingRecord;
use crate::Error;
use crate::types::Estimand;
use rayon::prelude::*;
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
    /// - ATT: `Anchors`
    /// - ATC: `Candidates`
    /// - ATE and ATM: `Pooled`
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
///
/// Units are numbered by position: the anchors in the order given, then the
/// candidates in the order given. [`subclass`](Self::subclass) is indexed by
/// that position. A unit that is dropped has no subclass and no weight. The
/// weights are not stored; each is derived from the anchor share of the unit's
/// subclass in [`subclasses`](Self::subclasses) when asked for.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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
    /// The id of each unit.
    pub ids: Vec<String>,
    /// Number of anchors. The units before this position are anchors.
    pub anchor_count: usize,
    /// Subclass of each unit, `None` for a unit that is dropped because its
    /// score is missing or not finite or because its subclass lacks a group.
    /// Subclasses are numbered from 0 in order of score, skipping any that no
    /// unit falls in.
    pub subclass: Vec<Option<usize>>,
    /// Subclass summaries.
    pub subclasses: Vec<SubclassSummary>,
    /// The units that common-support trimming left out before matching, empty
    /// when the match did not trim. They are not among the units above.
    #[serde(default)]
    pub trimmed: crate::TrimmedUnits,
}

impl SubclassificationOutcome {
    /// Number of units, kept or dropped.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether there are no units.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The id of the unit at `position`.
    ///
    /// # Panics
    ///
    /// When `position` is not below [`len`](Self::len).
    #[must_use]
    pub fn id(&self, position: usize) -> &str {
        &self.ids[position]
    }

    /// Whether the unit at `position` is an anchor.
    #[must_use]
    pub const fn is_anchor(&self, position: usize) -> bool {
        position < self.anchor_count
    }

    /// Number of retained units.
    #[must_use]
    pub fn retained_units(&self) -> usize {
        self.subclass.iter().filter(|s| s.is_some()).count()
    }

    /// The retained units with their subclass, in position order.
    pub fn assignments(&self) -> impl Iterator<Item = (&str, usize)> + '_ {
        self.ids
            .iter()
            .zip(&self.subclass)
            .filter_map(|(id, subclass)| subclass.map(|subclass| (id.as_str(), subclass)))
    }

    /// Ids of the units that were dropped, in position order.
    #[must_use]
    pub fn dropped_ids(&self) -> Vec<&str> {
        self.ids
            .iter()
            .zip(&self.subclass)
            .filter(|(_, subclass)| subclass.is_none())
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// The subclassification weight of the unit at `position`, `None` for a
    /// dropped unit.
    ///
    /// # Panics
    ///
    /// When `position` is not below [`len`](Self::len).
    #[must_use]
    pub fn weight(&self, position: usize) -> Option<f64> {
        let share = self.subclasses[self.subclass[position]?].subclass_propensity?;
        Some(subclass_weight_for_estimand(
            self.estimand,
            share,
            self.is_anchor(position),
        ))
    }

    /// The weights of all units by position, with 0 for a dropped unit.
    #[must_use]
    pub fn weights(&self) -> Vec<f64> {
        (0..self.len())
            .into_par_iter()
            .map(|position| self.weight(position).unwrap_or(0.0))
            .collect()
    }
}

/// One unit with a usable score.
struct ScoredUnit {
    position: usize,
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
///
/// # Algorithm
///
/// The scores of the units are read in parallel, the reference scores are
/// sorted, the cutpoints are read off the sorted scores and each unit finds its
/// interval by binary search among them. Intervals that hold no unit are
/// skipped by a pass over a table of the intervals, not a sort of the units.
/// The sort is `O(n log n)`, the rest `O(n log k)` for `k` subclasses, and the
/// outcome holds one id and one subclass number per unit.
///
/// # Errors
///
/// Returns [`Error::NoSubclasses`] when `config.subclasses`
/// is 0.
pub fn subclassify_by_propensity_score_map<R: MatchingRecord, H: BuildHasher + Sync>(
    anchors: &[R],
    candidates: &[R],
    propensity_scores: &HashMap<String, f64, H>,
    estimand: Estimand,
    config: &SubclassificationConfig,
) -> Result<SubclassificationOutcome, Error> {
    subclassify_with_set_aside_scores(
        anchors,
        candidates,
        propensity_scores,
        estimand,
        config,
        &[],
        &[],
    )
}

/// As [`subclassify_by_propensity_score_map`], with the scores of units that
/// were set aside by common-support trimming also entering the reference
/// group's quantiles.
///
/// `set_aside_anchors` and `set_aside_candidates` are the finite scores of the
/// trimmed anchors and candidates. They are not subclassified and get no
/// weight. `matchit()` takes the cutpoints over every unit of the reference
/// group, discarded ones included, and this reproduces that. Non-finite scores
/// are ignored.
///
/// # Errors
///
/// As [`subclassify_by_propensity_score_map`].
pub fn subclassify_with_set_aside_scores<R: MatchingRecord, H: BuildHasher + Sync>(
    anchors: &[R],
    candidates: &[R],
    propensity_scores: &HashMap<String, f64, H>,
    estimand: Estimand,
    config: &SubclassificationConfig,
    set_aside_anchors: &[f64],
    set_aside_candidates: &[f64],
) -> Result<SubclassificationOutcome, Error> {
    if config.subclasses == 0 {
        return Err(Error::NoSubclasses);
    }
    let requested_subclasses = config.subclasses;
    let reference_group = resolve_reference_group(config.reference_group, estimand);
    let scores: Vec<Option<f64>> = anchors
        .par_iter()
        .chain(candidates.par_iter())
        .map(|row| {
            propensity_scores
                .get(row.id())
                .copied()
                .filter(|score| score.is_finite())
        })
        .collect();
    let mut reference_scores = sorted_reference_scores(&scores, anchors.len(), reference_group);
    let extra = |values: &[f64]| -> Vec<f64> {
        values
            .iter()
            .copied()
            .filter(|score| score.is_finite())
            .collect()
    };
    match reference_group {
        SubclassReferenceGroup::Anchors => reference_scores.extend(extra(set_aside_anchors)),
        SubclassReferenceGroup::Candidates => reference_scores.extend(extra(set_aside_candidates)),
        SubclassReferenceGroup::Pooled => {
            reference_scores.extend(extra(set_aside_anchors));
            reference_scores.extend(extra(set_aside_candidates));
        }
    }
    reference_scores.par_sort_unstable_by(f64::total_cmp);
    if reference_scores.is_empty() {
        return Ok(dropped_only_outcome(
            anchors,
            candidates,
            estimand,
            requested_subclasses,
        ));
    }

    let quantile_cuts = quantile_cuts(&reference_scores, requested_subclasses);
    let interior = &quantile_cuts[1..quantile_cuts.len() - 1];

    let mut units: Vec<ScoredUnit> = scores
        .par_iter()
        .enumerate()
        .filter_map(|(position, score)| {
            score.map(|score| ScoredUnit {
                position,
                is_anchor: position < anchors.len(),
                score,
                subclass: interior.partition_point(|cut| *cut <= score),
            })
        })
        .collect();
    // Intervals that no unit falls in, which tied quantiles produce, are not
    // subclasses.
    let subclass_count = renumber_present_subclasses(&mut units, interior.len() + 1);

    if config.min_n > 0 && !fill_deficient_subclasses(&mut units, subclass_count, config.min_n) {
        let mut outcome = dropped_only_outcome(anchors, candidates, estimand, requested_subclasses);
        outcome.quantile_cuts = quantile_cuts;
        return Ok(outcome);
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

    let mut subclass = vec![None; scores.len()];
    for unit in &units {
        if valid_subclasses[unit.subclass]
            && subclasses[unit.subclass].subclass_propensity.is_some()
        {
            subclass[unit.position] = Some(unit.subclass);
        }
    }

    Ok(SubclassificationOutcome {
        estimand,
        requested_subclasses,
        realized_subclasses,
        quantile_cuts,
        ids: unit_ids(anchors, candidates),
        anchor_count: anchors.len(),
        subclass,
        subclasses,
        trimmed: crate::TrimmedUnits::default(),
    })
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

/// The finite scores of the reference group, sorted. `scores` holds the score
/// of every unit by position, with `None` for a score that is missing or not
/// finite, and the first `anchor_count` units are the anchors.
fn sorted_reference_scores(
    scores: &[Option<f64>],
    anchor_count: usize,
    reference_group: SubclassReferenceGroup,
) -> Vec<f64> {
    let (anchors, candidates) = scores.split_at(anchor_count);
    let mut scores: Vec<f64> = match reference_group {
        SubclassReferenceGroup::Anchors => anchors.iter().flatten().copied().collect(),
        SubclassReferenceGroup::Candidates => candidates.iter().flatten().copied().collect(),
        SubclassReferenceGroup::Pooled => scores.iter().flatten().copied().collect(),
    };
    // Scores equal under the total order are the same bits, so the order among
    // them cannot show.
    scores.par_sort_unstable_by(f64::total_cmp);
    scores
}

/// The `subclasses + 1` quantiles at probabilities `0, 1/k, ..., 1` of the
/// sorted scores, R's type 7. The probabilities are multiples of `1/k` as
/// `seq(0, 1, length.out = k + 1)` builds them, so a quantile that falls on an
/// order statistic is that statistic exactly. `sorted_scores` must be finite
/// and not empty.
fn quantile_cuts(sorted_scores: &[f64], subclasses: usize) -> Vec<f64> {
    let step = 1.0 / to_f64(subclasses);
    (0..=subclasses)
        .map(|bucket| {
            let probability = if bucket == subclasses {
                1.0
            } else {
                to_f64(bucket) * step
            };
            epistat_stats::quantile::type7(sorted_scores, probability)
                .expect("the scores are finite and non-empty and the probability lies in [0, 1]")
        })
        .collect()
}

/// Renumber the subclasses that some unit occupies as `0..count`, keeping
/// their order, and return `count`. `intervals` is the number of subclasses
/// before renumbering.
fn renumber_present_subclasses(units: &mut [ScoredUnit], intervals: usize) -> usize {
    let mut rank = vec![0_usize; intervals];
    for unit in units.iter() {
        rank[unit.subclass] = 1;
    }
    let mut count = 0;
    for slot in &mut rank {
        let present = *slot == 1;
        *slot = count;
        count += usize::from(present);
    }
    units
        .par_iter_mut()
        .for_each(|unit| unit.subclass = rank[unit.subclass]);
    count
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

pub(super) fn build_subclass_summaries(
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

pub(super) fn subclass_weight_for_estimand(
    estimand: Estimand,
    propensity: f64,
    is_anchor_unit: bool,
) -> f64 {
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
    SubclassificationOutcome {
        estimand,
        requested_subclasses,
        ids: unit_ids(anchors, candidates),
        anchor_count: anchors.len(),
        subclass: vec![None; anchors.len() + candidates.len()],
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

    fn weights(outcome: &SubclassificationOutcome) -> HashMap<String, f64> {
        let (ids, values) = outcome.weights_sorted();
        ids.into_iter().map(str::to_owned).zip(values).collect()
    }

    fn assigned(outcome: &SubclassificationOutcome) -> HashMap<String, usize> {
        outcome
            .assignments()
            .map(|(id, subclass)| (id.to_owned(), subclass))
            .collect()
    }

    /// The method as it was written, with a string id on every unit, maps by
    /// id for the outcome and a sort to find the occupied subclasses, kept to
    /// check the columnar one against.
    mod reference {
        use super::*;

        #[derive(Debug)]
        pub struct Maps {
            pub realized: usize,
            pub quantile_cuts: Vec<f64>,
            pub assignments: HashMap<String, usize>,
            pub weights: HashMap<String, f64>,
            pub dropped_ids: Vec<String>,
            pub subclasses: Vec<SubclassSummary>,
        }

        pub fn subclassify<R: MatchingRecord>(
            anchors: &[R],
            candidates: &[R],
            scores: &HashMap<String, f64>,
            estimand: Estimand,
            config: &SubclassificationConfig,
        ) -> Maps {
            let reference_group = resolve_reference_group(config.reference_group, estimand);
            let finite = |rows: &[R]| -> Vec<f64> {
                rows.iter()
                    .filter_map(|row| scores.get(row.id()).copied())
                    .filter(|score| score.is_finite())
                    .collect()
            };
            let mut reference_scores = match reference_group {
                SubclassReferenceGroup::Anchors => finite(anchors),
                SubclassReferenceGroup::Candidates => finite(candidates),
                SubclassReferenceGroup::Pooled => {
                    let mut pooled = finite(anchors);
                    pooled.extend(finite(candidates));
                    pooled
                }
            };
            reference_scores.sort_unstable_by(f64::total_cmp);
            let everyone = |dropped: Vec<String>| Maps {
                realized: 0,
                quantile_cuts: Vec::new(),
                assignments: HashMap::new(),
                weights: HashMap::new(),
                dropped_ids: dropped,
                subclasses: Vec::new(),
            };
            let all_ids = || -> Vec<String> {
                anchors
                    .iter()
                    .chain(candidates)
                    .map(|row| row.id().to_string())
                    .collect()
            };
            if reference_scores.is_empty() {
                return everyone(all_ids());
            }
            let cuts = quantile_cuts(&reference_scores, config.subclasses);
            let interior = &cuts[1..cuts.len() - 1];
            let mut dropped_ids = Vec::new();
            let mut units = Vec::new();
            let mut ids = Vec::new();
            for (rows, is_anchor) in [(anchors, true), (candidates, false)] {
                for row in rows {
                    match scores
                        .get(row.id())
                        .copied()
                        .filter(|score| score.is_finite())
                    {
                        Some(score) => {
                            units.push(ScoredUnit {
                                position: ids.len(),
                                is_anchor,
                                score,
                                subclass: interior.partition_point(|cut| *cut <= score),
                            });
                            ids.push(row.id().to_string());
                        }
                        None => dropped_ids.push(row.id().to_string()),
                    }
                }
            }
            let mut present: Vec<usize> = units.iter().map(|unit| unit.subclass).collect();
            present.sort_unstable();
            present.dedup();
            for unit in &mut units {
                unit.subclass = present.partition_point(|value| *value < unit.subclass);
            }
            let subclass_count = present.len();
            if config.min_n > 0
                && !fill_deficient_subclasses(&mut units, subclass_count, config.min_n)
            {
                let mut out = everyone(all_ids());
                out.quantile_cuts = cuts;
                return out;
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
            let valid = valid_subclasses(&anchor_counts, &candidate_counts);
            let subclasses = build_subclass_summaries(&anchor_counts, &candidate_counts);
            let realized = valid.iter().filter(|v| **v).count();
            let mut assignments = HashMap::new();
            let mut weights = HashMap::new();
            for unit in units {
                let id = ids[unit.position].clone();
                let propensity = subclasses[unit.subclass].subclass_propensity;
                match propensity.filter(|_| valid[unit.subclass]) {
                    Some(propensity) => {
                        weights.insert(
                            id.clone(),
                            subclass_weight_for_estimand(estimand, propensity, unit.is_anchor),
                        );
                        assignments.insert(id, unit.subclass);
                    }
                    None => dropped_ids.push(id),
                }
            }
            Maps {
                realized,
                quantile_cuts: cuts,
                assignments,
                weights,
                dropped_ids,
                subclasses,
            }
        }
    }

    #[test]
    fn the_columns_equal_the_maps_of_the_per_unit_version() {
        use crate::matching::exact::reference::Lcg;
        for (seed, anchors, candidates, nan_share, tied) in [
            (1_u64, 40_usize, 90_usize, 0_u64, false),
            (2, 500, 1500, 0, false),
            (3, 500, 1500, 20, true),
            (4, 20_000, 30_000, 0, false),
            (5, 3000, 3000, 50, true),
            (6, 5, 4, 0, false),
        ] {
            let mut rng = Lcg(seed);
            let make = |prefix: &str, count: usize| -> Vec<BaseRecord> {
                (0..count)
                    .map(|i| record(&format!("{prefix}{i}")))
                    .collect()
            };
            let a = make("a", anchors);
            let c = make("c", candidates);
            let mut scores = HashMap::new();
            for row in a.iter().chain(&c) {
                let draw = if tied {
                    // Few values, so that quantile cuts tie and units tie.
                    rng.below(8) as f64 / 8.0
                } else {
                    rng.below(1_000_000) as f64 / 1_000_000.0
                };
                let value = match rng.below(100) {
                    n if n < nan_share => f64::NAN,
                    _ => draw,
                };
                if rng.below(97) != 0 {
                    scores.insert(row.id().to_string(), value);
                }
            }
            for estimand in [Estimand::Att, Estimand::Atc, Estimand::Ate] {
                for subclasses in [1, 5, 6, 40] {
                    for min_n in [0, 1, 3] {
                        for reference_group in [
                            None,
                            Some(SubclassReferenceGroup::Candidates),
                            Some(SubclassReferenceGroup::Pooled),
                        ] {
                            let config = SubclassificationConfig {
                                subclasses,
                                reference_group,
                                min_n,
                            };
                            let context = format!(
                                "seed {seed} {estimand:?} {subclasses} {min_n} {reference_group:?}"
                            );
                            let new = subclassify_by_propensity_score_map(
                                &a, &c, &scores, estimand, &config,
                            )
                            .unwrap();
                            let old = reference::subclassify(&a, &c, &scores, estimand, &config);
                            assert_eq!(new.realized_subclasses, old.realized, "{context}");
                            assert_eq!(new.subclasses, old.subclasses, "{context}");
                            assert_eq!(
                                new.quantile_cuts
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                old.quantile_cuts
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                "{context}"
                            );
                            assert_eq!(assigned(&new), old.assignments, "{context}");
                            let new_weights = weights(&new);
                            assert_eq!(new_weights.len(), old.weights.len(), "{context}");
                            for (id, weight) in &old.weights {
                                assert_eq!(
                                    new_weights[id].to_bits(),
                                    weight.to_bits(),
                                    "{context}"
                                );
                            }
                            // The dropped units come in position order, where the
                            // per-unit version listed those without a score first.
                            let mut new_dropped: Vec<String> =
                                new.dropped_ids().into_iter().map(str::to_owned).collect();
                            let mut old_dropped = old.dropped_ids;
                            new_dropped.sort();
                            old_dropped.sort();
                            assert_eq!(new_dropped, old_dropped, "{context}");
                            assert_eq!(new.len(), anchors + candidates);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_outcome_is_the_same_on_any_number_of_threads() {
        use crate::matching::exact::reference::Lcg;
        let mut rng = Lcg(3);
        let a: Vec<BaseRecord> = (0..40_000).map(|i| record(&format!("a{i}"))).collect();
        let c: Vec<BaseRecord> = (0..60_000).map(|i| record(&format!("c{i}"))).collect();
        let scores: HashMap<String, f64> = a
            .iter()
            .chain(&c)
            .map(|row| (row.id().to_string(), rng.below(100_000) as f64 / 100_000.0))
            .collect();
        epistat_testkit::assert_same_on_pools(&[1, 2, 4, 12], || {
            let out = subclassify_by_propensity_score_map(
                &a,
                &c,
                &scores,
                Estimand::Ate,
                &SubclassificationConfig::default(),
            )
            .unwrap();
            let mut values: Vec<f64> = out
                .subclass
                .iter()
                .map(|s| s.map_or(-1.0, |s| s as f64))
                .collect();
            values.extend(out.weights());
            values.extend(&out.quantile_cuts);
            values
        });
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
        )
        .expect("valid subclassification settings");

        assert_eq!(outcome.realized_subclasses, 2);
        assert_eq!(outcome.retained_units(), 4);
        assert!(
            outcome.dropped_ids().is_empty(),
            "{:?}",
            outcome.dropped_ids()
        );
        assert_close(*weights(&outcome).get("a1").expect("anchor weight"), 1.0);
        assert_close(*weights(&outcome).get("c1").expect("candidate weight"), 1.0);
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
        )
        .expect("valid subclassification settings");

        assert_eq!(outcome.realized_subclasses, 0);
        assert_eq!(outcome.retained_units(), 0, "{:?}", outcome.subclass);
        assert_eq!(outcome.dropped_ids().len(), 4);
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
        )
        .expect("valid subclassification settings");

        assert_close(*weights(&outcome).get("a1").expect("a1 weight"), 1.5);
        assert_close(*weights(&outcome).get("a2").expect("a2 weight"), 1.5);
        assert_close(*weights(&outcome).get("c1").expect("c1 weight"), 3.0);
        assert_close(*weights(&outcome).get("a3").expect("a3 weight"), 3.0);
        assert_close(*weights(&outcome).get("c2").expect("c2 weight"), 1.5);
        assert_close(*weights(&outcome).get("c3").expect("c3 weight"), 1.5);
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
        )
        .expect("valid subclassification settings");
        // The pooled median is 0.3, and both units scored 0.3 are in the upper
        // subclass.
        assert_close(outcome.quantile_cuts[1], 0.3);
        assert_eq!(assigned(&outcome)["a1"], 0);
        assert_eq!(assigned(&outcome)["c1"], 0);
        assert_eq!(assigned(&outcome)["a2"], 1);
        assert_eq!(assigned(&outcome)["c2"], 1);
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
        )
        .expect("valid subclassification settings");
        // The middle subclass holds a2 and no candidate. Candidates sit below
        // (two) and above (two) it, so it looks left and takes c2, the lower
        // candidate closest to it.
        assert_eq!(assigned(&outcome)["a2"], 1);
        assert_eq!(assigned(&outcome)["c2"], 1);
        assert_eq!(assigned(&outcome)["c1"], 0);
        assert_eq!(assigned(&outcome)["c3"], 2);
        assert_eq!(outcome.realized_subclasses, 3);
        assert!(
            outcome.dropped_ids().is_empty(),
            "{:?}",
            outcome.dropped_ids()
        );
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
        )
        .expect("valid subclassification settings");
        assert_eq!(outcome.retained_units(), 0, "{:?}", outcome.subclass);
        assert_eq!(outcome.dropped_ids().len(), 3);
    }
}

#[cfg(test)]
mod quantile_cut_reference_tests {
    use super::{quantile_cuts, to_f64};

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

    fn reference_cuts(sorted_scores: &[f64], subclasses: usize) -> Vec<f64> {
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

    /// The cuts equal a direct evaluation of the interpolation, bit for bit,
    /// for sample sizes and class counts around the ends.
    #[test]
    fn cuts_equal_the_hand_written_interpolation() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        for n in [1_usize, 2, 3, 7, 10, 11, 64, 101, 1000, 4097] {
            let mut scores: Vec<f64> = (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    // Repeated values make ties between neighbours.
                    ((state >> 40) % 997) as f64 / 997.0
                })
                .collect();
            scores.sort_by(f64::total_cmp);
            for subclasses in [1_usize, 2, 3, 5, 6, 10, 17, 50] {
                let got = quantile_cuts(&scores, subclasses);
                let expected = reference_cuts(&scores, subclasses);
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&expected), "n {n}, k {subclasses}");
            }
        }
    }
}
