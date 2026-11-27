//! Exact matching: units are sorted into strata by their values of the
//! matching variables, and a stratum is a subclass.
//!
//! # Algorithm
//!
//! The values of each variable are numbered by first appearance, and the
//! columns of numbers are combined into one stratum number per unit, so two
//! units share a stratum when they agree on every variable. A unit with no
//! value for a variable is an error. The cases and controls of each
//! stratum are counted, and a stratum without both groups drops its units.
//! The weights of a subclass are those of
//! [`subclassification`](super::subclassification): with `p` the case share
//! of the stratum, an case weighs 1 and a control `p / (1 - p)` for the
//! ATT, and the reverse for the ATC.
//!
//! The work is `O(n v)` for `n` units and `v` variables.
//!
//! # References
//!
//! 1. Iacus SM, King G, Porro G. Causal inference without balance checking:
//!    coarsened exact matching. Political Analysis 2012;20:1-24.

use super::codes::{combine_codes, first_appearance_codes, first_missing};
use super::records::MatchingRecord;
use super::subclassification::{
    SubclassSummary, build_subclass_summaries, subclass_weight_for_estimand,
};
use crate::Error;
use crate::types::Estimand;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Units sorted into subclasses, with the weights of each subclass.
///
/// The result of exact, coarsened exact and quick matching. Units are
/// numbered by position: the cases in the order given, then the controls
/// in the order given. Every column is indexed by that position. A unit that
/// fits no subclass has no subclass and no weight.
///
/// The weights are not stored. Each is derived from the case share of the
/// unit's subclass in [`subclasses`](Self::subclasses) when asked for, so the
/// result holds one id and one subclass number per unit and nothing else.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SubclassedUnits {
    /// The id of each unit.
    pub ids: Vec<String>,
    /// Number of cases. The units before this position are cases.
    pub case_count: usize,
    /// Subclass of each unit, `None` for a unit that is dropped.
    pub subclass: Vec<Option<usize>>,
    /// Counts and case share of each subclass.
    pub subclasses: Vec<SubclassSummary>,
    /// The units that common-support trimming left out before matching, empty
    /// when the match did not trim. They are not among the units above.
    #[serde(default)]
    pub trimmed: crate::TrimmedUnits,
}

impl SubclassedUnits {
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

    /// Whether the unit at `position` is an case.
    #[must_use]
    pub const fn is_case(&self, position: usize) -> bool {
        position < self.case_count
    }

    /// Number of kept units.
    #[must_use]
    pub fn retained_units(&self) -> usize {
        self.subclass.iter().filter(|s| s.is_some()).count()
    }

    /// The kept units with their subclass, in position order.
    pub fn assignments(&self) -> impl Iterator<Item = (&str, usize)> + '_ {
        self.ids
            .iter()
            .zip(&self.subclass)
            .filter_map(|(id, subclass)| subclass.map(|subclass| (id.as_str(), subclass)))
    }

    /// Ids of the units that fit no subclass, cases first, each group in the
    /// order given.
    #[must_use]
    pub fn dropped_ids(&self) -> Vec<&str> {
        self.ids
            .iter()
            .zip(&self.subclass)
            .filter(|(_, subclass)| subclass.is_none())
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// The weight of the unit at `position` for an estimand. `None` for a
    /// dropped unit and for an estimand other than the ATT, the ATC and the
    /// ATE, such as the matched-sample effect, which these methods do not
    /// define separately.
    ///
    /// # Panics
    ///
    /// When `position` is not below [`len`](Self::len).
    #[must_use]
    pub fn weight(&self, estimand: Estimand, position: usize) -> Option<f64> {
        if !has_subclass_weights(estimand) {
            return None;
        }
        let share = self.subclasses[self.subclass[position]?].subclass_propensity?;
        Some(subclass_weight_for_estimand(
            estimand,
            share,
            self.is_case(position),
        ))
    }

    /// The weights of all units for an estimand, by position, with 0 for a
    /// dropped unit, or `None` for an estimand other than the ATT, the ATC and
    /// the ATE.
    ///
    /// Subclass weights for the effect in the case group, in the control
    /// group or in the whole population. They are not rescaled. MatchIt
    /// rescales the weights of each group to sum to the number of its kept
    /// units.
    #[must_use]
    pub fn weights(&self, estimand: Estimand) -> Option<Vec<f64>> {
        if !has_subclass_weights(estimand) {
            return None;
        }
        Some(
            (0..self.len())
                .into_par_iter()
                .map(|position| self.weight(estimand, position).unwrap_or(0.0))
                .collect(),
        )
    }
}

/// Whether subclasses carry weights for `estimand`: the ATT, the ATC and the
/// ATE.
pub(crate) const fn has_subclass_weights(estimand: Estimand) -> bool {
    matches!(estimand, Estimand::Att | Estimand::Atc | Estimand::Ate)
}

/// Ids of the cases and then the controls.
pub(super) fn unit_ids<R: MatchingRecord>(anchors: &[R], candidates: &[R]) -> Vec<String> {
    anchors
        .par_iter()
        .chain(candidates.par_iter())
        .map(|row| row.id().to_string())
        .collect()
}

/// Match exactly on the named strata fields.
///
/// A stratum is a combination of the values of `variables` in
/// [`MatchingRecord::strata`]. Every unit in a stratum that holds both cases
/// and controls is kept and the stratum is a subclass. Units in a stratum
/// that holds one group only are dropped. This is MatchIt's
/// `method = "exact"`, which differs from
/// [`with_exact_matches`](crate::MatchJob::with_exact_matches) in keeping all
/// units of a stratum, with weights, rather than pairing them.
///
/// The weights of each subclass are those of propensity-score
/// subclassification, with the case share of the subclass in the place of
/// the propensity: for the case effect an case weighs 1 and a control the
/// ratio of cases to controls in its subclass, for the control effect the
/// reverse, and for the whole population an case weighs the subclass size over
/// its cases and a control the subclass size over its controls. They are
/// not rescaled. MatchIt rescales the weights of each group to sum to the
/// number of its kept units.
///
/// MatchIt numbers the subclasses in the order of first appearance within the
/// group of the first row of its data, where this function uses the cases.
/// The strata and weights are the same.
///
/// # Algorithm
///
/// The values of each variable are numbered by first appearance, and the
/// columns of numbers are joined into one stratum number per unit, as
/// described in the module `codes`. Counting the cases and controls of each
/// stratum then gives the subclasses. The work is `O(n * v)` for `n` units and
/// `v` variables, in parallel apart from the merge of distinct values, and the
/// outcome holds one string and one number per unit.
///
/// # Errors
///
/// [`Error::NoVariables`] without variables,
/// [`Error::MissingValue`] when a unit lacks a variable, and
/// [`Error::NoSharedStratum`] when no stratum holds both groups.
pub fn exact_match_subclasses<R: MatchingRecord, V: AsRef<str>>(
    anchors: &[R],
    candidates: &[R],
    variables: &[V],
) -> Result<SubclassedUnits, Error> {
    if variables.is_empty() {
        return Err(Error::NoVariables);
    }
    let (numbers, count) =
        stratum_numbers(anchors, candidates, variables).map_err(|(position, variable)| {
            Error::MissingValue {
                id: unit_at(anchors, candidates, position).id().to_string(),
                variable: variables[variable].as_ref().to_string(),
            }
        })?;
    subclasses_of_strata(anchors, candidates, &numbers, count)
}

pub(super) fn unit_at<'a, R>(anchors: &'a [R], candidates: &'a [R], position: usize) -> &'a R {
    if position < anchors.len() {
        &anchors[position]
    } else {
        &candidates[position - anchors.len()]
    }
}

/// The stratum number of every unit, cases then controls, numbered from 0
/// in order of first appearance, and the number of strata. The error is the
/// position of the first unit that lacks a value, with the index of the first
/// variable it lacks.
pub(super) fn stratum_numbers<R: MatchingRecord, V: AsRef<str>>(
    anchors: &[R],
    candidates: &[R],
    variables: &[V],
) -> Result<(Vec<u32>, usize), (usize, usize)> {
    let mut first_error: Option<(usize, usize)> = None;
    let mut columns = Vec::with_capacity(variables.len());
    for (index, variable) in variables.iter().enumerate() {
        let name = variable.as_ref();
        let values: Vec<Option<&str>> = anchors
            .par_iter()
            .chain(candidates.par_iter())
            .map(|row| row.stratum(name))
            .collect();
        if let Some(position) = first_missing(&values) {
            if first_error.is_none_or(|best| (position, index) < best) {
                first_error = Some((position, index));
            }
            continue;
        }
        if first_error.is_some() {
            continue;
        }
        let present: Vec<&str> = values.into_iter().flatten().collect();
        columns.push(first_appearance_codes(&present).0);
    }
    if let Some(error) = first_error {
        return Err(error);
    }
    Ok(combine_codes(&columns))
}

/// Subclasses and weights of units that already have a stratum number.
///
/// `numbers` holds the stratum of the cases and then of the controls, from
/// 0 in order of first appearance, cases first, as `exact_match_subclasses`
/// numbers them, and `strata` is their count. The subclasses are the strata
/// that hold both groups, and they keep that order. Coarsened exact matching
/// uses this with strata from binned values.
pub(crate) fn subclasses_of_strata<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    numbers: &[u32],
    strata: usize,
) -> Result<SubclassedUnits, Error> {
    let mut anchor_counts = vec![0_usize; strata];
    let mut candidate_counts = vec![0_usize; strata];
    let (anchor_numbers, candidate_numbers) = numbers.split_at(anchors.len());
    for stratum in anchor_numbers {
        anchor_counts[*stratum as usize] += 1;
    }
    for stratum in candidate_numbers {
        candidate_counts[*stratum as usize] += 1;
    }

    // Subclasses are the strata with both groups. Every one of them has an
    // anchor, so numbering the strata in creation order keeps the order of
    // first appearance among the anchors.
    let mut subclass_of = vec![None; strata];
    let mut kept_anchor_counts = Vec::new();
    let mut kept_candidate_counts = Vec::new();
    for stratum in 0..strata {
        if anchor_counts[stratum] > 0 && candidate_counts[stratum] > 0 {
            subclass_of[stratum] = Some(kept_anchor_counts.len());
            kept_anchor_counts.push(anchor_counts[stratum]);
            kept_candidate_counts.push(candidate_counts[stratum]);
        }
    }
    if kept_anchor_counts.is_empty() {
        return Err(Error::NoSharedStratum);
    }
    Ok(SubclassedUnits {
        ids: unit_ids(anchors, candidates),
        case_count: anchors.len(),
        subclass: numbers
            .par_iter()
            .map(|stratum| subclass_of[*stratum as usize])
            .collect(),
        subclasses: build_subclass_summaries(&kept_anchor_counts, &kept_candidate_counts),
        trimmed: crate::TrimmedUnits::default(),
    })
}

#[cfg(test)]
pub(super) mod reference {
    //! The implementation of exact matching that keyed strata by owned
    //! strings and returned maps by id, kept to check the columnar one against.

    use super::*;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;
    use std::collections::HashMap;

    /// The outcome as maps by id.
    #[derive(Debug, Default)]
    pub struct Maps {
        pub assignments: HashMap<String, usize>,
        pub dropped_ids: Vec<String>,
        pub subclasses: Vec<SubclassSummary>,
        pub att_weights: HashMap<String, f64>,
        pub atc_weights: HashMap<String, f64>,
        pub ate_weights: HashMap<String, f64>,
    }

    pub fn exact<R: MatchingRecord, V: AsRef<str>>(
        anchors: &[R],
        candidates: &[R],
        variables: &[V],
    ) -> Result<Maps, Error> {
        if variables.is_empty() {
            return Err(Error::NoVariables);
        }
        let key = |row: &R| -> Result<Vec<String>, Error> {
            variables
                .iter()
                .map(|variable| {
                    row.stratum(variable.as_ref())
                        .map(str::to_owned)
                        .ok_or_else(|| Error::MissingValue {
                            id: row.id().to_string(),
                            variable: variable.as_ref().to_string(),
                        })
                })
                .collect()
        };
        let mut stratum_of: HashMap<Vec<String>, usize> = HashMap::new();
        let mut anchor_strata = Vec::with_capacity(anchors.len());
        let mut candidate_strata = Vec::with_capacity(candidates.len());
        for (rows, strata) in [
            (anchors, &mut anchor_strata),
            (candidates, &mut candidate_strata),
        ] {
            for row in rows {
                let next = stratum_of.len();
                strata.push(*stratum_of.entry(key(row)?).or_insert(next));
            }
        }
        of_strata(
            anchors,
            candidates,
            &anchor_strata,
            &candidate_strata,
            stratum_of.len(),
        )
    }

    pub fn of_strata<R: MatchingRecord>(
        anchors: &[R],
        candidates: &[R],
        anchor_strata: &[usize],
        candidate_strata: &[usize],
        strata: usize,
    ) -> Result<Maps, Error> {
        let mut anchor_counts = vec![0_usize; strata];
        let mut candidate_counts = vec![0_usize; strata];
        for stratum in anchor_strata {
            anchor_counts[*stratum] += 1;
        }
        for stratum in candidate_strata {
            candidate_counts[*stratum] += 1;
        }
        let mut subclass_of = vec![None; anchor_counts.len()];
        let mut kept_anchor_counts = Vec::new();
        let mut kept_candidate_counts = Vec::new();
        for stratum in 0..anchor_counts.len() {
            if anchor_counts[stratum] > 0 && candidate_counts[stratum] > 0 {
                subclass_of[stratum] = Some(kept_anchor_counts.len());
                kept_anchor_counts.push(anchor_counts[stratum]);
                kept_candidate_counts.push(candidate_counts[stratum]);
            }
        }
        if kept_anchor_counts.is_empty() {
            return Err(Error::NoSharedStratum);
        }
        let subclasses = build_subclass_summaries(&kept_anchor_counts, &kept_candidate_counts);
        let mut outcome = Maps {
            subclasses,
            ..Maps::default()
        };
        for (rows, strata, is_case) in [
            (anchors, anchor_strata, true),
            (candidates, candidate_strata, false),
        ] {
            for (row, stratum) in rows.iter().zip(strata) {
                let Some(subclass) = subclass_of[*stratum] else {
                    outcome.dropped_ids.push(row.id().to_string());
                    continue;
                };
                let share = outcome.subclasses[subclass]
                    .subclass_propensity
                    .expect("a kept subclass has units");
                let id = row.id().to_string();
                for (weights, estimand) in [
                    (&mut outcome.att_weights, Estimand::Att),
                    (&mut outcome.atc_weights, Estimand::Atc),
                    (&mut outcome.ate_weights, Estimand::Ate),
                ] {
                    weights.insert(
                        id.clone(),
                        subclass_weight_for_estimand(estimand, share, is_case),
                    );
                }
                outcome.assignments.insert(id, subclass);
            }
        }
        Ok(outcome)
    }

    /// The columnar outcome holds exactly what the maps hold, bit for bit.
    pub fn assert_same(new: &SubclassedUnits, old: &Maps, context: &str) {
        assert_eq!(new.subclasses, old.subclasses, "{context}: summaries");
        assert_eq!(
            new.dropped_ids()
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>(),
            old.dropped_ids,
            "{context}: dropped ids"
        );
        assert_eq!(
            new.assignments().count(),
            old.assignments.len(),
            "{context}"
        );
        for (id, subclass) in new.assignments() {
            assert_eq!(old.assignments.get(id), Some(&subclass), "{context}: {id}");
        }
        for (estimand, map) in [
            (Estimand::Att, &old.att_weights),
            (Estimand::Atc, &old.atc_weights),
            (Estimand::Ate, &old.ate_weights),
        ] {
            let column = new.weights(estimand).expect("weights");
            for (position, id) in new.ids.iter().enumerate() {
                if let Some(weight) = map.get(id) {
                    assert_eq!(column[position].to_bits(), weight.to_bits(), "{context}");
                    assert_eq!(
                        new.weight(estimand, position).map(f64::to_bits),
                        Some(weight.to_bits())
                    );
                } else {
                    assert_eq!(column[position].to_bits(), 0.0_f64.to_bits(), "{context}");
                    assert_eq!(new.weight(estimand, position), None);
                }
            }
        }
    }

    /// Every subclass number and weight of an outcome, for comparing runs.
    #[expect(
        clippy::cast_precision_loss,
        reason = "test subclass numbers are small"
    )]
    pub fn numbers(out: &SubclassedUnits) -> Vec<f64> {
        let mut values: Vec<f64> = out
            .subclass
            .iter()
            .map(|s| s.map_or(-1.0, |s| s as f64))
            .collect();
        for estimand in [Estimand::Att, Estimand::Atc, Estimand::Ate] {
            values.extend(out.weights(estimand).expect("weights"));
        }
        values
    }

    /// A pseudo-random sequence, the same on every run.
    pub struct Lcg(pub u64);

    impl Lcg {
        pub fn below(&mut self, bound: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % bound
        }
    }

    /// Cases and controls with `sex`, `area` and `region` strata and a
    /// numeric `x` taken from few values, so that many units tie exactly.
    /// With `holes`, about 1 in 60 values is missing.
    pub fn records(
        seed: u64,
        anchors: usize,
        candidates: usize,
        areas: u64,
        holes: bool,
    ) -> (Vec<BaseRecord>, Vec<BaseRecord>) {
        let mut rng = Lcg(seed);
        let mut make = |prefix: &str, count: usize, skew: u64| {
            (0..count)
                .map(|i| {
                    let mut record = BaseRecord::new(
                        format!("{prefix}{i}"),
                        NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid test date"),
                    );
                    let missing = |rng: &mut Lcg| holes && rng.below(60) == 0;
                    if !missing(&mut rng) {
                        record
                            .strata
                            .insert("sex".into(), format!("s{}", rng.below(2)));
                    }
                    if !missing(&mut rng) {
                        record
                            .strata
                            .insert("area".into(), format!("a{}", rng.below(areas + skew)));
                    }
                    if !missing(&mut rng) {
                        record
                            .strata
                            .insert("region".into(), format!("r{}", rng.below(3)));
                    }
                    if !missing(&mut rng) {
                        record.numerics.insert(
                            "x".into(),
                            f64::from(u32::try_from(rng.below(12)).unwrap()) * 0.5,
                        );
                    }
                    record
                })
                .collect::<Vec<_>>()
        };
        let a = make("a", anchors, 0);
        let c = make("c", candidates, 1);
        (a, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;
    use std::collections::HashMap;

    fn record(id: &str, sex: &str, area: &str) -> BaseRecord {
        let mut record = BaseRecord::new(
            id,
            NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid test date"),
        );
        record.strata.insert("sex".into(), sex.into());
        record.strata.insert("area".into(), area.into());
        record
    }

    fn map(out: &SubclassedUnits) -> HashMap<String, usize> {
        out.assignments()
            .map(|(id, subclass)| (id.to_owned(), subclass))
            .collect()
    }

    fn weight(out: &SubclassedUnits, estimand: Estimand, id: &str) -> f64 {
        let position = out.ids.iter().position(|x| x == id).expect("unit");
        out.weight(estimand, position).expect("kept unit")
    }

    #[test]
    fn the_columns_equal_the_maps_of_the_string_keyed_version() {
        let mut agreed = 0;
        for (seed, anchors, candidates, areas, holes) in [
            (1_u64, 40_usize, 90_usize, 3_u64, false),
            (2, 300, 700, 5, false),
            (3, 1000, 1000, 40, false),
            (4, 5000, 20_000, 6, false),
            (5, 500, 800, 4, true),
            (6, 30_000, 50_000, 80, false),
            (7, 3, 4, 2, false),
        ] {
            let (a, c) = reference::records(seed, anchors, candidates, areas, holes);
            for variables in [
                vec!["sex"],
                vec!["sex", "area"],
                vec!["area", "region", "sex"],
                vec!["sex", "missing_everywhere"],
            ] {
                let context = format!("seed {seed} {variables:?}");
                let new = exact_match_subclasses(&a, &c, &variables);
                let old = reference::exact(&a, &c, &variables);
                match (new, old) {
                    (Ok(new), Ok(old)) => {
                        agreed += 1;
                        reference::assert_same(&new, &old, &context);
                    }
                    (Err(new), Err(old)) => assert_eq!(new, old, "{context}"),
                    (new, old) => panic!("{context}: {new:?} against {old:?}"),
                }
            }
        }
        assert!(agreed >= 15, "only {agreed} runs produced an outcome");
    }

    #[test]
    fn the_first_missing_value_is_reported_in_unit_order() {
        let (mut a, c) = reference::records(9, 50, 60, 3, false);
        a[30].strata.remove("area");
        a[10].strata.remove("sex");
        let error = exact_match_subclasses(&a, &c, &["area", "sex"]).unwrap_err();
        assert_eq!(
            error,
            Error::MissingValue {
                id: "a10".into(),
                variable: "sex".into()
            }
        );
    }

    #[test]
    fn the_outcome_is_the_same_on_any_number_of_threads() {
        let (a, c) = reference::records(11, 40_000, 60_000, 30, false);
        epistat_testkit::assert_same_on_pools(&[1, 2, 4, 12], || {
            let out = exact_match_subclasses(&a, &c, &["sex", "area", "region"]).unwrap();
            reference::numbers(&out)
        });
    }

    #[test]
    fn keeps_strata_with_both_groups_and_weights_them() {
        let anchors = vec![
            record("a1", "f", "n"),
            record("a2", "m", "n"),
            record("a3", "f", "n"),
            record("a4", "f", "s"),
        ];
        let candidates = vec![
            record("c1", "f", "n"),
            record("c2", "m", "n"),
            record("c3", "m", "n"),
            record("c4", "m", "n"),
            record("c5", "f", "s"),
            record("c6", "f", "s"),
        ];
        let out = exact_match_subclasses(&anchors, &candidates, &["sex", "area"]).unwrap();
        assert_eq!(out.dropped_ids().len(), 0);
        assert_eq!(map(&out)["a1"], 0);
        assert_eq!(map(&out)["a2"], 1);
        assert_eq!(map(&out)["a4"], 2);
        assert_eq!(map(&out)["c5"], 2);
        assert!((weight(&out, Estimand::Att, "a1") - 1.0).abs() < 1e-12);
        assert!((weight(&out, Estimand::Att, "c1") - 2.0).abs() < 1e-12);
        assert!((weight(&out, Estimand::Att, "c3") - 1.0 / 3.0).abs() < 1e-12);
        assert!((weight(&out, Estimand::Ate, "a2") - 4.0).abs() < 1e-12);
        assert!((weight(&out, Estimand::Ate, "c2") - 4.0 / 3.0).abs() < 1e-12);
        assert!((weight(&out, Estimand::Atc, "a1") - 0.5).abs() < 1e-12);
        assert!((weight(&out, Estimand::Atc, "c1") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn drops_units_in_strata_with_one_group() {
        let anchors = vec![record("a1", "f", "n"), record("a2", "m", "s")];
        let candidates = vec![record("c1", "f", "n"), record("c2", "f", "s")];
        let out = exact_match_subclasses(&anchors, &candidates, &["sex", "area"]).unwrap();
        assert_eq!(out.dropped_ids(), vec!["a2", "c2"]);
        assert_eq!(out.retained_units(), 2);
        assert_eq!(out.weight(Estimand::Att, 1), None);
    }

    #[test]
    fn refuses_what_cannot_be_stratified() {
        let anchors = vec![record("a1", "f", "n")];
        let candidates = vec![record("c1", "m", "n")];
        assert_eq!(
            exact_match_subclasses::<_, &str>(&anchors, &candidates, &[]).unwrap_err(),
            Error::NoVariables
        );
        assert_eq!(
            exact_match_subclasses(&anchors, &candidates, &["sex"]).unwrap_err(),
            Error::NoSharedStratum
        );
        assert_eq!(
            exact_match_subclasses(&anchors, &candidates, &["age"]).unwrap_err(),
            Error::MissingValue {
                id: "a1".into(),
                variable: "age".into()
            }
        );
    }
}
