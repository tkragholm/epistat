use super::records::MatchingRecord;
use super::subclassification::{
    SubclassSummary, build_subclass_summaries, subclass_weight_for_estimand,
};
use crate::types::Estimand;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Why exact matching could not run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExactMatchError {
    /// No exact matching variable was given.
    NoVariables,
    /// A unit has no value for one of the variables.
    MissingValue {
        /// Id of the unit.
        id: String,
        /// Name of the variable.
        variable: String,
    },
    /// No stratum holds units of both groups.
    NoSharedStratum,
}

impl fmt::Display for ExactMatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoVariables => write!(f, "no exact matching variable was given"),
            Self::MissingValue { id, variable } => {
                write!(f, "unit {id} has no value for the variable {variable}")
            }
            Self::NoSharedStratum => write!(f, "no stratum holds units of both groups"),
        }
    }
}

impl std::error::Error for ExactMatchError {}

/// Result of exact matching into strata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExactMatchOutcome {
    /// Subclass of each kept unit. Subclasses are numbered from 0 in the
    /// order in which their stratum first appears among the anchors.
    pub assignments: HashMap<String, usize>,
    /// Ids of the units in a stratum that lacks one of the groups, anchors
    /// first.
    pub dropped_ids: Vec<String>,
    /// Counts and anchor share of each subclass.
    pub subclasses: Vec<SubclassSummary>,
    /// Subclass weights for the effect in the anchor group.
    pub att_weights: HashMap<String, f64>,
    /// Subclass weights for the effect in the candidate group.
    pub atc_weights: HashMap<String, f64>,
    /// Subclass weights for the effect in the whole population.
    pub ate_weights: HashMap<String, f64>,
}

impl ExactMatchOutcome {
    /// Number of kept units.
    #[must_use]
    pub fn retained_units(&self) -> usize {
        self.assignments.len()
    }

    /// The weights for an estimand, or `None` for the matched-sample effect,
    /// which exact matching does not define separately.
    #[must_use]
    pub const fn weights(&self, estimand: Estimand) -> Option<&HashMap<String, f64>> {
        match estimand {
            Estimand::Att => Some(&self.att_weights),
            Estimand::Atc => Some(&self.atc_weights),
            Estimand::Ate => Some(&self.ate_weights),
            Estimand::Atm => None,
        }
    }
}

/// Match exactly on the named strata fields.
///
/// A stratum is a combination of the values of `variables` in
/// [`MatchingRecord::strata`]. Every unit in a stratum that holds both anchors
/// and candidates is kept and the stratum is a subclass. Units in a stratum
/// that holds one group only are dropped. This is MatchIt's
/// `method = "exact"`, which differs from
/// [`with_exact_matches`](crate::MatchJob::with_exact_matches) in keeping all
/// units of a stratum, with weights, rather than pairing them.
///
/// The weights of each subclass are those of propensity-score
/// subclassification, with the anchor share of the subclass in the place of
/// the propensity: for the anchor effect an anchor weighs 1 and a candidate the
/// ratio of anchors to candidates in its subclass, for the candidate effect the
/// reverse, and for the whole population an anchor weighs the subclass size over
/// its anchors and a candidate the subclass size over its candidates. They are
/// not rescaled. MatchIt rescales the weights of each group to sum to the
/// number of its kept units.
///
/// MatchIt numbers the subclasses in the order of first appearance within the
/// group of the first row of its data, where this function uses the anchors.
/// The strata and weights are the same.
///
/// # Errors
///
/// [`ExactMatchError::NoVariables`] without variables,
/// [`ExactMatchError::MissingValue`] when a unit lacks a variable, and
/// [`ExactMatchError::NoSharedStratum`] when no stratum holds both groups.
pub fn exact_match_subclasses<R: MatchingRecord, V: AsRef<str>>(
    anchors: &[R],
    candidates: &[R],
    variables: &[V],
) -> Result<ExactMatchOutcome, ExactMatchError> {
    if variables.is_empty() {
        return Err(ExactMatchError::NoVariables);
    }
    let key = |row: &R| -> Result<Vec<String>, ExactMatchError> {
        variables
            .iter()
            .map(|variable| {
                row.strata().get(variable.as_ref()).cloned().ok_or_else(|| {
                    ExactMatchError::MissingValue {
                        id: row.id().to_string(),
                        variable: variable.as_ref().to_string(),
                    }
                })
            })
            .collect()
    };

    // Stratum numbers in order of first appearance, anchors first.
    let mut stratum_of: HashMap<Vec<String>, usize> = HashMap::new();
    let mut anchor_counts: Vec<usize> = Vec::new();
    let mut candidate_counts: Vec<usize> = Vec::new();
    let mut anchor_strata = Vec::with_capacity(anchors.len());
    let mut candidate_strata = Vec::with_capacity(candidates.len());
    for (rows, is_anchor) in [(anchors, true), (candidates, false)] {
        for row in rows {
            let next = anchor_counts.len();
            let stratum = *stratum_of.entry(key(row)?).or_insert(next);
            if stratum == next {
                anchor_counts.push(0);
                candidate_counts.push(0);
            }
            if is_anchor {
                anchor_counts[stratum] += 1;
                anchor_strata.push(stratum);
            } else {
                candidate_counts[stratum] += 1;
                candidate_strata.push(stratum);
            }
        }
    }

    // Subclasses are the strata with both groups. Every one of them has an
    // anchor, so numbering the strata in creation order keeps the order of
    // first appearance among the anchors.
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
        return Err(ExactMatchError::NoSharedStratum);
    }
    let subclasses = build_subclass_summaries(&kept_anchor_counts, &kept_candidate_counts);

    let mut outcome = ExactMatchOutcome {
        subclasses,
        ..ExactMatchOutcome::default()
    };
    for (rows, strata, is_anchor) in [
        (anchors, &anchor_strata, true),
        (candidates, &candidate_strata, false),
    ] {
        for (row, stratum) in rows.iter().zip(strata) {
            let Some(subclass) = subclass_of[*stratum] else {
                outcome.dropped_ids.push(row.id().to_string());
                continue;
            };
            // Both groups are present, so the share is defined.
            let Some(share) = outcome.subclasses[subclass].subclass_propensity else {
                unreachable!("a kept subclass has units");
            };
            let id = row.id().to_string();
            for (weights, estimand) in [
                (&mut outcome.att_weights, Estimand::Att),
                (&mut outcome.atc_weights, Estimand::Atc),
                (&mut outcome.ate_weights, Estimand::Ate),
            ] {
                weights.insert(
                    id.clone(),
                    subclass_weight_for_estimand(estimand, share, is_anchor),
                );
            }
            outcome.assignments.insert(id, subclass);
        }
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    fn record(id: &str, sex: &str, area: &str) -> BaseRecord {
        let mut record = BaseRecord::new(
            id,
            NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid test date"),
        );
        record.strata.insert("sex".into(), sex.into());
        record.strata.insert("area".into(), area.into());
        record
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
        assert_eq!(out.dropped_ids.len(), 0);
        assert_eq!(out.assignments["a1"], 0);
        assert_eq!(out.assignments["a2"], 1);
        assert_eq!(out.assignments["a4"], 2);
        assert_eq!(out.assignments["c5"], 2);
        assert_eq!(out.att_weights["a1"], 1.0);
        assert!((out.att_weights["c1"] - 2.0).abs() < 1e-12);
        assert!((out.att_weights["c3"] - 1.0 / 3.0).abs() < 1e-12);
        assert!((out.ate_weights["a2"] - 4.0).abs() < 1e-12);
        assert!((out.ate_weights["c2"] - 4.0 / 3.0).abs() < 1e-12);
        assert!((out.atc_weights["a1"] - 0.5).abs() < 1e-12);
        assert_eq!(out.atc_weights["c1"], 1.0);
    }

    #[test]
    fn drops_units_in_strata_with_one_group() {
        let anchors = vec![record("a1", "f", "n"), record("a2", "m", "s")];
        let candidates = vec![record("c1", "f", "n"), record("c2", "f", "s")];
        let out = exact_match_subclasses(&anchors, &candidates, &["sex", "area"]).unwrap();
        assert_eq!(out.dropped_ids, vec!["a2", "c2"]);
        assert_eq!(out.retained_units(), 2);
        assert!(!out.att_weights.contains_key("a2"));
    }

    #[test]
    fn refuses_what_cannot_be_stratified() {
        let anchors = vec![record("a1", "f", "n")];
        let candidates = vec![record("c1", "m", "n")];
        assert_eq!(
            exact_match_subclasses::<_, &str>(&anchors, &candidates, &[]).unwrap_err(),
            ExactMatchError::NoVariables
        );
        assert_eq!(
            exact_match_subclasses(&anchors, &candidates, &["sex"]).unwrap_err(),
            ExactMatchError::NoSharedStratum
        );
        assert_eq!(
            exact_match_subclasses(&anchors, &candidates, &["age"]).unwrap_err(),
            ExactMatchError::MissingValue {
                id: "a1".into(),
                variable: "age".into()
            }
        );
    }
}
