//! Output shapes of `matchit()`, as plain columns.
//!
//! [`match_matrix`], [`get_matches`] and [`match_data`] give the shapes of
//! MatchIt's `match.matrix`, `get_matches()` and `match_data()`. Each is a
//! struct of equal-length vectors in a fixed order, so a binding can turn it
//! into a data frame without looking at a hash map. They are built from a
//! [`MatchOutcome`] and the units the match was run on, in input order, which
//! the outcome does not keep: it holds the pairs and only the count of
//! anchors that found no control.
//!
//! The ordered views at the end of the module give the map-based outputs of
//! the crate in the order of their keys.

use crate::matching::{
    ExactMatchOutcome, MatchWeightMethod, PropensityScoreEstimate, SubclassificationOutcome,
    UnitRole, UnitWeightSet,
};
use crate::types::MatchOutcome;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// The units of a match in input order, with the group of each.
///
/// The anchors are the focal group of the match, the treated units for the
/// effect in the treated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Units {
    ids: Vec<String>,
    anchor: Vec<bool>,
}

impl Units {
    /// The units, in the order of the data, and whether each is an anchor.
    ///
    /// # Errors
    ///
    /// [`OutputError::LengthMismatch`] when the two lists differ in length and
    /// [`OutputError::DuplicateId`] when an id occurs twice.
    pub fn new(ids: Vec<String>, anchor: Vec<bool>) -> Result<Self, OutputError> {
        if ids.len() != anchor.len() {
            return Err(OutputError::LengthMismatch {
                ids: ids.len(),
                anchor: anchor.len(),
            });
        }
        let mut seen: HashSet<&str> = HashSet::with_capacity(ids.len());
        for id in &ids {
            if !seen.insert(id.as_str()) {
                return Err(OutputError::DuplicateId { id: id.clone() });
            }
        }
        Ok(Self { ids, anchor })
    }

    /// The ids in input order.
    #[must_use]
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// Whether each unit is an anchor, in input order.
    #[must_use]
    pub fn anchor_flags(&self) -> &[bool] {
        &self.anchor
    }

    /// Number of units.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether there are no units.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The anchors' ids in input order.
    #[must_use]
    pub fn anchor_ids(&self) -> Vec<&str> {
        self.ids
            .iter()
            .zip(&self.anchor)
            .filter(|(_, is_anchor)| **is_anchor)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    fn positions(&self) -> HashMap<&str, usize> {
        self.ids
            .iter()
            .enumerate()
            .map(|(position, id)| (id.as_str(), position))
            .collect()
    }
}

/// Why an output could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputError {
    /// The id list and the group list differ in length.
    LengthMismatch {
        /// Number of ids.
        ids: usize,
        /// Number of group flags.
        anchor: usize,
    },
    /// An id occurs twice among the units.
    DuplicateId {
        /// The id.
        id: String,
    },
    /// A pair names an anchor that is not among the anchors given.
    UnknownAnchor {
        /// The id.
        id: String,
    },
    /// A pair names a candidate that is not among the candidates given.
    UnknownCandidate {
        /// The id.
        id: String,
    },
}

impl Display for OutputError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { ids, anchor } => write!(
                f,
                "{ids} ids but {anchor} group flags; give one flag for every id"
            ),
            Self::DuplicateId { id } => write!(
                f,
                "the id {id} occurs more than once; the ids of the units must be unique"
            ),
            Self::UnknownAnchor { id } => write!(
                f,
                "a pair has the anchor {id}, which is not an anchor among the units given; \
                 pass the units the match was run on"
            ),
            Self::UnknownCandidate { id } => write!(
                f,
                "a pair has the candidate {id}, which is not a candidate among the units given; \
                 pass the units the match was run on"
            ),
        }
    }
}

impl Error for OutputError {}

/// MatchIt's `match.matrix`: one row per anchor, in input order, holding its
/// controls in the order they were assigned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchMatrix {
    /// The anchor of each row.
    pub anchor_id: Vec<String>,
    /// Column `j` holds the `j`-th control of each anchor, `None` where the
    /// anchor has fewer.
    pub columns: Vec<Vec<Option<String>>>,
}

impl MatchMatrix {
    /// Number of rows, one per anchor.
    #[must_use]
    pub fn nrow(&self) -> usize {
        self.anchor_id.len()
    }

    /// Number of columns.
    #[must_use]
    pub fn ncol(&self) -> usize {
        self.columns.len()
    }

    /// The controls of one row, in order of assignment.
    #[must_use]
    pub fn controls_of_row(&self, row: usize) -> Vec<&str> {
        self.columns
            .iter()
            .filter_map(|column| column[row].as_deref())
            .collect()
    }
}

/// The match matrix of an outcome.
///
/// Every anchor has a row, in the order of `units`. An anchor that found no
/// control has a row of `None`, as `match.matrix` has a row of `NA`. The
/// matrix has `ratio` columns, as MatchIt's has, or as many as the most any
/// anchor got when that is more.
///
/// # Errors
///
/// [`OutputError::UnknownAnchor`] when a pair's anchor is not an anchor in
/// `units`.
pub fn match_matrix(
    outcome: &MatchOutcome,
    units: &Units,
    ratio: usize,
) -> Result<MatchMatrix, OutputError> {
    let anchors = units.anchor_ids();
    let row_of: HashMap<&str, usize> = anchors
        .iter()
        .enumerate()
        .map(|(row, id)| (*id, row))
        .collect();
    let mut controls: Vec<Vec<&str>> = vec![Vec::new(); anchors.len()];
    for pair in &outcome.pairs {
        let row = *row_of
            .get(pair.anchor_id())
            .ok_or_else(|| OutputError::UnknownAnchor {
                id: pair.anchor_id().to_string(),
            })?;
        controls[row].push(pair.comparator_id());
    }
    let width = controls.iter().map(Vec::len).fold(ratio, usize::max);
    let columns = (0..width)
        .map(|column| {
            controls
                .iter()
                .map(|row| row.get(column).map(|id| (*id).to_string()))
                .collect()
        })
        .collect();
    Ok(MatchMatrix {
        anchor_id: anchors.iter().map(|id| (*id).to_string()).collect(),
        columns,
    })
}

/// MatchIt's `get_matches()`: one row per unit per pair.
#[derive(Debug, Clone, PartialEq)]
pub struct LongMatches {
    /// The unit of each row.
    pub id: Vec<String>,
    /// Whether the unit is the anchor or a control of its pair.
    pub role: Vec<UnitRole>,
    /// The pair, numbered from 1 in the order of the anchors. `None` for an
    /// anchor that found no control.
    pub pair_id: Vec<Option<usize>>,
    /// The weight of the unit in the pair.
    pub weight: Vec<f64>,
}

impl LongMatches {
    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.id.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }
}

/// The long form of an outcome.
///
/// Each anchor with controls is a pair, numbered from 1 in the order of
/// `units`. A pair has one row for the anchor and one for each of its controls,
/// the anchor first and the controls in the order of `units`. Pairs follow in
/// order of their number. With `include_unmatched`, an anchor without controls
/// follows the pairs with no pair number and weight 0, where MatchIt leaves it
/// out.
///
/// An anchor row weighs 1 and a control row its pair weight, the reciprocal of
/// the number of controls its anchor got. These are the weights of
/// `get_matches()` without rescaling. With replacement a control has a row in
/// each pair it is in.
///
/// MatchIt's table puts the treated unit of a pair first, which is the
/// control when the anchors are the controls. Here the anchor is always first.
///
/// # Errors
///
/// [`OutputError::UnknownAnchor`] or [`OutputError::UnknownCandidate`] when a
/// pair names a unit that is not in `units` in that role.
pub fn get_matches(
    outcome: &MatchOutcome,
    units: &Units,
    include_unmatched: bool,
) -> Result<LongMatches, OutputError> {
    let position = units.positions();
    let pair_weights = outcome.pair_weights(MatchWeightMethod::AnchorUnitCandidateFractional);
    let mut by_anchor: HashMap<&str, Vec<(usize, &str, f64)>> = HashMap::new();
    for entry in &pair_weights.pairs {
        let anchor = entry.pair.anchor_id();
        let control = entry.pair.comparator_id();
        if !position.get(anchor).is_some_and(|at| units.anchor[*at]) {
            return Err(OutputError::UnknownAnchor {
                id: anchor.to_string(),
            });
        }
        let control_position = position
            .get(control)
            .copied()
            .filter(|at| !units.anchor[*at])
            .ok_or_else(|| OutputError::UnknownCandidate {
                id: control.to_string(),
            })?;
        by_anchor
            .entry(anchor)
            .or_default()
            .push((control_position, control, entry.weight));
    }

    let mut out = LongMatches {
        id: Vec::new(),
        role: Vec::new(),
        pair_id: Vec::new(),
        weight: Vec::new(),
    };
    let mut next_pair = 0_usize;
    let mut unmatched: Vec<&str> = Vec::new();
    for anchor in units.anchor_ids() {
        let Some(mut controls) = by_anchor.remove(anchor) else {
            unmatched.push(anchor);
            continue;
        };
        next_pair += 1;
        controls.sort_by_key(|(position, _, _)| *position);
        out.id.push(anchor.to_string());
        out.role.push(UnitRole::Anchor);
        out.pair_id.push(Some(next_pair));
        out.weight.push(1.0);
        for (_, control, weight) in controls {
            out.id.push(control.to_string());
            out.role.push(UnitRole::Candidate);
            out.pair_id.push(Some(next_pair));
            out.weight.push(weight);
        }
    }
    if include_unmatched {
        for anchor in unmatched {
            out.id.push(anchor.to_string());
            out.role.push(UnitRole::Anchor);
            out.pair_id.push(None);
            out.weight.push(0.0);
        }
    }
    Ok(out)
}

/// What [`match_data`] puts in the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchDataOptions {
    /// The weighting. The default is the weighting of the effect in the
    /// anchors, MatchIt's for `estimand = "ATT"`.
    pub method: MatchWeightMethod,
    /// Rescale the weights of each group to a mean of 1, as `normalize = TRUE`
    /// does in `matchit()`. The default.
    pub normalize: bool,
    /// Leave out the units that are not matched, as `drop.unmatched = TRUE`
    /// does in `match_data()`. The default. When false they stay with
    /// weight 0.
    pub drop_unmatched: bool,
    /// The match was made with replacement, so a control can be in several
    /// pairs and no unit has one subclass. MatchIt gives no subclass then.
    pub replacement: bool,
}

impl Default for MatchDataOptions {
    fn default() -> Self {
        Self {
            method: MatchWeightMethod::AnchorUnitCandidateFractional,
            normalize: true,
            drop_unmatched: true,
            replacement: false,
        }
    }
}

/// MatchIt's `match_data()`: the units with their weight and subclass, in the
/// order of the data.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchData {
    /// The unit of each row.
    pub id: Vec<String>,
    /// Whether the unit is an anchor or a candidate.
    pub role: Vec<UnitRole>,
    /// The weight of the unit.
    pub weight: Vec<f64>,
    /// The pair the unit is in, numbered from 1 in the order of the anchors,
    /// as MatchIt's `subclass`. `None` for a unit with no pair and, with
    /// replacement, for every unit.
    pub subclass: Vec<Option<usize>>,
}

impl MatchData {
    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.id.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }
}

/// The matched data of an outcome.
///
/// The rows are the units of `units` in their order. A unit has a weight when
/// it is in a pair. Pairs are numbered as in [`get_matches`].
///
/// # Errors
///
/// [`OutputError::UnknownAnchor`] or [`OutputError::UnknownCandidate`] when a
/// pair names a unit that is not in `units` in that role.
pub fn match_data(
    outcome: &MatchOutcome,
    units: &Units,
    options: MatchDataOptions,
) -> Result<MatchData, OutputError> {
    let long = get_matches(outcome, units, false)?;
    let mut subclass_of: HashMap<&str, usize> = HashMap::new();
    for (id, pair) in long.id.iter().zip(&long.pair_id) {
        if let Some(pair) = pair {
            subclass_of.entry(id.as_str()).or_insert(*pair);
        }
    }
    let weights = {
        let raw = outcome.match_weights(options.method);
        if options.normalize {
            raw.normalized()
        } else {
            raw
        }
    };
    let mut out = MatchData {
        id: Vec::new(),
        role: Vec::new(),
        weight: Vec::new(),
        subclass: Vec::new(),
    };
    for (id, is_anchor) in units.ids.iter().zip(&units.anchor) {
        let group = if *is_anchor {
            &weights.anchor
        } else {
            &weights.candidate
        };
        let weight = group.get(id).copied();
        if weight.is_none() && options.drop_unmatched {
            continue;
        }
        out.id.push(id.clone());
        out.role.push(if *is_anchor {
            UnitRole::Anchor
        } else {
            UnitRole::Candidate
        });
        out.weight.push(weight.unwrap_or(0.0));
        out.subclass.push(if options.replacement {
            None
        } else {
            subclass_of.get(id.as_str()).copied()
        });
    }
    Ok(out)
}

/// Rows of a map sorted by key, as a pair of vectors.
fn sorted_by_key<'a, V: Copy + 'a>(
    entries: impl Iterator<Item = (&'a String, &'a V)>,
) -> (Vec<&'a str>, Vec<V>) {
    let mut rows: Vec<(&str, V)> = entries.map(|(key, value)| (key.as_str(), *value)).collect();
    rows.sort_unstable_by(|left, right| left.0.cmp(right.0));
    rows.into_iter().unzip()
}

impl UnitWeightSet {
    /// The anchors' weights sorted by id, as parallel vectors.
    #[must_use]
    pub fn anchor_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.anchor.iter())
    }

    /// The candidates' weights sorted by id, as parallel vectors.
    #[must_use]
    pub fn candidate_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.candidate.iter())
    }
}

impl ExactMatchOutcome {
    /// The subclass of each kept unit, sorted by id.
    #[must_use]
    pub fn assignments_sorted(&self) -> (Vec<&str>, Vec<usize>) {
        sorted_by_key(self.assignments.iter())
    }

    /// The weights for the effect in the anchors, sorted by id.
    #[must_use]
    pub fn att_weights_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.att_weights.iter())
    }

    /// The weights for the effect in the candidates, sorted by id.
    #[must_use]
    pub fn atc_weights_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.atc_weights.iter())
    }

    /// The weights for the effect in the whole population, sorted by id.
    #[must_use]
    pub fn ate_weights_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.ate_weights.iter())
    }
}

impl SubclassificationOutcome {
    /// The subclass of each retained unit, sorted by id.
    #[must_use]
    pub fn assignments_sorted(&self) -> (Vec<&str>, Vec<usize>) {
        sorted_by_key(self.assignments.iter())
    }

    /// The weight of each retained unit, sorted by id.
    #[must_use]
    pub fn weights_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.weights.iter())
    }
}

impl PropensityScoreEstimate {
    /// The scores matching used, sorted by id.
    #[must_use]
    pub fn distance_scores_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.distance_scores.iter())
    }

    /// The fitted probabilities, sorted by id.
    #[must_use]
    pub fn probabilities_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.probabilities.iter())
    }

    /// The fitted linear predictors, sorted by id.
    #[must_use]
    pub fn linear_predictor_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.linear_predictor.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_outcome;
    use crate::types::MatchedPair;

    fn units() -> Units {
        // Rows in data order: c1, a1, c2, a2, c3, a3.
        Units::new(
            ["c1", "a1", "c2", "a2", "c3", "a3"]
                .map(String::from)
                .to_vec(),
            vec![false, true, false, true, false, true],
        )
        .expect("valid units")
    }

    fn outcome() -> MatchOutcome {
        // a1 got c3 then c1, a2 got c2, a3 got nobody.
        test_outcome(
            vec![
                MatchedPair::new("a1", "c3"),
                MatchedPair::new("a2", "c2"),
                MatchedPair::new("a1", "c1"),
            ],
            2,
            1,
        )
    }

    #[test]
    fn matrix_has_a_row_per_anchor_in_input_order_with_controls_in_assignment_order() {
        let matrix = match_matrix(&outcome(), &units(), 3).expect("matrix");
        assert_eq!(matrix.anchor_id, ["a1", "a2", "a3"]);
        assert_eq!(matrix.ncol(), 3);
        assert_eq!(matrix.controls_of_row(0), ["c3", "c1"]);
        assert_eq!(matrix.controls_of_row(1), ["c2"]);
        assert_eq!(matrix.controls_of_row(2).len(), 0);
        assert_eq!(matrix.columns[2], [None, None, None]);
        // More controls than the ratio widens the matrix.
        assert_eq!(
            match_matrix(&outcome(), &units(), 1)
                .expect("matrix")
                .ncol(),
            2
        );
    }

    #[test]
    fn long_form_orders_pairs_by_anchor_and_controls_by_input_position() {
        let long = get_matches(&outcome(), &units(), false).expect("long");
        assert_eq!(long.id, ["a1", "c1", "c3", "a2", "c2"]);
        assert_eq!(long.pair_id, [Some(1), Some(1), Some(1), Some(2), Some(2)]);
        assert_eq!(long.weight, [1.0, 0.5, 0.5, 1.0, 1.0]);
        assert_eq!(long.role[0], UnitRole::Anchor);
        assert_eq!(long.role[1], UnitRole::Candidate);

        let with_unmatched = get_matches(&outcome(), &units(), true).expect("long");
        assert_eq!(with_unmatched.len(), 6);
        assert_eq!(with_unmatched.id[5], "a3");
        assert_eq!(with_unmatched.pair_id[5], None);
        assert!((with_unmatched.weight[5]).abs() < f64::EPSILON);
    }

    #[test]
    fn match_data_keeps_the_data_order_and_drops_unmatched_units() {
        let data = match_data(
            &outcome(),
            &units(),
            MatchDataOptions {
                normalize: false,
                ..MatchDataOptions::default()
            },
        )
        .expect("data");
        assert_eq!(data.id, ["c1", "a1", "c2", "a2", "c3"]);
        assert_eq!(data.weight, [0.5, 1.0, 1.0, 1.0, 0.5]);
        assert_eq!(data.subclass, [Some(1), Some(1), Some(2), Some(2), Some(1)]);

        let kept = match_data(
            &outcome(),
            &units(),
            MatchDataOptions {
                drop_unmatched: false,
                replacement: true,
                ..MatchDataOptions::default()
            },
        )
        .expect("data");
        assert_eq!(kept.len(), 6);
        assert!(kept.subclass.iter().all(Option::is_none));
        assert!(kept.weight[5].abs() < f64::EPSILON);
        // Normalized: the anchors' weights have mean 1 over the matched ones.
        assert!((kept.weight[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_pair_outside_the_units_is_an_error() {
        let stray = test_outcome(vec![MatchedPair::new("zz", "c1")], 1, 0);
        assert_eq!(
            match_matrix(&stray, &units(), 1),
            Err(OutputError::UnknownAnchor { id: "zz".into() })
        );
        let stray = test_outcome(vec![MatchedPair::new("a1", "a2")], 1, 0);
        assert_eq!(
            get_matches(&stray, &units(), false),
            Err(OutputError::UnknownCandidate { id: "a2".into() })
        );
        assert!(
            Units::new(vec!["x".into(), "x".into()], vec![true, false])
                .expect_err("duplicate")
                .to_string()
                .contains("more than once")
        );
        assert!(Units::new(vec!["x".into()], vec![]).is_err());
    }

    #[test]
    fn sorted_views_follow_the_key_order_whatever_the_map_order() {
        let mut set = UnitWeightSet::default();
        for (id, weight) in [("b", 2.0), ("c", 3.0), ("a", 1.0), ("d", 4.0)] {
            set.anchor.insert(id.to_string(), weight);
        }
        set.candidate.insert("z".into(), 9.0);
        let (ids, weights) = set.anchor_sorted();
        assert_eq!(ids, ["a", "b", "c", "d"]);
        assert_eq!(weights, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(set.candidate_sorted().0, ["z"]);
    }
}
