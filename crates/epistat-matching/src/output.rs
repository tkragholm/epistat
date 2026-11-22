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
    MatchWeightMethod, PropensityScoreEstimate, SubclassedUnits, SubclassificationOutcome,
    UnitRole, UnitWeightSet,
};
use crate::types::{Estimand, MatchOutcome};
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
    fixed_roles: bool,
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
        Ok(Self {
            ids,
            anchor,
            fixed_roles: true,
        })
    }

    /// The units of a match in which a unit has no fixed group, in the order
    /// of the data.
    ///
    /// Risk-set matching is one, since a future case is a control in the pairs
    /// of earlier cases. The role of a unit is read from each pair. Only
    /// [`get_matches`] takes these units, and it cannot list the anchors that
    /// found no control, since the outcome does not name them.
    ///
    /// # Errors
    ///
    /// [`OutputError::DuplicateId`] when an id occurs twice.
    pub fn without_fixed_roles(ids: Vec<String>) -> Result<Self, OutputError> {
        let anchor = vec![false; ids.len()];
        let mut units = Self::new(ids, anchor)?;
        units.fixed_roles = false;
        Ok(units)
    }

    /// Whether each unit has one group in every pair. False for units made
    /// with [`Units::without_fixed_roles`], whose anchor flags are all false
    /// and mean nothing.
    #[must_use]
    pub const fn has_fixed_roles(&self) -> bool {
        self.fixed_roles
    }

    fn require_fixed_roles(&self, output: &'static str) -> Result<(), OutputError> {
        if self.fixed_roles {
            Ok(())
        } else {
            Err(OutputError::NoFixedRoles { output })
        }
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
#[derive(Debug, Clone, PartialEq)]
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
    /// The units have no fixed group, and the output needs one.
    NoFixedRoles {
        /// The output that was asked for.
        output: &'static str,
    },
    /// A per-unit column is not as long as the list of units.
    ColumnLength {
        /// The column.
        column: &'static str,
        /// Number of units.
        units: usize,
        /// Length of the column.
        length: usize,
    },
    /// Pairs were asked for from a selection that was not paired.
    NotPaired,
    /// A weight is NaN, infinite or negative.
    InvalidWeight {
        /// The id of the unit.
        id: String,
        /// The weight.
        weight: f64,
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
            Self::NoFixedRoles { output } => write!(
                f,
                "{output} needs each unit to be an anchor or a candidate throughout; \
                 these units have a role in each pair"
            ),
            Self::ColumnLength {
                column,
                units,
                length,
            } => write!(
                f,
                "{column} has {length} values for {units} units; give one for every unit"
            ),
            Self::InvalidWeight { id, weight } => write!(
                f,
                "the weight {weight} of {id} is not a finite number of at least 0"
            ),
            Self::NotPaired => {
                f.write_str("the selection was not paired; give mahvars to pair the selected units")
            }
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
    units.require_fixed_roles("match_matrix")?;
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
/// the anchor first. Pairs follow in order of their number. With
/// `include_unmatched`, an anchor without controls follows the pairs with no
/// pair number and weight 0, where MatchIt leaves it out.
///
/// The controls of a pair are in the order MatchIt's `get_matches()` lists
/// them. Going through the anchors in order and each anchor's controls in the
/// order they were assigned, a control takes its place from the first time it
/// is met. Without replacement that is the order of assignment. With
/// replacement a control that an earlier pair used comes before one that is
/// new to this pair.
///
/// An anchor row weighs 1 and a control row its pair weight, the reciprocal of
/// the number of controls its anchor got. These are the weights of
/// `get_matches()` without rescaling. With replacement a control has a row in
/// each pair it is in.
///
/// MatchIt's table puts the treated unit of a pair first, which is the
/// control when the anchors are the controls. Here the anchor is always first.
///
/// With units made by [`Units::without_fixed_roles`], as in risk-set matching,
/// the anchors are the units that are the anchor of some pair, numbered in the
/// order of `units`, and a unit that is a control in one pair can be the
/// anchor of another. The `role` of each row says which it is in that pair.
///
/// # Errors
///
/// [`OutputError::UnknownAnchor`] or [`OutputError::UnknownCandidate`] when a
/// pair names a unit that is not in `units` in that role, and
/// [`OutputError::NoFixedRoles`] when `include_unmatched` is set for units with
/// no fixed roles.
pub fn get_matches(
    outcome: &MatchOutcome,
    units: &Units,
    include_unmatched: bool,
) -> Result<LongMatches, OutputError> {
    if include_unmatched {
        units.require_fixed_roles("get_matches with include_unmatched")?;
    }
    let position = units.positions();
    let pair_weights = outcome.pair_weights(MatchWeightMethod::AnchorUnitCandidateFractional);
    let mut by_anchor: HashMap<&str, Vec<(&str, f64)>> = HashMap::new();
    for entry in &pair_weights.pairs {
        let anchor = entry.pair.anchor_id();
        let control = entry.pair.comparator_id();
        let anchor_known = position
            .get(anchor)
            .is_some_and(|at| !units.fixed_roles || units.anchor[*at]);
        if !anchor_known {
            return Err(OutputError::UnknownAnchor {
                id: anchor.to_string(),
            });
        }
        let control_known = position
            .get(control)
            .is_some_and(|at| !units.fixed_roles || !units.anchor[*at]);
        if !control_known {
            return Err(OutputError::UnknownCandidate {
                id: control.to_string(),
            });
        }
        by_anchor
            .entry(anchor)
            .or_default()
            .push((control, entry.weight));
    }

    let anchors: Vec<&str> = if units.fixed_roles {
        units.anchor_ids()
    } else {
        units
            .ids
            .iter()
            .map(String::as_str)
            .filter(|id| by_anchor.contains_key(id))
            .collect()
    };
    let mut first_met: HashMap<&str, usize> = HashMap::new();
    for anchor in &anchors {
        for (control, _) in by_anchor.get(anchor).into_iter().flatten() {
            let next = first_met.len();
            first_met.entry(control).or_insert(next);
        }
    }

    let mut out = LongMatches {
        id: Vec::new(),
        role: Vec::new(),
        pair_id: Vec::new(),
        weight: Vec::new(),
    };
    let mut next_pair = 0_usize;
    let mut unmatched: Vec<&str> = Vec::new();
    for anchor in anchors {
        let Some(mut controls) = by_anchor.remove(anchor) else {
            unmatched.push(anchor);
            continue;
        };
        next_pair += 1;
        controls.sort_by_key(|(control, _)| first_met[control]);
        out.id.push(anchor.to_string());
        out.role.push(UnitRole::Anchor);
        out.pair_id.push(Some(next_pair));
        out.weight.push(1.0);
        for (control, weight) in controls {
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
    units.require_fixed_roles("match_data")?;
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

fn check_column(column: &'static str, units: &Units, length: usize) -> Result<(), OutputError> {
    if length == units.len() {
        Ok(())
    } else {
        Err(OutputError::ColumnLength {
            column,
            units: units.len(),
            length,
        })
    }
}

/// The matched data of a method that weights units and forms no pairs, or
/// forms them after selecting.
///
/// `weights` and `pair` hold one entry per unit of `units`, in its order. A
/// unit with weight 0 is not matched. `pair` holds labels of any size, and the
/// result numbers the distinct labels from 1 in increasing order, as
/// [`get_matches_from_pairs`] does, so labels from 0 and labels with gaps give
/// the same numbers there and here. The numbers are the `subclass` of
/// MatchIt's `match_data()`. The options are those of [`match_data`], and
/// `replacement` is not read.
///
/// # Errors
///
/// [`OutputError::ColumnLength`] when a column is not as long as `units`,
/// [`OutputError::InvalidWeight`] for a weight that is NaN, infinite or
/// negative, and [`OutputError::NoFixedRoles`] for units with no fixed group.
pub fn match_data_from_weights(
    weights: &[f64],
    pair: &[Option<usize>],
    units: &Units,
    options: MatchDataOptions,
) -> Result<MatchData, OutputError> {
    units.require_fixed_roles("match_data_from_weights")?;
    check_column("weights", units, weights.len())?;
    check_column("pair", units, pair.len())?;
    for (id, weight) in units.ids.iter().zip(weights) {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(OutputError::InvalidWeight {
                id: id.clone(),
                weight: *weight,
            });
        }
    }
    let numbers: std::collections::BTreeMap<usize, usize> = pair
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<usize>>()
        .into_iter()
        .enumerate()
        .map(|(rank, label)| (label, rank + 1))
        .collect();
    let mut set = UnitWeightSet::default();
    for ((id, is_anchor), weight) in units.ids.iter().zip(&units.anchor).zip(weights) {
        if *weight > 0.0 {
            let group = if *is_anchor {
                &mut set.anchor
            } else {
                &mut set.candidate
            };
            group.insert(id.clone(), *weight);
        }
    }
    let set = if options.normalize {
        set.normalized()
    } else {
        set
    };
    let mut out = MatchData {
        id: Vec::new(),
        role: Vec::new(),
        weight: Vec::new(),
        subclass: Vec::new(),
    };
    for ((id, is_anchor), pair) in units.ids.iter().zip(&units.anchor).zip(pair) {
        let group = if *is_anchor {
            &set.anchor
        } else {
            &set.candidate
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
        out.subclass.push(pair.map(|label| numbers[&label]));
    }
    Ok(out)
}

/// The long form of a paired selection, as [`get_matches`] gives it.
///
/// `pair` holds the pair label of each unit of `units`, `None` for a unit that
/// is not selected, and `weights` the weight of each unit. Labels may be of any
/// size and have gaps. Each pair has one row per unit, the anchors first and
/// then the candidates, each in the order of `units`. Pairs follow in
/// increasing order of their label and are numbered from 1, as
/// [`match_data_from_weights`] numbers them. With `include_unmatched`, the anchors that are not
/// selected follow the pairs with no pair number and weight 0.
///
/// # Errors
///
/// [`OutputError::ColumnLength`] when a column is not as long as `units`,
/// [`OutputError::InvalidWeight`] for a selected unit whose weight is NaN,
/// infinite or negative, and [`OutputError::NoFixedRoles`] for units with no
/// fixed group.
pub fn get_matches_from_pairs(
    pair: &[Option<usize>],
    weights: &[f64],
    units: &Units,
    include_unmatched: bool,
) -> Result<LongMatches, OutputError> {
    units.require_fixed_roles("get_matches_from_pairs")?;
    check_column("pair", units, pair.len())?;
    check_column("weights", units, weights.len())?;
    let mut members: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (position, pair) in pair.iter().enumerate() {
        if let Some(pair) = pair {
            let weight = weights[position];
            if !weight.is_finite() || weight < 0.0 {
                return Err(OutputError::InvalidWeight {
                    id: units.ids[position].clone(),
                    weight,
                });
            }
            members.entry(*pair).or_default().push(position);
        }
    }
    let mut out = LongMatches {
        id: Vec::new(),
        role: Vec::new(),
        pair_id: Vec::new(),
        weight: Vec::new(),
    };
    let role_of = |position: usize| {
        if units.anchor[position] {
            UnitRole::Anchor
        } else {
            UnitRole::Candidate
        }
    };
    for (number, positions) in members.values().enumerate() {
        let anchors = positions.iter().filter(|p| units.anchor[**p]);
        let candidates = positions.iter().filter(|p| !units.anchor[**p]);
        for position in anchors.chain(candidates) {
            out.id.push(units.ids[*position].clone());
            out.role.push(role_of(*position));
            out.pair_id.push(Some(number + 1));
            out.weight.push(weights[*position]);
        }
    }
    if include_unmatched {
        for (position, pair) in pair.iter().enumerate() {
            if pair.is_none() && units.anchor[position] {
                out.id.push(units.ids[position].clone());
                out.role.push(UnitRole::Anchor);
                out.pair_id.push(None);
                out.weight.push(0.0);
            }
        }
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

/// Kept units with their values, sorted by id.
fn sorted_pairs<'a, V>(entries: impl Iterator<Item = (&'a str, V)>) -> (Vec<&'a str>, Vec<V>) {
    let mut rows: Vec<(&str, V)> = entries.collect();
    rows.sort_unstable_by(|left, right| left.0.cmp(right.0));
    rows.into_iter().unzip()
}

impl SubclassedUnits {
    /// The subclass of each kept unit, sorted by id.
    #[must_use]
    pub fn assignments_sorted(&self) -> (Vec<&str>, Vec<usize>) {
        sorted_pairs(self.assignments())
    }

    /// The weights of the kept units for an estimand, sorted by id, or `None`
    /// for the matched-sample effect.
    #[must_use]
    pub fn weights_sorted(&self, estimand: Estimand) -> Option<(Vec<&str>, Vec<f64>)> {
        if estimand == Estimand::Atm {
            return None;
        }
        Some(sorted_pairs(self.ids.iter().enumerate().filter_map(
            |(position, id)| {
                self.weight(estimand, position)
                    .map(|weight| (id.as_str(), weight))
            },
        )))
    }

    /// The weights for the effect in the anchors, sorted by id.
    #[must_use]
    pub fn att_weights_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        self.weights_sorted(Estimand::Att).unwrap_or_default()
    }

    /// The weights for the effect in the candidates, sorted by id.
    #[must_use]
    pub fn atc_weights_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        self.weights_sorted(Estimand::Atc).unwrap_or_default()
    }

    /// The weights for the effect in the whole population, sorted by id.
    #[must_use]
    pub fn ate_weights_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        self.weights_sorted(Estimand::Ate).unwrap_or_default()
    }
}

impl SubclassedUnits {
    /// The weights of the kept units for an estimand as per-group maps by id,
    /// the shape the balance functions take, or `None` for the matched-sample
    /// effect. A dropped unit is in neither map.
    #[must_use]
    pub fn unit_weights(&self, estimand: Estimand) -> Option<UnitWeightSet> {
        let weights = self.weights(estimand)?;
        let mut set = UnitWeightSet::default();
        for (position, id) in self.ids.iter().enumerate() {
            if self.subclass[position].is_some() {
                let group = if self.is_anchor(position) {
                    &mut set.anchor
                } else {
                    &mut set.candidate
                };
                group.insert(id.clone(), weights[position]);
            }
        }
        Some(set)
    }

    /// MatchIt's `match_data()` for these units: each unit with its weight for
    /// the estimand and its subclass, in the order of the units. Subclasses are
    /// numbered from 1. `None` for the matched-sample effect.
    ///
    /// # Errors
    ///
    /// [`OutputError::DuplicateId`] when an id occurs twice, and
    /// [`OutputError::InvalidWeight`] for a weight that is not finite and
    /// non-negative.
    pub fn match_data(
        &self,
        estimand: Estimand,
        options: MatchDataOptions,
    ) -> Result<Option<MatchData>, OutputError> {
        let Some(weights) = self.weights(estimand) else {
            return Ok(None);
        };
        let units = Units::new(
            self.ids.clone(),
            (0..self.len()).map(|p| self.is_anchor(p)).collect(),
        )?;
        match_data_from_weights(&weights, &self.subclass, &units, options).map(Some)
    }
}

impl SubclassificationOutcome {
    /// The subclass of each retained unit, sorted by id.
    #[must_use]
    pub fn assignments_sorted(&self) -> (Vec<&str>, Vec<usize>) {
        sorted_pairs(self.assignments())
    }

    /// The weight of each retained unit, sorted by id.
    #[must_use]
    pub fn weights_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_pairs(
            self.ids.iter().enumerate().filter_map(|(position, id)| {
                self.weight(position).map(|weight| (id.as_str(), weight))
            }),
        )
    }
}

impl PropensityScoreEstimate {
    /// The scores matching used, sorted by id.
    #[must_use]
    pub fn distance_scores_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.ids.iter().zip(self.distance_scores.iter()))
    }

    /// The fitted probabilities, sorted by id.
    #[must_use]
    pub fn probabilities_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.ids.iter().zip(self.probabilities.iter()))
    }

    /// The fitted linear predictors, sorted by id.
    #[must_use]
    pub fn linear_predictor_sorted(&self) -> (Vec<&str>, Vec<f64>) {
        sorted_by_key(self.ids.iter().zip(self.linear_predictor.iter()))
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
    fn long_form_orders_pairs_by_anchor_and_controls_by_assignment() {
        let long = get_matches(&outcome(), &units(), false).expect("long");
        assert_eq!(long.id, ["a1", "c3", "c1", "a2", "c2"]);
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
    fn a_reused_control_is_listed_where_it_first_occurs() {
        // a1 took c1 then c3, a2 took c3 then c1. In a2's pair c3 comes first
        // although it was assigned second to a1, since MatchIt lists a control
        // by its first occurrence over all pairs. Here that is c1 before c3.
        let reused = test_outcome(
            vec![
                MatchedPair::new("a1", "c1"),
                MatchedPair::new("a1", "c3"),
                MatchedPair::new("a2", "c3"),
                MatchedPair::new("a2", "c2"),
                MatchedPair::new("a2", "c1"),
            ],
            2,
            0,
        );
        let long = get_matches(&reused, &units(), false).expect("long");
        assert_eq!(long.id, ["a1", "c1", "c3", "a2", "c1", "c3", "c2"]);
        // a2's controls by first occurrence: c1, c3, c2.
        assert_eq!(long.pair_id[3..], [Some(2); 4]);
    }

    #[test]
    fn risk_set_units_have_a_role_in_each_pair() {
        // p1 is a case with p2 and p3 as controls, p2 is a case with p3 and p1
        // as controls. p1 and p2 are controls in the other's pair.
        let ids = ["p3", "p1", "p2", "p4"].map(String::from).to_vec();
        let units = Units::without_fixed_roles(ids).expect("units");
        assert!(!units.has_fixed_roles());
        let outcome = test_outcome(
            vec![
                MatchedPair::new("p2", "p3"),
                MatchedPair::new("p1", "p2"),
                MatchedPair::new("p2", "p1"),
                MatchedPair::new("p1", "p3"),
            ],
            2,
            0,
        );
        let long = get_matches(&outcome, &units, false).expect("long");
        // Pairs follow the order of the units: p1 before p2.
        assert_eq!(long.id, ["p1", "p2", "p3", "p2", "p3", "p1"]);
        assert_eq!(
            long.pair_id,
            [Some(1), Some(1), Some(1), Some(2), Some(2), Some(2)]
        );
        assert_eq!(
            long.role,
            [
                UnitRole::Anchor,
                UnitRole::Candidate,
                UnitRole::Candidate,
                UnitRole::Anchor,
                UnitRole::Candidate,
                UnitRole::Candidate
            ]
        );
        assert_eq!(long.weight, [1.0, 0.5, 0.5, 1.0, 0.5, 0.5]);
        assert!(matches!(
            get_matches(&outcome, &units, true),
            Err(OutputError::NoFixedRoles { .. })
        ));
        assert!(matches!(
            match_matrix(&outcome, &units, 2),
            Err(OutputError::NoFixedRoles { .. })
        ));
        assert!(
            match_data(&outcome, &units, MatchDataOptions::default())
                .expect_err("roles")
                .to_string()
                .contains("a role in each pair")
        );
        let stray = test_outcome(vec![MatchedPair::new("zz", "p1")], 1, 0);
        assert_eq!(
            get_matches(&stray, &units, false),
            Err(OutputError::UnknownAnchor { id: "zz".into() })
        );
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

    #[test]
    fn match_data_from_weights_numbers_labels_from_one_with_gaps_closed() {
        let units = Units::new(
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            vec![true, false, true, false],
        )
        .unwrap();
        let pair = [Some(3), Some(3), Some(7), Some(7)];
        let weights = [1.0; 4];
        let data =
            match_data_from_weights(&weights, &pair, &units, MatchDataOptions::default()).unwrap();
        assert_eq!(data.subclass, [Some(1), Some(1), Some(2), Some(2)]);
        let long = get_matches_from_pairs(&pair, &weights, &units, false).unwrap();
        assert_eq!(long.pair_id, [Some(1), Some(1), Some(2), Some(2)]);
        let from_zero = [Some(0), Some(0), Some(1), Some(1)];
        let data =
            match_data_from_weights(&weights, &from_zero, &units, MatchDataOptions::default())
                .unwrap();
        assert_eq!(data.subclass, [Some(1), Some(1), Some(2), Some(2)]);
    }

    #[test]
    fn invalid_weights_are_errors() {
        let units = Units::new(vec!["a".into(), "b".into()], vec![true, false]).unwrap();
        let pair = [Some(0), Some(0)];
        for bad in [f64::NAN, -1.0, f64::INFINITY] {
            let weights = [1.0, bad];
            assert!(matches!(
                match_data_from_weights(&weights, &pair, &units, MatchDataOptions::default()),
                Err(OutputError::InvalidWeight { .. })
            ));
            assert!(matches!(
                get_matches_from_pairs(&pair, &weights, &units, false),
                Err(OutputError::InvalidWeight { .. })
            ));
        }
        // The weight of a unit that is not selected is not read.
        assert!(get_matches_from_pairs(&[None, None], &[f64::NAN, 1.0], &units, false).is_ok());
    }

    #[test]
    fn subclassed_units_give_match_data_and_weight_maps() {
        let units = SubclassedUnits {
            ids: ["a1", "a2", "a3", "c1", "c2", "c3", "c4"]
                .map(String::from)
                .to_vec(),
            anchor_count: 3,
            // a3 and c4 are dropped.
            subclass: vec![Some(0), Some(0), None, Some(0), Some(0), Some(1), None],
            subclasses: vec![
                crate::matching::SubclassSummary {
                    subclass: 0,
                    anchor_count: 2,
                    candidate_count: 2,
                    subclass_propensity: Some(0.5),
                },
                crate::matching::SubclassSummary {
                    subclass: 1,
                    anchor_count: 1,
                    candidate_count: 1,
                    subclass_propensity: Some(0.5),
                },
            ],
            trimmed: crate::TrimmedUnits::default(),
        };
        let weights = units.unit_weights(Estimand::Att).unwrap();
        assert_eq!(weights.anchor.len(), 2);
        assert_eq!(weights.candidate.len(), 3);
        assert!(units.unit_weights(Estimand::Atm).is_none());
        let options = MatchDataOptions {
            normalize: false,
            ..MatchDataOptions::default()
        };
        let data = units.match_data(Estimand::Att, options).unwrap().unwrap();
        assert_eq!(data.id, ["a1", "a2", "c1", "c2", "c3"]);
        assert_eq!(data.subclass, [Some(1), Some(1), Some(1), Some(1), Some(2)]);
        assert_eq!(data.weight, [1.0, 1.0, 1.0, 1.0, 1.0]);
        assert!(units.match_data(Estimand::Atm, options).unwrap().is_none());
    }
}
