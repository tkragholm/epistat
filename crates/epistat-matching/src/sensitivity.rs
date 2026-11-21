//! Sensitivity to hidden bias for matched samples: Rosenbaum bounds and
//! E-values.
//!
//! A matched analysis adjusts for the covariates that were matched on. An
//! unmeasured covariate could still change the odds of treatment within a
//! matched set. The functions here show how strong that change would have to
//! be to alter the conclusion. They work on the output of any matching
//! method that gives sets with one treated unit: [`matched_sets`] reads the
//! pairs and sets of [`crate::output::get_matches`], and [`MatchedSet`]s can
//! be built by hand. The numerical work is in
//! [`epistat_stats::rosenbaum`] and [`epistat_stats::evalue`], which are
//! re-exported here.
//!
//! # Bounds
//!
//! [`wilcoxon_bounds`] and [`hodges_lehmann_bounds`] take matched pairs: sets
//! with exactly one control. [`m_statistic_bounds`] takes sets with any number
//! of controls, and sets of different sizes. All of them assume the sets are
//! independent, which a match without replacement gives. After matching with
//! replacement a control is in several sets and the bounds do not apply as
//! they stand.
//!
//! # E-values
//!
//! The E-values need only an estimate and its limits, so they take plain
//! numbers: [`evalues_rr`], [`evalues_or`], [`evalues_hr`] and
//! [`evalues_md`].
//!
//! # Validation
//!
//! Checked against `rbounds::psens` and `hlsens`, `sensitivitymv::senmv`,
//! `sensitivitymw::senmw` and `EValue::evalues.RR`, `.OR`, `.HR` and `.MD` in
//! `tests/r_sensitivity.rs`.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub use epistat_stats::evalue::{
    EValueError, EValues, Outcome, evalues_hr, evalues_md, evalues_or, evalues_rr,
};
pub use epistat_stats::rosenbaum::{
    HodgesLehmannBounds, MScore, SensitivityError, SetBounds, WilcoxonBounds, WilcoxonDistribution,
};

use crate::effects::{EffectError, UnitData};
use crate::output::LongMatches;

/// A matched set: the outcome of the treated unit and of its controls.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchedSet {
    /// The outcome of the treated unit.
    pub treated: f64,
    /// The outcomes of the controls, at least one.
    pub controls: Vec<f64>,
}

/// What stops matched sets being made from the output of a match.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SetsError {
    /// A unit is not in the data.
    Unit(EffectError),
    /// A set with no treated unit or with several.
    NotOneTreated {
        /// The pair number.
        pair: usize,
        /// Its number of treated units.
        treated: usize,
    },
    /// A set with no control.
    NoControl {
        /// The pair number.
        pair: usize,
    },
    /// A set that is not a pair, where pairs are needed.
    NotAPair {
        /// The pair number.
        pair: usize,
        /// Its number of controls.
        controls: usize,
    },
    /// No sets.
    Empty,
}

impl Display for SetsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unit(e) => write!(f, "{e}"),
            Self::NotOneTreated { pair, treated } => {
                write!(f, "set {pair} has {treated} treated units, not one")
            }
            Self::NoControl { pair } => write!(f, "set {pair} has no control"),
            Self::NotAPair { pair, controls } => {
                write!(f, "set {pair} has {controls} controls, not one")
            }
            Self::Empty => write!(f, "there are no matched sets"),
        }
    }
}

impl Error for SetsError {}

impl From<EffectError> for SetsError {
    fn from(e: EffectError) -> Self {
        Self::Unit(e)
    }
}

/// The matched sets of a match in long form, with the outcomes from `units`.
///
/// Each pair number is a set. Rows with no pair number are left out. The
/// treated unit of a set is the one `units` marks as treated, whichever role
/// it has in the match.
///
/// # Errors
///
/// [`SetsError::Unit`] for a unit not in `units`, [`SetsError::NotOneTreated`]
/// and [`SetsError::NoControl`] for a set that is not one treated unit with
/// controls, and [`SetsError::Empty`].
pub fn matched_sets(long: &LongMatches, units: &UnitData) -> Result<Vec<MatchedSet>, SetsError> {
    let mut by_pair: BTreeMap<usize, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for row in 0..long.len() {
        let Some(pair) = long.pair_id[row] else {
            continue;
        };
        let (treated, outcome) = units.treatment_and_outcome(&long.id[row])?;
        let entry = by_pair.entry(pair).or_default();
        if treated {
            entry.0.push(outcome);
        } else {
            entry.1.push(outcome);
        }
    }
    if by_pair.is_empty() {
        return Err(SetsError::Empty);
    }
    by_pair
        .into_iter()
        .map(|(pair, (treated, controls))| {
            if treated.len() != 1 {
                return Err(SetsError::NotOneTreated {
                    pair,
                    treated: treated.len(),
                });
            }
            if controls.is_empty() {
                return Err(SetsError::NoControl { pair });
            }
            Ok(MatchedSet {
                treated: treated[0],
                controls,
            })
        })
        .collect()
}

/// The treated minus control differences of sets that are pairs.
///
/// # Errors
///
/// [`SetsError::NotAPair`] for a set with other than one control, and
/// [`SetsError::Empty`].
pub fn pair_differences(sets: &[MatchedSet]) -> Result<Vec<f64>, SetsError> {
    if sets.is_empty() {
        return Err(SetsError::Empty);
    }
    sets.iter()
        .enumerate()
        .map(|(i, s)| match s.controls.as_slice() {
            [c] => Ok(s.treated - c),
            other => Err(SetsError::NotAPair {
                pair: i + 1,
                controls: other.len(),
            }),
        })
        .collect()
}

/// What goes wrong in a bound.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BoundsError {
    /// The sets cannot be used.
    Sets(SetsError),
    /// The bound cannot be computed.
    Bound(SensitivityError),
}

impl Display for BoundsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sets(e) => write!(f, "{e}"),
            Self::Bound(e) => write!(f, "{e}"),
        }
    }
}

impl Error for BoundsError {}

impl From<SetsError> for BoundsError {
    fn from(e: SetsError) -> Self {
        Self::Sets(e)
    }
}

impl From<SensitivityError> for BoundsError {
    fn from(e: SensitivityError) -> Self {
        Self::Bound(e)
    }
}

/// The bounds of the signed rank test of matched pairs at each `gamma`, as
/// `rbounds::psens`.
///
/// # Errors
///
/// [`BoundsError`].
pub fn wilcoxon_bounds(
    sets: &[MatchedSet],
    gammas: &[f64],
    distribution: WilcoxonDistribution,
) -> Result<Vec<WilcoxonBounds>, BoundsError> {
    let diffs = pair_differences(sets)?;
    gammas
        .iter()
        .map(|&g| {
            epistat_stats::rosenbaum::wilcoxon_bounds(&diffs, g, distribution)
                .map_err(BoundsError::from)
        })
        .collect()
}

/// The bounds of the Hodges-Lehmann estimate of the effect in matched pairs
/// at each `gamma`, as `rbounds::hlsens`.
///
/// # Errors
///
/// [`BoundsError`].
pub fn hodges_lehmann_bounds(
    sets: &[MatchedSet],
    gammas: &[f64],
) -> Result<Vec<HodgesLehmannBounds>, BoundsError> {
    let diffs = pair_differences(sets)?;
    Ok(epistat_stats::rosenbaum::hodges_lehmann_bounds(
        &diffs, gammas,
    )?)
}

/// The bounds of the m-statistic over matched sets with any number of
/// controls at each `gamma`, as `sensitivitymv::senmv`.
///
/// # Errors
///
/// [`BoundsError`].
pub fn m_statistic_bounds(
    sets: &[MatchedSet],
    gammas: &[f64],
    spec: &MScore,
) -> Result<Vec<SetBounds>, BoundsError> {
    if sets.is_empty() {
        return Err(SetsError::Empty.into());
    }
    let raw: Vec<Vec<f64>> = sets
        .iter()
        .map(|s| {
            let mut v = Vec::with_capacity(1 + s.controls.len());
            v.push(s.treated);
            v.extend_from_slice(&s.controls);
            v
        })
        .collect();
    let scores = epistat_stats::rosenbaum::m_scores(&raw, spec)?;
    gammas
        .iter()
        .map(|&g| epistat_stats::rosenbaum::set_bounds(&scores, g).map_err(BoundsError::from))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnitRole;

    fn units() -> UnitData {
        UnitData::new(
            ["a", "b", "c", "d", "e"].map(String::from).to_vec(),
            vec![true, false, true, false, false],
            vec![5.0, 3.0, 7.0, 4.0, 1.0],
        )
        .unwrap()
    }

    #[test]
    fn sets_are_read_from_the_long_form() {
        let long = LongMatches {
            id: ["a", "b", "c", "d", "e"].map(String::from).to_vec(),
            role: vec![
                UnitRole::Anchor,
                UnitRole::Candidate,
                UnitRole::Anchor,
                UnitRole::Candidate,
                UnitRole::Candidate,
            ],
            pair_id: vec![Some(1), Some(1), Some(2), Some(2), Some(2)],
            weight: vec![1.0, 1.0, 1.0, 0.5, 0.5],
        };
        let sets = matched_sets(&long, &units()).unwrap();
        assert_eq!(sets.len(), 2);
        assert_eq!(sets[0].controls, vec![3.0]);
        assert_eq!(sets[1].controls, vec![4.0, 1.0]);
        assert_eq!(
            pair_differences(&sets),
            Err(SetsError::NotAPair {
                pair: 2,
                controls: 2
            })
        );
    }

    #[test]
    fn a_set_without_one_treated_unit_is_an_error() {
        let long = LongMatches {
            id: ["b", "d"].map(String::from).to_vec(),
            role: vec![UnitRole::Anchor, UnitRole::Candidate],
            pair_id: vec![Some(1), Some(1)],
            weight: vec![1.0, 1.0],
        };
        assert_eq!(
            matched_sets(&long, &units()),
            Err(SetsError::NotOneTreated {
                pair: 1,
                treated: 0
            })
        );
    }
}
