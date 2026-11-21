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
//! independent, which a match without replacement gives. A unit in more than
//! one set, as after matching with replacement, is an error.
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

#[cfg(test)]
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
    /// A unit in more than one set, as after matching with replacement. The
    /// bounds assume independent sets.
    Reused {
        /// The unit's id.
        id: String,
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
            Self::Reused { id } => write!(
                f,
                "unit {id} has more than one row in the matched sets, so the sets are not independent"
            ),
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
/// controls, [`SetsError::Reused`] for a unit that appears in more than one row
/// with a pair number, in one set or in several, and
/// [`SetsError::Empty`].
pub fn matched_sets(long: &LongMatches, units: &UnitData) -> Result<Vec<MatchedSet>, SetsError> {
    // Each row is read once: its unit's row in `units`, which identifies the
    // unit because `units` has no id twice, marks the unit as used.
    let mut used = vec![false; units.treated_flags().len()];
    let mut rows: Vec<(usize, bool, f64)> = Vec::new();
    for row in 0..long.len() {
        let Some(pair) = long.pair_id[row] else {
            continue;
        };
        let unit = units.row_of(&long.id[row])?;
        if std::mem::replace(&mut used[unit], true) {
            return Err(SetsError::Reused {
                id: long.id[row].clone(),
            });
        }
        let (treated, outcome) = units.treatment_and_outcome_at(unit);
        rows.push((pair, treated, outcome));
    }
    if rows.is_empty() {
        return Err(SetsError::Empty);
    }
    // Stable, so the rows of a set stay in the order of `long`.
    rows.sort_by_key(|&(pair, _, _)| pair);
    let mut sets = Vec::new();
    for group in rows.chunk_by(|left, right| left.0 == right.0) {
        let pair = group[0].0;
        let treated = group.iter().filter(|row| row.1).collect::<Vec<_>>();
        if treated.len() != 1 {
            return Err(SetsError::NotOneTreated {
                pair,
                treated: treated.len(),
            });
        }
        let controls = group
            .iter()
            .filter(|row| !row.1)
            .map(|row| row.2)
            .collect::<Vec<_>>();
        if controls.is_empty() {
            return Err(SetsError::NoControl { pair });
        }
        sets.push(MatchedSet {
            treated: treated[0].2,
            controls,
        });
    }
    Ok(sets)
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

    #[test]
    fn a_unit_twice_in_one_set_is_an_error() {
        let long = LongMatches {
            id: ["a", "b", "b"].map(String::from).to_vec(),
            role: vec![UnitRole::Anchor, UnitRole::Candidate, UnitRole::Candidate],
            pair_id: vec![Some(1), Some(1), Some(1)],
            weight: vec![1.0, 0.5, 0.5],
        };
        assert_eq!(
            matched_sets(&long, &units()),
            Err(SetsError::Reused { id: "b".into() })
        );
    }
    fn matched_sets_by_map(
        long: &LongMatches,
        units: &UnitData,
    ) -> Result<Vec<MatchedSet>, SetsError> {
        let mut by_pair: BTreeMap<usize, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        let mut set_of: BTreeMap<&str, usize> = BTreeMap::new();
        for row in 0..long.len() {
            let Some(pair) = long.pair_id[row] else {
                continue;
            };
            let id = long.id[row].as_str();
            if set_of.insert(id, pair).is_some() {
                return Err(SetsError::Reused { id: id.to_owned() });
            }
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

    /// The sets equal those of the reference that keeps the units in a map and
    /// the pairs in a tree, and so do the errors, on rows with unused pair
    /// numbers, rows without a pair, a unit used twice, a set with two treated
    /// units and a set with no control.
    #[test]
    fn matched_sets_equal_the_reference() {
        let ids: Vec<String> = (0..12).map(|i| format!("u{i}")).collect();
        let treated: Vec<bool> = (0..12).map(|i| i % 3 == 0).collect();
        let outcome: Vec<f64> = (0..12).map(|i| f64::from(i) * 1.5 - 4.0).collect();
        let data = UnitData::new(ids.clone(), treated, outcome).unwrap();
        let row = |id: &str, pair: Option<usize>| (id.to_string(), pair);
        let cases: Vec<Vec<(String, Option<usize>)>> = vec![
            // Pairs and a set of three, numbered with gaps and out of order.
            vec![
                row("u3", Some(7)),
                row("u4", Some(7)),
                row("u0", Some(2)),
                row("u1", Some(2)),
                row("u2", Some(2)),
                row("u6", Some(9)),
                row("u5", Some(9)),
                row("u7", None),
            ],
            // A unit twice.
            vec![row("u0", Some(1)), row("u1", Some(1)), row("u1", Some(2))],
            // Two treated units in one set.
            vec![row("u0", Some(1)), row("u3", Some(1)), row("u1", Some(1))],
            // No control.
            vec![row("u0", Some(1)), row("u3", Some(2)), row("u4", Some(2))],
            // An unknown unit.
            vec![row("u0", Some(1)), row("zz", Some(1))],
            // An unknown unit twice, then a known one.
            vec![row("zz", Some(1)), row("zz", Some(2))],
            // Nothing in a set.
            vec![row("u0", None)],
        ];
        for rows in cases {
            let long = LongMatches {
                id: rows.iter().map(|r| r.0.clone()).collect(),
                role: rows.iter().map(|_| UnitRole::Anchor).collect(),
                pair_id: rows.iter().map(|r| r.1).collect(),
                weight: rows.iter().map(|_| 1.0).collect(),
            };
            assert_eq!(
                format!("{:?}", matched_sets(&long, &data)),
                format!("{:?}", matched_sets_by_map(&long, &data)),
                "{rows:?}"
            );
        }
    }
}
