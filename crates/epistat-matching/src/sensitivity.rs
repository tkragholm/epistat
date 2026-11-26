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
//! [`binary_bounds`] takes sets whose outcomes are 0 or 1 and gives the
//! bounds of the Mantel-Haenszel test, in pairs the binomial tail of the
//! discordant pairs. [`censored_bounds`] takes [`CensoredSet`]s of times to
//! an event or to censoring and scores them by Gehan's comparison.
//!
//! # Sensitivity values, intervals and amplification
//!
//! [`wilcoxon_sensitivity_value`], [`m_statistic_sensitivity_value`],
//! [`binary_sensitivity_value`] and [`censored_sensitivity_value`] give the
//! `gamma` at which the upper bound of the p-value reaches the level, and
//! [`hodges_lehmann_gamma`] the `gamma` at which the lower bound of the
//! estimate reaches a value. [`m_statistic_interval`] gives the range of
//! point estimates and the confidence interval of the effect under a `gamma`.
//! [`amplify`] turns a `gamma` into the pairs of associations with treatment
//! and outcome it stands for.
//!
//! # E-values
//!
//! The E-values need only an estimate and its limits, so they take plain
//! numbers: [`evalues_rr`], [`evalues_or`], [`evalues_hr`] and
//! [`evalues_md`], and [`selection_evalues_rr`] for selection.
//! [`evalues_rd`] takes the table of a risk difference.
//! [`bounding_factor`] gives the most a confounder of two given strengths
//! can move a risk ratio, for benchmarks.
//!
//! # Validation
//!
//! Checked against `rbounds::psens` and `hlsens`, `sensitivitymv::senmv`,
//! `sensitivitymw::senmw` and `EValue::evalues.RR`, `.OR`, `.HR` and `.MD` in
//! `tests/r_sensitivity.rs`, and against `sensitivitymv::amplify` and
//! `separable1v`, `sensitivitymw::senmwCI`, `sensitivity2x2xk::mh` and
//! `mhLS`, the binomial tail and `uniroot` searches for the sensitivity
//! values in `tests/r_rosenbaum_methods.rs`.

#[cfg(test)]
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub use epistat_stats::evalue::{
    EValueError, EValues, Outcome, RdEValues, SelectionBias, bounding_factor, evalues_hr,
    evalues_md, evalues_or, evalues_rd, evalues_rr, selection_evalues_rr,
};
use epistat_stats::rosenbaum::{self, BinaryStratum};
pub use epistat_stats::rosenbaum::{
    BinaryBounds, BinaryDistribution, Censored, EffectInterval, EventDirection,
    HodgesLehmannBounds, MScore, SensitivityError, SetBounds, Sides, WilcoxonBounds,
    WilcoxonDistribution, amplify, sensitivity_value,
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
    /// An outcome other than 0 or 1 where a binary outcome is needed.
    NotBinary(f64),
}

impl Display for BoundsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sets(e) => write!(f, "{e}"),
            Self::Bound(e) => write!(f, "{e}"),
            Self::NotBinary(v) => write!(f, "the outcome {v} is not 0 or 1"),
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
    let scores = epistat_stats::rosenbaum::m_scores(&raw_sets(sets)?, spec)?;
    gammas
        .iter()
        .map(|&g| epistat_stats::rosenbaum::set_bounds(&scores, g).map_err(BoundsError::from))
        .collect()
}

/// The sensitivity value of the signed rank test of matched pairs: the
/// smallest `gamma` at which the upper bound of [`wilcoxon_bounds`] reaches
/// `alpha`; 1 when it does so without hidden bias. See
/// [`epistat_stats::rosenbaum::sensitivity_value`].
///
/// # Errors
///
/// [`BoundsError`].
pub fn wilcoxon_sensitivity_value(
    sets: &[MatchedSet],
    alpha: f64,
    distribution: WilcoxonDistribution,
) -> Result<f64, BoundsError> {
    let diffs = pair_differences(sets)?;
    Ok(rosenbaum::wilcoxon_sensitivity_value(
        &diffs,
        alpha,
        distribution,
    )?)
}

/// The smallest `gamma` at which the lower bound of the Hodges-Lehmann
/// estimate of matched pairs reaches `tau0`, 0 for no effect; 1 when it is
/// there without hidden bias and infinite when no `gamma` brings it there.
/// See [`epistat_stats::rosenbaum::hodges_lehmann_gamma`].
///
/// # Errors
///
/// [`BoundsError`].
pub fn hodges_lehmann_gamma(sets: &[MatchedSet], tau0: f64) -> Result<f64, BoundsError> {
    let diffs = pair_differences(sets)?;
    Ok(rosenbaum::hodges_lehmann_gamma(&diffs, tau0)?)
}

fn raw_sets(sets: &[MatchedSet]) -> Result<Vec<Vec<f64>>, SetsError> {
    if sets.is_empty() {
        return Err(SetsError::Empty);
    }
    Ok(sets
        .iter()
        .map(|s| {
            let mut v = Vec::with_capacity(1 + s.controls.len());
            v.push(s.treated);
            v.extend_from_slice(&s.controls);
            v
        })
        .collect())
}

/// The sensitivity value of the m-statistic over matched sets: the smallest
/// `gamma` at which the upper bound of [`m_statistic_bounds`] reaches
/// `alpha`. The scores are computed once.
///
/// # Errors
///
/// [`BoundsError`].
pub fn m_statistic_sensitivity_value(
    sets: &[MatchedSet],
    alpha: f64,
    spec: &MScore,
) -> Result<f64, BoundsError> {
    let scores = rosenbaum::m_scores(&raw_sets(sets)?, spec)?;
    Ok(rosenbaum::scores_sensitivity_value(&scores, alpha)?)
}

/// The range of point estimates and the confidence interval of an additive
/// effect under `gamma`, from the m-statistic over matched sets, as
/// `sensitivitymw::senmwCI`. See
/// [`epistat_stats::rosenbaum::m_statistic_interval`].
///
/// # Errors
///
/// [`BoundsError`].
pub fn m_statistic_interval(
    sets: &[MatchedSet],
    gamma: f64,
    spec: &MScore,
    alpha: f64,
    sides: Sides,
) -> Result<EffectInterval, BoundsError> {
    Ok(rosenbaum::m_statistic_interval(
        &raw_sets(sets)?,
        gamma,
        spec,
        alpha,
        sides,
    )?)
}

/// The strata of matched sets whose outcomes are 0 or 1.
fn binary_strata(sets: &[MatchedSet]) -> Result<Vec<BinaryStratum>, BoundsError> {
    if sets.is_empty() {
        return Err(SetsError::Empty.into());
    }
    #[expect(clippy::float_cmp, reason = "a binary outcome is exactly 0 or 1")]
    let case = |v: f64| -> Result<bool, BoundsError> {
        if v == 0.0 {
            Ok(false)
        } else if v == 1.0 {
            Ok(true)
        } else {
            Err(BoundsError::NotBinary(v))
        }
    };
    sets.iter()
        .map(|s| {
            if s.controls.is_empty() {
                return Err(SensitivityError::SetTooSmall.into());
            }
            let treated = case(s.treated)?;
            let mut control_cases = 0_u64;
            for &c in &s.controls {
                control_cases += u64::from(case(c)?);
            }
            let controls = s.controls.len() as u64;
            Ok(BinaryStratum {
                treated_cases: u64::from(treated),
                treated_noncases: u64::from(!treated),
                control_cases,
                control_noncases: controls - control_cases,
            })
        })
        .collect()
}

/// The bounds of the one-sided Mantel-Haenszel test over matched sets with a
/// binary outcome, 0 or 1, at each `gamma`, as `sensitivity2x2xk::mh`
/// ([`BinaryDistribution::Exact`]) or `mhLS` ([`BinaryDistribution::Normal`]).
/// In pairs this is the binomial tail of the discordant pairs. See
/// [`epistat_stats::rosenbaum::binary_bounds`].
///
/// # Errors
///
/// [`BoundsError::NotBinary`] for an outcome other than 0 or 1, and
/// [`BoundsError`].
pub fn binary_bounds(
    sets: &[MatchedSet],
    gammas: &[f64],
    distribution: BinaryDistribution,
) -> Result<Vec<BinaryBounds>, BoundsError> {
    let strata = binary_strata(sets)?;
    gammas
        .iter()
        .map(|&g| rosenbaum::binary_bounds(&strata, g, distribution).map_err(BoundsError::from))
        .collect()
}

/// The sensitivity value of the Mantel-Haenszel test over matched sets with
/// a binary outcome.
///
/// # Errors
///
/// As [`binary_bounds`].
pub fn binary_sensitivity_value(
    sets: &[MatchedSet],
    alpha: f64,
    distribution: BinaryDistribution,
) -> Result<f64, BoundsError> {
    let strata = binary_strata(sets)?;
    Ok(rosenbaum::binary_sensitivity_value(
        &strata,
        alpha,
        distribution,
    )?)
}

/// A matched set with a time to an event or to censoring for each unit.
#[derive(Debug, Clone, PartialEq)]
pub struct CensoredSet {
    /// The treated unit.
    pub treated: Censored,
    /// The controls, at least one.
    pub controls: Vec<Censored>,
}

fn gehan(sets: &[CensoredSet], direction: EventDirection) -> Result<Vec<Vec<f64>>, BoundsError> {
    if sets.is_empty() {
        return Err(SetsError::Empty.into());
    }
    let units: Vec<Vec<Censored>> = sets
        .iter()
        .map(|s| {
            let mut v = Vec::with_capacity(1 + s.controls.len());
            v.push(s.treated);
            v.extend_from_slice(&s.controls);
            v
        })
        .collect();
    Ok(rosenbaum::gehan_scores(&units, direction)?)
}

/// The bound of the one-sided test of censored times to an event over
/// matched sets, at each `gamma`: Gehan's scores of
/// [`epistat_stats::rosenbaum::gehan_scores`] summed for the treated units,
/// with the bound of [`epistat_stats::rosenbaum::set_bounds`]. `direction`
/// says whether the test is that the treated have the event earlier or
/// later. The null hypothesis is that the treatment changes neither the
/// times to the event nor the censoring.
///
/// # Errors
///
/// [`BoundsError`].
pub fn censored_bounds(
    sets: &[CensoredSet],
    gammas: &[f64],
    direction: EventDirection,
) -> Result<Vec<SetBounds>, BoundsError> {
    let scores = gehan(sets, direction)?;
    gammas
        .iter()
        .map(|&g| rosenbaum::set_bounds(&scores, g).map_err(BoundsError::from))
        .collect()
}

/// The sensitivity value of the test of [`censored_bounds`].
///
/// # Errors
///
/// [`BoundsError`].
pub fn censored_sensitivity_value(
    sets: &[CensoredSet],
    alpha: f64,
    direction: EventDirection,
) -> Result<f64, BoundsError> {
    let scores = gehan(sets, direction)?;
    Ok(rosenbaum::scores_sensitivity_value(&scores, alpha)?)
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
