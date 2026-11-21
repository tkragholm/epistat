//! Exposure density sampling.
//!
//! For each person who becomes exposed at time t, controls are drawn from the
//! people who are unexposed and still at risk at t, matched on time-fixed
//! covariates (Ohneberg, Beyersmann and Schumacher 2019). A person drawn as a
//! control stays in the cohort. If the person is exposed later, the person is
//! an exposed person in their own right, with their own controls, and an
//! analysis stops following them as a control on the day of that exposure.
//!
//! The sampling is risk-set matching: the exposure date is the transition
//! date, and the risk set of an exposed person is the people with no exposure
//! on or before the index date who are in the cohort and, when the job has the
//! alive and resident checks, alive and resident on it. This module adds the
//! output an analysis needs, which risk-set matching does not keep: the index date of each set and the date on which each control
//! is later exposed.
//!
//! Drawing with replacement gives the sampling of the paper, in which the sets
//! are drawn independently and one person may be a control in several of them.
//! Without replacement each person is a control at most once.

use super::{TransitionMatchRequest, match_transition};
use crate::matching::{ConstraintGroup, RoleIndexedRecord, SelectionStrategy};
use crate::types::MatchOutcome;
use chrono::NaiveDate;
use rapidhash::RapidHashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use super::RiskSetPolicy;

/// An exposed person and the date they became exposed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExposedPerson {
    /// The exposed person.
    pub id: String,
    /// The day the person became exposed. The controls of the set are
    /// unexposed on this day.
    pub index_date: NaiveDate,
    /// How many controls the set has.
    pub n_controls: usize,
}

/// A control in the set of one exposed person.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExposureDensityControl {
    /// The exposed person whose set the control is in.
    pub exposed_id: String,
    /// The control.
    pub control_id: String,
    /// The index date of the set, which is the control's start of follow-up.
    pub index_date: NaiveDate,
    /// The day the control becomes exposed, if ever. After the index date
    /// under the default risk-set policy, which admits a control only if its
    /// exposure is later than the index date. A policy of the caller's may
    /// admit others. An analysis censors the control from this day.
    pub exposure_date: Option<NaiveDate>,
}

impl ExposureDensityControl {
    /// Days from the index date to the control's own exposure, if any.
    #[must_use]
    pub fn days_to_exposure(&self) -> Option<i64> {
        self.exposure_date
            .map(|date| (date - self.index_date).num_days())
    }
}

/// The sets of exposure density sampling.
#[derive(Debug, Clone)]
pub struct ExposureDensityOutcome {
    /// The pairs and diagnostics of the underlying risk-set matching.
    pub outcome: MatchOutcome,
    /// The exposed people that have at least one control, in the order of
    /// their index dates, then ids.
    pub exposed: Vec<ExposedPerson>,
    /// The controls, in the order of the pairs.
    pub controls: Vec<ExposureDensityControl>,
}

/// Why the sets could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExposureDensityError {
    /// Two units share an id, so a pair does not say which unit it means.
    DuplicateId {
        /// The id that occurs twice.
        id: String,
    },
    /// A pair names a unit that is not in the cohort, or a unit without an
    /// exposure date as the exposed person.
    InconsistentPair {
        /// The exposed id of the pair.
        exposed_id: String,
        /// The control id of the pair.
        control_id: String,
    },
}

impl Display for ExposureDensityError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId { id } => write!(
                f,
                "the id `{id}` occurs more than once in the cohort; give every unit its own id"
            ),
            Self::InconsistentPair {
                exposed_id,
                control_id,
            } => write!(
                f,
                "the pair `{exposed_id}`, `{control_id}` does not match the cohort"
            ),
        }
    }
}

impl Error for ExposureDensityError {}

/// The index and exposure dates of the pairs of a risk-set match.
///
/// # Errors
///
/// [`ExposureDensityError::DuplicateId`] when two units share an id.
/// [`ExposureDensityError::InconsistentPair`] when a pair names a unit that is
/// not in the cohort, or an exposed person without an exposure date.
pub fn exposure_density_sets<R: RoleIndexedRecord>(
    cohort: &[R],
    outcome: MatchOutcome,
) -> Result<ExposureDensityOutcome, ExposureDensityError> {
    let mut position: RapidHashMap<&str, usize> = RapidHashMap::default();
    for (idx, unit) in cohort.iter().enumerate() {
        if position.insert(unit.id(), idx).is_some() {
            return Err(ExposureDensityError::DuplicateId {
                id: unit.id().to_string(),
            });
        }
    }
    let mut controls = Vec::with_capacity(outcome.pairs.len());
    let mut counts: RapidHashMap<&str, usize> = RapidHashMap::default();
    for pair in &outcome.pairs {
        let inconsistent = || ExposureDensityError::InconsistentPair {
            exposed_id: pair.anchor_id().to_string(),
            control_id: pair.comparator_id().to_string(),
        };
        let exposed = position
            .get(pair.anchor_id())
            .and_then(|idx| cohort[*idx].event_date())
            .ok_or_else(inconsistent)?;
        let control = position
            .get(pair.comparator_id())
            .map(|idx| &cohort[*idx])
            .ok_or_else(inconsistent)?;
        controls.push(ExposureDensityControl {
            exposed_id: pair.anchor_id().to_string(),
            control_id: pair.comparator_id().to_string(),
            index_date: exposed,
            exposure_date: control.event_date(),
        });
        *counts.entry(pair.anchor_id()).or_insert(0) += 1;
    }
    let mut exposed: Vec<ExposedPerson> = counts
        .into_iter()
        .filter_map(|(id, n_controls)| {
            let index_date = cohort[position[id]].event_date()?;
            Some(ExposedPerson {
                id: id.to_string(),
                index_date,
                n_controls,
            })
        })
        .collect();
    exposed.sort_by(|a, b| {
        a.index_date
            .cmp(&b.index_date)
            .then_with(|| a.id.cmp(&b.id))
    });
    Ok(ExposureDensityOutcome {
        outcome,
        exposed,
        controls,
    })
}

/// Exposure density sampling of a cohort.
///
/// Runs risk-set matching with the exposure date as the transition date and
/// returns the sets with their index dates and the controls' later exposure
/// dates.
///
/// # Errors
///
/// As [`exposure_density_sets`].
pub fn match_exposure_density<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R>,
>(
    cohort: &[R],
    request: TransitionMatchRequest<'_, S, G, P>,
) -> Result<ExposureDensityOutcome, ExposureDensityError> {
    exposure_density_sets(cohort, match_transition(cohort, request))
}
