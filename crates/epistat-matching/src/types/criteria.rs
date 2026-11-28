//! Matching criteria, their validation, and transition options.

use super::ids::{AgeLimitYears, BirthDateWindowDays, DistanceCaliper, MatchRatio};
use crate::Error;
use serde::{Deserialize, Serialize};
use std::ops::Deref;

pub use epistat_core::Estimand;

/// `estimand` when the matchers can target it: the ATT, the ATC, the ATE or
/// the ATM. A method that targets fewer refuses the others itself.
///
/// # Errors
///
/// [`Error::UnsupportedEstimand`] for any other estimand, such as the ATO,
/// naming `method`.
pub(crate) fn matched_estimand(
    estimand: Estimand,
    method: &'static str,
) -> Result<Estimand, Error> {
    match estimand {
        Estimand::Att | Estimand::Atc | Estimand::Ate | Estimand::Atm => Ok(estimand),
        other => Err(Error::UnsupportedEstimand {
            estimand: other,
            method,
        }),
    }
}

/// The ATT, the default estimand of `matchit()` and of every matcher here.
pub(crate) const fn default_estimand() -> Estimand {
    Estimand::Att
}

/// Common-support trimming policy.
///
/// A unit is trimmed when its score is strictly outside the range of the
/// scores of the other group, so a unit equal to an end of that range is
/// kept.
///
/// In standard matching the other group is all controls for an case and
/// all cases for a control. In risk-set matching a case is trimmed against
/// the scores of the records at risk on its own event date, and a control
/// against the scores of all cases. At risk means the risk-set rule and, when
/// the job has them, being alive and resident on that date. Strata, the birth
/// window and the other matching constraints are not read. A case with nobody
/// at risk is not trimmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommonSupport {
    /// Discard cases outside the overlap region.
    Treated,
    /// Discard controls outside the overlap region.
    Control,
    /// Discard both cases and controls outside overlap.
    Both,
}

/// Matching criteria used by the core engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingCriteria {
    /// Maximum absolute difference in days between record dates.
    pub birth_date_window_days: i32,
    /// Requested number of controls per case.
    pub match_ratio: usize,
    /// Required exact-match strata keys.
    pub required_strata: Vec<String>,
    /// Optional strata key for control uniqueness (fallbacks to `unique_key`).
    pub unique_by_key: Option<String>,
    /// Allow reusing controls across cases.
    pub allow_replacement: bool,
    /// Requested target estimand, the ATT by default. The matchers target
    /// the ATT, the ATC, the ATE and the ATM; [`MatchingCriteria::build`]
    /// refuses any other.
    #[serde(default = "default_estimand")]
    pub estimand: Estimand,
    /// Optional common-support trimming policy.
    #[serde(default)]
    pub common_support: Option<CommonSupport>,
}

impl Default for MatchingCriteria {
    fn default() -> Self {
        Self {
            birth_date_window_days: 30,
            match_ratio: 1,
            required_strata: Vec::new(),
            unique_by_key: None,
            allow_replacement: false,
            estimand: default_estimand(),
            common_support: None,
        }
    }
}

impl MatchingCriteria {
    /// Validate criteria and return an immutable validated wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when one or more criteria values are invalid.
    pub fn validate(&self) -> Result<ValidatedMatchingCriteria, Error> {
        self.clone().build()
    }

    /// Build and validate matching criteria.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when one or more criteria values are invalid.
    pub fn build(self) -> Result<ValidatedMatchingCriteria, Error> {
        if self.typed_birth_date_window().is_none() {
            return Err(Error::NegativeBirthDateWindow);
        }
        if self.match_ratio == 0 {
            return Err(Error::ZeroMatchRatio);
        }
        matched_estimand(self.estimand, "matching")?;
        Ok(ValidatedMatchingCriteria { inner: self })
    }

    /// Typed non-negative birth-date window when criteria is valid.
    #[must_use]
    pub const fn typed_birth_date_window(&self) -> Option<BirthDateWindowDays> {
        BirthDateWindowDays::new(self.birth_date_window_days)
    }

    /// Typed non-zero match ratio when criteria is valid.
    #[must_use]
    pub const fn typed_match_ratio(&self) -> Option<MatchRatio> {
        MatchRatio::new(self.match_ratio)
    }

    /// Typed non-negative day-window caliper when criteria is valid.
    #[must_use]
    pub fn typed_birth_date_caliper(&self) -> Option<DistanceCaliper> {
        self.typed_birth_date_window()
            .and_then(|days| DistanceCaliper::new(f64::from(days.get())))
    }
}

/// Validated matching criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedMatchingCriteria {
    inner: MatchingCriteria,
}

impl ValidatedMatchingCriteria {
    /// Access wrapped criteria.
    #[must_use]
    pub const fn criteria(&self) -> &MatchingCriteria {
        &self.inner
    }

    /// Consume wrapper and return raw criteria.
    #[must_use]
    pub fn into_inner(self) -> MatchingCriteria {
        self.inner
    }
}

impl Deref for ValidatedMatchingCriteria {
    type Target = MatchingCriteria;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl AsRef<MatchingCriteria> for ValidatedMatchingCriteria {
    fn as_ref(&self) -> &MatchingCriteria {
        &self.inner
    }
}

/// Generalized options for transition-based risk-set matching.
///
/// There is no age limit unless one is set: every record with a transition
/// date is a case.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoleTransitionOptions {
    /// A record is a case only if its transition comes before this age in
    /// whole years, `None` for no limit.
    pub transition_age_limit_years: Option<AgeLimitYears>,
    /// Optional descending fallback ratios, for example `[4, 3, 2]`.
    /// When empty, [`MatchingCriteria::match_ratio`] is used.
    pub ratio_fallback: Vec<MatchRatio>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_criteria_validation_rejects_invalid_values() {
        let negative_window = MatchingCriteria {
            birth_date_window_days: -1,
            ..MatchingCriteria::default()
        };
        assert!(matches!(
            negative_window.validate(),
            Err(Error::NegativeBirthDateWindow)
        ));

        let zero_ratio = MatchingCriteria {
            match_ratio: 0,
            ..MatchingCriteria::default()
        };
        assert!(matches!(zero_ratio.validate(), Err(Error::ZeroMatchRatio)));
    }

    #[test]
    fn builder_sets_fields_and_builds_validated_criteria() {
        let validated = MatchingCriteria {
            birth_date_window_days: 10,
            match_ratio: 2,
            required_strata: vec!["municipality".to_string()],
            unique_by_key: Some("family".to_string()),
            allow_replacement: true,
            estimand: Estimand::Ate,
            common_support: Some(CommonSupport::Both),
        }
        .validate()
        .expect("valid criteria");

        assert_eq!(validated.birth_date_window_days, 10);
        assert_eq!(validated.match_ratio, 2);
        assert_eq!(validated.required_strata, vec!["municipality".to_string()]);
        assert_eq!(validated.unique_by_key, Some("family".to_string()));
        assert!(validated.allow_replacement);
        assert_eq!(validated.estimand, Estimand::Ate);
        assert_eq!(validated.common_support, Some(CommonSupport::Both));
    }
}
