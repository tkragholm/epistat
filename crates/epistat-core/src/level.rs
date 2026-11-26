//! A confidence level, kept in the form the caller gave it.
//!
//! An interval at confidence `c` has probability `alpha = 1 - c` outside it,
//! `alpha / 2` in each tail. The two numbers are not interchangeable in
//! floating point: `1.0 - 0.95` is `0.050000000000000044`, not `0.05`, so a
//! function that converted one form into the other would move a type 7
//! percentile limit or a normal quantile in its last bits. [`Level`] keeps
//! the number and the form the caller gave and computes each probability
//! from that number, so no conversion happens unless a function asks for the
//! other form.
//!
//! # Formulas
//!
//! For a level given as a confidence `c`:
//!
//! ```text
//! confidence = c,    alpha = 1 - c,
//! lower tail = (1 - c) / 2,    upper probability = 1 - (1 - c) / 2,
//! ```
//!
//! and given as an alpha `a`:
//!
//! ```text
//! confidence = 1 - a,    alpha = a,
//! lower tail = a / 2,    upper probability = 1 - a / 2,
//! ```
//!
//! each evaluated in that order of operations. The upper probability is the
//! argument of the quantile in `estimate +- q(upper) se`: the expression
//! epistat-stats uses for a confidence (`1 - (1 - level) / 2`) and
//! epistat-inference uses for an alpha (`1 - alpha / 2`).
//!
//! The default is a confidence of 0.95, R's `conf.level` default.

use crate::check::InputError;

/// A confidence level, as the confidence or the alpha the caller gave.
///
/// Construct it with [`Level::confidence`] or [`Level::alpha`]; both refuse a
/// number outside `(0, 1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "LevelForm", into = "LevelForm"))]
pub struct Level(LevelForm);

/// The two forms a level is given in; serialised as `{"confidence": c}` or
/// `{"alpha": a}`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
enum LevelForm {
    Confidence(f64),
    Alpha(f64),
}

impl TryFrom<LevelForm> for Level {
    type Error = InputError;

    fn try_from(form: LevelForm) -> Result<Self, InputError> {
        match form {
            LevelForm::Confidence(c) => Self::confidence(c),
            LevelForm::Alpha(a) => Self::alpha(a),
        }
    }
}

impl From<Level> for LevelForm {
    fn from(level: Level) -> Self {
        level.0
    }
}

fn open_unit(what: &'static str, value: f64) -> Result<f64, InputError> {
    if value > 0.0 && value < 1.0 {
        Ok(value)
    } else {
        Err(InputError::Domain {
            what,
            value,
            domain: "(0, 1)",
        })
    }
}

impl Level {
    /// A level given as the confidence `c`, such as 0.95.
    ///
    /// # Errors
    ///
    /// [`InputError::Domain`] unless `0 < c < 1`.
    pub fn confidence(c: f64) -> Result<Self, InputError> {
        open_unit("confidence level", c).map(|c| Self(LevelForm::Confidence(c)))
    }

    /// A level given as the alpha `a`, the probability outside the interval,
    /// such as 0.05.
    ///
    /// # Errors
    ///
    /// [`InputError::Domain`] unless `0 < a < 1`.
    pub fn alpha(a: f64) -> Result<Self, InputError> {
        open_unit("alpha", a).map(|a| Self(LevelForm::Alpha(a)))
    }

    /// Whether the level was given as an alpha.
    #[must_use]
    pub const fn is_alpha(self) -> bool {
        matches!(self.0, LevelForm::Alpha(_))
    }

    /// The confidence: the number given, or `1 - a` for an alpha.
    #[must_use]
    pub fn confidence_value(self) -> f64 {
        match self.0 {
            LevelForm::Confidence(c) => c,
            LevelForm::Alpha(a) => 1.0 - a,
        }
    }

    /// The alpha: the number given, or `1 - c` for a confidence.
    #[must_use]
    pub fn alpha_value(self) -> f64 {
        match self.0 {
            LevelForm::Confidence(c) => 1.0 - c,
            LevelForm::Alpha(a) => a,
        }
    }

    /// The probability in the lower tail: `(1 - c) / 2` or `a / 2`.
    #[must_use]
    pub fn lower_probability(self) -> f64 {
        match self.0 {
            LevelForm::Confidence(c) => (1.0 - c) / 2.0,
            LevelForm::Alpha(a) => a / 2.0,
        }
    }

    /// The probability below the upper limit: `1 - (1 - c) / 2` or
    /// `1 - a / 2`.
    #[must_use]
    pub fn upper_probability(self) -> f64 {
        match self.0 {
            LevelForm::Confidence(c) => 1.0 - (1.0 - c) / 2.0,
            LevelForm::Alpha(a) => 1.0 - a / 2.0,
        }
    }
}

impl Default for Level {
    /// A confidence of 0.95.
    fn default() -> Self {
        Self(LevelForm::Confidence(0.95))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_a_confidence_of_95_percent() {
        let level = Level::default();
        assert!(!level.is_alpha());
        assert_eq!(level.confidence_value().to_bits(), 0.95_f64.to_bits());
        assert_eq!(level, Level::confidence(0.95).unwrap());
    }

    #[test]
    fn the_given_number_is_kept_and_the_other_form_computed() {
        let alpha = Level::alpha(0.05).unwrap();
        assert!(alpha.is_alpha());
        assert_eq!(alpha.alpha_value().to_bits(), 0.05_f64.to_bits());
        assert_eq!(
            alpha.confidence_value().to_bits(),
            (1.0_f64 - 0.05).to_bits()
        );
        let confidence = Level::confidence(0.95).unwrap();
        // 1 - 0.95 is not 0.05, which is why the form is kept.
        assert_eq!(
            confidence.alpha_value().to_bits(),
            (1.0_f64 - 0.95).to_bits()
        );
        assert_ne!(confidence.alpha_value().to_bits(), 0.05_f64.to_bits());
        assert_ne!(alpha, confidence);
    }

    #[test]
    fn tail_probabilities_follow_the_expressions_the_crates_use() {
        let a = 0.05_f64;
        let alpha = Level::alpha(a).unwrap();
        assert_eq!(
            alpha.upper_probability().to_bits(),
            (1.0 - a / 2.0).to_bits()
        );
        assert_eq!(alpha.lower_probability().to_bits(), (a / 2.0).to_bits());
        let c = 0.9_f64;
        let confidence = Level::confidence(c).unwrap();
        assert_eq!(
            confidence.upper_probability().to_bits(),
            (1.0 - (1.0 - c) / 2.0).to_bits()
        );
        assert_eq!(
            confidence.lower_probability().to_bits(),
            ((1.0 - c) / 2.0).to_bits()
        );
    }

    #[test]
    fn numbers_outside_the_open_unit_interval_are_refused() {
        for bad in [0.0, 1.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(Level::confidence(bad).is_err(), "{bad}");
            assert!(Level::alpha(bad).is_err(), "{bad}");
        }
        assert_eq!(
            Level::alpha(1.0),
            Err(InputError::Domain {
                what: "alpha",
                value: 1.0,
                domain: "(0, 1)"
            })
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_keeps_the_form_and_validates() {
        let alpha = Level::alpha(0.1).unwrap();
        let text = serde_json::to_string(&alpha).unwrap();
        assert_eq!(text, r#"{"alpha":0.1}"#);
        assert_eq!(serde_json::from_str::<Level>(&text).unwrap(), alpha);
        let confidence: Level = serde_json::from_str(r#"{"confidence":0.9}"#).unwrap();
        assert_eq!(confidence, Level::confidence(0.9).unwrap());
        assert!(serde_json::from_str::<Level>(r#"{"alpha":1.5}"#).is_err());
    }
}
