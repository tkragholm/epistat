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
//! # The default and its tails
//!
//! The default is a confidence of 0.95, R's `conf.level` default, and its
//! probabilities follow the confidence form. Its upper probability
//! `1 - (1 - 0.95) / 2` equals `1 - 0.05 / 2` to the bit, so a normal or t
//! quantile is the same under either form. Its lower tail
//! `(1 - 0.95) / 2 = 0.025000000000000022` and its alpha
//! `1 - 0.95 = 0.050000000000000044` lie 6 ulps above `0.05 / 2` and `0.05`.
//! A function that reads the lower tail or the alpha, such as a percentile
//! interval or a p-value threshold, therefore gives other last bits under
//! the default than under `Level::alpha(0.05)`. A port of an R function whose
//! argument is `alpha = 0.05` reproduces it to the bit with
//! [`Level::alpha`]; the default is not changed to that form, because the
//! crates' defaults and their recorded results follow the confidence form.

use crate::check::InputError;

/// A confidence level, as the confidence or the alpha the caller gave.
///
/// Construct it with [`Level::confidence`] or [`Level::alpha`], which refuse
/// a number outside `(0, 1)`, or take [`Level::P90`], [`Level::P95`],
/// [`Level::P99`] or [`Level::percent`]. Each probability is computed from the number in
/// the form it was given, so the default, a confidence of 0.95, has a lower
/// tail 6 ulps above that of `Level::alpha(0.05)` (see the module).
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
    } else if value > 1.0 && value < 100.0 {
        Err(InputError::Percentage { what, value })
    } else {
        Err(InputError::Domain {
            what,
            value,
            domain: "(0, 1)",
            row: None,
        })
    }
}

impl Level {
    /// A confidence of 0.90.
    pub const P90: Self = Self(LevelForm::Confidence(0.90));
    /// A confidence of 0.95, the default.
    pub const P95: Self = Self(LevelForm::Confidence(0.95));
    /// A confidence of 0.99.
    pub const P99: Self = Self(LevelForm::Confidence(0.99));

    /// A level given as the confidence `c`, such as 0.95.
    ///
    /// # Errors
    ///
    /// [`InputError::Percentage`] for a `c` between 1 and 100, which reads as
    /// a percentage given where the proportion was meant, and
    /// [`InputError::Domain`] for any other `c` outside `(0, 1)`.
    pub fn confidence(c: f64) -> Result<Self, InputError> {
        open_unit("confidence level", c).map(|c| Self(LevelForm::Confidence(c)))
    }

    /// A level given as the alpha `a`, the probability outside the interval,
    /// such as 0.05.
    ///
    /// # Errors
    ///
    /// [`InputError::Percentage`] for an `a` between 1 and 100 and
    /// [`InputError::Domain`] for any other `a` outside `(0, 1)`.
    pub fn alpha(a: f64) -> Result<Self, InputError> {
        open_unit("alpha", a).map(|a| Self(LevelForm::Alpha(a)))
    }

    /// The confidence `k` percent, as the confidence `k / 100`.
    ///
    /// `k / 100` is the same double as the literal `0.k` for every `k` from 1
    /// to 99, so `Level::percent(95)` equals `Level::confidence(0.95)` to the
    /// bit.
    ///
    /// ```
    /// use epistat_core::Level;
    ///
    /// const LEVEL: Level = Level::percent(90);
    /// assert_eq!(LEVEL, Level::confidence(0.9).unwrap());
    /// assert_eq!(Level::percent(95), Level::P95);
    /// assert_eq!(LEVEL.to_string(), "90%");
    /// ```
    ///
    /// # Panics
    ///
    /// For `k` outside 1 to 99; in a constant, that is a compile error.
    #[must_use]
    #[track_caller]
    pub const fn percent(k: u8) -> Self {
        assert!(k >= 1 && k <= 99, "Level::percent takes 1 to 99");
        Self(LevelForm::Confidence(k as f64 / 100.0))
    }

    /// Whether two levels give the same coverage, whichever form each was
    /// given in.
    ///
    /// Two levels given in the same form are the same when their numbers are
    /// equal. Across forms, `1 - c` and `a` are compared to within
    /// `f64::EPSILON`, the rounding of the subtraction from 1, so
    /// `Level::alpha(0.05)` has the coverage of `Level::confidence(0.95)`
    /// although the two are not equal (see the module).
    #[must_use]
    pub fn same_coverage(self, other: Self) -> bool {
        match (self.0, other.0) {
            (LevelForm::Confidence(a), LevelForm::Confidence(b))
            | (LevelForm::Alpha(a), LevelForm::Alpha(b)) => a.to_bits() == b.to_bits(),
            _ => (self.alpha_value() - other.alpha_value()).abs() <= f64::EPSILON,
        }
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
    /// A confidence of 0.95, [`Level::P95`].
    fn default() -> Self {
        Self::P95
    }
}

impl core::fmt::Display for Level {
    /// The confidence as a percentage, `95%`, whichever form the level was
    /// given in; the percentage is rounded to 12 significant digits, so the
    /// confidence `1 - 0.05` of `Level::alpha(0.05)` prints as `95%`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}%",
            crate::format::Trimmed(self.confidence_value() * 100.0, 12)
        )
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
    fn the_default_tails_are_those_of_the_confidence_form() {
        let default = Level::default();
        let alpha = Level::alpha(0.05).unwrap();
        assert_eq!(
            default.upper_probability().to_bits(),
            alpha.upper_probability().to_bits()
        );
        let ulps = |a: f64, b: f64| a.to_bits().abs_diff(b.to_bits());
        assert_eq!(
            ulps(default.lower_probability(), alpha.lower_probability()),
            6
        );
        assert_eq!(ulps(default.alpha_value(), alpha.alpha_value()), 6);
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
                domain: "(0, 1)",
                row: None
            })
        );
    }

    #[test]
    fn the_named_levels_are_the_literals() {
        assert_eq!(Level::P95, Level::default());
        assert_eq!(Level::P90, Level::confidence(0.9).unwrap());
        assert_eq!(Level::P99, Level::confidence(0.99).unwrap());
        for k in 1..=99_u8 {
            let literal: f64 = format!("0.{k:02}").parse().unwrap();
            assert_eq!(
                Level::percent(k).confidence_value().to_bits(),
                literal.to_bits(),
                "{k}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "Level::percent takes 1 to 99")]
    fn a_percent_of_100_panics() {
        let _ = Level::percent(100);
    }

    #[test]
    fn a_percentage_given_for_a_proportion_names_the_fix() {
        let error = Level::confidence(95.0).unwrap_err();
        assert_eq!(
            error,
            InputError::Percentage {
                what: "confidence level",
                value: 95.0
            }
        );
        assert_eq!(
            error.to_string(),
            "confidence level is 95, outside (0, 1); as a proportion, 95% is 0.95"
        );
        assert!(matches!(
            Level::alpha(5.0),
            Err(InputError::Percentage { what: "alpha", .. })
        ));
        assert!(matches!(
            Level::confidence(100.0),
            Err(InputError::Domain { .. })
        ));
    }

    #[test]
    fn coverage_is_compared_across_forms() {
        let alpha = Level::alpha(0.05).unwrap();
        assert_ne!(alpha, Level::P95);
        assert!(alpha.same_coverage(Level::P95));
        assert!(Level::P95.same_coverage(alpha));
        assert!(!alpha.same_coverage(Level::P90));
        assert!(!Level::P95.same_coverage(Level::P99));
        assert!(Level::alpha(0.1).unwrap().same_coverage(Level::P90));
    }

    #[test]
    fn a_level_prints_as_a_percentage() {
        assert_eq!(Level::P95.to_string(), "95%");
        assert_eq!(Level::alpha(0.05).unwrap().to_string(), "95%");
        assert_eq!(Level::confidence(0.975).unwrap().to_string(), "97.5%");
        assert_eq!(Level::alpha(0.07).unwrap().to_string(), "93%");
        assert_eq!(Level::percent(7).to_string(), "7%");
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
