//! Small helpers of the sensitivity engine: its confidence level, the checks
//! on it, the Wald interval and the conversion of counts.

/// The confidence level the sensitivity engine computes at, a probability in
/// `(0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InferenceConfig {
    /// Confidence level as a probability strictly between 0 and 1, such as
    /// 0.95. The default is 0.95.
    pub confidence_level: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
        }
    }
}

impl InferenceConfig {
    /// A configuration with the given confidence level.
    #[must_use]
    pub const fn new(confidence_level: f64) -> Self {
        Self { confidence_level }
    }
}

/// Whether the confidence level is finite and strictly between 0 and 1.
#[must_use]
pub(crate) fn validate_confidence_level(confidence_level: f64) -> bool {
    confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0
}

/// The Wald interval `estimate -/+ z se` at the given confidence level, `NaN`
/// at both ends when the level is outside (0, 1).
pub(crate) fn normal_interval(estimate: f64, se: f64, confidence_level: f64) -> (f64, f64) {
    let Ok(level) = epistat_core::Level::alpha(1.0 - confidence_level) else {
        return (f64::NAN, f64::NAN);
    };
    let interval = epistat_inference::interval::normal(estimate, se, level);
    (
        interval.lower.unwrap_or(f64::NAN),
        interval.upper.unwrap_or(f64::NAN),
    )
}

/// Convert a count to `f64` through an explicit `u32` bound.
///
/// # Panics
///
/// Panics if `value` exceeds `u32::MAX`.
#[must_use]
pub fn usize_to_f64(value: usize) -> f64 {
    u32::try_from(value).map_or_else(|error| panic!("count exceeds u32::MAX: {error}"), f64::from)
}
