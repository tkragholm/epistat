/// The bit pattern that stands for `None` in [`absent`]: a quiet NaN whose
/// payload no arithmetic produces.
const ABSENT_BITS: u64 = 0x7ff8_0000_dead_beef;

/// `value`, or one fixed NaN for `None`.
///
/// For comparing results by bit pattern: the NaN differs in its bits from
/// every number and from the NaN that arithmetic returns, so an output that
/// is `None` on one thread count and `NaN` or a number on another is caught.
#[must_use]
pub fn absent(value: Option<f64>) -> f64 {
    value.unwrap_or_else(|| f64::from_bits(ABSENT_BITS))
}

/// [`absent`] applied to each of `values`.
#[must_use]
pub fn absent_all(values: &[Option<f64>]) -> Vec<f64> {
    values.iter().copied().map(absent).collect()
}

/// `None` as NaN, for comparing against R's `NA` where a NaN is read as
/// undefined. Unlike [`absent`] it does not tell `None` from a computed NaN.
#[must_use]
pub fn none_as_nan(values: &[Option<f64>]) -> Vec<f64> {
    values.iter().map(|v| v.unwrap_or(f64::NAN)).collect()
}

/// The bit patterns of `values`, for exact comparison that also tells NaNs
/// and the signs of zero apart.
#[must_use]
pub fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_the_fixed_nan_and_some_is_unchanged() {
        assert_eq!(absent(None).to_bits(), 0x7ff8_0000_dead_beef);
        assert_eq!(absent(Some(2.5)).to_bits(), 2.5f64.to_bits());
        assert_eq!(absent(Some(-0.0)).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn none_differs_in_bits_from_an_arithmetic_nan() {
        assert_ne!(absent(None).to_bits(), f64::NAN.to_bits());
        assert_ne!(
            absent(None).to_bits(),
            (f64::INFINITY - f64::INFINITY).to_bits()
        );
        assert!(absent(None).is_nan());
    }

    #[test]
    fn a_slice_is_mapped_in_order() {
        let out = absent_all(&[Some(1.0), None, Some(3.0)]);
        assert_eq!(
            bits(&out),
            [1.0f64.to_bits(), 0x7ff8_0000_dead_beef, 3.0f64.to_bits()]
        );
    }

    #[test]
    fn none_as_nan_is_a_plain_nan() {
        let out = none_as_nan(&[Some(1.0), None]);
        assert_eq!(out[0].to_bits(), 1.0f64.to_bits());
        assert!(out[1].is_nan());
    }
}
