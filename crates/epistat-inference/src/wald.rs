//! The joint Wald test that estimates are zero.
//!
//! # Method
//!
//! For `k` estimates `b` with covariance `V`, the statistic is
//!
//! ```text
//! W = b' V^-1 b,
//! ```
//!
//! which under `b = 0` is asymptotically chi-square with `k` degrees of
//! freedom. `V^-1 b` comes from a full-pivot LU solve, so a covariance that is
//! positive semi-definite but nearly singular still gives an answer, and the
//! p-value is the chi-square survival function, computed directly rather than
//! as one minus the distribution function, so small p-values keep their
//! relative precision.

use epistat_linalg::{Attempt, Method, solve};
use epistat_linalg::{Mat, MatRef};
use epistat_stats::gamma::chi_square_sf;

use crate::{InferenceError, check_length};

/// The result of [`wald`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wald {
    pub statistic: f64,
    /// The number of estimates.
    pub df: usize,
    pub p_value: f64,
}

/// Tests that all of `estimates` are zero, given their covariance matrix.
///
/// # Errors
///
/// [`InferenceError::Empty`] for no estimates, [`InferenceError::Length`] when
/// the covariance is not `k x k`, [`InferenceError::Solve`] when the
/// covariance cannot be solved, and [`InferenceError::NotFinite`] for a
/// statistic that is not finite.
pub fn wald(estimates: &[f64], covariance: MatRef<'_, f64>) -> Result<Wald, InferenceError> {
    let k = estimates.len();
    if k == 0 {
        return Err(InferenceError::Empty);
    }
    check_length("covariance rows", k, covariance.nrows())?;
    check_length("covariance columns", k, covariance.ncols())?;
    let b = Mat::from_fn(k, 1, |i, _| estimates[i]);
    let solved = solve(covariance, b.as_ref(), &[Attempt::plain(Method::FullPivLu)])?;
    let statistic: f64 = (0..k).map(|i| estimates[i] * solved.x[(i, 0)]).sum();
    if !statistic.is_finite() {
        return Err(InferenceError::NotFinite);
    }
    Ok(Wald {
        statistic,
        df: k,
        p_value: chi_square_sf(statistic, k as f64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::mat;

    #[test]
    fn a_diagonal_covariance_sums_the_squared_z_scores() {
        let covariance = mat![[4.0, 0.0], [0.0, 0.25]];
        let test = wald(&[2.0, 1.0], covariance.as_ref()).unwrap();
        assert!((test.statistic - 5.0).abs() < 1e-14);
        assert_eq!(test.df, 2);
        // P(chi-square(2) > 5) = exp(-5 / 2).
        assert!((test.p_value - (-2.5f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn one_estimate_at_the_normal_critical_value() {
        let test = wald(&[1.959_963_984_540_054], mat![[1.0]].as_ref()).unwrap();
        assert!((test.p_value - 0.05).abs() < 1e-14);
    }

    #[test]
    fn errors() {
        assert_eq!(wald(&[], mat![[1.0]].as_ref()), Err(InferenceError::Empty));
        assert!(matches!(
            wald(&[1.0, 2.0], mat![[1.0]].as_ref()),
            Err(InferenceError::Length { .. })
        ));
    }
}
