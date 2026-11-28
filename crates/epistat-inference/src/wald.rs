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
//! freedom. `V^-1 b` comes from a full-pivot LU solve. A covariance whose
//! estimated reciprocal condition number is below the machine epsilon is
//! refused, the rule of R's `solve`, since the statistic of such a system has
//! no correct digits: `[[1, 1], [1, 1 + 2^-52]]` would otherwise give a
//! statistic near `10^16` and a p-value of 0. A covariance R accepts is
//! accepted. The p-value is the chi-square survival function, computed
//! directly rather than as one minus the distribution function, so small
//! p-values keep their relative precision.
//!
//! The covariance can be a faer matrix or an `epistat_core::Covariance`, such
//! as an influence function's variance gives ([`CovarianceInput`]):
//!
//! ```
//! use epistat_core::{Covariance, Reference};
//! use epistat_inference::wald::wald;
//!
//! let v = Covariance::from_values(2, &[4.0, 0.0, 0.0, 0.25])?;
//! let test = wald(&[2.0, 1.0], &v)?;
//! assert_eq!(test.distribution, Reference::ChiSquare);
//! assert_eq!(test.df, Some(2.0));
//! assert_eq!(test.to_string(), "chi-square = 5.00 on 2 df, p = 0.082");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use epistat_core::{InputError, Test, check};
use epistat_linalg::{Attempt, Mat, Method, SolveOptions, solve_with};
use epistat_stats::dist::chi_square::sf as chi_square_sf;

use crate::{CovarianceInput, Error, check_length, empty};

/// The result of [`wald`] in epistat-inference 0.3, now the shared
/// [`Test`]: read `test.df` as `Some(k as f64)` and `test.p_value` as an
/// `Option`.
#[deprecated(since = "0.4.0", note = "use epistat_core::Test")]
pub type Wald = Test;

/// Tests that all of `estimates` are zero, given their covariance matrix.
///
/// The result is a chi-square [`Test`] on `k` degrees of freedom, `k` the
/// number of estimates, with its statistic and p-value always defined.
///
/// # Errors
///
/// [`InputError::Empty`] for no estimates, [`InputError::Length`] when the
/// covariance is not `k x k`, [`InputError::NotFinite`] for an entry of the
/// covariance or an estimate that is NaN, infinite or undefined, [`Error::Solve`]
/// with `epistat_linalg::Error::IllConditioned` for a covariance whose
/// reciprocal condition number is below the machine epsilon and with
/// `Singular` for one that cannot be factorised, and [`Error::NotFinite`] for
/// a statistic that is not finite.
pub fn wald(estimates: &[f64], covariance: impl CovarianceInput) -> Result<Test, Error> {
    let k = estimates.len();
    if k == 0 {
        return Err(empty("estimates"));
    }
    check::finite("estimates", estimates)?;
    let covariance = covariance.faer()?;
    check_length("covariance rows", k, covariance.nrows())?;
    check_length("covariance columns", k, covariance.ncols())?;
    for i in 0..k {
        for j in 0..k {
            if !covariance[(i, j)].is_finite() {
                return Err(InputError::NotFinite {
                    what: "covariance",
                    row: i,
                    column: Some(j),
                }
                .into());
            }
        }
    }
    let b = Mat::from_fn(k, 1, |i, _| estimates[i]);
    let options = SolveOptions {
        max_condition: Some(1.0 / f64::EPSILON),
    };
    let solved = solve_with(
        covariance.as_ref(),
        b.as_ref(),
        &[Attempt::plain(Method::FullPivLu)],
        &options,
    )?;
    let statistic: f64 = (0..k).map(|i| estimates[i] * solved.x[(i, 0)]).sum();
    if !statistic.is_finite() {
        return Err(Error::NotFinite);
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "a number of estimates is far below 2^53"
    )]
    let df = k as f64;
    Ok(Test::chi_square(
        Some(statistic),
        df,
        Some(chi_square_sf(statistic, df)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::mat;

    #[test]
    fn a_diagonal_covariance_sums_the_squared_z_scores() {
        let covariance = mat![[4.0, 0.0], [0.0, 0.25]];
        let test = wald(&[2.0, 1.0], covariance.as_ref()).unwrap();
        assert!((test.statistic.unwrap() - 5.0).abs() < 1e-14);
        assert_eq!(test.df, Some(2.0));
        // P(chi-square(2) > 5) = exp(-5 / 2).
        assert!((test.p_value.unwrap() - (-2.5f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn one_estimate_at_the_normal_critical_value() {
        let test = wald(&[1.959_963_984_540_054], mat![[1.0]].as_ref()).unwrap();
        assert!((test.p_value.unwrap() - 0.05).abs() < 1e-14);
    }

    #[test]
    fn a_covariance_below_the_machine_epsilon_in_condition_is_refused() {
        let near = mat![[1.0, 1.0], [1.0, 1.0 + f64::EPSILON]];
        assert!(matches!(
            wald(&[1.0, 2.0], near.as_ref()),
            Err(Error::Solve(epistat_linalg::Error::IllConditioned { .. }))
        ));
        // R's solve accepts a reciprocal condition of about 2.5e-13, and so
        // does the test.
        let poor = mat![[1.0, 1.0], [1.0, 1.0 + 1e-12]];
        assert!(wald(&[1.0, 1.0], poor.as_ref()).is_ok());
    }

    #[test]
    fn a_core_covariance_gives_the_bits_of_the_matrix() {
        let m = mat![[2.0, 0.3], [0.3, 0.5]];
        let c = epistat_core::Covariance::from_values(2, &[2.0, 0.3, 0.3, 0.5]).unwrap();
        assert_eq!(
            wald(&[0.7, -0.2], &c).unwrap(),
            wald(&[0.7, -0.2], m.as_ref()).unwrap()
        );
        let undefined = epistat_core::Covariance::new(1, vec![None]).unwrap();
        assert!(matches!(
            wald(&[1.0], &undefined),
            Err(Error::Input(InputError::NotFinite { .. }))
        ));
        assert!(matches!(
            wald(&[f64::NAN], mat![[1.0]].as_ref()),
            Err(Error::Input(InputError::NotFinite { .. }))
        ));
        assert!(matches!(
            wald(&[1.0], mat![[f64::INFINITY]].as_ref()),
            Err(Error::Input(InputError::NotFinite { .. }))
        ));
    }

    #[test]
    fn errors() {
        assert!(matches!(
            wald(&[], mat![[1.0]].as_ref()),
            Err(Error::Input(InputError::Empty { .. }))
        ));
        assert!(matches!(
            wald(&[1.0, 2.0], mat![[1.0]].as_ref()),
            Err(Error::Input(InputError::Length { .. }))
        ));
    }
}
