//! The Wald pre-test of parallel trends that `did::att_gt` reports.
//!
//! # Method
//!
//! The pre-treatment cells are the `ATT(g,t)` with `t < g`. Under parallel
//! trends their estimates `b` are jointly zero, and with `V` their covariance
//! matrix from the influence functions the statistic is
//!
//! ```text
//! W = b' V^-1 b  ~  chi-square(q),
//! ```
//!
//! with `q` the number of cells used. `did` forms `V` as
//! `psi' psi / n^2` and, when `att_gt` was clustered and not bootstrapped, from
//! the cluster sums of `psi`; here the covariance is
//! [`did_covariance`](crate::inference::did_clusters::did_covariance), which
//! centres the influence functions. The two agree to rounding for influence
//! functions that average to zero.
//!
//! Cells whose standard error is at most `10 sqrt(eps)` are left out, as `did`
//! leaves out cells with a missing standard error, and so is the base cell of a
//! universal base period, which has no estimate here. `did` returns no
//! statistic when the matrix is singular, judged by its reciprocal condition
//! number against machine epsilon; this function returns
//! [`PretestError::SingularCovariance`] by the same rule, with the 1-norm
//! condition number. The p-value is the chi-square survival function, not
//! rounded to five decimals as `did` rounds it.
//!
//! # References
//!
//! Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time
//! periods. J Econom 2021;225(2):200-30. R package `did` 2.5.1, `att_gt`.

use epistat_inference::wald::wald;
use epistat_linalg::{Attempt, Mat, Method, solve};

use super::aggte::{AggteError, UnitPanel, checked_cluster_index};
use crate::inference::did_clusters::{covariance, standard_error};
use crate::types::AttGtEstimate;

/// The result of [`pretrend_wald_test`].
#[derive(Debug, Clone, PartialEq)]
pub struct PretrendWald {
    /// The Wald statistic.
    pub statistic: f64,
    /// Degrees of freedom, the number of pre-treatment cells used.
    pub df: usize,
    /// Upper tail probability of the statistic under chi-square with `df`
    /// degrees of freedom.
    pub p_value: f64,
    /// The `(group, time)` of the cells used, in input order.
    pub cells: Vec<(i32, i32)>,
}

/// Errors from [`pretrend_wald_test`].
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum PretestError {
    /// The inputs do not fit together.
    #[error(transparent)]
    Input(#[from] AggteError),
    /// No pre-treatment cell has a usable standard error.
    #[error("no pre-treatment cell with a standard error to test")]
    NoPreTreatmentCells,
    /// The covariance matrix is singular to machine precision.
    #[error("the covariance matrix of the pre-treatment cells is singular")]
    SingularCovariance,
    /// The statistic is not finite.
    #[error("the Wald statistic is not finite")]
    NotFinite,
}

/// Tests that the pre-treatment `ATT(g,t)` are jointly zero, as `att_gt` does.
///
/// `influence` and `panel` are as for [`aggregate_att_gt`](super::aggregate_att_gt);
/// set the clusters on `panel` for a clustered covariance.
///
/// # Errors
/// [`PretestError`].
pub fn pretrend_wald_test(
    estimates: &[AttGtEstimate],
    influence: &[Vec<f64>],
    panel: &UnitPanel,
) -> Result<PretrendWald, PretestError> {
    let clusters = checked_cluster_index(estimates, influence, panel)?;
    let floor = 10.0 * f64::EPSILON.sqrt();
    let used = (0..estimates.len())
        .filter(|&i| estimates[i].time < estimates[i].group)
        .filter(|&i| standard_error(&influence[i], &clusters) > floor)
        .collect::<Vec<_>>();
    if used.is_empty() {
        return Err(PretestError::NoPreTreatmentCells);
    }
    let columns = used
        .iter()
        .map(|&i| influence[i].clone())
        .collect::<Vec<_>>();
    if columns.iter().flatten().any(|value| !value.is_finite()) {
        return Err(PretestError::NotFinite);
    }
    let atts = used.iter().map(|&i| estimates[i].att).collect::<Vec<_>>();
    let k = atts.len();
    let cov = covariance(&columns, &clusters);
    let matrix = Mat::from_fn(k, k, |i, j| cov[i * k + j]);
    if reciprocal_condition(&matrix).is_none_or(|rcond| rcond <= f64::EPSILON) {
        return Err(PretestError::SingularCovariance);
    }
    let test = wald(&atts, matrix.as_ref()).map_err(|_| PretestError::NotFinite)?;
    Ok(PretrendWald {
        statistic: test.statistic,
        df: test.df,
        p_value: test.p_value,
        cells: used
            .iter()
            .map(|&i| (estimates[i].group, estimates[i].time))
            .collect(),
    })
}

/// `1 / (|A|_1 |A^-1|_1)`, or `None` when `A` cannot be inverted.
fn reciprocal_condition(matrix: &Mat<f64>) -> Option<f64> {
    let k = matrix.nrows();
    let identity = Mat::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 });
    let inverse = solve(
        matrix.as_ref(),
        identity.as_ref(),
        &[Attempt::plain(Method::FullPivLu)],
    )
    .ok()?
    .x;
    let norm = |m: &Mat<f64>| {
        (0..k)
            .map(|j| (0..k).map(|i| m[(i, j)].abs()).sum::<f64>())
            .fold(0.0, f64::max)
    };
    let condition = norm(matrix) * norm(&inverse);
    (condition.is_finite() && condition > 0.0).then(|| 1.0 / condition)
}
