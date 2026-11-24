//! `did::att_gt`'s overlap check of a cell.
//!
//! Before it fits a doubly robust or inverse probability weighted cell, `did`
//! 2.5.1 (`overlap_check_fail`) fits an unweighted logistic regression of the
//! arm on the cell's design (`fastglmPure`, tolerance `1e-8`, at most 100
//! iterations) and sets the cell to NA when the largest fitted value is
//! `0.999` or more. Without covariates it uses the treated share instead,
//! unless that share is within `1e-6` of the threshold, where it fits the
//! intercept-only model.

use epistat_linalg::MatRef;

use crate::estimators::common::logit::fit_logit;

/// The threshold of the check.
pub(super) const OVERLAP_THRESHOLD: f64 = 0.999;

/// The largest fitted propensity score of the check, on a row-major design
/// with `feature_count` columns, intercept first. `None` when the logistic
/// fit fails.
pub(super) fn max_propensity(design: &[f64], feature_count: usize, treated: &[f64]) -> Option<f64> {
    let n = treated.len();
    if n == 0 {
        return None;
    }
    if feature_count == 1 {
        let share = treated.iter().sum::<f64>() / crate::util::usize_to_f64(n);
        if (share - OVERLAP_THRESHOLD).abs() > 1e-6 {
            return Some(share);
        }
    }
    let matrix = MatRef::from_row_major_slice(design, n, feature_count);
    let fit = fit_logit(matrix, treated, None, 1e-8, 100).ok()?;
    (0..n)
        .map(|row| {
            let x = &design[row * feature_count..(row + 1) * feature_count];
            let eta = x
                .iter()
                .zip(&fit.coefficients)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            1.0 / (1.0 + (-eta).exp())
        })
        .reduce(f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_covariates_the_score_is_the_treated_share() {
        let treated = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(max_propensity(&[1.0; 4], 1, &treated), Some(0.5));
    }

    #[test]
    fn a_covariate_that_nearly_separates_the_arms_trips_the_check() {
        let x = [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
        let treated = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let design: Vec<f64> = x.iter().flat_map(|&v| [1.0, v]).collect();
        let max = max_propensity(&design, 2, &treated).expect("fit");
        assert!(max > 0.99 && max < 1.0, "{max}");
    }
}
