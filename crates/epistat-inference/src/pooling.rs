//! Rubin's rules for pooling an estimate over multiply imputed data sets.
//!
//! [`rubin`] pools the `m` estimates `Q_i` of one quantity and their
//! variances `U_i`, one pair per imputed data set, as Rubin \[1\] describes.
//! With
//!
//! ```text
//! Qbar = mean(Q_i)                      the pooled estimate
//! Ubar = mean(U_i)                      the within-imputation variance
//! B    = sum (Q_i - Qbar)^2 / (m - 1)   the between-imputation variance
//! T    = Ubar + (1 + 1/m) B             the total variance
//! ```
//!
//! the standard error of the pooled estimate is `sqrt(T)`. The statistics
//! reported with it are
//!
//! ```text
//! riv    = (1 + 1/m) B / Ubar           the relative increase in variance
//! lambda = (1 + 1/m) B / T              the proportion of variance due to
//!                                       the missing data
//! fmi    = (riv + 2/(df + 3)) / (riv + 1)  the fraction of missing information
//! ```
//!
//! # Degrees of freedom
//!
//! The reference distribution of `(Qbar - Q) / sqrt(T)` is a t distribution
//! with `df` degrees of freedom. Rubin's own value is `(m - 1) / lambda^2`.
//! Barnard and Rubin \[2\] correct it for a complete-data analysis on a small
//! sample, whose degrees of freedom `dfcom` the caller gives:
//!
//! ```text
//! df_old = (m - 1) / lambda^2
//! df_obs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lambda)
//! df     = df_old * df_obs / (df_old + df_obs)
//! ```
//!
//! With `dfcom = None`, the complete-data degrees of freedom are infinite and
//! `df = df_old`. With `B = 0` and so `lambda = 0`, `df_old` is infinite and
//! `df = df_obs`, or infinite when `dfcom` is. `mice::pool()` raises a
//! `lambda` below `1e-4` to `1e-4`, which turns an infinite `df_old` into a
//! very large one, so its `df` differs from this one by less than a part in
//! a million in that case and is finite where this one is infinite.
//!
//! An undefined statistic is `None`: `riv` when `Ubar` is zero, and `lambda`,
//! `fmi` and `df` when `T` is zero. An infinite `df` is `f64::INFINITY`.
//!
//! The pooling is checked against `mice::pool()` and `mice::pool.scalar()`
//! 3.19.0.
//!
//! # References
//!
//! 1. Rubin DB. Multiple Imputation for Nonresponse in Surveys. New York:
//!    Wiley; 1987.
//! 2. Barnard J, Rubin DB. Small-sample degrees of freedom with multiple
//!    imputation. Biometrika. 1999;86(4):948-55.

use epistat_stats::student::t_quantile;

use crate::{InferenceError, check_length};

/// An estimate pooled over imputed data sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pooled {
    /// The number of imputed data sets.
    pub m: usize,
    /// The pooled estimate.
    pub estimate: f64,
    /// The within-imputation variance.
    pub within: f64,
    /// The between-imputation variance.
    pub between: f64,
    /// The total variance.
    pub total: f64,
    /// The relative increase in variance, `None` when the within variance is
    /// zero.
    pub riv: Option<f64>,
    /// The proportion of the total variance due to the missing data, `None`
    /// when the total variance is zero.
    pub lambda: Option<f64>,
    /// The fraction of missing information, `None` when `lambda` is.
    pub fmi: Option<f64>,
    /// The degrees of freedom of the t reference distribution, `None` when
    /// the total variance is zero.
    pub df: Option<f64>,
}

impl Pooled {
    /// The standard error of the pooled estimate, `None` when the total
    /// variance is zero.
    #[must_use]
    pub fn se(&self) -> Option<f64> {
        (self.total > 0.0).then(|| self.total.sqrt())
    }

    /// The confidence interval `estimate -/+ t * se` at the level
    /// `1 - alpha`, with `t` the quantile of the t distribution at
    /// `1 - alpha / 2` on `df` degrees of freedom. `None` when the total
    /// variance is zero.
    ///
    /// # Errors
    ///
    /// [`InferenceError::Alpha`] when `alpha` is not in `(0, 1)`.
    pub fn interval(&self, alpha: f64) -> Result<Option<(f64, f64)>, InferenceError> {
        if alpha.is_nan() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(InferenceError::Alpha(alpha));
        }
        let (Some(se), Some(df)) = (self.se(), self.df) else {
            return Ok(None);
        };
        let t = if df.is_infinite() {
            epistat_stats::normal::quantile(1.0 - alpha / 2.0)
        } else {
            match t_quantile(1.0 - alpha / 2.0, df) {
                Some(t) => t,
                None => return Ok(None),
            }
        };
        Ok(Some((self.estimate - t * se, self.estimate + t * se)))
    }
}

/// Pools `estimates` and their `variances`, one of each per imputed data set,
/// by Rubin's rules with the Barnard-Rubin degrees of freedom for a
/// complete-data analysis on `dfcom` degrees of freedom (`None` for infinite).
///
/// # Errors
///
/// [`InferenceError::Empty`] for no data set, [`InferenceError::Length`] when
/// the two lists differ in length, [`InferenceError::NotFinite`] for an
/// estimate that is not finite, a variance that is negative or not finite, or
/// a `dfcom` that is not positive, and [`InferenceError::Length`] with
/// `expected` 2 for a single data set, whose between variance is not defined.
pub fn rubin(
    estimates: &[f64],
    variances: &[f64],
    dfcom: Option<f64>,
) -> Result<Pooled, InferenceError> {
    let m = estimates.len();
    if m == 0 {
        return Err(InferenceError::Empty);
    }
    check_length("variances", m, variances.len())?;
    if m < 2 {
        return Err(InferenceError::Length {
            what: "estimates",
            expected: 2,
            found: m,
        });
    }
    if estimates.iter().any(|q| !q.is_finite())
        || variances.iter().any(|u| !u.is_finite() || *u < 0.0)
        || dfcom.is_some_and(|d| d.is_nan() || d <= 0.0)
    {
        return Err(InferenceError::NotFinite);
    }
    let mf = m as f64;
    let mut sum = 0.0;
    for q in estimates {
        sum += q;
    }
    let estimate = sum / mf;
    let mut usum = 0.0;
    for u in variances {
        usum += u;
    }
    let within = usum / mf;
    let mut ss = 0.0;
    for q in estimates {
        ss += (q - estimate) * (q - estimate);
    }
    let between = ss / (mf - 1.0);
    let inflated = (1.0 + 1.0 / mf) * between;
    let total = within + inflated;
    let riv = (within > 0.0).then(|| inflated / within);
    let lambda = (total > 0.0).then(|| inflated / total);
    let df = lambda.map(|lambda| barnard_rubin(mf, lambda, dfcom));
    let fmi = match (riv, df) {
        (Some(riv), Some(df)) => Some((riv + 2.0 / (df + 3.0)) / (riv + 1.0)),
        // A zero within variance with a positive total has an infinite riv,
        // so every unit of variance is due to the missing data.
        (None, Some(_)) => Some(1.0),
        _ => None,
    };
    Ok(Pooled {
        m,
        estimate,
        within,
        between,
        total,
        riv,
        lambda,
        fmi,
        df,
    })
}

fn barnard_rubin(m: f64, lambda: f64, dfcom: Option<f64>) -> f64 {
    let old = if lambda > 0.0 {
        (m - 1.0) / (lambda * lambda)
    } else {
        f64::INFINITY
    };
    let Some(dfcom) = dfcom else {
        return old;
    };
    let observed = (dfcom + 1.0) / (dfcom + 3.0) * dfcom * (1.0 - lambda);
    if old.is_infinite() {
        observed
    } else {
        old * observed / (old + observed)
    }
}
