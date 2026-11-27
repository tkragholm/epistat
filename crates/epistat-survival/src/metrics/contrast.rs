//! The difference between two prediction models in the Brier score or the area
//! under the curve, with the standard error from the paired influence
//! functions, as the contrasts of `riskRegression::Score` (`contrasts = TRUE`).
//!
//! Two models scored on the same rows have influence functions `IF_a` and
//! `IF_b` for the same metric, each with the first-order error of the estimate
//! as its mean. The difference `a - b` has the influence function `IF_a -
//! IF_b`, and its standard error is `sd(IF_a - IF_b) / sqrt(N)`, `sd` with the
//! divisor `N - 1`, which allows for the two estimates being computed on the
//! same rows and, when the censoring distribution is estimated, from the same
//! weights. The limits are `difference -+ z se` with `z` the `1 - (1 - level) /
//! 2` quantile of the normal distribution, and the p-value is `2 (1 -
//! Phi(|difference / se|))`: the Wald test that the two scores are equal.
//!
//! Compare a model with the null model by scoring both with
//! [`ScoreOptions::keep_influence`](super::score::ScoreOptions::keep_influence)
//! and passing [`null_brier`](super::score::null_brier)'s score as the
//! reference, as `Score` does with `null.model = TRUE`. The influence of the
//! scores is that of the options they were made with, so a Cox censoring model
//! or the Kaplan-Meier estimate enters as it did there, and a conservative
//! standard error stays conservative.
//!
//! # Differences from `Score`
//!
//! - The difference is `None` where either metric is undefined, and the
//!   standard error and the p-value are `None` where the influence function of
//!   the difference is constant, so that its variance is 0.
//! - Both scores must come from the same data, rows and times. `Score` always
//!   scores its models together.
//! - The difference of the index of prediction accuracy is not computed;
//!   `Score` gives no contrast for it.
//!
//! # References
//!
//! 1. Gerds TA, Kattan MW. Medical Risk Prediction Models: With Ties to
//!    Machine Learning. Chapman and Hall/CRC, 2021.
//! 2. Ozenne B, Sorensen AL, Scheike T, Torp-Pedersen C, Gerds TA.
//!    riskRegression: predicting the risk of an event using Cox regression
//!    models. R J 2017;9(2):440-60. doi:10.32614/RJ-2017-062

use epistat_core::Level;
use epistat_inference::Influence;

use super::score::{Auc, Score, se_of};
use crate::error::{Error, Result};

/// What the contrasts take.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ContrastOptions {
    /// The confidence level of the limits, `conf.int` of `Score`.
    pub level: Level,
}

/// The difference between two models at each time.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ScoreContrast {
    /// The times.
    pub times: Vec<f64>,
    /// The model minus the reference.
    pub difference: Vec<Option<f64>>,
    /// The standard error of the difference.
    pub se: Vec<Option<f64>>,
    /// The lower confidence limit.
    pub lower: Vec<Option<f64>>,
    /// The upper confidence limit.
    pub upper: Vec<Option<f64>>,
    /// The p-value of the Wald test of a difference of 0.
    pub p_value: Vec<Option<f64>>,
    /// The number of rows of positive weight.
    pub n_used: usize,
}

/// `ScoreContrast`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `ScoreContrast`")]
pub type Contrast = ScoreContrast;

/// The Brier score of `model` minus that of `reference`.
///
/// Both scores must have been made with `keep_influence` and
/// `variance`, on the same data and times.
///
/// # Errors
///
/// An input error when a score has no influence function, when the two are
/// at different times or from data of different numbers of rows, when their
/// influence functions are of other rows, or when a score or influence
/// function does not have one value or column per time.
pub fn brier_difference(
    model: &Score,
    reference: &Score,
    opts: &ContrastOptions,
) -> Result<ScoreContrast> {
    if model.times != reference.times
        || model.n_rows != reference.n_rows
        || model.n_used != reference.n_used
    {
        return Err(Error::fit(
            "the two scores must be on the same data and times",
        ));
    }
    let values = |s: &Score| s.brier.iter().map(|&b| Some(b)).collect::<Vec<_>>();
    contrast(
        &model.times,
        model.n_used,
        &values(model),
        &values(reference),
        model.influence.as_ref(),
        reference.influence.as_ref(),
        *opts,
    )
}

/// The area under the curve of `model` minus that of `reference`.
///
/// # Errors
///
/// As [`brier_difference`].
pub fn auc_difference(
    model: &Auc,
    reference: &Auc,
    opts: &ContrastOptions,
) -> Result<ScoreContrast> {
    if model.times != reference.times
        || model.n_rows != reference.n_rows
        || model.n_used != reference.n_used
    {
        return Err(Error::fit(
            "the two areas must be on the same data and times",
        ));
    }
    contrast(
        &model.times,
        model.n_used,
        &model.auc,
        &reference.auc,
        model.influence.as_ref(),
        reference.influence.as_ref(),
        *opts,
    )
}

fn contrast(
    times: &[f64],
    n_used: usize,
    model: &[Option<f64>],
    reference: &[Option<f64>],
    model_influence: Option<&Influence>,
    reference_influence: Option<&Influence>,
    opts: ContrastOptions,
) -> Result<ScoreContrast> {
    let (Some(a), Some(b)) = (model_influence, reference_influence) else {
        return Err(Error::fit(
            "the scores need their influence functions: compute them with variance and \
             keep_influence",
        ));
    };
    if a.keys() != b.keys() {
        return Err(Error::fit(
            "the two influence functions must be of the same rows",
        ));
    }
    if model.len() != times.len() || reference.len() != times.len() {
        return Err(Error::fit("a score must have one value per time"));
    }
    if a.n_columns() != times.len() || b.n_columns() != times.len() {
        return Err(Error::fit(
            "the influence function of a score must have one column per time",
        ));
    }
    let z = epistat_stats::dist::normal::quantile(opts.level.upper_probability());
    let mut out = ScoreContrast {
        times: times.to_vec(),
        difference: Vec::new(),
        se: Vec::new(),
        lower: Vec::new(),
        upper: Vec::new(),
        p_value: Vec::new(),
        n_used,
    };
    for j in 0..times.len() {
        let (Some(x), Some(y)) = (model[j], reference[j]) else {
            for v in [
                &mut out.difference,
                &mut out.se,
                &mut out.lower,
                &mut out.upper,
                &mut out.p_value,
            ] {
                v.push(None);
            }
            continue;
        };
        let delta = x - y;
        let diff: Vec<f64> = a
            .column(j)
            .iter()
            .zip(b.column(j))
            .map(|(x, y)| x - y)
            .collect();
        let se = se_of(&diff, a.keys())?.filter(|&s| s > 0.0);
        out.difference.push(Some(delta));
        out.se.push(se);
        out.lower.push(se.map(|s| delta - z * s));
        out.upper.push(se.map(|s| delta + z * s));
        out.p_value
            .push(se.map(|s| 2.0 * epistat_stats::dist::normal::cdf(-(delta / s).abs())));
    }
    Ok(out)
}
