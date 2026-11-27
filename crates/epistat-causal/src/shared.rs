//! What the estimators share: the input checks, the nuisance model fits, the
//! influence functions and the contrast of two means.
//!
//! # Variance
//!
//! An estimator forms the influence function of its two means on the dfbeta
//! scale, `D_i` with `theta_hat - theta ~ sum_i D_i`, and [`influence`] turns
//! it into an [`Influence`] on the per-unit scale, `psi_i = n D_i`, with each
//! person (row) a unit coded by its row number. The variances are those of
//! `Influence::variance` under the [`ClusterRule`] of the estimator's R
//! reference: with `G` clusters, every person its own when there are none,
//!
//! ```text
//! V = c sum_g (S_g - n_g psi_bar)^2 / n^2
//! ```
//!
//! with `S_g` the sum of `psi` over cluster `g` and `c` the rule's factor.
//! [`rule`] gives the rule of the references that have the `n` divisor
//! without clusters and `G / (G - 1)` with them (`riskRegression::ate`,
//! `WeightIt`'s M-estimation variance, `survtmle`). Fewer than two clusters
//! leave the standard errors `None`.

use epistat_core::{Contrast, Estimand, Estimate, Interval, Level, Scale};
use epistat_glm::{Fit, Glm, GlmData};
use epistat_inference::{ClusterRule, Clusters, Influence};

use crate::error::{Error, Result};

pub(crate) fn compensated(values: impl IntoIterator<Item = f64>) -> f64 {
    epistat_stats::sum::compensated(values)
}

/// `found` values where `n` are needed.
pub(crate) fn check_len(name: &str, found: usize, n: usize) -> Result<()> {
    if found == n {
        Ok(())
    } else {
        Err(Error::input(format!(
            "{name}: {found} values for {n} persons"
        )))
    }
}

/// Every value finite.
pub(crate) fn check_finite(name: &str, values: &[f64]) -> Result<()> {
    match values.iter().position(|x| !x.is_finite()) {
        Some(i) => Err(Error::input(format!("{name}: value {i} is not finite"))),
        None => Ok(()),
    }
}

/// The case weights, or 1 for each of `n` persons, checked to be finite and
/// non-negative.
pub(crate) fn case_weights(name: &str, given: Option<&[f64]>, n: usize) -> Result<Vec<f64>> {
    match given {
        None => Ok(vec![1.0; n]),
        Some(w) => {
            check_len(name, w.len(), n)?;
            if let Some(i) = w.iter().position(|&x| !(x.is_finite() && x >= 0.0)) {
                return Err(Error::input(format!(
                    "{name}: value {i} is negative or not finite"
                )));
            }
            Ok(w.to_vec())
        }
    }
}

/// Both arms have a person of positive weight.
pub(crate) fn check_arms(weights: &[f64], treatment: &[bool]) -> Result<()> {
    let (mut treated, mut controls) = (false, false);
    for (&w, &t) in weights.iter().zip(treatment) {
        if w > 0.0 {
            treated |= t;
            controls |= !t;
        }
    }
    if treated && controls {
        Ok(())
    } else {
        Err(Error::input("treatment: both groups need persons"))
    }
}

/// The estimand is one of the effect in the population, on the treated and on
/// the controls.
pub(crate) fn check_estimand(estimand: Estimand) -> Result<()> {
    match estimand {
        Estimand::Ate | Estimand::Att | Estimand::Atc => Ok(()),
        other => Err(Error::input(format!(
            "estimand: {other} is not estimated; the estimands are ATE, ATT and ATC"
        ))),
    }
}

/// The design has a row per person, a column and finite values.
pub(crate) fn check_design(name: &str, m: epistat_linalg::MatRef<'_, f64>, n: usize) -> Result<()> {
    if m.nrows() != n {
        return Err(Error::input(format!(
            "{name}: {} rows for {n} persons",
            m.nrows()
        )));
    }
    if m.ncols() == 0 {
        return Err(Error::input(format!("{name}: no columns")));
    }
    if !epistat_linalg::is_finite(m) {
        return Err(Error::input(format!("{name}: a value is not finite")));
    }
    Ok(())
}

/// A fit of a nuisance model that converged and has no aliased column, with
/// its coefficients.
pub(crate) fn fit_glm(glm: &Glm, data: &GlmData<'_>, what: &str) -> Result<(Fit, Vec<f64>)> {
    let fit = glm
        .fit(data)
        .map_err(|e| Error::input(format!("{what}: {e}")))?;
    let coefficients = fit.full_rank().map_err(|e| match e {
        epistat_glm::Error::NotConverged { iterations, .. } => Error::NoConvergence(format!(
            "{what}: no convergence after {iterations} iterations"
        )),
        epistat_glm::Error::Aliased { column } => {
            Error::input(format!("{what}: design column {column} is aliased"))
        }
        other => Error::input(format!("{what}: {other}")),
    })?;
    Ok((fit, coefficients))
}

/// The data of a fit, as `GlmData` checks them.
pub(crate) fn glm_data<'a>(
    x: epistat_linalg::MatRef<'a, f64>,
    y: &'a [f64],
    weights: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
    what: &str,
) -> Result<GlmData<'a>> {
    let mut builder = GlmData::builder(x, y);
    if let Some(w) = weights {
        builder = builder.weights(w);
    }
    if let Some(o) = offset {
        builder = builder.offset(o);
    }
    builder
        .build()
        .map_err(|e| Error::input(format!("{what}: {e}")))
}

/// A fitted propensity of exactly 0 or 1 to within ten ulps of 1, which a
/// logistic fit reaches only under separation.
pub(crate) fn check_separation(propensity: &[f64]) -> Result<()> {
    let eps10 = 10.0 * f64::EPSILON;
    match propensity
        .iter()
        .position(|&p| p < eps10 || p > 1.0 - eps10)
    {
        Some(i) => Err(Error::input(format!(
            "propensity model: person {i} has fitted probability {} (separation)",
            propensity[i]
        ))),
        None => Ok(()),
    }
}

/// The rows `rows` as the `u32` keys of an influence function keyed by row.
pub(crate) fn row_keys(rows: impl IntoIterator<Item = usize>) -> Result<Vec<u32>> {
    rows.into_iter()
        .map(|i| {
            u32::try_from(i).map_err(|_| Error::input(format!("more than {} persons", u32::MAX)))
        })
        .collect()
}

fn inference_error(e: &epistat_inference::Error) -> Error {
    Error::input(format!("influence function: {e}"))
}

/// The clusters of every person keyed by row, for an influence function over
/// all persons or a subset of them.
pub(crate) fn clusters(cluster: Option<&[u32]>, n: usize) -> Result<Option<Clusters>> {
    let Some(cluster) = cluster else {
        return Ok(None);
    };
    check_len("cluster", cluster.len(), n)?;
    Clusters::per_row(cluster)
        .map(Some)
        .map_err(|e| inference_error(&e))
}

/// The rule of a reference with the `n` divisor without clusters and the
/// factor `G / (G - 1)` with them: `NoSmallSample` without clusters and
/// `SmallSample` with them.
pub(crate) fn rule(clustered: bool) -> ClusterRule {
    if clustered {
        ClusterRule::SmallSample
    } else {
        ClusterRule::NoSmallSample
    }
}

/// The influence function of the two means on the per-unit scale under
/// `rule`, from the values `d[0]` of mean 1 and `d[1]` of mean 0 on the dfbeta
/// scale over the persons of the rows `rows`.
pub(crate) fn influence(d: [&[f64]; 2], rows: Vec<u32>, rule: ClusterRule) -> Result<Influence> {
    let mut values = Vec::with_capacity(2 * rows.len());
    values.extend_from_slice(d[0]);
    values.extend_from_slice(d[1]);
    Influence::from_dfbeta_over_rows(values, rows, 2, rule).map_err(|e| inference_error(&e))
}

/// The variance of `influence`, which has one column.
fn variance_of(influence: &Influence, clusters: Option<&Clusters>) -> Result<Option<f64>> {
    let v = influence
        .variance(clusters)
        .map_err(|e| inference_error(&e))?;
    Ok(v.get(0, 0))
}

/// The variance of the combination `weights` of the columns of `influence`.
pub(crate) fn combined_variance(
    influence: &Influence,
    weights: &[f64],
    clusters: Option<&Clusters>,
) -> Result<Option<f64>> {
    let combined = influence
        .combine(weights)
        .map_err(|e| inference_error(&e))?;
    variance_of(&combined, clusters)
}

/// The standard error from a variance, `None` where the variance is.
fn se(variance: Option<f64>) -> Option<f64> {
    variance.map(|v| v.max(0.0).sqrt())
}

/// The Wald estimate of `value` against 0 at `level`.
pub(crate) fn wald(value: f64, variance: Option<f64>, level: Level) -> Estimate {
    epistat_stats::wald(value, se(variance), level, Scale::Identity)
}

/// The estimate of a ratio with the delta-method standard error, its interval
/// `ratio -/+ z se` and the p-value of the test of a ratio of 1.
pub(crate) fn ratio_estimate(ratio: f64, variance: Option<f64>, level: Level) -> Estimate {
    let se = se(variance);
    let point = Estimate::point(ratio).with_se(se);
    let Some(s) = se.filter(|s| s.is_finite()) else {
        return point;
    };
    let interval: Interval = epistat_inference::interval::normal(ratio, s, level);
    let p_value =
        (s > 0.0).then(|| 2.0 * epistat_stats::dist::normal::cdf(-((ratio - 1.0) / s).abs()));
    point.with_interval(Some(interval)).with_p_value(p_value)
}

/// The contrast of the means `theta[0]` (arm 1) and `theta[1]` (arm 0) whose
/// influence function is `influence`, with columns in the same order: the
/// difference with the influence `psi_1 - psi_0` and, unless `theta[1]` is 0,
/// the ratio with `(psi_1 - ratio psi_0) / theta[1]`.
pub(crate) fn contrast(
    theta: [f64; 2],
    influence: &Influence,
    clusters: Option<&Clusters>,
    level: Level,
) -> Result<Contrast> {
    let covariance = influence
        .variance(clusters)
        .map_err(|e| inference_error(&e))?;
    let mean1 = wald(theta[0], covariance.get(0, 0), level);
    let mean0 = wald(theta[1], covariance.get(1, 1), level);
    let difference = wald(
        theta[0] - theta[1],
        combined_variance(influence, &[1.0, -1.0], clusters)?,
        level,
    );
    let ratio = if theta[1] == 0.0 {
        None
    } else {
        let r = theta[0] / theta[1];
        let variance = combined_variance(influence, &[1.0 / theta[1], -r / theta[1]], clusters)?;
        Some(ratio_estimate(r, variance, level))
    };
    Ok(Contrast::new(mean1, mean0, difference, ratio))
}
