//! A Cox model weighted by estimated propensity weights, with the variance that
//! accounts for the estimation.
//!
//! # Algorithm
//!
//! [`fit`] gives every row the weight of its subject and fits the weighted Cox
//! model with [`crate::cox::fit`]. The coefficients `beta` solve the weighted
//! score equation `U(beta; w) = 0`. With `V` the inverse of the information,
//! `r_k` the score residuals of the rows (they contain the case weight, see
//! [`crate::cox::score_residuals`]), and a unit being a subject or, with a
//! cluster column, a cluster:
//!
//! - The variance that ignores the estimation of the weights is
//!   `V (sum_u s_u s_u') V` with `s_u = sum of r_k over the rows of unit u`.
//!   With one row per subject this is `vcov(coxph(weights =, robust = TRUE))`.
//! - The variance that accounts for it stacks the propensity score equations
//!   under the Cox score. With `theta = (alpha, beta)`, `A` the derivative of
//!   the stacked equations and `B` the sum over units of the outer products of
//!   the stacked estimating functions `(psi_u, s_u)`, it is the `beta` block of
//!   `A^-1 B A^-T`. `A` is block lower triangular, with `A_alpha` from the
//!   propensity model, `-V^-1` for `beta`, and the block
//!
//!   ```text
//!   d U / d alpha = sum over rows k of (r_k / w_k) (d w_k / d alpha)
//!   ```
//!
//!   which is where the Cox equation depends on the weights. `r_k / w_k` is
//!   the derivative of the score with respect to the weight of row `k`: the
//!   score is linear in the case weights through the sums over events, and
//!   through the risk-set averages and the baseline hazard, whose dependence on
//!   the weight of row `k` the residual contains.
//!
//! The ties are Breslow or Efron as in [`crate::cox::fit`]; the derivative of the
//! score with respect to a weight is that of the likelihood the fit maximises.
//!
//! # Where it differs from `coxph_weightit`
//!
//! `WeightIt::coxph_weightit` (2.1.0) forms the block `d U / d alpha` as the
//! Cox score residuals at unit weights times `d w / d alpha`, a shortcut that is
//! exact for a model whose score of one subject depends on that subject's weight
//! only, such as a generalised linear model. The Cox score of a subject depends
//! on the weights of everyone in its risk sets, so with weights that vary
//! the shortcut gives a variance that is a little off: on the lalonde data with
//! simulated survival, between 0.5 and 3 per cent of the variance of the
//! treatment coefficient. The block here is exact, and agrees with the numerical
//! Jacobian of the stacked equations and with finite differences of refits. The
//! tests reproduce WeightIt's number from its own shortcut. WeightIt also fits
//! and differentiates the Breslow likelihood whatever ties are asked for, so only
//! the Breslow fits are compared with it.
//!
//! The stacked matrix is block lower triangular, so its sandwich is taken with
//! the LU attempts of `stack::GENERAL`. The Cox information alone is symmetric
//! and is taken with the symmetric attempts.
//!
//! # Memory
//!
//! The stacked estimating functions are held as one row per unit and one
//! column per parameter of the propensity model and of the outcome model.
//!
//! # References
//!
//! Shu D, Young JG, Toh S, Wang R. Variance estimation in inverse probability
//! weighted Cox models. Biometrics 2021;77(3):1101-12. Lin DY, Wei LJ. The
//! robust inference for the Cox proportional hazards model. J Am Stat Assoc
//! 1989;84:1074-8.

use epistat_core::Covariance;
use epistat_inference::Clusters;
use epistat_inference::sandwich::{meat, sandwich};
use epistat_linalg::{Attempt, Mat, MatRef, inverse};

use super::estimated::{Weights, prepare};
use super::iptw::Iptw;
use super::stack::{GENERAL, check_subjects, units, with_weights};
use crate::covariance;
use crate::cox::{CoxModelFit, CoxOptions, fit as cox_fit, score_residuals};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// A weighted Cox fit and its two variances, under its name in 0.3.
#[deprecated(
    since = "0.4.0",
    note = "`fit` returns an `EstimatedCox` for every kind of weights"
)]
pub type WeightedCox = EstimatedCox;

/// Fit the Cox model of `data` on `x` weighted by the product of the estimated
/// weights of `weights`, and give the variance that accounts for each model of
/// the weights being estimated.
///
/// The weights are the propensity weights of the subject of each row and the
/// censoring weights of the row, see [`Weights`]; the censoring weights of the
/// Kaplan-Meier and Cox models are fitted here to `data`. With propensity
/// weights alone, the variance is that of the stacked estimating equations, and
/// [`EstimatedCox::weight_covariance`] holds the covariance of the parameters
/// of the weights. Otherwise the variance adds to the influence of each unit
/// the influence of the weights through every model, see the [module
/// documentation of `weighting`](super), and treats the models as separate
/// estimating equations. The two agree up to rounding for propensity weights.
///
/// The influence of a unit sums the score residuals of its rows, and the units
/// are the clusters of `data`, the subjects of the propensity weights, the `id`
/// of `data` or the rows, in that order of preference. `data` must carry no
/// case weights.
///
/// # Errors
///
/// An input error for the invalid input of [`crate::cox::fit`] and of the models
/// of the weights, for data with case weights, for [`Weights::known`], which
/// [`crate::cox::fit`] fits with its robust variance, for `subject` without
/// propensity weights or with the wrong length or an unknown subject, for rows of
/// one subject in several clusters, and for exact ties; [`Error::NotConverged`]
/// and [`Error::Aliased`] for a fit that does not converge or has an aliased
/// column, and the errors of a solve that fails.
pub fn fit(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    opts: &CoxOptions,
    weights: &Weights<'_>,
) -> Result<EstimatedCox> {
    if weights.is_known() {
        return Err(Error::not_applicable(
            "weights: no model of the weights is given; cox::fit takes case weights as known",
        ));
    }
    match (weights.iptw, weights.censoring) {
        (Some(iptw), None) => fit_iptw(data, x, opts, iptw, weights.subject),
        _ => fit_general(data, x, opts, weights),
    }
}

/// The fit of [`fit`] for `weights`, through the terms of every model of the
/// weights also for propensity weights alone, as 0.3 gave it.
///
/// # Errors
///
/// As [`fit`].
#[deprecated(since = "0.4.0", note = "use `fit`")]
pub fn fit_estimated(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    opts: &CoxOptions,
    weights: &Weights<'_>,
) -> Result<EstimatedCox> {
    fit_general(data, x, opts, weights)
}

/// The fit weighted by the propensity weights of `iptw` alone, with the
/// stacked variance.
fn fit_iptw(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    opts: &CoxOptions,
    iptw: &Iptw,
    subject: Option<&[u32]>,
) -> Result<EstimatedCox> {
    let n = data.n_rows();
    let n_subjects = iptw.len();
    check_subjects(n, n_subjects, subject)?;
    let subj = |row: usize| subject.map_or(row, |s| s[row] as usize);
    let weight: Vec<f64> = (0..n).map(|row| iptw.weights[subj(row)]).collect();
    let buf = with_weights(data, weight)?;
    let weighted = buf.as_data();
    let model = cox_fit(&weighted, x, opts)?;
    if !model.converged {
        return Err(Error::NotConverged {
            iterations: model.iterations as usize,
        });
    }
    let p = x.ncols();
    let coefficients: Vec<f64> = model
        .coefficients
        .iter()
        .enumerate()
        .map(|(j, c)| c.ok_or(Error::Aliased { column: j }))
        .collect::<Result<_>>()?;
    let v: Vec<f64> = model
        .covariance
        .as_slice()
        .iter()
        .map(|c| {
            c.ok_or_else(|| {
                Error::undefined("covariance", "the covariance of the fit is undefined")
            })
        })
        .collect::<Result<_>>()?;
    let residuals = score_residuals(&weighted, x, &model.coefficients, opts)?;
    let residuals: Vec<Vec<f64>> = residuals
        .into_iter()
        .map(|r| {
            r.ok_or_else(|| Error::undefined("score residual", "a score residual is undefined"))
        })
        .collect::<Result<_>>()?;

    let k = iptw.n_params();
    let (unit, n_units) = units(n_subjects, &subj, data.cluster(), data.n_clusters())?;
    let mut psi = Mat::<f64>::zeros(n_units, k + p);
    let mut cross = Mat::<f64>::zeros(p, k);
    let mut buffer = vec![0.0; k];
    for i in 0..n_subjects {
        iptw.score_into(i, &mut buffer);
        for (j, &b) in buffer.iter().enumerate() {
            psi[(unit[i], j)] += b;
        }
    }
    for row in 0..n {
        let i = subj(row);
        iptw.weight_gradient_into(i, &mut buffer);
        let w = weighted.weight().map_or(1.0, |w| w[row]);
        for j in 0..p {
            let r = residuals[j][row];
            psi[(unit[i], k + j)] += r;
            for (m, &g) in buffer.iter().enumerate() {
                cross[(j, m)] += r / w * g;
            }
        }
    }

    let vmat = Mat::from_fn(p, p, |i, j| v[i * p + j]);
    let information = inverse(vmat.as_ref(), &Attempt::SYMMETRIC)?.x;
    let bread_ps = iptw.bread()?;
    let size = k + p;
    let a = Mat::from_fn(size, size, |i, j| {
        if i < k && j < k {
            bread_ps[(i, j)]
        } else if i >= k && j >= k {
            -information[(i - k, j - k)]
        } else if i >= k {
            cross[(i - k, j)]
        } else {
            0.0
        }
    });
    let ones = Clusters::singletons(n_units);
    let b = meat(psi.as_ref(), |_| 1.0, Some(&ones))?;
    let stacked = sandwich(a.as_ref(), b.as_ref(), &GENERAL)?;
    let b_beta = Mat::from_fn(p, p, |i, j| b[(k + i, k + j)]);
    let minus_information = Mat::from_fn(p, p, |i, j| -information[(i, j)]);
    let ignoring = sandwich(
        minus_information.as_ref(),
        b_beta.as_ref(),
        &Attempt::SYMMETRIC,
    )?;

    let flat = |m: &Mat<f64>, offset: usize, size: usize| -> Vec<f64> {
        (0..size)
            .flat_map(|i| (0..size).map(move |j| (i, j)))
            .map(|(i, j)| m[(offset + i, offset + j)])
            .collect()
    };
    let covariance = flat(&stacked, k, p);
    let robust_covariance = flat(&ignoring, 0, p);
    let se = |c: &[f64]| (0..p).map(|i| c[i * p + i].max(0.0).sqrt()).collect();
    Ok(EstimatedCox {
        coefficients,
        se: se(&covariance),
        robust_se: se(&robust_covariance),
        covariance: covariance::from_values(p, &covariance)?,
        robust_covariance: covariance::from_values(p, &robust_covariance)?,
        weight_covariance: Some(covariance::from_values(k, &flat(&stacked, 0, k))?),
        n_units,
        fit: model,
    })
}

/// A Cox fit weighted by estimated weights and its two variances.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EstimatedCox {
    /// The weighted fit of [`crate::cox::fit`].
    pub fit: CoxModelFit,
    /// The coefficients.
    pub coefficients: Vec<f64>,
    /// The covariance of the coefficients that accounts for the estimation of
    /// every model of the weights, `p` by `p`.
    pub covariance: Covariance,
    /// The standard errors from `covariance`.
    pub se: Vec<f64>,
    /// The covariance that takes the weights as known, over the same units.
    pub robust_covariance: Covariance,
    /// The standard errors from `robust_covariance`.
    pub robust_se: Vec<f64>,
    /// For propensity weights alone, the covariance of the parameters of the
    /// weights from the stacked sandwich: the propensity coefficients, then
    /// the marginal probability of treatment for stabilised weights.
    /// `None` otherwise.
    pub weight_covariance: Option<Covariance>,
    /// The number of units the estimating functions were summed over.
    pub n_units: usize,
}

/// The fit weighted by the weights of `estimated`, with the terms of every model
/// of the weights.
fn fit_general(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    opts: &CoxOptions,
    estimated: &Weights<'_>,
) -> Result<EstimatedCox> {
    let n = data.n_rows();
    let prepared = prepare(data, estimated)?;
    let buf = with_weights(data, prepared.weight.clone())?;
    let weighted = buf.as_data();
    let model = cox_fit(&weighted, x, opts)?;
    if !model.converged {
        return Err(Error::NotConverged {
            iterations: model.iterations as usize,
        });
    }
    let p = x.ncols();
    let coefficients: Vec<f64> = model
        .coefficients
        .iter()
        .enumerate()
        .map(|(j, c)| c.ok_or(Error::Aliased { column: j }))
        .collect::<Result<_>>()?;
    let v: Vec<f64> = model
        .covariance
        .as_slice()
        .iter()
        .map(|c| {
            c.ok_or_else(|| {
                Error::undefined("covariance", "the covariance of the fit is undefined")
            })
        })
        .collect::<Result<_>>()?;
    let residuals = score_residuals(&weighted, x, &model.coefficients, opts)?;
    let residuals: Vec<Vec<f64>> = residuals
        .into_iter()
        .map(|r| {
            r.ok_or_else(|| Error::undefined("score residual", "a score residual is undefined"))
        })
        .collect::<Result<_>>()?;
    let a = Mat::from_fn(n, p, |row, j| residuals[j][row]);
    let mut known = Mat::<f64>::zeros(prepared.n_units, p);
    for row in 0..n {
        for j in 0..p {
            known[(prepared.unit_of_row[row], j)] += a[(row, j)];
        }
    }
    let mut influence = known.clone();
    for nuisance in &prepared.nuisances {
        nuisance.add(a.as_ref(), &mut influence)?;
    }
    let vmat = Mat::from_fn(p, p, |i, j| v[i * p + j]);
    let information = inverse(vmat.as_ref(), &Attempt::SYMMETRIC)?.x;
    let minus_information = Mat::from_fn(p, p, |i, j| -information[(i, j)]);
    let ones = Clusters::singletons(prepared.n_units);
    let form = |psi: &Mat<f64>| -> Result<Mat<f64>> {
        let b = meat(psi.as_ref(), |_| 1.0, Some(&ones))?;
        sandwich(minus_information.as_ref(), b.as_ref(), &Attempt::SYMMETRIC).map_err(Error::from)
    };
    let adjusted = form(&influence)?;
    let ignoring = form(&known)?;
    let flat = |m: &Mat<f64>| -> Vec<f64> {
        (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| m[(i, j)])
            .collect()
    };
    let covariance = flat(&adjusted);
    let robust_covariance = flat(&ignoring);
    let se = |c: &[f64]| (0..p).map(|i| c[i * p + i].max(0.0).sqrt()).collect();
    Ok(EstimatedCox {
        coefficients,
        se: se(&covariance),
        robust_se: se(&robust_covariance),
        covariance: covariance::from_values(p, &covariance)?,
        robust_covariance: covariance::from_values(p, &robust_covariance)?,
        weight_covariance: None,
        n_units: prepared.n_units,
        fit: model,
    })
}
