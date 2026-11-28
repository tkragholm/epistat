//! The outcome model of the stacked trials with a competing event, and the
//! cumulative incidence it implies.
//!
//! # Models
//!
//! A row of the trials (a person in a trial in one period) ends in the event
//! `Y`, in the competing event `D`, or in neither. The competing event is taken
//! to come first in a period, as in the g-formula: the hazard of `D` is a
//! weighted logistic model on every row,
//!
//! ```text
//! logit P(D_t = 1 | x_t) = x_t' gamma,
//! ```
//!
//! and the hazard of `Y` a weighted logistic model on the rows without `D`,
//!
//! ```text
//! logit P(Y_t = 1 | D_t = 0, x_t) = x_t' beta,
//! ```
//!
//! both with the design and the weights of [`MsmInput`]. Each is the model of
//! [`fit_msm`] on its rows, with its covariance clustered by person.
//!
//! # The covariance of the two models
//!
//! The coefficients of the two models are estimated from the rows of the same
//! persons, so their clustered covariance has a cross block. With `w_i` the
//! weight, `p_i` and `q_i` the fitted hazards of `Y` and `D` of row `i`, the
//! scores of person `g` are
//!
//! ```text
//! U_g = sum over rows of g without D of w_i (y_i - p_i) x_i,
//! V_g = sum over rows of g of w_i (d_i - q_i) x_i,
//! ```
//!
//! the information matrices `A = sum w p (1 - p) x x'` and `B = sum w q (1 - q)
//! x x'` over the rows of each model, and the cross block of the stacked
//! sandwich
//!
//! ```text
//! cov(beta, gamma) = G / (G - 1) A^-1 (sum over g of U_g V_g') B^-1
//! ```
//!
//! for `G` persons. The diagonal blocks of the same sandwich are the
//! covariances of [`fit_msm`] (`sandwich::vcovCL(type = "HC0")`).
//!
//! # Cumulative incidence
//!
//! For a row of the population, with `p_s` and `q_s` its hazards at follow-up
//! time `s`, the probability of being free of both events through `s` and the
//! cumulative incidence of `Y` by `t` are
//!
//! ```text
//! S_s = prod over k <= s of (1 - p_k)(1 - q_k),
//! F_t = sum over s <= t of p_s (1 - q_s) S_{s-1},     S_{-1} = 1,
//! ```
//!
//! the risk of the event in the presence of the competing event (the total
//! effect of Young et al. 2020), and the risk is the mean of `F_t` over the
//! rows. The incidence of `D` is `sum q_s S_{s-1}`. The delta-method standard
//! error uses the derivatives
//!
//! ```text
//! d p_s / d beta = p_s (1 - p_s) x_s,     d q_s / d gamma = q_s (1 - q_s) x_s,
//! ```
//!
//! carried through the recursions of `S` and `F`, and the stacked covariance
//! of `(beta, gamma)` with its cross block.
//!
//! # References
//!
//! 1. Young JG, Stensrud MJ, Tchetgen Tchetgen EJ, Hernán MA. A causal framework
//!    for classical statistical estimands in failure-time settings with
//!    competing events. Stat Med 2020;39(8):1199-236.
//! 2. Zeileis A, Koll S, Graham N. Various versatile variances: an
//!    object-oriented implementation of clustered covariances in R. J Stat Softw
//!    2020;95(1):1-36.

use epistat_core::{Covariance, InputError};
use epistat_inference::Clusters;
use epistat_inference::sandwich::SandwichAdjustment;
use epistat_linalg::{Attempt, Mat, inverse};

use super::{Msm, MsmInput, Population, check_design, dense, fit_msm, quadratic_form};
use crate::check::check_lengths;
use crate::covariance;
use crate::error::{Error, Result};

/// The two outcome models of the stacked trials with a competing event.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MsmCompeting {
    /// The model of the event, fitted on the rows without the competing event.
    pub event: Msm,
    /// The model of the competing event, fitted on every row.
    pub competing: Msm,
    /// The clustered covariance of the coefficients of both models,
    /// `(beta, gamma)`, `2p` by `2p`: the event model's `p` coefficients, then
    /// the competing model's. Its diagonal blocks are the covariances of
    /// `event` and `competing`, entry `(i, p + j)` the covariance of the
    /// event model's coefficient `i` with the competing model's coefficient
    /// `j`.
    pub covariance: Covariance,
}

/// Fit the outcome model of the event and that of the competing event, with
/// their covariance clustered by person. `input.outcome` is the event and
/// `competing` the competing event of each row.
///
/// # Errors
///
/// An input error for `competing` of another length than the rows, a row
/// with both events, the errors of [`fit_msm`] for either model, and an
/// information matrix that cannot be inverted.
pub fn fit_msm_competing(input: &MsmInput<'_>, competing: &[bool]) -> Result<MsmCompeting> {
    let n = input.outcome.len();
    check_design("design", input.design, n)?;
    check_lengths(&[
        ("outcome", n),
        ("competing", competing.len()),
        ("weight", input.weight.len()),
        ("unit", input.unit.len()),
    ])?;
    if let Some(r) = (0..n).find(|&r| input.outcome[r] && competing[r]) {
        return Err(Error::inconsistent(format!(
            "row {r} has both the event and the competing event; give each row at most one"
        )));
    }
    let p = input.design.ncols();
    let rows: Vec<usize> = (0..n).filter(|&r| !competing[r]).collect();
    let design = Mat::from_fn(rows.len(), p, |i, j| input.design[(rows[i], j)]);
    let outcome: Vec<bool> = rows.iter().map(|&r| input.outcome[r]).collect();
    let weight: Vec<f64> = rows.iter().map(|&r| input.weight[r]).collect();
    let unit: Vec<u32> = rows.iter().map(|&r| input.unit[r]).collect();
    let event = fit_msm(MsmInput {
        design: design.as_ref(),
        outcome: &outcome,
        weight: &weight,
        unit: &unit,
    })?;
    let competing_fit = fit_msm(MsmInput {
        design: input.design,
        outcome: competing,
        weight: input.weight,
        unit: input.unit,
    })?;

    let clusters = Clusters::from_labels_first_seen(input.unit)?;
    let g = clusters.count();
    let of = clusters.of();
    let mut u = vec![0.0; g * p];
    let mut v = vec![0.0; g * p];
    let mut a = Mat::<f64>::zeros(p, p);
    let mut b = Mat::<f64>::zeros(p, p);
    let mut x = vec![0.0; p];
    for r in 0..n {
        for (j, xj) in x.iter_mut().enumerate() {
            *xj = input.design[(r, j)];
        }
        let w = input.weight[r];
        let c = of[r] as usize;
        let q = expit(&x, &competing_fit.coefficients);
        let d = f64::from(u8::from(competing[r]));
        add_score(&mut v[c * p..(c + 1) * p], &x, w * (d - q));
        add_information(&mut b, &x, w * q * (1.0 - q));
        if !competing[r] {
            let pr = expit(&x, &event.coefficients);
            let y = f64::from(u8::from(input.outcome[r]));
            add_score(&mut u[c * p..(c + 1) * p], &x, w * (y - pr));
            add_information(&mut a, &x, w * pr * (1.0 - pr));
        }
    }
    let meat = Mat::from_fn(p, p, |i, j| {
        (0..g).map(|k| u[k * p + i] * v[k * p + j]).sum::<f64>()
    });
    let invert = |m: &Mat<f64>| {
        inverse(m.as_ref(), &Attempt::SYMMETRIC)
            .map(|s| s.x)
            .map_err(Error::from)
    };
    let a_inv = invert(&a)?;
    let b_inv = invert(&b)?;
    // The person-clustered factor `G / (G - 1)` of the outcome model's `Cr1`
    // covariance, which the cross-covariance shares.
    let factor = SandwichAdjustment::ClusterCount
        .factor(n, p, Some(g))
        .ok_or_else(|| Error::no_information("outcome model: at least two persons are needed"))?;
    let am = product(&a_inv, &meat);
    let cross = product(&am, &b_inv);
    let cross_covariance: Vec<f64> = (0..p * p).map(|k| cross[(k / p, k % p)] * factor).collect();
    let covariance = covariance::from_values(
        2 * p,
        &stacked_covariance(&event, &competing_fit, &cross_covariance)?,
    )?;
    Ok(MsmCompeting {
        event,
        competing: competing_fit,
        covariance,
    })
}

/// `a b` by sums in the order of the inner index, without fused operations.
fn product(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    Mat::from_fn(a.nrows(), b.ncols(), |i, j| {
        (0..a.ncols()).map(|k| a[(i, k)] * b[(k, j)]).sum::<f64>()
    })
}

fn expit(x: &[f64], coefficients: &[f64]) -> f64 {
    let eta: f64 = x.iter().zip(coefficients).map(|(a, b)| a * b).sum();
    epistat_glm::Link::Logit.inverse(eta)
}

fn add_score(score: &mut [f64], x: &[f64], scale: f64) {
    for (s, xj) in score.iter_mut().zip(x) {
        *s += scale * xj;
    }
}

fn add_information(m: &mut Mat<f64>, x: &[f64], scale: f64) {
    for i in 0..x.len() {
        for j in 0..x.len() {
            m[(i, j)] += scale * x[i] * x[j];
        }
    }
}

/// The cumulative incidence of the event and of the competing event at each
/// follow-up time, with their standard errors.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CompetingIncidence {
    /// The risk of the event by each time, in the presence of the competing
    /// event.
    pub risk: Vec<f64>,
    /// The delta-method standard error of `risk`.
    pub se: Vec<f64>,
    /// The risk of the competing event by each time.
    pub competing_risk: Vec<f64>,
    /// The delta-method standard error of `competing_risk`.
    pub competing_se: Vec<f64>,
}

/// The cumulative incidence under one treatment from the two models of
/// [`fit_msm_competing`] for a population, given as for
/// [`cumulative_incidence`](super::cumulative_incidence). Both models read the
/// same design.
///
/// The sums over rows are taken in the order of the rows, so the result does
/// not depend on the number of threads.
///
/// # Errors
///
/// An input error for no time, no row, models of different numbers of
/// coefficients, designs with another number of columns than the models or of
/// different numbers of rows, a covariance that is not `2p` by `2p` or has an undefined
/// entry, and a design value that is not finite.
pub fn cumulative_incidence_competing(
    fit: &MsmCompeting,
    population: &Population<'_>,
) -> Result<CompetingIncidence> {
    let p = fit.event.coefficients.len();
    if fit.competing.coefficients.len() != p || fit.covariance.dim() != 2 * p {
        return Err(Error::mismatch(
            "the two models need the same number of coefficients, and the covariance 2p by 2p",
        ));
    }
    let (rows, times, design) = population.rows(p)?;
    if times == 0 {
        return Err(Error::no_information("designs: no times"));
    }
    if rows == 0 {
        return Err(Error::from(InputError::Empty { what: "designs" }));
    }
    // Sums over rows of F and its gradient (beta then gamma), and of the
    // incidence of D and its gradient, at each time.
    let k = 2 * p;
    let mut f_sum = vec![0.0; times];
    let mut f_grad = vec![0.0; times * k];
    let mut c_sum = vec![0.0; times];
    let mut c_grad = vec![0.0; times * k];
    let mut x = vec![0.0; p];
    // Per row: S, F, C and their gradients.
    let mut ds = vec![0.0; k];
    let mut df = vec![0.0; k];
    let mut dc = vec![0.0; k];
    let mut dp = vec![0.0; k];
    let mut dq = vec![0.0; k];
    for r in 0..rows {
        let (mut s, mut f, mut c) = (1.0, 0.0, 0.0);
        ds.fill(0.0);
        df.fill(0.0);
        dc.fill(0.0);
        for t in 0..times {
            design(r, t, &mut x);
            if x.iter().any(|v| !v.is_finite()) {
                return Err(Error::value(format!(
                    "design: a value of row {r} at time {t} is not finite"
                )));
            }
            let hp = expit(&x, &fit.event.coefficients);
            let hq = expit(&x, &fit.competing.coefficients);
            dp.fill(0.0);
            dq.fill(0.0);
            for j in 0..p {
                dp[j] = hp * (1.0 - hp) * x[j];
                dq[p + j] = hq * (1.0 - hq) * x[j];
            }
            // F += p (1 - q) S, C += q S, S *= (1 - p)(1 - q), with S the value
            // before the period on the right-hand sides.
            for j in 0..k {
                df[j] += dp[j] * (1.0 - hq) * s - hp * dq[j] * s + hp * (1.0 - hq) * ds[j];
                dc[j] += dq[j] * s + hq * ds[j];
            }
            f += hp * (1.0 - hq) * s;
            c += hq * s;
            for j in 0..k {
                ds[j] = ds[j] * (1.0 - hp) * (1.0 - hq)
                    - s * dp[j] * (1.0 - hq)
                    - s * (1.0 - hp) * dq[j];
            }
            s *= (1.0 - hp) * (1.0 - hq);
            f_sum[t] += f;
            c_sum[t] += c;
            for j in 0..k {
                f_grad[t * k + j] += df[j];
                c_grad[t * k + j] += dc[j];
            }
        }
    }
    let covariance = dense(&fit.covariance, "fit")?;
    let nf = rows as f64;
    let mut out = CompetingIncidence {
        risk: Vec::with_capacity(times),
        se: Vec::with_capacity(times),
        competing_risk: Vec::with_capacity(times),
        competing_se: Vec::with_capacity(times),
    };
    for t in 0..times {
        out.risk.push(f_sum[t] / nf);
        out.competing_risk.push(c_sum[t] / nf);
        let g: Vec<f64> = f_grad[t * k..(t + 1) * k].iter().map(|v| v / nf).collect();
        out.se.push(quadratic_form(&g, &covariance).max(0.0).sqrt());
        let g: Vec<f64> = c_grad[t * k..(t + 1) * k].iter().map(|v| v / nf).collect();
        out.competing_se
            .push(quadratic_form(&g, &covariance).max(0.0).sqrt());
    }
    Ok(out)
}

/// The `2p * 2p` covariance of `(beta, gamma)` in row order, from the
/// covariances of the two models and the `p * p` cross-covariance `cross` of
/// the event model's coefficients (rows) with the competing model's
/// (columns).
fn stacked_covariance(event: &Msm, competing: &Msm, cross: &[f64]) -> Result<Vec<f64>> {
    let p = event.coefficients.len();
    let k = 2 * p;
    let event = dense(&event.covariance, "event model")?;
    let competing = dense(&competing.covariance, "competing model")?;
    let mut v = vec![0.0; k * k];
    for i in 0..p {
        for j in 0..p {
            v[i * k + j] = event[i * p + j];
            v[(p + i) * k + p + j] = competing[i * p + j];
            v[i * k + p + j] = cross[i * p + j];
            v[(p + j) * k + i] = cross[i * p + j];
        }
    }
    Ok(v)
}
