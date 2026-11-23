//! Predictions and contrasts with delta-method intervals.

use epistat_linalg::MatRef;
use epistat_stats::normal;

use super::link;
use super::{FlexParFit, Scale};
use crate::error::{Error, Result};

/// What [`predict`] returns for a covariate row at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predict {
    /// The survivor function.
    Survival,
    /// The cumulative hazard `-ln S`.
    CumulativeHazard,
    /// The hazard.
    Hazard,
}

/// What [`contrast`] returns for a covariate row against a reference row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Contrast {
    /// The ratio of the hazards, a function of time where the model has
    /// time-varying effects or is not on the hazard scale.
    HazardRatio,
    /// The difference of the hazards.
    HazardDifference,
    /// The difference of the survivor functions.
    SurvivalDifference,
}

/// Settings of a prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictOptions {
    /// The level of the limits is `1 - alpha`.
    pub alpha: f64,
    /// Use the cluster-robust covariance, which the fit must have.
    pub robust: bool,
    /// Predict from a fit whose derivative in log time is not positive
    /// everywhere on the data.
    pub allow_non_monotone: bool,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            robust: false,
            allow_non_monotone: false,
        }
    }
}

/// An estimate at a time with its delta-method standard error and limits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    /// The time.
    pub time: f64,
    /// The estimate.
    pub estimate: f64,
    /// The standard error of the estimate by the delta method.
    pub standard_error: f64,
    /// The lower limit, `None` where the hazard is not positive and its
    /// logarithm is undefined.
    pub lower: Option<f64>,
    /// The upper limit.
    pub upper: Option<f64>,
}

struct Point {
    z: Vec<f64>,
    w: Vec<f64>,
    eta: f64,
    slope: f64,
}

fn covariance<'a>(fit: &'a FlexParFit, opts: &PredictOptions) -> Result<&'a [Option<f64>]> {
    if !(opts.alpha > 0.0 && opts.alpha < 1.0) {
        return Err(Error::input("alpha must be in (0, 1)"));
    }
    if !fit.monotone_on_data && !opts.allow_non_monotone {
        return Err(Error::input(
            "the fitted spline is not increasing on the data, so its curves are not survivor functions: set allow_non_monotone to predict anyway",
        ));
    }
    let c = if opts.robust {
        fit.robust_covariance.as_ref()
    } else {
        fit.covariance.as_ref()
    };
    c.map(Vec::as_slice).ok_or_else(|| {
        Error::input(if opts.robust {
            "the fit has no robust covariance: it needs clusters and a positive definite information"
        } else {
            "the information of the fit is not positive definite, so it has no covariance"
        })
    })
}

fn point(fit: &FlexParFit, x: MatRef<'_, f64>, row: usize, time: f64) -> Result<Point> {
    if !(time > 0.0 && time.is_finite()) {
        return Err(Error::input("times must be positive and finite"));
    }
    let q = fit.design.dim();
    let (mut z, mut w) = (vec![0.0; q], vec![0.0; q]);
    fit.design.fill(x, row, time.ln(), &mut z, Some(&mut w));
    let eta = z.iter().zip(&fit.theta).map(|(a, b)| a * b).sum();
    let slope = w.iter().zip(&fit.theta).map(|(a, b)| a * b).sum();
    Ok(Point { z, w, eta, slope })
}

fn variance(fit: &FlexParFit, cov: &[Option<f64>], g: &[f64]) -> f64 {
    let q = g.len();
    let d = fit.dimension();
    let mut total = 0.0;
    for a in 0..q {
        let ia = fit.full_index(a);
        for b in 0..q {
            total += g[a] * g[b] * cov[ia * d + fit.full_index(b)].unwrap_or(f64::NAN);
        }
    }
    total
}

fn check_columns(fit: &FlexParFit, x: MatRef<'_, f64>) -> Result<()> {
    if x.ncols() != fit.n_covariates() {
        return Err(Error::input(format!(
            "the covariates have {} columns and the fit {}",
            x.ncols(),
            fit.n_covariates()
        )));
    }
    if (0..x.nrows()).any(|i| (0..x.ncols()).any(|j| !x[(i, j)].is_finite())) {
        return Err(Error::input("the covariates must be finite"));
    }
    Ok(())
}

/// The log hazard and its gradient in the estimated parameters, `None` where
/// the derivative in log time is not positive.
fn log_hazard(scale: Scale, p: &Point, time: f64) -> Option<(f64, Vec<f64>)> {
    if p.slope <= 0.0 {
        return None;
    }
    let (g, g1, _) = link::log_hazard_factor(scale, p.eta);
    let value = g + p.slope.ln() - time.ln();
    let grad =
        p.z.iter()
            .zip(&p.w)
            .map(|(z, w)| g1 * z + w / p.slope)
            .collect();
    Some((value, grad))
}

/// The hazard, its gradient, and the value of `m(eta) eta'(u) / t`.
fn hazard(scale: Scale, p: &Point, time: f64) -> (f64, Vec<f64>) {
    // The hazard factor is m = H' and its derivative in eta is H''.
    let (_, m, dm) = link::cumulative(scale, p.eta);
    let value = m * p.slope / time;
    let grad =
        p.z.iter()
            .zip(&p.w)
            .map(|(z, w)| (dm * z * p.slope + m * w) / time)
            .collect();
    (value, grad)
}

fn survival_gradient(scale: Scale, p: &Point) -> (f64, Vec<f64>) {
    let s = link::survival(scale, p.eta);
    let (_, h1, _) = link::cumulative(scale, p.eta);
    (s, p.z.iter().map(|z| -s * h1 * z).collect())
}

/// Predictions for each row of `x_new` at each of `times`, as one vector of
/// predictions per row.
///
/// # Errors
///
/// [`Error::Input`] for a time that is not positive and finite, an `alpha`
/// outside `(0, 1)`, covariates that do not have the columns of the fit, a
/// fit without the covariance asked for, a fit that is not monotone on the
/// data unless allowed, and a hazard at a time where the derivative in log time
/// is not positive unless allowed (its limits are then `None`).
pub fn predict(
    fit: &FlexParFit,
    x_new: MatRef<'_, f64>,
    times: &[f64],
    what: Predict,
    options: &PredictOptions,
) -> Result<Vec<Vec<Prediction>>> {
    let cov = covariance(fit, options)?;
    check_columns(fit, x_new)?;
    let z = normal::two_sided_z(1.0 - options.alpha);
    let scale = fit.scale;
    let mut out = Vec::with_capacity(x_new.nrows());
    for row in 0..x_new.nrows() {
        let mut curve = Vec::with_capacity(times.len());
        for &time in times {
            let p = point(fit, x_new, row, time)?;
            let eta_se = variance(fit, cov, &p.z).sqrt();
            curve.push(match what {
                Predict::Survival => {
                    let (s, grad) = survival_gradient(scale, &p);
                    let (h, h1, _) = link::cumulative(scale, p.eta);
                    let factor = (z * h1 / h * eta_se).exp();
                    Prediction {
                        time,
                        estimate: s,
                        standard_error: variance(fit, cov, &grad).sqrt(),
                        lower: Some((-h * factor).exp()),
                        upper: Some((-h / factor).exp()),
                    }
                }
                Predict::CumulativeHazard => {
                    let (h, h1, _) = link::cumulative(scale, p.eta);
                    let grad: Vec<f64> = p.z.iter().map(|v| h1 * v).collect();
                    let factor = (z * h1 / h * eta_se).exp();
                    Prediction {
                        time,
                        estimate: h,
                        standard_error: variance(fit, cov, &grad).sqrt(),
                        lower: Some(h / factor),
                        upper: Some(h * factor),
                    }
                }
                Predict::Hazard => {
                    let (h, grad) = hazard(scale, &p, time);
                    let se = variance(fit, cov, &grad).sqrt();
                    let limits = log_hazard(scale, &p, time).map(|(_, lg)| {
                        let s = variance(fit, cov, &lg).sqrt();
                        (h * (-z * s).exp(), h * (z * s).exp())
                    });
                    Prediction {
                        time,
                        estimate: h,
                        standard_error: se,
                        lower: limits.map(|l| l.0),
                        upper: limits.map(|l| l.1),
                    }
                }
            });
        }
        out.push(curve);
    }
    Ok(out)
}

/// Contrasts of each row of `x_new` against the reference row `x_ref` (one row,
/// or as many rows as `x_new`) at each of `times`.
///
/// The limits of the hazard ratio are normal on the log scale, and those of the
/// differences are normal on their own scale. A hazard ratio at a time where a
/// hazard is not positive has `None` limits and a `NaN` estimate and standard
/// error, whatever the options.
///
/// # Errors
///
/// As [`predict`], and [`Error::Input`] for a reference of another number of
/// rows.
pub fn contrast(
    fit: &FlexParFit,
    x_new: MatRef<'_, f64>,
    x_ref: MatRef<'_, f64>,
    times: &[f64],
    what: Contrast,
    options: &PredictOptions,
) -> Result<Vec<Vec<Prediction>>> {
    let cov = covariance(fit, options)?;
    check_columns(fit, x_new)?;
    check_columns(fit, x_ref)?;
    if x_ref.nrows() != 1 && x_ref.nrows() != x_new.nrows() {
        return Err(Error::input(
            "the reference has one row or as many as the rows to compare",
        ));
    }
    let z = normal::two_sided_z(1.0 - options.alpha);
    let scale = fit.scale;
    let mut out = Vec::with_capacity(x_new.nrows());
    for row in 0..x_new.nrows() {
        let ref_row = if x_ref.nrows() == 1 { 0 } else { row };
        let mut curve = Vec::with_capacity(times.len());
        for &time in times {
            let a = point(fit, x_new, row, time)?;
            let b = point(fit, x_ref, ref_row, time)?;
            let none = Prediction {
                time,
                estimate: f64::NAN,
                standard_error: f64::NAN,
                lower: None,
                upper: None,
            };
            curve.push(match what {
                Contrast::HazardRatio => {
                    match (log_hazard(scale, &a, time), log_hazard(scale, &b, time)) {
                        (Some((la, ga)), Some((lb, gb))) => {
                            let diff: Vec<f64> = ga.iter().zip(&gb).map(|(x, y)| x - y).collect();
                            let se_log = variance(fit, cov, &diff).sqrt();
                            let ratio = (la - lb).exp();
                            Prediction {
                                time,
                                estimate: ratio,
                                standard_error: ratio * se_log,
                                lower: Some(ratio * (-z * se_log).exp()),
                                upper: Some(ratio * (z * se_log).exp()),
                            }
                        }
                        _ => none,
                    }
                }
                Contrast::HazardDifference => {
                    let (ha, ga) = hazard(scale, &a, time);
                    let (hb, gb) = hazard(scale, &b, time);
                    let diff: Vec<f64> = ga.iter().zip(&gb).map(|(x, y)| x - y).collect();
                    wald(time, ha - hb, variance(fit, cov, &diff).sqrt(), z)
                }
                Contrast::SurvivalDifference => {
                    let (sa, ga) = survival_gradient(scale, &a);
                    let (sb, gb) = survival_gradient(scale, &b);
                    let diff: Vec<f64> = ga.iter().zip(&gb).map(|(x, y)| x - y).collect();
                    wald(time, sa - sb, variance(fit, cov, &diff).sqrt(), z)
                }
            });
        }
        out.push(curve);
    }
    Ok(out)
}

fn wald(time: f64, estimate: f64, se: f64, z: f64) -> Prediction {
    Prediction {
        time,
        estimate,
        standard_error: se,
        lower: Some(estimate - z * se),
        upper: Some(estimate + z * se),
    }
}
