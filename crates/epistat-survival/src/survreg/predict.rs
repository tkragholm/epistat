//! Predictions and residuals of a [`SurvregFit`], as `predict.survreg` and
//! `residuals.survreg`.

use faer::MatRef;

use super::dist::row_derivs;
use super::{SurvregFit, times_of};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// What [`SurvregFit::predict`] returns for each row.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictType {
    /// The linear predictor `x'beta`, the location of `ln T` or of `T`.
    LinearPredictor,
    /// The linear predictor on the scale of the time: `exp(x'beta)` for a
    /// transformed distribution, the linear predictor otherwise. The scale of
    /// `survreg`'s `type = "response"`, a median-like location and not a mean.
    Response,
    /// Quantiles of the time at the probabilities given, each in `(0, 1)`:
    /// `x'beta + q_p sigma`, with `q_p` the quantile of the standard
    /// distribution, on the scale of the time.
    Quantile {
        /// The probabilities.
        p: Vec<f64>,
    },
}

/// The options of [`SurvregFit::predict`].
#[derive(Debug, Clone, PartialEq)]
pub struct PredictOptions {
    /// What to predict.
    pub kind: PredictType,
    /// Whether to return standard errors, by the delta method from the
    /// variance of the fit. For a quantile with an estimated scale the
    /// variance of the log scale and its covariance with the coefficients
    /// enter.
    pub se_fit: bool,
    /// The dense stratum code (see [`SurvData::stratum`]) of each row, which
    /// gives the scale of a quantile. Needed for a quantile when the fit has
    /// more than one stratum, and not read otherwise.
    pub strata: Option<Vec<u32>>,
}

/// Predictions, row-major: `width` values for each row of the design, that is
/// one value, or one for each probability of a quantile prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// The predictions.
    pub fit: Vec<f64>,
    /// The standard errors, laid out as `fit`, when asked for.
    pub se: Option<Vec<f64>>,
    /// The number of values per row.
    pub width: usize,
}

/// The kinds of residual of [`SurvregFit::residuals`], named as in
/// `residuals.survreg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualKind {
    /// The observed time less the fitted, on the scale of the time. For an
    /// interval, the observation is the point where the density within the
    /// interval is largest.
    Response,
    /// The signed square root of twice the log likelihood of the row at the
    /// saturated model less at the fit, signed as the working residual.
    Deviance,
    /// `-g_eta / g_ee`: the step a Newton iteration would take for the row on
    /// its own.
    Working,
    /// The likelihood displacement of deleting the row, by the one-step
    /// approximation: `s' V s` with `s` the row's score for the coefficients
    /// and, when the scale is estimated, the log scale.
    LdCase,
    /// The likelihood displacement for a perturbation of the response.
    LdResp,
    /// The likelihood displacement for a perturbation of the shape, the
    /// scale.
    LdShape,
}

impl SurvregFit {
    fn dense_variance(&self) -> Result<Vec<f64>> {
        let q = self.n_params();
        let mut v = vec![0.0_f64; q * q];
        for (k, entry) in self.variance.iter().enumerate() {
            let (a, b) = (k / q, k % q);
            let aliased = |i: usize| i < self.n_coef() && self.coefficients[i].is_none();
            match entry {
                Some(value) => v[k] = *value,
                None if aliased(a) || aliased(b) => {}
                None => {
                    return Err(Error::input(
                        "the variance of the fit is not defined, so standard errors and \
                         displacements are not either"
                            .to_owned(),
                    ));
                }
            }
        }
        Ok(v)
    }

    fn linear_predictor(&self, row: &[f64]) -> f64 {
        let mut eta = 0.0;
        for (c, v) in self.coefficients.iter().zip(row) {
            if let Some(c) = c {
                eta += c * v;
            }
        }
        eta
    }

    /// Predict for the rows of `x`, which has the columns of the design the
    /// model was fitted on. An aliased column is left out of the prediction.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for a design with another number of columns or a
    /// value that is not finite; a probability outside `(0, 1)`; for a
    /// quantile of a fit with several strata, missing or wrong strata; and
    /// standard errors when the variance of the fit is not defined.
    ///
    /// # Panics
    ///
    /// Never: the variance the standard errors use is present when they are asked for.
    pub fn predict(&self, x: MatRef<'_, f64>, options: &PredictOptions) -> Result<Prediction> {
        let p = self.n_coef();
        let n = x.nrows();
        if x.ncols() != p {
            return Err(Error::input(format!(
                "the design has {} columns; the fit has {p} coefficients",
                x.ncols()
            )));
        }
        if (0..p).any(|j| (0..n).any(|i| !x[(i, j)].is_finite())) {
            return Err(Error::input(
                "the design has a value that is not finite".to_owned(),
            ));
        }
        let quantiles: &[f64] = match &options.kind {
            PredictType::Quantile { p } => {
                if p.is_empty() || p.iter().any(|&v| !(v > 0.0 && v < 1.0)) {
                    return Err(Error::input(
                        "quantile probabilities must be in (0, 1), at least one".to_owned(),
                    ));
                }
                p
            }
            _ => &[],
        };
        let strata: Vec<usize> = match (&options.kind, &options.strata) {
            (PredictType::Quantile { .. }, Some(codes)) => {
                if codes.len() != n || codes.iter().any(|&c| c as usize >= self.n_strata) {
                    return Err(Error::input(format!(
                        "strata need one code below {} for each of the {n} rows",
                        self.n_strata
                    )));
                }
                codes.iter().map(|&c| c as usize).collect()
            }
            (PredictType::Quantile { .. }, None) if self.n_strata > 1 => {
                return Err(Error::input(
                    "a quantile of a fit with strata on the scale needs the stratum of each row"
                        .to_owned(),
                ));
            }
            _ => vec![0; n],
        };
        let variance = if options.se_fit {
            Some(self.dense_variance()?)
        } else {
            None
        };
        let q = self.n_params();
        let log = self.distribution.is_log();
        let base = self.distribution.base();
        let offsets: Vec<f64> = quantiles.iter().map(|&pr| base.quantile(pr)).collect();
        let width = offsets.len().max(1);
        let mut fit = Vec::with_capacity(n * width);
        let mut se = options.se_fit.then(|| Vec::with_capacity(n * width));
        let mut row = vec![0.0_f64; p];
        for i in 0..n {
            for (j, r) in row.iter_mut().enumerate() {
                *r = x[(i, j)];
            }
            let eta = self.linear_predictor(&row);
            let mut t = vec![0.0_f64; q];
            for (j, tj) in t.iter_mut().enumerate().take(p) {
                *tj = if self.coefficients[j].is_some() {
                    row[j]
                } else {
                    0.0
                };
            }
            let quad = |t: &[f64]| -> f64 {
                let v = variance.as_ref().expect("asked for standard errors");
                let mut total = 0.0;
                for a in 0..q {
                    let mut inner = 0.0;
                    for b in 0..q {
                        inner += v[a * q + b] * t[b];
                    }
                    total += t[a] * inner;
                }
                // A quadratic form in a positive semi-definite matrix cannot
                // be below zero; rounding is the only way it is.
                total.max(0.0).sqrt()
            };
            match &options.kind {
                PredictType::LinearPredictor | PredictType::Response => {
                    let value = if matches!(options.kind, PredictType::Response) && log {
                        eta.exp()
                    } else {
                        eta
                    };
                    fit.push(value);
                    if let Some(se) = &mut se {
                        let s = quad(&t);
                        se.push(if matches!(options.kind, PredictType::Response) && log {
                            s * value
                        } else {
                            s
                        });
                    }
                }
                PredictType::Quantile { .. } => {
                    let s = strata[i];
                    let sigma = self.log_scale[s].exp();
                    for &offset in &offsets {
                        let on_log_scale = eta + offset * sigma;
                        let value = if log {
                            on_log_scale.exp()
                        } else {
                            on_log_scale
                        };
                        fit.push(value);
                        if let Some(se) = &mut se {
                            let mut tq = t.clone();
                            if self.scale_estimated {
                                tq[p + s] = offset * sigma;
                            }
                            let e = quad(&tq);
                            se.push(if log { e * value } else { e });
                        }
                    }
                }
            }
        }
        Ok(Prediction { fit, se, width })
    }

    /// Residuals of the model for `data`, the rows the model was fitted on
    /// and the same design `x`.
    ///
    /// With `weighted` each is multiplied by its case weight. A row is `None`
    /// where the residual is not defined: a time the distribution cannot
    /// take, a working residual with no curvature, a deviance below zero by
    /// more than rounding. The displacements (`LdCase`, `LdResp`, `LdShape`)
    /// use the model-based variance of the fit.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for data or a design that does not match the fit, and
    /// for a displacement when the variance of the fit is not defined.
    ///
    /// # Panics
    ///
    /// Never: the variance a displacement uses is present when it is asked for.
    pub fn residuals(
        &self,
        data: &SurvData<'_>,
        x: MatRef<'_, f64>,
        kind: ResidualKind,
        weighted: bool,
    ) -> Result<Vec<Option<f64>>> {
        let n = data.n_rows();
        let p = self.n_coef();
        if n != self.n_rows || x.nrows() != n || x.ncols() != p {
            return Err(Error::input(format!(
                "the data and design must be those of the fit: {} rows and {p} columns",
                self.n_rows
            )));
        }
        if data.n_strata().max(1) != self.n_strata {
            return Err(Error::input(
                "the strata of the data differ from those of the fit".to_owned(),
            ));
        }
        let times = times_of(data, self.distribution)?;
        let base = self.distribution.base();
        let log = self.distribution.is_log();
        let q = self.n_params();
        let displacement = matches!(
            kind,
            ResidualKind::LdCase | ResidualKind::LdResp | ResidualKind::LdShape
        );
        let variance = if displacement {
            Some(self.dense_variance()?)
        } else {
            None
        };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let s = data.stratum().map_or(0, |c| c[i] as usize);
            let ls = self.log_scale[s];
            let sigma = ls.exp();
            let eta = self.linear_predictors[i];
            let (y1, y2, status) = (times.y1[i], times.y2[i], times.status[i]);
            let d = row_derivs(base, status, y1, y2, eta, ls);
            let working = (-d.g_eta / d.g_ee).is_finite().then(|| -d.g_eta / d.g_ee);
            let value: Option<f64> = match kind {
                ResidualKind::Response => {
                    let (center, _) = base.saturated(status, y1, y2, sigma);
                    let value = if log {
                        center.exp() - eta.exp()
                    } else {
                        center - eta
                    };
                    value.is_finite().then_some(value)
                }
                ResidualKind::Working => working,
                ResidualKind::Deviance => {
                    let (_, saturated) = base.saturated(status, y1, y2, sigma);
                    let radicand = 2.0 * (saturated - d.g);
                    let rounding = 64.0 * f64::EPSILON * saturated.abs().max(d.g.abs()).max(1.0);
                    match working {
                        Some(w) if radicand.is_finite() && radicand >= -rounding => {
                            let root = radicand.max(0.0).sqrt();
                            Some(if w < 0.0 { -root } else { root })
                        }
                        _ => None,
                    }
                }
                ResidualKind::LdCase | ResidualKind::LdResp | ResidualKind::LdShape => {
                    let v = variance.as_ref().expect("a displacement has a variance");
                    let mut score = vec![0.0_f64; q];
                    for j in 0..p {
                        let xv = x[(i, j)];
                        score[j] = match kind {
                            ResidualKind::LdCase => d.g_eta * xv,
                            ResidualKind::LdResp => d.g_ee * sigma * xv,
                            _ => d.g_el * xv,
                        };
                    }
                    if self.scale_estimated {
                        score[p + s] = match kind {
                            ResidualKind::LdCase => d.g_ls,
                            ResidualKind::LdResp => d.g_el * sigma,
                            _ => d.g_ll,
                        };
                    }
                    let mut total = 0.0;
                    for a in 0..q {
                        let mut inner = 0.0;
                        for b in 0..q {
                            inner += v[a * q + b] * score[b];
                        }
                        total += score[a] * inner;
                    }
                    total.is_finite().then_some(total)
                }
            };
            let w = if weighted {
                data.weight().map_or(1.0, |w| w[i])
            } else {
                1.0
            };
            out.push(value.map(|v| v * w));
        }
        Ok(out)
    }
}
