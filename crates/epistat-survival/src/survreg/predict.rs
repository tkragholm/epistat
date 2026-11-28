//! Predictions and residuals of a [`SurvregFit`], as `predict.survreg` and
//! `residuals.survreg`.

use epistat_core::check;
use epistat_linalg::MatRef;

use super::dist::row_derivs;
use super::robust::{RowInput, dfbeta_row};
use super::{SurvregFit, times_of};
use crate::check::refuse_non_finite_matrix;
use crate::cox::residuals::Collapse;
use crate::error::{Error, Result};
use crate::order::{dense_index, dense_of_present};
use crate::surv::{OutcomeKind, SurvData};

/// What [`SurvregFit::predict`] returns for each row.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub enum SurvregPredictType {
    /// The linear predictor `x'beta`, the location of `ln T` or of `T`.
    #[default]
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
    /// The part of the linear predictor from each group of columns, as
    /// `type = "terms"`: for group `g` the sum of `(x_a - r_a) beta_a` over
    /// its columns, with `r` the reference. Aliased columns are left out of a
    /// group. The offset is not part of a term. One value per group and row.
    Terms {
        /// The column indices of each term.
        groups: Vec<Vec<usize>>,
        /// What the columns are measured from.
        reference: TermsReference,
    },
}

/// `SurvregPredictType`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `SurvregPredictType`")]
pub type PredictType = SurvregPredictType;

/// The point `type = "terms"` measures the columns from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TermsReference {
    /// The unweighted mean of each column over the rows the model was fitted
    /// on, `means` of `survreg`, which `predict.survreg` subtracts when the
    /// model has an intercept. A constant column centred on its mean
    /// contributes nothing to a term.
    #[default]
    Sample,
    /// Zero, as `predict.survreg` does for a model without an intercept.
    Zero,
}

/// The options of [`SurvregFit::predict`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SurvregPredictOptions {
    /// What to predict.
    pub kind: SurvregPredictType,
    /// Whether to return standard errors, by the delta method from the
    /// variance of the fit. For a quantile with an estimated scale the
    /// variance of the log scale and its covariance with the coefficients
    /// enter.
    pub se_fit: bool,
    /// The dense stratum code (see [`SurvData::stratum`]) of each row, which
    /// gives the scale of a quantile. Needed for a quantile when the fit has
    /// more than one stratum, and not read otherwise.
    pub strata: Option<Vec<u32>>,
    /// The offset of each row of the design, added to the linear predictor
    /// of a linear predictor, response or quantile. `None` is zero.
    /// `predict.survreg` leaves the offset of new data out of a quantile,
    /// which it includes for the fitted data; here it is included.
    pub offset: Option<Vec<f64>>,
    /// Whether the standard errors use the robust variance of the fit instead
    /// of the model-based one, as `predict.survreg` does for a fit with
    /// `robust = TRUE`.
    pub robust: bool,
}

/// `SurvregPredictOptions`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `SurvregPredictOptions`")]
pub type PredictOptions = SurvregPredictOptions;

/// Predictions, row-major: `width` values for each row of the design, that is
/// one value, or one for each probability of a quantile prediction.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SurvregPrediction {
    /// The predictions.
    pub fit: Vec<f64>,
    /// The standard errors, laid out as `fit`, when asked for.
    pub se: Option<Vec<f64>>,
    /// The number of values per row.
    pub width: usize,
}

/// `SurvregPrediction`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `SurvregPrediction`")]
pub type Prediction = SurvregPrediction;

/// The kinds of residual of [`SurvregFit::residuals`], named as in
/// `residuals.survreg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SurvregResidualKind {
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

/// `SurvregResidualKind`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `SurvregResidualKind`")]
pub type ResidualKind = SurvregResidualKind;

/// The options of [`SurvregFit::dfbeta`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DfbetaOptions {
    /// Divide each column by the model-based standard error of its
    /// parameter: `type = "dfbetas"` instead of `"dfbeta"`.
    pub standardise: bool,
    /// Multiply each row by its case weight, `weighted` of `residuals.survreg`.
    pub weighted: bool,
    /// Sum the rows within groups, `collapse` of `residuals.survreg`: within
    /// the clusters or the ids of the data, and the result has one row per
    /// distinct code, in ascending order of the caller's code. To sum within
    /// other groups, give their codes as the cluster or the id column of the
    /// data.
    pub collapse: Collapse,
}

/// The dfbeta residuals: for each row (or group of rows), the approximate
/// change in each parameter if the row were dropped, with the sign of
/// `residuals.survreg`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Dfbeta {
    /// Row-major, `width` values for each output row. `None` for an aliased
    /// column, for a row the fit gives no probability, and for a standardised
    /// value of a parameter with no variance.
    pub values: Vec<Option<f64>>,
    /// The number of output rows: the rows of the data, or the groups.
    pub n_rows: usize,
    /// The number of columns: the coefficients, then the log scales when they
    /// are estimated.
    pub width: usize,
    /// The code of each output row when the rows were collapsed.
    pub groups: Option<Vec<u32>>,
}

impl SurvregFit {
    fn dense_variance(&self) -> Result<Vec<f64>> {
        self.dense_of(self.variance.as_slice())
    }

    fn dense_of(&self, variance: &[Option<f64>]) -> Result<Vec<f64>> {
        let q = self.n_params();
        if variance.len() != q * q {
            return Err(Error::mismatch(format!(
                "the variance of the fit has {} entries for {q} parameters",
                variance.len()
            )));
        }
        let mut v = vec![0.0_f64; q * q];
        for (k, entry) in variance.iter().enumerate() {
            let (a, b) = (k / q, k % q);
            let aliased = |i: usize| i < self.n_coef() && self.coefficients[i].is_none();
            match entry {
                Some(value) => v[k] = *value,
                None if aliased(a) || aliased(b) => {}
                None => {
                    return Err(Error::fit(
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
    /// An input error for a design with another number of columns or a
    /// value that is not finite; a probability outside `(0, 1)`; for a
    /// quantile of a fit with several strata, missing or wrong strata; and
    /// standard errors when the variance of the fit is not defined.
    ///
    /// # Panics
    ///
    /// Never: the variance the standard errors use is present when they are asked for.
    pub fn predict(
        &self,
        x: MatRef<'_, f64>,
        options: &SurvregPredictOptions,
    ) -> Result<SurvregPrediction> {
        let p = self.n_coef();
        let n = x.nrows();
        if x.ncols() != p {
            return Err(Error::mismatch(format!(
                "the design has {} columns; the fit has {p} coefficients",
                x.ncols()
            )));
        }
        refuse_non_finite_matrix(x, "x")?;
        let quantiles: &[f64] = match &options.kind {
            SurvregPredictType::Quantile { p } => {
                if p.is_empty() || p.iter().any(|&v| !(v > 0.0 && v < 1.0)) {
                    return Err(Error::options(
                        "quantile probabilities must be in (0, 1), at least one".to_owned(),
                    ));
                }
                p
            }
            _ => &[],
        };
        let strata: Vec<usize> = match (&options.kind, &options.strata) {
            (SurvregPredictType::Quantile { .. }, Some(codes)) => {
                if codes.len() != n || codes.iter().any(|&c| c as usize >= self.n_strata) {
                    return Err(Error::out_of_range(format!(
                        "strata need one code below {} for each of the {n} rows",
                        self.n_strata
                    )));
                }
                codes.iter().map(|&c| c as usize).collect()
            }
            (SurvregPredictType::Quantile { .. }, None) if self.n_strata > 1 => {
                return Err(Error::missing(
                    "a quantile of a fit with strata on the scale needs the stratum of each row"
                        .to_owned(),
                ));
            }
            _ => vec![0; n],
        };
        if let Some(offset) = &options.offset {
            check::same_length("offset", n, offset.len())?;
            check::finite("offset", offset)?;
        }
        let terms: &[Vec<usize>] = match &options.kind {
            SurvregPredictType::Terms { groups, .. } => {
                if groups.is_empty() {
                    return Err(Error::options("terms needs at least one group".to_owned()));
                }
                if groups.iter().flatten().any(|&a| a >= p) {
                    return Err(Error::out_of_range(
                        "terms: an index is not a column of the design".to_owned(),
                    ));
                }
                groups
            }
            _ => &[],
        };
        let variance = if options.se_fit {
            Some(if options.robust {
                match &self.robust_variance {
                    Some(v) => self.dense_of(v.as_slice())?,
                    None => {
                        return Err(Error::fit("the fit has no robust variance".to_owned()));
                    }
                }
            } else {
                self.dense_variance()?
            })
        } else {
            None
        };
        let q = self.n_params();
        let log = self.distribution.is_log();
        let base = self.distribution.base(self.t_df);
        let offsets: Vec<f64> = quantiles.iter().map(|&pr| base.quantile(pr)).collect();
        let width = match &options.kind {
            SurvregPredictType::Terms { groups, .. } => groups.len(),
            _ => offsets.len().max(1),
        };
        let mut fit = Vec::with_capacity(n * width);
        let mut se = options.se_fit.then(|| Vec::with_capacity(n * width));
        let mut row = vec![0.0_f64; p];
        for i in 0..n {
            for (j, r) in row.iter_mut().enumerate() {
                *r = x[(i, j)];
            }
            let shift = options.offset.as_ref().map_or(0.0, |o| o[i]);
            let eta = self.linear_predictor(&row) + shift;
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
                SurvregPredictType::Terms { reference, .. } => {
                    let mut centred = row.clone();
                    if *reference == TermsReference::Sample {
                        for (c, m) in centred.iter_mut().zip(&self.column_means) {
                            *c -= m;
                        }
                    }
                    for group in terms {
                        let columns: Vec<usize> = group
                            .iter()
                            .copied()
                            .filter(|&a| self.coefficients[a].is_some())
                            .collect();
                        let mut value = 0.0;
                        for &a in &columns {
                            value += centred[a] * self.coefficients[a].unwrap_or(0.0);
                        }
                        fit.push(value);
                        if let Some(se) = &mut se {
                            let mut tg = vec![0.0_f64; q];
                            for &a in &columns {
                                tg[a] = centred[a];
                            }
                            se.push(quad(&tg));
                        }
                    }
                }
                SurvregPredictType::LinearPredictor | SurvregPredictType::Response => {
                    let value = if matches!(options.kind, SurvregPredictType::Response) && log {
                        eta.exp()
                    } else {
                        eta
                    };
                    fit.push(value);
                    if let Some(se) = &mut se {
                        let s = quad(&t);
                        se.push(
                            if matches!(options.kind, SurvregPredictType::Response) && log {
                                s * value
                            } else {
                                s
                            },
                        );
                    }
                }
                SurvregPredictType::Quantile { .. } => {
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
        Ok(SurvregPrediction { fit, se, width })
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
    /// An input error for data or a design that does not match the fit, and
    /// for a displacement when the variance of the fit is not defined.
    ///
    /// # Panics
    ///
    /// Never: the variance a displacement uses is present when it is asked for.
    pub fn residuals<K: OutcomeKind>(
        &self,
        data: &SurvData<'_, K>,
        x: MatRef<'_, f64>,
        kind: SurvregResidualKind,
        weighted: bool,
    ) -> Result<Vec<Option<f64>>> {
        let n = data.n_rows();
        let p = self.n_coef();
        if n != self.n_rows || x.nrows() != n || x.ncols() != p {
            return Err(Error::fit(format!(
                "the data and design must be those of the fit: {} rows and {p} columns",
                self.n_rows
            )));
        }
        if data.n_strata().max(1) != self.n_strata {
            return Err(Error::fit(
                "the strata of the data differ from those of the fit".to_owned(),
            ));
        }
        let times = times_of(data, self.distribution)?;
        let base = self.distribution.base(self.t_df);
        let log = self.distribution.is_log();
        let q = self.n_params();
        let displacement = matches!(
            kind,
            SurvregResidualKind::LdCase
                | SurvregResidualKind::LdResp
                | SurvregResidualKind::LdShape
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
                SurvregResidualKind::Response => {
                    let (center, _) = base.saturated(status, y1, y2, sigma);
                    let value = if log {
                        center.exp() - eta.exp()
                    } else {
                        center - eta
                    };
                    value.is_finite().then_some(value)
                }
                SurvregResidualKind::Working => working,
                SurvregResidualKind::Deviance => {
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
                SurvregResidualKind::LdCase
                | SurvregResidualKind::LdResp
                | SurvregResidualKind::LdShape => {
                    let v = variance.as_ref().expect("a displacement has a variance");
                    let mut score = vec![0.0_f64; q];
                    for j in 0..p {
                        let xv = x[(i, j)];
                        score[j] = match kind {
                            SurvregResidualKind::LdCase => d.g_eta * xv,
                            SurvregResidualKind::LdResp => d.g_ee * sigma * xv,
                            _ => d.g_el * xv,
                        };
                    }
                    if self.scale_estimated {
                        score[p + s] = match kind {
                            SurvregResidualKind::LdCase => d.g_ls,
                            SurvregResidualKind::LdResp => d.g_el * sigma,
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

    /// The dfbeta (or dfbetas) residuals of the model for `data`, the rows the
    /// model was fitted on, and the same design `x`: the score of each row
    /// times the model-based variance. The score of a row has `g_eta x` for
    /// the coefficients and `g_ls` for the log scale of its stratum. Where
    /// `survreg` reports 0 for an aliased column, this gives `None`.
    ///
    /// The model-based variance is used also when the fit has a robust one,
    /// as `residuals.survreg` uses `naive.var`.
    ///
    /// # Errors
    ///
    /// An input error for data or a design that does not match the fit, for
    /// a `collapse` of another length than the data, and when the variance of
    /// the fit is not defined.
    pub fn dfbeta<K: OutcomeKind>(
        &self,
        data: &SurvData<'_, K>,
        x: MatRef<'_, f64>,
        options: &DfbetaOptions,
    ) -> Result<Dfbeta> {
        let n = data.n_rows();
        let p = self.n_coef();
        if n != self.n_rows || x.nrows() != n || x.ncols() != p {
            return Err(Error::fit(format!(
                "the data and design must be those of the fit: {} rows and {p} columns",
                self.n_rows
            )));
        }
        if data.n_strata().max(1) != self.n_strata {
            return Err(Error::fit(
                "the strata of the data differ from those of the fit".to_owned(),
            ));
        }
        let collapse = match options.collapse {
            Collapse::None => None,
            Collapse::Cluster => Some((
                data.cluster()
                    .ok_or_else(|| Error::missing("collapse: the data have no clusters"))?,
                data.cluster_levels(),
            )),
            Collapse::Id => Some((
                data.id()
                    .ok_or_else(|| Error::missing("collapse: the data have no ids"))?,
                data.id_levels(),
            )),
        };
        let v = self.dense_variance()?;
        let q = self.n_params();
        let times = times_of(data, self.distribution)?;
        let base = self.distribution.base(self.t_df);
        let n_scale = if self.scale_estimated {
            self.n_strata
        } else {
            0
        };
        let mut rows = vec![0.0_f64; n * q];
        let mut x_row = vec![0.0_f64; p];
        for i in 0..n {
            for (j, value) in x_row.iter_mut().enumerate() {
                *value = x[(i, j)];
            }
            let s = data.stratum().map_or(0, |c| c[i] as usize);
            let out = &mut rows[i * q..(i + 1) * q];
            if times.y1[i].is_nan() {
                out.fill(f64::NAN);
                continue;
            }
            dfbeta_row(
                &RowInput {
                    base,
                    status: times.status[i],
                    y1: times.y1[i],
                    y2: times.y2[i],
                    eta: self.linear_predictors[i],
                    ls: self.log_scale[s],
                    stratum: s,
                    x: &x_row,
                    n_scale,
                },
                &v,
                out,
            );
            for (b, value) in out.iter_mut().enumerate() {
                if options.standardise {
                    *value /= v[b * q + b].sqrt();
                }
                if options.weighted
                    && let Some(w) = data.weight()
                {
                    *value *= w[i];
                }
            }
        }
        let (rows, n_rows, groups) = match collapse {
            None => (rows, n, None),
            Some((codes, labels)) => {
                let levels = dense_of_present(codes.iter().copied());
                let mut sums = vec![0.0_f64; levels.len() * q];
                for i in 0..n {
                    let g = dense_index(&levels, codes[i]);
                    for b in 0..q {
                        sums[g * q + b] += rows[i * q + b];
                    }
                }
                let count = levels.len();
                let codes = levels
                    .iter()
                    .map(|&c| labels.map_or(c, |l| l.code(c as usize).unwrap_or(c)))
                    .collect();
                (sums, count, Some(codes))
            }
        };
        let aliased = |b: usize| b < p && self.coefficients[b].is_none();
        let values = rows
            .iter()
            .enumerate()
            .map(|(k, &value)| (value.is_finite() && !aliased(k % q)).then_some(value))
            .collect();
        Ok(Dfbeta {
            values,
            n_rows,
            width: q,
            groups,
        })
    }
}
