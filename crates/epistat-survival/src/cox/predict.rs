//! The baseline hazard, predictions and predicted survival curves of a Cox fit,
//! as `basehaz`, `predict.coxph` and `survfit.coxph` of `survival`.
//!
//! All three read the fit of [`fit`](super::fit) together with the data and the
//! design matrix it was fitted on, which a [`FittedModel`] holds. None of them
//! changes the fit. The tie rule of `CoxOptions::timefix`, the event code, the
//! offset and the weights are those of the fit. Rows of weight zero are left
//! out of the baseline, as they are left out of the fit, and are still given a
//! prediction.
//!
//! # The baseline hazard
//!
//! With `theta_j = exp((x_j - c)' beta + o_j - o_c)`, the weighted sums over the
//! risk set `R(t)` of a stratum, at an event time `t`, are
//!
//! ```text
//! S_0(t) = sum over R(t) of w_j theta_j,    S_1(t) = sum over R(t) of w_j theta_j (x_j - c),
//! ```
//!
//! and `E_0, E_1` the same sums over the events at `t` only. `W(t)` is the sum
//! of the case weights of the events and `d(t)` their number. The reference
//! `c` is the vector of means (see below), and `o_c` the weighted mean of the
//! offsets, so that `theta` is the risk relative to a subject at the means. The
//! increment of the cumulative hazard of that subject is
//!
//! ```text
//! Breslow:  h(t) = W / S_0
//! Efron:    h(t) = W (1 / d) sum over j = 0..d-1 of 1 / (S_0 - (j / d) E_0).
//! ```
//!
//! The Breslow estimate \[1\] is the Nelson-Aalen estimate with the risk scores
//! as weights. The Efron form is the Fleming-Harrington estimate \[2\] for tied
//! events, which `survfit.coxph` uses when the model was fitted with Efron
//! ties. [`HazardType`] chooses, and the default follows the ties of the fit as
//! `survfit.coxph` does. The survival curve is `exp(-H)` (`stype = 2`), the
//! Kalbfleisch-Prentice product limit (`stype = 1`) is not provided.
//!
//! The variance of the curve has two parts \[3\]. With `beta` fixed, the
//! cumulative hazard has the variance `sum of v(s)` over the event times
//! `s <= t`, with `v = W / S_0^2` for Breslow and
//! `v = W (1 / d) sum of 1 / (S_0 - (j / d) E_0)^2` for Efron. The second part
//! is the variance of `beta` carried through the hazard. With
//! `xbar(s) = W (1 / d) sum of (S_1 - (j / d) E_1) / (S_0 - (j / d) E_0)^2`,
//! which is `h(s)` times the weighted mean of `x - c` over the risk set for
//! Breslow, and for a subject with `z = x - c` and `theta = exp(z' beta)`,
//!
//! ```text
//! var(theta H(t)) = theta^2 ( sum of v(s)  +  g(t)' V g(t) ),
//! g(t) = H(t) z - sum over s <= t of xbar(s).
//! ```
//!
//! `V` is the covariance of the coefficients that the fit holds, the
//! cluster-robust one when there are clusters, as in `coxph`'s `var`. The
//! standard error `std_err` of a curve is that of the cumulative hazard, which
//! is that of `log S`. The confidence limits are the transforms of `survfit`'s
//! `conf.type`, with the normal quantile of `conf.int`.
//!
//! [`basehaz`] returns the baseline at the event times of each stratum, with
//! these partial sums, and its `centered = FALSE` form is the hazard
//! multiplied by `exp(-c' beta)`. A stratum with no event has no time and a
//! hazard of zero everywhere. The curves are at the event times only, where
//! `survfit` returns every distinct exit time and `censor = FALSE` the event
//! times.
//!
//! # The means
//!
//! `predict.coxph` and `basehaz` measure from the means of the covariates that
//! `coxph` stores, and which are not one rule. Without entry times they are the
//! case-weighted means. With entry times they are the unweighted means of every
//! row, the case weights ignored. A column whose values are all in
//! `{-1, 0, 1}` has the mean 0. [`Centring::Coxph`] is that, and the default.
//! [`Centring::Weighted`] is the weighted mean of the rows of positive weight
//! for every column. [`Centring::Given`] takes the vector. The survival curve
//! of a subject does not depend on the choice, only the baseline `centered` or
//! not, `lp`, `risk` and `terms` do.
//!
//! # Predictions
//!
//! [`predict`] gives for every row, of the fitted data or of new data, the
//! linear predictor `(x - r)' beta + o - o_bar`, the risk `exp` of it, the
//! expected number of events, and the survival `exp(-expected)`. `r` is the
//! reference: [`Reference::Zero`] is 0, [`Reference::Sample`] the means, and
//! [`Reference::Strata`] the case-weighted mean of the row's own stratum, as in
//! `predict.coxph`, where it is the default for `lp` and `risk`. `o_bar` is the
//! mean offset over the rows of the fit. The expected number of events is
//! `(H(stop) - H(start)) theta` of the baseline above, and a right-censored fit
//! has `start = 0`. Aliased coefficients count as 0.
//!
//! `se` of the linear predictor is `sqrt(z' V z)` with `z = x - r`. `se` of the
//! risk is that times the risk. `survival` gives the standard error of
//! `exp(-expected)` as that of the expected number times the survival. The
//! standard error of the expected number is
//!
//! ```text
//! theta sqrt( v(stop) - v(start) + (g(stop) - g(start))' V (g(stop) - g(start)) ),
//! ```
//!
//! where `v` is the cumulative `sum of v(s)` and `g` is as above.
//!
//! [`predict_terms`] gives the part of the linear predictor from each group of
//! columns, which is `type = "terms"` and takes the groups as column indices.
//! Its constant is `sum of beta_a mean_a`.
//!
//! # Where this differs from `survival`
//!
//! Three results of `predict.coxph` do not follow from their definition, and
//! are not copied. `survival` 3.8-12 returns for the standard error of the risk
//! `se(lp) sqrt(risk)` where the delta method gives `se(lp) risk`. For the
//! expected number with entry times it returns `sqrt(v2 - v1)` with
//! `v = varh + g' V g` at each end, which is the difference of two variances,
//! where the variance of the increment has the difference inside the quadratic
//! form, as above. It can be negative. With no entry time the two agree. And
//! `predict(fit)` without `se.fit` or `newdata` returns the linear predictor
//! with the raw offset, where with `se.fit` the mean offset is taken out. The
//! mean offset is always taken out here. `predict(type = "expected")` for new
//! data adds the offset of the new data to the risk where it multiplies, and
//! here it multiplies. The tests compare each with the value `survival`
//! computes from its own pieces.
//!
//! For the fitted data, `predict(type = "expected")` is the event indicator
//! less the martingale residual. For a row with an event at a time shared with
//! other events, under Efron ties, that is the hazard before the time plus the
//! part `1 - j / d` of the hazard at it, where the expected number here is the
//! cumulative hazard at the time, as `survival` computes it for new data and
//! for `survival` and the standard errors of the fitted data. The two agree for
//! every other row. The tests show both.
//!
//! Rows of weight zero are dropped from the baseline, where `survival` keeps
//! them in the count of events at a time and in the times of the curve. The
//! curves have no `n.censor`.
//!
//! # Agreement with R
//!
//! The tests give [`basehaz`], [`predict`], [`predict_terms`] and [`survfit`]
//! the coefficients and the covariance matrix that `coxph` returned, so that the
//! differences are those of these functions. The sums over `T` event times
//! carry an error of about `T` times the machine epsilon, and the products
//! with `p` coefficients add `p` more. The bounds are in the test.
//!
//! # References
//!
//! 1. Breslow N. Covariance analysis of censored survival data. Biometrics
//!    1974;30(1):89-99. doi:10.2307/2529620
//! 2. Fleming TR, Harrington DP. Nonparametric estimation of the survival
//!    distribution in censored data. Commun Stat Theory Methods
//!    1984;13(20):2469-86. doi:10.1080/03610928408828837
//! 3. Tsiatis AA. A large sample study of Cox's regression model. Ann Stat
//!    1981;9(1):93-108. doi:10.1214/aos/1176345335
//! 4. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000.

// Rows and strata are counted in `u32`; `FittedModel` refuses more rows than
// that.
#![expect(
    clippy::cast_possible_truncation,
    reason = "the entry points check that the row count fits in u32"
)]

use std::borrow::Cow;

use epistat_stats::normal;
use epistat_stats::sum::compensated;
use faer::MatRef;

use super::{CoxModelFit, CoxOptions, Ties};
use crate::check::{check_lengths, check_row_count, refuse_non_finite};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// The fit with the data and the design it was fitted on.
///
/// `options` must be the options `fit` was called with: the event code, the
/// tie rule for the times, the offset and the ties are read from them.
#[derive(Debug, Clone, Copy)]
pub struct FittedModel<'a> {
    /// The data of the fit.
    pub data: &'a SurvData<'a>,
    /// The design matrix of the fit.
    pub x: MatRef<'a, f64>,
    /// The result of [`fit`](super::fit).
    pub fit: &'a CoxModelFit,
    /// The options of the fit.
    pub options: &'a CoxOptions<'a>,
}

/// The point the covariates are measured from. See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Centring<'a> {
    /// The means `coxph` stores: weighted without entry times, unweighted with
    /// them, and 0 for a column with all values in `{-1, 0, 1}`.
    #[default]
    Coxph,
    /// The case-weighted mean of every column over the rows of positive weight.
    Weighted,
    /// One value per column.
    Given(&'a [f64]),
}

/// How the cumulative hazard handles tied events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HazardType {
    /// `W / S_0` at each time, `ctype = 1`.
    Breslow,
    /// The Fleming-Harrington form for tied events, `ctype = 2`.
    Efron,
}

/// Which covariance matrix of the coefficients the standard errors use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Variance {
    /// The cluster-robust matrix when the fit has one, otherwise the
    /// model-based one, as `var` of a `coxph` fit.
    #[default]
    Auto,
    /// The model-based matrix.
    Model,
    /// The cluster-robust matrix, an error without clusters.
    Robust,
}

/// What a prediction is measured from, as `reference` of `predict.coxph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reference {
    /// The case-weighted mean of the covariates in the row's own stratum.
    /// Without strata it is the means.
    Strata,
    /// The means of the fit, see [`Centring`].
    Sample,
    /// Zero.
    Zero,
}

/// What [`predict`] returns for a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictType {
    /// `(x - reference)' beta + offset - mean offset`.
    LinearPredictor,
    /// The exponential of the linear predictor.
    Risk,
    /// The expected number of events between the start and the stop of the row.
    Expected,
    /// `exp(-expected)`.
    Survival,
}

/// New rows to predict for. The columns follow the design of the fit.
#[derive(Debug, Clone, Copy)]
pub struct NewData<'a> {
    /// The covariates, one row per new row.
    pub x: MatRef<'a, f64>,
    /// The caller's stratum code of each row, required when the data of the
    /// fit has strata, and not read otherwise.
    pub stratum: Option<&'a [u32]>,
    /// The start of the interval of each row, for `Expected` and `Survival`.
    /// Required when the fit has entry times, and refused otherwise.
    pub start: Option<&'a [f64]>,
    /// The stop of each row, for `Expected` and `Survival`.
    pub stop: Option<&'a [f64]>,
    /// The offset of each row, zero by default.
    pub offset: Option<&'a [f64]>,
}

impl<'a> NewData<'a> {
    /// Rows with covariates only.
    #[must_use]
    pub fn new(x: MatRef<'a, f64>) -> Self {
        Self {
            x,
            stratum: None,
            start: None,
            stop: None,
            offset: None,
        }
    }
}

/// The options of [`basehaz`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BaselineOptions<'a> {
    /// The reference the hazard is for.
    pub centring: Centring<'a>,
    /// The form of the hazard. `None` follows the ties of the fit: Efron for
    /// Efron ties and Breslow otherwise.
    pub hazard: Option<HazardType>,
    /// The covariance used for the variance parts.
    pub variance: Variance,
}

/// The options of [`predict`] and [`predict_terms`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictOptions<'a> {
    /// The reference. `None` is the default of `predict.coxph`: `Strata` for
    /// the linear predictor and the risk and for terms `Sample`; the expected
    /// number and the survival are always from `Sample`.
    pub reference: Option<Reference>,
    /// The means.
    pub centring: Centring<'a>,
    /// The covariance of the standard errors.
    pub variance: Variance,
    /// Whether to give standard errors, `se.fit`.
    pub se_fit: bool,
    /// The hazard form of the expected number, `None` as in [`BaselineOptions`].
    pub hazard: Option<HazardType>,
}

impl Default for PredictOptions<'_> {
    fn default() -> Self {
        Self {
            reference: None,
            centring: Centring::Coxph,
            variance: Variance::Auto,
            se_fit: false,
            hazard: None,
        }
    }
}

/// The confidence limits of a survival curve, `conf.type` of `survfit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfType {
    /// No limits.
    None,
    /// `S exp(-+ z se)`, the upper limit at most 1.
    #[default]
    Log,
    /// `exp(-exp(log(-log S) -+ z se / log S))`.
    LogLog,
    /// `S -+ z se S`, clipped to the unit interval.
    Plain,
    /// The logit scale.
    Logit,
    /// The arcsine-square-root scale.
    Arcsin,
}

/// The options of [`survfit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvfitOptions<'a> {
    /// The means the baseline is formed with. The curves do not depend on it.
    pub centring: Centring<'a>,
    /// The form of the hazard, as in [`BaselineOptions`], `ctype` of `survfit`.
    pub hazard: Option<HazardType>,
    /// The covariance of the standard error.
    pub variance: Variance,
    /// Whether to give the standard error, `se.fit`. Without it there are no
    /// confidence limits.
    pub se_fit: bool,
    /// The kind of confidence limit.
    pub conf_type: ConfType,
    /// The confidence level, `conf.int`, in `(0, 1)`.
    pub conf_level: f64,
}

impl Default for SurvfitOptions<'_> {
    fn default() -> Self {
        Self {
            centring: Centring::Coxph,
            hazard: None,
            variance: Variance::Auto,
            se_fit: true,
            conf_type: ConfType::Log,
            conf_level: 0.95,
        }
    }
}

/// The baseline of one stratum at its event times, in ascending order.
#[derive(Debug, Clone, PartialEq)]
pub struct StratumBaseline {
    /// The caller's code of the stratum, `None` without strata.
    pub code: Option<u32>,
    /// The distinct event times.
    pub time: Vec<f64>,
    /// The sum of the case weights of the events at each time.
    pub n_events: Vec<f64>,
    /// The sum of the case weights of the rows at risk at each time.
    pub n_risk: Vec<f64>,
    /// The increment of the cumulative hazard at each time.
    pub hazard: Vec<f64>,
    /// The cumulative hazard `H(t)` at each time.
    pub cumulative_hazard: Vec<f64>,
    /// The sum of `v(s)` up to each time: the variance of the cumulative
    /// hazard if the coefficients were known.
    pub cumulative_variance: Vec<f64>,
    /// The sum of `xbar(s)` up to each time, `p` values per time.
    pub cumulative_xbar: Vec<f64>,
}

impl StratumBaseline {
    /// The number of event times `s <= t`.
    fn count_up_to(&self, t: f64) -> usize {
        self.time.partition_point(|&s| s <= t)
    }
}

/// The baseline hazard of every stratum, the result of [`basehaz`].
#[derive(Debug, Clone, PartialEq)]
pub struct Baseline {
    /// One entry per stratum in the order of the codes, one entry without strata.
    pub strata: Vec<StratumBaseline>,
    /// The reference `c` of the covariates.
    pub centre: Vec<f64>,
    /// The weighted mean of the offsets `o_c`, 0 without an offset.
    pub offset_centre: f64,
    /// The form of the hazard.
    pub hazard_type: HazardType,
    /// `exp(-c' beta)`, the factor from the centred baseline to the baseline
    /// of a subject with every covariate 0 (`centered = FALSE`).
    pub uncentring_factor: f64,
    /// The weighted rows used, `n_used` of the data.
    pub n_used: usize,
    beta: Vec<f64>,
    variance: Vec<f64>,
}

/// The result of [`predict`]: a value per row, and its standard error when
/// requested.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// The predicted value of each row.
    pub fit: Vec<f64>,
    /// The standard error of each row when `se_fit` was set.
    pub se: Option<Vec<f64>>,
}

/// The result of [`predict_terms`].
#[derive(Debug, Clone, PartialEq)]
pub struct TermsPrediction {
    /// Row-major, one row per row and one column per group.
    pub fit: Vec<f64>,
    /// The standard errors laid out as `fit`, when requested.
    pub se: Option<Vec<f64>>,
    /// The number of groups.
    pub n_terms: usize,
    /// `sum of beta_a mean_a`, the constant `predict.coxph` returns as an
    /// attribute: the linear predictor is the sum of the terms plus this
    /// plus the offset, less the reference.
    pub constant: f64,
}

/// The times of the curves of one stratum.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveTimes {
    /// The caller's code of the stratum, `None` without strata.
    pub code: Option<u32>,
    /// The distinct event times, ascending.
    pub time: Vec<f64>,
    /// The sum of the case weights of the rows at risk.
    pub n_risk: Vec<f64>,
    /// The sum of the case weights of the events.
    pub n_event: Vec<f64>,
}

/// The curve of one row of new data.
#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    /// The index in [`SurvCurves::strata`] of the stratum of the row.
    pub stratum: usize,
    /// The cumulative hazard `theta H(t)` at the times of the stratum.
    pub cumulative_hazard: Vec<f64>,
    /// `exp(-cumulative_hazard)`.
    pub survival: Vec<f64>,
    /// The standard error of the cumulative hazard, that is of `log S`.
    pub std_err: Option<Vec<f64>>,
    /// The lower confidence limit, `None` where it is undefined.
    pub lower: Option<Vec<Option<f64>>>,
    /// The upper confidence limit.
    pub upper: Option<Vec<Option<f64>>>,
}

/// The curves of [`survfit`], one per row of the new data.
#[derive(Debug, Clone, PartialEq)]
pub struct SurvCurves {
    /// The times of every stratum.
    pub strata: Vec<CurveTimes>,
    /// One curve per row of new data, in order.
    pub curves: Vec<Curve>,
}

// -- sums

/// A running sum with a compensation term, so that adding and removing rows
/// from a risk set loses no more than a rounding error of the result.
#[derive(Clone, Copy, Default)]
struct Sum {
    sum: f64,
    carry: f64,
}

impl Sum {
    fn add(&mut self, value: f64) {
        let t = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.carry += (self.sum - t) + value;
        } else {
            self.carry += (value - t) + self.sum;
        }
        self.sum = t;
    }

    fn value(self) -> f64 {
        self.sum + self.carry
    }
}

fn quad(v: &[f64], p: usize, a: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..p {
        let mut row = 0.0;
        for j in 0..p {
            row += v[i * p + j] * a[j];
        }
        total += a[i] * row;
    }
    total
}

/// `x_i - centre` for a row of a matrix.
fn centred_row(x: MatRef<'_, f64>, i: usize, centre: &[f64], out: &mut [f64]) {
    for (a, (o, c)) in out.iter_mut().zip(centre).enumerate() {
        *o = x[(i, a)] - c;
    }
}

// -- the model, read once

/// The model's coefficients and the columns of the data an entry point reads.
struct Model<'a> {
    p: usize,
    rows: usize,
    x: MatRef<'a, f64>,
    beta: Vec<f64>,
    variance: Vec<f64>,
    data: SurvData<'a>,
    is_event: Vec<bool>,
    weight: Cow<'a, [f64]>,
    stratum: Cow<'a, [u32]>,
    n_strata: usize,
    has_start: bool,
    offset: Option<&'a [f64]>,
    ties: Ties,
}

impl<'a> Model<'a> {
    fn new(m: &FittedModel<'a>, variance: Variance) -> Result<Self> {
        let rows = m.data.n_rows();
        check_row_count(rows)?;
        check_lengths(&[("data", rows), ("x", m.x.nrows())])?;
        let p = m.x.ncols();
        check_lengths(&[("x columns", p), ("coefficients", m.fit.coefficients.len())])?;
        if p == 0 {
            return Err(Error::input("x has no columns"));
        }
        for j in 0..p {
            for i in 0..rows {
                if !m.x[(i, j)].is_finite() {
                    return Err(Error::input("x: a value is not finite"));
                }
            }
        }
        if m.fit.coefficients.iter().all(Option::is_none) {
            return Err(Error::input(
                "fit: the model has no estimate, so there is nothing to predict from",
            ));
        }
        let beta: Vec<f64> = m
            .fit
            .coefficients
            .iter()
            .map(|c| c.unwrap_or(0.0))
            .collect();
        let chosen = match variance {
            Variance::Auto => m
                .fit
                .robust_covariance
                .as_ref()
                .unwrap_or(&m.fit.covariance),
            Variance::Model => &m.fit.covariance,
            Variance::Robust => m.fit.robust_covariance.as_ref().ok_or_else(|| {
                Error::input("variance: the fit has no cluster-robust covariance")
            })?,
        };
        if chosen.len() != p * p {
            return Err(Error::input("fit: the covariance is not p by p"));
        }
        let variance: Vec<f64> = chosen.iter().map(|v| v.unwrap_or(0.0)).collect();
        if let Some(offset) = m.options.offset {
            check_lengths(&[("data", rows), ("offset", offset.len())])?;
        }
        let data = m.data.clone().with_timefix(m.options.timefix)?;
        let surv = data.followup()?;
        let code = match (m.options.cause, surv.n_codes()) {
            (None, 1) => 1,
            (Some(code), _) => code,
            (None, _) => {
                return Err(Error::input(
                    "cause: the status has several codes, so the event code is required",
                ));
            }
        };
        let is_event: Vec<bool> = data.cause(code)?.iter().collect();
        let has_start = surv.start().is_some();
        let weight = m
            .data
            .weight()
            .map_or_else(|| Cow::Owned(vec![1.0; rows]), Cow::Borrowed);
        let stratum = m
            .data
            .stratum()
            .map_or_else(|| Cow::Owned(vec![0; rows]), Cow::Borrowed);
        let n_strata = data.n_strata().max(1);
        Ok(Self {
            p,
            rows,
            x: m.x,
            beta,
            variance,
            data,
            is_event,
            weight,
            stratum,
            n_strata,
            has_start,
            offset: m.options.offset,
            ties: m.options.ties,
        })
    }

    fn has_strata(&self) -> bool {
        self.data.stratum().is_some()
    }

    fn stop(&self) -> &[f64] {
        self.data.followup().map_or(&[], |s| s.stop())
    }

    fn start(&self) -> Option<&[f64]> {
        self.data.followup().ok().and_then(|s| s.start())
    }

    /// The default hazard form follows the ties of the fit. Any rule other
    /// than Efron is Breslow, as `survfit.coxph` has it.
    fn hazard_type(&self, chosen: Option<HazardType>) -> HazardType {
        chosen.unwrap_or(if self.ties == Ties::Efron {
            HazardType::Efron
        } else {
            HazardType::Breslow
        })
    }

    fn means(&self, centring: Centring<'_>) -> Result<Vec<f64>> {
        let (p, rows) = (self.p, self.rows);
        match centring {
            Centring::Given(values) => {
                check_lengths(&[("x columns", p), ("centring", values.len())])?;
                refuse_non_finite(values, "centring")?;
                Ok(values.to_vec())
            }
            Centring::Weighted | Centring::Coxph => {
                let coxph = centring == Centring::Coxph;
                let weighted = !(coxph && self.has_start);
                let total = compensated(self.weight.iter().copied());
                let mut means = vec![0.0; p];
                for (a, mean) in means.iter_mut().enumerate() {
                    if coxph && (0..rows).all(|i| matches!(self.x[(i, a)], -1.0 | 0.0 | 1.0)) {
                        continue;
                    }
                    *mean = if weighted {
                        compensated((0..rows).map(|i| self.weight[i] * self.x[(i, a)])) / total
                    } else {
                        compensated((0..rows).map(|i| self.x[(i, a)])) / rows as f64
                    };
                }
                Ok(means)
            }
        }
    }

    /// The weighted mean of the offsets, 0 without an offset.
    fn weighted_offset_mean(&self) -> f64 {
        self.offset.map_or(0.0, |o| {
            let total = compensated(self.weight.iter().copied());
            compensated((0..self.rows).map(|i| self.weight[i] * o[i])) / total
        })
    }

    /// The unweighted mean of the offsets over every row, as `predict.coxph`
    /// has it.
    fn plain_offset_mean(&self) -> f64 {
        self.offset
            .map_or(0.0, |o| compensated(o.iter().copied()) / self.rows as f64)
    }

    fn dot_beta(&self, z: &[f64]) -> f64 {
        let mut total = 0.0;
        for (value, b) in z.iter().zip(&self.beta) {
            total += value * b;
        }
        total
    }

    fn levels(&self) -> Option<Vec<u32>> {
        self.data.stratum_levels().map(|l| l.to_vec())
    }

    /// The dense stratum of each row of new data.
    fn new_strata(&self, new: &NewData<'_>) -> Result<Vec<usize>> {
        let rows = new.x.nrows();
        let Some(levels) = self.levels() else {
            return Ok(vec![0; rows]);
        };
        let Some(codes) = new.stratum else {
            return Err(Error::input(
                "newdata: the stratum is required, the fit has strata",
            ));
        };
        check_lengths(&[("newdata x", rows), ("newdata stratum", codes.len())])?;
        codes
            .iter()
            .map(|code| {
                levels.binary_search(code).map_err(|_| {
                    Error::input(format!(
                        "newdata: stratum {code} is not a stratum of the fit"
                    ))
                })
            })
            .collect()
    }

    fn check_new(&self, new: &NewData<'_>) -> Result<()> {
        let rows = new.x.nrows();
        if new.x.ncols() != self.p {
            return Err(Error::input(format!(
                "newdata x has {} columns, the fit has {}",
                new.x.ncols(),
                self.p
            )));
        }
        for j in 0..self.p {
            for i in 0..rows {
                if !new.x[(i, j)].is_finite() {
                    return Err(Error::input("newdata x: a value is not finite"));
                }
            }
        }
        if let Some(offset) = new.offset {
            check_lengths(&[("newdata x", rows), ("newdata offset", offset.len())])?;
            refuse_non_finite(offset, "newdata offset")?;
        }
        Ok(())
    }

    /// The case-weighted mean of the covariates of every stratum, `p` per
    /// stratum.
    fn stratum_means(&self) -> Result<Vec<f64>> {
        let p = self.p;
        let mut sums = vec![Sum::default(); self.n_strata * p];
        let mut totals = vec![Sum::default(); self.n_strata];
        for i in 0..self.rows {
            let s = self.stratum[i] as usize;
            totals[s].add(self.weight[i]);
            for a in 0..p {
                sums[s * p + a].add(self.weight[i] * self.x[(i, a)]);
            }
        }
        let mut out = vec![0.0; self.n_strata * p];
        for s in 0..self.n_strata {
            let total = totals[s].value();
            if total <= 0.0 {
                return Err(Error::input(format!(
                    "reference: stratum level {s} has no weight, so its mean is undefined"
                )));
            }
            for a in 0..p {
                out[s * p + a] = sums[s * p + a].value() / total;
            }
        }
        Ok(out)
    }
}

// -- the baseline

/// The sums at one event time, and the increments they give.
struct Increment {
    hazard: f64,
    variance: f64,
    xbar: Vec<f64>,
}

fn increment(
    kind: HazardType,
    weight: f64,
    count: usize,
    s0: f64,
    s1: &[f64],
    e0: f64,
    e1: &[f64],
) -> Result<Increment> {
    let p = s1.len();
    let bad = || Error::input("the sum of the risk scores of a risk set is not positive");
    match kind {
        HazardType::Breslow => {
            if !(s0 > 0.0 && s0.is_finite()) {
                return Err(bad());
            }
            let hazard = weight / s0;
            Ok(Increment {
                hazard,
                variance: weight / (s0 * s0),
                xbar: s1.iter().map(|&s| (s / s0) * hazard).collect(),
            })
        }
        HazardType::Efron => {
            let d = count as f64;
            let (mut sum1, mut sum2) = (0.0, 0.0);
            let mut xbar = vec![0.0; p];
            for j in 0..count {
                let fraction = j as f64 / d;
                let denominator = s0 - e0 * fraction;
                if !(denominator > 0.0 && denominator.is_finite()) {
                    return Err(bad());
                }
                let t = 1.0 / denominator;
                sum1 += t / d;
                sum2 += t * t / d;
                for a in 0..p {
                    xbar[a] += ((s1[a] - e1[a] * fraction) * t * t) / d;
                }
            }
            Ok(Increment {
                hazard: weight * sum1,
                variance: weight * sum2,
                xbar: xbar.iter().map(|v| weight * v).collect(),
            })
        }
    }
}

/// The baseline of every stratum with `centre` and `offset_centre`.
fn build_baseline(
    model: &Model<'_>,
    centre: Vec<f64>,
    offset_centre: f64,
    kind: HazardType,
) -> Result<Baseline> {
    let p = model.p;
    let rows = model.rows;
    let stop = model.stop();
    let start = model.start();
    // The risk score of every row of positive weight, w * theta.
    let mut z = vec![0.0; p];
    let mut risk = vec![0.0; rows];
    for i in 0..rows {
        if model.weight[i] > 0.0 {
            centred_row(model.x, i, &centre, &mut z);
            let eta = model.dot_beta(&z) + model.offset.map_or(0.0, |o| o[i] - offset_centre);
            let theta = eta.exp();
            if !theta.is_finite() {
                return Err(Error::input(
                    "the risk score of a row is not finite; the offset or the coefficients are too large",
                ));
            }
            risk[i] = model.weight[i] * theta;
        }
    }
    let levels = model.levels();
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); model.n_strata];
    for i in 0..rows {
        if model.weight[i] > 0.0 {
            members[model.stratum[i] as usize].push(i as u32);
        }
    }
    let mut strata = Vec::with_capacity(model.n_strata);
    for (s, order) in members.iter_mut().enumerate() {
        let code = levels.as_ref().map(|l| l[s]);
        // Descending exit time, ties in input order.
        order.sort_by(|&a, &b| stop[b as usize].total_cmp(&stop[a as usize]));
        let len = order.len();
        let mut by_start: Vec<u32> = (0..len as u32).collect();
        if let Some(start) = start {
            by_start.sort_by(|&a, &b| {
                start[order[b as usize] as usize].total_cmp(&start[order[a as usize] as usize])
            });
        }
        let mut s0 = Sum::default();
        let mut s1 = vec![Sum::default(); p];
        let mut count = Sum::default();
        let (mut entering, mut leaving) = (0usize, 0usize);
        let mut time = Vec::new();
        let mut n_events = Vec::new();
        let mut n_risk = Vec::new();
        let mut hazard = Vec::new();
        let mut variance = Vec::new();
        let mut xbar: Vec<f64> = Vec::new();
        let mut position = 0;
        while position < len {
            let t = stop[order[position] as usize];
            let mut end = position;
            while end < len && stop[order[end] as usize].total_cmp(&t).is_eq() {
                end += 1;
            }
            let events: Vec<usize> = (position..end)
                .filter(|&k| model.is_event[order[k] as usize])
                .collect();
            position = end;
            if events.is_empty() {
                continue;
            }
            while entering < len && stop[order[entering] as usize] >= t {
                let i = order[entering] as usize;
                centred_row(model.x, i, &centre, &mut z);
                s0.add(risk[i]);
                count.add(model.weight[i]);
                for a in 0..p {
                    s1[a].add(risk[i] * z[a]);
                }
                entering += 1;
            }
            if let Some(start) = start {
                while leaving < len && start[order[by_start[leaving] as usize] as usize] >= t {
                    let i = order[by_start[leaving] as usize] as usize;
                    centred_row(model.x, i, &centre, &mut z);
                    s0.add(-risk[i]);
                    count.add(-model.weight[i]);
                    for a in 0..p {
                        s1[a].add(-(risk[i] * z[a]));
                    }
                    leaving += 1;
                }
            }
            let mut weight = 0.0;
            let mut e0 = 0.0;
            let mut e1 = vec![0.0; p];
            for &k in &events {
                let i = order[k] as usize;
                centred_row(model.x, i, &centre, &mut z);
                weight += model.weight[i];
                e0 += risk[i];
                for a in 0..p {
                    e1[a] += risk[i] * z[a];
                }
            }
            let sums1: Vec<f64> = s1.iter().map(|v| v.value()).collect();
            let inc = increment(kind, weight, events.len(), s0.value(), &sums1, e0, &e1)?;
            time.push(t);
            n_events.push(weight);
            n_risk.push(count.value());
            hazard.push(inc.hazard);
            variance.push(inc.variance);
            xbar.extend(inc.xbar);
        }
        // The sweep ran in descending time.
        time.reverse();
        n_events.reverse();
        n_risk.reverse();
        hazard.reverse();
        variance.reverse();
        let events = time.len();
        let mut reversed = vec![0.0; xbar.len()];
        for k in 0..events {
            let from = (events - 1 - k) * p;
            reversed[k * p..(k + 1) * p].copy_from_slice(&xbar[from..from + p]);
        }
        let mut cumulative_hazard = Vec::with_capacity(events);
        let mut cumulative_variance = Vec::with_capacity(events);
        let mut cumulative_xbar = Vec::with_capacity(events * p);
        let (mut h, mut v) = (0.0, 0.0);
        let mut x_sum = vec![0.0; p];
        for k in 0..events {
            h += hazard[k];
            v += variance[k];
            cumulative_hazard.push(h);
            cumulative_variance.push(v);
            for a in 0..p {
                x_sum[a] += reversed[k * p + a];
            }
            cumulative_xbar.extend_from_slice(&x_sum);
        }
        strata.push(StratumBaseline {
            code,
            time,
            n_events,
            n_risk,
            hazard,
            cumulative_hazard,
            cumulative_variance,
            cumulative_xbar,
        });
    }
    let shift = model.dot_beta(&centre);
    Ok(Baseline {
        strata,
        offset_centre,
        hazard_type: kind,
        uncentring_factor: (-shift).exp(),
        n_used: model.data.n_used(),
        beta: model.beta.clone(),
        variance: model.variance.clone(),
        centre,
    })
}

fn baseline_of(
    model: &Model<'_>,
    centring: Centring<'_>,
    hazard: Option<HazardType>,
) -> Result<Baseline> {
    let centre = model.means(centring)?;
    build_baseline(
        model,
        centre,
        model.weighted_offset_mean(),
        model.hazard_type(hazard),
    )
}

/// The means of the covariates a baseline and a prediction are measured from,
/// for the rule `centring`.
///
/// A curve of "the average subject" is [`survfit`] at these values.
///
/// # Errors
///
/// [`Error::Input`] as for [`basehaz`].
pub fn reference_means(model: &FittedModel<'_>, centring: Centring<'_>) -> Result<Vec<f64>> {
    Model::new(model, Variance::Auto)?.means(centring)
}

/// The weighted mean of the offsets of the fit, 0 without an offset.
///
/// The curve of the average subject is [`survfit`] at [`reference_means`] with
/// this offset, as `survfit.coxph` without `newdata` gives it.
///
/// # Errors
///
/// [`Error::Input`] as for [`basehaz`].
pub fn reference_offset(model: &FittedModel<'_>) -> Result<f64> {
    Ok(Model::new(model, Variance::Auto)?.weighted_offset_mean())
}

/// The baseline cumulative hazard of every stratum, as `basehaz` of
/// `survival`. See the module documentation for the definition and for how
/// it differs from `survival`.
///
/// # Errors
///
/// [`Error::Input`] when `x` has another number of rows than the data or a
/// value that is not finite, when the fit has no estimate or a covariance of
/// another size, when `variance` is `Robust` and the fit has no robust
/// covariance, when the offset has another length than the data, when the
/// event code is missing for a status with several codes or outside the
/// codes, when `Centring::Given` has another length than `x` has columns, and
/// when a risk score or a risk set sum is not finite or not positive.
pub fn basehaz(model: &FittedModel<'_>, opts: &BaselineOptions<'_>) -> Result<Baseline> {
    let m = Model::new(model, opts.variance)?;
    baseline_of(&m, opts.centring, opts.hazard)
}

// -- predict

fn expected_parts(
    baseline: &Baseline,
    stratum: usize,
    start: f64,
    stop: f64,
) -> (f64, f64, usize, usize) {
    let s = &baseline.strata[stratum];
    let (k1, k2) = (s.count_up_to(start), s.count_up_to(stop));
    let at = |k: usize| {
        if k == 0 {
            0.0
        } else {
            s.cumulative_hazard[k - 1]
        }
    };
    let var = |k: usize| {
        if k == 0 {
            0.0
        } else {
            s.cumulative_variance[k - 1]
        }
    };
    (at(k2) - at(k1), var(k2) - var(k1), k1, k2)
}

/// Predictions for the rows of the fitted data (`newdata` `None`) or for new
/// rows. See the module documentation for the definitions, the standard errors
/// and where they differ from `predict.coxph`.
///
/// The result has one value for every row, including rows of weight zero.
///
/// # Errors
///
/// [`Error::Input`] as for [`basehaz`], and when `newdata` has another number
/// of columns than the fit or a value that is not finite, a stratum that is not
/// one of the fit or missing for a fit with strata, a start or stop of another
/// length, a start for a fit without entry times or a missing start for a fit
/// with them, a missing stop for `Expected` and `Survival`, or a stop not after
/// its start. Also when the reference is `Strata` and a stratum carries no
/// weight.
pub fn predict(
    model: &FittedModel<'_>,
    kind: PredictType,
    newdata: Option<&NewData<'_>>,
    opts: &PredictOptions<'_>,
) -> Result<Prediction> {
    let m = Model::new(model, opts.variance)?;
    let p = m.p;
    let expected = matches!(kind, PredictType::Expected | PredictType::Survival);
    let reference = if expected {
        Reference::Sample
    } else {
        opts.reference.unwrap_or(Reference::Strata)
    };
    let (xm, rows, strata, offset): (MatRef<'_, f64>, usize, Vec<usize>, Option<&[f64]>) =
        match newdata {
            None => (
                m.x,
                m.rows,
                m.stratum.iter().map(|&s| s as usize).collect(),
                m.offset,
            ),
            Some(new) => {
                m.check_new(new)?;
                (new.x, new.x.nrows(), m.new_strata(new)?, new.offset)
            }
        };
    let means = m.means(opts.centring)?;
    let stratum_means = if reference == Reference::Strata && m.has_strata() {
        Some(m.stratum_means()?)
    } else {
        None
    };
    let reference_of = |i: usize, out: &mut [f64]| match reference {
        Reference::Zero => out.fill(0.0),
        Reference::Sample => out.copy_from_slice(&means),
        Reference::Strata => match &stratum_means {
            Some(all) => out.copy_from_slice(&all[strata[i] * p..(strata[i] + 1) * p]),
            None => out.copy_from_slice(&means),
        },
    };
    let mut fit = Vec::with_capacity(rows);
    let mut se = opts.se_fit.then(|| Vec::with_capacity(rows));
    let mut centre = vec![0.0; p];
    let mut z = vec![0.0; p];
    match kind {
        PredictType::LinearPredictor | PredictType::Risk => {
            let mean_offset = m.plain_offset_mean();
            for i in 0..rows {
                reference_of(i, &mut centre);
                centred_row(xm, i, &centre, &mut z);
                let lp = m.dot_beta(&z) + (offset.map_or(0.0, |o| o[i]) - mean_offset);
                let value = if kind == PredictType::Risk {
                    lp.exp()
                } else {
                    lp
                };
                fit.push(value);
                if let Some(se) = se.as_mut() {
                    let s = quad(&m.variance, p, &z).sqrt();
                    se.push(if kind == PredictType::Risk {
                        s * value
                    } else {
                        s
                    });
                }
            }
        }
        PredictType::Expected | PredictType::Survival => {
            let (starts, stops) = match newdata {
                None => (m.start().map(<[f64]>::to_vec), m.stop().to_vec()),
                Some(new) => {
                    let Some(stop) = new.stop else {
                        return Err(Error::input("newdata: the stop is required"));
                    };
                    check_lengths(&[("newdata x", rows), ("newdata stop", stop.len())])?;
                    refuse_non_finite(stop, "newdata stop")?;
                    match (m.has_start, new.start) {
                        (true, None) => {
                            return Err(Error::input(
                                "newdata: the fit has entry times, so the start is required",
                            ));
                        }
                        (false, Some(_)) => {
                            return Err(Error::input(
                                "newdata: the fit has no entry times, so the start is refused",
                            ));
                        }
                        _ => {}
                    }
                    if let Some(start) = new.start {
                        check_lengths(&[("newdata x", rows), ("newdata start", start.len())])?;
                        refuse_non_finite(start, "newdata start")?;
                        if start.iter().zip(stop).any(|(a, b)| a >= b) {
                            return Err(Error::input("newdata: a start is not before its stop"));
                        }
                    }
                    (new.start.map(<[f64]>::to_vec), stop.to_vec())
                }
            };
            let baseline = baseline_of(&m, opts.centring, opts.hazard)?;
            for i in 0..rows {
                centred_row(xm, i, &baseline.centre, &mut z);
                let eta = m.dot_beta(&z) + offset.map_or(0.0, |o| o[i] - baseline.offset_centre);
                let theta = eta.exp();
                if !theta.is_finite() {
                    return Err(Error::input("the risk score of a row is not finite"));
                }
                let begin = starts.as_ref().map_or(f64::NEG_INFINITY, |s| s[i]);
                let (increment, variance, k1, k2) =
                    expected_parts(&baseline, strata[i], begin, stops[i]);
                let value = increment * theta;
                let survival = (-value).exp();
                fit.push(if kind == PredictType::Survival {
                    survival
                } else {
                    value
                });
                if let Some(se) = se.as_mut() {
                    let s = &baseline.strata[strata[i]];
                    let mut g = vec![0.0; p];
                    for a in 0..p {
                        let upto = |k: usize| {
                            if k == 0 {
                                0.0
                            } else {
                                s.cumulative_xbar[(k - 1) * p + a]
                            }
                        };
                        g[a] = increment * z[a] - (upto(k2) - upto(k1));
                    }
                    let var = variance + quad(&m.variance, p, &g);
                    let s = theta * var.sqrt();
                    se.push(if kind == PredictType::Survival {
                        s * survival
                    } else {
                        s
                    });
                }
            }
        }
    }
    Ok(Prediction { fit, se })
}

/// The part of the linear predictor from each group of columns, as
/// `predict(type = "terms")`.
///
/// `groups` lists the column indices of each term. Aliased columns are left
/// out of a group. The default reference is `Sample`, and the offset is not
/// part of a term.
///
/// # Errors
///
/// [`Error::Input`] as for [`predict`], and when an index is not a column.
pub fn predict_terms(
    model: &FittedModel<'_>,
    groups: &[&[usize]],
    newdata: Option<&NewData<'_>>,
    opts: &PredictOptions<'_>,
) -> Result<TermsPrediction> {
    let m = Model::new(model, opts.variance)?;
    let p = m.p;
    if groups.iter().flat_map(|g| g.iter()).any(|&a| a >= p) {
        return Err(Error::input("groups: an index is not a column of x"));
    }
    let aliased: Vec<bool> = model.fit.coefficients.iter().map(Option::is_none).collect();
    let reference = opts.reference.unwrap_or(Reference::Sample);
    let (xm, rows, strata): (MatRef<'_, f64>, usize, Vec<usize>) = match newdata {
        None => (m.x, m.rows, m.stratum.iter().map(|&s| s as usize).collect()),
        Some(new) => {
            m.check_new(new)?;
            (new.x, new.x.nrows(), m.new_strata(new)?)
        }
    };
    let means = m.means(opts.centring)?;
    let stratum_means = if reference == Reference::Strata && m.has_strata() {
        Some(m.stratum_means()?)
    } else {
        None
    };
    let k = groups.len();
    let mut fit = Vec::with_capacity(rows * k);
    let mut se = opts.se_fit.then(|| Vec::with_capacity(rows * k));
    let mut centre = vec![0.0; p];
    let mut z = vec![0.0; p];
    for i in 0..rows {
        match reference {
            Reference::Zero => centre.fill(0.0),
            Reference::Sample => centre.copy_from_slice(&means),
            Reference::Strata => match &stratum_means {
                Some(all) => centre.copy_from_slice(&all[strata[i] * p..(strata[i] + 1) * p]),
                None => centre.copy_from_slice(&means),
            },
        }
        centred_row(xm, i, &centre, &mut z);
        for group in groups {
            let columns: Vec<usize> = group.iter().copied().filter(|&a| !aliased[a]).collect();
            let mut value = 0.0;
            for &a in &columns {
                value += z[a] * m.beta[a];
            }
            fit.push(value);
            if let Some(se) = se.as_mut() {
                let mut total = 0.0;
                for &a in &columns {
                    let mut row = 0.0;
                    for &b in &columns {
                        row += m.variance[a * p + b] * z[b];
                    }
                    total += z[a] * row;
                }
                se.push(total.sqrt());
            }
        }
    }
    let constant = (0..p)
        .filter(|&a| !aliased[a])
        .fold(0.0, |acc, a| acc + m.beta[a] * means[a]);
    Ok(TermsPrediction {
        fit,
        se,
        n_terms: k,
        constant,
    })
}

// -- survival curves

/// The limits of one curve point, `None` where undefined. The standard error
/// is that of `log S`.
fn limits(kind: ConfType, s: f64, se: f64, z: f64) -> (Option<f64>, Option<f64>) {
    let keep = |v: f64| v.is_finite().then_some(v);
    if kind == ConfType::None {
        return (None, None);
    }
    if se == 0.0 && kind != ConfType::Arcsin && kind != ConfType::Plain {
        return (Some(s), Some(s));
    }
    match kind {
        ConfType::None => (None, None),
        ConfType::Plain => {
            let width = se * s * z;
            (keep((s - width).max(0.0)), keep((s + width).min(1.0)))
        }
        ConfType::Log => {
            if s == 0.0 {
                return (None, None);
            }
            let l = s.ln();
            (keep((l - z * se).exp()), keep((l + z * se).exp().min(1.0)))
        }
        ConfType::LogLog => {
            if s <= 0.0 || s >= 1.0 {
                return (None, None);
            }
            let spread = z * se / s.ln();
            let l = (-s.ln()).ln();
            (
                keep((-(l - spread).exp()).exp()),
                keep((-(l + spread).exp()).exp()),
            )
        }
        ConfType::Logit => {
            if s == 0.0 {
                return (None, None);
            }
            let spread = z * se * (1.0 + s / (1.0 - s));
            let l = (s / (1.0 - s)).ln();
            (
                keep(1.0 - 1.0 / (1.0 + (l - spread).exp())),
                keep(1.0 - 1.0 / (1.0 + (l + spread).exp())),
            )
        }
        ConfType::Arcsin => {
            if s == 0.0 {
                return (None, None);
            }
            let spread = 0.5 * z * se * (s / (1.0 - s)).sqrt();
            let a = s.sqrt().asin();
            (
                keep((a - spread).max(0.0).sin().powi(2)),
                keep((a + spread).min(std::f64::consts::FRAC_PI_2).sin().powi(2)),
            )
        }
    }
}

/// Predicted survival curves for new covariate rows, as `survfit.coxph` with
/// `newdata`, at the event times of the stratum of each row.
///
/// `newdata.stratum` is required when the fit has strata, and the curve is
/// that of the row's stratum. For the curve of the average subject, pass the
/// row [`reference_means`] returns and the offset [`reference_offset`] returns. `newdata.start` and `stop` are not read.
///
/// # Errors
///
/// [`Error::Input`] as for [`predict`], and when `conf_level` is not in
/// `(0, 1)`.
pub fn survfit(
    model: &FittedModel<'_>,
    newdata: &NewData<'_>,
    opts: &SurvfitOptions<'_>,
) -> Result<SurvCurves> {
    if !(opts.conf_level > 0.0 && opts.conf_level < 1.0) {
        return Err(Error::input("conf_level: must be between 0 and 1"));
    }
    let m = Model::new(model, opts.variance)?;
    m.check_new(newdata)?;
    let strata = m.new_strata(newdata)?;
    let baseline = baseline_of(&m, opts.centring, opts.hazard)?;
    let p = m.p;
    let z_value = normal::quantile(1.0 - (1.0 - opts.conf_level) / 2.0);
    let mut z = vec![0.0; p];
    let mut g = vec![0.0; p];
    let mut curves = Vec::with_capacity(newdata.x.nrows());
    for (i, &s) in strata.iter().enumerate() {
        centred_row(newdata.x, i, &baseline.centre, &mut z);
        let eta = m.dot_beta(&z) + newdata.offset.map_or(0.0, |o| o[i]) - baseline.offset_centre;
        let theta = eta.exp();
        if !theta.is_finite() {
            return Err(Error::input("the risk score of a row is not finite"));
        }
        let base = &baseline.strata[s];
        let n = base.time.len();
        let cumulative_hazard: Vec<f64> =
            base.cumulative_hazard.iter().map(|h| h * theta).collect();
        let survival: Vec<f64> = cumulative_hazard.iter().map(|h| (-h).exp()).collect();
        let std_err = opts.se_fit.then(|| {
            (0..n)
                .map(|k| {
                    for a in 0..p {
                        g[a] = base.cumulative_hazard[k] * z[a] - base.cumulative_xbar[k * p + a];
                    }
                    ((base.cumulative_variance[k] + quad(&m.variance, p, &g)) * theta * theta)
                        .sqrt()
                })
                .collect::<Vec<f64>>()
        });
        let (lower, upper) = match (&std_err, opts.conf_type) {
            (Some(se), kind) if kind != ConfType::None => {
                let pairs: Vec<(Option<f64>, Option<f64>)> = (0..n)
                    .map(|k| limits(kind, survival[k], se[k], z_value))
                    .collect();
                (
                    Some(pairs.iter().map(|q| q.0).collect()),
                    Some(pairs.iter().map(|q| q.1).collect()),
                )
            }
            _ => (None, None),
        };
        curves.push(Curve {
            stratum: s,
            cumulative_hazard,
            survival,
            std_err,
            lower,
            upper,
        });
    }
    Ok(SurvCurves {
        strata: baseline
            .strata
            .into_iter()
            .map(|s| CurveTimes {
                code: s.code,
                time: s.time,
                n_risk: s.n_risk,
                n_event: s.n_events,
            })
            .collect(),
        curves,
    })
}
