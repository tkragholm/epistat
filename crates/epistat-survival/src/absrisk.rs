//! Absolute risk, the cumulative incidence of one cause, from cause-specific
//! Cox models, as `CSC` and `predict.CauseSpecificCox` of riskRegression.
//!
//! One Cox model is fitted for every cause, with the other causes as
//! censoring and a design matrix of its own ([`CauseSpecificCox::fit`], which
//! is [`cox::fit`](crate::cox::fit) once per cause). The absolute risk of a
//! cause `k` for a subject with covariates `x` is read from the baseline
//! hazards of all the models ([`CauseSpecificCox::predict`]).
//!
//! # The estimate
//!
//! With `theta_m = exp(x_m' beta_m)` the risk score of the subject in model
//! `m` (each model has its own columns) and `dL_m(s)` the increment of the
//! baseline cumulative hazard of model `m` at the event time `s`, the
//! cause-specific hazard of the subject is `theta_m dL_m(s)`. The grid of event
//! times `s_1 < s_2 < ...` is the union of the event times of all causes, and a
//! model has increment zero at a time of another cause. The absolute risk of
//! cause `k` at time `t` is
//!
//! ```text
//! F_k(t | x) = sum over s_j <= t of  S(s_j-) theta_k dL_k(s_j),
//! ```
//!
//! the sum of the probability of being event free just before `s_j`, times the
//! hazard of cause `k` there \[1, 2\]. The baseline increments are those of
//! [`cox::basehaz`](crate::cox::basehaz) for the covariates at zero, with the
//! form that follows the ties of the fit: the Fleming-Harrington form for
//! Efron ties and the Breslow form otherwise, as `predictCox` of riskRegression
//! does. Rows of weight zero are left out of the baselines, and case weights
//! enter them as in `basehaz`.
//!
//! The probability of being event free is, with `product_limit` (the default of
//! `predict.CauseSpecificCox`),
//!
//! ```text
//! S(s_j-) = product over l < j of ( 1 - sum over m of theta_m dL_m(s_l) ),
//! ```
//!
//! which is the Aalen-Johansen estimator with the model hazards, and without it
//! `S(s_j-) = exp(-sum over m of theta_m L_m(s_{j-1}))` with `L_m` the
//! cumulative baseline hazard. [`SurvType::Survival`] models the event-free
//! survival directly, as `CSC(surv.type = "survival")`. There are two Cox
//! models, one for the cause (other causes censored) and one for the time to
//! any event, and `S` is formed from the second alone: `1 - theta dL` or
//! `exp(-theta L)` with its baseline.
//!
//! The risk is `None` at a time after the last follow-up time of the stratum
//! of the subject, where the baselines are not defined, as `NA` in R. It is 0
//! before the first event. Where the factor `1 - sum theta_m dL_m` of the
//! product limit is negative, which a large risk score can make, `S` is not a
//! probability, and the risk from that time on is `None`.
//!
//! # The standard error
//!
//! With `std_err` the standard error is the square root of the sum of the
//! squares of the influence function of the estimate over the subjects of the
//! fit, the infinitesimal-jackknife variance. Every model contributes the
//! influence function of its coefficients and of its baseline hazard. For the
//! coefficients, the influence function of subject `i` is `V u_i`, with `V` the
//! covariance of the fit (the robust one when the fit has clusters) and `u_i`
//! the score residual of the Breslow partial likelihood at the estimate. For
//! the baseline, with `w_i` the case weight, `S0(s)` and `E(s)` the sum of
//! `w_i e_i` and the weighted mean of the covariates with those weights over
//! the risk set at `s`, `e_i = exp(x_i' beta)`, and `d_i` the event indicator
//! of the model, the influence function of the increment of the cumulative
//! hazard at `s` is
//!
//! ```text
//! IF_i(dL(s)) = - IF_i(beta)' E(s) dL(s)
//!               + w_i ( d_i 1(T_i = s) - e_i 1(a_i < s <= T_i) dL(s) ) / S0(s),
//! ```
//!
//! and that of the cumulative hazard is the sum of these over the event times
//! up to it. The risk set of a row with entry time `a_i` is the times in
//! `(a_i, T_i]`, as in `coxph` with counting-process data, and the influence
//! function of a row is its case weight times the derivative of the estimate
//! with respect to the weight, as in the score residual that `IF_i(beta)` is
//! made of. The influence function of `F_k(t | x)` follows from
//! `F_k = sum S theta_k dL_k` by the chain rule over the coefficients and the
//! baselines of every model, with the derivative of the event-free survival
//! `S(s_j-)` as follows.
//!
//! With the product limit, `S(s_j-) = product over l < j of (1 - h_l)` with
//! `h_l = sum over m of theta_m dL_m(s_l)`, and the derivative is
//!
//! ```text
//! d log S(s_j-) = - sum over l < j of d h_l / (1 - h_l),
//! ```
//!
//! so that every hazard increment of the subject enters with the weight
//! `1 / (1 - h_l)`. This is the derivative of the estimate, and it is the
//! default ([`InfluenceForm::Exact`]). Without the product limit
//! `S = exp(-sum theta_m L_m)`, the weights are 1, and that is exact. riskRegression
//! uses the weights 1 for the product limit as well, which is the derivative of
//! the exponential form and a first-order approximation of the exact one
//! ([`InfluenceForm::Exponential`]). The two differ by the order of the total
//! hazard `h_l`: on the test data, the standard errors of the exponential form
//! are off by up to 2 per cent relative to the exact ones at the times tested
//! and by up to 4 per cent on the simulated data of the standardise tests, as
//! the numerical derivative of the estimate with respect to the case weight of
//! each subject shows (`tests/standardise_checks.rs`). The exact derivative
//! is undefined where a factor `1 - h_l` is 0, and the standard error is
//! then `None` from the time after it.
//!
//! The influence function is evaluated by prefix sums over the event times, in
//! shared sums that [`standardise`](crate::standardise) uses as well: the cost for one new row is `n` per distinct time
//! of the result plus, for each model, the number of event times of the model
//! per distinct time of the result for the weights above, and the memory is
//! `O(n p + J)` per model for the training rows and `O(T J)` for the sums of
//! one row with `T` times, not the `n` by `J` matrices of riskRegression.
//!
//! The standard error is that of case weights as sampling weights: a person of
//! weight `w` is one unit of the influence function, `w` times the derivative
//! of the estimate with respect to the weight. Integer weights give the
//! estimates of repeated rows, and a standard error whose square is
//! `sum w_i^2 d_i^2` where the repeated rows give `sum w_i d_i^2`, with `d_i`
//! the derivative. The fit of the coefficients with tied times is the Efron one
//! when the options say so, and the influence function is that of the Breslow
//! likelihood, as in riskRegression. An offset enters the risk score of the rows of the fit and
//! of each new row ([`NewRows::offset`]), measured from the mean offset of the
//! fit that the baseline is at.
//!
//! # Confidence limits
//!
//! With `std_err` the limits are those of `confint.predictCSC`: on the scale
//! given by [`RiskLimits`], by default the log-log scale, `F^exp(+- z se / (F
//! |log F|))` with `z` the normal quantile of the level. Without a standard
//! error there are no limits.
//!
//! # Where this differs from riskRegression
//!
//! - The influence function of the coefficients of riskRegression
//!   (`iidCox`) counts a time with several tied events once in the sum over
//!   the risk set, where the score residual counts it for each event. It
//!   differs from `dfbeta` of `coxph` with a Breslow fit by more than the size
//!   of `dfbeta` on tied data and agrees with it to `1e-16` without ties (see
//!   `tests/data/absrisk.R`). The score residual is used here, so the
//!   standard errors agree with riskRegression's without tied times and
//!   differ with them.
//! - With the product limit the standard error is that of the exact derivative,
//!   not riskRegression's first-order one. [`InfluenceForm::Exponential`]
//!   gives riskRegression's number, and the tests against R use it.
//! - riskRegression's `predictCox` refuses case weights. Here the baselines
//!   and the standard error take them.
//! - With entry times (counting-process data) riskRegression's standard error
//!   is not the derivative of the estimate: it differs by up to 30 per cent at
//!   early times on the test data (`tests/data/influence_ext.R`), shrinking
//!   later. The standard error here is the derivative, checked against refitted
//!   numerical derivatives, and its risks agree with `predictCox`.
//! - A model with no covariate has no fit here (`cox::fit` needs a column).
//!   riskRegression then extrapolates the risk beyond the last follow-up time
//!   when the last row is an event, and here the risk is `None` there.
//! - The cause-specific models share one stratification, that of the data.
//!   `CSC` lets each model have its own formula and strata.
//! - Confidence bands, `landmark` and the average of the influence function
//!   are not provided.
//!
//! # Agreement with R
//!
//! The tests give the functions the coefficients and the covariance matrices
//! that `coxph` returned, so that the differences are those of the prediction
//! and not of the fit. The tolerances are derived in the test.
//!
//! # References
//!
//! 1. Benichou J, Gail MH. Estimates of absolute cause-specific risk in cohort
//!    studies. Biometrics 1990;46(3):813-26. doi:10.2307/2532100
//! 2. Ozenne B, Sorensen AL, Scheike T, Torp-Pedersen C, Gerds TA.
//!    riskRegression: predicting the risk of an event using Cox regression
//!    models. R Journal 2017;9(2):440-60. doi:10.32614/RJ-2017-062
//! 3. Fleming TR, Harrington DP. Nonparametric estimation of the survival
//!    distribution in censored data. Commun Stat Theory Methods
//!    1984;13(20):2469-86. doi:10.1080/03610928408828837
//! 4. Beyersmann J, Allignol A, Schumacher M. Competing risks and multistate
//!    models with R. New York: Springer; 2012.

use epistat_linalg::MatRef;
use epistat_stats::normal;
use rayon::prelude::*;

use crate::check::{check_lengths, refuse_non_finite};
use crate::cox::{CoxModelFit, CoxOptions, Ties};
use crate::error::{Error, Result};
use crate::surv::{SurvBuf, SurvData};
use crate::timefix::TimeFix;

pub(crate) mod influence;
mod limits;

pub use influence::InfluenceForm;
use influence::{Baselines, Influence, ModelInput, Path, Shape};
pub use limits::RiskLimits;

/// What the second model of `CSC` is, `surv.type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurvType {
    /// One model for every cause. The event-free survival is formed from all
    /// the cause-specific hazards.
    #[default]
    Hazard,
    /// Two models: one for the given cause, with the other causes censored,
    /// and one for the time to any event, from which the event-free survival is
    /// formed. Risks can be predicted for this cause only.
    Survival {
        /// The event code of the cause.
        cause: u32,
    },
}

/// The options of [`CauseSpecificCox::fit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CscOptions<'a> {
    /// How tied events enter the partial likelihood, and which baseline
    /// hazard follows. `Efron`, as in `coxph`, which `CSC` calls.
    pub ties: Ties,
    /// The rule that makes times equal when they differ by rounding error,
    /// as in [`CoxOptions`].
    pub timefix: TimeFix,
    /// What the models are, `surv.type`. `Hazard`, as in `CSC`.
    pub surv_type: SurvType,
    /// A term added to the linear predictor of every row of the data in every
    /// model, as `offset()` in the formulas of `CSC`; one value per row of the
    /// data. The default is no offset. A new row predicted for has the offset
    /// of [`NewRows::offset`].
    pub offset: Option<&'a [f64]>,
}

impl Default for CscOptions<'_> {
    fn default() -> Self {
        Self {
            ties: Ties::Efron,
            timefix: TimeFix::default(),
            surv_type: SurvType::Hazard,
            offset: None,
        }
    }
}

/// Cause-specific Cox models with the data and the designs they were fitted
/// on.
///
/// With [`SurvType::Hazard`] there is one model per event code of the data, in
/// the order of the codes. With [`SurvType::Survival`] the first is the model of
/// the cause and the second the model of the time to any event.
#[derive(Debug)]
pub struct CauseSpecificCox<'a> {
    data: SurvData<'a>,
    overall: Option<SurvBuf>,
    designs: Vec<MatRef<'a, f64>>,
    fits: Vec<CoxModelFit>,
    options: Vec<CoxOptions<'a>>,
    surv_type: SurvType,
}

/// The new rows to predict for.
#[derive(Debug, Clone, Copy)]
pub struct NewRows<'a> {
    /// The covariates of each model, in the order of the models, one row per
    /// new row. The columns follow the design the model was fitted on.
    pub designs: &'a [MatRef<'a, f64>],
    /// The caller's stratum code of each row, required when the data has
    /// strata and not read otherwise.
    pub stratum: Option<&'a [u32]>,
    /// The offset of each new row, in every model, as `offset()` of the formula
    /// has it for new data. Zero when absent. It is an error for models fitted
    /// without an offset.
    pub offset: Option<&'a [f64]>,
}

/// The options of [`CauseSpecificCox::predict`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskOptions {
    /// Whether the event-free survival is the product limit of one minus the
    /// hazard, as `product.limit` of `predict.CauseSpecificCox`, whose default
    /// this is, or the exponential of minus the cumulative hazard.
    pub product_limit: bool,
    /// Whether to give standard errors, `se`, from the influence functions of
    /// the models, see the module documentation.
    pub std_err: bool,
    /// How the standard error takes the derivative of the product limit.
    /// `Exact`, the derivative of the estimate. `InfluenceForm::Exponential`
    /// is riskRegression's, see the module documentation. It is not read
    /// without `product_limit` or `std_err`.
    pub influence: InfluenceForm,
    /// The scale of the confidence limits, which exist with `std_err`.
    pub limits: RiskLimits,
    /// The confidence level, in `(0, 1)`.
    pub conf_level: f64,
}

impl Default for RiskOptions {
    fn default() -> Self {
        Self {
            product_limit: true,
            std_err: false,
            influence: InfluenceForm::Exact,
            limits: RiskLimits::LogLog,
            conf_level: 0.95,
        }
    }
}

/// The absolute risks of [`CauseSpecificCox::predict`].
///
/// Every table has `n_rows` rows and `n_times` columns, stored by row. A value
/// is `None` after the last follow-up time of the stratum of the row and where
/// it is undefined.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsoluteRisk {
    /// The number of new rows.
    pub n_rows: usize,
    /// The number of times.
    pub n_times: usize,
    /// The absolute risk of the cause.
    pub risk: Vec<Option<f64>>,
    /// The standard error, with `std_err`.
    pub std_err: Option<Vec<Option<f64>>>,
    /// The lower confidence limit, with `std_err` and limits other than
    /// `RiskLimits::None`.
    pub lower: Option<Vec<Option<f64>>>,
    /// The upper confidence limit.
    pub upper: Option<Vec<Option<f64>>>,
}

impl<'a> CauseSpecificCox<'a> {
    /// Fit the cause-specific models, as `CSC`.
    ///
    /// `designs` has one matrix per event code of `data` for
    /// [`SurvType::Hazard`], the matrix of code `k` in position `k - 1`, and two
    /// for [`SurvType::Survival`], the cause and then the time to any event.
    /// Each matrix has one row per row of `data` and a column per covariate, and
    /// no column for the stratum, which is that of `data`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] as for [`cox::fit`](crate::cox::fit), for a number of
    /// designs other than the models need, for a data set with a single event
    /// code, and for a cause outside the codes.
    pub fn fit(
        data: &SurvData<'a>,
        designs: &[MatRef<'a, f64>],
        opts: &CscOptions<'a>,
    ) -> Result<Self> {
        let plan = Plan::new(data, designs, opts)?;
        let mut fits = Vec::with_capacity(plan.options.len());
        for (m, options) in plan.options.iter().enumerate() {
            let fit = match (&plan.overall, m) {
                (Some(buf), 1) => crate::cox::fit(&buf.as_data(), designs[m], options)?,
                _ => crate::cox::fit(&plan.data, designs[m], options)?,
            };
            fits.push(fit);
        }
        Ok(plan.into_model(designs, fits))
    }

    /// The models from fits made elsewhere, for example from
    /// [`cox::fit`](crate::cox::fit), with coefficients and covariances that
    /// are not the ones [`fit`](Self::fit) would give.
    ///
    /// `fits` is in the order of the designs, as for [`fit`](Self::fit), and
    /// each must have been fitted on `data` and its design with `opts`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] as for [`fit`](Self::fit), and for a number of fits
    /// other than the number of designs.
    pub fn from_fits(
        data: &SurvData<'a>,
        designs: &[MatRef<'a, f64>],
        fits: Vec<CoxModelFit>,
        opts: &CscOptions<'a>,
    ) -> Result<Self> {
        let plan = Plan::new(data, designs, opts)?;
        check_lengths(&[("designs", designs.len()), ("fits", fits.len())])?;
        Ok(plan.into_model(designs, fits))
    }

    /// The fit of each model, in the order of the designs.
    #[must_use]
    pub fn fits(&self) -> &[CoxModelFit] {
        &self.fits
    }

    /// Predict the absolute risk of `cause` at `times` for the new rows, as
    /// `predict.CauseSpecificCox` with `type = "absRisk"`.
    ///
    /// `times` need not be sorted and may repeat. The result has one column per
    /// time in the order given. See the module documentation for the estimate,
    /// the standard error and the limits.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `cause` is outside the codes of the data or, for
    /// [`SurvType::Survival`], is not the cause of the fit, when a time or a new
    /// covariate is not finite, when the new designs do not match the models in
    /// number or in columns, when a new stratum is not one of the data or is
    /// missing for data with strata, when a model has a coefficient that is
    /// `None` (drop the aliased column), when a risk score is not finite, when
    /// `conf_level` is not in `(0, 1)`, and with `std_err` for a covariance
    /// that is not defined.
    pub fn predict(
        &self,
        new: &NewRows<'_>,
        times: &[f64],
        cause: u32,
        opts: &RiskOptions,
    ) -> Result<AbsoluteRisk> {
        let ctx = self.context(new, times, cause, opts)?;
        let n_rows = new.designs[0].nrows();
        let n_times = times.len();
        let influence = if opts.std_err {
            Some(Influence::new(&self.inputs(), &ctx.base)?)
        } else {
            None
        };
        let per_row: Vec<Row> = (0..n_rows)
            .into_par_iter()
            .map(|r| ctx.row(r, new, times, influence.as_ref()))
            .collect();
        let mut risk = Vec::with_capacity(n_rows * n_times);
        let mut std_err = opts.std_err.then(|| Vec::with_capacity(n_rows * n_times));
        for row in &per_row {
            risk.extend_from_slice(&row.risk);
            if let Some(out) = std_err.as_mut() {
                out.extend_from_slice(&row.std_err);
            }
        }
        let (lower, upper) = match &std_err {
            Some(se) if opts.limits != RiskLimits::None => {
                let z = normal::two_sided_z(opts.conf_level);
                let (lo, up) = limits::limits(&risk, se, opts.limits, z);
                (Some(lo), Some(up))
            }
            _ => (None, None),
        };
        Ok(AbsoluteRisk {
            n_rows,
            n_times,
            risk,
            std_err,
            lower,
            upper,
        })
    }
}

/// The checked inputs of a fit.
struct Plan<'a> {
    data: SurvData<'a>,
    overall: Option<SurvBuf>,
    options: Vec<CoxOptions<'a>>,
    surv_type: SurvType,
}

impl<'a> Plan<'a> {
    fn new(
        data: &SurvData<'a>,
        designs: &[MatRef<'a, f64>],
        opts: &CscOptions<'a>,
    ) -> Result<Self> {
        let fixed = data.clone().with_timefix(opts.timefix)?;
        let n_codes = fixed.followup()?.n_codes();
        if n_codes < 2 {
            return Err(Error::input(
                "data: absolute risk needs competing causes, so the status needs at least two codes",
            ));
        }
        let expected = match opts.surv_type {
            SurvType::Hazard => n_codes as usize,
            SurvType::Survival { cause } => {
                if cause == 0 || cause > n_codes {
                    return Err(Error::input(format!(
                        "cause {cause} is outside 1..={n_codes}"
                    )));
                }
                2
            }
        };
        check_lengths(&[("models", expected), ("designs", designs.len())])?;
        if let Some(offset) = opts.offset {
            check_lengths(&[("data", fixed.n_rows()), ("offset", offset.len())])?;
        }
        let base = CoxOptions {
            ties: opts.ties,
            timefix: opts.timefix,
            offset: opts.offset,
            ..CoxOptions::default()
        };
        let (overall, options) = match opts.surv_type {
            SurvType::Hazard => (
                None,
                (1..=n_codes)
                    .map(|code| CoxOptions {
                        cause: Some(code),
                        ..base
                    })
                    .collect(),
            ),
            SurvType::Survival { cause } => {
                let buf = overall_data(&fixed)?;
                let any = CoxOptions {
                    cause: None,
                    // The times of the buffer are fixed already.
                    timefix: TimeFix::Exact,
                    ..base
                };
                (
                    Some(buf),
                    vec![
                        CoxOptions {
                            cause: Some(cause),
                            ..base
                        },
                        any,
                    ],
                )
            }
        };
        Ok(Self {
            data: fixed,
            overall,
            options,
            surv_type: opts.surv_type,
        })
    }

    fn into_model(
        self,
        designs: &[MatRef<'a, f64>],
        fits: Vec<CoxModelFit>,
    ) -> CauseSpecificCox<'a> {
        CauseSpecificCox {
            data: self.data,
            overall: self.overall,
            designs: designs.to_vec(),
            fits,
            options: self.options,
            surv_type: self.surv_type,
        }
    }
}

/// The data with every event as the event, for the model of the time to any
/// event.
pub(crate) fn overall_data(data: &SurvData<'_>) -> Result<SurvBuf> {
    let surv = data.followup()?;
    let status: Vec<i32> = surv.status().iter().map(|&s| i32::from(s != 0)).collect();
    let mut builder = SurvBuf::builder(surv.stop().to_vec(), status);
    if let Some(start) = surv.start() {
        builder = builder.start(start.to_vec());
    }
    if let Some(weight) = data.weight() {
        builder = builder.weight(weight.to_vec());
    }
    if let Some(stratum) = data.stratum() {
        builder = builder.stratum(stratum.to_vec());
    }
    if let Some(cluster) = data.cluster() {
        builder = builder.cluster(cluster.to_vec());
    }
    builder.build()
}

/// Everything the prediction of the rows reads.
struct Context {
    base: Baselines,
    /// Which model is the cause, and the models that form the survival.
    cause_model: usize,
    survival_models: Vec<usize>,
    /// The dense stratum of each new row.
    row_stratum: Vec<usize>,
    /// The risk score of each new row in each model, row-major.
    theta: Vec<f64>,
    product_limit: bool,
    form: InfluenceForm,
}

/// One new row: the risks and the standard errors at the times given.
struct Row {
    risk: Vec<Option<f64>>,
    std_err: Vec<Option<f64>>,
}

impl CauseSpecificCox<'_> {
    fn model_data(&self, m: usize) -> SurvData<'_> {
        match (&self.overall, m) {
            (Some(buf), 1) => buf.as_data(),
            _ => self.data.clone(),
        }
    }

    /// The models as the baselines and the influence functions read them.
    fn inputs(&self) -> Vec<ModelInput<'_>> {
        (0..self.fits.len())
            .map(|m| ModelInput {
                data: self.model_data(m),
                x: self.designs[m],
                fit: &self.fits[m],
                options: self.options[m],
            })
            .collect()
    }

    fn context(
        &self,
        new: &NewRows<'_>,
        times: &[f64],
        cause: u32,
        opts: &RiskOptions,
    ) -> Result<Context> {
        if !(opts.conf_level > 0.0 && opts.conf_level < 1.0) {
            return Err(Error::input("conf_level: must be in (0, 1)"));
        }
        refuse_non_finite(times, "times")?;
        let n_codes = self.data.followup()?.n_codes();
        if cause == 0 || cause > n_codes {
            return Err(Error::input(format!(
                "cause {cause} is outside 1..={n_codes}"
            )));
        }
        let (cause_model, survival_models) = match self.surv_type {
            SurvType::Hazard => ((cause - 1) as usize, (0..self.fits.len()).collect()),
            SurvType::Survival { cause: fitted } => {
                if cause != fitted {
                    return Err(Error::input(format!(
                        "cause {cause}: the models were fitted for cause {fitted}"
                    )));
                }
                (0, vec![1])
            }
        };
        check_lengths(&[
            ("models", self.fits.len()),
            ("new designs", new.designs.len()),
        ])?;
        let rows = new.designs[0].nrows();
        for (m, d) in new.designs.iter().enumerate() {
            if d.nrows() != rows {
                return Err(Error::input(format!(
                    "new designs: model {m} has {} rows against {rows}",
                    d.nrows()
                )));
            }
            if d.ncols() != self.designs[m].ncols() {
                return Err(Error::input(format!(
                    "new designs: model {m} has {} columns, the model has {}",
                    d.ncols(),
                    self.designs[m].ncols()
                )));
            }
            for j in 0..d.ncols() {
                for i in 0..rows {
                    if !d[(i, j)].is_finite() {
                        return Err(Error::input("new designs: a value is not finite"));
                    }
                }
            }
        }
        let levels = self.data.stratum_levels().map(|l| l.to_vec());
        if levels.is_some() {
            match new.stratum {
                Some(codes) => {
                    check_lengths(&[("new designs", rows), ("new stratum", codes.len())])?;
                }
                None => {
                    return Err(Error::input("new stratum: required, the data has strata"));
                }
            }
        }
        // The baselines at zero covariates, as predictCox(centered = FALSE).
        let base = Baselines::new(&self.inputs())?;
        let row_stratum = match (&levels, new.stratum) {
            (Some(levels), Some(codes)) => codes
                .iter()
                .map(|code| {
                    levels.binary_search(code).map_err(|_| {
                        Error::input(format!("new stratum: {code} is not a stratum of the data"))
                    })
                })
                .collect::<Result<Vec<usize>>>()?,
            _ => vec![0; rows],
        };
        if let Some(offset) = new.offset {
            check_lengths(&[("new designs", rows), ("new offset", offset.len())])?;
            refuse_non_finite(offset, "new offset")?;
            if self.options[0].offset.is_none() {
                return Err(Error::input(
                    "new offset: the models were fitted without an offset",
                ));
            }
        }
        let mut theta = Vec::with_capacity(rows * base.models.len());
        for r in 0..rows {
            for (m, model) in base.models.iter().enumerate() {
                let mut eta = 0.0;
                for (a, b) in model.beta.iter().enumerate() {
                    eta += new.designs[m][(r, a)] * b;
                }
                if self.options[0].offset.is_some() {
                    eta += new.offset.map_or(0.0, |o| o[r]) - model.offset_centre;
                }
                let value = eta.exp();
                if !value.is_finite() {
                    return Err(Error::input(format!(
                        "new designs: the risk score of row {r} in model {m} is not finite"
                    )));
                }
                theta.push(value);
            }
        }
        Ok(Context {
            base,
            cause_model,
            survival_models,
            row_stratum,
            theta,
            product_limit: opts.product_limit,
            form: opts.influence,
        })
    }
}

impl Context {
    fn n_models(&self) -> usize {
        self.base.models.len()
    }

    fn theta_of(&self, r: usize) -> &[f64] {
        &self.theta[r * self.n_models()..(r + 1) * self.n_models()]
    }

    fn row(
        &self,
        r: usize,
        new: &NewRows<'_>,
        times: &[f64],
        influence: Option<&Influence>,
    ) -> Row {
        let s = self.row_stratum[r];
        let grid = &self.base.grid;
        let path = self.base.path(
            self.theta_of(r),
            s,
            grid.len(),
            &self.survival_models,
            self.cause_model,
            self.product_limit,
            influence.is_some() && self.form == InfluenceForm::Exact,
        );
        let counts: Vec<Option<usize>> = times
            .iter()
            .map(|&t| (t <= self.base.last_time[s]).then(|| grid.partition_point(|&g| g <= t)))
            .collect();
        let risk: Vec<Option<f64>> = counts
            .iter()
            .map(|c| match *c {
                Some(0) => Some(0.0),
                Some(count) if count <= path.valid => Some(path.cif[count - 1]),
                _ => None,
            })
            .collect();
        let std_err = influence.map_or_else(Vec::new, |inf| {
            let x: Vec<Vec<f64>> = new
                .designs
                .iter()
                .map(|d| (0..d.ncols()).map(|a| d[(r, a)]).collect())
                .collect();
            let mut needed: Vec<usize> = counts
                .iter()
                .flatten()
                .copied()
                .filter(|&c| c > 0 && c <= path.std_valid)
                .collect();
            needed.sort_unstable();
            needed.dedup();
            let values = self.row_std_err(inf, s, self.theta_of(r), &x, &path, &needed);
            counts
                .iter()
                .map(|c| match *c {
                    Some(0) => Some(0.0),
                    Some(count) if count <= path.std_valid => {
                        needed.binary_search(&count).ok().map(|i| values[i])
                    }
                    _ => None,
                })
                .collect()
        });
        Row { risk, std_err }
    }

    /// The standard error of the risk of the row at each of the positive
    /// counts `needed` of grid times, sorted: the square root of the sum of the
    /// squares of the influence function over the training rows.
    fn row_std_err(
        &self,
        inf: &Influence,
        s: usize,
        theta: &[f64],
        x: &[Vec<f64>],
        path: &Path,
        needed: &[usize],
    ) -> Vec<f64> {
        if needed.is_empty() {
            return Vec::new();
        }
        let shape = Shape::new(
            &self.base,
            inf,
            self.cause_model,
            &self.survival_models,
            needed,
        );
        let mut acc = shape.new_acc();
        let mut scratch = shape.new_scratch();
        shape.add_row(&mut acc, 1.0, s, path, theta, x, &mut scratch);
        shape.finish(&mut acc);
        (0..needed.len())
            .map(|u| {
                let mut total = 0.0;
                for i in 0..inf.count_le.len() {
                    let value = shape.model_part(&acc, i, u);
                    total += value * value;
                }
                total.sqrt()
            })
            .collect()
    }
}
