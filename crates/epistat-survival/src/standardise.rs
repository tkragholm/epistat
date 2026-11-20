//! Standardised risks and average treatment effects by the g-formula, with
//! influence-function standard errors, as `ate` of riskRegression with
//! `estimator = "GFORMULA"`, and population-averaged predictions and
//! contrasts of a fitted model as `yates` of survival ([`yates`]).
//!
//! # The estimate
//!
//! An outcome model is fitted to the population: one Cox model
//! ([`standardise_cox`]) or cause-specific Cox models
//! ([`standardise_cause_specific`]). For each level `a` of the exposure the
//! caller gives the covariates of every person with the exposure set to `a`
//! ([`Level`]). The model gives each person a risk of the event by time `t`,
//! `F(t | x_i(a))`, and the standardised risk is the mean over the persons,
//!
//! ```text
//! F_a(t) = sum over i of w_i F(t | x_i(a)) / sum over i of w_i,
//! ```
//!
//! with `w_i` the case weight, so that it is the risk the whole population would
//! have had if everybody had the exposure `a` \[1, 2\]. For a Cox model
//! `F = 1 - exp(-theta L(t))`, with `theta = exp(x' beta)` and `L` the
//! baseline cumulative hazard of the stratum of the person, as `predictCox`
//! of riskRegression. For cause-specific models `F` is the absolute risk of
//! [`absrisk`](crate::absrisk) of the cause asked. A risk is `None` after the
//! last follow-up time of any stratum of the population, and where the risk of
//! a person is `None`.
//!
//! The contrasts between two levels are the risk difference `F_a - F_b` and the
//! risk ratio `F_a / F_b` ([`Standardised::contrast`]).
//!
//! # The standard error
//!
//! The estimate depends on the data twice: through the outcome model and
//! through the empirical distribution of the covariates over which the risks
//! are averaged. With `n` persons of weight one and the same persons as the
//! population, the influence function of `F_a(t)` for person `i` is
//!
//! ```text
//! IF_i = (1/n) sum over l of IF_i( F(t | x_l(a)) )  +  (F(t | x_i(a)) - F_a(t)) / n,
//! ```
//!
//! where the first term is the influence function of the risk of one covariate
//! row, as [`absrisk`](crate::absrisk) forms it, averaged over the rows, and the
//! second is the influence function of the mean \[2, 3\]. The standard error is
//! the square root of `sum_i IF_i^2`, and the covariance of the standardised
//! risks of two levels is `sum_i IF_{a,i} IF_{b,i}`, so that the standard error
//! of the difference is that of `IF_a - IF_b`. The ratio has the influence
//! function `IF_a / F_b - F_a IF_b / F_b^2`, the delta method, and
//! the limits are normal, `estimate +- z se`, as in `ate`. A p-value is that
//! of the Wald test of a difference of 0 and of a ratio of 1.
//!
//! The influence function of a risk is linear in running sums over the grid
//! of event times whose coefficients depend on the row only through its
//! risk path and covariates. The average over the rows is taken on those sums,
//! once per level, so that the influence function of the `n` persons is formed
//! from them in a second pass without an `n` by `n` product. For
//! cause-specific models the cost is `O(n J p)` for the risk paths of the
//! persons, done twice, and `O(n T J_m)` for the weights of the baseline
//! increments, with `T` the number of distinct times and `J_m` the number of
//! event times of a model, plus `O(n p)` per distinct time and level for the
//! influence; the memory is `O(J p + T J_m)` per level and stratum, not
//! `O(n J)`. Sums over persons are formed in blocks of a fixed size and added in
//! order, so that the bits do not depend on the number of threads.
//!
//! The influence functions of the outcome models are those of
//! [`absrisk`](crate::absrisk): for data without entry times, case weights of 1
//! and no offset, with the score residuals of the Breslow likelihood and the
//! baseline form of the fit, so that the standard errors differ from
//! riskRegression's with tied event times in the way described there.
//!
//! For cause-specific models with the product limit the influence function is
//! that of the product-limit survival itself, as described in
//! [`absrisk`](crate::absrisk) ([`InfluenceForm::Exact`], the default).
//! riskRegression takes the derivative of the exponential form instead, a
//! first-order approximation. [`InfluenceForm::Exponential`] gives its numbers:
//! on the cause-specific cases of the reference its standard errors are off by
//! up to 0.5 per cent from the exact ones, and by up to 4 per cent on the
//! simulated data of the tests, where the numerical derivative of the estimate
//! with respect to the case weight of each person agrees with the exact form.
//!
//! # Where this differs from riskRegression
//!
//! - The exposure is not a factor of the data frame. The caller gives the
//!   covariates under each level, either by setting columns of the design to
//!   constants ([`Counterfactual::Columns`]) or as a full design matrix
//!   ([`Counterfactual::Design`]), which allows interactions and
//!   transformations of the exposure.
//! - Only the g-formula estimator is provided, with the population of the
//!   fitted data. Inverse probability weighting and augmented estimators are
//!   in [`weighting`](crate::weighting).
//! - With the product limit the standard error is that of the exact derivative
//!   of the estimate, not riskRegression's first-order one, unless
//!   [`InfluenceForm::Exponential`] is chosen.
//! - Case weights enter the estimate as weights of the persons averaged,
//!   `ate` has none. The standard error refuses them.
//! - Cause-specific models with `surv.type = "survival"` are not provided.
//!
//! # References
//!
//! 1. Robins J. A new approach to causal inference in mortality studies with a
//!    sustained exposure period. Math Model 1986;7(9-12):1393-512.
//!    doi:10.1016/0270-0255(86)90088-6
//! 2. Ozenne BMH, Scheike TH, Staerk L, Gerds TA. On the estimation of average
//!    treatment effects with right-censored time to event outcome and competing
//!    risks. Biom J 2020;62(3):751-63. doi:10.1002/bimj.201800298
//! 3. Sjolander A, Dahlqwist E. stdReg: an R package for estimating
//!    standardized effects. J Stat Softw 2021;99(6):1-20.
//!    doi:10.18637/jss.v099.i06

use std::ops::Range;

use epistat_linalg::MatRef;
use epistat_stats::normal;

use crate::absrisk::{CscOptions, SurvType};
use crate::check::{check_lengths, refuse_non_finite};
use crate::cox::{CoxModelFit, CoxOptions};
use crate::error::{Error, Result};
use crate::par;
use crate::surv::SurvData;

mod cox;
mod csc;
pub mod yates;

pub use crate::absrisk::InfluenceForm;
use crate::absrisk::influence::{Baselines, Influence, ModelInput};

/// The covariates of one model with the exposure set to one level.
#[derive(Debug, Clone, Copy)]
pub enum Counterfactual<'a> {
    /// The design the model was fitted on, with these columns set to these
    /// values for every row.
    Columns(&'a [(usize, f64)]),
    /// A design matrix with one row per row of the data and the columns of the
    /// model.
    Design(MatRef<'a, f64>),
}

/// One level of the exposure: the covariates of every model under it.
#[derive(Debug, Clone)]
pub struct Level<'a> {
    /// One entry per model, in the order of the models.
    pub designs: Vec<Counterfactual<'a>>,
}

impl<'a> Level<'a> {
    /// A level for a single model.
    #[must_use]
    pub fn single(design: Counterfactual<'a>) -> Self {
        Self {
            designs: vec![design],
        }
    }
}

/// The options of [`standardise_cox`] and [`standardise_cause_specific`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardiseOptions {
    /// Whether to give standard errors and limits. It is an error for data with
    /// case weights other than 1, entry times or an offset.
    pub std_err: bool,
    /// The confidence level of the limits, in `(0, 1)`.
    pub conf_level: f64,
    /// Whether the event-free survival of cause-specific models is the product
    /// limit, as `product.limit` of `predict.CauseSpecificCox`, or the
    /// exponential of minus the cumulative hazard. Not read for one Cox model.
    pub product_limit: bool,
    /// How the standard error takes the derivative of the product limit of
    /// cause-specific models, see [`InfluenceForm`]. Not read for one Cox model
    /// or without `product_limit`.
    pub influence: InfluenceForm,
}

impl Default for StandardiseOptions {
    fn default() -> Self {
        Self {
            std_err: false,
            conf_level: 0.95,
            product_limit: true,
            influence: InfluenceForm::Exact,
        }
    }
}

/// The standardised risks of every level at every time.
///
/// The tables hold `n_levels` rows of `n_times` values, stored by level. A
/// value is `None` where it is not defined.
#[derive(Debug, Clone, PartialEq)]
pub struct Standardised {
    /// The number of levels.
    pub n_levels: usize,
    /// The number of times.
    pub n_times: usize,
    /// The rows of the data.
    pub n_rows: usize,
    /// The rows of positive weight, the population.
    pub n_used: usize,
    /// The standardised risk.
    pub risk: Vec<Option<f64>>,
    /// The standard error, with `std_err`.
    pub std_err: Option<Vec<Option<f64>>>,
    /// The lower normal limit, with `std_err`.
    pub lower: Option<Vec<Option<f64>>>,
    /// The upper normal limit, with `std_err`.
    pub upper: Option<Vec<Option<f64>>>,
    /// The covariance of the standardised risks of the levels at each time,
    /// `n_times` matrices of `n_levels` by `n_levels`, with `std_err`.
    pub covariance: Option<Vec<Option<f64>>>,
    /// The confidence level of the limits.
    pub conf_level: f64,
}

/// The risk difference and the risk ratio of two levels at every time.
#[derive(Debug, Clone, PartialEq)]
pub struct Contrast {
    /// The risk of the first level minus that of the second.
    pub difference: Vec<Option<f64>>,
    /// The risk of the first level over that of the second, `None` where the
    /// second is 0.
    pub ratio: Vec<Option<f64>>,
    /// Standard errors, limits and p-values, with `std_err`.
    pub difference_inference: Option<Inference>,
    /// As `difference_inference` for the ratio, tested against 1.
    pub ratio_inference: Option<Inference>,
}

/// The standard error, limits and p-value of a contrast at every time.
#[derive(Debug, Clone, PartialEq)]
pub struct Inference {
    /// The standard error.
    pub std_err: Vec<Option<f64>>,
    /// The lower normal limit.
    pub lower: Vec<Option<f64>>,
    /// The upper normal limit.
    pub upper: Vec<Option<f64>>,
    /// The two-sided p-value of the Wald test of no effect.
    pub p_value: Vec<Option<f64>>,
}

impl Standardised {
    /// The Wald test that the standardised risks of the levels are equal at time
    /// `k`, with the covariance of the risks of the levels from the influence
    /// function, as the tests of [`yates`]. The global test is of all levels, the
    /// others one per pair or per level against the mean.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] without `std_err`, for a time that does not exist or
    /// where a risk is not defined, and for fewer than two levels.
    pub fn tests(&self, k: usize, kind: yates::Test) -> Result<Vec<yates::WaldTest>> {
        let Some(cov) = &self.covariance else {
            return Err(Error::input("tests: the result has no standard errors"));
        };
        if k >= self.n_times {
            return Err(Error::input(format!("time {k} is outside the times")));
        }
        let l = self.n_levels;
        let estimate: Option<Vec<f64>> = (0..l).map(|a| self.risk[a * self.n_times + k]).collect();
        let block: Option<Vec<f64>> = cov[k * l * l..(k + 1) * l * l].iter().copied().collect();
        match (estimate, block) {
            (Some(e), Some(b)) => yates::wald_tests(&e, &b, kind),
            _ => Err(Error::input(format!("time {k}: a risk is not defined"))),
        }
    }

    /// The risk difference and the risk ratio of level `first` against level
    /// `second`, as `diffRisk` and `ratioRisk` of `ate`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for a level that does not exist.
    pub fn contrast(&self, first: usize, second: usize) -> Result<Contrast> {
        let l = self.n_levels;
        if first >= l || second >= l {
            return Err(Error::input(format!(
                "levels: {first} and {second} must be below {l}"
            )));
        }
        let t = self.n_times;
        let at = |level: usize, k: usize| self.risk[level * t + k];
        let mut difference = Vec::with_capacity(t);
        let mut ratio = Vec::with_capacity(t);
        for k in 0..t {
            if let (Some(a), Some(b)) = (at(first, k), at(second, k)) {
                difference.push(Some(a - b));
                ratio.push((b != 0.0).then(|| a / b));
            } else {
                difference.push(None);
                ratio.push(None);
            }
        }
        let (difference_inference, ratio_inference) = match &self.covariance {
            Some(cov) => {
                let z = normal::two_sided_z(self.conf_level);
                let mut d_se = Vec::with_capacity(t);
                let mut r_se = Vec::with_capacity(t);
                for k in 0..t {
                    let c = |a: usize, b: usize| cov[k * l * l + a * l + b];
                    let (Some(va), Some(vb), Some(vab)) =
                        (c(first, first), c(second, second), c(first, second))
                    else {
                        d_se.push(None);
                        r_se.push(None);
                        continue;
                    };
                    d_se.push(Some((va - 2.0 * vab + vb).max(0.0).sqrt()));
                    r_se.push(match (at(first, k), at(second, k)) {
                        (Some(fa), Some(fb)) if fb != 0.0 => {
                            let v = va / (fb * fb) - 2.0 * fa * vab / (fb * fb * fb)
                                + fa * fa * vb / (fb * fb * fb * fb);
                            Some(v.max(0.0).sqrt())
                        }
                        _ => None,
                    });
                }
                (
                    Some(inference(&difference, &d_se, 0.0, z)),
                    Some(inference(&ratio, &r_se, 1.0, z)),
                )
            }
            None => (None, None),
        };
        Ok(Contrast {
            difference,
            ratio,
            difference_inference,
            ratio_inference,
        })
    }
}

fn inference(estimate: &[Option<f64>], se: &[Option<f64>], null: f64, z: f64) -> Inference {
    let mut lower = Vec::with_capacity(estimate.len());
    let mut upper = Vec::with_capacity(estimate.len());
    let mut p_value = Vec::with_capacity(estimate.len());
    for (e, s) in estimate.iter().zip(se) {
        if let (Some(e), Some(s)) = (e, s) {
            lower.push(Some(e - z * s));
            upper.push(Some(e + z * s));
            p_value.push((*s > 0.0).then(|| 2.0 * normal::cdf(-((e - null) / s).abs())));
        } else {
            lower.push(None);
            upper.push(None);
            p_value.push(None);
        }
    }
    Inference {
        std_err: se.to_vec(),
        lower,
        upper,
        p_value,
    }
}

/// The standardised risk of the event of one Cox model, `1 - S(t)` averaged
/// over the persons, with `ate` of riskRegression as the reference.
///
/// `data`, `design`, `fit` and `options` are those of
/// [`cox::fit`](crate::cox::fit). `levels` have one model. See the module
/// documentation for the estimate and the standard error.
///
/// # Errors
///
/// [`Error::Input`] for no level, a level that does not match the design, a
/// value that is not finite, a time that is not finite, a coefficient that is
/// `None`, a risk score that is not finite, a `conf_level` outside `(0, 1)`, no
/// row of positive weight, and with `std_err` for the data the influence
/// function does not describe: entry times, case weights other than 1, an
/// offset, a covariance that is not defined.
pub fn standardise_cox<'a>(
    data: &SurvData<'a>,
    design: MatRef<'a, f64>,
    fit: &'a CoxModelFit,
    options: &CoxOptions<'a>,
    levels: &[Level<'_>],
    times: &[f64],
    opts: &StandardiseOptions,
) -> Result<Standardised> {
    let fixed = data.clone().with_timefix(options.timefix)?;
    let models = vec![ModelInput {
        data: fixed,
        x: design,
        fit,
        options: *options,
    }];
    run(&models, None, levels, times, opts)
}

/// The standardised absolute risk of `cause` from cause-specific Cox models,
/// the average over the persons of the risk of
/// [`CauseSpecificCox::predict`](crate::absrisk::CauseSpecificCox::predict).
///
/// `data`, `designs`, `fits` and `csc` are those of
/// [`CauseSpecificCox::from_fits`](crate::absrisk::CauseSpecificCox::from_fits),
/// with `SurvType::Hazard`.
///
/// # Errors
///
/// [`Error::Input`] as for [`standardise_cox`], for designs and fits that do
/// not match the event codes of the data, for a cause outside the codes, and for
/// `SurvType::Survival`.
#[expect(
    clippy::too_many_arguments,
    reason = "the models, the levels and the options of one estimate"
)]
pub fn standardise_cause_specific<'a>(
    data: &SurvData<'a>,
    designs: &[MatRef<'a, f64>],
    fits: &'a [CoxModelFit],
    cause: u32,
    csc: &CscOptions,
    levels: &[Level<'_>],
    times: &[f64],
    opts: &StandardiseOptions,
) -> Result<Standardised> {
    if csc.surv_type != SurvType::Hazard {
        return Err(Error::input(
            "surv_type: only SurvType::Hazard is provided for standardised risks",
        ));
    }
    let fixed = data.clone().with_timefix(csc.timefix)?;
    let n_codes = fixed.followup()?.n_codes();
    if n_codes < 2 {
        return Err(Error::input(
            "data: absolute risk needs competing causes, so the status needs at least two codes",
        ));
    }
    if cause == 0 || cause > n_codes {
        return Err(Error::input(format!(
            "cause {cause} is outside 1..={n_codes}"
        )));
    }
    check_lengths(&[
        ("models", n_codes as usize),
        ("designs", designs.len()),
        ("fits", fits.len()),
    ])?;
    let models: Vec<ModelInput<'a>> = (1..=n_codes)
        .map(|code| ModelInput {
            data: fixed.clone(),
            x: designs[code as usize - 1],
            fit: &fits[code as usize - 1],
            options: CoxOptions {
                ties: csc.ties,
                timefix: csc.timefix,
                cause: Some(code),
                ..CoxOptions::default()
            },
        })
        .collect();
    run(&models, Some(cause as usize - 1), levels, times, opts)
}

/// The rows of a level for one model.
pub(crate) enum Source<'a> {
    Design(MatRef<'a, f64>),
    Over(Vec<Option<f64>>),
}

pub(crate) struct LevelRows<'a> {
    models: Vec<Source<'a>>,
}

impl LevelRows<'_> {
    /// The covariates of `row` for model `m`.
    pub(crate) fn fill(&self, m: usize, x: MatRef<'_, f64>, row: usize, out: &mut [f64]) {
        match &self.models[m] {
            Source::Design(d) => {
                for (a, o) in out.iter_mut().enumerate() {
                    *o = d[(row, a)];
                }
            }
            Source::Over(over) => {
                for (a, o) in out.iter_mut().enumerate() {
                    *o = over[a].unwrap_or_else(|| x[(row, a)]);
                }
            }
        }
    }
}

fn prepare<'a>(
    levels: &'a [Level<'_>],
    models: &[ModelInput<'_>],
    rows: &[usize],
) -> Result<Vec<LevelRows<'a>>> {
    if levels.is_empty() {
        return Err(Error::input("levels: at least one level is needed"));
    }
    let mut out = Vec::with_capacity(levels.len());
    for (l, level) in levels.iter().enumerate() {
        check_lengths(&[
            ("models", models.len()),
            ("level designs", level.designs.len()),
        ])?;
        let mut sources = Vec::with_capacity(models.len());
        for (m, (model, cf)) in models.iter().zip(&level.designs).enumerate() {
            let p = model.x.ncols();
            match cf {
                Counterfactual::Columns(set) => {
                    let mut over = vec![None; p];
                    for &(col, value) in *set {
                        if col >= p {
                            return Err(Error::input(format!(
                                "level {l}, model {m}: column {col} is outside the {p} columns"
                            )));
                        }
                        if !value.is_finite() {
                            return Err(Error::input(format!(
                                "level {l}, model {m}: a value is not finite"
                            )));
                        }
                        over[col] = Some(value);
                    }
                    sources.push(Source::Over(over));
                }
                Counterfactual::Design(d) => {
                    if d.nrows() != model.x.nrows() || d.ncols() != p {
                        return Err(Error::input(format!(
                            "level {l}, model {m}: the design is {} by {}, the model's is {} by {p}",
                            d.nrows(),
                            d.ncols(),
                            model.x.nrows()
                        )));
                    }
                    for &i in rows {
                        for a in 0..p {
                            if !d[(i, a)].is_finite() {
                                return Err(Error::input(format!(
                                    "level {l}, model {m}: a value is not finite"
                                )));
                            }
                        }
                    }
                    sources.push(Source::Design(*d));
                }
            }
        }
        out.push(LevelRows { models: sources });
    }
    Ok(out)
}

fn run(
    models: &[ModelInput<'_>],
    cause_model: Option<usize>,
    levels: &[Level<'_>],
    times: &[f64],
    opts: &StandardiseOptions,
) -> Result<Standardised> {
    if !(opts.conf_level > 0.0 && opts.conf_level < 1.0) {
        return Err(Error::input("conf_level: must be in (0, 1)"));
    }
    refuse_non_finite(times, "times")?;
    let data = &models[0].data;
    for (m, model) in models.iter().enumerate() {
        check_lengths(&[
            ("data", data.n_rows()),
            (&format!("design {m}"), model.x.nrows()),
        ])?;
    }
    let rows: Vec<usize> = data.used_rows().map(|i| i as usize).collect();
    if rows.is_empty() {
        return Err(Error::input("data: no row has positive weight"));
    }
    let n = rows.len();
    let stratum: Vec<usize> = rows
        .iter()
        .map(|&i| data.stratum().map_or(0, |c| c[i] as usize))
        .collect();
    let raw: Vec<f64> = rows
        .iter()
        .map(|&i| data.weight().map_or(1.0, |w| w[i]))
        .collect();
    let total: f64 = raw.iter().sum();
    let weight: Vec<f64> = raw.iter().map(|w| w / total).collect();
    let rows_of_levels = prepare(levels, models, &rows)?;
    let base = Baselines::new(models)?;
    let infl = if opts.std_err {
        Some(Influence::new(models, &base)?)
    } else {
        None
    };
    // The latest time for which every stratum of the population has a
    // baseline.
    let t_max = base
        .last_time
        .iter()
        .copied()
        .filter(|t| t.is_finite())
        .fold(f64::INFINITY, f64::min);
    let counts: Vec<Option<usize>> = times
        .iter()
        .map(|&t| (t <= t_max).then(|| base.grid.partition_point(|&g| g <= t)))
        .collect();
    let mut needed: Vec<usize> = counts
        .iter()
        .flatten()
        .copied()
        .filter(|&c| c > 0)
        .collect();
    needed.sort_unstable();
    needed.dedup();
    let evaluated = match cause_model {
        None => cox::CoxEval {
            model: &models[0],
            base: &base,
            rows: &rows,
            stratum: &stratum,
            weight: &weight,
        }
        .evaluate(&rows_of_levels, infl.as_ref(), &needed)?,
        Some(k) => csc::Csc {
            models,
            base: &base,
            cause_model: k,
            product_limit: opts.product_limit,
            form: opts.influence,
            rows: &rows,
            stratum: &stratum,
            weight: &weight,
        }
        .evaluate(&rows_of_levels, infl.as_ref(), &needed)?,
    };
    let n_levels = levels.len();
    let n_times = times.len();
    let n_needed = needed.len();
    let mut risk = vec![None; n_levels * n_times];
    let mut std_err = infl.as_ref().map(|_| vec![None; n_levels * n_times]);
    let mut covariance = infl
        .as_ref()
        .map(|_| vec![None; n_times * n_levels * n_levels]);
    for (k, c) in counts.iter().enumerate() {
        let Some(c) = *c else {
            continue;
        };
        if c == 0 {
            for l in 0..n_levels {
                risk[l * n_times + k] = Some(0.0);
                if let Some(se) = std_err.as_mut() {
                    se[l * n_times + k] = Some(0.0);
                }
            }
            if let Some(cov) = covariance.as_mut() {
                for v in &mut cov[k * n_levels * n_levels..(k + 1) * n_levels * n_levels] {
                    *v = Some(0.0);
                }
            }
            continue;
        }
        let u = needed
            .binary_search(&c)
            .map_err(|_| Error::input("internal: count"))?;
        for l in 0..n_levels {
            risk[l * n_times + k] = evaluated.risk[l * n_needed + u];
        }
        if let (Some(cov), Some(gram)) = (covariance.as_mut(), evaluated.gram.as_ref()) {
            for a in 0..n_levels {
                for b in 0..n_levels {
                    if evaluated.std_ok[a * n_needed + u] && evaluated.std_ok[b * n_needed + u] {
                        cov[k * n_levels * n_levels + a * n_levels + b] =
                            Some(gram[u * n_levels * n_levels + a * n_levels + b]);
                    }
                }
            }
            if let Some(se) = std_err.as_mut() {
                for l in 0..n_levels {
                    se[l * n_times + k] =
                        cov[k * n_levels * n_levels + l * n_levels + l].map(f64::sqrt);
                }
            }
        }
    }
    let (lower, upper) = match &std_err {
        Some(se) => {
            let z = normal::two_sided_z(opts.conf_level);
            let lo = risk
                .iter()
                .zip(se)
                .map(|(r, s)| r.zip(*s).map(|(r, s)| r - z * s))
                .collect();
            let up = risk
                .iter()
                .zip(se)
                .map(|(r, s)| r.zip(*s).map(|(r, s)| r + z * s))
                .collect();
            (Some(lo), Some(up))
        }
        None => (None, None),
    };
    Ok(Standardised {
        n_levels,
        n_times,
        n_rows: data.n_rows(),
        n_used: n,
        risk,
        std_err,
        lower,
        upper,
        covariance,
        conf_level: opts.conf_level,
    })
}

/// The rows per block of a sum over the persons.
const BLOCK: usize = 128;
/// The blocks computed at once.
const WAVE: usize = 32;

/// Sum over the blocks of `0..n`: `make` forms the value of a block, and the
/// values are merged in the order of the blocks, so that the result does not
/// depend on the number of threads. `n` is at least 1.
pub(crate) fn ordered_reduce<T: Send>(
    n: usize,
    make: impl Fn(Range<usize>) -> Result<T> + Sync,
    merge: impl Fn(&mut T, T),
) -> Result<T> {
    par::try_ordered_reduce(n, BLOCK, WAVE, make, merge)
}
