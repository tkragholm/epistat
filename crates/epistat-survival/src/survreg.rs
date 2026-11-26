//! Accelerated failure time models, as R's `survival::survreg`: a
//! location-scale model for the log of a survival time, fitted by maximum
//! likelihood to exact, right-, left- and interval-censored times, with
//! case weights, offsets, a scale for each stratum or a fixed scale, the
//! model-based and the robust variance, and predictions and residuals.
//!
//! # The model
//!
//! For a time `T`, possibly transformed to `Y = ln T`, row `i` has
//!
//! ```text
//! Y_i = x_i' beta + sigma_s(i) W_i,
//! ```
//!
//! where `W` has a standard distribution, `x_i` is the row of the design
//! matrix the caller passes (an intercept column is the caller's to include)
//! and `sigma_s` is the scale of the stratum of the row. The distributions are
//!
//! | [`Distribution`] | `W` | transformed |
//! |---|---|---|
//! | `Weibull` | minimum extreme value | `ln T` |
//! | `Exponential` | minimum extreme value, `sigma = 1` | `ln T` |
//! | `Rayleigh` | minimum extreme value, `sigma = 1/2` | `ln T` |
//! | `ExtremeValue` | minimum extreme value | no |
//! | `LogNormal` | normal | `ln T` |
//! | `Gaussian` | normal | no |
//! | `LogLogistic` | logistic | `ln T` |
//! | `StudentT` | Student t with `df` degrees of freedom | no |
//! | `Logistic` | logistic | no |
//!
//! With density `f`, distribution function `F` and `S = 1 - F` of `W`,
//! `z = (y - x'beta) / sigma`, a row contributes
//!
//! ```text
//! exact           ln f(z) - ln sigma
//! right-censored  ln S(z)
//! left-censored   ln F(z)
//! interval (a, b] ln [ S(z_a) - S(z_b) ]
//! ```
//!
//! multiplied by its case weight. For a transformed distribution the density
//! is that of `T`, so the log likelihood of an exact time carries the Jacobian
//! `-ln t`, as `survreg`'s does and as a likelihood for `T` must, which makes
//! log likelihoods comparable between a Weibull and a Gaussian fit to the
//! same times. The derivatives of a row in `eta = x'beta` and `ln sigma` are
//! in the `dist` module's documentation for the code and in Lawless \[1\] for the
//! formulas.
//!
//! # Rows and outcomes
//!
//! [`fit`] reads a [`SurvData`] holding either follow-up data with a binary
//! status and no entry time (right-censored times, status 1 an event) or an
//! [`IntervalSurv`](crate::IntervalSurv) outcome. Left-censored data are
//! interval data with a `-inf` lower bound. Rows of weight 0 are left out, and
//! `n_rows` of the result counts every row of the data. The strata of the data
//! are the strata of the scale, one `ln sigma` each, labelled by their dense
//! code; [`SurvData::stratum_levels`] gives the caller's codes. Delayed entry
//! (start times) and several event codes are refused.
//!
//! The tie policy of `aeqSurv` is not applied, and the options have no field
//! for it: `survreg` has no `timefix` argument and never merges near-ties, and
//! a likelihood in the densities of exact times has no use for it. A caller
//! who wants the rule applies [`SurvData::with_timefix`] to follow-up data
//! before the fit. Interval-censored data have no tie rule.
//!
//! For a transformed distribution every time that enters must be positive.
//! An interval whose lower bound is 0 has no lower bound once logged and is
//! fitted as a left-censored time at its upper bound, as `survreg` does for the
//! Weibull, exponential, Rayleigh and log-normal distributions. `survreg` does
//! not do it for the log-logistic distribution, where its list of names has a
//! typing mistake, and stops with an error. Here the conversion applies to all
//! the transformed distributions. A time at or below 0 elsewhere is an error.
//!
//! # Maximum likelihood
//!
//! The parameters are `beta` and, unless the scale is fixed, `ln sigma_s` for
//! each stratum. The iteration is `survreg`'s own, so that the iterations and
//! not only the limit can be compared with R's:
//!
//! 1. The covariates are centred at their mean and divided by their standard
//!    deviation when the first column is an intercept and there is another
//!    column, except columns that hold only 0 and 1. The coefficients and
//!    their variance are put back on the scale of the columns given at the end.
//! 2. Starting values come from the mean and variance of the times (the
//!    midpoint of an interval): a location and a scale by moments,
//!    refined for the intercept alone by Newton iteration with at most 20
//!    trial points, and for `beta` by one weighted least-squares step on the
//!    working response `eta - g_eta / g_ee` at the times.
//! 3. Newton-Raphson on the log likelihood `l`, with the observed information
//!    `I = -d^2 l` and the score `u`. A trial point is the last accepted point
//!    plus `I^-1 u`. The fit has converged when, at a trial point that was not
//!    a halving, `|1 - l_last / l| <= 1e-9` or `|l_last - l| <= 1e-9`
//!    (`rel.tolerance`). When `l` falls, or a value is not finite, the trial
//!    moves a third of the way back, and the first time any log scale that has
//!    fallen by more than 1.1 is put back to 1.1 below the accepted value.
//!    After 30 trial points (`maxiter`) the fit stops unconverged.
//! 4. `I` is factorised as `L D L'`. A pivot below `1e-10` (`toler.chol`) of the
//!    largest diagonal element is zero, and its column is aliased: its
//!    coefficient and its row and column of the variance are `None`. When
//!    a pivot is clearly negative the step is taken from the outer product of
//!    the row scores, which cannot be negative, and the step is counted in
//!    `outer_product_steps`.
//!
//! `survreg` factorises with an absolute threshold of `1e-10` when every
//! diagonal element is positive, and relative to the most negative one
//! otherwise. The threshold here is relative to the largest diagonal element
//! as in the Cox fit of this crate, so that a change of the units of the
//! times does not change which columns are aliased.
//!
//! The estimates are `None`-free except for aliased coefficients. A fit that
//! does not converge in the iteration limit is returned with `converged` false
//! and the last accepted point, with the variance of that point.
//!
//! The variance is the inverse of the observed information at the estimate,
//! over the coefficients and then the log scales. It is `None` entirely when
//! the information is not positive semi-definite there.
//!
//! # Offsets
//!
//! [`SurvregOptions::offset`] gives each row a term added to its linear
//! predictor, `eta = x'beta + o`, on the scale of the model, as `offset()` in
//! a `survreg` formula does. The null model, the starting values, the
//! iteration, the linear predictors of the fit and with them every residual
//! use it. A prediction for new rows takes its offsets from
//! [`PredictOptions::offset`] and is zero without them.
//!
//! # The t distribution
//!
//! [`Distribution::StudentT`] is a Student t error on `T` itself, with the
//! degrees of freedom of [`SurvregOptions::df`], which must exceed 2 so that
//! the variance of the error exists (the starting scale divides the variance
//! of the times by `nu / (nu - 2)`). `survreg` has the default 4; here there is
//! none. The distribution function is a regularised incomplete beta function by
//! its continued fraction, and the quantile its root.
//!
//! # The robust variance
//!
//! With [`SurvregOptions::robust`], or with clusters in the data,
//! [`SurvregFit::robust_variance`] is `sum over clusters of s_g s_g'`, where
//! `s_g` is the sum over the rows of the cluster of `w_i u_i' V`, `u_i` the
//! score of row `i` for the coefficients and log scales and `V` the
//! model-based variance. Without clusters each row is its own. These are the
//! weighted dfbeta residuals summed by cluster, `crossprod(dfbeta)` of
//! `survreg`, with no small-sample factor. `epistat-inference`'s clustered
//! covariance multiplies by `G / (G - 1)` and divides by `n^2`, and its
//! sandwich solves with the information, which `survreg` does not, so neither
//! is used.
//! [`SurvregFit::variance`] stays the model-based variance, which the
//! residuals use, as they use `naive.var` in `survreg`.
//!
//! # Residuals and terms
//!
//! [`SurvregFit::dfbeta`] gives the dfbeta and dfbetas residuals, weighted and
//! summed by group as `residuals.survreg` does. [`PredictType::Terms`] gives
//! the part of the linear predictor of each group of columns with its standard
//! error. Both are described with their types.
//!
//! # Where this differs from `survival`
//!
//! Results of `survival` 3.8-12 that do not follow from their definitions and
//! are not copied. Each is shown in `tests/data/survreg_extra.R` and the tests
//! compare with the value computed from R's own pieces.
//!
//! - `residuals.survreg` has the wrong sign for the derivative in the log scale
//!   of an interval-censored row. Besides the displacements it reaches the
//!   dfbeta residuals, and so the robust variance and the standard errors from
//!   it, when the scale is estimated.
//! - `predict.survreg` takes the offset of new data as 0 for a linear
//!   predictor and a response, and leaves it out of a quantile, where for the
//!   fitted data it is part of all three. Here an offset given is used in all
//!   of them.
//! - The standard error of a term from `predict.survreg` is
//!   `(x * beta) V (x * beta)'`, with the coefficients inside, which depends on
//!   the units of the covariate. The delta method gives `x V x'`, which is
//!   used here.
//! - The deviance function of the t distribution is wrong for an
//!   interval-censored row: the log of a negative number as the saturated log
//!   likelihood, and a centre that averages the status column in. The
//!   saturated log likelihood is `log(2 F(width / 2) - 1)` and the centre the
//!   midpoint here.
//! - The aliased columns of the dfbeta residuals are 0 in `survreg` and `None`
//!   here.
//!
//! # Results that are errors
//!
//! A design with other than one row per data row, without columns, or with a
//! value that is not finite; an offset that is not finite or has another
//! length than the data; degrees of freedom missing or not above 2 for the t
//! distribution, or given for another; no row of positive weight; a fixed scale that is
//! not positive, or given with a distribution that has its own, or with more
//! than one stratum; a time that does not suit a transformed distribution;
//! a stratum without a row of positive weight; and a log likelihood or
//! derivative that is not finite at the starting values.
//!
//! # Threads
//!
//! The sums over rows run in row order on one thread, so the result is the
//! same however many threads the caller has.
//!
//! # References
//!
//! 1. Lawless JF. Statistical Models and Methods for Lifetime Data. 2nd ed.
//!    Wiley; 2003. Chapter 6.
//! 2. Therneau TM, Grambsch PM. Modeling Survival Data: Extending the Cox
//!    Model. Springer; 2000.
//! 3. Kalbfleisch JD, Prentice RL. The Statistical Analysis of Failure Time
//!    Data. 2nd ed. Wiley; 2002. Chapter 2.

#![expect(
    clippy::float_cmp,
    reason = "a column of exactly 0 and 1 or of exactly 1 is the case tested"
)]
#![expect(
    clippy::manual_midpoint,
    reason = "the midpoint is (a + b) / 2 as in survreg, and the times are far from overflow"
)]
#![expect(
    clippy::needless_range_loop,
    reason = "rows index several parallel columns and matrices in the order of the sums"
)]

mod dist;
mod newton;
mod predict;
mod robust;
mod tdist;

use dist::Base;
use epistat_linalg::MatRef;
use newton::{Control, Problem, factorise, newton};

pub use predict::{
    Dfbeta, DfbetaOptions, PredictOptions, PredictType, Prediction, ResidualKind, TermsReference,
};

use crate::error::{Error, Result};
use crate::ldl;
use crate::surv::{IntervalKind, Outcome, SurvData};

/// The distribution of the error term, and with it the transformation of the
/// times. See the [module documentation](self) for the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Distribution {
    /// Weibull: minimum extreme value on `ln T`. The default.
    #[default]
    Weibull,
    /// Exponential: the Weibull with the scale fixed at 1.
    Exponential,
    /// Rayleigh: the Weibull with the scale fixed at 1/2.
    Rayleigh,
    /// Minimum extreme value on `T` itself.
    ExtremeValue,
    /// Normal on `T` itself.
    Gaussian,
    /// Logistic on `T` itself.
    Logistic,
    /// Normal on `ln T`.
    LogNormal,
    /// Logistic on `ln T`.
    LogLogistic,
    /// Student t on `T` itself, with the degrees of freedom of
    /// [`SurvregOptions::df`], which must exceed 2.
    StudentT,
}

impl Distribution {
    fn base(self, df: Option<f64>) -> Base {
        match self {
            Self::StudentT => Base::StudentT(df.expect("checked with the options")),
            Self::Weibull | Self::Exponential | Self::Rayleigh | Self::ExtremeValue => {
                Base::ExtremeValue
            }
            Self::Gaussian | Self::LogNormal => Base::Gaussian,
            Self::Logistic | Self::LogLogistic => Base::Logistic,
        }
    }

    /// Whether the model is for `ln T`.
    #[must_use]
    pub fn is_log(self) -> bool {
        !matches!(
            self,
            Self::ExtremeValue | Self::Gaussian | Self::Logistic | Self::StudentT
        )
    }

    /// The scale that the distribution fixes, if it does.
    #[must_use]
    pub fn fixed_scale(self) -> Option<f64> {
        match self {
            Self::Exponential => Some(1.0),
            Self::Rayleigh => Some(0.5),
            _ => None,
        }
    }
}

/// The settings of [`fit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvregOptions<'a> {
    /// The distribution. Weibull.
    pub distribution: Distribution,
    /// The degrees of freedom of the t distribution, `parms` of `survreg`.
    /// Required, and greater than 2, for [`Distribution::StudentT`], and not
    /// allowed with another distribution. `survreg` defaults to 4, which has
    /// no more reason than any other value, so there is no default here.
    pub df: Option<f64>,
    /// An offset for each row of the data, added to the linear predictor
    /// `x'beta` as the `offset()` term of `survreg` does, on the scale of the
    /// model (the log of the time for a transformed distribution). `None` is
    /// no offset.
    pub offset: Option<&'a [f64]>,
    /// Whether to compute the robust variance, `robust = TRUE` of `survreg`.
    /// It is computed too when the data have clusters, as `survreg` does for
    /// a `cluster()` term, and then sums by cluster. See
    /// [`SurvregFit::robust_variance`].
    pub robust: bool,
    /// A fixed scale, as `scale` of `survreg`. `None` estimates one scale for
    /// each stratum. Not allowed with the exponential and Rayleigh
    /// distributions, which fix their own, nor with more than one stratum.
    pub scale: Option<f64>,
    /// Largest number of trial points, `maxiter` of `survreg.control`. 30.
    pub max_iterations: u32,
    /// Convergence threshold on the change of the log likelihood,
    /// `rel.tolerance` of `survreg.control`. 1e-9.
    pub rel_tolerance: f64,
    /// Relative size below which a pivot of the information matrix is zero,
    /// `toler.chol` of `survreg.control`. 1e-10.
    pub toler_chol: f64,
}

impl Default for SurvregOptions<'_> {
    fn default() -> Self {
        Self {
            distribution: Distribution::Weibull,
            df: None,
            offset: None,
            robust: false,
            scale: None,
            max_iterations: 30,
            rel_tolerance: 1e-9,
            toler_chol: 1e-10,
        }
    }
}

/// An accelerated failure time model.
#[derive(Debug, Clone, PartialEq)]
pub struct SurvregFit {
    /// One coefficient per column of the design, `None` for an aliased column.
    pub coefficients: Vec<Option<f64>>,
    /// The log of the scale of each stratum, one entry when there are no
    /// strata. All equal to the log of the fixed scale when it is fixed.
    pub log_scale: Vec<f64>,
    /// Whether the scale was estimated. False for a fixed scale, where the
    /// variance has no rows for it.
    pub scale_estimated: bool,
    /// The variance matrix over the coefficients and, when the scale is
    /// estimated, the log scales, row-major with `q * q` entries, `q` the
    /// number of coefficients plus the number of estimated log scales. `None`
    /// in the row and column of an aliased coefficient, and in every entry when
    /// the information matrix at the estimate is not positive semi-definite.
    pub variance: Vec<Option<f64>>,
    /// The robust (sandwich) variance, laid out as `variance`: the sum over
    /// the clusters of the products of the cluster totals of the weighted
    /// dfbeta residuals, which is `var` of `survreg(robust = TRUE)`. `None`
    /// unless [`SurvregOptions::robust`] was set or the data had clusters.
    /// `variance` stays the model-based one, `naive.var` of `survreg`, and is
    /// the one the residuals use. All entries are `None` when the model-based
    /// variance is not defined.
    pub robust_variance: Option<Vec<Option<f64>>>,
    /// The number of clusters of the robust variance: those of the data, or
    /// the rows used when there are none. `None` without a robust variance.
    pub n_clusters: Option<usize>,
    /// The log likelihood at the estimate, in the density of the times.
    pub loglik: f64,
    /// The log likelihood of the model with an intercept only and the same
    /// distribution and scale structure.
    pub loglik_null: f64,
    /// The trial points computed.
    pub iterations: u32,
    /// Whether the log likelihood stopped changing within the iteration limit.
    pub converged: bool,
    /// The Newton steps taken from the outer product of the row scores.
    pub outer_product_steps: u32,
    /// The columns that are not aliased.
    pub rank: usize,
    /// The linear predictor `x'beta` plus the offset of every row of the
    /// data, rows of weight 0 included. Aliased columns are left out.
    pub linear_predictors: Vec<f64>,
    /// The unweighted mean of each design column over every row of the data,
    /// `means` of `survreg`, which `type = "terms"` centres on.
    pub column_means: Vec<f64>,
    /// The degrees of freedom of the t distribution, when that is the
    /// distribution.
    pub t_df: Option<f64>,
    /// The rows of the data.
    pub n_rows: usize,
    /// The rows fitted: those with a weight above zero.
    pub n_used: usize,
    /// The distribution fitted.
    pub distribution: Distribution,
    /// The number of strata of the scale, 1 when the data have none.
    pub n_strata: usize,
}

impl SurvregFit {
    /// The scale of each stratum, `exp(log_scale)`.
    #[must_use]
    pub fn scale(&self) -> Vec<f64> {
        self.log_scale.iter().map(|ls| ls.exp()).collect()
    }

    /// The degrees of freedom of the model: the coefficients that are not
    /// aliased and the scales estimated. `survreg` counts aliased
    /// coefficients too.
    #[must_use]
    pub fn df(&self) -> usize {
        self.rank
            + if self.scale_estimated {
                self.n_strata
            } else {
                0
            }
    }

    fn n_coef(&self) -> usize {
        self.coefficients.len()
    }

    fn n_params(&self) -> usize {
        self.n_coef()
            + if self.scale_estimated {
                self.n_strata
            } else {
                0
            }
    }
}

/// The times of every row on the scale of the model.
pub(super) struct Times {
    pub y1: Vec<f64>,
    pub y2: Vec<f64>,
    pub status: Vec<u8>,
}

/// Read the outcome of `data` as times on the scale of the model, one entry
/// per row, with status 0 right, 1 exact, 2 left and 3 interval censored.
///
/// A time that a transformed distribution cannot take is NaN in `y1`, so that
/// the caller can refuse it when the row is used.
pub(super) fn times_of(data: &SurvData<'_>, distribution: Distribution) -> Result<Times> {
    let n = data.n_rows();
    let mut times = match data.outcome() {
        Outcome::Followup(surv) => {
            if surv.start().is_some() {
                return Err(Error::unsupported(
                    "survreg takes times from the origin; delayed entry is not supported"
                        .to_owned(),
                ));
            }
            if surv.n_codes() != 1 {
                return Err(Error::invalid(format!(
                    "survreg needs a binary status; the data has {} event codes",
                    surv.n_codes()
                )));
            }
            Times {
                y1: surv.stop().to_vec(),
                y2: vec![0.0; n],
                status: surv
                    .status()
                    .iter()
                    .map(|&s| u8::try_from(s).expect("status is 0 or 1 here"))
                    .collect(),
            }
        }
        Outcome::Interval(surv) => {
            let mut times = Times {
                y1: Vec::with_capacity(n),
                y2: vec![0.0; n],
                status: Vec::with_capacity(n),
            };
            for row in 0..n {
                let (lower, upper) = (surv.lower()[row], surv.upper()[row]);
                let (kind, y1) = match surv.kind()[row] {
                    IntervalKind::Right => (0, lower),
                    IntervalKind::Exact => (1, lower),
                    IntervalKind::Left => (2, upper),
                    IntervalKind::Interval if distribution.is_log() && lower == 0.0 => (2, upper),
                    IntervalKind::Interval => {
                        times.y2[row] = upper;
                        (3, lower)
                    }
                };
                times.y1.push(y1);
                times.status.push(kind);
            }
            times
        }
    };
    if distribution.is_log() {
        for row in 0..n {
            let positive = times.y1[row] > 0.0 && (times.status[row] != 3 || times.y2[row] > 0.0);
            if positive {
                times.y1[row] = times.y1[row].ln();
                times.y2[row] = if times.status[row] == 3 {
                    times.y2[row].ln()
                } else {
                    0.0
                };
            } else {
                times.y1[row] = f64::NAN;
            }
        }
    }
    Ok(times)
}

/// The design of the rows used, row-major, and how its columns were scaled.
struct Design {
    x: Vec<f64>,
    p: usize,
    /// Centre and divisor of each column when the covariates were scaled.
    scaling: Option<(Vec<f64>, Vec<f64>)>,
}

fn design_of(x: MatRef<'_, f64>, used: &[usize]) -> Design {
    let p = x.ncols();
    let mut flat = Vec::with_capacity(used.len() * p);
    for &row in used {
        for col in 0..p {
            flat.push(x[(row, col)]);
        }
    }
    let n = used.len();
    let first_is_intercept = (0..n).all(|i| flat[i * p] == 1.0);
    if !(first_is_intercept && p > 1) {
        return Design {
            x: flat,
            p,
            scaling: None,
        };
    }
    let mut centre = vec![0.0_f64; p];
    let mut divisor = vec![1.0_f64; p];
    for col in 0..p {
        let binary = (0..n).all(|i| flat[i * p + col] == 0.0 || flat[i * p + col] == 1.0);
        if binary {
            continue;
        }
        let mut sum = 0.0;
        for i in 0..n {
            sum += flat[i * p + col];
        }
        let mean = sum / n as f64;
        let mut squares = 0.0;
        for i in 0..n {
            let d = flat[i * p + col] - mean;
            squares += d * d;
        }
        let sd = (squares / (n as f64 - 1.0)).sqrt();
        // A constant column is aliased with the intercept and stays as it is.
        if sd > 0.0 {
            centre[col] = mean;
            divisor[col] = sd;
        }
    }
    for i in 0..n {
        for col in 0..p {
            flat[i * p + col] = (flat[i * p + col] - centre[col]) / divisor[col];
        }
    }
    Design {
        x: flat,
        p,
        scaling: Some((centre, divisor)),
    }
}

/// Fit an accelerated failure time model by maximum likelihood.
///
/// `x` has one row per row of `data` and one column per coefficient, with an
/// intercept column if the model has one. See the [module documentation](self)
/// for the model, the outcomes it reads and the iteration.
///
/// # Errors
///
/// An input error as listed in the module documentation.
///
/// # Panics
///
/// Never on valid data; the unwraps are on values the checks just made.
#[expect(
    clippy::too_many_lines,
    reason = "the steps of the fit read in order, each its own paragraph"
)]
pub fn fit(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    options: &SurvregOptions<'_>,
) -> Result<SurvregFit> {
    let dist = options.distribution;
    match (dist, options.df) {
        (Distribution::StudentT, None) => {
            return Err(Error::invalid(
                "the t distribution needs its degrees of freedom (df)".to_owned(),
            ));
        }
        (Distribution::StudentT, Some(df)) if !(df.is_finite() && df > 2.0) => {
            return Err(Error::invalid(
                "the degrees of freedom of the t distribution must be greater than 2".to_owned(),
            ));
        }
        (Distribution::StudentT, Some(_)) => {}
        (_, Some(_)) => {
            return Err(Error::invalid(
                "df applies to the t distribution only".to_owned(),
            ));
        }
        _ => {}
    }
    let base = dist.base(options.df);
    let n_all = data.n_rows();
    let p = x.ncols();
    if x.nrows() != n_all {
        return Err(Error::invalid(format!(
            "the design has {} rows against {n_all} rows of data",
            x.nrows()
        )));
    }
    if p == 0 {
        return Err(Error::invalid("the design has no columns".to_owned()));
    }
    for col in 0..p {
        if (0..n_all).any(|row| !x[(row, col)].is_finite()) {
            return Err(Error::invalid(format!(
                "design column {col} has a value that is not finite"
            )));
        }
    }
    if let Some(offset) = options.offset {
        if offset.len() != n_all {
            return Err(Error::invalid(format!(
                "the offset has {} values against {n_all} rows of data",
                offset.len()
            )));
        }
        if offset.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("an offset is not finite".to_owned()));
        }
    }
    if options.max_iterations == 0 {
        return Err(Error::invalid("max_iterations is 0".to_owned()));
    }
    for (name, value) in [
        ("rel_tolerance", options.rel_tolerance),
        ("toler_chol", options.toler_chol),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(Error::invalid(format!(
                "{name} must be positive and finite"
            )));
        }
    }
    let n_strata = data.n_strata().max(1);
    let fixed_scale = match (dist.fixed_scale(), options.scale) {
        (Some(_), Some(_)) => {
            return Err(Error::invalid(
                "this distribution has a fixed scale; the scale option does not apply".to_owned(),
            ));
        }
        (Some(fixed), None) => Some(fixed),
        (None, scale) => scale,
    };
    if let Some(scale) = fixed_scale {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(Error::invalid("a fixed scale must be positive".to_owned()));
        }
        if n_strata > 1 {
            return Err(Error::invalid(
                "a fixed scale cannot be combined with strata on the scale".to_owned(),
            ));
        }
    }
    let fixed_ls = fixed_scale.map(f64::ln);
    let n_scale = if fixed_ls.is_some() { 0 } else { n_strata };

    let times = times_of(data, dist)?;
    let used: Vec<usize> = data.used_rows().map(|row| row as usize).collect();
    if used.is_empty() {
        return Err(Error::invalid("no row has a positive weight".to_owned()));
    }
    let bad_times = used.iter().filter(|&&row| times.y1[row].is_nan()).count();
    if bad_times > 0 {
        let first = used.iter().find(|&&row| times.y1[row].is_nan()).unwrap();
        return Err(Error::invalid(format!(
            "{bad_times} times are not positive, which this distribution needs, first at row {first}"
        )));
    }
    let y1: Vec<f64> = used.iter().map(|&row| times.y1[row]).collect();
    let y2: Vec<f64> = used.iter().map(|&row| times.y2[row]).collect();
    let status: Vec<u8> = used.iter().map(|&row| times.status[row]).collect();
    let offset: Option<Vec<f64>> = options
        .offset
        .map(|o| used.iter().map(|&row| o[row]).collect());
    let weight: Vec<f64> = match data.weight() {
        Some(w) => used.iter().map(|&row| w[row]).collect(),
        None => vec![1.0; used.len()],
    };
    let stratum: Option<Vec<u32>> = data
        .stratum()
        .map(|codes| used.iter().map(|&row| codes[row]).collect());
    if let Some(codes) = &stratum {
        let mut seen = vec![false; n_strata];
        for &c in codes {
            seen[c as usize] = true;
        }
        if let Some(empty) = seen.iter().position(|&s| !s) {
            return Err(Error::invalid(format!(
                "stratum {empty} has no row with a positive weight"
            )));
        }
    }
    let n = used.len();
    // The Jacobian of the log transformation: an exact time t has the density
    // of ln t divided by t.
    let mut log_correction = 0.0;
    if dist.is_log() {
        for i in 0..n {
            if status[i] == 1 {
                log_correction -= weight[i] * y1[i];
            }
        }
    }

    let design = design_of(x, &used);
    let control = Control {
        max_iterations: options.max_iterations,
        rel_tolerance: options.rel_tolerance,
        toler_chol: options.toler_chol,
    };
    let ones = vec![1.0_f64; n];
    let meanonly = design.x.iter().all(|&v| v == 1.0) && p == 1;
    let intercept_problem = Problem {
        base,
        y1: &y1,
        y2: &y2,
        status: &status,
        weight: &weight,
        stratum: stratum.as_deref(),
        x: &ones,
        p: 1,
        offset: offset.as_deref(),
        n_scale,
        fixed_ls,
    };

    // Starting values: moments of the times, then a Newton fit of the
    // intercept alone when the model has more than that.
    let total_weight: f64 = weight.iter().sum();
    let centre_of_row = |i: usize| {
        if status[i] == 3 {
            0.5 * (y1[i] + y2[i])
        } else {
            y1[i]
        }
    };
    let mut mean = 0.0;
    for i in 0..n {
        mean += weight[i] * centre_of_row(i);
    }
    mean /= total_weight;
    let mut variance = 0.0;
    for i in 0..n {
        let d = centre_of_row(i) - mean;
        variance += weight[i] * d * d;
    }
    variance /= total_weight;
    let (_, scale_start) = base.start(mean, variance);
    let start_ls = fixed_ls.unwrap_or_else(|| (4.0 * scale_start).ln() / 2.0);
    // One weighted least squares step on the working response, at the times.
    let working_step = |x_rows: &[f64], pp: usize, ls: &[f64]| -> Result<Vec<f64>> {
        let mut a = vec![0.0_f64; pp * pp];
        let mut b = vec![0.0_f64; pp];
        for i in 0..n {
            let s = stratum.as_ref().map_or(0, |c| c[i] as usize);
            let d = dist::row_derivs(
                base,
                status[i],
                y1[i],
                y2[i],
                centre_of_row(i),
                ls[if ls.len() == 1 { 0 } else { s }],
            );
            let wt = -d.g_ee * weight[i];
            let eta_i = centre_of_row(i) - offset.as_ref().map_or(0.0, |o| o[i]);
            let target = wt * eta_i + weight[i] * d.g_eta;
            if !(wt.is_finite() && target.is_finite()) {
                return Err(Error::invalid(
                    "the starting values are not finite; check the times against the \
                     distribution"
                        .to_owned(),
                ));
            }
            let row = &x_rows[i * pp..(i + 1) * pp];
            for c in 0..pp {
                b[c] += target * row[c];
                for e in 0..=c {
                    a[c * pp + e] += wt * row[c] * row[e];
                }
            }
        }
        for c in 0..pp {
            for e in 0..c {
                a[e * pp + c] = a[c * pp + e];
            }
        }
        // The matrix is not necessarily positive definite away from the
        // maximum, so a column with no usable pivot is left at zero.
        factorise(&mut a, pp, options.toler_chol);
        ldl::solve(&a, pp, &mut b);
        Ok(b)
    };
    let (null_fit, start) = if meanonly {
        let intercept = working_step(&ones, 1, &[start_ls])?;
        let mut beta = intercept;
        beta.extend(std::iter::repeat_n(start_ls, n_scale));
        (None, beta)
    } else {
        let mut beta0 = working_step(&ones, 1, &[start_ls])?;
        beta0.extend(std::iter::repeat_n(start_ls, n_scale));
        let null_fit = newton(
            &intercept_problem,
            beta0,
            Control {
                max_iterations: 20,
                ..control
            },
        )?;
        let ls: Vec<f64> = if n_scale == 0 {
            vec![start_ls]
        } else {
            null_fit.beta[1..].to_vec()
        };
        let mut beta = working_step(&design.x, design.p, &ls)?;
        beta.extend(ls.iter().take(n_scale));
        (Some(null_fit), beta)
    };

    let problem = Problem {
        base,
        y1: &y1,
        y2: &y2,
        status: &status,
        weight: &weight,
        stratum: stratum.as_deref(),
        x: &design.x,
        p: design.p,
        offset: offset.as_deref(),
        n_scale,
        fixed_ls,
    };
    let fitted = newton(&problem, start, control)?;
    let q = p + n_scale;

    // The information on the scale of the iteration and its inverse.
    let mut factor = fitted.imat.clone();
    let nonneg = factorise(&mut factor, q, options.toler_chol);
    let aliased: Vec<bool> = (0..p).map(|j| factor[j * q + j] == 0.0).collect();
    let scaled_variance = nonneg.then(|| ldl::inverse(&factor, q));

    // Back to the scale of the columns given.
    let mut coefficient: Vec<f64> = fitted.beta[..p].to_vec();
    let mut back = vec![0.0_f64; q * q];
    for a in 0..q {
        back[a * q + a] = 1.0;
    }
    if let Some((centre, divisor)) = &design.scaling {
        for j in 1..p {
            back[j * q + j] = 1.0 / divisor[j];
            back[j] = -centre[j] / divisor[j];
        }
        let mut original = vec![0.0_f64; p];
        for (a, value) in original.iter_mut().enumerate() {
            let mut sum = 0.0;
            for c in 0..p {
                sum += back[a * q + c] * coefficient[c];
            }
            *value = sum;
        }
        coefficient = original;
    }
    let variance_out: Vec<Option<f64>> = match &scaled_variance {
        None => vec![None; q * q],
        Some(v) => {
            let mut left = vec![0.0_f64; q * q];
            for a in 0..q {
                for d in 0..q {
                    let mut sum = 0.0;
                    for c in 0..q {
                        sum += back[a * q + c] * v[c * q + d];
                    }
                    left[a * q + d] = sum;
                }
            }
            let mut out = vec![0.0_f64; q * q];
            for a in 0..q {
                for b in 0..q {
                    let mut sum = 0.0;
                    for d in 0..q {
                        sum += left[a * q + d] * back[b * q + d];
                    }
                    out[a * q + b] = sum;
                }
            }
            let dead = |i: usize| i < p && aliased[i];
            (0..q * q)
                .map(|k| {
                    let (a, b) = (k / q, k % q);
                    (!dead(a) && !dead(b)).then(|| 0.5 * (out[a * q + b] + out[b * q + a]))
                })
                .collect()
        }
    };

    // The linear predictor of every row, from the scaled covariates as the
    // iteration saw them.
    let linear_predictors: Vec<f64> = (0..n_all)
        .map(|row| {
            let mut eta = 0.0;
            for j in 0..p {
                let value = match &design.scaling {
                    Some((centre, divisor)) => (x[(row, j)] - centre[j]) / divisor[j],
                    None => x[(row, j)],
                };
                eta += fitted.beta[j] * value;
            }
            if let Some(offset) = options.offset {
                eta += offset[row];
            }
            eta
        })
        .collect();

    let log_scale = match fixed_ls {
        Some(ls) => vec![ls; n_strata],
        None => fitted.beta[p..].to_vec(),
    };
    let null_loglik = null_fit.map_or(fitted.loglik, |f| f.loglik) + log_correction;
    let rank = aliased.iter().filter(|&&a| !a).count();

    // The robust variance: the cluster totals of the weighted dfbeta rows,
    // `score' V` with `V` the model-based variance, and their cross-products.
    let clusters: Option<Vec<usize>> = data
        .cluster()
        .map(|codes| used.iter().map(|&row| codes[row] as usize).collect());
    let robust_variance = (options.robust || clusters.is_some()).then(|| {
        let sums_of = scaled_variance.as_ref().and_then(|_| {
            let v: Vec<f64> = variance_out.iter().map(|e| e.unwrap_or(0.0)).collect();
            let groups = clusters.as_ref().map_or(0, |_| data.n_clusters());
            let mut totals = vec![0.0_f64; groups * q];
            let mut cross = vec![0.0_f64; q * q];
            let mut x_row = vec![0.0_f64; p];
            let mut dfbeta = vec![0.0_f64; q];
            for k in 0..n {
                let row = used[k];
                for (j, value) in x_row.iter_mut().enumerate() {
                    *value = x[(row, j)];
                }
                let stratum_k = stratum.as_ref().map_or(0, |c| c[k] as usize);
                robust::dfbeta_row(
                    &robust::RowInput {
                        base,
                        status: status[k],
                        y1: y1[k],
                        y2: y2[k],
                        eta: linear_predictors[row],
                        ls: log_scale[stratum_k],
                        stratum: stratum_k,
                        x: &x_row,
                        n_scale,
                    },
                    &v,
                    &mut dfbeta,
                );
                for d in &mut dfbeta {
                    *d *= weight[k];
                }
                if dfbeta.iter().any(|d| !d.is_finite()) {
                    return None;
                }
                match &clusters {
                    Some(c) => {
                        for (t, d) in totals[c[k] * q..(c[k] + 1) * q].iter_mut().zip(&dfbeta) {
                            *t += d;
                        }
                    }
                    None => {
                        for a in 0..q {
                            for b in 0..=a {
                                cross[a * q + b] += dfbeta[a] * dfbeta[b];
                            }
                        }
                    }
                }
            }
            for total in totals.chunks_exact(q) {
                for a in 0..q {
                    for b in 0..=a {
                        cross[a * q + b] += total[a] * total[b];
                    }
                }
            }
            for a in 0..q {
                for b in 0..a {
                    cross[b * q + a] = cross[a * q + b];
                }
            }
            Some(cross)
        });
        let dead = |i: usize| i < p && aliased[i];
        (0..q * q)
            .map(|k| {
                let (a, b) = (k / q, k % q);
                sums_of
                    .as_ref()
                    .filter(|_| !dead(a) && !dead(b))
                    .map(|m| m[k])
            })
            .collect::<Vec<Option<f64>>>()
    });
    let n_clusters = robust_variance
        .as_ref()
        .map(|_| clusters.as_ref().map_or(n, |_| data.n_clusters()));
    let column_means: Vec<f64> = (0..p)
        .map(|j| {
            let mut sum = 0.0;
            for row in 0..n_all {
                sum += x[(row, j)];
            }
            sum / n_all as f64
        })
        .collect();
    Ok(SurvregFit {
        coefficients: coefficient
            .iter()
            .zip(&aliased)
            .map(|(&c, &dead)| (!dead).then_some(c))
            .collect(),
        log_scale,
        scale_estimated: n_scale > 0,
        variance: variance_out,
        robust_variance,
        n_clusters,
        loglik: fitted.loglik + log_correction,
        loglik_null: null_loglik,
        iterations: fitted.iterations,
        converged: fitted.converged,
        outer_product_steps: fitted.outer_product_steps,
        rank,
        linear_predictors,
        column_means,
        t_df: options.df,
        n_rows: n_all,
        n_used: n,
        distribution: dist,
        n_strata,
    })
}
