//! Royston and Sauerbrei's D, with the explained-variation measures that
//! `survival::royston` reports beside it.
//!
//! [`estimate`] takes the linear predictor of a Cox model, one value per row,
//! and [`from_fit`] takes a fit of [`cox::fit`](crate::cox::fit) and the design
//! it was fitted on. Both refit a Cox model on the normal scores of the
//! predictor, as `royston` does, and return D, its standard error, the
//! explained variation `R.D`, the measure `R.KO` of Kent and O'Quigley and the
//! concordance `C.GH` of Gonen and Heller. [`from_fit`] adds the Nagelkerke
//! `R.N`.
//!
//! # Method
//!
//! Let `eta_i` be the predictor of the `n` rows used, ranked `1..n` from
//! smallest to largest. The normal score of rank `k` is
//!
//! ```text
//! z_k = Phi^-1((k - 3/8) / (n + 1/4))
//! ```
//!
//! (Blom's approximation to the expected normal order statistic). A group of
//! `m` rows with equal `eta` takes the mean of the `m` scores of its ranks
//! (`ties`, the default, as in `royston`). Without `ties` each row takes the
//! score of the mean rank of its group. Call the score of row `i` `q_i`. It
//! has mean 0 and a variance close to 1. A Cox model with the single covariate
//! `q` (Efron's ties, no weights, strata or clusters, the times and status of
//! `data`) has coefficient `b` and variance `v`. Then
//!
//! ```text
//! D     = b sqrt(8 / pi)
//! se(D) = sqrt(8 v / pi)
//! R.D   = b^2 / (pi^2 / 6 + b^2)
//! R.KO  = s^2 / (pi^2 / 6 + s^2),   s^2 = the variance of eta, divisor n - 1
//! C.GH  = 2 / (n (n - 1)) sum over i < j of 1 / (1 + exp(eta_(i) - eta_(j)))
//! ```
//!
//! with `eta_(1) <= ... <= eta_(n)` the sorted predictor. `kappa = sqrt(8/pi)`
//! is the mean of the absolute value of a standard normal variable times 2,
//! which makes `D` the log hazard ratio between the two halves of a normal
//! predictor. `R.D` is the `R^2` of Royston and Sauerbrei with `b` standing for
//! the standard deviation of the predictor on the log hazard scale.
//!
//! The Nagelkerke measure is
//! `(1 - exp(-LR / n)) / (1 - exp(2 l_0 / n))` with `LR` the likelihood-ratio
//! statistic of the fit, `l_0` its log partial likelihood at the null and `n`
//! the number of rows given.
//!
//! # Adjustment
//!
//! The adjustment of Royston and Sauerbrei for overfitting is off by default,
//! as in `royston` (`adjust = FALSE`). It is requested with
//! [`RoystonAdjustment`], the number of events `e` and of coefficients `p` of the
//! fitted model. With `r = e / (e - p)` and `t = (1 + b^2 - r) / r`,
//!
//! ```text
//! D     = sign(b) sign(t) sqrt(8 |t| / pi)
//! se(D) = se(D) |b| / (r sqrt(|t|))
//! R.D   = 1 - r (1 - b^2 / (1 + b^2))
//! ```
//!
//! where the `se(D)` on the right is the unadjusted one. When `t` is 0 the
//! adjusted standard error is `None`.
//! [`from_fit`] reads `e` and `p` from the fit: the events of the rows used
//! and the number of columns of the design, aliased ones included, as `royston`
//! counts `length(coef(fit))`.
//!
//! # What is used
//!
//! Only the rows of positive weight. The weights themselves, the strata and
//! the clusters of `data` are not used: as in `royston`, the refit and the
//! variance of the predictor are unweighted, and the weights and strata enter
//! only through the predictor of the fit that produced it. `R.KO` and `C.GH`
//! do not change when a constant is added to the predictor, so it need not be
//! centred.
//!
//! # Cost
//!
//! `C.GH` is a sum over all pairs of rows, which is quadratic in the number
//! of rows used: about 10^9 terms at 45 000 rows. It is computed in parallel
//! and is the same on any number of threads. Set
//! [`RoystonOptions::gonen_heller`] to `false` to skip it; `c_gh` is then
//! `None`.
//!
//! # Errors of the computed numbers
//!
//! The normal scores have the error of the quantile function, a few eps, and
//! a mean of a group of ties adds at most `m eps`. The Cox coefficient
//! inherits those through the solve and the sums of the likelihood, so the
//! tests compare it with a bound derived from the condition number of the
//! information. `R.KO` is a variance, a sum of `n` squares with relative error
//! at most `n eps` per pass. `C.GH` sums `n (n - 1) / 2` terms in `(0, 1)`, each
//! with a relative error below 5 eps, in row sums of at most `n` terms and then
//! over rows, so its relative error is at most `(n + 5) eps` per program.
//!
//! # References
//!
//! 1. Royston P, Sauerbrei W. A new measure of prognostic separation in
//!    survival data. Stat Med 2004;23(5):723-48. doi:10.1002/sim.1621
//! 2. Kent JT, O'Quigley J. Measures of dependence for censored survival data.
//!    Biometrika 1988;75(3):525-34. doi:10.1093/biomet/75.3.525
//! 3. Gonen M, Heller G. Concordance probability and discriminatory power in
//!    proportional hazards regression. Biometrika 2005;92(4):965-70.
//!    doi:10.1093/biomet/92.4.965
//! 4. Nagelkerke NJD. A note on a general definition of the coefficient of
//!    determination. Biometrika 1991;78(3):691-2. doi:10.1093/biomet/78.3.691
//! 5. Blom G. Statistical Estimates and Transformed Beta-Variables. Wiley, 1958.
//! 6. Therneau TM. A Package for Survival Analysis in R. R package `survival`
//!    3.8: `royston` and its help page.

use std::f64::consts::PI;

use epistat_core::InputError;
use epistat_linalg::{Mat, MatRef};
use rayon::prelude::*;

use crate::check::{check_lengths, check_row_count, refuse_non_finite};
use crate::cox::predict::{CoxPredictOptions, CoxPredictType, FittedModel, predict};
use crate::cox::{CoxModelFit, CoxOptions, Ties, fit};
use crate::error::{Error, Result};
use crate::order::par_sort_by_key;
use crate::surv::{SurvBuf, SurvData};
use crate::timefix::TimeFix;
use epistat_stats::dist::normal;
use epistat_stats::sum::compensated;

/// The correction for overfitting of Royston and Sauerbrei: the number of
/// events and of coefficients of the model that produced the predictor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoystonAdjustment {
    /// The number of events the model was fitted on. It must exceed
    /// `n_coef`.
    pub n_events: f64,
    /// The number of coefficients of the model.
    pub n_coef: usize,
}

/// `RoystonAdjustment`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `RoystonAdjustment`")]
pub type Adjustment = RoystonAdjustment;

/// What [`estimate`] takes besides the data and the predictor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoystonOptions {
    /// The rule that makes times equal when they differ by rounding error,
    /// applied to the data first. The default is that of `survival::aeqSurv`.
    pub timefix: TimeFix,
    /// The status code that is the event. Required when the status has more
    /// than one code, as for [`CoxOptions::cause`]. The default is `None`.
    pub cause: Option<u32>,
    /// Give each group of equal predictors the mean of the normal scores of
    /// its ranks, `ties = TRUE` of `royston`. Without it a row takes the score
    /// of the mean rank of its group.
    pub ties: bool,
    /// The correction for overfitting, `adjust` of `royston`. The default is
    /// `None`, no correction, as `adjust = FALSE`.
    pub adjust: Option<RoystonAdjustment>,
    /// Compute `C.GH`, which costs time quadratic in the rows used. The
    /// default is `true`, as `royston` always computes it.
    pub gonen_heller: bool,
}

impl Default for RoystonOptions {
    fn default() -> Self {
        Self {
            timefix: TimeFix::default(),
            cause: None,
            ties: true,
            adjust: None,
            gonen_heller: true,
        }
    }
}

/// The result of [`estimate`] and [`from_fit`].
///
/// An entry is `None` where it is undefined: the Cox fit on the normal scores
/// has no estimate (no events, or a predictor that does not vary), or fewer
/// than two rows were used.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Royston {
    /// `D`, adjusted when an adjustment was asked for.
    pub d: Option<f64>,
    /// The standard error of `D`.
    pub se_d: Option<f64>,
    /// `R.D`, the explained variation of Royston and Sauerbrei, adjusted when
    /// an adjustment was asked for.
    pub r_d: Option<f64>,
    /// `R.KO`, the measure of Kent and O'Quigley, from the variance of the
    /// predictor. `None` for fewer than two rows.
    pub r_ko: Option<f64>,
    /// `R.N`, the Nagelkerke measure of the fit. `None` from [`estimate`],
    /// which has no fit, and from [`from_fit`] when the fit has no log
    /// likelihood.
    pub r_n: Option<f64>,
    /// `C.GH`, the concordance of Gonen and Heller. `None` when it was not
    /// asked for or fewer than two rows were used.
    pub c_gh: Option<f64>,
    /// The coefficient of the normal scores, `D` before scaling and
    /// adjustment.
    pub beta: Option<f64>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight.
    pub n_used: usize,
}

/// What [`from_fit`] takes besides the data, the design and the fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitOptions {
    /// Give a group of equal predictors the mean of its scores, `ties` of
    /// [`RoystonOptions`].
    pub ties: bool,
    /// Correct for overfitting with the events and coefficients of the fit,
    /// `adjust` of `royston`. The default is `false`.
    pub adjust: bool,
    /// Compute `C.GH`, see [`RoystonOptions::gonen_heller`].
    pub gonen_heller: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            ties: true,
            adjust: false,
            gonen_heller: true,
        }
    }
}

/// D and the measures beside it for the linear predictor `eta`, one value per
/// row of `data`.
///
/// `eta` must be finite. See the module documentation for the method and for
/// what is used of `data`.
///
/// # Errors
///
/// An input error when `eta` has another length than `data` or a value that
/// is not finite, when the status has several codes and `opts.cause` is
/// missing or outside them, when the rule for the times finds an interval of
/// length zero, and when an adjustment has `n_events <= n_coef` or a number
/// of events that is not finite.
pub fn estimate(data: &SurvData<'_>, eta: &[f64], opts: &RoystonOptions) -> Result<Royston> {
    if eta.is_empty() && data.n_rows() > 0 {
        return Err(Error::from(InputError::Empty { what: "eta" }));
    }
    check_lengths(&[("data", data.n_rows()), ("eta", eta.len())])?;
    refuse_non_finite(eta, "eta")?;
    if let Some(a) = opts.adjust
        && !(a.n_events.is_finite() && a.n_events > a.n_coef as f64)
    {
        return Err(Error::options(
            "adjust: the number of events must be finite and exceed the number of coefficients",
        ));
    }
    check_row_count(data.n_rows())?;
    let timefixed = data.clone().with_timefix(opts.timefix)?;
    let surv = timefixed.followup()?;
    let code = match (opts.cause, surv.n_codes()) {
        (None, 1) => 1,
        (Some(code), _) => code,
        (None, _) => {
            return Err(Error::missing(
                "cause: the status has several codes, so the event code is required",
            ));
        }
    };
    let event: Vec<bool> = timefixed.cause(code)?.iter().collect();

    // The rows used, in the order given.
    let used: Vec<usize> = timefixed.used_rows().map(|r| r as usize).collect();
    let n = used.len();
    let mut out = Royston {
        d: None,
        se_d: None,
        r_d: None,
        r_ko: None,
        r_n: None,
        c_gh: None,
        beta: None,
        n_rows: data.n_rows(),
        n_used: n,
    };
    if n < 2 {
        return Ok(out);
    }
    let e: Vec<f64> = used.iter().map(|&i| eta[i]).collect();

    out.r_ko = Some(kent_oquigley(&e));
    if opts.gonen_heller {
        out.c_gh = Some(gonen_heller(&e));
    }

    let q = normal_scores(&e, opts.ties);
    let stop: Vec<f64> = used.iter().map(|&i| surv.stop()[i]).collect();
    let status: Vec<i32> = used.iter().map(|&i| i32::from(event[i])).collect();
    let mut builder = SurvBuf::builder(stop, status);
    if let Some(start) = surv.start() {
        builder = builder.start(used.iter().map(|&i| start[i]).collect());
    }
    let buf = builder.build()?;
    let x = Mat::from_fn(n, 1, |i, _| q[i]);
    let refit = fit(
        &buf.as_data(),
        x.as_ref(),
        &CoxOptions {
            ties: Ties::Efron,
            timefix: TimeFix::Exact,
            ..CoxOptions::default()
        },
    )?;
    if let (Some(&Some(b)), Some(v)) = (refit.coefficients.first(), refit.covariance.get(0, 0)) {
        out.beta = Some(b);
        let kappa = (8.0 / PI).sqrt();
        let mut d = b * kappa;
        let mut se = (v * 8.0 / PI).sqrt();
        let mut r2 = b * b / (PI * PI / 6.0 + b * b);
        let mut se_defined = true;
        if let Some(a) = opts.adjust {
            let r = a.n_events / (a.n_events - a.n_coef as f64);
            let t = (1.0 + b * b - r) / r;
            d = sign(b) * sign(t) * (t.abs() * 8.0 / PI).sqrt();
            se_defined = t != 0.0;
            se = se * b.abs() / (r * t.abs().sqrt());
            r2 = 1.0 - r * (1.0 - b * b / (1.0 + b * b));
        }
        out.d = Some(d);
        out.se_d = se_defined.then_some(se);
        out.r_d = Some(r2);
    }
    Ok(out)
}

/// D and the measures beside it for a Cox fit, as `royston(fit)`.
///
/// The predictor is the linear predictor of
/// [`predict`] with its defaults, as
/// `predict.coxph`: centred within the stratum, an aliased coefficient read
/// as 0. `cox` holds the options the fit was made with, whose offset, event code
/// and tie rule for the times are used here. With
/// `opts.adjust` the events are those of the rows used and the coefficients
/// are the columns of `x`. `R.N` is computed from the log partial likelihoods
/// of `fit` and the number of rows of `data`.
///
/// # Errors
///
/// As [`estimate`] and [`predict`], and
/// An input error when `x` has a column count other than the fit's
/// coefficients, or the fit has no estimate (every coefficient `None`) while
/// `x` has columns.
pub fn from_fit(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    fitted: &CoxModelFit,
    cox: &CoxOptions,
    opts: &FitOptions,
) -> Result<Royston> {
    check_lengths(&[("data", data.n_rows()), ("x", x.nrows())])?;
    if x.ncols() != fitted.coefficients.len() {
        return Err(Error::mismatch(format!(
            "x has {} columns but the fit has {} coefficients",
            x.ncols(),
            fitted.coefficients.len()
        )));
    }
    if x.ncols() > 0 && fitted.coefficients.iter().all(Option::is_none) {
        return Err(Error::fit("fit: the fit has no estimate"));
    }
    let eta = predict(
        &FittedModel {
            data,
            x,
            fit: fitted,
            options: cox,
        },
        CoxPredictType::LinearPredictor,
        None,
        &CoxPredictOptions::default(),
    )?
    .fit;
    let code = cox.cause;
    let adjust = if opts.adjust {
        let timefixed = data.clone().with_timefix(cox.timefix)?;
        let ev = timefixed.followup()?;
        let k = match (code, ev.n_codes()) {
            (None, 1) => 1,
            (Some(k), _) => k,
            (None, _) => {
                return Err(Error::missing(
                    "cause: the status has several codes, so the event code is required",
                ));
            }
        };
        Some(RoystonAdjustment {
            n_events: timefixed.cause(k)?.n_events() as f64,
            n_coef: fitted.coefficients.len(),
        })
    } else {
        None
    };
    let mut out = estimate(
        data,
        &eta,
        &RoystonOptions {
            timefix: cox.timefix,
            cause: code,
            ties: opts.ties,
            adjust,
            gonen_heller: opts.gonen_heller,
        },
    )?;
    if let (Some(l1), Some(l0)) = (
        fitted.log_partial_likelihood,
        fitted.null_log_partial_likelihood,
    ) {
        let n = data.n_rows() as f64;
        let lr = -2.0 * (l0 - l1);
        out.r_n = Some((-lr / n).exp_m1() / (2.0 * l0 / n).exp_m1());
    }
    Ok(out)
}

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The normal score of each value, in the order given.
#[expect(
    clippy::float_cmp,
    reason = "values are tied when they are equal, as `duplicated` and `match` in R decide"
)]
fn normal_scores(eta: &[f64], ties: bool) -> Vec<f64> {
    let n = eta.len();
    let nf = n as f64;
    let mut order: Vec<usize> = (0..n).collect();
    par_sort_by_key(&mut order, |i| eta[i]);
    let score = |rank: f64| normal::quantile((rank - 0.375) / (nf + 0.25));
    let mut q = vec![0.0; n];
    let mut lo = 0;
    while lo < n {
        let mut hi = lo + 1;
        while hi < n && eta[order[hi]] == eta[order[lo]] {
            hi += 1;
        }
        let value = if hi - lo == 1 {
            score((lo + 1) as f64)
        } else if ties {
            let mut sum = 0.0;
            for k in lo..hi {
                sum += score((k + 1) as f64);
            }
            sum / (hi - lo) as f64
        } else {
            score(((lo + 1) + hi) as f64 / 2.0)
        };
        for &i in &order[lo..hi] {
            q[i] = value;
        }
        lo = hi;
    }
    q
}

/// `s^2 / (pi^2 / 6 + s^2)` for the variance `s^2` of the values, divisor
/// `n - 1`.
fn kent_oquigley(eta: &[f64]) -> f64 {
    let n = eta.len() as f64;
    let mean = compensated(eta.iter().copied()) / n;
    let v = compensated(eta.iter().map(|e| (e - mean) * (e - mean))) / (n - 1.0);
    v / (PI * PI / 6.0 + v)
}

/// The concordance of Gonen and Heller of the predictor.
fn gonen_heller(eta: &[f64]) -> f64 {
    let mut sorted = eta.to_vec();
    sorted.par_sort_unstable_by(f64::total_cmp);
    let n = sorted.len();
    let rows: Vec<f64> = (0..n - 1)
        .into_par_iter()
        .map(|i| {
            let mut s = 0.0;
            for &b in &sorted[i + 1..] {
                s += 1.0 / (1.0 + (sorted[i] - b).exp());
            }
            s
        })
        .collect();
    let nf = n as f64;
    compensated(rows) * 2.0 / (nf * (nf - 1.0))
}
