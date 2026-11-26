//! Calibration from pseudo-values of the risk: the nearest-neighbour curve of
//! `riskRegression::plotCalibration` and a calibration slope and intercept.
//!
//! The pseudo-value of the event by `t` of row `i` is the leave-one-out
//! jackknife `n F(t) - (n - 1) F_(-i)(t)` of the Kaplan-Meier or Aalen-Johansen
//! estimate `F` of the whole sample. It is a value whose mean given the
//! covariates is the risk by `t` whatever the censoring, so it can be compared
//! with a predicted risk by averaging or by regression, without weights. These
//! are the pseudo-values `Score` makes with `cens.method = "pseudo"` from
//! `prodlim::jackknife`.
//!
//! # What `riskRegression` offers
//!
//! `plotCalibration` has two methods. `method = "quantile"` groups the rows by
//! the predictions and estimates the risk of each group by product limit
//! ([`calibration::estimate`](super::calibration::estimate)); `pseudo = TRUE`
//! only draws the pseudo-values as points and leaves the curve unchanged.
//! `method = "nne"` smooths: with `cens.method = "local"` by a product-limit
//! estimate in a neighbourhood, and with another value by averaging the
//! pseudo-values in the neighbourhood. [`smoothed_curve`] is that average.
//! Neither `plotCalibration` nor `Score` computes a calibration slope or
//! intercept, or confidence limits of the observed risk, and none are computed
//! for the curve here.
//!
//! # Smoothed curve
//!
//! The predictions are rounded to `rounding` decimals (2 in `plotCalibration`)
//! unless the bandwidth is at least 1, with R's rule for `round`. Let `u_1 < ...
//! < u_m` be the distinct values and `c_k` the number of rows with a value up
//! to `u_k`. With `r = floor(bandwidth N)` the neighbourhood of `u_k` is the
//! rows at sorted positions `c_k - r` to `c_k + r`, clipped to `1..N`, and
//! widened or cut so that it holds whole groups of tied values: its first
//! position moves back to the first of its tie group, and its last moves back
//! to the end of the previous group when it would cut a group. The point of the
//! curve is `u_k` and the mean of the pseudo-values of the rows of the
//! neighbourhood, the sum taken in the order of the rows. With a bandwidth of 1
//! or more the curve is the one point, the mean prediction and the mean
//! pseudo-value. `prodlim::neighborhood` with a box kernel.
//!
//! The bandwidth is the caller's: `plotCalibration` chooses it by the direct
//! plug-in rule of `KernSmooth::dpik`, which is not implemented, and its
//! `"smooth"` value `N^(-1/4)` is [`Bandwidth::Smooth`].
//!
//! # Slope and intercept
//!
//! The pseudo-values `y_i` of the risk by `t` are regressed on the predictions
//! by a generalised estimating equation with independence, a Gaussian variance
//! and a link, as `geepack::geese.fit(mean.link = "logit"` or `"cloglog",
//! family = gaussian())` with one row per subject:
//!
//! ```text
//! E y_i = g^-1(a + b g(R_i)),     g = logit or complementary log-log.
//! ```
//!
//! A prediction that is right on average at every level has `a = 0` and `b =
//! 1`. The slope below 1 means predictions that are too extreme, above 1 too
//! cautious, on the scale of `g`. The complementary log-log scale is the scale
//! of a proportional-hazards model, on which the slope of a Cox model's linear
//! predictor is read. The covariance is the sandwich, `A^-1 M A^-1` with the
//! information `A = sum d_i^2 x_i x_i'` and `M = sum d_i^2 (y_i - mu_i)^2 x_i
//! x_i'`, `d_i = d mu_i / d eta_i` and one row per subject, which is `geese`'s
//! robust variance. This is the logistic recalibration of a risk prediction on
//! pseudo-values, not an estimate that `riskRegression` has a counterpart of; it
//! is checked against `geese.fit` on the pseudo-values of `prodlim`.
//!
//! The predictions must lie strictly between 0 and 1, since the link is
//! infinite at the ends.
//!
//! # Errors of the computed numbers
//!
//! The pseudo-values are sums over the risk sets of at most `n` terms. The
//! regression is solved to a relative step of `1e-8`, which the options can
//! tighten, and its coefficients are as accurate as the information's
//! condition number times the rounding unit allows. The tests state the
//! bounds.
//!
//! # References
//!
//! 1. Andersen PK, Perme MP. Pseudo-observations in survival analysis. Stat
//!    Methods Med Res 2010;19(1):71-99. doi:10.1177/0962280209105020
//! 2. Gerds TA, Andersen PK, Kattan MW. Calibration plots for risk prediction
//!    models in the presence of competing risks. Stat Med 2014;33(18):3191-203.
//!    doi:10.1002/sim.6152
//! 3. Austin PC, Harrell FE, van Klaveren D. Graphical calibration curves and
//!    the integrated calibration index (ICI) for survival models. Stat Med
//!    2020;39(21):2714-42. doi:10.1002/sim.8570
//! 4. Liang KY, Zeger SL. Longitudinal data analysis using generalized linear
//!    models. Biometrika 1986;73(1):13-22.

#![expect(
    deprecated,
    reason = "the fits take their data as separate arguments, the epistat-glm 0.2 form"
)]
#![expect(
    clippy::float_cmp,
    reason = "tied predictions are equal values, which is the definition of a tie"
)]

use epistat_glm::{Family, Glm};
use epistat_inference::sandwich::{meat, sandwich};
use epistat_linalg::{Attempt, Mat, inverse, weighted_gram};

use super::prepare::{check_times, sort_rows};
use crate::error::{Error, Result};
use crate::pseudo::{Method, PseudoOptions, Target, pseudo};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// The size of the neighbourhoods of the smoothed curve, as a fraction of the
/// number of rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bandwidth {
    /// This fraction, `bandwidth = ` a number of `neighborhood`. At least 1
    /// gives one point.
    Fixed(f64),
    /// `N^(-1/4)`, the `"smooth"` bandwidth of `prodlim`.
    Smooth,
}

/// What [`smoothed_curve`] takes besides the data, the predictions and the
/// time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothOptions {
    /// The rule that makes times equal when they differ by rounding error,
    /// applied to the data first. `Exact`: `Score` applies none.
    pub timefix: TimeFix,
    /// The event of interest. 1, the first cause as in `Score`.
    pub cause: u32,
    /// The size of the neighbourhoods. It has no default.
    pub bandwidth: Bandwidth,
    /// Round the predictions to this many decimals before smoothing, as
    /// `round` of `plotCalibration`, which is 2. `None` leaves them as they
    /// are. Not used when the bandwidth is at least 1.
    pub rounding: Option<i32>,
}

impl SmoothOptions {
    /// The options with the given bandwidth and the defaults of `Score` and
    /// `plotCalibration` for the others.
    #[must_use]
    pub fn new(bandwidth: Bandwidth) -> Self {
        Self {
            timefix: TimeFix::Exact,
            cause: 1,
            bandwidth,
            rounding: Some(2),
        }
    }
}

/// One point of the smoothed curve.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothedPoint {
    /// The distinct predicted risk, after rounding.
    pub predicted: f64,
    /// The mean pseudo-value of the rows of the neighbourhood.
    pub observed: f64,
    /// The number of rows of the neighbourhood.
    pub n_neighbors: usize,
}

/// The smoothed calibration curve at one time.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothedCurve {
    /// The time the risks are for.
    pub time: f64,
    /// The fraction of rows in a neighbourhood that was used.
    pub bandwidth: f64,
    /// One point for each distinct predicted risk, ascending.
    pub points: Vec<SmoothedPoint>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight.
    pub n_used: usize,
}

/// The pseudo-values of the risk by `time` of the used rows, in the order of the
/// data, with the rows they belong to.
fn pseudo_risk(
    data: &SurvData<'_>,
    time: f64,
    timefix: TimeFix,
    cause: u32,
) -> Result<(Vec<u32>, Vec<f64>)> {
    if data.cluster().is_some() || data.id().is_some() || data.stratum().is_some() {
        return Err(Error::input(
            "pseudo-values of the risk are of the whole sample: the data must have no \
             cluster, id or stratum column",
        ));
    }
    let sorted = sort_rows(data, timefix, cause)?;
    check_times(&[time], sorted.max_time)?;
    let options = PseudoOptions {
        method: Method::Jackknife,
        timefix,
        ..PseudoOptions::new(Target::CumulativeIncidence { cause })
    };
    let made = pseudo(data, &[time], &options)?;
    // Without a cluster or id column the subjects are the used rows, ascending.
    if made.subject.windows(2).any(|w| w[1] <= w[0]) {
        return Err(Error::input("internal: pseudo-value subjects are not rows"));
    }
    Ok((made.subject, made.values))
}

/// `floor(x)` of a value that is at least 0 and below the number of rows.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "x is a non-negative fraction of the number of rows, which fits a usize"
)]
fn floor_to_usize(x: f64) -> usize {
    x.floor() as usize
}

/// R's `round(x, digits)` of R 4.0.0 and later: of the two decimals either side
/// of `x` the nearer, and the one with the even last digit when they are
/// equally near.
fn round_like_r(x: f64, digits: i32) -> f64 {
    if !x.is_finite() || digits > 15 {
        return x;
    }
    let negative = x < 0.0;
    let x = x.abs();
    let p10 = 10f64.powi(digits);
    let scaled = x * p10;
    let lower = scaled.floor();
    let (xd, xu) = (lower / p10, scaled.ceil() / p10);
    let (dd, du) = (x - xd, xu - x);
    let even = (lower % 2.0) == 0.0;
    let r = if dd < du || (even && dd == du) {
        xd
    } else {
        xu
    };
    if negative { -r } else { r }
}

/// The calibration curve of predicted risk by nearest-neighbour smoothing of
/// pseudo-values, as `plotCalibration(method = "nne")` with pseudo-values.
///
/// `risk` has one value per row of the data: the predicted probability of the
/// event of interest by `time`.
///
/// # Errors
///
/// [`Error::Input`] when the data is interval-censored, has a start column, a
/// cluster, id or stratum column or a weight other than 0 and 1, has no row of
/// positive weight, or `cause` is outside the event codes; when `time` is not
/// finite or after the last observed time; when `risk` has another length than
/// the data or a value outside `[0, 1]`; when the bandwidth is not positive
/// and finite; and when fewer than two rows are used.
pub fn smoothed_curve(
    data: &SurvData<'_>,
    risk: &[f64],
    time: f64,
    opts: &SmoothOptions,
) -> Result<SmoothedCurve> {
    let (rows, y) = pseudo_risk(data, time, opts.timefix, opts.cause)?;
    let n = rows.len();
    if n < 2 {
        return Err(Error::input("at least two rows are needed to smooth"));
    }
    check_risk(risk, data.n_rows(), false)?;
    let nf = n as f64;
    let bandwidth = match opts.bandwidth {
        Bandwidth::Fixed(b) => b,
        Bandwidth::Smooth => nf.powf(-0.25),
    };
    if !(bandwidth.is_finite() && bandwidth > 0.0) {
        return Err(Error::input("the bandwidth must be positive and finite"));
    }
    let mut p: Vec<f64> = rows.iter().map(|&r| risk[r as usize]).collect();
    if bandwidth >= 1.0 {
        let mean_p = p.iter().sum::<f64>() / nf;
        let mean_y = y.iter().sum::<f64>() / nf;
        return Ok(SmoothedCurve {
            time,
            bandwidth,
            points: vec![SmoothedPoint {
                predicted: mean_p,
                observed: mean_y,
                n_neighbors: n,
            }],
            n_rows: data.n_rows(),
            n_used: n,
        });
    }
    if let Some(digits) = opts.rounding {
        for v in &mut p {
            *v = round_like_r(*v, digits);
        }
    }
    // Positions in ascending order of the prediction, ties by row.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]).then(a.cmp(&b)));
    // Per sorted position: the end (1-based count) and size of its tie group.
    let mut group_end = vec![0usize; n];
    let mut group_size = vec![0usize; n];
    let mut ends: Vec<usize> = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = start;
        while end < n && p[order[end]] == p[order[start]] {
            end += 1;
        }
        for k in start..end {
            group_end[k] = end;
            group_size[k] = end - start;
        }
        ends.push(end);
        start = end;
    }
    let radius = floor_to_usize(bandwidth * nf);
    let mut points = Vec::with_capacity(ends.len());
    for &c in &ends {
        // Positions are 1-based, as in the definition.
        let first = match c.saturating_sub(radius) {
            0 => 1,
            from => group_end[from - 1] - group_size[from - 1] + 1,
        };
        let last = if c + radius > n {
            n
        } else if group_end[c + radius - 1] > c + radius {
            group_end[c + radius - 1] - group_size[c + radius - 1]
        } else {
            c + radius
        };
        if last < first {
            return Err(Error::input("internal: an empty neighbourhood"));
        }
        let mut members: Vec<usize> = order[first - 1..last].to_vec();
        members.sort_unstable();
        let mean = members.iter().map(|&i| y[i]).sum::<f64>() / members.len() as f64;
        points.push(SmoothedPoint {
            predicted: p[order[c - 1]],
            observed: mean,
            n_neighbors: members.len(),
        });
    }
    Ok(SmoothedCurve {
        time,
        bandwidth,
        points,
        n_rows: data.n_rows(),
        n_used: n,
    })
}

fn check_risk(risk: &[f64], n_rows: usize, open: bool) -> Result<()> {
    if risk.len() != n_rows {
        return Err(Error::input(format!(
            "risk has {} values for {n_rows} rows",
            risk.len()
        )));
    }
    for (position, &p) in risk.iter().enumerate() {
        let ok = if open {
            p > 0.0 && p < 1.0
        } else {
            (0.0..=1.0).contains(&p)
        };
        if !ok {
            return Err(Error::input(format!(
                "risk: row {position} is {p}, which is not a probability{}",
                if open {
                    " strictly between 0 and 1"
                } else {
                    ""
                }
            )));
        }
    }
    Ok(())
}

/// The scale on which the slope and the intercept are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecalibrationScale {
    /// The log odds, `log(R / (1 - R))`.
    Logit,
    /// The log cumulative hazard, `log(-log(1 - R))`.
    Cloglog,
}

impl RecalibrationScale {
    const fn link(self) -> epistat_glm::Link {
        match self {
            Self::Logit => epistat_glm::Link::Logit,
            Self::Cloglog => epistat_glm::Link::Cloglog,
        }
    }
}

/// What [`slope_intercept`] takes besides the data, the predictions and the
/// time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecalibrationOptions {
    /// The rule that makes times equal when they differ by rounding error,
    /// applied to the data first. `Exact`.
    pub timefix: TimeFix,
    /// The event of interest. 1.
    pub cause: u32,
    /// The scale. It has no default.
    pub scale: RecalibrationScale,
    /// A step of Fisher scoring smaller than this, relative to the larger of 1
    /// and the largest coefficient, ends the iteration: `1e-8`, the tolerance of
    /// `glm.control`.
    pub tolerance: f64,
    /// The most steps taken: 25.
    pub max_iterations: usize,
}

impl RecalibrationOptions {
    /// The options for a scale with the defaults of `glm.control`.
    #[must_use]
    pub fn new(scale: RecalibrationScale) -> Self {
        Self {
            timefix: TimeFix::Exact,
            cause: 1,
            scale,
            tolerance: 1e-8,
            max_iterations: 25,
        }
    }
}

/// The calibration intercept and slope at one time.
#[derive(Debug, Clone, PartialEq)]
pub struct Recalibration {
    /// The time the risks are for.
    pub time: f64,
    /// The intercept `a`: 0 for predictions that are right on average.
    pub intercept: f64,
    /// The slope `b`: 1 for predictions that are right at every level.
    pub slope: f64,
    /// The sandwich covariance of `(a, b)`, row-major.
    pub covariance: [f64; 4],
    /// The information matrix `A` of the fit, row-major.
    pub information: [f64; 4],
    /// The condition number of `A` in the 1-norm. The coefficients lose about
    /// this many times the rounding unit to the solve.
    pub information_condition: f64,
    /// The scale.
    pub scale: RecalibrationScale,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight.
    pub n_used: usize,
    /// Whether the iteration stopped on its tolerance.
    pub converged: bool,
    /// The number of steps taken.
    pub iterations: usize,
}

impl Recalibration {
    /// The standard error of the intercept.
    #[must_use]
    pub fn std_err_intercept(&self) -> f64 {
        self.covariance[0].max(0.0).sqrt()
    }

    /// The standard error of the slope.
    #[must_use]
    pub fn std_err_slope(&self) -> f64 {
        self.covariance[3].max(0.0).sqrt()
    }
}

/// The calibration intercept and slope of the predicted risks by `time`, by
/// regression of their pseudo-values on the predictions. See the module
/// documentation.
///
/// # Errors
///
/// As [`smoothed_curve`], except that `risk` must lie strictly between 0 and 1,
/// and in addition when the mean of the pseudo-values is outside the range of
/// the link so that there is no start, the predictions are all equal, or the
/// iteration leaves the range of the link.
pub fn slope_intercept(
    data: &SurvData<'_>,
    risk: &[f64],
    time: f64,
    opts: &RecalibrationOptions,
) -> Result<Recalibration> {
    let (rows, y) = pseudo_risk(data, time, opts.timefix, opts.cause)?;
    let n = rows.len();
    check_risk(risk, data.n_rows(), true)?;
    if !(opts.tolerance.is_finite() && opts.tolerance > 0.0) {
        return Err(Error::input("the tolerance must be positive and finite"));
    }
    if opts.max_iterations == 0 {
        return Err(Error::input("at least one iteration must be allowed"));
    }
    let link = opts.scale.link();
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            1.0
        } else {
            link.link(risk[rows[i] as usize])
        }
    });
    let fit = gee_independence(&x, &y, link, opts)?;
    Ok(Recalibration {
        time,
        intercept: fit.beta[0],
        slope: fit.beta[1],
        covariance: fit.covariance,
        information: fit.information,
        information_condition: fit.condition,
        scale: opts.scale,
        n_rows: data.n_rows(),
        n_used: n,
        converged: fit.converged,
        iterations: fit.iterations,
    })
}

struct Gee {
    beta: [f64; 2],
    covariance: [f64; 4],
    information: [f64; 4],
    condition: f64,
    converged: bool,
    iterations: usize,
}

/// Fisher scoring of the Gaussian-variance model with `link`, one row per
/// subject, and its robust covariance.
fn gee_independence(
    x: &Mat<f64>,
    y: &[f64],
    link: epistat_glm::Link,
    opts: &RecalibrationOptions,
) -> Result<Gee> {
    let n = x.nrows();
    let aliased = || {
        Error::input(
            "the predictions are all equal on the scale of the link: the slope is not \
             estimable",
        )
    };
    let mean = y.iter().sum::<f64>() / n as f64;
    if !(mean > 0.0 && mean < 1.0) {
        return Err(Error::input(format!(
            "the mean {mean} of the pseudo-values is outside the range of the link, so \
             there is no start"
        )));
    }
    let eta0 = link.link(mean);
    let derivative = link.derivative(eta0);
    let z: Vec<f64> = y.iter().map(|&v| eta0 + (v - mean) / derivative).collect();
    let start = Glm::new(Family::Gaussian, epistat_glm::Link::Identity)
        .fit_slices(x.as_ref(), &z, None, None, None)
        .map_err(|e| Error::input(format!("the start could not be found: {e}")))?;
    let mut beta: Vec<f64> = start
        .coefficients
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(aliased)?;
    let model = Glm {
        max_iterations: 1,
        ..Glm::new(Family::Gaussian, link)
    };
    let (mut converged, mut iterations) = (false, 0);
    let mut solved = None;
    for _ in 0..opts.max_iterations {
        iterations += 1;
        let step = model
            .fit_slices(x.as_ref(), y, None, None, Some(&beta))
            .map_err(|e| Error::input(format!("the regression could not be fitted: {e}")))?;
        let next: Vec<f64> = step
            .coefficients
            .iter()
            .copied()
            .collect::<Option<_>>()
            .ok_or_else(aliased)?;
        let largest = beta.iter().fold(1.0f64, |m, b| m.max(b.abs()));
        converged = next
            .iter()
            .zip(&beta)
            .all(|(a, b)| (a - b).abs() <= opts.tolerance * largest);
        beta = next;
        solved = Some(step);
        if converged {
            break;
        }
    }
    let solved = solved.ok_or_else(|| Error::input("no iteration ran"))?;
    let fitted = solved.fitted;
    let d: Vec<f64> = solved
        .linear_predictor
        .iter()
        .map(|&e| link.derivative(e))
        .collect();
    let weight: Vec<f64> = d.iter().map(|v| v * v).collect();
    let a = weighted_gram(x.as_ref(), &weight)
        .map_err(|e| Error::input(format!("the information could not be formed: {e}")))?;
    let scored = Mat::from_fn(n, 2, |i, j| d[i] * x[(i, j)]);
    let m = meat(scored.as_ref(), |i| y[i] - fitted[i], None)
        .map_err(|e| Error::input(format!("the sandwich could not be formed: {e}")))?;
    let cov = sandwich(a.as_ref(), m.as_ref(), &Attempt::SYMMETRIC)
        .map_err(|e| Error::input(format!("the information could not be inverted: {e}")))?;
    let a_inverse = inverse(a.as_ref(), &Attempt::SYMMETRIC)
        .map_err(|e| Error::input(format!("the information could not be inverted: {e}")))?
        .x;
    let one_norm = |m: &Mat<f64>| {
        (0..2)
            .map(|j| (0..2).map(|i| m[(i, j)].abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    };
    Ok(Gee {
        beta: [beta[0], beta[1]],
        covariance: [cov[(0, 0)], cov[(0, 1)], cov[(1, 0)], cov[(1, 1)]],
        information: [a[(0, 0)], a[(0, 1)], a[(1, 0)], a[(1, 1)]],
        condition: one_norm(&a) * one_norm(&a_inverse),
        converged,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::round_like_r;

    #[derive(Deserialize)]
    struct Rounding {
        x: Vec<f64>,
        d0: Vec<f64>,
        d1: Vec<f64>,
        d2: Vec<f64>,
        d3: Vec<f64>,
    }

    #[derive(Deserialize)]
    struct Reference {
        rounding: Rounding,
    }

    /// `round(x, digits)` of R 4.6.2 on values that include exact halves
    /// (0.125) and values just either side of one (0.005, 0.015).
    #[test]
    fn rounding_is_that_of_r() {
        let r: Reference = epistat_testkit::load_json(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/metrics_recalibration.json"
        ));
        for (digits, want) in [
            (0, &r.rounding.d0),
            (1, &r.rounding.d1),
            (2, &r.rounding.d2),
            (3, &r.rounding.d3),
        ] {
            for (x, w) in r.rounding.x.iter().zip(want) {
                assert_eq!(
                    round_like_r(*x, digits).to_bits(),
                    w.to_bits(),
                    "round({x}, {digits})"
                );
            }
        }
    }
}
