//! Survival curves for a covariate path that changes over time, as
//! `survfit.coxph` with `newdata` in counting-process form and `id`.
//!
//! A subject is given as consecutive intervals `(start_i, stop_i]`, each with
//! its covariates, offset and stratum. The curve of the subject adds, over the
//! baseline times in each interval, the baseline increments of that interval's
//! stratum multiplied by the relative risk `theta_i` of the interval:
//!
//! ```text
//! H(t)   = sum over s <= t of theta(s) h(s)
//! S(t)   = exp(-H(t))            or     exp(sum over s <= t of theta(s) l(s))
//! var H  = sum over s <= t of theta(s)^2 v(s)  +  d(t)' V d(t)
//! d(t)   = sum over s <= t of theta(s) (h(s) z(s) - xbar(s)),
//! ```
//!
//! where `h`, `v`, `xbar` and `l` are the increments of the baseline of the
//! stratum the subject is in at `s`, `z(s)` the covariates less the reference
//! of the baseline, and `V` the covariance of the coefficients. This is the
//! delta-method variance of Tsiatis \[1\] with the covariates a step function
//! of time, which `survfit.coxph` gives for the same input.
//!
//! The times of the curve are the baseline times of the strata. Where the
//! intervals of a subject leave a gap, `survfit.coxph` shifts the times after
//! the gap back by its length. The times here are the baseline times as they
//! are, and no hazard accrues in the gap. Intervals of one subject must be in
//! time order and may not overlap.
//!
//! # References
//!
//! 1. Tsiatis AA. A large sample study of Cox's regression model. Ann Stat
//!    1981;9(1):93-108. doi:10.1214/aos/1176345335
//! 2. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000.

use std::collections::HashMap;

use epistat_stats::normal;
use faer::MatRef;

use super::{
    CurveType, FittedModel, Model, NewData, SurvfitOptions, bands, centred_row, curve_baseline,
    quad,
};
use crate::check::check_lengths;
use crate::error::{Error, Result};

/// The covariate path of the subjects, one row per interval.
#[derive(Debug, Clone, Copy)]
pub struct PathData<'a> {
    /// The covariates of each interval, in the design of the fit.
    pub x: MatRef<'a, f64>,
    /// The subject of each interval.
    pub id: &'a [u32],
    /// The start of each interval.
    pub start: &'a [f64],
    /// The stop of each interval.
    pub stop: &'a [f64],
    /// The caller's stratum code of each interval, required when the fit has
    /// strata.
    pub stratum: Option<&'a [u32]>,
    /// The offset of each interval, zero by default.
    pub offset: Option<&'a [f64]>,
}

/// The curve of one subject, at the baseline times inside its intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct PathCurve {
    /// The subject.
    pub id: u32,
    /// The times, ascending when the intervals are in order.
    pub time: Vec<f64>,
    /// The index in [`PathCurves::strata`] of the stratum each time is read from.
    pub stratum: Vec<usize>,
    /// The sum of the case weights at risk at each time, in its stratum.
    pub n_risk: Vec<f64>,
    /// The sum of the case weights of the events at each time.
    pub n_event: Vec<f64>,
    /// The sum of the case weights that leave without the event.
    pub n_censor: Vec<f64>,
    /// The cumulative hazard.
    pub cumulative_hazard: Vec<f64>,
    /// The survival.
    pub survival: Vec<f64>,
    /// The standard error of the cumulative hazard, that is of `log S`.
    pub std_err: Option<Vec<f64>>,
    /// The lower confidence limit, `None` where it is undefined.
    pub lower: Option<Vec<Option<f64>>>,
    /// The upper confidence limit.
    pub upper: Option<Vec<Option<f64>>>,
}

/// The curves of [`survfit_paths`], one per subject in order of first appearance.
#[derive(Debug, Clone, PartialEq)]
pub struct PathCurves {
    /// The caller's code of each stratum, `None` without strata.
    pub strata: Vec<Option<u32>>,
    /// One curve per subject.
    pub curves: Vec<PathCurve>,
}

/// One survival curve per subject from its covariate path, as `survfit.coxph`
/// with a counting-process `newdata` and `id`. See the module documentation.
///
/// The options are those of [`survfit`](super::survfit): the form of the
/// survival, the confidence limits, `start_time` and `censor_times`.
///
/// # Errors
///
/// [`Error::Input`] as for [`survfit`](super::survfit), and when the lengths of
/// the columns of `paths` differ, a start or stop is not finite, a start is not
/// before its stop, or the intervals of a subject are out of order or overlap.
pub fn survfit_paths(
    model: &FittedModel<'_>,
    paths: &PathData<'_>,
    opts: &SurvfitOptions<'_>,
) -> Result<PathCurves> {
    if !(opts.conf_level > 0.0 && opts.conf_level < 1.0) {
        return Err(Error::input("conf_level: must be between 0 and 1"));
    }
    let m = Model::new(model, opts.variance)?;
    let rows = paths.x.nrows();
    check_lengths(&[
        ("paths x", rows),
        ("paths id", paths.id.len()),
        ("paths start", paths.start.len()),
        ("paths stop", paths.stop.len()),
    ])?;
    for (&a, &b) in paths.start.iter().zip(paths.stop) {
        if !(a.is_finite() && b.is_finite()) {
            return Err(Error::input("paths: a start or stop is not finite"));
        }
        if a >= b {
            return Err(Error::input("paths: a start is not before its stop"));
        }
    }
    let as_new = NewData {
        x: paths.x,
        stratum: paths.stratum,
        start: None,
        stop: None,
        offset: paths.offset,
    };
    m.check_new(&as_new)?;
    let strata = m.new_strata(&as_new)?;
    let mut order: Vec<u32> = Vec::new();
    let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &id) in paths.id.iter().enumerate() {
        members
            .entry(id)
            .or_insert_with(|| {
                order.push(id);
                Vec::new()
            })
            .push(i);
    }
    for rows_of in members.values() {
        for pair in rows_of.windows(2) {
            if paths.start[pair[1]] < paths.stop[pair[0]] {
                return Err(Error::input(
                    "paths: the intervals of a subject must be in time order and not overlap",
                ));
            }
        }
    }
    let baseline = curve_baseline(&m, opts)?;
    let product_limit = opts.curve_type == CurveType::KalbfleischPrentice;
    let p = m.p;
    let z_value = normal::quantile(1.0 - (1.0 - opts.conf_level) / 2.0);
    let mut z = vec![0.0; p];
    let mut d = vec![0.0; p];
    let mut curves = Vec::with_capacity(order.len());
    for &id in &order {
        let mut curve = PathCurve {
            id,
            time: Vec::new(),
            stratum: Vec::new(),
            n_risk: Vec::new(),
            n_event: Vec::new(),
            n_censor: Vec::new(),
            cumulative_hazard: Vec::new(),
            survival: Vec::new(),
            std_err: None,
            lower: None,
            upper: None,
        };
        let mut se = Vec::new();
        d.fill(0.0);
        let (mut h, mut log_s, mut var1) = (0.0, 0.0, 0.0);
        for &i in &members[&id] {
            centred_row(paths.x, i, &baseline.centre, &mut z);
            let eta = m.dot_beta(&z) + paths.offset.map_or(0.0, |o| o[i]) - baseline.offset_centre;
            let theta = eta.exp();
            if !theta.is_finite() {
                return Err(Error::input("the risk score of a row is not finite"));
            }
            let s = strata[i];
            let base = &baseline.strata[s];
            let (lo, hi) = (
                base.count_up_to(paths.start[i]),
                base.count_up_to(paths.stop[i]),
            );
            for k in lo..hi {
                h += base.hazard[k] * theta;
                if product_limit {
                    log_s += base.log_km[k] * theta;
                }
                var1 += base.variance_increment[k] * theta * theta;
                for a in 0..p {
                    d[a] += (base.hazard[k] * z[a] - base.xbar_increment[k * p + a]) * theta;
                }
                curve.time.push(base.time[k]);
                curve.stratum.push(s);
                curve.n_risk.push(base.n_risk[k]);
                curve.n_event.push(base.n_events[k]);
                curve.n_censor.push(base.n_censor[k]);
                curve.cumulative_hazard.push(h);
                curve.survival.push(if product_limit {
                    log_s.exp()
                } else {
                    (-h).exp()
                });
                if opts.se_fit {
                    se.push((var1 + quad(&m.variance, p, &d)).sqrt());
                }
            }
        }
        if opts.se_fit {
            let (lower, upper) = bands(opts.conf_type, &curve.survival, Some(&se), z_value);
            curve.lower = lower;
            curve.upper = upper;
            curve.std_err = Some(se);
        }
        curves.push(curve);
    }
    Ok(PathCurves {
        strata: baseline.strata.iter().map(|s| s.code).collect(),
        curves,
    })
}
