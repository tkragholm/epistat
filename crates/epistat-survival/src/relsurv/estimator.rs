//! Net survival from the follow-up of a cohort and its reference hazards.

#![expect(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "times and weights are compared with the values they were copied from, and the interval of a time is a floor of a non-negative quotient"
)]

use super::ExpectedHazard;
use super::cohort::{self, ExpectedMethod, Weighting, sweep};
use super::expected::Pieces;
use crate::error::{Error, Result};
use crate::par;
use crate::surv::SurvData;
use epistat_core::Level;
use epistat_stats::dist::normal;

const BLOCK: usize = 1024;
const WAVE: usize = 64;
/// The most pairs of a person at risk and a time point the exact estimator
/// visits.
const EXACT_LIMIT: f64 = 3e8;

/// The estimator of the cumulative excess hazard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Ederer II: the observed hazard minus the mean reference hazard of the
    /// persons at risk. It is not a consistent estimator of net survival when
    /// the censoring depends on age or other determinants of the reference
    /// hazard.
    EdererII,
    /// Pohar Perme, Stare and Esteve: every person is weighted by the inverse
    /// of the expected survival since the origin, which makes the estimator
    /// consistent for the net survival.
    PoharPerme,
    /// The relative survival of Ederer I: the observed survival
    /// `exp(-Lambda_O)` divided by the expected survival of Ederer I, the
    /// mean expected survival of every person followed to the time
    /// ([`super::ExpectedMethod::EdererI`]). The expected part ignores
    /// censoring, so it suits a cohort followed to a common end.
    EdererI,
    /// The relative survival of Hakulinen: the observed survival divided by
    /// Hakulinen's expected survival, in which each person counts until its
    /// potential follow-up, [`NetOptions::potential`]
    /// ([`super::ExpectedMethod::Hakulinen`]).
    Hakulinen,
}

/// How the limits of the net survival are formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// Normal limits for `ln Lambda`, where `Lambda` is the cumulative excess
    /// hazard.
    LogLog,
    /// Normal limits for the net survival itself.
    Plain,
    /// Normal limits for `ln R`, the log net survival.
    Log,
}

/// The time points at which the estimate is formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Grid {
    /// The continuous estimator, evaluated exactly at every distinct exit and
    /// entry time. Its cost is the number of persons at risk summed over the
    /// time points, and it is refused above 3e8.
    EventTimes,
    /// Intervals of this width from time 0, in the units of the follow-up
    /// time. The estimate has person-time denominators and approximates the
    /// continuous one, as the module documentation describes, and its cost is
    /// linear in the follow-up covered.
    Width(f64),
}

/// The settings of [`net_survival`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetOptions<'a> {
    /// The estimator.
    pub method: Method,
    /// The time points.
    pub grid: Grid,
    /// The confidence level of the limits.
    pub level: Level,
    /// The scale of the limits.
    pub transform: Transform,
    /// The potential follow-up of every row of the data, which
    /// [`Method::Hakulinen`] needs and the other methods do not take: the time
    /// a row would have been followed to had it not died, its exit time if it
    /// was censored. `relsurv::rs.surv` sets it to `fin.date` less the date
    /// of diagnosis for those who died.
    pub potential: Option<&'a [f64]>,
}

impl Default for NetOptions<'_> {
    fn default() -> Self {
        Self {
            method: Method::PoharPerme,
            grid: Grid::EventTimes,
            level: Level::default(),
            transform: Transform::LogLog,
            potential: None,
        }
    }
}

/// The net survival of one stratum.
///
/// All vectors have one entry for each reported time. With
/// [`Grid::EventTimes`] these are the distinct exit times; with
/// [`Grid::Width`] the end of each interval in which someone is at risk.
#[derive(Clone, Debug, PartialEq)]
pub struct NetSurvival {
    /// The level of the stratum, 0 when the data has no strata.
    pub stratum: usize,
    /// The reported times.
    pub times: Vec<f64>,
    /// The estimate of the cumulative excess hazard at each time.
    pub cumulative_excess_hazard: Vec<f64>,
    /// The variance of the cumulative excess hazard.
    pub variance: Vec<f64>,
    /// The net survival `exp(-Lambda)`.
    pub survival: Vec<f64>,
    /// The lower limit, `None` where the transform is undefined, as the log
    /// log scale is for a cumulative hazard that is not positive.
    pub lower: Vec<Option<f64>>,
    /// The upper limit, `None` where the lower is.
    pub upper: Vec<Option<f64>>,
    /// The number of persons at risk at the time, or with an interval the
    /// person-time at risk in it divided by its width.
    pub n_at_risk: Vec<f64>,
    /// The number of events at the time, or in the interval.
    pub events: Vec<f64>,
}

struct Raw {
    times: Vec<f64>,
    lambda: Vec<f64>,
    variance: Vec<f64>,
    at_risk: Vec<f64>,
    events: Vec<f64>,
}

fn check(data: &SurvData<'_>, expected: &ExpectedHazard, opts: &NetOptions<'_>) -> Result<()> {
    if expected.n_rows() != data.n_rows() {
        return Err(Error::invalid(format!(
            "the expected hazards are for {} rows and the data has {}",
            expected.n_rows(),
            data.n_rows()
        )));
    }
    if let Grid::Width(w) = opts.grid
        && !(w.is_finite() && w > 0.0)
    {
        return Err(Error::invalid("the grid width must be finite and positive"));
    }
    if data.followup()?.n_codes() != 1 {
        return Err(Error::invalid(
            "net survival needs a binary status: use SurvData::cause for competing events",
        ));
    }
    if let Some(w) = data.weight()
        && w.iter().any(|&v| v != 0.0 && v != 1.0)
    {
        return Err(Error::invalid(
            "net survival takes weights of 0 and 1 only, which select rows",
        ));
    }
    let hakulinen = opts.method == Method::Hakulinen;
    if hakulinen && opts.potential.is_none() {
        return Err(Error::invalid(
            "the Hakulinen estimator needs the potential follow-up of every row",
        ));
    }
    if !hakulinen && opts.potential.is_some() {
        return Err(Error::invalid(
            "only the Hakulinen estimator takes a potential follow-up",
        ));
    }
    Ok(())
}

/// Net survival by Ederer II or Pohar Perme, or relative survival by Ederer I
/// or Hakulinen, for each stratum of the data.
///
/// `data` is the follow-up with entry (`start`) and exit (`stop`) times in the
/// time scale of the reference hazards of `expected`, and status 1 for death
/// from any cause and 0 for censoring. The strata of the data, if any, give
/// one curve each.
///
/// # Errors
///
/// An input error for expected hazards of another number of rows than the
/// data, a status with more than one event code, weights other than 0 and 1,
/// a grid width that is not positive, an entry
/// before time 0, follow-up beyond what `expected` covers, a potential
/// follow-up missing for Hakulinen or given for another method, delayed entry
/// or a walk off the table for Ederer I and Hakulinen (see
/// [`super::expected_survival`]), and, for
/// [`Grid::EventTimes`], data whose
/// exact evaluation would visit more than 3e8 pairs of a person at risk and a
/// time point.
pub fn net_survival(
    data: &SurvData<'_>,
    expected: &ExpectedHazard,
    opts: &NetOptions<'_>,
) -> Result<Vec<NetSurvival>> {
    check(data, expected, opts)?;
    let surv = data.followup()?;
    let (stop, status) = (surv.stop(), surv.status());
    let start = surv.start();
    let n_strata = data.n_strata().max(1);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
    for row in data.used_rows() {
        let i = row as usize;
        if start.is_some_and(|s| s[i] < 0.0) {
            return Err(Error::invalid(format!("row {i} has a negative entry time")));
        }
        if expected.horizon(i) < stop[i] {
            return Err(Error::invalid(format!(
                "the expected hazard of row {i} ends at {} before its follow-up at {}",
                expected.horizon(i),
                stop[i]
            )));
        }
        let g = data.stratum().map_or(0, |s| s[i] as usize);
        groups[g].push(i);
    }
    let z = normal::quantile(opts.level.upper_probability());
    groups
        .into_iter()
        .enumerate()
        .map(|(stratum, rows)| {
            let raw = match (opts.method, opts.grid) {
                (Method::EdererI | Method::Hakulinen, grid) => {
                    relative(&rows, start, stop, status, expected, opts, grid)?
                }
                (method, Grid::EventTimes) => exact(&rows, start, stop, status, expected, method)?,
                (method, Grid::Width(w)) => {
                    interval(&rows, start, stop, status, expected, method, w)
                }
            };
            Ok(finish(stratum, raw, z, opts.transform))
        })
        .collect()
}

fn exact(
    rows: &[usize],
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    expected: &ExpectedHazard,
    method: Method,
) -> Result<Raw> {
    let begin = |i: usize| start.map_or(0.0, |s| s[i]);
    let mut points: Vec<f64> = rows.iter().map(|&i| stop[i]).collect();
    let mut stops = points.clone();
    stops.sort_by(f64::total_cmp);
    stops.dedup();
    points.extend(rows.iter().map(|&i| begin(i)).filter(|&s| s > 0.0));
    points.sort_by(f64::total_cmp);
    points.dedup();
    let work: f64 = rows
        .iter()
        .map(|&i| {
            let lo = points.partition_point(|&p| p <= begin(i));
            let hi = points.partition_point(|&p| p <= stop[i]);
            (hi - lo) as f64
        })
        .sum();
    if work > EXACT_LIMIT {
        return Err(Error::invalid(format!(
            "the exact estimator would visit {work:.3e} pairs of a person and a time point, \
             above the limit of {EXACT_LIMIT:.0e}: use a grid width"
        )));
    }
    let mut by_start: Vec<usize> = rows.to_vec();
    by_start.sort_by(|&a, &b| begin(a).total_cmp(&begin(b)).then(a.cmp(&b)));
    let mut next_entrant = 0;
    // The persons at risk, each with its cumulative reference hazard at the
    // previous time point and where its pieces are in `pieces`. The pieces of
    // a person are formed when it enters.
    let mut active: Vec<(usize, f64, (usize, usize))> = Vec::new();
    let mut reader = expected.reader();
    let mut pieces = Pieces::default();
    let mut raw = Raw {
        times: Vec::new(),
        lambda: Vec::new(),
        variance: Vec::new(),
        at_risk: Vec::new(),
        events: Vec::new(),
    };
    let (mut lambda, mut variance, mut previous) = (0.0_f64, 0.0_f64, 0.0_f64);
    for &b in &points {
        while next_entrant < by_start.len() && begin(by_start[next_entrant]) < b {
            let i = by_start[next_entrant];
            let at = reader.append(i, &mut pieces);
            let h = pieces.slice(at.clone()).cumulative_within(previous);
            active.push((i, h, (at.start, at.end)));
            next_entrant += 1;
        }
        if active.is_empty() {
            previous = b;
            continue;
        }
        let n = active.len() as f64;
        // Pohar Perme: W(a) = sum exp(H(a)), W(b) - W(a) = sum exp(H(a)) expm1(H(b) - H(a)),
        // and the deaths weighted by exp(H(b)). Ederer II: the sum of H(b) - H(a).
        let (mut w_a, mut w_gain, mut dh) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut deaths, mut dw, mut dw2) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (i, h_a, at) in &mut active {
            let h_b = pieces.slice(at.0..at.1).cumulative_within(b);
            let dead = stop[*i] == b && status[*i] != 0;
            match method {
                Method::PoharPerme => {
                    let e_a = h_a.exp();
                    w_a += e_a;
                    w_gain += e_a * (h_b - *h_a).exp_m1();
                    if dead {
                        let wb = h_b.exp();
                        deaths += 1.0;
                        dw += wb;
                        dw2 += wb * wb;
                    }
                }
                Method::EdererII | Method::EdererI | Method::Hakulinen => {
                    dh += h_b - *h_a;
                    if dead {
                        deaths += 1.0;
                    }
                }
            }
            *h_a = h_b;
        }
        match method {
            Method::PoharPerme => {
                let w_b = w_a + w_gain;
                lambda += dw / w_b - (w_gain / w_a).ln_1p();
                variance += dw2 / (w_b * w_b);
            }
            Method::EdererII | Method::EdererI | Method::Hakulinen => {
                lambda += deaths / n - dh / n;
                variance += deaths / (n * n);
            }
        }
        if stops.binary_search_by(|p| p.total_cmp(&b)).is_ok() {
            raw.times.push(b);
            raw.lambda.push(lambda);
            raw.variance.push(variance);
            raw.at_risk.push(n);
            raw.events.push(deaths);
        }
        active.retain(|&(i, _, _)| stop[i] > b);
        previous = b;
    }
    Ok(raw)
}

/// The relative survival of Ederer I or Hakulinen: the cumulative observed
/// hazard `Lambda_O` with its variance `sum d / n^2`, at every distinct exit
/// time or at the end of every interval, less the cumulative expected hazard
/// `-ln S_e` of the method at the same times.
fn relative(
    rows: &[usize],
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    expected: &ExpectedHazard,
    opts: &NetOptions<'_>,
    grid: Grid,
) -> Result<Raw> {
    if let Some(s) = start
        && let Some(&i) = rows.iter().find(|&&i| s[i] > 0.0)
    {
        return Err(Error::invalid(format!(
            "row {i} has delayed entry, which Ederer I and Hakulinen do not take"
        )));
    }
    let mut raw = match grid {
        Grid::EventTimes => observed(rows, stop, status),
        Grid::Width(w) => interval_observed(rows, stop, status, w),
    };
    let method = match (opts.method, opts.potential) {
        (Method::Hakulinen, Some(potential)) => {
            if potential.len() != stop.len() {
                return Err(Error::invalid(format!(
                    "{} potential follow-up times for {} rows",
                    potential.len(),
                    stop.len()
                )));
            }
            if let Some(&i) = rows
                .iter()
                .find(|&&i| !(potential[i].is_finite() && potential[i] >= 0.0))
            {
                return Err(Error::invalid(format!(
                    "the potential follow-up of row {i} is not finite and not negative"
                )));
            }
            ExpectedMethod::Hakulinen { potential }
        }
        _ => ExpectedMethod::EdererI,
    };
    let (survival, _) = match method {
        ExpectedMethod::Hakulinen { potential } => {
            let entry = |_: usize| 0.0;
            let exit = |i: usize| potential[i];
            sweep(
                rows,
                expected,
                &raw.times,
                &entry,
                &exit,
                Weighting::Survival,
            )?
        }
        _ => cohort::ederer_one(rows, expected, &raw.times)?,
    };
    for (k, s) in survival.into_iter().enumerate() {
        let s = s.ok_or_else(|| {
            Error::invalid(format!(
                "no row has potential follow-up to time {}, where the observed survival is estimated",
                raw.times[k]
            ))
        })?;
        raw.lambda[k] += s.ln();
    }
    Ok(raw)
}

/// The cumulative observed hazard of Nelson and Aalen at every distinct exit
/// time of rows that enter at 0.
fn observed(rows: &[usize], stop: &[f64], status: &[i32]) -> Raw {
    let mut order: Vec<usize> = rows.to_vec();
    order.sort_by(|&a, &b| stop[a].total_cmp(&stop[b]).then(a.cmp(&b)));
    let mut raw = Raw {
        times: Vec::new(),
        lambda: Vec::new(),
        variance: Vec::new(),
        at_risk: Vec::new(),
        events: Vec::new(),
    };
    let (mut lambda, mut variance) = (0.0_f64, 0.0_f64);
    let mut k = 0;
    while k < order.len() {
        let t = stop[order[k]];
        let n = (order.len() - k) as f64;
        let mut deaths = 0.0_f64;
        while k < order.len() && stop[order[k]] == t {
            if status[order[k]] != 0 {
                deaths += 1.0;
            }
            k += 1;
        }
        lambda += deaths / n;
        variance += deaths / (n * n);
        raw.times.push(t);
        raw.lambda.push(lambda);
        raw.variance.push(variance);
        raw.at_risk.push(n);
        raw.events.push(deaths);
    }
    raw
}

/// The cumulative observed hazard with person-time denominators in intervals
/// of `width`, the observed part of [`interval`].
fn interval_observed(rows: &[usize], stop: &[f64], status: &[i32], width: f64) -> Raw {
    let t_max = rows.iter().map(|&i| stop[i]).fold(0.0_f64, f64::max);
    let g = if t_max > 0.0 {
        (t_max / width).ceil() as usize
    } else {
        0
    };
    let (mut y, mut d) = (vec![0.0_f64; g], vec![0.0_f64; g]);
    for &i in rows {
        let mut k = 0;
        while (k as f64) * width < stop[i] && k < g {
            let hi = stop[i].min(((k + 1) as f64) * width);
            y[k] += hi - (k as f64) * width;
            if status[i] != 0 && hi == stop[i] {
                d[k] += 1.0;
            }
            k += 1;
        }
    }
    let mut raw = Raw {
        times: Vec::new(),
        lambda: Vec::new(),
        variance: Vec::new(),
        at_risk: Vec::new(),
        events: Vec::new(),
    };
    let (mut lambda, mut variance) = (0.0_f64, 0.0_f64);
    for k in 0..g {
        if y[k] <= 0.0 {
            continue;
        }
        let edge = (k as f64) * width;
        let end = ((k + 1) as f64 * width).min(t_max);
        let w = end - edge;
        lambda += d[k] / y[k] * w;
        variance += w * w * d[k] / (y[k] * y[k]);
        raw.times.push(end);
        raw.lambda.push(lambda);
        raw.variance.push(variance);
        raw.at_risk.push(y[k] / w);
        raw.events.push(d[k]);
    }
    raw
}

struct Cells {
    wy: Vec<f64>,
    wd: Vec<f64>,
    w2d: Vec<f64>,
    wh: Vec<f64>,
    y: Vec<f64>,
    d: Vec<f64>,
}

impl Cells {
    fn zero(g: usize) -> Self {
        Self {
            wy: vec![0.0; g],
            wd: vec![0.0; g],
            w2d: vec![0.0; g],
            wh: vec![0.0; g],
            y: vec![0.0; g],
            d: vec![0.0; g],
        }
    }

    fn add(&mut self, o: &Self) {
        for (a, b) in [
            (&mut self.wy, &o.wy),
            (&mut self.wd, &o.wd),
            (&mut self.w2d, &o.w2d),
            (&mut self.wh, &o.wh),
            (&mut self.y, &o.y),
            (&mut self.d, &o.d),
        ] {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
    }
}

fn interval(
    rows: &[usize],
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    expected: &ExpectedHazard,
    method: Method,
    width: f64,
) -> Raw {
    let t_max = rows.iter().map(|&i| stop[i]).fold(0.0_f64, f64::max);
    let g = if t_max > 0.0 {
        (t_max / width).ceil() as usize
    } else {
        0
    };
    let cells = par::ordered_reduce(
        rows.len(),
        BLOCK,
        WAVE,
        |range| {
            let mut c = Cells::zero(g);
            let mut reader = expected.reader();
            for &i in &rows[range] {
                let pieces = reader.row(i);
                let begin = start.map_or(0.0, |s| s[i]);
                let mut k = (begin / width).floor() as usize;
                while (k as f64) * width < stop[i] && k < g {
                    let edge = (k as f64) * width;
                    let lo = begin.max(edge);
                    let hi = stop[i].min(((k + 1) as f64) * width);
                    let y = hi - lo;
                    if y > 0.0 {
                        let h_lo = pieces.cumulative_within(lo);
                        let h_hi = pieces.cumulative_within(hi);
                        let w = match method {
                            Method::PoharPerme => {
                                pieces.cumulative_within(f64::midpoint(lo, hi)).exp()
                            }
                            Method::EdererII | Method::EdererI | Method::Hakulinen => 1.0,
                        };
                        let d = f64::from(u8::from(status[i] != 0 && hi == stop[i]));
                        c.wy[k] += w * y;
                        c.wd[k] += w * d;
                        c.w2d[k] += w * w * d;
                        c.wh[k] += w * (h_hi - h_lo);
                        c.y[k] += y;
                        c.d[k] += d;
                    }
                    k += 1;
                }
            }
            c
        },
        |t: &mut Cells, p| t.add(&p),
    );
    let mut raw = Raw {
        times: Vec::new(),
        lambda: Vec::new(),
        variance: Vec::new(),
        at_risk: Vec::new(),
        events: Vec::new(),
    };
    let (mut lambda, mut variance) = (0.0_f64, 0.0_f64);
    for k in 0..g {
        if cells.wy[k] <= 0.0 {
            continue;
        }
        let edge = (k as f64) * width;
        let end = ((k + 1) as f64 * width).min(t_max);
        let w = end - edge;
        lambda += (cells.wd[k] - cells.wh[k]) / cells.wy[k] * w;
        variance += w * w * cells.w2d[k] / (cells.wy[k] * cells.wy[k]);
        raw.times.push(end);
        raw.lambda.push(lambda);
        raw.variance.push(variance);
        raw.at_risk.push(cells.y[k] / w);
        raw.events.push(cells.d[k]);
    }
    raw
}

fn finish(stratum: usize, raw: Raw, z: f64, transform: Transform) -> NetSurvival {
    let survival: Vec<f64> = raw.lambda.iter().map(|l| (-l).exp()).collect();
    let mut lower = Vec::with_capacity(survival.len());
    let mut upper = Vec::with_capacity(survival.len());
    for ((&l, &v), &s) in raw.lambda.iter().zip(&raw.variance).zip(&survival) {
        let se = v.sqrt();
        let limits = match transform {
            Transform::Plain => Some((s - z * se * s, s + z * se * s)),
            Transform::Log => Some((s * (-z * se).exp(), s * (z * se).exp())),
            Transform::LogLog if l > 0.0 => {
                let shift = z * se / l;
                Some((
                    (-(l.ln() + shift).exp()).exp(),
                    (-(l.ln() - shift).exp()).exp(),
                ))
            }
            Transform::LogLog => None,
        };
        lower.push(limits.map(|x| x.0));
        upper.push(limits.map(|x| x.1));
    }
    NetSurvival {
        stratum,
        times: raw.times,
        cumulative_excess_hazard: raw.lambda,
        variance: raw.variance,
        survival,
        lower,
        upper,
        n_at_risk: raw.at_risk,
        events: raw.events,
    }
}
