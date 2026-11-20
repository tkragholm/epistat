//! The curves of `survfit.coxph` in survival 3.8-12 that `r_cox_predict.rs`
//! leaves out, against [`epistat_survival::cox::predict`]: the
//! Kalbfleisch-Prentice curve (`stype = 1`), `start.time`, the counts of
//! censored rows at every exit time (`censor = TRUE`), and curves for
//! covariates that change over time (`newdata` in counting-process form with
//! `id`).
//!
//! The functions are given the coefficients and the covariance matrix that
//! `coxph` returned, so what is compared is the curve and not the fit.
//!
//! # Tolerances
//!
//! The bounds of `r_cox_predict.rs` carry over: with `T` the number of distinct
//! exit times, `p` the number of columns and `xb` the largest value of
//! `sum_a 2 |x_a beta_a|` over the rows plus the largest offset,
//!
//! - the cumulative hazard, the counts and the curves with `stype = 2` have the
//!   relative bound `b_sum = 8 eps (T + p + 4 + xb)`;
//! - a standard error has `b_se = b_sum p kappa`, relative to the value, with
//!   `kappa` the condition number of the covariance matrix;
//! - a confidence limit has the absolute bound `b_se`.
//!
//! The Kalbfleisch-Prentice curve differs in three ways.
//!
//! 1. A time with a single event has the increment `ln(1 - x) / r` with
//!    `x = w r / s0` for the event's case weight `w`, risk `r` and risk-set sum
//!    `s0`. The error of `s0` is `b_sum s0` and `1 - x` is the share of the
//!    others in `s0`, so the error of the increment is amplified by
//!    `A = s0 / (s0 - e0)`, with `e0` the risk of the events. The log of the
//!    curve has the relative bound `b_ln = 8 eps (T + p + 4 + xb + A_max)`.
//! 2. A time with tied events has the increment solving
//!    `sum w_j r_j / (1 - a^r_j) = s0`. `survival` solves it by 35 bisection
//!    steps from 0.5, so its `a` is off by up to `2^-36` and the log of the
//!    curve by `theta 2^-36 / a` per such time. `a` is found here by bisection
//!    to the spacing of doubles on the same equation.
//! 3. The variance of an increment is `W / (s0 (s0 - W))`, with `W` the case
//!    weight of the events, which has the amplification `s0 / (s0 - W)` and
//!    gives `b_se` the factor `(T + p + 4 + xb + A'_max) / (T + p + 4 + xb)`.
//!
//! A curve that reaches a time where the risk set is exactly the set of events
//! has survival 0 mathematically, and a value of rounding size from either
//! program. From that point the survival and the limits are not compared, and
//! both values must be below `max(16 eps, 2^-35) ^ theta`.
//!
//! The limits of the `log` type move with the log of the curve as the curve
//! does, `L (d ln S + z d se)`. Those of the `log-log` type are
//! `exp(-exp(u -+ v))` with `u = ln(-ln S)` and `v = z se / ln S`, so their
//! change is at most `e^-1 [(1 + |v|) d ln S + z d se] / |ln S|`. These are added
//! to `b_se`.
//!
//! The bounds are not changed after a run. The worst deviation in units of its
//! bound is printed for each output.

use std::collections::{BTreeMap, HashMap};

use epistat_survival::SurvData;
use epistat_survival::cox::predict::{
    ConfType, CurveType, FittedModel, NewData, PathData, SurvfitOptions, survfit, survfit_paths,
};
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_testkit::load_json;
use faer::{Mat, Side};
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Reference_ {
    versions: Versions,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per option of the R call"
)]
struct RFit {
    name: String,
    zero: bool,
    ties: String,
    entry: bool,
    strata: bool,
    weights: bool,
    offset: bool,
    x: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    var: Vec<Vec<f64>>,
    means: Vec<f64>,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    stratum: Vec<u32>,
    weight: Vec<f64>,
    offset_values: Vec<f64>,
    new_x: Vec<Vec<f64>>,
    new_stratum: Vec<u32>,
    new_offset: Vec<f64>,
    new_theta: Vec<f64>,
    scenarios: Vec<Scenario>,
    path: Option<RPath>,
    path_scenarios: Option<Vec<PathScenario>>,
}

#[derive(Deserialize)]
struct RPath {
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    x: Vec<Vec<f64>>,
    stratum: Vec<u32>,
    offset: Vec<f64>,
    gap_start: f64,
    gap_length: f64,
}

#[derive(Deserialize)]
struct Scenario {
    stype: u8,
    start_time: Option<f64>,
    censor: bool,
    curves: Vec<RCurve>,
}

#[derive(Deserialize)]
struct PathScenario {
    stype: u8,
    start_time: Option<f64>,
    censor: bool,
    ids: Vec<RCurve>,
}

#[derive(Deserialize)]
struct RLimits {
    lower: Vec<Option<f64>>,
    upper: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct RCurve {
    #[serde(default)]
    id: u32,
    time: Vec<f64>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    n_censor: Vec<f64>,
    surv: Vec<f64>,
    cumhaz: Vec<f64>,
    std_err: Vec<f64>,
    limits: BTreeMap<String, RLimits>,
}

/// What a program gives for one curve.
struct Got {
    time: Vec<f64>,
    stratum_code: Vec<u32>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    n_censor: Vec<f64>,
    cumhaz: Vec<f64>,
    surv: Vec<f64>,
    std_err: Vec<f64>,
    log: (Vec<Option<f64>>, Vec<Option<f64>>),
    loglog: (Vec<Option<f64>>, Vec<Option<f64>>),
}

/// What one risk set looks like at a time of a stratum.
#[derive(Clone, Copy)]
enum Time {
    /// No event.
    Censor,
    /// One event.
    Single,
    /// Tied events, the increment `a` of the curve.
    Tied(f64),
    /// The risk set is the set of events.
    Exhausted,
}

struct Table {
    kind: HashMap<(u32, u64), Time>,
    /// `s0 / (s0 - e0)` of every event time that is not exhausted.
    variance_amplification: f64,
    single_amplification: f64,
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn flatten(m: &[Vec<f64>]) -> Vec<Option<f64>> {
    m.iter().flatten().copied().map(Some).collect()
}

/// The largest deviation of `got` from `expected` in units of
/// `bound * max(|expected|, floor)`.
fn deviation(got: &[f64], expected: &[f64], bound: f64, floor: f64) -> f64 {
    assert_eq!(got.len(), expected.len(), "lengths");
    got.iter()
        .zip(expected)
        .map(|(g, e)| {
            assert!(g.is_finite() && e.is_finite(), "not finite: {g} {e}");
            (g - e).abs() / (bound * e.abs().max(floor))
        })
        .fold(0.0, f64::max)
}

/// The largest deviation in units of an allowance per element.
fn deviation_each(got: &[Option<f64>], expected: &[Option<f64>], allowed: &[f64]) -> f64 {
    assert_eq!(got.len(), expected.len(), "lengths");
    got.iter()
        .zip(expected)
        .zip(allowed)
        .map(|((g, e), a)| match (g, e) {
            (Some(g), Some(e)) => (g - e).abs() / a,
            (None, None) => 0.0,
            _ => panic!("limits differ in being defined: {g:?} {e:?}"),
        })
        .fold(0.0, f64::max)
}

/// The root `a` in (0, 1) of `sum w r / (1 - a^r) = s0` by bisection.
fn product_limit_root(w: &[f64], r: &[f64], s0: f64) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let mid = f64::midpoint(lo, hi);
        let f: f64 = w
            .iter()
            .zip(r)
            .map(|(w, r)| w * r / (1.0 - mid.powf(*r)))
            .sum();
        if f < s0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

/// The risk sets of every stratum at every exit time, by direct summation.
fn table(r: &RFit, theta: &[f64]) -> Table {
    let weight = |i: usize| if r.weights { r.weight[i] } else { 1.0 };
    let code = |i: usize| if r.strata { r.stratum[i] } else { 0 };
    let eps = f64::EPSILON;
    let mut kind = HashMap::new();
    let mut variance_amplification = 1.0f64;
    let mut single_amplification = 1.0f64;
    let mut seen = std::collections::BTreeSet::new();
    for (i, t) in r.stop.iter().enumerate() {
        seen.insert((code(i), t.to_bits()));
    }
    for (c, bits) in seen {
        let t = f64::from_bits(bits);
        let mut s0 = 0.0;
        let mut e0 = 0.0;
        let (mut ws, mut rs) = (Vec::new(), Vec::new());
        for (i, &risk) in theta.iter().enumerate() {
            if code(i) != c || r.stop[i] < t || (r.entry && r.start[i] >= t) {
                continue;
            }
            s0 += weight(i) * risk;
            if r.stop[i].to_bits() == bits && r.event[i] == 1 {
                e0 += weight(i) * risk;
                ws.push(weight(i));
                rs.push(risk);
            }
        }
        let entry = if ws.is_empty() {
            Time::Censor
        } else {
            if s0 - e0 > 8.0 * eps * s0 {
                variance_amplification = variance_amplification.max(s0 / (s0 - e0));
            }
            if s0 - e0 <= 8.0 * eps * s0 {
                Time::Exhausted
            } else if ws.len() == 1 {
                single_amplification = single_amplification.max(s0 / (s0 - e0));
                Time::Single
            } else {
                Time::Tied(product_limit_root(&ws, &rs, s0))
            }
        };
        kind.insert((c, bits), entry);
    }
    Table {
        kind,
        variance_amplification,
        single_amplification,
    }
}

#[derive(Default)]
struct Worst(BTreeMap<String, f64>);

impl Worst {
    fn put(&mut self, name: &str, value: f64) {
        let entry = self.0.entry(name.to_string()).or_insert(0.0);
        *entry = entry.max(value);
    }
}

struct Bounds {
    b_sum: f64,
    b_se: f64,
    b_se_kp: f64,
    b_ln: f64,
    n: f64,
    z: f64,
}

/// Compares one curve with R's. `theta` is the relative risk at each point.
fn compare(
    tag: &str,
    stype: u8,
    got: &Got,
    want: &RCurve,
    theta: &[f64],
    table: &Table,
    b: &Bounds,
    exact_se: bool,
    note: &mut dyn FnMut(&str, f64),
) {
    let eps = f64::EPSILON;
    let what = |s: &str| format!("{tag} {s}");
    assert_eq!(got.time, want.time, "{tag}: times");
    note(
        &what("n_risk"),
        deviation(&got.n_risk, &want.n_risk, 8.0 * eps * b.n, 0.0),
    );
    note(
        &what("n_event"),
        deviation(&got.n_event, &want.n_event, b.b_sum, 0.0),
    );
    note(
        &what("n_censor"),
        deviation(&got.n_censor, &want.n_censor, b.b_sum, 0.0),
    );
    note(
        &what("cumhaz"),
        deviation(&got.cumhaz, &want.cumhaz, b.b_sum, 0.0),
    );
    let b_se = if stype == 1 { b.b_se_kp } else { b.b_se };
    if exact_se {
        note(
            &what("std_err"),
            deviation(&got.std_err, &want.std_err, b_se, 0.0),
        );
    }
    let len = got.time.len();
    // The error of the log of the survival at each point, and where the
    // survival stops being defined.
    let mut d_ln = vec![0.0; len];
    let mut cut = len;
    if stype == 1 {
        let mut tied = 0.0;
        for k in 0..len {
            let key = (got.stratum_code[k], got.time[k].to_bits());
            match table.kind[&key] {
                Time::Tied(a) => tied += theta[k] * 2f64.powi(-36) / a,
                Time::Exhausted if cut == len => cut = k,
                _ => {}
            }
            let ln_s = want.surv[k].ln().abs();
            d_ln[k] = b.b_ln * ln_s + tied;
        }
    } else {
        for (k, d) in d_ln.iter_mut().enumerate() {
            *d = b.b_sum * want.cumhaz[k];
        }
    }
    // survival
    let mut worst_surv = 0.0f64;
    for ((g, w), d) in got.surv.iter().zip(&want.surv).zip(&d_ln).take(cut) {
        let allowed = if stype == 1 {
            w * d + 8.0 * eps * w
        } else {
            b.b_sum * w.max(1.0)
        };
        worst_surv = worst_surv.max((g - w).abs() / allowed);
    }
    note(&what("surv"), worst_surv);
    let noise = (16.0 * eps).max(2f64.powi(-35));
    for ((g, w), t) in got.surv.iter().zip(&want.surv).zip(theta).skip(cut) {
        let limit = noise.powf(*t);
        assert!(
            *g <= limit && *w <= limit,
            "{tag}: survival at an exhausted risk set is {g} and {w}"
        );
    }
    // limits
    let compared = if exact_se {
        vec![("log", &got.log), ("log-log", &got.loglog)]
    } else {
        Vec::new()
    };
    for (key, (lo, hi)) in compared {
        let rl = &want.limits[key];
        let allowed: Vec<f64> = (0..cut)
            .map(|k| {
                let s = want.surv[k];
                let se = want.std_err[k];
                let extra = if s <= 0.0 || s >= 1.0 {
                    0.0
                } else if key == "log" {
                    let upper = (s.ln() + b.z * se).exp();
                    upper * (d_ln[k] + b.z * b_se * se)
                } else {
                    let ln_s = s.ln().abs();
                    let v = b.z * se / ln_s;
                    (1.0 + v) * d_ln[k] / ln_s / std::f64::consts::E
                        + b.z * b_se * se / ln_s / std::f64::consts::E
                };
                b_se + extra
            })
            .collect();
        let dev = deviation_each(&lo[..cut], &rl.lower[..cut], &allowed).max(deviation_each(
            &hi[..cut],
            &rl.upper[..cut],
            &allowed,
        ));
        note(&what(&format!("limits {key}")), dev);
    }
}

/// The standard error of the product-limit curve of a row with centred
/// covariates `z` and relative risk `theta_new` in stratum `code`, at `times`,
/// from direct sums over the fitted rows: the delta method of the Breslow
/// hazard with the increments `W / (s0 (s0 - e0))`.
fn brute_se(
    r: &RFit,
    theta: &[f64],
    z: &[f64],
    theta_new: f64,
    code: u32,
    times: &[f64],
    start_time: Option<f64>,
) -> Vec<f64> {
    let p = z.len();
    let weight = |i: usize| if r.weights { r.weight[i] } else { 1.0 };
    let in_stratum = |i: usize| !r.strata || r.stratum[i] == code;
    let mut event_times: Vec<f64> = (0..r.stop.len())
        .filter(|&i| in_stratum(i) && r.event[i] == 1)
        .map(|i| r.stop[i])
        .filter(|&t| start_time.is_none_or(|s| t >= s))
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup();
    let mut steps = Vec::new();
    let (mut cum_v, mut d) = (0.0, vec![0.0; p]);
    for &t in &event_times {
        let (mut s0, mut e0, mut w) = (0.0, 0.0, 0.0);
        let mut s1 = vec![0.0; p];
        for (i, &theta_i) in theta.iter().enumerate() {
            if !in_stratum(i) || r.stop[i] < t || (r.entry && r.start[i] >= t) {
                continue;
            }
            let risk = weight(i) * theta_i;
            s0 += risk;
            for (a, v) in s1.iter_mut().enumerate() {
                *v += risk * (r.x[i][a] - r.means[a]);
            }
            if r.stop[i].to_bits() == t.to_bits() && r.event[i] == 1 {
                e0 += risk;
                w += weight(i);
            }
        }
        let h = w / s0;
        cum_v += if s0 - e0 > 8.0 * f64::EPSILON * s0 {
            w / (s0 * (s0 - e0))
        } else {
            w / (s0 * s0)
        };
        for a in 0..p {
            d[a] += h * z[a] - s1[a] / s0 * h;
        }
        let mut q = 0.0;
        for a in 0..p {
            for c in 0..p {
                q += d[a] * r.var[a][c] * d[c];
            }
        }
        steps.push((t, theta_new * (cum_v + q).sqrt()));
    }
    times
        .iter()
        .map(|&t| {
            steps
                .iter()
                .take_while(|(s, _)| *s <= t)
                .last()
                .map_or(0.0, |&(_, v)| v)
        })
        .collect()
}

fn from_curves(
    c: &epistat_survival::cox::predict::SurvCurves,
    i: usize,
    ll: &epistat_survival::cox::predict::SurvCurves,
) -> Got {
    let curve = &c.curves[i];
    let t = &c.strata[curve.stratum];
    let len = t.time.len();
    Got {
        time: t.time.clone(),
        stratum_code: vec![t.code.unwrap_or(0); len],
        n_risk: t.n_risk.clone(),
        n_event: t.n_event.clone(),
        n_censor: t.n_censor.clone(),
        cumhaz: curve.cumulative_hazard.clone(),
        surv: curve.survival.clone(),
        std_err: curve.std_err.clone().unwrap(),
        log: (curve.lower.clone().unwrap(), curve.upper.clone().unwrap()),
        loglog: (
            ll.curves[i].lower.clone().unwrap(),
            ll.curves[i].upper.clone().unwrap(),
        ),
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the scenarios of each fit"
)]
fn curves_match_r() {
    let reference: Reference_ = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cox_curve.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    for r in &reference.fits {
        let n = r.stop.len();
        let p = r.coefficients.len();
        let x = matrix(&r.x);
        let new_x = matrix(&r.new_x);
        let status = r.event.clone();
        let mut builder = SurvData::builder(&r.stop, &status);
        if r.entry {
            builder = builder.start(&r.start);
        }
        if r.strata {
            builder = builder.stratum(&r.stratum);
        }
        if r.weights {
            builder = builder.weight(&r.weight);
        }
        let data = builder.build().unwrap();
        let opts = CoxOptions {
            ties: if r.ties == "efron" {
                Ties::Efron
            } else {
                Ties::Breslow
            },
            offset: r.offset.then_some(r.offset_values.as_slice()),
            ..CoxOptions::default()
        };
        let mut model = fit(&data, x.as_ref(), &opts).unwrap();
        model.coefficients = r.coefficients.iter().copied().map(Some).collect();
        model.covariance = flatten(&r.var);
        model.robust_covariance = None;
        let fitted = FittedModel {
            data: &data,
            x: x.as_ref(),
            fit: &model,
            options: &opts,
        };

        // Relative risks of the fitted rows, the new rows and the path rows.
        let case_weight = |i: usize| if r.weights { r.weight[i] } else { 1.0 };
        let offset_mean = if r.offset {
            (0..n)
                .map(|i| case_weight(i) * r.offset_values[i])
                .sum::<f64>()
                / (0..n).map(case_weight).sum::<f64>()
        } else {
            0.0
        };
        let theta_of = |row: &[f64], off: f64| {
            let eta: f64 = (0..p)
                .map(|a| (row[a] - r.means[a]) * r.coefficients[a])
                .sum();
            (eta + off - offset_mean).exp()
        };
        let theta: Vec<f64> = (0..n)
            .map(|i| theta_of(&r.x[i], if r.offset { r.offset_values[i] } else { 0.0 }))
            .collect();
        for (i, row) in r.new_x.iter().enumerate() {
            let t = theta_of(row, if r.offset { r.new_offset[i] } else { 0.0 });
            assert!(
                (t - r.new_theta[i]).abs() <= 8.0 * eps * 20.0 * t,
                "{}: relative risk of new row {i}",
                r.name
            );
        }

        // The bounds.
        let mut distinct = r.stop.clone();
        distinct.sort_by(f64::total_cmp);
        distinct.dedup();
        let t_max = distinct.len() as f64;
        let pf = p as f64;
        let mut xb = 0.0f64;
        for a in 0..p {
            let big =
                r.x.iter()
                    .chain(&r.new_x)
                    .chain(r.path.iter().flat_map(|q| q.x.iter()))
                    .map(|row| row[a].abs())
                    .fold(0.0, f64::max);
            xb += 2.0 * big * r.coefficients[a].abs();
        }
        if r.offset {
            xb += r
                .offset_values
                .iter()
                .chain(&r.new_offset)
                .fold(0.0f64, |m, v| m.max(v.abs()));
        }
        let v = Mat::from_fn(p, p, |i, j| r.var[i][j]);
        let eigen = v.self_adjoint_eigenvalues(Side::Lower).unwrap();
        let kappa = eigen[p - 1] / eigen[0];
        let tab = table(r, &theta);
        let base = t_max + pf + 4.0 + xb;
        let bounds = Bounds {
            b_sum: 8.0 * eps * base,
            b_se: 8.0 * eps * base * pf * kappa,
            b_se_kp: 8.0 * eps * (base + tab.variance_amplification) * pf * kappa,
            b_ln: 8.0 * eps * (base + tab.single_amplification),
            n: n as f64,
            z: epistat_stats::normal::quantile(0.975),
        };
        let name = &r.name;
        let mut note = |what: &str, value: f64| {
            assert!(value <= 1.0, "{name}: {what} is {value} bounds");
            worst.put(what, value);
        };
        eprintln!(
            "{name}: amplification of the single-event increments {:.3}, of the variance {:.3}",
            tab.single_amplification, tab.variance_amplification
        );

        // Curves of new rows.
        let mut differs = 0.0f64;
        let levels: Vec<u32> = r.new_stratum.clone();
        let mut new = NewData::new(new_x.as_ref());
        new.stratum = r.strata.then_some(levels.as_slice());
        new.offset = r.offset.then_some(r.new_offset.as_slice());
        for sc in &r.scenarios {
            let mk = |conf_type| SurvfitOptions {
                conf_type,
                curve_type: if sc.stype == 1 {
                    CurveType::KalbfleischPrentice
                } else {
                    CurveType::ExpCumulativeHazard
                },
                start_time: sc.start_time,
                censor_times: sc.censor,
                ..SurvfitOptions::default()
            };
            let log = survfit(&fitted, &new, &mk(ConfType::Log)).unwrap();
            let ll = survfit(&fitted, &new, &mk(ConfType::LogLog)).unwrap();
            for (i, rc) in sc.curves.iter().enumerate() {
                let got = from_curves(&log, i, &ll);
                let thetas = vec![r.new_theta[i]; got.time.len()];
                let tag = format!(
                    "stype {} start {} censor {}",
                    sc.stype,
                    sc.start_time.is_some(),
                    sc.censor
                );
                let exact = sc.stype == 2 || r.zero;
                compare(
                    &tag, sc.stype, &got, rc, &thetas, &tab, &bounds, exact, &mut note,
                );
                if sc.stype == 1 {
                    let z: Vec<f64> = (0..p).map(|a| r.new_x[i][a] - r.means[a]).collect();
                    let own = brute_se(
                        r,
                        &theta,
                        &z,
                        r.new_theta[i],
                        got.stratum_code[0],
                        &got.time,
                        sc.start_time,
                    );
                    note(
                        "std_err against direct sums",
                        deviation(&got.std_err, &own, bounds.b_se_kp, 0.0),
                    );
                    if !r.zero {
                        differs = differs.max(
                            got.std_err
                                .iter()
                                .zip(&rc.std_err)
                                .map(|(g, w)| (g - w).abs() / w.abs().max(1e-300))
                                .fold(0.0, f64::max),
                        );
                    }
                }
            }
        }

        if r.zero {
            assert!(r.coefficients.iter().all(|&c| c == 0.0));
        } else {
            // survival's variance of the product-limit curve is not the one
            // with consistent units.
            assert!(
                differs > 1e-6,
                "{name}: the variance of survival and the one here do not differ ({differs})"
            );
            eprintln!("{name}: largest relative difference from survival's variance {differs:.3e}");
        }

        // Curves of covariate paths.
        let (Some(path), Some(scenarios)) = (&r.path, &r.path_scenarios) else {
            continue;
        };
        let px = matrix(&path.x);
        let path_theta: Vec<f64> = (0..path.id.len())
            .map(|i| theta_of(&path.x[i], if r.offset { path.offset[i] } else { 0.0 }))
            .collect();
        for sc in scenarios {
            let mk = |conf_type| SurvfitOptions {
                conf_type,
                curve_type: if sc.stype == 1 {
                    CurveType::KalbfleischPrentice
                } else {
                    CurveType::ExpCumulativeHazard
                },
                start_time: sc.start_time,
                censor_times: sc.censor,
                ..SurvfitOptions::default()
            };
            let data = PathData {
                x: px.as_ref(),
                id: &path.id,
                start: &path.start,
                stop: &path.stop,
                stratum: r.strata.then_some(path.stratum.as_slice()),
                offset: r.offset.then_some(path.offset.as_slice()),
            };
            let log = survfit_paths(&fitted, &data, &mk(ConfType::Log)).unwrap();
            let ll = survfit_paths(&fitted, &data, &mk(ConfType::LogLog)).unwrap();
            assert_eq!(log.curves.len(), sc.ids.len());
            for (k, rc) in sc.ids.iter().enumerate() {
                let c = &log.curves[k];
                assert_eq!(c.id, rc.id);
                // The interval of each point, for its relative risk.
                let rows: Vec<usize> = (0..path.id.len()).filter(|&i| path.id[i] == c.id).collect();
                let thetas: Vec<f64> = c
                    .time
                    .iter()
                    .map(|&t| {
                        let i = rows
                            .iter()
                            .copied()
                            .find(|&i| path.start[i] < t && t <= path.stop[i])
                            .unwrap();
                        path_theta[i]
                    })
                    .collect();
                // survfit.coxph shifts the times after a gap back by its length.
                let mut want = RCurve {
                    id: rc.id,
                    time: rc.time.clone(),
                    n_risk: rc.n_risk.clone(),
                    n_event: rc.n_event.clone(),
                    n_censor: rc.n_censor.clone(),
                    surv: rc.surv.clone(),
                    cumhaz: rc.cumhaz.clone(),
                    std_err: rc.std_err.clone(),
                    limits: BTreeMap::new(),
                };
                for (key, v) in &rc.limits {
                    want.limits.insert(
                        key.clone(),
                        RLimits {
                            lower: v.lower.clone(),
                            upper: v.upper.clone(),
                        },
                    );
                }
                if rc.id == 4 {
                    for t in &mut want.time {
                        if *t > path.gap_start {
                            *t += path.gap_length;
                        }
                    }
                }
                let got = Got {
                    time: c.time.clone(),
                    stratum_code: c
                        .stratum
                        .iter()
                        .map(|&s| log.strata[s].unwrap_or(0))
                        .collect(),
                    n_risk: c.n_risk.clone(),
                    n_event: c.n_event.clone(),
                    n_censor: c.n_censor.clone(),
                    cumhaz: c.cumulative_hazard.clone(),
                    surv: c.survival.clone(),
                    std_err: c.std_err.clone().unwrap(),
                    log: (c.lower.clone().unwrap(), c.upper.clone().unwrap()),
                    loglog: (
                        ll.curves[k].lower.clone().unwrap(),
                        ll.curves[k].upper.clone().unwrap(),
                    ),
                };
                let tag = format!(
                    "path stype {} start {} censor {}",
                    sc.stype,
                    sc.start_time.is_some(),
                    sc.censor
                );
                compare(
                    &tag,
                    sc.stype,
                    &got,
                    &want,
                    &thetas,
                    &tab,
                    &bounds,
                    sc.stype == 2 || r.zero,
                    &mut note,
                );
            }
        }
    }
    for (what, value) in &worst.0 {
        eprintln!("worst deviation, {what}: {value:.3} of its bound");
    }
}
