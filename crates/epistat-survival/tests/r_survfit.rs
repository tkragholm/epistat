//! `survfit::estimate` checked against R's `survival::survfit` 3.8-12: the
//! Kaplan-Meier and Nelson-Aalen curves in every stype and ctype combination,
//! Greenwood and infinitesimal jackknife standard errors, every confidence
//! limit transformation, `quantile.survfit`, and the restricted mean of
//! `survmean`.
//!
//! The data are `survival::lung` and a simulated set with ties, delayed entry,
//! three groups, whole-number, binary-exact and decimal case weights, a cluster
//! column, subjects with several rows, and times that tie only up to rounding
//! error. See `data/survfit.R`.
//!
//! # Tolerances
//!
//! Each is a bound on how far two implementations of the same formula can
//! differ in double precision, computed from R's own counts, and none was set
//! by trying values.
//!
//! - Times and counts are compared exactly. So are quantile times, which are
//!   data values or midpoints of two of them.
//! - A weighted count is a sum of at most `n` positive weights, each addition
//!   rounding by at most `eps / 2` times a running value no larger than the
//!   group's total weight `W`, so two orders of summation differ by at most
//!   `dw = n eps W`. Whole-number weights and multiples of 1/8 sum exactly, and
//!   `dw` is zero for them.
//! - An increment is a function `g(Y, d)` of the weight at risk and the weight
//!   of events. With `p = dg/dd` and `q = -dg/dY`, an error `dw` in both moves
//!   it by at most `dw (p + q)`, and the arithmetic adds `3 eps` of the
//!   increment. The cumulative hazard is a sum of `m` increments, so its bound
//!   is the sum of the increments' bounds and `m eps H` for the additions.
//!   The Kaplan-Meier curve is a product, and the same bound on the logarithm of
//!   each factor, with `p = 1 / (Y - d)` and `q = d / (Y (Y - d))`, adds up to
//!   its relative error. With stype 2 the bound on `exp(-H)` is `S` times that
//!   on `H`.
//! - A variance that is a sum of positive terms has the relative error of its
//!   worst term plus `m eps`, and a standard error half of it plus `2 eps`.
//! - The jackknife variance is `P - 2 Q C + Q^2 R` in the implementation and a
//!   recursion in R. Each rounds relative to the size of what it adds, which
//!   is bounded by the sum over clusters of the squared absolute sum of the
//!   cluster's terms. The test forms that sum for every time, `M`, directly
//!   from the rows, and takes `8 (n + m + 8) eps M` as the bound on the
//!   variance, which allows each implementation `n + m + 8` operations on each
//!   of four terms. The standard error follows from
//!   `|sqrt(a) - sqrt(b)| = |a - b| / (sqrt(a) + sqrt(b))`.
//! - A confidence limit is a smooth function of the survival probability, its
//!   standard error and the normal quantile. Its bound is the largest change of
//!   the limit over the corners of the bounds on those three, plus `16 eps`.
//! - The restricted mean is a sum of rectangles whose heights have the bounds
//!   of the curve, and its variance a sum of squared areas times Greenwood
//!   increments. Both bounds follow from the bounds on the curve.

#![expect(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::manual_midpoint,
    clippy::redundant_guards,
    clippy::similar_names,
    clippy::unused_self,
    reason = "the comparisons are of values that must be equal or are whole counts and flags, \
              and the casts are of whole numbers a few hundred at most"
)]

use std::collections::HashMap;

use epistat_survival::SurvData;
use epistat_survival::survfit::{
    ConfLower, ConfType, HazardType, SurvfitOptions, SurvivalType, Variance, estimate,
};
use epistat_testkit::{load_json, na_f64_vec};
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: HashMap<String, Dataset>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    survival: String,
}

#[derive(Deserialize)]
struct Dataset {
    stop: Vec<f64>,
    status: Vec<f64>,
    start: Vec<f64>,
    group: Vec<u32>,
    weight: Vec<f64>,
    id: Vec<u32>,
    cluster: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    taus: Vec<f64>,
    robust_arg: Option<bool>,
    fits: Vec<Fit>,
}

#[derive(Deserialize)]
struct Fit {
    stype: u8,
    ctype: u8,
    conf_type: String,
    conf_lower: String,
    conf_int: f64,
    robust: bool,
    bug: bool,
    groups: Vec<RGroup>,
}

#[derive(Deserialize)]
struct RGroup {
    group: u32,
    n: usize,
    n_id: Option<usize>,
    time: Vec<f64>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    n_censor: Vec<f64>,
    count_risk: Vec<f64>,
    count_event: Vec<f64>,
    count_censor: Vec<f64>,
    surv: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    se: Vec<f64>,
    cumhaz: Vec<f64>,
    se_cumhaz: Vec<f64>,
    #[serde(default, deserialize_with = "na_f64_vec")]
    lower: Vec<f64>,
    #[serde(default, deserialize_with = "na_f64_vec")]
    upper: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    q: Vec<f64>,
    #[serde(default, deserialize_with = "na_f64_vec")]
    q_lower: Vec<f64>,
    #[serde(default, deserialize_with = "na_f64_vec")]
    q_upper: Vec<f64>,
    rmean: Vec<f64>,
    rmean_se: Vec<f64>,
}

/// The largest deviation seen per output, in units of its tolerance.
#[derive(Default)]
struct Worst(HashMap<&'static str, (f64, String)>);

impl Worst {
    #[track_caller]
    fn see(&mut self, output: &'static str, label: &str, got: f64, want: f64, tolerance: f64) {
        let deviation = (got - want).abs();
        let ratio = if tolerance > 0.0 {
            deviation / tolerance
        } else if deviation == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        assert!(
            ratio <= 1.0 || (got.is_nan() && want.is_nan()),
            "{output} at {label}: got {got:e}, R {want:e}, deviation {deviation:e} against tolerance {tolerance:e}"
        );
        let entry = self.0.entry(output).or_insert((0.0, String::new()));
        if ratio >= entry.0 {
            *entry = (ratio, label.to_string());
        }
    }

    fn undefined(&mut self, output: &'static str, label: &str, got: Option<f64>, want: f64) {
        assert!(
            got.is_none() == want.is_nan(),
            "{output} at {label}: got {got:?}, R {want}"
        );
    }
}

/// The rows of one group, with positive weight.
struct GroupRows {
    start: Option<Vec<f64>>,
    stop: Vec<f64>,
    event: Vec<bool>,
    weight: Vec<f64>,
    cluster: Vec<usize>,
}

fn group_rows(d: &Dataset, group: u32) -> GroupRows {
    let n = d.stop.len();
    let rows: Vec<usize> = (0..n)
        .filter(|&i| {
            (d.group.is_empty() || d.group[i] == group)
                && (d.weight.is_empty() || d.weight[i] > 0.0)
        })
        .collect();
    let key = |i: usize| {
        if !d.cluster.is_empty() {
            d.cluster[i] as usize
        } else if !d.id.is_empty() {
            d.id[i] as usize
        } else {
            i
        }
    };
    GroupRows {
        start: (!d.start.is_empty()).then(|| rows.iter().map(|&i| d.start[i]).collect()),
        stop: rows.iter().map(|&i| d.stop[i]).collect(),
        event: rows.iter().map(|&i| d.status[i] == 1.0).collect(),
        weight: rows
            .iter()
            .map(|&i| {
                if d.weight.is_empty() {
                    1.0
                } else {
                    d.weight[i]
                }
            })
            .collect(),
        cluster: rows.iter().map(|&i| key(i)).collect(),
    }
}

/// `(p, q)` of the hazard increment: `dg/dd` and `-dg/dY`.
fn hazard_pq(ctype: u8, y: f64, d1: f64, d0: f64) -> (f64, f64) {
    if ctype == 1 {
        return (1.0 / y, d1 / (y * y));
    }
    let a = d1 / d0;
    let (mut sum_yb2, mut sum_b2) = (0.0, 0.0);
    for j in 0..d0 as u32 {
        let b = 1.0 / (y - f64::from(j) * a);
        sum_yb2 += y * b * b;
        sum_b2 += b * b;
    }
    (sum_yb2 / d0, a * sum_b2)
}

fn km_pq(y: f64, d1: f64) -> (f64, f64) {
    (1.0 / (y - d1), d1 / (y * (y - d1)))
}

/// What R's counts allow each output to differ by.
struct Bounds {
    s: Vec<f64>,
    h: Vec<f64>,
    se_s: Vec<f64>,
    se_h: Vec<f64>,
    /// The standard error the limits are formed from, with its tolerance, and
    /// whether it is on the scale of S.
    raw: Vec<f64>,
    raw_tol: Vec<f64>,
    raw_surv_scale: bool,
}

/// The sum over clusters of the squared absolute sums of the cluster's terms,
/// at every time, for increments with the given derivatives at the event times.
fn envelope(
    rows: &GroupRows,
    times: &[f64],
    event_times: &[f64],
    p: &[f64],
    q: &[f64],
) -> Vec<f64> {
    let cumulative: Vec<f64> = q
        .iter()
        .scan(0.0, |acc, &v| {
            *acc += v;
            Some(*acc)
        })
        .collect();
    let big_q = |t: f64| {
        let k = event_times.partition_point(|&e| e <= t);
        if k == 0 { 0.0 } else { cumulative[k - 1] }
    };
    let mut out = Vec::with_capacity(times.len());
    for &t in times {
        let mut by_cluster: HashMap<usize, f64> = HashMap::new();
        for r in 0..rows.stop.len() {
            let start = rows.start.as_ref().map_or(f64::NEG_INFINITY, |s| s[r]);
            if start >= t {
                continue;
            }
            let u = rows.stop[r];
            let entry = if rows.start.is_some() {
                big_q(start)
            } else {
                0.0
            };
            let mut term = entry + big_q(u.min(t));
            if rows.event[r] && u <= t {
                let j = event_times.partition_point(|&e| e < u);
                if j < p.len() {
                    term += p[j];
                }
            }
            *by_cluster.entry(rows.cluster[r]).or_insert(0.0) += rows.weight[r] * term;
        }
        out.push(by_cluster.values().map(|b| b * b).sum());
    }
    out
}

fn bounds(g: &RGroup, fit: &Fit, dw: f64, rows: &GroupRows) -> Bounds {
    let m_total = g.time.len();
    let n = rows.stop.len() as f64;
    let (mut h_abs, mut km_rel, mut steps) = (0.0, 0.0, 0.0);
    let (mut rel_t1, mut rel_t2): (f64, f64) = (0.0, 0.0);
    let mut zero = false;
    let mut b = Bounds {
        s: Vec::new(),
        h: Vec::new(),
        se_s: Vec::new(),
        se_h: Vec::new(),
        raw: Vec::new(),
        raw_tol: Vec::new(),
        raw_surv_scale: false,
    };
    // Derivatives at the event times, for the jackknife.
    let (mut event_times, mut p_h, mut q_h, mut p_k, mut q_k) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..m_total {
        if g.count_event[i] > 0.0 {
            let (p, q) = hazard_pq(fit.ctype, g.n_risk[i], g.n_event[i], g.count_event[i]);
            event_times.push(g.time[i]);
            p_h.push(p);
            q_h.push(q);
            if g.n_risk[i] - g.n_event[i] > 0.0 && !zero {
                let (pk, qk) = km_pq(g.n_risk[i], g.n_event[i]);
                p_k.push(pk);
                q_k.push(qk);
            }
            if g.n_risk[i] - g.n_event[i] <= 0.0 {
                zero = true;
            }
        }
    }
    let jackknife = fit.robust;
    let times = &g.time;
    let m_h = if jackknife {
        envelope(rows, times, &event_times, &p_h, &q_h)
    } else {
        Vec::new()
    };
    let m_k = if jackknife && fit.stype == 1 {
        envelope(rows, times, &event_times[..p_k.len()], &p_k, &q_k)
    } else {
        Vec::new()
    };
    zero = false;
    for i in 0..m_total {
        let (y, d1, d0) = (g.n_risk[i], g.n_event[i], g.count_event[i]);
        if d0 > 0.0 {
            steps += 1.0;
            let (p, q) = hazard_pq(fit.ctype, y, d1, d0);
            let step = g.cumhaz[i] - if i == 0 { 0.0 } else { g.cumhaz[i - 1] };
            h_abs += dw * (p + q) + 3.0 * EPS * step;
            if y - d1 > 0.0 && !zero {
                let (pk, qk) = km_pq(y, d1);
                km_rel += dw * (pk + qk) + 3.0 * EPS;
                rel_t1 = rel_t1.max(dw * (1.0 / d1 + 2.0 / (y - d1) + 1.0 / y) + 5.0 * EPS);
            } else {
                zero = true;
            }
            let rel_h = if fit.ctype == 1 {
                dw * (1.0 / d1 + 2.0 / y) + 4.0 * EPS
            } else {
                let a = d1 / d0;
                dw * (1.0 / d1 + 4.0 / (y - d1 + a)) + 6.0 * EPS
            };
            rel_t2 = rel_t2.max(rel_h);
        }
        let s_ref = g.surv[i];
        let h_tol = h_abs + steps * EPS * g.cumhaz[i];
        let s_tol = if fit.stype == 1 {
            if zero {
                0.0
            } else {
                s_ref * (km_rel + steps * EPS)
            }
        } else {
            s_ref * (h_tol + 2.0 * EPS)
        };
        b.h.push(h_tol);
        b.s.push(s_tol);
        // Standard errors.
        let (se_h_tol, se_log, se_log_tol);
        if jackknife {
            let c = 8.0 * (n + m_total as f64 + 8.0) * EPS;
            let v_tol = c * m_h[i];
            let se_h = g.se_cumhaz[i];
            se_h_tol = sqrt_tolerance(se_h, v_tol);
            if fit.stype == 1 {
                let v_tol_k = c * m_k[i];
                let phi = if s_ref > 0.0 { g.se[i] / s_ref } else { 0.0 };
                se_log = phi;
                se_log_tol = if zero {
                    0.0
                } else {
                    sqrt_tolerance(phi, v_tol_k)
                };
            } else {
                se_log = se_h;
                se_log_tol = se_h_tol;
            }
        } else {
            let v2_rel = rel_t2 + steps * EPS;
            let se_h = g.se_cumhaz[i];
            se_h_tol = se_h * (0.5 * v2_rel + 2.0 * EPS);
            if fit.stype == 1 {
                se_log = if s_ref > 0.0 {
                    g.se[i] / s_ref
                } else {
                    f64::NAN
                };
                se_log_tol = se_log * (0.5 * (rel_t1 + steps * EPS) + 2.0 * EPS);
            } else {
                se_log = se_h;
                se_log_tol = se_h_tol;
            }
        }
        b.se_h.push(se_h_tol);
        let se_s_tol = if se_log.is_nan() {
            0.0
        } else {
            s_tol * se_log + s_ref * se_log_tol + s_tol * se_log_tol
        };
        b.se_s.push(se_s_tol);
        // The standard error the limits use.
        if jackknife && fit.stype == 1 {
            b.raw.push(g.se[i]);
            b.raw_tol.push(se_s_tol);
            b.raw_surv_scale = true;
        } else {
            b.raw.push(se_log);
            b.raw_tol.push(se_log_tol);
        }
    }
    b
}

/// The tolerance on a standard error whose variance has the tolerance
/// `v_tol`: `v_tol / se`, never more than `sqrt(v_tol)`.
fn sqrt_tolerance(se: f64, v_tol: f64) -> f64 {
    if se > 0.0 {
        (v_tol / se).min(v_tol.sqrt())
    } else {
        v_tol.sqrt()
    }
}

/// `survfit_confint`, for the limits' sensitivity.
fn limits_ref(
    kind: &str,
    z: f64,
    p: f64,
    se: f64,
    surv_scale: bool,
    lower_se: Option<f64>,
) -> (f64, f64) {
    let stretch = match lower_se {
        None => 1.0,
        Some(l) if l == 0.0 => 1.0,
        Some(l) => l / se,
    };
    let se = if surv_scale {
        if se == 0.0 { 0.0 } else { se / p }
    } else {
        se
    };
    let nan_to = |x: f64| if x.is_nan() { f64::NAN } else { x };
    let positive = if p == 0.0 { f64::NAN } else { p };
    let interior = if p == 0.0 || p == 1.0 { f64::NAN } else { p };
    match kind {
        "plain" => {
            let half = se * p * z;
            ((p - half * stretch).max(0.0), (p + half).min(1.0))
        }
        "log" => {
            let half = z * se;
            let lo = if se == 0.0 {
                p
            } else {
                (positive.ln() - half * stretch).exp()
            };
            let hi = if se == 0.0 {
                p
            } else {
                (positive.ln() + half).exp()
            };
            (lo, nan_to(hi.min(1.0)))
        }
        "log-log" => {
            let half = z * se / interior.ln();
            let base = (-interior.ln()).ln();
            let lo = if se == 0.0 {
                p
            } else {
                (-(base - half * stretch).exp()).exp()
            };
            let hi = if se == 0.0 {
                p
            } else {
                (-(base + half).exp()).exp()
            };
            (lo, hi)
        }
        "logit" => {
            let half = z * se * (1.0 + positive / (1.0 - positive));
            let centre = (p / (1.0 - p)).ln();
            let lo = if se == 0.0 {
                p
            } else {
                1.0 - 1.0 / (1.0 + (centre - half * stretch).exp())
            };
            let hi = if se == 0.0 {
                p
            } else {
                1.0 - 1.0 / (1.0 + (centre + half).exp())
            };
            (lo, hi)
        }
        "arcsin" => {
            let half = 0.5 * z * se * (positive / (1.0 - positive)).sqrt();
            let centre = positive.sqrt().asin();
            let lo = (centre - half * stretch).max(0.0).sin();
            let hi = (centre + half).min(std::f64::consts::FRAC_PI_2).sin();
            (lo * lo, hi * hi)
        }
        other => panic!("conf type {other}"),
    }
}

fn parse_options(case: &Case, fit: &Fit) -> SurvfitOptions {
    SurvfitOptions {
        survival: if fit.stype == 1 {
            SurvivalType::KaplanMeier
        } else {
            SurvivalType::ExpCumulativeHazard
        },
        hazard: if fit.ctype == 1 {
            HazardType::NelsonAalen
        } else {
            HazardType::FlemingHarrington
        },
        conf_type: match fit.conf_type.as_str() {
            "log" => ConfType::Log,
            "log-log" => ConfType::LogLog,
            "plain" => ConfType::Plain,
            "logit" => ConfType::Logit,
            "arcsin" => ConfType::Arcsin,
            "none" => ConfType::None,
            other => panic!("conf type {other}"),
        },
        conf_level: fit.conf_int,
        conf_lower: match fit.conf_lower.as_str() {
            "usual" => ConfLower::Usual,
            "peto" => ConfLower::Peto,
            "modified" => ConfLower::Modified,
            other => panic!("conf.lower {other}"),
        },
        robust: case.robust_arg,
        ..SurvfitOptions::default()
    }
}

#[test]
fn survfit_matches_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/survfit.json"
    ));
    eprintln!(
        "{}, survival {}",
        reference.versions.r, reference.versions.survival
    );
    let mut worst = Worst::default();
    let mut compared = 0usize;
    for case in &reference.cases {
        let d = &reference.datasets[&case.name];
        let status: Vec<i32> = d.status.iter().map(|&s| s as i32).collect();
        let mut builder = SurvData::builder(&d.stop, &status);
        if !d.start.is_empty() {
            builder = builder.start(&d.start);
        }
        if !d.weight.is_empty() {
            builder = builder.weight(&d.weight);
        }
        if !d.group.is_empty() {
            builder = builder.stratum(&d.group);
        }
        if !d.id.is_empty() {
            builder = builder.id(&d.id);
        }
        if !d.cluster.is_empty() {
            builder = builder.cluster(&d.cluster);
        }
        let data = builder.build().unwrap();
        // Weights that are not multiples of 1/8 do not sum exactly.
        let inexact = d.weight.iter().any(|w| (w * 8.0).fract() != 0.0);
        for (fit_index, fit) in case.fits.iter().enumerate() {
            let opts = parse_options(case, fit);
            // survfit mislabels the scale of its standard error exactly when the
            // jackknife is used with stype and ctype that differ.
            assert_eq!(
                fit.bug,
                fit.robust && fit.conf_type != "none" && fit.stype != fit.ctype,
                "{}: R's scale flag",
                case.name
            );
            let got = estimate(&data, &opts).unwrap();
            let name = format!(
                "{} fit {fit_index} (stype {}, ctype {}, {}, {})",
                case.name, fit.stype, fit.ctype, fit.conf_type, fit.conf_lower
            );
            assert_eq!(
                got.variance,
                if fit.robust {
                    Variance::InfinitesimalJackknife
                } else {
                    Variance::Greenwood
                },
                "{name}: variance"
            );
            assert_eq!(got.groups.len(), fit.groups.len(), "{name}: groups");
            let z = epistat_stats::normal::quantile(1.0 - (1.0 - fit.conf_int) / 2.0);
            for (g, r) in got.groups.iter().zip(&fit.groups) {
                compared += 1;
                assert_eq!(g.group, r.group, "{name}: group code");
                assert_eq!(g.n, r.n, "{name}: n");
                assert_eq!(g.n_id, r.n_id, "{name}: n_id");
                assert_eq!(g.time, r.time, "{name} group {}: times", r.group);
                let rows = group_rows(d, r.group);
                let w_total: f64 = rows.weight.iter().sum();
                let dw = if inexact {
                    rows.stop.len() as f64 * EPS * w_total
                } else {
                    0.0
                };
                for i in 0..r.time.len() {
                    let label = format!("{name} group {} time {}", r.group, r.time[i]);
                    assert_eq!(
                        f64::from(g.count_risk[i]),
                        r.count_risk[i],
                        "{label}: count_risk"
                    );
                    assert_eq!(
                        f64::from(g.count_event[i]),
                        r.count_event[i],
                        "{label}: count_event"
                    );
                    assert_eq!(
                        f64::from(g.count_censor[i]),
                        r.count_censor[i],
                        "{label}: count_censor"
                    );
                    worst.see("n_risk", &label, g.n_risk[i], r.n_risk[i], dw);
                    worst.see("n_event", &label, g.n_event[i], r.n_event[i], dw);
                    worst.see("n_censor", &label, g.n_censor[i], r.n_censor[i], dw);
                }
                let b = bounds(r, fit, dw, &rows);
                for i in 0..r.time.len() {
                    let label = format!("{name} group {} time {}", r.group, r.time[i]);
                    worst.see("surv", &label, g.surv[i], r.surv[i], b.s[i]);
                    worst.see("cumhaz", &label, g.cumhaz[i], r.cumhaz[i], b.h[i]);
                    worst.see(
                        "se_cumhaz",
                        &label,
                        g.se_cumhaz[i],
                        r.se_cumhaz[i],
                        b.se_h[i],
                    );
                    worst.undefined("se", &label, g.se[i], r.se[i]);
                    if let Some(se) = g.se[i] {
                        worst.see("se", &label, se, r.se[i], b.se_s[i]);
                    }
                }
                // Limits.
                if fit.conf_type == "none" {
                    assert!(
                        g.lower.is_empty() && g.upper.is_empty(),
                        "{name}: limits without a type"
                    );
                } else {
                    let mut last_event_risk = r.n_risk[0];
                    for i in 0..r.time.len() {
                        if r.n_event[i] > 0.0 {
                            last_event_risk = r.n_risk[i];
                        }
                        let label = format!("{name} group {} time {}", r.group, r.time[i]);
                        worst.undefined("lower", &label, g.lower[i], r.lower[i]);
                        worst.undefined("upper", &label, g.upper[i], r.upper[i]);
                        let lower_se = |p: f64, se: f64| match fit.conf_lower.as_str() {
                            "peto" => Some(((1.0 - p) / r.n_risk[i]).sqrt()),
                            "modified" => Some(se * (last_event_risk / r.n_risk[i]).sqrt()),
                            _ => None,
                        };
                        let (lo0, hi0) = limits_ref(
                            &fit.conf_type,
                            z,
                            r.surv[i],
                            b.raw[i],
                            b.raw_surv_scale,
                            lower_se(r.surv[i], b.raw[i]),
                        );
                        let (mut lo_tol, mut hi_tol) = (0.0_f64, 0.0_f64);
                        for ds in [-1.0, 1.0] {
                            for de in [-1.0, 1.0] {
                                for dz in [-1.0, 1.0] {
                                    let p = (r.surv[i] + ds * b.s[i]).clamp(0.0, 1.0);
                                    let se = b.raw[i] + de * b.raw_tol[i];
                                    let (lo, hi) = limits_ref(
                                        &fit.conf_type,
                                        z * (1.0 + dz * 8.0 * EPS),
                                        p,
                                        se,
                                        b.raw_surv_scale,
                                        lower_se(p, se),
                                    );
                                    if lo.is_finite() && lo0.is_finite() {
                                        lo_tol = lo_tol.max((lo - lo0).abs());
                                    }
                                    if hi.is_finite() && hi0.is_finite() {
                                        hi_tol = hi_tol.max((hi - hi0).abs());
                                    }
                                }
                            }
                        }
                        if let (Some(lo), false) = (g.lower[i], r.lower[i].is_nan()) {
                            worst.see(
                                "lower",
                                &label,
                                lo,
                                r.lower[i],
                                lo_tol + 16.0 * EPS * lo.abs().max(1.0),
                            );
                        }
                        if let (Some(hi), false) = (g.upper[i], r.upper[i].is_nan()) {
                            worst.see(
                                "upper",
                                &label,
                                hi,
                                r.upper[i],
                                hi_tol + 16.0 * EPS * hi.abs().max(1.0),
                            );
                        }
                    }
                }
                // Quantiles.
                let probs = [0.25, 0.5, 0.75];
                let q = g.quantiles(&probs).unwrap();
                for (k, quantile) in q.iter().enumerate() {
                    let label = format!("{name} group {} quantile {}", r.group, probs[k]);
                    let same = |got: Option<f64>, want: f64| match got {
                        Some(v) => v == want,
                        None => want.is_nan(),
                    };
                    assert!(
                        same(quantile.time, r.q[k]),
                        "{label}: {:?} against {}",
                        quantile.time,
                        r.q[k]
                    );
                    if fit.conf_type == "none" {
                        assert!(
                            quantile.lower.is_none() && quantile.upper.is_none(),
                            "{label}"
                        );
                    } else {
                        assert!(
                            same(quantile.lower, r.q_lower[k]),
                            "{label}: lower {:?} against {}",
                            quantile.lower,
                            r.q_lower[k]
                        );
                        assert!(
                            same(quantile.upper, r.q_upper[k]),
                            "{label}: upper {:?} against {}",
                            quantile.upper,
                            r.q_upper[k]
                        );
                    }
                }
                // Restricted means.
                for (k, &tau) in case.taus.iter().enumerate() {
                    let label = format!("{name} group {} rmean {tau}", r.group);
                    let got = g.rmean(tau).unwrap();
                    let (mean_tol, se_tol) = rmean_bounds(r, &b, tau, dw);
                    worst.see("rmean", &label, got.mean, r.rmean[k], mean_tol);
                    worst.see("rmean se", &label, got.se, r.rmean_se[k], se_tol);
                }
            }
        }
    }
    eprintln!("{compared} curves compared");
    let mut rows: Vec<_> = worst.0.iter().collect();
    rows.sort_by(|a, b| a.0.cmp(b.0));
    for (output, (ratio, label)) in rows {
        eprintln!("worst {output}: {ratio:.3} of tolerance at {label}");
    }
}

/// The bounds on the restricted mean and its standard error.
fn rmean_bounds(r: &RGroup, b: &Bounds, tau: f64, dw: f64) -> (f64, f64) {
    let kept = r.time.partition_point(|&t| t <= tau);
    let origin = 0.0;
    let mut widths = Vec::new();
    let mut heights = Vec::new();
    let mut height_tol = Vec::new();
    let (mut previous, mut s_prev, mut s_tol_prev) = (origin, 1.0, 0.0);
    for i in 0..kept {
        widths.push(r.time[i] - previous);
        heights.push(s_prev);
        height_tol.push(s_tol_prev);
        previous = r.time[i];
        s_prev = r.surv[i];
        s_tol_prev = b.s[i];
    }
    widths.push(tau - previous);
    heights.push(s_prev);
    height_tol.push(s_tol_prev);
    let area: f64 = widths.iter().zip(&heights).map(|(w, h)| w * h).sum();
    let count = widths.len() as f64;
    let mean_tol: f64 = widths
        .iter()
        .zip(&height_tol)
        .map(|(w, t)| w * t)
        .sum::<f64>()
        + 2.0 * (count + 2.0) * EPS * area;
    // Variance: sum over times of (area after)^2 * increment.
    let mut variance = 0.0;
    let mut variance_tol = 0.0;
    let mut after = 0.0;
    let mut after_tol = 0.0;
    for i in (0..kept).rev() {
        after += widths[i + 1] * heights[i + 1];
        after_tol += widths[i + 1] * height_tol[i + 1];
        let (y, d) = (r.n_risk[i], r.n_event[i]);
        let (inc, rel) = if y - d == 0.0 || d == 0.0 {
            (0.0, 0.0)
        } else {
            (
                d / (y * (y - d)),
                dw * (1.0 / d + 2.0 / (y - d) + 1.0 / y) + 5.0 * EPS,
            )
        };
        let a_err = after_tol + count * EPS * after;
        variance += after * after * inc;
        variance_tol += 2.0 * after * a_err * inc + after * after * inc * rel;
    }
    variance_tol += 2.0 * (count + 2.0) * EPS * variance;
    (mean_tol, sqrt_tolerance(variance.sqrt(), variance_tol))
}
