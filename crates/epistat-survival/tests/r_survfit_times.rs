//! `GroupCurve::at` and `GroupCurve::with_start` checked against R's
//! `summary.survfit(times = )` and `survfit0` in survival 3.8-12, on
//! `survival::lung` by sex and a simulated counting-process set with weights and
//! subjects with several rows. See `data/survfit_times.R`.
//!
//! # What is compared, and with what tolerance
//!
//! Reading a curve copies its values and adds counts, so the reader is checked
//! on R's own curve: the columns of R's `survfit` are loaded into a
//! `GroupCurve`, read, and compared with R's `summary` output. That leaves no
//! estimator error in the comparison.
//!
//! - Times, counts of rows, the values picked from the curve (survival,
//!   standard error, limits, cumulative hazard) and the weight at risk are
//!   compared exactly: they are copies of one double in both programs.
//! - A weight of events or censorings summed over an interval is a sum of at
//!   most `m` weights that are multiples of 1/8, so it is exact in both
//!   programs however it is ordered. The tolerance is zero. (R forms it as a
//!   difference of running sums, and for these weights the running sums are
//!   exact too.)
//!
//! The end-to-end check runs `estimate` on lung and on the counting-process
//! set with unit weights, and compares the read curve
//! with R's. All its counts are whole numbers, so counts are exact. The
//! survival curve is a product of at most `m` factors, each in error by at most
//! `3 eps` relative, so its tolerance is `4 m eps S`. The cumulative hazard is
//! a sum of `m` increments with relative error `3 eps` each and `m` additions,
//! so `4 m eps H`. The standard errors and limits of the estimator are checked
//! in `r_survfit.rs` and are not repeated here.

#![expect(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "copies and whole counts are compared exactly"
)]

use std::collections::HashMap;

use epistat_survival::SurvData;
use epistat_survival::survfit::{
    ConfType, CurveIndex, GroupCurve, HazardType, ReadOptions, SurvfitOptions, SurvivalType,
    estimate,
};
use epistat_testkit::{load_json, na_f64_vec};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    datasets: HashMap<String, Dataset>,
    cases: Vec<Case>,
    cr: Cr,
}

#[derive(Deserialize)]
struct Dataset {
    stop: Vec<f64>,
    status: Vec<f64>,
    g: Vec<u32>,
    start: Vec<f64>,
    id: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    stype: u8,
    ctype: u8,
    conf_type: String,
    curves: Vec<Curve>,
    starts: Vec<Read>,
    reads: Vec<Reading>,
}

#[derive(Deserialize)]
struct Cr {
    stop: Vec<f64>,
    cause: Vec<f64>,
    g: Vec<u32>,
    times: Vec<f64>,
    reads: Vec<CrRead>,
}

#[derive(Deserialize)]
struct CrRead {
    extend: bool,
    groups: Vec<CrGroup>,
}

#[derive(Deserialize)]
struct CrGroup {
    group: u32,
    time: Vec<f64>,
    pstate: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Curve {
    group: u32,
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
}

#[derive(Deserialize)]
struct Read {
    group: u32,
    time: Vec<f64>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    n_censor: Vec<f64>,
    surv: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    se: Vec<f64>,
    cumhaz: Vec<f64>,
    se_cumhaz: Vec<f64>,
    #[serde(default, deserialize_with = "na_f64_vec")]
    lower: Vec<f64>,
    #[serde(default, deserialize_with = "na_f64_vec")]
    upper: Vec<f64>,
}

#[derive(Deserialize)]
struct Reading {
    set: String,
    times: Vec<f64>,
    extend: bool,
    dosum: Option<bool>,
    groups: Vec<Read>,
}

fn opt(v: &[f64]) -> Vec<Option<f64>> {
    v.iter().map(|&x| (!x.is_nan()).then_some(x)).collect()
}

fn to_curve(c: &Curve) -> GroupCurve {
    GroupCurve {
        group: c.group,
        n: 0,
        n_id: None,
        origin: c.time.iter().copied().fold(0.0, f64::min),
        time: c.time.clone(),
        n_risk: c.n_risk.clone(),
        n_event: c.n_event.clone(),
        n_censor: c.n_censor.clone(),
        count_risk: c.count_risk.iter().map(|&v| v as u32).collect(),
        count_event: c.count_event.iter().map(|&v| v as u32).collect(),
        count_censor: c.count_censor.iter().map(|&v| v as u32).collect(),
        surv: c.surv.clone(),
        se: opt(&c.se),
        lower: opt(&c.lower),
        upper: opt(&c.upper),
        cumhaz: c.cumhaz.clone(),
        se_cumhaz: c.se_cumhaz.clone(),
    }
}

#[track_caller]
fn same(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g == w || (g.is_nan() && w.is_nan()),
            "{label}[{i}]: got {g:e}, R {w:e}"
        );
    }
}

#[track_caller]
fn same_opt(label: &str, got: &[Option<f64>], want: &[f64]) {
    let got: Vec<f64> = got.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    same(label, &got, want);
}

#[test]
fn reads_and_starting_points_match_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/survfit_times.json"
    ));
    let (mut n_reads, mut n_starts) = (0, 0);
    for case in &r.cases {
        let curves: Vec<GroupCurve> = case.curves.iter().map(to_curve).collect();
        // survfit0
        for (c, s) in curves.iter().zip(&case.starts) {
            let name = format!(
                "{} {} survfit0 group {}",
                case.name, case.conf_type, s.group
            );
            let got = c.with_start();
            assert_eq!(got.group, s.group);
            same(&format!("{name} time"), &got.time, &s.time);
            same(&format!("{name} n_risk"), &got.n_risk, &s.n_risk);
            same(&format!("{name} n_event"), &got.n_event, &s.n_event);
            same(&format!("{name} n_censor"), &got.n_censor, &s.n_censor);
            same(&format!("{name} surv"), &got.surv, &s.surv);
            same_opt(&format!("{name} se"), &got.se, &s.se);
            same(&format!("{name} cumhaz"), &got.cumhaz, &s.cumhaz);
            same(&format!("{name} se_cumhaz"), &got.se_cumhaz, &s.se_cumhaz);
            same_opt(&format!("{name} lower"), &got.lower, &s.lower);
            same_opt(&format!("{name} upper"), &got.upper, &s.upper);
            assert_eq!(got.with_start(), got, "{name}: idempotent");
            n_starts += 1;
        }
        // summary(times = )
        for rd in &case.reads {
            let opts = ReadOptions {
                extend: rd.extend,
                sum: rd.dosum,
            };
            for (c, s) in curves.iter().zip(&rd.groups) {
                let name = format!(
                    "{} {} {} extend {} dosum {:?} group {}",
                    case.name, case.conf_type, rd.set, rd.extend, rd.dosum, s.group
                );
                let got = c.at(&rd.times, &opts).unwrap();
                assert_eq!(got.group, s.group);
                same(&format!("{name} time"), &got.time, &s.time);
                same(&format!("{name} n_risk"), &got.n_risk, &s.n_risk);
                same(&format!("{name} n_event"), &got.n_event, &s.n_event);
                same(&format!("{name} n_censor"), &got.n_censor, &s.n_censor);
                same(&format!("{name} surv"), &got.surv, &s.surv);
                same_opt(&format!("{name} se"), &got.se, &s.se);
                same(&format!("{name} cumhaz"), &got.cumhaz, &s.cumhaz);
                same(&format!("{name} se_cumhaz"), &got.se_cumhaz, &s.se_cumhaz);
                same_opt(&format!("{name} lower"), &got.lower, &s.lower);
                same_opt(&format!("{name} upper"), &got.upper, &s.upper);
                n_reads += 1;
            }
        }
    }
    eprintln!("{n_starts} curves with a starting point, {n_reads} curves read");
    assert!(n_starts >= 20 && n_reads > 400);
}

/// The rules, checked on one curve by a plain scan of its rows.
#[test]
fn reads_equal_a_scan_of_the_rows() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/survfit_times.json"
    ));
    for case in &r.cases {
        for c in &case.curves {
            let g = to_curve(c);
            let c0 = g.with_start();
            let m = c0.time.len();
            let last_time = c0.time[m - 1];
            let times = [
                -2.0,
                0.0,
                c.time[0],
                c.time[0] + 0.01,
                c.time[m / 2],
                last_time,
                last_time + 1.0,
            ];
            for sum in [true, false] {
                let got = g
                    .at(
                        &times,
                        &ReadOptions {
                            extend: true,
                            sum: Some(sum),
                        },
                    )
                    .unwrap();
                for (k, &t) in times.iter().enumerate() {
                    let at = (0..m).rev().find(|&j| c0.time[j] <= t).unwrap_or(0);
                    assert_eq!(got.surv[k], c0.surv[at]);
                    assert_eq!(
                        got.count_risk[k],
                        (0..m)
                            .find(|&j| c0.time[j] >= t)
                            .map_or(0, |j| c0.count_risk[j])
                    );
                    let ev: u32 = if sum {
                        let prev = if k == 0 {
                            f64::NEG_INFINITY
                        } else {
                            times[k - 1]
                        };
                        (0..m)
                            .filter(|&j| c0.time[j] > prev && c0.time[j] <= t)
                            .map(|j| c0.count_event[j])
                            .sum()
                    } else {
                        c0.count_event[at]
                    };
                    assert_eq!(got.count_event[k], ev, "{} t {t} sum {sum}", case.name);
                }
            }
        }
    }
}

#[test]
fn estimated_lung_curve_read_matches_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/survfit_times.json"
    ));
    for case in r.cases.iter().filter(|c| c.name != "cp") {
        let d = &r.datasets[&case.name];
        let status: Vec<i32> = d.status.iter().map(|&s| s as i32).collect();
        let mut builder = SurvData::builder(&d.stop, &status).stratum(&d.g);
        if !d.start.is_empty() {
            builder = builder.start(&d.start).id(&d.id);
        }
        let data = builder.build().unwrap();
        let opts = SurvfitOptions {
            survival: if case.stype == 1 {
                SurvivalType::KaplanMeier
            } else {
                SurvivalType::ExpCumulativeHazard
            },
            hazard: if case.ctype == 1 {
                HazardType::NelsonAalen
            } else {
                HazardType::FlemingHarrington
            },
            conf_type: match case.conf_type.as_str() {
                "log" => ConfType::Log,
                "log-log" => ConfType::LogLog,
                "plain" => ConfType::Plain,
                "logit" => ConfType::Logit,
                "none" => ConfType::None,
                other => panic!("{other}"),
            },
            ..SurvfitOptions::default()
        };
        let fit = estimate(&data, &opts).unwrap();
        for rd in &case.reads {
            let ropts = ReadOptions {
                extend: rd.extend,
                sum: rd.dosum,
            };
            let got = fit.at(&rd.times, &ropts).unwrap();
            for (g, s) in got.iter().zip(&rd.groups) {
                let name = format!(
                    "{} {} {} group {}",
                    case.name, case.conf_type, rd.set, s.group
                );
                same(&format!("{name} time"), &g.time, &s.time);
                same(&format!("{name} n_risk"), &g.n_risk, &s.n_risk);
                same(&format!("{name} n_event"), &g.n_event, &s.n_event);
                same(&format!("{name} n_censor"), &g.n_censor, &s.n_censor);
                let m = fit.group(s.group).unwrap().time.len() as f64;
                for (i, (a, b)) in g.surv.iter().zip(&s.surv).enumerate() {
                    assert!(
                        (a - b).abs() <= 4.0 * m * f64::EPSILON * b,
                        "{name} surv[{i}]: {a:e} against {b:e}"
                    );
                }
                for (i, (a, b)) in g.cumhaz.iter().zip(&s.cumhaz).enumerate() {
                    assert!(
                        (a - b).abs() <= 4.0 * m * f64::EPSILON * b,
                        "{name} cumhaz[{i}]: {a:e} against {b:e}"
                    );
                }
            }
        }
    }
}

/// The Aalen-Johansen curves of `aalen_johansen::estimate`, read with the
/// same index as a survfit curve, against `summary.survfitms`.
///
/// Each value is a product of at most `m` factors or a sum of at most `m` terms
/// that are at most 1, so the bound on the absolute difference is `(4 m + 4)
/// eps`: relative `3 eps` per factor of the survival curve, and the cumulative
/// incidence adds terms no larger than the survival before them.
#[test]
fn aalen_johansen_curves_are_read_by_the_same_index() {
    use epistat_survival::aalen_johansen::{AalenJohansenOptions, estimate as aj};
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/survfit_times.json"
    ));
    let cr = &r.cr;
    let status: Vec<i32> = cr.cause.iter().map(|&c| c as i32).collect();
    let data = SurvData::builder(&cr.stop, &status)
        .n_codes(2)
        .stratum(&cr.g)
        .build()
        .unwrap();
    let curves = aj(&data, &AalenJohansenOptions::default()).unwrap();
    let mut worst = 0.0_f64;
    for rd in &cr.reads {
        for want in &rd.groups {
            let rows: Vec<usize> = (0..curves.group.len())
                .filter(|&i| curves.group[i] == want.group)
                .collect();
            let m = rows.len();
            let origin = 0.0_f64.min(curves.time[rows[0]]);
            let mut time = vec![origin];
            let mut free = vec![1.0];
            let mut cif = vec![vec![0.0]; 2];
            for &i in &rows {
                time.push(curves.time[i]);
                free.push(curves.event_free[i]);
                for (k, c) in cif.iter_mut().enumerate() {
                    c.push(curves.cif[k][i]);
                }
            }
            let opts = ReadOptions {
                extend: rd.extend,
                sum: None,
            };
            let index = CurveIndex::new(&time, &r.cr.times, &opts).unwrap();
            same("time", index.time(), &want.time);
            let bound = (4.0 * m as f64 + 4.0) * f64::EPSILON;
            let mut columns = vec![index.pick(&free).unwrap()];
            columns.extend(cif.iter().map(|c| index.pick(c).unwrap()));
            for (j, column) in columns.iter().enumerate() {
                for (i, (a, b)) in column.iter().zip(&want.pstate[j]).enumerate() {
                    worst = worst.max((a - b).abs() / bound);
                    assert!(
                        (a - b).abs() <= bound,
                        "group {} state {j} time {i}: {a:e} against {b:e}",
                        want.group
                    );
                }
            }
        }
    }
    eprintln!("Aalen-Johansen read: worst deviation {worst:.3e} of its tolerance");
}
