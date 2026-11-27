//! `crossover::build_windows` and `crossover::exposure_in_windows` checked
//! against explicit R loops over calendar days, and the analysis of the
//! windows against `survival::clogit` (survival 3.8.12, R 4.6.2).
//! `tests/data/crossover.R` makes the data and the references.
//!
//! The windows, the counts of what was dropped and the days of exposure are
//! integers and are compared exactly. The analysis fits the exact likelihood
//! and, for the cluster-robust variance, `ClogitMethod::Approximate` against
//! `clogit(method = "approximate")` with `cluster(person)`, which share the
//! likelihood with one case per set. Both sides iterate to a relative change of
//! the log likelihood of `1e-12`, so the last Newton step has information norm
//! at most `sqrt(2e-12 |ln L|)` and the coefficients agree to about
//! `2e-12 |ln L|` standard errors, below `1e-8` for the `|ln L|` of a few
//! hundred here. The covariances agree relative to `sqrt(V_jj V_kk)` to the
//! same order.

use epistat_linalg::Mat;
use epistat_survival::clogit::{ClogitMethod, ClogitOptions, clogit};
use epistat_survival::crossover::{
    ExposureIntervals, Measure, Observation, Referent, StratumUnit, build_windows,
    case_time_control_design, exposure_in_windows,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    survival: String,
}

#[derive(Deserialize)]
struct Case {
    label: String,
    h: i64,
    lag: i64,
    referent: Ref,
    event_day: Vec<i64>,
    person: Vec<u32>,
    obs_start: Vec<i64>,
    obs_end: Vec<i64>,
    group: Option<Vec<u8>>,
    exposure: Exposure,
    windows: Win,
    sets_without_referent: Vec<u32>,
    sets_case_outside: Vec<u32>,
    n_referents_outside: usize,
    n_referents_on_event_day: usize,
    fit: Fit,
}

#[derive(Deserialize)]
struct Ref {
    #[serde(rename = "type")]
    kind: String,
    unit: Option<String>,
    n: Option<usize>,
    step: Option<i64>,
}

#[derive(Deserialize)]
struct Exposure {
    person: Vec<u32>,
    start: Vec<i64>,
    end: Vec<i64>,
}

#[derive(Deserialize)]
struct Win {
    set: Vec<u32>,
    person: Vec<u32>,
    start: Vec<i64>,
    end: Vec<i64>,
    is_case: Vec<bool>,
    days: Vec<f64>,
}

#[derive(Deserialize)]
struct Fit {
    coef: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    loglik: Vec<f64>,
    approx_coef: Vec<f64>,
    approx_robust_vcov: Vec<Vec<f64>>,
}

/// Equality with a short message: the first position that differs.
#[track_caller]
fn same<T: PartialEq + std::fmt::Debug>(got: &[T], want: &[T], what: &str, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length of {what}");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{label}: {what} differs at {i}: {:?} against {:?}",
            got[i], want[i]
        );
    }
}

fn referent(r: &Ref) -> Referent {
    match r.kind.as_str() {
        "uni" => Referent::Unidirectional {
            n: r.n.unwrap(),
            step: r.step.unwrap(),
        },
        "bi" => Referent::Bidirectional {
            n: r.n.unwrap(),
            step: r.step.unwrap(),
        },
        _ => Referent::TimeStratified {
            unit: if r.unit.as_deref() == Some("year") {
                StratumUnit::Year
            } else {
                StratumUnit::Month
            },
        },
    }
}

#[test]
fn windows_and_exposure_equal_the_explicit_loops() {
    let reference: Reference = load_json("tests/data/crossover.json");
    assert!(reference.versions.r.starts_with("R version 4.6"));
    for c in &reference.cases {
        let w = build_windows(
            &c.event_day,
            &c.person,
            c.h,
            c.lag,
            referent(&c.referent),
            Some(Observation {
                start: &c.obs_start,
                end: &c.obs_end,
            }),
        )
        .unwrap();
        same(&w.set, &c.windows.set, "set", &c.label);
        same(&w.person, &c.windows.person, "person", &c.label);
        same(&w.start, &c.windows.start, "start", &c.label);
        same(&w.end, &c.windows.end, "end", &c.label);
        same(&w.is_case, &c.windows.is_case, "is_case", &c.label);
        assert_eq!(
            w.sets_without_referent, c.sets_without_referent,
            "{}",
            c.label
        );
        assert_eq!(
            w.sets_case_outside_observation, c.sets_case_outside,
            "{}",
            c.label
        );
        assert_eq!(
            w.n_referents_outside_observation, c.n_referents_outside,
            "{}",
            c.label
        );
        assert_eq!(
            w.n_referents_on_event_day, c.n_referents_on_event_day,
            "{}",
            c.label
        );
        let intervals = ExposureIntervals {
            person: &c.exposure.person,
            start: &c.exposure.start,
            end: &c.exposure.end,
        };
        let days = exposure_in_windows(&w, &intervals, Measure::Days).unwrap();
        same(&days, &c.windows.days, "days", &c.label);
        let any = exposure_in_windows(&w, &intervals, Measure::Indicator).unwrap();
        assert!(
            any.iter()
                .zip(&days)
                .all(|(&a, &d)| a.to_bits() == f64::from(u8::from(d > 0.0)).to_bits())
        );
    }
}

fn worst(got: &[Option<f64>], want_beta: &[f64], want_var: &[Vec<f64>]) -> f64 {
    let p = want_beta.len();
    let mut out = 0.0f64;
    for j in 0..p {
        out = out.max((got[j].unwrap() - want_beta[j]).abs() / want_var[j][j].sqrt());
    }
    out
}

#[test]
fn the_analysis_equals_clogit_on_the_same_windows() {
    let reference: Reference = load_json("tests/data/crossover.json");
    assert_eq!(reference.versions.survival, "3.8.12");
    for c in &reference.cases {
        let w = build_windows(
            &c.event_day,
            &c.person,
            c.h,
            c.lag,
            referent(&c.referent),
            Some(Observation {
                start: &c.obs_start,
                end: &c.obs_end,
            }),
        )
        .unwrap();
        let intervals = ExposureIntervals {
            person: &c.exposure.person,
            start: &c.exposure.start,
            end: &c.exposure.end,
        };
        let exposed = exposure_in_windows(&w, &intervals, Measure::Indicator).unwrap();
        let x: Mat<f64> = match &c.group {
            None => Mat::from_fn(exposed.len(), 1, |i, _| exposed[i]),
            Some(g) => {
                let per_window = w
                    .gather(&g.iter().map(|&v| v == 1).collect::<Vec<_>>())
                    .unwrap();
                case_time_control_design(&exposed, &per_window).unwrap()
            }
        };
        let p = x.ncols();
        let opts = ClogitOptions {
            tolerance: 1e-12,
            max_iterations: 100,
            ..ClogitOptions::default()
        };
        let exact = clogit(&w.is_case, &w.set, x.as_ref(), &opts).unwrap();
        let db = worst(&exact.coefficients, &c.fit.coef, &c.fit.vcov);
        assert!(db <= 1e-8, "{}: coefficients {db} s", c.label);
        let ll = exact.log_partial_likelihood.unwrap();
        assert!(
            (ll - c.fit.loglik[1]).abs() <= 1e-10 * ll.abs(),
            "{}",
            c.label
        );
        for j in 0..p {
            for k in 0..p {
                let got = exact.covariance[j * p + k].unwrap();
                let want = c.fit.vcov[j][k];
                let scale = (c.fit.vcov[j][j] * c.fit.vcov[k][k]).sqrt();
                assert!(
                    (got - want).abs() <= 1e-8 * scale,
                    "{}: covariance",
                    c.label
                );
            }
        }
        let cluster = ClogitOptions {
            method: ClogitMethod::Approximate,
            cluster: Some(&w.person),
            ..opts
        };
        let robust = clogit(&w.is_case, &w.set, x.as_ref(), &cluster).unwrap();
        let dr = worst(
            &robust.coefficients,
            &c.fit.approx_coef,
            &c.fit.approx_robust_vcov,
        );
        assert!(dr <= 1e-8, "{}: approximate coefficients {dr} s", c.label);
        let rcov = robust.robust_covariance.as_ref().unwrap();
        for j in 0..p {
            for k in 0..p {
                let got = rcov[j * p + k].unwrap();
                let want = c.fit.approx_robust_vcov[j][k];
                let scale =
                    (c.fit.approx_robust_vcov[j][j] * c.fit.approx_robust_vcov[k][k]).sqrt();
                assert!(
                    (got - want).abs() <= 1e-8 * scale,
                    "{}: robust covariance",
                    c.label
                );
            }
        }
    }
}
