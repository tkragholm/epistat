//! Stratified referents of `crossover` checked against CRAN and against
//! explicit R loops. `tests/data/crossover_strata.R` makes the data and the
//! references and records the versions: R 4.6.2, rSPARCS 0.1.1, season
//! 0.3.16, survival 3.8.12.
//!
//! - `rSPARCS::CXover.data` lists the referent days of single-day events in
//!   the same month on the same weekday, 7 or 14 days apart; the shifts are
//!   integers and are compared exactly.
//! - `season::casecross` fits `coxph` with the daily count as weight and a
//!   stratum per case day on its own referent days: the same weekday in fixed
//!   28-day and 35-day strata from the first day of the series, and in the
//!   month. `casecross` takes the exposure of the event day, and a hazard
//!   window here ends the day before the event, so each one-day window stands
//!   for the day after it; with that, one set per event and the exposure of the
//!   day have the likelihood of the `casecross` fit. Every day of a stratum
//!   (`matchdow = FALSE`) is not compared: the day after the event would be the
//!   window that holds the event day, which is dropped here.
//!   `coxph` stops at a relative change of the log likelihood of `1e-9`, so
//!   with `|ln L|` about 2,000 the coefficient agrees to about `4e-6`
//!   standard errors; the bound is ten times that. Its variance is formed at
//!   its last iterate, first order in the last step, and is bounded at `1e-4`
//!   relative.
//! - The explicit loops over calendar days build quarters, years with a
//!   35-day spacing and a 30-day hazard window, fixed strata with a lag and
//!   strata given per event, with observation periods that drop referents and
//!   sets; every window and count is an integer and is compared exactly.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "a few thousand events and days, far inside the ranges of the casts"
)]

use epistat_linalg::Mat;
use epistat_survival::clogit::{ClogitOptions, clogit};
use epistat_survival::crossover::{
    EventStrata, Observation, Referent, StratumUnit, Windows, build_windows,
    build_windows_in_strata,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    sparcs: Vec<Sparcs>,
    casecross: Casecross,
    loops: Vec<Loop>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "rSPARCS")]
    rsparcs: String,
    season: String,
}

#[derive(Deserialize)]
struct Sparcs {
    spacing: i64,
    event_day: Vec<i64>,
    shifts: Vec<Vec<i64>>,
}

#[derive(Deserialize)]
struct Fit {
    coef: f64,
    var: f64,
    ncases: f64,
}

#[derive(Deserialize)]
struct Casecross {
    first: i64,
    x: Vec<f64>,
    y: Vec<u32>,
    fixed_dow: Fit,
    fixed_35: Fit,
    month_dow: Fit,
}

#[derive(Deserialize)]
struct Loop {
    label: String,
    unit: String,
    h: i64,
    lag: i64,
    spacing: i64,
    origin: Option<i64>,
    length: Option<i64>,
    custom: bool,
    event_day: Vec<i64>,
    obs_start: Vec<i64>,
    obs_end: Vec<i64>,
    first: Vec<i64>,
    last: Vec<i64>,
    windows: Want,
}

#[derive(Deserialize)]
struct Want {
    set: Vec<u32>,
    start: Vec<i64>,
    end: Vec<i64>,
    is_case: Vec<bool>,
    sets_without_referent: Vec<u32>,
    sets_case_outside_observation: Vec<u32>,
    n_referents_outside_observation: usize,
}

fn reference() -> Reference {
    let r: Reference = load_json("tests/data/crossover_strata.json");
    assert_eq!(r.versions.rsparcs, "0.1.1");
    assert_eq!(r.versions.season, "0.3.16");
    r
}

#[test]
fn month_referents_match_rsparcs() {
    let r = reference();
    for s in &r.sparcs {
        let person: Vec<u32> = (0..s.event_day.len() as u32).collect();
        let referent = if s.spacing == 7 {
            Referent::TimeStratified {
                unit: StratumUnit::Month,
            }
        } else {
            Referent::Stratified {
                unit: StratumUnit::Month,
                spacing: s.spacing,
            }
        };
        let w = build_windows(&s.event_day, &person, 1, 0, referent, None).unwrap();
        for (i, want) in s.shifts.iter().enumerate() {
            let got: Vec<i64> = (0..w.len())
                .filter(|&k| w.set[k] == i as u32 && !w.is_case[k])
                .map(|k| w.start[k] + 1 - s.event_day[i])
                .collect();
            assert_eq!(&got, want, "spacing {}, event {i}", s.spacing);
        }
    }
}

/// The clogit fit of the daily exposure on the windows of the events of the
/// series, one event per count.
fn fit_series(c: &Casecross, referent: Referent) -> (f64, f64, usize) {
    let mut day = Vec::new();
    for (k, &y) in c.y.iter().enumerate() {
        for _ in 0..y {
            day.push(c.first + k as i64);
        }
    }
    let n = day.len();
    let person: Vec<u32> = (0..n as u32).collect();
    // The hazard window of an event on day t is [t - 1, t - 1], and casecross
    // takes the exposure of the event day itself, so each window stands for the
    // day after it, and the series [first, last] is observation [first - 1,
    // last - 1] of the windows.
    let start = vec![c.first - 1; n];
    let end = vec![c.first + c.x.len() as i64 - 2; n];
    let w: Windows = build_windows(
        &day,
        &person,
        1,
        0,
        referent,
        Some(Observation {
            start: &start,
            end: &end,
        }),
    )
    .unwrap();
    let x = Mat::from_fn(w.len(), 1, |i, _| c.x[(w.start[i] + 1 - c.first) as usize]);
    let f = clogit(&w.is_case, &w.set, x.as_ref(), &ClogitOptions::default()).unwrap();
    let se = f.se[0].unwrap();
    let sets = w.is_case.iter().filter(|&&b| b).count();
    (f.coefficients[0].unwrap(), se * se, sets)
}

#[test]
fn stratified_case_crossover_matches_casecross() {
    let r = reference();
    let c = &r.casecross;
    let fixed = StratumUnit::Fixed {
        origin: c.first,
        length: 28,
    };
    for (name, want, referent) in [
        (
            "fixed strata, same weekday",
            &c.fixed_dow,
            Referent::Stratified {
                unit: fixed,
                spacing: 7,
            },
        ),
        (
            "fixed 35-day strata, same weekday",
            &c.fixed_35,
            Referent::Stratified {
                unit: StratumUnit::Fixed {
                    origin: c.first,
                    length: 35,
                },
                spacing: 7,
            },
        ),
        (
            "month, same weekday",
            &c.month_dow,
            Referent::TimeStratified {
                unit: StratumUnit::Month,
            },
        ),
    ] {
        let (b, v, sets) = fit_series(c, referent);
        let db = (b - want.coef).abs() / want.var.sqrt();
        let dv = (v - want.var).abs() / want.var;
        eprintln!("{name}: coef {db:e} s, var {dv:e}");
        assert!(db <= 4e-5, "{name}: coefficient {db} s");
        assert!(dv <= 1e-4, "{name}: variance {dv}");
        // casecross counts the events of the case days that keep a referent.
        assert_eq!(sets as f64, want.ncases, "{name}");
    }
}

#[test]
fn stratified_windows_equal_the_explicit_loops() {
    let r = reference();
    for l in &r.loops {
        let n = l.event_day.len();
        let person: Vec<u32> = (0..n as u32).collect();
        let obs = Some(Observation {
            start: &l.obs_start,
            end: &l.obs_end,
        });
        let w = if l.custom {
            build_windows_in_strata(
                &l.event_day,
                &person,
                l.h,
                l.lag,
                EventStrata {
                    first: &l.first,
                    last: &l.last,
                },
                l.spacing,
                obs,
            )
        } else {
            let unit = match l.unit.as_str() {
                "quarter" => StratumUnit::Quarter,
                "year" => StratumUnit::Year,
                "month" => StratumUnit::Month,
                _ => StratumUnit::Fixed {
                    origin: l.origin.unwrap(),
                    length: l.length.unwrap(),
                },
            };
            build_windows(
                &l.event_day,
                &person,
                l.h,
                l.lag,
                Referent::Stratified {
                    unit,
                    spacing: l.spacing,
                },
                obs,
            )
        }
        .unwrap();
        let want = &l.windows;
        assert_eq!(w.set, want.set, "{}", l.label);
        assert_eq!(w.start, want.start, "{}", l.label);
        assert_eq!(w.end, want.end, "{}", l.label);
        assert_eq!(w.is_case, want.is_case, "{}", l.label);
        assert_eq!(
            w.sets_without_referent, want.sets_without_referent,
            "{}",
            l.label
        );
        assert_eq!(
            w.sets_case_outside_observation, want.sets_case_outside_observation,
            "{}",
            l.label
        );
        assert_eq!(
            w.n_referents_outside_observation, want.n_referents_outside_observation,
            "{}",
            l.label
        );
    }
}
