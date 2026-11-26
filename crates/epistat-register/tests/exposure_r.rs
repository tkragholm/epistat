//! The observed and fixed dose rules, `exposed_on` and `cumulative_dose`
//! against base R and `AdhereR::compute.treatment.episodes`
//! (`tests/data/exposure_reference.R`).
//!
//! Durations, episode ends and points are compared exactly: every duration in
//! the reference is a whole number of days, and R computes the observed one
//! with the same two divisions as the crate. The prorated cumulative dose is
//! summed in another order, day by day in R and supply by supply here, so it
//! is compared to a relative tolerance: R adds at most 2 000 daily rates, an
//! error of at most `2000 * eps` of the total, below `5e-13`; the crate's sum
//! of at most eight supply shares adds less. `1e-12` covers both.

use epistat_register::exposure::{
    Assumption, Dispensings, DoseMode, DoseRule, PointTable, Stock, Supply, Windows,
    cumulative_dose, exposed_on, exposure_periods,
};
use epistat_testkit::{Tolerance, assert_close, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Dose {
    default: f64,
    min: f64,
    max: f64,
    fixed: f64,
}

#[derive(Deserialize)]
struct Episodes {
    rule: String,
    gap: f64,
    id: Vec<u32>,
    start: Vec<i64>,
    end: Vec<f64>,
}

#[derive(Deserialize)]
struct Points {
    person: Vec<u32>,
    day: Vec<i64>,
    lag: Vec<i64>,
    exposed: Vec<bool>,
}

#[derive(Deserialize)]
struct WindowRef {
    person: Vec<u32>,
    start: Vec<i64>,
    end: Vec<i64>,
    lag: Vec<i64>,
    at_dispensing: Vec<f64>,
    prorated: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    dose: Dose,
    id: Vec<u32>,
    day: Vec<i64>,
    quantity: Vec<f64>,
    episodes: Vec<Episodes>,
    points: Points,
    windows: WindowRef,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/exposure_reference.json"
    ))
}

fn offsets(id: &[u32]) -> Vec<usize> {
    let persons = *id.last().unwrap() as usize + 1;
    let mut o = vec![0_usize; persons + 1];
    for &p in id {
        o[p as usize + 1] += 1;
    }
    for p in 0..persons {
        o[p + 1] += o[p];
    }
    o
}

/// The doses of the reference, no cap on stock and a permitted gap.
fn assumption(r: &Reference, gap: f64) -> [Assumption; 1] {
    [Assumption {
        dose_default: r.dose.default,
        dose_min: r.dose.min,
        dose_max: r.dose.max,
        max_stock_days: f64::INFINITY,
        grace_days: gap,
    }]
}

#[test]
fn observed_and_fixed_rules_give_the_episodes_of_adherer() {
    let r = reference();
    let o = offsets(&r.id);
    let group = vec![Some(0); r.day.len()];
    let d = Dispensings {
        person_offsets: &o,
        group: &group,
        day: &r.day,
        quantity: &r.quantity,
    };
    assert_eq!(r.episodes.len(), 4);
    for e in &r.episodes {
        let rule = match e.rule.as_str() {
            "observed" => DoseRule::Observed,
            _ => DoseRule::Fixed { days: r.dose.fixed },
        };
        let assumptions = assumption(&r, e.gap);
        let s = Supply {
            assumptions: &assumptions,
            rule,
            stock: Stock::CarryForward,
            pauses: None,
        };
        let p = exposure_periods(&d, &s).unwrap();
        let label = format!("{} rule, gap {}", e.rule, e.gap);
        assert_eq!(p.len(), e.id.len(), "{label}: number of episodes");
        assert_eq!(p.person, e.id, "{label}");
        assert_eq!(p.start, e.start, "{label}");
        assert_close(&p.end, &e.end, Tolerance::exact());
        for i in 0..p.len() {
            assert_eq!(p.end_day(i) as f64, e.end[i], "{label}: whole end days");
        }
    }
}

#[test]
fn exposure_on_a_day_and_cumulative_doses_agree_with_base_r() {
    let r = reference();
    let o = offsets(&r.id);
    let group = vec![Some(0); r.day.len()];
    let d = Dispensings {
        person_offsets: &o,
        group: &group,
        day: &r.day,
        quantity: &r.quantity,
    };
    let assumptions = assumption(&r, 30.0);
    let s = Supply {
        assumptions: &assumptions,
        rule: DoseRule::Observed,
        stock: Stock::CarryForward,
        pauses: None,
    };
    let periods = exposure_periods(&d, &s).unwrap();

    // Each point with its own lag.
    let pts = &r.points;
    for i in 0..pts.day.len() {
        let q = PointTable {
            person: &pts.person[i..=i],
            group: &[0],
            day: &pts.day[i..=i],
        };
        assert_eq!(
            exposed_on(&periods, &q, pts.lag[i]).unwrap()[0],
            pts.exposed[i],
            "point {i}"
        );
    }
    assert!(pts.exposed.iter().filter(|&&x| x).count() > 20);

    let w = &r.windows;
    for i in 0..w.start.len() {
        let one = Windows {
            person: &w.person[i..=i],
            group: &[0],
            start: &w.start[i..=i],
            end: &w.end[i..=i],
        };
        let at = cumulative_dose(&d, &one, w.lag[i], DoseMode::AtDispensing).unwrap();
        assert_eq!(at[0].to_bits(), w.at_dispensing[i].to_bits(), "window {i}");
        let spread =
            cumulative_dose(&d, &one, w.lag[i], DoseMode::ProratedOverSupply(&s)).unwrap();
        assert_close(&spread, &w.prorated[i..=i], Tolerance::exact().rel(1e-12));
    }
    assert!(w.prorated.iter().any(|&x| x != x.round()));
}
