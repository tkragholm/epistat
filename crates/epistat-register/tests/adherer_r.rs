//! Exposure episodes checked against `AdhereR::compute.treatment.episodes`
//! 0.8.3 (`tests/data/adherer_reference.R`).
//!
//! AdhereR starts a new episode when the gap after the supply, with surplus
//! carried over, is longer than the permitted gap. It takes whole-day
//! durations, and the reference gives each dispensing the duration `quantity /
//! 2`. The comparison is on the start and the end of every episode, which are
//! whole days here, so agreement is exact.

use epistat_register::exposure::{
    Assumption, Dispensings, DoseRule, Stock, Supply, exposure_periods,
};
use epistat_testkit::{Fingerprint, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Episodes {
    gap: f64,
    id: Vec<u32>,
    start: Vec<f64>,
    end: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    adherer_version: String,
    id: Vec<u32>,
    day: Vec<f64>,
    quantity: Vec<f64>,
    episodes: Vec<Episodes>,
}

#[test]
fn episodes_agree_with_adherer_at_every_permitted_gap() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/adherer_reference.json"
    ));
    assert_eq!(r.adherer_version, "0.8.3");
    let persons = *r.id.last().unwrap() as usize + 1;
    let mut offsets = vec![0_usize; persons + 1];
    for &p in &r.id {
        offsets[p as usize + 1] += 1;
    }
    for p in 0..persons {
        offsets[p + 1] += offsets[p];
    }
    #[expect(clippy::cast_possible_truncation, reason = "whole days from R")]
    let day: Vec<i64> = r.day.iter().map(|&d| d as i64).collect();
    let group = vec![Some(0); day.len()];
    let d = Dispensings {
        person_offsets: &offsets,
        group: &group,
        day: &day,
        quantity: &r.quantity,
    };
    let mut fp = Fingerprint::new();
    for e in &r.episodes {
        let assumptions = [Assumption {
            dose_default: 2.0,
            dose_min: 1.0,
            dose_max: 4.0,
            max_stock_days: f64::INFINITY,
            grace_days: e.gap,
        }];
        let s = Supply {
            assumptions: &assumptions,
            rule: DoseRule::Default,
            stock: Stock::CarryForward,
            pauses: None,
        };
        let p = exposure_periods(&d, &s).unwrap();
        assert_eq!(p.len(), e.id.len(), "number of episodes at gap {}", e.gap);
        assert!(e.id.len() > persons, "some person has several episodes");
        for i in 0..p.len() {
            let label = format!("gap {} episode {i}", e.gap);
            assert_eq!(p.person[i], e.id[i], "{label} person");
            assert_eq!(p.start[i] as f64, e.start[i], "{label} start");
            assert_eq!(p.end[i], e.end[i], "{label} end");
            fp = fp.u64(p.start[i] as u64).f64(p.end[i]);
        }
    }
    assert_eq!(fp.finish(), 7_957_771_175_707_055_590, "golden hash");
}
