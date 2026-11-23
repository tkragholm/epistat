//! Exposure periods checked against `heaven::medicinMacro` (type "dynamic")
//! (`tests/data/heaven_reference.R`).
//!
//! The reference uses one strength and equal minimum, maximum and default
//! dose, the setting in which heaven's dose estimation reduces to a constant
//! daily dose and the two rules coincide. Adjacent exposure intervals of heaven
//! are joined, since heaven writes a row at every change of dose or admission.
//! Both count whole days here, so agreement is exact. The differences are in
//! the module documentation of `epistat_register::exposure`.

use epistat_register::exposure::{
    Assumption, Dispensings, DoseRule, Pauses, Stock, Supply, exposure_periods,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Periods {
    id: Vec<u32>,
    start: Vec<i64>,
    end: Vec<i64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    depot_days: Option<f64>,
    pauses: bool,
    periods: Periods,
}

#[derive(Deserialize)]
struct Reference {
    heaven_version: String,
    dose: f64,
    id: Vec<u32>,
    day: Vec<i64>,
    quantity: Vec<f64>,
    pause_ok: Vec<u32>,
    pause_id: Vec<u32>,
    pause_start: Vec<i64>,
    pause_end: Vec<i64>,
    cases: Vec<Case>,
}

fn offsets(ids: &[u32], persons: usize) -> Vec<usize> {
    let mut offsets = vec![0_usize; persons + 1];
    for &p in ids {
        offsets[p as usize + 1] += 1;
    }
    for p in 0..persons {
        offsets[p + 1] += offsets[p];
    }
    offsets
}

#[test]
fn periods_agree_with_heaven() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/heaven_reference.json"
    ));
    assert_eq!(r.heaven_version, "2026.10.30");
    let persons = *r.id.last().unwrap() as usize + 1;
    let dispensing_offsets = offsets(&r.id, persons);
    let group = vec![Some(0); r.day.len()];
    let d = Dispensings {
        person_offsets: &dispensing_offsets,
        group: &group,
        day: &r.day,
        quantity: &r.quantity,
    };
    let pause_offsets = offsets(&r.pause_id, persons);
    for case in &r.cases {
        let assumptions = [Assumption {
            dose_default: r.dose,
            dose_min: r.dose,
            dose_max: r.dose,
            max_stock_days: case.depot_days.unwrap_or(f64::INFINITY),
            grace_days: 0.0,
        }];
        let pauses = Pauses {
            person_offsets: &pause_offsets,
            start: &r.pause_start,
            end: &r.pause_end,
        };
        let s = Supply {
            assumptions: &assumptions,
            rule: DoseRule::Default,
            stock: Stock::CarryForward,
            pauses: case.pauses.then_some(&pauses),
        };
        let p = exposure_periods(&d, &s).unwrap();
        let want = &case.periods;
        // With pauses, only the persons for whom the two rules coincide.
        let compared = |person: u32| !case.pauses || r.pause_ok.contains(&person);
        let got: Vec<(u32, i64, i64)> = (0..p.len())
            .filter(|&i| compared(p.person[i]))
            .map(|i| (p.person[i], p.start[i], p.end_day(i)))
            .collect();
        let want: Vec<(u32, i64, i64)> = (0..want.id.len())
            .filter(|&i| compared(want.id[i]))
            .map(|i| (want.id[i], want.start[i], want.end[i]))
            .collect();
        assert_eq!(got, want, "{}", case.name);
    }
}
