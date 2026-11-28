//! Person-time tables compared with `survival::pyears` without a rate table:
//! counts, events of unit weight and the number of contributing rows exactly,
//! person-time and weighted events within a bound derived from rounding.

use epistat_survival::SurvData;
use epistat_survival::pyears::{Dimension, PyearsOptions, pyears};
use epistat_survival::timefix::TimeFix;
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    r_version: String,
    survival_version: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Spec {
    kind: String,
    codes: Option<Vec<u32>>,
    levels: Option<usize>,
    origin: Option<Vec<f64>>,
    cuts: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    scale: f64,
    start: Option<Vec<f64>>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weight: Option<Vec<f64>>,
    dims: Vec<Spec>,
    pyears: Vec<f64>,
    n: Vec<f64>,
    event: Vec<f64>,
    offtable: f64,
}

#[test]
fn pyears_matches_survival() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/pyears.json"
    ));
    println!(
        "{}, survival {}",
        reference.r_version, reference.survival_version
    );
    let mut worst_time = 0.0_f64;
    let mut worst_off = 0.0_f64;
    for case in &reference.cases {
        let mut builder = SurvData::builder(&case.stop, &case.status);
        if let Some(start) = &case.start {
            builder = builder.start(start);
        }
        if let Some(weight) = &case.weight {
            builder = builder.weight(weight);
        }
        let data = builder.build().unwrap();
        let dims: Vec<Dimension<'_>> = case
            .dims
            .iter()
            .map(|spec| match spec.kind.as_str() {
                "factor" => Dimension::Factor {
                    codes: spec.codes.as_ref().unwrap(),
                    levels: spec.levels.unwrap(),
                },
                _ => Dimension::Time {
                    origin: spec.origin.as_ref().unwrap(),
                    cuts: spec.cuts.as_ref().unwrap(),
                },
            })
            .collect();
        // R does not apply the tie rule.
        let opts = PyearsOptions {
            scale: case.scale,
            cause: None,
            timefix: TimeFix::Exact,
            ..Default::default()
        };
        let got = pyears(&data, &dims, &opts).unwrap();
        let name = &case.name;
        assert_eq!(got.person_time.len(), case.pyears.len(), "{name}: cells");

        // Largest magnitude among the numbers that make up a boundary
        // offset. Each is formed by one or two additions and a subtraction, so
        // each piece of follow-up is off by at most 4 eps times it, in both
        // implementations (R also accumulates the starting value piece by
        // piece, which adds a further few eps per piece). The bound for a
        // cell is 16 eps M for each row and unit of weight that touches it.
        let mut magnitude = case.stop.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if let Some(start) = &case.start {
            magnitude += start.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        }
        for spec in &case.dims {
            if let Some(origin) = &spec.origin {
                magnitude += origin.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            }
            if let Some(cuts) = &spec.cuts {
                magnitude = magnitude.max(cuts.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
            }
        }
        let wmax = case
            .weight
            .as_ref()
            .map_or(1.0, |w| w.iter().fold(0.0_f64, |m, &v| m.max(v)));
        for cell in 0..case.pyears.len() {
            assert!(
                (got.n[cell] as f64 - case.n[cell]).abs() < 0.5,
                "{name}: n in cell {cell} is {}, R gives {}",
                got.n[cell],
                case.n[cell]
            );
            let bound = 16.0 * f64::EPSILON * magnitude * wmax * case.n[cell] / case.scale;
            let deviation = (got.person_time[cell] - case.pyears[cell]).abs();
            assert!(
                deviation <= bound,
                "{name}: person-time in cell {cell} is {}, R gives {}, bound {bound:e}",
                got.person_time[cell],
                case.pyears[cell]
            );
            if bound > 0.0 {
                worst_time = worst_time.max(deviation / bound);
            }
            // Events are sums of weights, whole numbers added in row order
            // for unit weights and sums of at most n weights otherwise.
            let event_bound = 4.0 * f64::EPSILON * wmax * (case.n[cell] + 1.0);
            assert!(
                (got.events[cell] - case.event[cell]).abs() <= event_bound,
                "{name}: events in cell {cell} are {}, R gives {}",
                got.events[cell],
                case.event[cell]
            );
        }
        let n_total: f64 = case.n.iter().sum();
        let bound = 16.0 * f64::EPSILON * magnitude * wmax * (n_total + 1.0) / case.scale;
        let deviation = (got.offtable - case.offtable).abs();
        assert!(
            deviation <= bound,
            "{name}: off table is {}, R gives {}",
            got.offtable,
            case.offtable
        );
        worst_off = worst_off.max(deviation / bound);
    }
    println!("worst person-time deviation in units of its bound: {worst_time:e}");
    println!("worst off-table deviation in units of its bound: {worst_off:e}");
}
