//! The public functions of `aalen_johansen` give the same bits as when this
//! test was recorded.
//!
//! Inputs come from a seeded stream. The curves are run with and without
//! delayed entry and weights, with weights of zero, with groups, with ties on a
//! grid of tenths, with several causes (one of them unobserved in one input),
//! with an entry time equal to the exit time, and on empty and all-censored
//! input. The bootstrap runs in both draw modes over overlapping cells, a cell
//! with an empty arm, a cell with no rows, zero weights and ages before the
//! first and after the last event time.
//!
//! The bits depend on the platform's `ln` and `exp`, which come from its maths
//! library, so the test runs only on the target it was recorded on, aarch64
//! macOS. CI's macOS runner is that target. Thread-count independence, which
//! does not depend on the platform, is checked everywhere by the thread tests.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the grid values and counts are far below the limits of the casts"
)]
#![expect(
    deprecated,
    reason = "the test is of the deprecated functions, whose numbers must not move"
)]

use epistat_rng::Stream;
use epistat_survival::aalen_johansen::{
    ArmColumns, Draw, RowWindows, bootstrap_arm_incidence, curves,
};

mod golden_values;

struct Input {
    time: Vec<f64>,
    event: Vec<i32>,
    weight: Vec<f64>,
    group: Vec<u32>,
    entry: Vec<f64>,
}

fn input(n: usize, seed: u64, causes: usize, zero_weights: bool) -> Input {
    let mut rng = Stream::seeded(seed);
    let mut s = Input {
        time: Vec::new(),
        event: Vec::new(),
        weight: Vec::new(),
        group: Vec::new(),
        entry: Vec::new(),
    };
    for _ in 0..n {
        let entry = (rng.uniform() * 30.0).floor() / 10.0;
        s.entry.push(entry);
        s.time
            .push(entry + 0.1 + (rng.exp1() * 40.0).floor() / 10.0);
        s.event.push(match rng.below(10) {
            0..=5 => 1 + rng.below(causes) as i32,
            _ => 0,
        });
        let w = if zero_weights && rng.below(5) == 0 {
            0.0
        } else {
            0.25 + rng.uniform() * 2.0
        };
        s.weight.push(w);
        s.group.push(rng.below(4) as u32 * 3);
    }
    s
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn push_curves(
    out: &mut Vec<(String, Vec<u64>)>,
    name: &str,
    c: &epistat_survival::aalen_johansen::Curves,
) {
    out.push((
        format!("{name} causes"),
        c.causes
            .iter()
            .map(|&k| u64::from(k.unsigned_abs()))
            .collect(),
    ));
    out.push((
        format!("{name} group"),
        c.group.iter().map(|&g| u64::from(g)).collect(),
    ));
    for (label, column) in [
        ("time", &c.time),
        ("n_risk", &c.n_risk),
        ("n_at_risk", &c.n_at_risk),
        ("n_event", &c.n_event),
        ("event_free", &c.event_free),
    ] {
        out.push((format!("{name} {label}"), bits(column)));
    }
    for (k, column) in c.cif.iter().enumerate() {
        out.push((format!("{name} cif {k}"), bits(column)));
    }
}

fn curve_cases(out: &mut Vec<(String, Vec<u64>)>) {
    let a = input(240, 1, 3, false);
    let b = input(240, 2, 2, true);
    let ones = vec![1.0; 240];
    let one_group = vec![0u32; 240];
    for (name, s, weight, group, with_entry) in [
        ("plain", &a, &ones, &one_group, false),
        ("groups", &a, &ones, &a.group, false),
        ("entry", &a, &ones, &one_group, true),
        ("weights groups entry", &b, &b.weight, &b.group, true),
        ("weights groups", &b, &b.weight, &b.group, false),
        ("weights", &b, &b.weight, &one_group, false),
    ] {
        let entry = with_entry.then_some(s.entry.as_slice());
        let c = curves(&s.time, &s.event, weight, group, entry).unwrap();
        push_curves(out, name, &c);
    }

    // Three causes declared by the data, the third unobserved.
    let mut u = input(120, 3, 2, true);
    u.event[0] = 2;
    let c = curves(&u.time, &u.event, &u.weight, &u.group, Some(&u.entry)).unwrap();
    push_curves(out, "two causes", &c);

    // An entry time equal to the exit time, and a group of weight zero only.
    let time = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
    let entry = [0.0, 2.0, 0.0, 1.0, 4.0, 0.0];
    let event = [1, 2, 0, 1, 2, 0];
    let weight = [1.0, 0.0, 2.0, 0.5, 0.0, 0.0];
    let group = [0, 1, 0, 0, 1, 2];
    let c = curves(&time, &event, &weight, &group, Some(&entry)).unwrap();
    push_curves(out, "degenerate", &c);

    let c = curves(&time, &[0; 6], &weight, &group, None).unwrap();
    push_curves(out, "all censored", &c);
    let c = curves(&[], &[], &[], &[], None).unwrap();
    push_curves(out, "empty", &c);

    out.push((
        "errors".into(),
        vec![
            u64::from(curves(&[1.0], &[1, 1], &[1.0], &[0], None).is_err()),
            u64::from(curves(&[f64::NAN], &[1], &[1.0], &[0], None).is_err()),
            u64::from(curves(&[1.0], &[1], &[1.0], &[0], Some(&[f64::NAN])).is_err()),
        ],
    ));
}

fn frame(n: usize) -> (ArmColumns, RowWindows) {
    let s = input(n, 4, 3, true);
    let mut rng = Stream::seeded(5);
    let exposed: Vec<bool> = (0..n).map(|_| rng.below(2) == 0).collect();
    let family: Vec<u32> = (0..n).map(|_| rng.below(40) as u32).collect();
    let columns = ArmColumns::new(s.time, s.event, s.weight, family, exposed.clone()).unwrap();
    let mut rows = Vec::new();
    let mut offsets = vec![0u32];
    for (cell, len) in [150usize, 20, 90, 300, 0, 60, 41].into_iter().enumerate() {
        let start = cell * 37 % (n - len.max(1));
        for (row, &is_exposed) in exposed.iter().enumerate().skip(start).take(len) {
            // Cell 5 holds only exposed rows, cell 6 only comparators.
            if cell != 5 && cell != 6 || (cell == 5) == is_exposed {
                rows.push(row as u32);
            }
        }
        offsets.push(rows.len() as u32);
    }
    (columns, RowWindows::new(rows, offsets, n, "cell").unwrap())
}

fn bootstrap_cases(out: &mut Vec<(String, Vec<u64>)>) {
    let (columns, cells) = frame(400);
    let seeds: Vec<u64> = (0..cells.len() as u64).map(|c| 11 + 7919 * c).collect();
    let ages = vec![2.5, 0.05, 1.0, 40.0, 4.0];
    for (name, draw) in [("per cell", Draw::PerCell), ("shared", Draw::Shared)] {
        for cause in [1, 3] {
            let (exposed, comparator, estimable) =
                bootstrap_arm_incidence(&columns, &cells, &seeds, ages.clone(), cause, 9, draw)
                    .unwrap();
            out.push((format!("{name} cause {cause} exposed"), bits(&exposed)));
            out.push((
                format!("{name} cause {cause} comparator"),
                bits(&comparator),
            ));
            out.push((
                format!("{name} cause {cause} estimable"),
                estimable.iter().map(|&e| u64::from(e)).collect(),
            ));
        }
        let (e, c, ok) =
            bootstrap_arm_incidence(&columns, &cells, &seeds, vec![], 1, 9, draw).unwrap();
        out.push((
            format!("{name} no ages"),
            vec![e.len() as u64, c.len() as u64, ok.len() as u64],
        ));
        let (e, c, ok) =
            bootstrap_arm_incidence(&columns, &cells, &seeds, ages.clone(), 1, 0, draw).unwrap();
        out.push((
            format!("{name} no replicates"),
            vec![e.len() as u64, c.len() as u64, ok.len() as u64],
        ));
    }
    let none = RowWindows::new(vec![], vec![0], 400, "cell").unwrap();
    let r = bootstrap_arm_incidence(&columns, &none, &[], vec![1.0], 1, 3, Draw::PerCell).unwrap();
    out.push(("no cells".into(), vec![r.0.len() as u64]));
    out.push((
        "errors".into(),
        vec![
            u64::from(
                bootstrap_arm_incidence(&columns, &cells, &seeds[..2], ages, 1, 3, Draw::Shared)
                    .is_err(),
            ),
            u64::from(
                bootstrap_arm_incidence(
                    &columns,
                    &RowWindows::new(vec![], vec![0], 7, "cell").unwrap(),
                    &[],
                    vec![],
                    1,
                    3,
                    Draw::Shared,
                )
                .is_err(),
            ),
            u64::from(
                ArmColumns::new(vec![1.0], vec![1], vec![1.0], vec![0], vec![true, false]).is_err(),
            ),
        ],
    ));
}

fn compute() -> Vec<(String, Vec<u64>)> {
    let mut out = Vec::new();
    curve_cases(&mut out);
    bootstrap_cases(&mut out);
    out
}

#[test]
fn outputs_have_the_recorded_bits() {
    let got = compute();
    let expected = golden_values::EXPECTED;
    assert_eq!(got.len(), expected.len(), "number of recorded outputs");
    for ((name, values), (expected_name, expected_values)) in got.iter().zip(expected) {
        assert_eq!(name, expected_name);
        assert_eq!(values.as_slice(), *expected_values, "{name}");
    }
}

/// Prints the recorded values as the source of `golden_values/mod.rs`.
#[test]
#[ignore = "records the golden values"]
fn record() {
    use std::fmt::Write;

    let mut text = String::from("pub const EXPECTED: &[(&str, &[u64])] = &[\n");
    for (name, values) in compute() {
        let _ = write!(text, "    ({name:?}, &[");
        for (i, v) in values.iter().enumerate() {
            text.push_str(if i % 3 == 0 { "\n        " } else { " " });
            let hex = format!("{v:016x}");
            let grouped: Vec<&str> = (0..4).map(|g| &hex[4 * g..4 * g + 4]).collect();
            let _ = write!(text, "0x{},", grouped.join("_"));
        }
        text.push_str("\n    ]),\n");
    }
    text.push_str("];\n");
    std::fs::write(std::env::var("GOLDEN_OUT").unwrap(), text).unwrap();
}
