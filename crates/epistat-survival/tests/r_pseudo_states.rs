//! `pseudo::pseudo_states` checked against `survival::pseudo` 3.8-12 on
//! multi-state `survfit` objects: the occupation probabilities (`pstate`), the
//! time in each state (`auc`) and the cumulative transition hazards (`cumhaz`),
//! for an illness-death process with plain follow-up, subjects starting in either
//! of the first two states and two strata. See `data/pseudo_states.R`.
//!
//! # Tolerances
//!
//! Each stratum has `m` distinct exit times, `h` transitions that occur and
//! `S` states, and `Y_j(l)` and `n_j(l)` are the number at risk in state `j` and
//! the number leaving it at exit time `l`, all whole numbers computed from the
//! data. With `A` the largest over states of the sum over exit times of
//! `n_j / (Y_j (Y_j - n_j))` and `b` the largest `1 / (Y_j - n_j)` (a state with
//! everyone leaving at once contributes `1 / Y_j`):
//!
//! - An estimate is a product of at most `m` factors, each a vector-matrix
//!   product of `S` terms, or a sum of at most `m` rectangles or increments. Its
//!   error is at most `16 (m + 8) (S + 1) eps` times its scale: 1 for an occupation
//!   probability, the time from the origin to the time for a time in state, and the
//!   larger of 1 and its value for a hazard.
//! - The influence of a subject is a sum over its `r` rows, and for each row of
//!   `m (h + 2)` terms of absolute size at most `A`, `b` or 1 times the scale,
//!   each formed with a few roundings in either program, so its error is at most
//!   `2 (7 m + 20) (h + 2) eps r scale (2 A + b + 1)`, and the pseudo-observation,
//!   which is `n` times it plus the estimate, has that times `n` plus the error of
//!   the estimate.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "labels and counts are whole numbers"
)]

use std::collections::HashMap;

use epistat_survival::SurvData;
use epistat_survival::pseudo::{StateOptions, StateTarget, pseudo_states};
use epistat_testkit::load_json;
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
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    state: Vec<f64>,
    ist: Vec<f64>,
    group: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    times: Vec<f64>,
    origin: f64,
    outputs: Vec<String>,
    group: Vec<u32>,
    subject: Vec<u32>,
    n_subjects: Vec<usize>,
    values: Vec<f64>,
    estimate: Vec<f64>,
}

/// `m`, `h`, `A` and `b` of the rows `rows`.
fn tables(d: &Dataset, rows: &[usize]) -> (f64, f64, f64, f64) {
    let mut times: Vec<f64> = rows.iter().map(|&i| d.stop[i]).collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let mut seen = std::collections::BTreeSet::new();
    let (mut a, mut b) = ([0.0f64; 3], 0.0f64);
    for &t in &times {
        for j in 1..=3u32 {
            let at_risk = rows
                .iter()
                .filter(|&&i| d.ist[i] as u32 == j && d.start[i] < t && t <= d.stop[i])
                .count() as f64;
            let moved = rows
                .iter()
                .filter(|&&i| {
                    d.ist[i] as u32 == j && d.start[i] < t && d.stop[i] == t && d.state[i] > 0.0
                })
                .count() as f64;
            if moved > 0.0 {
                let rest = if at_risk - moved > 0.0 {
                    at_risk - moved
                } else {
                    at_risk
                };
                a[j as usize - 1] += moved / (at_risk * rest);
                b = b.max(1.0 / rest);
            }
        }
    }
    for &i in rows {
        if d.state[i] > 0.0 {
            seen.insert((d.ist[i] as u32, d.state[i] as u32));
        }
    }
    (
        times.len() as f64,
        seen.len() as f64,
        a.iter().fold(0.0f64, |m, &v| m.max(v)),
        b,
    )
}

#[test]
fn multistate_pseudo_agrees_with_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/pseudo_states.json"
    ));
    eprintln!(
        "{}, survival {}",
        reference.versions.r, reference.versions.survival
    );
    let mut worst: HashMap<String, (f64, String)> = HashMap::new();
    let mut see = |output: &str, label: &str, got: f64, want: f64, tol: f64| {
        let ratio = (got - want).abs() / tol;
        assert!(
            ratio <= 1.0,
            "{output} at {label}: got {got:e}, R {want:e}, tolerance {tol:e}"
        );
        let entry = worst
            .entry(output.to_string())
            .or_insert((0.0, String::new()));
        if ratio >= entry.0 {
            *entry = (ratio, label.to_string());
        }
    };
    for case in &reference.cases {
        let d = &reference.datasets[&case.name];
        let status: Vec<i32> = d.state.iter().map(|&s| s as i32).collect();
        let ist: Vec<u32> = d.ist.iter().map(|&s| s as u32).collect();
        let mut builder = SurvData::builder(&d.stop, &status)
            .n_codes(3)
            .start(&d.start)
            .id(&d.id)
            .istate(&ist);
        let multi = d.group.iter().any(|&g| g != d.group[0]);
        if multi {
            builder = builder.stratum(&d.group);
        }
        let data = builder.build().unwrap();
        let target = match case.kind.as_str() {
            "pstate" => StateTarget::Occupation,
            "auc" => StateTarget::TimeInState,
            _ => StateTarget::CumulativeHazard,
        };
        let label = format!("{} {}", case.name, case.kind);
        let got = pseudo_states(&data, &case.times, &StateOptions::new(target, case.origin))
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let k = case.times.len();
        let width = case.outputs.len();
        assert_eq!(got.width, width, "{label}: width");
        assert_eq!(got.group, case.group, "{label}: groups");
        assert_eq!(got.subject, case.subject, "{label}: subjects");
        assert_eq!(got.values.len(), case.values.len(), "{label}: shape");
        let mut first = 0;
        for (g, estimate) in got.groups.iter().enumerate() {
            assert_eq!(estimate.n_subjects, case.n_subjects[g], "{label}: n");
            let n = estimate.n_subjects;
            let rows: Vec<usize> = (0..d.stop.len())
                .filter(|&i| !multi || d.group[i] == estimate.group)
                .collect();
            let (m, h, a, b) = tables(d, &rows);
            let sc = |t: f64, value: f64| match target {
                StateTarget::Occupation => 1.0,
                StateTarget::TimeInState => (t - case.origin).max(1.0),
                StateTarget::CumulativeHazard => value.max(1.0),
            };
            for c in 0..n {
                let subject = case.subject[first + c];
                let r = rows.iter().filter(|&&i| d.id[i] == subject).count() as f64;
                for (j, &t) in case.times.iter().enumerate() {
                    for o in 0..width {
                        let idx = (first + c) * k * width + j * width + o;
                        let want_est = case.estimate[g * k * width + j * width + o];
                        let scale = sc(t, want_est);
                        let est_tol = 16.0 * (m + 8.0) * 4.0 * EPS * scale;
                        let infl_scale = if target == StateTarget::CumulativeHazard {
                            1.0
                        } else {
                            scale
                        };
                        let tol = n as f64
                            * 2.0
                            * (7.0 * m + 20.0)
                            * (h + 2.0)
                            * EPS
                            * r
                            * infl_scale
                            * (2.0 * a + b + 1.0)
                            + est_tol;
                        let at = format!("{label} subject {subject} time {t} output {o}");
                        see("value", &at, got.values[idx], case.values[idx], tol);
                        if c == 0 {
                            see(
                                "estimate",
                                &at,
                                estimate.estimate[j * width + o],
                                want_est,
                                est_tol,
                            );
                        }
                    }
                }
            }
            first += n;
        }
    }
    let mut rows: Vec<_> = worst.into_iter().collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (output, (ratio, label)) in rows {
        eprintln!("worst {output}: {ratio:.4} of tolerance at {label}");
    }
}
