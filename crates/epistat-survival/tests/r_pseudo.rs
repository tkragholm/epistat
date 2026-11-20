//! `pseudo::pseudo` checked against `survival::pseudo` 3.8-12, against
//! leave-one-subject-out refits of `survival::survfit`, and against the `pseudo`
//! package's `pseudosurv`, `pseudomean` and `pseudoci`. Seven data sets (plain,
//! strata, delayed entry, subjects with several rows, case weights, case weights
//! with several rows, clusters with weights and strata) are used as single-event
//! data and as data with two causes, for the survival probability, the
//! incidences, the restricted mean and the restricted time lost to each cause,
//! each by both jackknives. See `data/pseudo.R`.
//!
//! # Tolerances
//!
//! The case weights are multiples of 1/4, so every weight at risk and of events is
//! exact, and every difference is rounding in products and sums. With `m` the
//! number of exit times of a group, `n` its subjects, and `A` and `b` the
//! largest value of the sum `A = sum d / (Y (Y - d))` of Greenwood increments
//! and of `1 / (Y - d)` over its event times:
//!
//! - An estimate of the survival probability or the incidence is a product or a
//!   sum of at most `m` factors of three roundings each in either program. Its
//!   error is at most `12 (m + 4) eps` and that of an area, at most `12 (m + 4) eps`
//!   times the horizon less the origin, `r`. A value of the target is at most
//!   `s = 1` or `r`.
//! - An exact pseudo-value is `n theta - (n - 1) theta(-i)`, so it carries the error
//!   of the estimates multiplied by `2 n`, `2 n 12 (m + 4) eps s`.
//! - An infinitesimal pseudo-value is `theta + n U`. The influence of a row is the
//!   weight, the target's scale `s`, and a difference of sums over at most `m`
//!   terms each of absolute size at most `A` or `b`. Its error is at most
//!   `(7 m + 20) eps s w (2 A + b + 1)` per row in either program, `w` the
//!   weight, so a subject with the weight `W` has at most `2 (7 m + 20) eps s W
//!   (2 A + b + 1)`, and the pseudo-value that times `n` plus the error of
//!   `theta`. For the incidence and the time lost, whose influence comes from a
//!   recursion that adds a term at every time, the factor is three times
//!   greater.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "labels and counts are whole numbers"
)]

use std::collections::HashMap;

use epistat_survival::SurvData;
use epistat_survival::pseudo::{Method, PseudoOptions, Target, pseudo};
use epistat_survival::survfit::{ConfType, SurvfitOptions, estimate};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: HashMap<String, Dataset>,
    cases: Vec<Case>,
    pkg: Pkg,
    plain: Plain,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    survival: String,
    pseudo: String,
}

#[derive(Deserialize)]
struct Dataset {
    stop: Vec<f64>,
    start: Vec<f64>,
    status: Vec<f64>,
    group: Vec<u32>,
    weight: Vec<f64>,
    id: Vec<u32>,
    cluster: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    kind: String,
    target: String,
    method: String,
    times: Vec<f64>,
    group: Vec<u32>,
    subject: Vec<u32>,
    values: Vec<f64>,
    estimate: Vec<f64>,
    n_subjects: Vec<usize>,
    r_rows: Option<usize>,
    r_pseudo: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Pkg {
    times: Vec<f64>,
    surv: Vec<f64>,
    mean: Vec<f64>,
    ci1: Vec<f64>,
    ci2: Vec<f64>,
}

#[derive(Deserialize)]
struct Plain {
    time: Vec<f64>,
    status: Vec<f64>,
}

#[derive(Default)]
struct Worst(HashMap<String, (f64, String)>);

impl Worst {
    #[track_caller]
    fn see(&mut self, output: &str, label: &str, got: f64, want: f64, tolerance: f64) {
        let deviation = (got - want).abs();
        let ratio = deviation / tolerance;
        assert!(
            ratio <= 1.0,
            "{output} at {label}: got {got:e}, R {want:e}, deviation {deviation:e} against tolerance {tolerance:e}"
        );
        let entry = self
            .0
            .entry(output.to_string())
            .or_insert((0.0, String::new()));
        if ratio >= entry.0 {
            *entry = (ratio, label.to_string());
        }
    }

    fn report(self) {
        let mut rows: Vec<_> = self.0.into_iter().collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        for (output, (ratio, label)) in rows {
            eprintln!("worst {output}: {ratio:.4} of tolerance at {label}");
        }
    }
}

fn target_of(name: &str) -> Target {
    match name {
        "survival" => Target::Survival,
        "cif1" => Target::CumulativeIncidence { cause: 1 },
        "cif2" => Target::CumulativeIncidence { cause: 2 },
        "rmst" => Target::RestrictedMean,
        "rmtl1" => Target::RestrictedTimeLost { cause: 1 },
        "rmtl2" => Target::RestrictedTimeLost { cause: 2 },
        _ => unreachable!(),
    }
}

/// The Greenwood sum and the largest `1 / (Y - d)` of a group's event times.
fn greenwood(d: &Dataset, group: u32) -> (usize, f64, f64) {
    let rows: Vec<usize> = (0..d.stop.len())
        .filter(|&i| d.group.is_empty() || d.group[i] == group)
        .collect();
    let stop: Vec<f64> = rows.iter().map(|&i| d.stop[i]).collect();
    let status: Vec<i32> = rows.iter().map(|&i| i32::from(d.status[i] > 0.0)).collect();
    let start: Vec<f64> = rows
        .iter()
        .map(|&i| d.start.get(i).copied().unwrap_or(0.0))
        .collect();
    let weight: Vec<f64> = rows
        .iter()
        .map(|&i| d.weight.get(i).copied().unwrap_or(1.0))
        .collect();
    let mut builder = SurvData::builder(&stop, &status).weight(&weight);
    if !d.start.is_empty() {
        builder = builder.start(&start);
    }
    let id: Vec<u32> = rows
        .iter()
        .map(|&i| d.id.get(i).copied().unwrap_or(0))
        .collect();
    if !d.id.is_empty() {
        builder = builder.id(&id);
    }
    let data = builder.build().unwrap();
    let curve = estimate(
        &data,
        &SurvfitOptions {
            conf_type: ConfType::None,
            robust: Some(false),
            ..SurvfitOptions::default()
        },
    );
    // A cluster or an id makes the model-based variance unavailable for some
    // inputs, and only the risk sets are used here.
    let curve = curve.unwrap_or_else(|_| {
        estimate(
            &data,
            &SurvfitOptions {
                conf_type: ConfType::None,
                robust: None,
                ..SurvfitOptions::default()
            },
        )
        .unwrap()
    });
    let g = &curve.groups[0];
    let (mut a, mut b) = (0.0f64, 0.0f64);
    for i in 0..g.time.len() {
        let (y, e) = (g.n_risk[i], g.n_event[i]);
        if e > 0.0 {
            let rest = if y - e > 0.0 { y - e } else { y };
            a += e / (y * rest);
            b = b.max(1.0 / rest);
        }
    }
    (g.time.len(), a, b)
}

#[test]
fn pseudo_agrees_with_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/pseudo.json"
    ));
    eprintln!(
        "{}, survival {}, pseudo {}",
        reference.versions.r, reference.versions.survival, reference.versions.pseudo
    );
    let mut worst = Worst::default();
    let mut compared = 0;
    let mut same_as_pseudo = 0;
    for case in &reference.cases {
        let d = &reference.datasets[&case.name];
        let status: Vec<i32> = d
            .status
            .iter()
            .map(|&s| {
                if case.kind == "km" {
                    i32::from(s > 0.0)
                } else {
                    s as i32
                }
            })
            .collect();
        let mut builder =
            SurvData::builder(&d.stop, &status).n_codes(if case.kind == "km" { 1 } else { 2 });
        if !d.start.is_empty() {
            builder = builder.start(&d.start);
        }
        if !d.group.is_empty() {
            builder = builder.stratum(&d.group);
        }
        if !d.weight.is_empty() {
            builder = builder.weight(&d.weight);
        }
        if !d.id.is_empty() {
            builder = builder.id(&d.id);
        }
        if !d.cluster.is_empty() {
            builder = builder.cluster(&d.cluster);
        }
        let data = builder.build().unwrap();
        let mut opts = PseudoOptions::new(target_of(&case.target));
        opts.method = if case.method == "ij" {
            Method::InfinitesimalJackknife
        } else {
            Method::Jackknife
        };
        let label = format!(
            "{} {} {} {}",
            case.name, case.kind, case.target, case.method
        );
        let got = pseudo(&data, &case.times, &opts).unwrap_or_else(|e| panic!("{label}: {e}"));
        compared += 1;
        let k = case.times.len();

        // Subjects: same groups and labels as R, R counting rows from 1.
        let row_labels = d.id.is_empty() && d.cluster.is_empty();
        let want_labels: Vec<u32> = case
            .subject
            .iter()
            .map(|&s| if row_labels { s - 1 } else { s })
            .collect();
        assert_eq!(got.subject, want_labels, "{label}: subjects");
        assert_eq!(got.group, case.group, "{label}: groups");
        assert_eq!(got.values.len(), case.values.len(), "{label}: shape");
        // R's values are written subject by subject, the times within each.
        let mut start = 0;
        let area = matches!(
            opts.target,
            Target::RestrictedMean | Target::RestrictedTimeLost { .. }
        );
        let engine = matches!(
            opts.target,
            Target::CumulativeIncidence { .. } | Target::RestrictedTimeLost { .. }
        ) && opts.method == Method::InfinitesimalJackknife;
        for (g, estimate_g) in got.groups.iter().enumerate() {
            assert_eq!(estimate_g.n_subjects, case.n_subjects[g], "{label}: n");
            let n = estimate_g.n_subjects;
            let (m, a, b) = greenwood(d, estimate_g.group);
            let (m, nf) = (m as f64, n as f64);
            for (j, &t) in case.times.iter().enumerate() {
                let scale = if area { t.max(1.0) } else { 1.0 };
                let est_tol = 12.0 * (m + 4.0) * EPS * scale;
                worst.see(
                    "estimate",
                    &format!("{label} group {} time {t}", estimate_g.group),
                    estimate_g.estimate[j],
                    case.estimate[g * k + j],
                    est_tol,
                );
                let last_time = d
                    .stop
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| d.group.is_empty() || d.group[i] == estimate_g.group)
                    .map(|(_, &s)| s)
                    .fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(
                    estimate_g.beyond[j],
                    t > last_time,
                    "{label}: beyond at {t}"
                );
                for c in 0..n {
                    let row = start + c;
                    let value = got.values[row * k + j];
                    let want = case.values[row * k + j];
                    let tol = if case.method == "jack" {
                        2.0 * nf * est_tol
                    } else {
                        // The weight of the subject's rows.
                        let w: f64 = if d.weight.is_empty() {
                            1.0
                        } else {
                            let label_of = |i: usize| {
                                if !d.id.is_empty() {
                                    d.id[i]
                                } else if !d.cluster.is_empty() {
                                    d.cluster[i]
                                } else {
                                    i as u32 + 1
                                }
                            };
                            (0..d.stop.len())
                                .filter(|&i| {
                                    label_of(i) == case.subject[row]
                                        && (d.group.is_empty() || d.group[i] == estimate_g.group)
                                })
                                .map(|i| d.weight[i])
                                .sum()
                        };
                        let factor = if engine { 3.0 } else { 1.0 };
                        factor * nf * 2.0 * (7.0 * m + 20.0) * EPS * scale * w * (2.0 * a + b + 1.0)
                            + est_tol
                    };
                    worst.see(
                        &format!(
                            "{} {}",
                            case.method,
                            if engine { "engine" } else { "closed" }
                        ),
                        &format!("{label} subject {} time {t}", case.subject[row]),
                        value,
                        want,
                        tol,
                    );
                }
            }
            start += n;
        }

        // survival::pseudo itself, where it has the same subjects.
        if let Some(r) = &case.r_pseudo {
            same_as_pseudo += 1;
            for (i, (&mine, &theirs)) in got.values.iter().zip(r).enumerate() {
                assert!(
                    (mine - theirs).abs() <= 1e-9 * (1.0 + theirs.abs()),
                    "{label}: survival::pseudo differs from the pseudo-values here at {i}"
                );
            }
        } else if case.method == "ij" {
            // survival::pseudo does not collapse a cluster column without an id.
            assert!(!d.cluster.is_empty() && d.id.is_empty(), "{label}");
            if let Some(rows) = case.r_rows {
                assert_eq!(rows, d.stop.len(), "{label}: R returns one value per row");
            }
        }
    }
    eprintln!("compared {compared} cases, {same_as_pseudo} against survival::pseudo itself");
    worst.report();
    assert_eq!(compared, 112);
}

/// The exact jackknife against `pseudosurv`, `pseudomean` and `pseudoci` of
/// the `pseudo` package, on single-event and two-cause data without weights.
/// Tolerance as for the exact jackknife above, with `m` the number of distinct
/// times: `2 n 12 (m + 4) eps` times the scale of the target.
#[test]
fn jackknife_agrees_with_the_pseudo_package() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/pseudo.json"
    ));
    let (time, pkg) = (&reference.plain.time, &reference.pkg);
    let n = time.len();
    let mut distinct = time.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    let m = distinct.len() as f64;
    let binary: Vec<i32> = reference
        .plain
        .status
        .iter()
        .map(|&s| i32::from(s > 0.0))
        .collect();
    let causes: Vec<i32> = reference.plain.status.iter().map(|&s| s as i32).collect();
    let km = SurvData::builder(time, &binary).build().unwrap();
    let cr = SurvData::builder(time, &causes).n_codes(2).build().unwrap();
    let mut worst = Worst::default();
    let k = pkg.times.len();
    for (name, data, target, want, area) in [
        ("pseudosurv", &km, Target::Survival, &pkg.surv, false),
        ("pseudomean", &km, Target::RestrictedMean, &pkg.mean, true),
        (
            "pseudoci cause 1",
            &cr,
            Target::CumulativeIncidence { cause: 1 },
            &pkg.ci1,
            false,
        ),
        (
            "pseudoci cause 2",
            &cr,
            Target::CumulativeIncidence { cause: 2 },
            &pkg.ci2,
            false,
        ),
    ] {
        let mut opts = PseudoOptions::new(target);
        opts.method = Method::Jackknife;
        let got = pseudo(data, &pkg.times, &opts).unwrap();
        assert_eq!(got.values.len(), n * k);
        for (i, (&mine, &theirs)) in got.values.iter().zip(want).enumerate() {
            let scale = if area { pkg.times[i % k] } else { 1.0 };
            let tolerance = 2.0 * n as f64 * 12.0 * (m + 4.0) * EPS * scale;
            worst.see(name, &format!("value {i}"), mine, theirs, tolerance);
        }
    }
    worst.report();
}
