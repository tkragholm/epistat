//! `metrics::rttright` and `metrics::brier` on counting-process data with an id
//! and one common start, against `survival::rttright` and `survival::brier` in
//! survival 3.8-12 (`tests/data/metrics_counting.R`).
//!
//! Subjects are cut into one to three rows, the intermediate rows censored.
//! The reference that does not depend on the cuts is the same subjects in one
//! row each without a start column, which `rttright` handles correctly: a
//! subject's weight, summed over its rows, must equal that.
//!
//! `rttright` with reporting times is wrong in `survival` for these data, and the
//! test asserts the defect (see the module documentation of
//! `metrics::rttright`). Where R is right, in every column of the weights
//! without reporting times and in every entry of the others that the defect does
//! not touch, the weights equal R's. The defective entries are two kinds:
//! a last row that is censored before the time, which R weighs and the weights
//! here leave at 0, and a last row that starts at or after the time and stops at
//! or after it, which R weighs although the row that covers the time already
//! carries the subject's weight.
//!
//! `brier` of R counts every row of an id as a subject. With one row per id and
//! a start it equals the score of the same data without the start, and the test
//! shows that the split data has another score, so that the data is refused.
//!
//! # Tolerances
//!
//! `n` rows of the data. As in `r_metrics_survival.rs`, a weight has the relative
//! error `2 (3 n + 4) eps` between the programs, the score `b_B = 4 (3 n + 3) eps
//! B`, the score of the Kaplan-Meier model `b_B0 = b_B(B_0) + 4 (2 n + 3) eps`,
//! the R-squared `(b_B + (B / B_0) b_B0) / B_0` and the effective sample size
//! `(2 (2 n + 3) + n) eps` twice over, relative.
//!
//! The bounds are not changed after a run.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::metrics::brier::{self, BrierOptions};
use epistat_survival::metrics::rttright::{self, RttrightOptions};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Split {
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    cause: Vec<i32>,
    grp: Vec<u32>,
    w: Vec<f64>,
}

#[derive(Deserialize)]
struct Subjects {
    id: Vec<u32>,
    stop: Vec<f64>,
    cause: Vec<i32>,
}

#[derive(Deserialize)]
struct Case {
    competing: bool,
    renorm: bool,
    strata: bool,
    none_split: Vec<f64>,
    none_unsplit: Vec<f64>,
    times_split: Vec<Vec<f64>>,
    times_unsplit: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct BrierRun {
    brier: Vec<f64>,
    rsquared: Vec<f64>,
    #[serde(default)]
    p0: Vec<f64>,
    #[serde(default)]
    eff_n: Vec<f64>,
    #[serde(default)]
    phat: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct BrierRef {
    times: Vec<f64>,
    unsplit: BrierRun,
    start: BrierRun,
    split: BrierRun,
}

#[derive(Deserialize)]
struct Reference {
    versions: std::collections::BTreeMap<String, String>,
    split: Split,
    subjects: Subjects,
    times: Vec<f64>,
    origin: f64,
    cases: Vec<Case>,
    brier: BrierRef,
}

fn status_of(cause: &[i32], competing: bool) -> Vec<i32> {
    cause
        .iter()
        .map(|&c| if competing { c } else { i32::from(c > 0) })
        .collect()
}

#[test]
fn rttright_with_a_start_against_survival() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/metrics_counting.json"
    ));
    eprintln!("reference: {:?}", r.versions);
    let s = &r.split;
    let n = s.id.len();
    let eps = f64::EPSILON;
    let bound = 2.0 * (3.0 * n as f64 + 4.0) * eps;
    let grp0: Vec<u32> = s.grp.iter().map(|g| g - 1).collect();
    let mut last = vec![false; n];
    {
        let mut best: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for i in 0..n {
            let e = best.entry(s.id[i]).or_insert(i);
            if s.stop[i] > s.stop[*e] {
                *e = i;
            }
        }
        for &i in best.values() {
            last[i] = true;
        }
    }
    let (mut worst, mut worst_sum) = (0.0f64, 0.0f64);
    let (mut n_d1, mut n_d2) = (0, 0);
    for case in &r.cases {
        let status = status_of(&s.cause, case.competing);
        let mut builder = SurvData::builder(&s.stop, &status)
            .start(&s.start)
            .id(&s.id)
            .weight(&s.w)
            .n_codes(if case.competing { 2 } else { 1 });
        if case.strata {
            builder = builder.stratum(&grp0);
        }
        let data = builder.build().unwrap();
        let opts = RttrightOptions {
            renorm: case.renorm,
            ..RttrightOptions::default()
        };
        let tag = format!(
            "competing {} renorm {} strata {}",
            case.competing, case.renorm, case.strata
        );
        // Without reporting times R is right.
        let none = rttright::estimate(&data, None, &opts).unwrap();
        for i in 0..n {
            let (got, want) = (none.weights[(i, 0)], case.none_split[i]);
            if want == 0.0 {
                assert!(got == 0.0, "{tag}: row {i} is {got}, R has 0");
            } else {
                let dev = (got - want).abs() / (bound * want);
                assert!(dev <= 1.0, "{tag}: row {i} is {got}, R has {want}");
                worst = worst.max(dev);
            }
        }
        // The same subjects in one row each.
        let mut sum_by_id = vec![0.0; r.subjects.id.len()];
        for i in 0..n {
            sum_by_id[(s.id[i] - 1) as usize] += none.weights[(i, 0)];
        }
        for (k, &w) in case.none_unsplit.iter().enumerate() {
            let dev = (sum_by_id[k] - w).abs() / (bound * w.max(1e-300));
            assert!(
                w == 0.0 && sum_by_id[k] == 0.0 || dev <= 1.0,
                "{tag}: subject {k} weighs {} against {w}",
                sum_by_id[k]
            );
        }

        let got = rttright::estimate(&data, Some(&r.times), &opts).unwrap();
        for (j, &t) in r.times.iter().enumerate() {
            assert!(t > r.origin);
            let mut sum_by_id = vec![0.0; r.subjects.id.len()];
            for i in 0..n {
                let (g, w) = (got.weights[(i, j)], case.times_split[i][j]);
                sum_by_id[(s.id[i] - 1) as usize] += g;
                let censored = status[i] == 0;
                let d1 = last[i] && censored && s.stop[i] < t;
                let d2 = last[i] && s.stop[i] >= t && s.start[i] >= t;
                if d1 || d2 {
                    // R weighs a row that carries no weight.
                    assert!(g == 0.0, "{tag}: row {i} at {t} is {g}, expected 0");
                    assert!(
                        w > 0.0,
                        "{tag}: R has {w} for row {i} at {t}, expected a weight"
                    );
                    if d1 {
                        n_d1 += 1;
                    } else {
                        n_d2 += 1;
                    }
                } else if w == 0.0 {
                    assert!(g == 0.0, "{tag}: row {i} at {t} is {g}, R has 0");
                } else {
                    let dev = (g - w).abs() / (bound * w);
                    assert!(dev <= 1.0, "{tag}: row {i} at {t} is {g}, R has {w}");
                    worst = worst.max(dev);
                }
            }
            for (k, row) in case.times_unsplit.iter().enumerate() {
                let w = row[j];
                let g = sum_by_id[k];
                if w == 0.0 {
                    assert!(g == 0.0, "{tag}: subject {k} at {t} weighs {g}, expected 0");
                } else {
                    let dev = (g - w).abs() / (bound * w);
                    assert!(
                        dev <= 1.0,
                        "{tag}: subject {k} at {t} weighs {g} against {w}"
                    );
                    worst_sum = worst_sum.max(dev);
                }
            }
        }
        if case.renorm && !case.strata {
            for j in 0..r.times.len() {
                let rust: f64 = (0..n).map(|i| got.weights[(i, j)]).sum();
                let theirs: f64 = case.times_split.iter().map(|row| row[j]).sum();
                assert!((rust - 1.0).abs() <= 8.0 * bound, "{tag}: sums to {rust}");
                assert!(theirs > 1.0 + 1e-6, "{tag}: R sums to {theirs}");
            }
        }
    }
    assert!(
        n_d1 > 20 && n_d2 > 5,
        "the defect must be exercised: {n_d1} {n_d2}"
    );
    eprintln!(
        "worst weight {worst:.3}, worst subject sum {worst_sum:.3} of the bound; defective R \
         entries seen: {n_d1} censored last rows, {n_d2} extra last rows"
    );
}

#[test]
fn brier_with_a_start_against_survival() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/metrics_counting.json"
    ));
    let b = &r.brier;
    let subj = &r.subjects;
    let n = subj.id.len();
    let nf = n as f64;
    let eps = f64::EPSILON;
    let status = status_of(&subj.cause, false);
    let start = vec![r.origin; n];
    let data = SurvData::builder(&subj.stop, &status)
        .start(&start)
        .id(&subj.id)
        .build()
        .unwrap();
    let t = b.times.len();
    // R's predictions: one row per time, one column per row.
    let predicted = Mat::from_fn(n, t, |i, j| b.start.phat[j][i]);
    let got = brier::estimate(
        &data,
        predicted.as_ref(),
        &b.times,
        &BrierOptions::default(),
    )
    .unwrap();
    let b_b = |value: f64| 4.0 * (3.0 * nf + 3.0) * eps * value;
    for j in 0..t {
        let (bj, b0) = (got.brier[j], got.null_brier[j]);
        let want = b.start.brier[j];
        assert!((bj - want).abs() <= b_b(want), "brier {bj} against {want}");
        let r2 = got.rsquared[j].unwrap();
        let b_r2 = (b_b(want) + (want / b0) * (b_b(b0) + 4.0 * (2.0 * nf + 3.0) * eps)) / b0;
        assert!(
            (r2 - b.start.rsquared[j]).abs() <= b_r2,
            "rsquared {r2} against {}",
            b.start.rsquared[j]
        );
        assert!(
            (got.p0[j] - b.start.p0[j]).abs() <= 2.0 * (2.0 * nf + 3.0) * eps,
            "p0"
        );
        let rel = 2.0 * (2.0 * (2.0 * nf + 3.0) + nf) * eps;
        assert!(
            (got.eff_n[j] - b.start.eff_n[j]).abs() <= rel * b.start.eff_n[j],
            "eff_n"
        );
        // Without the start column R and the crate give the same: the start only
        // says when follow-up began.
        assert!((b.unsplit.brier[j] - b.start.brier[j]).abs() <= b_b(want));
    }
    // R's score of the split data differs, since every row is a subject.
    for j in 0..t {
        assert!(
            (b.split.brier[j] - b.unsplit.brier[j]).abs() > 1e-3,
            "R's score of the split data should differ at time {}",
            b.times[j]
        );
    }
    let s = &r.split;
    let split_status = status_of(&s.cause, false);
    let split = SurvData::builder(&s.stop, &split_status)
        .start(&s.start)
        .id(&s.id)
        .build()
        .unwrap();
    let m = Mat::from_fn(s.id.len(), t, |_, _| 0.5);
    let e = brier::estimate(&split, m.as_ref(), &b.times, &BrierOptions::default()).unwrap_err();
    assert!(e.to_string().contains("several rows"), "{e}");
    // A start without an id, and subjects that start at different times.
    let no_id = SurvData::builder(&subj.stop, &status)
        .start(&start)
        .build()
        .unwrap();
    let e = brier::estimate(
        &no_id,
        predicted.as_ref(),
        &b.times,
        &BrierOptions::default(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("needs an id"), "{e}");
    let mut uneven = start.clone();
    uneven[3] = r.origin - 0.25;
    let different = SurvData::builder(&subj.stop, &status)
        .start(&uneven)
        .id(&subj.id)
        .build()
        .unwrap();
    let e = brier::estimate(
        &different,
        predicted.as_ref(),
        &b.times,
        &BrierOptions::default(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("different times"), "{e}");
}
