//! The bits of the outcome models of sequential trials and of
//! clone-censor-weight on simulated person-period data, with the stacked rows
//! built in full and formed block by block by `fit_msm_trials` and
//! `fit_msm_clones`, and of the deviation weights from held designs and from
//! designs formed row by row by `deviation_weights_rows`.
//!
//! The two ways give the same bits on every platform. The recorded values are
//! an FNV-1a digest of the `to_bits` of each output, with its length, taken
//! from the rows built in full before the rows were formed block by block.
//! They must not change. To record again, run with `GOLDEN_RECORD=1 --
//! --nocapture` and paste the lines. The bits depend on the platform's `ln`
//! and `exp`, so the digests are checked only on aarch64 macOS, where they
//! were recorded.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::weighting::clone::{
    Adherence, CloneInput, DeviationDesign, DeviationModel, DeviationRows, DeviationWeights,
    Strategy, clone_censor, deviation_weights, deviation_weights_rows, fit_msm_clones,
};
use epistat_survival::weighting::trials::{
    Msm, MsmInput, PersonPeriod, TrialRows, TrialSpans, WeightModel, fit_msm, fit_msm_trials,
    switch_weights, trial_weights,
};

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Person-period rows with a covariate that depends on the earlier treatment,
/// a persistent treatment and an outcome that ends the follow-up.
struct Periods {
    id: Vec<u64>,
    period: Vec<i64>,
    a: Vec<bool>,
    am1: Vec<bool>,
    l: Vec<f64>,
    l0: Vec<f64>,
    x1: Vec<f64>,
    outcome: Vec<bool>,
}

fn periods(n: usize, length: usize, seed: u64) -> Periods {
    let mut rng = Stream::seeded(seed);
    let mut d = Periods {
        id: Vec::new(),
        period: Vec::new(),
        a: Vec::new(),
        am1: Vec::new(),
        l: Vec::new(),
        l0: Vec::new(),
        x1: Vec::new(),
        outcome: Vec::new(),
    };
    for id in 0..n {
        let x1 = f64::from(u8::from(rng.uniform() < 0.5));
        let l0 = rng.normal();
        let (mut l, mut prev) = (l0, false);
        for p in 0..length {
            l = 0.5 * l + 0.4 * f64::from(u8::from(prev)) + 0.7 * rng.normal();
            let pa = if prev {
                expit(2.4 - 0.5 * l)
            } else {
                expit(-2.2 + 0.7 * l + 0.1 * p as f64 + 0.3 * x1)
            };
            let a = rng.uniform() < pa;
            let y = rng.uniform()
                < expit(-3.0 - 0.6 * f64::from(u8::from(a)) + 0.4 * l + 0.05 * p as f64);
            d.id.push(u64::try_from(id).unwrap());
            d.period.push(i64::try_from(p).unwrap());
            d.a.push(a);
            d.am1.push(prev);
            d.l.push(l);
            d.l0.push(l0);
            d.x1.push(x1);
            d.outcome.push(y);
            prev = a;
            if y {
                break;
            }
        }
    }
    d
}

fn flatten(m: &Msm) -> Vec<f64> {
    let mut v = m.coefficients.clone();
    v.extend(&m.covariance);
    v.extend([m.n_rows as f64, m.n_persons as f64]);
    v
}

/// The trials of the benchmark: one starting in every period, running to the
/// last row of the person, fitted from the stacked rows built in full and by
/// `fit_msm_trials`.
fn trial_msm_both(d: &Periods) -> (Msm, Msm) {
    let rows = d.id.len();
    let den = Mat::from_fn(rows, 3, |i, j| [1.0, d.l[i], d.x1[i]][j]);
    let num = Mat::from_fn(rows, 2, |i, j| [1.0, d.x1[i]][j]);
    let data = PersonPeriod {
        id: &d.id,
        period: &d.period,
        treatment: &d.a,
        previous: &d.am1,
    };
    let w = switch_weights(
        &data,
        &WeightModel {
            numerator: num.as_ref(),
            denominator: den.as_ref(),
        },
    )
    .unwrap();
    let (mut tid, mut tstart, mut tfollow, mut src) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut first = 0;
    while first < rows {
        let mut last = first;
        while last + 1 < rows && d.id[last + 1] == d.id[first] {
            last += 1;
        }
        for s in first..=last {
            for r in s..=last {
                tid.push(d.id[first]);
                tstart.push(d.period[s]);
                tfollow.push(d.period[r] - d.period[s]);
                src.push((s, r));
            }
        }
        first = last + 1;
    }
    let tw = trial_weights(
        &data,
        &w.weight,
        &TrialRows {
            id: &tid,
            trial_period: &tstart,
            follow_up: &tfollow,
        },
    )
    .unwrap();
    let design = Mat::from_fn(tid.len(), 5, |i, j| {
        let (s, _) = src[i];
        let f = tfollow[i] as f64;
        [1.0, f64::from(u8::from(d.a[s])), f, f * f, d.x1[s]][j]
    });
    let y: Vec<bool> = src.iter().map(|&(_, r)| d.outcome[r]).collect();
    let full = fit_msm(&MsmInput {
        design: design.as_ref(),
        outcome: &y,
        weight: &tw,
        person: &tid,
    })
    .unwrap();
    // The same trials as spans: each starts at a row and runs to the last row
    // of its person.
    let mut start = Vec::new();
    let mut end = Vec::new();
    for s in 0..rows {
        let mut e = s;
        while e < rows && d.id[e] == d.id[s] {
            e += 1;
        }
        start.push(s);
        end.push(e);
    }
    let spans = fit_msm_trials(
        &data,
        &w.weight,
        &TrialSpans {
            start: &start,
            end: &end,
        },
        &d.outcome,
        5,
        &|s, r, out| {
            let f = (d.period[r] - d.period[s]) as f64;
            out.copy_from_slice(&[1.0, f64::from(u8::from(d.a[s])), f, f * f, d.x1[s]]);
        },
    )
    .unwrap();
    (full, spans)
}

/// The clone-censor-weight analysis of the benchmark, with the design of the
/// outcome model built in full and formed by `fit_msm_clones`.
fn ccw_both(d: &Periods) -> (Msm, Msm) {
    let period = &d.period;
    let strategies = [
        Strategy::InitiateWithin {
            grace: 3,
            adherence: Adherence::Continuous,
        },
        Strategy::NeverInitiate { until: None },
    ];
    let c = clone_censor(
        &CloneInput {
            id: &d.id,
            period,
            treatment: &d.a,
            outcome: &d.outcome,
            covariates: &[],
        },
        &strategies,
    )
    .unwrap();
    let m = c.len();
    let den = Mat::from_fn(m, 5, |i, j| {
        let s = c.source[i];
        let p = period[s] as f64;
        [1.0, d.l[s], d.l0[s], p, p * p][j]
    });
    let num = Mat::from_fn(m, 2, |i, j| [1.0, period[c.source[i]] as f64][j]);
    let model = DeviationModel {
        denominator: den.as_ref(),
        numerator: Some(num.as_ref()),
    };
    let w = deviation_weights(&c, &[model, model], None).unwrap();
    let keep: Vec<usize> = (0..m).filter(|&i| !c.deviation[i]).collect();
    let design = Mat::from_fn(keep.len(), 5, |k, j| {
        let i = keep[k];
        let p = period[c.source[i]] as f64;
        let arm = c.arm[i] as f64;
        [1.0, arm, p, p * p, arm * p][j]
    });
    let y: Vec<bool> = keep.iter().map(|&i| c.outcome[i]).collect();
    let wt: Vec<f64> = keep.iter().map(|&i| w.weight[i]).collect();
    let person: Vec<u64> = keep.iter().map(|&i| u64::from(c.person[i])).collect();
    let full = fit_msm(&MsmInput {
        design: design.as_ref(),
        outcome: &y,
        weight: &wt,
        person: &person,
    })
    .unwrap();
    // The weights again from designs formed row by row, which must be the
    // same, with and without case weights and a numerator.
    let den_fill = |i: usize, out: &mut [f64]| {
        let s = c.source[i];
        let p = period[s] as f64;
        out.copy_from_slice(&[1.0, d.l[s], d.l0[s], p, p * p]);
    };
    let num_fill = |i: usize, out: &mut [f64]| {
        out.copy_from_slice(&[1.0, period[c.source[i]] as f64]);
    };
    let rows_model = DeviationRows {
        denominator: DeviationDesign {
            columns: 5,
            fill: &den_fill,
        },
        numerator: Some(DeviationDesign {
            columns: 2,
            fill: &num_fill,
        }),
    };
    let by_rows = deviation_weights_rows(&c, &[rows_model, rows_model], None).unwrap();
    assert_eq!(weights_bits(&by_rows), weights_bits(&w));
    let counts: Vec<f64> = (0..m)
        .map(|i| f64::from(u8::try_from(c.person[i] % 3).unwrap()))
        .collect();
    let unstabilised = DeviationModel {
        numerator: None,
        ..model
    };
    let rows_unstabilised = DeviationRows {
        numerator: None,
        ..rows_model
    };
    assert_eq!(
        weights_bits(
            &deviation_weights_rows(&c, &[rows_unstabilised, rows_model], Some(&counts)).unwrap()
        ),
        weights_bits(&deviation_weights(&c, &[unstabilised, model], Some(&counts)).unwrap())
    );
    let rows = fit_msm_clones(&c, &by_rows.weight, 5, &|i, out| {
        let p = period[c.source[i]] as f64;
        let arm = c.arm[i] as f64;
        out.copy_from_slice(&[1.0, arm, p, p * p, arm * p]);
    })
    .unwrap();
    (full, rows)
}

fn weights_bits(w: &DeviationWeights) -> Vec<u64> {
    w.weight
        .iter()
        .chain(&w.probability)
        .chain(w.coefficients.iter().flatten())
        .map(|v| v.to_bits())
        .collect()
}

/// `(name, length, digest)` of each recorded output.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const EXPECTED: &[(&str, usize, u64)] = &[
    ("trial msm", 32, 0xe0e2_53ac_13f3_a6d9),
    ("trial msm", 32, 0xe0e2_53ac_13f3_a6d9),
    ("ccw", 32, 0xa72a_7c84_5364_6e98),
    ("ccw", 32, 0xa72a_7c84_5364_6e98),
];

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn outputs() -> Vec<(String, Vec<f64>)> {
    let trials = periods(3000, 8, 20_261_124);
    let ccw = periods(3000, 10, 20_261_125);
    let (trial_full, trial_rows) = trial_msm_both(&trials);
    let (ccw_full, ccw_rows) = ccw_both(&ccw);
    vec![
        ("trial msm".to_string(), flatten(&trial_full)),
        ("trial msm".to_string(), flatten(&trial_rows)),
        ("ccw".to_string(), flatten(&ccw_full)),
        ("ccw".to_string(), flatten(&ccw_rows)),
    ]
}

fn bits(m: &Msm) -> Vec<u64> {
    flatten(m).iter().map(|v| v.to_bits()).collect()
}

#[test]
fn rows_formed_by_blocks_give_the_bits_of_rows_built_in_full() {
    // Several sizes, so that the stack has one block, a block and a row, and
    // many blocks with persons across their ends.
    for (persons, seed) in [(40, 1), (520, 2), (3000, 3)] {
        let (full, rows) = trial_msm_both(&periods(persons, 8, seed));
        assert_eq!(bits(&rows), bits(&full), "trials, {persons} persons");
        let (full, rows) = ccw_both(&periods(persons, 10, seed + 100));
        assert_eq!(bits(&rows), bits(&full), "clones, {persons} persons");
    }
}

#[test]
fn spans_are_checked() {
    let d = periods(30, 6, 5);
    let data = PersonPeriod {
        id: &d.id,
        period: &d.period,
        treatment: &d.a,
        previous: &d.am1,
    };
    let w = vec![1.0; d.id.len()];
    let design = |_: usize, _: usize, out: &mut [f64]| out.fill(1.0);
    let fit = |start: &[usize], end: &[usize]| {
        fit_msm_trials(
            &data,
            &w,
            &TrialSpans { start, end },
            &d.outcome,
            1,
            &design,
        )
    };
    let last = d.id.iter().position(|&i| i != d.id[0]).unwrap();
    assert!(fit(&[0], &[0]).is_err());
    assert!(fit(&[0], &[last + 1]).is_err());
    assert!(fit(&[1, 0], &[2, 1]).is_err());
    assert!(fit(&[0, 1], &[1]).is_err());
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn bits_are_unchanged() {
    use epistat_testkit::Fingerprint;

    let got: Vec<(String, usize, u64)> = outputs()
        .into_iter()
        .map(|(name, v)| {
            let digest = Fingerprint::new().u64(v.len() as u64).f64s(&v).finish();
            (name, v.len(), digest)
        })
        .collect();
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        for (name, len, digest) in &got {
            println!("    ({name:?}, {len}, {digest:#020x}),");
        }
        return;
    }
    let expected: Vec<(String, usize, u64)> = EXPECTED
        .iter()
        .map(|&(n, l, d)| (n.to_string(), l, d))
        .collect();
    assert_eq!(got, expected);
}
