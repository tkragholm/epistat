//! The bits of the outcome models of sequential trials and of
//! clone-censor-weight on simulated person-period data.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output,
//! with its length, taken from the stacked rows built in full before the
//! outcome model is fitted. They must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the digests are
//! checked only on aarch64 macOS, where they were recorded.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::weighting::clone::{
    Adherence, CloneInput, DeviationModel, Strategy, clone_censor, deviation_weights,
};
use epistat_survival::weighting::trials::{
    Msm, MsmInput, PersonPeriod, TrialRows, WeightModel, fit_msm, switch_weights, trial_weights,
};
use epistat_testkit::Fingerprint;

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Person-period rows with a covariate that depends on the earlier treatment,
/// a persistent treatment and an outcome that ends the follow-up.
struct Periods {
    id: Vec<u32>,
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
            d.id.push(u32::try_from(id).unwrap());
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
/// last row of the person, fitted from the stacked rows built in full.
fn trial_msm_full(d: &Periods) -> Msm {
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
    fit_msm(&MsmInput {
        design: design.as_ref(),
        outcome: &y,
        weight: &tw,
        person: &tid,
    })
    .unwrap()
}

/// The clone-censor-weight analysis of the benchmark, with the design of the
/// outcome model built in full.
fn ccw_full(d: &Periods) -> Msm {
    let period: Vec<u32> = d
        .period
        .iter()
        .map(|&p| u32::try_from(p).unwrap())
        .collect();
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
            period: &period,
            treatment: &d.a,
            outcome: &d.outcome,
        },
        &strategies,
    )
    .unwrap();
    let m = c.len();
    let den = Mat::from_fn(m, 5, |i, j| {
        let s = c.source[i];
        let p = f64::from(period[s]);
        [1.0, d.l[s], d.l0[s], p, p * p][j]
    });
    let num = Mat::from_fn(m, 2, |i, j| [1.0, f64::from(period[c.source[i]])][j]);
    let model = DeviationModel {
        denominator: den.as_ref(),
        numerator: Some(num.as_ref()),
    };
    let w = deviation_weights(&c, &[model, model], None).unwrap();
    let keep: Vec<usize> = (0..m).filter(|&i| !c.deviation[i]).collect();
    let design = Mat::from_fn(keep.len(), 5, |k, j| {
        let i = keep[k];
        let p = f64::from(period[c.source[i]]);
        let arm = c.arm[i] as f64;
        [1.0, arm, p, p * p, arm * p][j]
    });
    let y: Vec<bool> = keep.iter().map(|&i| c.outcome[i]).collect();
    let wt: Vec<f64> = keep.iter().map(|&i| w.weight[i]).collect();
    let person: Vec<u32> = keep.iter().map(|&i| c.person[i]).collect();
    fit_msm(&MsmInput {
        design: design.as_ref(),
        outcome: &y,
        weight: &wt,
        person: &person,
    })
    .unwrap()
}

/// `(name, length, digest)` of each recorded output.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const EXPECTED: &[(&str, usize, u64)] = &[
    ("trial msm", 32, 0xe0e2_53ac_13f3_a6d9),
    ("ccw", 32, 0xa72a_7c84_5364_6e98),
];

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn outputs() -> Vec<(String, Vec<f64>)> {
    let trials = periods(3000, 8, 20_261_124);
    let ccw = periods(3000, 10, 20_261_125);
    vec![
        ("trial msm".to_string(), flatten(&trial_msm_full(&trials))),
        ("ccw".to_string(), flatten(&ccw_full(&ccw))),
    ]
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn bits_are_unchanged() {
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
