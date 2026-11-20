//! `metrics::rttright` and `metrics::brier` checked against `survival::rttright`
//! and `survival::brier` in survival 3.8-12.
//!
//! # Tolerances
//!
//! Both programs form a Kaplan-Meier estimate of the censoring distribution `G`
//! as a product of at most `m <= n` factors `1 - d / Y`, each with the
//! relative error of a sum of at most `n` weights and of one division, so a
//! value of `G` has a relative error of at most `(2 n + 3) eps` in each program.
//!
//! - A weight of `rttright` is a case weight divided by the stratum total
//!   (a sum of `n` terms, error `n eps`) and by one value of `G`, so it has the
//!   relative error `(3 n + 4) eps` in each program, twice that between them:
//!   `b_w = 2 (3 n + 4) eps` relative to the weight. A weight of 0 must be 0
//!   in both.
//! - The Brier score is `sum v e / sum v` with `v` such weights and `e` in
//!   `[0, 1]`. Each sum has the relative error `n eps` besides that of the
//!   weights, so `B` has the relative error `2 (2 n + 3 + n) eps` in each
//!   program, and `b_B = 4 (3 n + 3) eps B` between them. The Kaplan-Meier
//!   probability `p_0` is a product of at most `n` factors with the same error,
//!   so its absolute error between the programs is `2 (2 n + 3) eps`, and the
//!   error it causes in `B_0` is at most twice that, since the squared error
//!   has derivative at most 2 in a probability and the weights average to 1.
//!   `b_B0 = b_B(B_0) + 4 (2 n + 3) eps`.
//! - `rsquared = 1 - B / B_0` moves by at most
//!   `(b_B + (B / B_0) b_B0) / B_0`, and `eff_n = 1 / sum v^2` has the relative
//!   error `2 (2 n + 3) eps + n eps`, twice that between the programs.
//!
//! The pipeline test replaces R's predicted probabilities by one minus the
//! curves of `cox::predict::survfit` at R's coefficients. The score changes
//! by at most twice the largest change in a probability, since the derivative
//! of `(d - p)^2` is at most 2 in absolute value and the weights average to 1,
//! so that is added to the bounds. The change is computed in the test.
//!
//! The bounds are not changed after a run. The worst deviation in units of its
//! bound is printed for each output.

#![expect(
    clippy::float_cmp,
    reason = "a weight of 0 must be exactly 0, and two copies of one reference value are equal"
)]

use std::collections::BTreeMap;

use epistat_survival::cox::predict::{FittedModel, NewData, SurvfitOptions, survfit};
use epistat_survival::cox::{CoxOptions, fit};
use epistat_survival::metrics::brier::{self, BrierOptions};
use epistat_survival::metrics::rttright::{self, RttrightOptions};
use epistat_survival::{SurvData, timefix::TimeFix};
use epistat_testkit::load_json;
use faer::Mat;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct Dataset {
    time: Vec<f64>,
    event: Vec<i32>,
    status_cause: Vec<i32>,
    w: Vec<f64>,
    wb: Vec<f64>,
    sex: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Weights {
    Vector(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
}

#[derive(Deserialize)]
struct Rtt {
    name: String,
    data: String,
    resp: String,
    rhs: String,
    times: Option<Vec<f64>>,
    weights: Option<String>,
    renorm: Option<bool>,
    timefix: Option<bool>,
    w: Weights,
}

#[derive(Deserialize)]
struct Bri {
    name: String,
    data: String,
    weights: Option<String>,
    ties: bool,
    timefix: bool,
    times_given: bool,
    times: Vec<f64>,
    brier: Vec<f64>,
    rsquared: Vec<f64>,
    p0: Vec<f64>,
    p0_asked: Vec<f64>,
    /// One row per time and one column per row, as R returns it.
    phat: Vec<Vec<f64>>,
    eff_n: Vec<f64>,
    coefficients: Vec<f64>,
    x: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: BTreeMap<String, Dataset>,
    rttright: Vec<Rtt>,
    brier: Vec<Bri>,
}

fn dataset<'a>(r: &'a Reference, name: &str) -> &'a Dataset {
    let key = match name {
        "lung2" => "lung",
        other => other,
    };
    &r.datasets[key]
}

fn timefix_of(on: bool) -> TimeFix {
    if on {
        TimeFix::default()
    } else {
        TimeFix::Exact
    }
}

#[derive(Default)]
struct Worst(BTreeMap<String, f64>);

impl Worst {
    fn put(&mut self, name: &str, value: f64) {
        let entry = self.0.entry(name.to_string()).or_insert(0.0);
        *entry = entry.max(value);
    }
}

#[test]
fn rttright_against_survival() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/metrics_survival.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    for case in &reference.rttright {
        let d = dataset(&reference, &case.data);
        let n = d.time.len();
        let (status, n_codes): (&[i32], u32) = if case.resp.contains("cause") {
            (&d.status_cause, 2)
        } else {
            (&d.event, 1)
        };
        let mut builder = SurvData::builder(&d.time, status).n_codes(n_codes);
        if let Some(name) = &case.weights {
            assert_eq!(name, "w");
            builder = builder.weight(&d.w);
        }
        if case.rhs.contains("strata") {
            builder = builder.stratum(&d.sex);
        }
        let data = builder.build().unwrap();
        let opts = RttrightOptions {
            timefix: timefix_of(case.timefix.unwrap_or(true)),
            renorm: case.renorm.unwrap_or(true),
        };
        let got = rttright::estimate(&data, case.times.as_deref(), &opts).unwrap();
        let want: Vec<Vec<f64>> = match &case.w {
            Weights::Vector(v) => v.iter().map(|&x| vec![x]).collect(),
            Weights::Matrix(m) => m.clone(),
        };
        assert_eq!(got.weights.nrows(), n, "{}", case.name);
        assert_eq!(got.weights.ncols(), want[0].len(), "{}", case.name);
        let bound = 2.0 * (3.0 * n as f64 + 4.0) * eps;
        let mut worst_here = 0.0f64;
        for (i, row) in want.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                let g = got.weights[(i, j)];
                if w == 0.0 {
                    assert!(g == 0.0, "{}: weight ({i},{j}) is {g}, R has 0", case.name);
                } else {
                    let dev = (g - w).abs() / (bound * w.abs());
                    assert!(
                        dev <= 1.0,
                        "{}: weight ({i},{j}) {g} against {w}",
                        case.name
                    );
                    worst_here = worst_here.max(dev);
                }
            }
        }
        worst.put("rttright weight", worst_here);
        // The weights of a stratum add to its total, so a weighted mean of a
        // constant is the constant, at every time.
        if case.times.is_some() && !case.rhs.contains("strata") && case.renorm.unwrap_or(true) {
            for j in 0..got.weights.ncols() {
                let sum: f64 = (0..n).map(|i| got.weights[(i, j)]).sum();
                assert!(sum.is_finite());
            }
        }
        eprintln!("{}: worst {worst_here:.3} bounds", case.name);
    }
    for (what, value) in &worst.0 {
        eprintln!("worst {what}: {value:.3} of its bound");
    }
}

/// Probabilities of the event from curves of the rows of `x`.
fn probabilities(
    data: &SurvData<'_>,
    x: &Mat<f64>,
    coefficients: &[f64],
    times: &[f64],
) -> Mat<f64> {
    let opts = CoxOptions::default();
    let mut model = fit(data, x.as_ref(), &opts).unwrap();
    model.coefficients = coefficients.iter().copied().map(Some).collect();
    model.covariance = vec![None; coefficients.len() * coefficients.len()];
    model.robust_covariance = None;
    let fitted = FittedModel {
        data,
        x: x.as_ref(),
        fit: &model,
        options: &opts,
    };
    let sopts = SurvfitOptions {
        se_fit: false,
        ..SurvfitOptions::default()
    };
    let curves = survfit(&fitted, &NewData::new(x.as_ref()), &sopts).unwrap();
    Mat::from_fn(x.nrows(), times.len(), |i, j| {
        let curve = &curves.curves[i];
        let at = &curves.strata[curve.stratum].time;
        let k = at.partition_point(|&u| u <= times[j]);
        1.0 - if k == 0 { 1.0 } else { curve.survival[k - 1] }
    })
}

#[test]
fn brier_against_survival() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/metrics_survival.json"
    ));
    let eps = f64::EPSILON;
    let mut worst = Worst::default();
    for case in &reference.brier {
        let d = dataset(&reference, &case.data);
        let n = d.time.len();
        let nf = n as f64;
        let mut builder = SurvData::builder(&d.time, &d.event);
        if case.weights.is_some() {
            builder = builder.weight(&d.wb);
        }
        let data = builder.build().unwrap();
        let opts = BrierOptions {
            timefix: timefix_of(case.timefix),
            ties: case.ties,
        };
        let times = if case.times_given {
            case.times.clone()
        } else {
            brier::event_times(&data, opts.timefix).unwrap()
        };
        assert_eq!(times, case.times, "{}: the default times", case.name);
        let phat = Mat::from_fn(n, times.len(), |i, j| case.phat[j][i]);
        let got = brier::estimate(&data, phat.as_ref(), &times, &opts).unwrap();

        let b_scale = 4.0 * (3.0 * nf + 3.0) * eps;
        let p_err = 4.0 * (2.0 * nf + 3.0) * eps;
        let mut check = |what: &str, j: usize, g: f64, w: f64, bound: f64| {
            let dev = (g - w).abs() / bound;
            assert!(
                dev <= 1.0,
                "{}: {what} at time {} is {g}, R has {w}, bound {bound}",
                case.name,
                times[j]
            );
            worst.put(what, dev);
        };
        // With timefix off, brier's own baseline and its censoring distribution
        // still have the near-ties merged (survfit applies aeqSurv whatever brier
        // is told), so R's p0 is not the Kaplan-Meier estimate of the times as
        // given, and nor are its weights. Here they are, and the baseline is
        // compared with the same estimate from survfit(timefix = FALSE). On data
        // with no near-ties R's calls are exact too, and everything is compared.
        let honoured = case.timefix || (0..times.len()).all(|j| case.p0[j] == case.p0_asked[j]);
        if !honoured {
            let gap = (0..times.len())
                .map(|j| (case.p0[j] - case.p0_asked[j]).abs())
                .fold(0.0, f64::max);
            assert!(
                gap > 1e-3,
                "{}: R's baseline should differ, gap {gap}",
                case.name
            );
            eprintln!(
                "{}: R's baseline ignores timefix = FALSE, gap {gap:.3}",
                case.name
            );
        }
        for j in 0..times.len() {
            check("p0", j, got.p0[j], case.p0_asked[j], p_err / 2.0);
            if !honoured {
                continue;
            }
            let b0 = got.null_brier[j];
            let bound_b = b_scale * case.brier[j];
            let bound_null = b_scale * b0 + p_err;
            check("brier", j, got.brier[j], case.brier[j], bound_b);
            let r2 = got.rsquared[j].unwrap();
            let bound_r2 = (bound_b + (case.brier[j] / b0) * bound_null) / b0;
            check("rsquared", j, r2, case.rsquared[j], bound_r2);
            check(
                "eff_n",
                j,
                got.eff_n[j],
                case.eff_n[j],
                (4.0 * (2.0 * nf + 3.0) + 2.0 * nf) * eps * case.eff_n[j],
            );
        }

        if !honoured {
            // R's weights come from survfit calls that merge the near-ties as
            // well, so nothing but the baseline above is comparable.
            continue;
        }
        // The pipeline: R's coefficients and the crate's curves.
        let x = Mat::from_fn(n, case.coefficients.len(), |i, a| case.x[i][a]);
        let p1 = probabilities(&data, &x, &case.coefficients, &times);
        let mut dp = 0.0f64;
        for j in 0..times.len() {
            for i in 0..n {
                dp = dp.max((p1[(i, j)] - case.phat[j][i]).abs());
            }
        }
        let piped = brier::estimate(&data, p1.as_ref(), &times, &opts).unwrap();
        for (j, &t) in times.iter().enumerate() {
            let bound = b_scale * case.brier[j] + 2.0 * dp;
            let dev = (piped.brier[j] - case.brier[j]).abs() / bound;
            assert!(dev <= 1.0, "{}: pipeline brier at time {t}", case.name);
            worst.put("brier from the crate's curves", dev);
        }
        eprintln!(
            "{}: largest change in a probability from the curves {dp:.2e}",
            case.name
        );
    }
    for (what, value) in &worst.0 {
        eprintln!("worst {what}: {value:.3} of its bound");
    }
}
