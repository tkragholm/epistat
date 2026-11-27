//! `flexpar::fit`, `predict` and `contrast` give identical bits in rayon pools
//! of 1, 2, 3, 4 and 8 threads, for rows that span several blocks.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::flexpar::{
    Contrast, FlexParFit, FlexParOptions, Predict, PredictOptions, Scale, TimeVarying, contrast,
    fit, predict,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

fn flat(f: &FlexParFit) -> Vec<f64> {
    let mut out = f.gamma.clone();
    out.extend(f.beta.iter().map(|b| b.unwrap_or(f64::NAN)));
    out.extend(f.tvc_coefficients.iter().flatten());
    out.extend(f.covariance.iter().flatten().map(|v| v.unwrap_or(f64::NAN)));
    out.extend(
        f.robust_covariance
            .iter()
            .flatten()
            .map(|v| v.unwrap_or(f64::NAN)),
    );
    out.extend([
        f.log_likelihood,
        f.null_log_likelihood.unwrap_or(f64::NAN),
        f64::from(f.iterations),
        f.n_events,
    ]);
    out
}

/// Weights, delayed entry, clusters, a duplicated column and a time-varying effect.
#[test]
fn flexpar_fit_predict_contrast() {
    let n = 3500;
    let mut rng = Stream::seeded(41);
    let (mut time, mut start, mut status, mut weight, mut cluster) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut x = Mat::<f64>::zeros(n, 3);
    for i in 0..n {
        let a = rng.normal();
        let b = f64::from(u8::from(rng.below(5) < 2));
        let t = rng.exp1().powf(0.7) * (-(0.4 * a - 0.5 * b)).exp() * 3.0 + 0.001;
        let c = rng.exp1() * 6.0 + 0.001;
        let entry = if rng.below(3) == 0 && 0.3 * t < t.min(c) {
            0.3 * t
        } else {
            0.0
        };
        time.push(t.min(c));
        status.push(i32::from(t <= c));
        start.push(entry);
        weight.push(1.0 + f64::from(u32::try_from(rng.below(3)).unwrap()));
        cluster.push(u32::try_from(i / 5).unwrap());
        x[(i, 0)] = a;
        x[(i, 1)] = b;
        x[(i, 2)] = 2.0 * a;
    }
    let data = SurvData::builder(&time, &status)
        .start(&start)
        .weight(&weight)
        .cluster(&cluster)
        .build()
        .unwrap();
    let tvc = [TimeVarying { column: 1, df: 2 }];
    let xs = Mat::from_fn(2, 3, |i, j| [[0.3, 1.0, 9.0], [-0.5, 0.0, 9.0]][i][j]);
    let x0 = Mat::from_fn(1, 3, |_, j| [-0.5, 0.0, 9.0][j]);
    let times = [0.1, 0.5, 1.0, 2.5];
    for scale in [Scale::Hazard, Scale::Odds, Scale::Probit] {
        let options = FlexParOptions {
            tvc: &tvc,
            ..FlexParOptions::new(scale, 3)
        };
        assert_same_on_pools(&POOLS, || {
            let f = fit(&data, x.as_ref(), &options).unwrap();
            assert!(f.converged && f.beta[2].is_none());
            let mut out = flat(&f);
            let o = PredictOptions {
                robust: true,
                ..PredictOptions::default()
            };
            for what in [
                Predict::Survival,
                Predict::CumulativeHazard,
                Predict::Hazard,
            ] {
                for curve in predict(&f, xs.as_ref(), &times, what, &o).unwrap() {
                    for p in curve {
                        out.extend([p.estimate, p.se, p.lower.unwrap(), p.upper.unwrap()]);
                    }
                }
            }
            for what in [
                Contrast::HazardRatio,
                Contrast::HazardDifference,
                Contrast::SurvivalDifference,
            ] {
                for curve in contrast(&f, xs.as_ref(), x0.as_ref(), &times, what, &o).unwrap() {
                    for p in curve {
                        out.extend([p.estimate, p.se, p.lower.unwrap(), p.upper.unwrap()]);
                    }
                }
            }
            out
        });
    }
}
