//! `survreg::fit`, `predict` and `residuals` give identical bits in rayon pools
//! of 1, 2, 3, 4 and 8 threads.

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::survreg::{
    Distribution, PredictOptions, PredictType, ResidualKind, SurvregOptions, fit,
};
use epistat_testkit::assert_same_on_pools;
use faer::Mat;

const POOLS: [usize; 5] = [1, 2, 3, 4, 8];

/// Interval-censored times with strata, case weights and a duplicated column.
#[test]
fn survreg_fit_predict_residuals() {
    let n = 1500;
    let mut rng = Stream::seeded(31);
    let (mut lower, mut upper, mut weight, mut stratum, mut a) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let x = rng.normal();
        let t = (0.5 * x + 0.8 * rng.exp1().ln()).exp() * 3.0;
        let k = (t / 1.5).ceil();
        match rng.below(6) {
            0 => {
                lower.push(t);
                upper.push(t);
            }
            1 => {
                lower.push(k * 1.5);
                upper.push(f64::INFINITY);
            }
            2 => {
                lower.push(f64::NEG_INFINITY);
                upper.push(1.5);
            }
            _ => {
                lower.push((k - 1.0) * 1.5);
                upper.push(k * 1.5);
            }
        }
        weight.push(if rng.below(9) == 0 {
            0.0
        } else {
            1.0 + f64::from(u32::try_from(rng.below(3)).unwrap())
        });
        stratum.push(u32::try_from(rng.below(3)).unwrap());
        a.push(x);
    }
    let design = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        _ => a[i],
    });
    let data = SurvData::interval_builder(&lower, &upper)
        .weight(&weight)
        .stratum(&stratum)
        .build()
        .unwrap();
    for distribution in [Distribution::Weibull, Distribution::Gaussian] {
        let options = SurvregOptions {
            distribution,
            ..SurvregOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let f = fit(&data, design.as_ref(), &options).unwrap();
            assert!(f.converged && f.coefficients[2].is_none());
            let mut out: Vec<f64> = f
                .coefficients
                .iter()
                .map(|c| c.unwrap_or(f64::NAN))
                .collect();
            out.extend(&f.log_scale);
            out.extend(f.variance.iter().map(|v| v.unwrap_or(f64::NAN)));
            out.extend([f.loglik, f.loglik_null, f64::from(f.iterations)]);
            out.extend(&f.linear_predictors);
            let p = f
                .predict(
                    design.as_ref(),
                    &PredictOptions {
                        kind: PredictType::Quantile { p: vec![0.25, 0.5] },
                        se_fit: true,
                        strata: Some(stratum.clone()),
                    },
                )
                .unwrap();
            out.extend(p.fit);
            out.extend(p.se.unwrap());
            for kind in [
                ResidualKind::Response,
                ResidualKind::Deviance,
                ResidualKind::Working,
                ResidualKind::LdCase,
                ResidualKind::LdResp,
                ResidualKind::LdShape,
            ] {
                out.extend(
                    f.residuals(&data, design.as_ref(), kind, true)
                        .unwrap()
                        .iter()
                        .map(|v| v.unwrap_or(f64::NAN)),
                );
            }
            out
        });
    }
}
