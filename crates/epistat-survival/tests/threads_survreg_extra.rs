//! The offset, the t distribution, the robust variance, the dfbeta residuals
//! and the term predictions of `survreg` give identical bits in rayon pools of
//! 1, 2, 3, 4 and 8 threads.

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::survreg::{
    DfbetaOptions, Distribution, PredictOptions, PredictType, SurvregOptions, TermsReference, fit,
};
use epistat_testkit::assert_same_on_pools;
use faer::Mat;

const POOLS: [usize; 5] = [1, 2, 3, 4, 8];

#[test]
fn survreg_extra_options() {
    let n = 1200;
    let mut rng = Stream::seeded(47);
    let (mut lower, mut upper, mut weight, mut stratum, mut cluster, mut a, mut offset) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for _ in 0..n {
        let x = rng.normal();
        let t = 4.0 + 1.5 * x + 1.2 * rng.normal();
        match rng.below(6) {
            0 => {
                lower.push(t);
                upper.push(t);
            }
            1 => {
                lower.push(t);
                upper.push(f64::INFINITY);
            }
            2 => {
                lower.push(f64::NEG_INFINITY);
                upper.push(t);
            }
            _ => {
                lower.push(t.floor());
                upper.push(t.floor() + 1.0);
            }
        }
        weight.push(1.0 + f64::from(u32::try_from(rng.below(3)).unwrap()));
        stratum.push(u32::try_from(rng.below(2)).unwrap());
        cluster.push(u32::try_from(rng.below(90)).unwrap());
        offset.push(0.1 * rng.normal());
        a.push(x);
    }
    let design = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { a[i] });
    let data = SurvData::interval_builder(&lower, &upper)
        .weight(&weight)
        .stratum(&stratum)
        .cluster(&cluster)
        .build()
        .unwrap();
    let options = SurvregOptions {
        distribution: Distribution::StudentT,
        df: Some(5.0),
        offset: Some(&offset),
        ..SurvregOptions::default()
    };
    assert_same_on_pools(&POOLS, || {
        let f = fit(&data, design.as_ref(), &options).unwrap();
        assert!(f.converged && f.robust_variance.is_some());
        let mut out: Vec<f64> = f
            .coefficients
            .iter()
            .map(|c| c.unwrap_or(f64::NAN))
            .collect();
        out.extend(&f.log_scale);
        out.extend(f.variance.iter().map(|v| v.unwrap_or(f64::NAN)));
        out.extend(
            f.robust_variance
                .as_ref()
                .unwrap()
                .iter()
                .map(|v| v.unwrap_or(f64::NAN)),
        );
        out.extend([f.loglik, f.loglik_null, f64::from(f.iterations)]);
        out.extend(&f.linear_predictors);
        out.extend(&f.column_means);
        for kind in [
            PredictType::Quantile { p: vec![0.1, 0.9] },
            PredictType::Terms {
                groups: vec![vec![0], vec![1]],
                reference: TermsReference::Sample,
            },
        ] {
            let p = f
                .predict(
                    design.as_ref(),
                    &PredictOptions {
                        kind,
                        se_fit: true,
                        strata: Some(stratum.clone()),
                        offset: Some(offset.clone()),
                        robust: true,
                    },
                )
                .unwrap();
            out.extend(p.fit);
            out.extend(p.se.unwrap());
        }
        for options in [
            DfbetaOptions::default(),
            DfbetaOptions {
                standardise: true,
                weighted: true,
                collapse: Some(&cluster),
            },
        ] {
            out.extend(
                f.dfbeta(&data, design.as_ref(), &options)
                    .unwrap()
                    .values
                    .iter()
                    .map(|v| v.unwrap_or(f64::NAN)),
            );
        }
        out
    });
}
