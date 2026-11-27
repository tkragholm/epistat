//! The refusals of the offset, the t distribution, the robust variance, the
//! dfbeta residuals and the term predictions of `survreg`, and the cases the
//! reference comparison does not reach.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::survreg::{
    DfbetaOptions, Distribution, SurvregOptions, SurvregPredictOptions, SurvregPredictType,
    TermsReference, fit,
};

fn message<T>(result: epistat_survival::Result<T>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error.to_string(),
    }
}

fn data() -> (Vec<f64>, Vec<i32>, Vec<u32>, Mat<f64>) {
    let time: Vec<f64> = (1..=40).map(|i| f64::from(i) * 0.37 + 0.5).collect();
    let status: Vec<i32> = (0..40).map(|i| i32::from(i % 4 != 0)).collect();
    let cluster: Vec<u32> = (0..40_u32).map(|i| i / 4).collect();
    let x = Mat::from_fn(40, 2, |i, j| {
        if j == 0 {
            1.0
        } else {
            ((i * 7) % 11) as f64 / 3.0
        }
    });
    (time, status, cluster, x)
}

#[test]
fn inputs_are_refused() {
    let (time, status, _, x) = data();
    let d = SurvData::builder(&time, &status).build().unwrap();
    let t = |df| SurvregOptions {
        distribution: Distribution::StudentT,
        df,
        ..SurvregOptions::default()
    };
    assert!(message(fit(&d, x.as_ref(), &t(None))).contains("degrees of freedom"));
    assert!(message(fit(&d, x.as_ref(), &t(Some(2.0)))).contains("greater than 2"));
    assert!(message(fit(&d, x.as_ref(), &t(Some(f64::NAN)))).contains("greater than 2"));
    let weibull_df = SurvregOptions {
        df: Some(4.0),
        ..SurvregOptions::default()
    };
    assert!(message(fit(&d, x.as_ref(), &weibull_df)).contains("t distribution only"));
    let short = [0.0; 3];
    let refused = SurvData::builder(&time, &status).offset(&short).build();
    assert!(message(refused).contains("offset"));
    let mut bad = vec![0.0; 40];
    bad[5] = f64::INFINITY;
    let refused = SurvData::builder(&time, &status).offset(&bad).build();
    assert!(message(refused).contains("not finite"));
}

#[test]
fn a_constant_offset_moves_the_intercept_and_nothing_else() {
    let (time, status, _, x) = data();
    let d = SurvData::builder(&time, &status).build().unwrap();
    let plain = fit(&d, x.as_ref(), &SurvregOptions::default()).unwrap();
    let shift = vec![0.7; 40];
    let shifted_data = SurvData::builder(&time, &status)
        .offset(&shift)
        .build()
        .unwrap();
    let shifted = fit(&shifted_data, x.as_ref(), &SurvregOptions::default()).unwrap();
    let (a, b) = (&plain.coefficients, &shifted.coefficients);
    assert!((a[0].unwrap() - 0.7 - b[0].unwrap()).abs() < 1e-7);
    assert!((a[1].unwrap() - b[1].unwrap()).abs() < 1e-7);
    assert!((plain.loglik - shifted.loglik).abs() < 1e-8);
    for (p, s) in plain
        .linear_predictors
        .iter()
        .zip(&shifted.linear_predictors)
    {
        assert!((p - s).abs() < 1e-7);
    }
}

#[test]
fn the_robust_variance_needs_robust_or_clusters() {
    let (time, status, cluster, x) = data();
    let plain = SurvData::builder(&time, &status).build().unwrap();
    let f = fit(&plain, x.as_ref(), &SurvregOptions::default()).unwrap();
    assert!(f.robust_variance.is_none() && f.n_clusters.is_none());
    let robust = fit(
        &plain,
        x.as_ref(),
        &SurvregOptions {
            robust: true,
            ..SurvregOptions::default()
        },
    )
    .unwrap();
    assert_eq!(robust.n_clusters, Some(40));
    // Every row its own cluster is the same as no clusters.
    let own: Vec<u32> = (0..40).collect();
    let singletons = SurvData::builder(&time, &status)
        .cluster(&own)
        .build()
        .unwrap();
    let by_cluster = fit(&singletons, x.as_ref(), &SurvregOptions::default()).unwrap();
    assert_eq!(by_cluster.robust_variance, robust.robust_variance);
    // Clusters alone switch it on, and change it.
    let grouped = SurvData::builder(&time, &status)
        .cluster(&cluster)
        .build()
        .unwrap();
    let g = fit(&grouped, x.as_ref(), &SurvregOptions::default()).unwrap();
    assert_eq!(g.n_clusters, Some(10));
    assert_ne!(g.robust_variance, robust.robust_variance);
    assert_eq!(g.variance, robust.variance);
    // Standard errors can be asked of either variance.
    let ask = |robust| {
        g.predict(
            x.as_ref(),
            &SurvregPredictOptions {
                kind: SurvregPredictType::LinearPredictor,
                se_fit: true,
                robust,
                ..SurvregPredictOptions::default()
            },
        )
        .unwrap()
        .se
        .unwrap()
    };
    assert_ne!(ask(true), ask(false));
    assert!(
        message(f.predict(
            x.as_ref(),
            &SurvregPredictOptions {
                se_fit: true,
                robust: true,
                ..SurvregPredictOptions::default()
            },
        ))
        .contains("robust")
    );
}

#[test]
fn dfbeta_collapsed_by_the_cluster_of_the_fit_sums_to_the_robust_meat() {
    // The robust variance is the cross-product of the weighted dfbeta rows
    // collapsed by the cluster, `crossprod(dfbeta)` of survreg.
    let (time, status, cluster, x) = data();
    let d = SurvData::builder(&time, &status)
        .cluster(&cluster)
        .build()
        .unwrap();
    let f = fit(&d, x.as_ref(), &SurvregOptions::default()).unwrap();
    let collapsed = f
        .dfbeta(
            &d,
            x.as_ref(),
            &DfbetaOptions {
                weighted: true,
                collapse: Some(&cluster),
                ..DfbetaOptions::default()
            },
        )
        .unwrap();
    let q = collapsed.width;
    let robust = f.robust_variance.as_ref().unwrap();
    for a in 0..q {
        for b in 0..q {
            let mut sum = 0.0;
            for g in 0..collapsed.n_rows {
                sum += collapsed.values[g * q + a].unwrap() * collapsed.values[g * q + b].unwrap();
            }
            let want = robust[a * q + b].unwrap();
            assert!(
                (sum - want).abs() <= 1e-12 * want.abs().max(1e-12),
                "{a} {b}"
            );
        }
    }
    let wrong = [0_u32; 3];
    assert!(
        message(f.dfbeta(
            &d,
            x.as_ref(),
            &DfbetaOptions {
                collapse: Some(&wrong),
                ..DfbetaOptions::default()
            },
        ))
        .contains("collapse")
    );
}

#[test]
fn terms_sum_to_the_centred_linear_predictor() {
    let (time, status, _, x) = data();
    let d = SurvData::builder(&time, &status).build().unwrap();
    let f = fit(&d, x.as_ref(), &SurvregOptions::default()).unwrap();
    let ask = |groups: Vec<Vec<usize>>, reference| {
        f.predict(
            x.as_ref(),
            &SurvregPredictOptions {
                kind: SurvregPredictType::Terms { groups, reference },
                se_fit: true,
                ..SurvregPredictOptions::default()
            },
        )
        .unwrap()
    };
    let sample = ask(vec![vec![0], vec![1]], TermsReference::Sample);
    let zero = ask(vec![vec![0, 1]], TermsReference::Zero);
    let (b0, b1) = (f.coefficients[0].unwrap(), f.coefficients[1].unwrap());
    for i in 0..40 {
        // The intercept is centred on its mean of 1 and gives nothing.
        assert!(sample.fit[2 * i].abs() < 1e-15);
        let centred = (x[(i, 1)] - f.column_means[1]) * b1;
        assert!((sample.fit[2 * i + 1] - centred).abs() < 1e-13);
        assert!((zero.fit[i] - (b0 + x[(i, 1)] * b1)).abs() < 1e-13);
    }
    // The standard error of one term is |x - mean| times that of its
    // coefficient. The variance has the two coefficients and the log scale.
    let se_b1 = f.variance[3 + 1].unwrap().sqrt();
    for i in 0..40 {
        let want = (x[(i, 1)] - f.column_means[1]).abs() * se_b1;
        assert!((sample.se.as_ref().unwrap()[2 * i + 1] - want).abs() < 1e-12);
    }
    assert!(
        message(f.predict(
            x.as_ref(),
            &SurvregPredictOptions {
                kind: SurvregPredictType::Terms {
                    groups: vec![vec![2]],
                    reference: TermsReference::Sample,
                },
                ..SurvregPredictOptions::default()
            },
        ))
        .contains("not a column")
    );
}
