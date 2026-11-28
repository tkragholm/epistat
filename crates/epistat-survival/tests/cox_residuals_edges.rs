//! Refusals and conventions of `cox::residuals` and `cox::zph` that R has no
//! counterpart for: rows of weight zero, invalid options, and fits without an
//! estimate.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{
    Collapse, CoxModelFit, CoxOptions, CoxResidualKind, ResidualOptions, ZphOptions, ZphTransform,
    fit, residuals, zph,
};

const STOP: [f64; 12] = [1., 2., 2., 3., 4., 4., 5., 6., 7., 7., 8., 9.];
const STATUS: [i32; 12] = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1];
const X: [f64; 12] = [
    0.3, -0.5, 1.2, 0.1, 0.8, -1.1, 0.4, 0.0, 0.9, -0.2, 0.6, -0.7,
];

fn x_matrix(values: &[f64]) -> Mat<f64> {
    let n = values.len();
    Mat::from_fn(n, 2, |i, j| if j == 0 { values[i] } else { (i % 3) as f64 })
}

fn model(data: &SurvData<'_>, x: &Mat<f64>) -> CoxModelFit {
    fit(data, x.as_ref(), &CoxOptions::default()).unwrap()
}

fn options(kind: CoxResidualKind) -> ResidualOptions<'static> {
    ResidualOptions::new(kind, CoxOptions::default())
}

#[test]
fn rows_of_weight_zero_are_dropped() {
    // Rows appended with weight 0 change nothing for the rows with weight.
    let weight: Vec<f64> = (0..15).map(|i| if i < 12 { 1.0 } else { 0.0 }).collect();
    let mut stop = STOP.to_vec();
    let mut status = STATUS.to_vec();
    let mut values = X.to_vec();
    for (t, e, v) in [(3.0, 1, 5.0), (6.0, 1, -4.0), (2.0, 0, 2.0)] {
        stop.push(t);
        status.push(e);
        values.push(v);
    }
    let small = SurvData::builder(&STOP, &STATUS).build().unwrap();
    let big = SurvData::builder(&stop, &status)
        .weight(&weight)
        .build()
        .unwrap();
    let (xs, xb) = (x_matrix(&X), x_matrix(&values));
    let (ms, mb) = (model(&small, &xs), model(&big, &xb));
    assert_eq!(mb.n_rows, 15);
    assert_eq!(mb.n_used, 12);
    for kind in [
        CoxResidualKind::Martingale,
        CoxResidualKind::Deviance,
        CoxResidualKind::Score,
        CoxResidualKind::Schoenfeld,
        CoxResidualKind::ScaledSchoenfeld,
        CoxResidualKind::Dfbeta,
        CoxResidualKind::Dfbetas,
    ] {
        let a = residuals(&small, xs.as_ref(), &ms, &options(kind)).unwrap();
        let b = residuals(&big, xb.as_ref(), &mb, &options(kind)).unwrap();
        assert_eq!((b.n_rows, b.n_used), (15, 12), "{kind:?}");
        assert_eq!(a.rows, b.rows, "{kind:?}");
        for (ca, cb) in a.columns.iter().zip(&b.columns) {
            for (u, v) in ca.as_ref().unwrap().iter().zip(cb.as_ref().unwrap()) {
                assert!(
                    (u - v).abs() <= 1e-12 * u.abs().max(1.0),
                    "{kind:?}: {u} {v}"
                );
            }
        }
    }
    let a = zph(&small, xs.as_ref(), &ms, &ZphOptions::default()).unwrap();
    let b = zph(&big, xb.as_ref(), &mb, &ZphOptions::default()).unwrap();
    assert_eq!(a.time, b.time);
    assert_eq!(a.n_used, b.n_used);
    let (ta, tb) = (a.global.unwrap(), b.global.unwrap());
    assert!((ta.statistic - tb.statistic).abs() <= 1e-10 * ta.statistic.max(1.0));
}

#[test]
fn collapsing_adds_up() {
    let cluster: Vec<u32> = (0..12u32).map(|i| i % 4 + 10).collect();
    let data = SurvData::builder(&STOP, &STATUS)
        .cluster(&cluster)
        .build()
        .unwrap();
    let x = x_matrix(&X);
    let fitted = model(&data, &x);
    let rows = residuals(
        &data,
        x.as_ref(),
        &fitted,
        &options(CoxResidualKind::Martingale),
    )
    .unwrap();
    let grouped = residuals(
        &data,
        x.as_ref(),
        &fitted,
        &ResidualOptions {
            collapse: Collapse::Cluster,
            ..options(CoxResidualKind::Martingale)
        },
    )
    .unwrap();
    // The groups come out under the caller's codes.
    assert_eq!(grouped.groups, [10, 11, 12, 13]);
    let by_hand = |g: usize| -> f64 {
        rows.columns[0]
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i % 4 == g)
            .map(|(_, v)| v)
            .sum()
    };
    for g in 0..4 {
        let got = grouped.columns[0].as_ref().unwrap()[g];
        assert!((got - by_hand(g)).abs() < 1e-14, "{got} {}", by_hand(g));
    }
    // The martingale residuals of an Efron or Breslow fit add up to the number
    // of events less the expected number, which is 0 at the maximum.
    let total: f64 = rows.columns[0].as_ref().unwrap().iter().sum();
    assert!(total.abs() < 1e-9, "{total}");
}

#[test]
fn invalid_options_are_refused() {
    let data = SurvData::builder(&STOP, &STATUS).build().unwrap();
    let x = x_matrix(&X);
    let fitted = model(&data, &x);
    let run = |opts: &ResidualOptions<'_>| residuals(&data, x.as_ref(), &fitted, opts);
    for kind in [
        CoxResidualKind::Schoenfeld,
        CoxResidualKind::ScaledSchoenfeld,
    ] {
        let codes = [0u32; 12];
        let opts = ResidualOptions {
            collapse: Collapse::Codes(&codes),
            ..options(kind)
        };
        assert!(run(&opts).is_err(), "{kind:?} collapsed");
    }
    for collapse in [Collapse::Cluster, Collapse::Id] {
        let opts = ResidualOptions {
            collapse,
            ..options(CoxResidualKind::Martingale)
        };
        assert!(run(&opts).is_err(), "{collapse:?} without the column");
    }
    let short = [0u32; 3];
    let opts = ResidualOptions {
        collapse: Collapse::Codes(&short),
        ..options(CoxResidualKind::Score)
    };
    assert!(run(&opts).is_err());
    let opts = ResidualOptions {
        weighted: Some(true),
        ..options(CoxResidualKind::Deviance)
    };
    assert!(run(&opts).is_err());
    // A fit of other data.
    let other = SurvData::builder(&STOP[..8], &STATUS[..8]).build().unwrap();
    let x8 = x_matrix(&X[..8]);
    assert!(
        residuals(
            &other,
            x8.as_ref(),
            &fitted,
            &options(CoxResidualKind::Martingale)
        )
        .is_err()
    );
    assert!(zph(&other, x8.as_ref(), &fitted, &ZphOptions::default()).is_err());
}

#[test]
fn a_fit_without_an_estimate_has_no_residuals() {
    let status = [0; 12];
    let data = SurvData::builder(&STOP, &status).build().unwrap();
    let x = x_matrix(&X);
    let fitted = model(&data, &x);
    assert!(fitted.coefficients.iter().all(Option::is_none));
    assert!(
        residuals(
            &data,
            x.as_ref(),
            &fitted,
            &options(CoxResidualKind::Martingale)
        )
        .is_err()
    );
    assert!(zph(&data, x.as_ref(), &fitted, &ZphOptions::default()).is_err());
}

#[test]
fn zph_refuses_invalid_terms_and_a_log_of_zero() {
    let data = SurvData::builder(&STOP, &STATUS).build().unwrap();
    let x = x_matrix(&X);
    let fitted = model(&data, &x);
    for groups in [
        vec![vec![0], vec![0, 1]],
        vec![vec![0], vec![]],
        vec![vec![0], vec![2]],
        vec![vec![0, 0]],
    ] {
        let opts = ZphOptions {
            groups: Some(groups.clone()),
            ..ZphOptions::default()
        };
        assert!(
            zph(&data, x.as_ref(), &fitted, &opts).is_err(),
            "{groups:?}"
        );
    }
    let zero = [0.0, 2., 2., 3., 4., 4., 5., 6., 7., 7., 8., 9.];
    let status = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1];
    let start = [-1.0; 12];
    let with_zero = SurvData::builder(&zero, &status)
        .start(&start)
        .build()
        .unwrap();
    let fitted = model(&with_zero, &x);
    let opts = ZphOptions {
        transform: ZphTransform::Log,
        ..ZphOptions::default()
    };
    assert!(zph(&with_zero, x.as_ref(), &fitted, &opts).is_err());
    let opts = ZphOptions {
        transform: ZphTransform::Identity,
        ..ZphOptions::default()
    };
    assert!(zph(&with_zero, x.as_ref(), &fitted, &opts).is_ok());
}
