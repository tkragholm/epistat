//! Properties of the time-transform expansion that need no reference: an
//! identity transform leaves the model unchanged, the forms of a transform
//! agree, and the refusals.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, TimeTransform, TtColumn, TtOptions, fit, fit_tt};

struct Rows {
    stop: Vec<f64>,
    status: Vec<i32>,
    start: Vec<f64>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    x: Mat<f64>,
}

fn data() -> Rows {
    let stop = vec![3.0, 5.0, 5.0, 8.0, 2.0, 9.0, 6.0, 7.0, 4.0, 8.0, 10.0, 1.0];
    let status = vec![1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1];
    let start = vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0];
    let weight = vec![1.0, 2.0, 1.0, 1.5, 1.0, 1.0, 0.0, 1.0, 1.0, 2.5, 1.0, 1.0];
    let stratum = vec![0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let x = Mat::from_fn(12, 2, |i, j| {
        if j == 0 {
            [
                0.3, -1.2, 0.8, 1.7, -0.4, 0.1, 0.9, -0.7, 1.1, -1.5, 0.5, 0.2,
            ][i]
        } else {
            f64::from(u8::from(i % 3 == 0))
        }
    });
    Rows {
        stop,
        status,
        start,
        weight,
        stratum,
        x,
    }
}

#[test]
fn an_identity_transform_is_the_plain_fit() {
    let Rows {
        stop,
        status,
        start,
        weight,
        stratum,
        x,
    } = data();
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .build()
        .unwrap();
    let identity = |v: f64, _t: f64| v;
    let transforms = [TtColumn {
        column: 0,
        transform: TimeTransform::Function(&identity),
    }];
    let tt = fit_tt(&data, x.as_ref(), &transforms, &TtOptions::default()).unwrap();
    let plain = fit(&data, x.as_ref(), &CoxOptions::default()).unwrap();
    // The two sum the same terms in another order.
    for (a, b) in tt.fit.coefficients.iter().zip(&plain.coefficients) {
        assert!((a.unwrap() - b.unwrap()).abs() < 1e-12, "{a:?} {b:?}");
    }
    for (a, b) in tt.fit.standard_errors.iter().zip(&plain.standard_errors) {
        assert!((a.unwrap() - b.unwrap()).abs() < 1e-12);
    }
    assert_eq!(tt.fit.n_rows, 12);
    assert_eq!(tt.fit.n_used, 11);
    assert!(tt.n_expanded > 11);
    assert!((tt.fit.n_events - plain.n_events).abs() < 1e-12);
}

#[test]
fn the_named_transforms_agree_with_their_closures() {
    let Rows {
        stop,
        status,
        weight,
        stratum,
        x,
        ..
    } = data();
    let data = SurvData::builder(&stop, &status)
        .weight(&weight)
        .stratum(&stratum)
        .build()
        .unwrap();
    let product = |v: f64, t: f64| v * t;
    let after_four = |v: f64, t: f64| if t > 4.0 { v } else { 0.0 };
    let set = |_t: f64, v: &[f64], _w: &[f64], out: &mut [f64]| {
        out.copy_from_slice(v);
        for o in out {
            *o *= 2.0;
        }
    };
    let doubled = |v: f64, _t: f64| 2.0 * v;
    for (named, closure) in [
        (TimeTransform::Product, TimeTransform::Function(&product)),
        (
            TimeTransform::Step { cut: 4.0 },
            TimeTransform::Function(&after_four),
        ),
        (
            TimeTransform::RiskSet(&set),
            TimeTransform::Function(&doubled),
        ),
    ] {
        let a = fit_tt(
            &data,
            x.as_ref(),
            &[TtColumn {
                column: 0,
                transform: named,
            }],
            &TtOptions::default(),
        )
        .unwrap();
        let b = fit_tt(
            &data,
            x.as_ref(),
            &[TtColumn {
                column: 0,
                transform: closure,
            }],
            &TtOptions::default(),
        )
        .unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn no_events_gives_the_fit_without_an_expansion() {
    let stop = [1.0, 2.0, 3.0];
    let status = [0, 0, 0];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let x = Mat::from_fn(3, 1, |i, _| i as f64);
    let tt = fit_tt(
        &data,
        x.as_ref(),
        &[TtColumn {
            column: 0,
            transform: TimeTransform::Product,
        }],
        &TtOptions::default(),
    )
    .unwrap();
    assert_eq!(tt.n_expanded, 0);
    assert_eq!(tt.n_risk_sets, 0);
    assert!(tt.fit.coefficients[0].is_none());
}

#[test]
fn refusals() {
    let Rows {
        stop, status, x, ..
    } = data();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let col = |column, transform| TtColumn { column, transform };
    let opts = TtOptions::default();
    let msg = |r: epistat_survival::Result<_>| match r {
        Err(e) => e.to_string(),
        Ok(_) => panic!("accepted"),
    };
    assert!(msg(fit_tt(&data, x.as_ref(), &[], &opts)).contains("transforms is empty"));
    assert!(
        msg(fit_tt(
            &data,
            x.as_ref(),
            &[col(2, TimeTransform::Product)],
            &opts
        ))
        .contains("outside the 2 columns")
    );
    assert!(
        msg(fit_tt(
            &data,
            x.as_ref(),
            &[
                col(0, TimeTransform::Product),
                col(0, TimeTransform::LogProduct)
            ],
            &opts
        ))
        .contains("named twice")
    );
    let limited = TtOptions {
        max_expanded_rows: Some(10),
        ..TtOptions::default()
    };
    assert!(
        msg(fit_tt(
            &data,
            x.as_ref(),
            &[col(0, TimeTransform::Product)],
            &limited
        ))
        .contains("max_expanded_rows")
    );
    // ln(0) is not finite.
    let zero = [0.0, 1.0, 2.0];
    let ev = [1, 1, 1];
    let d0 = SurvData::builder(&zero, &ev).build().unwrap();
    let x0 = Mat::from_fn(3, 1, |i, _| i as f64);
    assert!(
        msg(fit_tt(
            &d0,
            x0.as_ref(),
            &[col(0, TimeTransform::LogProduct)],
            &opts
        ))
        .contains("not finite at time 0")
    );
}
