//! `attributable_logistic`: what it refuses and the results that are not
//! numbers, on small fits.

use epistat_glm::attributable::{AfOptions, Design, IntervalScale, attributable_logistic};
use epistat_glm::{Family, Glm, Link};
use epistat_linalg::Mat;

/// Exposure `x` and confounder `z` over 400 rows with a fixed pattern.
fn data(protective: bool) -> (Mat<f64>, Mat<f64>, Vec<f64>) {
    let n = 400;
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => f64::from(u8::from(i % 3 == 0)),
        _ => ((i * 7919) % 101) as f64 / 50.0 - 1.0,
    });
    let x0 = Mat::from_fn(n, 3, |i, j| if j == 1 { 0.0 } else { x[(i, j)] });
    let y = (0..n)
        .map(|i| {
            let e = x[(i, 1)];
            let z = x[(i, 2)];
            let eta = -0.5 + if protective { -0.9 } else { 0.9 } * e + 0.4 * z;
            // A fixed quasi-random draw.
            let u = ((i * 2_654_435_761) % 1000) as f64 / 1000.0;
            f64::from(u8::from(u < 1.0 / (1.0 + (-eta).exp())))
        })
        .collect();
    (x, x0, y)
}

fn fitted(x: &Mat<f64>, y: &[f64]) -> epistat_glm::Fit {
    Glm::new(Family::Binomial, Link::Logit)
        .fit(x.as_ref(), y, None, None, None)
        .unwrap()
}

#[test]
fn a_harmful_exposure_gives_a_positive_fraction_and_a_protective_one_a_negative() {
    for (protective, sign) in [(false, 1.0), (true, -1.0)] {
        let (x, x0, y) = data(protective);
        let fit = fitted(&x, &y);
        let a = attributable_logistic(&fit, x.as_ref(), x0.as_ref(), &y, &AfOptions::default())
            .unwrap();
        assert!(a.af.unwrap() * sign > 0.0);
        assert!(a.se.unwrap() > 0.0);
        assert!(a.lower.unwrap() < a.af.unwrap() && a.af.unwrap() < a.upper.unwrap());
    }
}

#[test]
fn nobody_exposed_gives_zero_and_no_events_gives_nothing() {
    let (x, _, y) = data(false);
    let fit = fitted(&x, &y);
    let same =
        attributable_logistic(&fit, x.as_ref(), x.as_ref(), &y, &AfOptions::default()).unwrap();
    assert!(same.af.unwrap().abs() < 1e-12);
    let zeros = vec![0.0; y.len()];
    let none_fit = Glm::new(Family::Binomial, Link::Logit)
        .fit(x.as_ref(), &y, Some(&zeros), None, None)
        .unwrap();
    let a = attributable_logistic(
        &none_fit,
        x.as_ref(),
        x.as_ref(),
        &y,
        &AfOptions {
            weights: Some(&zeros),
            ..AfOptions::default()
        },
    )
    .unwrap();
    assert!(a.af.is_none() && a.se.is_none());
    assert_eq!(a.n_cases, 0);
}

#[test]
fn an_exposure_effect_of_one_sided_size_keeps_the_log_limits_below_one() {
    let (x, x0, y) = data(false);
    let fit = fitted(&x, &y);
    let a = attributable_logistic(
        &fit,
        x.as_ref(),
        x0.as_ref(),
        &y,
        &AfOptions {
            design: Design::CaseControl,
            scale: IntervalScale::LogComplement,
            level: 0.999_999,
            ..AfOptions::default()
        },
    )
    .unwrap();
    assert!(a.upper.unwrap() < 1.0);
}

#[test]
fn unusable_input_is_refused() {
    let (x, x0, y) = data(false);
    let fit = fitted(&x, &y);
    let run = |x: &Mat<f64>, x0: &Mat<f64>, y: &[f64], opts: &AfOptions<'_>| {
        attributable_logistic(&fit, x.as_ref(), x0.as_ref(), y, opts)
    };
    let d = AfOptions::default();
    assert!(run(&x, &x0, &y[..399], &d).is_err());
    let narrow = Mat::from_fn(400, 2, |i, j| x0[(i, j)]);
    assert!(run(&x, &narrow, &y, &d).is_err());
    let mut bad_y = y.clone();
    bad_y[3] = 0.5;
    assert!(run(&x, &x0, &bad_y, &d).is_err());
    for level in [0.0, 1.0, f64::NAN] {
        assert!(run(&x, &x0, &y, &AfOptions { level, ..d }).is_err());
    }
    let negative = vec![-1.0; 400];
    assert!(
        run(
            &x,
            &x0,
            &y,
            &AfOptions {
                weights: Some(&negative),
                ..d
            }
        )
        .is_err()
    );
    assert!(
        run(
            &x,
            &x0,
            &y,
            &AfOptions {
                weights: Some(&negative[..10]),
                ..d
            }
        )
        .is_err()
    );
    assert!(
        run(
            &x,
            &x0,
            &y,
            &AfOptions {
                cluster: Some(&[1, 2]),
                ..d
            }
        )
        .is_err()
    );
    // A fit with another link is not logistic.
    let probit = Glm::new(Family::Binomial, Link::Probit)
        .fit(x.as_ref(), &y, None, None, None)
        .unwrap();
    assert!(attributable_logistic(&probit, x.as_ref(), x0.as_ref(), &y, &d).is_err());
}

#[test]
fn one_cluster_has_no_standard_error() {
    let (x, x0, y) = data(false);
    let fit = fitted(&x, &y);
    let one = vec![1_u32; 400];
    let a = attributable_logistic(
        &fit,
        x.as_ref(),
        x0.as_ref(),
        &y,
        &AfOptions {
            cluster: Some(&one),
            ..AfOptions::default()
        },
    )
    .unwrap();
    assert!(a.af.is_some() && a.se.is_none() && a.lower.is_none());
}
