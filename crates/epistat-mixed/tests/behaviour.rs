//! The closed-form gradients against finite differences, the defaults of the
//! optional inputs, the order of the rows, and the errors.
//!
//! Gradient tolerance. A central difference at step `h` has a truncation
//! error of about `h² |f'''| / 6` and a rounding error of about `ε |f| / h`;
//! with `h = 1e-5` (relative) both are below `1e-8` of the gradient's scale in
//! these data, where `|f|` is a few thousand and the gradient's entries are
//! at least of order 1. The comparison is at `1e-6` of the largest entry.

mod common;

use common::simulate;
use epistat_glm::{Family, Link};
use epistat_linalg::Mat;
use epistat_mixed::{Criterion, Glmm, Lmm, MixedError};

fn largest(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

#[test]
fn the_lmm_gradient_is_the_derivative_of_the_criterion() {
    let d = simulate(400, "gaussian", 1);
    for criterion in [Criterion::Reml, Criterion::Ml] {
        for slope in [None, Some(&d.slope[..])] {
            let model = Lmm::new(criterion);
            let theta: Vec<f64> = if slope.is_some() {
                vec![0.7, 0.2, 0.25]
            } else {
                vec![0.6]
            };
            let at = |t: &[f64]| {
                model
                    .evaluate(
                        d.x.as_ref(),
                        &d.y,
                        &d.cluster,
                        slope,
                        Some(&d.weights),
                        Some(&d.offset),
                        t,
                    )
                    .unwrap()
            };
            let got = at(&theta).gradient;
            let mut fd = vec![0.0; theta.len()];
            for j in 0..theta.len() {
                let h = 1e-5;
                let (mut a, mut b) = (theta.clone(), theta.clone());
                a[j] += h;
                b[j] -= h;
                fd[j] = (at(&a).criterion - at(&b).criterion) / (2.0 * h);
            }
            let scale = largest(&fd).max(1.0);
            for j in 0..theta.len() {
                assert!(
                    (got[j] - fd[j]).abs() <= 1e-6 * scale,
                    "{criterion:?} {j}: {got:?} against {fd:?}"
                );
            }
        }
    }
}

#[test]
fn the_glmm_gradient_is_the_derivative_of_the_likelihood() {
    for (kind, family, link) in [
        ("binomial", Family::Binomial, Link::Logit),
        ("poisson", Family::Poisson, Link::Log),
    ] {
        let d = simulate(300, kind, 2);
        for nodes in [1, 7] {
            let model = Glmm::new(family, link, nodes);
            let par = [0.7, 0.25, 0.45, -0.35];
            let at = |p: &[f64]| {
                model
                    .evaluate(
                        d.x.as_ref(),
                        &d.y,
                        &d.cluster,
                        Some(&d.weights),
                        Some(&d.offset),
                        p[0],
                        &p[1..],
                    )
                    .unwrap()
            };
            let got = at(&par).gradient;
            let mut fd = vec![0.0; par.len()];
            for j in 0..par.len() {
                let h = 1e-5;
                let (mut a, mut b) = (par.to_vec(), par.to_vec());
                a[j] += h;
                b[j] -= h;
                fd[j] = -2.0 * (at(&a).log_likelihood - at(&b).log_likelihood) / (2.0 * h);
            }
            let scale = largest(&fd).max(1.0);
            for j in 0..par.len() {
                assert!(
                    (got[j] - fd[j]).abs() <= 1e-6 * scale,
                    "{kind} {nodes} {j}: {got:?} against {fd:?}"
                );
            }
        }
    }
}

#[test]
fn absent_weights_and_offset_are_ones_and_zeros() {
    let d = simulate(200, "gaussian", 3);
    let (ones, zeros) = (vec![1.0; d.y.len()], vec![0.0; d.y.len()]);
    let model = Lmm::new(Criterion::Reml);
    let a = model
        .fit(d.x.as_ref(), &d.y, &d.cluster, Some(&d.slope), None, None)
        .unwrap();
    let b = model
        .fit(
            d.x.as_ref(),
            &d.y,
            &d.cluster,
            Some(&d.slope),
            Some(&ones),
            Some(&zeros),
        )
        .unwrap();
    assert_eq!(a.beta, b.beta);
    assert_eq!(a.theta, b.theta);
    assert_eq!(a.criterion, b.criterion);
    let d = simulate(200, "binomial", 3);
    let (ones, zeros) = (vec![1.0; d.y.len()], vec![0.0; d.y.len()]);
    let model = Glmm::new(Family::Binomial, Link::Logit, 3);
    let a = model
        .fit(d.x.as_ref(), &d.y, &d.cluster, None, None)
        .unwrap();
    let b = model
        .fit(d.x.as_ref(), &d.y, &d.cluster, Some(&ones), Some(&zeros))
        .unwrap();
    assert_eq!(a.beta, b.beta);
    assert_eq!(a.theta, b.theta);
    assert_eq!(a.log_likelihood, b.log_likelihood);
}

#[test]
fn the_order_of_the_rows_does_not_matter() {
    let d = simulate(300, "poisson", 4);
    let n = d.y.len();
    let order: Vec<usize> = (0..n).rev().collect();
    let pick = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let x = Mat::from_fn(n, 3, |i, j| d.x[(order[i], j)]);
    let cluster: Vec<u32> = order.iter().map(|&i| d.cluster[i]).collect();
    let model = Glmm::new(Family::Poisson, Link::Log, 5);
    let a = model
        .fit(
            d.x.as_ref(),
            &d.y,
            &d.cluster,
            Some(&d.weights),
            Some(&d.offset),
        )
        .unwrap();
    let b = model
        .fit(
            x.as_ref(),
            &pick(&d.y),
            &cluster,
            Some(&pick(&d.weights)),
            Some(&pick(&d.offset)),
        )
        .unwrap();
    assert_eq!(a.groups, b.groups);
    for (u, v) in a
        .beta
        .iter()
        .chain(&a.modes)
        .zip(b.beta.iter().chain(&b.modes))
    {
        assert!((u - v).abs() < 1e-8 * (1.0 + v.abs()));
    }
    let d = simulate(300, "gaussian", 4);
    let x = Mat::from_fn(d.y.len(), 3, |i, j| d.x[(d.y.len() - 1 - i, j)]);
    let rev = |v: &[f64]| v.iter().rev().copied().collect::<Vec<f64>>();
    let cluster: Vec<u32> = d.cluster.iter().rev().copied().collect();
    let model = Lmm::new(Criterion::Ml);
    let a = model
        .fit(
            d.x.as_ref(),
            &d.y,
            &d.cluster,
            Some(&d.slope),
            Some(&d.weights),
            None,
        )
        .unwrap();
    let b = model
        .fit(
            x.as_ref(),
            &rev(&d.y),
            &cluster,
            Some(&rev(&d.slope)),
            Some(&rev(&d.weights)),
            None,
        )
        .unwrap();
    for (u, v) in a
        .beta
        .iter()
        .chain(&a.theta)
        .zip(b.beta.iter().chain(&b.theta))
    {
        assert!((u - v).abs() < 1e-8 * (1.0 + v.abs()));
    }
}

#[test]
fn a_model_without_cluster_variation_is_singular() {
    // Every cluster has the same mean, so the random intercept vanishes.
    let n = 400;
    let x = Mat::from_fn(n, 1, |_, _| 1.0);
    let cluster: Vec<u32> = (0..n).map(|i| u32::try_from(i / 4).unwrap()).collect();
    let y: Vec<f64> = (0..n).map(|i| [1.0, -1.0, 0.5, -0.5][i % 4]).collect();
    let fit = Lmm::new(Criterion::Reml)
        .fit(x.as_ref(), &y, &cluster, None, None, None)
        .unwrap();
    assert!(fit.singular);
    assert_eq!(fit.theta, [0.0]);
    assert_eq!(fit.icc, Some(0.0));
    assert!(fit.modes.iter().all(|&b| b == 0.0));
    let yb: Vec<f64> = (0..n).map(|i| [1.0, 0.0, 1.0, 0.0][i % 4]).collect();
    let fit = Glmm::new(Family::Binomial, Link::Logit, 1)
        .fit(x.as_ref(), &yb, &cluster, None, None)
        .unwrap();
    assert!(fit.singular);
    assert_eq!(fit.theta, 0.0);
    assert_eq!(fit.theta_se, None);
    assert!(fit.beta_covariance.is_some());
}

#[test]
fn invalid_input_is_an_error() {
    let d = simulate(50, "binomial", 5);
    let n = d.y.len();
    let lmm = Lmm::new(Criterion::Reml);
    let glmm = Glmm::new(Family::Binomial, Link::Logit, 1);
    assert!(matches!(
        lmm.fit(d.x.as_ref(), &d.y[1..], &d.cluster, None, None, None),
        Err(MixedError::Length { what: "y", .. })
    ));
    let mut w = vec![1.0; n];
    w[3] = 0.0;
    assert!(matches!(
        lmm.fit(d.x.as_ref(), &d.y, &d.cluster, None, Some(&w), None),
        Err(MixedError::Weight { row: 3, .. })
    ));
    w[3] = -1.0;
    assert!(matches!(
        glmm.fit(d.x.as_ref(), &d.y, &d.cluster, Some(&w), None),
        Err(MixedError::Weight { row: 3, .. })
    ));
    let mut y = d.y.clone();
    y[2] = f64::NAN;
    assert!(matches!(
        lmm.fit(d.x.as_ref(), &y, &d.cluster, None, None, None),
        Err(MixedError::NotFinite { what: "y", row: 2 })
    ));
    y[2] = 1.5;
    assert!(matches!(
        glmm.fit(d.x.as_ref(), &y, &d.cluster, None, None),
        Err(MixedError::Response { row: 2, .. })
    ));
    let doubled = Mat::from_fn(n, 4, |i, j| {
        if j == 3 {
            2.0 * d.x[(i, 1)]
        } else {
            d.x[(i, j)]
        }
    });
    assert_eq!(
        lmm.fit(doubled.as_ref(), &d.y, &d.cluster, None, None, None)
            .unwrap_err(),
        MixedError::RankDeficient
    );
    assert!(matches!(
        Glmm::new(Family::Gaussian, Link::Identity, 1).fit(
            d.x.as_ref(),
            &d.y,
            &d.cluster,
            None,
            None
        ),
        Err(MixedError::Family)
    ));
    assert!(matches!(
        Glmm::new(Family::Binomial, Link::Probit, 1).fit(
            d.x.as_ref(),
            &d.y,
            &d.cluster,
            None,
            None
        ),
        Err(MixedError::Family)
    ));
    for nodes in [0, 101] {
        assert_eq!(
            Glmm::new(Family::Binomial, Link::Logit, nodes)
                .fit(d.x.as_ref(), &d.y, &d.cluster, None, None)
                .unwrap_err(),
            MixedError::Nodes(nodes)
        );
    }
    let small = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
    assert!(matches!(
        lmm.fit(
            small.as_ref(),
            &[1.0, 2.0, 3.0],
            &[1, 1, 2],
            None,
            None,
            None
        ),
        Err(MixedError::Size(_))
    ));
    assert_eq!(
        lmm.evaluate(
            d.x.as_ref(),
            &d.y,
            &d.cluster,
            Some(&d.slope),
            None,
            None,
            &[1.0]
        )
        .unwrap_err(),
        MixedError::Parameters(3)
    );
    assert_eq!(
        glmm.evaluate(
            d.x.as_ref(),
            &d.y,
            &d.cluster,
            None,
            None,
            f64::NAN,
            &[0.0; 3]
        )
        .unwrap_err(),
        MixedError::Parameters(4)
    );
    let mut bad = lmm;
    bad.tolerance = 0.0;
    assert!(matches!(
        bad.fit(d.x.as_ref(), &d.y, &d.cluster, None, None, None),
        Err(MixedError::Control(_))
    ));
}
