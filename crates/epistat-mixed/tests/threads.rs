//! The fits give the same bits on 1, 2, 8 and again 1 thread, on data with
//! more clusters than one block of the parallel sums.

mod common;

use common::{simulate, view};
use epistat_core::Covariance;
use epistat_glm::{Family, Link};
use epistat_mixed::{Criterion, Glmm, GlmmFit, Lmm, LmmFit};
use epistat_testkit::assert_same_on_threads;

fn lmm_values(fit: &LmmFit) -> Vec<f64> {
    let mut v = fit.beta.clone();
    v.extend(entries(&fit.beta_covariance));
    v.extend(&fit.theta);
    v.extend(entries(&fit.random_covariance));
    v.extend([fit.sigma, fit.criterion]);
    v.extend(&fit.modes);
    v.extend(&fit.mode_covariances);
    v
}

fn glmm_values(fit: &GlmmFit) -> Vec<f64> {
    let mut v = fit.beta.clone();
    v.extend(entries(fit.beta_covariance.as_ref().unwrap()));
    v.extend([fit.theta, fit.theta_se.unwrap(), fit.log_likelihood]);
    v.extend(&fit.gradient);
    v.extend(&fit.modes);
    v.extend(&fit.mode_variances);
    v
}

#[test]
fn linear_mixed_models_do_not_depend_on_the_thread_count() {
    let d = simulate(1500, "gaussian", 11);
    for criterion in [Criterion::Reml, Criterion::Ml] {
        assert_same_on_threads(|| {
            let fit = Lmm::new(criterion)
                .fit(
                    &view(
                        d.x.as_ref(),
                        &d.y,
                        &d.cluster,
                        Some(&d.slope),
                        Some(&d.weights),
                        Some(&d.offset),
                    )
                    .unwrap(),
                )
                .unwrap();
            lmm_values(&fit)
        });
    }
}

#[test]
fn generalised_linear_mixed_models_do_not_depend_on_the_thread_count() {
    for (kind, family, link, nodes) in [
        ("binomial", Family::Binomial, Link::Logit, 1),
        ("poisson", Family::Poisson, Link::Log, 7),
    ] {
        let d = simulate(1500, kind, 12);
        assert_same_on_threads(|| {
            let fit = Glmm::new(family, link, nodes)
                .fit(
                    &view(
                        d.x.as_ref(),
                        &d.y,
                        &d.cluster,
                        None,
                        Some(&d.weights),
                        Some(&d.offset),
                    )
                    .unwrap(),
                )
                .unwrap();
            glmm_values(&fit)
        });
    }
}

/// The entries of a covariance, every one defined.
fn entries(covariance: &Covariance) -> Vec<f64> {
    covariance
        .as_slice()
        .iter()
        .map(|v| v.expect("a defined entry"))
        .collect()
}
