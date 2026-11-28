//! `Glm::fit_rows` and `RowsFit::robust_covariance` on rows formed block by
//! block give the bits of `Glm::fit` and `robust_covariance` on the same rows
//! held as a matrix: the coefficients, the deviance, the iterations, the
//! dispersion, the model covariance and every sandwich, for several families,
//! for designs of one block and of several, with an aliased column, with prior
//! weights of 0, and with clusters that are and are not contiguous.

use epistat_core::{Covariance, InputError};
use epistat_glm::robust::{SandwichType, robust_covariance};
use epistat_glm::rows::{Block, Rows};
use epistat_glm::{Error, Family, Glm, Link};
use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_testkit::{SplitMix64, assert_same_on_pools};

mod common;
use common::*;

struct Held {
    x: Mat<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    cluster: Vec<usize>,
}

impl Rows for Held {
    fn n_rows(&self) -> usize {
        self.y.len()
    }

    fn n_columns(&self) -> usize {
        self.x.ncols()
    }

    fn fill(&self, start: usize, mut block: Block<'_>) {
        for k in 0..block.y.len() {
            for j in 0..self.x.ncols() {
                block.x[(k, j)] = self.x[(start + k, j)];
            }
            block.y[k] = self.y[start + k];
            block.weight[k] = self.w[start + k];
            block.cluster[k] = self.cluster[start + k];
        }
    }
}

fn normal(rng: &mut SplitMix64) -> f64 {
    let (u, v) = (1.0 - rng.unit(), rng.unit());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

/// `n` rows of an intercept, two covariates and, with `aliased`, a column
/// that is twice the first covariate, between it and a fourth covariate, so
/// that a column after the aliased one is tested too, a response of the family, weights with zeros,
/// and clusters of about five rows, in order or scattered.
fn problem(n: usize, family: Family, aliased: bool, scattered: bool, seed: u64) -> Held {
    let mut rng = SplitMix64::new(seed);
    let p = if aliased { 5 } else { 3 };
    let mut x = Mat::<f64>::zeros(n, p);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b) = (normal(&mut rng), f64::from(u8::from(rng.unit() < 0.4)));
        x[(i, 0)] = 1.0;
        x[(i, 1)] = a;
        if aliased {
            x[(i, 2)] = 2.0 * a;
            x[(i, 3)] = b;
            x[(i, 4)] = normal(&mut rng);
        } else {
            x[(i, 2)] = b;
        }
        let eta = 0.3 * a - 0.5 * b;
        y.push(match family {
            Family::Binomial => f64::from(u8::from(rng.unit() < 1.0 / (1.0 + (-eta).exp()))),
            Family::Poisson => {
                let mean = (0.5 + eta).exp();
                let (mut k, mut t) = (0.0, -rng.unit().ln());
                while t < mean {
                    k += 1.0;
                    t -= rng.unit().ln();
                }
                k
            }
            Family::Gamma => (1.0 + eta).exp() * (0.5 + rng.unit()),
            _ => 1.0 + eta + normal(&mut rng),
        });
        w.push(if i % 11 == 5 { 0.0 } else { 0.5 + rng.unit() });
        labels.push(if scattered {
            usize::try_from(rng.below((n / 5).max(1) as u64)).unwrap()
        } else {
            i / 5
        });
    }
    let cluster = Clusters::from_labels_first_seen(&labels)
        .unwrap()
        .of()
        .iter()
        .map(|&c| c as usize)
        .collect();
    Held { x, y, w, cluster }
}

fn flatten(v: &[Option<f64>]) -> Vec<u64> {
    v.iter().map(|c| c.map_or(u64::MAX, f64::to_bits)).collect()
}

fn flatten_covariance(v: &Covariance) -> Vec<u64> {
    flatten(v.as_slice())
}

fn same(model: Glm, held: &Held) {
    let fit = model
        .fit5(held.x.as_ref(), &held.y, Some(&held.w), None, None)
        .unwrap();
    let rows = model.fit_rows(held).unwrap();
    assert_eq!(flatten(&rows.coefficients), flatten(&fit.coefficients));
    assert_eq!(rows.deviance.to_bits(), fit.deviance.to_bits());
    assert_eq!(
        (
            rows.iterations,
            rows.converged,
            rows.boundary,
            rows.rank,
            rows.residual_df
        ),
        (
            fit.iterations,
            fit.converged,
            fit.boundary,
            fit.rank,
            fit.residual_df
        )
    );
    assert_eq!(
        rows.dispersion.map(f64::to_bits),
        fit.dispersion.map(f64::to_bits)
    );
    assert_eq!(
        flatten_covariance(&rows.covariance),
        flatten_covariance(&fit.covariance)
    );
    let clusters = Clusters::from_labels_first_seen(&held.cluster).unwrap();
    for (clustered, kind) in [
        (false, SandwichType::Hc0),
        (false, SandwichType::Hc1),
        (false, SandwichType::Hc3),
        (true, SandwichType::Hc0),
        (true, SandwichType::Hc1),
    ] {
        let expected = robust_covariance(
            &model,
            &fit,
            held.x.as_ref(),
            &held.y,
            clustered.then_some(&clusters),
            kind,
        );
        let got = rows.robust_covariance(held, clustered, kind);
        match (got, expected) {
            (Ok(g), Ok(e)) => assert_eq!(
                flatten_covariance(&g),
                flatten_covariance(&e),
                "{kind:?} {clustered}"
            ),
            (g, e) => assert_eq!(g.err(), e.err(), "{kind:?} {clustered}"),
        }
    }
}

#[test]
fn the_fit_and_its_covariances_have_the_bits_of_the_held_design() {
    for (family, link) in [
        (Family::Binomial, Link::Logit),
        (Family::Binomial, Link::Probit),
        (Family::Poisson, Link::Log),
        (Family::Gaussian, Link::Identity),
        (Family::Gamma, Link::Log),
    ] {
        for n in [50, 4_096, 4_097, 13_000] {
            for (aliased, scattered) in [(false, false), (true, false), (false, true)] {
                same(
                    Glm::new(family, link),
                    &problem(n, family, aliased, scattered, n as u64),
                );
            }
        }
    }
}

#[test]
fn a_ridge_is_added_as_by_the_fit() {
    let held = problem(9_000, Family::Poisson, false, false, 3);
    same(
        Glm {
            ridge: 2.5,
            ..Glm::new(Family::Poisson, Link::Log)
        },
        &held,
    );
}

#[test]
fn errors_are_those_of_the_fit() {
    let model = Glm::new(Family::Binomial, Link::Logit);
    let mut held = problem(6_000, Family::Binomial, false, false, 9);
    held.w[5_000] = -1.0;
    held.w[5_100] = f64::NAN;
    let expected = model
        .fit5(held.x.as_ref(), &held.y, Some(&held.w), None, None)
        .unwrap_err();
    assert_eq!(model.fit_rows(&held).unwrap_err(), expected);
    assert!(matches!(
        expected,
        Error::Input(InputError::Domain {
            what: "weights",
            ..
        })
    ));
    held.w[5_000] = 1.0;
    held.w[5_100] = 1.0;
    held.y[4_500] = 2.0;
    let expected = model
        .fit5(held.x.as_ref(), &held.y, Some(&held.w), None, None)
        .unwrap_err();
    assert_eq!(model.fit_rows(&held).unwrap_err(), expected);
}

#[test]
fn the_fit_does_not_depend_on_the_thread_count() {
    let held = problem(20_000, Family::Binomial, true, true, 17);
    let model = Glm::new(Family::Binomial, Link::Logit);
    assert_same_on_pools(&[1, 3, 8], || {
        let f = model.fit_rows(&held).unwrap();
        let mut v: Vec<f64> = f.coefficients.iter().flatten().copied().collect();
        v.push(f.deviance);
        v.extend(
            f.robust_covariance(&held, true, SandwichType::Hc1)
                .unwrap()
                .as_slice()
                .iter()
                .flatten(),
        );
        v
    });
}
