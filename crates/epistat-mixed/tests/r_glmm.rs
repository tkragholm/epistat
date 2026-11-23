//! Generalised linear mixed models with a random intercept checked against
//! R 4.6's `lme4::glmer` 2.0.6, and the Gauss-Hermite rules against
//! `lme4::GHrule`.
//!
//! The reference is `tests/data/glmm.json`, written by `tests/data/glmm.R`:
//! cbpp (binomial with trials), simulated sibling pairs with a binary outcome,
//! a simulated Poisson model with an offset, prior weights and scrambled rows,
//! and a simulated singular binary model, with `nAGQ = 1` and more nodes.
//!
//! Fixed parameters. At `glmer`'s optimum and at two other points the
//! reference holds `lme4`'s deviance function, the conditional modes and
//! variances, and the Hessian of the deviance function by `numDeriv` with
//! Richardson extrapolation, and the quadrature transcribed in R at `lme4`'s
//! modes. The transcription is the formula of this crate evaluated at modes
//! that both iterate to convergence, so the two agree to rounding: the worst
//! deviation observed is about `1e-14` relative, and the comparison is at
//! `1e-11`. `lme4`'s own deviance takes the curvature `a_i`, and so the scale
//! of the nodes, from the PIRLS iterate before the last, one Newton step behind
//! the mode, which moves it by up to `3e-9` relative in these data; it is
//! compared at `1e-8`. The modes and their variances agree to `1.4e-12` of
//! their largest value, set by `lme4`'s PIRLS stopping at a relative change of
//! `1e-13` in its penalised residual sum of squares, and are compared at
//! `1e-10`. The covariance `2 H^{-1}` is compared with the one from
//! `numDeriv`'s Hessian at `1e-6` relative: both are difference
//! approximations, this crate's with an error of about `1e-9` (central
//! differences of an exact gradient at a step of `1e-5`) and `numDeriv`'s of
//! about `1e-7`, the worst deviation observed.
//!
//! Optimum. With `g` the gradient of this crate at `glmer`'s estimate and `H`
//! `numDeriv`'s Hessian there, `δ = H^{-1} g` is `glmer`'s distance from the
//! optimum to first order. The fitted `(θ, β)` must lie within
//! `2 |δ| + 1e-7 |(θ, β)|` of `glmer`'s, and its log likelihood must not be
//! below `glmer`'s by more than rounding. The modes and their variances move
//! with the estimate and are compared with what `glmer` reports at `1e-5`
//! relative, about 100 times `δ`. `glmer`'s `vcov` comes from second
//! differences of its deviance at a step of `1e-4`, whose error is of the
//! order of `1e-5` (`8.8e-6` observed), and is compared at `5e-5`.
//!
//! Approximation. With 25 nodes the quadrature agrees with the marginal
//! likelihood integrated cluster by cluster by `integrate()` at a relative
//! tolerance of `1e-13` to better than `1e-8` relative in every case.

use epistat_glm::{Family, Link};
use epistat_linalg::Mat;
use epistat_mixed::quadrature::gauss_hermite;
use epistat_mixed::{Glmm, GlmmFit};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    rules: Vec<Rule>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    lme4: String,
}

#[derive(Deserialize)]
struct Rule {
    nodes: usize,
    #[serde(deserialize_with = "one_or_many")]
    z: Vec<f64>,
    #[serde(deserialize_with = "one_or_many")]
    w: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    family: String,
    saturated: f64,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    cluster: Vec<u32>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    fits: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    nodes: usize,
    theta: f64,
    #[serde(deserialize_with = "one_or_many")]
    beta: Vec<f64>,
    log_lik: f64,
    singular: bool,
    vcov: Vec<f64>,
    modes: Vec<f64>,
    mode_var: Vec<f64>,
    exact_log_lik: f64,
    at: Vec<Point>,
}

#[derive(Deserialize)]
struct Point {
    par: Vec<f64>,
    deviance: f64,
    transcribed: f64,
    modes: Vec<f64>,
    mode_var: Vec<f64>,
    hessian: Vec<f64>,
}

fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Numbers {
        One(f64),
        Many(Vec<f64>),
    }
    Ok(match Numbers::deserialize(d)? {
        Numbers::One(v) => vec![v],
        Numbers::Many(v) => v,
    })
}

fn reference() -> Reference {
    let r: Reference = load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/glmm.json"));
    assert!(r.versions.r.starts_with("R version 4.6"));
    assert_eq!(r.versions.lme4, "2.0.6");
    r
}

/// `max |a - b| / max |b|`.
fn deviation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

fn model(case: &Case, nodes: usize) -> Glmm {
    match case.family.as_str() {
        "binomial" => Glmm::new(Family::Binomial, Link::Logit, nodes),
        _ => Glmm::new(Family::Poisson, Link::Log, nodes),
    }
}

fn design(case: &Case) -> Mat<f64> {
    Mat::from_fn(case.x.len(), case.x[0].len(), |i, j| case.x[i][j])
}

fn evaluate(case: &Case, x: &Mat<f64>, nodes: usize, par: &[f64]) -> GlmmFit {
    model(case, nodes)
        .evaluate(
            x.as_ref(),
            &case.y,
            &case.cluster,
            Some(&case.weights),
            Some(&case.offset),
            par[0],
            &par[1..],
        )
        .unwrap()
}

/// `lme4`'s deviance function as a log likelihood with the constants: its
/// Laplace deviance has them, its quadrature deviance leaves out the
/// saturated log likelihood.
fn lme4_log_likelihood(case: &Case, nodes: usize, deviance: f64) -> f64 {
    if nodes == 1 {
        -deviance / 2.0
    } else {
        -deviance / 2.0 + case.saturated
    }
}

/// The inverse of a small symmetric matrix by Gauss-Jordan elimination.
fn invert(h: &[f64], k: usize) -> Vec<f64> {
    let mut a = h.to_vec();
    let mut inv: Vec<f64> = (0..k * k)
        .map(|i| if i / k == i % k { 1.0 } else { 0.0 })
        .collect();
    for c in 0..k {
        let pivot = a[c * k + c];
        for j in 0..k {
            a[c * k + j] /= pivot;
            inv[c * k + j] /= pivot;
        }
        for r in 0..k {
            if r != c {
                let f = a[r * k + c];
                for j in 0..k {
                    a[r * k + j] -= f * a[c * k + j];
                    inv[r * k + j] -= f * inv[c * k + j];
                }
            }
        }
    }
    inv
}

#[test]
fn the_rules_are_lme4s() {
    let r = reference();
    for rule in &r.rules {
        let (z, w) = gauss_hermite(rule.nodes).unwrap();
        for i in 0..rule.nodes {
            assert!((z[i] - rule.z[i]).abs() <= 1e-13 * rule.z[i].abs().max(1.0));
            assert!(
                (w[i] - rule.w[i]).abs() <= 1e-13 * rule.w[i],
                "{} {i}",
                rule.nodes
            );
        }
    }
}

#[test]
fn fixed_parameters_match_glmer() {
    let r = reference();
    let (mut worst_dev, mut worst_modes, mut worst_cov) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut worst_lag = 0.0_f64;
    for case in &r.cases {
        let x = design(case);
        for entry in &case.fits {
            for point in &entry.at {
                let got = evaluate(case, &x, entry.nodes, &point.par);
                let lme4 = lme4_log_likelihood(case, entry.nodes, point.deviance);
                let d_lme4 = deviation(&[got.log_likelihood], &[lme4]);
                let d = deviation(&[got.log_likelihood], &[point.transcribed]);
                assert!(
                    d < 1e-11,
                    "{} nAGQ {}: {} against {}",
                    case.name,
                    entry.nodes,
                    got.log_likelihood,
                    point.transcribed
                );
                worst_dev = worst_dev.max(d);
                assert!(
                    d_lme4 < 1e-8,
                    "{} nAGQ {}: {} against lme4's {lme4}",
                    case.name,
                    entry.nodes,
                    got.log_likelihood
                );
                worst_lag = worst_lag.max(d_lme4);
                let dm = deviation(&got.modes, &point.modes)
                    .max(deviation(&got.mode_variances, &point.mode_var));
                assert!(
                    dm < 1e-10,
                    "{} nAGQ {}: modes deviate by {dm:e}",
                    case.name,
                    entry.nodes
                );
                worst_modes = worst_modes.max(dm);
                let k = point.par.len();
                let p = k - 1;
                let singular = point.par[0] == 0.0;
                let (cov_beta, theta_se) = if singular {
                    let block: Vec<f64> = (0..p * p)
                        .map(|i| point.hessian[(1 + i / p) * k + 1 + i % p])
                        .collect();
                    (
                        invert(&block, p)
                            .iter()
                            .map(|v| 2.0 * v)
                            .collect::<Vec<f64>>(),
                        None,
                    )
                } else {
                    let inv: Vec<f64> = invert(&point.hessian, k).iter().map(|v| 2.0 * v).collect();
                    (
                        (0..p * p)
                            .map(|i| inv[(1 + i / p) * k + 1 + i % p])
                            .collect(),
                        Some(inv[0].sqrt()),
                    )
                };
                let dc = deviation(got.beta_covariance.as_ref().unwrap(), &cov_beta);
                assert!(
                    dc < 1e-6,
                    "{} nAGQ {}: covariance deviates by {dc:e}",
                    case.name,
                    entry.nodes
                );
                worst_cov = worst_cov.max(dc);
                if let (Some(a), Some(b)) = (got.theta_se, theta_se) {
                    let d = deviation(&[a], &[b]);
                    assert!(
                        d < 1e-6,
                        "{} nAGQ {}: se of theta deviates by {d:e}",
                        case.name,
                        entry.nodes
                    );
                    worst_cov = worst_cov.max(d);
                } else {
                    assert_eq!(got.theta_se.is_some(), theta_se.is_some());
                }
            }
        }
    }
    eprintln!(
        "worst deviations: log likelihood {worst_dev:e}, lme4's deviance {worst_lag:e}, modes {worst_modes:e}, covariance {worst_cov:e}"
    );
}

#[test]
fn the_optimum_matches_glmer() {
    let r = reference();
    for case in &r.cases {
        let x = design(case);
        for entry in &case.fits {
            let fit = model(case, entry.nodes)
                .fit(
                    x.as_ref(),
                    &case.y,
                    &case.cluster,
                    Some(&case.weights),
                    Some(&case.offset),
                )
                .unwrap();
            assert_eq!(
                fit.singular, entry.singular,
                "{} nAGQ {}",
                case.name, entry.nodes
            );
            assert!(
                fit.convergence.unwrap().converged,
                "{} nAGQ {}: {:?}",
                case.name,
                entry.nodes,
                fit.convergence
            );
            let point = &entry.at[0];
            let k = point.par.len();
            let at_r = evaluate(case, &x, entry.nodes, &point.par);
            let free: Vec<usize> = if entry.singular {
                (1..k).collect()
            } else {
                (0..k).collect()
            };
            let h: Vec<f64> = free
                .iter()
                .flat_map(|&a| free.iter().map(move |&b| point.hessian[a * k + b]))
                .collect();
            let inv = invert(&h, free.len());
            let delta: Vec<f64> = (0..free.len())
                .map(|a| {
                    (0..free.len())
                        .map(|b| inv[a * free.len() + b] * at_r.gradient[free[b]])
                        .sum()
                })
                .collect();
            let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mut ours = vec![fit.theta];
            ours.extend_from_slice(&fit.beta);
            let mut theirs = vec![entry.theta];
            theirs.extend_from_slice(&entry.beta);
            let distance = norm(
                &ours
                    .iter()
                    .zip(&theirs)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<f64>>(),
            );
            let bound = 2.0 * norm(&delta) + 1e-7 * norm(&theirs);
            eprintln!(
                "{} nAGQ {}: |psi - glmer| {distance:e}, first-order distance of glmer {:e}, log likelihood {:e} above glmer's",
                case.name,
                entry.nodes,
                norm(&delta),
                fit.log_likelihood - lme4_log_likelihood(case, entry.nodes, point.deviance)
            );
            assert!(
                distance <= bound,
                "{} nAGQ {}: {ours:?} against {theirs:?}",
                case.name,
                entry.nodes
            );
            let want = lme4_log_likelihood(case, entry.nodes, point.deviance);
            assert!(
                fit.log_likelihood >= want - 1e-12 * want.abs(),
                "{} nAGQ {}",
                case.name,
                entry.nodes
            );
            // glmer's own reports at its estimate.
            let reported = if entry.nodes == 1 {
                entry.log_lik
            } else {
                entry.log_lik + case.saturated
            };
            assert!(deviation(&[fit.log_likelihood], &[reported]) < 1e-9);
            let dm = deviation(&fit.modes, &entry.modes)
                .max(deviation(&fit.mode_variances, &entry.mode_var));
            let dv = deviation(fit.beta_covariance.as_ref().unwrap(), &entry.vcov);
            eprintln!("    modes {dm:e}, vcov {dv:e}");
            assert!(
                dm < 1e-5 && dv < 5e-5,
                "{} nAGQ {}: modes {dm:e}, vcov {dv:e}",
                case.name,
                entry.nodes
            );
        }
    }
}

#[test]
fn quadrature_with_25_nodes_is_the_integral() {
    let r = reference();
    for case in &r.cases {
        let x = design(case);
        for entry in &case.fits {
            let mut par = vec![entry.theta];
            par.extend_from_slice(&entry.beta);
            let mut values = Vec::new();
            for nodes in [1, 5, 15, 25] {
                let got = evaluate(case, &x, nodes, &par);
                values.push(got.log_likelihood);
            }
            eprintln!(
                "{} at the nAGQ {} estimate: log likelihood by 1, 5, 15, 25 nodes {values:?}, integrate() {}",
                case.name, entry.nodes, entry.exact_log_lik
            );
            let d = deviation(&[values[3]], &[entry.exact_log_lik]);
            assert!(
                d < 1e-8,
                "{}: 25 nodes deviate from the integral by {d:e}",
                case.name
            );
        }
    }
}
