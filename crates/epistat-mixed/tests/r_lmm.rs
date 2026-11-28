//! Linear mixed models checked against R 4.6's `lme4::lmer` 2.0.6.
//!
//! The reference is `tests/data/lmm.json`, written by `tests/data/lmm.R`:
//! sleepstudy with a random intercept and slope and with a random intercept
//! only, a simulated random-intercept model with 1000 unbalanced clusters in
//! scrambled row order, prior weights and an offset, a simulated model with a
//! correlated random slope, and a simulated singular fit, each by REML and ML.
//!
//! Fixed `θ`. At `lmer`'s optimum and at two other values of `θ` the
//! reference holds what `lme4` reports for a model at that `θ`. Both programs
//! evaluate the same closed forms, `lme4` through a sparse Cholesky factor
//! and this crate through per-cluster blocks, so they differ by rounding,
//! amplified by the condition number of `M`, below `1e4` here. The worst
//! deviation observed is below `2e-12` of the largest value of a vector; the
//! comparison is at `1e-10`.
//!
//! Optimum. `lmer` stops where BOBYQA's trust region has shrunk to
//! `1e-14`, but its criterion is flat to rounding within about
//! `sqrt(ε |d| / d'')` of the optimum, so its `θ` is not exact. The test takes
//! the gradient `g` of the criterion at `lmer`'s `θ` from this crate's closed
//! form (checked against `lme4` through the fixed-`θ` comparison and against
//! finite differences in `behaviour.rs`) and the Hessian `H` from central
//! differences of it, so that `δ = H^{-1} g` is the distance of `lmer`'s `θ`
//! from the optimum to first order. The fitted `θ` must lie within
//! `2 |δ| + 1e-7 |θ|` of `lmer`'s, where `1e-7` covers this crate's own
//! stopping rule, and its criterion must not exceed `lmer`'s by more than
//! rounding. The other estimates move with `θ`; they are compared at a
//! relative `1e-6`, above `δ` times their derivatives in `θ` in these data.

use epistat_core::Covariance;
use epistat_linalg::{Mat, MatRef};
use epistat_mixed::{Criterion, Error, Lmm, LmmFit, MixedData};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    lme4: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    cluster: Vec<u32>,
    slope: Option<Vec<f64>>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    fits: Fits,
}

#[derive(Deserialize)]
struct Fits {
    reml: Entry,
    ml: Entry,
}

#[derive(Deserialize)]
struct Entry {
    fit: Report,
    singular: bool,
    at: Vec<Report>,
}

#[derive(Deserialize, Clone)]
struct Report {
    #[serde(deserialize_with = "one_or_many")]
    theta: Vec<f64>,
    criterion: f64,
    log_lik: f64,
    #[serde(deserialize_with = "one_or_many")]
    beta: Vec<f64>,
    sigma: f64,
    #[serde(deserialize_with = "one_or_many")]
    vcov: Vec<f64>,
    #[serde(deserialize_with = "one_or_many")]
    random_cov: Vec<f64>,
    modes: Vec<f64>,
    mode_cov: Vec<f64>,
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
    let r: Reference = load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/lmm.json"));
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

fn design(case: &Case) -> Mat<f64> {
    let p = case.x[0].len();
    Mat::from_fn(case.x.len(), p, |i, j| case.x[i][j])
}

fn evaluate(model: Lmm, case: &Case, x: &Mat<f64>, theta: &[f64]) -> LmmFit {
    model
        .evaluate(
            &view(
                x.as_ref(),
                &case.y,
                &case.cluster,
                case.slope.as_deref(),
                Some(&case.weights),
                Some(&case.offset),
            )
            .unwrap(),
            theta,
        )
        .unwrap()
}

/// The worst deviation of each reported quantity.
fn compare(got: &LmmFit, want: &Report) -> [f64; 8] {
    [
        deviation(&[got.criterion], &[want.criterion]),
        deviation(&[got.log_likelihood], &[want.log_lik]),
        deviation(&got.beta, &want.beta),
        deviation(&[got.sigma], &[want.sigma]),
        deviation(&entries(&got.beta_covariance), &want.vcov),
        deviation(&entries(&got.random_covariance), &want.random_cov),
        deviation(&got.modes, &want.modes),
        deviation(&got.mode_covariances, &want.mode_cov),
    ]
}

const NAMES: [&str; 8] = [
    "criterion",
    "logLik",
    "beta",
    "sigma",
    "vcov",
    "VarCorr",
    "modes",
    "condVar",
];

#[test]
fn fixed_theta_matches_lmer() {
    let r = reference();
    let mut worst = 0.0_f64;
    for case in &r.cases {
        let x = design(case);
        for (criterion, entry) in [
            (Criterion::Reml, &case.fits.reml),
            (Criterion::Ml, &case.fits.ml),
        ] {
            let model = Lmm::new(criterion);
            for want in std::iter::once(&entry.fit).chain(&entry.at) {
                let got = evaluate(model, case, &x, &want.theta);
                for (name, d) in NAMES.iter().zip(compare(&got, want)) {
                    assert!(
                        d < 1e-10,
                        "{} {criterion:?} at {:?}: {name} deviates by {d:e}",
                        case.name,
                        want.theta
                    );
                    worst = worst.max(d);
                }
            }
        }
    }
    eprintln!("worst fixed-theta deviation {worst:e}");
}

#[test]
fn the_optimum_matches_lmer() {
    let r = reference();
    for case in &r.cases {
        let x = design(case);
        for (criterion, entry) in [
            (Criterion::Reml, &case.fits.reml),
            (Criterion::Ml, &case.fits.ml),
        ] {
            let model = Lmm::new(criterion);
            let fit = model
                .fit(
                    &view(
                        x.as_ref(),
                        &case.y,
                        &case.cluster,
                        case.slope.as_deref(),
                        Some(&case.weights),
                        Some(&case.offset),
                    )
                    .unwrap(),
                )
                .unwrap();
            let want = &entry.fit;
            assert_eq!(fit.singular, entry.singular, "{} {criterion:?}", case.name);
            assert!(
                fit.convergence.unwrap().converged,
                "{} {criterion:?}: {:?} gradient {:?} theta {:?}",
                case.name,
                fit.convergence,
                fit.gradient,
                fit.theta
            );
            // lmer's distance from the optimum, to first order.
            let at_r = evaluate(model, case, &x, &want.theta);
            let k = want.theta.len();
            let free: Vec<usize> = (0..k)
                .filter(|&j| !(entry.singular && want.theta[j] == 0.0 && (j == 0 || j == 2)))
                .collect();
            let mut h = vec![0.0; free.len() * free.len()];
            for (a, &j) in free.iter().enumerate() {
                let step = 1e-5 * want.theta[j].abs().max(1e-2);
                let mut plus = want.theta.clone();
                let mut minus = want.theta.clone();
                plus[j] += step;
                minus[j] -= step;
                let gp = evaluate(model, case, &x, &plus).gradient;
                let gm = evaluate(model, case, &x, &minus).gradient;
                for (b, &l) in free.iter().enumerate() {
                    h[a * free.len() + b] = (gp[l] - gm[l]) / (2.0 * step);
                }
            }
            let g: Vec<f64> = free.iter().map(|&j| at_r.gradient[j]).collect();
            let delta = solve(&h, &g);
            let delta_norm = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
            let theta_norm = want.theta.iter().map(|d| d * d).sum::<f64>().sqrt();
            let distance = fit
                .theta
                .iter()
                .zip(&want.theta)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            let bound = 2.0 * delta_norm + 1e-7 * theta_norm.max(1e-3);
            eprintln!(
                "{} {criterion:?}: |theta - lmer| {distance:e}, bound {bound:e}, criterion {:e} below lmer",
                case.name,
                want.criterion - fit.criterion
            );
            assert!(
                distance <= bound,
                "{} {criterion:?}: theta {:?} against {:?}",
                case.name,
                fit.theta,
                want.theta
            );
            assert!(
                fit.criterion <= want.criterion + 1e-11 * want.criterion.abs(),
                "{} {criterion:?}: criterion {} above lmer's {}",
                case.name,
                fit.criterion,
                want.criterion
            );
            for (name, d) in NAMES.iter().zip(compare(&fit, want)) {
                assert!(
                    d < 1e-6,
                    "{} {criterion:?}: {name} deviates by {d:e}",
                    case.name
                );
            }
        }
    }
}

/// `H^{-1} g` for a small symmetric `H`, by Gaussian elimination.
fn solve(h: &[f64], g: &[f64]) -> Vec<f64> {
    let k = g.len();
    let mut a = h.to_vec();
    let mut b = g.to_vec();
    for c in 0..k {
        for r in c + 1..k {
            let f = a[r * k + c] / a[c * k + c];
            for j in c..k {
                a[r * k + j] -= f * a[c * k + j];
            }
            b[r] -= f * b[c];
        }
    }
    for c in (0..k).rev() {
        let mut s = b[c];
        for j in c + 1..k {
            s -= a[c * k + j] * b[j];
        }
        b[c] = s / a[c * k + c];
    }
    b
}

/// The data view of the given columns.
fn view<'a>(
    x: MatRef<'a, f64>,
    y: &'a [f64],
    groups: &'a [u32],
    slope: Option<&'a [f64]>,
    weights: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
) -> Result<MixedData<'a>, Error> {
    let mut data = MixedData::builder(x, y, groups);
    if let Some(slope) = slope {
        data = data.slope(slope);
    }
    if let Some(weights) = weights {
        data = data.weights(weights);
    }
    if let Some(offset) = offset {
        data = data.offset(offset);
    }
    data.build()
}

/// The entries of a covariance, every one defined.
fn entries(covariance: &Covariance) -> Vec<f64> {
    covariance
        .as_slice()
        .iter()
        .map(|v| v.expect("a defined entry"))
        .collect()
}
