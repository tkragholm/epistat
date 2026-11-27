//! The augmented inverse probability weighting estimators checked against R 4.6.2:
//! estimators written out by hand with `glm` (the efficient variance, and the
//! infinitesimal jackknife of the whole procedure for the stacked variance, found
//! by numerical derivatives of refits with case weights), and
//! `riskRegression::ate(estimator = "AIPTW")` for a binary outcome. See
//! `data/aipw.R` for the construction.
//!
//! # The tolerances
//!
//! The glm fits run IRLS from the same start with the same stopping rule as R's,
//! so a coefficient differs by `bound = 8 kappa eps` relative to `max(|beta|, 1)`,
//! with `kappa` the largest condition number of the three information matrices on
//! their scaled columns (see `r_weighting.rs` of `epistat-survival`). A mean is a
//! smooth function of the coefficients, with `L = sum_j |d theta / d beta_j|
//! max(|beta_j|, 1)` recorded by R, so it differs by at most `bound L`, and the sums
//! over `n` persons add `16 n eps max(|theta|, 1)`.
//!
//! The efficient variance is a sum of squares of influences, each of which moves
//! by the same relative amount as the means do, so for the core, which takes R's
//! own predictions, the two sides differ by the rounding of the sums, `64 n eps`
//! relative. The stacked variance gets the fits of this crate, so it adds the
//! difference `bound L sqrt(n_units) / se` relative in each influence (a
//! Cauchy-Schwarz bound for a sum of `n_units` products with an error in one
//! factor, with `L` summed over the means), the truncation of the reference's
//! finite differences (twice the difference between the variance from the
//! plain central differences and from the Richardson-extrapolated ones), and the
//! rounding of its refits, `3 * 1e-13 / h` per influence, times `sqrt(n_units V)`
//! twice.
//!
//! # The small-sample factor
//!
//! The references give the variances `sum IF^2`, with clusters `G / (G - 1) sum
//! S_g^2`, as `riskRegression::ate` does without clusters. The estimators centre
//! the influences and apply `G / (G - 1)` with every person its own cluster when
//! there are none, so their unclustered standard errors are the reference's
//! times `sqrt(n / (n - 1))`, which [`se_factor`] applies to the reference
//! before the comparison. The influences have mean 0 up to the rounding of
//! their sums, so the centring changes the variance by the square of that
//! rounding, far below the tolerances.

use epistat_causal::Estimand;
use epistat_causal::aipw::{Aipw, AipwInput, AipwOptions, GlmInput, Variance, aipw, aipw_glm};
use epistat_glm::Family;
use epistat_linalg::Mat;
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;
const H: f64 = 1e-4;
const FIT_TOLERANCE: f64 = 1e-13;

#[derive(Deserialize)]
struct Versions {
    r: String,
    #[serde(rename = "riskRegression")]
    risk_regression: String,
}

#[derive(Deserialize)]
struct Summary {
    mean1: f64,
    mean0: f64,
    diff: f64,
    ratio: f64,
    se_mean1: f64,
    se_mean0: f64,
    se_diff: f64,
    se_ratio: f64,
}

#[derive(Deserialize)]
struct Scenario {
    name: String,
    outcome: String,
    target: String,
    trunc: Option<Vec<f64>>,
    n_clipped: usize,
    y: Vec<f64>,
    weights: Vec<f64>,
    cluster: Option<Vec<u32>>,
    mu1: Vec<f64>,
    mu0: Vec<f64>,
    propensity: Vec<f64>,
    efficient: Summary,
    stacked: Summary,
    stacked_plain: Summary,
    n_units: usize,
    lipschitz: Vec<f64>,
    kappa: f64,
    weighted: bool,
}

#[derive(Deserialize)]
struct RiskRegressionSummary {
    mean0: f64,
    mean1: f64,
    se_mean0: f64,
    se_mean1: f64,
    diff: f64,
    se_diff: f64,
    ratio: f64,
    se_ratio: f64,
}

#[derive(Deserialize)]
struct RiskRegression {
    y: Vec<f64>,
    a: Vec<u8>,
    mu1: Vec<f64>,
    mu0: Vec<f64>,
    propensity: Vec<f64>,
    known: RiskRegressionSummary,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    x: Vec<Vec<f64>>,
    treatment: Vec<u8>,
    scenarios: Vec<Scenario>,
    riskregression: RiskRegression,
}

fn load() -> Reference {
    let r: Reference = load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/aipw.json"));
    assert!(r.versions.r.starts_with("R version 4.6"));
    assert_eq!(r.versions.risk_regression, "2026.3.11");
    r
}

fn target(name: &str) -> Estimand {
    match name {
        "ATE" => Estimand::Ate,
        "ATT" => Estimand::Att,
        "ATC" => Estimand::Atc,
        other => panic!("target {other}"),
    }
}

fn family(name: &str) -> Family {
    match name {
        "binary" => Family::Binomial,
        "gaussian" => Family::Gaussian,
        "poisson" => Family::Poisson,
        other => panic!("family {other}"),
    }
}

fn options(s: &Scenario) -> AipwOptions {
    AipwOptions {
        truncation: s.trunc.as_ref().map(|t| (t[0], t[1])),
        ..AipwOptions::default()
    }
}

/// The deviation of the point estimates and standard errors of `got` from the
/// reference, in units of their tolerances.
/// The factor from the reference's standard errors to the estimators': `sqrt(n
/// / (n - 1))` without clusters, 1 with them.
fn se_factor(n: usize, clustered: bool) -> f64 {
    if clustered {
        1.0
    } else {
        (n as f64 / (n as f64 - 1.0)).sqrt()
    }
}

fn compare(
    name: &str,
    got: &Aipw,
    want: &Summary,
    point_tol: [f64; 2],
    factor: f64,
    se_tol: impl Fn(usize, f64) -> f64,
) -> f64 {
    let mut worst = 0.0f64;
    let mut check = |what: &str, g: f64, w: f64, tol: f64| {
        let d = (g - w).abs() / tol;
        assert!(
            d <= 1.0,
            "{name} {what}: {g} against {w} (tolerance {tol:e})"
        );
        worst = worst.max(d);
    };
    let [t1, t0] = point_tol;
    check("mean1", got.contrast.mean1.estimate, want.mean1, t1);
    check("mean0", got.contrast.mean0.estimate, want.mean0, t0);
    check(
        "difference",
        got.contrast.difference.estimate,
        want.diff,
        t1 + t0,
    );
    let r = got.contrast.ratio.unwrap();
    check(
        "ratio",
        r.estimate,
        want.ratio,
        (t1 + want.ratio.abs() * t0) / want.mean0.abs(),
    );
    for (k, (what, g, w)) in [
        ("se mean1", got.contrast.mean1.se.unwrap(), want.se_mean1),
        ("se mean0", got.contrast.mean0.se.unwrap(), want.se_mean0),
        (
            "se difference",
            got.contrast.difference.se.unwrap(),
            want.se_diff,
        ),
        ("se ratio", r.se.unwrap(), want.se_ratio),
    ]
    .into_iter()
    .enumerate()
    {
        let w = w * factor;
        check(what, g, w, se_tol(k, w));
    }
    worst
}

fn point_tolerances(s: &Scenario, n: usize, theta: [f64; 2]) -> [f64; 2] {
    let bound = 8.0 * s.kappa * EPS;
    [0, 1].map(|m| bound * s.lipschitz[m] + 16.0 * n as f64 * EPS * theta[m].abs().max(1.0))
}

#[test]
fn core_matches_the_hand_written_estimator() {
    let reference = load();
    let treatment: Vec<bool> = reference.treatment.iter().map(|&a| a == 1).collect();
    let n = treatment.len();
    for s in &reference.scenarios {
        let input = AipwInput {
            outcome: &s.y,
            treatment: &treatment,
            mu1: &s.mu1,
            mu0: &s.mu0,
            propensity: &s.propensity,
            weight: s.weighted.then_some(s.weights.as_slice()),
            cluster: s.cluster.as_deref(),
        };
        let got = aipw(
            &input,
            &AipwOptions {
                estimand: target(&s.target),
                ..options(s)
            },
        )
        .unwrap();
        assert_eq!(got.variance, Variance::Efficient);
        assert_eq!(got.n_trimmed, s.n_clipped, "{}", s.name);
        // The predictions are R's own, so only the rounding of the sums differs.
        let theta = [s.efficient.mean1, s.efficient.mean0];
        let point = [0, 1].map(|m| 16.0 * n as f64 * EPS * theta[m].abs().max(1.0));
        let factor = se_factor(n, s.cluster.is_some());
        let worst = compare(&s.name, &got, &s.efficient, point, factor, |_, w| {
            64.0 * n as f64 * EPS * w.abs().max(1e-3)
        });
        println!(
            "{}: core, worst deviation in tolerances {worst:.3e}",
            s.name
        );
    }
}

#[test]
fn glm_wrapper_matches_the_jackknife_of_refits() {
    let reference = load();
    let treatment: Vec<bool> = reference.treatment.iter().map(|&a| a == 1).collect();
    let n = treatment.len();
    let x = Mat::from_fn(n, reference.x[0].len(), |i, j| reference.x[i][j]);
    for s in &reference.scenarios {
        let input = GlmInput {
            outcome: &s.y,
            treatment: &treatment,
            outcome_design: x.as_ref(),
            propensity_design: x.as_ref(),
            weight: s.weighted.then_some(s.weights.as_slice()),
            cluster: s.cluster.as_deref(),
        };
        let got = aipw_glm(
            &input,
            family(&s.outcome),
            &AipwOptions {
                estimand: target(&s.target),
                ..options(s)
            },
        )
        .unwrap();
        assert_eq!(got.variance, Variance::Stacked);
        assert_eq!(got.n_trimmed, s.n_clipped, "{}", s.name);
        let theta = [s.stacked.mean1, s.stacked.mean0];
        let point = point_tolerances(s, n, theta);
        let bound = 8.0 * s.kappa * EPS;
        let units = s.n_units as f64;
        // The tolerance of the variance `V = se^2` is the truncation of the
        // finite differences, the rounding of the refits and the relative error of
        // the influences.
        let plain = [
            s.stacked_plain.se_mean1,
            s.stacked_plain.se_mean0,
            s.stacked_plain.se_diff,
            s.stacked_plain.se_ratio,
        ];
        let factor = se_factor(n, s.cluster.is_some());
        let worst = compare(&s.name, &got, &s.stacked, point, factor, |k, w| {
            let v = w * w;
            let truncation = (plain[k] * plain[k] - v).abs();
            let rounding = 3.0 * FIT_TOLERANCE / H * (units * v).sqrt() * 2.0;
            let rel = 4.0
                * (bound * s.lipschitz.iter().sum::<f64>() * units.sqrt() / w.max(1e-3)
                    + 16.0 * n as f64 * EPS);
            // From a tolerance of the variance to one of the standard error.
            (2.0 * truncation + rounding + rel * v) / (2.0 * w)
        });
        // Accounting for the estimation of the models changes the standard error.
        assert!(
            (got.contrast.difference.se.unwrap() - s.efficient.se_diff).abs()
                > 1e-6 * s.efficient.se_diff,
            "{}: the stacked and the efficient standard errors are the same",
            s.name
        );
        println!(
            "{}: stacked, worst deviation in tolerances {worst:.3e}",
            s.name
        );
    }
}

#[test]
fn core_matches_riskregression_ate() {
    let reference = load();
    let rr = &reference.riskregression;
    let treatment: Vec<bool> = rr.a.iter().map(|&a| a == 1).collect();
    let n = treatment.len();
    let input = AipwInput {
        outcome: &rr.y,
        treatment: &treatment,
        mu1: &rr.mu1,
        mu0: &rr.mu0,
        propensity: &rr.propensity,
        weight: None,
        cluster: None,
    };
    let got = aipw(
        &input,
        &AipwOptions {
            estimand: Estimand::Ate,
            ..AipwOptions::default()
        },
    )
    .unwrap();
    // `ate` takes the glm predictions as they are, so the two sides differ by the
    // rounding of the sums only. Its standard errors are those of the influence
    // functions with the nuisance models taken as known, `sqrt(sum IF^2)`, so
    // ours are theirs times `sqrt(n / (n - 1))` (see the module documentation).
    let k = &rr.known;
    let f = se_factor(n, false);
    let tol = 64.0 * n as f64 * EPS;
    let mut worst = 0.0f64;
    for (what, g, w) in [
        ("mean1", got.contrast.mean1.estimate, k.mean1),
        ("mean0", got.contrast.mean0.estimate, k.mean0),
        ("difference", got.contrast.difference.estimate, k.diff),
        ("ratio", got.contrast.ratio.unwrap().estimate, k.ratio),
        ("se mean1", got.contrast.mean1.se.unwrap(), f * k.se_mean1),
        ("se mean0", got.contrast.mean0.se.unwrap(), f * k.se_mean0),
        (
            "se difference",
            got.contrast.difference.se.unwrap(),
            f * k.se_diff,
        ),
        (
            "se ratio",
            got.contrast.ratio.unwrap().se.unwrap(),
            f * k.se_ratio,
        ),
    ] {
        let d = (g - w).abs() / (tol * w.abs());
        assert!(d <= 1.0, "ate {what}: {g} against {w}");
        worst = worst.max(d);
    }
    println!("riskRegression::ate: worst deviation in tolerances {worst:.3e}");
}
