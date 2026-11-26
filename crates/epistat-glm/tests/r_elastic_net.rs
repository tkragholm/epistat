//! Elastic-net fits checked against R 4.6.2 with glmnet 5.2 (`glmnet.R`).

use epistat_glm::{ElasticNet, Family, ResponseScaling};
use epistat_linalg::Mat;
use epistat_testkit::load_json;
use serde::Deserialize;

mod common;
use common::*;

#[derive(Deserialize)]
struct Reference {
    x: Vec<Vec<f64>>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    penalty: Vec<f64>,
    y_gaussian: Vec<f64>,
    y_binomial: Vec<f64>,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
struct RFit {
    family: String,
    alpha: f64,
    variant: String,
    lambda: f64,
    weighted: bool,
    factors: bool,
    offset: bool,
    coefficients: Vec<f64>,
    sy: f64,
    thresh: f64,
}

/// glmnet stops when `v_j (Δb_j)^2 <= thresh` on the response scaled to unit
/// standard deviation, with `v_j` the weighted variance of column `j`, so a
/// coefficient is determined to about `sqrt(thresh / v_j)` times `sy` in the
/// original units, times a factor for the correlation of the columns, which is
/// well under 10 here. The columns of the design have weighted variances
/// from 1e-7 to 1e6.
fn bound(thresh: f64, scale: f64, moment: f64) -> f64 {
    10.0 * (thresh * scale * scale / moment).sqrt()
}

#[test]
fn fits_match_glmnet() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/glmnet.json"
    ));
    let n = reference.x.len();
    let p = reference.x[0].len() + 1;
    let x = Mat::from_fn(
        n,
        p,
        |i, j| if j == 0 { 1.0 } else { reference.x[i][j - 1] },
    );
    let mut worst_ratio = 0.0_f64;
    let mut worst_abs = 0.0_f64;
    for r in &reference.fits {
        let (family, y) = match r.family.as_str() {
            "gaussian" => (Family::Gaussian, &reference.y_gaussian),
            "binomial" => (Family::Binomial, &reference.y_binomial),
            other => panic!("unknown family {other}"),
        };
        let label = format!(
            "{} alpha {} {} lambda {}",
            r.family, r.alpha, r.variant, r.lambda
        );
        let weights = r.weighted.then_some(reference.weights.as_slice());
        let offset = r.offset.then_some(reference.offset.as_slice());
        let factors: Vec<f64> = std::iter::once(0.0)
            .chain(
                reference
                    .penalty
                    .iter()
                    .map(|&f| if r.factors { f } else { 1.0 }),
            )
            .collect();
        let enet = ElasticNet {
            tolerance: 1e-20,
            scaling: ResponseScaling::Glmnet,
            ..ElasticNet::new(family, r.lambda, r.alpha)
        };
        let fit = enet
            .fit6(x.as_ref(), y, weights, offset, Some(&factors), None)
            .unwrap();
        assert!(fit.converged, "{label}");
        assert_eq!(fit.coefficients.len(), r.coefficients.len());
        let total: f64 = weights.map_or(n as f64, |w| w.iter().sum());
        let (scale, curvature): (f64, Vec<f64>) = if family == Family::Gaussian {
            (r.sy, vec![1.0; n])
        } else {
            (1.0, fit.fitted.iter().map(|p| p * (1.0 - p)).collect())
        };
        // glmnet's intercept follows from its other coefficients, so it
        // carries their errors times the weighted column means.
        let mut allowed = vec![0.0; p];
        for j in 1..p {
            // The weighted variance of the column at the working weights,
            // which is what glmnet's rule uses after centring.
            let w: Vec<f64> = (0..n)
                .map(|i| weights.map_or(1.0, |w| w[i]) / total * curvature[i])
                .collect();
            let sum: f64 = w.iter().sum();
            let mean = (0..n).map(|i| w[i] * x[(i, j)]).sum::<f64>() / sum;
            let moment: f64 = (0..n)
                .map(|i| w[i] * (x[(i, j)] - mean) * (x[(i, j)] - mean))
                .sum();
            allowed[j] = bound(r.thresh, scale, moment);
        }
        allowed[0] = (1..p)
            .map(|j| {
                let mean: f64 = (0..n)
                    .map(|i| weights.map_or(1.0, |w| w[i]) / total * x[(i, j)])
                    .sum();
                mean.abs() * allowed[j]
            })
            .sum::<f64>()
            + bound(r.thresh, scale, 1.0);
        for (j, ((got, expected), bound)) in fit
            .coefficients
            .iter()
            .zip(&r.coefficients)
            .zip(&allowed)
            .enumerate()
        {
            let diff = (got - expected).abs();
            worst_ratio = worst_ratio.max(diff / bound);
            worst_abs = worst_abs.max(diff);
            assert!(
                diff <= *bound,
                "{label}, coefficient {j}: {got} against {expected}, bound {bound}"
            );
        }
    }
    eprintln!(
        "worst absolute deviation {worst_abs:e}, worst fraction of the bound {worst_ratio:e}"
    );
}

/// Without glmnet's scaling the Gaussian fit is a different problem, and the
/// ridge-only fits differ by the factor `sy` in lambda.
#[test]
fn unscaled_gaussian_equals_glmnet_at_a_rescaled_ridge() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/glmnet.json"
    ));
    let n = reference.x.len();
    let p = reference.x[0].len() + 1;
    let x = Mat::from_fn(
        n,
        p,
        |i, j| if j == 0 { 1.0 } else { reference.x[i][j - 1] },
    );
    let r = reference
        .fits
        .iter()
        .find(|r| r.family == "gaussian" && r.alpha == 0.0 && r.variant == "plain")
        .unwrap();
    let mut factors = vec![1.0; p];
    factors[0] = 0.0;
    let at = |lambda: f64, scaling| {
        ElasticNet {
            tolerance: 1e-24,
            scaling,
            ..ElasticNet::new(Family::Gaussian, lambda, 0.0)
        }
        .fit6(
            x.as_ref(),
            &reference.y_gaussian,
            None,
            None,
            Some(&factors),
            None,
        )
        .unwrap()
        .coefficients
    };
    let scaled = at(r.lambda, ResponseScaling::Glmnet);
    let unscaled = at(r.lambda / r.sy, ResponseScaling::None);
    for (a, b) in scaled.iter().zip(&unscaled) {
        assert!((a - b).abs() <= 1e-9 * a.abs().max(1e-3), "{a} {b}");
    }
}

/// At glmnet's default tolerance every fit converges in a few passes per step,
/// although one column has mean 50 and standard deviation 1 and two have
/// scales of 1e-3 and 1e3.
#[test]
fn default_tolerance_converges_in_few_passes() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/glmnet.json"
    ));
    let n = reference.x.len();
    let p = reference.x[0].len() + 1;
    let x = Mat::from_fn(
        n,
        p,
        |i, j| {
            if j == 0 { 1.0 } else { reference.x[i][j - 1] }
        },
    );
    let mut most = 0;
    for r in &reference.fits {
        let y = if r.family == "gaussian" {
            &reference.y_gaussian
        } else {
            &reference.y_binomial
        };
        let family = if r.family == "gaussian" {
            Family::Gaussian
        } else {
            Family::Binomial
        };
        let mut factors = vec![1.0; p];
        factors[0] = 0.0;
        let fit = ElasticNet::new(family, r.lambda, r.alpha)
            .fit6(x.as_ref(), y, None, None, Some(&factors), None)
            .unwrap();
        assert!(fit.converged);
        most = most.max(fit.passes.div_ceil(fit.iterations));
    }
    eprintln!("most passes per step {most}");
    assert!(most <= 60);
}
