//! `pseudo::regression::fit` checked against `geepack::geese.fit` 1.3-13 with
//! independence working correlation and a gaussian variance, on pseudo-values of
//! `survival::pseudo`, and against `eventglm` 1.4.5's `cumincglm` and `rmeanglm`
//! (coefficients and robust covariance) on the pseudo-values and the design
//! they made. Identity, log and complementary log-log links, one and several
//! times. See `data/pseudo_glm.R`.
//!
//! # Tolerances
//!
//! Both programs solve the same estimating equation, so the difference is how
//! far each is from its solution. Both are run to a tolerance of `1e-13`, which
//! is a step of Fisher scoring for `geese` and for the fit here, and a relative
//! change of the deviance for `glm.fit`. Fisher scoring converges linearly
//! with a rate `rho` below 1, which leaves an error of at most `rho / (1 - rho)`
//! times the last step, and `rho <= 0.9` bounds it for these problems, as the
//! pseudo-values are not far from the model. Each program then is within
//! `9e-13` of the solution, times the larger of 1 and the largest coefficient,
//! and `2e-12` allows for both. The solve here adds `16 kappa (n + p) eps` times
//! the largest coefficient, `kappa` the condition number of the information
//! matrix reported by the fit, `n` rows and `p` columns.
//!
//! `glm.fit`, which `cumincglm` and `rmeanglm` use for one time, stops on the
//! deviance, which is a sum of squares here, when it changes by less than `1e-13`
//! of its size plus 0.1. Near the solution the sum of squares changes by
//! about `2 d' A d` for a step `d`, so the last step it takes is at most
//! `sqrt(1e-13 (SS + 0.1) / (2 lambda_min))` with `lambda_min` the smallest
//! eigenvalue of the information matrix `A`, at least the reciprocal of the 1-norm
//! of its inverse, and the iteration, whose rate is at most 0.9, leaves at most
//! 9 times that. That is the allowance for the `glm.fit` fits with a log or
//! complementary log-log link, where it exceeds the `2e-12` above, and
//! the identity link is solved in one step.
//!
//! The covariance is a smooth function of the coefficients and a product of
//! sums over `n` rows. Its entry `(i, j)` may differ by
//! `(32 kappa (n + p) eps + 8 tc) sqrt(V_ii V_jj)`, with `tc` the allowance
//! of the coefficients.

use epistat_glm::Link;
use epistat_glm::robust::SandwichType;
use epistat_linalg::Mat;
use epistat_survival::pseudo::regression::{PseudoRegressionOptions, fit};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    geese: Vec<Case>,
    eventglm: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    geepack: String,
    eventglm: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    link: String,
    y: Vec<f64>,
    x: Vec<f64>,
    p: usize,
    id: Vec<u32>,
    coef: Vec<f64>,
    vcov: Vec<f64>,
    vcov_hc3: Option<Vec<f64>>,
    converged: bool,
    method: Option<String>,
}

fn link_of(name: &str) -> Link {
    match name {
        "identity" => Link::Identity,
        "log" => Link::Log,
        "cloglog" => Link::Cloglog,
        _ => unreachable!(),
    }
}

fn run(case: &Case, worst: &mut (f64, f64, String, f64)) {
    assert!(case.converged, "{}: R converged", case.name);
    let n = case.y.len();
    let p = case.p;
    let x = Mat::from_fn(n, p, |i, j| case.x[i * p + j]);
    let mut opts = PseudoRegressionOptions::new(link_of(&case.link));
    opts.tolerance = 1e-13;
    opts.max_iterations = 200;
    let got = fit(x.as_ref(), &case.y, &case.id, None, None, None, &opts)
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));
    assert!(got.converged, "{}: converged", case.name);
    let kappa = got.information_condition;
    let big = case.coef.iter().fold(1.0f64, |m, b| m.max(b.abs()));
    let mut tc = 2e-12 * big + 16.0 * kappa * (n + p) as f64 * EPS * big;
    if case.method.as_deref() == Some("glm.fit") && case.link != "identity" {
        let norm_a = (0..p)
            .map(|j| {
                (0..p)
                    .map(|i| got.information[i * p + j].abs())
                    .sum::<f64>()
            })
            .fold(0.0f64, f64::max);
        let lambda_min = norm_a / kappa;
        let ss: f64 = case
            .y
            .iter()
            .zip(&got.fitted)
            .map(|(y, m)| (y - m).powi(2))
            .sum();
        tc += 9.0 * (1e-13 * (ss + 0.1) / (2.0 * lambda_min)).sqrt();
    }
    for j in 0..p {
        let ratio = (got.coefficients[j] - case.coef[j]).abs() / tc;
        assert!(
            ratio <= 1.0,
            "{} coefficient {j}: got {:e}, R {:e}, tolerance {tc:e}",
            case.name,
            got.coefficients[j],
            case.coef[j]
        );
        worst.0 = worst.0.max(ratio);
    }
    for i in 0..p {
        for j in 0..p {
            let tv = (32.0 * kappa * (n + p) as f64 * EPS + 8.0 * tc)
                * (case.vcov[i * p + i] * case.vcov[j * p + j]).sqrt();
            let diff = (got.covariance.get(i, j).unwrap() - case.vcov[i * p + j]).abs();
            let ratio = diff / tv;
            assert!(
                ratio <= 1.0,
                "{} covariance ({i}, {j}): got {:e}, R {:e}, tolerance {tv:e}",
                case.name,
                got.covariance.get(i, j).unwrap(),
                case.vcov[i * p + j]
            );
            if ratio >= worst.1 {
                worst.1 = ratio;
                worst.2 = format!("{} ({i}, {j})", case.name);
            }
        }
    }
    // The covariance `eventglm` reports for one time, HC3. Each score is divided by
    // one less its leverage, so a relative error `delta` of a leverage `h` moves a
    // score by `delta h / (1 - h)`, and the meat, a sum of squares, by twice that
    // and the rounding of the solve, so the bound above is multiplied by
    // `1 + 2 h_max / (1 - h_max)` for the factor and by the factor itself,
    // `1 / (1 - h_max)^2`, for the size of the entries relative to HC0.
    if case.method.as_deref() == Some("glm.fit") {
        let want = case.vcov_hc3.as_ref().expect("eventglm's covariance");
        let mut hc3 = opts;
        hc3.covariance = SandwichType::Hc3;
        let got = fit(x.as_ref(), &case.y, &case.id, None, None, None, &hc3)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let h_max = got
            .leverage
            .as_ref()
            .unwrap()
            .iter()
            .fold(0.0f64, |m, &h| m.max(h));
        let amplification = (1.0 + 2.0 * h_max / (1.0 - h_max)) / (1.0 - h_max).powi(2);
        for i in 0..p {
            for j in 0..p {
                let tv = (32.0 * kappa * (n + p) as f64 * EPS + 8.0 * tc)
                    * amplification
                    * (want[i * p + i] * want[j * p + j]).sqrt();
                let ratio = (got.covariance.get(i, j).unwrap() - want[i * p + j]).abs() / tv;
                assert!(
                    ratio <= 1.0,
                    "{} HC3 covariance ({i}, {j}): got {:e}, R {:e}, tolerance {tv:e}",
                    case.name,
                    got.covariance.get(i, j).unwrap(),
                    want[i * p + j]
                );
                if ratio >= worst.3 {
                    worst.3 = ratio;
                }
            }
        }
    }
}

#[test]
fn regression_agrees_with_geese_and_eventglm() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/pseudo_glm.json"
    ));
    eprintln!(
        "{}, geepack {}, eventglm {}",
        reference.versions.r, reference.versions.geepack, reference.versions.eventglm
    );
    let mut worst = (0.0, 0.0, String::new(), 0.0);
    for case in reference.geese.iter().chain(&reference.eventglm) {
        run(case, &mut worst);
    }
    eprintln!(
        "worst coefficient {:.4} of tolerance, worst covariance {:.4} at {}, worst HC3 covariance {:.4}",
        worst.0, worst.1, worst.2, worst.3
    );
}
