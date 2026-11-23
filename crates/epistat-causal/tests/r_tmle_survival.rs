//! The targeted estimator of the risk by a time with censoring checked against
//! the estimator written out by hand in R 4.6.2, on R's glm predictions of the
//! event and censoring hazards and of the propensity. See
//! `data/tmle_survival.R`.
//!
//! # The tolerances
//!
//! The two sides get the same predictions and take the same number of targeting
//! rounds, which the test checks. They differ by the rounding of the sums and
//! products, `16 (n K) eps max(|v|, 1)` for `K` intervals, and by the
//! fluctuation coefficients, which come from logistic fits that run IRLS from
//! the same start with the same stopping rule as R's: a coefficient differs by
//! `8 eps` relative to `max(|e|, 1)` (one column, so the scaled information has
//! condition number 1), and R records `L = sum_j |dv / de_j| max(|e_j|, 1)` for
//! every reported number `v`. Later rounds take their clever covariate from the
//! earlier ones, which `L` includes.

use epistat_causal::tmle::PropensityBound;
use epistat_causal::tmle::survival::{SurvivalInput, SurvivalOptions, tmle_survival};
use epistat_linalg::Mat;
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Versions {
    r: String,
}

#[derive(Deserialize)]
struct Scenario {
    name: String,
    horizon: usize,
    bound: f64,
    weighted: bool,
    weights: Vec<f64>,
    cluster: Option<Vec<u32>>,
    hazard1: Vec<Vec<f64>>,
    hazard0: Vec<Vec<f64>>,
    censoring1: Vec<Vec<f64>>,
    censoring0: Vec<Vec<f64>>,
    propensity: Vec<f64>,
    numbers: Vec<f64>,
    iterations: Vec<usize>,
    epsilon1: Vec<f64>,
    epsilon0: Vec<f64>,
    scores: Vec<f64>,
    one_step: Vec<f64>,
    lipschitz: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    time: Vec<usize>,
    event: Vec<bool>,
    treatment: Vec<u8>,
    scenarios: Vec<Scenario>,
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

#[test]
fn matches_the_hand_written_estimator() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/tmle_survival.json"
    ));
    assert!(r.versions.r.starts_with("R version 4.6"));
    let treatment: Vec<bool> = r.treatment.iter().map(|&a| a == 1).collect();
    let n = treatment.len();
    for s in &r.scenarios {
        let (h1, h0) = (matrix(&s.hazard1), matrix(&s.hazard0));
        let (c1, c0) = (matrix(&s.censoring1), matrix(&s.censoring0));
        let input = SurvivalInput {
            time: &r.time,
            event: &r.event,
            treatment: &treatment,
            hazard1: h1.as_ref(),
            hazard0: h0.as_ref(),
            censoring1: c1.as_ref(),
            censoring0: c0.as_ref(),
            propensity: &s.propensity,
            weight: s.weighted.then_some(s.weights.as_slice()),
            cluster: s.cluster.as_deref(),
        };
        let opts = SurvivalOptions {
            propensity_bound: if s.bound > 0.0 {
                PropensityBound::At(s.bound)
            } else {
                PropensityBound::None
            },
            ..SurvivalOptions::new(s.horizon)
        };
        let got = tmle_survival(&input, &opts).unwrap();
        assert_eq!(got.iterations.to_vec(), s.iterations, "{}", s.name);
        let ratio = got.ratio.unwrap();
        let values = [
            got.risk1.estimate,
            got.risk0.estimate,
            got.difference.estimate,
            ratio.estimate,
            got.risk1.se,
            got.risk0.se,
            got.difference.se,
            ratio.se,
        ];
        let labels = [
            "risk1",
            "risk0",
            "difference",
            "ratio",
            "se risk1",
            "se risk0",
            "se difference",
            "se ratio",
        ];
        let cells = (n * s.horizon) as f64;
        let mut worst = 0.0f64;
        for (k, (label, v)) in labels.iter().zip(values).enumerate() {
            let want = s.numbers[k];
            let tol = 8.0 * EPS * s.lipschitz[k] + 16.0 * cells * EPS * want.abs().max(1.0);
            let d = (v - want).abs() / tol;
            assert!(
                d <= 1.0,
                "{} {label}: {v} against {want} (tolerance {tol:e})",
                s.name
            );
            worst = worst.max(d);
        }
        for (got, want) in got.epsilon.iter().zip([&s.epsilon1, &s.epsilon0]) {
            for (g, w) in got.iter().zip(want) {
                let tol = 16.0 * cells * EPS * w.abs().max(1.0);
                let d = (g - w).abs() / tol;
                assert!(d <= 1.0, "{} epsilon: {g} against {w}", s.name);
                worst = worst.max(d);
            }
        }
        // Targeting stopped by its rule, at the mean R found.
        let rule = (n as f64).sqrt() * (n as f64).ln();
        for ((g, w), se) in got
            .score
            .iter()
            .zip(&s.scores)
            .zip([got.risk1.se, got.risk0.se])
        {
            assert!(g.abs() <= se / rule, "{}: score {g}", s.name);
            let tol = 8.0 * EPS * s.lipschitz[2] + 16.0 * cells * EPS;
            assert!((g - w).abs() <= tol, "{}: score {g} against {w}", s.name);
        }
        // The targeted risks differ from the one-step (augmented) estimates on the
        // same predictions by a term of second order, here below a tenth of the
        // standard error.
        for (g, w) in [got.risk1.estimate, got.risk0.estimate]
            .iter()
            .zip(&s.one_step)
        {
            assert!((g - w).abs() < 0.1 * got.difference.se, "{}", s.name);
        }
        println!("{}: worst deviation in tolerances {worst:.3e}", s.name);
    }
}
