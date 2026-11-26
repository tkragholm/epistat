//! `attributable_logistic` checked against `AFglm` of the R package AF 0.1.5,
//! run from its source because AF was archived from CRAN, and against a
//! sandwich of the stacked estimating equations with a numerical Jacobian
//! (`numDeriv` 2016.8.1.1) for the weighted fits. `tests/data/af.R` makes the
//! data and the references.
//!
//! Tolerances. The estimate is a closed form of the coefficients, which the
//! fit finds to an IRLS tolerance of `1e-14` on the deviance, so the
//! coefficients differ from R's `glm` (the same tolerance) by about `1e-12`
//! relative and the estimate by less than `1e-10` absolute. The variance
//! against `AFglm` is analytic on both sides and differs only through the
//! coefficients and the rounding of sums of 1,500 terms: relative `1e-9`. The
//! variance against the stacked-equation sandwich is limited by the Jacobian
//! of `numDeriv`, a Richardson extrapolation with a relative step of `1e-4`
//! and 6 levels, whose truncation error is far below `1e-6` for these smooth
//! functions, and by the same rounding: `1e-6`. The data meet both bounds by
//! two to five orders.

use epistat_core::Level;
use epistat_glm::attributable::{AfOptions, Design, IntervalScale, attributable_logistic};
use epistat_glm::{Family, Glm, Link};
use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_testkit::load_json;
use serde::Deserialize;

mod common;
use common::*;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    glm: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    #[serde(rename = "AF")]
    af: String,
    #[serde(rename = "numDeriv")]
    num_deriv: String,
}

#[derive(Deserialize)]
struct Case {
    label: String,
    #[serde(rename = "case_control")]
    is_case_control: bool,
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    x0: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
    cluster: Option<Vec<u32>>,
    af: f64,
    var: f64,
    afpkg: Option<Package>,
    z: f64,
}

#[derive(Deserialize)]
struct Package {
    af: f64,
    var: f64,
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

#[test]
fn logistic_fractions_match_af_and_the_stacked_sandwich() {
    let reference: Reference = load_json("tests/data/af.json");
    assert!(reference.versions.r.starts_with("R version 4.6"));
    assert!(reference.versions.af.starts_with("0.1.5"));
    assert_ne!(reference.versions.num_deriv, "");
    let mut model = Glm::new(Family::Binomial, Link::Logit);
    model.tolerance = 1e-14;
    model.max_iterations = 100;
    let mut worst = (0.0f64, 0.0f64, 0.0f64);
    for c in &reference.glm {
        let x = matrix(&c.x);
        let x0 = matrix(&c.x0);
        let fit = model
            .fit5(x.as_ref(), &c.y, c.weights.as_deref(), None, None)
            .unwrap();
        let design = if c.is_case_control {
            Design::CaseControl
        } else {
            Design::CrossSectional
        };
        let opts = AfOptions {
            design,
            level: Level::confidence(0.95).unwrap(),
            scale: IntervalScale::Identity,
        };
        let data = glm_data(x.as_ref(), &c.y, c.weights.as_deref(), None, None).unwrap();
        let clusters = c.cluster.as_deref().map(Clusters::from_labels);
        let got =
            attributable_logistic(&fit, &data, x0.as_ref(), clusters.as_ref(), &opts).unwrap();
        let estimate = got.af.unwrap();
        let af = estimate.estimate;
        let se = estimate.se.unwrap();
        let d_af = (af - c.af).abs();
        assert!(d_af <= 1e-10, "{}: AF {af} against {}", c.label, c.af);
        let d_var = (se * se - c.var).abs() / c.var;
        assert!(d_var <= 1e-6, "{}: variance, relative {d_var:e}", c.label);
        worst.0 = worst.0.max(d_af);
        worst.1 = worst.1.max(d_var);
        if let Some(p) = &c.afpkg {
            assert!((af - p.af).abs() <= 1e-10, "{}: AF against AFglm", c.label);
            let d = (se * se - p.var).abs() / p.var;
            assert!(
                d <= 1e-9,
                "{}: variance against AFglm, relative {d:e}",
                c.label
            );
            worst.2 = worst.2.max(d);
        }
        assert!((estimate.lower().unwrap() - (af - c.z * se)).abs() <= 1e-14);
        assert!((estimate.upper().unwrap() - (af + c.z * se)).abs() <= 1e-14);
        let log = attributable_logistic(
            &fit,
            &data,
            x0.as_ref(),
            clusters.as_ref(),
            &AfOptions {
                scale: IntervalScale::LogComplement,
                ..opts
            },
        )
        .unwrap()
        .af
        .unwrap();
        if af < 1.0 {
            let half = c.z * se / (1.0 - af);
            let (lo, hi) = (
                1.0 - (1.0 - af) * half.exp(),
                1.0 - (1.0 - af) * (-half).exp(),
            );
            assert!((log.lower().unwrap() - lo).abs() <= 1e-14, "{}", c.label);
            assert!((log.upper().unwrap() - hi).abs() <= 1e-14, "{}", c.label);
            assert!(log.upper().unwrap() < 1.0);
        }
        assert_eq!(
            got.n_clusters,
            c.cluster.as_ref().map_or(0, |v| {
                let mut u = v.clone();
                u.sort_unstable();
                u.dedup();
                u.len()
            })
        );
    }
    eprintln!(
        "worst: AF {:e}, variance against the sandwich {:e}, against AFglm {:e}",
        worst.0, worst.1, worst.2
    );
}
