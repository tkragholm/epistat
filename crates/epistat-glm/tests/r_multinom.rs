//! Multinomial logit fits checked against R 4.6's `nnet::multinom()` 7.3-21.
//!
//! The reference is `tests/data/multinom.json`, written by
//! `tests/data/multinom.R`. It holds four fits to 500 simulated rows, with
//! three and four classes, each unweighted and with case weights: the
//! coefficients, the fitted probabilities, the residual deviance and the
//! standard errors of `summary()`. R runs BFGS with a relative tolerance of
//! 1e-14 so that its stopping rule does not limit the comparison.
//!
//! Tolerances. The fit is the solution of the score equations, so two correct
//! solvers differ by the precision to which each solves them. R's BFGS at
//! `reltol = 1e-14` leaves the criterion within about 1e-14 relative of its
//! optimum, and the criterion is quadratic around it, so a coefficient is
//! known to about the square root of that, 1e-7, scaled by the curvature. The
//! coefficients are compared at 1e-6 relative with an absolute floor of 1e-6,
//! the probabilities, which depend on them through a bounded function, at
//! 1e-6 absolute, the deviance, which is quadratic in the error, at 1e-9
//! relative, and the standard errors, which come from the information at the
//! fit, at 1e-5 relative.

use epistat_glm::{Error, Multinom};
use epistat_linalg::Mat;
use epistat_testkit::{Tolerance, assert_close, load_json};
use serde::Deserialize;

mod common;
use common::*;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    x: Vec<Vec<f64>>,
    weights: Vec<f64>,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    nnet: String,
}

#[derive(Deserialize)]
struct RFit {
    classes: usize,
    weighted: bool,
    y: Vec<usize>,
    coefficients: Vec<Vec<f64>>,
    se: Vec<Vec<f64>>,
    fitted: Vec<Vec<f64>>,
    deviance: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/multinom.json"
    ))
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let r = reference();
    assert!(r.versions.r.starts_with("R version 4.6"));
    assert_eq!(r.versions.nnet, "7.3.21");
}

#[test]
fn fits_match_nnet_multinom() {
    let r = reference();
    let n = r.x.len();
    let x = Mat::from_fn(n, 4, |i, j| if j == 0 { 1.0 } else { r.x[i][j - 1] });
    for f in &r.fits {
        let weights = f.weighted.then_some(r.weights.as_slice());
        let fit = Multinom::default()
            .fit4(x.as_ref(), &f.y, f.classes, weights)
            .unwrap();
        let label = format!("{} classes, weighted {}", f.classes, f.weighted);
        assert!(fit.converged, "{label}");
        assert!(!fit.fitted_at_boundary, "{label}");
        let want: Vec<f64> = f.coefficients.iter().flatten().copied().collect();
        eprintln!(
            "{label}: worst coefficient deviation {:e}",
            fit.coefficients
                .iter()
                .zip(f.coefficients.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        );
        assert_close(
            &fit.coefficients,
            &want,
            Tolerance::exact().rel(1e-6).abs(1e-6),
        );
        let want: Vec<f64> = f.fitted.iter().flatten().copied().collect();
        assert_close(&fit.probabilities, &want, Tolerance::exact().abs(1e-6));
        assert_close(&[fit.deviance], &[f.deviance], Tolerance::exact().rel(1e-9));
        let want: Vec<f64> = f.se.iter().flatten().copied().collect();
        let se: Vec<f64> = fit.standard_errors().into_iter().map(Option::unwrap).collect();
        assert_close(&se, &want, Tolerance::exact().rel(1e-5));
        let rows: f64 = (0..n)
            .map(|i| fit.row_probabilities(i).unwrap().iter().sum::<f64>())
            .sum();
        assert!((rows - n as f64).abs() < 1e-9, "probabilities sum to one");
    }
}

#[test]
fn separation_does_not_converge_and_a_collinear_design_is_an_error() {
    // The class is determined by the sign of the covariate.
    let x = Mat::from_fn(8, 2, |i, j| if j == 0 { 1.0 } else { i as f64 - 3.5 });
    let y: Vec<usize> = (0..8).map(|i| usize::from(i >= 4)).collect();
    let separated = Multinom::default().fit4(x.as_ref(), &y, 2, None);
    assert!(
        matches!(separated, Err(Error::Separation)),
        "{separated:?}"
    );

    let x = Mat::from_fn(8, 3, |i, j| match j {
        0 => 1.0,
        1 => i as f64,
        _ => 2.0 * i as f64,
    });
    let y = [0, 1, 0, 1, 1, 0, 1, 0];
    assert!(Multinom::default().fit4(x.as_ref(), &y, 2, None).is_err());
    // A label outside the classes, an empty class and a negative weight.
    let x = Mat::from_fn(4, 1, |_, _| 1.0);
    assert!(
        Multinom::default()
            .fit4(x.as_ref(), &[0, 1, 2, 0], 2, None)
            .is_err()
    );
    assert!(
        Multinom::default()
            .fit4(x.as_ref(), &[0, 1, 0, 1], 3, None)
            .is_err()
    );
    assert!(
        Multinom::default()
            .fit4(x.as_ref(), &[0, 1, 0, 1], 2, Some(&[1.0, -1.0, 1.0, 1.0]))
            .is_err()
    );
}
