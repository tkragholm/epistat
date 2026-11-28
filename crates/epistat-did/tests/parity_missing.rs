//! The difference-in-differences estimator for a pre-treatment outcome missing
//! at random, checked against the authors' Python replication code
//! (github.com/testalorenzo/MissingDiD, `dgp.py`, with numpy 2.5.5, pandas
//! 3.1.1 and statsmodels 0.15.0; `tests/data/generate_missing_reference.py`
//! records the commit).
//!
//! Three checks per data set and nuisance specification:
//!
//! 1. The estimator on the reference's own nuisance predictions: the estimate,
//!    the influence function and the standard error agree to the rounding of
//!    the two computations. Under the outcome-dependent assumption the
//!    reference's nested regression is the one of its code, fitted on all
//!    units.
//! 2. The nuisance fits: the predictions of `att_missing_pre`'s logistic and
//!    least-squares models agree with the reference's, within the
//!    reference's solver gap and the forward error of the normal equations.
//!    The nested regression is compared with the paper's, fitted among the
//!    controls, which the script adds.
//! 3. The estimate, influence function and standard error of
//!    `att_missing_pre` against those of the estimator on the reference
//!    predictions, within the propagation of the tolerances of check 2. Under
//!    the outcome-independent assumption the reference code is the paper's
//!    estimator, so this is compared with its estimate directly.
//!
//! None of the reference's clipped probabilities binds in these data sets,
//! which the test checks.

use std::collections::HashMap;

use epistat_did::{
    MissingAssumption, MissingAtt, MissingFolds, MissingModels, MissingOptions, MissingPreData,
    MissingPredictions, NestedOutcome, att_missing_pre, att_missing_pre_from_predictions,
};
use epistat_linalg::Mat;
use epistat_testkit::{Tolerance, load_json};
use serde::Deserialize;

const U: f64 = f64::EPSILON / 2.0;

#[derive(Deserialize)]
struct Reference {
    missingdid_commit: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Fit {
    values: Vec<f64>,
    gap: Option<f64>,
    kappa: Option<f64>,
    clipped: Option<usize>,
}

#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the reference's flags of correct specification, one per model"
)]
struct Spec {
    mu: bool,
    gamma: bool,
    pi: bool,
    eta: bool,
    augmentation: bool,
    nuisance: HashMap<String, String>,
    estimate: f64,
    std: f64,
    terms: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    assumption: String,
    n: usize,
    z: Vec<f64>,
    x: Vec<f64>,
    y0: Vec<Option<f64>>,
    y1: Vec<f64>,
    treated: Vec<bool>,
    fits: HashMap<String, Fit>,
    specs: Vec<Spec>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/missing_ref.json"
    ))
}

const Z: [usize; 4] = [0, 1, 2, 3];
const X: [usize; 4] = [4, 5, 6, 7];

impl Case {
    fn assumption(&self) -> MissingAssumption {
        if self.assumption == "MarX" {
            MissingAssumption::MarX
        } else {
            MissingAssumption::MarXY1
        }
    }

    /// The covariates z1..z4 then x1..x4.
    fn covariates(&self) -> Mat<f64> {
        Mat::from_fn(self.n, 8, |i, j| {
            if j < 4 {
                self.z[i * 4 + j]
            } else {
                self.x[i * 4 + j - 4]
            }
        })
    }

    fn data(&self) -> MissingPreData<'_> {
        MissingPreData {
            treated: &self.treated,
            outcome_pre: &self.y0,
            outcome_post: &self.y1,
            clusters: None,
        }
    }

    fn fit(&self, spec: &Spec, key: &str) -> &Fit {
        &self.fits[&spec.nuisance[key]]
    }

    fn predictions<'a>(&'a self, spec: &Spec, nested: &str) -> MissingPredictions<'a> {
        MissingPredictions {
            propensity: &self.fit(spec, "pi").values,
            gamma_treated: &self.fit(spec, "gamma1").values,
            gamma_control: &self.fit(spec, "gamma0").values,
            pre_treated: &self.fit(spec, "mu01").values,
            post_control: &self.fit(spec, "mu10").values,
            pre_control: &self.fit(spec, "mu00").values,
            nested_control: (self.assumption() == MissingAssumption::MarXY1)
                .then(|| &self.fits[&spec.nuisance[nested]].values[..]),
        }
    }
}

fn cols(correct: bool) -> &'static [usize] {
    if correct { &Z } else { &X }
}

fn name(spec: &Spec) -> String {
    format!(
        "mu {} gamma {} pi {} eta {} augmented {}",
        spec.mu, spec.gamma, spec.pi, spec.eta, spec.augmentation
    )
}

/// Per unit: the sum of the absolute values of the quantities in its term,
/// times its weight, and the absolute derivatives of the term with respect to
/// each prediction, in the order pi, gamma1, gamma0, mu01, mu10, mu00, eta.
struct Sensitivity {
    magnitude: Vec<f64>,
    term: Vec<f64>,
    derivative: Vec<[f64; 7]>,
}

fn sensitivity(case: &Case, p: &MissingPredictions<'_>) -> Sensitivity {
    let mar_x = case.assumption() == MissingAssumption::MarX;
    let mut out = Sensitivity {
        magnitude: Vec::new(),
        term: Vec::new(),
        derivative: Vec::new(),
    };
    for i in 0..case.n {
        let y1 = case.y1[i];
        let (r, y0) = case.y0[i].map_or((0.0, 0.0), |v| (1.0, v));
        let eta = p.nested_control.map_or(p.pre_control[i], |e| e[i]);
        let mut d = [0.0; 7];
        let (magnitude, term) = if case.treated[i] {
            let (m, g) = (p.pre_treated[i], p.gamma_treated[i]);
            d[1] = r * (y0 - m).abs() / (g * g);
            d[3] = (1.0 - r / g).abs();
            d[4] = 1.0;
            if mar_x {
                d[5] = 1.0;
            } else {
                d[6] = 1.0;
            }
            (
                y1.abs()
                    + m.abs()
                    + r * (y0.abs() + m.abs()) / g
                    + p.post_control[i].abs()
                    + eta.abs(),
                y1 - m - r * (y0 - m) / g - p.post_control[i] + eta,
            )
        } else {
            let (pi, m, g) = (p.propensity[i], p.pre_control[i], p.gamma_control[i]);
            let w = pi / (1.0 - pi);
            let bracket = if mar_x {
                y1 - p.post_control[i] - r * (y0 - m) / g
            } else {
                y1 - m - r * (y0 - m) / g - p.post_control[i] + eta
            };
            d[0] = bracket.abs() / ((1.0 - pi) * (1.0 - pi));
            d[2] = w * r * (y0 - m).abs() / (g * g);
            d[4] = w;
            d[5] = if mar_x {
                w * r / g
            } else {
                w * (1.0 - r / g).abs()
            };
            if !mar_x {
                d[6] = w;
            }
            (
                w * (y1.abs()
                    + p.post_control[i].abs()
                    + 2.0 * m.abs()
                    + eta.abs()
                    + r * (y0.abs() + m.abs()) / g),
                -w * bracket,
            )
        };
        out.magnitude.push(magnitude);
        out.term.push(term);
        out.derivative.push(d);
    }
    out
}

/// A bound on the difference between two computations of the estimator from
/// the same predictions, each in floating point. A term is at most 12
/// roundings from its inputs, so it errs by at most `12 u` times the sum of
/// the absolute values it combines; a sum of `n` terms in any order errs by at
/// most `(n - 1) u` times the sum of their absolute values (Higham, Accuracy
/// and Stability of Numerical Algorithms, 2nd ed., section 4.2). Both sides
/// carry these errors, hence the factor 2.
fn rounding_bound(s: &Sensitivity, n_treated: f64) -> (f64, Vec<f64>) {
    let n = s.term.len() as f64;
    let terms: f64 = s.magnitude.iter().map(|m| 12.0 * U * m).sum();
    let sums = (n - 1.0) * U * s.term.iter().map(|t| t.abs()).sum::<f64>();
    let theta = 2.0 * (terms + sums) / n_treated;
    let per_unit = s.magnitude.iter().map(|m| 2.0 * 12.0 * U * m).collect();
    (theta, per_unit)
}

/// The reference's influence function on the crate's scale, from the terms
/// its code sums: `phi_i = n (term_i - A_i theta / sum A)`.
fn reference_influence(case: &Case, spec: &Spec, n_treated: f64) -> Vec<f64> {
    let n = case.n as f64;
    spec.terms
        .iter()
        .zip(&case.treated)
        .map(|(&t, &a)| n * (t - if a { spec.estimate / n_treated } else { 0.0 }))
        .collect()
}

/// Checks the estimate, the influence function and the standard error of
/// `got` against `theta`, `phi` and `se`, given the bound on the estimate and
/// the per-unit bounds on the terms. The influence of unit `i` is
/// `(n / n1)(term_i - A_i theta)`, so its bound is `(n / n1)(term bound + A_i
/// theta bound)` plus the rounding of that product. The standard error of the
/// crate centres the influence function and the reference does not; the
/// difference between the two is computed here and added, with the bound on
/// the norm of the difference of the influence functions over `n`.
fn check(
    what: &str,
    got: &MissingAtt,
    theta: f64,
    phi: &[f64],
    se: f64,
    theta_bound: f64,
    term_bounds: &[f64],
    treated: &[bool],
) {
    let n = phi.len() as f64;
    let n1 = treated.iter().filter(|&&a| a).count() as f64;
    let theta_bound = theta_bound + 2.0 * U * theta.abs();
    assert!(
        (got.estimate.estimate - theta).abs() <= theta_bound,
        "{what}: att {} vs {theta}, bound {theta_bound:e}",
        got.estimate.estimate
    );
    let mut squared = 0.0;
    for i in 0..phi.len() {
        let a = if treated[i] { 1.0 } else { 0.0 };
        let bound = n / n1 * (term_bounds[i] + a * theta_bound)
            + 8.0 * U * phi[i].abs().max(got.influence[i].abs());
        let diff = (got.influence[i] - phi[i]).abs();
        assert!(
            diff <= bound,
            "{what}: influence {i} {} vs {}, bound {bound:e}",
            got.influence[i],
            phi[i]
        );
        squared += bound * bound;
    }
    let uncentred = got.influence.iter().map(|v| v * v).sum::<f64>().sqrt() / n;
    let se_bound =
        (got.estimate.se.unwrap() - uncentred).abs() + squared.sqrt() / n + 4.0 * n * U * se;
    assert!(
        (got.estimate.se.unwrap() - se).abs() <= se_bound,
        "{what}: se {} vs {se}, bound {se_bound:e}",
        got.estimate.se.unwrap()
    );
}

#[test]
fn the_estimator_on_the_reference_predictions_is_the_reference_code() {
    let reference = reference();
    assert_eq!(reference.missingdid_commit.len(), 40);
    for case in &reference.cases {
        let n_treated = case.treated.iter().filter(|&&a| a).count() as f64;
        for spec in &case.specs {
            let what = format!("{} {}", case.assumption, name(spec));
            let p = case.predictions(spec, "eta_all_units");
            let got = att_missing_pre_from_predictions(
                &case.data(),
                case.assumption(),
                &p,
                &MissingOptions::default(),
            )
            .unwrap();
            let s = sensitivity(case, &p);
            let (theta_bound, term_bounds) = rounding_bound(&s, n_treated);
            let phi = reference_influence(case, spec, n_treated);
            check(
                &what,
                &got,
                spec.estimate,
                &phi,
                spec.std,
                theta_bound,
                &term_bounds,
                &case.treated,
            );
        }
    }
}

/// The bound on the difference between `att_missing_pre`'s prediction of a
/// nuisance and the reference's, per unit.
///
/// The reference's own error is its recorded gap. The crate solves the normal
/// equations of the scaled design by Cholesky, whose computed solution has a
/// relative forward error of at most about `c p kappa^2 u` for the design's
/// condition number `kappa` (Higham, section 20.4); `c = 64` covers the
/// accumulation of the Gram matrix and the triangular solves. For a logistic
/// model the system is weighted by `mu (1 - mu)`, which can raise the
/// condition number of the weighted design by `1 / sqrt(min mu (1 - mu))`,
/// a probability moves by at most a quarter of its linear predictor, and the
/// linear predictor is bounded by the largest logit among the predictions.
/// Newton's method stops after a step of at most `1e-10` on the scaled
/// coefficients, which leaves an error of the order of its square, far below
/// the rounding.
fn prediction_bound(fit: &Fit, logistic: bool, fit_rows: &[bool], p: usize) -> f64 {
    let kappa = fit.kappa.expect("a fit with a recorded condition number");
    let gap = fit.gap.expect("a fit with a recorded gap");
    assert_eq!(fit.clipped.unwrap_or(0), 0, "a clipped probability binds");
    let scale = fit.values.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if logistic {
        let weight = fit
            .values
            .iter()
            .zip(fit_rows)
            .filter(|&(_, &r)| r)
            .map(|(&mu, _)| mu * (1.0 - mu))
            .fold(f64::INFINITY, f64::min);
        let linear = fit
            .values
            .iter()
            .map(|&mu| (mu / (1.0 - mu)).ln().abs())
            .fold(0.0, f64::max);
        gap + 0.25 * 64.0 * p as f64 * kappa * kappa / weight * U * (1.0 + linear)
    } else {
        gap + 64.0 * p as f64 * kappa * kappa * U * scale
    }
}

/// A nuisance to compare: its key, the crate's predictions, whether the
/// model is logistic, the units it is fitted on and its number of
/// coefficients.
type Check<'a> = (&'static str, &'a [f64], bool, Vec<bool>, usize);

fn assert_close_each(what: &str, got: &[f64], expected: &[f64], bound: f64) {
    assert_eq!(got.len(), expected.len(), "{what}: lengths differ");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            Tolerance::exact().abs(bound).accepts(g, e),
            "{what}: unit {i}: {g} vs {e}, bound {bound:e}"
        );
    }
}

#[test]
fn the_nuisance_fits_and_the_estimate_agree_with_the_reference() {
    let reference = reference();
    for case in &reference.cases {
        let covariates = case.covariates();
        let n = case.n;
        let n_treated = case.treated.iter().filter(|&&a| a).count() as f64;
        let observed: Vec<bool> = case.y0.iter().map(Option::is_some).collect();
        let mar_xy1 = case.assumption() == MissingAssumption::MarXY1;
        let extra = usize::from(mar_xy1);
        for spec in &case.specs {
            let what = format!("{} {}", case.assumption, name(spec));
            let models = MissingModels {
                propensity: cols(spec.pi),
                missingness: cols(spec.gamma),
                outcome: cols(spec.mu),
                nested: cols(spec.eta),
            };
            let config = MissingOptions {
                nested_outcome: if spec.augmentation {
                    NestedOutcome::Augmented
                } else {
                    NestedOutcome::Plain
                },
                ..MissingOptions::default()
            };
            let got = att_missing_pre(
                &case.data(),
                covariates.as_ref(),
                case.assumption(),
                &models,
                MissingFolds::None,
                &config,
            )
            .unwrap();
            assert_eq!(got.folds, 1);

            // Check 2: the predictions, model by model.
            let all = vec![true; n];
            let treated: Vec<bool> = case.treated.clone();
            let controls: Vec<bool> = case.treated.iter().map(|&a| !a).collect();
            let both = |a: &[bool], b: &[bool]| -> Vec<bool> {
                a.iter().zip(b).map(|(&x, &y)| x && y).collect()
            };
            let mut bounds = [0.0; 7];
            let nu = &got.nuisance;
            let checks: [Check<'_>; 6] = [
                ("pi", &nu.propensity, true, all.clone(), 5),
                (
                    "gamma1",
                    &nu.gamma_treated,
                    true,
                    treated.clone(),
                    5 + extra,
                ),
                (
                    "gamma0",
                    &nu.gamma_control,
                    true,
                    controls.clone(),
                    5 + extra,
                ),
                (
                    "mu01",
                    &nu.pre_treated,
                    false,
                    both(&treated, &observed),
                    5 + extra,
                ),
                ("mu10", &nu.post_control, false, controls.clone(), 5),
                (
                    "mu00",
                    &nu.pre_control,
                    false,
                    both(&controls, &observed),
                    5 + extra,
                ),
            ];
            for (k, (key, values, logistic, rows, p)) in checks.iter().enumerate() {
                let fit = case.fit(spec, key);
                bounds[k] = prediction_bound(fit, *logistic, rows, *p);
                assert_close_each(&format!("{what} {key}"), values, &fit.values, bounds[k]);
            }
            if mar_xy1 {
                // The pseudo-outcome carries the errors of mu00 and gamma0:
                // `|d pseudo| <= |d m| (1 + R / g) + R |y0 - m| / g^2 |d g|`.
                // A least-squares fit is a projection, whose infinity norm is
                // at most `sqrt(rows)` times its 2-norm of 1.
                let fit = &case.fits[&spec.nuisance["eta_controls"]];
                let rows = controls.iter().filter(|&&c| c).count() as f64;
                let gamma0 = &case.fit(spec, "gamma0").values;
                let mu00 = &case.fit(spec, "mu00").values;
                let mut pseudo: f64 = 0.0;
                for i in (0..n).filter(|&i| controls[i]) {
                    let (r, y0) = case.y0[i].map_or((0.0, 0.0), |v| (1.0, v));
                    let aug = if spec.augmentation { 1.0 } else { 0.0 };
                    pseudo = pseudo.max(
                        bounds[5] * (1.0 + aug * r / gamma0[i])
                            + aug * r * (y0 - mu00[i]).abs() / (gamma0[i] * gamma0[i]) * bounds[2],
                    );
                }
                bounds[6] = rows.sqrt() * pseudo + prediction_bound(fit, false, &controls, 5);
                assert_close_each(
                    &format!("{what} eta"),
                    nu.nested_control.as_ref().unwrap(),
                    &fit.values,
                    bounds[6],
                );
            }

            // Check 3: the estimate. The reference is the estimator on the
            // reference predictions, the paper's nested regression included;
            // under MarX that is the reference code's own estimate, so its
            // values are used. The first-order propagation of the prediction
            // bounds is doubled to cover the second-order terms, which are
            // smaller by a further factor of the bounds.
            let p = case.predictions(spec, "eta_controls");
            let s = sensitivity(case, &p);
            let (rounding, term_rounding) = rounding_bound(&s, n_treated);
            let term_bounds: Vec<f64> = (0..n)
                .map(|i| {
                    let propagated: f64 = s.derivative[i]
                        .iter()
                        .zip(&bounds)
                        .map(|(d, b)| d * b)
                        .sum();
                    2.0 * propagated + term_rounding[i]
                })
                .collect();
            let theta_bound = term_bounds.iter().sum::<f64>() / n_treated + rounding;
            if mar_xy1 {
                let on_reference =
                    att_missing_pre_from_predictions(&case.data(), case.assumption(), &p, &config)
                        .unwrap();
                check(
                    &what,
                    &got,
                    on_reference.estimate.estimate,
                    &on_reference.influence,
                    on_reference.estimate.se.unwrap(),
                    theta_bound,
                    &term_bounds,
                    &case.treated,
                );
            } else {
                let phi = reference_influence(case, spec, n_treated);
                check(
                    &what,
                    &got,
                    spec.estimate,
                    &phi,
                    spec.std,
                    theta_bound,
                    &term_bounds,
                    &case.treated,
                );
            }
        }
    }
}

/// Under MarXY1 the reference code regresses the nested outcome on the
/// covariates of every unit; the paper regresses it among the controls. With
/// a correctly specified propensity score the nested regression cancels in
/// expectation, so the two estimates are close; with a misspecified one they
/// are not. This records that the difference is real in the data set, so that
/// check 1 above (which uses the code's regression) and check 3 (the paper's)
/// test different estimators there.
#[test]
fn the_reference_code_and_the_paper_differ_in_the_nested_regression() {
    let reference = reference();
    let case = reference
        .cases
        .iter()
        .find(|c| c.assumption == "MarXY1")
        .unwrap();
    let mut largest: f64 = 0.0;
    for spec in &case.specs {
        let paper = att_missing_pre_from_predictions(
            &case.data(),
            case.assumption(),
            &case.predictions(spec, "eta_controls"),
            &MissingOptions::default(),
        )
        .unwrap();
        largest = largest.max((paper.estimate.estimate - spec.estimate).abs());
    }
    assert!(largest > 0.1, "largest difference {largest}");
}
