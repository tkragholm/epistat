//! Every estimator meets a bad column with an error of kind input and the
//! code of the shared check, never a panic.
//!
//! The treatment fixes the number of persons, so an empty column is a column
//! of the wrong length and has the code of a length error. Order does not
//! matter to any estimator here, so `Unsorted` is not tried. The g-formula's
//! columns are checked by its own tests, since its key is a `Panel` that
//! refuses bad keys before the estimator sees them.

use epistat_causal::Error;
use epistat_causal::aipw::{AipwInput, AipwOptions, aipw};
use epistat_causal::balance::cbps::{CbpsInput, CbpsOptions, cbps};
use epistat_causal::balance::entropy::{
    EntropyInput, EntropyOptions, balance_columns, entropy_balance,
};
use epistat_causal::tmle::survival::{SurvivalInput, SurvivalOptions, tmle_survival};
use epistat_causal::tmle::{TmleOptions, tmle};
use epistat_core::{CaseWeights, Classify};
use epistat_linalg::Mat;
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

const N: usize = 40;

/// A covariate in `(-1, 1)` that is not a function of the treatment alone.
fn x(i: usize) -> f64 {
    ((i * 7 % 13) as f64 - 6.0) / 6.5
}

fn treatment() -> Vec<bool> {
    (0..N).map(|i| (i * 5 + 3) % 7 < 3).collect()
}

fn outcome() -> Vec<f64> {
    (0..N)
        .map(|i| f64::from(u8::from((i * 3 + 1) % 5 < 2)))
        .collect()
}

fn propensity() -> Vec<f64> {
    (0..N).map(|i| 0.35 + 0.2 * x(i)).collect()
}

fn ones() -> Vec<f64> {
    vec![1.0; N]
}

/// The code of a bad column beside the treatment.
fn code(bad: BadInput) -> &'static str {
    match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    }
}

/// The code of bad case weights, which [`CaseWeights`] checks itself, an
/// empty column among them.
fn weight_code(bad: BadInput) -> &'static str {
    bad.code()
}

/// The code of a bad column named `name`.
fn code_of(name: &str) -> fn(BadInput) -> &'static str {
    if name == "weight" { weight_code } else { code }
}

/// The kinds a column of numbers can carry.
fn column(bad: BadInput, good: &[f64]) -> Option<Vec<f64>> {
    (bad != BadInput::Unsorted).then(|| bad.column(good))
}

#[test]
fn the_augmented_and_targeted_estimators_refuse_bad_columns() {
    let (a, y, e) = (treatment(), outcome(), propensity());
    let mu: Vec<f64> = (0..N).map(|i| 0.4 + 0.1 * x(i)).collect();
    let input = AipwInput {
        outcome: &y,
        treatment: &a,
        mu1: &mu,
        mu0: &mu,
        propensity: &e,
        weight: None,
        cluster: None,
    };
    for (name, good) in [
        ("outcome", &y),
        ("mu1", &mu),
        ("mu0", &mu),
        ("propensity", &e),
        ("weight", &ones()),
    ] {
        let label = format!("aipw {name}");
        assert_rejects_bad_input_with_codes(&label, code_of(name), |bad| {
            column(bad, good).map(|c| {
                let input = match name {
                    "outcome" => AipwInput {
                        outcome: &c,
                        ..input
                    },
                    "mu1" => AipwInput { mu1: &c, ..input },
                    "mu0" => AipwInput { mu0: &c, ..input },
                    "propensity" => AipwInput {
                        propensity: &c,
                        ..input
                    },
                    _ => {
                        return CaseWeights::new(&c).map_err(Error::from).and_then(|w| {
                            let input = AipwInput {
                                weight: Some(w),
                                ..input
                            };
                            aipw(&input, &AipwOptions::default()).map(|_| ())
                        });
                    }
                };
                aipw(&input, &AipwOptions::default()).map(|_| ())
            })
        });
    }
    for (name, good) in [("outcome", &y), ("propensity", &e), ("weight", &ones())] {
        let label = format!("tmle {name}");
        assert_rejects_bad_input_with_codes(&label, code_of(name), |bad| {
            column(bad, good).map(|c| {
                let input = match name {
                    "outcome" => AipwInput {
                        outcome: &c,
                        ..input
                    },
                    "propensity" => AipwInput {
                        propensity: &c,
                        ..input
                    },
                    _ => {
                        return CaseWeights::new(&c).map_err(Error::from).and_then(|w| {
                            let input = AipwInput {
                                weight: Some(w),
                                ..input
                            };
                            tmle(&input, &TmleOptions::default()).map(|_| ())
                        });
                    }
                };
                tmle(&input, &TmleOptions::default()).map(|_| ())
            })
        });
    }
}

#[test]
fn the_balancing_weights_refuse_bad_columns() {
    let a = treatment();
    let design = Mat::from_fn(N, 2, |i, j| if j == 0 { 1.0 } else { x(i) });
    let covariates = Mat::from_fn(N, 1, |i, _| x(i));
    assert_rejects_bad_input_with_codes("cbps sample_weight", weight_code, |bad| {
        column(bad, &ones()).map(|w| {
            CaseWeights::new(&w).map_err(Error::from).and_then(|w| {
                let input = CbpsInput {
                    treatment: &a,
                    design: design.as_ref(),
                    sample_weight: Some(w),
                };
                cbps(&input, &CbpsOptions::default()).map(|_| ())
            })
        })
    });
    for base in [false, true] {
        let label = if base {
            "entropy_balance base_weight"
        } else {
            "entropy_balance sample_weight"
        };
        let codes = if base { code } else { weight_code };
        assert_rejects_bad_input_with_codes(label, codes, |bad| {
            column(bad, &ones()).map(|w| {
                let sample = (!base)
                    .then(|| CaseWeights::new(&w))
                    .transpose()
                    .map_err(Error::from);
                sample.and_then(|sample_weight| {
                    let input = EntropyInput {
                        treatment: &a,
                        covariates: covariates.as_ref(),
                        sample_weight,
                        base_weight: base.then_some(w.as_slice()),
                    };
                    entropy_balance(&input, &EntropyOptions::default()).map(|_| ())
                })
            })
        });
    }
}

#[test]
fn the_balancing_weights_refuse_a_bad_design() {
    let a = treatment();
    assert_rejects_bad_input_with_codes("cbps design", code, |bad| {
        let design = match bad {
            BadInput::Nan | BadInput::Infinite => {
                let value = bad.column(&[0.5])[0];
                Mat::from_fn(N, 2, |i, j| if (i, j) == (4, 1) { value } else { x(i) })
            }
            BadInput::Empty => Mat::zeros(0, 2),
            BadInput::Mismatched => Mat::from_fn(N - 1, 2, |i, _| x(i)),
            BadInput::Unsorted => return None,
        };
        let input = CbpsInput {
            treatment: &a,
            design: design.as_ref(),
            sample_weight: None,
        };
        Some(cbps(&input, &CbpsOptions::default()).map(|_| ()))
    });
}

#[test]
fn the_survival_estimator_refuses_bad_columns() {
    let a = treatment();
    let e = propensity();
    let time: Vec<usize> = (0..N).map(|i| 1 + i % 3).collect();
    let event: Vec<bool> = (0..N).map(|i| i % 4 == 0).collect();
    let hazard = Mat::from_fn(N, 3, |i, k| 0.1 + 0.02 * k as f64 + 0.03 * x(i));
    let censoring = Mat::from_fn(N, 2, |_, _| 0.05);
    let input = SurvivalInput {
        time: &time,
        event: &event,
        treatment: &a,
        hazard1: hazard.as_ref(),
        hazard0: hazard.as_ref(),
        censoring1: censoring.as_ref(),
        censoring0: censoring.as_ref(),
        propensity: &e,
        weight: None,
        cluster: None,
    };
    for (name, good) in [("propensity", &e), ("weight", &ones())] {
        let label = format!("tmle_survival {name}");
        assert_rejects_bad_input_with_codes(&label, code_of(name), |bad| {
            column(bad, good).map(|c| {
                if name == "propensity" {
                    let input = SurvivalInput {
                        propensity: &c,
                        ..input
                    };
                    return tmle_survival(&input, &SurvivalOptions::new(3)).map(|_| ());
                }
                CaseWeights::new(&c).map_err(Error::from).and_then(|w| {
                    let input = SurvivalInput {
                        weight: Some(w),
                        ..input
                    };
                    tmle_survival(&input, &SurvivalOptions::new(3)).map(|_| ())
                })
            })
        });
    }
}

#[test]
fn an_empty_treatment_is_an_empty_input() {
    let input = AipwInput {
        outcome: &[],
        treatment: &[],
        mu1: &[],
        mu0: &[],
        propensity: &[],
        weight: None,
        cluster: None,
    };
    let error = aipw(&input, &AipwOptions::default()).unwrap_err();
    assert_eq!(error.code(), "input.empty");
}

#[test]
fn the_balance_columns_refuse_a_bad_design() {
    assert_rejects_bad_input_with_codes("balance_columns", BadInput::code, |bad| {
        matches!(bad, BadInput::Nan | BadInput::Infinite).then(|| {
            let spoilt = bad.column(&[0.5])[0];
            let covariates = Mat::from_fn(
                N,
                2,
                |i, j| if (i, j) == (7, 1) { spoilt } else { x(i + j) },
            );
            balance_columns(covariates.as_ref(), 2, true)
        })
    });
}
