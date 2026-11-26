//! The g-formula with a censoring model against gfoRmula 1.1.1: the outcome,
//! competing-event and censoring models with censored rows, and the observed
//! risk weighted by the inverse probability of remaining uncensored (its "IP
//! weighted risk"). See `data/gformula_censored.R`.
//!
//! # The tolerances
//!
//! The models are fitted by IRLS from the same start with the same stopping
//! rule as `glm`, so a coefficient differs by `8 kappa eps` relative to
//! `max(|c|, 1)`, with `kappa` below `10^3` for these designs of at most five
//! terms: `2e-12`, checked to `1e-9`. A censoring probability then differs by
//! `eta / 4` with `eta = 3 * 5 * 2e-12`, a weight (a product of at most five
//! factors `1 / (1 - p)`, `p < 0.6`) by `5 * 2.5 * eta / 4 = 1e-10` relative,
//! and the observed risk, built from ratios of weighted sums and products of
//! five of them, by `20` times that, `2e-9`; it is checked to `1e-8`.

use epistat_causal::gformula::{
    Censoring, Competing, Covariate, CovariateKind, GFormulaOptions, Outcome, PersonPeriods,
    Term, gformula,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
#[allow(non_snake_case)]
struct Data {
    id: Vec<u64>,
    t0: Vec<usize>,
    V: Vec<f64>,
    L1: Vec<f64>,
    L2: Vec<f64>,
    A: Vec<f64>,
    C: Vec<u8>,
    D: Vec<Option<f64>>,
    Y: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct Reference {
    data: Data,
    outcome_coef: Vec<f64>,
    competing_coef: Vec<f64>,
    censoring_coef: Vec<f64>,
    observed: Vec<f64>,
}

const L1: usize = 0;
const L2: usize = 1;
const A: usize = 2;

fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

#[test]
fn the_models_and_the_weighted_observed_risk_agree_with_gformula() {
    use Term::{Baseline, Current, Intercept, Lag, Period};
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/gformula_censored.json"
    ));
    let d = &r.data;
    let censored: Vec<bool> = d.C.iter().map(|&c| c == 1).collect();
    let event: Vec<bool> = d.Y.iter().map(|y| *y == Some(1.0)).collect();
    let competing: Vec<bool> = d.D.iter().map(|v| *v == Some(1.0)).collect();
    let baseline: [&[f64]; 1] = [&d.V];
    let time_varying: [&[f64]; 3] = [&d.L1, &d.L2, &d.A];
    let data = PersonPeriods {
        id: &d.id,
        period: &d.t0,
        baseline: &baseline,
        time_varying: &time_varying,
    };
    let covariates = [
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![Intercept, Lag(L1, 1), Lag(A, 1), Baseline(0)],
            restriction: None,
        },
        Covariate {
            kind: CovariateKind::Normal,
            terms: vec![Intercept, Lag(L2, 1), Current(L1), Lag(A, 1)],
            restriction: None,
        },
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![Intercept, Lag(A, 1), Current(L1), Current(L2)],
            restriction: None,
        },
    ];
    let outcome = Outcome::Survival {
        event: &event,
        terms: vec![Intercept, Current(L1), Current(L2), Current(A), Baseline(0)],
        competing: Some(Competing {
            event: &competing,
            terms: vec![Intercept, Current(L1), Period],
        }),
        censoring: Some(Censoring {
            censored: &censored,
            terms: vec![Intercept, Current(L1), Current(L2)],
        }),
    };
    let g = gformula(
        &data,
        &covariates,
        &outcome,
        &[],
        &GFormulaOptions {
            simulated: Some(200),
            ..GFormulaOptions::default()
        },
    )
    .expect("g-formula");
    assert!(worst(&g.models.outcome.coefficients, &r.outcome_coef) < 1e-9);
    let competing_model = g.models.competing.as_ref().expect("competing model");
    assert!(worst(&competing_model.coefficients, &r.competing_coef) < 1e-9);
    let censoring_model = g.models.censoring.as_ref().expect("censoring model");
    assert!(worst(&censoring_model.coefficients, &r.censoring_coef) < 1e-9);
    let observed: Vec<f64> = g.observed.iter().map(|o| o.expect("defined")).collect();
    assert!(
        worst(&observed, &r.observed) < 1e-8,
        "{observed:?} against {:?}",
        r.observed
    );
}

#[test]
fn a_row_after_censoring_is_an_error() {
    let id = [1, 1];
    let period = [0, 1];
    let x = [0.0, 1.0];
    let baseline: [&[f64]; 0] = [];
    let time_varying: [&[f64]; 1] = [&x];
    let flags = [false, false];
    let censored = [true, false];
    let outcome = Outcome::Survival {
        event: &flags,
        terms: vec![Term::Intercept],
        competing: None,
        censoring: Some(Censoring {
            censored: &censored,
            terms: vec![Term::Intercept],
        }),
    };
    let covariates = [Covariate {
        kind: CovariateKind::Normal,
        terms: vec![Term::Intercept],
        restriction: None,
    }];
    assert!(
        gformula(
            &PersonPeriods {
                id: &id,
                period: &period,
                baseline: &baseline,
                time_varying: &time_varying,
            },
            &covariates,
            &outcome,
            &[],
            &GFormulaOptions::default(),
        )
        .is_err()
    );
}
