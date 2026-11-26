//! A categorical covariate in the g-formula against gfoRmula 1.1.1 and
//! nnet 7.3-21. See `data/gformula_categorical.R`.
//!
//! # The tolerances
//!
//! The multinomial model of the categorical covariate is fitted by Newton's
//! method to a step below `1e-10 (1 + max |b|)`, and R's refit runs BFGS to a
//! relative change of `1e-14` in the deviance; at a deviance of about `10^4`
//! that leaves R's coefficients within `sqrt(2 * 1e-14 * 10^4 / lambda)` of the
//! maximum, with `lambda`, the smallest eigenvalue of the information, above
//! `50` here: `2e-6`. They are checked to `1e-5`.
//!
//! The risks are simulated by both sides with different random numbers, so
//! they agree within Monte Carlo error: the difference of the two is within
//! four of its standard errors, `sqrt(se_r^2 + se^2)`, at every period and in
//! both strategies (a chance of `6e-5` for each comparison when the two
//! estimate the same risk). gfoRmula fits the categorical model with nnet's
//! default stopping rule, coefficients `1e-4` from the maximum, which moves its
//! risks by about `1e-5`, below a tenth of the standard errors.

use epistat_causal::gformula::{
    Covariate, CovariateKind, GFormulaOptions, Intervention, Outcome, PersonPeriods, Term,
    gformula,
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
    L3: Vec<f64>,
    A: Vec<f64>,
    Y: Vec<u8>,
}

#[derive(Deserialize)]
struct Reference {
    data: Data,
    l3_coef: Vec<f64>,
    natural: Vec<f64>,
    natural_se: Vec<f64>,
    treat: Vec<f64>,
    treat_se: Vec<f64>,
}

const L1: usize = 0;
const L3: usize = 1;
const A: usize = 2;

#[test]
fn a_categorical_covariate_agrees_with_gformula() {
    use Term::{Baseline, Current, Intercept, Lag};
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/gformula_categorical.json"
    ));
    let d = &r.data;
    let event: Vec<bool> = d.Y.iter().map(|&y| y == 1).collect();
    let baseline: [&[f64]; 1] = [&d.V];
    let time_varying: [&[f64]; 3] = [&d.L1, &d.L3, &d.A];
    let data = PersonPeriods {
        id: &d.id,
        period: &d.t0,
        baseline: &baseline,
        time_varying: &time_varying,
    };
    let class = |t: Term, c: f64| Term::is(t, c);
    let covariates = [
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![Intercept, Lag(L1, 1), Lag(A, 1), Baseline(0)],
            restriction: None,
        },
        Covariate {
            kind: CovariateKind::Categorical(3),
            terms: vec![
                Intercept,
                class(Lag(L3, 1), 1.0),
                class(Lag(L3, 1), 2.0),
                Current(L1),
            ],
            restriction: None,
        },
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![
                Intercept,
                Lag(A, 1),
                Current(L1),
                class(Current(L3), 1.0),
                class(Current(L3), 2.0),
            ],
            restriction: None,
        },
    ];
    let outcome = Outcome::Survival {
        event: &event,
        terms: vec![
            Intercept,
            Current(L1),
            class(Current(L3), 1.0),
            class(Current(L3), 2.0),
            Current(A),
            Baseline(0),
        ],
        competing: None,
        censoring: None,
    };
    let g = gformula(
        &data,
        &covariates,
        &outcome,
        &[Intervention::set(A, 1.0)],
        &GFormulaOptions {
            simulated: Some(40_000),
            ..GFormulaOptions::default()
        },
    )
    .expect("g-formula");
    let model = g.models.covariates[L3].as_ref().expect("a model of L3");
    assert_eq!(model.coefficients.len(), r.l3_coef.len());
    for (b, want) in model.coefficients.iter().zip(&r.l3_coef) {
        assert!((b - want).abs() < 1e-5, "{b} against {want}");
    }
    for (s, (want, se)) in [(&r.natural, &r.natural_se), (&r.treat, &r.treat_se)]
        .into_iter()
        .enumerate()
    {
        for (k, e) in g.strategies[s].value.iter().enumerate() {
            let mine = e.estimate.expect("a risk");
            let own = e.monte_carlo_se.expect("an error");
            let tol = 4.0 * (se[k] * se[k] + own * own).sqrt();
            assert!(
                (mine - want[k]).abs() < tol,
                "strategy {s}, period {k}: {mine} against {} (tolerance {tol:e})",
                want[k]
            );
        }
    }
}

#[test]
fn a_class_outside_the_levels_is_an_error() {
    let id = [1, 1];
    let period = [0, 1];
    let x = [0.0, 3.0];
    let baseline: [&[f64]; 0] = [];
    let time_varying: [&[f64]; 1] = [&x];
    let flags = [false, false];
    let covariates = [Covariate {
        kind: CovariateKind::Categorical(3),
        terms: vec![Term::Intercept],
        restriction: None,
    }];
    let outcome = Outcome::Survival {
        event: &flags,
        terms: vec![Term::Intercept],
        competing: None,
        censoring: None,
    };
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
