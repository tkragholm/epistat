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
    Censoring, Competing, Covariate, CovariateKind, GFormulaOptions, Outcome, PersonPeriods, Term,
    WeightCutoff, gformula,
};
use epistat_causal::{Error, Panel};
use epistat_testkit::load_json;
use serde::Deserialize;

/// The root seed of the simulation and the bootstrap of every run here.
const SEED: u64 = 20_261_124;

#[derive(Deserialize)]
#[expect(non_snake_case, reason = "the columns keep the names of the R data")]
struct Data {
    id: Vec<u32>,
    t0: Vec<i64>,
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
    observed_quantile_95: Vec<f64>,
    observed_value_3: Vec<f64>,
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
        key: Panel::new(&d.id, &d.t0).unwrap(),
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
            cutoff: WeightCutoff::None,
        }),
    };
    let run = |outcome: &Outcome<'_>| {
        gformula(
            &data,
            &covariates,
            outcome,
            &[],
            &GFormulaOptions {
                simulated: Some(200),
                ..GFormulaOptions::new(SEED)
            },
        )
        .expect("g-formula")
    };
    let g = run(&outcome);
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
    // Cut weights: a cut at a value moves no weight by more than its error,
    // and a type 7 quantile is a weighted mean of two neighbouring weights,
    // so both keep the tolerance of the weights.
    for (cutoff, want) in [
        (WeightCutoff::Quantile(0.95), &r.observed_quantile_95),
        (WeightCutoff::Value(3.0), &r.observed_value_3),
    ] {
        let Outcome::Survival {
            event,
            terms,
            competing,
            censoring,
        } = &outcome
        else {
            unreachable!("a survival outcome")
        };
        let cut = Outcome::Survival {
            event,
            terms: terms.clone(),
            competing: competing.clone(),
            censoring: censoring.clone().map(|c| Censoring { cutoff, ..c }),
        };
        let got: Vec<f64> = run(&cut)
            .observed
            .iter()
            .map(|o| o.expect("defined"))
            .collect();
        assert!(
            worst(&got, want) < 1e-8,
            "{cutoff:?}: {got:?} against {want:?}"
        );
    }
    // Cutoffs outside their ranges are errors, not a panic or every weight set
    // to the cutoff.
    for cutoff in [
        WeightCutoff::Quantile(1.5),
        WeightCutoff::Quantile(-0.1),
        WeightCutoff::Quantile(f64::NAN),
        WeightCutoff::Value(-1.0),
        WeightCutoff::Value(0.0),
        WeightCutoff::Value(f64::INFINITY),
    ] {
        let Outcome::Survival {
            event,
            terms,
            competing,
            censoring,
        } = &outcome
        else {
            unreachable!("a survival outcome")
        };
        let cut = Outcome::Survival {
            event,
            terms: terms.clone(),
            competing: competing.clone(),
            censoring: censoring.clone().map(|c| Censoring { cutoff, ..c }),
        };
        let result = gformula(&data, &covariates, &cut, &[], &GFormulaOptions::new(SEED));
        assert!(
            matches!(result, Err(Error::Input(ref e)) if e.to_string().contains("weight cutoff")),
            "{cutoff:?}"
        );
    }
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
            cutoff: WeightCutoff::None,
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
                key: Panel::new(&id, &period).unwrap(),
                baseline: &baseline,
                time_varying: &time_varying,
            },
            &covariates,
            &outcome,
            &[],
            &GFormulaOptions::new(SEED),
        )
        .is_err()
    );
}
