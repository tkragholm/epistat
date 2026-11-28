//! The parametric g-formula checked against `gfoRmula::gformula` 1.1.1 and
//! against quantities written out in R (`data/gformula.R`).
//!
//! # The tolerances
//!
//! The models are fitted by `epistat_glm`, whose normal equations agree with
//! R's QR to about `1e-11` relative on designs this well conditioned, so the
//! coefficients are compared at `1e-11` relative and `1e-13` absolute. Where
//! every simulated variable is set by a static intervention and each person is
//! simulated once, the result does not depend on the random draws of either
//! side. A risk is then a mean of sums of products of at most `2 K = 12`
//! probabilities, each of which moves by at most a quarter of `|x' d beta|`;
//! with `|x|_1 <= 12` for these designs and `|d beta| <= 1e-11 |beta| <= 3e-11`,
//! a risk moves by at most `12 * 12 / 4 * 3e-11`, about `1.1e-9`, which is the
//! bound. The
//! nonparametric risk is arithmetic on counts and agrees to `1e-14` relative.
//!
//! Where covariates are simulated, the two sides use different random numbers
//! and agree within the Monte Carlo error: both report the standard error of the
//! simulation from the simulated persons, and the comparison allows four of
//! their combined standard error. The exact two-period g-formula has no error of
//! its own. A bootstrap standard error from `B` replicates has the Monte Carlo
//! error `se / sqrt(2 (B - 1))`; the comparison of two of them allows five of
//! the combined error. The truth of the data-generating process is compared with
//! four bootstrap standard errors, together with its own simulation error from
//! `10^6` persons.

use std::collections::BTreeMap;

use epistat_causal::Panel;
use epistat_causal::gformula::{
    Action, Competing, Covariate, CovariateKind, GFormula, GFormulaOptions, History, Intervention,
    Outcome, PersonPeriods, Restriction, Rule, Term, gformula,
};
use epistat_causal::{Bootstrap as Replicates, Seed};
use epistat_testkit::{
    Tolerance, assert_close, assert_within_standard_errors, load_json, na_f64_vec,
};
use serde::Deserialize;

/// The root seed of the simulation and the bootstrap of every run here.
const SEED: u64 = 20_261_124;

#[derive(Deserialize)]
struct Reference {
    main: Main,
    eof: Eof,
    two: Two,
}

#[derive(Deserialize)]
struct MainData {
    id: Vec<f64>,
    t0: Vec<f64>,
    #[serde(rename = "V")]
    v: Vec<f64>,
    #[serde(rename = "W")]
    w: Vec<f64>,
    #[serde(rename = "L1")]
    l1: Vec<f64>,
    #[serde(rename = "L2")]
    l2: Vec<f64>,
    #[serde(rename = "A")]
    a: Vec<f64>,
    #[serde(rename = "D")]
    d: Vec<f64>,
    #[serde(rename = "Y", deserialize_with = "na_f64_vec")]
    y: Vec<f64>,
}

#[derive(Deserialize)]
struct Curve {
    estimate: Vec<f64>,
    mc_se: Vec<f64>,
    difference_mc_se: Vec<f64>,
}

#[derive(Deserialize)]
struct Bootstrap {
    replicates: usize,
    risk_se: Vec<Vec<f64>>,
    difference_se: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Exact {
    coefficients: BTreeMap<String, Vec<f64>>,
    risk: Vec<Vec<f64>>,
    observed: Vec<f64>,
}

#[derive(Deserialize)]
struct Main {
    data: MainData,
    periods: usize,
    coefficients: BTreeMap<String, Vec<f64>>,
    rmse: BTreeMap<String, f64>,
    observed: Vec<f64>,
    simulated: usize,
    mc: Vec<Curve>,
    bootstrap: Bootstrap,
    truth: BTreeMap<String, Vec<f64>>,
    truth_n: f64,
    exact1: Exact,
    exact2: Exact,
}

#[derive(Deserialize)]
struct EofData {
    id: Vec<f64>,
    t0: Vec<f64>,
    #[serde(rename = "V")]
    v: Vec<f64>,
    #[serde(rename = "L")]
    l: Vec<f64>,
    #[serde(rename = "Z")]
    z: Vec<f64>,
    #[serde(rename = "Y", deserialize_with = "na_f64_vec")]
    y: Vec<f64>,
    #[serde(rename = "Yc", deserialize_with = "na_f64_vec")]
    yc: Vec<f64>,
}

#[derive(Deserialize)]
struct Point {
    estimate: f64,
    mc_se: f64,
    difference_mc_se: f64,
}

#[derive(Deserialize)]
struct EofRun {
    coefficients: BTreeMap<String, Vec<f64>>,
    rmse: BTreeMap<String, f64>,
    observed: f64,
    mc: Vec<Point>,
}

#[derive(Deserialize)]
struct EofExact {
    coefficients: BTreeMap<String, Vec<f64>>,
    value: Vec<f64>,
}

#[derive(Deserialize)]
struct Eof {
    data: EofData,
    simulated: usize,
    binary: EofRun,
    continuous: EofRun,
    exact_binary: EofExact,
    exact_continuous: EofExact,
}

#[derive(Deserialize)]
struct TwoData {
    id: Vec<f64>,
    t0: Vec<f64>,
    #[serde(rename = "V")]
    v: Vec<f64>,
    #[serde(rename = "L")]
    l: Vec<f64>,
    #[serde(rename = "A")]
    a: Vec<f64>,
    #[serde(rename = "D")]
    d: Vec<f64>,
    #[serde(rename = "Y", deserialize_with = "na_f64_vec")]
    y: Vec<f64>,
}

#[derive(Deserialize)]
struct Two {
    data: TwoData,
    coefficients: BTreeMap<String, Vec<f64>>,
    exact: BTreeMap<String, Vec<f64>>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/gformula.json"
    ))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "ids and periods written by R are small whole numbers"
)]
fn ids(values: &[f64]) -> Vec<u32> {
    values.iter().map(|&v| v as u32).collect()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "periods written by R are small whole numbers"
)]
fn periods(values: &[f64]) -> Vec<i64> {
    values.iter().map(|&v| v as i64).collect()
}

fn flags(values: &[f64]) -> Vec<bool> {
    values.iter().map(|&v| v > 0.5).collect()
}

const COEF: Tolerance = Tolerance::exact().rel(1e-11).abs(1e-13);
const DETERMINISTIC: Tolerance = Tolerance::exact().abs(1.1e-9);
const COUNTS: Tolerance = Tolerance::exact().rel(1e-14).abs(1e-16);

fn coefficients(model: &epistat_causal::gformula::FittedModel) -> &[f64] {
    &model.coefficients
}

// main: time-varying L1 = 0, L2 = 1, A = 2; baseline V = 0, W = 1.
const L1: usize = 0;
const L2: usize = 1;
const A: usize = 2;
const V: usize = 0;
const W: usize = 1;

fn main_covariates() -> Vec<Covariate> {
    use Term::{Baseline, Current, Intercept, Lag, LaggedCumulativeAverage};
    vec![
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![
                Intercept,
                Lag(L1, 1),
                LaggedCumulativeAverage(A, 1),
                Baseline(V),
                Baseline(W),
            ],
            restriction: None,
        },
        Covariate {
            kind: CovariateKind::Normal,
            terms: vec![Intercept, Lag(L2, 1), Current(L1), Lag(A, 1), Baseline(W)],
            restriction: None,
        },
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![Intercept, Lag(A, 1), Current(L1), Current(L2), Baseline(V)],
            restriction: None,
        },
    ]
}

/// `Y ~ L1 + cumavg_L2 + A + V + A:V + t0`, in R's order of the columns.
fn main_outcome_terms() -> Vec<Term> {
    use Term::{Baseline, CumulativeAverage, Current, Intercept, Period};
    vec![
        Intercept,
        Current(L1),
        CumulativeAverage(L2),
        Current(A),
        Baseline(V),
        Period,
        Term::product(Current(A), Baseline(V)),
    ]
}

fn main_competing_terms() -> Vec<Term> {
    vec![
        Term::Intercept,
        Term::Current(L1),
        Term::Baseline(W),
        Term::Period,
    ]
}

fn main_interventions() -> Vec<Intervention> {
    vec![
        Intervention::set(A, 0.0),
        Intervention::set(A, 1.0),
        Intervention {
            rules: vec![Rule {
                variable: A,
                action: Action::Dynamic(Box::new(|h: &History<'_>, _natural: f64| {
                    f64::from(u8::from(h.current(L2) > 0.5))
                })),
                periods: None,
            }],
        },
        Intervention {
            rules: vec![Rule {
                variable: A,
                action: Action::Set(1.0),
                periods: Some(vec![2, 3, 4, 5]),
            }],
        },
    ]
}

/// Runs `f` on the main data with the main models' outcome and competing event.
fn with_main<T>(
    m: &Main,
    covariates: &[Covariate],
    outcome_terms: Vec<Term>,
    competing_terms: Vec<Term>,
    interventions: &[Intervention],
    options: &GFormulaOptions,
    f: impl FnOnce(GFormula) -> T,
) -> T {
    let d = &m.data;
    let id = ids(&d.id);
    let period = periods(&d.t0);
    let baseline = [&d.v[..], &d.w[..]];
    let all = [&d.l1[..], &d.l2[..], &d.a[..]];
    let only_a = [&d.a[..]];
    let time_varying: &[&[f64]] = if covariates.len() == 3 { &all } else { &only_a };
    let event = flags(&d.y);
    let competing = flags(&d.d);
    let data = PersonPeriods {
        key: Panel::new(&id, &period).unwrap(),
        baseline: &baseline,
        time_varying,
    };
    let outcome = Outcome::Survival {
        event: &event,
        terms: outcome_terms,
        competing: Some(Competing {
            event: &competing,
            terms: competing_terms,
        }),
        censoring: None,
    };
    f(gformula(&data, covariates, &outcome, interventions, options).expect("g-formula"))
}

#[test]
fn main_models_and_observed_risk_match_gformula() {
    let r = reference();
    let m = &r.main;
    with_main(
        m,
        &main_covariates(),
        main_outcome_terms(),
        main_competing_terms(),
        &main_interventions(),
        &GFormulaOptions::new(Seed(SEED)),
        |g| {
            for (name, j) in [("L1", 0), ("L2", 1), ("A", 2)] {
                let fit = g.models.covariates[j].as_ref().expect("fitted");
                assert_close(coefficients(fit), &m.coefficients[name], COEF);
            }
            assert_close(&g.models.outcome.coefficients, &m.coefficients["Y"], COEF);
            let d = g.models.competing.as_ref().expect("competing");
            assert_close(&d.coefficients, &m.coefficients["D"], COEF);
            let rmse = g.models.covariates[1].as_ref().unwrap().rmse.unwrap();
            assert_close(&[rmse], &[m.rmse["L2"]], COEF);
            let observed: Vec<f64> = g.observed.iter().map(|o| o.unwrap()).collect();
            assert_close(&observed, &m.observed, COUNTS);
            assert_eq!(
                g.periods,
                (0..i64::try_from(m.periods).unwrap()).collect::<Vec<_>>()
            );
        },
    );
}

#[test]
fn main_risks_agree_within_monte_carlo_error() {
    let r = reference();
    let m = &r.main;
    let options = GFormulaOptions {
        simulated: Some(4 * m.simulated),
        ..GFormulaOptions::new(Seed(SEED))
    };
    with_main(
        m,
        &main_covariates(),
        main_outcome_terms(),
        main_competing_terms(),
        &main_interventions(),
        &options,
        |g| {
            assert_eq!(g.simulated, 4 * m.simulated);
            for (s, curve) in m.mc.iter().enumerate() {
                for t in 0..m.periods {
                    let ours = &g.strategies[s].value[t];
                    assert_within_standard_errors(
                        ours.estimate.unwrap(),
                        ours.monte_carlo_se.unwrap(),
                        curve.estimate[t],
                        curve.mc_se[t],
                        4.0,
                    );
                    if s > 0 {
                        let d = &g.strategies[s].difference[t];
                        assert_within_standard_errors(
                            d.estimate.unwrap(),
                            d.monte_carlo_se.unwrap(),
                            curve.estimate[t] - m.mc[0].estimate[t],
                            curve.difference_mc_se[t],
                            4.0,
                        );
                    }
                }
            }
            // The rule of strategy 4 applies from period 2 only.
            let late = &g.strategies[4];
            assert!(late.intervened_person_periods < late.intervened_persons);
        },
    );
}

#[test]
fn deterministic_risks_match_gformula() {
    use Term::{
        Baseline, CumulativeAverage, Current, Intercept, Lag, LaggedCumulativeAverage, Period,
    };
    let r = reference();
    let m = &r.main;
    let a_only = |terms| {
        vec![Covariate {
            kind: CovariateKind::Binary,
            terms,
            restriction: None,
        }]
    };
    let static_strategies = |periods: usize| {
        vec![
            Intervention::set(0, 0.0),
            Intervention::set(0, 1.0),
            Intervention {
                rules: vec![Rule {
                    variable: 0,
                    action: Action::Schedule(
                        (0..periods)
                            .map(|k| f64::from(u8::from(k % 2 == 0)))
                            .collect(),
                    ),
                    periods: None,
                }],
            },
        ]
    };
    // Y ~ A + lag1_A + cumavg_A + V + W + A:W + t0 and D ~ lag1_A + W + t0.
    with_main(
        m,
        &a_only(vec![Intercept, Lag(0, 1), Baseline(V)]),
        vec![
            Intercept,
            Current(0),
            Lag(0, 1),
            CumulativeAverage(0),
            Baseline(V),
            Baseline(W),
            Period,
            Term::product(Current(0), Baseline(W)),
        ],
        vec![Intercept, Lag(0, 1), Baseline(W), Period],
        &static_strategies(m.periods),
        &GFormulaOptions::new(Seed(SEED)),
        |g| check_exact(&g, &m.exact1, 6),
    );
    // With baseline lags and four periods: Y ~ A + lag1_A + lag2_A +
    // lag_cumavg1_A + V + t0 and D ~ lag2_A + W.
    with_main(
        m,
        &a_only(vec![Intercept, Lag(0, 1), Baseline(V)]),
        vec![
            Intercept,
            Current(0),
            Lag(0, 1),
            Lag(0, 2),
            LaggedCumulativeAverage(0, 1),
            Baseline(V),
            Period,
        ],
        vec![Intercept, Lag(0, 2), Baseline(W)],
        &static_strategies(4),
        &GFormulaOptions {
            baseline_lags: true,
            periods: Some(4),
            ..GFormulaOptions::new(Seed(SEED))
        },
        |g| check_exact(&g, &m.exact2, 4),
    );
}

fn check_exact(g: &GFormula, e: &Exact, periods: usize) {
    assert_eq!(g.periods.len(), periods);
    let a = g.models.covariates[0].as_ref().unwrap();
    assert_close(&a.coefficients, &e.coefficients["A"], COEF);
    assert_close(&g.models.outcome.coefficients, &e.coefficients["Y"], COEF);
    let d = g.models.competing.as_ref().unwrap();
    assert_close(&d.coefficients, &e.coefficients["D"], COEF);
    for s in 1..=3 {
        let risk: Vec<f64> = g.strategies[s]
            .value
            .iter()
            .map(|v| v.estimate.unwrap())
            .collect();
        assert_close(&risk, &e.risk[s], DETERMINISTIC);
    }
    let observed: Vec<f64> = g.observed.iter().map(|o| o.unwrap()).collect();
    assert_close(&observed, &e.observed, COUNTS);
}

#[test]
fn bootstrap_matches_gformula_and_covers_the_truth() {
    let r = reference();
    let m = &r.main;
    let b = m.bootstrap.replicates;
    let options = GFormulaOptions {
        bootstrap: Some(Replicates::new(b, Seed(SEED))),
        ..GFormulaOptions::new(Seed(SEED))
    };
    with_main(
        m,
        &main_covariates(),
        main_outcome_terms(),
        main_competing_terms(),
        &main_interventions(),
        &options,
        |g| {
            assert_eq!(g.failed_replicates, 0);
            let relative = (1.0 / (2.0 * (b as f64 - 1.0))).sqrt();
            for s in 0..g.strategies.len() {
                for t in 0..m.periods {
                    let v = &g.strategies[s].value[t];
                    let (se, se_r) = (v.se.unwrap(), m.bootstrap.risk_se[s][t]);
                    assert_within_standard_errors(se, se * relative, se_r, se_r * relative, 5.0);
                    assert!(v.interval.and_then(|i| i.lower).unwrap() <= v.estimate.unwrap());
                    assert!(v.estimate.unwrap() <= v.interval.and_then(|i| i.upper).unwrap());
                    if s > 0 {
                        let d = &g.strategies[s].difference[t];
                        let (se, se_r) = (d.se.unwrap(), m.bootstrap.difference_se[s][t]);
                        assert_within_standard_errors(
                            se,
                            se * relative,
                            se_r,
                            se_r * relative,
                            5.0,
                        );
                    }
                }
            }
            for (name, s) in [("never", 1), ("always", 2), ("dynamic", 3), ("from_two", 4)] {
                for (t, &truth) in m.truth[name].iter().enumerate() {
                    let v = &g.strategies[s].value[t];
                    let truth_se = (truth * (1.0 - truth) / m.truth_n).sqrt();
                    assert_within_standard_errors(
                        v.estimate.unwrap(),
                        v.se.unwrap(),
                        truth,
                        truth_se,
                        4.0,
                    );
                }
            }
            // The natural course reproduces the observed risk.
            for t in 0..m.periods {
                let v = &g.strategies[0].value[t];
                assert_within_standard_errors(
                    v.estimate.unwrap(),
                    v.se.unwrap(),
                    m.observed[t],
                    0.0,
                    4.0,
                );
            }
        },
    );
}

/// eof: time-varying L = 0 (absorbing), Z = 1; baseline V = 0.
fn eof_run(e: &Eof, binary: bool, simulated: Option<usize>) -> GFormula {
    use Term::{Baseline, CumulativeAverage, Current, Intercept, Lag};
    let d = &e.data;
    let id = ids(&d.id);
    let period = periods(&d.t0);
    let baseline = [&d.v[..]];
    let time_varying = [&d.l[..], &d.z[..]];
    let covariates = vec![
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![Intercept, Lag(1, 1)],
            restriction: Some(Restriction::Absorbing),
        },
        Covariate {
            kind: CovariateKind::Normal,
            terms: vec![Intercept, Lag(1, 1), Current(0), Baseline(0)],
            restriction: None,
        },
    ];
    let terms = vec![
        Intercept,
        Current(0),
        Current(1),
        CumulativeAverage(1),
        Baseline(0),
    ];
    let outcome = if binary {
        Outcome::BinaryEnd { value: &d.y, terms }
    } else {
        Outcome::ContinuousEnd {
            value: &d.yc,
            terms,
        }
    };
    let interventions = vec![
        Intervention {
            rules: vec![Rule {
                variable: 1,
                action: Action::Threshold {
                    lower: -0.5,
                    upper: 0.5,
                },
                periods: None,
            }],
        },
        Intervention::set(1, 0.0),
    ];
    let data = PersonPeriods {
        key: Panel::new(&id, &period).unwrap(),
        baseline: &baseline,
        time_varying: &time_varying,
    };
    let options = GFormulaOptions {
        simulated,
        ..GFormulaOptions::new(Seed(SEED))
    };
    gformula(&data, &covariates, &outcome, &interventions, &options).expect("g-formula")
}

#[test]
fn end_of_follow_up_agrees_with_gformula() {
    let r = reference();
    let e = &r.eof;
    for (binary, run, y) in [(true, &e.binary, "Y"), (false, &e.continuous, "Yc")] {
        let g = eof_run(e, binary, Some(4 * e.simulated));
        assert_eq!(g.periods, vec![3]);
        let l = g.models.covariates[0].as_ref().unwrap();
        let z = g.models.covariates[1].as_ref().unwrap();
        assert_close(&l.coefficients, &run.coefficients["L"], COEF);
        assert_close(&z.coefficients, &run.coefficients["Z"], COEF);
        assert_close(&[z.rmse.unwrap()], &[run.rmse["Z"]], COEF);
        assert_close(&g.models.outcome.coefficients, &run.coefficients[y], COEF);
        if !binary {
            assert_close(&[g.models.outcome.rmse.unwrap()], &[run.rmse[y]], COEF);
        }
        assert_close(&[g.observed[0].unwrap()], &[run.observed], COUNTS);
        for (s, point) in run.mc.iter().enumerate() {
            let v = &g.strategies[s].value[0];
            assert_within_standard_errors(
                v.estimate.unwrap(),
                v.monte_carlo_se.unwrap(),
                point.estimate,
                point.mc_se,
                4.0,
            );
            if s > 0 {
                let d = &g.strategies[s].difference[0];
                assert_within_standard_errors(
                    d.estimate.unwrap(),
                    d.monte_carlo_se.unwrap(),
                    point.estimate - run.mc[0].estimate,
                    point.difference_mc_se,
                    4.0,
                );
            }
        }
    }
}

#[test]
fn deterministic_end_of_follow_up_matches_gformula() {
    use Term::{Baseline, CumulativeAverage, Current, Intercept, Lag};
    let r = reference();
    let e = &r.eof;
    let d = &e.data;
    let id = ids(&d.id);
    let period = periods(&d.t0);
    let baseline = [&d.v[..]];
    let time_varying = [&d.z[..]];
    let data = PersonPeriods {
        key: Panel::new(&id, &period).unwrap(),
        baseline: &baseline,
        time_varying: &time_varying,
    };
    let covariates = vec![Covariate {
        kind: CovariateKind::Normal,
        terms: vec![Intercept, Lag(0, 1), Baseline(0)],
        restriction: None,
    }];
    let terms = vec![
        Intercept,
        Current(0),
        Lag(0, 1),
        CumulativeAverage(0),
        Baseline(0),
    ];
    let interventions = vec![Intervention::set(0, 0.0), Intervention::set(0, 1.0)];
    for (binary, exact, y) in [
        (true, &e.exact_binary, "Y"),
        (false, &e.exact_continuous, "Yc"),
    ] {
        let outcome = if binary {
            Outcome::BinaryEnd {
                value: &d.y,
                terms: terms.clone(),
            }
        } else {
            Outcome::ContinuousEnd {
                value: &d.yc,
                terms: terms.clone(),
            }
        };
        let g = gformula(
            &data,
            &covariates,
            &outcome,
            &interventions,
            &GFormulaOptions::new(Seed(SEED)),
        )
        .unwrap();
        let z = g.models.covariates[0].as_ref().unwrap();
        assert_close(&z.coefficients, &exact.coefficients["Z"], COEF);
        assert_close(&g.models.outcome.coefficients, &exact.coefficients[y], COEF);
        for s in 1..=2 {
            let v = g.strategies[s].value[0].estimate.unwrap();
            assert_close(&[v], &[exact.value[s]], DETERMINISTIC);
        }
    }
}

#[test]
fn two_periods_match_the_exact_g_formula() {
    use Term::{Baseline, Current, Intercept, Lag, Period};
    let r = reference();
    let t = &r.two;
    let d = &t.data;
    let id = ids(&d.id);
    let period = periods(&d.t0);
    let baseline = [&d.v[..]];
    let time_varying = [&d.l[..], &d.a[..]];
    let event = flags(&d.y);
    let competing = flags(&d.d);
    let data = PersonPeriods {
        key: Panel::new(&id, &period).unwrap(),
        baseline: &baseline,
        time_varying: &time_varying,
    };
    let covariates = vec![
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![Intercept, Lag(0, 1), Lag(1, 1), Baseline(0)],
            restriction: None,
        },
        Covariate {
            kind: CovariateKind::Binary,
            terms: vec![Intercept, Lag(1, 1), Current(0)],
            restriction: None,
        },
    ];
    let outcome = Outcome::Survival {
        event: &event,
        terms: vec![Intercept, Current(0), Current(1), Baseline(0), Period],
        competing: Some(Competing {
            event: &competing,
            terms: vec![Intercept, Current(0), Period],
        }),
        censoring: None,
    };
    let interventions = vec![
        Intervention::set(1, 0.0),
        Intervention::set(1, 1.0),
        Intervention {
            rules: vec![Rule {
                variable: 1,
                action: Action::Dynamic(Box::new(|h: &History<'_>, _| h.current(0))),
                periods: None,
            }],
        },
    ];
    let g = gformula(
        &data,
        &covariates,
        &outcome,
        &interventions,
        &GFormulaOptions {
            simulated: Some(400_000),
            ..GFormulaOptions::new(Seed(SEED))
        },
    )
    .unwrap();
    assert_close(
        &g.models.covariates[0].as_ref().unwrap().coefficients,
        &t.coefficients["L"],
        COEF,
    );
    assert_close(
        &g.models.covariates[1].as_ref().unwrap().coefficients,
        &t.coefficients["A"],
        COEF,
    );
    assert_close(&g.models.outcome.coefficients, &t.coefficients["Y"], COEF);
    assert_close(
        &g.models.competing.as_ref().unwrap().coefficients,
        &t.coefficients["D"],
        COEF,
    );
    for (name, s) in [
        ("natural", 0),
        ("never", 1),
        ("always", 2),
        ("treat_if_l", 3),
    ] {
        for (k, &exact) in t.exact[name].iter().enumerate() {
            let v = &g.strategies[s].value[k];
            assert_within_standard_errors(
                v.estimate.unwrap(),
                v.monte_carlo_se.unwrap(),
                exact,
                0.0,
                4.0,
            );
        }
    }
}
