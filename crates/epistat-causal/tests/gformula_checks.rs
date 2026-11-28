//! Properties of the g-formula that hold exactly, and its refusals of input it
//! cannot use.

mod gformula_sim;

use epistat_causal::gformula::Competing;
use epistat_causal::gformula::{
    Action, Covariate, CovariateKind, GFormula, GFormulaOptions, History, IntervalMethod,
    Intervention, Otherwise, Outcome, PersonPeriods, Restriction, Rule, Term, gformula,
};
use epistat_causal::{Bootstrap, Seed};
use epistat_causal::{Error, Panel};
use epistat_core::{Classify, ErrorKind};
use gformula_sim::{A, L, Sim, Z, complete, covariates, flatten, interventions, simulate};

/// The root seed of the simulation and the bootstrap of every run here.
const SEED: u64 = 20_261_124;

fn run(sim: &Sim, interventions: &[Intervention], options: &GFormulaOptions) -> GFormula {
    let baseline = sim.baseline();
    let tv = sim.time_varying();
    gformula(
        &sim.data(&baseline, &tv),
        &covariates(),
        &sim.survival(),
        interventions,
        options,
    )
    .unwrap()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn values(g: &GFormula, s: usize) -> Vec<f64> {
    g.strategies[s]
        .value
        .iter()
        .map(|e| e.estimate.unwrap())
        .collect()
}

#[test]
fn a_rule_that_changes_nothing_is_the_natural_course() {
    let sim = simulate(600, 4, 1);
    let unchanged = vec![
        Intervention {
            rules: vec![Rule {
                variable: Z,
                action: Action::Threshold {
                    lower: f64::NEG_INFINITY,
                    upper: f64::INFINITY,
                },
                periods: None,
            }],
        },
        Intervention {
            rules: vec![Rule {
                variable: A,
                action: Action::Dynamic(Box::new(|_: &History<'_>, natural| natural)),
                periods: None,
            }],
        },
    ];
    let g = run(&sim, &unchanged, &GFormulaOptions::new(SEED));
    for s in 1..=2 {
        assert_eq!(bits(&values(&g, s)), bits(&values(&g, 0)));
        assert_eq!(
            g.strategies[s].intervened_persons.to_bits(),
            0.0_f64.to_bits()
        );
        for d in &g.strategies[s].difference {
            assert_eq!(d.estimate, Some(0.0));
        }
    }
}

#[test]
fn a_constant_schedule_is_a_set_value() {
    let sim = simulate(500, 4, 2);
    let g = run(
        &sim,
        &[
            Intervention::set(A, 1.0),
            Intervention {
                rules: vec![Rule {
                    variable: A,
                    action: Action::Schedule(vec![1.0; 4]),
                    periods: None,
                }],
            },
        ],
        &GFormulaOptions::new(SEED),
    );
    assert_eq!(bits(&values(&g, 1)), bits(&values(&g, 2)));
    assert!(g.strategies[1].intervened_persons > 0.0);
}

#[test]
fn fewer_periods_give_the_first_periods_of_the_longer_run() {
    // Every person draws from its own stream, the same number of draws in each
    // period, so the first periods do not depend on how many follow.
    let sim = simulate(500, 4, 3);
    for simulated in [None, Some(1_100)] {
        let long = run(
            &sim,
            &interventions(),
            &GFormulaOptions {
                simulated,
                ..GFormulaOptions::new(SEED)
            },
        );
        let short = run(
            &sim,
            &interventions(),
            &GFormulaOptions {
                simulated,
                periods: Some(2),
                ..GFormulaOptions::new(SEED)
            },
        );
        assert_eq!(short.periods, vec![0, 1]);
        for s in 0..long.strategies.len() {
            assert_eq!(bits(&values(&short, s)), bits(&values(&long, s)[..2]));
        }
        assert_eq!(short.observed, long.observed[..2]);
    }
}

#[test]
fn the_seed_moves_only_simulated_values() {
    let sim = simulate(400, 4, 4);
    let a = run(&sim, &interventions(), &GFormulaOptions::new(SEED));
    let b = run(
        &sim,
        &interventions(),
        &GFormulaOptions {
            seed: 99,
            ..GFormulaOptions::new(SEED)
        },
    );
    assert_eq!(
        bits(&flatten(&a)),
        bits(&flatten(&run(
            &sim,
            &interventions(),
            &GFormulaOptions::new(SEED)
        )))
    );
    assert_ne!(values(&a, 0), values(&b, 0));
    assert_eq!(a.models, b.models);
    // Period 0 has no simulated covariates.
    for s in 0..a.strategies.len() {
        assert_eq!(values(&a, s)[0].to_bits(), values(&b, s)[0].to_bits());
    }
}

#[test]
fn bootstrap_limits_follow_the_method() {
    let sim = simulate(400, 3, 6);
    let options = GFormulaOptions {
        bootstrap: Some(Bootstrap::new(30, Seed(SEED))),
        interval: IntervalMethod::Normal,
        ..GFormulaOptions::new(SEED)
    };
    let g = run(&sim, &interventions(), &options);
    let z = epistat_stats::dist::normal::quantile(0.975);
    for s in &g.strategies {
        for e in s.value.iter().chain(&s.difference).chain(&s.ratio) {
            let (x, se) = (e.estimate.unwrap(), e.se.unwrap());
            assert!((e.lower.unwrap() - (x - z * se)).abs() <= 1e-15);
            assert!((e.upper.unwrap() - (x + z * se)).abs() <= 1e-15);
        }
    }
    let p = run(
        &sim,
        &interventions(),
        &GFormulaOptions {
            interval: IntervalMethod::Percentile,
            ..options
        },
    );
    // The same replicates: the standard errors agree, the limits do not.
    assert_eq!(g.strategies[1].value[2].se, p.strategies[1].value[2].se);
    assert_ne!(
        g.strategies[1].value[2].lower,
        p.strategies[1].value[2].lower
    );
    // The reference's own difference is 0 in every replicate.
    assert_eq!(g.strategies[0].difference[2].se, Some(0.0));
}

#[test]
fn restrictions_fit_only_where_they_apply() {
    let sim = simulate(500, 4, 7);
    let g = run(&sim, &interventions(), &GFormulaOptions::new(SEED));
    // L is absorbing: its model is fitted to the rows after period 0 whose
    // previous L is 0.
    let absorbing_rows = (1..sim.id.len())
        .filter(|&r| sim.period[r] > 0 && sim.l[r - 1] == 0.0)
        .count();
    assert_eq!(
        g.models.covariates[L].as_ref().unwrap().rows,
        absorbing_rows
    );
    // Z modelled only where its previous value is below 1, 0 elsewhere.
    let mut c = covariates();
    c[Z].restriction = Some(Restriction::When {
        applies: Box::new(|h: &History<'_>| h.lag(Z, 1) < 1.0),
        otherwise: Otherwise::Value(0.0),
    });
    let baseline = sim.baseline();
    let tv = sim.time_varying();
    let g = gformula(
        &sim.data(&baseline, &tv),
        &c,
        &sim.survival(),
        &interventions(),
        &GFormulaOptions::new(SEED),
    )
    .unwrap();
    let rows = (0..sim.id.len())
        .filter(|&r| sim.period[r] > 0 && sim.z[r - 1] < 1.0)
        .count();
    assert_eq!(g.models.covariates[Z].as_ref().unwrap().rows, rows);
}

#[test]
fn an_outcome_at_the_end_reports_one_period() {
    let sim = complete(&simulate(800, 3, 8), 3);
    let baseline = sim.baseline();
    let tv = sim.time_varying();
    let g = gformula(
        &sim.data(&baseline, &tv),
        &covariates(),
        &sim.binary_end(),
        &interventions(),
        &GFormulaOptions::new(SEED),
    )
    .unwrap();
    assert_eq!(g.periods, vec![2]);
    assert!(g.models.competing.is_none());
    for s in &g.strategies {
        let v = s.value[0].estimate.unwrap();
        assert!(v > 0.0 && v < 1.0);
    }
}

fn expect_input(result: Result<GFormula, Error>, needle: &str) {
    match result {
        Err(error) if error.kind() == ErrorKind::Input => {
            let message = error.to_string();
            assert!(
                message.contains(needle),
                "{message:?} does not mention {needle:?}"
            );
        }
        other => panic!("expected an input error about {needle:?}, got {other:?}"),
    }
}

#[test]
fn unusable_input_is_refused() {
    let sim = simulate(300, 3, 9);
    let baseline = sim.baseline();
    let tv = sim.time_varying();
    let data = sim.data(&baseline, &tv);
    let outcome = sim.survival();
    let ok = |c: &[Covariate], o: &Outcome<'_>, i: &[Intervention], opt: &GFormulaOptions| {
        gformula(&data, c, o, i, opt)
    };
    let default = GFormulaOptions::new(SEED);

    // A model reading a later variable of the same period.
    let mut c = covariates();
    c[L].terms.push(Term::Current(Z));
    expect_input(ok(&c, &outcome, &[], &default), "simulated after it");
    let mut c = covariates();
    c[Z].terms.push(Term::CumulativeAverage(Z));
    expect_input(ok(&c, &outcome, &[], &default), "simulated after it");
    let mut c = covariates();
    c[Z].terms.push(Term::Lag(A, 0));
    expect_input(ok(&c, &outcome, &[], &default), "at least 1");
    let mut c = covariates();
    c[A].terms.push(Term::Baseline(3));
    expect_input(ok(&c, &outcome, &[], &default), "does not exist");
    let mut c = covariates();
    c[Z].restriction = Some(Restriction::Absorbing);
    expect_input(ok(&c, &outcome, &[], &default), "absorbing");

    // Interventions.
    expect_input(
        ok(
            &covariates(),
            &outcome,
            &[Intervention::set(5, 1.0)],
            &default,
        ),
        "does not exist",
    );
    let two = Intervention {
        rules: vec![
            Rule {
                variable: A,
                action: Action::Set(1.0),
                periods: None,
            },
            Rule {
                variable: A,
                action: Action::Set(0.0),
                periods: None,
            },
        ],
    };
    expect_input(ok(&covariates(), &outcome, &[two], &default), "two rules");
    let short = Intervention {
        rules: vec![Rule {
            variable: A,
            action: Action::Schedule(vec![1.0]),
            periods: None,
        }],
    };
    expect_input(ok(&covariates(), &outcome, &[short], &default), "schedule");
    let nan = Intervention {
        rules: vec![Rule {
            variable: A,
            action: Action::Dynamic(Box::new(|_: &History<'_>, _| f64::NAN)),
            periods: None,
        }],
    };
    expect_input(ok(&covariates(), &outcome, &[nan], &default), "not finite");
    expect_input(
        ok(
            &covariates(),
            &outcome,
            &[],
            &GFormulaOptions {
                reference: 1,
                ..default
            },
        ),
        "reference",
    );
    expect_input(
        ok(
            &covariates(),
            &outcome,
            &[],
            &GFormulaOptions {
                periods: Some(9),
                ..default
            },
        ),
        "periods",
    );
    expect_input(
        ok(
            &covariates(),
            &outcome,
            &[],
            &GFormulaOptions {
                simulated: Some(0),
                ..default
            },
        ),
        "simulated",
    );

    // The data.
    let mut gap = sim.period.clone();
    let r = (1..gap.len()).find(|&r| gap[r] == 1).unwrap();
    for k in r..gap.len() {
        if sim.id[k] != sim.id[r] {
            break;
        }
        gap[k] += 1;
    }
    let d = PersonPeriods {
        key: Panel::new(&sim.id, &gap).unwrap(),
        ..data
    };
    expect_input(
        gformula(&d, &covariates(), &outcome, &[], &default),
        "consecutive",
    );
    let mut late_id = sim.id.clone();
    let last = late_id.len() - 1;
    late_id[last] = late_id[0];
    // A person whose rows are not together is not a sorted key.
    assert!(Panel::new(&late_id, &sim.period).is_err());
    let mut late_start = sim.period.clone();
    for k in late_start.iter_mut().take(3) {
        *k += 1;
    }
    let d = PersonPeriods {
        key: Panel::new(&sim.id, &late_start).unwrap(),
        ..data
    };
    expect_input(
        gformula(&d, &covariates(), &outcome, &[], &default),
        "not at period 0",
    );
    let mut binary = sim.l.clone();
    binary[3] = 0.5;
    let tv_bad = [&binary[..], &sim.z[..], &sim.a[..]];
    let d = PersonPeriods {
        time_varying: &tv_bad,
        ..data
    };
    expect_input(
        gformula(&d, &covariates(), &outcome, &[], &default),
        "{0, 1}",
    );
    let mut v = sim.v.clone();
    let r = (1..v.len()).find(|&r| sim.period[r] == 1).unwrap();
    v[r] = 1.0 - v[r];
    let b_bad = [&v[..]];
    let d = PersonPeriods {
        baseline: &b_bad,
        ..data
    };
    expect_input(
        gformula(&d, &covariates(), &outcome, &[], &default),
        "not constant within",
    );
    let mut after = sim.y.clone();
    let r = (0..after.len() - 1)
        .find(|&r| sim.id[r] == sim.id[r + 1] && !sim.d[r])
        .unwrap();
    after[r] = true;
    let o = Outcome::Survival {
        event: &after,
        terms: gformula_sim::outcome_terms(),
        competing: Some(Competing {
            event: &sim.d,
            terms: gformula_sim::competing_terms(),
        }),
        censoring: None,
    };
    expect_input(
        gformula(&data, &covariates(), &o, &[], &default),
        "after an event",
    );
    let wrong = vec![Covariate {
        kind: CovariateKind::Binary,
        terms: vec![Term::Intercept],
        restriction: None,
    }];
    expect_input(
        gformula(&data, &wrong, &outcome, &[], &default),
        "covariates",
    );
}

/// The simulated courses are those the estimates are means of, and keeping
/// them changes no estimate.
#[test]
fn the_simulated_courses_give_the_estimates() {
    let sim = simulate(700, 4, 9);
    let kept = run(
        &sim,
        &interventions(),
        &GFormulaOptions {
            simulated: Some(900),
            keep_simulated: true,
            ..GFormulaOptions::new(SEED)
        },
    );
    let plain = run(
        &sim,
        &interventions(),
        &GFormulaOptions {
            simulated: Some(900),
            ..GFormulaOptions::new(SEED)
        },
    );
    assert!(plain.courses.is_none());
    let courses = kept.courses.clone().expect("kept");
    let mut without = kept.clone();
    without.courses = None;
    assert_eq!(without, plain);
    assert_eq!(courses.len(), plain.strategies.len());
    let periods = plain.periods.len();
    for (s, c) in courses.iter().enumerate() {
        assert_eq!(c.person.len(), 900 * periods);
        assert_eq!(c.values.len(), sim.time_varying().len());
        for (k, want) in values(&plain, s).iter().enumerate() {
            let at: Vec<f64> = (0..c.person.len())
                .filter(|&r| c.period[r] == i64::try_from(k).unwrap())
                .map(|r| c.value[r].expect("a survival risk is defined at every period"))
                .collect();
            let mean = at.iter().sum::<f64>() / at.len() as f64;
            // A mean of 900 values in [0, 1] by two summation orders differs by
            // at most 2 * 900 eps.
            assert!((mean - want).abs() <= 1800.0 * f64::EPSILON, "{s} {k}");
        }
    }
    // Under "always treat", the second intervention, the treatment is 1 in
    // every period, period 0 included.
    let c = &courses[2];
    assert!((0..c.person.len()).all(|r| c.values[A][r].to_bits() == 1.0_f64.to_bits()));
}
