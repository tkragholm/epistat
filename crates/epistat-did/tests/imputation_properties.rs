//! Properties of the imputation and two-stage estimators that the R
//! references do not cover: results that do not depend on the number of
//! threads, rows that cannot be imputed, and the rejection of invalid input.

use epistat_core::InputError;
use epistat_did::{
    Error, ImputationOptions, ImputationTarget, ImputationTerm, Pretrends, StaggeredPanel,
    StaggeredPanelBuilder, TwoStageOptions, TwoStageSpec, imputation, two_stage,
};
use epistat_linalg::Mat;
use epistat_testkit::{BadInput, SplitMix64, assert_rejects_bad_input, assert_same_on_threads};

/// A staggered panel: `units` units over periods 1..=periods, cohorts 3, 5,
/// 7 and never, one covariate, row weights and a fifth of the rows dropped.
struct Sim {
    unit: Vec<u32>,
    time: Vec<i64>,
    g: Vec<Option<i64>>,
    y: Vec<f64>,
    w: Vec<f64>,
    x: Mat<f64>,
    cl: Vec<u32>,
}

fn simulate(units: u32, periods: i64, seed: u64) -> Sim {
    let mut rng = SplitMix64::new(seed);
    let (mut unit, mut time, mut g, mut y, mut w, mut x, mut cl) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for u in 0..units {
        let cohort = [Some(3), Some(5), Some(7), None][usize::try_from(rng.below(4)).unwrap()];
        let alpha = rng.uniform(-1.0, 1.0);
        for t in 1..=periods {
            if rng.unit() < 0.2 {
                continue;
            }
            let treated = cohort.is_some_and(|c| t >= c);
            let xv = rng.uniform(-1.0, 1.0);
            let tau = if treated {
                1.0 + 0.1 * (t - cohort.unwrap_or(0)) as f64
            } else {
                0.0
            };
            unit.push(3 * u + 1);
            time.push(t);
            g.push(cohort);
            y.push(alpha + 0.2 * t as f64 + 0.5 * xv + tau + rng.uniform(-1.0, 1.0));
            w.push(rng.uniform(0.5, 2.0));
            x.push(xv);
            cl.push(u % 7);
        }
    }
    let n = y.len();
    Sim {
        unit,
        time,
        g,
        y,
        w,
        x: Mat::from_fn(n, 1, |i, _| x[i]),
        cl,
    }
}

impl Sim {
    fn columns(&self) -> StaggeredPanelBuilder<'_> {
        StaggeredPanel::builder(&self.unit, &self.time, &self.g, &self.y)
            .weights(&self.w)
            .covariates(self.x.as_ref())
    }

    fn panel(&self) -> StaggeredPanel<'_> {
        self.columns().build().unwrap()
    }
}

/// The event study of `did2s` with reference period -1.
fn event_study() -> TwoStageOptions {
    TwoStageOptions {
        spec: TwoStageSpec::EventStudy {
            references: vec![-1],
        },
        ..TwoStageOptions::default()
    }
}

#[test]
fn results_do_not_depend_on_the_number_of_threads() {
    let sim = simulate(3000, 9, 7);
    let panel = sim.columns().clusters(&sim.cl).build().unwrap();
    assert_same_on_threads(|| {
        let config = ImputationOptions {
            target: ImputationTarget::Horizons(None),
            pretrends: Some(Pretrends::All),
            ..ImputationOptions::default()
        };
        let imp = imputation::estimate(&panel, &config).unwrap();
        let two = two_stage::estimate(&panel, &event_study()).unwrap();
        let mut out = Vec::new();
        let estimates = imp
            .effects
            .iter()
            .map(|e| e.estimate)
            .chain(
                imp.pretrends
                    .iter()
                    .flat_map(|p| p.terms.iter().map(|t| t.estimate)),
            )
            .chain(two.terms.iter().map(|t| t.estimate));
        for e in estimates.flatten() {
            out.extend([
                e.estimate,
                e.se.unwrap(),
                e.lower().unwrap(),
                e.upper().unwrap(),
            ]);
        }
        for inf in imp.influence.iter().chain(&two.influence) {
            for j in 0..inf.n_columns() {
                out.extend(inf.column(j));
            }
        }
        out
    });
}

#[test]
fn a_row_whose_effects_lie_in_different_components_is_dropped() {
    // Units 0 and 1 are untreated in periods 1 and 2 only, units 2 and 3 in
    // periods 3 and 4 only, so the untreated rows form two components. Unit
    // 0 is treated in period 3, whose effect lies in the other component: the
    // sum of the two effects is not identified and the row is dropped. Unit 2
    // is treated in period 5, which has no untreated row.
    let unit = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3];
    let time = [1, 2, 3, 1, 2, 3, 4, 5, 3, 4];
    let g = [
        Some(3),
        Some(3),
        Some(3),
        None,
        None,
        Some(5),
        Some(5),
        Some(5),
        None,
        None,
    ];
    let y = [1.0, 2.0, 5.0, 1.5, 2.5, 3.0, 4.0, 9.0, 3.5, 4.5];
    let panel = StaggeredPanel::builder(&unit, &time, &g, &y)
        .build()
        .unwrap();
    let out = imputation::estimate(&panel, &ImputationOptions::default()).unwrap();
    assert_eq!(out.dropped_rows, vec![2, 7]);
    assert_eq!(out.effects[0].estimate, None);
    assert_eq!(out.effects[0].n_rows, 0);
    assert!(out.influence.is_none());
}

#[test]
fn invalid_input_is_an_error() {
    let sim = simulate(50, 8, 3);
    let n = sim.y.len();
    let panel = sim.panel();
    let mut g = sim.g.clone();
    let first = g[0];
    let other = sim.unit.iter().position(|&u| u == sim.unit[0]).unwrap();
    let last_of_first = (0..n).rev().find(|&i| sim.unit[i] == sim.unit[0]).unwrap();
    assert_ne!(other, last_of_first);
    g[last_of_first] = if first == Some(3) { Some(5) } else { Some(3) };
    assert_eq!(
        StaggeredPanel::builder(&sim.unit, &sim.time, &g, &sim.y)
            .build()
            .unwrap_err(),
        Error::Input(InputError::NotConstant {
            what: "first_treated",
            unit: sim.unit[0]
        })
    );
    let mut cl = sim.cl.clone();
    cl[last_of_first] += 1;
    assert_eq!(
        sim.columns().clusters(&cl).build().unwrap_err(),
        Error::Input(InputError::NotConstant {
            what: "clusters",
            unit: sim.unit[0]
        })
    );
    assert_eq!(
        imputation::estimate(
            &panel,
            &ImputationOptions {
                target: ImputationTarget::Horizons(Some(vec![0, -1])),
                ..ImputationOptions::default()
            }
        )
        .unwrap_err(),
        Error::Horizon(-1)
    );
    assert_eq!(
        imputation::estimate(
            &panel,
            &ImputationOptions {
                pretrends: Some(Pretrends::Periods(vec![-2, 0])),
                ..ImputationOptions::default()
            }
        )
        .unwrap_err(),
        Error::Pretrend(0)
    );
    let untreated = (0..n)
        .find(|&i| sim.g[i].is_none_or(|g| sim.time[i] < g))
        .unwrap();
    let mut custom = vec![0.0; n];
    custom[untreated] = 1.0;
    assert_eq!(
        imputation::estimate(
            &panel,
            &ImputationOptions {
                target: ImputationTarget::Weights(vec![custom.clone()]),
                ..ImputationOptions::default()
            }
        )
        .unwrap_err(),
        Error::TargetOnUntreated {
            column: 0,
            row: untreated
        }
    );
    assert_eq!(
        imputation::estimate(
            &panel,
            &ImputationOptions {
                target: ImputationTarget::Weights(vec![custom[1..].to_vec()]),
                ..ImputationOptions::default()
            }
        )
        .unwrap_err(),
        Error::Input(InputError::Length {
            what: "target weights",
            expected: n,
            found: n - 1
        })
    );
    assert_eq!(
        two_stage::estimate(
            &panel,
            &TwoStageOptions {
                spec: TwoStageSpec::EventStudy {
                    references: vec![0]
                },
                ..TwoStageOptions::default()
            }
        )
        .unwrap_err(),
        Error::Reference(0)
    );
    let mut y = sim.y.clone();
    y[4] = f64::NAN;
    assert_eq!(
        StaggeredPanel::builder(&sim.unit, &sim.time, &sim.g, &y)
            .build()
            .unwrap_err(),
        Error::Input(InputError::NotFinite {
            what: "outcome",
            row: 4,
            column: None
        })
    );
}

#[test]
fn bad_target_weights_are_input_errors() {
    let sim = simulate(50, 8, 5);
    let panel = sim.panel();
    let treated: Vec<f64> = (0..sim.y.len())
        .map(|i| f64::from(u8::from(sim.g[i].is_some_and(|g| sim.time[i] >= g))))
        .collect();
    assert_rejects_bad_input("imputation::estimate target weights", |bad| {
        // Reversed weights fall on untreated rows, which is its own error.
        (bad != BadInput::Unsorted).then(|| {
            imputation::estimate(
                &panel,
                &ImputationOptions {
                    target: ImputationTarget::Weights(vec![bad.column(&treated)]),
                    ..ImputationOptions::default()
                },
            )
            .map(|out| out.effects.len())
        })
    });
}

#[test]
fn horizons_without_imputable_rows_have_no_estimate() {
    // Without never-treated units the periods from the last cohort's first
    // treated period on have no untreated rows, so horizons reached only
    // there have no estimate, and the requested horizon 40 has no rows.
    let mut sim = simulate(400, 9, 11);
    for g in &mut sim.g {
        if g.is_none() {
            *g = Some(7);
        }
    }
    let out = imputation::estimate(
        &sim.panel(),
        &ImputationOptions {
            target: ImputationTarget::Horizons(Some(vec![0, 1, 5, 40])),
            ..ImputationOptions::default()
        },
    )
    .unwrap();
    let estimate = |h: i64| {
        out.effects
            .iter()
            .find(|e| e.term == ImputationTerm::Horizon(h))
            .unwrap()
            .estimate
    };
    assert!(estimate(0).is_some());
    assert!(estimate(1).is_some());
    // Horizon 5 is reached by cohort 3 in period 8 only, which every unit
    // has reached by then.
    assert_eq!(estimate(5), None);
    assert_eq!(estimate(40), None);
}
