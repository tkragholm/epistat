//! Properties of the imputation and two-stage estimators that the R
//! references do not cover: results that do not depend on the number of
//! threads, rows that cannot be imputed, and the rejection of invalid input.

use epistat_core::InputError;
use epistat_did::{
    Error, ImputationConfig, ImputationError, ImputationTarget, ImputationTerm, Pretrends,
    StaggeredPanel, StaggeredPanelBuilder, TwoStageConfig, TwoStageSpec, estimate_did_imputation,
    estimate_did2s,
};
use epistat_linalg::Mat;
use epistat_testkit::{SplitMix64, assert_same_on_threads};

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

#[test]
fn results_do_not_depend_on_the_number_of_threads() {
    let sim = simulate(3000, 9, 7);
    let panel = sim.columns().clusters(&sim.cl).build().unwrap();
    assert_same_on_threads(|| {
        let config = ImputationConfig {
            pretrends: Some(Pretrends::All),
            ..ImputationConfig::event_study()
        };
        let imp = estimate_did_imputation(&panel, &config).unwrap();
        let two = estimate_did2s(&panel, &TwoStageConfig::event_study()).unwrap();
        let mut out = Vec::new();
        for e in &imp.estimates {
            out.extend(e.estimate);
            out.extend(e.std_error);
        }
        for inf in imp.influence.iter().chain(&two.influence).flatten() {
            out.extend(inf);
        }
        for t in &imp.pretrends.unwrap().terms {
            out.extend(t.estimate);
            out.extend(t.std_error);
        }
        for t in &two.terms {
            out.extend(t.estimate);
            out.extend(t.std_error);
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
    let out = estimate_did_imputation(&panel, &ImputationConfig::static_effect()).unwrap();
    assert_eq!(out.dropped_rows, vec![2, 7]);
    assert_eq!(out.estimates[0].estimate, None);
    assert_eq!(out.estimates[0].n_rows, 0);
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
        estimate_did_imputation(
            &panel,
            &ImputationConfig {
                target: ImputationTarget::Horizons(Some(&[0, -1])),
                ..ImputationConfig::static_effect()
            }
        )
        .unwrap_err(),
        ImputationError::Horizon(-1)
    );
    assert_eq!(
        estimate_did_imputation(
            &panel,
            &ImputationConfig {
                pretrends: Some(Pretrends::Periods(&[-2, 0])),
                ..ImputationConfig::static_effect()
            }
        )
        .unwrap_err(),
        ImputationError::Pretrend(0)
    );
    let untreated = (0..n)
        .find(|&i| sim.g[i].is_none_or(|g| sim.time[i] < g))
        .unwrap();
    let mut custom = vec![0.0; n];
    custom[untreated] = 1.0;
    let columns = [custom.as_slice()];
    assert_eq!(
        estimate_did_imputation(
            &panel,
            &ImputationConfig {
                target: ImputationTarget::Weights(&columns),
                ..ImputationConfig::static_effect()
            }
        )
        .unwrap_err(),
        ImputationError::TargetOnUntreated {
            column: 0,
            row: untreated
        }
    );
    assert_eq!(
        estimate_did2s(
            &panel,
            &TwoStageConfig {
                spec: TwoStageSpec::EventStudy { references: &[0] },
                ..TwoStageConfig::event_study()
            }
        )
        .unwrap_err(),
        ImputationError::Reference(0)
    );
    let mut y = sim.y.clone();
    y[4] = f64::NAN;
    assert_eq!(
        StaggeredPanel::builder(&sim.unit, &sim.time, &sim.g, &y)
            .build()
            .unwrap_err(),
        Error::Input(InputError::NotFinite {
            what: "outcome",
            row: 4
        })
    );
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
    let out = estimate_did_imputation(
        &sim.panel(),
        &ImputationConfig {
            target: ImputationTarget::Horizons(Some(&[0, 1, 5, 40])),
            ..ImputationConfig::static_effect()
        },
    )
    .unwrap();
    let estimate = |h: i64| {
        out.estimates
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
