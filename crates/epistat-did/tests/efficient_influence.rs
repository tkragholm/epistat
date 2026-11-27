//! The influence functions of the efficient ATT(g,t) and of its aggregations,
//! checked against the infinitesimal jackknife of the whole procedure on an
//! unbalanced, weighted panel whose cells use different units.
//!
//! The jackknife multiplies the sampling weight of one unit by `1 + h` and by
//! `1 - h`, reruns the estimator and the aggregation, and takes
//! `IJ_i = N (f(1 + h) - f(1 - h)) / (2 h)`, which is `N` times the derivative
//! of the estimate in `log w_i`, the quantity the influence function is.
//!
//! The derivative also contains the response of the estimated baseline weights,
//! `sum_b theta_b d a_b / d log w_i`, which the influence function leaves out
//! because it is of smaller order. The panel is built so that this term is zero:
//! the noise is centred, with the sampling weights, within each group of units
//! that share a cohort and a pattern of observed periods. Every cell is a union
//! of such groups, so each weighted group mean of the noise is zero, every
//! baseline gives the same `theta_b`, and `sum_b theta_b d a_b = theta d (sum_b
//! a_b) = 0`. The jackknife then equals the influence function up to the error
//! of the central difference.
//!
//! # The tolerance
//!
//! A weighted mean is a Moebius function of `s = log w_i` near 0, whose third
//! derivative is at most 6 times its first, so the truncation of the central
//! difference is at most `h^2` times the largest influence value; the baseline
//! weights are smooth in the weights with derivatives of the same order. Each
//! evaluation is a sum of at most `N` terms, with a rounding error at most
//! `2 N eps` times the largest outcome change, and the difference quotient
//! multiplies it by `N / h`.

#[path = "common/att_gt_rows.rs"]
#[expect(dead_code, reason = "the shared helpers serve several tests")]
mod rows;

use epistat_core::InputError;
use epistat_did::Error;
use epistat_did::att_gt::{AggteOptions, AggteType, AggteWeights, AttGt, AttGtOptions, Estimator};
use epistat_testkit::{SplitMix64, on_threads};
use rows::Obs;

const EPS: f64 = f64::EPSILON;
const H: f64 = 1e-4;

const COHORTS: [Option<i32>; 4] = [Some(2), Some(3), Some(4), None];

/// The periods each pattern observes: complete, attrition after 4 and after 3,
/// entry at 2, and a gap at 2.
const PATTERNS: [&[i32]; 5] = [
    &[1, 2, 3, 4, 5],
    &[1, 2, 3, 4],
    &[1, 2, 3],
    &[2, 3, 4, 5],
    &[1, 3, 4, 5],
];

const UNITS_PER_GROUP: usize = 6;

/// The rows and the largest absolute outcome.
fn panel(seed: u64) -> (Vec<Obs>, f64) {
    let mut rng = SplitMix64::new(seed);
    let lambda = [0.0, 0.3, 0.5, 0.9, 1.2];
    let mut rows = Vec::new();
    let mut unit_id = 0_i64;
    for cohort in COHORTS {
        for pattern in PATTERNS {
            let mut units = Vec::new();
            for _ in 0..UNITS_PER_GROUP {
                let weight = rng.uniform(0.5, 2.0);
                let alpha = rng.uniform(-3.0, 3.0);
                // A shock in period 1 enters every change from baseline 1, so the
                // cells of cohort 2 are strongly correlated.
                let shock = rng.uniform(-3.0, 3.0);
                let noise = pattern
                    .iter()
                    .map(|&t| rng.uniform(-1.0, 1.0) + if t == 1 { shock } else { 0.0 })
                    .collect::<Vec<_>>();
                units.push((unit_id, weight, alpha, noise));
                unit_id += 1;
            }
            let weight_sum = units.iter().map(|u| u.1).sum::<f64>();
            for (k, &t) in pattern.iter().enumerate() {
                let mean = units.iter().map(|u| u.1 * u.3[k]).sum::<f64>() / weight_sum;
                for (id, weight, alpha, noise) in &units {
                    let effect = match cohort {
                        Some(g) if t >= g => 1.0 + 0.5 * f64::from(t - g) + 0.25 * f64::from(g),
                        _ => 0.0,
                    };
                    let idx = usize::try_from(t - 1).expect("period");
                    rows.push(Obs {
                        weight: *weight,
                        ..Obs::new(
                            Some(*id),
                            cohort,
                            t,
                            alpha + lambda[idx] + effect + (noise[k] - mean),
                        )
                    });
                }
            }
        }
    }
    let largest = rows.iter().fold(0.0_f64, |m, r| m.max(r.outcome.abs()));
    (rows, largest)
}

fn unit_count(rows: &[Obs]) -> usize {
    let mut ids = rows.iter().filter_map(|r| r.unit).collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    ids.len()
}

fn reweighted(rows: &[Obs], unit_id: i64, factor: f64) -> Vec<Obs> {
    rows.iter()
        .map(|r| {
            let mut r = r.clone();
            if r.unit == Some(unit_id) {
                r.weight *= factor;
            }
            r
        })
        .collect()
}

fn efficient(rows: &[Obs]) -> AttGt {
    let options = AttGtOptions {
        estimator: Estimator::Efficient,
        ..AttGtOptions::default()
    };
    rows::fit(rows, &options).expect("cells")
}

/// Each summary under the equal and the treated-count weights: the event
/// times, cohorts and periods, then the overall effect, with the standard
/// error and the influence function of each.
fn aggregates(fit: &AttGt) -> Vec<(f64, f64, Vec<f64>)> {
    let mut out = Vec::new();
    for weights in [AggteWeights::Equal, AggteWeights::TreatedCount] {
        for aggregation in [
            AggteType::Dynamic,
            AggteType::Group,
            AggteType::Calendar,
            AggteType::Simple,
        ] {
            let result = fit
                .aggregate(&AggteOptions {
                    aggregation,
                    weights,
                    ..AggteOptions::default()
                })
                .expect("aggregate");
            let influence = result.influence();
            let estimates = result
                .points
                .iter()
                .map(|p| p.estimate)
                .chain(result.overall.map(|o| o.estimate));
            for (j, estimate) in estimates.enumerate() {
                out.push((
                    estimate.estimate,
                    estimate.se.expect("se"),
                    influence.column(j).to_vec(),
                ));
            }
        }
    }
    out
}

fn summaries(rows: &[Obs]) -> Vec<f64> {
    let fit = efficient(rows);
    let mut out: Vec<f64> = rows::cells(&fit).iter().map(|c| c.att).collect();
    out.extend(aggregates(&fit).iter().map(|a| a.0));
    out
}

/// The influence function and the standard error of every cell and summary.
fn reported(fit: &AttGt) -> (Vec<Option<Vec<f64>>>, Vec<f64>) {
    let mut influence: Vec<Option<Vec<f64>>> = rows::influence(fit).into_iter().map(Some).collect();
    let mut se: Vec<f64> = rows::cells(fit).iter().map(|c| c.se).collect();
    for (_, s, psi) in aggregates(fit) {
        influence.push(Some(psi));
        se.push(s);
    }
    (influence, se)
}

#[test]
fn influence_functions_match_the_jackknife_on_an_unbalanced_weighted_panel() {
    let (rows, largest_outcome) = panel(20_261_124);
    let n = unit_count(&rows);
    let n_f = n as f64;
    let fit = efficient(&rows);
    let cells = rows::cells(&fit);
    let psi = rows::influence(&fit);

    // The cells of one cohort use different units, and a cell's influence is
    // zero at every unit it does not use.
    let cohort_two = cells.iter().filter(|c| c.group == 2).collect::<Vec<_>>();
    assert!(
        cohort_two
            .windows(2)
            .any(|pair| pair[0].treated_n != pair[1].treated_n)
    );
    let efficient_cells = fit.cells().iter().filter(|c| c.estimate.is_some());
    for ((cell, column), full) in cells.iter().zip(&psi).zip(efficient_cells) {
        assert_eq!(column.len(), n);
        let nonzero = column.iter().filter(|v| **v != 0.0).count();
        assert!(nonzero <= cell.treated_n + cell.control_n);
        assert_eq!(
            full.efficient.as_ref().unwrap().baselines.len(),
            usize::try_from(cell.group - 1).unwrap()
        );
    }

    let (influence, se) = reported(&fit);
    let per_unit = (0..n)
        .map(|i| {
            let unit_id = i64::try_from(i).unwrap();
            let up = summaries(&reweighted(&rows, unit_id, 1.0 + H));
            let down = summaries(&reweighted(&rows, unit_id, 1.0 - H));
            up.iter()
                .zip(&down)
                .map(|(u, d)| n_f * (u - d) / (2.0 * H))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let jackknife = (0..influence.len())
        .map(|k| per_unit.iter().map(|unit| unit[k]).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let largest_psi = influence
        .iter()
        .flatten()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let rounding = n_f * (2.0 * n_f * EPS * 2.0 * largest_outcome) / H;
    let bound = H * H * largest_psi + rounding;
    for (k, ij) in jackknife.iter().enumerate() {
        if let Some(psi) = &influence[k] {
            for (i, (got, want)) in psi.iter().zip(ij).enumerate() {
                assert!(
                    (got - want).abs() <= bound,
                    "summary {k}, unit {i}: {got} against the jackknife's {want}, bound {bound:e}"
                );
            }
        }
        let se_ij = ij.iter().map(|v| v * v).sum::<f64>().sqrt() / n_f;
        assert!(
            (se[k] - se_ij).abs() <= bound / n_f.sqrt(),
            "summary {k}: se {} against the jackknife's {se_ij}",
            se[k]
        );
    }
}

#[test]
fn independent_combination_understates_the_variance_of_correlated_cells() {
    let (rows, _) = panel(20_261_125);
    let fit = efficient(&rows);
    let cells = rows::cells(&fit);
    let cohort_two = cells.iter().filter(|c| c.group == 2).collect::<Vec<_>>();
    let k = cohort_two.len() as f64;
    let naive = (cohort_two.iter().map(|c| c.se * c.se).sum::<f64>() / (k * k)).sqrt();
    let group = fit
        .aggregate(&AggteOptions {
            aggregation: AggteType::Group,
            weights: AggteWeights::Equal,
            ..AggteOptions::default()
        })
        .unwrap();
    let se = group.point(2).unwrap().estimate.se.unwrap();
    // The shared period-1 shock makes the cells of cohort 2 nearly perfectly
    // correlated, so the variance is close to that of a single cell, about k
    // times the independent one.
    assert!(se > 1.5 * naive, "{se} against {naive}");
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let (rows, _) = panel(7);
    let runs = on_threads(&[1, 4, 12], || {
        let (influence, se) = reported(&efficient(&rows));
        let mut bits = summaries(&rows)
            .iter()
            .chain(&se)
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        bits.extend(influence.iter().flatten().flatten().map(|v| v.to_bits()));
        bits
    });
    assert_eq!(runs[0], runs[1]);
    assert_eq!(runs[0], runs[2]);
}

#[test]
fn a_common_factor_in_the_weights_changes_nothing() {
    // The weights enter through ratios w_i / W, and multiplying every weight by
    // a power of two scales each sum exactly, so the results are bit-identical.
    let (rows, _) = panel(11);
    let scaled = rows
        .iter()
        .map(|r| Obs {
            weight: 4.0 * r.weight,
            ..r.clone()
        })
        .collect::<Vec<_>>();
    let (a, b) = (efficient(&rows), efficient(&scaled));
    assert_eq!(a.influence(), b.influence());
    for (x, y) in rows::cells(&a).iter().zip(&rows::cells(&b)) {
        assert_eq!(x.att.to_bits(), y.att.to_bits());
        assert_eq!(x.se.to_bits(), y.se.to_bits());
        assert_eq!((4.0 * x.total_weight).to_bits(), y.total_weight.to_bits());
    }
}

#[test]
fn a_weight_of_two_is_a_unit_counted_twice_in_the_estimate() {
    let (rows, largest_outcome) = panel(13);
    let n_f = unit_count(&rows) as f64;
    let doubled = rows
        .iter()
        .map(|r| Obs {
            weight: 2.0,
            ..r.clone()
        })
        .collect::<Vec<_>>();
    let copied = rows
        .iter()
        .flat_map(|r| {
            let id = r.unit.unwrap();
            [
                Obs {
                    weight: 1.0,
                    ..r.clone()
                },
                Obs {
                    unit: Some(id + 1_000_000),
                    weight: 1.0,
                    ..r.clone()
                },
            ]
        })
        .collect::<Vec<_>>();
    let a = rows::cells(&efficient(&doubled));
    let b = rows::cells(&efficient(&copied));
    let bound = 4.0 * 2.0 * n_f * EPS * 2.0 * largest_outcome;
    for (x, y) in a.iter().zip(&b) {
        assert_eq!((x.group, x.time), (y.group, y.time));
        assert!(
            (x.att - y.att).abs() <= bound,
            "{} against {}",
            x.att,
            y.att
        );
        assert_eq!(x.treated_n * 2, y.treated_n);
    }
}

#[test]
fn inconsistent_panels_are_errors() {
    let row = |id: i64, g: Option<i32>, t: i32, w: f64| Obs {
        weight: w,
        ..Obs::new(Some(id), g, t, f64::from(t))
    };
    let base = vec![
        row(1, Some(2), 1, 1.0),
        row(1, Some(2), 2, 1.0),
        row(2, None, 1, 1.0),
        row(2, None, 2, 1.0),
    ];
    let options = AttGtOptions {
        estimator: Estimator::Efficient,
        ..AttGtOptions::default()
    };
    let refused = |rows: &[Obs]| rows::fit(rows, &options).map(|_| ()).unwrap_err();

    let mut duplicate = base.clone();
    duplicate.push(row(2, None, 2, 1.0));
    assert!(matches!(
        refused(&duplicate),
        Error::Input(InputError::Duplicate { .. })
    ));

    let mut weight = base.clone();
    weight[1].weight = 2.0;
    assert_eq!(
        refused(&weight),
        Error::Input(InputError::NotConstant {
            what: "weights",
            unit: 1
        })
    );

    let mut cohort = base;
    cohort[3].cohort = Some(3);
    assert_eq!(
        refused(&cohort),
        Error::Input(InputError::NotConstant {
            what: "first_treated",
            unit: 2
        })
    );
}
