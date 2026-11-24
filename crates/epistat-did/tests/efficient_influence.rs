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

use epistat_did::{
    AttGtAggregationConfig, AttGtAggregationWeighting, AttGtObservation, EfficientDidConfig,
    EfficientDidError, EfficientDidEstimate, aggregate_efficient_att_gt_by_calendar_time,
    aggregate_efficient_att_gt_by_cohort, aggregate_efficient_att_gt_event_time_with_influence,
    aggregate_efficient_att_gt_overall, estimate_att_gt_efficient,
};
use epistat_testkit::{SplitMix64, on_threads};

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
fn panel(seed: u64) -> (Vec<AttGtObservation>, f64) {
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
                    rows.push(AttGtObservation {
                        unit_id: Some(*id),
                        first_treated_time: cohort,
                        time: t,
                        outcome: alpha + lambda[idx] + effect + (noise[k] - mean),
                        weight: *weight,
                    });
                }
            }
        }
    }
    let largest = rows.iter().fold(0.0_f64, |m, r| m.max(r.outcome.abs()));
    (rows, largest)
}

fn unit_count(rows: &[AttGtObservation]) -> usize {
    let mut ids = rows.iter().filter_map(|r| r.unit_id).collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    ids.len()
}

fn reweighted(rows: &[AttGtObservation], unit_id: i64, factor: f64) -> Vec<AttGtObservation> {
    rows.iter()
        .map(|r| {
            let mut r = *r;
            if r.unit_id == Some(unit_id) {
                r.weight *= factor;
            }
            r
        })
        .collect()
}

fn config(weighting: AttGtAggregationWeighting) -> AttGtAggregationConfig {
    AttGtAggregationConfig {
        weighting,
        ..AttGtAggregationConfig::default()
    }
}

/// Every number the jackknife differentiates, in a fixed order: the cells'
/// estimates, then the event-time, cohort, calendar and overall aggregates
/// under equal and treated-count weighting.
fn summaries(rows: &[AttGtObservation]) -> Vec<f64> {
    let cells = estimate_att_gt_efficient(rows, EfficientDidConfig::default()).expect("cells");
    let mut out = cells.iter().map(|c| c.att).collect::<Vec<_>>();
    for weighting in [
        AttGtAggregationWeighting::Equal,
        AttGtAggregationWeighting::ByTreatedCount,
    ] {
        let config = config(weighting);
        let event = aggregate_efficient_att_gt_event_time_with_influence(&cells, config).unwrap();
        out.extend(event.estimates.iter().map(|e| e.summary.estimate));
        let cohort = aggregate_efficient_att_gt_by_cohort(&cells, config).unwrap();
        out.extend(cohort.iter().map(|e| e.summary.estimate));
        let calendar = aggregate_efficient_att_gt_by_calendar_time(&cells, config).unwrap();
        out.extend(calendar.iter().map(|e| e.summary.estimate));
        out.push(
            aggregate_efficient_att_gt_overall(&cells, config)
                .unwrap()
                .estimate,
        );
    }
    out
}

/// The influence vectors and standard errors the estimator reports, in the
/// order of `summaries`.
fn reported(cells: &[EfficientDidEstimate]) -> (Vec<Option<Vec<f64>>>, Vec<f64>) {
    let mut influence = cells
        .iter()
        .map(|c| Some(c.influence_function.clone()))
        .collect::<Vec<_>>();
    let mut se = cells.iter().map(|c| c.se).collect::<Vec<_>>();
    for weighting in [
        AttGtAggregationWeighting::Equal,
        AttGtAggregationWeighting::ByTreatedCount,
    ] {
        let config = config(weighting);
        let event = aggregate_efficient_att_gt_event_time_with_influence(cells, config).unwrap();
        influence.extend(event.influence_functions.iter().cloned().map(Some));
        se.extend(event.estimates.iter().map(|e| e.summary.se));
        let cohort = aggregate_efficient_att_gt_by_cohort(cells, config).unwrap();
        influence.extend(cohort.iter().map(|_| None));
        se.extend(cohort.iter().map(|e| e.summary.se));
        let calendar = aggregate_efficient_att_gt_by_calendar_time(cells, config).unwrap();
        influence.extend(calendar.iter().map(|_| None));
        se.extend(calendar.iter().map(|e| e.summary.se));
        influence.push(None);
        se.push(
            aggregate_efficient_att_gt_overall(cells, config)
                .unwrap()
                .se,
        );
    }
    (influence, se)
}

#[test]
fn influence_functions_match_the_jackknife_on_an_unbalanced_weighted_panel() {
    let (rows, largest_outcome) = panel(20_261_124);
    let n = unit_count(&rows);
    let n_f = n as f64;
    let cells = estimate_att_gt_efficient(&rows, EfficientDidConfig::default()).unwrap();

    // The cells of one cohort use different units, and a cell's influence is
    // zero at every unit it does not use.
    let cohort_two = cells.iter().filter(|c| c.group == 2).collect::<Vec<_>>();
    assert!(
        cohort_two
            .windows(2)
            .any(|pair| pair[0].treated_n != pair[1].treated_n)
    );
    for cell in &cells {
        assert_eq!(cell.influence_function.len(), n);
        let nonzero = cell
            .influence_function
            .iter()
            .filter(|v| **v != 0.0)
            .count();
        assert!(nonzero <= cell.treated_n + cell.control_n);
        assert_eq!(
            cell.baseline_weights.len(),
            usize::try_from(cell.group - 1).unwrap()
        );
    }

    let (influence, se) = reported(&cells);
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
    let cells = estimate_att_gt_efficient(&rows, EfficientDidConfig::default()).unwrap();
    let cohort_two = cells.iter().filter(|c| c.group == 2).collect::<Vec<_>>();
    let k = cohort_two.len() as f64;
    let naive = (cohort_two.iter().map(|c| c.se * c.se).sum::<f64>() / (k * k)).sqrt();
    let summary =
        aggregate_efficient_att_gt_by_cohort(&cells, config(AttGtAggregationWeighting::Equal))
            .unwrap()
            .into_iter()
            .find(|c| c.group == 2)
            .unwrap()
            .summary;
    // The shared period-1 shock makes the cells of cohort 2 nearly perfectly
    // correlated, so the variance is close to that of a single cell, about k
    // times the independent one.
    assert!(summary.se > 1.5 * naive, "{} against {naive}", summary.se);
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let (rows, _) = panel(7);
    let runs = on_threads(&[1, 4, 12], || {
        let cells = estimate_att_gt_efficient(&rows, EfficientDidConfig::default()).unwrap();
        let (influence, se) = reported(&cells);
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
        .map(|r| AttGtObservation {
            weight: 4.0 * r.weight,
            ..*r
        })
        .collect::<Vec<_>>();
    let a = estimate_att_gt_efficient(&rows, EfficientDidConfig::default()).unwrap();
    let b = estimate_att_gt_efficient(&scaled, EfficientDidConfig::default()).unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.att.to_bits(), y.att.to_bits());
        assert_eq!(x.se.to_bits(), y.se.to_bits());
        assert_eq!(x.influence_function, y.influence_function);
        assert_eq!((4.0 * x.total_weight).to_bits(), y.total_weight.to_bits());
    }
}

#[test]
fn a_weight_of_two_is_a_unit_counted_twice_in_the_estimate() {
    let (rows, largest_outcome) = panel(13);
    let n_f = unit_count(&rows) as f64;
    let doubled = rows
        .iter()
        .map(|r| AttGtObservation { weight: 2.0, ..*r })
        .collect::<Vec<_>>();
    let copied = rows
        .iter()
        .flat_map(|r| {
            let id = r.unit_id.unwrap();
            [
                AttGtObservation { weight: 1.0, ..*r },
                AttGtObservation {
                    unit_id: Some(id + 1_000_000),
                    weight: 1.0,
                    ..*r
                },
            ]
        })
        .collect::<Vec<_>>();
    let a = estimate_att_gt_efficient(&doubled, EfficientDidConfig::default()).unwrap();
    let b = estimate_att_gt_efficient(&copied, EfficientDidConfig::default()).unwrap();
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
    let row = |id: i64, g: Option<i32>, t: i32, w: f64| AttGtObservation {
        unit_id: Some(id),
        first_treated_time: g,
        time: t,
        outcome: f64::from(t),
        weight: w,
    };
    let base = vec![
        row(1, Some(2), 1, 1.0),
        row(1, Some(2), 2, 1.0),
        row(2, None, 1, 1.0),
        row(2, None, 2, 1.0),
    ];
    let config = EfficientDidConfig::default();

    let mut duplicate = base.clone();
    duplicate.push(row(2, None, 2, 1.0));
    assert_eq!(
        estimate_att_gt_efficient(&duplicate, config),
        Err(EfficientDidError::DuplicateObservation {
            unit_id: 2,
            time: 2
        })
    );

    let mut weight = base.clone();
    weight[1].weight = 2.0;
    assert_eq!(
        estimate_att_gt_efficient(&weight, config),
        Err(EfficientDidError::TimeVaryingWeight { unit_id: 1 })
    );

    let mut cohort = base;
    cohort[3].first_treated_time = Some(3);
    assert_eq!(
        estimate_att_gt_efficient(&cohort, config),
        Err(EfficientDidError::TimeVaryingCohort { unit_id: 2 })
    );
}
