use std::collections::BTreeMap;

use crate::types::{AttGtAggregationConfig, AttGtAggregationWeighting, BasePeriod, DrDidConfig};

use super::*;

fn row(g: Option<i32>, t: i32, y: f64) -> AttGtObservation {
    AttGtObservation::new(g, t, y)
}

fn panel_row(id: i64, g: Option<i32>, t: i32, y: f64) -> AttGtObservation {
    AttGtObservation::with_unit_id(id, g, t, y)
}

fn row_dr(g: Option<i32>, t: i32, y: f64, x: f64) -> AttGtDrObservation {
    AttGtDrObservation {
        covariates: vec![x],
        ..AttGtDrObservation::new(g, t, y)
    }
}

#[test]
fn estimates_group_time_att_with_never_treated_controls() {
    let mut rows = Vec::new();
    for _ in 0..50 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            rows.push(row(None, t, 10.0 + t_f));
            rows.push(row(Some(2), t, 10.0 + t_f + if t >= 2 { 2.0 } else { 0.0 }));
            rows.push(row(Some(3), t, 10.0 + t_f + if t >= 3 { 1.0 } else { 0.0 }));
        }
    }

    let estimates = estimate_att_gt(
        &rows,
        AttGtConfig {
            base_period: BasePeriod::Universal,
            ..AttGtConfig::default()
        },
    )
    .expect("estimate att(g,t)");
    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(att_map.len(), 6);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 1e-12);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 1e-12);
    assert!((att_map[&(2, 4)] - 2.0).abs() < 1e-12);
    assert!(att_map[&(3, 1)].abs() < 1e-12);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 1e-12);
    assert!((att_map[&(3, 4)] - 1.0).abs() < 1e-12);
}

#[test]
fn rejects_missing_never_treated_group() {
    let rows = vec![row(Some(2), 1, 10.0), row(Some(2), 2, 12.0)];
    let err = estimate_att_gt(&rows, AttGtConfig::default()).expect_err("must fail");
    assert_eq!(err, AttGtError::MissingNeverTreatedGroup);
}

#[test]
fn strict_mode_errors_on_missing_cells() {
    let rows = vec![
        row(None, 1, 11.0),
        row(None, 2, 12.0),
        row(Some(2), 1, 10.0),
        // missing treated_post at t=2
    ];
    let err = estimate_att_gt(
        &rows,
        AttGtConfig {
            base_period: BasePeriod::Universal,
            skip_incomplete_pairs: false,
            ..AttGtConfig::default()
        },
    )
    .expect_err("must fail");
    assert!(matches!(err, AttGtError::MissingCell { cell, .. } if cell == "treated_post"));
}

#[test]
fn estimates_group_time_att_with_not_yet_treated_controls() {
    let mut rows = Vec::new();
    for _ in 0..50 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            rows.push(row(Some(2), t, 10.0 + t_f + if t >= 2 { 2.0 } else { 0.0 }));
            rows.push(row(Some(3), t, 10.0 + t_f + if t >= 3 { 1.0 } else { 0.0 }));
            rows.push(row(Some(4), t, 10.0 + t_f));
        }
    }
    let estimates = estimate_att_gt(
        &rows,
        AttGtConfig {
            base_period: BasePeriod::Universal,
            comparison_group: ComparisonGroup::NotYetTreated,
            ..AttGtConfig::default()
        },
    )
    .expect("estimate att(g,t)");
    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(att_map.len(), 6);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 1e-12);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 1e-12);
    assert!((att_map[&(3, 1)] - (2.0 / 3.0)).abs() < 1e-12);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 1e-12);
    assert!((att_map[&(4, 1)] - 1.0).abs() < 1e-12);
    assert!((att_map[&(4, 2)] - 0.5).abs() < 1e-12);
}

#[test]
fn uses_panel_unit_differences_for_standard_errors_when_ids_are_present() {
    let rows = vec![
        panel_row(1, Some(2), 1, 10.0),
        panel_row(1, Some(2), 2, 13.0),
        panel_row(2, Some(2), 1, 8.0),
        panel_row(2, Some(2), 2, 12.0),
        panel_row(3, None, 1, 7.0),
        panel_row(3, None, 2, 8.0),
        panel_row(4, None, 1, 9.0),
        panel_row(4, None, 2, 11.0),
    ];

    let estimates = estimate_att_gt(&rows, AttGtConfig::default()).expect("panel att(g,t)");
    let estimate = estimates
        .iter()
        .find(|estimate| estimate.group == 2 && estimate.time == 2)
        .expect("g=2,t=2 estimate");

    assert!((estimate.att - 2.0).abs() < 1e-12);
    assert!((estimate.se - 0.5).abs() < 1e-12);
}

#[test]
fn aggregation_helpers_work() {
    let estimates = vec![
        AttGtEstimate {
            group: 2,
            time: 2,
            event_time: 0,
            att: 2.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 10,
            control_n: 10,
            total_weight: 20.0,
        },
        AttGtEstimate {
            group: 2,
            time: 3,
            event_time: 1,
            att: 2.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 10,
            control_n: 10,
            total_weight: 20.0,
        },
        AttGtEstimate {
            group: 2,
            time: 4,
            event_time: 2,
            att: 2.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 10,
            control_n: 10,
            total_weight: 20.0,
        },
        AttGtEstimate {
            group: 3,
            time: 3,
            event_time: 0,
            att: 1.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 10,
            control_n: 10,
            total_weight: 20.0,
        },
        AttGtEstimate {
            group: 3,
            time: 4,
            event_time: 1,
            att: 1.0,
            se: 0.1,
            ci_low: 0.0,
            ci_high: 0.0,
            treated_n: 10,
            control_n: 10,
            total_weight: 20.0,
        },
    ];

    let cfg = AttGtAggregationConfig::default();
    // One unit in each cohort: equal cohort shares, so treated-count weighting
    // weights every cell equally.
    let influence = vec![vec![0.1, -0.1]; estimates.len()];
    let panel = UnitPanel::unweighted(vec![Some(2), Some(3)]);
    let by_event = aggregate_att_gt_event_time_with_influence(&estimates, &influence, cfg)
        .expect("event aggregation");
    let event_map = by_event
        .estimates
        .into_iter()
        .map(|entry| (entry.event_time, entry.summary.estimate))
        .collect::<BTreeMap<_, _>>();
    assert!((event_map[&0] - 1.5).abs() < 1e-12);
    assert!((event_map[&1] - 1.5).abs() < 1e-12);
    assert!((event_map[&2] - 2.0).abs() < 1e-12);

    let by_group = aggregate_att_gt_by_cohort(&estimates, &influence, &panel, cfg)
        .expect("cohort aggregation");
    let group_map = by_group
        .into_iter()
        .map(|entry| (entry.group, entry.summary.estimate))
        .collect::<BTreeMap<_, _>>();
    assert!((group_map[&2] - 2.0).abs() < 1e-12);
    assert!((group_map[&3] - 1.0).abs() < 1e-12);

    let by_time = aggregate_att_gt_by_calendar_time(&estimates, &influence, &panel, cfg)
        .expect("calendar aggregation");
    let time_map = by_time
        .into_iter()
        .map(|entry| (entry.time, entry.summary.estimate))
        .collect::<BTreeMap<_, _>>();
    assert!((time_map[&2] - 2.0).abs() < 1e-12);
    assert!((time_map[&3] - 1.5).abs() < 1e-12);
    assert!((time_map[&4] - 1.5).abs() < 1e-12);

    let overall =
        aggregate_att_gt_overall(&estimates, &influence, &panel, cfg).expect("overall aggregation");
    assert!((overall.estimate - 1.6).abs() < 1e-12);

    // `total_weight` is the sum of the components' total weight under every
    // weighting and on the route with influence functions as well.
    for weighting in [
        AttGtAggregationWeighting::Equal,
        AttGtAggregationWeighting::ByTreatedCount,
        AttGtAggregationWeighting::ByTotalWeight,
    ] {
        let cfg = AttGtAggregationConfig {
            weighting,
            ..AttGtAggregationConfig::default()
        };
        let overall = aggregate_att_gt_overall(&estimates, &influence, &panel, cfg)
            .expect("overall aggregation");
        assert_eq!(overall.total_weight.to_bits(), 100.0_f64.to_bits());
        for entry in aggregate_att_gt_by_cohort(&estimates, &influence, &panel, cfg)
            .expect("cohort aggregation")
        {
            let want: f64 = if entry.group == 2 { 60.0 } else { 40.0 };
            assert_eq!(entry.summary.total_weight.to_bits(), want.to_bits());
        }
    }
}

#[test]
fn estimates_dr_group_time_att_with_never_treated_controls() {
    let mut rows = Vec::new();
    for _ in 0..40 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            for x in [0.0, 1.0] {
                let base = 0.5f64.mul_add(x, 10.0 + t_f);
                rows.push(row_dr(None, t, base, x));
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }

    let estimates = estimate_att_gt_dr_with_influence(
        &rows,
        AttGtDrConfig {
            att_gt: AttGtConfig {
                base_period: BasePeriod::Universal,
                ..AttGtConfig::default()
            },
            drdid: DrDidConfig::builder()
                .bootstrap_reps(49)
                .bootstrap_seed(101)
                .build(),
        },
    )
    .map(|out| out.estimates)
    .expect("dr att(g,t)");
    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(att_map.len(), 6);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 0.2);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 0.2);
    assert!((att_map[&(2, 4)] - 2.0).abs() < 0.2);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 0.2);
    assert!((att_map[&(3, 4)] - 1.0).abs() < 0.2);
}

#[test]
fn estimates_dr_group_time_att_with_not_yet_treated_controls() {
    let mut rows = Vec::new();
    for _ in 0..40 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            for x in [0.0, 1.0] {
                let base = 0.4f64.mul_add(x, 10.0 + t_f);
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(4), t, base, x));
            }
        }
    }

    let estimates = estimate_att_gt_dr_with_influence(
        &rows,
        AttGtDrConfig {
            att_gt: AttGtConfig::builder()
                .comparison_group(ComparisonGroup::NotYetTreated)
                .base_period(BasePeriod::Universal)
                .build(),
            drdid: DrDidConfig::builder()
                .bootstrap_reps(49)
                .bootstrap_seed(102)
                .build(),
        },
    )
    .map(|out| out.estimates)
    .expect("dr att(g,t)");
    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    // Four cells, not six. Every unit here is eventually treated (groups 2, 3
    // and 4; there is no never-treated arm), and a not-yet-treated control must
    // be untreated at the LATER of the two periods a cell compares. That empties
    // the cells whose later period is 4, since no group has G > 4:
    //   (2,4) base 1 -> needs G > 4   gone
    //   (3,4) base 2 -> needs G > 4   gone
    // and all of group 4's own cells, whose base is 3, need G > 3 with only
    // group 4 itself above it. What survives is (2,2), (2,3), (3,1), (3,3).
    // Six was the count under the older rule that judged eligibility at `time`
    // alone, which admitted units already treated by the baseline period.
    assert_eq!(att_map.len(), 4);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 0.2);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 0.2);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 0.2);
    // The surviving pre-treatment placebo, which must be flat.
    assert!(att_map[&(3, 1)].abs() < 0.2);
}

#[test]
fn dr_group_time_rejects_inconsistent_covariates() {
    let rows = vec![
        row_dr(None, 1, 10.0, 0.0),
        AttGtDrObservation {
            covariates: vec![0.0, 1.0],
            ..AttGtDrObservation::new(Some(2), 1, 10.0)
        },
    ];
    let err = estimate_att_gt_dr_with_influence(&rows, AttGtDrConfig::default())
        .map(|out| out.estimates)
        .expect_err("must fail");
    assert_eq!(
        err,
        AttGtError::InconsistentCovariateCount {
            expected: 1,
            actual: 2
        }
    );
}

#[test]
fn dr_with_influence_outputs_aligned_vectors() {
    let mut rows = Vec::new();
    for _ in 0..20 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            for x in [0.0, 1.0] {
                let base = 0.5f64.mul_add(x, 10.0 + t_f);
                rows.push(row_dr(None, t, base, x));
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }

    let out = estimate_att_gt_dr_with_influence(
        &rows,
        AttGtDrConfig {
            drdid: DrDidConfig::builder()
                .bootstrap_reps(49)
                .bootstrap_seed(111)
                .build(),
            ..AttGtDrConfig::default()
        },
    )
    .expect("dr with influence");

    assert_eq!(out.estimates.len(), out.influence_functions.len());
    let n = rows.len();
    assert!(
        out.influence_functions
            .iter()
            .all(|influence| influence.len() == n)
    );
    assert!(
        out.influence_functions
            .iter()
            .all(|influence| influence.iter().any(|value| value.abs() > 0.0))
    );
}

#[test]
fn or_with_influence_outputs_aligned_vectors() {
    let mut rows = Vec::new();
    for _ in 0..20 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            for x in [0.0, 1.0] {
                let base = 0.75f64.mul_add(x, 20.0 + t_f);
                rows.push(row_dr(None, t, base, x));
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }
    let out = estimate_att_gt_or_with_influence(&rows, AttGtDrConfig::default())
        .expect("or with influence");
    assert_eq!(out.estimates.len(), out.influence_functions.len());
    assert!(
        out.influence_functions
            .iter()
            .all(|influence| influence.len() == rows.len())
    );
}

#[test]
fn estimates_or_group_time_att() {
    let mut rows = Vec::new();
    for _ in 0..50 {
        for t in 1..=4 {
            let t_f = f64::from(t);
            for x in [0.0, 1.0] {
                let base = 0.75f64.mul_add(x, 20.0 + t_f);
                rows.push(row_dr(None, t, base, x));
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }

    let estimates = estimate_att_gt_or_with_influence(
        &rows,
        AttGtDrConfig {
            att_gt: AttGtConfig {
                base_period: BasePeriod::Universal,
                ..AttGtConfig::default()
            },
            ..AttGtDrConfig::default()
        },
    )
    .map(|out| out.estimates)
    .expect("or att(g,t)");
    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(att_map.len(), 6);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 0.15);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 0.15);
    assert!((att_map[&(2, 4)] - 2.0).abs() < 0.15);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 0.15);
    assert!((att_map[&(3, 4)] - 1.0).abs() < 0.15);
}

#[test]
fn estimates_ipw_group_time_att() {
    let mut rows = Vec::new();
    for i in 0..800 {
        let x = if i % 2 == 0 { 0.0 } else { 1.0 };
        let noise = if i % 5 == 0 { -0.1 } else { 0.1 };
        for t in 1..=4 {
            let t_f = f64::from(t);
            let base = 0.8f64.mul_add(x, 10.0 + t_f) + noise;
            rows.push(row_dr(None, t, base, x));
            if i % 4 != 0 {
                rows.push(row_dr(Some(2), t, base + if t >= 2 { 2.0 } else { 0.0 }, x));
            }
            if i % 3 == 0 {
                rows.push(row_dr(Some(3), t, base + if t >= 3 { 1.0 } else { 0.0 }, x));
            }
        }
    }

    let estimates = estimate_att_gt_ipw_with_influence(
        &rows,
        AttGtDrConfig {
            att_gt: AttGtConfig {
                base_period: BasePeriod::Universal,
                ..AttGtConfig::default()
            },
            drdid: DrDidConfig::builder()
                .propensity_clip(0.01)
                .max_iter(300)
                .tol(1e-8)
                .build(),
        },
    )
    .map(|out| out.estimates)
    .expect("ipw att(g,t)");

    let att_map = estimates
        .iter()
        .map(|est| ((est.group, est.time), est.att))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(att_map.len(), 6);
    assert!((att_map[&(2, 2)] - 2.0).abs() < 0.35);
    assert!((att_map[&(2, 3)] - 2.0).abs() < 0.35);
    assert!((att_map[&(2, 4)] - 2.0).abs() < 0.35);
    assert!((att_map[&(3, 3)] - 1.0).abs() < 0.35);
    assert!((att_map[&(3, 4)] - 1.0).abs() < 0.35);
}
