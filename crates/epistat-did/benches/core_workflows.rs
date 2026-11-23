use std::time::Duration;

use criterion::{
    BenchmarkGroup, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use epistat_did::inference::multiplier_bootstrap_ci;
use epistat_did::{
    AttGtConfig, AttGtDrConfig, AttGtDrObservation, BootstrapConfig, DidCcConfig, DrDidConfig,
    DrDidObservation, DrDidRepeatedObservation, HonestEventStudyInput, HonestJointPathConfig,
    HonestPostFunctional, HonestSensitivity, HonestWorkflowConfig, HonestWorkflowDirectionMode,
    InferenceConfig, RelativeMagnitudeConfidenceSetConfig,
    assess_honest_event_study_directional_region_with_config,
    assess_honest_event_study_joint_path_region_with_config,
    assess_honest_event_study_post_functional_multi_flci_with_config,
    assess_honest_event_study_post_functional_with_config, assess_honest_event_study_workflow,
    estimate_att_gt_dr, estimate_did_cc_robust, estimate_did_cc_stationary,
    estimate_drdid_improved_repeated_cross_section, estimate_drdid_panel, test_did_cc_stationarity,
};

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark size fits u32"))
}

fn build_drdid_panel_rows(num_units: usize, num_covariates: usize) -> Vec<DrDidObservation> {
    (0..num_units)
        .map(|unit_index| {
            let treated = unit_index % 4 == 0;
            let unit_index_f64 = usize_to_f64(unit_index);
            let covariates = (0..num_covariates)
                .map(|covariate_index| {
                    let covariate_index_f64 = usize_to_f64(covariate_index);
                    (unit_index_f64 * (covariate_index_f64 + 1.0)).sin()
                })
                .collect::<Vec<_>>();
            let baseline = unit_index_f64.rem_euclid(23.0) * 0.1;
            let covariate_signal = covariates
                .iter()
                .enumerate()
                .map(|(covariate_index, value)| (usize_to_f64(covariate_index) + 1.0) * value)
                .sum::<f64>()
                * 0.05;
            let treatment_effect = if treated { 1.75 } else { 0.0 };
            DrDidObservation {
                treated,
                delta_outcome: baseline + covariate_signal + treatment_effect,
                weight: 1.0,
                covariates,
            }
        })
        .collect()
}

fn build_att_gt_rows(num_units: usize, start_year: i32, periods: usize) -> Vec<AttGtDrObservation> {
    let mut observations = Vec::with_capacity(num_units * periods);
    for unit_index in 0..num_units {
        let first_treated_time = match unit_index % 5 {
            0 => Some(start_year + 2),
            1 => Some(start_year + 3),
            2 => Some(start_year + 4),
            _ => None,
        };
        let baseline = usize_to_f64(unit_index % 17) * 0.2;
        for period_offset in 0..periods {
            let time = start_year + i32::try_from(period_offset).expect("period offset fits i32");
            let event_time = first_treated_time.map_or(-99, |group| time - group);
            let treated_effect = if event_time >= 0 {
                0.8 + f64::from(event_time) * 0.15
            } else {
                0.0
            };
            let trend = f64::from(time - start_year) * 0.05;
            observations.push(AttGtDrObservation {
                unit_id: i64::try_from(unit_index).ok(),
                first_treated_time,
                time,
                outcome: baseline + trend + treated_effect,
                weight: 1.0,
                covariates: vec![
                    usize_to_f64(unit_index % 7) * 0.1,
                    usize_to_f64(period_offset) * 0.05,
                ],
                comparison_cohort: None,
            });
        }
    }
    observations
}

fn build_honest_input(num_pre_periods: usize, num_post_periods: usize) -> HonestEventStudyInput {
    let total_periods = num_pre_periods + num_post_periods;
    let betahat = (0..total_periods)
        .map(|index| {
            if index < num_pre_periods {
                (usize_to_f64(index) - usize_to_f64(num_pre_periods)) * 0.05
            } else {
                usize_to_f64(index - num_pre_periods).mul_add(0.3, 0.4)
            }
        })
        .collect::<Vec<_>>();

    let covariance = (0..total_periods)
        .map(|row_index| {
            (0..total_periods)
                .map(|column_index| {
                    let distance = usize_to_f64(row_index.abs_diff(column_index));
                    if row_index == column_index {
                        usize_to_f64(row_index).mul_add(0.01, 0.16)
                    } else {
                        0.03 / (1.0 + distance)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    HonestEventStudyInput {
        betahat,
        covariance,
        pre_periods: (0..num_pre_periods)
            .map(|index| -i32::try_from(num_pre_periods - index).expect("pre index fits i32"))
            .collect(),
        post_periods: (0..num_post_periods)
            .map(|index| i32::try_from(index).expect("post index fits i32"))
            .collect(),
    }
}

fn build_did_cc_rows(num_units_per_cell: usize) -> Vec<DrDidRepeatedObservation> {
    let mut observations = Vec::with_capacity(num_units_per_cell * 4);
    for unit_index in 0..num_units_per_cell {
        let unit_index_f64 = usize_to_f64(unit_index);
        let control_x = usize_to_f64(unit_index % 9) * 0.25;
        let treated_pre_x = usize_to_f64((unit_index + 2) % 9) * 0.25;
        let treated_post_x = usize_to_f64((unit_index + 5) % 9) * 0.25;

        observations.push(DrDidRepeatedObservation {
            treated: false,
            post_period: false,
            outcome: unit_index_f64.mul_add(0.001, control_x),
            weight: 1.0,
            covariates: vec![control_x, control_x * control_x],
        });
        observations.push(DrDidRepeatedObservation {
            treated: false,
            post_period: true,
            outcome: unit_index_f64.mul_add(0.001, control_x + 0.9),
            weight: 1.0,
            covariates: vec![control_x, control_x * control_x],
        });
        observations.push(DrDidRepeatedObservation {
            treated: true,
            post_period: false,
            outcome: unit_index_f64.mul_add(0.001, treated_pre_x + 0.2),
            weight: 1.0,
            covariates: vec![treated_pre_x, treated_pre_x * treated_pre_x],
        });
        observations.push(DrDidRepeatedObservation {
            treated: true,
            post_period: true,
            outcome: unit_index_f64.mul_add(0.001, treated_post_x + 2.1 + treated_post_x * 0.6),
            weight: 1.0,
            covariates: vec![treated_post_x, treated_post_x * treated_post_x],
        });
    }
    observations
}

fn build_improved_repeated_rows(num_units_per_period: usize) -> Vec<DrDidRepeatedObservation> {
    let mut rows = Vec::with_capacity(2 * num_units_per_period);
    for post_period in [false, true] {
        for unit_index in 0..num_units_per_period {
            let unit_f = usize_to_f64(unit_index + 1);
            let x1 = (0.013 * unit_f).sin();
            let x2 = (0.021 * unit_f).cos();
            let score = 0.4f64.mul_add(x1, -(0.25 * x2)) + if post_period { 0.12 } else { -0.08 };
            let treated = score > 0.0;
            let base =
                0.7f64.mul_add(-x2, 1.2f64.mul_add(x1, 1.8)) + if post_period { 0.9 } else { 0.0 };
            let treatment_effect = if treated && post_period { 1.5 } else { 0.0 };
            let noise = (0.033 * unit_f).sin() * 0.1;
            rows.push(DrDidRepeatedObservation {
                treated,
                post_period,
                outcome: base + treatment_effect + noise,
                weight: 1.0,
                covariates: vec![1.0, x1, x2],
            });
        }
    }
    rows
}

fn bench_drdid_panel(criterion: &mut Criterion) {
    let rows = build_drdid_panel_rows(20_000, 6);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let mut group = criterion.benchmark_group("drdid_panel");
    group.throughput(Throughput::Elements(throughput));
    group.bench_function("20k_rows_6_covariates", |bencher| {
        bencher.iter(|| estimate_drdid_panel(&rows, DrDidConfig::default()).expect("panel drdid"));
    });
    group.finish();
}

fn bench_drdid_improved_repeated(criterion: &mut Criterion) {
    let rows = build_improved_repeated_rows(8_000);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let mut group = criterion.benchmark_group("drdid_improved_repeated");
    group.throughput(Throughput::Elements(throughput));
    group.bench_function("8k_per_period_3_covariates", |bencher| {
        bencher.iter(|| {
            estimate_drdid_improved_repeated_cross_section(&rows, DrDidConfig::default())
                .expect("improved repeated drdid")
        });
    });
    group.finish();
}

fn bench_att_gt_dr(criterion: &mut Criterion) {
    let rows = build_att_gt_rows(8_000, 2000, 6);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let mut group = criterion.benchmark_group("att_gt_dr");
    group.throughput(Throughput::Elements(throughput));
    group.bench_function("8k_units_6_periods", |bencher| {
        bencher.iter(|| {
            estimate_att_gt_dr(
                &rows,
                AttGtDrConfig {
                    att_gt: AttGtConfig::default(),
                    drdid: DrDidConfig::default(),
                },
            )
            .expect("att_gt dr")
        });
    });
    group.finish();
}

fn bench_honest_workflow(criterion: &mut Criterion) {
    let input = build_honest_input(3, 2);
    let functionals = vec![HonestPostFunctional::Period(0)];
    let mut group = criterion.benchmark_group("honest_workflow");
    group.throughput(Throughput::Elements(
        u64::try_from(input.post_periods.len()).expect("post period count fits u64"),
    ));
    group.bench_function("smoothness_3pre_2post", |bencher| {
        bencher.iter(|| {
            assess_honest_event_study_workflow(
                &input,
                &functionals,
                HonestSensitivity::Smoothness(0.5),
                InferenceConfig::new(0.95),
                0.0,
            )
            .expect("honest workflow")
        });
    });
    group.finish();
}

fn build_relative_magnitude_functionals(num_post_periods: usize) -> Vec<HonestPostFunctional> {
    let mid_period = i32::try_from(num_post_periods / 2).expect("mid period fits i32");
    let last_period = i32::try_from(num_post_periods.saturating_sub(1)).expect("last fits i32");
    let tail_start =
        i32::try_from(num_post_periods.saturating_sub(4)).expect("tail start fits i32");
    vec![
        HonestPostFunctional::Period(0),
        HonestPostFunctional::Period(mid_period),
        HonestPostFunctional::AverageWindow {
            start_period: 0,
            end_period: 3.min(last_period),
        },
        HonestPostFunctional::AverageWindow {
            start_period: tail_start,
            end_period: last_period,
        },
        HonestPostFunctional::Weighted(vec![(0, 0.15), (mid_period, 0.35), (last_period, 0.50)]),
    ]
}

fn configure_heavy_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

fn build_basis_directions(num_post_periods: usize) -> Vec<Vec<f64>> {
    let mut directions = vec![vec![0.0; num_post_periods]; num_post_periods];
    for (index, direction) in directions.iter_mut().enumerate() {
        direction[index] = 1.0;
    }
    directions
}

fn bench_relative_magnitude_scalar_functionals(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let mut group = criterion.benchmark_group("relative_magnitude_scalar_functionals");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        let functionals = build_relative_magnitude_functionals(num_post);
        let selected = [
            ("period0", functionals[0].clone()),
            ("mid_period", functionals[1].clone()),
            ("weighted_tail", functionals[4].clone()),
        ];
        for (label, functional) in selected {
            group.throughput(Throughput::Elements(1));
            group.bench_function(format!("{num_pre}pre_{num_post}post/{label}"), |bencher| {
                bencher.iter(|| {
                    let output = assess_honest_event_study_post_functional_with_config(
                        &input,
                        &functional,
                        HonestSensitivity::RelativeMagnitude(1.0),
                        inference,
                        0.0,
                        relative_magnitude_config,
                    )
                    .expect("relative-magnitude scalar functional");
                    std::hint::black_box(output);
                });
            });
        }
    }
    group.finish();
}

fn bench_smoothness_scalar_functionals(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let mut group = criterion.benchmark_group("smoothness_scalar_functionals");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        let functionals = build_relative_magnitude_functionals(num_post);
        let selected = [
            ("period0", functionals[0].clone()),
            ("mid_period", functionals[1].clone()),
            ("weighted_tail", functionals[4].clone()),
        ];
        for (label, functional) in selected {
            group.throughput(Throughput::Elements(1));
            group.bench_function(format!("{num_pre}pre_{num_post}post/{label}"), |bencher| {
                bencher.iter(|| {
                    let output = epistat_did::assess_honest_event_study_post_functional(
                        &input,
                        &functional,
                        HonestSensitivity::Smoothness(0.5),
                        inference,
                        0.0,
                    )
                    .expect("smoothness scalar functional");
                    std::hint::black_box(output);
                });
            });
        }
    }
    group.finish();
}

fn bench_relative_magnitude_multi_flci(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let joint = HonestJointPathConfig::default_for_production();
    let mut group = criterion.benchmark_group("relative_magnitude_multi_flci");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        let functionals = build_relative_magnitude_functionals(num_post)
            .into_iter()
            .take(3)
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(functionals.len()).expect("functional count fits u64"),
        ));
        group.bench_function(format!("{num_pre}pre_{num_post}post"), |bencher| {
            bencher.iter(|| {
                let output = assess_honest_event_study_post_functional_multi_flci_with_config(
                    &input,
                    &functionals,
                    1.0,
                    inference,
                    relative_magnitude_config,
                    joint,
                )
                .expect("relative-magnitude multi flci");
                std::hint::black_box(output);
            });
        });
    }
    group.finish();
}

fn bench_relative_magnitude_workflow(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let joint = HonestJointPathConfig::default_for_production();
    let mut group = criterion.benchmark_group("relative_magnitude_workflow");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        let functionals = build_relative_magnitude_functionals(num_post)
            .into_iter()
            .take(3)
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(
            u64::try_from(functionals.len() * num_post).expect("throughput fits u64"),
        ));
        group.bench_function(format!("{num_pre}pre_{num_post}post"), |bencher| {
            bencher.iter(|| {
                let output = assess_honest_event_study_workflow(
                    &input,
                    &functionals,
                    HonestSensitivity::RelativeMagnitude(1.0),
                    inference,
                    0.0,
                )
                .expect("relative-magnitude workflow");
                std::hint::black_box(output);
            });
        });
        group.bench_function(
            format!("{num_pre}pre_{num_post}post_basis_cfg"),
            |bencher| {
                bencher.iter(|| {
                    let output = epistat_did::assess_honest_event_study_workflow_with_config(
                        &input,
                        &functionals,
                        HonestSensitivity::RelativeMagnitude(1.0),
                        inference,
                        0.0,
                        &HonestWorkflowConfig {
                            relative_magnitude: relative_magnitude_config,
                            joint,
                            direction_mode: HonestWorkflowDirectionMode::Basis,
                        },
                    )
                    .expect("relative-magnitude configured workflow");
                    std::hint::black_box(output);
                });
            },
        );
    }
    group.finish();
}

fn bench_relative_magnitude_joint_path_region(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let joint = HonestJointPathConfig::default_for_production();
    let mut group = criterion.benchmark_group("relative_magnitude_joint_path_region");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        group.throughput(Throughput::Elements(
            u64::try_from(num_post).expect("post-period count fits u64"),
        ));
        group.bench_function(format!("{num_pre}pre_{num_post}post"), |bencher| {
            bencher.iter(|| {
                let output = assess_honest_event_study_joint_path_region_with_config(
                    &input,
                    HonestSensitivity::RelativeMagnitude(1.0),
                    inference,
                    0.0,
                    relative_magnitude_config,
                    joint,
                )
                .expect("relative-magnitude joint path region");
                std::hint::black_box(output);
            });
        });
    }
    group.finish();
}

fn bench_relative_magnitude_directional_basis_region(criterion: &mut Criterion) {
    let inference = InferenceConfig::new(0.95);
    let relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(inference);
    let joint = HonestJointPathConfig::default_for_production();
    let mut group = criterion.benchmark_group("relative_magnitude_directional_basis_region");
    configure_heavy_group(&mut group);
    for (num_pre, num_post) in [(3usize, 6usize), (4usize, 8usize)] {
        let input = build_honest_input(num_pre, num_post);
        let directions = build_basis_directions(num_post);
        group.throughput(Throughput::Elements(
            u64::try_from(directions.len()).expect("direction count fits u64"),
        ));
        group.bench_function(format!("{num_pre}pre_{num_post}post"), |bencher| {
            bencher.iter(|| {
                let output = assess_honest_event_study_directional_region_with_config(
                    &input,
                    &directions,
                    HonestSensitivity::RelativeMagnitude(1.0),
                    inference,
                    0.0,
                    relative_magnitude_config,
                    joint,
                )
                .expect("relative-magnitude directional basis region");
                std::hint::black_box(output);
            });
        });
    }
    group.finish();
}

fn bench_did_cc(criterion: &mut Criterion) {
    let rows = build_did_cc_rows(4_000);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let config = DidCcConfig::default();
    let cross_fit_config = DidCcConfig {
        drdid: DrDidConfig {
            bootstrap_reps: 19,
            max_iter: 1_000,
            tol: 1e-6,
            ..DrDidConfig::default()
        },
        cross_fit_folds: 4,
        cross_fit_seed: 17,
        ..DidCcConfig::default()
    };
    let cross_fit_rows = build_did_cc_rows(1_000);
    let mut group = criterion.benchmark_group("did_cc");
    group.throughput(Throughput::Elements(throughput));

    group.bench_function("robust_4k_per_cell", |bencher| {
        bencher.iter(|| estimate_did_cc_robust(&rows, config).expect("did_cc robust"));
    });
    group.bench_function("stationary_4k_per_cell", |bencher| {
        bencher.iter(|| estimate_did_cc_stationary(&rows, config).expect("did_cc stationary"));
    });
    group.bench_function("hausman_4k_per_cell", |bencher| {
        bencher.iter(|| test_did_cc_stationarity(&rows, config).expect("did_cc hausman"));
    });
    group.bench_function("robust_cross_fit_1k_per_cell", |bencher| {
        bencher.iter(|| {
            estimate_did_cc_robust(&cross_fit_rows, cross_fit_config)
                .expect("did_cc robust cross-fit")
        });
    });
    group.bench_function("stationary_cross_fit_1k_per_cell", |bencher| {
        bencher.iter(|| {
            estimate_did_cc_stationary(&cross_fit_rows, cross_fit_config)
                .expect("did_cc stationary cross-fit")
        });
    });
    group.bench_function("hausman_cross_fit_1k_per_cell", |bencher| {
        bencher.iter(|| {
            test_did_cc_stationarity(&cross_fit_rows, cross_fit_config)
                .expect("did_cc hausman cross-fit")
        });
    });

    group.finish();
}

fn bench_multiplier_bootstrap_ci(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("multiplier_bootstrap_ci");
    for n in [200usize, 1_000, 5_000] {
        let influence_function = (0..n)
            .map(|index| {
                let index_f = usize_to_f64(index);
                0.25f64.mul_add((0.17 * index_f).cos(), (0.03 * index_f).sin())
            })
            .collect::<Vec<_>>();
        let throughput = u64::try_from(n).expect("influence length fits u64");
        group.throughput(Throughput::Elements(throughput));
        group.bench_function(format!("n{n}_reps999"), |bencher| {
            bencher.iter(|| {
                multiplier_bootstrap_ci(
                    0.7,
                    &influence_function,
                    InferenceConfig::new(0.95),
                    BootstrapConfig {
                        reps: 999,
                        seed: 17_431,
                    },
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_drdid_panel,
    bench_drdid_improved_repeated,
    bench_att_gt_dr,
    bench_honest_workflow,
    bench_relative_magnitude_scalar_functionals,
    bench_smoothness_scalar_functionals,
    bench_relative_magnitude_multi_flci,
    bench_relative_magnitude_workflow,
    bench_relative_magnitude_joint_path_region,
    bench_relative_magnitude_directional_basis_region,
    bench_did_cc,
    bench_multiplier_bootstrap_ci,
);
criterion_main!(benches);
