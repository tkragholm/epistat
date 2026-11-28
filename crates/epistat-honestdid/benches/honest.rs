//! Timings of the HonestDiD paths on simulated event studies: the
//! relative-magnitude and smoothness summaries of single targets, the joint
//! path, the region over the unit directions and the joint FLCI.

use std::time::Duration;

use criterion::{
    BenchmarkGroup, Criterion, SamplingMode, criterion_group, criterion_main, measurement::WallTime,
};
use epistat_core::{Bootstrap, Covariance, EventStudy, Seed};
use epistat_honestdid::honest::{
    self, Calibration, Directions, Grid, HonestOptions, JointMethod, JointPathOptions, Restriction,
    RmBound, Target,
};

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark size fits u32"))
}

fn event_study(n_pre: usize, n_post: usize) -> EventStudy {
    let k = n_pre + n_post;
    let coefficients = (0..k)
        .map(|index| {
            if index < n_pre {
                (usize_to_f64(index) - usize_to_f64(n_pre)) * 0.05
            } else {
                usize_to_f64(index - n_pre).mul_add(0.3, 0.4)
            }
        })
        .collect();
    let covariance: Vec<f64> = (0..k * k)
        .map(|index| {
            let (row, column) = (index / k, index % k);
            if row == column {
                usize_to_f64(row).mul_add(0.01, 0.16)
            } else {
                0.03 / (1.0 + usize_to_f64(row.abs_diff(column)))
            }
        })
        .collect();
    let periods = (0..k)
        .map(|index| i64::try_from(index).expect("fits") - i64::try_from(n_pre).expect("fits") - 1)
        .map(|period| if period >= -1 { period + 1 } else { period })
        .collect();
    EventStudy::new(
        coefficients,
        Covariance::from_values(k, &covariance).expect("covariance"),
        periods,
        -1,
    )
    .expect("event study")
}

fn targets(n_post: usize) -> Vec<Target> {
    let last = i64::try_from(n_post - 1).expect("fits");
    vec![
        Target::Period(0),
        Target::Period(last / 2),
        Target::Average {
            first: 0,
            last: 3.min(last),
        },
    ]
}

fn configure_heavy_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

fn bench_summaries(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("honest_summaries");
    configure_heavy_group(&mut group);
    for (n_pre, n_post) in [(3, 6), (4, 8)] {
        let study = event_study(n_pre, n_post);
        for (name, restriction) in [
            (
                "relative_magnitude",
                Restriction::RelativeMagnitude {
                    mbar: Grid::Values(vec![0.5, 1.0]),
                    bound: RmBound::ParallelTrends,
                },
            ),
            (
                "smoothness",
                Restriction::Smoothness {
                    m: Grid::Values(vec![0.05, 0.1]),
                },
            ),
        ] {
            group.bench_function(format!("{name}_{n_pre}pre_{n_post}post"), |bencher| {
                bencher.iter(|| {
                    for target in targets(n_post) {
                        honest::summarize(
                            &study,
                            &target,
                            &restriction,
                            &HonestOptions {
                                least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
                                ..HonestOptions::default()
                            },
                        )
                        .expect("summary");
                    }
                });
            });
        }
    }
    group.finish();
}

fn bench_joint_regions(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("honest_joint_regions");
    configure_heavy_group(&mut group);
    let n_post = 6;
    let study = event_study(3, n_post);
    let restriction = Restriction::RelativeMagnitude {
        mbar: Grid::Values(vec![1.0]),
        bound: RmBound::ParallelTrends,
    };
    let basis: Vec<Vec<f64>> = (0..n_post)
        .map(|i| (0..n_post).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    let gaussian = Calibration::Gaussian(Bootstrap::new(20_000, Seed(42)));
    for (name, options) in [
        (
            "joint_path",
            JointPathOptions {
                calibration: gaussian,
                least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
                ..JointPathOptions::default()
            },
        ),
        (
            "directional_basis",
            JointPathOptions {
                directions: Directions::Weights(basis),
                calibration: gaussian,
                least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
                ..JointPathOptions::default()
            },
        ),
        (
            "joint_flci",
            JointPathOptions {
                method: JointMethod::Flci,
                calibration: gaussian,
                least_favorable: Some(Bootstrap::new(1_000, Seed(0))),
                ..JointPathOptions::default()
            },
        ),
    ] {
        group.bench_function(name, |bencher| {
            bencher.iter(|| honest::joint_path(&study, &restriction, &options).expect("region"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_summaries, bench_joint_regions);
criterion_main!(benches);
