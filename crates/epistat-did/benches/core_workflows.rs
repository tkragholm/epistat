use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use epistat_core::{Bootstrap, Seed};
use epistat_did::att_gt::{self, AttGtOptions, Estimator};
use epistat_did::did_cc::{self, Assumption, CrossFitting, DidCcOptions};
use epistat_did::two_period::{self, NuisanceOptions, TwoPeriodOptions};
use epistat_did::{StaggeredPanel, TwoPeriodData};
use epistat_linalg::MatRef;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark size fits u32"))
}

/// Two-period columns: arm, outcome (or change), period, and the covariates
/// column-major with `k` columns.
struct TwoPeriodColumns {
    treated: Vec<bool>,
    post: Vec<bool>,
    outcome: Vec<f64>,
    covariates: Vec<f64>,
    k: usize,
}

impl TwoPeriodColumns {
    fn from_rows(rows: &[(bool, bool, f64, Vec<f64>)]) -> Self {
        let k = rows.first().map_or(0, |row| row.3.len());
        Self {
            treated: rows.iter().map(|row| row.0).collect(),
            post: rows.iter().map(|row| row.1).collect(),
            outcome: rows.iter().map(|row| row.2).collect(),
            covariates: (0..k)
                .flat_map(|j| rows.iter().map(move |row| row.3[j]))
                .collect(),
            k,
        }
    }

    fn len(&self) -> usize {
        self.treated.len()
    }

    fn x(&self) -> MatRef<'_, f64> {
        MatRef::from_column_major_slice(&self.covariates, self.len(), self.k)
    }

    fn panel(&self) -> TwoPeriodData<'_> {
        TwoPeriodData::panel(&self.treated, &self.outcome)
            .covariates(self.x())
            .build()
            .expect("panel")
    }

    fn repeated(&self) -> TwoPeriodData<'_> {
        TwoPeriodData::repeated_cross_section(&self.treated, &self.post, &self.outcome)
            .covariates(self.x())
            .build()
            .expect("repeated cross-section")
    }
}

fn build_drdid_panel_rows(num_units: usize, num_covariates: usize) -> TwoPeriodColumns {
    let rows: Vec<_> = (0..num_units)
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
            (
                treated,
                true,
                baseline + covariate_signal + treatment_effect,
                covariates,
            )
        })
        .collect();
    TwoPeriodColumns::from_rows(&rows)
}

/// A repeated cross-section of `num_units * periods` rows, each its own
/// observation: the unit, period, cohort and outcome columns and two
/// covariates, column-major.
struct AttGtColumns {
    unit: Vec<u32>,
    period: Vec<i64>,
    cohort: Vec<Option<i64>>,
    outcome: Vec<f64>,
    covariates: Vec<f64>,
}

fn build_att_gt_rows(num_units: usize, start_year: i64, periods: usize) -> AttGtColumns {
    let n = num_units * periods;
    let mut columns = AttGtColumns {
        unit: Vec::with_capacity(n),
        period: Vec::with_capacity(n),
        cohort: Vec::with_capacity(n),
        outcome: Vec::with_capacity(n),
        covariates: vec![0.0; 2 * n],
    };
    for unit_index in 0..num_units {
        let first_treated_time = match unit_index % 5 {
            0 => Some(start_year + 2),
            1 => Some(start_year + 3),
            2 => Some(start_year + 4),
            _ => None,
        };
        let baseline = usize_to_f64(unit_index % 17) * 0.2;
        for period_offset in 0..periods {
            let row = columns.unit.len();
            let time = start_year + i64::try_from(period_offset).expect("period offset fits i64");
            let event_time = first_treated_time.map_or(-99, |group| time - group);
            let treated_effect = if event_time >= 0 {
                0.8 + event_time as f64 * 0.15
            } else {
                0.0
            };
            let trend = (time - start_year) as f64 * 0.05;
            columns
                .unit
                .push(u32::try_from(row).expect("row count fits u32"));
            columns.period.push(time);
            columns.cohort.push(first_treated_time);
            columns.outcome.push(baseline + trend + treated_effect);
            columns.covariates[row] = usize_to_f64(unit_index % 7) * 0.1;
            columns.covariates[n + row] = usize_to_f64(period_offset) * 0.05;
        }
    }
    columns
}

fn build_did_cc_rows(num_units_per_cell: usize) -> TwoPeriodColumns {
    let mut observations = Vec::with_capacity(num_units_per_cell * 4);
    for unit_index in 0..num_units_per_cell {
        let unit_index_f64 = usize_to_f64(unit_index);
        let control_x = usize_to_f64(unit_index % 9) * 0.25;
        let treated_pre_x = usize_to_f64((unit_index + 2) % 9) * 0.25;
        let treated_post_x = usize_to_f64((unit_index + 5) % 9) * 0.25;

        observations.push((
            false,
            false,
            unit_index_f64.mul_add(0.001, control_x),
            vec![control_x, control_x * control_x],
        ));
        observations.push((
            false,
            true,
            unit_index_f64.mul_add(0.001, control_x + 0.9),
            vec![control_x, control_x * control_x],
        ));
        observations.push((
            true,
            false,
            unit_index_f64.mul_add(0.001, treated_pre_x + 0.2),
            vec![treated_pre_x, treated_pre_x * treated_pre_x],
        ));
        observations.push((
            true,
            true,
            unit_index_f64.mul_add(0.001, treated_post_x + 2.1 + treated_post_x * 0.6),
            vec![treated_post_x, treated_post_x * treated_post_x],
        ));
    }
    TwoPeriodColumns::from_rows(&observations)
}

fn build_improved_repeated_rows(num_units_per_period: usize) -> TwoPeriodColumns {
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
            rows.push((
                treated,
                post_period,
                base + treatment_effect + noise,
                vec![x1, x2],
            ));
        }
    }
    TwoPeriodColumns::from_rows(&rows)
}

fn bench_drdid_panel(criterion: &mut Criterion) {
    let rows = build_drdid_panel_rows(20_000, 6);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let mut group = criterion.benchmark_group("drdid_panel");
    group.throughput(Throughput::Elements(throughput));
    group.bench_function("20k_rows_6_covariates", |bencher| {
        let data = rows.panel();
        let options = TwoPeriodOptions {
            estimator: two_period::Estimator::DoublyRobust,
            ..TwoPeriodOptions::default()
        };
        bencher.iter(|| two_period::estimate(&data, &options).expect("panel drdid"));
    });
    group.finish();
}

fn bench_drdid_improved_repeated(criterion: &mut Criterion) {
    let rows = build_improved_repeated_rows(8_000);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let mut group = criterion.benchmark_group("drdid_improved_repeated");
    group.throughput(Throughput::Elements(throughput));
    group.bench_function("8k_per_period_3_covariates", |bencher| {
        let data = rows.repeated();
        bencher.iter(|| {
            two_period::estimate(&data, &TwoPeriodOptions::default())
                .expect("improved repeated drdid")
        });
    });
    group.finish();
}

fn bench_att_gt_dr(criterion: &mut Criterion) {
    let rows = build_att_gt_rows(8_000, 2000, 6);
    let n = rows.unit.len();
    let covariates = epistat_linalg::MatRef::from_column_major_slice(&rows.covariates, n, 2);
    let panel = StaggeredPanel::cross_sections(&rows.period, &rows.cohort, &rows.outcome)
        .covariates(covariates)
        .build()
        .expect("repeated cross-sections");
    let options = AttGtOptions {
        estimator: Estimator::TraditionalDoublyRobust,
        ..AttGtOptions::default()
    };
    let throughput = u64::try_from(n).expect("row count fits u64");
    let mut group = criterion.benchmark_group("att_gt_dr");
    group.throughput(Throughput::Elements(throughput));
    group.bench_function("8k_units_6_periods", |bencher| {
        bencher.iter(|| att_gt::estimate(&panel, &options).expect("att_gt dr"));
    });
    group.finish();
}

fn bench_did_cc(criterion: &mut Criterion) {
    let rows = build_did_cc_rows(4_000);
    let throughput = u64::try_from(rows.len()).expect("row count fits u64");
    let options = |assumption| DidCcOptions {
        assumption,
        ..DidCcOptions::default()
    };
    let cross_fit = |assumption| DidCcOptions {
        assumption,
        nuisance: NuisanceOptions {
            max_iterations: 1_000,
            tolerance: 1e-6,
            ..NuisanceOptions::default()
        },
        bootstrap: Some(Bootstrap::new(19, Seed(17_431))),
        cross_fitting: Some(CrossFitting { folds: 4, seed: 17 }),
        ..DidCcOptions::default()
    };
    let cross_fit_rows = build_did_cc_rows(1_000);
    let (data, cross_fit_data) = (rows.repeated(), cross_fit_rows.repeated());
    let mut group = criterion.benchmark_group("did_cc");
    group.throughput(Throughput::Elements(throughput));

    group.bench_function("robust_4k_per_cell", |bencher| {
        bencher.iter(|| did_cc::estimate(&data, &options(Assumption::Robust)).expect("robust"));
    });
    group.bench_function("stationary_4k_per_cell", |bencher| {
        bencher.iter(|| {
            did_cc::estimate(&data, &options(Assumption::Stationary)).expect("stationary")
        });
    });
    group.bench_function("hausman_4k_per_cell", |bencher| {
        bencher.iter(|| did_cc::test(&data, &options(Assumption::Robust)).expect("hausman"));
    });
    group.bench_function("robust_cross_fit_1k_per_cell", |bencher| {
        bencher.iter(|| {
            did_cc::estimate(&cross_fit_data, &cross_fit(Assumption::Robust))
                .expect("robust cross-fit")
        });
    });
    group.bench_function("stationary_cross_fit_1k_per_cell", |bencher| {
        bencher.iter(|| {
            did_cc::estimate(&cross_fit_data, &cross_fit(Assumption::Stationary))
                .expect("stationary cross-fit")
        });
    });
    group.bench_function("hausman_cross_fit_1k_per_cell", |bencher| {
        bencher.iter(|| {
            did_cc::test(&cross_fit_data, &cross_fit(Assumption::Robust))
                .expect("hausman cross-fit")
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_drdid_panel,
    bench_drdid_improved_repeated,
    bench_att_gt_dr,
    bench_did_cc,
);
criterion_main!(benches);
