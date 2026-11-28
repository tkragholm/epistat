use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use epistat_core::{Level, Matrix};
use epistat_semicontinuous::two_part::percentile_intervals;

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark size fits u32"))
}

fn bootstrap_percentile_ci_sorted_reference(betas: &Matrix, alpha: f64) -> Vec<(f64, f64)> {
    betas
        .columns()
        .map(|column| {
            let mut values = column.collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            let lower = epistat_stats::quantile::type7(&values, alpha / 2.0).unwrap();
            let upper = epistat_stats::quantile::type7(&values, 1.0 - alpha / 2.0).unwrap();
            (lower, upper)
        })
        .collect()
}

/// Draws with one row per replicate and one column per coefficient.
fn build_betas(reps: usize, coefficients: usize) -> Matrix {
    Matrix::from_fn(reps, coefficients, |draw, coef| {
        let d = usize_to_f64(draw);
        let c = usize_to_f64(coef);
        0.001f64.mul_add(-d, 0.05f64.mul_add(c, (0.17f64.mul_add(d, 0.31 * c)).sin()))
    })
    .expect("finite draws")
}

fn bench_two_part_bootstrap_percentile_ci(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("two_part_bootstrap_percentile_ci");
    let coefficients = 12usize;
    let level = Level::alpha(0.05).expect("level");
    for reps in [200usize, 999, 5_000] {
        let betas = build_betas(reps, coefficients);
        group.throughput(Throughput::Elements((reps as u64) * (coefficients as u64)));
        group.bench_function(format!("current_select/reps={reps}"), |bench| {
            bench.iter(|| {
                let _ = percentile_intervals(&betas, level);
            });
        });
        group.bench_function(format!("reference_sort/reps={reps}"), |bench| {
            bench.iter(|| {
                let _ = bootstrap_percentile_ci_sorted_reference(&betas, 0.05);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_two_part_bootstrap_percentile_ci);
criterion_main!(benches);
