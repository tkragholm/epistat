use rayon::ThreadPoolBuilder;

/// Runs `f` once inside a rayon pool of each of `thread_counts` threads and
/// returns the results in that order.
///
/// # Panics
///
/// If a pool cannot be built.
pub fn on_threads<T: Send>(thread_counts: &[usize], f: impl Fn() -> T + Sync) -> Vec<T> {
    thread_counts
        .iter()
        .map(|&threads| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build a rayon pool")
                .install(&f)
        })
        .collect()
}

/// Checks that `f` returns bit-identical values in pools of 1, 2 and 8
/// threads and when called again on 1.
///
/// # Panics
///
/// At the first value that differs, naming its index and the thread count.
#[track_caller]
pub fn assert_same_on_threads(f: impl Fn() -> Vec<f64> + Sync) {
    let counts = [1, 2, 8, 1];
    let runs = on_threads(&counts, f);
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let first = bits(&runs[0]);
    for (run, threads) in runs.iter().zip(counts).skip(1) {
        let other = bits(run);
        assert_eq!(
            first.len(),
            other.len(),
            "length differs on {threads} threads"
        );
        if let Some(index) = first.iter().zip(&other).position(|(a, b)| a != b) {
            panic!(
                "index {index} differs on {threads} threads: {:e} against {:e} on 1",
                run[index], runs[0][index]
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn pools_have_the_requested_size() {
        assert_eq!(on_threads(&[1, 3], rayon::current_num_threads), [1, 3]);
    }

    #[test]
    fn an_ordered_reduction_passes() {
        assert_same_on_threads(|| {
            let terms: Vec<f64> = (1..10_000).map(|i| 1.0 / f64::from(i)).collect();
            vec![terms.iter().sum()]
        });
    }

    #[test]
    #[should_panic(expected = "differs")]
    fn a_reduction_that_follows_the_threads_fails() {
        assert_same_on_threads(|| {
            let threads = rayon::current_num_threads();
            let terms: Vec<f64> = (1..100_000).map(|i| 1.0 / f64::from(i)).collect();
            let size = terms.len().div_ceil(threads);
            vec![terms.par_chunks(size).map(|c| c.iter().sum::<f64>()).sum()]
        });
    }
}
