use rayon::ThreadPoolBuilder;

/// The pool sizes a thread-independence test runs on: one thread, an even and
/// an odd small count, and a count that does not divide typical input sizes.
///
/// Pass it to [`assert_same_on_pools`] or [`on_threads`].
pub const POOLS: [usize; 5] = [1, 2, 3, 4, 8];

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
    assert_same_on_pools(&[1, 2, 8, 1], f);
}

/// Checks that `f` returns bit-identical values in a rayon pool of each of
/// `thread_counts` threads, each compared with the first.
///
/// [`assert_same_on_threads`] fixes the sizes at 1, 2, 8 and 1. Use this when
/// the code under test splits work in a way that depends on the pool size,
/// for example at a size that does not divide the input evenly.
///
/// # Panics
///
/// If `thread_counts` is empty, or at the first value that differs, naming its
/// index and the thread count.
#[track_caller]
pub fn assert_same_on_pools(thread_counts: &[usize], f: impl Fn() -> Vec<f64> + Sync) {
    assert!(!thread_counts.is_empty(), "no pool sizes given");
    let runs = on_threads(thread_counts, f);
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let first = bits(&runs[0]);
    for (run, &threads) in runs.iter().zip(thread_counts).skip(1) {
        let other = bits(run);
        assert_eq!(
            first.len(),
            other.len(),
            "length differs on {threads} threads"
        );
        if let Some(index) = first.iter().zip(&other).position(|(a, b)| a != b) {
            panic!(
                "index {index} differs on {threads} threads: {:e} against {:e} on {}",
                run[index], runs[0][index], thread_counts[0]
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
    fn the_standard_pools_start_with_one_thread() {
        assert_eq!(POOLS, [1, 2, 3, 4, 8]);
        assert_same_on_pools(&POOLS, || vec![1.0]);
    }

    #[test]
    fn an_ordered_reduction_passes() {
        assert_same_on_threads(|| {
            let terms: Vec<f64> = (1..10_000).map(|i| 1.0 / f64::from(i)).collect();
            vec![terms.iter().sum()]
        });
    }

    #[test]
    fn pool_sizes_are_free() {
        assert_same_on_pools(&[1, 2, 3, 12], || {
            let terms: Vec<f64> = (1..10_000).map(|i| 1.0 / f64::from(i)).collect();
            vec![terms.iter().sum()]
        });
    }

    #[test]
    #[should_panic(expected = "differs on 3 threads")]
    fn a_pool_size_that_changes_the_result_is_named() {
        assert_same_on_pools(&[1, 1, 3], || vec![rayon::current_num_threads() as f64]);
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
