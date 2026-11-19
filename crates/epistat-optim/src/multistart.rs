//! Independent restarts of a fit, with a selection that does not depend on the
//! number of threads.
//!
//! A multi-start fit runs the same estimator from `n` starting points and keeps
//! the best result. [`run`] evaluates attempts `0..n` and returns the results in
//! attempt order. [`select`] scans such results in that order and keeps one.
//! [`best`] does both.
//!
//! # Seeding
//!
//! The closure receives only the attempt index. The caller derives the seed of
//! attempt `i` from the index, for example with `epistat_rng::replicate_seed`,
//! so each attempt owns its stream and no attempt reads state another one wrote.
//! This crate does not depend on `epistat-rng`.
//!
//! # Selection and ties
//!
//! A failed attempt is `None` and is skipped. The first successful attempt is
//! the incumbent. Each later successful attempt replaces it only when
//! `better(candidate, incumbent)` is true, so with a strict comparison the
//! lowest index wins a tie. The scan is serial and in index order, so the
//! answer is a function of the results alone and not of which attempt finished
//! first. A comparison that is not a total order, such as one involving NaN or a
//! tolerance, is applied exactly as the scan meets it: `x < y` is false when
//! either is NaN, so a NaN incumbent is replaced by nothing.
//!
//! # Parallelism
//!
//! With the `rayon` feature, [`run`] evaluates attempts on the rayon pool and
//! collects them in index order. Without it the attempts run in a loop. Both
//! give the same `Vec`, provided the closure is a function of the index, which
//! is the caller's side of the contract. The bounds `Sync` on the closure and
//! `Send` on the results are required in both builds so that code compiles the
//! same way with and without the feature.
//!
//! # Edge cases
//!
//! - `n = 0`: [`run`] returns an empty vector and [`best`] returns `None`.
//! - Every attempt fails: [`best`] and [`select`] return `None`.
//! - A closure that panics propagates the panic. Under `rayon` other attempts
//!   may or may not have run.

/// Evaluate attempts `0..n` and return the results in attempt order.
///
/// Parallel under the `rayon` feature, serial otherwise, with the same result.
pub fn run<R, F>(n: usize, attempt: F) -> Vec<R>
where
    F: Fn(usize) -> R + Sync + Send,
    R: Send,
{
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(attempt).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..n).map(attempt).collect()
    }
}

/// Keep one result from `results`, scanned in order, and its index.
///
/// `None` entries are skipped. The first `Some` is the incumbent and a later one
/// replaces it only when `better(candidate, incumbent)`. Returns `None` when no
/// entry is `Some`.
pub fn select<T, I, B>(results: I, mut better: B) -> Option<(usize, T)>
where
    I: IntoIterator<Item = Option<T>>,
    B: FnMut(&T, &T) -> bool,
{
    let mut best: Option<(usize, T)> = None;
    for (index, result) in results.into_iter().enumerate() {
        let Some(candidate) = result else { continue };
        match &best {
            Some((_, incumbent)) if !better(&candidate, incumbent) => {}
            _ => best = Some((index, candidate)),
        }
    }
    best
}

/// Run `n` attempts and keep the best successful one with its index.
///
/// Equivalent to [`select`] applied to [`run`].
pub fn best<T, F, B>(n: usize, attempt: F, better: B) -> Option<(usize, T)>
where
    F: Fn(usize) -> Option<T> + Sync + Send,
    T: Send,
    B: FnMut(&T, &T) -> bool,
{
    select(run(n, attempt), better)
}

#[cfg(test)]
mod tests {
    use super::{best, run, select};

    #[test]
    fn results_come_back_in_attempt_order() {
        assert_eq!(run(5, |i| i * i), vec![0, 1, 4, 9, 16]);
    }

    #[test]
    fn the_lowest_index_wins_a_tie() {
        let picked = best(4, |i| Some([3.0, 2.0, 2.0, 5.0][i]), |c, b| c < b);
        assert_eq!(picked, Some((1, 2.0)));
    }

    #[test]
    fn failed_attempts_are_skipped() {
        let picked = best(4, |i| (i >= 2).then_some(10 - i), |c, b| c < b);
        assert_eq!(picked, Some((3, 7)));
    }

    #[test]
    fn nothing_to_pick() {
        assert_eq!(best(0, |_| Some(1), |c: &i32, b| c < b), None);
        assert_eq!(best(3, |_| None::<i32>, |c, b| c < b), None);
    }

    #[test]
    fn a_nan_first_is_kept_by_a_strict_comparison() {
        let picked = select([Some(f64::NAN), Some(1.0)], |c, b| c < b);
        assert!(picked.is_some_and(|(i, v)| i == 0 && v.is_nan()));
    }
}
