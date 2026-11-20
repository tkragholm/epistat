//! Sums over blocks of rows that do not depend on the number of threads.

use std::ops::Range;

use rayon::prelude::*;

/// Merges the values of the blocks of `0..n`, `block` rows each, in the order of
/// the blocks. `make` forms the value of a block and `merge` adds a later block
/// to the total. The first block is the start of the total, so no zero is
/// added to it.
///
/// `wave` blocks are formed at once and merged before the next wave, which
/// bounds the memory held. The merge order is that of the blocks whatever the
/// wave, so the result depends on `block` and not on `wave` or the number of
/// threads. A `make` that fails ends the reduction with the error of the first
/// failing block. `n == 0` gives `make(0..0)`.
pub(crate) fn try_ordered_reduce<T: Send, E: Send>(
    n: usize,
    block: usize,
    wave: usize,
    make: impl Fn(Range<usize>) -> Result<T, E> + Sync,
    merge: impl Fn(&mut T, T),
) -> Result<T, E> {
    let starts: Vec<usize> = (0..n).step_by(block).collect();
    let mut total: Option<T> = None;
    for wave in starts.chunks(wave) {
        let parts: Vec<Result<T, E>> = wave
            .par_iter()
            .map(|&s| make(s..(s + block).min(n)))
            .collect();
        for part in parts {
            let part = part?;
            match total.as_mut() {
                Some(t) => merge(t, part),
                None => total = Some(part),
            }
        }
    }
    match total {
        Some(t) => Ok(t),
        None => make(0..0),
    }
}

/// [`try_ordered_reduce`] for a `make` that cannot fail.
pub(crate) fn ordered_reduce<T: Send>(
    n: usize,
    block: usize,
    wave: usize,
    make: impl Fn(Range<usize>) -> T + Sync,
    merge: impl Fn(&mut T, T),
) -> T {
    match try_ordered_reduce(
        n,
        block,
        wave,
        |range| Ok::<T, std::convert::Infallible>(make(range)),
        merge,
    ) {
        Ok(total) => total,
        Err(never) => match never {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_are_merged_in_order_whatever_the_wave() {
        let values: Vec<f64> = (0..1000).map(|i| 1.0 / f64::from(i + 1)).collect();
        let sum = |wave| {
            try_ordered_reduce::<f64, ()>(
                values.len(),
                7,
                wave,
                |r| Ok(values[r].iter().sum::<f64>()),
                |t, p| *t += p,
            )
            .unwrap()
        };
        assert_eq!(sum(1).to_bits(), sum(5).to_bits());
        assert_eq!(sum(1).to_bits(), sum(1000).to_bits());
    }

    #[test]
    fn the_first_failing_block_ends_the_reduction_and_an_empty_range_makes_one_block() {
        let failed = try_ordered_reduce(
            10,
            3,
            2,
            |r| if r.start >= 3 { Err(r.start) } else { Ok(0.0) },
            |t: &mut f64, p| *t += p,
        );
        assert_eq!(failed, Err(3));
        let empty = try_ordered_reduce::<usize, ()>(0, 3, 2, |r| Ok(r.len()), |t, p| *t += p);
        assert_eq!(empty, Ok(0));
    }
}
