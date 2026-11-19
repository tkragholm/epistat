//! `SplitMix64` streams and the bootstrap draws built on them.
//!
//! [`Stream`] is `SplitMix64` (Steele, Lea and Flood, OOPSLA 2014), the
//! generator of `java.util.SplittableRandom`. Its draws use only integer
//! arithmetic and scaling by powers of two, so they are identical on every
//! platform and can be reproduced bit for bit in another language. The
//! exceptions are [`Stream::exp1`] and [`dirichlet_weights`], which call `ln`
//! and so agree across platforms only as far as their libm does.
//!
//! Bootstrap code seeds replicate `r` with [`replicate_seed`]`(base, r)`, so
//! the draws of a replicate do not depend on which thread runs it or in what
//! order.

/// The Weyl increment of `SplitMix64`: the odd integer nearest `2^64` divided by
/// the golden ratio.
pub const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The first output of a stream seeded with `value`.
#[must_use]
pub const fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed of replicate `replicate`: `splitmix64(base ^ replicate * GAMMA)`.
#[must_use]
pub const fn replicate_seed(base: u64, replicate: u64) -> u64 {
    splitmix64(base ^ replicate.wrapping_mul(GAMMA))
}

/// A hash of `bytes` that is the same in every build and on every platform:
/// `FNV-1a`, then [`splitmix64`] of the result XOR `seed`.
///
/// Sorting by it gives a seeded, reproducible order of identifiers.
#[must_use]
pub fn seeded_id_hash(bytes: &[u8], seed: u64) -> u64 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    splitmix64(hash ^ seed)
}

/// A `SplitMix64` stream.
///
/// The state is a counter that advances by [`GAMMA`] per draw, so every
/// stream has period `2^64`. Seeds are used as given; two seeds that differ
/// by a multiple of `GAMMA` give shifted copies of one stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    state: u64,
}

impl Stream {
    #[must_use]
    pub const fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        let output = splitmix64(self.state);
        self.state = self.state.wrapping_add(GAMMA);
        output
    }

    /// Uniform on `[0, 1)`: the top 53 bits of the next output times `2^-53`,
    /// which is exact.
    #[allow(clippy::cast_precision_loss)] // 53 bits fit an f64 significand.
    pub fn uniform(&mut self) -> f64 {
        const SCALE: f64 = 9_007_199_254_740_992.0;
        (self.next_u64() >> 11) as f64 / SCALE
    }

    /// An integer on `[0, bound)`, or 0 when `bound` is 0.
    ///
    /// Lemire's multiply-and-shift without the rejection step. Values differ
    /// in probability by a factor of at most about `1 + bound / 2^64`.
    #[allow(clippy::cast_possible_truncation)] // The result is below `bound`.
    pub fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        ((u128::from(self.next_u64()) * bound as u128) >> 64) as usize
    }

    /// Fisher-Yates, from the last position to the first.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            items.swap(index, self.below(index + 1));
        }
    }

    /// One element of `items`, or `None` when it is empty.
    pub fn choice<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            Some(&items[self.below(items.len())])
        }
    }

    /// An `Exp(1)` variate, `-ln(1 - u)`, which is always finite.
    pub fn exp1(&mut self) -> f64 {
        -(1.0 - self.uniform()).ln()
    }
}

/// How often each of `n` units is drawn in one resample of `n` with
/// replacement, as one count per unit in index order.
#[must_use]
pub fn resample_counts(n: usize, seed: u64) -> Vec<u32> {
    let mut stream = Stream::seeded(seed);
    let mut counts = vec![0u32; n];
    for _ in 0..n {
        counts[stream.below(n)] += 1;
    }
    counts
}

/// Bayesian bootstrap weights: `n` draws of `Exp(1)` scaled to sum to `n`,
/// which is Dirichlet(1, ..., 1) scaled to mean 1.
///
/// The variates are summed in index order into a plain `f64`. `out` is
/// cleared first, so a replicate loop can reuse one buffer.
#[allow(clippy::cast_precision_loss)]
pub fn dirichlet_weights(n: usize, seed: u64, out: &mut Vec<f64>) {
    out.clear();
    out.reserve(n);
    let mut stream = Stream::seeded(seed);
    let mut sum = 0.0;
    for _ in 0..n {
        let variate = stream.exp1();
        out.push(variate);
        sum += variate;
    }
    let scale = if sum > 0.0 { n as f64 / sum } else { 1.0 };
    for weight in out.iter_mut() {
        *weight *= scale;
    }
}

/// `sum(s_i * values_i)` with independent signs `s_i = ±1`, the draw of a
/// Rademacher multiplier bootstrap.
///
/// Each output of `stream` gives the signs of 64 values, from the lowest bit
/// up; a set bit is `+1`.
pub fn rademacher_sum(values: &[f64], stream: &mut Stream) -> f64 {
    let mut sum = 0.0;
    for chunk in values.chunks(64) {
        let mut bits = stream.next_u64();
        for &value in chunk {
            sum += if bits & 1 == 1 { value } else { -value };
            bits >>= 1;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_zero_gives_the_published_outputs() {
        let mut stream = Stream::seeded(0);
        let first: Vec<u64> = (0..4).map(|_| stream.next_u64()).collect();
        assert_eq!(
            first,
            [
                0xE220_A839_7B1D_CDAF,
                0x6E78_9E6A_A1B9_65F4,
                0x06C4_5D18_8009_454F,
                0xF88B_B8A8_724C_81EC,
            ]
        );
    }

    #[test]
    fn the_id_hash_starts_from_fnv1a() {
        // FNV-1a of "" and of "a" are the published test vectors.
        assert_eq!(seeded_id_hash(b"", 0), splitmix64(0xCBF2_9CE4_8422_2325));
        assert_eq!(seeded_id_hash(b"a", 0), splitmix64(0xAF63_DC4C_8601_EC8C));
        assert_ne!(seeded_id_hash(b"a", 1), seeded_id_hash(b"a", 2));
    }

    #[test]
    fn uniform_and_below_stay_in_range() {
        let mut stream = Stream::seeded(1);
        let mut seen = [false; 7];
        for _ in 0..10_000 {
            assert!((0.0..1.0).contains(&stream.uniform()));
            seen[stream.below(7)] = true;
        }
        assert!(seen.iter().all(|&hit| hit));
        assert_eq!(stream.below(0), 0);
    }

    #[test]
    fn shuffle_permutes() {
        let mut items: Vec<usize> = (0..64).collect();
        Stream::seeded(99).shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        assert_ne!(items, sorted);
    }

    #[test]
    fn choice_of_nothing_is_none() {
        assert_eq!(Stream::seeded(3).choice::<u8>(&[]), None);
        assert_eq!(Stream::seeded(3).choice(&[5]), Some(&5));
    }

    #[test]
    fn resample_counts_add_up() {
        let counts = resample_counts(1_000, 4);
        assert_eq!(counts.iter().sum::<u32>(), 1_000);
        assert!(resample_counts(0, 4).is_empty());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn dirichlet_weights_have_mean_one() {
        let mut weights = vec![9.0];
        dirichlet_weights(500, 11, &mut weights);
        assert_eq!(weights.len(), 500);
        assert!(weights.iter().all(|&w| w > 0.0 && w.is_finite()));
        assert!((weights.iter().sum::<f64>() - 500.0).abs() < 1e-9);
        dirichlet_weights(0, 11, &mut weights);
        assert!(weights.is_empty());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn rademacher_signs_come_from_the_bits() {
        let values = vec![1.0; 70];
        let mut stream = Stream::seeded(8);
        let sum = rademacher_sum(&values, &mut stream);

        let mut reference = Stream::seeded(8);
        let first = reference.next_u64();
        let second = reference.next_u64() & ((1 << 6) - 1);
        let plus = first.count_ones() + second.count_ones();
        assert_eq!(sum, f64::from(plus) - f64::from(70 - plus));
        assert_eq!(stream, reference);
    }
}
