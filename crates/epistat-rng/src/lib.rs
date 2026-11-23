//! `SplitMix64` streams and the bootstrap draws built on them.
//!
//! Every draw of [`Stream`] is integer arithmetic or scaling by a power of
//! two, so it is the same on every platform and another language can
//! reproduce it bit for bit. A test can then compare a kernel with a reference
//! implementation draw for draw rather than only in distribution.
//!
//! # Algorithm
//!
//! [`Stream`] is `SplitMix64` \[1\], the generator of
//! `java.util.SplittableRandom`. Let `γ = 0x9E37_79B9_7F4A_7C15` and let `mix`
//! be the 64-bit mixing function
//!
//! ```text
//! z = (z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9
//! z = (z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB
//! z =  z ^ (z >> 31)
//! ```
//!
//! with all arithmetic modulo 2^64, the `mix64` of `SplittableRandom`. The
//! state is a Weyl counter, and each draw advances it by `γ` and returns the
//! mixed state:
//!
//! ```text
//! s(0) = seed,    s(n+1) = s(n) + γ,    output n+1 = mix(s(n+1)).
//! ```
//!
//! Seeded with 0, the first four outputs are `0xE220_A839_7B1D_CDAF`,
//! `0x6E78_9E6A_A1B9_65F4`, `0x06C4_5D18_8009_454F` and
//! `0xF88B_B8A8_724C_81EC`, which the tests check. [`splitmix64`] computes
//! `mix(x + γ)`, the first output of a stream seeded with `x`.
//!
//! # Period
//!
//! `γ` is odd, so the counter visits every 64-bit value before it repeats, and
//! `mix` is a bijection, so every stream has period 2^64 \[1\]. Steele, Lea and
//! Flood report that the generator passes the BigCrush battery of TestU01 \[1\].
//!
//! # Seeding
//!
//! [`Stream::seeded`] takes any 64-bit value as `s(0)`, unmixed. Two streams
//! whose seeds differ by a multiple of `γ` walk the same counter and reproduce
//! each other's outputs with a shift; seeds are not checked for such overlap.
//! Bootstrap code seeds replicate `r` with [`replicate_seed`]`(base, r)`,
//! which is `splitmix64(base ^ (r * γ))`, so the draws of a replicate depend
//! only on the base seed and `r`, not on which thread runs it or in what
//! order, and a caller that loops replicates itself gets the same streams.
//!
//! # Derived draws
//!
//! - [`Stream::uniform`] is `(x >> 11) * 2^-53`, a multiple of 2^-53 in
//!   `[0, 1)`. The shifted value and the scale are both exact in an `f64`, so
//!   the result is exact.
//! - [`Stream::below`] maps `x` to `floor(x * b / 2^64)` on `[0, b)`, the
//!   multiply-and-shift method of \[2\] without its rejection step. The result
//!   is therefore biased. For a uniform 64-bit `x`, each value in `[0, b)` is
//!   the image of either `floor(2^64 / b)` or `ceil(2^64 / b)` inputs, so its
//!   probability differs from `1 / b` by less than 2^-64, and the most likely
//!   value is at most `1 + 1 / floor(2^64 / b)` times as likely as the least
//!   likely, about `1 + b / 2^64`. For `b = 10^6` that factor is about
//!   `1 + 5.4e-14`.
//! - [`Stream::shuffle`] is the Fisher-Yates shuffle, from the last position
//!   to the first, with `below(i + 1)` choosing the element for position `i`.
//!   It carries the bias of `below`.
//! - [`Stream::exp1`] is `-ln(1 - u)` for the next uniform `u`, and
//!   [`dirichlet_weights`] scales `n` such variates to sum to `n`. These call
//!   `ln`, which comes from the platform's libm and may differ in the last bit
//!   between platforms and languages, so they agree across them only as far as
//!   libm does.
//! - [`rademacher_sum`] takes the signs of 64 values from each output, lowest
//!   bit first, a set bit being `+1`.
//!
//! - [`dirichlet_scale`] is the factor [`dirichlet_weights`] scales by, without
//!   keeping the weights.
//!
//! Normal, gamma, chi-square, Beta, Wishart and inverse Wishart draws, and
//! a sign fill, are on [`Stream`] and described in the `draws` module. They
//! call `ln`, `cos` and the like, with the caveat of `exp1`.
//!
//! # Python reference
//!
//! The draws above, written in Python. Python integers do not wrap, so each
//! step that wraps in a `u64` is masked. The outputs in the comments are the
//! values the tests of this crate check.
//!
//! ```python
//! MASK = (1 << 64) - 1
//! GAMMA = 0x9E3779B97F4A7C15
//!
//! def splitmix64(value):
//!     z = (value + GAMMA) & MASK
//!     z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
//!     z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
//!     return z ^ (z >> 31)
//!
//! class Stream:
//!     def __init__(self, seed):          # Stream::seeded
//!         self.state = seed & MASK
//!
//!     def next_u64(self):
//!         out = splitmix64(self.state)
//!         self.state = (self.state + GAMMA) & MASK
//!         return out
//!
//!     def uniform(self):
//!         return (self.next_u64() >> 11) / 2.0**53
//!
//! def replicate_seed(base, replicate):
//!     return splitmix64((base & MASK) ^ ((replicate * GAMMA) & MASK))
//!
//! Stream(0).next_u64()      # 0xE220A8397B1DCDAF
//! Stream(0).uniform()       # 0.8833108082136426
//! replicate_seed(1, 2)      # 0xF893A2EEFB32555E
//! ```
//!
//! # References
//!
//! 1. Steele GL, Lea D, Flood CH. Fast splittable pseudorandom number
//!    generators. In: Proceedings of the 2014 ACM International Conference on
//!    Object Oriented Programming Systems Languages and Applications
//!    (OOPSLA 2014). New York: ACM; 2014. p. 453-72.
//!    doi:10.1145/2660193.2660195
//! 2. Lemire D. Fast random integer generation in an interval. ACM Trans
//!    Model Comput Simul. 2019;29(1):3. doi:10.1145/3230636
#![warn(missing_docs)]

mod draws;

pub use draws::DrawError;

/// The Weyl increment of `SplitMix64`: the odd integer nearest `2^64` divided by
/// the golden ratio.
pub const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The `SplitMix64` output function: two xor-shift-multiply rounds and a final
/// xor-shift, with no increment of its own. [`splitmix64`] is this applied to
/// `value + GAMMA`.
#[must_use]
pub const fn mix64(value: u64) -> u64 {
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The first output of a stream seeded with `value`.
#[must_use]
pub const fn splitmix64(value: u64) -> u64 {
    mix64(value.wrapping_add(GAMMA))
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
    /// A stream that starts from `seed`.
    #[must_use]
    pub const fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let output = splitmix64(self.state);
        self.state = self.state.wrapping_add(GAMMA);
        output
    }

    /// Uniform on `[0, 1)`: the top 53 bits of the next output times `2^-53`,
    /// which is exact.
    pub fn uniform(&mut self) -> f64 {
        const SCALE: f64 = 9_007_199_254_740_992.0;
        (self.next_u64() >> 11) as f64 / SCALE
    }

    /// An integer on `[0, bound)`, or 0 when `bound` is 0.
    ///
    /// Lemire's multiply-and-shift without the rejection step. Values differ
    /// in probability by a factor of at most about `1 + bound / 2^64`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the result is below `bound`"
    )]
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

/// With the `rand` feature, a [`Stream`] is a `rand_core` generator, so the
/// distributions of `rand` and `rand_distr` can draw from it. `next_u64` is the
/// stream's own output, `next_u32` its upper 32 bits, and bytes are filled
/// eight at a time from successive outputs, little-endian.
/// `SeedableRng::seed_from_u64(s)` is [`Stream::seeded`]`(s)`, so a seed gives
/// the same stream either way.
#[cfg(feature = "rand")]
mod rand_impl {
    use core::convert::Infallible;

    use rand_core::{SeedableRng, TryRng};

    use super::Stream;

    impl TryRng for Stream {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Infallible> {
            Ok((self.next_u64() >> 32) as u32)
        }

        fn try_next_u64(&mut self) -> Result<u64, Infallible> {
            Ok(self.next_u64())
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
            for chunk in dst.chunks_mut(8) {
                let bytes = self.next_u64().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
            Ok(())
        }
    }

    impl SeedableRng for Stream {
        type Seed = [u8; 8];

        fn from_seed(seed: [u8; 8]) -> Self {
            Self::seeded(u64::from_le_bytes(seed))
        }

        fn seed_from_u64(state: u64) -> Self {
            Self::seeded(state)
        }
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
    let scale = weight_scale(n, sum);
    for weight in out.iter_mut() {
        *weight *= scale;
    }
}

/// The factor [`dirichlet_weights`] multiplies its variates by: `n` divided by
/// the sum of the `n` `Exp(1)` variates of the stream seeded with `seed`,
/// summed in index order, or 1 when that sum is not positive.
///
/// The weight of unit `i` is the `i`-th variate times this scale, so a
/// caller that needs only the scale, or draws the variates itself, does not
/// store the weights.
#[must_use]
pub fn dirichlet_scale(n: usize, seed: u64) -> f64 {
    let mut stream = Stream::seeded(seed);
    let mut sum = 0.0;
    for _ in 0..n {
        sum += stream.exp1();
    }
    weight_scale(n, sum)
}

fn weight_scale(n: usize, sum: f64) -> f64 {
    if sum > 0.0 { n as f64 / sum } else { 1.0 }
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

/// [`rademacher_sum`] for `L` streams over the same values at once. Lane `l`
/// returns what `rademacher_sum(values, &mut streams[l])` returns and leaves
/// `streams[l]` where it leaves it, bit for bit: each lane adds the same
/// values in the same order, and a clear sign bit flips the sign of the value,
/// which is what negation does.
pub fn rademacher_sums<const L: usize>(values: &[f64], streams: &mut [Stream; L]) -> [f64; L] {
    let mut sum = [0.0_f64; L];
    for chunk in values.chunks(64) {
        let mut bits: [u64; L] = std::array::from_fn(|lane| streams[lane].next_u64());
        for &value in chunk {
            let magnitude = value.to_bits();
            for lane in 0..L {
                let flip = (!bits[lane] & 1) << 63;
                sum[lane] += f64::from_bits(magnitude ^ flip);
                bits[lane] >>= 1;
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rand")]
    #[test]
    fn the_rand_core_view_is_the_same_stream() {
        use rand_core::{Rng, SeedableRng};
        let mut via_rand = Stream::seed_from_u64(5);
        let mut direct = Stream::seeded(5);
        assert_eq!(Rng::next_u64(&mut via_rand), direct.next_u64());
        assert_eq!(
            Rng::next_u32(&mut via_rand),
            u32::try_from(direct.next_u64() >> 32).unwrap()
        );
        let mut bytes = [0u8; 12];
        via_rand.fill_bytes(&mut bytes);
        let first = direct.next_u64().to_le_bytes();
        let second = direct.next_u64().to_le_bytes();
        assert_eq!(&bytes[..8], &first);
        assert_eq!(&bytes[8..], &second[..4]);
    }

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
    fn the_python_reference_in_the_docs_gives_these_values() {
        assert_eq!(Stream::seeded(0).next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(
            Stream::seeded(0).uniform().to_bits(),
            0.883_310_808_213_642_6_f64.to_bits()
        );
        assert_eq!(replicate_seed(1, 2), 0xF893_A2EE_FB32_555E);
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
        assert_eq!(resample_counts(0, 4), Vec::<u32>::new());
    }

    #[test]
    fn dirichlet_weights_have_mean_one() {
        let mut weights = vec![9.0];
        dirichlet_weights(500, 11, &mut weights);
        assert_eq!(weights.len(), 500);
        assert!(weights.iter().all(|&w| w > 0.0 && w.is_finite()));
        assert!((weights.iter().sum::<f64>() - 500.0).abs() < 1e-9);
        dirichlet_weights(0, 11, &mut weights);
        assert_eq!(weights, Vec::<f64>::new());
    }

    #[test]
    fn rademacher_signs_come_from_the_bits() {
        let values = vec![1.0; 70];
        let mut stream = Stream::seeded(8);
        let sum = rademacher_sum(&values, &mut stream);

        let mut reference = Stream::seeded(8);
        let first = reference.next_u64();
        let second = reference.next_u64() & ((1 << 6) - 1);
        let plus = first.count_ones() + second.count_ones();
        assert_eq!(
            sum.to_bits(),
            (f64::from(plus) - f64::from(70 - plus)).to_bits()
        );
        assert_eq!(stream, reference);
    }

    /// Equal bits, or both NaN: the sign of a NaN produced by arithmetic is
    /// not specified and differs between x86 and ARM.
    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn lanes_equal_the_single_stream_draw() {
        let specials = [f64::NAN, -0.0, 0.0, f64::INFINITY, -f64::INFINITY, 1e308];
        for n in [0usize, 1, 7, 63, 64, 65, 127, 128, 130, 1000] {
            let values: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 17 == 5 {
                        specials[i % specials.len()]
                    } else {
                        (i as f64 * 0.37).sin() * 1e-3 * (1.0 + i as f64)
                    }
                })
                .collect();
            let mut lanes: [Stream; 8] =
                std::array::from_fn(|l| Stream::seeded(replicate_seed(9, l as u64)));
            let mut singles = lanes.clone();
            let got = rademacher_sums(&values, &mut lanes);
            for l in 0..8 {
                let want = rademacher_sum(&values, &mut singles[l]);
                assert!(same(got[l], want), "n = {n}, lane {l}");
                assert_eq!(lanes[l], singles[l]);
            }
            let mut four: [Stream; 4] = std::array::from_fn(|l| Stream::seeded(l as u64 + 100));
            let mut ref_four = four.clone();
            let got = rademacher_sums(&values, &mut four);
            for l in 0..4 {
                let want = rademacher_sum(&values, &mut ref_four[l]);
                assert!(same(got[l], want));
            }
        }
    }
}
