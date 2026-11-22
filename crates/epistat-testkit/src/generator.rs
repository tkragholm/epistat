//! A `SplitMix64` generator for drawing test problems.
//!
//! The stream is the counter `state += 0x9E3779B97F4A7C15` followed by the
//! `SplitMix64` output function, the same stream as `epistat_rng::Stream`.
//! The testkit does not depend on that crate, so that it can be a
//! dev-dependency of it.
//!
//! Tests that draw from a linear congruential or xorshift generator of their
//! own keep it: their expectations were fixed against that data, and a
//! different stream would change what they check.

/// A `SplitMix64` stream seeded with `seed` as given.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A stream whose state starts at `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on `[0, 1)` from the top 53 bits of the next output.
    #[expect(clippy::cast_precision_loss, reason = "53 bits are exact")]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Uniform on `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// The next output modulo `bound`, which must be positive.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_outputs_are_the_published_splitmix64_values() {
        let mut stream = SplitMix64::new(0);
        assert_eq!(stream.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(stream.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn unit_draws_lie_in_the_half_open_interval() {
        let mut stream = SplitMix64::new(3);
        for _ in 0..1000 {
            let u = stream.unit();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
