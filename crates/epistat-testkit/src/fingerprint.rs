/// A 64-bit FNV-1a hash of exact bit patterns, for pinning results that must
/// not change: any change to a value's last bit changes the hash.
///
/// The hash depends on the platform's libm and floating-point kernels as much
/// as on the code, so pin it on one target.
#[derive(Clone, Debug)]
pub struct Fingerprint(u64);

impl Default for Fingerprint {
    fn default() -> Self {
        Self(0xCBF2_9CE4_8422_2325)
    }
}

impl Fingerprint {
    /// The hash of nothing, FNV-1a's offset basis.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mixes in the eight little-endian bytes of `value`.
    #[must_use]
    pub fn u64(mut self, value: u64) -> Self {
        for byte in value.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
        }
        self
    }

    /// Mixes in the bit pattern of `value`, so `-0.0` and `0.0` differ.
    #[must_use]
    pub fn f64(self, value: f64) -> Self {
        self.u64(value.to_bits())
    }

    /// Mixes in `value`, or [`absent`](crate::absent) for `None`, so that an
    /// undefined value differs from every defined one.
    #[must_use]
    pub fn option_f64(self, value: Option<f64>) -> Self {
        self.f64(crate::absent(value))
    }

    /// Mixes in each of `values` in order.
    #[must_use]
    pub fn f64s(self, values: &[f64]) -> Self {
        values.iter().fold(self, |hash, &v| hash.f64(v))
    }

    /// The hash of everything mixed in so far.
    #[must_use]
    pub fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hash_is_fnv1a_over_little_endian_bytes() {
        // FNV-1a 64 of eight zero bytes.
        assert_eq!(Fingerprint::new().u64(0).finish(), 0xA8C7_F832_281A_39C5);
    }

    #[test]
    fn none_differs_from_every_number() {
        let none = Fingerprint::new().option_f64(None).finish();
        for v in [0.0, -0.0, 1.0, f64::NAN, f64::INFINITY] {
            assert_ne!(none, Fingerprint::new().option_f64(Some(v)).finish());
        }
        assert_eq!(none, Fingerprint::new().u64(0x7ff8_0000_dead_beef).finish());
    }

    #[test]
    fn the_last_bit_matters() {
        let one = Fingerprint::new().f64s(&[1.0, 2.0]).finish();
        let off = Fingerprint::new()
            .f64s(&[1.0, f64::from_bits(2.0f64.to_bits() + 1)])
            .finish();
        assert_ne!(one, off);
    }
}
