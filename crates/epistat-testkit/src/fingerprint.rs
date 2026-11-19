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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn u64(mut self, value: u64) -> Self {
        for byte in value.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
        }
        self
    }

    #[must_use]
    pub fn f64(self, value: f64) -> Self {
        self.u64(value.to_bits())
    }

    #[must_use]
    pub fn f64s(self, values: &[f64]) -> Self {
        values.iter().fold(self, |hash, &v| hash.f64(v))
    }

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
    fn the_last_bit_matters() {
        let one = Fingerprint::new().f64s(&[1.0, 2.0]).finish();
        let off = Fingerprint::new()
            .f64s(&[1.0, f64::from_bits(2.0f64.to_bits() + 1)])
            .finish();
        assert_ne!(one, off);
    }
}
