//! The settings of a bootstrap: how many replicates, from which root seed.
//!
//! A randomised call takes a [`Bootstrap`]. Replicate `r` draws from the seed
//! `epistat_rng::replicate_seed(seed, r)`, so its draws depend on the root seed
//! and `r` alone, not on the thread that runs it or the order the replicates
//! run in. The stream a seed gives is part of each method's API.
//!
//! The root seed is a [`Seed`], not a bare integer, so the two numbers of a
//! bootstrap cannot be given in the wrong order:
//!
//! ```
//! use epistat_core::{Bootstrap, Seed};
//!
//! let bootstrap = Bootstrap::new(999, Seed(20_381));
//! assert_eq!(bootstrap.seed.value(), 20_381);
//! assert_eq!(bootstrap.to_string(), "999 replicates from seed 20381");
//! ```
//!
//! `Bootstrap::new(Seed(20_381), 999)` does not compile. There is no default:
//! a bootstrap without a seed the caller chose does not compile, and two
//! analyses cannot share a seed by accident.

use core::fmt;

/// The root seed of a randomised call.
///
/// A seed is the number the caller chose, kept as it is: `Seed(20_381)` gives
/// the draws the integer `20_381` gives the random number generator, and every
/// stream derived from it follows from that integer. It serialises as the bare
/// integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Seed(pub u64);

impl Seed {
    /// The seed as an integer, the argument of the random number generator.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

impl From<u64> for Seed {
    fn from(seed: u64) -> Self {
        Self(seed)
    }
}

impl From<Seed> for u64 {
    fn from(seed: Seed) -> Self {
        seed.0
    }
}

impl fmt::Display for Seed {
    /// The integer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// The number of bootstrap replicates and the root seed of their draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct Bootstrap {
    /// The number of replicates, R's `R` or `B`.
    pub replicates: usize,
    /// The root seed; replicate `r` draws from `replicate_seed(seed, r)`.
    pub seed: Seed,
}

impl Bootstrap {
    /// `replicates` replicates from the root seed `seed`.
    #[must_use]
    pub const fn new(replicates: usize, seed: Seed) -> Self {
        Self { replicates, seed }
    }
}

impl fmt::Display for Bootstrap {
    /// `999 replicates from seed 20381`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let noun = if self.replicates == 1 {
            "replicate"
        } else {
            "replicates"
        };
        write!(f, "{} {noun} from seed {}", self.replicates, self.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bootstrap_is_its_two_numbers() {
        let bootstrap = Bootstrap::new(999, Seed(20_381));
        assert_eq!(
            bootstrap,
            Bootstrap {
                replicates: 999,
                seed: Seed(20_381)
            }
        );
        assert_eq!(u64::from(bootstrap.seed), 20_381);
        assert_eq!(Seed::from(20_381_u64), bootstrap.seed);
        assert_eq!(
            Bootstrap::new(1, Seed(0)).to_string(),
            "1 replicate from seed 0"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_names_both_fields_and_refuses_others() {
        let bootstrap = Bootstrap::new(200, Seed(u64::MAX));
        let text = serde_json::to_string(&bootstrap).unwrap();
        assert_eq!(text, r#"{"replicates":200,"seed":18446744073709551615}"#);
        assert_eq!(serde_json::from_str::<Bootstrap>(&text).unwrap(), bootstrap);
        assert!(
            serde_json::from_str::<Bootstrap>(r#"{"replicates":2,"seed":1,"reps":3}"#).is_err()
        );
        assert!(serde_json::from_str::<Bootstrap>(r#"{"replicates":2}"#).is_err());
        assert_eq!(serde_json::to_string(&Seed(7)).unwrap(), "7");
    }
}
