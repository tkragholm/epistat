//! The settings of a bootstrap: how many replicates, from which root seed.
//!
//! A randomised call takes a [`Bootstrap`]. Replicate `r` draws from the seed
//! `epistat_rng::replicate_seed(seed, r)`, so its draws depend on the root seed
//! and `r` alone, not on the thread that runs it or the order the replicates
//! run in. The stream a seed gives is part of each method's API.
//!
//! There is no default: a bootstrap without a seed the caller chose does not
//! compile, and two analyses cannot share a seed by accident.

/// The number of bootstrap replicates and the root seed of their draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct Bootstrap {
    /// The number of replicates, R's `R` or `B`.
    pub replicates: usize,
    /// The root seed; replicate `r` draws from `replicate_seed(seed, r)`.
    pub seed: u64,
}

impl Bootstrap {
    /// `replicates` replicates from the root seed `seed`.
    #[must_use]
    pub const fn new(replicates: usize, seed: u64) -> Self {
        Self { replicates, seed }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bootstrap_is_its_two_numbers() {
        let bootstrap = Bootstrap::new(999, 20_381);
        assert_eq!(
            bootstrap,
            Bootstrap {
                replicates: 999,
                seed: 20_381
            }
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_names_both_fields_and_refuses_others() {
        let bootstrap = Bootstrap::new(200, u64::MAX);
        let text = serde_json::to_string(&bootstrap).unwrap();
        assert_eq!(text, r#"{"replicates":200,"seed":18446744073709551615}"#);
        assert_eq!(serde_json::from_str::<Bootstrap>(&text).unwrap(), bootstrap);
        assert!(
            serde_json::from_str::<Bootstrap>(r#"{"replicates":2,"seed":1,"reps":3}"#).is_err()
        );
        assert!(serde_json::from_str::<Bootstrap>(r#"{"replicates":2}"#).is_err());
    }
}
