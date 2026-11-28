//! The seeds of the restarts of a k-medoids fit.

use epistat_rng::replicate_seed;

use crate::Error;

/// One seed per restart of a k-medoids fit.
///
/// Restart `i` seeds its weighted k-means++ draws from
/// `Stream::seeded(seeds[i])`, and the restart with the least objective is
/// kept. The seeds are given explicitly or derived from one with
/// [`from_seed`](Self::from_seed). There is at least one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Restarts(Vec<u64>);

impl Restarts {
    /// `count` restarts whose seeds are [`replicate_seed`]`(seed, i)` for
    /// `i` in `0..count`, so that the streams of two restarts do not overlap
    /// and the seeds do not depend on `count`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `count` is zero.
    pub fn from_seed(seed: u64, count: usize) -> Result<Self, Error> {
        Self::from_seeds((0..count as u64).map(|i| replicate_seed(seed, i)).collect())
    }

    /// Restarts with exactly these seeds, in this order. The earliest of
    /// equally good restarts is kept, so the order can matter.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `seeds` is empty.
    pub fn from_seeds(seeds: Vec<u64>) -> Result<Self, Error> {
        if seeds.is_empty() {
            return Err(Error::NoRestarts);
        }
        Ok(Self(seeds))
    }

    /// The seeds, one per restart.
    #[must_use]
    pub fn seeds(&self) -> &[u64] {
        &self.0
    }

    /// The number of restarts, at least one.
    #[must_use]
    #[expect(
        clippy::len_without_is_empty,
        reason = "there is always at least one restart"
    )]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_seeds_are_replicate_seeds_and_do_not_depend_on_the_count() {
        let three = Restarts::from_seed(9, 3).unwrap();
        let five = Restarts::from_seed(9, 5).unwrap();
        assert_eq!(three.seeds(), &five.seeds()[..3]);
        assert_eq!(three.seeds()[2], replicate_seed(9, 2));
        assert_eq!(three.len(), 3);
    }

    #[test]
    fn no_restart_is_an_error() {
        assert!(Restarts::from_seed(1, 0).is_err());
        assert!(Restarts::from_seeds(Vec::new()).is_err());
        assert_eq!(Restarts::from_seeds(vec![4, 4]).unwrap().seeds(), [4, 4]);
    }
}
