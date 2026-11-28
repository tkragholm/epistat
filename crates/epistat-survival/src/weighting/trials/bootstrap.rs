//! A bootstrap over persons for sequential trials.

use epistat_core::Bootstrap;
use rayon::prelude::*;

use crate::error::{Error, Result};

/// How often each person is drawn in one resample of the persons with
/// replacement, for the case weights of a bootstrap replicate.
///
/// The persons are the distinct unit codes of the ids, in increasing order, and the counts of
/// replicate `r` of a [`Bootstrap`] are those of
/// `epistat_inference::bootstrap::unit_counts_of`, that is
/// `resample_counts(n, replicate_seed(seed, r))`, so a replicate depends on
/// the root seed, its number and the set of ids only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersonCounts {
    persons: Vec<u32>,
    counts: Vec<u32>,
}

impl PersonCounts {
    /// The counts of the distinct ids of `ids` in replicate `replicate` of
    /// `bootstrap`, the counts [`bootstrap_trials`] gives that replicate.
    #[must_use]
    pub fn draw(ids: &[u32], bootstrap: Bootstrap, replicate: usize) -> Self {
        let persons = distinct(ids);
        let counts =
            epistat_inference::bootstrap::unit_counts_of(persons.len(), bootstrap, replicate);
        Self { persons, counts }
    }

    /// The persons, in increasing order.
    #[must_use]
    pub fn persons(&self) -> &[u32] {
        &self.persons
    }

    /// The count of each person, in the order of [`Self::persons`].
    #[must_use]
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// The count of the person `id`, `None` for an id that was not among the
    /// persons drawn from.
    #[must_use]
    pub fn count(&self, id: u32) -> Option<u32> {
        self.persons
            .binary_search(&id)
            .ok()
            .map(|at| self.counts[at])
    }

    /// The case weight of each row of an id column: the count of its person.
    /// The rows of the person-period data, of the trials and of the population
    /// of a cumulative incidence all take their weights from the same counts.
    ///
    /// # Errors
    ///
    /// An input error for an id that was not among the persons drawn from.
    pub fn case_weights(&self, ids: &[u32]) -> Result<Vec<f64>> {
        ids.iter()
            .map(|&id| {
                self.count(id).map(f64::from).ok_or_else(|| {
                    Error::out_of_range(format!("id {id} is not among the persons resampled"))
                })
            })
            .collect()
    }
}

fn distinct(ids: &[u32]) -> Vec<u32> {
    let mut persons = ids.to_vec();
    persons.sort_unstable();
    persons.dedup();
    persons
}

/// Apply `analysis` to `bootstrap.replicates` resamples of the persons of
/// `ids`.
///
/// A resample draws as many persons as there are, with replacement, and gives
/// each person a count ([`PersonCounts`]). The analysis turns the counts into
/// case weights of its rows ([`PersonCounts::case_weights`], or
/// [`Cloned::case_weights`](crate::weighting::clone::Cloned::case_weights)
/// for the clones of a clone-censor-weight analysis): of the rows of the
/// weight models through [`WeightRows::case_weight`](super::WeightRows), of the
/// rows of the trials by multiplying their weights in [`fit_msm`](super::fit_msm),
/// and of the population of the cumulative incidence by repeating or
/// weighting its rows. A logistic model fitted with the counts as case weights
/// has the coefficients of the model fitted to the resampled rows, so this is
/// the bootstrap that expands, weights and fits again, with the person as the
/// unit of resampling.
///
/// The persons are the distinct ids in increasing order. Replicate `r` draws
/// its counts with `epistat_inference::bootstrap::unit_counts_of`, that is
/// `resample_counts(n, replicate_seed(bootstrap.seed, r))`, one replicate at
/// a time so that the counts of every replicate are not held at once. The
/// replicates run in parallel and each depends on its seed only, so the
/// results do not depend on the number of threads; they are in replicate
/// order.
pub fn bootstrap_trials<T: Send>(
    ids: &[u32],
    bootstrap: Bootstrap,
    analysis: impl Fn(&PersonCounts) -> Result<T> + Sync,
) -> Vec<Result<T>> {
    let persons = distinct(ids);
    (0..bootstrap.replicates)
        .into_par_iter()
        .map(|r| {
            let counts = PersonCounts {
                persons: persons.clone(),
                counts: epistat_inference::bootstrap::unit_counts_of(persons.len(), bootstrap, r),
            };
            analysis(&counts)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_follow_the_ids_and_the_seed() {
        let ids = [9, 9, 3, 3, 3, 12];
        let a = PersonCounts::draw(&ids, Bootstrap::new(1, 5), 0);
        assert_eq!(a.persons(), &[3, 9, 12]);
        assert_eq!(a.counts().iter().sum::<u32>(), 3);
        assert_eq!(a, PersonCounts::draw(&[12, 3, 9], Bootstrap::new(1, 5), 0));
        let w = a.case_weights(&ids).expect("known ids");
        assert_eq!(w[0], f64::from(a.count(9).expect("drawn from")));
        assert!(a.case_weights(&[4]).is_err());
        let runs = bootstrap_trials(&ids, Bootstrap::new(2, 5), |c| Ok(c.counts().to_vec()));
        for (r, run) in runs.iter().enumerate() {
            let drawn = PersonCounts::draw(&ids, Bootstrap::new(2, 5), r);
            assert_eq!(run.as_ref().expect("runs"), drawn.counts());
        }
    }
}
