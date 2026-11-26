//! A bootstrap over persons for sequential trials.

use rayon::prelude::*;

use crate::error::{Error, Result};

/// How often each person is drawn in one resample of the persons with
/// replacement, for the case weights of a bootstrap replicate.
///
/// The persons are the distinct ids, in increasing order, and the counts are
/// those of `epistat_rng::resample_counts` with the replicate's seed, so a
/// replicate depends on its seed and the set of ids only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersonCounts {
    persons: Vec<u64>,
    counts: Vec<u32>,
}

impl PersonCounts {
    /// The counts of the distinct ids of `ids` in the resample of `seed`.
    #[must_use]
    pub fn draw(ids: &[u64], seed: u64) -> Self {
        let persons = distinct(ids);
        let counts = epistat_rng::resample_counts(persons.len(), seed);
        Self { persons, counts }
    }

    /// The persons, in increasing order.
    #[must_use]
    pub fn persons(&self) -> &[u64] {
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
    pub fn count(&self, id: u64) -> Option<u32> {
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
    /// [`Error::Input`] for an id that was not among the persons drawn from.
    pub fn case_weights(&self, ids: &[u64]) -> Result<Vec<f64>> {
        ids.iter()
            .map(|&id| {
                self.count(id).map(f64::from).ok_or_else(|| {
                    Error::input(format!("id {id} is not among the persons resampled"))
                })
            })
            .collect()
    }
}

fn distinct(ids: &[u64]) -> Vec<u64> {
    let mut persons = ids.to_vec();
    persons.sort_unstable();
    persons.dedup();
    persons
}

/// Apply `analysis` to resamples of the persons of `ids`, one for each seed.
///
/// A resample draws as many persons as there are, with replacement, and gives
/// each person a count ([`PersonCounts`]). The analysis turns the counts into
/// case weights of its rows ([`PersonCounts::case_weights`]): of the rows of the
/// weight models through [`WeightRows::case_weight`](super::WeightRows), of the
/// rows of the trials by multiplying their weights in [`fit_msm`](super::fit_msm),
/// and of the population of the cumulative incidence by repeating or
/// weighting its rows. A logistic model fitted with the counts as case weights
/// has the coefficients of the model fitted to the resampled rows, so this is
/// the bootstrap that expands, weights and fits again, with the person as the
/// unit of resampling.
///
/// The replicates run in parallel, each from its own seed, so the results do
/// not depend on the number of threads; they are in the order of the seeds. A
/// seed for replicate `b` of a run with seed `s` is
/// `epistat_rng::replicate_seed(s, b)`.
pub fn bootstrap_trials<T: Send>(
    ids: &[u64],
    seeds: &[u64],
    analysis: impl Fn(&PersonCounts) -> Result<T> + Sync,
) -> Vec<Result<T>> {
    let persons = distinct(ids);
    seeds
        .par_iter()
        .map(|&seed| {
            let counts = PersonCounts {
                persons: persons.clone(),
                counts: epistat_rng::resample_counts(persons.len(), seed),
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
        let a = PersonCounts::draw(&ids, 5);
        assert_eq!(a.persons(), &[3, 9, 12]);
        assert_eq!(a.counts().iter().sum::<u32>(), 3);
        assert_eq!(a, PersonCounts::draw(&[12, 3, 9], 5));
        let w = a.case_weights(&ids).expect("known ids");
        assert_eq!(w[0], f64::from(a.count(9).expect("drawn from")));
        assert!(a.case_weights(&[4]).is_err());
        let runs = bootstrap_trials(&ids, &[5, 6], |c| Ok(c.counts().to_vec()));
        assert_eq!(runs[0].as_ref().expect("runs"), a.counts());
    }
}
