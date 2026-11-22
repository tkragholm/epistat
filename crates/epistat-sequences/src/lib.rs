//! Distances between state sequences, weighted k-medoids over them, the
//! silhouette of the result and null-model validation of a typology.
//!
//! A sequence is a list of states, one per time step, for example the
//! labour-market state of a person in each year. A typology groups sequences
//! into types of similar course. This crate computes the distances, fits the
//! types, scores the fit and checks it against data that contain no types.
//!
//! Sequences arrive rank-encoded: each state is a `u8` rank in the alphabet,
//! and a set of sequences is one flat slice. Every sequence carries a weight,
//! for example the number of people who share it.
//!
//! # Two families of distance
//!
//! The two families need opposite arrangements, and the crate keeps them apart.
//!
//! - [`optimal_matching`] does not factorise over positions. It is the
//!   Needleman-Wunsch cost of aligning two sequences with substitution and
//!   insertion-deletion costs \[1\] \[2\]. A fit has to read distances pair by
//!   pair, so they are computed once into a [`CondensedMatrix`], the strict
//!   upper triangle in single precision. Its size is quadratic in the number
//!   of distinct sequences.
//! - [`positionwise`] holds Hamming distance and dynamic Hamming distance \[3\],
//!   which compare position `t` with position `t` and never shift a sequence
//!   in time. They factorise, so there is no matrix. The medoid update and the
//!   silhouette reduce to per-position symbol-weight tables and are linear in
//!   the number of sequences.
//!
//! [`silhouette`] turns either family's weighted cluster distance sums into the
//! weighted average silhouette width, [`nulls`] draws sequences from four null
//! models, and [`validation`] runs the replicate loops that compare a fit with
//! those nulls and with refits under replicate weights.
//!
//! # Algorithm
//!
//! Weighted k-medoids with `k` clusters and weights `w` picks `k` sequences as
//! medoids to minimise the objective
//!
//! ```text
//! sum_i w_i * d(i, medoid(i)),    medoid(i) = the nearest medoid to i.
//! ```
//!
//! Each restart seeds by weighted k-means++ \[4\] and then alternates two steps
//! until no medoid moves or `max_iterations` is reached:
//!
//! ```text
//! assign:  label(i) = rank of the nearest medoid to i   (the earliest on a tie)
//! update:  medoid(c) = argmin over members j of c of  sum_{l in c} w_l d(j, l)
//!                      (the earliest member on a tie)
//! ```
//!
//! The first medoid is the first sequence of the input, so a caller that sorts
//! heaviest first seeds the heaviest. Each further medoid is drawn with
//! probability proportional to `d(i, nearest medoid)^2 * w_i`. A step at which
//! every score is zero takes the first sequence not yet chosen and draws
//! nothing. Restart `i` uses the stream `Stream::seeded(seeds[i])` of
//! `epistat-rng`, where the seeds are those of a [`Restarts`]. They are derived
//! from one seed as [`replicate_seed`](epistat_rng::replicate_seed)`(seed, i)`
//! or given one by one, for a caller that has to reproduce earlier seeds. The
//! restart with the least objective is kept, the earliest on a tie
//! ([`best_attempt`]). The restarts run in parallel. Each owns
//! its stream and the choice does not depend on which finishes first, so a fit
//! is the same on any number of threads.
//!
//! # Floating point
//!
//! Sums are taken in a fixed order, and no operation is fused. A medoid cost
//! is a sum over positions and symbols in the order the symbols were first met
//! scanning members in ascending index, which is not the order of the ranks.
//! Where a result is the same in exact arithmetic by two routes, such as the
//! Hamming cost `W - f_p(c_p)` against the sum over the other symbols, the code
//! follows the route stated in each function, because the medoid is chosen by
//! comparing these values and the two routes differ in the last bit. Where a
//! sum is compensated it is [`epistat_stats::sum::compensated`].
//!
//! # Edge cases
//!
//! - Fewer than two distinct labels give a silhouette of 0.
//! - A cluster of zero total weight is skipped in the silhouette, and so is a
//!   sequence whose own cluster has zero weight.
//! - `k` greater than the number of sequences, or zero, is refused.
//! - Optimal matching holds `n (n - 1) / 2` single-precision values, 4 bytes
//!   each. At 10 000 sequences that is 200 MB, and at 100 000 it is 20 GB.
//! - A negative or NaN weight is refused by the validation loops, because the
//!   seeding draws in proportion to it.
//!
//! # References
//!
//! 1. Abbott A, Tsay A. Sequence analysis and optimal matching methods in
//!    sociology: review and prospect. Sociol Methods Res. 2000;29(1):3-33.
//!    doi:10.1177/0049124100029001001
//! 2. Needleman SB, Wunsch CD. A general method applicable to the search for
//!    similarities in the amino acid sequence of two proteins. J Mol Biol.
//!    1970;48(3):443-53. doi:10.1016/0022-2836(70)90057-4
//! 3. Lesnard L. Setting cost in optimal matching to uncover contemporaneous
//!    socio-temporal patterns. Sociol Methods Res. 2010;38(3):389-419.
//!    doi:10.1177/0049124110362526
//! 4. Arthur D, Vassilvitskii S. k-means++: the advantages of careful seeding.
//!    In: Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete
//!    Algorithms. Philadelphia: SIAM; 2007. p. 1027-35.
//! 5. Kaufman L, Rousseeuw PJ. Finding groups in data: an introduction to
//!    cluster analysis. New York: Wiley; 1990. doi:10.1002/9780470316801
//! 6. Studer M, Ritschard G. What matters in differences between life
//!    trajectories: a comparative review of sequence dissimilarity measures.
//!    J R Stat Soc Ser A. 2016;179(2):481-511. doi:10.1111/rssa.12125

mod error;
mod fit;
pub mod nulls;
pub mod optimal_matching;
pub mod positionwise;
mod restarts;
pub mod silhouette;
pub mod validation;

pub use error::SequenceError;
pub use nulls::{NullModel, null_sequences};
pub use optimal_matching::{CondensedMatrix, condensed};
pub use positionwise::Panel;
pub use restarts::Restarts;
pub use validation::{
    Agreement, Distance, FitSettings, RestartSeeds, indel_for, null_silhouettes,
    partition_agreement, refit_agreements, substitution_costs, transition_costs,
};

/// What one seeding and its alternating loop settle.
///
/// Both fits return the best of several of these, and the selection between
/// restarts is the same comparison whichever distance produced them.
#[derive(Clone, Debug, PartialEq)]
pub struct Attempt {
    /// The medoids, as positions in the order the caller gave the sequences,
    /// one per cluster. The cluster number of a sequence is the index of its
    /// medoid here.
    pub medoids: Vec<u32>,
    /// The cluster of each sequence, the rank of its nearest medoid.
    pub labels: Vec<u32>,
    /// The iterations of the alternating loop that ran.
    pub iterations: usize,
    /// Whether the loop ended because no medoid moved.
    pub converged: bool,
    /// The weighted sum of distances to the nearest medoid.
    pub objective: f64,
}

/// The attempt with the least objective, the earliest on a tie.
///
/// Strictly less, scanning in attempt order, so the choice does not depend on
/// which thread finished first. A NaN objective is never less than another, so
/// it wins only when it comes first and no later attempt is less than it.
/// `None` when `attempts` is empty.
#[must_use]
pub fn best_attempt(attempts: Vec<Attempt>) -> Option<Attempt> {
    epistat_optim::multistart::select(attempts.into_iter().map(Some), |candidate, incumbent| {
        candidate.objective < incumbent.objective
    })
    .map(|(_, attempt)| attempt)
}

#[cfg(test)]
mod tests {
    use super::{Attempt, best_attempt};

    fn attempt(objective: f64, iterations: usize) -> Attempt {
        Attempt {
            medoids: vec![0],
            labels: vec![0],
            iterations,
            converged: true,
            objective,
        }
    }

    #[test]
    fn the_least_objective_wins_and_the_earliest_takes_a_tie() {
        let picked = best_attempt(vec![attempt(3.0, 1), attempt(2.0, 2), attempt(2.0, 3)]).unwrap();
        assert_eq!(picked.iterations, 2);
        assert_eq!(picked.objective.to_bits(), 2.0_f64.to_bits());
    }

    #[test]
    fn no_attempt_has_no_best() {
        assert_eq!(best_attempt(Vec::new()), None);
    }
}
