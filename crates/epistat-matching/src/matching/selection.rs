use super::nearest::NearestSelection;
use super::records::MatchingRecord;
use epistat_rng::{Stream, replicate_seed};

/// How anchors share the candidates when each takes more than one.
///
/// Without replacement and with a ratio above 1, the order in which anchors
/// receive their controls decides who is left short when candidates run out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Assignment {
    /// Each anchor takes all its controls before the next anchor starts.
    #[default]
    Sequential,
    /// Every anchor takes a first control before any takes a second, and so on
    /// up to the ratio, as `matchit()` does.
    Rounds,
}

/// Strategy for selecting one control from a candidate index set.
pub trait SelectionStrategy<R: MatchingRecord> {
    /// Returns selected candidate position index from `candidate_indices`.
    fn select_position(
        &mut self,
        case: &R,
        controls: &[R],
        candidate_indices: &[usize],
    ) -> Option<usize>;

    /// The order in which anchors receive their controls in standard matching
    /// without replacement. Risk-set matching always assigns sequentially.
    fn assignment(&self) -> Assignment {
        Assignment::Sequential
    }

    /// The nearest-neighbour configuration, when this strategy ranks
    /// candidates by a distance. Standard matching then hands the whole job to
    /// the distance-ranked matcher instead of calling
    /// [`Self::select_position`].
    fn nearest(&self) -> Option<&NearestSelection> {
        None
    }

    /// A copy of this strategy for the work unit numbered `index` (a stratum
    /// group or an anchor), so that units matched in parallel do not repeat
    /// each other's draws.
    ///
    /// A strategy with no random state returns a plain copy. A random one
    /// starts a stream derived from its seed and `index`, which depends on
    /// the position of the unit in the input and on nothing else, so the
    /// result is the same on any number of threads.
    #[must_use]
    fn for_unit(&self, _index: usize) -> Self
    where
        Self: Sized + Clone,
    {
        self.clone()
    }

    /// The [`RandomSelection`] that makes this strategy's choices, when its
    /// choice is a uniform draw of a position that reads nothing but the
    /// number of candidates. Risk-set matching then draws from it without
    /// listing the candidates. Answer `Some` only when
    /// [`Self::select_position`] is that selection's own.
    #[doc(hidden)]
    fn as_random(&mut self) -> Option<&mut RandomSelection> {
        None
    }
}

/// Wraps a strategy so that anchors take their controls in rounds.
///
/// The wrapped strategy still chooses each control. Only the order of
/// assignment changes, and only in standard matching without replacement at a
/// ratio above 1.
#[derive(Clone)]
pub struct Rounds<S>(pub S);

impl<R: MatchingRecord, S: SelectionStrategy<R> + Clone> SelectionStrategy<R> for Rounds<S> {
    fn select_position(
        &mut self,
        case: &R,
        controls: &[R],
        candidate_indices: &[usize],
    ) -> Option<usize> {
        self.0.select_position(case, controls, candidate_indices)
    }

    fn assignment(&self) -> Assignment {
        Assignment::Rounds
    }

    fn for_unit(&self, index: usize) -> Self
    where
        Self: Sized + Clone,
    {
        Self(SelectionStrategy::<R>::for_unit(&self.0, index))
    }

    fn as_random(&mut self) -> Option<&mut RandomSelection> {
        self.0.as_random()
    }
}

/// Random control selection using a reproducible RNG seed.
#[derive(Clone)]
pub struct RandomSelection {
    seed: u64,
    rng: Stream,
}

impl RandomSelection {
    /// A random selection that draws from the seeded stream of `seed`.
    #[must_use]
    pub fn seeded(seed: u64) -> Self {
        Self {
            seed,
            rng: Stream::seeded(seed),
        }
    }

    /// The seed the stream started from.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// A position below `len`, or `None` when `len` is 0.
    pub(crate) fn draw(&mut self, len: usize) -> Option<usize> {
        (len > 0).then(|| self.rng.below(len))
    }
}

impl<R: MatchingRecord> SelectionStrategy<R> for RandomSelection {
    fn select_position(
        &mut self,
        _case: &R,
        _controls: &[R],
        candidate_indices: &[usize],
    ) -> Option<usize> {
        self.draw(candidate_indices.len())
    }

    fn for_unit(&self, index: usize) -> Self {
        Self::seeded(replicate_seed(self.seed, index as u64))
    }

    fn as_random(&mut self) -> Option<&mut RandomSelection> {
        Some(self)
    }
}

/// Select the candidate with smallest birth-date distance to the case.
#[derive(Clone)]
pub struct NearestBirthDateSelection;

impl<R: MatchingRecord> SelectionStrategy<R> for NearestBirthDateSelection {
    fn select_position(
        &mut self,
        case: &R,
        controls: &[R],
        candidate_indices: &[usize],
    ) -> Option<usize> {
        candidate_indices
            .iter()
            .enumerate()
            .min_by(|(left_pos, left_idx), (right_pos, right_idx)| {
                let left_distance = (controls[**left_idx].birth_date() - case.birth_date())
                    .num_days()
                    .unsigned_abs();
                let right_distance = (controls[**right_idx].birth_date() - case.birth_date())
                    .num_days()
                    .unsigned_abs();

                left_distance
                    .cmp(&right_distance)
                    .then_with(|| controls[**left_idx].id().cmp(controls[**right_idx].id()))
                    .then_with(|| left_idx.cmp(right_idx))
                    .then_with(|| left_pos.cmp(right_pos))
            })
            .map(|(pos, _)| pos)
    }
}

/// Deterministic selection: lowest candidate index first.
#[derive(Clone)]
pub struct DeterministicSelection;

impl<R: MatchingRecord> SelectionStrategy<R> for DeterministicSelection {
    fn select_position(
        &mut self,
        _case: &R,
        controls: &[R],
        candidate_indices: &[usize],
    ) -> Option<usize> {
        candidate_indices
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                left.cmp(right)
                    .then_with(|| controls[**left].id().cmp(controls[**right].id()))
            })
            .map(|(pos, _)| pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{date, record};

    #[test]
    fn deterministic_selection_returns_first_or_none() {
        let mut strategy = DeterministicSelection;
        let case = record("case", date(2010, 1, 2));
        let controls = vec![record("a", date(2010, 1, 1)), record("b", date(2010, 1, 3))];
        assert_eq!(strategy.select_position(&case, &controls, &[1, 0]), Some(1));
        assert_eq!(strategy.select_position(&case, &controls, &[]), None);
    }

    #[test]
    fn nearest_birth_date_selection_prefers_smallest_distance() {
        let mut strategy = NearestBirthDateSelection;
        let case = record("case", date(2010, 1, 5));
        let controls = vec![
            record("a", date(2010, 1, 1)),
            record("b", date(2010, 1, 6)),
            record("c", date(2010, 1, 10)),
        ];
        assert_eq!(
            strategy.select_position(&case, &controls, &[0, 1, 2]),
            Some(1)
        );
    }

    #[test]
    fn nearest_birth_date_selection_is_stable_on_ties() {
        let mut strategy = NearestBirthDateSelection;
        let case = record("case", date(2010, 1, 5));
        let controls = vec![record("a", date(2010, 1, 4)), record("b", date(2010, 1, 6))];
        assert_eq!(strategy.select_position(&case, &controls, &[1, 0]), Some(1));
    }

    #[test]
    fn nearest_birth_date_selection_breaks_distance_ties_by_id() {
        let mut strategy = NearestBirthDateSelection;
        let case = record("case", date(2010, 1, 5));
        let controls = vec![record("b", date(2010, 1, 4)), record("a", date(2010, 1, 6))];
        assert_eq!(strategy.select_position(&case, &controls, &[0, 1]), Some(1));
    }

    #[test]
    fn random_selection_is_seeded_and_reproducible() {
        let case = record("case", date(2010, 1, 1));
        let controls = vec![record("a", date(2010, 1, 1)); 4];
        let candidates = [0usize, 1, 2, 3];
        let mut left = RandomSelection::seeded(42);
        let mut right = RandomSelection::seeded(42);

        let left_positions = (0..5)
            .map(|_| {
                left.select_position(&case, &controls, &candidates)
                    .expect("position from non-empty candidates")
            })
            .collect::<Vec<_>>();
        let right_positions = (0..5)
            .map(|_| {
                right
                    .select_position(&case, &controls, &candidates)
                    .expect("position from non-empty candidates")
            })
            .collect::<Vec<_>>();
        assert_eq!(left_positions, right_positions);
    }
}
