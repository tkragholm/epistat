use crate::matching::MatchingRecord;
use crate::matching::UnitWeightSet;
use crate::types::{Estimand, MatchOutcome, MatchedPair};
use std::collections::{HashMap, HashSet};

/// The weights a balance summary is computed on.
///
/// Each unit has a matching weight. A unit without one has weight zero, so it
/// is unmatched, and a unit with a positive one is matched. The statistics
/// after matching use the matching weights multiplied by the sampling weights
/// and the statistics before matching use the sampling weights alone, as
/// `summary()` of MatchIt does. Without sampling weights every unit has
/// sampling weight one.
///
/// The weights may come from any method: pair counts, subclassification, the
/// exact method, full matching, or none at all ([`BalanceWeights::unmatched`]).
#[derive(Debug, Clone)]
pub struct BalanceWeights {
    /// The estimand that chooses the standardiser of the differences.
    pub estimand: Estimand,
    /// Matching weight of each unit, by group and identifier.
    pub matching: UnitWeightSet,
    /// Sampling weight of each unit by identifier. A unit without one has
    /// weight one. `None` when there are no sampling weights.
    pub sampling: Option<HashMap<String, f64>>,
    /// Identifiers of units discarded before matching, which the sample size
    /// table counts apart from the unmatched ones.
    pub discarded: HashSet<String>,
    /// Matched pairs, for the pair distances. Empty when the method does not
    /// pair units.
    pub pairs: Vec<MatchedPair>,
}

/// A weight that cannot be used.
#[derive(Debug, Clone, PartialEq)]
pub enum BalanceInputError {
    /// A matching or sampling weight is negative or not finite.
    InvalidWeight {
        /// Identifier of the unit.
        id: String,
        /// The weight.
        weight: f64,
    },
}

impl std::fmt::Display for BalanceInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWeight { id, weight } => write!(
                f,
                "the weight of unit {id} is {weight}, which is negative or not finite"
            ),
        }
    }
}

impl std::error::Error for BalanceInputError {}

impl BalanceWeights {
    /// Weights from matching weights alone.
    #[must_use]
    pub fn new(estimand: Estimand, matching: UnitWeightSet) -> Self {
        Self {
            estimand,
            matching,
            sampling: None,
            discarded: HashSet::new(),
            pairs: Vec::new(),
        }
    }

    /// No matching: every unit has weight one, so the statistics after
    /// matching equal those before.
    #[must_use]
    pub fn unmatched<R: MatchingRecord>(cases: &[R], controls: &[R], estimand: Estimand) -> Self {
        let ones = |records: &[R]| {
            records
                .iter()
                .map(|record| (record.id().to_owned(), 1.0))
                .collect()
        };
        Self::new(
            estimand,
            UnitWeightSet {
                anchor: ones(cases),
                candidate: ones(controls),
            },
        )
    }

    /// The matching weights of a pair-based outcome for its realised estimand,
    /// with its pairs.
    #[must_use]
    pub fn from_outcome(outcome: &MatchOutcome) -> Self {
        let mut weights = Self::new(
            outcome.diagnostics.realized_estimand,
            outcome.match_weights_for_realized_estimand(),
        );
        weights.pairs.clone_from(&outcome.pairs);
        weights
    }

    /// One weight per unit identifier, as subclassification and the exact method
    /// give them, split into the two groups. Units without a weight are
    /// unmatched.
    #[must_use]
    pub fn from_unit_weights<R: MatchingRecord>(
        cases: &[R],
        controls: &[R],
        weights: &HashMap<String, f64>,
        estimand: Estimand,
    ) -> Self {
        let pick = |records: &[R]| {
            records
                .iter()
                .filter_map(|record| {
                    weights
                        .get(record.id())
                        .map(|weight| (record.id().to_owned(), *weight))
                })
                .collect()
        };
        Self::new(
            estimand,
            UnitWeightSet {
                anchor: pick(cases),
                candidate: pick(controls),
            },
        )
    }

    /// Add sampling weights.
    #[must_use]
    pub fn with_sampling_weights(mut self, sampling: HashMap<String, f64>) -> Self {
        self.sampling = Some(sampling);
        self
    }

    /// Record the units discarded before matching.
    #[must_use]
    pub fn with_discarded(mut self, discarded: impl IntoIterator<Item = String>) -> Self {
        self.discarded = discarded.into_iter().collect();
        self
    }

    /// Give the matched pairs, for the pair distances.
    #[must_use]
    pub fn with_pairs(mut self, pairs: Vec<MatchedPair>) -> Self {
        self.pairs = pairs;
        self
    }

    /// Check that every weight is finite and not negative.
    ///
    /// # Errors
    ///
    /// [`BalanceInputError::InvalidWeight`] for the weight that is not, the one
    /// of the unit whose identifier sorts first when there are several.
    pub fn validate(&self) -> Result<(), BalanceInputError> {
        let mut checks: Vec<(&String, f64)> = Vec::new();
        for group in [&self.matching.anchor, &self.matching.candidate] {
            checks.extend(group.iter().map(|(id, weight)| (id, *weight)));
        }
        if let Some(sampling) = &self.sampling {
            checks.extend(sampling.iter().map(|(id, weight)| (id, *weight)));
        }
        let mut bad: Vec<(&String, f64)> = checks
            .into_iter()
            .filter(|(_, weight)| !weight.is_finite() || *weight < 0.0)
            .collect();
        bad.sort_by(|left, right| left.0.cmp(right.0));
        match bad.first() {
            Some((id, weight)) => Err(BalanceInputError::InvalidWeight {
                id: (*id).clone(),
                weight: *weight,
            }),
            None => Ok(()),
        }
    }
}
