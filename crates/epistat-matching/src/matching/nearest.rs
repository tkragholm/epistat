//! Greedy nearest neighbour matching, ranked by a distance.
//!
//! Standard matching with the other strategies restricts candidates by a
//! distance and then picks among those that remain. Here the distance ranks
//! them: each anchor takes the candidate nearest to it that no constraint
//! refuses, as `matchit(method = "nearest")` does.
//!
//! # Rules
//!
//! * **Ranking.** The distance is the configured one: the absolute difference
//!   of two scores (a propensity score on either scale, or any numeric score),
//!   the absolute difference of two birth dates, or the Mahalanobis distance.
//! * **Ties.** Candidates at the same distance are ranked by their position in
//!   the candidate slice, lowest first. `matchit()` documents the same rule
//!   but does not follow it where candidates at the same distance lie on the
//!   same side of the anchor's score; see `tests/r_nearest.rs`.
//! * **Calipers.** A caliper on the distance is inclusive. With
//!   [`NearestSelection::with_std_caliper`] its width is in standard deviation
//!   units of the distance variable, computed over all anchors and candidates
//!   that common-support trimming leaves, with `n - 1` in the denominator.
//!   Calipers on other fields are ordinary constraints.
//! * **Mahalanobis within a caliper.** [`NearestSelection::with_mahvars`] ranks
//!   by a second distance and keeps the configured one as a caliper and as the
//!   scale of the anchor order, which is `mahvars` in `matchit()`.
//! * **Order of anchors.** [`MatchOrder`]. The order of anchors with equal
//!   distance is their order in the anchor slice.
//! * **Assignment.** Without replacement and with a ratio above 1, every
//!   anchor takes a first control before any takes a second, and so on, unless
//!   [`Assignment::Sequential`] is asked for.
//! * **Variable ratio.** [`VariableRatio`] sets the number of controls of each
//!   anchor from its score before matching, as `min.controls` and
//!   `max.controls` do in `matchit()`. It needs a score distance.
//! * **Calipers on a covariate distance.** `matchit()` accepts calipers only
//!   on named covariates when the distance is Mahalanobis or Euclidean. A
//!   caliper in standard deviation units of such a distance has no score to
//!   take the deviation of, and the outcome is invalid. A caliper in the
//!   distance's own units is accepted.
//! * **Ratio fallbacks.** Fallback ratios choose a smaller ratio for a stratum
//!   that cannot reach the full one. Ranking by distance in rounds gives every
//!   anchor its controls one at a time and leaves an anchor with fewer when
//!   the candidates run out, and a variable ratio sets each anchor's number
//!   itself, so a tier has nothing left to decide. A job that sets fallback
//!   ratios with the nearest mode returns an invalid outcome (`nearest_options`).
//! * **Exact strata.** Anchors and candidates are split by the exact-match
//!   keys first, and the strata are matched in parallel. Within a stratum the
//!   matching is sequential, since each assignment depends on the last.
//!
//! # Cost
//!
//! With a score distance, a stratum's candidates are sorted once, and an anchor
//! finds its nearest by walking out from its own score. Candidates already used
//! are skipped in near-constant time. The walk stops at the caliper, so a
//! caliper bounds the work per anchor. Without one, an anchor whose nearest
//! candidates are all refused by a constraint walks as far as it must.
//!
//! With the Mahalanobis distance each anchor computes the distance to every
//! available candidate that passes the caliper, which costs the number of
//! such candidates per anchor and per round. A caliper on a score
//! (`mahvars`) narrows that range by binary search. Without one, a stratum of
//! `a` anchors and `c` candidates costs `a * c` distances.

use super::constraints::{ConstraintContext, ConstraintGroup, UsedControlsVec, unique_value};
use super::distance::{
    ConfiguredMetric, DistanceConfig, DistanceMetric, IdMapMahalanobisDistance, ScalarLookup,
};
use super::engine::{
    TrimmedIndexConfig, build_outcome, finalize_estimand_diagnostics, intern_unique,
    invalid_criteria_outcome,
};
use super::index::StrataValues;
use super::records::MatchingRecord;
use super::selection::{Assignment, SelectionStrategy};
use crate::constraints::sample_sd;
use crate::types::{
    ControlIdx, Estimand, InvalidCriteriaReason, MatchDiagnostics, MatchOutcome, MatchedPair,
    MatchingCriteria, UniqueValueId,
};
use epistat_rng::Stream;
use ordered_float::OrderedFloat;
use rapidhash::RapidHashMap;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// The order in which anchors are matched.
///
/// The names and meanings are those of `m.order` in `matchit()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchOrder {
    /// By decreasing distance score, so the anchors hardest to match come
    /// first. Needs a score distance.
    Largest,
    /// By increasing distance score. Needs a score distance.
    Smallest,
    /// In the order of the anchor slice.
    Data,
    /// In an order drawn from the job's seed.
    Random,
    /// By the distance to the nearest available candidate, smallest first.
    ///
    /// Every anchor finds its nearest candidate, the pairs are taken in order
    /// of their distance, and an anchor whose candidate has been used meanwhile
    /// finds the next nearest and takes its place in the order by the new
    /// distance. Needs rounds, and has no effect with replacement.
    Closest,
    /// As [`Self::Closest`], with the largest distance first.
    Farthest,
}

/// Nearest neighbour matching ranked by a distance, as a selection strategy.
///
/// Make it with [`MatchJob::with_nearest`](crate::MatchJob::with_nearest).
/// Standard matching hands the whole job to the distance-ranked matcher when
/// the strategy is this one. The default order of anchors is as in `matchit()`:
/// largest score first for a propensity score or any numeric score (smallest
/// first when the controls are the anchors, estimand `Atc`), and the order of
/// the anchor slice for the birth date and the Mahalanobis distance.
#[derive(Debug, Clone)]
pub struct NearestSelection {
    seed: u64,
    order: Option<MatchOrder>,
    assignment: Assignment,
    std_caliper: bool,
    mahvars: Option<DistanceConfig>,
    variable_ratio: Option<VariableRatio>,
}

/// Why a variable ratio is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableRatioError {
    /// The average is not a finite number above 1.
    AverageNotAboveOne,
    /// The minimum is below 1.
    MinimumBelowOne,
    /// The minimum is not below the average.
    MinimumNotBelowAverage,
    /// The maximum is not above the average.
    MaximumNotAboveAverage,
}

impl std::fmt::Display for VariableRatioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::AverageNotAboveOne => "the average ratio must be a number above 1",
            Self::MinimumBelowOne => "the minimum number of controls must be at least 1",
            Self::MinimumNotBelowAverage => {
                "the minimum number of controls must be below the average ratio"
            }
            Self::MaximumNotAboveAverage => {
                "the maximum number of controls must be above the average ratio"
            }
        })
    }
}

impl std::error::Error for VariableRatioError {}

/// Extremal variable ratio matching (Ming and Rosenbaum, 2000), the
/// `min.controls` and `max.controls` of `matchit()`.
///
/// Anchors take between `min` and `max` controls, and in total
/// `round(average * anchors)` controls are asked for (halves rounded to even,
/// as in R), as many as fixed ratio matching at the average would ask for. The
/// numbers are set before matching, from the scores alone and without regard to
/// calipers or trimming: as many anchors as possible get `max`, one anchor
/// gets the number in between that makes the total, never below `min`, and the
/// others get `min`. `matchit()` gives that one anchor fewer than `min` for
/// some totals when `min` is above 1, which the minimum forbids; the numbers
/// agree otherwise. The anchors are ranked by score. When the anchors' mean score is
/// above the candidates', the anchors with the highest scores, those furthest
/// from the candidates, get the fewest controls, and otherwise the anchors with
/// the lowest. Equal scores rank by position in the anchor slice. The ranking
/// covers every anchor of the slice, trimmed to common support or not, as the
/// mean does with the candidates.
///
/// Used with [`NearestSelection::with_variable_ratio`], it replaces the job's
/// ratio. It needs a score distance, a propensity score or any numeric score,
/// possibly with `mahvars`, and the job must not set fallback ratios.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariableRatio {
    average: f64,
    min: usize,
    max: usize,
}

impl VariableRatio {
    /// A variable ratio with the average (`ratio`) and the least and most
    /// controls per anchor (`min.controls`, `max.controls`).
    ///
    /// # Errors
    ///
    /// Returns a [`VariableRatioError`] unless `1 <= min < average < max`.
    pub fn new(average: f64, min: usize, max: usize) -> Result<Self, VariableRatioError> {
        if !(average.is_finite() && average > 1.0) {
            return Err(VariableRatioError::AverageNotAboveOne);
        }
        if min < 1 {
            return Err(VariableRatioError::MinimumBelowOne);
        }
        #[expect(clippy::cast_precision_loss, reason = "control counts are small")]
        let (low, high) = (min as f64, max as f64);
        if low >= average {
            return Err(VariableRatioError::MinimumNotBelowAverage);
        }
        if high <= average {
            return Err(VariableRatioError::MaximumNotAboveAverage);
        }
        Ok(Self { average, min, max })
    }

    /// The numbers of controls of `anchors` anchors, in the order of the
    /// ranking by score (fewest first), or `None` when no assignment of the
    /// total exists within the bounds.
    fn counts(&self, anchors: usize) -> Option<Vec<usize>> {
        let n = i64::try_from(anchors).ok()?;
        if n == 0 {
            return None;
        }
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            reason = "the total is a count of units"
        )]
        let total = (self.average * anchors as f64).round_ties_even() as i64;
        let low = i64::try_from(self.min).ok()?;
        let high = i64::try_from(self.max).ok()?;
        // `at_max` anchors get the maximum and `at_min` the minimum, and one
        // anchor `between` the two, so that the numbers add up to the total:
        // at_max is the most anchors the maximum can be given to while every
        // other anchor still gets at least the minimum.
        let at_max = (total - low * n).div_euclid(high - low);
        let at_min = n - at_max - 1;
        if at_min < 0 || at_max < 0 {
            return None;
        }
        let between = total - (low * at_min + high * at_max);
        std::iter::repeat_n(low, usize::try_from(at_min).ok()?)
            .chain(std::iter::once(between))
            .chain(std::iter::repeat_n(high, usize::try_from(at_max).ok()?))
            .map(|count| usize::try_from(count).ok())
            .collect()
    }

    /// The number of controls of every anchor, by position in the slice.
    fn per_anchor<R: MatchingRecord>(
        &self,
        anchors: &[R],
        candidates: &[R],
        lookup: ScalarLookup<'_>,
    ) -> Option<Vec<usize>> {
        let counts = self.counts(anchors.len())?;
        let anchor_scores: Vec<f64> = anchors
            .iter()
            .map(|record| lookup.value(record).filter(|value| value.is_finite()))
            .collect::<Option<_>>()?;
        let mut candidate_sum = 0.0;
        for record in candidates {
            candidate_sum += lookup.value(record).filter(|value| value.is_finite())?;
        }
        let mut anchor_sum = 0.0;
        for value in &anchor_scores {
            anchor_sum += value;
        }
        #[expect(clippy::cast_precision_loss, reason = "counts of units")]
        let anchors_above =
            anchor_sum / anchors.len() as f64 > candidate_sum / candidates.len() as f64;
        let mut ranked: Vec<usize> = (0..anchors.len()).collect();
        ranked.sort_by(|&a, &b| {
            let by_score = if anchors_above {
                cmp_f64(anchor_scores[b], anchor_scores[a])
            } else {
                cmp_f64(anchor_scores[a], anchor_scores[b])
            };
            by_score.then(a.cmp(&b))
        });
        let mut per_anchor = vec![0_usize; anchors.len()];
        for (position, anchor) in ranked.into_iter().enumerate() {
            per_anchor[anchor] = counts[position];
        }
        Some(per_anchor)
    }
}

impl NearestSelection {
    /// Nearest neighbour matching with rounds, a random order drawn from `seed`
    /// if one is asked for, and every caliper in the units of its distance.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            seed,
            order: None,
            assignment: Assignment::Rounds,
            std_caliper: false,
            mahvars: None,
            variable_ratio: None,
        }
    }

    /// Set the order of anchors, replacing the default.
    #[must_use]
    pub const fn with_order(mut self, order: MatchOrder) -> Self {
        self.order = Some(order);
        self
    }

    /// Choose rounds or sequential assignment.
    #[must_use]
    pub const fn with_assignment(mut self, assignment: Assignment) -> Self {
        self.assignment = assignment;
        self
    }

    /// Read the distance caliper in standard deviations of the distance
    /// variable instead of its own units. Needs a score distance.
    #[must_use]
    pub const fn with_std_caliper(mut self, std_caliper: bool) -> Self {
        self.std_caliper = std_caliper;
        self
    }

    /// Match a variable number of controls to each anchor instead of the
    /// job's ratio. See [`VariableRatio`].
    #[must_use]
    pub const fn with_variable_ratio(mut self, ratio: VariableRatio) -> Self {
        self.variable_ratio = Some(ratio);
        self
    }

    /// Rank candidates by this distance, usually Mahalanobis, and use the
    /// job's own distance only as a caliper and for the order of anchors.
    #[must_use]
    pub fn with_mahvars(mut self, ranking: DistanceConfig) -> Self {
        self.mahvars = Some(ranking);
        self
    }
}

impl<R: MatchingRecord> SelectionStrategy<R> for NearestSelection {
    /// The lowest index. Standard matching never calls this, since it ranks
    /// by distance; risk-set matching does not rank by distance.
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

    fn assignment(&self) -> Assignment {
        self.assignment
    }

    fn nearest(&self) -> Option<&NearestSelection> {
        Some(self)
    }
}

/// The inputs of a distance-ranked standard match.
pub(crate) struct NearestRequest<'a, R: MatchingRecord, G: ?Sized> {
    pub anchors: &'a [R],
    pub candidates: &'a [R],
    pub criteria: &'a MatchingCriteria,
    pub constraints: &'a G,
    pub distance_config: Option<&'a DistanceConfig>,
    pub trimmed: TrimmedIndexConfig<'a>,
}

/// How candidates are ranked for an anchor.
enum Rank<'a> {
    /// The distance is the absolute difference of one number per record, so
    /// candidates are sorted on it.
    Scalar {
        lookup: ScalarLookup<'a>,
        caliper: Option<f64>,
    },
    /// Distances are computed pair by pair. A caliper on one number per record
    /// narrows the candidates first.
    Pairs {
        gate: Option<(ScalarLookup<'a>, f64)>,
        metric: ConfiguredMetric<'a>,
        caliper: Option<f64>,
    },
}

struct Plan<'a> {
    order: MatchOrder,
    assignment: Assignment,
    order_lookup: Option<ScalarLookup<'a>>,
    rank: Rank<'a>,
    /// The number of controls of each anchor by position in the anchor slice,
    /// where the ratio is variable.
    ratios: Option<Vec<usize>>,
    /// The most controls any anchor takes.
    max_ratio: usize,
}

fn invalid(anchors: usize) -> MatchOutcome {
    invalid_criteria_outcome(anchors, InvalidCriteriaReason::NearestOptions)
}

/// Settle the options against the distance, or say they contradict it.
fn resolve_plan<'a, R: MatchingRecord, G: ?Sized>(
    request: &NearestRequest<'a, R, G>,
    nearest: &'a NearestSelection,
    order_config: &'a DistanceConfig,
) -> Option<Plan<'a>> {
    let criteria = request.criteria;
    let order_lookup = order_config.scalar();

    let default_order = if matches!(order_config, DistanceConfig::PropensityScoreMap { .. }) {
        if criteria.estimand == Estimand::Atc {
            MatchOrder::Smallest
        } else {
            MatchOrder::Largest
        }
    } else {
        MatchOrder::Data
    };
    let mut order = nearest.order.unwrap_or(default_order);
    if matches!(order, MatchOrder::Largest | MatchOrder::Smallest) && order_lookup.is_none() {
        return None;
    }
    if criteria.allow_replacement && matches!(order, MatchOrder::Closest | MatchOrder::Farthest) {
        order = MatchOrder::Data;
    }
    if matches!(order, MatchOrder::Closest | MatchOrder::Farthest)
        && nearest.assignment == Assignment::Sequential
    {
        return None;
    }

    let scale = if nearest.std_caliper && order_config.caliper().is_some() {
        let lookup = order_lookup?;
        let values: Vec<f64> = request
            .anchors
            .iter()
            .enumerate()
            .filter(|(idx, _)| !is_trimmed(request.trimmed.anchors, *idx))
            .map(|(_, record)| record)
            .chain(
                request
                    .candidates
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| !is_trimmed(request.trimmed.candidates, *idx))
                    .map(|(_, record)| record),
            )
            .filter_map(|record| lookup.value(record))
            .collect();
        sample_sd(&values)?
    } else {
        1.0
    };

    let rank = match &nearest.mahvars {
        None => match order_lookup {
            Some(lookup) => Rank::Scalar {
                lookup,
                caliper: order_config.caliper().map(|width| width * scale),
            },
            None => Rank::Pairs {
                gate: None,
                metric: order_config.metric(),
                caliper: order_config.caliper(),
            },
        },
        Some(ranking) => {
            let gate = match (order_config.caliper(), order_lookup) {
                (None, _) => None,
                (Some(width), Some(lookup)) => Some((lookup, width * scale)),
                (Some(_), None) => return None,
            };
            Rank::Pairs {
                gate,
                metric: ranking.metric(),
                caliper: ranking.caliper(),
            }
        }
    };

    let ratios = match &nearest.variable_ratio {
        None => None,
        Some(variable) => {
            Some(variable.per_anchor(request.anchors, request.candidates, order_lookup?)?)
        }
    };
    let max_ratio = ratios.as_ref().map_or(criteria.match_ratio, |ratios| {
        ratios.iter().copied().max().unwrap_or(0)
    });

    Some(Plan {
        order,
        assignment: nearest.assignment,
        order_lookup,
        rank,
        ratios,
        max_ratio,
    })
}

fn is_trimmed(trimmed: Option<&std::collections::HashSet<usize>>, idx: usize) -> bool {
    trimmed.is_some_and(|set| set.contains(&idx))
}

/// What every stratum needs and none changes.
struct Env<'a, R: MatchingRecord, G: ?Sized> {
    anchors: &'a [R],
    candidates: &'a [R],
    criteria: &'a MatchingCriteria,
    constraints: &'a G,
    plan: Plan<'a>,
    candidate_strata: Option<StrataValues<'a>>,
    random_rank: Option<Vec<usize>>,
    /// For a Mahalanobis ranking, the row of each candidate's transformed
    /// vector by position in the candidate slice, `NO_ROW` where it has none.
    candidate_rows: Option<Vec<usize>>,
}

/// Selects the pair-by-pair scan, which the tests hold the whitened scan to.
#[cfg(test)]
static USE_REFERENCE_SCAN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Marks a candidate with no transformed vector.
const NO_ROW: usize = usize::MAX;

/// The units used so far in one stratum, as the constraints read them.
#[derive(Default)]
struct UseState {
    used_unique: FxHashSet<UniqueValueId>,
    interner: RapidHashMap<String, UniqueValueId>,
}

/// Which of the next positions to the left and right of a point are still
/// available, with removal. Both directions are union-find forests with path
/// halving, so skipping a run of used positions costs near-constant time.
struct Availability {
    /// `right[i]`: the smallest available position at or after `i`, or `n`.
    right: Vec<usize>,
    /// `left[i]`: one more than the largest available position below `i`, or 0.
    left: Vec<usize>,
    remaining: usize,
    available: Vec<bool>,
}

impl Availability {
    fn new(n: usize) -> Self {
        Self {
            right: (0..=n).collect(),
            left: (0..=n).collect(),
            remaining: n,
            available: vec![true; n],
        }
    }

    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }

    fn at_or_after(&mut self, position: usize) -> usize {
        Self::root(&mut self.right, position)
    }

    fn at_or_before(&mut self, position: usize) -> Option<usize> {
        match Self::root(&mut self.left, position + 1) {
            0 => None,
            slot => Some(slot - 1),
        }
    }

    fn remove(&mut self, position: usize) {
        if self.available[position] {
            self.available[position] = false;
            self.remaining -= 1;
            self.right[position] = position + 1;
            self.left[position + 1] = position;
        }
    }
}

/// The candidates of one stratum, in the order they are searched.
struct Pool {
    /// Candidate index in the slice, by position.
    candidate: Vec<usize>,
    /// The sort key by position: the score the candidates are sorted on, or
    /// empty when they are in slice order.
    keys: Vec<f64>,
    avail: Availability,
    /// For a Mahalanobis ranking, the whitened vectors of the candidates in
    /// position order, `dimension` numbers each and NaN for a candidate with
    /// no vector. Filled on the first search.
    whitened: Option<Vec<f64>>,
}

/// One anchor's view of the stratum: what it may not be matched to.
struct Probe<'p, R: MatchingRecord, G: ?Sized> {
    env: &'p Env<'p, R, G>,
    anchor: &'p R,
    case_strata: Option<&'p [Option<&'p str>]>,
    used: &'p UsedControlsVec,
    state: &'p UseState,
}

impl<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized> Probe<'_, R, G> {
    /// Whether the anchor may be matched to this candidate, apart from the
    /// distance and the candidate being available.
    fn admits(&self, candidate_idx: usize) -> bool {
        let control = &self.env.candidates[candidate_idx];
        if self.anchor.id() == control.id() {
            return false;
        }
        let criteria = self.env.criteria;
        if criteria.unique_by_key.is_some()
            && unique_value(control, criteria).is_some_and(|value| {
                self.state
                    .interner
                    .get(value)
                    .is_some_and(|id| self.state.used_unique.contains(id))
            })
        {
            return false;
        }
        let context = ConstraintContext {
            criteria,
            used_controls: self.used,
            used_unique: &self.state.used_unique,
            unique_interner: &self.state.interner,
            control_idx: ControlIdx::new(candidate_idx),
            case_strata_values: self.case_strata,
            control_strata_values: self
                .env
                .candidate_strata
                .as_ref()
                .and_then(|values| values.row(candidate_idx)),
        };
        self.env
            .constraints
            .first_blocking_reason(self.anchor, control, &context)
            .is_none()
    }
}

fn cmp_f64(left: f64, right: f64) -> std::cmp::Ordering {
    left.partial_cmp(&right)
        .unwrap_or(std::cmp::Ordering::Equal)
}

impl Pool {
    fn build<R: MatchingRecord, G: ?Sized>(
        env: &Env<'_, R, G>,
        candidate_indices: &[usize],
    ) -> Self {
        let lookup = match &env.plan.rank {
            Rank::Scalar { lookup, .. }
            | Rank::Pairs {
                gate: Some((lookup, _)),
                ..
            } => Some(*lookup),
            Rank::Pairs { gate: None, .. } => None,
        };
        let Some(lookup) = lookup else {
            return Self {
                candidate: candidate_indices.to_vec(),
                keys: Vec::new(),
                avail: Availability::new(candidate_indices.len()),
                whitened: None,
            };
        };
        let mut entries: Vec<(f64, usize)> = candidate_indices
            .iter()
            .filter_map(|&idx| lookup.value(&env.candidates[idx]).map(|value| (value, idx)))
            .collect();
        entries.sort_unstable_by(|left, right| cmp_f64(left.0, right.0).then(left.1.cmp(&right.1)));
        Self {
            avail: Availability::new(entries.len()),
            keys: entries.iter().map(|entry| entry.0).collect(),
            candidate: entries.into_iter().map(|entry| entry.1).collect(),
            whitened: None,
        }
    }

    /// The `k` available candidates nearest to `x` that the probe admits, with
    /// their distances, nearest first and by slice position among equals.
    /// The distance is the absolute difference of the keys.
    fn nearest_by_key<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized>(
        &mut self,
        x: f64,
        caliper: Option<f64>,
        k: usize,
        probe: &Probe<'_, R, G>,
    ) -> Vec<(usize, f64)> {
        let n = self.candidate.len();
        let split = self.keys.partition_point(|&key| key < x);
        let mut left = if split == 0 {
            None
        } else {
            self.avail.at_or_before(split - 1)
        };
        let mut right = {
            let position = self.avail.at_or_after(split);
            (position < n).then_some(position)
        };
        let mut found: Vec<(f64, usize, usize)> = Vec::new();
        let mut cutoff: Option<f64> = None;
        loop {
            let left_distance = left.map(|position| x - self.keys[position]);
            let right_distance = right.map(|position| self.keys[position] - x);
            let (take_left, distance) = match (left_distance, right_distance) {
                (None, None) => break,
                (Some(distance), None) => (true, distance),
                (None, Some(distance)) => (false, distance),
                (Some(l), Some(r)) => {
                    if l <= r {
                        (true, l)
                    } else {
                        (false, r)
                    }
                }
            };
            if caliper.is_some_and(|width| distance > width) {
                break;
            }
            if cutoff.is_some_and(|limit| distance > limit) {
                break;
            }
            let position = if take_left {
                left.unwrap_or(0)
            } else {
                right.unwrap_or(0)
            };
            if take_left {
                left = if position == 0 {
                    None
                } else {
                    self.avail.at_or_before(position - 1)
                };
            } else {
                let next = self.avail.at_or_after(position + 1);
                right = (next < n).then_some(next);
            }
            let candidate_idx = self.candidate[position];
            if !probe.admits(candidate_idx) {
                continue;
            }
            found.push((distance, candidate_idx, position));
            if cutoff.is_none() && found.len() >= k {
                cutoff = Some(distance);
            }
        }
        found.sort_by(|a, b| cmp_f64(a.0, b.0).then(a.1.cmp(&b.1)));
        found.truncate(k);
        found
            .into_iter()
            .map(|(distance, _, position)| (position, distance))
            .collect()
    }

    /// The positions `low..high` whose keys lie within the gate, or all
    /// positions without one.
    fn gate_bounds(&self, gate: Option<(f64, f64)>) -> (usize, usize) {
        let n = self.candidate.len();
        match gate {
            None => (0, n),
            Some((x, width)) => {
                let keys = &self.keys;
                let mut low = keys.partition_point(|&key| key < x - width);
                let mut high = keys.partition_point(|&key| key <= x + width);
                // The bounds above can be off by a rounding of `x -+ width`, so
                // they are corrected against the test itself.
                while low > 0 && (x - keys[low - 1]).abs() <= width {
                    low -= 1;
                }
                while low < high && (x - keys[low]).abs() > width {
                    low += 1;
                }
                while high < n && (keys[high] - x).abs() <= width {
                    high += 1;
                }
                while high > low && (keys[high - 1] - x).abs() > width {
                    high -= 1;
                }
                (low, high)
            }
        }
    }

    /// The Mahalanobis form of [`Self::nearest_by_pairs`]. The distance is
    /// computed first, on whitened vectors stored one coordinate after the
    /// other, and the constraints are tested only for a candidate that would
    /// enter the `k` nearest, so the result is the same as testing every
    /// candidate. A long range is scanned in chunks on the rayon pool and the
    /// chunks' nearest sets merged under the same total order, so the result
    /// does not depend on the number of threads.
    #[expect(clippy::too_many_arguments)]
    fn nearest_by_whitened<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized>(
        &mut self,
        gate: Option<(f64, f64)>,
        mahalanobis: &IdMapMahalanobisDistance<'_>,
        anchor_row: usize,
        candidate_rows: &[usize],
        caliper: Option<f64>,
        k: usize,
        probe: &Probe<'_, R, G>,
    ) -> Vec<(usize, f64)> {
        let Some(dimension) = mahalanobis.dimension() else {
            return Vec::new();
        };
        if k == 0 {
            return Vec::new();
        }
        let (low, high) = self.gate_bounds(gate);
        let candidate = &self.candidate;
        let columns = self.whitened.get_or_insert_with(|| {
            let n = candidate.len();
            let mut columns = vec![f64::NAN; n * dimension];
            for (position, &idx) in candidate.iter().enumerate() {
                if candidate_rows[idx] != NO_ROW {
                    let row = mahalanobis.whitened_row(candidate_rows[idx], dimension);
                    for (j, value) in row.iter().enumerate() {
                        columns[j * n + position] = *value;
                    }
                }
            }
            columns
        });
        let scan = WhitenedScan {
            columns,
            anchor: mahalanobis.whitened_row(anchor_row, dimension),
            candidate,
            available: &self.avail.available,
            caliper,
            k,
            probe,
        };
        let best = if high - low <= SCAN_CHUNK {
            scan.range(low, high)
        } else {
            let starts: Vec<usize> = (low..high).step_by(SCAN_CHUNK).collect();
            let parts: Vec<Vec<(f64, usize, usize)>> = starts
                .par_iter()
                .map(|&start| scan.range(start, (start + SCAN_CHUNK).min(high)))
                .collect();
            let mut merged: Vec<(f64, usize, usize)> = parts.into_iter().flatten().collect();
            merged.sort_by(|a, b| cmp_f64(a.0, b.0).then(a.1.cmp(&b.1)));
            merged.truncate(k);
            merged
        };
        best.into_iter()
            .map(|(distance, _, position)| (position, distance))
            .collect()
    }

    /// As [`Self::nearest_by_key`], with the distance computed for each pair
    /// that passes the gate and the probe.
    fn nearest_by_pairs<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized>(
        &mut self,
        gate: Option<(f64, f64)>,
        metric: &ConfiguredMetric<'_>,
        rows: Option<(&IdMapMahalanobisDistance<'_>, usize, &[usize])>,
        caliper: Option<f64>,
        k: usize,
        probe: &Probe<'_, R, G>,
    ) -> Vec<(usize, f64)> {
        let (low, high) = self.gate_bounds(gate);
        let mut best: Vec<(f64, usize, usize)> = Vec::with_capacity(k + 1);
        let mut position = self.avail.at_or_after(low);
        while position < high {
            let candidate_idx = self.candidate[position];
            let distance = if probe.admits(candidate_idx) {
                match rows {
                    Some((mahalanobis, anchor_row, candidate_rows)) => {
                        match candidate_rows[candidate_idx] {
                            NO_ROW => None,
                            row => mahalanobis.row_distance(anchor_row, row),
                        }
                    }
                    None => DistanceMetric::<R>::distance(
                        metric,
                        probe.anchor,
                        &probe.env.candidates[candidate_idx],
                    ),
                }
                .filter(|distance| distance.is_finite())
                .filter(|distance| caliper.is_none_or(|width| *distance <= width))
            } else {
                None
            };
            if let Some(distance) = distance {
                let at = best.partition_point(|entry| {
                    cmp_f64(entry.0, distance)
                        .then(entry.1.cmp(&candidate_idx))
                        .is_lt()
                });
                if at < k {
                    best.insert(at, (distance, candidate_idx, position));
                    best.truncate(k);
                }
            }
            position = self.avail.at_or_after(position + 1);
        }
        best.into_iter()
            .map(|(distance, _, position)| (position, distance))
            .collect()
    }
}

/// Positions scanned by one task of a Mahalanobis search, and the range
/// length above which the scan is split. The result does not depend on it.
const SCAN_CHUNK: usize = 8192;

/// Positions whose distances are computed together.
const SCAN_BLOCK: usize = 256;

/// The scan of one anchor over a range of pool positions.
struct WhitenedScan<'s, R: MatchingRecord, G: ?Sized> {
    /// Whitened coordinates by coordinate, then position.
    columns: &'s [f64],
    anchor: &'s [f64],
    candidate: &'s [usize],
    available: &'s [bool],
    caliper: Option<f64>,
    k: usize,
    probe: &'s Probe<'s, R, G>,
}

impl<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized> WhitenedScan<'_, R, G> {
    /// The up to `k` nearest admitted candidates among the positions
    /// `from..to` as (distance, candidate index, position), ordered by
    /// distance and then candidate index.
    fn range(&self, from: usize, to: usize) -> Vec<(f64, usize, usize)> {
        let n = self.candidate.len();
        let k = self.k;
        let mut best: Vec<(f64, usize, usize)> = Vec::with_capacity(k + 1);
        // A squared distance above this cannot enter: the margin is far above
        // the rounding of the square and the root.
        let loose = |distance: f64| distance * distance * (1.0 + 1e-9);
        let mut limit = self.caliper.map_or(f64::INFINITY, loose);
        let mut squares = [0.0_f64; SCAN_BLOCK];
        let mut start = from;
        while start < to {
            let len = SCAN_BLOCK.min(to - start);
            let squares = &mut squares[..len];
            squares.fill(0.0);
            for (j, a) in self.anchor.iter().enumerate() {
                let column = &self.columns[j * n + start..j * n + start + len];
                for (sum, value) in squares.iter_mut().zip(column) {
                    let delta = a - value;
                    *sum += delta * delta;
                }
            }
            for (offset, &sum) in squares.iter().enumerate() {
                if sum > limit || !sum.is_finite() {
                    continue;
                }
                let position = start + offset;
                if !self.available[position] {
                    continue;
                }
                let distance = sum.sqrt();
                if self.caliper.is_some_and(|width| distance > width) {
                    continue;
                }
                let candidate_idx = self.candidate[position];
                if best.len() == k {
                    let worst = best[k - 1];
                    if !cmp_f64(distance, worst.0)
                        .then(candidate_idx.cmp(&worst.1))
                        .is_lt()
                    {
                        continue;
                    }
                }
                if !self.probe.admits(candidate_idx) {
                    continue;
                }
                let at = best.partition_point(|entry| {
                    cmp_f64(entry.0, distance)
                        .then(entry.1.cmp(&candidate_idx))
                        .is_lt()
                });
                best.insert(at, (distance, candidate_idx, position));
                best.truncate(k);
                if best.len() == k {
                    limit = limit.min(loose(best[k - 1].0));
                }
            }
            start += len;
        }
        best
    }
}

/// What a stratum produced.
#[derive(Default)]
struct StratumResult {
    /// Each matched anchor with its controls in order of assignment.
    matched: Vec<(usize, Vec<usize>)>,
    unmatched: usize,
    below_ratio: usize,
    distinct_candidates: usize,
}

/// Heap entry for the closest and farthest orders: larger pops first.
type QueueItem = (OrderedFloat<f64>, Reverse<usize>);

/// One stratum being matched.
struct Stratum<'e, 'a, R: MatchingRecord, G: ?Sized> {
    env: &'e Env<'a, R, G>,
    used: &'e mut UsedControlsVec,
    /// The anchors that can be searched, in slice order.
    usable: Vec<usize>,
    case_strata: Option<Vec<Option<&'a str>>>,
    pool: Pool,
    /// The unique-key values taken so far. With a unique key it is shared by
    /// every stratum of the match.
    state: &'e mut UseState,
    /// The controls of each usable anchor, in order of assignment.
    controls_of: Vec<Vec<usize>>,
    /// Candidates marked used in `used`, to be cleared at the end.
    assigned: Vec<usize>,
    touched: FxHashSet<usize>,
}

impl<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized> Stratum<'_, '_, R, G> {
    /// The nearest `k` available candidates of the anchor in `slot`.
    fn search(&mut self, slot: usize, k: usize) -> Vec<(usize, f64)> {
        let env = self.env;
        let anchor = &env.anchors[self.usable[slot]];
        let probe = Probe {
            env,
            anchor,
            case_strata: self.case_strata.as_deref(),
            used: self.used,
            state: &*self.state,
        };
        match &env.plan.rank {
            Rank::Scalar { lookup, caliper } => {
                let Some(x) = lookup.value(anchor) else {
                    return Vec::new();
                };
                self.pool.nearest_by_key(x, *caliper, k, &probe)
            }
            Rank::Pairs {
                gate,
                metric,
                caliper,
            } => {
                let gate = match gate {
                    None => None,
                    Some((lookup, width)) => match lookup.value(anchor) {
                        Some(x) => Some((x, *width)),
                        None => return Vec::new(),
                    },
                };
                let rows = match (metric, &env.candidate_rows) {
                    (ConfiguredMetric::Mahalanobis(metric), Some(rows)) => {
                        match metric.row(anchor.id()) {
                            Some(anchor_row) => Some((metric, anchor_row, rows.as_slice())),
                            None => return Vec::new(),
                        }
                    }
                    _ => None,
                };
                #[cfg(test)]
                let reference = USE_REFERENCE_SCAN.load(std::sync::atomic::Ordering::Relaxed);
                #[cfg(not(test))]
                let reference = false;
                match rows {
                    Some((mahalanobis, anchor_row, candidate_rows)) if !reference => {
                        self.pool.nearest_by_whitened(
                            gate,
                            mahalanobis,
                            anchor_row,
                            candidate_rows,
                            *caliper,
                            k,
                            &probe,
                        )
                    }
                    _ => self
                        .pool
                        .nearest_by_pairs(gate, metric, rows, *caliper, k, &probe),
                }
            }
        }
    }

    /// The number of controls the anchor in `slot` is to get.
    fn ratio_of(&self, slot: usize) -> usize {
        self.env
            .plan
            .ratios
            .as_ref()
            .map_or(self.env.criteria.match_ratio, |ratios| {
                ratios[self.usable[slot]]
            })
    }

    /// Whether the unique value of the candidate at `position` has been taken.
    fn unique_value_taken(&self, position: usize) -> bool {
        let criteria = self.env.criteria;
        criteria.unique_by_key.is_some()
            && unique_value(
                &self.env.candidates[self.pool.candidate[position]],
                criteria,
            )
            .is_some_and(|value| {
                self.state
                    .interner
                    .get(value)
                    .is_some_and(|id| self.state.used_unique.contains(id))
            })
    }

    /// Give the anchor in `slot` the candidate at `position`.
    fn assign(&mut self, slot: usize, position: usize) {
        let criteria = self.env.criteria;
        let candidate_idx = self.pool.candidate[position];
        self.controls_of[slot].push(candidate_idx);
        self.touched.insert(position);
        if !criteria.allow_replacement {
            self.pool.avail.remove(position);
            self.used.insert(ControlIdx::new(candidate_idx));
            self.assigned.push(candidate_idx);
        }
        if criteria.unique_by_key.is_some() {
            if let Some(value) = unique_value(&self.env.candidates[candidate_idx], criteria) {
                self.state
                    .used_unique
                    .insert(intern_unique(&mut self.state.interner, value));
            }
        }
    }

    /// Every anchor, in turn, takes all its controls.
    fn take_all_in_turn(&mut self, order: &[usize]) {
        for &slot in order {
            let ratio = self.ratio_of(slot);
            if self.env.criteria.unique_by_key.is_some() {
                // Each control taken closes its unique value to the anchor's
                // next one, so the controls are taken one at a time.
                for _ in 0..ratio {
                    let Some(&(position, _)) = self.search(slot, 1).first() else {
                        break;
                    };
                    self.assign(slot, position);
                }
            } else {
                for (position, _) in self.search(slot, ratio) {
                    self.assign(slot, position);
                }
            }
        }
    }

    /// Every anchor takes one control per round, in the same order each round.
    fn take_in_rounds(&mut self, order: &[usize]) {
        let mut active: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&slot| self.ratio_of(slot) > 0)
            .collect();
        for _ in 0..self.env.plan.max_ratio {
            if self.pool.avail.remaining == 0 || active.is_empty() {
                break;
            }
            let mut next = Vec::with_capacity(active.len());
            for &slot in &active {
                if let Some(&(position, _)) = self.search(slot, 1).first() {
                    self.assign(slot, position);
                    if self.controls_of[slot].len() < self.ratio_of(slot) {
                        next.push(slot);
                    }
                }
            }
            active = next;
        }
    }

    /// Rounds in which the pairs are taken in order of their distance.
    fn take_by_pair_distance(&mut self, order: &[usize], closest: bool) {
        let key = |distance: f64, entry: usize| -> QueueItem {
            (
                OrderedFloat(if closest { -distance } else { distance }),
                Reverse(entry),
            )
        };
        let mut active: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&slot| self.ratio_of(slot) > 0)
            .collect();
        for _ in 0..self.env.plan.max_ratio {
            if self.pool.avail.remaining == 0 || active.is_empty() {
                break;
            }
            let mut entries: Vec<(usize, usize)> = Vec::with_capacity(active.len());
            let mut queue: BinaryHeap<QueueItem> = BinaryHeap::with_capacity(active.len());
            for &slot in &active {
                if let Some(&(position, distance)) = self.search(slot, 1).first() {
                    queue.push(key(distance, entries.len()));
                    entries.push((slot, position));
                }
            }
            let mut matched_now: Vec<usize> = Vec::new();
            while let Some((_, Reverse(entry))) = queue.pop() {
                if self.pool.avail.remaining == 0 {
                    break;
                }
                let (slot, position) = entries[entry];
                if self.pool.avail.available[position] && !self.unique_value_taken(position) {
                    self.assign(slot, position);
                    matched_now.push(slot);
                } else if let Some(&(next_position, distance)) = self.search(slot, 1).first() {
                    entries[entry] = (slot, next_position);
                    queue.push(key(distance, entry));
                }
            }
            matched_now.retain(|&slot| self.controls_of[slot].len() < self.ratio_of(slot));
            matched_now.sort_unstable();
            active = matched_now;
        }
    }
}

fn run_stratum<'a, R: MatchingRecord, G: ConstraintGroup<R> + ?Sized>(
    env: &Env<'a, R, G>,
    used: &mut UsedControlsVec,
    state: &mut UseState,
    anchor_indices: &[usize],
    candidate_indices: &[usize],
) -> StratumResult {
    let criteria = env.criteria;
    let plan = &env.plan;
    let anchors: &'a [R] = env.anchors;

    // The anchors that have what the search needs: a score, where one is used.
    let usable: Vec<usize> = anchor_indices
        .iter()
        .copied()
        .filter(|&idx| anchor_key(plan, &anchors[idx]).is_some())
        .filter(|&idx| {
            !matches!(plan.order, MatchOrder::Largest | MatchOrder::Smallest)
                || plan
                    .order_lookup
                    .is_some_and(|lookup| lookup.value(&anchors[idx]).is_some())
        })
        .collect();
    let mut result = StratumResult {
        unmatched: anchor_indices.len() - usable.len(),
        ..StratumResult::default()
    };
    if usable.is_empty() {
        return result;
    }

    let mut order: Vec<usize> = (0..usable.len()).collect();
    match plan.order {
        MatchOrder::Data | MatchOrder::Closest | MatchOrder::Farthest => {}
        MatchOrder::Largest | MatchOrder::Smallest => {
            let values: Vec<f64> = usable
                .iter()
                .map(|&idx| {
                    plan.order_lookup
                        .and_then(|lookup| lookup.value(&anchors[idx]))
                        .unwrap_or(f64::NAN)
                })
                .collect();
            if plan.order == MatchOrder::Largest {
                order.sort_by(|&a, &b| cmp_f64(values[b], values[a]));
            } else {
                order.sort_by(|&a, &b| cmp_f64(values[a], values[b]));
            }
        }
        MatchOrder::Random => {
            if let Some(rank) = &env.random_rank {
                order.sort_by_key(|&slot| rank[usable[slot]]);
            }
        }
    }

    let case_strata = (!criteria.required_strata.is_empty()).then(|| {
        criteria
            .required_strata
            .iter()
            .map(|key| anchors[usable[0]].strata().get(key).map(String::as_str))
            .collect::<Vec<_>>()
    });
    let mut stratum = Stratum {
        env,
        used,
        controls_of: vec![Vec::new(); usable.len()],
        usable,
        case_strata,
        pool: Pool::build(env, candidate_indices),
        state,
        assigned: Vec::new(),
        touched: FxHashSet::default(),
    };

    let in_rounds = !criteria.allow_replacement && plan.assignment == Assignment::Rounds;
    match plan.order {
        MatchOrder::Closest | MatchOrder::Farthest if in_rounds => {
            stratum.take_by_pair_distance(&order, plan.order == MatchOrder::Closest);
        }
        _ if in_rounds => stratum.take_in_rounds(&order),
        _ => stratum.take_all_in_turn(&order),
    }

    for &candidate_idx in &stratum.assigned {
        stratum.used.remove(ControlIdx::new(candidate_idx));
    }
    result.distinct_candidates = stratum.touched.len();
    for slot in order {
        let controls = std::mem::take(&mut stratum.controls_of[slot]);
        if controls.is_empty() {
            result.unmatched += 1;
        } else {
            if controls.len() < stratum.ratio_of(slot) {
                result.below_ratio += 1;
            }
            result.matched.push((stratum.usable[slot], controls));
        }
    }
    result
}

/// The number a record is searched by: its score, or nothing where candidates
/// are searched in full.
fn anchor_key<R: MatchingRecord>(plan: &Plan<'_>, anchor: &R) -> Option<f64> {
    match &plan.rank {
        Rank::Scalar { lookup, .. }
        | Rank::Pairs {
            gate: Some((lookup, _)),
            ..
        } => lookup.value(anchor),
        Rank::Pairs { gate: None, .. } => Some(0.0),
    }
}

/// Rows grouped by their exact-match values: each key with its row indices.
type Groups<'a> = Vec<(Vec<Option<&'a str>>, Vec<usize>)>;

/// Group row indices by their values of the exact-match keys, in order of
/// first appearance, skipping the trimmed rows.
fn group_rows<'a, R: MatchingRecord>(
    rows: &'a [R],
    values: &StrataValues<'a>,
    trimmed: Option<&std::collections::HashSet<usize>>,
) -> (Groups<'a>, FxHashMap<Vec<Option<&'a str>>, usize>) {
    let mut index: FxHashMap<Vec<Option<&str>>, usize> = FxHashMap::default();
    let mut groups: Vec<(Vec<Option<&str>>, Vec<usize>)> = Vec::new();
    for idx in 0..rows.len() {
        if is_trimmed(trimmed, idx) {
            continue;
        }
        let key: Vec<Option<&str>> = values.row(idx).unwrap_or(&[]).to_vec();
        if let Some(&position) = index.get(&key) {
            groups[position].1.push(idx);
        } else {
            index.insert(key.clone(), groups.len());
            groups.push((key, vec![idx]));
        }
    }
    (groups, index)
}

/// Match standard anchors to candidates by distance.
pub(crate) fn match_nearest<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized>(
    request: &NearestRequest<'_, R, G>,
    nearest: &NearestSelection,
    mut diagnostics: MatchDiagnostics,
) -> MatchOutcome {
    let criteria = request.criteria;
    let date_default = DistanceConfig::date(criteria.typed_birth_date_caliper());
    let order_config = request.distance_config.unwrap_or(&date_default);
    let Some(plan) = resolve_plan(request, nearest, order_config) else {
        return invalid(request.anchors.len());
    };

    let required = &criteria.required_strata;
    let anchor_strata = StrataValues::of_controls(request.anchors, required);
    let candidate_strata = StrataValues::of_controls(request.candidates, required);
    let (anchor_groups, _) = group_rows(request.anchors, &anchor_strata, request.trimmed.anchors);
    let (candidate_groups, candidate_index) = group_rows(
        request.candidates,
        &candidate_strata,
        request.trimmed.candidates,
    );
    let eligible_anchors: usize = anchor_groups.iter().map(|group| group.1.len()).sum();

    let random_rank = (plan.order == MatchOrder::Random).then(|| {
        let mut ids: Vec<usize> = anchor_groups
            .iter()
            .flat_map(|group| group.1.iter().copied())
            .collect();
        ids.sort_unstable();
        Stream::seeded(nearest.seed).shuffle(&mut ids);
        let mut rank = vec![0_usize; request.anchors.len()];
        for (position, idx) in ids.into_iter().enumerate() {
            rank[idx] = position;
        }
        rank
    });

    let candidate_rows = match &plan.rank {
        Rank::Pairs {
            metric: ConfiguredMetric::Mahalanobis(metric),
            ..
        } => {
            let rows: Vec<usize> = request
                .candidates
                .iter()
                .map(|candidate| metric.row(candidate.id()).unwrap_or(NO_ROW))
                .collect();
            rows.iter().any(|row| *row != NO_ROW).then_some(rows)
        }
        _ => None,
    };

    let env = Env {
        anchors: request.anchors,
        candidates: request.candidates,
        criteria,
        constraints: request.constraints,
        plan,
        candidate_strata: (!required.is_empty()).then_some(candidate_strata),
        random_rank,
        candidate_rows,
    };

    let candidates_of = |key: &Vec<Option<&str>>| -> &[usize] {
        candidate_index
            .get(key)
            .map_or(&[], |&group| candidate_groups[group].1.as_slice())
    };
    let results: Vec<StratumResult> = if criteria.unique_by_key.is_some() {
        // A unique value is used once in the whole match, so the strata share
        // what has been taken and run in order.
        let mut used = UsedControlsVec::with_capacity(request.candidates.len());
        let mut state = UseState::default();
        anchor_groups
            .iter()
            .map(|(key, anchor_indices)| {
                run_stratum(
                    &env,
                    &mut used,
                    &mut state,
                    anchor_indices,
                    candidates_of(key),
                )
            })
            .collect()
    } else {
        anchor_groups
            .par_iter()
            .map_init(
                || UsedControlsVec::with_capacity(request.candidates.len()),
                |used, (key, anchor_indices)| {
                    let mut state = UseState::default();
                    run_stratum(&env, used, &mut state, anchor_indices, candidates_of(key))
                },
            )
            .collect()
    };

    let mut pairs = Vec::new();
    let mut matched_anchors = 0_usize;
    let mut distinct = 0_usize;
    for result in results {
        distinct += result.distinct_candidates;
        diagnostics.anchors_with_no_candidates += result.unmatched;
        diagnostics.anchors_below_required_ratio += result.below_ratio;
        for (anchor_idx, controls) in result.matched {
            matched_anchors += 1;
            let anchor_id = request.anchors[anchor_idx].id();
            for candidate_idx in controls {
                pairs.push(MatchedPair::new(
                    anchor_id,
                    request.candidates[candidate_idx].id(),
                ));
            }
        }
    }
    diagnostics.matched_anchors = matched_anchors;
    diagnostics.pairs_selected = pairs.len();
    finalize_estimand_diagnostics(&mut diagnostics, criteria.estimand, eligible_anchors, None);
    build_outcome(
        pairs,
        matched_anchors,
        eligible_anchors,
        distinct,
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_skips_removed_positions_in_both_directions() {
        let mut avail = Availability::new(6);
        for position in [1, 2, 4] {
            avail.remove(position);
        }
        assert_eq!(avail.at_or_after(1), 3);
        assert_eq!(avail.at_or_after(4), 5);
        assert_eq!(avail.at_or_after(6), 6);
        assert_eq!(avail.at_or_before(2), Some(0));
        assert_eq!(avail.at_or_before(4), Some(3));
        avail.remove(0);
        assert_eq!(avail.at_or_before(2), None);
        assert_eq!(avail.remaining, 2);
    }

    /// The whitened scan returns the pairs of the pair-by-pair scan, with
    /// exact ties in the distances, with and without replacement, at ratios 1
    /// and 3, with exact strata and a caliper, on one and on four threads.
    #[test]
    fn whitened_scan_equals_the_pair_by_pair_scan() {
        use crate::prelude::*;
        use std::sync::Mutex;
        use std::sync::atomic::Ordering;
        static SERIAL: Mutex<()> = Mutex::new(());
        let _guard = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let birth = crate::date(2000, 1, 1);
        let names = ["x1", "x2", "x3"];
        for seed in [1_u64, 2, 3] {
            let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let mut next = move || {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^ (z >> 31)
            };
            // Small integer coordinates make many pairs tie exactly. Chosen
            // so the pool is longer than one scan chunk.
            let units: Vec<BalanceRecord> = (0..14_000)
                .map(|i| {
                    let mut r = BalanceRecord::new(format!("{i}"), birth);
                    for name in names {
                        let value = (next() % 4) as f64 + if seed == 3 { 0.5 } else { 0.0 };
                        r.covariates
                            .insert(name.to_string(), CovariateValue::Numeric(value));
                        r.core.numerics.insert(name.to_string(), value);
                    }
                    r.core.strata.insert("s".into(), format!("s{}", next() % 3));
                    r
                })
                .collect();
            let (anchors, candidates) = units.split_at(2_000);
            for (replace, ratio, exact, caliper) in [
                (false, 1, false, None),
                (false, 3, true, None),
                (true, 1, false, Some(1.5)),
                (true, 3, true, Some(2.0)),
            ] {
                let spec = MatchSpec {
                    method: MatchMethod::Nearest,
                    distance: DistanceSpec::Mahalanobis {
                        covariates: CovariatesSpec {
                            names: names.iter().map(|n| (*n).to_string()).collect(),
                            missing: MissingSpec::Error,
                        },
                        covariance: CovarianceSpec::default(),
                        transform: TransformSpec::default(),
                        caliper,
                    },
                    exact: if exact { vec!["s".into()] } else { vec![] },
                    birth_window_days: 36_500,
                    replace,
                    ratio,
                    ..MatchSpec::default()
                };
                let run = |reference: bool, threads: usize| {
                    USE_REFERENCE_SCAN.store(reference, Ordering::Relaxed);
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .expect("thread pool");
                    let out = pool.install(|| {
                        run_spec(
                            &spec,
                            SpecInput::Standard {
                                anchors,
                                candidates,
                            },
                        )
                    });
                    USE_REFERENCE_SCAN.store(false, Ordering::Relaxed);
                    match out.expect("match") {
                        SpecOutput::Pairs { outcome, .. } => outcome
                            .pairs
                            .iter()
                            .map(|p| (p.case_id.clone(), p.control_id.clone()))
                            .collect::<Vec<_>>(),
                        _ => panic!("pairs expected"),
                    }
                };
                let reference = run(true, 1);
                assert!(reference.len() > 100);
                for threads in [1, 4] {
                    assert_eq!(
                        run(false, threads),
                        reference,
                        "seed {seed}, replace {replace}, ratio {ratio}, exact {exact}, {threads} threads"
                    );
                }
            }
        }
    }

    #[test]
    fn std_caliper_without_a_score_distance_is_invalid() {
        use crate::types::BaseRecord;
        use rapidhash::RapidHashMap;
        let anchors = vec![BaseRecord::new("a", crate::date(2010, 1, 1))];
        let candidates = vec![BaseRecord::new("c", crate::date(2010, 1, 1))];
        let config = DistanceConfig::mahalanobis_map(
            RapidHashMap::default(),
            vec![1.0],
            1,
            crate::types::DistanceCaliper::new(1.0),
        )
        .expect("valid");
        let outcome = crate::MatchJob::new_standard(&anchors, &candidates, 0)
            .with_nearest()
            .with_std_caliper(true)
            .with_distance_config(config)
            .run();
        assert_eq!(outcome.matched_cases, 0);
        assert!(
            outcome
                .diagnostics
                .exclusion_counts
                .keys()
                .any(|reason| reason.to_string() == "invalid_criteria:nearest_options")
        );
    }

    #[test]
    fn ratio_fallbacks_with_the_nearest_mode_are_invalid() {
        use crate::types::{BaseRecord, MatchRatio};
        let anchors = vec![BaseRecord::new("a", crate::date(2010, 1, 1))];
        let candidates = vec![BaseRecord::new("c", crate::date(2010, 1, 1))];
        for variable in [false, true] {
            let mut job = crate::MatchJob::new_standard(&anchors, &candidates, 0)
                .with_nearest()
                .with_ratio_fallback(vec![MatchRatio::new(1).expect("ratio")]);
            if variable {
                job = job.with_variable_ratio(VariableRatio::new(2.0, 1, 3).expect("valid"));
            }
            let outcome = job.run();
            assert_eq!(outcome.matched_cases, 0);
            assert!(
                outcome
                    .diagnostics
                    .exclusion_counts
                    .keys()
                    .any(|reason| reason.to_string() == "invalid_criteria:nearest_options")
            );
        }
    }

    #[test]
    fn variable_ratio_numbers_add_up_and_keep_to_the_bounds() {
        for (n, average, min, max) in [
            (60, 2.0, 1, 4),
            (60, 2.5, 2, 5),
            (139, 2.5, 2, 5),
            (7, 1.5, 1, 2),
            (50, 3.0, 2, 12),
        ] {
            let ratio = VariableRatio::new(average, min, max).expect("valid");
            let counts = ratio.counts(n).expect("counts");
            assert_eq!(counts.len(), n);
            assert!(counts.iter().all(|c| (min..=max).contains(c)));
            assert!(counts.windows(2).all(|w| w[0] <= w[1]));
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "small positive numbers"
            )]
            let total = (average * n as f64).round_ties_even() as usize;
            assert_eq!(counts.iter().sum::<usize>(), total, "{n} {average}");
        }
    }
}
