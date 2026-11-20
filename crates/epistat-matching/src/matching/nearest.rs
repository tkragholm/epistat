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
//!   the candidate slice, lowest first. `matchit()` documents the same rule.
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
use super::distance::{ConfiguredMetric, DistanceConfig, DistanceMetric, ScalarLookup};
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

    let scale = if nearest.std_caliper {
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
        match order_config.caliper() {
            None => 1.0,
            Some(_) => sample_sd(&values)?,
        }
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

    Some(Plan {
        order,
        assignment: nearest.assignment,
        order_lookup,
        rank,
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
}

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
            Rank::Scalar { lookup, .. } => Some(*lookup),
            Rank::Pairs {
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

    /// As [`Self::nearest_by_key`], with the distance computed for each pair
    /// that passes the gate and the probe.
    fn nearest_by_pairs<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized>(
        &mut self,
        gate: Option<(f64, f64)>,
        metric: &ConfiguredMetric<'_>,
        caliper: Option<f64>,
        k: usize,
        probe: &Probe<'_, R, G>,
    ) -> Vec<(usize, f64)> {
        let n = self.candidate.len();
        let (low, high) = match gate {
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
        };
        let mut best: Vec<(f64, usize, usize)> = Vec::with_capacity(k + 1);
        let mut position = self.avail.at_or_after(low);
        while position < high {
            let candidate_idx = self.candidate[position];
            if probe.admits(candidate_idx)
                && let Some(distance) = DistanceMetric::<R>::distance(
                    metric,
                    probe.anchor,
                    &probe.env.candidates[candidate_idx],
                )
                && distance.is_finite()
                && caliper.is_none_or(|width| distance <= width)
            {
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
    state: UseState,
    /// The controls of each usable anchor, in order of assignment.
    controls_of: Vec<Vec<usize>>,
    /// Candidates marked used in `used`, to be cleared at the end.
    assigned: Vec<usize>,
    touched: FxHashSet<usize>,
}

impl<'a, R: MatchingRecord, G: ConstraintGroup<R> + ?Sized> Stratum<'_, 'a, R, G> {
    /// The nearest `k` available candidates of the anchor in `slot`.
    fn search(&mut self, slot: usize, k: usize) -> Vec<(usize, f64)> {
        let env = self.env;
        let anchor = &env.anchors[self.usable[slot]];
        let probe = Probe {
            env,
            anchor,
            case_strata: self.case_strata.as_deref(),
            used: self.used,
            state: &self.state,
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
                self.pool
                    .nearest_by_pairs(gate, metric, *caliper, k, &probe)
            }
        }
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
        if criteria.unique_by_key.is_some()
            && let Some(value) = unique_value(&self.env.candidates[candidate_idx], criteria)
        {
            self.state
                .used_unique
                .insert(intern_unique(&mut self.state.interner, value));
        }
    }

    /// Every anchor, in turn, takes all its controls.
    fn take_all_in_turn(&mut self, order: &[usize]) {
        let ratio = self.env.criteria.match_ratio;
        for &slot in order {
            for (position, _) in self.search(slot, ratio) {
                self.assign(slot, position);
            }
        }
    }

    /// Every anchor takes one control per round, in the same order each round.
    fn take_in_rounds(&mut self, order: &[usize]) {
        let ratio = self.env.criteria.match_ratio;
        let mut active = order.to_vec();
        for _ in 0..ratio {
            if self.pool.avail.remaining == 0 || active.is_empty() {
                break;
            }
            let mut next = Vec::with_capacity(active.len());
            for &slot in &active {
                if let Some(&(position, _)) = self.search(slot, 1).first() {
                    self.assign(slot, position);
                    if self.controls_of[slot].len() < ratio {
                        next.push(slot);
                    }
                }
            }
            active = next;
        }
    }

    /// Rounds in which the pairs are taken in order of their distance.
    fn take_by_pair_distance(&mut self, order: &[usize], closest: bool) {
        let ratio = self.env.criteria.match_ratio;
        let key = |distance: f64, entry: usize| -> QueueItem {
            (
                OrderedFloat(if closest { -distance } else { distance }),
                Reverse(entry),
            )
        };
        let mut active = order.to_vec();
        for _ in 0..ratio {
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
                if self.pool.avail.available[position] {
                    self.assign(slot, position);
                    matched_now.push(slot);
                } else if let Some(&(next_position, distance)) = self.search(slot, 1).first() {
                    entries[entry] = (slot, next_position);
                    queue.push(key(distance, entry));
                }
            }
            matched_now.retain(|&slot| self.controls_of[slot].len() < ratio);
            matched_now.sort_unstable();
            active = matched_now;
        }
    }
}

fn run_stratum<'a, R: MatchingRecord, G: ConstraintGroup<R> + ?Sized>(
    env: &Env<'a, R, G>,
    used: &mut UsedControlsVec,
    anchor_indices: &[usize],
    candidate_indices: &[usize],
) -> StratumResult {
    let criteria = env.criteria;
    let ratio = criteria.match_ratio;
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
        state: UseState::default(),
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
            if controls.len() < ratio {
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

/// Group row indices by their values of the exact-match keys, in order of
/// first appearance, skipping the trimmed rows.
fn group_rows<'a, R: MatchingRecord>(
    rows: &'a [R],
    values: &StrataValues<'a>,
    trimmed: Option<&std::collections::HashSet<usize>>,
) -> (
    Vec<(Vec<Option<&'a str>>, Vec<usize>)>,
    FxHashMap<Vec<Option<&'a str>>, usize>,
) {
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

    let env = Env {
        anchors: request.anchors,
        candidates: request.candidates,
        criteria,
        constraints: request.constraints,
        plan,
        candidate_strata: (!required.is_empty()).then_some(candidate_strata),
        random_rank,
    };

    let results: Vec<StratumResult> = anchor_groups
        .par_iter()
        .map_init(
            || UsedControlsVec::with_capacity(request.candidates.len()),
            |used, (key, anchor_indices)| {
                let candidate_indices: &[usize] = candidate_index
                    .get(key)
                    .map_or(&[], |&group| candidate_groups[group].1.as_slice());
                run_stratum(&env, used, anchor_indices, candidate_indices)
            },
        )
        .collect();

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
