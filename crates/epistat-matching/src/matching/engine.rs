use super::constraints::{
    CaliperConstraint, Constraint, ConstraintContext, ConstraintGroup, ExactMatchConstraint,
    NoSelfMatchConstraint, ReplacementConstraint, UniqueKeyConstraint, UsedControlsVec,
    build_strata_values, unique_value,
};
use super::distance::{DateDistance, DistanceChannel, DistanceConfig, DistanceMetric};
use super::index::{CandidateIndex, StrataValues};
use super::ratio::RatioPolicy;
use super::records::MatchingRecord;
use super::selection::SelectionStrategy;
use crate::types::{
    CommonSupport, CommonSupportFailureReason, ConstraintReason, ControlIdx, DistanceCaliper,
    DistanceCaliperReason, Estimand, EstimandDriftReason, ExclusionReason, InvalidCriteriaReason,
    MatchDiagnostics, MatchOutcome, MatchRatio, MatchedPair, MatchingCriteria, UniqueValueId,
};
use rapidhash::RapidHashMap;
use std::collections::HashSet;
use std::sync::OnceLock;

fn initial_diagnostics(
    total_anchors: usize,
    requested_estimand: Estimand,
    common_support_policy: Option<CommonSupport>,
) -> MatchDiagnostics {
    MatchDiagnostics {
        total_anchors_evaluated: total_anchors,
        requested_estimand,
        realized_estimand: requested_estimand,
        common_support_policy,
        ..MatchDiagnostics::default()
    }
}

fn push_drift_reason(diagnostics: &mut MatchDiagnostics, reason: EstimandDriftReason) {
    if diagnostics.estimand_drift_reasons.contains(&reason) {
        return;
    }
    diagnostics.estimand_drift_reasons.push(reason);
}

pub fn finalize_estimand_diagnostics(
    diagnostics: &mut MatchDiagnostics,
    requested_estimand: Estimand,
    eligible_anchors: usize,
    distance_caliper_reason: Option<&str>,
) {
    diagnostics.requested_estimand = requested_estimand;

    if diagnostics.common_support_trimmed_anchors > 0
        || diagnostics.common_support_trimmed_candidates > 0
    {
        push_drift_reason(diagnostics, EstimandDriftReason::CommonSupportTrimming);
    }
    if eligible_anchors > diagnostics.matched_anchors {
        push_drift_reason(diagnostics, EstimandDriftReason::UnmatchedAnchors);
    }
    if diagnostics.anchors_below_required_ratio > 0 {
        push_drift_reason(diagnostics, EstimandDriftReason::RatioShortfall);
    }
    if let Some(reason) = distance_caliper_reason {
        let caliper_exclusions = diagnostics
            .exclusion_counts
            .get(&ExclusionReason::DistanceCaliper(
                DistanceCaliperReason::from_reason_str(reason),
            ))
            .copied()
            .unwrap_or(0);
        if caliper_exclusions > 0 {
            push_drift_reason(diagnostics, EstimandDriftReason::DistanceCaliperExclusion);
            if diagnostics.common_support_trimmed_anchors > 0
                || diagnostics.common_support_trimmed_candidates > 0
            {
                push_drift_reason(
                    diagnostics,
                    EstimandDriftReason::CommonSupportCaliperInteraction,
                );
            }
        }
    }

    diagnostics.realized_estimand = if diagnostics.estimand_drift_reasons.is_empty() {
        requested_estimand
    } else {
        Estimand::Atm
    };
}

/// Reusable matching engine with precomputed lookups and pluggable selection.
pub struct MatchEngine<'a, R: MatchingRecord, S: SelectionStrategy<R>> {
    pub(crate) controls: &'a [R],
    pub(crate) criteria: &'a MatchingCriteria,
    pub(crate) precomputed: MatchingPrecomputed<'a>,
    pub(crate) candidate_index: CandidateIndex<'a>,
    pub(crate) selector: S,
    id_ranks: OnceLock<Option<IdRanks>>,
}

/// Unified request for standard anchor-to-candidate matching.
#[derive(Clone, Copy)]
pub struct StandardMatchRequest<'a, S, G: ?Sized> {
    pub criteria: &'a MatchingCriteria,
    pub strategy: S,
    pub constraints: &'a G,
    pub ratio_fallback: &'a [MatchRatio],
    pub distance_config: Option<&'a DistanceConfig>,
}

impl<'a, S, G: ?Sized> StandardMatchRequest<'a, S, G> {
    #[must_use]
    pub const fn new(criteria: &'a MatchingCriteria, strategy: S, constraints: &'a G) -> Self {
        Self {
            criteria,
            strategy,
            constraints,
            ratio_fallback: &[],
            distance_config: None,
        }
    }

    #[must_use]
    pub const fn with_ratio_fallback(mut self, ratio_fallback: &'a [MatchRatio]) -> Self {
        self.ratio_fallback = ratio_fallback;
        self
    }

    crate::impl_with_distance_config!();
}

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Internal candidate-pool request used by role-transition and engine helpers.
#[derive(Clone, Copy)]
pub struct CandidatePoolRequest<'a, R: MatchingRecord, D: DistanceMetric<R>> {
    pub case: &'a R,
    pub used_controls: &'a UsedControlsVec,
    pub used_unique: &'a FxHashSet<UniqueValueId>,
    pub unique_interner: &'a RapidHashMap<String, UniqueValueId>,
    pub distance_channel: DistanceChannel<'a, R, D>,
    pub use_birth_date_index: bool,
}

#[derive(Clone, Copy)]
pub struct CandidatePoolConfig<'a, D> {
    pub used_controls: &'a UsedControlsVec,
    pub used_unique: &'a FxHashSet<UniqueValueId>,
    pub unique_interner: &'a RapidHashMap<String, UniqueValueId>,
    pub distance_metric: &'a D,
    pub distance_caliper: Option<DistanceCaliper>,
    pub use_birth_date_index: bool,
    pub caliper_reason: &'static str,
}

#[derive(Clone, Copy)]
pub struct DistanceRunConfig<'a, D> {
    pub distance_metric: &'a D,
    pub distance_caliper: Option<DistanceCaliper>,
    pub use_birth_date_index: bool,
    pub caliper_reason: &'static str,
}

#[derive(Clone, Copy, Default)]
pub struct TrimmedIndexConfig<'a> {
    pub anchors: Option<&'a HashSet<usize>>,
    pub candidates: Option<&'a HashSet<usize>>,
}

fn intersect_candidate_indices(candidate_indices: &mut Vec<usize>, indexed_indices: Vec<usize>) {
    let pre_filter_count = candidate_indices.len();
    if indexed_indices.len() < pre_filter_count {
        let set: HashSet<usize> = candidate_indices.drain(..).collect();
        candidate_indices.extend(indexed_indices.into_iter().filter(|idx| set.contains(idx)));
    } else {
        let set: HashSet<usize> = indexed_indices.into_iter().collect();
        candidate_indices.retain(|idx| set.contains(idx));
    }
}

fn first_blocking_constraint_reason<R: MatchingRecord, D: DistanceMetric<R> + Sync>(
    case: &R,
    control: &R,
    builtin_constraints: &[&dyn Constraint<R>],
    caliper_constraint: Option<&CaliperConstraint<'_, R, D>>,
    extra_constraints: &(impl ConstraintGroup<R> + ?Sized),
    context: &ConstraintContext<'_>,
) -> Option<&'static str> {
    for constraint in builtin_constraints.iter().copied().chain(
        caliper_constraint
            .as_ref()
            .map(|constraint| *constraint as &dyn Constraint<R>),
    ) {
        if !constraint.allows(case, control, context) {
            return Some(constraint.reason());
        }
    }
    extra_constraints.first_blocking_reason(case, control, context)
}

impl<'a, R: MatchingRecord + Sync, S: SelectionStrategy<R> + Clone + Send + Sync>
    MatchEngine<'a, R, S>
{
    /// Construct a reusable engine for a fixed control pool and criteria.
    #[must_use]
    pub fn new(controls: &'a [R], criteria: &'a MatchingCriteria, selector: S) -> Self {
        let strata_values = StrataValues::of_controls(controls, &criteria.required_strata);
        let candidate_index =
            CandidateIndex::from_values(controls, &criteria.required_strata, &strata_values);
        let precomputed = MatchingPrecomputed {
            criteria,
            control_strata_values: (!criteria.required_strata.is_empty()).then_some(strata_values),
        };

        Self {
            controls,
            criteria,
            precomputed,
            candidate_index,
            selector,
            id_ranks: OnceLock::new(),
        }
    }

    #[must_use]
    pub const fn controls(&self) -> &'a [R] {
        self.controls
    }

    pub(crate) fn candidate_pool_for_request_and_constraint_group<
        F,
        D: DistanceMetric<R> + Sync,
        G: ConstraintGroup<R> + ?Sized,
    >(
        &self,
        request: &CandidatePoolRequest<'_, R, D>,
        extra_constraints: &G,
        additional_filter: F,
        diagnostics: &mut MatchDiagnostics,
    ) -> Vec<usize>
    where
        F: Fn(usize, &R) -> bool + Sync,
    {
        self.candidate_pool_inner(
            request.case,
            &CandidatePoolConfig {
                used_controls: request.used_controls,
                used_unique: request.used_unique,
                unique_interner: request.unique_interner,
                distance_metric: request.distance_channel.metric(),
                distance_caliper: request.distance_channel.typed_caliper(),
                use_birth_date_index: request.use_birth_date_index,
                caliper_reason: request.distance_channel.reason(),
            },
            extra_constraints,
            additional_filter,
            diagnostics,
            PoolScan::Wide,
        )
    }

    pub(crate) fn candidate_pool_with_distance_and_constraint_group<
        F,
        D: DistanceMetric<R> + Sync,
        G: ConstraintGroup<R> + ?Sized,
    >(
        &self,
        case: &R,
        pool_config: &CandidatePoolConfig<'_, D>,
        extra_constraints: &G,
        additional_filter: F,
        diagnostics: &mut MatchDiagnostics,
    ) -> Vec<usize>
    where
        F: Fn(usize, &R) -> bool + Sync,
    {
        self.candidate_pool_inner(
            case,
            pool_config,
            extra_constraints,
            additional_filter,
            diagnostics,
            PoolScan::Narrow,
        )
    }

    /// The controls that may be matched to `case`, in the order of their ids.
    ///
    /// A `Wide` scan splits a long candidate list over the rayon pool and sorts
    /// by a precomputed id rank. Both produce the pool a `Narrow` scan does,
    /// element for element.
    fn candidate_pool_inner<F, D: DistanceMetric<R> + Sync, G: ConstraintGroup<R> + ?Sized>(
        &self,
        case: &R,
        pool_config: &CandidatePoolConfig<'_, D>,
        extra_constraints: &G,
        additional_filter: F,
        diagnostics: &mut MatchDiagnostics,
        scan: PoolScan,
    ) -> Vec<usize>
    where
        F: Fn(usize, &R) -> bool + Sync,
    {
        let case_strata_values = (!self.criteria.required_strata.is_empty())
            .then(|| build_strata_values(case, &self.criteria.required_strata));

        let mut candidate_indices = if pool_config.use_birth_date_index {
            let window_days = self
                .criteria
                .typed_birth_date_window()
                .map_or(0_i64, |days| i64::from(days.get()));
            self.candidate_index.candidate_indices(
                case.birth_date(),
                window_days,
                case_strata_values.as_deref(),
            )
        } else {
            (0..self.controls.len()).collect()
        };

        let indexed_indices = pool_config.distance_caliper.and_then(|max_distance| {
            pool_config
                .distance_metric
                .candidate_indices(case, max_distance.get())
        });
        if let Some(indexed_indices) = indexed_indices {
            let pre_filter_count = candidate_indices.len();
            intersect_candidate_indices(&mut candidate_indices, indexed_indices);
            let excluded_by_index = pre_filter_count.saturating_sub(candidate_indices.len());
            if excluded_by_index > 0 {
                let reason = pool_config.caliper_reason;
                *diagnostics
                    .exclusion_counts
                    .entry(ExclusionReason::DistanceCaliper(
                        DistanceCaliperReason::from_reason_str(reason),
                    ))
                    .or_insert(0) += excluded_by_index;
            }
        }
        let caliper_constraint = pool_config.distance_caliper.map(|max_distance| {
            CaliperConstraint::new(
                pool_config.distance_metric,
                max_distance.get(),
                pool_config.caliper_reason,
            )
        });
        // Every candidate the strata index returns has the case's value for each
        // required stratum, so the exact-match constraint cannot block it.
        let strata_indexed = pool_config.use_birth_date_index && case_strata_values.is_some();

        let self_match = match (scan, self.id_ranks()) {
            (PoolScan::Wide, Some(ranks)) => ranks
                .position_of(self.controls, case.id())
                .map_or(SelfMatch::Nobody, SelfMatch::At),
            _ => SelfMatch::ById,
        };
        let scan_chunk = |chunk: &[usize]| {
            self.scan_candidates(
                chunk,
                &ScanContext {
                    case,
                    pool_config,
                    extra_constraints,
                    case_strata_values: case_strata_values.as_deref(),
                    caliper_constraint: caliper_constraint.as_ref(),
                    strata_indexed,
                    self_match,
                },
                &additional_filter,
            )
        };
        let tally = match scan {
            PoolScan::Narrow => scan_chunk(&candidate_indices),
            PoolScan::Wide if candidate_indices.len() < 2 * SCAN_CHUNK => {
                let mut tally = scan_chunk(&candidate_indices);
                if let Some(ranks) = self.id_ranks() {
                    ranks.sort(&mut tally.eligible);
                }
                tally
            }
            PoolScan::Wide => {
                let ranks = self.id_ranks();
                let tallies: Vec<ScanTally> = candidate_indices
                    .par_chunks(SCAN_CHUNK)
                    .map(|chunk| {
                        let mut tally = scan_chunk(chunk);
                        if let Some(ranks) = ranks {
                            ranks.sort(&mut tally.eligible);
                        }
                        tally
                    })
                    .collect();
                ScanTally::join_all(tallies, ranks)
            }
        };
        let ScanTally {
            mut eligible,
            filtered_out,
            constraint_exclusions,
        } = tally;

        if filtered_out > 0 {
            *diagnostics
                .exclusion_counts
                .entry(ExclusionReason::AdditionalFilter)
                .or_insert(0) += filtered_out;
        }
        for (reason, count) in constraint_exclusions {
            *diagnostics
                .exclusion_counts
                .entry(ExclusionReason::Constraint(
                    ConstraintReason::from_reason_str(reason),
                ))
                .or_insert(0) += count;
        }

        // The pool is ordered by control id. The seeded selector draws a position
        // into it, so the order must not depend on the order of the control slice,
        // which callers may build through hash-backed collections. The sort also
        // gives distance-based selection a stable tie-break and does not change
        // which candidates are eligible.
        match (scan, self.id_ranks()) {
            // A wide scan has ordered the pool already.
            (PoolScan::Wide, Some(_)) => {}
            (PoolScan::Narrow, _) | (PoolScan::Wide, None) => {
                sort_by_id(self.controls, &mut eligible);
            }
        }
        eligible
    }

    /// Apply the additional filter and every constraint to `chunk`, in order.
    ///
    /// Exclusions are tallied by reason string here and entered into the
    /// diagnostics once per pool. `ConstraintReason::from_reason_str`
    /// allocates for every string it has no variant for, which is most of
    /// the built-in ones, and this loop runs once per candidate.
    fn scan_candidates<F, D: DistanceMetric<R> + Sync, G: ConstraintGroup<R> + ?Sized>(
        &self,
        chunk: &[usize],
        context: &ScanContext<'_, R, D, G>,
        additional_filter: &F,
    ) -> ScanTally
    where
        F: Fn(usize, &R) -> bool,
    {
        let ScanContext {
            case,
            pool_config,
            extra_constraints,
            case_strata_values,
            caliper_constraint,
            strata_indexed,
            self_match,
        } = *context;

        let replacement_constraint = ReplacementConstraint;
        let no_self_match_constraint = NoSelfMatchConstraint;
        let self_match_at = SelfMatchAt(match self_match {
            SelfMatch::At(idx) => Some(idx),
            SelfMatch::ById | SelfMatch::Nobody => None,
        });
        let no_self_match: &dyn Constraint<R> = match self_match {
            SelfMatch::ById => &no_self_match_constraint,
            SelfMatch::At(_) | SelfMatch::Nobody => &self_match_at,
        };
        let strata_constraint = ExactMatchConstraint;
        let unique_constraint = UniqueKeyConstraint;
        let all_builtin: [&dyn Constraint<R>; 4] = [
            &replacement_constraint,
            no_self_match,
            &strata_constraint,
            &unique_constraint,
        ];
        let indexed_builtin: [&dyn Constraint<R>; 3] =
            [&replacement_constraint, no_self_match, &unique_constraint];
        let builtin_constraints: &[&dyn Constraint<R>] = if strata_indexed {
            &indexed_builtin
        } else {
            &all_builtin
        };

        let mut tally = ScanTally::default();
        for &idx in chunk {
            let control = &self.controls[idx];

            if !additional_filter(idx, control) {
                tally.filtered_out += 1;
                continue;
            }

            let control_strata_values = self
                .precomputed
                .control_strata_values
                .as_ref()
                .and_then(|values| values.row(idx));
            let constraint_context = ConstraintContext {
                criteria: self.precomputed.criteria,
                used_controls: pool_config.used_controls,
                used_unique: pool_config.used_unique,
                unique_interner: pool_config.unique_interner,
                control_idx: ControlIdx::new(idx),
                case_strata_values,
                control_strata_values,
            };

            if let Some(reason) = first_blocking_constraint_reason(
                case,
                control,
                builtin_constraints,
                caliper_constraint,
                extra_constraints,
                &constraint_context,
            ) {
                tally.exclude(reason, 1);
                continue;
            }

            tally.eligible.push(idx);
        }
        tally
    }

    /// Id ranks of the controls, or `None` when two controls share an id.
    fn id_ranks(&self) -> Option<&IdRanks> {
        self.id_ranks
            .get_or_init(|| IdRanks::of_controls(self.controls))
            .as_ref()
    }

    pub fn select_control(
        &mut self,
        case: &R,
        candidate_indices: &mut Vec<usize>,
    ) -> Option<ControlIdx> {
        let selected_pos = self
            .selector
            .select_position(case, self.controls, candidate_indices)?;
        Some(ControlIdx::new(candidate_indices.swap_remove(selected_pos)))
    }

    pub(crate) fn run_anchor_matching_with_distance_internal<
        D: DistanceMetric<R> + Sync,
        G: ConstraintGroup<R> + ?Sized,
    >(
        &self,
        anchors: &[R],
        distance_config: &DistanceRunConfig<'_, D>,
        extra_constraints: &G,
        ratio_policy: &RatioPolicy,
        trimmed_indices: TrimmedIndexConfig<'_>,
        mut diagnostics: MatchDiagnostics,
    ) -> MatchOutcome {
        struct GroupResult {
            pairs: Vec<MatchedPair>,
            matched_cases: usize,
            eligible_anchors: usize,
            used_controls_count: usize,
            diagnostics: MatchDiagnostics,
        }

        if self.criteria.allow_replacement && self.criteria.unique_by_key.is_none() {
            return self.run_anchor_matching_parallel(
                anchors,
                distance_config,
                extra_constraints,
                ratio_policy,
                trimmed_indices,
                diagnostics,
            );
        }

        // Anchors are grouped by strata key. Controls in different groups are never
        // interchangeable, since matching is exact on the strata, so the groups run
        // independently in parallel and `used_controls` is local to a group. A unique
        // value is used once in the whole match, so with a unique key the groups share
        // `used_unique` and run in order.
        let n_controls = self.controls.len();
        let strata_groups =
            group_anchors_by_strata(anchors, &self.criteria.required_strata, &trimmed_indices);

        let run_group = |group_index: usize,
                         anchor_indices: Vec<usize>,
                         used_unique: &mut FxHashSet<UniqueValueId>,
                         unique_interner: &mut RapidHashMap<String, UniqueValueId>|
         -> GroupResult {
            let mut used_controls = UsedControlsVec::with_capacity(n_controls);
            let mut distinct_used = FxHashSet::<usize>::default();
            let mut pairs = Vec::new();
            let mut matched_cases = 0usize;
            let mut eligible_anchors = 0usize;
            let mut local_diagnostics = MatchDiagnostics::default();
            let mut local_selector = self.selector.for_unit(group_index);

            if self.selector.assignment() == super::selection::Assignment::Rounds {
                // Every anchor takes a first control before any takes a
                // second. The ratio is the criteria's own: a fallback tier
                // is a property of one anchor's pool, and the pool changes
                // between rounds.
                let ratio = self.criteria.match_ratio;
                let mut taken: Vec<Vec<usize>> = vec![Vec::new(); anchor_indices.len()];
                let mut alive = vec![true; anchor_indices.len()];
                eligible_anchors += anchor_indices.len();
                for _ in 0..ratio {
                    for (slot, &anchor_idx) in anchor_indices.iter().enumerate() {
                        if !alive[slot] {
                            continue;
                        }
                        let case = &anchors[anchor_idx];
                        let mut candidates = self
                            .candidate_pool_with_distance_and_constraint_group(
                                case,
                                &CandidatePoolConfig {
                                    used_controls: &used_controls,
                                    used_unique,
                                    unique_interner,
                                    distance_metric: distance_config.distance_metric,
                                    distance_caliper: distance_config.distance_caliper,
                                    use_birth_date_index: distance_config.use_birth_date_index,
                                    caliper_reason: distance_config.caliper_reason,
                                },
                                extra_constraints,
                                |idx, _control| {
                                    !trimmed_indices
                                        .candidates
                                        .is_some_and(|trimmed| trimmed.contains(&idx))
                                },
                                &mut local_diagnostics,
                            );
                        let Some(pos) =
                            local_selector.select_position(case, self.controls, &candidates)
                        else {
                            alive[slot] = false;
                            continue;
                        };
                        let raw_idx = candidates.swap_remove(pos);
                        let control = &self.controls[raw_idx];
                        used_controls.insert(ControlIdx::new(raw_idx));
                        distinct_used.insert(raw_idx);
                        if let Some(value) = unique_value(control, self.criteria) {
                            used_unique.insert(intern_unique(&mut *unique_interner, value));
                        }
                        taken[slot].push(raw_idx);
                    }
                }
                for (slot, controls) in taken.into_iter().enumerate() {
                    let case = &anchors[anchor_indices[slot]];
                    if controls.is_empty() {
                        local_diagnostics.anchors_with_no_candidates += 1;
                        continue;
                    }
                    if controls.len() < ratio {
                        local_diagnostics.anchors_below_required_ratio += 1;
                    }
                    matched_cases += 1;
                    for raw_idx in controls {
                        pairs.push(MatchedPair::new(case.id(), self.controls[raw_idx].id()));
                    }
                }
                return GroupResult {
                    pairs,
                    matched_cases,
                    eligible_anchors,
                    used_controls_count: distinct_used.len(),
                    diagnostics: local_diagnostics,
                };
            }

            for anchor_idx in anchor_indices {
                let case = &anchors[anchor_idx];
                eligible_anchors += 1;

                let mut candidates = self.candidate_pool_with_distance_and_constraint_group(
                    case,
                    &CandidatePoolConfig {
                        used_controls: &used_controls,
                        used_unique,
                        unique_interner,
                        distance_metric: distance_config.distance_metric,
                        distance_caliper: distance_config.distance_caliper,
                        use_birth_date_index: distance_config.use_birth_date_index,
                        caliper_reason: distance_config.caliper_reason,
                    },
                    extra_constraints,
                    |idx, _control| {
                        !trimmed_indices
                            .candidates
                            .is_some_and(|trimmed| trimmed.contains(&idx))
                    },
                    &mut local_diagnostics,
                );
                if candidates.is_empty() {
                    local_diagnostics.anchors_with_no_candidates += 1;
                    continue;
                }

                let Some(target_ratio) = ratio_policy.target_ratio(candidates.len()) else {
                    local_diagnostics.anchors_below_required_ratio += 1;
                    continue;
                };
                if ratio_policy.is_shortfall(candidates.len()) {
                    local_diagnostics.anchors_below_required_ratio += 1;
                }
                if ratio_policy.is_lower_tier(candidates.len()) {
                    local_diagnostics.anchors_on_lower_tier += 1;
                }

                let mut local_matches = 0usize;
                for _ in 0..target_ratio {
                    let Some(pos) =
                        local_selector.select_position(case, self.controls, &candidates)
                    else {
                        break;
                    };
                    let raw_idx = candidates.swap_remove(pos);
                    let selected = ControlIdx::new(raw_idx);
                    let control = &self.controls[raw_idx];
                    pairs.push(MatchedPair::new(case.id(), control.id()));
                    local_matches += 1;
                    distinct_used.insert(raw_idx);

                    if !self.criteria.allow_replacement {
                        used_controls.insert(selected);
                    }
                    if let Some(value) = unique_value(control, self.criteria) {
                        used_unique.insert(intern_unique(&mut *unique_interner, value));
                        // The anchor's other candidates of this value are closed.
                        if self.criteria.unique_by_key.is_some() {
                            candidates.retain(|&other| {
                                unique_value(&self.controls[other], self.criteria) != Some(value)
                            });
                        }
                    }
                }
                if local_matches > 0 {
                    matched_cases += 1;
                }
            }

            GroupResult {
                pairs,
                matched_cases,
                eligible_anchors,
                used_controls_count: distinct_used.len(),
                diagnostics: local_diagnostics,
            }
        };
        let group_results: Vec<GroupResult> = if self.criteria.unique_by_key.is_some() {
            let mut used_unique = FxHashSet::<UniqueValueId>::default();
            let mut unique_interner = RapidHashMap::<String, UniqueValueId>::default();
            strata_groups
                .into_iter()
                .enumerate()
                .map(|(group_index, anchor_indices)| {
                    run_group(
                        group_index,
                        anchor_indices,
                        &mut used_unique,
                        &mut unique_interner,
                    )
                })
                .collect()
        } else {
            strata_groups
                .into_par_iter()
                .enumerate()
                .map(|(group_index, anchor_indices)| {
                    let mut used_unique = FxHashSet::<UniqueValueId>::default();
                    let mut unique_interner = RapidHashMap::<String, UniqueValueId>::default();
                    run_group(
                        group_index,
                        anchor_indices,
                        &mut used_unique,
                        &mut unique_interner,
                    )
                })
                .collect()
        };

        let mut pairs = Vec::new();
        let mut matched_cases = 0usize;
        let mut eligible_anchors = 0usize;
        let mut used_controls_count = 0usize;
        for gr in group_results {
            pairs.extend(gr.pairs);
            matched_cases += gr.matched_cases;
            eligible_anchors += gr.eligible_anchors;
            used_controls_count += gr.used_controls_count;
            diagnostics.merge(gr.diagnostics);
        }

        diagnostics.matched_anchors = matched_cases;
        diagnostics.pairs_selected = pairs.len();
        finalize_estimand_diagnostics(
            &mut diagnostics,
            self.criteria.estimand,
            eligible_anchors,
            distance_config
                .distance_caliper
                .is_some()
                .then_some(distance_config.caliper_reason),
        );
        build_outcome(
            pairs,
            matched_cases,
            eligible_anchors,
            used_controls_count,
            diagnostics,
        )
    }

    fn run_anchor_matching_parallel<D: DistanceMetric<R> + Sync, G: ConstraintGroup<R> + ?Sized>(
        &self,
        anchors: &[R],
        distance_config: &DistanceRunConfig<'_, D>,
        extra_constraints: &G,
        ratio_policy: &RatioPolicy,
        trimmed_indices: TrimmedIndexConfig<'_>,
        mut diagnostics: MatchDiagnostics,
    ) -> MatchOutcome
    where
        S: Clone + Send + Sync,
        R: Sync,
    {
        let empty_controls = UsedControlsVec::with_capacity(0);
        let empty_unique = FxHashSet::<UniqueValueId>::default();
        let empty_interner = RapidHashMap::<String, UniqueValueId>::default();

        let results: Vec<(Vec<MatchedPair>, Vec<usize>, MatchDiagnostics)> = anchors
            .par_iter()
            .enumerate()
            .filter_map(|(anchor_idx, case)| {
                if trimmed_indices
                    .anchors
                    .is_some_and(|trimmed| trimmed.contains(&anchor_idx))
                {
                    return None;
                }

                let mut local_diagnostics = MatchDiagnostics::default();
                let mut local_selector = self.selector.for_unit(anchor_idx);

                let mut candidates = self.candidate_pool_with_distance_and_constraint_group(
                    case,
                    &CandidatePoolConfig {
                        used_controls: &empty_controls,
                        used_unique: &empty_unique,
                        unique_interner: &empty_interner,
                        distance_metric: distance_config.distance_metric,
                        distance_caliper: distance_config.distance_caliper,
                        use_birth_date_index: distance_config.use_birth_date_index,
                        caliper_reason: distance_config.caliper_reason,
                    },
                    extra_constraints,
                    |idx, _control| {
                        !trimmed_indices
                            .candidates
                            .is_some_and(|trimmed| trimmed.contains(&idx))
                    },
                    &mut local_diagnostics,
                );

                if candidates.is_empty() {
                    local_diagnostics.anchors_with_no_candidates += 1;
                    return Some((Vec::new(), Vec::new(), local_diagnostics));
                }

                let Some(target_ratio) = ratio_policy.target_ratio(candidates.len()) else {
                    local_diagnostics.anchors_below_required_ratio += 1;
                    return Some((Vec::new(), Vec::new(), local_diagnostics));
                };
                if ratio_policy.is_shortfall(candidates.len()) {
                    local_diagnostics.anchors_below_required_ratio += 1;
                }
                if ratio_policy.is_lower_tier(candidates.len()) {
                    local_diagnostics.anchors_on_lower_tier += 1;
                }

                let mut local_pairs = Vec::new();
                let mut local_controls = Vec::new();
                for _ in 0..target_ratio {
                    let Some(selected_idx) =
                        local_selector.select_position(case, self.controls, &candidates)
                    else {
                        break;
                    };
                    let control_idx = candidates[selected_idx];
                    let control = &self.controls[control_idx];
                    local_pairs.push(MatchedPair::new(case.id(), control.id()));
                    local_controls.push(control_idx);
                    candidates.remove(selected_idx);
                }

                if !local_pairs.is_empty() {
                    local_diagnostics.matched_anchors = 1;
                }
                local_diagnostics.pairs_selected = local_pairs.len();

                Some((local_pairs, local_controls, local_diagnostics))
            })
            .collect();

        // Every anchor that was not trimmed has an entry, matched or not.
        let eligible_anchors = results.len();
        let mut all_pairs = Vec::new();
        let mut distinct_controls = FxHashSet::<usize>::default();
        for (mut pairs, controls, diag) in results {
            all_pairs.append(&mut pairs);
            distinct_controls.extend(controls);
            diagnostics.merge(diag);
        }

        finalize_estimand_diagnostics(
            &mut diagnostics,
            self.criteria.estimand,
            eligible_anchors,
            distance_config
                .distance_caliper
                .is_some()
                .then_some(distance_config.caliper_reason),
        );

        build_outcome(
            all_pairs,
            diagnostics.matched_anchors,
            eligible_anchors,
            distinct_controls.len(),
            diagnostics,
        )
    }
}

/// Units left out of a standard match, given by id.
///
/// This is `matchit()`'s `discard` as a list: the caller decides which units
/// go, for instance from a first propensity fit, and matches the others on a
/// distance fitted again. Ids that belong to no anchor or candidate are
/// ignored.
#[derive(Debug, Clone, Default)]
pub struct DiscardedUnits {
    /// Ids of the anchors to leave out.
    pub anchors: HashSet<String>,
    /// Ids of the candidates to leave out.
    pub candidates: HashSet<String>,
}

/// Run anchor/candidate matching with an explicit control selection strategy.
#[must_use]
pub fn match_standard<
    R: MatchingRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
>(
    anchors: &[R],
    candidates: &[R],
    request: StandardMatchRequest<'_, S, G>,
) -> MatchOutcome {
    match_standard_with_discard(anchors, candidates, request, None)
}

/// As [`match_standard`], leaving out the units in `discarded`.
///
/// The listed units take the place of common-support trimming: the criteria's
/// common-support policy is not applied again, and the distance need not be a
/// propensity score. They are counted as trimmed in the diagnostics.
#[must_use]
pub fn match_standard_with_discard<
    R: MatchingRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
>(
    anchors: &[R],
    candidates: &[R],
    request: StandardMatchRequest<'_, S, G>,
    discarded: Option<&DiscardedUnits>,
) -> MatchOutcome {
    let validated = match request.criteria.validate() {
        Ok(validated) => validated,
        Err(err) => {
            return invalid_criteria_outcome(anchors.len(), InvalidCriteriaReason::from(err));
        }
    };
    let criteria = validated.as_ref();
    let engine = MatchEngine::new(candidates, criteria, request.strategy);
    if RatioPolicy::fallback_above_primary(validated.match_ratio, request.ratio_fallback) {
        return invalid_criteria_outcome(
            anchors.len(),
            InvalidCriteriaReason::FallbackRatioAbovePrimary,
        );
    }
    let ratio_policy = RatioPolicy::from_fallback(validated.match_ratio, request.ratio_fallback);
    let mut diagnostics =
        initial_diagnostics(anchors.len(), criteria.estimand, criteria.common_support);
    let listed = discarded.map(|units| {
        let indices = |rows: &[R], ids: &HashSet<String>| -> HashSet<usize> {
            rows.iter()
                .enumerate()
                .filter(|(_, row)| ids.contains(row.id()))
                .map(|(idx, _)| idx)
                .collect()
        };
        (
            indices(anchors, &units.anchors),
            indices(candidates, &units.candidates),
        )
    });
    if let Some((trimmed_anchors, trimmed_candidates)) = &listed {
        diagnostics.common_support_trimmed_anchors = trimmed_anchors.len();
        diagnostics.common_support_trimmed_candidates = trimmed_candidates.len();
    }
    let support = match criteria.common_support {
        _ if listed.is_some() => None,
        None => None,
        Some(policy) => {
            let Some(scores) = request
                .distance_config
                .and_then(DistanceConfig::propensity_scores)
            else {
                return invalid_common_support_outcome(
                    anchors.len(),
                    CommonSupportFailureReason::RequiresPropensityScoreMap,
                );
            };
            match standard_common_support(anchors, candidates, policy, scores) {
                Ok(support) => {
                    support.record_in(&mut diagnostics);
                    Some(support)
                }
                Err(reason) => return invalid_common_support_outcome(anchors.len(), reason),
            }
        }
    };
    let trimmed = match &listed {
        Some((trimmed_anchors, trimmed_candidates)) => TrimmedIndexConfig {
            anchors: Some(trimmed_anchors),
            candidates: Some(trimmed_candidates),
        },
        None => TrimmedIndexConfig {
            anchors: support.as_ref().map(|support| &support.anchors),
            candidates: support.as_ref().map(|support| &support.candidates),
        },
    };

    if let Some(nearest) = engine.selector.nearest() {
        if !request.ratio_fallback.is_empty() {
            return invalid_criteria_outcome(anchors.len(), InvalidCriteriaReason::NearestOptions);
        }
        return super::nearest::match_nearest(
            &super::nearest::NearestRequest {
                anchors,
                candidates,
                criteria,
                constraints: request.constraints,
                distance_config: request.distance_config,
                trimmed,
            },
            nearest,
            diagnostics,
        );
    }
    if engine.selector.assignment() == super::selection::Assignment::Rounds
        && !request.ratio_fallback.is_empty()
    {
        return invalid_criteria_outcome(anchors.len(), InvalidCriteriaReason::NearestOptions);
    }

    // The unconfigured path is the common one, so it keeps the concrete date
    // metric rather than paying `ConfiguredMetric`'s match on every candidate.
    match request.distance_config {
        None => engine.run_anchor_matching_with_distance_internal(
            anchors,
            &DistanceRunConfig {
                distance_metric: &DateDistance,
                distance_caliper: criteria.typed_birth_date_caliper(),
                use_birth_date_index: true,
                caliper_reason: "date_caliper",
            },
            request.constraints,
            &ratio_policy,
            trimmed,
            diagnostics,
        ),
        Some(config) => engine.run_anchor_matching_with_distance_internal(
            anchors,
            &DistanceRunConfig {
                distance_metric: &config.metric(),
                distance_caliper: config.typed_caliper(),
                use_birth_date_index: false,
                caliper_reason: config.reason(),
            },
            request.constraints,
            &ratio_policy,
            trimmed,
            diagnostics,
        ),
    }
}

/// The units that common-support trimming removes from a standard match.
pub(crate) struct StandardCommonSupport {
    pub(crate) anchors: HashSet<usize>,
    pub(crate) candidates: HashSet<usize>,
    overlap: (f64, f64),
    anchor_bounds: (f64, f64),
    candidate_bounds: (f64, f64),
}

impl StandardCommonSupport {
    fn record_in(&self, diagnostics: &mut MatchDiagnostics) {
        diagnostics.common_support_trimmed_anchors = self.anchors.len();
        diagnostics.common_support_trimmed_candidates = self.candidates.len();
        diagnostics.common_support_overlap = Some(self.overlap);
        diagnostics.common_support_anchor_score_bounds = Some(self.anchor_bounds);
        diagnostics.common_support_candidate_score_bounds = Some(self.candidate_bounds);
    }
}

/// The smallest and largest finite score among `rows`, or `None` when no row
/// has one.
fn finite_score_bounds<R: MatchingRecord>(
    rows: &[R],
    scores: &RapidHashMap<String, f64>,
) -> Option<(f64, f64)> {
    rows.iter()
        .filter_map(|row| scores.get(row.id()).copied())
        .filter(|score| score.is_finite())
        .fold(None, |bounds, score| match bounds {
            None => Some((score, score)),
            Some((low, high)) => Some((low.min(score), high.max(score))),
        })
}

/// Trim to the region where the anchors' and the candidates' scores overlap.
///
/// The region is the interval from the larger of the two minima to the smaller
/// of the two maxima, with its ends inside. `Treated` removes the anchors and
/// `Control` the candidates outside it, which for each group is the same as
/// removing the units outside the other group's range. `Both` removes both.
/// A unit without a finite score lies outside the region. This is MatchIt's
/// documented `discard` rule: a unit is discarded when its distance is outside
/// the range of the other group's distances, so a unit equal to an end of that
/// range is kept.
pub(crate) fn standard_common_support<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    policy: CommonSupport,
    scores: &RapidHashMap<String, f64>,
) -> Result<StandardCommonSupport, CommonSupportFailureReason> {
    let anchor_bounds =
        finite_score_bounds(anchors, scores).ok_or(CommonSupportFailureReason::NoOverlap)?;
    let candidate_bounds =
        finite_score_bounds(candidates, scores).ok_or(CommonSupportFailureReason::NoOverlap)?;
    let lower = anchor_bounds.0.max(candidate_bounds.0);
    let upper = anchor_bounds.1.min(candidate_bounds.1);
    if lower > upper {
        return Err(CommonSupportFailureReason::NoOverlap);
    }
    let outside = |rows: &[R]| -> HashSet<usize> {
        rows.iter()
            .enumerate()
            .filter(|(_, row)| {
                !scores
                    .get(row.id())
                    .is_some_and(|score| score.is_finite() && *score >= lower && *score <= upper)
            })
            .map(|(idx, _)| idx)
            .collect()
    };
    Ok(StandardCommonSupport {
        anchors: if matches!(policy, CommonSupport::Treated | CommonSupport::Both) {
            outside(anchors)
        } else {
            HashSet::new()
        },
        candidates: if matches!(policy, CommonSupport::Control | CommonSupport::Both) {
            outside(candidates)
        } else {
            HashSet::new()
        },
        overlap: (lower, upper),
        anchor_bounds,
        candidate_bounds,
    })
}

/// Partition `anchors` by strata key, returning one `Vec<usize>` of anchor indices per group.
/// Anchors trimmed by `trimmed_indices` are excluded.
fn group_anchors_by_strata<R: MatchingRecord>(
    anchors: &[R],
    required_strata: &[String],
    trimmed_indices: &TrimmedIndexConfig<'_>,
) -> Vec<Vec<usize>> {
    // Groups come back in the order of their first anchor, so the order of the
    // pairs is a function of the input order and of nothing else, in every
    // process. Sorting the keys would need an order on `Vec<Option<String>>`
    // and gains nothing.
    let mut index_of: FxHashMap<Vec<Option<&str>>, usize> = FxHashMap::default();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (anchor_idx, case) in anchors.iter().enumerate() {
        if trimmed_indices
            .anchors
            .is_some_and(|trimmed| trimmed.contains(&anchor_idx))
        {
            continue;
        }
        let key: Vec<Option<&str>> = required_strata.iter().map(|k| case.stratum(k)).collect();
        if let Some(&position) = index_of.get(&key) {
            groups[position].push(anchor_idx);
        } else {
            index_of.insert(key, groups.len());
            groups.push(vec![anchor_idx]);
        }
    }
    groups
}

/// The id of a uniqueness value, numbered in order of first use.
pub(crate) fn intern_unique(
    interner: &mut RapidHashMap<String, UniqueValueId>,
    value: &str,
) -> UniqueValueId {
    if let Some(id) = interner.get(value) {
        return *id;
    }
    let id = UniqueValueId::new(interner.len());
    interner.insert(value.to_string(), id);
    id
}

/// The first eight bytes of an id, zero-padded, as a big-endian integer.
///
/// For two ids, the order of these integers is the order of the ids whenever
/// the integers differ, so a comparison only reads the ids when they agree.
fn id_prefix(id: &str) -> u64 {
    let bytes = id.as_bytes();
    let mut word = [0_u8; 8];
    let len = bytes.len().min(8);
    word[..len].copy_from_slice(&bytes[..len]);
    u64::from_be_bytes(word)
}

/// Candidates per task when a pool is scanned in parallel.
const SCAN_CHUNK: usize = 1024;

/// How a candidate pool is scanned and ordered.
#[derive(Clone, Copy)]
enum PoolScan {
    /// One thread, ordered by comparing ids.
    Narrow,
    /// Long candidate lists are split over the rayon pool and ordered by rank.
    Wide,
}

struct ScanContext<'a, R: MatchingRecord, D: DistanceMetric<R>, G: ?Sized> {
    case: &'a R,
    pool_config: &'a CandidatePoolConfig<'a, D>,
    extra_constraints: &'a G,
    case_strata_values: Option<&'a [Option<&'a str>]>,
    caliper_constraint: Option<&'a CaliperConstraint<'a, R, D>>,
    strata_indexed: bool,
    self_match: SelfMatch,
}

/// How a control that is the case itself is recognised.
#[derive(Clone, Copy)]
enum SelfMatch {
    /// By comparing ids.
    ById,
    /// Ids are distinct and this control has the case's id.
    At(usize),
    /// Ids are distinct and no control has the case's id.
    Nobody,
}

/// Blocks the one control that is the case itself, found by position.
struct SelfMatchAt(Option<usize>);

impl<R: MatchingRecord> Constraint<R> for SelfMatchAt {
    fn reason(&self) -> &'static str {
        Constraint::<R>::reason(&NoSelfMatchConstraint)
    }

    fn allows(&self, _case: &R, _control: &R, ctx: &ConstraintContext<'_>) -> bool {
        self.0 != Some(ctx.control_idx.get())
    }
}

impl<R: MatchingRecord, D: DistanceMetric<R>, G: ?Sized> Clone for ScanContext<'_, R, D, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: MatchingRecord, D: DistanceMetric<R>, G: ?Sized> Copy for ScanContext<'_, R, D, G> {}

/// The outcome of scanning a run of candidates.
#[derive(Default)]
struct ScanTally {
    eligible: Vec<usize>,
    filtered_out: usize,
    constraint_exclusions: Vec<(&'static str, usize)>,
}

impl ScanTally {
    fn exclude(&mut self, reason: &'static str, count: usize) {
        match self
            .constraint_exclusions
            .iter_mut()
            .find(|(seen, _)| *seen == reason)
        {
            Some((_, total)) => *total += count,
            None => self.constraint_exclusions.push((reason, count)),
        }
    }

    /// The tallies of consecutive runs as one. When the ranks are given, every
    /// run's eligible controls are in id order and so are the merged ones.
    fn join_all(tallies: Vec<Self>, ranks: Option<&IdRanks>) -> Self {
        let mut joined = Self::default();
        let mut runs: Vec<Vec<usize>> = Vec::with_capacity(tallies.len());
        for tally in tallies {
            joined.filtered_out += tally.filtered_out;
            for (reason, count) in tally.constraint_exclusions {
                joined.exclude(reason, count);
            }
            runs.push(tally.eligible);
        }
        joined.eligible = match ranks {
            Some(ranks) => ranks.merge(runs),
            None => runs.concat(),
        };
        joined
    }
}

/// Position of every control in the order of ids, for controls with distinct ids.
struct IdRanks {
    rank: Vec<u32>,
    by_rank: Vec<u32>,
}

impl IdRanks {
    fn of_controls<R: MatchingRecord>(controls: &[R]) -> Option<Self> {
        let mut order: Vec<usize> = (0..controls.len()).collect();
        sort_by_id(controls, &mut order);
        if order
            .windows(2)
            .any(|pair| controls[pair[0]].id() == controls[pair[1]].id())
        {
            return None;
        }
        let mut rank = vec![0_u32; controls.len()];
        let mut by_rank = Vec::with_capacity(controls.len());
        for (position, &idx) in order.iter().enumerate() {
            rank[idx] = u32::try_from(position).ok()?;
            by_rank.push(u32::try_from(idx).ok()?);
        }
        Some(Self { rank, by_rank })
    }

    /// The control with this id, if any.
    fn position_of<R: MatchingRecord>(&self, controls: &[R], id: &str) -> Option<usize> {
        let at = self
            .by_rank
            .partition_point(|&idx| controls[idx as usize].id() < id);
        let idx = *self.by_rank.get(at)? as usize;
        (controls[idx].id() == id).then_some(idx)
    }

    /// Sort control indices by id.
    fn sort(&self, indices: &mut Vec<usize>) {
        let mut ranks: Vec<u32> = indices.iter().map(|&idx| self.rank[idx]).collect();
        ranks.sort_unstable();
        indices.clear();
        indices.extend(
            ranks
                .into_iter()
                .map(|rank| self.by_rank[rank as usize] as usize),
        );
    }

    /// Merge runs of control indices that are each in id order into one run.
    fn merge(&self, mut runs: Vec<Vec<usize>>) -> Vec<usize> {
        while runs.len() > 1 {
            let mut next = Vec::with_capacity(runs.len().div_ceil(2));
            let mut pending = runs.into_iter();
            while let Some(first) = pending.next() {
                match pending.next() {
                    Some(second) => next.push(self.merge_two(&first, &second)),
                    None => next.push(first),
                }
            }
            runs = next;
        }
        runs.pop().unwrap_or_default()
    }

    fn merge_two(&self, left: &[usize], right: &[usize]) -> Vec<usize> {
        let mut out = Vec::with_capacity(left.len() + right.len());
        let (mut i, mut j) = (0, 0);
        while i < left.len() && j < right.len() {
            if self.rank[left[i]] <= self.rank[right[j]] {
                out.push(left[i]);
                i += 1;
            } else {
                out.push(right[j]);
                j += 1;
            }
        }
        out.extend_from_slice(&left[i..]);
        out.extend_from_slice(&right[j..]);
        out
    }
}

/// Sort control indices by the id of the control, as `str::cmp` orders them.
fn sort_by_id<R: MatchingRecord>(controls: &[R], indices: &mut Vec<usize>) {
    let mut keyed: Vec<(u64, usize)> = indices
        .iter()
        .map(|&idx| (id_prefix(controls[idx].id()), idx))
        .collect();
    keyed.sort_unstable_by(|&(left_prefix, left), &(right_prefix, right)| {
        left_prefix
            .cmp(&right_prefix)
            .then_with(|| controls[left].id().cmp(controls[right].id()))
    });
    indices.clear();
    indices.extend(keyed.into_iter().map(|(_, idx)| idx));
}

pub struct MatchingPrecomputed<'a> {
    pub criteria: &'a MatchingCriteria,
    pub(crate) control_strata_values: Option<StrataValues<'a>>,
}

pub fn build_outcome(
    pairs: Vec<MatchedPair>,
    matched_cases: usize,
    anchor_count: usize,
    used_controls: usize,
    diagnostics: MatchDiagnostics,
) -> MatchOutcome {
    let avg_controls_per_case = if matched_cases == 0 {
        0.0
    } else {
        usize_to_f64(pairs.len()) / usize_to_f64(matched_cases)
    };

    MatchOutcome {
        pairs,
        unmatched_cases: anchor_count.saturating_sub(matched_cases),
        used_controls,
        matched_cases,
        avg_controls_per_case,
        diagnostics,
    }
}

pub fn to_f64(n: usize) -> f64 {
    usize_to_f64(n)
}

fn usize_to_f64(n: usize) -> f64 {
    // Exact up to 2^53.
    n as f64
}

pub fn invalid_criteria_outcome(
    anchor_count: usize,
    reason: InvalidCriteriaReason,
) -> MatchOutcome {
    let mut diagnostics = MatchDiagnostics {
        total_anchors_evaluated: anchor_count,
        ..MatchDiagnostics::default()
    };
    diagnostics
        .exclusion_counts
        .insert(ExclusionReason::InvalidCriteria(reason), 1);

    MatchOutcome {
        pairs: Vec::new(),
        unmatched_cases: anchor_count,
        used_controls: 0,
        matched_cases: 0,
        avg_controls_per_case: 0.0,
        diagnostics,
    }
}

/// The outcome of a cohort refused for a record-level reason, with no pairs.
pub fn invalid_cohort_outcome(anchor_count: usize, reason: ExclusionReason) -> MatchOutcome {
    let mut diagnostics = MatchDiagnostics {
        total_anchors_evaluated: anchor_count,
        ..MatchDiagnostics::default()
    };
    diagnostics.exclusion_counts.insert(reason, 1);

    MatchOutcome {
        pairs: Vec::new(),
        unmatched_cases: anchor_count,
        used_controls: 0,
        matched_cases: 0,
        avg_controls_per_case: 0.0,
        diagnostics,
    }
}

pub fn invalid_common_support_outcome(
    anchor_count: usize,
    reason: CommonSupportFailureReason,
) -> MatchOutcome {
    let mut diagnostics = MatchDiagnostics {
        total_anchors_evaluated: anchor_count,
        ..MatchDiagnostics::default()
    };
    diagnostics
        .exclusion_counts
        .insert(ExclusionReason::CommonSupportFailure(reason), 1);

    MatchOutcome {
        pairs: Vec::new(),
        unmatched_cases: anchor_count,
        used_controls: 0,
        matched_cases: 0,
        avg_controls_per_case: 0.0,
        diagnostics,
    }
}

#[cfg(test)]
mod strata_group_order_tests {
    use super::{TrimmedIndexConfig, group_anchors_by_strata};
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    fn anchor(id: &str, site: &str) -> BaseRecord {
        let mut record =
            BaseRecord::new(id, NaiveDate::from_ymd_opt(2005, 1, 1).expect("valid date"));
        record.strata.insert("site".to_string(), site.to_string());
        record
    }

    /// Strata groups come back in first-appearance order, in every process.
    ///
    /// A hash-ordered grouping would differ between processes, and one process
    /// cannot see that by running twice. The test asserts the order of the
    /// anchors instead, with three interleaved strata.
    #[test]
    fn groups_follow_first_appearance_of_their_strata() {
        // Interleaved on purpose: first appearances are b, a, c.
        let anchors: Vec<BaseRecord> = ["b", "a", "c", "a", "b", "c", "a"]
            .iter()
            .enumerate()
            .map(|(i, site)| anchor(&format!("p{i}"), site))
            .collect();
        let required = vec!["site".to_string()];
        let trimmed = TrimmedIndexConfig {
            anchors: None,
            candidates: None,
        };

        let groups = group_anchors_by_strata(&anchors, &required, &trimmed);

        assert_eq!(
            groups,
            vec![vec![0, 4], vec![1, 3, 6], vec![2, 5]],
            "groups must follow first appearance (b, a, c), each keeping anchor order"
        );
    }
}

#[cfg(test)]
mod id_order_tests {
    use super::{id_prefix, intern_unique, sort_by_id};
    use crate::types::BaseRecord;
    use chrono::NaiveDate;
    use rapidhash::RapidHashMap;

    fn records(ids: &[&str]) -> Vec<BaseRecord> {
        let date = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
        ids.iter().map(|id| BaseRecord::new(*id, date)).collect()
    }

    /// The prefix order agrees with `str` order wherever the prefixes differ,
    /// and the sort agrees with `str` order everywhere.
    #[test]
    fn sorting_by_prefix_then_id_is_sorting_by_id() {
        let ids = [
            "",
            "a",
            "a\0",
            "a\0\0",
            "ab",
            "abcdefgh",
            "abcdefgh0",
            "abcdefgh1",
            "abcdefgi",
            "abcdefg",
            "b",
            "0000000001",
            "0000000002",
            "00000000011",
            "\u{e5}",
            "z",
            "zz",
            "abcdefgh\0",
            "abcdefg\0h",
        ];
        for (left, right) in ids.iter().flat_map(|l| ids.iter().map(move |r| (l, r))) {
            let (lp, rp) = (id_prefix(left), id_prefix(right));
            if lp != rp {
                assert_eq!(lp.cmp(&rp), left.cmp(right), "{left:?} vs {right:?}");
            }
        }
        let controls = records(&ids);
        let mut indices: Vec<usize> = (0..ids.len()).rev().collect();
        sort_by_id(&controls, &mut indices);
        let sorted: Vec<&str> = indices.iter().map(|&i| ids[i]).collect();
        let mut expected = ids.to_vec();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn unique_values_are_numbered_in_order_of_first_use() {
        let mut interner = RapidHashMap::default();
        let first = intern_unique(&mut interner, "f1");
        let second = intern_unique(&mut interner, "f2");
        assert_ne!(first, second);
        assert_eq!(intern_unique(&mut interner, "f1"), first);
        assert_eq!(intern_unique(&mut interner, "f2"), second);
        assert_eq!(interner.len(), 2);
    }
}

#[cfg(test)]
mod pool_scan_tests {
    use super::*;
    use crate::matching::{DateDistance, RandomSelection};
    use crate::types::{BaseRecord, RoleTransitionRecord};
    use chrono::{Days, NaiveDate};
    use std::collections::BTreeMap;

    type Member = RoleTransitionRecord<BaseRecord>;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self, bound: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % bound
        }
    }

    fn origin() -> NaiveDate {
        NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date")
    }

    /// A cohort with few distinct dates, so that many events and entries tie.
    ///
    /// Ids share a long prefix, so that ordering them needs the full string.
    fn cohort(seed: u64, n: usize, strata: u64, duplicate_ids: bool) -> Vec<Member> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|i| {
                let id = if duplicate_ids && i % 97 == 0 {
                    "person_dup".to_string()
                } else {
                    format!("person_{:07}", rng.next(10_000_000) * 100_000 + i as u64)
                };
                let mut record = BaseRecord::new(&id, origin() + Days::new(rng.next(5)));
                record
                    .strata
                    .insert("s".to_string(), format!("s{}", rng.next(strata)));
                if rng.next(4) == 0 {
                    record.entry_date = Some(origin() + Days::new(rng.next(20) * 5));
                }
                if rng.next(5) == 0 {
                    record.death_date = Some(origin() + Days::new(50 + rng.next(30) * 5));
                }
                let event = (rng.next(8) == 0).then(|| origin() + Days::new(20 + rng.next(30) * 5));
                RoleTransitionRecord::from_record(record, event)
            })
            .collect()
    }

    fn at_risk(control: &Member, event: NaiveDate) -> bool {
        control.record.entry_date.is_none_or(|entry| entry <= event)
            && control.transition_date.is_none_or(|date| date > event)
            && control.record.death_date.is_none_or(|death| death > event)
    }

    /// The pool as a single thread builds it, with the order given by comparing ids.
    fn reference_pool<G: ConstraintGroup<Member> + ?Sized>(
        engine: &MatchEngine<'_, Member, RandomSelection>,
        case: &Member,
        pool_config: &CandidatePoolConfig<'_, DateDistance>,
        extra_constraints: &G,
        additional_filter: impl Fn(usize, &Member) -> bool,
        diagnostics: &mut MatchDiagnostics,
    ) -> Vec<usize> {
        let case_strata_values = (!engine.criteria.required_strata.is_empty())
            .then(|| build_strata_values(case, &engine.criteria.required_strata));
        let window_days = engine
            .criteria
            .typed_birth_date_window()
            .map_or(0_i64, |days| i64::from(days.get()));
        let candidate_indices = engine.candidate_index.candidate_indices(
            case.birth_date(),
            window_days,
            case_strata_values.as_deref(),
        );
        let caliper_constraint = pool_config.distance_caliper.map(|max_distance| {
            CaliperConstraint::new(
                pool_config.distance_metric,
                max_distance.get(),
                pool_config.caliper_reason,
            )
        });
        let builtin: [&dyn Constraint<Member>; 3] = [
            &ReplacementConstraint,
            &NoSelfMatchConstraint,
            &UniqueKeyConstraint,
        ];
        let mut eligible = Vec::new();
        let mut filtered_out = 0_usize;
        let mut exclusions: BTreeMap<&'static str, usize> = BTreeMap::new();
        for idx in candidate_indices {
            let control = &engine.controls[idx];
            if !additional_filter(idx, control) {
                filtered_out += 1;
                continue;
            }
            let context = ConstraintContext {
                criteria: engine.criteria,
                used_controls: pool_config.used_controls,
                used_unique: pool_config.used_unique,
                unique_interner: pool_config.unique_interner,
                control_idx: ControlIdx::new(idx),
                case_strata_values: case_strata_values.as_deref(),
                control_strata_values: engine
                    .precomputed
                    .control_strata_values
                    .as_ref()
                    .and_then(|values| values.row(idx)),
            };
            if let Some(reason) = first_blocking_constraint_reason(
                case,
                control,
                &builtin,
                caliper_constraint.as_ref(),
                extra_constraints,
                &context,
            ) {
                *exclusions.entry(reason).or_insert(0) += 1;
                continue;
            }
            eligible.push(idx);
        }
        if filtered_out > 0 {
            *diagnostics
                .exclusion_counts
                .entry(ExclusionReason::AdditionalFilter)
                .or_insert(0) += filtered_out;
        }
        for (reason, count) in exclusions {
            *diagnostics
                .exclusion_counts
                .entry(ExclusionReason::Constraint(
                    ConstraintReason::from_reason_str(reason),
                ))
                .or_insert(0) += count;
        }
        sort_by_id(engine.controls, &mut eligible);
        eligible
    }

    /// Walk the cases in event order, as risk-set matching does, and compare
    /// every pool and its exclusion counts with the single-threaded reference.
    /// Controls are drawn from the pool in the same way for both, so a
    /// difference in order would send the walks apart.
    fn walk(seed: u64, n: usize, strata: u64, replace: bool, duplicate_ids: bool) {
        let members = cohort(seed, n, strata, duplicate_ids);
        let criteria = MatchingCriteria::builder()
            .required_strata(vec!["s".to_string()])
            .birth_date_window_days(1000)
            .allow_replacement(replace)
            .build()
            .validate()
            .expect("valid criteria");
        let engine = MatchEngine::new(&members, criteria.as_ref(), RandomSelection::seeded(seed));
        let mut selector = RandomSelection::seeded(seed);
        let mut used = UsedControlsVec::with_capacity(n);
        let unique = FxHashSet::default();
        let interner = RapidHashMap::default();
        let channel = DistanceChannel::new(&DateDistance, criteria.typed_birth_date_caliper());

        let mut cases: Vec<usize> = (0..n)
            .filter(|&i| members[i].transition_date.is_some())
            .collect();
        cases.sort_by(|&a, &b| {
            members[a]
                .transition_date
                .cmp(&members[b].transition_date)
                .then_with(|| members[a].record.id.cmp(&members[b].record.id))
        });
        assert!(cases.len() > 10);
        // Every case of a short cohort, and the first ones of a long cohort.
        cases.truncate(if n > 5_000 { 150 } else { usize::MAX });
        let mut compared_wide = false;
        for case_idx in cases {
            let case = &members[case_idx];
            let event = case.transition_date.expect("a case has an event date");
            let filter = |_: usize, control: &Member| at_risk(control, event);

            let mut new_diagnostics = MatchDiagnostics::default();
            let mut pool = engine.candidate_pool_for_request_and_constraint_group(
                &CandidatePoolRequest {
                    case,
                    used_controls: &used,
                    used_unique: &unique,
                    unique_interner: &interner,
                    distance_channel: channel,
                    use_birth_date_index: true,
                },
                &(),
                filter,
                &mut new_diagnostics,
            );
            let mut reference_diagnostics = MatchDiagnostics::default();
            let reference = reference_pool(
                &engine,
                case,
                &CandidatePoolConfig {
                    used_controls: &used,
                    used_unique: &unique,
                    unique_interner: &interner,
                    distance_metric: channel.metric(),
                    distance_caliper: channel.typed_caliper(),
                    use_birth_date_index: true,
                    caliper_reason: channel.reason(),
                },
                &(),
                filter,
                &mut reference_diagnostics,
            );
            compared_wide |= reference.len() >= 2 * SCAN_CHUNK;
            assert_eq!(pool, reference, "pool of case {case_idx}");
            assert_eq!(
                new_diagnostics.exclusion_counts, reference_diagnostics.exclusion_counts,
                "exclusion counts of case {case_idx}"
            );
            if let Some(selected) = engine_select(&mut selector, case, &members, &mut pool) {
                if !replace {
                    used.insert(ControlIdx::new(selected));
                }
            }
        }
        if n >= 20_000 {
            assert!(compared_wide, "a pool must be long enough to split");
        }
    }

    fn engine_select(
        selector: &mut RandomSelection,
        case: &Member,
        members: &[Member],
        pool: &mut Vec<usize>,
    ) -> Option<usize> {
        let position = selector.select_position(case, members, pool)?;
        Some(pool.swap_remove(position))
    }

    #[test]
    fn short_pools_with_and_without_replacement() {
        for seed in [1, 2, 3, 4, 5] {
            for replace in [false, true] {
                walk(seed, 600, 3, replace, false);
            }
        }
    }

    #[test]
    fn long_pools_split_over_threads_are_the_same() {
        for seed in [11, 12] {
            for replace in [false, true] {
                walk(seed, 24_000, 2, replace, false);
            }
        }
    }

    #[test]
    fn repeated_ids_fall_back_to_comparing_ids() {
        for replace in [false, true] {
            walk(21, 600, 3, replace, true);
            walk(22, 24_000, 2, replace, true);
        }
    }

    #[test]
    fn a_pool_is_the_same_on_every_thread_count() {
        let members = cohort(31, 24_000, 2, false);
        let criteria = MatchingCriteria::builder()
            .required_strata(vec!["s".to_string()])
            .birth_date_window_days(1000)
            .allow_replacement(true)
            .build()
            .validate()
            .expect("valid criteria");
        let engine = MatchEngine::new(&members, criteria.as_ref(), RandomSelection::seeded(1));
        let case = members
            .iter()
            .find(|m| m.transition_date.is_some())
            .expect("a case");
        let event = case.transition_date.expect("event date");
        let channel = DistanceChannel::new(&DateDistance, criteria.typed_birth_date_caliper());
        let used = UsedControlsVec::with_capacity(members.len());
        let unique = FxHashSet::default();
        let interner = RapidHashMap::default();
        let pool_with = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| {
                    let mut diagnostics = MatchDiagnostics::default();
                    let pool = engine.candidate_pool_for_request_and_constraint_group(
                        &CandidatePoolRequest {
                            case,
                            used_controls: &used,
                            used_unique: &unique,
                            unique_interner: &interner,
                            distance_channel: channel,
                            use_birth_date_index: true,
                        },
                        &(),
                        |_, control: &Member| at_risk(control, event),
                        &mut diagnostics,
                    );
                    (pool, diagnostics.exclusion_counts)
                })
        };
        let one = pool_with(1);
        assert!(one.0.len() > 2 * SCAN_CHUNK);
        assert_eq!(one, pool_with(3));
        assert_eq!(one, pool_with(12));
    }
}
