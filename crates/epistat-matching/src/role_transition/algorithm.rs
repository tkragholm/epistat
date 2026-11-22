use super::IdOrder;
use super::sweep::{RiskSetSweep, list_limit};
use crate::matching::{
    CandidatePoolRequest, Constraint, ConstraintGroup, DateDistance, DistanceChannel,
    DistanceConfig, DistanceMetric, MatchEngine, ReplacementConstraint, RoleIndexedRecord,
    ScoreTable, SelectionStrategy, UsedControlsVec, build_outcome, build_strata_values,
    finalize_estimand_diagnostics, intern_unique, invalid_cohort_outcome,
    invalid_common_support_outcome, invalid_criteria_outcome, ratio::RatioPolicy, unique_value,
};
use crate::types::{
    CommonSupport, CommonSupportFailureReason, ConstraintReason, ControlIdx, ExclusionReason,
    InvalidCriteriaReason, MatchDiagnostics, MatchOutcome, MatchRatio, MatchedPair,
    MatchingCriteria, RoleTransitionOptions, UniqueValueId,
};
use chrono::{Datelike, NaiveDate};
use epistat_register::{AgeRule, LeapBirthday, age_in_years};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rapidhash::RapidHashMap;
use std::collections::{BTreeSet, HashSet};

/// Policy hook for role-transition risk-set construction.
///
/// This allows downstream orchestrators to inject study-specific entry/exit logic
/// (e.g. alive/resident checks) while keeping matching mechanics generic.
/// Candidate lists are scanned on several threads, so a policy is `Sync`.
pub trait RiskSetPolicy<R: RoleIndexedRecord>: Sync {
    /// Whether a record should be treated as an eligible case.
    ///
    /// Default behavior requires an event date before the configured age limit.
    fn is_case_eligible(&self, row: &R, age_limit_years: u8) -> bool {
        is_case_before_age_limit(row, age_limit_years)
    }

    /// Whether a record is at risk (eligible as a candidate) at the case event date.
    ///
    /// Default behavior allows candidates that have entered on or before the
    /// case event date (a record without an entry date has always been at risk)
    /// and that have no event date or an event date strictly after the case
    /// event date.
    fn is_control_at_risk(&self, _case: &R, control: &R, case_event_date: NaiveDate) -> bool {
        default_control_is_at_risk(control, case_event_date)
    }

    /// `Some` when [`Self::is_control_at_risk`] is the default rule, which
    /// reads only the entry and event dates of the control. Common-support
    /// trimming then finds the scores at risk on each case date in one sweep,
    /// and so does the matching where the job allows it (see
    /// [`match_transition`]).
    ///
    /// [`DefaultAtRiskRule`] can only be made inside this crate, so only
    /// [`DefaultRiskSetPolicy`] answers `Some` and a policy of your own cannot
    /// claim the sweep for a rule it has changed. Every other policy answers
    /// `None`, and the trimming asks [`Self::is_control_at_risk`] of every
    /// record for every case.
    fn default_at_risk_rule(&self) -> Option<DefaultAtRiskRule> {
        None
    }

    /// The span of days on which `control` is at risk, for a policy whose
    /// [`Self::is_control_at_risk`] depends on the control and the date alone
    /// and is true on the days of one span. `None`, the default, when the
    /// policy does not say.
    ///
    /// A policy that answers `Some` for every record promises that
    /// `is_control_at_risk(case, control, date)` is
    /// `control_at_risk_span(control).contains(date)` for every case. Matching
    /// then keeps the risk sets by a sweep over the case dates, as it does
    /// under the default rule (see [`match_transition`]), and gives the pairs
    /// and diagnostics of the scan. A policy that answers `None` for any
    /// record is scanned.
    fn control_at_risk_span(&self, _control: &R) -> Option<AtRiskSpan> {
        None
    }
}

/// The days on which a control is at risk, whatever the case; see
/// [`RiskSetPolicy::control_at_risk_span`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtRiskSpan {
    /// Not at risk on any day.
    Never,
    /// At risk on the days from `from` up to but not including `until`. A
    /// missing `from` is before every day and a missing `until` after every
    /// day.
    Between {
        /// The first day at risk.
        from: Option<NaiveDate>,
        /// The first day no longer at risk.
        until: Option<NaiveDate>,
    },
}

impl AtRiskSpan {
    /// The span of the default rule: from the entry date up to but not
    /// including the event date.
    #[must_use]
    pub fn of_default_rule<R: RoleIndexedRecord>(control: &R) -> Self {
        Self::Between {
            from: control.entry_date(),
            until: control.event_date(),
        }
    }

    /// The span cut to end on `until` at the latest, which is not a day of
    /// it. A missing `until` leaves the span as it is.
    #[must_use]
    pub fn ending_by(self, until: Option<NaiveDate>) -> Self {
        match (self, until) {
            (Self::Between { from, until: end }, Some(cut)) => Self::Between {
                from,
                until: Some(end.map_or(cut, |end| end.min(cut))),
            },
            (span, _) => span,
        }
    }

    /// Whether `date` is a day of the span.
    #[must_use]
    pub fn contains(self, date: NaiveDate) -> bool {
        match self {
            Self::Never => false,
            Self::Between { from, until } => {
                from.is_none_or(|from| from <= date) && until.is_none_or(|until| date < until)
            }
        }
    }
}

/// Proof that a risk-set policy applies the default at-risk rule. Only
/// [`DefaultRiskSetPolicy`] holds one.
#[derive(Debug, Clone, Copy)]
pub struct DefaultAtRiskRule(());

/// Default role-transition risk-set policy used by existing APIs.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultRiskSetPolicy;

impl<R: RoleIndexedRecord> RiskSetPolicy<R> for DefaultRiskSetPolicy {
    fn default_at_risk_rule(&self) -> Option<DefaultAtRiskRule> {
        Some(DefaultAtRiskRule(()))
    }
}

/// Unified request for role-transition matching.
#[derive(Clone, Copy)]
pub struct TransitionMatchRequest<'a, S, G: ?Sized, P> {
    pub criteria: &'a MatchingCriteria,
    pub options: &'a RoleTransitionOptions,
    pub strategy: S,
    pub constraints: &'a G,
    pub distance_config: Option<&'a DistanceConfig>,
    pub risk_set_policy: &'a P,
}

impl<'a, S, G: ?Sized> TransitionMatchRequest<'a, S, G, DefaultRiskSetPolicy> {
    #[must_use]
    pub const fn new(
        criteria: &'a MatchingCriteria,
        options: &'a RoleTransitionOptions,
        strategy: S,
        constraints: &'a G,
    ) -> Self {
        Self {
            criteria,
            options,
            strategy,
            constraints,
            distance_config: None,
            risk_set_policy: &DefaultRiskSetPolicy,
        }
    }
}

impl<'a, S, G: ?Sized, P> TransitionMatchRequest<'a, S, G, P> {
    crate::impl_with_distance_config!();

    #[must_use]
    pub fn with_risk_set_policy<P2>(
        self,
        risk_set_policy: &'a P2,
    ) -> TransitionMatchRequest<'a, S, G, P2> {
        TransitionMatchRequest {
            criteria: self.criteria,
            options: self.options,
            strategy: self.strategy,
            constraints: self.constraints,
            distance_config: self.distance_config,
            risk_set_policy,
        }
    }
}

/// Risk-set matching with role transitions using neutral terminology.
///
/// With `common_support` set in the criteria the scores of the distance
/// configuration decide the trimming: a case whose score is strictly outside
/// the range of the scores of the records at risk on its event date (by the
/// risk-set rule and the alive and resident checks) is not matched, and a
/// candidate whose score is outside the range of the cases' scores is not
/// used. See [`CommonSupport`].
///
/// Cases are matched in order of event date. The risk sets are kept by a
/// sweep over the case dates rather than scanned for each case when the
/// at-risk rule is given as a span of each control ([`DefaultRiskSetPolicy`],
/// or a policy that answers [`RiskSetPolicy::control_at_risk_span`]), the
/// birth date is the distance (no distance configuration), the selection is a
/// seeded [`RandomSelection`](crate::RandomSelection), there is no unique key
/// and every constraint is an alive check, a residence check by periods
/// ([`MatchJob::with_resident_spells`](crate::MatchJob::with_resident_spells))
/// or a caliper on a numeric field. Without a caliper the sweep takes time
/// close to linear in the cohort; a caliper is tested on the controls of each
/// case's window. Any other job scans the candidates of each case's stratum
/// and birth window, in time proportional to the cases times the candidates
/// per case. Both give the same pairs and diagnostics.
#[must_use]
pub fn match_transition<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R>,
>(
    cohort: &[R],
    request: TransitionMatchRequest<'_, S, G, P>,
) -> MatchOutcome {
    match_transition_ordered(cohort, request, None)
}

/// [`match_transition`] with the order of the cohort's ids, when the caller
/// has it. An order of another length is not used.
pub(crate) fn match_transition_ordered<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R>,
>(
    cohort: &[R],
    request: TransitionMatchRequest<'_, S, G, P>,
    id_order: Option<&IdOrder>,
) -> MatchOutcome {
    let id_order = id_order.filter(|order| order.len() == cohort.len());
    // Without a configuration the date metric runs off the birth-date index and
    // the criteria's own window. That path is the common one, so it keeps the
    // concrete metric rather than paying `ConfiguredMetric`'s match on every
    // candidate.
    let Some(config) = request.distance_config else {
        let caliper = request.criteria.typed_birth_date_caliper();
        return match_with_distance_channel(
            cohort,
            request,
            DistanceChannel::new(&DateDistance, caliper),
            true,
            None,
            id_order,
        );
    };
    let scores = config.propensity_scores();
    if scores.is_none() && request.criteria.common_support.is_some() {
        return invalid_common_support_outcome(
            cohort.len(),
            CommonSupportFailureReason::RequiresPropensityScoreMap,
        );
    }
    let metric = config.metric();
    match_with_distance_channel(
        cohort,
        request,
        DistanceChannel::new(&metric, config.typed_caliper()).with_reason(config.reason()),
        false,
        scores,
        id_order,
    )
}

fn match_with_distance_channel<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R>,
>(
    cohort: &[R],
    request: TransitionMatchRequest<'_, S, G, P>,
    distance_channel: DistanceChannel<'_, R, D>,
    use_birth_date_index: bool,
    common_support_scores: Option<&ScoreTable>,
    id_order: Option<&IdOrder>,
) -> MatchOutcome {
    match_with_role_indexing_with_channel_internal(
        cohort,
        request.strategy,
        &RoleIndexingConfig {
            criteria: request.criteria,
            age_limit_years: request.options.transition_age_limit_years.get(),
            ratio_fallback: &request.options.ratio_fallback,
            extra_constraints: request.constraints,
            distance_channel,
            use_birth_date_index,
            common_support_scores,
            risk_set_policy: request.risk_set_policy,
            id_order,
        },
    )
}

#[derive(Clone, Copy)]
struct RoleIndexingConfig<
    'a,
    R: RoleIndexedRecord,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R> + ?Sized,
> {
    criteria: &'a MatchingCriteria,
    age_limit_years: u8,
    ratio_fallback: &'a [MatchRatio],
    extra_constraints: &'a G,
    distance_channel: DistanceChannel<'a, R, D>,
    use_birth_date_index: bool,
    common_support_scores: Option<&'a ScoreTable>,
    risk_set_policy: &'a P,
    /// The order of the cohort's ids, when the caller has it.
    id_order: Option<&'a IdOrder>,
}

#[derive(Default)]
struct TransitionCommonSupportFilters {
    trimmed_case_indices: HashSet<usize>,
    trimmed_candidate_indices: HashSet<usize>,
    overlap: Option<(f64, f64)>,
    case_score_bounds: Option<(f64, f64)>,
    candidate_score_bounds: Option<(f64, f64)>,
    policy: Option<CommonSupport>,
}

fn score_bounds_for_indices<R: RoleIndexedRecord>(
    cohort: &[R],
    indices: &[usize],
    scores: &ScoreTable,
) -> Option<(f64, f64)> {
    let mut min_score = f64::INFINITY;
    let mut max_score = f64::NEG_INFINITY;
    let mut found = false;

    for idx in indices {
        let Some(row) = cohort.get(*idx) else {
            continue;
        };
        let Some(score) = scores.get(row.id()).copied() else {
            continue;
        };
        if !score.is_finite() {
            continue;
        }

        min_score = min_score.min(score);
        max_score = max_score.max(score);
        found = true;
    }

    found.then_some((min_score, max_score))
}

fn support_interval(left: (f64, f64), right: (f64, f64)) -> Option<(f64, f64)> {
    let lower = left.0.max(right.0);
    let upper = left.1.min(right.1);
    (lower <= upper).then_some((lower, upper))
}

fn score_inside_interval<R: RoleIndexedRecord>(
    row: &R,
    scores: &ScoreTable,
    interval: (f64, f64),
) -> bool {
    scores
        .get(row.id())
        .copied()
        .is_some_and(|score| score.is_finite() && score >= interval.0 && score <= interval.1)
}

/// For each case, the smallest and largest finite score among the records at
/// risk on the case's event date, or `None` where nobody is at risk or the
/// case has no event date. Cases are given in order of event date.
///
/// A record is at risk when the risk-set policy says so and the status
/// constraints of `constraints` (alive, resident) allow it, as of the case's
/// event date. No other constraint is read.
///
/// With the default policy rule a record is at risk from its entry date up to
/// but not including its event date. The sweep visits the case dates in
/// order, adds the records that have entered and removes those whose event is
/// on or before the date, keeping the scores in a sorted set. The status
/// constraints depend on the case and may be any function of the date, so the
/// ends of the set are read from the outside in, skipping records they refuse.
fn at_risk_score_ranges<
    R: RoleIndexedRecord,
    P: RiskSetPolicy<R> + ?Sized,
    G: ConstraintGroup<R> + ?Sized,
>(
    cohort: &[R],
    case_indices: &[usize],
    scores: &ScoreTable,
    policy: &P,
    constraints: &G,
) -> Vec<Option<(f64, f64)>> {
    let score_of = |idx: usize| {
        scores
            .get(cohort[idx].id())
            .copied()
            .filter(|score| score.is_finite())
    };
    if policy.default_at_risk_rule().is_none() {
        return case_indices
            .iter()
            .map(|&case_idx| {
                let case = &cohort[case_idx];
                let date = case.event_date()?;
                let mut range: Option<(f64, f64)> = None;
                for (idx, control) in cohort.iter().enumerate() {
                    if let Some(score) = score_of(idx) {
                        if policy.is_control_at_risk(case, control, date)
                            && constraints.allows_at_risk(case, control)
                        {
                            range = Some(range.map_or((score, score), |(low, high)| {
                                (low.min(score), high.max(score))
                            }));
                        }
                    }
                }
                range
            })
            .collect();
    }

    // Records with a score, by entry date (no entry date first) and by event
    // date. Ties keep cohort order, which does not matter for a set.
    let with_score: Vec<usize> = (0..cohort.len())
        .filter(|&idx| score_of(idx).is_some())
        .collect();
    let by_entry: Vec<usize> = with_score
        .iter()
        .copied()
        .sorted_by_key(|&idx| cohort[idx].entry_date())
        .collect();
    let by_event: Vec<usize> = with_score
        .iter()
        .copied()
        .filter(|&idx| cohort[idx].event_date().is_some())
        .sorted_by_key(|&idx| cohort[idx].event_date())
        .collect();

    let mut at_risk: BTreeSet<(OrderedFloat<f64>, usize)> = BTreeSet::new();
    let (mut next_entry, mut next_event) = (0, 0);
    case_indices
        .iter()
        .map(|&case_idx| {
            let case = &cohort[case_idx];
            let date = case.event_date()?;
            while next_entry < by_entry.len()
                && cohort[by_entry[next_entry]]
                    .entry_date()
                    .is_none_or(|entry| entry <= date)
            {
                let idx = by_entry[next_entry];
                next_entry += 1;
                if cohort[idx].event_date().is_none_or(|event| event > date) {
                    at_risk.insert((OrderedFloat(score_of(idx).expect("scored record")), idx));
                }
            }
            while next_event < by_event.len()
                && cohort[by_event[next_event]]
                    .event_date()
                    .is_some_and(|event| event <= date)
            {
                let idx = by_event[next_event];
                next_event += 1;
                at_risk.remove(&(OrderedFloat(score_of(idx).expect("scored record")), idx));
            }
            let admitted = |entry: &&(OrderedFloat<f64>, usize)| {
                constraints.allows_at_risk(case, &cohort[entry.1])
            };
            let low = at_risk.iter().find(admitted)?.0.0;
            let high = at_risk.iter().rev().find(admitted)?.0.0;
            Some((low, high))
        })
        .collect()
}

/// Common-support trimming of a risk-set match.
///
/// A candidate is trimmed under `Control` or `Both` when its score is outside
/// the range of the scores of all cases. A case is trimmed under `Treated` or
/// `Both` when its score is strictly outside the range of the scores of the
/// records at risk on its event date. At risk means the risk-set rule and, as
/// of the event date, the alive and resident checks when the job has them.
/// Strata, the birth window and every other matching constraint are not read.
/// A score equal to an end of the range is kept. A case with nobody at risk is
/// not trimmed, since it has no range to be outside of, and is counted among
/// the cases without candidates. A record without a finite score lies outside
/// every range.
fn transition_common_support_filters<
    R: RoleIndexedRecord,
    P: RiskSetPolicy<R> + ?Sized,
    G: ConstraintGroup<R> + ?Sized,
>(
    cohort: &[R],
    case_indices: &[usize],
    policy: CommonSupport,
    scores: &ScoreTable,
    risk_set_policy: &P,
    constraints: &G,
) -> Option<TransitionCommonSupportFilters> {
    let candidate_indices = (0..cohort.len()).collect_vec();
    let case_bounds = score_bounds_for_indices(cohort, case_indices, scores)?;
    let candidate_bounds = score_bounds_for_indices(cohort, &candidate_indices, scores)?;
    let overlap = support_interval(case_bounds, candidate_bounds)?;

    let mut filters = TransitionCommonSupportFilters {
        overlap: Some(overlap),
        case_score_bounds: Some(case_bounds),
        candidate_score_bounds: Some(candidate_bounds),
        policy: Some(policy),
        ..TransitionCommonSupportFilters::default()
    };

    if matches!(policy, CommonSupport::Treated | CommonSupport::Both) {
        let ranges =
            at_risk_score_ranges(cohort, case_indices, scores, risk_set_policy, constraints);
        filters.trimmed_case_indices = case_indices
            .iter()
            .zip(ranges)
            .filter_map(|(&idx, range)| {
                let range = range?;
                (!score_inside_interval(&cohort[idx], scores, range)).then_some(idx)
            })
            .collect();
    }

    if matches!(policy, CommonSupport::Control | CommonSupport::Both) {
        filters.trimmed_candidate_indices = candidate_indices
            .into_iter()
            .filter(|idx| {
                cohort
                    .get(*idx)
                    .is_some_and(|row| !score_inside_interval(row, scores, overlap))
            })
            .collect();
    }

    Some(filters)
}

fn resolve_transition_common_support<
    R: RoleIndexedRecord,
    P: RiskSetPolicy<R> + ?Sized,
    G: ConstraintGroup<R> + ?Sized,
>(
    cohort: &[R],
    case_indices: &[usize],
    criteria: &MatchingCriteria,
    scores: Option<&ScoreTable>,
    risk_set_policy: &P,
    constraints: &G,
) -> Result<TransitionCommonSupportFilters, CommonSupportFailureReason> {
    if criteria.common_support.is_some() && scores.is_none() {
        return Err(CommonSupportFailureReason::RequiresPropensityScoreMap);
    }

    if let (Some(policy), Some(scores)) = (criteria.common_support, scores) {
        return transition_common_support_filters(
            cohort,
            case_indices,
            policy,
            scores,
            risk_set_policy,
            constraints,
        )
        .ok_or(CommonSupportFailureReason::NoOverlap);
    }

    Ok(TransitionCommonSupportFilters::default())
}

/// The ids of the rows at `indices`, in the order of the cohort.
fn trimmed_row_ids<R: RoleIndexedRecord>(cohort: &[R], indices: &HashSet<usize>) -> Vec<String> {
    cohort
        .iter()
        .enumerate()
        .filter(|(idx, _)| indices.contains(idx))
        .map(|(_, row)| row.id().to_string())
        .collect()
}

fn transition_initial_diagnostics<R: RoleIndexedRecord>(
    cohort: &[R],
    criteria: &MatchingCriteria,
    case_count: usize,
    filters: &TransitionCommonSupportFilters,
) -> MatchDiagnostics {
    MatchDiagnostics {
        total_anchors_evaluated: case_count,
        requested_estimand: criteria.estimand,
        realized_estimand: criteria.estimand,
        common_support_trimmed_anchors: filters.trimmed_case_indices.len(),
        common_support_trimmed_candidates: filters.trimmed_candidate_indices.len(),
        common_support_trimmed_ids: crate::TrimmedUnits {
            anchors: trimmed_row_ids(cohort, &filters.trimmed_case_indices),
            candidates: trimmed_row_ids(cohort, &filters.trimmed_candidate_indices),
        },
        common_support_overlap: filters.overlap,
        common_support_policy: filters.policy.or(criteria.common_support),
        common_support_anchor_score_bounds: filters.case_score_bounds,
        common_support_candidate_score_bounds: filters.candidate_score_bounds,
        ..MatchDiagnostics::default()
    }
}

fn match_with_role_indexing_with_channel_internal<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R> + ?Sized,
>(
    cohort: &[R],
    strategy: S,
    config: &RoleIndexingConfig<'_, R, D, G, P>,
) -> MatchOutcome {
    let validated = match config.criteria.validate() {
        Ok(validated) => validated,
        Err(err) => {
            return invalid_criteria_outcome(cohort.len(), InvalidCriteriaReason::from(err));
        }
    };

    let computed_order;
    let id_order = if let Some(order) = config.id_order {
        Some(order)
    } else {
        computed_order = IdOrder::of(cohort);
        computed_order.as_ref()
    };

    if config.risk_set_policy.default_at_risk_rule().is_some() {
        if let Some(reason) = invalid_record(cohort, id_order) {
            return invalid_cohort_outcome(cohort.len(), reason);
        }
    }

    if RatioPolicy::fallback_above_primary(validated.match_ratio, config.ratio_fallback) {
        return invalid_criteria_outcome(
            cohort.len(),
            InvalidCriteriaReason::FallbackRatioAbovePrimary,
        );
    }
    let ratio_policy = RatioPolicy::from_fallback(validated.match_ratio, config.ratio_fallback);
    let case_indices = eligible_case_indices(
        cohort,
        config.age_limit_years,
        config.risk_set_policy,
        id_order,
    );
    let common_support_filters = match resolve_transition_common_support(
        cohort,
        &case_indices,
        validated.as_ref(),
        config.common_support_scores,
        config.risk_set_policy,
        config.extra_constraints,
    ) {
        Ok(filters) => filters,
        Err(reason) => {
            return invalid_common_support_outcome(case_indices.len(), reason);
        }
    };

    let mut engine = MatchEngine::new(cohort, validated.as_ref(), strategy);
    if let Some(order) = id_order {
        engine.use_id_order(order);
    }

    let mut run_state = TransitionRunState::new(cohort.len());
    run_state.diagnostics = transition_initial_diagnostics(
        cohort,
        validated.as_ref(),
        case_indices.len(),
        &common_support_filters,
    );

    run_transition_case_loop(
        &TransitionLoopConfig {
            cohort,
            case_indices: &case_indices,
            config,
            validated: validated.as_ref(),
            ratio_policy: &ratio_policy,
            common_support_filters: &common_support_filters,
            id_order,
        },
        &mut engine,
        &mut run_state,
    );

    run_state.diagnostics.matched_anchors = run_state.matched_cases;
    run_state.diagnostics.pairs_selected = run_state.pairs.len();
    finalize_estimand_diagnostics(
        &mut run_state.diagnostics,
        validated.estimand,
        run_state.eligible_cases,
        config
            .distance_channel
            .caliper()
            .is_some()
            .then_some(config.distance_channel.reason()),
    );
    build_outcome(
        run_state.pairs,
        run_state.matched_cases,
        run_state.eligible_cases,
        run_state.used_controls.len(),
        run_state.diagnostics,
    )
}

use rustc_hash::FxHashSet;

struct TransitionRunState {
    pairs: Vec<MatchedPair>,
    used_controls: UsedControlsVec,
    used_unique: FxHashSet<UniqueValueId>,
    unique_interner: RapidHashMap<String, UniqueValueId>,
    matched_cases: usize,
    eligible_cases: usize,
    diagnostics: MatchDiagnostics,
}

impl TransitionRunState {
    fn new(n_controls: usize) -> Self {
        Self {
            pairs: Vec::new(),
            used_controls: UsedControlsVec::with_capacity(n_controls),
            used_unique: FxHashSet::default(),
            unique_interner: RapidHashMap::default(),
            matched_cases: 0,
            eligible_cases: 0,
            diagnostics: MatchDiagnostics::default(),
        }
    }
}

struct TransitionLoopConfig<
    'a,
    R: RoleIndexedRecord,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R> + ?Sized,
> {
    cohort: &'a [R],
    case_indices: &'a [usize],
    config: &'a RoleIndexingConfig<'a, R, D, G, P>,
    validated: &'a MatchingCriteria,
    ratio_policy: &'a RatioPolicy,
    common_support_filters: &'a TransitionCommonSupportFilters,
    id_order: Option<&'a IdOrder>,
}

fn run_transition_case_loop<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R> + ?Sized,
>(
    loop_config: &TransitionLoopConfig<'_, R, D, G, P>,
    engine: &mut MatchEngine<'_, R, S>,
    state: &mut TransitionRunState,
) {
    if let Some(mut sweep) = risk_set_sweep(loop_config, engine) {
        run_swept_case_loop(loop_config, engine, state, &mut sweep);
        return;
    }
    for case_idx in loop_config.case_indices {
        if loop_config
            .common_support_filters
            .trimmed_case_indices
            .contains(case_idx)
        {
            continue;
        }
        state.eligible_cases += 1;

        let case = &loop_config.cohort[*case_idx];
        let Some(case_event_date) = case.event_date() else {
            continue;
        };

        let mut candidates = engine.candidate_pool_for_request_and_constraint_group(
            &CandidatePoolRequest {
                case,
                used_controls: &state.used_controls,
                used_unique: &state.used_unique,
                unique_interner: &state.unique_interner,
                distance_channel: loop_config.config.distance_channel,
                use_birth_date_index: loop_config.config.use_birth_date_index,
            },
            loop_config.config.extra_constraints,
            |idx, control| {
                loop_config.config.risk_set_policy.is_control_at_risk(
                    case,
                    control,
                    case_event_date,
                ) && !loop_config
                    .common_support_filters
                    .trimmed_candidate_indices
                    .contains(&idx)
            },
            &mut state.diagnostics,
        );

        if candidates.is_empty() {
            state.diagnostics.anchors_with_no_candidates += 1;
            continue;
        }

        let Some(target_ratio) = loop_config.ratio_policy.target_ratio(candidates.len()) else {
            state.diagnostics.anchors_below_required_ratio += 1;
            continue;
        };
        if loop_config.ratio_policy.is_shortfall(candidates.len()) {
            state.diagnostics.anchors_below_required_ratio += 1;
        }
        if loop_config.ratio_policy.is_lower_tier(candidates.len()) {
            state.diagnostics.anchors_on_lower_tier += 1;
        }

        let mut local_matches = 0usize;
        for _ in 0..target_ratio {
            let Some(selected_idx) = engine.select_control(case, &mut candidates) else {
                break;
            };
            let control = &engine.controls()[selected_idx.get()];
            state.pairs.push(MatchedPair::new(case.id(), control.id()));
            local_matches += 1;

            if !loop_config.validated.allow_replacement {
                state.used_controls.insert(selected_idx);
            }
            if let Some(value) = unique_value(control, loop_config.validated) {
                let id = intern_unique(&mut state.unique_interner, value);
                state.used_unique.insert(id);
                if loop_config.validated.unique_by_key.is_some() {
                    // The case's other candidates of this value are closed.
                    candidates.retain(|&other| {
                        unique_value(&engine.controls()[other], loop_config.validated)
                            != Some(value)
                    });
                }
            }
        }

        if local_matches > 0 {
            state.matched_cases += 1;
        }
    }
}

/// The sweep of [`super::sweep`], when it gives what the scan of every
/// candidate gives: an at-risk rule given as a span of each control (the
/// default rule, or a policy's [`RiskSetPolicy::control_at_risk_span`]), the
/// birth-date index with the date distance (whose caliper is the birth window,
/// so it refuses nobody the index returns), no unique key, no common-support
/// trimming, a seeded random draw, distinct ids, and constraints that each
/// have a rule the sweep reads (see [`Constraint::status_rule`]). Every other
/// job scans.
fn risk_set_sweep<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R> + ?Sized,
>(
    loop_config: &TransitionLoopConfig<'_, R, D, G, P>,
    engine: &mut MatchEngine<'_, R, S>,
) -> Option<RiskSetSweep> {
    let config = loop_config.config;
    let filters = loop_config.common_support_filters;
    if !config.use_birth_date_index
        || loop_config.validated.unique_by_key.is_some()
        || !filters.trimmed_case_indices.is_empty()
        || !filters.trimmed_candidate_indices.is_empty()
        || engine.selector.as_random().is_none()
    {
        return None;
    }
    let mut rules = Vec::new();
    if !config.extra_constraints.status_rules(&mut rules) {
        return None;
    }
    let spans = at_risk_spans(
        loop_config.cohort,
        config.risk_set_policy,
        loop_config.case_indices,
    )?;
    RiskSetSweep::new(
        loop_config.cohort,
        &engine.candidate_index,
        &rules,
        loop_config.id_order?,
        spans,
    )
}

/// The at-risk span of every record, or `None` when the policy does not give
/// them all, or gives a case a span that holds its own event date (the scan
/// would refuse it as the case itself, which the sweep does not count).
fn at_risk_spans<R: RoleIndexedRecord, P: RiskSetPolicy<R> + ?Sized>(
    cohort: &[R],
    policy: &P,
    case_indices: &[usize],
) -> Option<Vec<AtRiskSpan>> {
    if policy.default_at_risk_rule().is_some() {
        return Some(cohort.iter().map(AtRiskSpan::of_default_rule).collect());
    }
    let spans: Vec<AtRiskSpan> = cohort
        .iter()
        .map(|record| policy.control_at_risk_span(record))
        .collect::<Option<_>>()?;
    let own_date_at_risk = case_indices.iter().any(|&idx| {
        cohort[idx]
            .event_date()
            .is_some_and(|date| spans[idx].contains(date))
    });
    (!own_date_at_risk).then_some(spans)
}

/// The case loop of [`run_transition_case_loop`] with the risk sets of a
/// sweep. The counts, the pool length and the draws are those of the scan:
/// the pool a scan builds is the eligible controls of the case's stratum and
/// window in id order, and the control at a drawn position is the eligible
/// control of that rank.
fn run_swept_case_loop<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R> + ?Sized,
>(
    loop_config: &TransitionLoopConfig<'_, R, D, G, P>,
    engine: &mut MatchEngine<'_, R, S>,
    state: &mut TransitionRunState,
    sweep: &mut RiskSetSweep,
) {
    #[cfg(test)]
    super::sweep::SWEPT_RUNS.with(|runs| runs.set(runs.get() + 1));
    let validated = loop_config.validated;
    let cohort = loop_config.cohort;
    let window_days = validated
        .typed_birth_date_window()
        .map_or(0_i64, |days| i64::from(days.get()));
    let required = &validated.required_strata;
    let drawn_reason = Constraint::<R>::reason(&ReplacementConstraint);

    // The stratum and window of each case. A stratum is counted with trees
    // when some case's window in it holds more than the list limit of
    // controls.
    let places: Vec<Option<(usize, std::ops::Range<usize>)>> = loop_config
        .case_indices
        .iter()
        .map(|&case_idx| {
            let case = &cohort[case_idx];
            case.event_date()?;
            let case_strata_values =
                (!required.is_empty()).then(|| build_strata_values(case, required));
            let stratum = engine
                .candidate_index
                .stratum_of(case_strata_values.as_deref())?;
            let window = engine
                .candidate_index
                .window(stratum, case.birth_date(), window_days)?;
            Some((stratum, window))
        })
        .collect();
    let limit = list_limit();
    let mut counted = vec![false; engine.candidate_index.strata().len()];
    for (stratum, window) in places.iter().flatten() {
        counted[*stratum] |= window.len() > limit;
    }
    sweep.count_strata(&counted);

    let mut drawn_now: Vec<usize> = Vec::new();
    for (case_idx, place) in loop_config.case_indices.iter().zip(places) {
        state.eligible_cases += 1;

        let case = &cohort[*case_idx];
        let Some(case_event_date) = case.event_date() else {
            continue;
        };
        sweep.advance(case_event_date);

        let Some((stratum, window)) = place else {
            state.diagnostics.anchors_with_no_candidates += 1;
            continue;
        };
        let (counts, mut pool) = sweep.pool(stratum, window, *case_idx);

        let exclusions = &mut state.diagnostics.exclusion_counts;
        if counts.not_at_risk > 0 {
            *exclusions
                .entry(ExclusionReason::AdditionalFilter)
                .or_insert(0) += counts.not_at_risk;
        }
        for (reason, count) in std::iter::once((drawn_reason, counts.drawn)).chain(counts.refused) {
            if count > 0 {
                *exclusions
                    .entry(ExclusionReason::Constraint(
                        ConstraintReason::from_reason_str(reason),
                    ))
                    .or_insert(0) += count;
            }
        }

        let eligible = counts.eligible;
        if eligible == 0 {
            state.diagnostics.anchors_with_no_candidates += 1;
            continue;
        }

        let Some(target_ratio) = loop_config.ratio_policy.target_ratio(eligible) else {
            state.diagnostics.anchors_below_required_ratio += 1;
            continue;
        };
        if loop_config.ratio_policy.is_shortfall(eligible) {
            state.diagnostics.anchors_below_required_ratio += 1;
        }
        if loop_config.ratio_policy.is_lower_tier(eligible) {
            state.diagnostics.anchors_on_lower_tier += 1;
        }

        let mut local_matches = 0usize;
        drawn_now.clear();
        for _ in 0..target_ratio {
            let Some(position) = engine
                .selector
                .as_random()
                .and_then(|random| random.draw(pool.len()))
            else {
                break;
            };
            let selected_idx = sweep.take(stratum, &mut pool, position);
            let control = &cohort[selected_idx];
            state.pairs.push(MatchedPair::new(case.id(), control.id()));
            local_matches += 1;

            if !validated.allow_replacement {
                state.used_controls.insert(ControlIdx::new(selected_idx));
                drawn_now.push(selected_idx);
            }
            if let Some(value) = unique_value(control, validated) {
                let id = intern_unique(&mut state.unique_interner, value);
                state.used_unique.insert(id);
            }
        }
        for &idx in &drawn_now {
            sweep.mark_drawn(idx);
        }

        if local_matches > 0 {
            state.matched_cases += 1;
        }
    }
}

/// The first record, in cohort order, that the default policy refuses: an id
/// that occurs twice, or a case whose event is before its birth. A record that
/// is both is reported for its id.
fn invalid_record<R: RoleIndexedRecord>(
    cohort: &[R],
    id_order: Option<&IdOrder>,
) -> Option<ExclusionReason> {
    let Some(order) = id_order else {
        return invalid_record_by_hashing(cohort);
    };
    let repeat = order.first_repeat(cohort);
    let before_birth = cohort.iter().position(|row| {
        row.event_date()
            .is_some_and(|event| event < row.birth_date())
    });
    match (repeat, before_birth) {
        (Some(repeat), Some(before)) if before < repeat => Some(ExclusionReason::EventBeforeBirth(
            cohort[before].id().to_owned(),
        )),
        (Some(repeat), _) => Some(ExclusionReason::DuplicateId(cohort[repeat].id().to_owned())),
        (None, Some(before)) => Some(ExclusionReason::EventBeforeBirth(
            cohort[before].id().to_owned(),
        )),
        (None, None) => None,
    }
}

/// [`invalid_record`] by a set of the ids seen so far, for a cohort too large
/// for an [`IdOrder`].
fn invalid_record_by_hashing<R: RoleIndexedRecord>(cohort: &[R]) -> Option<ExclusionReason> {
    let mut seen: HashSet<&str> = HashSet::with_capacity(cohort.len());
    for row in cohort {
        if !seen.insert(row.id()) {
            return Some(ExclusionReason::DuplicateId(row.id().to_owned()));
        }
        if row
            .event_date()
            .is_some_and(|event| event < row.birth_date())
        {
            return Some(ExclusionReason::EventBeforeBirth(row.id().to_owned()));
        }
    }
    None
}

/// The cases, by event date and then by id, records with equal ids in cohort
/// order.
fn eligible_case_indices<R: RoleIndexedRecord, P: RiskSetPolicy<R> + ?Sized>(
    cohort: &[R],
    age_limit_years: u8,
    risk_set_policy: &P,
    id_order: Option<&IdOrder>,
) -> Vec<usize> {
    let cases = cohort.iter().enumerate().filter_map(|(idx, row)| {
        if risk_set_policy.is_case_eligible(row, age_limit_years) {
            Some(idx)
        } else {
            None
        }
    });
    let Some(order) = id_order else {
        return cases_by_comparing_ids(cohort, cases);
    };
    // A rank orders records by id and then by index, so the order of
    // `(event date, rank)` is that of a stable sort by `(event date, id)`.
    let mut keyed: Vec<(Option<NaiveDate>, u32, usize)> = cases
        .map(|idx| (cohort[idx].event_date(), order.rank(idx), idx))
        .collect();
    keyed.sort_unstable_by_key(|&(event, rank, _)| (event, rank));
    keyed.into_iter().map(|(_, _, idx)| idx).collect()
}

/// [`eligible_case_indices`] by comparing ids, for a cohort too large for an
/// [`IdOrder`].
fn cases_by_comparing_ids<R: RoleIndexedRecord>(
    cohort: &[R],
    cases: impl Iterator<Item = usize>,
) -> Vec<usize> {
    cases
        .sorted_by(|left_idx, right_idx| {
            let left = &cohort[*left_idx];
            let right = &cohort[*right_idx];
            left.event_date()
                .cmp(&right.event_date())
                .then_with(|| left.id().cmp(right.id()))
        })
        .collect_vec()
}

/// `NaiveDate::num_days_from_ce` of 1970-01-01.
const UNIX_EPOCH_DAYS_FROM_CE: i32 = 719_163;

fn is_case_before_age_limit<R: RoleIndexedRecord>(row: &R, age_limit_years: u8) -> bool {
    let Some(event_date) = row.event_date() else {
        return false;
    };
    let day = |date: NaiveDate| i64::from(date.num_days_from_ce() - UNIX_EPOCH_DAYS_FROM_CE);
    // An event before birth is below any age limit.
    age_in_years(
        day(row.birth_date()),
        day(event_date),
        AgeRule::CivilYears(LeapBirthday::February28),
    )
    .is_none_or(|age| age < u32::from(age_limit_years))
}

fn default_control_is_at_risk<R: RoleIndexedRecord>(
    control: &R,
    case_event_date: NaiveDate,
) -> bool {
    control
        .entry_date()
        .is_none_or(|entry| entry <= case_event_date)
        && control
            .event_date()
            .is_none_or(|event_date| event_date > case_event_date)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;
    use crate::matching::{DeterministicSelection, DistanceConfig, RandomSelection};
    use crate::types::{AgeLimitYears, BaseRecord, Estimand, MatchRatio, RoleTransitionRecord};
    use chrono::{Months, NaiveDate};

    #[test]
    fn age_limit_agrees_with_adding_whole_years_to_the_birth_date() {
        for birth in [
            NaiveDate::from_ymd_opt(2000, 2, 29).unwrap(),
            NaiveDate::from_ymd_opt(2001, 2, 28).unwrap(),
            NaiveDate::from_ymd_opt(2003, 12, 31).unwrap(),
            NaiveDate::from_ymd_opt(2004, 3, 1).unwrap(),
        ] {
            for limit in [1_u8, 4, 6, 18] {
                let limit_date = birth
                    .checked_add_months(Months::new(u32::from(limit) * 12))
                    .unwrap();
                let first = birth - chrono::Duration::days(400);
                for offset in 0..9000 {
                    let event = first + chrono::Duration::days(offset);
                    let row =
                        RoleTransitionRecord::from_record(BaseRecord::new("a", birth), Some(event));
                    assert_eq!(
                        is_case_before_age_limit(&row, limit),
                        event < limit_date,
                        "{birth} {event} {limit}"
                    );
                }
            }
        }
    }

    fn ratio(value: usize) -> MatchRatio {
        MatchRatio::new(value).expect("non-zero ratio")
    }

    fn age_limit(value: u8) -> AgeLimitYears {
        AgeLimitYears::new(value).expect("positive age limit")
    }

    fn run_transition<R, S, G, P>(
        cohort: &[R],
        criteria: &MatchingCriteria,
        options: &RoleTransitionOptions,
        strategy: S,
        constraints: &G,
        distance_config: Option<&DistanceConfig>,
        risk_set_policy: &P,
    ) -> MatchOutcome
    where
        R: RoleIndexedRecord,
        S: SelectionStrategy<R> + Clone + Send + Sync,
        G: ConstraintGroup<R> + ?Sized,
        P: RiskSetPolicy<R>,
    {
        match_transition(
            cohort,
            TransitionMatchRequest {
                criteria,
                options,
                strategy,
                constraints,
                distance_config,
                risk_set_policy,
            },
        )
    }

    #[derive(Clone)]
    struct ResidencyRecord {
        base: BaseRecord,
        resident_until: Option<NaiveDate>,
    }

    impl crate::MatchingRecord for ResidencyRecord {
        crate::delegate_matching_record!(base);
    }

    impl ResidencyRecord {
        fn new(id: &str, birth_date: NaiveDate, resident_until: Option<NaiveDate>) -> Self {
            Self {
                base: BaseRecord::new(id, birth_date),
                resident_until,
            }
        }
    }

    struct ResidencyPolicy;

    impl RiskSetPolicy<RoleTransitionRecord<ResidencyRecord>> for ResidencyPolicy {
        fn is_case_eligible(
            &self,
            row: &RoleTransitionRecord<ResidencyRecord>,
            age_limit_years: u8,
        ) -> bool {
            is_case_before_age_limit(row, age_limit_years)
                && row.record.resident_until.is_none_or(|until| {
                    row.event_date()
                        .is_some_and(|event_date| until >= event_date)
                })
        }

        fn is_control_at_risk(
            &self,
            case: &RoleTransitionRecord<ResidencyRecord>,
            control: &RoleTransitionRecord<ResidencyRecord>,
            case_event_date: NaiveDate,
        ) -> bool {
            default_control_is_at_risk(control, case_event_date)
                && control
                    .record
                    .resident_until
                    .is_none_or(|until| until >= case_event_date)
                && case.record.base.id != control.record.base.id
        }
    }

    #[test]
    fn transition_matching_is_generic_over_record_shape() {
        let criteria = MatchingCriteria::default();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };

        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            None,
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
    }

    #[test]
    fn transition_works_with_base_record_default_type() {
        let criteria = MatchingCriteria::default();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };

        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            RandomSelection::seeded(9),
            &(),
            None,
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
    }

    #[test]
    fn transition_policy_filters_non_resident_candidates() {
        let criteria = MatchingCriteria::default();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };
        let cohort = vec![
            RoleTransitionRecord::from_record(
                ResidencyRecord::new("case", date(2010, 1, 1), None),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(
                ResidencyRecord::new(
                    "candidate_not_resident",
                    date(2010, 1, 2),
                    Some(date(2013, 12, 31)),
                ),
                None,
            ),
            RoleTransitionRecord::from_record(
                ResidencyRecord::new("candidate_resident", date(2010, 1, 3), None),
                None,
            ),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            None,
            &ResidencyPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
        assert_eq!(outcome.pairs[0].comparator_id(), "candidate_resident");
    }

    #[test]
    fn transition_policy_filters_non_resident_cases() {
        let criteria = MatchingCriteria::builder().allow_replacement(true).build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };
        let cohort = vec![
            RoleTransitionRecord::from_record(
                ResidencyRecord::new(
                    "ineligible_case",
                    date(2010, 1, 1),
                    Some(date(2013, 12, 31)),
                ),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(
                ResidencyRecord::new("eligible_case", date(2010, 1, 2), None),
                Some(date(2014, 2, 1)),
            ),
            RoleTransitionRecord::from_record(
                ResidencyRecord::new("candidate", date(2010, 1, 3), None),
                None,
            ),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            None,
            &ResidencyPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
        assert_eq!(outcome.pairs[0].anchor_id(), "eligible_case");
    }

    #[test]
    fn transition_supports_reusable_distance_channels() {
        let criteria = MatchingCriteria::builder()
            .birth_date_window_days(0)
            .build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };
        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 7, 1)), None),
        ];

        let scores: RapidHashMap<String, f64> = [
            ("case".to_string(), 0.20_f64),
            ("candidate".to_string(), 0.24_f64),
        ]
        .into_iter()
        .collect();
        let config = DistanceConfig::propensity_score_map(
            scores,
            Some(crate::types::DistanceCaliper::new(0.05).expect("valid positive caliper")),
        )
        .with_reason("ps_transition");
        let outcome_with_config = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            Some(&config),
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome_with_config.matched_cases, 1);
        assert_eq!(outcome_with_config.pairs.len(), 1);
    }

    #[test]
    fn transition_default_ratio_behavior_allows_partial_matching() {
        let criteria = MatchingCriteria::builder()
            .birth_date_window_days(10)
            .match_ratio(2)
            .build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: Vec::new(),
        };

        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            None,
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
        assert_eq!(outcome.diagnostics.anchors_below_required_ratio, 1);
    }

    #[test]
    fn transition_explicit_ratio_fallback_is_strict() {
        let criteria = MatchingCriteria::builder()
            .birth_date_window_days(10)
            .match_ratio(2)
            .build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(2)],
        };

        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            None,
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome.matched_cases, 0);
        assert!(outcome.pairs.is_empty(), "{:?}", outcome.pairs);
        assert_eq!(outcome.diagnostics.anchors_below_required_ratio, 1);
    }

    #[test]
    fn transition_common_support_requires_propensity_score_map() {
        let criteria = MatchingCriteria::builder()
            .common_support(CommonSupport::Both)
            .build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };
        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            RandomSelection::seeded(12),
            &(),
            None,
            &DefaultRiskSetPolicy,
        );
        assert!(outcome.pairs.is_empty(), "{:?}", outcome.pairs);
        assert_eq!(
            outcome
                .diagnostics
                .exclusion_counts
                .get(&crate::types::ExclusionReason::CommonSupportFailure(
                    crate::types::CommonSupportFailureReason::RequiresPropensityScoreMap,
                ))
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn transition_propensity_common_support_updates_estimand_diagnostics() {
        let criteria = MatchingCriteria::builder()
            .birth_date_window_days(0)
            .common_support(CommonSupport::Control)
            .build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };
        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(
                BaseRecord::new("candidate_overlap", date(2010, 1, 2)),
                None,
            ),
            RoleTransitionRecord::from_record(
                BaseRecord::new("candidate_trimmed", date(2010, 1, 3)),
                None,
            ),
        ];
        let scores: RapidHashMap<String, f64> = [
            ("case".to_string(), 0.35_f64),
            ("candidate_overlap".to_string(), 0.35_f64),
            ("candidate_trimmed".to_string(), 0.90_f64),
        ]
        .into_iter()
        .collect();
        let config = DistanceConfig::propensity_score_map(
            scores,
            Some(crate::types::DistanceCaliper::new(1.0).expect("valid positive caliper")),
        );

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            Some(&config),
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.diagnostics.common_support_trimmed_candidates, 1);
        assert_eq!(
            outcome.diagnostics.common_support_policy,
            Some(CommonSupport::Control)
        );
        assert_eq!(
            outcome.diagnostics.common_support_anchor_score_bounds,
            Some((0.35, 0.35))
        );
        assert_eq!(
            outcome.diagnostics.common_support_candidate_score_bounds,
            Some((0.35, 0.90))
        );
        assert_eq!(outcome.diagnostics.requested_estimand, Estimand::Att);
        assert_eq!(outcome.diagnostics.realized_estimand, Estimand::Atm);
    }

    /// Records with repeated ids, cases before their birth and tied event
    /// dates, in an order unrelated to the ids.
    fn tangled_cohort(seed: u64, n: usize) -> Vec<RoleTransitionRecord<BaseRecord>> {
        let mut state = seed;
        let mut next = |bound: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };
        (0..n)
            .map(|_| {
                let birth = date(2000, 1, 1) + chrono::Days::new(next(400));
                let id = format!("{}", next(n as u64 * 4));
                let event = match next(10) {
                    0 => Some(birth - chrono::Days::new(1 + next(30))),
                    1..=4 => Some(date(2001, 1, 1) + chrono::Days::new(next(5) * 30)),
                    _ => None,
                };
                RoleTransitionRecord::from_record(BaseRecord::new(id, birth), event)
            })
            .collect()
    }

    /// The validation and the order of the cases read from the job's id
    /// order equal those found by hashing and comparing the ids.
    #[test]
    fn the_id_order_validates_and_orders_cases_as_comparing_ids_does() {
        let mut reported = 0;
        for seed in 0..200_u64 {
            let n = [1, 2, 5, 30, 200][usize::try_from(seed % 5).expect("small")];
            let mut cohort = tangled_cohort(seed, n);
            if seed % 3 == 0 {
                // Without repeats or cases before birth, so that a case
                // order exists to compare with nothing reported.
                for (i, row) in cohort.iter_mut().enumerate() {
                    row.record.id = format!("u{}", (i * 7919) % 100_003);
                    if row.event_date().is_some_and(|e| e < row.record.birth_date) {
                        row.transition_date = None;
                    }
                }
            }
            let order = IdOrder::of(&cohort).expect("fits");
            let by_order = invalid_record(&cohort, Some(&order));
            assert_eq!(by_order, invalid_record_by_hashing(&cohort), "seed {seed}");
            reported += usize::from(by_order.is_some());
            for limit in [1_u8, 6] {
                assert_eq!(
                    eligible_case_indices(&cohort, limit, &DefaultRiskSetPolicy, Some(&order)),
                    cases_by_comparing_ids(
                        &cohort,
                        (0..cohort.len()).filter(|&i| is_case_before_age_limit(&cohort[i], limit))
                    ),
                    "seed {seed}"
                );
            }
        }
        assert!(reported > 50, "{reported}");
    }
}
