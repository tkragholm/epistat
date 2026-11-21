use crate::matching::constraints::BirthWindowConstraint;
use crate::matching::engine::{StandardMatchRequest, match_standard};
use crate::matching::{
    Assignment, Chained, Constraint, ConstraintGroup, DistanceConfig, MatchOrder, MatchingRecord,
    NearestSelection, RandomSelection, ResidentAtIndexRecord, RoleIndexedRecord, Rounds,
    SelectionStrategy,
};
use crate::role_transition::{
    DefaultRiskSetPolicy, IdOrder, RiskSetPolicy, TransitionMatchRequest, match_transition_ordered,
};
use crate::types::{
    AgeLimitYears, BirthDateWindowDays, Estimand, MatchOutcome, MatchRatio, MatchingCriteria,
    RoleTransitionOptions,
};
use std::marker::PhantomData;

/// Marker and data for standard matching.
pub struct StandardMode<'a, R: MatchingRecord> {
    anchors: &'a [R],
    candidates: &'a [R],
    ratio_fallback: Vec<MatchRatio>,
    distance_config: Option<DistanceConfig>,
    birth_window_set: bool,
}

/// Marker and data for role-transition matching.
pub struct TransitionMode<'a, R: RoleIndexedRecord> {
    cohort: &'a [R],
    options: RoleTransitionOptions,
    distance_config: Option<DistanceConfig>,
    birth_window_set: bool,
    id_order: Option<&'a IdOrder>,
}

/// A job mode that remembers whether the birth window was set by the caller.
///
/// The window then holds under a distance configuration too. Without a
/// configuration the window is the matching distance itself.
pub trait BirthWindowMode {
    #[doc(hidden)]
    fn note_birth_window(&mut self);
}

impl<R: MatchingRecord> BirthWindowMode for StandardMode<'_, R> {
    fn note_birth_window(&mut self) {
        self.birth_window_set = true;
    }
}

impl<R: RoleIndexedRecord> BirthWindowMode for TransitionMode<'_, R> {
    fn note_birth_window(&mut self) {
        self.birth_window_set = true;
    }
}

type AliveSelector<R> = fn(&R) -> Option<chrono::NaiveDate>;
type ResidentSelector<R> = fn(&R, chrono::NaiveDate) -> bool;
type WithAliveConstraint<Mode, R, S, G, P> =
    MatchJob<Mode, R, S, (G, crate::constraints::MustBeAlive<AliveSelector<R>>), P>;
type WithResidentConstraint<Mode, R, S, G, P> =
    MatchJob<Mode, R, S, (G, crate::constraints::MustBeResident<ResidentSelector<R>>), P>;

/// Unified entrypoint for matching jobs.
///
/// `MatchJob` is a builder for configuring and running cohort matching. A
/// typestate keeps standard matching and role-transition matching to their own
/// methods.
///
/// ### Example (Standard)
/// ```rust,ignore
/// use epistat_matching::prelude::*;
///
/// let job = MatchJob::new_standard(&anchors, &candidates, 42)
///     .with_ratio(MatchRatio::new(4).expect("non-zero ratio"))
///     .with_birth_window(BirthDateWindowDays::new(30).expect("non-negative birth window"))
///     .with_gender_match();
///
/// let result = job.run();
/// ```
///
/// ### Example (Role-Transition)
/// ```rust,ignore
/// use epistat_matching::prelude::*;
///
/// let job = MatchJob::new_transition(&records, 42)
///     .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
///     .with_age_limit(AgeLimitYears::new(6).expect("positive age limit"))
///     .with_alive_check();
///
/// let result = job.run();
/// ```
pub struct MatchJob<Mode, R, S, G = (), P = DefaultRiskSetPolicy> {
    mode: Mode,
    criteria: MatchingCriteria,
    strategy: S,
    constraints: G,
    risk_set_policy: P,
    record: PhantomData<R>,
}

impl<'a, R: RoleIndexedRecord>
    MatchJob<TransitionMode<'a, R>, R, RandomSelection, (), DefaultRiskSetPolicy>
{
    /// Create a new matching job for role-transition risk-set matching.
    #[must_use]
    pub fn new_transition(cohort: &'a [R], seed: u64) -> Self {
        Self {
            mode: TransitionMode {
                cohort,
                options: RoleTransitionOptions::default(),
                distance_config: None,
                birth_window_set: false,
                id_order: None,
            },
            criteria: MatchingCriteria::default(),
            strategy: RandomSelection::seeded(seed),
            constraints: (),
            risk_set_policy: DefaultRiskSetPolicy,
            record: PhantomData,
        }
    }
}

impl<'a, R: MatchingRecord>
    MatchJob<StandardMode<'a, R>, R, RandomSelection, (), DefaultRiskSetPolicy>
{
    /// Create a new matching job for standard anchor-to-candidate matching.
    #[must_use]
    pub fn new_standard(anchors: &'a [R], candidates: &'a [R], seed: u64) -> Self {
        Self {
            mode: StandardMode {
                anchors,
                candidates,
                ratio_fallback: Vec::new(),
                distance_config: None,
                birth_window_set: false,
            },
            criteria: MatchingCriteria::default(),
            strategy: RandomSelection::seeded(seed),
            constraints: (),
            risk_set_policy: DefaultRiskSetPolicy,
            record: PhantomData,
        }
    }
}

impl<Mode, R: MatchingRecord, S, G, P> MatchJob<Mode, R, S, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Set the matching ratio (number of controls per case).
    ///
    /// Without replacement, anchors take their controls one anchor at a time
    /// by default: each takes all its controls before the next starts. When
    /// controls run short this leaves the last anchors with fewer controls or
    /// none. `matchit()` gives every anchor one control before any gets a
    /// second, which leaves every anchor with at least one for as long as
    /// controls last. That order is the default in nearest mode
    /// (`with_nearest`) and can be chosen with
    /// `with_round_assignment`. An anchor that finds fewer than the ratio gets
    /// what is available.
    #[must_use]
    pub const fn with_ratio(mut self, ratio: MatchRatio) -> Self {
        self.criteria.match_ratio = ratio.get();
        self
    }

    /// Set the birth date caliper window in days.
    ///
    /// Candidates must have a birth date within `+/- days` of the case.
    ///
    /// The window also holds when a distance configuration is set, where it
    /// is applied as a constraint on top of the distance. A job that never
    /// calls this has the default window of 30 days only when it matches on
    /// the birth date, that is, with no distance configuration.
    #[must_use]
    pub fn with_birth_window(mut self, days: BirthDateWindowDays) -> Self
    where
        Mode: BirthWindowMode,
    {
        self.criteria.birth_date_window_days = days.get();
        self.mode.note_birth_window();
        self
    }

    /// Add a required exact-match strata key.
    ///
    /// The engine will only consider candidates that have the same value for this
    /// strata key as the case. A record with no value for the key has the
    /// value "absent", and matches another record with none.
    #[must_use]
    pub fn with_exact_match(mut self, key: impl Into<String>) -> Self {
        self.criteria.required_strata.push(key.into());
        self
    }

    /// Add multiple required exact-match strata keys.
    ///
    /// Equivalent to calling [`Self::with_exact_match`] once per key.
    #[must_use]
    pub fn with_exact_matches<I, K>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = K>,
        K: Into<String>,
    {
        self.criteria
            .required_strata
            .extend(keys.into_iter().map(Into::into));
        self
    }

    /// Enable or disable replacement (candidate reuse).
    ///
    /// If `true`, a candidate can be matched to multiple cases.
    #[must_use]
    pub const fn with_replacement(mut self, allow: bool) -> Self {
        self.criteria.allow_replacement = allow;
        self
    }

    /// Set optional uniqueness by a strata key.
    ///
    /// When set, controls sharing the same value for this key are treated as mutually exclusive.
    #[must_use]
    pub fn with_unique_by_key(mut self, key: Option<String>) -> Self {
        self.criteria.unique_by_key = key;
        self
    }

    /// Set the target estimand (e.g., ATT, ATE).
    #[must_use]
    pub const fn with_estimand(mut self, estimand: Estimand) -> Self {
        self.criteria.estimand = estimand;
        self
    }

    /// Set a custom selection strategy.
    #[must_use]
    pub fn with_strategy<S2>(self, strategy: S2) -> MatchJob<Mode, R, S2, G, P> {
        MatchJob::<Mode, R, S2, G, P> {
            mode: self.mode,
            criteria: self.criteria,
            strategy,
            constraints: self.constraints,
            risk_set_policy: self.risk_set_policy,
            record: PhantomData,
        }
    }

    /// Append a whole constraint group, checked after everything already added.
    ///
    /// [`Self::with_constraint`] composes into a tuple, so the number of
    /// constraints is fixed at compile time. This takes a
    /// [`ConstraintGroup`], which `Vec<C>` implements, so a caller can decide
    /// at run time how many constraints there are, one per field named in a
    /// configuration file, say. See [`Self::with_constraints`] for the common
    /// case.
    #[must_use]
    pub fn with_constraint_group<G2>(self, group: G2) -> MatchJob<Mode, R, S, Chained<G, G2>, P>
    where
        G2: ConstraintGroup<R>,
    {
        MatchJob::<Mode, R, S, Chained<G, G2>, P> {
            mode: self.mode,
            criteria: self.criteria,
            strategy: self.strategy,
            constraints: Chained {
                first: self.constraints,
                second: group,
            },
            risk_set_policy: self.risk_set_policy,
            record: PhantomData,
        }
    }

    /// Append however many constraints of one type the caller has.
    ///
    /// The case this exists for: a study whose matching rules are configuration
    /// rather than code, holding a list of `(field, window)` pairs.
    /// [`crate::constraints::caliper_on_field`] returns the same type for every
    /// field, so the calipers collect into one `Vec` and arrive here.
    ///
    /// An empty list adds nothing and refuses nothing.
    ///
    /// ```rust,ignore
    /// let calipers: Vec<_> = cfg
    ///     .calipers
    ///     .iter()
    ///     .map(|(field, window)| caliper_on_field(field, *window))
    ///     .collect();
    ///
    /// let outcome = MatchJob::new_transition(&cohort, seed)
    ///     .with_exact_matches(["sex", "municipality"])
    ///     .with_constraints(calipers)
    ///     .run();
    /// ```
    #[must_use]
    pub fn with_constraints<C>(
        self,
        constraints: Vec<C>,
    ) -> MatchJob<Mode, R, S, Chained<G, Vec<C>>, P>
    where
        C: Constraint<R>,
    {
        self.with_constraint_group(constraints)
    }

    /// Add a matching constraint.
    #[must_use]
    pub fn with_constraint<C>(self, constraint: C) -> MatchJob<Mode, R, S, (G, C), P>
    where
        C: Constraint<R>,
    {
        MatchJob::<Mode, R, S, (G, C), P> {
            mode: self.mode,
            criteria: self.criteria,
            strategy: self.strategy,
            constraints: (self.constraints, constraint),
            risk_set_policy: self.risk_set_policy,
            record: PhantomData,
        }
    }

    /// Require matched units to differ on a key of the strata map
    /// (`antiexact`). See [`AntiExact`](crate::constraints::AntiExact).
    #[must_use]
    pub fn with_antiexact(
        self,
        key: impl Into<String>,
    ) -> MatchJob<Mode, R, S, (G, crate::constraints::AntiExact), P> {
        self.with_constraint(crate::constraints::AntiExact::on_key(key))
    }

    /// Add a gender match constraint.
    ///
    /// Adds a [`GenderMatch`](crate::constraints::GenderMatch) constraint on
    /// the "sex" strata key.
    #[must_use]
    pub fn with_gender_match(
        self,
    ) -> MatchJob<Mode, R, S, (G, crate::constraints::GenderMatch), P> {
        use crate::constraints::GenderMatch;
        self.with_constraint(GenderMatch::on_key("sex"))
    }
}

impl<'a, R: MatchingRecord, S, G, P> MatchJob<StandardMode<'a, R>, R, S, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Give every anchor a first control before any gets a second, and so on
    /// up to the ratio, as `matchit()` does.
    ///
    /// This applies to standard matching without replacement at a ratio above
    /// 1, with the strategy still choosing each control. It cannot be combined
    /// with ratio fallbacks, and a job that does so returns an
    /// [`InvalidCriteriaReason::NearestOptions`](crate::InvalidCriteriaReason)
    /// outcome. Each round builds every anchor's candidate pool again, so the
    /// cost is the ratio times that of the default order.
    #[must_use]
    pub fn with_round_assignment(self) -> MatchJob<StandardMode<'a, R>, R, Rounds<S>, G, P> {
        MatchJob {
            mode: self.mode,
            criteria: self.criteria,
            strategy: Rounds(self.strategy),
            constraints: self.constraints,
            risk_set_policy: self.risk_set_policy,
            record: PhantomData,
        }
    }
}

impl<'a, R: MatchingRecord, G, P> MatchJob<StandardMode<'a, R>, R, RandomSelection, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Rank candidates by the distance instead of choosing among them, as
    /// `matchit(method = "nearest")` does. See [`NearestSelection`] for the
    /// rules. The job's seed is used for the random order of anchors.
    ///
    /// The distance is the one set with [`Self::with_distance_config`], or the
    /// birth date. Anchors take their controls in rounds unless
    /// [`Self::with_assignment`] says otherwise.
    #[must_use]
    pub fn with_nearest(self) -> MatchJob<StandardMode<'a, R>, R, NearestSelection, G, P> {
        let seed = self.strategy.seed();
        self.with_strategy(NearestSelection::new(seed))
    }
}

impl<R: MatchingRecord, G, P> MatchJob<StandardMode<'_, R>, R, NearestSelection, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Set the order in which anchors are matched (`m.order`).
    #[must_use]
    pub fn with_m_order(mut self, order: MatchOrder) -> Self {
        self.strategy = self.strategy.with_order(order);
        self
    }

    /// Choose rounds or sequential assignment.
    #[must_use]
    pub fn with_assignment(mut self, assignment: Assignment) -> Self {
        self.strategy = self.strategy.with_assignment(assignment);
        self
    }

    /// Read the distance caliper in standard deviations of the distance
    /// (`std.caliper`).
    #[must_use]
    pub fn with_std_caliper(mut self, std_caliper: bool) -> Self {
        self.strategy = self.strategy.with_std_caliper(std_caliper);
        self
    }

    /// Match a variable number of controls to each anchor, between the
    /// ratio's minimum and maximum, with the ratio's average as the mean
    /// (`ratio`, `min.controls` and `max.controls`). The job's own ratio is
    /// not used. See [`VariableRatio`](crate::VariableRatio). It cannot be
    /// combined with ratio fallbacks.
    #[must_use]
    pub fn with_variable_ratio(mut self, ratio: crate::VariableRatio) -> Self {
        self.strategy = self.strategy.with_variable_ratio(ratio);
        self
    }

    /// Rank candidates by this distance and use the job's distance as a
    /// caliper and for the order of anchors (`mahvars`).
    #[must_use]
    pub fn with_mahvars(mut self, ranking: DistanceConfig) -> Self {
        self.strategy = self.strategy.with_mahvars(ranking);
        self
    }
}

impl<Mode, R: RoleIndexedRecord, S, G, P> MatchJob<Mode, R, S, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Add a constraint ensuring controls are alive at each case's index date.
    ///
    /// This is equivalent to:
    /// `with_constraint(MustBeAlive::at_index_date(|record: &R| record.death_date()))`.
    ///
    /// Matching behavior for each case/control pair:
    /// - If the case has no `event_date`, the pair is allowed by this constraint.
    /// - If the control has no `death_date`, the pair is allowed by this constraint.
    /// - Otherwise, the pair is allowed only when `control.death_date > case.event_date`.
    #[must_use]
    pub fn with_alive_check(self) -> WithAliveConstraint<Mode, R, S, G, P> {
        use crate::constraints::MustBeAlive;
        fn death_selector<R: RoleIndexedRecord>(record: &R) -> Option<chrono::NaiveDate> {
            record.death_date()
        }
        self.with_constraint(MustBeAlive::at_index_date(death_selector::<R>))
    }

    /// Add a custom alive-at-index constraint.
    ///
    /// Use this when death dates are provided by an external map.
    #[must_use]
    pub fn with_alive_check_by<F>(
        self,
        selector: F,
    ) -> MatchJob<Mode, R, S, (G, crate::constraints::MustBeAlive<F>), P>
    where
        F: Fn(&R) -> Option<chrono::NaiveDate> + Send + Sync,
    {
        use crate::constraints::MustBeAlive;
        self.with_constraint(MustBeAlive::at_index_date(selector))
    }

    /// Add a constraint ensuring controls are resident at each case's index date.
    ///
    /// This uses [`ResidentAtIndexRecord::is_resident_at`] as the residency source.
    #[must_use]
    pub fn with_resident_check(self) -> WithResidentConstraint<Mode, R, S, G, P>
    where
        R: ResidentAtIndexRecord,
    {
        use crate::constraints::MustBeResident;
        fn resident_selector<R: ResidentAtIndexRecord>(
            record: &R,
            index_date: chrono::NaiveDate,
        ) -> bool {
            record.is_resident_at(index_date)
        }
        self.with_constraint(MustBeResident::at_index_date(resident_selector::<R>))
    }

    /// Add a residency-at-index constraint with residence given as periods;
    /// see [`MustBeResidentInSpells`](crate::constraints::MustBeResidentInSpells).
    ///
    /// Use this when residence comes from a register of migrations. It refuses
    /// what [`Self::with_resident_check_by`] with the same periods refuses, and
    /// lets risk-set matching take the sweep.
    #[must_use]
    pub fn with_resident_spells<'s, F>(
        self,
        spells: F,
    ) -> MatchJob<Mode, R, S, (G, crate::constraints::MustBeResidentInSpells<'s, F>), P>
    where
        F: Fn(&R) -> Option<&'s [crate::constraints::ResidenceSpell]> + Send + Sync,
    {
        self.with_constraint(crate::constraints::MustBeResidentInSpells::at_index_date(
            spells,
        ))
    }

    /// Add a custom residency-at-index constraint.
    ///
    /// Use this when residency is derived from study-specific fields.
    #[must_use]
    pub fn with_resident_check_by<F>(
        self,
        check: F,
    ) -> MatchJob<Mode, R, S, (G, crate::constraints::MustBeResident<F>), P>
    where
        F: Fn(&R, chrono::NaiveDate) -> bool + Send + Sync,
    {
        use crate::constraints::MustBeResident;
        self.with_constraint(MustBeResident::at_index_date(check))
    }
}

impl<'a, R: RoleIndexedRecord, S, G, P> MatchJob<TransitionMode<'a, R>, R, S, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Set the transition age limit in years (for role-transition matching).
    #[must_use]
    pub const fn with_age_limit(mut self, years: AgeLimitYears) -> Self {
        self.mode.options.transition_age_limit_years = years;
        self
    }

    /// Set optional descending fallback ratios for role-transition matching.
    #[must_use]
    pub fn with_ratio_fallback(mut self, fallback: Vec<MatchRatio>) -> Self {
        self.mode.options.ratio_fallback = fallback;
        self
    }

    /// Set a custom distance metric configuration.
    #[must_use]
    pub fn with_distance_config(mut self, config: DistanceConfig) -> Self {
        self.mode.distance_config = Some(config);
        self
    }

    /// Trim cases and candidates outside the common support of the scores
    /// of the distance configuration. A case is judged against the people at
    /// risk on its event date; see [`CommonSupport`](crate::CommonSupport).
    /// Needs a score distance; without one the job returns an invalid outcome.
    #[must_use]
    pub const fn with_common_support(mut self, policy: Option<crate::CommonSupport>) -> Self {
        self.criteria.common_support = policy;
        self
    }

    /// Use `order`, the order of the cohort's ids, instead of sorting the
    /// ids again.
    #[must_use]
    pub(crate) const fn with_id_order(mut self, order: Option<&'a IdOrder>) -> Self {
        self.mode.id_order = order;
        self
    }

    /// Set a custom risk-set policy.
    #[must_use]
    pub fn with_risk_set_policy<P2>(
        self,
        risk_set_policy: P2,
    ) -> MatchJob<TransitionMode<'a, R>, R, S, G, P2> {
        MatchJob::<TransitionMode<'a, R>, R, S, G, P2> {
            mode: self.mode,
            criteria: self.criteria,
            strategy: self.strategy,
            constraints: self.constraints,
            risk_set_policy,
            record: PhantomData,
        }
    }

    /// Execute role-transition matching.
    pub fn run(self) -> MatchOutcome
    where
        S: SelectionStrategy<R> + Clone + Send + Sync,
        P: RiskSetPolicy<R> + Send + Sync,
    {
        let window = BirthWindowConstraint::new(
            (self.mode.birth_window_set && self.mode.distance_config.is_some())
                .then_some(self.criteria.birth_date_window_days),
        );
        match_transition_ordered(
            self.mode.cohort,
            TransitionMatchRequest {
                criteria: &self.criteria,
                options: &self.mode.options,
                strategy: self.strategy,
                constraints: &(&self.constraints, window),
                distance_config: self.mode.distance_config.as_ref(),
                risk_set_policy: &self.risk_set_policy,
            },
            self.mode.id_order,
        )
    }

    /// Execute role-transition matching and compute balance diagnostics in one call.
    pub fn run_with_balance<B: crate::CovariateRecord>(
        self,
        cases: &[B],
        controls: &[B],
    ) -> (MatchOutcome, crate::types::BalanceReport)
    where
        S: SelectionStrategy<R> + Clone + Send + Sync,
        P: RiskSetPolicy<R> + Send + Sync,
    {
        let outcome = self.run();
        let report = crate::balance_report(cases, controls, &outcome);
        (outcome, report)
    }
}

impl<R: MatchingRecord, S, G, P> MatchJob<StandardMode<'_, R>, R, S, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Trim the units outside the common support of the propensity scores
    /// before matching (`discard`). Needs a score distance, a propensity score
    /// or any numeric score; without one the job returns an invalid outcome.
    #[must_use]
    pub const fn with_common_support(mut self, policy: Option<crate::CommonSupport>) -> Self {
        self.criteria.common_support = policy;
        self
    }

    /// Set optional descending fallback ratios for standard matching.
    ///
    /// They cannot be combined with the nearest mode or with round assignment:
    /// the job returns an invalid outcome (`nearest_options`).
    #[must_use]
    pub fn with_ratio_fallback(mut self, fallback: Vec<MatchRatio>) -> Self {
        self.mode.ratio_fallback = fallback;
        self
    }

    /// Set a custom distance metric configuration.
    #[must_use]
    pub fn with_distance_config(mut self, config: DistanceConfig) -> Self {
        self.mode.distance_config = Some(config);
        self
    }

    /// Execute standard matching.
    pub fn run(self) -> MatchOutcome
    where
        S: SelectionStrategy<R> + Clone + Send + Sync,
    {
        let window = BirthWindowConstraint::new(
            (self.mode.birth_window_set && self.mode.distance_config.is_some())
                .then_some(self.criteria.birth_date_window_days),
        );
        match_standard(
            self.mode.anchors,
            self.mode.candidates,
            StandardMatchRequest {
                criteria: &self.criteria,
                strategy: self.strategy,
                constraints: &(&self.constraints, window),
                ratio_fallback: &self.mode.ratio_fallback,
                distance_config: self.mode.distance_config.as_ref(),
            },
        )
    }

    /// Execute standard matching and compute balance diagnostics in one call.
    pub fn run_with_balance<B: crate::CovariateRecord>(
        self,
        cases: &[B],
        controls: &[B],
    ) -> (MatchOutcome, crate::types::BalanceReport)
    where
        S: SelectionStrategy<R> + Clone + Send + Sync,
    {
        let outcome = self.run();
        let report = crate::balance_report(cases, controls, &outcome);
        (outcome, report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;
    use crate::types::{BaseRecord, EstimandDriftReason};

    #[test]
    fn match_job_runs_transition() {
        use crate::types::RoleTransitionRecord;

        let records = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("a1", date(2010, 1, 1)),
                Some(date(2012, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("c1", date(2010, 1, 2)), None),
        ];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs.len(), 1);
    }

    #[test]
    fn with_replacement_counts_anchors_without_candidates_as_unmatched() {
        let anchors = vec![
            BaseRecord::new("a1", date(2010, 1, 1)),
            BaseRecord::new("a2", date(2010, 1, 1)),
            BaseRecord::new("a3", date(2015, 1, 1)),
        ];
        let candidates = vec![BaseRecord::new("c1", date(2010, 1, 2))];

        for replacement in [false, true] {
            let result = MatchJob::new_standard(&anchors, &candidates, 42)
                .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
                .with_replacement(replacement)
                .run();
            // a3 is outside the birth window of every candidate. Without
            // replacement a2 finds c1 used already.
            assert_eq!(result.unmatched_cases, if replacement { 1 } else { 2 });
            assert_eq!(result.diagnostics.total_anchors_evaluated, 3);
            assert!(
                result
                    .diagnostics
                    .estimand_drift_reasons
                    .contains(&EstimandDriftReason::UnmatchedAnchors)
            );
        }
    }

    #[test]
    fn match_job_runs_standard() {
        let anchors = vec![BaseRecord::new("a1", date(2010, 1, 1))];
        let candidates = vec![BaseRecord::new("c1", date(2010, 1, 2))];

        let result = MatchJob::new_standard(&anchors, &candidates, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs.len(), 1);
    }

    #[test]
    fn match_job_with_gender_match_helper() {
        let mut a1 = BaseRecord::new("a1", date(2010, 1, 1));
        a1.strata.insert("sex".to_string(), "M".to_string());

        let mut c1_m = BaseRecord::new("c1_m", date(2010, 1, 1));
        c1_m.strata.insert("sex".to_string(), "M".to_string());

        let mut c2_f = BaseRecord::new("c2_f", date(2010, 1, 1));
        c2_f.strata.insert("sex".to_string(), "F".to_string());

        let anchors = vec![a1];
        let candidates = vec![c1_m, c2_f];

        let result = MatchJob::new_standard(&anchors, &candidates, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_gender_match()
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c1_m");
    }

    #[test]
    fn match_job_with_exact_matches_helper() {
        let mut a1 = BaseRecord::new("a1", date(2010, 1, 1));
        a1.strata.insert("sex".to_string(), "M".to_string());
        a1.strata.insert("region".to_string(), "A".to_string());

        let mut c1_match = BaseRecord::new("c1_match", date(2010, 1, 1));
        c1_match.strata.insert("sex".to_string(), "M".to_string());
        c1_match
            .strata
            .insert("region".to_string(), "A".to_string());

        let mut c2_wrong_region = BaseRecord::new("c2_wrong_region", date(2010, 1, 1));
        c2_wrong_region
            .strata
            .insert("sex".to_string(), "M".to_string());
        c2_wrong_region
            .strata
            .insert("region".to_string(), "B".to_string());

        let anchors = vec![a1];
        let candidates = vec![c1_match, c2_wrong_region];

        let result = MatchJob::new_standard(&anchors, &candidates, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_exact_matches(["sex", "region"])
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c1_match");
    }

    #[test]
    fn match_job_with_alive_check_helper() {
        use crate::types::RoleTransitionRecord;

        let a1 = RoleTransitionRecord::from_record(
            BaseRecord::new("a1", date(2010, 1, 1)),
            Some(date(2012, 1, 1)),
        );

        let mut c1_dead = BaseRecord::new("c1_dead", date(2010, 1, 1));
        c1_dead.death_date = Some(date(2011, 1, 1)); // dead before index
        let c1_dead = RoleTransitionRecord::from_record(c1_dead, None);

        let mut c2_alive = BaseRecord::new("c2_alive", date(2010, 1, 1));
        c2_alive.death_date = Some(date(2015, 1, 1)); // alive at index
        let c2_alive = RoleTransitionRecord::from_record(c2_alive, None);

        let records = vec![a1, c1_dead, c2_alive];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_alive_check()
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c2_alive");
    }

    #[test]
    fn match_job_with_resident_check_helper() {
        use crate::types::RoleTransitionRecord;
        use chrono::NaiveDate;

        #[derive(Clone)]
        struct ResidencyRecord {
            inner: RoleTransitionRecord<BaseRecord>,
            resident_until: Option<NaiveDate>,
        }

        impl crate::matching::MatchingRecord for ResidencyRecord {
            crate::delegate_matching_record!(inner);
        }

        impl crate::matching::RoleIndexedRecord for ResidencyRecord {
            fn event_date(&self) -> Option<NaiveDate> {
                self.inner.transition_date
            }
        }

        impl crate::matching::ResidentAtIndexRecord for ResidencyRecord {
            fn is_resident_at(&self, index_date: NaiveDate) -> bool {
                self.resident_until.is_none_or(|until| until > index_date)
            }
        }

        let a1 = ResidencyRecord {
            inner: RoleTransitionRecord::from_record(
                BaseRecord::new("a1", date(2010, 1, 1)),
                Some(date(2012, 1, 1)),
            ),
            resident_until: None,
        };
        let c1_non_resident = ResidencyRecord {
            inner: RoleTransitionRecord::from_record(
                BaseRecord::new("c1_non_resident", date(2010, 1, 1)),
                None,
            ),
            resident_until: Some(date(2011, 1, 1)),
        };
        let c2_resident = ResidencyRecord {
            inner: RoleTransitionRecord::from_record(
                BaseRecord::new("c2_resident", date(2010, 1, 1)),
                None,
            ),
            resident_until: None,
        };

        let records = vec![a1, c1_non_resident, c2_resident];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_resident_check()
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c2_resident");
    }

    #[test]
    fn match_job_with_resident_check_by_helper() {
        use crate::types::RoleTransitionRecord;

        let a1 = RoleTransitionRecord::from_record(
            BaseRecord::new("a1", date(2010, 1, 1)),
            Some(date(2012, 1, 1)),
        );
        let c1_non_resident = RoleTransitionRecord::from_record(
            BaseRecord::new("c1_non_resident", date(2010, 1, 1)),
            None,
        );
        let c2_resident = RoleTransitionRecord::from_record(
            BaseRecord::new("c2_resident", date(2010, 1, 1)),
            None,
        );

        let records = vec![a1, c1_non_resident, c2_resident];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_resident_check_by(|record: &RoleTransitionRecord<BaseRecord>, _index_date| {
                record.record.id == "c2_resident"
            })
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c2_resident");
    }

    #[test]
    fn match_job_with_resident_check_uses_base_record_emigration_date() {
        use crate::types::RoleTransitionRecord;

        let a1 = RoleTransitionRecord::from_record(
            BaseRecord::new("a1", date(2010, 1, 1)),
            Some(date(2012, 1, 1)),
        );
        // Emigrated before the case index date -> non-resident at index.
        let c1_emigrated = RoleTransitionRecord::from_record(
            BaseRecord::new("c1_emigrated", date(2010, 1, 1))
                .with_emigration_date(date(2011, 1, 1)),
            None,
        );
        // Emigrated after the case index date -> still resident at index.
        let c2_resident = RoleTransitionRecord::from_record(
            BaseRecord::new("c2_resident", date(2010, 1, 1)).with_emigration_date(date(2015, 1, 1)),
            None,
        );

        let records = vec![a1, c1_emigrated, c2_resident];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_resident_check()
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c2_resident");
    }

    #[test]
    fn resident_check_excludes_control_emigrating_on_the_index_date() {
        use crate::types::RoleTransitionRecord;

        let a1 = RoleTransitionRecord::from_record(
            BaseRecord::new("a1", date(2010, 1, 1)),
            Some(date(2012, 1, 1)),
        );
        // Emigration exactly on the case index date: residency is strict (> index),
        // so this control is non-resident and the only candidate is dropped.
        let c_boundary = RoleTransitionRecord::from_record(
            BaseRecord::new("c_boundary", date(2010, 1, 1)).with_emigration_date(date(2012, 1, 1)),
            None,
        );

        let records = vec![a1, c_boundary];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_resident_check()
            .run();

        assert_eq!(result.matched_cases, 0);
        assert!(result.pairs.is_empty(), "{:?}", result.pairs);
    }

    /// The record the parental-caliper case needs: a child matched on its own
    /// birth date, carrying its parents' birth years as named numerics.
    fn child(
        id: &str,
        transition: Option<chrono::NaiveDate>,
        mother: f64,
        father: f64,
    ) -> crate::types::RoleTransitionRecord<BaseRecord> {
        crate::types::RoleTransitionRecord::from_record(
            BaseRecord::new(id, date(2010, 1, 1))
                .with_numeric("mother_birth_year", mother)
                .with_numeric("father_birth_year", father),
            transition,
        )
    }

    #[test]
    fn with_constraints_binds_a_named_caliper_during_matching() {
        use crate::constraints::caliper_on_field;

        // The rule: comparators whose parents were born within one year, imposed
        // during matching. The far control is outside it.
        let records = vec![
            child("case", Some(date(2012, 1, 1)), 1980.0, 1978.0),
            child("far", None, 1996.0, 1978.0),
            child("near", None, 1981.0, 1978.0),
        ];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_constraints(vec![caliper_on_field("mother_birth_year", 1.0)])
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "near");
    }

    #[test]
    fn with_constraints_applies_every_caliper_in_the_list() {
        use crate::constraints::caliper_on_field;
        use crate::types::RoleTransitionRecord;

        // Two calipers, and the control that satisfies only the first must be
        // refused. This is the whole point of taking a list: a caller with two
        // numeric rules had to bake the second into its record type.
        let rules = || {
            vec![
                caliper_on_field::<RoleTransitionRecord<BaseRecord>>("mother_birth_year", 1.0),
                caliper_on_field("father_birth_year", 1.0),
            ]
        };

        // Asserted on a pool where the only candidate fails the SECOND caliper,
        // so the case can only stay unmatched. With a second, admissible control
        // in the pool the selection could pick it for reasons of its own and
        // the assertion would hold whether or not the caliper bound.
        let refused = vec![
            child("case", Some(date(2012, 1, 1)), 1980.0, 1978.0),
            child("mother_only", None, 1980.0, 1990.0),
        ];
        let result = MatchJob::new_transition(&refused, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_constraints(rules())
            .run();
        assert_eq!(result.matched_cases, 0);
        assert!(result.pairs.is_empty(), "{:?}", result.pairs);

        // And a control satisfying both is taken, so the refusal above is the
        // caliper and not something else about the pool.
        let admitted = vec![
            child("case", Some(date(2012, 1, 1)), 1980.0, 1978.0),
            child("mother_only", None, 1980.0, 1990.0),
            child("both", None, 1980.0, 1978.0),
        ];
        let result = MatchJob::new_transition(&admitted, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_constraints(rules())
            .run();
        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "both");
    }

    #[test]
    fn with_constraints_on_an_empty_list_refuses_nothing() {
        use crate::constraints::caliper_on_field;
        use crate::types::RoleTransitionRecord;

        // A configuration that names no calipers must not quietly become a
        // configuration that refuses every pair.
        let records = vec![
            child("case", Some(date(2012, 1, 1)), 1980.0, 1978.0),
            child("far", None, 1996.0, 1990.0),
        ];

        // Written the way a caller with an empty config list gets there: the
        // closure type is opaque, so `Vec<Caliper<_, _>>` cannot be spelled and
        // the type comes from the `map` instead.
        let configured: Vec<(String, f64)> = Vec::new();
        let empty: Vec<_> = configured
            .into_iter()
            .map(|(field, window)| {
                caliper_on_field::<RoleTransitionRecord<BaseRecord>>(&field, window)
            })
            .collect();

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_constraints(empty)
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "far");
    }

    #[test]
    fn with_constraints_keeps_the_constraints_added_before_it() {
        use crate::constraints::caliper_on_field;
        use crate::types::RoleTransitionRecord;

        // `Chained` asks the statically composed group first and only then the
        // list. Both must still bind: a control that satisfies the list and
        // fails the earlier constraint has to be refused.
        //
        // The constraint added first is a real `Constraint` on purpose.
        // `with_exact_match` sets strata criteria and is applied by the
        // candidate index, not by the constraint group, so composing after it
        // proves nothing about `Chained`.
        let records = vec![
            child("case", Some(date(2012, 1, 1)), 1980.0, 1978.0),
            child("fails_the_earlier_one", None, 1980.0, 1990.0),
            child("passes_both", None, 1980.0, 1978.0),
        ];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_constraint(caliper_on_field::<RoleTransitionRecord<BaseRecord>>(
                "father_birth_year",
                1.0,
            ))
            .with_constraints(vec![caliper_on_field("mother_birth_year", 1.0)])
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "passes_both");
    }

    #[test]
    fn with_constraints_takes_a_list_of_boxed_constraints() {
        use crate::constraints::{GenderMatch, caliper_on_field};
        use crate::types::RoleTransitionRecord;

        // The nameable form, and the only one that mixes constraint kinds in a
        // single runtime list: `Box<C>` is a `Constraint`, so the vector is a
        // group. A caller whose config names both an exact key and a caliper
        // ends up here.
        type Rule = Box<dyn Constraint<RoleTransitionRecord<BaseRecord>>>;

        let mut case = child("case", Some(date(2012, 1, 1)), 1980.0, 1978.0);
        case.record
            .strata
            .insert("sex".to_string(), "F".to_string());
        let mut wrong_sex = child("wrong_sex", None, 1980.0, 1978.0);
        wrong_sex
            .record
            .strata
            .insert("sex".to_string(), "M".to_string());
        let mut far = child("far", None, 1996.0, 1978.0);
        far.record.strata.insert("sex".to_string(), "F".to_string());
        let mut right = child("right", None, 1981.0, 1978.0);
        right
            .record
            .strata
            .insert("sex".to_string(), "F".to_string());

        let rules: Vec<Rule> = vec![
            Box::new(GenderMatch::on_key("sex")),
            Box::new(caliper_on_field::<RoleTransitionRecord<BaseRecord>>(
                "mother_birth_year",
                1.0,
            )) as Rule,
        ];

        let result = MatchJob::new_transition(&[case, wrong_sex, far, right], 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_constraints(rules)
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "right");
    }
}
