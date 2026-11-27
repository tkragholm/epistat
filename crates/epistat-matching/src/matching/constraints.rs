use super::distance::DistanceMetric;
use super::records::MatchingRecord;
use crate::types::{ControlIdx, MatchingCriteria, UniqueValueId};
use chrono::NaiveDate;
use rapidhash::RapidHashMap;
use rustc_hash::FxHashSet;
use std::marker::PhantomData;

/// Compact used-control tracking backed by a flat `Vec<bool>`.
///
/// Indexed by raw control index, so `contains` is one array lookup.
pub struct UsedControlsVec {
    bits: Vec<bool>,
    count: usize,
}

impl UsedControlsVec {
    /// An empty set for a cohort of `n` controls.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            bits: vec![false; n],
            count: 0,
        }
    }

    /// Whether the control is marked as used.
    #[inline]
    #[must_use]
    pub fn contains(&self, idx: ControlIdx) -> bool {
        self.bits.get(idx.get()).copied().unwrap_or(false)
    }

    /// Mark a control as used. Returns `true` if it was not already marked.
    #[inline]
    pub fn insert(&mut self, idx: ControlIdx) -> bool {
        let Some(slot) = self.bits.get_mut(idx.get()) else {
            return false;
        };
        if *slot {
            return false;
        }
        *slot = true;
        self.count += 1;
        true
    }

    /// Mark a control as unused again.
    pub(crate) fn remove(&mut self, idx: ControlIdx) {
        if let Some(slot) = self.bits.get_mut(idx.get())
            && *slot
        {
            *slot = false;
            self.count -= 1;
        }
    }

    /// The number of controls marked as used.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Whether no control is marked as used.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// A rule that a case and a control must satisfy to be matched.
pub trait Constraint<R: MatchingRecord>: Send + Sync {
    /// The reason recorded in the exclusion counts when the rule refuses a pair.
    fn reason(&self) -> &'static str;
    /// Whether the pair satisfies the rule.
    fn allows(&self, case: &R, control: &R, ctx: &ConstraintContext<'_>) -> bool;

    /// Whether the control is in the population at risk as far as this
    /// constraint is concerned, for a constraint that is a condition of the
    /// control's status on the case's index date, such as being alive or
    /// resident. Every other constraint allows.
    fn allows_at_risk(&self, _case: &R, _control: &R) -> bool {
        true
    }

    /// How risk-set matching under the default policy may read this
    /// constraint without asking it of every control, or `None` when it has
    /// to ask. Only constraints of this crate answer `Some`.
    #[doc(hidden)]
    fn status_rule(&self) -> Option<StatusRule<'_, R>> {
        None
    }

    /// Whether the constraint's own settings can be used, such as a caliper
    /// whose window is finite and not negative. A job checks its constraints
    /// before it matches.
    ///
    /// # Errors
    ///
    /// The error that names the setting. A constraint without settings to
    /// check returns `Ok`.
    fn check(&self) -> Result<(), crate::Error> {
        Ok(())
    }
}

/// A constraint whose verdict on a pair is a function of the control and the
/// case's index date alone: it refuses the control on and after a date of the
/// control's own.
pub(crate) trait RefusesFromDate<R>: Sync {
    /// The first index date on which the control is refused, if any.
    fn refused_from(&self, control: &R) -> Option<NaiveDate>;
}

/// A constraint that allows a control on the index dates inside its periods
/// of residence and refuses it on the others.
pub(crate) trait ResidenceSpellSource<R>: Sync {
    /// The periods of residence of the control, or `None` when it is resident
    /// on every date.
    fn spells_of(&self, control: &R) -> Option<&[crate::constraints::ResidenceSpell]>;
}

/// A constraint that compares one numeric value of the case with the same
/// value of the control, as [`caliper_allows`] does.
pub(crate) trait FieldValueSource<R>: Sync {
    /// The value of the record.
    fn value_of(&self, record: &R) -> Option<f64>;
}

/// Whether a case with value `case` and a control with value `control` are
/// within `window` of each other. Two missing values are within it when
/// `match_missing`, and one missing value never is.
#[inline]
pub(crate) fn caliper_allows(
    case: Option<f64>,
    control: Option<f64>,
    window: f64,
    match_missing: bool,
) -> bool {
    match (case, control) {
        (Some(c), Some(ctrl)) => (c - ctrl).abs() <= window,
        (None, None) => match_missing,
        _ => false,
    }
}

/// What a constraint decides, in a form that risk-set matching under the
/// default policy can count with a sweep over the index dates. See
/// [`Constraint::status_rule`].
#[doc(hidden)]
pub struct StatusRule<'a, R>(pub(crate) StatusRuleKind<'a, R>);

pub(crate) enum StatusRuleKind<'a, R> {
    /// Allows every pair.
    AllowsAll,
    /// Refuses a control, for `reason`, on every index date from the date
    /// `source` gives for it, and allows it before. A control with no date is
    /// never refused.
    RefusesFrom {
        reason: &'static str,
        source: &'a dyn RefusesFromDate<R>,
    },
    /// Refuses a control, for `reason`, on the index dates outside the
    /// periods of residence `source` gives for it.
    OutsideSpells {
        reason: &'static str,
        source: &'a dyn ResidenceSpellSource<R>,
    },
    /// Refuses a pair, for `reason`, when the values `source` gives for the
    /// case and the control are not within `window` by [`caliper_allows`].
    ComparesValues {
        reason: &'static str,
        source: &'a dyn FieldValueSource<R>,
        window: f64,
        match_missing: bool,
    },
}

/// Add the rule of `constraint` to `rules`, or answer `false` when it has none.
fn push_status_rule<'s, R: MatchingRecord, C: Constraint<R> + ?Sized>(
    constraint: &'s C,
    rules: &mut Vec<StatusRule<'s, R>>,
) -> bool {
    match constraint.status_rule() {
        None => false,
        Some(StatusRule(StatusRuleKind::AllowsAll)) => true,
        Some(rule) => {
            rules.push(rule);
            true
        }
    }
}

/// Statically-dispatched collection of constraints.
///
/// This enables callers to compose constraints as tuples (heterogeneous)
/// or slices/arrays (homogeneous) without requiring `dyn Constraint`.
pub trait ConstraintGroup<R: MatchingRecord>: Send + Sync {
    /// The reason of the first constraint that refuses the pair, or `None` when all allow it.
    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str>;

    /// Whether every status constraint of the group (see
    /// [`Constraint::allows_at_risk`]) allows the control.
    fn allows_at_risk(&self, _case: &R, _control: &R) -> bool {
        true
    }

    /// Add the rules of the group's constraints to `rules`, in the order the
    /// group asks them, and answer `true`; or answer `false` when one of them
    /// has no rule (see [`Constraint::status_rule`]). A group of your own
    /// answers `false`.
    #[doc(hidden)]
    fn status_rules<'s>(&'s self, _rules: &mut Vec<StatusRule<'s, R>>) -> bool {
        false
    }

    /// [`Constraint::check`] of every constraint of the group, in the order
    /// the group asks them.
    ///
    /// # Errors
    ///
    /// The first constraint's error.
    fn check(&self) -> Result<(), crate::Error> {
        Ok(())
    }
}

impl<R: MatchingRecord> ConstraintGroup<R> for () {
    fn first_blocking_reason(
        &self,
        _case: &R,
        _control: &R,
        _ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        None
    }

    fn status_rules<'s>(&'s self, _rules: &mut Vec<StatusRule<'s, R>>) -> bool {
        true
    }
}

impl<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized> ConstraintGroup<R> for &G {
    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        (*self).first_blocking_reason(case, control, ctx)
    }

    fn allows_at_risk(&self, case: &R, control: &R) -> bool {
        (**self).allows_at_risk(case, control)
    }

    fn status_rules<'s>(&'s self, rules: &mut Vec<StatusRule<'s, R>>) -> bool {
        (**self).status_rules(rules)
    }

    fn check(&self) -> Result<(), crate::Error> {
        (**self).check()
    }
}

impl<R: MatchingRecord> ConstraintGroup<R> for [&dyn Constraint<R>] {
    fn allows_at_risk(&self, case: &R, control: &R) -> bool {
        self.iter().all(|c| c.allows_at_risk(case, control))
    }

    fn status_rules<'s>(&'s self, rules: &mut Vec<StatusRule<'s, R>>) -> bool {
        self.iter().all(|c| push_status_rule(*c, rules))
    }

    fn check(&self) -> Result<(), crate::Error> {
        self.iter().try_for_each(|c| c.check())
    }

    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        self.iter().find_map(|constraint| {
            (!constraint.allows(case, control, ctx)).then_some(constraint.reason())
        })
    }
}

impl<R: MatchingRecord, C, const N: usize> ConstraintGroup<R> for [C; N]
where
    C: Constraint<R>,
{
    fn allows_at_risk(&self, case: &R, control: &R) -> bool {
        self.iter().all(|c| c.allows_at_risk(case, control))
    }

    fn status_rules<'s>(&'s self, rules: &mut Vec<StatusRule<'s, R>>) -> bool {
        self.iter().all(|c| push_status_rule(c, rules))
    }

    fn check(&self) -> Result<(), crate::Error> {
        self.iter().try_for_each(Constraint::check)
    }

    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        self.iter().find_map(|constraint| {
            (!constraint.allows(case, control, ctx)).then_some(constraint.reason())
        })
    }
}

impl<R: MatchingRecord, C: Constraint<R> + ?Sized> Constraint<R> for Box<C> {
    fn reason(&self) -> &'static str {
        (**self).reason()
    }

    fn allows(&self, case: &R, control: &R, ctx: &ConstraintContext<'_>) -> bool {
        (**self).allows(case, control, ctx)
    }

    fn allows_at_risk(&self, case: &R, control: &R) -> bool {
        (**self).allows_at_risk(case, control)
    }

    fn status_rule(&self) -> Option<StatusRule<'_, R>> {
        (**self).status_rule()
    }

    fn check(&self) -> Result<(), crate::Error> {
        (**self).check()
    }
}

/// A homogeneous group whose length is decided at run time.
///
/// The array impl above needs `N` at compile time, and the tuple impls need the
/// shape at compile time. Neither can express "the constraints named in the
/// configuration file", which is the shape a caller ends up with once
/// [`crate::constraints::caliper_on_field`] lets it build one per named field.
///
/// `caliper_on_field` returns an unnameable opaque type, so a `Vec` of them can
/// be inferred from a `map`/`collect` but not written down. A caller that needs
/// to name the type, or to mix constraint kinds in one list, can use
/// `Vec<Box<dyn Constraint<R>>>`: `Box<C>` is itself a `Constraint`.
impl<R: MatchingRecord, C> ConstraintGroup<R> for Vec<C>
where
    C: Constraint<R>,
{
    fn allows_at_risk(&self, case: &R, control: &R) -> bool {
        self.iter().all(|c| c.allows_at_risk(case, control))
    }

    fn status_rules<'s>(&'s self, rules: &mut Vec<StatusRule<'s, R>>) -> bool {
        self.iter().all(|c| push_status_rule(c, rules))
    }

    fn check(&self) -> Result<(), crate::Error> {
        self.iter().try_for_each(Constraint::check)
    }

    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        self.iter().find_map(|constraint| {
            (!constraint.allows(case, control, ctx)).then_some(constraint.reason())
        })
    }
}

/// Two groups checked in order, `first` before `second`.
///
/// `MatchJob` composes constraints into a tuple, so its group type is built at
/// compile time: `.with_constraint(a).with_constraint(b)` is `(((), A), B)`.
/// That leaves no way to append a group whose own length is not known until run
/// time, because the `(G, C)` impl takes a single `Constraint` on the right.
/// `Chained` takes a `ConstraintGroup` on both sides, which is what
/// [`MatchJob::with_constraint_group`](crate::MatchJob::with_constraint_group) appends through.
///
/// The first blocking reason is reported, and `first` is asked before `second`,
/// so a caller that puts its cheap constraints first keeps them first.
pub struct Chained<G1, G2> {
    /// The group asked first.
    pub first: G1,
    /// The group asked when `first` allows the pair.
    pub second: G2,
}

impl<R: MatchingRecord, G1, G2> ConstraintGroup<R> for Chained<G1, G2>
where
    G1: ConstraintGroup<R>,
    G2: ConstraintGroup<R>,
{
    fn allows_at_risk(&self, case: &R, control: &R) -> bool {
        self.first.allows_at_risk(case, control) && self.second.allows_at_risk(case, control)
    }

    fn status_rules<'s>(&'s self, rules: &mut Vec<StatusRule<'s, R>>) -> bool {
        self.first.status_rules(rules) && self.second.status_rules(rules)
    }

    fn check(&self) -> Result<(), crate::Error> {
        self.first.check()?;
        self.second.check()
    }

    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        self.first
            .first_blocking_reason(case, control, ctx)
            .or_else(|| self.second.first_blocking_reason(case, control, ctx))
    }
}

impl<R: MatchingRecord, G, C> ConstraintGroup<R> for (G, C)
where
    G: ConstraintGroup<R>,
    C: Constraint<R>,
{
    fn allows_at_risk(&self, case: &R, control: &R) -> bool {
        self.0.allows_at_risk(case, control) && self.1.allows_at_risk(case, control)
    }

    fn status_rules<'s>(&'s self, rules: &mut Vec<StatusRule<'s, R>>) -> bool {
        self.0.status_rules(rules) && push_status_rule(&self.1, rules)
    }

    fn check(&self) -> Result<(), crate::Error> {
        self.0.check()?;
        self.1.check()
    }

    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        let (group, constraint) = self;
        if let Some(reason) = group.first_blocking_reason(case, control, ctx) {
            return Some(reason);
        }
        (!constraint.allows(case, control, ctx)).then_some(constraint.reason())
    }
}

macro_rules! impl_constraint_group_tuple {
    ($( $ty:ident : $value:ident ),+ $(,)?) => {
        impl<R: MatchingRecord, $($ty),+> ConstraintGroup<R> for ($($ty,)+)
        where
            $($ty: Constraint<R>,)+
        {
            fn allows_at_risk(&self, case: &R, control: &R) -> bool {
                let ($($value,)+) = self;
                true $(&& $value.allows_at_risk(case, control))+
            }

            fn status_rules<'s>(&'s self, rules: &mut Vec<StatusRule<'s, R>>) -> bool {
                let ($($value,)+) = self;
                true $(&& push_status_rule($value, rules))+
            }

            fn check(&self) -> Result<(), crate::Error> {
                let ($($value,)+) = self;
                $($value.check()?;)+
                Ok(())
            }

            fn first_blocking_reason(
                &self,
                case: &R,
                control: &R,
                ctx: &ConstraintContext<'_>,
            ) -> Option<&'static str> {
                let ($($value,)+) = self;
                $(
                    if !$value.allows(case, control, ctx) {
                        return Some($value.reason());
                    }
                )+
                None
            }
        }
    };
}

impl_constraint_group_tuple!(C1: c1);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3, C4: c4);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3, C4: c4, C5: c5);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3, C4: c4, C5: c5, C6: c6);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3, C4: c4, C5: c5, C6: c6, C7: c7);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3, C4: c4, C5: c5, C6: c6, C7: c7, C8: c8);

/// The state of a match that a constraint may read: the criteria, the controls used so far and the strata values of the pair.
pub struct ConstraintContext<'a> {
    /// The criteria of the job.
    pub criteria: &'a MatchingCriteria,
    /// The controls already taken by a case.
    pub used_controls: &'a UsedControlsVec,
    /// The unique-key values already taken, as ids from `unique_interner`.
    pub used_unique: &'a FxHashSet<UniqueValueId>,
    /// The ids of the unique-key values.
    pub unique_interner: &'a RapidHashMap<String, UniqueValueId>,
    /// The position of the control being tested.
    pub control_idx: ControlIdx,
    /// The case's values of the required strata, in the order of the criteria, when they have been built.
    pub case_strata_values: Option<&'a [Option<&'a str>]>,
    /// The control's values of the required strata, in the order of the criteria, when they have been built.
    pub control_strata_values: Option<&'a [Option<&'a str>]>,
}

pub struct ReplacementConstraint;

impl<R: MatchingRecord> Constraint<R> for ReplacementConstraint {
    fn reason(&self) -> &'static str {
        "used_control_replacement_disabled"
    }

    fn allows(&self, _case: &R, _control: &R, ctx: &ConstraintContext<'_>) -> bool {
        ctx.criteria.allow_replacement || !ctx.used_controls.contains(ctx.control_idx)
    }
}

pub struct NoSelfMatchConstraint;

impl<R: MatchingRecord> Constraint<R> for NoSelfMatchConstraint {
    fn reason(&self) -> &'static str {
        "self_match"
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        case.id() != control.id()
    }
}

/// Requires the case and the control to agree on every required stratum of the criteria.
pub struct ExactMatchConstraint;

impl<R: MatchingRecord> Constraint<R> for ExactMatchConstraint {
    fn reason(&self) -> &'static str {
        "strata_mismatch"
    }

    fn allows(&self, case: &R, control: &R, ctx: &ConstraintContext<'_>) -> bool {
        strata_match_fast(
            case,
            control,
            &ctx.criteria.required_strata,
            ctx.case_strata_values,
            ctx.control_strata_values,
        )
    }
}

/// The name `ExactMatchConstraint` goes by in code that speaks of strata.
pub type StrataExactConstraint = ExactMatchConstraint;

/// The birth-date window of the criteria, as a constraint, for a job whose
/// distance is not the birth date. With no window it allows every pair.
pub struct BirthWindowConstraint {
    days: Option<i64>,
}

impl BirthWindowConstraint {
    pub(crate) fn new(days: Option<i32>) -> Self {
        Self {
            days: days.map(i64::from),
        }
    }
}

impl<R: MatchingRecord> Constraint<R> for BirthWindowConstraint {
    fn reason(&self) -> &'static str {
        "date_window_mismatch"
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        self.days
            .is_none_or(|days| (control.birth_date() - case.birth_date()).num_days().abs() <= days)
    }

    fn status_rule(&self) -> Option<StatusRule<'_, R>> {
        self.days
            .is_none()
            .then_some(StatusRule(StatusRuleKind::AllowsAll))
    }
}

pub struct UniqueKeyConstraint;

impl<R: MatchingRecord> Constraint<R> for UniqueKeyConstraint {
    fn reason(&self) -> &'static str {
        "unique_key_reused"
    }

    fn allows(&self, _case: &R, control: &R, ctx: &ConstraintContext<'_>) -> bool {
        if ctx.criteria.unique_by_key.is_none() {
            return true;
        }

        let value = unique_value(control, ctx.criteria);
        value.is_none_or(|value| {
            !ctx.unique_interner
                .get(value)
                .is_some_and(|id| ctx.used_unique.contains(id))
        })
    }
}

/// Generic caliper constraint over a distance channel.
pub struct CaliperConstraint<'a, R: MatchingRecord, D: DistanceMetric<R>> {
    /// The distance that is measured.
    pub metric: &'a D,
    /// The largest distance a pair may have.
    pub max_distance: f64,
    /// The reason recorded when the pair is refused.
    pub reason: &'static str,
    marker: PhantomData<R>,
}

impl<'a, R: MatchingRecord, D: DistanceMetric<R>> CaliperConstraint<'a, R, D> {
    /// A constraint that refuses pairs whose distance under `metric` is above `max_distance`.
    #[must_use]
    pub const fn new(metric: &'a D, max_distance: f64, reason: &'static str) -> Self {
        Self {
            metric,
            max_distance,
            reason,
            marker: PhantomData,
        }
    }
}

impl<R: MatchingRecord, D: DistanceMetric<R> + Sync> Constraint<R> for CaliperConstraint<'_, R, D> {
    fn reason(&self) -> &'static str {
        self.reason
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        if !self.max_distance.is_finite() || self.max_distance < 0.0 {
            return false;
        }
        self.metric
            .distance(case, control)
            .is_some_and(|distance| distance.is_finite() && distance <= self.max_distance)
    }
}

pub fn unique_value<'a, R: MatchingRecord>(
    record: &'a R,
    criteria: &'a MatchingCriteria,
) -> Option<&'a str> {
    criteria
        .unique_by_key
        .as_ref()
        .and_then(|key| record.stratum(key))
        .or_else(|| record.unique_key())
}

/// The values of the `required` strata of `record`, in their order.
pub fn build_strata_values<'a, R: MatchingRecord>(
    record: &'a R,
    required: &[String],
) -> Vec<Option<&'a str>> {
    required.iter().map(|key| record.stratum(key)).collect()
}

pub(super) fn strata_match_fast<R: MatchingRecord>(
    case: &R,
    control: &R,
    required: &[String],
    case_values: Option<&[Option<&str>]>,
    control_values: Option<&[Option<&str>]>,
) -> bool {
    if required.is_empty() {
        return true;
    }
    let Some(case_values) = case_values else {
        return strata_match(case, control, required);
    };
    let Some(control_values) = control_values else {
        return strata_match(case, control, required);
    };
    if case_values.len() != control_values.len() {
        return false;
    }

    case_values
        .iter()
        .zip(control_values.iter())
        .all(|(case_val, control_val)| case_val == control_val)
}

fn strata_match<R: MatchingRecord>(case: &R, control: &R, required: &[String]) -> bool {
    required
        .iter()
        .all(|key| case.stratum(key) == control.stratum(key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::DateDistance;
    use crate::types::MatchingCriteria;
    use crate::{date, record};

    fn context<'a>(
        criteria: &'a MatchingCriteria,
        used_controls: &'a UsedControlsVec,
        used_unique: &'a FxHashSet<UniqueValueId>,
        unique_interner: &'a RapidHashMap<String, UniqueValueId>,
        control_idx: ControlIdx,
        case_strata_values: Option<&'a [Option<&'a str>]>,
        control_strata_values: Option<&'a [Option<&'a str>]>,
    ) -> ConstraintContext<'a> {
        ConstraintContext {
            criteria,
            used_controls,
            used_unique,
            unique_interner,
            control_idx,
            case_strata_values,
            control_strata_values,
        }
    }

    fn empty_used_controls() -> UsedControlsVec {
        UsedControlsVec::with_capacity(0)
    }

    #[test]
    fn replacement_constraint_blocks_reuse_without_replacement() {
        let criteria = MatchingCriteria::builder().build().validate().unwrap();
        let case = record("a", date(2010, 1, 1));
        let control = record("b", date(2010, 1, 2));
        let mut used_controls = UsedControlsVec::with_capacity(3);
        used_controls.insert(ControlIdx::new(2));
        let empty_unique = FxHashSet::default();
        let empty_interner = RapidHashMap::default();
        let ctx = context(
            &criteria,
            &used_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(2),
            None,
            None,
        );
        assert!(!ReplacementConstraint.allows(&case, &control, &ctx));

        let criteria = MatchingCriteria::builder()
            .allow_replacement(true)
            .build()
            .validate()
            .unwrap();
        let ctx = context(
            &criteria,
            &used_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(2),
            None,
            None,
        );
        assert!(ReplacementConstraint.allows(&case, &control, &ctx));
    }

    #[test]
    fn no_self_match_constraint_blocks_self_pairs() {
        let criteria = MatchingCriteria::builder().build().validate().unwrap();
        let row = record("same", date(2010, 1, 1));
        let empty_controls = empty_used_controls();
        let empty_unique = FxHashSet::default();
        let empty_interner = RapidHashMap::default();
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(!NoSelfMatchConstraint.allows(&row, &row, &ctx));
    }

    #[test]
    fn exact_match_constraint_uses_strata_paths() {
        let criteria = MatchingCriteria::builder()
            .required_strata(vec!["municipality".to_string()])
            .build()
            .validate()
            .unwrap();
        let mut case = record("case", date(2010, 1, 1));
        case.strata
            .insert("municipality".to_string(), "0101".to_string());
        let mut control = record("control", date(2010, 1, 2));
        control
            .strata
            .insert("municipality".to_string(), "0202".to_string());
        let empty_controls = empty_used_controls();
        let empty_unique = FxHashSet::default();
        let empty_interner = RapidHashMap::default();
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(!ExactMatchConstraint.allows(&case, &control, &ctx));

        let case_vals = build_strata_values(&case, &criteria.required_strata);
        let control_vals = build_strata_values(&case, &criteria.required_strata);
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(0),
            Some(&case_vals),
            Some(&control_vals),
        );
        assert!(ExactMatchConstraint.allows(&case, &control, &ctx));
    }

    #[test]
    fn unique_key_constraint_blocks_reused_unique_values() {
        let criteria = MatchingCriteria::builder()
            .unique_by_key("family".to_string())
            .build()
            .validate()
            .unwrap();
        let mut control = record("control", date(2010, 1, 1));
        control
            .strata
            .insert("family".to_string(), "F1".to_string());

        let mut unique_interner = RapidHashMap::default();
        unique_interner.insert("F1".to_string(), UniqueValueId::new(1));
        let mut used_unique = FxHashSet::default();
        used_unique.insert(UniqueValueId::new(1));
        let empty_controls = empty_used_controls();
        let ctx = context(
            &criteria,
            &empty_controls,
            &used_unique,
            &unique_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(!UniqueKeyConstraint.allows(&control, &control, &ctx));

        let criteria = MatchingCriteria::builder().build().validate().unwrap();
        let ctx = context(
            &criteria,
            &empty_controls,
            &used_unique,
            &unique_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(UniqueKeyConstraint.allows(&control, &control, &ctx));
    }

    #[test]
    fn date_caliper_constraint_uses_birth_window() {
        let criteria = MatchingCriteria::builder()
            .birth_date_window_days(5)
            .build()
            .validate()
            .unwrap();
        let case = record("case", date(2010, 1, 1));
        let control_close = record("control_close", date(2010, 1, 5));
        let control_far = record("control_far", date(2010, 1, 10));
        let metric = DateDistance;
        let caliper = CaliperConstraint::new(
            &metric,
            f64::from(criteria.birth_date_window_days),
            "date_caliper",
        );
        let empty_controls = empty_used_controls();
        let empty_unique = FxHashSet::default();
        let empty_interner = RapidHashMap::default();
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(caliper.allows(&case, &control_close, &ctx));
        assert!(!caliper.allows(&case, &control_far, &ctx));
    }

    #[test]
    fn generic_caliper_constraint_works_with_custom_metric() {
        let criteria = MatchingCriteria::builder().build().validate().unwrap();
        let case = record("case", date(2010, 1, 1));
        let near = record("near", date(2010, 1, 2));
        let far = record("far", date(2010, 1, 10));
        let metric = DateDistance;
        let caliper = CaliperConstraint::new(&metric, 3.0, "date_caliper");
        let empty_controls = empty_used_controls();
        let empty_unique = FxHashSet::default();
        let empty_interner = RapidHashMap::default();
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(caliper.allows(&case, &near, &ctx));
        assert!(!caliper.allows(&case, &far, &ctx));
    }

    #[test]
    fn unique_value_prefers_strata_key_then_fallback_key() {
        let criteria = MatchingCriteria {
            unique_by_key: Some("family".to_string()),
            ..MatchingCriteria::default()
        };
        let mut record_with_strata = record("a", date(2010, 1, 1));
        record_with_strata
            .strata
            .insert("family".to_string(), "fam_a".to_string());
        assert_eq!(unique_value(&record_with_strata, &criteria), Some("fam_a"));

        let mut fallback = record("b", date(2010, 1, 1));
        fallback.unique_key = Some("uk_b".to_string());
        assert_eq!(unique_value(&fallback, &criteria), Some("uk_b"));
    }
}
