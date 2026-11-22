//! Risk sets kept up to date by a sweep over the case dates instead of being
//! scanned for every case.
//!
//! # Algorithm
//!
//! Cases are matched in order of event date. A control is at risk on day `d`
//! when `d` lies in its at-risk span `from <= d < until`, which depends on the
//! control alone: under the default rule `from` is the entry date (or none)
//! and `until` the event date (or none), and a policy of the caller's may give
//! its own span (see [`AtRiskSpan`](super::AtRiskSpan)). The constraints this
//! sweep accepts are of three kinds, kept in the order of the constraint
//! group:
//!
//! * a status rule refuses a control on the dates of a set that depends on the
//!   control alone: from a date of its own on (alive at the index date), or
//!   outside its periods of residence;
//! * a value rule compares a value of the case with the same value of the
//!   control (a caliper on a numeric field);
//! * a rule that allows every pair.
//!
//! A control that has been drawn is refused when drawing without replacement.
//! For a case on day `d`, each control of its stratum and birth window is in
//! exactly one of these states, checked in the order the scan checks them:
//!
//! 1. not at risk, counted as excluded by the risk-set filter;
//! 2. at risk and already drawn, counted under the replacement constraint;
//! 3. at risk and refused by a rule, counted under the first rule of the
//!    group that refuses it;
//! 4. eligible.
//!
//! The state a control has as far as the status rules go changes only when
//! `d` passes the start or end of its span or of a refusal, or when it is
//! drawn. The sweep keeps each list of those dates sorted and, as `d`
//! advances, moves the controls whose dates it passes to their new state, the
//! first status rule that refuses them included. A stratum keeps its controls
//! in two orders: by birth date, so that a birth window is a range of
//! positions, and by id, which is the order of the pool a scan builds.
//!
//! A window is then read in one of three ways, with the same counts and the
//! same pool:
//!
//! * Listed: the controls of the window are visited in birth order, their
//!   states read, the value rules tested where they come in the order of the
//!   group, and the id ranks of the eligible controls sorted. This is the way
//!   for the strata in which no case's window holds more than [`LIST_LIMIT`]
//!   controls, and, when the group has a value rule, more than
//!   [`REACH_LIMIT`].
//! * Counted: Fenwick trees over the birth order count the controls of each
//!   state in any range, and the eligible controls are counted by difference.
//!   The seeded draw picks a position in the pool by its length alone, so the
//!   pool is never listed: the control at a position is the eligible control
//!   of that rank by id, found with a Fenwick tree over the id order when the
//!   window holds the whole stratum and with a [`WaveletMatrix`] over the
//!   birth order, built for the stratum the first time it is needed,
//!   otherwise. Drawing more than one control removes a drawn position from
//!   the pool as `Vec::swap_remove` does, which a small map of moved
//!   positions follows. This is the way for the wide strata of a group with
//!   no value rule.
//! * Reached: for the wide strata of a group with one value rule, whose
//!   eligible controls depend on the case and cannot be counted ahead. The
//!   controls are kept in chunks of [`CHUNK`] birth positions, each sorted by
//!   the value the rule compares and counting its controls by state. A window
//!   visits the controls of the chunks at its ends and finds, in each whole
//!   chunk, the controls the rule allows by binary search on the value. The
//!   others are counted from the chunk's counts: each state decides which
//!   rule refuses a control first, so a control that is at risk, not drawn,
//!   refused by a status rule after the value rule and outside the value
//!   window is counted under the value rule, and one refused by a status
//!   rule before it under that rule. The pool is listed from the allowed
//!   controls alone.
//!
//! # Complexity
//!
//! For `n` records in strata of at most `m`, `L = ceil(log2 m)`, `k` rules,
//! `s` periods of residence per record at most, and `c` cases drawing `r`
//! controls each: sorting the dates is `O(n (k + s) log n)`; every record
//! changes state `O(k + s)` times, at `O(k s)` to find its state plus, in a
//! counted stratum, `O(log m)` per tree or `O(L log m)` with a wavelet matrix.
//! A counted case costs `O(k log m)` for its counts and `O(r L log m)` for its
//! draws; a listed case with `w` controls in its window costs
//! `O(w k + w log w)`. A reached case costs `O((w / C) (k + log C) + C + h log h)`
//! with `C` the chunk size and `h` the controls the value rule allows. Memory
//! is `O(n (k + s))` words, with `k + 3` trees and possibly a wavelet matrix of
//! `2 L` words per record in a counted stratum, and a position, a value and
//! `(k + 3) / C` counts per record in a reached one.
//!
//! # References
//!
//! - Fenwick, P. M. (1994). A new data structure for cumulative frequency
//!   tables. Software: Practice and Experience 24(3), 327-336.
//! - Claude, F., Navarro, G. and Ordóñez, A. (2015). The wavelet matrix: an
//!   efficient wavelet tree for large alphabets. Information Systems 47,
//!   15-32.

use super::order_statistics::{Fenwick, WaveletMatrix};
use super::{AtRiskSpan, IdOrder};
use crate::constraints::ResidenceSpell;
use crate::matching::{
    CandidateIndex, RoleIndexedRecord, StatusRule, StatusRuleKind, caliper_allows,
};
use chrono::{Datelike, NaiveDate};
use rustc_hash::FxHashMap;
use std::ops::Range;

/// The largest window, in controls, that a stratum lists rather than counts.
pub(super) const LIST_LIMIT: usize = 2048;

/// The largest window, in controls, that a stratum with a value rule lists
/// control by control rather than from the chunks of its [`Reach`].
const REACH_LIMIT: usize = 256;

/// [`REACH_LIMIT`], or the smaller of it and the limit a test has set.
fn reach_limit() -> usize {
    list_limit().min(REACH_LIMIT)
}

/// [`LIST_LIMIT`], or in tests the limit a test has set on its thread.
fn list_limit() -> usize {
    #[cfg(test)]
    return LIST_LIMIT_IN_TESTS.with(std::cell::Cell::get);
    #[cfg(not(test))]
    LIST_LIMIT
}

const NOT_AT_RISK: u8 = 0;
const DRAWN: u8 = 1;
const ELIGIBLE: u8 = 2;
/// `REFUSED + p` is refused by the status rule at position `p` of the group.
const REFUSED: u8 = 3;

/// The counting trees of a counted stratum.
struct Trees {
    at_risk: Fenwick,
    drawn: Fenwick,
    /// One per refusal reason.
    refused: Vec<Fenwick>,
    /// Eligible controls by id rank.
    eligible_by_id: Fenwick,
    /// Eligible controls by birth position, once a window narrower than the
    /// stratum has asked.
    eligible_by_birth: Option<WaveletMatrix>,
}

/// The number of birth positions in a chunk of a [`Reach`].
const CHUNK: usize = 64;

/// What a counted stratum keeps when the group has one value rule, so that a
/// window lists only the controls the rule allows.
///
/// The birth positions are cut into chunks of [`CHUNK`]. Each chunk holds its
/// controls in order of the value the rule compares, those with a value first,
/// then those without, then those whose value is not a number, and counts its
/// controls by state. A window covers whole chunks and a part of the chunk at
/// each end. The controls of the end parts are visited. For each whole chunk
/// the controls with a value within the window of the case are found by two
/// binary searches, and the others are counted from the chunk's counts.
struct Reach {
    /// Position of the value rule among the rules.
    rule: usize,
    /// The birth position of each control, chunk by chunk, in the order above.
    positions: Vec<u32>,
    /// The value of each control with one, in the order of `positions`.
    values: Vec<f64>,
    /// For each chunk, the number of controls with a value and the number
    /// without.
    valued: Vec<u8>,
    without: Vec<u8>,
    /// For each chunk, the controls at risk, those at risk and drawn, and
    /// those at risk and not drawn by state, the state minus `ELIGIBLE` being
    /// the index. [`Self::width`] numbers per chunk.
    counts: Vec<u32>,
    width: usize,
}

impl Reach {
    fn new(rule: usize, states: usize, values: &[Option<f64>]) -> Self {
        let m = values.len();
        let chunks = m.div_ceil(CHUNK);
        let mut positions = Vec::with_capacity(m);
        let mut sorted = Vec::with_capacity(m);
        let mut valued = Vec::with_capacity(chunks);
        let mut without = Vec::with_capacity(chunks);
        for chunk in 0..chunks {
            let range = chunk * CHUNK..m.min((chunk + 1) * CHUNK);
            let position = |p: usize| u32::try_from(p).expect("position fits u32");
            let mut with: Vec<usize> = range
                .clone()
                .filter(|&p| values[p].is_some_and(|value| !value.is_nan()))
                .collect();
            let value_at = |p: usize| values[p].unwrap_or(f64::NAN);
            with.sort_unstable_by(|&a, &b| value_at(a).total_cmp(&value_at(b)).then(a.cmp(&b)));
            let missing: Vec<usize> = range.clone().filter(|&p| values[p].is_none()).collect();
            let not_a_number: Vec<usize> = range
                .filter(|&p| values[p].is_some_and(f64::is_nan))
                .collect();
            valued.push(u8::try_from(with.len()).expect("a chunk holds at most 64"));
            without.push(u8::try_from(missing.len()).expect("a chunk holds at most 64"));
            for p in with.into_iter().chain(missing).chain(not_a_number) {
                positions.push(position(p));
                sorted.push(value_at(p));
            }
        }
        let width = 2 + states;
        Self {
            rule,
            positions,
            values: sorted,
            valued,
            without,
            counts: vec![0; chunks * width],
            width,
        }
    }

    /// The entries of `chunk`, as a range of `positions`, that hold every
    /// control a case with value `case` may be paired with by the rule, and
    /// perhaps controls it may not.
    fn candidates(
        &self,
        chunk: usize,
        case: Option<f64>,
        window: f64,
        match_missing: bool,
    ) -> Range<usize> {
        let first = chunk * CHUNK;
        let valued = usize::from(self.valued[chunk]);
        let without = usize::from(self.without[chunk]);
        match case {
            None if match_missing => first + valued..first + valued + without,
            None => first..first,
            Some(c) if c.is_nan() => first..first,
            Some(c) => {
                // Left of `c` the controls the window refuses come first, and
                // right of it they come last, because a difference rounded to
                // a double does not decrease as the value moves away.
                let values = &self.values[first..first + valued];
                let low = values.partition_point(|&x| x < c && !((c - x).abs() <= window));
                let high = values.partition_point(|&x| x <= c || (c - x).abs() <= window);
                first + low..first + high.max(low)
            }
        }
    }

    fn counts_of(&self, chunk: usize) -> &[u32] {
        &self.counts[chunk * self.width..(chunk + 1) * self.width]
    }
}

/// The controls of one stratum and, when it is counted, their counts.
struct Stratum {
    /// Record index of each id rank.
    by_id: Vec<u32>,
    /// Record index of each birth position.
    by_birth: Vec<u32>,
    /// Id rank of each birth position.
    id_rank_by_birth: Vec<u32>,
    /// State of each birth position, as in [`Unit::state`], so that a listed
    /// window reads its states in one run of memory.
    states: Vec<u8>,
    /// For each value rule, in the order of the group, the value of each
    /// birth position.
    values: Vec<Vec<Option<f64>>>,
    trees: Option<Trees>,
    reach: Option<Reach>,
}

#[derive(Clone, Copy)]
struct Unit {
    stratum: u32,
    birth_position: u32,
    id_rank: u32,
    state: u8,
    drawn: bool,
}

/// The counts of a case's stratum and window.
#[derive(Default)]
pub(super) struct WindowCounts {
    /// Controls not at risk.
    pub(super) not_at_risk: usize,
    /// Controls at risk and already drawn.
    pub(super) drawn: usize,
    /// Controls refused, by reason.
    pub(super) refused: Vec<(&'static str, usize)>,
    /// Controls the case may draw.
    pub(super) eligible: usize,
}

/// The pool of a case: the eligible controls in id order, as a list or as a
/// position in the counting trees.
pub(super) enum Pool {
    /// Id ranks of the eligible controls in increasing order.
    Listed(Vec<u32>),
    /// The eligible controls of this window of the stratum, read from the
    /// trees, and the positions drawn so far.
    Counted(Range<usize>, SwapRemovePool),
}

impl Pool {
    /// The number of controls left to draw.
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Listed(ranks) => ranks.len(),
            Self::Counted(_, drawn) => drawn.len(),
        }
    }
}

/// A list of `(date, record)` sorted by date, and how far the sweep has
/// passed through it.
struct DateQueue {
    entries: Vec<(i32, u32)>,
    next: usize,
}

impl DateQueue {
    /// The entries whose date is not `NEVER`, by date. Those on `ALWAYS`
    /// are taken on the first date.
    fn new(mut entries: Vec<(i32, u32)>) -> Self {
        entries.retain(|&(date, _)| date != NEVER);
        entries.sort_unstable();
        Self { entries, next: 0 }
    }

    /// The records of the entries up to `today`, which have not been taken.
    fn take_through(&mut self, today: i32, passed: &mut Vec<u32>) {
        while let Some(&(date, idx)) = self.entries.get(self.next) {
            if date > today {
                break;
            }
            self.next += 1;
            passed.push(idx);
        }
    }
}

/// A date after every real one.
const NEVER: i32 = i32::MAX;
/// A date before every real one.
const ALWAYS: i32 = i32::MIN;

fn day(date: NaiveDate) -> i32 {
    date.num_days_from_ce()
}

fn day_or(date: Option<NaiveDate>, missing: i32) -> i32 {
    date.map_or(missing, day)
}

/// How a status rule decides, per record.
enum StatusDates {
    /// Refused on and after the date; `NEVER` when never refused.
    From(Vec<i32>),
    /// Refused outside the periods `spells[ranges[idx]]`, as `[from, to)`;
    /// a record without a range is never refused.
    Outside {
        ranges: Vec<Option<(u32, u32)>>,
        spells: Vec<(i32, i32)>,
    },
}

impl StatusDates {
    fn refuses(&self, idx: usize, today: i32) -> bool {
        match self {
            Self::From(dates) => dates[idx] <= today,
            Self::Outside { ranges, spells } => ranges[idx].is_some_and(|(start, end)| {
                !spells[start as usize..end as usize]
                    .iter()
                    .any(|&(from, to)| from <= today && today < to)
            }),
        }
    }
}

/// One rule of the group, in its order.
enum Rule {
    Status(StatusDates),
    /// The value rule numbered `number` among the value rules, with the value
    /// of every record and its window.
    Values {
        number: usize,
        values: Vec<Option<f64>>,
        window: f64,
        match_missing: bool,
    },
}

/// The risk-set state of every record on the current case date.
pub(super) struct RiskSetSweep {
    strata: Vec<Stratum>,
    units: Vec<Unit>,
    /// At-risk span of each record, `[from, until)`.
    spans: Vec<(i32, i32)>,
    /// The dates at which some record's state may change.
    queue: DateQueue,
    rules: Vec<Rule>,
    /// The reason slot of each rule.
    rule_slot: Vec<usize>,
    reasons: Vec<&'static str>,
    /// Whether the group has a value rule, so that every window is listed.
    has_values: bool,
    today: i32,
}

impl RiskSetSweep {
    /// The sweep of `cohort` with the strata of `index`, which holds every
    /// record, the rules of the constraint group, the order of the ids,
    /// which must be distinct, and the at-risk span `[from, until)` of each
    /// record. Every record is out of the risk set until the first call of
    /// [`Self::advance`]; no stratum is counted until [`Self::count_strata`].
    pub(super) fn new<R: RoleIndexedRecord>(
        cohort: &[R],
        index: &CandidateIndex<'_>,
        rules: &[StatusRule<'_, R>],
        id_order: &IdOrder,
        spans: Vec<AtRiskSpan>,
    ) -> Option<Self> {
        let n = cohort.len();
        u32::try_from(n).ok()?;
        if !id_order.distinct() || id_order.len() != n || spans.len() != n {
            return None;
        }
        // The state is a byte: three fixed states and one per rule.
        u8::try_from(rules.len() + usize::from(REFUSED)).ok()?;
        let record = |idx: usize| u32::try_from(idx).expect("record index fits u32");

        let spans: Vec<(i32, i32)> = spans
            .into_iter()
            .map(|span| match span {
                AtRiskSpan::Never => (NEVER, ALWAYS),
                AtRiskSpan::Between { from, until } => (day_or(from, ALWAYS), day_or(until, NEVER)),
            })
            .collect();
        let mut changes: Vec<(i32, u32)> = Vec::with_capacity(2 * n);
        for (idx, &(from, until)) in spans.iter().enumerate() {
            changes.push((from, record(idx)));
            changes.push((until, record(idx)));
        }

        let mut reasons: Vec<&'static str> = Vec::new();
        let mut rule_slot = Vec::with_capacity(rules.len());
        let mut kept = Vec::with_capacity(rules.len());
        let mut value_rules = 0_usize;
        for StatusRule(kind) in rules {
            let (reason, rule) = match kind {
                StatusRuleKind::AllowsAll => continue,
                StatusRuleKind::RefusesFrom { reason, source } => {
                    let dates: Vec<i32> = cohort
                        .iter()
                        .map(|record| day_or(source.refused_from(record), NEVER))
                        .collect();
                    changes.extend(dates.iter().enumerate().map(|(idx, &d)| (d, record(idx))));
                    (*reason, Rule::Status(StatusDates::From(dates)))
                }
                StatusRuleKind::OutsideSpells { reason, source } => {
                    let mut ranges = Vec::with_capacity(n);
                    let mut flat: Vec<(i32, i32)> = Vec::new();
                    for (idx, control) in cohort.iter().enumerate() {
                        let Some(spells) = source.spells_of(control) else {
                            ranges.push(None);
                            continue;
                        };
                        let start = u32::try_from(flat.len()).ok()?;
                        for &ResidenceSpell { from, to } in spells {
                            let (from, to) = (day_or(from, ALWAYS), day_or(to, NEVER));
                            flat.push((from, to));
                            changes.push((from, record(idx)));
                            changes.push((to, record(idx)));
                        }
                        ranges.push(Some((start, u32::try_from(flat.len()).ok()?)));
                    }
                    let dates = StatusDates::Outside {
                        ranges,
                        spells: flat,
                    };
                    (*reason, Rule::Status(dates))
                }
                StatusRuleKind::ComparesValues {
                    reason,
                    source,
                    window,
                    match_missing,
                } => {
                    let values = cohort
                        .iter()
                        .map(|record| source.value_of(record))
                        .collect();
                    let rule = Rule::Values {
                        number: value_rules,
                        values,
                        window: *window,
                        match_missing: *match_missing,
                    };
                    value_rules += 1;
                    (*reason, rule)
                }
            };
            let slot = reasons
                .iter()
                .position(|seen| *seen == reason)
                .unwrap_or_else(|| {
                    reasons.push(reason);
                    reasons.len() - 1
                });
            rule_slot.push(slot);
            kept.push(rule);
        }

        let mut units = vec![
            Unit {
                stratum: 0,
                birth_position: 0,
                id_rank: 0,
                state: NOT_AT_RISK,
                drawn: false,
            };
            n
        ];
        for (number, entries) in index.strata().iter().enumerate() {
            let stratum = u32::try_from(number).expect("stratum number fits u32");
            for (position, &(_, idx)) in entries.iter().enumerate() {
                let unit = &mut units[idx];
                unit.stratum = stratum;
                unit.birth_position = u32::try_from(position).expect("position fits u32");
            }
        }
        // The controls of each stratum in id order, read off the order of the
        // whole cohort.
        let mut by_id_of: Vec<Vec<u32>> = index
            .strata()
            .iter()
            .map(|entries| Vec::with_capacity(entries.len()))
            .collect();
        for &idx in id_order.by_id() {
            let unit = &mut units[idx as usize];
            let members = &mut by_id_of[unit.stratum as usize];
            unit.id_rank = u32::try_from(members.len()).expect("rank fits u32");
            members.push(idx);
        }
        let strata = index
            .strata()
            .iter()
            .zip(by_id_of)
            .map(|(entries, by_id)| {
                let by_birth: Vec<u32> = entries.iter().map(|&(_, idx)| record(idx)).collect();
                let id_rank_by_birth = by_birth
                    .iter()
                    .map(|&idx| units[idx as usize].id_rank)
                    .collect();
                let values = kept
                    .iter()
                    .filter_map(|rule| match rule {
                        Rule::Values { values, .. } => {
                            Some(by_birth.iter().map(|&idx| values[idx as usize]).collect())
                        }
                        Rule::Status(_) => None,
                    })
                    .collect();
                Stratum {
                    by_id,
                    states: vec![NOT_AT_RISK; by_birth.len()],
                    by_birth,
                    id_rank_by_birth,
                    values,
                    trees: None,
                    reach: None,
                }
            })
            .collect();

        Some(Self {
            strata,
            units,
            spans,
            queue: DateQueue::new(changes),
            rules: kept,
            rule_slot,
            reasons,
            has_values: value_rules > 0,
            today: ALWAYS,
        })
    }

    /// Count the strata with trees from now on, before the first call of
    /// [`Self::advance`]. `widest` is the number of controls in the widest
    /// window of each stratum. A stratum whose widest window is over
    /// [`LIST_LIMIT`] is counted with trees, and with a value rule over
    /// [`REACH_LIMIT`] with a [`Reach`]. Any other is listed.
    pub(super) fn count_strata(&mut self, widest: &[usize]) {
        debug_assert_eq!(self.today, ALWAYS, "before the sweep starts");
        if self.has_values {
            self.reach_strata(widest);
            return;
        }
        let slots = self.reasons.len();
        let limit = list_limit();
        for (stratum, _) in self
            .strata
            .iter_mut()
            .zip(widest)
            .filter(|(_, widest)| **widest > limit)
        {
            let m = stratum.by_birth.len();
            stratum.trees = Some(Trees {
                at_risk: Fenwick::zeros(m),
                drawn: Fenwick::zeros(m),
                refused: (0..slots).map(|_| Fenwick::zeros(m)).collect(),
                eligible_by_id: Fenwick::zeros(m),
                eligible_by_birth: None,
            });
        }
    }

    /// Give the strata with a wide window a [`Reach`] when the group has
    /// exactly one value rule. With more, every window is listed.
    fn reach_strata(&mut self, widest: &[usize]) {
        let mut value_rules = self
            .rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| matches!(rule, Rule::Values { .. }))
            .map(|(position, _)| position);
        let (Some(rule), None) = (value_rules.next(), value_rules.next()) else {
            return;
        };
        let states = self.rules.len() + 1;
        let limit = reach_limit();
        for (stratum, _) in self
            .strata
            .iter_mut()
            .zip(widest)
            .filter(|(_, widest)| **widest > limit)
        {
            stratum.reach = Some(Reach::new(rule, states, &stratum.values[0]));
        }
    }

    /// The state of record `idx` on the current date.
    fn state_of(&self, idx: usize) -> u8 {
        let today = self.today;
        let (from, until) = self.spans[idx];
        if from > today || until <= today {
            return NOT_AT_RISK;
        }
        if self.units[idx].drawn {
            return DRAWN;
        }
        self.rules
            .iter()
            .position(|rule| matches!(rule, Rule::Status(dates) if dates.refuses(idx, today)))
            .map_or(ELIGIBLE, |position| {
                REFUSED + u8::try_from(position).expect("fewer rules than states, checked in new")
            })
    }

    /// Add `delta` to the counts of `state` for record `idx`.
    fn count(&mut self, idx: usize, state: u8, delta: i32) {
        let unit = self.units[idx];
        let position = unit.birth_position as usize;
        let stratum = &mut self.strata[unit.stratum as usize];
        if let Some(reach) = &mut stratum.reach {
            if state != NOT_AT_RISK {
                let base = position / CHUNK * reach.width;
                let slot = if state == DRAWN {
                    1
                } else {
                    2 + usize::from(state - ELIGIBLE)
                };
                for index in [base, base + slot] {
                    reach.counts[index] = reach.counts[index].wrapping_add_signed(delta);
                }
            }
            return;
        }
        let Some(trees) = &mut stratum.trees else {
            return;
        };
        match state {
            NOT_AT_RISK => return,
            DRAWN => trees.drawn.add(position, delta),
            ELIGIBLE => {
                trees.eligible_by_id.add(unit.id_rank as usize, delta);
                if let Some(matrix) = &mut trees.eligible_by_birth {
                    matrix.add(position, delta);
                }
            }
            refused => {
                let slot = self.rule_slot[usize::from(refused - REFUSED)];
                trees.refused[slot].add(position, delta);
            }
        }
        trees.at_risk.add(position, delta);
    }

    /// Bring record `idx` to its state on the current date.
    fn refresh(&mut self, idx: usize) {
        let state = self.state_of(idx);
        let old = self.units[idx].state;
        if state != old {
            self.count(idx, old, -1);
            self.count(idx, state, 1);
            let unit = &mut self.units[idx];
            unit.state = state;
            self.strata[unit.stratum as usize].states[unit.birth_position as usize] = state;
        }
    }

    /// Move to `date`, which is not before the previous one.
    pub(super) fn advance(&mut self, date: NaiveDate) {
        let today = day(date);
        debug_assert!(today >= self.today, "case dates are in order");
        self.today = today;
        let mut passed = Vec::new();
        self.queue.take_through(today, &mut passed);
        for idx in passed {
            self.refresh(idx as usize);
        }
    }

    /// The counts of `window`, a range of birth positions of `stratum`, and
    /// the pool of the case `case_idx`.
    pub(super) fn pool(
        &self,
        stratum: usize,
        window: Range<usize>,
        case_idx: usize,
    ) -> (WindowCounts, Pool) {
        let s = &self.strata[stratum];
        let (start, end) = (window.start, window.end);
        if s.reach.is_some() {
            return self.reached(stratum, window, case_idx);
        }
        let Some(trees) = &s.trees else {
            return self.list(stratum, window, case_idx);
        };
        let at_risk = trees.at_risk.range(start, end) as usize;
        let drawn = trees.drawn.range(start, end) as usize;
        let refused: Vec<(&'static str, usize)> = self
            .reasons
            .iter()
            .zip(&trees.refused)
            .map(|(&reason, tree)| (reason, tree.range(start, end) as usize))
            .collect();
        let refused_total: usize = refused.iter().map(|&(_, count)| count).sum();
        let eligible = at_risk - drawn - refused_total;
        (
            WindowCounts {
                not_at_risk: (end - start) - at_risk,
                drawn,
                refused,
                eligible,
            },
            Pool::Counted(window, SwapRemovePool::new(eligible)),
        )
    }

    /// The counts and the pool of a window of a stratum with a [`Reach`].
    ///
    /// The controls the value rule allows are listed from the chunks. The
    /// others are not visited, except in the chunks at the ends of the window:
    /// those at risk and not drawn are counted by state, and each state
    /// decides which rule refuses them first, as [`Self::first_refusal`] does
    /// for one control. A status rule earlier in the group than the value rule
    /// refuses its controls before the value rule is asked, and one after it
    /// only those the value rule allows.
    fn reached(
        &self,
        stratum: usize,
        window: Range<usize>,
        case_idx: usize,
    ) -> (WindowCounts, Pool) {
        let s = &self.strata[stratum];
        let reach = s.reach.as_ref().expect("a stratum with a reach");
        let Rule::Values {
            values,
            window: width,
            match_missing,
            ..
        } = &self.rules[reach.rule]
        else {
            unreachable!("the rule of a reach is a value rule");
        };
        let (start, end) = (window.start, window.end);
        let m = s.by_birth.len();
        let case = values[case_idx];
        let states = reach.width - 2;
        let mut counts = WindowCounts::default();
        // Controls at risk and not drawn, and those among them the rule
        // allows, by state.
        let mut live = vec![0_usize; states];
        let mut whole_live = vec![0_usize; states];
        let mut inside = vec![0_usize; states];
        let mut ranks = Vec::new();
        let mut visit = |position: usize, counts: &mut WindowCounts, whole: bool| {
            let state = s.states[position];
            if whole && state < ELIGIBLE {
                return;
            }
            match state {
                NOT_AT_RISK => counts.not_at_risk += 1,
                DRAWN => counts.drawn += 1,
                _ => {
                    let index = usize::from(state - ELIGIBLE);
                    let allowed =
                        caliper_allows(case, s.values[0][position], *width, *match_missing);
                    live[index] += usize::from(!whole);
                    if allowed {
                        inside[index] += 1;
                        if state == ELIGIBLE {
                            ranks.push(s.id_rank_by_birth[position]);
                        }
                    }
                }
            }
        };
        // The chunks the window covers whole, the last of the stratum being
        // short.
        let first = start.div_ceil(CHUNK);
        let last = if end == m {
            m.div_ceil(CHUNK)
        } else {
            end / CHUNK
        };
        let (first, last) = if first < last { (first, last) } else { (0, 0) };
        let (from, to) = (first * CHUNK, m.min(last * CHUNK));
        if first == last {
            for position in start..end {
                visit(position, &mut counts, false);
            }
        } else {
            for position in (start..from).chain(to..end) {
                visit(position, &mut counts, false);
            }
            for chunk in first..last {
                let size = m.min((chunk + 1) * CHUNK) - chunk * CHUNK;
                let held = reach.counts_of(chunk);
                counts.not_at_risk += size - held[0] as usize;
                counts.drawn += held[1] as usize;
                for (total, &count) in whole_live.iter_mut().zip(&held[2..]) {
                    *total += count as usize;
                }
                for entry in reach.candidates(chunk, case, *width, *match_missing) {
                    visit(reach.positions[entry] as usize, &mut counts, true);
                }
            }
        }
        let mut refused = vec![0_usize; self.reasons.len()];
        for (index, &partial) in live.iter().enumerate() {
            let outside = partial + whole_live[index] - inside[index];
            let Some(status) = index.checked_sub(1) else {
                refused[self.rule_slot[reach.rule]] += outside;
                continue;
            };
            refused[self.rule_slot[status]] += inside[index];
            let first = if reach.rule < status {
                reach.rule
            } else {
                status
            };
            refused[self.rule_slot[first]] += outside;
        }
        ranks.sort_unstable();
        counts.refused = self.reasons.iter().copied().zip(refused).collect();
        counts.eligible = ranks.len();
        (counts, Pool::Listed(ranks))
    }

    /// The counts and the pool of a window, by visiting its controls.
    fn list(&self, stratum: usize, window: Range<usize>, case_idx: usize) -> (WindowCounts, Pool) {
        let s = &self.strata[stratum];
        let mut counts = WindowCounts {
            refused: self.reasons.iter().map(|&reason| (reason, 0)).collect(),
            ..WindowCounts::default()
        };
        let mut ranks = Vec::new();
        for position in window {
            let state = s.states[position];
            match state {
                NOT_AT_RISK => counts.not_at_risk += 1,
                DRAWN => counts.drawn += 1,
                _ => match self.first_refusal(s, state, case_idx, position) {
                    Some(slot) => counts.refused[slot].1 += 1,
                    None => ranks.push(s.id_rank_by_birth[position]),
                },
            }
        }
        ranks.sort_unstable();
        counts.eligible = ranks.len();
        (counts, Pool::Listed(ranks))
    }

    /// The reason slot of the first rule that refuses the control at birth
    /// `position` of stratum `s`, at risk and not drawn and in `state`, for
    /// case `case_idx`, or `None` when none does.
    #[inline]
    fn first_refusal(
        &self,
        s: &Stratum,
        state: u8,
        case_idx: usize,
        position: usize,
    ) -> Option<usize> {
        let first_status = (state >= REFUSED).then(|| usize::from(state - REFUSED));
        if !self.has_values {
            return first_status.map(|position| self.rule_slot[position]);
        }
        // The status rules before the first one that refuses allow the
        // control, so only the value rules before it are asked.
        let last = first_status.unwrap_or(self.rules.len());
        self.rules[..last]
            .iter()
            .position(|rule| match rule {
                Rule::Values {
                    number,
                    values,
                    window,
                    match_missing,
                } => !caliper_allows(
                    values[case_idx],
                    s.values[*number][position],
                    *window,
                    *match_missing,
                ),
                Rule::Status(_) => false,
            })
            .or(first_status)
            .map(|position| self.rule_slot[position])
    }

    /// The record index of the control drawn at `position` of `pool`, which
    /// is below the pool's length, removed from the pool as
    /// `Vec::swap_remove` removes it.
    pub(super) fn take(&mut self, stratum: usize, pool: &mut Pool, position: usize) -> usize {
        match pool {
            Pool::Listed(ranks) => {
                self.strata[stratum].by_id[ranks.swap_remove(position) as usize] as usize
            }
            Pool::Counted(window, drawn) => {
                let rank = drawn.swap_remove(position);
                self.nth_eligible(stratum, window.clone(), rank)
            }
        }
    }

    /// The record index of the eligible control of rank `rank` by id among
    /// those of `window` in the counted `stratum`. `rank` is below their
    /// number.
    fn nth_eligible(&mut self, stratum: usize, window: Range<usize>, rank: usize) -> usize {
        let rank = u32::try_from(rank).expect("rank fits u32");
        let s = &mut self.strata[stratum];
        let whole = window.start == 0 && window.end == s.by_birth.len();
        let units = &self.units;
        let trees = s.trees.as_mut().expect("a counted stratum");
        let id_rank = if whole {
            trees.eligible_by_id.kth(rank)
        } else {
            let (by_birth, id_rank_by_birth) = (&s.by_birth, &s.id_rank_by_birth);
            let matrix = trees.eligible_by_birth.get_or_insert_with(|| {
                WaveletMatrix::new(id_rank_by_birth, |position| {
                    units[by_birth[position] as usize].state == ELIGIBLE
                })
            });
            matrix.kth(window.start, window.end, rank) as usize
        };
        s.by_id[id_rank] as usize
    }

    /// Record that control `idx` has been drawn, without replacement.
    pub(super) fn mark_drawn(&mut self, idx: usize) {
        self.units[idx].drawn = true;
        self.refresh(idx);
    }
}

/// The positions of a pool of `len` entries, initially `0..len`, as
/// `Vec::swap_remove` leaves them.
pub(super) struct SwapRemovePool {
    len: usize,
    moved: FxHashMap<usize, usize>,
}

impl SwapRemovePool {
    pub(super) fn new(len: usize) -> Self {
        Self {
            len,
            moved: FxHashMap::default(),
        }
    }

    pub(super) const fn len(&self) -> usize {
        self.len
    }

    fn at(&self, position: usize) -> usize {
        self.moved.get(&position).copied().unwrap_or(position)
    }

    /// Remove the entry at `position`, which is below the length, and return it.
    pub(super) fn swap_remove(&mut self, position: usize) -> usize {
        let last = self.len - 1;
        let taken = self.at(position);
        let moved_in = self.at(last);
        self.moved.remove(&last);
        if position != last {
            self.moved.insert(position, moved_in);
        }
        self.len = last;
        taken
    }
}

#[cfg(test)]
thread_local! {
    /// Matching runs on this thread that took the sweep.
    pub(super) static SWEPT_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The list limit of the sweeps on this thread.
    static LIST_LIMIT_IN_TESTS: std::cell::Cell<usize> = const { std::cell::Cell::new(LIST_LIMIT) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::ResidenceSpell;
    use crate::matching::MatchingRecord;
    use crate::matching::{DeterministicSelection, MatchJob, RandomSelection, Rounds};
    use crate::role_transition::{AtRiskSpan, RiskSetPolicy};
    use crate::types::{
        AgeLimitYears, BaseRecord, BirthDateWindowDays, MatchOutcome, MatchRatio,
        RoleTransitionRecord,
    };
    use chrono::Days;
    use std::collections::{HashMap, HashSet};

    type Member = RoleTransitionRecord<BaseRecord>;

    /// The default policy's rules without its token, so that matching scans
    /// every candidate as it does for a policy of the caller's.
    struct Scanned;

    impl<R: RoleIndexedRecord> RiskSetPolicy<R> for Scanned {}

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self, bound: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % bound
        }

        fn chance(&mut self, percent: u64) -> bool {
            self.next(100) < percent
        }
    }

    fn origin() -> NaiveDate {
        NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date")
    }

    /// A cohort with many ties: birth dates on few days, case dates on few
    /// days, entry, death and emigration dates on and around the case dates,
    /// absent strata values, and ids whose order is not the cohort order.
    fn cohort(seed: u64, n: usize, birth_days: u64) -> Vec<Member> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|i| {
                let birth = origin() + Days::new(rng.next(birth_days));
                let id = format!("{:x}-{i}", rng.next(1 << 20));
                let mut base = BaseRecord::new(id, birth);
                if !rng.chance(5) {
                    let sex = if rng.chance(50) { "F" } else { "M" };
                    base.strata.insert("sex".into(), sex.into());
                }
                if !rng.chance(5) {
                    base.strata
                        .insert("region".into(), format!("r{}", rng.next(4)));
                }
                let around = |rng: &mut Lcg| birth + Days::new(rng.next(60) * 30);
                if rng.chance(20) {
                    base.entry_date = Some(around(&mut rng));
                }
                if rng.chance(25) {
                    base.death_date = Some(around(&mut rng));
                }
                if rng.chance(15) {
                    base.emigration_date = Some(around(&mut rng));
                }
                let event = rng.chance(20).then(|| around(&mut rng));
                RoleTransitionRecord::from_record(base, event)
            })
            .collect()
    }

    #[derive(Clone, Copy, Debug)]
    struct Setup {
        ratio: usize,
        fallback: bool,
        replace: bool,
        strata: usize,
        window: i32,
        alive: usize,
        rounds: bool,
    }

    fn emigration(record: &Member) -> Option<NaiveDate> {
        record.record.emigration_date
    }

    fn run<P: RiskSetPolicy<Member> + Send + Sync>(
        cohort: &[Member],
        setup: Setup,
        seed: u64,
        policy: P,
    ) -> MatchOutcome {
        let keys = ["sex", "region"];
        let job = MatchJob::new_transition(cohort, seed)
            .with_ratio(MatchRatio::new(setup.ratio).expect("ratio"))
            .with_ratio_fallback(if setup.fallback {
                (1..setup.ratio)
                    .rev()
                    .map(|r| MatchRatio::new(r).expect("ratio"))
                    .collect()
            } else {
                Vec::new()
            })
            .with_birth_window(BirthDateWindowDays::new(setup.window).expect("window"))
            .with_age_limit(AgeLimitYears::new(6).expect("age limit"))
            .with_replacement(setup.replace)
            .with_exact_matches(keys[..setup.strata].iter().copied());
        let job = job.with_risk_set_policy(policy);
        match (setup.alive, setup.rounds) {
            (0, false) => job.run(),
            (0, true) => job
                .with_strategy(Rounds(RandomSelection::seeded(seed)))
                .run(),
            (1, _) => job.with_alive_check().run(),
            _ => job
                .with_alive_check()
                .with_alive_check_by(emigration as fn(&Member) -> Option<NaiveDate>)
                .run(),
        }
    }

    /// Every pair and every diagnostic of the sweep equal those of the scan,
    /// over cohorts with tied dates, windows from none to wider than the
    /// cohort, with and without strata, replacement, fallback ratios and one
    /// or two date constraints.
    #[test]
    fn the_sweep_gives_what_the_scan_gives() {
        let mut checked = 0;
        // Every stratum counted, some, and none.
        for ((seed, n, birth_days), limit) in [(1, 60, 5), (2, 300, 40), (3, 900, 400), (4, 900, 3)]
            .into_iter()
            .flat_map(|cohort| [0, 64, LIST_LIMIT].map(|limit| (cohort, limit)))
        {
            LIST_LIMIT_IN_TESTS.with(|cell| cell.set(limit));
            let people = cohort(seed, n, birth_days);
            for ratio in [1, 3] {
                for fallback in [false, true] {
                    for replace in [false, true] {
                        for strata in [0, 1, 2] {
                            for window in [0, 30, 400, 36_500] {
                                for (alive, rounds) in
                                    [(0, false), (0, true), (1, false), (2, false)]
                                {
                                    if fallback && ratio == 1 {
                                        continue;
                                    }
                                    let setup = Setup {
                                        ratio,
                                        fallback,
                                        replace,
                                        strata,
                                        window,
                                        alive,
                                        rounds,
                                    };
                                    let before = SWEPT_RUNS.with(std::cell::Cell::get);
                                    let swept = run(
                                        &people,
                                        setup,
                                        seed + 10,
                                        super::super::DefaultRiskSetPolicy,
                                    );
                                    assert_eq!(
                                        SWEPT_RUNS.with(std::cell::Cell::get),
                                        before + 1,
                                        "{setup:?} took the sweep"
                                    );
                                    let scanned = run(&people, setup, seed + 10, Scanned);
                                    assert_eq!(
                                        format!("{swept:?}"),
                                        format!("{scanned:?}"),
                                        "seed {seed}, {setup:?}"
                                    );
                                    checked += usize::from(!swept.pairs.is_empty());
                                }
                            }
                        }
                    }
                }
            }
        }
        LIST_LIMIT_IN_TESTS.with(|cell| cell.set(LIST_LIMIT));
        assert!(checked > 900, "{checked} runs with pairs");
    }

    /// A policy of the caller's in the shape of Study II's: a control is at
    /// risk under the default rule, until a date of its own when it has one,
    /// and never when it is banned. With `declare` it gives its spans.
    struct Spanned<'a> {
        until: &'a HashMap<String, NaiveDate>,
        banned: &'a HashSet<String>,
        declare: bool,
    }

    impl RiskSetPolicy<Member> for Spanned<'_> {
        fn is_control_at_risk(&self, case: &Member, control: &Member, date: NaiveDate) -> bool {
            !self.banned.contains(control.id())
                && self
                    .until
                    .get(control.id())
                    .is_none_or(|until| *until > date)
                && super::super::DefaultRiskSetPolicy.is_control_at_risk(case, control, date)
        }

        fn control_at_risk_span(&self, control: &Member) -> Option<AtRiskSpan> {
            self.declare.then(|| {
                if self.banned.contains(control.id()) {
                    AtRiskSpan::Never
                } else {
                    AtRiskSpan::of_default_rule(control)
                        .ending_by(self.until.get(control.id()).copied())
                }
            })
        }
    }

    /// The inputs of a job of Study II's shape on `cohort`: an end of
    /// eligibility for some records, banned records, periods of residence
    /// with gaps for some, none for others and an empty list for a few, and a
    /// numeric field that some records lack.
    struct Shape {
        until: HashMap<String, NaiveDate>,
        banned: HashSet<String>,
        spells: HashMap<String, Vec<ResidenceSpell>>,
    }

    fn shape(seed: u64, people: &mut [Member]) -> Shape {
        let mut rng = Lcg(seed);
        let mut shape = Shape {
            until: HashMap::new(),
            banned: HashSet::new(),
            spells: HashMap::new(),
        };
        for person in people.iter_mut() {
            let birth = person.record.birth_date;
            let id = person.record.id.clone();
            let around = |rng: &mut Lcg| birth + Days::new(rng.next(60) * 30);
            if rng.chance(15) {
                shape.until.insert(id.clone(), around(&mut rng));
            }
            if rng.chance(5) {
                shape.banned.insert(id.clone());
            }
            match rng.next(10) {
                0 => {
                    shape.spells.insert(id.clone(), Vec::new());
                }
                1..=3 => {
                    let out = around(&mut rng);
                    let back = out + Days::new(rng.next(20) * 30);
                    let mut periods = vec![
                        ResidenceSpell {
                            from: None,
                            to: Some(out),
                        },
                        ResidenceSpell {
                            from: Some(back),
                            to: None,
                        },
                    ];
                    if rng.chance(30) {
                        periods.pop();
                    }
                    shape.spells.insert(id.clone(), periods);
                }
                _ => {}
            }
            if !rng.chance(10) {
                person
                    .record
                    .numerics
                    .insert("parent_year".into(), (1960 + rng.next(8)) as f64);
            }
        }
        shape
    }

    /// The jobs of Study II's shape, by how they are given: `0` the policy
    /// with its spans and residence as periods, which take the sweep; `1` the
    /// same policy without spans and residence as a check, which scan as
    /// Study II's job does.
    fn shaped(
        people: &[Member],
        shape: &Shape,
        seed: u64,
        given: usize,
        window: i32,
        order: usize,
        replace: bool,
    ) -> MatchOutcome {
        use crate::constraints::{MissingPolicy, caliper_on_field};
        let missing = if order == 1 {
            MissingPolicy::MatchMissing
        } else {
            MissingPolicy::Refuse
        };
        let calipers = || vec![caliper_on_field::<Member>("parent_year", 2.0).on_missing(missing)];
        let base = MatchJob::new_transition(people, seed)
            .with_ratio(MatchRatio::new(3).expect("ratio"))
            .with_ratio_fallback(vec![
                MatchRatio::new(2).expect("ratio"),
                MatchRatio::new(1).expect("ratio"),
            ])
            .with_birth_window(BirthDateWindowDays::new(window).expect("window"))
            .with_age_limit(AgeLimitYears::new(6).expect("age limit"))
            .with_replacement(replace)
            .with_exact_matches(["sex"]);
        let spells = |record: &Member| shape.spells.get(record.id()).map(Vec::as_slice);
        let resident = |record: &Member, date: NaiveDate| {
            spells(record).is_none_or(|periods| periods.iter().any(|p| p.contains(date)))
        };
        let policy = Spanned {
            until: &shape.until,
            banned: &shape.banned,
            declare: given == 0,
        };
        match (given, order) {
            // Alive, residence, caliper: Study II's order.
            (0, 0 | 1) => base
                .with_alive_check()
                .with_resident_spells(spells)
                .with_constraints(calipers())
                .with_risk_set_policy(policy)
                .run(),
            (_, 0 | 1) => base
                .with_alive_check()
                .with_resident_check_by(resident)
                .with_constraints(calipers())
                .with_risk_set_policy(policy)
                .run(),
            // The caliper first, then residence and alive.
            (0, _) => base
                .with_constraints(calipers())
                .with_resident_spells(spells)
                .with_alive_check()
                .with_risk_set_policy(policy)
                .run(),
            (_, _) => base
                .with_constraints(calipers())
                .with_resident_check_by(resident)
                .with_alive_check()
                .with_risk_set_policy(policy)
                .run(),
        }
    }

    /// A job of Study II's shape, given with spans and periods, takes the
    /// sweep and gives every pair and diagnostic of the same job given as
    /// Study II gives it, which scans; with the caliper before or after the
    /// status constraints, with and without replacement, and windows that are
    /// listed and counted.
    #[test]
    fn a_job_of_study_two_shape_sweeps_as_it_scans() {
        let mut checked = 0;
        // Every stratum counted, those with a window of over 64 controls, and
        // none.
        for (limit, (seed, n, birth_days)) in [0, 64, LIST_LIMIT].into_iter().flat_map(|limit| {
            [
                (21, 300, 40),
                (22, 900, 400),
                (23, 1500, 3),
                (24, 2000, 900),
            ]
            .map(|cohort| (limit, cohort))
        }) {
            LIST_LIMIT_IN_TESTS.with(|cell| cell.set(limit));
            let mut people = cohort(seed, n, birth_days);
            let shape = shape(seed, &mut people);
            for window in [0, 30, 400, 36_500] {
                for order in [0, 1, 2] {
                    for replace in [false, true] {
                        let before = SWEPT_RUNS.with(std::cell::Cell::get);
                        let swept = shaped(&people, &shape, seed, 0, window, order, replace);
                        assert_eq!(SWEPT_RUNS.with(std::cell::Cell::get), before + 1);
                        let scanned = shaped(&people, &shape, seed, 1, window, order, replace);
                        assert_eq!(SWEPT_RUNS.with(std::cell::Cell::get), before + 1);
                        assert_eq!(
                            format!("{swept:?}"),
                            format!("{scanned:?}"),
                            "seed {seed}, window {window}, order {order}, replace {replace}, \
                             limit {limit}"
                        );
                        checked += usize::from(swept.pairs.len() > 10);
                    }
                }
            }
        }
        LIST_LIMIT_IN_TESTS.with(|cell| cell.set(LIST_LIMIT));
        assert!(checked > 180, "{checked} runs with pairs");
    }

    /// The default policy with residence as periods and a caliper, whose
    /// windows are listed, sweeps as the scan does.
    #[test]
    fn periods_and_calipers_under_the_default_policy_sweep_as_they_scan() {
        use crate::constraints::caliper_on_field;
        for (seed, n, birth_days, limit) in [
            (31, 600, 40, 0),
            (32, 1200, 5, 64),
            (33, 900, 40, LIST_LIMIT),
        ] {
            LIST_LIMIT_IN_TESTS.with(|cell| cell.set(limit));
            let mut people = cohort(seed, n, birth_days);
            let shape = shape(seed, &mut people);
            let spells = |record: &Member| shape.spells.get(record.id()).map(Vec::as_slice);
            for window in [10, 36_500] {
                let job = |policy| {
                    let job = MatchJob::new_transition(&people, seed)
                        .with_ratio(MatchRatio::new(2).expect("ratio"))
                        .with_birth_window(BirthDateWindowDays::new(window).expect("window"))
                        .with_age_limit(AgeLimitYears::new(6).expect("age limit"))
                        .with_resident_spells(spells)
                        .with_alive_check();
                    if policy {
                        job.run()
                    } else {
                        job.with_risk_set_policy(Scanned).run()
                    }
                };
                assert_eq!(format!("{:?}", job(true)), format!("{:?}", job(false)));
                let with_caliper = |policy| {
                    let job = MatchJob::new_transition(&people, seed)
                        .with_birth_window(BirthDateWindowDays::new(window).expect("window"))
                        .with_age_limit(AgeLimitYears::new(6).expect("age limit"))
                        .with_constraints(vec![caliper_on_field::<Member>("parent_year", 1.0)]);
                    if policy {
                        job.run()
                    } else {
                        job.with_risk_set_policy(Scanned).run()
                    }
                };
                assert_eq!(
                    format!("{:?}", with_caliper(true)),
                    format!("{:?}", with_caliper(false))
                );
            }
        }
    }

    /// A policy whose span holds a case's own event date sends the job to the
    /// scan, which refuses the case as its own control.
    #[test]
    fn a_span_holding_the_case_date_scans() {
        struct Entered;
        impl RiskSetPolicy<Member> for Entered {
            fn is_control_at_risk(
                &self,
                _case: &Member,
                control: &Member,
                date: NaiveDate,
            ) -> bool {
                control.entry_date().is_none_or(|entry| entry <= date)
            }
            fn control_at_risk_span(&self, control: &Member) -> Option<AtRiskSpan> {
                Some(AtRiskSpan::Between {
                    from: control.entry_date(),
                    until: None,
                })
            }
        }
        let people = cohort(41, 300, 30);
        let before = SWEPT_RUNS.with(std::cell::Cell::get);
        let outcome = MatchJob::new_transition(&people, 3)
            .with_risk_set_policy(Entered)
            .run();
        assert_eq!(SWEPT_RUNS.with(std::cell::Cell::get), before);
        assert_ne!(outcome.pairs.len(), 0);
    }

    /// A constraint the sweep cannot count, a unique key or a selection that
    /// is not a seeded draw sends the job to the scan.
    #[test]
    fn other_jobs_scan() {
        let people = cohort(7, 300, 30);
        let before = SWEPT_RUNS.with(std::cell::Cell::get);
        let _ = MatchJob::new_transition(&people, 3)
            .with_resident_check_by(|_: &Member, _| true)
            .run();
        let _ = MatchJob::new_transition(&people, 3)
            .with_unique_by_key(Some("region".into()))
            .run();
        let _ = MatchJob::new_transition(&people, 3)
            .with_strategy(DeterministicSelection)
            .run();
        assert_eq!(SWEPT_RUNS.with(std::cell::Cell::get), before);
    }

    #[test]
    fn the_pool_follows_swap_remove() {
        let mut state = 9_u64;
        let mut next = |bound: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % bound as u64) as usize
        };
        for len in [1_usize, 2, 5, 40] {
            let mut plain: Vec<usize> = (0..len).collect();
            let mut pool = SwapRemovePool::new(len);
            while !plain.is_empty() {
                let position = next(plain.len());
                assert_eq!(pool.swap_remove(position), plain.swap_remove(position));
                assert_eq!(pool.len(), plain.len());
            }
        }
    }
}
