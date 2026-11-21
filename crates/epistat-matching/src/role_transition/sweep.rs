//! Risk sets under the default at-risk rule, kept up to date by a sweep over
//! the case dates instead of being scanned for every case.
//!
//! # Algorithm
//!
//! Cases are matched in order of event date. Under the default rule a record
//! is at risk on day `d` when it has entered (`entry <= d`, or no entry date)
//! and has no event on or before `d`. The other constraints this sweep accepts
//! refuse a control from a date of its own on (alive at the index date is
//! one); a control that has been drawn is refused when drawing without
//! replacement. For a case on day `d`, each control of its stratum and birth
//! window is therefore in exactly one of these states, checked in the order
//! the scan checks them:
//!
//! 1. not at risk, counted as excluded by the risk-set filter;
//! 2. at risk and already drawn, counted under the replacement constraint;
//! 3. at risk and refused by a date constraint, counted under the first such
//!    constraint, in the order of the constraint group, whose date is on or
//!    before `d`;
//! 4. eligible.
//!
//! The state of a control changes only when `d` passes its entry, event or
//! refusal dates, or when it is drawn. The sweep keeps each list of those
//! dates sorted and, as `d` advances, moves the controls whose dates it
//! passes to their new state. A stratum keeps its controls in two orders: by
//! birth date, so that a birth window is a range of positions, and by id,
//! which is the order of the pool a scan builds. Fenwick trees over the birth
//! order count the controls in states 1 to 3 in any range, and the eligible
//! controls are counted by difference. The seeded draw picks a position in the
//! pool by its length alone, so the pool is never listed: the control at a
//! position is the eligible control of that rank by id, found with a Fenwick
//! tree over the id order when the window holds the whole stratum and with a
//! [`WaveletMatrix`] over the birth order, built for the stratum the first time
//! it is needed, otherwise. Drawing more than one control removes a drawn
//! position from the pool as `Vec::swap_remove` does, which a small map of
//! moved positions follows.
//!
//! # Complexity
//!
//! For `n` records in strata of at most `m`, `L = ceil(log2 m)`, `k` date
//! constraints and `c` cases drawing `r` controls each: sorting the dates is
//! `O(n log n)`; every record changes state at most `k + 3` times, at
//! `O(log m)` each, or `O(L log m)` in a stratum with a wavelet matrix; each
//! case costs `O(k log m)` for its counts and `O(r L log m)` for its draws.
//! In all `O((n (k + L) + c r L) log m + n log n)` time, against `O(c m)` for
//! the scan, and `O(n (k + 8))` words with `2 L` more per record of a stratum
//! with a wavelet matrix.
//!
//! # References
//!
//! - Fenwick, P. M. (1994). A new data structure for cumulative frequency
//!   tables. Software: Practice and Experience 24(3), 327-336.
//! - Claude, F., Navarro, G. and Ordóñez, A. (2015). The wavelet matrix: an
//!   efficient wavelet tree for large alphabets. Information Systems 47,
//!   15-32.

use super::IdOrder;
use super::order_statistics::{Fenwick, WaveletMatrix};
use crate::matching::{CandidateIndex, RoleIndexedRecord, StatusRule, StatusRuleKind};
use chrono::{Datelike, NaiveDate};
use rustc_hash::FxHashMap;
use std::ops::Range;

const NOT_AT_RISK: u8 = 0;
const DRAWN: u8 = 1;
const ELIGIBLE: u8 = 2;
/// `REFUSED + slot` is refused for the reason of that slot.
const REFUSED: u8 = 3;

/// The controls of one stratum and their counts.
struct Stratum {
    /// Record index of each id rank.
    by_id: Vec<u32>,
    /// Record index of each birth position.
    by_birth: Vec<u32>,
    /// Id rank of each birth position.
    id_rank_by_birth: Vec<u32>,
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

/// A list of records sorted by one of their dates, and how far the sweep has
/// passed through it.
struct DateQueue {
    order: Vec<u32>,
    dates: Vec<i32>,
    next: usize,
}

impl DateQueue {
    /// The records whose date is not `NEVER`, by date.
    fn new(dates: Vec<i32>) -> Self {
        let mut order: Vec<u32> = (0..dates.len())
            .filter(|&idx| dates[idx] != NEVER)
            .map(|idx| u32::try_from(idx).expect("record index fits u32"))
            .collect();
        order.sort_unstable_by_key(|&idx| (dates[idx as usize], idx));
        Self {
            order,
            dates,
            next: 0,
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

/// The risk-set state of every record on the current case date.
pub(super) struct RiskSetSweep {
    strata: Vec<Stratum>,
    units: Vec<Unit>,
    entry: DateQueue,
    event: DateQueue,
    /// One queue per date constraint, in the order of the group.
    refusals: Vec<DateQueue>,
    /// The reason slot of each date constraint.
    refusal_slot: Vec<u8>,
    reasons: Vec<&'static str>,
    today: i32,
}

impl RiskSetSweep {
    /// The sweep of `cohort` with the strata of `index`, which holds every
    /// record, the date constraints of `rules` and the order of the ids,
    /// which must be distinct. Every record is out of the risk set until the
    /// first call of [`Self::advance`].
    pub(super) fn new<R: RoleIndexedRecord>(
        cohort: &[R],
        index: &CandidateIndex<'_>,
        rules: &[StatusRule<'_, R>],
        id_order: &IdOrder,
    ) -> Option<Self> {
        let n = cohort.len();
        u32::try_from(n).ok()?;
        if !id_order.distinct() || id_order.len() != n {
            return None;
        }
        let mut reasons: Vec<&'static str> = Vec::new();
        let mut refusal_slot = Vec::with_capacity(rules.len());
        let mut refusal_dates = Vec::with_capacity(rules.len());
        for StatusRule(kind) in rules {
            let StatusRuleKind::RefusesFrom { reason, source } = kind else {
                continue;
            };
            let slot = reasons
                .iter()
                .position(|seen| seen == reason)
                .unwrap_or_else(|| {
                    reasons.push(reason);
                    reasons.len() - 1
                });
            refusal_slot.push(u8::try_from(slot).ok()?);
            refusal_dates.push(
                cohort
                    .iter()
                    .map(|record| source.refused_from(record).map_or(NEVER, day))
                    .collect::<Vec<i32>>(),
            );
        }
        // The state is a byte: three fixed states and one per reason.
        u8::try_from(reasons.len() + usize::from(REFUSED)).ok()?;

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
        let mut strata = Vec::with_capacity(index.strata().len());
        for (entries, by_id) in index.strata().iter().zip(by_id_of) {
            let by_birth: Vec<u32> = entries
                .iter()
                .map(|&(_, idx)| u32::try_from(idx).expect("record index fits u32"))
                .collect();
            let id_rank_by_birth = by_birth
                .iter()
                .map(|&idx| units[idx as usize].id_rank)
                .collect();
            let m = by_birth.len();
            strata.push(Stratum {
                by_id,
                by_birth,
                id_rank_by_birth,
                at_risk: Fenwick::zeros(m),
                drawn: Fenwick::zeros(m),
                refused: (0..reasons.len()).map(|_| Fenwick::zeros(m)).collect(),
                eligible_by_id: Fenwick::zeros(m),
                eligible_by_birth: None,
            });
        }

        let entry = cohort
            .iter()
            .map(|record| record.entry_date().map_or(ALWAYS, day))
            .collect();
        let event = cohort
            .iter()
            .map(|record| record.event_date().map_or(NEVER, day))
            .collect();
        Some(Self {
            strata,
            units,
            entry: DateQueue::new(entry),
            event: DateQueue::new(event),
            refusals: refusal_dates.into_iter().map(DateQueue::new).collect(),
            refusal_slot,
            reasons,
            today: ALWAYS,
        })
    }

    /// The state of record `idx` on the current date.
    fn state_of(&self, idx: usize) -> u8 {
        let today = self.today;
        if self.entry.dates[idx] > today || self.event.dates[idx] <= today {
            return NOT_AT_RISK;
        }
        if self.units[idx].drawn {
            return DRAWN;
        }
        self.refusals
            .iter()
            .zip(&self.refusal_slot)
            .find(|(queue, _)| queue.dates[idx] <= today)
            .map_or(ELIGIBLE, |(_, &slot)| REFUSED + slot)
    }

    /// Add `delta` to the counts of `state` for record `idx`.
    fn count(&mut self, idx: usize, state: u8, delta: i32) {
        let unit = self.units[idx];
        let stratum = &mut self.strata[unit.stratum as usize];
        let position = unit.birth_position as usize;
        match state {
            NOT_AT_RISK => return,
            DRAWN => stratum.drawn.add(position, delta),
            ELIGIBLE => {
                stratum.eligible_by_id.add(unit.id_rank as usize, delta);
                if let Some(matrix) = &mut stratum.eligible_by_birth {
                    matrix.add(position, delta);
                }
            }
            refused => stratum.refused[usize::from(refused - REFUSED)].add(position, delta),
        }
        stratum.at_risk.add(position, delta);
    }

    /// Bring record `idx` to its state on the current date.
    fn refresh(&mut self, idx: usize) {
        let state = self.state_of(idx);
        let old = self.units[idx].state;
        if state != old {
            self.count(idx, old, -1);
            self.count(idx, state, 1);
            self.units[idx].state = state;
        }
    }

    /// Move to `date`, which is not before the previous one.
    pub(super) fn advance(&mut self, date: NaiveDate) {
        let today = day(date);
        debug_assert!(today >= self.today, "case dates are in order");
        self.today = today;
        let mut passed = Vec::new();
        for queue in std::iter::once(&mut self.entry)
            .chain(std::iter::once(&mut self.event))
            .chain(self.refusals.iter_mut())
        {
            while let Some(&idx) = queue.order.get(queue.next) {
                if queue.dates[idx as usize] > today {
                    break;
                }
                queue.next += 1;
                passed.push(idx);
            }
        }
        for idx in passed {
            self.refresh(idx as usize);
        }
    }

    /// The counts of `window`, a range of birth positions of `stratum`.
    pub(super) fn counts(&self, stratum: usize, window: Range<usize>) -> WindowCounts {
        let s = &self.strata[stratum];
        let (start, end) = (window.start, window.end);
        let at_risk = s.at_risk.range(start, end) as usize;
        let drawn = s.drawn.range(start, end) as usize;
        let refused: Vec<(&'static str, usize)> = self
            .reasons
            .iter()
            .zip(&s.refused)
            .map(|(&reason, tree)| (reason, tree.range(start, end) as usize))
            .collect();
        let refused_total: usize = refused.iter().map(|&(_, count)| count).sum();
        WindowCounts {
            not_at_risk: (end - start) - at_risk,
            drawn,
            refused,
            eligible: at_risk - drawn - refused_total,
        }
    }

    /// The record index of the eligible control of rank `rank` by id among
    /// those of `window` in `stratum`. `rank` is below their number.
    pub(super) fn nth_eligible(
        &mut self,
        stratum: usize,
        window: Range<usize>,
        rank: usize,
    ) -> usize {
        let rank = u32::try_from(rank).expect("rank fits u32");
        let whole = window.start == 0 && window.end == self.strata[stratum].by_birth.len();
        let id_rank = if whole {
            self.strata[stratum].eligible_by_id.kth(rank)
        } else {
            if self.strata[stratum].eligible_by_birth.is_none() {
                let s = &self.strata[stratum];
                let units = &self.units;
                let matrix = WaveletMatrix::new(&s.id_rank_by_birth, |position| {
                    units[s.by_birth[position] as usize].state == ELIGIBLE
                });
                self.strata[stratum].eligible_by_birth = Some(matrix);
            }
            let matrix = self.strata[stratum]
                .eligible_by_birth
                .as_ref()
                .expect("built above");
            matrix.kth(window.start, window.end, rank) as usize
        };
        self.strata[stratum].by_id[id_rank] as usize
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::{DeterministicSelection, MatchJob, RandomSelection, Rounds};
    use crate::role_transition::RiskSetPolicy;
    use crate::types::{
        AgeLimitYears, BaseRecord, BirthDateWindowDays, MatchOutcome, MatchRatio,
        RoleTransitionRecord,
    };
    use chrono::Days;

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
        for (seed, n, birth_days) in [(1, 60, 5), (2, 300, 40), (3, 900, 400), (4, 900, 3)] {
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
        assert!(checked > 300, "{checked} runs with pairs");
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
