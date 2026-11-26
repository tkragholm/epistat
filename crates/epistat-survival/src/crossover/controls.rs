//! Control persons for the case-time-control and case-case-time-control
//! designs, sampled at the date of each case, and the windows of cases and
//! controls together.
//!
//! # Algorithm
//!
//! For each case event on day `t`, [`sample_controls`] draws `per_case`
//! distinct people from the candidates with the same matching key who are
//! under observation on day `t` (`start <= t <= end`) and are not the case's
//! own person, and who meet the [`ControlRule`]:
//!
//! - [`ControlRule::EventFree`], the case-time-control design \[1\]: no event
//!   on or before `t`, as in risk-set sampling. A person may be a control for
//!   several cases and may become a case later.
//! - [`ControlRule::FutureCases`], the case-case-time-control design \[2\]: a
//!   first event after `t + gap`, so the controls are people who become cases
//!   later, whose exposure at `t` is not yet changed by their own event.
//!
//! The draw is uniform over the eligible candidates and without replacement
//! within a case. The candidates of a key are sorted by their first event, so
//! those without an event by `t` (or by `t + gap`) are a suffix found by
//! binary search; within it a candidate is drawn at random and kept when it is
//! under observation on `t` and not drawn already. After `64 + 16 per_case`
//! draws without enough controls the eligible candidates are listed and the
//! rest are drawn from the list, which is uniform as well. Each case has its
//! own stream, `replicate_seed(seed, case)`, so the controls do not depend on
//! the number of threads. A case with fewer eligible candidates than
//! `per_case` gets all of them and is listed in [`Controls::cases_short`].
//!
//! [`case_time_control_windows`] builds the windows of the case events and of
//! their controls in one [`Windows`], the cases' sets first, and the group of
//! each window for [`super::case_time_control_design`].
//!
//! # References
//!
//! 1. Suissa S. The case-time-control design. Epidemiology 1995;6(3):248-53.
//! 2. Wang S, Linkletter C, Maclure M, Dore D, Mor V, Buka S, et al. Future
//!    cases as present controls to adjust for exposure trend bias in
//!    case-only studies. Epidemiology 2011;22(4):568-74.

use epistat_rng::{Stream, replicate_seed};
use rayon::prelude::*;

use super::{Observation, Referent, Windows, build_windows};
use crate::error::{Error, Result};

/// The case events at whose dates controls are sampled.
#[derive(Debug, Clone, Copy)]
pub struct CaseEvents<'a> {
    /// The day of each case event.
    pub day: &'a [i64],
    /// The person of each case event.
    pub person: &'a [u32],
    /// The matching key of each case event, such as a code of sex and birth
    /// year; `None` matches every candidate.
    pub key: Option<&'a [u64]>,
}

/// The people who may be drawn as controls, one entry per person.
#[derive(Debug, Clone, Copy)]
pub struct Candidates<'a> {
    /// The person code, as in the exposure data.
    pub person: &'a [u32],
    /// The first observed day.
    pub start: &'a [i64],
    /// The last observed day.
    pub end: &'a [i64],
    /// The day of the person's first event, `None` without one.
    pub first_event: &'a [Option<i64>],
    /// The matching key, compared with that of the case; `None` when the
    /// cases have none.
    pub key: Option<&'a [u64]>,
}

/// Who may serve as the control of a case event on day `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlRule {
    /// People without an event on or before `t`: the case-time-control
    /// design.
    EventFree,
    /// People whose first event is after `t + gap`: the case-case-time-control
    /// design, with future cases as controls.
    FutureCases {
        /// The days after `t` within which a first event rules a person out,
        /// at least 0.
        gap: i64,
    },
}

/// The sampled controls, one entry per control, in the order of the cases.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Controls {
    /// The position of the control's case event in the input.
    pub case: Vec<u32>,
    /// The control's person code.
    pub person: Vec<u32>,
    /// The index day, the day of the case event.
    pub day: Vec<i64>,
    /// The control's first observed day.
    pub start: Vec<i64>,
    /// The control's last observed day.
    pub end: Vec<i64>,
    /// The case events with fewer than `per_case` controls.
    pub cases_short: Vec<u32>,
}

impl Controls {
    /// The number of controls.
    #[must_use]
    pub fn len(&self) -> usize {
        self.case.len()
    }

    /// Whether there is no control.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.case.is_empty()
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::input(message)
}

/// The candidates of one key, sorted by first event with `None` last.
struct Group {
    key: u64,
    /// Candidate positions.
    members: Vec<usize>,
    /// The first event of each member, `i64::MAX` for none.
    first: Vec<i64>,
}

/// Samples `per_case` control persons for each case event under `rule`. See
/// the module documentation.
///
/// # Errors
///
/// [`Error::Input`] when the columns of the cases or the candidates differ in
/// length, one side has a key and the other not, an observation period ends
/// before it starts, `per_case` is 0, a gap is negative, or there are more
/// cases than a `u32` addresses.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the number of cases is checked against u32 first"
)]
pub fn sample_controls(
    cases: &CaseEvents<'_>,
    candidates: &Candidates<'_>,
    rule: ControlRule,
    per_case: usize,
    seed: u64,
) -> Result<Controls> {
    let n = cases.day.len();
    let m = candidates.person.len();
    if cases.person.len() != n || cases.key.is_some_and(|k| k.len() != n) {
        return Err(invalid(
            "day, person and key of the cases must have one entry per case",
        ));
    }
    if candidates.start.len() != m
        || candidates.end.len() != m
        || candidates.first_event.len() != m
        || candidates.key.is_some_and(|k| k.len() != m)
    {
        return Err(invalid(
            "person, start, end, first_event and key of the candidates must have one entry per candidate",
        ));
    }
    if cases.key.is_some() != candidates.key.is_some() {
        return Err(invalid(
            "the cases and the candidates must both have a key or both have none",
        ));
    }
    if let Some(i) = (0..m).find(|&i| candidates.end[i] < candidates.start[i]) {
        return Err(invalid(format!(
            "candidate {i} has an observation period that ends before it starts"
        )));
    }
    if per_case == 0 {
        return Err(invalid("per_case must be at least 1"));
    }
    let gap = match rule {
        ControlRule::EventFree => 0,
        ControlRule::FutureCases { gap } if gap >= 0 => gap,
        ControlRule::FutureCases { gap } => {
            return Err(invalid(format!("the gap is {gap}, which is negative")));
        }
    };
    if u32::try_from(n).is_err() {
        return Err(invalid("more cases than a u32 addresses"));
    }
    let key_of = |i: usize| candidates.key.map_or(0, |k| k[i]);
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by_key(|&i| (key_of(i), candidates.first_event[i].unwrap_or(i64::MAX), i));
    let mut groups: Vec<Group> = Vec::new();
    for &i in &order {
        let k = key_of(i);
        if groups.last().is_none_or(|g| g.key != k) {
            groups.push(Group {
                key: k,
                members: Vec::new(),
                first: Vec::new(),
            });
        }
        let g = groups.last_mut().expect("a group was pushed");
        g.members.push(i);
        g.first.push(candidates.first_event[i].unwrap_or(i64::MAX));
    }
    let per: Vec<Vec<usize>> = (0..n)
        .into_par_iter()
        .map(|c| {
            let t = cases.day[c];
            let key = cases.key.map_or(0, |k| k[c]);
            let Ok(g) = groups.binary_search_by_key(&key, |g| g.key) else {
                return Vec::new();
            };
            let g = &groups[g];
            // Eligible by the rule: first event after t (event-free) or after
            // t + gap, and for future cases an event at all.
            let from = g.first.partition_point(|&f| f <= t.saturating_add(gap));
            let to = match rule {
                ControlRule::EventFree => g.members.len(),
                ControlRule::FutureCases { .. } => g.first.partition_point(|&f| f < i64::MAX),
            };
            let pool = &g.members[from..to.max(from)];
            let ok = |i: usize| {
                candidates.start[i] <= t
                    && t <= candidates.end[i]
                    && candidates.person[i] != cases.person[c]
            };
            draw(pool, per_case, &ok, replicate_seed(seed, c as u64))
        })
        .collect();
    let mut out = Controls::default();
    for (c, chosen) in per.into_iter().enumerate() {
        if chosen.len() < per_case {
            out.cases_short.push(c as u32);
        }
        for i in chosen {
            out.case.push(c as u32);
            out.person.push(candidates.person[i]);
            out.day.push(cases.day[c]);
            out.start.push(candidates.start[i]);
            out.end.push(candidates.end[i]);
        }
    }
    Ok(out)
}

/// `k` distinct members of `pool` that pass `ok`, uniformly, or all that pass
/// when fewer do.
fn draw(pool: &[usize], k: usize, ok: &impl Fn(usize) -> bool, seed: u64) -> Vec<usize> {
    let mut rng = Stream::seeded(seed);
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    if pool.is_empty() {
        return chosen;
    }
    let mut tries = 64 + 16 * k;
    while chosen.len() < k && tries > 0 {
        tries -= 1;
        let i = pool[rng.below(pool.len())];
        if ok(i) && !chosen.contains(&i) {
            chosen.push(i);
        }
    }
    if chosen.len() < k {
        let mut rest: Vec<usize> = pool
            .iter()
            .copied()
            .filter(|&i| ok(i) && !chosen.contains(&i))
            .collect();
        while chosen.len() < k && !rest.is_empty() {
            let j = rng.below(rest.len());
            chosen.push(rest.swap_remove(j));
        }
    }
    chosen
}

/// The windows of a case-time-control or case-case-time-control analysis:
/// those of the case events, with their observation, and those of the
/// controls at their index days, built alike.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseTimeControlWindows {
    /// The windows. Sets `0..n_cases` are the case events, in their order,
    /// and set `n_cases + k` is control `k`.
    pub windows: Windows,
    /// For each window, whether it belongs to a case's set.
    pub case_group: Vec<bool>,
}

/// Builds the windows of the case events and of their controls with one
/// hazard window, lag and referent, as [`build_windows`] does for each.
///
/// `case_observation`, when given, has one entry per case event; the controls'
/// observation is their own, from [`Controls`].
///
/// # Errors
///
/// As [`build_windows`], and when the cases' columns differ in length.
pub fn case_time_control_windows(
    case_day: &[i64],
    case_person: &[u32],
    case_observation: Option<Observation<'_>>,
    controls: &Controls,
    hazard_days: i64,
    lag: i64,
    referent: Referent,
) -> Result<CaseTimeControlWindows> {
    let n = case_day.len();
    if case_person.len() != n
        || case_observation.is_some_and(|o| o.start.len() != n || o.end.len() != n)
    {
        return Err(invalid(
            "the case columns and their observation must have one entry per case",
        ));
    }
    let day: Vec<i64> = case_day.iter().chain(&controls.day).copied().collect();
    let person: Vec<u32> = case_person
        .iter()
        .chain(&controls.person)
        .copied()
        .collect();
    let (start, end): (Vec<i64>, Vec<i64>) = match case_observation {
        Some(o) => (
            o.start.iter().chain(&controls.start).copied().collect(),
            o.end.iter().chain(&controls.end).copied().collect(),
        ),
        None => (
            std::iter::repeat_n(i64::MIN, n)
                .chain(controls.start.iter().copied())
                .collect(),
            std::iter::repeat_n(i64::MAX, n)
                .chain(controls.end.iter().copied())
                .collect(),
        ),
    };
    let windows = build_windows(
        &day,
        &person,
        hazard_days,
        lag,
        referent,
        Some(Observation {
            start: &start,
            end: &end,
        }),
    )?;
    let case_group = windows.set.iter().map(|&s| (s as usize) < n).collect();
    Ok(CaseTimeControlWindows {
        windows,
        case_group,
    })
}
