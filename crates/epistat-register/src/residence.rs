//! Residence spells from dated emigrations and immigrations, and the end of
//! observation.
//!
//! Whether a person lived in the country on a date depends on the order of
//! their moves, not on the first or last. [`residence_spells`] turns each
//! person's dated events of two kinds into closed periods of residence, and
//! [`resident_on`] tests dates against them. It is general for any register
//! of arrivals and departures. The mapping of a register's own codes to
//! [`Direction`] is the caller's.
//!
//! # Rules
//!
//! 1. Events are read per person in order of day. Days must not decrease.
//! 2. A day with both an `Out` and an `In` is a trip that began and ended that
//!    day. Under [`SameDay::Cancel`] all events of that day are dropped and the
//!    person is in the state they had before it. [`SameDay::KeepOut`] and
//!    [`SameDay::KeepIn`] read the day as that one direction instead. The
//!    registers do not order events within a day, so no reading is the
//!    register's own, and the result of a change of reading is a change of who
//!    is censored; the caller names it.
//! 3. A run of events in the same direction is collapsed to its first, so
//!    directions alternate. Two departures with no return between them are
//!    one departure recorded twice.
//! 4. Every `In` opens a spell that the next event, an `Out`, closes, or that
//!    stays open. A first event that is an `Out` closes a spell that began at
//!    an unknown time, so a person who first emigrates was resident before.
//!    A person whose first event is an `In` was not resident before it.
//!
//! Spells are closed at both ends: the day of the emigration is a day of
//! residence, and the day of the immigration is too. A person with no usable
//! event has no spell and [`resident_on`] reads them as resident on every day,
//! the usual reading of the Civil Registration System. An unbounded start or
//! end is `None`, not a sentinel day.
//!
//! # Observation end
//!
//! [`observation_end`] is the earliest of death, emigration, a cut of the data
//! and the end of the study. A tie is reported by the fixed precedence
//! [`EndReason::Death`], [`EndReason::Emigration`], [`EndReason::DataCut`],
//! [`EndReason::Administrative`], so the same data always give the same reason.
//!
//! # Algorithm
//!
//! One pass per person over the events: `O(events)`. [`resident_on`] is a
//! binary search per query in the person's spells.
//!
//! # References
//!
//! 1. Pedersen CB. The Danish Civil Registration System. Scand J Public Health.
//!    2011;39(7 Suppl):22-5.
//! 2. Schmidt M, Pedersen L, Sorensen HT. The Danish Civil Registration System
//!    as a tool in epidemiology. Eur J Epidemiol. 2014;29(8):541-9.

use crate::RegisterError;
use crate::intervals::SpanSet;

/// The direction of a move.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Leaving.
    Out,
    /// Arriving.
    In,
}

/// Dated moves of persons, in CSR layout.
#[derive(Clone, Copy, Debug)]
pub struct Events<'a> {
    /// Event range of each person, of length persons plus one.
    pub person_offsets: &'a [usize],
    /// The day of each event; within a person it must not decrease.
    pub day: &'a [i64],
    /// The direction of each event.
    pub direction: &'a [Direction],
}

/// How a day with both an emigration and an immigration of one person is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SameDay {
    /// A trip that began and ended that day: every event of the day is dropped.
    Cancel,
    /// As one `Out`.
    KeepOut,
    /// As one `In`.
    KeepIn,
}

/// Closed periods of residence per person, sorted by start with an unknown
/// start first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Spells {
    /// Spell range of each person, of length persons plus one.
    pub person_offsets: Vec<usize>,
    /// First day of residence, or `None` for residence from an unknown time.
    pub from: Vec<Option<i64>>,
    /// Last day of residence, or `None` for residence that has not ended.
    pub to: Vec<Option<i64>>,
}

impl Spells {
    /// The days of residence of each person inside the window
    /// `[window_start[p], window_end[p])`, as a half-open [`SpanSet`].
    ///
    /// An unknown start is the start of the window, a spell that has not ended
    /// runs to the end of the window, and a closed last day `d` becomes the
    /// half-open end `d + 1`. A person with no spell is resident in the whole
    /// window, as [`resident_on`] reads them, so their set is the window and
    /// not empty.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when the window columns do not have one entry
    /// per person, a window ends before it starts, or the spells are not a
    /// layout of sorted, disjoint spells, which [`residence_spells`] always
    /// gives.
    pub fn span_set(
        &self,
        window_start: &[i64],
        window_end: &[i64],
    ) -> Result<SpanSet, RegisterError> {
        let persons = self.person_offsets.len().saturating_sub(1);
        if window_start.len() != persons || window_end.len() != persons {
            return Err(RegisterError::input(format!(
                "the window needs one entry per person, {persons}; got {} and {}",
                window_start.len(),
                window_end.len()
            )));
        }
        if self.from.len() != self.to.len() {
            return Err(RegisterError::input(
                "from and to must have one entry per spell",
            ));
        }
        // A person with no spell is given one spell open at both ends.
        let mut from = Vec::with_capacity(self.from.len() + persons);
        let mut to = Vec::with_capacity(self.to.len() + persons);
        let mut offsets = Vec::with_capacity(persons + 1);
        offsets.push(0);
        for p in 0..persons {
            let (a, b) = (self.person_offsets[p], self.person_offsets[p + 1]);
            if a > b || b > self.from.len() {
                return Err(RegisterError::input(
                    "person offsets must be non-decreasing from 0 to the number of spells",
                ));
            }
            if a == b {
                from.push(None);
                to.push(None);
            } else {
                from.extend_from_slice(&self.from[a..b]);
                to.extend_from_slice(&self.to[a..b]);
            }
            offsets.push(from.len());
        }
        SpanSet::from_closed_within(&from, &to, &offsets, window_start, window_end)
    }
}

impl Events<'_> {
    fn check(&self) -> Result<(), RegisterError> {
        let n = self.day.len();
        if self.direction.len() != n {
            return Err(RegisterError::input(
                "day and direction must have one entry per event",
            ));
        }
        if self.person_offsets.first() != Some(&0)
            || self.person_offsets.last() != Some(&n)
            || self.person_offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(RegisterError::input(
                "person offsets must be non-decreasing from 0 to the number of events",
            ));
        }
        for (person, w) in self.person_offsets.windows(2).enumerate() {
            if self.day[w[0]..w[1]].windows(2).any(|d| d[0] > d[1]) {
                return Err(RegisterError::input(format!(
                    "person {person} has events that are not in order of day"
                )));
            }
        }
        Ok(())
    }

    fn persons(&self) -> usize {
        self.person_offsets.len().saturating_sub(1)
    }
}

/// The usable events of one person, alternating in direction, in order of day.
fn alternating(e: &Events<'_>, person: usize, same_day: SameDay, out: &mut Vec<(i64, Direction)>) {
    out.clear();
    let (lo, hi) = (e.person_offsets[person], e.person_offsets[person + 1]);
    let mut i = lo;
    while i < hi {
        let day = e.day[i];
        let mut j = i;
        let (mut has_out, mut has_in) = (false, false);
        while j < hi && e.day[j] == day {
            match e.direction[j] {
                Direction::Out => has_out = true,
                Direction::In => has_in = true,
            }
            j += 1;
        }
        i = j;
        let direction = match (has_out, has_in, same_day) {
            (true, true, SameDay::Cancel) => continue,
            (true, true, SameDay::KeepOut) | (true, false, _) => Direction::Out,
            _ => Direction::In,
        };
        if out.last().is_none_or(|&(_, last)| last != direction) {
            out.push((day, direction));
        }
    }
}

/// The residence spells of every person.
///
/// # Errors
///
/// [`RegisterError::Input`] when the columns differ in length, the offsets are
/// not a layout over the events, or a person's days decrease.
pub fn residence_spells(e: &Events<'_>, same_day: SameDay) -> Result<Spells, RegisterError> {
    e.check()?;
    let mut out = Spells {
        person_offsets: vec![0],
        ..Spells::default()
    };
    let mut events = Vec::new();
    for person in 0..e.persons() {
        alternating(e, person, same_day, &mut events);
        for (k, &(day, direction)) in events.iter().enumerate() {
            match direction {
                Direction::Out if k == 0 => {
                    out.from.push(None);
                    out.to.push(Some(day));
                }
                Direction::Out => {}
                Direction::In => {
                    out.from.push(Some(day));
                    out.to.push(events.get(k + 1).map(|&(next, _)| next));
                }
            }
        }
        out.person_offsets.push(out.from.len());
    }
    Ok(out)
}

/// Whether each `(person, day)` query falls in a spell of the person, ends
/// included. A person with no spell is resident on every day.
///
/// # Errors
///
/// [`RegisterError::Input`] when the query columns differ in length or a person
/// is not below the number of persons of `spells`.
pub fn resident_on(
    spells: &Spells,
    person: &[u32],
    day: &[i64],
) -> Result<Vec<bool>, RegisterError> {
    if person.len() != day.len() {
        return Err(RegisterError::input(
            "person and day must have one entry per query",
        ));
    }
    let persons = spells.person_offsets.len().saturating_sub(1);
    person
        .iter()
        .zip(day)
        .map(|(&p, &d)| {
            let p = p as usize;
            if p >= persons {
                return Err(RegisterError::input(format!(
                    "person {p} is not below the number of persons, {persons}"
                )));
            }
            let range = spells.person_offsets[p]..spells.person_offsets[p + 1];
            if range.is_empty() {
                return Ok(true);
            }
            let from = &spells.from[range.clone()];
            // Spells are sorted and disjoint, so only the last one that has
            // started by `d` can hold it.
            let started = from.partition_point(|f| f.is_none_or(|f| f <= d));
            Ok(started > 0 && spells.to[range.start + started - 1].is_none_or(|to| d <= to))
        })
        .collect()
}

/// The day of each person's first emigration on or after a day, among the
/// usable events of [`residence_spells`], or `None` when there is none. A
/// second departure with no return between is not counted.
///
/// # Errors
///
/// As [`residence_spells`], and when a person is not below the number of
/// persons or the query columns differ in length.
pub fn first_departure_on_or_after(
    e: &Events<'_>,
    same_day: SameDay,
    person: &[u32],
    day: &[i64],
) -> Result<Vec<Option<i64>>, RegisterError> {
    e.check()?;
    if person.len() != day.len() {
        return Err(RegisterError::input(
            "person and day must have one entry per query",
        ));
    }
    let mut events = Vec::new();
    person
        .iter()
        .zip(day)
        .map(|(&p, &d)| {
            let p = p as usize;
            if p >= e.persons() {
                return Err(RegisterError::input(format!(
                    "person {p} is not below the number of persons, {}",
                    e.persons()
                )));
            }
            alternating(e, p, same_day, &mut events);
            Ok(events
                .iter()
                .find(|&&(day, direction)| direction == Direction::Out && day >= d)
                .map(|&(day, _)| day))
        })
        .collect()
}

/// Why observation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EndReason {
    /// Death.
    Death,
    /// Emigration.
    Emigration,
    /// A date on which the delivered data stop, earlier than the study end.
    DataCut,
    /// The end of the study period.
    Administrative,
}

/// The earliest of death, emigration, a data cut and the study end, with the
/// reason. On a tie the reason is the first of death, emigration, data cut,
/// administrative end that has the earliest day.
#[must_use]
pub fn observation_end(
    study_end: i64,
    death: Option<i64>,
    emigration: Option<i64>,
    cut: Option<i64>,
) -> (i64, EndReason) {
    let mut best = (study_end, EndReason::Administrative);
    for (day, reason) in [
        (cut, EndReason::DataCut),
        (emigration, EndReason::Emigration),
        (death, EndReason::Death),
    ] {
        if let Some(day) = day
            && day <= best.0
        {
            best = (day, reason);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use Direction::{In, Out};

    type Spell = (Option<i64>, Option<i64>);

    #[test]
    fn spells_become_a_span_set_in_the_window() {
        // Person 0 leaves on day 9 and returns on day 20; person 1 has no
        // move; person 2 arrives on day 50, after the window.
        let day = [9, 20, 50];
        let direction = [Out, In, In];
        let e = Events {
            person_offsets: &[0, 2, 2, 3],
            day: &day,
            direction: &direction,
        };
        let spells = residence_spells(&e, SameDay::Cancel).unwrap();
        let set = spells.span_set(&[0; 3], &[30; 3]).unwrap();
        assert_eq!(set.key_offsets, [0, 2, 3, 3]);
        assert_eq!(set.start, [0, 20, 0]);
        assert_eq!(set.end, [10, 30, 30]);
        // Every day in the window agrees with resident_on.
        for p in 0..3_u32 {
            for d in 0..30 {
                let rows = set.key_offsets[p as usize]..set.key_offsets[p as usize + 1];
                let in_set = rows.clone().any(|r| set.start[r] <= d && d < set.end[r]);
                assert_eq!(in_set, resident_on(&spells, &[p], &[d]).unwrap()[0]);
            }
        }
        assert!(spells.span_set(&[0; 2], &[30; 2]).is_err());
    }

    fn run(events: &[&[(i64, Direction)]], same_day: SameDay) -> Vec<Vec<Spell>> {
        let mut offsets = vec![0];
        let (mut day, mut direction) = (Vec::new(), Vec::new());
        for person in events {
            for &(d, dir) in *person {
                day.push(d);
                direction.push(dir);
            }
            offsets.push(day.len());
        }
        let e = Events {
            person_offsets: &offsets,
            day: &day,
            direction: &direction,
        };
        let s = residence_spells(&e, same_day).unwrap();
        (0..events.len())
            .map(|p| {
                (s.person_offsets[p]..s.person_offsets[p + 1])
                    .map(|i| (s.from[i], s.to[i]))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn hand_cases() {
        let got = run(
            &[
                &[(10, Out), (20, In)],
                &[(10, Out), (15, Out), (20, In), (30, Out)],
                &[(10, Out), (10, In)],
                &[(10, In)],
                &[(10, Out)],
                &[],
                &[(5, In), (5, Out), (9, In)],
            ],
            SameDay::Cancel,
        );
        assert_eq!(got[0], [(None, Some(10)), (Some(20), None)]);
        assert_eq!(got[1], [(None, Some(10)), (Some(20), Some(30))]);
        assert_eq!(got[2], []);
        assert_eq!(got[3], [(Some(10), None)]);
        assert_eq!(got[4], [(None, Some(10))]);
        assert_eq!(got[5], []);
        assert_eq!(got[6], [(Some(9), None)]);
    }

    #[test]
    fn the_three_readings_of_a_same_day_pair() {
        let events: &[&[(i64, Direction)]] = &[&[(3, Out), (10, Out), (10, In)]];
        assert_eq!(run(events, SameDay::Cancel)[0], [(None, Some(3))]);
        assert_eq!(run(events, SameDay::KeepOut)[0], [(None, Some(3))]);
        assert_eq!(
            run(events, SameDay::KeepIn)[0],
            [(None, Some(3)), (Some(10), None)]
        );
    }

    #[test]
    fn residence_queries_include_both_ends() {
        let events: &[&[(i64, Direction)]] = &[&[(10, Out), (20, In), (30, Out)], &[], &[(5, In)]];
        let mut offsets = vec![0];
        let (mut day, mut direction) = (Vec::new(), Vec::new());
        for p in events {
            for &(d, dir) in *p {
                day.push(d);
                direction.push(dir);
            }
            offsets.push(day.len());
        }
        let e = Events {
            person_offsets: &offsets,
            day: &day,
            direction: &direction,
        };
        let s = residence_spells(&e, SameDay::Cancel).unwrap();
        let q: Vec<u32> = vec![0; 8];
        let d = [-100, 10, 11, 19, 20, 30, 31, 1000];
        assert_eq!(
            resident_on(&s, &q, &d).unwrap(),
            [true, true, false, false, true, true, false, false]
        );
        assert_eq!(resident_on(&s, &[1, 1], &[-5, 5]).unwrap(), [true, true]);
        assert_eq!(resident_on(&s, &[2, 2], &[4, 5]).unwrap(), [false, true]);
        assert!(resident_on(&s, &[3], &[0]).is_err());
        let f = first_departure_on_or_after(&e, SameDay::Cancel, &[0, 0, 0, 1], &[0, 11, 31, 0])
            .unwrap();
        assert_eq!(f, [Some(10), Some(30), None, None]);
    }

    #[test]
    fn observation_end_ties_follow_the_precedence() {
        assert_eq!(
            observation_end(100, Some(50), Some(50), Some(50)),
            (50, EndReason::Death)
        );
        assert_eq!(
            observation_end(100, None, Some(50), Some(50)),
            (50, EndReason::Emigration)
        );
        assert_eq!(
            observation_end(100, None, None, Some(100)),
            (100, EndReason::DataCut)
        );
        assert_eq!(
            observation_end(100, Some(120), None, None),
            (100, EndReason::Administrative)
        );
    }

    #[test]
    fn refusals() {
        let bad = Events {
            person_offsets: &[0, 2],
            day: &[5, 3],
            direction: &[Out, In],
        };
        assert!(residence_spells(&bad, SameDay::Cancel).is_err());
        let bad = Events {
            person_offsets: &[0, 1],
            day: &[5, 3],
            direction: &[Out, In],
        };
        assert!(residence_spells(&bad, SameDay::Cancel).is_err());
    }
}
