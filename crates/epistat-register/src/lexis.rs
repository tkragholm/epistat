//! Cutting follow-up at calendar dates, birthdays and other instants, on
//! integer days.
//!
//! A follow-up interval `[start, end)` of days is cut into pieces that each
//! lie within one calendar year, one age year, one period, or one stretch
//! since entry, so that person-time can be tabulated or a rate model fitted
//! on the pieces. This is Lexis expansion, as `Epi::splitLexis` and
//! `popEpi::splitMulti` do it in R, with two differences.
//!
//! - The scales are integer days, so a cut is an exact day and the pieces tile
//!   the interval with no rounding.
//! - An age cut is a civil birthday: [`crate::age::birthday_on_age`] gives the
//!   day on which a person reaches an age under [`AgeRule`], so 29 February
//!   births are handled by the named convention, and a person born on
//!   1 January 2000 is 18 on 1 January 2018, 6 574 days later, not on the day
//!   `18 * 365.25` falls on. In R the age scale is fractional years that
//!   advance at unit speed, and a birthday is `birth + 18` years.
//!
//! `epistat-survival` has a Lexis splitter on `f64` scales that advance at
//! unit speed (`split_lexis`), which suits scales that are not dates. The two are used side by side: this module produces
//! pieces on dates and ages, and the survival one splits scales of any
//! other kind.
//!
//! # Algorithm
//!
//! For each row, the instants of every [`Cuts`] source that fall strictly
//! inside `(start, end)` are gathered, sorted and made distinct, and the
//! interval is cut at each. A piece is `[a, b)`, so an instant belongs to the
//! piece that starts at it. The calendar-year cuts are the 1 January of each
//! year after the year of `start` up to the year of `end - 1`. An age cut
//! for age `a` is `birthday_on_age(birth, a, rule)`. The work is
//! `O(pieces)` plus the sort of the instants of one row.
//!
//! # Edge cases
//!
//! - An empty interval (`start == end`) gives no piece. An interval with
//!   `end < start` is refused, as is a missing cut column of the wrong length.
//! - A cut equal to `start` or to `end` gives no extra piece.
//! - An age whose birthday does not fit in an `i64` is refused.
//! - Pieces of one row are in time order and the rows are in input order.
//! - Missing end of follow-up (open follow-up) is for the caller to close with
//!   an explicit end day before the call.
//!
//! # Labels
//!
//! [`labels`] gives each piece the left end of its band on each source: the
//! calendar year, the age reached, the period or the time since entry, as
//! `Epi::timeBand(x, scale, "left")` does after `Epi::splitLexis`. A piece
//! that starts before the first cut of a source has no label on it.
//!
//! # Validation
//!
//! The pieces are compared with `Epi::splitLexis` and `popEpi::splitMulti` in
//! `tests/lexis_r.rs`, with calendar years and with ages on the
//! [`AgeRule::DaysOver365_25`] scale, where R's day scales are exact, and the
//! labels with `Epi::timeBand` on the same pieces.

use crate::Error;
use crate::age::{AgeRule, birthday_days};
use crate::error::{csr, length, row_count};
use crate::person_time::{days_from_civil, year_from_days};

/// A source of cut instants for [`split_at`].
#[derive(Clone, Copy, Debug)]
pub enum Cuts<'a> {
    /// Each 1 January.
    CalendarYears,
    /// The day each person reaches each of `ages` whole years: `birth` has one
    /// day per row.
    AgeYears {
        /// Birth day of each row.
        birth: &'a [i64],
        /// The ages, in years, at which to cut, the same for every row.
        ages: &'a [u32],
        /// How an age is reckoned.
        rule: AgeRule,
    },
    /// The same days for every row.
    CalendarPeriods(&'a [i64]),
    /// `entry + offset` for each offset, with one entry day per row.
    TimeSinceEntry {
        /// Entry day of each row.
        entry: &'a [i64],
        /// Days since entry at which to cut, the same for every row.
        offsets: &'a [i64],
    },
    /// A list of cuts per row: row `r` has `at[offsets[r]..offsets[r + 1]]`.
    PerRow {
        /// CSR offsets, one more than there are rows.
        offsets: &'a [usize],
        /// The cut days.
        at: &'a [i64],
    },
}

/// The pieces of a split, in row order and in time order within a row.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Split {
    /// The input row of each piece.
    pub source_row: Vec<u32>,
    /// First day of each piece.
    pub start: Vec<i64>,
    /// One past the last day of each piece.
    pub end: Vec<i64>,
}

/// Cuts each interval `[start, end)` at the instants of `cuts`. A cut belongs
/// to the later piece.
///
/// # Errors
///
/// [`Error::Input`] when the columns differ in length, there are more rows
/// than a `u32` holds or a cut column does not have one entry per row,
/// [`Error::EndBeforeStart`] when an interval ends before it starts, and
/// [`Error::DayOverflow`] when an age has no birthday that fits in an
/// `i64`.
pub fn split_at(start: &[i64], end: &[i64], cuts: &[Cuts<'_>]) -> Result<Split, Error> {
    let mut out = Split::default();
    split_at_into(start, end, cuts, |row, s, e| {
        out.source_row.push(row);
        out.start.push(s);
        out.end.push(e);
    })?;
    Ok(out)
}

/// As [`split_at`], handing each piece to `emit(row, start, end)` instead of
/// collecting them.
///
/// # Errors
///
/// As [`split_at`]. Pieces handed out before an error is found are not
/// withdrawn, so a caller that needs all or nothing validates first with
/// [`split_at`] or collects.
pub fn split_at_into(
    start: &[i64],
    end: &[i64],
    cuts: &[Cuts<'_>],
    mut emit: impl FnMut(u32, i64, i64),
) -> Result<(), Error> {
    let n = start.len();
    length("end", n, end.len())?;
    let n32 = row_count("start", n)?;
    for cut in cuts {
        check_cut(cut, n)?;
    }
    if let Some(row) = (0..n).find(|&r| end[r] < start[r]) {
        return Err(Error::EndBeforeStart { what: "end", row });
    }
    let mut at: Vec<i64> = Vec::new();
    for row in 0..n32 {
        let r = row as usize;
        let (s, e) = (start[r], end[r]);
        if s == e {
            continue;
        }
        at.clear();
        for cut in cuts {
            gather(cut, r, s, e, &mut at)?;
        }
        at.sort_unstable();
        at.dedup();
        let mut from = s;
        for &c in &at {
            emit(row, from, c);
            from = c;
        }
        emit(row, from, e);
    }
    Ok(())
}

/// The label of each piece on each cut source: the left end of the band the
/// piece lies in, as `Epi::timeBand(x, scale, "left")` gives it after
/// `Epi::splitLexis`.
///
/// `cuts` are the sources the split was made with, in the same order, and the
/// result has one column per source with one value per piece.
///
/// - [`Cuts::CalendarYears`]: the calendar year of the piece.
/// - [`Cuts::AgeYears`]: the largest of `ages` that the person has reached on
///   the first day of the piece, under the rule of the cut.
/// - [`Cuts::CalendarPeriods`]: the latest cut day on or before the first day
///   of the piece.
/// - [`Cuts::TimeSinceEntry`]: the largest offset with `entry + offset` on or
///   before the first day of the piece.
/// - [`Cuts::PerRow`]: the latest of the row's cut days on or before the first
///   day of the piece.
///
/// A piece that starts before the first cut of a source is [`None`] on it, as
/// R gives `NA` below the first break. Every day of a piece has the label of
/// its first day, because the split cut the interval at every instant where a
/// label changes, so a piece is in one band of each source. To label each
/// piece with the completed age in years, list every age from the youngest in
/// the data, as the ages that fall before a row's start add no piece.
///
/// # Errors
///
/// [`Error::Input`] when the columns of the split differ in length,
/// [`Error::OutOfRange`] when a piece names a row beyond a cut source, and
/// [`Error::DayOverflow`] when an age has no birthday that fits in an
/// `i64`.
pub fn labels(split: &Split, cuts: &[Cuts<'_>]) -> Result<Vec<Vec<Option<i64>>>, Error> {
    let pieces = split.start.len();
    length("source_row", pieces, split.source_row.len())?;
    length("end", pieces, split.end.len())?;
    let latest = |values: &mut dyn Iterator<Item = (i64, i64)>, day: i64| {
        values
            .filter(|&(at, _)| at <= day)
            .max_by_key(|&(at, label)| (at, label))
            .map(|(_, label)| label)
    };
    let rows_needed = split
        .source_row
        .iter()
        .map(|&r| r as usize + 1)
        .max()
        .unwrap_or(0);
    let mut out = Vec::with_capacity(cuts.len());
    for cut in cuts {
        let rows = match *cut {
            Cuts::AgeYears { birth, .. } => Some(("birth", birth.len())),
            Cuts::TimeSinceEntry { entry, .. } => Some(("entry", entry.len())),
            Cuts::PerRow { offsets, .. } => Some(("cut offsets", offsets.len().saturating_sub(1))),
            _ => None,
        };
        if let Some((what, limit)) = rows
            && rows_needed > limit
            && let Some(piece) = split.source_row.iter().position(|&r| r as usize >= limit)
        {
            return Err(Error::OutOfRange {
                what,
                row: Some(piece),
                value: u64::from(split.source_row[piece]),
                limit: limit as u64,
            });
        }
        if let Cuts::PerRow { offsets, at } = *cut {
            csr("cut offsets", offsets, at.len())?;
        }
        let mut column = Vec::with_capacity(pieces);
        for i in 0..pieces {
            let (row, day) = (split.source_row[i] as usize, split.start[i]);
            let label = match *cut {
                Cuts::CalendarYears => Some(year_from_days(day)),
                Cuts::AgeYears { birth, ages, rule } => {
                    let mut reached = None;
                    for &age in ages {
                        let at =
                            birthday_days(birth[row], age, rule).ok_or(Error::DayOverflow {
                                what: "birthday",
                                row,
                            })?;
                        if at <= day {
                            reached = reached.max(Some(i64::from(age)));
                        }
                    }
                    reached
                }
                Cuts::CalendarPeriods(days) => latest(&mut days.iter().map(|&c| (c, c)), day),
                Cuts::TimeSinceEntry { entry, offsets } => latest(
                    &mut offsets
                        .iter()
                        .filter_map(|&o| entry[row].checked_add(o).map(|c| (c, o))),
                    day,
                ),
                Cuts::PerRow { offsets, at } => latest(
                    &mut at[offsets[row]..offsets[row + 1]].iter().map(|&c| (c, c)),
                    day,
                ),
            };
            column.push(label);
        }
        out.push(column);
    }
    Ok(out)
}

fn check_cut(cut: &Cuts<'_>, n: usize) -> Result<(), Error> {
    match *cut {
        Cuts::CalendarYears | Cuts::CalendarPeriods(_) => Ok(()),
        Cuts::AgeYears { birth, .. } => length("birth", n, birth.len()),
        Cuts::TimeSinceEntry { entry, .. } => length("entry", n, entry.len()),
        Cuts::PerRow { offsets, at } => {
            length("cut offsets", n + 1, offsets.len())?;
            csr("cut offsets", offsets, at.len())
        }
    }
}

/// Appends the cuts of one source that lie strictly inside `(s, e)`.
fn gather(cut: &Cuts<'_>, row: usize, s: i64, e: i64, at: &mut Vec<i64>) -> Result<(), Error> {
    let mut keep = |c: i64| {
        if c > s && c < e {
            at.push(c);
        }
    };
    match *cut {
        Cuts::CalendarYears => {
            for year in year_from_days(s) + 1..=year_from_days(e - 1) {
                keep(days_from_civil(year, 1, 1));
            }
        }
        Cuts::AgeYears { birth, ages, rule } => {
            for &age in ages {
                let day = birthday_days(birth[row], age, rule).ok_or(Error::DayOverflow {
                    what: "birthday",
                    row,
                })?;
                keep(day);
            }
        }
        Cuts::CalendarPeriods(days) => days.iter().for_each(|&c| keep(c)),
        Cuts::TimeSinceEntry { entry, offsets } => {
            for &offset in offsets {
                if let Some(c) = entry[row].checked_add(offset) {
                    keep(c);
                }
            }
        }
        Cuts::PerRow { offsets, at: list } => {
            list[offsets[row]..offsets[row + 1]]
                .iter()
                .for_each(|&c| keep(c));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::cast_possible_wrap,
        reason = "small whole numbers in test data"
    )]

    use super::*;
    use crate::age::{LeapBirthday, age_in_years};

    const MARCH1: AgeRule = AgeRule::CivilYears(LeapBirthday::March1);
    const FEB28: AgeRule = AgeRule::CivilYears(LeapBirthday::February28);

    fn d(y: i64, m: i64, day: i64) -> i64 {
        days_from_civil(y, m, day)
    }

    fn pieces(s: &Split) -> Vec<(u32, i64, i64)> {
        (0..s.start.len())
            .map(|i| (s.source_row[i], s.start[i], s.end[i]))
            .collect()
    }

    #[test]
    fn calendar_years_cut_at_new_year() {
        let s = split_at(&[d(2019, 11, 1)], &[d(2022, 2, 1)], &[Cuts::CalendarYears]).unwrap();
        assert_eq!(
            pieces(&s),
            [
                (0, d(2019, 11, 1), d(2020, 1, 1)),
                (0, d(2020, 1, 1), d(2021, 1, 1)),
                (0, d(2021, 1, 1), d(2022, 1, 1)),
                (0, d(2022, 1, 1), d(2022, 2, 1)),
            ]
        );
    }

    #[test]
    fn a_cut_at_start_or_end_adds_no_piece() {
        let s = split_at(
            &[d(2020, 1, 1)],
            &[d(2021, 1, 1)],
            &[
                Cuts::CalendarYears,
                Cuts::CalendarPeriods(&[d(2020, 1, 1), d(2021, 1, 1)]),
            ],
        )
        .unwrap();
        assert_eq!(pieces(&s), [(0, d(2020, 1, 1), d(2021, 1, 1))]);
    }

    #[test]
    fn follow_up_that_starts_on_a_birthday() {
        let birth = [d(2000, 3, 5)];
        let s = split_at(
            &[d(2018, 3, 5)],
            &[d(2020, 3, 5)],
            &[Cuts::AgeYears {
                birth: &birth,
                ages: &[18, 19, 20],
                rule: MARCH1,
            }],
        )
        .unwrap();
        assert_eq!(
            pieces(&s),
            [
                (0, d(2018, 3, 5), d(2019, 3, 5)),
                (0, d(2019, 3, 5), d(2020, 3, 5))
            ]
        );
    }

    #[test]
    fn labels_are_the_left_ends_of_the_bands() {
        // Born 29 February 2000, followed from 1 June 2017 to 31 May 2020.
        let birth = [d(2000, 2, 29)];
        let cuts = [
            Cuts::CalendarYears,
            Cuts::AgeYears {
                birth: &birth,
                ages: &[17, 18, 19, 20],
                rule: MARCH1,
            },
            Cuts::CalendarPeriods(&[d(2019, 1, 1)]),
            Cuts::TimeSinceEntry {
                entry: &[d(2017, 6, 1)],
                offsets: &[0, 365],
            },
        ];
        let s = split_at(&[d(2017, 6, 1)], &[d(2020, 6, 1)], &cuts).unwrap();
        let l = labels(&s, &cuts).unwrap();
        assert_eq!(s.start.len(), 8);
        let years: Vec<i64> = l[0].iter().map(|y| y.unwrap()).collect();
        assert_eq!(years, [2017, 2018, 2018, 2018, 2019, 2019, 2020, 2020]);
        let ages: Vec<i64> = l[1].iter().map(|a| a.unwrap()).collect();
        assert_eq!(ages, [17, 17, 18, 18, 18, 19, 19, 20]);
        // The age label is the completed age on the first day of each piece.
        for (i, &age) in ages.iter().enumerate() {
            assert_eq!(
                age_in_years(birth[0], s.start[i], MARCH1).map(i64::from),
                Some(age)
            );
        }
        assert_eq!(l[2][..4], [None, None, None, None]);
        assert_eq!(l[2][4..], [Some(d(2019, 1, 1)); 4]);
        assert_eq!(l[3][..3], [Some(0); 3]);
        assert_eq!(l[3][3..], [Some(365); 5]);
        // Ages that start above the follow-up leave the first pieces unlabelled.
        let late = [Cuts::AgeYears {
            birth: &birth,
            ages: &[19],
            rule: MARCH1,
        }];
        let s = split_at(&[d(2017, 6, 1)], &[d(2020, 6, 1)], &late).unwrap();
        assert_eq!(labels(&s, &late).unwrap()[0], [None, Some(19)]);
        let short: [i64; 0] = [];
        let bad = [Cuts::AgeYears {
            birth: &short,
            ages: &[19],
            rule: MARCH1,
        }];
        assert!(labels(&s, &bad).is_err());
    }

    #[test]
    fn a_leap_day_birth_under_both_conventions() {
        let birth = [d(2000, 2, 29)];
        let cut = |rule| {
            let s = split_at(
                &[d(2017, 1, 1)],
                &[d(2020, 1, 1)],
                &[Cuts::AgeYears {
                    birth: &birth,
                    ages: &[17, 18, 19, 20],
                    rule,
                }],
            )
            .unwrap();
            s.start[1..].to_vec()
        };
        // 17 on 1 March 2017, 18 on 1 March 2018, 19 on 1 March 2019.
        assert_eq!(cut(MARCH1), [d(2017, 3, 1), d(2018, 3, 1), d(2019, 3, 1)]);
        // The convention of moving 29 February to 28 February in a common year.
        assert_eq!(cut(FEB28), [d(2017, 2, 28), d(2018, 2, 28), d(2019, 2, 28)]);
    }

    #[test]
    fn the_birthday_is_the_first_day_of_the_age() {
        for rule in [MARCH1, FEB28, AgeRule::DaysOver365_25] {
            for birth in d(1996, 1, 1)..d(2005, 1, 1) {
                for age in 0..=105_u32 {
                    let day = crate::age::birthday_days(birth, age, rule).unwrap();
                    assert_eq!(age_in_years(birth, day, rule), Some(age));
                    if age > 0 {
                        assert_eq!(age_in_years(birth, day - 1, rule), Some(age - 1));
                    }
                }
            }
        }
    }

    #[test]
    fn pieces_tile_each_interval_and_never_cross_a_cut() {
        let mut state = 7_u64;
        let mut next = move |m: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % m
        };
        let n = 300;
        let birth: Vec<i64> = (0..n)
            .map(|_| d(1950, 1, 1) + next(20_000) as i64)
            .collect();
        let start: Vec<i64> = birth.iter().map(|b| b + next(9_000) as i64).collect();
        let end: Vec<i64> = start.iter().map(|s| s + next(9_000) as i64).collect();
        let periods = [d(2000, 7, 1), d(2010, 7, 1)];
        let cuts = [
            Cuts::CalendarYears,
            Cuts::AgeYears {
                birth: &birth,
                ages: &[5, 18, 40, 65],
                rule: MARCH1,
            },
            Cuts::CalendarPeriods(&periods),
            Cuts::TimeSinceEntry {
                entry: &start,
                offsets: &[30, 365],
            },
        ];
        let s = split_at(&start, &end, &cuts).unwrap();
        let mut length = vec![0_i64; n];
        let mut last_end = start.clone();
        for (row, a, b) in pieces(&s) {
            let r = row as usize;
            assert!(a < b);
            assert_eq!(a, last_end[r], "pieces are contiguous");
            last_end[r] = b;
            length[r] += b - a;
            // No cut strictly inside a piece.
            assert_eq!(year_from_days(a), year_from_days(b - 1));
            for age in [5, 18, 40, 65] {
                let c = crate::age::birthday_days(birth[r], age, MARCH1).unwrap();
                assert!(!(a < c && c < b));
            }
            assert!(!periods.iter().any(|&c| a < c && c < b));
            for off in [30, 365] {
                assert!(!(a < start[r] + off && start[r] + off < b));
            }
        }
        for r in 0..n {
            assert_eq!(length[r], end[r] - start[r]);
            if end[r] > start[r] {
                assert_eq!(last_end[r], end[r]);
            }
        }
    }

    #[test]
    fn per_row_cuts_and_refusals() {
        let s = split_at(
            &[0, 0],
            &[10, 10],
            &[Cuts::PerRow {
                offsets: &[0, 2, 2],
                at: &[3, 3],
            }],
        )
        .unwrap();
        assert_eq!(pieces(&s), [(0, 0, 3), (0, 3, 10), (1, 0, 10)]);
        assert!(split_at(&[5], &[4], &[]).is_err());
        assert!(split_at(&[5], &[5, 6], &[]).is_err());
        assert!(
            split_at(
                &[0],
                &[9],
                &[Cuts::AgeYears {
                    birth: &[],
                    ages: &[1],
                    rule: MARCH1
                }]
            )
            .is_err()
        );
        assert!(
            split_at(
                &[0],
                &[9],
                &[Cuts::PerRow {
                    offsets: &[0, 3],
                    at: &[1]
                }]
            )
            .is_err()
        );
        assert_eq!(
            split_at(&[3], &[3], &[Cuts::CalendarYears]).unwrap().start,
            Vec::<i64>::new()
        );
    }
}
