//! Civil dates as day counts: days since 1970-01-01 in the proleptic
//! Gregorian calendar, negative before it.
//!
//! A date column is `i64` days throughout the workspace, the type registers
//! deliver and Arrow's `Date32` widens to. This module converts day counts to
//! and from year, month and day, and adds whole years and months.
//!
//! # Algorithm
//!
//! The conversions are the algorithms of Hinnant \[1\], exact over every day
//! count whose intermediate products fit an `i64`. A Gregorian era is 400
//! years, or 146 097 days. Dates are counted from 1 March, so that the leap
//! day is the last day of the year. With `y` the year, shifted back by one for
//! January and February, the era `e = floor(y / 400)`, the year of the era
//! `yoe = y - 400 e`, and the day of the shifted year
//! `doy = (153 m' + 2) / 5 + d - 1`, where `m' = m - 3` for `m > 2` and
//! `m + 9` otherwise,
//!
//! ```text
//! doe  = 365 yoe + yoe/4 - yoe/100 + doy        day of the era, 0 to 146 096
//! days = 146 097 e + doe - 719 468              719 468 days from 0000-03-01 to 1970-01-01
//! ```
//!
//! with integer division. [`to_civil`] inverts it.
//!
//! A year is a leap year when it is divisible by 4 and not by 100, or by
//! 400. [`add_years`] keeps the month and day; a 29 February with no
//! counterpart in the target year goes to 1 March or to 28 February, as the
//! caller's [`LeapBirthday`] says. [`add_months`] keeps the day of the month
//! or, when the target month is shorter, takes its last day.
//!
//! # References
//!
//! 1. Hinnant H. chrono-Compatible Low-Level Date Algorithms.
//!    howardhinnant.github.io/date_algorithms.html

/// Days from 1970-01-01 to the proleptic Gregorian date `(y, m, d)`, with the
/// month from 1 to 12 and the day from 1.
#[must_use]
pub const fn from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The proleptic Gregorian date `(year, month, day)` of a day count from
/// 1970-01-01.
#[must_use]
pub const fn to_civil(day: i64) -> (i64, i64, i64) {
    let z = day + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The calendar year of a day count.
#[must_use]
pub const fn year_of(day: i64) -> i64 {
    to_civil(day).0
}

/// Whether `year` has a 29 February.
#[must_use]
pub const fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// The number of days in month `month` (1 to 12) of `year`.
#[must_use]
pub const fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 => 28 + is_leap_year(year) as i64,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Where a 29 February falls in a year that has none.
///
/// The rule is always named, because two conventions are in use and a
/// cohort's ages depend on which one it uses.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum LeapBirthday {
    /// On 1 March: a year is completed when the month and day compare as not
    /// earlier than the birth month and day.
    March1,
    /// On 28 February, the date that adding whole years to 29 February gives
    /// in most date libraries.
    February28,
}

/// The day `years` calendar years after `day`, on the same month and day; a
/// negative `years` goes back.
///
/// A 29 February that has no counterpart in the target year goes to 1 March
/// or to 28 February, as `leap` says. The result is the day on which a person
/// born on `day` completes `years` years under the same convention.
#[must_use]
pub const fn add_years(day: i64, years: i64, leap: LeapBirthday) -> i64 {
    let (y, m, d) = to_civil(day);
    let target = y + years;
    if m == 2 && d == 29 && !is_leap_year(target) {
        return match leap {
            LeapBirthday::March1 => from_civil(target, 3, 1),
            LeapBirthday::February28 => from_civil(target, 2, 28),
        };
    }
    from_civil(target, m, d)
}

/// The day `months` calendar months after `day`, on the same day of the month
/// or on the last day of the target month when it is shorter; a negative
/// `months` goes back.
#[must_use]
pub const fn add_months(day: i64, months: i64) -> i64 {
    let (y, m, d) = to_civil(day);
    let index = y * 12 + (m - 1) + months;
    let (ty, tm) = (index.div_euclid(12), index.rem_euclid(12) + 1);
    let last = days_in_month(ty, tm);
    from_civil(ty, tm, if d < last { d } else { last })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_dates() {
        assert_eq!(from_civil(1970, 1, 1), 0);
        assert_eq!(from_civil(1969, 12, 31), -1);
        assert_eq!(from_civil(2000, 3, 1), 11_017);
        assert_eq!(from_civil(1960, 1, 1), -3_653);
        assert_eq!(to_civil(19_782), (2024, 2, 29));
        assert_eq!(year_of(-1), 1969);
    }

    /// A day-by-day walk of the calendar from 1599-01-01 to 2401-12-31 agrees
    /// with both conversions on every day.
    #[test]
    fn the_conversions_agree_with_a_walk_of_the_calendar() {
        let mut day = from_civil(1599, 1, 1);
        for year in 1599..=2401 {
            for month in 1..=12 {
                for d in 1..=days_in_month(year, month) {
                    assert_eq!(from_civil(year, month, d), day);
                    assert_eq!(to_civil(day), (year, month, d));
                    day += 1;
                }
            }
        }
    }

    #[test]
    fn conversions_round_trip_far_from_the_epoch() {
        for day in (-200_000_000_i64..200_000_000).step_by(9_973) {
            let (y, m, d) = to_civil(day);
            assert!((1..=12).contains(&m) && (1..=days_in_month(y, m)).contains(&d));
            assert_eq!(from_civil(y, m, d), day);
        }
    }

    #[test]
    fn leap_years() {
        assert!(is_leap_year(2000) && is_leap_year(2024) && is_leap_year(-4));
        assert!(!is_leap_year(1900) && !is_leap_year(2023) && !is_leap_year(2100));
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2023, 4), 30);
    }

    #[test]
    fn whole_years_and_months_keep_the_calendar_date() {
        let leap_day = from_civil(2000, 2, 29);
        assert_eq!(
            add_years(leap_day, 19, LeapBirthday::March1),
            from_civil(2019, 3, 1)
        );
        assert_eq!(
            add_years(leap_day, 19, LeapBirthday::February28),
            from_civil(2019, 2, 28)
        );
        assert_eq!(
            add_years(leap_day, 4, LeapBirthday::March1),
            from_civil(2004, 2, 29)
        );
        assert_eq!(
            add_years(from_civil(2010, 6, 15), -10, LeapBirthday::March1),
            from_civil(2000, 6, 15)
        );
        assert_eq!(
            add_months(from_civil(2023, 1, 31), 1),
            from_civil(2023, 2, 28)
        );
        assert_eq!(
            add_months(from_civil(2024, 1, 31), 1),
            from_civil(2024, 2, 29)
        );
        assert_eq!(
            add_months(from_civil(2024, 3, 15), -15),
            from_civil(2022, 12, 15)
        );
    }
}
