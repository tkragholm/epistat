//! Civil date arithmetic on day counts, and the split of an interval of days
//! across the calendar years it spans.
//!
//! Dates are counts of days since 1970-01-01, negative before it, in the
//! proleptic Gregorian calendar. The conversions to and from year, month and
//! day are the algorithms of Hinnant \[1\], which are exact over the whole
//! range of `i64` that the intermediate products allow.
//!
//! # Algorithm
//!
//! A Gregorian era is 400 years, or 146 097 days. Dates are counted from
//! 1 March, so that the leap day is the last day of the year. With `y` the
//! year, shifted back by one for January and February, the era `e = floor(y /
//! 400)`, the year of the era `yoe = y - 400 e`, and the day of the shifted
//! year `doy = (153 m' + 2) / 5 + d - 1`, where `m' = m - 3` for `m > 2` and
//! `m + 9` otherwise,
//!
//! ```text
//! doe  = 365 yoe + yoe/4 - yoe/100 + doy        day of the era, 0 to 146 096
//! days = 146 097 e + doe - 719 468              719 468 days from 0000-03-01 to 1970-01-01
//! ```
//!
//! with integer division. [`civil_from_days`] inverts it.
//!
//! The interval `[start, end]` has both days included. Its share of calendar
//! year `y`, optionally clipped below at a floor day, is
//!
//! ```text
//! overlap(y) = max(0, min(end, 31 Dec y) - max(start, 1 Jan y, floor) + 1)
//! length(y)  = 366 if y is a leap year, else 365
//! ```
//!
//! for every year from the year of `start` to the year of `end`. The ratio
//! `overlap / length` prorates an annual value to the part of the year the
//! interval covers. The overlaps of one interval sum to its inclusive length
//! when there is no floor.
//!
//! # SAS epoch
//!
//! SAS stores a date as days since 1960-01-01 and a datetime as seconds since
//! 1960-01-01 00:00:00. The Unix epoch is 3 653 days, or 315 619 200 seconds,
//! later (10 years with 3 leap days, 1960, 1964 and 1968, times 86 400).
//! [`SAS_EPOCH_OFFSET_DAYS`] and [`SAS_EPOCH_OFFSET_SECONDS`] are those
//! offsets. [`days_from_civil_sas`], [`civil_from_days_sas`],
//! [`civil_from_sas_seconds`] and the unit conversions add or subtract them
//! around the functions above, in integer arithmetic.
//!
//! # Edge cases
//!
//! - An `end` in an earlier year than `start` spans no year and gives nothing.
//! - An `end` earlier than `start` in the same year gives that year with an
//!   overlap of zero, which is kept only when `keep_empty` is set.
//! - A floor after the end clips every overlap to zero.
//! - A year that does not fit an `i32` is reported as `i32::MAX`.
//!
//! # References
//!
//! 1. Hinnant H. chrono-Compatible Low-Level Date Algorithms.
//!    howardhinnant.github.io/date_algorithms.html

/// Days from 1970-01-01 to the proleptic Gregorian date `(y, m, d)`, with the
/// month from 1 to 12 and the day from 1.
#[must_use]
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// The proleptic Gregorian `(year, month, day)` of a day count from 1970-01-01.
#[must_use]
pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The calendar year of a day count from 1970-01-01.
#[must_use]
pub fn year_from_days(z: i64) -> i64 {
    civil_from_days(z).0
}

/// Days from 1960-01-01, the SAS date epoch, to 1970-01-01: 3 653.
pub const SAS_EPOCH_OFFSET_DAYS: i64 = 3_653;

/// Seconds from 1960-01-01 00:00:00, the SAS datetime epoch, to
/// 1970-01-01 00:00:00: 315 619 200.
pub const SAS_EPOCH_OFFSET_SECONDS: i64 = SAS_EPOCH_OFFSET_DAYS * 86_400;

/// A SAS date, in days from 1960-01-01, as days from 1970-01-01.
#[must_use]
pub const fn sas_days_to_unix(sas_days: i64) -> i64 {
    sas_days - SAS_EPOCH_OFFSET_DAYS
}

/// A day count from 1970-01-01 as a SAS date, in days from 1960-01-01.
#[must_use]
pub const fn unix_days_to_sas(unix_days: i64) -> i64 {
    unix_days + SAS_EPOCH_OFFSET_DAYS
}

/// A SAS datetime, in seconds from 1960-01-01 00:00:00, as seconds from
/// 1970-01-01 00:00:00.
#[must_use]
pub const fn sas_seconds_to_unix(sas_seconds: i64) -> i64 {
    sas_seconds - SAS_EPOCH_OFFSET_SECONDS
}

/// Seconds from 1970-01-01 00:00:00 as a SAS datetime.
#[must_use]
pub const fn unix_seconds_to_sas(unix_seconds: i64) -> i64 {
    unix_seconds + SAS_EPOCH_OFFSET_SECONDS
}

/// Days from 1960-01-01, the value SAS stores for a date, to the proleptic
/// Gregorian date `(y, m, d)`.
#[must_use]
pub fn days_from_civil_sas(y: i64, m: i64, d: i64) -> i64 {
    unix_days_to_sas(days_from_civil(y, m, d))
}

/// The proleptic Gregorian `(year, month, day)` of a SAS date.
#[must_use]
pub fn civil_from_days_sas(sas_days: i64) -> (i64, i64, i64) {
    civil_from_days(sas_days_to_unix(sas_days))
}

/// The `((year, month, day), seconds_since_midnight)` of a SAS datetime. The
/// seconds are from 0 to 86 399, also before 1960.
#[must_use]
pub fn civil_from_sas_seconds(sas_seconds: i64) -> ((i64, i64, i64), i64) {
    let unix = sas_seconds_to_unix(sas_seconds);
    (
        civil_from_days(unix.div_euclid(86_400)),
        unix.rem_euclid(86_400),
    )
}

/// One calendar year's share of an interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YearOverlap {
    /// The calendar year.
    pub calendar_year: i32,
    /// Days of the interval in this year on or after the floor, both ends
    /// included, and zero rather than negative.
    pub overlap_days: i64,
    /// Length of the calendar year, 365 or 366.
    pub calendar_days: i64,
}

/// Append to `out` the calendar years `[start, end]` spans, with its overlap in
/// each.
///
/// The overlap with year `y` is
/// `min(end, 31 Dec y) - max(start, 1 Jan y, floor) + 1`, clipped at zero. A
/// `None` floor sets no bound. Years with no overlap are appended only when
/// `keep_empty` is set. An `end` in an earlier year than `start` spans no
/// year and appends nothing, and an `end` earlier in the same year appends that
/// year with an overlap of zero.
pub fn calendar_year_overlaps(
    start: i64,
    end: i64,
    floor: Option<i64>,
    keep_empty: bool,
    out: &mut Vec<YearOverlap>,
) {
    let lower = floor.map_or(start, |floor| start.max(floor));
    for year in year_from_days(start)..=year_from_days(end) {
        let calendar_start = days_from_civil(year, 1, 1);
        let calendar_end = days_from_civil(year, 12, 31);
        let overlap_days = (end.min(calendar_end) - lower.max(calendar_start) + 1).max(0);
        if overlap_days > 0 || keep_empty {
            out.push(YearOverlap {
                // A year that does not fit an `i32` is reported as `i32::MAX`.
                calendar_year: i32::try_from(year).unwrap_or(i32::MAX),
                overlap_days,
                calendar_days: calendar_end - calendar_start + 1,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(start: i64, end: i64, floor: Option<i64>, keep_empty: bool) -> Vec<YearOverlap> {
        let mut out = Vec::new();
        calendar_year_overlaps(start, end, floor, keep_empty, &mut out);
        out
    }

    #[test]
    fn epoch_round_trips_known_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 1, 1), 10957);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(18262), (2020, 1, 1));
        assert_eq!(year_from_days(days_from_civil(2019, 6, 15)), 2019);
        assert_eq!(year_from_days(-1), 1969);
    }

    #[test]
    fn leap_year_has_366_days() {
        let whole = |year| {
            split(
                days_from_civil(year, 1, 1),
                days_from_civil(year, 12, 31),
                None,
                false,
            )
        };
        assert_eq!(whole(2020)[0].calendar_days, 366);
        assert_eq!(whole(2020)[0].overlap_days, 366);
        assert_eq!(whole(2021)[0].calendar_days, 365);
        assert_eq!(whole(1900)[0].calendar_days, 365);
        assert_eq!(whole(2000)[0].calendar_days, 366);
    }

    #[test]
    fn an_interval_spanning_two_years_is_split_between_them() {
        let out = split(
            days_from_civil(2020, 7, 1),
            days_from_civil(2021, 6, 30),
            Some(days_from_civil(2020, 1, 1)),
            false,
        );
        assert_eq!(out.len(), 2);
        // Jul 1 .. Dec 31 of a leap year, then Jan 1 .. Jun 30.
        assert_eq!((out[0].calendar_year, out[0].overlap_days), (2020, 184));
        assert_eq!((out[1].calendar_year, out[1].overlap_days), (2021, 181));
    }

    #[test]
    fn the_floor_clips_early_days() {
        let jan1 = days_from_civil(2020, 1, 1);
        let out = split(jan1, days_from_civil(2020, 12, 31), Some(jan1 + 90), false);
        // 2020 is a leap year: 366 days less the 90 before the floor.
        assert_eq!(out[0].overlap_days, 276);
    }

    #[test]
    fn years_before_the_floor_are_kept_only_on_request() {
        let start = days_from_civil(2019, 6, 1);
        let end = days_from_civil(2021, 6, 1);
        let floor = Some(days_from_civil(2020, 3, 1));
        let dropped = split(start, end, floor, false);
        assert_eq!(dropped[0].calendar_year, 2020);
        let kept = split(start, end, floor, true);
        assert_eq!(kept.len(), 3);
        assert_eq!((kept[0].calendar_year, kept[0].overlap_days), (2019, 0));
    }

    #[test]
    fn an_end_before_the_start_spans_nothing_or_an_empty_year() {
        let day = days_from_civil(2020, 5, 5);
        assert_eq!(split(day, days_from_civil(2019, 5, 5), None, true), []);
        let same_year = split(day, day - 3, None, true);
        assert_eq!(same_year.len(), 1);
        assert_eq!(same_year[0].overlap_days, 0);
        assert_eq!(split(day, day - 3, None, false), []);
    }

    #[test]
    fn the_overlaps_sum_to_the_inclusive_length() {
        let start = days_from_civil(1996, 2, 29);
        for length in [0_i64, 1, 305, 366, 1461, 4000] {
            let total: i64 = split(start, start + length, None, false)
                .iter()
                .map(|year| year.overlap_days)
                .sum();
            assert_eq!(total, length + 1);
        }
    }

    #[expect(
        clippy::cast_possible_wrap,
        reason = "a copy of the tool's code, where a month and a day are small"
    )]
    /// `days_since_sas_epoch` of `sas7bdat-writer` (and `days_from_civil` of
    /// `register-delivery`, which is the same arithmetic in another type).
    fn tool_days_since_sas_epoch(year: i32, month: u32, day: u32) -> f64 {
        let adjusted_year = if month <= 2 { year - 1 } else { year };
        let era = if adjusted_year >= 0 {
            adjusted_year
        } else {
            adjusted_year - 399
        } / 400;
        let year_of_era = adjusted_year - era * 400;
        let day_of_year =
            (153 * (if month > 2 { month - 3 } else { month + 9 }) as i32 + 2) / 5 + day as i32 - 1;
        let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
        f64::from(era * 146_097 + day_of_era - 719_468 + 3_653)
    }

    /// `iso_date` of `dst-disclose`, applied to days from 1970-01-01.
    fn tool_iso_date(days: i64) -> String {
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + i64::from(month <= 2);
        format!("{year:04}-{month:02}-{day:02}")
    }

    #[test]
    fn the_sas_epoch_is_3653_days_and_the_tools_agree() {
        assert_eq!(SAS_EPOCH_OFFSET_DAYS, 3_653);
        assert_eq!(SAS_EPOCH_OFFSET_SECONDS, 315_619_200);
        assert_eq!(days_from_civil_sas(1960, 1, 1), 0);
        assert_eq!(days_from_civil_sas(1970, 1, 1), 3_653);
        assert_eq!(civil_from_days_sas(-1), (1959, 12, 31));
        for sas in -40_000..60_000 {
            let (y, m, d) = civil_from_days_sas(sas);
            assert_eq!(days_from_civil_sas(y, m, d), sas);
            let tool = tool_days_since_sas_epoch(
                i32::try_from(y).unwrap(),
                u32::try_from(m).unwrap(),
                u32::try_from(d).unwrap(),
            );
            assert_eq!(tool.to_bits(), (sas as f64).to_bits());
            assert_eq!(
                tool_iso_date(sas_days_to_unix(sas)),
                format!("{y:04}-{m:02}-{d:02}")
            );
        }
    }

    #[test]
    fn sas_seconds_split_into_a_date_and_a_time_of_day() {
        assert_eq!(civil_from_sas_seconds(0), ((1960, 1, 1), 0));
        assert_eq!(
            civil_from_sas_seconds(SAS_EPOCH_OFFSET_SECONDS + 86_399),
            ((1970, 1, 1), 86_399)
        );
        assert_eq!(civil_from_sas_seconds(-1), ((1959, 12, 31), 86_399));
        assert_eq!(sas_seconds_to_unix(unix_seconds_to_sas(-5)), -5);
        assert_eq!(unix_days_to_sas(sas_days_to_unix(77)), 77);
    }
}
