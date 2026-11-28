//! Civil date arithmetic on day counts, and the split of an interval of days
//! across the calendar years it spans.
//!
//! Dates are counts of days since 1970-01-01, negative before it, in the
//! proleptic Gregorian calendar. The conversions to and from year, month and
//! day, and the addition of whole years and months, are those of
//! [`epistat_core::day`], re-exported here under this crate's names:
//! [`days_from_civil`] is `from_civil`, [`civil_from_days`] is `to_civil` and
//! [`year_from_days`] is `year_of`. The algorithm and its reference are in
//! that module.
//!
//! # Algorithm
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
//! offsets. A SAS date read as a Unix day is ten years off and nothing in the
//! number shows it, so SAS dates enter through `epistat_core::Day`, which
//! names the epoch: [`Day::from_sas_days`] for one date,
//! [`Day::from_sas_column`] for a column, and [`day_from_sas_seconds`] for a
//! datetime. The `i64` forms of release 0.1 are deprecated.
//!
//! ```
//! use epistat_core::Day;
//! use epistat_register::person_time::day_from_sas_seconds;
//!
//! let day = Day::from_sas_days(23_435).expect("a day");
//! assert_eq!(day.to_string(), "2024-02-29");
//! assert_eq!(day.unix_days(), 19_782);
//! let (when, seconds) = day_from_sas_seconds(23_435 * 86_400 + 3_600).expect("a day");
//! assert_eq!((when, seconds), (day, 3_600));
//! ```
//!
//! # Edge cases
//!
//! - An `end` in an earlier year than `start` spans no year and gives nothing.
//! - An `end` earlier than `start` in the same year gives that year with an
//!   overlap of zero, which is kept only when `keep_empty` is set.
//! - A floor after the end clips every overlap to zero.
//!

use epistat_core::Day;

pub use epistat_core::day::year_of as year_from_days;
pub use epistat_core::day::{add_months, add_years};
pub use epistat_core::day::{from_civil as days_from_civil, to_civil as civil_from_days};

/// Days from 1960-01-01, the SAS date epoch, to 1970-01-01: 3 653.
pub const SAS_EPOCH_OFFSET_DAYS: i64 = 3_653;

/// Seconds from 1960-01-01 00:00:00, the SAS datetime epoch, to
/// 1970-01-01 00:00:00: 315 619 200.
pub const SAS_EPOCH_OFFSET_SECONDS: i64 = SAS_EPOCH_OFFSET_DAYS * 86_400;

/// A SAS date, in days from 1960-01-01, as days from 1970-01-01.
#[deprecated(
    since = "0.2.0",
    note = "use epistat_core::Day::from_sas_days(d).map(Day::unix_days), or Day::from_sas_column"
)]
#[must_use]
pub const fn sas_days_to_unix(sas_days: i64) -> i64 {
    sas_days - SAS_EPOCH_OFFSET_DAYS
}

/// A day count from 1970-01-01 as a SAS date, in days from 1960-01-01.
#[deprecated(since = "0.2.0", note = "use epistat_core::Day::sas_days")]
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
#[deprecated(
    since = "0.2.0",
    note = "use epistat_core::Day::from_civil(y, m, d).map(Day::sas_days), which refuses a date that does not exist"
)]
#[must_use]
pub fn days_from_civil_sas(y: i64, m: i64, d: i64) -> i64 {
    days_from_civil(y, m, d) + SAS_EPOCH_OFFSET_DAYS
}

/// The proleptic Gregorian `(year, month, day)` of a SAS date.
#[deprecated(
    since = "0.2.0",
    note = "use epistat_core::Day::from_sas_days(d).map(Day::to_civil)"
)]
#[must_use]
pub fn civil_from_days_sas(sas_days: i64) -> (i64, i64, i64) {
    civil_from_days(sas_days - SAS_EPOCH_OFFSET_DAYS)
}

/// The day of a SAS datetime, in seconds from 1960-01-01 00:00:00, and the
/// seconds since midnight of that day, from 0 to 86 399, also before 1960.
/// `None` for a day beyond [`Day::MIN`] to [`Day::MAX`].
#[must_use]
pub fn day_from_sas_seconds(sas_seconds: i64) -> Option<(Day, i64)> {
    let unix = sas_seconds_to_unix(sas_seconds);
    Some((
        Day::from_unix_days(unix.div_euclid(86_400))?,
        unix.rem_euclid(86_400),
    ))
}

/// The `((year, month, day), seconds_since_midnight)` of a SAS datetime. The
/// seconds are from 0 to 86 399, also before 1960.
#[deprecated(
    since = "0.2.0",
    note = "use day_from_sas_seconds, which returns the day as an epistat_core::Day"
)]
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
#[non_exhaustive]
pub struct YearOverlap {
    /// The calendar year.
    pub calendar_year: i64,
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
                calendar_year: year,
                overlap_days,
                calendar_days: calendar_end - calendar_start + 1,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn adding_years_and_months() {
        use crate::LeapBirthday::{February28, March1};
        let leap_day = days_from_civil(2000, 2, 29);
        assert_eq!(add_years(leap_day, 4, March1), days_from_civil(2004, 2, 29));
        assert_eq!(add_years(leap_day, 18, March1), days_from_civil(2018, 3, 1));
        assert_eq!(
            add_years(leap_day, 18, February28),
            days_from_civil(2018, 2, 28)
        );
        assert_eq!(
            add_years(days_from_civil(2020, 5, 17), -3, March1),
            days_from_civil(2017, 5, 17)
        );
        let jan31 = days_from_civil(2021, 1, 31);
        assert_eq!(add_months(jan31, 1), days_from_civil(2021, 2, 28));
        assert_eq!(add_months(jan31, 13), days_from_civil(2022, 2, 28));
        assert_eq!(
            add_months(days_from_civil(2020, 1, 31), 1),
            days_from_civil(2020, 2, 29)
        );
        assert_eq!(add_months(jan31, -2), days_from_civil(2020, 11, 30));
        assert_eq!(add_months(jan31, 0), jan31);
    }

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
        assert_eq!(SAS_EPOCH_OFFSET_DAYS, epistat_core::day::SAS_OFFSET_DAYS);
        assert_eq!(SAS_EPOCH_OFFSET_SECONDS, 315_619_200);
        let sas_of = |y, m, d| Day::from_civil(y, m, d).unwrap().sas_days();
        assert_eq!(sas_of(1960, 1, 1), 0);
        assert_eq!(sas_of(1970, 1, 1), 3_653);
        assert_eq!(Day::from_sas_days(-1).unwrap().to_civil(), (1959, 12, 31));
        for sas in -40_000..60_000 {
            let day = Day::from_sas_days(sas).unwrap();
            let (y, m, d) = day.to_civil();
            assert_eq!(sas_of(y, m, d), sas);
            let tool = tool_days_since_sas_epoch(
                i32::try_from(y).unwrap(),
                u32::try_from(m).unwrap(),
                u32::try_from(d).unwrap(),
            );
            assert_eq!(tool.to_bits(), (sas as f64).to_bits());
            assert_eq!(
                tool_iso_date(day.unix_days()),
                format!("{y:04}-{m:02}-{d:02}")
            );
            assert_eq!(day.to_string(), format!("{y:04}-{m:02}-{d:02}"));
        }
    }

    #[test]
    fn sas_seconds_split_into_a_day_and_a_time_of_day() {
        let day = |y, m, d| Day::from_civil(y, m, d).unwrap();
        assert_eq!(day_from_sas_seconds(0), Some((day(1960, 1, 1), 0)));
        assert_eq!(
            day_from_sas_seconds(SAS_EPOCH_OFFSET_SECONDS + 86_399),
            Some((day(1970, 1, 1), 86_399))
        );
        assert_eq!(day_from_sas_seconds(-1), Some((day(1959, 12, 31), 86_399)));
        assert_eq!(day_from_sas_seconds(i64::MAX), None);
        assert_eq!(sas_seconds_to_unix(unix_seconds_to_sas(-5)), -5);
    }

    #[test]
    #[expect(
        deprecated,
        reason = "the 0.1 forms keep their results for one release"
    )]
    fn the_deprecated_sas_forms_agree_with_day() {
        for sas in [-40_000, -1, 0, 3_653, 23_435, 60_000] {
            let day = Day::from_sas_days(sas).unwrap();
            assert_eq!(civil_from_days_sas(sas), day.to_civil());
            let (y, m, d) = day.to_civil();
            assert_eq!(days_from_civil_sas(y, m, d), sas);
            assert_eq!(sas_days_to_unix(sas), day.unix_days());
            assert_eq!(unix_days_to_sas(day.unix_days()), sas);
            let seconds = sas * 86_400 + 7;
            let (when, time) = day_from_sas_seconds(seconds).unwrap();
            assert_eq!(civil_from_sas_seconds(seconds), (when.to_civil(), time));
        }
    }
}
