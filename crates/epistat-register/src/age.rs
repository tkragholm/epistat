//! Age in whole years at a date, under a named rule.
//!
//! Dates are counts of days since 1970-01-01, as in [`crate::person_time`].
//!
//! # Rules
//!
//! The rule is always named. There is no default, because two conventions for
//! a 29 February birthday are in use and a cohort's ages depend on which one
//! it uses.
//!
//! [`AgeRule::CivilYears`] counts the birthdays that have passed. With birth
//! date `(by, bm, bd)` and reference date `(y, m, d)`,
//!
//! ```text
//! age = y - by - 1   if (m, d) < (bm, bd)
//! age = y - by       otherwise
//! ```
//!
//! This is exact except for a birth date of 29 February in a year that is not a
//! leap year, where there is no birthday to compare with. [`LeapBirthday`]
//! says where it falls.
//!
//! - [`LeapBirthday::March1`] compares month and day as they are, so the
//!   birthday of a 29 February birth falls on 1 March of a common year. This
//!   is the formula above applied unchanged.
//! - [`LeapBirthday::February28`] moves the birthday to 28 February of a
//!   common year, the date that adding whole years to 29 February gives in
//!   most date libraries.
//!
//! For a person born on 29 February 2000, the age on 28 February 2019 is 18
//! under `March1` and 19 under `February28`. On 1 March 2019 both give 19, and
//! in a leap year both change on 29 February.
//!
//! [`AgeRule::DaysOver365_25`] is `floor(days / 365.25)`, with `days` the
//! number of days from birth to the reference date. It agrees with the civil
//! rules for most dates and differs by one year on some days near a birthday,
//! because a civil year is 365 or 366 days.
//!
//! # Edge cases
//!
//! - A reference date before the birth date gives [`None`].
//! - On the birth date the age is 0 under both rules.

use crate::person_time::civil_from_days;

/// Where the birthday of a person born on 29 February falls in a year that is
/// not a leap year.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LeapBirthday {
    /// On 1 March: the year is completed when the month and day compare as
    /// not earlier than the birth month and day.
    March1,
    /// On 28 February.
    February28,
}

/// The rule for counting whole years of age.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AgeRule {
    /// Birthdays passed, with the given convention for a 29 February birth.
    CivilYears(LeapBirthday),
    /// Days elapsed since birth divided by 365.25, rounded down.
    DaysOver365_25,
}

/// Age in whole years on `at_day` for a person born on `birth_day`, both as
/// days since 1970-01-01.
///
/// Returns [`None`] when `at_day` is before `birth_day`.
#[must_use]
pub fn age_in_years(birth_day: i64, at_day: i64, rule: AgeRule) -> Option<u32> {
    if at_day < birth_day {
        return None;
    }
    let years = match rule {
        AgeRule::CivilYears(leap) => civil_years(birth_day, at_day, leap),
        // 4 * days / 1461 is floor(days / 365.25) in integers.
        AgeRule::DaysOver365_25 => 4 * (at_day - birth_day) / 1461,
    };
    u32::try_from(years).ok()
}

fn civil_years(birth: i64, at: i64, leap: LeapBirthday) -> i64 {
    let (birth_year, birth_month, birth_dom) = civil_from_days(birth);
    let (year, month, dom) = civil_from_days(at);
    let anniversary = if leap == LeapBirthday::February28
        && birth_month == 2
        && birth_dom == 29
        && !is_leap(year)
    {
        (2, 28)
    } else {
        (birth_month, birth_dom)
    };
    let passed = (month, dom) >= anniversary;
    year - birth_year - i64::from(!passed)
}

const fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::person_time::days_from_civil;

    type Date = (i64, i64, i64);

    const MARCH1: AgeRule = AgeRule::CivilYears(LeapBirthday::March1);
    const FEB28: AgeRule = AgeRule::CivilYears(LeapBirthday::February28);

    fn age(rule: AgeRule, birth: Date, at: Date) -> Option<u32> {
        age_in_years(
            days_from_civil(birth.0, birth.1, birth.2),
            days_from_civil(at.0, at.1, at.2),
            rule,
        )
    }

    #[test]
    fn birth_date_is_age_zero() {
        for rule in [MARCH1, FEB28, AgeRule::DaysOver365_25] {
            assert_eq!(age(rule, (2000, 5, 17), (2000, 5, 17)), Some(0));
            assert_eq!(age(rule, (2000, 2, 29), (2000, 2, 29)), Some(0));
        }
    }

    #[test]
    fn before_birth_is_none() {
        for rule in [MARCH1, FEB28, AgeRule::DaysOver365_25] {
            assert_eq!(age(rule, (2000, 5, 17), (2000, 5, 16)), None);
            assert_eq!(age(rule, (2000, 5, 17), (1990, 1, 1)), None);
        }
    }

    #[test]
    fn birthday_is_reached_on_the_day() {
        for rule in [MARCH1, FEB28] {
            assert_eq!(age(rule, (2000, 5, 17), (2010, 5, 16)), Some(9));
            assert_eq!(age(rule, (2000, 5, 17), (2010, 5, 17)), Some(10));
            assert_eq!(age(rule, (2000, 12, 31), (2001, 12, 30)), Some(0));
            assert_eq!(age(rule, (2000, 12, 31), (2001, 12, 31)), Some(1));
        }
    }

    #[test]
    fn march_1_matches_the_documented_examples_of_the_python_function() {
        // completed_years in dansk_register: 2019-02-28, 2019-03-01 and
        // 2020-02-29 for a birth on 2000-02-29 give 18, 19 and 20.
        let birth = (2000, 2, 29);
        assert_eq!(age(MARCH1, birth, (2019, 2, 28)), Some(18));
        assert_eq!(age(MARCH1, birth, (2019, 3, 1)), Some(19));
        assert_eq!(age(MARCH1, birth, (2020, 2, 29)), Some(20));
    }

    #[test]
    fn february_28_completes_the_year_on_28_february_in_common_years() {
        assert_eq!(age(FEB28, (2000, 2, 29), (2001, 2, 27)), Some(0));
        assert_eq!(age(FEB28, (2000, 2, 29), (2001, 2, 28)), Some(1));
        assert_eq!(age(FEB28, (2000, 2, 29), (2001, 3, 1)), Some(1));
        assert_eq!(age(FEB28, (2000, 2, 29), (2004, 2, 28)), Some(3));
        assert_eq!(age(FEB28, (2000, 2, 29), (2004, 2, 29)), Some(4));
        assert_eq!(age(FEB28, (2000, 2, 29), (2019, 2, 28)), Some(19));
        // 1900 is not a leap year, 2000 is.
        assert_eq!(age(FEB28, (1896, 2, 29), (1900, 2, 28)), Some(4));
        assert_eq!(age(FEB28, (1996, 2, 29), (2100, 2, 28)), Some(104));
    }

    #[test]
    fn the_conventions_differ_only_on_28_february_of_a_common_year() {
        let birth = days_from_civil(2000, 2, 29);
        for at in birth..birth + 366 * 130 {
            let (_, month, dom) = civil_from_days(at);
            let (m, f) = (
                age_in_years(birth, at, MARCH1),
                age_in_years(birth, at, FEB28),
            );
            let year = civil_from_days(at).0;
            if (month, dom) == (2, 28) && !is_leap(year) {
                assert_eq!(f.unwrap(), m.unwrap() + 1);
            } else {
                assert_eq!(m, f);
            }
        }
    }

    #[test]
    fn days_rule_uses_365_25() {
        let rule = AgeRule::DaysOver365_25;
        // 365 days is not yet a year, 366 days is.
        assert_eq!(age(rule, (2001, 1, 1), (2001, 12, 31)), Some(0));
        assert_eq!(age(rule, (2001, 1, 1), (2002, 1, 1)), Some(0));
        assert_eq!(age(rule, (2001, 1, 1), (2002, 1, 2)), Some(1));
        // 4 * 365.25 = 1461 days.
        let birth = days_from_civil(2001, 1, 1);
        assert_eq!(age_in_years(birth, birth + 1460, rule), Some(3));
        assert_eq!(age_in_years(birth, birth + 1461, rule), Some(4));
    }

    #[test]
    fn days_rule_matches_floor_of_float_division() {
        for elapsed in 0..40_000_i64 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the quotient is small and not negative"
            )]
            let expected = (elapsed as f64 / 365.25).floor() as u32;
            assert_eq!(
                age_in_years(0, elapsed, AgeRule::DaysOver365_25),
                Some(expected)
            );
        }
    }

    /// Age by comparing year, month and day directly, with the birthday of a
    /// 29 February birth in a common year placed by `leap`.
    fn naive(birth: Date, at: Date, leap: LeapBirthday) -> i64 {
        let mut age = 0;
        loop {
            let year = birth.0 + age + 1;
            let birthday = if birth.1 == 2 && birth.2 == 29 && !is_leap(year) {
                match leap {
                    LeapBirthday::March1 => (year, 3, 1),
                    LeapBirthday::February28 => (year, 2, 28),
                }
            } else {
                (year, birth.1, birth.2)
            };
            if at >= birthday {
                age += 1;
            } else {
                return age;
            }
        }
    }

    fn check_against_naive(birth: i64, at: i64) {
        let b = civil_from_days(birth);
        let a = civil_from_days(at);
        for leap in [LeapBirthday::March1, LeapBirthday::February28] {
            let expected = if at < birth {
                None
            } else {
                Some(naive(b, a, leap))
            };
            let got = age_in_years(birth, at, AgeRule::CivilYears(leap));
            assert_eq!(got.map(i64::from), expected, "{leap:?} {b:?} -> {a:?}");
        }
    }

    #[test]
    fn civil_years_match_a_naive_comparison() {
        // Every birth date from 1999 to 2001, which includes a leap day,
        // against every reference date from 1998 to 2032, which includes eight.
        let start = days_from_civil(1998, 1, 1);
        let end = days_from_civil(2032, 12, 31);
        for birth in days_from_civil(1999, 1, 1)..=days_from_civil(2001, 12, 31) {
            for at in start..=end {
                check_against_naive(birth, at);
            }
        }
    }

    #[test]
    fn civil_years_match_a_naive_comparison_over_century_leap_rules() {
        let births = [
            (1896, 2, 29),
            (1900, 2, 28),
            (1900, 3, 1),
            (1999, 12, 31),
            (2000, 2, 29),
            (2096, 2, 29),
        ];
        for b in births {
            let birth = days_from_civil(b.0, b.1, b.2);
            for at in birth - 5..birth + 366 * 210 {
                check_against_naive(birth, at);
            }
        }
    }
}
