//! Age in whole years at a date, under a named rule.
//!
//! Dates are counts of days since 1970-01-01, as in [`crate::person_time`].
//!
//! # Rules
//!
//! [`AgeRule::CivilYears`] counts the birthdays that have passed. With birth
//! date `(by, bm, bd)` and reference date `(y, m, d)`,
//!
//! ```text
//! age = y - by - 1   if (m, d) < (bm, bd)
//! age = y - by       otherwise
//! ```
//!
//! A person born on 29 February has the birthday on 28 February in a year that
//! is not a leap year, the same date that adding whole years to 29 February
//! gives in most date libraries. So the person is a year older on 28 February
//! of a common year and on 29 February of a leap year.
//!
//! [`AgeRule::DaysOver365_25`] is `floor(days / 365.25)`, with `days` the
//! number of days from birth to the reference date. It agrees with the civil
//! rule for most dates and differs by one year on some days near a birthday,
//! because a civil year is 365 or 366 days.
//!
//! # Edge cases
//!
//! - A reference date before the birth date gives [`None`].
//! - On the birth date the age is 0 under both rules.

use crate::person_time::civil_from_days;

/// The rule for counting whole years of age.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AgeRule {
    /// Birthdays passed. A 29 February birthday falls on 28 February in a year
    /// that is not a leap year.
    CivilYears,
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
        AgeRule::CivilYears => civil_years(birth_day, at_day),
        // 4 * days / 1461 is floor(days / 365.25) in integers.
        AgeRule::DaysOver365_25 => 4 * (at_day - birth_day) / 1461,
    };
    u32::try_from(years).ok()
}

fn civil_years(birth: i64, at: i64) -> i64 {
    let (birth_year, birth_month, birth_dom) = civil_from_days(birth);
    let (year, month, dom) = civil_from_days(at);
    let anniversary = if birth_month == 2 && birth_dom == 29 && !is_leap(year) {
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

    fn civil(birth: (i64, i64, i64), at: (i64, i64, i64)) -> Option<u32> {
        age_in_years(
            days_from_civil(birth.0, birth.1, birth.2),
            days_from_civil(at.0, at.1, at.2),
            AgeRule::CivilYears,
        )
    }

    fn days365(birth: (i64, i64, i64), at: (i64, i64, i64)) -> Option<u32> {
        age_in_years(
            days_from_civil(birth.0, birth.1, birth.2),
            days_from_civil(at.0, at.1, at.2),
            AgeRule::DaysOver365_25,
        )
    }

    #[test]
    fn birth_date_is_age_zero() {
        assert_eq!(civil((2000, 5, 17), (2000, 5, 17)), Some(0));
        assert_eq!(days365((2000, 5, 17), (2000, 5, 17)), Some(0));
        assert_eq!(civil((2000, 2, 29), (2000, 2, 29)), Some(0));
    }

    #[test]
    fn before_birth_is_none() {
        assert_eq!(civil((2000, 5, 17), (2000, 5, 16)), None);
        assert_eq!(days365((2000, 5, 17), (1990, 1, 1)), None);
    }

    #[test]
    fn birthday_is_reached_on_the_day() {
        assert_eq!(civil((2000, 5, 17), (2010, 5, 16)), Some(9));
        assert_eq!(civil((2000, 5, 17), (2010, 5, 17)), Some(10));
        assert_eq!(civil((2000, 12, 31), (2001, 12, 30)), Some(0));
        assert_eq!(civil((2000, 12, 31), (2001, 12, 31)), Some(1));
    }

    #[test]
    fn leap_day_birthday_falls_on_28_february_in_common_years() {
        assert_eq!(civil((2000, 2, 29), (2001, 2, 27)), Some(0));
        assert_eq!(civil((2000, 2, 29), (2001, 2, 28)), Some(1));
        assert_eq!(civil((2000, 2, 29), (2001, 3, 1)), Some(1));
        assert_eq!(civil((2000, 2, 29), (2004, 2, 28)), Some(3));
        assert_eq!(civil((2000, 2, 29), (2004, 2, 29)), Some(4));
        // 1900 is not a leap year, 2000 is.
        assert_eq!(civil((1896, 2, 29), (1900, 2, 28)), Some(4));
        assert_eq!(civil((1996, 2, 29), (2100, 2, 28)), Some(104));
    }

    #[test]
    fn days_rule_uses_365_25() {
        // 365 days is not yet a year, 366 days is.
        assert_eq!(days365((2001, 1, 1), (2001, 12, 31)), Some(0));
        assert_eq!(days365((2001, 1, 1), (2002, 1, 1)), Some(0));
        assert_eq!(days365((2001, 1, 1), (2002, 1, 2)), Some(1));
        // 4 * 365.25 = 1461 days.
        let birth = days_from_civil(2001, 1, 1);
        assert_eq!(
            age_in_years(birth, birth + 1460, AgeRule::DaysOver365_25),
            Some(3)
        );
        assert_eq!(
            age_in_years(birth, birth + 1461, AgeRule::DaysOver365_25),
            Some(4)
        );
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

    /// Age by comparing year, month and day directly, with a 29 February
    /// birthday moved to 28 February in a common year.
    fn naive(birth: (i64, i64, i64), at: (i64, i64, i64)) -> i64 {
        let mut age = 0;
        loop {
            let year = birth.0 + age + 1;
            let birthday = if birth.1 == 2 && birth.2 == 29 && !is_leap(year) {
                (year, 2, 28)
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

    #[test]
    fn civil_years_match_a_naive_comparison() {
        // Every birth date from 1996 to 2004, which includes two leap days,
        // against reference dates from 1990 to 2110 at a stride of seven days.
        let mut births = Vec::new();
        for year in 1996..=2004 {
            for day in 0..366 {
                let z = days_from_civil(year, 1, 1) + day;
                if civil_from_days(z).0 == year {
                    births.push(z);
                }
            }
        }
        let start = days_from_civil(1990, 1, 1);
        let end = days_from_civil(2110, 12, 31);
        for &birth in &births {
            let b = civil_from_days(birth);
            for at in (start..=end).step_by(7) {
                let a = civil_from_days(at);
                let expected = if at < birth { None } else { Some(naive(b, a)) };
                let got = age_in_years(birth, at, AgeRule::CivilYears);
                assert_eq!(got.map(i64::from), expected, "{b:?} -> {a:?}");
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
                let a = civil_from_days(at);
                let expected = if at < birth { None } else { Some(naive(b, a)) };
                let got = age_in_years(birth, at, AgeRule::CivilYears);
                assert_eq!(got.map(i64::from), expected, "{b:?} -> {a:?}");
            }
        }
    }
}
