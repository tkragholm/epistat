//! Scores for censored times to an event in matched sets: Gehan's
//! comparison of every two units.
//!
//! Without hidden bias, and under the null hypothesis that the treatment
//! changes neither the time to the event nor the time to censoring, the
//! observed times and event indicators are fixed, and so is any score
//! computed from them within a set. The treated unit's score summed over the
//! sets is then a sum statistic, and its bound under `gamma` is that of
//! [`super::set_bounds`] \[1, chapter 4\], \[2\].
//!
//! # Algorithm
//!
//! Unit `j` had the event definitely before unit `k` when `j` had the event
//! and either `t_j < t_k`, or `t_j = t_k` and `k` was censored: a unit
//! censored at the time of another's event is taken to have outlasted it, as
//! by Gehan \[3\]. The score of `j` in its set is the number of units it
//! definitely preceded minus the number that definitely preceded it. Two
//! events at the same time, and a pair in which the earlier time is a
//! censoring, are not ordered and add nothing. With
//! [`EventDirection::Later`] the scores change sign, so the test is that the
//! treated have the event later. In a pair the scores are 1, 0 and -1, and
//! the statistic is the sign test of the ordered pairs.
//!
//! The null hypothesis includes censoring. Censoring that the treatment
//! could change, such as loss to follow-up that differs by treatment, is not
//! covered; administrative censoring at a date shared by a set is.
//!
//! # References
//!
//! 1. Rosenbaum PR. Observational Studies, 2nd ed. Springer; 2002.
//! 2. Gastwirth JL, Krieger AM, Rosenbaum PR. Asymptotic separability in
//!    sensitivity analysis. J R Stat Soc Series B. 2000;62(3):545-55.
//! 3. Gehan EA. A generalized Wilcoxon test for comparing arbitrarily
//!    singly-censored samples. Biometrika. 1965;52(1-2):203-23.

use super::SensitivityError;

/// A time to an event or to censoring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Censored {
    /// The time.
    pub time: f64,
    /// Whether the time is that of the event; otherwise it is a censoring.
    pub event: bool,
}

/// Which direction of the effect a test of censored times looks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventDirection {
    /// The treated have the event earlier, as for a harmful exposure.
    Earlier,
    /// The treated have the event later.
    Later,
}

/// Whether `a` definitely had the event before `b`.
fn precedes(a: Censored, b: Censored) -> bool {
    a.event && (a.time < b.time || (a.time.total_cmp(&b.time).is_eq() && !b.event))
}

/// Gehan's scores of every unit of every set, the treated unit first in
/// each, for [`super::set_bounds`].
///
/// # Errors
///
/// [`SensitivityError::NoInformation`] for no sets,
/// [`SensitivityError::SetTooSmall`] and [`SensitivityError::NotFinite`]
/// for a time that is not finite.
pub fn gehan_scores(
    sets: &[Vec<Censored>],
    direction: EventDirection,
) -> Result<Vec<Vec<f64>>, SensitivityError> {
    if sets.is_empty() {
        return Err(SensitivityError::NoInformation);
    }
    let sign = match direction {
        EventDirection::Earlier => 1.0,
        EventDirection::Later => -1.0,
    };
    sets.iter()
        .map(|set| {
            if set.len() < 2 {
                return Err(SensitivityError::SetTooSmall);
            }
            if set.iter().any(|u| !u.time.is_finite()) {
                return Err(SensitivityError::NotFinite);
            }
            Ok(set
                .iter()
                .enumerate()
                .map(|(j, a)| {
                    let mut score = 0_i64;
                    for (k, b) in set.iter().enumerate() {
                        if j != k {
                            score += i64::from(precedes(*a, *b)) - i64::from(precedes(*b, *a));
                        }
                    }
                    sign * score as f64
                })
                .collect())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(time: f64, event: bool) -> Censored {
        Censored { time, event }
    }

    #[test]
    fn scores_order_the_units_that_can_be_ordered() {
        let set = vec![
            unit(2.0, true),
            unit(5.0, false),
            unit(1.0, false),
            unit(2.0, true),
            unit(2.0, false),
        ];
        let s = gehan_scores(&[set], EventDirection::Earlier).unwrap();
        // The first preceded the units censored at 5 and at 2; it is tied
        // with the other event at 2 and not ordered with the censoring at 1.
        assert_eq!(s[0], vec![2.0, -2.0, 0.0, 2.0, -2.0]);
        // The scores of a set add to zero.
        assert_eq!(s[0].iter().sum::<f64>(), 0.0);
        let later = gehan_scores(
            &[vec![unit(2.0, true), unit(5.0, false)]],
            EventDirection::Later,
        )
        .unwrap();
        assert_eq!(later[0], vec![-1.0, 1.0]);
    }

    #[test]
    fn invalid_sets_are_errors() {
        assert!(gehan_scores(&[], EventDirection::Earlier).is_err());
        assert!(gehan_scores(&[vec![unit(1.0, true)]], EventDirection::Earlier).is_err());
        assert!(
            gehan_scores(
                &[vec![unit(f64::NAN, true), unit(1.0, true)]],
                EventDirection::Earlier
            )
            .is_err()
        );
    }
}
