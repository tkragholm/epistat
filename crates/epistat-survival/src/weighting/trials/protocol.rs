//! The rows that the models of the per-protocol weights are fitted on, as
//! `TrialEmulation` chooses them.

use crate::error::{Error, Result};

/// The panel of a sequential trial emulation in columns: one row per person and
/// period, ordered by person and then period.
#[derive(Debug, Clone, Copy)]
pub struct Panel<'a> {
    /// The person of each row. The rows of a person are next to each other.
    pub id: &'a [u64],
    /// The period of each row, increasing within a person.
    pub period: &'a [i64],
    /// The treatment in the period.
    pub treatment: &'a [bool],
    /// Whether the person is eligible for a trial that starts in the period.
    pub eligible: &'a [bool],
    /// Whether the outcome happens in the period.
    pub outcome: &'a [bool],
}

/// The rows of the models of the switch weights, from [`protocol_rows`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolRows {
    /// Whether each row is one the models are fitted on, for
    /// [`WeightRows::fitted`](super::WeightRows::fitted).
    pub fitted: Vec<bool>,
    /// The treatment of the row before, as `TrialEmulation`'s `am_1`: `false`
    /// in the first row of a person from which the person is followed (the
    /// first eligible row) and in the rows before it, the treatment of the row
    /// before otherwise. It is the `previous` of
    /// [`PersonPeriod`](super::PersonPeriod) that reproduces `TrialEmulation`.
    pub previous: Vec<bool>,
}

/// The rows that `TrialEmulation` (version 0.0.5) fits the models of the
/// switch weights on in a per-protocol analysis.
///
/// # The rule
///
/// For each person the rows before the first eligible row and after the first
/// outcome are not used. On the rows that remain, with `a_t` the treatment of a
/// row, `a_{t-1}` that of the row before (0 in the first) and a switch where the
/// two differ, a row is in the fits while the person follows a regime that
/// started at an eligible row and has not stopped:
///
/// * at an eligible row a person who follows no regime starts the regime of
///   the row's treatment;
/// * a switch at an eligible row starts the regime of the new treatment;
/// * a switch at a row that is not eligible stops the regime. The row of the
///   switch is still in the fits, and the rows after it are not, until the
///   next eligible row starts a regime again.
///
/// This is `TrialEmulation:::data_manipulation` with its C++ `censor_func`.
/// Every row of the per-protocol follow-up of every trial lies in these rows,
/// since a trial starts at an eligible row and its follow-up ends before the
/// first switch.
///
/// # Errors
///
/// [`Error::Input`] for columns of different lengths and rows that are not
/// ordered by person and period.
pub fn protocol_rows(panel: &Panel<'_>) -> Result<ProtocolRows> {
    let n = panel.id.len();
    for (name, len) in [
        ("period", panel.period.len()),
        ("treatment", panel.treatment.len()),
        ("eligible", panel.eligible.len()),
        ("outcome", panel.outcome.len()),
    ] {
        if len != n {
            return Err(Error::input(format!("{name}: {len} values for {n} rows")));
        }
    }
    for i in 1..n {
        if panel.id[i] < panel.id[i - 1] {
            return Err(Error::input(format!(
                "id: the rows of a person must be next to each other and ordered, row {i}"
            )));
        }
        if panel.id[i] == panel.id[i - 1] && panel.period[i] <= panel.period[i - 1] {
            return Err(Error::input(format!(
                "period: row {i} is not after the row before it for person {}",
                panel.id[i]
            )));
        }
    }
    let mut fitted = vec![false; n];
    let mut previous = vec![false; n];
    let mut start = 0;
    while start < n {
        let end = (start..n)
            .find(|&i| panel.id[i] != panel.id[start])
            .unwrap_or(n);
        previous[start + 1..end].copy_from_slice(&panel.treatment[start..end - 1]);
        if let Some(first) = (start..end).find(|&i| panel.eligible[i]) {
            previous[first] = false;
            let last = (first..end).find(|&i| panel.outcome[i]).unwrap_or(end - 1);
            follow(panel, first, last, &mut fitted);
        }
        start = end;
    }
    Ok(ProtocolRows { fitted, previous })
}

/// The state machine of `censor_func` over the used rows `first..=last` of one
/// person.
fn follow(panel: &Panel<'_>, first: usize, last: usize, fitted: &mut [bool]) {
    let (mut started0, mut started1, mut stop0, mut stop1) = (false, false, false, false);
    for i in first..=last {
        let a = panel.treatment[i];
        let eligible = panel.eligible[i];
        let switch = i > first && panel.treatment[i - 1] != a;
        if stop0 || stop1 {
            (started0, started1, stop0, stop1) = (false, false, false, false);
        }
        if !started0 && !started1 && eligible {
            if a {
                started1 = true;
            } else {
                started0 = true;
            }
        }
        let (mut e0, mut e1) = if started0 && !stop0 {
            (true, false)
        } else if started1 && !stop1 {
            (false, true)
        } else {
            (false, false)
        };
        if switch {
            if eligible {
                if a {
                    (started1, stop1, started0, stop0) = (true, false, false, false);
                    e1 = true;
                } else {
                    (started0, stop0, started1, stop1) = (true, false, false, false);
                    e0 = true;
                }
            } else {
                stop0 = started0;
                stop1 = started1;
            }
        }
        fitted[i] = e0 || e1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(treatment: &[u8], eligible: &[u8], outcome: &[u8]) -> ProtocolRows {
        let n = treatment.len();
        let id = vec![7; n];
        let period: Vec<i64> = (0..i64::try_from(n).expect("small")).collect();
        let t: Vec<bool> = treatment.iter().map(|&v| v == 1).collect();
        let e: Vec<bool> = eligible.iter().map(|&v| v == 1).collect();
        let o: Vec<bool> = outcome.iter().map(|&v| v == 1).collect();
        protocol_rows(&Panel {
            id: &id,
            period: &period,
            treatment: &t,
            eligible: &e,
            outcome: &o,
        })
        .expect("valid")
    }

    fn mask(r: &ProtocolRows) -> Vec<u8> {
        r.fitted.iter().map(|&f| u8::from(f)).collect()
    }

    #[test]
    fn a_stop_after_starting_ends_the_rows_until_the_next_eligible_row() {
        // Eligible while never treated: start in period 2, stop in period 4.
        let r = rows(
            &[0, 0, 1, 1, 0, 0, 1],
            &[1, 1, 1, 0, 0, 0, 0],
            &[0, 0, 0, 0, 0, 0, 0],
        );
        assert_eq!(mask(&r), vec![1, 1, 1, 1, 1, 0, 0]);
        assert_eq!(
            r.previous,
            vec![false, false, false, true, true, false, false]
        );
    }

    #[test]
    fn rows_before_eligibility_and_after_the_outcome_are_left_out() {
        let r = rows(&[1, 0, 0, 0, 0], &[0, 1, 1, 1, 1], &[0, 0, 1, 0, 0]);
        assert_eq!(mask(&r), vec![0, 1, 1, 0, 0]);
        // The first eligible row has no previous treatment, as in TrialEmulation.
        assert_eq!(r.previous, vec![false, false, false, false, false]);
    }

    #[test]
    fn an_eligible_row_starts_a_regime_again() {
        // Always eligible: every switch starts a new regime.
        let r = rows(&[0, 1, 0, 1], &[1, 1, 1, 1], &[0, 0, 0, 0]);
        assert_eq!(mask(&r), vec![1, 1, 1, 1]);
        // Eligible again after a stop: the row after the stop is left out, the
        // eligible row after it starts again.
        let r = rows(&[0, 1, 1, 0], &[1, 0, 0, 1], &[0, 0, 0, 0]);
        assert_eq!(mask(&r), vec![1, 1, 0, 1]);
    }

    #[test]
    fn unordered_rows_are_errors() {
        let id = [2, 1];
        let period = [0, 0];
        let b = [false, false];
        assert!(
            protocol_rows(&Panel {
                id: &id,
                period: &period,
                treatment: &b,
                eligible: &b,
                outcome: &b,
            })
            .is_err()
        );
    }
}
