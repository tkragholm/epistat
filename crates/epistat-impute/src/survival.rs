//! The Nelson-Aalen cumulative hazard as an imputation predictor.
//!
//! To impute a covariate of a Cox model, the imputation model has to contain
//! the outcome. White and Royston \[1\] show that for a small covariate effect
//! the conditional distribution of the covariate given the survival time `T`
//! and the event indicator `d` is approximately linear in `d` and in the
//! cumulative hazard `H(T)`, not in `log T`. The two columns `d` and `H(T)`
//! are therefore added as complete predictors of every incomplete covariate,
//! where `H` is the Nelson-Aalen estimate from the observed `(T, d)`.
//!
//! # Algorithm
//!
//! [`nelson_aalen`] gives each person `i` the value at their own time `T_i`
//! of
//!
//! ```text
//! H(t) = sum over event times s <= t of  h_s
//! h_s  = d_s / n_s                        (Ties::Breslow)
//! h_s  = sum_{k=0}^{d_s-1} 1 / (n_s - k)  (Ties::Efron)
//! ```
//!
//! with `d_s` the events at time `s` and `n_s` the number with `T >= s`.
//! The two agree without tied event times. The sort is `O(n log n)`.
//! `Ties::Efron` is `mice::nelsonaalen`, which evaluates
//! `survival::basehaz(coxph(Surv(time, status) ~ 1))` at each person's time,
//! and `coxph` handles ties as Efron does. `Ties::Breslow` is the textbook
//! Nelson-Aalen estimator, `coxph(ties = "breslow")`. A person's own event at
//! `T_i` is included in `H(T_i)`.
//!
//! The time must start at 0 for every person: there is no delayed entry, and
//! with delayed entry the risk set would need the entry times.
//!
//! # References
//!
//! 1. White IR, Royston P. Imputing missing covariate values for the Cox
//!    model. Stat Med. 2009;28(15):1982-98.

use epistat_core::{InputError, check};

use crate::error::{Result, length};

/// How the hazard of a time with tied events is formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ties {
    /// `d / n`: every tied event has the whole risk set as denominator.
    Breslow,
    /// `1/n + 1/(n-1) + ... + 1/(n-d+1)`: each tied event leaves the risk set
    /// before the next, as in `coxph`'s default and so in `mice::nelsonaalen`.
    Efron,
}

/// The Nelson-Aalen cumulative hazard of `status` (1 an event, 0 censored) at
/// each person's own `time`.
///
/// # Errors
///
/// [`Error::Input`](crate::Error::Input): no rows
/// ([`InputError::Empty`]), a status of a different length than the times
/// ([`InputError::Length`]), a time that is not finite
/// ([`InputError::NotFinite`]) or negative, or a status other than 0 and 1
/// ([`InputError::Domain`], with its row).
pub fn nelson_aalen(time: &[f64], status: &[i32], ties: Ties) -> Result<Vec<f64>> {
    check::non_empty("time", time.len())?;
    length("status", time.len(), status.len())?;
    check::finite("time", time)?;
    if let Some(row) = time.iter().position(|&t| t < 0.0) {
        return Err(InputError::Domain {
            what: "time",
            value: time[row],
            domain: "[0, inf)",
            row: Some(row),
        }
        .into());
    }
    if let Some(row) = status.iter().position(|&s| s != 0 && s != 1) {
        return Err(InputError::Domain {
            what: "status",
            value: f64::from(status[row]),
            domain: "{0, 1}",
            row: Some(row),
        }
        .into());
    }
    let n = time.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| time[a].total_cmp(&time[b]));
    let mut hazard = vec![0.0; n];
    let mut cumulative = 0.0;
    let mut at_risk = n;
    let mut start = 0;
    while start < n {
        let t = time[order[start]];
        let mut end = start;
        let mut events = 0_usize;
        while end < n && time[order[end]].total_cmp(&t).is_eq() {
            events += usize::from(status[order[end]] == 1);
            end += 1;
        }
        match ties {
            Ties::Breslow => cumulative += events as f64 / at_risk as f64,
            Ties::Efron => {
                for k in 0..events {
                    cumulative += 1.0 / (at_risk - k) as f64;
                }
            }
        }
        for &i in &order[start..end] {
            hazard[i] = cumulative;
        }
        at_risk -= end - start;
        start = end;
    }
    Ok(hazard)
}
