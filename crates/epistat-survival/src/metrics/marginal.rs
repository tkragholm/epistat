//! The marginal risk of one cause at chosen times: one minus the
//! Kaplan-Meier estimate for single-event data and the Aalen-Johansen
//! cumulative incidence when another event is present.

#![expect(
    clippy::float_cmp,
    reason = "times tie when they are equal, which is the definition of a tie"
)]

use super::prepare::Kind;

/// The risk of the cause at each of `times`, for rows sorted by ascending time.
///
/// A death at a tied time comes before a censoring, as in `prodlim` and
/// `survfit`. A time before the first death has risk 0, and one after the last
/// row keeps the last value.
pub(super) fn marginal_risk(time: &[f64], kind: &[Kind], times: &[f64]) -> Vec<f64> {
    let n = time.len();
    let competing = kind.contains(&Kind::Competing);
    let mut out = Vec::with_capacity(times.len());
    let mut next = 0;
    let mut surv = 1.0;
    let mut cuminc = 0.0;
    let mut i = 0;
    let at_risk_total = n as f64;
    let mut left = 0.0;
    while i < n {
        let u = time[i];
        while next < times.len() && times[next] < u {
            out.push(if competing { cuminc } else { 1.0 - surv });
            next += 1;
        }
        let mut j = i;
        let (mut d_cause, mut d_other) = (0.0, 0.0);
        while j < n && time[j] == u {
            match kind[j] {
                Kind::Cause => d_cause += 1.0,
                Kind::Competing => d_other += 1.0,
                Kind::Censored => {}
            }
            j += 1;
        }
        let at_risk = at_risk_total - left;
        let d = d_cause + d_other;
        if d > 0.0 {
            cuminc += surv * d_cause / at_risk;
            surv *= 1.0 - d / at_risk;
        }
        left += (j - i) as f64;
        i = j;
    }
    while next < times.len() {
        out.push(if competing { cuminc } else { 1.0 - surv });
        next += 1;
    }
    out
}
