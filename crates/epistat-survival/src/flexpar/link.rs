//! The three scales of the model as functions of the linear predictor `eta`.
//!
//! With `S` the survivor function, the model is `g(S) = eta` for
//! `g(S) = ln(-ln S)`, `ln((1 - S) / S)` or `-Phi^-1(S)`. The cumulative
//! hazard is `H(eta) = -ln S` and the hazard is `m(eta) * eta'(u) / t` with
//! `m = H'`. A row's log likelihood needs `ln m` and `H`, with the derivatives
//! of both in `eta`:
//!
//! ```text
//! scale   H(eta)             ln m(eta)                  (ln m)'          (ln m)''
//! hazard  e^eta              eta                        1                0
//! odds    ln(1 + e^eta)      eta - ln(1 + e^eta)        1 - p            -p (1 - p)
//! probit  -ln Phi(-eta)      ln phi(eta) + H(eta)       -eta + l         -1 + l (l - eta)
//! ```
//!
//! where `p = 1 / (1 + e^-eta)` and `l = phi(eta) / Phi(-eta)` is the inverse
//! Mills ratio, so `H' = p` for the odds and `H' = l` for the probit, and
//! `H'' = p (1 - p)` and `l (l - eta)`.

use epistat_stats::dist::normal;

use super::FlexparScale;

const HALF_LN_2PI: f64 = 0.918_938_533_204_672_7;

/// Above this `eta` the lower tail `Phi(-eta)` is evaluated by its continued
/// fraction, whose terms converge quickly there. Below it the tail is the
/// distribution function itself, which has full relative accuracy as far as
/// `Phi` is not subnormal.
const TAIL_START: f64 = 5.0;

/// `ln(1 + e^x)` without overflow or cancellation.
pub(super) fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

pub(super) use epistat_stats::logistic::expit;

/// `Phi(-y) / phi(y)` by the continued fraction
/// `1 / (y + 1 / (y + 2 / (y + 3 / (y + ...))))`, evaluated from the tail
/// with 80 terms. Valid for `y >= TAIL_START`, where 80 terms reach the
/// rounding error.
fn mills_ratio(y: f64) -> f64 {
    let mut tail = y;
    for k in (1..=80).rev() {
        tail = y + f64::from(k) / tail;
    }
    1.0 / tail
}

/// `ln Phi(-y)` and the inverse Mills ratio `phi(y) / Phi(-y)`.
pub(super) fn log_upper_tail(y: f64) -> (f64, f64) {
    if y > TAIL_START {
        let ratio = mills_ratio(y);
        (-0.5 * y * y - HALF_LN_2PI + ratio.ln(), 1.0 / ratio)
    } else if y < 0.0 {
        let lower = normal::cdf(y);
        ((-lower).ln_1p(), normal::pdf(y) / (1.0 - lower))
    } else {
        let tail = normal::cdf(-y);
        (tail.ln(), normal::pdf(y) / tail)
    }
}

/// The cumulative hazard `H(eta)` and its first two derivatives.
pub(super) fn cumulative(scale: FlexparScale, eta: f64) -> (f64, f64, f64) {
    match scale {
        FlexparScale::Hazard => {
            let h = eta.exp();
            (h, h, h)
        }
        FlexparScale::Odds => {
            let p = expit(eta);
            (softplus(eta), p, p * (1.0 - p))
        }
        FlexparScale::Probit => {
            let (log_tail, l) = log_upper_tail(eta);
            (-log_tail, l, l * (l - eta))
        }
    }
}

/// `ln m(eta)` and its first two derivatives, where the hazard is
/// `m(eta) eta'(u) / t`.
pub(super) fn log_hazard_factor(scale: FlexparScale, eta: f64) -> (f64, f64, f64) {
    match scale {
        FlexparScale::Hazard => (eta, 1.0, 0.0),
        FlexparScale::Odds => {
            let p = expit(eta);
            (eta - softplus(eta), 1.0 - p, -p * (1.0 - p))
        }
        FlexparScale::Probit => {
            let (log_tail, l) = log_upper_tail(eta);
            (
                -0.5 * eta * eta - HALF_LN_2PI - log_tail,
                -eta + l,
                -1.0 + l * (l - eta),
            )
        }
    }
}

/// The survivor function at `eta`.
pub(super) fn survival(scale: FlexparScale, eta: f64) -> f64 {
    match scale {
        FlexparScale::Hazard => (-eta.exp()).exp(),
        FlexparScale::Odds => expit(-eta),
        FlexparScale::Probit => normal::cdf(-eta),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn the_tail_is_continuous_where_the_method_changes() {
        let (below, l_below) = log_upper_tail(TAIL_START - 1e-12);
        let (above, l_above) = log_upper_tail(TAIL_START + 1e-12);
        assert!(close(below, above, 1e-11), "{below} {above}");
        assert!(close(l_below, l_above, 1e-11));
    }

    #[test]
    fn the_far_tail_matches_the_known_asymptote() {
        // ln Phi(-40) = -804.608442013754 (to 15 digits, from the asymptotic series).
        let (value, l) = log_upper_tail(40.0);
        assert!(close(value, -804.608_442_013_754, 1e-13), "{value}");
        assert!(close(l, 40.0 + 1.0 / 40.0, 1e-3));
    }

    #[test]
    fn derivatives_agree_with_differences() {
        for scale in [
            FlexparScale::Hazard,
            FlexparScale::Odds,
            FlexparScale::Probit,
        ] {
            for &eta in &[-6.0, -1.3, 0.0, 0.7, 4.0, 5.5, 9.0] {
                let h = 1e-5;
                let (_, h1, h2) = cumulative(scale, eta);
                let d1 = (cumulative(scale, eta + h).0 - cumulative(scale, eta - h).0) / (2.0 * h);
                let d2 = (cumulative(scale, eta + h).1 - cumulative(scale, eta - h).1) / (2.0 * h);
                assert!(close(h1, d1, 1e-7), "{scale:?} {eta} {h1} {d1}");
                assert!(close(h2, d2, 1e-6), "{scale:?} {eta} {h2} {d2}");
                let (_, g1, g2) = log_hazard_factor(scale, eta);
                let e1 = (log_hazard_factor(scale, eta + h).0
                    - log_hazard_factor(scale, eta - h).0)
                    / (2.0 * h);
                let e2 = (log_hazard_factor(scale, eta + h).1
                    - log_hazard_factor(scale, eta - h).1)
                    / (2.0 * h);
                assert!(close(g1, e1, 1e-7), "{scale:?} {eta} {g1} {e1}");
                assert!(close(g2, e2, 1e-6), "{scale:?} {eta} {g2} {e2}");
            }
        }
    }

    #[test]
    fn survival_is_the_exponential_of_minus_the_cumulative_hazard() {
        for scale in [
            FlexparScale::Hazard,
            FlexparScale::Odds,
            FlexparScale::Probit,
        ] {
            for &eta in &[-3.0, 0.0, 2.0] {
                let s = survival(scale, eta);
                assert!(close(s, (-cumulative(scale, eta).0).exp(), 1e-14));
            }
        }
    }
}
