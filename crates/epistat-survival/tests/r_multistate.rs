//! `multistate::state_occupation_curve` and `multistate::integrate` checked
//! against R 4.6.2's `survival::survfit(Surv(start, stop, state) ~ 1, id =,
//! istate =)` and the restricted mean time in state of `summary(fit, rmean =)`.
//! The bootstrap has no R counterpart and is not checked here.

use epistat_survival::multistate::{Intervals, integrate, state_occupation_curve};
use epistat_testkit::load_json;
use serde::Deserialize;

const STATES: usize = 3;

/// A summed leaving rate above one by more than this is refused as malformed
/// input. It is a bound on a check, not on a result: every rate of well-formed
/// data is at most one up to a few units of rounding.
const RATE_TOLERANCE: f64 = 1e-9;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    survival: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    weights: String,
    start: Vec<f64>,
    stop: Vec<f64>,
    from: Vec<u32>,
    to: Vec<u32>,
    weight: Vec<f64>,
    time: Vec<f64>,
    p0: Vec<f64>,
    t0: f64,
    n_risk: Vec<f64>,
    pstate: Vec<Vec<f64>>,
    tau: Vec<f64>,
    rmean: Vec<Vec<f64>>,
    rmean_common: Vec<f64>,
}

/// The largest deviation seen, in units of its tolerance.
#[derive(Default)]
struct Worst {
    value: f64,
    case: String,
}

impl Worst {
    #[track_caller]
    fn see(&mut self, case: &str, deviation: f64, tolerance: f64) {
        let ratio = if tolerance > 0.0 {
            deviation / tolerance
        } else if deviation == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        assert!(
            ratio <= 1.0,
            "{case}: deviation {deviation:e} against tolerance {tolerance:e}"
        );
        if ratio >= self.value {
            self.value = ratio;
            self.case = case.to_string();
        }
    }
}

/// The bound on how far two implementations of the product integral can differ
/// in the sum over the states of the absolute differences of the probabilities,
/// after each of the times.
///
/// The step is `p <- p A`, with `A` row-stochastic, so it does not enlarge an
/// error carried in from before: the sum of the absolute differences of `p A`
/// and `p' A` is at most that of `p` and `p'`. What a step adds is the
/// difference between the two matrices and the rounding of the product.
///
/// - A weighted count is a sum of at most `n` weights, or a running balance of
///   entries and exits. Each addition rounds by at most `eps / 2` times a
///   running value no larger than the total weight `W`, so two implementations
///   differ by at most `d_w = n eps W`. Weights that are multiples of 1/8 sum
///   exactly, then `d_w` is zero.
/// - An off-diagonal entry is `a = n_ij / Y_i`. The two differ by `eps a` from
///   the division and by `(d_w + a d_w) / Y_i` from the counts. `Y_i` is at
///   least the smallest weight `w`.
/// - The diagonal is `1 - sum_j a_ij`. It differs by the sum of the off-diagonal
///   differences, plus `eps` for the subtraction in each implementation.
/// - A row of `A` therefore differs in the sum of absolute values by at most
///   `2 (eps + 2 (S - 1) d_w / w) + eps`, since the off-diagonal sum of rates
///   is at most one.
/// - The product `p A` adds `S eps` for the sums, so a step adds at most
///   `3 eps + 4 (S - 1) d_w / w + S eps`.
///
/// The bound after `t` times is `t` steps' worth. The probabilities are on the
/// scale of 1, so it is absolute.
fn step_bound(weight_bound: f64, smallest_weight: f64) -> f64 {
    let eps = f64::EPSILON;
    let s = STATES as f64;
    3.0 * eps + 4.0 * (s - 1.0) * weight_bound / smallest_weight + s * eps
}

#[test]
fn occupation_and_restricted_time_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/multistate.json"
    ));
    eprintln!(
        "{}, survival {}",
        reference.versions.r, reference.versions.survival
    );
    let eps = f64::EPSILON;
    let mut at_risk_worst = Worst::default();
    let mut probability_worst = Worst::default();
    let mut rmean_worst = Worst::default();
    let mut common_worst = Worst::default();

    for case in &reference.cases {
        let intervals = Intervals::new(&case.start, &case.stop, &case.from, &case.to).unwrap();
        let got = state_occupation_curve(
            intervals,
            &case.weight,
            &case.time,
            &case.p0,
            STATES,
            RATE_TOLERANCE,
        )
        .unwrap();
        assert!(
            (case.p0.iter().sum::<f64>() - 1.0).abs() <= 4.0 * eps,
            "{}",
            case.name
        );

        let rows = case.start.len() as f64;
        let total: f64 = case.weight.iter().sum();
        let smallest = case.weight.iter().copied().fold(f64::INFINITY, f64::min);
        let weight_bound = if case.weights == "decimal" {
            rows * eps * total
        } else {
            0.0
        };
        let steps = step_bound(weight_bound, smallest);
        let times = case.time.len();

        let mut deviation = 0.0f64;
        for (a, b) in got.at_risk.iter().zip(&case.n_risk) {
            deviation = deviation.max((a - b).abs());
        }
        // The total at risk is a sum of up to `STATES` counts, each as above.
        at_risk_worst.see(&case.name, deviation, STATES as f64 * weight_bound);

        let mut cumulated = Vec::with_capacity(times);
        for t in 0..times {
            let bound = steps * (t as f64 + 1.0);
            cumulated.push(bound);
            let mut sum = 0.0f64;
            for (s, column) in case.pstate.iter().enumerate() {
                sum += (got.probabilities[t * STATES + s] - column[t]).abs();
            }
            probability_worst.see(&case.name, sum, bound);
        }

        // The integral of a step function. The probabilities differ by at most
        // the cumulated bound at each time, the widths are differences of
        // integers or half-integers and exact, and the sum of positive terms
        // adds `times eps` of the length of the interval in each
        // implementation.
        for (k, &tau) in case.tau.iter().enumerate() {
            let length = tau - case.t0;
            let last = case.time.partition_point(|&t| t <= tau);
            let bound =
                length * cumulated[last.max(1) - 1] + 2.0 * (times as f64 + 1.0) * eps * length;
            let mine = integrate(
                &got.probabilities,
                &case.time,
                &case.p0,
                STATES,
                tau,
                case.t0,
            );
            for (value, expected) in mine.iter().zip(&case.rmean[k]) {
                rmean_worst.see(&case.name, (value - expected).abs(), bound);
            }
        }
        // "common" is the last time.
        let tau = case.time[times - 1];
        let length = tau - case.t0;
        let bound = length * cumulated[times - 1] + 2.0 * (times as f64 + 1.0) * eps * length;
        let mine = integrate(
            &got.probabilities,
            &case.time,
            &case.p0,
            STATES,
            tau,
            case.t0,
        );
        for (value, expected) in mine.iter().zip(&case.rmean_common) {
            common_worst.see(&case.name, (value - expected).abs(), bound);
        }
    }
    for (name, w) in [
        ("weight at risk", &at_risk_worst),
        ("probabilities", &probability_worst),
        ("restricted mean", &rmean_worst),
        ("rmean = last time", &common_worst),
    ] {
        eprintln!(
            "{name:18} worst {:.3e} of the tolerance ({})",
            w.value, w.case
        );
    }
}
