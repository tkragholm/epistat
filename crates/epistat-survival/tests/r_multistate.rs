//! `multistate::estimate` and `lowlevel::integrate` checked
//! against R 4.6.2's `survival::survfit(Surv(start, stop, state) ~ 1, id =,
//! istate =)` and the restricted mean time in state of `summary(fit, rmean =)`.
//! The bootstrap has no R counterpart and is not checked here.

use epistat_survival::SurvData;
use epistat_survival::lowlevel::integrate;
use epistat_survival::multistate::{MultistateOptions, estimate};
use epistat_testkit::load_json;
use serde::Deserialize;

const STATES: usize = 3;

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

/// The reference stores the rows of a subject one after the other and not the
/// subject. A row belongs to the subject of the row before it when it starts
/// where that one stopped, in the state it moved to.
#[expect(clippy::float_cmp, reason = "the reference times are integers")]
fn subject_ids(case: &Case) -> Vec<u32> {
    let mut id = 0_u32;
    let mut ids = Vec::with_capacity(case.start.len());
    for row in 0..case.start.len() {
        let continues =
            row > 0 && case.start[row] == case.stop[row - 1] && case.from[row] == case.to[row - 1];
        if row > 0 && !continues {
            id += 1;
        }
        ids.push(id);
    }
    ids
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
        // The states are the codes 1 to 3, the ranks plus one, and status 0 is
        // the censoring that the reference marks as a move to state rank 3.
        let id = subject_ids(case);
        let status: Vec<i32> = case
            .to
            .iter()
            .map(|&to| {
                if to as usize >= STATES {
                    0
                } else {
                    i32::try_from(to).unwrap() + 1
                }
            })
            .collect();
        let istate: Vec<u32> = case.from.iter().map(|&from| from + 1).collect();
        let builder = SurvData::builder(&case.stop, &status)
            .start(&case.start)
            .n_codes(u32::try_from(STATES).unwrap())
            .id(&id)
            .istate(&istate);
        // The cases without weights give every row the weight 1, which is
        // the default.
        let data = if case.weights == "none" {
            builder
        } else {
            builder.weight(&case.weight)
        }
        .build()
        .unwrap();
        let options = MultistateOptions {
            initial: Some(case.p0.clone()),
            ..MultistateOptions::default()
        };
        let got = estimate(&data, &case.time, &options).unwrap();
        assert_eq!(got.n_rows, case.start.len(), "{}", case.name);
        assert_eq!(got.n_used, case.start.len(), "{}", case.name);
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
            )
            .unwrap();
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
        )
        .unwrap();
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
