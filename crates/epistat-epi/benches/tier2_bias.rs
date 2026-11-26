//! Wall time of the probabilistic bias analysis, set against
//! `episensr::probsens` (`bench/tier2/bench.R`). The driver and the
//! environment are those of `bench/tier1/common.rs`; `bench/run.sh` is
//! the driver.
//!
//! `N` is the number of draws. Methods: `bias_exposure` (non-differential
//! exposure misclassification with a trapezoidal sensitivity and a triangular
//! specificity, with random error) and `bias_outcome` (outcome
//! misclassification with a truncated normal sensitivity). The two sides draw
//! differently, so the numbers are compared as information only.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::Lines;
use epistat_epi::Counts2x2;
use epistat_core::{Estimate, Level};
use epistat_epi::bias::{Dist, Kind, Pair, ProbOptions, probabilistic_misclass};

const TABLE: Counts2x2 = Counts2x2 {
    exposed_cases: 215.0,
    unexposed_cases: 83.0,
    exposed_noncases: 1021.0,
    unexposed_noncases: 1202.0,
};

fn push(lines: &mut Lines, name: &str, s: Option<Estimate>) {
    lines.push_opt(&format!("info_{name}_median"), s.map(|s| s.estimate));
    lines.push_opt(&format!("info_{name}_lower"), s.and_then(|s| s.lower()));
    lines.push_opt(&format!("info_{name}_upper"), s.and_then(|s| s.upper()));
}

fn main() {
    let (method, n) = (common::method(), common::size());
    let mut lines = Lines::default();
    let opts = ProbOptions {
        draws: n,
        seed: 20_261_223,
        random_error: true,
        level: Level::alpha(0.05).expect("a level"),
    };
    let (kind, se, sp) = match method.as_str() {
        "bias_exposure" => (
            Kind::Exposure,
            Pair::Same(Dist::Trapezoidal {
                a: 0.75,
                b: 0.85,
                c: 0.95,
                d: 1.0,
            }),
            Pair::Same(Dist::Triangular {
                lo: 0.9,
                mode: 0.97,
                hi: 1.0,
            }),
        ),
        "bias_outcome" => (
            Kind::Outcome,
            Pair::Same(Dist::TruncatedNormal {
                lo: 0.7,
                hi: 0.95,
                mean: 0.85,
                sd: 0.05,
            }),
            Pair::Same(Dist::Uniform { lo: 0.95, hi: 1.0 }),
        ),
        other => panic!("unknown method {other}"),
    };
    let (times, result) = common::timed(common::reps(), || {
        probabilistic_misclass(&TABLE, kind, &se, &sp, &opts).expect("bias analysis")
    });
    push(&mut lines, "rr", result.risk_ratio);
    push(&mut lines, "or", result.odds_ratio);
    push(&mut lines, "rr_total", result.risk_ratio_total);
    push(&mut lines, "or_total", result.odds_ratio_total);
    lines.push("info_discarded", result.n_discarded as f64);
    common::finish(&method, n, &times, &lines);
}
