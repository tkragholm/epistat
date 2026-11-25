//! Wall time of the imputation and two-stage estimators, set against
//! `didimputation::did_imputation` and `did2s::did2s` (`bench/tier1/bench.R`).
//! The driver and the environment are described in `bench/tier1/common.rs`.
//!
//! The data (`stag`) are `N / 10` units over ten periods with a tenth of the
//! rows missing, cohorts first treated in periods 4, 6 and 8 and a
//! never-treated group, effects that grow with the time since treatment, one
//! covariate correlated with treatment and clusters of 50 units. Methods:
//! `imputation_es` (every horizon, clustered) and `did2s_es` (relative-period
//! indicators with reference -1, clustered). The timed part includes the
//! first stage, the estimates and the variance.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a benchmark program: sizes, codes and periods are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_did::{
    ImputationConfig, ImputationTerm, StaggeredPanel, TwoStageConfig, estimate_did_imputation,
    estimate_did2s,
};
use epistat_linalg::Mat;

const SEED: u64 = 20_261_125;
const PERIODS: usize = 10;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("stag", n), "id,t,g,cl,x1,y");
    let period_effect: Vec<f64> = (0..PERIODS).map(|_| rng.normal()).collect();
    for id in 0..n / PERIODS {
        let u = rng.normal();
        let g = match rng.below(4) {
            0 => 4.0,
            1 => 6.0,
            2 => 8.0,
            _ => 0.0,
        };
        for (s, period) in period_effect.iter().enumerate() {
            if rng.bernoulli(0.1) {
                continue;
            }
            let t = (s + 1) as f64;
            let treated = g > 0.0 && t >= g;
            let effect = if treated { 1.0 + 0.2 * (t - g) } else { 0.0 };
            let x1 = rng.normal() + if treated { 0.3 } else { 0.0 };
            let y = u + period + 0.5 * x1 + effect + rng.normal();
            out.row(&[id as f64, t, g, (id / 50) as f64, x1, y]);
        }
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_stag" {
        generate(n);
        return;
    }
    let t = Table::read_sized(&common::data_path("stag", n));
    let unit: Vec<u32> = t.ints("id");
    let time: Vec<i32> = t.ints("t");
    let cl: Vec<u32> = t.ints("cl");
    let g: Vec<Option<i32>> = t
        .col("g")
        .iter()
        .map(|&g| (g > 0.0).then_some(g as i32))
        .collect();
    let x = Mat::from_fn(t.len(), 1, |i, _| t.col("x1")[i]);
    let panel = StaggeredPanel {
        unit: &unit,
        time: &time,
        first_treated: &g,
        outcome: t.col("y"),
        weights: None,
        covariates: Some(x.as_ref()),
    };
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "imputation_es" => {
            let config = ImputationConfig {
                clusters: Some(&cl),
                ..ImputationConfig::event_study()
            };
            let (times, out) = common::timed(common::reps(), || {
                estimate_did_imputation(&panel, &config).expect("imputation")
            });
            for e in &out.estimates {
                if let ImputationTerm::Horizon(h) = e.term {
                    lines.push_opt(&format!("est_{h}"), e.estimate);
                    lines.push_opt(&format!("se_{h}"), e.std_error);
                }
            }
            times
        }
        "did2s_es" => {
            let config = TwoStageConfig {
                clusters: Some(&cl),
                ..TwoStageConfig::event_study()
            };
            let (times, out) = common::timed(common::reps(), || {
                estimate_did2s(&panel, &config).expect("did2s")
            });
            for term in &out.terms {
                let r = term.relative_period.expect("event study");
                lines.push_opt(&format!("est_{r}"), term.estimate);
                lines.push_opt(&format!("se_{r}"), term.std_error);
            }
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
