//! Wall time of the imputation, two-stage, interaction-weighted and extended
//! two-way fixed-effects estimators, set against
//! `didimputation::did_imputation`, `did2s::did2s`, `fixest::sunab` and
//! `etwfe::etwfe` with `etwfe::emfx` (`bench/tier1/bench.R`).
//! The driver and the environment are described in `bench/tier1/common.rs`.
//!
//! The data (`stag`) are `N / 10` units over ten periods with a tenth of the
//! rows missing, cohorts first treated in periods 4, 6 and 8 and a
//! never-treated group, effects that grow with the time since treatment, one
//! covariate correlated with treatment and clusters of 50 units. Methods:
//! `imputation_es` (every horizon, clustered) and `did2s_es` (relative-period
//! indicators with reference -1, clustered), `sunab_es` (cohort by relative
//! period cells with the covariate, clustered, and their summaries by
//! relative period and overall) and `etwfe_es` (not-yet-treated controls,
//! the covariate, clustered, and the overall and event-time effects). The
//! data `stagc` have the same layout with a count outcome `yc`, Poisson with
//! a log mean that grows by 0.2 at treatment and 0.05 per period after, and a
//! binary outcome `yb` with the same effects on the logit scale:
//! `etwfe_poisson` and `etwfe_logit` fit them as `etwfe_es` does, with the
//! cohort and period dummies of the nonlinear models. The timed part includes
//! the fits, the estimates and the variances.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a benchmark program: sizes, codes and periods are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_did::{
    EtwfeConfig, EtwfeFamily, EtwfeGlmConfig, EtwfeType, ImputationConfig, ImputationTerm,
    StaggeredPanel, SunabConfig, SunabTerm, TwoStageConfig, estimate_did_imputation,
    estimate_did2s, estimate_etwfe, estimate_etwfe_glm, estimate_sunab,
};
use epistat_glm::Vcov;
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

/// The panel of `generate` with a count and a binary outcome.
fn generate_counts(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut out = Writer::create(&common::data_path("stagc", n), "id,t,g,cl,x1,yc,yb");
    let period_effect: Vec<f64> = (0..PERIODS).map(|_| 0.2 * rng.normal()).collect();
    for id in 0..n / PERIODS {
        let u = 0.3 * rng.normal();
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
            let effect = if treated { 0.2 + 0.05 * (t - g) } else { 0.0 };
            let x1 = rng.normal() + if treated { 0.3 } else { 0.0 };
            let eta = u + period + 0.2 * x1 + effect;
            let yc = rng.poisson((0.5 + eta).exp());
            let yb = f64::from(u8::from(rng.bernoulli(1.0 / (1.0 + (0.2 - eta).exp()))));
            out.row(&[id as f64, t, g, (id / 50) as f64, x1, yc, yb]);
        }
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_stag" {
        generate(n);
        return;
    }
    if method == "gen_stagc" {
        generate_counts(n);
        return;
    }
    let kind = if method.starts_with("etwfe_poisson") || method.starts_with("etwfe_logit") {
        "stagc"
    } else {
        "stag"
    };
    let t = Table::read_sized(&common::data_path(kind, n));
    let unit: Vec<u32> = t.ints("id");
    let time: Vec<i64> = t.ints("t");
    let cl: Vec<u32> = t.ints("cl");
    let g: Vec<Option<i64>> = t
        .col("g")
        .iter()
        .map(|&g| (g > 0.0).then_some(g as i64))
        .collect();
    let x = Mat::from_fn(t.len(), 1, |i, _| t.col("x1")[i]);
    let outcome = t.col(match method.as_str() {
        "etwfe_poisson" => "yc",
        "etwfe_logit" => "yb",
        _ => "y",
    });
    let columns = StaggeredPanel::builder(&unit, &time, &g, outcome).covariates(x.as_ref());
    let panel = columns.build().expect("panel");
    let clustered = columns.clusters(&cl).build().expect("panel");
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "imputation_es" => {
            let config = ImputationConfig::event_study();
            let (times, out) = common::timed(common::reps(), || {
                estimate_did_imputation(&clustered, &config).expect("imputation")
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
            let config = TwoStageConfig::event_study();
            let (times, out) = common::timed(common::reps(), || {
                estimate_did2s(&clustered, &config).expect("did2s")
            });
            for term in &out.terms {
                let r = term.relative_period.expect("event study");
                lines.push_opt(&format!("est_{r}"), term.estimate);
                lines.push_opt(&format!("se_{r}"), term.std_error);
            }
            times
        }
        "sunab_es" => {
            let config = SunabConfig::new();
            let (times, out) = common::timed(common::reps(), || {
                estimate_sunab(&clustered, &config).expect("sunab")
            });
            for e in &out.estimates {
                let name = match e.term {
                    SunabTerm::Period(r) => format!("{r}"),
                    SunabTerm::Att => "att".to_string(),
                    SunabTerm::Cohort(_) => continue,
                };
                lines.push(&format!("est_{name}"), e.estimate);
                lines.push_opt(&format!("se_{name}"), e.std_error);
            }
            times
        }
        "etwfe_es" => {
            let ways = [&cl[..]];
            let config = EtwfeConfig {
                vcov: Vcov::Cluster(&ways),
                ..EtwfeConfig::new()
            };
            let (times, out) = common::timed(common::reps(), || {
                estimate_etwfe(&panel, &config).expect("etwfe")
            });
            for e in &out.effects {
                let name = match e.term {
                    EtwfeType::Simple => "simple".to_string(),
                    EtwfeType::Event(r) => format!("{r}"),
                    _ => continue,
                };
                lines.push(&format!("est_{name}"), e.estimate);
                lines.push_opt(&format!("se_{name}"), e.std_error);
            }
            times
        }
        "etwfe_poisson" | "etwfe_logit" => {
            let ways = [&cl[..]];
            let family = if method == "etwfe_poisson" {
                EtwfeFamily::Poisson
            } else {
                EtwfeFamily::Logit
            };
            let config = EtwfeGlmConfig {
                vcov: Vcov::Cluster(&ways),
                ..EtwfeGlmConfig::new(family)
            };
            let (times, out) = common::timed(common::reps(), || {
                estimate_etwfe_glm(&panel, &config).expect("etwfe glm")
            });
            for e in &out.effects {
                let name = match e.term {
                    EtwfeType::Simple => "simple".to_string(),
                    EtwfeType::Event(r) => format!("{r}"),
                    _ => continue,
                };
                lines.push(&format!("est_{name}"), e.estimate);
                lines.push_opt(&format!("se_{name}"), e.std_error);
            }
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
