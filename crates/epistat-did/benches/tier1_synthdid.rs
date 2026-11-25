//! Wall time of synthetic difference-in-differences and synthetic control,
//! set against R's `synthdid` (`bench/tier1/synthdid.R`). The driver and the
//! environment are described in `bench/tier1/common.rs`.
//!
//! `N` is the number of units, a tenth of them treated, over 20 periods
//! (`sdid20`) or 100 (`sdid100`), the last fifth after adoption. The data
//! file is wide: `unit,treated,y0,y1,...`, the controls first. The outcome
//! is a three-factor model with unit and period effects, and the treated
//! units load more on the first factor. Methods: `synthdid_t20`,
//! `synthdid_t100` and `sc_t100` (the estimate), `synthdid_jk_t100` (the
//! estimate and its jackknife standard error) and `synthdid_placebo_t20`
//! (the estimate and a placebo standard error of 20 replicates; the draws
//! differ from R's, so its standard error is reported apart).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a benchmark program: sizes and counts are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_did::{
    SynthDidConfig, SynthDraws, SynthPanel, SynthVariance, estimate_synthdid, synthdid_variance,
};

const SEED: u64 = 20_261_125;
const FACTORS: usize = 3;

fn periods_of(kind: &str) -> usize {
    match kind {
        "sdid20" => 20,
        "sdid100" => 100,
        _ => panic!("unknown data kind {kind}"),
    }
}

fn generate(kind: &str, n: usize) {
    let periods = periods_of(kind);
    let pre = periods * 4 / 5;
    let treated = n / 10;
    let mut rng = Rng(SEED ^ periods as u64);
    let v: Vec<Vec<f64>> = (0..periods)
        .map(|_| (0..FACTORS).map(|_| rng.normal()).collect())
        .collect();
    let mut trend = 0.0;
    let period_effect: Vec<f64> = (0..periods)
        .map(|_| {
            trend += 0.5 * rng.normal();
            trend
        })
        .collect();
    let header = std::iter::once("unit,treated".to_string())
        .chain((0..periods).map(|t| format!("y{t}")))
        .collect::<Vec<_>>()
        .join(",");
    let mut out = Writer::create(&common::data_path(kind, n), &header);
    for unit in 0..n {
        let is_treated = unit >= n - treated;
        let mut u: Vec<f64> = (0..FACTORS).map(|_| rng.normal()).collect();
        if is_treated {
            u[0] += 1.0;
        }
        let alpha = 2.0 * rng.normal();
        let mut row = vec![unit as f64, f64::from(u8::from(is_treated))];
        for t in 0..periods {
            let factor: f64 = u.iter().zip(&v[t]).map(|(a, b)| a * b).sum();
            let effect = if is_treated && t >= pre { 1.0 } else { 0.0 };
            row.push(alpha + period_effect[t] + factor + effect + 0.5 * rng.normal());
        }
        out.row(&row);
    }
}

fn read(kind: &str, n: usize) -> SynthPanel {
    let periods = periods_of(kind);
    let t = Table::read_sized(&common::data_path(kind, n));
    let treated = t.flags("treated");
    let controls = treated.iter().filter(|&&w| !w).count();
    let columns: Vec<&[f64]> = (0..periods).map(|p| t.col(&format!("y{p}"))).collect();
    let mut y = Vec::with_capacity(n * periods);
    for unit in 0..t.len() {
        for column in &columns {
            y.push(column[unit]);
        }
    }
    SynthPanel::new(y, t.len(), periods, controls, periods * 4 / 5).expect("panel")
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if let Some(kind) = method.strip_prefix("gen_") {
        generate(kind, n);
        return;
    }
    let (kind, config) = match method.as_str() {
        "synthdid_t20" | "synthdid_placebo_t20" => ("sdid20", SynthDidConfig::synthdid()),
        "synthdid_t100" | "synthdid_jk_t100" => ("sdid100", SynthDidConfig::synthdid()),
        "sc_t100" => ("sdid100", SynthDidConfig::synthetic_control()),
        _ => panic!("unknown method {method}"),
    };
    let panel = read(kind, n);
    let (times, (estimate, se)) = common::timed(common::reps(), || {
        let estimate = estimate_synthdid(&panel, &config).expect("estimate");
        let se = match method.as_str() {
            "synthdid_jk_t100" => Some(
                synthdid_variance(&panel, &estimate, SynthVariance::Jackknife { clusters: None })
                    .expect("jackknife")
                    .se,
            ),
            "synthdid_placebo_t20" => Some(
                synthdid_variance(
                    &panel,
                    &estimate,
                    SynthVariance::Placebo(SynthDraws::Seeded {
                        replications: 20,
                        seed: SEED,
                    }),
                )
                .expect("placebo")
                .se,
            ),
            _ => None,
        };
        (estimate, se)
    });
    let mut lines = Lines::default();
    lines.push("estimate", estimate.estimate);
    lines.push_all("lambda", &estimate.lambda);
    lines.push_all("omega", &estimate.omega);
    lines.push_all("curve", &estimate.effect_curve);
    lines.push("info_lambda_steps", estimate.trace.lambda.len() as f64);
    lines.push("info_omega_steps", estimate.trace.omega.len() as f64);
    match method.as_str() {
        "synthdid_jk_t100" => lines.push_opt("se", se.flatten()),
        "synthdid_placebo_t20" => lines.push_opt("info_se", se.flatten()),
        _ => {}
    }
    common::finish(&method, n, &times, &lines);
}
