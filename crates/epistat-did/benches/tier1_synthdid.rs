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
    clippy::cast_possible_wrap,
    reason = "a benchmark program: sizes and counts are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_core::{Bootstrap, Seed};
use epistat_did::{StaggeredPanel, SynthDidOptions, SynthDraws, SynthVariance, synthdid};

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

/// The wide file as long columns: unit `i` is row `i`, observed in periods
/// `0..T`, the treated first treated in period `4 T / 5`.
struct Long {
    unit: Vec<u32>,
    period: Vec<i64>,
    first_treated: Vec<Option<i64>>,
    outcome: Vec<f64>,
}

impl Long {
    fn panel(&self) -> StaggeredPanel<'_> {
        StaggeredPanel::builder(&self.unit, &self.period, &self.first_treated, &self.outcome)
            .build()
            .expect("panel")
    }
}

fn read(kind: &str, n: usize) -> Long {
    let periods = periods_of(kind);
    let t = Table::read_sized(&common::data_path(kind, n));
    let treated = t.flags("treated");
    let columns: Vec<&[f64]> = (0..periods).map(|p| t.col(&format!("y{p}"))).collect();
    let adoption = (periods * 4 / 5) as i64;
    let mut long = Long {
        unit: Vec::with_capacity(n * periods),
        period: Vec::with_capacity(n * periods),
        first_treated: Vec::with_capacity(n * periods),
        outcome: Vec::with_capacity(n * periods),
    };
    for unit in 0..t.len() {
        for (p, column) in columns.iter().enumerate() {
            long.unit.push(unit as u32);
            long.period.push(p as i64);
            long.first_treated.push(treated[unit].then_some(adoption));
            long.outcome.push(column[unit]);
        }
    }
    long
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if let Some(kind) = method.strip_prefix("gen_") {
        generate(kind, n);
        return;
    }
    let (kind, options) = match method.as_str() {
        "synthdid_t20" => ("sdid20", SynthDidOptions::default()),
        "synthdid_placebo_t20" => (
            "sdid20",
            SynthDidOptions {
                variance: SynthVariance::Placebo(SynthDraws::Seeded(Bootstrap::new(
                    20,
                    Seed(SEED),
                ))),
                ..SynthDidOptions::default()
            },
        ),
        "synthdid_t100" => ("sdid100", SynthDidOptions::default()),
        "synthdid_jk_t100" => (
            "sdid100",
            SynthDidOptions {
                variance: SynthVariance::Jackknife,
                ..SynthDidOptions::default()
            },
        ),
        "sc_t100" => ("sdid100", SynthDidOptions::synthetic_control()),
        _ => panic!("unknown method {method}"),
    };
    let long = read(kind, n);
    let panel = long.panel();
    let (times, estimate) = common::timed(common::reps(), || {
        synthdid::estimate(&panel, &options).expect("estimate")
    });
    let mut lines = Lines::default();
    lines.push("estimate", estimate.estimate.estimate);
    lines.push_all("lambda", &estimate.fit.lambda);
    lines.push_all("omega", &estimate.fit.omega);
    let curve: Vec<f64> = estimate.effect_curve.iter().map(|e| e.estimate).collect();
    lines.push_all("curve", &curve);
    lines.push("info_lambda_steps", estimate.fit.trace.lambda.len() as f64);
    lines.push("info_omega_steps", estimate.fit.trace.omega.len() as f64);
    match method.as_str() {
        "synthdid_jk_t100" => lines.push_opt("se", estimate.estimate.se),
        "synthdid_placebo_t20" => lines.push_opt("info_se", estimate.estimate.se),
        _ => {}
    }
    common::finish(&method, n, &times, &lines);
}
