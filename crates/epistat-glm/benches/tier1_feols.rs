//! Wall time of linear models with absorbed fixed effects, set against
//! `fixest::feols` (`bench/tier1/bench.R`). The driver and the environment are
//! described in `bench/tier1/common.rs`.
//!
//! The data (`fe`) are `N / 10` units observed for ten consecutive periods
//! out of fifteen, starting at a period drawn for each unit, so that the panel
//! is unbalanced in calendar time and the projections iterate. Each row has a
//! firm, which a unit leaves for a random one with probability 0.2 a period.
//! Covariates: `x1` correlated with the unit effect, `x2` with the period
//! effect. Methods: `feols_fe2` (`y ~ x1 + x2 | id + t`, clustered by unit)
//! and `feols_fe3` (`| id + t + firm`, clustered by unit). The timed part
//! includes setting up the factors, the fit and the covariance.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a benchmark program: sizes, codes and days are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_glm::{Feols, FixedEffects, Singletons, Ssc, Vcov};
use epistat_linalg::Mat;

const SEED: u64 = 20_261_125;
const PERIODS: usize = 10;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("fe", n), "id,t,firm,x1,x2,y");
    let units = n / PERIODS;
    let firms = (units / 20).max(2) as u64;
    let period_effect: Vec<f64> = (0..15).map(|_| rng.normal()).collect();
    for id in 0..units {
        let u = rng.normal();
        let start = rng.below(6) as usize;
        let mut firm = rng.below(firms);
        for s in 0..PERIODS {
            let t = start + s;
            if rng.bernoulli(0.2) {
                firm = rng.below(firms);
            }
            let f = ((firm % 97) as f64 - 48.0) / 48.0;
            let x1 = rng.normal() + 0.5 * u;
            let x2 = rng.normal() + period_effect[t];
            let y = u + period_effect[t] + f + 0.5 * x1 - 0.3 * x2 + rng.normal();
            out.row(&[id as f64, t as f64, firm as f64, x1, x2, y]);
        }
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_fe" {
        generate(n);
        return;
    }
    let t = Table::read_sized(&common::data_path("fe", n));
    let codes = |name: &str| -> Vec<u32> { t.col(name).iter().map(|&v| v as u32).collect() };
    let (id, time, firm) = (codes("id"), codes("t"), codes("firm"));
    let x = Mat::from_fn(t.len(), 2, |i, j| {
        t.col(if j == 0 { "x1" } else { "x2" })[i]
    });
    let y = t.col("y");
    let factors: Vec<&[u32]> = match method.as_str() {
        "feols_fe2" => vec![&id, &time],
        "feols_fe3" => vec![&id, &time, &firm],
        other => panic!("unknown method {other}"),
    };
    let ways: [&[u32]; 1] = [&id];
    let (times, (coefficients, covariance)) = common::timed(common::reps(), || {
        let fe = FixedEffects::new(&factors, None, Singletons::Remove).expect("factors");
        let fit = Feols::default().fit(y, x.as_ref(), &fe).expect("feols");
        let cov = fit.vcov(Vcov::Cluster(&ways), Ssc::new()).expect("vcov");
        (fit.coefficients, cov.matrix)
    });
    let mut lines = Lines::default();
    for (j, name) in ["x1", "x2"].iter().enumerate() {
        lines.push_opt(&format!("coef_{name}"), coefficients[j]);
        lines.push_opt(&format!("se_{name}"), covariance[j * 2 + j].map(f64::sqrt));
    }
    common::finish(&method, n, &times, &lines);
}
