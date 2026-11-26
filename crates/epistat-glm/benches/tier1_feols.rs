//! Wall time of linear models with absorbed fixed effects, set against
//! `fixest::feols` (`bench/tier1/bench.R`). The driver and the environment are
//! described in `bench/tier1/common.rs`.
//!
//! The data (`fe`) are `N / 10` units observed for ten consecutive periods
//! out of fifteen, starting at a period drawn for each unit, so that the panel
//! is unbalanced in calendar time and the solver iterates. Each row has a
//! firm, which a unit leaves for a random one with probability 0.2 a period.
//! Covariates: `x1` correlated with the unit effect, `x2` with the period
//! effect. Methods: `feols_fe2` (`y ~ x1 + x2 | id + t`, clustered by unit)
//! and `feols_fe3` (`| id + t + firm`, clustered by unit). The timed part
//! includes setting up the factors, the fit and the covariance.
//!
//! Two poorly connected worker-firm designs, `N / 5` workers observed for five
//! consecutive years from a year drawn among the first six, with `x1`
//! correlated with the firm effect and the fits `y ~ x1 | worker + firm`
//! (clustered by worker) and `| worker + firm + year`:
//!
//! - `hfe`: a worker moves to a random one of `N / 20` firms with probability
//!   0.05 a year, a sparse random graph of firms. Methods `feols_sparse2` and
//!   `feols_sparse3` with the projections, `feols_sparse2_cg` and
//!   `feols_sparse3_cg` with the conjugate gradients.
//! - `chn`: the firms, at most 1000, lie on a line and a worker moves to a
//!   neighbouring firm with probability 0.1 a year, a chain. Methods
//!   `feols_chain2_cg` and `feols_chain3_cg` with the conjugate gradients;
//!   the projections do not reach the tolerance in their 10 000 iterations
//!   there. `fixest` stops by its own criterion at its default tolerance,
//!   which on these designs leaves it further from the exact projection.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a benchmark program: sizes, codes and days are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_glm::{Absorb, AbsorbMethod, Feols, FixedEffects, Singletons, Ssc, Vcov};
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

/// Workers for five years from a start among the first six, moving between
/// firms as `move_firm` says; `kind` is `hfe` or `chn`.
fn generate_workers(kind: &str, n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path(kind, n), "worker,firm,year,x1,y");
    let workers = n / 5;
    let chain = kind == "chn";
    let firms = if chain {
        (workers / 20).clamp(2, 1000)
    } else {
        (n / 20).max(2)
    } as u64;
    let firm_effect: Vec<f64> = (0..firms).map(|_| rng.normal()).collect();
    let year_effect: Vec<f64> = (0..10).map(|_| rng.normal()).collect();
    for worker in 0..workers {
        let u = rng.normal();
        let start = rng.below(6) as usize;
        let mut firm = rng.below(firms);
        for s in 0..5 {
            if chain {
                if rng.bernoulli(0.1) {
                    firm = if rng.bernoulli(0.5) {
                        firm.saturating_sub(1)
                    } else {
                        (firm + 1).min(firms - 1)
                    };
                }
            } else if rng.bernoulli(0.05) {
                firm = rng.below(firms);
            }
            let year = start + s;
            let f = firm_effect[firm as usize];
            let x1 = rng.normal() + 0.5 * f;
            let y = u + f + year_effect[year] + 0.5 * x1 + rng.normal();
            out.row(&[worker as f64, firm as f64, year as f64, x1, y]);
        }
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    match method.as_str() {
        "gen_fe" => return generate(n),
        "gen_hfe" => return generate_workers("hfe", n),
        "gen_chn" => return generate_workers("chn", n),
        m if m.starts_with("feols_sparse") || m.starts_with("feols_chain") => {
            return workers(&method, n);
        }
        _ => {}
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
        let fit = Feols::default().fit(x.as_ref(), y, &fe).expect("feols");
        let cov = fit.vcov(&fe, Vcov::Cluster(&ways), Ssc::new()).expect("vcov");
        (fit.coefficients, cov.matrix)
    });
    let mut lines = Lines::default();
    for (j, name) in ["x1", "x2"].iter().enumerate() {
        lines.push_opt(&format!("coef_{name}"), coefficients[j]);
        lines.push_opt(&format!("se_{name}"), covariance.get(j, j).map(f64::sqrt));
    }
    common::finish(&method, n, &times, &lines);
}

/// The worker-firm designs.
fn workers(method: &str, n: usize) {
    let kind = if method.starts_with("feols_chain") {
        "chn"
    } else {
        "hfe"
    };
    let t = Table::read_sized(&common::data_path(kind, n));
    let codes = |name: &str| -> Vec<u32> { t.col(name).iter().map(|&v| v as u32).collect() };
    let (worker, firm, year) = (codes("worker"), codes("firm"), codes("year"));
    let x = Mat::from_fn(t.len(), 1, |i, _| t.col("x1")[i]);
    let y = t.col("y");
    let factors: Vec<&[u32]> = if method.contains('3') {
        vec![&worker, &firm, &year]
    } else {
        vec![&worker, &firm]
    };
    let absorb = Absorb {
        method: if method.ends_with("_cg") {
            AbsorbMethod::ConjugateGradient
        } else {
            AbsorbMethod::Projections
        },
        ..Absorb::new()
    };
    let ways: [&[u32]; 1] = [&worker];
    let (times, (coefficients, covariance, iterations)) = common::timed(common::reps(), || {
        let fe = FixedEffects::new(&factors, None, Singletons::Remove).expect("factors");
        let fit = Feols { absorb }.fit(x.as_ref(), y, &fe).expect("feols");
        let cov = fit.vcov(&fe, Vcov::Cluster(&ways), Ssc::new()).expect("vcov");
        (fit.coefficients, cov.matrix, fit.iterations)
    });
    let mut lines = Lines::default();
    lines.push_opt("coef_x1", coefficients[0]);
    lines.push_opt("se_x1", covariance.get(0, 0).map(f64::sqrt));
    eprintln!("{method} {n}: iterations {iterations:?}");
    common::finish(method, n, &times, &lines);
}
