//! Wall time of the augmented inverse probability weighted estimator of the
//! average treatment effect, set against the same estimator written out in R
//! from `glm` fits (`bench/tier1/bench.R`). The driver and the environment are
//! described in `bench/tier1/common.rs`.
//!
//! The data (`aipw`) have two covariates, a treatment and a continuous
//! outcome. Method: `aipw` (Gaussian outcome models in each arm and a
//! logistic propensity model, the ATE). The point estimates are compared. The
//! standard error of the crate is the stacked one, which accounts for the
//! estimated models, and R's is the efficient one, so the two are written as
//! `info_se_*` and not compared.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_causal::Estimand;
use epistat_causal::aipw::{AipwOptions, GlmInput, aipw_glm};
use epistat_glm::Family;
use epistat_linalg::Mat;

const SEED: u64 = 20_261_207;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("aipw", n), "a,x1,x2,y");
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.bernoulli(0.4)));
        let p = 1.0 / (1.0 + (-(0.4 * x1 - 0.5 * x2 - 0.1)).exp());
        let a = f64::from(u8::from(rng.bernoulli(p)));
        let y = 1.0 + 0.5 * x1 - 0.8 * x2 + 1.2 * a + rng.normal();
        out.row(&[a, x1, x2, y]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_aipw" {
        generate(n);
        return;
    }
    assert_eq!(method, "aipw", "unknown method {method}");
    let t = Table::read(&common::data_path("aipw", n));
    let treated = t.flags("a");
    let (x1, x2) = (t.col("x1"), t.col("x2"));
    let design = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => x1[i],
        _ => x2[i],
    });
    let input = GlmInput {
        outcome: t.col("y"),
        treatment: &treated,
        outcome_design: design.as_ref(),
        propensity_design: design.as_ref(),
        weight: None,
        cluster: None,
    };
    let (times, fit) = common::timed(common::reps(), || {
        aipw_glm(
            &input,
            Family::Gaussian,
            &AipwOptions {
                estimand: Estimand::Ate,
                ..AipwOptions::default()
            },
        )
        .expect("aipw")
    });
    let mut lines = Lines::default();
    lines.push("mean1", fit.contrast.mean1.estimate);
    lines.push("mean0", fit.contrast.mean0.estimate);
    lines.push("difference", fit.contrast.difference.estimate);
    lines.push("info_se_mean1", fit.contrast.mean1.se.unwrap());
    lines.push("info_se_mean0", fit.contrast.mean0.se.unwrap());
    lines.push("info_se_difference", fit.contrast.difference.se.unwrap());
    common::finish(&method, n, &times, &lines);
}
