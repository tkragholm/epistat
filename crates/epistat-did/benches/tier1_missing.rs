//! Wall time of the difference-in-differences estimator for a pre-treatment
//! outcome missing at random, set against the authors' Python replication
//! code, `fit_missingdid` of `dgp.py` (`bench/tier1/missing_did.py`).
//! The driver and the environment are described in `bench/tier1/common.rs`.
//!
//! The data (`mdid`) follow the simulation design of the paper (Testa,
//! Kennedy and Reimherr, Section 4): four standard normal covariates, a
//! logistic propensity, a linear pre-treatment outcome, an ATT of 5, and two
//! indicators of an observed baseline, `rx` logistic in the covariates and
//! the treatment and `ry` also in the post-treatment outcome. Every model is
//! fitted on the four covariates, so all are correctly specified. Methods:
//! `missing_marx` (outcome-independent assumption, `rx`), `missing_marxy1`
//! (outcome-dependent, `ry`, augmented nested regression) and
//! `missing_marxy1_cf5` (the same, cross-fitted over five folds, Rust only).
//! The Python code regresses the nested outcome on all units where the paper
//! and this crate regress it among the controls, so the outcome-dependent
//! estimates are written as `info_` values. The timed part is the nuisance
//! fits, the estimate and its standard error.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a benchmark program: sizes are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_did::{
    MissingAssumption, MissingConfig, MissingFolds, MissingModels, MissingPreData, att_missing_pre,
};
use epistat_linalg::Mat;

const SEED: u64 = 20_261_125;

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("mdid", n), "z1,z2,z3,z4,t,y0,y1,rx,ry");
    for _ in 0..n {
        let z: Vec<f64> = (0..4).map(|_| rng.normal()).collect();
        let t = rng.bernoulli(expit(-z[0] + 0.5 * z[1] - 0.25 * z[2] - 0.1 * z[3]));
        let y0 = 210.0 + 27.4 * z[0] + 13.7 * (z[1] + z[2] + z[3]) + rng.normal();
        let y1 = y0 + rng.normal() + if t { 5.0 } else { 0.0 };
        let index = -0.25 * z[0] - 0.1 * z[1] - 0.5 * z[2] + 0.3 * z[3] - if t { 0.2 } else { 0.0 };
        let rx = rng.bernoulli(expit(index));
        let ry = rng.bernoulli(expit(index + 0.01 * y1));
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        out.row(&[z[0], z[1], z[2], z[3], flag(t), y0, y1, flag(rx), flag(ry)]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_mdid" {
        generate(n);
        return;
    }
    let t = Table::read_sized(&common::data_path("mdid", n));
    let z = Mat::from_fn(t.len(), 4, |i, j| t.col(["z1", "z2", "z3", "z4"][j])[i]);
    let treated = t.flags("t");
    let (assumption, observed, folds) = match method.as_str() {
        "missing_marx" => (MissingAssumption::MarX, "rx", MissingFolds::None),
        "missing_marxy1" => (MissingAssumption::MarXY1, "ry", MissingFolds::None),
        "missing_marxy1_cf5" => (
            MissingAssumption::MarXY1,
            "ry",
            MissingFolds::CrossFit { folds: 5, seed: 1 },
        ),
        other => panic!("unknown method {other}"),
    };
    let pre: Vec<Option<f64>> = t
        .flags(observed)
        .iter()
        .zip(t.col("y0"))
        .map(|(&r, &y)| r.then_some(y))
        .collect();
    let data = MissingPreData {
        treated: &treated,
        outcome_pre: &pre,
        outcome_post: t.col("y1"),
        clusters: None,
    };
    let all = [0, 1, 2, 3];
    let models = MissingModels {
        propensity: &all,
        missingness: &all,
        outcome: &all,
        nested: &all,
    };
    let config = MissingConfig::default();
    let (times, est) = common::timed(common::reps(), || {
        att_missing_pre(&data, z.as_ref(), assumption, &models, folds, &config).expect("estimate")
    });
    let mut lines = Lines::default();
    let prefix = if assumption == MissingAssumption::MarX {
        ""
    } else {
        "info_"
    };
    lines.push(&format!("{prefix}att"), est.att);
    lines.push(&format!("{prefix}se"), est.se);
    common::finish(&method, n, &times, &lines);
}
