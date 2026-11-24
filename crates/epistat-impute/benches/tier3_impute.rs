//! Wall time of one chain of five iterations of chained equations for a
//! single incomplete column, set against `mice::mice` (`bench/tier3/impute.R`).
//! The driver and the environment are those of `bench/tier1/common.rs`, run by
//! `bench/run.sh`.
//!
//! Data kind `imp` (`METHOD=gen_imp` writes it): `x1` and `x2` normal and
//! `x3` binary, complete, and three incomplete columns about 30 per cent
//! missing at random given the complete ones: `inc`, a skewed positive
//! number, `flag`, binary, and `cat`, a three-level code. Methods `pmm` and
//! `norm` impute `inc`, `logreg` imputes `flag` and `polyreg` imputes `cat`,
//! each from `x1`, `x2` and `x3` alone, with `m = 1` and five iterations. The
//! numbers written for comparison are the mean and variance of the imputed
//! cells (the share of each level for `cat`), which agree between the two
//! implementations only in distribution.
//!
//! `pmm_chains` is `pmm` with ten chains, `m = 10`, for the memory that grows
//! with the number of chains; it has no R counterpart in the harness.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "test code: codes and counts are small whole numbers, and exact comparisons are intended"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_impute::{Column, ImputeOptions, ImputeSpec, Incomplete, Method, impute};

const SEED: u64 = 20_261_209;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("imp", n), "x1,x2,x3,inc,flag,cat");
    for _ in 0..n {
        let (x1, x2) = (rng.normal(), rng.normal());
        let x3 = f64::from(u8::from(rng.bernoulli(0.4)));
        let inc = (3.0 + 0.3 * x1 + 0.2 * x2 + 0.5 * rng.normal()).exp();
        let flag = f64::from(u8::from(
            rng.bernoulli(1.0 / (1.0 + (-(0.4 * x1 - 0.3 * x3)).exp())),
        ));
        let cat = match rng.uniform() {
            u if u < 0.4 => 0.0,
            u if u < 0.4 + 0.3 + 0.1 * x1.tanh() => 1.0,
            _ => 2.0,
        };
        let mut missing = |p: f64| rng.uniform() < p / (1.0 + (-(x1 + 0.5 * x3)).exp()) * 1.4;
        let (mi, mf, mc) = (missing(0.6), missing(0.6), missing(0.6));
        out.row(&[
            x1,
            x2,
            x3,
            if mi { f64::NAN } else { inc },
            if mf { f64::NAN } else { flag },
            if mc { f64::NAN } else { cat },
        ]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_imp" {
        generate(n);
        return;
    }
    let table = Table::read(&common::data_path("imp", n));
    let target = match method.as_str() {
        "pmm" | "pmm_chains" | "norm" => "inc",
        "logreg" => "flag",
        "polyreg" => "cat",
        other => panic!("unknown method {other}"),
    };
    let y = table.col(target);
    let missing: Vec<bool> = y.iter().map(|v| v.is_nan()).collect();
    let filled: Vec<f64> = y
        .iter()
        .map(|v| if v.is_nan() { 0.0 } else { *v })
        .collect();
    let column = match target {
        "inc" => Column::Continuous(filled),
        "flag" => Column::Binary(filled.iter().map(|&v| v as u8).collect()),
        _ => Column::Nominal {
            codes: filled.iter().map(|&v| v as u32).collect(),
            levels: 3,
        },
    };
    let none = vec![false; n];
    let data = Incomplete::new(
        vec![
            Column::Continuous(table.col("x1").to_vec()),
            Column::Continuous(table.col("x2").to_vec()),
            Column::Binary(table.col("x3").iter().map(|&v| v as u8).collect()),
            column,
        ],
        &[none.clone(), none.clone(), none, missing],
    )
    .expect("data");
    let mut spec = ImputeSpec::default_for(&data);
    spec.methods[3] = Some(match method.as_str() {
        "pmm" | "pmm_chains" => Method::pmm(),
        "norm" => Method::Norm,
        "logreg" => Method::Logreg,
        _ => Method::Polyreg { proper: true },
    });
    let options = ImputeOptions {
        m: if method == "pmm_chains" { 10 } else { 1 },
        max_iterations: 5,
        seed: SEED,
    };
    let (times, imputed) = common::timed(common::reps(), || {
        impute(&data, &spec, &options).expect("impute")
    });
    let cells = &imputed.values[3][0];
    let count = cells.len() as f64;
    let mean = cells.iter().sum::<f64>() / count;
    let mut lines = Lines::default();
    lines.push("imputed", count);
    if target == "cat" {
        for level in [1.0, 2.0] {
            lines.push(
                &format!("share{level}"),
                cells.iter().filter(|&&v| v == level).count() as f64 / count,
            );
        }
    } else {
        lines.push("mean", mean);
        lines.push(
            "variance",
            cells.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (count - 1.0),
        );
    }
    common::finish(&method, n, &times, &lines);
}
