//! Wall time of entropy balancing and CBPS, set against `WeightIt` and `CBPS`
//! on the same data (`bench/tier3/balance.R`). The driver and the environment are
//! described in `bench/tier1/common.rs`.
//!
//! The data (`bal`) have four covariates (two continuous, one binary, one
//! skewed), a treatment and a continuous outcome. Methods:
//!
//! * `ebal_att`: entropy balancing of the means for the effect on the treated;
//! * `ebal_ate_m2`: entropy balancing of the means and squares, the effect in
//!   the population;
//! * `cbps_exact`: just identified CBPS, the effect in the population;
//! * `cbps_over`: over identified CBPS, two-step, the effect on the treated.
//!
//! The numbers compared are the sum, the sum of squares and the largest of the
//! weights, and the weighted difference of the outcome. Over identified CBPS is
//! slow, and the driver runs it to a smaller size.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_causal::Estimand;
use epistat_causal::balance::OutcomeOptions;
use epistat_causal::balance::{
    BalanceWeights, CbpsInput, CbpsOptions, EntropyInput, EntropyOptions, Identification,
    WeightVariance, balance_columns, cbps, entropy_balance,
};
use epistat_linalg::Mat;

const SEED: u64 = 20_261_208;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("bal", n), "a,x1,x2,x3,x4,y");
    for _ in 0..n {
        let (x1, x2) = (rng.normal(), rng.normal());
        let x3 = f64::from(u8::from(rng.bernoulli(0.4)));
        let x4 = rng.gamma(2.0);
        let p = 1.0 / (1.0 + (-(-0.3 + 0.5 * x1 - 0.3 * x2 - 0.4 * x3 + 0.2 * x4)).exp());
        let a = f64::from(u8::from(rng.bernoulli(p)));
        let y = 1.0 + 0.5 * x1 - 0.8 * x3 + 0.3 * x4 + 1.2 * a + rng.normal();
        out.row(&[a, x1, x2, x3, x4, y]);
    }
}

fn report(lines: &mut Lines, w: &BalanceWeights, y: &[f64]) {
    let sum: f64 = w.weights().iter().sum();
    lines.push("sum_w", sum);
    lines.push("sum_w_square", w.weights().iter().map(|x| x * x).sum());
    lines.push(
        "max_w",
        w.weights()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let fit = w
        .mean_difference(
            y,
            None,
            &OutcomeOptions {
                variance: WeightVariance::Estimated,
                ..OutcomeOptions::default()
            },
        )
        .expect("mean difference");
    lines.push("difference", fit.contrast.difference.estimate);
    lines.push("info_se_difference", fit.contrast.difference.se.unwrap());
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_bal" {
        generate(n);
        return;
    }
    let t = Table::read(&common::data_path("bal", n));
    let treated = t.flags("a");
    let (x1, x2, x3, x4) = (t.col("x1"), t.col("x2"), t.col("x3"), t.col("x4"));
    let raw = Mat::from_fn(n, 4, |i, j| [x1[i], x2[i], x3[i], x4[i]][j]);
    let design = Mat::from_fn(n, 5, |i, j| if j == 0 { 1.0 } else { raw[(i, j - 1)] });
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "ebal_att" | "ebal_ate_m2" => {
            let (moments, target) = if method == "ebal_att" {
                (1, Estimand::Att)
            } else {
                (2, Estimand::Ate)
            };
            let cols = balance_columns(raw.as_ref(), moments, false).expect("columns");
            let (times, r) = common::timed(common::reps(), || {
                entropy_balance(
                    &EntropyInput {
                        treatment: &treated,
                        covariates: cols.as_ref(),
                        sample_weight: None,
                        base_weight: None,
                    },
                    &EntropyOptions {
                        estimand: target,
                        ..EntropyOptions::default()
                    },
                )
                .expect("entropy balancing")
            });
            report(&mut lines, &r.weights, t.col("y"));
            times
        }
        "cbps_exact" | "cbps_over" => {
            let (identification, target) = if method == "cbps_exact" {
                (Identification::Exact, Estimand::Ate)
            } else {
                (Identification::Over, Estimand::Att)
            };
            let (times, r) = common::timed(common::reps(), || {
                cbps(
                    &CbpsInput {
                        treatment: &treated,
                        design: design.as_ref(),
                        sample_weight: None,
                    },
                    &CbpsOptions {
                        estimand: target,
                        identification,
                        ..CbpsOptions::default()
                    },
                )
                .expect("cbps")
            });
            report(&mut lines, &r.weights, t.col("y"));
            lines.push_all("coef", &r.coefficients);
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
