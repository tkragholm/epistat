//! Wall time of the two-part model of a semicontinuous outcome, set against
//! the two `glm` fits it stands for (`bench/tier1/bench.R`). The driver and
//! the environment are described in `bench/tier1/common.rs`.
//!
//! The data (`sc`) have a uniform and a binary covariate and an outcome that
//! is zero or exponential. Method: `two_part` (a logit for the outcome being
//! positive and a gamma regression with log link for the positive values, with
//! the model-based standard errors at unit dispersion for the gamma part).

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_linalg::Mat;
use epistat_semicontinuous::{FitOptions, ModelInput, fit_two_part_input};

const SEED: u64 = 20_261_209;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("sc", n), "x1,x2,y");
    for _ in 0..n {
        let x1 = 2.0 * rng.uniform() - 1.0;
        let x2 = f64::from(u8::from(rng.bernoulli(0.3)));
        let prob = 1.0 / (1.0 + (-(0.2 + 0.8 * x1 - 0.5 * x2)).exp());
        let y = if rng.bernoulli(prob) {
            (1.0 + 0.3 * x1 + 0.2 * x2).exp() * -rng.uniform().ln()
        } else {
            0.0
        };
        out.row(&[x1, x2, y]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_sc" {
        generate(n);
        return;
    }
    assert_eq!(method, "two_part", "unknown method {method}");
    let t = Table::read(&common::data_path("sc", n));
    let (x1, x2, y) = (t.col("x1"), t.col("x2"), t.col("y"));
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => x1[i],
        _ => x2[i],
    });
    let outcome = Mat::from_fn(n, 1, |i, _| y[i]);
    let input = ModelInput::new(x, outcome);
    let (times, (model, report)) = common::timed(common::reps(), || {
        fit_two_part_input(&input, FitOptions::default()).expect("two-part fit")
    });
    let mut lines = Lines::default();
    for (part, beta, se) in [
        ("logit", &model.beta_logit, &report.se_logit),
        ("gamma", &model.beta_gamma, &report.se_gamma),
    ] {
        for (j, name) in ["int", "x1", "x2"].iter().enumerate() {
            lines.push(&format!("coef_{part}_{name}"), beta[(j, 0)]);
            lines.push_opt(&format!("se_{part}_{name}"), se.as_ref().map(|s| s[(j, 0)]));
        }
    }
    common::finish(&method, n, &times, &lines);
}
