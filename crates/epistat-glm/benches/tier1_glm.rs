//! Wall time of the negative binomial model, risk regression with robust and
//! clustered variance, and the within-family estimator, set against
//! `MASS::glm.nb`, `glm` with `sandwich` and `fixest::feols`
//! (`bench/tier1/bench.R`). The driver and the environment are described in
//! `bench/tier1/common.rs`.
//!
//! The data (`reg`) are families of one to four members with a family effect:
//! three covariates, an offset, an overdispersed count `y`, a binary `b` and a
//! continuous `c`. Methods: `negbin` (`y` on the covariates with the offset,
//! the `glm.nb` stopping rules), `risk_robust` (risk ratios of `b`, HC1),
//! `risk_cluster` (the same with the sum of scores within families) and
//! `within` (`c` on the covariates with a family effect, clustered variance).

#![allow(
    clippy::cast_possible_truncation,
    reason = "a benchmark program: sizes, codes and days are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_glm::{NegBin, RiskScale, SandwichType, ThetaRule, Within, risk_regression};
use epistat_inference::Clusters;
use epistat_linalg::Mat;

const SEED: u64 = 20_261_202;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("reg", n), "fam,x1,x2,x3,off,y,b,c");
    let (mut fam, mut written) = (0.0, 0);
    while written < n {
        let size = (1 + rng.below(4) as usize).min(n - written);
        let u = 0.3 * rng.normal();
        for _ in 0..size {
            let (x1, x2) = (rng.normal(), rng.normal());
            let x3 = f64::from(u8::from(rng.bernoulli(0.45)));
            let off = 0.1 * rng.normal();
            let lambda = (0.3 + 0.4 * x1 - 0.3 * x2 + 0.5 * x3 + off + u).exp();
            let shape = rng.gamma(2.0) / 2.0;
            let y = rng.poisson(lambda * shape);
            let p = (-2.2 + 0.35 * x1 + 0.4 * x3 + u).exp().min(0.9);
            let b = f64::from(u8::from(rng.bernoulli(p)));
            let c = 1.0 + 0.5 * x1 - 0.8 * x2 + x3 + u + rng.normal();
            out.row(&[fam, x1, x2, x3, off, y, b, c]);
        }
        fam += 1.0;
        written += size;
    }
}

fn design(t: &Table, intercept: bool) -> Mat<f64> {
    let cols = [t.col("x1"), t.col("x2"), t.col("x3")];
    let shift = usize::from(intercept);
    Mat::from_fn(t.len(), 3 + shift, |i, j| {
        if intercept && j == 0 {
            1.0
        } else {
            cols[j - shift][i]
        }
    })
}

fn coefficients(lines: &mut Lines, names: &[&str], coef: &[Option<f64>], cov: &[Option<f64>]) {
    let p = coef.len();
    for (j, name) in names.iter().enumerate() {
        lines.push_opt(&format!("coef_{name}"), coef[j]);
        lines.push_opt(&format!("se_{name}"), cov[j * p + j].map(f64::sqrt));
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_reg" {
        generate(n);
        return;
    }
    let t = Table::read(&common::data_path("reg", n));
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "negbin" => {
            let x = design(&t, true);
            let model = NegBin {
                theta_rule: ThetaRule::RParity,
                ..NegBin::new()
            };
            let (times, fit) = common::timed(common::reps(), || {
                model
                    .fit(x.as_ref(), t.col("y"), None, Some(t.col("off")))
                    .expect("negbin")
            });
            coefficients(
                &mut lines,
                &["int", "x1", "x2", "x3"],
                &fit.glm.coefficients,
                &fit.glm.covariance,
            );
            lines.push_opt("theta", fit.theta);
            lines.push_opt("se_theta", fit.theta_se);
            times
        }
        "risk_robust" | "risk_cluster" => {
            let x = design(&t, true);
            let fam: Vec<u64> = t.ints("fam");
            let clusters = Clusters::from_labels(&fam);
            let clustered = method == "risk_cluster";
            let (times, fit) = common::timed(common::reps(), || {
                risk_regression(
                    RiskScale::Ratio,
                    x.as_ref(),
                    t.col("b"),
                    None,
                    None,
                    clustered.then_some(&clusters),
                    SandwichType::Hc1,
                )
                .expect("risk regression")
            });
            coefficients(
                &mut lines,
                &["int", "x1", "x2", "x3"],
                &fit.fit.coefficients,
                &fit.robust_covariance,
            );
            times
        }
        "within" => {
            let x = design(&t, false);
            let fam: Vec<u64> = t.ints("fam");
            let clusters = Clusters::from_labels(&fam);
            let model = Within::new();
            let (times, fit) = common::timed(common::reps(), || {
                model
                    .fit(x.as_ref(), t.col("c"), &clusters, None)
                    .expect("within")
            });
            coefficients(
                &mut lines,
                &["x1", "x2", "x3"],
                &fit.coefficients,
                &fit.robust_covariance,
            );
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
