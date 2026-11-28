//! Wall time of linear and generalised linear mixed models with one grouping
//! factor, set against `lme4::lmer` and `lme4::glmer` at their default
//! controls (`bench/tier3/mixed.R`). The driver and the environment are those
//! of `bench/tier1/common.rs`, run by `bench/run.sh`.
//!
//! Data kinds (`METHOD=gen_KIND` writes them): `mix` has clusters of 1 to 9
//! rows (about `N / 5` clusters), `x1` normal, `x2` binary, `t` uniform on
//! (0, 5), a Gaussian `yg` with a correlated random intercept and slope on
//! `t`, a binary `yb` and a count `yc` with a random intercept. `pair` has
//! sibling pairs, `N / 2` clusters of 2 rows, with `x1`, `x2` and a binary
//! `yb` with a random intercept of standard deviation 1.5. Methods:
//! `lmm_intercept` and `lmm_slope` (REML), `glmm_laplace` and `glmm_agq`
//! (`yb` on `mix`, 1 and 7 nodes), `poisson_laplace` (`yc` on `mix`) and
//! `pairs_agq` (`yb` on `pair`, 7 nodes). The numbers written for comparison
//! are `θ`, `β` and the log likelihood. At its default controls `glmer` stops
//! about `1e-3` (relative) from the optimum of the Laplace fits on `mix` at
//! `10^4` and `10^5` rows, where its fit at `rhoend = 1e-12` and
//! `tolPwrss = 1e-13` agrees with this crate to `1e-7`, so the comparison of
//! those methods shows `glmer`'s stopping rule.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a benchmark program: sizes and labels are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_glm::{Family, Link};
use epistat_linalg::{Mat, MatRef};
use epistat_mixed::{Criterion, Error, Glmm, Lmm, MixedData};

const SEED: u64 = 20_261_223;

fn expit(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

fn generate_mix(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("mix", n), "g,x1,x2,t,yg,yb,yc");
    let (mut id, mut written) = (1.0, 0);
    while written < n {
        let size = (1 + rng.below(9) as usize).min(n - written);
        let b0 = 0.8 * rng.normal();
        let b1 = 0.3 * b0 + 0.2 * rng.normal();
        let c = 0.6 * rng.normal();
        for _ in 0..size {
            let (x1, x2, t) = (
                rng.normal(),
                f64::from(u8::from(rng.bernoulli(0.4))),
                5.0 * rng.uniform(),
            );
            let yg = 1.0 + 0.5 * x1 - 0.4 * x2 + 0.2 * t + b0 + b1 * t + rng.normal();
            let yb = f64::from(u8::from(
                rng.bernoulli(expit(-0.5 + 0.6 * x1 - 0.4 * x2 + b0)),
            ));
            let yc = rng.poisson((0.2 + 0.3 * x1 + 0.2 * x2 + c).exp());
            out.row(&[id, x1, x2, t, yg, yb, yc]);
        }
        id += 1.0;
        written += size;
    }
}

fn generate_pair(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut out = Writer::create(&common::data_path("pair", n), "g,x1,x2,yb");
    for pair in 0..n / 2 {
        let b = 1.5 * rng.normal();
        for _ in 0..2 {
            let (x1, x2) = (rng.normal(), f64::from(u8::from(rng.bernoulli(0.5))));
            let yb = f64::from(u8::from(
                rng.bernoulli(expit(-0.5 + 0.6 * x1 - 0.4 * x2 + b)),
            ));
            out.row(&[(pair + 1) as f64, x1, x2, yb]);
        }
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    match method.as_str() {
        "gen_mix" => return generate_mix(n),
        "gen_pair" => return generate_pair(n),
        _ => {}
    }
    let kind = if method == "pairs_agq" { "pair" } else { "mix" };
    let table = Table::read(&common::data_path(kind, n));
    let rows = table.len();
    let cluster: Vec<u32> = table.ints("g");
    let (x1, x2) = (table.col("x1"), table.col("x2"));
    let x = Mat::from_fn(rows, 3, |i, j| [1.0, x1[i], x2[i]][j]);
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "lmm_intercept" | "lmm_slope" => {
            let slope = (method == "lmm_slope").then(|| table.col("t"));
            let xt = Mat::from_fn(rows, 4, |i, j| [1.0, x1[i], x2[i], table.col("t")[i]][j]);
            let (times, fit) = common::timed(common::reps(), || {
                Lmm::new(Criterion::Reml)
                    .fit(&view(xt.as_ref(), table.col("yg"), &cluster, slope, None, None).unwrap())
                    .expect("lmer")
            });
            lines.push_all("theta", &fit.theta);
            lines.push_all("beta", &fit.beta);
            lines.push("loglik", fit.log_likelihood);
            times
        }
        _ => {
            let (family, link, response, nodes) = match method.as_str() {
                "glmm_laplace" => (Family::Binomial, Link::Logit, "yb", 1),
                "glmm_agq" | "pairs_agq" => (Family::Binomial, Link::Logit, "yb", 7),
                "poisson_laplace" => (Family::Poisson, Link::Log, "yc", 1),
                other => panic!("unknown method {other}"),
            };
            let (times, fit) = common::timed(common::reps(), || {
                Glmm::new(family, link, nodes)
                    .fit(
                        &view(x.as_ref(), table.col(response), &cluster, None, None, None).unwrap(),
                    )
                    .expect("glmer")
            });
            lines.push_all("theta", &[fit.theta]);
            lines.push_all("beta", &fit.beta);
            lines.push("loglik", fit.log_likelihood);
            times
        }
    };
    common::finish(&method, n, &times, &lines);
}

/// The data view of the given columns.
fn view<'a>(
    x: MatRef<'a, f64>,
    y: &'a [f64],
    groups: &'a [u32],
    slope: Option<&'a [f64]>,
    weights: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
) -> Result<MixedData<'a>, Error> {
    let mut data = MixedData::builder(x, y, groups);
    if let Some(slope) = slope {
        data = data.slope(slope);
    }
    if let Some(weights) = weights {
        data = data.weights(weights);
    }
    if let Some(offset) = offset {
        data = data.offset(offset);
    }
    data.build()
}
