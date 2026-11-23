//! Wall time of targeted maximum likelihood estimation, set against `tmle::tmle`
//! with its learners replaced by `glm` on the same data (`bench/tier3/tmle.R`).
//! The driver and the environment are described in `bench/tier1/common.rs`.
//!
//! The data (`tml`) have three covariates (two continuous, one binary), a
//! treatment, a binary and a continuous outcome. Methods, each with the
//! main-terms outcome model and `tmle`'s propensity bound:
//!
//! * `tmle_ate_binary`: the effect in the population on the binary outcome;
//! * `tmle_ate_gaussian`: the same on the continuous outcome;
//! * `tmle_all_binary`: the effects in the population, on the treated and on
//!   the controls, which one call of `tmle` gives.
//!
//! The numbers compared are the estimates. The standard errors follow another
//! convention in `tmle` (see `epistat_causal::tmle`) and are reported apart.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_causal::aipw::{GlmInput, Target};
use epistat_causal::tmle::{OutcomeModel, PropensityBound, TmleOptions, tmle_glm};
use epistat_glm::Family;
use epistat_linalg::Mat;

const SEED: u64 = 20_261_209;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("tml", n), "a,x1,x2,x3,yb,yc");
    let expit = |z: f64| 1.0 / (1.0 + (-z).exp());
    for _ in 0..n {
        let (x1, x2) = (rng.normal(), rng.normal());
        let x3 = f64::from(u8::from(rng.bernoulli(0.4)));
        let a = f64::from(u8::from(
            rng.bernoulli(expit(-0.2 + 0.6 * x1 - 0.3 * x2 - 0.5 * x3)),
        ));
        let yb = f64::from(u8::from(
            rng.bernoulli(expit(-0.5 + 0.8 * x1 + 0.2 * x2 + 0.6 * x3 + 0.7 * a)),
        ));
        let yc = 1.0 + 0.5 * x1 - 0.3 * x2 - 0.8 * x3 + 1.2 * a + rng.normal();
        out.row(&[a, x1, x2, x3, yb, yc]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_tml" {
        generate(n);
        return;
    }
    let t = Table::read(&common::data_path("tml", n));
    let treated = t.flags("a");
    let (x1, x2, x3) = (t.col("x1"), t.col("x2"), t.col("x3"));
    let design = Mat::from_fn(n, 4, |i, j| [1.0, x1[i], x2[i], x3[i]][j]);
    let (outcome, family, targets): (&[f64], Family, &[Target]) = match method.as_str() {
        "tmle_ate_binary" => (t.col("yb"), Family::Binomial, &[Target::Ate]),
        "tmle_ate_gaussian" => (t.col("yc"), Family::Gaussian, &[Target::Ate]),
        "tmle_all_binary" => (
            t.col("yb"),
            Family::Binomial,
            &[Target::Ate, Target::Att, Target::Atc],
        ),
        other => panic!("unknown method {other}"),
    };
    let input = GlmInput {
        outcome,
        treatment: &treated,
        outcome_design: design.as_ref(),
        propensity_design: design.as_ref(),
        weight: None,
        cluster: None,
    };
    let opts = TmleOptions {
        propensity_bound: PropensityBound::Tmle,
        ..TmleOptions::default()
    };
    let (times, results) = common::timed(common::reps(), || {
        targets
            .iter()
            .map(|&target| {
                tmle_glm(&input, family, OutcomeModel::Pooled, target, &opts).expect("tmle")
            })
            .collect::<Vec<_>>()
    });
    let mut lines = Lines::default();
    for (target, r) in targets.iter().zip(&results) {
        let name = match target {
            Target::Ate => "ate",
            Target::Att => "att",
            Target::Atc => "atc",
        };
        lines.push(name, r.difference.estimate);
        lines.push(&format!("info_se_{name}"), r.difference.se);
    }
    common::finish(&method, n, &times, &lines);
}
