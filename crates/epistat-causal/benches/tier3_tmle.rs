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
//!   the controls, which one call of `tmle` gives;
//! * `tmle_survival`: the risks by the end of interval 12 of a follow-up in 12
//!   intervals with censoring, on hazards and a propensity given by the formulas
//!   of the simulation. R has no package for it (survtmle is not on CRAN), so
//!   only the Rust side runs.
//!
//! The numbers compared are the estimates. The standard errors follow another
//! convention in `tmle` (see `epistat_causal::tmle`) and are reported apart.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_causal::aipw::{GlmInput, Target};
use epistat_causal::tmle::survival::{SurvivalInput, SurvivalOptions, tmle_survival};
use epistat_causal::tmle::{OutcomeModel, PropensityBound, TmleOptions, tmle_glm};
use epistat_glm::Family;
use epistat_linalg::Mat;

const SEED: u64 = 20_261_209;
/// The intervals of the follow-up.
const INTERVALS: usize = 12;

fn expit(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn propensity(x1: f64, x2: f64, x3: f64) -> f64 {
    expit(-0.2 + 0.6 * x1 - 0.3 * x2 - 0.5 * x3)
}

/// The event hazard of interval `k` (from 1).
fn event_hazard(k: usize, x1: f64, x3: f64, a: f64) -> f64 {
    expit(-3.0 + 0.05 * k as f64 + 0.6 * x1 + 0.4 * x3 - 0.5 * a)
}

/// The censoring hazard after interval `k`.
fn censoring_hazard(x2: f64, a: f64) -> f64 {
    expit(-3.5 + 0.3 * x2 + 0.3 * a)
}

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("tml", n), "a,x1,x2,x3,yb,yc,time,event");
    for _ in 0..n {
        let (x1, x2) = (rng.normal(), rng.normal());
        let x3 = f64::from(u8::from(rng.bernoulli(0.4)));
        let a = f64::from(u8::from(rng.bernoulli(propensity(x1, x2, x3))));
        let yb = f64::from(u8::from(
            rng.bernoulli(expit(-0.5 + 0.8 * x1 + 0.2 * x2 + 0.6 * x3 + 0.7 * a)),
        ));
        let yc = 1.0 + 0.5 * x1 - 0.3 * x2 - 0.8 * x3 + 1.2 * a + rng.normal();
        let (mut time, mut event) = (INTERVALS, 0.0);
        for k in 1..=INTERVALS {
            if rng.bernoulli(event_hazard(k, x1, x3, a)) {
                (time, event) = (k, 1.0);
                break;
            }
            if k < INTERVALS && rng.bernoulli(censoring_hazard(x2, a)) {
                time = k;
                break;
            }
        }
        out.row(&[a, x1, x2, x3, yb, yc, time as f64, event]);
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
    if method == "tmle_survival" {
        survival(&t, &treated);
        return;
    }
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

fn survival(t: &Table, treated: &[bool]) {
    let n = t.len();
    let (x1, x2, x3) = (t.col("x1"), t.col("x2"), t.col("x3"));
    let time: Vec<usize> = t.ints("time");
    let event = t.flags("event");
    let hazard = |a: f64| Mat::from_fn(n, INTERVALS, |i, k| event_hazard(k + 1, x1[i], x3[i], a));
    let censoring = |a: f64| Mat::from_fn(n, INTERVALS, |i, _| censoring_hazard(x2[i], a));
    let (h1, h0, c1, c0) = (hazard(1.0), hazard(0.0), censoring(1.0), censoring(0.0));
    let g: Vec<f64> = (0..n).map(|i| propensity(x1[i], x2[i], x3[i])).collect();
    let input = SurvivalInput {
        time: &time,
        event: &event,
        treatment: treated,
        hazard1: h1.as_ref(),
        hazard0: h0.as_ref(),
        censoring1: c1.as_ref(),
        censoring0: c0.as_ref(),
        propensity: &g,
        weight: None,
        cluster: None,
    };
    let (times, r) = common::timed(common::reps(), || {
        tmle_survival(&input, &SurvivalOptions::new(INTERVALS)).expect("tmle_survival")
    });
    let mut lines = Lines::default();
    lines.push("risk1", r.risk1.estimate);
    lines.push("risk0", r.risk0.estimate);
    lines.push("info_se_difference", r.difference.se);
    common::finish("tmle_survival", n, &times, &lines);
}
