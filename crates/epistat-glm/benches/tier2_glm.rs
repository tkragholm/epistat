//! Wall time of generalised estimating equations and ordinal regression, set
//! against `geepack::geeglm` and `ordinal::clm` (`bench/tier2/glm_bench.R`).
//! The driver and the environment are those of `bench/tier1/common.rs`.
//!
//! Data kinds: `gpanel` has clusters of 2 to 6 rows at distinct waves of 1 to
//! 6, three covariates, a count `y` and a binary `b`, both with a cluster
//! effect. `gord` has five covariates, two more for the category-specific
//! terms and a five-category outcome `y`. Methods: `gee_exch` (`b`, logit,
//! exchangeable), `gee_ar1` (`y`, log, AR(1)), `gee_unstr` (`y`, log,
//! unstructured), `ordinal` (proportional odds, logit), `ordinal_ppo` (the
//! last two covariates category-specific) and `brant`. `brant` is set
//! against `brant::brant`, and `geeglm` has no counterpart for `gee_unstr`
//! because it stops with a segmentation fault when the waves of a cluster have
//! gaps.

#![allow(
    clippy::cast_possible_truncation,
    reason = "a benchmark program: sizes, codes and waves are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_core::Covariance;
use epistat_glm::{
    Correlation, Family, Gee, GeeData, Link, Ordinal, OrdinalData, OrdinalLink, brant,
};
use epistat_linalg::Mat;

const SEED: u64 = 20_261_203;

fn generate_panel(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("gpanel", n), "id,wave,x1,x2,x3,y,b");
    let (mut id, mut written) = (1.0, 0);
    while written < n {
        let size = (2 + rng.below(5) as usize).min(n - written);
        let mut waves: Vec<u32> = (1..=6).collect();
        for k in 0..size {
            let pick = k + rng.below((6 - k) as u64) as usize;
            waves.swap(k, pick);
        }
        let mut chosen = waves[..size].to_vec();
        chosen.sort_unstable();
        let u = 0.4 * rng.normal();
        for &wave in &chosen {
            let (x1, x2) = (rng.normal(), rng.normal());
            let x3 = f64::from(u8::from(rng.bernoulli(0.45)));
            let y = rng.poisson((0.2 + 0.3 * x1 - 0.2 * x2 + 0.4 * x3 + u).exp());
            let p = 1.0 / (1.0 + (-(-0.5 + 0.4 * x1 + 0.3 * x3 + u)).exp());
            let b = f64::from(u8::from(rng.bernoulli(p)));
            out.row(&[id, f64::from(wave), x1, x2, x3, y, b]);
        }
        id += 1.0;
        written += size;
    }
}

fn generate_ordinal(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut out = Writer::create(&common::data_path("gord", n), "x1,x2,x3,x4,x5,z1,z2,y");
    for _ in 0..n {
        let (x1, x2, x4, x5, z1) = (
            rng.normal(),
            rng.normal(),
            rng.normal(),
            rng.normal(),
            rng.normal(),
        );
        let x3 = f64::from(u8::from(rng.bernoulli(0.4)));
        let z2 = f64::from(u8::from(rng.bernoulli(0.5)));
        let u = rng.uniform();
        let noise = (u / (1.0 - u)).ln();
        let latent = 0.5 * x1 - 0.4 * x2 + 0.6 * x3 + 0.2 * x4 + 0.5 * z1 + 0.8 * z2 + noise;
        let y = [-1.5, -0.3, 0.8, 2.0]
            .iter()
            .filter(|&&c| latent > c)
            .count();
        out.row(&[x1, x2, x3, x4, x5, z1, z2, y as f64]);
    }
}

fn columns(t: &Table, names: &[&str], intercept: bool) -> Mat<f64> {
    let cols: Vec<&[f64]> = names.iter().map(|&name| t.col(name)).collect();
    let shift = usize::from(intercept);
    Mat::from_fn(t.len(), names.len() + shift, |i, j| {
        if intercept && j == 0 {
            1.0
        } else {
            cols[j - shift][i]
        }
    })
}

fn report(
    lines: &mut Lines,
    names: &[&str],
    coef: &[Option<f64>],
    cov: &Covariance,
    offset: usize,
) {
    for (j, name) in names.iter().enumerate() {
        lines.push_opt(&format!("coef_{name}"), coef[j]);
        lines.push_opt(
            &format!("se_{name}"),
            cov.get(offset + j, offset + j).map(f64::sqrt),
        );
    }
}

fn gee(method: &str, t: &Table) -> (Vec<f64>, Lines) {
    let x = columns(t, &["x1", "x2", "x3"], true);
    let cluster: Vec<u32> = t.ints("id");
    let wave: Vec<u32> = t.ints("wave");
    let (family, link, correlation, response) = match method {
        "gee_exch" => (
            Family::Binomial,
            Link::Logit,
            Correlation::Exchangeable,
            "b",
        ),
        "gee_ar1" => (Family::Poisson, Link::Log, Correlation::Ar1, "y"),
        _ => (Family::Poisson, Link::Log, Correlation::Unstructured, "y"),
    };
    let model = Gee::new(family, link, correlation);
    let (times, fit) = common::timed(common::reps(), || {
        let data = GeeData::builder(x.as_ref(), t.col(response), &cluster)
            .waves(&wave)
            .build()
            .expect("data");
        model.fit(&data).expect("gee")
    });
    let mut lines = Lines::default();
    report(
        &mut lines,
        &["int", "x1", "x2", "x3"],
        &fit.coefficients,
        &fit.covariance,
        0,
    );
    lines.push_all("alpha", &fit.alpha);
    lines.push("scale", fit.scale);
    (times, lines)
}

fn main() {
    let (method, n) = (common::method(), common::size());
    match method.as_str() {
        "gen_gpanel" => return generate_panel(n),
        "gen_gord" => return generate_ordinal(n),
        _ => {}
    }
    let (times, lines) = match method.as_str() {
        "gee_exch" | "gee_ar1" | "gee_unstr" => {
            gee(&method, &Table::read(&common::data_path("gpanel", n)))
        }
        "ordinal" | "ordinal_ppo" | "brant" => {
            let t = Table::read(&common::data_path("gord", n));
            let y: Vec<u32> = t.ints("y");
            let ppo = method == "ordinal_ppo";
            let names: &[&str] = if ppo {
                &["x1", "x2", "x3", "x4", "x5"]
            } else {
                &["x1", "x2", "x3", "x4", "x5", "z1", "z2"]
            };
            let x = columns(&t, names, false);
            let mut lines = Lines::default();
            if method == "brant" {
                let (times, test) = common::timed(common::reps(), || {
                    let data = OrdinalData::builder(x.as_ref(), &y).build().expect("data");
                    brant(&data).expect("brant")
                });
                lines.push("omnibus", test.omnibus.statistic.unwrap_or(f64::NAN));
                (times, lines)
            } else {
                let model = Ordinal::new(OrdinalLink::Logit);
                let w = columns(&t, &["z1", "z2"], false);
                let (times, fit) = common::timed(common::reps(), || {
                    let builder = OrdinalData::builder(x.as_ref(), &y);
                    let builder = if ppo {
                        builder.nominal(w.as_ref())
                    } else {
                        builder
                    };
                    model.fit(&builder.build().expect("data")).expect("ordinal")
                });
                let k1 = fit.thresholds.len();
                for (k, th) in fit.thresholds.iter().enumerate() {
                    lines.push(&format!("theta{k}"), *th);
                    lines.push_opt(
                        &format!("se_theta{k}"),
                        fit.covariance.get(k, k).map(f64::sqrt),
                    );
                }
                report(&mut lines, names, &fit.coefficients, &fit.covariance, k1);
                (times, lines)
            }
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
