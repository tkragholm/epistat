//! Wall time of the flexible parametric survival model of `epistat_survival::flexpar`
//! on the three scales, with and without delayed entry and a time-varying effect,
//! set against `flexsurv::flexsurvspline` and `rstpm2::stpm2`
//! (`bench/tier3/survival.R`). The driver environment is described in
//! `bench/tier1/common.rs`: one process times one method on one size.
//!
//! Run by hand, with `SCRATCH` outside the repository, and not above 10^5 rows
//! on the Mac:
//!
//! ```text
//! N=100000 METHOD=gen_flex SCRATCH=/x cargo bench -p epistat-survival --bench tier3_survival
//! N=100000 METHOD=flexpar_hazard SCRATCH=/x REPS=3 cargo bench -p epistat-survival --bench tier3_survival
//! ```
//!
//! Methods, all on the data `flex` (`N` rows with exit and entry times, status
//! and two covariates), with a baseline of 4 degrees of freedom:
//!
//! * `flexpar_hazard`, `flexpar_odds`, `flexpar_probit`: the model with both
//!   covariates and no entry. The numbers compared are the log likelihood,
//!   the coefficients of the covariates and the survivor function at three times.
//! * `flexpar_entry`: the hazard scale with delayed entry.
//! * `flexpar_tvc`: the hazard scale with delayed entry and a time-varying effect
//!   of the binary covariate with 2 degrees of freedom. The numbers compared are
//!   the log likelihood and the survivor function and the hazard ratio at three
//!   times, since the coefficients depend on the basis.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::flexpar::{
    FlexParOptions, FlexparContrast, FlexparPredictOptions, FlexparScale, Predict, TimeVarying,
    contrast, fit, predict,
};

const SEED: u64 = 20_261_208;
const DF: usize = 4;
const AT: [f64; 3] = [0.5, 1.5, 3.0];

/// Weibull times with covariate effects, exponential censoring and, for a third
/// of the rows, an entry time before the exit.
fn generate_flex(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("flex", n), "time,start,status,x1,x2");
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.bernoulli(0.4)));
        let t = (-rng.uniform().ln()).powf(1.0 / 1.3) * (-(0.5 * x1 - 0.4 * x2) / 1.3).exp() * 2.0;
        let c = -rng.uniform().ln() / 0.2;
        let stop = t.min(c);
        let entry = if rng.bernoulli(1.0 / 3.0) {
            0.4 * stop * rng.uniform()
        } else {
            0.0
        };
        out.row(&[stop, entry, f64::from(u8::from(t <= c)), x1, x2]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_flex" {
        return generate_flex(n);
    }
    let t = Table::read(&common::data_path("flex", n));
    let status: Vec<i32> = t.ints("status");
    let (x1, x2) = (t.col("x1"), t.col("x2"));
    let x = Mat::from_fn(n, 2, |i, j| [x1[i], x2[i]][j]);
    let entry = matches!(method.as_str(), "flexpar_entry" | "flexpar_tvc");
    let mut builder = SurvData::builder(t.col("time"), &status);
    if entry {
        builder = builder.start(t.col("start"));
    }
    let data = builder.build().expect("data");
    let scale = match method.as_str() {
        "flexpar_odds" => FlexparScale::Odds,
        "flexpar_probit" => FlexparScale::Probit,
        "flexpar_hazard" | "flexpar_entry" | "flexpar_tvc" => FlexparScale::Hazard,
        other => panic!("unknown method {other}"),
    };
    let tvc = [TimeVarying { column: 1, df: 2 }];
    let options = FlexParOptions {
        tvc: if method == "flexpar_tvc" {
            tvc.to_vec()
        } else {
            Vec::new()
        },
        ..FlexParOptions::new(scale, DF)
    };
    let (times, f) = common::timed(common::reps(), || {
        fit(&data, x.as_ref(), None, &options).expect("flexpar")
    });
    let mut lines = Lines::default();
    lines.push("loglik", f.log_likelihood);
    if method != "flexpar_tvc" {
        lines.push_all(
            "beta",
            &f.beta
                .iter()
                .map(|b| b.unwrap_or(f64::NAN))
                .collect::<Vec<_>>(),
        );
    }
    let profile = Mat::from_fn(1, 2, |_, j| [0.5, 1.0][j]);
    let o = FlexparPredictOptions::default();
    let s = predict(&f, profile.as_ref(), &AT, Predict::Survival, &o).expect("predict");
    for (p, &tau) in s[0].iter().zip(&AT) {
        lines.push(&format!("surv_{tau}"), p.estimate);
    }
    if method == "flexpar_tvc" {
        let reference = Mat::from_fn(1, 2, |_, j| [0.5, 0.0][j]);
        let hr = contrast(
            &f,
            profile.as_ref(),
            reference.as_ref(),
            &AT,
            FlexparContrast::HazardRatio,
            &o,
        )
        .expect("contrast");
        for (p, &tau) in hr[0].iter().zip(&AT) {
            lines.push(&format!("hr_{tau}"), p.estimate);
        }
    }
    common::finish(&method, n, &times, &lines);
}
