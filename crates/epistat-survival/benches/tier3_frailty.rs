//! Wall time of the shared gamma frailty Cox model of `epistat_survival::frailty`
//! set against `survival::coxph` with `frailty()` (`bench/tier3/frailty.R`). The
//! driver environment is described in `bench/tier1/common.rs`: one process times
//! one method on one size. `bench/run.sh` drives it.
//!
//! Run by hand, with `SCRATCH` outside the repository, and not above 10^5 rows
//! on the Mac:
//!
//! ```text
//! N=100000 METHOD=gen_frailty SCRATCH=/x cargo bench -p epistat-survival --bench tier3_frailty
//! N=100000 METHOD=frailty_fixed SCRATCH=/x REPS=3 cargo bench -p epistat-survival --bench tier3_frailty
//! ```
//!
//! The data `frailty` has `N` rows in clusters of 2 to 6 rows, about `N / 4`
//! clusters, with a gamma frailty of variance 0.5, a covariate of the row and a
//! binary covariate of the cluster, exit times on a grid of 0.01 (so that some
//! are tied), and for a third of the rows an entry time. Methods, all with Efron
//! ties:
//!
//! * `frailty_fixed`: theta fixed at 0.5, against `coxph` with
//!   `frailty(cluster, theta = 0.5)`, whose default `sparse = TRUE` gives the
//!   diagonal-form variance. The numbers compared are the coefficients, the
//!   diagonal-form standard errors, the partial and the corrected log
//!   likelihood. The exact standard errors are written as `info_`.
//! * `frailty_dense`: the same fit, against `coxph` with `sparse = FALSE`, whose
//!   variance is the exact one compared here; R's dense information has a row
//!   and a column per cluster, so it is run only at the smallest sizes.
//! * `frailty_search`: theta chosen by `coxph`'s own search
//!   ([`Theta::coxph`](epistat_survival::frailty::Theta::coxph)) against `coxph`'s
//!   default `method = "em"`, compared on theta and the coefficients.
//! * `frailty_profile`: theta at the maximum of the profile likelihood,
//!   compared on the corrected log likelihood only (`coxph` has no such search,
//!   and its own stops short of the maximum, so its theta is written as `info_`).
//! * `frailty_entry`: theta fixed at 0.5 with the entry times.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::Ties;
use epistat_survival::frailty::{FrailtyOptions, Theta, fit};

const SEED: u64 = 20_261_123;
const THETA: f64 = 0.5;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(
        &common::data_path("frailty", n),
        "time,start,status,x1,x2,cluster",
    );
    let mut row = 0usize;
    let mut cluster = 0u32;
    while row < n {
        cluster += 1;
        let size = (2 + usize::try_from(rng.below(5)).expect("small")).min(n - row);
        let z = rng.gamma(1.0 / THETA) * THETA;
        let x2 = f64::from(u8::from(rng.bernoulli(0.4)));
        for _ in 0..size {
            let x1 = rng.normal();
            let rate = 0.2 * z * (0.5 * x1 - 0.4 * x2).exp();
            let t = -rng.uniform().ln() / rate;
            let c = -rng.uniform().ln() / 0.1;
            let stop = (t.min(c) * 100.0).ceil() / 100.0;
            let entry = if rng.bernoulli(1.0 / 3.0) {
                (0.5 * stop * rng.uniform() * 100.0).floor() / 100.0
            } else {
                0.0
            };
            out.row(&[
                stop,
                entry,
                f64::from(u8::from(t <= c)),
                x1,
                x2,
                f64::from(cluster),
            ]);
            row += 1;
        }
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_frailty" {
        return generate(n);
    }
    let t = Table::read(&common::data_path("frailty", n));
    let status: Vec<i32> = t.ints("status");
    let cluster: Vec<u32> = t.ints("cluster");
    let (x1, x2) = (t.col("x1"), t.col("x2"));
    let x = Mat::from_fn(n, 2, |i, j| [x1[i], x2[i]][j]);
    let mut builder = SurvData::builder(t.col("time"), &status).cluster(&cluster);
    if method == "frailty_entry" {
        builder = builder.start(t.col("start"));
    }
    let data = builder.build().expect("data");
    let theta = match method.as_str() {
        "frailty_fixed" | "frailty_dense" | "frailty_entry" => Theta::Fixed(THETA),
        "frailty_search" => Theta::coxph(),
        "frailty_profile" => Theta::Profile,
        other => panic!("unknown method {other}"),
    };
    let options = FrailtyOptions {
        ties: Ties::Efron,
        theta,
        ..FrailtyOptions::default()
    };
    let (times, f) = common::timed(common::reps(), || {
        fit(&data, x.as_ref(), &options).expect("frailty")
    });
    let value = |v: Option<f64>| v.unwrap_or(f64::NAN);
    let mut lines = Lines::default();
    let dense = method == "frailty_dense";
    if method == "frailty_profile" {
        lines.push_opt("c_loglik", f.marginal_log_likelihood);
        lines.push("info_theta", f.theta);
    } else {
        lines.push_all(
            "beta",
            &f.coefficients.iter().map(|&b| value(b)).collect::<Vec<_>>(),
        );
        let (se, other) = if dense {
            (&f.standard_errors, &f.diagonal_standard_errors)
        } else {
            (&f.diagonal_standard_errors, &f.standard_errors)
        };
        lines.push_all("se", &se.iter().map(|&s| value(s)).collect::<Vec<_>>());
        lines.push_all(
            "info_se_other",
            &other.iter().map(|&s| value(s)).collect::<Vec<_>>(),
        );
        lines.push("loglik", f.log_partial_likelihood);
        lines.push_opt("c_loglik", f.marginal_log_likelihood);
        if method == "frailty_search" {
            lines.push("theta", f.theta);
        }
    }
    common::finish(&method, n, &times, &lines);
}
