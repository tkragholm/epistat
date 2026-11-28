//! Wall time of net survival and excess hazard models of
//! `epistat_survival::relsurv`, set against `relsurv::rs.surv`,
//! `popEpi::survtab`, `popEpi::relpois` and `rstpm2::stpm2` with `bhazard`
//! (`bench/tier3/relsurv.R`). The driver environment is described in
//! `bench/tier1/common.rs`: one process times one method on one size.
//!
//! Run by hand, with `SCRATCH` outside the repository, and not above 10^5 rows
//! on the Mac:
//!
//! ```text
//! N=100000 METHOD=gen_rel SCRATCH=/x cargo bench -p epistat-survival --bench tier3_relsurv
//! N=100000 METHOD=net_pp SCRATCH=/x REPS=3 cargo bench -p epistat-survival --bench tier3_relsurv
//! ```
//!
//! The data `rel` has `N` persons with age in days and date at diagnosis, sex,
//! entry and exit in days, status and two covariates; the rate table is the
//! synthetic daily table of `tests/relsurv_sim`, which the R script builds from
//! the same formula. The expected hazards are built inside the timed part.
//!
//! * `net_pp`, `net_e2`: Pohar Perme and Ederer II at every exit time, without
//!   delayed entry, against `rs.surv(precision = 1)`. The numbers compared are
//!   the net survival at 1, 3 and 5 years. They agree to about 1e-5: relsurv's
//!   steps of a day start at the half days of the censorings and so straddle
//!   the cuts of the table, and its Pohar Perme integral is an approximation
//!   (see the `relsurv` module).
//! * `net_pp_monthly`: Pohar Perme on monthly intervals, against `survtab` on
//!   monthly breaks, whose weights differ (see the `relsurv` module), so its
//!   numbers are `info_`.
//! * `excess_poisson`: the piecewise excess hazard model on yearly intervals to
//!   5 years with both covariates and delayed entry, against `relpois` on the
//!   data split at the same intervals and at the cuts of the table, with its
//!   default convergence of 1e-8. The numbers compared are the coefficients.
//! * `excess_flexpar`: the flexible excess hazard model with 4 degrees of
//!   freedom, against `stpm2(bhazard = )`: the log likelihood and the
//!   coefficients of the covariates.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::flexpar::{FlexParOptions, FlexparScale};
use epistat_survival::ratetable::{RateDim, RateTable};
use epistat_survival::relsurv::{
    ExcessOptions, ExpectedHazard, Grid, NetMethod, NetOptions, excess_hazard, net_survival,
};

const SEED: u64 = 20_261_123;
const DAY0: f64 = 3652.0;
const YEAR: f64 = 365.25;

fn table() -> RateTable {
    let mut age: Vec<f64> = (0..=110).map(|a| f64::from(a) * 365.0).collect();
    age.push(1e9);
    let mut year: Vec<f64> = (0..=45).map(|y| DAY0 + f64::from(y) * 365.0).collect();
    year.push(1e9);
    let mut rates = Vec::new();
    for y in 0..year.len() - 1 {
        for s in 0..2 {
            for a in 0..age.len() - 1 {
                let log =
                    -10.5 + 0.09 * a as f64 + if s == 0 { 0.35 } else { 0.0 } - 0.012 * y as f64;
                rates.push(log.exp() / 365.0);
            }
        }
    }
    RateTable::new(
        vec![
            RateDim::Time { cuts: age },
            RateDim::Factor { levels: 2 },
            RateDim::Time { cuts: year },
        ],
        rates,
        1.0,
    )
    .expect("table")
}

/// Deaths on whole days with an excess hazard of about 0.2 a year, censoring
/// from 150 to 2650 days, and, for a third, an entry before the exit.
fn generate_rel(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(
        &common::data_path("rel", n),
        "age0,date0,sex,start,time,status,x1,x2",
    );
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.bernoulli(0.4)));
        let t = 1.0 + (2000.0 * (-(0.4 * x1 + 0.3 * x2)).exp() * -rng.uniform().ln()).floor();
        let cens = 150.5 + (rng.uniform() * 2500.0).floor();
        let time = t.min(cens);
        let start = if rng.bernoulli(1.0 / 3.0) {
            (0.5 * time).floor() + 0.25
        } else {
            0.0
        };
        let age0 = (rng.uniform() * 40.0 * 365.0).floor() + 40.0 * 365.0;
        let date0 = DAY0 + 5.0 * 365.0 + (rng.uniform() * 15.0 * 365.0).floor();
        let sex = f64::from(u8::from(rng.bernoulli(0.5)));
        out.row(&[
            age0,
            date0,
            sex,
            start,
            time,
            f64::from(u8::from(t <= cens)),
            x1,
            x2,
        ]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_rel" {
        return generate_rel(n);
    }
    let t = Table::read(&common::data_path("rel", n));
    let status: Vec<i32> = t.ints("status");
    let sex: Vec<u32> = t.ints("sex");
    let (age0, date0, time, start) = (t.col("age0"), t.col("date0"), t.col("time"), t.col("start"));
    let (x1, x2) = (t.col("x1"), t.col("x2"));
    let rates = table();
    let entry = matches!(method.as_str(), "excess_poisson");
    let mut builder = SurvData::builder(time, &status);
    if entry {
        builder = builder.start(start);
    }
    let data = builder.build().expect("data");
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "net_pp" | "net_e2" | "net_pp_monthly" => {
            let opts = NetOptions {
                method: if method == "net_e2" {
                    NetMethod::EdererII
                } else {
                    NetMethod::PoharPerme
                },
                grid: if method == "net_pp_monthly" {
                    Grid::Width(30.4375)
                } else {
                    Grid::EventTimes
                },
                ..NetOptions::default()
            };
            let (times, curve) = common::timed(common::reps(), || {
                let e = ExpectedHazard::from_table(&rates, &data, &[age0, date0], &[&sex])
                    .expect("expected");
                net_survival(&data, &e, &opts)
                    .expect("net survival")
                    .remove(0)
            });
            let prefix = if method == "net_pp_monthly" {
                "info_"
            } else {
                ""
            };
            for years in [1.0, 3.0, 5.0] {
                let k = curve.times.partition_point(|&s| s <= years * YEAR);
                lines.push(&format!("{prefix}surv_{years}"), curve.survival[k - 1]);
            }
            times
        }
        "excess_poisson" => {
            let cuts: Vec<f64> = (0..=5).map(|y| f64::from(y) * YEAR).collect();
            let (times, f) = common::timed(common::reps(), || {
                let e = ExpectedHazard::from_table(&rates, &data, &[age0, date0], &[&sex])
                    .expect("expected");
                excess_hazard(&data, &[x1, x2], &e, &ExcessOptions::new(&cuts)).expect("excess")
            });
            let b: Vec<f64> = f
                .coefficients
                .iter()
                .map(|v| v.unwrap_or(f64::NAN))
                .collect();
            lines.push_all("coef", &b);
            times
        }
        "excess_flexpar" => {
            let years: Vec<f64> = time.iter().map(|v| v / YEAR).collect();
            let fit_data = SurvData::builder(&years, &status).build().expect("data");
            let x = Mat::from_fn(n, 2, |i, j| [x1[i], x2[i]][j]);
            let (times, f) = common::timed(common::reps(), || {
                let e = ExpectedHazard::from_table(&rates, &data, &[age0, date0], &[&sex])
                    .expect("expected");
                let bh: Vec<f64> = (0..n)
                    .map(|i| e.hazard(i, time[i]).expect("hazard") * YEAR)
                    .collect();
                let opts = FlexParOptions::new(FlexparScale::Hazard, 4);
                epistat_survival::flexpar::fit(&fit_data, x.as_ref(), Some(&bh), &opts)
                    .expect("flexpar")
            });
            lines.push("loglik", f.log_likelihood);
            lines.push_all(
                "beta",
                &f.beta
                    .iter()
                    .map(|b| b.unwrap_or(f64::NAN))
                    .collect::<Vec<_>>(),
            );
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
