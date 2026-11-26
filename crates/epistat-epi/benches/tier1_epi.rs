//! Wall time of the Mantel-Haenszel estimators, set against
//! `metafor::rma.mh` (`bench/tier1/bench.R`). The driver and the environment
//! are described in `bench/tier1/common.rs`.
//!
//! Methods: `mh_or` (the 2 x 2 tables of 200 strata built from one record per
//! person, then the common odds ratio, risk ratio and risk difference) and
//! `mh_rr` (one stratum per row, the common rate ratio and rate difference).

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_core::Level;
use epistat_epi::PersonTime2;
use epistat_epi::mantel_haenszel::{
    Pooled, odds_ratio, rate_difference, rate_ratio, risk_difference, risk_ratio,
    tables_from_records,
};

const SEED: u64 = 20_261_203;
const STRATA: u64 = 200;

fn generate_mh(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("mh", n), "stratum,exposed,case");
    for _ in 0..n {
        let s = rng.below(STRATA);
        let level = (s as f64 / STRATA as f64) - 0.5;
        let exposed = rng.bernoulli(0.2 + 0.3 * (level + 0.5));
        let p = 0.04 * (1.0 + 2.0 * level + if exposed { 0.6 } else { 0.0 });
        let case = rng.bernoulli(p.max(0.005));
        out.row(&[
            s as f64,
            f64::from(u8::from(exposed)),
            f64::from(u8::from(case)),
        ]);
    }
}

fn generate_mhr(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut out = Writer::create(&common::data_path("mhr", n), "x1,t1,x0,t0");
    for _ in 0..n {
        let (t1, t0) = (10.0 + 100.0 * rng.uniform(), 10.0 + 100.0 * rng.uniform());
        let rate = 0.02 * (0.5 * rng.normal()).exp();
        out.row(&[rng.poisson(t1 * rate * 1.5), t1, rng.poisson(t0 * rate), t0]);
    }
}

fn pooled(lines: &mut Lines, name: &str, p: &Pooled) {
    let e = p.estimate;
    lines.push_opt(&format!("{name}_estimate"), e.map(|e| e.estimate));
    lines.push_opt(&format!("{name}_se"), e.and_then(|e| e.se));
    lines.push_opt(&format!("{name}_lower"), e.and_then(|e| e.lower()));
    lines.push_opt(&format!("{name}_upper"), e.and_then(|e| e.upper()));
}

fn main() {
    let (method, n) = (common::method(), common::size());
    match method.as_str() {
        "gen_mh" => return generate_mh(n),
        "gen_mhr" => return generate_mhr(n),
        _ => {}
    }
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "mh_or" => {
            let t = Table::read(&common::data_path("mh", n));
            let stratum: Vec<usize> = t.ints("stratum");
            let (exposed, case) = (t.flags("exposed"), t.flags("case"));
            let (times, results) = common::timed(common::reps(), || {
                let tables = tables_from_records(&stratum, &exposed, &case).expect("tables");
                (
                    odds_ratio(&tables, Level::default()).expect("or"),
                    risk_ratio(&tables, Level::default()).expect("rr"),
                    risk_difference(&tables, Level::default()).expect("rd"),
                )
            });
            pooled(&mut lines, "or", &results.0);
            pooled(&mut lines, "rr", &results.1);
            pooled(&mut lines, "rd", &results.2);
            times
        }
        "mh_rr" => {
            let t = Table::read(&common::data_path("mhr", n));
            let strata: Vec<PersonTime2> = (0..t.len())
                .map(|i| PersonTime2 {
                    exposed_events: t.col("x1")[i],
                    exposed_time: t.col("t1")[i],
                    unexposed_events: t.col("x0")[i],
                    unexposed_time: t.col("t0")[i],
                })
                .collect();
            let (times, results) = common::timed(common::reps(), || {
                (
                    rate_ratio(&strata, Level::default()).expect("rate ratio"),
                    rate_difference(&strata, Level::default()).expect("rate difference"),
                )
            });
            pooled(&mut lines, "irr", &results.0);
            pooled(&mut lines, "ird", &results.1);
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
