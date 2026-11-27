//! Wall time of person-time tables and rates, standardised incidence ratios
//! against a rate table, overlap weights and the weights of a treatment that
//! changes over time, set against `survival::pyears`, `WeightIt::weightit`
//! and `ipw::ipwtm` (`bench/tier1/bench.R`). The driver and the environment
//! are described in `bench/tier1/common.rs`.
//!
//! Methods:
//!
//! * `pyears_rates`: person-time and events by 18 age bands, 23 calendar
//!   years and sex, then the directly standardised rate of each sex over the
//!   age bands and the rate ratio of the sexes (`pyr`, one row per person).
//! * `sir_ratetable`: the expected events of the same persons from a rate
//!   table of age, sex and year, and the SIR with its exact interval.
//! * `overlap`: overlap weights of a logistic propensity model (`ps`).
//! * `ipwtm`: stabilised weights of a treatment that is a Bernoulli variable
//!   in each of eight intervals (`ppw`, `N` is the number of rows).

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_core::Level;
use epistat_epi::rates::{direct_standardised, rate_ratio};
use epistat_epi::sir::{SirInterval, sir};
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::pyears::{Dimension, PyearsOptions, pyears};
use epistat_survival::ratetable::{RateDim, RateTable};
use epistat_survival::weighting::pooled::PooledOptions;
use epistat_survival::weighting::pooled::{PooledEvent, TreatmentHistory};
use epistat_survival::weighting::{Estimand, IptwOptions, iptw};

const SEED: u64 = 20_261_206;
const YEAR_DAYS: f64 = 365.25;
const AGE_BANDS: usize = 18;
const INTERVALS: usize = 8;

fn year_starts(first: i64, last: i64) -> Vec<f64> {
    (first..=last)
        .map(|y| epistat_core::day::from_civil(y, 1, 1) as f64)
        .collect()
}

fn age_cuts(bands: usize, years_per_band: f64) -> Vec<f64> {
    (0..=bands)
        .map(|i| (i as f64 * years_per_band * YEAR_DAYS).round())
        .collect()
}

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn generate_pyr(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("pyr", n), "age0,date0,dur,ev,sex");
    for _ in 0..n {
        // A fraction of a day, so that the age and the calendar date never reach
        // a cut point at the same instant, which R and this crate may order
        // differently.
        let age0 = (rng.uniform() * 80.0 * YEAR_DAYS).floor() + 0.37;
        let date0 =
            epistat_core::day::from_civil(2000, 1, 1) as f64 + (rng.uniform() * 4747.0).floor();
        let dur = 30.0 + (rng.uniform() * 2800.0).floor();
        let ev = f64::from(u8::from(
            rng.bernoulli(0.05 + 0.1 * age0 / (80.0 * YEAR_DAYS)),
        ));
        let sex = f64::from(u8::from(rng.bernoulli(0.5)));
        out.row(&[age0, date0, dur, ev, sex]);
    }
}

fn generate_ps(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut out = Writer::create(&common::data_path("ps", n), "a,x1,x2,x3,x4");
    for _ in 0..n {
        let (x1, x2) = (rng.normal(), rng.normal());
        let x3 = f64::from(u8::from(rng.bernoulli(0.4)));
        let x4 = rng.normal();
        let a = rng.bernoulli(expit(-0.2 + 0.5 * x1 - 0.4 * x2 + 0.3 * x3 + 0.2 * x4));
        out.row(&[f64::from(u8::from(a)), x1, x2, x3, x4]);
    }
}

fn generate_ppw(n: usize) {
    let mut rng = Rng(SEED + 2);
    let mut out = Writer::create(&common::data_path("ppw", n), "id,k,a,aprev,l,x1");
    let persons = n / INTERVALS;
    for id in 0..persons {
        let x1 = rng.normal();
        let (mut l, mut aprev) = (rng.normal(), 0.0);
        for k in 1..=INTERVALS {
            l = 0.5 * l + 0.8 * rng.normal() + 0.3 * aprev;
            let a = f64::from(u8::from(
                rng.bernoulli(expit(-1.4 + 0.7 * l + 0.3 * x1 + 1.2 * aprev)),
            ));
            out.row(&[id as f64, k as f64, a, aprev, l, x1]);
            aprev = a;
        }
    }
}

/// The daily rate of the table of the benchmark, by age in years, sex (0 or 1)
/// and calendar years since 1990.
fn table_rate(age: usize, sex: usize, year: usize) -> f64 {
    1e-6 * (0.09 * age as f64).exp() * (1.0 + 0.3 * sex as f64) * (1.0 - 0.01 * year as f64)
}

fn rate_table() -> RateTable {
    let (ages, years) = (120, 40);
    let mut rates = Vec::with_capacity(ages * 2 * years);
    for y in 0..years {
        for s in 0..2 {
            for a in 0..ages {
                rates.push(table_rate(a, s, y));
            }
        }
    }
    RateTable::new(
        vec![
            RateDim::Time {
                cuts: age_cuts(ages, 1.0),
            },
            RateDim::Factor { levels: 2 },
            RateDim::Time {
                cuts: year_starts(1990, 2030),
            },
        ],
        rates,
        1.0,
    )
    .expect("rate table")
}

fn main() {
    let (method, n) = (common::method(), common::size());
    match method.as_str() {
        "gen_pyr" => return generate_pyr(n),
        "gen_ps" => return generate_ps(n),
        "gen_ppw" => return generate_ppw(n),
        _ => {}
    }
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "pyears_rates" | "sir_ratetable" => {
            let t = Table::read(&common::data_path("pyr", n));
            let status: Vec<i32> = t.ints("ev");
            let sex: Vec<u32> = t.ints("sex");
            let data = SurvData::builder(t.col("dur"), &status)
                .build()
                .expect("data");
            if method == "pyears_rates" {
                let ages = age_cuts(AGE_BANDS, 5.0);
                let years = year_starts(2000, 2023);
                let dims = [
                    Dimension::Time {
                        origin: t.col("age0"),
                        cuts: &ages,
                    },
                    Dimension::Time {
                        origin: t.col("date0"),
                        cuts: &years,
                    },
                    Dimension::Factor {
                        codes: &sex,
                        levels: 2,
                    },
                ];
                let weights: Vec<f64> = (1..=AGE_BANDS).map(|a| a as f64).collect();
                let (times, out) = common::timed(common::reps(), || {
                    let table = pyears(&data, &dims, &PyearsOptions::default()).expect("pyears");
                    let mut by_sex = Vec::new();
                    for s in 0..2 {
                        let mut events = vec![0.0; AGE_BANDS];
                        let mut time = vec![0.0; AGE_BANDS];
                        for y in 0..years.len() - 1 {
                            for a in 0..AGE_BANDS {
                                let cell = table.index(&[a, y, s]).expect("cell");
                                events[a] += table.events[cell];
                                time[a] += table.person_time[cell];
                            }
                        }
                        let standardised =
                            direct_standardised(&events, &time, &weights, Level::default())
                                .expect("rate");
                        by_sex.push((events, time, standardised));
                    }
                    let (e0, t0) = (by_sex[0].0.iter().sum::<f64>(), by_sex[0].1.iter().sum());
                    let (e1, t1) = (by_sex[1].0.iter().sum::<f64>(), by_sex[1].1.iter().sum());
                    let ratio = rate_ratio(e1, t1, e0, t0, Level::default()).expect("rate ratio");
                    (by_sex, ratio)
                });
                for (s, (events, time, standardised)) in out.0.iter().enumerate() {
                    lines.push(&format!("events_sex{s}"), events.iter().sum());
                    lines.push(&format!("person_time_sex{s}"), time.iter().sum());
                    let rate = standardised.map(|s| s.rate);
                    lines.push_opt(&format!("std_rate_sex{s}"), rate.map(|r| r.estimate));
                    lines.push_opt(&format!("std_se_sex{s}"), rate.and_then(|r| r.se));
                }
                lines.push_opt("rate_ratio", out.1.map(|r| r.estimate));
                times
            } else {
                let table = rate_table();
                let origins = [t.col("age0"), t.col("date0")];
                let observed: f64 = status.iter().map(|&s| f64::from(s)).sum();
                let (times, result) = common::timed(common::reps(), || {
                    let expected = table.expected(&data, &origins, &[&sex]).expect("expected");
                    let ratio = sir(
                        observed,
                        expected.total,
                        Level::default(),
                        SirInterval::Exact,
                    )
                    .expect("sir");
                    (expected, ratio)
                });
                lines.push("observed", observed);
                lines.push("expected", result.0.total);
                lines.push("offtable", result.0.offtable);
                lines.push_opt("sir", result.1.map(|r| r.estimate));
                lines.push_opt("lower", result.1.and_then(|r| r.lower()));
                lines.push_opt("upper", result.1.and_then(|r| r.upper()));
                times
            }
        }
        "overlap" => {
            let t = Table::read(&common::data_path("ps", n));
            let treated = t.flags("a");
            let cols = [t.col("x1"), t.col("x2"), t.col("x3"), t.col("x4")];
            let design = Mat::from_fn(n, 5, |i, j| if j == 0 { 1.0 } else { cols[j - 1][i] });
            let options = IptwOptions {
                estimand: Estimand::Ato,
                stabilise: false,
                alpha: None,
            };
            let (times, fit) = common::timed(common::reps(), || {
                iptw::fit(&treated, design.as_ref(), &options).expect("overlap weights")
            });
            lines.push_all("coef", &fit.coefficients);
            let (mut w1, mut w0, mut wx1, mut wx0) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..n {
                if treated[i] {
                    w1 += fit.weights[i];
                    wx1 += fit.weights[i] * cols[0][i];
                } else {
                    w0 += fit.weights[i];
                    wx0 += fit.weights[i] * cols[0][i];
                }
            }
            lines.push("sum_w_treated", w1);
            lines.push("sum_w_control", w0);
            lines.push("sum_wx1_treated", wx1);
            lines.push("sum_wx1_control", wx0);
            lines.push(
                "mean_propensity",
                fit.propensity.iter().sum::<f64>() / n as f64,
            );
            times
        }
        "ipwtm" => {
            let t = Table::read(&common::data_path("ppw", n));
            let rows = t.len();
            let id: Vec<u32> = t.ints("id");
            let stop = t.col("k").to_vec();
            let start: Vec<f64> = stop.iter().map(|k| k - 1.0).collect();
            let status = vec![0_i32; rows];
            let treated = t.flags("a");
            let (l, x1, aprev) = (t.col("l"), t.col("x1"), t.col("aprev"));
            let z = Mat::from_fn(rows, 4, |i, j| match j {
                0 => 1.0,
                1 => l[i],
                2 => x1[i],
                _ => aprev[i],
            });
            let zn = Mat::from_fn(rows, 2, |i, j| if j == 0 { 1.0 } else { x1[i] });
            let data = SurvData::builder(&stop, &status)
                .start(&start)
                .id(&id)
                .build()
                .expect("data");
            let (times, fit) = common::timed(common::reps(), || {
                epistat_survival::weighting::pooled::fit(
                    &data,
                    PooledEvent::Treatment {
                        treated: &treated,
                        history: TreatmentHistory::All,
                    },
                    z.as_ref(),
                    Some(zn.as_ref()),
                    &PooledOptions { truncation: None },
                )
                .expect("ipwtm")
            });
            lines.push_all("coef_den", &fit.coefficients);
            lines.push_all(
                "coef_num",
                fit.numerator_coefficients.as_deref().unwrap_or(&[]),
            );
            lines.push("sum_w", fit.weight.iter().sum());
            lines.push("sum_w_square", fit.weight.iter().map(|w| w * w).sum());
            lines.push("max_w", fit.weight.iter().copied().fold(f64::MIN, f64::max));
            for i in [0, 7, 8, rows / 2, rows - 1] {
                lines.push(&format!("w{i}"), fit.weight[i]);
            }
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
