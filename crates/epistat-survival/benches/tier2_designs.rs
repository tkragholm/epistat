//! Wall time of the self-controlled case series, the case-crossover windows
//! with their exposure, and the attributable fraction function of a Cox model.
//! The driver and the environment are those of `bench/tier1/common.rs`, run by
//! `bench/run.sh`; the data are written by `METHOD=gen_KIND` from a fixed seed
//! and read by both sides, and are not timed. The R side is
//! `bench/tier2/designs.R`.
//!
//! Methods:
//!
//! * `sccs` (data `scc`): `N` people with one to three events, at most one
//!   exposure date, three risk windows and four age bands: the pieces and the
//!   fit with the robust variance.
//! * `crossover` (data `cxo`): `N` events of 600 people with time-stratified referents by
//!   month, a three-day hazard window and observation periods, and the days of
//!   exposure in each window from `N / 4` intervals.
//! * `af_cox` (data `afc`): `N` people, a Cox model with an interaction, the standardised
//!   risks at five times with standard errors, and the attributable fraction
//!   function from them (the fit is not timed).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::range_plus_one,
    reason = "a benchmark program: sizes, codes and days are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_glm::attributable::IntervalScale;
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_survival::crossover::{
    ExposureIntervals, Measure, Observation, Referent, StratumUnit, build_windows,
    exposure_in_windows,
};
use epistat_survival::sccs::{
    ExposureSeries, PieceSpec, SccsOptions, WindowSpec, build_pieces, fit as sccs_fit,
};
use epistat_survival::standardise::attributable::af_from_standardised;
use epistat_survival::standardise::{Counterfactual, Level, StandardiseOptions, standardise_cox};

const SEED: u64 = 20_261_207;

fn generate_sccs(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(
        &common::data_path("scc", n),
        "id,start,end,expo,ev1,ev2,ev3",
    );
    for id in 0..n {
        let a = rng.below(100) as f64;
        let b = a + 300.0 + rng.below(700) as f64;
        let expo = if rng.below(2) == 1 {
            a + rng.below(500) as f64
        } else {
            f64::NAN
        };
        let mut ev = [f64::NAN; 3];
        for k in 0..1 + rng.below(3) as usize {
            // distinct days: standardsccs keeps one event per person and day
            ev[k] = loop {
                let day = a + rng.below((b - a) as u64) as f64;
                if !ev[..k].contains(&day) {
                    break day;
                }
            };
        }
        out.row(&[id as f64, a, b, expo, ev[0], ev[1], ev[2]]);
    }
}

fn generate_crossover(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut out = Writer::create(&common::data_path("cxo", n), "set,person,event,start,end");
    let people = 600_u64;
    let event: Vec<i64> = (0..n).map(|_| 17_532 + rng.below(1_461) as i64).collect();
    let person: Vec<u64> = (0..n).map(|_| rng.below(people)).collect();
    let start: Vec<i64> = event.iter().map(|&t| t - rng.below(400) as i64).collect();
    let end: Vec<i64> = event.iter().map(|&t| t + rng.below(400) as i64).collect();
    for i in 0..n {
        out.row(&[
            0.0,
            person[i] as f64,
            event[i] as f64,
            start[i] as f64,
            end[i] as f64,
        ]);
    }
    for _ in 0..n / 4 {
        let p = rng.below(people);
        let s = 17_000 + rng.below(2_500) as i64;
        let e = s + rng.below(40) as i64;
        out.row(&[1.0, p as f64, f64::NAN, s as f64, e as f64]);
    }
}

fn generate_af_cox(n: usize) {
    let mut rng = Rng(SEED + 2);
    let mut out = Writer::create(&common::data_path("afc", n), "time,status,e,z");
    for _ in 0..n {
        let z = rng.normal();
        let e = f64::from(u8::from(rng.bernoulli(1.0 / (1.0 + (-0.4 * z).exp()))));
        let t = -rng.uniform().ln() / (0.1 * (0.6 * e + 0.4 * z).exp());
        let time = t.min(3.0 + 30.0 * rng.uniform());
        let status = f64::from(u8::from(rng.bernoulli(0.7)));
        out.row(&[time, status, e, z]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    match method.as_str() {
        "gen_scc" => return generate_sccs(n),
        "gen_cxo" => return generate_crossover(n),
        "gen_afc" => return generate_af_cox(n),
        _ => {}
    }
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "sccs" => {
            let t = Table::read(&common::data_path("scc", n));
            let (start, end) = (t.col("start"), t.col("end"));
            let (mut event_offsets, mut events) = (vec![0_usize], Vec::new());
            let (mut exposure_offsets, mut exposures) = (vec![0_usize], Vec::new());
            for i in 0..n {
                let expo = t.col("expo")[i];
                if !expo.is_nan() {
                    exposures.push(expo);
                }
                exposure_offsets.push(exposures.len());
                for name in ["ev1", "ev2", "ev3"] {
                    let e = t.col(name)[i];
                    if !e.is_nan() {
                        events.push(e);
                    }
                }
                event_offsets.push(events.len());
            }
            let windows = [
                WindowSpec {
                    from: -14.0,
                    to: 0.0,
                },
                WindowSpec {
                    from: 0.0,
                    to: 14.0,
                },
                WindowSpec {
                    from: 14.0,
                    to: 60.0,
                },
            ];
            let spec = PieceSpec {
                start,
                end,
                event_offsets: &event_offsets,
                events: &events,
                exposures: &[ExposureSeries {
                    offsets: &exposure_offsets,
                    dates: &exposures,
                    windows: &windows,
                }],
                age_cuts: &[200.0, 400.0, 700.0],
                calendar: None,
            };
            let opts = SccsOptions {
                robust: true,
                ..SccsOptions::default()
            };
            let (times, result) = common::timed(common::reps(), || {
                let pieces = build_pieces(&spec).expect("pieces");
                let length = pieces.length();
                let x = pieces.design();
                let f = sccs_fit(&pieces.data(&length), x.as_ref(), &opts).expect("sccs");
                (pieces.person.len(), f)
            });
            lines.push("pieces", result.0 as f64);
            for (j, c) in result.1.coefficients.iter().enumerate() {
                lines.push_opt(&format!("coef{j}"), *c);
            }
            lines.push_opt("loglik", result.1.log_likelihood);
            lines.push("iterations", result.1.iterations as f64);
            times
        }
        "crossover" => {
            let t = Table::read(&common::data_path("cxo", n));
            let set: Vec<u8> = t.ints("set");
            let rows_of = |k: u8| -> Vec<usize> { (0..t.len()).filter(|&i| set[i] == k).collect() };
            let (case_rows, interval_rows) = (rows_of(0), rows_of(1));
            let pick_i64 = |name: &str, rows: &[usize]| -> Vec<i64> {
                rows.iter().map(|&i| t.col(name)[i] as i64).collect()
            };
            let event = pick_i64("event", &case_rows);
            let person: Vec<u32> = pick_i64("person", &case_rows)
                .iter()
                .map(|&p| p as u32)
                .collect();
            let start = pick_i64("start", &case_rows);
            let end = pick_i64("end", &case_rows);
            let ex_person: Vec<u32> = pick_i64("person", &interval_rows)
                .iter()
                .map(|&p| p as u32)
                .collect();
            let ex_start = pick_i64("start", &interval_rows);
            let ex_end = pick_i64("end", &interval_rows);
            let referent = Referent::TimeStratified {
                unit: StratumUnit::Month,
            };
            let (times, result) = common::timed(common::reps(), || {
                let w = build_windows(
                    &event,
                    &person,
                    3,
                    0,
                    referent,
                    Some(Observation {
                        start: &start,
                        end: &end,
                    }),
                )
                .expect("windows");
                let days = exposure_in_windows(
                    &w,
                    &ExposureIntervals {
                        person: &ex_person,
                        start: &ex_start,
                        end: &ex_end,
                    },
                    Measure::Days,
                )
                .expect("exposure");
                (w.len(), days.iter().sum::<f64>())
            });
            lines.push("windows", result.0 as f64);
            lines.push("exposure_days", result.1);
            times
        }
        "af_cox" => {
            let t = Table::read(&common::data_path("afc", n));
            let (e, z, time) = (t.col("e"), t.col("z"), t.col("time"));
            let status: Vec<i32> = t.ints("status");
            let x = Mat::from_fn(n, 3, |i, j| [e[i], z[i], e[i] * z[i]][j]);
            let x0 = Mat::from_fn(n, 3, |i, j| if j == 1 { x[(i, j)] } else { 0.0 });
            let data = SurvData::builder(time, &status).build().expect("data");
            let options = CoxOptions {
                ties: Ties::Breslow,
                ..CoxOptions::default()
            };
            let model = fit(&data, x.as_ref(), &options).expect("cox");
            let at = [1.0, 2.0, 4.0, 8.0, 16.0];
            let (times, curve) = common::timed(common::reps(), || {
                let s = standardise_cox(
                    &data,
                    x.as_ref(),
                    &model,
                    &options,
                    &[
                        Level::single(Counterfactual::Design(x0.as_ref())),
                        Level::single(Counterfactual::Columns(&[])),
                    ],
                    &at,
                    &StandardiseOptions {
                        se: true,
                        ..StandardiseOptions::default()
                    },
                )
                .expect("standardise");
                af_from_standardised(
                    &s,
                    &at,
                    0,
                    1,
                    epistat_survival::Level::default(),
                    IntervalScale::LogComplement,
                )
                .expect("af")
            });
            for k in 0..at.len() {
                lines.push_opt(&format!("af{k}"), curve.af[k]);
                lines.push_opt(&format!("se{k}"), curve.se[k]);
            }
            times
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
