//! Wall time of the self-controlled case series, the case-crossover windows
//! with their exposure, and the attributable fraction function of a Cox model.
//! The driver and the environment are those of `bench/tier1/common.rs`, run by
//! `bench/tier2/run.sh`; the data are generated in the process from a fixed
//! seed and are not timed.
//!
//! Methods:
//!
//! * `sccs`: `N` people with one to three events, up to two exposure dates,
//!   three risk windows and four age bands: the pieces and the fit with the
//!   robust variance.
//! * `crossover`: `N` events of 600 people with time-stratified referents by
//!   month, a three-day hazard window and observation periods, and the days of
//!   exposure in each window from `N / 4` intervals.
//! * `af_cox`: `N` people, a Cox model with an interaction, the standardised
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

use common::{Lines, Rng};
use epistat_glm::attributable::IntervalScale;
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_survival::crossover::{
    ExposureIntervals, Measure, Observation, Referent, StratumUnit, build_windows,
    exposure_in_windows,
};
use epistat_survival::sccs::{PieceSpec, SccsOptions, WindowSpec, build_pieces, fit as sccs_fit};
use epistat_survival::standardise::attributable::af_from_standardised;
use epistat_survival::standardise::{Counterfactual, Level, StandardiseOptions, standardise_cox};

const SEED: u64 = 20_261_207;

fn main() {
    let (method, n) = (common::method(), common::size());
    let mut rng = Rng(SEED);
    let mut lines = Lines::default();
    let times = match method.as_str() {
        "sccs" => {
            let (mut start, mut end) = (Vec::new(), Vec::new());
            let (mut event_offsets, mut events) = (vec![0_usize], Vec::new());
            let (mut exposure_offsets, mut exposures) = (vec![0_usize], Vec::new());
            for _ in 0..n {
                let a = rng.below(100) as f64;
                let b = a + 300.0 + rng.below(700) as f64;
                start.push(a);
                end.push(b);
                for _ in 0..rng.below(3) {
                    exposures.push(a + rng.below(500) as f64);
                }
                exposure_offsets.push(exposures.len());
                for _ in 0..1 + rng.below(3) {
                    events.push(a + rng.below((b - a) as u64) as f64);
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
                start: &start,
                end: &end,
                event_offsets: &event_offsets,
                events: &events,
                exposure_offsets: &exposure_offsets,
                exposures: &exposures,
                age_cuts: &[200.0, 400.0, 700.0],
                windows: &windows,
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
            let people = 600_u64;
            let event: Vec<i64> = (0..n).map(|_| 17_532 + rng.below(1_461) as i64).collect();
            let person: Vec<u32> = (0..n).map(|_| rng.below(people) as u32).collect();
            let start: Vec<i64> = event.iter().map(|&t| t - rng.below(400) as i64).collect();
            let end: Vec<i64> = event.iter().map(|&t| t + rng.below(400) as i64).collect();
            let m = n / 4;
            let ex_person: Vec<u32> = (0..m).map(|_| rng.below(people) as u32).collect();
            let ex_start: Vec<i64> = (0..m).map(|_| 17_000 + rng.below(2_500) as i64).collect();
            let ex_end: Vec<i64> = ex_start.iter().map(|&s| s + rng.below(40) as i64).collect();
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
            let z: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let e: Vec<f64> = (0..n)
                .map(|i| f64::from(u8::from(rng.bernoulli(1.0 / (1.0 + (-0.4 * z[i]).exp())))))
                .collect();
            let time: Vec<f64> = (0..n)
                .map(|i| {
                    let t = -rng.uniform().ln() / (0.1 * (0.6 * e[i] + 0.4 * z[i]).exp());
                    t.min(3.0 + 30.0 * rng.uniform())
                })
                .collect();
            let status: Vec<i32> = (0..n).map(|_| i32::from(rng.bernoulli(0.7))).collect();
            let x = Mat::from_fn(n, 3, |i, j| [e[i], z[i], e[i] * z[i]][j]);
            let x0 = Mat::from_fn(n, 3, |i, j| if j == 1 { x[(i, j)] } else { 0.0 });
            let data = SurvData::builder(&time, &status).build().expect("data");
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
                        std_err: true,
                        ..StandardiseOptions::default()
                    },
                )
                .expect("standardise");
                af_from_standardised(&s, &at, 0, 1, 0.95, IntervalScale::LogComplement).expect("af")
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
