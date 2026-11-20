//! Curves read at chosen times give identical bits in rayon pools of 1, 2, 3, 4
//! and 8 threads, for a fit of seven strata built in the pool.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::survfit::{ReadOptions, SurvfitOptions, estimate};
use epistat_testkit::{POOLS, assert_same_on_pools};

const ABSENT: f64 = f64::from_bits(0x7ff8_0000_dead_beef);

#[test]
fn reading_does_not_depend_on_the_number_of_threads() {
    let mut rng = Stream::seeded(41);
    let n = 3001;
    let stop: Vec<f64> = (0..n)
        .map(|_| 0.1 + (rng.exp1() * 40.0).floor() / 10.0)
        .collect();
    let status: Vec<i32> = (0..n).map(|_| i32::from(rng.below(10) < 4)).collect();
    let weight: Vec<f64> = (0..n).map(|_| 0.5 + rng.uniform() * 2.0).collect();
    let stratum: Vec<u32> = (0..n).map(|_| 5 * rng.below(7) as u32 + 3).collect();
    let times: Vec<f64> = (-2..70).map(|i| f64::from(i) * 0.37).collect();
    for extend in [false, true] {
        for sum in [None, Some(false)] {
            assert_same_on_pools(&POOLS, || {
                let data = SurvData::builder(&stop, &status)
                    .weight(&weight)
                    .stratum(&stratum)
                    .build()
                    .unwrap();
                let fit = estimate(&data, &SurvfitOptions::default()).unwrap();
                let mut out = Vec::new();
                for c in fit.at(&times, &ReadOptions { extend, sum }).unwrap() {
                    for column in [
                        &c.time,
                        &c.n_risk,
                        &c.n_event,
                        &c.n_censor,
                        &c.surv,
                        &c.cumhaz,
                        &c.se_cumhaz,
                    ] {
                        out.extend_from_slice(column);
                    }
                    for column in [&c.se, &c.lower, &c.upper] {
                        out.extend(column.iter().map(|v| v.unwrap_or(ABSENT)));
                    }
                    for column in [&c.count_risk, &c.count_event, &c.count_censor] {
                        out.extend(column.iter().map(|&v| f64::from(v)));
                    }
                }
                out
            });
        }
    }
}
