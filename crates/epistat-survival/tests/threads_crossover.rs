//! `crossover::build_windows` and `crossover::exposure_in_windows` give
//! identical results in rayon pools of 1, 2, 3, 4 and 8 threads.
//!
//! The data are seeded, with 5,000 events so that the construction runs in
//! five tasks of 1,024 events, and 3,000 exposure intervals.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "the data are seeded days and counts of a few thousand, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::crossover::{
    ExposureIntervals, Measure, Observation, Referent, StratumUnit, build_windows,
    exposure_in_windows,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn crossover_does_not_depend_on_the_number_of_threads() {
    let mut rng = Stream::seeded(63);
    let n = 5_000_usize;
    let day = |rng: &mut Stream, a: i64, b: i64| a + rng.below((b - a) as usize) as i64;
    let event: Vec<i64> = (0..n).map(|_| day(&mut rng, 17_532, 18_993)).collect();
    let person: Vec<u32> = (0..n).map(|_| rng.below(500) as u32).collect();
    let start: Vec<i64> = event.iter().map(|&t| t - rng.below(120) as i64).collect();
    let end: Vec<i64> = event.iter().map(|&t| t + rng.below(120) as i64).collect();
    let m = 3_000_usize;
    let ex_person: Vec<u32> = (0..m).map(|_| rng.below(500) as u32).collect();
    let ex_start: Vec<i64> = (0..m).map(|_| day(&mut rng, 17_500, 19_000)).collect();
    let ex_end: Vec<i64> = ex_start.iter().map(|&s| s + rng.below(40) as i64).collect();
    let referent = Referent::TimeStratified {
        unit: StratumUnit::Month,
    };
    assert_same_on_pools(&POOLS, || {
        let w = build_windows(
            &event,
            &person,
            2,
            1,
            referent,
            Some(Observation {
                start: &start,
                end: &end,
            }),
        )
        .unwrap();
        let exposure = ExposureIntervals {
            person: &ex_person,
            start: &ex_start,
            end: &ex_end,
        };
        let days = exposure_in_windows(&w, &exposure, Measure::Days).unwrap();
        let mut out: Vec<f64> = vec![];
        out.extend(w.set.iter().map(|&v| f64::from(v)));
        out.extend(w.start.iter().map(|&v| v as f64));
        out.extend(w.end.iter().map(|&v| v as f64));
        out.extend(w.is_case.iter().map(|&v| f64::from(u8::from(v))));
        out.extend(days);
        out.push(w.n_referents_outside_observation as f64);
        out.push(w.sets_without_referent.len() as f64);
        out
    });
}
