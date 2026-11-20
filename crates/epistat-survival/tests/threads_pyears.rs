//! The person-time table gives identical bits in rayon pools of 1, 2, 3, 4 and
//! 8 threads, with weights, delayed entry, a factor and two time dimensions.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the draws and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::pyears::{Dimension, PyearsOptions, pyears};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn pyears_does_not_depend_on_the_number_of_threads() {
    let n = 100_003;
    let mut rng = Stream::seeded(2026);
    let mut start = Vec::with_capacity(n);
    let mut stop = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    let mut weight = Vec::with_capacity(n);
    let mut sex = Vec::with_capacity(n);
    let mut age = Vec::with_capacity(n);
    let mut year = Vec::with_capacity(n);
    for _ in 0..n {
        let begin = rng.below(300) as f64 / 100.0;
        start.push(begin);
        stop.push(begin + (1 + rng.below(900)) as f64 / 100.0);
        status.push(i32::from(rng.below(3) == 0));
        weight.push(rng.below(800) as f64 / 100.0);
        sex.push(rng.below(2) as u32);
        age.push(rng.below(9000) as f64 / 100.0 - 5.0);
        year.push(1980.0 + rng.below(3500) as f64 / 100.0);
    }
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .build()
        .unwrap();
    let age_cuts: Vec<f64> = (0..=18).map(|i| f64::from(i) * 5.0).collect();
    let year_cuts: Vec<f64> = (0..=8).map(|i| 1985.0 + f64::from(i) * 5.0).collect();
    let dims = [
        Dimension::Factor {
            codes: &sex,
            levels: 2,
        },
        Dimension::Time {
            origin: &age,
            cuts: &age_cuts,
        },
        Dimension::Time {
            origin: &year,
            cuts: &year_cuts,
        },
    ];
    let opts = PyearsOptions {
        scale: 365.25,
        ..PyearsOptions::default()
    };
    assert_same_on_pools(&POOLS, || {
        let table = pyears(&data, &dims, &opts).unwrap();
        let mut out = table.person_time;
        out.extend(table.n.iter().map(|&c| c as f64));
        out.extend(table.events);
        out.push(table.offtable);
        out
    });
}
