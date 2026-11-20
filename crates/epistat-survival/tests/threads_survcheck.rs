//! The structural check gives identical results in rayon pools of 1, 2, 3, 4
//! and 8 threads.
//!
//! The rows are drawn at random from a small grid of times, so that many rows
//! of one subject share a stop or a start and only the row number breaks the
//! tie in the sort. The number of rows is odd, and no pool divides it evenly.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::survcheck::{Flagged, survcheck};
use epistat_testkit::{POOLS, assert_same_on_pools};

fn push_flag(out: &mut Vec<f64>, flag: &Flagged) {
    out.push(flag.count() as f64);
    out.extend(flag.rows.iter().map(|&r| f64::from(r)));
    out.extend(flag.ids.iter().map(|&i| f64::from(i)));
}

#[test]
fn survcheck_does_not_depend_on_the_number_of_threads() {
    let n = 100_003;
    let k = 3_u32;
    let mut rng = Stream::seeded(2026);
    let mut start = Vec::with_capacity(n);
    let mut stop = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    let mut id = Vec::with_capacity(n);
    let mut istate = Vec::with_capacity(n);
    for _ in 0..n {
        let begin = rng.below(30) as f64;
        start.push(begin);
        stop.push(begin + 1.0 + rng.below(4) as f64);
        status.push(rng.below(k as usize + 1) as i32);
        id.push(1000 * rng.below(20_011) as u32 + 7);
        istate.push(rng.below(k as usize + 1) as u32);
    }
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(k)
        .id(&id)
        .istate(&istate)
        .build()
        .unwrap();
    assert_same_on_pools(&POOLS, || {
        let check = survcheck(&data).unwrap();
        let mut out = Vec::new();
        for flag in [&check.overlap, &check.gap, &check.jump, &check.teleport] {
            push_flag(&mut out, flag);
        }
        for t in &check.transitions {
            out.extend([f64::from(t.from), f64::from(t.to), t.count as f64]);
        }
        out.extend(check.istate.iter().map(|&s| f64::from(s)));
        assert!(check.overlap.count() > 1000 && check.jump.count() > 1000);
        out
    });
}
