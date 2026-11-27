//! `rmst2` gives identical bits in rayon pools of 1, 2, 3, 4 and 8 threads, on
//! tied times with delayed entry, case weights, some of weight 0, and subjects
//! with several rows.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::rmst::{Rmst2, RmstContrast, RmstOptions, Tau, rmst2};
use epistat_testkit::{POOLS, absent, assert_same_on_pools};

fn flatten(r: &Rmst2, out: &mut Vec<f64>) {
    out.extend([r.tau, r.tau_largest, r.n_rows as f64, r.n_used as f64]);
    for g in [&r.group0, &r.group1] {
        out.extend([
            f64::from(g.group),
            g.n as f64,
            g.rmst,
            g.se,
            g.rmst_lower,
            g.rmst_upper,
            g.rmtl,
            g.rmtl_lower,
            g.rmtl_upper,
            g.last_time,
            f64::from(u8::from(g.ends_in_event)),
        ]);
    }
    for c in [&r.difference, &r.ratio, &r.loss_ratio] {
        let RmstContrast {
            estimate,
            se,
            lower,
            upper,
            p_value,
        } = *c;
        out.extend([
            absent(estimate),
            absent(se),
            absent(lower),
            absent(upper),
            absent(p_value),
        ]);
    }
}

#[test]
fn rmst2_does_not_depend_on_the_number_of_threads() {
    let n = 4001;
    let mut rng = Stream::seeded(41);
    let (mut start, mut stop, mut status, mut weight, mut arm, mut id) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for row in 0..n {
        let s = (rng.uniform() * 30.0).floor() / 10.0;
        start.push(s);
        stop.push(s + 0.1 + (rng.exp1() * 40.0).floor() / 10.0);
        status.push(i32::from(rng.below(10) < 4));
        weight.push(if rng.below(25) == 0 {
            0.0
        } else {
            0.5 + rng.uniform() * 2.0
        });
        arm.push(5 * rng.below(2) as u32 + 2);
        id.push((row / 2) as u32);
    }
    // A subject belongs to one arm.
    for row in (1..n).step_by(2) {
        arm[row] = arm[row - 1];
    }
    for with_start in [false, true] {
        let mut builder = SurvData::builder(&stop, &status)
            .weight(&weight)
            .stratum(&arm)
            .id(&id);
        if with_start {
            builder = builder.start(&start);
        }
        let data = builder.build().unwrap();
        for tau in [Tau::Fixed(2.0), Tau::Fixed(5.0)] {
            assert_same_on_pools(&POOLS, || {
                let mut out = Vec::new();
                match rmst2(&data, &RmstOptions::new(tau)) {
                    Ok(r) => flatten(&r, &mut out),
                    Err(_) => out.push(absent(None)),
                }
                out
            });
        }
        let r = rmst2(&data, &RmstOptions::new(Tau::Fixed(2.0))).unwrap();
        assert_eq!((r.group0.group, r.group1.group), (2, 7));
    }
}
