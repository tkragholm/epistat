//! `survdiff::test` and `survdiff::one_sample` give identical bits in rayon
//! pools of 1, 2, 3, 4 and 8 threads.
//!
//! The data are seeded, with 2,500 strata so that they fall into three blocks
//! of 1,024 strata, each summed by its own task, and times on a grid of tenths
//! so that times tie.

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::survdiff::{SurvdiffOptions, one_sample, test};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn survdiff_does_not_depend_on_the_number_of_threads() {
    let mut rng = Stream::seeded(53);
    let n = 30_000_usize;
    let (mut time, mut status, mut group, mut stratum) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let g = u32::try_from(rng.below(4)).unwrap();
        time.push(0.1 + (rng.exp1() * (8.0 + f64::from(g))).floor() / 10.0);
        status.push(i32::from(rng.below(3) != 0));
        group.push(g * 3 + 1);
        stratum.push(u32::try_from(rng.below(2500)).unwrap());
    }
    let data = SurvData::builder(&time, &status)
        .stratum(&stratum)
        .build()
        .unwrap();
    let offset: Vec<f64> = time.iter().map(|t| (-0.05 * t).exp()).collect();
    let plain = SurvData::builder(&time, &status).build().unwrap();
    for rho in [0.0, 1.0, 0.37] {
        let opts = SurvdiffOptions {
            rho,
            ..SurvdiffOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let r = test(&data, &group, &opts).unwrap();
            let mut out = vec![];
            out.extend(&r.observed);
            out.extend(&r.expected);
            out.extend(&r.variance);
            out.extend(&r.observed_by_stratum);
            out.extend(&r.expected_by_stratum);
            out.extend(r.statistic);
            out.extend(r.p_value);
            let s = one_sample(&plain, &offset, &opts).unwrap();
            out.extend([s.observed, s.expected, s.variance]);
            out.extend(s.statistic);
            out
        });
    }
}
