//! The regression of the restricted mean on covariates gives identical bits in
//! rayon pools of 1, 2, 3, 4 and 8 threads, for the three effects. The sample is
//! larger than the row block of the Gram matrices, with tied times on a grid of
//! tenths and two covariates.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the grid values are far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::rmst::{RmstOptions, Tau, rmst2_adjusted};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn adjusted_rmst_does_not_depend_on_the_number_of_threads() {
    let n = 9001;
    let mut rng = Stream::seeded(71);
    let (mut stop, mut status, mut arm) = (Vec::new(), Vec::new(), Vec::new());
    let x = {
        let mut cells = Vec::new();
        for _ in 0..n {
            let a = rng.below(2) as u32;
            let x1 = rng.uniform();
            let x2 = f64::from(rng.below(2) as u32);
            arm.push(a);
            stop.push(0.1 + (rng.exp1() * (20.0 - 8.0 * f64::from(a) - 5.0 * x1)).floor() / 10.0);
            status.push(i32::from(rng.below(10) < 6));
            cells.push((x1, x2));
        }
        Mat::from_fn(n, 2, |i, j| if j == 0 { cells[i].0 } else { cells[i].1 })
    };
    let data = SurvData::builder(&stop, &status)
        .stratum(&arm)
        .build()
        .unwrap();
    let opts = RmstOptions::new(Tau::Fixed(3.0));
    assert_same_on_pools(&POOLS, || {
        let r = rmst2_adjusted(&data, Some(x.as_ref()), &opts).unwrap();
        let mut out = Vec::new();
        for f in [&r.difference, &r.ratio, &r.loss_ratio] {
            out.extend_from_slice(&f.coefficients);
            out.extend_from_slice(&f.se);
            out.extend(f.covariance.as_slice().iter().map(|v| v.unwrap()));
            out.extend_from_slice(&f.p_value);
            out.extend_from_slice(&f.lower);
            out.extend_from_slice(&f.upper);
        }
        out
    });
}
