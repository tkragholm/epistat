//! `metrics::royston` gives identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads.
//!
//! 6001 rows (an odd number above rayon's cut-off for a parallel sort) with
//! times and a predictor that tie, delayed entry, weights of 0 on some rows and
//! a status with two codes. The pairwise sum of `C.GH` runs over about 18
//! million pairs and is split into rows, so pools of different sizes group the
//! rows differently.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the grid values are far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::cox::fit;
use epistat_survival::metrics::royston::{
    Adjustment, FitOptions, RoystonOptions, estimate, from_fit,
};
use epistat_testkit::{POOLS, absent, assert_same_on_pools};

#[test]
fn royston_does_not_depend_on_the_number_of_threads() {
    let n = 6001usize;
    let mut rng = Stream::seeded(4242);
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut x = Mat::<f64>::zeros(n, 2);
    for i in 0..n {
        let t = 1.0 + rng.below(400) as f64 / 10.0;
        let entry = if rng.below(4) == 0 {
            (rng.below(100) as f64 / 10.0).min(t - 0.1)
        } else {
            0.0
        };
        stop.push(t);
        start.push(entry);
        status.push(rng.below(3) as i32);
        weight.push(if rng.below(50) == 0 { 0.0 } else { 1.0 });
        x[(i, 0)] = rng.below(5) as f64 / 2.0;
        x[(i, 1)] = rng.below(30) as f64 / 10.0;
    }
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .n_codes(2)
        .build()
        .unwrap();
    let cox = CoxOptions {
        cause: Some(1),
        ..CoxOptions::default()
    };
    let model = fit(&data, x.as_ref(), &cox).unwrap();
    let eta: Vec<f64> = (0..n).map(|i| 0.4 * x[(i, 0)] + 0.3 * x[(i, 1)]).collect();
    let pack = |r: &epistat_survival::metrics::royston::Royston| {
        [r.d, r.se_d, r.r_d, r.r_ko, r.r_n, r.c_gh, r.beta]
            .map(absent)
            .to_vec()
    };
    assert_same_on_pools(&POOLS, || {
        let mut out = Vec::new();
        for ties in [true, false] {
            let r = estimate(
                &data,
                &eta,
                &RoystonOptions {
                    cause: Some(1),
                    ties,
                    adjust: Some(Adjustment {
                        n_events: 900.0,
                        n_coef: 2,
                    }),
                    ..RoystonOptions::default()
                },
            )
            .unwrap();
            out.extend(pack(&r));
        }
        let r = from_fit(&data, x.as_ref(), &model, &cox, &FitOptions::default()).unwrap();
        out.extend(pack(&r));
        out
    });
}
