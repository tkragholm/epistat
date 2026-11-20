//! `predict_expected_martingale` gives identical bits in rayon pools of 1, 2, 3,
//! 4 and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::predict::{FittedModel, predict_expected_martingale};
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn martingale_form_is_the_same_on_every_pool() {
    let n = 1800;
    let mut rng = Stream::seeded(59);
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut stratum = Vec::new();
    let mut a = Vec::new();
    for _ in 0..n {
        let u = rng.normal();
        a.push(u);
        stop.push((rng.exp1() * (-0.3 * u).exp() * 8.0).ceil() / 10.0 + 0.1);
        status.push(i32::from(rng.below(4) != 0));
        stratum.push(rng.below(3) as u32);
    }
    let x = Mat::from_fn(n, 1, |i, _| a[i]);
    let data = SurvData::builder(&stop, &status)
        .stratum(&stratum)
        .build()
        .unwrap();
    let opts = CoxOptions {
        ties: Ties::Efron,
        ..CoxOptions::default()
    };
    let model = fit(&data, x.as_ref(), &opts).unwrap();
    let fitted = FittedModel {
        data: &data,
        x: x.as_ref(),
        fit: &model,
        options: &opts,
    };
    assert_same_on_pools(&POOLS, || predict_expected_martingale(&fitted).unwrap());
}
