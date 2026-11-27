//! `af_from_standardised` of `standardise_cox` output gives identical bits in
//! rayon pools of 1, 2, 3, 4 and 8 threads.

use epistat_glm::attributable::IntervalScale;
use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_survival::standardise::attributable::af_from_standardised;
use epistat_survival::standardise::{
    Counterfactual, ExposureLevel, StandardiseOptions, standardise_cox,
};
use epistat_testkit::{POOLS, absent_all, assert_same_on_pools};

#[test]
fn the_attributable_fraction_function_does_not_depend_on_the_number_of_threads() {
    let mut rng = Stream::seeded(66);
    let n = 3_000_usize;
    let z: Vec<f64> = (0..n)
        .map(|_| {
            let (a, b) = (rng.uniform(), rng.uniform());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        })
        .collect();
    let e: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(rng.uniform() < 1.0 / (1.0 + (-0.4 * z[i]).exp()))))
        .collect();
    let time: Vec<f64> = (0..n)
        .map(|i| {
            (rng.exp1() / (0.1 * (0.6 * e[i] + 0.4 * z[i]).exp())).min(30.0 * rng.uniform() + 3.0)
        })
        .collect();
    let status: Vec<i32> = (0..n).map(|_| i32::from(rng.uniform() < 0.7)).collect();
    let x = Mat::from_fn(n, 3, |i, j| [e[i], z[i], e[i] * z[i]][j]);
    let x0 = Mat::from_fn(n, 3, |i, j| if j == 1 { x[(i, j)] } else { 0.0 });
    let data = SurvData::builder(&time, &status).build().unwrap();
    let options = CoxOptions {
        ties: Ties::Breslow,
        ..CoxOptions::default()
    };
    let model = fit(&data, x.as_ref(), &options).unwrap();
    let times = [1.0, 2.0, 4.0, 8.0];
    assert_same_on_pools(&POOLS, || {
        let s = standardise_cox(
            &data,
            x.as_ref(),
            &model,
            &options,
            &[
                ExposureLevel::single(Counterfactual::Design(x0.as_ref())),
                ExposureLevel::single(Counterfactual::Columns(&[])),
            ],
            &times,
            &StandardiseOptions {
                se: true,
                ..StandardiseOptions::default()
            },
        )
        .unwrap();
        let c = af_from_standardised(
            &s,
            &times,
            0,
            1,
            epistat_survival::Level::default(),
            IntervalScale::LogComplement,
        )
        .unwrap();
        let mut out = vec![];
        for v in [&c.af, &c.se, &c.lower, &c.upper, &c.ratio] {
            out.extend(absent_all(v));
        }
        out
    });
}
