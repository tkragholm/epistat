//! `cox::fit` with an offset, starting values, an iteration limit and its
//! tests gives identical bits in rayon pools of 1, 2, 3, 4 and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::{ChiSquareTest, CoxOptions, Ties, fit};
use epistat_testkit::{POOLS, assert_same_on_pools};
use faer::Mat;

/// A value standing for `None`, so that an absent number differs from any
/// present one.
const ABSENT: u64 = 0x7ff8_0000_dead_beef;

fn absent(value: Option<f64>) -> f64 {
    value.unwrap_or_else(|| f64::from_bits(ABSENT))
}

fn test(value: Option<ChiSquareTest>) -> [f64; 3] {
    value.map_or([f64::from_bits(ABSENT); 3], |t| {
        [t.statistic, t.df as f64, t.p_value]
    })
}

#[test]
fn cox_control_and_tests() {
    let n = 2203;
    let mut rng = Stream::seeded(41);
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut stratum = Vec::new();
    let mut cluster = Vec::new();
    let mut offset = Vec::new();
    let mut base = Vec::new();
    for _ in 0..n {
        let a = rng.normal();
        let t = (rng.exp1() * (-0.4 * a).exp() * 8.0).ceil() / 10.0 + 0.1;
        stop.push(t);
        start.push((rng.uniform() * t * 8.0).floor() / 10.0 * 0.9);
        status.push(i32::from(rng.below(3) != 0));
        weight.push(if rng.below(7) == 0 { 2.0 } else { 1.0 });
        stratum.push(rng.below(3) as u32);
        cluster.push(rng.below(31) as u32);
        offset.push(0.3 * rng.normal());
        base.push(a);
    }
    let x = Mat::from_fn(n, 3, |i, j| match j {
        2 => base[i] + base[i],
        1 => f64::from(u8::from(i % 3 == 0)),
        _ => base[i],
    });
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .cluster(&cluster)
        .build()
        .unwrap();
    let init = [0.4, -0.3, 0.2];
    for (iter_max, init) in [
        (20, None),
        (20, Some(&init[..])),
        (0, Some(&init[..])),
        (2, None),
    ] {
        for ties in [Ties::Breslow, Ties::Efron] {
            let options = CoxOptions {
                ties,
                offset: Some(&offset),
                init,
                iter_max,
                ..CoxOptions::default()
            };
            assert_same_on_pools(&POOLS, || {
                let model = fit(&data, x.as_ref(), &options).unwrap();
                assert!(model.tests.score.is_some() && model.tests.wald.is_some());
                let mut out: Vec<f64> = model.coefficients.iter().copied().map(absent).collect();
                out.extend(model.standard_errors.iter().copied().map(absent));
                out.extend([
                    absent(model.log_partial_likelihood),
                    absent(model.null_log_partial_likelihood),
                    f64::from(model.iterations),
                ]);
                let t = model.tests;
                for value in [
                    t.wald,
                    t.robust_wald,
                    t.score,
                    t.robust_score,
                    t.likelihood_ratio,
                ] {
                    out.extend(test(value));
                }
                out
            });
        }
    }
}
