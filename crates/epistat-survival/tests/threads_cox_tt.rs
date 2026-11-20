//! `cox::fit_tt` gives identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties, TimeTransform, TtColumn, TtOptions, fit_tt};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn fit_tt_is_the_same_on_every_pool() {
    let n = 300;
    let mut rng = Stream::seeded(77);
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut stratum = Vec::new();
    let mut cluster = Vec::new();
    let mut a = Vec::new();
    for _ in 0..n {
        let v = rng.normal();
        let t = (rng.exp1() * (-0.3 * v).exp() * 6.0).ceil() / 5.0 + 0.2;
        stop.push(t);
        start.push((rng.uniform() * t * 4.0).floor() / 5.0 * 0.8);
        status.push(i32::from(rng.below(3) != 0));
        weight.push(if rng.below(5) == 0 { 2.0 } else { 1.0 });
        stratum.push(rng.below(2) as u32);
        cluster.push(rng.below(17) as u32);
        a.push(v);
    }
    let x = Mat::from_fn(n, 3, |i, j| {
        if j == 2 {
            f64::from(u8::from(i % 3 == 0))
        } else {
            a[i]
        }
    });
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .cluster(&cluster)
        .build()
        .unwrap();
    let rank_sum = |_t: f64, v: &[f64], w: &[f64], out: &mut [f64]| {
        let total: f64 = w.iter().sum();
        for (o, x) in out.iter_mut().zip(v) {
            *o = x - total / 1e3;
        }
    };
    let transforms = [
        TtColumn {
            column: 0,
            transform: TimeTransform::LogProduct,
        },
        TtColumn {
            column: 1,
            transform: TimeTransform::OBrien,
        },
        TtColumn {
            column: 2,
            transform: TimeTransform::RiskSet(&rank_sum),
        },
    ];
    for ties in [Ties::Breslow, Ties::Efron] {
        let opts = TtOptions {
            cox: CoxOptions {
                ties,
                ..CoxOptions::default()
            },
            max_expanded_rows: None,
        };
        assert_same_on_pools(&POOLS, || {
            let r = fit_tt(&data, x.as_ref(), &transforms, &opts).unwrap();
            let f = r.fit;
            let mut out: Vec<f64> = f.coefficients.iter().map(|c| c.unwrap()).collect();
            out.extend(f.standard_errors.iter().map(|c| c.unwrap()));
            out.extend(f.robust_standard_errors.unwrap().iter().map(|c| c.unwrap()));
            out.extend([
                f.log_partial_likelihood.unwrap(),
                f64::from(f.iterations),
                r.n_expanded as f64,
            ]);
            out
        });
    }
}
