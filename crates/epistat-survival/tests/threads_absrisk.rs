//! The absolute risks, their standard errors and their limits give identical
//! bits in rayon pools of 1, 2, 3, 4 and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::absrisk::{
    CauseSpecificCox, CscOptions, InfluenceForm, NewRows, RiskOptions, SurvType,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

fn flat(values: &[Option<f64>]) -> impl Iterator<Item = f64> + '_ {
    values.iter().map(|v| v.unwrap_or(-1.0))
}

#[test]
fn risk_standard_error_and_limits() {
    let n = 611;
    let mut rng = Stream::seeded(71);
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut stratum = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for _ in 0..n {
        let u = rng.normal();
        let v = f64::from(u8::from(rng.below(2) == 0));
        let t1 = rng.exp1() / (0.10 * (0.5 * u).exp());
        let t2 = rng.exp1() / (0.06 * (-0.6 * v).exp());
        let c = 2.0 + 16.0 * rng.uniform();
        let t = t1.min(t2).min(c);
        // Rounded times, so that events are tied.
        stop.push((t * 4.0).ceil() / 4.0);
        status.push(if c < t1.min(t2) {
            0
        } else if t1 < t2 {
            1
        } else {
            2
        });
        stratum.push(rng.below(3) as u32);
        a.push(u);
        b.push(v);
    }
    let x1 = Mat::from_fn(n, 1, |i, _| a[i]);
    let x2 = Mat::from_fn(n, 2, |i, j| if j == 0 { a[i] } else { b[i] });
    let rows = 37;
    let new_x1 = Mat::from_fn(rows, 1, |i, _| 0.1 * i as f64 - 1.5);
    let new_x2 = Mat::from_fn(rows, 2, |i, j| 0.05 * (i * (j + 1)) as f64 - 0.8);
    let new_stratum: Vec<u32> = (0..rows).map(|i| (i % 3) as u32).collect();
    let times = [0.5, 2.0, 5.5, 9.0, 14.0, 30.0];
    let data = SurvData::builder(&stop, &status)
        .n_codes(2)
        .stratum(&stratum)
        .build()
        .unwrap();
    for surv_type in [SurvType::Hazard, SurvType::Survival { cause: 1 }] {
        for (product_limit, influence) in [
            (true, InfluenceForm::Exact),
            (true, InfluenceForm::Exponential),
            (false, InfluenceForm::Exact),
        ] {
            assert_same_on_pools(&POOLS, || {
                let designs = [x1.as_ref(), x2.as_ref()];
                let model = CauseSpecificCox::fit(
                    &data,
                    &designs,
                    &CscOptions {
                        surv_type,
                        ..CscOptions::default()
                    },
                )
                .unwrap();
                let new_designs = [new_x1.as_ref(), new_x2.as_ref()];
                let new = NewRows {
                    designs: &new_designs,
                    stratum: Some(&new_stratum),
                    offset: None,
                };
                let out = model
                    .predict(
                        &new,
                        &times,
                        1,
                        &RiskOptions {
                            product_limit,
                            influence,
                            std_err: true,
                            ..RiskOptions::default()
                        },
                    )
                    .unwrap();
                let mut all: Vec<f64> = flat(&out.risk).collect();
                all.extend(flat(out.std_err.as_ref().unwrap()));
                all.extend(flat(out.lower.as_ref().unwrap()));
                all.extend(flat(out.upper.as_ref().unwrap()));
                all
            });
        }
    }
}
