//! The standardised risks, standard errors and covariances give identical bits
//! in rayon pools of 1, 2, 3, 4 and 8 threads, for one Cox model and for
//! cause-specific models, with strata and tied times, on more rows than one
//! block of the sums holds.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::absrisk::{CauseSpecificCox, CscOptions};
use epistat_survival::cox::{self, CoxOptions};
use epistat_survival::standardise::{
    Counterfactual, ExposureLevel, InfluenceForm, StandardiseOptions, Standardised,
    standardise_cause_specific, standardise_cox,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

fn flat(out: &Standardised) -> Vec<f64> {
    let mut all: Vec<f64> = out.risk.iter().map(|v| v.unwrap_or(-1.0)).collect();
    for table in [&out.se, &out.lower, &out.upper].into_iter().flatten() {
        all.extend(table.iter().map(|v| v.unwrap_or(-1.0)));
    }
    for matrix in out.covariance.iter().flatten() {
        all.extend(matrix.as_slice().iter().map(|v| v.unwrap_or(-1.0)));
    }
    all
}

#[test]
fn standardised_risks_are_the_same_on_every_pool() {
    let n = 1103;
    let mut rng = Stream::seeded(77);
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut stratum = Vec::new();
    let mut cols = Vec::new();
    for _ in 0..n {
        let x1 = rng.normal();
        let a = f64::from(u8::from(rng.below(2) == 0));
        let t1 = rng.exp1() / (0.10 * (0.5 * x1 - 0.4 * a).exp());
        let t2 = rng.exp1() / (0.06 * (-0.3 * a).exp());
        let c = 2.0 + 16.0 * rng.uniform();
        let t = t1.min(t2).min(c);
        stop.push((t * 4.0).ceil() / 4.0);
        status.push(if c < t1.min(t2) {
            0
        } else if t1 < t2 {
            1
        } else {
            2
        });
        stratum.push(rng.below(3) as u32);
        cols.push([a, x1]);
    }
    let x = Mat::from_fn(n, 2, |i, j| cols[i][j]);
    let one = [(0, 0.0)];
    let two = [(0, 1.0)];
    let times = [0.5, 2.0, 4.0, 5.5];
    let data = SurvData::builder(&stop, &status)
        .n_codes(2)
        .stratum(&stratum)
        .build()
        .unwrap();
    assert_same_on_pools(&POOLS, || {
        let options = CoxOptions {
            cause: Some(1),
            ..CoxOptions::default()
        };
        let fit = cox::fit(&data, x.as_ref(), &options).unwrap();
        let levels = [
            ExposureLevel::single(Counterfactual::Columns(&one)),
            ExposureLevel::single(Counterfactual::Columns(&two)),
        ];
        let out = standardise_cox(
            &data,
            x.as_ref(),
            &fit,
            &options,
            &levels,
            &times,
            &StandardiseOptions {
                se: true,
                ..StandardiseOptions::default()
            },
        )
        .unwrap();
        flat(&out)
    });
    for (product_limit, influence) in [
        (true, InfluenceForm::Exact),
        (true, InfluenceForm::Exponential),
        (false, InfluenceForm::Exact),
    ] {
        assert_same_on_pools(&POOLS, || {
            let designs = [x.as_ref(), x.as_ref()];
            let csc = CscOptions::default();
            let model = CauseSpecificCox::fit(&data, &designs, &csc).unwrap();
            let levels = [
                ExposureLevel {
                    designs: vec![Counterfactual::Columns(&one), Counterfactual::Columns(&one)],
                },
                ExposureLevel {
                    designs: vec![Counterfactual::Columns(&two), Counterfactual::Columns(&two)],
                },
            ];
            let out = standardise_cause_specific(
                &data,
                &designs,
                model.fits(),
                1,
                &csc,
                &levels,
                &times,
                &StandardiseOptions {
                    se: true,
                    product_limit,
                    influence,
                    ..StandardiseOptions::default()
                },
            )
            .unwrap();
            flat(&out)
        });
    }
}
