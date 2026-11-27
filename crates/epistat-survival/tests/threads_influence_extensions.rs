//! The standardised risks and standard errors with case weights, entry times,
//! an offset in every model and `surv.type = "survival"` give identical bits in rayon pools of
//! 1, 2, 3, 4 and 8 threads, on more rows than one block of the sums holds. The
//! absolute risk's standard error with weights, entry times and an offset does as well.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::absrisk::{CauseSpecificCox, CscOptions, NewRows, RiskOptions, SurvType};
use epistat_survival::cox::{self, CoxOptions};
use epistat_survival::standardise::{
    Counterfactual, Level, StandardiseOptions, Standardised, standardise_cause_specific,
    standardise_cox,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

fn flat(out: &Standardised) -> Vec<f64> {
    let mut all: Vec<f64> = out.risk.iter().map(|v| v.unwrap_or(-1.0)).collect();
    for table in [&out.se, &out.lower, &out.upper, &out.covariance]
        .into_iter()
        .flatten()
    {
        all.extend(table.iter().map(|v| v.unwrap_or(-1.0)));
    }
    all
}

#[test]
fn weights_entry_offset_and_survival_type_are_the_same_on_every_pool() {
    let n = 1103;
    let mut rng = Stream::seeded(78);
    let (mut stop, mut start, mut status) = (Vec::new(), Vec::new(), Vec::new());
    let (mut stratum, mut weight, mut offset, mut cols) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let x1 = rng.normal();
        let a = f64::from(u8::from(rng.below(2) == 0));
        let t1 = rng.exp1() / (0.10 * (0.5 * x1 - 0.4 * a).exp());
        let t2 = rng.exp1() / (0.06 * (-0.3 * a).exp());
        let c = 2.0 + 16.0 * rng.uniform();
        let t = ((t1.min(t2).min(c)) * 4.0).ceil() / 4.0;
        stop.push(t);
        start.push(if rng.uniform() < 0.5 {
            0.3 * t * rng.uniform()
        } else {
            0.0
        });
        status.push(if c < t1.min(t2) {
            0
        } else if t1 < t2 {
            1
        } else {
            2
        });
        stratum.push(rng.below(3) as u32);
        weight.push(0.5 + 2.5 * rng.uniform());
        offset.push(0.3 * rng.normal());
        cols.push([a, x1]);
    }
    let x = Mat::from_fn(n, 2, |i, j| cols[i][j]);
    let one = [(0, 0.0)];
    let two = [(0, 1.0)];
    let times = [0.5, 2.0, 4.0, 5.5];
    let data = SurvData::builder(&stop, &status)
        .n_codes(2)
        .stratum(&stratum)
        .weight(&weight)
        .start(&start)
        .offset(&offset)
        .build()
        .unwrap();
    let levels = [
        Level::single(Counterfactual::Columns(&one)),
        Level::single(Counterfactual::Columns(&two)),
    ];
    let std_err = StandardiseOptions {
        se: true,
        ..StandardiseOptions::default()
    };
    assert_same_on_pools(&POOLS, || {
        let options = CoxOptions {
            cause: Some(1),
            ..CoxOptions::default()
        };
        let fit = cox::fit(&data, x.as_ref(), &options).unwrap();
        let out =
            standardise_cox(&data, x.as_ref(), &fit, &options, &levels, &times, &std_err).unwrap();
        flat(&out)
    });
    let pair = |set: &'static [(usize, f64); 1]| Level {
        designs: vec![Counterfactual::Columns(set), Counterfactual::Columns(set)],
    };
    let pairs = [pair(&[(0, 0.0)]), pair(&[(0, 1.0)])];
    let designs = [x.as_ref(), x.as_ref()];
    assert_same_on_pools(&POOLS, || {
        let csc = CscOptions {
            ..CscOptions::default()
        };
        let model = CauseSpecificCox::fit(&data, &designs, &csc).unwrap();
        let out = standardise_cause_specific(
            &data,
            &designs,
            model.fits(),
            1,
            &csc,
            &pairs,
            &times,
            &std_err,
        )
        .unwrap();
        flat(&out)
    });
    assert_same_on_pools(&POOLS, || {
        let csc = CscOptions {
            surv_type: SurvType::Survival { cause: 1 },
            ..CscOptions::default()
        };
        let model = CauseSpecificCox::fit(&data, &designs, &csc).unwrap();
        let out = standardise_cause_specific(
            &data,
            &designs,
            model.fits(),
            1,
            &csc,
            &pairs,
            &times,
            &std_err,
        )
        .unwrap();
        flat(&out)
    });
    assert_same_on_pools(&POOLS, || {
        let csc = CscOptions {
            ..CscOptions::default()
        };
        let model = CauseSpecificCox::fit(&data, &designs, &csc).unwrap();
        let new1 = Mat::from_fn(3, 2, |i, j| [[1.0, 0.2], [0.0, -0.5], [1.0, 1.1]][i][j]);
        let rows = [0u32, 1, 2];
        let new_offset = [0.3, -0.2, 0.0];
        let new_designs = [new1.as_ref(), new1.as_ref()];
        let out = model
            .predict(
                &NewRows {
                    designs: &new_designs,
                    stratum: Some(&rows),
                    offset: Some(&new_offset),
                },
                &times,
                1,
                &RiskOptions {
                    se: true,
                    ..RiskOptions::default()
                },
            )
            .unwrap();
        let mut all: Vec<f64> = out.risk.iter().map(|v| v.unwrap_or(-1.0)).collect();
        all.extend(out.se.unwrap().iter().map(|v| v.unwrap_or(-1.0)));
        all
    });
}
