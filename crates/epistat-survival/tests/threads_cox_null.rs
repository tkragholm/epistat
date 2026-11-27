//! The fit, the baseline hazard, the predictions and the survival curves of a
//! Cox model without covariates give identical bits in rayon pools of 1, 2, 3,
//! 4 and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::predict::{
    BaselineOptions, ConfType, FittedModel, NewData, PredictOptions, PredictType, SurvfitOptions,
    basehaz, predict, survfit,
};
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn model_without_covariates() {
    let n = 1907;
    let mut rng = Stream::seeded(59);
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut stratum = Vec::new();
    let mut cluster = Vec::new();
    let mut offset = Vec::new();
    for _ in 0..n {
        let t = (rng.exp1() * 8.0).ceil() / 10.0 + 0.1;
        stop.push(t);
        start.push((rng.uniform() * t * 8.0).floor() / 10.0 * 0.9);
        status.push(i32::from(rng.below(3) != 0));
        weight.push(if rng.below(7) == 0 { 2.0 } else { 1.0 });
        stratum.push(rng.below(3) as u32);
        cluster.push(rng.below(40) as u32);
        offset.push(0.3 * rng.normal());
    }
    let x = Mat::<f64>::zeros(n, 0);
    let new_x = Mat::<f64>::zeros(3, 0);
    let new_stratum = [0u32, 1, 2];
    let new_offset = [0.1, 0.0, -0.1];
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .cluster(&cluster)
        .build()
        .unwrap();
    for ties in [Ties::Breslow, Ties::Efron] {
        let opts = CoxOptions {
            ties,
            offset: Some(&offset),
            ..CoxOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let model = fit(&data, x.as_ref(), &opts).unwrap();
            let mut out = vec![
                model.log_partial_likelihood.unwrap(),
                model.null_log_partial_likelihood.unwrap(),
                model.n_events,
            ];
            let fitted = FittedModel {
                data: &data,
                x: x.as_ref(),
                fit: &model,
                options: &opts,
            };
            let b = basehaz(&fitted, &BaselineOptions::default()).unwrap();
            for s in &b.strata {
                out.extend(&s.time);
                out.extend(&s.cumulative_hazard);
                out.extend(&s.cumulative_variance);
            }
            let mut new = NewData::new(new_x.as_ref());
            new.stratum = Some(&new_stratum);
            new.offset = Some(&new_offset);
            let popts = PredictOptions {
                se_fit: true,
                ..PredictOptions::default()
            };
            for kind in [
                PredictType::LinearPredictor,
                PredictType::Risk,
                PredictType::Expected,
                PredictType::Survival,
            ] {
                let p = predict(&fitted, kind, None, &popts).unwrap();
                out.extend(&p.fit);
                out.extend(p.se.unwrap());
            }
            for conf_type in [ConfType::Log, ConfType::LogLog] {
                let c = survfit(
                    &fitted,
                    &new,
                    &SurvfitOptions {
                        conf_type,
                        ..SurvfitOptions::default()
                    },
                )
                .unwrap();
                for curve in &c.curves {
                    out.extend(&curve.cumulative_hazard);
                    out.extend(&curve.survival);
                    out.extend(curve.se.clone().unwrap());
                }
            }
            out
        });
    }
}
