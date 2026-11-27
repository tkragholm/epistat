//! The baseline hazard, the predictions and the survival curves of a Cox fit
//! give identical bits in rayon pools of 1, 2, 3, 4 and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::predict::{
    BaselineOptions, ConfType, CoxPredictOptions, CoxPredictType, CoxSurvfitOptions, FittedModel,
    NewData, Reference, basehaz, predict, predict_terms, survfit,
};
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_testkit::{POOLS, assert_same_on_pools};

fn options(value: Option<Vec<Option<f64>>>) -> Vec<f64> {
    value.map_or_else(Vec::new, |v| {
        v.into_iter().map(|x| x.unwrap_or(-1.0)).collect()
    })
}

#[test]
fn predictions_and_curves() {
    let n = 1907;
    let mut rng = Stream::seeded(53);
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut stratum = Vec::new();
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
        offset.push(0.3 * rng.normal());
        base.push(a);
    }
    let x = Mat::from_fn(n, 3, |i, j| match j {
        2 => base[i] * base[i],
        1 => f64::from(u8::from(i % 3 == 0)),
        _ => base[i],
    });
    let new_x = Mat::from_fn(4, 3, |i, j| 0.3 * (i as f64) - 0.2 * (j as f64));
    let new_stratum = [0u32, 1, 2, 1];
    let new_start = [0.1, 0.2, 0.0, 0.3];
    let new_stop = [1.0, 2.0, 3.0, 4.0];
    let new_offset = [0.1, 0.0, -0.1, 0.2];
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .offset(&offset)
        .build()
        .unwrap();
    for ties in [Ties::Breslow, Ties::Efron] {
        let opts = CoxOptions {
            ties,
            ..CoxOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let model = fit(&data, x.as_ref(), &opts).unwrap();
            let fitted = FittedModel {
                data: &data,
                x: x.as_ref(),
                fit: &model,
                options: &opts,
            };
            let mut out = Vec::new();
            let b = basehaz(&fitted, &BaselineOptions::default()).unwrap();
            for s in &b.strata {
                out.extend(&s.time);
                out.extend(&s.cumulative_hazard);
                out.extend(&s.cumulative_variance);
                out.extend(&s.cumulative_xbar);
            }
            let mut new = NewData::new(new_x.as_ref());
            new.stratum = Some(&new_stratum);
            new.start = Some(&new_start);
            new.stop = Some(&new_stop);
            new.offset = Some(&new_offset);
            let popts = CoxPredictOptions {
                se_fit: true,
                ..CoxPredictOptions::default()
            };
            for kind in [
                CoxPredictType::LinearPredictor,
                CoxPredictType::Risk,
                CoxPredictType::Expected,
                CoxPredictType::Survival,
            ] {
                for newdata in [None, Some(&new)] {
                    for reference in [Reference::Strata, Reference::Sample, Reference::Zero] {
                        let o = CoxPredictOptions {
                            reference: Some(reference),
                            ..popts
                        };
                        let p = predict(&fitted, kind, newdata, &o).unwrap();
                        out.extend(&p.fit);
                        out.extend(p.se.unwrap());
                    }
                }
            }
            let t = predict_terms(&fitted, &[&[0, 1], &[2]], Some(&new), &popts).unwrap();
            out.extend(&t.fit);
            out.extend(t.se.unwrap());
            out.push(t.constant);
            for conf_type in [ConfType::Log, ConfType::LogLog, ConfType::Arcsin] {
                let c = survfit(
                    &fitted,
                    &new,
                    &CoxSurvfitOptions {
                        conf_type,
                        ..CoxSurvfitOptions::default()
                    },
                )
                .unwrap();
                for curve in &c.curves {
                    out.extend(&curve.cumulative_hazard);
                    out.extend(&curve.survival);
                    out.extend(curve.se.clone().unwrap());
                    out.extend(options(curve.lower.clone()));
                    out.extend(options(curve.upper.clone()));
                }
            }
            out
        });
    }
}
