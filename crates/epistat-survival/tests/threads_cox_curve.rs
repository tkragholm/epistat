//! The Kalbfleisch-Prentice curve, the curves from a start time and the
//! curves of covariate paths give identical bits in rayon pools of 1, 2, 3, 4
//! and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::predict::{
    ConfType, CurveType, FittedModel, NewData, PathData, SurvfitOptions, survfit, survfit_paths,
};
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_testkit::assert_same_on_pools;
use faer::Mat;

const POOLS: [usize; 5] = [1, 2, 3, 4, 8];

fn options(value: Option<Vec<Option<f64>>>) -> Vec<f64> {
    value.map_or_else(Vec::new, |v| {
        v.into_iter().map(|x| x.unwrap_or(-1.0)).collect()
    })
}

#[test]
fn product_limit_start_time_and_paths() {
    let n = 1907;
    let mut rng = Stream::seeded(59);
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut stratum = Vec::new();
    let mut base = Vec::new();
    for _ in 0..n {
        let a = rng.normal();
        let t = (rng.exp1() * (-0.4 * a).exp() * 8.0).ceil() / 10.0 + 0.1;
        stop.push(t);
        start.push((rng.uniform() * t * 8.0).floor() / 10.0 * 0.9);
        status.push(i32::from(rng.below(3) != 0));
        weight.push(if rng.below(7) == 0 { 2.0 } else { 1.0 });
        stratum.push(rng.below(3) as u32);
        base.push(a);
    }
    let x = Mat::from_fn(
        n,
        2,
        |i, j| if j == 1 { base[i] * base[i] } else { base[i] },
    );
    let new_x = Mat::from_fn(4, 2, |i, j| 0.3 * (i as f64) - 0.2 * (j as f64));
    let new_stratum = [0u32, 1, 2, 1];
    let path_id = [1u32, 1, 2, 2, 2, 3];
    let path_start = [0.0, 0.5, 0.2, 0.9, 1.5, 0.0];
    let path_stop = [0.5, 3.0, 0.9, 1.5, 4.0, 2.0];
    let path_stratum = [0u32, 0, 1, 2, 2, 1];
    let path_x = Mat::from_fn(6, 2, |i, j| 0.25 * (i as f64) - 0.1 * (j as f64));
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
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
            let mut new = NewData::new(new_x.as_ref());
            new.stratum = Some(&new_stratum);
            let paths = PathData {
                x: path_x.as_ref(),
                id: &path_id,
                start: &path_start,
                stop: &path_stop,
                stratum: Some(&path_stratum),
                offset: None,
            };
            let mut out = Vec::new();
            for curve_type in [
                CurveType::ExpCumulativeHazard,
                CurveType::KalbfleischPrentice,
            ] {
                for start_time in [None, Some(1.0)] {
                    for conf_type in [ConfType::Log, ConfType::LogLog] {
                        let o = SurvfitOptions {
                            conf_type,
                            curve_type,
                            start_time,
                            censor_times: true,
                            ..SurvfitOptions::default()
                        };
                        let c = survfit(&fitted, &new, &o).unwrap();
                        for t in &c.strata {
                            out.extend(&t.time);
                            out.extend(&t.n_risk);
                            out.extend(&t.n_censor);
                        }
                        for curve in &c.curves {
                            out.extend(&curve.cumulative_hazard);
                            out.extend(&curve.survival);
                            out.extend(curve.std_err.clone().unwrap());
                            out.extend(options(curve.lower.clone()));
                            out.extend(options(curve.upper.clone()));
                        }
                        let q = survfit_paths(&fitted, &paths, &o).unwrap();
                        for curve in &q.curves {
                            out.extend(&curve.time);
                            out.extend(&curve.cumulative_hazard);
                            out.extend(&curve.survival);
                            out.extend(curve.std_err.clone().unwrap());
                            out.extend(options(curve.lower.clone()));
                            out.extend(options(curve.upper.clone()));
                        }
                    }
                }
            }
            out
        });
    }
}
