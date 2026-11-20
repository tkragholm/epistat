//! `metrics` give identical bits in rayon pools of 1, 2, 3, 4 and 8 threads.
//!
//! The inputs are drawn from a seeded stream on a grid of tenths so that times
//! tie, with a competing event, rows of weight 0, strata, case weights and
//! predictions that tie. The number of rows is odd and above rayon's cut-off
//! for a parallel sort, and there are more times than threads, so that the
//! split of the times over the pool differs between pool sizes.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::metrics::brier::{self, BrierOptions};
use epistat_survival::metrics::calibration::{self, CalibrationOptions, Groups};
use epistat_survival::metrics::rttright::{self, RttrightOptions};
use epistat_survival::metrics::score::{
    ControlSet, HorizonRule, ScoreOptions, auc, brier as score_brier, ipa, null_brier,
};
use epistat_testkit::{POOLS, absent, assert_same_on_pools};
use faer::Mat;

struct Sample {
    time: Vec<f64>,
    status: Vec<i32>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    risk: Mat<f64>,
}

fn sample(n: usize, n_times: usize, seed: u64) -> Sample {
    let mut rng = Stream::seeded(seed);
    let mut s = Sample {
        time: Vec::new(),
        status: Vec::new(),
        weight: Vec::new(),
        stratum: Vec::new(),
        risk: Mat::zeros(n, n_times),
    };
    for i in 0..n {
        s.time.push(0.1 + (rng.exp1() * 50.0).floor() / 10.0);
        s.status.push(match rng.below(5) {
            0 | 1 => 0,
            2 | 3 => 1,
            _ => 2,
        });
        s.weight.push(if rng.below(20) == 0 { 0.0 } else { 1.0 });
        s.stratum.push(rng.below(4) as u32 * 7);
        for j in 0..n_times {
            s.risk[(i, j)] = (rng.below(40) as f64 + 5.0 * j as f64) / 100.0;
        }
    }
    s
}

#[test]
fn score_metrics_and_calibration() {
    let n_times = 11;
    let s = sample(4001, n_times, 17);
    let data = SurvData::builder(&s.time, &s.status)
        .n_codes(2)
        .weight(&s.weight)
        .build()
        .unwrap();
    let times: Vec<f64> = (1..=n_times).map(|j| 0.4 * j as f64).collect();
    let column: Vec<f64> = (0..s.time.len()).map(|i| s.risk[(i, 3)]).collect();
    for controls in [ControlSet::EventFree, ControlSet::EventFreeAndCompeting] {
        for horizon in [HorizonRule::Inclusive, HorizonRule::Exclusive] {
            let opts = ScoreOptions {
                keep_influence: true,
                controls,
                horizon,
                ..ScoreOptions::default()
            };
            assert_same_on_pools(&POOLS, || {
                let b = score_brier(&data, s.risk.as_ref(), &times, &opts).unwrap();
                let z = null_brier(&data, &times, &opts).unwrap();
                let p = ipa(&b, &z).unwrap();
                let a = auc(&data, s.risk.as_ref(), &times, &opts).unwrap();
                let mut out = Vec::new();
                out.extend(&b.brier);
                out.extend(b.std_err.iter().map(|&v| absent(v)));
                out.extend(&z.brier);
                out.extend(p.ipa.iter().map(|&v| absent(v)));
                out.extend(p.std_err.iter().map(|&v| absent(v)));
                out.extend(a.auc.iter().map(|&v| absent(v)));
                out.extend(a.std_err.iter().map(|&v| absent(v)));
                out.extend(a.n_cases.iter().map(|&v| v as f64));
                let inf = b.influence.unwrap();
                out.extend(
                    (0..inf.nrows())
                        .flat_map(|i| (0..inf.ncols()).map(move |j| (i, j)))
                        .map(|(i, j)| inf[(i, j)]),
                );
                let inf = a.influence.unwrap();
                out.extend(
                    (0..inf.nrows())
                        .flat_map(|i| (0..inf.ncols()).map(move |j| (i, j)))
                        .map(|(i, j)| inf[(i, j)]),
                );
                out
            });
        }
    }
    assert_same_on_pools(&POOLS, || {
        let c = calibration::estimate(
            &data,
            &column,
            2.0,
            &CalibrationOptions::new(Groups::Quantiles(10)),
        )
        .unwrap();
        let mut out = c.breaks.clone();
        for g in &c.groups {
            out.extend([
                g.lower,
                g.upper,
                g.n as f64,
                g.predicted,
                absent(g.observed),
            ]);
        }
        out
    });
}

#[test]
fn rttright_and_brier() {
    let n_times = 11;
    let s = sample(4001, n_times, 23);
    let single: Vec<i32> = s.status.iter().map(|&v| i32::from(v == 1)).collect();
    let probability = Mat::from_fn(s.time.len(), n_times, |i, j| s.risk[(i, j)].clamp(0.0, 1.0));
    let times: Vec<f64> = (1..=n_times).map(|j| 0.4 * j as f64).collect();
    let competing = SurvData::builder(&s.time, &s.status)
        .n_codes(2)
        .weight(&s.weight)
        .stratum(&s.stratum)
        .build()
        .unwrap();
    let binary = SurvData::builder(&s.time, &single)
        .weight(&s.weight)
        .build()
        .unwrap();
    for ties in [true, false] {
        assert_same_on_pools(&POOLS, || {
            let w =
                rttright::estimate(&competing, Some(&times), &RttrightOptions::default()).unwrap();
            let end = rttright::estimate(&competing, None, &RttrightOptions::default()).unwrap();
            let b = brier::estimate(
                &binary,
                probability.as_ref(),
                &times,
                &BrierOptions {
                    ties,
                    ..BrierOptions::default()
                },
            )
            .unwrap();
            let mut out = Vec::new();
            out.extend(
                (0..w.weights.nrows())
                    .flat_map(|i| (0..w.weights.ncols()).map(move |j| (i, j)))
                    .map(|(i, j)| w.weights[(i, j)]),
            );
            out.extend((0..end.weights.nrows()).map(|i| end.weights[(i, 0)]));
            out.extend(&b.brier);
            out.extend(&b.null_brier);
            out.extend(b.rsquared.iter().map(|&v| absent(v)));
            out.extend(&b.p0);
            out.extend(&b.eff_n);
            out
        });
    }
}
