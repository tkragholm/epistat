//! The Cox censoring model of `metrics::score`, `metrics::contrast`,
//! `metrics::recalibration` and `metrics::rttright` with a start column give
//! identical bits in rayon pools of 1, 2, 3, 4 and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::metrics::contrast::{ContrastOptions, auc_difference, brier_difference};
use epistat_survival::metrics::recalibration::{
    Bandwidth, RecalibrationOptions, RecalibrationScale, SmoothOptions, slope_intercept,
    smoothed_curve,
};
use epistat_survival::metrics::rttright::{self, RttrightOptions};
use epistat_survival::metrics::score::{
    Censoring, CoxCensoring, ScoreOptions, auc, brier, null_brier,
};
use epistat_testkit::{POOLS, absent, assert_same_on_pools};

/// Times, status, censoring covariates and the predictions of two models.
type Grid = (Vec<f64>, Vec<i32>, Mat<f64>, Mat<f64>, Mat<f64>);

fn grid_sample(n: usize, n_times: usize, seed: u64) -> Grid {
    let mut rng = Stream::seeded(seed);
    let (mut time, mut status) = (Vec::new(), Vec::new());
    let mut x = Mat::zeros(n, 2);
    let mut a = Mat::zeros(n, n_times);
    let mut b = Mat::zeros(n, n_times);
    for i in 0..n {
        time.push(0.1 + (rng.exp1() * 50.0).floor() / 10.0);
        status.push(match rng.below(5) {
            0 | 1 => 0,
            2 | 3 => 1,
            _ => 2,
        });
        x[(i, 0)] = rng.below(9) as f64 / 4.0 - 1.0;
        x[(i, 1)] = rng.below(2) as f64;
        for j in 0..n_times {
            a[(i, j)] = (rng.below(40) as f64 + 5.0 * j as f64 + 1.0) / 100.0;
            b[(i, j)] = (rng.below(50) as f64 + 4.0 * j as f64 + 1.0) / 100.0;
        }
    }
    (time, status, x, a, b)
}

#[test]
fn cox_censoring_contrasts_and_calibration() {
    let n_times = 11;
    let n = 3001;
    let (time, status, x, a, b) = grid_sample(n, n_times, 31);
    let data = SurvData::builder(&time, &status)
        .n_codes(2)
        .build()
        .unwrap();
    let times: Vec<f64> = (1..=n_times).map(|j| 0.4 * j as f64).collect();
    let opts = ScoreOptions {
        keep_influence: true,
        censoring: Censoring::Cox(CoxCensoring {
            x: x.as_ref(),
            options: CoxOptions::default(),
        }),
        ..ScoreOptions::default()
    };
    assert_same_on_pools(&POOLS, || {
        let sa = brier(&data, a.as_ref(), &times, &opts).unwrap();
        let sb = brier(&data, b.as_ref(), &times, &opts).unwrap();
        let s0 = null_brier(&data, &times, &opts).unwrap();
        let aa = auc(&data, a.as_ref(), &times, &opts).unwrap();
        let ab = auc(&data, b.as_ref(), &times, &opts).unwrap();
        let level = ContrastOptions::default();
        let mut out = Vec::new();
        for s in [&sa, &sb, &s0] {
            out.extend(&s.brier);
            out.extend(s.se.iter().map(|&v| absent(v)));
        }
        for c in [
            brier_difference(&sb, &sa, &level).unwrap(),
            brier_difference(&sa, &s0, &level).unwrap(),
            auc_difference(&ab, &aa, &level).unwrap(),
        ] {
            for column in [&c.difference, &c.se, &c.lower, &c.upper, &c.p_value] {
                out.extend(column.iter().map(|&v| absent(v)));
            }
        }
        let inf = sa.influence.unwrap();
        out.extend((0..inf.n_columns()).flat_map(|j| inf.column(j).to_vec()));
        out
    });

    let single: Vec<i32> = status.iter().map(|&v| i32::from(v == 1)).collect();
    let one = SurvData::builder(&time, &single).build().unwrap();
    let risk: Vec<f64> = (0..n).map(|i| a[(i, 4)]).collect();
    assert_same_on_pools(&POOLS, || {
        let curve = smoothed_curve(
            &one,
            &risk,
            2.0,
            &SmoothOptions::new(Bandwidth::Fixed(0.05)),
        )
        .unwrap();
        let mut out = vec![curve.bandwidth];
        for p in &curve.points {
            out.extend([p.predicted, p.observed, p.n_neighbors as f64]);
        }
        for scale in [RecalibrationScale::Logit, RecalibrationScale::Cloglog] {
            let r = slope_intercept(&one, &risk, 2.0, &RecalibrationOptions::new(scale)).unwrap();
            out.extend([r.intercept, r.slope, r.information_condition]);
            out.extend(r.covariance.as_slice().iter().map(|v| v.unwrap()));
        }
        out
    });
}

#[test]
fn rttright_with_a_start() {
    let n: u32 = 2501;
    let mut rng = Stream::seeded(41);
    let (mut id, mut start, mut stop, mut status, mut weight, mut stratum) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for subject in 0..n {
        let end = (11.0 + (rng.exp1() * 40.0).floor()) / 10.0;
        let event = match rng.below(5) {
            0 | 1 => 0,
            2 | 3 => 1,
            _ => 2,
        };
        let w = (1 + rng.below(3)) as f64;
        let g = rng.below(3) as u32;
        let mut edges = vec![1.0];
        for _ in 0..rng.below(3) {
            let cut = (10 + rng.below(100)) as f64 / 10.0;
            if cut > 1.0 && cut < end && !edges.contains(&cut) {
                edges.push(cut);
            }
        }
        edges.sort_by(f64::total_cmp);
        edges.push(end);
        for r in 0..edges.len() - 1 {
            id.push(subject * 3 + 1);
            start.push(edges[r]);
            stop.push(edges[r + 1]);
            status.push(if r + 2 == edges.len() { event } else { 0 });
            weight.push(w);
            stratum.push(g);
        }
    }
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .weight(&weight)
        .stratum(&stratum)
        .n_codes(2)
        .build()
        .unwrap();
    let times: Vec<f64> = (1..=11).map(|j| f64::from(10 + 4 * j) / 10.0).collect();
    assert_same_on_pools(&POOLS, || {
        let mut out = Vec::new();
        for given in [None, Some(times.as_slice())] {
            let w = rttright::estimate(&data, given, &RttrightOptions::default()).unwrap();
            out.extend(
                (0..w.weights.nrows())
                    .flat_map(|i| (0..w.weights.ncols()).map(move |j| (i, j)))
                    .map(|(i, j)| w.weights[(i, j)]),
            );
        }
        out
    });
}
