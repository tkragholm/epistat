//! The inclusion probabilities, the weighted fit and the absolute risk of
//! nested case-control data give identical bits in rayon pools of 1, 2, 3, 4
//! and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::ncc::{
    NccDesign, absolute_risk, baseline, fit_weighted, inclusion_probabilities,
};
use epistat_testkit::{POOLS, absent, assert_same_on_pools};

#[test]
fn nested_case_control_pipeline() {
    let n = 3000;
    let m = 2;
    let mut rng = Stream::seeded(91);
    let (mut entry, mut exit, mut case, mut stratum) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.below(3) == 0));
        let t = rng.exp1() / (0.1 * (0.5 * x1 - 0.4 * x2).exp());
        let c = rng.exp1() / 0.04;
        let e = t.min(c).min(6.0);
        entry.push(if rng.below(3) == 0 {
            rng.uniform() * 0.5 * e
        } else {
            0.0
        });
        exit.push(e);
        case.push(t <= c.min(6.0));
        stratum.push(rng.below(3) as u32);
        a.push(x1);
        b.push(x2);
    }
    // Controls sampled by a coin of the probability that matches roughly m
    // per case, which is enough for a fixed set of rows.
    let sampled: Vec<bool> = (0..n).map(|i| case[i] || rng.below(4) == 0).collect();
    let design = NccDesign {
        entry: Some(&entry),
        exit: &exit,
        case: &case,
        controls_per_case: m,
        stratum: Some(&stratum),
    };
    let probability = inclusion_probabilities(&design).unwrap().probability;
    let rows: Vec<usize> = (0..n)
        .filter(|&i| sampled[i] && (case[i] || probability[i] > 0.0))
        .collect();
    let stop: Vec<f64> = rows.iter().map(|&i| exit[i]).collect();
    let start: Vec<f64> = rows.iter().map(|&i| entry[i]).collect();
    let status: Vec<i32> = rows.iter().map(|&i| i32::from(case[i])).collect();
    let x = Mat::from_fn(
        rows.len(),
        2,
        |i, j| if j == 0 { a[rows[i]] } else { b[rows[i]] },
    );
    let profile = Mat::from_fn(2, 2, |i, j| [[0.0, 0.0], [1.0, 1.0]][i][j]);
    assert_same_on_pools(&POOLS, || {
        let inc = inclusion_probabilities(&design).unwrap();
        let w: Vec<f64> = rows
            .iter()
            .map(|&i| {
                if case[i] {
                    1.0
                } else {
                    1.0 / inc.probability[i]
                }
            })
            .collect();
        let data = SurvData::builder(&stop, &status)
            .start(&start)
            .build()
            .unwrap();
        let opts = CoxOptions::default();
        let model = fit_weighted(&data, x.as_ref(), &w, &opts).unwrap();
        let base = baseline(&model, &data, x.as_ref(), &w, &opts).unwrap();
        let risk = absolute_risk(
            &base,
            &model.coefficients,
            profile.as_ref(),
            &[1.0, 3.0, 5.0],
        )
        .unwrap();
        let mut out: Vec<f64> = inc.probability.clone();
        out.extend(inc.risk_set_size.iter().map(|&s| s as f64));
        out.extend(w.iter().copied());
        out.extend(model.coefficients.iter().copied().map(absent));
        out.extend(
            model
                .robust_covariance
                .as_ref()
                .unwrap()
                .as_slice()
                .iter()
                .copied()
                .map(absent),
        );
        out.extend(base.cumulative_hazard.iter().copied());
        out.extend(risk.iter().flatten().copied().map(absent));
        out
    });
}
