//! `MsModel::msfit_many`, the Greenwood variances, `probtrans` at a fixed
//! horizon with its covariance, and `probtrans_many` give identical bits in
//! rayon pools of 1, 2, 3, 4 and 8 threads.
//!
//! Illness-death data for 3001 subjects with two covariates and weights,
//! integer times so that events and censorings tie, and several profiles.

#![expect(
    clippy::cast_precision_loss,
    reason = "the counts and grid values are far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::msreg::{
    CumulativeHazards, MsFitOptions, MsOptions, MsProfile, MsVarType, ProbTrans,
    ProbTransDirection, ProbTransMethod, ProbTransOptions, fit, probtrans_many,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

fn push_hazards(out: &mut Vec<f64>, h: &CumulativeHazards) {
    out.extend_from_slice(&h.time);
    for c in &h.cumulative_hazard {
        out.extend_from_slice(c);
    }
    for c in h.covariance.as_ref().unwrap() {
        out.extend_from_slice(c);
    }
}

fn push_probabilities(out: &mut Vec<f64>, pt: &ProbTrans) {
    out.extend_from_slice(&pt.time);
    out.extend_from_slice(&pt.probability);
    out.extend(
        pt.se
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );
    if let Some(c) = &pt.covariance {
        out.extend_from_slice(c);
    }
}

#[test]
fn msreg_more_does_not_depend_on_the_number_of_threads() {
    let n = 3001_u32;
    let mut rng = Stream::seeded(2027);
    let (mut id, mut start, mut stop, mut status, mut istate, mut weight) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut covariates = Vec::new();
    for i in 0..n {
        let x1 = f64::from(rng.below(5) < 2);
        let x2 = rng.below(5) as f64 / 2.0;
        let w = 0.5 + rng.below(4) as f64 / 2.0;
        let (mut t, mut state) = (0.0, 1_u32);
        loop {
            let stop_at = t + 1.0 + rng.below(12) as f64;
            id.push(1000 * i + 7);
            start.push(t);
            stop.push(stop_at);
            istate.push(state);
            weight.push(w);
            covariates.push([x1, x2]);
            let u = rng.below(100);
            let to: u32 = if state == 1 && u < 40 {
                2
            } else if u < 60 {
                3
            } else {
                0
            };
            status.push(i32::try_from(to).unwrap());
            if to == 0 || to == 3 || stop_at > 60.0 {
                break;
            }
            t = stop_at;
            state = to;
        }
    }
    let x = Mat::from_fn(covariates.len(), 2, |i, j| covariates[i][j]);
    let none = Mat::<f64>::zeros(covariates.len(), 0);
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .istate(&istate)
        .n_codes(3)
        .build()
        .unwrap();
    let weighted = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .istate(&istate)
        .n_codes(3)
        .weight(&weight)
        .build()
        .unwrap();
    let rows = [vec![1.0, 1.0], vec![0.0, 0.5], vec![1.0, 0.0]];
    let by_transition = [vec![1.0, 0.0], vec![0.0, 2.0], vec![1.0, 1.0]];
    let profiles = [
        MsProfile::Same(&rows[0]),
        MsProfile::Same(&rows[1]),
        MsProfile::ByTransition(&by_transition),
        MsProfile::Same(&rows[2]),
    ];
    let options = [
        ProbTransOptions {
            covariance: true,
            ..ProbTransOptions::default()
        },
        ProbTransOptions {
            direction: ProbTransDirection::FixedHorizon,
            method: ProbTransMethod::Greenwood,
            covariance: true,
            ..ProbTransOptions::default()
        },
        ProbTransOptions {
            direction: ProbTransDirection::FixedHorizon,
            ..ProbTransOptions::default()
        },
    ];
    assert_same_on_pools(&POOLS, || {
        let mut out = Vec::new();
        for d in [&data, &weighted] {
            let model = fit(d, x.as_ref(), &MsOptions::default()).unwrap();
            let many = model
                .msfit_many(&profiles, &MsFitOptions::default())
                .unwrap();
            for h in &many {
                push_hazards(&mut out, h);
            }
            for opts in &options {
                for predt in [3.0, 8.0] {
                    out.push(predt);
                    for p in &probtrans_many(&many, predt, opts).unwrap() {
                        push_probabilities(&mut out, p);
                    }
                }
            }
        }
        // The Greenwood variance of a model without coefficients.
        let null = fit(&data, none.as_ref(), &MsOptions::default()).unwrap();
        let h = null
            .msfit(
                &[],
                &MsFitOptions {
                    vartype: MsVarType::Greenwood,
                    ..MsFitOptions::default()
                },
            )
            .unwrap();
        push_hazards(&mut out, &h);
        out
    });
}
