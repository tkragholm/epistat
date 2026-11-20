//! `msreg::fit`, `MsModel::msfit` and `msreg::probtrans` give identical bits in
//! rayon pools of 1, 2, 3, 4 and 8 threads.
//!
//! Illness-death data for 4001 subjects with two covariates, integer times so
//! that events and censorings tie, separate and shared coefficients, with the
//! robust covariance, the cumulative hazards, their covariances, and the
//! transition probabilities with standard errors.

#![expect(
    clippy::cast_precision_loss,
    reason = "the counts and grid values are far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::msreg::{
    MsBaseline, MsFitOptions, MsModel, MsOptions, MsTerms, ProbTransOptions, fit, probtrans,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

fn push(out: &mut Vec<f64>, values: &[Option<f64>]) {
    out.extend(values.iter().map(|v| v.unwrap_or(f64::NAN)));
}

fn bits_of(model: &MsModel, profile: &[f64]) -> Vec<f64> {
    let mut out = Vec::new();
    push(&mut out, &model.fit.coefficients);
    push(&mut out, &model.fit.covariance);
    push(&mut out, model.fit.robust_covariance.as_ref().unwrap());
    out.push(model.fit.log_partial_likelihood.unwrap());
    let h = model.msfit(profile, &MsFitOptions::default()).unwrap();
    out.extend_from_slice(&h.time);
    for c in &h.cumulative_hazard {
        out.extend_from_slice(c);
    }
    for c in h.covariance.as_ref().unwrap() {
        out.extend_from_slice(c);
    }
    for predt in [0.0, 5.0] {
        let pt = probtrans(&h, predt, &ProbTransOptions::default()).unwrap();
        out.extend_from_slice(&pt.probability);
        push(&mut out, pt.std_err.as_ref().unwrap());
    }
    out
}

#[test]
fn msreg_does_not_depend_on_the_number_of_threads() {
    let n = 4001_u32;
    let mut rng = Stream::seeded(2026);
    let (mut id, mut start, mut stop, mut status, mut istate) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut covariates = Vec::new();
    for i in 0..n {
        let x1 = f64::from(rng.below(5) < 2);
        let x2 = rng.below(5) as f64 / 2.0;
        let (mut t, mut state) = (0.0, 1_u32);
        loop {
            let wait = 1.0 + rng.below(12) as f64;
            let stop_at = t + wait;
            id.push(1000 * i + 7);
            start.push(t);
            stop.push(stop_at);
            istate.push(state);
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
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .istate(&istate)
        .n_codes(3)
        .build()
        .unwrap();
    for (terms, baseline) in [
        (MsTerms::Separate, MsBaseline::Separate),
        (MsTerms::Common, MsBaseline::Shared),
    ] {
        let opts = MsOptions {
            terms,
            baseline,
            ..MsOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let model = fit(&data, x.as_ref(), &opts).unwrap();
            bits_of(&model, &[1.0, 1.0])
        });
    }
    // The model without covariates is a fit of a design with no column.
    let none = Mat::<f64>::zeros(covariates.len(), 0);
    assert_same_on_pools(&POOLS, || {
        let model = fit(&data, none.as_ref(), &MsOptions::default()).unwrap();
        bits_of(&model, &[])
    });
}
