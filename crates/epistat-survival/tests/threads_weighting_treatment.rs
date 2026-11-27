//! The overlap, matching and optimal-subset weights and the pooled logistic
//! treatment weights, with the Cox fit weighted by them, give identical bits in
//! rayon pools of 1, 2, 3, 4 and 8 threads.

mod treatment_sim;

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::weighting::cox as wcox;
use epistat_survival::weighting::iptw::{self, Estimand, IptwOptions};
use epistat_survival::weighting::pooled::PooledOptions;
use epistat_survival::weighting::pooled::{PooledEvent, TreatmentHistory};
use epistat_survival::weighting::{Censoring, WeightAt, Weights};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn overlap_weights_are_the_same_on_every_pool() {
    let s = treatment_sim::simulate();
    for estimand in [Estimand::Ato, Estimand::Atm, Estimand::Atos] {
        assert_same_on_pools(&POOLS, || {
            let w = iptw::fit(
                &s.treat,
                s.z.as_ref(),
                &IptwOptions {
                    estimand,
                    stabilise: false,
                    alpha: None,
                },
            )
            .unwrap();
            let mut out = w.weights.clone();
            out.extend(&w.propensity);
            out.extend(&w.coefficients);
            out.extend(w.alpha);
            for g in [w.summary.treated.unwrap(), w.summary.control.unwrap()] {
                out.extend([
                    g.sum,
                    g.min,
                    g.max,
                    g.mean,
                    g.coefficient_of_variation.unwrap(),
                    g.effective_sample_size,
                ]);
            }
            out
        });
    }
}

#[test]
fn treatment_weights_are_the_same_on_every_pool() {
    let s = treatment_sim::simulate();
    let rows = s.stop.len();
    let x = Mat::from_fn(rows, 2, |i, j| {
        if j == 0 {
            f64::from(u8::from(s.row_treated[i]))
        } else {
            s.design[(i, 1)]
        }
    });
    let data = SurvData::builder(&s.stop, &s.status)
        .start(&s.start)
        .id(&s.id)
        .build()
        .unwrap();
    let numerator = Mat::from_fn(rows, 2, |i, j| s.design[(i, 3 * j)]);
    for (history, stabilise, trunc) in [
        (TreatmentHistory::All, false, None),
        (TreatmentHistory::All, true, Some((0.02, 0.98))),
        (TreatmentHistory::First, true, None),
    ] {
        assert_same_on_pools(&POOLS, || {
            let w = epistat_survival::weighting::pooled::fit(
                &data,
                PooledEvent::Treatment {
                    treated: &s.row_treated,
                    history,
                },
                s.design.as_ref(),
                stabilise.then(|| numerator.as_ref()),
                &PooledOptions { truncation: trunc },
            )
            .unwrap();
            let mut out = w.weight.clone();
            out.extend(&w.weight_before);
            out.extend(&w.probability);
            out.extend(&w.coefficients);
            out.extend(w.untruncated.iter().flatten());
            let estimated = Weights {
                iptw: None,
                subject: None,
                censoring: Some(Censoring::Pooled {
                    fit: &w,
                    at: WeightAt::Stop,
                }),
            };
            let cox = wcox::fit(&data, x.as_ref(), &CoxOptions::default(), &estimated).unwrap();
            out.extend(&cox.coefficients);
            out.extend(cox.covariance.as_slice().iter().map(|v| v.unwrap()));
            out.extend(cox.robust_covariance.as_slice().iter().map(|v| v.unwrap()));
            out
        });
    }
}
