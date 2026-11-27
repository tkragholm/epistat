//! Pseudo-values by inverse probability of censoring weighting, and the
//! regression on them with the HC3 covariance, give identical bits in rayon pools
//! of 1, 2, 3, 4 and 8 threads, for the Binder and Hajek forms and every target.
//! The data have two causes, tied times on a grid of tenths and two censoring
//! covariates, with more rows than a block of the Cox fit.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_glm::Link;
use epistat_glm::robust::SandwichType;
use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::pseudo::{
    IpcwMethod, IpcwOptions, PseudoRegressionOptions, Target, pseudo_ipcw,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn ipcw_pseudo_values_do_not_depend_on_the_number_of_threads() {
    let n = 5003;
    let mut rng = Stream::seeded(97);
    let (mut stop, mut status) = (Vec::new(), Vec::new());
    let mut z = Mat::<f64>::zeros(n, 2);
    for i in 0..n {
        let x1 = f64::from(rng.below(2) as u32);
        let x2 = rng.uniform();
        z[(i, 0)] = x1;
        z[(i, 1)] = x2;
        let event = (rng.exp1() * 40.0).floor() / 10.0;
        let censor = (rng.exp1() * (60.0 - 20.0 * x1 + 10.0 * x2)).floor() / 10.0;
        stop.push(0.1 + event.min(censor));
        status.push(if event <= censor {
            1 + rng.below(2) as i32
        } else {
            0
        });
    }
    let times = [0.5, 1.5, 3.0, 6.0];
    let competing = SurvData::builder(&stop, &status)
        .n_codes(2)
        .build()
        .unwrap();
    let single_status: Vec<i32> = status.iter().map(|&s| i32::from(s > 0)).collect();
    let single = SurvData::builder(&stop, &single_status).build().unwrap();
    for (data, target) in [
        (&competing, Target::CumulativeIncidence { cause: 2 }),
        (&competing, Target::RestrictedTimeLost { cause: 1 }),
        (&single, Target::Survival),
        (&single, Target::RestrictedMean),
    ] {
        for method in [IpcwMethod::Binder, IpcwMethod::Hajek] {
            let mut opts = IpcwOptions::new(target);
            opts.method = method;
            assert_same_on_pools(&POOLS, || {
                let p = pseudo_ipcw(data, z.as_ref(), &times, &opts).unwrap();
                let mut out = p.values.clone();
                out.extend_from_slice(&p.groups[0].estimate);
                out
            });
        }
    }
    // The HC3 regression on the pseudo-values at one time.
    let opts = IpcwOptions::new(Target::CumulativeIncidence { cause: 1 });
    assert_same_on_pools(&POOLS, || {
        let p = pseudo_ipcw(&competing, z.as_ref(), &[3.0], &opts).unwrap();
        let mut gee = PseudoRegressionOptions::new(Link::Identity);
        gee.covariance = SandwichType::Hc3;
        let fit = p.regress(Some(z.as_ref()), false, &gee).unwrap();
        let mut out = fit.coefficients.clone();
        out.extend(fit.covariance.as_slice().iter().map(|v| v.unwrap()));
        out.extend_from_slice(fit.leverage.as_ref().unwrap());
        out
    });
}
