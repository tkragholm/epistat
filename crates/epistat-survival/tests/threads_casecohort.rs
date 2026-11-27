//! The five case-cohort methods give identical bits in rayon pools of 1, 2, 3, 4
//! and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::casecohort::{CaseCohortMethod, CaseCohortOptions, fit};
use epistat_testkit::{POOLS, absent, assert_same_on_pools};

#[test]
fn case_cohort_methods() {
    let cohort = 2400_usize;
    let mut rng = Stream::seeded(77);
    let (mut stop, mut start, mut status, mut sub, mut stratum) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let mut sizes = [0.0_f64; 3];
    for _ in 0..cohort {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.below(3) == 0));
        let t = (rng.exp1() * (-0.5 * x1 + 0.3 * x2).exp() * 8.0).ceil() / 10.0 + 0.1;
        let c = (rng.exp1() * 12.0).ceil() / 10.0 + 0.1;
        let s = rng.below(3);
        sizes[s] += 1.0;
        let in_sub = rng.uniform() < 0.2;
        let event = t <= c;
        if event || in_sub {
            stop.push(t.min(c));
            start.push((rng.uniform() * t.min(c) * 8.0).floor() / 10.0 * 0.9);
            status.push(i32::from(event));
            sub.push(in_sub);
            stratum.push(s as u32);
            a.push(x1);
            b.push(x2);
        }
    }
    let rows = stop.len();
    let x = Mat::from_fn(rows, 2, |i, j| if j == 0 { a[i] } else { b[i] });
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .build()
        .unwrap();
    for method in [
        CaseCohortMethod::Prentice,
        CaseCohortMethod::SelfPrentice,
        CaseCohortMethod::LinYing,
        CaseCohortMethod::BorganI,
        CaseCohortMethod::BorganII,
    ] {
        let total = [cohort as f64];
        let cohort_size: &[f64] = if matches!(
            method,
            CaseCohortMethod::BorganI | CaseCohortMethod::BorganII
        ) {
            &sizes
        } else {
            &total
        };
        let strata = (cohort_size.len() > 1).then_some(stratum.as_slice());
        let options = CaseCohortOptions::new(method, cohort_size);
        assert_same_on_pools(&POOLS, || {
            let model = fit(&data, x.as_ref(), &sub, strata, &options)
                .unwrap_or_else(|e| panic!("{method:?}: {e}"));
            let mut out: Vec<f64> = Vec::new();
            for m in [
                &model.coefficients,
                &model.covariance,
                &model.naive_covariance,
                &model.phase_two_covariance,
            ] {
                out.extend(m.iter().copied().map(absent));
            }
            if let Some(opt) = &model.optimal_allocation {
                out.extend(opt.iter().copied().map(absent));
            }
            out.push(f64::from(u32::try_from(model.iterations).unwrap()));
            out
        });
    }
}
