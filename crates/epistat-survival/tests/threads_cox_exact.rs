//! `cox::fit` with exact ties and `clogit` give identical bits in rayon pools of
//! 1, 2, 3, 4 and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::clogit::{ClogitMethod, ClogitOptions, clogit};
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_testkit::{POOLS, absent, absent_all, assert_same_on_pools};
use faer::Mat;

fn summary(model: &epistat_survival::cox::CoxModelFit) -> Vec<f64> {
    let mut out = absent_all(&model.coefficients);
    out.extend(absent_all(&model.standard_errors));
    out.extend(absent_all(&model.covariance));
    out.extend([
        absent(model.log_partial_likelihood),
        absent(model.null_log_partial_likelihood),
        model.n_events,
        model.n_rows as f64,
        model.rank as f64,
        f64::from(model.iterations),
        f64::from(u8::from(model.converged)),
    ]);
    out.extend(model.tests.score.map(|t| t.statistic));
    out.extend(model.tests.wald.map(|t| t.statistic));
    out
}

/// Strata, delayed entry, tied times, an offset and a duplicated column.
#[test]
fn cox_exact() {
    let n = 900;
    let mut rng = Stream::seeded(31);
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut stratum = Vec::new();
    let mut base = Vec::new();
    for _ in 0..n {
        let a = rng.normal();
        let t = (rng.exp1() * (-0.4 * a).exp() * 6.0).ceil() + 1.0;
        stop.push(t);
        start.push((rng.uniform() * t * 0.5).floor());
        status.push(i32::from(rng.below(4) != 0));
        stratum.push(rng.below(9) as u32);
        base.push(a);
    }
    let offset: Vec<f64> = (0..n).map(|_| 0.3 * rng.normal()).collect();
    let x = Mat::from_fn(n, 3, |i, j| {
        if j == 2 {
            base[i]
        } else {
            base[i] * (j as f64 + 1.0)
        }
    });
    for entry in [false, true] {
        let mut builder = SurvData::builder(&stop, &status).stratum(&stratum);
        if entry {
            builder = builder.start(&start);
        }
        let data = builder.build().unwrap();
        let options = CoxOptions {
            ties: Ties::Exact,
            offset: Some(&offset),
            ..CoxOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let model = fit(&data, x.as_ref(), &options).unwrap();
            assert!(model.converged && model.coefficients[0].is_some());
            assert!(model.coefficients[2].is_none(), "the duplicate is aliased");
            summary(&model)
        });
    }
}

/// Matched sets with one to three cases.
#[test]
fn clogit_methods() {
    let n_sets = 400usize;
    let mut rng = Stream::seeded(32);
    let mut case = Vec::new();
    let mut set = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for s in 0..n_sets {
        let size = 3 + rng.below(4);
        let cases = 1 + rng.below(3);
        for k in 0..size {
            case.push(k < cases.min(size - 1));
            set.push(s as u32);
            a.push(rng.normal());
            b.push(f64::from(rng.below(2) as u32));
        }
    }
    let rows = case.len();
    let x = Mat::from_fn(rows, 2, |i, j| if j == 0 { a[i] } else { b[i] });
    for method in [
        ClogitMethod::Exact,
        ClogitMethod::Approximate,
        ClogitMethod::Efron,
        ClogitMethod::Breslow,
    ] {
        let options = ClogitOptions {
            method,
            ..ClogitOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let model = clogit(&case, &set, x.as_ref(), &options).unwrap();
            assert!(model.converged);
            summary(&model)
        });
    }
}
