//! `cox::fit` and `cox::score_residuals` give identical bits in rayon pools of
//! 1, 2, 3, 4 and 8 threads.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties, fit, score_residuals};
use epistat_testkit::{POOLS, assert_same_on_pools};
use faer::Mat;

/// A value standing for `None`, so that an absent number differs from any
/// present one.
const ABSENT: u64 = 0x7ff8_0000_dead_beef;

fn opts(values: &[Option<f64>]) -> Vec<f64> {
    values
        .iter()
        .map(|v| v.unwrap_or_else(|| f64::from_bits(ABSENT)))
        .collect()
}

/// Strata, delayed entry, case weights with zeros, clusters, tied times, three
/// event codes and a design of four columns of which one is aliased.
#[test]
fn cox_fit_and_residuals() {
    let n = 2503;
    let mut rng = Stream::seeded(12);
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut stratum = Vec::new();
    let mut cluster = Vec::new();
    let mut base = Vec::new();
    for _ in 0..n {
        let a = rng.normal();
        let t = (rng.exp1() * (-0.4 * a).exp() * 8.0).ceil() / 10.0 + 0.1;
        stop.push(t);
        start.push((rng.uniform() * t * 8.0).floor() / 10.0 * 0.9);
        status.push(match rng.below(5) {
            0 | 1 => 1,
            2 => 2,
            _ => 0,
        });
        weight.push(match rng.below(7) {
            0 => 0.0,
            1 => 2.0,
            _ => 1.0,
        });
        stratum.push(rng.below(3) as u32);
        cluster.push(rng.below(37) as u32);
        base.push(a);
    }
    let noise: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
    let x = Mat::from_fn(n, 4, |i, j| match j {
        1 => noise[i],
        2 => base[i] + noise[i] * 0.5,
        _ => base[i],
    });
    let data = SurvData::builder(&stop, &status)
        .n_codes(2)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .cluster(&cluster)
        .build()
        .unwrap();
    for ties in [Ties::Breslow, Ties::Efron] {
        let options = CoxOptions {
            ties,
            cause: Some(1),
            ..CoxOptions::default()
        };
        assert_same_on_pools(&POOLS, || {
            let model = fit(&data, x.as_ref(), &options).unwrap();
            assert!(model.converged && model.coefficients[0].is_some());
            assert!(model.coefficients[3].is_none(), "the duplicate is aliased");
            let mut out = opts(&model.coefficients);
            out.extend(opts(&model.standard_errors));
            out.extend(opts(model.robust_standard_errors.as_ref().unwrap()));
            out.extend([
                model
                    .log_partial_likelihood
                    .unwrap_or_else(|| f64::from_bits(ABSENT)),
                model
                    .null_log_partial_likelihood
                    .unwrap_or_else(|| f64::from_bits(ABSENT)),
                model.n_events,
                model.n_rows as f64,
                model.n_used as f64,
                model.rank as f64,
                f64::from(model.iterations),
                f64::from(u8::from(model.converged)),
            ]);
            out.extend(model.monotone_columns.iter().map(|&c| c as f64));
            out.extend(opts(&model.covariance));
            out.extend(opts(model.robust_covariance.as_ref().unwrap()));
            let residuals =
                score_residuals(&data, x.as_ref(), &model.coefficients, &options).unwrap();
            for column in residuals {
                out.extend(column.unwrap_or_default());
            }
            out
        });
    }
}
