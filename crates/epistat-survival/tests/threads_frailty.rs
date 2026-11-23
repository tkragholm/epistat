//! `frailty::fit` gives identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads, with the profile search, strata, entry times and Efron ties.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvBuf;
use epistat_survival::frailty::{FrailtyFit, FrailtyOptions, fit};
use epistat_testkit::{POOLS, assert_same_on_pools};

fn flat(f: &FrailtyFit) -> Vec<f64> {
    let mut out: Vec<f64> = f
        .coefficients
        .iter()
        .map(|b| b.unwrap_or(f64::NAN))
        .collect();
    for m in [
        &f.covariance,
        &f.covariance_sandwich,
        &f.diagonal_covariance,
        &f.diagonal_covariance_sandwich,
    ] {
        out.extend(m.iter().map(|v| v.unwrap_or(f64::NAN)));
    }
    out.extend(f.frailties.iter().map(|v| v.unwrap_or(f64::NAN)));
    out.extend(f.frailty_variances.iter().map(|v| v.unwrap_or(f64::NAN)));
    out.extend([
        f.theta,
        f.log_partial_likelihood,
        f.penalised_log_likelihood,
    ]);
    out.push(f.marginal_log_likelihood.unwrap_or(f64::NAN));
    out.extend(f.history.iter().map(|h| h.marginal_log_likelihood));
    out
}

#[test]
fn frailty_profile() {
    let n = 3000;
    let mut rng = Stream::seeded(17);
    let (mut time, mut start, mut status, mut cluster, mut stratum) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut z = 1.0;
    for i in 0..n {
        if i % 4 == 0 {
            z = rng.exp1() * 0.5 + 0.5;
        }
        let a = rng.normal();
        let b = f64::from(u8::from(rng.below(5) < 2));
        let t = rng.exp1() / (0.2 * z * (0.5 * a - 0.4 * b).exp());
        let c = rng.exp1() * 10.0;
        let stop = (t.min(c) * 10.0).ceil() / 10.0;
        time.push(stop);
        start.push(if rng.below(3) == 0 {
            (0.3 * stop * 10.0).floor() / 10.0
        } else {
            0.0
        });
        status.push(i32::from(t <= c));
        cluster.push(u32::try_from(i / 4).unwrap());
        stratum.push(u32::try_from(rng.below(2)).unwrap());
        x[(i, 0)] = a;
        x[(i, 1)] = b;
    }
    let buf = SurvBuf::builder(time, status)
        .start(start)
        .cluster(cluster)
        .stratum(stratum)
        .build()
        .unwrap();
    let data = buf.as_data();
    assert_same_on_pools(&POOLS, || {
        flat(&fit(&data, x.as_ref(), &FrailtyOptions::default()).unwrap())
    });
}
