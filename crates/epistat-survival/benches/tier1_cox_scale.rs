//! Wall time and peak memory of a Cox fit by `cox::fit` with Efron ties on
//! `N` rows, for the scaling of the fit itself. The program makes its data in
//! memory, so the peak memory of the process is the data, the design and the
//! fit. The driver and the environment are described in
//! `bench/tier1/common.rs`; there is no R counterpart.
//!
//! Methods:
//!
//! * `cox_pP`: `P` standard normal columns, an exponential event time with
//!   log hazard `0.1 x_1 - 0.1 x_2 + 0.2 x_3 ...` and a uniform censoring
//!   time, both in whole days up to ten years, so that many events tie.
//! * `cox_entry_pP`: the same with a delayed entry of up to two years, ten
//!   strata and clusters of three consecutive rows, so the fit also forms the
//!   cluster-robust variance.
//!
//! The numbers written are the coefficients, both standard errors, the log
//! partial likelihood and the iteration count, so two builds can be compared
//! for equality.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng};
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, fit};

const SEED: u64 = 20_261_125;
const HORIZON: f64 = 3652.0;

struct Data {
    x: Mat<f64>,
    stop: Vec<f64>,
    start: Option<Vec<f64>>,
    status: Vec<i32>,
    stratum: Option<Vec<u32>>,
    cluster: Option<Vec<u32>>,
}

fn slope(j: usize) -> f64 {
    let size = 0.1 * (1 + j % 3) as f64;
    if j.is_multiple_of(2) { size } else { -size }
}

fn data(n: usize, p: usize, entry: bool) -> Data {
    let mut rng = Rng(SEED);
    let mut x = Mat::<f64>::zeros(n, p);
    for j in 0..p {
        for i in 0..n {
            x[(i, j)] = rng.normal();
        }
    }
    let (mut stop, mut status) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut start = Vec::new();
    for i in 0..n {
        let eta: f64 = (0..p).map(|j| slope(j) * x[(i, j)]).sum();
        let event = -rng.uniform().ln() / (eta.exp() / 2000.0);
        let censor = rng.uniform() * HORIZON;
        let begin = if entry {
            (rng.uniform() * 730.0).floor()
        } else {
            0.0
        };
        let end = event.min(censor).ceil().max(begin + 1.0);
        stop.push(end);
        status.push(i32::from(event <= censor));
        start.push(begin);
    }
    Data {
        x,
        stop,
        start: entry.then_some(start),
        status,
        stratum: entry.then(|| {
            (0..n)
                .map(|i| u32::try_from(i % 10).expect("a small code"))
                .collect()
        }),
        cluster: entry.then(|| {
            (0..n)
                .map(|i| u32::try_from(i / 3).expect("fewer than 2^32 rows"))
                .collect()
        }),
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    let (entry, p) = if let Some(p) = method.strip_prefix("cox_entry_p") {
        (true, p)
    } else if let Some(p) = method.strip_prefix("cox_p") {
        (false, p)
    } else {
        panic!("unknown method {method}")
    };
    let p: usize = p.parse().expect("a column count");
    let d = data(n, p, entry);
    let mut builder = SurvData::builder(&d.stop, &d.status);
    if let Some(start) = &d.start {
        builder = builder.start(start);
    }
    if let Some(stratum) = &d.stratum {
        builder = builder.stratum(stratum);
    }
    if let Some(cluster) = &d.cluster {
        builder = builder.cluster(cluster);
    }
    let surv = builder.build().expect("data");
    let opts = CoxOptions::default();
    let (times, f) = common::timed(common::reps(), || {
        fit(&surv, d.x.as_ref(), &opts).expect("cox fit")
    });
    let mut lines = Lines::default();
    for (j, (&coef, &se)) in f.coefficients.iter().zip(&f.se).enumerate() {
        lines.push_opt(&format!("coef_{j}"), coef);
        lines.push_opt(&format!("se_{j}"), se);
        lines.push_opt(
            &format!("robust_se_{j}"),
            f.robust_se.as_ref().and_then(|robust| robust[j]),
        );
    }
    lines.push_opt("loglik", f.log_partial_likelihood);
    lines.push("iterations", f64::from(f.iterations));
    common::finish(&method, n, &times, &lines);
}
