//! Wall time of the attributable fraction of a logistic model, cross-sectional
//! and case-control, with a confounder, an interaction and clusters of four
//! rows. The driver and the environment are those of `bench/tier1/common.rs`,
//! run by `bench/run.sh`; the data are generated in the process from a
//! fixed seed and the fit is not timed.
//!
//! Methods: `af_cross` and `af_case_control`, `N` rows.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng};
use epistat_glm::attributable::{AfOptions, Design, attributable_logistic};
use epistat_glm::{Family, Glm, Link};
use epistat_linalg::Mat;

const SEED: u64 = 20_261_208;

fn main() {
    let (method, n) = (common::method(), common::size());
    let design = match method.as_str() {
        "af_cross" => Design::CrossSectional,
        "af_case_control" => Design::CaseControl,
        other => panic!("unknown method {other}"),
    };
    let mut rng = Rng(SEED);
    let z: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let e: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(rng.bernoulli(1.0 / (1.0 + (-0.4 * z[i]).exp())))))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let eta = -1.0 + 0.8 * e[i] + 0.5 * z[i] + 0.3 * e[i] * z[i];
            f64::from(u8::from(rng.bernoulli(1.0 / (1.0 + (-eta).exp()))))
        })
        .collect();
    let cluster: Vec<u32> = (0..n).map(|i| u32::try_from(i / 4).unwrap()).collect();
    let x = Mat::from_fn(n, 4, |i, j| [1.0, e[i], z[i], e[i] * z[i]][j]);
    let x0 = Mat::from_fn(n, 4, |i, j| if j == 1 || j == 3 { 0.0 } else { x[(i, j)] });
    let fit = Glm::new(Family::Binomial, Link::Logit)
        .fit(x.as_ref(), &y, None, None, None)
        .expect("fit");
    let opts = AfOptions {
        design,
        cluster: Some(&cluster),
        ..AfOptions::default()
    };
    let (times, result) = common::timed(common::reps(), || {
        attributable_logistic(&fit, x.as_ref(), x0.as_ref(), &y, &opts).expect("af")
    });
    let mut lines = Lines::default();
    lines.push_opt("af", result.af);
    lines.push_opt("se", result.se);
    lines.push_opt("lower", result.lower);
    lines.push_opt("upper", result.upper);
    common::finish(&method, n, &times, &lines);
}
