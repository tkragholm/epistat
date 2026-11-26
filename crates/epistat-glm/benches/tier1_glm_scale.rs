//! Wall time and peak memory of a logistic regression fitted by `Glm::fit` on
//! a column-major design of `N` rows, for the scaling of the fit itself. The
//! program makes its data in memory, so the peak memory of the process is the
//! design, the response and the fit. The driver and the environment are
//! described in `bench/tier1/common.rs`; there is no R counterpart.
//!
//! Methods: `glm_logit_pP`, an intercept and `P - 1` standard normal columns,
//! with a binary response from the logistic model with coefficients
//! `-1, 0.1, -0.1, 0.2, ...`. Every 50th column after the first two is a copy
//! of the second scaled by two, so that the aliasing test meets one aliased
//! column when `P` exceeds 50. The numbers written are the coefficients, the
//! standard errors, the deviance, the iteration count and a hash of the bits
//! of the fitted means, the linear predictor and the working weights, so two
//! builds can be compared for equality.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng};
use epistat_glm::{Family, Glm, GlmData, Link};
use epistat_linalg::Mat;

const SEED: u64 = 20_261_124;

fn columns_of(method: &str) -> usize {
    method
        .strip_prefix("glm_logit_p")
        .and_then(|p| p.parse().ok())
        .unwrap_or_else(|| panic!("unknown method {method}"))
}

fn slope(j: usize) -> f64 {
    let size = 0.1 * (1 + (j - 1) % 3) as f64;
    if j.is_multiple_of(2) { -size } else { size }
}

fn data(n: usize, p: usize) -> (Mat<f64>, Vec<f64>) {
    let mut rng = Rng(SEED);
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        x[(i, 0)] = 1.0;
    }
    for j in 1..p {
        if j >= 2 && j.is_multiple_of(50) {
            for i in 0..n {
                x[(i, j)] = 2.0 * x[(i, 1)];
            }
        } else {
            for i in 0..n {
                x[(i, j)] = rng.normal();
            }
        }
    }
    let y = (0..n)
        .map(|i| {
            let eta: f64 = -1.0
                + (1..p)
                    .filter(|&j| j < 2 || !j.is_multiple_of(50))
                    .map(|j| slope(j) * x[(i, j)])
                    .sum::<f64>();
            f64::from(u8::from(rng.uniform() < 1.0 / (1.0 + (-eta).exp())))
        })
        .collect();
    (x, y)
}

/// FNV-1a over the bits of the values.
fn hash(values: &[f64]) -> f64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for v in values {
        h ^= v.to_bits();
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h >> 11) as f64
}

fn main() {
    let (method, n) = (common::method(), common::size());
    let p = columns_of(&method);
    let (x, y) = data(n, p);
    let model = Glm::new(Family::Binomial, Link::Logit);
    let (times, fit) = common::timed(common::reps(), || {
        let data = GlmData::builder(x.as_ref(), &y).build().expect("data");
        model.fit(&data).expect("logistic fit")
    });
    let mut lines = Lines::default();
    let se = fit.standard_errors();
    for (j, (&coef, &se)) in fit.coefficients.iter().zip(&se).enumerate() {
        lines.push_opt(&format!("coef_{j}"), coef);
        lines.push_opt(&format!("se_{j}"), se);
    }
    lines.push("deviance", fit.deviance);
    lines.push("iterations", fit.iterations as f64);
    lines.push("fitted_hash", hash(&fit.fitted));
    lines.push("eta_hash", hash(&fit.linear_predictor));
    lines.push("weights_hash", hash(&fit.working_weights));
    common::finish(&method, n, &times, &lines);
}
