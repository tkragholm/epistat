//! The simulated data of the thread and golden tests.

use epistat_causal::aipw::{Aipw, GlmInput};
use epistat_linalg::Mat;
use epistat_rng::Stream;

pub struct Simulated {
    pub y: Vec<f64>,
    pub a: Vec<bool>,
    pub weight: Vec<f64>,
    pub cluster: Vec<u32>,
    pub x: Mat<f64>,
}

pub fn simulate() -> Simulated {
    let n = 3000;
    let mut rng = Stream::seeded(24);
    let (mut y, mut a, mut weight, mut cluster) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut cols = [Vec::new(), Vec::new()];
    for i in 0..n {
        let (u, v) = (rng.normal(), f64::from(u8::from(rng.uniform() < 0.4)));
        let e = 1.0 / (1.0 + (-(0.4 * u - 0.5 * v - 0.1)).exp());
        let t = rng.uniform() < e;
        let p = 1.0 / (1.0 + (-(-0.5 + 0.8 * u + 0.6 * v + 0.7 * f64::from(u8::from(t)))).exp());
        y.push(f64::from(u8::from(rng.uniform() < p)));
        a.push(t);
        weight.push(0.5 + 1.5 * rng.uniform());
        cluster.push(u32::try_from(i % 500).unwrap());
        cols[0].push(u);
        cols[1].push(v);
    }
    let x = Mat::from_fn(n, 3, |i, j| if j == 0 { 1.0 } else { cols[j - 1][i] });
    Simulated {
        y,
        a,
        weight,
        cluster,
        x,
    }
}

impl Simulated {
    pub fn input(&self) -> GlmInput<'_> {
        GlmInput {
            outcome: &self.y,
            treatment: &self.a,
            outcome_design: self.x.as_ref(),
            propensity_design: self.x.as_ref(),
            weight: Some(&self.weight),
            cluster: Some(&self.cluster),
        }
    }

    /// Predictions that depend on the covariates only through simple formulas.
    pub fn models(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.y.len();
        let eta = |i: usize| 0.8 * self.x[(i, 1)] + 0.6 * self.x[(i, 2)] - 0.5;
        let expit = |z: f64| 1.0 / (1.0 + (-z).exp());
        (
            (0..n).map(|i| expit(eta(i) + 0.7)).collect(),
            (0..n).map(|i| expit(eta(i))).collect(),
            (0..n)
                .map(|i| expit(0.4 * self.x[(i, 1)] - 0.5 * self.x[(i, 2)] - 0.1))
                .collect(),
        )
    }
}

pub fn flatten(r: &Aipw) -> Vec<f64> {
    let mut out = vec![r.n_trimmed as f64, r.min_propensity, r.max_propensity];
    for e in [r.mean1, r.mean0, r.difference].into_iter().chain(r.ratio) {
        out.extend([
            e.estimate,
            e.se,
            e.lower,
            e.upper,
            e.p_value.unwrap_or(f64::NAN),
        ]);
    }
    out.extend(&r.influence);
    out
}
