//! The simulated data of the thread and golden tests of the balance module.

use epistat_causal::balance::BalanceWeights;
use epistat_linalg::Mat;
use epistat_rng::Stream;

pub struct Simulated {
    pub y: Vec<f64>,
    pub a: Vec<bool>,
    pub s: Vec<f64>,
    pub q: Vec<f64>,
    pub cluster: Vec<u32>,
    /// Three covariates, no intercept.
    pub c: Mat<f64>,
    /// The same with an intercept.
    pub x: Mat<f64>,
}

pub fn simulate(n: usize) -> Simulated {
    let mut rng = Stream::seeded(31);
    let (mut y, mut a, mut s, mut q, mut cluster) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut cols = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..n {
        let u = rng.normal();
        let v = f64::from(u8::from(rng.uniform() < 0.4));
        let w = (rng.normal() * 0.5).exp();
        let e = 1.0 / (1.0 + (-(-0.3 + 0.6 * u - 0.5 * v + 0.25 * w)).exp());
        let t = rng.uniform() < e;
        y.push(1.0 + 0.5 * u - 0.8 * v + 0.3 * w + 1.2 * f64::from(u8::from(t)) + rng.normal());
        a.push(t);
        s.push(0.5 + 1.5 * rng.uniform());
        q.push(0.5 + rng.uniform());
        cluster.push(u32::try_from(i % 600).unwrap());
        cols[0].push(u);
        cols[1].push(v);
        cols[2].push(w);
    }
    let c = Mat::from_fn(n, 3, |i, j| cols[j][i]);
    let x = Mat::from_fn(n, 4, |i, j| if j == 0 { 1.0 } else { cols[j - 1][i] });
    Simulated {
        y,
        a,
        s,
        q,
        cluster,
        c,
        x,
    }
}

/// Everything a thread or golden test compares for a set of weights and the
/// estimates made with them.
pub fn flatten(w: &BalanceWeights, s: &Simulated) -> Vec<f64> {
    use epistat_causal::balance::WeightVariance;
    let mut out = w.weights.clone();
    for g in [w.controls, w.treated] {
        out.extend([g.sum, g.min, g.max, g.effective_sample_size]);
    }
    for v in [WeightVariance::Estimated, WeightVariance::Fixed] {
        let r = w.mean_difference(&s.y, Some(&s.cluster), v, 0.05).unwrap();
        for e in [r.mean1, r.mean0, r.difference] {
            out.extend([
                e.estimate,
                e.se,
                e.lower,
                e.upper,
                e.p_value.unwrap_or(f64::NAN),
            ]);
        }
    }
    let design = Mat::from_fn(s.y.len(), 3, |i, j| match j {
        0 => 1.0,
        1 => f64::from(u8::from(s.a[i])),
        _ => s.c[(i, 0)],
    });
    let reg = w
        .weighted_regression(&s.y, design.as_ref(), None, WeightVariance::Estimated)
        .unwrap();
    out.extend(reg.coefficients);
    out.extend(reg.covariance);
    out
}
