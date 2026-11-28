//! Simulated clustered data for the behaviour, thread and golden tests.

use epistat_linalg::{Mat, MatRef};
use epistat_mixed::{Error, MixedData};
use epistat_testkit::SplitMix64;

/// A data set: design, response, cluster labels, slope variable, weights
/// and offset.
pub struct Data {
    pub x: Mat<f64>,
    pub y: Vec<f64>,
    pub cluster: Vec<u32>,
    pub slope: Vec<f64>,
    pub weights: Vec<f64>,
    pub offset: Vec<f64>,
}

fn normal(rng: &mut SplitMix64) -> f64 {
    let (a, b) = (1.0 - rng.unit(), rng.unit());
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

/// `clusters` clusters of 1 to 6 rows with labels in a scrambled order, three
/// columns (intercept, a normal and a binary covariate) and a response of
/// the given kind: "gaussian" with a random intercept and slope, "binomial"
/// (0/1) or "poisson" with a random intercept.
pub fn simulate(clusters: usize, kind: &str, seed: u64) -> Data {
    let mut rng = SplitMix64::new(seed);
    let (mut rows, mut cluster, mut slope, mut y, mut weights, mut offset) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for c in 0..clusters {
        let size = 1 + usize::try_from(rng.below(6)).unwrap();
        let label = u32::try_from((c * 7919) % 100_003).unwrap();
        let (b0, b1) = (0.8 * normal(&mut rng), 0.3 * normal(&mut rng));
        for _ in 0..size {
            let x1 = normal(&mut rng);
            let x2 = f64::from(u8::from(rng.unit() < 0.4));
            let t = 3.0 * rng.unit();
            let o = 0.2 * normal(&mut rng);
            let eta = 0.3 + 0.5 * x1 - 0.4 * x2 + o + b0;
            let response = match kind {
                "gaussian" => eta + 0.2 * t + b1 * t + normal(&mut rng),
                "binomial" => f64::from(u8::from(rng.unit() < 1.0 / (1.0 + (-eta).exp()))),
                _ => {
                    // Poisson by inversion.
                    let mean = eta.exp();
                    let (mut k, mut p, u) = (0.0, (-mean).exp(), rng.unit());
                    let mut cdf = p;
                    while u > cdf && k < 1000.0 {
                        k += 1.0;
                        p *= mean / k;
                        cdf += p;
                    }
                    k
                }
            };
            rows.push([1.0, x1, x2]);
            cluster.push(label);
            slope.push(t);
            y.push(response);
            weights.push(0.5 + rng.unit());
            offset.push(o);
        }
    }
    let n = rows.len();
    Data {
        x: Mat::from_fn(n, 3, |i, j| rows[i][j]),
        y,
        cluster,
        slope,
        weights,
        offset,
    }
}

/// The data view of the given columns.
pub fn view<'a>(
    x: MatRef<'a, f64>,
    y: &'a [f64],
    groups: &'a [u32],
    slope: Option<&'a [f64]>,
    weights: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
) -> Result<MixedData<'a>, Error> {
    let mut data = MixedData::builder(x, y, groups);
    if let Some(slope) = slope {
        data = data.slope(slope);
    }
    if let Some(weights) = weights {
        data = data.weights(weights);
    }
    if let Some(offset) = offset {
        data = data.offset(offset);
    }
    data.build()
}
