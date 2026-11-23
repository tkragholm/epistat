//! The simulated person-period data of the tests of treatment weights.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;

pub struct Simulated {
    pub z: Mat<f64>,
    pub treat: Vec<bool>,
    pub stop: Vec<f64>,
    pub start: Vec<f64>,
    pub status: Vec<i32>,
    pub id: Vec<u32>,
    pub row_treated: Vec<bool>,
    pub design: Mat<f64>,
}

pub fn simulate() -> Simulated {
    let n = 1500;
    let mut rng = Stream::seeded(81);
    let mut cols = [Vec::new(), Vec::new()];
    let mut treat = Vec::new();
    let (mut stop, mut start, mut status, mut id) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut row_treated, mut design_rows) = (Vec::new(), Vec::new());
    for i in 0..n {
        let (a, b) = (rng.normal(), f64::from(u8::from(rng.uniform() < 0.4)));
        let p = 1.0 / (1.0 + (-(0.3 * a + 0.5 * b - 0.2)).exp());
        treat.push(rng.uniform() < p);
        cols[0].push(a);
        cols[1].push(b);
        let mut l = rng.normal();
        let mut prev = false;
        for k in 0..6 {
            l = 0.5 * l + 0.8 * rng.normal() + 0.3 * f64::from(u8::from(prev));
            let pa =
                1.0 / (1.0 + (-(-1.0 + 0.7 * l + 0.3 * a + 1.0 * f64::from(u8::from(prev)))).exp());
            let t = rng.uniform() < pa;
            let ev = rng.uniform() < 0.06;
            start.push(f64::from(k));
            stop.push(f64::from(k) + 1.0);
            status.push(i32::from(ev));
            id.push(i as u32);
            row_treated.push(t);
            design_rows.push([1.0, a, l, f64::from(k)]);
            prev = t;
            if ev {
                break;
            }
        }
    }
    let z = Mat::from_fn(n, 3, |i, j| if j == 0 { 1.0 } else { cols[j - 1][i] });
    let design = Mat::from_fn(design_rows.len(), 4, |i, j| design_rows[i][j]);
    Simulated {
        z,
        treat,
        stop,
        start,
        status,
        id,
        row_treated,
        design,
    }
}
