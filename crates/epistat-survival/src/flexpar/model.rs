//! The log likelihood of the model on a follow-up table, its score and
//! information, and the maximisation.

use epistat_linalg::MatRef;
use epistat_optim::bfgs::Bfgs;

use super::Scale;
use super::design::Design;
use super::link;
use crate::ldl;
use crate::par;

const BLOCK: usize = 1024;
const WAVE: usize = 64;

/// The rows of a fit: the columns of the follow-up table and the design.
pub(super) struct Model<'a> {
    pub design: &'a Design,
    pub scale: Scale,
    pub x: MatRef<'a, f64>,
    pub stop: &'a [f64],
    pub start: Option<&'a [f64]>,
    pub event: &'a [i32],
    pub weight: Option<&'a [f64]>,
    pub offset: Option<&'a [f64]>,
}

/// Buffers for one row.
pub(super) struct Work {
    pub z: Vec<f64>,
    pub w: Vec<f64>,
    pub z0: Vec<f64>,
}

impl Work {
    pub(super) fn new(q: usize) -> Self {
        Self {
            z: vec![0.0; q],
            w: vec![0.0; q],
            z0: vec![0.0; q],
        }
    }
}

/// One row's log likelihood and the coefficients with which its score and
/// information are built from `z`, `w` and `z0`:
/// `score = sz z + sw w + s0 z0` and
/// `information = a z z' + b w w' + a0 z0 z0'`.
pub(super) struct RowTerms {
    pub ll: f64,
    pub sz: f64,
    pub sw: f64,
    pub s0: f64,
    pub a: f64,
    pub b: f64,
    pub a0: f64,
    pub has_entry: bool,
}

/// The sum over rows.
#[derive(Clone, Debug)]
pub(super) struct Totals {
    pub ll: f64,
    pub score: Vec<f64>,
    /// Row-major, lower triangle only until [`Totals::symmetrise`].
    pub info: Vec<f64>,
}

impl Totals {
    fn zero(q: usize, derivatives: bool) -> Self {
        Self {
            ll: 0.0,
            score: vec![0.0; if derivatives { q } else { 0 }],
            info: vec![0.0; if derivatives { q * q } else { 0 }],
        }
    }

    fn add(&mut self, other: &Self) {
        self.ll += other.ll;
        for (a, b) in self.score.iter_mut().zip(&other.score) {
            *a += b;
        }
        for (a, b) in self.info.iter_mut().zip(&other.info) {
            *a += b;
        }
    }

    fn symmetrise(&mut self, q: usize) {
        for i in 0..q {
            for j in 0..i {
                self.info[j * q + i] = self.info[i * q + j];
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn rank_one(info: &mut [f64], q: usize, coefficient: f64, v: &[f64]) {
    for i in 0..q {
        let c = coefficient * v[i];
        if c != 0.0 {
            let row = &mut info[i * q..=i * q + i];
            for (slot, vj) in row.iter_mut().zip(v) {
                *slot += c * vj;
            }
        }
    }
}

impl Model<'_> {
    pub(super) fn n(&self) -> usize {
        self.stop.len()
    }

    pub(super) fn dim(&self) -> usize {
        self.design.dim()
    }

    /// The terms of row `i` at `theta`, with `work.z`, `work.w` and `work.z0`
    /// set. `None` for a row of weight 0, `Err(())` where the derivative of
    /// the spline in an event row is not positive or the value is not finite.
    pub(super) fn row(
        &self,
        i: usize,
        theta: &[f64],
        work: &mut Work,
    ) -> std::result::Result<Option<RowTerms>, ()> {
        let c = self.weight.map_or(1.0, |w| w[i]);
        if c == 0.0 {
            return Ok(None);
        }
        let event = self.event[i] != 0;
        let u = self.stop[i].ln();
        let off = self.offset.map_or(0.0, |o| o[i]);
        self.design.fill(
            self.x,
            i,
            u,
            &mut work.z,
            if event { Some(&mut work.w) } else { None },
        );
        let eta = dot(&work.z, theta) + off;
        let (h, h1, h2) = link::cumulative(self.scale, eta);
        let mut ll = -h;
        let (mut sz, mut sw, mut a, mut b) = (-h1, 0.0, h2, 0.0);
        if event {
            let ws = dot(&work.w, theta);
            if ws.is_nan() || ws <= 0.0 {
                return Err(());
            }
            let (g, g1, g2) = link::log_hazard_factor(self.scale, eta);
            ll += g + ws.ln() - u;
            sz += g1;
            sw = 1.0 / ws;
            a -= g2;
            b = 1.0 / (ws * ws);
        }
        let start = self.start.map_or(0.0, |s| s[i]);
        let has_entry = start > 0.0;
        let (mut s0, mut a0) = (0.0, 0.0);
        if has_entry {
            let u0 = start.ln();
            self.design.fill(self.x, i, u0, &mut work.z0, None);
            let eta0 = dot(&work.z0, theta) + off;
            let (h0, h01, h02) = link::cumulative(self.scale, eta0);
            ll += h0;
            s0 = h01;
            a0 = -h02;
        }
        if !ll.is_finite() {
            return Err(());
        }
        Ok(Some(RowTerms {
            ll: c * ll,
            sz: c * sz,
            sw: c * sw,
            s0: c * s0,
            a: c * a,
            b: c * b,
            a0: c * a0,
            has_entry,
        }))
    }

    /// The log likelihood and, with `derivatives`, the score and information.
    /// `None` where the model is undefined: a derivative of the spline that is
    /// not positive at an event, or a non-finite value.
    pub(super) fn evaluate(&self, theta: &[f64], derivatives: bool) -> Option<Totals> {
        let q = self.dim();
        let make = |range: std::ops::Range<usize>| -> std::result::Result<Totals, ()> {
            let mut work = Work::new(q);
            let mut t = Totals::zero(q, derivatives);
            for i in range {
                let Some(r) = self.row(i, theta, &mut work)? else {
                    continue;
                };
                t.ll += r.ll;
                if derivatives {
                    for k in 0..q {
                        t.score[k] += r.sz * work.z[k]
                            + r.sw * work.w[k]
                            + if r.has_entry { r.s0 * work.z0[k] } else { 0.0 };
                    }
                    rank_one(&mut t.info, q, r.a, &work.z);
                    if r.b != 0.0 {
                        rank_one(&mut t.info, q, r.b, &work.w);
                    }
                    if r.has_entry {
                        rank_one(&mut t.info, q, r.a0, &work.z0);
                    }
                }
            }
            Ok(t)
        };
        let mut total =
            par::try_ordered_reduce(self.n(), BLOCK, WAVE, make, |t: &mut Totals, p| t.add(&p))
                .ok()?;
        if derivatives {
            total.symmetrise(q);
        }
        Some(total)
    }
}

/// How the maximisation ended.
pub(super) struct Maximum {
    pub theta: Vec<f64>,
    pub totals: Totals,
    pub iterations: u32,
    pub converged: bool,
}

/// Newton's method with step halving in the region where the model is
/// defined. The step solves `(I + r D) delta = score`, with the information
/// `I` and a ridge `r` of zero where `I` is positive definite, otherwise the
/// smallest of a ladder that makes it so. A fit that does not converge in
/// `max_iterations` or cannot improve along a step goes to BFGS and then back
/// to Newton.
pub(super) fn maximise(
    model: &Model<'_>,
    start: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Option<Maximum> {
    let mut theta = start.to_vec();
    let mut used = 0_u32;
    let mut best = newton(model, &mut theta, tolerance, max_iterations, &mut used)?;
    if !best.converged {
        let q = theta.len();
        let f = |x: &[f64]| match model.evaluate(x, true) {
            Some(t) => (-t.ll, t.score.iter().map(|s| -s).collect()),
            None => (1e300, vec![0.0; q]),
        };
        let minimum = Bfgs {
            max_iterations: 200,
            ..Bfgs::default()
        }
        .minimize(&best.theta, f, None);
        if let Ok(m) = minimum
            && m.value.is_finite()
            && m.value < 1e299
            && -m.value >= best.totals.ll
        {
            let mut again = m.x;
            if let Some(n) = newton(model, &mut again, tolerance, max_iterations, &mut used) {
                best = n;
            }
        }
    }
    best.iterations = used;
    Some(best)
}

fn newton(
    model: &Model<'_>,
    theta: &mut [f64],
    tolerance: f64,
    max_iterations: usize,
    used: &mut u32,
) -> Option<Maximum> {
    let q = theta.len();
    let mut current = model.evaluate(theta, true)?;
    let mut converged = false;
    for _ in 0..max_iterations {
        *used += 1;
        let (delta, ridge) = step(&current, q);
        let decrement = dot(&delta, &current.score);
        let scale = 1.0 + current.ll.abs();
        let finishing = ridge == 0.0 && decrement <= tolerance * scale;
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..40 {
            let candidate: Vec<f64> = theta
                .iter()
                .zip(&delta)
                .map(|(t, d)| t + alpha * d)
                .collect();
            if let Some(t) = model.evaluate(&candidate, false)
                && (t.ll >= current.ll || (finishing && t.ll >= current.ll - 1e-12 * scale))
            {
                theta.copy_from_slice(&candidate);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if accepted {
            current = model.evaluate(theta, true)?;
        }
        if finishing {
            converged = true;
            break;
        }
        if !accepted {
            break;
        }
    }
    Some(Maximum {
        theta: theta.to_vec(),
        totals: current,
        iterations: 0,
        converged,
    })
}

/// The Newton step and the ridge used.
fn step(totals: &Totals, q: usize) -> (Vec<f64>, f64) {
    let largest = (0..q)
        .map(|i| totals.info[i * q + i])
        .fold(0.0_f64, f64::max);
    let mut ridge = 0.0;
    for attempt in 0..14 {
        let mut a = totals.info.clone();
        for i in 0..q {
            a[i * q + i] += ridge * largest.max(1e-300);
        }
        let pivots = ldl::factorise(&mut a, q, 1e-13);
        if pivots.clean && pivots.rank == q {
            let mut y = totals.score.clone();
            ldl::solve(&a, q, &mut y);
            return (y, ridge);
        }
        ridge = if attempt == 0 { 1e-8 } else { ridge * 10.0 };
    }
    // A ridge as large as this makes the step a short gradient step.
    let scale = 1.0 / (largest.max(1e-300) * ridge);
    (totals.score.iter().map(|s| s * scale).collect(), ridge)
}
