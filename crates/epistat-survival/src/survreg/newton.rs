//! The log likelihood of a location-scale model with its score and
//! information, and the Newton-Raphson iteration with step halving.

use super::dist::{Base, row_derivs};
use crate::error::{Error, Result};
use crate::ldl;

/// The rows of a fit, one entry per row used, in order.
pub(super) struct Problem<'a> {
    pub base: Base,
    pub y1: &'a [f64],
    pub y2: &'a [f64],
    pub status: &'a [u8],
    pub weight: &'a [f64],
    /// Dense stratum code of each row, absent for a single scale.
    pub stratum: Option<&'a [u32]>,
    /// The design, row-major, `p` columns.
    pub x: &'a [f64],
    pub p: usize,
    /// The offset of each row, added to the linear predictor.
    pub offset: Option<&'a [f64]>,
    /// The number of scale parameters estimated: the number of strata, or 0
    /// when the scale is fixed.
    pub n_scale: usize,
    /// The log of the fixed scale, when it is fixed.
    pub fixed_ls: Option<f64>,
}

/// The log likelihood at a point, its score `u`, the information matrix
/// (minus the Hessian) and the outer product of the row scores, all over the
/// `q = p + n_scale` parameters, matrices row-major and symmetric.
pub(super) struct Eval {
    pub loglik: f64,
    pub u: Vec<f64>,
    pub imat: Vec<f64>,
    pub jj: Vec<f64>,
}

impl Problem<'_> {
    pub(super) fn q(&self) -> usize {
        self.p + self.n_scale
    }

    /// The log scale of a row and the index of its parameter.
    fn scale_of(&self, beta: &[f64], row: usize) -> (f64, usize) {
        let s = self.stratum.map_or(0, |codes| codes[row] as usize);
        match self.fixed_ls {
            Some(ls) => (ls, s),
            None => (beta[self.p + s], s),
        }
    }

    /// Evaluate at `beta`: the coefficients, then the log scale of each
    /// stratum when it is estimated. Rows are summed in order. A row with no
    /// probability makes the log likelihood `-inf` and ends the evaluation.
    pub(super) fn eval(&self, beta: &[f64]) -> Eval {
        let (p, q) = (self.p, self.q());
        let mut out = Eval {
            loglik: 0.0,
            u: vec![0.0; q],
            imat: vec![0.0; q * q],
            jj: vec![0.0; q * q],
        };
        for row in 0..self.y1.len() {
            let x = &self.x[row * p..(row + 1) * p];
            let mut eta = 0.0;
            for (b, xv) in beta[..p].iter().zip(x) {
                eta += b * xv;
            }
            if let Some(offset) = self.offset {
                eta += offset[row];
            }
            let (ls, s) = self.scale_of(beta, row);
            let d = row_derivs(
                self.base,
                self.status[row],
                self.y1[row],
                self.y2[row],
                eta,
                ls,
            );
            let w = self.weight[row];
            out.loglik += w * d.g;
            if !d.g.is_finite() {
                out.loglik = f64::NEG_INFINITY;
                return out;
            }
            for a in 0..p {
                let xa = x[a];
                out.u[a] += w * d.g_eta * xa;
                for b in 0..=a {
                    out.imat[a * q + b] -= w * d.g_ee * xa * x[b];
                    out.jj[a * q + b] += w * d.g_eta * d.g_eta * xa * x[b];
                }
            }
            if self.n_scale > 0 {
                let k = p + s;
                out.u[k] += w * d.g_ls;
                for a in 0..p {
                    out.imat[k * q + a] -= w * d.g_el * x[a];
                    out.jj[k * q + a] += w * d.g_ls * d.g_eta * x[a];
                }
                out.imat[k * q + k] -= w * d.g_ll;
                out.jj[k * q + k] += w * d.g_ls * d.g_ls;
            }
        }
        for m in [&mut out.imat, &mut out.jj] {
            for a in 0..q {
                for b in 0..a {
                    m[b * q + a] = m[a * q + b];
                }
            }
        }
        out
    }
}

/// Whether the information matrix is positive semi-definite, after
/// factorising it in place as `coxph` does.
pub(super) fn factorise(a: &mut [f64], q: usize, tolerance: f64) -> bool {
    ldl::factorise(a, q, tolerance).clean
}

/// The settings of the iteration.
#[derive(Clone, Copy)]
pub(super) struct Control {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub toler_chol: f64,
}

/// Where the iteration ended.
pub(super) struct Newton {
    pub beta: Vec<f64>,
    pub loglik: f64,
    pub iterations: usize,
    pub converged: bool,
    /// The information matrix at `beta`.
    pub imat: Vec<f64>,
    /// The Newton steps that had to be taken with the outer product of the
    /// row scores because the information matrix was not positive
    /// semi-definite.
    pub outer_product_steps: u32,
}

fn all_finite(eval: &Eval, q: usize) -> bool {
    eval.loglik.is_finite()
        && eval.u.iter().all(|v| v.is_finite())
        && (0..q).all(|i| eval.imat[i * q + i].is_finite())
}

/// The Newton step `I^-1 u`, from the outer product of the row scores when
/// the information is not positive semi-definite.
fn step(eval: &Eval, q: usize, toler: f64, outer: &mut u32) -> Result<Vec<f64>> {
    let mut a = eval.imat.clone();
    if !factorise(&mut a, q, toler) {
        a.copy_from_slice(&eval.jj);
        *outer += 1;
        if !factorise(&mut a, q, toler) {
            return Err(Error::value(
                "the likelihood has no usable derivatives at the starting values or along the \
                 iteration; check the times and the design"
                    .to_owned(),
            ));
        }
    }
    let mut s = eval.u.clone();
    ldl::solve(&a, q, &mut s);
    Ok(s)
}

/// Maximise the log likelihood from `start` by Newton-Raphson with step
/// halving, as `survreg` does.
///
/// A trial point is the last accepted point plus `I^-1 u` from the last
/// accepted point. After a trial the iteration stops when the log likelihood
/// has changed by at most `tolerance`, relatively or absolutely, since the
/// last accepted point and the trial was not a halving. A trial whose log
/// likelihood is lower, or that has a non-finite value or derivative, is moved
/// a third of the way back to the last accepted point (the first time, a log
/// scale that fell by more than 1.1 is moved back to a fall of 1.1). The
/// iteration count is the number of trial points, the first being 1.
pub(super) fn newton(problem: &Problem<'_>, start: Vec<f64>, control: Control) -> Result<Newton> {
    let q = problem.q();
    let toler = control.toler_chol;
    let mut outer = 0_u32;
    let mut beta = start;
    let first = problem.eval(&beta);
    if !all_finite(&first, q) {
        return Err(Error::value(
            "the log likelihood or its derivatives are not finite at the starting values; \
             check the times against the distribution"
                .to_owned(),
        ));
    }
    let mut loglik = first.loglik;
    let first_step = step(&first, q, toler, &mut outer)?;
    let mut newbeta: Vec<f64> = beta.iter().zip(&first_step).map(|(b, s)| b + s).collect();
    let mut current = problem.eval(&newbeta);
    let mut halving = 0_u32;
    for iteration in 1..=control.max_iterations {
        let bad = !all_finite(&current, q);
        let newlk = current.loglik;
        if !bad
            && halving == 0
            && ((1.0 - loglik / newlk).abs() <= control.tolerance
                || (loglik - newlk).abs() <= control.tolerance)
        {
            return Ok(Newton {
                beta: newbeta,
                loglik: newlk,
                iterations: iteration,
                converged: true,
                imat: current.imat,
                outer_product_steps: outer,
            });
        }
        if bad || newlk < loglik {
            halving += 1;
            for (nb, b) in newbeta.iter_mut().zip(&beta) {
                *nb = (*nb + 2.0 * b) / 3.0;
            }
            if halving == 1 {
                for i in problem.p..q {
                    if beta[i] - newbeta[i] > 1.1 {
                        newbeta[i] = beta[i] - 1.1;
                    }
                }
            }
        } else {
            halving = 0;
            loglik = newlk;
            let s = step(&current, q, toler, &mut outer)?;
            beta.copy_from_slice(&newbeta);
            for (nb, d) in newbeta.iter_mut().zip(&s) {
                *nb += d;
            }
        }
        current = problem.eval(&newbeta);
    }
    // Out of iterations: keep the last trial when it is an improvement.
    let newlk = current.loglik;
    let (loglik, imat) = if halving == 0 && newlk.is_finite() && newlk >= loglik {
        beta = newbeta;
        (newlk, current.imat)
    } else {
        (loglik, problem.eval(&beta).imat)
    };
    Ok(Newton {
        beta,
        loglik,
        iterations: control.max_iterations,
        converged: false,
        imat,
        outer_product_steps: outer,
    })
}
