//! The design of the linear predictor and of its derivative in log time.
//!
//! The parameters are ordered as the intercept `gamma_0`, the `df` baseline
//! spline coefficients `gamma_1, ...`, one coefficient for each kept column of
//! `x`, and then, for each time-varying term in order, the coefficients of the
//! spline that multiplies its column. `fill` writes the row `z` of the design
//! at a log time, so that `eta = z' theta`, and its derivative `w = dz/du`, so
//! that `eta'(u) = w' theta`.

use epistat_linalg::MatRef;
use epistat_linalg::basis::{Basis, RestrictedCubic};

use crate::error::{Error, Result};

/// A spline in log time without an intercept: `u` itself and, for three knots
/// or more, the restricted cubic functions of Harrell's basis on the knots.
#[derive(Clone, Debug)]
pub(super) struct Spline {
    knots: Vec<f64>,
    cubic: Option<RestrictedCubic>,
}

impl Spline {
    /// The spline on `knots` (boundary knots included), which has
    /// `knots.len() - 1` functions.
    pub(super) fn new(knots: &[f64]) -> Result<Self> {
        if knots.len() < 2 {
            return Err(Error::input("a spline needs at least two knots"));
        }
        if knots.iter().any(|k| !k.is_finite()) || !knots.windows(2).all(|p| p[0] < p[1]) {
            return Err(Error::input(
                "the knots of the spline in log time must be finite and strictly increasing, \
                 which fails when the event times are too few or too tied for the degrees of freedom",
            ));
        }
        let cubic = if knots.len() >= 3 {
            Some(
                RestrictedCubic::new(knots, true)
                    .map_err(|e| Error::input(format!("knots of the spline: {e:?}")))?,
            )
        } else {
            None
        };
        Ok(Self {
            knots: knots.to_vec(),
            cubic,
        })
    }

    pub(super) fn knots(&self) -> &[f64] {
        &self.knots
    }

    pub(super) fn len(&self) -> usize {
        self.knots.len() - 1
    }

    pub(super) fn values_into(&self, u: f64, out: &mut [f64]) {
        match &self.cubic {
            Some(c) => c.values_into(u, out),
            None => out[0] = u,
        }
    }

    pub(super) fn derivatives_into(&self, u: f64, out: &mut [f64]) {
        match &self.cubic {
            Some(c) => c.derivatives_into(u, out),
            None => out[0] = 1.0,
        }
    }
}

/// One time-varying effect: the column of `x` and the spline that multiplies it.
#[derive(Clone, Debug)]
pub(super) struct TvcTerm {
    pub column: usize,
    pub spline: Spline,
}

/// The layout of the parameters and the rule that builds a design row.
#[derive(Clone, Debug)]
pub(super) struct Design {
    pub baseline: Spline,
    /// The columns of `x` that are estimated, ascending.
    pub kept: Vec<usize>,
    pub tvc: Vec<TvcTerm>,
    /// The number of columns of `x`.
    pub n_columns: usize,
}

impl Design {
    /// The number of parameters.
    pub(super) fn dim(&self) -> usize {
        1 + self.baseline.len()
            + self.kept.len()
            + self.tvc.iter().map(|t| t.spline.len()).sum::<usize>()
    }

    /// The position of the first coefficient of `x`.
    pub(super) fn x_start(&self) -> usize {
        1 + self.baseline.len()
    }

    /// The position of the first coefficient of the time-varying term `j`.
    pub(super) fn tvc_start(&self, j: usize) -> usize {
        self.x_start()
            + self.kept.len()
            + self.tvc[..j].iter().map(|t| t.spline.len()).sum::<usize>()
    }

    /// Writes the design row of row `row` of `x` at log time `u` into `z`, and
    /// its derivative in `u` into `w` when given.
    pub(super) fn fill(
        &self,
        x: MatRef<'_, f64>,
        row: usize,
        u: f64,
        z: &mut [f64],
        mut w: Option<&mut [f64]>,
    ) {
        let df = self.baseline.len();
        z[0] = 1.0;
        self.baseline.values_into(u, &mut z[1..=df]);
        if let Some(w) = w.as_deref_mut() {
            w[0] = 0.0;
            self.baseline.derivatives_into(u, &mut w[1..=df]);
        }
        let xs = self.x_start();
        for (slot, &col) in self.kept.iter().enumerate() {
            z[xs + slot] = x[(row, col)];
            if let Some(w) = w.as_deref_mut() {
                w[xs + slot] = 0.0;
            }
        }
        let mut at = xs + self.kept.len();
        for term in &self.tvc {
            let m = term.spline.len();
            let value = x[(row, term.column)];
            term.spline.values_into(u, &mut z[at..at + m]);
            z[at..at + m].iter_mut().for_each(|v| *v *= value);
            if let Some(w) = w.as_deref_mut() {
                term.spline.derivatives_into(u, &mut w[at..at + m]);
                w[at..at + m].iter_mut().for_each(|v| *v *= value);
            }
            at += m;
        }
    }
}
