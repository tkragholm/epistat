//! The row scores of the fit times its variance, which are the dfbeta
//! residuals, and the robust variance built from them.

use super::dist::{Base, row_derivs};

/// What one row needs to be scored.
pub(super) struct RowInput<'a> {
    pub base: Base,
    pub status: u8,
    pub y1: f64,
    pub y2: f64,
    /// The linear predictor of the row, offset included.
    pub eta: f64,
    /// The log scale of the row's stratum.
    pub ls: f64,
    /// The index of the row's stratum.
    pub stratum: usize,
    /// The row of the design.
    pub x: &'a [f64],
    /// The number of strata whose scale is estimated, 0 for a fixed scale.
    pub n_scale: usize,
}

/// The score of a row, over the coefficients and the log scales, times the
/// variance `v` (`q * q`, row-major): the change in the estimates that
/// dropping the row would make to first order. The row's score for
/// coefficient `a` is `g_eta x_a` and for the log scale of its stratum
/// `g_ls`. `out` has `q` entries and is written whole, with NaN for a row
/// that has no probability under the fit.
pub(super) fn dfbeta_row(row: &RowInput<'_>, v: &[f64], out: &mut [f64]) {
    let p = row.x.len();
    let q = p + row.n_scale;
    let d = row_derivs(row.base, row.status, row.y1, row.y2, row.eta, row.ls);
    if !d.g.is_finite() {
        out[..q].fill(f64::NAN);
        return;
    }
    let score = |a: usize| -> f64 {
        if a < p {
            d.g_eta * row.x[a]
        } else if a == p + row.stratum {
            d.g_ls
        } else {
            0.0
        }
    };
    for (b, o) in out.iter_mut().enumerate().take(q) {
        let mut sum = 0.0;
        for a in 0..q {
            sum += score(a) * v[a * q + b];
        }
        *o = sum;
    }
}
