//! The contrast curve of a spline term against a reference value.
//!
//! # Method
//!
//! A model with a spline term has `g(E y) = ... + B(x)' b`, with `B` a
//! [`Basis`] and `b` the coefficients of its functions, and a covariance `V`
//! of `b` from the fit (model-based or robust, as the caller prefers). The
//! contrast of `x` against the reference value `x0` is
//!
//! ```text
//! l(x)  = d(x)' b,      d(x) = B(x) - B(x0)
//! se(x) = sqrt(d(x)' V d(x))
//! ```
//!
//! The intercept and every other term of the model cancel, so the curve is 0
//! with a standard error of 0 at `x0`. This is `rms::Predict` with
//! `ref.zero = TRUE`, and `Epi::ci.lin` with a contrast matrix of the rows
//! `d(x)`. On a log or logit link, the exponential of the curve and its limits
//! is a ratio.
//!
//! [`Band::Pointwise`] gives the limits `l(x) -+ z se(x)` with
//! `z = Phi^-1(1 - alpha / 2)`, each covering its own `l(x)` with probability
//! `1 - alpha`. [`Band::SupT`] gives limits with the sup-t critical value `c`
//! of [`crate::simultaneous::sup_t_from_covariance`], the `1 - alpha` quantile
//! of the largest absolute standardised deviation over the grid, so that the
//! band covers the whole curve over the grid with probability `1 - alpha`. The
//! covariance of the curve over the grid is `D V D'`, with `D` the stacked rows
//! `d(x)`. The reference point has no variance and no deviation, so it does not
//! enter the maximum, and a curve with no other point has `c = z`. The
//! covariance over a grid has at most the rank of `V`, so the replicates are
//! drawn for the coefficients, `b = diag(s) L z` with `L L'` the correlation
//! of `V` and `s` its standard deviations, and mapped to the curve, which gives
//! the same distribution as the `G x G` covariance without factoring it. The
//! coefficients must have positive variances for the band.
//!
//! # Accuracy
//!
//! The values are sums of a few products, so they agree with R to the rounding
//! of the basis and of the covariance that R prints, about `1e-13` relative.
//! The curve takes the basis as it is, outside the knots too: a restricted
//! cubic or natural spline is linear there, which is a modelling assumption
//! and not a result. The uncertainty from choosing the knots by the data is
//! not part of `V`, as in R.
//!
//! # References
//!
//! 1. Harrell FE. Regression Modeling Strategies. 2nd ed. Springer; 2015.
//!    Section 2.4.
//! 2. Montiel Olea JL, Plagborg-Moller M. Simultaneous confidence bands:
//!    theory, implementation, and an application to SVARs. J Appl Econom.
//!    2019;34(1):1-17.

use epistat_linalg::MatRef;
use epistat_linalg::basis::Basis;
use epistat_rng::{Stream, replicate_seed};
use epistat_stats::{lanes, normal, quantile};
use rayon::prelude::*;

use crate::simultaneous::{correlation, lower_cholesky};
use crate::{InferenceError, check_length};

/// How far a covariance matrix may be from symmetric, relative to the
/// geometric mean of its two diagonal entries (see `pooling`).
const SYMMETRY: f64 = 1e-8;

/// The kind of confidence limits of a curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    /// Each point has its own interval of coverage `1 - alpha`.
    Pointwise,
    /// The band covers the curve over the whole grid at once, with a sup-t
    /// critical value from `draws` simulated replicates seeded from `seed`.
    SupT {
        /// The number of simulated replicates.
        draws: usize,
        /// The seed of the replicates.
        seed: u64,
    },
}

/// A contrast curve and its limits.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastCurve {
    /// The grid.
    pub x: Vec<f64>,
    /// The contrast of each grid point against the reference.
    pub estimate: Vec<f64>,
    /// Its standard error.
    pub se: Vec<f64>,
    /// The lower limit.
    pub lower: Vec<f64>,
    /// The upper limit.
    pub upper: Vec<f64>,
    /// The critical value that multiplies the standard errors: the normal
    /// quantile for [`Band::Pointwise`], the sup-t value for [`Band::SupT`].
    pub critical: f64,
}

/// The contrast curve of `basis` on `grid` against `reference`, with the
/// spline coefficients `coefficients` and their covariance block `covariance`
/// in the order of the basis, and limits of the confidence level `1 - alpha`.
///
/// # Errors
///
/// [`InferenceError::Empty`] for an empty grid or no draws,
/// [`InferenceError::Length`] when the coefficients or the covariance do not
/// match the basis, [`InferenceError::Alpha`] for `alpha` outside `(0, 1)`,
/// [`InferenceError::NotSymmetric`] for a covariance that is not symmetric,
/// and [`InferenceError::NotFinite`] for a grid point, the reference, a
/// coefficient or a covariance entry that is not finite, a negative variance,
/// or a curve variance that is negative.
pub fn contrast_curve<B: Basis + ?Sized>(
    basis: &B,
    grid: &[f64],
    reference: f64,
    coefficients: &[f64],
    covariance: MatRef<'_, f64>,
    alpha: f64,
    band: Band,
) -> Result<ContrastCurve, InferenceError> {
    let p = basis.len();
    if grid.is_empty() || p == 0 {
        return Err(InferenceError::Empty);
    }
    check_length("coefficients", p, coefficients.len())?;
    check_length("covariance rows", p, covariance.nrows())?;
    check_length("covariance columns", p, covariance.ncols())?;
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(InferenceError::Alpha(alpha));
    }
    if !reference.is_finite()
        || grid.iter().any(|v| !v.is_finite())
        || coefficients.iter().any(|v| !v.is_finite())
    {
        return Err(InferenceError::NotFinite);
    }
    // The symmetrised covariance, row-major.
    let mut v = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            let (a, b) = (covariance[(i, j)], covariance[(j, i)]);
            if !a.is_finite() || !b.is_finite() {
                return Err(InferenceError::NotFinite);
            }
            v[i * p + j] = a.midpoint(b);
        }
    }
    for i in 0..p {
        if v[i * p + i] < 0.0 {
            return Err(InferenceError::NotFinite);
        }
        for j in 0..i {
            let scale = (v[i * p + i] * v[j * p + j]).sqrt();
            if (covariance[(i, j)] - covariance[(j, i)]).abs() > SYMMETRY * scale {
                return Err(InferenceError::NotSymmetric);
            }
        }
    }

    let at_reference = basis.values(reference);
    let g = grid.len();
    let mut rows = vec![0.0; g * p];
    let mut estimate = Vec::with_capacity(g);
    let mut vd = vec![0.0; p];
    let mut variance = Vec::with_capacity(g);
    for (r, &x) in grid.iter().enumerate() {
        let row = &mut rows[r * p..(r + 1) * p];
        basis.values_into(x, row);
        for (d, b0) in row.iter_mut().zip(&at_reference) {
            *d -= b0;
        }
        let mut e = 0.0;
        for (d, b) in row.iter().zip(coefficients) {
            e += d * b;
        }
        for (i, slot) in vd.iter_mut().enumerate() {
            let mut s = 0.0;
            for (j, d) in row.iter().enumerate() {
                s += v[i * p + j] * d;
            }
            *slot = s;
        }
        let mut var = 0.0;
        for (d, w) in row.iter().zip(&vd) {
            var += d * w;
        }
        if !(e.is_finite() && var.is_finite()) || var < 0.0 {
            return Err(InferenceError::NotFinite);
        }
        estimate.push(e);
        variance.push(var);
    }
    let se: Vec<f64> = variance.iter().map(|v| v.sqrt()).collect();

    let z = normal::quantile(1.0 - alpha / 2.0);
    let critical = match band {
        Band::Pointwise => z,
        Band::SupT { draws, seed } => {
            // The points with a variance; the reference has none.
            let free: Vec<usize> = (0..g).filter(|&r| variance[r] > 0.0).collect();
            if free.is_empty() {
                z
            } else {
                sup_t(&rows, &free, &se, &v, p, alpha, draws, seed)?
            }
        }
    };
    let lower = estimate
        .iter()
        .zip(&se)
        .map(|(e, s)| e - critical * s)
        .collect();
    let upper = estimate
        .iter()
        .zip(&se)
        .map(|(e, s)| e + critical * s)
        .collect();
    Ok(ContrastCurve {
        x: grid.to_vec(),
        estimate,
        se,
        lower,
        upper,
        critical,
    })
}

/// The `1 - alpha` quantile of the maximum over the points `free` of
/// `|d' beta| / se`, with `beta` normal with covariance `v` (`p x p`). With
/// the correlation `C = L L'` of `v` and its standard deviations `s`, `beta`
/// is `diag(s) L z` for `z` standard normal in `p` dimensions, so the matrix
/// `A = diag(1 / se) D diag(s) L` has the curve's deviations `A z`. The curve
/// has at most rank `p`, which a factor of its own covariance over the grid
/// could not express.
#[allow(clippy::too_many_arguments)]
fn sup_t(
    rows: &[f64],
    free: &[usize],
    se: &[f64],
    v: &[f64],
    p: usize,
    alpha: f64,
    draws: usize,
    seed: u64,
) -> Result<f64, InferenceError> {
    if draws == 0 {
        return Err(InferenceError::Empty);
    }
    let covariance = epistat_linalg::Mat::from_fn(p, p, |i, j| v[i * p + j]);
    let lower = lower_cholesky(&correlation(covariance.as_ref())?, p)?;
    let sd: Vec<f64> = (0..p).map(|i| v[i * p + i].sqrt()).collect();
    let m = free.len();
    let mut a = vec![0.0; m * p];
    for (k, &r) in free.iter().enumerate() {
        let d = &rows[r * p..(r + 1) * p];
        for c in 0..p {
            let mut s = 0.0;
            for i in c..p {
                s += d[i] * sd[i] * lower[i * p + c];
            }
            a[k * p + c] = s / se[r];
        }
    }
    let mut maxima: Vec<f64> = (0..draws as u64)
        .into_par_iter()
        .map_init(
            || vec![0.0; p],
            |z, r| {
                Stream::seeded(replicate_seed(seed, r)).fill_normals(z);
                (0..m)
                    .map(|k| lanes::dot(&a[k * p..(k + 1) * p], z).abs())
                    .fold(0.0, f64::max)
            },
        )
        .collect();
    quantile::type7_unsorted(&mut maxima, 1.0 - alpha).ok_or(InferenceError::NotFinite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::basis::Polynomial;
    use epistat_linalg::mat;

    /// A straight line `b1 x` through a basis `1, x`: the contrast is
    /// `b1 (x - x0)` with the standard error `|x - x0| sqrt(v11)`, where the
    /// intercept and its covariances cancel.
    #[test]
    fn a_line_has_the_contrast_of_its_slope() {
        let basis = Polynomial { degree: 1 };
        let cov = mat![[4.0, 0.5], [0.5, 0.25]];
        let curve = contrast_curve(
            &basis,
            &[1.0, 3.0, 5.0],
            3.0,
            &[7.0, 2.0],
            cov.as_ref(),
            0.05,
            Band::Pointwise,
        )
        .unwrap();
        assert_eq!(curve.estimate, vec![-4.0, 0.0, 4.0]);
        assert_eq!(curve.se, vec![1.0, 0.0, 1.0]);
        assert_eq!((curve.lower[1], curve.upper[1]), (0.0, 0.0));
        assert!((curve.critical - 1.959_963_984_540_054).abs() < 1e-14);
    }
}
