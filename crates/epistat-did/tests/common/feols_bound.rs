//! The tolerances of a comparison with `fixest` for a linear model with two
//! absorbed factors, derived as in `parity_twfe.rs`, for any design.
//!
//! Each side's demeaned columns are off by at most `e` times the column's
//! root mean square per row: `fixest`'s measured error (from the exact
//! projection in R) and this side's bound from its criterion. At the returned
//! point the level means of the residual are at most `tol s`, so the level
//! sums `gamma` have `||gamma|| <= tol s sqrt(sum W_qg^2)`, the error has
//! `||e||_W <= ||gamma|| / sqrt(lambda_+)` (epistat-glm's absorb module) and
//! each row's error is at most `||e||_W / sqrt(min w)`. The residual itself is
//! a sum of three terms of the size of the centred column, which adds 16 eps
//! of its largest value. The coefficients move at second order in `e`, with
//! the rounding of two least-squares solutions, `100 kappa eps`; the clustered
//! covariance moves at first order: `|dV| <= |B| |dM| |B|` times the
//! small-sample factors, with `|dM|` bounded from the score sums and the
//! errors of the scores, and rounding of 100 eps of the largest entry.

use epistat_glm::{Absorb, Feols, FixedEffects, Singletons, Ssc, Vcov};
use epistat_inference::Clusters;
use epistat_linalg::Mat;

pub const EPS: f64 = f64::EPSILON;

/// The covariance whose error is bounded.
#[derive(Clone, Copy)]
pub enum Meat<'a> {
    /// Clustered by one code per input row.
    Cluster(&'a [u32]),
    /// Homoskedastic, `rss / (n - K) B`: the error of `rss` is at most
    /// `2 sum w |r| d` with `d` the error of a residual, and `B` moves at
    /// second order.
    Iid,
}

/// The bounds of one fit.
pub struct FitBound {
    /// The coefficients of the plain fit, `None` for aliased columns.
    pub coefficients: Vec<Option<f64>>,
    /// The columns that are not aliased.
    pub kept: Vec<usize>,
    /// The bound on each coefficient's difference.
    pub beta: f64,
    /// The bound on each entry of the clustered covariance over `kept`,
    /// row-major.
    pub covariance: Vec<f64>,
}

/// The fit of `outcome` on `design` with the factors `factors` and the
/// bounds of its comparison with `fixest`, whose demeaning error is
/// `fixest_error` and whose `D'WD` has the smallest positive eigenvalue
/// `lambda_plus`. `factor` bounds the small-sample factors of the clustered
/// covariance, or is `1 / (n - K)` for the iid one.
#[expect(clippy::too_many_arguments, reason = "the inputs of one fit")]
pub fn feols_bound(
    factors: &[&[u32]],
    weights: Option<&[f64]>,
    design: &Mat<f64>,
    outcome: &[f64],
    covariance_kind: Meat<'_>,
    absorb: Absorb,
    lambda_plus: f64,
    fixest_error: f64,
    factor: f64,
) -> FitBound {
    let fe = FixedEffects::new(factors, weights, Singletons::Remove).unwrap();
    let fit = Feols { absorb }.fit(design.as_ref(), outcome, &fe).unwrap();
    let rows = fe.rows();
    let m = rows.len();
    let w: Vec<f64> = (0..m).map(|k| fe.weights().map_or(1.0, |w| w[k])).collect();
    let tw: f64 = w.iter().sum();
    let mean_of = |v: &dyn Fn(usize) -> f64| (0..m).map(|k| w[k] * v(rows[k])).sum::<f64>() / tw;
    let rms = |v: &dyn Fn(usize) -> f64| {
        let mean = mean_of(v);
        ((0..m)
            .map(|k| w[k] * (v(rows[k]) - mean).powi(2))
            .sum::<f64>()
            / tw)
            .sqrt()
    };
    let spread = |v: &dyn Fn(usize) -> f64, s: f64| {
        let mean = mean_of(v);
        (0..m)
            .map(|k| (v(rows[k]) - mean).abs())
            .fold(0.0, f64::max)
            / s
    };
    let kept = fit.kept_columns().to_vec();
    let k = kept.len();
    let rms_x: Vec<f64> = kept.iter().map(|&j| rms(&|i| design[(i, j)])).collect();
    let rms_y = rms(&|i| outcome[i]);
    let ratio = kept
        .iter()
        .zip(&rms_x)
        .map(|(&j, &r)| spread(&|i| design[(i, j)], r))
        .fold(spread(&|i| outcome[i], rms_y), f64::max);
    let mut sum_w2 = 0.0;
    for q in 0..fe.n_factors() {
        let mut level = vec![0.0; fe.levels()[q]];
        for (kk, &c) in fe.codes(q).iter().enumerate() {
            level[c as usize] += w[kk];
        }
        sum_w2 += level.iter().map(|v| v * v).sum::<f64>();
    }
    let w_min = w.iter().copied().fold(f64::MAX, f64::min);
    let mine = absorb.tolerance * sum_w2.sqrt() / (lambda_plus * w_min).sqrt() + 16.0 * EPS * ratio;
    let e = fixest_error + mine;
    let delta_e = e
        * (rms_y
            + kept
                .iter()
                .zip(&rms_x)
                .map(|(&j, r)| fit.coefficients[j].unwrap().abs() * r)
                .sum::<f64>());
    let d = fit.design();
    let bread = fit.bread();
    let clusters = match covariance_kind {
        Meat::Cluster(c) => c,
        Meat::Iid => {
            let cov = fit.vcov(&fe, Vcov::Iid, Ssc::new()).unwrap();
            let largest = cov
                .matrix
                .as_slice()
                .iter()
                .flatten()
                .fold(0.0_f64, |a, &b| a.max(b.abs()));
            let drss: f64 = (0..m)
                .map(|r| 2.0 * w[r] * fit.residuals[r].abs() * delta_e)
                .sum();
            let covariance = (0..k * k)
                .map(|ab| {
                    2.0 * factor * drss * bread[(ab / k, ab % k)].abs() + 100.0 * EPS * largest
                })
                .collect();
            return finish(&fit, kept, covariance, e, &w, d);
        }
    };
    let fit_clusters: Vec<u32> = rows.iter().map(|&i| clusters[i]).collect();
    let of = Clusters::from_labels_first_seen(&fit_clusters);
    let g = of.count();
    let mut s = vec![0.0; g * k];
    let mut u = vec![0.0; g * k];
    for r in 0..m {
        let c = of.of()[r];
        for a in 0..k {
            s[c * k + a] += w[r] * fit.residuals[r] * d[(r, a)];
            u[c * k + a] +=
                w[r] * (fit.residuals[r].abs() * e * rms_x[a] + d[(r, a)].abs() * delta_e);
        }
    }
    let mut dm = vec![0.0; k * k];
    for c in 0..g {
        for a in 0..k {
            for b in 0..k {
                dm[a * k + b] +=
                    s[c * k + a].abs() * u[c * k + b] + u[c * k + a] * s[c * k + b].abs();
            }
        }
    }
    let ways = [clusters];
    let cov = fit.vcov(&fe, Vcov::Cluster(&ways), Ssc::new()).unwrap();
    let largest = cov
        .matrix
        .as_slice()
        .iter()
        .flatten()
        .fold(0.0_f64, |a, &b| a.max(b.abs()));
    // |B| |dM| |B|, as two products.
    let mut half = vec![0.0; k * k];
    for c in 0..k {
        for b in 0..k {
            half[c * k + b] = (0..k).map(|dd| dm[c * k + dd] * bread[(dd, b)].abs()).sum();
        }
    }
    let mut covariance = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            let total: f64 = (0..k).map(|c| bread[(a, c)].abs() * half[c * k + b]).sum();
            covariance[a * k + b] = 2.0 * factor * total + 100.0 * EPS * largest;
        }
    }
    finish(&fit, kept, covariance, e, &w, d)
}

/// The coefficient bound from the condition number of the demeaned design.
fn finish(
    fit: &epistat_glm::FeolsFit,
    kept: Vec<usize>,
    covariance: Vec<f64>,
    e: f64,
    w: &[f64],
    d: epistat_linalg::MatRef<'_, f64>,
) -> FitBound {
    let k = kept.len();
    let m = w.len();
    // The condition number of the Gram matrix of the demeaned design with
    // its columns scaled to unit norm.
    let gram: Vec<f64> = (0..k * k)
        .map(|ab| (0..m).map(|r| w[r] * d[(r, ab / k)] * d[(r, ab % k)]).sum())
        .collect();
    let diag: Vec<f64> = (0..k).map(|a| gram[a * k + a].sqrt()).collect();
    let scaled: Vec<f64> = (0..k * k)
        .map(|ab| gram[ab] / (diag[ab / k] * diag[ab % k]))
        .collect();
    let (values, _) = epistat_linalg::eigen::symmetric_eigen_jacobi(&scaled, k);
    let kappa = values.iter().copied().fold(f64::MIN, f64::max)
        / values.iter().copied().fold(f64::MAX, f64::min);
    let largest_beta = fit
        .coefficients
        .iter()
        .flatten()
        .fold(0.0_f64, |a, &b| a.max(b.abs()));
    FitBound {
        coefficients: fit.coefficients.clone(),
        kept,
        beta: (100.0 * EPS + e * e) * kappa * largest_beta,
        covariance,
    }
}
