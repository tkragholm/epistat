//! Shared by the distribution and coverage tests: statistics of imputed values
//! per data set and comparisons within Monte Carlo error.

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "each test program uses part of this module; indices are small whole numbers"
)]

use epistat_glm::{Family, Glm, Link};
use epistat_impute::Imputed;
use epistat_linalg::Mat;
use epistat_testkit::assert_within_standard_errors;

/// R's default quantile (type 7).
pub fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

pub fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)
}

/// Per data set: mean, variance, and the 0.1, 0.5 and 0.9 quantiles.
pub fn numeric_stats(imputed: &Imputed, column: usize) -> Vec<Vec<f64>> {
    imputed.values[column]
        .iter()
        .map(|cells| {
            let mut s = cells.clone();
            s.sort_by(f64::total_cmp);
            vec![
                mean(cells),
                variance(cells),
                quantile7(&s, 0.1),
                quantile7(&s, 0.5),
                quantile7(&s, 0.9),
            ]
        })
        .collect()
}

/// Per data set: the share of each of `levels` codes after the first, or the
/// mean for a binary column (`levels` 2 gives one entry, the share of 1s).
pub fn share_stats(imputed: &Imputed, column: usize, levels: usize) -> Vec<Vec<f64>> {
    imputed.values[column]
        .iter()
        .map(|cells| {
            (1..levels)
                .map(|l| {
                    cells.iter().filter(|&&v| v == l as f64).count() as f64 / cells.len() as f64
                })
                .collect()
        })
        .collect()
}

pub fn column_of(rows: &[Vec<f64>], k: usize) -> Vec<f64> {
    rows.iter().map(|r| r[k]).collect()
}

/// Each column of `a` against the same column of `b`: the means over data
/// sets differ by at most `k` combined Monte Carlo standard errors, taken from
/// the spread over the data sets.
#[track_caller]
pub fn compare_stats(label: &str, a: &[Vec<f64>], b: &[Vec<f64>], k: f64) {
    for c in 0..a[0].len() {
        let (xa, xb) = (column_of(a, c), column_of(b, c));
        let se = |x: &[f64]| (variance(x) / x.len() as f64).sqrt();
        eprintln!(
            "{label} stat {c}: epistat {:.5} (se {:.5}) mice {:.5} (se {:.5})",
            mean(&xa),
            se(&xa),
            mean(&xb),
            se(&xb)
        );
        assert_within_standard_errors(mean(&xa), se(&xa), mean(&xb), se(&xb), k);
    }
}

/// Pooled quantities of one coefficient from its per-data-set estimates and
/// variances: the mean estimate, the mean within variance, the between
/// variance and the fraction of missing information, with Monte Carlo
/// standard errors.
pub struct PooledStats {
    pub q: (f64, f64),
    pub u: (f64, f64),
    pub b: (f64, f64),
    pub fmi: (f64, f64),
}

pub fn pooled_stats(estimates: &[f64], variances: &[f64]) -> PooledStats {
    let m = estimates.len() as f64;
    let q = mean(estimates);
    let u = mean(variances);
    let b = variance(estimates);
    let se_q = (b / m).sqrt();
    let se_u = (variance(variances) / m).sqrt();
    // The sample variance of m values has relative standard error sqrt(2 / (m - 1))
    // when they are normal.
    let se_b = b * (2.0 / (m - 1.0)).sqrt();
    let infl = 1.0 + 1.0 / m;
    let t = u + infl * b;
    let fmi = infl * b / t;
    let d_b = infl * u / (t * t);
    let d_u = -infl * b / (t * t);
    let se_fmi = ((d_b * se_b).powi(2) + (d_u * se_u).powi(2)).sqrt();
    PooledStats {
        q: (q, se_q),
        u: (u, se_u),
        b: (b, se_b),
        fmi: (fmi, se_fmi),
    }
}

#[track_caller]
pub fn compare_pooled(label: &str, a: &PooledStats, b: &PooledStats, k: f64) {
    for (name, x, y) in [
        ("estimate", &a.q, &b.q),
        ("within", &a.u, &b.u),
        ("between", &a.b, &b.b),
        ("fmi", &a.fmi, &b.fmi),
    ] {
        eprintln!(
            "{label} {name}: epistat {:.6} (se {:.6}) mice {:.6} (se {:.6})",
            x.0, x.1, y.0, y.1
        );
        assert_within_standard_errors(x.0, x.1, y.0, y.1, k);
    }
}

/// Least squares of `y` on an intercept and `columns`: the coefficients and
/// their variances.
pub fn lm(columns: &[&[f64]], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let x = Mat::from_fn(n, columns.len() + 1, |i, k| {
        if k == 0 { 1.0 } else { columns[k - 1][i] }
    });
    let fit = Glm::new(Family::Gaussian, Link::Identity)
        .fit(x.as_ref(), y, None, None, None)
        .unwrap();
    let p = columns.len() + 1;
    let coef = fit.coefficients.iter().map(|c| c.unwrap()).collect();
    let var = (0..p).map(|i| fit.covariance[i * p + i].unwrap()).collect();
    (coef, var)
}
