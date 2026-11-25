//! The first stage shared by the imputation and two-stage estimators: unit
//! and period fixed effects and covariates fitted on the untreated rows, the
//! imputed untreated outcomes, and the projections that carry the first
//! stage's estimation error into the variance.
//!
//! # Projections
//!
//! Both estimators need, for a vector `b` of sums over the regressors, the
//! fitted values `h = Z0 M0^-1 b` on the untreated rows, with `Z0 = [X0 D0]`
//! the covariates and the fixed-effect dummies of those rows and
//! `M0 = Z0' W Z0`. Here `b = Z0' W base + Z1' c`: a column `base` on the
//! untreated rows and a vector `c` on the treated rows. `M0` is not formed.
//! The level sums `b_D = D0' W base + D1' c` of every factor are solved for
//! coefficients `alpha` with `D0' W D0 alpha = b_D` by
//! [`FixedEffects::solve_sums`], the alternating projections of
//! [`FixedEffects::demean`] run on the sums, and `f_D = D0 alpha`
//! ([`FixedEffects::fitted`]) is unique although `alpha` is not. The sums lie
//! in the range of `D0' W D0` when every treated row's levels are connected
//! to untreated rows, which is what makes the row imputable. Nothing in this
//! depends on the number of factors.
//!
//! With `X~0` the demeaned covariates and `B = (X~0' W X~0)^-1`, the block
//! solution of the normal equations is
//!
//! ```text
//! gamma_X = B (X0' W base + X1' c - X0' W f_D),      h = f_D + X~0 gamma_X.
//! ```
//!
//! `f_D` is a sum of fixed effects by construction, so `h` lies in the span
//! of `Z0` to rounding, and the iteration's error appears only in the normal
//! equations `Z0' W h = b`, which bound the distance to the exact projection
//! (the tests check that bound).

use epistat_glm::{Absorb, AbsorbError, FeolsFit, FixedEffects};
use epistat_linalg::MatRef;
use rayon::prelude::*;

/// The rows of a staggered panel after the first stage.
pub(crate) struct FirstStage {
    /// Whether the row is treated, `t >= g`.
    pub(crate) treated: Vec<bool>,
    /// `y - y_hat(0)`, `None` for rows whose untreated outcome cannot be
    /// imputed.
    pub(crate) adjusted: Vec<Option<f64>>,
    /// The prior weight of each row, 1 without weights.
    pub(crate) weight: Vec<f64>,
    /// The kept covariates' coefficients, `None` for aliased covariates.
    pub(crate) coefficients: Vec<Option<f64>>,
}

/// The compact level of each caller code of a factor.
fn level_map(labels: &[u32]) -> Vec<u32> {
    let size = labels.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut map = vec![u32::MAX; size];
    for (level, &code) in labels.iter().enumerate() {
        map[code as usize] = u32::try_from(level).unwrap_or(u32::MAX);
    }
    map
}

impl FirstStage {
    /// The imputed outcomes from a first-stage fit.
    pub(crate) fn new(
        unit: &[u32],
        time: &[u32],
        treated: Vec<bool>,
        outcome: &[f64],
        weight: Vec<f64>,
        covariates: MatRef<'_, f64>,
        fit: &FeolsFit<'_>,
    ) -> Self {
        let values = fit.fixed_effects();
        let coefficients = fit.coefficients.clone();
        let p = covariates.ncols();
        let adjusted: Vec<Option<f64>> = (0..unit.len())
            .into_par_iter()
            .map(|i| {
                let mut fitted = values.sum(&[unit[i], time[i]])?;
                for j in 0..p {
                    if let Some(b) = coefficients[j] {
                        fitted += b * covariates[(i, j)];
                    }
                }
                Some(outcome[i] - fitted)
            })
            .collect();
        Self {
            treated,
            adjusted,
            weight,
            coefficients,
        }
    }
}

/// One projection: a column on the untreated rows and a vector on the
/// treated rows, each with one value per input row.
pub(crate) struct Projection<'a> {
    /// Read on the rows of the fit; `None` for zero.
    pub(crate) base: Option<&'a [f64]>,
    /// Read on the treated rows that can be imputed.
    pub(crate) treated: &'a [f64],
}

/// The projections of the first stage.
pub(crate) struct Projector<'a> {
    fe: &'a FixedEffects,
    fit: &'a FeolsFit<'a>,
    covariates: MatRef<'a, f64>,
    /// The caller's codes of each factor, one per input row.
    factors: Vec<&'a [u32]>,
    /// The level in the fit of each caller code, by factor.
    level: Vec<Vec<u32>>,
}

impl<'a> Projector<'a> {
    pub(crate) fn new(
        fe: &'a FixedEffects,
        fit: &'a FeolsFit<'a>,
        covariates: MatRef<'a, f64>,
        factors: &[&'a [u32]],
    ) -> Self {
        Self {
            fe,
            fit,
            covariates,
            factors: factors.to_vec(),
            level: (0..fe.n_factors())
                .map(|q| level_map(fe.labels(q)))
                .collect(),
        }
    }

    /// `h = Z0 M0^-1 (Z0' W base + Z1' c)` on the rows of the fit
    /// ([`FixedEffects::rows`]), for the treated rows `rows` that can be
    /// imputed.
    pub(crate) fn project(
        &self,
        target: &Projection<'_>,
        rows: &[usize],
        control: &Absorb,
    ) -> Result<Vec<f64>, AbsorbError> {
        let fe = self.fe;
        let fit_rows = fe.rows();
        let weights = fe.weights();
        let w = |k: usize| weights.map_or(1.0, |w| w[k]);
        let kept = self.fit.kept_columns();
        let r = kept.len();

        // The level sums D0' W base + D1' c, and the treated rows' covariate
        // sums X1' c.
        let mut sums: Vec<Vec<f64>> = fe.levels().iter().map(|&g| vec![0.0; g]).collect();
        if let Some(base) = target.base {
            for (k, &i) in fit_rows.iter().enumerate() {
                let value = w(k) * base[i];
                for (q, s) in sums.iter_mut().enumerate() {
                    s[fe.codes(q)[k] as usize] += value;
                }
            }
        }
        let mut b_x = vec![0.0; r];
        for &i in rows {
            let c = target.treated[i];
            if c == 0.0 {
                continue;
            }
            for (q, s) in sums.iter_mut().enumerate() {
                s[self.level[q][self.factors[q][i] as usize] as usize] += c;
            }
            for (a, &j) in kept.iter().enumerate() {
                b_x[a] += c * self.covariates[(i, j)];
            }
        }
        let refs: Vec<&[f64]> = sums.iter().map(Vec::as_slice).collect();
        let alpha = fe.solve_sums(&refs, control)?;
        let f_d = fe.fitted(&alpha);
        let mut h = f_d;
        if r > 0 {
            // gamma_X = B (X0' W base + X1' c - X0' W f_D)
            let mut rhs = b_x;
            for (k, &i) in fit_rows.iter().enumerate() {
                let base = target.base.map_or(0.0, |b| b[i]);
                let d = w(k) * (base - h[k]);
                for (a, &j) in kept.iter().enumerate() {
                    rhs[a] += d * self.covariates[(i, j)];
                }
            }
            let bread = self.fit.bread();
            let gamma: Vec<f64> = (0..r)
                .map(|a| {
                    let mut total = 0.0;
                    for (b, value) in rhs.iter().enumerate() {
                        total += bread[(a, b)] * value;
                    }
                    total
                })
                .collect();
            let design = self.fit.design();
            for (k, value) in h.iter_mut().enumerate() {
                let mut total = 0.0;
                for (a, g) in gamma.iter().enumerate() {
                    total += design[(k, a)] * g;
                }
                *value += total;
            }
        }
        Ok(h)
    }
}
