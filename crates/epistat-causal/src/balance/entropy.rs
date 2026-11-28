//! Entropy balancing.
//!
//! # Algorithm
//!
//! Take the groups to weight: the controls for the effect on the treated (ATT),
//! the treated for the effect on the controls (ATC), both for the effect in the
//! population (ATE). The target is the treated, the controls or everyone. With
//! sample weights `s_i`, base weights `q_i` and balance columns `c_i`
//! (the covariates, and their higher moments and products if asked for), the
//! weights of a group `g` solve
//!
//! ```text
//! min  sum_{i in g} s_i w_i log(w_i / q_i)   subject to
//! sum_{i in g} s_i w_i = sum_{i in T} s_i,   sum_{i in g} s_i w_i c_i = sum_{i in T} s_i c_i
//! ```
//!
//! (Hainmueller 2012), where `T` is the target. With `u_i = (c_i - m) / sd` for
//! the mean `m` and standard deviation `sd` of each column in the target, the
//! solution is `w_i = kappa q_i exp(lambda' u_i)`, where `lambda` minimises the
//! convex function
//!
//! ```text
//! F(lambda) = log sum_{i in g} s_i q_i exp(lambda' u_i)
//! ```
//!
//! and `kappa` makes the weights sum to the size of the target. The gradient of
//! `F` is the weighted mean of `u` in `g`, which is the standardised difference
//! from the target, and the Hessian is its weighted covariance. They are
//! computed in blocks of rows that do not depend on the number of threads.
//!
//! Damped Newton steps `-H^-1 grad` are taken from `lambda = 0`, with the step
//! halved until `F` falls by the Armijo amount (`c = 1e-4`) plus eight units of
//! rounding. Newton's method is invariant to the scaling of the columns, so the
//! standardisation serves the stopping rule: the iteration stops when the
//! largest standardised difference of the weighted means from the target is at
//! most [`EntropyOptions::tolerance`]. The balance is therefore exact to that
//! tolerance by construction, and it is recomputed from the returned weights and
//! reported as [`EntropyBalance::max_imbalance`].
//!
//! # When balance is not possible
//!
//! The weights can match the target only if it lies in the interior of the convex
//! hull of the group's columns. A column whose target mean is outside the range
//! of the group, or on its edge, is reported before the iteration with its
//! index. Otherwise the multipliers diverge: `F` has no minimum. The iteration
//! returns [`Error::Infeasible`] when a multiplier exceeds 200 standardised
//! units, where the weights are in effect on one person. That bound also catches
//! a target so close to the boundary that no weight vector with reasonable
//! effective sample size balances it. A column that does not vary in the target,
//! or that is a linear combination of the others in the group, is an
//! error ([`Error::Invalid`] or [`Error::Aliased`]), not dropped. No weights are returned from a problem that
//! was not solved.
//!
//! # Variance
//!
//! The multipliers solve `sum psi_i = 0` with `psi_i = s_i b_i (1[i in T] - 1[i in
//! g] w_i)`, `b_i = (1, u_i)`, where the constant in `b_i` is the normalisation.
//! With `G_g = sum_{i in g} s_i w_i b_i b_i'` the influence of the multipliers
//! is `G_g^-1 psi_i`, and `d log w_i / d lambda_g = b_i` for `i in g`. The target
//! mean and standard deviation enter as constants of the parametrisation, and the
//! mean of the target is estimated through `psi_i`. These feed
//! [`BalanceWeights::weighted_regression`](super::BalanceWeights::weighted_regression)
//! and [`BalanceWeights::mean_difference`](super::BalanceWeights::mean_difference).
//! Zhao and Percival (2017) show that an estimate with these weights is
//! consistent if the outcome is linear in the balanced columns or the weights are
//! log-linear in them, so the stacked variance is the one to use.
//!
//! # References
//!
//! 1. Hainmueller J. Entropy balancing for causal effects: a multivariate
//!    reweighting method to produce balanced samples in observational studies.
//!    Polit Anal. 2012;20(1):25-46.
//! 2. Zhao Q, Percival D. Entropy balancing is doubly robust. J Causal Inference.
//!    2017;5(1).
//! 3. Boyd S, Vandenberghe L. Convex Optimization. Cambridge University Press;
//!    2004. Section 9.5.

#![expect(
    clippy::float_cmp,
    reason = "a column of exactly 0 and 1 is binary, and equal bounds mean a constant column"
)]

use epistat_linalg::{
    Attempt, Mat, MatRef, Method,
    design::{independent_columns, select_rows},
    inverse, matvec_into, weighted_gram, xtv,
};

use super::{BalanceWeights, Nuisance, compensated, resolve_weights, summarise};
use epistat_core::{CaseWeights, Estimand};

use epistat_core::check;

use crate::error::{Error, Model, Result, domain, empty};
use crate::shared::{check_design, check_estimand, finite_design};

/// The data of [`entropy_balance`].
#[derive(Debug, Clone, Copy)]
pub struct EntropyInput<'a> {
    /// Whether each person is treated.
    pub treatment: &'a [bool],
    /// The columns to balance, one row per person, without an intercept. Build
    /// higher moments and products with [`balance_columns`].
    pub covariates: MatRef<'a, f64>,
    /// Sample weights, 1 for each person when `None`.
    pub sample_weight: Option<CaseWeights<'a>>,
    /// The base weights `q` the weights stay close to, 1 for each person when
    /// `None`. A person with base weight 0 gets weight 0.
    pub base_weight: Option<&'a [f64]>,
}

/// What [`entropy_balance`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntropyOptions {
    /// The population to balance to: the ATE (the default), ATT or ATC.
    pub estimand: Estimand,
    /// The largest standardised difference between the weighted means and the
    /// target that is accepted. The default is `1e-10`.
    pub tolerance: f64,
    /// The most Newton iterations in one group. The default is 100.
    pub max_iterations: usize,
}

impl Default for EntropyOptions {
    fn default() -> Self {
        Self {
            estimand: Estimand::Ate,
            tolerance: 1e-10,
            max_iterations: 100,
        }
    }
}

/// The result of [`entropy_balance`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct EntropyBalance {
    /// The weights and what the outcome estimates need.
    pub weights: BalanceWeights,
    /// The largest absolute difference, in standard deviations of the target,
    /// between a weighted mean of a column and its target, found from the
    /// returned weights.
    pub max_imbalance: f64,
    /// The multipliers `lambda` of each weighted group, on the standardised
    /// columns: the controls first, then the treated.
    pub multipliers: Vec<Vec<f64>>,
}

const MULTIPLIER_LIMIT: f64 = 200.0;

/// The raw columns, then the powers `2..=moments` of each column that is not
/// binary, then, if `interactions`, the products of each pair of columns. Powers
/// and products use columns centred at their means, which spans the same space
/// as the raw ones. A product that is constant is not generated.
///
/// This is the choice `WeightIt` makes with `moments` and `int = TRUE`.
///
/// # Errors
///
/// [`Error::Input`] for `moments` of 0 and a value that is not finite.
pub fn balance_columns(
    covariates: MatRef<'_, f64>,
    moments: usize,
    interactions: bool,
) -> Result<Mat<f64>> {
    if moments == 0 {
        return Err(domain("moments", 0.0, "at least 1"));
    }
    finite_design("covariates", covariates)?;
    let (n, p) = (covariates.nrows(), covariates.ncols());
    let nf = n as f64;
    let mean: Vec<f64> = (0..p)
        .map(|j| compensated((0..n).map(|i| covariates[(i, j)])) / nf)
        .collect();
    let binary: Vec<bool> = (0..p)
        .map(|j| (0..n).all(|i| covariates[(i, j)] == 0.0 || covariates[(i, j)] == 1.0))
        .collect();
    let mut cols: Vec<Vec<f64>> = (0..p)
        .map(|j| (0..n).map(|i| covariates[(i, j)]).collect())
        .collect();
    for k in 2..=moments {
        for j in 0..p {
            if !binary[j] {
                cols.push(
                    (0..n)
                        .map(|i| {
                            (covariates[(i, j)] - mean[j])
                                .powi(i32::try_from(k).unwrap_or(i32::MAX))
                        })
                        .collect(),
                );
            }
        }
    }
    if interactions {
        for a in 0..p {
            for b in (a + 1)..p {
                let col: Vec<f64> = (0..n)
                    .map(|i| {
                        let (x, y) = (covariates[(i, a)], covariates[(i, b)]);
                        let cx = if binary[a] { x } else { x - mean[a] };
                        let cy = if binary[b] { y } else { y - mean[b] };
                        cx * cy
                    })
                    .collect();
                if col.iter().any(|&v| v != col[0]) {
                    cols.push(col);
                }
            }
        }
    }
    Ok(Mat::from_fn(n, cols.len(), |i, j| cols[j][i]))
}

struct Armsolve {
    lambda: Vec<f64>,
    iterations: usize,
}

/// `F` and the weights `a_i exp(eta_i - max)` at `lambda`.
fn evaluate(u: MatRef<'_, f64>, a: &[f64], lambda: &[f64]) -> (f64, Vec<f64>, f64) {
    let m = u.nrows();
    let p = u.ncols();
    let mut eta = vec![0.0; m];
    matvec_into(&mut eta, u, MatRef::from_column_major_slice(lambda, p, 1));
    let mx = eta
        .iter()
        .zip(a)
        .filter(|(_, ai)| **ai > 0.0)
        .map(|(e, _)| *e)
        .fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = eta
        .iter()
        .zip(a)
        .map(|(&et, &ai)| if ai > 0.0 { ai * (et - mx).exp() } else { 0.0 })
        .collect();
    let z = compensated(e.iter().copied());
    (mx + z.ln(), e, z)
}

fn solve_arm(u: MatRef<'_, f64>, a: &[f64], opts: &EntropyOptions) -> Result<Armsolve> {
    let model = Model::Balance;
    let p = u.ncols();
    let mut lambda = vec![0.0; p];
    let (mut f, mut e, mut z) = evaluate(u, a, &lambda);
    for iteration in 0..=opts.max_iterations {
        let sum = xtv(u, &e);
        let g: Vec<f64> = (0..p).map(|j| sum[(j, 0)] / z).collect();
        let gmax = g.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
        if gmax <= opts.tolerance {
            return Ok(Armsolve {
                lambda,
                iterations: iteration,
            });
        }
        if iteration == opts.max_iterations {
            break;
        }
        let mut h = weighted_gram(u, &e).map_err(|err| Error::linalg(model, err))?;
        for r in 0..p {
            for c in 0..p {
                h[(r, c)] = h[(r, c)] / z - g[r] * g[c];
            }
        }
        let rhs = Mat::from_fn(p, 1, |r, _| -g[r]);
        let step = match epistat_linalg::solve(
            h.as_ref(),
            rhs.as_ref(),
            &[Attempt::plain(Method::Cholesky)],
        ) {
            Ok(s) => s.x,
            Err(_) if iteration == 0 => {
                return Err(Error::invalid(
                    "covariates",
                    "the columns are collinear among the persons to weight",
                ));
            }
            Err(_) => {
                return Err(Error::Infeasible {
                    model,
                    why: "the weights have collapsed onto a few persons, so the target is on the boundary of the group",
                });
            }
        };
        let slope: f64 = (0..p).map(|j| g[j] * step[(j, 0)]).sum();
        let mut t = 1.0;
        let mut accepted = false;
        for _ in 0..60 {
            let trial: Vec<f64> = (0..p).map(|j| lambda[j] + t * step[(j, 0)]).collect();
            let (f2, e2, z2) = evaluate(u, a, &trial);
            if f2.is_finite() && f2 <= f + 1e-4 * t * slope + 8.0 * f64::EPSILON * (f.abs() + 1.0) {
                lambda = trial;
                (f, e, z) = (f2, e2, z2);
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            return Err(Error::NotConverged {
                model,
                iterations: Some(iteration + 1),
            });
        }
        if lambda.iter().any(|x| x.abs() > MULTIPLIER_LIMIT) {
            return Err(Error::Infeasible {
                model,
                why: "the multipliers grow without bound, so the target cannot be reached with weights on the group",
            });
        }
    }
    Err(Error::NotConverged {
        model,
        iterations: Some(opts.max_iterations),
    })
}

/// Entropy balancing of a binary treatment.
///
/// # Errors
///
/// [`Error::Input`] for values of different lengths, a value that is not
/// finite, a negative sample or base weight and a tolerance that is not
/// positive; [`Error::Invalid`] for a group or target without persons of
/// positive weight and a column that does not vary in the target;
/// [`Error::Aliased`] for a column collinear among the persons to weight.
/// [`Error::Infeasible`] when the target moments cannot be matched, as when
/// a target mean is outside the range of the group. [`Error::NotConverged`]
/// after the iteration limit or when the line search fails.
pub fn entropy_balance(input: &EntropyInput<'_>, opts: &EntropyOptions) -> Result<EntropyBalance> {
    let n = input.treatment.len();
    if n == 0 {
        return Err(empty("treatment"));
    }
    check_design("covariates", input.covariates, n)?;
    let p = input.covariates.ncols();
    check::positive("tolerance", opts.tolerance)?;
    check_estimand(opts.estimand)?;
    let s = resolve_weights("sample_weight", input.sample_weight, n)?;
    let q = crate::shared::weight_column("base_weight", input.base_weight, n)?;
    let t = input.treatment;
    let in_target = |i: usize| {
        s[i] > 0.0
            && match opts.estimand {
                Estimand::Att => t[i],
                Estimand::Atc => !t[i],
                _ => true,
            }
    };
    let target_rows: Vec<usize> = (0..n).filter(|&i| in_target(i)).collect();
    if target_rows.is_empty() {
        return Err(Error::invalid(
            "sample_weight",
            "the target population has weight 0",
        ));
    }
    let size = compensated(target_rows.iter().map(|&i| s[i]));
    let c = input.covariates;
    let mean: Vec<f64> = (0..p)
        .map(|j| compensated(target_rows.iter().map(|&i| s[i] * c[(i, j)])) / size)
        .collect();
    let sd: Vec<f64> = (0..p)
        .map(|j| {
            (compensated(
                target_rows
                    .iter()
                    .map(|&i| s[i] * (c[(i, j)] - mean[j]).powi(2)),
            ) / size)
                .sqrt()
        })
        .collect();
    if let Some(j) = sd.iter().position(|&x| x == 0.0) {
        return Err(Error::invalid_at(
            "covariate column",
            j,
            "does not vary in the target",
        ));
    }
    let u = Mat::from_fn(n, p, |i, j| (c[(i, j)] - mean[j]) / sd[j]);

    // Groups to weight: false = controls, true = treated.
    let groups: Vec<bool> = match opts.estimand {
        Estimand::Att => vec![false],
        Estimand::Atc => vec![true],
        _ => vec![false, true],
    };
    let mut weights = vec![1.0; n];
    let mut iterations = 0;
    let mut multipliers = Vec::new();
    let width = p + 1;
    let mut influence = Mat::<f64>::zeros(n, width * groups.len());
    let mut dlog = Mat::<f64>::zeros(n, width * groups.len());
    let mut max_imbalance = 0.0_f64;
    let b_of = |i: usize, j: usize| if j == 0 { 1.0 } else { u[(i, j - 1)] };
    for (slot, &arm) in groups.iter().enumerate() {
        let name = if arm { "treated" } else { "controls" };
        let rows: Vec<usize> = (0..n)
            .filter(|&i| t[i] == arm && s[i] * q[i] > 0.0)
            .collect();
        if rows.is_empty() {
            return Err(Error::invalid(name, "has no persons with positive weight"));
        }
        let ug = select_rows(u.as_ref(), &rows);
        for j in 0..p {
            let lo = (0..rows.len())
                .map(|r| ug[(r, j)])
                .fold(f64::INFINITY, f64::min);
            let hi = (0..rows.len())
                .map(|r| ug[(r, j)])
                .fold(f64::NEG_INFINITY, f64::max);
            if lo == hi {
                return Err(if lo == 0.0 {
                    Error::invalid_at(
                        "covariate column",
                        j,
                        "does not vary in the group and equals the target",
                    )
                } else {
                    Error::Infeasible {
                        model: Model::Balance,
                        why: "a column is constant in the group and the target mean differs",
                    }
                });
            }
            if !(lo < 0.0 && hi > 0.0) {
                return Err(Error::Infeasible {
                    model: Model::Balance,
                    why: "the target mean of a column is not inside the range of the group",
                });
            }
        }
        let a: Vec<f64> = rows.iter().map(|&i| s[i] * q[i]).collect();
        let keep = independent_columns(ug.as_ref(), &a, None, 1e-8)
            .map_err(|e| Error::linalg(Model::Balance, e))?;
        if let Some(column) = keep.iter().position(|&x| !x) {
            return Err(Error::Aliased {
                model: Model::Balance,
                column,
            });
        }
        let sol = solve_arm(ug.as_ref(), &a, opts)?;
        iterations += sol.iterations;
        // The weights of everyone in the group, from the multipliers.
        let eta: Vec<f64> = (0..n)
            .map(|i| (0..p).map(|j| sol.lambda[j] * u[(i, j)]).sum())
            .collect();
        let mx = rows
            .iter()
            .map(|&i| eta[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let denom = compensated(rows.iter().map(|&i| s[i] * q[i] * (eta[i] - mx).exp()));
        for i in (0..n).filter(|&i| t[i] == arm) {
            weights[i] = size / denom * q[i] * (eta[i] - mx).exp();
        }
        // Imbalance from the returned weights.
        for j in 0..p {
            let diff = compensated(rows.iter().map(|&i| s[i] * weights[i] * u[(i, j)])) / size;
            max_imbalance = max_imbalance.max(diff.abs());
        }
        // The influence of the multipliers and the derivative of the log weights.
        let off = slot * width;
        let total: Vec<f64> = (0..n)
            .map(|i| if t[i] == arm { s[i] * weights[i] } else { 0.0 })
            .collect();
        let b = Mat::from_fn(n, width, &b_of);
        let gram =
            weighted_gram(b.as_ref(), &total).map_err(|e| Error::linalg(Model::Balance, e))?;
        let ginv = inverse(gram.as_ref(), &Attempt::SYMMETRIC)
            .map_err(|_| Error::Singular {
                model: Model::Balance,
            })?
            .x;
        for i in 0..n {
            let member = t[i] == arm;
            let target_part = f64::from(u8::from(match opts.estimand {
                Estimand::Att => t[i] && s[i] > 0.0,
                Estimand::Atc => !t[i] && s[i] > 0.0,
                _ => s[i] > 0.0,
            }));
            let factor = s[i] * (target_part - if member { weights[i] } else { 0.0 });
            if factor != 0.0 {
                for r in 0..width {
                    let mut x = 0.0;
                    for cc in 0..width {
                        x += ginv[(r, cc)] * b[(i, cc)];
                    }
                    influence[(i, off + r)] = x * factor;
                }
            }
            if member {
                for r in 0..width {
                    dlog[(i, off + r)] = b[(i, r)];
                }
            }
        }
        multipliers.push(sol.lambda);
    }
    let weighted = |arm: bool| groups.contains(&arm);
    let controls = summarise(weighted(false), false, t, &s, &weights);
    let treated = summarise(weighted(true), true, t, &s, &weights);
    Ok(EntropyBalance {
        weights: BalanceWeights::new(
            opts.estimand,
            weights,
            s,
            t.to_vec(),
            controls,
            treated,
            iterations,
            Nuisance { influence, dlog },
        ),
        max_imbalance,
        multipliers,
    })
}
