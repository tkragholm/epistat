//! The weights of one adoption cohort and its estimate: `synthdid_estimate`,
//! `sc.weight.fw` and `sc.weight.fw.covariates` of R's `synthdid` 0.0.9.

use epistat_optim::frank_wolfe::{FrankWolfeOptions, SimplexLeastSquares};

use crate::error::Error;

use super::SynthTrace;

/// The panel in the layout the fits read: outcomes by unit (`units × periods`,
/// unit-major) and the covariates as `count` such blocks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Data<'a> {
    pub outcome: &'a [f64],
    pub covariates: &'a [f64],
    pub count: usize,
    pub units: usize,
    pub periods: usize,
}

impl Data<'_> {
    pub fn outcome_row(&self, unit: usize) -> &[f64] {
        &self.outcome[unit * self.periods..(unit + 1) * self.periods]
    }

    pub fn covariate_row(&self, k: usize, unit: usize) -> &[f64] {
        let start = (k * self.units + unit) * self.periods;
        &self.covariates[start..start + self.periods]
    }
}

/// The options a fit runs with once its defaults are resolved, the `opts`
/// that `synthdid` stores with an estimate and passes to its refits.
#[derive(Debug, Clone, Copy, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the flags of synthdid_estimate, one per argument"
)]
pub(crate) struct Opts {
    pub zeta_omega: f64,
    pub zeta_lambda: f64,
    pub omega_intercept: bool,
    pub lambda_intercept: bool,
    pub update_omega: bool,
    pub update_lambda: bool,
    pub min_decrease: f64,
    pub max_iter: usize,
    pub max_iter_pre_sparsify: usize,
    pub sparsify: bool,
}

/// Where the weights start, or what they are held at when not updated.
#[derive(Debug, Clone, Default)]
pub(crate) struct Start {
    pub lambda: Option<Vec<f64>>,
    pub omega: Option<Vec<f64>>,
}

/// One fitted cohort.
#[derive(Debug, Clone)]
pub(crate) struct Fit {
    pub estimate: f64,
    pub curve: Vec<f64>,
    pub omega: Vec<f64>,
    pub lambda: Vec<f64>,
    pub beta: Vec<f64>,
    pub trace: SynthTrace,
    pub converged: bool,
}

/// R's `mean`: the sum divided by `n`, corrected by the mean of the
/// deviations from it. Doubles throughout, as R's long double is a double on
/// aarch64.
pub(crate) fn r_mean(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let mut n = 0usize;
    let mut sum = 0.0;
    for v in values.clone() {
        sum += v;
        n += 1;
    }
    let mean = sum / n as f64;
    let correction: f64 = values.map(|v| v - mean).sum();
    mean + correction / n as f64
}

/// `sd(apply(Y[1:N0, 1:T0], 1, diff))`: the standard deviation, with divisor
/// `n - 1`, of the first differences of the control outcomes over the
/// pre-treatment periods. `None` with fewer than two differences.
pub(crate) fn noise_level(data: &Data<'_>, controls: &[usize], pre: usize) -> Option<f64> {
    let diffs = || {
        controls.iter().flat_map(move |&unit| {
            let row = &data.outcome_row(unit)[..pre];
            row.windows(2).map(|w| w[1] - w[0])
        })
    };
    let n = controls.len() * pre.saturating_sub(1);
    if n < 2 {
        return None;
    }
    let mean = r_mean(diffs());
    let squares: f64 = diffs().map(|d| (d - mean) * (d - mean)).sum();
    Some((squares / (n - 1) as f64).sqrt())
}

/// `collapsed.form`: the controls' pre-period values with the mean of their
/// post periods, and a last row of the treated units' means. `(n0 + 1) ×
/// (pre + 1)`, row-major. `row` gives a unit's values over all periods.
fn collapse<'r>(
    row: impl Fn(usize) -> &'r [f64],
    controls: &[usize],
    treated: &[usize],
    pre: usize,
    periods: usize,
) -> Vec<f64> {
    let width = pre + 1;
    let post = (periods - pre) as f64;
    let mut out = vec![0.0; (controls.len() + 1) * width];
    for (j, &unit) in controls.iter().enumerate() {
        let values = row(unit);
        let target = &mut out[j * width..(j + 1) * width];
        target[..pre].copy_from_slice(&values[..pre]);
        target[pre] = values[pre..].iter().sum::<f64>() / post;
    }
    let last = &mut out[controls.len() * width..];
    let n1 = treated.len() as f64;
    for (t, cell) in last[..pre].iter_mut().enumerate() {
        *cell = treated.iter().map(|&unit| row(unit)[t]).sum::<f64>() / n1;
    }
    // `mean(Y[treated, post])` reads the block by column, period by period.
    last[pre] = r_mean(
        (pre..periods)
            .flat_map(|t| treated.iter().map(move |&unit| (unit, t)))
            .map(|(unit, t)| row(unit)[t]),
    );
    out
}

/// The time-weight problem of `sc.weight.fw(Yc[1:N0, ], ...)`: `A` has a
/// column per pre-period over the controls, `b` the controls' post means,
/// both centred over the controls when `intercept`.
fn lambda_problem(yc: &[f64], n0: usize, pre: usize, intercept: bool) -> (Vec<f64>, Vec<f64>) {
    let width = pre + 1;
    let mut a = vec![0.0; n0 * pre];
    let mut b = vec![0.0; n0];
    for t in 0..=pre {
        let column = (0..n0).map(|j| yc[j * width + t]);
        let centre = if intercept {
            r_mean(column.clone())
        } else {
            0.0
        };
        let target = if t < pre {
            &mut a[t * n0..(t + 1) * n0]
        } else {
            &mut b[..]
        };
        for (cell, value) in target.iter_mut().zip(column) {
            *cell = value - centre;
        }
    }
    (a, b)
}

/// The unit-weight problem of `sc.weight.fw(t(Yc[, 1:T0]), ...)`: `A` has a
/// column per control over the pre-periods, `b` the treated means, each
/// centred over the pre-periods when `intercept`.
fn omega_problem(yc: &[f64], n0: usize, pre: usize, intercept: bool) -> (Vec<f64>, Vec<f64>) {
    let width = pre + 1;
    let mut a = vec![0.0; pre * n0];
    let mut b = vec![0.0; pre];
    for j in 0..=n0 {
        let column = &yc[j * width..j * width + pre];
        let centre = if intercept {
            r_mean(column.iter().copied())
        } else {
            0.0
        };
        let target = if j < n0 {
            &mut a[j * pre..(j + 1) * pre]
        } else {
            &mut b[..]
        };
        for (cell, &value) in target.iter_mut().zip(column) {
            *cell = value - centre;
        }
    }
    (a, b)
}

fn residual(ax: &[f64], b: &[f64]) -> Vec<f64> {
    ax.iter().zip(b).map(|(&v, &bv)| v - bv).collect()
}

/// `A x` for `A` stored by column with `rows` rows.
fn product(a: &[f64], rows: usize, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; rows];
    for (column, &xj) in a.chunks_exact(rows).zip(x) {
        for (cell, &value) in out.iter_mut().zip(column) {
            *cell += xj * value;
        }
    }
    out
}

/// `sparsify_function`: weights at most a quarter of the largest set to zero,
/// the rest rescaled to sum to one.
fn sparsify(mut v: Vec<f64>) -> Vec<f64> {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for value in &mut v {
        if *value <= max / 4.0 {
            *value = 0.0;
        }
    }
    let total: f64 = v.iter().sum();
    for value in &mut v {
        *value /= total;
    }
    v
}

/// `sum_normalize`: `x / sum(x)`, or uniform weights when the sum is zero.
pub(crate) fn sum_normalize(mut x: Vec<f64>) -> Vec<f64> {
    let total: f64 = x.iter().sum();
    if total == 0.0 {
        let uniform = 1.0 / x.len() as f64;
        x.fill(uniform);
    } else {
        for v in &mut x {
            *v /= total;
        }
    }
    x
}

/// The weights of one problem without covariates: a first run of at most
/// `max_iter_pre_sparsify` steps, then, when `sparsify`, a second run from the
/// sparsified result. Returns the weights, the trace of the last run and
/// whether it met its tolerance.
fn solve_weights(
    a: &[f64],
    b: &[f64],
    rows: usize,
    cols: usize,
    zeta: f64,
    start: Option<Vec<f64>>,
    opts: &Opts,
) -> Result<(Vec<f64>, Vec<f64>, bool), Error> {
    let eta = rows as f64 * (zeta * zeta);
    let problem = SimplexLeastSquares::new(a, rows, cols, b, eta)?;
    let tolerance = opts.min_decrease * opts.min_decrease;
    let x0 = start.unwrap_or_else(|| vec![1.0 / cols as f64; cols]);
    let first_limit = if opts.sparsify {
        opts.max_iter_pre_sparsify
    } else {
        opts.max_iter
    };
    let first = problem.solve(
        x0,
        FrankWolfeOptions {
            tolerance,
            max_iter: first_limit,
        },
    )?;
    if !opts.sparsify {
        return Ok((first.x, first.values, first.converged));
    }
    let second = problem.solve(
        sparsify(first.x),
        FrankWolfeOptions {
            tolerance,
            max_iter: opts.max_iter,
        },
    )?;
    Ok((second.x, second.values, second.converged))
}

/// The weights and estimate of the cohort whose treated units are `treated`,
/// first treated after `pre` periods, against `controls`. Rows may repeat, as
/// in a bootstrap sample.
pub(crate) fn fit(
    data: &Data<'_>,
    controls: &[usize],
    treated: &[usize],
    pre: usize,
    opts: &Opts,
    start: Start,
) -> Result<Fit, Error> {
    let n0 = controls.len();
    let periods = data.periods;
    let yc = collapse(|u| data.outcome_row(u), controls, treated, pre, periods);
    let (lambda, omega, beta, trace, converged) =
        if data.count == 0 {
            let lambda_job = || -> Result<(Vec<f64>, Vec<f64>, bool), Error> {
                if opts.update_lambda {
                    let (a, b) = lambda_problem(&yc, n0, pre, opts.lambda_intercept);
                    solve_weights(
                        &a,
                        &b,
                        n0,
                        pre,
                        opts.zeta_lambda,
                        start.lambda.clone(),
                        opts,
                    )
                } else {
                    Ok((
                        start.lambda.clone().ok_or(Error::Input(
                            epistat_core::InputError::Empty { what: "weights" },
                        ))?,
                        Vec::new(),
                        true,
                    ))
                }
            };
            let omega_job = || -> Result<(Vec<f64>, Vec<f64>, bool), Error> {
                if opts.update_omega {
                    let (a, b) = omega_problem(&yc, n0, pre, opts.omega_intercept);
                    solve_weights(&a, &b, pre, n0, opts.zeta_omega, start.omega.clone(), opts)
                } else {
                    Ok((
                        start.omega.clone().ok_or(Error::Input(
                            epistat_core::InputError::Empty { what: "weights" },
                        ))?,
                        Vec::new(),
                        true,
                    ))
                }
            };
            let (l, o) = rayon::join(lambda_job, omega_job);
            let (lambda, lambda_trace, lambda_ok) = l?;
            let (omega, omega_trace, omega_ok) = o?;
            (
                lambda,
                omega,
                Vec::new(),
                SynthTrace {
                    lambda: lambda_trace,
                    omega: omega_trace,
                    joint: Vec::new(),
                },
                lambda_ok && omega_ok,
            )
        } else {
            let xc: Vec<Vec<f64>> = (0..data.count)
                .map(|k| {
                    collapse(
                        |u| data.covariate_row(k, u),
                        controls,
                        treated,
                        pre,
                        periods,
                    )
                })
                .collect();
            let fitted = fit_covariates(&yc, &xc, n0, pre, opts, start)?;
            (
                fitted.lambda,
                fitted.omega,
                fitted.beta,
                SynthTrace {
                    lambda: Vec::new(),
                    omega: Vec::new(),
                    joint: fitted.values,
                },
                fitted.converged,
            )
        };
    let (estimate, curve) =
        estimate_and_curve(data, controls, treated, pre, &omega, &lambda, &beta);
    Ok(Fit {
        estimate,
        curve,
        omega,
        lambda,
        beta,
        trace,
        converged,
    })
}

/// `tau.sc = c(-omega, 1/N1)' (Y - X beta)` over all periods, then the
/// estimate `tau.sc c(-lambda, 1/T1)` and the effect curve
/// `tau.sc[T0 + s] - tau.sc[1:T0] lambda`, as `synthdid_estimate` and
/// `synthdid_effect_curve` form them.
pub(crate) fn estimate_and_curve(
    data: &Data<'_>,
    controls: &[usize],
    treated: &[usize],
    pre: usize,
    omega: &[f64],
    lambda: &[f64],
    beta: &[f64],
) -> (f64, Vec<f64>) {
    let periods = data.periods;
    let mut tau = vec![0.0; periods];
    let adjusted = |unit: usize, t: usize| -> f64 {
        let mut xb = 0.0;
        for (k, &b) in beta.iter().enumerate() {
            xb += b * data.covariate_row(k, unit)[t];
        }
        data.outcome_row(unit)[t] - xb
    };
    let share = 1.0 / treated.len() as f64;
    for (t, cell) in tau.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (&w, &unit) in omega.iter().zip(controls) {
            sum += -w * adjusted(unit, t);
        }
        for &unit in treated {
            sum += share * adjusted(unit, t);
        }
        *cell = sum;
    }
    let post_share = 1.0 / (periods - pre) as f64;
    let mut estimate = 0.0;
    for (t, &value) in tau.iter().enumerate() {
        let weight = if t < pre { -lambda[t] } else { post_share };
        estimate += value * weight;
    }
    let baseline: f64 = tau[..pre].iter().zip(lambda).map(|(&v, &l)| v * l).sum();
    let curve = tau[pre..].iter().map(|&v| v - baseline).collect();
    (estimate, curve)
}

/// The weights and objective of one iteration of `sc.weight.fw.covariates`.
struct Update {
    val: f64,
    err_lambda: Vec<f64>,
    err_omega: Vec<f64>,
}

struct CovariateFit {
    lambda: Vec<f64>,
    omega: Vec<f64>,
    beta: Vec<f64>,
    values: Vec<f64>,
    converged: bool,
}

/// `sc.weight.fw.covariates` on the collapsed outcome and covariates: one
/// Frank-Wolfe step for each set of weights per iteration, alternated with a
/// gradient step of size `1 / t` on `beta`, stopping when the objective
/// changes by no more than `min_decrease^2` either way.
fn fit_covariates(
    yc: &[f64],
    xc: &[Vec<f64>],
    n0: usize,
    pre: usize,
    opts: &Opts,
    start: Start,
) -> Result<CovariateFit, Error> {
    let width = pre + 1;
    let k = xc.len();
    let mut lambda = start.lambda.unwrap_or_else(|| vec![1.0 / pre as f64; pre]);
    let mut omega = start.omega.unwrap_or_else(|| vec![1.0 / n0 as f64; n0]);
    let mut beta = vec![0.0; k];
    let tolerance = opts.min_decrease * opts.min_decrease;
    let update =
        |y: &[f64], lambda: &mut Vec<f64>, omega: &mut Vec<f64>| -> Result<Update, Error> {
            let (a, b) = lambda_problem(y, n0, pre, opts.lambda_intercept);
            let eta = n0 as f64 * (opts.zeta_lambda * opts.zeta_lambda);
            let problem = SimplexLeastSquares::new(&a, n0, pre, &b, eta)?;
            let err_lambda = if opts.update_lambda {
                let mut state = problem.start(std::mem::take(lambda))?;
                problem.step(&mut state);
                *lambda = state.x;
                residual(&state.ax, &b)
            } else {
                residual(&product(&a, n0, lambda), &b)
            };
            let (a, b) = omega_problem(y, n0, pre, opts.omega_intercept);
            let eta = pre as f64 * (opts.zeta_omega * opts.zeta_omega);
            let problem = SimplexLeastSquares::new(&a, pre, n0, &b, eta)?;
            let err_omega = if opts.update_omega {
                let mut state = problem.start(std::mem::take(omega))?;
                problem.step(&mut state);
                *omega = state.x;
                residual(&state.ax, &b)
            } else {
                residual(&product(&a, pre, omega), &b)
            };
            let sq = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
            let val = opts.zeta_omega * opts.zeta_omega * sq(omega)
                + opts.zeta_lambda * opts.zeta_lambda * sq(lambda)
                + sq(&err_omega) / pre as f64
                + sq(&err_lambda) / n0 as f64;
            Ok(Update {
                val,
                err_lambda,
                err_omega,
            })
        };
    // `Y - contract3(X, beta)`: the sum of `beta_k X_k` first, then the
    // difference.
    let adjust = |beta: &[f64]| -> Vec<f64> {
        let mut xb = vec![0.0; yc.len()];
        for (b, x) in beta.iter().zip(xc) {
            for (cell, &xv) in xb.iter_mut().zip(x) {
                *cell += b * xv;
            }
        }
        yc.iter().zip(&xb).map(|(&y, &v)| y - v).collect()
    };

    let mut weights = update(&adjust(&beta), &mut lambda, &mut omega)?;
    let mut values: Vec<f64> = Vec::new();
    let mut converged = false;
    while values.len() < opts.max_iter {
        let t = values.len() + 1;
        let alpha = 1.0 / t as f64;
        for (bk, x) in beta.iter_mut().zip(xc) {
            // err_lambda' X[1:N0, ] c(lambda, -1) / N0
            let mut first = 0.0;
            for (j, &e) in weights.err_lambda.iter().enumerate() {
                let row = &x[j * width..(j + 1) * width];
                let mut contracted = 0.0;
                for (&xv, &l) in row[..pre].iter().zip(&lambda) {
                    contracted += xv * l;
                }
                contracted -= row[pre];
                first += e * contracted;
            }
            // err_omega' t(X[, 1:T0]) c(omega, -1) / T0
            let mut second = 0.0;
            for (s, &e) in weights.err_omega.iter().enumerate() {
                let mut contracted = 0.0;
                for (j, &w) in omega.iter().enumerate() {
                    contracted += x[j * width + s] * w;
                }
                contracted -= x[n0 * width + s];
                second += e * contracted;
            }
            let gradient = -(first / n0 as f64 + second / pre as f64);
            *bk -= alpha * gradient;
        }
        weights = update(&adjust(&beta), &mut lambda, &mut omega)?;
        values.push(weights.val);
        let n = values.len();
        if n >= 2 && (values[n - 2] - values[n - 1]).abs() <= tolerance {
            converged = true;
            break;
        }
    }
    Ok(CovariateFit {
        lambda,
        omega,
        beta,
        values,
        converged,
    })
}
