use epistat_glm::expit;
use epistat_rng::{Stream, replicate_seed, splitmix64};
use rayon::prelude::*;

use crate::data::{Column, Incomplete};
use crate::error::{ImputeError, Result};
use crate::parts::{
    SortedDonors, augmentation, class_from_uniform, class_probabilities, design_matrix,
    draw_coefficients, draw_linear, linear_fit, logistic_fit, multinomial_fit,
};
use crate::spec::{ImputeOptions, ImputeSpec, MatchType, Method};

/// The imputed values of every incomplete column, the diagnostics of the
/// chains and what the imputation had to repair.
///
/// Only the imputed cells are stored, as `mids$imp` of `mice` does, not `m`
/// copies of the data. [`completed`](Imputed::completed) builds a completed
/// data set.
#[derive(Clone, Debug, PartialEq)]
pub struct Imputed {
    /// The number of data sets.
    pub m: usize,
    /// The number of iterations of each chain.
    pub max_iterations: usize,
    /// For each column the rows that were missing, in increasing order. Empty
    /// for a complete column.
    pub missing_rows: Vec<Vec<usize>>,
    /// The imputed values, by column, data set and missing cell in the order of
    /// `missing_rows`. A binary or categorical value is its code.
    pub values: Vec<Vec<Vec<f64>>>,
    /// The mean of the imputed cells of each column after each iteration, by
    /// column and then data set times `max_iterations` plus iteration, as the
    /// chain means of `mice`.
    pub chain_mean: Vec<Vec<f64>>,
    /// The variance of the imputed cells, laid out as `chain_mean`, `None`
    /// where fewer than two cells are imputed.
    pub chain_variance: Vec<Vec<Option<f64>>>,
    /// The number of regressions fitted with a column of the design left out
    /// because it was a linear combination of earlier ones.
    pub n_aliased_dropped: usize,
    /// The number of regressions that had the rows of `mice:::augment` added.
    pub n_augmented: usize,
}

struct Context<'a> {
    data: &'a Incomplete,
    spec: &'a ImputeSpec,
    options: &'a ImputeOptions,
    observed: Vec<Vec<usize>>,
    missing: Vec<Vec<usize>>,
}

struct ChainOutput {
    values: Vec<Vec<f64>>,
    mean: Vec<Vec<f64>>,
    variance: Vec<Vec<Option<f64>>>,
    aliased: usize,
    augmented: usize,
}

/// Imputes the missing cells of `data` by chained equations.
///
/// Each chain starts from a random draw of an observed value for every missing
/// cell. An iteration then visits the incomplete columns in `spec.visit_order`:
/// the column is regressed on its predictors as currently completed, using the
/// rows where it is observed, and its missing cells are replaced by draws from
/// the fitted model. After `max_iterations` iterations the missing cells are
/// the imputation. The `m` chains are independent. See the crate
/// documentation for the methods, the streams and the survival predictors.
///
/// # Errors
///
/// [`ImputeError`]: no data set or iteration, a method list or predictor list
/// that does not fit the data, an incomplete column without a method or a
/// complete one with a method, a column with no observed value, a method that
/// does not apply to the column's type, a column with too few observed rows
/// for its model, a categorical level that never occurs among the observed
/// values, a regression that did not converge, and a covariance that is not
/// positive definite.
pub fn impute(data: &Incomplete, spec: &ImputeSpec, options: &ImputeOptions) -> Result<Imputed> {
    validate(data, spec, options)?;
    let n_columns = data.n_columns();
    let observed: Vec<Vec<usize>> = (0..n_columns)
        .map(|j| (0..data.rows()).filter(|&r| !data.missing(j).get(r)).collect())
        .collect();
    let missing: Vec<Vec<usize>> = (0..n_columns).map(|j| data.missing(j).rows()).collect();
    let ctx = Context {
        data,
        spec,
        options,
        observed,
        missing,
    };
    let chains: Vec<ChainOutput> = (0..options.m)
        .into_par_iter()
        .map(|chain| run_chain(&ctx, chain))
        .collect::<Result<_>>()?;

    let m = options.m;
    let maxit = options.max_iterations;
    let mut values = vec![Vec::with_capacity(m); n_columns];
    let mut chain_mean = vec![Vec::with_capacity(m * maxit); n_columns];
    let mut chain_variance = vec![Vec::with_capacity(m * maxit); n_columns];
    let (mut aliased, mut augmented) = (0, 0);
    for chain in chains {
        aliased += chain.aliased;
        augmented += chain.augmented;
        for j in 0..n_columns {
            values[j].push(chain.values[j].clone());
            chain_mean[j].extend_from_slice(&chain.mean[j]);
            chain_variance[j].extend_from_slice(&chain.variance[j]);
        }
    }
    Ok(Imputed {
        m,
        max_iterations: maxit,
        missing_rows: ctx.missing,
        values,
        chain_mean,
        chain_variance,
        n_aliased_dropped: aliased,
        n_augmented: augmented,
    })
}

fn validate(data: &Incomplete, spec: &ImputeSpec, options: &ImputeOptions) -> Result<()> {
    if options.m == 0 {
        return Err(ImputeError::Count { what: "m" });
    }
    if options.max_iterations == 0 {
        return Err(ImputeError::Count {
            what: "max_iterations",
        });
    }
    let n = data.n_columns();
    for (what, found) in [
        ("methods", spec.methods.len()),
        ("predictors", spec.predictors.len()),
    ] {
        if found != n {
            return Err(ImputeError::Length {
                what,
                expected: n,
                found,
            });
        }
    }
    let incomplete: Vec<usize> = (0..n).filter(|&j| data.missing(j).count() > 0).collect();
    let mut visited = spec.visit_order.clone();
    visited.sort_unstable();
    if visited != incomplete {
        return Err(ImputeError::VisitOrder);
    }
    for j in 0..n {
        let missing = data.missing(j).count();
        let column = &data.columns()[j];
        match (&spec.methods[j], missing) {
            (None, 0) => continue,
            (Some(_), 0) => {
                return Err(ImputeError::Method {
                    column: j,
                    reason: "a column without missing cells has a method",
                });
            }
            (None, _) => {
                return Err(ImputeError::Method {
                    column: j,
                    reason: "an incomplete column has no method",
                });
            }
            (Some(method), _) => {
                if missing == data.rows() {
                    return Err(ImputeError::NothingObserved { column: j });
                }
                let fits = matches!(
                    (method, column),
                    (
                        Method::Norm | Method::Pmm { .. } | Method::Passive(_),
                        Column::Continuous(_)
                    ) | (Method::Logreg, Column::Binary(_))
                        | (Method::Polyreg { .. }, Column::Nominal { .. })
                        | (Method::Sample, _)
                );
                if !fits {
                    return Err(ImputeError::Unsupported { column: j });
                }
                if let Method::Pmm { donors, .. } = method {
                    if *donors == 0 {
                        return Err(ImputeError::Method {
                            column: j,
                            reason: "predictive mean matching needs at least one donor",
                        });
                    }
                }
            }
        }
    }
    for j in 0..n {
        let mut seen = vec![false; n];
        for &k in &spec.predictors[j] {
            if k >= n || k == j || seen[k] {
                return Err(ImputeError::Predictor {
                    column: j,
                    predictor: k,
                });
            }
            seen[k] = true;
        }
    }
    // Every level of a categorical column that is imputed or used to impute
    // must occur among the observed values.
    for j in 0..n {
        let used = data.missing(j).count() > 0
            || spec
                .predictors
                .iter()
                .enumerate()
                .any(|(i, p)| spec.methods[i].is_some() && p.contains(&j));
        if let (true, Column::Nominal { codes, levels }) = (used, &data.columns()[j]) {
            let mut seen = vec![false; *levels];
            for r in (0..data.rows()).filter(|&r| !data.missing(j).get(r)) {
                seen[codes[r] as usize] = true;
            }
            if let Some(level) = seen.iter().position(|&s| !s) {
                return Err(ImputeError::EmptyLevel { column: j, level });
            }
        }
    }
    Ok(())
}

/// The seed of one column in one iteration of one chain.
fn column_seed(seed: u64, chain: usize, iteration: usize, column: usize) -> u64 {
    let mut s = splitmix64(seed);
    for part in [chain, iteration, column] {
        s = replicate_seed(s, part as u64);
    }
    s
}

/// The seed of the draws that belong to the column as a whole, such as the
/// parameters of its model, which no row index takes.
const PARAMETER_ROW: u64 = u64::MAX;

fn row_stream(column_seed: u64, row: usize) -> Stream {
    Stream::seeded(replicate_seed(column_seed, row as u64))
}

fn run_chain(ctx: &Context<'_>, chain: usize) -> Result<ChainOutput> {
    let (data, spec, options) = (ctx.data, ctx.spec, ctx.options);
    let n_columns = data.n_columns();
    let maxit = options.max_iterations;
    let mut completed: Vec<Vec<f64>> = (0..n_columns)
        .map(|j| (0..data.rows()).map(|r| data.columns()[j].value(r)).collect())
        .collect();
    // Starting values: a random observed value for each missing cell.
    for j in 0..n_columns {
        let (obs, mis) = (&ctx.observed[j], &ctx.missing[j]);
        if mis.is_empty() {
            continue;
        }
        let seed = column_seed(options.seed, chain, 0, j);
        let source = &completed[j];
        let draws: Vec<f64> = mis
            .par_iter()
            .with_min_len(4096)
            .map(|&r| source[obs[row_stream(seed, r).below(obs.len())]])
            .collect();
        for (&r, v) in mis.iter().zip(draws) {
            completed[j][r] = v;
        }
    }
    let mut out = ChainOutput {
        values: vec![Vec::new(); n_columns],
        mean: vec![Vec::with_capacity(maxit); n_columns],
        variance: vec![Vec::with_capacity(maxit); n_columns],
        aliased: 0,
        augmented: 0,
    };
    for iteration in 1..=maxit {
        for &j in &spec.visit_order {
            let seed = column_seed(options.seed, chain, iteration, j);
            let cells = impute_column(ctx, &completed, j, seed, &mut out)?;
            for (&r, &v) in ctx.missing[j].iter().zip(&cells) {
                completed[j][r] = v;
            }
            let (mean, variance) = moments(&cells);
            out.mean[j].push(mean);
            out.variance[j].push(variance);
        }
    }
    for j in 0..n_columns {
        out.values[j] = ctx.missing[j].iter().map(|&r| completed[j][r]).collect();
    }
    Ok(out)
}

fn moments(values: &[f64]) -> (f64, Option<f64>) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 {
        return (mean, None);
    }
    let ss: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
    (mean, Some(ss / (n - 1.0)))
}

/// The predictors of column `j` as numeric columns without an intercept:
/// indicators for the levels of a nominal predictor after the first.
fn predictor_columns(
    data: &Incomplete,
    completed: &[Vec<f64>],
    predictors: &[usize],
) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for &k in predictors {
        match &data.columns()[k] {
            Column::Nominal { levels, .. } => {
                for level in 1..*levels {
                    let target = level as f64;
                    out.push(completed[k].iter().map(|&v| f64::from(u8::from(v == target))).collect());
                }
            }
            _ => out.push(completed[k].clone()),
        }
    }
    out
}

fn predict(columns: &[Vec<f64>], rows: &[usize], beta: &[f64]) -> Vec<f64> {
    rows.iter()
        .map(|&r| {
            let mut s = beta[0];
            for (k, col) in columns.iter().enumerate() {
                s += beta[k + 1] * col[r];
            }
            s
        })
        .collect()
}

fn impute_column(
    ctx: &Context<'_>,
    completed: &[Vec<f64>],
    j: usize,
    seed: u64,
    out: &mut ChainOutput,
) -> Result<Vec<f64>> {
    let (data, spec) = (ctx.data, ctx.spec);
    let (obs, mis) = (&ctx.observed[j], &ctx.missing[j]);
    let method = spec.methods[j].as_ref().expect("validated");
    let y_obs: Vec<f64> = obs.iter().map(|&r| data.columns()[j].value(r)).collect();
    let uniform = |row: usize| row_stream(seed, row).uniform();

    match method {
        Method::Sample => Ok(mis
            .iter()
            .map(|&r| y_obs[row_stream(seed, r).below(y_obs.len())])
            .collect()),
        Method::Passive(f) => {
            let column = (f.0)(completed);
            if column.len() != data.rows() || mis.iter().any(|&r| !column[r].is_finite()) {
                return Err(ImputeError::Passive { column: j });
            }
            Ok(mis.iter().map(|&r| column[r]).collect())
        }
        Method::Norm | Method::Pmm { .. } => {
            let cols = predictor_columns(data, completed, &spec.predictors[j]);
            let x = design_matrix(&cols, obs);
            let fit = linear_fit(x.as_ref(), &y_obs, j)?;
            out.aliased += usize::from(fit.n_active() < fit.active.len());
            let mut stream = Stream::seeded(replicate_seed(seed, PARAMETER_ROW));
            let chi = stream.chi_square(fit.df)?;
            let mut z = vec![0.0; fit.n_active()];
            stream.fill_normals(&mut z);
            let draw = draw_linear(&fit, chi, &z, j)?;
            if let Method::Pmm { donors, match_type } = method {
                let (obs_beta, mis_beta) = match match_type {
                    MatchType::Fitted => (&fit.coefficients, &fit.coefficients),
                    MatchType::Mixed => (&fit.coefficients, &draw.beta),
                    MatchType::Drawn => (&draw.beta, &draw.beta),
                };
                let donors_sorted = SortedDonors::new(&predict(&cols, obs, obs_beta), j)?;
                let targets = predict(&cols, mis, mis_beta);
                if targets.iter().any(|t| !t.is_finite()) {
                    return Err(ImputeError::NotFinite { column: j });
                }
                Ok(mis
                    .par_iter()
                    .zip(targets.par_iter())
                    .with_min_len(2048)
                    .map(|(&r, &t)| y_obs[donors_sorted.pick(t, *donors, uniform(r))])
                    .collect())
            } else {
                let means = predict(&cols, mis, &draw.beta);
                let cells: Vec<f64> = mis
                    .par_iter()
                    .zip(means.par_iter())
                    .with_min_len(2048)
                    .map(|(&r, &m)| m + row_stream(seed, r).normal() * draw.sigma)
                    .collect();
                if cells.iter().any(|v| !v.is_finite()) {
                    return Err(ImputeError::NotFinite { column: j });
                }
                Ok(cells)
            }
        }
        Method::Logreg => {
            let cols = predictor_columns(data, completed, &spec.predictors[j]);
            let (x_aug, y_aug, w_aug, used) = augmented_problem(&cols, obs, &y_obs, &[0.0, 1.0]);
            out.augmented += usize::from(used);
            let fit = logistic_fit(x_aug.as_ref(), &y_aug, &w_aug, j)?;
            out.aliased += usize::from(fit.active.iter().any(|&a| !a));
            let mut stream = Stream::seeded(replicate_seed(seed, PARAMETER_ROW));
            let mut z = vec![0.0; fit.active.iter().filter(|&&a| a).count()];
            stream.fill_normals(&mut z);
            let beta = draw_coefficients(&fit.coefficients, &fit.active, &fit.covariance, &z, 1.0, j)?;
            let eta = predict(&cols, mis, &beta);
            Ok(mis
                .par_iter()
                .zip(eta.par_iter())
                .with_min_len(2048)
                .map(|(&r, &e)| f64::from(u8::from(uniform(r) <= expit(e))))
                .collect())
        }
        Method::Polyreg { proper } => {
            let Column::Nominal { levels, .. } = &data.columns()[j] else {
                unreachable!("validated");
            };
            let cols = predictor_columns(data, completed, &spec.predictors[j]);
            let outcomes: Vec<f64> = (0..*levels).map(|l| l as f64).collect();
            let (x_aug, y_aug, w_aug, used) = augmented_problem(&cols, obs, &y_obs, &outcomes);
            out.augmented += usize::from(used);
            let y_class: Vec<usize> = y_aug.iter().map(|&v| v as usize).collect();
            let fit = multinomial_fit(x_aug.as_ref(), &y_class, *levels, &w_aug)?;
            out.aliased += usize::from(fit.active.iter().any(|&a| !a));
            let coefficients = if *proper {
                let mut stream = Stream::seeded(replicate_seed(seed, PARAMETER_ROW));
                let mut z = vec![0.0; fit.n_free()];
                stream.fill_normals(&mut z);
                fit.draw(&z, j)?
            } else {
                fit.coefficients.clone()
            };
            let p = cols.len() + 1;
            Ok(mis
                .par_iter()
                .with_min_len(1024)
                .map(|&r| {
                    let mut x = Vec::with_capacity(p);
                    x.push(1.0);
                    x.extend(cols.iter().map(|c| c[r]));
                    let probabilities = class_probabilities(&coefficients, *levels, &x);
                    class_from_uniform(&probabilities, uniform(r)) as f64
                })
                .collect())
        }
    }
}

/// The observed rows of the design with the intercept, followed by the
/// augmentation rows, with the outcome and weight of each.
fn augmented_problem(
    cols: &[Vec<f64>],
    obs: &[usize],
    y_obs: &[f64],
    outcomes: &[f64],
) -> (epistat_linalg::Mat<f64>, Vec<f64>, Vec<f64>, bool) {
    let n_obs = obs.len();
    let base = design_matrix(cols, obs);
    let mut present: Vec<f64> = outcomes
        .iter()
        .copied()
        .filter(|o| y_obs.contains(o))
        .collect();
    present.sort_unstable_by(f64::total_cmp);
    match augmentation(cols, &present) {
        None => (base, y_obs.to_vec(), vec![1.0; n_obs], false),
        Some(aug) => {
            let p = cols.len();
            let x = epistat_linalg::Mat::from_fn(n_obs + aug.rows, p + 1, |i, k| {
                if i < n_obs {
                    base[(i, k)]
                } else if k == 0 {
                    1.0
                } else {
                    aug.x[(i - n_obs) * p + (k - 1)]
                }
            });
            let mut y = y_obs.to_vec();
            y.extend_from_slice(&aug.y);
            let mut w = vec![1.0; n_obs];
            w.extend(std::iter::repeat_n(aug.weight, aug.rows));
            (x, y, w, true)
        }
    }
}
