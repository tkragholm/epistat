//! The parametric nuisance models of [`super::att_missing_pre`]: designs with
//! scaled covariates, logistic regression by Newton's method and least
//! squares by the normal equations, fitted on a subset of the units and
//! predicting for another.

use std::collections::HashMap;

use epistat_linalg::cholesky::{Pivot, cholesky_lower};
use epistat_linalg::{MatRef, weighted_gram, xtv};
use epistat_rng::Stream;
use rayon::prelude::*;

use super::{
    MissingAssumption, MissingConfig, MissingError, MissingFolds, MissingModels, MissingNuisance,
    MissingPreData, NestedOutcome, compensated_sum,
};

/// A Cholesky pivot that leaves at most this share of a column's squared norm
/// marks the column as collinear with those before it.
const COLLINEAR: f64 = 1e-10;

/// A row-major design: an intercept, then the chosen covariates centred and
/// scaled over all units.
struct Design {
    data: Vec<f64>,
    cols: usize,
}

impl Design {
    fn view(&self) -> MatRef<'_, f64> {
        MatRef::from_row_major_slice(&self.data, self.data.len() / self.cols, self.cols)
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    fn predict(&self, i: usize, beta: &[f64]) -> f64 {
        let mut acc = 0.0;
        for (x, b) in self.row(i).iter().zip(beta) {
            acc += x * b;
        }
        acc
    }
}

/// Rows per block of the passes over the units. The blocks are processed in
/// parallel and their sums added in block order, so the results do not
/// depend on the number of threads.
const BLOCK: usize = 1 << 14;

fn block_range(n: usize, b: usize) -> std::ops::Range<usize> {
    b * BLOCK..n.min((b + 1) * BLOCK)
}

/// The sum of `f(i)` over `0..n`: compensated within each block of
/// [`BLOCK`] units, the block sums added in order with compensation.
fn block_sum(n: usize, f: impl Fn(usize) -> f64 + Sync) -> f64 {
    let partial: Vec<f64> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| compensated_sum(block_range(n, b).map(&f)))
        .collect();
    compensated_sum(partial.into_iter())
}

/// `out[i] = f(i)` for every unit, block by block in parallel.
fn fill(out: &mut [f64], f: impl Fn(usize) -> f64 + Sync) {
    out.par_chunks_mut(BLOCK)
        .enumerate()
        .for_each(|(b, chunk)| {
            for (k, v) in chunk.iter_mut().enumerate() {
                *v = f(b * BLOCK + k);
            }
        });
}

/// `out[i] = f(i)` for the units of `rows`, block by block in parallel; the
/// other entries are left as they are.
fn fill_where(out: &mut [f64], rows: &[bool], f: impl Fn(usize) -> f64 + Sync) {
    out.par_chunks_mut(BLOCK)
        .enumerate()
        .for_each(|(b, chunk)| {
            for (k, v) in chunk.iter_mut().enumerate() {
                let i = b * BLOCK + k;
                if rows[i] {
                    *v = f(i);
                }
            }
        });
}

/// The scaled design of the chosen columns, with `y1` appended when given.
fn design(
    covariates: MatRef<'_, f64>,
    columns: &[usize],
    y1: Option<&[f64]>,
) -> Result<Design, MissingError> {
    let n = covariates.nrows();
    let k = covariates.ncols();
    if let Some(&column) = columns.iter().find(|&&c| c >= k) {
        return Err(MissingError::Column { column, columns: k });
    }
    let q = columns.len() + usize::from(y1.is_some());
    let cols = q + 1;
    let value = |i: usize, j: usize| match (j < columns.len(), y1) {
        (true, _) => covariates[(i, columns[j])],
        (false, Some(y1)) => y1[i],
        (false, None) => unreachable!("only y1 follows the covariates"),
    };
    let mut centre = vec![0.0; q];
    let mut scale = vec![0.0; q];
    for j in 0..q {
        if let Some(unit) = (0..n).find(|&i| !value(i, j).is_finite()) {
            return Err(MissingError::NotFinite {
                what: "covariates",
                unit,
            });
        }
        let mean = block_sum(n, |i| value(i, j)) / n as f64;
        let var = block_sum(n, |i| {
            let d = value(i, j) - mean;
            d * d
        }) / n as f64;
        if var <= 0.0 || !var.is_finite() {
            return Err(MissingError::Constant {
                what: if j < columns.len() {
                    format!("covariate column {}", columns[j])
                } else {
                    "the post-treatment outcome".to_string()
                },
            });
        }
        centre[j] = mean;
        scale[j] = var.sqrt();
    }
    let mut data = vec![0.0; n * cols];
    data.par_chunks_mut(BLOCK * cols)
        .enumerate()
        .for_each(|(b, chunk)| {
            for (k, row) in chunk.chunks_mut(cols).enumerate() {
                let i = b * BLOCK + k;
                row[0] = 1.0;
                for j in 0..q {
                    row[j + 1] = (value(i, j) - centre[j]) / scale[j];
                }
            }
        });
    Ok(Design { data, cols })
}

/// Solves `G beta = b` for the Gram matrix of a fit, refusing a pivot that
/// marks a collinear column.
fn solve(
    gram: &epistat_linalg::Mat<f64>,
    b: &[f64],
    model: &'static str,
) -> Result<Vec<f64>, MissingError> {
    let p = b.len();
    let mut flat = vec![0.0; p * p];
    for r in 0..p {
        for c in 0..p {
            flat[r * p + c] = gram[(r, c)];
        }
    }
    let singular = MissingError::Fit {
        model,
        reason: "a covariate is collinear with the others among the units of the fit",
    };
    let lower = cholesky_lower(&flat, p, Pivot::Strict).ok_or_else(|| singular.clone())?;
    for j in 0..p {
        let pivot = lower[j * p + j];
        if pivot * pivot <= COLLINEAR * flat[j * p + j] {
            return Err(singular);
        }
    }
    let mut z = vec![0.0; p];
    for r in 0..p {
        let mut s = b[r];
        for c in 0..r {
            s -= lower[r * p + c] * z[c];
        }
        z[r] = s / lower[r * p + r];
    }
    let mut beta = vec![0.0; p];
    for r in (0..p).rev() {
        let mut s = z[r];
        for c in r + 1..p {
            s -= lower[c * p + r] * beta[c];
        }
        beta[r] = s / lower[r * p + r];
    }
    Ok(beta)
}

/// Least squares of `y` on the design over the units of `fit_rows`.
fn least_squares(
    design: &Design,
    y: &[f64],
    fit_rows: &[bool],
    model: &'static str,
) -> Result<Vec<f64>, MissingError> {
    let mut weights = vec![0.0; y.len()];
    fill(&mut weights, |i| f64::from(u8::from(fit_rows[i])));
    let mut weighted = vec![0.0; y.len()];
    fill(&mut weighted, |i| if fit_rows[i] { y[i] } else { 0.0 });
    let x = design.view();
    let gram = weighted_gram(x, &weights).map_err(|_| MissingError::Fit {
        model,
        reason: "the design could not be formed",
    })?;
    let rhs = xtv(x, &weighted);
    let b: Vec<f64> = (0..design.cols).map(|j| rhs[(j, 0)]).collect();
    solve(&gram, &b, model)
}

pub(super) fn expit(eta: f64) -> f64 {
    if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

/// The linear predictor `eta` and `exp(-|eta|)` of the units of `fit_rows`
/// at `beta`, written into `eta` and `damped`, and the log-likelihood
/// `sum y eta - log(1 + exp(eta))` over those units.
fn evaluate(
    design: &Design,
    y: &[f64],
    fit_rows: &[bool],
    beta: &[f64],
    eta: &mut [f64],
    damped: &mut [f64],
) -> f64 {
    let n = y.len();
    let partial: Vec<f64> = eta
        .par_chunks_mut(BLOCK)
        .zip(damped.par_chunks_mut(BLOCK))
        .enumerate()
        .map(|(b, (eta, damped))| {
            let range = block_range(n, b);
            for (k, i) in range.clone().enumerate() {
                if fit_rows[i] {
                    let e = design.predict(i, beta);
                    eta[k] = e;
                    damped[k] = (-e.abs()).exp();
                } else {
                    eta[k] = 0.0;
                    damped[k] = 0.0;
                }
            }
            compensated_sum(
                range
                    .enumerate()
                    .filter(|&(_, i)| fit_rows[i])
                    .map(|(k, i)| y[i] * eta[k] - (eta[k].max(0.0) + damped[k].ln_1p())),
            )
        })
        .collect();
    compensated_sum(partial.into_iter())
}

/// Logistic regression of the 0/1 `y` on the design over the units of
/// `fit_rows`, by Newton's method with step halving.
fn logistic(
    design: &Design,
    y: &[f64],
    fit_rows: &[bool],
    config: &MissingConfig,
    model: &'static str,
) -> Result<Vec<f64>, MissingError> {
    let n = y.len();
    let p = design.cols;
    let x = design.view();
    let mut beta = vec![0.0; p];
    let (mut eta, mut damped) = (vec![0.0; n], vec![0.0; n]);
    let (mut trial_eta, mut trial_damped) = (vec![0.0; n], vec![0.0; n]);
    let mut ll = evaluate(design, y, fit_rows, &beta, &mut eta, &mut damped);
    let mut weights = vec![0.0; n];
    let mut residual = vec![0.0; n];
    for _ in 0..config.max_iterations {
        // The fitted probability from `exp(-|eta|)`, without another
        // exponential: `1 / (1 + d)` for `eta >= 0`, `d / (1 + d)` below.
        weights
            .par_chunks_mut(BLOCK)
            .zip(residual.par_chunks_mut(BLOCK))
            .enumerate()
            .for_each(|(b, (w, r))| {
                for (k, i) in block_range(n, b).enumerate() {
                    if fit_rows[i] {
                        let d = damped[i];
                        let mu = if eta[i] >= 0.0 {
                            1.0 / (1.0 + d)
                        } else {
                            d / (1.0 + d)
                        };
                        w[k] = mu * (1.0 - mu);
                        r[k] = y[i] - mu;
                    } else {
                        w[k] = 0.0;
                        r[k] = 0.0;
                    }
                }
            });
        let gram = weighted_gram(x, &weights).map_err(|_| MissingError::Fit {
            model,
            reason: "the design could not be formed",
        })?;
        let score = xtv(x, &residual);
        let b: Vec<f64> = (0..p).map(|j| score[(j, 0)]).collect();
        let step = solve(&gram, &b, model)?;
        let largest = step.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
        if !largest.is_finite() {
            return Err(MissingError::Fit {
                model,
                reason: "the Newton step is not finite",
            });
        }
        // Halve until the likelihood does not fall by more than the rounding
        // of its compensated sum. In the quadratic region near the maximum
        // the change is below that rounding and the full step is taken.
        let mut factor = 1.0;
        let mut accepted = false;
        let mut trial = vec![0.0; p];
        for _ in 0..60 {
            for j in 0..p {
                trial[j] = beta[j] + factor * step[j];
            }
            let trial_ll = evaluate(
                design,
                y,
                fit_rows,
                &trial,
                &mut trial_eta,
                &mut trial_damped,
            );
            if trial_ll.is_finite() && trial_ll >= ll - 1e-12 * ll.abs() {
                beta.copy_from_slice(&trial);
                std::mem::swap(&mut eta, &mut trial_eta);
                std::mem::swap(&mut damped, &mut trial_damped);
                ll = trial_ll;
                accepted = true;
                break;
            }
            factor *= 0.5;
        }
        if !accepted {
            return Err(MissingError::Fit {
                model,
                reason: "step halving did not raise the likelihood",
            });
        }
        if largest <= config.tolerance {
            return Ok(beta);
        }
    }
    Err(MissingError::Fit {
        model,
        reason: "Newton's method did not converge, possibly because of separation",
    })
}

/// The units of the data, by role.
struct Roles {
    treated: Vec<bool>,
    observed: Vec<bool>,
    y0: Vec<f64>,
    y1: Vec<f64>,
    a: Vec<f64>,
    r: Vec<f64>,
}

/// The designs of the models, each distinct set of columns built once.
struct Designs {
    store: Vec<Design>,
    propensity: usize,
    missingness: usize,
    pre: usize,
    post: usize,
    nested: Option<usize>,
}

impl Designs {
    fn new(
        data: &MissingPreData<'_>,
        covariates: MatRef<'_, f64>,
        assumption: MissingAssumption,
        models: &MissingModels<'_>,
    ) -> Result<Self, MissingError> {
        let y1 = (assumption == MissingAssumption::MarXY1).then_some(data.outcome_post);
        let mut built: HashMap<(Vec<usize>, bool), usize> = HashMap::new();
        let mut store: Vec<Design> = Vec::new();
        let mut get = |columns: &[usize], with_y1: bool| -> Result<usize, MissingError> {
            let key = (columns.to_vec(), with_y1);
            if let Some(&index) = built.get(&key) {
                return Ok(index);
            }
            store.push(design(
                covariates,
                columns,
                if with_y1 { y1 } else { None },
            )?);
            built.insert(key, store.len() - 1);
            Ok(store.len() - 1)
        };
        let with_y1 = y1.is_some();
        let propensity = get(models.propensity, false)?;
        let missingness = get(models.missingness, with_y1)?;
        let pre = get(models.outcome, with_y1)?;
        let post = get(models.outcome, false)?;
        let nested = if with_y1 {
            Some(get(models.nested, false)?)
        } else {
            None
        };
        Ok(Self {
            store,
            propensity,
            missingness,
            pre,
            post,
            nested,
        })
    }

    fn propensity(&self) -> &Design {
        &self.store[self.propensity]
    }

    fn missingness(&self) -> &Design {
        &self.store[self.missingness]
    }

    fn pre(&self) -> &Design {
        &self.store[self.pre]
    }

    fn post(&self) -> &Design {
        &self.store[self.post]
    }

    fn nested(&self) -> Option<&Design> {
        self.nested.map(|index| &self.store[index])
    }
}

/// The fold of each unit and its half within the training units, from a
/// random order of the units, or of the clusters when there are clusters.
fn fold_assignment(
    data: &MissingPreData<'_>,
    folds: usize,
    seed: u64,
) -> Result<(Vec<usize>, Vec<bool>), MissingError> {
    let n = data.treated.len();
    let groups = data.clusters.map_or(n, epistat_inference::Clusters::count);
    if folds < 2 || folds > groups {
        return Err(MissingError::Config(
            "the number of folds must be at least 2 and at most the number of units or clusters",
        ));
    }
    let mut order: Vec<usize> = (0..groups).collect();
    Stream::seeded(seed).shuffle(&mut order);
    let mut position = vec![0; groups];
    for (pos, &g) in order.iter().enumerate() {
        position[g] = pos;
    }
    let group = |i: usize| data.clusters.map_or(i, |c| c.of()[i]);
    let fold = (0..n).map(|i| position[group(i)] % folds).collect();
    let half = (0..n)
        .map(|i| (position[group(i)] / folds) % 2 == 1)
        .collect();
    Ok((fold, half))
}

/// Fits the nuisance models on all units, or fold by fold, and returns the
/// predictions of every unit with the number of folds.
pub(super) fn fit_nuisance(
    data: &MissingPreData<'_>,
    covariates: MatRef<'_, f64>,
    assumption: MissingAssumption,
    models: &MissingModels<'_>,
    folds: MissingFolds,
    config: &MissingConfig,
) -> Result<(MissingNuisance, usize), MissingError> {
    let n = data.treated.len();
    let roles = Roles {
        treated: data.treated.to_vec(),
        observed: data.outcome_pre.iter().map(Option::is_some).collect(),
        y0: data.outcome_pre.iter().map(|v| v.unwrap_or(0.0)).collect(),
        y1: data.outcome_post.to_vec(),
        a: data
            .treated
            .iter()
            .map(|&t| f64::from(u8::from(t)))
            .collect(),
        r: data
            .outcome_pre
            .iter()
            .map(|v| f64::from(u8::from(v.is_some())))
            .collect(),
    };
    for (arm, name) in [(true, "treated unit"), (false, "control")] {
        if !(0..n).any(|i| roles.treated[i] == arm && roles.observed[i]) {
            return Err(MissingError::NoObservedPre(name));
        }
    }
    let designs = Designs::new(data, covariates, assumption, models)?;
    let mut out = MissingNuisance {
        propensity: vec![0.0; n],
        gamma_treated: vec![0.0; n],
        gamma_control: vec![0.0; n],
        pre_treated: vec![0.0; n],
        post_control: vec![0.0; n],
        pre_control: vec![0.0; n],
        nested_control: (assumption == MissingAssumption::MarXY1).then(|| vec![0.0; n]),
    };
    match folds {
        MissingFolds::None => {
            let all = vec![true; n];
            fit_fold(&roles, &designs, &all, None, &all, config, &mut out)?;
            Ok((out, 1))
        }
        MissingFolds::CrossFit { folds, seed } => {
            let (fold, half) = fold_assignment(data, folds, seed)?;
            for j in 0..folds {
                let train: Vec<bool> = fold.iter().map(|&f| f != j).collect();
                let eval: Vec<bool> = fold.iter().map(|&f| f == j).collect();
                fit_fold(
                    &roles,
                    &designs,
                    &train,
                    Some(&half),
                    &eval,
                    config,
                    &mut out,
                )?;
            }
            Ok((out, folds))
        }
    }
}

fn and(a: &[bool], b: impl Fn(usize) -> bool) -> Vec<bool> {
    a.iter().enumerate().map(|(i, &x)| x && b(i)).collect()
}

/// `gamma` of one arm on the units of `eval`: 1 when every unit of the arm
/// among `train` has `Y0`, otherwise the logistic fit.
fn missingness(
    roles: &Roles,
    design: &Design,
    train: &[bool],
    arm: bool,
    config: &MissingConfig,
    model: &'static str,
) -> Result<Option<Vec<f64>>, MissingError> {
    let rows = and(train, |i| roles.treated[i] == arm);
    if !(0..rows.len()).any(|i| rows[i] && roles.observed[i]) {
        return Err(MissingError::Fit {
            model,
            reason: "no unit of the arm among the training units has the pre-treatment outcome",
        });
    }
    if (0..rows.len()).all(|i| !rows[i] || roles.observed[i]) {
        return Ok(None);
    }
    logistic(design, &roles.r, &rows, config, model).map(Some)
}

/// Fits every model on `train` and writes the predictions for `eval`. Under
/// `MarXY1` with `half` given, the pseudo-outcome of the nested regression
/// comes from models fitted on the training units of the first half, and the
/// nested regression is fitted on the controls of the second.
fn fit_fold(
    roles: &Roles,
    designs: &Designs,
    train: &[bool],
    half: Option<&[bool]>,
    eval: &[bool],
    config: &MissingConfig,
    out: &mut MissingNuisance,
) -> Result<(), MissingError> {
    let n = train.len();
    let propensity = logistic(designs.propensity(), &roles.a, train, config, "propensity")?;
    let gamma1 = missingness(
        roles,
        designs.missingness(),
        train,
        true,
        config,
        "missingness among the treated",
    )?;
    let gamma0 = missingness(
        roles,
        designs.missingness(),
        train,
        false,
        config,
        "missingness among the controls",
    )?;
    let pre1 = least_squares(
        designs.pre(),
        &roles.y0,
        &and(train, |i| roles.treated[i] && roles.observed[i]),
        "pre-treatment outcome among the treated",
    )?;
    let post0 = least_squares(
        designs.post(),
        &roles.y1,
        &and(train, |i| !roles.treated[i]),
        "post-treatment outcome among the controls",
    )?;
    let pre0_rows = and(train, |i| !roles.treated[i] && roles.observed[i]);
    let pre0 = least_squares(
        designs.pre(),
        &roles.y0,
        &pre0_rows,
        "pre-treatment outcome among the controls",
    )?;
    let gamma_at = |beta: &Option<Vec<f64>>, i: usize| {
        beta.as_ref()
            .map_or(1.0, |b| expit(designs.missingness().predict(i, b)))
    };
    fill_where(&mut out.propensity, eval, |i| {
        expit(designs.propensity().predict(i, &propensity))
    });
    fill_where(&mut out.gamma_treated, eval, |i| gamma_at(&gamma1, i));
    fill_where(&mut out.gamma_control, eval, |i| gamma_at(&gamma0, i));
    fill_where(&mut out.pre_treated, eval, |i| {
        designs.pre().predict(i, &pre1)
    });
    fill_where(&mut out.post_control, eval, |i| {
        designs.post().predict(i, &post0)
    });
    fill_where(&mut out.pre_control, eval, |i| {
        designs.pre().predict(i, &pre0)
    });
    let (Some(nested_design), Some(nested_out)) = (designs.nested(), out.nested_control.as_mut())
    else {
        return Ok(());
    };
    // The models of the pseudo-outcome, and the rows of the nested fit.
    let (pseudo_pre0, pseudo_gamma0, nested_rows) = match half {
        None => (pre0, gamma0, and(train, |i| !roles.treated[i])),
        Some(half) => {
            let first = and(train, |i| !half[i]);
            let pre0 = least_squares(
                designs.pre(),
                &roles.y0,
                &and(&first, |i| !roles.treated[i] && roles.observed[i]),
                "pre-treatment outcome among the controls of the nested split",
            )?;
            let gamma0 = missingness(
                roles,
                designs.missingness(),
                &first,
                false,
                config,
                "missingness among the controls of the nested split",
            )?;
            (pre0, gamma0, and(train, |i| half[i] && !roles.treated[i]))
        }
    };
    if !nested_rows.iter().any(|&r| r) {
        return Err(MissingError::Fit {
            model: "nested regression",
            reason: "no controls among its training units",
        });
    }
    let mut pseudo = vec![0.0; n];
    for i in (0..n).filter(|&i| nested_rows[i]) {
        let m = designs.pre().predict(i, &pseudo_pre0);
        pseudo[i] = if config.nested_outcome == NestedOutcome::Augmented && roles.observed[i] {
            let gamma = gamma_at(&pseudo_gamma0, i);
            super::check_gamma(i, gamma, config)?;
            m + (roles.y0[i] - m) / gamma
        } else {
            m
        };
    }
    let eta = least_squares(nested_design, &pseudo, &nested_rows, "nested regression")?;
    fill_where(nested_out, eval, |i| nested_design.predict(i, &eta));
    Ok(())
}
