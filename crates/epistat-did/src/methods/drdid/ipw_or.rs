//! Two-period inverse probability weighting and outcome regression `DiD`.
//!
//! The four estimators `DRDID` 1.3.0 provides beside the doubly robust ones,
//! and that `did::att_gt` calls for `est_method = "ipw"` and `"reg"`:
//! `std_ipw_did_panel`, `reg_did_panel`, `std_ipw_did_rc` and `reg_did_rc`;
//! and, from the same pieces, the traditional doubly robust repeated
//! cross-section estimator `drdid_rc1`, which combines the IPW weights with
//! the outcome residuals `Y - m_T(X)` and adds the projections of both
//! regressions and of the propensity score to its influence function.
//!
//! # Algorithm
//!
//! Weights are normalised to mean one, `w_i = v_i / mean(v)`, so every mean
//! below is over the whole sample. `D_i` is treatment, `T_i` the post period
//! of a repeated cross-section, `X_i` the design row with its intercept.
//!
//! The propensity score `p(X)` is the logistic regression of `D` on `X` with
//! prior weights `w`, by maximum likelihood, capped at `1 - propensity_clip`.
//! Comparators with `p >= 0.995` are trimmed (`DRDID`'s `trim.level`); treated
//! units never are. Its asymptotic linear representation is
//! `l_i = n w_i (D_i - p_i) X_i' H^-1` with `H = sum_i w_i p_i (1 - p_i) X_i X_i'`.
//!
//! Panel IPW, on `dY = Y_1 - Y_0`, with `a_i = trim_i w_i D_i` and
//! `b_i = trim_i w_i p_i (1 - D_i) / (1 - p_i)`:
//!
//! ```text
//! ATT    = mean(a dY) / mean(a) - mean(b dY) / mean(b) = eta_1 - eta_0
//! psi_i  = a_i (dY_i - eta_1) / mean(a)
//!        - [ b_i (dY_i - eta_0) + l_i M ] / mean(b),
//! M      = sum_i b_i (dY_i - eta_0) X_i / n
//! ```
//!
//! Repeated cross-section IPW splits `a` and `b` by period, `a_t = a T`,
//! `a_0 = a (1 - T)` and so on, with each cell's normalised mean
//! `att_c = mean(c Y) / mean(c)`:
//!
//! ```text
//! ATT    = (att_{a1} - att_{a0}) - (att_{b1} - att_{b0})
//! psi_i  = sum over the four cells of +-c_i (Y_i - att_c) / mean(c)
//!        - l_i (M_1 - M_0),
//! M_t    = sum_i b_{t,i} (Y_i - att_{bt}) X_i / (n mean(b_t))
//! ```
//!
//! Panel outcome regression fits `dY ~ X` by weighted least squares among the
//! comparators, `m(X) = X beta`, with representation
//! `r_i = w_i (1 - D_i) (dY_i - m_i) X_i' B^-1`, `B = sum_i w_i (1 - D_i) X_i X_i' / n`.
//! With `a_i = w_i D_i`,
//!
//! ```text
//! ATT    = mean(a dY) / mean(a) - mean(a m) / mean(a)
//! psi_i  = a_i (dY_i - eta_1) / mean(a)
//!        - [ a_i (m_i - eta_0) + r_i N ] / mean(a),   N = sum_i a_i X_i / n
//! ```
//!
//! Repeated cross-section outcome regression fits `Y ~ X` among the
//! comparators of each period, `m_0` and `m_1`, averages `m_1 - m_0` over all
//! treated rows, and subtracts both regressions' representations projected on
//! `N`:
//!
//! ```text
//! ATT    = (mean(a_1 Y)/mean(a_1) - mean(a_0 Y)/mean(a_0))
//!        - mean(a (m_1 - m_0)) / mean(a)
//! psi_i  = treated terms - [ a_i (m_1 - m_0 - eta_0)_i + (r_1 - r_0)_i N ] / mean(a)
//! ```
//!
//! The standard error is `sqrt(mean((psi - mean(psi))^2) / n)`, `DRDID`'s
//! `sd(psi) sqrt(n - 1) / n`. Each projection `l M` is formed as
//! `n w_i (D_i - p_i) X_i' (H^-1 M)`, one solve per term rather than the
//! `n x k` representation.
//!
//! The columns of `X` are pruned to those the fit can identify: over every
//! row for the propensity score, and within the comparators (of each period,
//! for a repeated cross-section) for the outcome regression. A full-rank
//! design is left as it is.
//!
//! # References
//!
//! 1. Sant'Anna PHC, Zhao J. Doubly robust difference-in-differences
//!    estimators. J Econom 2020;219(1):101-22.
//!    doi:10.1016/j.jeconom.2020.06.003
//! 2. Abadie A. Semiparametric difference-in-differences estimators. Rev Econ
//!    Stud 2005;72(1):1-19. doi:10.1111/0034-6527.00321
//! 3. Heckman JJ, Ichimura H, Todd PE. Matching as an econometric evaluation
//!    estimator: evidence from evaluating a job training programme. Rev Econ
//!    Stud 1997;64(4):605-54. doi:10.2307/2971733

use epistat_linalg::{MatRef, weighted_gram};

use super::design::{DEPENDENCE_TOLERANCE, prune_design};
use super::moments::normalize_weights_to_n;
use super::panel::{
    LogisticIrlsInputs, LogisticIrlsSettings, PanelFlatInput, fit_logistic_irls,
    fit_weighted_linear, predict_linear, predict_logistic_scores, validate_drdid_config,
};
use crate::estimators::common::linalg::solve_spd_system;
use crate::inference::standard_error_from_influence;
use crate::types::{DidCell, DrDidConfig, DrDidError, DrDidEstimate, DrDidRepeatedObservation};
use crate::util::usize_to_f64;

/// Comparators at or above this propensity score are trimmed, as at
/// `DRDID`'s default `trim.level`.
const TRIM_LEVEL: f64 = 0.995;

/// Which nuisance model an estimator fits, which decides the rows a design
/// column must be identified on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Nuisance {
    Propensity,
    Outcome,
}

/// One two-period sample, validated, with weights normalised to mean one and
/// the design pruned for the estimator's nuisance model.
struct Sample {
    design: Vec<f64>,
    feature_count: usize,
    treated: Vec<f64>,
    /// The post-period indicator of a repeated cross-section; empty on a panel.
    post: Vec<f64>,
    /// `dY` on a panel, `Y` on a repeated cross-section.
    outcome: Vec<f64>,
    weights: Vec<f64>,
    treated_n: usize,
    control_n: usize,
    total_weight: f64,
    columns_dropped: usize,
}

impl Sample {
    fn n(&self) -> usize {
        self.treated.len()
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.design[i * self.feature_count..(i + 1) * self.feature_count]
    }
}

/// Validate the rows and build the sample. `post` is empty for a panel.
fn sample(
    treated: Vec<f64>,
    post: Vec<f64>,
    outcome: Vec<f64>,
    raw_weights: &[f64],
    design: Vec<f64>,
    feature_count: usize,
    nuisance: Nuisance,
) -> Result<Sample, DrDidError> {
    let n = treated.len();
    if n == 0 {
        return Err(DrDidError::EmptyInput);
    }
    for value in &outcome {
        if !value.is_finite() {
            return Err(DrDidError::InvalidOutcome { value: *value });
        }
    }
    for &weight in raw_weights {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(DrDidError::InvalidWeight { value: weight });
        }
    }
    for value in &design {
        if !value.is_finite() {
            return Err(DrDidError::InvalidCovariate { value: *value });
        }
    }
    let treated_n = treated.iter().filter(|&&d| d > 0.5).count();
    let control_n = n - treated_n;
    if treated_n == 0 {
        return Err(DrDidError::NoTreated);
    }
    if control_n == 0 {
        return Err(DrDidError::NoControl);
    }
    if !post.is_empty() {
        for (cell, is_treated, is_post) in [
            (DidCell::TreatedPre, true, false),
            (DidCell::TreatedPost, true, true),
            (DidCell::ControlPre, false, false),
            (DidCell::ControlPost, false, true),
        ] {
            let present =
                (0..n).any(|i| (treated[i] > 0.5) == is_treated && (post[i] > 0.5) == is_post);
            if !present {
                return Err(DrDidError::MissingCell { cell });
            }
        }
    }
    let total_weight = raw_weights.iter().sum::<f64>();
    let weights = normalize_weights_to_n(raw_weights)
        .map_err(|_| DrDidError::InvalidConfig("sum of weights must be > 0".to_string()))?;

    // The rows the nuisance model reads, and the groups a column must be
    // identified in: every row for the propensity score; the comparators, by
    // period on a repeated cross-section, for the outcome regression.
    let (rows, groups): (Vec<usize>, Vec<usize>) = match nuisance {
        Nuisance::Propensity => ((0..n).collect(), vec![0; n]),
        Nuisance::Outcome => (0..n)
            .filter(|&i| treated[i] < 0.5)
            .map(|i| (i, usize::from(post.get(i).is_some_and(|&t| t > 0.5))))
            .unzip(),
    };
    let mut fit_design = Vec::with_capacity(rows.len() * feature_count);
    let mut fit_weights = Vec::with_capacity(rows.len());
    for &i in &rows {
        fit_design.extend_from_slice(&design[i * feature_count..(i + 1) * feature_count]);
        fit_weights.push(raw_weights[i]);
    }
    let pruned = prune_design(
        &fit_design,
        feature_count,
        &groups,
        &fit_weights,
        DEPENDENCE_TOLERANCE,
    )?;
    let (design, kept_count) = if pruned.dropped == 0 {
        (design, feature_count)
    } else {
        let kept: Vec<usize> = (0..feature_count).filter(|&c| pruned.kept[c]).collect();
        let mut selected = Vec::with_capacity(n * kept.len());
        for i in 0..n {
            let x = &design[i * feature_count..(i + 1) * feature_count];
            selected.extend(kept.iter().map(|&c| x[c]));
        }
        (selected, kept.len())
    };

    Ok(Sample {
        design,
        feature_count: kept_count,
        treated,
        post,
        outcome,
        weights,
        treated_n,
        control_n,
        total_weight,
        columns_dropped: pruned.dropped,
    })
}

fn panel_flat_sample(input: PanelFlatInput<'_>, nuisance: Nuisance) -> Result<Sample, DrDidError> {
    let n = input.treated.len();
    let feature_count = input.feature_count.max(1);
    if input.delta_outcome.len() != n
        || input.weight.len() != n
        || input.design_matrix_flat.len() != n * feature_count
    {
        return Err(DrDidError::InconsistentCovariateCount {
            expected: n * feature_count,
            actual: input.design_matrix_flat.len(),
        });
    }
    sample(
        input
            .treated
            .iter()
            .map(|&d| f64::from(u8::from(d)))
            .collect(),
        Vec::new(),
        input.delta_outcome.to_vec(),
        input.weight,
        input.design_matrix_flat.to_vec(),
        feature_count,
        nuisance,
    )
}

/// A repeated cross-section sample, with the intercept prepended to the
/// covariates as the traditional repeated cross-section estimator does.
fn repeated_sample(
    observations: &[DrDidRepeatedObservation],
    nuisance: Nuisance,
) -> Result<Sample, DrDidError> {
    let covariate_count = observations.first().map_or(0, |row| row.covariates.len());
    let feature_count = covariate_count + 1;
    let mut design = Vec::with_capacity(observations.len() * feature_count);
    for row in observations {
        if row.covariates.len() != covariate_count {
            return Err(DrDidError::InconsistentCovariateCount {
                expected: covariate_count,
                actual: row.covariates.len(),
            });
        }
        design.push(1.0);
        design.extend_from_slice(&row.covariates);
    }
    let weights: Vec<f64> = observations.iter().map(|row| row.weight).collect();
    sample(
        observations
            .iter()
            .map(|row| f64::from(u8::from(row.treated)))
            .collect(),
        observations
            .iter()
            .map(|row| f64::from(u8::from(row.post_period)))
            .collect(),
        observations.iter().map(|row| row.outcome).collect(),
        &weights,
        design,
        feature_count,
        nuisance,
    )
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / usize_to_f64(values.len())
}

/// `sum_i c_i X_i / n`.
fn moment(sample: &Sample, c: &[f64]) -> Vec<f64> {
    let n_f = usize_to_f64(sample.n());
    let mut out = vec![0.0; sample.feature_count];
    for (i, &ci) in c.iter().enumerate() {
        for (slot, x) in out.iter_mut().zip(sample.row(i)) {
            *slot += ci * x;
        }
    }
    for slot in &mut out {
        *slot /= n_f;
    }
    out
}

/// `sum_i c_i X_i X_i'`, row-major.
fn gram(sample: &Sample, c: &[f64]) -> Result<Vec<f64>, DrDidError> {
    let k = sample.feature_count;
    let design = MatRef::from_row_major_slice(&sample.design, sample.n(), k);
    let matrix = weighted_gram(design, c).map_err(|_| DrDidError::SingularSystem)?;
    let mut out = vec![0.0; k * k];
    for row in 0..k {
        for col in 0..k {
            out[row * k + col] = matrix[(row, col)];
        }
    }
    Ok(out)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>()
}

/// The fitted propensity score of every row, capped at `1 - propensity_clip`.
fn propensity(sample: &Sample, config: DrDidConfig) -> Result<Vec<f64>, DrDidError> {
    let fit = fit_logistic_irls(
        LogisticIrlsInputs {
            design_matrix_flat: &sample.design,
            feature_count: sample.feature_count,
            treated_indicator: &sample.treated,
            normalized_weights: &sample.weights,
        },
        LogisticIrlsSettings {
            max_iter: config.max_iter,
            tol: config.tol,
        },
    )?;
    Ok(predict_logistic_scores(
        &sample.design,
        sample.feature_count,
        &fit.coefficients,
        config.propensity_clip,
    ))
}

/// `l_i M` for every row, with `l` the propensity score's asymptotic linear
/// representation: `n w_i (D_i - p_i) X_i' H^-1 M`.
fn propensity_projection(
    sample: &Sample,
    scores: &[f64],
    m: &[f64],
) -> Result<Vec<f64>, DrDidError> {
    let hessian_weights: Vec<f64> = (0..sample.n())
        .map(|i| scores[i] * (1.0 - scores[i]) * sample.weights[i])
        .collect();
    let hessian = gram(sample, &hessian_weights)?;
    let solved = solve_spd_system(&hessian, m).map_err(|_| DrDidError::SingularSystem)?;
    let n_f = usize_to_f64(sample.n());
    Ok((0..sample.n())
        .map(|i| {
            n_f * sample.weights[i] * (sample.treated[i] - scores[i]) * dot(sample.row(i), &solved)
        })
        .collect())
}

/// Weighted least squares of the outcome on the design over the rows where
/// `fit_weight` is positive, and its prediction for every row.
fn outcome_fit(sample: &Sample, fit_weight: &[f64], ridge: f64) -> Result<Vec<f64>, DrDidError> {
    let k = sample.feature_count;
    let rows: Vec<usize> = (0..sample.n()).filter(|&i| fit_weight[i] > 0.0).collect();
    if rows.len() < k {
        return Err(DrDidError::SingularSystem);
    }
    let mut design = Vec::with_capacity(rows.len() * k);
    let mut outcome = Vec::with_capacity(rows.len());
    let mut weights = Vec::with_capacity(rows.len());
    for &i in &rows {
        design.extend_from_slice(sample.row(i));
        outcome.push(sample.outcome[i]);
        weights.push(fit_weight[i]);
    }
    let beta = fit_weighted_linear(&design, k, &outcome, &weights, ridge)?;
    Ok(predict_linear(&sample.design, k, &beta))
}

/// `r_i N` for every row, with `r` the asymptotic linear representation of
/// the regression [`outcome_fit`] fitted with `fit_weight`, whose
/// predictions are `prediction`.
fn outcome_projection(
    sample: &Sample,
    fit_weight: &[f64],
    prediction: &[f64],
    n_vector: &[f64],
) -> Result<Vec<f64>, DrDidError> {
    let n_f = usize_to_f64(sample.n());
    let mut bread = gram(sample, fit_weight)?;
    for value in &mut bread {
        *value /= n_f;
    }
    let solved = solve_spd_system(&bread, n_vector).map_err(|_| DrDidError::SingularSystem)?;
    Ok((0..sample.n())
        .map(|i| fit_weight[i] * (sample.outcome[i] - prediction[i]) * dot(sample.row(i), &solved))
        .collect())
}

/// [`outcome_fit`] and its [`outcome_projection`] on `n_vector`.
fn outcome_regression(
    sample: &Sample,
    fit_weight: &[f64],
    n_vector: &[f64],
    ridge: f64,
) -> Result<(Vec<f64>, Vec<f64>), DrDidError> {
    let prediction = outcome_fit(sample, fit_weight, ridge)?;
    let projection = outcome_projection(sample, fit_weight, &prediction, n_vector)?;
    Ok((prediction, projection))
}

fn estimate(
    sample: &Sample,
    att: f64,
    influence_function: Vec<f64>,
    config: DrDidConfig,
) -> DrDidEstimate {
    let se = standard_error_from_influence(&influence_function);
    let (ci_low, ci_high) = super::confidence_interval(att, se, &influence_function, config);
    DrDidEstimate {
        att,
        se,
        ci_low,
        ci_high,
        treated_n: sample.treated_n,
        control_n: sample.control_n,
        total_weight: sample.total_weight,
        influence_function,
        design_columns_dropped: sample.columns_dropped,
    }
}

fn ipw_panel(sample: &Sample, config: DrDidConfig) -> Result<DrDidEstimate, DrDidError> {
    let n = sample.n();
    let scores = propensity(sample, config)?;
    let trim: Vec<f64> = (0..n)
        .map(|i| {
            if sample.treated[i] > 0.5 || scores[i] < TRIM_LEVEL {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let a: Vec<f64> = (0..n)
        .map(|i| trim[i] * sample.weights[i] * sample.treated[i])
        .collect();
    let b: Vec<f64> = (0..n)
        .map(|i| {
            trim[i] * sample.weights[i] * scores[i] * (1.0 - sample.treated[i]) / (1.0 - scores[i])
        })
        .collect();
    let (mean_a, mean_b) = (mean(&a), mean(&b));
    if mean_a <= 0.0 {
        return Err(DrDidError::NoTreated);
    }
    if mean_b <= 0.0 {
        return Err(DrDidError::NoControl);
    }
    let dy = &sample.outcome;
    let a_dy: Vec<f64> = (0..n).map(|i| a[i] * dy[i]).collect();
    let b_dy: Vec<f64> = (0..n).map(|i| b[i] * dy[i]).collect();
    let eta_treated = mean(&a_dy) / mean_a;
    let eta_control = mean(&b_dy) / mean_b;
    let att = eta_treated - eta_control;

    let centred: Vec<f64> = (0..n).map(|i| b[i] * (dy[i] - eta_control)).collect();
    let m = moment(sample, &centred);
    let ps_term = propensity_projection(sample, &scores, &m)?;
    let influence = (0..n)
        .map(|i| {
            let treated_part = (a_dy[i] - a[i] * eta_treated) / mean_a;
            let control_part = (b_dy[i] - b[i] * eta_control + ps_term[i]) / mean_b;
            treated_part - control_part
        })
        .collect();
    Ok(estimate(sample, att, influence, config))
}

fn ipw_repeated(sample: &Sample, config: DrDidConfig) -> Result<DrDidEstimate, DrDidError> {
    let n = sample.n();
    let scores = propensity(sample, config)?;
    let (d, t, y) = (&sample.treated, &sample.post, &sample.outcome);
    let trim: Vec<f64> = (0..n)
        .map(|i| {
            if d[i] > 0.5 || scores[i] < TRIM_LEVEL {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let treated_weight = |i: usize| trim[i] * sample.weights[i] * d[i];
    let control_weight =
        |i: usize| trim[i] * sample.weights[i] * scores[i] * (1.0 - d[i]) / (1.0 - scores[i]);
    let a0: Vec<f64> = (0..n).map(|i| treated_weight(i) * (1.0 - t[i])).collect();
    let a1: Vec<f64> = (0..n).map(|i| treated_weight(i) * t[i]).collect();
    let b0: Vec<f64> = (0..n).map(|i| control_weight(i) * (1.0 - t[i])).collect();
    let b1: Vec<f64> = (0..n).map(|i| control_weight(i) * t[i]).collect();

    // Each cell's `eta_i = c_i Y_i / mean(c)`, its mean `att_c`, and its
    // influence `eta_i - c_i att_c / mean(c)`.
    let cell = |c: &[f64], missing: DidCell| -> Result<(f64, f64, Vec<f64>), DrDidError> {
        let mc = mean(c);
        if mc <= 0.0 {
            return Err(DrDidError::MissingCell { cell: missing });
        }
        let eta: Vec<f64> = (0..n).map(|i| c[i] * y[i] / mc).collect();
        let att_c = mean(&eta);
        let influence = (0..n).map(|i| eta[i] - c[i] * att_c / mc).collect();
        Ok((mc, att_c, influence))
    };
    let (_, treated_pre, treated_pre_psi) = cell(&a0, DidCell::TreatedPre)?;
    let (_, treated_post, treated_post_psi) = cell(&a1, DidCell::TreatedPost)?;
    let (control_pre_weight, control_pre, control_pre_psi) = cell(&b0, DidCell::ControlPre)?;
    let (control_post_weight, control_post, control_post_psi) = cell(&b1, DidCell::ControlPost)?;
    let att = (treated_post - treated_pre) - (control_post - control_pre);

    let centred =
        |c: &[f64], att_c: f64| -> Vec<f64> { (0..n).map(|i| c[i] * (y[i] - att_c)).collect() };
    let moment_pre = moment(sample, &centred(&b0, control_pre));
    let moment_post = moment(sample, &centred(&b1, control_post));
    let m: Vec<f64> = (0..sample.feature_count)
        .map(|j| moment_post[j] / control_post_weight - moment_pre[j] / control_pre_weight)
        .collect();
    let ps_term = propensity_projection(sample, &scores, &m)?;
    let influence = (0..n)
        .map(|i| {
            let treated_part = treated_post_psi[i] - treated_pre_psi[i];
            let control_part = control_post_psi[i] - control_pre_psi[i] + ps_term[i];
            treated_part - control_part
        })
        .collect();
    Ok(estimate(sample, att, influence, config))
}

fn or_panel(sample: &Sample, config: DrDidConfig) -> Result<DrDidEstimate, DrDidError> {
    let n = sample.n();
    let (d, dy) = (&sample.treated, &sample.outcome);
    let a: Vec<f64> = (0..n).map(|i| sample.weights[i] * d[i]).collect();
    let mean_a = mean(&a);
    if mean_a <= 0.0 {
        return Err(DrDidError::NoTreated);
    }
    let comparator: Vec<f64> = (0..n).map(|i| sample.weights[i] * (1.0 - d[i])).collect();
    let n_vector = moment(sample, &a);
    let (prediction, or_term) = outcome_regression(sample, &comparator, &n_vector, config.ridge)?;

    let a_dy: Vec<f64> = (0..n).map(|i| a[i] * dy[i]).collect();
    let a_m: Vec<f64> = (0..n).map(|i| a[i] * prediction[i]).collect();
    let eta_treated = mean(&a_dy) / mean_a;
    let eta_control = mean(&a_m) / mean_a;
    let att = eta_treated - eta_control;
    let influence = (0..n)
        .map(|i| {
            let treated_part = (a_dy[i] - a[i] * eta_treated) / mean_a;
            let control_part = (a_m[i] - a[i] * eta_control + or_term[i]) / mean_a;
            treated_part - control_part
        })
        .collect();
    Ok(estimate(sample, att, influence, config))
}

fn or_repeated(sample: &Sample, config: DrDidConfig) -> Result<DrDidEstimate, DrDidError> {
    let n = sample.n();
    let (d, t, y) = (&sample.treated, &sample.post, &sample.outcome);
    let a: Vec<f64> = (0..n).map(|i| sample.weights[i] * d[i]).collect();
    let a0: Vec<f64> = (0..n).map(|i| a[i] * (1.0 - t[i])).collect();
    let a1: Vec<f64> = (0..n).map(|i| a[i] * t[i]).collect();
    let (mean_a, mean_a0, mean_a1) = (mean(&a), mean(&a0), mean(&a1));
    if mean_a0 <= 0.0 {
        return Err(DrDidError::MissingCell {
            cell: DidCell::TreatedPre,
        });
    }
    if mean_a1 <= 0.0 {
        return Err(DrDidError::MissingCell {
            cell: DidCell::TreatedPost,
        });
    }
    let comparator_pre: Vec<f64> = (0..n)
        .map(|i| sample.weights[i] * (1.0 - d[i]) * (1.0 - t[i]))
        .collect();
    let comparator_post: Vec<f64> = (0..n)
        .map(|i| sample.weights[i] * (1.0 - d[i]) * t[i])
        .collect();
    let n_vector = moment(sample, &a);
    let (prediction_pre, or_term_pre) =
        outcome_regression(sample, &comparator_pre, &n_vector, config.ridge)?;
    let (prediction_post, or_term_post) =
        outcome_regression(sample, &comparator_post, &n_vector, config.ridge)?;

    let a0_y: Vec<f64> = (0..n).map(|i| a0[i] * y[i]).collect();
    let a1_y: Vec<f64> = (0..n).map(|i| a1[i] * y[i]).collect();
    let a_gap: Vec<f64> = (0..n)
        .map(|i| a[i] * (prediction_post[i] - prediction_pre[i]))
        .collect();
    let eta_pre = mean(&a0_y) / mean_a0;
    let eta_post = mean(&a1_y) / mean_a1;
    let eta_control = mean(&a_gap) / mean_a;
    let att = (eta_post - eta_pre) - eta_control;
    let influence = (0..n)
        .map(|i| {
            let treated_part =
                (a1_y[i] - a1[i] * eta_post) / mean_a1 - (a0_y[i] - a0[i] * eta_pre) / mean_a0;
            let control_part =
                (a_gap[i] - a[i] * eta_control + or_term_post[i] - or_term_pre[i]) / mean_a;
            treated_part - control_part
        })
        .collect();
    Ok(estimate(sample, att, influence, config))
}

fn dr_repeated(sample: &Sample, config: DrDidConfig) -> Result<DrDidEstimate, DrDidError> {
    let n = sample.n();
    let k = sample.feature_count;
    let (d, t, y) = (&sample.treated, &sample.post, &sample.outcome);
    let scores = propensity(sample, config)?;
    let trim: Vec<f64> = (0..n)
        .map(|i| {
            if d[i] > 0.5 || scores[i] < TRIM_LEVEL {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let comparator_pre: Vec<f64> = (0..n)
        .map(|i| sample.weights[i] * (1.0 - d[i]) * (1.0 - t[i]))
        .collect();
    let comparator_post: Vec<f64> = (0..n)
        .map(|i| sample.weights[i] * (1.0 - d[i]) * t[i])
        .collect();
    let prediction_pre = outcome_fit(sample, &comparator_pre, config.ridge)?;
    let prediction_post = outcome_fit(sample, &comparator_post, config.ridge)?;
    let residual: Vec<f64> = (0..n)
        .map(|i| {
            let prediction = if t[i] > 0.5 {
                prediction_post[i]
            } else {
                prediction_pre[i]
            };
            y[i] - prediction
        })
        .collect();

    let treated_weight = |i: usize| trim[i] * sample.weights[i] * d[i];
    let control_weight =
        |i: usize| trim[i] * sample.weights[i] * scores[i] * (1.0 - d[i]) / (1.0 - scores[i]);
    let a0: Vec<f64> = (0..n).map(|i| treated_weight(i) * (1.0 - t[i])).collect();
    let a1: Vec<f64> = (0..n).map(|i| treated_weight(i) * t[i]).collect();
    let b0: Vec<f64> = (0..n).map(|i| control_weight(i) * (1.0 - t[i])).collect();
    let b1: Vec<f64> = (0..n).map(|i| control_weight(i) * t[i]).collect();

    // Each cell's `eta_i = c_i (Y_i - m_i) / mean(c)`, its mean and its
    // influence `eta_i - c_i att_c / mean(c)`.
    let cell = |c: &[f64], missing: DidCell| -> Result<(f64, f64, Vec<f64>), DrDidError> {
        let mc = mean(c);
        if mc <= 0.0 {
            return Err(DrDidError::MissingCell { cell: missing });
        }
        let eta: Vec<f64> = (0..n).map(|i| c[i] * residual[i] / mc).collect();
        let att_c = mean(&eta);
        let influence = (0..n).map(|i| eta[i] - c[i] * att_c / mc).collect();
        Ok((mc, att_c, influence))
    };
    let (treated_pre_weight, treated_pre, treated_pre_psi) = cell(&a0, DidCell::TreatedPre)?;
    let (treated_post_weight, treated_post, treated_post_psi) = cell(&a1, DidCell::TreatedPost)?;
    let (control_pre_weight, control_pre, control_pre_psi) = cell(&b0, DidCell::ControlPre)?;
    let (control_post_weight, control_post, control_post_psi) = cell(&b1, DidCell::ControlPost)?;
    let att = (treated_post - treated_pre) - (control_post - control_pre);

    // `-sum_i c_i X_i / (n mean(c))`, the derivative of a cell's mean with
    // respect to the regression coefficients.
    let scaled_moment = |c: &[f64], by: f64| -> Vec<f64> {
        moment(sample, c).into_iter().map(|v| -v / by).collect()
    };
    let m1_post = scaled_moment(&a1, treated_post_weight);
    let m1_pre = scaled_moment(&a0, treated_pre_weight);
    let m3_post = scaled_moment(&b1, control_post_weight);
    let m3_pre = scaled_moment(&b0, control_pre_weight);
    let treated_or_post = outcome_projection(sample, &comparator_post, &prediction_post, &m1_post)?;
    let treated_or_pre = outcome_projection(sample, &comparator_pre, &prediction_pre, &m1_pre)?;
    let control_or_post = outcome_projection(sample, &comparator_post, &prediction_post, &m3_post)?;
    let control_or_pre = outcome_projection(sample, &comparator_pre, &prediction_pre, &m3_pre)?;

    let centred = |c: &[f64], att_c: f64| -> Vec<f64> {
        (0..n).map(|i| c[i] * (residual[i] - att_c)).collect()
    };
    let moment_pre = moment(sample, &centred(&b0, control_pre));
    let moment_post = moment(sample, &centred(&b1, control_post));
    let m2: Vec<f64> = (0..k)
        .map(|j| moment_post[j] / control_post_weight - moment_pre[j] / control_pre_weight)
        .collect();
    let ps_term = propensity_projection(sample, &scores, &m2)?;

    let influence = (0..n)
        .map(|i| {
            let treated_part =
                treated_post_psi[i] - treated_pre_psi[i] + (treated_or_post[i] + treated_or_pre[i]);
            let control_part = control_post_psi[i] - control_pre_psi[i]
                + ps_term[i]
                + (control_or_post[i] + control_or_pre[i]);
            treated_part - control_part
        })
        .collect();
    Ok(estimate(sample, att, influence, config))
}

/// The traditional doubly robust repeated cross-section ATT,
/// `DRDID::drdid_rc1`: the weighted propensity score and the comparators'
/// outcome regression in each period, comparators with a score of 0.995 or
/// more trimmed, and an influence function that includes the estimation of
/// all three models. The intercept is prepended to the covariates.
pub(crate) fn estimate_dr_did_repeated_traditional(
    observations: &[DrDidRepeatedObservation],
    config: DrDidConfig,
) -> Result<DrDidEstimate, DrDidError> {
    validate_drdid_config(config)?;
    dr_repeated(&repeated_sample(observations, Nuisance::Outcome)?, config)
}

/// The panel inverse probability weighted ATT with normalised weights,
/// `DRDID::std_ipw_did_panel`, on a flat design intercept included. See the
/// module documentation for the estimator and its influence function, which
/// includes the estimation of the propensity score.
///
/// # Errors
/// [`DrDidError`] for an invalid configuration or input, an arm without
/// rows, or a nuisance model that cannot be fitted, plus
/// [`DrDidError::InconsistentCovariateCount`] for a design of the wrong length.
pub(crate) fn estimate_ipw_did_panel_flat(
    input: PanelFlatInput<'_>,
    config: DrDidConfig,
) -> Result<DrDidEstimate, DrDidError> {
    validate_drdid_config(config)?;
    ipw_panel(&panel_flat_sample(input, Nuisance::Propensity)?, config)
}

/// The panel outcome regression ATT, `DRDID::reg_did_panel`, on a flat design
/// intercept included. The influence function includes the estimation of the
/// comparators' regression.
///
/// # Errors
/// [`DrDidError`] for an invalid configuration or input, an arm without
/// rows, or a nuisance model that cannot be fitted, plus
/// [`DrDidError::InconsistentCovariateCount`] for a design of the wrong length.
pub(crate) fn estimate_or_did_panel_flat(
    input: PanelFlatInput<'_>,
    config: DrDidConfig,
) -> Result<DrDidEstimate, DrDidError> {
    validate_drdid_config(config)?;
    or_panel(&panel_flat_sample(input, Nuisance::Outcome)?, config)
}

/// The repeated cross-section inverse probability weighted ATT with
/// normalised weights, `DRDID::std_ipw_did_rc`.
///
/// The intercept is prepended to the covariates. The influence function
/// includes the estimation of the propensity score.
///
/// # Errors
/// [`DrDidError`] for an invalid configuration or input, a (group, period)
/// cell without rows, or a propensity score that cannot be fitted.
pub fn estimate_ipw_did_repeated_cross_section(
    observations: &[DrDidRepeatedObservation],
    config: DrDidConfig,
) -> Result<DrDidEstimate, DrDidError> {
    validate_drdid_config(config)?;
    ipw_repeated(
        &repeated_sample(observations, Nuisance::Propensity)?,
        config,
    )
}

/// The repeated cross-section outcome regression ATT, `DRDID::reg_did_rc`.
///
/// The intercept is prepended to the covariates. The comparators' outcome is
/// regressed on the covariates in each period, and the difference of the two
/// predictions is averaged over every treated row, both periods. The influence
/// function includes the estimation of both regressions.
///
/// # Errors
/// [`DrDidError`] for an invalid configuration or input, a (group, period)
/// cell without rows, or a regression that cannot be solved.
pub fn estimate_or_did_repeated_cross_section(
    observations: &[DrDidRepeatedObservation],
    config: DrDidConfig,
) -> Result<DrDidEstimate, DrDidError> {
    validate_drdid_config(config)?;
    or_repeated(&repeated_sample(observations, Nuisance::Outcome)?, config)
}
