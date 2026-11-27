//! The efficient `ATT(g,t)` of Chen, Sant'Anna and Xie, which combines every
//! pre-treatment baseline, under parallel trends in all periods (`PT-All`) with
//! never-treated controls: [`Estimator::Efficient`](super::Estimator).
//!
//! # Algorithm
//!
//! On a panel ([`Design::Panel`](super::Design)) the estimator works without
//! covariates. Its units
//! are numbered `i = 1, ..., N` in ascending order of identifier, and unit `i`
//! carries one sampling weight `w_i`, the same on all of its rows. For the
//! cohort `g`, a period `t >= g` and the cohort's pre-treatment periods
//! `B = {b < g}`, the cell uses the units of `g` (the set `T`) and the never
//! treated units (`C`) that are observed at `t` and at every `b` in `B`. For
//! each baseline `b`, with `D_ib = Y_it - Y_ib`,
//!
//! ```text
//! theta_b = mu_Tb - mu_Cb,      mu_Sb = sum_{i in S} w_i D_ib / W_S,
//!                               W_S   = sum_{i in S} w_i,
//! psi_ib  =  N w_i (D_ib - mu_Tb) / W_T     for i in T,
//!         = -N w_i (D_ib - mu_Cb) / W_C     for i in C,
//!         =  0                              for every other unit.
//! ```
//!
//! `psi_ib` is `N` times the derivative of `theta_b` in `log w_i`, and it is
//! the influence function that `did::att_gt` returns for the same comparison
//! (the panel `DRDID` estimator with an intercept only, scaled by `n / n_cell`).
//! With every weight 1 it is `(D_ib - mu_Tb) / (n_T / N)` and
//! `-(D_ib - mu_Cb) / (n_C / N)`. The baselines are then combined by
//! generalised least squares,
//!
//! ```text
//! Omega_bc  = sum_i psi_ib psi_ic / N,       v = Omega^{-1} 1,   a_b = v_b / sum_c v_c,
//! ATT(g,t)  = sum_b a_b theta_b,             psi_i(g,t) = sum_b a_b psi_ib,
//! se(g,t)   = sqrt(sum_i psi_i(g,t)^2 / N) / sqrt(N).
//! ```
//!
//! The weights `a_b` are estimated but add nothing to the influence function
//! to first order: since `sum_b a_b = 1`, `ATT - theta = sum_b a_b (theta_b -
//! theta)`, and the error in `a` multiplies `theta_b - theta`, which is of
//! order `N^{-1/2}` under `PT-All`, so the product is of order `1 / N`. Every
//! `psi_i(g,t)` is indexed by unit over the whole panel, so the influence
//! vectors of all cells of one call line up unit by unit. A singular `Omega`
//! is solved with a ridge, and an `Omega` of zero gives equal weights.
//!
//! On a repeated cross-section ([`Design::RepeatedCrossSection`](super::Design))
//! it replaces `theta_b` by the traditional doubly robust repeated
//! cross-section estimator of each baseline pair, `DRDID::drdid_rc1`. Its `psi` is
//! indexed by row of the input, `N` is the number of rows, and `Omega`, `a` and
//! the standard error are formed in the same way.
//!
//! Neither design applies `did`'s pre-processing: every unit, or every row,
//! is a position of the influence vectors, and the cells are aggregated by
//! [`AttGt::aggregate`](super::AttGt::aggregate) as any other.
//!
//! # References
//!
//! 1. Chen X, Sant'Anna PHC, Xie H. Efficient difference-in-differences and
//!    event study estimators. Working paper, 2025.
//! 2. Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time
//!    periods. J Econom 2021;225(2):200-30. doi:10.1016/j.jeconom.2020.12.001

use std::collections::{BTreeMap, BTreeSet};

use epistat_linalg::matmul;
use epistat_linalg::{Accum, Attempt as SolveAttempt, Mat, MatRef, Method, Par, Ridge, solve};

use crate::error::Error;
use crate::inference::normal_interval;
use crate::methods::drdid::repeated::estimate_drdid_repeated_cross_section;
use crate::util::usize_to_f64;

use super::pair_estimators::{build_pair_rows, missing_pair_cell};
use super::preprocess::row_code;
use super::rows::Rows;
use super::{
    AttGtSample, Attempt, BasePeriod, CellFit, ComparisonGroup, Config, EfficientBaseline,
    EfficientWeights, Output, SkipReason,
};

#[derive(Debug, Clone)]
struct UnitSeries {
    first_treated_time: Option<i64>,
    weight: f64,
    outcomes_by_time: BTreeMap<i64, f64>,
}

/// The units in ascending order of code. A unit's position here is its
/// index in every influence vector of the panel route.
type Panel = Vec<UnitSeries>;

#[derive(Debug, Clone)]
struct BaselineSample {
    baseline_time: i64,
    att: f64,
    treated_n: usize,
    control_n: usize,
    total_weight: f64,
}

#[derive(Debug, Clone)]
struct BaselineSamples {
    samples: Vec<BaselineSample>,
    influence_matrix_flat: Vec<f64>,
    influence_len: usize,
}

#[derive(Debug, Clone)]
struct EfficientWeightSolution {
    normalized_weights: Vec<f64>,
    raw_precision_solution: Vec<f64>,
    ridge_penalty: Option<f64>,
}

/// The cell of a combination of baselines: its fit, its influence vector and
/// its weights.
struct Combined {
    fit: CellFit,
    influence: Vec<f64>,
    weights: EfficientWeights,
}

/// The efficient `ATT(g,t)` on a panel without covariates.
///
/// Each unit's sampling weight weights its outcome changes and its influence
/// value; the caller checks that it is the same on all of the unit's rows.
/// The influence vector of every cell has one entry per unit of the panel, in
/// ascending order of code, zero for the units the cell does not use.
///
/// # Errors
/// [`Error::NoNeverTreated`] without never-treated units,
/// [`Error::NoPrePeriods`] or [`Error::NoPostPeriods`] for a cohort without
/// periods before or after it, [`Error::CellFailed`] for a cell without
/// treated or never-treated units observed at all its periods, and
/// [`Error::Singular`] when the baseline weighting system cannot be solved.
pub(super) fn estimate_panel(rows: &Rows<'_>, level: f64) -> Result<Output, Error> {
    let (units, panel) = build_panel(rows);
    let treated_groups = collect_treated_groups(&panel);
    let never_treated_ids = collect_group_positions(&panel, None);
    if never_treated_ids.is_empty() {
        return Err(Error::NoNeverTreated);
    }

    let mut attempts = Vec::new();
    let mut influence = Vec::new();
    for &group in &treated_groups {
        let pre_times = collect_pre_times(&panel, group);
        if pre_times.is_empty() {
            return Err(Error::NoPrePeriods { group });
        }
        let post_times = collect_post_times(&panel, group);
        if post_times.is_empty() {
            return Err(Error::NoPostPeriods { group });
        }

        let treated_ids = collect_group_positions(&panel, Some(group));
        for time in post_times {
            let combined = estimate_group_time(
                &panel,
                &treated_ids,
                &never_treated_ids,
                (group, time),
                &pre_times,
                level,
            )?;
            influence.extend_from_slice(&combined.influence);
            attempts.push(Attempt {
                group,
                time,
                base: pre_times.last().copied().unwrap_or(group),
                result: Ok(combined.fit),
                overlap: None,
                efficient: Some(combined.weights),
            });
        }
    }

    let mut periods: Vec<i64> = rows.period.to_vec();
    periods.sort_unstable();
    periods.dedup();
    let sample = AttGtSample {
        units,
        groups: panel
            .iter()
            .map(|series| series.first_treated_time)
            .collect(),
        weights: panel.iter().map(|series| series.weight).collect(),
        periods,
        cohorts: treated_groups,
        ..AttGtSample::default()
    };
    super::finish(attempts, influence, sample)
}

/// The covariate-adjusted efficient `ATT(g,t)` on a repeated cross-section,
/// combining the traditional doubly robust estimates of every baseline pair
/// by generalised least squares. The influence vector of every cell has one
/// entry per row, unscaled, so that `sqrt(sum psi_i^2) / n` is its standard
/// error.
///
/// # Errors
/// [`Error::NoNeverTreated`] under the never-treated comparison group without
/// never-treated rows, [`Error::CellFailed`] for a baseline pair that cannot
/// be fitted (or that has an empty arm when cells are not skipped),
/// [`Error::Singular`] when the weighting system cannot be solved, and
/// [`Error::NoEstimableCells`] if no cell is fitted.
pub(super) fn estimate_repeated(rows: &Rows<'_>, config: Config) -> Result<Output, Error> {
    let all_times: Vec<i64> = rows
        .period
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let treated_groups: Vec<i64> = rows
        .cohort
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if config.comparison_group == ComparisonGroup::NeverTreated
        && rows.cohort.iter().all(Option::is_some)
    {
        return Err(Error::NoNeverTreated);
    }

    let mut attempts = Vec::new();
    let mut influence = Vec::new();
    for &group in &treated_groups {
        let all_pre_treatment_times = all_times
            .iter()
            .copied()
            .filter(|time| *time < group)
            .collect::<Vec<_>>();
        for &time in &all_times {
            let baselines =
                baseline_times_for_pair_efficient_dr(time, group, &all_pre_treatment_times, config);
            if baselines.is_empty() || (baselines.len() == 1 && baselines[0] == time) {
                continue;
            }
            let base = baselines.last().copied().unwrap_or(group);
            let result = estimate_group_time_efficient_dr(rows, (group, time), &baselines, config)?;
            let attempt = match result {
                Some(combined) => {
                    influence.extend_from_slice(&combined.influence);
                    Attempt {
                        group,
                        time,
                        base,
                        result: Ok(combined.fit),
                        overlap: None,
                        efficient: Some(combined.weights),
                    }
                }
                None => Attempt {
                    group,
                    time,
                    base,
                    result: Err(SkipReason::NoBaselines),
                    overlap: None,
                    efficient: None,
                },
            };
            attempts.push(attempt);
        }
    }

    let sample = AttGtSample {
        units: (0..rows.len()).map(row_code).collect(),
        groups: rows.cohort.to_vec(),
        weights: (0..rows.len()).map(|i| rows.weight(i)).collect(),
        periods: all_times,
        cohorts: treated_groups,
        ..AttGtSample::default()
    };
    super::finish(attempts, influence, sample)
}

fn baseline_times_for_pair_efficient_dr(
    time: i64,
    group: i64,
    all_pre_treatment_times: &[i64],
    config: Config,
) -> Vec<i64> {
    if time < group && matches!(config.base_period, BasePeriod::Varying) {
        vec![time - 1]
    } else {
        all_pre_treatment_times.to_vec()
    }
}

/// The efficient weights of a combination, for the result.
fn weights_of(
    samples: &BaselineSamples,
    covariance: &[f64],
    solution: EfficientWeightSolution,
) -> Result<EfficientWeights, Error> {
    let dimension = samples.samples.len();
    Ok(EfficientWeights {
        baselines: samples
            .samples
            .iter()
            .zip(&solution.normalized_weights)
            .map(|(sample, weight)| EfficientBaseline {
                base: sample.baseline_time,
                estimate: sample.att,
                weight: *weight,
                n_treated: sample.treated_n,
                n_control: sample.control_n,
            })
            .collect(),
        covariance: epistat_core::Covariance::new(
            dimension,
            covariance.iter().map(|&value| Some(value)).collect(),
        )?,
        precision_solution: solution.raw_precision_solution,
        ridge: solution.ridge_penalty,
    })
}

fn estimate_group_time_efficient_dr(
    rows: &Rows<'_>,
    (group, time): (i64, i64),
    baselines: &[i64],
    config: Config,
) -> Result<Option<Combined>, Error> {
    let baseline_samples = collect_dr_baseline_samples(rows, (group, time), baselines, config)?;
    if baseline_samples.samples.is_empty() {
        return Ok(None);
    }

    let full_n = rows.len();
    let covariance = influence_covariance_matrix(&baseline_samples);
    let weight_solution = efficient_weights(&covariance)?;
    let weights = &weight_solution.normalized_weights;
    let att = baseline_samples
        .samples
        .iter()
        .zip(weights.iter())
        .map(|(sample, weight)| sample.att * weight)
        .sum::<f64>();
    let mut influence = vec![0.0; full_n];
    combine_influences_into(&baseline_samples, weights, &mut influence);
    let variance = influence.iter().map(|value| value * value).sum::<f64>() / usize_to_f64(full_n);
    let se = (variance / usize_to_f64(full_n)).sqrt();
    let (ci_low, ci_high) = normal_interval(att, se, config.drdid.confidence_level);
    let reference_counts = &baseline_samples.samples[0];
    let fit = CellFit {
        att,
        se,
        ci_low,
        ci_high,
        treated_n: reference_counts.treated_n,
        control_n: reference_counts.control_n,
        total_weight: reference_counts.total_weight,
        columns_dropped: 0,
    };
    let weights = weights_of(&baseline_samples, &covariance, weight_solution)?;
    Ok(Some(Combined {
        fit,
        influence,
        weights,
    }))
}

fn collect_dr_baseline_samples(
    rows: &Rows<'_>,
    (group, time): (i64, i64),
    baselines: &[i64],
    config: Config,
) -> Result<BaselineSamples, Error> {
    let full_n = rows.len();
    let mut baseline_samples = Vec::new();
    let mut influence_matrix_flat = Vec::with_capacity(baselines.len() * full_n);
    for &baseline_time in baselines {
        let cell = (group, time, baseline_time);
        let (pair_rows, pair_indices) = build_pair_rows(rows, cell, config);
        if let Some(reason) = missing_pair_cell(&pair_rows) {
            if config.skip_failed_cells {
                continue;
            }
            return Err(Error::CellFailed {
                group,
                time,
                base: baseline_time,
                reason,
            });
        }

        let estimate =
            estimate_drdid_repeated_cross_section(&pair_rows, config.drdid).map_err(|error| {
                Error::CellFailed {
                    group,
                    time,
                    base: baseline_time,
                    reason: SkipReason::FitFailed(super::panel_pairs::fit_failure(&error)),
                }
            })?;
        if estimate.influence_function.len() != pair_rows.len() {
            return Err(Error::CellFailed {
                group,
                time,
                base: baseline_time,
                reason: SkipReason::FitFailed(super::FitFailure::Other),
            });
        }

        let row_start = influence_matrix_flat.len();
        influence_matrix_flat.resize(row_start + full_n, 0.0);
        let aligned = &mut influence_matrix_flat[row_start..row_start + full_n];
        for (local_idx, global_idx) in pair_indices.iter().enumerate() {
            aligned[*global_idx] = estimate.influence_function[local_idx];
        }
        baseline_samples.push(BaselineSample {
            baseline_time,
            att: estimate.att,
            treated_n: estimate.treated_n,
            control_n: estimate.control_n,
            total_weight: estimate.total_weight,
        });
    }
    Ok(BaselineSamples {
        samples: baseline_samples,
        influence_matrix_flat,
        influence_len: full_n,
    })
}

/// The distinct unit codes in ascending order and each unit's series. The
/// rows of a validated panel are sorted by unit and then period, so each
/// unit's rows are adjacent.
fn build_panel(rows: &Rows<'_>) -> (Vec<u32>, Panel) {
    let mut units: Vec<u32> = Vec::new();
    let mut panel: Panel = Vec::new();
    for i in 0..rows.len() {
        if units.last() != Some(&rows.unit[i]) {
            units.push(rows.unit[i]);
            panel.push(UnitSeries {
                first_treated_time: rows.cohort[i],
                weight: rows.weight(i),
                outcomes_by_time: BTreeMap::new(),
            });
        }
        if let Some(series) = panel.last_mut() {
            series
                .outcomes_by_time
                .insert(rows.period[i], rows.outcome[i]);
        }
    }
    (units, panel)
}

fn collect_treated_groups(panel: &Panel) -> Vec<i64> {
    panel
        .iter()
        .filter_map(|series| series.first_treated_time)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// The positions of the units of cohort `group`, `None` for never treated.
fn collect_group_positions(panel: &Panel, group: Option<i64>) -> Vec<usize> {
    panel
        .iter()
        .enumerate()
        .filter_map(|(position, series)| (series.first_treated_time == group).then_some(position))
        .collect()
}

fn collect_pre_times(panel: &Panel, group: i64) -> Vec<i64> {
    panel
        .iter()
        .filter(|series| series.first_treated_time == Some(group))
        .flat_map(|series| series.outcomes_by_time.keys().copied())
        .filter(|time| *time < group)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn collect_post_times(panel: &Panel, group: i64) -> Vec<i64> {
    panel
        .iter()
        .filter(|series| series.first_treated_time == Some(group))
        .flat_map(|series| series.outcomes_by_time.keys().copied())
        .filter(|time| *time >= group)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn estimate_group_time(
    panel: &Panel,
    treated_ids: &[usize],
    control_ids: &[usize],
    (group, time): (i64, i64),
    baselines: &[i64],
    level: f64,
) -> Result<Combined, Error> {
    let base = baselines.last().copied().unwrap_or(group);
    let treated_panel = filter_complete_ids(panel, treated_ids, time, baselines);
    if treated_panel.is_empty() {
        return Err(Error::CellFailed {
            group,
            time,
            base,
            reason: SkipReason::NoTreated,
        });
    }
    let control_panel = filter_complete_ids(panel, control_ids, time, baselines);
    if control_panel.is_empty() {
        return Err(Error::CellFailed {
            group,
            time,
            base,
            reason: SkipReason::NoComparison,
        });
    }

    let unit_count = panel.len();
    let mut baseline_rows = Vec::with_capacity(baselines.len());
    let mut influence_matrix_flat = vec![0.0; baselines.len() * unit_count];
    for (baseline_idx, baseline_time) in baselines.iter().copied().enumerate() {
        let row_start = baseline_idx * unit_count;
        let influence_row = &mut influence_matrix_flat[row_start..row_start + unit_count];
        baseline_rows.push(build_baseline_sample(
            panel,
            &treated_panel,
            &control_panel,
            time,
            baseline_time,
            influence_row,
        ));
    }
    let baseline_samples = BaselineSamples {
        samples: baseline_rows,
        influence_matrix_flat,
        influence_len: unit_count,
    };

    let covariance = influence_covariance_matrix(&baseline_samples);
    let weight_solution = efficient_weights(&covariance)?;
    let weights = &weight_solution.normalized_weights;
    let att = baseline_samples
        .samples
        .iter()
        .zip(weights.iter())
        .map(|(sample, weight)| sample.att * weight)
        .sum::<f64>();
    let mut influence = vec![0.0; unit_count];
    combine_influences_into(&baseline_samples, weights, &mut influence);
    let se = standard_error_uncentred(&influence);
    let (ci_low, ci_high) = normal_interval(att, se, level);
    let reference = &baseline_samples.samples[0];
    let fit = CellFit {
        att,
        se,
        ci_low,
        ci_high,
        treated_n: treated_panel.len(),
        control_n: control_panel.len(),
        total_weight: reference.total_weight,
        columns_dropped: 0,
    };
    let weights = weights_of(&baseline_samples, &covariance, weight_solution)?;
    Ok(Combined {
        fit,
        influence,
        weights,
    })
}

/// `sqrt(sum psi_i^2 / n) / sqrt(n)`, the standard error of an estimate whose
/// influence function `psi` has mean zero by construction.
fn standard_error_uncentred(influence: &[f64]) -> f64 {
    let n = usize_to_f64(influence.len());
    let variance = influence.iter().map(|value| value * value).sum::<f64>() / n;
    (variance / n).sqrt()
}

fn filter_complete_ids(
    panel: &Panel,
    positions: &[usize],
    time: i64,
    baselines: &[i64],
) -> Vec<usize> {
    positions
        .iter()
        .copied()
        .filter(|position| {
            let outcomes = &panel[*position].outcomes_by_time;
            outcomes.contains_key(&time)
                && baselines
                    .iter()
                    .all(|baseline| outcomes.contains_key(baseline))
        })
        .collect()
}

/// The weighted mean of `Y_t - Y_b` over `positions` and the sum of their weights.
fn weighted_change(
    panel: &Panel,
    positions: &[usize],
    time: i64,
    baseline_time: i64,
) -> (f64, f64) {
    let mut weight_sum = 0.0;
    let mut weighted_sum = 0.0;
    for position in positions {
        let series = &panel[*position];
        let delta = series.outcomes_by_time[&time] - series.outcomes_by_time[&baseline_time];
        weight_sum += series.weight;
        weighted_sum += series.weight * delta;
    }
    (weighted_sum / weight_sum, weight_sum)
}

/// Writes `sign N w_i (D_ib - mean) / W` at the position of each unit in
/// `positions`; `influence_out` holds one entry per unit of the panel.
fn write_group_influence(
    panel: &Panel,
    positions: &[usize],
    time: i64,
    baseline_time: i64,
    (mean, weight_sum): (f64, f64),
    sign: f64,
    influence_out: &mut [f64],
) {
    let scale = sign * usize_to_f64(panel.len()) / weight_sum;
    for position in positions {
        let series = &panel[*position];
        let delta = series.outcomes_by_time[&time] - series.outcomes_by_time[&baseline_time];
        influence_out[*position] = scale * (series.weight * (delta - mean));
    }
}

fn build_baseline_sample(
    panel: &Panel,
    treated_ids: &[usize],
    control_ids: &[usize],
    time: i64,
    baseline_time: i64,
    influence_out: &mut [f64],
) -> BaselineSample {
    debug_assert_eq!(influence_out.len(), panel.len());
    let treated = weighted_change(panel, treated_ids, time, baseline_time);
    let control = weighted_change(panel, control_ids, time, baseline_time);
    write_group_influence(
        panel,
        treated_ids,
        time,
        baseline_time,
        treated,
        1.0,
        influence_out,
    );
    write_group_influence(
        panel,
        control_ids,
        time,
        baseline_time,
        control,
        -1.0,
        influence_out,
    );

    BaselineSample {
        baseline_time,
        att: treated.0 - control.0,
        treated_n: treated_ids.len(),
        control_n: control_ids.len(),
        total_weight: treated.1 + control.1,
    }
}

fn influence_covariance_matrix(samples: &BaselineSamples) -> Vec<f64> {
    let dimension = samples.samples.len();
    let sample_len = samples.influence_len;
    let if_matrix =
        MatRef::from_row_major_slice(&samples.influence_matrix_flat, dimension, sample_len);
    let mut covariance = Mat::<f64>::zeros(dimension, dimension);
    matmul(
        covariance.as_mut(),
        Accum::Replace,
        if_matrix,
        if_matrix.transpose(),
        1.0 / usize_to_f64(sample_len),
        Par::Seq,
    );
    let mut out = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            out[row * dimension + col] = covariance[(row, col)];
        }
    }
    out
}

fn efficient_weights(covariance: &[f64]) -> Result<EfficientWeightSolution, Error> {
    let dimension = checked_square_dimension(covariance.len()).ok_or(Error::Singular)?;
    let covariance_scale = covariance.iter().map(|value| value.abs()).sum::<f64>();
    if covariance_scale <= f64::EPSILON {
        let equal_weight = 1.0 / usize_to_f64(dimension);
        return Ok(EfficientWeightSolution {
            normalized_weights: vec![equal_weight; dimension],
            raw_precision_solution: vec![1.0; dimension],
            ridge_penalty: None,
        });
    }
    let ones = vec![1.0; dimension];
    let (mut weights, ridge_penalty) =
        solve_with_ridge_fallback(covariance, &ones).ok_or(Error::Singular)?;
    let raw_precision_solution = weights.clone();
    let normalizer = weights.iter().sum::<f64>();
    if !normalizer.is_finite() || normalizer.abs() < f64::EPSILON {
        return Err(Error::Singular);
    }
    for weight in &mut weights {
        *weight /= normalizer;
    }
    Ok(EfficientWeightSolution {
        normalized_weights: weights,
        raw_precision_solution,
        ridge_penalty,
    })
}

/// Solves `covariance x = rhs` by the symmetric fallback chain, and if that
/// fails by Cholesky with a diagonal ridge starting at `1e-8` times the mean
/// absolute diagonal and growing by 100 for up to six tries. The second element
/// is the ridge that was used.
fn solve_with_ridge_fallback(covariance: &[f64], rhs: &[f64]) -> Option<(Vec<f64>, Option<f64>)> {
    let dimension = rhs.len();
    let mean_diagonal = (0..dimension)
        .map(|index| covariance[index * dimension + index].abs())
        .sum::<f64>()
        / usize_to_f64(dimension).max(1.0);
    let ridge = Ridge {
        start: 1e-8 * mean_diagonal,
        growth: 100.0,
        tries: 6,
    };
    let attempts = [
        SolveAttempt::SYMMETRIC[0],
        SolveAttempt::SYMMETRIC[1],
        SolveAttempt::SYMMETRIC[2],
        SolveAttempt::SYMMETRIC[3],
        SolveAttempt::ridged(Method::Cholesky, ridge),
    ];
    let system = MatRef::from_row_major_slice(covariance, dimension, dimension);
    let solution = solve(
        system,
        MatRef::from_column_major_slice(rhs, dimension, 1),
        &attempts,
    )
    .ok()?;
    let ridge_used = (solution.ridge > 0.0).then_some(solution.ridge);
    Some((solution.x.col(0).iter().copied().collect(), ridge_used))
}

fn combine_influences_into(samples: &BaselineSamples, weights: &[f64], out: &mut [f64]) {
    debug_assert_eq!(samples.samples.len(), weights.len());
    debug_assert_eq!(samples.influence_len, out.len());
    out.fill(0.0);
    for (baseline_idx, weight) in weights.iter().copied().enumerate() {
        let row_start = baseline_idx * samples.influence_len;
        let row = &samples.influence_matrix_flat[row_start..row_start + samples.influence_len];
        for (combined_value, influence_value) in out.iter_mut().zip(row.iter()) {
            *combined_value = weight.mul_add(*influence_value, *combined_value);
        }
    }
}

fn checked_square_dimension(value: usize) -> Option<usize> {
    let mut candidate = 0usize;
    while candidate.saturating_mul(candidate) < value {
        candidate = candidate.saturating_add(1);
    }
    (candidate.saturating_mul(candidate) == value).then_some(candidate)
}
