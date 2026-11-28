//! Counterfactual effect helpers derived from MTP posterior draws.

use epistat_linalg::MatRef;

use crate::utils::{EffectIntervalSummary, summarize_draws_in_place, usize_to_f64};

use super::likelihood::logistic_stable;
use super::posterior::MtpPosteriorSamples;
use super::types::validate_draw_dimensions;
use crate::Error;
use epistat_core::InputError;

/// Exposed/unexposed design scenarios for posterior contrast calculations,
/// one row per period.
#[derive(Debug, Clone, Copy)]
pub struct CounterfactualScenario<'a> {
    /// The binary-part design with every row exposed.
    pub binary_design_exposed: MatRef<'a, f64>,
    /// The mean design with every row exposed.
    pub mean_design_exposed: MatRef<'a, f64>,
    /// The binary-part design with every row unexposed.
    pub binary_design_unexposed: MatRef<'a, f64>,
    /// The mean design with every row unexposed.
    pub mean_design_unexposed: MatRef<'a, f64>,
}

/// Per-period effect estimates.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct PeriodEffect {
    /// The period, in the order of the input's times.
    pub period_index: usize,
    /// The mean outcome with every row exposed.
    pub mean_exposed: f64,
    /// The mean outcome with every row unexposed.
    pub mean_unexposed: f64,
    /// The difference of the exposed and unexposed means.
    pub additive_effect: f64,
    /// The ratio of the exposed to the unexposed mean, `None` when the
    /// unexposed mean is 0.
    pub multiplicative_effect: Option<f64>,
    /// The odds ratio of a positive outcome, exposed against unexposed,
    /// `None` when an odds is undefined or the unexposed odds is 0.
    pub odds_ratio_positive: Option<f64>,
}

/// Counterfactual effects across all periods in a scenario.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct CounterfactualEffects {
    /// The effects in each period.
    pub per_period: Vec<PeriodEffect>,
    /// The sum of the additive effects over the periods.
    pub cumulative_additive_effect: f64,
}

/// Posterior summaries for period-specific effects.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PeriodEffectSummary {
    /// The period, in the order of the input's times.
    pub period_index: usize,
    /// The posterior of the exposed mean.
    pub mean_exposed: EffectIntervalSummary,
    /// The posterior of the unexposed mean.
    pub mean_unexposed: EffectIntervalSummary,
    /// The posterior of the additive effect.
    pub additive_effect: EffectIntervalSummary,
    /// The posterior of the multiplicative effect.
    pub multiplicative_effect: EffectIntervalSummary,
    /// The posterior of the odds ratio of a positive outcome.
    pub odds_ratio_positive: EffectIntervalSummary,
}

/// Posterior summaries for all counterfactual effects.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct CounterfactualEffectsSummary {
    /// The posterior summaries in each period.
    pub per_period: Vec<PeriodEffectSummary>,
    /// The posterior of the cumulative additive effect.
    pub cumulative_additive_effect: EffectIntervalSummary,
}

/// # Errors
///
/// Returns `Error` if draws are missing or design dimensions are inconsistent.
pub fn counterfactual_effects(
    samples: &MtpPosteriorSamples,
    scenario: &CounterfactualScenario,
) -> Result<CounterfactualEffects, Error> {
    if samples.is_empty() {
        return Err(Error::Input(InputError::Empty {
            what: "posterior draws",
        }));
    }

    validate_counterfactual_design(scenario)?;

    let alpha = posterior_mean_alpha(samples);
    let beta = posterior_mean_beta(samples);

    validate_coef_dims(scenario.binary_design_exposed.ncols(), alpha.len())?;
    validate_coef_dims(scenario.binary_design_unexposed.ncols(), alpha.len())?;
    validate_coef_dims(scenario.mean_design_exposed.ncols(), beta.len())?;
    validate_coef_dims(scenario.mean_design_unexposed.ncols(), beta.len())?;

    let periods = scenario.binary_design_exposed.nrows();
    let mut per_period = Vec::with_capacity(periods);
    let mut cumulative_additive = 0.0;

    for row in 0..periods {
        let p_exposed = logistic_stable(dot_row(scenario.binary_design_exposed, row, &alpha));
        let p_unexposed = logistic_stable(dot_row(scenario.binary_design_unexposed, row, &alpha));

        // MTP beta parameterizes the marginal mean directly on log-scale.
        let mean_exposed = dot_row(scenario.mean_design_exposed, row, &beta).exp();
        let mean_unexposed = dot_row(scenario.mean_design_unexposed, row, &beta).exp();

        let additive_effect = mean_exposed - mean_unexposed;
        let multiplicative_effect = ratio(mean_exposed, mean_unexposed);
        let odds_ratio_positive = odds_ratio(p_exposed, p_unexposed);

        cumulative_additive += additive_effect;
        per_period.push(PeriodEffect {
            period_index: row,
            mean_exposed,
            mean_unexposed,
            additive_effect,
            multiplicative_effect,
            odds_ratio_positive,
        });
    }

    Ok(CounterfactualEffects {
        per_period,
        cumulative_additive_effect: cumulative_additive,
    })
}

/// Compute posterior effect summaries across all retained draws.
///
/// # Errors
///
/// Returns `Error` if draws are missing or design dimensions are inconsistent.
pub fn counterfactual_effects_summary(
    samples: &MtpPosteriorSamples,
    scenario: &CounterfactualScenario,
) -> Result<CounterfactualEffectsSummary, Error> {
    if samples.is_empty() {
        return Err(Error::Input(InputError::Empty {
            what: "posterior draws",
        }));
    }
    validate_counterfactual_design(scenario)?;

    let alpha_len = samples.draws.first().map_or(0, |draw| draw.alpha.len());
    let beta_len = samples.draws.first().map_or(0, |draw| draw.beta.len());
    validate_coef_dims(scenario.binary_design_exposed.ncols(), alpha_len)?;
    validate_coef_dims(scenario.binary_design_unexposed.ncols(), alpha_len)?;
    validate_coef_dims(scenario.mean_design_exposed.ncols(), beta_len)?;
    validate_coef_dims(scenario.mean_design_unexposed.ncols(), beta_len)?;

    let periods = scenario.binary_design_exposed.nrows();
    let draw_count = samples.len();

    let mut mean_exposed_draws = vec![Vec::with_capacity(draw_count); periods];
    let mut mean_unexposed_draws = vec![Vec::with_capacity(draw_count); periods];
    let mut additive_draws = vec![Vec::with_capacity(draw_count); periods];
    let mut multiplicative_draws = vec![Vec::with_capacity(draw_count); periods];
    let mut odds_ratio_draws = vec![Vec::with_capacity(draw_count); periods];
    let mut cumulative_additive_draws = Vec::with_capacity(draw_count);

    for draw in &samples.draws {
        validate_draw_dimensions(alpha_len, beta_len, draw.alpha.len(), draw.beta.len())?;

        let mut cumulative_additive = 0.0;
        for row in 0..periods {
            let p_exposed =
                logistic_stable(dot_row(scenario.binary_design_exposed, row, &draw.alpha));
            let p_unexposed =
                logistic_stable(dot_row(scenario.binary_design_unexposed, row, &draw.alpha));

            let mean_exposed = dot_row(scenario.mean_design_exposed, row, &draw.beta).exp();
            let mean_unexposed = dot_row(scenario.mean_design_unexposed, row, &draw.beta).exp();
            let additive_effect = mean_exposed - mean_unexposed;
            // An undefined ratio enters the draws as NaN, so that its summary is
            // undefined rather than formed from a floored denominator.
            let multiplicative_effect = ratio(mean_exposed, mean_unexposed).unwrap_or(f64::NAN);
            let odds_ratio_positive = odds_ratio(p_exposed, p_unexposed).unwrap_or(f64::NAN);

            mean_exposed_draws[row].push(mean_exposed);
            mean_unexposed_draws[row].push(mean_unexposed);
            additive_draws[row].push(additive_effect);
            multiplicative_draws[row].push(multiplicative_effect);
            odds_ratio_draws[row].push(odds_ratio_positive);
            cumulative_additive += additive_effect;
        }
        cumulative_additive_draws.push(cumulative_additive);
    }

    // The per-period draw vectors are owned here and not needed afterwards, so
    // they are sorted in place for the quantiles rather than copied again.
    let per_period = (0..periods)
        .map(|row| PeriodEffectSummary {
            period_index: row,
            mean_exposed: summarize_draws_in_place(&mut mean_exposed_draws[row]),
            mean_unexposed: summarize_draws_in_place(&mut mean_unexposed_draws[row]),
            additive_effect: summarize_draws_in_place(&mut additive_draws[row]),
            multiplicative_effect: summarize_draws_in_place(&mut multiplicative_draws[row]),
            odds_ratio_positive: summarize_draws_in_place(&mut odds_ratio_draws[row]),
        })
        .collect();

    Ok(CounterfactualEffectsSummary {
        per_period,
        cumulative_additive_effect: summarize_draws_in_place(&mut cumulative_additive_draws),
    })
}

/// Row `row` of `matrix` times the coefficients.
fn dot_row(matrix: MatRef<'_, f64>, row: usize, coefficients: &[f64]) -> f64 {
    (0..matrix.ncols())
        .map(|col| matrix[(row, col)] * coefficients[col])
        .sum()
}

/// `a / b`, `None` when `b` is 0 or the ratio is not finite.
fn ratio(a: f64, b: f64) -> Option<f64> {
    let r = a / b;
    (b != 0.0 && r.is_finite()).then_some(r)
}

/// The odds ratio of the probabilities `p1` and `p0`, `None` when an odds or
/// the ratio is undefined.
fn odds_ratio(p1: f64, p0: f64) -> Option<f64> {
    ratio(ratio(p1, 1.0 - p1)?, ratio(p0, 1.0 - p0)?)
}

fn validate_counterfactual_design(scenario: &CounterfactualScenario) -> Result<(), Error> {
    let rows = scenario.binary_design_exposed.nrows();
    if rows == 0
        || scenario.mean_design_exposed.nrows() == 0
        || scenario.binary_design_unexposed.nrows() == 0
        || scenario.mean_design_unexposed.nrows() == 0
    {
        return Err(Error::Input(InputError::Empty {
            what: "counterfactual design",
        }));
    }

    if scenario.mean_design_exposed.nrows() != rows
        || scenario.binary_design_unexposed.nrows() != rows
        || scenario.mean_design_unexposed.nrows() != rows
    {
        return Err(Error::Input(InputError::Length {
            what: "rows of the counterfactual designs",
            expected: rows,
            found: [
                scenario.mean_design_exposed.nrows(),
                scenario.binary_design_unexposed.nrows(),
                scenario.mean_design_unexposed.nrows(),
            ]
            .into_iter()
            .find(|&n| n != rows)
            .unwrap_or(rows),
        }));
    }

    Ok(())
}

const fn validate_coef_dims(design_cols: usize, coef_len: usize) -> Result<(), Error> {
    if design_cols != coef_len {
        return Err(Error::Input(InputError::Length {
            what: "coefficients",
            expected: design_cols,
            found: coef_len,
        }));
    }
    Ok(())
}

fn posterior_mean_alpha(samples: &MtpPosteriorSamples) -> Vec<f64> {
    let len = samples.draws.first().map_or(0, |draw| draw.alpha.len());
    let mut mean = vec![0.0; len];

    for draw in &samples.draws {
        for (index, value) in draw.alpha.iter().copied().enumerate() {
            mean[index] += value;
        }
    }

    let n = usize_to_f64(samples.len());
    for value in &mut mean {
        *value /= n;
    }

    mean
}

fn posterior_mean_beta(samples: &MtpPosteriorSamples) -> Vec<f64> {
    let len = samples.draws.first().map_or(0, |draw| draw.beta.len());
    let mut mean = vec![0.0; len];

    for draw in &samples.draws {
        for (index, value) in draw.beta.iter().copied().enumerate() {
            mean[index] += value;
        }
    }

    let n = usize_to_f64(samples.len());
    for value in &mut mean {
        *value /= n;
    }

    mean
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtp::posterior::MtpPosteriorDraw;
    use epistat_linalg::Mat;

    /// The scenario with `exposed` as both designs of the exposed rows and
    /// `unexposed` as both designs of the unexposed rows.
    fn scenario<'a>(exposed: &'a Mat<f64>, unexposed: &'a Mat<f64>) -> CounterfactualScenario<'a> {
        CounterfactualScenario {
            binary_design_exposed: exposed.as_ref(),
            mean_design_exposed: exposed.as_ref(),
            binary_design_unexposed: unexposed.as_ref(),
            mean_design_unexposed: unexposed.as_ref(),
        }
    }

    /// An intercept and an exposure of 0 in the first row and 1 in the
    /// second, and the same rows unexposed.
    fn two_periods() -> (Mat<f64>, Mat<f64>) {
        (
            Mat::from_fn(2, 2, |i, j| if j == 0 || i == 1 { 1.0 } else { 0.0 }),
            Mat::from_fn(2, 2, |_i, j| if j == 0 { 1.0 } else { 0.0 }),
        )
    }

    #[test]
    fn compute_effects_requires_draws() {
        let design = Mat::from_fn(1, 1, |_i, _j| 1.0);
        let scenario = scenario(&design, &design);

        let result = counterfactual_effects(&MtpPosteriorSamples::default(), &scenario);
        assert!(matches!(
            result,
            Err(Error::Input(InputError::Empty {
                what: "posterior draws",
            }))
        ));
    }

    #[test]
    fn compute_effects_runs_for_matching_dims() {
        let samples = MtpPosteriorSamples {
            draws: vec![MtpPosteriorDraw {
                alpha: vec![0.0, 0.5],
                beta: vec![0.0, 0.25],
                kappa: 0.1,
                omega_sq: 1.0,
            }],
        };

        let (exposed, unexposed) = two_periods();
        let scenario = scenario(&exposed, &unexposed);

        let effects = counterfactual_effects(&samples, &scenario)
            .expect("matching dimensions should produce effects");
        assert_eq!(effects.per_period.len(), 2);
    }

    #[test]
    fn compute_effects_summary_requires_draws() {
        let design = Mat::from_fn(1, 1, |_i, _j| 1.0);
        let scenario = scenario(&design, &design);

        let result = counterfactual_effects_summary(&MtpPosteriorSamples::default(), &scenario);
        assert!(matches!(
            result,
            Err(Error::Input(InputError::Empty {
                what: "posterior draws",
            }))
        ));
    }

    #[test]
    fn compute_effects_summary_returns_intervals() {
        let samples = MtpPosteriorSamples {
            draws: vec![
                MtpPosteriorDraw {
                    alpha: vec![0.0, 0.5],
                    beta: vec![0.0, 0.25],
                    kappa: 0.1,
                    omega_sq: 1.0,
                },
                MtpPosteriorDraw {
                    alpha: vec![0.0, 0.4],
                    beta: vec![0.0, 0.2],
                    kappa: 0.2,
                    omega_sq: 1.1,
                },
            ],
        };

        let (exposed, unexposed) = two_periods();
        let scenario = scenario(&exposed, &unexposed);

        let summary =
            counterfactual_effects_summary(&samples, &scenario).expect("summary should compute");
        assert_eq!(summary.per_period.len(), 2);
        assert!(summary.cumulative_additive_effect.mean.unwrap().is_finite());
        assert!(
            summary.per_period[0].additive_effect.q975.unwrap()
                >= summary.per_period[0].additive_effect.q025.unwrap()
        );
    }
}
