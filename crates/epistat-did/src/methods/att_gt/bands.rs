use epistat_core::{Bootstrap, Level};
use epistat_inference::Clusters;
use epistat_inference::simultaneous::multiplier_bands;

use crate::inference::validate_confidence_level;
use crate::types::{AttGtBandConfig, AttGtBandError, AttGtBandEstimate, AttGtEstimate};

fn validate_band_inputs(
    estimates: &[AttGtEstimate],
    config: AttGtBandConfig,
) -> Result<(), AttGtBandError> {
    if estimates.is_empty() {
        return Err(AttGtBandError::EmptyInput);
    }
    if !validate_confidence_level(config.confidence_level.confidence_level) {
        return Err(AttGtBandError::InvalidConfidenceLevel);
    }
    if config.reps == 0 {
        return Err(AttGtBandError::InvalidReps);
    }
    for estimate in estimates {
        if !estimate.att.is_finite() {
            return Err(AttGtBandError::InvalidEstimate);
        }
        if !estimate.se.is_finite() || estimate.se < 0.0 {
            return Err(AttGtBandError::InvalidSe);
        }
    }
    Ok(())
}

/// Simultaneous bands by the multiplier bootstrap, as `did:::mboot` does it.
///
/// # The bootstrap
///
/// The draws are not scaled by the analytic influence standard error:
///
/// * The scale of each estimate is the interquartile range of its bootstrap
///   draws divided by `qnorm(0.75) - qnorm(0.25)`, so a few extreme draws do not
///   move it.
/// * The critical value is the type 1 quantile, the inverse empirical CDF, of the
///   maximum absolute t statistic over all cells.
/// * One weight is drawn per cluster, or per unit without clustering, and applied
///   to the cluster's summed influence. Clusters are numbered in increasing label
///   order, as `did` numbers them.
///
/// The bands are `att +- crit * se`, with `se = sigma * sqrt(G) / n` for `G`
/// clusters and `n` units. An estimate whose draws are degenerate takes no part
/// in the maximum, as in `did`, and has `NaN` for its standard error and band.
///
/// # Errors
///
/// Returns [`AttGtBandError`] when inputs are malformed or every influence vector
/// is degenerate.
pub fn att_gt_mboot_bands(
    estimates: &[AttGtEstimate],
    influence_functions: &[Vec<f64>],
    clusters: Option<&[i64]>,
    config: AttGtBandConfig,
) -> Result<Vec<AttGtBandEstimate>, AttGtBandError> {
    validate_band_inputs(estimates, config)?;
    if influence_functions.len() != estimates.len() {
        return Err(AttGtBandError::InfluenceCountMismatch);
    }
    if influence_functions.is_empty() {
        return Err(AttGtBandError::EmptyInput);
    }
    let n = influence_functions[0].len();
    if n == 0 {
        return Err(AttGtBandError::EmptyInfluence);
    }
    for influence in influence_functions {
        if influence.len() != n {
            return Err(AttGtBandError::InconsistentInfluenceLength);
        }
        if influence.iter().any(|value| !value.is_finite()) {
            return Err(AttGtBandError::InvalidInfluence);
        }
    }
    let cluster_numbers = match clusters {
        None => None,
        Some(labels) if labels.len() == n => Some(Clusters::from_sorted_labels(labels)),
        Some(_) => return Err(AttGtBandError::InconsistentInfluenceLength),
    };

    // The weights are Rademacher, as in BMisc::multiplier_bootstrap, which is the
    // kernel did calls, and not the Mammen weights its documentation names.
    let alpha = (1.0 - config.confidence_level.confidence_level).clamp(0.0, 1.0);
    let columns: Vec<&[f64]> = influence_functions.iter().map(Vec::as_slice).collect();
    let level = Level::alpha(alpha).map_err(|_| AttGtBandError::DegenerateInfluence)?;
    let bands = multiplier_bands(
        &columns,
        cluster_numbers.as_ref(),
        level,
        Bootstrap::new(config.reps, config.seed),
    )
    .map_err(|_| AttGtBandError::DegenerateInfluence)?;
    debug_assert_eq!(bands.se.len(), estimates.len());

    Ok(estimates
        .iter()
        .zip(&bands.se)
        .map(|(estimate, se)| {
            let se = se.unwrap_or(f64::NAN);
            let margin = bands.critical * se;
            AttGtBandEstimate {
                group: estimate.group,
                time: estimate.time,
                event_time: estimate.event_time,
                att: estimate.att,
                se,
                band_low: estimate.att - margin,
                band_high: estimate.att + margin,
            }
        })
        .collect())
}
