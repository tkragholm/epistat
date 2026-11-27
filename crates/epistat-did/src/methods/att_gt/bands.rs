//! Simultaneous bands over the cells of an ATT(g,t) by the multiplier
//! bootstrap, as `did:::mboot` forms them.
//!
//! # Algorithm
//!
//! Replicate `b` draws a Rademacher weight `w_gb` per cluster, or per unit
//! without clusters, from `replicate_seed(seed, b)`, and forms
//! `T_bj = sqrt(G) mean_g(w_gb S_gj)` for every cell `j`, with `S_gj` the sum
//! of the cell's influence function in cluster `g`. The draws are not scaled
//! by the analytic standard error:
//!
//! - the scale `sigma_j` of each cell is the interquartile range of its draws
//!   divided by `qnorm(0.75) - qnorm(0.25)`, so a few extreme draws do not
//!   move it, and its standard error is `sigma_j sqrt(G) / n`;
//! - the critical value is the type 1 quantile at `1 - alpha`, the inverse
//!   empirical CDF, of `max_j |T_bj| / sigma_j` over the cells.
//!
//! The bands are `att +- crit * se`. A cell whose draws are degenerate takes
//! no part in the maximum, as in `did`, and has no standard error or band.
//! Clusters are numbered in increasing code, as `did` numbers them, and the
//! result does not depend on the number of threads.
//!
//! # References
//!
//! Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time
//! periods. J Econom 2021;225(2):200-30. R package `did` 2.5.1, `mboot`.

use epistat_core::{Bootstrap, Level};
use epistat_inference::Clusters;
use epistat_inference::simultaneous::multiplier_bands;

use super::AttGt;
use crate::error::Error;

/// The simultaneous band of one cell.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttGtBand {
    /// The cell's cohort.
    pub group: i64,
    /// The cell's later period.
    pub time: i64,
    /// `time - group`.
    pub event_time: i64,
    /// The cell's estimate.
    pub estimate: f64,
    /// The bootstrap standard error, `None` where the cell's draws are
    /// degenerate.
    pub se: Option<f64>,
    /// `estimate - critical_value * se`.
    pub lower: Option<f64>,
    /// `estimate + critical_value * se`.
    pub upper: Option<f64>,
}

/// Simultaneous bands over the estimated cells.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct AttGtBands {
    /// The simultaneous critical value, a multiplier of the standard errors.
    pub critical_value: f64,
    /// The band of each estimated cell, in the order of the cells.
    pub cells: Vec<AttGtBand>,
    /// The joint confidence level.
    pub level: Level,
}

/// [`AttGt::bands`].
pub(super) fn bands(fit: &AttGt, level: Level, bootstrap: Bootstrap) -> Result<AttGtBands, Error> {
    let influence = fit.kept_influence()?;
    let columns: Vec<&[f64]> = (0..influence.n_columns())
        .map(|j| influence.column(j))
        .collect();
    let clusters = fit.clusters.as_deref().map(Clusters::from_codes);
    // The bands take the level as an alpha, which is the number `did` gives
    // them; a level given as a confidence passes its own `1 - c`.
    let alpha = Level::alpha(level.alpha_value())?;
    let bands = multiplier_bands(&columns, clusters.as_ref(), alpha, bootstrap)?;
    let cells = fit
        .estimated()
        .zip(&bands.se)
        .map(|((cell, estimate), &se)| {
            let margin = se.map(|se| bands.critical * se);
            AttGtBand {
                group: cell.group,
                time: cell.time,
                event_time: cell.event_time,
                estimate: estimate.estimate,
                se,
                lower: margin.map(|margin| estimate.estimate - margin),
                upper: margin.map(|margin| estimate.estimate + margin),
            }
        })
        .collect();
    Ok(AttGtBands {
        critical_value: bands.critical,
        cells,
        level,
    })
}
