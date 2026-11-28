//! Sandwich covariance shared by the two-part, Tweedie and log-normal models.

use epistat_inference::Clusters;

use crate::error::Error;
use epistat_inference::sandwich as epistat_sandwich;
use epistat_linalg::Attempt;
use epistat_linalg::{Mat, MatRef};

/// The sandwich meat built from per-row score residuals `r_i = residual(i)`.
///
/// Without clusters this is `Σ_i r_i² x_i x_iᵀ`. With cluster labels it is
/// `Σ_g s_g s_gᵀ` with `s_g = Σ_{i∈g} r_i x_i`, and the number of clusters is
/// returned alongside. Clusters are numbered in order of first appearance and score
/// sums are accumulated in row order, so the meat does not depend on hashing.
///
/// # Errors
///
/// [`Error::Input`] for more than `u32::MAX` cluster labels.
///
/// # Panics
///
/// Panics if `clusters` does not have one label per row of `x`. The fitters validate
/// this before they get here.
pub(crate) fn score_meat(
    x: MatRef<'_, f64>,
    residual: impl Fn(usize) -> f64 + Sync,
    clusters: Option<&[u32]>,
) -> Result<(Mat<f64>, Option<usize>), Error> {
    if let Some(labels) = clusters {
        epistat_core::check::fits_u32("clusters", labels.len())?;
    }
    // With at most `u32::MAX` labels there are at most `u32::MAX` clusters,
    // the only case the numbering refuses.
    let clusters = clusters
        .map(|labels| Clusters::from_labels_first_seen(labels).expect("at most u32::MAX clusters"));
    let meat =
        epistat_sandwich::meat(x, residual, clusters.as_ref()).expect("one cluster label per row");
    Ok((meat, clusters.map(|clusters| clusters.count())))
}

/// `A⁻¹ M A⁻ᵀ` for the information matrix `A` and meat `M`, through two solves with
/// the symmetric fallback sequence. `None` when no factorisation gives a finite solution.
pub(crate) fn sandwich(information: MatRef<'_, f64>, meat: MatRef<'_, f64>) -> Option<Mat<f64>> {
    epistat_sandwich::sandwich(information, meat, &Attempt::SYMMETRIC).ok()
}
