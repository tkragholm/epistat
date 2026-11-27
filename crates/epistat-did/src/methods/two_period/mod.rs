//! The two-period difference-in-differences estimators of `DRDID`.
//!
//! [`estimate`] takes a [`TwoPeriodData`], a panel of outcome changes or a
//! repeated cross-section of outcomes, and [`TwoPeriodOptions`], and returns
//! a [`TwoPeriod`]: the ATT as an [`Estimate`], the counts of the arms and the
//! influence function. The estimator is an option:
//!
//! | [`Estimator`] | panel | repeated cross-section |
//! |---|---|---|
//! | [`Estimator::Improved`], the default | `drdid_imp_panel` | `drdid_imp_rc` |
//! | [`Estimator::DoublyRobust`] | `drdid_panel` | `drdid_rc` |
//! | [`Estimator::TraditionalDoublyRobust`] | refused | `drdid_rc1` |
//! | [`Estimator::InverseProbabilityWeighting`] | `std_ipw_did_panel` | `std_ipw_did_rc` |
//! | [`Estimator::OutcomeRegression`] | `reg_did_panel` | `reg_did_rc` |
//!
//! The covariates are given without an intercept and every estimator adds it
//! as the first column of its design. The nuisance models of the panel
//! doubly robust estimator are public in [`nuisance`].
//!
//! # Algorithm
//!
//! With `D` the treatment indicator, `X` the design row with its intercept,
//! `p(X)` the logistic propensity score fitted by maximum likelihood with the
//! prior weights, and `m(X)` the weighted least-squares regression of the
//! outcome change on `X` among the comparators, the panel doubly robust ATT of
//! Sant'Anna and Zhao \[1\] is
//!
//! ```text
//! ATT    = E[ (w_1 - w_0(X)) (dY - m(X)) ],
//! w_1    = D / E[D],
//! w_0(X) = (p(X) (1 - D) / (1 - p(X))) / E[p(X) (1 - D) / (1 - p(X))],
//! ```
//!
//! weights normalised to mean one and comparators with `p >= 0.995` trimmed,
//! as `DRDID`'s default `trim.level`. The improved estimators replace the
//! maximum-likelihood score by the calibrated score of Sant'Anna and Zhao and
//! weight the outcome regression by the odds of the score; on a repeated
//! cross-section the outcome is regressed by arm and period. Inverse
//! probability weighting drops `m`, and outcome regression drops the weights
//! `w_0`. The design is pruned to the columns each fit can identify: a
//! column constant within an arm, or a combination of the columns before it,
//! is dropped and counted in [`TwoPeriod::columns_dropped`].
//!
//! # Inference
//!
//! The influence function `psi_i` is on the per-unit scale of [`Influence`],
//! one value per row, and includes the estimation of the nuisance models. The
//! standard error is `sqrt(mean((psi - mean(psi))^2) / n)`, `DRDID`'s
//! `sd(psi) sqrt(n - 1) / n`, and the interval is `att +- z se` at
//! [`TwoPeriodOptions::level`]; with [`TwoPeriodOptions::bootstrap`] it is the
//! percentile interval of Rademacher multiplier draws of `psi`, `DRDID`'s
//! `boot = TRUE` with `boot.type = "multiplier"`. The improved estimators'
//! interval is `att +- 1.96 se` and is labelled 95% whatever the level. With
//! clusters in the data the standard error is `sqrt(sum_c s_c^2)`, `s_c` the
//! sum of the centred `psi_i` of cluster `c` divided by `n`, without the
//! factor `G / (G - 1)`, as `did` reports it with `clustervars`, and the
//! interval is the normal one. Sums run in a fixed order, so the results do
//! not depend on the number of threads.
//!
//! # References
//!
//! 1. Sant'Anna PHC, Zhao J. Doubly robust difference-in-differences
//!    estimators. J Econom 2020;219(1):101-22.
//!    doi:10.1016/j.jeconom.2020.06.003
//!
//! The R reference is `DRDID` 1.2.2.

mod data;
pub mod nuisance;

pub use data::{Design, TwoPeriodData, TwoPeriodDataBuilder};

use epistat_core::{Bootstrap, Estimate, InputError, Level};
use epistat_inference::{ClusterRule, Clusters, Influence};

use crate::error::Error;
use crate::inference::did_clusters::standard_error;
use crate::inference::{estimate_from, normal_interval};
use crate::methods::drdid;
use crate::types::{
    DrDidConfig, DrDidError, DrDidEstimate, DrDidObservation, DrDidRepeatedObservation,
};

/// The two-period estimator, `DRDID`'s estimator functions.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Estimator {
    /// The improved, locally efficient doubly robust estimator with the
    /// calibrated propensity score, `drdid_imp_panel` and `drdid_imp_rc`,
    /// `DRDID::drdid`'s default `estMethod = "imp"`.
    #[default]
    Improved,
    /// The doubly robust estimator with the maximum-likelihood propensity
    /// score, `drdid_panel` on a panel and the locally efficient `drdid_rc`
    /// on a repeated cross-section, `estMethod = "trad"` and `did`'s `"dr"`.
    DoublyRobust,
    /// The traditional doubly robust repeated cross-section estimator,
    /// `drdid_rc1`, which is not locally efficient. Refused on a panel.
    TraditionalDoublyRobust,
    /// Inverse probability weighting with normalised weights,
    /// `std_ipw_did_panel` and `std_ipw_did_rc`.
    InverseProbabilityWeighting,
    /// Outcome regression, `reg_did_panel` and `reg_did_rc`.
    OutcomeRegression,
}

/// The nuisance models of the two-period estimators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NuisanceOptions {
    /// Propensity scores are clipped to `[propensity_clip, 1 - propensity_clip]`;
    /// in `(0, 0.5)`. The default is 1e-6.
    pub propensity_clip: f64,
    /// Ridge penalty on the diagonal of the outcome regressions' normal
    /// equations, finite and not negative. The default is 1e-8.
    pub ridge: f64,
    /// Iterations of the propensity fit, at least 1. `glm`'s `maxit` is 25;
    /// the default of 200 lets a fit near separation finish.
    pub max_iterations: usize,
    /// Convergence tolerance of the propensity fit, positive, `glm`'s
    /// `epsilon`. The default is 1e-8, `glm`'s.
    pub tolerance: f64,
    /// Whether the traditional repeated cross-section doubly robust estimator,
    /// and the efficient ATT(g,t) estimator on a repeated cross-section, which
    /// uses it, compute as `epistat-did` 0.1 and `did_methods` 0.2 did, for
    /// reproducing results made with them: the propensity score fitted
    /// without the sampling weights and clipped, no trimming, and an influence
    /// function without the estimation of the nuisance models. Off, the
    /// default, the estimator is `DRDID::drdid_rc1`. The other estimators
    /// ignore it.
    pub legacy_repeated_cross_section: bool,
}

impl Default for NuisanceOptions {
    fn default() -> Self {
        Self {
            propensity_clip: 1e-6,
            ridge: 1e-8,
            max_iterations: 200,
            tolerance: 1e-8,
            legacy_repeated_cross_section: false,
        }
    }
}

impl NuisanceOptions {
    /// The options as an error when one is outside its domain.
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if !(self.propensity_clip > 0.0 && self.propensity_clip < 0.5) {
            return Err(domain("propensity_clip", self.propensity_clip, "(0, 0.5)"));
        }
        if !(self.ridge.is_finite() && self.ridge >= 0.0) {
            return Err(domain("ridge", self.ridge, "[0, inf)"));
        }
        if self.max_iterations == 0 {
            return Err(domain("max_iterations", 0.0, "[1, inf)"));
        }
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(domain("tolerance", self.tolerance, "(0, inf)"));
        }
        Ok(())
    }

    /// The internal configuration of the estimators at `level`, with
    /// `bootstrap` for their intervals.
    pub(crate) fn config(&self, level: Level, bootstrap: Option<Bootstrap>) -> DrDidConfig {
        let (replicates, seed) =
            bootstrap.map_or((0, 0), |bootstrap| (bootstrap.replicates, bootstrap.seed));
        DrDidConfig {
            confidence_level: level.confidence_value(),
            propensity_clip: self.propensity_clip,
            ridge: self.ridge,
            max_iter: self.max_iterations,
            tol: self.tolerance,
            bootstrap_reps: replicates,
            bootstrap_seed: seed,
            legacy_repeated_cross_section: self.legacy_repeated_cross_section,
        }
    }
}

pub(crate) fn domain(what: &'static str, value: f64, domain: &'static str) -> Error {
    InputError::Domain {
        what,
        value,
        domain,
    }
    .into()
}

/// The options of [`estimate`], with `DRDID::drdid`'s defaults.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TwoPeriodOptions {
    /// The estimator, `estMethod` and the estimator functions.
    pub estimator: Estimator,
    /// The nuisance models.
    pub nuisance: NuisanceOptions,
    /// The confidence level of the interval.
    pub level: Level,
    /// A multiplier bootstrap for the interval, `boot = TRUE` with
    /// `boot.type = "multiplier"`: the percentile interval of Rademacher
    /// draws of the influence function, replicate `r` seeded with
    /// `replicate_seed(seed, r)`. `None`, the default, gives the normal
    /// interval. Not with clusters, and not used by the improved estimators.
    pub bootstrap: Option<Bootstrap>,
}

/// A two-period ATT.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct TwoPeriod {
    /// The estimator that formed it.
    pub estimator: Estimator,
    /// The ATT with its standard error, interval and normal p-value; a part
    /// that is not finite is `None`.
    pub estimate: Estimate,
    /// Treated rows.
    pub n_treated: usize,
    /// Comparison rows.
    pub n_control: usize,
    /// The sum of the prior weights.
    pub total_weight: f64,
    /// Design columns dropped because the fit's own rows could not identify
    /// them: constant within an arm, or a combination of the columns before
    /// them. Zero for a full-rank design.
    pub columns_dropped: usize,
    /// The influence function, one value per row, under
    /// [`ClusterRule::NoSmallSample`], the rule of `DRDID` and `did`.
    pub influence: Influence,
}

/// The two-period ATT of `data` by `options.estimator`.
///
/// ```
/// use epistat_core::Level;
/// use epistat_did::two_period::{self, Estimator, TwoPeriodData, TwoPeriodOptions};
///
/// // Two periods of a repeated cross-section, the treated rising by 2 more.
/// let treated = [true, true, true, true, false, false, false, false];
/// let post = [false, false, true, true, false, false, true, true];
/// let y = [1.0, 1.2, 4.1, 3.9, 1.1, 0.9, 2.0, 2.1];
/// let data = TwoPeriodData::repeated_cross_section(&treated, &post, &y).build()?;
/// let fit = two_period::estimate(&data, &TwoPeriodOptions {
///     estimator: Estimator::OutcomeRegression,
///     level: Level::confidence(0.9)?,
///     ..TwoPeriodOptions::default()
/// })?;
/// // The default ridge of 1e-8 moves it by about 5e-9.
/// assert!((fit.estimate.estimate - 1.85).abs() < 1e-6);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// - [`Error::Input`] for a nuisance option outside its domain, or a
///   bootstrap without replicates.
/// - [`Error::Unsupported`] for the traditional doubly robust estimator on a
///   panel, and a bootstrap with clusters.
/// - [`Error::NoTreated`] and [`Error::NoComparison`] for an arm without rows,
///   or one that the trim leaves without weight, and [`Error::EmptyCell`] for
///   a repeated cross-section without rows in one arm and period.
/// - [`Error::Singular`] for a nuisance model that cannot be fitted.
pub fn estimate(data: &TwoPeriodData<'_>, options: &TwoPeriodOptions) -> Result<TwoPeriod, Error> {
    options.nuisance.validate()?;
    if let Some(bootstrap) = options.bootstrap {
        if bootstrap.replicates == 0 {
            return Err(domain("bootstrap replicates", 0.0, "[1, inf)"));
        }
        if data.clusters().is_some() {
            return Err(Error::Unsupported {
                what: "a two-period bootstrap with clusters",
            });
        }
    }
    let config = options.nuisance.config(options.level, options.bootstrap);
    let fit = if data.is_panel() {
        panel(data, options.estimator, config)?
    } else {
        repeated(data, options.estimator, config)?
    };
    let level = match options.estimator {
        Estimator::Improved => Level::confidence(0.95)?,
        _ => options.level,
    };
    finish(data, options.estimator, fit, level)
}

/// The canonical 2x2 difference in means of a repeated cross-section without
/// covariates: the change of the treated arm's weighted mean minus the
/// comparison arm's, with the standard error
/// `sqrt(sum over the four cells of v_c / n_c)`, `v_c` the weighted variance of
/// the cell with divisor `sum w` and `n_c = (sum w)^2 / sum w^2` its effective
/// size, and the normal interval at `level`.
///
/// # Errors
///
/// [`Error::Unsupported`] for a panel or covariates, and [`Error::EmptyCell`]
/// for an arm and period without rows.
pub fn two_by_two(data: &TwoPeriodData<'_>, level: Level) -> Result<Estimate, Error> {
    let Design::RepeatedCrossSection { post, outcome } = data.design() else {
        return Err(Error::Unsupported {
            what: "the 2x2 difference in means on a panel",
        });
    };
    if data.covariates().is_some() {
        return Err(Error::Unsupported {
            what: "covariates in the 2x2 difference in means",
        });
    }
    let rows = (0..data.len()).map(|i| (data.treated()[i], post[i], outcome[i], data.weight(i)));
    let summary = crate::methods::standard::summarize_two_by_two(rows)?;
    let (att, se, ci) =
        crate::methods::standard::estimate_att_from_summary(summary, level.confidence_value());
    Ok(estimate_from(att, se, ci, level))
}

fn panel(
    data: &TwoPeriodData<'_>,
    estimator: Estimator,
    config: DrDidConfig,
) -> Result<DrDidEstimate, Error> {
    let Design::Panel { change } = data.design() else {
        unreachable!("a panel")
    };
    let weights = data.weight_column();
    let (design, feature_count) = data.design_flat();
    let input = drdid::panel::PanelFlatInput {
        treated: data.treated(),
        delta_outcome: change,
        weight: &weights,
        design_matrix_flat: &design,
        feature_count,
    };
    let fit = match estimator {
        Estimator::DoublyRobust => drdid::panel::estimate_drdid_panel_flat(input, config),
        Estimator::InverseProbabilityWeighting => {
            drdid::ipw_or::estimate_ipw_did_panel_flat(input, config)
        }
        Estimator::OutcomeRegression => drdid::ipw_or::estimate_or_did_panel_flat(input, config),
        Estimator::Improved => {
            let rows: Vec<DrDidObservation> = (0..data.len())
                .map(|i| DrDidObservation {
                    treated: data.treated()[i],
                    delta_outcome: change[i],
                    weight: weights[i],
                    covariates: design[i * feature_count..(i + 1) * feature_count].to_vec(),
                })
                .collect();
            drdid::improved::estimate_drdid_improved_panel(&rows, config)
        }
        Estimator::TraditionalDoublyRobust => {
            return Err(Error::Unsupported {
                what: "the traditional doubly robust estimator on a panel",
            });
        }
    };
    fit.map_err(Error::from)
}

fn repeated(
    data: &TwoPeriodData<'_>,
    estimator: Estimator,
    config: DrDidConfig,
) -> Result<DrDidEstimate, Error> {
    let Design::RepeatedCrossSection { post, outcome } = data.design() else {
        unreachable!("a repeated cross-section")
    };
    let rows = repeated_rows(data, post, outcome);
    let fit = match estimator {
        Estimator::Improved => {
            drdid::improved::estimate_drdid_improved_repeated_cross_section(&rows, config)
        }
        Estimator::DoublyRobust => {
            drdid::repeated_efficient::estimate_drdid_repeated_efficient(&rows, config)
        }
        Estimator::TraditionalDoublyRobust => {
            drdid::repeated::estimate_drdid_repeated_cross_section(&rows, config)
        }
        Estimator::InverseProbabilityWeighting => {
            drdid::ipw_or::estimate_ipw_did_repeated_cross_section(&rows, config)
        }
        Estimator::OutcomeRegression => {
            drdid::ipw_or::estimate_or_did_repeated_cross_section(&rows, config)
        }
    };
    fit.map_err(Error::from)
}

/// The rows of a repeated cross-section in the estimators' internal form,
/// covariates without the intercept, which they prepend.
pub(crate) fn repeated_rows(
    data: &TwoPeriodData<'_>,
    post: &[bool],
    outcome: &[f64],
) -> Vec<DrDidRepeatedObservation> {
    (0..data.len())
        .map(|i| DrDidRepeatedObservation {
            treated: data.treated()[i],
            post_period: post[i],
            outcome: outcome[i],
            weight: data.weight(i),
            covariates: data.covariate_row(i),
        })
        .collect()
}

/// The influence function of `data`'s rows, one column.
pub(crate) fn row_influence(values: Vec<f64>) -> Result<Influence, Error> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::NotFinite);
    }
    let units = (0..values.len())
        .map(|i| u32::try_from(i).expect("the data have at most u32::MAX rows"))
        .collect();
    Ok(Influence::new(
        values,
        units,
        1,
        ClusterRule::NoSmallSample,
    )?)
}

/// The standard error and interval of an estimate from its influence
/// function, clustered when `data` has clusters, else as the estimator gave
/// them.
pub(crate) fn clustered(
    data: &TwoPeriodData<'_>,
    att: f64,
    se: f64,
    ci: (f64, f64),
    influence: &[f64],
    level: Level,
) -> (f64, (f64, f64)) {
    match data.clusters() {
        Some(codes) => {
            let se = standard_error(influence, &Clusters::from_codes(codes));
            (se, normal_interval(att, se, level.confidence_value()))
        }
        None => (se, ci),
    }
}

fn finish(
    data: &TwoPeriodData<'_>,
    estimator: Estimator,
    fit: DrDidEstimate,
    level: Level,
) -> Result<TwoPeriod, Error> {
    let (se, ci) = clustered(
        data,
        fit.att,
        fit.se,
        (fit.ci_low, fit.ci_high),
        &fit.influence_function,
        level,
    );
    Ok(TwoPeriod {
        estimator,
        estimate: estimate_from(fit.att, se, ci, level),
        n_treated: fit.treated_n,
        n_control: fit.control_n,
        total_weight: fit.total_weight,
        columns_dropped: fit.design_columns_dropped,
        influence: row_influence(fit.influence_function)?,
    })
}

impl From<DrDidError> for Error {
    fn from(error: DrDidError) -> Self {
        match error {
            DrDidError::EmptyInput => InputError::Empty { what: "rows" }.into(),
            DrDidError::NoTreated => Self::NoTreated,
            DrDidError::NoControl => Self::NoComparison,
            DrDidError::MissingCell { cell } => {
                let (treated, post) = cell.flags();
                Self::EmptyCell { treated, post }
            }
            DrDidError::InvalidWeight { value } => domain("weights", value, "(0, inf)"),
            DrDidError::InvalidOutcome { .. } => InputError::NotFinite {
                what: "outcome",
                row: 0,
            }
            .into(),
            DrDidError::InvalidCovariate { .. } => InputError::NotFinite {
                what: "covariates",
                row: 0,
            }
            .into(),
            DrDidError::InconsistentCovariateCount { expected, actual } => InputError::Length {
                what: "covariates",
                expected,
                found: actual,
            }
            .into(),
            DrDidError::InvalidConfig(_) => Self::NotFinite,
            DrDidError::SingularSystem => Self::Singular,
        }
    }
}

#[cfg(test)]
mod tests {
    use epistat_linalg::MatRef;

    use super::{Estimator, TwoPeriod, TwoPeriodData, TwoPeriodOptions, estimate};

    fn bits(fit: &TwoPeriod) -> Vec<u64> {
        let estimate = &fit.estimate;
        let mut out = vec![
            estimate.estimate.to_bits(),
            estimate.se.map_or(0, f64::to_bits),
            estimate.lower().map_or(0, f64::to_bits),
            estimate.upper().map_or(0, f64::to_bits),
            fit.total_weight.to_bits(),
            fit.n_treated as u64,
            fit.n_control as u64,
            fit.columns_dropped as u64,
        ];
        out.extend(fit.influence.column(0).iter().map(|value| value.to_bits()));
        out
    }

    /// A caller that passes the intercept in a finished panel design, as a
    /// design `[1, x]`, gets the bits of the covariates `x` to which the
    /// estimators add it.
    #[test]
    fn covariates_match_a_finished_design_with_its_intercept() {
        let n = 40_usize;
        let rows: Vec<f64> = (0..n).map(crate::util::usize_to_f64).collect();
        let treated: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let x1: Vec<f64> = rows.iter().map(|&i| (0.37 * i).sin()).collect();
        let x2: Vec<f64> = rows.iter().map(|&i| (0.11 * i).cos() + 0.5).collect();
        let weight: Vec<f64> = rows.iter().map(|&i| 1.0 + (0.7 * i).sin().abs()).collect();
        let change: Vec<f64> = (0..n)
            .map(|i| {
                let effect = if treated[i] { 1.5 } else { 0.0 };
                x1[i] - x2[i] / 2.0 + effect + (1.3 * rows[i]).cos() / 10.0
            })
            .collect();
        let covariates: Vec<f64> = x1.iter().chain(&x2).copied().collect();
        let design: Vec<f64> = std::iter::repeat_n(1.0, n)
            .chain(x1.iter().copied())
            .chain(x2.iter().copied())
            .collect();
        for estimator in [Estimator::DoublyRobust, Estimator::Improved] {
            let options = TwoPeriodOptions {
                estimator,
                ..TwoPeriodOptions::default()
            };
            let added = TwoPeriodData::panel(&treated, &change)
                .weights(&weight)
                .covariates(MatRef::from_column_major_slice(&covariates, n, 2))
                .build()
                .expect("covariates");
            let finished = TwoPeriodData::panel(&treated, &change)
                .weights(&weight)
                .design(MatRef::from_column_major_slice(&design, n, 3))
                .build()
                .expect("design");
            let added = estimate(&added, &options).expect("added intercept");
            let finished = estimate(&finished, &options).expect("finished design");
            assert_eq!(bits(&added), bits(&finished), "{estimator:?}");
        }
    }
}
