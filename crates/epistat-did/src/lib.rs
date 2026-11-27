//! Difference-in-differences: group-time average treatment effects for
//! staggered adoption, their aggregations, two-period doubly robust
//! estimators, with influence-function inference.
//!
//! # The estimators
//!
//! - ATT(g,t), the average effect at time `t` on the cohort first treated at
//!   `g` \[1\], in [`att_gt`]: [`att_gt::estimate`] on a [`StaggeredPanel`]
//!   by the doubly robust estimator, inverse probability weighting or outcome
//!   regression, on a panel or repeated cross-sections, after
//!   `did::att_gt`'s pre-processing, and the efficient estimator over all
//!   baseline periods \[3\]. The result aggregates by event time, cohort,
//!   calendar time or overall as `did::aggte` does, with the multiplier
//!   bootstrap and uniform bands ([`att_gt::AttGt::aggregate`]), and gives
//!   simultaneous bands over the cells, the Wald pre-test of `att_gt` and the
//!   event study HonestDiD takes.
//! - The two-period estimators of `DRDID` \[2\] on a [`TwoPeriodData`], a
//!   panel of outcome changes or a repeated cross-section:
//!   [`two_period::estimate`] by the improved, doubly robust, traditional
//!   doubly robust, inverse probability weighting or outcome regression
//!   estimator, with the nuisance models of the panel doubly robust
//!   estimator in [`two_period::nuisance`].
//! - Compositional changes in repeated cross-sections \[4\]:
//!   [`did_cc::estimate`] under either assumption and the Hausman test of
//!   stationarity [`did_cc::test`].
//! - Doubly robust triple differences \[5\]: [`triple_did::estimate`].
//! - The average causal response to a continuous treatment \[6\]:
//!   [`continuous::estimate`], with the sieve's basis as a
//!   [`continuous::BasisSpec`].
//! - Two-way fixed-effects regressions, static and event study with binned
//!   end points, as `fixest::feols` fits them: [`twfe::fit`], and the
//!   Goodman-Bacon decomposition \[11\] of the static estimate into its 2x2
//!   comparisons as `bacondecomp::bacon`: [`bacon::decompose`].
//! - Event-study effects for treatments that switch on and off or take
//!   several values, as `DIDmultiplegtDYN` of de Chaisemartin and
//!   D'Haultfoeuille \[8\]: [`dcdh::estimate`] on a [`SwitchingPanel`],
//!   whose treatment need not be absorbing and whose outcomes and
//!   treatments may be missing.
//! - Synthetic difference-in-differences \[9\], synthetic control and the
//!   plain difference-in-differences beside them, as R's `synthdid` computes
//!   them: [`synthdid::estimate`] with [`SynthDidOptions`], standard errors
//!   by placebo, bootstrap or jackknife through [`SynthDidOptions::variance`],
//!   and staggered adoption cohort by cohort in
//!   [`synthdid::estimate_staggered`]. The module [`synthdid`] gives the
//!   algorithm.
//! - The power of the pre-trends test and the distortion after passing it,
//!   after Roth \[10\], are in the crate `epistat-honestdid`, which takes
//!   the event study of [`att_gt::AttGt::event_study`].
//! - The imputation estimator of Borusyak, Jaravel and Spiess \[12\]:
//!   [`imputation::estimate`]; and the two-stage estimator of Gardner \[13\]:
//!   [`two_stage::estimate`]. Both absorb the unit and period effects of the
//!   untreated rows with `epistat_glm::absorb`, so their cost is linear in
//!   the rows.
//! - The interaction-weighted event study of Sun and Abraham \[14\], as
//!   `fixest::sunab` and `fixest::aggregate` compute it: [`sunab::fit`], with
//!   the cohort and relative period coefficients and their aggregates by
//!   relative period, by cohort and overall.
//! - Wooldridge's extended two-way fixed-effects regression \[15\] and its
//!   average effects, as `etwfe::etwfe` and `etwfe::emfx` compute them:
//!   [`etwfe::fit`], for linear models and, by [`EtwfeOptions::family`], for
//!   the Poisson and logit models.
//!
//! The fixed-effects family takes a validated [`StaggeredPanel`], with the
//! clusters of its covariances in the panel, and returns its estimates as
//! [`epistat_core::Estimate`]s with intervals at the options' `level`, its
//! covariances as [`epistat_core::Covariance`] and its influence functions as
//! [`epistat_inference::Influence`], each under the cluster rule of its R
//! reference.
//! - The two-period ATT when the pre-treatment outcome is missing at random,
//!   after Testa, Kennedy and Reimherr \[16\]: [`att_missing_pre`] with
//!   parametric nuisance models, with or without cross-fitting, and
//!   [`att_missing_pre_from_predictions`] for out-of-fold predictions of any
//!   learner; the module [`missing`] gives the algorithm.
//! - The sensitivity analysis of Rambachan and Roth \[7\] is in the crate
//!   `epistat-honestdid`, on the same event study.
//!
//! # Inference
//!
//! Each estimator returns the influence function `psi_i` of every unit, and
//! the standard error is `sqrt(mean((psi_i - mean(psi))^2) / n)`. The intervals
//! are `att +- z se`, or percentile intervals of a Rademacher multiplier
//! bootstrap when the options ask for a `Bootstrap`; those of the
//! improved two-period estimators are `att +- 1.96 se`. With clusters, the
//! units are replaced by the sums of the centred `psi_i` within each cluster,
//! divided by `n`, without the factor `G / (G - 1)`, as in `did`;
//! [`did_standard_error`] and [`did_covariance`] are that variance. The
//! multiplier bootstrap draws a Rademacher weight per unit or cluster and
//! replicate, each replicate seeded from the caller's seed and its index.
//! Sums run in a fixed order, so the results do not depend on the number of
//! threads.
//!
//! # References
//!
//! 1. Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time
//!    periods. J Econom 2021;225(2):200-30. doi:10.1016/j.jeconom.2020.12.001
//! 2. Sant'Anna PHC, Zhao J. Doubly robust difference-in-differences
//!    estimators. J Econom 2020;219(1):101-22.
//!    doi:10.1016/j.jeconom.2020.06.003
//! 3. Chen X, Sant'Anna PHC, Xie H. Efficient difference-in-differences and
//!    event study estimators. Working paper, 2025.
//! 4. Sant'Anna PHC, Xu Q. Difference-in-differences with compositional
//!    changes. Working paper, 2023.
//! 5. Ortiz-Villavicencio M, Sant'Anna PHC. Better understanding triple
//!    differences estimators. Working paper, 2025.
//! 6. Callaway B, Goodman-Bacon A, Sant'Anna PHC. Difference-in-differences
//!    with a continuous treatment. Working paper, 2024.
//! 7. Rambachan A, Roth J. A more credible approach to parallel trends. Rev
//!    Econ Stud 2023;90(5):2555-91. doi:10.1093/restud/rdad018
//! 8. de Chaisemartin C, D'Haultfoeuille X. Difference-in-differences
//!    estimators of intertemporal treatment effects. Rev Econ Stat 2024.
//!    doi:10.1162/rest_a_01414
//! 9. Arkhangelsky D, Athey S, Hirshberg DA, Imbens GW, Wager S. Synthetic
//!    difference-in-differences. Am Econ Rev 2021;111(12):4088-118.
//!    doi:10.1257/aer.20190159
//! 10. Roth J. Pretest with caution: event-study estimates after testing for
//!     parallel trends. Am Econ Rev Insights 2022;4(3):305-22.
//!     doi:10.1257/aeri.20210236
//! 11. Goodman-Bacon A. Difference-in-differences with variation in
//!     treatment timing. J Econom 2021;225(2):254-77.
//!     doi:10.1016/j.jeconom.2021.03.014
//! 12. Borusyak K, Jaravel X, Spiess J. Revisiting event-study designs:
//!     robust and efficient estimation. Rev Econ Stud 2024;91(6):3253-85.
//!     doi:10.1093/restud/rdae007
//! 13. Gardner J. Two-stage differences in differences. Working paper,
//!     2021. arXiv:2207.05943
//! 14. Sun L, Abraham S. Estimating dynamic treatment effects in event
//!     studies with heterogeneous treatment effects. J Econom
//!     2021;225(2):175-99. doi:10.1016/j.jeconom.2020.09.006
//! 15. Wooldridge JM. Two-way fixed effects, the two-way Mundlak regression,
//!     and difference-in-differences estimators. Empir Econ
//!     2025;69:2545-87. doi:10.1007/s00181-025-02807-z
//! 16. Testa L, Kennedy EH, Reimherr M. Efficient difference-in-differences
//!     estimation when outcomes are missing at random. arXiv:2509.25009
//!     [stat.ME]; 2025.
#![warn(missing_docs)]

pub(crate) mod error;
pub(crate) mod estimators;
pub mod inference;
pub(crate) mod methods;
#[cfg(test)]
mod tests;
pub(crate) mod types;
pub(crate) mod util;

pub use error::Error;
pub use inference::did_clusters::{did_cluster_sums, did_covariance, did_standard_error};
pub use methods::att_gt;
pub use methods::bacon;
pub use methods::bacon::{BaconComparison, BaconDecomposition, BaconKind, BaconWithin};
pub use methods::continuous;
pub use methods::dcdh;
pub use methods::dcdh::{
    Dcdh, DcdhAvgTimePeriods, DcdhBootstrap, DcdhBootstrapTerm, DcdhDesign, DcdhEffectsEqual,
    DcdhHeterogeneity, DcdhOptions, DcdhPath, DcdhPathResult, DcdhSubgroup, DcdhSwitchDate,
    DcdhSwitchDates, DcdhSwitchers, DcdhTerm, SwitchingPanel, SwitchingPanelBuilder,
};
pub use methods::did_cc;
pub use methods::etwfe;
pub use methods::etwfe::{
    EtwfeCoefficient, EtwfeControl, EtwfeEffect, EtwfeEffects, EtwfeFamily, EtwfeFit, EtwfeOptions,
    EtwfeTerm, EtwfeType, EtwfeVcov,
};
pub use methods::imputation;
pub use methods::imputation::{
    Imputation, ImputationEffect, ImputationOptions, ImputationPretrends, ImputationTarget,
    ImputationTerm, PretrendTerm, Pretrends,
};
pub use methods::missing;
pub use methods::missing::{
    CrossFitFolds, MissingAssumption, MissingAtt, MissingFolds, MissingModels, MissingNuisance,
    MissingOptions, MissingPreData, MissingPredictions, NestedOutcome, att_missing_pre,
    att_missing_pre_from_predictions, missing_cross_fit_folds,
};
pub use methods::staggered::{StaggeredPanel, StaggeredPanelBuilder};
pub use methods::sunab;
pub use methods::sunab::{
    SunabAggregate, SunabBins, SunabCell, SunabControl, SunabFit, SunabOptions, SunabReference,
    SunabTerm,
};
pub use methods::synthdid;
pub use methods::synthdid::{
    StaggeredSynthDid, SynthCohort, SynthDid, SynthDidOptions, SynthDraws, SynthFit,
    SynthReplicates, SynthTrace, SynthUndefined, SynthVariance, SynthWeights,
};
pub use methods::triple_did;
pub use methods::twfe;
pub use methods::twfe::{TwfeFit, TwfeOptions, TwfeSpec, TwfeTerm};
pub use methods::two_period;
pub use methods::two_period::{TwoPeriodData, TwoPeriodDataBuilder};
pub use methods::two_stage;
pub use methods::two_stage::{TwoStage, TwoStageOptions, TwoStageSpec, TwoStageTerm};
