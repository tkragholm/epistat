//! Difference-in-differences: group-time average treatment effects for
//! staggered adoption, their aggregations, two-period doubly robust
//! estimators and the HonestDiD sensitivity analysis, with influence-function
//! inference.
//!
//! # The estimators
//!
//! - ATT(g,t), the average effect at time `t` on the cohort first treated at
//!   `g` \[1\]: [`estimate_att_gt`] without covariates;
//!   [`estimate_att_gt_dr_panel_with_influence`],
//!   [`estimate_att_gt_ipw_panel_with_influence`] and
//!   [`estimate_att_gt_or_panel_with_influence`] by the doubly robust
//!   estimator, inverse probability weighting and outcome regression on a
//!   panel; [`estimate_att_gt_dr_efficient_with_influence`],
//!   [`estimate_att_gt_ipw_with_influence`] and
//!   [`estimate_att_gt_or_with_influence`] on repeated cross-sections, with
//!   [`estimate_att_gt_dr_with_influence`] for the traditional doubly robust
//!   one. These apply `did::att_gt`'s pre-processing first ([`AttGtSample`]).
//! - Their aggregations as `aggte` of R's `did`: [`aggregate_att_gt`] by
//!   event time, cohort, calendar time or overall, with the influence of the
//!   estimated cohort shares, with the multiplier bootstrap and uniform bands
//!   of `aggte(bstrap, cband)` through [`AggteBootstrap`], and
//!   [`att_gt_mboot_bands`] for simultaneous bands over the cells.
//!   [`pretrend_wald_test`] is the Wald pre-test of `att_gt`.
//! - The two-period estimators of `DRDID` \[2\]: [`estimate_drdid_panel`],
//!   [`estimate_drdid_repeated_cross_section`], [`estimate_ipw_did_panel`],
//!   [`estimate_or_did_panel`], [`estimate_ipw_did_repeated_cross_section`],
//!   [`estimate_or_did_repeated_cross_section`], the improved estimators
//!   [`estimate_drdid_improved_panel`] and
//!   [`estimate_drdid_improved_repeated_cross_section`], and the locally
//!   efficient [`estimate_drdid_repeated_efficient`].
//! - The efficient ATT(g,t) over all baseline periods \[3\]:
//!   [`estimate_att_gt_efficient`] and [`estimate_att_gt_efficient_dr`].
//! - Compositional changes in repeated cross-sections \[4\]:
//!   [`estimate_did_cc_robust`], [`estimate_did_cc_stationary`] and
//!   [`test_did_cc_stationarity`].
//! - Doubly robust triple differences \[5\]: [`estimate_dr_ddd`].
//! - The average causal response to a continuous treatment \[6\]:
//!   [`estimate_acrt_sieve`].
//! - Two-way fixed-effects regressions, static and event study with binned
//!   end points, as `fixest::feols` fits them: [`estimate_twfe`], and the
//!   Goodman-Bacon decomposition \[11\] of the static estimate into its 2x2
//!   comparisons as `bacondecomp::bacon`: [`bacon_decomposition`].
//! - Event-study effects for treatments that switch on and off or take
//!   several values, as `DIDmultiplegtDYN` of de Chaisemartin and
//!   D'Haultfoeuille \[8\]: [`estimate_dcdh`].
//! - Synthetic difference-in-differences \[9\], synthetic control and the
//!   plain difference-in-differences beside them, as R's `synthdid` computes
//!   them: [`estimate_synthdid`] with [`SynthDidConfig`], standard errors by
//!   placebo, bootstrap or jackknife in [`synthdid_variance`], and staggered
//!   adoption cohort by cohort in [`estimate_synthdid_staggered`]. The
//!   module [`synthdid`] gives the algorithm.
//! - The power of the pre-trends test and the distortion after passing it,
//!   after Roth \[10\], in [`pretrends`].
//! - The imputation estimator of Borusyak, Jaravel and Spiess \[12\]:
//!   [`estimate_did_imputation`], with the algorithm in [`imputation`]; and
//!   the two-stage estimator of Gardner \[13\]: [`estimate_did2s`], in
//!   [`two_stage`]. Both absorb the unit and period effects of the untreated
//!   rows with `epistat_glm::absorb`, so their cost is linear in the rows.
//! - The interaction-weighted event study of Sun and Abraham \[14\], as
//!   `fixest::sunab` and `fixest::aggregate` compute it: [`estimate_sunab`],
//!   with the cohort and relative period coefficients and their summaries by
//!   relative period, by cohort and overall.
//! - With the `honest` feature, the sensitivity analysis of Rambachan and
//!   Roth \[7\] in [`inference::sensitivity`].
//!
//! # The estimate
//!
//! For the cohort `g`, a comparison group `C` (never treated, or not yet
//! treated by `t`) and the base period `b` (the last period before
//! `g - anticipation` for a universal base, and before treatment the period
//! before `t` for a varying one), the doubly robust ATT(g,t) is
//!
//! ```text
//! ATT(g,t) = E[ (w_g - w_C(X)) (Y_t - Y_b - m_C(X)) ],
//! w_g      = G_g / E[G_g],
//! w_C(X)   = (p(X) C / (1 - p(X))) / E[p(X) C / (1 - p(X))],
//! ```
//!
//! with `p(X)` the logistic propensity of belonging to `g` among the units of
//! `g` and `C`, fitted by maximum likelihood as R's `glm`, and `m_C(X)` the
//! weighted least-squares regression of `Y_t - Y_b` on `X` among the units of
//! `C` \[2\]. Outcome regression drops the weights `w_C` and inverse
//! probability weighting drops `m_C`. On a repeated cross-section the
//! outcome is not differenced within a unit, and the regressions are fitted
//! by group and period.
//!
//! An aggregation is a weighted mean of the cells. By event time `e`,
//!
//! ```text
//! theta(e) = sum over g of P(G = g | G + e <= T) ATT(g, g + e),
//! ```
//!
//! and the cohort, calendar and overall summaries are formed as in `aggte`
//! \[1\]. The cohort shares are estimated, so their influence function enters
//! the variance of the aggregate, as `did:::wif` adds it.
//!
//! # Inference
//!
//! Each estimator returns the influence function `psi_i` of every unit, and
//! the standard error is `sqrt(mean((psi_i - mean(psi))^2) / n)`. The intervals
//! are `att +- z se`, or percentile intervals of a Rademacher multiplier
//! bootstrap when [`DrDidConfig::bootstrap_reps`] is positive; those of the
//! improved two-period estimators are `att +- 1.96 se`. In [`aggregate_att_gt`],
//! [`att_gt_clustered_standard_errors`] and the bands, clusters replace the
//! units by the sums of the centred `psi_i` within each cluster, divided by
//! `n`, without the factor `G / (G - 1)`, as in `did`; [`did_standard_error`] and
//! [`did_covariance`] are that variance. The deprecated `EventStudyResult` and
//! the validated dataset use [`clustered_variance_from_index`] and its
//! siblings, which are the cluster-robust variance with that factor. The
//! multiplier bootstrap draws a Rademacher weight per unit or cluster and
//! replicate, each replicate seeded from the caller's seed and its index, and
//! a simultaneous band uses the `1 - alpha` quantile of the largest
//! studentised deviation, with the bootstrap standard error from the
//! interquartile range as `did::mboot` does. Sums run in a fixed order, so the
//! results do not depend on the number of threads.
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
#![warn(missing_docs)]

#[expect(
    deprecated,
    reason = "the deprecated workflow is built from its own deprecated types"
)]
pub(crate) mod dataset;
pub(crate) mod error;
pub(crate) mod estimators;
pub mod inference;
pub(crate) mod methods;
pub mod pretrends;
#[cfg(test)]
mod tests;
pub(crate) mod types;
pub(crate) mod util;

pub use dataset::{Diagnostic, PanelRecord, Severity, ValidationReport};
#[expect(
    deprecated,
    reason = "re-exports deprecated types until they are removed"
)]
pub use dataset::{InferenceDataset, ValidatedInferenceDataset};
pub use error::{DatasetError, InternalDidError};
pub use estimators::common::basis::{BSplineBasis, Basis, PolynomialBasis};
pub use inference::did_clusters::{
    DidClusterError, did_cluster_sums, did_covariance, did_standard_error,
};
pub use inference::export::{
    InferenceCovarianceRow, InferenceInfluenceRow, build_clustered_inference_covariance_rows,
    build_inference_covariance_rows, build_inference_influence_rows,
};
#[cfg(feature = "honest")]
pub use inference::sensitivity::{
    HonestAssessment, HonestBiasDirection, HonestConditionalConfidenceSet,
    HonestDirectionalFunctionalPoint, HonestDirectionalRegion, HonestDirectionalRegionDiagnostics,
    HonestDirectionalReport, HonestDirectionalReportRow, HonestEventStudyInput,
    HonestEventStudyPoint, HonestFunctionalAssessmentPoint, HonestIdentifiedSet,
    HonestJointPathConfig, HonestJointPathMethod, HonestJointPathPoint, HonestJointPathRegion,
    HonestJointPathReport, HonestJointPathReportRow, HonestMonotonicityDirection,
    HonestMultiFlciReport, HonestMultiFlciReportRow, HonestOptimizationSurfaceAdaptiveConfig,
    HonestOptimizationSurfaceAdaptiveRunConfig, HonestOptimizationSurfaceConfig,
    HonestOriginalConfidenceSet, HonestPostFunctional, HonestRelativeMagnitudeBound, HonestResult,
    HonestSensitivity, HonestStudyWorkflowResult, HonestWorkflowConfig,
    HonestWorkflowDirectionMode, LeastFavorableSimulation, RelativeMagnitudeConfidenceSetConfig,
    RelativeMagnitudeFlciProblem, RelativeMagnitudeFlciResult, RelativeMagnitudeHybrid,
    RelativeMagnitudeMultiFlciPoint, RelativeMagnitudeMultiFlciProblem,
    RelativeMagnitudeMultiFlciResult, SensitivitySummary, SensitivitySummaryMethod,
    SensitivitySummaryRow, SolverCounts, assess_honest_event_study_directional_region,
    assess_honest_event_study_directional_region_with_config, assess_honest_event_study_functional,
    assess_honest_event_study_functional_with_config, assess_honest_event_study_joint_path_region,
    assess_honest_event_study_joint_path_region_with_config,
    assess_honest_event_study_optimization_surface_region,
    assess_honest_event_study_optimization_surface_region_adaptive,
    assess_honest_event_study_optimization_surface_region_adaptive_with_config,
    assess_honest_event_study_optimization_surface_region_with_config,
    assess_honest_event_study_period, assess_honest_event_study_period_with_config,
    assess_honest_event_study_post_functional, assess_honest_event_study_post_functional_flci,
    assess_honest_event_study_post_functional_flci_with_config,
    assess_honest_event_study_post_functional_multi_flci,
    assess_honest_event_study_post_functional_multi_flci_with_config,
    assess_honest_event_study_post_functional_with_config, assess_honest_event_study_workflow,
    assess_honest_event_study_workflow_with_config, build_honest_directional_report_rows,
    build_honest_joint_path_report_rows, build_honest_multi_flci_report_rows,
    build_relative_magnitude_flci_problem, build_relative_magnitude_multi_flci_problem,
    compute_honest_ci_scalar, compute_honest_event_study_scalar,
    compute_honest_event_study_scalar_for_input,
    compute_monotone_smoothness_confidence_set_with_simulation, compute_original_confidence_set,
    compute_relative_magnitude_confidence_set,
    compute_relative_magnitude_confidence_set_with_config, compute_relative_magnitude_flci,
    compute_relative_magnitude_flci_with_config, compute_relative_magnitude_identified_set,
    compute_relative_magnitude_multi_flci, compute_relative_magnitude_multi_flci_with_config,
    compute_signed_smoothness_confidence_set_with_simulation,
    compute_smoothness_confidence_set_with_simulation,
    estimate_smoothness_upper_bound_from_pretrends, render_honest_directional_report_csv,
    render_honest_directional_report_json, render_honest_joint_path_report_csv,
    render_honest_joint_path_report_json, render_honest_multi_flci_report_csv,
    render_honest_multi_flci_report_json, summarize_post_period_sensitivity,
    summarize_post_period_sensitivity_with_simulation, summarize_relative_magnitude_sensitivity,
    summarize_relative_magnitude_sensitivity_many,
    summarize_relative_magnitude_sensitivity_many_with_simulation,
    summarize_relative_magnitude_sensitivity_with_simulation, summarize_smoothness_sensitivity,
    summarize_smoothness_sensitivity_with_simulation,
};
pub use inference::{
    ClusterIndex, ClusterSpec, InferenceError, clustered_ci_from_index,
    clustered_standard_error_from_index, clustered_variance_from_index,
};
#[cfg(feature = "honest")]
pub use methods::att_gt::WindowAssessment;
pub use methods::att_gt::{
    AggteBootstrap, AggteConfig, AggteError, AggteEstimate, AggteResult, AggteType, PretestError,
    PretrendWald, UnitPanel, aggregate_att_gt, aggregate_att_gt_by_calendar_time,
    aggregate_att_gt_by_cohort, aggregate_att_gt_event_time_with_influence,
    aggregate_att_gt_overall, att_gt_clustered_standard_errors, att_gt_mboot_bands,
    estimate_att_gt, estimate_att_gt_dr, estimate_att_gt_dr_efficient_with_influence,
    estimate_att_gt_dr_panel, estimate_att_gt_dr_panel_with_influence,
    estimate_att_gt_dr_with_influence, estimate_att_gt_ipw,
    estimate_att_gt_ipw_panel_with_influence, estimate_att_gt_ipw_with_influence,
    estimate_att_gt_or, estimate_att_gt_or_panel_with_influence, estimate_att_gt_or_with_influence,
    pretrend_wald_test, row_panel, unit_panel,
};
#[expect(
    deprecated,
    reason = "re-exports deprecated items until they are removed"
)]
pub use methods::att_gt::{
    EventStudyResult, EventTimeResult, aggregate_att_gt_event_time, att_gt_simultaneous_bands,
    att_gt_simultaneous_bands_with_influence,
};
pub use methods::bacon::{
    BaconComparison, BaconDecomposition, BaconError, BaconKind, BaconWithin, bacon_decomposition,
};
pub use methods::continuous::sieve::estimate_acrt_sieve;
pub use methods::dcdh::{
    DcdhConfig, DcdhEffectsEqual, DcdhError, DcdhEstimate, DcdhObservation, DcdhResult,
    DcdhSwitchers, estimate_dcdh,
};
pub use methods::did_cc::{
    DidCcObservationView, estimate_did_cc_robust, estimate_did_cc_robust_from_views,
    estimate_did_cc_stationary, estimate_did_cc_stationary_from_views, test_did_cc_stationarity,
    test_did_cc_stationarity_from_views,
};
pub use methods::drdid::improved::{
    estimate_drdid_improved_panel, estimate_drdid_improved_repeated_cross_section,
};
pub use methods::drdid::ipw_or::{
    estimate_ipw_did_panel, estimate_ipw_did_panel_flat, estimate_ipw_did_repeated_cross_section,
    estimate_or_did_panel, estimate_or_did_panel_flat, estimate_or_did_repeated_cross_section,
};
pub use methods::drdid::panel::{
    PanelFlatInput, PanelNuisanceFit, PanelPropensityCoefficients, estimate_drdid_panel,
    estimate_drdid_panel_flat, fit_drdid_panel_nuisance, fit_drdid_panel_propensity_coefficients,
};
pub use methods::drdid::repeated::estimate_drdid_repeated_cross_section;
pub use methods::drdid::repeated_efficient::estimate_drdid_repeated_efficient;
pub use methods::drdid::{DesignPruning, independent_design_columns};
pub use methods::efficient::{
    aggregate_efficient_att_gt_by_calendar_time, aggregate_efficient_att_gt_by_cohort,
    aggregate_efficient_att_gt_event_time, aggregate_efficient_att_gt_event_time_with_influence,
    aggregate_efficient_att_gt_overall, estimate_att_gt_efficient, estimate_att_gt_efficient_dr,
};
pub use methods::imputation;
pub use methods::imputation::{
    ImputationConfig, ImputationError, ImputationEstimate, ImputationResult, ImputationTarget,
    ImputationTerm, PretrendResult, PretrendTerm, Pretrends, estimate_did_imputation,
};
pub use methods::staggered::StaggeredPanel;
#[expect(
    deprecated,
    reason = "re-exports a deprecated function until it is removed"
)]
pub use methods::standard::aggregate_event_time;
pub use methods::standard::{
    estimate_att_from_summary, estimate_att_two_by_two, summarize_two_by_two,
};
pub use methods::sunab::{
    SUNAB_DEFAULT_REFERENCES, SunabBins, SunabCell, SunabConfig, SunabControl, SunabError,
    SunabEstimate, SunabReference, SunabResult, SunabTerm, estimate_sunab,
};
pub use methods::synthdid;
pub use methods::synthdid::{
    StaggeredSynthEstimate, StaggeredSynthPanel, SynthCohort, SynthDidConfig, SynthDidError,
    SynthDidEstimate, SynthDidVariance, SynthDraws, SynthObservation, SynthPanel, SynthTrace,
    SynthUndefined, SynthVariance, SynthWeights, estimate_synthdid, estimate_synthdid_staggered,
    synthdid_staggered_variance, synthdid_variance,
};
pub use methods::triple_did::aggregator::estimate_dr_ddd;
pub use methods::twfe::{TwfeConfig, TwfeError, TwfeResult, TwfeSpec, TwfeTerm, estimate_twfe};
pub use methods::two_stage;
pub use methods::two_stage::{
    TwoStageConfig, TwoStageResult, TwoStageSpec, TwoStageTerm, estimate_did2s,
};
pub use types::*;
