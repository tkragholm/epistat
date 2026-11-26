mod diagnostics;
mod input;
mod projected;
mod report;
mod stats;

pub use diagnostics::balance_diagnostics;
pub use input::{BalanceInputError, BalanceWeights};
pub use projected::{
    CategoricalCovariateSpec, NumericCovariateSpec, ProjectedBalanceRow,
    build_projected_balance_rows, categorical_max_abs_diff, smd_numeric_from_values,
};
pub use report::{
    balance_improvement, balance_report, balance_report_on_weights, balance_report_with_options,
    balance_summary, balance_threshold_summary,
};

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_ecdf_distance_stats(
    case_values: &[f64],
    control_values: &[f64],
) -> Option<(f64, f64)> {
    epistat_epi::balance::ecdf_distance(case_values, control_values)
}

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_eqq_distance_stats(
    case_values: &[f64],
    control_values: &[f64],
) -> Option<(f64, f64)> {
    epistat_epi::balance::eqq_distance(case_values, control_values)
}
