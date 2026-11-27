mod diagnostics;
mod input;
mod report;
mod stats;

pub use diagnostics::balance_diagnostics;
pub use input::BalanceWeights;
pub use report::{balance_improvement, balance_report, balance_summary, balance_threshold_summary};

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
