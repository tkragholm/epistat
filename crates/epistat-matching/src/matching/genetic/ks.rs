//! The two-sample Kolmogorov-Smirnov test of [`epistat_epi::balance`], in the
//! shape the genetic search reads it.

pub(crate) use epistat_epi::balance::{ks_statistic, ks_statistic_sorted as ks_d_sorted};

/// The statistic, the p-value and whether the p-value is exact.
pub(crate) fn ks_test(x: &[f64], y: &[f64]) -> Option<(f64, f64, bool)> {
    epistat_epi::balance::ks_test(x, y).map(|t| (t.statistic, t.p_value, t.exact))
}
