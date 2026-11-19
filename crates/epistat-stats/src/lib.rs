//! Sample quantiles, the normal distribution, floating-point sums,
//! log-likelihood helpers and covariate balance statistics, checked against R
//! and Python and computed the same way on every platform.

pub mod balance;
pub mod likelihood;
pub mod normal;
pub mod quantile;
pub mod sum;
