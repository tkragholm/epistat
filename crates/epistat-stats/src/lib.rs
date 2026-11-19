//! Sample quantiles, the normal distribution, floating-point sums,
//! log-likelihood helpers, covariate balance statistics and MCMC diagnostics,
//! checked against R and Python and computed the same way on every platform.

pub mod balance;
pub mod likelihood;
pub mod mcmc;
pub mod normal;
pub mod quantile;
pub mod sum;
