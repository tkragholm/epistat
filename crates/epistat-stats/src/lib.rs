//! Sample quantiles, the normal and chi-square distributions, the gamma
//! function, floating-point sums, log-likelihood helpers, covariate balance
//! statistics, MCMC diagnostics, weighted means and variances, direct
//! standardisation and the Kitagawa decomposition, Poisson confidence intervals,
//! the gamma quantile, the Student t distribution and the incomplete beta
//! function with its quantile, k-means clustering, eight-lane sums and
//! quantitative bias analysis, checked against R and Python and computed
//! the same way on every platform.
#![warn(missing_docs)]

pub mod balance;
pub mod bias;
pub mod cluster;
pub mod decomposition;
mod discrete;
pub mod evalue;
pub mod gamma;
pub mod lanes;
pub mod likelihood;
pub mod mantel_haenszel;
pub mod mcmc;
pub mod normal;
pub mod poisson;
pub mod quantile;
pub mod rates;
pub mod rosenbaum;
pub mod signed_rank;
pub mod sir;
pub mod standardisation;
pub mod student;
pub mod sum;
pub mod weighted;
