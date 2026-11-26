//! Distribution functions, sample quantiles, floating-point sums, weighted
//! means and variances, log-likelihood helpers, MCMC diagnostics and k-means
//! clustering, checked against R and Python and computed the same way on
//! every platform.
//!
//! | module | what it holds |
//! |---|---|
//! | [`dist`] | the normal, t, chi-square, F, beta, gamma, Poisson and binomial distributions |
//! | [`quantile`] | R's sample quantiles of types 1 and 7, and a weighted type 1 |
//! | [`sum`] | sums in the order Python and polars compute them, and over eight fixed lanes |
//! | [`weighted`] | weighted means and variances with reliability weights, effective sample size, weight limits |
//! | [`moments`] | the mean, variance and standard deviation over fixed lanes |
//! | [`logistic`] | the logistic function, the logit and their logs |
//! | [`wald`] | an estimate with its Wald interval and p-value |
//! | [`likelihood`] | log-sum-exp, softmax and information criteria |
//! | [`mcmc`] | autocorrelation, effective sample size and split R-hat |
//! | [`cluster`] | k-means from given centres |
//! | [`compat`] | the approximations that Studies I and II reproduce |
//!
//! The distribution functions return NaN for a parameter outside the domain,
//! as R's do; every other function reports an undefined result as `None` and
//! invalid input as an [`Error`].
//!
//! The modules `normal`, `gamma`, `student` and `poisson` hold the paths of
//! 0.2 as deprecated forwarders for one release.
#![warn(missing_docs)]

pub mod balance;
pub mod bias;
pub mod cluster;
pub mod compat;
pub mod decomposition;
pub mod dist;
mod error;
pub mod evalue;
pub mod gamma;
pub mod likelihood;
pub mod logistic;
pub mod mantel_haenszel;
pub mod mcmc;
pub mod moments;
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
mod wald;
pub mod weighted;

pub use error::Error;
pub use wald::wald;
