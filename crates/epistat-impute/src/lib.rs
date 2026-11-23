//! Multiple imputation by chained equations, as `mice` does it, with results
//! that do not depend on the number of threads.
//!
//! [`impute`] fills the missing cells of an [`Incomplete`] data set `m` times.
//! Each incomplete column is imputed from a regression on the others, in turn
//! and repeatedly, so that mixed variable types need no joint model. The
//! imputed data sets are analysed one by one and pooled by Rubin's rules:
//! [`Imputed::pool`] and [`Imputed::pool_vector`] hand the completed data
//! to an analysis function and pool its estimates with
//! `epistat_inference::pooling`.
//!
//! # Methods
//!
//! | [`Method`] | Column | `mice` |
//! |---|---|---|
//! | `Norm` | continuous | `norm` |
//! | `Pmm` | continuous | `pmm`, with `donors` and `matchtype` |
//! | `Logreg` | binary | `logreg` |
//! | `Polyreg` | nominal | `polyreg`, drawing the coefficients |
//! | `Sample` | any | `sample` |
//! | `Passive` | continuous | a passive `~ I(...)` formula |
//!
//! Each method draws the model's parameters from their sampling distribution
//! before the missing values, as Rubin's rules require. `mice`'s `polyreg`
//! does not draw them, so it ignores the uncertainty in the coefficients and
//! understates the between-imputation variance. `Polyreg { proper: true }`
//! draws them. `proper: false` is `mice`'s version, for comparison.
//!
//! Predictive mean matching is the usual choice for a skewed continuous
//! variable such as income, because it only imputes values that occur in the
//! data. Not implemented: ordered logistic imputation (`polr`), post-processing,
//! multivariate blocks, imputing a chosen set of cells other than the missing
//! ones, and substantive-model-compatible imputation.
//!
//! # Algorithm
//!
//! Chain `c = 1..m`, iteration `t = 1..maxit`, the incomplete columns `j` in
//! the visit order. With `O_j` the rows where column `j` is observed, `M_j` the
//! rows where it is missing, and `X_j` the predictors as currently completed
//! (nominal predictors as indicators of the levels after the first):
//!
//! ```text
//! Norm:    (b, V, RSS) from least squares on O_j, sigma* = sqrt(RSS / chi2_(n-p)),
//!          beta* = b + chol(V) z sigma*,   y_mis = X_M beta* + sigma* z'
//! Pmm:     as Norm, then a donor from O_j with predicted value near X_M beta*
//! Logreg:  augmented IRLS, beta* = b + chol(V) z,  y_mis = 1 with probability
//!          expit(X_M beta*)
//! Polyreg: augmented multinomial logit, beta* = b + chol(V) z, the class by
//!          inverse distribution function
//! ```
//!
//! The formulas and the matching rule are in [`parts`]. Before the first
//! iteration each missing cell holds a random draw of an observed value of its
//! column. A column whose predictors include one that is a linear combination
//! of the others is fitted without it, and the number of such fits is
//! [`Imputed::n_aliased_dropped`]. The logistic and multinomial models add
//! the rows of `mice:::augment`, which keep the fit finite when the predictors
//! separate the outcome, and the number of fits that did so is
//! [`Imputed::n_augmented`]. Where `mice` goes on with a warning and a ridge, a
//! column with too few observed rows, a regression that does not converge and a
//! covariance without a Cholesky factor are errors.
//!
//! # Streams
//!
//! Every draw comes from a [`SplitMix64`](epistat_rng::Stream) stream seeded
//! from the seed, the chain, the iteration, the column and the row: the seed
//! is mixed by `splitmix64` and then by `replicate_seed` with the chain, the
//! iteration, the column and the row in turn. The parameters of a column's model
//! use the row index `u64::MAX`. A draw therefore does not depend on the thread
//! that makes it or on how the rows are divided, and the chains, the rows of a
//! column and the regressions of `epistat-glm` all give the same values on any
//! number of threads.
//!
//! # Survival outcomes
//!
//! To impute a covariate of a Cox model the imputation model must contain the
//! outcome. White and Royston \[5\] show that this means the event indicator and
//! the Nelson-Aalen cumulative hazard at each person's own time, not `log T`.
//! [`nelson_aalen`] gives the second as a complete column. Put both in the data
//! and in the predictors of every incomplete covariate. See [`survival`]. With
//! competing risks the cumulative hazards of every cause and the indicators of
//! all causes are needed, which this crate does not compute.
//!
//! # Congeniality
//!
//! The imputation model has to contain every variable of the analysis model,
//! and every interaction and non-linear term of it \[8\]. A covariate with an
//! interaction in the analysis needs it in the imputation model, either by
//! imputing within groups or as a passive column, and a passive column does not
//! restore consistency with the joint distribution \[9\]. This crate does not
//! remove that bias, and substantive-model-compatible imputation \[7\], which
//! does, is not part of it.
//!
//! # Memory
//!
//! Each chain holds the completed data as `f64` columns, `8 n` bytes per
//! column, and chains run in parallel, so memory is about `m_parallel` copies
//! of the data. The result holds only the imputed cells.
//!
//! # References
//!
//! 1. Rubin DB. Multiple Imputation for Nonresponse in Surveys. New York:
//!    Wiley; 1987.
//! 2. van Buuren S, Groothuis-Oudshoorn K. mice: multivariate imputation by
//!    chained equations in R. J Stat Softw. 2011;45(3):1-67.
//! 3. van Buuren S. Flexible Imputation of Missing Data. 2nd ed. Boca Raton:
//!    Chapman and Hall/CRC; 2018.
//! 4. White IR, Royston P, Wood AM. Multiple imputation using chained
//!    equations: issues and guidance for practice. Stat Med. 2011;30(4):377-99.
//! 5. White IR, Royston P. Imputing missing covariate values for the Cox
//!    model. Stat Med. 2009;28(15):1982-98.
//! 6. Morris TP, White IR, Royston P. Tuning multiple imputation by predictive
//!    mean matching and local residual draws. BMC Med Res Methodol.
//!    2014;14:75.
//! 7. Bartlett JW, Seaman SR, White IR, Carpenter JR. Multiple imputation of
//!    covariates by fully conditional specification: accommodating the
//!    substantive model. Stat Methods Med Res. 2015;24(4):462-87.
//! 8. Meng XL. Multiple-imputation inferences with uncongenial sources of
//!    input. Stat Sci. 1994;9(4):538-58.
//! 9. von Hippel PT. How to impute interactions, squares, and other
//!    transformed variables. Sociol Methodol. 2009;39(1):265-91.
#![warn(missing_docs)]

mod data;
mod error;
mod impute;
pub mod parts;
mod pool;
mod spec;
pub mod survival;

pub use data::{Column, Incomplete, Mask};
pub use error::{ImputeError, Result};
pub use impute::{Imputed, impute};
pub use spec::{ImputeOptions, ImputeSpec, MatchType, Method, PassiveFn};
pub use survival::{Ties, nelson_aalen};
