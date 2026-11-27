//! How much can parallel trends fail before an event study's conclusion
//! changes? The sensitivity analysis of Rambachan and Roth \[1\] in
//! [`honest`], and the power of the pre-trends test with the bias after
//! passing it, after Roth \[2\], in [`pretrends`].
//!
//! Both take an [`EventStudy`]: the coefficients by period relative to
//! treatment, their covariance and the reference period, from any estimator.
//! epistat-did's `AttGt::event_study` and its fixed-effects estimators
//! return one.
//!
//! ```
//! use epistat_core::{Covariance, EventStudy, Level};
//! use epistat_honestdid::honest::{self, HonestOptions, Restriction, RmBound, Target};
//!
//! // Three pre-treatment and two post-treatment coefficients, reference -1.
//! let variances = [0.04, 0.04, 0.04, 0.05, 0.06];
//! let covariance: Vec<f64> = (0..25)
//!     .map(|i| if i / 5 == i % 5 { variances[i / 5] } else { 0.005 })
//!     .collect();
//! let study = EventStudy::new(
//!     vec![0.02, -0.03, 0.01, 0.4, 0.5],
//!     Covariance::from_values(5, &covariance)?,
//!     vec![-4, -3, -2, 0, 1],
//!     -1,
//! )?;
//! let target = Target::Average { first: 0, last: 1 };
//! let original = honest::original(&study, &target, Level::default())?;
//! let summary = honest::summarize(
//!     &study,
//!     &target,
//!     &Restriction::RelativeMagnitude { mbar: vec![0.0, 0.5, 1.0], bound: RmBound::ParallelTrends },
//!     &HonestOptions::default(),
//! )?;
//! assert!(original.lower().is_some());
//! assert_eq!(summary.rows.len(), 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # References
//!
//! 1. Rambachan A, Roth J. A more credible approach to parallel trends. Rev
//!    Econ Stud 2023;90(5):2555-91. doi:10.1093/restud/rdad018
//! 2. Roth J. Pretest with caution: event-study estimates after testing for
//!    parallel trends. Am Econ Rev Insights 2022;4(3):305-22.
//!    doi:10.1257/aeri.20210236
//!
//! [`EventStudy`]: epistat_core::EventStudy
#![warn(missing_docs)]

mod error;
pub mod honest;
pub mod pretrends;
mod sensitivity;
#[cfg(test)]
mod tests;
mod util;

pub use error::Error;
