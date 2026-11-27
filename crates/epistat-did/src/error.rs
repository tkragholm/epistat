//! The errors of the crate.
//!
//! [`Error`] is the error of the data views and of the estimators that take
//! them. A [`StaggeredPanel`] that does not build returns [`Error::Input`]
//! with the shared [`InputError`], as every epistat crate reports invalid
//! input. The other variants say why accepted data do not support an
//! estimate, which option cannot be used with them, or which lower step
//! failed, with that crate's error as the source. [`Classify`] gives a
//! binding the kind and a stable code.
//!
//! [`StaggeredPanel`]: crate::StaggeredPanel

use std::fmt;

use epistat_core::{Classify, ErrorKind, InputError};

use crate::methods::att_gt::SkipReason;

/// Why the input of an estimator could not be used.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// A column of the wrong length, a value that is not finite, a negative
    /// weight, rows out of order, or a unit-level column that changes within
    /// a unit.
    #[error(transparent)]
    Input(#[from] InputError),
    /// No row carries a treatment indicator, so the regression has no
    /// treatment term.
    #[error("no row carries a treatment indicator")]
    NoTreatment,
    /// Never-treated comparators were asked for and no unit is never
    /// treated, before or after `did`'s pre-processing.
    #[error("no unit is never treated in the panel")]
    NoNeverTreated,
    /// The periods span more than 2^32 values, or a period relative to
    /// treatment does not fit in 64 bits.
    #[error("the periods span more than 2^32 values")]
    Periods,
    /// The bins of an event study are empty, leave out the reference
    /// period, or have a width that is not positive.
    #[error("the bins need lo <= hi, the reference inside them, or a positive width")]
    Bins,
    /// A requested horizon is negative.
    #[error("horizon {0} is negative")]
    Horizon(i64),
    /// A requested pre-trend period is not negative.
    #[error("pre-trend period {0} is not negative")]
    Pretrend(i64),
    /// A requested reference period is not negative.
    #[error("reference period {0} is not negative")]
    Reference(i64),
    /// A column of target weights is not zero on an untreated row.
    #[error("target weight column {column} is not zero on the untreated row {row}")]
    TargetOnUntreated {
        /// The column.
        column: usize,
        /// The row.
        row: usize,
    },
    /// Some unit lacks a period where the method needs a balanced panel.
    #[error("the panel is not balanced")]
    Unbalanced,
    /// Fewer than two timing groups.
    #[error("the decomposition needs at least two timing groups")]
    Groups,
    /// The estimator does not do what the options or the data ask for, such
    /// as prior weights in the Goodman-Bacon decomposition.
    #[error("not available: {what}")]
    Unsupported {
        /// What was asked for.
        what: &'static str,
    },
    /// No ATT(g,t) cell could be estimated.
    #[error("no ATT(g,t) cell could be estimated")]
    NoEstimableCells,
    /// The never-treated comparison group has fewer rows per period than the
    /// number of covariates plus 5, which `did::att_gt` refuses.
    #[error(
        "the never-treated group is too small to serve as a reliable comparison \
         ({per_period} rows per period, fewer than {required}, the number of covariates \
         plus 5); the not-yet-treated comparison group adds the units not yet treated"
    )]
    SmallNeverTreatedGroup {
        /// Never-treated rows divided by the number of periods.
        per_period: f64,
        /// The number of covariates plus 5.
        required: usize,
    },
    /// A cell could not be fitted and failed cells are not skipped.
    #[error("the ATT(g,t) cell (group {group}, time {time}, base {base}) failed: {reason}")]
    CellFailed {
        /// The cell's cohort.
        group: i64,
        /// The cell's later period.
        time: i64,
        /// The cell's base period.
        base: i64,
        /// Why the cell failed.
        reason: SkipReason,
    },
    /// The efficient estimator found a cohort without pre-treatment periods.
    #[error("cohort {group} has no pre-treatment period")]
    NoPrePeriods {
        /// The cohort.
        group: i64,
    },
    /// The efficient estimator found a cohort without post-treatment periods.
    #[error("cohort {group} has no post-treatment period")]
    NoPostPeriods {
        /// The cohort.
        group: i64,
    },
    /// An aggregation, band, pre-test or event study was asked of an ATT(g,t)
    /// whose influence function was not kept.
    #[error("the ATT(g,t) influence function was not kept, so it cannot be aggregated")]
    NoInfluence,
    /// No post-treatment cell, or for a fixed-weight event study no cell,
    /// survives the restrictions of a summary.
    #[error("no post-treatment cell survives the requested aggregation")]
    NoPostTreatmentCells,
    /// No pre-treatment cell has a standard error to test.
    #[error("no pre-treatment cell with a standard error to test")]
    NoPreTreatmentCells,
    /// A system to solve was singular.
    #[error("a system to solve is singular")]
    Singular,
    /// A result was not finite.
    #[error("a result is not finite")]
    NotFinite,
    /// A regression, an absorption of fixed effects or a covariance of the
    /// fit failed.
    #[error(transparent)]
    Glm(#[from] epistat_glm::Error),
    /// An influence function could not be formed.
    #[error(transparent)]
    Inference(#[from] epistat_inference::Error),
    /// After the sample rules of dCDH no period has both switchers and
    /// controls (Design Restriction 1 of de Chaisemartin and
    /// D'Haultfoeuille fails), so no effect can be estimated.
    #[error("no treatment effect can be estimated")]
    NoEffect,
    /// After the sample rules of dCDH a period inside the panel has no rows.
    #[error("a period inside the panel has no rows after the sample rules")]
    EmptyPeriod,
    /// The controls of dCDH cannot be used for a baseline treatment.
    #[error("controls at baseline treatment {baseline}: {reason}")]
    Controls {
        /// The baseline treatment.
        baseline: f64,
        /// What is wrong.
        reason: &'static str,
    },
    /// The regressors of a regression are collinear, as the covariates of
    /// dCDH's heterogeneity regression with its cells.
    #[error("collinear: {what}")]
    Collinear {
        /// The regression.
        what: &'static str,
    },
    /// The default noise level of synthdid needs at least two first
    /// differences of the controls before adoption; give the noise level
    /// instead.
    #[error("the noise level needs two first differences before adoption; give it explicitly")]
    NoiseLevel,
    /// The placebo standard error of synthdid needs more controls than
    /// treated units.
    #[error("the placebo method needs more controls than treated units")]
    PlaceboControls,
    /// A given draw of synthdid's placebo or bootstrap is not a permutation
    /// of the controls (placebo), or holds an index outside the panel, no
    /// control or no treated unit (bootstrap).
    #[error("draw {replicate} is not a valid draw for this method")]
    Draw {
        /// The replicate, from 0.
        replicate: usize,
    },
    /// A unit is first treated in the first period of the panel, or in a
    /// period that is not one of the panel's, so synthdid has no pre-period
    /// or no column for its adoption.
    #[error("unit {unit} is not first treated in a period of the panel after the first")]
    Adoption {
        /// The unit, as its code.
        unit: u32,
    },
    /// A solver refused its problem, as the Frank-Wolfe steps of synthdid.
    #[error(transparent)]
    Optim(#[from] epistat_optim::Error),
    /// A two-period sample has no treated row, or none the trim leaves with
    /// weight.
    #[error("no treated row of positive weight")]
    NoTreated,
    /// A two-period sample has no comparison row, or none the trim leaves
    /// with weight.
    #[error("no comparison row of positive weight")]
    NoComparison,
    /// A repeated cross-section has no row in one arm and period.
    #[error("no row with treated = {treated} and post = {post}")]
    EmptyCell {
        /// The arm of the empty cell.
        treated: bool,
        /// The period of the empty cell.
        post: bool,
    },
    /// An arm and period of a repeated cross-section has fewer rows than the
    /// cross-fitting folds.
    #[error("{folds} folds need {folds} rows with treated = {treated} and post = {post}")]
    CrossFitCell {
        /// The number of folds.
        folds: usize,
        /// The arm of the cell.
        treated: bool,
        /// The period of the cell.
        post: bool,
    },
    /// The variance of a difference of two estimates is not positive, so it
    /// cannot be tested.
    #[error("the variance of the difference is not positive")]
    DegenerateVariance,
    /// A propensity logit could not be fitted on the covariates, and the
    /// estimator was not allowed to fall back, as
    /// [`crate::did_cc::DidCcOptions::propensity_fallback`] allows.
    #[error("the propensity logit could not be fitted on the covariates{}", fold_suffix(*fold))]
    PropensityFit {
        /// The cross-fitting fold whose model failed, `None` for the model
        /// fitted on every row.
        fold: Option<usize>,
    },
}

fn fold_suffix(fold: Option<usize>) -> String {
    fold.map_or_else(String::new, |fold| format!(" in cross-fitting fold {fold}"))
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Glm(error) => error.kind(),
            Self::Inference(error) => error.kind(),
            Self::Optim(error) => error.kind(),
            Self::NoEffect
            | Self::EmptyPeriod
            | Self::Controls { .. }
            | Self::NoiseLevel
            | Self::PlaceboControls
            | Self::Draw { .. }
            | Self::Adoption { .. }
            | Self::NoTreatment
            | Self::NoNeverTreated
            | Self::Periods
            | Self::Bins
            | Self::Horizon(_)
            | Self::Pretrend(_)
            | Self::Reference(_)
            | Self::TargetOnUntreated { .. }
            | Self::Unbalanced
            | Self::Groups
            | Self::NoEstimableCells
            | Self::SmallNeverTreatedGroup { .. }
            | Self::NoPrePeriods { .. }
            | Self::NoPostPeriods { .. }
            | Self::NoPostTreatmentCells
            | Self::NoPreTreatmentCells
            | Self::NoTreated
            | Self::NoComparison
            | Self::EmptyCell { .. }
            | Self::CrossFitCell { .. } => ErrorKind::Input,
            Self::Unsupported { .. } | Self::NoInfluence => ErrorKind::Unsupported,
            Self::CellFailed { .. }
            | Self::Singular
            | Self::NotFinite
            | Self::Collinear { .. }
            | Self::DegenerateVariance
            | Self::PropensityFit { .. } => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Glm(error) => error.code(),
            Self::Inference(error) => error.code(),
            Self::Optim(error) => error.code(),
            Self::NoEffect => "did.no_effect",
            Self::EmptyPeriod => "did.empty_period",
            Self::Controls { .. } => "did.controls",
            Self::Collinear { .. } => "did.collinear",
            Self::NoiseLevel => "did.noise_level",
            Self::PlaceboControls => "did.placebo_controls",
            Self::Draw { .. } => "did.draw",
            Self::Adoption { .. } => "did.adoption",
            Self::NoTreatment => "did.no_treatment",
            Self::NoNeverTreated => "did.no_never_treated",
            Self::Periods => "did.periods",
            Self::Bins => "did.bins",
            Self::Horizon(_) => "did.horizon",
            Self::Pretrend(_) => "did.pretrend",
            Self::Reference(_) => "did.reference",
            Self::TargetOnUntreated { .. } => "did.target_on_untreated",
            Self::Unbalanced => "did.unbalanced",
            Self::Groups => "did.groups",
            Self::Unsupported { .. } => "did.unsupported",
            Self::NoEstimableCells => "did.no_estimable_cells",
            Self::SmallNeverTreatedGroup { .. } => "did.small_never_treated",
            Self::CellFailed { .. } => "did.cell_failed",
            Self::NoPrePeriods { .. } => "did.no_pre_periods",
            Self::NoPostPeriods { .. } => "did.no_post_periods",
            Self::NoInfluence => "did.no_influence",
            Self::NoPostTreatmentCells => "did.no_post_treatment_cells",
            Self::NoPreTreatmentCells => "did.no_pre_treatment_cells",
            Self::Singular => "did.singular",
            Self::NotFinite => "did.not_finite",
            Self::NoTreated => "did.no_treated",
            Self::NoComparison => "did.no_comparison",
            Self::EmptyCell { .. } => "did.empty_cell",
            Self::CrossFitCell { .. } => "did.cross_fit_cell",
            Self::DegenerateVariance => "did.degenerate_variance",
            Self::PropensityFit { .. } => "did.propensity_fit",
        }
    }
}
/// Errors from internal estimation steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InternalDidError {
    /// An estimation step failed, with a description.
    Estimation(String),
    /// An iterative procedure did not converge, with a description.
    Convergence(String),
}

impl fmt::Display for InternalDidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Estimation(message) => write!(f, "Estimation failed: {message}"),
            Self::Convergence(message) => write!(f, "Convergence error: {message}"),
        }
    }
}

impl std::error::Error for InternalDidError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique_and_kinds_follow_the_cause() {
        let errors = [
            Error::NoTreatment,
            Error::NoNeverTreated,
            Error::Periods,
            Error::Bins,
            Error::Horizon(-1),
            Error::Pretrend(0),
            Error::Reference(1),
            Error::TargetOnUntreated { column: 0, row: 3 },
            Error::Unbalanced,
            Error::Groups,
            Error::Unsupported { what: "weights" },
            Error::NoEstimableCells,
            Error::SmallNeverTreatedGroup {
                per_period: 1.0,
                required: 6,
            },
            Error::CellFailed {
                group: 2,
                time: 3,
                base: 1,
                reason: SkipReason::NoTreated,
            },
            Error::NoPrePeriods { group: 2 },
            Error::NoPostPeriods { group: 2 },
            Error::NoInfluence,
            Error::NoPostTreatmentCells,
            Error::NoPreTreatmentCells,
            Error::Singular,
            Error::NotFinite,
            Error::NoEffect,
            Error::EmptyPeriod,
            Error::Controls {
                baseline: 0.0,
                reason: "collinear",
            },
            Error::Collinear { what: "covariates" },
            Error::NoiseLevel,
            Error::PlaceboControls,
            Error::Draw { replicate: 1 },
            Error::Adoption { unit: 2 },
            Error::NoTreated,
            Error::NoComparison,
            Error::EmptyCell {
                treated: true,
                post: false,
            },
            Error::CrossFitCell {
                folds: 2,
                treated: false,
                post: true,
            },
            Error::DegenerateVariance,
            Error::PropensityFit { fold: Some(1) },
        ];
        epistat_testkit::assert_codes_unique("did", &errors);
        assert_eq!(Error::Bins.kind(), ErrorKind::Input);
        assert_eq!(
            Error::Unsupported { what: "weights" }.kind(),
            ErrorKind::Unsupported
        );
        let glm = Error::from(epistat_glm::Error::Singular);
        assert_eq!(
            (glm.kind(), glm.code()),
            (ErrorKind::Numerical, "glm.singular")
        );
    }
}
