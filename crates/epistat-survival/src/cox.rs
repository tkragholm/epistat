//! Cox regression: the partial likelihood fit, its tests, residuals and
//! predictions, as `survival::coxph` and its methods.
//!
//! [`fit`] fits a design matrix on a [`SurvData`](crate::SurvData) with
//! Breslow, Efron or exact ties, delayed entry, strata, case weights, offsets
//! and a cluster-robust variance, following `coxph` in its iteration; it is
//! documented in the [`design`] module. [`residuals`](mod@residuals) and
//! [`zph`](mod@zph) give `residuals.coxph` and `cox.zph`, [`predict`](mod@predict)
//! gives `basehaz`, `predict.coxph` and `survfit.coxph`, and [`tt`] fits
//! time-transform covariates as `coxph(tt =)`.
//!
//! [`fit_cox`] and [`CoxDesign`] are deprecated wrappers of [`fit`] with exact
//! ties. [`fit_cox_single_covariate`] and [`CoxData`], the fit on one covariate
//! that came before the design matrix, are deprecated too; its algorithm is
//! described in the private `single` module.

// Rows, levels and strata are counted in `u32`; the entry point refuses more
// rows than that.
#![expect(
    clippy::cast_possible_truncation,
    reason = "the entry point checks that the row count fits in u32"
)]

pub mod anova;
pub mod design;
pub mod predict;
pub mod residuals;
mod single;
pub mod tt;
pub mod zph;
pub use anova::{AnovaRow, AnovaTable, anova, anova_terms};
pub use design::{ChiSquareTest, CoxModelFit, CoxOptions, CoxTests, fit, score_residuals};
#[expect(
    deprecated,
    reason = "the deprecated names stay exported for a release"
)]
pub use design::{CoxDesign, cox_score_residuals, fit_cox};
pub use predict::predict_expected_martingale;
pub use predict::{
    Baseline, BaselineOptions, Centring, ConfType, Curve, CurveTimes, FittedModel, HazardType,
    NewData, PredictOptions, PredictType, Prediction, Reference, StratumBaseline, SurvCurves,
    SurvfitOptions, TermsPrediction, Variance, basehaz, predict, predict_terms, reference_means,
    reference_offset, survfit,
};
pub use residuals::{Collapse, CoxResiduals, ResidualKind, ResidualOptions, residuals};
#[expect(
    deprecated,
    reason = "the deprecated single-covariate fit stays public until it is removed"
)]
pub use single::{CoxData, CoxFit, fit_cox_single_covariate};
pub use tt::{CoxTtFit, RiskSetTransform, TimeTransform, TtColumn, TtOptions, fit_tt};
pub use zph::{CoxZph, ZphOptions, ZphTransform, zph};

/// How events that share a time are placed in the partial likelihood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ties {
    /// Every event at a time has the whole risk set as its denominator.
    Breslow,
    /// Efron's approximation, with the weights of tied events averaged as in
    /// `survival::coxph`: the `k`-th of `d` tied events has the risk set less
    /// `k / d` of the weight of the tied events.
    Efron,
    /// The exact partial likelihood, the discrete-time conditional logistic
    /// likelihood: the `d` tied events at a time are one of the equally
    /// likely sets of `d` rows from the risk set, each weighted by the product
    /// of its risk scores. It is available in [`fit`] only, for case weights
    /// of 1, without clusters, and without score residuals, as in `coxph`.
    /// The cost at a time with `d` tied events among `n` at risk is
    /// proportional to `d * n` times the square of the number of covariates.
    Exact,
}
