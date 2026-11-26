//! Quantitative bias analysis: corrections of a 2 x 2 table for exposure or
//! outcome misclassification, selection and an unmeasured binary confounder,
//! and their probabilistic versions.
//!
//! The table is a [`Counts2x2`], written `a, b, c, d` here with `a` exposed cases, `b`
//! unexposed cases, `c` exposed non-cases and `d` unexposed non-cases. The
//! risk ratio is `(a / (a + c)) / (b / (b + d))`, the odds ratio `a d / (b c)`
//! and the risk difference `a / (a + c) - b / (b + d)`. Bias parameters are
//! assumed values, and nothing here estimates them.
//!
//! # Algorithm
//!
//! Exposure misclassification, with sensitivity `Se_k` and specificity `Sp_k`
//! of the exposure classification among cases (`k = 0`) and non-cases
//! (`k = 1`). The true numbers of exposed cases and non-cases are
//!
//! ```text
//! A = ( a - (1 - Sp_0) (a + b) ) / ( Se_0 - (1 - Sp_0) ),      B = (a + b) - A
//! C = ( c - (1 - Sp_1) (c + d) ) / ( Se_1 - (1 - Sp_1) ),      D = (c + d) - C
//! ```
//!
//! and with predictive values `PPV_k` and `NPV_k` in place of `Se` and `Sp`,
//! `A = a PPV_0 + b (1 - NPV_0)` and `C = c PPV_1 + d (1 - NPV_1)`. With the
//! positive predictive value and the sensitivity, the truly exposed are the
//! recorded exposed times `PPV_k / Se_k`. Outcome
//! misclassification is the same with the roles of exposure and outcome
//! exchanged, `Se_k` and `Sp_k` for exposed (`k = 0`) and unexposed (`k = 1`)
//! persons and `A = ( a - (1 - Sp_0) (a + c) ) / ( Se_0 - (1 - Sp_0) )` for the
//! true cases among the exposed, `A = a PPV_0 + c (1 - NPV_0)` from
//! predictive values and `A = a PPV_0 / Se_0` from the positive predictive
//! value and the sensitivity. `Se + Sp - 1` must be positive, or the
//! classification is no better than chance.
//!
//! Selection with the probability `s` of being in the table by cell: the
//! corrected cells are `a / s_a`, `b / s_b`, `c / s_c` and `d / s_d`, and the
//! odds ratio is multiplied by `(s_b s_c) / (s_a s_d)`.
//!
//! A binary confounder with prevalence `p_1` among the exposed and `p_0` among
//! the unexposed and association `R` with the outcome (risk ratio, odds ratio
//! or risk difference, in the unexposed) divides the crude risk ratio by the
//! bias factor `(p_1 (R - 1) + 1) / (p_0 (R - 1) + 1)`. [`adjust_confounding`]
//! builds the two confounder strata of the table and gives the standardised
//! morbidity ratio and Mantel-Haenszel estimates, as `episensr::confounders`
//! does.
//!
//! The probabilistic versions draw the bias parameters from [`Dist`]
//! distributions, correct, and take the median and the `alpha / 2` and
//! `1 - alpha / 2` percentiles (type 7) of the corrected measures: the
//! systematic error. With `random_error`, the sampling error of the table is
//! added as well. For misclassification this follows Fox, Lash and Greenland
//! \[2\]: the true exposure prevalences among cases and non-cases are drawn
//! from `Beta(A, B)` and `Beta(C, D)`, the predictive values follow from them,
//! the observed cells are reclassified by binomial draws, and the corrected
//! log measure is shifted by a standard normal draw times its standard error.
//! For selection and confounding the shift is a standard normal draw times the
//! standard error of the observed log measure.
//!
//! A draw is discarded when a corrected cell is not positive or a parameter is
//! outside its range, which for a classification includes `Se + Sp <= 1`.
//! The number kept and the number discarded are returned, and so is every
//! draw ([`Draws`]): its bias parameters and its corrected measures, so that a
//! draw can be checked against the simple correction, plotted against a
//! parameter, or summarised otherwise. The risk difference is summarised as
//! the ratios are; with random error it is shifted by a standard normal draw
//! times its standard error `sqrt(a c / (a + c)^3 + b d / (b + d)^3)`.
//!
//! Rates (cases and person-time) are corrected in [`PersonTime2`]'s functions,
//! several biases in one draw by [`probabilistic_multiple`], and an adjusted
//! estimate from any model by the record-level correction of
//! [`record_level_misclass`].
//!
//! # Random numbers
//!
//! Draw `r` uses `Stream::seeded(replicate_seed(seed, r))`, so the draws do
//! not depend on the thread count, the order of work or the number of other
//! draws. Within a draw, a parameter pair is drawn first (sensitivity, then
//! specificity, then the other bias parameters), then the sampling error. A
//! distribution is sampled by its quantile function at a uniform draw. Two
//! parameters with a correlation `rho` are drawn as correlated standard
//! normals, mapped through the normal distribution function to uniforms, and
//! through the two quantile functions.
//!
//! # Not included
//!
//! Regression calibration and SIMEX are not here. A correction that gives a
//! negative cell is `None`, not a number.
//!
//! # References
//!
//! 1. Lash TL, Fox MP, Fink AK. Applying Quantitative Bias Analysis to
//!    Epidemiologic Data. New York: Springer; 2009.
//! 2. Fox MP, Lash TL, Greenland S. A method to automate probabilistic
//!    sensitivity analyses of misclassified binary variables. Int J
//!    Epidemiol. 2005;34(6):1370-6.
//! 3. Lin DY, Psaty BM, Kronmal RA. Assessing the sensitivity of regression
//!    results to unmeasured confounders in observational studies. Biometrics.
//!    1998;54(3):948-63.
//! 4. Greenland S. Basic methods for sensitivity analysis of biases. Int J
//!    Epidemiol. 1996;25(6):1107-16.
//! 5. Knuth DE. The Art of Computer Programming, vol 2. 3rd ed. Addison-Wesley;
//!    1997. Section 3.4.1, exercise 22 (binomial variates by beta splitting).

use epistat_rng::{Stream, replicate_seed};

mod multiple;
mod rates;
mod record;

pub use multiple::{BiasStep, ConfoundingStep, probabilistic_multiple};
pub use rates::{
    CorrectedRates, correct_exposure_rates, correct_outcome_rates_ppv, probabilistic_misclass_rates,
};
pub use record::record_level_misclass;

use epistat_stats::dist::beta;
use epistat_stats::dist::normal;
use epistat_stats::quantile::type7;

use crate::tables::Counts2x2;

/// `x`, or `None` for NaN.
fn finite(x: f64) -> Option<f64> {
    (!x.is_nan()).then_some(x)
}

/// What is wrong with the input of a bias analysis.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum BiasError {
    /// A table that is not finite and non-negative, or has a margin of zero.
    #[error("invalid table: {0}")]
    Table(&'static str),
    /// A bias parameter or distribution outside its range.
    #[error("invalid parameter: {0}")]
    Parameter(&'static str),
    /// The assumed parameters give a negative cell in a corrected table.
    #[error("the parameters give a negative cell in the corrected table")]
    NegativeCell,
    /// Every draw was discarded.
    #[error("all {discarded} draws were discarded: the parameters are incompatible with the table")]
    NoValidDraws {
        /// The number of draws discarded.
        discarded: usize,
    },
}

impl Counts2x2 {
    fn check(&self) -> Result<(), BiasError> {
        let cells = [
            self.exposed_cases,
            self.unexposed_cases,
            self.exposed_noncases,
            self.unexposed_noncases,
        ];
        if cells.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(BiasError::Table("cells must be finite and not negative"));
        }
        if self.exposed_cases + self.unexposed_cases <= 0.0
            || self.exposed_noncases + self.unexposed_noncases <= 0.0
            || self.exposed_cases + self.exposed_noncases <= 0.0
            || self.unexposed_cases + self.unexposed_noncases <= 0.0
        {
            return Err(BiasError::Table("every row and column needs a count"));
        }
        Ok(())
    }

    fn is_integer(&self) -> bool {
        [
            self.exposed_cases,
            self.unexposed_cases,
            self.exposed_noncases,
            self.unexposed_noncases,
        ]
        .iter()
        .all(|x| x.fract() == 0.0)
    }
}

/// A corrected table and its measures. `cells` is `None`, and so are the
/// measures, when a corrected cell is negative: the assumed parameters are then
/// incompatible with the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corrected {
    /// The corrected cells.
    pub cells: Option<Counts2x2>,
    /// The corrected risk ratio.
    pub risk_ratio: Option<f64>,
    /// The corrected odds ratio.
    pub odds_ratio: Option<f64>,
    /// The corrected risk difference.
    pub risk_difference: Option<f64>,
}

impl Corrected {
    fn from_cells(t: Counts2x2) -> Self {
        if [
            t.exposed_cases,
            t.unexposed_cases,
            t.exposed_noncases,
            t.unexposed_noncases,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Self {
                cells: None,
                risk_ratio: None,
                odds_ratio: None,
                risk_difference: None,
            };
        }
        Self {
            cells: Some(t),
            risk_ratio: t.risk_ratio(),
            odds_ratio: t.odds_ratio(),
            risk_difference: t.risk_difference(),
        }
    }
}

/// Sensitivity and specificity of a classification in two strata. For
/// exposure misclassification the strata are cases (index 0) and non-cases
/// (1). For outcome misclassification they are exposed (0) and unexposed (1)
/// persons. Equal entries give a non-differential misclassification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Misclass {
    /// Sensitivity in each stratum.
    pub se: [f64; 2],
    /// Specificity in each stratum.
    pub sp: [f64; 2],
}

impl Misclass {
    fn check(&self) -> Result<(), BiasError> {
        for k in 0..2 {
            if !(0.0..=1.0).contains(&self.se[k]) || !(0.0..=1.0).contains(&self.sp[k]) {
                return Err(BiasError::Parameter(
                    "sensitivity and specificity must be in [0, 1]",
                ));
            }
            if self.se[k] - (1.0 - self.sp[k]) <= 0.0 {
                return Err(BiasError::Parameter(
                    "sensitivity plus specificity must exceed 1",
                ));
            }
        }
        Ok(())
    }
}

/// The corrected cells for exposure misclassification.
fn exposure_cells(t: &Counts2x2, se: [f64; 2], sp: [f64; 2]) -> Counts2x2 {
    let big_a = (t.exposed_cases - (1.0 - sp[0]) * (t.exposed_cases + t.unexposed_cases))
        / (se[0] - (1.0 - sp[0]));
    let big_c = (t.exposed_noncases - (1.0 - sp[1]) * (t.exposed_noncases + t.unexposed_noncases))
        / (se[1] - (1.0 - sp[1]));
    Counts2x2 {
        exposed_cases: big_a,
        unexposed_cases: (t.exposed_cases + t.unexposed_cases) - big_a,
        exposed_noncases: big_c,
        unexposed_noncases: (t.exposed_noncases + t.unexposed_noncases) - big_c,
    }
}

/// The corrected cells for outcome misclassification.
fn outcome_cells(t: &Counts2x2, se: [f64; 2], sp: [f64; 2]) -> Counts2x2 {
    let big_a = (t.exposed_cases - (1.0 - sp[0]) * (t.exposed_cases + t.exposed_noncases))
        / (se[0] - (1.0 - sp[0]));
    let big_b = (t.unexposed_cases - (1.0 - sp[1]) * (t.unexposed_cases + t.unexposed_noncases))
        / (se[1] - (1.0 - sp[1]));
    Counts2x2 {
        exposed_cases: big_a,
        unexposed_cases: big_b,
        exposed_noncases: (t.exposed_cases + t.exposed_noncases) - big_a,
        unexposed_noncases: (t.unexposed_cases + t.unexposed_noncases) - big_b,
    }
}

/// Corrects a table for misclassification of the exposure, with the
/// sensitivity and specificity among cases (stratum 0) and non-cases
/// (stratum 1).
///
/// # Errors
///
/// [`BiasError`] when the table or a parameter is invalid.
pub fn correct_exposure(t: &Counts2x2, m: &Misclass) -> Result<Corrected, BiasError> {
    t.check()?;
    m.check()?;
    Ok(Corrected::from_cells(exposure_cells(t, m.se, m.sp)))
}

/// Corrects a table for misclassification of the outcome, with the
/// sensitivity and specificity among the exposed (stratum 0) and the
/// unexposed (stratum 1).
///
/// # Errors
///
/// [`BiasError`] when the table or a parameter is invalid.
pub fn correct_outcome(t: &Counts2x2, m: &Misclass) -> Result<Corrected, BiasError> {
    t.check()?;
    m.check()?;
    Ok(Corrected::from_cells(outcome_cells(t, m.se, m.sp)))
}

/// Corrects a table for misclassification of the exposure from predictive
/// values: `ppv` and `npv` of the classification among cases (index 0) and
/// non-cases (index 1).
///
/// # Errors
///
/// [`BiasError`] when the table is invalid or a predictive value is outside
/// `[0, 1]`.
pub fn correct_exposure_predictive(
    t: &Counts2x2,
    ppv: [f64; 2],
    npv: [f64; 2],
) -> Result<Corrected, BiasError> {
    t.check()?;
    if ppv.iter().chain(&npv).any(|x| !(0.0..=1.0).contains(x)) {
        return Err(BiasError::Parameter("predictive values must be in [0, 1]"));
    }
    let big_a = t.exposed_cases * ppv[0] + t.unexposed_cases * (1.0 - npv[0]);
    let big_c = t.exposed_noncases * ppv[1] + t.unexposed_noncases * (1.0 - npv[1]);
    Ok(Corrected::from_cells(Counts2x2 {
        exposed_cases: big_a,
        unexposed_cases: (t.exposed_cases + t.unexposed_cases) - big_a,
        exposed_noncases: big_c,
        unexposed_noncases: (t.exposed_noncases + t.unexposed_noncases) - big_c,
    }))
}

/// Corrects a table for misclassification of the outcome from predictive
/// values: `ppv` and `npv` of the outcome classification among the exposed
/// (index 0) and the unexposed (index 1). The true cases among the exposed
/// are `a PPV_0 + c (1 - NPV_0)`, and among the unexposed `b PPV_1 + d (1 -
/// NPV_1)`.
///
/// # Errors
///
/// [`BiasError`] when the table is invalid or a predictive value is outside
/// `[0, 1]`.
pub fn correct_outcome_predictive(
    t: &Counts2x2,
    ppv: [f64; 2],
    npv: [f64; 2],
) -> Result<Corrected, BiasError> {
    t.check()?;
    if ppv.iter().chain(&npv).any(|x| !(0.0..=1.0).contains(x)) {
        return Err(BiasError::Parameter("predictive values must be in [0, 1]"));
    }
    let big_a = t.exposed_cases * ppv[0] + t.exposed_noncases * (1.0 - npv[0]);
    let big_b = t.unexposed_cases * ppv[1] + t.unexposed_noncases * (1.0 - npv[1]);
    Ok(Corrected::from_cells(Counts2x2 {
        exposed_cases: big_a,
        unexposed_cases: big_b,
        exposed_noncases: (t.exposed_cases + t.exposed_noncases) - big_a,
        unexposed_noncases: (t.unexposed_cases + t.unexposed_noncases) - big_b,
    }))
}

fn check_ppv_sensitivity(ppv: [f64; 2], se: [f64; 2]) -> Result<(), BiasError> {
    if ppv.iter().any(|x| !(0.0..=1.0).contains(x)) {
        return Err(BiasError::Parameter("predictive values must be in [0, 1]"));
    }
    if se.iter().any(|x| !(*x > 0.0 && *x <= 1.0)) {
        return Err(BiasError::Parameter("sensitivities must be in (0, 1]"));
    }
    Ok(())
}

/// Corrects a table for misclassification of the outcome from the positive
/// predictive value and the sensitivity of the outcome classification among
/// the exposed (index 0) and the unexposed (index 1), as when a validation
/// study reports the share of recorded cases that are real.
///
/// The recorded cases of a group that are real are `PPV` times the recorded,
/// and they are the share `Se` of all real cases, so the true cases are the
/// recorded times `PPV / Se`. The specificity this implies differs between
/// groups of different risk. With the same `PPV` and `Se` in both groups the
/// risk ratio is unchanged.
///
/// # Errors
///
/// [`BiasError`] when the table is invalid, a predictive value is outside
/// `[0, 1]` or a sensitivity outside `(0, 1]`.
pub fn correct_outcome_ppv(
    t: &Counts2x2,
    ppv: [f64; 2],
    se: [f64; 2],
) -> Result<Corrected, BiasError> {
    t.check()?;
    check_ppv_sensitivity(ppv, se)?;
    let big_a = t.exposed_cases * ppv[0] / se[0];
    let big_b = t.unexposed_cases * ppv[1] / se[1];
    Ok(Corrected::from_cells(Counts2x2 {
        exposed_cases: big_a,
        unexposed_cases: big_b,
        exposed_noncases: (t.exposed_cases + t.exposed_noncases) - big_a,
        unexposed_noncases: (t.unexposed_cases + t.unexposed_noncases) - big_b,
    }))
}

/// Corrects a table for misclassification of the exposure from the positive
/// predictive value and the sensitivity of the exposure classification among
/// cases (index 0) and non-cases (index 1): the truly exposed are the
/// recorded exposed times `PPV / Se`, as in [`correct_outcome_ppv`].
///
/// # Errors
///
/// As [`correct_outcome_ppv`].
pub fn correct_exposure_ppv(
    t: &Counts2x2,
    ppv: [f64; 2],
    se: [f64; 2],
) -> Result<Corrected, BiasError> {
    t.check()?;
    check_ppv_sensitivity(ppv, se)?;
    let big_a = t.exposed_cases * ppv[0] / se[0];
    let big_c = t.exposed_noncases * ppv[1] / se[1];
    Ok(Corrected::from_cells(Counts2x2 {
        exposed_cases: big_a,
        unexposed_cases: (t.exposed_cases + t.unexposed_cases) - big_a,
        exposed_noncases: big_c,
        unexposed_noncases: (t.exposed_noncases + t.unexposed_noncases) - big_c,
    }))
}

/// Selection probabilities by cell of the table: `[[a, b], [c, d]]`, the
/// probability that a person of that cell is in the table.
pub type Selection = [[f64; 2]; 2];

fn check_selection(s: &Selection) -> Result<(), BiasError> {
    if s.iter().flatten().any(|x| !(*x > 0.0 && *x <= 1.0)) {
        return Err(BiasError::Parameter(
            "selection probabilities must be in (0, 1]",
        ));
    }
    Ok(())
}

/// The factor `(s_b s_c) / (s_a s_d)` that the observed odds ratio is
/// multiplied by to correct for selection.
///
/// # Errors
///
/// [`BiasError::Parameter`] when a probability is not in `(0, 1]`.
pub fn selection_bias_factor(s: &Selection) -> Result<f64, BiasError> {
    check_selection(s)?;
    Ok((s[0][1] * s[1][0]) / (s[0][0] * s[1][1]))
}

/// Corrects a table for selection: each cell is divided by its selection
/// probability.
///
/// # Errors
///
/// [`BiasError`] when the table or a probability is invalid.
pub fn correct_selection(t: &Counts2x2, s: &Selection) -> Result<Corrected, BiasError> {
    t.check()?;
    check_selection(s)?;
    Ok(Corrected::from_cells(selection_cells(t, s)))
}

fn selection_cells(t: &Counts2x2, s: &Selection) -> Counts2x2 {
    Counts2x2 {
        exposed_cases: t.exposed_cases / s[0][0],
        unexposed_cases: t.unexposed_cases / s[0][1],
        exposed_noncases: t.exposed_noncases / s[1][0],
        unexposed_noncases: t.unexposed_noncases / s[1][1],
    }
}

/// The bias factor of an unmeasured binary confounder, `(p1 (R - 1) + 1) / (p0
/// (R - 1) + 1)`: the observed risk ratio divided by it is the adjusted one.
/// `p1` and `p0` are the prevalences among the exposed and the unexposed, and
/// `rr_cd` the risk ratio between confounder and outcome among the unexposed.
///
/// # Errors
///
/// [`BiasError::Parameter`] when a prevalence is outside `[0, 1]` or `rr_cd`
/// is not positive.
pub fn confounding_bias_factor(p1: f64, p0: f64, rr_cd: f64) -> Result<f64, BiasError> {
    if !(0.0..=1.0).contains(&p1) || !(0.0..=1.0).contains(&p0) {
        return Err(BiasError::Parameter("prevalences must be in [0, 1]"));
    }
    if !(rr_cd > 0.0 && rr_cd.is_finite()) {
        return Err(BiasError::Parameter(
            "the confounder-outcome risk ratio must be positive",
        ));
    }
    Ok((p1 * (rr_cd - 1.0) + 1.0) / (p0 * (rr_cd - 1.0) + 1.0))
}

/// The measure of a confounding analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfoundingMeasure {
    /// The risk ratio.
    RiskRatio,
    /// The odds ratio.
    OddsRatio,
    /// The risk difference.
    RiskDifference,
}

/// The assumed confounder: its association with the outcome among the
/// unexposed, as a risk ratio, an odds ratio or a risk difference as the
/// measure says, and its prevalence among the exposed and unexposed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confounder {
    /// The association with the outcome among the unexposed.
    pub association: f64,
    /// The prevalence among the exposed.
    pub prevalence_exposed: f64,
    /// The prevalence among the unexposed.
    pub prevalence_unexposed: f64,
}

/// A table adjusted for an unmeasured binary confounder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confounded {
    /// The crude measure of the table.
    pub crude: f64,
    /// The adjusted measure by standardised morbidity ratio, for the ratios.
    pub adjusted_smr: Option<f64>,
    /// The adjusted measure by Mantel-Haenszel weighting.
    pub adjusted_mh: f64,
    /// The cells of persons with the confounder.
    pub present: Counts2x2,
    /// The cells of persons without it.
    pub absent: Counts2x2,
}

/// Splits a table by an unmeasured binary confounder and adjusts the crude
/// measure for it. With the risk ratio the adjusted value is the crude one
/// divided by [`confounding_bias_factor`].
///
/// # Errors
///
/// [`BiasError`] when the table or a parameter is invalid, or the parameters
/// give a negative cell in a stratum ([`BiasError::NegativeCell`]).
pub fn adjust_confounding(
    t: &Counts2x2,
    measure: ConfoundingMeasure,
    c: &Confounder,
) -> Result<Confounded, BiasError> {
    t.check()?;
    if !(0.0..=1.0).contains(&c.prevalence_exposed)
        || !(0.0..=1.0).contains(&c.prevalence_unexposed)
    {
        return Err(BiasError::Parameter("prevalences must be in [0, 1]"));
    }
    if measure != ConfoundingMeasure::RiskDifference
        && !(c.association > 0.0 && c.association.is_finite())
    {
        return Err(BiasError::Parameter(
            "the confounder-outcome association must be positive",
        ));
    }
    if !c.association.is_finite() {
        return Err(BiasError::Parameter("the association must be finite"));
    }
    let (a, b, cc, d) = (
        t.exposed_cases,
        t.unexposed_cases,
        t.exposed_noncases,
        t.unexposed_noncases,
    );
    let r = c.association;
    let (present, absent, crude, smr, mh) = match measure {
        ConfoundingMeasure::RiskRatio => {
            let m1 = (a + cc) * c.prevalence_exposed;
            let n1 = (b + d) * c.prevalence_unexposed;
            let a1 = (r * m1 * a) / (r * m1 + (a + cc) - m1);
            let b1 = (r * n1 * b) / (r * n1 + (b + d) - n1);
            let (c1, d1) = (m1 - a1, n1 - b1);
            let (m0, n0) = (a + cc - m1, b + d - n1);
            let (a0, b0, c0, d0) = (a - a1, b - b1, cc - c1, d - d1);
            let smr = a / ((m1 * b1 / n1) + (m0 * b0 / n0));
            let mh = (a1 * n1 / (m1 + n1) + a0 * n0 / (m0 + n0))
                / (b1 * m1 / (m1 + n1) + b0 * m0 / (m0 + n0));
            (
                Counts2x2 {
                    exposed_cases: a1,
                    unexposed_cases: b1,
                    exposed_noncases: c1,
                    unexposed_noncases: d1,
                },
                Counts2x2 {
                    exposed_cases: a0,
                    unexposed_cases: b0,
                    exposed_noncases: c0,
                    unexposed_noncases: d0,
                },
                t.risk_ratio(),
                Some(smr),
                mh,
            )
        }
        ConfoundingMeasure::OddsRatio => {
            let c1 = cc * c.prevalence_exposed;
            let d1 = d * c.prevalence_unexposed;
            let a1 = (r * c1 * a) / (r * c1 + cc - c1);
            let b1 = (r * d1 * b) / (r * d1 + d - d1);
            let (c0, d0) = (cc - c1, d - d1);
            let (a0, b0) = (a - a1, b - b1);
            let (m1, n1) = (a1 + c1, b1 + d1);
            let (m0, n0) = (a0 + c0, b0 + d0);
            let smr = a / ((c1 * b1 / d1) + (c0 * b0 / d0));
            let mh = (a1 * d1 / (m1 + n1) + a0 * d0 / (m0 + n0))
                / (b1 * c1 / (m1 + n1) + b0 * c0 / (m0 + n0));
            (
                Counts2x2 {
                    exposed_cases: a1,
                    unexposed_cases: b1,
                    exposed_noncases: c1,
                    unexposed_noncases: d1,
                },
                Counts2x2 {
                    exposed_cases: a0,
                    unexposed_cases: b0,
                    exposed_noncases: c0,
                    unexposed_noncases: d0,
                },
                t.odds_ratio(),
                Some(smr),
                mh,
            )
        }
        ConfoundingMeasure::RiskDifference => {
            let m1 = (a + cc) * c.prevalence_exposed;
            let n1 = (b + d) * c.prevalence_unexposed;
            let (m0, n0) = (a + cc - m1, b + d - n1);
            let a1 = (r * m1 * m0 + m1 * a) / (a + cc);
            let b1 = (r * n1 * n0 + n1 * b) / (b + d);
            let (c1, d1) = (m1 - a1, n1 - b1);
            let (a0, b0, c0, d0) = (a - a1, b - b1, cc - c1, d - d1);
            let mh = (((a1 * n1 - b1 * m1) / (m1 + n1)) + ((a0 * n0 - b0 * m0) / (m0 + n0)))
                / ((m1 * n1 / (m1 + n1)) + (m0 * n0 / (m0 + n0)));
            (
                Counts2x2 {
                    exposed_cases: a1,
                    unexposed_cases: b1,
                    exposed_noncases: c1,
                    unexposed_noncases: d1,
                },
                Counts2x2 {
                    exposed_cases: a0,
                    unexposed_cases: b0,
                    exposed_noncases: c0,
                    unexposed_noncases: d0,
                },
                t.risk_difference(),
                None,
                mh,
            )
        }
    };
    let cells = [present, absent].map(|s| {
        [
            s.exposed_cases,
            s.unexposed_cases,
            s.exposed_noncases,
            s.unexposed_noncases,
        ]
    });
    if cells.iter().flatten().any(|x| x.is_nan() || *x < 0.0) {
        return Err(BiasError::NegativeCell);
    }
    let crude = crude.ok_or(BiasError::Table("the crude measure is undefined"))?;
    Ok(Confounded {
        crude,
        adjusted_smr: smr,
        adjusted_mh: mh,
        present,
        absent,
    })
}

/// A distribution for a bias parameter.
///
/// Each is sampled by its quantile function at a uniform draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dist {
    /// A fixed value.
    Constant(f64),
    /// Uniform on `[lo, hi]`.
    Uniform {
        /// Lower limit.
        lo: f64,
        /// Upper limit.
        hi: f64,
    },
    /// Triangular with the given mode.
    Triangular {
        /// Lower limit.
        lo: f64,
        /// Mode.
        mode: f64,
        /// Upper limit.
        hi: f64,
    },
    /// Trapezoidal: rising from `a` to `b`, flat to `c`, falling to `d`.
    Trapezoidal {
        /// Lower limit.
        a: f64,
        /// Start of the flat part.
        b: f64,
        /// End of the flat part.
        c: f64,
        /// Upper limit.
        d: f64,
    },
    /// A normal distribution truncated to `[lo, hi]`.
    TruncatedNormal {
        /// Lower limit.
        lo: f64,
        /// Upper limit.
        hi: f64,
        /// Mean of the untruncated normal.
        mean: f64,
        /// Standard deviation of the untruncated normal.
        sd: f64,
    },
    /// The logistic transform of a normal variate, on `(0, 1)`.
    LogitNormal {
        /// Mean on the logit scale.
        mean: f64,
        /// Standard deviation on the logit scale.
        sd: f64,
    },
    /// Beta with shapes `alpha` and `beta`.
    Beta {
        /// First shape.
        alpha: f64,
        /// Second shape.
        beta: f64,
    },
    /// The exponential of a normal variate, on `(0, inf)`.
    LogNormal {
        /// Mean of the log.
        mu: f64,
        /// Standard deviation of the log.
        sigma: f64,
    },
}

impl Dist {
    /// Checks the parameters.
    ///
    /// # Errors
    ///
    /// [`BiasError::Parameter`] when they do not define a distribution.
    pub fn validate(&self) -> Result<(), BiasError> {
        let ok = match *self {
            Self::Constant(x) => x.is_finite(),
            Self::Uniform { lo, hi } => lo.is_finite() && hi.is_finite() && lo < hi,
            Self::Triangular { lo, mode, hi } => {
                [lo, mode, hi].iter().all(|x| x.is_finite()) && lo <= mode && mode <= hi && lo < hi
            }
            Self::Trapezoidal { a, b, c, d } => {
                [a, b, c, d].iter().all(|x| x.is_finite()) && a <= b && b <= c && c <= d && a < d
            }
            Self::TruncatedNormal { lo, hi, mean, sd } => {
                [lo, hi, mean].iter().all(|x| x.is_finite())
                    && lo < hi
                    && sd > 0.0
                    && sd.is_finite()
            }
            Self::LogitNormal { mean, sd }
            | Self::LogNormal {
                mu: mean,
                sigma: sd,
            } => mean.is_finite() && sd >= 0.0 && sd.is_finite(),
            Self::Beta { alpha, beta } => {
                alpha > 0.0 && beta > 0.0 && alpha.is_finite() && beta.is_finite()
            }
        };
        if ok {
            Ok(())
        } else {
            Err(BiasError::Parameter(
                "a distribution has invalid parameters",
            ))
        }
    }

    /// The quantile at `u`, in `[0, 1]`, or `None` when it cannot be computed.
    #[must_use]
    pub fn quantile(&self, u: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        match *self {
            Self::Constant(x) => Some(x),
            Self::Uniform { lo, hi } => Some(lo + u * (hi - lo)),
            Self::Triangular { lo, mode, hi } => {
                let split = (mode - lo) / (hi - lo);
                Some(if u < split {
                    lo + (u * (hi - lo) * (mode - lo)).sqrt()
                } else {
                    hi - ((1.0 - u) * (hi - lo) * (hi - mode)).sqrt()
                })
            }
            Self::Trapezoidal { a, b, c, d } => {
                // The density is `h` on [b, c] and rises or falls linearly to
                // zero at `a` and `d`.
                let h = 2.0 / ((d + c) - (a + b));
                let rise = h * (b - a) / 2.0;
                let flat = rise + h * (c - b);
                Some(if u < rise {
                    a + (2.0 * u * (b - a) / h).sqrt()
                } else if u <= flat {
                    b + (u - rise) / h
                } else {
                    d - (2.0 * (1.0 - u) * (d - c) / h).sqrt()
                })
            }
            Self::TruncatedNormal { lo, hi, mean, sd } => {
                let z = finite(normal::truncated_quantile(
                    u,
                    (lo - mean) / sd,
                    (hi - mean) / sd,
                ))?;
                Some((mean + sd * z).clamp(lo, hi))
            }
            Self::LogitNormal { mean, sd } => {
                let x = mean + sd * normal::quantile(u);
                Some(1.0 / (1.0 + (-x).exp()))
            }
            Self::Beta { alpha, beta } => finite(beta::quantile(u, alpha, beta)),
            Self::LogNormal { mu, sigma } => Some((mu + sigma * normal::quantile(u)).exp()),
        }
    }

    fn draw(&self, stream: &mut Stream) -> Option<f64> {
        self.quantile(stream.uniform())
    }
}

/// How a pair of bias parameters, one for each stratum, is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pair {
    /// One draw used for both strata: a non-differential parameter.
    Same(Dist),
    /// Independent draws.
    Independent([Dist; 2]),
    /// Draws with a correlation `rho` on the normal scale, a Gaussian copula.
    Correlated {
        /// The distributions of the two strata.
        dists: [Dist; 2],
        /// The correlation, strictly between 0 and 1.
        rho: f64,
    },
}

impl Pair {
    fn validate(&self) -> Result<(), BiasError> {
        match self {
            Self::Same(d) => d.validate(),
            Self::Independent(ds) => ds.iter().try_for_each(Dist::validate),
            Self::Correlated { dists, rho } => {
                if !(*rho > 0.0 && *rho < 1.0) {
                    return Err(BiasError::Parameter("a correlation must be in (0, 1)"));
                }
                if dists.iter().any(|d| matches!(d, Dist::Constant(_))) {
                    return Err(BiasError::Parameter(
                        "a constant cannot be correlated with another draw",
                    ));
                }
                dists.iter().try_for_each(Dist::validate)
            }
        }
    }

    fn draw(&self, stream: &mut Stream) -> Option<[f64; 2]> {
        match self {
            Self::Same(d) => d.draw(stream).map(|x| [x, x]),
            Self::Independent([d0, d1]) => Some([d0.draw(stream)?, d1.draw(stream)?]),
            Self::Correlated { dists, rho } => {
                let z0 = stream.normal();
                let z1 = rho * z0 + (1.0 - rho * rho).sqrt() * stream.normal();
                Some([
                    dists[0].quantile(normal::cdf(z0))?,
                    dists[1].quantile(normal::cdf(z1))?,
                ])
            }
        }
    }
}

/// Options of a probabilistic bias analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbOptions {
    /// The number of draws.
    pub draws: usize,
    /// The seed: draw `r` uses `Stream::seeded(replicate_seed(seed, r))`.
    pub seed: u64,
    /// Whether to add the sampling error of the table to the systematic error.
    pub random_error: bool,
    /// The two-sided level of the limits: they are the `alpha / 2` and
    /// `1 - alpha / 2` percentiles. 0.05 gives 95% limits.
    pub alpha: f64,
}

/// The median and the limits of a set of corrected values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// The median.
    pub median: f64,
    /// The `alpha / 2` percentile.
    pub lower: f64,
    /// The `1 - alpha / 2` percentile.
    pub upper: f64,
}

/// The result of a probabilistic bias analysis, for the risk ratio, the odds
/// ratio and the risk difference, with every draw.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbResult {
    /// The risk ratio with systematic error only.
    pub risk_ratio: Summary,
    /// The odds ratio with systematic error only.
    pub odds_ratio: Summary,
    /// The risk difference with systematic error only.
    pub risk_difference: Summary,
    /// The risk ratio with systematic and random error, when requested.
    pub risk_ratio_total: Option<Summary>,
    /// The odds ratio with systematic and random error, when requested.
    pub odds_ratio_total: Option<Summary>,
    /// The risk difference with systematic and random error, when requested.
    pub risk_difference_total: Option<Summary>,
    /// The draws kept.
    pub n_draws_used: usize,
    /// The draws discarded because a parameter or a corrected cell was out of
    /// range.
    pub n_discarded: usize,
    /// Every draw, kept or discarded, in the order drawn.
    pub draws: Draws<Measures>,
}

/// The corrected measures of one draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measures {
    /// The risk ratio.
    pub risk_ratio: f64,
    /// The odds ratio.
    pub odds_ratio: f64,
    /// The risk difference.
    pub risk_difference: f64,
}

/// Every draw of a probabilistic bias analysis, in the order drawn, so that
/// draw `r` is the one made from `replicate_seed(seed, r)`.
///
/// The parameters of draw `r` are `parameters[r * names.len()..(r + 1) *
/// names.len()]`, in the order of `names`; a parameter that could not be
/// drawn is NaN. A discarded draw has `None` among the corrected values. The
/// memory is about `8 (names.len() + 6)` bytes a draw.
#[derive(Debug, Clone, PartialEq)]
pub struct Draws<M> {
    /// The names of the bias parameters.
    pub names: Vec<String>,
    /// The bias parameters of every draw, `names.len()` a draw.
    pub parameters: Vec<f64>,
    /// The corrected value of every draw with systematic error only.
    pub systematic: Vec<Option<M>>,
    /// The corrected value of every draw with systematic and random error,
    /// empty when random error was not requested.
    pub total: Vec<Option<M>>,
}

impl<M> Draws<M> {
    fn new(names: Vec<String>, draws: usize, random_error: bool) -> Self {
        Self {
            parameters: Vec::with_capacity(names.len() * draws),
            names,
            systematic: Vec::with_capacity(draws),
            total: if random_error {
                Vec::with_capacity(draws)
            } else {
                Vec::new()
            },
        }
    }

    /// The number of draws.
    #[must_use]
    pub fn len(&self) -> usize {
        self.systematic.len()
    }

    /// Whether there are no draws.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.systematic.is_empty()
    }

    /// The parameters of draw `r`, in the order of `names`.
    ///
    /// # Panics
    ///
    /// When `r` is not below [`Draws::len`].
    #[must_use]
    pub fn parameters_of(&self, r: usize) -> &[f64] {
        let k = self.names.len();
        &self.parameters[r * k..(r + 1) * k]
    }
}

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_owned()).collect()
}

/// Which variable is misclassified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The exposure: strata are cases and non-cases.
    Exposure,
    /// The outcome: strata are exposed and unexposed.
    Outcome,
}

/// The corrected values of one draw: systematic, and total when random error
/// is added.
struct Draw {
    systematic: Measures,
    total: Option<Measures>,
}

struct Collector {
    draws: Draws<Measures>,
    random_error: bool,
    discarded: usize,
}

impl Collector {
    fn new(names: Vec<String>, opts: &ProbOptions) -> Self {
        Self {
            draws: Draws::new(names, opts.draws, opts.random_error),
            random_error: opts.random_error,
            discarded: 0,
        }
    }

    fn push(&mut self, parameters: &[f64], d: Option<Draw>) {
        self.draws.parameters.extend_from_slice(parameters);
        match d {
            Some(d)
                if d.systematic.risk_ratio.is_finite() && d.systematic.odds_ratio.is_finite() =>
            {
                self.draws.systematic.push(Some(d.systematic));
                if self.random_error {
                    self.draws.total.push(d.total);
                }
            }
            _ => {
                self.discarded += 1;
                self.draws.systematic.push(None);
                if self.random_error {
                    self.draws.total.push(None);
                }
            }
        }
    }

    fn finish(self, alpha: f64) -> Result<ProbResult, BiasError> {
        let kept: Vec<Measures> = self.draws.systematic.iter().flatten().copied().collect();
        if kept.is_empty() {
            return Err(BiasError::NoValidDraws {
                discarded: self.discarded,
            });
        }
        let totals: Vec<Measures> = self.draws.total.iter().flatten().copied().collect();
        let total = self.random_error && !totals.is_empty();
        let pick = |values: &[Measures], f: fn(&Measures) -> f64| -> Summary {
            let mut v: Vec<f64> = values.iter().map(f).collect();
            summarise(&mut v, alpha)
        };
        Ok(ProbResult {
            risk_ratio: pick(&kept, |m| m.risk_ratio),
            odds_ratio: pick(&kept, |m| m.odds_ratio),
            risk_difference: pick(&kept, |m| m.risk_difference),
            risk_ratio_total: total.then(|| pick(&totals, |m| m.risk_ratio)),
            odds_ratio_total: total.then(|| pick(&totals, |m| m.odds_ratio)),
            risk_difference_total: total.then(|| pick(&totals, |m| m.risk_difference)),
            n_draws_used: kept.len(),
            n_discarded: self.discarded,
            draws: self.draws,
        })
    }
}

fn summarise(values: &mut [f64], alpha: f64) -> Summary {
    values.sort_unstable_by(f64::total_cmp);
    let at = |p: f64| type7(values, p).unwrap_or(f64::NAN);
    Summary {
        median: at(0.5),
        lower: at(alpha / 2.0),
        upper: at(1.0 - alpha / 2.0),
    }
}

fn check_options(o: &ProbOptions) -> Result<(), BiasError> {
    if o.draws == 0 {
        return Err(BiasError::Parameter("the number of draws must be positive"));
    }
    if !(o.alpha > 0.0 && o.alpha < 1.0) {
        return Err(BiasError::Parameter("alpha must be in (0, 1)"));
    }
    Ok(())
}

/// A binomial variate by splitting at a beta variate: with `x ~ Beta(k, n + 1
/// - k)` for `k = 1 + n / 2`, the `k`-th smallest of `n` uniforms, the count
/// of uniforms below `p` is `k` plus a binomial on the larger half when
/// `x < p`, and a binomial on the smaller half otherwise. Exact, and
/// logarithmic in `n` in beta draws.
fn binomial(stream: &mut Stream, mut n: u64, mut p: f64) -> Option<u64> {
    if !(0.0..=1.0).contains(&p) {
        return None;
    }
    let mut k = 0_u64;
    while n > 10 {
        let half = 1 + n / 2;
        let x = stream.beta(half as f64, (n + 1 - half) as f64).ok()?;
        if x >= p {
            n = half - 1;
            p /= x;
        } else {
            k += half;
            n -= half;
            p = (p - x) / (1.0 - x);
        }
    }
    for _ in 0..n {
        if stream.uniform() < p {
            k += 1;
        }
    }
    Some(k)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "cells are checked to be whole non-negative counts before they are used as counts"
)]
fn count(x: f64) -> u64 {
    x as u64
}

fn measures(t: &Counts2x2) -> Option<Measures> {
    Some(Measures {
        risk_ratio: t.risk_ratio()?,
        odds_ratio: t.odds_ratio()?,
        risk_difference: t.risk_difference()?,
    })
}

/// The standard error of the risk difference of a table.
fn rd_standard_error(t: &Counts2x2) -> f64 {
    (t.exposed_cases * t.exposed_noncases / (t.exposed_cases + t.exposed_noncases).powi(3)
        + t.unexposed_cases * t.unexposed_noncases
            / (t.unexposed_cases + t.unexposed_noncases).powi(3))
    .sqrt()
}

/// The measures shifted by the sampling error `z` times the standard errors
/// `se` of the log risk ratio, the log odds ratio and the risk difference.
fn shifted(m: &Measures, z: f64, se: [f64; 3]) -> Measures {
    Measures {
        risk_ratio: (m.risk_ratio.ln() - z * se[0]).exp(),
        odds_ratio: (m.odds_ratio.ln() - z * se[1]).exp(),
        risk_difference: m.risk_difference - z * se[2],
    }
}

fn positive_cells(t: &Counts2x2) -> bool {
    [
        t.exposed_cases,
        t.unexposed_cases,
        t.exposed_noncases,
        t.unexposed_noncases,
    ]
    .iter()
    .all(|x| *x > 0.0)
}

/// The standard errors of the log risk ratio, the log odds ratio and the risk
/// difference of a table.
fn standard_errors(t: &Counts2x2) -> [f64; 3] {
    let rr = (1.0 / t.exposed_cases + 1.0 / t.unexposed_cases
        - 1.0 / (t.exposed_cases + t.exposed_noncases)
        - 1.0 / (t.unexposed_cases + t.unexposed_noncases))
        .sqrt();
    let or = (1.0 / t.exposed_cases
        + 1.0 / t.unexposed_cases
        + 1.0 / t.exposed_noncases
        + 1.0 / t.unexposed_noncases)
        .sqrt();
    [rr, or, rd_standard_error(t)]
}

/// A probabilistic correction for misclassification of the exposure or the
/// outcome.
///
/// `se` and `sp` are the parameter pairs of [`Misclass`]. See the module
/// documentation for the draws and the random error. The parameters of each
/// draw are named `sensitivity 0`, `sensitivity 1`, `specificity 0` and
/// `specificity 1`, for the strata of [`Misclass`].
///
/// # Errors
///
/// [`BiasError`] when the table, a distribution or an option is invalid, when
/// random error is requested for a table with fractional cells, and
/// [`BiasError::NoValidDraws`] when every draw is discarded.
pub fn probabilistic_misclass(
    t: &Counts2x2,
    kind: Kind,
    se: &Pair,
    sp: &Pair,
    opts: &ProbOptions,
) -> Result<ProbResult, BiasError> {
    t.check()?;
    se.validate()?;
    sp.validate()?;
    check_options(opts)?;
    if opts.random_error && !t.is_integer() {
        return Err(BiasError::Table("random error needs whole counts"));
    }
    let mut out = Collector::new(
        names(&[
            "sensitivity 0",
            "sensitivity 1",
            "specificity 0",
            "specificity 1",
        ]),
        opts,
    );
    for r in 0..opts.draws {
        let mut stream = Stream::seeded(replicate_seed(opts.seed, r as u64));
        let mut parameters = [f64::NAN; 4];
        let draw = misclass_draw(
            t,
            kind,
            se,
            sp,
            opts.random_error,
            &mut stream,
            &mut parameters,
        );
        out.push(&parameters, draw);
    }
    out.finish(opts.alpha)
}

fn misclass_draw(
    t: &Counts2x2,
    kind: Kind,
    se: &Pair,
    sp: &Pair,
    random_error: bool,
    stream: &mut Stream,
    parameters: &mut [f64; 4],
) -> Option<Draw> {
    let se = se.draw(stream)?;
    parameters[..2].copy_from_slice(&se);
    let sp = sp.draw(stream)?;
    parameters[2..].copy_from_slice(&sp);
    if se.iter().chain(&sp).any(|x| !(0.0..=1.0).contains(x))
        || se.iter().zip(&sp).any(|(e, p)| e - (1.0 - p) <= 0.0)
    {
        return None;
    }
    let cells = match kind {
        Kind::Exposure => exposure_cells(t, se, sp),
        Kind::Outcome => outcome_cells(t, se, sp),
    };
    if !positive_cells(&cells) {
        return None;
    }
    let systematic = measures(&cells)?;
    if !random_error {
        return Some(Draw {
            systematic,
            total: None,
        });
    }
    // Strata 0 and 1 hold the cells (A, B) and (C, D) for the exposure, and
    // (A, C) and (B, D) for the outcome.
    let (true_0, rest_0, true_1, rest_1) = match kind {
        Kind::Exposure => (
            cells.exposed_cases,
            cells.unexposed_cases,
            cells.exposed_noncases,
            cells.unexposed_noncases,
        ),
        Kind::Outcome => (
            cells.exposed_cases,
            cells.exposed_noncases,
            cells.unexposed_cases,
            cells.unexposed_noncases,
        ),
    };
    let prevalence = [
        stream.beta(true_0, rest_0).ok()?,
        stream.beta(true_1, rest_1).ok()?,
    ];
    let mut reclassified = [0_u64; 2];
    for k in 0..2 {
        let (p, s_e, s_p) = (prevalence[k], se[k], sp[k]);
        let ppv = (s_e * p) / (s_e * p + (1.0 - s_p) * (1.0 - p));
        let npv = (s_p * (1.0 - p)) / ((1.0 - s_e) * p + s_p * (1.0 - p));
        // Observed cells of the stratum: those classified positive and negative.
        let (pos, neg) = match (kind, k) {
            (Kind::Exposure, 0) => (t.exposed_cases, t.unexposed_cases),
            (Kind::Exposure, _) => (t.exposed_noncases, t.unexposed_noncases),
            (Kind::Outcome, 0) => (t.exposed_cases, t.exposed_noncases),
            (Kind::Outcome, _) => (t.unexposed_cases, t.unexposed_noncases),
        };
        reclassified[k] =
            binomial(stream, count(pos), ppv)? + binomial(stream, count(neg), 1.0 - npv)?;
    }
    let totals = match kind {
        Kind::Exposure => [
            t.exposed_cases + t.unexposed_cases,
            t.exposed_noncases + t.unexposed_noncases,
        ],
        Kind::Outcome => [
            t.exposed_cases + t.exposed_noncases,
            t.unexposed_cases + t.unexposed_noncases,
        ],
    };
    let stratum = [
        (reclassified[0] as f64, totals[0] - reclassified[0] as f64),
        (reclassified[1] as f64, totals[1] - reclassified[1] as f64),
    ];
    let again = match kind {
        Kind::Exposure => Counts2x2 {
            exposed_cases: stratum[0].0,
            unexposed_cases: stratum[0].1,
            exposed_noncases: stratum[1].0,
            unexposed_noncases: stratum[1].1,
        },
        Kind::Outcome => Counts2x2 {
            exposed_cases: stratum[0].0,
            unexposed_cases: stratum[1].0,
            exposed_noncases: stratum[0].1,
            unexposed_noncases: stratum[1].1,
        },
    };
    if !positive_cells(&again) {
        return None;
    }
    let reclassified_measures = measures(&again)?;
    let z = stream.normal();
    Some(Draw {
        systematic,
        total: Some(shifted(&reclassified_measures, z, standard_errors(&again))),
    })
}

/// A probabilistic correction for selection: `s` holds the distribution of the
/// selection probability of each cell, `[[a, b], [c, d]]`, drawn
/// independently. The parameters of each draw are named `selection a` to
/// `selection d`.
///
/// # Errors
///
/// As [`probabilistic_misclass`], without the restriction on counts.
pub fn probabilistic_selection(
    t: &Counts2x2,
    s: &[[Dist; 2]; 2],
    opts: &ProbOptions,
) -> Result<ProbResult, BiasError> {
    t.check()?;
    s.iter().flatten().try_for_each(Dist::validate)?;
    check_options(opts)?;
    let observed = (t.risk_ratio(), t.odds_ratio());
    let (Some(_), Some(_)) = observed else {
        return Err(BiasError::Table("the observed measures are undefined"));
    };
    let se_log = standard_errors(t);
    let mut out = Collector::new(
        names(&["selection a", "selection b", "selection c", "selection d"]),
        opts,
    );
    for r in 0..opts.draws {
        let mut stream = Stream::seeded(replicate_seed(opts.seed, r as u64));
        let mut parameters = [f64::NAN; 4];
        let draw = (|| {
            let p = draw_selection(s, &mut stream, &mut parameters)?;
            let systematic = measures(&selection_cells(t, &p))?;
            let total = opts
                .random_error
                .then(|| shifted(&systematic, stream.normal(), se_log));
            Some(Draw { systematic, total })
        })();
        out.push(&parameters, draw);
    }
    out.finish(opts.alpha)
}

/// Draws the four selection probabilities into `parameters`, `None` when one
/// cannot be drawn or is outside `(0, 1]`.
fn draw_selection(
    s: &[[Dist; 2]; 2],
    stream: &mut Stream,
    parameters: &mut [f64],
) -> Option<Selection> {
    let mut p = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            p[i][j] = s[i][j].draw(stream)?;
            parameters[2 * i + j] = p[i][j];
        }
    }
    if p.iter().flatten().any(|x| !(*x > 0.0 && *x <= 1.0)) {
        return None;
    }
    Some(p)
}

/// The result of a probabilistic adjustment for one measure.
#[derive(Debug, Clone, PartialEq)]
pub struct AdjustedResult {
    /// The adjusted measure with systematic error only.
    pub adjusted: Summary,
    /// The adjusted measure with systematic and random error, when requested.
    pub adjusted_total: Option<Summary>,
    /// The draws kept.
    pub n_draws_used: usize,
    /// The draws discarded because a parameter was out of range or a stratum
    /// cell negative.
    pub n_discarded: usize,
    /// Every draw, kept or discarded, in the order drawn.
    pub draws: Draws<f64>,
}

/// The adjusted value of every draw, summarised.
fn finish_single(draws: Draws<f64>, alpha: f64) -> Result<AdjustedResult, BiasError> {
    let mut systematic: Vec<f64> = draws.systematic.iter().flatten().copied().collect();
    let discarded = draws.len() - systematic.len();
    if systematic.is_empty() {
        return Err(BiasError::NoValidDraws { discarded });
    }
    let mut total: Vec<f64> = draws.total.iter().flatten().copied().collect();
    let random_error = !draws.total.is_empty();
    Ok(AdjustedResult {
        n_draws_used: systematic.len(),
        n_discarded: discarded,
        adjusted: summarise(&mut systematic, alpha),
        adjusted_total: random_error.then(|| summarise(&mut total, alpha)),
        draws,
    })
}

/// The shift of a measure of `kind` by the sampling error `z` times its
/// standard error: on the log scale for the ratios.
fn shift_measure(measure: ConfoundingMeasure, value: f64, z: f64, se: [f64; 3]) -> f64 {
    match measure {
        ConfoundingMeasure::RiskRatio => (value.ln() - z * se[0]).exp(),
        ConfoundingMeasure::OddsRatio => (value.ln() - z * se[1]).exp(),
        ConfoundingMeasure::RiskDifference => value - z * se[2],
    }
}

/// A probabilistic adjustment for an unmeasured binary confounder: the
/// prevalences among the exposed (stratum 0) and the unexposed are the
/// [`Pair`], and `association` is the distribution of the confounder-outcome
/// association among the unexposed, on the scale of `measure`. Each draw gives
/// the Mantel-Haenszel adjusted measure of [`adjust_confounding`]. With random
/// error, the adjusted log measure is shifted by a standard normal draw times
/// the standard error of the observed log measure (the standard error of the
/// risk difference, for it). The parameters of each draw are named
/// `prevalence exposed`, `prevalence unexposed` and `association`.
///
/// # Errors
///
/// [`BiasError`] when the table, a distribution or an option is invalid, and
/// [`BiasError::NoValidDraws`] when every draw is discarded.
pub fn probabilistic_confounding(
    t: &Counts2x2,
    measure: ConfoundingMeasure,
    prevalence: &Pair,
    association: &Dist,
    opts: &ProbOptions,
) -> Result<AdjustedResult, BiasError> {
    t.check()?;
    prevalence.validate()?;
    association.validate()?;
    check_options(opts)?;
    let se = standard_errors(t);
    let mut draws = Draws::new(
        names(&["prevalence exposed", "prevalence unexposed", "association"]),
        opts.draws,
        opts.random_error,
    );
    for r in 0..opts.draws {
        let mut stream = Stream::seeded(replicate_seed(opts.seed, r as u64));
        let mut parameters = [f64::NAN; 3];
        let adjusted = draw_confounding(
            t,
            measure,
            prevalence,
            association,
            &mut stream,
            &mut parameters,
        );
        draws.parameters.extend_from_slice(&parameters);
        draws.systematic.push(adjusted);
        if opts.random_error {
            draws
                .total
                .push(adjusted.map(|value| shift_measure(measure, value, stream.normal(), se)));
        }
    }
    finish_single(draws, opts.alpha)
}

/// Draws a confounder into `parameters` and gives the Mantel-Haenszel
/// adjusted measure, `None` when a draw fails or a stratum cell is negative.
fn draw_confounding(
    t: &Counts2x2,
    measure: ConfoundingMeasure,
    prevalence: &Pair,
    association: &Dist,
    stream: &mut Stream,
    parameters: &mut [f64],
) -> Option<f64> {
    let p = prevalence.draw(stream);
    if let Some(p) = p {
        parameters[..2].copy_from_slice(&p);
    }
    let assoc = association.draw(stream);
    if let Some(assoc) = assoc {
        parameters[2] = assoc;
    }
    let (p, assoc) = p.zip(assoc)?;
    adjust_confounding(
        t,
        measure,
        &Confounder {
            association: assoc,
            prevalence_exposed: p[0],
            prevalence_unexposed: p[1],
        },
    )
    .ok()
    .map(|c| c.adjusted_mh)
    .filter(|x| x.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRUE: Counts2x2 = Counts2x2 {
        exposed_cases: 300.0,
        unexposed_cases: 200.0,
        exposed_noncases: 2000.0,
        unexposed_noncases: 3500.0,
    };

    fn close(x: f64, y: f64) -> bool {
        (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0)
    }

    /// Observed exposure counts when each exposed person is classified with
    /// sensitivity `se` and each unexposed with specificity `sp`.
    fn observed_exposure(t: &Counts2x2, m: &Misclass) -> Counts2x2 {
        let seen = |exposed: f64, unexposed: f64, k: usize| {
            (
                m.se[k] * exposed + (1.0 - m.sp[k]) * unexposed,
                (1.0 - m.se[k]) * exposed + m.sp[k] * unexposed,
            )
        };
        let (a, b) = seen(t.exposed_cases, t.unexposed_cases, 0);
        let (c, d) = seen(t.exposed_noncases, t.unexposed_noncases, 1);
        Counts2x2 {
            exposed_cases: a,
            unexposed_cases: b,
            exposed_noncases: c,
            unexposed_noncases: d,
        }
    }

    #[test]
    fn exposure_correction_inverts_the_misclassification() {
        let m = Misclass {
            se: [0.9, 0.8],
            sp: [0.95, 0.97],
        };
        let corrected = correct_exposure(&observed_exposure(&TRUE, &m), &m).unwrap();
        let cells = corrected.cells.unwrap();
        for (got, want) in [
            cells.exposed_cases,
            cells.unexposed_cases,
            cells.exposed_noncases,
            cells.unexposed_noncases,
        ]
        .into_iter()
        .zip([
            TRUE.exposed_cases,
            TRUE.unexposed_cases,
            TRUE.exposed_noncases,
            TRUE.unexposed_noncases,
        ]) {
            assert!(close(got, want), "{got} {want}");
        }
        assert!(close(
            corrected.odds_ratio.unwrap(),
            TRUE.odds_ratio().unwrap()
        ));
    }

    #[test]
    fn outcome_correction_inverts_the_misclassification() {
        // Exchange the roles: strata are exposed and unexposed, and the
        // classification is of the outcome.
        let m = Misclass {
            se: [0.85, 0.75],
            sp: [0.99, 0.96],
        };
        let flipped = Counts2x2 {
            exposed_cases: TRUE.exposed_cases,
            unexposed_cases: TRUE.exposed_noncases,
            exposed_noncases: TRUE.unexposed_cases,
            unexposed_noncases: TRUE.unexposed_noncases,
        };
        let seen = observed_exposure(&flipped, &m);
        let observed = Counts2x2 {
            exposed_cases: seen.exposed_cases,
            unexposed_cases: seen.exposed_noncases,
            exposed_noncases: seen.unexposed_cases,
            unexposed_noncases: seen.unexposed_noncases,
        };
        let cells = correct_outcome(&observed, &m).unwrap().cells.unwrap();
        for (got, want) in [
            cells.exposed_cases,
            cells.unexposed_cases,
            cells.exposed_noncases,
            cells.unexposed_noncases,
        ]
        .into_iter()
        .zip([
            TRUE.exposed_cases,
            TRUE.unexposed_cases,
            TRUE.exposed_noncases,
            TRUE.unexposed_noncases,
        ]) {
            assert!(close(got, want), "{got} {want}");
        }
    }

    #[test]
    fn perfect_classification_is_the_identity() {
        let m = Misclass {
            se: [1.0; 2],
            sp: [1.0; 2],
        };
        assert_eq!(correct_exposure(&TRUE, &m).unwrap().cells, Some(TRUE));
        assert_eq!(correct_outcome(&TRUE, &m).unwrap().cells, Some(TRUE));
        assert_eq!(
            correct_selection(&TRUE, &[[1.0; 2]; 2]).unwrap().cells,
            Some(TRUE)
        );
    }

    #[test]
    fn a_negative_cell_is_not_a_number() {
        // Observing 50 exposed cases of 500 cannot come from a classification
        // that is 99% specific, which would give at least 5 false positives
        // among 500 and then a negative number of true exposed cases.
        let t = Counts2x2 {
            exposed_cases: 2.0,
            unexposed_cases: 498.0,
            exposed_noncases: 100.0,
            unexposed_noncases: 900.0,
        };
        let m = Misclass {
            se: [0.9, 0.9],
            sp: [0.95, 0.95],
        };
        let corrected = correct_exposure(&t, &m).unwrap();
        assert_eq!(corrected.cells, None);
        assert_eq!(corrected.odds_ratio, None);
        assert!(
            correct_exposure(
                &t,
                &Misclass {
                    se: [0.5, 0.5],
                    sp: [0.5, 0.5]
                }
            )
            .is_err(),
            "no better than chance"
        );
        assert!(
            correct_exposure(
                &t,
                &Misclass {
                    se: [1.1, 0.5],
                    sp: [0.9, 0.9]
                }
            )
            .is_err()
        );
    }

    /// The correction of a risk difference for exposure missed with known
    /// sensitivity and perfect specificity: the true difference is the
    /// observed one times `(1 - p) / (1 - p / Se)`.
    #[test]
    fn the_known_sensitivity_case_is_the_inflation_factor() {
        let (pi, se, r1, r0, n) = (0.2, 0.7, 0.3, 0.1, 1_000_000.0);
        let truth = Counts2x2 {
            exposed_cases: r1 * pi * n,
            unexposed_cases: r0 * (1.0 - pi) * n,
            exposed_noncases: (1.0 - r1) * pi * n,
            unexposed_noncases: (1.0 - r0) * (1.0 - pi) * n,
        };
        // Exposure is classified for everyone with the same sensitivity and no
        // false positives; the strata are cases and non-cases.
        let m = Misclass {
            se: [se; 2],
            sp: [1.0; 2],
        };
        // The table above has exposed in `a`, `c` and unexposed in `b`, `d`
        // within outcome rows; reshape to cases (a, b) and non-cases (c, d)
        // with exposure columns.
        let by_outcome = Counts2x2 {
            exposed_cases: truth.exposed_cases,
            unexposed_cases: r0 * (1.0 - pi) * n,
            exposed_noncases: truth.exposed_noncases,
            unexposed_noncases: truth.unexposed_noncases,
        };
        let observed = observed_exposure(&by_outcome, &m);
        let p = (observed.exposed_cases + observed.exposed_noncases)
            / (observed.exposed_cases
                + observed.unexposed_cases
                + observed.exposed_noncases
                + observed.unexposed_noncases);
        let rd_observed = observed.risk_difference().unwrap();
        let corrected = correct_exposure(&observed, &m).unwrap();
        let factor = (1.0 - p) / (1.0 - p / se);
        assert!(close(
            corrected.risk_difference.unwrap(),
            rd_observed * factor
        ));
        assert!(close(corrected.risk_difference.unwrap(), r1 - r0));
    }

    #[test]
    fn selection_multiplies_the_odds_ratio() {
        let s = [[0.9, 0.7], [0.8, 0.95]];
        let corrected = correct_selection(&TRUE, &s).unwrap();
        let factor = selection_bias_factor(&s).unwrap();
        assert!(close(
            corrected.odds_ratio.unwrap(),
            TRUE.odds_ratio().unwrap() * factor
        ));
        assert!(correct_selection(&TRUE, &[[0.0, 1.0], [1.0, 1.0]]).is_err());
        assert!(correct_selection(&TRUE, &[[1.5, 1.0], [1.0, 1.0]]).is_err());
    }

    #[test]
    fn the_bias_factor_is_the_crude_over_the_adjusted_risk_ratio() {
        let c = Confounder {
            association: 2.0,
            prevalence_exposed: 0.4,
            prevalence_unexposed: 0.2,
        };
        let adjusted = adjust_confounding(&TRUE, ConfoundingMeasure::RiskRatio, &c).unwrap();
        let factor = confounding_bias_factor(0.4, 0.2, 2.0).unwrap();
        assert!(close(factor, 1.4 / 1.2));
        assert!(close(adjusted.crude / factor, adjusted.adjusted_mh));
        assert!(close(adjusted.adjusted_smr.unwrap(), adjusted.adjusted_mh));
        // A confounder with no association or equal prevalence changes nothing.
        assert!(close(confounding_bias_factor(0.3, 0.3, 5.0).unwrap(), 1.0));
        assert!(close(confounding_bias_factor(0.4, 0.2, 1.0).unwrap(), 1.0));
        assert!(confounding_bias_factor(1.2, 0.2, 2.0).is_err());
        assert!(confounding_bias_factor(0.2, 0.2, 0.0).is_err());
    }

    #[test]
    fn a_confounder_stratum_with_a_negative_cell_is_refused() {
        let c = Confounder {
            association: 100.0,
            prevalence_exposed: 0.001,
            prevalence_unexposed: 0.5,
        };
        let t = Counts2x2 {
            exposed_cases: 40.0,
            unexposed_cases: 100.0,
            exposed_noncases: 2000.0,
            unexposed_noncases: 3500.0,
        };
        assert_eq!(
            adjust_confounding(&t, ConfoundingMeasure::RiskRatio, &c).unwrap_err(),
            BiasError::NegativeCell
        );
    }

    /// The distribution function of the triangular and trapezoidal laws,
    /// written out, to check the quantile functions by `cdf(quantile(u)) = u`.
    fn trapezoid_cdf(a: f64, b: f64, c: f64, d: f64, x: f64) -> f64 {
        let h = 2.0 / ((d + c) - (a + b));
        if x <= a {
            0.0
        } else if x < b {
            h * (x - a).powi(2) / (2.0 * (b - a))
        } else if x <= c {
            h * (b - a) / 2.0 + h * (x - b)
        } else if x < d {
            1.0 - h * (d - x).powi(2) / (2.0 * (d - c))
        } else {
            1.0
        }
    }

    #[test]
    fn quantile_functions_invert_their_distribution_functions() {
        let trapezoid = Dist::Trapezoidal {
            a: 0.7,
            b: 0.8,
            c: 0.9,
            d: 1.0,
        };
        let triangle = Dist::Triangular {
            lo: 0.2,
            mode: 0.3,
            hi: 0.9,
        };
        for k in 1..100 {
            let u = f64::from(k) / 100.0;
            let x = trapezoid.quantile(u).unwrap();
            assert!(
                (trapezoid_cdf(0.7, 0.8, 0.9, 1.0, x) - u).abs() < 1e-12,
                "{u}"
            );
            // A triangle is a trapezoid with the two modes equal.
            let y = triangle.quantile(u).unwrap();
            assert!(
                (trapezoid_cdf(0.2, 0.3, 0.3, 0.9, y) - u).abs() < 1e-12,
                "{u}"
            );
        }
        assert_eq!(trapezoid.quantile(0.0), Some(0.7));
        assert_eq!(trapezoid.quantile(1.0), Some(1.0));
        let truncated = Dist::TruncatedNormal {
            lo: 0.75,
            hi: 0.95,
            mean: 0.85,
            sd: 0.05,
        };
        assert!((truncated.quantile(0.5).unwrap() - 0.85).abs() < 1e-12);
        assert!(truncated.quantile(0.0).unwrap() >= 0.75);
        assert!(truncated.quantile(1.0).unwrap() <= 0.95);
        let logit = Dist::LogitNormal { mean: 0.0, sd: 1.0 };
        assert!((logit.quantile(0.5).unwrap() - 0.5).abs() < 1e-12);
        assert!(Dist::Uniform { lo: 1.0, hi: 0.0 }.validate().is_err());
        assert!(
            Dist::Trapezoidal {
                a: 0.0,
                b: 0.5,
                c: 0.4,
                d: 1.0
            }
            .validate()
            .is_err()
        );
        assert_eq!(trapezoid.quantile(1.5), None);
    }

    #[test]
    fn binomial_has_the_right_mean_and_variance() {
        let mut stream = Stream::seeded(7);
        for (n, p) in [(7_u64, 0.4), (30, 0.1), (1000, 0.3), (1_000_000, 0.02)] {
            let reps = 20_000_u32;
            let draws: Vec<f64> = (0..reps)
                .map(|_| binomial(&mut stream, n, p).unwrap() as f64)
                .collect();
            let mean = draws.iter().sum::<f64>() / f64::from(reps);
            let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / f64::from(reps - 1);
            let (m, v) = (n as f64 * p, n as f64 * p * (1.0 - p));
            assert!(
                (mean - m).abs() < 5.0 * (v / f64::from(reps)).sqrt(),
                "{n} {p} mean {mean}"
            );
            assert!((var / v - 1.0).abs() < 0.06, "{n} {p} var {var} {v}");
            assert!(draws.iter().all(|&x| x <= n as f64));
        }
        assert_eq!(binomial(&mut stream, 5, 0.0), Some(0));
        assert_eq!(binomial(&mut stream, 5, 1.0), Some(5));
        assert_eq!(binomial(&mut stream, 5, 1.5), None);
    }

    fn options(draws: usize, random_error: bool) -> ProbOptions {
        ProbOptions {
            draws,
            seed: 11,
            random_error,
            alpha: 0.05,
        }
    }

    #[test]
    fn constant_parameters_give_the_simple_correction_in_every_draw() {
        let m = Misclass {
            se: [0.9, 0.85],
            sp: [0.95, 0.9],
        };
        let t = observed_exposure(&TRUE, &m);
        let simple = correct_exposure(&t, &m).unwrap();
        let r = probabilistic_misclass(
            &t,
            Kind::Exposure,
            &Pair::Independent([Dist::Constant(0.9), Dist::Constant(0.85)]),
            &Pair::Independent([Dist::Constant(0.95), Dist::Constant(0.9)]),
            &options(50, false),
        )
        .unwrap();
        assert!(close(r.risk_ratio.median, simple.risk_ratio.unwrap()));
        assert_eq!(r.risk_ratio.lower.to_bits(), r.risk_ratio.upper.to_bits());
        assert_eq!((r.n_draws_used, r.n_discarded), (50, 0));
        assert!(r.risk_ratio_total.is_none());
    }

    #[test]
    fn discards_are_counted_and_the_seed_fixes_the_result() {
        // Se and Sp wide enough that some draws give a negative cell.
        let t = Counts2x2 {
            exposed_cases: 30.0,
            unexposed_cases: 470.0,
            exposed_noncases: 500.0,
            unexposed_noncases: 4500.0,
        };
        let se = Pair::Same(Dist::Uniform { lo: 0.55, hi: 1.0 });
        let sp = Pair::Same(Dist::Uniform { lo: 0.9, hi: 1.0 });
        let run = |seed| {
            probabilistic_misclass(
                &t,
                Kind::Exposure,
                &se,
                &sp,
                &ProbOptions {
                    seed,
                    ..options(2000, true)
                },
            )
            .unwrap()
        };
        let (x, y) = (run(5), run(5));
        assert_eq!(x, y);
        assert!(x.n_discarded > 0, "{x:?}");
        assert_eq!(x.n_draws_used + x.n_discarded, 2000);
        assert_ne!(x, run(6));
        let total = x.risk_ratio_total.unwrap();
        assert!(total.lower < x.risk_ratio.lower && total.upper > x.risk_ratio.upper);
    }

    #[test]
    fn every_draw_discarded_is_an_error() {
        let t = Counts2x2 {
            exposed_cases: 2.0,
            unexposed_cases: 498.0,
            exposed_noncases: 100.0,
            unexposed_noncases: 900.0,
        };
        let r = probabilistic_misclass(
            &t,
            Kind::Exposure,
            &Pair::Same(Dist::Constant(0.9)),
            &Pair::Same(Dist::Constant(0.95)),
            &options(20, false),
        );
        assert_eq!(r.unwrap_err(), BiasError::NoValidDraws { discarded: 20 });
    }

    #[test]
    fn refusals() {
        let ok = Pair::Same(Dist::Constant(0.9));
        let bad = Counts2x2 {
            exposed_cases: -1.0,
            ..TRUE
        };
        assert!(
            probabilistic_misclass(&bad, Kind::Exposure, &ok, &ok, &options(5, false)).is_err()
        );
        assert!(
            probabilistic_misclass(&TRUE, Kind::Exposure, &ok, &ok, &options(0, false)).is_err()
        );
        let fractional = Counts2x2 {
            exposed_cases: 300.5,
            ..TRUE
        };
        assert!(
            probabilistic_misclass(&fractional, Kind::Exposure, &ok, &ok, &options(5, true))
                .is_err()
        );
        let corr = Pair::Correlated {
            dists: [Dist::Constant(0.9), Dist::Uniform { lo: 0.0, hi: 1.0 }],
            rho: 0.5,
        };
        assert!(
            probabilistic_misclass(&TRUE, Kind::Exposure, &corr, &ok, &options(5, false)).is_err()
        );
        let rho = Pair::Correlated {
            dists: [Dist::Uniform { lo: 0.0, hi: 1.0 }; 2],
            rho: 1.0,
        };
        assert!(
            probabilistic_misclass(&TRUE, Kind::Exposure, &rho, &ok, &options(5, false)).is_err()
        );
    }

    #[test]
    fn outcome_predictive_values_invert_the_misclassification() {
        // Classify the true outcome with sensitivity and specificity, and
        // take the predictive values of the classification from the truth.
        let m = Misclass {
            se: [0.8, 0.7],
            sp: [0.99, 0.97],
        };
        let recorded = |cases: f64, noncases: f64, k: usize| {
            let tp = m.se[k] * cases;
            let fp = (1.0 - m.sp[k]) * noncases;
            let tn = m.sp[k] * noncases;
            let fneg = (1.0 - m.se[k]) * cases;
            (tp + fp, fneg + tn, tp / (tp + fp), tn / (tn + fneg))
        };
        let (a, c, ppv0, npv0) = recorded(TRUE.exposed_cases, TRUE.exposed_noncases, 0);
        let (b, d, ppv1, npv1) = recorded(TRUE.unexposed_cases, TRUE.unexposed_noncases, 1);
        let observed = Counts2x2 {
            exposed_cases: a,
            unexposed_cases: b,
            exposed_noncases: c,
            unexposed_noncases: d,
        };
        let back = correct_outcome_predictive(&observed, [ppv0, ppv1], [npv0, npv1])
            .unwrap()
            .cells
            .unwrap();
        for (got, want) in [
            back.exposed_cases,
            back.unexposed_cases,
            back.exposed_noncases,
            back.unexposed_noncases,
        ]
        .into_iter()
        .zip([
            TRUE.exposed_cases,
            TRUE.unexposed_cases,
            TRUE.exposed_noncases,
            TRUE.unexposed_noncases,
        ]) {
            assert!(close(got, want), "{got} {want}");
        }
        // The positive predictive value and the sensitivity give the true
        // cases as well, and agree with the correction by Se and Sp.
        let by_ppv = correct_outcome_ppv(&observed, [ppv0, ppv1], m.se)
            .unwrap()
            .cells
            .unwrap();
        assert!(
            close(by_ppv.exposed_cases, TRUE.exposed_cases)
                && close(by_ppv.unexposed_cases, TRUE.unexposed_cases)
        );
        let by_se_sp = correct_outcome(&observed, &m).unwrap().cells.unwrap();
        assert!(close(by_ppv.exposed_cases, by_se_sp.exposed_cases));
        // With the same PPV and Se in both groups the risk ratio is kept.
        let same = correct_outcome_ppv(&TRUE, [0.9, 0.9], [0.8, 0.8]).unwrap();
        assert!(close(same.risk_ratio.unwrap(), TRUE.risk_ratio().unwrap()));
        assert!(correct_outcome_ppv(&TRUE, [0.9, 0.9], [0.0, 0.8]).is_err());
    }

    #[test]
    fn exposure_ppv_and_sensitivity_give_the_true_exposed() {
        let m = Misclass {
            se: [0.9, 0.8],
            sp: [0.95, 0.97],
        };
        let observed = observed_exposure(&TRUE, &m);
        let ppv = [
            m.se[0] * TRUE.exposed_cases / observed.exposed_cases,
            m.se[1] * TRUE.exposed_noncases / observed.exposed_noncases,
        ];
        let cells = correct_exposure_ppv(&observed, ppv, m.se)
            .unwrap()
            .cells
            .unwrap();
        assert!(
            close(cells.exposed_cases, TRUE.exposed_cases)
                && close(cells.exposed_noncases, TRUE.exposed_noncases)
        );
        assert!(
            close(cells.unexposed_cases, TRUE.unexposed_cases)
                && close(cells.unexposed_noncases, TRUE.unexposed_noncases)
        );
    }

    #[test]
    fn each_draw_is_the_simple_correction_of_its_parameters() {
        let t = Counts2x2 {
            exposed_cases: 150.0,
            unexposed_cases: 900.0,
            exposed_noncases: 9_850.0,
            unexposed_noncases: 89_100.0,
        };
        let se = Pair::Independent([
            Dist::Uniform { lo: 0.7, hi: 0.9 },
            Dist::Uniform { lo: 0.6, hi: 0.9 },
        ]);
        let sp = Pair::Same(Dist::Uniform { lo: 0.985, hi: 1.0 });
        let r = probabilistic_misclass(&t, Kind::Outcome, &se, &sp, &options(500, true)).unwrap();
        assert_eq!(r.draws.len(), 500);
        assert_eq!(r.draws.total.len(), 500);
        let mut discarded = 0;
        for k in 0..r.draws.len() {
            let p = r.draws.parameters_of(k);
            let simple = correct_outcome(
                &t,
                &Misclass {
                    se: [p[0], p[1]],
                    sp: [p[2], p[3]],
                },
            )
            .unwrap();
            if let Some(m) = r.draws.systematic[k] {
                assert_eq!(m.risk_ratio.to_bits(), simple.risk_ratio.unwrap().to_bits());
                assert_eq!(
                    m.risk_difference.to_bits(),
                    simple.risk_difference.unwrap().to_bits()
                );
                assert!(r.draws.total[k].is_some());
            } else {
                discarded += 1;
                assert!(simple.cells.is_none() || simple.risk_ratio.is_none());
            }
        }
        assert_eq!(discarded, r.n_discarded);
        assert!(r.n_discarded > 0);
        // The risk difference is summarised from the same draws.
        let mut rd: Vec<f64> = r
            .draws
            .systematic
            .iter()
            .flatten()
            .map(|m| m.risk_difference)
            .collect();
        let s = summarise(&mut rd, 0.05);
        assert_eq!(s, r.risk_difference);
        assert!(r.risk_difference_total.is_some());
    }

    #[test]
    fn one_step_of_the_multiple_analysis_is_the_single_analysis() {
        let t = Counts2x2 {
            exposed_cases: 150.0,
            unexposed_cases: 900.0,
            exposed_noncases: 9_850.0,
            unexposed_noncases: 89_100.0,
        };
        let se = Pair::Same(Dist::Trapezoidal {
            a: 0.7,
            b: 0.75,
            c: 0.85,
            d: 0.9,
        });
        let sp = Pair::Same(Dist::Uniform { lo: 0.99, hi: 1.0 });
        let systematic = options(400, false);
        let single = probabilistic_misclass(&t, Kind::Outcome, &se, &sp, &systematic).unwrap();
        let step = BiasStep::Misclassification {
            kind: Kind::Outcome,
            se,
            sp,
        };
        let multiple = probabilistic_multiple(
            &t,
            &[step],
            None,
            ConfoundingMeasure::RiskRatio,
            &systematic,
        )
        .unwrap();
        assert_eq!(multiple.adjusted, single.risk_ratio);
        assert_eq!(multiple.draws.names[0], "step 1: sensitivity 0");

        let s = [
            [
                Dist::Uniform { lo: 0.7, hi: 0.9 },
                Dist::Uniform { lo: 0.85, hi: 0.95 },
            ],
            [
                Dist::Uniform { lo: 0.93, hi: 0.97 },
                Dist::Uniform { lo: 0.93, hi: 0.97 },
            ],
        ];
        let random = options(400, true);
        let single = probabilistic_selection(&t, &s, &random).unwrap();
        let multiple = probabilistic_multiple(
            &t,
            &[BiasStep::Selection(s)],
            None,
            ConfoundingMeasure::OddsRatio,
            &random,
        )
        .unwrap();
        assert_eq!(multiple.adjusted, single.odds_ratio);
        assert_eq!(multiple.adjusted_total, single.odds_ratio_total);

        let prevalence = Pair::Independent([
            Dist::Beta {
                alpha: 30.0,
                beta: 70.0,
            },
            Dist::Beta {
                alpha: 20.0,
                beta: 80.0,
            },
        ]);
        let association = Dist::LogNormal {
            mu: 2f64.ln(),
            sigma: 0.2,
        };
        let single = probabilistic_confounding(
            &t,
            ConfoundingMeasure::RiskRatio,
            &prevalence,
            &association,
            &random,
        )
        .unwrap();
        let multiple = probabilistic_multiple(
            &t,
            &[],
            Some(&ConfoundingStep {
                prevalence,
                association,
            }),
            ConfoundingMeasure::RiskRatio,
            &random,
        )
        .unwrap();
        assert_eq!(multiple.adjusted, single.adjusted);
        assert_eq!(multiple.adjusted_total, single.adjusted_total);
        assert!(
            probabilistic_multiple(&t, &[], None, ConfoundingMeasure::RiskRatio, &random).is_err()
        );
    }

    #[test]
    fn chained_steps_are_the_chained_simple_corrections() {
        let t = Counts2x2 {
            exposed_cases: 150.0,
            unexposed_cases: 900.0,
            exposed_noncases: 9_850.0,
            unexposed_noncases: 89_100.0,
        };
        let steps = [
            BiasStep::Misclassification {
                kind: Kind::Outcome,
                se: Pair::Same(Dist::Uniform { lo: 0.7, hi: 0.9 }),
                sp: Pair::Same(Dist::Uniform { lo: 0.995, hi: 1.0 }),
            },
            BiasStep::Selection([[Dist::Uniform { lo: 0.8, hi: 0.95 }; 2]; 2]),
        ];
        let confounder = ConfoundingStep {
            prevalence: Pair::Same(Dist::Uniform { lo: 0.2, hi: 0.3 }),
            association: Dist::Uniform { lo: 1.5, hi: 2.5 },
        };
        let r = probabilistic_multiple(
            &t,
            &steps,
            Some(&confounder),
            ConfoundingMeasure::RiskRatio,
            &options(200, false),
        )
        .unwrap();
        assert_eq!(r.draws.names.len(), 11);
        for k in 0..r.draws.len() {
            let p = r.draws.parameters_of(k);
            let Some(value) = r.draws.systematic[k] else {
                continue;
            };
            let m = Misclass {
                se: [p[0], p[1]],
                sp: [p[2], p[3]],
            };
            let first = correct_outcome(&t, &m).unwrap().cells.unwrap();
            let second = correct_selection(&first, &[[p[4], p[5]], [p[6], p[7]]])
                .unwrap()
                .cells
                .unwrap();
            let third = adjust_confounding(
                &second,
                ConfoundingMeasure::RiskRatio,
                &Confounder {
                    association: p[10],
                    prevalence_exposed: p[8],
                    prevalence_unexposed: p[9],
                },
            )
            .unwrap();
            assert_eq!(value.to_bits(), third.adjusted_mh.to_bits());
        }
    }

    #[test]
    fn correlated_draws_are_correlated() {
        let pair = Pair::Correlated {
            dists: [Dist::Uniform { lo: 0.0, hi: 1.0 }; 2],
            rho: 0.8,
        };
        let mut stream = Stream::seeded(3);
        let n = 20_000_u32;
        let draws: Vec<[f64; 2]> = (0..n).map(|_| pair.draw(&mut stream).unwrap()).collect();
        let mean = |k: usize| draws.iter().map(|d| d[k]).sum::<f64>() / f64::from(n);
        let (m0, m1) = (mean(0), mean(1));
        let cov = draws.iter().map(|d| (d[0] - m0) * (d[1] - m1)).sum::<f64>() / f64::from(n);
        let r = cov / ((1.0 / 12.0) * (1.0 / 12.0_f64)).sqrt();
        // The correlation of the uniforms is 6 / pi * asin(rho / 2), 0.786.
        assert!((r - 0.7859).abs() < 0.02, "{r}");
    }
}
