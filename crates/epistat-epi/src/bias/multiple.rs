//! Several biases corrected in one draw.
//!
//! # Algorithm
//!
//! Greenland's multiple-bias modelling \[1\] corrects for each bias in turn
//! within a draw, in the reverse of the order in which the biases arose: the
//! last error made in producing the data is undone first. A draw draws the
//! parameters of each [`BiasStep`] in the order given, applies its
//! correction to the table the step before left, and finally adjusts for an
//! unmeasured confounder, when one is given, by the Mantel-Haenszel estimate
//! of [`super::adjust_confounding`]. A draw is discarded when a parameter is
//! out of range or a corrected cell is not positive.
//!
//! With random error, the final measure is shifted by a standard normal draw
//! times the standard error of the observed measure, on the log scale for the
//! ratios, as [`super::probabilistic_selection`] does. The reclassification
//! of [`super::probabilistic_misclass`] is not used, because after the first
//! step the table no longer holds counts.
//!
//! With one step, or a confounder alone, a draw uses the random numbers that
//! [`super::probabilistic_misclass`] (without random error),
//! [`super::probabilistic_selection`] or [`super::probabilistic_confounding`]
//! use, and gives the same value.
//!
//! # References
//!
//! 1. Greenland S. Multiple-bias modelling for analysis of observational data.
//!    J R Stat Soc Ser A. 2005;168(2):267-306.

use crate::Error;
use epistat_rng::{Stream, replicate_seed};

use super::{
    AdjustedResult, BiasError, Confounder, ConfoundingMeasure, Counts2x2, Dist, Draws, Kind, Pair,
    ProbOptions, adjust_confounding, check_options, draw_selection, exposure_cells, finish_single,
    outcome_cells, positive_cells, selection_cells, shift_measure, standard_errors,
};

/// One correction of a table within a draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BiasStep {
    /// Misclassification of the exposure or the outcome, with the parameter
    /// pairs of [`super::probabilistic_misclass`].
    Misclassification {
        /// Which variable.
        kind: Kind,
        /// The sensitivities.
        se: Pair,
        /// The specificities.
        sp: Pair,
    },
    /// Selection, with the distribution of the probability of each cell.
    Selection(super::Cells<Dist>),
}

/// An unmeasured binary confounder adjusted for last.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfoundingStep {
    /// The prevalences among the exposed (stratum 0) and the unexposed.
    pub prevalence: Pair,
    /// The association with the outcome among the unexposed, on the scale of
    /// the measure.
    pub association: Dist,
}

impl BiasStep {
    fn validate(&self) -> Result<(), Error> {
        match self {
            Self::Misclassification { se, sp, .. } => {
                se.validate()?;
                sp.validate()
            }
            Self::Selection(s) => super::selection_order(s)
                .iter()
                .try_for_each(Dist::validate),
        }
    }

    fn names(&self, step: usize, out: &mut Vec<String>) {
        let own: &[&str] = match self {
            Self::Misclassification { .. } => &[
                "sensitivity 0",
                "sensitivity 1",
                "specificity 0",
                "specificity 1",
            ],
            Self::Selection(_) => &super::SELECTION_NAMES,
        };
        out.extend(own.iter().map(|n| format!("step {step}: {n}")));
    }

    /// Draws the step's parameters into `parameters`, four of them, and
    /// corrects `t`.
    fn apply(
        &self,
        t: &Counts2x2,
        stream: &mut Stream,
        parameters: &mut [f64],
    ) -> Option<Counts2x2> {
        let cells = match self {
            Self::Misclassification { kind, se, sp } => {
                let se = se.draw(stream)?;
                parameters[..2].copy_from_slice(&se);
                let sp = sp.draw(stream)?;
                parameters[2..4].copy_from_slice(&sp);
                if se.iter().chain(&sp).any(|x| !(0.0..=1.0).contains(x))
                    || se.iter().zip(&sp).any(|(e, p)| e - (1.0 - p) <= 0.0)
                {
                    return None;
                }
                match kind {
                    Kind::Exposure => exposure_cells(t, se, sp),
                    Kind::Outcome => outcome_cells(t, se, sp),
                }
            }
            Self::Selection(s) => selection_cells(t, &draw_selection(s, stream, parameters)?),
        };
        positive_cells(&cells).then_some(cells)
    }
}

/// A probabilistic correction for several biases in one draw: the `steps` in
/// the order given, then the `confounding` adjustment when there is one, for
/// `measure`. The parameters of each draw are named `step k: ...` after the
/// names of the single analyses, with `k` from 1, and then `prevalence
/// exposed`, `prevalence unexposed` and `association`.
///
/// # Errors
///
/// [`crate::Error`] when the table, a distribution or an option is invalid,
/// [`crate::Error`] when there is neither a step nor a confounder,
/// and [`crate::Error`] when every draw is discarded.
pub fn probabilistic_multiple(
    t: &Counts2x2,
    steps: &[BiasStep],
    confounding: Option<&ConfoundingStep>,
    measure: ConfoundingMeasure,
    opts: &ProbOptions,
) -> Result<AdjustedResult, Error> {
    t.check()?;
    check_options(opts)?;
    if steps.is_empty() && confounding.is_none() {
        return Err(BiasError::Parameter("there is no bias to correct for").into());
    }
    steps.iter().try_for_each(BiasStep::validate)?;
    if let Some(c) = confounding {
        c.prevalence.validate()?;
        c.association.validate()?;
    }
    let observed = match measure {
        ConfoundingMeasure::RiskRatio => t.risk_ratio(),
        ConfoundingMeasure::OddsRatio => t.odds_ratio(),
        ConfoundingMeasure::RiskDifference => t.risk_difference(),
    };
    if observed.is_none() {
        return Err(BiasError::Table("the observed measure is undefined").into());
    }
    let se = standard_errors(t);
    let mut names = Vec::new();
    for (k, step) in steps.iter().enumerate() {
        step.names(k + 1, &mut names);
    }
    if confounding.is_some() {
        names.extend(
            ["prevalence exposed", "prevalence unexposed", "association"].map(str::to_owned),
        );
    }
    let width = names.len();
    let mut draws = Draws::new(names, opts.bootstrap.replicates, opts.random_error);
    let mut parameters = vec![f64::NAN; width];
    for r in 0..opts.bootstrap.replicates {
        let mut stream = Stream::seeded(replicate_seed(opts.bootstrap.seed.value(), r as u64));
        parameters.fill(f64::NAN);
        let value = (|| {
            let mut cells = *t;
            for (k, step) in steps.iter().enumerate() {
                cells = step.apply(&cells, &mut stream, &mut parameters[4 * k..4 * k + 4])?;
            }
            let value = match confounding {
                Some(c) => {
                    let base = 4 * steps.len();
                    let p = c.prevalence.draw(&mut stream);
                    if let Some(p) = p {
                        parameters[base..base + 2].copy_from_slice(&p);
                    }
                    let assoc = c.association.draw(&mut stream);
                    if let Some(assoc) = assoc {
                        parameters[base + 2] = assoc;
                    }
                    let (p, assoc) = p.zip(assoc)?;
                    adjust_confounding(
                        &cells,
                        measure,
                        &Confounder {
                            association: assoc,
                            prevalence_exposed: p[0],
                            prevalence_unexposed: p[1],
                        },
                    )
                    .ok()?
                    .adjusted_mh
                }
                None => match measure {
                    ConfoundingMeasure::RiskRatio => cells.risk_ratio()?,
                    ConfoundingMeasure::OddsRatio => cells.odds_ratio()?,
                    ConfoundingMeasure::RiskDifference => cells.risk_difference()?,
                },
            };
            value.is_finite().then_some(value)
        })();
        draws.push_parameters(&parameters);
        draws.systematic.push(value);
        if opts.random_error {
            draws
                .total
                .push(value.map(|v| shift_measure(measure, v, stream.normal(), se)));
        }
    }
    finish_single(draws, opts.level)
}
