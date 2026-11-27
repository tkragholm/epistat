//! Staggered adoption, one adoption cohort at a time against the
//! never-treated units, aggregated by treated cells.

use epistat_core::Estimate;

use crate::error::Error;
use crate::methods::staggered::StaggeredPanel;

use super::fit;
use super::panel::Grid;
use super::variance::{self, CohortValue, Plan, PlanCohort};
use super::{SynthDidOptions, SynthFit, SynthReplicates, SynthWeights, resolve, with_se};

/// One adoption cohort of a staggered estimate.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SynthCohort {
    /// The period the cohort is first treated in.
    pub adoption: i64,
    /// The codes of the cohort's units.
    pub treated: Vec<u32>,
    /// Its share of the treated cells, the weight in the aggregate.
    pub weight: f64,
    /// The cohort's estimate against the never-treated units.
    pub estimate: f64,
    /// The cohort's effect in each of its post periods.
    pub effect_curve: Vec<f64>,
    /// The cohort's weights, whose unit weights follow
    /// [`StaggeredSynthDid::controls`].
    pub fit: SynthFit,
}

/// A staggered estimate.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct StaggeredSynthDid {
    /// The average effect over all treated cells, with the standard error of
    /// [`SynthDidOptions::variance`], its normal interval and p-value.
    pub estimate: Estimate,
    /// The effect at each event time `0, 1, ...` (periods since adoption),
    /// the cohorts' effect curves averaged with weights `N_a`, with a
    /// standard error from the same replicates. A bootstrap replicate that
    /// lacks a cohort has no value at the event times only that cohort
    /// reaches; those event times take their standard error over the
    /// replicates that have one.
    pub event_time: Vec<Estimate>,
    /// The codes of the never-treated units.
    pub controls: Vec<u32>,
    /// The cohorts, by adoption period.
    pub cohorts: Vec<SynthCohort>,
    /// The replicates of the standard error, `None` without a variance.
    pub variance: Option<SynthReplicates>,
}

/// The cohort-by-cohort estimate described in the module documentation, each
/// cohort estimated with `options`, and its standard error by replaying the
/// whole procedure. Weights given as vectors ([`SynthWeights::Start`],
/// [`SynthWeights::Fixed`]) do not carry across cohorts and are refused.
///
/// # Errors
///
/// [`Error::Unsupported`] for weights given as vectors, and the errors of
/// [`estimate`](super::estimate) for any cohort.
pub fn estimate_staggered(
    panel: &StaggeredPanel<'_>,
    options: &SynthDidOptions,
) -> Result<StaggeredSynthDid, Error> {
    let vector = |w: &SynthWeights| matches!(w, SynthWeights::Start(_) | SynthWeights::Fixed(_));
    if vector(&options.omega) || vector(&options.lambda) {
        return Err(Error::Unsupported {
            what: "weights given as vectors in a staggered synthdid, which do not apply to every cohort",
        });
    }
    let grid = Grid::new(panel)?;
    let data = grid.data();
    let periods = grid.periods.len();
    let controls = grid.controls();
    let mut cohorts = Vec::new();
    let mut values = Vec::new();
    let mut plan = Vec::new();
    for (adoption, treated) in grid.cohorts() {
        let (opts, start, noise) = resolve(options, &data, &controls, treated.len(), adoption)?;
        let fitted = fit::fit(&data, &controls, &treated, adoption, &opts, start)?;
        values.push(CohortValue {
            treated: treated.len(),
            post: periods - adoption,
            estimate: fitted.estimate,
            curve: fitted.curve.clone(),
        });
        plan.push(PlanCohort {
            pre: adoption,
            treated: treated.clone(),
            opts,
            omega: fitted.omega.clone(),
            lambda: fitted.lambda.clone(),
        });
        cohorts.push(SynthCohort {
            adoption: grid.periods[adoption],
            treated: grid.codes(&treated),
            weight: 0.0,
            estimate: fitted.estimate,
            effect_curve: fitted.curve,
            fit: SynthFit {
                omega: fitted.omega,
                lambda: fitted.lambda,
                beta: fitted.beta,
                noise_level: noise,
                zeta_omega: opts.zeta_omega,
                zeta_lambda: opts.zeta_lambda,
                min_decrease: opts.min_decrease,
                trace: fitted.trace,
                converged: fitted.converged,
            },
        });
    }
    let cells: f64 = values.iter().map(|v| (v.treated * v.post) as f64).sum();
    for (cohort, v) in cohorts.iter_mut().zip(&values) {
        cohort.weight = (v.treated * v.post) as f64 / cells;
    }
    let length = values.iter().map(|v| v.post).max().unwrap_or(0);
    let (estimate, event_time) = variance::aggregate(&values, length);
    let variance = match options.variance.method(&grid) {
        None => None,
        Some(method) => {
            let plan = Plan {
                data,
                controls: controls.clone(),
                cohorts: plan,
            };
            Some(variance::variance(&plan, method)?)
        }
    };
    let level = options.level;
    let se = variance.as_ref().and_then(|v| v.se);
    let curve_se = |e: usize| variance.as_ref().and_then(|v| v.curve_se[e]);
    Ok(StaggeredSynthDid {
        estimate: with_se(estimate, se, level),
        event_time: event_time
            .iter()
            .enumerate()
            .map(|(e, &v)| with_se(v, curve_se(e), level))
            .collect(),
        controls: grid.codes(&controls),
        cohorts,
        variance: variance.map(|v| v.replicates(&grid.units)).transpose()?,
    })
}
