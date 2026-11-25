//! Staggered adoption, one adoption cohort at a time against the
//! never-treated units, aggregated by treated cells.

use super::fit::{self, Data};
use super::variance::{self, CohortValue, Plan, PlanCohort};
use super::{
    SynthDidConfig, SynthDidError, SynthDidEstimate, SynthDidVariance, SynthVariance, SynthWeights,
    resolve,
};

/// A balanced panel with absorbing treatment: `units × periods` outcomes by
/// unit, and for each unit the period (0-based column) it is first treated
/// in, or `None` when never treated.
#[derive(Debug, Clone, PartialEq)]
pub struct StaggeredSynthPanel {
    outcome: Vec<f64>,
    covariates: Vec<f64>,
    covariate_count: usize,
    units: usize,
    periods: usize,
    adoption: Vec<Option<usize>>,
    pub(crate) unit_ids: Vec<i64>,
    pub(crate) times: Vec<i32>,
}

impl StaggeredSynthPanel {
    /// The panel from its outcomes, unit by unit, and the adoption periods.
    ///
    /// # Errors
    ///
    /// [`SynthDidError::Dimension`], [`SynthDidError::NotFinite`], or
    /// [`SynthDidError::Design`] when no unit is never treated, none is
    /// treated, or a unit is treated in the first period or after the last.
    pub fn new(
        outcome: Vec<f64>,
        units: usize,
        periods: usize,
        adoption: Vec<Option<usize>>,
    ) -> Result<Self, SynthDidError> {
        if units.checked_mul(periods) != Some(outcome.len()) || adoption.len() != units {
            return Err(SynthDidError::Dimension("outcome is not units × periods"));
        }
        if outcome.iter().any(|v| !v.is_finite()) {
            return Err(SynthDidError::NotFinite);
        }
        let never = adoption.iter().filter(|a| a.is_none()).count();
        if never == 0
            || never == units
            || adoption.iter().flatten().any(|&a| a == 0 || a >= periods)
        {
            return Err(SynthDidError::Design);
        }
        Ok(Self {
            outcome,
            covariates: Vec::new(),
            covariate_count: 0,
            units,
            periods,
            adoption,
            unit_ids: (0..units).map(crate::util::usize_to_i64).collect(),
            times: (0..periods).map(period_label).collect(),
        })
    }

    /// Adds `count` covariates, laid out as in
    /// [`SynthPanel::with_covariates`](super::SynthPanel::with_covariates).
    ///
    /// # Errors
    ///
    /// [`SynthDidError::Dimension`] or [`SynthDidError::NotFinite`].
    pub fn with_covariates(
        mut self,
        covariates: Vec<f64>,
        count: usize,
    ) -> Result<Self, SynthDidError> {
        if self.outcome.len().checked_mul(count) != Some(covariates.len()) {
            return Err(SynthDidError::Dimension(
                "covariates are not count × units × periods",
            ));
        }
        if covariates.iter().any(|v| !v.is_finite()) {
            return Err(SynthDidError::NotFinite);
        }
        self.covariates = covariates;
        self.covariate_count = count;
        Ok(self)
    }

    /// The adoption period of each unit.
    #[must_use]
    pub fn adoption(&self) -> &[Option<usize>] {
        &self.adoption
    }

    /// The unit of each row.
    #[must_use]
    pub fn unit_ids(&self) -> &[i64] {
        &self.unit_ids
    }

    /// The period of each column.
    #[must_use]
    pub fn times(&self) -> &[i32] {
        &self.times
    }

    /// Number of units.
    #[must_use]
    pub fn units(&self) -> usize {
        self.units
    }

    /// Number of periods.
    #[must_use]
    pub fn periods(&self) -> usize {
        self.periods
    }

    fn data(&self) -> Data<'_> {
        Data {
            outcome: &self.outcome,
            covariates: &self.covariates,
            count: self.covariate_count,
            units: self.units,
            periods: self.periods,
        }
    }

    fn controls(&self) -> Vec<usize> {
        (0..self.units)
            .filter(|&u| self.adoption[u].is_none())
            .collect()
    }

    /// The adoption periods with their units, in increasing order.
    fn cohorts(&self) -> Vec<(usize, Vec<usize>)> {
        let mut periods: Vec<usize> = self.adoption.iter().flatten().copied().collect();
        periods.sort_unstable();
        periods.dedup();
        periods
            .into_iter()
            .map(|a| {
                let units = (0..self.units)
                    .filter(|&u| self.adoption[u] == Some(a))
                    .collect();
                (a, units)
            })
            .collect()
    }
}

pub(crate) fn period_label(t: usize) -> i32 {
    i32::try_from(t).unwrap_or_else(|error| panic!("period index exceeds i32::MAX: {error}"))
}

/// One adoption cohort of a staggered estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthCohort {
    /// The period the cohort is first treated in, as a 0-based column.
    pub adoption: usize,
    /// The cohort's units, in panel order.
    pub treated: Vec<usize>,
    /// Its share of the treated cells, the weight in the aggregate.
    pub weight: f64,
    /// The cohort's estimate against the never-treated units, whose unit
    /// weights follow the never-treated units in panel order.
    pub fit: SynthDidEstimate,
}

/// A staggered estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct StaggeredSynthEstimate {
    /// The average effect over all treated cells.
    pub estimate: f64,
    /// The effect at each event time `0, 1, ...` (periods since adoption),
    /// the cohorts' effect curves averaged with weights `N_a`.
    pub event_time: Vec<f64>,
    /// The never-treated units, in panel order.
    pub controls: Vec<usize>,
    /// The cohorts, by adoption period.
    pub cohorts: Vec<SynthCohort>,
    shape: (usize, usize, usize),
}

/// The cohort-by-cohort estimate described in the module documentation, each
/// cohort estimated with `config`. Weights given as vectors
/// ([`SynthWeights::Start`], [`SynthWeights::Fixed`]) do not carry across
/// cohorts and are refused.
///
/// # Errors
///
/// [`SynthDidError::Setting`] for weights given as vectors, and the errors of
/// [`estimate_synthdid`](super::estimate_synthdid) for any cohort.
pub fn estimate_synthdid_staggered(
    panel: &StaggeredSynthPanel,
    config: &SynthDidConfig,
) -> Result<StaggeredSynthEstimate, SynthDidError> {
    let vector = |w: &SynthWeights| matches!(w, SynthWeights::Start(_) | SynthWeights::Fixed(_));
    if vector(&config.omega) || vector(&config.lambda) {
        return Err(SynthDidError::Setting(
            "weights given as vectors do not apply to every cohort",
        ));
    }
    let data = panel.data();
    let controls = panel.controls();
    let mut cohorts = Vec::new();
    let mut values = Vec::new();
    for (adoption, treated) in panel.cohorts() {
        let (opts, start, noise) = resolve(config, &data, &controls, treated.len(), adoption)?;
        let fitted = fit::fit(&data, &controls, &treated, adoption, &opts, start)?;
        values.push(CohortValue {
            treated: treated.len(),
            post: panel.periods - adoption,
            estimate: fitted.estimate,
            curve: fitted.curve.clone(),
        });
        cohorts.push(SynthCohort {
            adoption,
            treated,
            weight: 0.0,
            fit: SynthDidEstimate {
                estimate: fitted.estimate,
                effect_curve: fitted.curve,
                omega: fitted.omega,
                lambda: fitted.lambda,
                beta: fitted.beta,
                noise_level: noise,
                zeta_omega: opts.zeta_omega,
                zeta_lambda: opts.zeta_lambda,
                min_decrease: opts.min_decrease,
                trace: fitted.trace,
                converged: fitted.converged,
                opts,
                shape: (
                    panel.units,
                    panel.periods,
                    controls.len(),
                    adoption,
                    panel.covariate_count,
                ),
            },
        });
    }
    let cells: f64 = values.iter().map(|v| (v.treated * v.post) as f64).sum();
    for (cohort, v) in cohorts.iter_mut().zip(&values) {
        cohort.weight = (v.treated * v.post) as f64 / cells;
    }
    let length = values.iter().map(|v| v.post).max().unwrap_or(0);
    let (estimate, event_time) = variance::aggregate(&values, length);
    Ok(StaggeredSynthEstimate {
        estimate,
        event_time,
        controls,
        cohorts,
        shape: (panel.units, panel.periods, panel.covariate_count),
    })
}

/// The standard error of a staggered estimate by replaying the whole
/// procedure, as described in the module documentation, with a standard error
/// for each event time beside it. A bootstrap replicate that lacks a cohort
/// has no value at the event times only that cohort reaches; those event
/// times take their standard error over the replicates that have one.
///
/// # Errors
///
/// [`SynthDidError::Mismatch`] when `estimate` was not computed on `panel`,
/// and the errors of [`synthdid_variance`](super::synthdid_variance).
pub fn synthdid_staggered_variance(
    panel: &StaggeredSynthPanel,
    estimate: &StaggeredSynthEstimate,
    method: SynthVariance<'_>,
) -> Result<SynthDidVariance, SynthDidError> {
    if estimate.shape != (panel.units, panel.periods, panel.covariate_count)
        || estimate.controls != panel.controls()
        || estimate
            .cohorts
            .iter()
            .map(|c| (c.adoption, c.treated.clone()))
            .ne(panel.cohorts())
    {
        return Err(SynthDidError::Mismatch);
    }
    let plan = Plan {
        data: panel.data(),
        controls: estimate.controls.clone(),
        cohorts: estimate
            .cohorts
            .iter()
            .map(|c| PlanCohort {
                pre: c.adoption,
                treated: c.treated.clone(),
                opts: c.fit.opts,
                omega: c.fit.omega.clone(),
                lambda: c.fit.lambda.clone(),
            })
            .collect(),
    };
    variance::variance(&plan, method)
}
