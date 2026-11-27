//! The placebo, bootstrap and jackknife standard errors of `synthdid`'s
//! `vcov`, on one cohort or on the cohorts of a staggered panel.

use epistat_rng::{Stream, replicate_seed};
use rayon::prelude::*;

use super::fit::{self, Data, Opts, Start, sum_normalize};
use crate::error::Error;

use super::{Draws, Method, SynthUndefined, Variance};

/// A fitted cohort as the variance methods replay it.
#[derive(Debug, Clone)]
pub(crate) struct PlanCohort {
    pub pre: usize,
    pub treated: Vec<usize>,
    pub opts: Opts,
    pub omega: Vec<f64>,
    pub lambda: Vec<f64>,
}

/// The never-treated units and the fitted cohorts.
#[derive(Debug, Clone)]
pub(crate) struct Plan<'a> {
    pub data: Data<'a>,
    pub controls: Vec<usize>,
    pub cohorts: Vec<PlanCohort>,
}

/// One cohort's estimate and effect curve with its size.
pub(crate) struct CohortValue {
    pub treated: usize,
    pub post: usize,
    pub estimate: f64,
    pub curve: Vec<f64>,
}

/// The aggregate over cohorts: weights `N_a T1_a` for the estimate, and `N_a`
/// over the cohorts observed at each event time for the curve, each
/// normalised before it multiplies, so that one cohort is returned unchanged.
/// The curve is as long as the longest cohort's; event times no cohort
/// reaches are NaN.
pub(crate) fn aggregate(values: &[CohortValue], length: usize) -> (f64, Vec<f64>) {
    let cells: f64 = values.iter().map(|v| (v.treated * v.post) as f64).sum();
    let mut estimate = 0.0;
    for v in values {
        estimate += ((v.treated * v.post) as f64 / cells) * v.estimate;
    }
    let curve = (0..length)
        .map(|e| {
            let size: f64 = values
                .iter()
                .filter(|v| v.post > e)
                .map(|v| v.treated as f64)
                .sum();
            if size == 0.0 {
                return f64::NAN;
            }
            let mut sum = 0.0;
            for v in values.iter().filter(|v| v.post > e) {
                sum += (v.treated as f64 / size) * v.curve[e];
            }
            sum
        })
        .collect();
    (estimate, curve)
}

impl Plan<'_> {
    fn curve_length(&self) -> usize {
        self.cohorts
            .iter()
            .map(|c| self.data.periods - c.pre)
            .max()
            .unwrap_or(0)
    }

    fn treated_total(&self) -> usize {
        self.cohorts.iter().map(|c| c.treated.len()).sum()
    }

    /// Refits every cohort on the given controls and treated units, starting
    /// from the cohort's weights with the unit weights taken at
    /// `control_positions` (positions in `self.controls`) and rescaled.
    fn refit(
        &self,
        controls: &[usize],
        control_positions: &[usize],
        treated: &[Vec<usize>],
    ) -> Result<(f64, Vec<f64>), Error> {
        let mut values = Vec::with_capacity(self.cohorts.len());
        for (cohort, treated) in self.cohorts.iter().zip(treated) {
            if treated.is_empty() {
                continue;
            }
            let omega = sum_normalize(control_positions.iter().map(|&p| cohort.omega[p]).collect());
            let start = Start {
                lambda: Some(cohort.lambda.clone()),
                omega: Some(omega),
            };
            let fitted = fit::fit(
                &self.data,
                controls,
                treated,
                cohort.pre,
                &cohort.opts,
                start,
            )?;
            values.push(CohortValue {
                treated: treated.len(),
                post: self.data.periods - cohort.pre,
                estimate: fitted.estimate,
                curve: fitted.curve,
            });
        }
        Ok(aggregate(&values, self.curve_length()))
    }

    /// The cohort each unit belongs to, `None` for the controls.
    fn cohort_of(&self) -> Vec<Option<usize>> {
        let mut of = vec![None; self.data.units];
        for (a, cohort) in self.cohorts.iter().enumerate() {
            for &unit in &cohort.treated {
                of[unit] = Some(a);
            }
        }
        of
    }

    /// The placebo replicate of a permutation of the controls' positions.
    fn placebo(&self, permutation: &[usize]) -> Result<(f64, Vec<f64>), Error> {
        let n_treated = self.treated_total();
        let kept = self.controls.len() - n_treated;
        let positions = &permutation[..kept];
        let controls: Vec<usize> = positions.iter().map(|&p| self.controls[p]).collect();
        // The pseudo-treated take the cohorts of the treated units in panel
        // order.
        let mut order: Vec<(usize, usize)> = self
            .cohorts
            .iter()
            .enumerate()
            .flat_map(|(a, c)| c.treated.iter().map(move |&u| (u, a)))
            .collect();
        order.sort_unstable();
        let mut treated = vec![Vec::new(); self.cohorts.len()];
        for (&p, &(_, a)) in permutation[kept..].iter().zip(&order) {
            treated[a].push(self.controls[p]);
        }
        self.refit(&controls, positions, &treated)
    }

    /// The bootstrap replicate of a draw of units, `None` when the draw has no
    /// control or no treated unit.
    fn bootstrap(
        &self,
        draw: &[usize],
        cohort_of: &[Option<usize>],
        control_position: &[Option<usize>],
    ) -> Option<Result<(f64, Vec<f64>), Error>> {
        let mut sorted = draw.to_vec();
        sorted.sort_unstable();
        let mut controls = Vec::new();
        let mut positions = Vec::new();
        let mut treated = vec![Vec::new(); self.cohorts.len()];
        for &unit in &sorted {
            if let Some(a) = cohort_of[unit] {
                treated[a].push(unit);
            } else {
                controls.push(unit);
                positions.push(control_position[unit].expect("a unit is a control or treated"));
            }
        }
        if controls.is_empty() || treated.iter().all(Vec::is_empty) {
            return None;
        }
        Some(self.refit(&controls, &positions, &treated))
    }

    /// The position of each control in `self.controls`, `None` for the
    /// treated units.
    fn control_positions(&self) -> Vec<Option<usize>> {
        let mut position = vec![None; self.data.units];
        for (p, &unit) in self.controls.iter().enumerate() {
            position[unit] = Some(p);
        }
        position
    }
}

/// `sqrt((B - 1)/B) sd(u)`, the standard error of the placebo and the
/// bootstrap, over the finite values.
fn replicate_se(values: impl Iterator<Item = f64> + Clone) -> Option<f64> {
    let finite = values.filter(|v| v.is_finite());
    let b = finite.clone().count();
    if b < 2 {
        return None;
    }
    let mean = fit::r_mean(finite.clone());
    let squares: f64 = finite.map(|v| (v - mean) * (v - mean)).sum();
    let var = squares / (b - 1) as f64;
    Some(((b - 1) as f64 / b as f64).sqrt() * var.sqrt())
}

/// `sqrt(((n - 1)/n) (n - 1) var(u))`, the jackknife standard error.
fn jackknife_se(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let mean = fit::r_mean(values.iter().copied());
    let squares: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
    let var = squares / (n - 1) as f64;
    Some((((n - 1) as f64 / n as f64) * (n - 1) as f64 * var).sqrt())
}

fn collect(
    plan: &Plan<'_>,
    replicates: &[(f64, Vec<f64>)],
    se: impl Fn(&[f64]) -> Option<f64>,
    influence: Option<Vec<f64>>,
) -> Variance {
    let estimates: Vec<f64> = replicates.iter().map(|r| r.0).collect();
    let effect_curve_se = (0..plan.curve_length())
        .map(|e| {
            let column: Vec<f64> = replicates.iter().map(|r| r.1[e]).collect();
            se(&column)
        })
        .collect();
    Variance {
        se: se(&estimates),
        curve_se: effect_curve_se,
        replicates: estimates,
        influence,
        undefined: None,
    }
}

fn undefined(plan: &Plan<'_>, reason: SynthUndefined) -> Variance {
    Variance {
        se: None,
        curve_se: vec![None; plan.curve_length()],
        replicates: Vec::new(),
        influence: None,
        undefined: Some(reason),
    }
}

pub(crate) fn variance(plan: &Plan<'_>, method: Method<'_>) -> Result<Variance, Error> {
    match method {
        Method::Placebo(draws) => placebo(plan, draws),
        Method::Bootstrap(draws) => bootstrap(plan, draws),
        Method::Jackknife(clusters) => jackknife(plan, clusters),
    }
}

fn replications(draws: Draws<'_>) -> Result<usize, Error> {
    let b = match draws {
        Draws::Seeded { replications, .. } => replications,
        Draws::Given(d) => d.len(),
    };
    if b < 2 {
        return Err(epistat_core::InputError::Domain {
            what: "replicates",
            value: b as f64,
            domain: "2 or more",
            row: None,
        }
        .into());
    }
    Ok(b)
}

fn placebo(plan: &Plan<'_>, draws: Draws<'_>) -> Result<Variance, Error> {
    let n0 = plan.controls.len();
    if n0 <= plan.treated_total() {
        return Err(Error::PlaceboControls);
    }
    let b = replications(draws)?;
    if let Draws::Given(given) = draws {
        for (r, draw) in given.iter().enumerate() {
            let mut seen = vec![false; n0];
            let ok = draw.len() == n0
                && draw
                    .iter()
                    .all(|&p| p < n0 && !std::mem::replace(&mut seen[p], true));
            if !ok {
                return Err(Error::Draw { replicate: r });
            }
        }
    }
    let replicates = (0..b)
        .into_par_iter()
        .map(|r| match draws {
            Draws::Given(given) => plan.placebo(&given[r]),
            Draws::Seeded { seed, .. } => {
                let mut stream = Stream::seeded(replicate_seed(seed, r as u64));
                let mut permutation: Vec<usize> = (0..n0).collect();
                stream.shuffle(&mut permutation);
                plan.placebo(&permutation)
            }
        })
        // Gathered in replicate order before the first error is taken, so the
        // error reported does not depend on the number of threads.
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    Ok(collect(
        plan,
        &replicates,
        |v| replicate_se(v.iter().copied()),
        None,
    ))
}

fn bootstrap(plan: &Plan<'_>, draws: Draws<'_>) -> Result<Variance, Error> {
    if plan.treated_total() == 1 {
        return Ok(undefined(plan, SynthUndefined::OneTreatedUnit));
    }
    let b = replications(draws)?;
    let n = plan.data.units;
    let cohort_of = plan.cohort_of();
    let positions = plan.control_positions();
    let replicates = (0..b)
        .into_par_iter()
        .map(|r| match draws {
            Draws::Given(given) => {
                let draw = &given[r];
                if draw.len() != n || draw.iter().any(|&u| u >= n) {
                    return Err(Error::Draw { replicate: r });
                }
                plan.bootstrap(draw, &cohort_of, &positions)
                    .unwrap_or(Err(Error::Draw { replicate: r }))
            }
            Draws::Seeded { seed, .. } => {
                let mut stream = Stream::seeded(replicate_seed(seed, r as u64));
                let mut draw = vec![0; n];
                loop {
                    for unit in &mut draw {
                        *unit = stream.below(n);
                    }
                    if let Some(result) = plan.bootstrap(&draw, &cohort_of, &positions) {
                        return result;
                    }
                }
            }
        })
        // Gathered in replicate order before the first error is taken, so the
        // error reported does not depend on the number of threads.
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    Ok(collect(
        plan,
        &replicates,
        |v| replicate_se(v.iter().copied()),
        None,
    ))
}

/// Per cohort, `h_j = sum_t c_t Y_jt` and the curve terms `Y_j,T0+s -
/// sum_t lambda_t Y_jt` of every unit of the panel, for the closed-form
/// jackknife.
struct Contrasts {
    h: Vec<f64>,
    curve: Vec<Vec<f64>>,
}

fn contrasts(plan: &Plan<'_>, cohort: &PlanCohort) -> Contrasts {
    let periods = plan.data.periods;
    let pre = cohort.pre;
    let share = 1.0 / (periods - pre) as f64;
    let mut h = vec![0.0; plan.data.units];
    let mut curve = vec![Vec::new(); plan.data.units];
    for unit in 0..plan.data.units {
        let row = plan.data.outcome_row(unit);
        let baseline: f64 = row[..pre]
            .iter()
            .zip(&cohort.lambda)
            .map(|(&y, &l)| y * l)
            .sum();
        let mut value = 0.0;
        for (t, &y) in row.iter().enumerate() {
            value += y * if t < pre { -cohort.lambda[t] } else { share };
        }
        h[unit] = value;
        curve[unit] = row[pre..].iter().map(|&y| y - baseline).collect();
    }
    Contrasts { h, curve }
}

fn jackknife(
    plan: &Plan<'_>,
    clusters: Option<&epistat_inference::Clusters>,
) -> Result<Variance, Error> {
    let n = plan.data.units;
    let groups: Vec<Vec<usize>> = match clusters {
        None => (0..n).map(|u| vec![u]).collect(),
        Some(c) => {
            if c.len() != n {
                return Err(epistat_core::InputError::Length {
                    what: "clusters",
                    expected: n,
                    found: c.len(),
                }
                .into());
            }
            let mut groups = vec![Vec::new(); c.count()];
            for (unit, &g) in c.of().iter().enumerate() {
                groups[g].push(unit);
            }
            groups
        }
    };
    if clusters.is_none() {
        if plan.cohorts.iter().any(|c| c.treated.len() == 1) {
            return Ok(undefined(plan, SynthUndefined::OneTreatedUnit));
        }
        if plan
            .cohorts
            .iter()
            .any(|c| c.omega.iter().filter(|&&w| w != 0.0).count() == 1)
        {
            return Ok(undefined(plan, SynthUndefined::OneWeightedControl));
        }
    }
    let cohort_of = plan.cohort_of();
    let positions = plan.control_positions();
    let length = plan.curve_length();
    // Every leave-out set must leave each cohort a treated unit and a control
    // with positive weight.
    for group in &groups {
        for (a, cohort) in plan.cohorts.iter().enumerate() {
            let removed = group.iter().filter(|&&u| cohort_of[u] == Some(a)).count();
            if removed == cohort.treated.len() {
                return Ok(undefined(plan, SynthUndefined::OneTreatedUnit));
            }
            let weighted = |u: &usize| positions[*u].is_some_and(|p| cohort.omega[p] != 0.0);
            let positive = cohort.omega.iter().filter(|&&w| w != 0.0).count();
            if group.iter().filter(|u| weighted(u)).count() == positive {
                return Ok(undefined(plan, SynthUndefined::OneWeightedControl));
            }
        }
    }
    let replicates: Vec<(f64, Vec<f64>)> = if plan.data.count == 0 {
        let parts: Vec<Contrasts> = plan.cohorts.iter().map(|c| contrasts(plan, c)).collect();
        groups
            .par_iter()
            .map(|group| {
                let values: Vec<CohortValue> = plan
                    .cohorts
                    .iter()
                    .zip(&parts)
                    .enumerate()
                    .map(|(a, (cohort, part))| {
                        closed_form(plan, cohort, part, group, a, &cohort_of, &positions)
                    })
                    .collect();
                aggregate(&values, length)
            })
            .collect()
    } else {
        groups
            .par_iter()
            .map(|group| {
                let controls: Vec<usize> = plan
                    .controls
                    .iter()
                    .copied()
                    .filter(|u| !group.contains(u))
                    .collect();
                let mut values = Vec::with_capacity(plan.cohorts.len());
                for cohort in &plan.cohorts {
                    let treated: Vec<usize> = cohort
                        .treated
                        .iter()
                        .copied()
                        .filter(|u| !group.contains(u))
                        .collect();
                    let omega = sum_normalize(
                        controls
                            .iter()
                            .map(|&u| cohort.omega[positions[u].expect("a control")])
                            .collect(),
                    );
                    let opts = Opts {
                        update_omega: false,
                        update_lambda: false,
                        ..cohort.opts
                    };
                    let start = Start {
                        lambda: Some(cohort.lambda.clone()),
                        omega: Some(omega),
                    };
                    let fitted =
                        fit::fit(&plan.data, &controls, &treated, cohort.pre, &opts, start)?;
                    values.push(CohortValue {
                        treated: treated.len(),
                        post: plan.data.periods - cohort.pre,
                        estimate: fitted.estimate,
                        curve: fitted.curve,
                    });
                }
                Ok(aggregate(&values, length))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>, Error>>()?
    };
    let influence = clusters.is_none().then(|| {
        let estimates: Vec<f64> = replicates.iter().map(|r| r.0).collect();
        let mean = fit::r_mean(estimates.iter().copied());
        let scale = ((n * (n - 1)) as f64).sqrt();
        estimates.iter().map(|&u| scale * (mean - u)).collect()
    });
    Ok(collect(plan, &replicates, jackknife_se, influence))
}

/// The leave-out estimate of one cohort with fixed weights, without
/// covariates: the treated mean over the treated units kept, less the unit
/// weights of the controls kept, rescaled to sum to one.
fn closed_form(
    plan: &Plan<'_>,
    cohort: &PlanCohort,
    part: &Contrasts,
    group: &[usize],
    a: usize,
    cohort_of: &[Option<usize>],
    positions: &[Option<usize>],
) -> CohortValue {
    let post = plan.data.periods - cohort.pre;
    let removed_treated: Vec<usize> = group
        .iter()
        .copied()
        .filter(|&u| cohort_of[u] == Some(a))
        .collect();
    let removed_controls: Vec<(usize, usize)> = group
        .iter()
        .filter_map(|&u| positions[u].map(|p| (u, p)))
        .collect();
    let n1 = (cohort.treated.len() - removed_treated.len()) as f64;
    let total_weight: f64 = cohort.omega.iter().sum();
    let removed_weight: f64 = removed_controls.iter().map(|&(_, p)| cohort.omega[p]).sum();
    let kept_weight = total_weight - removed_weight;
    let treated_sum = |values: &dyn Fn(usize) -> f64| -> f64 {
        let all: f64 = cohort.treated.iter().map(|&u| values(u)).sum();
        let gone: f64 = removed_treated.iter().map(|&u| values(u)).sum();
        (all - gone) / n1
    };
    let control_sum = |values: &dyn Fn(usize) -> f64| -> f64 {
        let all: f64 = plan
            .controls
            .iter()
            .zip(&cohort.omega)
            .map(|(&u, &w)| w * values(u))
            .sum();
        let gone: f64 = removed_controls
            .iter()
            .map(|&(u, p)| cohort.omega[p] * values(u))
            .sum();
        (all - gone) / kept_weight
    };
    let estimate = treated_sum(&|u| part.h[u]) - control_sum(&|u| part.h[u]);
    let curve = (0..post)
        .map(|s| treated_sum(&|u| part.curve[u][s]) - control_sum(&|u| part.curve[u][s]))
        .collect();
    CohortValue {
        treated: cohort.treated.len() - removed_treated.len(),
        post,
        estimate,
        curve,
    }
}
