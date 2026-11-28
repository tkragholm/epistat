//! The Monte Carlo simulation of the strategies.

use epistat_rng::{Stream, replicate_seed};
use rayon::prelude::*;

use super::fit::{Fitted, applies};
use super::history::History;
use super::prepare::Prepared;
use super::{Action, Covariate, CovariateKind, Otherwise, Outcome, Restriction, SimulatedCourse};
use crate::error::{Error, Result};

/// The persons of a block, whose moments are merged in block order.
const BLOCK: usize = 512;

/// The count, mean and sum of squared deviations of a set of values.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct Moments {
    n: f64,
    pub(super) mean: f64,
    m2: f64,
}

impl Moments {
    /// Welford's update.
    fn push(&mut self, x: f64) {
        self.n += 1.0;
        let d = x - self.mean;
        self.mean += d / self.n;
        self.m2 += d * (x - self.mean);
    }

    /// Chan, Golub and LeVeque's merge of two sets.
    fn merge(&mut self, other: &Self) {
        if other.n == 0.0 {
            return;
        }
        if self.n == 0.0 {
            *self = *other;
            return;
        }
        let n = self.n + other.n;
        let d = other.mean - self.mean;
        self.mean += d * other.n / n;
        self.m2 += other.m2 + d * d * self.n * other.n / n;
        self.n = n;
    }

    fn variance(&self) -> Option<f64> {
        (self.n >= 2.0).then(|| self.m2 / (self.n - 1.0))
    }

    /// The standard error of the mean.
    pub(super) fn standard_error(&self) -> Option<f64> {
        self.variance().map(|v| (v / self.n).sqrt())
    }
}

/// The Monte Carlo standard error of the ratio of the means of `a` and `b`,
/// given the moments of `a`, `b` and of `a - b`: that of the mean of
/// `a_i - R b_i`, over the mean of `b`.
pub(super) fn ratio_standard_error(a: &Moments, b: &Moments, d: &Moments) -> Option<f64> {
    let (va, vb, vd) = (a.variance()?, b.variance()?, d.variance()?);
    let q = a.mean / b.mean;
    let covariance = (va + vb - vd) / 2.0;
    let v = (va - 2.0 * q * covariance + q * q * vb).max(0.0);
    Some((v / a.n).sqrt() / b.mean.abs())
}

/// One run of the simulation: the moments of each strategy `s` at each
/// reported period `t`, of its values and of their differences from the
/// reference strategy.
pub(super) struct Run {
    /// The simulated courses of each strategy, when they are kept.
    pub(super) courses: Option<Vec<SimulatedCourse>>,
    pub(super) value: Vec<Vec<Moments>>,
    pub(super) difference: Vec<Vec<Moments>>,
    pub(super) intervened_persons: Vec<f64>,
    pub(super) intervened_person_periods: Vec<f64>,
}

impl Run {
    /// The mean of each strategy at each reported period.
    pub(super) fn means(&self) -> Vec<Vec<f64>> {
        self.value
            .iter()
            .map(|s| s.iter().map(|m| m.mean).collect())
            .collect()
    }
}

/// The observed persons the simulated persons start from: each person as
/// often as its count in `counts` (a bootstrap resample, or 1 each), or
/// `simulated` persons drawn with replacement from those.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the counts are small whole numbers"
)]
pub(super) fn sources(counts: &[f64], simulated: Option<usize>, seed: u64) -> Vec<usize> {
    let pool: Vec<usize> = counts
        .iter()
        .enumerate()
        .flat_map(|(p, &c)| std::iter::repeat_n(p, c as usize))
        .collect();
    match simulated {
        None => pool,
        Some(m) => {
            let mut stream = Stream::seeded(replicate_seed(seed, 1));
            (0..m).map(|_| pool[stream.below(pool.len())]).collect()
        }
    }
}

struct Block {
    courses: Option<Vec<SimulatedCourse>>,
    value: Vec<Moments>,
    difference: Vec<Moments>,
    persons: Vec<usize>,
    person_periods: Vec<usize>,
}

/// Simulates every strategy for the persons of `sources`, with simulated
/// person `j` drawing from `replicate_seed(seed, j + 2)`.
pub(super) fn run(
    prepared: &Prepared<'_>,
    covariates: &[Covariate],
    outcome: &Outcome<'_>,
    fitted: &Fitted,
    sources: &[usize],
    seed: u64,
    keep: bool,
) -> Result<Run> {
    let strategies = prepared.rules.len();
    let reported = prepared.reported_periods().len();
    let blocks = sources
        .par_chunks(BLOCK)
        .enumerate()
        .map(|(b, chunk)| {
            simulate_block(
                prepared,
                covariates,
                outcome,
                fitted,
                chunk,
                b * BLOCK,
                seed,
                keep,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut value = vec![Moments::default(); strategies * reported];
    let mut difference = vec![Moments::default(); strategies * reported];
    let mut persons = vec![0usize; strategies];
    let mut person_periods = vec![0usize; strategies];
    for block in &blocks {
        for (m, o) in value.iter_mut().zip(&block.value) {
            m.merge(o);
        }
        for (m, o) in difference.iter_mut().zip(&block.difference) {
            m.merge(o);
        }
        for s in 0..strategies {
            persons[s] += block.persons[s];
            person_periods[s] += block.person_periods[s];
        }
    }
    let courses = keep.then(|| {
        let mut all = vec![SimulatedCourse::new(prepared.width); strategies];
        for block in &blocks {
            if let Some(c) = &block.courses {
                for (a, b) in all.iter_mut().zip(c) {
                    a.extend(b);
                }
            }
        }
        all
    });
    let n = sources.len() as f64;
    let by_strategy = |m: Vec<Moments>| m.chunks(reported).map(<[Moments]>::to_vec).collect();
    Ok(Run {
        courses,
        value: by_strategy(value),
        difference: by_strategy(difference),
        intervened_persons: persons.iter().map(|&c| c as f64 / n).collect(),
        intervened_person_periods: person_periods
            .iter()
            .map(|&c| c as f64 / (n * prepared.periods as f64))
            .collect(),
    })
}

fn simulate_block(
    prepared: &Prepared<'_>,
    covariates: &[Covariate],
    outcome: &Outcome<'_>,
    fitted: &Fitted,
    sources: &[usize],
    first: usize,
    seed: u64,
    keep: bool,
) -> Result<Block> {
    let strategies = prepared.rules.len();
    let reported = prepared.reported_periods().len();
    let cells = prepared.periods * prepared.width;
    let mut values = vec![0.0; cells];
    let mut sums = vec![0.0; cells];
    let mut person = vec![0.0; strategies * reported];
    let mut block = Block {
        courses: keep.then(|| vec![SimulatedCourse::new(prepared.width); strategies]),
        value: vec![Moments::default(); strategies * reported],
        difference: vec![Moments::default(); strategies * reported],
        persons: vec![0; strategies],
        person_periods: vec![0; strategies],
    };
    let reference = prepared.reference;
    for (i, &source) in sources.iter().enumerate() {
        let person_seed = replicate_seed(seed, (first + i) as u64 + 2);
        for s in 0..strategies {
            let mut stream = Stream::seeded(person_seed);
            let intervened = simulate_person(
                prepared,
                covariates,
                outcome,
                fitted,
                source,
                s,
                &mut stream,
                &mut values,
                &mut sums,
                &mut person[s * reported..(s + 1) * reported],
                block.courses.as_mut().map(|c| (&mut c[s], first + i)),
            )?;
            if intervened > 0 {
                block.persons[s] += 1;
                block.person_periods[s] += intervened;
            }
        }
        for s in 0..strategies {
            for t in 0..reported {
                let x = person[s * reported + t];
                block.value[s * reported + t].push(x);
                block.difference[s * reported + t].push(x - person[reference * reported + t]);
            }
        }
    }
    Ok(block)
}

/// Simulates one person under strategy `s`, writing its value at each reported
/// period to `out`, and returns the number of periods in which a rule changed
/// a value.
fn simulate_person(
    prepared: &Prepared<'_>,
    covariates: &[Covariate],
    outcome: &Outcome<'_>,
    fitted: &Fitted,
    source: usize,
    s: usize,
    stream: &mut Stream,
    values: &mut [f64],
    sums: &mut [f64],
    out: &mut [f64],
    mut course: Option<(&mut SimulatedCourse, usize)>,
) -> Result<usize> {
    let w = prepared.width;
    let nb = prepared.n_baseline;
    let baseline = &prepared.baseline[source * nb..(source + 1) * nb];
    let row0 = prepared.start[source];
    let periods = prepared.periods;
    let (mut risk, mut survival) = (0.0, 1.0);
    let mut intervened = 0;
    for k in 0..periods {
        let mut changed = false;
        for v in 0..w {
            let h = History {
                values: &values[..(k + 1) * w],
                sums: &sums[..(k + 1) * w],
                baseline,
                width: w,
                period: k,
                visible: v,
                baseline_lags: prepared.baseline_lags,
            };
            let mut x = if k == 0 {
                prepared.values[row0 * w + v]
            } else {
                let covariate = &covariates[v];
                let model = fitted.covariates[v]
                    .as_ref()
                    .expect("covariate models are fitted when more than one period is simulated");
                let draw = match covariate.kind {
                    CovariateKind::Binary | CovariateKind::Categorical(_) => stream.uniform(),
                    CovariateKind::Normal => stream.normal(),
                };
                if applies(covariate.restriction.as_ref(), v, &h) {
                    match covariate.kind {
                        CovariateKind::Categorical(_) => model.class(&covariate.terms, &h, draw),
                        CovariateKind::Binary => {
                            f64::from(u8::from(draw < model.mean(&covariate.terms, &h)))
                        }
                        CovariateKind::Normal => {
                            let mean = model.mean(&covariate.terms, &h);
                            let sd = model.model.rmse.unwrap_or(0.0);
                            let x = mean + sd * draw;
                            match prepared.ranges[v] {
                                Some((lo, hi)) if prepared.truncate => x.max(lo).min(hi),
                                _ => x,
                            }
                        }
                    }
                } else {
                    match &covariate.restriction {
                        Some(Restriction::When {
                            otherwise: Otherwise::Value(c),
                            ..
                        }) => *c,
                        _ => values[(k - 1) * w + v],
                    }
                }
            };
            if let Some(rule) = &prepared.rules[s][v]
                && rule.at[k]
            {
                let set = match rule.action {
                    Action::Set(c) => *c,
                    Action::Schedule(schedule) => schedule[k],
                    Action::Threshold { lower, upper } => x.max(*lower).min(*upper),
                    Action::Dynamic(f) => f(&h, x),
                };
                if !set.is_finite() {
                    return Err(Error::invalid_at(
                        "intervention",
                        s,
                        "a rule gave a value that is not finite",
                    ));
                }
                changed |= set.to_bits() != x.to_bits();
                x = set;
            }
            values[k * w + v] = x;
            sums[k * w + v] = if k == 0 { x } else { sums[(k - 1) * w + v] + x };
        }
        intervened += usize::from(changed);
        let h = History {
            values: &values[..(k + 1) * w],
            sums: &sums[..(k + 1) * w],
            baseline,
            width: w,
            period: k,
            visible: w,
            baseline_lags: prepared.baseline_lags,
        };
        let (py, pd) = match outcome {
            Outcome::Survival {
                terms, competing, ..
            } => {
                let pd = match (competing, &fitted.competing) {
                    (Some(c), Some(model)) => model.mean(&c.terms, &h),
                    _ => 0.0,
                };
                let py = fitted.outcome.mean(terms, &h);
                risk += py * (1.0 - pd) * survival;
                survival *= (1.0 - py) * (1.0 - pd);
                out[k] = risk;
                (py, pd)
            }
            Outcome::BinaryEnd { terms, .. } | Outcome::ContinuousEnd { terms, .. } => {
                if k + 1 == periods {
                    out[0] = fitted.outcome.mean(terms, &h);
                    risk = out[0];
                    (out[0], f64::NAN)
                } else {
                    risk = f64::NAN;
                    (f64::NAN, f64::NAN)
                }
            }
        };
        if let Some((c, person)) = course.as_mut() {
            c.person.push(*person);
            c.source.push(source);
            c.period.push(i64::try_from(k).unwrap_or(i64::MAX));
            for (v, column) in c.values.iter_mut().enumerate() {
                column.push(values[k * w + v]);
            }
            let defined = |x: f64| (!x.is_nan()).then_some(x);
            c.outcome_probability.push(defined(py));
            c.competing_probability.push(defined(pd));
            c.value.push(defined(risk));
        }
    }
    Ok(intervened)
}
