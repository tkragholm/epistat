//! Checks of the input, the observed histories and the observed risk.

use super::history::{History, check_terms};
use super::{
    Action, Covariate, CovariateKind, GFormulaOptions, Intervention, Outcome, PersonPeriods,
    Restriction,
};
use epistat_core::{InputError, check};

use crate::error::{Error, Model, Result, domain, domain_at};

/// The rule of one variable under one strategy, with the periods it applies at.
pub(super) struct ActiveRule<'a> {
    pub(super) action: &'a Action,
    pub(super) at: Vec<bool>,
}

/// The observed data, checked and laid out by person.
pub(super) struct Prepared<'a> {
    /// The first row of each person, and the number of rows at the end.
    pub(super) start: Vec<usize>,
    /// `rows x width` time-varying values.
    pub(super) values: Vec<f64>,
    /// Their running sums within each person.
    pub(super) sums: Vec<f64>,
    /// `persons x baseline` baseline values.
    pub(super) baseline: Vec<f64>,
    pub(super) width: usize,
    pub(super) n_baseline: usize,
    /// The number of periods simulated.
    pub(super) periods: usize,
    /// The last period in the data.
    pub(super) last_period: usize,
    /// Strategy `s`, variable `v`: its rule, if any. Strategy 0 is the natural
    /// course.
    pub(super) rules: Vec<Vec<Option<ActiveRule<'a>>>>,
    /// The range of each normal covariate in the data.
    pub(super) ranges: Vec<Option<(f64, f64)>>,
    pub(super) observed: Vec<Option<f64>>,
    pub(super) reference: usize,
    pub(super) baseline_lags: bool,
    pub(super) truncate: bool,
    pub(super) end_of_follow_up: bool,
}

impl Prepared<'_> {
    pub(super) fn persons(&self) -> usize {
        self.start.len() - 1
    }

    pub(super) fn rows_of(&self, p: usize) -> std::ops::Range<usize> {
        self.start[p]..self.start[p + 1]
    }

    /// The observed history of person `p` at period `k`, with `visible`
    /// variables of period `k` visible.
    pub(super) fn history(&self, p: usize, k: usize, visible: usize) -> History<'_> {
        let rows = self.rows_of(p);
        let w = self.width;
        History {
            values: &self.values[rows.start * w..rows.end * w],
            sums: &self.sums[rows.start * w..rows.end * w],
            baseline: &self.baseline[p * self.n_baseline..(p + 1) * self.n_baseline],
            width: w,
            period: k,
            visible,
            baseline_lags: self.baseline_lags,
        }
    }

    pub(super) fn reported_periods(&self) -> Vec<usize> {
        if self.end_of_follow_up {
            vec![self.periods - 1]
        } else {
            (0..self.periods).collect()
        }
    }
}

#[expect(clippy::float_cmp, reason = "a binary value is exactly 0 or 1")]
fn is_binary(x: f64) -> bool {
    x == 0.0 || x == 1.0
}

fn check_length(what: &'static str, expected: usize, found: usize) -> Result<()> {
    Ok(check::same_length(what, expected, found)?)
}

pub(super) fn prepare<'a>(
    data: &PersonPeriods<'_>,
    covariates: &[Covariate],
    outcome: &Outcome<'_>,
    interventions: &'a [Intervention],
    options: &GFormulaOptions,
) -> Result<Prepared<'a>> {
    let id = data.key.unit();
    let rows = id.len();
    let width = data.time_varying.len();
    let n_baseline = data.baseline.len();
    for column in data.baseline {
        check_length("baseline", rows, column.len())?;
    }
    for column in data.time_varying {
        check_length("time_varying", rows, column.len())?;
    }
    check_length("covariates", width, covariates.len())?;
    if options.simulated == Some(0) {
        return Err(domain("simulated", 0.0, "at least 1"));
    }
    if options.reference > interventions.len() {
        return Err(Error::invalid(
            "reference",
            "names no strategy: it is 0 or the number of an intervention",
        ));
    }

    // Persons: the key sorts the rows by person and period, and each person's
    // periods must be 0, 1, 2, ...
    let mut start = vec![0];
    let mut period = Vec::with_capacity(rows);
    for (r, &k) in data.key.period().iter().enumerate() {
        let first = r == 0 || id[r] != id[r - 1];
        if first && r > 0 {
            start.push(r);
        }
        let expected = if first { 0 } else { period[r - 1] + 1 };
        if usize::try_from(k).ok() != Some(expected) {
            return Err(Error::invalid_at(
                "row",
                r,
                if first {
                    "the first row of a person is not at period 0"
                } else {
                    "the periods of a person must be consecutive"
                },
            ));
        }
        period.push(expected);
    }
    start.push(rows);
    let persons = start.len() - 1;
    let last_period = *period.iter().max().unwrap_or(&0);
    let periods = options.periods.unwrap_or(last_period + 1);
    if periods == 0 || periods > last_period + 1 {
        #[expect(clippy::cast_precision_loss, reason = "the value is reported only")]
        return Err(domain(
            "periods",
            periods as f64,
            "1 to the number of periods in the data",
        ));
    }

    // Baseline variables, constant within a person.
    let mut baseline = Vec::with_capacity(persons * n_baseline);
    for p in 0..persons {
        for (j, column) in data.baseline.iter().enumerate() {
            let value = column[start[p]];
            if !value.is_finite() {
                return Err(InputError::NotFinite {
                    what: "baseline",
                    row: start[p],
                    column: Some(j),
                }
                .into());
            }
            if let Some(r) =
                (start[p]..start[p + 1]).find(|&r| column[r].to_bits() != value.to_bits())
            {
                return Err(InputError::NotConstant {
                    what: "baseline",
                    unit: id[r],
                }
                .into());
            }
            baseline.push(value);
        }
    }

    // Time-varying values and their running sums.
    let mut values = vec![0.0; rows * width];
    let mut sums = vec![0.0; rows * width];
    for (j, column) in data.time_varying.iter().enumerate() {
        let binary = covariates[j].kind == CovariateKind::Binary;
        if let CovariateKind::Categorical(levels) = covariates[j].kind
            && levels < 2
        {
            return Err(Error::invalid_at(
                "variable",
                j,
                "a categorical covariate needs at least two classes",
            ));
        }
        for r in 0..rows {
            let x = column[r];
            if !x.is_finite() {
                return Err(InputError::NotFinite {
                    what: "time_varying",
                    row: r,
                    column: Some(j),
                }
                .into());
            }
            if binary && !is_binary(x) {
                return Err(domain_at("time_varying", x, "{0, 1}", r));
            }
            if let CovariateKind::Categorical(levels) = covariates[j].kind
                && !(x >= 0.0 && x.fract() == 0.0 && x < levels as f64)
            {
                return Err(domain_at(
                    "time_varying",
                    x,
                    "the classes 0, 1, ... of the variable",
                    r,
                ));
            }
            values[r * width + j] = x;
            let before = if period[r] == 0 {
                0.0
            } else {
                sums[(r - 1) * width + j]
            };
            sums[r * width + j] = before + x;
        }
    }

    // The models' terms.
    for (j, c) in covariates.iter().enumerate() {
        check_terms(&c.terms, n_baseline, width, j, Model::Variable(j))?;
        if matches!(c.restriction, Some(Restriction::Absorbing)) && c.kind != CovariateKind::Binary
        {
            return Err(Error::invalid_at(
                "variable",
                j,
                "only a binary covariate can be absorbing",
            ));
        }
        if let Some(Restriction::When {
            otherwise: super::Otherwise::Value(v),
            ..
        }) = &c.restriction
            && !v.is_finite()
        {
            return Err(Error::invalid_at(
                "variable",
                j,
                "the value outside the restriction is not finite",
            ));
        }
    }
    let (end_of_follow_up, observed) = match outcome {
        Outcome::Survival {
            event,
            terms,
            competing,
            censoring,
        } => {
            check_length("the outcome", rows, event.len())?;
            if let Some(c) = censoring {
                check_length("the censoring", rows, c.censored.len())?;
                match c.cutoff {
                    super::WeightCutoff::Quantile(q) if !(0.0..=1.0).contains(&q) => {
                        return Err(domain("weight cutoff quantile", q, "[0, 1]"));
                    }
                    super::WeightCutoff::Value(v) if !(v.is_finite() && v > 0.0) => {
                        return Err(domain("weight cutoff value", v, "(0, inf)"));
                    }
                    _ => {}
                }
                check_terms(&c.terms, n_baseline, width, width, Model::Censoring)?;
                for p in 0..persons {
                    for r in start[p]..start[p + 1] {
                        if !c.censored[r] {
                            continue;
                        }
                        if r + 1 != start[p + 1] {
                            return Err(Error::invalid_at(
                                "row",
                                r,
                                "a person has rows after censoring",
                            ));
                        }
                        if event[r] || competing.as_ref().is_some_and(|d| d.event[r]) {
                            return Err(Error::invalid_at(
                                "row",
                                r,
                                "a censored row has no observed event",
                            ));
                        }
                    }
                }
            }
            check_terms(terms, n_baseline, width, width, Model::Outcome)?;
            if let Some(c) = competing {
                check_length("the competing event", rows, c.event.len())?;
                check_terms(&c.terms, n_baseline, width, width, Model::Competing)?;
            }
            for p in 0..persons {
                for r in start[p]..start[p + 1] - 1 {
                    let d = competing.as_ref().is_some_and(|c| c.event[r]);
                    if event[r] || d {
                        return Err(Error::invalid_at(
                            "row",
                            r,
                            "a person has rows after an event",
                        ));
                    }
                }
            }
            let observed =
                observed_risk(&period, event, competing.as_ref().map(|c| c.event), periods);
            (false, observed)
        }
        Outcome::BinaryEnd { value, terms } | Outcome::ContinuousEnd { value, terms } => {
            check_length("the outcome", rows, value.len())?;
            check_terms(terms, n_baseline, width, width, Model::Outcome)?;
            if periods != last_period + 1 {
                return Err(Error::invalid(
                    "periods",
                    "an outcome at the end of follow-up needs every period simulated",
                ));
            }
            let binary = matches!(outcome, Outcome::BinaryEnd { .. });
            let mut sum = 0.0;
            let mut count = 0usize;
            for r in 0..rows {
                if period[r] != last_period {
                    continue;
                }
                let y = value[r];
                if y.is_nan() {
                    continue;
                }
                if !y.is_finite() {
                    return Err(InputError::NotFinite {
                        what: "outcome",
                        row: r,
                        column: None,
                    }
                    .into());
                }
                if binary && !is_binary(y) {
                    return Err(domain_at("outcome", y, "{0, 1}", r));
                }
                sum += y;
                count += 1;
            }
            if count == 0 {
                return Err(Error::invalid(
                    "outcome",
                    "no outcome is observed at the last period",
                ));
            }
            (true, vec![Some(sum / count as f64)])
        }
    };

    // The strategies: the natural course, then the interventions.
    let mut rules = vec![(0..width).map(|_| None).collect::<Vec<_>>()];
    for (i, intervention) in interventions.iter().enumerate() {
        let mut by_variable: Vec<Option<ActiveRule<'a>>> = (0..width).map(|_| None).collect();
        for rule in &intervention.rules {
            let bad = |rule| Err(Error::invalid_at("intervention", i + 1, rule));
            if rule.variable >= width {
                return bad("names a time-varying variable that does not exist");
            }
            if by_variable[rule.variable].is_some() {
                return bad("has two rules for one variable");
            }
            match &rule.action {
                Action::Set(v) if !v.is_finite() => {
                    return bad("sets a value that is not finite");
                }
                Action::Schedule(s)
                    if s.len() < periods || s[..periods].iter().any(|v| !v.is_finite()) =>
                {
                    return bad("the schedule needs a finite value for each period simulated");
                }
                Action::Threshold { lower, upper }
                    if lower.is_nan() || upper.is_nan() || lower > upper =>
                {
                    return bad("the threshold needs lower <= upper");
                }
                _ => {}
            }
            let at = match &rule.periods {
                None => vec![true; periods],
                Some(list) => {
                    let mut at = vec![false; periods];
                    for &k in list.iter().filter(|&&k| k < periods) {
                        at[k] = true;
                    }
                    at
                }
            };
            by_variable[rule.variable] = Some(ActiveRule {
                action: &rule.action,
                at,
            });
        }
        rules.push(by_variable);
    }

    let ranges = covariates
        .iter()
        .enumerate()
        .map(|(j, c)| {
            (c.kind == CovariateKind::Normal).then(|| {
                let column = data.time_varying[j];
                let lo = column.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (lo, hi)
            })
        })
        .collect();

    Ok(Prepared {
        start,
        values,
        sums,
        baseline,
        width,
        n_baseline,
        periods,
        last_period,
        rules,
        ranges,
        observed,
        reference: options.reference,
        baseline_lags: options.baseline_lags,
        truncate: options.truncate,
        end_of_follow_up,
    })
}

/// The nonparametric risk by period: the discrete-time product of the event
/// and competing-event proportions among the rows of each period.
fn observed_risk(
    period: &[usize],
    event: &[bool],
    competing: Option<&[bool]>,
    periods: usize,
) -> Vec<Option<f64>> {
    let mut rows = vec![0usize; periods];
    let mut free = vec![0usize; periods];
    let mut events = vec![0usize; periods];
    let mut deaths = vec![0usize; periods];
    for r in 0..period.len() {
        let k = period[r];
        if k >= periods {
            continue;
        }
        rows[k] += 1;
        let d = competing.is_some_and(|c| c[r]);
        if d {
            deaths[k] += 1;
        } else {
            free[k] += 1;
            events[k] += usize::from(event[r]);
        }
    }
    let mut out = Vec::with_capacity(periods);
    let (mut risk, mut survival) = (0.0, 1.0);
    let mut defined = true;
    for k in 0..periods {
        defined &= rows[k] > 0 && free[k] > 0;
        if !defined {
            out.push(None);
            continue;
        }
        let h = events[k] as f64 / free[k] as f64;
        let d = deaths[k] as f64 / rows[k] as f64;
        risk += h * (1.0 - d) * survival;
        survival *= (1.0 - h) * (1.0 - d);
        out.push(Some(risk));
    }
    out
}

/// The observed risk by period weighted by the inverse probability of
/// remaining uncensored (see the module documentation of the g-formula).
pub(super) fn weighted_observed_risk(
    prepared: &Prepared<'_>,
    event: &[bool],
    competing: Option<&[bool]>,
    censoring: &super::Censoring<'_>,
    model: &super::fit::Linear,
) -> Vec<Option<f64>> {
    let periods = prepared.periods;
    let w = prepared.width;
    // The weight of every row, 0 in a censored row.
    let rows = prepared.start[prepared.start.len() - 1];
    let mut weight = vec![0.0; rows];
    for p in 0..prepared.start.len() - 1 {
        let mut cumulative = 1.0;
        for (k, r) in prepared.rows_of(p).enumerate() {
            let h = prepared.history(p, k, w);
            cumulative /= 1.0 - model.mean(&censoring.terms, &h);
            weight[r] = if censoring.censored[r] {
                0.0
            } else {
                cumulative
            };
        }
    }
    let cut = match censoring.cutoff {
        super::WeightCutoff::None => None,
        super::WeightCutoff::Value(v) => Some(v),
        super::WeightCutoff::Quantile(q) => {
            let mut sorted = weight.clone();
            sorted.sort_by(f64::total_cmp);
            // The quantile was checked to lie in [0, 1] and the weights are
            // finite and not empty, so it exists.
            epistat_stats::quantile::type7(&sorted, q)
        }
    };
    if let Some(c) = cut {
        for v in &mut weight {
            *v = v.min(c);
        }
    }
    // Weighted sums over the rows of each period: of the rows without the
    // competing event and of their events, of all rows and of the competing
    // events.
    let mut free = vec![0.0; periods];
    let mut events = vec![0.0; periods];
    let mut all = vec![0.0; periods];
    let mut deaths = vec![0.0; periods];
    for p in 0..prepared.start.len() - 1 {
        for (k, r) in prepared.rows_of(p).enumerate() {
            if k >= periods || censoring.censored[r] {
                continue;
            }
            let wr = weight[r];
            all[k] += wr;
            if competing.is_some_and(|c| c[r]) {
                deaths[k] += wr;
            } else {
                free[k] += wr;
                if event[r] {
                    events[k] += wr;
                }
            }
        }
    }
    let mut out = Vec::with_capacity(periods);
    let (mut risk, mut survival) = (0.0, 1.0);
    let mut defined = true;
    for k in 0..periods {
        defined &= all[k] > 0.0 && free[k] > 0.0;
        if !defined {
            out.push(None);
            continue;
        }
        let h = events[k] / free[k];
        let d = deaths[k] / all[k];
        risk += h * (1.0 - d) * survival;
        survival *= (1.0 - h) * (1.0 - d);
        out.push(Some(risk));
    }
    out
}
