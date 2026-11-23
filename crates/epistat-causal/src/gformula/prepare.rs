//! Checks of the input, the observed histories and the observed risk.

use super::history::{History, check_terms};
use super::{
    Action, Covariate, CovariateKind, GFormulaOptions, Intervention, Outcome, PersonPeriods,
    Restriction,
};
use crate::error::{Error, Result};

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

fn check_length(what: &str, expected: usize, found: usize) -> Result<()> {
    if expected == found {
        Ok(())
    } else {
        Err(Error::input(format!(
            "{what} has {found} entries where {expected} were expected"
        )))
    }
}

pub(super) fn prepare<'a>(
    data: &PersonPeriods<'_>,
    covariates: &[Covariate],
    outcome: &Outcome<'_>,
    interventions: &'a [Intervention],
    options: &GFormulaOptions,
) -> Result<Prepared<'a>> {
    let rows = data.id.len();
    let width = data.time_varying.len();
    let n_baseline = data.baseline.len();
    if rows == 0 {
        return Err(Error::input("the data have no rows"));
    }
    check_length("period", rows, data.period.len())?;
    for (j, column) in data.baseline.iter().enumerate() {
        check_length(&format!("baseline variable {j}"), rows, column.len())?;
    }
    for (j, column) in data.time_varying.iter().enumerate() {
        check_length(&format!("time-varying variable {j}"), rows, column.len())?;
    }
    check_length("covariates", width, covariates.len())?;
    if !(options.alpha > 0.0 && options.alpha < 1.0) {
        return Err(Error::input("alpha must lie in (0, 1)"));
    }
    if options.simulated == Some(0) {
        return Err(Error::input("at least one person must be simulated"));
    }
    if options.reference > interventions.len() {
        return Err(Error::input(format!(
            "reference {} names no strategy; there are {} interventions",
            options.reference,
            interventions.len()
        )));
    }

    // Persons: contiguous rows at periods 0, 1, 2, ...
    let mut start = vec![0];
    for r in 0..rows {
        let first = r == 0 || data.id[r] != data.id[r - 1];
        if first {
            if r > 0 {
                start.push(r);
            }
            if data.period[r] != 0 {
                return Err(Error::input(format!(
                    "row {r}: the first row of person {} is at period {}, not 0",
                    data.id[r], data.period[r]
                )));
            }
        } else if data.period[r] != data.period[r - 1] + 1 {
            return Err(Error::input(format!(
                "row {r}: person {} goes from period {} to {}; the rows of a person \
                 must be contiguous and at consecutive periods",
                data.id[r],
                data.period[r - 1],
                data.period[r]
            )));
        }
    }
    start.push(rows);
    let persons = start.len() - 1;
    {
        let mut seen = std::collections::HashSet::with_capacity(persons);
        for &first in &start[..persons] {
            if !seen.insert(data.id[first]) {
                return Err(Error::input(format!(
                    "the rows of person {} are not contiguous",
                    data.id[first]
                )));
            }
        }
    }
    let last_period = *data.period.iter().max().unwrap_or(&0);
    let periods = options.periods.unwrap_or(last_period + 1);
    if periods == 0 || periods > last_period + 1 {
        return Err(Error::input(format!(
            "{periods} periods asked for; the data have {}",
            last_period + 1
        )));
    }

    // Baseline variables, constant within a person.
    let mut baseline = Vec::with_capacity(persons * n_baseline);
    for p in 0..persons {
        for (j, column) in data.baseline.iter().enumerate() {
            let value = column[start[p]];
            if !value.is_finite() {
                return Err(Error::input(format!(
                    "row {}: baseline variable {j} is not finite",
                    start[p]
                )));
            }
            if let Some(r) =
                (start[p]..start[p + 1]).find(|&r| column[r].to_bits() != value.to_bits())
            {
                return Err(Error::input(format!(
                    "row {r}: baseline variable {j} changes within person {}",
                    data.id[r]
                )));
            }
            baseline.push(value);
        }
    }

    // Time-varying values and their running sums.
    let mut values = vec![0.0; rows * width];
    let mut sums = vec![0.0; rows * width];
    for (j, column) in data.time_varying.iter().enumerate() {
        let binary = covariates[j].kind == CovariateKind::Binary;
        for r in 0..rows {
            let x = column[r];
            if !x.is_finite() {
                return Err(Error::input(format!(
                    "row {r}: time-varying variable {j} is not finite"
                )));
            }
            if binary && !is_binary(x) {
                return Err(Error::input(format!(
                    "row {r}: binary variable {j} is {x}, not 0 or 1"
                )));
            }
            values[r * width + j] = x;
            let before = if data.period[r] == 0 {
                0.0
            } else {
                sums[(r - 1) * width + j]
            };
            sums[r * width + j] = before + x;
        }
    }

    // The models' terms.
    for (j, c) in covariates.iter().enumerate() {
        check_terms(
            &c.terms,
            n_baseline,
            width,
            j,
            &format!("the model of variable {j}"),
        )?;
        if matches!(c.restriction, Some(Restriction::Absorbing)) && c.kind != CovariateKind::Binary
        {
            return Err(Error::input(format!(
                "variable {j}: only a binary covariate can be absorbing"
            )));
        }
        if let Some(Restriction::When {
            otherwise: super::Otherwise::Value(v),
            ..
        }) = &c.restriction
            && !v.is_finite()
        {
            return Err(Error::input(format!(
                "variable {j}: the value outside the restriction is not finite"
            )));
        }
    }
    let (end_of_follow_up, observed) = match outcome {
        Outcome::Survival {
            event,
            terms,
            competing,
        } => {
            check_length("the outcome", rows, event.len())?;
            check_terms(terms, n_baseline, width, width, "the outcome model")?;
            if let Some(c) = competing {
                check_length("the competing event", rows, c.event.len())?;
                check_terms(
                    &c.terms,
                    n_baseline,
                    width,
                    width,
                    "the competing-event model",
                )?;
            }
            for p in 0..persons {
                for r in start[p]..start[p + 1] - 1 {
                    let d = competing.as_ref().is_some_and(|c| c.event[r]);
                    if event[r] || d {
                        return Err(Error::input(format!(
                            "row {r}: person {} has rows after an event",
                            data.id[r]
                        )));
                    }
                }
            }
            let observed = observed_risk(
                data.period,
                event,
                competing.as_ref().map(|c| c.event),
                periods,
            );
            (false, observed)
        }
        Outcome::BinaryEnd { value, terms } | Outcome::ContinuousEnd { value, terms } => {
            check_length("the outcome", rows, value.len())?;
            check_terms(terms, n_baseline, width, width, "the outcome model")?;
            if periods != last_period + 1 {
                return Err(Error::input(
                    "an outcome at the end of follow-up needs every period simulated",
                ));
            }
            let binary = matches!(outcome, Outcome::BinaryEnd { .. });
            let mut sum = 0.0;
            let mut count = 0usize;
            for r in 0..rows {
                if data.period[r] != last_period {
                    continue;
                }
                let y = value[r];
                if y.is_nan() {
                    continue;
                }
                if !y.is_finite() || (binary && !is_binary(y)) {
                    return Err(Error::input(format!(
                        "row {r}: the outcome is {y}, which the model cannot take"
                    )));
                }
                sum += y;
                count += 1;
            }
            if count == 0 {
                return Err(Error::input("no outcome is observed at the last period"));
            }
            (true, vec![Some(sum / count as f64)])
        }
    };

    // The strategies: the natural course, then the interventions.
    let mut rules = vec![(0..width).map(|_| None).collect::<Vec<_>>()];
    for (i, intervention) in interventions.iter().enumerate() {
        let mut by_variable: Vec<Option<ActiveRule<'a>>> = (0..width).map(|_| None).collect();
        for rule in &intervention.rules {
            let what = format!("intervention {}", i + 1);
            if rule.variable >= width {
                return Err(Error::input(format!(
                    "{what}: time-varying variable {} does not exist",
                    rule.variable
                )));
            }
            if by_variable[rule.variable].is_some() {
                return Err(Error::input(format!(
                    "{what}: two rules for variable {}",
                    rule.variable
                )));
            }
            match &rule.action {
                Action::Set(v) if !v.is_finite() => {
                    return Err(Error::input(format!("{what}: the value is not finite")));
                }
                Action::Schedule(s)
                    if s.len() < periods || s[..periods].iter().any(|v| !v.is_finite()) =>
                {
                    return Err(Error::input(format!(
                        "{what}: the schedule needs a finite value for each of {periods} periods"
                    )));
                }
                Action::Threshold { lower, upper }
                    if lower.is_nan() || upper.is_nan() || lower > upper =>
                {
                    return Err(Error::input(format!(
                        "{what}: the threshold needs lower <= upper"
                    )));
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
