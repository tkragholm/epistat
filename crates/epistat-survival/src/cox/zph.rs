//! The test of proportional hazards of a Cox fit, as `cox.zph` of `survival`
//! 3.x.
//!
//! [`zph`] tests, for each term of the model and for the model as a whole,
//! whether the coefficients change with a transform `g` of time. It is the
//! score test of Grambsch and Therneau \[1\] and not the correlation test of
//! earlier versions of `cox.zph`: for each term the model is extended by the
//! term times `g(t)`, and the score of the added coefficients at zero, with the
//! information of the extended model at the estimate of the original, gives a
//! chi-square statistic. It returns the scaled Schoenfeld residuals against the
//! transformed times, which are what the plot of `cox.zph` shows.
//!
//! # The test
//!
//! Let `x_i` be the covariates of an event row, `xbar(t)` the mean over the risk
//! set at its time, as for the Schoenfeld residual of [`residuals`](mod@super::residuals),
//! and `g(t)` the transformed time, centred at its mean over the event rows. The
//! score for the coefficient of `g(t) x` is
//!
//! ```text
//! u = sum over events i of g(t_i) w_i (x_i - xbar(t_i)).
//! ```
//!
//! With `V(t)` the weighted covariance of `x` over the risk set at an event
//! time, summed over the terms of Breslow's or Efron's likelihood as in
//! [`fit`](super::fit), the information of the extended model at its null is the
//! `2p` by `2p` matrix with blocks
//!
//! ```text
//! [ sum V(t)      sum g V(t)   ]
//! [ sum g V(t)    sum g^2 V(t) ].
//! ```
//!
//! The statistic of a term with columns `J` is `u_J' (I^-1)_JJ u_J`, the
//! quadratic form of the score on the columns of `J` in the inverse of the
//! information of the model with the columns of all terms, with `|J|` degrees
//! of freedom. The statistic of the model is the same over every column, with
//! `p` degrees of freedom. With `singledf`, a term of several columns is tested
//! as the one linear predictor of its columns: with `b` its coefficients,
//! `(b' u_J)^2 (b' (I^-1)_JJ b)` on one degree of freedom, where `b` is the
//! estimate. Columns whose coefficient is `None` are left out of everything.
//!
//! # The transforms of time
//!
//! `Km` is one minus the left-continuous Kaplan-Meier estimate of survival at
//! the time, `Rank` the rank of the time with ties averaged, `Identity` the
//! time, and `Log` its logarithm. They are computed over the rows of positive
//! weight, ignoring weights, strata and clusters, as in `cox.zph`, and the
//! Kaplan-Meier risk set has delayed entry.
//!
//! # Scaled residuals
//!
//! `y` holds, for each event and each term, the Schoenfeld residual scaled as
//! `cox.zph` does: within a stratum, by the inverse of the information of the
//! model divided by the number of events of the stratum, and shifted by the
//! coefficient. For a term of several columns it is the residual of the linear
//! predictor of the term's columns, shifted by 1. A term whose columns do not
//! vary in a stratum has `None` in that stratum. `variance` is the inverse of
//! the matrix that scales them.
//!
//! # Terms
//!
//! A term is a set of columns of the design, as the columns of a factor.
//! `ZphOptions::groups` lists them. The default is one term for each column.
//!
//! # Differences from R
//!
//! Rows of weight zero are dropped, as in the fit, so they are not events of the
//! test, not in the transforms and not in the rows returned. A test that cannot
//! be computed, because the information of the extended model is singular or
//! the statistic is negative, is `None` where `cox.zph` fails or reports it. A
//! stratum with events in which no column varies has `None` in every term of
//! `y`. Penalised and frailty terms, a transform given as a function, and the
//! exact ties of `coxph` are not covered.
//!
//! # References
//!
//! 1. Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics based
//!    on weighted residuals. Biometrika 1994;81(3):515-26. doi:10.1093/biomet/81.3.515
//! 2. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000. Section 6.2.

use epistat_linalg::MatRef;
use epistat_stats::sum::compensated;

use super::design::{ChiSquareTest, CoxModelFit, CoxOptions};
use super::moments;
use super::residuals::{Prepared, prepare, schoenfeld_rows};
use super::sweep::Sums;
use crate::check::check_lengths;
use crate::error::{Error, Result};
use crate::step::Steps;
use crate::surv::SurvData;

/// The transform of time that the coefficients are tested against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZphTransform {
    /// One minus the left-continuous Kaplan-Meier survival, `transform = "km"`.
    #[default]
    Km,
    /// The rank of the time, ties averaged, `"rank"`.
    Rank,
    /// The time itself, `"identity"`.
    Identity,
    /// The natural logarithm of the time, `"log"`. Event times must be
    /// positive.
    Log,
}

/// What [`zph`] takes besides the data, the design and the fit.
#[derive(Debug, Clone, PartialEq)]
pub struct ZphOptions<'a> {
    /// The options of the fit, which must be those it was made with.
    pub cox: CoxOptions<'a>,
    /// The transform of time. `Km`, as in `cox.zph`.
    pub transform: ZphTransform,
    /// The terms, each a set of columns of the design. `None` makes one term of
    /// each column. A column may be in one term only.
    pub groups: Option<&'a [Vec<usize>]>,
    /// Test a term of several columns as its one linear predictor, `singledf`
    /// of `cox.zph`.
    pub singledf: bool,
}

impl Default for ZphOptions<'_> {
    fn default() -> Self {
        Self {
            cox: CoxOptions::default(),
            transform: ZphTransform::Km,
            groups: None,
            singledf: false,
        }
    }
}

/// The result of [`zph`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoxZph {
    /// The transform that was used.
    pub transform: ZphTransform,
    /// The columns of the design in each term, aliased columns left out and
    /// terms with no column left dropped.
    pub terms: Vec<Vec<usize>>,
    /// The test of each term, `None` when it cannot be computed.
    pub tests: Vec<Option<ChiSquareTest>>,
    /// The test of the model.
    pub global: Option<ChiSquareTest>,
    /// The time of each event, in order of stratum, time and then the row of the
    /// data.
    pub time: Vec<f64>,
    /// The transformed time of each event.
    pub x: Vec<f64>,
    /// The row of the data of each event.
    pub rows: Vec<u32>,
    /// The stratum code of each event as the caller gave it, empty without
    /// strata.
    pub stratum: Vec<u32>,
    /// The scaled Schoenfeld residuals, one vector per term with one value per
    /// event, `None` where the term plays no part in the stratum.
    pub y: Vec<Vec<Option<f64>>>,
    /// The inverse of the matrix that scaled `y`, `terms` by `terms`, row by
    /// row, with `None` in every entry when it cannot be inverted.
    pub variance: Vec<Option<f64>>,
    /// The rows of the data.
    pub n_rows: usize,
    /// The rows with a weight above zero.
    pub n_used: usize,
}

/// The transform of time at a time.
enum Transformer {
    Identity,
    Log,
    Rank(Vec<f64>),
    /// The Kaplan-Meier curve over the event times, read just before a time.
    Km(Steps),
}

impl Transformer {
    fn new(kind: ZphTransform, prep: &Prepared) -> Self {
        match kind {
            ZphTransform::Identity => Self::Identity,
            ZphTransform::Log => Self::Log,
            ZphTransform::Rank => {
                let mut stops: Vec<f64> = prep
                    .strata
                    .iter()
                    .flat_map(|s| s.stop.iter().copied())
                    .collect();
                stops.sort_unstable_by(f64::total_cmp);
                Self::Rank(stops)
            }
            ZphTransform::Km => {
                let mut stops: Vec<f64> = Vec::with_capacity(prep.n_used);
                let mut starts: Vec<f64> = Vec::new();
                let mut events: Vec<f64> = Vec::with_capacity(prep.n_events);
                for s in &prep.strata {
                    stops.extend_from_slice(&s.stop);
                    if s.start.iter().any(|&a| a != f64::NEG_INFINITY) {
                        starts.extend(s.start.iter().copied());
                    }
                    for (&t, &e) in s.stop.iter().zip(&s.is_event) {
                        if e {
                            events.push(t);
                        }
                    }
                }
                stops.sort_unstable_by(f64::total_cmp);
                starts.sort_unstable_by(f64::total_cmp);
                events.sort_unstable_by(f64::total_cmp);
                let mut times = Vec::new();
                let mut survival = Vec::new();
                let mut i = 0;
                while i < events.len() {
                    let t = events[i];
                    let mut j = i;
                    while j < events.len() && events[j].total_cmp(&t).is_eq() {
                        j += 1;
                    }
                    let deaths = (j - i) as f64;
                    let at_risk_stop = (stops.len() - stops.partition_point(|&s| s < t)) as f64;
                    let entered_later = (starts.len() - starts.partition_point(|&a| a < t)) as f64;
                    let at_risk = at_risk_stop - entered_later;
                    let last = survival.last().copied().unwrap_or(1.0);
                    times.push(t);
                    survival.push(last * ((at_risk - deaths) / at_risk));
                    i = j;
                }
                Self::Km(Steps {
                    time: times,
                    first: 1.0,
                    value: survival,
                })
            }
        }
    }

    fn at(&self, t: f64) -> f64 {
        match self {
            Self::Identity => t,
            Self::Log => t.ln(),
            Self::Rank(sorted) => {
                let below = sorted.partition_point(|&s| s < t) as f64;
                let through = sorted.partition_point(|&s| s <= t) as f64;
                (below + through + 1.0) / 2.0
            }
            Self::Km(curve) => 1.0 - curve.before(t),
        }
    }
}

/// The LU factorisation of a symmetric matrix after scaling its diagonal to 1,
/// with partial pivoting.
struct Factor {
    n: usize,
    lu: Vec<f64>,
    pivot: Vec<usize>,
    scale: Vec<f64>,
}

impl Factor {
    /// `None` when a diagonal entry is not positive or a pivot of the scaled
    /// matrix, whose diagonal is 1, is at most `n` times the machine epsilon.
    fn new(a: &[f64], n: usize) -> Option<Self> {
        let scale: Vec<f64> = (0..n)
            .map(|i| {
                let d = a[i * n + i];
                (d > 0.0 && d.is_finite()).then(|| 1.0 / d.sqrt())
            })
            .collect::<Option<_>>()?;
        let mut lu: Vec<f64> = (0..n * n)
            .map(|k| a[k] * scale[k / n] * scale[k % n])
            .collect();
        let mut pivot = Vec::with_capacity(n);
        for c in 0..n {
            let mut best = c;
            for r in c + 1..n {
                if lu[r * n + c].abs() > lu[best * n + c].abs() {
                    best = r;
                }
            }
            let size = lu[best * n + c].abs();
            if size.is_nan() || size <= n as f64 * f64::EPSILON {
                return None;
            }
            if best != c {
                for k in 0..n {
                    lu.swap(c * n + k, best * n + k);
                }
            }
            pivot.push(best);
            for r in c + 1..n {
                let f = lu[r * n + c] / lu[c * n + c];
                lu[r * n + c] = f;
                for k in c + 1..n {
                    lu[r * n + k] -= f * lu[c * n + k];
                }
            }
        }
        Some(Self {
            n,
            lu,
            pivot,
            scale,
        })
    }

    /// Solve `A x = b` in place.
    fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        for (v, s) in b.iter_mut().zip(&self.scale) {
            *v *= s;
        }
        for c in 0..n {
            b.swap(c, self.pivot[c]);
            for r in c + 1..n {
                b[r] -= self.lu[r * n + c] * b[c];
            }
        }
        for c in (0..n).rev() {
            for k in c + 1..n {
                b[c] -= self.lu[c * n + k] * b[k];
            }
            b[c] /= self.lu[c * n + c];
        }
        for (v, s) in b.iter_mut().zip(&self.scale) {
            *v *= s;
        }
    }
}

/// The terms as positions among the kept columns, and as columns of the design.
type Terms = (Vec<Vec<usize>>, Vec<Vec<usize>>);

/// The terms as sets of positions among the kept columns, and as the columns
/// of the design.
fn terms_of(groups: Option<&[Vec<usize>]>, keep: &[usize], columns: usize) -> Result<Terms> {
    let position = |j: usize| keep.iter().position(|&k| k == j);
    let groups: Vec<Vec<usize>> = groups.map_or_else(
        || keep.iter().map(|&j| vec![j]).collect(),
        <[Vec<usize>]>::to_vec,
    );
    let mut seen = vec![false; columns];
    let mut by_position = Vec::new();
    let mut by_column = Vec::new();
    for group in &groups {
        if group.is_empty() {
            return Err(Error::invalid("groups: a term has no column"));
        }
        for &j in group {
            if j >= columns {
                return Err(Error::invalid(format!(
                    "groups: column {j} is outside the {columns} columns of x"
                )));
            }
            if std::mem::replace(&mut seen[j], true) {
                return Err(Error::invalid(format!(
                    "groups: column {j} is in more than one term, or twice in one"
                )));
            }
        }
        let kept: Vec<usize> = group
            .iter()
            .copied()
            .filter(|&j| position(j).is_some())
            .collect();
        if !kept.is_empty() {
            by_position.push(kept.iter().filter_map(|&j| position(j)).collect());
            by_column.push(kept);
        }
    }
    Ok((by_position, by_column))
}

/// The test of proportional hazards of a Cox fit, as `cox.zph`.
///
/// `data`, `x` and `opts.cox` are those the fit was made with, and `fit` is its
/// result. See the module documentation for the test, the transforms and the
/// scaled residuals.
///
/// # Errors
///
/// An input error for the invalid input of [`fit`](super::fit), for a fit
/// without a coefficient or without an event, for ties other than Breslow and
/// Efron, for `groups` that name a column outside the design, name a column
/// twice or hold an empty term, for the `Log` transform with an event time that
/// is not positive, and for coefficients at which a risk score or the
/// denominator of a term of the partial likelihood is not finite and positive.
pub fn zph(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    fit: &CoxModelFit,
    opts: &ZphOptions<'_>,
) -> Result<CoxZph> {
    let columns = x.ncols();
    check_lengths(&[
        ("x columns", columns),
        ("fit coefficients", fit.coefficients.len()),
    ])?;
    if fit.n_used != data.n_used() {
        return Err(Error::invalid(
            "fit: the number of rows used differs from the data's, so it is not a fit of these data",
        ));
    }
    let prep = prepare(data, x, &opts.cox, &fit.coefficients)?;
    if prep.n_events == 0 {
        return Err(Error::invalid("the data have no event"));
    }
    let p = prep.p;
    let (terms, term_columns) = terms_of(opts.groups, &prep.keep, columns)?;
    let nterm = terms.len();
    let coefficient: Vec<f64> = prep
        .keep
        .iter()
        .map(|&j| fit.coefficients[j].unwrap_or(0.0))
        .collect();

    // The transformed times.
    let transformer = Transformer::new(opts.transform, &prep);
    if opts.transform == ZphTransform::Log
        && prep
            .strata
            .iter()
            .any(|s| s.event_times.iter().any(|&t| t <= 0.0))
    {
        return Err(Error::invalid(
            "transform: the log of a time that is not positive is not defined",
        ));
    }
    let mut event_values = Vec::with_capacity(prep.n_events);
    for s in &prep.strata {
        for (&t, &e) in s.stop.iter().zip(&s.is_event) {
            if e {
                event_values.push(transformer.at(t));
            }
        }
    }
    let centre = compensated(event_values.iter().copied()) / prep.n_events as f64;

    // The score and information of the extended model at the null.
    let mut u = vec![0.0f64; p];
    let mut q = [
        vec![0.0f64; p * p],
        vec![0.0f64; p * p],
        vec![0.0f64; p * p],
    ];
    let mut active = Vec::new();
    let mut sums = Sums::new(p, true);
    let mut mean = vec![0.0f64; p];
    let mut d1 = vec![0.0f64; p];
    let mut quotient = vec![0.0f64; p];
    for stratum in &prep.strata {
        let mut failed = false;
        stratum.sweep(p, &mut active, &mut sums, |position, at| {
            let g = transformer.at(stratum.event_times[position]) - centre;
            if stratum
                .at_time(position, at, prep.efron, &mut mean)
                .is_none()
            {
                failed = true;
                return;
            }
            let range = stratum.dying_offset[position]..stratum.dying_offset[position + 1];
            let deadwt: f64 = stratum.dying[range.clone()]
                .iter()
                .map(|&k| stratum.weight[k as usize])
                .sum();
            for a in 0..p {
                let observed: f64 = stratum.dying[range.clone()]
                    .iter()
                    .map(|&k| stratum.weight[k as usize] * stratum.x[k as usize * p + a])
                    .sum();
                u[a] += g * (observed - deadwt * mean[a]);
            }
            for (weight, fraction) in stratum.terms(position, prep.efron) {
                let d0 = at.risk.m0 - fraction * at.dying.m0;
                for (a, d) in d1.iter_mut().enumerate() {
                    *d = at.risk.m1[a] - fraction * at.dying.m1[a];
                }
                moments::quotients(&d1, d0, &mut quotient);
                for a in 0..p {
                    for b in 0..=a {
                        let d2 = at.risk.m2[a * p + b] - fraction * at.dying.m2[a * p + b];
                        let v = weight * (d2 / d0 - quotient[a] * quotient[b]);
                        q[0][a * p + b] += v;
                        q[1][a * p + b] += g * v;
                        q[2][a * p + b] += g * g * v;
                    }
                }
            }
        });
        if failed {
            return Err(Error::undefined(
                "the risk score or the denominator of a partial-likelihood term is not finite and positive at the coefficients",
            ));
        }
    }
    for block in &mut q {
        for a in 0..p {
            for b in 0..a {
                block[b * p + a] = block[a * p + b];
            }
        }
    }

    // The information of the extended model over the x columns and the g
    // columns `cols`, and the score, which is zero on the x columns.
    let extended = |cols: &[usize]| -> (Vec<f64>, Vec<f64>) {
        let m = p + cols.len();
        let mut a = vec![0.0f64; m * m];
        for r in 0..m {
            for c in 0..m {
                a[r * m + c] = match (r < p, c < p) {
                    (true, true) => q[0][r * p + c],
                    (true, false) => q[1][r * p + cols[c - p]],
                    (false, true) => q[1][cols[r - p] * p + c],
                    (false, false) => q[2][cols[r - p] * p + cols[c - p]],
                };
            }
        }
        let mut score = vec![0.0f64; m];
        for (k, &j) in cols.iter().enumerate() {
            score[p + k] = u[j];
        }
        (a, score)
    };
    let quadratic = |cols: &[usize]| -> Option<f64> {
        let (a, score) = extended(cols);
        let factor = Factor::new(&a, p + cols.len())?;
        let mut solved = score.clone();
        factor.solve(&mut solved);
        Some(score.iter().zip(&solved).map(|(s, z)| s * z).sum())
    };
    let tests: Vec<Option<ChiSquareTest>> = terms
        .iter()
        .map(|cols| {
            if opts.singledf && cols.len() > 1 {
                let (a, _) = extended(cols);
                let m = p + cols.len();
                let factor = Factor::new(&a, m)?;
                // The block of the inverse on the added columns.
                let mut block = vec![0.0f64; cols.len() * cols.len()];
                for c in 0..cols.len() {
                    let mut e = vec![0.0f64; m];
                    e[p + c] = 1.0;
                    factor.solve(&mut e);
                    for r in 0..cols.len() {
                        block[r * cols.len() + c] = e[p + r];
                    }
                }
                let t1: f64 = cols.iter().map(|&j| coefficient[j] * u[j]).sum();
                let mut quad = 0.0f64;
                for r in 0..cols.len() {
                    for c in 0..cols.len() {
                        quad +=
                            coefficient[cols[r]] * block[r * cols.len() + c] * coefficient[cols[c]];
                    }
                }
                ChiSquareTest::new(t1 * t1 * quad, 1)
            } else {
                ChiSquareTest::new(quadratic(cols)?, cols.len())
            }
        })
        .collect();
    let all: Vec<usize> = (0..p).collect();
    let global = quadratic(&all).and_then(|s| ChiSquareTest::new(s, p));

    // Scaled residuals.
    let rows = schoenfeld_rows(&prep)?;
    let events = rows.time.len();
    // The events a column was used for, by stratum: the number of events of the
    // stratum when the column varies in it, else 0.
    let mut used: Vec<Vec<usize>> = prep
        .strata
        .iter()
        .map(|s| {
            let n = s.is_event.iter().filter(|&&e| e).count();
            s.varies.iter().map(|&v| if v { n } else { 0 }).collect()
        })
        .collect();
    for cols in &terms {
        if cols.len() > 1 && used.iter().any(|row| cols.iter().any(|&j| row[j] == 0)) {
            for row in &mut used {
                let top = cols.iter().map(|&j| row[j]).max().unwrap_or(0);
                for &j in cols {
                    row[j] = top;
                }
            }
        }
    }
    let mut vmean = vec![0.0f64; p * p];
    for a in 0..p {
        for b in 0..p {
            let weight: usize = used.iter().map(|row| row[a].min(row[b])).sum();
            vmean[a * p + b] = q[0][a * p + b] / if weight == 0 { 1.0 } else { weight as f64 };
        }
    }
    // The residuals and the information collapsed to the terms. A term of one
    // column has the column itself, and a term of several the linear predictor
    // with the coefficients as weights.
    let weights: Vec<Vec<(usize, f64)>> = terms
        .iter()
        .map(|cols| {
            cols.iter()
                .map(|&j| (j, if cols.len() == 1 { 1.0 } else { coefficient[j] }))
                .collect()
        })
        .collect();
    let sres: Vec<Vec<f64>> = weights
        .iter()
        .map(|term| {
            (0..events)
                .map(|e| term.iter().map(|&(j, w)| rows.values[j][e] * w).sum())
                .collect()
        })
        .collect();
    let mut vterm = vec![0.0f64; nterm * nterm];
    for (i, ti) in weights.iter().enumerate() {
        for (k, tk) in weights.iter().enumerate() {
            let mut total = 0.0f64;
            for &(a, wa) in ti {
                for &(b, wb) in tk {
                    total += wa * vmean[a * p + b] * wb;
                }
            }
            vterm[i * nterm + k] = total;
        }
    }
    let used_terms: Vec<Vec<usize>> = used
        .iter()
        .map(|row| terms.iter().map(|cols| row[cols[0]]).collect())
        .collect();
    let mut y: Vec<Vec<Option<f64>>> = vec![vec![None; events]; nterm];
    let mut start = 0usize;
    for (s, stratum) in prep.strata.iter().enumerate() {
        let n_stratum = rows.stratum[start..]
            .iter()
            .take_while(|&&c| c == stratum.code)
            .count();
        let k: Vec<usize> = (0..nterm).filter(|&i| used_terms[s][i] > 0).collect();
        if !k.is_empty() && n_stratum > 0 {
            let m = k.len();
            let sub: Vec<f64> = (0..m * m)
                .map(|c| vterm[k[c / m] * nterm + k[c % m]])
                .collect();
            if let Some(factor) = Factor::new(&sub, m) {
                for e in start..start + n_stratum {
                    let mut b: Vec<f64> = k.iter().map(|&i| sres[i][e]).collect();
                    factor.solve(&mut b);
                    for (&i, v) in k.iter().zip(b) {
                        let shift = if terms[i].len() == 1 {
                            coefficient[terms[i][0]]
                        } else {
                            1.0
                        };
                        y[i][e] = Some(v + shift);
                    }
                }
            }
        }
        start += n_stratum;
    }
    let mut variance = vec![None; nterm * nterm];
    if nterm > 0
        && let Some(factor) = Factor::new(&vterm, nterm)
    {
        for c in 0..nterm {
            let mut e = vec![0.0f64; nterm];
            e[c] = 1.0;
            factor.solve(&mut e);
            for r in 0..nterm {
                variance[r * nterm + c] = Some(e[r]);
            }
        }
    }

    let x_values: Vec<f64> = rows.time.iter().map(|&t| transformer.at(t)).collect();
    let stratum = data.stratum_levels().map_or_else(Vec::new, |levels| {
        rows.stratum
            .iter()
            .map(|&s| levels.code(s as usize).unwrap_or(s))
            .collect()
    });
    Ok(CoxZph {
        transform: opts.transform,
        terms: term_columns,
        tests,
        global,
        time: rows.time,
        x: x_values,
        rows: rows.row,
        stratum,
        y,
        variance,
        n_rows: prep.n_rows,
        n_used: prep.n_used,
    })
}
