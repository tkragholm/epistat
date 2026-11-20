//! Cox regression with time-transform covariates, as `coxph(..., tt = )`.
//!
//! A covariate whose effect depends on time enters the partial likelihood as a
//! function `g(x, t)` of its value and of the event time `t` at which it is
//! evaluated, for example `x * log(t)` for a hazard ratio that drifts with log
//! time \[1\]. The value of `g` is not a property of the row, so the data cannot
//! be fitted as they stand. [`fit_tt`] expands them as `coxph` does and fits the
//! expansion with [`fit`].
//!
//! # Algorithm
//!
//! Take the weighted rows, apply the tie rule to the times, and work within
//! each stratum. For each distinct time `t` at which a row of the stratum has
//! the event, the risk set is `R(t)`, the rows with `a_j < t <= t_j` (every row
//! with `t_j >= t` when there is no entry time). Each row of `R(t)` gives one
//! row of the expansion, with
//!
//! ```text
//! exit time   t
//! status      1 if the row has the event at t, otherwise 0
//! covariates  g_k(x_jk, t) for a transformed column k, x_jk otherwise
//! weight, cluster and offset of row j
//! stratum     the risk set (stratum, t)
//! ```
//!
//! The expansion is fitted as right-censored data in which every risk set is a
//! stratum. A stratum with one event time has the partial likelihood term
//! `sum over D(t) of (beta x - log S_0(t))` of that time and no other, so the
//! sum over the risk sets is the partial likelihood of the original data with
//! the covariates evaluated at each event time \[2\]. Ties among events at one
//! time are handled inside the risk set by the rule of `CoxOptions::ties`, so
//! Breslow and Efron are as in a fit without a transform. Rows of weight zero
//! are left out. A row with another status code than the cause modelled is
//! censored in the risk sets and not an event.
//!
//! The model-based and robust variances, the tests and the iteration are those
//! of the fit of the expansion. With a cluster, the robust variance sums the
//! score residuals of the expanded rows within the cluster, so a subject
//! appears once for each event time it is at risk at, as in `coxph`.
//!
//! # The transform
//!
//! `coxph` takes an R function of the covariate, the time, the risk set and the
//! weights. A closure cannot cross a binding to R or Python, so
//! [`TimeTransform`] has the transforms that a binding can name:
//! `x * t`, `x * log(t)`, `x * (t > cut)` and the default of `coxph`, the
//! logit of the rank of `x` in the risk set (O'Brien's transform \[3\]). It
//! also takes a closure of `(x, t)` and a closure of a whole risk set, which
//! the transforms that look across rows need. The closures see one column at
//! a time, as `coxph`'s `tt` does, and cannot depend on other covariates.
//!
//! # Size of the expansion
//!
//! The expansion has one row for each pair of an event time and a row at risk
//! at it. With `n` rows and `m` distinct event times it is about `n m / 2` rows
//! when nobody is censored early, which is 10^12 rows for 10^7 rows and 10^5
//! times. [`fit_tt`] counts the rows first, in one sort and `m` binary
//! searches, and refuses when `TtOptions::max_expanded_rows` is exceeded or the
//! count is more than a `u32` addresses. Each row of the expansion takes
//! `8 p + 8` bytes for the covariates and the exit time, 4 for its stratum, 4
//! for the index of its source row, 4 for the status, and 8 more for each of a
//! weight, a cluster code and an offset when present. The expansion is built in
//! memory and not streamed, because [`fit`] reads its design matrix many times.
//! At register scale, coarsen the times first (by calendar month, for
//! example) or sample the risk sets, which this function does not do.
//!
//! The rows of a stratum with entry times are found by a scan of the rows with
//! `t_j >= t`, which costs the rows that have not yet entered as well, so a
//! stratum with long delayed entry costs more than its expansion.
//!
//! # References
//!
//! 1. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000. Section 6.2, time-dependent
//!    coefficients, and the survival vignette "Using time dependent covariates
//!    and time dependent coefficients in the Cox model".
//! 2. Cox DR. Regression models and life-tables. J R Stat Soc B
//!    1972;34(2):187-220.
//! 3. O'Brien PC. A non-parametric test for association with a censored
//!    variable. Biometrics 1978;34(2):243-50. doi:10.2307/2530013

use std::cmp::Ordering;
use std::fmt;

use faer::{Mat, MatRef};

use super::design::{CoxModelFit, CoxOptions, fit};
use crate::check::{check_lengths, check_row_count};
use crate::error::{Error, Result};
use crate::surv::{SurvBuf, SurvData};
use crate::timefix::TimeFix;

/// A function of the values of one covariate in a risk set. It receives the
/// event time, the covariate values and the case weights of the risk set, and
/// writes one transformed value per row into the last argument. The three
/// slices have the same length. Closures are `Sync` so that the transforms can
/// be shared across the threads of a caller.
pub type RiskSetTransform<'a> = dyn Fn(f64, &[f64], &[f64], &mut [f64]) + Sync + 'a;

/// How one covariate is transformed at an event time `t`.
#[derive(Clone, Copy)]
pub enum TimeTransform<'a> {
    /// `x * t`.
    Product,
    /// `x * ln(t)`. The event times must be positive.
    LogProduct,
    /// `x * (t > cut)`, a hazard ratio that changes at `cut`.
    Step {
        /// The time after which the covariate has its value.
        cut: f64,
    },
    /// The transform of `coxph` when no `tt` is given: for the `r`-th smallest
    /// of the `m` values in the risk set, with the rank of tied values the
    /// mean of their positions, `(r - 1/2) / (m - r + 1/2)`.
    OBrien,
    /// A closure of the covariate value and the event time.
    Function(&'a (dyn Fn(f64, f64) -> f64 + Sync + 'a)),
    /// A closure of a whole risk set.
    RiskSet(&'a RiskSetTransform<'a>),
}

impl fmt::Debug for TimeTransform<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Product => f.write_str("Product"),
            Self::LogProduct => f.write_str("LogProduct"),
            Self::Step { cut } => f.debug_struct("Step").field("cut", cut).finish(),
            Self::OBrien => f.write_str("OBrien"),
            Self::Function(_) => f.write_str("Function(..)"),
            Self::RiskSet(_) => f.write_str("RiskSet(..)"),
        }
    }
}

/// A column of the design matrix and the transform that replaces it. Columns
/// without an entry keep their values. To use both a covariate and its
/// transform, give the matrix the column twice.
#[derive(Debug, Clone, Copy)]
pub struct TtColumn<'a> {
    /// The column, counted from 0.
    pub column: usize,
    /// The transform applied to it.
    pub transform: TimeTransform<'a>,
}

/// What [`fit_tt`] takes besides the data, the design and the transforms.
#[derive(Debug, Clone, Copy, Default)]
pub struct TtOptions<'a> {
    /// The options of the fit of the expansion. `cox.timefix` is applied to
    /// the data before the expansion, and the expansion is not fixed again.
    /// `cox.offset` has one value per row of the data.
    pub cox: CoxOptions<'a>,
    /// The largest number of rows of the expansion to build. `None` sets no
    /// limit but the `u32` row count and the memory of the machine, which
    /// the function cannot check. There is no principled default.
    pub max_expanded_rows: Option<usize>,
}

/// A Cox fit with time-transform covariates.
#[derive(Debug, Clone, PartialEq)]
pub struct CoxTtFit {
    /// The fit of the expansion. `n_rows` and `n_used` count the rows of the
    /// data, not of the expansion. `n_events` and every other field is as in
    /// [`fit`].
    pub fit: CoxModelFit,
    /// The rows of the expansion, one for each event time and row at risk.
    pub n_expanded: usize,
    /// The risk sets: the distinct event times within strata.
    pub n_risk_sets: usize,
}

/// One risk set: where its rows start among the rows of the stratum sorted by
/// exit time, where that stratum ends, its time and its number of rows.
struct Plan {
    first: usize,
    end: usize,
    time: f64,
    size: usize,
}

/// Fit a Cox model in which some covariates are functions of time, as
/// `coxph` with `tt`.
///
/// `x` has one row per row of `data`, and `transforms` names the columns that
/// are replaced by their transform at each event time. See the module
/// documentation for the expansion, its size and the options. When the data
/// have no event there is no expansion and the fit is that of `x` as given,
/// with `n_expanded` 0.
///
/// # Errors
///
/// [`Error::Input`] when `x` has no column, has a number of rows other than
/// `data`, or holds a value that is not finite, when `transforms` is empty,
/// names a column that does not exist or names one twice, when a transform
/// gives a value that is not finite, when the expansion would have more rows
/// than `max_expanded_rows` or than a `u32` addresses, when `opts.cox.offset`
/// has another length than `data`, and for every reason that [`fit`] refuses.
pub fn fit_tt(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    transforms: &[TtColumn<'_>],
    opts: &TtOptions<'_>,
) -> Result<CoxTtFit> {
    check_lengths(&[("data", data.n_rows()), ("x", x.nrows())])?;
    if x.ncols() == 0 {
        return Err(Error::input("x has no columns"));
    }
    if transforms.is_empty() {
        return Err(Error::input(
            "transforms is empty; without a time transform use cox::fit",
        ));
    }
    let mut seen = vec![false; x.ncols()];
    for t in transforms {
        match seen.get_mut(t.column) {
            None => {
                return Err(Error::input(format!(
                    "transforms: column {} is outside the {} columns of x",
                    t.column,
                    x.ncols()
                )));
            }
            Some(flag) if *flag => {
                return Err(Error::input(format!(
                    "transforms: column {} is named twice",
                    t.column
                )));
            }
            Some(flag) => *flag = true,
        }
    }
    for j in 0..x.ncols() {
        if (0..x.nrows()).any(|i| !x[(i, j)].is_finite()) {
            return Err(Error::input(format!(
                "x: column {j} has a value that is not finite"
            )));
        }
    }
    if let Some(offset) = opts.cox.offset {
        check_lengths(&[("data", data.n_rows()), ("offset", offset.len())])?;
    }

    let data = data.clone().with_timefix(opts.cox.timefix)?;
    let surv = data.followup()?;
    let code = match (opts.cox.cause, surv.n_codes()) {
        (None, 1) => 1,
        (Some(code), _) => code,
        (None, _) => {
            return Err(Error::input(
                "cause: the status has several codes, so the event code is required",
            ));
        }
    };
    let is_event: Vec<bool> = data.cause(code)?.iter().collect();
    let stop = surv.stop();
    let start = surv.start();
    let stratum = data.stratum();

    // The weighted rows by stratum, then by exit time. The row number settles
    // equal keys, so the order does not depend on the sort.
    let mut order: Vec<u32> = data.used_rows().collect();
    order.sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        let by_stratum = stratum.map_or(Ordering::Equal, |s| s[a].cmp(&s[b]));
        by_stratum.then(stop[a].total_cmp(&stop[b])).then(a.cmp(&b))
    });

    let mut plans: Vec<Plan> = Vec::new();
    let mut lo = 0;
    while lo < order.len() {
        let key = |i: usize| stratum.map_or(0, |s| s[order[i] as usize]);
        let mut hi = lo + 1;
        while hi < order.len() && key(hi) == key(lo) {
            hi += 1;
        }
        let stops: Vec<f64> = order[lo..hi].iter().map(|&r| stop[r as usize]).collect();
        let starts: Option<Vec<f64>> = start.map(|a| {
            let mut v: Vec<f64> = order[lo..hi].iter().map(|&r| a[r as usize]).collect();
            v.sort_unstable_by(f64::total_cmp);
            v
        });
        let mut previous: Option<f64> = None;
        for (k, &row) in order[lo..hi].iter().enumerate() {
            let t = stop[row as usize];
            if !is_event[row as usize] || previous.is_some_and(|p| p.total_cmp(&t).is_eq()) {
                continue;
            }
            previous = Some(t);
            let first = stops.partition_point(|&s| s.total_cmp(&t).is_lt());
            debug_assert!(first <= k);
            let mut size = stops.len() - first;
            if let Some(starts) = &starts {
                size -= starts.len() - starts.partition_point(|&a| a.total_cmp(&t).is_lt());
            }
            plans.push(Plan {
                first: lo + first,
                end: hi,
                time: t,
                size,
            });
        }
        lo = hi;
    }

    let total: usize = plans.iter().map(|p| p.size).sum();
    if let Some(limit) = opts.max_expanded_rows.filter(|&limit| total > limit) {
        return Err(Error::input(format!(
            "the expansion has {total} rows, more than max_expanded_rows = {limit}"
        )));
    }
    check_row_count(total)?;
    if total == 0 {
        let fit = fit(&data, x, &opts.cox)?;
        return Ok(CoxTtFit {
            fit,
            n_expanded: 0,
            n_risk_sets: 0,
        });
    }

    // The source row of every row of the expansion.
    let mut source: Vec<u32> = Vec::with_capacity(total);
    let mut time: Vec<f64> = Vec::with_capacity(total);
    let mut stratum_e: Vec<u32> = Vec::with_capacity(total);
    for (id, plan) in plans.iter().enumerate() {
        for &row in &order[plan.first..plan.end] {
            if start.is_some_and(|a| a[row as usize].total_cmp(&plan.time).is_ge()) {
                continue;
            }
            source.push(row);
            time.push(plan.time);
            stratum_e.push(id as u32);
        }
    }
    drop(order);
    debug_assert_eq!(source.len(), total);

    let status: Vec<i32> = source
        .iter()
        .zip(&time)
        .map(|(&row, t)| {
            i32::from(is_event[row as usize] && stop[row as usize].total_cmp(t).is_eq())
        })
        .collect();

    let mut design = Mat::<f64>::zeros(total, x.ncols());
    for j in 0..x.ncols() {
        match transforms.iter().find(|t| t.column == j) {
            None => {
                for (k, &row) in source.iter().enumerate() {
                    design[(k, j)] = x[(row as usize, j)];
                }
            }
            Some(t) => {
                transform_column(
                    &mut design,
                    j,
                    t.transform,
                    x,
                    &source,
                    &time,
                    &plans,
                    data.weight(),
                );
                if let Some(k) = (0..total).find(|&k| !design[(k, j)].is_finite()) {
                    return Err(Error::input(format!(
                        "transforms: column {j} is not finite at time {}",
                        time[k]
                    )));
                }
            }
        }
    }

    let mut builder = SurvBuf::builder(time, status).stratum(stratum_e);
    if let Some(w) = data.weight() {
        builder = builder.weight(source.iter().map(|&r| w[r as usize]).collect());
    }
    if let Some(c) = data.cluster() {
        builder = builder.cluster(source.iter().map(|&r| c[r as usize]).collect());
    }
    let expanded = builder.build()?;
    let offset: Option<Vec<f64>> = opts
        .cox
        .offset
        .map(|o| source.iter().map(|&r| o[r as usize]).collect());
    drop(source);
    let options = CoxOptions {
        timefix: TimeFix::Exact,
        cause: None,
        offset: offset.as_deref(),
        ..opts.cox
    };
    let mut result = fit(&expanded.as_data(), design.as_ref(), &options)?;
    result.n_rows = data.n_rows();
    result.n_used = data.n_used();
    Ok(CoxTtFit {
        fit: result,
        n_expanded: total,
        n_risk_sets: plans.len(),
    })
}

/// Write the transform of column `j` for every row of the expansion.
fn transform_column(
    design: &mut Mat<f64>,
    j: usize,
    transform: TimeTransform<'_>,
    x: MatRef<'_, f64>,
    source: &[u32],
    time: &[f64],
    plans: &[Plan],
    weight: Option<&[f64]>,
) {
    let each = |design: &mut Mat<f64>, g: &dyn Fn(f64, f64) -> f64| {
        for (k, (&row, &t)) in source.iter().zip(time).enumerate() {
            design[(k, j)] = g(x[(row as usize, j)], t);
        }
    };
    match transform {
        TimeTransform::Product => each(design, &|v, t| v * t),
        TimeTransform::LogProduct => each(design, &|v, t| v * t.ln()),
        TimeTransform::Step { cut } => each(design, &|v, t| if t > cut { v } else { 0.0 }),
        TimeTransform::Function(g) => each(design, g),
        TimeTransform::OBrien | TimeTransform::RiskSet(_) => {
            let mut at = 0;
            let (mut values, mut weights, mut out) = (Vec::new(), Vec::new(), Vec::new());
            for plan in plans {
                let rows = &source[at..at + plan.size];
                values.clear();
                values.extend(rows.iter().map(|&r| x[(r as usize, j)]));
                weights.clear();
                weights.extend(rows.iter().map(|&r| weight.map_or(1.0, |w| w[r as usize])));
                out.clear();
                out.resize(plan.size, 0.0);
                match transform {
                    TimeTransform::RiskSet(g) => g(plan.time, &values, &weights, &mut out),
                    _ => obrien(&values, &mut out),
                }
                for (k, v) in out.iter().enumerate() {
                    design[(at + k, j)] = *v;
                }
                at += plan.size;
            }
        }
    }
}

/// `(r - 1/2) / (m - r + 1/2)` for the rank `r` of each value among `m`, tied
/// values sharing the mean of their positions.
#[expect(
    clippy::float_cmp,
    reason = "values are tied when they are the same number"
)]
fn obrien(values: &[f64], out: &mut [f64]) {
    let m = values.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]).then(a.cmp(&b)));
    let mut i = 0;
    while i < m {
        let mut j = i + 1;
        while j < m && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        // Positions i+1 ..= j, so their mean is (i + 1 + j) / 2.
        let r = (i + 1 + j) as f64 / 2.0;
        let value = (r - 0.5) / (0.5 + m as f64 - r);
        for &k in &idx[i..j] {
            out[k] = value;
        }
        i = j;
    }
}
