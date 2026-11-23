//! Case-cohort analysis: the Cox model on the cases and a random subcohort, with
//! the variance of the two-phase sampling, as `survival::cch`.
//!
//! In a case-cohort study covariates are known for every case and for a
//! subcohort sampled from the whole cohort at baseline, and the cohort size is
//! known. [`fit`] takes the rows of those people and the indicator of the
//! subcohort, and gives the coefficients and a covariance matrix that has the
//! model part and a part for sampling the subcohort. The five methods of `cch`
//! are the five [`CaseCohortMethod`] values.
//!
//! # Notation
//!
//! `N` is the cohort size, `n_c` the subcohort size, `n_d` the number of cases
//! and `n_cd` the cases inside the subcohort. A row is a subcohort non-case
//! (`cc = 0`), a subcohort case (`cc = 1`) or a case outside the subcohort
//! (`cc = 2`). `delta` is half the smallest gap between two distinct failure
//! times.
//!
//! # Algorithm
//!
//! Every method is a sequence of [`cox::fit`](crate::cox::fit) calls on
//! rearranged rows, followed by a few cross-products of the `dfbeta` or score
//! residuals. A row of the Cox data is `(entry, exit, event)`, and a row that
//! `cch` gives offset `-100` is one that must enter no denominator while still
//! providing its event.
//!
//! - Prentice \[1\]. The coefficients are those of a Cox fit on all rows where
//!   a case outside the subcohort enters at `exit - delta`, so it is in the risk
//!   set at its own failure time only. The variance is that of a second fit on
//!   stacked rows, all cases as events with offset `-100` (so they contribute
//!   an event and nothing to a denominator) followed by the subcohort as
//!   non-events with offset 0, started at the first fit's coefficients. With
//!   `V_m` the model-based covariance of the second fit and `db_i` the `dfbeta`
//!   of its subcohort rows,
//!
//!   ```text
//!   phase2 = (1 - n_c / N) * sum_i db_i db_i'
//!   V      = V_m + phase2
//!   ```
//!
//! - Self-Prentice \[2\]. The second fit gives the coefficients as well, so the
//!   risk sets hold the subcohort members at risk and nothing else. `phase2` and
//!   `V` are as for Prentice.
//! - Lin-Ying \[3\]. One fit on all rows with offset `log w`, `w = (N - n_d) /
//!   (n_c - n_cd)` for subcohort non-cases and 1 for cases. With `db` the
//!   `dfbeta` of the rows, `db0` those of the non-cases centred at their mean,
//!
//!   ```text
//!   phase2 = (1 - (n_c - n_cd) / (N - n_d)) * db0' db0
//!   V      = V_m + phase2,     or with `robust`   V = sum_i db_i db_i' / w_i + phase2
//!   ```
//!
//!   The robust form is Barlow's \[4\] variance for the sampling of the first
//!   phase as well.
//! - Borgan I and II \[5\]. Subcohort stratum `j` has cohort size `N_j`, `m_j`
//!   members of which `m_0j` are non-cases, and `d_j` cases. Method I is the
//!   Self-Prentice rows with weight `N_j / m_j` on the subcohort rows and 1 on
//!   the cases. Method II is the Lin-Ying rows with weight `(N_j - d_j) / m_0j`
//!   on the non-cases and 1 on the cases, and no offset. With `V_n` the
//!   model-based covariance of the weighted fit, `S_j` the covariance of the
//!   unweighted score residuals of the subcohort rows (the non-cases for II)
//!   within the stratum, centred at the stratum mean and divided by `m_j - 1`
//!   (`m_0j - 1` for II), and `w_j` the weight of the stratum,
//!
//!   ```text
//!   D = sum_j (w_j - 1) N_j S_j      (N_j - d_j for II)
//!   phase2 = V_n D V_n,              V = V_n + phase2
//!   ```
//!
//!   The proportions `N_j sqrt(diag(V_n S_j V_n))`, scaled to sum to 1 over the
//!   strata, are returned as `optimal_allocation`, the `opt` of `cch`.
//!
//! The entry of a case outside the subcohort at `exit - delta` is a device of
//! `cch`: such a case must be in the risk set at its own failure time and a
//! start-stop row cannot start where it stops. `delta` is half the smallest gap
//! between two distinct failure times, so the entry is after every earlier
//! failure and before its own, in any time unit. Only Prentice's method uses it.
//!
//! # Where it differs from cch
//!
//! - `cch` warns and goes on for `robust` with a stratified method and for
//!   `stratum` with an unstratified one. Both are an error here, since the
//!   request has no effect.
//! - With one distinct failure time `cch` uses `delta = 1`. Prentice's method
//!   has no estimate there, and is an error here. The other methods do not use
//!   `delta` and are unaffected.
//! - A stratum of Borgan's methods with one member gives `0/0` in `cch`. Here
//!   its term is 0 when its weight is 1, which is the exact answer when the
//!   whole stratum is sampled, and an error otherwise. A stratum without
//!   subcohort members (without non-case members for II) is an error.
//! - In the stacked rows of Self-Prentice, of Prentice's variance and of Borgan I
//!   a case outside the subcohort provides an event, and the risk set is the
//!   subcohort at risk. When no subcohort member is at risk at its time, `cch`
//!   forms the risk set from the other cases, whose offsets are all `-100` and
//!   which therefore count as fully as any, and adds a term for it, an artefact
//!   of the offset. Here such a case is not an event and adds nothing, as the
//!   estimator is defined. It does not arise when every failure time has a
//!   subcohort member at risk, as in the `nwtco` data.
//! - The times are not changed by a tie rule, as `cch` passes `timefix = FALSE`.
//!   Apply [`timefix`](crate::timefix) beforehand if the times need it.
//! - `cch` fits Efron ties only. [`Ties::Breslow`] is available and is checked
//!   against `cch`'s own fitters with Breslow ties; exact ties are refused.
//!
//! # Scale
//!
//! Each fit is a Cox fit on at most `n_c + n_d` rows, plus a product of
//! `dfbeta` matrices linear in the rows. The cohort is never read, only its
//! size.
//!
//! # Agreement with R
//!
//! `tests/r_casecohort.rs` compares coefficients, `var`, `phase2var`, the model
//! covariance, `opt` and the iteration count of the five methods with
//! `survival::cch` 3.8.12 on the `nwtco` data of its help page, with a
//! stratified subcohort, with delayed entry, with failure times without ties,
//! with `robust = TRUE`, and with Breslow ties.
//!
//! # References
//!
//! 1. Prentice RL. A case-cohort design for epidemiologic cohort studies and
//!    disease prevention trials. Biometrika 1986;73(1):1-11.
//! 2. Self SG, Prentice RL. Asymptotic distribution theory and efficiency
//!    results for case-cohort studies. Ann Stat 1988;16(1):64-81.
//! 3. Lin DY, Ying Z. Cox regression with incomplete covariate measurements.
//!    J Am Stat Assoc 1993;88(424):1341-9.
//! 4. Barlow WE. Robust variance estimation for the case-cohort design.
//!    Biometrics 1994;50(4):1064-72.
//! 5. Borgan O, Langholz B, Samuelsen SO, Goldstein L, Pogoda J. Exposure
//!    stratified case-cohort designs. Lifetime Data Anal 2000;6(1):39-58.
//! 6. Therneau TM, Li H. Computing the Cox model for case-cohort designs.
//!    Lifetime Data Anal 1999;5(2):99-112.

use epistat_linalg::{Mat, MatRef};

use crate::check::{check_lengths, refuse_non_finite, refuse_non_finite_matrix};
use crate::cox::{
    Collapse, CoxModelFit, CoxOptions, ResidualKind, ResidualOptions, Ties, fit as cox_fit,
    residuals,
};
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// The offset that keeps a row out of every denominator of the partial
/// likelihood while it still provides its event, as `cch` writes it.
const OUT_OF_RISK_SETS: f64 = -100.0;

/// Most trial points of the fits, as `cch` sets them: `coxph`'s 20 for the
/// first fit and the rows of Self-Prentice and Lin-Ying, 35 for the variance
/// fit of Prentice and 25 for Borgan's methods.
const ITER_DEFAULT: u32 = 20;
const ITER_PRENTICE_VARIANCE: u32 = 35;
const ITER_BORGAN: u32 = 25;

/// The method of the analysis, named as `method` of `cch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseCohortMethod {
    /// `"Prentice"`: the subcohort members at risk and the failing case when it
    /// is outside the subcohort.
    Prentice,
    /// `"SelfPrentice"`: the subcohort members at risk only.
    SelfPrentice,
    /// `"LinYing"`: all subcohort members and all cases at risk, the
    /// subcohort non-cases weighted by `(N - n_d) / (n_c - n_cd)`.
    LinYing,
    /// `"I.Borgan"`: Self-Prentice for a stratified subcohort.
    BorganI,
    /// `"II.Borgan"`: Lin-Ying for a stratified subcohort.
    BorganII,
}

impl CaseCohortMethod {
    fn stratified(self) -> bool {
        matches!(self, Self::BorganI | Self::BorganII)
    }
}

/// What [`fit`] takes besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaseCohortOptions<'a> {
    /// The method.
    pub method: CaseCohortMethod,
    /// The cohort size: one value for the unstratified methods, one per
    /// stratum in ascending order of the stratum codes for Borgan's.
    pub cohort_size: &'a [f64],
    /// Barlow's variance for the first phase as well, as `robust` of `cch`.
    /// Lin-Ying only; it is an error with any other method.
    pub robust: bool,
    /// How tied events enter the partial likelihood: Efron, as `coxph`, or
    /// Breslow. Exact ties are refused.
    pub ties: Ties,
    /// The status code to model when the data have several, as for the Cox
    /// fit. `None` for one code.
    pub cause: Option<u32>,
    /// The relative change of the log partial likelihood at which the fits
    /// stop, `eps` of `coxph.control`.
    pub eps: f64,
}

impl<'a> CaseCohortOptions<'a> {
    /// The options of `method` for a cohort of `cohort_size`, with Efron ties,
    /// no robust variance and the default convergence limit.
    #[must_use]
    pub fn new(method: CaseCohortMethod, cohort_size: &'a [f64]) -> Self {
        Self {
            method,
            cohort_size,
            robust: false,
            ties: Ties::Efron,
            cause: None,
            eps: 1e-9,
        }
    }
}

/// A case-cohort fit.
///
/// Matrices are row-major with `p * p` entries, `None` where the row or the
/// column is a coefficient that is `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseCohortFit {
    /// One coefficient per column of the design, `None` for an aliased column
    /// and for every column when the fit has no estimate.
    pub coefficients: Vec<Option<f64>>,
    /// The covariance matrix of the coefficients, `var` of `cch`: the model
    /// part plus the sampling part, or Barlow's for `robust`.
    pub covariance: Vec<Option<f64>>,
    /// The model-based covariance of the fit that gives the variance, before
    /// the sampling part is added. `cch` stores the sum in `naive.var`.
    pub naive_covariance: Vec<Option<f64>>,
    /// The part of the variance for sampling the subcohort, `phase2var`.
    pub phase_two_covariance: Vec<Option<f64>>,
    /// For Borgan's methods, the proportion of the subcohort to take from each
    /// stratum to minimise the variance of each coefficient, `opt` of `cch`: one
    /// row per stratum and one column per coefficient, row-major. `None`
    /// otherwise.
    pub optimal_allocation: Option<Vec<Option<f64>>>,
    /// The method.
    pub method: CaseCohortMethod,
    /// The cohort size, summed over the strata.
    pub n_cohort: f64,
    /// The number of subcohort members among the rows.
    pub n_subcohort: usize,
    /// The number of cases among the rows.
    pub n_cases: usize,
    /// The number of cases outside the subcohort.
    pub n_cases_outside: usize,
    /// The trial points of the fit that gives the coefficients.
    pub iterations: u32,
    /// Whether every fit stopped on the convergence limit.
    pub converged: bool,
}

/// The rows of a Cox fit with their covariates, the offset and the weight.
struct Stack {
    stop: Vec<f64>,
    start: Vec<f64>,
    status: Vec<i32>,
    weight: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    x: Mat<f64>,
}

/// The fixed settings of every Cox fit of one analysis.
#[derive(Clone, Copy)]
struct Control {
    ties: Ties,
    eps: f64,
}

impl Stack {
    fn data(&self) -> Result<SurvData<'_>> {
        let mut builder = SurvData::builder(&self.stop, &self.status).start(&self.start);
        if let Some(weight) = &self.weight {
            builder = builder.weight(weight);
        }
        builder.build()
    }

    fn options<'a>(
        &'a self,
        control: Control,
        init: Option<&'a [f64]>,
        iter_max: u32,
    ) -> CoxOptions<'a> {
        CoxOptions {
            ties: control.ties,
            timefix: TimeFix::Exact,
            cause: None,
            offset: self.offset.as_deref(),
            init,
            iter_max,
            eps: control.eps,
        }
    }

    fn fit(&self, control: Control, init: Option<&[f64]>, iter_max: u32) -> Result<CoxModelFit> {
        cox_fit(
            &self.data()?,
            self.x.as_ref(),
            &self.options(control, init, iter_max),
        )
    }

    /// One column per coefficient: the `dfbeta` (not weighted) or the
    /// unweighted score residuals of every row.
    fn residuals(
        &self,
        control: Control,
        model: &CoxModelFit,
        kind: ResidualKind,
    ) -> Result<Vec<Option<Vec<f64>>>> {
        let mut options = ResidualOptions::new(kind, self.options(control, None, ITER_DEFAULT));
        options.weighted = Some(false);
        options.collapse = Collapse::None;
        Ok(residuals(&self.data()?, self.x.as_ref(), model, &options)?.columns)
    }
}

/// The rows of the analysis, in the order of the input.
struct Sample<'a> {
    x: MatRef<'a, f64>,
    stop: &'a [f64],
    start: Vec<f64>,
    event: Vec<bool>,
    subcohort: &'a [bool],
}

impl Sample<'_> {
    fn rows(&self) -> usize {
        self.stop.len()
    }

    /// The rows listed, each with its event indicator, offset and weight.
    fn stack(&self, rows: &[(usize, bool, f64, f64)], weighted: bool, offset: bool) -> Stack {
        let n = rows.len();
        Stack {
            stop: rows.iter().map(|r| self.stop[r.0]).collect(),
            start: rows.iter().map(|r| self.start[r.0]).collect(),
            status: rows.iter().map(|r| i32::from(r.1)).collect(),
            weight: weighted.then(|| rows.iter().map(|r| r.3).collect()),
            offset: offset.then(|| rows.iter().map(|r| r.2).collect()),
            x: Mat::from_fn(n, self.x.ncols(), |i, j| self.x[(rows[i].0, j)]),
        }
    }

    /// Whether a subcohort member is at risk at the failure time of row `i`:
    /// some subcohort row with `start < t <= stop`. A case in the subcohort is
    /// its own member.
    fn subcohort_at_risk(&self) -> impl Fn(usize) -> bool {
        let mut stops: Vec<f64> = Vec::new();
        let mut starts: Vec<f64> = Vec::new();
        for i in (0..self.rows()).filter(|&i| self.subcohort[i]) {
            stops.push(self.stop[i]);
            starts.push(self.start[i]);
        }
        stops.sort_by(f64::total_cmp);
        starts.sort_by(f64::total_cmp);
        let time: Vec<f64> = self.stop.to_vec();
        move |i| {
            let t = time[i];
            // Rows with `stop >= t`, less those with `start >= t`.
            let stopped_before = stops.partition_point(|&s| s < t);
            let started_after = starts.partition_point(|&s| s < t);
            (stops.len() - stopped_before) > (starts.len() - started_after)
        }
    }

    /// Cases first, as events outside every denominator, then the subcohort as
    /// non-events, as `cch` stacks them. `weight` gives the weight of a
    /// subcohort row by its position among the rows. A case outside the
    /// subcohort at whose time no subcohort member is at risk is not an event,
    /// since the subcohort provides the whole risk set.
    fn stacked(&self, weight: Option<&dyn Fn(usize) -> f64>) -> (Stack, usize) {
        let mut rows = Vec::new();
        let at_risk = self.subcohort_at_risk();
        for (i, _) in self.event.iter().enumerate().filter(|&(_, &e)| e) {
            rows.push((i, at_risk(i), OUT_OF_RISK_SETS, 1.0));
        }
        let cases = rows.len();
        for (i, _) in self.subcohort.iter().enumerate().filter(|&(_, &s)| s) {
            rows.push((i, false, 0.0, weight.map_or(1.0, |w| w(i))));
        }
        (self.stack(&rows, weight.is_some(), true), cases)
    }
}

/// A dense `q * q` matrix over the columns kept, row-major.
type Dense = Vec<f64>;

fn mat_mul(a: &[f64], b: &[f64], q: usize) -> Dense {
    let mut out = vec![0.0; q * q];
    for i in 0..q {
        for k in 0..q {
            let aik = a[i * q + k];
            for j in 0..q {
                out[i * q + j] += aik * b[k * q + j];
            }
        }
    }
    out
}

/// `sum_i f_i row_i row_i'` over the rows of `rows`, a `q`-vector each.
fn weighted_crossproduct(rows: &[Vec<f64>], factor: impl Fn(usize) -> f64, q: usize) -> Dense {
    let mut out = vec![0.0; q * q];
    for (i, row) in rows.iter().enumerate() {
        let f = factor(i);
        for a in 0..q {
            for b in 0..q {
                out[a * q + b] += f * row[a] * row[b];
            }
        }
    }
    out
}

/// The residual rows over the columns kept: row `i` of the result has the value
/// of each kept column at the position `rows[i]`.
fn gather(columns: &[Option<Vec<f64>>], kept: &[usize], rows: &[usize]) -> Result<Vec<Vec<f64>>> {
    let kept_columns: Vec<&Vec<f64>> = kept
        .iter()
        .map(|&j| {
            columns[j]
                .as_ref()
                .ok_or_else(|| Error::input("fit: a residual column is missing for a coefficient"))
        })
        .collect::<Result<_>>()?;
    Ok(rows
        .iter()
        .map(|&r| kept_columns.iter().map(|c| c[r]).collect())
        .collect())
}

/// The covariance of a fit over the kept columns.
fn kept_covariance(model: &CoxModelFit, kept: &[usize]) -> Result<Dense> {
    let p = model.coefficients.len();
    let mut out = Vec::with_capacity(kept.len() * kept.len());
    for &a in kept {
        for &b in kept {
            out.push(
                model.covariance[a * p + b]
                    .ok_or_else(|| Error::input("fit: the model covariance has no value"))?,
            );
        }
    }
    Ok(out)
}

fn kept_columns(model: &CoxModelFit) -> Vec<usize> {
    model
        .coefficients
        .iter()
        .enumerate()
        .filter_map(|(j, c)| c.map(|_| j))
        .collect()
}

/// A `q * q` matrix on the kept columns as a `p * p` one with `None` elsewhere.
fn embed(matrix: &[f64], kept: &[usize], p: usize) -> Vec<Option<f64>> {
    let mut out = vec![None; p * p];
    let q = kept.len();
    for (a, &ja) in kept.iter().enumerate() {
        for (b, &jb) in kept.iter().enumerate() {
            out[ja * p + jb] = Some(matrix[a * q + b]);
        }
    }
    out
}

/// What the method adds to a Cox fit: the pieces of the result.
struct Pieces {
    coefficients: Vec<Option<f64>>,
    naive: Vec<Option<f64>>,
    phase_two: Vec<Option<f64>>,
    covariance: Vec<Option<f64>>,
    optimal: Option<Vec<Option<f64>>>,
    iterations: u32,
    converged: bool,
}

/// A fit with no estimate: every entry `None`.
fn no_estimate(p: usize, strata: Option<usize>, fits: &[&CoxModelFit]) -> Pieces {
    Pieces {
        coefficients: vec![None; p],
        naive: vec![None; p * p],
        phase_two: vec![None; p * p],
        covariance: vec![None; p * p],
        optimal: strata.map(|j| vec![None; j * p]),
        iterations: fits.first().map_or(0, |f| f.iterations),
        converged: fits.iter().all(|f| f.converged),
    }
}

fn pieces_from(
    coefficients: Vec<Option<f64>>,
    kept: &[usize],
    naive: &[f64],
    phase_two: &[f64],
    optimal: Option<Vec<Option<f64>>>,
    iterations: u32,
    converged: bool,
) -> Pieces {
    let p = coefficients.len();
    let q = kept.len();
    let total: Vec<f64> = (0..q * q).map(|k| naive[k] + phase_two[k]).collect();
    Pieces {
        coefficients,
        naive: embed(naive, kept, p),
        phase_two: embed(phase_two, kept, p),
        covariance: embed(&total, kept, p),
        optimal,
        iterations,
        converged,
    }
}

/// The case-cohort fit of the rows in `data`.
///
/// `data` holds one row for every case and every subcohort member, with the
/// stop time, the optional start time, the status and optionally the id;
/// `x` the covariates, one row per row of `data`; and `subcohort` says which
/// rows are in the subcohort. `stratum` is the stratum of each row for
/// Borgan's methods and `None` for the others. `data` may not carry case
/// weights, strata or clusters, which the design takes over. See the module
/// documentation for the methods.
///
/// A column that is aliased has `None` for its coefficient and its row and
/// column of the matrices. When the Cox fit has no estimate every entry is
/// `None`.
///
/// # Errors
///
/// [`Error::Input`] for the invalid input of the Cox fit; for `data` with case
/// weights, strata or clusters, or an id that occurs twice; for a row outside
/// the subcohort that is not a case; for `subcohort` or `stratum` of another
/// length than the data; for `robust` with a method other than Lin-Ying; for a
/// stratum with an unstratified method, and none with a stratified one; for a
/// cohort size that is not finite and positive, has another length than the
/// number of strata (one without strata), or is smaller than the rows of its
/// stratum; for no case, or no subcohort non-case for Lin-Ying and for Borgan's
/// second method; for fewer than two distinct failure times with Prentice's
/// method; for a stratum without the members its weight needs, or with one
/// member and a weight other than 1; for exact ties; and for a design without
/// columns.
pub fn fit(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    subcohort: &[bool],
    stratum: Option<&[u32]>,
    opts: &CaseCohortOptions<'_>,
) -> Result<CaseCohortFit> {
    let rows = data.n_rows();
    check_lengths(&[
        ("data", rows),
        ("x", x.nrows()),
        ("subcohort", subcohort.len()),
    ])?;
    refuse_non_finite_matrix(x, "x")?;
    if x.ncols() == 0 {
        return Err(Error::input("x: the design has no columns"));
    }
    check_inputs(data, stratum, opts)?;
    let surv = data.followup()?;
    let code = match (opts.cause, surv.n_codes()) {
        (None, 1) => 1,
        (Some(code), _) => code,
        (None, _) => {
            return Err(Error::input(
                "cause: the status has several codes, so the event code is required",
            ));
        }
    };
    let event: Vec<bool> = data.cause(code)?.iter().collect();
    if let Some(i) = (0..rows).find(|&i| !event[i] && !subcohort[i]) {
        let count = (0..rows).filter(|&i| !event[i] && !subcohort[i]).count();
        return Err(Error::input(format!(
            "subcohort: {count} censored observations are outside the subcohort, the first is row {i}"
        )));
    }
    let stop = surv.stop();
    let start = surv
        .start()
        .map_or_else(|| vec![0.0; rows], <[f64]>::to_vec);
    let sample = Sample {
        x,
        stop,
        start,
        event,
        subcohort,
    };
    let control = Control {
        ties: opts.ties,
        eps: opts.eps,
    };
    let n_cases = sample.event.iter().filter(|&&e| e).count();
    let n_subcohort = subcohort.iter().filter(|&&s| s).count();
    let n_cases_outside = (0..rows)
        .filter(|&i| sample.event[i] && !subcohort[i])
        .count();
    let n_cases_inside = n_cases - n_cases_outside;
    if n_cases == 0 {
        return Err(Error::input("data: there is no case"));
    }
    let counts = Counts {
        cases: n_cases,
        subcohort: n_subcohort,
        cases_inside: n_cases_inside,
    };
    let pieces = match opts.method {
        CaseCohortMethod::Prentice => {
            prentice(&sample, control, opts.cohort_size[0], counts, true)?
        }
        CaseCohortMethod::SelfPrentice => {
            prentice(&sample, control, opts.cohort_size[0], counts, false)?
        }
        CaseCohortMethod::LinYing => lin_ying(&sample, control, opts, counts)?,
        CaseCohortMethod::BorganI | CaseCohortMethod::BorganII => {
            let stratum = stratum.ok_or_else(|| Error::input("stratum: required for Borgan"))?;
            borgan(&sample, control, opts, stratum)?
        }
    };
    Ok(CaseCohortFit {
        coefficients: pieces.coefficients,
        covariance: pieces.covariance,
        naive_covariance: pieces.naive,
        phase_two_covariance: pieces.phase_two,
        optimal_allocation: pieces.optimal,
        method: opts.method,
        n_cohort: opts.cohort_size.iter().sum(),
        n_subcohort,
        n_cases,
        n_cases_outside,
        iterations: pieces.iterations,
        converged: pieces.converged,
    })
}

#[derive(Clone, Copy)]
struct Counts {
    cases: usize,
    subcohort: usize,
    cases_inside: usize,
}

/// The checks that do not read the status.
fn check_inputs(
    data: &SurvData<'_>,
    stratum: Option<&[u32]>,
    opts: &CaseCohortOptions<'_>,
) -> Result<()> {
    let rows = data.n_rows();
    if opts.ties == Ties::Exact {
        return Err(Error::input("ties: the exact method is not supported"));
    }
    if data.weight().is_some() || data.stratum().is_some() || data.cluster().is_some() {
        return Err(Error::input(
            "data: case weights, strata and clusters are set by the design, so the data may not carry them",
        ));
    }
    if let Some(id) = data.id() {
        let mut sorted = id.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::input(
                "id: more than one row per id is not supported",
            ));
        }
    }
    if opts.robust && opts.method != CaseCohortMethod::LinYing {
        return Err(Error::input(
            "robust: only the Lin-Ying method has a robust variance",
        ));
    }
    refuse_non_finite(opts.cohort_size, "cohort_size")?;
    if opts.cohort_size.iter().any(|&n| n <= 0.0) {
        return Err(Error::input("cohort_size: every value must be positive"));
    }
    if !(opts.eps.is_finite() && opts.eps > 0.0) {
        return Err(Error::input("eps: must be finite and positive"));
    }
    match (opts.method.stratified(), stratum) {
        (false, Some(_)) => {
            return Err(Error::input(
                "stratum: the unstratified methods take no stratum",
            ));
        }
        (true, None) => {
            return Err(Error::input("stratum: Borgan's methods require it"));
        }
        (false, None) => {
            if opts.cohort_size.len() != 1 {
                return Err(Error::input(
                    "cohort_size: the unstratified methods take one value",
                ));
            }
            if opts.cohort_size[0] < rows as f64 {
                return Err(Error::input("cohort_size: smaller than the number of rows"));
            }
        }
        (true, Some(stratum)) => {
            check_lengths(&[("data", rows), ("stratum", stratum.len())])?;
        }
    }
    Ok(())
}

/// The second fit of the stacked rows and the sampling part of Prentice's and
/// Self-Prentice's variance.
fn prentice(
    sample: &Sample<'_>,
    control: Control,
    cohort: f64,
    counts: Counts,
    entry_at_failure: bool,
) -> Result<Pieces> {
    let p = sample.x.ncols();
    let rows = sample.rows();
    let mut first: Option<CoxModelFit> = None;
    if entry_at_failure {
        let mut times: Vec<f64> = (0..rows)
            .filter(|&i| sample.event[i])
            .map(|i| sample.stop[i])
            .collect();
        times.sort_by(f64::total_cmp);
        times.dedup();
        if times.len() < 2 {
            return Err(Error::input(
                "data: Prentice's method needs at least two distinct failure times",
            ));
        }
        let delta = times
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::INFINITY, f64::min)
            / 2.0;
        let all: Vec<(usize, bool, f64, f64)> =
            (0..rows).map(|i| (i, sample.event[i], 0.0, 1.0)).collect();
        let mut stack = sample.stack(&all, false, false);
        for (k, &(i, ..)) in all.iter().enumerate() {
            if sample.event[i] && !sample.subcohort[i] {
                stack.start[k] = sample.stop[i] - delta;
            }
        }
        first = Some(stack.fit(control, None, ITER_DEFAULT)?);
    }
    let (stacked, _) = sample.stacked(None);
    let init: Option<Vec<f64>> = first.as_ref().map(|f| {
        f.coefficients
            .iter()
            .map(|c| c.unwrap_or(0.0))
            .collect::<Vec<f64>>()
    });
    let second = stacked.fit(
        control,
        init.as_deref(),
        if entry_at_failure {
            ITER_PRENTICE_VARIANCE
        } else {
            ITER_DEFAULT
        },
    )?;
    let mut fits = vec![&second];
    let coefficients_fit = if let Some(f) = first.as_ref() {
        fits.insert(0, f);
        f
    } else {
        &second
    };
    if coefficients_fit.rank == 0 || second.rank == 0 {
        return Ok(no_estimate(p, None, &fits));
    }
    let kept = kept_columns(&second);
    if kept_columns(coefficients_fit) != kept {
        return Err(Error::input(
            "x: the first and the second fit alias different columns",
        ));
    }
    let model = kept_covariance(&second, &kept)?;
    let dfbeta = stacked.residuals(control, &second, ResidualKind::Dfbeta)?;
    let subcohort_rows: Vec<usize> = (counts.cases..stacked.stop.len()).collect();
    let db = gather(&dfbeta, &kept, &subcohort_rows)?;
    let factor = 1.0 - counts.subcohort as f64 / cohort;
    let mut phase_two = weighted_crossproduct(&db, |_| 1.0, kept.len());
    for v in &mut phase_two {
        *v *= factor;
    }
    Ok(pieces_from(
        coefficients_fit.coefficients.clone(),
        &kept,
        &model,
        &phase_two,
        None,
        coefficients_fit.iterations,
        fits.iter().all(|f| f.converged),
    ))
}

fn lin_ying(
    sample: &Sample<'_>,
    control: Control,
    opts: &CaseCohortOptions<'_>,
    counts: Counts,
) -> Result<Pieces> {
    let p = sample.x.ncols();
    let rows = sample.rows();
    let cohort = opts.cohort_size[0];
    let non_cases = counts.subcohort - counts.cases_inside;
    if non_cases == 0 {
        return Err(Error::input(
            "subcohort: Lin-Ying's weights need a subcohort member that is not a case",
        ));
    }
    let cases = counts.cases as f64;
    if cohort <= cases {
        return Err(Error::input("cohort_size: must exceed the number of cases"));
    }
    let weight = (cohort - cases) / non_cases as f64;
    let all: Vec<(usize, bool, f64, f64)> = (0..rows)
        .map(|i| {
            let w = if sample.event[i] { 1.0 } else { weight };
            (i, sample.event[i], w.ln(), w)
        })
        .collect();
    let stack = sample.stack(&all, false, true);
    let model_fit = stack.fit(control, None, ITER_DEFAULT)?;
    if model_fit.rank == 0 {
        return Ok(no_estimate(p, None, &[&model_fit]));
    }
    let kept = kept_columns(&model_fit);
    let q = kept.len();
    let model = kept_covariance(&model_fit, &kept)?;
    let dfbeta = stack.residuals(control, &model_fit, ResidualKind::Dfbeta)?;
    let everyone: Vec<usize> = (0..rows).collect();
    let db = gather(&dfbeta, &kept, &everyone)?;
    let noncase_rows: Vec<usize> = (0..rows).filter(|&i| !sample.event[i]).collect();
    let mut db0: Vec<Vec<f64>> = noncase_rows.iter().map(|&i| db[i].clone()).collect();
    let mean: Vec<f64> = (0..q)
        .map(|a| db0.iter().map(|r| r[a]).sum::<f64>() / db0.len() as f64)
        .collect();
    for row in &mut db0 {
        for a in 0..q {
            row[a] -= mean[a];
        }
    }
    let factor = 1.0 - non_cases as f64 / (cohort - cases);
    let mut phase_two = weighted_crossproduct(&db0, |_| 1.0, q);
    for v in &mut phase_two {
        *v *= factor;
    }
    let mut out = pieces_from(
        model_fit.coefficients.clone(),
        &kept,
        &model,
        &phase_two,
        None,
        model_fit.iterations,
        model_fit.converged,
    );
    if opts.robust {
        let mut robust = weighted_crossproduct(&db, |i| 1.0 / all[i].3, q);
        for k in 0..q * q {
            robust[k] += phase_two[k];
        }
        out.covariance = embed(&robust, &kept, p);
    }
    Ok(out)
}

/// The dense strata `0..J` of the stratum codes, in ascending order.
fn dense_strata(stratum: &[u32]) -> (Vec<usize>, usize) {
    let mut codes = stratum.to_vec();
    codes.sort_unstable();
    codes.dedup();
    let dense = stratum
        .iter()
        .map(|c| codes.binary_search(c).expect("code is present"))
        .collect();
    (dense, codes.len())
}

fn borgan(
    sample: &Sample<'_>,
    control: Control,
    opts: &CaseCohortOptions<'_>,
    stratum: &[u32],
) -> Result<Pieces> {
    let second = opts.method == CaseCohortMethod::BorganII;
    let p = sample.x.ncols();
    let rows = sample.rows();
    let (dense, strata) = dense_strata(stratum);
    if opts.cohort_size.len() != strata {
        return Err(Error::input(format!(
            "cohort_size: {} values for {strata} strata",
            opts.cohort_size.len()
        )));
    }
    let mut in_stratum = vec![0usize; strata];
    let mut members = vec![0usize; strata];
    let mut non_cases = vec![0usize; strata];
    let mut cases = vec![0usize; strata];
    for i in 0..rows {
        let j = dense[i];
        in_stratum[j] += 1;
        if sample.subcohort[i] {
            members[j] += 1;
            if !sample.event[i] {
                non_cases[j] += 1;
            }
        }
        if sample.event[i] {
            cases[j] += 1;
        }
    }
    for j in 0..strata {
        if opts.cohort_size[j] < in_stratum[j] as f64 {
            return Err(Error::input(format!(
                "cohort_size: stratum {j} has {} rows and a cohort size of {}",
                in_stratum[j], opts.cohort_size[j]
            )));
        }
    }
    // The weight of a subcohort row by stratum, the number the stratum
    // covariance is divided by (less one), and the size the stratum's
    // contribution is multiplied by.
    let (weight, divisor, size): (Vec<f64>, Vec<usize>, Vec<f64>) = {
        let mut weight = Vec::new();
        let mut divisor = Vec::new();
        let mut size = Vec::new();
        for j in 0..strata {
            let (count, n) = if second {
                (non_cases[j], opts.cohort_size[j] - cases[j] as f64)
            } else {
                (members[j], opts.cohort_size[j])
            };
            if count == 0 || n <= 0.0 {
                return Err(Error::input(format!(
                    "stratum: stratum {j} has no {} or no cohort members to weight",
                    if second {
                        "subcohort non-case"
                    } else {
                        "subcohort member"
                    }
                )));
            }
            weight.push(n / count as f64);
            divisor.push(count);
            size.push(n);
        }
        (weight, divisor, size)
    };
    let (stack, kept_rows, model_fit) = if second {
        let all: Vec<(usize, bool, f64, f64)> = (0..rows)
            .map(|i| {
                let w = if sample.event[i] {
                    1.0
                } else {
                    weight[dense[i]]
                };
                (i, sample.event[i], 0.0, w)
            })
            .collect();
        let stack = sample.stack(&all, true, false);
        let kept_rows: Vec<usize> = (0..rows).filter(|&i| !sample.event[i]).collect();
        let model_fit = stack.fit(control, None, ITER_BORGAN)?;
        (stack, kept_rows, model_fit)
    } else {
        let w = |i: usize| weight[dense[i]];
        let (stack, cases_stacked) = sample.stacked(Some(&w));
        let kept_rows: Vec<usize> = (cases_stacked..stack.stop.len()).collect();
        let model_fit = stack.fit(control, None, ITER_BORGAN)?;
        (stack, kept_rows, model_fit)
    };
    if model_fit.rank == 0 {
        return Ok(no_estimate(p, Some(strata), &[&model_fit]));
    }
    let kept = kept_columns(&model_fit);
    let q = kept.len();
    let vn = kept_covariance(&model_fit, &kept)?;
    let scores = stack.residuals(control, &model_fit, ResidualKind::Score)?;
    let score_rows = gather(&scores, &kept, &kept_rows)?;
    // The stratum of each score row: the subcohort rows in input order.
    let row_stratum: Vec<usize> = if second {
        kept_rows.iter().map(|&i| dense[i]).collect()
    } else {
        (0..rows)
            .filter(|&i| sample.subcohort[i])
            .map(|i| dense[i])
            .collect()
    };
    let mut delta = vec![0.0; q * q];
    let mut contributions = Vec::with_capacity(strata);
    for j in 0..strata {
        let idx: Vec<usize> = (0..score_rows.len())
            .filter(|&k| row_stratum[k] == j)
            .collect();
        let m = idx.len() as f64;
        let mean: Vec<f64> = (0..q)
            .map(|a| idx.iter().map(|&k| score_rows[k][a]).sum::<f64>() / m)
            .collect();
        let centred: Vec<Vec<f64>> = idx
            .iter()
            .map(|&k| (0..q).map(|a| score_rows[k][a] - mean[a]).collect())
            .collect();
        let mut s = weighted_crossproduct(&centred, |_| 1.0, q);
        if divisor[j] < 2 {
            if weight[j] - 1.0 != 0.0 {
                return Err(Error::input(format!(
                    "stratum: stratum {j} has one member and a weight other than 1, so its covariance is not defined"
                )));
            }
            s.fill(0.0);
        } else {
            let d = (divisor[j] - 1) as f64;
            for v in &mut s {
                *v /= d;
            }
        }
        for k in 0..q * q {
            delta[k] += (weight[j] - 1.0) * size[j] * s[k];
        }
        contributions.push(s);
    }
    let left = mat_mul(&vn, &delta, q);
    let phase_two = mat_mul(&left, &vn, q);
    // The allocation: N_j sqrt(diag(Vn S_j Vn)), scaled over the strata.
    let mut allocation = vec![vec![0.0; q]; strata];
    for (j, s) in contributions.iter().enumerate() {
        let t = mat_mul(&mat_mul(&vn, s, q), &vn, q);
        for a in 0..q {
            allocation[j][a] = size[j] * t[a * q + a].sqrt();
        }
    }
    let mut optimal = vec![None; strata * p];
    for (a, &col) in kept.iter().enumerate() {
        let total: f64 = (0..strata).map(|j| allocation[j][a]).sum();
        for j in 0..strata {
            optimal[j * p + col] = Some(allocation[j][a] / total);
        }
    }
    Ok(pieces_from(
        model_fit.coefficients.clone(),
        &kept,
        &vn,
        &phase_two,
        Some(optimal),
        model_fit.iterations,
        model_fit.converged,
    ))
}
