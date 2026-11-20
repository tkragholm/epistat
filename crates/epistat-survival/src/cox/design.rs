//! Cox regression on a design matrix, with Breslow or Efron ties, delayed
//! entry, strata, case weights and a cluster-robust variance.
//!
//! The fit follows `survival::coxph` step by step: the same standardisation of
//! the covariates, the same factorisation of the information matrix, the same
//! step halving and the same stopping rule, so that the iterations, and not
//! only the limit, can be compared with R's.
//!
//! # Algorithm
//!
//! Row `i` has a covariate vector `x_i` of length `p`, a case weight `w_i > 0`,
//! an entry time `a_i`, an exit time `t_i` and an event indicator `delta_i`.
//! Rows are in strata, and a stratum has its own baseline hazard. At an event
//! time `t` the risk set `R(t)` of a stratum is the rows with
//! `a_j < t <= t_j`. With `theta_j = exp(beta' x_j)` the weighted sums over the
//! risk set are
//!
//! ```text
//! S_0(t) = sum over R(t) of w_j theta_j,
//! S_1(t) = sum over R(t) of w_j theta_j x_j,
//! S_2(t) = sum over R(t) of w_j theta_j x_j x_j'.
//! ```
//!
//! Let `D(t)` be the events at `t`, `d` their number and `w_bar` the mean of
//! their weights, and let `E_0, E_1, E_2` be the sums above taken over `D(t)`
//! only. Breslow's approximation \[1\] gives every tied event the whole risk set
//! as its denominator. Efron's approximation \[2\] gives the `k`-th tied event
//! the risk set less `k / d` of the tied events, `k = 0, ..., d - 1`:
//!
//! ```text
//! Breslow:  one term,   weight sum of w over D(t),  denominators S_r
//! Efron:    d terms,    weight w_bar each,          denominators S_r - (k / d) E_r
//! ```
//!
//! For a term with weight `u` and denominators `D_0, D_1, D_2`, the log
//! partial likelihood \[3\] gains `u (beta' x - log D_0)` over the covariates of
//! the events, so
//!
//! ```text
//! l(beta) = sum over t of [ sum over D(t) of w beta' x  -  sum over terms of u log D_0 ]
//! U(beta) = sum over t of [ sum over D(t) of w x  -  sum over terms of u D_1 / D_0 ]
//! I(beta) = sum over t, terms of u [ D_2 / D_0  -  (D_1 / D_0)(D_1 / D_0)' ].
//! ```
//!
//! Strata add their terms. A stratum with no event adds nothing.
//!
//! # Newton-Raphson with step halving
//!
//! The covariates are centred at their weighted mean and divided by their
//! weighted mean absolute deviation, as `coxph` does, and the iteration runs on
//! these. A column that is constant over the rows is set to zero. Starting at
//! `beta = 0`, with `l_0` the log partial likelihood there:
//!
//! 1. The trial point is `beta + I^{-1} U`, with `I` and `U` taken at the last
//!    accepted `beta`. The first trial is the first Newton step.
//! 2. The log partial likelihood `l` at the trial point is computed. When
//!    `|1 - l_prev / l| <= 1e-9` and the trial was not a halving, the fit has
//!    converged at the trial point. This is `coxph.control`'s `eps`.
//! 3. Otherwise, when `l` is below the last accepted value, or is not finite,
//!    the trial point moves half way back to the last accepted `beta` and step
//!    2 is repeated. When it is not, the trial point is accepted, and the next
//!    trial is a Newton step from it.
//! 4. After 20 trial points without convergence the fit stops, with
//!    `converged` false. This is `coxph.control`'s `iter.max`. `iterations`
//!    counts the trial points, as `iter` does in a `coxph` fit.
//!
//! The Newton step solves `I s = U` by the factorisation `I = L D L'` of
//! `coxph`. A pivot at most `1.8e-12` (`.Machine$double.eps^0.75`, `toler.chol`)
//! of the largest diagonal element of `I` is treated as zero, and its column is
//! removed from the solve. A column removed this way is aliased with the
//! columns before it, as when `coxph` reports `NA`, and its coefficient and
//! standard error are `None`. The standard errors come from the diagonal of
//! the inverse of `I` over the other columns, with `1 / sqrt` of the scale
//! undone, so they are on the scale of the columns given.
//!
//! # Covariance matrices
//!
//! `covariance` is the whole inverse `V`, not only its diagonal, on the scale
//! of the columns given: entry `(a, b)` is `V_ab f_a f_b` with `f` the factor
//! that undoes the scaling of the columns. Row-major, `p * p` entries, and
//! `None` in every entry whose row or column is an aliased coefficient, as the
//! standard errors are. It is exactly symmetric. The standard errors are
//! computed as `sqrt(V_aa) f_a`, which differs from the square root of the
//! diagonal entry in the last bits.
//!
//! # Cluster-robust variance
//!
//! With cluster codes the robust variance is the grouped sandwich estimate of
//! Lin and Wei \[4\]:
//!
//! ```text
//! V_robust = V (sum over clusters g of U_g U_g') V,    V = I(beta_hat)^{-1}
//! ```
//!
//! with no small-sample factor, as `coxph` computes it. `U_g` is the sum within
//! cluster `g` of the score residuals, which are defined below. The sum of the
//! cross-products `U_g U_g'` is the meat of the sandwich. It is formed by
//! `epistat_inference::sandwich::meat` from the matrix with one row `U_g` per
//! cluster, and `V M V` is then formed here, in a fixed order and averaged with
//! its transpose, since `V` has zero rows for aliased columns and is not the
//! inverse of a matrix that `epistat-inference` could factorise. The
//! covariance of `epistat_inference::influence` is not used: it multiplies by
//! `G / (G - 1)` and divides by `n^2`, which `coxph` does not. With a
//! cluster code that no row carries, the cluster has no row and adds nothing.
//! `robust_covariance` is the whole matrix `V M V` on the scale of the columns
//! given, laid out as `covariance`.
//!
//! # Score residuals
//!
//! [`cox_score_residuals`] returns, for every row, the derivative of the log
//! partial likelihood with respect to each coefficient through the case weight
//! of the row, on the scale of the columns given. A row with an event
//! contributes `w_j (x_j - m)` with `m` the mean over the terms at its time of
//! `D_1 / D_0`. Every row contributes `-w_j theta_j sum_t (x_j h_t - c_t)` over
//! the event times it is at risk at, where `h_t` is the sum of `u / D_0` over
//! the terms at `t` and `c_t` the sum of `u D_1 / D_0^2`. The sums over the
//! event times are differences of running sums over the times of the stratum.
//! Under Efron's approximation a tied event is at risk in only part of the
//! terms of its own time, and a final correction adds that back \[5\]. The
//! residuals add up to the score, so at the estimate their sum is zero up to
//! the stopping rule. They contain the case weight. `residuals(fit, "score")`
//! of `survival` leaves it out by default, and `weighted = TRUE` puts it in,
//! so that column divided by the weights of the rows is R's default. The
//! residual of a row whose weight is zero, and of a row in a stratum without
//! events, is zero.
//!
//! The residuals are a separate function and not a field of the fit because
//! they are `rows * p` numbers, which at `10^7` rows is `80` MB per column,
//! and most fits do not need them. The robust variance does not hold them
//! either: it sums them by cluster as they are produced. The function repeats
//! the sort of the rows by stratum and exit time, which costs a sort of the
//! rows, and takes the coefficients as an argument, so it also gives the
//! residuals at a point other than the estimate.
//!
//! # Results that are not numbers
//!
//! The estimate is `None`, with zero iterations and `converged` false, when no
//! row has a positive weight, no row has an event, every column is aliased, the
//! information matrix is not positive semi-definite, or the log partial
//! likelihood is not finite at `beta = 0`.
//!
//! The maximum of the partial likelihood does not exist when a column
//! separates the events, for example when every row with `x = 1` is censored.
//! The likelihood then rises towards a limit, the coefficient grows without
//! bound, and `coxph` stops when the likelihood stops changing and warns that
//! the coefficient may be infinite. It warns for column `j` when, at the end,
//! the Newton step `|(I^{-1} U)_j|` on the scaled covariates exceeds
//! `1e-9 + sqrt(1e-9)`. The fit does the same test, lists those columns in
//! `monotone_columns`, and reports every estimate and standard error as `None`.
//! The other coefficients are conditional on a value of the diverging one that
//! only the stopping rule chose, so they are not estimates either. The log
//! partial likelihood is `None` as well, and `converged` is false. The
//! iteration count is that of the run, which `coxph` also reports.
//!
//! The robust standard errors are `None` with fewer than two clusters, where
//! the sum of residuals within the one cluster is the score and the variance
//! is zero by construction.
//!
//! # Invalid input
//!
//! Columns of different lengths, a covariate, weight or exit time that is not
//! finite, a negative weight, an entry time that is NaN or not before the exit
//! time, a design without columns, and more rows than a `u32` addresses are
//! errors. Rows with a weight of zero are dropped before anything else, so
//! `n_rows` counts the rows fitted. Without delayed entry, pass negative
//! infinity as every entry time.
//!
//! # Agreement with R
//!
//! The tests compare coefficients, both standard errors, both covariance
//! matrices, the score residuals with and without the case weight, the log
//! partial likelihood and the iteration count with `survival::coxph` for
//! Breslow and Efron ties, delayed entry, strata, weights, clusters, aliased
//! columns and a diverging coefficient. Two implementations of one Newton
//! iteration that take the same steps differ by the error of the solve, which
//! is about the condition number of the information matrix on the scaled
//! covariates times the machine epsilon. The tolerance is 8 times that for a
//! coefficient and a standard error. The covariance tolerance is `p` times
//! that, since each entry of a row of an inverse carries it, and the robust
//! covariance twice as much again, since the error of `V` enters twice. The
//! tolerance of a residual adds `n` times the machine epsilon, the error of a
//! running sum over at most `n` event times, to `p` times the 8 above.
//!
//! # References
//!
//! 1. Breslow N. Covariance analysis of censored survival data. Biometrics
//!    1974;30(1):89-99. doi:10.2307/2529620
//! 2. Efron B. The efficiency of Cox's likelihood function for censored data.
//!    J Am Stat Assoc 1977;72(359):557-65. doi:10.1080/01621459.1977.10480613
//! 3. Cox DR. Regression models and life-tables. J R Stat Soc Series B
//!    1972;34(2):187-220. doi:10.1111/j.2517-6161.1972.tb00899.x
//! 4. Lin DY, Wei LJ. The robust inference for the Cox proportional hazards
//!    model. J Am Stat Assoc 1989;84(408):1074-8.
//!    doi:10.1080/01621459.1989.10478874
//! 5. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000.

// Rows and clusters are counted in `u32`; the entry point refuses more rows
// than that.
#![expect(
    clippy::cast_possible_truncation,
    reason = "the entry point checks that the row count fits in u32"
)]

use epistat_inference::sandwich::meat;
use faer::{Mat, MatRef};

use super::Ties;
use crate::check::{check_lengths, check_row_count, refuse_non_finite};
use crate::error::{Error, Result};

/// Largest number of trial points, `iter.max` of `coxph.control`.
const MAX_ITERATIONS: u32 = 20;
/// Relative change of the log partial likelihood at convergence, `eps`.
const EPS: f64 = 1e-9;
/// Smallest pivot of the information matrix relative to its largest diagonal
/// element, `.Machine$double.eps^0.75`, which is `2^-39`.
const TOLER_CHOL: f64 = 1.818_989_403_545_856_5e-12;
/// `sqrt(EPS)`, added to `EPS` in the test for a diverging coefficient.
const TOLER_INF: f64 = 3.162_277_660_168_379_5e-5;

/// The rows of a Cox fit on a design matrix. One entry per row, all of the same
/// length.
#[derive(Debug, Clone, Copy)]
pub struct CoxDesign<'a> {
    /// The covariates, one row per observation and one column per covariate.
    /// There is no intercept column: a column that is constant over the rows is
    /// aliased.
    pub x: MatRef<'a, f64>,
    /// Whether the row ends in the event of interest.
    pub is_event: &'a [bool],
    /// Entry time. Minus infinity when there is no delayed entry.
    pub start: &'a [f64],
    /// Exit time.
    pub stop: &'a [f64],
    /// Case weight. Rows with a weight of zero are dropped.
    pub weight: &'a [f64],
    /// Stratum codes. Strata are summed in code order.
    pub stratum: &'a [u32],
    /// How tied events enter the partial likelihood.
    pub ties: Ties,
    /// Cluster codes for the robust variance, or `None` for the model-based
    /// variance alone.
    pub cluster: Option<&'a [u32]>,
}

/// One Cox fit on a design matrix. See [`fit_cox`] for when entries are `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct CoxModelFit {
    /// One coefficient per column of the design, `None` for an aliased column
    /// and for every column when the fit is degenerate or a coefficient
    /// diverges.
    pub coefficients: Vec<Option<f64>>,
    /// The model-based standard errors, `None` where the coefficient is.
    pub standard_errors: Vec<Option<f64>>,
    /// The model-based covariance matrix of the coefficients, row-major with
    /// `p * p` entries on the scale of the columns given. An entry is `None`
    /// where its row or its column is a coefficient that is `None`.
    pub covariance: Vec<Option<f64>>,
    /// Standard errors from the cluster-robust variance. `None` when no
    /// cluster was given or there are fewer than two clusters.
    pub robust_standard_errors: Option<Vec<Option<f64>>>,
    /// The cluster-robust covariance matrix, laid out as `covariance`. `None`
    /// exactly when `robust_standard_errors` is, and with every entry `None`
    /// when clusters were given and the fit has no robust variance.
    pub robust_covariance: Option<Vec<Option<f64>>>,
    /// The log partial likelihood at the estimate, `None` when there is no
    /// estimate.
    pub log_partial_likelihood: Option<f64>,
    /// The log partial likelihood at zero, `None` when it is not finite or
    /// there are no events.
    pub null_log_partial_likelihood: Option<f64>,
    /// The summed weight of the events among the rows fitted.
    pub n_events: f64,
    /// The rows fitted, after dropping rows whose weight is zero.
    pub n_rows: usize,
    /// The columns that are not aliased, 0 when there is no estimate.
    pub rank: usize,
    /// The trial points computed, as `iter` in a `coxph` fit.
    pub iterations: u32,
    /// Whether the log partial likelihood stopped changing within the
    /// iteration limit.
    pub converged: bool,
    /// The columns whose coefficient diverges, which `coxph` warns about as
    /// possibly infinite. Empty when none do.
    pub monotone_columns: Vec<usize>,
}

impl CoxModelFit {
    fn degenerate(p: usize, n_events: f64, n_rows: usize, with_clusters: bool) -> Self {
        Self {
            coefficients: vec![None; p],
            standard_errors: vec![None; p],
            covariance: vec![None; p * p],
            robust_standard_errors: with_clusters.then(|| vec![None; p]),
            robust_covariance: with_clusters.then(|| vec![None; p * p]),
            log_partial_likelihood: None,
            null_log_partial_likelihood: None,
            n_events,
            n_rows,
            rank: 0,
            iterations: 0,
            converged: false,
            monotone_columns: Vec::new(),
        }
    }
}

/// Newton-Raphson with step halving on the weighted partial likelihood of a
/// design matrix, Breslow or Efron, as `survival::coxph` fits it.
///
/// See the module documentation for the method, the robust variance, aliased
/// columns and diverging coefficients.
///
/// # Errors
///
/// [`Error::Input`] when the columns differ in length, the design has no
/// column, a covariate, weight or exit time is not finite, a weight is
/// negative, an entry time is NaN or not before its exit time, or there are
/// more rows than a `u32` addresses.
pub fn fit_cox(data: &CoxDesign<'_>) -> Result<CoxModelFit> {
    check_design(data)?;
    Ok(fit_checked(data))
}

/// The checks every entry point makes on a design.
fn check_design(data: &CoxDesign<'_>) -> Result<()> {
    let rows = data.x.nrows();
    check_lengths(&[
        ("x", rows),
        ("is_event", data.is_event.len()),
        ("start", data.start.len()),
        ("stop", data.stop.len()),
        ("weight", data.weight.len()),
        ("stratum", data.stratum.len()),
        ("cluster", data.cluster.map_or(rows, <[u32]>::len)),
    ])?;
    check_row_count(rows)?;
    if data.x.ncols() == 0 {
        return Err(Error::input("x has no columns"));
    }
    refuse_non_finite(data.stop, "stop")?;
    refuse_non_finite(data.weight, "weight")?;
    if let Some(index) = data.weight.iter().position(|&w| w < 0.0) {
        return Err(Error::input(format!("weight: row {index} is negative")));
    }
    let non_finite = (0..rows)
        .flat_map(|i| (0..data.x.ncols()).map(move |j| (i, j)))
        .filter(|&(i, j)| !data.x[(i, j)].is_finite())
        .count();
    if non_finite > 0 {
        return Err(Error::input(format!(
            "x: {non_finite} non-finite values; drop or resolve them before estimating"
        )));
    }
    for i in 0..rows {
        if data.weight[i] > 0.0 && (data.start[i].is_nan() || data.start[i] >= data.stop[i]) {
            return Err(Error::input(format!(
                "start: row {i} has an entry time that is NaN or not before its exit time"
            )));
        }
    }
    Ok(())
}

/// One stratum's rows in descending exit order.
struct Stratum {
    /// Standardised covariates, `p` per row, in descending exit order.
    x: Vec<f64>,
    weight: Vec<f64>,
    stop: Vec<f64>,
    start: Vec<f64>,
    is_event: Vec<bool>,
    /// Index in the input of each row.
    row: Vec<u32>,
    /// Cluster code of each row, empty without clusters.
    cluster: Vec<u32>,
    /// Positions in descending entry order, and the entry times in that order.
    by_start: Vec<u32>,
    start_sorted: Vec<f64>,
    /// Distinct event times, descending.
    event_times: Vec<f64>,
    /// The positions of the rows with their event at each time, in the order
    /// of `event_times`: `dying_rows[dying_offset[k]..dying_offset[k + 1]]`.
    dying_rows: Vec<u32>,
    dying_offset: Vec<usize>,
}

/// The sums `sum c x^r` for `r = 0, 1, 2`, with the lower triangle of `r = 2`.
struct Moments {
    m0: f64,
    m1: Vec<f64>,
    m2: Vec<f64>,
}

impl Moments {
    fn new(p: usize) -> Self {
        Self {
            m0: 0.0,
            m1: vec![0.0; p],
            m2: vec![0.0; p * p],
        }
    }

    fn clear(&mut self) {
        self.m0 = 0.0;
        self.m1.fill(0.0);
        self.m2.fill(0.0);
    }

    fn add(&mut self, c: f64, x: &[f64]) {
        let p = x.len();
        self.m0 += c;
        for a in 0..p {
            let cx = c * x[a];
            self.m1[a] += cx;
            for (b, xb) in x.iter().enumerate().take(a + 1) {
                self.m2[a * p + b] += cx * xb;
            }
        }
    }
}

/// The risk-set sums and the sums over the tied events at one event time.
struct Sums {
    risk: Moments,
    dying: Moments,
}

impl Stratum {
    /// Visit every event time in descending order with the sums at the
    /// per-row weights `risk`, which are `w exp(beta' x)`.
    fn sweep(
        &self,
        p: usize,
        risk: &[f64],
        active: &mut Vec<bool>,
        sums: &mut Sums,
        mut visit: impl FnMut(usize, &Sums),
    ) {
        let size = self.stop.len();
        active.clear();
        active.resize(size, false);
        sums.risk.clear();
        let (mut entering, mut leaving) = (0usize, 0usize);
        for (position, &time) in self.event_times.iter().enumerate() {
            while entering < size && self.stop[entering] >= time {
                sums.risk
                    .add(risk[entering], &self.x[entering * p..(entering + 1) * p]);
                active[entering] = true;
                entering += 1;
            }
            while leaving < size && self.start_sorted[leaving] >= time {
                let row = self.by_start[leaving] as usize;
                if active[row] {
                    sums.risk.add(-risk[row], &self.x[row * p..(row + 1) * p]);
                    active[row] = false;
                }
                leaving += 1;
            }
            sums.dying.clear();
            for &row in
                &self.dying_rows[self.dying_offset[position]..self.dying_offset[position + 1]]
            {
                let row = row as usize;
                sums.dying.add(risk[row], &self.x[row * p..(row + 1) * p]);
            }
            visit(position, sums);
        }
    }

    /// The terms of the log partial likelihood at event time `position`, as
    /// `(weight, fraction)`: the term has denominator `S - fraction * E` and
    /// is multiplied by `weight`.
    fn terms(&self, position: usize, ties: Ties) -> impl Iterator<Item = (f64, f64)> + use<> {
        let range = self.dying_offset[position]..self.dying_offset[position + 1];
        let count = range.len();
        let total: f64 = self.dying_rows[range]
            .iter()
            .map(|&row| self.weight[row as usize])
            .sum();
        let (terms, weight, spread) = match ties {
            Ties::Breslow => (1, total, 0.0),
            Ties::Efron => (count, total / count as f64, 1.0 / count as f64),
        };
        (0..terms).map(move |k| (weight, k as f64 * spread))
    }
}

/// The log partial likelihood, the score and the information at one beta.
struct Evaluation {
    loglik: f64,
    score: Vec<f64>,
    /// `p` by `p`, symmetric.
    information: Vec<f64>,
}

struct Model {
    p: usize,
    ties: Ties,
    strata: Vec<Stratum>,
    n_clusters: usize,
}

impl Model {
    fn row_risk(stratum: &Stratum, p: usize, beta: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        let size = stratum.stop.len();
        let mut eta = Vec::with_capacity(size);
        let mut risk = Vec::with_capacity(size);
        for row in 0..size {
            let value: f64 = stratum.x[row * p..(row + 1) * p]
                .iter()
                .zip(beta)
                .map(|(x, b)| x * b)
                .sum();
            let weighted = stratum.weight[row] * value.exp();
            if !weighted.is_finite() {
                return None;
            }
            eta.push(value);
            risk.push(weighted);
        }
        Some((eta, risk))
    }

    /// The log partial likelihood, score and information at `beta`, or `None`
    /// when any of them is not finite.
    fn evaluate(&self, beta: &[f64]) -> Option<Evaluation> {
        let p = self.p;
        let mut loglik = 0.0f64;
        let mut score = vec![0.0f64; p];
        let mut information = vec![0.0f64; p * p];
        let mut active = Vec::new();
        let mut sums = Sums {
            risk: Moments::new(p),
            dying: Moments::new(p),
        };
        let mut d1 = vec![0.0f64; p];
        for stratum in &self.strata {
            let (eta, risk) = Self::row_risk(stratum, p, beta)?;
            let mut failed = false;
            stratum.sweep(p, &risk, &mut active, &mut sums, |position, at| {
                let range = stratum.dying_offset[position]..stratum.dying_offset[position + 1];
                for &row in &stratum.dying_rows[range] {
                    let row = row as usize;
                    loglik += stratum.weight[row] * eta[row];
                    for (total, x) in score.iter_mut().zip(&stratum.x[row * p..(row + 1) * p]) {
                        *total += stratum.weight[row] * x;
                    }
                }
                for (u, fraction) in stratum.terms(position, self.ties) {
                    let d0 = at.risk.m0 - fraction * at.dying.m0;
                    if d0 <= 0.0 || !d0.is_finite() {
                        failed = true;
                        return;
                    }
                    loglik -= u * d0.ln();
                    for a in 0..p {
                        d1[a] = at.risk.m1[a] - fraction * at.dying.m1[a];
                        score[a] -= u * d1[a] / d0;
                    }
                    for a in 0..p {
                        for b in 0..=a {
                            let d2 = at.risk.m2[a * p + b] - fraction * at.dying.m2[a * p + b];
                            information[a * p + b] += u * (d2 / d0 - (d1[a] / d0) * (d1[b] / d0));
                        }
                    }
                }
            });
            if failed {
                return None;
            }
        }
        for a in 0..p {
            for b in 0..a {
                information[b * p + a] = information[a * p + b];
            }
        }
        let finite = loglik.is_finite()
            && score.iter().all(|v| v.is_finite())
            && information.iter().all(|v| v.is_finite());
        finite.then_some(Evaluation {
            loglik,
            score,
            information,
        })
    }

    /// The score residuals summed within cluster, `p` per cluster.
    fn cluster_scores(&self, beta: &[f64]) -> Option<Vec<f64>> {
        let p = self.p;
        let mut totals = vec![0.0f64; self.n_clusters * p];
        self.score_rows(beta, |stratum, j, residual| {
            let cluster = stratum.cluster[j] as usize;
            for (total, r) in totals[cluster * p..(cluster + 1) * p]
                .iter_mut()
                .zip(residual)
            {
                *total += r;
            }
        })?;
        Some(totals)
    }

    /// Hand the score residual of every row to `emit`, as the stratum, the
    /// position in it and the `p` values, in stratum order and then in the
    /// order of the stratum. `None` when a risk score or a denominator is not
    /// finite or not positive.
    fn score_rows(
        &self,
        beta: &[f64],
        mut emit: impl FnMut(&Stratum, usize, &[f64]),
    ) -> Option<()> {
        let p = self.p;
        let mut residual = vec![0.0f64; p];
        let mut active = Vec::new();
        let mut sums = Sums {
            risk: Moments::new(p),
            dying: Moments::new(p),
        };
        for stratum in &self.strata {
            let (_, risk) = Self::row_risk(stratum, p, beta)?;
            let count = stratum.event_times.len();
            // Per event time: `hazard` is the sum of weight / d0 over the
            // terms, `centre` the sum of weight * d1 / d0^2, `late_hazard` and
            // `late_centre` the same two sums with the fraction as a factor,
            // and `mean` the mean of d1 / d0.
            let mut hazard = vec![0.0f64; count];
            let mut late_hazard = vec![0.0f64; count];
            let mut centre = vec![0.0f64; count * p];
            let mut late_centre = vec![0.0f64; count * p];
            let mut mean = vec![0.0f64; count * p];
            let mut failed = false;
            stratum.sweep(p, &risk, &mut active, &mut sums, |position, at| {
                let mut terms = 0.0f64;
                for (u, fraction) in stratum.terms(position, self.ties) {
                    let d0 = at.risk.m0 - fraction * at.dying.m0;
                    if d0 <= 0.0 || !d0.is_finite() {
                        failed = true;
                        return;
                    }
                    hazard[position] += u / d0;
                    late_hazard[position] += u * fraction / d0;
                    for a in 0..p {
                        let d1 = at.risk.m1[a] - fraction * at.dying.m1[a];
                        centre[position * p + a] += u * d1 / (d0 * d0);
                        late_centre[position * p + a] += u * fraction * d1 / (d0 * d0);
                        mean[position * p + a] += d1 / d0;
                    }
                    terms += 1.0;
                }
                for a in 0..p {
                    mean[position * p + a] /= terms;
                }
            });
            if failed {
                return None;
            }
            // Prefix sums over the descending event times.
            let mut hazard_sum = vec![0.0f64; count + 1];
            let mut centre_sum = vec![0.0f64; (count + 1) * p];
            for k in 0..count {
                hazard_sum[k + 1] = hazard_sum[k] + hazard[k];
                for a in 0..p {
                    centre_sum[(k + 1) * p + a] = centre_sum[k * p + a] + centre[k * p + a];
                }
            }
            for (j, &risk_j) in risk.iter().enumerate() {
                let x = &stratum.x[j * p..(j + 1) * p];
                // Event times t with start < t <= stop.
                let first = stratum
                    .event_times
                    .partition_point(|&t| t > stratum.stop[j]);
                let last = stratum
                    .event_times
                    .partition_point(|&t| t > stratum.start[j]);
                let span = hazard_sum[last] - hazard_sum[first];
                for a in 0..p {
                    let around = centre_sum[last * p + a] - centre_sum[first * p + a];
                    let mut value = -risk_j * (x[a] * span - around);
                    if stratum.is_event[j] {
                        // The row has its event at `stop`, an event time of the
                        // stratum.
                        value += stratum.weight[j] * (x[a] - mean[first * p + a]);
                        value += risk_j * (x[a] * late_hazard[first] - late_centre[first * p + a]);
                    }
                    residual[a] = value;
                }
                emit(stratum, j, &residual);
            }
        }
        Some(())
    }
}

/// The `L D L'` factorisation of `coxph`, in place on a symmetric matrix.
///
/// A pivot below `TOLER_CHOL` times the largest diagonal element is set to
/// zero together with the column of `L` under it. Returns the number of
/// pivots kept, or `None` when a pivot is not finite or is clearly negative.
fn factorise(a: &mut [f64], p: usize) -> Option<usize> {
    let largest = (0..p).map(|i| a[i * p + i]).fold(0.0f64, f64::max);
    let floor = if largest == 0.0 {
        TOLER_CHOL
    } else {
        largest * TOLER_CHOL
    };
    let mut rank = 0;
    for i in 0..p {
        let pivot = a[i * p + i];
        if !pivot.is_finite() || pivot < -8.0 * floor {
            return None;
        }
        if pivot < floor {
            a[i * p + i] = 0.0;
            for j in i + 1..p {
                a[j * p + i] = 0.0;
            }
        } else {
            rank += 1;
            for j in i + 1..p {
                let t = a[j * p + i] / pivot;
                a[j * p + i] = t;
                a[j * p + j] -= t * t * pivot;
                for k in j + 1..p {
                    a[k * p + j] -= t * a[k * p + i];
                }
            }
        }
    }
    Some(rank)
}

/// Solve `L D L' z = y` in place, with zero where `D` is zero.
fn solve_factor(a: &[f64], p: usize, y: &mut [f64]) {
    for i in 0..p {
        let mut t = y[i];
        for j in 0..i {
            t -= y[j] * a[i * p + j];
        }
        y[i] = t;
    }
    for i in (0..p).rev() {
        if a[i * p + i] == 0.0 {
            y[i] = 0.0;
        } else {
            let mut t = y[i] / a[i * p + i];
            for j in i + 1..p {
                t -= y[j] * a[j * p + i];
            }
            y[i] = t;
        }
    }
}

/// The inverse of the factorised matrix over its kept columns, with zero rows
/// and columns for the others, symmetric.
fn inverse_factor(a: &[f64], p: usize) -> Vec<f64> {
    let mut inverse = vec![0.0f64; p * p];
    for k in 0..p {
        if a[k * p + k] == 0.0 {
            continue;
        }
        let mut column = vec![0.0f64; p];
        column[k] = 1.0;
        solve_factor(a, p, &mut column);
        for (i, value) in column.iter().enumerate() {
            inverse[i * p + k] = *value;
        }
    }
    for i in 0..p {
        for j in 0..i {
            inverse[j * p + i] = inverse[i * p + j];
        }
    }
    inverse
}

/// The model on the scaled covariates, and the factor that scales each
/// column: a column's coefficient on the scale given is its coefficient on the
/// scaled column times the factor, and a column without variation has factor 0.
struct Prepared {
    model: Model,
    factor: Vec<f64>,
}

/// Centre and scale the covariates of the rows `kept`, which are not empty,
/// and sort them into strata.
fn prepare(input: &CoxDesign<'_>, kept: &[usize]) -> Prepared {
    let p = input.x.ncols();
    // Centre and scale as coxph does.
    let total_weight: f64 = kept.iter().map(|&i| input.weight[i]).sum();
    let mut centre = vec![0.0f64; p];
    let mut factor = vec![0.0f64; p];
    for a in 0..p {
        let first = input.x[(kept[0], a)];
        let constant = kept.iter().all(|&i| input.x[(i, a)] - first == 0.0);
        if constant {
            continue;
        }
        let mean: f64 = kept
            .iter()
            .map(|&i| input.weight[i] * input.x[(i, a)])
            .sum::<f64>()
            / total_weight;
        let deviation: f64 = kept
            .iter()
            .map(|&i| input.weight[i] * (input.x[(i, a)] - mean).abs())
            .sum::<f64>()
            / total_weight;
        centre[a] = mean;
        factor[a] = if deviation > 0.0 {
            1.0 / deviation
        } else {
            1.0
        };
    }

    let mut order = kept.to_vec();
    order.sort_by_key(|&i| input.stratum[i]);
    let mut strata = Vec::new();
    let mut begin = 0;
    while begin < order.len() {
        let code = input.stratum[order[begin]];
        let end = begin
            + order[begin..]
                .iter()
                .take_while(|&&i| input.stratum[i] == code)
                .count();
        let members = &mut order[begin..end];
        begin = end;
        if !members.iter().any(|&i| input.is_event[i]) {
            // A stratum without events has an unidentified baseline hazard and
            // adds nothing.
            continue;
        }
        // Stable, so rows with the same exit keep their input order.
        members.sort_by(|&a, &b| input.stop[b].total_cmp(&input.stop[a]));
        let size = members.len();
        let mut by_start: Vec<u32> = (0..size as u32).collect();
        by_start.sort_by(|&a, &b| {
            input.start[members[b as usize]].total_cmp(&input.start[members[a as usize]])
        });
        let start_sorted = by_start
            .iter()
            .map(|&row| input.start[members[row as usize]])
            .collect();
        let mut event_times = Vec::new();
        let mut dying_rows = Vec::new();
        let mut dying_offset = vec![0usize];
        for (position, &i) in members.iter().enumerate() {
            if !input.is_event[i] {
                continue;
            }
            if event_times.last() != Some(&input.stop[i]) {
                if !event_times.is_empty() {
                    dying_offset.push(dying_rows.len());
                }
                event_times.push(input.stop[i]);
            }
            dying_rows.push(position as u32);
        }
        dying_offset.push(dying_rows.len());
        let mut x = Vec::with_capacity(size * p);
        for &i in members.iter() {
            for a in 0..p {
                x.push(if factor[a] == 0.0 {
                    0.0
                } else {
                    (input.x[(i, a)] - centre[a]) * factor[a]
                });
            }
        }
        strata.push(Stratum {
            x,
            row: members.iter().map(|&i| i as u32).collect(),
            weight: members.iter().map(|&i| input.weight[i]).collect(),
            stop: members.iter().map(|&i| input.stop[i]).collect(),
            start: members.iter().map(|&i| input.start[i]).collect(),
            is_event: members.iter().map(|&i| input.is_event[i]).collect(),
            cluster: input
                .cluster
                .map(|cluster| members.iter().map(|&i| cluster[i]).collect())
                .unwrap_or_default(),
            by_start,
            start_sorted,
            event_times,
            dying_rows,
            dying_offset,
        });
    }
    let n_clusters = input.cluster.map_or(0, |cluster| {
        kept.iter()
            .map(|&i| cluster[i] as usize + 1)
            .max()
            .unwrap_or(0)
    });
    Prepared {
        model: Model {
            p,
            ties: input.ties,
            strata,
            n_clusters,
        },
        factor,
    }
}

fn fit_checked(input: &CoxDesign<'_>) -> CoxModelFit {
    let p = input.x.ncols();
    let with_clusters = input.cluster.is_some();
    let kept: Vec<usize> = (0..input.x.nrows())
        .filter(|&i| input.weight[i] > 0.0)
        .collect();
    if kept.is_empty() {
        return CoxModelFit::degenerate(p, 0.0, 0, with_clusters);
    }
    let n_rows = kept.len();
    let n_events: f64 = kept
        .iter()
        .filter(|&&i| input.is_event[i])
        .map(|&i| input.weight[i])
        .sum();
    if !kept.iter().any(|&i| input.is_event[i]) {
        return CoxModelFit::degenerate(p, n_events, n_rows, with_clusters);
    }

    let Prepared { model, factor } = prepare(input, &kept);
    let n_clusters = model.n_clusters;
    let undefined = || CoxModelFit::degenerate(p, n_events, n_rows, with_clusters);

    let mut beta = vec![0.0f64; p];
    let Some(start) = model.evaluate(&beta) else {
        return undefined();
    };
    let null_loglik = start.loglik;
    let mut accepted_loglik = start.loglik;
    let mut matrix = start.information;
    let Some(rank) = factorise(&mut matrix, p) else {
        return undefined();
    };
    if rank == 0 {
        return undefined();
    }
    let mut step = start.score;
    solve_factor(&matrix, p, &mut step);
    let mut trial: Vec<f64> = beta.iter().zip(&step).map(|(b, s)| b + s).collect();

    let mut halving = false;
    let mut iterations = 0u32;
    let mut converged = false;
    let mut last: Option<Evaluation> = None;
    for count in 1..=MAX_ITERATIONS {
        iterations = count;
        let evaluation = model.evaluate(&trial);
        if let Some(e) = &evaluation {
            if (1.0 - accepted_loglik / e.loglik).abs() <= EPS && !halving {
                converged = true;
                last = evaluation;
                break;
            }
        }
        if count == MAX_ITERATIONS {
            last = evaluation;
            break;
        }
        match evaluation {
            Some(e) if e.loglik >= accepted_loglik => {
                halving = false;
                accepted_loglik = e.loglik;
                beta.clone_from(&trial);
                matrix = e.information;
                if factorise(&mut matrix, p).is_none() {
                    return undefined();
                }
                step = e.score;
                solve_factor(&matrix, p, &mut step);
                for (t, s) in trial.iter_mut().zip(&step) {
                    *t += s;
                }
            }
            _ => {
                halving = true;
                for (t, b) in trial.iter_mut().zip(&beta) {
                    *t = (*t + b) / 2.0;
                }
            }
        }
    }
    let Some(last) = last else {
        let mut fit = undefined();
        fit.iterations = iterations;
        return fit;
    };

    let mut final_matrix = last.information.clone();
    let Some(rank) = factorise(&mut final_matrix, p) else {
        let mut fit = undefined();
        fit.iterations = iterations;
        return fit;
    };
    if rank == 0 {
        let mut fit = undefined();
        fit.iterations = iterations;
        return fit;
    }
    let aliased = |a: usize| final_matrix[a * p + a] == 0.0;

    // The coefficient of a column whose Newton step is still large when the
    // likelihood has stopped changing is not finite.
    let mut newton = last.score.clone();
    solve_factor(&final_matrix, p, &mut newton);
    let monotone_columns: Vec<usize> = if converged {
        (0..p)
            .filter(|&a| !aliased(a) && newton[a].abs() > EPS + TOLER_INF)
            .collect()
    } else {
        Vec::new()
    };
    if !monotone_columns.is_empty() {
        let mut fit = undefined();
        fit.iterations = iterations;
        fit.null_log_partial_likelihood = null_loglik.is_finite().then_some(null_loglik);
        fit.monotone_columns = monotone_columns;
        return fit;
    }

    let variance = inverse_factor(&final_matrix, p);
    let coefficients: Vec<Option<f64>> = (0..p)
        .map(|a| (!aliased(a)).then(|| trial[a] * factor[a]))
        .collect();
    let standard_errors: Vec<Option<f64>> = (0..p)
        .map(|a| (!aliased(a)).then(|| variance[a * p + a].sqrt() * factor[a]))
        .collect();
    let covariance = on_input_scale(&variance, &factor, &aliased);

    let robust = input.cluster.and_then(|cluster| {
        let mut present = vec![false; n_clusters];
        for &i in &kept {
            present[cluster[i] as usize] = true;
        }
        if present.iter().filter(|&&seen| seen).count() < 2 {
            return None;
        }
        let totals = model.cluster_scores(&trial)?;
        let sums = Mat::from_fn(n_clusters, p, |g, a| totals[g * p + a]);
        // Every cluster is a row of `sums` already, so the meat is the plain
        // cross-product of the rows, `sum_g U_g U_g'`.
        let meat = meat(sums.as_ref(), |_| 1.0, None)
            .expect("the meat of rows without clusters has no error to report");
        let sandwich = sandwich(&variance, &meat, p);
        Some((
            (0..p)
                .map(|a| (!aliased(a)).then(|| sandwich[a * p + a].sqrt() * factor[a]))
                .collect::<Vec<_>>(),
            on_input_scale(&sandwich, &factor, &aliased),
        ))
    });
    let (robust_standard_errors, robust_covariance) = match (with_clusters, robust) {
        (true, None) => (Some(vec![None; p]), Some(vec![None; p * p])),
        (_, Some((se, covariance))) => (Some(se), Some(covariance)),
        (false, None) => (None, None),
    };

    CoxModelFit {
        coefficients,
        standard_errors,
        covariance,
        robust_standard_errors,
        robust_covariance,
        log_partial_likelihood: Some(last.loglik),
        null_log_partial_likelihood: null_loglik.is_finite().then_some(null_loglik),
        n_events,
        n_rows,
        rank,
        iterations,
        converged,
        monotone_columns,
    }
}

/// `V M V'` for the symmetric `p` by `p` matrices `variance` and `meat`, in a
/// fixed order, averaged with its transpose so that it is exactly symmetric.
fn sandwich(variance: &[f64], meat: &Mat<f64>, p: usize) -> Vec<f64> {
    let mut left = vec![0.0f64; p * p];
    for a in 0..p {
        for b in 0..p {
            left[a * p + b] = (0..p).map(|c| variance[a * p + c] * meat[(c, b)]).sum();
        }
    }
    let mut full = vec![0.0f64; p * p];
    for a in 0..p {
        for b in 0..p {
            full[a * p + b] = (0..p).map(|c| left[a * p + c] * variance[c * p + b]).sum();
        }
    }
    let mut symmetric = vec![0.0f64; p * p];
    for a in 0..p {
        for b in 0..p {
            symmetric[a * p + b] = full[a * p + b].midpoint(full[b * p + a]);
        }
    }
    symmetric
}

/// A covariance matrix of the scaled coefficients on the scale of the columns
/// given, `None` in the rows and columns of an aliased column.
fn on_input_scale(
    scaled: &[f64],
    factor: &[f64],
    aliased: &impl Fn(usize) -> bool,
) -> Vec<Option<f64>> {
    let p = factor.len();
    (0..p * p)
        .map(|k| {
            let (a, b) = (k / p, k % p);
            (!aliased(a) && !aliased(b)).then(|| scaled[k] * (factor[a] * factor[b]))
        })
        .collect()
}

/// The score residual of every row at `coefficients`, one vector of `rows`
/// values per column, as `residuals(fit, type = "score")` of `survival`.
///
/// Row `i` of column `j` is the derivative of the log partial likelihood with
/// respect to the coefficient of column `j` through the case weight of row
/// `i`, on the scale of the columns given. The rows are in the order of the
/// input. A row whose weight is zero, and a row in a stratum without events,
/// has residuals of zero. A column whose coefficient is `None`, such as one
/// that is aliased, is `None`.
///
/// `coefficients` is normally `CoxModelFit::coefficients`. Any other value is
/// the residual at that point, with a column left out where it is `None`. The
/// `cluster` of `data` is not used.
///
/// The residuals contain the case weight; see the module documentation for
/// how they relate to `residuals(fit, "score")` of `survival`.
///
/// # Errors
///
/// [`Error::Input`] for the invalid input of [`fit_cox`], for `coefficients`
/// of another length than the design has columns, for a coefficient that is
/// not finite, and for coefficients at which a risk score or the denominator
/// of a partial-likelihood term is not finite and positive.
pub fn cox_score_residuals(
    data: &CoxDesign<'_>,
    coefficients: &[Option<f64>],
) -> Result<Vec<Option<Vec<f64>>>> {
    check_design(data)?;
    let (rows, p) = (data.x.nrows(), data.x.ncols());
    if coefficients.len() != p {
        return Err(Error::input(format!(
            "coefficients: {} values for {p} columns",
            coefficients.len()
        )));
    }
    if coefficients.iter().flatten().any(|c| !c.is_finite()) {
        return Err(Error::input("coefficients: a value is not finite"));
    }
    let mut columns: Vec<Option<Vec<f64>>> = coefficients
        .iter()
        .map(|c| c.map(|_| vec![0.0f64; rows]))
        .collect();
    let kept: Vec<usize> = (0..rows).filter(|&i| data.weight[i] > 0.0).collect();
    if kept.is_empty() {
        return Ok(columns);
    }
    let Prepared { model, factor } = prepare(data, &kept);
    // A column without variation has a scaled column of zeros and a residual
    // of zero whatever its coefficient.
    let beta: Vec<f64> = coefficients
        .iter()
        .zip(&factor)
        .map(|(c, &f)| if f == 0.0 { 0.0 } else { c.unwrap_or(0.0) / f })
        .collect();
    model
        .score_rows(&beta, |stratum, j, residual| {
            let row = stratum.row[j] as usize;
            for (a, column) in columns.iter_mut().enumerate() {
                if let Some(column) = column {
                    column[row] = if factor[a] == 0.0 {
                        0.0
                    } else {
                        residual[a] / factor[a]
                    };
                }
            }
        })
        .ok_or_else(|| {
            Error::input(
                "coefficients: a risk score or a denominator of the partial likelihood \
                 is not finite and positive at these values",
            )
        })?;
    Ok(columns)
}

#[cfg(test)]
mod tests {
    use faer::Mat;

    use super::{CoxDesign, cox_score_residuals, fit_cox};
    use crate::cox::Ties;

    const STOP: [f64; 8] = [1., 2., 2., 3., 4., 5., 6., 7.];
    const EVENT: [bool; 8] = [true, true, true, false, true, true, false, true];

    fn design(x: &Mat<f64>, cluster: Option<&[u32]>) -> super::Result<super::CoxModelFit> {
        fit_cox(&CoxDesign {
            x: x.as_ref(),
            is_event: &EVENT,
            start: &[f64::NEG_INFINITY; 8],
            stop: &STOP,
            weight: &[1.0; 8],
            stratum: &[0; 8],
            ties: Ties::Efron,
            cluster,
        })
    }

    fn column(values: [f64; 8]) -> Mat<f64> {
        Mat::from_fn(8, 1, |i, _| values[i])
    }

    #[test]
    fn one_cluster_has_no_robust_standard_error() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let fit = design(&x, Some(&[0; 8])).unwrap();
        assert!(fit.converged && fit.coefficients[0].is_some());
        assert_eq!(fit.robust_standard_errors, Some(vec![None]));
    }

    #[test]
    fn a_column_without_variation_is_aliased() {
        let x = Mat::from_fn(8, 2, |i, j| {
            if j == 0 {
                [0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3][i]
            } else {
                0.1
            }
        });
        let fit = design(&x, None).unwrap();
        assert_eq!(fit.rank, 1);
        assert!(fit.coefficients[0].is_some() && fit.coefficients[1].is_none());
        assert!(fit.standard_errors[1].is_none());
    }

    #[test]
    fn no_events_gives_no_estimate() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let fit = fit_cox(&CoxDesign {
            x: x.as_ref(),
            is_event: &[false; 8],
            start: &[f64::NEG_INFINITY; 8],
            stop: &STOP,
            weight: &[1.0; 8],
            stratum: &[0; 8],
            ties: Ties::Breslow,
            cluster: None,
        })
        .unwrap();
        assert_eq!(fit.coefficients, vec![None]);
        assert!(!fit.converged && fit.iterations == 0);
    }

    #[test]
    fn invalid_input_is_an_error() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, f64::NAN]);
        assert!(
            design(&x, None)
                .unwrap_err()
                .to_string()
                .starts_with("x: 1 non-finite")
        );
        let none = Mat::<f64>::zeros(8, 0);
        assert_eq!(
            design(&none, None).unwrap_err().to_string(),
            "x has no columns"
        );
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let mut weight = [1.0; 8];
        weight[3] = -1.0;
        let error = fit_cox(&CoxDesign {
            x: x.as_ref(),
            is_event: &EVENT,
            start: &[f64::NEG_INFINITY; 8],
            stop: &STOP,
            weight: &weight,
            stratum: &[0; 8],
            ties: Ties::Breslow,
            cluster: None,
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "weight: row 3 is negative");
        let error = fit_cox(&CoxDesign {
            x: x.as_ref(),
            is_event: &EVENT,
            start: &[0., 0., 2., 0., 0., 0., 0., 0.],
            stop: &STOP,
            weight: &[1.0; 8],
            stratum: &[0; 8],
            ties: Ties::Breslow,
            cluster: None,
        })
        .unwrap_err();
        assert!(error.to_string().starts_with("start: row 2"));
    }

    #[test]
    fn covariance_matrices_are_symmetric_and_hold_the_standard_errors() {
        let x = Mat::from_fn(8, 3, |i, j| match j {
            0 => [0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3][i],
            1 => [1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 1.0, 0.5][i],
            _ => 0.1,
        });
        let fit = design(&x, Some(&[0, 0, 1, 1, 2, 2, 3, 3])).unwrap();
        let robust = fit.robust_covariance.as_ref().unwrap();
        let robust_se = fit.robust_standard_errors.as_ref().unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert_eq!(fit.covariance[a * 3 + b], fit.covariance[b * 3 + a]);
                assert_eq!(robust[a * 3 + b], robust[b * 3 + a]);
                assert_eq!(
                    fit.covariance[a * 3 + b].is_none(),
                    a == 2 || b == 2,
                    "the constant column is aliased"
                );
            }
            if let (Some(variance), Some(se)) = (fit.covariance[a * 3 + a], fit.standard_errors[a])
            {
                assert!((variance.sqrt() - se).abs() <= 4.0 * f64::EPSILON * se);
            }
            if let (Some(variance), Some(se)) = (robust[a * 3 + a], robust_se[a]) {
                assert!((variance.sqrt() - se).abs() <= 4.0 * f64::EPSILON * se);
            }
        }
    }

    #[test]
    fn one_cluster_has_no_robust_covariance() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let fit = design(&x, Some(&[0; 8])).unwrap();
        assert_eq!(fit.robust_covariance, Some(vec![None]));
        assert!(design(&x, None).unwrap().robust_covariance.is_none());
    }

    #[test]
    fn score_residuals_add_up_to_the_score() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let fit = design(&x, None).unwrap();
        // At the estimate the score is zero, and at zero it is the sum of the
        // residuals there, which is the first Newton step times the information.
        let at_estimate = cox_score_residuals(&data(&x), &fit.coefficients).unwrap();
        let sum: f64 = at_estimate[0].as_ref().unwrap().iter().sum();
        assert!(sum.abs() < 1e-8, "{sum}");
        let at_zero = cox_score_residuals(&data(&x), &[Some(0.0)]).unwrap();
        let sum: f64 = at_zero[0].as_ref().unwrap().iter().sum();
        assert!(sum.abs() > 0.1);
    }

    #[test]
    fn score_residuals_of_dropped_columns_and_rows() {
        let x = Mat::from_fn(8, 2, |i, j| {
            if j == 0 {
                [0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3][i]
            } else {
                0.1
            }
        });
        let fit = design(&x, None).unwrap();
        let mut weight = [1.0; 8];
        weight[4] = 0.0;
        let residuals = cox_score_residuals(
            &CoxDesign {
                weight: &weight,
                ..data(&x)
            },
            &fit.coefficients,
        )
        .unwrap();
        assert!(residuals[1].is_none());
        assert_eq!(
            residuals[0].as_ref().unwrap()[4].to_bits(),
            0.0f64.to_bits()
        );
    }

    #[test]
    fn score_residuals_refuse_invalid_coefficients() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let wrong = cox_score_residuals(&data(&x), &[Some(0.1), Some(0.2)]).unwrap_err();
        assert_eq!(wrong.to_string(), "coefficients: 2 values for 1 columns");
        let nan = cox_score_residuals(&data(&x), &[Some(f64::NAN)]).unwrap_err();
        assert_eq!(nan.to_string(), "coefficients: a value is not finite");
        let huge = cox_score_residuals(&data(&x), &[Some(1e4)]).unwrap_err();
        assert!(huge.to_string().starts_with("coefficients: a risk score"));
        let none = cox_score_residuals(&data(&x), &[None]).unwrap();
        assert_eq!(none, vec![None]);
    }

    fn data(x: &Mat<f64>) -> CoxDesign<'_> {
        CoxDesign {
            x: x.as_ref(),
            is_event: &EVENT,
            start: &[f64::NEG_INFINITY; 8],
            stop: &STOP,
            weight: &[1.0; 8],
            stratum: &[0; 8],
            ties: Ties::Efron,
            cluster: None,
        }
    }
}
