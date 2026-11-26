//! The inverse-probability-of-censoring weighted Brier score, the index of
//! prediction accuracy and the time-dependent area under the ROC curve, for
//! predicted risks of a survival or a competing-risks event, as
//! `riskRegression::Score` and `timeROC::timeROC`.
//!
//! The predictions are supplied, one column per evaluation time `t`: the
//! predicted absolute risk of the event by `t` for every row, or for the AUC
//! any marker where a larger value means a higher risk. Nothing here fits a
//! model.
//!
//! # Weights
//!
//! Let `T_i` be the time of row `i` and `G` the censoring distribution. A row
//! with the event of interest by `t` has weight `1 / G(T_i-)`, a row with
//! another event by `t` has weight `1 / G(T_i-)`, a row censored by `t` has
//! weight 0, and a row still at risk after `t` has weight `1 / G(t)`. The
//! default `G` is the Kaplan-Meier estimate of the censoring times, in which a
//! death at a tied time leaves the risk set before the censoring does, as in
//! `prodlim` with `reverse = TRUE`, which `Score` calls. It is
//! [`Censoring::KaplanMeier`]. A Cox model for the censoring times, fitted here
//! by [`cox::fit`](crate::cox::fit), is [`Censoring::Cox`], `cens.model =
//! "cox"`: `G_i(s) = exp(-exp(x_i' b) L0(s))` with the Breslow baseline `L0`.
//! Weights from any other model can be supplied with [`Censoring::Supplied`],
//! and then the standard errors are the conservative ones: they treat the
//! weights as known.
//!
//! # Brier score
//!
//! With `N` the number of rows and `R_i(t)` the predicted risk,
//!
//! ```text
//! B(t) = (1/N) sum_i r_i
//! r_i = (1 - R_i)^2 / G(T_i-)   event of interest by t
//!     = R_i^2 / G(T_i-)         competing event by t
//!     = 0                       censored by t
//!     = R_i^2 / G(t)            T_i > t
//! ```
//!
//! The null model predicts to everybody the marginal risk of the cause: one
//! minus the Kaplan-Meier estimate, or the Aalen-Johansen cumulative incidence
//! when there is a competing event ([`null_brier`]). The index of prediction
//! accuracy is `IPA(t) = 1 - B(t) / B_0(t)` ([`ipa`]), Kattan and Gerds'
//! R-squared.
//!
//! # Area under the curve
//!
//! The cases at `t` are the rows with the event of interest at or before `t`,
//! with weights `a_i = 1 / (N G(T_i-))`, and the controls the rows still at
//! risk after `t`, with `b_j = 1 / (N G(t))`. For competing risks,
//! [`ControlSet`] chooses whether rows with a competing event by `t` are also
//! controls, with `b_j = 1 / (N G(T_j-))`. The area is
//!
//! ```text
//! AUC(t) = sum_i sum_j a_i b_j h(R_i, R_j) / (sum_i a_i sum_j b_j),
//! h = 1 if R_i > R_j, 1/2 if R_i = R_j, 0 otherwise,
//! ```
//!
//! which is the area under the empirical ROC curve by the trapezoid rule.
//! [`ControlSet::EventFreeAndCompeting`] is the area `Score` reports and
//! `AUC_2` of `timeROC`, which `timeROC`'s paper writes with controls that
//! have not had the event of interest by `t`. [`ControlSet::EventFree`] is
//! `AUC_1` of `timeROC`: the controls are the rows that are still event-free.
//! For data without a competing event the two are the same. `timeROC` leaves
//! rows with `T_i = t` out of the cases and the controls, where `Score` counts
//! the event as a case. [`HorizonRule`] chooses.
//!
//! # Standard errors
//!
//! The standard error of each metric is `sd(IF) / sqrt(N)`, `sd` taking the
//! divisor `N - 1`, for an influence function `IF` with `sum IF / N` the
//! estimate's first-order error. It is the form `Score` reports as `se`. The
//! influence function has two parts. The first holds the weights fixed. For
//! the Brier score it is `r_i - mean(r)`. For the area it is the
//! first-order term of a ratio of two weighted sums and a weighted U-statistic,
//!
//! ```text
//! IF_k = (N (a_k L_k + b_k U_k) - 2 Num) / (Dc Dn)
//!        - AUC ((N a_k - Dc) / Dc + (N b_k - Dn) / Dn),
//! ```
//!
//! with `L_k = sum_j b_j h(R_k, R_j)`, `U_k = sum_i a_i h(R_i, R_k)`, `Num`
//! the numerator and `Dc`, `Dn` the two sums of weights. The second part is
//! the effect of having estimated `G`. If a metric is a function of weights
//! `1 / G(s_i)` with derivative `g_i` with respect to the relative size of
//! each weight, the Kaplan-Meier estimate of `G` adds
//!
//! ```text
//! IF_k += d_k A(T_k) / Y(T_k) - sum_{u <= T_k} dC(u) A(u) / Y(u)^2,
//! A(u) = sum_i g_i 1(u <= s_i),
//! ```
//!
//! where `d_k` is 1 for a censored row, `dC(u)` the number censored at `u`,
//! `Y(u)` the number of rows with a time at or after `u`, and `s_i` is `T_i-`
//! for rows with an event by `t` (so the sum runs over `u < T_i`) and `t` for the
//! others. This is the influence function of the Kaplan-Meier estimate
//! substituted into the weights. `Y` counts a death at a tied time as at risk of
//! censoring, which is how `Score` forms the function, while the weights
//! themselves have the death leave first. At a tie the function is therefore
//! that of the plain Kaplan-Meier estimate of the censoring times, applied to
//! weights from the other convention. This is reproduced so that the standard
//! errors equal `Score`'s; the difference from the consistent function is of the
//! order of the share of tied times. [`ScoreOptions::conservative`] leaves the
//! term out. As `Score` does, the area's censoring term is left out at `t` at or
//! beyond the last time.
//!
//! With a Cox model for the censoring times `G_i(s) = exp(-q_i L0(s))`, with
//! `q_i = exp(x_i' b)` and `L0` the Breslow estimate whose jumps are `d_l /
//! S0(u_l)` at the distinct censoring times `u_l`, `S0(u)` the sum of `q_j` over
//! the rows with a time at or after `u` (a death at a censoring time stays in the
//! risk set), the term is `IF_k += sum_i g_i f_ik(s_i)`, with `f_ik(s)` the
//! influence of row `k` on the cumulative hazard `q_i L0(s)` of row `i`:
//!
//! ```text
//! f_ik(s) = q_i { c_k(s) + (L0(s) x_i - H(s))' f_k },
//! c_k(s) = d_k 1(T_k <= s) / S0(T_k) - q_k C(min(T_k, s)),
//! C(v) = sum over u_l <= v of d_l / S0(u_l)^2,
//! H(s) = sum over u_l <= s of d_l E(u_l) / S0(u_l),
//! ```
//!
//! where `d_k` is 1 for a row that was censored, `E` the `q`-weighted mean of
//! `x` in the risk set and `f_k = V u_k` the influence of row `k` on the
//! coefficients, with the model covariance `V` and the Breslow score residual
//! `u_k`. The sum over `i` is formed with cumulative sums over the censoring
//! times, `O(n + m p)` for each time. `Score` takes the same function from
//! `iidCox`. With tied censoring times `iidCox` counts a tied time once, and the
//! function here is the exact derivative of the score with respect to the case
//! weights, which the tests check by numerical differences; without ties the
//! two are equal to rounding error.
//!
//! The standard error of the IPA uses the influence functions of the model and
//! the null model, `IF_IPA = -IF_B / B_0 + B IF_B0 / B_0^2`, as `Score` does,
//! and takes the null model's predictions as fixed.
//!
//! # Differences from `riskRegression` and `timeROC`
//!
//! - Weights other than 0 and 1 are an error, a row of weight 0 is dropped, and
//!   a start column is an error. `Score` has no case weights.
//! - Times that are not strictly increasing, or lie after the last observed
//!   time, are an error. `Score` sorts them and drops those after the end of
//!   follow-up.
//! - An undefined area, because no case or no control exists, is `None`, as is
//!   a standard error that needs more than one row.
//! - Predicted risks outside `[0, 1]` are an error for the Brier score, where
//!   `Score` warns.
//! - The tie rule of `aeqSurv` is not applied by default, since neither
//!   function applies it.
//! - The Cox model for the censoring times is fitted by [`cox::fit`](crate::cox::fit)
//!   with the options given, `Efron` ties by default as `coxph`; `Score` fits it
//!   with `rms::cph`, which stops at a relative change of `1e-6` of the log
//!   partial likelihood, so `eps = 1e-6` reproduces its coefficients. The
//!   baseline is Breslow's and the survival is `exp(-L)`, as `Score` has it, and
//!   the influence of the coefficients is `V u` with the model covariance and
//!   the Breslow score residual. A model with strata, an offset or the exact
//!   ties method is not supported.
//!
//! # Errors of the computed numbers
//!
//! Every metric is a sum of `n` terms bounded by the largest weight
//! `w_max = 1 / min G`, normalised by `N` or by the sums of weights. A sum of
//! `n` non-negative terms has a relative error of at most `n eps`, so `B` has
//! an absolute error of at most `n eps w_max max(r)`, doubled for a second
//! implementation. The area is a ratio of such sums, bounded by `3 n eps`
//! relative to each, and an influence function is a sum of the same kind, so
//! a standard error has a relative error of the order `n eps w_max`. The
//! tolerances of the tests are derived from this.
//!
//! # References
//!
//! 1. Gerds TA, Cai T, Schumacher M. The performance of risk prediction models.
//!    Biom J 2008;50(4):457-79. doi:10.1002/bimj.200810443
//! 2. Blanche P, Dartigues JF, Jacqmin-Gadda H. Estimating and comparing
//!    time-dependent areas under receiver operating characteristic curves for
//!    censored event times with competing risks. Stat Med 2013;32(30):5381-97.
//!    doi:10.1002/sim.5958
//! 3. Blanche P, Kattan MW, Gerds TA. The c-index is not proper for the
//!    evaluation of t-year predicted risks. Biostatistics 2019;20(2):347-57.
//!    doi:10.1093/biostatistics/kxy006
//! 4. Kattan MW, Gerds TA. The index of prediction accuracy: an intuitive
//!    measure useful for evaluating risk prediction models. Diagn Progn Res
//!    2018;2:7. doi:10.1186/s41512-018-0029-2
//! 5. Ozenne B, Sorensen AL, Scheike T, Torp-Pedersen C, Gerds TA.
//!    riskRegression: predicting the risk of an event using Cox regression
//!    models. R J 2017;9(2):440-60. doi:10.32614/RJ-2017-062
//! 6. Gerds TA, Kattan MW. Medical Risk Prediction Models: With Ties to
//!    Machine Learning. Chapman and Hall/CRC, 2021.

#![expect(
    clippy::float_cmp,
    reason = "times and markers tie when they are equal, which is the definition of a tie"
)]

use epistat_core::InputError;
use epistat_linalg::{Mat, MatRef};
use rayon::prelude::*;

use super::censoring::CensoringKm;
use super::coxcens::CoxCens;
pub use super::coxcens::CoxCensoring;
use super::marginal::marginal_risk;
use super::prepare::{Kind, Sorted, check_matrix, check_times, sort_rows};
use crate::error::{Error, Result};
use crate::order::sort_by_key;
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// How the censoring distribution is obtained.
#[derive(Debug, Clone, Copy, Default)]
pub enum Censoring<'a> {
    /// The Kaplan-Meier estimate of the censoring times, `cens.model = "km"`.
    #[default]
    KaplanMeier,
    /// Probabilities the caller computed from another model of the censoring
    /// times. The standard errors are the conservative ones.
    Supplied(SuppliedCensoring<'a>),
    /// A Cox model for the censoring times, fitted here, `cens.model = "cox"`.
    /// The standard errors include the estimation of the model unless
    /// [`ScoreOptions::conservative`] is set.
    Cox(CoxCensoring<'a>),
}

/// Censoring probabilities from a model of the caller's.
#[derive(Debug, Clone, Copy)]
pub struct SuppliedCensoring<'a> {
    /// `G_i(T_i-)`: the probability of being uncensored just before the row's
    /// own time, one value per row of the data, in the data's order.
    pub before_own_time: &'a [f64],
    /// `G_i(t_j)`: the probability of being uncensored at each evaluation
    /// time, one row per row of the data and one column per time.
    pub at_times: MatRef<'a, f64>,
}

/// The scale the predictions are on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Predicted {
    /// The probability of the event by the time.
    #[default]
    AbsoluteRisk,
    /// The probability of being free of the event by the time. Only for data
    /// with one event code.
    EventFreeProbability,
}

/// Which rows are controls when there is a competing event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ControlSet {
    /// Rows still at risk after the time, and rows with a competing event by
    /// the time: `AUC_2` of `timeROC` and the area of `Score`.
    #[default]
    EventFreeAndCompeting,
    /// Rows still at risk after the time only: `AUC_1` of `timeROC`.
    EventFree,
}

/// How a row whose time is exactly the evaluation time is counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HorizonRule {
    /// An event at the time is a case, `Score`.
    #[default]
    Inclusive,
    /// A row at exactly the time is neither a case nor a control, `timeROC`.
    Exclusive,
}

/// What the metrics take besides the data, the predictions and the times.
#[derive(Debug, Clone, Copy)]
pub struct ScoreOptions<'a> {
    /// The rule that makes times equal when they differ by rounding error,
    /// applied to the data first. `Exact`: neither `Score` nor `timeROC`
    /// applies one.
    pub timefix: TimeFix,
    /// The event of interest. 1, which is the first cause as in `Score`.
    pub cause: u32,
    /// How the weights are obtained. Kaplan-Meier, as `cens.model = "km"`.
    pub censoring: Censoring<'a>,
    /// Compute the standard errors, `se.fit = TRUE` of `Score`.
    pub variance: bool,
    /// Leave the estimation of the censoring distribution out of the standard
    /// errors, `conservative = TRUE` of `Score`.
    pub conservative: bool,
    /// Keep the influence function of every row, which an IPA standard error
    /// needs. Costs 8 bytes per row and time.
    pub keep_influence: bool,
    /// The scale of the predictions, Brier score only.
    pub predicted: Predicted,
    /// Which rows are controls of the area, competing risks only.
    pub controls: ControlSet,
    /// How a row at exactly the evaluation time is counted, area only.
    pub horizon: HorizonRule,
}

impl Default for ScoreOptions<'_> {
    fn default() -> Self {
        Self {
            timefix: TimeFix::Exact,
            cause: 1,
            censoring: Censoring::KaplanMeier,
            variance: true,
            conservative: false,
            keep_influence: false,
            predicted: Predicted::AbsoluteRisk,
            controls: ControlSet::EventFreeAndCompeting,
            horizon: HorizonRule::Inclusive,
        }
    }
}

/// The Brier score at each time.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    /// The times, as given.
    pub times: Vec<f64>,
    /// The Brier score.
    pub brier: Vec<f64>,
    /// Its standard error, `None` unless asked for and defined (at least two
    /// rows).
    pub std_err: Vec<Option<f64>>,
    /// The influence function of each used row at each time, one row per row
    /// of positive weight in the order of the data. `None` unless asked for.
    pub influence: Option<Mat<f64>>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight.
    pub n_used: usize,
}

/// The index of prediction accuracy at each time.
#[derive(Debug, Clone, PartialEq)]
pub struct Ipa {
    /// The times, as given.
    pub times: Vec<f64>,
    /// `1 - B / B_0`, `None` where the null model has score 0.
    pub ipa: Vec<Option<f64>>,
    /// Its standard error, `None` unless both scores kept their influence
    /// functions and it is defined.
    pub std_err: Vec<Option<f64>>,
}

/// The area under the ROC curve at each time.
#[derive(Debug, Clone, PartialEq)]
pub struct Auc {
    /// The times, as given.
    pub times: Vec<f64>,
    /// The area, `None` where there is no case or no control.
    pub auc: Vec<Option<f64>>,
    /// Its standard error, `None` unless asked for and defined.
    pub std_err: Vec<Option<f64>>,
    /// The number of cases at each time.
    pub n_cases: Vec<usize>,
    /// The number of rows still at risk after each time.
    pub n_survivors: Vec<usize>,
    /// The number of rows with a competing event by each time.
    pub n_competing: Vec<usize>,
    /// The influence function, as for [`Score::influence`].
    pub influence: Option<Mat<f64>>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight.
    pub n_used: usize,
}

/// The sorted rows and the censoring weights shared by every time.
struct Core<'a> {
    sorted: Sorted,
    km: CensoringKm,
    /// `G(T_i-)` at each sorted position.
    g_own: Vec<f64>,
    supplied: Option<SuppliedCensoring<'a>>,
    cox: Option<CoxCens>,
    /// The baseline hazard of the censoring Cox model at each evaluation time.
    lam_times: Vec<f64>,
    n: usize,
}

impl<'a> Core<'a> {
    fn new(data: &SurvData<'_>, opts: &ScoreOptions<'a>, times: &[f64]) -> Result<Self> {
        let n_times = times.len();
        let sorted = sort_rows(data, opts.timefix, opts.cause)?;
        let n = sorted.n();
        let w = vec![1.0; n];
        let km = CensoringKm::new(&sorted.time, &sorted.is_censored(), &w, true);
        let mut cox = None;
        let mut lam_times = Vec::new();
        let (supplied, g_own) = match opts.censoring {
            Censoring::Cox(spec) => {
                let model = CoxCens::new(&sorted, &spec)?;
                let g_own = (0..n)
                    .map(|pos| model.survival(pos, model.lambda_before(sorted.time[pos])))
                    .collect();
                lam_times = times.iter().map(|&t| model.lambda_through(t)).collect();
                cox = Some(model);
                (None, g_own)
            }
            Censoring::KaplanMeier => (
                None,
                sorted
                    .time
                    .iter()
                    .map(|&t| km.before(t))
                    .collect::<Vec<_>>(),
            ),
            Censoring::Supplied(s) => {
                if s.before_own_time.len() != sorted.n_rows {
                    return Err(Error::from(InputError::Length {
                        what: "before_own_time",
                        expected: sorted.n_rows,
                        found: s.before_own_time.len(),
                    }));
                }
                check_matrix("at_times", s.at_times, sorted.n_rows, n_times)?;
                let g_own: Vec<f64> = sorted
                    .row
                    .iter()
                    .map(|&r| s.before_own_time[r as usize])
                    .collect();
                let at = s.at_times;
                let at_values = sorted
                    .row
                    .iter()
                    .flat_map(|&r| (0..n_times).map(move |j| at[(r as usize, j)]));
                for g in g_own.iter().copied().chain(at_values) {
                    if !(g > 0.0 && g <= 1.0) {
                        return Err(Error::invalid(format!(
                            "censoring probabilities must be in (0, 1], got {g}"
                        )));
                    }
                }
                (Some(s), g_own)
            }
        };
        Ok(Self {
            sorted,
            km,
            g_own,
            supplied,
            cox,
            lam_times,
            n,
        })
    }

    /// `G(t)` for the row at sorted position `pos`, evaluation time `t` in
    /// column `j`.
    fn g_at(&self, pos: usize, j: usize, t: f64) -> f64 {
        if let Some(cox) = &self.cox {
            return cox.survival(pos, self.lam_times[j]);
        }
        match &self.supplied {
            None => self.km.at(t),
            Some(s) => s.at_times[(self.sorted.row[pos] as usize, j)],
        }
    }

    /// Whether the estimation of `G` enters the standard error at time `t`.
    fn uses_censoring_term(&self, opts: &ScoreOptions<'_>, t: f64) -> bool {
        self.supplied.is_none() && !opts.conservative && opts.variance && t < self.sorted.max_time
    }

    /// The influence of the Kaplan-Meier estimate of `G` on a metric, for the
    /// derivative `g` with respect to each row's relative weight. See the
    /// module documentation.
    fn censoring_term(&self, t: f64, g: &[f64]) -> Vec<f64> {
        if let Some(cox) = &self.cox {
            return cox.term(&self.sorted, t, g);
        }
        let sorted = &self.sorted;
        let km = &self.km;
        let n = sorted.n();
        let through = sorted.count_through(t);
        // Sums over the rows at or before t from each position to the end of
        // that range, and the sum over the rows after t.
        let mut suffix = vec![0.0; through + 1];
        for i in (0..through).rev() {
            suffix[i] = suffix[i + 1] + g[i];
        }
        let tail: f64 = g[through..n].iter().sum();
        let m = km.curve.time.len();
        let mut a = Vec::with_capacity(m);
        let mut cumulative = vec![0.0; m + 1];
        for (index, &u) in km.curve.time.iter().enumerate() {
            let first_after = sorted.time.partition_point(|&x| x <= u);
            let before_t = if first_after < through {
                suffix[first_after]
            } else {
                0.0
            };
            let a_u = before_t + if u <= t { tail } else { 0.0 };
            a.push(a_u);
            let y = km.at_risk_plain[index];
            cumulative[index + 1] = cumulative[index] + km.n_censored[index] / (y * y) * a_u;
        }
        (0..n)
            .map(|k| {
                if sorted.kind[k] == Kind::Censored {
                    let index = km.curve.time.partition_point(|&u| u < sorted.time[k]);
                    a[index] / km.at_risk_plain[index] - cumulative[index + 1]
                } else {
                    let index = km.curve.time.partition_point(|&u| u <= sorted.time[k]);
                    -cumulative[index]
                }
            })
            .collect()
    }

    /// The influence functions of the used rows, one row per used row in the
    /// order of the data, from columns in sorted order.
    fn influence_matrix(&self, columns: &[Option<Vec<f64>>]) -> Mat<f64> {
        let n = self.n;
        let mut by_row: Vec<usize> = (0..n).collect();
        by_row.sort_unstable_by_key(|&p| self.sorted.row[p]);
        let mut rank = vec![0; n];
        for (k, &p) in by_row.iter().enumerate() {
            rank[p] = k;
        }
        let mut out = Mat::<f64>::zeros(n, columns.len());
        for (j, column) in columns.iter().enumerate() {
            if let Some(values) = column {
                for (pos, &v) in values.iter().enumerate() {
                    out[(rank[pos], j)] = v;
                }
            }
        }
        out
    }
}

/// The standard error `sd(IF) / sqrt(N)`, `sd` with divisor `N - 1`.
pub(super) fn std_err_of(influence: &[f64]) -> Option<f64> {
    let n = influence.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mean = influence.iter().sum::<f64>() / nf;
    let ss: f64 = influence.iter().map(|&v| (v - mean) * (v - mean)).sum();
    Some((ss / (nf - 1.0)).sqrt() / nf.sqrt())
}

/// What one time of the Brier score returns.
struct BrierColumn {
    brier: f64,
    std_err: Option<f64>,
    influence: Option<Vec<f64>>,
}

/// The score at time `t`, column `j`, for the risk at each sorted position.
fn brier_column(
    core: &Core<'_>,
    opts: &ScoreOptions<'_>,
    j: usize,
    t: f64,
    risk: &dyn Fn(usize) -> f64,
) -> BrierColumn {
    let sorted = &core.sorted;
    let n = core.n;
    let mut residual = vec![0.0; n];
    for (pos, slot) in residual.iter_mut().enumerate() {
        let r = risk(pos);
        *slot = if sorted.time[pos] <= t {
            match sorted.kind[pos] {
                Kind::Cause => (1.0 - r) * (1.0 - r) / core.g_own[pos],
                Kind::Competing => r * r / core.g_own[pos],
                Kind::Censored => 0.0,
            }
        } else {
            r * r / core.g_at(pos, j, t)
        };
    }
    let nf = n as f64;
    let mean = residual.iter().sum::<f64>() / nf;
    let brier = mean;
    if !opts.variance {
        return BrierColumn {
            brier,
            std_err: None,
            influence: None,
        };
    }
    let mut influence: Vec<f64> = residual.iter().map(|&r| r - mean).collect();
    if core.uses_censoring_term(opts, t) {
        let term = core.censoring_term(t, &residual);
        for (i, c) in influence.iter_mut().zip(&term) {
            *i += c;
        }
    }
    BrierColumn {
        brier,
        std_err: std_err_of(&influence),
        influence: Some(influence),
    }
}

fn collect_score(
    core: &Core<'_>,
    opts: &ScoreOptions<'_>,
    times: &[f64],
    columns: &[BrierColumn],
) -> Score {
    let influence = (opts.keep_influence && opts.variance).then(|| {
        let cols: Vec<Option<Vec<f64>>> = columns.iter().map(|c| c.influence.clone()).collect();
        core.influence_matrix(&cols)
    });
    Score {
        times: times.to_vec(),
        brier: columns.iter().map(|c| c.brier).collect(),
        std_err: columns.iter().map(|c| c.std_err).collect(),
        influence,
        n_rows: core.sorted.n_rows,
        n_used: core.n,
    }
}

/// The IPCW Brier score of the predictions at `times`.
///
/// `predicted` has one row per row of the data and one column per time. See
/// the module documentation for the definition, the weights and the standard
/// error.
///
/// # Errors
///
/// An input error when the data is interval-censored, has a start column, has
/// a weight other than 0 and 1, or has no row of positive weight; when `cause`
/// is outside the event codes; when `times` is empty, not strictly increasing,
/// not finite or after the last observed time; when `predicted` has another
/// shape than rows by times or a value outside `[0, 1]`; when
/// [`Predicted::EventFreeProbability`] is used with several event codes; or
/// when supplied censoring probabilities are not in `(0, 1]`.
pub fn brier(
    data: &SurvData<'_>,
    predicted: MatRef<'_, f64>,
    times: &[f64],
    opts: &ScoreOptions<'_>,
) -> Result<Score> {
    let core = Core::new(data, opts, times)?;
    check_times(times, core.sorted.max_time)?;
    check_matrix("predicted", predicted, core.sorted.n_rows, times.len())?;
    let complement = match opts.predicted {
        Predicted::AbsoluteRisk => false,
        Predicted::EventFreeProbability => {
            if core.sorted.kind.contains(&Kind::Competing) || data.followup()?.n_codes() > 1 {
                return Err(Error::invalid(
                    "event-free probabilities are for data with one event code; with competing \
                     risks give the absolute risk of the cause",
                ));
            }
            true
        }
    };
    for j in 0..times.len() {
        for i in 0..predicted.nrows() {
            if !(0.0..=1.0).contains(&predicted[(i, j)]) {
                return Err(Error::invalid(format!(
                    "predicted: row {i}, column {j} is {}, which is not a probability",
                    predicted[(i, j)]
                )));
            }
        }
    }
    let columns: Vec<BrierColumn> = times
        .par_iter()
        .enumerate()
        .map(|(j, &t)| {
            let risk = |pos: usize| {
                let p = predicted[(core.sorted.row[pos] as usize, j)];
                if complement { 1.0 - p } else { p }
            };
            brier_column(&core, opts, j, t, &risk)
        })
        .collect();
    Ok(collect_score(&core, opts, times, &columns))
}

/// The Brier score of the null model, which predicts to everyone the marginal
/// risk of the cause: one minus the Kaplan-Meier estimate, or the
/// Aalen-Johansen cumulative incidence with a competing event. `Score`'s
/// `null.model`.
///
/// # Errors
///
/// As for [`brier`], except those about `predicted`.
pub fn null_brier(data: &SurvData<'_>, times: &[f64], opts: &ScoreOptions<'_>) -> Result<Score> {
    let core = Core::new(data, opts, times)?;
    check_times(times, core.sorted.max_time)?;
    let marginal = marginal_risk(&core.sorted.time, &core.sorted.kind, times);
    let columns: Vec<BrierColumn> = times
        .par_iter()
        .enumerate()
        .map(|(j, &t)| brier_column(&core, opts, j, t, &|_| marginal[j]))
        .collect();
    Ok(collect_score(&core, opts, times, &columns))
}

/// The index of prediction accuracy, `1 - B / B_0`, of a model against the
/// null model, with the standard error of `Score` when both kept their
/// influence functions (`keep_influence`).
///
/// # Errors
///
/// An input error when the two scores are at different times or from data of
/// different numbers of rows.
pub fn ipa(model: &Score, null: &Score) -> Result<Ipa> {
    if model.times != null.times || model.n_rows != null.n_rows || model.n_used != null.n_used {
        return Err(Error::invalid(
            "the model and the null model must be scored on the same data and times",
        ));
    }
    let mut out = Ipa {
        times: model.times.clone(),
        ipa: Vec::new(),
        std_err: Vec::new(),
    };
    for j in 0..model.times.len() {
        let (b, b0) = (model.brier[j], null.brier[j]);
        if b0 <= 0.0 {
            out.ipa.push(None);
            out.std_err.push(None);
            continue;
        }
        out.ipa.push(Some(1.0 - b / b0));
        let se = match (&model.influence, &null.influence) {
            (Some(m), Some(z)) => {
                let combined: Vec<f64> = (0..m.nrows())
                    .map(|i| -m[(i, j)] / b0 + b * z[(i, j)] / (b0 * b0))
                    .collect();
                std_err_of(&combined)
            }
            _ => None,
        };
        out.std_err.push(se);
    }
    Ok(out)
}

/// What one time of the area returns.
struct AucColumn {
    auc: Option<f64>,
    std_err: Option<f64>,
    influence: Option<Vec<f64>>,
    n_cases: usize,
    n_survivors: usize,
    n_competing: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Neither,
    Case,
    Control,
}

fn auc_column(
    core: &Core<'_>,
    opts: &ScoreOptions<'_>,
    j: usize,
    t: f64,
    marker: &dyn Fn(usize) -> f64,
) -> AucColumn {
    let sorted = &core.sorted;
    let n = core.n;
    let nf = n as f64;
    let inclusive = opts.horizon == HorizonRule::Inclusive;
    let mut role = vec![Role::Neither; n];
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let (mut n_cases, mut n_survivors, mut n_competing) = (0, 0, 0);
    for pos in 0..n {
        let time = sorted.time[pos];
        let before = if inclusive { time <= t } else { time < t };
        if time > t {
            n_survivors += 1;
            role[pos] = Role::Control;
            b[pos] = 1.0 / (nf * core.g_at(pos, j, t));
        } else if before {
            match sorted.kind[pos] {
                Kind::Cause => {
                    n_cases += 1;
                    role[pos] = Role::Case;
                    a[pos] = 1.0 / (nf * core.g_own[pos]);
                }
                Kind::Competing => {
                    n_competing += 1;
                    if opts.controls == ControlSet::EventFreeAndCompeting {
                        role[pos] = Role::Control;
                        b[pos] = 1.0 / (nf * core.g_own[pos]);
                    }
                }
                Kind::Censored => {}
            }
        }
    }
    let dc: f64 = a.iter().sum();
    let dn: f64 = b.iter().sum();
    let counts = |auc, std_err, influence| AucColumn {
        auc,
        std_err,
        influence,
        n_cases,
        n_survivors,
        n_competing,
    };
    if dc == 0.0 || dn == 0.0 {
        return counts(None, None, None);
    }

    let mut order: Vec<usize> = (0..n).collect();
    sort_by_key(&mut order, marker);
    // Weight of the controls below, or equal to, a case's marker, and of the
    // cases above, or equal to, a control's.
    let mut lc = vec![0.0; n];
    let mut uc = vec![0.0; n];
    let mut below = 0.0;
    let mut start = 0;
    while start < n {
        let mut end = start;
        let mut in_group = 0.0;
        while end < n && marker(order[end]) == marker(order[start]) {
            in_group += b[order[end]];
            end += 1;
        }
        for &p in &order[start..end] {
            if role[p] == Role::Case {
                lc[p] = below + 0.5 * in_group;
            }
        }
        below += in_group;
        start = end;
    }
    let mut above = 0.0;
    let mut end = n;
    while end > 0 {
        let mut start = end;
        let mut in_group = 0.0;
        while start > 0 && marker(order[start - 1]) == marker(order[end - 1]) {
            start -= 1;
            in_group += a[order[start]];
        }
        for &p in &order[start..end] {
            if role[p] == Role::Control {
                uc[p] = above + 0.5 * in_group;
            }
        }
        above += in_group;
        end = start;
    }
    let numerator: f64 = (0..n)
        .filter(|&p| role[p] == Role::Case)
        .map(|p| a[p] * lc[p])
        .sum();
    let denominator = dc * dn;
    let auc = numerator / denominator;
    if !opts.variance {
        return counts(Some(auc), None, None);
    }
    let mut influence: Vec<f64> = (0..n)
        .map(|p| {
            (nf * (a[p] * lc[p] + b[p] * uc[p]) - 2.0 * numerator) / denominator
                - auc * ((nf * a[p] - dc) / dc + (nf * b[p] - dn) / dn)
        })
        .collect();
    if core.uses_censoring_term(opts, t) {
        let g: Vec<f64> = (0..n)
            .map(|p| match role[p] {
                Role::Case => nf * a[p] * (lc[p] / denominator - auc / dc),
                Role::Control => nf * b[p] * (uc[p] / denominator - auc / dn),
                Role::Neither => 0.0,
            })
            .collect();
        let term = core.censoring_term(t, &g);
        for (i, c) in influence.iter_mut().zip(&term) {
            *i += c;
        }
    }
    counts(Some(auc), std_err_of(&influence), Some(influence))
}

/// The time-dependent area under the ROC curve of the markers at `times`.
///
/// `marker` has one row per row of the data and one column per time. A larger
/// value means a higher risk; any finite value is allowed. See the module
/// documentation for the cases, the controls, the weights and the standard
/// error.
///
/// # Errors
///
/// An input error as for [`brier`], except that a marker may be any finite
/// number, and [`ScoreOptions::predicted`] is not read.
pub fn auc(
    data: &SurvData<'_>,
    marker: MatRef<'_, f64>,
    times: &[f64],
    opts: &ScoreOptions<'_>,
) -> Result<Auc> {
    let core = Core::new(data, opts, times)?;
    check_times(times, core.sorted.max_time)?;
    check_matrix("marker", marker, core.sorted.n_rows, times.len())?;
    let columns: Vec<AucColumn> = times
        .par_iter()
        .enumerate()
        .map(|(j, &t)| {
            let m = |pos: usize| marker[(core.sorted.row[pos] as usize, j)];
            auc_column(&core, opts, j, t, &m)
        })
        .collect();
    let influence = (opts.keep_influence && opts.variance).then(|| {
        let cols: Vec<Option<Vec<f64>>> = columns.iter().map(|c| c.influence.clone()).collect();
        core.influence_matrix(&cols)
    });
    Ok(Auc {
        times: times.to_vec(),
        auc: columns.iter().map(|c| c.auc).collect(),
        std_err: columns.iter().map(|c| c.std_err).collect(),
        n_cases: columns.iter().map(|c| c.n_cases).collect(),
        n_survivors: columns.iter().map(|c| c.n_survivors).collect(),
        n_competing: columns.iter().map(|c| c.n_competing).collect(),
        influence,
        n_rows: core.sorted.n_rows,
        n_used: core.n,
    })
}
