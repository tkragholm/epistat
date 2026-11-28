//! Nested case-control data analysed with inverse probability weights, as
//! `multipleNCC` (Stoer and Samuelsen), and the absolute risk from the weighted
//! fit.
//!
//! A nested case-control study samples, at the failure time of each case, `m`
//! controls from the other people at risk. The conditional logistic analysis
//! ([`clogit`](crate::clogit)) uses the matched sets only. Samuelsen's
//! pseudolikelihood \[1\] gives every sampled person an inclusion probability
//! `p_i` and fits a Cox model on the cases and the sampled controls over their
//! whole follow-up with weights `1 / p_i`. It uses more of the information in
//! the controls, which can be reused for other outcomes, and it gives a baseline
//! hazard and so an absolute risk.
//!
//! # Inclusion probabilities
//!
//! The risk set of the case `j` at its time `t_j` is
//! `R_j = {i : entry_i < t_j <= exit_i}`, of the case's matching stratum when
//! there is one, with `n_j` people. A person `i` who is not the case `j` is
//! sampled for it with probability `q_j = m / (n_j - 1)`. The probability that
//! `i` is sampled for at least one case is
//!
//! ```text
//! p_i = 1 - prod over j with i in R_j, j != i, of (1 - q_j),    w_i = 1 / p_i
//! ```
//!
//! for a control, and a case has `p_i = 1` and `w_i = 1`. The weighted Cox model
//! of the sampled controls and the cases, with a cluster per person, is
//! [`fit_weighted`], and its robust variance is that of `multipleNCC`'s
//! `variance = "robust"`. It does not account for `p_i` being known from the
//! design with sampling that is not independent between cases, and the
//! variance of Samuelsen's paper with that correction (`"Modelbased"`) is not
//! provided.
//!
//! The risk-set sizes are computed here, since the matching engine reports the
//! controls and not the size of the set they were drawn from. With the case
//! times of a stratum sorted, `n_j` is the number of exits at or after `t_j`
//! less the number of entries at or after `t_j`, each from a binary search in a
//! sorted array. The product over cases is the sum of `log(1 - q_j)` over the
//! cases whose time lies in `(entry_i, exit_i]`, a contiguous range of the sorted
//! case times, so a prefix sum gives it in `O(log K)` per person. The whole
//! computation is `O(N log N)` in time and `O(N)` in memory, with no loop of
//! people over cases.
//!
//! # Numerical choices
//!
//! The prefix sums are held in double-double (a sum and its rounding error, by
//! two-sum), so that the difference of two prefix sums for a short range is
//! accurate to its own size and not to the size of the whole prefix. `p_i` is
//! `-expm1(sum)`, which is accurate for small `p_i`. A case with `n_j - 1 = m` is
//! sampled with probability 1 and gives `p_i = 1` to everyone in its risk set,
//! which is counted separately because `log(0)` is not a number. A case whose
//! risk set holds fewer than `m + 1` people cannot have had `m` controls, and
//! is an error.
//!
//! # Absolute risk
//!
//! The Breslow hazard of the weighted fit is
//! `dLambda_0(t_j) = 1 / sum over sampled i in R_j of w_i exp(x_i' beta)`, and the
//! absolute risk of covariates `x` by `t` is `1 - exp(-Lambda_0(t) exp(x' beta))`.
//! [`baseline`] reads the cumulative hazard from [`cox::basehaz`](crate::cox::basehaz)
//! of the weighted fit, so it follows the ties of the fit, and [`absolute_risk`]
//! evaluates the risk of covariate profiles. The estimate of Langholz and Borgan
//! \[2\], which scales the sampled risk set by `n_j / (m + 1)` instead of
//! weighting, and a variance of the absolute risk are not provided.
//!
//! # Not covered
//!
//! The kernel and model-based estimates of the inclusion probabilities of
//! `multipleNCC` (`weight.method` `"gam"`, `"glm"`, `"Chen"`), a number of controls
//! that differs between cases, and matching on a continuous variable within an
//! interval. Matching on a categorical variable is the `stratum`.
//!
//! # Agreement with R
//!
//! `tests/r_ncc.rs` compares the risk-set sizes and inclusion probabilities
//! with the explicit loop of the formula and with `multipleNCC::KMprob`, the
//! weighted fit with `multipleNCC::wpl`, and the baseline hazard and absolute
//! risk with `survival::basehaz` of the same weighted fit, on four simulated
//! cohorts: without entry times, with delayed entry, with matching strata, and
//! with tied times. `tests/ncc_checks.rs` repeats 200 simulated studies and
//! compares the weighted estimates with the full-cohort Cox fit.
//!
//! # References
//!
//! 1. Samuelsen SO. A pseudolikelihood approach to analysis of nested
//!    case-control studies. Biometrika 1997;84(2):379-94.
//! 2. Langholz B, Borgan O. Estimation of absolute risk from nested
//!    case-control data. Biometrics 1997;53(2):767-74.
//! 3. Stoer NC, Samuelsen SO. Inverse probability weighting in nested
//!    case-control studies with additional matching: a simulation study. Stat
//!    Med 2013;32(30):5328-39.
//! 4. Stoer NC, Samuelsen SO. multipleNCC: inverse probability weighting of
//!    nested case-control data. R J 2016;8(2):5-18.

// Rows are indexed in `u32`; `inclusion_probabilities` refuses more rows than
// that, and `SurvData` does the same for the fit.
#![expect(
    clippy::cast_possible_truncation,
    reason = "the row counts are checked to fit in u32"
)]

use epistat_core::InputError;
use epistat_linalg::MatRef;

use crate::check::{check_lengths, refuse_non_finite, refuse_non_finite_matrix};
use crate::cox::{
    BaselineOptions, Centring, CoxModelFit, CoxOptions, FittedModel, basehaz, fit as cox_fit,
};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// The cohort and the sampling of an analysis of nested case-control data.
#[derive(Debug, Clone, Copy)]
pub struct NccDesign<'a> {
    /// The entry time of each person of the whole cohort, `None` when everyone
    /// is at risk from the start. A person is at risk at `t` when
    /// `entry < t <= exit`.
    pub entry: Option<&'a [f64]>,
    /// The exit time of each person: the failure time of a case, the
    /// censoring time of the others.
    pub exit: &'a [f64],
    /// Whether each person is a case.
    pub case: &'a [bool],
    /// The number of controls sampled at each case time.
    pub controls_per_case: usize,
    /// The matching stratum of each person, `None` without matching. Controls
    /// are sampled from the case's stratum, and the risk set and the
    /// probabilities are those of the stratum.
    pub stratum: Option<&'a [u32]>,
}

/// The risk-set sizes and inclusion probabilities of a design.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct InclusionProbabilities {
    /// The probability of each person of the cohort that they were sampled as
    /// a control for at least one case, 1 for a case. A person who is at risk
    /// at no case time has 0.
    pub probability: Vec<f64>,
    /// The rows of the cases in ascending order.
    pub case_rows: Vec<u32>,
    /// The size `n_j` of the risk set of each case, in the order of
    /// `case_rows`, the case itself included.
    pub risk_set_size: Vec<usize>,
    /// The probability `m / (n_j - 1)` that a given other member of the risk set
    /// is sampled for each case, in the order of `case_rows`.
    pub sampling_probability: Vec<f64>,
}

/// A sum with its rounding error.
#[derive(Clone, Copy)]
struct Double {
    hi: f64,
    lo: f64,
}

impl Double {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    fn add(self, x: f64) -> Self {
        let (s, e) = two_sum(self.hi, x);
        let (hi, lo) = quick_two_sum(s, e + self.lo);
        Self { hi, lo }
    }

    /// `self - other` as one number.
    fn difference(self, other: Self) -> f64 {
        let (s, e) = two_sum(self.hi, -other.hi);
        s + (e + (self.lo - other.lo))
    }
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// The risk-set sizes and the inclusion probabilities of Samuelsen's
/// weights. See the module documentation for the formula and the algorithm.
///
/// # Errors
///
/// An input error for columns of different lengths, an exit time that is not
/// finite, an entry time that is not finite or is not before the exit, no case,
/// `controls_per_case` of 0, and a case whose risk set has fewer than
/// `controls_per_case + 1` people.
pub fn inclusion_probabilities(design: &NccDesign<'_>) -> Result<InclusionProbabilities> {
    let n = design.exit.len();
    check_lengths(&[("exit", n), ("case", design.case.len())])?;
    if let Some(entry) = design.entry {
        check_lengths(&[("exit", n), ("entry", entry.len())])?;
        refuse_non_finite(entry, "entry")?;
        if entry.iter().zip(design.exit).any(|(a, b)| a >= b) {
            return Err(Error::value(
                "entry: a person enters at or after their exit",
            ));
        }
    }
    if let Some(stratum) = design.stratum {
        check_lengths(&[("exit", n), ("stratum", stratum.len())])?;
    }
    refuse_non_finite(design.exit, "exit")?;
    if u32::try_from(n).is_err() {
        return Err(InputError::TooLarge {
            what: "exit",
            limit: u64::from(u32::MAX),
            found: n as u64,
        }
        .into());
    }
    let m = design.controls_per_case;
    if m == 0 {
        return Err(Error::options("controls_per_case: must be at least 1"));
    }
    if !design.case.iter().any(|&c| c) {
        return Err(Error::no_information("case: there is no case"));
    }

    // The rows of each stratum, in the order of the stratum codes.
    let mut order: Vec<u32> = (0..n as u32).collect();
    if let Some(stratum) = design.stratum {
        order.sort_by_key(|&i| stratum[i as usize]);
    }
    let mut probability = vec![0.0_f64; n];
    let mut case_info: Vec<(u32, usize, f64)> = Vec::new();
    let mut from = 0;
    while from < n {
        let mut to = from + 1;
        if let Some(stratum) = design.stratum {
            let code = stratum[order[from] as usize];
            while to < n && stratum[order[to] as usize] == code {
                to += 1;
            }
        } else {
            to = n;
        }
        stratum_probabilities(design, &order[from..to], &mut probability, &mut case_info)?;
        from = to;
    }
    case_info.sort_by_key(|c| c.0);
    Ok(InclusionProbabilities {
        probability,
        case_rows: case_info.iter().map(|c| c.0).collect(),
        risk_set_size: case_info.iter().map(|c| c.1).collect(),
        sampling_probability: case_info.iter().map(|c| c.2).collect(),
    })
}

/// The probabilities of the people of one stratum, and the risk set of each of
/// its cases.
fn stratum_probabilities(
    design: &NccDesign<'_>,
    members: &[u32],
    probability: &mut [f64],
    case_info: &mut Vec<(u32, usize, f64)>,
) -> Result<()> {
    let m = design.controls_per_case;
    let exits: Vec<f64> = {
        let mut v: Vec<f64> = members.iter().map(|&i| design.exit[i as usize]).collect();
        v.sort_by(f64::total_cmp);
        v
    };
    let entries: Vec<f64> = design.entry.map_or_else(Vec::new, |entry| {
        let mut v: Vec<f64> = members.iter().map(|&i| entry[i as usize]).collect();
        v.sort_by(f64::total_cmp);
        v
    });
    let mut cases: Vec<u32> = members
        .iter()
        .copied()
        .filter(|&i| design.case[i as usize])
        .collect();
    cases.sort_by(|&a, &b| design.exit[a as usize].total_cmp(&design.exit[b as usize]));
    let times: Vec<f64> = cases.iter().map(|&i| design.exit[i as usize]).collect();

    // n_j, the sampling probability, and prefix sums over the cases in time
    // order of log(1 - q_j) and of the number of cases with q_j = 1.
    let mut log_survival = vec![Double::ZERO; cases.len() + 1];
    let mut certain = vec![0usize; cases.len() + 1];
    for (k, &row) in cases.iter().enumerate() {
        let t = times[k];
        let at_or_after_exit = exits.len() - exits.partition_point(|&e| e < t);
        let at_or_after_entry = entries.len() - entries.partition_point(|&e| e < t);
        let size = at_or_after_exit - at_or_after_entry;
        if size < m + 1 {
            return Err(Error::no_information(format!(
                "case: the risk set of row {row} has {size} people, fewer than the {} needed for {m} controls",
                m + 1
            )));
        }
        let q = m as f64 / (size - 1) as f64;
        case_info.push((row, size, q));
        if q >= 1.0 {
            log_survival[k + 1] = log_survival[k];
            certain[k + 1] = certain[k] + 1;
        } else {
            log_survival[k + 1] = log_survival[k].add((-q).ln_1p());
            certain[k + 1] = certain[k];
        }
    }
    for &row in members {
        let i = row as usize;
        if design.case[i] {
            probability[i] = 1.0;
            continue;
        }
        let lo = design
            .entry
            .map_or(0, |entry| times.partition_point(|&t| t <= entry[i]));
        let hi = times.partition_point(|&t| t <= design.exit[i]);
        probability[i] = if hi <= lo {
            0.0
        } else if certain[hi] > certain[lo] {
            1.0
        } else {
            -log_survival[hi].difference(log_survival[lo]).exp_m1()
        };
    }
    Ok(())
}

/// The weight of each person of the cohort: 1 for a case, `1 / p` for a
/// sampled control and 0 for a control that was not sampled.
///
/// # Errors
///
/// An input error for columns of different lengths, and for a sampled control
/// whose inclusion probability is 0, which means no case time falls in their
/// follow-up and they cannot have been sampled.
pub fn weights(
    probabilities: &InclusionProbabilities,
    sampled: &[bool],
    case: &[bool],
) -> Result<Vec<f64>> {
    let n = probabilities.probability.len();
    check_lengths(&[
        ("probabilities", n),
        ("sampled", sampled.len()),
        ("case", case.len()),
    ])?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(if case[i] {
            1.0
        } else if sampled[i] {
            let p = probabilities.probability[i];
            if p <= 0.0 {
                return Err(Error::inconsistent(format!(
                    "sampled: row {i} is a sampled control with inclusion probability 0"
                )));
            }
            1.0 / p
        } else {
            0.0
        });
    }
    Ok(out)
}

/// The data with the weights and one cluster per row.
fn weighted_data<'a>(
    data: &'a SurvData<'a>,
    weights: &'a [f64],
    clusters: &'a [u32],
) -> Result<SurvData<'a>> {
    check_lengths(&[("data", data.n_rows()), ("weights", weights.len())])?;
    if data.weight().is_some() || data.cluster().is_some() {
        return Err(Error::not_applicable(
            "data: the weights and the clusters are set by the analysis, so the data may not carry them",
        ));
    }
    refuse_non_finite(weights, "weights")?;
    let surv = data.followup();
    let mut builder = SurvData::builder(surv.stop(), surv.status())
        .n_codes(surv.n_codes())
        .weight(weights)
        .cluster(clusters);
    if let Some(start) = surv.start() {
        builder = builder.start(start);
    }
    if let Some(stratum) = data.stratum() {
        builder = builder.stratum(stratum);
    }
    builder.build()
}

/// The weighted Cox model of the cases and the sampled controls, with a robust
/// variance for each person, as `multipleNCC::wpl`.
///
/// `data` has one row per case and sampled control with their whole follow-up
/// `(entry, exit]` and the status, `x` their covariates and `weights` the
/// weights of [`weights`] for those rows. `data` may have strata, which are
/// strata of the Cox model, but no weights or clusters. A row of weight 0 is
/// left out. The result is that of [`cox::fit`](crate::cox::fit) with
/// `robust_covariance` and `robust_se` set, one cluster per row.
///
/// # Errors
///
/// An input error for the invalid input of the Cox fit, for `data` with
/// weights or clusters, and for weights of another length than the data or
/// that are not finite.
pub fn fit_weighted(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    weights: &[f64],
    opts: &CoxOptions,
) -> Result<CoxModelFit> {
    let clusters: Vec<u32> = (0..data.n_rows() as u32).collect();
    let weighted = weighted_data(data, weights, &clusters)?;
    cox_fit(&weighted, x, opts)
}

/// The cumulative baseline hazard of a weighted fit, for the subject whose
/// covariates are all 0.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct NccBaseline {
    /// The distinct event times in ascending order.
    pub times: Vec<f64>,
    /// The cumulative hazard at each of them.
    pub cumulative_hazard: Vec<f64>,
    /// The largest exit time among the rows of the fit. The baseline is not
    /// known after it.
    pub last_time: f64,
}

/// The baseline hazard of the weighted fit of [`fit_weighted`], at the
/// covariates 0 and the offset 0.
///
/// `fit`, `data`, `x`, `weights` and `opts` are those of the fit.
///
/// # Errors
///
/// An input error as for [`cox::basehaz`](crate::cox::basehaz), for `data`
/// with weights or clusters, and for data with strata.
pub fn baseline(
    fit: &CoxModelFit,
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    weights: &[f64],
    opts: &CoxOptions,
) -> Result<NccBaseline> {
    if data.stratum().is_some() {
        return Err(Error::not_applicable(
            "data: the baseline of a stratified fit is not provided",
        ));
    }
    let clusters: Vec<u32> = (0..data.n_rows() as u32).collect();
    let weighted = weighted_data(data, weights, &clusters)?;
    let zeros = vec![0.0; x.ncols()];
    let model = FittedModel {
        data: &weighted,
        x,
        fit,
        options: opts,
    };
    let result = basehaz(
        &model,
        &BaselineOptions {
            centring: Centring::Given(zeros.clone()),
            ..BaselineOptions::default()
        },
    )?;
    let stratum = result
        .strata
        .first()
        .ok_or_else(|| Error::no_information("fit: the baseline has no stratum"))?;
    let last_time = weighted
        .followup()
        .stop()
        .iter()
        .zip(weights)
        .filter(|&(_, &w)| w > 0.0)
        .map(|(&t, _)| t)
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(NccBaseline {
        times: stratum.time.clone(),
        cumulative_hazard: stratum
            .cumulative_hazard
            .iter()
            .map(|h| h * result.uncentring_factor)
            .collect(),
        last_time,
    })
}

/// The absolute risk `1 - exp(-Lambda_0(t) exp(x' beta))` of each covariate
/// profile at each time.
///
/// `new_x` has one profile per row on the design of the fit, and `times` are the
/// times to evaluate at, in any order. The result has one vector per profile.
/// A coefficient that is `None` counts as 0. The risk before the first event
/// time is 0, and it is `None` for a time after the last exit time of the fit,
/// where the baseline is not known.
///
/// # Errors
///
/// An input error for profiles of another width than the coefficients, a
/// profile value or a time that is not finite.
pub fn absolute_risk(
    baseline: &NccBaseline,
    coefficients: &[Option<f64>],
    new_x: MatRef<'_, f64>,
    times: &[f64],
) -> Result<Vec<Vec<Option<f64>>>> {
    check_lengths(&[
        ("coefficients", coefficients.len()),
        ("new_x columns", new_x.ncols()),
    ])?;
    refuse_non_finite_matrix(new_x, "new_x")?;
    refuse_non_finite(times, "times")?;
    let hazard: Vec<Option<f64>> = times
        .iter()
        .map(|&t| {
            if t > baseline.last_time {
                return None;
            }
            let k = baseline.times.partition_point(|&s| s <= t);
            Some(if k == 0 {
                0.0
            } else {
                baseline.cumulative_hazard[k - 1]
            })
        })
        .collect();
    Ok((0..new_x.nrows())
        .map(|i| {
            let mut eta = 0.0;
            for (j, c) in coefficients.iter().enumerate() {
                eta += c.unwrap_or(0.0) * new_x[(i, j)];
            }
            let risk = eta.exp();
            hazard
                .iter()
                .map(|h| h.map(|h| -(-h * risk).exp_m1()))
                .collect()
        })
        .collect())
}
