//! Survival curves for single-event data: the Kaplan-Meier product limit, the
//! Nelson-Aalen and Fleming-Harrington cumulative hazards and the survival
//! curve `exp(-H)`, with standard errors, confidence limits, quantiles and
//! restricted means, as `survival::survfit` computes them.
//!
//! # Algorithm
//!
//! A group is a stratum of the data, or all rows when there is no stratum
//! column. Row `i` is at risk on `(start_i, stop_i]`, with `start_i` absent
//! meaning from the beginning of follow-up, and has weight `w_i` and status 0
//! or 1. At each time `t` in the group's reporting times, with `Y(t)` the weight
//! at risk, `d(t)` the weight of events and `m(t)` their number, the estimates
//! are
//!
//! ```text
//! Kaplan-Meier (stype 1)        S(t) = prod over s <= t of (1 - d(s) / Y(s))
//! Nelson-Aalen (ctype 1)        H(t) = sum over s <= t of d(s) / Y(s)
//! Fleming-Harrington (ctype 2)  H(t) = sum over s <= t, j = 0..m(s)-1 of
//!                                         (d(s) / m(s)) / (Y(s) - j d(s) / m(s))
//! exp(-H) (stype 2)             S(t) = exp(-H(t))
//! ```
//!
//! The reporting times are the times at which an event happens or a chain of
//! intervals of one subject ends, so that a subject with several rows, given
//! through the id column, is censored once and not at each row's end. Only rows
//! of positive weight are used. The tie rule of [`TimeFix`] is applied first.
//!
//! `survfit` calls the combinations by the number `2 stype + ctype - 2`. The
//! default, stype 1 with ctype 1, is the Kaplan-Meier curve with the
//! Nelson-Aalen hazard. Stype 2 with ctype 1 is what `survfit` calls
//! `fleming-harrington` and stype 2 with ctype 2 `fh2`. Stype 1 with ctype 2 is
//! a Kaplan-Meier curve with the Fleming-Harrington hazard.
//!
//! # Variance
//!
//! `survfit` chooses between two variances. The default rule, which `robust:
//! None` follows, is the infinitesimal jackknife when there is a cluster
//! column, or a case weight that is not a whole number, or a subject with more
//! than one event row, and the model-based variance otherwise.
//!
//! - Greenwood [1] for the product limit: `Var log S(t) = sum d / (Y (Y - d))`.
//!   For the hazard `Var H(t) = sum d / Y^2`, and for ctype 2 the sum of
//!   `(d/m) / (Y - j d/m)^2` over the events. With stype 2 the variance of
//!   `log S` is that of `H`. With case weights the weights are used as they are,
//!   so whole-number weights act as counts of identical subjects.
//! - The infinitesimal jackknife, in [`jackknife`]'s notation: the sum over
//!   clusters of the squared derivative of the curve with respect to the weight
//!   of the cluster. The clusters are the cluster column, else the id column,
//!   else each row. It is computed in a sort and one pass.
//!
//! The standard error reported is that of `S`, which is what
//! `summary.survfit` prints. `survfit` stores that of `log S` for the
//! Greenwood variance, and for the jackknife that of `S`.
//!
//! The confidence limits follow `survfit`'s rules at the edges: an undefined
//! limit is `None`, and a zero standard error gives the point estimate itself.
//! The limits of type `Log` are `S exp(-+ z se)`, of type `LogLog`
//! `exp(-exp(log(-log S) -+ z se / log S))`, `Plain` `S -+ z se S`, `Logit` and
//! `Arcsin` the usual transformations [2][3]. The lower limit of the Peto and
//! modified variants uses another standard error [4][5].
//!
//! # Quantiles and restricted means
//!
//! [`GroupCurve::quantiles`] gives the smallest time at which the curve is at
//! or below `1 - p`, the midpoint of a flat part of the curve that lies exactly
//! at the level, `None` when the curve does not reach it, and the same for the
//! limits [6]. [`GroupCurve::rmean`] integrates the curve from its origin to
//! a horizon `tau`, holding the last value beyond the last time, with the
//! variance `sum_i A_i^2 d_i / (Y_i (Y_i - d_i))`, where `A_i` is the area under
//! the curve from the `i`-th time to `tau` [7]. That variance is Greenwood's
//! whatever variance the curve itself uses, as in `survfit`.
//!
//! # Differences from survfit
//!
//! - Rows of weight 0 are not used, so they appear in no count. `survfit`
//!   counts them in `n.risk` and lists their times.
//! - Under the jackknife variance `survfit` records the scale of its standard
//!   error wrongly when stype and ctype differ. For stype 1 with ctype 2 it
//!   treats the standard error of `S` as that of `log S`, and for stype 2 with
//!   ctype 1 the standard error of `log S` as that of `S`. Its summary standard
//!   error and its limits are wrong in those two cases. The standard errors and
//!   limits here are formed on the right scale, and so differ from `survfit`'s
//!   there. The value `survfit` stores in `std.err` is the right one.
//! - A restricted mean up to a horizon before the first time is computed,
//!   `survfit`'s print method refuses it.
//! - A `robust` setting of `Some(false)` with a cluster column is an error,
//!   where `survfit` warns and ignores the cluster.
//! - The median of `print.survfit` is not computed. It differs from
//!   `quantile` only where the curve ends exactly at one half.
//!
//! # References
//!
//! 1. Greenwood M. (1926) The natural duration of cancer, Reports on Public
//!    Health and Medical Subjects 33.
//! 2. Klein J. P., Moeschberger M. L. (2003) Survival Analysis, 2nd ed.,
//!    sections 4.2 to 4.3.
//! 3. Borgan Ø., Liestøl K. (1990) A note on confidence intervals and bands for
//!    the survival function based on transformations, Scand J Stat 17: 35-41.
//! 4. Peto R. et al. (1977) Design and analysis of randomized clinical trials
//!    requiring prolonged observation of each patient II, Br J Cancer 35: 1-39.
//! 5. Dorey F. J., Korn E. L. (1987) Effective sample sizes for confidence
//!    intervals for survival probabilities, Stat Med 6: 679-87.
//! 6. Brookmeyer R., Crowley J. (1982) A confidence interval for the median
//!    survival time, Biometrics 38: 29-41.
//! 7. Miller R. G. (1981) Survival Analysis, Wiley, pp. 195-6.
//! 8. Nelson W. (1972) Theory and applications of hazard plotting, Technometrics
//!    14: 945-66. Aalen O. O. (1978) Nonparametric inference for a family of
//!    counting processes, Ann Stat 6: 701-26. Fleming T. R., Harrington D. P.
//!    (1984) Nonparametric estimation of the survival distribution in censored
//!    data, Commun Stat A 13: 2469-86.
//! 9. Therneau T. M., Grambsch P. M. (2000) Modeling Survival Data, Springer.

mod confint;
mod group;
mod jackknife;
mod quantile;
mod rmean;

use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

pub use rmean::RestrictedMean;

/// The estimator of the survival curve, `stype` in `survfit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurvivalType {
    /// The Kaplan-Meier product limit, `stype = 1`.
    #[default]
    KaplanMeier,
    /// `exp(-H)` of the cumulative hazard, `stype = 2`.
    ExpCumulativeHazard,
}

/// The estimator of the cumulative hazard, `ctype` in `survfit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HazardType {
    /// Nelson-Aalen, `ctype = 1`.
    #[default]
    NelsonAalen,
    /// Fleming-Harrington, which spreads tied events over the risk set,
    /// `ctype = 2`.
    FlemingHarrington,
}

/// The transformation of the confidence limits, `conf.type` in `survfit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfType {
    /// Limits on the log scale.
    #[default]
    Log,
    /// Limits on the log-log scale.
    LogLog,
    /// Limits on the probability scale.
    Plain,
    /// Limits on the logit scale.
    Logit,
    /// Limits on the arcsine square-root scale.
    Arcsin,
    /// No limits.
    None,
}

/// The standard error of the lower limit, `conf.lower` in `survfit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfLower {
    /// The curve's own standard error.
    #[default]
    Usual,
    /// Peto's, `sqrt((1 - S) / Y)` on the log scale.
    Peto,
    /// The curve's standard error scaled by `sqrt(Y_last / Y)`, where `Y_last`
    /// is the weight at risk at the last time with an event.
    Modified,
}

/// Which variance a curve's standard errors come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variance {
    /// Greenwood's formula and the variances of the hazard estimators.
    Greenwood,
    /// The infinitesimal jackknife over clusters.
    InfinitesimalJackknife,
}

/// What [`estimate`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvfitOptions {
    /// How near-equal times are treated. The default is `survival`'s rule,
    /// see [`TimeFix`].
    pub timefix: TimeFix,
    /// The survival estimator.
    pub survival: SurvivalType,
    /// The hazard estimator.
    pub hazard: HazardType,
    /// The confidence limit transformation.
    pub conf_type: ConfType,
    /// The confidence level, in `(0, 1)`. Not used with [`ConfType::None`].
    pub conf_level: f64,
    /// The standard error of the lower limit.
    pub conf_lower: ConfLower,
    /// `Some(true)` for the infinitesimal jackknife variance, `Some(false)`
    /// for the model-based one, and `None` for `survfit`'s rule, see the
    /// [module documentation](self).
    pub robust: Option<bool>,
}

impl Default for SurvfitOptions {
    /// `survfit`'s defaults: Kaplan-Meier with Nelson-Aalen, log limits at
    /// 95 per cent, the usual lower limit and `survfit`'s choice of variance.
    fn default() -> Self {
        Self {
            timefix: TimeFix::default(),
            survival: SurvivalType::default(),
            hazard: HazardType::default(),
            conf_type: ConfType::default(),
            conf_level: 0.95,
            conf_lower: ConfLower::default(),
            robust: None,
        }
    }
}

/// The curve of one group, one entry per reporting time in each vector.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupCurve {
    /// The caller's stratum code, 0 without a stratum column.
    pub group: u32,
    /// The number of rows used in the group.
    pub n: usize,
    /// The number of subjects, when there is an id column.
    pub n_id: Option<usize>,
    /// Where the curve starts, the smaller of 0 and every time.
    pub origin: f64,
    /// The reporting times, ascending.
    pub time: Vec<f64>,
    /// The weight at risk at each time, `n.risk` in `survfit`.
    pub n_risk: Vec<f64>,
    /// The weight of events at each time, `n.event`.
    pub n_event: Vec<f64>,
    /// The weight censored at each time, `n.censor`.
    pub n_censor: Vec<f64>,
    /// The number of rows at risk, `counts[, "nrisk"]`.
    pub count_risk: Vec<u32>,
    /// The number of events, `counts[, "nevent"]`.
    pub count_event: Vec<u32>,
    /// The number of censorings, `counts[, "ncensor"]`.
    pub count_censor: Vec<u32>,
    /// The survival probability.
    pub surv: Vec<f64>,
    /// The standard error of the survival probability, `None` where it is
    /// infinite, which is where Greenwood's variance meets a curve at zero.
    pub se: Vec<Option<f64>>,
    /// The lower confidence limit, `None` where undefined. Empty with
    /// [`ConfType::None`].
    pub lower: Vec<Option<f64>>,
    /// The upper confidence limit, `None` where undefined. Empty with
    /// [`ConfType::None`].
    pub upper: Vec<Option<f64>>,
    /// The cumulative hazard, of the estimator chosen as the hazard.
    pub cumhaz: Vec<f64>,
    /// The standard error of the cumulative hazard.
    pub se_cumhaz: Vec<f64>,
}

/// The estimate of every group, and how it was made.
#[derive(Debug, Clone, PartialEq)]
pub struct Survfit {
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight, which are the rows used.
    pub n_used: usize,
    /// The variance the standard errors come from.
    pub variance: Variance,
    /// The confidence level of the limits.
    pub conf_level: f64,
    /// The transformation of the limits.
    pub conf_type: ConfType,
    /// One curve per stratum that has rows, in the order of the stratum codes.
    pub groups: Vec<GroupCurve>,
}

/// One quantile of a curve, with the limits of its confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantile {
    /// The probability asked for.
    pub prob: f64,
    /// The time at which the curve reaches `1 - prob`, `None` when it does
    /// not.
    pub time: Option<f64>,
    /// The time at which the lower confidence limit of the curve does. `None`
    /// when it does not, and without confidence limits.
    pub lower: Option<f64>,
    /// The time at which the upper confidence limit of the curve does.
    pub upper: Option<f64>,
}

impl GroupCurve {
    /// The quantiles of the curve at the probabilities `probs`, each with the
    /// confidence interval that the curve's limits give.
    ///
    /// Time `t` is the quantile of `p` where the distribution function
    /// `1 - S` first reaches `p`. Where it is flat at `p`, within `sqrt(eps)`,
    /// it is the midpoint of the flat part, and at a flat part that ends the
    /// curve, the midpoint to the last time. As `survfit`'s `quantile`, a
    /// probability of 0 gives 0, and one that is positive but below `sqrt(eps)`
    /// gives `None`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when a probability is outside `[0, 1]` or NaN.
    pub fn quantiles(&self, probs: &[f64]) -> Result<Vec<Quantile>> {
        if let Some(bad) = probs.iter().find(|p| !(0.0..=1.0).contains(*p)) {
            return Err(Error::input(format!(
                "quantile probability {bad} is outside [0, 1]"
            )));
        }
        let mut x = Vec::with_capacity(self.time.len() + 1);
        x.push(0.0);
        x.extend_from_slice(&self.time);
        let complement = |values: &mut dyn Iterator<Item = f64>| -> Vec<f64> {
            std::iter::once(0.0)
                .chain(values.map(|v| 1.0 - v))
                .collect()
        };
        let centre = quantile::find(&x, &complement(&mut self.surv.iter().copied()), probs);
        let (lower, upper) = if self.lower.is_empty() {
            (vec![None; probs.len()], vec![None; probs.len()])
        } else {
            let nan = |v: &Option<f64>| v.unwrap_or(f64::NAN);
            (
                quantile::find(&x, &complement(&mut self.lower.iter().map(nan)), probs),
                quantile::find(&x, &complement(&mut self.upper.iter().map(nan)), probs),
            )
        };
        Ok(probs
            .iter()
            .enumerate()
            .map(|(i, &prob)| Quantile {
                prob,
                time: centre[i],
                lower: lower[i],
                upper: upper[i],
            })
            .collect())
    }

    /// The median and the limits of its confidence interval, which is
    /// `quantiles(&[0.5])`.
    ///
    /// # Errors
    ///
    /// Never, in practice. The signature follows [`quantiles`](Self::quantiles).
    pub fn median(&self) -> Result<Quantile> {
        Ok(self.quantiles(&[0.5])?[0])
    }

    /// The restricted mean survival time up to `tau` and its standard error.
    ///
    /// See the [module documentation](self) for the formulas.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `tau` is not finite or is before the origin of
    /// the curve.
    pub fn rmean(&self, tau: f64) -> Result<RestrictedMean> {
        rmean::restricted_mean(self, tau)
    }
}

impl Survfit {
    /// The curve of the stratum with the caller's code `group`.
    #[must_use]
    pub fn group(&self, group: u32) -> Option<&GroupCurve> {
        self.groups.iter().find(|g| g.group == group)
    }
}

/// The survival curves of every stratum of `data`.
///
/// Status 0 is censoring and 1 the event, so `data` must have one event code.
/// Use [`aalen_johansen::estimate`](crate::aalen_johansen::estimate) for
/// competing events. The strata are the groups, labelled by the caller's
/// codes, and without a stratum column there is one group, coded 0. The tie
/// rule of `opts` is applied to the times first. Rows of weight 0 are not
/// used. They count in `n_rows` and not in `n_used`.
///
/// The id column gives the subjects of a counting-process data set. It decides
/// which times are reported and the censoring counts, and it is the cluster of
/// the jackknife variance when there is no cluster column.
///
/// # Errors
///
/// [`Error::Input`] when `data` has more than one event code, no row has
/// positive weight, the tie rule fails (see [`SurvData::with_timefix`]), the
/// confidence level is not in `(0, 1)`, `robust` is `Some(false)` with a
/// cluster column, or `robust` is `Some(true)` for counting-process data with
/// neither a cluster nor an id column.
///
/// # Panics
///
/// Never on valid data. The `expect`s mark invariants of the checked input.
#[expect(
    clippy::float_cmp,
    reason = "a weight is whole when it equals its floor"
)]
pub fn estimate(data: &SurvData<'_>, opts: &SurvfitOptions) -> Result<Survfit> {
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup()?;
    if surv.n_codes() != 1 {
        return Err(Error::input(format!(
            "survfit takes one event code and the data has {}; \
             use aalen_johansen::estimate for competing events",
            surv.n_codes()
        )));
    }
    if opts.conf_type != ConfType::None && !(opts.conf_level > 0.0 && opts.conf_level < 1.0) {
        return Err(Error::input(format!(
            "the confidence level {} is not in (0, 1)",
            opts.conf_level
        )));
    }
    if data.n_used() == 0 {
        return Err(Error::input("no row has positive weight"));
    }
    let counting = surv.start().is_some();
    let has_cluster = data.cluster().is_some();
    let has_id = data.id().is_some();
    let robust = match opts.robust {
        Some(true) => {
            if counting && !has_cluster && !has_id {
                return Err(Error::input(
                    "the jackknife variance of counting-process data needs a cluster or an id column",
                ));
            }
            true
        }
        Some(false) => {
            if has_cluster {
                return Err(Error::input(
                    "a cluster column needs the jackknife variance, but robust is false",
                ));
            }
            false
        }
        None => {
            has_cluster
                || data
                    .weight()
                    .is_some_and(|w| w.iter().any(|&w| w != w.floor()))
                || (has_id && has_repeated_events(&data))
        }
    };
    let variance = if robust {
        Variance::InfinitesimalJackknife
    } else {
        Variance::Greenwood
    };
    let z = if opts.conf_type == ConfType::None {
        f64::NAN
    } else {
        epistat_stats::normal::quantile(1.0 - (1.0 - opts.conf_level) / 2.0)
    };
    let settings = group::Settings {
        survival: opts.survival,
        hazard: opts.hazard,
        conf_type: opts.conf_type,
        conf_lower: opts.conf_lower,
        z,
        variance,
    };

    let origin = used_origin(&data, surv);
    let n_groups = data.n_strata().max(1);
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_groups];
    for row in data.used_rows() {
        let g = data.stratum().map_or(0, |s| s[row as usize] as usize);
        members[g].push(row);
    }
    let cluster_codes = data.cluster().or_else(|| data.id());
    let n_levels = if data.cluster().is_some() {
        data.n_clusters()
    } else {
        data.n_ids()
    };
    let mut lookup = vec![u32::MAX; if robust { n_levels } else { 0 }];
    let weight = data.weight();
    let mut groups = Vec::new();
    for (g, rows) in members.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let pick = |column: &[f64]| rows.iter().map(|&r| column[r as usize] + 0.0).collect();
        let cluster: Vec<u32>;
        let n_clusters;
        if robust {
            if let Some(codes) = cluster_codes {
                let mut next = 0u32;
                cluster = rows
                    .iter()
                    .map(|&r| {
                        let slot = &mut lookup[codes[r as usize] as usize];
                        if *slot == u32::MAX {
                            *slot = next;
                            next += 1;
                        }
                        *slot
                    })
                    .collect();
                for &r in rows {
                    lookup[codes[r as usize] as usize] = u32::MAX;
                }
                n_clusters = next as usize;
            } else {
                cluster = (0..u32::try_from(rows.len()).expect("rows fit u32")).collect();
                n_clusters = rows.len();
            }
        } else {
            cluster = Vec::new();
            n_clusters = 0;
        }
        let input = group::GroupRows {
            start: surv.start().map(&pick),
            stop: pick(surv.stop()),
            event: rows
                .iter()
                .map(|&r| surv.status()[r as usize] == 1)
                .collect(),
            weight: rows
                .iter()
                .map(|&r| weight.map_or(1.0, |w| w[r as usize]))
                .collect(),
            id: data
                .id()
                .map(|id| rows.iter().map(|&r| id[r as usize]).collect()),
            cluster,
            n_clusters,
        };
        let mut curve = group::fit(&input, &settings, origin);
        curve.group = match data.stratum_levels() {
            Some(levels) => levels.code(g).expect("a group is a level of the stratum"),
            None => 0,
        };
        groups.push(curve);
    }
    Ok(Survfit {
        n_rows: data.n_rows(),
        n_used: data.n_used(),
        variance,
        conf_level: opts.conf_level,
        conf_type: opts.conf_type,
        groups,
    })
}

/// Whether some subject has more than one event row, which `survfit` takes as
/// a reason for the jackknife variance.
fn has_repeated_events(data: &SurvData<'_>) -> bool {
    let (Some(id), Ok(surv)) = (data.id(), data.followup()) else {
        return false;
    };
    let mut seen = vec![false; data.n_ids()];
    for row in data.used_rows() {
        let r = row as usize;
        if surv.status()[r] == 1 {
            if seen[id[r] as usize] {
                return true;
            }
            seen[id[r] as usize] = true;
        }
    }
    false
}

/// The origin of the curves: the smaller of 0 and every start and stop time.
fn used_origin(data: &SurvData<'_>, surv: &crate::surv::Surv<'_>) -> f64 {
    let mut origin = 0.0_f64;
    for row in data.used_rows() {
        let r = row as usize;
        origin = origin.min(surv.stop()[r]);
        if let Some(start) = surv.start() {
            origin = origin.min(start[r]);
        }
    }
    origin
}

#[cfg(test)]
mod tests;
