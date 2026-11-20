//! Pseudo-observations of survival, cumulative incidence and restricted means,
//! from the infinitesimal jackknife as `survival::pseudo` makes them or from the
//! exact leave-one-out jackknife as the `pseudo` package does, and regression on
//! them.
//!
//! # Algorithm
//!
//! A pseudo-observation turns a censored subject into a value of the target
//! `theta` that a regression can use, without a model for the censoring \[1\]. For
//! the targets
//!
//! ```text
//! Survival                         S(t)          the event-free probability
//! CumulativeIncidence(k)           F_k(t)        the incidence of cause k
//! RestrictedMean                   integral of S from the origin to t
//! RestrictedTimeLost(k)            integral of F_k from the origin to t
//! ```
//!
//! at the times `t` asked for, within each group (stratum) of the data, with
//! `theta_hat` the estimate of the group from its `n` subjects, the
//! pseudo-observation of subject `i` is
//!
//! ```text
//! exact jackknife              n theta_hat - (n - 1) theta_hat(-i)
//! infinitesimal jackknife      theta_hat + n U_i,   U_i = sum over rows of i of
//!                                                     w dtheta_hat / dw
//! ```
//!
//! where `theta_hat(-i)` is the estimate without subject `i`, and `U_i` is the
//! influence of the subject, the derivative of the estimate with respect to the
//! case weight of the rows times the weight, summed over the subject's rows \[2\].
//! The infinitesimal one is what `survival::pseudo` computes, with its
//! `collapse = TRUE`. The estimate is the Kaplan-Meier curve of the group for the
//! survival probability and the restricted mean, which for competing causes treat
//! every cause as the event, and the Aalen-Johansen estimate for the incidence
//! and the time lost. The estimates are those of [`survfit`](crate::survfit) and
//! [`aalen_johansen`](crate::aalen_johansen), read at the times as
//! `summary.survfit(times = , extend = TRUE)` reads them, and the restricted
//! mean as `summary(, rmean = )`.
//!
//! A subject is the cluster column, else the id column, else the row. Rows of
//! weight 0 are not used. The number of subjects of a group is `n`.
//!
//! # Which jackknife
//!
//! The two are the same to first order, and differ in the second: the exact
//! jackknife is `n` times the estimate less `n - 1` times the one with the
//! subject deleted, and the infinitesimal one is `n` times the derivative. The
//! exact jackknife is the one of the theory of the regression \[3\], and the
//! infinitesimal one the one `survival` offers. `eventglm` uses neither, but a
//! censoring-weighted construction, which is not done here.
//!
//! *Cost.* The infinitesimal jackknife of the survival probability and the
//! restricted mean is in closed form, `O(rows log events + rows times)` for the
//! times asked for. The incidence and the time lost come from the recursion of
//! [`aalen_johansen::variance`](crate::aalen_johansen::variance), which costs
//! `O(subjects exit times states)`. The time lost needs the influence at every
//! exit time before the largest horizon, which takes subjects times exit times
//! times states values of memory, and
//! [`PseudoOptions::max_influence_values`] bounds it. The exact jackknife costs
//! `O(subjects exit times)` for every target. It does not refit the curve for
//! each subject, nor does it take the direct `O(n^2)` route. It edits the table
//! of risk sets and events that the estimate is read from, needs no memory beyond
//! that table and a few columns for each thread, and runs the subjects in
//! parallel. It has no size guard, since its cost is that of the influence.
//!
//! # Tie rule, weights and scale
//!
//! The tie rule of [`TimeFix`] is applied to the times first, `survival`'s by
//! default. Case weights enter as in the estimate. Under the infinitesimal
//! jackknife the influence of a row is its weight times the derivative, which is
//! the scale of the variance of the curve, the influence of a subject is the sum
//! over its rows, and the multiplier `n` is the number of subjects, as in
//! `survival::pseudo`. That function returns one value per row when subjects
//! are given by a cluster column and no id column, and multiplies by the number of rows. Here
//! the rows of a cluster are collapsed, with the number of clusters as the
//! multiplier, whichever column gives the subject.
//!
//! # Times beyond the data
//!
//! A time after the last exit time of a group is carried forward as the curve's
//! last value. [`GroupEstimate::beyond`] says which times that applies to, as
//! `pseudo` warns.
//!
//! # Regression
//!
//! [`Pseudo::regress`] fits the regression of the pseudo-values on covariates
//! and the time, see [`regression`]. Compute the pseudo-values on the whole
//! sample, without a stratum column, for a regression on a covariate that
//! defines groups.
//!
//! # References
//!
//! 1. Andersen P. K., Klein J. P., Rosthoj S. (2003) Generalised linear models
//!    for correlated pseudo-observations, with applications to multi-state
//!    models, Biometrika 90: 15-27.
//! 2. Therneau T. M., Grambsch P. M. (2000) Modeling Survival Data, Springer,
//!    section 2.9, and the vignette `methods` of the survival package.
//! 3. Overgaard M., Parner E. T., Pedersen J. (2017) Asymptotic theory of
//!    generalized estimating equations based on jack-knife pseudo-observations,
//!    Ann Stat 45: 1988-2015.

mod engine;
mod infinitesimal;
mod jackknife;
pub mod regression;
mod table;

use faer::{Mat, MatRef};

use crate::error::{Error, Result};
use crate::multistate::check_ascending;
use crate::surv::SurvData;
use crate::timefix::TimeFix;

pub use regression::{GeeFit, GeeOptions, Link};

/// What is estimated at each time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// The event-free probability, `pstate` of `survival::pseudo`. Every cause
    /// is the event.
    Survival,
    /// The cumulative incidence of a cause.
    CumulativeIncidence {
        /// The cause, from 1 to `n_codes`.
        cause: u32,
    },
    /// The restricted mean survival time up to the time, `auc` of
    /// `survival::pseudo`.
    RestrictedMean,
    /// The restricted mean time lost to a cause up to the time, the area under
    /// the cumulative incidence.
    RestrictedTimeLost {
        /// The cause, from 1 to `n_codes`.
        cause: u32,
    },
}

impl Target {
    pub(crate) fn cause(self) -> Option<u32> {
        match self {
            Self::CumulativeIncidence { cause } | Self::RestrictedTimeLost { cause } => Some(cause),
            _ => None,
        }
    }
}

/// How the pseudo-values are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Method {
    /// The infinitesimal jackknife, as `survival::pseudo`.
    #[default]
    InfinitesimalJackknife,
    /// The leave-one-out jackknife, as the `pseudo` package.
    Jackknife,
}

/// What [`pseudo`] does besides the data and the times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PseudoOptions {
    /// What is estimated. It has no default.
    pub target: Target,
    /// How the pseudo-values are made. The default is `survival::pseudo`'s.
    pub method: Method,
    /// How near-equal times are treated. The default is `survival`'s rule,
    /// see [`TimeFix`].
    pub timefix: TimeFix,
    /// The most values of influence the incidence and time-lost targets may
    /// hold at once under the infinitesimal jackknife, which is subjects times
    /// exit times times states. `None`, the default, sets no bound.
    pub max_influence_values: Option<usize>,
}

impl PseudoOptions {
    /// Options for a target, with the infinitesimal jackknife, `survival`'s tie
    /// rule and no bound on memory.
    #[must_use]
    pub fn new(target: Target) -> Self {
        Self {
            target,
            method: Method::default(),
            timefix: TimeFix::default(),
            max_influence_values: None,
        }
    }
}

/// The estimate of one group at the times.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupEstimate {
    /// The caller's stratum code, 0 without a stratum column.
    pub group: u32,
    /// The number of subjects.
    pub n_subjects: usize,
    /// The estimate at each time.
    pub estimate: Vec<f64>,
    /// Whether the time is after the last exit time of the group, so that the
    /// estimate is the curve's last value carried forward.
    pub beyond: Vec<bool>,
}

/// Pseudo-observations at a set of times.
#[derive(Debug, Clone, PartialEq)]
pub struct Pseudo {
    /// The times, ascending.
    pub times: Vec<f64>,
    /// The group of each subject, as in [`GroupEstimate::group`].
    pub group: Vec<u32>,
    /// The caller's label of each subject: the cluster column, else the id
    /// column, else the row number of the data. Subjects are in order of group,
    /// and in a group of label for the columns, and of row for rows.
    pub subject: Vec<u32>,
    /// The pseudo-observations, row-major by subject and then time.
    pub values: Vec<f64>,
    /// The estimate of each group.
    pub groups: Vec<GroupEstimate>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight, which are the rows used.
    pub n_used: usize,
}

/// The pseudo-observations of the subjects of every group at `times`.
///
/// See the [module documentation](self) for the targets, the methods and what
/// the options mean.
///
/// # Errors
///
/// [`Error::Input`] when `times` is empty, not finite or not strictly
/// ascending, or a time of a restricted mean is before the origin of the
/// curves; the data have no row of positive weight, a subject has rows in more
/// than one stratum, the tie rule fails (see [`SurvData::with_timefix`]), a
/// cause is outside `1..=n_codes`, the exact jackknife meets a group of fewer
/// than two subjects, or the bound on influence values is exceeded.
pub fn pseudo(data: &SurvData<'_>, times: &[f64], opts: &PseudoOptions) -> Result<Pseudo> {
    if times.is_empty() {
        return Err(Error::input("times is empty"));
    }
    check_ascending(times, "times", true)?;
    let data = data.clone().with_timefix(opts.timefix)?;
    if let Some(cause) = opts.target.cause() {
        data.cause(cause)?;
    }
    let groups = table::build(&data, opts.target.cause())?;
    let origin = groups[0].origin;
    if matches!(
        opts.target,
        Target::RestrictedMean | Target::RestrictedTimeLost { .. }
    ) && times[0] < origin
    {
        return Err(Error::input(format!(
            "the time {} of a restricted mean is before the start of the curves, {origin}",
            times[0]
        )));
    }
    let k = times.len();
    let mut reads = Vec::with_capacity(groups.len());
    let mut estimates = Vec::with_capacity(groups.len());
    for group in &groups {
        let read: Vec<usize> = times
            .iter()
            .map(|&t| group.times.partition_point(|&x| x <= t))
            .collect();
        let mut estimate = vec![0.0; k];
        table::evaluate(
            &table::Columns {
                times: &group.times,
                risk: &group.risk,
                events: &group.events,
                cause_events: &group.cause_events,
            },
            group.origin,
            &table::Reading {
                target: opts.target,
                at: times,
                reads: &read,
            },
            &mut estimate,
        );
        reads.push(read);
        estimates.push(estimate);
    }

    // The influence of every subject, for each group, when that is the method.
    let influences: Vec<Vec<f64>> = match opts.method {
        Method::Jackknife => Vec::new(),
        Method::InfinitesimalJackknife => match opts.target {
            Target::Survival | Target::RestrictedMean => groups
                .iter()
                .zip(&reads)
                .map(|(g, read)| {
                    infinitesimal::run(
                        g,
                        &table::Reading {
                            target: opts.target,
                            at: times,
                            reads: read,
                        },
                    )
                })
                .collect(),
            _ => engine::run(
                &data,
                &groups,
                &table::Reading {
                    target: opts.target,
                    at: times,
                    reads: &reads[0],
                },
                opts.max_influence_values,
            )?,
        },
    };

    let mut out = Pseudo {
        times: times.to_vec(),
        group: Vec::new(),
        subject: Vec::new(),
        values: Vec::new(),
        groups: Vec::new(),
        n_rows: data.n_rows(),
        n_used: data.n_used(),
    };
    for (gi, group) in groups.iter().enumerate() {
        let n = group.n_clusters();
        let reading = table::Reading {
            target: opts.target,
            at: times,
            reads: &reads[gi],
        };
        match opts.method {
            Method::Jackknife => {
                if n < 2 {
                    return Err(Error::input(format!(
                        "the jackknife needs two subjects in a group, and group {} has {n}",
                        group.code
                    )));
                }
                out.values
                    .extend(jackknife::run(group, &reading, &estimates[gi]));
            }
            Method::InfinitesimalJackknife => {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a subject count is far below 2^53"
                )]
                let n_f = n as f64;
                for c in 0..n {
                    for slot in 0..k {
                        out.values
                            .push(estimates[gi][slot] + n_f * influences[gi][c * k + slot]);
                    }
                }
            }
        }
        out.group.extend(std::iter::repeat_n(group.code, n));
        out.subject.extend_from_slice(&group.labels);
        out.groups.push(GroupEstimate {
            group: group.code,
            n_subjects: n,
            estimate: estimates[gi].clone(),
            beyond: times
                .iter()
                .map(|&t| t > group.times[group.times.len() - 1])
                .collect(),
        });
    }
    Ok(out)
}

/// The rows of a regression on pseudo-observations: one row for each subject
/// and time.
#[derive(Debug, Clone)]
pub struct Stacked {
    /// The response, the pseudo-observations by subject and then time.
    pub y: Vec<f64>,
    /// The design: an intercept, then an indicator of each time after the
    /// first when the time is a factor, then the covariates.
    pub x: Mat<f64>,
    /// The subject of each row, the cluster of the covariance.
    pub cluster: Vec<u32>,
}

impl Pseudo {
    /// The number of subjects.
    #[must_use]
    pub fn n_subjects(&self) -> usize {
        self.subject.len()
    }

    /// The stacked rows for a regression on `covariates`, one row per subject
    /// in the order of [`subject`](Self::subject), without an intercept column.
    ///
    /// With `time_effects` and more than one time the design has an indicator
    /// for each time after the first, as `eventglm`'s `factor(time)`. Without
    /// it the times share the intercept.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `covariates` does not have one row per subject.
    pub fn stacked(
        &self,
        covariates: Option<MatRef<'_, f64>>,
        time_effects: bool,
    ) -> Result<Stacked> {
        let n = self.n_subjects();
        let k = self.times.len();
        let q = covariates.map_or(0, |c| c.ncols());
        if let Some(c) = covariates {
            if c.nrows() != n {
                return Err(Error::input(format!(
                    "the covariates have {} rows and there are {n} subjects",
                    c.nrows()
                )));
            }
        }
        let t = if time_effects { k - 1 } else { 0 };
        let x = Mat::from_fn(n * k, 1 + t + q, |row, col| {
            let (subject, time) = (row / k, row % k);
            if col == 0 {
                1.0
            } else if col <= t {
                f64::from(u8::from(time == col))
            } else {
                covariates.map_or(0.0, |c| c[(subject, col - 1 - t)])
            }
        });
        Ok(Stacked {
            y: self.values.clone(),
            x,
            cluster: (0..n * k)
                .map(|row| {
                    u32::try_from(row / k).map_err(|_| Error::input("more subjects than u32::MAX"))
                })
                .collect::<Result<_>>()?,
        })
    }

    /// The regression of the pseudo-observations on `covariates`, an
    /// intercept and, for several times, an effect of each time. See
    /// [`stacked`](Self::stacked) for the design and [`regression`] for the fit.
    ///
    /// # Errors
    ///
    /// As [`stacked`](Self::stacked) and [`regression::fit`].
    pub fn regress(
        &self,
        covariates: Option<MatRef<'_, f64>>,
        time_effects: bool,
        opts: &GeeOptions,
    ) -> Result<GeeFit> {
        let s = self.stacked(covariates, time_effects)?;
        regression::fit(s.x.as_ref(), &s.y, &s.cluster, None, None, None, opts)
    }
}
