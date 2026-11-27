//! The cumulative hazards of a covariate profile, `mstate::msfit`.

use epistat_core::Covariance;
use rayon::prelude::*;

use super::MsModel;
use crate::covariance;
use crate::cox::predict::{
    Baseline, BaselineOptions, Centring, CoxHazardType, CoxVariance, FittedModel, StratumBaseline,
    basehaz,
};
use crate::cox::{CoxModelFit, CoxOptions};
use crate::error::{Error, Result};
use crate::timefix::TimeFix;

/// The options of [`MsModel::msfit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsFitOptions {
    /// The form of the baseline hazard at tied events. `None` follows the ties
    /// of the fit, as `survfit.coxph` does and `msfit` with it.
    pub hazard: Option<CoxHazardType>,
    /// The covariance of the coefficients the variances use. The model-based
    /// matrix by default, which is the one `mstate` takes since it refuses a
    /// cluster term.
    pub variance: CoxVariance,
    /// Whether to give the variances and covariances, `variance` of `msfit`.
    pub se: bool,
    /// The form of the variance of the cumulative hazards, `vartype` of
    /// `msfit`.
    pub vartype: MsVarType,
    /// An offset of the profile, added to the linear predictor of every
    /// transition, as the offset of `newdata`.
    pub offset: f64,
}

impl Default for MsFitOptions {
    fn default() -> Self {
        Self {
            hazard: None,
            variance: CoxVariance::Model,
            se: true,
            vartype: MsVarType::Aalen,
            offset: 0.0,
        }
    }
}

/// The form of the variance of the cumulative hazards of [`MsModel::msfit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MsVarType {
    /// The variance of the Cox baseline hazard and the coefficients, the
    /// delta method of the module documentation. `vartype = "aalen"` of
    /// `msfit`.
    #[default]
    Aalen,
    /// The Greenwood form for a model without coefficients, in which each
    /// transition has a baseline hazard of its own: the variance of the
    /// cumulative hazard of a transition is the sum over its event times of
    /// `(n - d) d / n^3`, and two transitions out of one state have the
    /// covariance minus the sum of `d_a d_b / n^3` over the times at which both
    /// happen, with `n` the number at risk and `d` the number of events.
    /// Transitions out of different states are independent. `vartype =
    /// "greenwood"` of `msfit`.
    Greenwood,
}

/// The covariate values of a prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MsProfile<'a> {
    /// One value for each covariate column, the same for every transition.
    Same(&'a [f64]),
    /// One vector of values for the covariate columns for each transition, in
    /// the order of [`MsModel::transitions`]: the rows of the `newdata` of
    /// `msfit`, one for each transition.
    ByTransition(&'a [Vec<f64>]),
}

/// The cumulative hazards of every transition for one covariate profile, at
/// the event times of the model.
#[derive(Debug, Clone, PartialEq)]
pub struct CumulativeHazards {
    /// The transitions, from state and to state, in the order of the model.
    pub transitions: Vec<(u32, u32)>,
    /// The number of states, the codes being `1..=n_states`.
    pub n_states: u32,
    /// The event times of every transition, ascending, and the last time of
    /// follow-up when it is later than the last event.
    pub time: Vec<f64>,
    /// The cumulative hazard of each transition at each time, by transition.
    pub cumulative_hazard: Vec<Vec<f64>>,
    /// The covariance of the cumulative hazards at each time, one matrix over
    /// the transitions for each entry of `time`, in the order of
    /// `transitions`. `None` without variances.
    pub covariance: Option<Vec<Covariance>>,
}

impl CumulativeHazards {
    /// The position of the pair `(a, b)` of transitions, for `a <= b`, in
    /// the pairs taken in the order of `a` and then `b`.
    #[deprecated(
        since = "0.4.0",
        note = "the covariance is one matrix per time: use `covariance[t].get(a, b)` or `covariance_of`"
    )]
    #[must_use]
    pub fn pair_index(&self, a: usize, b: usize) -> usize {
        let k = self.transitions.len();
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        a * k - (a * a - a) / 2 + (b - a)
    }

    /// The covariance of the cumulative hazards of transitions `a` and `b` at
    /// each time, `None` without variances, for a transition out of range and
    /// where an entry is undefined. With `a == b` it is the variance.
    #[must_use]
    pub fn covariance_of(&self, a: usize, b: usize) -> Option<Vec<f64>> {
        self.covariance
            .as_ref()?
            .iter()
            .map(|matrix| matrix.get(a, b))
            .collect()
    }
}

/// The covariance matrices at each of `n_t` times of the pairs of `k`
/// transitions `(a, b)`, `a <= b`, taken in the order of `a` and then `b`,
/// each pair holding its `n_t` covariances.
fn by_time(pairs: &[Vec<f64>], k: usize, n_t: usize) -> Result<Vec<Covariance>> {
    let mut values = vec![0.0; n_t * k * k];
    let mut pair = 0;
    for a in 0..k {
        for b in a..k {
            for (t, &value) in pairs[pair].iter().enumerate() {
                values[(t * k + a) * k + b] = value;
                values[(t * k + b) * k + a] = value;
            }
            pair += 1;
        }
    }
    covariance::blocks(n_t, k, &values)
}

/// The covariance matrix of the coefficients, row-major with zeros for aliased
/// coefficients.
fn covariance_matrix(fit: &CoxModelFit, variance: CoxVariance) -> Result<Vec<f64>> {
    let matrix = match variance {
        CoxVariance::Model => &fit.covariance,
        CoxVariance::Robust => fit.robust_covariance.as_ref().ok_or_else(|| {
            Error::invalid("variance: the robust covariance needs `robust` in the fit")
        })?,
        CoxVariance::Auto => fit.robust_covariance.as_ref().unwrap_or(&fit.covariance),
    };
    Ok(matrix.as_slice().iter().map(|v| v.unwrap_or(0.0)).collect())
}

/// What one covariance evaluation needs of the model, computed once for any
/// number of profiles.
struct Prepared {
    baseline: Baseline,
    beta: Vec<f64>,
    v: Vec<f64>,
}

impl MsModel {
    /// The cumulative hazards of every transition for a covariate profile, the
    /// values of the covariate columns of the fit, the same for every
    /// transition, as `msfit` of `mstate`.
    ///
    /// The hazards are at every event time of any transition and, when it is
    /// later, at the last follow-up time, with the hazard carried forward. See
    /// the [module documentation](super) for the variances.
    ///
    /// # Errors
    ///
    /// An input error when `profile` is not as long as the number of covariate
    /// columns, has a value that is not finite, or `opts.offset` is not finite,
    /// when `opts.variance` asks for a robust covariance the fit lacks, when
    /// `opts.vartype` is Greenwood and the model does not allow it (see
    /// [`MsVarType::Greenwood`]), and as [`basehaz`] for the baseline.
    pub fn msfit(&self, profile: &[f64], opts: &MsFitOptions) -> Result<CumulativeHazards> {
        self.msfit_profile(&MsProfile::Same(profile), opts)
    }

    /// The cumulative hazards for one [`MsProfile`], which may differ between
    /// the transitions, as the `newdata` of `msfit` with a row for each
    /// transition.
    ///
    /// # Errors
    ///
    /// As [`msfit`](Self::msfit), and an input error when a profile by
    /// transition has not one vector for each transition.
    pub fn msfit_profile(
        &self,
        profile: &MsProfile<'_>,
        opts: &MsFitOptions,
    ) -> Result<CumulativeHazards> {
        self.check_profile(profile, opts)?;
        let prepared = self.prepare(opts)?;
        self.evaluate(&prepared, profile, opts)
    }

    /// The cumulative hazards for each of several profiles, in the order given.
    /// The baseline hazards are computed once, so this costs much less than a
    /// call of [`msfit_profile`](Self::msfit_profile) for each profile, and the
    /// profiles are evaluated in parallel with a result that does not depend on
    /// the number of threads.
    ///
    /// # Errors
    ///
    /// As [`msfit_profile`](Self::msfit_profile), for the first profile in the
    /// order given that fails.
    pub fn msfit_many(
        &self,
        profiles: &[MsProfile<'_>],
        opts: &MsFitOptions,
    ) -> Result<Vec<CumulativeHazards>> {
        for profile in profiles {
            self.check_profile(profile, opts)?;
        }
        let prepared = self.prepare(opts)?;
        let results: Vec<Result<CumulativeHazards>> = profiles
            .par_iter()
            .map(|profile| self.evaluate(&prepared, profile, opts))
            .collect();
        results.into_iter().collect()
    }

    fn check_profile(&self, profile: &MsProfile<'_>, opts: &MsFitOptions) -> Result<()> {
        let rows: Vec<&[f64]> = match profile {
            MsProfile::Same(values) => vec![values],
            MsProfile::ByTransition(rows) => {
                if rows.len() != self.transitions.len() {
                    return Err(Error::invalid(format!(
                        "profile: {} vectors for {} transitions",
                        rows.len(),
                        self.transitions.len()
                    )));
                }
                rows.iter().map(Vec::as_slice).collect()
            }
        };
        for values in rows {
            if values.len() != self.n_covariates {
                return Err(Error::invalid(format!(
                    "profile: {} values for {} covariate columns",
                    values.len(),
                    self.n_covariates
                )));
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::invalid("profile: a value is not finite"));
            }
        }
        if !opts.offset.is_finite() {
            return Err(Error::invalid("profile: a value is not finite"));
        }
        if opts.se && opts.vartype == MsVarType::Greenwood {
            if !self.terms.is_empty() {
                return Err(Error::invalid(
                    "vartype: the Greenwood variance is for a model without coefficients",
                ));
            }
            let mut groups: Vec<usize> = self.transitions.iter().map(|t| t.baseline).collect();
            groups.sort_unstable();
            if groups.windows(2).any(|w| w[0] == w[1]) {
                return Err(Error::invalid(
                    "vartype: the Greenwood variance needs a baseline hazard of its own for \
                     each transition",
                ));
            }
        }
        Ok(())
    }

    fn prepare(&self, opts: &MsFitOptions) -> Result<Prepared> {
        let cox_options = CoxOptions {
            ties: self.ties,
            timefix: TimeFix::Exact,
            cause: None,
            ..CoxOptions::default()
        };
        let data = self.stacked.as_data();
        let model = FittedModel {
            data: &data,
            x: self.design.as_ref(),
            fit: &self.fit,
            options: &cox_options,
        };
        let baseline = basehaz(
            &model,
            &BaselineOptions {
                centring: Centring::Coxph,
                hazard: opts.hazard,
                variance: opts.variance,
            },
        )?;
        let beta: Vec<f64> = self
            .fit
            .coefficients
            .iter()
            .map(|b| b.unwrap_or(0.0))
            .collect();
        let v = covariance_matrix(&self.fit, opts.variance)?;
        Ok(Prepared { baseline, beta, v })
    }

    fn evaluate(
        &self,
        prepared: &Prepared,
        profile: &MsProfile<'_>,
        opts: &MsFitOptions,
    ) -> Result<CumulativeHazards> {
        let Prepared { baseline, beta, v } = prepared;
        let p = self.design.ncols();
        let k_n = self.transitions.len();

        // The baseline of each transition and its relative risk and covariate
        // vector relative to the reference.
        let mut strata: Vec<Option<&StratumBaseline>> = Vec::with_capacity(k_n);
        let mut theta = Vec::with_capacity(k_n);
        let mut z: Vec<Vec<f64>> = Vec::with_capacity(k_n);
        for (k, transition) in self.transitions.iter().enumerate() {
            let code = u32::try_from(transition.baseline)
                .map_err(|_| Error::invalid("more baselines than a u32 counts"))?;
            strata.push(baseline.strata.iter().find(|s| s.code == Some(code)));
            let values = match profile {
                MsProfile::Same(values) => values,
                MsProfile::ByTransition(rows) => rows[k].as_slice(),
            };
            let mut row = vec![0.0; p];
            for (t, term) in self.terms.iter().enumerate() {
                if term.transitions.contains(&k) {
                    row[t] = term.column.map_or(1.0, |c| values[c]);
                }
            }
            for (r, c) in row.iter_mut().zip(&baseline.centre) {
                *r -= c;
            }
            let lp: f64 = row.iter().zip(beta).map(|(a, b)| a * b).sum();
            theta.push((lp + opts.offset - baseline.offset_centre).exp());
            z.push(row);
        }

        // The grid: every event time, then the last time of follow-up.
        let mut time: Vec<f64> = strata
            .iter()
            .flatten()
            .flat_map(|s| s.time.iter().copied())
            .collect();
        time.sort_by(f64::total_cmp);
        time.dedup();
        if time.last().is_none_or(|&t| self.last_time > t) {
            time.push(self.last_time);
        }
        let n_t = time.len();

        let greenwood = opts.se && opts.vartype == MsVarType::Greenwood;
        let mut cumulative_hazard = vec![vec![0.0; n_t]; k_n];
        // The partial derivative of the cumulative hazard of each transition
        // with respect to the coefficients, at each time, and the baseline
        // variance part.
        let with_gradient = opts.se && !greenwood;
        let mut gradient = vec![vec![0.0; n_t * p]; if with_gradient { k_n } else { 0 }];
        let mut base_var = vec![vec![0.0; n_t]; if with_gradient { k_n } else { 0 }];
        for k in 0..k_n {
            let Some(s) = strata[k] else { continue };
            let mut j = 0_usize;
            for (ti, &t) in time.iter().enumerate() {
                while j < s.time.len() && s.time[j] <= t {
                    j += 1;
                }
                if j == 0 {
                    continue;
                }
                let h = s.cumulative_hazard[j - 1];
                cumulative_hazard[k][ti] = theta[k] * h;
                if with_gradient {
                    base_var[k][ti] = s.cumulative_variance[j - 1];
                    for a in 0..p {
                        gradient[k][ti * p + a] = h * z[k][a] - s.cumulative_xbar[(j - 1) * p + a];
                    }
                }
            }
        }

        let covariance = if greenwood {
            Some(self.greenwood_covariance(&strata, &theta, &time))
        } else {
            opts.se.then(|| {
                let mut pairs = Vec::with_capacity(k_n * (k_n + 1) / 2);
                for a in 0..k_n {
                    for b in a..k_n {
                        let same_baseline =
                            self.transitions[a].baseline == self.transitions[b].baseline;
                        let scale = theta[a] * theta[b];
                        let mut values = vec![0.0; n_t];
                        for (ti, value) in values.iter_mut().enumerate() {
                            let ga = &gradient[a][ti * p..(ti + 1) * p];
                            let gb = &gradient[b][ti * p..(ti + 1) * p];
                            let mut quad = 0.0;
                            for i in 0..p {
                                let mut row = 0.0;
                                for l in 0..p {
                                    row += v[i * p + l] * gb[l];
                                }
                                quad += ga[i] * row;
                            }
                            let own = if same_baseline && strata[a].is_some() {
                                base_var[a][ti]
                            } else {
                                0.0
                            };
                            *value = scale * (quad + own);
                        }
                        pairs.push(values);
                    }
                }
                pairs
            })
        };
        let covariance = covariance
            .map(|pairs| by_time(&pairs, k_n, time.len()))
            .transpose()?;
        Ok(CumulativeHazards {
            transitions: self.transitions.iter().map(|t| (t.from, t.to)).collect(),
            n_states: self.n_states,
            time,
            cumulative_hazard,
            covariance,
        })
    }

    /// The Greenwood covariances of the cumulative hazards on the grid, by pair
    /// of transitions `(a, b)`, `a <= b`, in the order of `a` and then `b`.
    fn greenwood_covariance(
        &self,
        strata: &[Option<&StratumBaseline>],
        theta: &[f64],
        time: &[f64],
    ) -> Vec<Vec<f64>> {
        let k_n = self.transitions.len();
        let n_t = time.len();
        // The events and the number at risk of each transition on the grid, 0
        // where the transition has no event.
        let mut events = vec![vec![0.0; n_t]; k_n];
        let mut risk = vec![vec![0.0; n_t]; k_n];
        for (k, stratum) in strata.iter().enumerate() {
            let Some(s) = stratum else { continue };
            for (j, &t) in s.time.iter().enumerate() {
                let ti = time
                    .binary_search_by(|g| g.total_cmp(&t))
                    .expect("the grid holds every event time");
                events[k][ti] = s.n_events[j];
                risk[k][ti] = s.n_risk[j];
            }
        }
        let mut pairs = Vec::with_capacity(k_n * (k_n + 1) / 2);
        for a in 0..k_n {
            for b in a..k_n {
                let mut values = vec![0.0; n_t];
                let same_state = self.transitions[a].from == self.transitions[b].from;
                if a == b || same_state {
                    let scale = theta[a] * theta[b];
                    let mut sum = 0.0;
                    for (ti, value) in values.iter_mut().enumerate() {
                        let n = risk[a][ti];
                        let d = events[a][ti];
                        if d > 0.0 {
                            sum += if a == b {
                                (n - d) * d / (n * n * n)
                            } else {
                                -d * events[b][ti] / (n * n * n)
                            };
                        }
                        *value = scale * sum;
                    }
                }
                pairs.push(values);
            }
        }
        pairs
    }
}
