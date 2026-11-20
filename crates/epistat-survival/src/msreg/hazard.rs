//! The cumulative hazards of a covariate profile, `mstate::msfit`.

use super::MsModel;
use crate::cox::predict::{
    BaselineOptions, Centring, FittedModel, HazardType, StratumBaseline, Variance, basehaz,
};
use crate::cox::{CoxModelFit, CoxOptions};
use crate::error::{Error, Result};
use crate::timefix::TimeFix;

/// The options of [`MsModel::msfit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsFitOptions {
    /// The form of the baseline hazard at tied events. `None` follows the ties
    /// of the fit, as `survfit.coxph` does and `msfit` with it.
    pub hazard: Option<HazardType>,
    /// The covariance of the coefficients the variances use. The model-based
    /// matrix by default, which is the one `mstate` takes since it refuses a
    /// cluster term.
    pub variance: Variance,
    /// Whether to give the variances and covariances, `variance` of `msfit`.
    pub se: bool,
    /// An offset of the profile, added to the linear predictor of every
    /// transition, as the offset of `newdata`.
    pub offset: f64,
}

impl Default for MsFitOptions {
    fn default() -> Self {
        Self {
            hazard: None,
            variance: Variance::Model,
            se: true,
            offset: 0.0,
        }
    }
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
    /// The covariance of the cumulative hazards at each time, by pair of
    /// transitions `(a, b)` with `a <= b` in the order of `a` and then `b`, see
    /// [`pair_index`](Self::pair_index). `None` without variances.
    pub covariance: Option<Vec<Vec<f64>>>,
}

impl CumulativeHazards {
    /// The index of the pair `(a, b)` of transitions in
    /// [`covariance`](Self::covariance), for `a <= b`.
    #[must_use]
    pub fn pair_index(&self, a: usize, b: usize) -> usize {
        let k = self.transitions.len();
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        a * k - (a * a - a) / 2 + (b - a)
    }

    /// The covariance of the cumulative hazards of transitions `a` and `b` at
    /// each time, `None` without variances. With `a == b` it is the variance.
    #[must_use]
    pub fn covariance_of(&self, a: usize, b: usize) -> Option<&[f64]> {
        self.covariance
            .as_ref()
            .map(|c| c[self.pair_index(a, b)].as_slice())
    }
}

/// The covariance matrix of the coefficients, row-major with zeros for aliased
/// coefficients.
fn covariance_matrix(fit: &CoxModelFit, variance: Variance) -> Result<Vec<f64>> {
    let matrix = match variance {
        Variance::Model => &fit.covariance,
        Variance::Robust => fit.robust_covariance.as_ref().ok_or_else(|| {
            Error::input("variance: the robust covariance needs `robust` in the fit")
        })?,
        Variance::Auto => fit.robust_covariance.as_ref().unwrap_or(&fit.covariance),
    };
    Ok(matrix.iter().map(|v| v.unwrap_or(0.0)).collect())
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
    /// [`Error::Input`] when `profile` is not as long as the number of covariate
    /// columns, has a value that is not finite, or `opts.offset` is not finite,
    /// when `opts.variance` asks for a robust covariance the fit lacks, and as
    /// [`basehaz`] for the baseline.
    pub fn msfit(&self, profile: &[f64], opts: &MsFitOptions) -> Result<CumulativeHazards> {
        if profile.len() != self.n_covariates {
            return Err(Error::input(format!(
                "profile: {} values for {} covariate columns",
                profile.len(),
                self.n_covariates
            )));
        }
        if profile.iter().any(|v| !v.is_finite()) || !opts.offset.is_finite() {
            return Err(Error::input("profile: a value is not finite"));
        }
        let cox_options = CoxOptions {
            ties: self.ties,
            timefix: TimeFix::Exact,
            cause: None,
            offset: self.offset.as_deref(),
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
        let p = self.design.ncols();
        let beta: Vec<f64> = self
            .fit
            .coefficients
            .iter()
            .map(|b| b.unwrap_or(0.0))
            .collect();
        let v = covariance_matrix(&self.fit, opts.variance)?;
        let k_n = self.transitions.len();

        // The baseline of each transition and its relative risk and covariate
        // vector relative to the reference.
        let mut strata: Vec<Option<&StratumBaseline>> = Vec::with_capacity(k_n);
        let mut theta = Vec::with_capacity(k_n);
        let mut z: Vec<Vec<f64>> = Vec::with_capacity(k_n);
        for (k, transition) in self.transitions.iter().enumerate() {
            let code = u32::try_from(transition.baseline)
                .map_err(|_| Error::input("more baselines than a u32 counts"))?;
            strata.push(baseline.strata.iter().find(|s| s.code == Some(code)));
            let mut row = vec![0.0; p];
            for (t, term) in self.terms.iter().enumerate() {
                if term.transitions.contains(&k) {
                    row[t] = term.column.map_or(1.0, |c| profile[c]);
                }
            }
            for (r, c) in row.iter_mut().zip(&baseline.centre) {
                *r -= c;
            }
            let lp: f64 = row.iter().zip(&beta).map(|(a, b)| a * b).sum();
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

        let mut cumulative_hazard = vec![vec![0.0; n_t]; k_n];
        // The partial derivative of the cumulative hazard of each transition
        // with respect to the coefficients, at each time, and the baseline
        // variance part.
        let mut gradient = vec![vec![0.0; n_t * p]; if opts.se { k_n } else { 0 }];
        let mut base_var = vec![vec![0.0; n_t]; if opts.se { k_n } else { 0 }];
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
                if opts.se {
                    base_var[k][ti] = s.cumulative_variance[j - 1];
                    for a in 0..p {
                        gradient[k][ti * p + a] = h * z[k][a] - s.cumulative_xbar[(j - 1) * p + a];
                    }
                }
            }
        }

        let covariance = opts.se.then(|| {
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
        });
        Ok(CumulativeHazards {
            transitions: self.transitions.iter().map(|t| (t.from, t.to)).collect(),
            n_states: self.n_states,
            time,
            cumulative_hazard,
            covariance,
        })
    }
}
