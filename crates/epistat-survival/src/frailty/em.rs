//! The gamma frailty model by the EM algorithm of Nielsen et al. and Klein,
//! with the marginal log likelihood written out from the gamma integral. See
//! the module documentation of [`crate::frailty`].

use epistat_linalg::MatRef;
use epistat_stats::dist::gamma::ln_gamma;

use super::engine::{Layout, State};
use crate::cox::{CoxOptions, Ties};
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// What the EM iteration needs besides the layout.
pub(super) struct EmInput<'a> {
    /// The data with the rule for the times already applied.
    pub(super) data: &'a SurvData<'a>,
    /// The fitted columns of the design, one row per row of the data.
    pub(super) x: MatRef<'a, f64>,
    pub(super) cause: Option<u32>,
    /// The caller's offset, or zero, one per row of the data.
    pub(super) offset: &'a [f64],
    /// Dense cluster index of each row of the data, `u32::MAX` for a row of
    /// weight zero.
    pub(super) cluster: &'a [u32],
    pub(super) max_iterations: usize,
    pub(super) tolerance: f64,
}

/// The EM solution at one theta.
pub(super) struct EmFit {
    pub(super) state: State,
    /// The marginal log likelihood at the solution.
    pub(super) marginal: f64,
    pub(super) iterations: usize,
    pub(super) converged: bool,
}

impl Layout {
    /// The Breslow cumulative hazard of each cluster without its frailty,
    /// `Lambda_i`, and the part `sum over events of (x'beta + o + ln dH_0)` of
    /// the marginal log likelihood, at `state`, for case weights of 1.
    pub(super) fn cluster_hazards(&self, state: &State) -> Option<(Vec<f64>, f64)> {
        self.breslow_cluster_hazards(state)
    }
}

/// The gamma part of the marginal log likelihood,
/// `sum_i [ln Gamma(nu + D_i) - ln Gamma(nu) + D_i ln theta - (nu + D_i) ln(1 + theta Lambda_i)]`.
fn gamma_part(events: &[u32], lambda: &[f64], theta: f64) -> f64 {
    let nu = 1.0 / theta;
    let lg_nu = ln_gamma(nu);
    let mut total = 0.0;
    for (&d, &l) in events.iter().zip(lambda) {
        let d = f64::from(d);
        total += ln_gamma(nu + d) - lg_nu + d * theta.ln() - (nu + d) * (theta * l).ln_1p();
    }
    total
}

/// Fit the gamma frailty model at `theta` by EM from `start`.
pub(super) fn fit_em(
    layout: &Layout,
    input: &EmInput<'_>,
    theta: f64,
    start: &State,
) -> Result<EmFit> {
    let nu = 1.0 / theta;
    let rows = input.data.n_rows();
    let mut state = start.clone();
    let mut offset = vec![0.0; rows];
    let mut previous: Option<f64> = None;
    let mut marginal = f64::NAN;
    let mut converged = false;
    let mut iterations = 0;
    while iterations < input.max_iterations {
        iterations += 1;
        for (j, o) in offset.iter_mut().enumerate() {
            let c = input.cluster[j];
            *o = input.offset[j]
                + if c == u32::MAX {
                    0.0
                } else {
                    state.w[c as usize]
                };
        }
        let options = CoxOptions {
            ties: Ties::Breslow,
            timefix: TimeFix::Exact,
            cause: input.cause,
            start: Some(state.beta.clone()),
            max_iterations: 100,
            tolerance: 1e-14,
        };
        let shifted = input.data.with_offset(&offset)?;
        let fit = crate::cox::fit(&shifted, input.x, &options)?;
        let mut beta = Vec::with_capacity(layout.p);
        for c in &fit.coefficients {
            beta.push(c.ok_or_else(|| {
                Error::invalid("the Cox fit of an EM step has no estimate for a coefficient")
            })?);
        }
        state.beta = beta;
        let (lambda, events_part) = layout
            .cluster_hazards(&state)
            .ok_or_else(|| Error::undefined("a risk score of an EM step is not finite"))?;
        marginal = events_part + gamma_part(&layout.events, &lambda, theta);
        for (i, w) in state.w.iter_mut().enumerate() {
            let d = f64::from(layout.events[i]);
            *w = ((nu + d) / (nu + lambda[i])).ln();
        }
        if let Some(prev) = previous
            && (marginal - prev).abs() <= input.tolerance * (1.0 + marginal.abs())
        {
            converged = true;
            break;
        }
        previous = Some(marginal);
    }
    Ok(EmFit {
        state,
        marginal,
        iterations,
        converged,
    })
}
