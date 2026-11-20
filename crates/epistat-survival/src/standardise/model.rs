//! The baselines of the outcome models on one grid of event times, and the
//! influence functions of the models at the training rows. See the module
//! documentation of [`standardise`](super) for the formulas.
//!
//! Index conventions: the grid positions are `0..J`, the position `j` is the
//! event time `s_j`. For a training row `i` with exit time `T_i`, `q_i` is the
//! number of grid times at or before `T_i`, so that `s_j <= T_i` is `j < q_i`
//! and the position of `T_i` itself, when it is a grid time, is `q_i - 1`.

use faer::MatRef;

use crate::cox::predict::{BaselineOptions, Centring, FittedModel, basehaz};
use crate::cox::{CoxModelFit, CoxOptions, Ties, score_residuals};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// One outcome model with the data and the design it was fitted on.
pub(super) struct ModelInput<'a> {
    pub(super) data: SurvData<'a>,
    pub(super) x: MatRef<'a, f64>,
    pub(super) fit: &'a CoxModelFit,
    pub(super) options: CoxOptions<'a>,
}

/// The baseline of one model in one stratum on the grid.
pub(super) struct StratumGrid {
    /// The hazard increment at each grid time, zero at a time without an event
    /// of the model in the stratum.
    pub(super) lam: Vec<f64>,
    /// The cumulative hazard at each grid time.
    pub(super) cum: Vec<f64>,
    /// The grid positions of the event times of the model in the stratum.
    pub(super) jumps: Vec<usize>,
}

pub(super) struct ModelGrid {
    pub(super) beta: Vec<f64>,
    pub(super) strata: Vec<StratumGrid>,
}

pub(super) struct Baselines {
    pub(super) grid: Vec<f64>,
    pub(super) models: Vec<ModelGrid>,
    /// The last follow-up time of every stratum, minus infinity for a stratum
    /// without a row of positive weight.
    pub(super) last_time: Vec<f64>,
    pub(super) n_strata: usize,
}

impl Baselines {
    /// The baseline hazards of the models at zero covariates, on the union of
    /// their event times.
    pub(super) fn new(models: &[ModelInput<'_>]) -> Result<Self> {
        let first = &models[0].data;
        let n_strata = first.n_strata().max(1);
        let mut bases = Vec::with_capacity(models.len());
        for (m, model) in models.iter().enumerate() {
            if model.fit.coefficients.iter().any(Option::is_none) {
                return Err(Error::input(format!(
                    "model {m}: a coefficient is not defined; drop the aliased column"
                )));
            }
            let fitted = FittedModel {
                data: &model.data,
                x: model.x,
                fit: model.fit,
                options: &model.options,
            };
            let zero = vec![0.0; model.x.ncols()];
            let opts = BaselineOptions {
                centring: Centring::Given(&zero),
                ..BaselineOptions::default()
            };
            bases.push(basehaz(&fitted, &opts)?);
        }
        let mut grid: Vec<f64> = bases
            .iter()
            .flat_map(|b| b.strata.iter().flat_map(|s| s.time.iter().copied()))
            .collect();
        grid.sort_by(f64::total_cmp);
        grid.dedup();
        let mut out = Vec::with_capacity(bases.len());
        for (m, base) in bases.iter().enumerate() {
            let beta: Vec<f64> = models[m]
                .fit
                .coefficients
                .iter()
                .map(|c| c.unwrap_or(0.0))
                .collect();
            let mut strata = Vec::with_capacity(n_strata);
            for b in &base.strata {
                let mut lam = vec![0.0; grid.len()];
                let mut jumps = Vec::with_capacity(b.time.len());
                for (t, h) in b.time.iter().zip(&b.hazard) {
                    let j = grid.binary_search_by(|g| g.total_cmp(t)).map_err(|_| {
                        Error::input("internal: an event time is missing from the grid")
                    })?;
                    lam[j] = *h;
                    jumps.push(j);
                }
                let mut cum = Vec::with_capacity(grid.len());
                let mut running = 0.0;
                for &h in &lam {
                    running += h;
                    cum.push(running);
                }
                strata.push(StratumGrid { lam, cum, jumps });
            }
            out.push(ModelGrid { beta, strata });
        }
        let surv = first.followup()?;
        let mut last_time = vec![f64::NEG_INFINITY; n_strata];
        let stratum = first.stratum();
        for i in first.used_rows() {
            let i = i as usize;
            let s = stratum.map_or(0, |c| c[i] as usize);
            last_time[s] = last_time[s].max(surv.stop()[i]);
        }
        Ok(Self {
            grid,
            models: out,
            last_time,
            n_strata,
        })
    }
}

/// The risk-set sums of one model in one stratum on the grid.
pub(super) struct StratumSums {
    /// `dL(s_j) / S0(s_j)`, 0 at a position without an event of the model.
    pub(super) lam_over_s0: Vec<f64>,
    /// The running sum of `lam_over_s0`.
    pub(super) cum_lam_over_s0: Vec<f64>,
    /// `E(s_j) dL(s_j)`, `p` values per position.
    pub(super) e_lam: Vec<f64>,
    /// The running sum of `e_lam`, `p` values per position.
    pub(super) cum_e_lam: Vec<f64>,
}

/// What one model contributes about the training rows.
pub(super) struct TrainModel {
    pub(super) p: usize,
    /// `exp(x_i' beta)` of every training row.
    pub(super) risk: Vec<f64>,
    /// The influence function of the coefficients, `n` rows of `p`.
    pub(super) if_beta: Vec<f64>,
    /// `1 / S0(T_i)` for a row with the event of the model at `T_i`, else 0.
    pub(super) event_over_s0: Vec<f64>,
    /// Whether the row has the event of the model.
    pub(super) event: Vec<bool>,
    pub(super) strata: Vec<StratumSums>,
}

/// The influence functions of the models at the training rows of positive
/// weight.
pub(super) struct Influence {
    /// `q_i` of every training row.
    pub(super) count_le: Vec<usize>,
    pub(super) stratum: Vec<usize>,
    pub(super) models: Vec<TrainModel>,
}

impl Influence {
    pub(super) fn new(models: &[ModelInput<'_>], base: &Baselines) -> Result<Self> {
        let data = &models[0].data;
        if data.followup()?.start().is_some() {
            return Err(Error::input(
                "std_err: the influence function is for data without entry times",
            ));
        }
        if data
            .weight()
            .is_some_and(|w| w.iter().any(|&w| w > 0.0 && w - 1.0 != 0.0))
        {
            return Err(Error::input(
                "std_err: the influence function is for case weights of 1",
            ));
        }
        if models.iter().any(|m| m.options.offset.is_some()) {
            return Err(Error::input(
                "std_err: the influence function is for models without an offset",
            ));
        }
        let rows: Vec<usize> = data.used_rows().map(|i| i as usize).collect();
        let stop = data.followup()?.stop();
        let stratum: Vec<usize> = rows
            .iter()
            .map(|&i| data.stratum().map_or(0, |c| c[i] as usize))
            .collect();
        let count_le: Vec<usize> = rows
            .iter()
            .map(|&i| base.grid.partition_point(|&g| g <= stop[i]))
            .collect();
        let mut trained = Vec::with_capacity(models.len());
        for (m, model) in models.iter().enumerate() {
            trained.push(train_model(
                model,
                &base.models[m],
                base,
                m,
                &rows,
                &stratum,
                stop,
            )?);
        }
        Ok(Self {
            count_le,
            stratum,
            models: trained,
        })
    }
}

fn train_model(
    model: &ModelInput<'_>,
    grid_model: &ModelGrid,
    base: &Baselines,
    m: usize,
    rows: &[usize],
    stratum: &[usize],
    stop: &[f64],
) -> Result<TrainModel> {
    let x = model.x;
    let p = x.ncols();
    let n = rows.len();
    let beta = &grid_model.beta;
    let event: Vec<bool> = {
        let surv = model.data.followup()?;
        let code = model.options.cause.unwrap_or(1);
        let status = surv.status();
        rows.iter()
            .map(|&i| i64::from(status[i]) == i64::from(code))
            .collect()
    };
    let risk: Vec<f64> = rows
        .iter()
        .map(|&i| {
            let mut eta = 0.0;
            for (a, b) in beta.iter().enumerate() {
                eta += x[(i, a)] * b;
            }
            eta.exp()
        })
        .collect();
    if risk.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(Error::input(format!(
            "model {m}: a risk score of the fit is not finite"
        )));
    }
    let v = covariance(model, m)?;
    let options = CoxOptions {
        ties: Ties::Breslow,
        ..model.options
    };
    let coefficients: Vec<Option<f64>> = beta.iter().copied().map(Some).collect();
    let scores = score_residuals(&model.data, x, &coefficients, &options)?;
    let mut if_beta = vec![0.0; n * p];
    for (i, &row) in rows.iter().enumerate() {
        for a in 0..p {
            let mut total = 0.0;
            for (b, column) in scores.iter().enumerate() {
                let u = column.as_ref().map_or(0.0, |c| c[row]);
                total += v[a * p + b] * u;
            }
            if_beta[i * p + a] = total;
        }
    }
    let n_grid = base.grid.len();
    let mut sums = Vec::with_capacity(base.n_strata);
    let mut event_over_s0 = vec![0.0; n];
    for (s, grid_stratum) in grid_model.strata.iter().enumerate() {
        let jumps = &grid_stratum.jumps;
        let lam = &grid_stratum.lam;
        let (s0, e) = risk_sets(
            rows,
            stratum,
            s,
            stop,
            &risk,
            x,
            p,
            jumps.iter().map(|&j| base.grid[j]),
        );
        let mut lam_over_s0 = vec![0.0; n_grid];
        let mut e_lam = vec![0.0; n_grid * p];
        for (l, &j) in jumps.iter().enumerate() {
            lam_over_s0[j] = lam[j] / s0[l];
            for a in 0..p {
                e_lam[j * p + a] = e[l * p + a] * lam[j];
            }
        }
        let mut cum_lam_over_s0 = Vec::with_capacity(n_grid);
        let mut running = 0.0;
        for &value in &lam_over_s0 {
            running += value;
            cum_lam_over_s0.push(running);
        }
        let mut cum_e_lam = vec![0.0; e_lam.len()];
        let mut running = vec![0.0; p];
        for j in 0..n_grid {
            for a in 0..p {
                running[a] += e_lam[j * p + a];
                cum_e_lam[j * p + a] = running[a];
            }
        }
        for (i, &row) in rows.iter().enumerate() {
            if stratum[i] == s && event[i] {
                let t = stop[row];
                let l = jumps
                    .binary_search_by(|&j| base.grid[j].total_cmp(&t))
                    .map_err(|_| {
                        Error::input("internal: an event time is not a jump of its model")
                    })?;
                event_over_s0[i] = 1.0 / s0[l];
            }
        }
        sums.push(StratumSums {
            lam_over_s0,
            cum_lam_over_s0,
            e_lam,
            cum_e_lam,
        });
    }
    Ok(TrainModel {
        p,
        risk,
        if_beta,
        event_over_s0,
        event,
        strata: sums,
    })
}

/// The covariance of the coefficients of a model: the robust one when the fit
/// has it, as `var` of a `coxph` fit.
fn covariance(model: &ModelInput<'_>, m: usize) -> Result<Vec<f64>> {
    let chosen = model
        .fit
        .robust_covariance
        .as_ref()
        .unwrap_or(&model.fit.covariance);
    let p = model.x.ncols();
    if chosen.len() != p * p {
        return Err(Error::input(format!(
            "model {m}: the covariance is not p by p"
        )));
    }
    chosen
        .iter()
        .map(|v| v.ok_or_else(|| Error::input(format!("model {m}: the covariance is not defined"))))
        .collect()
}

/// `S0` and `E` of the rows of stratum `s` at the event times, which are
/// ascending: the sum of the risk scores of the rows with exit time at or after
/// the time, and the risk-score weighted mean of their covariates. The rows are
/// added from the latest exit time down.
#[expect(
    clippy::too_many_arguments,
    reason = "the columns of one risk-set pass"
)]
fn risk_sets(
    rows: &[usize],
    stratum: &[usize],
    s: usize,
    stop: &[f64],
    risk: &[f64],
    x: MatRef<'_, f64>,
    p: usize,
    times: impl DoubleEndedIterator<Item = f64> + ExactSizeIterator,
) -> (Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..rows.len()).filter(|&i| stratum[i] == s).collect();
    order.sort_by(|&a, &b| stop[rows[a]].total_cmp(&stop[rows[b]]));
    let n_times = times.len();
    let mut s0 = vec![0.0; n_times];
    let mut e = vec![0.0; n_times * p];
    let mut s1 = vec![0.0; p];
    let mut acc = 0.0;
    let mut ptr = order.len();
    for (l, t) in times.enumerate().rev() {
        while ptr > 0 && stop[rows[order[ptr - 1]]] >= t {
            ptr -= 1;
            let i = order[ptr];
            acc += risk[i];
            for a in 0..p {
                s1[a] += risk[i] * x[(rows[i], a)];
            }
        }
        s0[l] = acc;
        for a in 0..p {
            e[l * p + a] = s1[a] / acc;
        }
    }
    (s0, e)
}
