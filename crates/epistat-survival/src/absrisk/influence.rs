//! The baselines of cause-specific Cox models on one grid of event times, the
//! influence functions of the models at the training rows, and the influence
//! function of an absolute risk assembled from them. [`absrisk`](super) and
//! [`standardise`](crate::standardise) both read this module. See the module
//! documentation of [`absrisk`](super) for the formulas.
//!
//! Index conventions: the grid positions are `0..J`, the position `j` is the
//! event time `s_j`. For a training row `i` with exit time `T_i`, `q_i` is the
//! number of grid times at or before `T_i`, so that `s_j <= T_i` is `j < q_i`
//! and the position of `T_i` itself, when it is a grid time, is `q_i - 1`.

use epistat_linalg::MatRef;
use epistat_stats::sum::Compensated;

use crate::cox::predict::{BaselineOptions, Centring, FittedModel, basehaz, reference_offset};
use crate::cox::{CoxModelFit, CoxOptions, Ties, score_residuals};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// One outcome model with the data and the design it was fitted on.
pub(crate) struct ModelInput<'a> {
    pub(crate) data: SurvData<'a>,
    pub(crate) x: MatRef<'a, f64>,
    pub(crate) fit: &'a CoxModelFit,
    pub(crate) options: CoxOptions<'a>,
}

/// The baseline of one model in one stratum on the grid.
pub(crate) struct StratumGrid {
    /// The hazard increment at each grid time, zero at a time without an event
    /// of the model in the stratum.
    pub(crate) lam: Vec<f64>,
    /// The cumulative hazard at each grid time.
    pub(crate) cum: Vec<f64>,
    /// The grid positions of the event times of the model in the stratum.
    pub(crate) jumps: Vec<usize>,
}

pub(crate) struct ModelGrid {
    pub(crate) beta: Vec<f64>,
    /// The mean offset of the fit, which the baseline is measured from, 0
    /// without an offset. The risk score of a person with offset `o` is
    /// `exp(x' beta + o - offset_centre)`.
    pub(crate) offset_centre: f64,
    pub(crate) strata: Vec<StratumGrid>,
}

pub(crate) struct Baselines {
    pub(crate) grid: Vec<f64>,
    pub(crate) models: Vec<ModelGrid>,
    /// The last follow-up time of every stratum, minus infinity for a stratum
    /// without a row of positive weight.
    pub(crate) last_time: Vec<f64>,
    pub(crate) n_strata: usize,
}

impl Baselines {
    /// The baseline hazards of the models at zero covariates, on the union of
    /// their event times.
    pub(crate) fn new(models: &[ModelInput<'_>]) -> Result<Self> {
        let first = &models[0].data;
        let n_strata = first.n_strata().max(1);
        let mut bases = Vec::with_capacity(models.len());
        let mut offset_centres = Vec::with_capacity(models.len());
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
            offset_centres.push(if model.options.offset.is_some() {
                reference_offset(&fitted)?
            } else {
                0.0
            });
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
            out.push(ModelGrid {
                beta,
                offset_centre: offset_centres[m],
                strata,
            });
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
pub(crate) struct StratumSums {
    /// `dL(s_j) / S0(s_j)`, 0 at a position without an event of the model.
    pub(crate) lam_over_s0: Vec<f64>,
    /// The running sum of `lam_over_s0`.
    pub(crate) cum_lam_over_s0: Vec<f64>,
    /// `E(s_j) dL(s_j)`, `p` values per position.
    pub(crate) e_lam: Vec<f64>,
    /// The running sum of `e_lam`, `p` values per position.
    pub(crate) cum_e_lam: Vec<f64>,
}

/// What one model contributes about the training rows.
pub(crate) struct TrainModel {
    pub(crate) p: usize,
    /// `exp(x_i' beta)` of every training row.
    pub(crate) risk: Vec<f64>,
    /// The influence function of the coefficients, `n` rows of `p`.
    pub(crate) if_beta: Vec<f64>,
    /// `1 / S0(T_i)` for a row with the event of the model at `T_i`, else 0.
    pub(crate) event_over_s0: Vec<f64>,
    /// Whether the row has the event of the model.
    pub(crate) event: Vec<bool>,
    /// The number of event times of the model, in the stratum of the row, that
    /// are before the exit time of the row. For a row with the event of the
    /// model, the last of them is the event.
    pub(crate) rank_q: Vec<usize>,
    /// The number of event times of the model, in the stratum of the row, at
    /// or before the entry time of the row, 0 without entry times.
    pub(crate) rank_q0: Vec<usize>,
    pub(crate) strata: Vec<StratumSums>,
}

/// The influence functions of the models at the training rows of positive
/// weight.
pub(crate) struct Influence {
    /// `q_i` of every training row.
    pub(crate) count_le: Vec<usize>,
    /// The number of grid times at or before the entry time of every training
    /// row, 0 without entry times.
    pub(crate) count_le0: Vec<usize>,
    /// The case weight of every training row.
    pub(crate) weight: Vec<f64>,
    pub(crate) stratum: Vec<usize>,
    pub(crate) models: Vec<TrainModel>,
}

impl Influence {
    pub(crate) fn new(models: &[ModelInput<'_>], base: &Baselines) -> Result<Self> {
        let data = &models[0].data;
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
        let count_le0: Vec<usize> = match data.followup()?.start() {
            Some(start) => rows
                .iter()
                .map(|&i| base.grid.partition_point(|&g| g <= start[i]))
                .collect(),
            None => vec![0; rows.len()],
        };
        let weight: Vec<f64> = rows
            .iter()
            .map(|&i| data.weight().map_or(1.0, |w| w[i]))
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
                &count_le,
                &count_le0,
                &weight,
                stop,
            )?);
        }
        Ok(Self {
            count_le,
            count_le0,
            weight,
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
    count_le: &[usize],
    count_le0: &[usize],
    weight: &[f64],
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
    // The baseline of `basehaz` is at the mean offset of the fit, so the risk
    // scores are measured from it and a new row has the mean offset.
    let mean_offset = grid_model.offset_centre;
    let risk: Vec<f64> = rows
        .iter()
        .map(|&i| {
            let mut eta = 0.0;
            for (a, b) in beta.iter().enumerate() {
                eta += x[(i, a)] * b;
            }
            if let Some(offset) = model.options.offset {
                eta += offset[i] - mean_offset;
            }
            eta.exp()
        })
        .collect();
    let weighted_risk: Vec<f64> = risk.iter().zip(weight).map(|(r, w)| r * w).collect();
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
            &model.data,
            rows,
            stratum,
            s,
            &weighted_risk,
            x,
            p,
            jumps.iter().map(|&j| base.grid[j]),
        )?;
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
    let rank_q: Vec<usize> = (0..n)
        .map(|i| {
            grid_model.strata[stratum[i]]
                .jumps
                .partition_point(|&j| j < count_le[i])
        })
        .collect();
    let rank_q0: Vec<usize> = (0..n)
        .map(|i| {
            grid_model.strata[stratum[i]]
                .jumps
                .partition_point(|&j| j < count_le0[i])
        })
        .collect();
    Ok(TrainModel {
        p,
        risk,
        if_beta,
        event_over_s0,
        event,
        rank_q,
        rank_q0,
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
/// ascending: the sum of `weighted_risk`, the case weight times the risk score,
/// over the rows at risk at the time, those with entry time before it and exit
/// time at or after it, and the mean of their covariates with those weights.
/// The rows are added from the latest exit time down, and with entry times the
/// rows not yet entered are taken out again, in compensated sums.
#[expect(
    clippy::too_many_arguments,
    reason = "the columns of one risk-set pass"
)]
fn risk_sets(
    data: &SurvData<'_>,
    rows: &[usize],
    stratum: &[usize],
    s: usize,
    weighted_risk: &[f64],
    x: MatRef<'_, f64>,
    p: usize,
    times: impl DoubleEndedIterator<Item = f64> + ExactSizeIterator,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let surv = data.followup()?;
    let stop = surv.stop();
    let start = surv.start();
    let mut order: Vec<usize> = (0..rows.len()).filter(|&i| stratum[i] == s).collect();
    order.sort_by(|&a, &b| stop[rows[a]].total_cmp(&stop[rows[b]]));
    let mut by_start: Vec<usize> = if start.is_some() {
        order.clone()
    } else {
        Vec::new()
    };
    if let Some(start) = start {
        by_start.sort_by(|&a, &b| start[rows[a]].total_cmp(&start[rows[b]]));
    }
    let n_times = times.len();
    let mut s0 = vec![0.0; n_times];
    let mut e = vec![0.0; n_times * p];
    let mut acc = Compensated::new();
    let mut s1 = vec![Compensated::new(); p];
    let mut ptr = order.len();
    let mut out = by_start.len();
    for (l, t) in times.enumerate().rev() {
        while ptr > 0 && stop[rows[order[ptr - 1]]] >= t {
            ptr -= 1;
            let i = order[ptr];
            acc.add(weighted_risk[i]);
            for a in 0..p {
                s1[a].add(weighted_risk[i] * x[(rows[i], a)]);
            }
        }
        if let Some(start) = start {
            while out > 0 && start[rows[by_start[out - 1]]] >= t {
                out -= 1;
                let i = by_start[out];
                acc.add(-weighted_risk[i]);
                for a in 0..p {
                    s1[a].add(-(weighted_risk[i] * x[(rows[i], a)]));
                }
            }
        }
        s0[l] = acc.value();
        for a in 0..p {
            e[l * p + a] = s1[a].value() / s0[l];
        }
    }
    Ok((s0, e))
}

/// How the influence function treats the product-limit survival of the
/// cause-specific models, `S(s_j-) = product over l < j of (1 - h_l)`, with
/// `h_l` the total hazard of the subject at the event time `s_l`.
///
/// Without the product limit the survival is the exponential of minus the
/// cumulative hazard and the two forms are the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InfluenceForm {
    /// The derivative of the product limit itself, `d log S(s_j-) = - sum over
    /// l < j of d h_l / (1 - h_l)`. The influence function is that of the
    /// estimate.
    #[default]
    Exact,
    /// The derivative of the exponential form, `d log S(s_j-) = - sum over
    /// l < j of d h_l`, as riskRegression takes it for the product limit as
    /// well. It is a first-order approximation of the exact derivative that is
    /// exact only when every `h_l` is small.
    Exponential,
}

/// The event-free survival and the absolute risk along the grid for one row.
pub(crate) struct Path {
    /// `S(s_j-)` at every grid position below `top`, valid below `valid`.
    pub(crate) surv: Vec<f64>,
    /// The risk after the jump at every grid position.
    pub(crate) cif: Vec<f64>,
    /// The first grid position at which the survival is not a probability.
    pub(crate) valid: usize,
    /// The first count of grid times at which the exact influence function is
    /// not defined, because a factor `1 - h_l` of the product limit is 0 or
    /// negative. Equal to `valid` or later when the form is not exact.
    pub(crate) std_valid: usize,
    /// `1 / (1 - h_l)` for the exact form of the product limit, else empty,
    /// which stands for 1 everywhere.
    r: Vec<f64>,
}

impl Path {
    fn weight(&self, l: usize) -> f64 {
        self.r.get(l).copied().unwrap_or(1.0)
    }
}

impl Baselines {
    /// The path of a subject with the risk scores `theta` in stratum `s` along
    /// the first `top` grid positions. The models in `survival` form the
    /// event-free survival, `cause_model` is the cause. With `exact` the
    /// weights of the exact derivative of the product limit are formed.
    #[expect(clippy::too_many_arguments, reason = "the choices of one risk path")]
    pub(crate) fn path(
        &self,
        theta: &[f64],
        s: usize,
        top: usize,
        survival: &[usize],
        cause_model: usize,
        product_limit: bool,
        exact: bool,
    ) -> Path {
        let k = cause_model;
        let hazard_k = &self.models[k].strata[s].lam;
        let mut surv = Vec::with_capacity(top);
        let mut cif = Vec::with_capacity(top);
        let mut valid = top;
        let mut std_valid = usize::MAX;
        let mut r = if exact && product_limit {
            vec![1.0; top]
        } else {
            Vec::new()
        };
        let mut s_j = 1.0;
        let mut total = 0.0;
        for (j, &increment) in hazard_k.iter().enumerate().take(top) {
            if j > 0 {
                s_j = if product_limit {
                    let mut hazard = 0.0;
                    for &m in survival {
                        hazard += theta[m] * self.models[m].strata[s].lam[j - 1];
                    }
                    if !r.is_empty() {
                        if 1.0 - hazard > 0.0 {
                            r[j - 1] = 1.0 / (1.0 - hazard);
                        } else {
                            std_valid = std_valid.min(j + 1);
                        }
                    }
                    s_j * (1.0 - hazard)
                } else {
                    let mut cumulative = 0.0;
                    for &m in survival {
                        cumulative += self.models[m].strata[s].cum[j - 1] * theta[m];
                    }
                    (-cumulative).exp()
                };
            }
            if valid == top && (s_j.is_nan() || s_j < 0.0) {
                valid = j;
            }
            total += s_j * (increment * theta[k]);
            surv.push(s_j);
            cif.push(total);
        }
        Path {
            surv,
            cif,
            valid,
            std_valid: std_valid.min(valid),
            r,
        }
    }
}

/// The fixed part of the influence function of an absolute risk: the
/// baselines, the training influence, which models form the survival, and the
/// grid counts the function is wanted at.
pub(crate) struct Shape<'a> {
    base: &'a Baselines,
    infl: &'a Influence,
    cause_model: usize,
    survival: &'a [usize],
    /// The sorted distinct positive counts of grid times.
    needed: &'a [usize],
    /// The largest of `needed`.
    top: usize,
    /// Per model and stratum, per needed count, the number of event times of
    /// the model in the stratum before the count.
    rank_jt: Vec<Vec<Vec<usize>>>,
}

/// Running sums of one stratum over the rows added.
#[derive(Default)]
struct StratumAcc {
    /// `sum w_j dL_k(s_j) / S0(s_j)` over positions below each count.
    hk: Vec<f64>,
    /// `S(s_j-) theta_k` at each position.
    w: Vec<f64>,
    /// Per model, per needed count and event time of the model, the weight of
    /// the baseline increment at the event time in the derivative of the risk
    /// at the count, `(needed index) * (jumps + 1) + (jump index)`.
    mat: Vec<Vec<f64>>,
    /// Per model, `mat` times `dL / S0` summed over the event times before
    /// each one, `(needed index) * (jumps + 1) + (jumps before)`.
    prefix: Vec<Vec<f64>>,
}

/// Sums over the rows added, the average of the influence of the models on
/// the risk when the weights of the rows add to one.
pub(crate) struct Acc {
    strata: Vec<StratumAcc>,
    /// Per model, the coefficient of the influence function of the
    /// coefficients, `(top + 1) * p`.
    g: Vec<Vec<f64>>,
}

/// The work arrays of adding one row.
pub(crate) struct Scratch {
    hk: Vec<f64>,
    wl: Vec<f64>,
    we: Vec<f64>,
    w: Vec<f64>,
    e: Vec<f64>,
    eb: Vec<f64>,
    pv: Vec<Vec<f64>>,
    et: Vec<f64>,
}

fn add(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

impl<'a> Shape<'a> {
    pub(crate) fn new(
        base: &'a Baselines,
        infl: &'a Influence,
        cause_model: usize,
        survival: &'a [usize],
        needed: &'a [usize],
    ) -> Self {
        let top = needed.last().copied().unwrap_or(0);
        let rank_jt = base
            .models
            .iter()
            .map(|model| {
                model
                    .strata
                    .iter()
                    .map(|st| {
                        needed
                            .iter()
                            .map(|&jt| st.jumps.partition_point(|&j| j < jt))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self {
            base,
            infl,
            cause_model,
            survival,
            needed,
            top,
            rank_jt,
        }
    }

    pub(crate) fn new_acc(&self) -> Acc {
        Acc {
            strata: (0..self.base.n_strata)
                .map(|_| StratumAcc::default())
                .collect(),
            g: self
                .infl
                .models
                .iter()
                .map(|t| vec![0.0; (self.top + 1) * t.p])
                .collect(),
        }
    }

    pub(crate) fn new_scratch(&self) -> Scratch {
        let pk = self.infl.models.iter().map(|t| t.p).max().unwrap_or(0);
        let top = self.top;
        Scratch {
            hk: vec![0.0; top + 1],
            wl: vec![0.0; top + 1],
            we: vec![0.0; (top + 1) * self.infl.models[self.cause_model].p],
            w: vec![0.0; top],
            e: vec![0.0; top],
            eb: vec![0.0; top + 1],
            pv: self
                .infl
                .models
                .iter()
                .map(|t| vec![0.0; (top + 1) * t.p])
                .collect(),
            et: vec![0.0; pk],
        }
    }

    fn touch(&self, acc: &mut Acc, s: usize) {
        let sa = &mut acc.strata[s];
        if !sa.hk.is_empty() {
            return;
        }
        sa.hk = vec![0.0; self.top + 1];
        sa.w = vec![0.0; self.top];
        sa.mat = self
            .base
            .models
            .iter()
            .map(|m| vec![0.0; self.needed.len() * (m.strata[s].jumps.len() + 1)])
            .collect();
    }

    /// Add the subject with covariates `xs` (one row per model), risk scores
    /// `theta` and weight `wt` in stratum `s`, whose risk path is `path`,
    /// to the sums.
    pub(crate) fn add_row(
        &self,
        acc: &mut Acc,
        wt: f64,
        s: usize,
        path: &Path,
        theta: &[f64],
        xs: &[Vec<f64>],
        sc: &mut Scratch,
    ) {
        let k = self.cause_model;
        let top = self.top;
        let pk = self.infl.models[k].p;
        let lam_k = &self.base.models[k].strata[s].lam;
        let sums_k = &self.infl.models[k].strata[s];
        sc.w.fill(0.0);
        sc.e.fill(0.0);
        for (j, &increment) in lam_k.iter().enumerate().take(top) {
            let hazard = theta[k] * increment;
            if hazard > 0.0 {
                sc.w[j] = path.surv[j] * theta[k];
                if j >= 1 {
                    sc.e[j] = path.surv[j] * hazard;
                }
            }
        }
        sc.hk[0] = 0.0;
        sc.wl[0] = 0.0;
        sc.we[..pk].fill(0.0);
        sc.eb[0] = 0.0;
        for (j, &increment) in lam_k.iter().enumerate().take(top) {
            sc.hk[j + 1] = sc.hk[j] + sc.w[j] * sums_k.lam_over_s0[j];
            sc.wl[j + 1] = sc.wl[j] + sc.w[j] * increment;
            for a in 0..pk {
                sc.we[(j + 1) * pk + a] = sc.we[j * pk + a] + sc.w[j] * sums_k.e_lam[j * pk + a];
            }
            sc.eb[j + 1] = sc.eb[j] + sc.e[j];
        }
        self.touch(acc, s);
        for &m in self.survival {
            let p = self.infl.models[m].p;
            let sums = &self.infl.models[m].strata[s];
            let lam = &self.base.models[m].strata[s].lam;
            sc.pv[m][..p].fill(0.0);
            sc.et[..p].fill(0.0);
            let mut lam_t = 0.0;
            for (j, &lam_j) in lam.iter().enumerate().take(top) {
                let c = sc.e[j] * theta[m];
                for (a, &x) in xs[m].iter().enumerate() {
                    sc.pv[m][(j + 1) * p + a] = sc.pv[m][j * p + a] + c * (sc.et[a] - x * lam_t);
                }
                let r = path.weight(j);
                for a in 0..p {
                    sc.et[a] += r * sums.e_lam[j * p + a];
                }
                lam_t += r * lam_j;
            }
            let jumps = &self.base.models[m].strata[s].jumps;
            let width = jumps.len() + 1;
            let mat = &mut acc.strata[s].mat[m];
            for (u, &jt) in self.needed.iter().enumerate() {
                for (jr, &l) in jumps[..self.rank_jt[m][s][u]].iter().enumerate() {
                    mat[u * width + jr] +=
                        wt * theta[m] * path.weight(l) * (sc.eb[jt] - sc.eb[l + 1]);
                }
            }
        }
        let sa = &mut acc.strata[s];
        for jt in 0..=top {
            sa.hk[jt] += wt * sc.hk[jt];
        }
        for j in 0..top {
            sa.w[j] += wt * sc.w[j];
        }
        for (m, train) in self.infl.models.iter().enumerate() {
            let p = train.p;
            let in_survival = self.survival.contains(&m);
            for jt in 0..=top {
                for (a, &x) in xs[m].iter().enumerate().take(p) {
                    let mut value = if in_survival {
                        sc.pv[m][jt * p + a]
                    } else {
                        0.0
                    };
                    if m == k {
                        value += x * sc.wl[jt] - sc.we[jt * pk + a];
                    }
                    acc.g[m][jt * p + a] += wt * value;
                }
            }
        }
    }

    /// Complete the sums once every row has been added.
    pub(crate) fn finish(&self, acc: &mut Acc) {
        let n_needed = self.needed.len();
        for (s, sa) in acc.strata.iter_mut().enumerate() {
            if sa.hk.is_empty() {
                continue;
            }
            sa.prefix = self
                .base
                .models
                .iter()
                .enumerate()
                .map(|(m, model)| {
                    let jumps = &model.strata[s].jumps;
                    let width = jumps.len() + 1;
                    let lam_over_s0 = &self.infl.models[m].strata[s].lam_over_s0;
                    let mut prefix = vec![0.0; n_needed * width];
                    for u in 0..n_needed {
                        for (jr, &l) in jumps.iter().enumerate() {
                            prefix[u * width + jr + 1] =
                                prefix[u * width + jr] + lam_over_s0[l] * sa.mat[m][u * width + jr];
                        }
                    }
                    prefix
                })
                .collect();
        }
    }

    /// Add the sums of another set of rows.
    pub(crate) fn merge(acc: &mut Acc, other: Acc) {
        for (a, b) in acc.strata.iter_mut().zip(other.strata) {
            if b.hk.is_empty() {
                continue;
            }
            if a.hk.is_empty() {
                *a = b;
                continue;
            }
            add(&mut a.hk, &b.hk);
            add(&mut a.w, &b.w);
            for (x, y) in a.mat.iter_mut().zip(&b.mat) {
                add(x, y);
            }
        }
        for (a, b) in acc.g.iter_mut().zip(&other.g) {
            add(a, b);
        }
    }

    /// The influence of the models on the risk at the `u`-th needed count, for
    /// the training row `i`: the case weight of the row times the derivative of
    /// the risk with respect to it.
    pub(crate) fn model_part(&self, acc: &Acc, i: usize, u: usize) -> f64 {
        let k = self.cause_model;
        let jt = self.needed[u];
        let q = self.infl.count_le[i];
        let q0 = self.infl.count_le0[i];
        let s = self.infl.stratum[i];
        // The coefficients' part carries the case weight in the score
        // residual already.
        let mut beta_part = 0.0;
        let mut rest = 0.0;
        for (train, g) in self.infl.models.iter().zip(&acc.g) {
            let p = train.p;
            let row = &train.if_beta[i * p..(i + 1) * p];
            let coefficient = &g[jt * p..(jt + 1) * p];
            for (r, c) in row.iter().zip(coefficient) {
                beta_part += r * c;
            }
        }
        let sa = &acc.strata[s];
        if sa.hk.is_empty() {
            return beta_part;
        }
        let tk = &self.infl.models[k];
        rest -= tk.risk[i] * (sa.hk[jt.min(q)] - sa.hk[jt.min(q0)]);
        if tk.event[i] && q <= jt {
            rest += sa.w[q - 1] * tk.event_over_s0[i];
        }
        for &m in self.survival {
            let train = &self.infl.models[m];
            let width = self.base.models[m].strata[s].jumps.len() + 1;
            let at = u * width;
            let rank = train.rank_q[i].min(self.rank_jt[m][s][u]);
            let rank0 = train.rank_q0[i].min(self.rank_jt[m][s][u]);
            rest += train.risk[i] * (sa.prefix[m][at + rank] - sa.prefix[m][at + rank0]);
            if train.event[i] && q <= jt {
                rest -= train.event_over_s0[i] * sa.mat[m][at + train.rank_q[i] - 1];
            }
        }
        beta_part + self.infl.weight[i] * rest
    }
}
