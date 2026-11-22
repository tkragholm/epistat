//! The Fine-Gray partial likelihood, its score, information and score
//! residuals, by one sweep over the event times of the cause, without the
//! weighted rows. The parent module documents the method.

use crate::cox::Ties;
use crate::cox::design::{Evaluation, PartialLikelihood};
use crate::cox::moments;
use crate::cox::sweep::terms;
use crate::step::Steps;

/// The value of `curve` just before `t`, with `position` the number of its
/// times below the last `t` asked for. The times asked for must not
/// decrease.
fn before_ascending(curve: &Steps, position: &mut usize, t: f64) -> f64 {
    while *position < curve.time.len() && curve.time[*position] < t {
        *position += 1;
    }
    match *position {
        0 => curve.first,
        k => curve.value[k - 1],
    }
}

/// Adds `c`, `c x` and `c x x'` to the moment cells `sums`: one cell for the
/// sum of `c`, then `p` and then `p * p`, of which the lower triangle is read.
fn add(c: f64, x: &[f64], sums: &mut [f64]) {
    sums[0] += c;
    let (m1, m2) = sums[1..].split_at_mut(x.len());
    moments::add(c, x, m1, m2);
}

/// The data of the sweep, on the scaled covariates. Individuals are numbered
/// as in the data used by the fit.
pub(super) struct Sweep {
    pub(super) p: usize,
    pub(super) ties: Ties,
    /// Covariates, `p` per individual.
    pub(super) x: Vec<f64>,
    pub(super) weight: Vec<f64>,
    /// The last time of each individual at their own weight: the time for an
    /// individual with the cause or censored, and for a competing event the
    /// event time, or the first censoring time at or after it when `G` is
    /// zero before the event.
    pub(super) end: Vec<f64>,
    pub(super) is_event: Vec<bool>,
    /// Individuals in descending order of `end`, ties in their order.
    pub(super) by_end: Vec<u32>,
    /// The competing events with `G(x-) > 0`, in order of censoring stratum
    /// and then of event time, ties in the order of the individuals:
    /// individual, event time and `w / G(x-)`. Stratum `g` is
    /// `late_offset[g]..late_offset[g + 1]`.
    pub(super) late: Vec<u32>,
    pub(super) late_time: Vec<f64>,
    pub(super) late_ratio: Vec<f64>,
    pub(super) late_offset: Vec<usize>,
    /// The censoring distribution of each stratum.
    pub(super) curves: Vec<Steps>,
    /// The distinct event times of the cause, ascending, and the individuals
    /// with the event at each: `dying[dying_offset[k]..dying_offset[k + 1]]`.
    pub(super) times: Vec<f64>,
    pub(super) dying: Vec<u32>,
    pub(super) dying_offset: Vec<usize>,
    /// The cluster code of each individual, empty without clusters, and the
    /// number of codes.
    pub(super) cluster: Vec<u32>,
    pub(super) n_clusters: usize,
}

/// The risk scores at one point: `w exp(eta)` per individual, and
/// `w exp(eta) / G(x-)` per competing event in the order of
/// [`Sweep::late`].
struct Risk {
    eta: Vec<f64>,
    own: Vec<f64>,
    late: Vec<f64>,
}

/// The per-time quantities of the score residuals: `hazard` is the sum of
/// `u / D_0` over the terms at the time, `centre` the sum of `u D_1 / D_0^2`,
/// `late_hazard` and `late_centre` the same with the fraction of the term as
/// a factor, and `mean` the mean over the terms of `D_1 / D_0`.
struct Increments {
    hazard: Vec<f64>,
    centre: Vec<f64>,
    late_hazard: Vec<f64>,
    late_centre: Vec<f64>,
    mean: Vec<f64>,
}

impl Sweep {
    fn cells(&self) -> usize {
        1 + self.p + self.p * self.p
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.x[i * self.p..(i + 1) * self.p]
    }

    fn risk(&self, beta: &[f64]) -> Option<Risk> {
        let n = self.weight.len();
        let mut eta = Vec::with_capacity(n);
        let mut risk = Vec::with_capacity(n);
        for i in 0..n {
            let value: f64 = self.row(i).iter().zip(beta).map(|(x, b)| x * b).sum();
            let weighted = self.weight[i] * value.exp();
            if !weighted.is_finite() {
                return None;
            }
            eta.push(value);
            risk.push(weighted);
        }
        let mut late = Vec::with_capacity(self.late.len());
        for (&i, &ratio) in self.late.iter().zip(&self.late_ratio) {
            let weighted = ratio * eta[i as usize].exp();
            if !weighted.is_finite() {
                return None;
            }
            late.push(weighted);
        }
        Some(Risk {
            eta,
            own: risk,
            late,
        })
    }

    /// Visit every event time in descending order with its position, the
    /// moment cells of the risk set and those of the events at the time.
    ///
    /// The risk set at `t` is the individuals with `end >= t` at their own
    /// weight, summed as the time descends, and in each censoring stratum
    /// `G(t-)` times the competing events before `t` at `w / G(x-)`, summed as
    /// the time ascends. The ascending sums are kept at checkpoints, one per
    /// block of about `sqrt(times * strata)` event times, and recomputed from
    /// there for each block, so the memory is that of the checkpoints and one
    /// block and the sums are those of one ascending pass.
    fn sweep(&self, risk: &Risk, mut visit: impl FnMut(usize, &[f64], &[f64])) {
        let cells = self.cells();
        let n_times = self.times.len();
        let groups = self.curves.len();
        if n_times == 0 {
            return;
        }
        let block = (n_times * groups).isqrt().max(1);
        let n_blocks = n_times.div_ceil(block);

        // Per stratum: the ascending sums, the next competing event to add
        // and the position in the censoring curve.
        let mut sums = vec![0.0f64; groups * cells];
        let mut next: Vec<usize> = self.late_offset[..groups].to_vec();
        let mut position = vec![0usize; groups];
        let mut g_before = vec![0.0f64; groups];
        let advance = |k: usize,
                       sums: &mut [f64],
                       next: &mut [usize],
                       position: &mut [usize],
                       g_before: &mut [f64]| {
            let t = self.times[k];
            for g in 0..groups {
                while next[g] < self.late_offset[g + 1] && self.late_time[next[g]] < t {
                    let j = next[g];
                    add(
                        risk.late[j],
                        self.row(self.late[j] as usize),
                        &mut sums[g * cells..(g + 1) * cells],
                    );
                    next[g] += 1;
                }
                g_before[g] = before_ascending(&self.curves[g], &mut position[g], t);
            }
        };

        let mut saved_sums = Vec::with_capacity(n_blocks * groups * cells);
        let mut saved_next = Vec::with_capacity(n_blocks * groups);
        let mut saved_position = Vec::with_capacity(n_blocks * groups);
        for b in 0..n_blocks {
            saved_sums.extend_from_slice(&sums);
            saved_next.extend_from_slice(&next);
            saved_position.extend_from_slice(&position);
            for k in b * block..((b + 1) * block).min(n_times) {
                advance(k, &mut sums, &mut next, &mut position, &mut g_before);
            }
        }

        let mut late_sums = vec![0.0f64; block * cells];
        let mut own = vec![0.0f64; cells];
        let mut at_risk = vec![0.0f64; cells];
        let mut dying = vec![0.0f64; cells];
        let mut entering = 0usize;
        for b in (0..n_blocks).rev() {
            let first = b * block;
            let last = ((b + 1) * block).min(n_times);
            sums.copy_from_slice(&saved_sums[b * groups * cells..(b + 1) * groups * cells]);
            next.copy_from_slice(&saved_next[b * groups..(b + 1) * groups]);
            position.copy_from_slice(&saved_position[b * groups..(b + 1) * groups]);
            for k in first..last {
                advance(k, &mut sums, &mut next, &mut position, &mut g_before);
                let slot = &mut late_sums[(k - first) * cells..(k - first + 1) * cells];
                slot.fill(0.0);
                for g in 0..groups {
                    if next[g] == self.late_offset[g] {
                        continue;
                    }
                    for (cell, &s) in slot.iter_mut().zip(&sums[g * cells..(g + 1) * cells]) {
                        *cell += g_before[g] * s;
                    }
                }
            }
            for k in (first..last).rev() {
                let t = self.times[k];
                while entering < self.by_end.len() && self.end[self.by_end[entering] as usize] >= t
                {
                    let i = self.by_end[entering] as usize;
                    add(risk.own[i], self.row(i), &mut own);
                    entering += 1;
                }
                let slot = &late_sums[(k - first) * cells..(k - first + 1) * cells];
                for ((cell, &a), &c) in at_risk.iter_mut().zip(&own).zip(slot) {
                    *cell = a + c;
                }
                dying.fill(0.0);
                for &i in &self.dying[self.dying_offset[k]..self.dying_offset[k + 1]] {
                    let i = i as usize;
                    add(risk.own[i], self.row(i), &mut dying);
                }
                visit(k, &at_risk, &dying);
            }
        }
    }

    fn terms(&self, k: usize) -> impl Iterator<Item = (f64, f64)> + use<> {
        terms(
            &self.weight,
            &self.dying[self.dying_offset[k]..self.dying_offset[k + 1]],
            self.ties == Ties::Efron,
        )
    }

    fn increments(&self, risk: &Risk) -> Option<Increments> {
        let p = self.p;
        let count = self.times.len();
        let mut out = Increments {
            hazard: vec![0.0; count],
            centre: vec![0.0; count * p],
            late_hazard: vec![0.0; count],
            late_centre: vec![0.0; count * p],
            mean: vec![0.0; count * p],
        };
        let mut failed = false;
        self.sweep(risk, |k, at_risk, dying| {
            let mut n_terms = 0.0f64;
            for (u, fraction) in self.terms(k) {
                let d0 = at_risk[0] - fraction * dying[0];
                if d0 <= 0.0 || !d0.is_finite() {
                    failed = true;
                    return;
                }
                out.hazard[k] += u / d0;
                out.late_hazard[k] += u * fraction / d0;
                for a in 0..p {
                    let d1 = at_risk[1 + a] - fraction * dying[1 + a];
                    out.centre[k * p + a] += u * d1 / (d0 * d0);
                    out.late_centre[k * p + a] += u * fraction * d1 / (d0 * d0);
                    out.mean[k * p + a] += d1 / d0;
                }
                n_terms += 1.0;
            }
            for a in 0..p {
                out.mean[k * p + a] /= n_terms;
            }
        });
        (!failed).then_some(out)
    }

    /// Hand the score residual of every individual to `emit`, as the
    /// individual and the `p` values. `None` when a risk score or a
    /// denominator is not finite or not positive.
    ///
    /// The residual is that of the individual's event, `w (x - mean)` with
    /// the correction of Efron's approximation, less `w exp(eta)` times
    /// `x h_t - c_t` summed over the event times at the individual's own
    /// weight, less `w exp(eta) / G(x-)` times `G(t-) (x h_t - c_t)` summed
    /// over the event times after a competing event.
    fn score_residuals(&self, beta: &[f64], mut emit: impl FnMut(usize, &[f64])) -> Option<()> {
        let p = self.p;
        let risk = self.risk(beta)?;
        let inc = self.increments(&risk)?;
        let count = self.times.len();
        let mut hazard_sum = vec![0.0f64; count + 1];
        let mut centre_sum = vec![0.0f64; (count + 1) * p];
        for k in 0..count {
            hazard_sum[k + 1] = hazard_sum[k] + inc.hazard[k];
            for a in 0..p {
                centre_sum[(k + 1) * p + a] = centre_sum[k * p + a] + inc.centre[k * p + a];
            }
        }
        let mut residual = vec![0.0f64; p];
        for i in 0..self.weight.len() {
            let x = self.row(i);
            // Event times t <= end.
            let span = self.times.partition_point(|&t| t <= self.end[i]);
            for a in 0..p {
                residual[a] = -risk.own[i] * (x[a] * hazard_sum[span] - centre_sum[span * p + a]);
            }
            if self.is_event[i] {
                // The individual's time is the event time `span - 1`.
                let k = span - 1;
                for a in 0..p {
                    residual[a] += self.weight[i] * (x[a] - inc.mean[k * p + a]);
                    residual[a] +=
                        risk.own[i] * (x[a] * inc.late_hazard[k] - inc.late_centre[k * p + a]);
                }
            }
            emit(i, &residual);
        }

        // After a competing event: per stratum, the event times after the
        // event in descending order, with `G(t-)` as a factor.
        let mut late_centre = vec![0.0f64; p];
        for (g, curve) in self.curves.iter().enumerate() {
            let mut late_hazard = 0.0f64;
            late_centre.fill(0.0);
            let mut k = count;
            for j in (self.late_offset[g]..self.late_offset[g + 1]).rev() {
                while k > 0 && self.times[k - 1] > self.late_time[j] {
                    k -= 1;
                    let factor = curve.before(self.times[k]);
                    late_hazard += factor * inc.hazard[k];
                    for (cell, &c) in late_centre.iter_mut().zip(&inc.centre[k * p..(k + 1) * p]) {
                        *cell += factor * c;
                    }
                }
                let i = self.late[j] as usize;
                let x = self.row(i);
                for a in 0..p {
                    residual[a] = -risk.late[j] * (x[a] * late_hazard - late_centre[a]);
                }
                emit(i, &residual);
            }
        }
        Some(())
    }
}

impl PartialLikelihood for Sweep {
    fn evaluate(&self, beta: &[f64]) -> Option<Evaluation> {
        let p = self.p;
        let risk = self.risk(beta)?;
        let mut loglik = 0.0f64;
        let mut score = vec![0.0f64; p];
        let mut information = vec![0.0f64; p * p];
        let mut d1 = vec![0.0f64; p];
        let mut quotient = vec![0.0f64; p];
        let mut failed = false;
        self.sweep(&risk, |k, at_risk, dying| {
            if failed {
                return;
            }
            for &i in &self.dying[self.dying_offset[k]..self.dying_offset[k + 1]] {
                let i = i as usize;
                loglik += self.weight[i] * risk.eta[i];
                for (total, x) in score.iter_mut().zip(self.row(i)) {
                    *total += self.weight[i] * x;
                }
            }
            for (u, fraction) in self.terms(k) {
                let d0 = at_risk[0] - fraction * dying[0];
                if d0 <= 0.0 || !d0.is_finite() {
                    failed = true;
                    return;
                }
                loglik -= u * d0.ln();
                for a in 0..p {
                    d1[a] = at_risk[1 + a] - fraction * dying[1 + a];
                    score[a] -= u * d1[a] / d0;
                }
                moments::quotients(&d1, d0, &mut quotient);
                moments::add_information(
                    p,
                    u,
                    d0,
                    fraction,
                    &at_risk[1 + p..],
                    &dying[1 + p..],
                    &quotient,
                    &mut information,
                );
            }
        });
        if failed {
            return None;
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

    fn cluster_scores(&self, beta: &[f64]) -> Option<Vec<f64>> {
        let p = self.p;
        let mut totals = vec![0.0f64; self.n_clusters * p];
        self.score_residuals(beta, |i, residual| {
            let cluster = self.cluster[i] as usize;
            for (total, r) in totals[cluster * p..(cluster + 1) * p]
                .iter_mut()
                .zip(residual)
            {
                *total += r;
            }
        })?;
        Some(totals)
    }
}
