//! The penalised partial likelihood of the shared gamma frailty model: the
//! layout of the rows, one evaluation with its score and the pieces of its
//! information, the product of the whole information with a vector, the
//! preconditioned conjugate gradient solve, the Newton iteration and the
//! covariance matrices. The formulas are in the module documentation of
//! [`crate::frailty`].

use crate::cox::moments;
use crate::ldl;

/// Pivot floor of the factorisation of the coefficient block, relative to its
/// largest diagonal element, `toler.chol` of `coxph.control`.
const TOLER_CHOL: f64 = 1.818_989_403_545_856_5e-12;
/// Relative residual, in the norm of the preconditioner, at which the
/// conjugate gradient solve stops.
const CG_TOLERANCE: f64 = 1e-12;
/// Most conjugate gradient iterations for one solve.
const CG_MAX_ITERATIONS: usize = 500;
/// Most step halvings in one Newton iteration.
const MAX_HALVINGS: u32 = 40;

/// One stratum of the layout.
pub(super) struct StratumLayout {
    /// Positions of its rows in the layout, sorted by exit index, events first.
    rows: std::ops::Range<usize>,
    /// Index of its first event time among all event times.
    time0: usize,
    /// Position of its first entry in the prefix arrays, `time0` plus the
    /// stratum's number.
    base: usize,
    /// Its number of distinct event times.
    times: usize,
    /// Positions of its rows that leave the risk set before the earliest event
    /// time, in ascending order of the index at which they leave.
    removal: Vec<u32>,
    /// Number of events at each of its event times.
    dying: Vec<u32>,
}

/// The rows of the fit, in the order of the sweep.
///
/// Event times of a stratum are numbered from the latest, `k = 0`, to the
/// earliest. A row is at risk at the event times with index in `[e, s)`, where
/// `e` counts the event times after its exit and `s` those after its entry.
/// Rows never at risk at an event time are left out of the layout.
pub(super) struct Layout {
    /// Columns of the design that are fitted.
    pub(super) p: usize,
    /// Clusters with a row of positive weight.
    pub(super) g: usize,
    pub(super) efron: bool,
    /// Covariates centred at their weighted mean, `p` per row.
    x: Vec<f64>,
    weight: Vec<f64>,
    /// Offsets less their weighted mean.
    offset: Vec<f64>,
    event: Vec<bool>,
    /// Dense cluster index of each row, `0..g`.
    cluster: Vec<u32>,
    /// Position in the prefix arrays of the row's exit index and entry index.
    pe: Vec<u32>,
    ps: Vec<u32>,
    /// Index among all event times of the first event time at or before the
    /// row's exit, which for an event row is its own time.
    te: Vec<u32>,
    strata: Vec<StratumLayout>,
    /// Number of event times over all strata.
    n_times: usize,
    /// For the sum of squared cluster risk totals: the change points of each
    /// cluster within a stratum, grouped, each `(index, position, adds)`.
    points: Vec<(u32, u32, bool)>,
    group_offset: Vec<usize>,
    /// Stratum of each group.
    group_stratum: Vec<u32>,
    /// Number of events of each cluster, unweighted.
    pub(super) events: Vec<u32>,
    /// Whether every case weight is 1.
    pub(super) unit_weights: bool,
}

/// What [`Layout::new`] reads.
pub(super) struct Rows<'a> {
    pub(super) x: &'a [f64],
    pub(super) p: usize,
    pub(super) start: &'a [f64],
    pub(super) stop: &'a [f64],
    pub(super) event: &'a [bool],
    pub(super) weight: &'a [f64],
    pub(super) stratum: &'a [u32],
    pub(super) n_strata: usize,
    /// Dense cluster index of each row, `u32::MAX` for a row left out.
    pub(super) cluster: &'a [u32],
    pub(super) g: usize,
    pub(super) offset: Option<&'a [f64]>,
    pub(super) efron: bool,
}

fn to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("the entry point checks that the row count fits in u32")
}

impl Layout {
    /// The layout of the rows of positive weight; `x` holds the fitted columns
    /// only, `p` per row of the data.
    pub(super) fn new(rows: &Rows<'_>) -> Self {
        let n = rows.stop.len();
        let p = rows.p;
        let used: Vec<usize> = (0..n).filter(|&i| rows.weight[i] > 0.0).collect();
        let total: f64 = used.iter().map(|&i| rows.weight[i]).sum();
        let mut mean = vec![0.0; p];
        let mut mean_offset = 0.0;
        for &i in &used {
            let c = rows.weight[i];
            for (m, &v) in mean.iter_mut().zip(&rows.x[i * p..(i + 1) * p]) {
                *m += c * v;
            }
            if let Some(o) = rows.offset {
                mean_offset += c * o[i];
            }
        }
        for m in &mut mean {
            *m /= total;
        }
        mean_offset /= total;

        let mut events = vec![0u32; rows.g];
        let mut unit_weights = true;
        for &i in &used {
            if rows.event[i] {
                events[rows.cluster[i] as usize] += 1;
            }
            if rows.weight[i] - 1.0 != 0.0 {
                unit_weights = false;
            }
        }

        let mut by_stratum: Vec<Vec<usize>> = vec![Vec::new(); rows.n_strata.max(1)];
        for &i in &used {
            by_stratum[rows.stratum[i] as usize].push(i);
        }
        let mut layout = Self {
            p,
            g: rows.g,
            efron: rows.efron,
            x: Vec::new(),
            weight: Vec::new(),
            offset: Vec::new(),
            event: Vec::new(),
            cluster: Vec::new(),
            pe: Vec::new(),
            ps: Vec::new(),
            te: Vec::new(),
            strata: Vec::new(),
            n_times: 0,
            points: Vec::new(),
            group_offset: vec![0],
            group_stratum: Vec::new(),
            events,
            unit_weights,
        };
        for (s, members) in by_stratum.iter().enumerate() {
            let mut times: Vec<f64> = members
                .iter()
                .filter(|&&i| rows.event[i])
                .map(|&i| rows.stop[i])
                .collect();
            times.sort_by(|a, b| b.total_cmp(a));
            times.dedup();
            let k = times.len();
            let first = layout.n_times;
            let base = first + s;
            // (exit index, entry index, row)
            let mut keyed: Vec<(usize, usize, usize)> = members
                .iter()
                .map(|&i| {
                    let e = times.partition_point(|&t| t > rows.stop[i]);
                    let st = times.partition_point(|&t| t > rows.start[i]);
                    (e, st, i)
                })
                .filter(|&(e, st, _)| e < st)
                .collect();
            keyed.sort_by(|a, b| {
                a.0.cmp(&b.0)
                    .then(rows.event[b.2].cmp(&rows.event[a.2]))
                    .then(a.2.cmp(&b.2))
            });
            let first_row = layout.weight.len();
            let mut dying = vec![0u32; k];
            for &(e, st, i) in &keyed {
                layout.x.extend(
                    rows.x[i * p..(i + 1) * p]
                        .iter()
                        .zip(&mean)
                        .map(|(v, m)| v - m),
                );
                layout.weight.push(rows.weight[i]);
                layout
                    .offset
                    .push(rows.offset.map_or(0.0, |o| o[i] - mean_offset));
                layout.event.push(rows.event[i]);
                layout.cluster.push(rows.cluster[i]);
                layout.pe.push(to_u32(base + e));
                layout.ps.push(to_u32(base + st));
                layout.te.push(to_u32(first + e.min(k.saturating_sub(1))));
                if rows.event[i] {
                    dying[e] += 1;
                }
            }
            let last = layout.weight.len();
            let mut removal: Vec<u32> = (first_row..last)
                .filter(|&pos| (layout.ps[pos] as usize - base) < k)
                .map(to_u32)
                .collect();
            removal.sort_by_key(|&pos| (layout.ps[pos as usize], pos));
            // Change points of each cluster in this stratum.
            let mut by_cluster: Vec<usize> = (first_row..last).collect();
            by_cluster.sort_by_key(|&pos| (layout.cluster[pos], pos));
            let mut start = 0;
            while start < by_cluster.len() {
                let c = layout.cluster[by_cluster[start]];
                let mut end = start;
                while end < by_cluster.len() && layout.cluster[by_cluster[end]] == c {
                    end += 1;
                }
                let mut pts: Vec<(u32, u32, bool)> = Vec::with_capacity(2 * (end - start));
                for &pos in &by_cluster[start..end] {
                    pts.push((to_u32(layout.pe[pos] as usize - base), to_u32(pos), true));
                    let st = layout.ps[pos] as usize - base;
                    if st < k {
                        pts.push((to_u32(st), to_u32(pos), false));
                    }
                }
                pts.sort_by_key(|&(index, pos, adds)| (index, adds, pos));
                layout.points.extend(pts);
                layout.group_offset.push(layout.points.len());
                layout.group_stratum.push(to_u32(s));
                start = end;
            }
            layout.strata.push(StratumLayout {
                rows: first_row..last,
                time0: first,
                base,
                times: k,
                removal,
                dying,
            });
            layout.n_times += k;
        }
        layout
    }

    fn n(&self) -> usize {
        self.weight.len()
    }

    fn n_prefix(&self) -> usize {
        self.n_times + self.strata.len()
    }
}

/// The penalty on the log frailties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Penalty {
    /// No frailty: the log frailties are held at zero.
    None,
    /// Gamma frailty with variance `theta = 1 / nu`.
    Gamma { nu: f64 },
}

/// One evaluation of the penalised partial likelihood.
pub(super) struct Eval {
    /// The partial log likelihood.
    pub(super) loglik: f64,
    /// The penalised log likelihood, the partial less the penalty.
    pub(super) ppl: f64,
    /// The gradient of the penalised log likelihood, `p` coefficients and
    /// then `g` log frailties.
    pub(super) grad: Vec<f64>,
    /// The coefficient block of the information, `p * p`, symmetric.
    a: Vec<f64>,
    /// The cross block, row `i` the derivatives of the score of cluster `i`.
    b: Vec<f64>,
    /// The diagonal of the frailty block, penalty included.
    hdiag: Vec<f64>,
    /// The diagonal of the quadratic part of the frailty block.
    fdiag: Vec<f64>,
    /// The second derivative of the penalty in each log frailty.
    pub(super) curvature: Vec<f64>,
    risk: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
}

/// Coefficients and log frailties.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct State {
    pub(super) beta: Vec<f64>,
    pub(super) w: Vec<f64>,
}

impl State {
    pub(super) fn zero(p: usize, g: usize) -> Self {
        Self {
            beta: vec![0.0; p],
            w: vec![0.0; g],
        }
    }

    /// Shift the log frailties so that the mean frailty is 1, which raises
    /// the gamma penalty and leaves the partial likelihood as it is.
    fn recenter(&mut self) {
        if self.w.is_empty() {
            return;
        }
        let mean = self.w.iter().map(|w| w.exp()).sum::<f64>() / self.w.len() as f64;
        let shift = mean.ln();
        for w in &mut self.w {
            *w -= shift;
        }
    }
}

/// The factorised preconditioner, the information with its frailty block
/// replaced by the diagonal.
pub(super) struct Preconditioner {
    /// `A - B' diag(1 / hdiag) B`, factorised by `ldl`.
    schur: Vec<f64>,
    rank: usize,
}

impl Layout {
    /// The partial likelihood, its score and the pieces of its information at
    /// `state`, or `None` when a risk score or a denominator is not finite.
    #[expect(
        clippy::too_many_lines,
        reason = "one sweep that forms every sum of the evaluation"
    )]
    pub(super) fn evaluate(&self, state: &State, penalty: Penalty) -> Option<Eval> {
        let (p, n, g) = (self.p, self.n(), self.g);
        let frail = !matches!(penalty, Penalty::None);
        let mut eta = Vec::with_capacity(n);
        let mut risk = Vec::with_capacity(n);
        for j in 0..n {
            let mut value: f64 = self.x[j * p..(j + 1) * p]
                .iter()
                .zip(&state.beta)
                .map(|(x, b)| x * b)
                .sum();
            value += self.offset[j];
            if frail {
                value += state.w[self.cluster[j] as usize];
            }
            let r = self.weight[j] * value.exp();
            if !r.is_finite() {
                return None;
            }
            eta.push(value);
            risk.push(r);
        }
        let t = self.n_times;
        let mut h = vec![0.0; t];
        let mut gq = vec![0.0; t];
        let mut k1 = vec![0.0; t];
        let mut k2 = vec![0.0; t];
        let mut k3 = vec![0.0; t];
        let mut m1 = vec![0.0; t * p];
        let mut m2 = vec![0.0; t * p];
        let mut loglik = 0.0;
        let mut score = vec![0.0; p];
        let mut a = vec![0.0; p * p];
        let mut s1 = vec![0.0; p];
        let mut s2 = vec![0.0; p * p];
        let mut e1 = vec![0.0; p];
        let mut e2 = vec![0.0; p * p];
        let mut d1 = vec![0.0; p];
        let mut quot = vec![0.0; p];
        for stratum in &self.strata {
            let mut s0 = 0.0;
            s1.fill(0.0);
            s2.fill(0.0);
            let mut cursor = stratum.rows.start;
            let mut leaving = 0usize;
            for k in 0..stratum.times {
                let block = cursor;
                while cursor < stratum.rows.end && self.pe[cursor] as usize - stratum.base == k {
                    s0 += risk[cursor];
                    moments::add(
                        risk[cursor],
                        &self.x[cursor * p..(cursor + 1) * p],
                        &mut s1,
                        &mut s2,
                    );
                    cursor += 1;
                }
                while leaving < stratum.removal.len() {
                    let pos = stratum.removal[leaving] as usize;
                    if self.ps[pos] as usize - stratum.base != k {
                        break;
                    }
                    s0 -= risk[pos];
                    moments::add(
                        -risk[pos],
                        &self.x[pos * p..(pos + 1) * p],
                        &mut s1,
                        &mut s2,
                    );
                    leaving += 1;
                }
                let d = stratum.dying[k] as usize;
                let mut e0 = 0.0;
                e1.fill(0.0);
                e2.fill(0.0);
                let mut wsum = 0.0;
                for pos in block..block + d {
                    let c = self.weight[pos];
                    wsum += c;
                    loglik += c * eta[pos];
                    for (sc, &x) in score.iter_mut().zip(&self.x[pos * p..(pos + 1) * p]) {
                        *sc += c * x;
                    }
                    e0 += risk[pos];
                    moments::add(risk[pos], &self.x[pos * p..(pos + 1) * p], &mut e1, &mut e2);
                }
                let (terms, u, spread) = if self.efron {
                    (d, wsum / d as f64, 1.0 / d as f64)
                } else {
                    (1, wsum, 0.0)
                };
                let tt = stratum.time0 + k;
                for term in 0..terms {
                    let f = term as f64 * spread;
                    let d0 = s0 - f * e0;
                    if !(d0.is_finite() && d0 > 0.0) {
                        return None;
                    }
                    for ((v, &r), &e) in d1.iter_mut().zip(&s1).zip(&e1) {
                        *v = r - f * e;
                    }
                    loglik -= u * d0.ln();
                    moments::quotients(&d1, d0, &mut quot);
                    for (sc, &q) in score.iter_mut().zip(&quot) {
                        *sc -= u * q;
                    }
                    moments::add_information(p, u, d0, f, &s2, &e2, &quot, &mut a);
                    let inv = u / d0;
                    let inv2 = inv / d0;
                    h[tt] += inv;
                    gq[tt] += f * inv;
                    k1[tt] += inv2;
                    k2[tt] += f * inv2;
                    k3[tt] += f * f * inv2;
                    for (col, &v) in d1.iter().enumerate() {
                        m1[tt * p + col] += inv2 * v;
                        m2[tt * p + col] += f * inv2 * v;
                    }
                }
            }
        }
        for r in 0..p {
            for c in 0..r {
                a[c * p + r] = a[r * p + c];
            }
        }
        if !(loglik.is_finite() && score.iter().all(|v| v.is_finite())) {
            return None;
        }
        let mut grad = score;
        grad.resize(p + g, 0.0);
        let mut b = vec![0.0; g * p];
        let mut hdiag = vec![0.0; g];
        let mut fdiag = vec![0.0; g];
        let mut curvature = vec![0.0; g];
        let mut ppl = loglik;
        if frail {
            let hp = self.prefix(&h, 1);
            let k1p = self.prefix(&k1, 1);
            let m1p = self.prefix(&m1, p);
            let mut dm = vec![0.0; p];
            for (j, &r) in risk.iter().enumerate() {
                let (pe, ps, te) = (
                    self.pe[j] as usize,
                    self.ps[j] as usize,
                    self.te[j] as usize,
                );
                let ev = self.event[j];
                let mut dh = hp[ps] - hp[pe];
                if ev {
                    dh -= gq[te];
                }
                for (col, v) in dm.iter_mut().enumerate() {
                    *v = m1p[ps * p + col] - m1p[pe * p + col];
                    if ev {
                        *v -= m2[te * p + col];
                    }
                }
                let i = self.cluster[j] as usize;
                let mut gw = -r * dh;
                if ev {
                    gw += self.weight[j];
                }
                grad[p + i] += gw;
                hdiag[i] += r * dh;
                let x = &self.x[j * p..(j + 1) * p];
                for (col, (&xv, &mv)) in x.iter().zip(&dm).enumerate() {
                    b[i * p + col] += r * (xv * dh - mv);
                }
            }
            self.squared_totals(&risk, &k1p, &k2, &k3, &mut fdiag);
            for i in 0..g {
                hdiag[i] -= fdiag[i];
            }
            match penalty {
                Penalty::Gamma { nu } => {
                    let mut pen = 0.0;
                    for (i, &w) in state.w.iter().enumerate() {
                        let z = w.exp();
                        pen += z - w;
                        grad[p + i] -= nu * (z - 1.0);
                        curvature[i] = nu * z;
                        hdiag[i] += nu * z;
                    }
                    ppl -= nu * pen;
                }
                Penalty::None => unreachable!("frail implies a penalty"),
            }
        }
        if !(ppl.is_finite() && grad.iter().all(|v| v.is_finite())) {
            return None;
        }
        Some(Eval {
            loglik,
            ppl,
            grad,
            a,
            b,
            hdiag,
            fdiag,
            curvature,
            risk,
            k1,
            k2,
            k3,
        })
    }

    /// Prefix sums over each stratum's event times of `values`, `width` per
    /// time: entry `base + k` is the sum over the first `k` times of the
    /// stratum, with `base = time0 + s`.
    fn prefix(&self, values: &[f64], width: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.n_prefix() * width];
        for stratum in &self.strata {
            let base = stratum.base;
            for k in 0..stratum.times {
                for c in 0..width {
                    out[(base + k + 1) * width + c] =
                        out[(base + k) * width + c] + values[(stratum.time0 + k) * width + c];
                }
            }
        }
        out
    }

    /// The diagonal of the quadratic part of the frailty block: for each
    /// cluster the sum over event times of `k1 R^2 - 2 k2 R E + k3 E^2`, with
    /// `R` the cluster's risk total and `E` its total over the events at the
    /// time, from the change points of the cluster's risk total.
    fn squared_totals(&self, risk: &[f64], k1p: &[f64], k2: &[f64], k3: &[f64], out: &mut [f64]) {
        for (group, &s) in self.group_stratum.iter().enumerate() {
            let stratum = &self.strata[s as usize];
            let base = stratum.base;
            let points = &self.points[self.group_offset[group]..self.group_offset[group + 1]];
            let cluster = self.cluster[points[0].1 as usize] as usize;
            let mut total = 0.0;
            let mut sum = 0.0;
            let mut i = 0;
            while i < points.len() {
                let index = points[i].0 as usize;
                let mut dying = 0.0;
                while i < points.len() && points[i].0 as usize == index {
                    let (_, pos, adds) = points[i];
                    let r = risk[pos as usize];
                    if adds {
                        total += r;
                        if self.event[pos as usize] {
                            dying += r;
                        }
                    } else {
                        total -= r;
                    }
                    i += 1;
                }
                let next = if i < points.len() {
                    points[i].0 as usize
                } else {
                    stratum.times
                };
                sum += total * total * (k1p[base + next] - k1p[base + index]);
                if self.efron && dying != 0.0 {
                    let tt = stratum.time0 + index;
                    sum += -2.0 * k2[tt] * total * dying + k3[tt] * dying * dying;
                }
            }
            out[cluster] += sum;
        }
    }

    /// `out = H v` for the whole information `H` at `ev`, `v` and `out` of
    /// length `p + g`.
    pub(super) fn multiply(&self, ev: &Eval, v: &[f64], out: &mut [f64]) {
        let (p, g) = (self.p, self.g);
        let (vb, vw) = v.split_at(p);
        let (ob, ow) = out.split_at_mut(p);
        for (r, o) in ob.iter_mut().enumerate() {
            let mut s = 0.0;
            for (&a, &vc) in ev.a[r * p..(r + 1) * p].iter().zip(vb) {
                s += a * vc;
            }
            *o = s;
        }
        for i in 0..g {
            let row = &ev.b[i * p..(i + 1) * p];
            let mut s = 0.0;
            for (c, &bv) in row.iter().enumerate() {
                ob[c] += bv * vw[i];
                s += bv * vb[c];
            }
            ow[i] = s + (ev.hdiag[i] + ev.fdiag[i]) * vw[i];
        }
        // The quadratic part F vw by a sweep over the event times.
        let n = self.n();
        let mut q = vec![0.0; self.n_prefix()];
        let mut qe = vec![0.0; self.n_times];
        for j in 0..n {
            let y = ev.risk[j] * vw[self.cluster[j] as usize];
            q[self.pe[j] as usize] += y;
            q[self.ps[j] as usize] -= y;
            if self.event[j] {
                qe[self.te[j] as usize] += y;
            }
        }
        let mut phi = vec![0.0; self.n_prefix()];
        let mut psi = vec![0.0; self.n_times];
        for stratum in &self.strata {
            let base = stratum.base;
            let mut running = 0.0;
            for k in 0..stratum.times {
                running += q[base + k];
                let tt = stratum.time0 + k;
                let f = running * ev.k1[tt] - qe[tt] * ev.k2[tt];
                psi[tt] = running * ev.k2[tt] - qe[tt] * ev.k3[tt];
                phi[base + k + 1] = phi[base + k] + f;
            }
        }
        for j in 0..n {
            let mut v = phi[self.ps[j] as usize] - phi[self.pe[j] as usize];
            if self.event[j] {
                v -= psi[self.te[j] as usize];
            }
            ow[self.cluster[j] as usize] -= ev.risk[j] * v;
        }
    }

    /// The preconditioner at `ev`, or `None` when the diagonal of the frailty
    /// block is not positive or the Schur complement is not positive definite.
    pub(super) fn preconditioner(&self, ev: &Eval) -> Option<Preconditioner> {
        let (p, g) = (self.p, self.g);
        if ev.hdiag.iter().any(|&d| !(d.is_finite() && d > 0.0)) {
            return None;
        }
        let mut schur = ev.a.clone();
        for i in 0..g {
            let row = &ev.b[i * p..(i + 1) * p];
            let inv = 1.0 / ev.hdiag[i];
            for r in 0..p {
                let br = row[r] * inv;
                for c in 0..=r {
                    schur[r * p + c] -= br * row[c];
                }
            }
        }
        for r in 0..p {
            for c in 0..r {
                schur[c * p + r] = schur[r * p + c];
            }
        }
        let pivots = ldl::factorise(&mut schur, p, TOLER_CHOL);
        if !pivots.clean || pivots.rank < p {
            return None;
        }
        Some(Preconditioner {
            schur,
            rank: pivots.rank,
        })
    }

    /// `z = M^-1 r` for the preconditioner `M`.
    fn precondition(&self, ev: &Eval, m: &Preconditioner, r: &[f64], z: &mut [f64]) {
        let (p, g) = (self.p, self.g);
        debug_assert_eq!(m.rank, p);
        let (rb, rw) = r.split_at(p);
        let mut t = rb.to_vec();
        for (i, &rwi) in rw.iter().enumerate() {
            let yw = rwi / ev.hdiag[i];
            for (c, tv) in t.iter_mut().enumerate() {
                *tv -= ev.b[i * p + c] * yw;
            }
        }
        ldl::solve(&m.schur, p, &mut t);
        let (zb, zw) = z.split_at_mut(p);
        zb.copy_from_slice(&t);
        for i in 0..g {
            let row = &ev.b[i * p..(i + 1) * p];
            let mut s = rw[i];
            for (c, &bv) in row.iter().enumerate() {
                s -= bv * t[c];
            }
            zw[i] = s / ev.hdiag[i];
        }
    }

    /// Solve `H x = rhs` by conjugate gradients preconditioned by `m`.
    /// Returns the solution and whether the residual fell below the
    /// tolerance.
    pub(super) fn solve(&self, ev: &Eval, m: &Preconditioner, rhs: &[f64]) -> (Vec<f64>, bool) {
        let dim = rhs.len();
        let mut x = vec![0.0; dim];
        let mut r = rhs.to_vec();
        let mut z = vec![0.0; dim];
        self.precondition(ev, m, &r, &mut z);
        let mut dir = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let rz0 = rz;
        if rz0 == 0.0 {
            return (x, true);
        }
        let mut q = vec![0.0; dim];
        for _ in 0..CG_MAX_ITERATIONS.min(2 * dim + 10) {
            self.multiply(ev, &dir, &mut q);
            let dq: f64 = dir.iter().zip(&q).map(|(a, b)| a * b).sum();
            if !(dq.is_finite() && dq > 0.0) {
                return (x, false);
            }
            let alpha = rz / dq;
            for k in 0..dim {
                x[k] += alpha * dir[k];
                r[k] -= alpha * q[k];
            }
            self.precondition(ev, m, &r, &mut z);
            let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            if rz_new <= CG_TOLERANCE * CG_TOLERANCE * rz0 {
                return (x, true);
            }
            let beta = rz_new / rz;
            rz = rz_new;
            for k in 0..dim {
                dir[k] = z[k] + beta * dir[k];
            }
        }
        (x, false)
    }
}

impl Layout {
    /// The Breslow cumulative hazard of each cluster with its frailty left
    /// out, and the sum over the events of `x'beta + o + ln dH_0`, where `dH_0`
    /// is the Breslow increment with the frailties in the risk scores, less
    /// `sum_k (d_k ln d_k - d_k)` over the event times, which puts the
    /// likelihood on the scale of the partial likelihood.
    pub(super) fn breslow_cluster_hazards(&self, state: &State) -> Option<(Vec<f64>, f64)> {
        let (p, n) = (self.p, self.n());
        let mut eta0 = Vec::with_capacity(n);
        let mut q = vec![0.0; self.n_prefix()];
        let mut wk = vec![0.0; self.n_times];
        for j in 0..n {
            let mut value: f64 = self.x[j * p..(j + 1) * p]
                .iter()
                .zip(&state.beta)
                .map(|(x, b)| x * b)
                .sum();
            value += self.offset[j];
            let r = self.weight[j] * (value + state.w[self.cluster[j] as usize]).exp();
            if !r.is_finite() {
                return None;
            }
            q[self.pe[j] as usize] += r;
            q[self.ps[j] as usize] -= r;
            if self.event[j] {
                wk[self.te[j] as usize] += self.weight[j];
            }
            eta0.push(value);
        }
        let mut dh = vec![0.0; self.n_times];
        let mut hp = vec![0.0; self.n_prefix()];
        for stratum in &self.strata {
            let mut s0 = 0.0;
            for k in 0..stratum.times {
                s0 += q[stratum.base + k];
                let tt = stratum.time0 + k;
                dh[tt] = wk[tt] / s0;
                if !(dh[tt].is_finite() && dh[tt] > 0.0) {
                    return None;
                }
                hp[stratum.base + k + 1] = hp[stratum.base + k] + dh[tt];
            }
        }
        let scale: f64 = wk.iter().map(|&d| d * d.ln() - d).sum();
        let mut lambda = vec![0.0; self.g];
        let mut events = 0.0;
        for j in 0..n {
            let base = self.weight[j] * eta0[j].exp();
            lambda[self.cluster[j] as usize] +=
                base * (hp[self.ps[j] as usize] - hp[self.pe[j] as usize]);
            if self.event[j] {
                events += self.weight[j] * (eta0[j] + dh[self.te[j] as usize].ln());
            }
        }
        Some((lambda, events - scale))
    }
}

/// The result of the Newton iteration at one theta.
pub(super) struct Inner {
    pub(super) state: State,
    pub(super) eval: Eval,
    pub(super) iterations: u32,
    pub(super) converged: bool,
}

/// Why the Newton iteration has no result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Failure {
    /// A risk score or a denominator is not finite at the start.
    NotFinite,
    /// The information is not positive definite.
    Information,
}

impl Layout {
    /// The Newton direction at `ev`: `H^-1 grad` and whether the solve met its
    /// tolerance. Without frailty the coefficient block is solved directly.
    fn direction(&self, ev: &Eval, penalty: Penalty) -> Result<(Vec<f64>, bool), Failure> {
        let p = self.p;
        if matches!(penalty, Penalty::None) {
            let mut a = ev.a.clone();
            let pivots = ldl::factorise(&mut a, p, TOLER_CHOL);
            if !pivots.clean || pivots.rank < p {
                return Err(Failure::Information);
            }
            let mut d = ev.grad[..p].to_vec();
            ldl::solve(&a, p, &mut d);
            d.resize(p + self.g, 0.0);
            return Ok((d, true));
        }
        let m = self.preconditioner(ev).ok_or(Failure::Information)?;
        Ok(self.solve(ev, &m, &ev.grad))
    }

    /// Maximise the penalised partial likelihood from `start` by Newton's
    /// method with step halving. Converged when the decrement `grad' H^-1
    /// grad` at the last accepted point is at most `tolerance`; the step that
    /// it predicts is then taken and the iteration stops.
    pub(super) fn maximise(
        &self,
        start: &State,
        penalty: Penalty,
        tolerance: f64,
        max_iterations: u32,
    ) -> Result<Inner, Failure> {
        let frail = !matches!(penalty, Penalty::None);
        let mut state = start.clone();
        if frail {
            state.recenter();
        } else {
            state.w.fill(0.0);
        }
        let mut ev = self.evaluate(&state, penalty).ok_or(Failure::NotFinite)?;
        let mut iterations = 0;
        let mut converged = false;
        while iterations < max_iterations {
            let (step, _) = self.direction(&ev, penalty)?;
            let decrement: f64 = step.iter().zip(&ev.grad).map(|(a, b)| a * b).sum();
            if !decrement.is_finite() {
                return Err(Failure::Information);
            }
            let mut scale = 1.0;
            let mut accepted = None;
            for _ in 0..=MAX_HALVINGS {
                iterations += 1;
                let mut trial = state.clone();
                for (b, d) in trial.beta.iter_mut().zip(&step) {
                    *b += scale * d;
                }
                if frail {
                    for (w, d) in trial.w.iter_mut().zip(&step[self.p..]) {
                        *w += scale * d;
                    }
                    trial.recenter();
                }
                if let Some(next) = self.evaluate(&trial, penalty) {
                    let noise = 1e-12 * (1.0 + ev.ppl.abs());
                    if next.ppl >= ev.ppl || (decrement < 1e-6 && next.ppl >= ev.ppl - noise) {
                        accepted = Some((trial, next));
                        break;
                    }
                }
                scale *= 0.5;
                if iterations >= max_iterations {
                    break;
                }
            }
            let Some((trial, next)) = accepted else {
                break;
            };
            state = trial;
            ev = next;
            if decrement <= tolerance {
                converged = true;
                break;
            }
        }
        Ok(Inner {
            state,
            eval: ev,
            iterations,
            converged,
        })
    }
}

/// The covariance matrices of the coefficients at a solution.
pub(super) struct Covariances {
    /// The coefficient block of the inverse of the whole information.
    pub(super) exact: Option<Vec<f64>>,
    /// `exact` less `Y' P Y`, with `Y` the frailty rows of the inverse and `P`
    /// the penalty's curvature, `var2` of `coxph`.
    pub(super) exact_sandwich: Option<Vec<f64>>,
    /// The same two with the frailty block taken as its diagonal.
    pub(super) diagonal: Vec<f64>,
    pub(super) diagonal_sandwich: Vec<f64>,
    /// The diagonal of the frailty block of the inverse of the information
    /// with the diagonal frailty block, `fvar` of `coxph`.
    pub(super) frailty_diagonal: Vec<f64>,
}

impl Layout {
    /// The covariance matrices at `ev`, or `None` when the information is not
    /// positive definite. Without frailty both are the inverse of the
    /// coefficient block.
    pub(super) fn covariances(&self, ev: &Eval, penalty: Penalty) -> Option<Covariances> {
        let (p, g) = (self.p, self.g);
        if matches!(penalty, Penalty::None) {
            let mut a = ev.a.clone();
            let pivots = ldl::factorise(&mut a, p, TOLER_CHOL);
            if !pivots.clean || pivots.rank < p {
                return None;
            }
            let v = ldl::inverse(&a, p);
            return Some(Covariances {
                exact: Some(v.clone()),
                exact_sandwich: Some(v.clone()),
                diagonal: v.clone(),
                diagonal_sandwich: v,
                frailty_diagonal: vec![0.0; g],
            });
        }
        let m = self.preconditioner(ev)?;
        let vs = ldl::inverse(&m.schur, p);
        // Y_s = -diag(1 / hdiag) B V_s, g by p.
        let mut ys = vec![0.0; g * p];
        let mut frailty_diagonal = vec![0.0; g];
        for i in 0..g {
            let row = &ev.b[i * p..(i + 1) * p];
            let inv = 1.0 / ev.hdiag[i];
            let mut quad = 0.0;
            for c in 0..p {
                let mut s = 0.0;
                for (k, &bv) in row.iter().enumerate() {
                    s += bv * vs[k * p + c];
                }
                ys[i * p + c] = -inv * s;
                quad += s * row[c];
            }
            frailty_diagonal[i] = inv + quad * inv * inv;
        }
        let diagonal_sandwich = sandwich(&vs, &ys, &ev.curvature, p, g);
        let mut exact = vec![0.0; p * p];
        let mut y = vec![0.0; g * p];
        let mut ok = true;
        for c in 0..p {
            let mut rhs = vec![0.0; p + g];
            rhs[c] = 1.0;
            let (x, converged) = self.solve(ev, &m, &rhs);
            ok &= converged;
            for r in 0..p {
                exact[r * p + c] = x[r];
            }
            for i in 0..g {
                y[i * p + c] = x[p + i];
            }
        }
        let (exact, exact_sandwich) = if ok {
            symmetrise(&mut exact, p);
            let s = sandwich(&exact, &y, &ev.curvature, p, g);
            (Some(exact), Some(s))
        } else {
            (None, None)
        };
        Some(Covariances {
            exact,
            exact_sandwich,
            diagonal: vs,
            diagonal_sandwich,
            frailty_diagonal,
        })
    }
}

fn symmetrise(m: &mut [f64], p: usize) {
    for r in 0..p {
        for c in 0..r {
            let v = f64::midpoint(m[r * p + c], m[c * p + r]);
            m[r * p + c] = v;
            m[c * p + r] = v;
        }
    }
}

/// `V - Y' diag(curvature) Y`, made exactly symmetric.
fn sandwich(v: &[f64], y: &[f64], curvature: &[f64], p: usize, g: usize) -> Vec<f64> {
    let mut out = v.to_vec();
    for i in 0..g {
        let row = &y[i * p..(i + 1) * p];
        let c = curvature[i];
        for r in 0..p {
            for k in 0..p {
                out[r * p + k] -= row[r] * c * row[k];
            }
        }
    }
    symmetrise(&mut out, p);
    out
}
