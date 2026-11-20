//! The infinitesimal jackknife of the Aalen-Johansen estimate, shared by the
//! multi-state and the competing-risks entry points.
//!
//! # Algorithm
//!
//! The estimate is a function of the case weights `w_i` of the rows. The
//! influence of row `i` on the occupation probabilities `p(t)` is the
//! derivative `U_i(t) = w_i dp(t) / dw_i`, a vector over the states. The
//! influence of a cluster `g`, which is the sum of `U_i` over its rows, is
//! `U_g(t)`, and the variance of `p_j(t)` is
//!
//! ```text
//! var p_j(t) = sum over g of U_gj(t)^2.
//! ```
//!
//! This is the robust variance of the product integral \[1\]\[2\]. Without a cluster
//! or an id every row is its own cluster.
//!
//! With `p(t) = p(t-) (I + H(t))`, where `H` has `H_jk = n_jk / Y_j` for
//! `j != k` and `H_jj = -sum_k H_jk`, the product rule gives
//!
//! ```text
//! U_g(t) = U_g(t-) (I + H(t)) + Q_g(t),
//! ```
//!
//! and `Q_g` has two parts. A row of `g` that moves from `j` to `k` at `t`
//! adds `+- w p_j(t-) / Y_j` to the entries `k` and `j`. Every row of `g`
//! at risk in `j` at `t` adds `-+ w_gj p_j(t-) n_jk / Y_j^2` to the same two
//! entries, where `w_gj` is the weight of `g` at risk in `j`, for each
//! transition `j -> k` that occurs at `t`. The cumulative hazard
//! `Lambda_jk = sum n_jk / Y_j` has the influence
//!
//! ```text
//! C_g,jk(t) = C_g,jk(t-) + sum over rows of g with a j -> k move at t of w / Y_j
//!                         - w_gj n_jk / Y_j^2 (when the transition occurs at t),
//! ```
//!
//! and the restricted time in state `j` from the origin `t0`, the integral of
//! `p_j`, has the influence `A_g(t) = integral of U_g(s) ds`, a step function
//! integrated exactly. A horizon between two event times uses the influence at
//! the earlier one. The standard errors are the square roots of the sums of
//! squares of `U`, `C` and `A` over the clusters. The initial distribution, when
//! it is estimated from the data, has the influence
//! `U_g(t0)_j = sum over its opening rows of w (1[state = j] - p0_j) / W`,
//! and none when it is given \[3\].
//!
//! # Memory and order
//!
//! A full influence array is clusters by times by states, and at register scale
//! it does not fit. The recursion above is the same for every cluster and
//! involves no other cluster, so the clusters are cut into blocks of
//! [`BLOCK`] and each block runs the whole recursion on its own, holding only
//! its `U`, `A` and `C`, which are `O(block * (3 states + transitions))`. A block adds
//! the squares into a per-time partial of `O(times * (2 states +
//! transitions))`, and the partials are added in block order. The influence
//! values of chosen times are written out as they pass, and nothing else of the
//! influence is kept. Peak memory is the data, the global tables of size
//! `times * (states + transitions)` and `O(waves * block)` for the blocks in
//! flight, where a wave is as many blocks as there are threads. The
//! time cost is `O(clusters * times * states)`, as for `survfit`, since every
//! cluster's influence moves at every event time.
//!
//! The sum of squares within a block runs over its clusters in code order, and
//! the blocks are added in order, whatever the number of threads or waves, so
//! the bits do not depend on the pool.
//!
//! # References
//!
//! 1. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000.
//! 2. Therneau TM. Methods of the survfit function in the survival package,
//!    vignette `methods`, section on the infinitesimal jackknife for the
//!    Aalen-Johansen estimate.
//! 3. Andersen PK, Borgan O, Gill RD, Keiding N. Statistical models based on
//!    counting processes. New York: Springer; 1993.

#![expect(
    clippy::cast_possible_truncation,
    reason = "row, cluster and time counts are checked to fit in u32 by the entry points"
)]

use rayon::prelude::*;

use crate::order::par_sort_by_key;

/// The clusters in one block. A block is the unit of parallel work and of the
/// order of the sums, so changing it changes the last bits of a variance.
pub(crate) const BLOCK: usize = 256;

const NONE: u32 = u32::MAX;

/// The rows of one curve, one entry per row.
///
/// `to` at or above `states` is no transition, a censored row. `cluster` is a
/// dense code below `n_clusters`.
pub(crate) struct Process<'a> {
    pub start: &'a [f64],
    pub stop: &'a [f64],
    pub from: &'a [u32],
    pub to: &'a [u32],
    pub weight: &'a [f64],
    pub cluster: &'a [u32],
}

/// What the engine is asked for.
pub(crate) struct Request<'a> {
    pub states: usize,
    pub n_clusters: usize,
    /// The reporting times, strictly increasing and not before `origin`.
    pub times: &'a [f64],
    pub p0: &'a [f64],
    /// The influence of the initial distribution, cluster by state.
    pub u0: Option<&'a [f64]>,
    pub origin: f64,
    /// Ascending.
    pub horizons: &'a [f64],
    /// For each influence time, the index of the last reporting time not after
    /// it, or `None` when it is before the first. Ascending.
    pub snapshots: &'a [Option<usize>],
}

/// The result, all row-major in the order named.
pub(crate) struct Output {
    /// `(from, to)` of each cumulative hazard, ordered by `to` and then `from`.
    pub transitions: Vec<(u32, u32)>,
    /// Time by transition.
    pub cumhaz: Vec<f64>,
    /// Time by state.
    pub std_p: Vec<f64>,
    /// Time by transition.
    pub std_c: Vec<f64>,
    /// Time by state.
    pub std_a: Vec<f64>,
    /// State.
    pub std_initial: Vec<f64>,
    /// Horizon by state.
    pub std_h: Vec<f64>,
    /// Cluster by snapshot by state.
    pub influence: Vec<f64>,
    /// The probabilities after each time, time by state. The estimate the
    /// variance belongs to, for checking against the other entry points.
    pub p: Vec<f64>,
}

/// One transition at one time: its index, endpoints and the hazard over the
/// weight at risk.
struct Tr {
    jk: u32,
    j: u32,
    k: u32,
    htemp: f64,
}

/// One entry of `H`, the diagonal ones included.
struct Hentry {
    j: u32,
    k: u32,
    value: f64,
}

/// What every cluster shares: weights at risk, hazards and the probabilities.
struct Global {
    risk: Vec<f64>,
    phat_before: Vec<f64>,
    tr_off: Vec<usize>,
    tr: Vec<Tr>,
    h_off: Vec<usize>,
    h: Vec<Hentry>,
}

/// Runs the recursion for one curve.
pub(crate) fn run(process: &Process<'_>, req: &Request<'_>) -> Output {
    let s = req.states;
    let n = process.stop.len();
    let t_count = req.times.len();

    // Transitions that occur, ordered by destination and then origin.
    let mut seen = vec![false; s * s];
    for row in 0..n {
        let (from, to) = (process.from[row] as usize, process.to[row] as usize);
        if to < s {
            seen[to * s + from] = true;
        }
    }
    let mut transitions = Vec::new();
    let mut jk_of = vec![NONE; s * s];
    for to in 0..s {
        for from in 0..s {
            if seen[to * s + from] {
                jk_of[from * s + to] = transitions.len() as u32;
                transitions.push((from as u32, to as u32));
            }
        }
    }
    let h_count = transitions.len();

    // The reporting position of each move, and its transition.
    let mut ev_pos = vec![NONE; n];
    let mut ev_jk = vec![NONE; n];
    let mut ntrans = vec![0.0f64; t_count * h_count];
    for row in 0..n {
        let (from, to) = (process.from[row] as usize, process.to[row] as usize);
        if to >= s {
            continue;
        }
        if let Ok(position) = req
            .times
            .binary_search_by(|time| time.total_cmp(&process.stop[row]))
        {
            let jk = jk_of[from * s + to];
            ev_pos[row] = position as u32;
            ev_jk[row] = jk;
            ntrans[position * h_count + jk as usize] += process.weight[row];
        }
    }

    // Weight at risk in each state, accumulated backwards in time.
    let mut by_start: Vec<u32> = (0..n as u32).collect();
    par_sort_by_key(&mut by_start, |row| process.start[row as usize]);
    let mut by_stop: Vec<u32> = (0..n as u32).collect();
    par_sort_by_key(&mut by_stop, |row| process.stop[row as usize]);
    let mut risk = vec![0.0f64; t_count * s];
    let mut running = vec![0.0f64; s];
    let (mut left_start, mut left_stop) = (n, n);
    for i in (0..t_count).rev() {
        let time = req.times[i];
        while left_start > 0 && process.start[by_start[left_start - 1] as usize] >= time {
            let row = by_start[left_start - 1] as usize;
            running[process.from[row] as usize] -= process.weight[row];
            left_start -= 1;
        }
        while left_stop > 0 && process.stop[by_stop[left_stop - 1] as usize] >= time {
            let row = by_stop[left_stop - 1] as usize;
            running[process.from[row] as usize] += process.weight[row];
            left_stop -= 1;
        }
        risk[i * s..(i + 1) * s].copy_from_slice(&running);
    }

    // The estimate, its hazards and the sparse `H` at each time.
    let mut phat = req.p0.to_vec();
    let mut chaz = vec![0.0f64; h_count];
    let mut p = Vec::with_capacity(t_count * s);
    let mut cumhaz = Vec::with_capacity(t_count * h_count);
    let mut phat_before = Vec::with_capacity(t_count * s);
    let (mut tr_off, mut tr) = (vec![0usize], Vec::new());
    let (mut h_off, mut h) = (vec![0usize], Vec::new());
    let mut leaving = vec![0.0f64; s];
    for i in 0..t_count {
        phat_before.extend_from_slice(&phat);
        let before = phat.clone();
        leaving.fill(0.0);
        for (jk, &(from, to)) in transitions.iter().enumerate() {
            let count = ntrans[i * h_count + jk];
            if count > 0.0 {
                let (j, k) = (from as usize, to as usize);
                let haz = count / risk[i * s + j];
                tr.push(Tr {
                    jk: jk as u32,
                    j: from,
                    k: to,
                    htemp: haz / risk[i * s + j],
                });
                chaz[jk] += haz;
                if j != k {
                    phat[j] -= before[j] * haz;
                    phat[k] += before[j] * haz;
                    leaving[j] += haz;
                    h.push(Hentry {
                        j: from,
                        k: to,
                        value: haz,
                    });
                }
            }
        }
        for (j, &total) in leaving.iter().enumerate() {
            if total > 0.0 {
                h.push(Hentry {
                    j: j as u32,
                    k: j as u32,
                    value: -total,
                });
            }
        }
        tr_off.push(tr.len());
        h_off.push(h.len());
        p.extend_from_slice(&phat);
        cumhaz.extend_from_slice(&chaz);
    }
    let global = Global {
        risk,
        phat_before,
        tr_off,
        tr,
        h_off,
        h,
    };

    // Clusters in code order, each with its rows in row order.
    let g_count = req.n_clusters;
    let mut cluster_offset = vec![0usize; g_count + 1];
    for &g in process.cluster {
        cluster_offset[g as usize + 1] += 1;
    }
    for g in 0..g_count {
        cluster_offset[g + 1] += cluster_offset[g];
    }
    let mut fill = cluster_offset.clone();
    let mut cluster_rows = vec![0u32; n];
    for (row, &g) in process.cluster.iter().enumerate() {
        cluster_rows[fill[g as usize]] = row as u32;
        fill[g as usize] += 1;
    }

    let n_snap = req.snapshots.len();
    let per_time = 2 * s + h_count;
    let horizon_base = t_count * per_time;
    let init_base = horizon_base + req.horizons.len() * s;
    let total_len = init_base + s;

    let n_blocks = g_count.div_ceil(BLOCK);
    let mut influence = vec![0.0f64; g_count * n_snap * s];
    let per_block_influence = BLOCK * n_snap * s;
    let slices: Vec<&mut [f64]> = if per_block_influence > 0 {
        influence.chunks_mut(per_block_influence).collect()
    } else {
        (0..n_blocks).map(|_| <&mut [f64]>::default()).collect()
    };
    let mut slices = slices;
    let wave = (rayon::current_num_threads() * 2).max(1);

    let shared = Shared {
        process,
        req,
        global: &global,
        h_count,
        ev_pos: &ev_pos,
        ev_jk: &ev_jk,
        cluster_offset: &cluster_offset,
        cluster_rows: &cluster_rows,
    };
    let mut totals = vec![0.0f64; total_len];
    for (wave_index, group) in slices.chunks_mut(wave).enumerate() {
        let partials: Vec<Vec<f64>> = group
            .par_iter_mut()
            .enumerate()
            .map(|(offset, slice)| shared.block((wave_index * wave + offset) * BLOCK, slice))
            .collect();
        for partial in &partials {
            for (total, value) in totals.iter_mut().zip(partial) {
                *total += value;
            }
        }
    }

    let root = |values: &[f64]| values.iter().map(|v| v.sqrt()).collect::<Vec<_>>();
    let mut std_p = Vec::with_capacity(t_count * s);
    let mut std_c = Vec::with_capacity(t_count * h_count);
    let mut std_a = Vec::with_capacity(t_count * s);
    for i in 0..t_count {
        let row = &totals[i * per_time..(i + 1) * per_time];
        std_p.extend(root(&row[..s]));
        std_c.extend(root(&row[s..s + h_count]));
        std_a.extend(root(&row[s + h_count..]));
    }
    Output {
        transitions,
        cumhaz,
        std_p,
        std_c,
        std_a,
        std_initial: root(&totals[init_base..]),
        std_h: root(&totals[horizon_base..init_base]),
        influence,
        p,
    }
}

/// Everything a block reads.
struct Shared<'a> {
    process: &'a Process<'a>,
    req: &'a Request<'a>,
    global: &'a Global,
    h_count: usize,
    ev_pos: &'a [u32],
    ev_jk: &'a [u32],
    cluster_offset: &'a [usize],
    cluster_rows: &'a [u32],
}

impl Shared<'_> {
    /// The whole recursion for the clusters from `g0` on, one block. Returns
    /// the sums of squares: per time `[U (states), C (transitions), A
    /// (states)]`, then per horizon the states, then the initial `U`. The
    /// influence values of the snapshots go to `out`, cluster by snapshot by
    /// state.
    fn block(&self, g0: usize, out: &mut [f64]) -> Vec<f64> {
        let s = self.req.states;
        let h_count = self.h_count;
        let process = self.process;
        let req = self.req;
        let global = self.global;
        let g1 = (g0 + BLOCK).min(req.n_clusters);
        let b = g1 - g0;
        let t_count = req.times.len();
        let per_time = 2 * s + h_count;
        let horizon_base = t_count * per_time;
        let init_base = horizon_base + req.horizons.len() * s;
        let mut part = vec![0.0f64; init_base + s];

        let rows = &self.cluster_rows[self.cluster_offset[g0]..self.cluster_offset[g1]];
        let local = |row: u32| process.cluster[row as usize] as usize - g0;
        let mut by_start = rows.to_vec();
        par_sort_by_key(&mut by_start, |row| process.start[row as usize]);
        let mut by_stop = rows.to_vec();
        par_sort_by_key(&mut by_stop, |row| process.stop[row as usize]);
        let mut by_event: Vec<u32> = rows
            .iter()
            .copied()
            .filter(|&row| self.ev_pos[row as usize] != NONE)
            .collect();
        by_event.sort_by_key(|&row| (self.ev_pos[row as usize], row));

        let mut u = match req.u0 {
            Some(u0) => u0[g0 * s..g1 * s].to_vec(),
            None => vec![0.0f64; b * s],
        };
        let mut ua = vec![0.0f64; b * s];
        let mut c = vec![0.0f64; b * h_count];
        let mut wg = vec![0.0f64; b * s];
        let mut count = vec![0u32; b * s];
        let mut old = vec![0.0f64; s];

        let mut cur_p = vec![0.0f64; s];
        for row in u.chunks_exact(s) {
            for (sum, &value) in cur_p.iter_mut().zip(row) {
                *sum += value * value;
            }
        }
        part[init_base..].copy_from_slice(&cur_p);
        let mut cur_c = vec![0.0f64; h_count];

        let n_snap = req.snapshots.len();
        let write = |out: &mut [f64], k: usize, u: &[f64]| {
            for gl in 0..b {
                let at = (gl * n_snap + k) * s;
                out[at..at + s].copy_from_slice(&u[gl * s..(gl + 1) * s]);
            }
        };
        let mut snap = 0usize;
        while snap < n_snap && req.snapshots[snap].is_none() {
            write(out, snap, &u);
            snap += 1;
        }

        let horizon_sums = |part: &mut [f64], horizon: usize, width: f64, ua: &[f64], u: &[f64]| {
            let mut sums = vec![0.0f64; s];
            for gl in 0..b {
                for j in 0..s {
                    let value = ua[gl * s + j] + width * u[gl * s + j];
                    sums[j] += value * value;
                }
            }
            part[horizon_base + horizon * s..horizon_base + (horizon + 1) * s]
                .copy_from_slice(&sums);
        };
        let mut next_horizon = req.horizons.partition_point(|&h| h <= req.origin);

        let mut t_prev = req.origin;
        let (mut at_start, mut at_stop, mut at_event) = (0usize, 0usize, 0usize);
        for i in 0..t_count {
            let time = req.times[i];
            while next_horizon < req.horizons.len() && req.horizons[next_horizon] < time {
                horizon_sums(
                    &mut part,
                    next_horizon,
                    req.horizons[next_horizon] - t_prev,
                    &ua,
                    &u,
                );
                next_horizon += 1;
            }

            // The integral of U up to this time, from U just before it.
            let delta = time - t_prev;
            let mut sum_a = vec![0.0f64; s];
            for gl in 0..b {
                for j in 0..s {
                    ua[gl * s + j] += delta * u[gl * s + j];
                    sum_a[j] += ua[gl * s + j] * ua[gl * s + j];
                }
            }

            // Weight at risk in each state, per cluster, strictly after the
            // start and up to the stop.
            while at_start < by_start.len() && process.start[by_start[at_start] as usize] < time {
                let row = by_start[at_start];
                let at = local(row) * s + process.from[row as usize] as usize;
                wg[at] += process.weight[row as usize];
                count[at] += 1;
                at_start += 1;
            }
            while at_stop < by_stop.len() && process.stop[by_stop[at_stop] as usize] < time {
                let row = by_stop[at_stop];
                let at = local(row) * s + process.from[row as usize] as usize;
                count[at] -= 1;
                wg[at] = if count[at] == 0 {
                    0.0
                } else {
                    wg[at] - process.weight[row as usize]
                };
                at_stop += 1;
            }

            let trs = &global.tr[global.tr_off[i]..global.tr_off[i + 1]];
            let mut events_end = at_event;
            while events_end < by_event.len()
                && self.ev_pos[by_event[events_end] as usize] as usize == i
            {
                events_end += 1;
            }
            if !trs.is_empty() {
                let risk = &global.risk[i * s..(i + 1) * s];
                let before = &global.phat_before[i * s..(i + 1) * s];
                let events = &by_event[at_event..events_end];
                for &row in events {
                    let r = row as usize;
                    let j = process.from[r] as usize;
                    c[local(row) * h_count + self.ev_jk[r] as usize] += process.weight[r] / risk[j];
                }
                // U <- U (I + H)
                let entries = &global.h[global.h_off[i]..global.h_off[i + 1]];
                for gl in 0..b {
                    let row = &mut u[gl * s..(gl + 1) * s];
                    old.copy_from_slice(row);
                    for e in entries {
                        row[e.k as usize] += old[e.j as usize] * e.value;
                    }
                }
                for &row in events {
                    let r = row as usize;
                    let (j, k) = (process.from[r] as usize, process.to[r] as usize);
                    if j != k {
                        let q = process.weight[r] * before[j] / risk[j];
                        let gl = local(row);
                        u[gl * s + j] -= q;
                        u[gl * s + k] += q;
                    }
                }
                for e in trs {
                    let (j, k) = (e.j as usize, e.k as usize);
                    for gl in 0..b {
                        let at_risk = wg[gl * s + j];
                        if at_risk > 0.0 {
                            c[gl * h_count + e.jk as usize] -= at_risk * e.htemp;
                            if j != k {
                                let q = at_risk * before[j] * e.htemp;
                                u[gl * s + j] += q;
                                u[gl * s + k] -= q;
                            }
                        }
                    }
                }
                cur_p.fill(0.0);
                for row in u.chunks_exact(s) {
                    for (sum, &value) in cur_p.iter_mut().zip(row) {
                        *sum += value * value;
                    }
                }
                cur_c.fill(0.0);
                for row in c.chunks_exact(h_count.max(1)) {
                    for (sum, &value) in cur_c.iter_mut().zip(row) {
                        *sum += value * value;
                    }
                }
            }
            at_event = events_end;

            let row = &mut part[i * per_time..(i + 1) * per_time];
            row[..s].copy_from_slice(&cur_p);
            row[s..s + h_count].copy_from_slice(&cur_c);
            row[s + h_count..].copy_from_slice(&sum_a);

            while snap < n_snap && req.snapshots[snap] == Some(i) {
                write(out, snap, &u);
                snap += 1;
            }
            t_prev = time;
        }
        while next_horizon < req.horizons.len() {
            horizon_sums(
                &mut part,
                next_horizon,
                req.horizons[next_horizon] - t_prev,
                &ua,
                &u,
            );
            next_horizon += 1;
        }
        part
    }
}
