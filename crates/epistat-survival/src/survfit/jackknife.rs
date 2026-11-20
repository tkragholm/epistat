//! The infinitesimal jackknife variance of a curve built from event times.
//!
//! # Algorithm
//!
//! The estimates are functions of the weights of the rows. Weight every row of
//! cluster `k` by `1 + e` and differentiate at `e = 0`. At event time `t_j`,
//! with `Y` the weight at risk, `d` the weight of events, `Y_k` the weight of
//! cluster `k` at risk and `d_k` its weight of events, every estimate treated
//! here has an increment `g(Y, d)`, and the derivative of the curve with
//! respect to cluster `k` is, up to a factor common to all clusters,
//!
//! ```text
//! phi_k(J) = sum over j <= J of  p_j d_kj - q_j Y_kj,
//! p_j = dg/dd,  q_j = -dg/dY.
//! ```
//!
//! The variance is the sum of `phi_k(J)^2` over the clusters, times that factor
//! squared. The caller supplies `p_j` and `q_j`. Each row `r` with weight
//! `w`, start `s`, stop `u` and event indicator `delta` adds to its cluster
//!
//! ```text
//! w [ delta p(u) 1(u <= J) - (Q(min(u, J)) - Q(s)) 1(s < J) ],
//! Q(t) = sum of q_j over event times t_j <= t.
//! ```
//!
//! While the row is at risk this is `w Q(s) - w Q(J)`, linear in `Q(J)`. So
//! each cluster keeps a constant `a_k` that collects the finished and entry
//! terms, and the weight `W_k` of its rows at risk, and
//! `phi_k(J) = a_k - Q(J) W_k`. A row changes its cluster's pair twice, when it
//! enters and when it leaves. Three running sums
//!
//! ```text
//! P = sum a_k^2,   C = sum a_k W_k,   R = sum W_k^2
//! ```
//!
//! then give `sum phi_k^2 = P - 2 Q C + Q^2 R` at every event time in constant
//! time, so the whole curve costs a sort and one pass over the rows, where
//! updating every cluster at every event time costs the number of clusters
//! times the number of event times.
//!
//! The sums are Neumaier-compensated, so their rounding error does not grow
//! with the number of updates. The combination `P - 2QC + Q^2 R` can still
//! lose digits when the variance is far below `P`. It is a sum of squares and
//! is not negative, so a computed value below zero, which only rounding can
//! produce, is returned as zero.
//!
//! # References
//!
//! Therneau T. M., Grambsch P. M. (2000) Modeling Survival Data, section 2.9
//! gives the influence of an observation on the Kaplan-Meier and
//! Nelson-Aalen estimates. Reid N. (1981) Estimating the median survival time,
//! Biometrika 68: 601-8 introduced the influence function of the
//! Kaplan-Meier estimate.

/// A running sum with Neumaier compensation.
#[derive(Debug, Clone, Copy, Default)]
struct Sum {
    value: f64,
    error: f64,
}

impl Sum {
    fn add(&mut self, x: f64) {
        let t = self.value + x;
        if self.value.abs() >= x.abs() {
            self.error += (self.value - t) + x;
        } else {
            self.error += (x - t) + self.value;
        }
        self.value = t;
    }

    fn get(&self) -> f64 {
        self.value + self.error
    }
}

/// The rows of one group and their order.
pub(super) struct Rows<'a> {
    /// Start of each risk interval, `None` for no delayed entry.
    pub(super) start: Option<&'a [f64]>,
    pub(super) stop: &'a [f64],
    /// Whether the row ends with the event.
    pub(super) event: &'a [bool],
    pub(super) weight: &'a [f64],
    /// Dense cluster code of each row, below `n_clusters`.
    pub(super) cluster: &'a [u32],
    pub(super) n_clusters: usize,
    /// Row indices by ascending start, when there is a start.
    pub(super) by_start: &'a [u32],
    /// Row indices by ascending stop.
    pub(super) by_stop: &'a [u32],
}

/// The state of every cluster and the sums over them.
struct State {
    a: Vec<f64>,
    w: Vec<f64>,
    alive: Vec<u32>,
    p: Sum,
    c: Sum,
    r: Sum,
}

impl State {
    fn change(&mut self, k: usize, da: f64, dw: f64, dalive: i64) {
        let (a, w) = (self.a[k], self.w[k]);
        self.p.add(-(a * a));
        self.c.add(-(a * w));
        self.r.add(-(w * w));
        self.a[k] = a + da;
        self.alive[k] = u32::try_from(i64::from(self.alive[k]) + dalive)
            .expect("a cluster leaves only rows it entered");
        // With no row at risk the weight is exactly zero, whatever the
        // additions and subtractions left.
        self.w[k] = if self.alive[k] == 0 { 0.0 } else { w + dw };
        let (a, w) = (self.a[k], self.w[k]);
        self.p.add(a * a);
        self.c.add(a * w);
        self.r.add(w * w);
    }
}

/// The sum over clusters of `phi_k(J)^2`, for each event time.
///
/// `times` are the event times in ascending order, and `p` and `q` the
/// derivatives of the increment at each of them. The result has one entry per
/// event time.
pub(super) fn sum_of_squares(rows: &Rows<'_>, times: &[f64], p: &[f64], q: &[f64]) -> Vec<f64> {
    let n = rows.stop.len();
    let mut state = State {
        a: vec![0.0; rows.n_clusters],
        w: vec![0.0; rows.n_clusters],
        alive: vec![0; rows.n_clusters],
        p: Sum::default(),
        c: Sum::default(),
        r: Sum::default(),
    };
    let mut cumulative = Sum::default();
    let mut entered = 0usize;
    let mut left = 0usize;
    let mut out = Vec::with_capacity(times.len());
    for (j, &t) in times.iter().enumerate() {
        let q_before = cumulative.get();
        // Rows that start before t are at risk at t. Without a start column
        // every row is at risk from the beginning.
        match rows.start {
            Some(start) => {
                while entered < n && start[rows.by_start[entered] as usize] < t {
                    let r = rows.by_start[entered] as usize;
                    let w = rows.weight[r];
                    state.change(rows.cluster[r] as usize, w * q_before, w, 1);
                    entered += 1;
                }
            }
            None => {
                while entered < n {
                    let r = rows.by_stop[entered] as usize;
                    state.change(rows.cluster[r] as usize, 0.0, rows.weight[r], 1);
                    entered += 1;
                }
            }
        }
        // Rows that ended before t without being at risk at an event time.
        while left < n && rows.stop[rows.by_stop[left] as usize] < t {
            let r = rows.by_stop[left] as usize;
            let w = rows.weight[r];
            state.change(rows.cluster[r] as usize, -(w * q_before), -w, -1);
            left += 1;
        }
        cumulative.add(q[j]);
        let q_now = cumulative.get();
        // Rows that end at t leave, an event adding its share.
        while left < n && rows.stop[rows.by_stop[left] as usize] <= t {
            let r = rows.by_stop[left] as usize;
            let w = rows.weight[r];
            let event = if rows.event[r] { w * p[j] } else { 0.0 };
            state.change(rows.cluster[r] as usize, event - w * q_now, -w, -1);
            left += 1;
        }
        let total = state.p.get() - 2.0 * q_now * state.c.get() + q_now * q_now * state.r.get();
        out.push(total.max(0.0));
    }
    out
}
