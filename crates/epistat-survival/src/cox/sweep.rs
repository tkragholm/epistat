//! The sweep over the event times of a stratum with the risk-set sums, shared
//! by the fit and by the residuals.
//!
//! A stratum's rows are in descending exit order. At each event time, in
//! descending order, rows enter the risk set when their exit is at or after the
//! time and leave it when their entry is at or after the time, and the sums of
//! the weights `c`, of `c x` and of `c x x'` over the risk set and over the
//! events at the time are handed to a visitor.

use super::moments;

/// The sums `sum c x^r` for `r = 0, 1, 2`, with the lower triangle of `r = 2`
/// when the second moment is wanted.
pub(crate) struct Moments {
    pub(crate) m0: f64,
    pub(crate) m1: Vec<f64>,
    pub(crate) m2: Vec<f64>,
}

impl Moments {
    fn new(p: usize, second: bool) -> Self {
        Self {
            m0: 0.0,
            m1: vec![0.0; p],
            m2: vec![0.0; if second { p * p } else { 0 }],
        }
    }

    fn clear(&mut self) {
        self.m0 = 0.0;
        self.m1.fill(0.0);
        self.m2.fill(0.0);
    }

    fn add(&mut self, c: f64, x: &[f64]) {
        self.m0 += c;
        moments::add(c, x, &mut self.m1, &mut self.m2);
    }
}

/// The sums over the risk set and over the events at one time.
pub(crate) struct Sums {
    pub(crate) risk: Moments,
    pub(crate) dying: Moments,
}

impl Sums {
    pub(crate) fn new(p: usize, second: bool) -> Self {
        Self {
            risk: Moments::new(p, second),
            dying: Moments::new(p, second),
        }
    }
}

/// What the sweep reads of one stratum.
pub(crate) struct RiskSets<'a> {
    /// Covariates, `p` per row, in descending exit order.
    pub(crate) x: &'a [f64],
    /// Exit times in descending order.
    pub(crate) stop: &'a [f64],
    /// Positions in descending entry order, and the entry times in that order.
    /// Both are empty when no row has a delayed entry.
    pub(crate) by_start: &'a [u32],
    pub(crate) start_sorted: &'a [f64],
    /// Distinct event times, descending.
    pub(crate) event_times: &'a [f64],
    /// The positions of the events at each time:
    /// `dying[dying_offset[k]..dying_offset[k + 1]]`.
    pub(crate) dying: &'a [u32],
    pub(crate) dying_offset: &'a [usize],
}

impl RiskSets<'_> {
    /// Visit every event time in descending order with the risk-set sums and
    /// the sums over the events at that time, at the per-row weights `risk`,
    /// which are `w exp(beta' x)`.
    pub(crate) fn sweep(
        &self,
        p: usize,
        risk: &[f64],
        active: &mut Vec<bool>,
        sums: &mut Sums,
        mut visit: impl FnMut(usize, &Sums),
    ) {
        let size = self.stop.len();
        active.clear();
        active.resize(size, false);
        sums.risk.clear();
        let (mut entering, mut leaving) = (0usize, 0usize);
        for (position, &time) in self.event_times.iter().enumerate() {
            while entering < size && self.stop[entering] >= time {
                sums.risk
                    .add(risk[entering], &self.x[entering * p..(entering + 1) * p]);
                active[entering] = true;
                entering += 1;
            }
            while leaving < self.by_start.len() && self.start_sorted[leaving] >= time {
                let row = self.by_start[leaving] as usize;
                if active[row] {
                    sums.risk.add(-risk[row], &self.x[row * p..(row + 1) * p]);
                    active[row] = false;
                }
                leaving += 1;
            }
            sums.dying.clear();
            for &row in &self.dying[self.dying_offset[position]..self.dying_offset[position + 1]] {
                let row = row as usize;
                sums.dying.add(risk[row], &self.x[row * p..(row + 1) * p]);
            }
            visit(position, sums);
        }
    }
}

/// The terms of the partial likelihood at one event time as `(weight,
/// fraction)`: the denominator is the risk-set sum less `fraction` of the sum
/// over the events, and the term is multiplied by `weight`. `dying` holds the
/// positions of the events at the time and `weight` the case weight of each
/// position. Breslow's approximation has one term, Efron's one per event.
pub(crate) fn terms(
    weight: &[f64],
    dying: &[u32],
    efron: bool,
) -> impl Iterator<Item = (f64, f64)> + use<> {
    let count = dying.len();
    let total: f64 = dying.iter().map(|&row| weight[row as usize]).sum();
    let (terms, weight, spread) = if efron {
        (count, total / count as f64, 1.0 / count as f64)
    } else {
        (1, total, 0.0)
    };
    (0..terms).map(move |k| (weight, k as f64 * spread))
}
