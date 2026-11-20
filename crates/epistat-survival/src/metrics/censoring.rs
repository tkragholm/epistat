//! The Kaplan-Meier estimate of the censoring distribution, shared by the
//! weights of the metrics.

#![expect(
    clippy::float_cmp,
    reason = "times tie when they are equal, which is the definition of a tie"
)]

/// The censoring distribution `G(t) = P(C > t)` as a step function.
///
/// The risk set at a censoring time `c` is the rows censored at `c` and the
/// rows with a later time. A death at `c` has left it: deaths come before
/// censorings at a tied time. `survival::rttright` and `brier` reach this by
/// moving every censoring time right by less than the smallest gap, and
/// `prodlim` with `reverse = TRUE`, which `riskRegression::Score` uses, does
/// the same at a tie. With `deaths_first` false the deaths at `c` are in the
/// risk set, the plain Kaplan-Meier estimate of the censoring times, which is
/// what `survfit(Surv(time, 1 - status) ~ 1)` gives.
#[derive(Debug, Clone)]
pub(super) struct CensoringKm {
    /// The distinct censoring times, ascending.
    pub time: Vec<f64>,
    /// `G` just after each time.
    pub surv: Vec<f64>,
    /// The weight at risk at each time, as the estimate itself counts it.
    pub at_risk: Vec<f64>,
    /// The weight of every row with a time at or after each time, deaths at
    /// the time included. It equals `at_risk` when no death ties a censoring.
    pub at_risk_plain: Vec<f64>,
    /// The weight censored at each time.
    pub n_censored: Vec<f64>,
}

impl CensoringKm {
    /// Build from rows sorted by ascending time. `weight` is the case weight
    /// of each row, `censored` says whether it was censored.
    pub(super) fn new(time: &[f64], censored: &[bool], weight: &[f64], deaths_first: bool) -> Self {
        let n = time.len();
        // Weight of the rows from each position to the end, summed from the
        // end so that the order is fixed.
        let mut suffix = vec![0.0; n + 1];
        for i in (0..n).rev() {
            suffix[i] = suffix[i + 1] + weight[i];
        }
        let mut out = Self {
            time: Vec::new(),
            surv: Vec::new(),
            at_risk: Vec::new(),
            at_risk_plain: Vec::new(),
            n_censored: Vec::new(),
        };
        let mut g = 1.0;
        let mut i = 0;
        while i < n {
            let mut j = i;
            let mut deaths = 0.0;
            let mut cens = 0.0;
            while j < n && time[j] == time[i] {
                if censored[j] {
                    cens += weight[j];
                } else {
                    deaths += weight[j];
                }
                j += 1;
            }
            if cens > 0.0 {
                let risk = if deaths_first {
                    suffix[i] - deaths
                } else {
                    suffix[i]
                };
                g *= 1.0 - cens / risk;
                out.time.push(time[i]);
                out.surv.push(g);
                out.at_risk.push(risk);
                out.at_risk_plain.push(suffix[i]);
                out.n_censored.push(cens);
            }
            i = j;
        }
        out
    }

    /// `G(t)`: the censorings at or before `t` are counted.
    pub(super) fn at(&self, t: f64) -> f64 {
        let k = self.time.partition_point(|&u| u <= t);
        if k == 0 { 1.0 } else { self.surv[k - 1] }
    }

    /// `G(t-)`: only the censorings before `t` are counted.
    pub(super) fn before(&self, t: f64) -> f64 {
        let k = self.time.partition_point(|&u| u < t);
        if k == 0 { 1.0 } else { self.surv[k - 1] }
    }
}
