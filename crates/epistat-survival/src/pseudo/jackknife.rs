//! The exact leave-one-out jackknife, one cluster at a time.
//!
//! # Algorithm
//!
//! With `theta` the estimate from all `n` clusters of a group and
//! `theta_(-i)` the estimate with cluster `i` left out, the pseudo-value of
//! cluster `i` is `n theta - (n - 1) theta_(-i)` \[1\]\[2\].
//!
//! Leaving a cluster out changes the table of the group, and nothing else. A
//! row at risk on the exit times `first..=last` takes its weight and one row
//! from the weight and count at risk at each of them, and a row that ends with
//! an event takes its weight and one event from the events at `last`. All the
//! rows of a cluster are applied as differences, whose running sum gives the
//! change at every time, so the table without the cluster costs a pass over the
//! exit times and the rows of the cluster, and the estimate is read off it by
//! the arithmetic of the estimate itself. A cluster costs `O(times + rows)`, the
//! whole jackknife `O(clusters times)`, which is the cost of the infinitesimal
//! jackknife, and no sort, count or search is repeated. An exit time that
//! only the cluster carries stays in the table with nobody at risk and no
//! events, where the estimate does not move, as it does not in a fit on the
//! rest of the data. Counts of rows are kept beside the weights, so that a
//! weight at risk or of events that has to be zero is exactly zero and not what
//! the subtraction leaves.
//!
//! Each cluster is independent of the others and is computed by one thread
//! alone, and the results are collected in cluster order, so the bits do not
//! depend on the number of threads.
//!
//! # References
//!
//! 1. Andersen P. K., Klein J. P., Rosthoj S. (2003) Generalised linear models
//!    for correlated pseudo-observations, with applications to multi-state
//!    models, Biometrika 90: 15-27.
//! 2. Klein J. P., Gerster M., Andersen P. K., Tarima S., Perme M. P. (2008)
//!    SAS and R functions to compute pseudo-values for censored data regression,
//!    Comput Methods Programs Biomed 89: 289-300.

use rayon::prelude::*;

use super::table::{Columns, Group, Reading, evaluate};

/// The table of a group with one cluster left out, and the differences that
/// make it.
struct Scratch {
    risk: Vec<f64>,
    events: Vec<f64>,
    cause_events: Vec<f64>,
    risk_diff: Vec<f64>,
    count_diff: Vec<i64>,
    event_diff: Vec<f64>,
    event_count_diff: Vec<i64>,
    cause_diff: Vec<f64>,
    cause_count_diff: Vec<i64>,
    out: Vec<f64>,
}

impl Scratch {
    fn new(group: &Group, n_times: usize) -> Self {
        let m = group.times.len();
        let c = group.cause_events.len();
        Self {
            risk: vec![0.0; m],
            events: vec![0.0; m],
            cause_events: vec![0.0; c],
            risk_diff: vec![0.0; m + 1],
            count_diff: vec![0; m + 1],
            event_diff: vec![0.0; m],
            event_count_diff: vec![0; m],
            cause_diff: vec![0.0; c],
            cause_count_diff: vec![0; c],
            out: vec![0.0; n_times],
        }
    }
}

/// The pseudo-values of every cluster of the group at the times in `reading`,
/// row-major by cluster. `estimate` is the estimate of the whole group at
/// those times.
pub(super) fn run(group: &Group, reading: &Reading<'_>, estimate: &[f64]) -> Vec<f64> {
    let k = reading.at.len();
    let n = group.n_clusters();
    let cause = reading.target.cause();
    let m = group.times.len();
    #[expect(
        clippy::cast_precision_loss,
        reason = "a cluster count is far below 2^53"
    )]
    let (n_f, n_less) = (n as f64, (n - 1) as f64);
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map_init(
            || Scratch::new(group, k),
            |s, c| {
                let rows = group.cluster_rows(c);
                for row in rows {
                    let (first, last) = (row.first as usize, row.last as usize);
                    s.risk_diff[first] += row.weight;
                    s.risk_diff[last + 1] -= row.weight;
                    s.count_diff[first] += 1;
                    s.count_diff[last + 1] -= 1;
                    if row.status > 0 {
                        s.event_diff[last] += row.weight;
                        s.event_count_diff[last] += 1;
                        if cause == Some(row.status) {
                            s.cause_diff[last] += row.weight;
                            s.cause_count_diff[last] += 1;
                        }
                    }
                }
                let (mut run, mut run_count) = (0.0f64, 0i64);
                for j in 0..m {
                    run += s.risk_diff[j];
                    run_count += s.count_diff[j];
                    s.risk[j] = if i64::from(group.risk_count[j]) - run_count == 0 {
                        0.0
                    } else {
                        group.risk[j] - run
                    };
                    s.events[j] = if i64::from(group.event_count[j]) - s.event_count_diff[j] == 0 {
                        0.0
                    } else {
                        group.events[j] - s.event_diff[j]
                    };
                    if !s.cause_events.is_empty() {
                        s.cause_events[j] =
                            if i64::from(group.cause_count[j]) - s.cause_count_diff[j] == 0 {
                                0.0
                            } else {
                                group.cause_events[j] - s.cause_diff[j]
                            };
                    }
                }
                for row in rows {
                    let (first, last) = (row.first as usize, row.last as usize);
                    s.risk_diff[first] = 0.0;
                    s.risk_diff[last + 1] = 0.0;
                    s.count_diff[first] = 0;
                    s.count_diff[last + 1] = 0;
                    s.event_diff[last] = 0.0;
                    s.event_count_diff[last] = 0;
                    if !s.cause_diff.is_empty() {
                        s.cause_diff[last] = 0.0;
                        s.cause_count_diff[last] = 0;
                    }
                }
                evaluate(
                    &Columns {
                        times: &group.times,
                        risk: &s.risk,
                        events: &s.events,
                        cause_events: &s.cause_events,
                    },
                    group.origin,
                    reading,
                    &mut s.out,
                );
                (0..k)
                    .map(|i| n_f * estimate[i] - n_less * s.out[i])
                    .collect()
            },
        )
        .collect();
    rows.concat()
}
