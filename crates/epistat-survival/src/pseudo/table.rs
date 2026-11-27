//! The rows of each group tabulated by exit time, and the estimators read off
//! the table.
//!
//! A group's table holds the distinct exit times, the weight and number of
//! rows at risk at each, and the weight and number of events, of any cause and
//! of the cause asked for. Both pseudo-value methods use it: the exact
//! jackknife edits it for each cluster left out, and the closed forms of the
//! infinitesimal jackknife read the risk sets and hazards from it. One function
//! turns a table into the estimate at chosen times, so the estimate and the
//! leave-one-out estimates are computed by the same arithmetic.

use crate::aalen_johansen::advance_step;
use crate::error::{Error, Result};
use crate::surv::SurvData;

use super::Target;

/// One row of a group.
#[derive(Debug, Clone, Copy)]
pub(super) struct Row {
    /// The first exit time (by index) at which the row is at risk.
    pub first: u32,
    /// The exit time (by index) at which the row ends.
    pub last: u32,
    pub start: f64,
    pub stop: f64,
    pub weight: f64,
    /// 0 for censored, otherwise the cause.
    pub status: u32,
}

/// The rows and the risk and event tables of one group.
#[derive(Debug)]
pub(super) struct Group {
    /// The caller's stratum code.
    pub code: u32,
    /// The distinct exit times, ascending.
    pub times: Vec<f64>,
    pub risk: Vec<f64>,
    pub risk_count: Vec<u32>,
    /// The weight of events of any cause.
    pub events: Vec<f64>,
    pub event_count: Vec<u32>,
    /// The weight of events of the cause asked for, empty without a cause.
    pub cause_events: Vec<f64>,
    pub cause_count: Vec<u32>,
    /// The rows, ordered by cluster and then by row.
    pub rows: Vec<Row>,
    /// `rows[offsets[c]..offsets[c + 1]]` are the rows of cluster `c`.
    pub offsets: Vec<usize>,
    /// The caller's label of each cluster: the cluster column, else the id
    /// column, else the row number.
    pub labels: Vec<u32>,
    /// Where the curves start: the smaller of 0 and every start and stop time
    /// of the rows used, in every group, as `survfit` takes its origin.
    pub origin: f64,
}

impl Group {
    pub(super) fn n_clusters(&self) -> usize {
        self.labels.len()
    }

    pub(super) fn cluster_rows(&self, cluster: usize) -> &[Row] {
        &self.rows[self.offsets[cluster]..self.offsets[cluster + 1]]
    }
}

/// Tabulates every group of `data`, whose times already follow the tie rule.
///
/// # Errors
///
/// An input error when no row has positive weight or a cluster has rows in
/// more than one group.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the data type checks that the row count fits in u32, and an exit time index is below it"
)]
#[expect(clippy::float_cmp, reason = "rows share an exit time or they do not")]
pub(super) fn build(data: &SurvData<'_>, cause: Option<u32>) -> Result<Vec<Group>> {
    let surv = data.followup()?;
    if data.n_used() == 0 {
        return Err(Error::no_information("no row has positive weight"));
    }
    let n_groups = data.n_strata().max(1);
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_groups];
    for row in data.used_rows() {
        let g = data.stratum().map_or(0, |s| s[row as usize] as usize);
        members[g].push(row);
    }
    let cluster_codes = data.cluster().or_else(|| data.id());
    let cluster_levels = if data.cluster().is_some() {
        data.cluster_levels()
    } else {
        data.id_levels()
    };
    if let Some(codes) = cluster_codes {
        let mut owner = vec![
            u32::MAX;
            data.cluster()
                .map_or_else(|| data.n_ids(), |_| data.n_clusters())
        ];
        for (g, rows) in members.iter().enumerate() {
            for &r in rows {
                let slot = &mut owner[codes[r as usize] as usize];
                if *slot == u32::MAX {
                    *slot = g as u32;
                } else if *slot != g as u32 {
                    return Err(Error::inconsistent(
                        "a cluster has rows in more than one stratum, so its pseudo-values \
                         cannot be collapsed",
                    ));
                }
            }
        }
    }
    let weight = data.weight();
    let mut origin = 0.0f64;
    for row in data.used_rows() {
        let r = row as usize;
        origin = origin.min(surv.stop()[r]);
        if let Some(start) = surv.start() {
            origin = origin.min(start[r]);
        }
    }
    let mut groups = Vec::new();
    for (g, rows) in members.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let mut times: Vec<f64> = rows.iter().map(|&r| surv.stop()[r as usize]).collect();
        times.sort_by(f64::total_cmp);
        times.dedup_by(|a, b| *a == *b);
        let m = times.len();

        // The rows ordered by cluster, rows of a cluster in row order.
        let key = |r: u32| cluster_codes.map_or(r, |codes| codes[r as usize]);
        let mut ordered: Vec<u32> = rows.clone();
        ordered.sort_by_key(|&r| (key(r), r));
        let mut table_rows = Vec::with_capacity(ordered.len());
        let mut offsets = vec![0usize];
        let mut labels = Vec::new();
        let mut risk_diff = vec![0.0f64; m + 1];
        let mut count_diff = vec![0i64; m + 1];
        let mut events = vec![0.0f64; m];
        let mut event_count = vec![0u32; m];
        let mut cause_events = vec![0.0f64; if cause.is_some() { m } else { 0 }];
        let mut cause_count = vec![0u32; cause_events.len()];
        for (position, &r) in ordered.iter().enumerate() {
            if position > 0 && key(ordered[position - 1]) != key(r) {
                offsets.push(position);
            }
            if position == 0 || key(ordered[position - 1]) != key(r) {
                let k = key(r);
                labels.push(cluster_codes.map_or(k, |_| {
                    cluster_levels.map_or(k, |l| l.code(k as usize).unwrap_or(k))
                }));
            }
        }
        offsets.push(ordered.len());
        for &r in &ordered {
            let i = r as usize;
            let stop = surv.stop()[i];
            let start = surv.start().map_or(f64::NEG_INFINITY, |s| s[i]);
            let w = weight.map_or(1.0, |w| w[i]);
            let first = times.partition_point(|&t| t <= start);
            let last = times
                .binary_search_by(|t| t.total_cmp(&stop))
                .expect("the stop time of a used row is an exit time");
            let status = u32::try_from(surv.status()[i]).expect("a status is not negative");
            risk_diff[first] += w;
            risk_diff[last + 1] -= w;
            count_diff[first] += 1;
            count_diff[last + 1] -= 1;
            if status > 0 {
                events[last] += w;
                event_count[last] += 1;
                if cause == Some(status) {
                    cause_events[last] += w;
                    cause_count[last] += 1;
                }
            }
            table_rows.push(Row {
                first: first as u32,
                last: last as u32,
                start,
                stop,
                weight: w,
                status,
            });
        }
        let mut risk = Vec::with_capacity(m);
        let mut risk_count = Vec::with_capacity(m);
        let (mut run, mut run_count) = (0.0f64, 0i64);
        for j in 0..m {
            run += risk_diff[j];
            run_count += count_diff[j];
            risk_count.push(u32::try_from(run_count).expect("a count at risk is not negative"));
            // With nobody at risk the weight is exactly zero, whatever the
            // additions and subtractions left.
            risk.push(if run_count == 0 { 0.0 } else { run });
        }
        groups.push(Group {
            code: data
                .stratum_levels()
                .map_or(0, |l| l.code(g).expect("a group is a level of the stratum")),
            times,
            risk,
            risk_count,
            events,
            event_count,
            cause_events,
            cause_count,
            rows: table_rows,
            offsets,
            labels,
            origin,
        });
    }
    Ok(groups)
}

/// The columns of a table, or of a table with a cluster left out.
#[derive(Debug, Clone, Copy)]
pub(super) struct Columns<'a> {
    pub times: &'a [f64],
    pub risk: &'a [f64],
    pub events: &'a [f64],
    /// Empty without a cause.
    pub cause_events: &'a [f64],
}

/// What is read from the curves: the target and the times, with
/// `reads[k]` the number of exit times not after `at[k]`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Reading<'a> {
    pub target: Target,
    pub at: &'a [f64],
    pub reads: &'a [usize],
}

/// The estimate of the target at each of the times asked for, written to
/// `out`.
///
/// The curves start at `origin`, with the event-free probability 1 and no
/// incidence, and are held at their last value beyond the last time.
pub(super) fn evaluate(columns: &Columns<'_>, origin: f64, reading: &Reading<'_>, out: &mut [f64]) {
    let Columns {
        times,
        risk,
        events,
        cause_events,
    } = *columns;
    let Reading { target, at, reads } = *reading;
    let with_cause = !cause_events.is_empty();
    let mut survival = 1.0f64;
    let mut incidence = [0.0f64];
    let (mut area_s, mut area_f) = (0.0f64, 0.0f64);
    let mut cumulative_hazard = 0.0f64;
    let mut previous = origin;
    let mut k = 0;
    let read =
        |tau: f64, s: f64, f: f64, area_s: f64, area_f: f64, previous: f64, h: f64| match target {
            Target::Survival => s,
            Target::CumulativeIncidence { .. } => f,
            Target::RestrictedMean => area_s + s * (tau - previous),
            Target::RestrictedTimeLost { .. } => area_f + f * (tau - previous),
            Target::CumulativeHazard | Target::CauseHazard { .. } => h,
        };
    for j in 0..times.len() {
        while k < at.len() && reads[k] == j {
            out[k] = read(
                at[k],
                survival,
                incidence[0],
                area_s,
                area_f,
                previous,
                cumulative_hazard,
            );
            k += 1;
        }
        let step = times[j] - previous;
        area_s += survival * step;
        area_f += incidence[0] * step;
        previous = times[j];
        match target {
            Target::CumulativeHazard if events[j] > 0.0 => {
                cumulative_hazard += events[j] / risk[j];
            }
            Target::CauseHazard { .. } if with_cause && cause_events[j] > 0.0 => {
                cumulative_hazard += cause_events[j] / risk[j];
            }
            _ => {}
        }
        if with_cause {
            advance_step(
                &mut survival,
                risk[j],
                events[j],
                &cause_events[j..=j],
                &mut incidence,
            );
        } else {
            advance_step(&mut survival, risk[j], events[j], &[], &mut incidence);
        }
    }
    while k < at.len() {
        out[k] = read(
            at[k],
            survival,
            incidence[0],
            area_s,
            area_f,
            previous,
            cumulative_hazard,
        );
        k += 1;
    }
}
