//! The stacked data of a multi-state model: one copy of each row for every
//! transition that leaves the state the row starts in.

use faer::{Mat, MatRef};

use super::{MsBaseline, MsModel, MsOptions, MsTerm, MsTerms, MsTransition, resolve};
use crate::check::{check_lengths, check_row_count, refuse_non_finite};
use crate::cox::{self, CoxOptions};
use crate::error::{Error, Result};
use crate::surv::{SurvBuf, SurvData};
use crate::timefix::TimeFix;

/// The transitions of the rows of positive weight, sorted by the state left and
/// the state entered.
pub(super) fn observed(data: &SurvData<'_>, istate: &[u32]) -> Result<Vec<(u32, u32)>> {
    let surv = data.followup()?;
    let status = surv.status();
    let mut seen = Vec::new();
    for row in data.used_rows() {
        let row = row as usize;
        let from = istate[row];
        if from == 0 {
            return Err(Error::input(format!(
                "row {row} starts in an unknown state: give `istate` for the first row of each subject"
            )));
        }
        let to = u32::try_from(status[row]).expect("the builder keeps the status from below zero");
        if to == from {
            return Err(Error::input(format!(
                "row {row} ends in state {to}, the state it starts in"
            )));
        }
        if to != 0 {
            seen.push((from, to));
        }
    }
    seen.sort_unstable();
    seen.dedup();
    if seen.is_empty() {
        return Err(Error::input("no row of positive weight has a transition"));
    }
    Ok(seen)
}

/// The transitions of the model: the caller's, checked, or those observed.
fn transitions_of(
    opts: &MsOptions<'_>,
    data: &SurvData<'_>,
    istate: &[u32],
    n_states: u32,
) -> Result<Vec<(u32, u32)>> {
    // The rows are checked whichever transitions are asked for.
    let seen = observed(data, istate)?;
    let Some(given) = opts.transitions else {
        return Ok(seen);
    };
    let mut list = given.to_vec();
    list.sort_unstable();
    if list.is_empty() {
        return Err(Error::input("transitions: the list is empty"));
    }
    for window in list.windows(2) {
        if window[0] == window[1] {
            return Err(Error::input(format!(
                "transitions: ({}, {}) is listed twice",
                window[0].0, window[0].1
            )));
        }
    }
    for &(from, to) in &list {
        if from == 0 || from > n_states || to == 0 || to > n_states {
            return Err(Error::input(format!(
                "transitions: ({from}, {to}) is outside the states 1 to {n_states}"
            )));
        }
        if from == to {
            return Err(Error::input(format!(
                "transitions: ({from}, {to}) leaves and enters one state"
            )));
        }
    }
    Ok(list)
}

/// The terms of the model, checked against the covariates and the transitions.
fn terms_of(opts: &MsOptions<'_>, n_columns: usize, n_transitions: usize) -> Result<Vec<MsTerm>> {
    let all: Vec<usize> = (0..n_transitions).collect();
    let terms = match opts.terms {
        MsTerms::Separate => (0..n_columns)
            .flat_map(|c| (0..n_transitions).map(move |k| MsTerm::covariate(c, &[k])))
            .collect(),
        MsTerms::Common => (0..n_columns).map(|c| MsTerm::covariate(c, &all)).collect(),
        MsTerms::Custom(terms) => terms.to_vec(),
    };
    let mut covered = vec![false; (n_columns + 1) * n_transitions];
    for (t, term) in terms.iter().enumerate() {
        if let Some(column) = term.column {
            if column >= n_columns {
                return Err(Error::input(format!(
                    "terms: term {t} uses column {column} of {n_columns}"
                )));
            }
        }
        if term.transitions.is_empty() {
            return Err(Error::input(format!("terms: term {t} has no transition")));
        }
        for &k in &term.transitions {
            if k >= n_transitions {
                return Err(Error::input(format!(
                    "terms: term {t} uses transition {k} of {n_transitions}"
                )));
            }
            let slot = term.column.unwrap_or(n_columns) * n_transitions + k;
            if covered[slot] {
                return Err(Error::input(format!(
                    "terms: term {t} repeats a coefficient for transition {k}"
                )));
            }
            covered[slot] = true;
        }
    }
    Ok(terms)
}

/// The baseline group of each transition as a dense index.
fn groups_of(opts: &MsOptions<'_>, n_transitions: usize) -> Result<Vec<usize>> {
    let codes: Vec<u32> = match opts.baseline {
        MsBaseline::Separate => (0..n_transitions)
            .map(|k| u32::try_from(k).expect("a transition count is below u32::MAX"))
            .collect(),
        MsBaseline::Shared => vec![0; n_transitions],
        MsBaseline::Groups(codes) => {
            check_lengths(&[("transitions", n_transitions), ("baseline", codes.len())])?;
            codes.to_vec()
        }
    };
    let mut sorted = codes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    Ok(codes
        .iter()
        .map(|c| sorted.binary_search(c).expect("the code is in the list"))
        .collect())
}

pub(super) fn fit(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    opts: &MsOptions<'_>,
) -> Result<MsModel> {
    let rows = data.n_rows();
    check_lengths(&[("data", rows), ("x", x.nrows())])?;
    check_row_count(rows)?;
    if let Some(offset) = opts.offset {
        check_lengths(&[("data", rows), ("offset", offset.len())])?;
        refuse_non_finite(offset, "offset")?;
    }
    let non_finite = (0..x.nrows())
        .flat_map(|i| (0..x.ncols()).map(move |j| (i, j)))
        .filter(|&(i, j)| !x[(i, j)].is_finite())
        .count();
    if non_finite > 0 {
        return Err(Error::input(format!(
            "x: {non_finite} non-finite values; drop or resolve them before estimating"
        )));
    }

    let (data, istate) = resolve(data, opts.timefix)?;
    let surv = data.followup()?;
    let n_states = surv.n_codes();
    let status = surv.status();
    let pairs = transitions_of(opts, &data, &istate, n_states)?;
    let n_transitions = pairs.len();
    let terms = terms_of(opts, x.ncols(), n_transitions)?;
    let groups = groups_of(opts, n_transitions)?;

    // The rows of positive weight by the state they start in.
    let mut by_state: Vec<Vec<u32>> = vec![Vec::new(); n_states as usize + 1];
    for row in data.used_rows() {
        by_state[istate[row as usize] as usize].push(row);
    }
    let n_stacked: usize = pairs
        .iter()
        .map(|&(from, _)| by_state[from as usize].len())
        .sum();
    check_row_count(n_stacked)?;

    // Without a term the design has no column and the fit is the baseline
    // hazards only.
    let p = terms.len();
    if p == 0 && opts.init.is_some() {
        return Err(Error::input("init: the model has no coefficient"));
    }
    let mut design = Mat::<f64>::zeros(n_stacked, p);
    let mut stop = Vec::with_capacity(n_stacked);
    let mut start = surv.start().map(|_| Vec::with_capacity(n_stacked));
    let mut event = Vec::with_capacity(n_stacked);
    let mut weight = data.weight().map(|_| Vec::with_capacity(n_stacked));
    let mut stratum = Vec::with_capacity(n_stacked);
    let mut offset = opts.offset.map(|_| Vec::with_capacity(n_stacked));
    let clusters = data
        .cluster()
        .or_else(|| opts.robust.then(|| data.id()).flatten());
    let mut cluster = (opts.robust && clusters.is_some()).then(|| Vec::with_capacity(n_stacked));
    let mut n_events = vec![0_usize; n_transitions];
    let mut last_time = f64::NEG_INFINITY;
    let mut out = 0_usize;
    for (k, &(from, to)) in pairs.iter().enumerate() {
        let group = u32::try_from(groups[k]).expect("a group index is below u32::MAX");
        let columns: Vec<(usize, Option<usize>)> = terms
            .iter()
            .enumerate()
            .filter(|(_, term)| term.transitions.contains(&k))
            .map(|(t, term)| (t, term.column))
            .collect();
        for &row in &by_state[from as usize] {
            let i = row as usize;
            stop.push(surv.stop()[i]);
            last_time = last_time.max(surv.stop()[i]);
            if let (Some(start), Some(s)) = (start.as_mut(), surv.start()) {
                start.push(s[i]);
            }
            let hit = u32::try_from(status[i]).expect("the status is not below zero") == to;
            n_events[k] += usize::from(hit);
            event.push(i32::from(hit));
            if let (Some(weight), Some(w)) = (weight.as_mut(), data.weight()) {
                weight.push(w[i]);
            }
            stratum.push(group);
            if let (Some(offset), Some(o)) = (offset.as_mut(), opts.offset) {
                offset.push(o[i]);
            }
            if let (Some(cluster), Some(c)) = (cluster.as_mut(), clusters) {
                cluster.push(c[i]);
            }
            for &(t, column) in &columns {
                design[(out, t)] = column.map_or(1.0, |c| x[(i, c)]);
            }
            out += 1;
        }
    }

    let mut builder = SurvBuf::builder(stop, event).stratum(stratum);
    if let Some(start) = start {
        builder = builder.start(start);
    }
    if let Some(weight) = weight {
        builder = builder.weight(weight);
    }
    if let Some(cluster) = cluster {
        builder = builder.cluster(cluster);
    }
    let stacked = builder.build()?;
    let cox_options = CoxOptions {
        ties: opts.ties,
        // The tie rule has been applied to the rows before they were stacked.
        timefix: TimeFix::Exact,
        cause: None,
        offset: offset.as_deref(),
        init: opts.init,
        iter_max: opts.iter_max,
        eps: opts.eps,
    };
    let mut fit = cox::fit(&stacked.as_data(), design.as_ref(), &cox_options)?;
    fit.n_rows = data.n_rows();
    fit.n_used = data.n_used();
    let transitions = pairs
        .iter()
        .enumerate()
        .map(|(k, &(from, to))| MsTransition {
            from,
            to,
            baseline: groups[k],
            n_events: n_events[k],
        })
        .collect();
    let n_ids = data.n_ids();
    Ok(MsModel {
        transitions,
        terms,
        n_states,
        fit,
        n_stacked,
        n_ids,
        stacked,
        design,
        n_covariates: x.ncols(),
        offset,
        ties: opts.ties,
        last_time,
    })
}
