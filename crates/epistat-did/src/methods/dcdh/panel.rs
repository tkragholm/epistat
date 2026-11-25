//! The sample rules of `did_multiplegt_main`: which rows and groups enter,
//! the baseline treatment, the first-switch date, the missing-data
//! conventions and the balanced grid of groups and periods.

use std::collections::HashMap;

use super::{DcdhConfig, DcdhError, DcdhObservation};

/// Not available, as R's `NA`. Every quantity that R stores with `NA` is NaN
/// here, so that sums with `na.rm = TRUE` skip the same terms.
pub(super) const NA: f64 = f64::NAN;

/// The balanced grid of the groups that remain and the periods `t_min..=t_max`
/// (ranks of the distinct input periods, from 1), with the group-level
/// variables of the estimator.
pub(super) struct Panel {
    /// Input identifiers of the groups in the grid, increasing.
    pub group_ids: Vec<i64>,
    /// Input identifiers of all groups counted in `n_groups`, increasing.
    pub counted_ids: Vec<i64>,
    /// `G`, the number of groups after the baseline rule, which divides every
    /// estimate.
    pub n_groups: f64,
    pub t_min: i64,
    pub t_max: i64,
    /// Number of periods in the grid.
    pub nt: usize,
    /// Outcome and treatment of each cell, NaN where missing or filled in.
    pub y: Vec<f64>,
    pub d: Vec<f64>,
    /// `N_gt`: the weight where outcome and treatment are both observed, else 0.
    pub n_gt: Vec<f64>,
    /// Sampling weight of the cell, NaN for a cell the balancing filled in.
    pub weight: Vec<f64>,
    /// Whether the treatment has differed from the baseline by this row
    /// (1 or 0), NaN for a filled cell.
    pub ever_change: Vec<f64>,
    /// Controls of each cell, `controls[k][cell]`, NaN for a filled cell.
    pub controls: Vec<Vec<f64>>,
    /// Group variables.
    pub d_sq: Vec<f64>,
    pub f_g: Vec<i64>,
    pub t_g: Vec<i64>,
    /// 1 for a switcher in, 0 for a switcher out, NaN otherwise.
    pub s_g: Vec<f64>,
    pub l_g: Vec<i64>,
    /// Number of placebos the group allows, `None` where `F_g < 3`.
    pub l_g_placebo: Vec<Option<i64>>,
    /// Index of `(d_sq, stratum)`.
    pub base_id: Vec<usize>,
    pub n_base: usize,
    /// Index of `d_sq` alone, the baseline levels of the controls.
    pub level: Vec<usize>,
    pub n_level: usize,
    /// Index of `(d_sq, F_g, d_fg, stratum)`, the switchers' cohorts.
    pub cohort_id: Vec<usize>,
    pub n_cohort: usize,
    /// Cluster index of each group, `None` without clustering.
    pub cluster: Option<Vec<usize>>,
    pub n_cluster: usize,
}

impl Panel {
    pub fn n_grid_groups(&self) -> usize {
        self.group_ids.len()
    }

    pub fn time_of(&self, tp: usize) -> i64 {
        self.t_min + i64::try_from(tp).expect("period index fits i64")
    }
}

/// One input row after validation, with the period as a rank.
#[derive(Clone)]
struct Row {
    group: usize,
    rank: i64,
    y: f64,
    d: f64,
    weight: f64,
    controls: Vec<f64>,
}

fn key(value: f64) -> u64 {
    // R groups by value, for which -0 and 0 are equal.
    (value + 0.0).to_bits()
}

/// Dense indices of the distinct keys, in increasing order of the keys.
fn index_of<K: Ord + Clone + std::hash::Hash>(keys: &[K]) -> (Vec<usize>, usize) {
    let mut distinct: Vec<K> = keys.to_vec();
    distinct.sort();
    distinct.dedup();
    let map: HashMap<K, usize> = distinct
        .iter()
        .enumerate()
        .map(|(i, k)| (k.clone(), i))
        .collect();
    (keys.iter().map(|k| map[k]).collect(), distinct.len())
}

fn ordered(x: f64) -> i64 {
    // A total order on finite values for use as a sort key.
    let bits = key(x).cast_signed();
    if bits < 0 { bits ^ i64::MAX } else { bits }
}

#[expect(
    clippy::too_many_lines,
    reason = "the sample rules of did_multiplegt_main, in R's order"
)]
pub(super) fn prepare(
    observations: &[DcdhObservation],
    config: &DcdhConfig,
) -> Result<Panel, DcdhError> {
    if observations.is_empty() {
        return Err(DcdhError::Empty);
    }
    let n_controls = observations[0].controls.len();
    let clustered = observations.iter().any(|o| o.cluster.is_some());
    for (i, o) in observations.iter().enumerate() {
        if o.controls.len() != n_controls {
            return Err(DcdhError::InvalidRow {
                row: i,
                reason: "every row needs the same number of controls",
            });
        }
        if o.controls.iter().any(|x| !x.is_finite()) {
            return Err(DcdhError::InvalidRow {
                row: i,
                reason: "a control is not finite",
            });
        }
        if !(o.weight.is_finite() && o.weight >= 0.0) {
            return Err(DcdhError::InvalidRow {
                row: i,
                reason: "the weight must be finite and not negative",
            });
        }
        if o.outcome.is_some_and(|y| !y.is_finite()) || o.treatment.is_some_and(|d| !d.is_finite())
        {
            return Err(DcdhError::InvalidRow {
                row: i,
                reason: "an outcome or treatment is not finite; give a missing value as None",
            });
        }
        if clustered && o.cluster.is_none() {
            return Err(DcdhError::InvalidRow {
                row: i,
                reason: "with clusters, every row needs a cluster",
            });
        }
    }

    // Group-level constants: the stratum and the cluster.
    let mut stratum_of: HashMap<i64, i64> = HashMap::new();
    let mut cluster_of: HashMap<i64, i64> = HashMap::new();
    for o in observations {
        if *stratum_of.entry(o.group).or_insert(o.stratum) != o.stratum {
            return Err(DcdhError::VaryingStratum { group: o.group });
        }
        if let Some(c) = o.cluster
            && *cluster_of.entry(o.group).or_insert(c) != c
        {
            return Err(DcdhError::ClusterNotNested { group: o.group });
        }
    }

    // Groups whose treatment or outcome is never observed are dropped.
    let mut seen_d: HashMap<i64, bool> = HashMap::new();
    let mut seen_y: HashMap<i64, bool> = HashMap::new();
    for o in observations {
        *seen_d.entry(o.group).or_insert(false) |= o.treatment.is_some();
        *seen_y.entry(o.group).or_insert(false) |= o.outcome.is_some();
    }
    let keep = |o: &DcdhObservation| seen_d[&o.group] && seen_y[&o.group];

    // Period ranks over the rows that remain.
    let mut times: Vec<i64> = observations
        .iter()
        .filter(|o| keep(o))
        .map(|o| o.time)
        .collect();
    times.sort_unstable();
    times.dedup();
    if times.is_empty() {
        return Err(DcdhError::NoEffect);
    }
    let rank_of: HashMap<i64, i64> = times
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, i64::try_from(i).expect("rank fits i64") + 1))
        .collect();

    let mut ids: Vec<i64> = observations
        .iter()
        .filter(|o| keep(o))
        .map(|o| o.group)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let pos_of: HashMap<i64, usize> = ids.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let ng = ids.len();

    let mut rows: Vec<Row> = observations
        .iter()
        .filter(|o| keep(o))
        .map(|o| Row {
            group: pos_of[&o.group],
            rank: rank_of[&o.time],
            y: o.outcome.unwrap_or(NA),
            d: o.treatment.unwrap_or(NA),
            weight: o.weight,
            controls: o.controls.clone(),
        })
        .collect();
    rows.sort_by_key(|r| (r.group, r.rank));
    for pair in rows.windows(2) {
        if pair[0].group == pair[1].group && pair[0].rank == pair[1].rank {
            return Err(DcdhError::DuplicateCell {
                group: ids[pair[0].group],
                time: times[usize::try_from(pair[0].rank - 1).expect("rank is positive")],
            });
        }
    }

    // Per group: first and last period with a treatment, first period with an
    // outcome, and the first period with a missing treatment after that.
    let mut min_d = vec![i64::MAX; ng];
    let mut max_d = vec![i64::MIN; ng];
    let mut min_y = vec![i64::MAX; ng];
    for r in &rows {
        if !r.d.is_nan() {
            min_d[r.group] = min_d[r.group].min(r.rank);
            max_d[r.group] = max_d[r.group].max(r.rank);
        }
        if !r.y.is_nan() {
            min_y[r.group] = min_y[r.group].min(r.rank);
        }
    }
    let mut min_d_miss_after_y: Vec<Option<i64>> = vec![None; ng];
    for r in &rows {
        if r.d.is_nan() && r.rank >= min_y[r.group] {
            let slot = &mut min_d_miss_after_y[r.group];
            *slot = Some(slot.map_or(r.rank, |m| m.min(r.rank)));
        }
    }

    // The baseline treatment: the first observed one.
    let mut d_sq = vec![NA; ng];
    for r in &rows {
        if r.rank == min_d[r.group] {
            d_sq[r.group] = r.d;
        }
    }

    // A group whose treatment has been both above and below its baseline
    // loses its rows from the period when the second of the two first happens.
    if !config.dont_drop_larger_lower {
        let mut up = vec![false; ng];
        let mut down = vec![false; ng];
        rows.retain(|r| {
            if !r.d.is_nan() {
                let diff = r.d - d_sq[r.group];
                up[r.group] |= diff > 0.0;
                down[r.group] |= diff < 0.0;
            }
            !(up[r.group] && down[r.group])
        });
    }

    // Whether the treatment has changed by each row, and the first period of
    // a change, 0 for none.
    let mut ever: Vec<bool> = Vec::with_capacity(rows.len());
    let mut f_g = vec![0_i64; ng];
    {
        let mut changed = vec![false; ng];
        for r in &rows {
            let g = r.group;
            let now = !r.d.is_nan() && (r.d - d_sq[g]).abs() > 0.0;
            if now && !changed[g] {
                f_g[g] = r.rank;
            }
            changed[g] |= now;
            ever.push(changed[g]);
        }
    }

    // Baseline levels, and baseline-by-stratum cells whose groups all share
    // one first-switch date, are dropped. R takes the standard deviation of
    // F_g over the rows of the cell, so a cell of one row goes too.
    let stratum: Vec<i64> = ids.iter().map(|g| stratum_of[g]).collect();
    let base_key = |g: usize| (ordered(d_sq[g]), stratum[g]);
    let mut cell_f: HashMap<(i64, i64), (i64, bool, usize)> = HashMap::new();
    for r in &rows {
        let e = cell_f
            .entry(base_key(r.group))
            .or_insert((f_g[r.group], false, 0));
        e.1 |= e.0 != f_g[r.group];
        e.2 += 1;
    }
    let keep_row: Vec<bool> = rows
        .iter()
        .map(|r| {
            let e = cell_f[&base_key(r.group)];
            e.1 && e.2 > 1
        })
        .collect();
    let mut kept_rows = Vec::with_capacity(rows.len());
    let mut kept_ever = Vec::with_capacity(rows.len());
    for ((r, e), k) in rows.into_iter().zip(ever).zip(keep_row) {
        if k {
            kept_rows.push(r);
            kept_ever.push(e);
        }
    }
    let mut rows = kept_rows;
    let mut ever = kept_ever;
    let mut counted = vec![false; ng];
    for r in &rows {
        counted[r.group] = true;
    }
    let n_groups = counted.iter().filter(|&&c| c).count();
    if n_groups == 0 {
        return Err(DcdhError::NoEffect);
    }
    let counted_ids: Vec<i64> = (0..ng).filter(|&g| counted[g]).map(|g| ids[g]).collect();

    // Periods without a group of the same baseline and stratum whose
    // treatment has not changed are dropped for that baseline and stratum.
    let mut has_control: HashMap<(i64, i64, i64), bool> = HashMap::new();
    for (r, &e) in rows.iter().zip(&ever) {
        let (a, b) = base_key(r.group);
        *has_control.entry((a, b, r.rank)).or_insert(false) |= !e;
    }
    let keep_row: Vec<bool> = rows
        .iter()
        .map(|r| {
            let (a, b) = base_key(r.group);
            has_control[&(a, b, r.rank)]
        })
        .collect();
    let mut kept_rows = Vec::with_capacity(rows.len());
    let mut kept_ever = Vec::with_capacity(rows.len());
    for ((r, e), k) in rows.into_iter().zip(ever).zip(keep_row) {
        if k {
            kept_rows.push(r);
            kept_ever.push(e);
        }
    }
    rows = kept_rows;
    ever = kept_ever;
    let t_min = rows
        .iter()
        .map(|r| r.rank)
        .min()
        .ok_or(DcdhError::NoEffect)?;
    let t_max = rows
        .iter()
        .map(|r| r.rank)
        .max()
        .ok_or(DcdhError::NoEffect)?;
    {
        let mut present = vec![false; usize::try_from(t_max - t_min + 1).expect("periods")];
        for r in &rows {
            present[usize::try_from(r.rank - t_min).expect("rank")] = true;
        }
        if present.iter().any(|&p| !p) {
            return Err(DcdhError::EmptyPeriod);
        }
    }
    let never_f = t_max + 1;
    for f in &mut f_g {
        if *f == 0 {
            *f = never_f;
        }
    }

    // Missing treatments and outcomes.
    if config.drop_if_d_miss_before_first_switch {
        for r in &mut rows {
            if let Some(m) = min_d_miss_after_y[r.group]
                && m < f_g[r.group]
                && r.rank >= m
            {
                r.y = NA;
            }
        }
    }
    let mut last_before = vec![i64::MIN; ng];
    for r in &rows {
        if r.rank < f_g[r.group] && !r.d.is_nan() {
            last_before[r.group] = last_before[r.group].max(r.rank);
        }
    }
    for r in &mut rows {
        let g = r.group;
        if r.rank < min_d[g] {
            r.y = NA;
        }
        if f_g[g] < never_f && r.d.is_nan() && r.rank < last_before[g] && r.rank > min_d[g] {
            r.d = d_sq[g];
        }
        if f_g[g] < never_f && r.rank > last_before[g] && last_before[g] < f_g[g] - 1 {
            r.y = NA;
        }
    }
    let mut trunc: Vec<Option<i64>> = vec![None; ng];
    for g in 0..ng {
        if f_g[g] < never_f && last_before[g] < f_g[g] - 1 {
            trunc[g] = Some(last_before[g] + 1);
            f_g[g] = never_f;
        }
    }
    let mut d_at_f = vec![NA; ng];
    for r in &rows {
        if r.rank == f_g[r.group] {
            d_at_f[r.group] = r.d;
        }
    }
    for r in &mut rows {
        let g = r.group;
        if f_g[g] < never_f && r.d.is_nan() && r.rank > f_g[g] && last_before[g] == f_g[g] - 1 {
            r.d = d_at_f[g];
        }
        if f_g[g] == never_f && r.d.is_nan() && r.rank > min_d[g] && r.rank < max_d[g] {
            r.d = d_sq[g];
        }
        if f_g[g] == never_f && r.rank > max_d[g] {
            r.y = NA;
        }
    }
    for g in 0..ng {
        if f_g[g] == never_f {
            trunc[g] = Some(max_d[g] + 1);
        }
    }

    // The balanced grid of the groups that still have rows.
    let mut in_grid = vec![false; ng];
    for r in &rows {
        in_grid[r.group] = true;
    }
    let nt = usize::try_from(t_max - t_min + 1).expect("periods");
    let grid_groups: Vec<usize> = (0..ng).filter(|&g| in_grid[g]).collect();
    let mut grid_pos = vec![usize::MAX; ng];
    for (i, &g) in grid_groups.iter().enumerate() {
        grid_pos[g] = i;
    }
    let gn = grid_groups.len();
    let cells = gn * nt;
    let mut y = vec![NA; cells];
    let mut d = vec![NA; cells];
    let mut weight = vec![NA; cells];
    let mut ever_change = vec![NA; cells];
    let mut controls = vec![vec![NA; cells]; n_controls];
    let mut filled = vec![true; cells];
    for (r, &e) in rows.iter().zip(&ever) {
        let c = grid_pos[r.group] * nt + usize::try_from(r.rank - t_min).expect("rank");
        y[c] = r.y;
        d[c] = r.d;
        weight[c] = r.weight;
        ever_change[c] = if e { 1.0 } else { 0.0 };
        for (k, x) in r.controls.iter().enumerate() {
            controls[k][c] = *x;
        }
        filled[c] = false;
    }
    let n_gt: Vec<f64> = (0..cells)
        .map(|c| {
            if y[c].is_nan() || d[c].is_nan() {
                0.0
            } else {
                weight[c]
            }
        })
        .collect();

    // T_g: the last period in which the group's baseline-and-stratum cell
    // still has a control, from the truncated first-switch dates. A filled
    // cell enters with F_g itself, as in R, where the balancing leaves the
    // truncation date missing.
    let gd_sq: Vec<f64> = grid_groups.iter().map(|&g| d_sq[g]).collect();
    let gstratum: Vec<i64> = grid_groups.iter().map(|&g| stratum[g]).collect();
    let gf: Vec<i64> = grid_groups.iter().map(|&g| f_g[g]).collect();
    let (base_id, n_base) = index_of(
        &(0..gn)
            .map(|i| (ordered(gd_sq[i]), gstratum[i]))
            .collect::<Vec<_>>(),
    );
    let mut base_max = vec![i64::MIN; n_base];
    for i in 0..gn {
        let g = grid_groups[i];
        let truncated = trunc[g].map_or(f_g[g], |t| f_g[g].min(t));
        for tp in 0..nt {
            let v = if filled[i * nt + tp] {
                f_g[g]
            } else {
                truncated
            };
            base_max[base_id[i]] = base_max[base_id[i]].max(v);
        }
    }
    let t_g: Vec<i64> = (0..gn).map(|i| base_max[base_id[i]] - 1).collect();

    // S_g from the mean treatment between F_g and T_g. A switcher whose mean
    // equals its baseline is neither a switcher in nor out and is dropped.
    let mut s_g = vec![NA; gn];
    let mut drop = vec![false; gn];
    for i in 0..gn {
        let (mut sum, mut count) = (0.0, 0.0);
        for tp in 0..nt {
            let t = t_min + i64::try_from(tp).expect("period");
            let c = i * nt + tp;
            if t >= gf[i] && t <= t_g[i] && !d[c].is_nan() {
                sum += d[c];
                count += 1.0;
            }
        }
        let avg = sum / count;
        if !avg.is_nan() && avg == gd_sq[i] && gf[i] != t_g[i] + 1 {
            drop[i] = true;
        }
        if gf[i] != never_f && !avg.is_nan() {
            s_g[i] = if avg > gd_sq[i] { 1.0 } else { 0.0 };
        }
    }
    if drop.iter().any(|&x| x) {
        let keep_idx: Vec<usize> = (0..gn).filter(|&i| !drop[i]).collect();
        return Ok(finish(
            &Subset {
                keep: &keep_idx,
                nt,
                y: &y,
                d: &d,
                n_gt: &n_gt,
                weight: &weight,
                ever_change: &ever_change,
                controls: &controls,
            },
            &Groups {
                ids: grid_groups.iter().map(|&g| ids[g]).collect(),
                d_sq: gd_sq,
                stratum: gstratum,
                f_g: gf,
                t_g,
                s_g,
                cluster: grid_groups
                    .iter()
                    .map(|&g| cluster_of.get(&ids[g]).copied())
                    .collect(),
            },
            counted_ids,
            n_groups,
            t_min,
            t_max,
            clustered,
        ));
    }
    let all: Vec<usize> = (0..gn).collect();
    Ok(finish(
        &Subset {
            keep: &all,
            nt,
            y: &y,
            d: &d,
            n_gt: &n_gt,
            weight: &weight,
            ever_change: &ever_change,
            controls: &controls,
        },
        &Groups {
            ids: grid_groups.iter().map(|&g| ids[g]).collect(),
            d_sq: gd_sq,
            stratum: gstratum,
            f_g: gf,
            t_g,
            s_g,
            cluster: grid_groups
                .iter()
                .map(|&g| cluster_of.get(&ids[g]).copied())
                .collect(),
        },
        counted_ids,
        n_groups,
        t_min,
        t_max,
        clustered,
    ))
}

struct Subset<'a> {
    keep: &'a [usize],
    nt: usize,
    y: &'a [f64],
    d: &'a [f64],
    n_gt: &'a [f64],
    weight: &'a [f64],
    ever_change: &'a [f64],
    controls: &'a [Vec<f64>],
}

struct Groups {
    ids: Vec<i64>,
    d_sq: Vec<f64>,
    stratum: Vec<i64>,
    f_g: Vec<i64>,
    t_g: Vec<i64>,
    s_g: Vec<f64>,
    cluster: Vec<Option<i64>>,
}

fn take(values: &[f64], keep: &[usize], nt: usize) -> Vec<f64> {
    keep.iter()
        .flat_map(|&i| values[i * nt..(i + 1) * nt].iter().copied())
        .collect()
}

/// The group variables that follow S_g, on the groups that remain.
fn finish(
    sub: &Subset<'_>,
    groups: &Groups,
    counted_ids: Vec<i64>,
    n_groups: usize,
    t_min: i64,
    t_max: i64,
    clustered: bool,
) -> Panel {
    let &Subset {
        keep,
        nt,
        y,
        d,
        n_gt,
        weight,
        ever_change,
        controls,
    } = sub;
    let y = take(y, keep, nt);
    let d = take(d, keep, nt);
    let n_gt = take(n_gt, keep, nt);
    let weight = take(weight, keep, nt);
    let ever_change = take(ever_change, keep, nt);
    let controls: Vec<Vec<f64>> = controls.iter().map(|x| take(x, keep, nt)).collect();
    let pick = |v: &[i64]| keep.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let pickf = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let group_ids = pick(&groups.ids);
    let d_sq = pickf(&groups.d_sq);
    let stratum = pick(&groups.stratum);
    let f_g = pick(&groups.f_g);
    let t_g = pick(&groups.t_g);
    let s_g = pickf(&groups.s_g);
    let gn = group_ids.len();
    let never_f = t_max + 1;

    // D at F_g; the baseline for a group that never switches.
    let d_fg: Vec<f64> = (0..gn)
        .map(|i| {
            let tp = f_g[i] - t_min;
            if f_g[i] <= t_max && tp >= 0 {
                let v = d[i * nt + usize::try_from(tp).expect("period")];
                if v.is_nan() && f_g[i] == never_f {
                    d_sq[i]
                } else {
                    v
                }
            } else {
                d_sq[i]
            }
        })
        .collect();
    let l_g: Vec<i64> = (0..gn).map(|i| t_g[i] - f_g[i] + 1).collect();
    let l_g_placebo: Vec<Option<i64>> = (0..gn)
        .map(|i| (f_g[i] >= 3).then(|| l_g[i].min(f_g[i] - 2)))
        .collect();
    let (base_id, n_base) = index_of(
        &(0..gn)
            .map(|i| (ordered(d_sq[i]), stratum[i]))
            .collect::<Vec<_>>(),
    );
    let (level, n_level) = index_of(&d_sq.iter().map(|&v| ordered(v)).collect::<Vec<_>>());
    let (cohort_id, n_cohort) = index_of(
        &(0..gn)
            .map(|i| (ordered(d_sq[i]), f_g[i], ordered(d_fg[i]), stratum[i]))
            .collect::<Vec<_>>(),
    );
    let (cluster, n_cluster) = if clustered {
        let labels: Vec<i64> = keep
            .iter()
            .map(|&i| groups.cluster[i].expect("every row has a cluster"))
            .collect();
        let (idx, n) = index_of(&labels);
        (Some(idx), n)
    } else {
        (None, 0)
    };
    Panel {
        group_ids,
        counted_ids,
        n_groups: n_groups as f64,
        t_min,
        t_max,
        nt,
        y,
        d,
        n_gt,
        weight,
        ever_change,
        controls,
        d_sq,
        f_g,
        t_g,
        s_g,
        l_g,
        l_g_placebo,
        base_id,
        n_base,
        level,
        n_level,
        cohort_id,
        n_cohort,
        cluster,
        n_cluster,
    }
}
