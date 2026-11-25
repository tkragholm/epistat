//! The Goodman-Bacon decomposition of the static two-way fixed-effects
//! estimate into its 2x2 comparisons, as `bacondecomp::bacon` 0.1.1 computes
//! it.
//!
//! # Without covariates
//!
//! On a balanced panel where each unit's treatment switches on once and stays
//! on, the static TWFE coefficient is a weighted mean of the 2x2
//! difference-in-differences estimates between timing groups \[1\]. For an
//! ordered pair of groups (a "treated" group `k`, a comparison group `l`), the
//! comparison is fitted on the rows of the two groups in a window:
//!
//! ```text
//! l never treated            all periods          treated vs untreated
//! k < l                      periods before l     earlier vs later treated
//! k > l                      periods from l on    later vs earlier treated
//! l treated from the start   all periods          later vs always treated
//! ```
//!
//! The estimate is the TWFE coefficient of the treatment on that window,
//! fitted by [`epistat_glm::feols`]. With `n_k`, `n_l` the rows of the two
//! groups in the window, `n_kl = n_k / (n_k + n_l)` and `D_k`, `D_l` their
//! shares of treated rows there, the weights of `bacondecomp` are
//!
//! ```text
//! treated vs untreated   (n_k + n_l)^2 n_kl (1 - n_kl) D_k (1 - D_k)
//! k < l                  ((n_k + n_l)(1 - D_l))^2 n_kl (1 - n_kl) (D_k - D_l)/(1 - D_l) (1 - D_k)/(1 - D_l)
//! k > l                  ((n_k + n_l) D_k)^2 n_kl (1 - n_kl) (D_l / D_k) (D_k - D_l)/D_k
//! ```
//!
//! where for `k > l` the roles are those of `bacondecomp`: `n_k`, `D_k` belong
//! to the earlier group `l` and `n_l`, `D_l` to the later group `k`. The
//! weights are scaled to sum to one, and the weighted mean of the estimates is
//! then the static TWFE coefficient \[1, theorem 1\]. A group treated in the
//! first period is never the treated group of a comparison.
//!
//! # With covariates
//!
//! With time-varying covariates `X`, `bacondecomp` follows \[1, section 5\]:
//! the treatment is regressed on `X` and the unit and period effects, with
//! fitted values `p` and residuals `d`. With `d_i`, `d_t`, `d_kt`, `d_k` and
//! `d_bar` the means of `d` by unit, period, group and period, group, and
//! overall,
//!
//! ```text
//! d~_it  = d - d_i - d_t + d_bar
//! d~_ikt = d - d_i - (d_kt - d_k)
//! Omega  = var(d~_ikt) / var(d~_it),       beta_w = cov(y, d~_ikt) / var(d~_ikt)
//! ```
//!
//! give the within-group component and its weight (population variances).
//! The between-group comparisons are the unordered pairs of groups `k < l`
//! (the never treated last), each on all periods of the two groups. With `D~`
//! the treatment and `p~` the group-by-period mean of `p` after the unit and
//! period effects are removed in the pair, `g~` the same for the group-by-period
//! means of each covariate, `R^2` and `p^` the fit of `D~` on the `g~` with an
//! intercept, `dp = p^ - p~`, `B_D` the TWFE coefficient of the treatment in
//! `y` on the treatment and the group means of the covariates, and `B_b` that
//! of `dp` in `y` on `dp`,
//!
//! ```text
//! s_kl     = N^2 ((1 - R^2) var(D~) + var(dp))
//! beta_kl  = ((1 - R^2) var(D~) B_D + var(dp) B_b) / ((1 - R^2) var(D~) + var(dp))
//! ```
//!
//! with `N` the rows of the pair, and the `s_kl` are scaled to sum to one. In
//! a pair where the treatment is absorbed by the effects (a group treated
//! from the first period against the never treated), `var(D~)` is zero and
//! the estimate is `B_b`.
//! These are the formulas of `bacondecomp` 0.1.1, which does not check that
//! `Omega beta_w + (1 - Omega) sum s_kl beta_kl` reproduces the TWFE
//! coefficient; this module does not claim it either.
//!
//! # References
//!
//! 1. Goodman-Bacon A. Difference-in-differences with variation in treatment
//!    timing. J Econom. 2021;225(2):254-77.
//!    doi:10.1016/j.jeconom.2021.03.014
//! 2. Flack E, Jee E. bacondecomp: Goodman-Bacon decomposition. R package
//!    version 0.1.1, 2020.

use std::collections::{BTreeMap, BTreeSet};

use epistat_glm::{Absorb, AbsorbError, Feols, FixedEffects, Singletons};
use epistat_linalg::{Mat, MatRef};

use crate::methods::staggered::StaggeredPanel;

/// The rows of the decomposition, with the treatment of each row.
#[derive(Clone, Copy, Debug)]
struct Rows<'a> {
    unit: &'a [u32],
    time: &'a [i32],
    treated: &'a [bool],
    outcome: &'a [f64],
    controls: Option<MatRef<'a, f64>>,
}

/// The kind of a 2x2 comparison, with the labels of `bacondecomp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaconKind {
    /// Against the never treated.
    TreatedVsUntreated,
    /// The earlier group against the later one before it is treated.
    EarlierVsLaterTreated,
    /// The later group against the earlier one after it is treated.
    LaterVsEarlierTreated,
    /// Against the group treated from the first period.
    LaterVsAlwaysTreated,
    /// With covariates: two groups that are both treated at some point.
    BothTreated,
}

/// One 2x2 comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct BaconComparison {
    /// The first treated period of the treated group.
    pub treated: i32,
    /// The first treated period of the comparison group, `None` for never.
    pub untreated: Option<i32>,
    /// The kind.
    pub kind: BaconKind,
    /// The 2x2 estimate, `None` when the treatment is collinear in the window.
    pub estimate: Option<f64>,
    /// The weight, scaled with the others to sum to one.
    pub weight: f64,
}

/// The within-group component of the decomposition with covariates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaconWithin {
    /// The weight `Omega` of the within-group estimate.
    pub omega: f64,
    /// The within-group estimate `beta_w`.
    pub estimate: f64,
}

/// The decomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct BaconDecomposition {
    /// The comparisons, by treated group and then comparison group, the
    /// never treated last.
    pub comparisons: Vec<BaconComparison>,
    /// With covariates, the within-group component.
    pub within: Option<BaconWithin>,
}

/// Why the decomposition could not be computed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BaconError {
    /// An input has the wrong length.
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        /// The input.
        what: &'static str,
        /// The length it needed.
        expected: usize,
        /// The length it had.
        found: usize,
    },
    /// Some unit lacks a period, or has one twice.
    #[error("the panel is not balanced")]
    Unbalanced,
    /// A unit has rows with different first treated periods.
    #[error("unit {unit} has rows with different first treated periods")]
    Cohort {
        /// The unit.
        unit: u32,
    },
    /// Prior weights were given; the decomposition is unweighted, as in
    /// `bacondecomp`.
    #[error("the decomposition takes no weights")]
    Weights,
    /// A value is not finite.
    #[error("a value of {0} is not finite")]
    NotFinite(&'static str),
    /// Fewer than two timing groups.
    #[error("the decomposition needs at least two timing groups")]
    Groups,
    /// A regression failed.
    #[error(transparent)]
    Fit(#[from] AbsorbError),
}

/// The tolerance of the demeaning: on a balanced panel the two-way
/// projections are exact after one sweep, so this only stops the iteration.
const ABSORB: Absorb = Absorb {
    tolerance: 1e-13,
    max_iterations: 10_000,
};

/// The rows of one regression with unit and period effects.
struct Window {
    unit: Vec<u32>,
    time: Vec<u32>,
}

impl Window {
    fn fixed_effects(&self) -> Result<FixedEffects, AbsorbError> {
        FixedEffects::new(&[&self.unit, &self.time], None, Singletons::Keep)
    }
}

/// The coefficient of the first column of `x` in the TWFE regression of `y`.
fn first_coefficient(
    fe: &FixedEffects,
    y: &[f64],
    x: MatRef<'_, f64>,
) -> Result<Option<f64>, AbsorbError> {
    let fit = Feols { absorb: ABSORB }.fit(y, x, fe)?;
    Ok(fit.coefficients[0])
}

/// The residuals of the columns on the unit and period effects.
fn residuals(fe: &FixedEffects, columns: &[&[f64]]) -> Result<Vec<Vec<f64>>, AbsorbError> {
    let out = fe.demean(columns, &ABSORB)?;
    Ok((0..columns.len())
        .map(|j| out.columns.col_as_slice(j).to_vec())
        .collect())
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// The population variance, as `var(x) (N - 1) / N`.
fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64
}

fn covariance(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / a.len() as f64
}

/// The means of `v` within the groups of `key`, spread back to the rows.
fn group_means<K: Ord + Copy>(v: &[f64], key: impl Fn(usize) -> K) -> Vec<f64> {
    let mut sums: BTreeMap<K, (f64, usize)> = BTreeMap::new();
    for (i, x) in v.iter().enumerate() {
        let e = sums.entry(key(i)).or_insert((0.0, 0));
        e.0 += x;
        e.1 += 1;
    }
    (0..v.len())
        .map(|i| {
            let (s, c) = sums[&key(i)];
            s / c as f64
        })
        .collect()
}

/// The Goodman-Bacon decomposition of `panel`, as `bacondecomp::bacon`. The
/// panel must be balanced and unweighted; its covariates are the time-varying
/// controls of the decomposition. A unit whose first treated period lies after
/// the last period is never treated, as it is to `bacondecomp`, which sees only
/// the treatment of each row.
///
/// # Errors
///
/// [`BaconError`] for inputs of the wrong length or not finite, weights, a
/// panel that is not balanced, a first treated period that varies within a
/// unit, fewer than two timing groups, and the errors of the regressions.
pub fn bacon_decomposition(panel: &StaggeredPanel<'_>) -> Result<BaconDecomposition, BaconError> {
    let n = panel.unit.len();
    for (what, found) in [
        ("time", panel.time.len()),
        ("first_treated", panel.first_treated.len()),
        ("outcome", panel.outcome.len()),
        ("covariates", panel.covariates.map_or(n, |x| x.nrows())),
    ] {
        if found != n {
            return Err(BaconError::Length {
                what,
                expected: n,
                found,
            });
        }
    }
    if panel.weights.is_some() {
        return Err(BaconError::Weights);
    }
    let mut first: BTreeMap<u32, Option<i32>> = BTreeMap::new();
    for (&u, &g) in panel.unit.iter().zip(panel.first_treated) {
        if *first.entry(u).or_insert(g) != g {
            return Err(BaconError::Cohort { unit: u });
        }
    }
    let treated: Vec<bool> = (0..n)
        .map(|i| panel.first_treated[i].is_some_and(|g| panel.time[i] >= g))
        .collect();
    decompose(&Rows {
        unit: panel.unit,
        time: panel.time,
        treated: &treated,
        outcome: panel.outcome,
        controls: panel.covariates,
    })
}

/// The decomposition of rows whose treatment stays on once it starts.
#[expect(
    clippy::similar_names,
    reason = "n_k, n_l, d_k and d_l are the notation of bacondecomp"
)]
fn decompose(panel: &Rows<'_>) -> Result<BaconDecomposition, BaconError> {
    let n = panel.unit.len();
    if panel.outcome.iter().any(|v| !v.is_finite()) {
        return Err(BaconError::NotFinite("the outcome"));
    }
    if let Some(x) = panel.controls
        && (0..x.ncols()).any(|j| (0..n).any(|i| !x[(i, j)].is_finite()))
    {
        return Err(BaconError::NotFinite("the controls"));
    }
    let periods: BTreeSet<i32> = panel.time.iter().copied().collect();
    let units: BTreeSet<u32> = panel.unit.iter().copied().collect();
    let mut seen: BTreeSet<(u32, i32)> = BTreeSet::new();
    for (&u, &t) in panel.unit.iter().zip(panel.time) {
        if !seen.insert((u, t)) {
            return Err(BaconError::Unbalanced);
        }
    }
    if seen.len() != units.len() * periods.len() {
        return Err(BaconError::Unbalanced);
    }
    let t_min = *periods.first().ok_or(BaconError::Groups)?;
    // The first treated period of each unit, as bacondecomp finds it from
    // the treated rows.
    let mut cohort_of: BTreeMap<u32, Option<i32>> = units.iter().map(|&u| (u, None)).collect();
    for i in 0..n {
        if panel.treated[i] {
            let c = cohort_of
                .get_mut(&panel.unit[i])
                .ok_or(BaconError::Groups)?;
            *c = Some(c.map_or(panel.time[i], |g: i32| g.min(panel.time[i])));
        }
    }
    let cohort: Vec<Option<i32>> = (0..n).map(|i| cohort_of[&panel.unit[i]]).collect();
    // The groups, the never treated last.
    let mut groups: Vec<Option<i32>> = cohort_of
        .values()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    groups.sort_by_key(|g| g.map_or((1, 0), |g| (0, g)));
    if groups.len() < 2 {
        return Err(BaconError::Groups);
    }
    let window = |rows: &[usize]| Window {
        unit: rows.iter().map(|&i| panel.unit[i]).collect(),
        time: rows
            .iter()
            .map(|&i| (panel.time[i] - t_min).unsigned_abs())
            .collect(),
    };
    let d: Vec<f64> = panel
        .treated
        .iter()
        .map(|&b| f64::from(u8::from(b)))
        .collect();

    match panel.controls {
        None => {
            let mut comparisons = Vec::new();
            for &k in &groups {
                let Some(k) = k else { continue };
                if k == t_min {
                    continue;
                }
                for &l in &groups {
                    if l == Some(k) {
                        continue;
                    }
                    let later = l.is_none_or(|l| k < l);
                    let rows: Vec<usize> = (0..n)
                        .filter(|&i| cohort[i] == Some(k) || cohort[i] == l)
                        .filter(|&i| match l {
                            Some(l) if k < l => panel.time[i] < l,
                            Some(l) => panel.time[i] >= l,
                            None => true,
                        })
                        .collect();
                    let share = |g: Option<i32>| {
                        let mine: Vec<usize> =
                            rows.iter().copied().filter(|&i| cohort[i] == g).collect();
                        let treated = mine.iter().filter(|&&i| panel.treated[i]).count();
                        (mine.len() as f64, treated as f64 / mine.len() as f64)
                    };
                    let (n_k_rows, d_k) = share(Some(k));
                    let (n_l_rows, d_l) = share(l);
                    let (weight, kind) = match l {
                        None => {
                            let s = n_k_rows / (n_k_rows + n_l_rows);
                            let v = s * (1.0 - s) * d_k * (1.0 - d_k);
                            (
                                (n_k_rows + n_l_rows).powi(2) * v,
                                BaconKind::TreatedVsUntreated,
                            )
                        }
                        Some(_) if later => {
                            let s = n_k_rows / (n_k_rows + n_l_rows);
                            let v = s * (1.0 - s) * (d_k - d_l) / (1.0 - d_l) * (1.0 - d_k)
                                / (1.0 - d_l);
                            (
                                ((n_k_rows + n_l_rows) * (1.0 - d_l)).powi(2) * v,
                                BaconKind::EarlierVsLaterTreated,
                            )
                        }
                        Some(l) => {
                            // bacondecomp's names: k is the earlier group l.
                            let (n_e, d_e, n_a, d_a) = (n_l_rows, d_l, n_k_rows, d_k);
                            let s = n_e / (n_e + n_a);
                            let v = s * (1.0 - s) * (d_a / d_e) * (d_e - d_a) / d_e;
                            let kind = if l == t_min {
                                BaconKind::LaterVsAlwaysTreated
                            } else {
                                BaconKind::LaterVsEarlierTreated
                            };
                            (((n_e + n_a) * d_e).powi(2) * v, kind)
                        }
                    };
                    let w = window(&rows);
                    let fe = w.fixed_effects()?;
                    let y: Vec<f64> = rows.iter().map(|&i| panel.outcome[i]).collect();
                    let x = Mat::from_fn(rows.len(), 1, |r, _| d[rows[r]]);
                    let estimate = first_coefficient(&fe, &y, x.as_ref())?;
                    comparisons.push(BaconComparison {
                        treated: k,
                        untreated: l,
                        kind,
                        estimate,
                        weight,
                    });
                }
            }
            scale(&mut comparisons);
            Ok(BaconDecomposition {
                comparisons,
                within: None,
            })
        }
        Some(controls) => controlled(panel, controls, &cohort, &groups, &d, t_min, &window),
    }
}

/// Scales the weights to sum to one.
fn scale(comparisons: &mut [BaconComparison]) {
    let total: f64 = comparisons.iter().map(|c| c.weight).sum();
    for c in comparisons {
        c.weight /= total;
    }
}

/// The decomposition with covariates.
#[expect(
    clippy::similar_names,
    reason = "d_i, d_t, d_it, d_kt and d_ikt are the notation of the paper"
)]
fn controlled(
    panel: &Rows<'_>,
    controls: MatRef<'_, f64>,
    cohort: &[Option<i32>],
    groups: &[Option<i32>],
    d: &[f64],
    t_min: i32,
    window: &dyn Fn(&[usize]) -> Window,
) -> Result<BaconDecomposition, BaconError> {
    let n = panel.unit.len();
    let q = controls.ncols();
    let all: Vec<usize> = (0..n).collect();
    let fe_all = window(&all).fixed_effects()?;
    // The treatment on the covariates and both effects.
    let fit = Feols { absorb: ABSORB }.fit(d, controls, &fe_all)?;
    let resid = fit.residuals.clone();
    let p: Vec<f64> = d.iter().zip(&resid).map(|(a, b)| a - b).collect();
    let key_g = |i: usize| cohort[i].unwrap_or(i32::MAX);
    let d_i = group_means(&resid, |i| panel.unit[i]);
    let d_t = group_means(&resid, |i| panel.time[i]);
    let d_bar = mean(&resid);
    let d_kt = group_means(&resid, |i| (key_g(i), panel.time[i]));
    let d_k = group_means(&resid, key_g);
    let d_it: Vec<f64> = (0..n).map(|i| resid[i] - d_i[i] - d_t[i] + d_bar).collect();
    let d_ikt: Vec<f64> = (0..n)
        .map(|i| resid[i] - d_i[i] - (d_kt[i] - d_k[i]))
        .collect();
    let within = BaconWithin {
        omega: variance(&d_ikt) / variance(&d_it),
        estimate: covariance(panel.outcome, &d_ikt) / variance(&d_ikt),
    };
    // Group-by-period means of the covariates and of the fitted treatment.
    let g_x: Vec<Vec<f64>> = (0..q)
        .map(|j| {
            let column: Vec<f64> = (0..n).map(|i| controls[(i, j)]).collect();
            group_means(&column, |i| (key_g(i), panel.time[i]))
        })
        .collect();
    let g_p = group_means(&p, |i| (key_g(i), panel.time[i]));

    let mut comparisons = Vec::new();
    for (a, &k) in groups.iter().enumerate() {
        let Some(k) = k else { continue };
        for &l in &groups[a + 1..] {
            let rows: Vec<usize> = (0..n)
                .filter(|&i| cohort[i] == Some(k) || cohort[i] == l)
                .collect();
            let m = rows.len();
            let fe = window(&rows).fixed_effects()?;
            let take = |v: &[f64]| -> Vec<f64> { rows.iter().map(|&i| v[i]).collect() };
            let d_w = take(d);
            let mut columns: Vec<Vec<f64>> = vec![d_w.clone(), take(&g_p)];
            columns.extend(g_x.iter().map(|g| take(g)));
            let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
            let tilde = residuals(&fe, &refs)?;
            let (d_tilde, p_tilde) = (&tilde[0], &tilde[1]);
            let vd = variance(d_tilde);
            // D~ on the demeaned group means of the covariates, with an
            // intercept: one constant factor absorbs it.
            let constant = vec![0u32; m];
            let fe_one = FixedEffects::new(&[&constant], None, Singletons::Keep)?;
            let gx_tilde = Mat::from_fn(m, q, |r, j| tilde[2 + j][r]);
            let fit_p = Feols { absorb: ABSORB }.fit(d_tilde, gx_tilde.as_ref(), &fe_one)?;
            let pgj: Vec<f64> = d_tilde
                .iter()
                .zip(&fit_p.residuals)
                .map(|(a, b)| a - b)
                .collect();
            let rsq = 1.0 - fit_p.rss / (variance(d_tilde) * m as f64);
            let dp: Vec<f64> = pgj.iter().zip(p_tilde).map(|(a, b)| a - b).collect();
            let vdp = variance(&dp);
            let y = take(panel.outcome);
            let x_bd = Mat::from_fn(
                m,
                1 + q,
                |r, j| if j == 0 { d_w[r] } else { g_x[j - 1][rows[r]] },
            );
            let bd = first_coefficient(&fe, &y, x_bd.as_ref())?;
            let x_bb = Mat::from_fn(m, 1, |r, _| dp[r]);
            let bb = first_coefficient(&fe, &y, x_bb.as_ref())?;
            // When the treatment is absorbed by the unit and period effects
            // of the pair (a group treated throughout against the never
            // treated), var(D~) is zero in exact arithmetic and B_D has no
            // weight; R's lm keeps the column by dropping a dummy instead and
            // multiplies its coefficient by the zero variance.
            let a_part = if bd.is_some() { (1.0 - rsq) * vd } else { 0.0 };
            let weight = (m as f64).powi(2) * (a_part + vdp);
            // As in R, an aliased B_b leaves the estimate undefined.
            let estimate = match (bd, bb) {
                (Some(bd), Some(bb)) => Some((a_part * bd + vdp * bb) / (a_part + vdp)),
                (None, Some(bb)) => Some(bb),
                _ => None,
            };
            let kind = if l.is_none() {
                BaconKind::TreatedVsUntreated
            } else if k == t_min || l == Some(t_min) {
                BaconKind::LaterVsAlwaysTreated
            } else {
                BaconKind::BothTreated
            };
            comparisons.push(BaconComparison {
                treated: k,
                untreated: l,
                kind,
                estimate,
                weight,
            });
        }
    }
    scale(&mut comparisons);
    Ok(BaconDecomposition {
        comparisons,
        within: Some(within),
    })
}
