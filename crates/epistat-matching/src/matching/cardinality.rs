//! Cardinality matching and profile matching.
//!
//! Neither method pairs units. Each selects a subset of the units, by an integer
//! program with one binary variable per unit, so that the means of the covariates
//! in the selected subset satisfy balance constraints, and returns the subset as
//! weights of 0 and 1.
//!
//! - **Cardinality matching** (Zubizarreta, Paredes and Rosenbaum \[1\]) selects
//!   the largest subset in which the ratio of selected non-focal to selected focal
//!   units is `ratio`, and in which the mean of each covariate differs between
//!   the two groups by at most its tolerance. Not every focal unit need be
//!   selected, so the estimand is no longer the effect in the focal group.
//! - **Profile matching** (Cohn and Zubizarreta \[2\]) selects the largest subset
//!   of each group whose covariate means are within the tolerance of a fixed
//!   target. For the effect in the treated (`ratio = None`, `Estimand::Att`) the
//!   target is the mean of the treated, which are all kept, and the subset is of
//!   the controls. For the effect in the whole sample (`Estimand::Ate`) the
//!   target is the mean of the whole sample and both groups are cut down to it,
//!   to equal size ratios when `ratio` is given. The estimand is kept.
//!
//! The programs are those that `MatchIt` documents for
//! `matchit(method = "cardinality")`. Let `x_i ∈ {0, 1}` select unit `i`, `X_ic`
//! be its covariate `c`, standardised when asked, and `s_i` its sampling weight.
//!
//! ```text
//! cardinality:   max Σ x_i
//!                Σ_{focal} x_i = k,    Σ_{other} x_i = ratio · k
//!                |Σ_{focal} x_i X_ic - (1/ratio) Σ_{other} x_i X_ic| ≤ tol_c · k
//! profile, ATT:  max Σ_{other} x_i
//!                Σ_{other} s_i x_i = t,    |Σ_{other} s_i x_i X_ic - t μ_c| ≤ tol_c · t
//! profile, ATE:  max Σ x_i
//!                Σ_{g} s_i x_i = t_g,    |Σ_{g} s_i x_i X_ic - t_g μ_c| ≤ tol_c/2 · t_g
//!                (with ratio:  Σ_{focal} x_i = k,  Σ_{other} x_i = ratio · k)
//! ```
//!
//! where `μ_c` is the (sampling weighted) mean of `X_c` in the focal group for the
//! ATT and in the whole sample for the ATE, and `t`, `t_g` and `k` are continuous
//! variables that carry the sizes. A tolerance is a difference of means in
//! standard deviations when `std_tols` is true, the default. The standard
//! deviation is the sampling weighted one of the focal group for the ATT and the
//! ATC, and the root of the mean of the two group variances for the ATE, a binary
//! covariate using `p (1 - p)` in place of the sample variance.
//!
//! # Solver
//!
//! The programs have up to a million columns and a few dozen rows. They are
//! solved by [`epistat_optim::branch_and_bound`], an own bounded dual simplex
//! with Gomory cuts, local search and branch and bound by objective level, not
//! a general solver. Its limits are the node and time limits of
//! [`CardinalityRequest`]. A stratum whose search stops at a limit returns the
//! best subset found, [`StratumReport::proven_optimal`] is false, and
//! [`StratumReport::bound`] is the largest objective the search could not rule
//! out. `MatchIt`'s default time limit is 120 seconds, and this is the default
//! here.
//!
//! The columns of one binary program are dense: for 10^6 units and 30 rows the
//! matrix takes 240 MB.
//!
//! # Exact strata and pairing
//!
//! With `exact`, the program is solved within each stratum that holds both
//! groups. Units in other strata get weight 0. With `pair_on`, the selected units
//! of a stratum are then paired, optimal pair matching on the Mahalanobis distance
//! of the `pair_on` columns, with the pooled within-group covariance of the
//! non-discarded units. Each focal unit gets `ratio` others, and the selection does
//! not change. This needs a whole-number `ratio` and the effect in the treated or
//! in the whole sample. The distance matrix of a stratum is dense.
//!
//! # Weights
//!
//! A selected unit has weight 1. In each stratum the weights of the larger group
//! are scaled to the total of the smaller group (the focal group for profile
//! matching for the ATT), and then, over all strata, the weights of each group
//! are scaled to sum to the number of its units that have a weight, as
//! `matchit()` does. With one stratum the weights are 0 and 1 again. With
//! several, a group's weights are redistributed among the strata in proportion
//! to the other group's selected units.
//!
//! # Checked against R
//!
//! The sizes of the selected subsets equal `MatchIt`'s with HiGHS wherever HiGHS
//! proves its optimum, on `lalonde` and on simulated cohorts, and every balance
//! constraint holds
//! (`tests/r_cardinality.rs`). The subsets themselves can differ where the
//! optimum is not unique. The pairing is checked against `optmatch::pairmatch`
//! on its total distance.
//!
//! # References
//!
//! 1. Zubizarreta JR, Paredes RD, Rosenbaum PR. Matching for balance, pairing for
//!    heterogeneity, and the question of whether there is a mass of uninsured
//!    people in Chicago. Ann Appl Stat. 2014;8(1):204-231.
//! 2. Cohn ER, Zubizarreta JR. Profile matching for the generalization and
//!    personalization of causal inferences. Epidemiology. 2022;33(5):678-688.
//! 3. Niknam BA, Zubizarreta JR. Using cardinality matching to design balanced
//!    and representative samples for observational studies. JAMA.
//!    2022;327(2):173-174.

#![expect(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "indicator columns and unit weights are compared with 0 and 1 as given, and the \
              ratio is checked to be whole before it is cast"
)]

use super::quick::cholesky_lower;
use crate::types::Estimand;
use epistat_optim::branch_and_bound::{
    MipError, MixedBinaryProgram, Options as SolverOptions, Status as SolverStatus, solve,
};
use epistat_optim::min_cost_flow::{FlowError, MinCostFlow};
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

const SD_FLOOR: f64 = 1e-10;

/// A cardinality or profile matching problem. See [`match_cardinality`]
/// documentation.
#[derive(Debug, Clone)]
pub struct CardinalityRequest<'a> {
    /// Whether each unit is treated.
    pub treated: &'a [bool],
    /// The covariates to balance, one slice per column, each as long as
    /// `treated`. A categorical covariate is passed as its indicator columns.
    pub covariates: &'a [Vec<f64>],
    /// The effect to keep. `Estimand::Atm` is not defined here.
    pub estimand: Estimand,
    /// The number of others per focal unit, or `None` to select the largest
    /// subsets without a fixed ratio (profile matching). `Some(1.0)` is 1:1
    /// cardinality matching for the ATT and the ATC.
    pub ratio: Option<f64>,
    /// The tolerance of each covariate, or one value for all.
    pub tols: Vec<f64>,
    /// Whether each tolerance is in standard deviations, or one value for all.
    pub std_tols: Vec<bool>,
    /// The exact stratum of each unit. Strata are solved separately.
    pub exact: Option<&'a [u64]>,
    /// Sampling weights, for profile matching only.
    pub sampling_weights: Option<&'a [f64]>,
    /// Units that take no part, for example from common support.
    pub discarded: Option<&'a [bool]>,
    /// Covariates to pair the selected units on, one slice per column.
    pub pair_on: &'a [Vec<f64>],
    /// The limits of each stratum's search. The time limit is 120 seconds by
    /// default, as in `MatchIt`.
    pub limits: CardinalityLimits,
}

/// The limits of the solver in each stratum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CardinalityLimits {
    /// The most time for one stratum. `None` is no limit.
    pub time: Option<Duration>,
    /// The most relaxations for one stratum. `None` is no limit.
    pub nodes: Option<u64>,
}

impl Default for CardinalityLimits {
    fn default() -> Self {
        Self {
            time: Some(Duration::from_secs(120)),
            nodes: None,
        }
    }
}

impl<'a> CardinalityRequest<'a> {
    /// 1:1 cardinality matching for the ATT with tolerance 0.05 standard
    /// deviations on every covariate.
    #[must_use]
    pub fn new(treated: &'a [bool], covariates: &'a [Vec<f64>]) -> Self {
        Self {
            treated,
            covariates,
            estimand: Estimand::Att,
            ratio: Some(1.0),
            tols: vec![0.05],
            std_tols: vec![true],
            exact: None,
            sampling_weights: None,
            discarded: None,
            pair_on: &[],
            limits: CardinalityLimits::default(),
        }
    }
}

/// How the search of one stratum went.
#[derive(Debug, Clone, PartialEq)]
pub struct StratumReport {
    /// The exact key of the stratum, 0 without strata.
    pub key: u64,
    /// Units in the stratum, discarded ones excluded.
    pub units: usize,
    /// Treated units selected.
    pub selected_treated: usize,
    /// Control units selected.
    pub selected_control: usize,
    /// Whether the search closed: no larger subset exists.
    pub proven_optimal: bool,
    /// The best upper bound on the size of the integer program's objective that
    /// the search proved.
    pub bound: f64,
    /// Relaxations solved.
    pub nodes: u64,
    /// Simplex pivots.
    pub pivots: u64,
    /// The largest amount by which a constraint of the program is broken by the
    /// selection, in the program's own units, zero when all hold.
    pub max_violation: f64,
}

/// The result of cardinality or profile matching.
#[derive(Debug, Clone)]
pub struct CardinalityOutcome {
    /// The weight of each unit: 0 for a unit that is not selected.
    pub weights: Vec<f64>,
    /// Whether each unit is selected.
    pub selected: Vec<bool>,
    /// The pair of each selected unit when `pair_on` was given, numbered from 0.
    pub pairs: Option<Vec<Option<usize>>>,
    /// One report per solved stratum, in order of the key.
    pub strata: Vec<StratumReport>,
}

/// Why cardinality or profile matching could not run.
#[derive(Debug, Clone, PartialEq)]
pub enum CardinalityError {
    /// An input has the wrong length, or there are no covariates.
    Shape(String),
    /// A covariate, weight or tolerance is not finite.
    NonFinite(String),
    /// A tolerance is negative.
    NegativeTolerance,
    /// The ratio is not positive and finite, or is not a whole number where one
    /// is needed.
    InvalidRatio(String),
    /// Sampling weights with cardinality matching, which has no fixed target.
    SamplingWeightsNeedProfileMatching,
    /// The estimand is not one of the ATT, the ATC and the ATE.
    UnsupportedEstimand,
    /// No stratum holds both groups.
    NoCommonStratum,
    /// The program of a stratum has no solution: relax the tolerances.
    Infeasible {
        /// The stratum.
        key: u64,
    },
    /// A limit ended the search of a stratum before any subset was found.
    NoSolutionInTime {
        /// The stratum.
        key: u64,
    },
    /// A group has no variance where one is needed to standardise.
    ZeroVarianceGroup,
    /// The covariance of the pairing covariates is singular.
    SingularCovariance,
    /// Pairing needs the ATT or the ATE and a whole-number ratio.
    PairingNeedsWholeRatio,
    /// The solver failed.
    Solver(String),
}

impl fmt::Display for CardinalityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(why) => write!(f, "malformed input: {why}"),
            Self::NonFinite(what) => write!(f, "{what} is not finite"),
            Self::NegativeTolerance => f.write_str("a tolerance is negative"),
            Self::InvalidRatio(why) => write!(f, "invalid ratio: {why}"),
            Self::SamplingWeightsNeedProfileMatching => f.write_str(
                "sampling weights need profile matching (ratio unset or the ATE): \
                 cardinality matching has no fixed target population",
            ),
            Self::UnsupportedEstimand => f.write_str("the estimand must be ATT, ATC or ATE"),
            Self::NoCommonStratum => f.write_str("no exact stratum holds both groups"),
            Self::Infeasible { key } => write!(
                f,
                "no subset satisfies the balance constraints in stratum {key}; relax the tolerances"
            ),
            Self::NoSolutionInTime { key } => {
                write!(
                    f,
                    "the solver found no subset in stratum {key} within its limits"
                )
            }
            Self::ZeroVarianceGroup => f.write_str("a group has no variance to standardise by"),
            Self::SingularCovariance => {
                f.write_str("the covariance of the pairing covariates is singular")
            }
            Self::PairingNeedsWholeRatio => {
                f.write_str("pairing needs the ATT or the ATE and a whole-number ratio")
            }
            Self::Solver(why) => write!(f, "the solver failed: {why}"),
        }
    }
}

impl std::error::Error for CardinalityError {}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Cardinality,
    ProfileAtt,
    ProfileAte,
}

/// Weighted variance as `MatchIt` defines it: with weights summing to 1, a
/// binary column gives `m (1 - m)` and any other `Σ w (x - m)² / (1 - Σ w²)`.
fn weighted_variance(values: &[f64], weights: Option<&[f64]>, binary: bool) -> f64 {
    let n = values.len();
    let total: f64 = weights.map_or(n as f64, |w| w.iter().sum());
    let w = |i: usize| weights.map_or(1.0, |w| w[i]) / total;
    let mean: f64 = (0..n).map(|i| w(i) * values[i]).sum();
    if binary {
        return mean * (1.0 - mean);
    }
    let ss: f64 = (0..n).map(|i| w(i) * (values[i] - mean).powi(2)).sum();
    let sq: f64 = (0..n).map(|i| w(i) * w(i)).sum();
    ss / (1.0 - sq)
}

fn is_binary(values: &[f64]) -> bool {
    values.iter().all(|&v| v == 0.0 || v == 1.0)
}

/// Scale each covariate whose tolerance is standardised by its standard
/// deviation, over all units, as `MatchIt` does before it builds the programs.
fn standardise(
    request: &CardinalityRequest<'_>,
    columns: &mut [Vec<f64>],
    focal: bool,
    weights: Option<&[f64]>,
    std_tols: &[bool],
) -> Result<(), CardinalityError> {
    let ate = request.estimand == Estimand::Ate;
    for (c, column) in columns.iter_mut().enumerate() {
        if !std_tols[c] {
            continue;
        }
        let binary = is_binary(column);
        let variance = if ate {
            let mut total = 0.0;
            for group in [false, true] {
                let idx: Vec<usize> = (0..column.len())
                    .filter(|&i| request.treated[i] == group)
                    .collect();
                let values: Vec<f64> = idx.iter().map(|&i| column[i]).collect();
                let w: Option<Vec<f64>> = weights.map(|w| idx.iter().map(|&i| w[i]).collect());
                total += weighted_variance(&values, w.as_deref(), binary);
            }
            total / 2.0
        } else {
            let idx: Vec<usize> = (0..column.len())
                .filter(|&i| request.treated[i] == focal)
                .collect();
            let values: Vec<f64> = idx.iter().map(|&i| column[i]).collect();
            let w: Option<Vec<f64>> = weights.map(|w| idx.iter().map(|&i| w[i]).collect());
            weighted_variance(&values, w.as_deref(), is_binary(&values))
        };
        if !variance.is_finite() {
            return Err(CardinalityError::ZeroVarianceGroup);
        }
        let sd = variance.sqrt();
        if sd >= SD_FLOOR {
            for v in column.iter_mut() {
                *v /= sd;
            }
        }
    }
    Ok(())
}

fn expand<T: Copy>(values: &[T], p: usize, what: &str) -> Result<Vec<T>, CardinalityError> {
    match values.len() {
        1 => Ok(vec![values[0]; p]),
        k if k == p => Ok(values.to_vec()),
        _ => Err(CardinalityError::Shape(format!(
            "{what} must have one entry or one per covariate"
        ))),
    }
}

/// One stratum's program, built and solved. Returns the selection of its units
/// and the report.
struct StratumInput<'a> {
    /// Units of the stratum: index into the full arrays.
    units: &'a [usize],
    treated: &'a [bool],
    x: &'a [Vec<f64>],
    s: &'a [f64],
    kind: Kind,
    ratio: Option<f64>,
    focal: bool,
    tols: &'a [f64],
}

fn solve_stratum(
    input: &StratumInput<'_>,
    key: u64,
    limits: CardinalityLimits,
) -> Result<(Vec<bool>, StratumReport), CardinalityError> {
    let p = input.x.len();
    let nu = input.units.len();
    let group = |i: usize| usize::from(input.treated[input.units[i]]);
    let focal_group = usize::from(input.focal);
    let value = |c: usize, i: usize| input.x[c][input.units[i]];
    let group_size = [
        (0..nu).filter(|&i| group(i) == 0).count(),
        (0..nu).filter(|&i| group(i) == 1).count(),
    ];
    // Sampling weights, scaled within the stratum so each group's sum is its
    // size.
    let scaled: Vec<f64> = {
        let mut out: Vec<f64> = (0..nu).map(|i| input.s[input.units[i]]).collect();
        for (g, &size) in group_size.iter().enumerate() {
            let total: f64 = (0..nu).filter(|&i| group(i) == g).map(|i| out[i]).sum();
            for i in (0..nu).filter(|&i| group(i) == g) {
                out[i] = size as f64 * out[i] / total;
            }
        }
        out
    };

    let weight = |i: usize| scaled[i];

    // Columns: the units that take part, then the continuous columns.
    let (unit_cols, rows): (Vec<usize>, usize) = match input.kind {
        Kind::Cardinality => ((0..nu).collect(), 2 + 2 * p),
        Kind::ProfileAtt => (
            (0..nu).filter(|&i| group(i) != focal_group).collect(),
            1 + 2 * p,
        ),
        Kind::ProfileAte => {
            let count_rows = if input.ratio.is_some() { 2 } else { 0 };
            ((0..nu).collect(), 2 + 4 * p + count_rows)
        }
    };
    let n_units = unit_cols.len();
    let n_extra = match input.kind {
        Kind::Cardinality | Kind::ProfileAtt => 1,
        Kind::ProfileAte => 2 + usize::from(input.ratio.is_some()),
    };
    let n = n_units + n_extra;
    let mut a = vec![0.0; n * rows];
    let mut cost = vec![0.0; n];
    let mut lo = vec![0.0; n];
    let mut hi = vec![1.0; n];
    let mut row_lo = vec![0.0; rows];
    let mut row_hi = vec![0.0; rows];
    let mut binary = vec![false; n];
    // The size column of a fixed ratio counts the focal units, so it is whole.
    let mut integer = vec![false; n];
    // With unit weights a size column is a count, so it is whole.
    let unit_weights = scaled.iter().all(|&w| w == 1.0);
    for col in 0..n_units {
        cost[col] = 1.0;
        binary[col] = true;
    }
    let weighted_mean = |members: &dyn Fn(usize) -> bool, c: usize| -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for i in 0..nu {
            if members(i) {
                num += weight(i) * value(c, i);
                den += weight(i);
            }
        }
        num / den
    };

    match input.kind {
        Kind::Cardinality => {
            let ratio = input.ratio.unwrap_or(1.0);
            // Rows: sizes of groups 0 and 1, then `≤` and `≥` per covariate.
            for (col, &i) in unit_cols.iter().enumerate() {
                let g = group(i);
                a[col * rows + g] = 1.0;
                for c in 0..p {
                    let coef = if g == focal_group {
                        value(c, i)
                    } else {
                        -value(c, i) / ratio
                    };
                    a[col * rows + 2 + c] = coef;
                    a[col * rows + 2 + p + c] = coef;
                }
            }
            let k = n_units;
            for g in 0..2 {
                a[k * rows + g] = if g == focal_group { -1.0 } else { -ratio };
            }
            for c in 0..p {
                a[k * rows + 2 + c] = -input.tols[c];
                a[k * rows + 2 + p + c] = input.tols[c];
                row_lo[2 + c] = f64::NEG_INFINITY;
                row_hi[2 + p + c] = f64::INFINITY;
            }
            let smaller = group_size[0].min(group_size[1]) as f64;
            hi[k] = smaller;
            integer[k] = true;
        }
        Kind::ProfileAtt => {
            let target: Vec<f64> = (0..p)
                .map(|c| weighted_mean(&|i| group(i) == focal_group, c))
                .collect();
            let mut total = 0.0;
            for (col, &i) in unit_cols.iter().enumerate() {
                let s = weight(i);
                total += s;
                a[col * rows] = s;
                for c in 0..p {
                    a[col * rows + 1 + c] = s * value(c, i);
                    a[col * rows + 1 + p + c] = s * value(c, i);
                }
            }
            let t = n_units;
            a[t * rows] = -1.0;
            for c in 0..p {
                a[t * rows + 1 + c] = -(target[c] + input.tols[c]);
                a[t * rows + 1 + p + c] = -(target[c] - input.tols[c]);
                row_lo[1 + c] = f64::NEG_INFINITY;
                row_hi[1 + p + c] = f64::INFINITY;
            }
            hi[t] = total.floor();
            integer[t] = unit_weights;
        }
        Kind::ProfileAte => {
            let target: Vec<f64> = (0..p).map(|c| weighted_mean(&|_| true, c)).collect();
            // Rows: size of group g; `≤` and `≥` per group and covariate; the
            // counts when there is a ratio.
            for (col, &i) in unit_cols.iter().enumerate() {
                let g = group(i);
                let s = weight(i);
                a[col * rows + g] = s;
                for c in 0..p {
                    let r1 = 2 + g * 2 * p + c;
                    a[col * rows + r1] = s * value(c, i);
                    a[col * rows + r1 + p] = s * value(c, i);
                }
                if input.ratio.is_some() {
                    a[col * rows + 2 + 4 * p + g] = 1.0;
                }
            }
            for g in 0..2 {
                let t = n_units + g;
                a[t * rows + g] = -1.0;
                lo[t] = 1.0;
                hi[t] = group_size[g] as f64;
                integer[t] = unit_weights;
                for (c, &mu) in target.iter().enumerate() {
                    let r1 = 2 + g * 2 * p + c;
                    a[t * rows + r1] = -(mu + input.tols[c] / 2.0);
                    a[t * rows + r1 + p] = -(mu - input.tols[c] / 2.0);
                    row_lo[r1] = f64::NEG_INFINITY;
                    row_hi[r1 + p] = f64::INFINITY;
                }
            }
            if let Some(ratio) = input.ratio {
                let k = n_units + 2;
                for g in 0..2 {
                    a[k * rows + 2 + 4 * p + g] = if g == focal_group { -1.0 } else { -ratio };
                }
                let limit = (0..2)
                    .map(|g| group_size[g] as f64 / if g == focal_group { 1.0 } else { ratio })
                    .fold(f64::INFINITY, f64::min);
                hi[k] = limit.floor();
                integer[k] = true;
            }
        }
    }
    lo[..n_units].fill(0.0);

    let program = MixedBinaryProgram {
        rows,
        columns: &a,
        cost: &cost,
        col_lo: &lo,
        col_hi: &hi,
        row_lo: &row_lo,
        row_hi: &row_hi,
        binary: &binary,
        integer: &integer,
    };
    let options = SolverOptions {
        node_limit: limits.nodes,
        time_limit: limits.time,
        ..SolverOptions::default()
    };
    let solution = solve(&program, &options).map_err(|e| match e {
        MipError::Infeasible => CardinalityError::Infeasible { key },
        MipError::NoSolution { .. } => CardinalityError::NoSolutionInTime { key },
        other => CardinalityError::Solver(other.to_string()),
    })?;

    let mut selected = vec![false; nu];
    for (col, &i) in unit_cols.iter().enumerate() {
        selected[i] = solution.x[col] > 0.5;
    }
    if input.kind == Kind::ProfileAtt {
        for (i, sel) in selected.iter_mut().enumerate() {
            if group(i) == focal_group {
                *sel = true;
            }
        }
    }
    let count = |g: usize| (0..nu).filter(|&i| selected[i] && group(i) == g).count();
    let report = StratumReport {
        key,
        units: nu,
        selected_treated: count(1),
        selected_control: count(0),
        proven_optimal: solution.status == SolverStatus::Optimal,
        bound: solution.bound,
        nodes: solution.nodes,
        pivots: solution.pivots,
        max_violation: solution.max_row_violation,
    };
    Ok((selected, report))
}

/// Cardinality or profile matching. See [`match_cardinality`].
///
/// # Errors
/// A [`CardinalityError`] for malformed input, for a stratum whose program has no
/// solution, or when the solver fails or runs out of time before it has one.
pub fn match_cardinality(
    request: &CardinalityRequest<'_>,
) -> Result<CardinalityOutcome, CardinalityError> {
    let n = request.treated.len();
    let p = request.covariates.len();
    if p == 0 {
        return Err(CardinalityError::Shape("no covariates".to_string()));
    }
    for column in request.covariates {
        if column.len() != n {
            return Err(CardinalityError::Shape(
                "a covariate has the wrong length".to_string(),
            ));
        }
        if column.iter().any(|v| !v.is_finite()) {
            return Err(CardinalityError::NonFinite("a covariate".to_string()));
        }
    }
    if request.exact.is_some_and(|e| e.len() != n) {
        return Err(CardinalityError::Shape(
            "exact has the wrong length".to_string(),
        ));
    }
    if request.discarded.is_some_and(|d| d.len() != n) {
        return Err(CardinalityError::Shape(
            "discarded has the wrong length".to_string(),
        ));
    }
    if request.estimand == Estimand::Atm {
        return Err(CardinalityError::UnsupportedEstimand);
    }
    let tols = expand(&request.tols, p, "tols")?;
    if tols.iter().any(|t| !t.is_finite()) {
        return Err(CardinalityError::NonFinite("a tolerance".to_string()));
    }
    if tols.iter().any(|&t| t < 0.0) {
        return Err(CardinalityError::NegativeTolerance);
    }
    let std_tols = expand(&request.std_tols, p, "std_tols")?;
    let mut ratio = request.ratio;
    if ratio.is_some_and(|r| !(r.is_finite() && r > 0.0)) {
        return Err(CardinalityError::InvalidRatio(
            "the ratio must be positive and finite, or unset".to_string(),
        ));
    }
    let kind = if request.estimand == Estimand::Ate {
        Kind::ProfileAte
    } else if ratio.is_some() {
        Kind::Cardinality
    } else {
        Kind::ProfileAtt
    };
    if kind == Kind::Cardinality && ratio.is_some_and(|r| r.fract() != 0.0) {
        return Err(CardinalityError::InvalidRatio(
            "cardinality matching needs a whole-number ratio".to_string(),
        ));
    }
    if kind == Kind::ProfileAtt {
        ratio = None;
    }
    // Sampling weights as given; each stratum scales them to its group sizes.
    let s: Vec<f64> = match request.sampling_weights {
        None => vec![1.0; n],
        Some(w) => {
            if w.len() != n {
                return Err(CardinalityError::Shape(
                    "sampling_weights has the wrong length".to_string(),
                ));
            }
            if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(CardinalityError::NonFinite("a sampling weight".to_string()));
            }
            if kind == Kind::Cardinality && w.iter().any(|&v| v != w[0]) {
                return Err(CardinalityError::SamplingWeightsNeedProfileMatching);
            }
            w.to_vec()
        }
    };
    let focal = request.estimand != Estimand::Atc;

    let mut columns: Vec<Vec<f64>> = request.covariates.to_vec();
    standardise(
        request,
        &mut columns,
        focal,
        request.sampling_weights.map(|_| s.as_slice()),
        &std_tols,
    )?;

    let discarded = |i: usize| request.discarded.is_some_and(|d| d[i]);
    let mut by_key: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        if !discarded(i) {
            by_key
                .entry(request.exact.map_or(0, |e| e[i]))
                .or_default()
                .push(i);
        }
    }
    by_key.retain(|_, units| {
        units.iter().any(|&i| request.treated[i]) && units.iter().any(|&i| !request.treated[i])
    });
    if by_key.is_empty() {
        return Err(CardinalityError::NoCommonStratum);
    }

    let mut weights = vec![0.0; n];
    let mut selected_all = vec![false; n];
    let mut reports = Vec::new();
    for (&key, units) in &by_key {
        let input = StratumInput {
            units,
            treated: request.treated,
            x: &columns,
            s: &s,
            kind,
            ratio,
            focal,
            tols: &tols,
        };
        let (selected, report) = solve_stratum(&input, key, request.limits)?;
        let mut w: Vec<f64> = selected.iter().map(|&b| f64::from(u8::from(b))).collect();
        // Scale one group to the other's total, as MatchIt does.
        let size = |g: bool| units.iter().filter(|&&i| request.treated[i] == g).count();
        let total = |g: bool| {
            units
                .iter()
                .zip(&w)
                .filter(|&(&i, _)| request.treated[i] == g)
                .map(|(_, v)| *v)
                .sum::<f64>()
        };
        let reference = if kind == Kind::ProfileAtt {
            focal
        } else {
            // The smaller group of the stratum by size, the first (controls) on a tie.
            size(true) < size(false)
        };
        let reference_total = total(reference);
        let other_total = total(!reference);
        if other_total > 0.0 {
            for (idx, &i) in units.iter().enumerate() {
                if request.treated[i] != reference {
                    w[idx] *= reference_total / other_total;
                }
            }
        }
        for (idx, &i) in units.iter().enumerate() {
            weights[i] = w[idx];
            selected_all[i] = selected[idx];
        }
        reports.push(report);
    }

    // As `matchit()` does for every method, scale the weights of each group to
    // sum to the number of its units that keep a weight.
    for group in [false, true] {
        let members: Vec<usize> = (0..n).filter(|&i| request.treated[i] == group).collect();
        let kept = members.iter().filter(|&&i| weights[i] > 0.0).count() as f64;
        let total: f64 = members.iter().map(|&i| weights[i]).sum();
        if total > 0.0 {
            for &i in &members {
                weights[i] *= kept / total;
            }
        }
    }

    let pairs = if request.pair_on.is_empty() {
        None
    } else {
        Some(pair_selected(request, ratio, kind, &by_key, &selected_all)?)
    };
    Ok(CardinalityOutcome {
        weights,
        selected: selected_all,
        pairs,
        strata: reports,
    })
}

/// Optimal pair matching of the selected units of each stratum on the
/// Mahalanobis distance of the pairing covariates.
fn pair_selected(
    request: &CardinalityRequest<'_>,
    ratio: Option<f64>,
    kind: Kind,
    by_key: &BTreeMap<u64, Vec<usize>>,
    selected: &[bool],
) -> Result<Vec<Option<usize>>, CardinalityError> {
    let Some(r) = ratio else {
        return Err(CardinalityError::PairingNeedsWholeRatio);
    };
    if r.fract() != 0.0 || request.estimand == Estimand::Atc || kind == Kind::ProfileAtt {
        return Err(CardinalityError::PairingNeedsWholeRatio);
    }
    let strata: Vec<&[usize]> = by_key.values().map(Vec::as_slice).collect();
    pair_units(
        request.treated,
        selected,
        request.pair_on,
        r as i64,
        &strata,
        request.discarded,
        request.sampling_weights,
    )
}

/// Optimal pair matching of selected units on the Mahalanobis distance of the
/// `pair_on` columns, within each stratum in `strata` (positions of its units).
///
/// Each selected treated unit gets `ratio` selected controls, and the total
/// distance of the pairs is the least possible. The distance is the Mahalanobis
/// distance with the pooled within-group covariance of the units that are not
/// `discarded`, weighted by `sampling_weights` when given. Columns without
/// variance are left out. Returns the pair of each unit, numbered from 0 in
/// order of the strata and the treated units, and `None` for a unit that is not
/// paired. The distance matrix of a stratum is dense.
///
/// # Errors
/// [`CardinalityError::PairingNeedsWholeRatio`] when a stratum does not hold
/// `ratio` selected controls per selected treated unit, and
/// [`CardinalityError::SingularCovariance`] when the covariance cannot be
/// inverted.
pub fn pair_units(
    treated: &[bool],
    selected: &[bool],
    pair_on: &[Vec<f64>],
    ratio_int: i64,
    strata: &[&[usize]],
    discarded: Option<&[bool]>,
    sampling_weights: Option<&[f64]>,
) -> Result<Vec<Option<usize>>, CardinalityError> {
    let n = treated.len();
    for column in pair_on {
        if column.len() != n {
            return Err(CardinalityError::Shape(
                "a pairing covariate has the wrong length".to_string(),
            ));
        }
        if column.iter().any(|v| !v.is_finite()) {
            return Err(CardinalityError::NonFinite(
                "a pairing covariate".to_string(),
            ));
        }
    }
    // Columns without variance are dropped.
    let cols: Vec<&Vec<f64>> = pair_on
        .iter()
        .filter(|c| {
            let (mn, mx) = c
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            (mx - mn) >= f64::EPSILON.sqrt()
        })
        .collect();
    let d = cols.len();
    let all: Vec<usize> = (0..n)
        .filter(|&i| !discarded.is_some_and(|dd| dd[i]))
        .collect();
    let weight = |i: usize| sampling_weights.map_or(1.0, |w| w[i]);
    // Pooled within-group scatter matrix of the non-discarded units.
    let mut scatter = vec![0.0; d * d];
    if d > 0 {
        for group in [false, true] {
            let members: Vec<usize> = all
                .iter()
                .copied()
                .filter(|&i| treated[i] == group)
                .collect();
            let total: f64 = members.iter().map(|&i| weight(i)).sum();
            let means: Vec<f64> = cols
                .iter()
                .map(|c| members.iter().map(|&i| weight(i) * c[i]).sum::<f64>() / total)
                .collect();
            for &i in &members {
                for a in 0..d {
                    for b in 0..=a {
                        scatter[a * d + b] +=
                            weight(i) * (cols[a][i] - means[a]) * (cols[b][i] - means[b]);
                    }
                }
            }
        }
        let wsum: f64 = all.iter().map(|&i| weight(i)).sum();
        for a in 0..d {
            for b in 0..=a {
                scatter[a * d + b] /= wsum;
                scatter[b * d + a] = scatter[a * d + b];
            }
        }
    }
    // The Mahalanobis distance is the Euclidean one after L⁻¹ with `S = L Lᵀ`.
    let chol = if d == 0 {
        Vec::new()
    } else {
        cholesky_lower(&scatter, d).ok_or(CardinalityError::SingularCovariance)?
    };
    let transformed = |i: usize| -> Vec<f64> {
        // Solve L z = x for z.
        let x: Vec<f64> = cols.iter().map(|c| c[i]).collect();
        let mut z = vec![0.0; d];
        for a in 0..d {
            let mut s = x[a];
            for b in 0..a {
                s -= chol[a * d + b] * z[b];
            }
            z[a] = s / chol[a * d + a];
        }
        z
    };

    let mut assigned: Vec<Option<usize>> = vec![None; n];
    let mut next_pair = 0_usize;
    for units in strata {
        let tr: Vec<usize> = units
            .iter()
            .copied()
            .filter(|&i| selected[i] && treated[i])
            .collect();
        let controls: Vec<usize> = units
            .iter()
            .copied()
            .filter(|&i| selected[i] && !treated[i])
            .collect();
        if tr.is_empty() {
            continue;
        }
        let nt = tr.len();
        let nc = controls.len();
        let want = i64::try_from(nt).unwrap_or(i64::MAX) * ratio_int;
        if i64::try_from(nc).unwrap_or(-1) != want {
            return Err(CardinalityError::PairingNeedsWholeRatio);
        }
        let zt: Vec<Vec<f64>> = tr.iter().map(|&i| transformed(i)).collect();
        let zc: Vec<Vec<f64>> = controls.iter().map(|&i| transformed(i)).collect();
        let dist = |a: usize, b: usize| -> f64 {
            zt[a]
                .iter()
                .zip(&zc[b])
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt()
        };
        let mut max_distance = 0.0_f64;
        for a in 0..nt {
            for b in 0..nc {
                max_distance = max_distance.max(dist(a, b));
            }
        }
        // Integer costs: the largest distance maps to about 2^30.
        let scale = if max_distance > 0.0 {
            2.0_f64.powi(30 - max_distance.log2().ceil() as i32)
        } else {
            1.0
        };
        let mut flow = MinCostFlow::new(nt + nc);
        flow.reserve_arcs(nt * nc);
        for a in 0..nt {
            flow.set_supply(a, ratio_int);
        }
        for b in 0..nc {
            flow.set_supply(nt + b, -1);
        }
        for a in 0..nt {
            for b in 0..nc {
                flow.add_arc(a, nt + b, 1, (dist(a, b) * scale).round() as i64);
            }
        }
        let solution = flow.solve().map_err(|e| match e {
            FlowError::Infeasible => CardinalityError::PairingNeedsWholeRatio,
            other => CardinalityError::Solver(other.to_string()),
        })?;
        for a in 0..nt {
            let id = next_pair;
            next_pair += 1;
            assigned[tr[a]] = Some(id);
            for b in 0..nc {
                if solution.flow[a * nc + b] == 1 {
                    assigned[controls[b]] = Some(id);
                }
            }
        }
    }
    Ok(assigned)
}
