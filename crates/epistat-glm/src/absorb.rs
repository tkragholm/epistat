//! Absorption of any number of fixed-effect factors by preconditioned
//! conjugate gradients.
//!
//! [`FixedEffects`] holds the factors of a linear model (unit, time, firm and
//! so on) for the rows that enter it, and [`FixedEffects::demean`] removes them
//! from columns: each column is replaced by its residual from the weighted
//! least-squares projection on the dummies of every factor, without forming
//! the dummies. [`crate::feols`] fits a linear model on the demeaned columns.
//!
//! # Algorithm
//!
//! With `Q` factors, `g_q(i)` the level of factor `q` on row `i`, prior weights
//! `w_i` and a column `v`, the projection finds coefficients `alpha_q` that
//! solve the normal equations
//!
//! ```text
//! sum_{i: g_q(i) = g} w_i (v_i - sum_r alpha_r[g_r(i)]) = 0     for every q and level g
//! ```
//!
//! and the demeaned column is the residual `r_i = v_i - sum_q alpha_q[g_q(i)]`.
//! With `S_q[g] = sum_{i: g_q(i) = g} w_i v_i` the level sums of the column and
//! `W_q[g]` those of the weights, the column is read once for the `S_q`, and
//! each iteration reads only the level codes and the weights. Each column is
//! first centred at its weighted mean, which lies in the span of every factor
//! and leaves the residual unchanged, so that the criterion below does not
//! depend on the column's location.
//!
//! The classical solver is the method of alternating projections \[1, 2\],
//! block Gauss-Seidel on the normal equations, which updates one factor at a
//! time to the weighted level means of what the others leave. Its error falls
//! by about `1 - 1 / k` a sweep, `k` the condition number below, which is slow
//! on poorly connected designs.
//!
//! Eliminating the factor with the most levels, `1`, exactly, `alpha_1 =
//! W_1^{-1} (S_1 - C_1x x)` with `C_qr` the weighted cross tabulation of
//! factors `q` and `r` and `x` the coefficients of the others, leaves
//!
//! ```text
//! S x = c,     S = A_xx - C_x1 W_1^{-1} C_1x,     c = S_x - C_x1 W_1^{-1} S_1
//! ```
//!
//! with `A_xx` the normal equations of the other factors among themselves.
//! `S` is positive semi-definite and `c` lies in its range, so conjugate
//! gradients \[4\] preconditioned by the level weights `M = diag(W_q)`
//! converge to a solution from `x = 0` \[5\]. The preconditioned residual
//! `M^{-1} (c - S x)` is exactly the vector of level means `m_q` of the
//! residual at the point `(alpha_1(x), x)`, where the largest factor's means
//! are zero, so the criterion below applies unchanged. The error falls by
//! about the factor `(sqrt(k) - 1) / (sqrt(k) + 1)` an iteration, `k` the
//! ratio of the extreme positive eigenvalues of `M^{-1} S`. On a chain of 1000
//! firms with few movers between neighbours `k` is large: at 10^5 rows the
//! gradients took about 1200 iterations, where alternating projections did not
//! meet the criterion at `1e-10` in 30 000. An iteration applies `S` once, one
//! pass over the rows for each factor. The residual `c - S x` is carried by
//! its recursion; when that passes the criterion, the means are computed
//! afresh at the point, and the iteration continues from them when they do not
//! pass. LSQR or LSMR on the column-scaled dummies, as some packages use, is
//! conjugate gradients (or MINRES) on the full equations `A` with the same
//! preconditioner, without the elimination; on the test designs it took about
//! twice the iterations. Accelerated alternating projections met the same
//! criterion no faster on any design measured: as fast on well connected
//! panels, where both converge in a few iterations, and 4 to 5 times slower on
//! a sparse worker-firm graph at 10^7 rows.
//!
//! # Convergence criterion
//!
//! The iteration stops at a point where every weighted level mean of the
//! residual is small:
//!
//! ```text
//! m_q[g] = sum_{i: g_q(i) = g} w_i r_i / W_q[g],      max_{q, g} |m_q[g]| <= tol s
//! ```
//!
//! with `s = sqrt(sum w v_c^2 / sum w)` the weighted root mean square of the
//! centred column. The `m_q` are the steps a projection of each factor would
//! take, and they vanish exactly at the solution. The criterion is checked at
//! the point that is returned.
//!
//! The criterion bounds the error of the demeaned column. Let `D` be the
//! dummies of all factors, `A = D' W D` and `r*` the exact residual. The error
//! `e = r - r*` lies in the span of `D`, and `D' W e = D' W r = (W_q[g] m_q[g])`,
//! so
//!
//! ```text
//! ||r - r*||_W^2 = gamma' A^+ gamma <= ||gamma||^2 / lambda_+,      gamma = (W_q[g] m_q[g])
//! ```
//!
//! with `lambda_+` the smallest positive eigenvalue of `A`, which measures how
//! well the factors are connected. A regression on demeaned columns is
//! affected at second order: with `X~ = P X + E` and `y~ = P y + e`, where `E`
//! and `e` lie in the span of `D` and `P X` is orthogonal to it, `X~' W X~ =
//! X' P W P X + E' W E` and `X~' W y~ = X' P W P y + E' W e`.
//!
//! # Singletons
//!
//! A level with one row fits that row exactly and takes a degree of freedom
//! without information. [`Singletons::Remove`] drops the rows of such levels,
//! again and again until no factor has a level with one row, as `fixest`
//! (`fixef.rm = "perfect_fit"` or `"singletons"`, the default) does \[3\].
//! Rows with weight zero are dropped before that, as `fixest` drops them.
//!
//! # Fixed effects and their normalisation
//!
//! The coefficients are identified only up to the redundancies of the
//! factors; the fitted sums `sum_q alpha_q[g_q(i)]` are unique.
//! [`FixedEffects::fixed_effects`] and `FeolsFit::fixed_effects` set
//! references and recover the other coefficients from the fitted sums.
//!
//! With two factors, the levels joined by rows form a bipartite graph. Its
//! connected components are found by union-find and numbered in the order of
//! their first rows. In each, the level of the second factor on the
//! component's first row is set to zero, and a breadth-first pass from it
//! fixes every other level of the component, each from one row:
//! `alpha_1[g_1] + alpha_2[g_2]` equals the row's fitted sum. The number of
//! references, the number of components, is the number of redundant
//! coefficients \[6\].
//!
//! With three or more factors no rule on the graph gives the redundancies in
//! general, and the coefficients follow a propagation: the first row's levels
//! of the second and later factors are set to zero, then every coefficient
//! that one row determines from the others (a row with one coefficient not
//! known) is found from its fitted sum; when none is, the first row in row
//! order with the most coefficients known supplies new references in the same
//! way, its unknown levels of every factor but the first among them set to
//! zero. The first factor never holds a reference. This is the count `fixest`
//! reports \[3\], which the small-sample correction of [`crate::feols`] reads,
//! and it need not equal the redundancies. When the normalised coefficients do
//! not reproduce the fitted sums, within a running bound on the rounding of
//! each coefficient, the solver's own coefficients are returned with
//! [`FixedEffectValues::normalised`] false and the count unchanged.
//!
//! # Determinism
//!
//! A factor with at most 4096 levels is summed by blocks of
//! `16 * ROW_BLOCK` rows into one partial vector per block, and the partial
//! vectors are added in block order. A factor with more levels is summed level
//! by level over its rows in row order; when its codes are not sorted, the
//! other factors' codes and the weights are kept in its order as well, so that
//! a pass reads them in sequence. Inner products run over fixed blocks added in
//! block order. The normalisation runs on one thread. Nothing depends on the
//! number of threads.
//!
//! # References
//!
//! 1. Guimarães P, Portugal P. A simple feasible procedure to fit models with
//!    high-dimensional fixed effects. Stata J. 2010;10(4):628-49.
//! 2. Gaure S. OLS with multiple high dimensional category variables. Comput
//!    Stat Data Anal. 2013;66:8-18. doi:10.1016/j.csda.2013.03.024
//! 3. Bergé L. Efficient estimation of maximum likelihood models with multiple
//!    fixed-effects: the R package FENmlm. CREA Discussion Paper 2018-13; and
//!    the documentation of the R package `fixest` 0.14.2 (`feols`, `fixef`).
//! 4. Hestenes MR, Stiefel E. Methods of conjugate gradients for solving
//!    linear systems. J Res Natl Bur Stand. 1952;49(6):409-36.
//!    doi:10.6028/jres.049.044
//! 5. Kaasschieter EF. Preconditioned conjugate gradients for solving
//!    singular systems. J Comput Appl Math. 1988;24(1-2):265-75.
//!    doi:10.1016/0377-0427(88)90358-5
//! 6. Abowd JM, Creecy RH, Kramarz F. Computing person and firm effects using
//!    linked longitudinal employer-employee data. US Census Bureau Technical
//!    Paper TP-2002-06; 2002.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use epistat_core::InputError;
use epistat_linalg::ROW_BLOCK;
use rayon::prelude::*;

use crate::Error;
use crate::error::{length, weight_error};

/// The rows per block of the dense sums.
const DENSE_BLOCK: usize = 16 * ROW_BLOCK;

/// Factors with at most this many levels are summed into one dense partial
/// vector per block of `16 * ROW_BLOCK` rows.
pub(crate) const DENSE_LEVELS: usize = DENSE_BLOCK / 4;

/// The levels handled by one task when a factor is summed level by level.
const LEVELS_PER_TASK: usize = 4096;

/// What to do with levels that have one row.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Singletons {
    /// Remove their rows, repeatedly, as `fixest` does by default.
    Remove,
    /// Keep them.
    Keep,
}

/// How the normal equations of the projection are solved. Every method
/// stops by the criterion of the module documentation, so every method
/// carries the same bound on the error.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AbsorbMethod {
    /// Preconditioned conjugate gradients on the equations of the factors
    /// other than the largest, which is eliminated exactly. Much faster on
    /// poorly connected designs, such as workers who rarely move between
    /// firms, and as fast on well connected ones.
    #[default]
    ConjugateGradient,
}

/// The controls of the iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Absorb {
    /// The criterion: every weighted level mean of the residual at most
    /// `tolerance` times the root mean square of the centred column.
    pub tolerance: f64,
    /// The largest number of iterations, each of one sweep's work: one pass
    /// over the rows for each factor.
    pub max_iterations: usize,
    /// The solver.
    pub method: AbsorbMethod,
}

impl Absorb {
    /// A tolerance of `1e-10`, 10 000 iterations and
    /// [`AbsorbMethod::ConjugateGradient`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 10_000,
            method: AbsorbMethod::ConjugateGradient,
        }
    }

    fn check(&self) -> Result<(), Error> {
        crate::error::tolerance("absorb tolerance", self.tolerance)?;
        crate::error::iterations("absorb max_iterations", self.max_iterations)
    }
}

impl Default for Absorb {
    fn default() -> Self {
        Self::new()
    }
}

/// How a factor is summed by level.
#[derive(Clone, Debug)]
enum Plan {
    /// Dense partial vectors per block of rows.
    Dense,
    /// Level by level over the rows `bounds[g]..bounds[g + 1]`, for codes
    /// that do not decrease along the rows.
    Sorted { bounds: Vec<usize> },
    /// Level by level over the positions `bounds[g]..bounds[g + 1]` of
    /// `order`, the rows sorted by level and stable within a level, with the
    /// other factors' codes and the weights copied in that order so that a
    /// pass reads them in sequence.
    Ordered {
        bounds: Vec<usize>,
        order: Vec<u32>,
        /// The codes of each factor in this order; empty for this factor.
        others: Vec<Vec<u32>>,
        weights: Option<Vec<f64>>,
    },
}

/// One factor on the rows that enter.
#[derive(Clone, Debug)]
struct Level {
    /// The level of each row, numbered `0..levels` in the order of the
    /// caller's codes.
    codes: Vec<u32>,
    /// The caller's code of each level.
    labels: Vec<u32>,
    /// One more than the largest code the caller gave.
    input_levels: usize,
    /// The weight of each level.
    weight: Vec<f64>,
    plan: Plan,
}

impl Level {
    fn levels(&self) -> usize {
        self.labels.len()
    }

    /// `out[g]` is the sum of `term(k)` over the positions `k` of level `g`:
    /// the rows themselves, or the positions of [`Plan::Ordered`]'s order.
    fn accumulate(
        &self,
        term: impl Fn(usize) -> f64 + Sync,
        scratch: &mut Vec<f64>,
        out: &mut [f64],
    ) {
        let n = self.codes.len();
        let g = self.levels();
        let by_level = |bounds: &[usize], out: &mut [f64]| {
            out.par_chunks_mut(LEVELS_PER_TASK)
                .enumerate()
                .for_each(|(task, chunk)| {
                    for (k, o) in chunk.iter_mut().enumerate() {
                        let level = task * LEVELS_PER_TASK + k;
                        let mut total = 0.0;
                        for j in bounds[level]..bounds[level + 1] {
                            total += term(j);
                        }
                        *o = total;
                    }
                });
        };
        match &self.plan {
            Plan::Dense => {
                let blocks = n.div_ceil(DENSE_BLOCK);
                scratch.clear();
                scratch.resize(blocks * g, 0.0);
                let codes = &self.codes;
                scratch
                    .par_chunks_mut(g)
                    .enumerate()
                    .for_each(|(block, part)| {
                        for i in block * DENSE_BLOCK..((block + 1) * DENSE_BLOCK).min(n) {
                            part[codes[i] as usize] += term(i);
                        }
                    });
                out.fill(0.0);
                for part in scratch.chunks(g) {
                    for (o, p) in out.iter_mut().zip(part) {
                        *o += p;
                    }
                }
            }
            Plan::Sorted { bounds } | Plan::Ordered { bounds, .. } => by_level(bounds, out),
        }
    }

    /// `out[g] = sum over the rows of level g of term(row)`.
    fn sums(&self, term: impl Fn(usize) -> f64 + Sync, scratch: &mut Vec<f64>, out: &mut [f64]) {
        match &self.plan {
            Plan::Ordered { order, .. } => {
                self.accumulate(|j| term(order[j] as usize), scratch, out);
            }
            _ => self.accumulate(term, scratch, out),
        }
    }
}

/// The fixed-effect factors of the rows that enter a model.
#[derive(Clone, Debug)]
pub struct FixedEffects {
    n_input: usize,
    /// The input rows that enter, in increasing order.
    kept: Vec<usize>,
    /// The weights of the rows that enter; `None` for unit weights.
    weights: Option<Vec<f64>>,
    total_weight: f64,
    factors: Vec<Level>,
    removed_zero_weight: usize,
    removed_singletons: usize,
}

/// A column with its fixed effects removed.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Demeaned {
    /// The residuals, one per row that enters ([`FixedEffects::rows`]), one
    /// column per input column.
    pub columns: epistat_linalg::Mat<f64>,
    /// The root mean square `s` of each centred column, as in the criterion.
    pub scale: Vec<f64>,
    /// The iterations each column took.
    pub iterations: Vec<usize>,
    /// The coefficients of each column: factor by factor, level by level in
    /// the order of [`FixedEffects::levels`], before normalisation.
    pub(crate) coefficients: Vec<Vec<Vec<f64>>>,
}

/// The fixed effects of a fit, normalised by the rule of the module
/// documentation.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct FixedEffectValues {
    /// For each factor, the coefficient of each code the caller gave, `None`
    /// for a code without rows in the fit.
    pub values: Vec<Vec<Option<f64>>>,
    /// The number of coefficients set to zero as references in each factor,
    /// also when [`FixedEffectValues::normalised`] is false.
    pub references: Vec<usize>,
    /// Whether the values are the normalised coefficients. False only with
    /// three or more factors whose normalised coefficients do not reproduce
    /// the fitted sums; the values are then the solver's own.
    pub normalised: bool,
    /// With two factors, the connected component of each code of each factor,
    /// `None` for codes without rows.
    component: Option<Vec<Vec<Option<u32>>>>,
}

impl FixedEffectValues {
    /// The sum of the fixed effects of one row, from the caller's codes of each
    /// factor. `None` when a code has no rows in the fit, and with two factors
    /// when the two levels lie in different connected components, where the
    /// sum is not identified. With three or more factors only the first
    /// condition is checked.
    #[must_use]
    pub fn sum(&self, codes: &[u32]) -> Option<f64> {
        if codes.len() != self.values.len() {
            return None;
        }
        let mut total = 0.0;
        for (q, &code) in codes.iter().enumerate() {
            total += (*self.values[q].get(code as usize)?)?;
        }
        if let Some(component) = &self.component {
            let a = (*component[0].get(codes[0] as usize)?)?;
            let b = (*component[1].get(codes[1] as usize)?)?;
            if a != b {
                return None;
            }
        }
        Some(total)
    }

    /// [`FixedEffectValues::sum`] for every row of `factors`, one slice of
    /// codes per factor.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the factors differ in number or length.
    pub fn predict(&self, factors: &[&[u32]]) -> Result<Vec<Option<f64>>, Error> {
        length("factors", self.values.len(), factors.len())?;
        let n = factors.first().map_or(0, |f| f.len());
        for f in factors {
            length("factor codes", n, f.len())?;
        }
        Ok((0..n)
            .into_par_iter()
            .map(|i| {
                let codes: Vec<u32> = factors.iter().map(|f| f[i]).collect();
                self.sum(&codes)
            })
            .collect())
    }
}

/// `sum_i term(i)` over `0..n` by blocks of [`ROW_BLOCK`], added in block
/// order.
pub(crate) fn blocked_total(n: usize, term: impl Fn(usize) -> f64 + Sync) -> f64 {
    let partial: Vec<f64> = (0..n.div_ceil(ROW_BLOCK))
        .into_par_iter()
        .map(|block| {
            (block * ROW_BLOCK..((block + 1) * ROW_BLOCK).min(n))
                .map(&term)
                .sum::<f64>()
        })
        .collect();
    partial.iter().sum()
}

/// The rows of each level of `codes`: bounds and, unless the codes do not
/// decrease, the order of the rows by level, stable within a level.
fn grouping(codes: &[u32], levels: usize) -> (Vec<usize>, Option<Vec<u32>>) {
    let mut bounds = vec![0usize; levels + 1];
    for &c in codes {
        bounds[c as usize + 1] += 1;
    }
    for g in 0..levels {
        bounds[g + 1] += bounds[g];
    }
    let sorted = codes.windows(2).all(|pair| pair[0] <= pair[1]);
    let order = (!sorted).then(|| {
        let mut next = bounds.clone();
        let mut order = vec![0u32; codes.len()];
        for (i, &c) in codes.iter().enumerate() {
            order[next[c as usize]] = u32::try_from(i).unwrap_or(u32::MAX);
            next[c as usize] += 1;
        }
        order
    });
    (bounds, order)
}

/// The number of rows of each code among `rows`.
fn counts(codes: &[u32], rows: &[usize], levels: usize) -> Vec<u32> {
    let mut count = vec![0u32; levels];
    for &i in rows {
        let c = &mut count[codes[i] as usize];
        *c = c.saturating_add(1);
    }
    count
}

impl FixedEffects {
    /// The factors of a model: one slice of level codes per factor, with one
    /// code per input row. Codes are any numbers; a code's coefficient is
    /// reported at that position, so the memory is proportional to the largest
    /// code. `weights` are prior weights, 1 when `None`. Rows with weight zero
    /// do not enter, so 0/1 weights fit a subset of the rows, and with
    /// [`Singletons::Remove`] the rows of levels with one row are removed until
    /// none is left.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for no factors, slices of the wrong length, a weight
    /// that is negative or not finite, more rows than 32-bit row numbers
    /// hold, and no rows left after the rows of weight zero and the
    /// singletons are removed.
    pub fn new(
        factors: &[&[u32]],
        weights: Option<&[f64]>,
        singletons: Singletons,
    ) -> Result<Self, Error> {
        let Some(first) = factors.first() else {
            return Err(InputError::Empty { what: "factors" }.into());
        };
        let n = first.len();
        if u32::try_from(n).is_err() {
            return Err(InputError::TooLarge {
                what: "rows",
                limit: u64::from(u32::MAX),
                found: n as u64,
            }
            .into());
        }
        for f in factors {
            if f.len() != n {
                return Err(length_error("factor codes", n, f.len()));
            }
        }
        if let Some(w) = weights {
            if w.len() != n {
                return Err(length_error("weights", n, w.len()));
            }
            if let Some((index, &value)) = w
                .par_iter()
                .enumerate()
                .find_first(|&(_, &v)| !(v.is_finite() && v >= 0.0))
            {
                return Err(weight_error("weights", index, value));
            }
        }
        let input_levels: Vec<usize> = factors
            .iter()
            .map(|f| f.par_iter().max().map_or(0, |&m| m as usize + 1))
            .collect();
        let mut kept: Vec<usize> = match weights {
            Some(w) => (0..n).filter(|&i| w[i] > 0.0).collect(),
            None => (0..n).collect(),
        };
        let removed_zero_weight = n - kept.len();
        if singletons == Singletons::Remove {
            loop {
                let count: Vec<Vec<u32>> = factors
                    .par_iter()
                    .zip(&input_levels)
                    .map(|(f, &levels)| counts(f, &kept, levels))
                    .collect();
                let before = kept.len();
                kept.retain(|&i| {
                    factors
                        .iter()
                        .zip(&count)
                        .all(|(f, c)| c[f[i] as usize] > 1)
                });
                if kept.len() == before {
                    break;
                }
            }
        }
        let removed_singletons = n - removed_zero_weight - kept.len();
        if kept.is_empty() {
            return Err(InputError::Empty {
                what: "rows left after rows of weight zero and singletons are removed",
            }
            .into());
        }
        let kept_weights: Option<Vec<f64>> = weights.map(|w| kept.iter().map(|&i| w[i]).collect());
        let levels: Vec<Level> = factors
            .par_iter()
            .zip(&input_levels)
            .map(|(f, &input)| {
                let mut compact = vec![u32::MAX; input];
                for &i in &kept {
                    compact[f[i] as usize] = 0;
                }
                let mut labels = Vec::new();
                for (code, slot) in compact.iter_mut().enumerate() {
                    if *slot == 0 {
                        *slot = u32::try_from(labels.len()).unwrap_or(u32::MAX);
                        labels.push(u32::try_from(code).unwrap_or(u32::MAX));
                    }
                }
                let codes: Vec<u32> = kept.iter().map(|&i| compact[f[i] as usize]).collect();
                let g = labels.len();
                let mut weight = vec![0.0; g];
                match &kept_weights {
                    Some(w) => {
                        for (k, &c) in codes.iter().enumerate() {
                            weight[c as usize] += w[k];
                        }
                    }
                    None => {
                        for &c in &codes {
                            weight[c as usize] += 1.0;
                        }
                    }
                }
                let plan = if g <= DENSE_LEVELS {
                    Plan::Dense
                } else {
                    let (bounds, order) = grouping(&codes, g);
                    match order {
                        None => Plan::Sorted { bounds },
                        Some(order) => Plan::Ordered {
                            bounds,
                            order,
                            others: Vec::new(),
                            weights: None,
                        },
                    }
                };
                Level {
                    codes,
                    labels,
                    input_levels: input,
                    weight,
                    plan,
                }
            })
            .collect();
        let mut levels = levels;
        for q in 0..levels.len() {
            if let Plan::Ordered { order, .. } = &levels[q].plan {
                let copies: Vec<Vec<u32>> = (0..levels.len())
                    .map(|r| {
                        if r == q {
                            Vec::new()
                        } else {
                            let codes = &levels[r].codes;
                            order.par_iter().map(|&k| codes[k as usize]).collect()
                        }
                    })
                    .collect();
                let w = kept_weights
                    .as_ref()
                    .map(|w| order.par_iter().map(|&k| w[k as usize]).collect());
                if let Plan::Ordered {
                    others, weights, ..
                } = &mut levels[q].plan
                {
                    *others = copies;
                    *weights = w;
                }
            }
        }
        let total_weight = kept_weights
            .as_ref()
            .map_or(kept.len() as f64, |w| blocked_total(w.len(), |k| w[k]));
        Ok(Self {
            n_input: n,
            kept,
            weights: kept_weights,
            total_weight,
            factors: levels,
            removed_zero_weight,
            removed_singletons,
        })
    }

    /// The number of input rows.
    #[must_use]
    pub const fn n_input(&self) -> usize {
        self.n_input
    }

    /// The input rows that enter, in increasing order.
    #[must_use]
    pub fn rows(&self) -> &[usize] {
        &self.kept
    }

    /// The number of rows that enter.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.kept.len()
    }

    /// The number of factors.
    #[must_use]
    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }

    /// The number of levels of each factor among the rows that enter.
    #[must_use]
    pub fn levels(&self) -> Vec<usize> {
        self.factors.iter().map(Level::levels).collect()
    }

    /// The rows removed for a weight of zero.
    #[must_use]
    pub const fn removed_zero_weight(&self) -> usize {
        self.removed_zero_weight
    }

    /// The rows removed as singletons.
    #[must_use]
    pub const fn removed_singletons(&self) -> usize {
        self.removed_singletons
    }

    /// The weights of the rows that enter, `None` for unit weights.
    #[must_use]
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// The level of each row that enters in factor `q`, numbered `0..levels`
    /// in the order of the caller's codes.
    #[must_use]
    pub fn codes(&self, q: usize) -> &[u32] {
        &self.factors[q].codes
    }

    /// The caller's code of each level of factor `q`.
    #[must_use]
    pub fn labels(&self, q: usize) -> &[u32] {
        &self.factors[q].labels
    }

    fn weight(&self, k: usize) -> f64 {
        self.weights.as_ref().map_or(1.0, |w| w[k])
    }

    /// Removes the fixed effects from each column of `columns`, which have
    /// one value per input row; only the rows that enter are read.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for an unusable control, a column of the wrong length
    /// or a value that is not finite on a row that enters, and
    /// [`Error::NotConverged`].
    pub fn demean(&self, columns: &[&[f64]], control: &Absorb) -> Result<Demeaned, Error> {
        control.check()?;
        for c in columns {
            if c.len() != self.n_input {
                return Err(length_error("column", self.n_input, c.len()));
            }
        }
        for c in columns {
            if let Some(&row) = self.kept.par_iter().find_first(|&&i| !c[i].is_finite()) {
                return Err(InputError::NotFinite {
                    what: "columns",
                    row,
                    column: None,
                }
                .into());
            }
        }
        let n = self.kept.len();
        let mut out = epistat_linalg::Mat::<f64>::zeros(n, columns.len());
        // The columns are demeaned in parallel, each by its own deterministic
        // iteration, so the order in which they finish does not matter.
        let results: Vec<Result<ColumnResult, Error>> = out
            .par_col_iter_mut()
            .zip(columns.par_iter())
            .enumerate()
            .map(|(j, (target, c))| {
                let target = target
                    .try_as_col_major_mut()
                    .ok_or(Error::Invalid("a column is not contiguous"))?
                    .as_slice_mut();
                target
                    .par_iter_mut()
                    .zip(self.kept.par_iter())
                    .for_each(|(t, &i)| *t = c[i]);
                self.demean_in_place(target, control, j)
            })
            .collect();
        let mut scale = Vec::with_capacity(columns.len());
        let mut iterations = Vec::with_capacity(columns.len());
        let mut coefficients = Vec::with_capacity(columns.len());
        for result in results {
            let column = result?;
            scale.push(column.scale);
            iterations.push(column.iterations);
            coefficients.push(column.coefficients);
        }
        Ok(Demeaned {
            columns: out,
            scale,
            iterations,
            coefficients,
        })
    }

    /// [`FixedEffects::demean`] of the columns of `x` and then `y`, with the
    /// columns of `x` demeaned in their own storage: each column keeps the
    /// rows that enter, in order, at its top, and `x` is cut to those rows.
    /// The values, the errors and their order are those of `demean` on the
    /// same columns. The returned [`Demeaned`] holds no columns; the demeaned
    /// `y` is the returned vector.
    pub(crate) fn demean_matrix(
        &self,
        x: &mut epistat_linalg::Mat<f64>,
        y: &[f64],
        control: &Absorb,
    ) -> Result<(Vec<f64>, Demeaned), Error> {
        control.check()?;
        let p = x.ncols();
        for found in [x.nrows(), y.len()] {
            if found != self.n_input {
                return Err(length_error("column", self.n_input, found));
            }
        }
        for j in 0..=p {
            let c: &[f64] = if j < p { x.col_as_slice(j) } else { y };
            if let Some(&row) = self.kept.par_iter().find_first(|&&i| !c[i].is_finite()) {
                return Err(InputError::NotFinite {
                    what: "columns",
                    row,
                    column: None,
                }
                .into());
            }
        }
        let n = self.kept.len();
        let compact = n < self.n_input;
        let mut y_tilde: Vec<f64> = self.kept.par_iter().map(|&i| y[i]).collect();
        let (results, last) = rayon::join(
            || {
                x.par_col_iter_mut()
                    .enumerate()
                    .map(|(j, column)| {
                        let column = column
                            .try_as_col_major_mut()
                            .ok_or(Error::Invalid("a column is not contiguous"))?
                            .as_slice_mut();
                        // The rows that enter are increasing, so each moves
                        // up or stays and none is overwritten before it moves.
                        if compact {
                            for (k, &i) in self.kept.iter().enumerate() {
                                column[k] = column[i];
                            }
                        }
                        self.demean_in_place(&mut column[..n], control, j)
                    })
                    .collect::<Vec<Result<ColumnResult, Error>>>()
            },
            || self.demean_in_place(&mut y_tilde, control, p),
        );
        let mut scale = Vec::with_capacity(p + 1);
        let mut iterations = Vec::with_capacity(p + 1);
        let mut coefficients = Vec::with_capacity(p + 1);
        for result in results.into_iter().chain(std::iter::once(last)) {
            let column = result?;
            scale.push(column.scale);
            iterations.push(column.iterations);
            coefficients.push(column.coefficients);
        }
        x.truncate(n, p);
        Ok((
            y_tilde,
            Demeaned {
                columns: epistat_linalg::Mat::<f64>::zeros(0, 0),
                scale,
                iterations,
                coefficients,
            },
        ))
    }

    /// The coefficients `alpha` that solve the normal equations
    /// `D' W D alpha = b` for given level sums `b`, one slice per factor and
    /// one value per level in the order of [`FixedEffects::levels`], by the
    /// iteration of [`FixedEffects::demean`]. With `b = D' W v` this is the
    /// projection of a column `v`, but `b` need not come from a column: it
    /// can carry sums over other rows, as the variance of the imputation
    /// estimator needs. `b` must lie in the range of `D' W D`, so that with
    /// two factors the sums over each connected component agree between the
    /// factors; otherwise the iteration does not converge.
    ///
    /// The criterion is that of [`Absorb`] with the scale `s` replaced by the
    /// largest level mean `|b_q[g]| / W_q[g]`: every level mean of
    /// `b - D' W D alpha` is at most `tolerance` times it. The coefficients
    /// are not normalised; [`FixedEffects::fitted`] gives `D alpha`, which is
    /// unique.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for an unusable control, sums of the wrong shape or a
    /// sum that is not finite, and [`Error::NotConverged`].
    pub fn solve_sums(&self, sums: &[&[f64]], control: &Absorb) -> Result<Vec<Vec<f64>>, Error> {
        control.check()?;
        let levels = self.levels();
        if sums.len() != levels.len() {
            return Err(length_error("level sums", levels.len(), sums.len()));
        }
        for (s, &g) in sums.iter().zip(&levels) {
            if s.len() != g {
                return Err(length_error("level sums", g, s.len()));
            }
            if let Some(row) = s.iter().position(|v| !v.is_finite()) {
                return Err(InputError::NotFinite {
                    what: "level sums",
                    row,
                    column: None,
                }
                .into());
            }
        }
        let owned: Vec<Vec<f64>> = sums.iter().map(|s| s.to_vec()).collect();
        let scale = owned
            .iter()
            .zip(&self.factors)
            .flat_map(|(s, f)| s.iter().zip(&f.weight).map(|(b, w)| (b / w).abs()))
            .fold(0.0_f64, f64::max);
        if scale == 0.0 {
            return Ok(levels.iter().map(|&g| vec![0.0; g]).collect());
        }
        if levels.len() == 1 {
            return Ok(vec![
                owned[0]
                    .iter()
                    .zip(&self.factors[0].weight)
                    .map(|(b, w)| b / w)
                    .collect(),
            ]);
        }
        let mut solver = Iteration::new(self, &owned, control.tolerance * scale);
        solver
            .solve(control)
            .map(|(point, _)| point)
            .ok_or(Error::NotConverged {
                iterations: control.max_iterations,
                column: Some(0),
            })
    }

    /// `sum_q alpha_q[g_q(i)]` for each row that enters, from coefficients by
    /// factor and level as [`FixedEffects::solve_sums`] returns them, added
    /// in factor order.
    ///
    /// # Panics
    ///
    /// If `alpha` does not have one vector per factor with one value per
    /// level.
    #[must_use]
    pub fn fitted(&self, alpha: &[Vec<f64>]) -> Vec<f64> {
        assert_eq!(alpha.len(), self.factors.len(), "one vector per factor");
        for (a, f) in alpha.iter().zip(&self.factors) {
            assert_eq!(a.len(), f.levels(), "one value per level");
        }
        (0..self.kept.len())
            .into_par_iter()
            .map(|k| {
                let mut total = 0.0;
                for (a, f) in alpha.iter().zip(&self.factors) {
                    total += a[f.codes[k] as usize];
                }
                total
            })
            .collect()
    }

    /// The fixed effects of column `column` of `demeaned`, normalised by the
    /// rule of the module documentation.
    ///
    /// # Panics
    ///
    /// If `column` is not a column of `demeaned`.
    #[must_use]
    pub fn fixed_effects(&self, demeaned: &Demeaned, column: usize) -> FixedEffectValues {
        self.normalise(&demeaned.coefficients[column])
    }

    /// One column, given on the rows that enter, replaced by its residual.
    pub(crate) fn demean_in_place(
        &self,
        v: &mut [f64],
        control: &Absorb,
        column: usize,
    ) -> Result<ColumnResult, Error> {
        let n = v.len();
        let mean = blocked_total(n, |k| self.weight(k) * v[k]) / self.total_weight;
        v.par_iter_mut().for_each(|x| *x -= mean);
        let s = (blocked_total(n, |k| self.weight(k) * v[k] * v[k]) / self.total_weight).sqrt();
        let q_count = self.factors.len();
        let mut alpha: Vec<Vec<f64>> = self.factors.iter().map(|f| vec![0.0; f.levels()]).collect();
        let mut iterations = 0;
        if s > 0.0 {
            let mut scratch = Vec::new();
            let sums: Vec<Vec<f64>> = self
                .factors
                .iter()
                .map(|f| {
                    let mut out = vec![0.0; f.levels()];
                    f.sums(|k| self.weight(k) * v[k], &mut scratch, &mut out);
                    out
                })
                .collect();
            if q_count == 1 {
                for (a, (s1, w)) in alpha[0]
                    .iter_mut()
                    .zip(sums[0].iter().zip(&self.factors[0].weight))
                {
                    *a = s1 / w;
                }
            } else {
                let mut solver = Iteration::new(self, &sums, control.tolerance * s);
                let (point, its) = solver.solve(control).ok_or(Error::NotConverged {
                    iterations: control.max_iterations,
                    column: Some(column),
                })?;
                iterations = its;
                alpha = point;
            }
            let codes: Vec<&[u32]> = self.factors.iter().map(|f| f.codes.as_slice()).collect();
            let alpha_ref = &alpha;
            v.par_chunks_mut(ROW_BLOCK)
                .enumerate()
                .for_each(|(block, chunk)| {
                    for (k, x) in chunk.iter_mut().enumerate() {
                        let i = block * ROW_BLOCK + k;
                        let mut fitted = 0.0;
                        for (q, c) in codes.iter().enumerate() {
                            fitted += alpha_ref[q][c[i] as usize];
                        }
                        *x -= fitted;
                    }
                });
        }
        // The centring mean belongs to the fixed effects; any one factor
        // carries it.
        for a in &mut alpha[0] {
            *a += mean;
        }
        Ok(ColumnResult {
            scale: s,
            iterations,
            coefficients: alpha,
        })
    }

    /// `out[g]` is the sum over the rows of level `g` of factor `q` of
    /// `w_i sum_{r in others} current[r][g_r(i)]`, the factors `others`
    /// in increasing order and without `q`.
    fn other_sums(
        &self,
        q: usize,
        others: &[usize],
        current: &[&[f64]],
        scratch: &mut Vec<f64>,
        out: &mut [f64],
    ) {
        let level = &self.factors[q];
        let (codes, weights): (Vec<&[u32]>, Option<&[f64]>) = match &level.plan {
            Plan::Ordered {
                others: copies,
                weights,
                ..
            } => (
                others.iter().map(|&r| copies[r].as_slice()).collect(),
                weights.as_deref(),
            ),
            _ => (
                others
                    .iter()
                    .map(|&r| self.factors[r].codes.as_slice())
                    .collect(),
                self.weights.as_deref(),
            ),
        };
        let coef: Vec<&[f64]> = others.iter().map(|&r| current[r]).collect();
        match (codes.as_slice(), coef.as_slice(), weights) {
            ([c0], [a0], None) => level.accumulate(|k| a0[c0[k] as usize], scratch, out),
            ([c0], [a0], Some(w)) => {
                level.accumulate(|k| w[k] * a0[c0[k] as usize], scratch, out);
            }
            ([c0, c1], [a0, a1], None) => {
                level.accumulate(|k| a0[c0[k] as usize] + a1[c1[k] as usize], scratch, out);
            }
            ([c0, c1], [a0, a1], Some(w)) => level.accumulate(
                |k| w[k] * (a0[c0[k] as usize] + a1[c1[k] as usize]),
                scratch,
                out,
            ),
            (codes, coef, w) => level.accumulate(
                |k| {
                    let mut total = 0.0;
                    for (c, a) in codes.iter().zip(coef) {
                        total += a[c[k] as usize];
                    }
                    w.map_or(total, |w| w[k] * total)
                },
                scratch,
                out,
            ),
        }
    }

    /// The fixed effects of the coefficients `alpha` (factor by factor, level
    /// by level), normalised by the rule of the module documentation.
    pub(crate) fn normalise(&self, alpha: &[Vec<f64>]) -> FixedEffectValues {
        match self.factors.len() {
            1 => FixedEffectValues {
                values: vec![self.by_code(0, &alpha[0])],
                references: vec![0],
                normalised: true,
                component: None,
            },
            2 => self.normalise_two(alpha),
            _ => self.normalise_many(alpha),
        }
    }

    /// Values by level of factor `q` placed at the caller's codes, `None` at
    /// codes without rows.
    fn by_code<T: Copy>(&self, q: usize, values: &[T]) -> Vec<Option<T>> {
        let level = &self.factors[q];
        let mut out = vec![None; level.input_levels];
        for (&label, &v) in level.labels.iter().zip(values) {
            out[label as usize] = Some(v);
        }
        out
    }

    /// The rows of each level of factor `q`.
    fn members(&self, q: usize) -> Members {
        let codes = &self.factors[q].codes;
        let (bounds, order) = grouping(codes, self.factors[q].levels());
        let order = order.unwrap_or_else(|| {
            (0..codes.len())
                .map(|k| u32::try_from(k).unwrap_or(u32::MAX))
                .collect()
        });
        Members { bounds, order }
    }

    /// The fitted sum `sum_q alpha_q[g_q(k)]` of row `k`, added in factor
    /// order as in [`FixedEffects::fitted`].
    fn fitted_row(&self, alpha: &[Vec<f64>], k: usize) -> f64 {
        let mut total = 0.0;
        for (a, f) in alpha.iter().zip(&self.factors) {
            total += a[f.codes[k] as usize];
        }
        total
    }

    /// Two factors: the connected components of the levels joined by rows,
    /// found by union-find and numbered in the order of their first rows; in
    /// each, the level of the second factor on the first row is the
    /// reference, and a breadth-first pass from it fixes every other level
    /// from one row's fitted sum.
    fn normalise_two(&self, alpha: &[Vec<f64>]) -> FixedEffectValues {
        let codes = [&self.factors[0].codes, &self.factors[1].codes];
        let levels = [self.factors[0].levels(), self.factors[1].levels()];
        let n = self.kept.len();
        // The levels of the first factor are nodes 0..G_0, those of the
        // second follow.
        let node = |q: usize, g: usize| if q == 0 { g } else { levels[0] + g };
        let mut parent: Vec<usize> = (0..levels[0] + levels[1]).collect();
        for k in 0..n {
            let a = find_root(&mut parent, node(0, codes[0][k] as usize));
            let b = find_root(&mut parent, node(1, codes[1][k] as usize));
            parent[a.max(b)] = a.min(b);
        }
        let mut number = vec![u32::MAX; parent.len()];
        let mut first_rows = Vec::new();
        for (k, &c) in codes[0].iter().enumerate() {
            let root = find_root(&mut parent, node(0, c as usize));
            if number[root] == u32::MAX {
                number[root] = u32::try_from(first_rows.len()).unwrap_or(u32::MAX);
                first_rows.push(k);
            }
        }
        let mut component = [Vec::new(), Vec::new()];
        for (q, c) in component.iter_mut().enumerate() {
            *c = (0..levels[q])
                .map(|g| number[find_root(&mut parent, node(q, g))])
                .collect();
        }
        let members = [self.members(0), self.members(1)];
        let mut values = [vec![0.0; levels[0]], vec![0.0; levels[1]]];
        let mut known = [vec![false; levels[0]], vec![false; levels[1]]];
        let mut queue = VecDeque::new();
        for &k in &first_rows {
            let reference = codes[1][k] as usize;
            known[1][reference] = true;
            queue.push_back((1, reference));
            while let Some((q, g)) = queue.pop_front() {
                let other = 1 - q;
                for &row in members[q].rows(g) {
                    let row = row as usize;
                    let h = codes[other][row] as usize;
                    if !known[other][h] {
                        values[other][h] = self.fitted_row(alpha, row) - values[q][g];
                        known[other][h] = true;
                        queue.push_back((other, h));
                    }
                }
            }
        }
        FixedEffectValues {
            values: vec![self.by_code(0, &values[0]), self.by_code(1, &values[1])],
            references: vec![0, first_rows.len()],
            normalised: true,
            component: Some(vec![
                self.by_code(0, &component[0]),
                self.by_code(1, &component[1]),
            ]),
        }
    }

    /// Three or more factors: the propagation of the module documentation,
    /// with a running bound on the rounding of each coefficient, against
    /// which the fitted sums are checked.
    fn normalise_many(&self, alpha: &[Vec<f64>]) -> FixedEffectValues {
        let factors = self.factors.len();
        let levels = self.levels();
        let mut state = Propagation {
            fe: self,
            alpha,
            members: (0..factors).map(|q| self.members(q)).collect(),
            values: levels.iter().map(|&g| vec![0.0; g]).collect(),
            error: levels.iter().map(|&g| vec![0.0; g]).collect(),
            known: levels.iter().map(|&g| vec![false; g]).collect(),
            unknown: levels.iter().sum(),
            count: vec![0; self.kept.len()],
            ready: VecDeque::new(),
            waiting: vec![BinaryHeap::new(); factors - 1],
            cursor: 0,
        };
        let mut references = vec![0usize; factors];
        loop {
            while let Some(row) = state.ready.pop_front() {
                if state.count[row as usize] as usize == factors - 1 {
                    state.determine(row as usize);
                }
            }
            if state.unknown == 0 {
                break;
            }
            let row = state.next_reference_row();
            let unknown: Vec<usize> = (0..factors)
                .filter(|&q| !state.known[q][self.factors[q].codes[row] as usize])
                .collect();
            for &q in &unknown[1..] {
                state.set(q, self.factors[q].codes[row] as usize, 0.0, 0.0);
                references[q] += 1;
            }
            state.determine(row);
        }
        let normalised = (0..self.kept.len()).all(|k| state.reproduces(k));
        let values = if normalised { &state.values } else { alpha };
        FixedEffectValues {
            values: (0..factors).map(|q| self.by_code(q, &values[q])).collect(),
            references,
            normalised,
            component: None,
        }
    }
}

/// The rows of each level of a factor: those of level `g` are
/// `order[bounds[g]..bounds[g + 1]]`, in increasing order.
struct Members {
    bounds: Vec<usize>,
    order: Vec<u32>,
}

impl Members {
    fn rows(&self, g: usize) -> &[u32] {
        &self.order[self.bounds[g]..self.bounds[g + 1]]
    }
}

/// The root of `x` in the union-find forest `parent`, halving the path.
fn find_root(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// `gamma_m = m u / (1 - m u)`, `u` the unit roundoff: a sum of `m + 1`
/// terms in floating point is within `gamma_m` times the sum of their
/// magnitudes of the exact sum (Higham, Accuracy and Stability of Numerical
/// Algorithms, 2nd ed., section 4.2).
fn gamma(m: usize) -> f64 {
    let mu = m as f64 * f64::EPSILON / 2.0;
    mu / (1.0 - mu)
}

/// The state of the normalisation of three or more factors.
struct Propagation<'a> {
    fe: &'a FixedEffects,
    /// The solver's coefficients, whose fitted sums the values reproduce.
    alpha: &'a [Vec<f64>],
    members: Vec<Members>,
    /// The normalised coefficients, by factor and level.
    values: Vec<Vec<f64>>,
    /// A bound on the rounding error of each value: the distance to the
    /// value that exact arithmetic gives from the same references.
    error: Vec<Vec<f64>>,
    known: Vec<Vec<bool>>,
    /// The number of coefficients not yet known.
    unknown: usize,
    /// The number of known coefficients of each row.
    count: Vec<u32>,
    /// Rows that reached one unknown coefficient, in the order they did.
    ready: VecDeque<u32>,
    /// For `c` in `1..Q - 1`, rows that reached `c` known coefficients; an
    /// entry is stale once the row's count has moved on.
    waiting: Vec<BinaryHeap<Reverse<u32>>>,
    /// No row before this one has no known coefficient.
    cursor: usize,
}

impl Propagation<'_> {
    fn code(&self, q: usize, row: usize) -> usize {
        self.fe.factors[q].codes[row] as usize
    }

    /// Sets level `g` of factor `q` to `value`, with the error bound `error`.
    fn set(&mut self, q: usize, g: usize, value: f64, error: f64) {
        self.values[q][g] = value;
        self.error[q][g] = error;
        self.known[q][g] = true;
        self.unknown -= 1;
        let factors = self.known.len();
        let Self {
            members,
            count,
            ready,
            waiting,
            ..
        } = self;
        for &row in members[q].rows(g) {
            let c = &mut count[row as usize];
            *c += 1;
            let c = *c as usize;
            if c == factors - 1 {
                ready.push_back(row);
            } else if c < factors - 1 {
                waiting[c].push(Reverse(row));
            }
        }
    }

    /// Fixes the one unknown coefficient of `row` from its fitted sum.
    fn determine(&mut self, row: usize) {
        let factors = self.known.len();
        let Some(q) = (0..factors).find(|&q| !self.known[q][self.code(q, row)]) else {
            return;
        };
        let fitted = self.fe.fitted_row(self.alpha, row);
        let (mut value, mut error, mut magnitude) = (fitted, 0.0, fitted.abs());
        for r in (0..factors).filter(|&r| r != q) {
            let g = self.code(r, row);
            value -= self.values[r][g];
            error += self.error[r][g];
            magnitude += self.values[r][g].abs();
        }
        let error = error + gamma(factors) * magnitude;
        self.set(q, self.code(q, row), value, error);
    }

    /// The first row, in row order, among those with the most known
    /// coefficients short of all but one. Called only when no row has one
    /// unknown coefficient and some coefficient is unknown, so that such a
    /// row exists.
    fn next_reference_row(&mut self) -> usize {
        for c in (1..self.waiting.len()).rev() {
            while let Some(&Reverse(row)) = self.waiting[c].peek() {
                if self.count[row as usize] as usize == c {
                    return row as usize;
                }
                self.waiting[c].pop();
            }
        }
        while self.count[self.cursor] != 0 {
            self.cursor += 1;
        }
        self.cursor
    }

    /// Whether the values reproduce the fitted sum of `row` within the
    /// rounding their error bounds and the sum allow. When the references
    /// leave the values consistent, the values computed in exact arithmetic
    /// reproduce every fitted sum exactly; each computed value lies within
    /// its bound of them, and the sum of `Q` values adds at most `gamma_Q`
    /// times their magnitudes.
    fn reproduces(&self, row: usize) -> bool {
        let factors = self.known.len();
        let (mut total, mut error, mut magnitude) = (0.0, 0.0, 0.0);
        for q in 0..factors {
            let g = self.code(q, row);
            total += self.values[q][g];
            error += self.error[q][g];
            magnitude += self.values[q][g].abs();
        }
        let fitted = self.fe.fitted_row(self.alpha, row);
        (total - fitted).abs() <= error + gamma(factors) * magnitude
    }
}

/// The result of one column.
pub(crate) struct ColumnResult {
    pub(crate) scale: f64,
    pub(crate) iterations: usize,
    pub(crate) coefficients: Vec<Vec<f64>>,
}

/// The solution of the normal equations for one column with two or more
/// factors by conjugate gradients.
struct Iteration<'a> {
    fe: &'a FixedEffects,
    /// The level sums of the column, one vector per factor.
    sums: &'a [Vec<f64>],
    /// The bound on the level means of the residual.
    bound: f64,
    /// The factor eliminated exactly, the one with the most levels (the
    /// first of them on a tie).
    first: usize,
    /// The other factors, in order; the state is their coefficients, one
    /// after the other.
    state: Vec<usize>,
    offsets: Vec<usize>,
    scratch: Vec<f64>,
}

/// The largest absolute value of `v`.
fn largest(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, &a| m.max(a.abs()))
}

/// `sum_k a_k b_k` by blocks added in block order.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    blocked_total(a.len(), |k| a[k] * b[k])
}

impl<'a> Iteration<'a> {
    fn new(fe: &'a FixedEffects, sums: &'a [Vec<f64>], bound: f64) -> Self {
        let levels = fe.levels();
        let mut first = 0;
        for (q, &g) in levels.iter().enumerate() {
            if g > levels[first] {
                first = q;
            }
        }
        let state: Vec<usize> = (0..levels.len()).filter(|&q| q != first).collect();
        let mut offsets = vec![0];
        for &q in &state {
            offsets.push(offsets.last().copied().unwrap_or(0) + levels[q]);
        }
        Self {
            fe,
            sums,
            bound,
            first,
            state,
            offsets,
            scratch: Vec::new(),
        }
    }

    /// Runs the solver of `control` to convergence; the point and the
    /// iterations, `None` when the iterations run out.
    fn solve(&mut self, control: &Absorb) -> Option<(Vec<Vec<f64>>, usize)> {
        match control.method {
            AbsorbMethod::ConjugateGradient => self.conjugate_gradient(control.max_iterations),
        }
    }

    fn dimension(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0)
    }

    /// The level means of factor `q` of what the other factors' coefficients
    /// `current` leave of the column (its own are not read); with
    /// `homogeneous`, as if the column's level sums were zero.
    fn project_with(&mut self, q: usize, current: &[&[f64]], out: &mut [f64], homogeneous: bool) {
        let fe = self.fe;
        let others: Vec<usize> = (0..fe.factors.len()).filter(|&r| r != q).collect();
        fe.other_sums(q, &others, current, &mut self.scratch, out);
        if homogeneous {
            for (o, w) in out.iter_mut().zip(&fe.factors[q].weight) {
                *o = -*o / w;
            }
        } else {
            for ((o, s), w) in out.iter_mut().zip(&self.sums[q]).zip(&fe.factors[q].weight) {
                *o = (s - *o) / w;
            }
        }
    }

    /// The coefficients of every factor as slices, from the first factor's
    /// and the state.
    fn view<'b>(&self, first: &'b [f64], x: &'b [f64]) -> Vec<&'b [f64]> {
        let mut out: Vec<&[f64]> = vec![&[]; self.fe.factors.len()];
        out[self.first] = first;
        for (s, &q) in self.state.iter().enumerate() {
            out[q] = &x[self.offsets[s]..self.offsets[s + 1]];
        }
        out
    }

    /// The point `(first, x)` as coefficients of every factor.
    fn point(&self, first: Vec<f64>, x: &[f64]) -> Vec<Vec<f64>> {
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); self.fe.factors.len()];
        out[self.first] = first;
        for (s, &q) in self.state.iter().enumerate() {
            out[q] = x[self.offsets[s]..self.offsets[s + 1]].to_vec();
        }
        out
    }

    /// The coefficients of the first factor from the state `x`, and the
    /// level means of the residual of the other factors at that
    /// point, in the layout of the state. With `homogeneous` the column's
    /// sums are dropped, and the means are `-W^{-1} S x`.
    fn means(&mut self, x: &[f64], homogeneous: bool) -> (Vec<f64>, Vec<f64>) {
        let levels = self.fe.levels();
        let zero = vec![0.0; levels[self.first]];
        let mut first = vec![0.0; levels[self.first]];
        {
            let current = self.view(&zero, x);
            self.project_with(self.first, &current, &mut first, homogeneous);
        }
        let mut means = vec![0.0; self.dimension()];
        for s in 0..self.state.len() {
            let q = self.state[s];
            let range = self.offsets[s]..self.offsets[s + 1];
            let mut out = vec![0.0; levels[q]];
            let current = self.view(&first, x);
            self.project_with(q, &current, &mut out, homogeneous);
            for ((m, o), old) in means[range.clone()].iter_mut().zip(&out).zip(&x[range]) {
                *m = o - old;
            }
        }
        (first, means)
    }

    /// Conjugate gradients on the equations of the state with the first
    /// factor eliminated, `S x = c`, preconditioned by the level weights.
    fn conjugate_gradient(&mut self, max_iterations: usize) -> Option<(Vec<Vec<f64>>, usize)> {
        let weight: Vec<f64> = self
            .state
            .iter()
            .flat_map(|&q| self.fe.factors[q].weight.iter().copied())
            .collect();
        let mut x = vec![0.0; self.dimension()];
        let (first, mut z) = self.means(&x, false);
        if largest(&z) <= self.bound {
            return Some((self.point(first, &x), 0));
        }
        let mut r: Vec<f64> = z.iter().zip(&weight).map(|(z, w)| z * w).collect();
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let mut restarted = false;
        for iteration in 1..=max_iterations {
            let (_, h) = self.means(&p, true);
            let sp: Vec<f64> = h.iter().zip(&weight).map(|(h, w)| -(h * w)).collect();
            let curvature = dot(&p, &sp);
            let step = rz / curvature;
            let usable = curvature > 0.0 && step.is_finite();
            if usable {
                for (xk, pk) in x.iter_mut().zip(&p) {
                    *xk += step * pk;
                }
                for ((rk, zk), (sk, w)) in
                    r.iter_mut().zip(z.iter_mut()).zip(sp.iter().zip(&weight))
                {
                    *rk -= step * sk;
                    *zk = *rk / w;
                }
            }
            if !usable || largest(&z) <= self.bound {
                // The recursive residual can drift from the residual at the
                // point; the criterion is checked on the latter.
                let (first, true_means) = self.means(&x, false);
                if largest(&true_means) <= self.bound {
                    return Some((self.point(first, &x), iteration));
                }
                if !usable && restarted {
                    return None;
                }
                restarted = !usable;
                z = true_means;
                r = z.iter().zip(&weight).map(|(z, w)| z * w).collect();
                p.clone_from(&z);
                rz = dot(&r, &z);
                continue;
            }
            restarted = false;
            let next = dot(&r, &z);
            let beta = next / rz;
            rz = next;
            for (pk, zk) in p.iter_mut().zip(&z) {
                *pk = zk + beta * *pk;
            }
        }
        None
    }
}

/// The error of a value of `found` entries where `expected` were needed.
const fn length_error(what: &'static str, expected: usize, found: usize) -> Error {
    Error::Input(InputError::Length {
        what,
        expected,
        found,
    })
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::cast_possible_truncation,
        reason = "test panels have fewer than 2^32 rows"
    )]

    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn singleton_removal_repeats_until_none_is_left() {
        // Row 3 is the only row of unit 2. Removing it leaves period 9 with
        // one row, row 4, and removing that leaves unit 1 with row 2 alone.
        let unit = [0u32, 0, 1, 2, 1, 3, 3];
        let time = [0u32, 1, 0, 9, 9, 0, 1];
        let fe = FixedEffects::new(&[&unit, &time], None, Singletons::Remove).unwrap();
        assert_eq!(fe.rows(), &[0, 1, 5, 6]);
        assert_eq!(fe.removed_singletons(), 3);
        assert_eq!(fe.levels(), vec![2, 2]);
    }

    #[test]
    fn a_balanced_panel_converges_in_one_sweep_to_the_two_way_residual() {
        let (units, periods) = (40u32, 6u32);
        let unit: Vec<u32> = (0..units * periods).map(|i| i / periods).collect();
        let time: Vec<u32> = (0..units * periods).map(|i| i % periods).collect();
        let mut state = 7u64;
        let v: Vec<f64> = (0..unit.len())
            .map(|_| (xorshift(&mut state) >> 11) as f64 / (1u64 << 53) as f64)
            .collect();
        let fe = FixedEffects::new(&[&unit, &time], None, Singletons::Keep).unwrap();
        let out = fe.demean(&[&v], &Absorb::new()).unwrap();
        let n = v.len() as f64;
        let mean: f64 = v.iter().sum::<f64>() / n;
        for i in 0..v.len() {
            let u = unit[i];
            let t = time[i];
            let mu: f64 = (0..periods)
                .map(|s| v[(u * periods + s) as usize])
                .sum::<f64>()
                / f64::from(periods);
            let mt: f64 = (0..units)
                .map(|s| v[(s * periods + t) as usize])
                .sum::<f64>()
                / f64::from(units);
            let exact = v[i] - mu - mt + mean;
            // Sums of at most 240 values of size 1: 1e-13 is far above their
            // rounding.
            assert!((out.columns[(i, 0)] - exact).abs() < 1e-13);
        }
        assert!(out.iterations[0] <= 1);
    }

    /// A balanced panel of `units` units and `periods` periods, its rows in
    /// unit order or shuffled, with a column and its exact two-way residual.
    fn large_panel(
        units: u32,
        periods: u32,
        shuffle: bool,
    ) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>) {
        let n = (units * periods) as usize;
        let mut state = 99u64;
        let value: Vec<f64> = (0..n)
            .map(|_| (xorshift(&mut state) >> 11) as f64 / (1u64 << 53) as f64)
            .collect();
        let unit_mean: Vec<f64> = (0..units as usize)
            .map(|u| {
                (0..periods as usize)
                    .map(|t| value[u * periods as usize + t])
                    .sum::<f64>()
                    / f64::from(periods)
            })
            .collect();
        let time_mean: Vec<f64> = (0..periods as usize)
            .map(|t| {
                (0..units as usize)
                    .map(|u| value[u * periods as usize + t])
                    .sum::<f64>()
                    / f64::from(units)
            })
            .collect();
        let mean = value.iter().sum::<f64>() / n as f64;
        let mut order: Vec<usize> = (0..n).collect();
        if shuffle {
            for i in (1..n).rev() {
                let j = (xorshift(&mut state) % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
        }
        let p = periods as usize;
        let unit: Vec<u32> = order.iter().map(|&i| (i / p) as u32).collect();
        let time: Vec<u32> = order.iter().map(|&i| (i % p) as u32).collect();
        let v: Vec<f64> = order.iter().map(|&i| value[i]).collect();
        let exact: Vec<f64> = order
            .iter()
            .map(|&i| value[i] - unit_mean[i / p] - time_mean[i % p] + mean)
            .collect();
        (unit, time, v, exact)
    }

    #[test]
    fn factors_with_many_levels_sorted_or_not_give_the_two_way_residual() {
        // 20 000 units are more than DENSE_LEVELS, so the units are summed
        // level by level: in place when the rows are in unit order, through
        // the copied codes when they are shuffled.
        for shuffle in [false, true] {
            let (unit, time, v, exact) = large_panel(20_000, 3, shuffle);
            let fe = FixedEffects::new(&[&unit, &time], None, Singletons::Keep).unwrap();
            assert!(matches!(
                (&fe.factors[0].plan, shuffle),
                (Plan::Sorted { .. }, false) | (Plan::Ordered { .. }, true)
            ));
            let out = fe.demean(&[&v], &Absorb::new()).unwrap();
            for (k, e) in exact.iter().enumerate() {
                // Means of 3 and of 20 000 values in [0, 1): far below 1e-12.
                assert!((out.columns[(k, 0)] - e).abs() < 1e-12, "{shuffle} {k}");
            }
        }
    }

    /// Workers on shuffled rows, five years each, who move to a neighbouring
    /// one of `firms` firms on a line with probability 0.1 a year: a chain.
    fn chain(workers: u32, firms: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f64>) {
        let mut state = 5u64;
        let mut rows = Vec::new();
        for i in 0..workers {
            let mut firm = (xorshift(&mut state) % u64::from(firms)) as u32;
            let start = (xorshift(&mut state) % 6) as u32;
            for s in 0..5 {
                if xorshift(&mut state).is_multiple_of(10) {
                    firm = if xorshift(&mut state).is_multiple_of(2) {
                        firm.saturating_sub(1)
                    } else {
                        (firm + 1).min(firms - 1)
                    };
                }
                let v = (xorshift(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
                rows.push((i, firm, start + s, v + f64::from(firm) / f64::from(firms)));
            }
        }
        for i in (1..rows.len()).rev() {
            let j = (xorshift(&mut state) % (i as u64 + 1)) as usize;
            rows.swap(i, j);
        }
        (
            rows.iter().map(|r| r.0).collect(),
            rows.iter().map(|r| r.1).collect(),
            rows.iter().map(|r| r.2).collect(),
            rows.iter().map(|r| r.3).collect(),
        )
    }

    #[test]
    fn a_chain_meets_the_criterion_and_does_not_depend_on_the_number_of_threads() {
        // 20 000 workers are more than DENSE_LEVELS and their rows are
        // shuffled, so they are summed through the copied codes; the years
        // and the 30 firms are summed by dense blocks. Few workers move, and
        // only to a neighbouring firm, so the factors are poorly connected.
        let (worker, firm, year, v) = chain(20_000, 30);
        let w: Vec<f64> = (0..v.len()).map(|i| 0.5 + (i % 5) as f64 / 4.0).collect();
        let control = Absorb::new();
        for factors in [vec![&worker, &firm], vec![&worker, &firm, &year]] {
            let factors: Vec<&[u32]> = factors.iter().map(|f| f.as_slice()).collect();
            let fe = FixedEffects::new(&factors, Some(&w), Singletons::Remove).unwrap();
            let out = fe.demean(&[&v], &control).unwrap();
            let r = out.columns.col_as_slice(0);
            let kept = fe.weights().unwrap();
            let largest_r = largest(r);
            for q in 0..fe.n_factors() {
                let g = fe.levels()[q];
                let (mut sum, mut weight, mut count) =
                    (vec![0.0; g], vec![0.0; g], vec![0usize; g]);
                for (k, &c) in fe.codes(q).iter().enumerate() {
                    sum[c as usize] += kept[k] * r[k];
                    weight[c as usize] += kept[k];
                    count[c as usize] += 1;
                }
                for c in 0..g {
                    // Recomputed here, a level mean carries the rounding of
                    // its sum of `count` terms of at most 1.25 |r|: at most
                    // count * 1.25 * eps * |r|_max over a weight of at least
                    // 0.5 * count, so 2.5 eps |r|_max.
                    let mean = sum[c] / weight[c];
                    let allowed = control.tolerance * out.scale[0] + 2.5 * f64::EPSILON * largest_r;
                    assert!(mean.abs() <= allowed, "factor {q} level {c}: {mean:e}");
                }
            }
            assert!(out.iterations[0] > 10);
        }
        epistat_testkit::assert_same_on_threads(|| {
            let fe =
                FixedEffects::new(&[&worker, &firm, &year], Some(&w), Singletons::Remove).unwrap();
            let out = fe.demean(&[&v], &control).unwrap();
            let mut bits = out.columns.col_as_slice(0).to_vec();
            bits.push(out.iterations[0] as f64);
            let fe = FixedEffects::new(&[&worker, &firm], Some(&w), Singletons::Remove).unwrap();
            let out = fe.demean(&[&v], &control).unwrap();
            bits.extend(out.columns.col_as_slice(0));
            let fx = fe.fixed_effects(&out, 0);
            bits.extend(fx.values.iter().flatten().flatten());
            bits.extend(fx.references.iter().map(|&r| r as f64));
            bits
        });
    }

    #[test]
    fn the_level_sums_do_not_depend_on_the_number_of_threads() {
        let (unit, time, v, _) = large_panel(20_000, 3, true);
        // A third factor of 50 levels crossing the units. Chains of units and
        // firms with many levels each converge slowly (fixest takes 26 000
        // iterations at 1e-10 for 17 000 firms of three rows); this design is
        // well connected.
        let firm: Vec<u32> = unit
            .iter()
            .zip(&time)
            .map(|(u, t)| (u * 7 + t * 13) % 50)
            .collect();
        let w: Vec<f64> = (0..v.len()).map(|i| 0.5 + (i % 7) as f64 / 4.0).collect();
        epistat_testkit::assert_same_on_threads(|| {
            let fe =
                FixedEffects::new(&[&unit, &time, &firm], Some(&w), Singletons::Remove).unwrap();
            let out = fe.demean(&[&v], &Absorb::new()).unwrap();
            let mut bits = out.columns.col_as_slice(0).to_vec();
            bits.extend(
                fe.normalise(&out.coefficients[0])
                    .values
                    .iter()
                    .flatten()
                    .flatten(),
            );
            bits
        });
    }

    /// The normalised values of factor `q` at the caller's codes `0..`, which
    /// all have rows in these tests.
    fn values(fx: &FixedEffectValues, q: usize) -> Vec<f64> {
        fx.values[q].iter().map(|v| v.unwrap()).collect()
    }

    /// Checks that the values reproduce the fitted sums of `alpha` exactly:
    /// with whole numbers far below 2^53 every sum and difference is exact.
    fn assert_reproduces_exactly(fe: &FixedEffects, alpha: &[Vec<f64>], fx: &FixedEffectValues) {
        let fitted = fe.fitted(alpha);
        for (k, &f) in fitted.iter().enumerate() {
            let codes: Vec<u32> = (0..fe.n_factors())
                .map(|q| fe.labels(q)[fe.codes(q)[k] as usize])
                .collect();
            assert_eq!(fx.sum(&codes), Some(f), "row {k}");
        }
    }

    #[test]
    fn two_components_have_two_references_and_reproduce_the_fitted_sums_exactly() {
        // Units 2 and 3 with periods 2 and 3 form the component of the first
        // row, units 0 and 1 with periods 0 and 1 the other. The references
        // are the periods of each component's first row, 2 and 0.
        let unit = [2u32, 0, 0, 1, 3, 3];
        let time = [2u32, 0, 1, 1, 2, 3];
        let fe = FixedEffects::new(&[&unit, &time], None, Singletons::Keep).unwrap();
        let alpha = vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]];
        let fx = fe.normalise(&alpha);
        assert_eq!(fx.references, vec![0, 2]);
        assert!(fx.normalised);
        // Component 0: t2 = 0, u2 = 3 + 30, u3 = 4 + 30, t3 = 4 + 40 - 34.
        // Component 1: t0 = 0, u0 = 1 + 10, t1 = 1 + 20 - 11, u1 = 2 + 20 - 10.
        assert_eq!(values(&fx, 0), vec![11.0, 12.0, 33.0, 34.0]);
        assert_eq!(values(&fx, 1), vec![0.0, 10.0, 0.0, 10.0]);
        assert_reproduces_exactly(&fe, &alpha, &fx);
        let component = fx.component.as_ref().unwrap();
        assert_eq!(component[0], vec![Some(1), Some(1), Some(0), Some(0)]);
        assert_eq!(component[1], vec![Some(1), Some(1), Some(0), Some(0)]);
        // A unit and a period of different components: not identified.
        assert_eq!(fx.sum(&[0, 2]), None);
        assert_eq!(fx.sum(&[0, 7]), None);
    }

    #[test]
    fn three_factors_with_a_nested_one_follow_the_propagation() {
        // Workers, firms and sectors, with firms 0 and 1 in sector 0 and
        // firms 2 and 3 in sector 1: the sectors are nested in the firms.
        // The redundancies are 3: one of the connected worker-firm graph and
        // the two sectors. The first row sets firm 0 and sector 0 to zero and
        // fixes worker 0; rows 1, 2 and 7 then fix firm 1, worker 1 and
        // worker 3. Rows 3 and 6 have one coefficient known; the first, row
        // 3, sets sector 1 to zero and fixes firm 2, and rows 4 and 5 fix
        // worker 2 and firm 3. References 0, 1 and 2, as many as the
        // redundancies, so the values reproduce row 6 as well.
        let worker = [0u32, 0, 1, 1, 2, 2, 3, 3];
        let firm = [0u32, 1, 1, 2, 2, 3, 3, 0];
        let sector = [0u32, 0, 0, 1, 1, 1, 1, 0];
        let fe = FixedEffects::new(&[&worker, &firm, &sector], None, Singletons::Keep).unwrap();
        let alpha = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![100.0, 200.0],
        ];
        let fx = fe.normalise(&alpha);
        assert_eq!(fx.references, vec![0, 1, 2]);
        assert!(fx.normalised);
        assert_eq!(values(&fx, 0), vec![111.0, 112.0, 113.0, 114.0]);
        assert_eq!(values(&fx, 1), vec![0.0, 10.0, 120.0, 130.0]);
        assert_eq!(values(&fx, 2), vec![0.0, 0.0]);
        assert_reproduces_exactly(&fe, &alpha, &fx);
    }

    #[test]
    fn more_references_than_redundancies_leave_the_solver_coefficients() {
        // Two levels in each of three factors on four rows: the dummies have
        // rank 4 of 6, so 2 redundancies. The first row sets b0 and c0 to
        // zero and fixes a0; no row then has one unknown coefficient, and
        // row 1, the first with one known, sets c1 to zero and fixes a1; row
        // 2 fixes b1. Three references: row 3 is not reproduced
        // (a0 + b1 + c1 = 11 - 2 + 0 against 17).
        let a = [0u32, 1, 1, 0];
        let b = [0u32, 0, 1, 1];
        let c = [0u32, 1, 0, 1];
        let fe = FixedEffects::new(&[&a, &b, &c], None, Singletons::Keep).unwrap();
        let alpha = vec![vec![1.0, 2.0], vec![3.0, 5.0], vec![7.0, 11.0]];
        let fx = fe.normalise(&alpha);
        assert_eq!(fx.references, vec![0, 1, 2]);
        assert!(!fx.normalised);
        assert_eq!(values(&fx, 0), alpha[0]);
        assert_eq!(values(&fx, 1), alpha[1]);
        assert_eq!(values(&fx, 2), alpha[2]);
    }

    /// A fraction of whole numbers in lowest terms with a positive
    /// denominator, for the exact projections of the small designs.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Ratio(i128, i128);

    impl Ratio {
        fn new(n: i128, d: i128) -> Self {
            let (mut a, mut b) = (n.abs(), d.abs());
            while b != 0 {
                (a, b) = (b, a % b);
            }
            let g = a.max(1) * d.signum();
            Self(n / g, d / g)
        }
        fn int(n: i128) -> Self {
            Self(n, 1)
        }
        fn sub(self, o: Self) -> Self {
            Self::new(self.0 * o.1 - o.0 * self.1, self.1 * o.1)
        }
        fn mul(self, o: Self) -> Self {
            Self::new(self.0 * o.0, self.1 * o.1)
        }
        fn div(self, o: Self) -> Self {
            Self::new(self.0 * o.1, self.1 * o.0)
        }
        fn to_f64(self) -> f64 {
            self.0 as f64 / self.1 as f64
        }
    }

    /// The exact weighted residual of `v` on the dummies of `factors`, by
    /// Gauss-Jordan elimination of the normal equations in exact arithmetic,
    /// and the number of redundant coefficients.
    fn exact_residual(factors: &[&[u32]], w: &[i128], v: &[i128]) -> (Vec<Ratio>, usize) {
        let levels: Vec<usize> = factors
            .iter()
            .map(|f| f.iter().max().map_or(0, |&m| m as usize + 1))
            .collect();
        let offset: Vec<usize> = levels
            .iter()
            .scan(0, |s, &g| {
                *s += g;
                Some(*s - g)
            })
            .collect();
        let m: usize = levels.iter().sum();
        let column = |k: usize, q: usize| offset[q] + factors[q][k] as usize;
        let mut a = vec![vec![Ratio::int(0); m + 1]; m];
        for k in 0..v.len() {
            for q in 0..factors.len() {
                for r in 0..factors.len() {
                    let (i, j) = (column(k, q), column(k, r));
                    a[i][j] = Ratio::int(a[i][j].0 + w[k]);
                }
                let i = column(k, q);
                a[i][m] = Ratio::int(a[i][m].0 + w[k] * v[k]);
            }
        }
        let mut pivots = Vec::new();
        let mut row = 0;
        for col in 0..m {
            let Some(p) = (row..m).find(|&i| a[i][col].0 != 0) else {
                continue;
            };
            a.swap(row, p);
            let pivot = a[row][col];
            for x in &mut a[row] {
                *x = x.div(pivot);
            }
            for i in 0..m {
                if i != row && a[i][col].0 != 0 {
                    let factor = a[i][col];
                    for j in 0..=m {
                        a[i][j] = a[i][j].sub(factor.mul(a[row][j]));
                    }
                }
            }
            pivots.push(col);
            row += 1;
        }
        // The free coefficients are zero; each pivot's is its right side.
        let mut alpha = vec![Ratio::int(0); m];
        for (i, &col) in pivots.iter().enumerate() {
            alpha[col] = a[i][m];
        }
        let residual = (0..v.len())
            .map(|k| (0..factors.len()).fold(Ratio::int(v[k]), |r, q| r.sub(alpha[column(k, q)])))
            .collect();
        (residual, m - pivots.len())
    }

    #[test]
    fn small_designs_meet_the_bound_against_their_exact_projections() {
        // A loose tolerance, so that the error is well above rounding and the
        // bound of the module documentation is tested against the exact
        // residual r*: ||r - r*||_W <= ||gamma|| / sqrt(lambda_+), gamma the
        // level sums of w r. lambda_+ is the (d + 1)-th smallest eigenvalue of
        // A = D' W D, d the redundancies found in exact arithmetic; the
        // Jacobi eigenvalues are those of A + E with ||E|| <= p(m) eps ||A||
        // for a modest polynomial p (Golub and Van Loan, section 8.5), so
        // m^2 eps ||A||_F is taken off. The computed residual and gamma carry
        // the rounding of sums of at most n terms of the size of |v|, so
        // n m eps ||v||_W is added, as in the comparisons with fixest.
        let unbalanced: (Vec<Vec<u32>>, Vec<i128>) = (
            vec![
                vec![0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5],
                vec![0, 1, 2, 1, 2, 0, 2, 3, 3, 4, 4, 5, 4, 5, 6],
            ],
            vec![1, 2, 1, 3, 1, 1, 2, 1, 1, 4, 1, 1, 2, 1, 3],
        );
        let components: (Vec<Vec<u32>>, Vec<i128>) = (
            vec![vec![0, 0, 1, 1, 2, 3, 3, 2], vec![0, 1, 0, 1, 2, 3, 2, 3]],
            vec![1, 2, 3, 1, 1, 2, 1, 2],
        );
        let nested: (Vec<Vec<u32>>, Vec<i128>) = (
            vec![
                vec![0, 0, 1, 1, 2, 2, 3, 3, 0, 2],
                vec![0, 1, 1, 2, 2, 3, 3, 0, 2, 0],
                vec![0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
            ],
            vec![1, 1, 2, 1, 3, 1, 1, 2, 1, 1],
        );
        let complete: (Vec<Vec<u32>>, Vec<i128>) = (
            vec![
                (0..12).map(|i| i / 3).collect(),
                (0..12).map(|i| i % 3).collect(),
            ],
            vec![1; 12],
        );
        for (name, (factors, w)) in [
            ("unbalanced", unbalanced),
            ("components", components),
            ("nested", nested),
            ("complete", complete),
        ] {
            let n = w.len();
            let v: Vec<i128> = (0..n as i128).map(|k| (k * 37 + 11) % 23 - 9).collect();
            let factors: Vec<&[u32]> = factors.iter().map(Vec::as_slice).collect();
            let (exact, redundancies) = exact_residual(&factors, &w, &v);
            let wf: Vec<f64> = w.iter().map(|&x| x as f64).collect();
            let vf: Vec<f64> = v.iter().map(|&x| x as f64).collect();
            let fe = FixedEffects::new(&factors, Some(&wf), Singletons::Keep).unwrap();
            let control = Absorb {
                tolerance: 1e-6,
                ..Absorb::new()
            };
            let out = fe.demean(&[&vf], &control).unwrap();
            let r = out.columns.col_as_slice(0);
            let m: usize = fe.levels().iter().sum();
            let offset: Vec<usize> = fe
                .levels()
                .iter()
                .scan(0, |s, &g| {
                    *s += g;
                    Some(*s - g)
                })
                .collect();
            let mut a = vec![0.0; m * m];
            let mut gamma2 = 0.0;
            for q in 0..fe.n_factors() {
                let mut sum = vec![0.0; fe.levels()[q]];
                for k in 0..n {
                    sum[fe.codes(q)[k] as usize] += wf[k] * r[k];
                    for p in 0..fe.n_factors() {
                        let i = offset[q] + fe.codes(q)[k] as usize;
                        let j = offset[p] + fe.codes(p)[k] as usize;
                        a[i * m + j] += wf[k];
                    }
                }
                gamma2 += sum.iter().map(|s| s * s).sum::<f64>();
            }
            let (mut eigen, _) = epistat_linalg::eigen::symmetric_eigen_jacobi(&a, m);
            eigen.sort_by(f64::total_cmp);
            let frobenius = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let lambda = eigen[redundancies] - (m * m) as f64 * f64::EPSILON * frobenius;
            assert!(lambda > 0.0, "{name}");
            let norm_v = (0..n).map(|k| wf[k] * vf[k] * vf[k]).sum::<f64>().sqrt();
            let error = (0..n)
                .map(|k| wf[k] * (r[k] - exact[k].to_f64()).powi(2))
                .sum::<f64>()
                .sqrt();
            let bound = gamma2.sqrt() / lambda.sqrt() + (n * m) as f64 * f64::EPSILON * norm_v;
            assert!(error <= bound, "{name}: {error:e} above {bound:e}");
            let fx = fe.fixed_effects(&out, 0);
            assert!(fx.normalised, "{name}");
            if fe.n_factors() == 2 {
                assert_eq!(fx.references[1], redundancies, "{name}");
            }
        }
    }
}
