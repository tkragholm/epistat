//! The first stage shared by the imputation and two-stage estimators: unit
//! and period fixed effects and covariates fitted on the untreated rows, the
//! imputed untreated outcomes, and the projections that carry the first
//! stage's estimation error into the variance.
//!
//! # Projections
//!
//! Both estimators need, for a vector `b` of sums over the regressors, the
//! fitted values `h = Z0 M0^-1 b` on the untreated rows, with `Z0 = [X0 D0]`
//! the covariates and the unit and period dummies of those rows and
//! `M0 = Z0' W Z0`. Here `b = Z0' W base + Z1' c`: a column `base` on the
//! untreated rows and a vector `c` on the treated rows. `M0` is not formed.
//! Instead a column `u` on the untreated rows with `D0' W u = D0' W base +
//! D1' c` is built, so that its projection on the dummies is
//! `D0 (D0' W D0)^+ (D0' W u)`, which the alternating projections of
//! [`FixedEffects::demean`] compute. With two factors this column exists
//! whenever every treated row's unit and period lie in one connected
//! component of the untreated rows' bipartite graph of units and periods, and
//! a spanning forest of that graph gives it in linear time: the sums
//! `D1' c` are demands on the nodes, and peeling the forest from its leaves
//! puts on each tree edge (an untreated row) the flow that meets the demand
//! of the node below it. The demands of a component add to zero on each side
//! of the graph, so the root's demand is met as well.
//!
//! With `f_D = u - u~` the projection of `u` on the dummies (`u~` the
//! demeaned column), `X~0` the demeaned covariates and `B = (X~0' W X~0)^-1`,
//! the block solution of the normal equations is
//!
//! ```text
//! gamma_X = B (X0' W base + X1' c - X0' W f_D),      h = f_D + X~0 gamma_X.
//! ```
//!
//! `f_D` is a sum of unit and period effects by construction, so `h` lies in
//! the span of `Z0` to rounding, and the iteration's error appears only in the
//! normal equations `Z0' W h = b`, which bound the distance to the exact
//! projection (the tests check that bound).

use epistat_glm::{Absorb, AbsorbError, FeolsFit, FixedEffects};
use epistat_linalg::MatRef;
use rayon::prelude::*;

/// The rows of a staggered panel after the first stage.
pub(crate) struct FirstStage {
    /// Whether the row is treated, `t >= g`.
    pub(crate) treated: Vec<bool>,
    /// `y - y_hat(0)`, `None` for rows whose untreated outcome cannot be
    /// imputed.
    pub(crate) adjusted: Vec<Option<f64>>,
    /// The prior weight of each row, 1 without weights.
    pub(crate) weight: Vec<f64>,
    /// The kept covariates' coefficients, `None` for aliased covariates.
    pub(crate) coefficients: Vec<Option<f64>>,
}

/// The compact level of each caller code of a factor.
fn level_map(labels: &[u32]) -> Vec<u32> {
    let size = labels.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut map = vec![u32::MAX; size];
    for (level, &code) in labels.iter().enumerate() {
        map[code as usize] = u32::try_from(level).unwrap_or(u32::MAX);
    }
    map
}

impl FirstStage {
    /// The imputed outcomes from a first-stage fit.
    pub(crate) fn new(
        unit: &[u32],
        time: &[u32],
        treated: Vec<bool>,
        outcome: &[f64],
        weight: Vec<f64>,
        covariates: MatRef<'_, f64>,
        fit: &FeolsFit<'_>,
    ) -> Self {
        let values = fit.fixed_effects();
        let coefficients = fit.coefficients.clone();
        let p = covariates.ncols();
        let adjusted: Vec<Option<f64>> = (0..unit.len())
            .into_par_iter()
            .map(|i| {
                let mut fitted = values.sum(&[unit[i], time[i]])?;
                for j in 0..p {
                    if let Some(b) = coefficients[j] {
                        fitted += b * covariates[(i, j)];
                    }
                }
                Some(outcome[i] - fitted)
            })
            .collect();
        Self {
            treated,
            adjusted,
            weight,
            coefficients,
        }
    }
}

/// One projection: a column on the untreated rows and a vector on the
/// treated rows, each with one value per input row.
pub(crate) struct Projection<'a> {
    /// Read on the rows of the fit; `None` for zero.
    pub(crate) base: Option<&'a [f64]>,
    /// Read on the treated rows that can be imputed.
    pub(crate) treated: &'a [f64],
}

/// The spanning forest of the untreated rows' graph of units and periods.
struct Forest {
    /// The nodes (unit levels, then period levels) in breadth-first order.
    order: Vec<u32>,
    /// The parent of each node and the position (in the fit's rows) of the
    /// row joining them; `u32::MAX` for roots.
    parent: Vec<u32>,
    edge: Vec<u32>,
}

impl Forest {
    fn new(fe: &FixedEffects) -> Self {
        let levels = fe.levels();
        let (g0, g1) = (levels[0], levels[1]);
        let nodes = g0 + g1;
        let c0 = fe.codes(0);
        let c1 = fe.codes(1);
        let n = c0.len();
        // Adjacency in compressed rows: the fit positions of each node.
        let mut start = vec![0usize; nodes + 1];
        for k in 0..n {
            start[c0[k] as usize + 1] += 1;
            start[g0 + c1[k] as usize + 1] += 1;
        }
        for v in 0..nodes {
            start[v + 1] += start[v];
        }
        let mut fill = start.clone();
        let mut rows = vec![0u32; 2 * n];
        for k in 0..n {
            let k32 = u32::try_from(k).unwrap_or(u32::MAX);
            let a = c0[k] as usize;
            rows[fill[a]] = k32;
            fill[a] += 1;
            let b = g0 + c1[k] as usize;
            rows[fill[b]] = k32;
            fill[b] += 1;
        }
        let mut parent = vec![u32::MAX; nodes];
        let mut edge = vec![u32::MAX; nodes];
        let mut seen = vec![false; nodes];
        let mut order = Vec::with_capacity(nodes);
        let other = |v: usize, k: usize| -> usize {
            if v < g0 {
                g0 + c1[k] as usize
            } else {
                c0[k] as usize
            }
        };
        for root in 0..nodes {
            if seen[root] {
                continue;
            }
            seen[root] = true;
            let mut head = order.len();
            order.push(u32::try_from(root).unwrap_or(u32::MAX));
            while head < order.len() {
                let v = order[head] as usize;
                head += 1;
                for &k in &rows[start[v]..start[v + 1]] {
                    let w = other(v, k as usize);
                    if !seen[w] {
                        seen[w] = true;
                        parent[w] = u32::try_from(v).unwrap_or(u32::MAX);
                        edge[w] = k;
                        order.push(u32::try_from(w).unwrap_or(u32::MAX));
                    }
                }
            }
        }
        Self {
            order,
            parent,
            edge,
        }
    }
}

/// The projections of the first stage.
pub(crate) struct Projector<'a> {
    fe: &'a FixedEffects,
    fit: &'a FeolsFit<'a>,
    covariates: MatRef<'a, f64>,
    unit: &'a [u32],
    time: &'a [u32],
    forest: Forest,
    unit_level: Vec<u32>,
    time_level: Vec<u32>,
}

impl<'a> Projector<'a> {
    pub(crate) fn new(
        fe: &'a FixedEffects,
        fit: &'a FeolsFit<'a>,
        covariates: MatRef<'a, f64>,
        unit: &'a [u32],
        time: &'a [u32],
    ) -> Self {
        Self {
            fe,
            fit,
            covariates,
            unit,
            time,
            forest: Forest::new(fe),
            unit_level: level_map(fe.labels(0)),
            time_level: level_map(fe.labels(1)),
        }
    }

    /// `h = Z0 M0^-1 (Z0' W base + Z1' c)` on the rows of the fit
    /// ([`FixedEffects::rows`]), for the treated rows `rows` that can be
    /// imputed.
    pub(crate) fn project(
        &self,
        target: &Projection<'_>,
        rows: &[usize],
        control: &Absorb,
    ) -> Result<Vec<f64>, AbsorbError> {
        let fe = self.fe;
        let levels = fe.levels();
        let g0 = levels[0];
        let nodes = g0 + levels[1];
        let fit_rows = fe.rows();
        let weights = fe.weights();
        let w = |k: usize| weights.map_or(1.0, |w| w[k]);
        let kept = self.fit.kept_columns();
        let r = kept.len();

        // Demands of the treated rows on the nodes, and their covariate sums.
        let mut demand = vec![0.0; nodes];
        let mut b_x = vec![0.0; r];
        for &i in rows {
            let c = target.treated[i];
            if c == 0.0 {
                continue;
            }
            demand[self.unit_level[self.unit[i] as usize] as usize] += c;
            demand[g0 + self.time_level[self.time[i] as usize] as usize] += c;
            for (a, &j) in kept.iter().enumerate() {
                b_x[a] += c * self.covariates[(i, j)];
            }
        }
        // Flows on the tree edges, from the leaves up.
        let mut u = vec![0.0; fe.n_input()];
        if let Some(base) = target.base {
            for &i in fit_rows {
                u[i] = base[i];
            }
        }
        for &v in self.forest.order.iter().rev() {
            let v = v as usize;
            let parent = self.forest.parent[v];
            if parent == u32::MAX {
                continue;
            }
            let k = self.forest.edge[v] as usize;
            let flow = demand[v];
            demand[parent as usize] -= flow;
            u[fit_rows[k]] += flow / w(k);
        }
        let demeaned = fe.demean(&[&u], control)?;
        let residual = demeaned.columns.col_as_slice(0);
        let f_d: Vec<f64> = fit_rows
            .iter()
            .zip(residual)
            .map(|(&i, &e)| u[i] - e)
            .collect();
        let mut h = f_d;
        if r > 0 {
            // gamma_X = B (X0' W base + X1' c - X0' W f_D)
            let mut rhs = b_x;
            for (k, &i) in fit_rows.iter().enumerate() {
                let base = target.base.map_or(0.0, |b| b[i]);
                let d = w(k) * (base - h[k]);
                for (a, &j) in kept.iter().enumerate() {
                    rhs[a] += d * self.covariates[(i, j)];
                }
            }
            let bread = self.fit.bread();
            let gamma: Vec<f64> = (0..r)
                .map(|a| {
                    let mut total = 0.0;
                    for (b, value) in rhs.iter().enumerate() {
                        total += bread[(a, b)] * value;
                    }
                    total
                })
                .collect();
            let design = self.fit.design();
            for (k, value) in h.iter_mut().enumerate() {
                let mut total = 0.0;
                for (a, g) in gamma.iter().enumerate() {
                    total += design[(k, a)] * g;
                }
                *value += total;
            }
        }
        Ok(h)
    }
}
