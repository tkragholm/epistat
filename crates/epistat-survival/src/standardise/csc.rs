//! The standardised absolute risk of a cause from cause-specific Cox models
//! and its influence function. See the module documentation of
//! [`standardise`](super) for the formulas.
//!
//! The influence function of the absolute risk of one covariate row is linear
//! in a set of arrays indexed by the grid position: the running sums behind
//! [`absrisk`](crate::absrisk)'s variance, which
//! [`Shape`] forms. Their coefficients depend on the row only through its risk
//! path and its covariates. The average over the rows of the population is
//! therefore taken once, on those arrays, and the influence function of the
//! training rows is formed from the averages in one further pass.

use super::{LevelRows, ordered_reduce};
use crate::absrisk::influence::{
    Acc, Baselines, Influence, InfluenceForm, ModelInput, Scratch, Shape,
};
use crate::error::{Error, Result};

/// What the evaluation reads.
pub(super) struct Csc<'a> {
    pub(super) models: &'a [ModelInput<'a>],
    pub(super) base: &'a Baselines,
    pub(super) cause_model: usize,
    /// The models that form the event-free survival.
    pub(super) survival: &'a [usize],
    pub(super) product_limit: bool,
    pub(super) form: InfluenceForm,
    /// The rows of positive weight, their strata and weights divided by the
    /// total weight.
    pub(super) rows: &'a [usize],
    pub(super) stratum: &'a [usize],
    pub(super) weight: &'a [f64],
}

/// The averages of one level.
struct LevelAcc {
    /// The standardised risk at each needed count.
    fbar: Vec<f64>,
    min_valid: usize,
    min_std_valid: usize,
    acc: Option<Acc>,
}

pub(super) fn add(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

/// The covariates and risk scores of one row.
struct RowBuf {
    xs: Vec<Vec<f64>>,
    theta: Vec<f64>,
}

/// The result of the evaluation: the risk of every level at every needed
/// count and, with the influence function, the Gram matrix of the levels at
/// every needed count.
pub(super) struct Evaluated {
    /// `levels * needed`, `None` where the risk is not defined.
    pub(super) risk: Vec<Option<f64>>,
    /// Whether the influence function is defined, `levels * needed`.
    pub(super) std_ok: Vec<bool>,
    /// `needed * levels * levels`, zero where a risk is not defined.
    pub(super) gram: Option<Vec<f64>>,
}

impl Csc<'_> {
    fn new_row(&self) -> RowBuf {
        RowBuf {
            xs: self.models.iter().map(|m| vec![0.0; m.x.ncols()]).collect(),
            theta: vec![0.0; self.models.len()],
        }
    }

    fn fill_row(&self, level: &LevelRows<'_>, row: usize, buf: &mut RowBuf) -> Result<()> {
        for (m, model) in self.models.iter().enumerate() {
            level.fill(m, model.x, row, &mut buf.xs[m]);
            let mut eta = 0.0;
            for (a, b) in self.base.models[m].beta.iter().enumerate() {
                eta += buf.xs[m][a] * b;
            }
            if let Some(offset) = model.data.offset() {
                eta += offset[row] - self.base.models[m].offset_centre;
            }
            let value = f64::exp(eta);
            if !value.is_finite() {
                return Err(Error::undefined(format!(
                    "levels: the risk score of row {row} in model {m} is not finite"
                )));
            }
            buf.theta[m] = value;
        }
        Ok(())
    }

    /// Evaluate the levels at the sorted distinct counts `needed`, all positive.
    pub(super) fn evaluate(
        &self,
        levels: &[LevelRows<'_>],
        infl: Option<&Influence>,
        needed: &[usize],
    ) -> Result<Evaluated> {
        let n_levels = levels.len();
        let top = needed.last().copied().unwrap_or(0);
        let n_needed = needed.len();
        let n = self.rows.len();
        let survival = self.survival;
        let shape = infl
            .filter(|_| n_needed > 0)
            .map(|infl| Shape::new(self.base, infl, self.cause_model, survival, needed));
        let exact = infl.is_some() && self.form == InfluenceForm::Exact;
        let path_of = |buf: &RowBuf, s: usize, exact: bool| {
            self.base.path(
                &buf.theta,
                s,
                top,
                survival,
                self.cause_model,
                self.product_limit,
                exact,
            )
        };
        let mut accs = Vec::with_capacity(n_levels);
        if n_needed > 0 {
            for level in levels {
                let mut acc = ordered_reduce(
                    n,
                    |range| {
                        let mut acc = LevelAcc {
                            fbar: vec![0.0; n_needed],
                            min_valid: usize::MAX,
                            min_std_valid: usize::MAX,
                            acc: shape.as_ref().map(Shape::new_acc),
                        };
                        let mut scratch: Option<Scratch> = shape.as_ref().map(Shape::new_scratch);
                        let mut buf = self.new_row();
                        for i in range {
                            let s = self.stratum[i];
                            self.fill_row(level, self.rows[i], &mut buf)?;
                            let path = path_of(&buf, s, exact);
                            let wt = self.weight[i];
                            acc.min_valid = acc.min_valid.min(path.valid);
                            acc.min_std_valid = acc.min_std_valid.min(path.std_valid);
                            for (u, &c) in needed.iter().enumerate() {
                                if c <= path.valid {
                                    acc.fbar[u] += wt * path.cif[c - 1];
                                }
                            }
                            if let (Some(shape), Some(sums), Some(sc)) =
                                (shape.as_ref(), acc.acc.as_mut(), scratch.as_mut())
                            {
                                shape.add_row(sums, wt, s, &path, &buf.theta, &buf.xs, sc);
                            }
                        }
                        Ok(acc)
                    },
                    |a, b| {
                        add(&mut a.fbar, &b.fbar);
                        a.min_valid = a.min_valid.min(b.min_valid);
                        a.min_std_valid = a.min_std_valid.min(b.min_std_valid);
                        if let (Some(x), Some(y)) = (a.acc.as_mut(), b.acc) {
                            Shape::merge(x, y);
                        }
                    },
                )?;
                if let (Some(shape), Some(sums)) = (shape.as_ref(), acc.acc.as_mut()) {
                    shape.finish(sums);
                }
                accs.push(acc);
            }
        }
        let mut risk = vec![None; n_levels * n_needed];
        let mut std_ok = vec![false; n_levels * n_needed];
        for (l, acc) in accs.iter().enumerate() {
            for (u, &c) in needed.iter().enumerate() {
                if c <= acc.min_valid {
                    risk[l * n_needed + u] = Some(acc.fbar[u]);
                    std_ok[l * n_needed + u] = c <= acc.min_std_valid;
                }
            }
        }
        let (Some(infl), Some(shape)) = (infl, shape.as_ref()) else {
            let gram = infl.map(|_| Vec::new());
            return Ok(Evaluated { risk, std_ok, gram });
        };
        let _ = infl;
        let size = n_needed * n_levels * n_levels;
        let gram = ordered_reduce(
            n,
            |range| {
                let mut gram = vec![0.0; size];
                let mut buf = self.new_row();
                let mut own = vec![0.0; n_levels];
                for i in range {
                    // The row's own risk at each needed count, per level.
                    let mut own_all = vec![0.0; n_levels * n_needed];
                    for (l, level) in levels.iter().enumerate() {
                        self.fill_row(level, self.rows[i], &mut buf)?;
                        let path = path_of(&buf, self.stratum[i], false);
                        for (u, &c) in needed.iter().enumerate() {
                            own_all[l * n_needed + u] = if c <= path.valid {
                                path.cif[c - 1]
                            } else {
                                f64::NAN
                            };
                        }
                    }
                    let wt = self.weight[i];
                    for (u, _) in needed.iter().enumerate() {
                        for l in 0..n_levels {
                            own[l] = match (risk[l * n_needed + u], std_ok[l * n_needed + u]) {
                                (Some(fbar), true) => {
                                    let acc = accs[l].acc.as_ref().expect("influence sums");
                                    shape.model_part(acc, i, u)
                                        + wt * (own_all[l * n_needed + u] - fbar)
                                }
                                _ => 0.0,
                            };
                        }
                        let base = u * n_levels * n_levels;
                        for a in 0..n_levels {
                            for b in 0..n_levels {
                                gram[base + a * n_levels + b] += own[a] * own[b];
                            }
                        }
                    }
                }
                Ok(gram)
            },
            |a, b| add(a, &b),
        )?;
        Ok(Evaluated {
            risk,
            std_ok,
            gram: Some(gram),
        })
    }
}
