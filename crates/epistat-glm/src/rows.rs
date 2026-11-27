//! Generalised linear models on rows that are formed block by block rather
//! than held in a design matrix.
//!
//! A design whose rows are many and cheap to form, such as the rows of
//! stacked trials or of clones, each a person's row of the data with a few
//! columns added, takes more memory as a matrix than the data it comes from,
//! and the fit holds several more values for each row. [`Glm::fit_rows`] fits
//! the model of [`Glm::fit`] to rows that a [`Rows`] forms on request, and
//! [`RowsFit::robust_covariance`] gives the covariance of
//! [`robust_covariance`](crate::robust::robust_covariance), so that the memory
//! is that of the data the rows come from and of a few blocks of rows.
//!
//! # Algorithm
//!
//! The model, the iterations and the rules for convergence, step halving and
//! aliased columns are those of [`Glm::fit`], described in the [`fit`](crate::fit)
//! module, without an offset or starting coefficients. What `Glm::fit` holds
//! for every row is formed again from the coefficients each time it is
//! needed, by the same operations in the same order:
//!
//! ```text
//! eta = x'b            each row's terms added in column order from -0, then + 0
//! z   = eta - 0 + (y - mu) / mu'(eta)
//! W   = w mu'(eta)^2 / V(mu)
//! ```
//!
//! with `eta` at the start the link of the family's starting mean. The sums
//! over rows are taken in the order of `Glm::fit`: `X'WX` and `X'Wz` as terms
//! of blocks of [`ROW_BLOCK`] rows added in block order with [`BlockSum`],
//! which is how `epistat_linalg` forms them over a whole design, and the
//! deviance, the Pearson sum, the inner products and norms of the test for
//! aliased columns and the cluster sums of the scores as sums over the rows
//! in order. The fit and the covariance therefore have the bits of `Glm::fit`
//! and `robust_covariance` on the design the rows make, on any number of
//! threads.
//!
//! # Cost
//!
//! Blocks are formed in waves of 32 in parallel, so the memory beyond
//! the source is about `WAVE * ROW_BLOCK * (p + 8)` values. Each iteration
//! forms the rows once, at the new coefficients, for the deviance and the
//! products of the next iteration together, and again when a step is halved.
//! The test for aliased columns at the first iteration reads the Gram matrix
//! `X'WX` at the starting means, which that iteration forms anyway, and
//! settles every column whose decision its error bound makes clear, as
//! [`Glm::fit`] does (see the [`fit`](crate::fit) module). Up to a column it
//! does not settle, which is every aliased column, the decisions come from
//! modified Gram-Schmidt with a second pass, whose inner products depend on
//! each other in turn, so it forms the rows about `m^2` times for the first
//! `m` columns, the norm left in one column together with the first inner
//! product of the next. When a column is aliased the first iteration forms
//! the products again over the columns kept, as `Glm::fit` forms its normal
//! equations on the reduced design.
//!
//! # References
//!
//! 1. R Core Team. Fitting generalized linear models: `glm`, `glm.fit` and
//!    `glm.control`. Help pages of the stats package of R.
//! 2. Golub GH, Van Loan CF. Matrix Computations. 4th ed. Baltimore: Johns
//!    Hopkins University Press; 2013. Section 5.2.8, modified Gram-Schmidt.

use epistat_core::Covariance;
use epistat_inference::sandwich::sandwich_with_inverse;
use epistat_linalg::{
    Attempt, BlockSum, Factor, Mat, MatMut, MatRef, ROW_BLOCK, block_scaled_gram, block_xtv,
    row_scaled_gram,
};
use rayon::prelude::*;

use crate::error::weight_error;
use crate::fit::{covariance, screened_columns};
use crate::robust::{SandwichType, place};
use crate::{Error, Glm};

/// The blocks formed at once.
pub(crate) const WAVE: usize = 32;

/// One block of rows to be filled by [`Rows::fill`].
pub struct Block<'a> {
    /// The design of the rows, one row each, all columns.
    pub x: MatMut<'a, f64>,
    /// The response of each row.
    pub y: &'a mut [f64],
    /// The prior weight of each row.
    pub weight: &'a mut [f64],
    /// The cluster of each row, numbered from 0 in the order in which the
    /// clusters first appear among the rows, as
    /// `epistat_inference::Clusters::from_labels_first_seen` numbers labels. Read only by
    /// a clustered covariance.
    pub cluster: &'a mut [usize],
}

/// Rows of a model that are formed when they are needed.
///
/// `fill` must give the same values for a row each time it is asked, since
/// the fit forms every row many times.
pub trait Rows: Sync {
    /// The number of rows.
    fn n_rows(&self) -> usize;
    /// The number of columns of the design.
    fn n_columns(&self) -> usize;
    /// Fills the rows `start..start + block.y.len()`.
    fn fill(&self, start: usize, block: Block<'_>);
}

/// One block as filled, with its rows' linear predictors at up to two sets of
/// coefficients.
struct Filled {
    start: usize,
    x: Mat<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    cluster: Vec<usize>,
}

fn fill_block(rows: &dyn Rows, start: usize, len: usize, p: usize) -> Filled {
    let mut x = Mat::<f64>::zeros(len, p);
    let mut y = vec![0.0; len];
    let mut w = vec![0.0; len];
    let mut cluster = vec![0; len];
    rows.fill(
        start,
        Block {
            x: x.as_mut(),
            y: &mut y,
            weight: &mut w,
            cluster: &mut cluster,
        },
    );
    Filled {
        start,
        x,
        y,
        w,
        cluster,
    }
}

/// Forms the blocks of `rows` in waves, `make` on each in parallel, and hands
/// the results to `take` in block order.
fn in_order<T: Send>(rows: &dyn Rows, make: impl Fn(&Filled) -> T + Sync, mut take: impl FnMut(T)) {
    let (n, p) = (rows.n_rows(), rows.n_columns());
    let starts: Vec<usize> = (0..n).step_by(ROW_BLOCK).collect();
    for wave in starts.chunks(WAVE) {
        let parts: Vec<T> = wave
            .par_iter()
            .map(|&s| make(&fill_block(rows, s, ROW_BLOCK.min(n - s), p)))
            .collect();
        for part in parts {
            take(part);
        }
    }
}

/// The value of an empty sum of `f64`, from which `Iterator::sum` starts.
fn empty_sum() -> f64 {
    std::iter::empty::<f64>().sum()
}

/// The coefficients a linear predictor comes from.
#[derive(Clone, Debug)]
enum Source {
    /// The link of the family's starting mean.
    Start,
    /// These coefficients, one per column of the design.
    Beta(Vec<f64>),
}

/// A fit of [`Glm::fit_rows`]: the fields of [`Fit`](crate::Fit) that do not
/// have one value per row.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RowsFit {
    /// The coefficients, `None` for an aliased column.
    pub coefficients: Vec<Option<f64>>,
    /// The number of coefficients that are not aliased.
    pub rank: usize,
    /// The number of rows with positive prior weight less the rank.
    pub residual_df: usize,
    /// The residual deviance.
    pub deviance: f64,
    /// The number of IRLS iterations.
    pub iterations: usize,
    /// Whether the deviance converged within the iteration limit.
    pub converged: bool,
    /// Whether a step was halved in the last iteration.
    pub boundary: bool,
    /// The Pearson dispersion, or 1 for the binomial and Poisson families.
    pub dispersion: Option<f64>,
    /// `dispersion * (X'WX + ridge)^-1`, `p x p`, as
    /// [`Fit::covariance`](crate::Fit::covariance).
    pub covariance: Covariance,
    model: Glm,
    columns: Vec<usize>,
    beta: Vec<f64>,
    weights_from: Source,
}

/// What a row gives at a set of coefficients.
#[derive(Clone, Copy)]
struct AtRow {
    eta: f64,
    mu: f64,
    slope: f64,
}

impl Glm {
    fn at(&self, source: &Source, x: MatRef<'_, f64>, k: usize, y: f64, w: f64) -> AtRow {
        let eta = match source {
            Source::Start => self.link.link(self.family.start(y, w)),
            Source::Beta(beta) => {
                let mut e = -0.0_f64;
                for (j, &b) in beta.iter().enumerate() {
                    e += x[(k, j)] * b;
                }
                e + 0.0
            }
        };
        let (mu, slope) = self.link.inverse_and_derivative(eta);
        AtRow { eta, mu, slope }
    }

    /// The working response and weight of a row.
    fn working(&self, r: AtRow, y: f64, w: f64) -> (f64, f64) {
        if w > 0.0 && r.slope != 0.0 {
            (
                r.eta - 0.0 + (y - r.mu) / r.slope,
                w * r.slope * r.slope / self.family.variance(r.mu),
            )
        } else {
            (0.0, 0.0)
        }
    }

    /// The terms of `X'WX` and `X'Wz` of a block at `source`, over the
    /// `columns` of the design, and the first row whose working weight is
    /// negative or not finite.
    fn products(&self, b: &Filled, source: &Source, columns: &[usize]) -> Products {
        let len = b.y.len();
        let mut factors = Vec::with_capacity(len);
        let mut wz = Vec::with_capacity(len);
        let mut bad = None;
        for k in 0..len {
            let r = self.at(source, b.x.as_ref(), k, b.y[k], b.w[k]);
            let (z, w2) = self.working(r, b.y[k], b.w[k]);
            if bad.is_none() && !(w2.is_finite() && w2 >= 0.0) {
                bad = Some(b.start + k);
            }
            factors.push(w2.sqrt());
            wz.push(w2 * z);
        }
        let design = Mat::from_fn(len, columns.len(), |i, a| b.x[(i, columns[a])]);
        Products {
            gram: block_scaled_gram(design.as_ref(), &factors),
            cross: block_xtv(design.as_ref(), &wz),
            bad,
        }
    }

    /// Fits the model to the rows formed by `rows`, as [`Glm::fit`] fits it
    /// to the design, response and prior weights they make, without an
    /// offset or starting coefficients. See the [module](self).
    ///
    /// # Errors
    ///
    /// As [`Glm::fit`].
    pub fn fit_rows(&self, rows: &dyn Rows) -> Result<RowsFit, Error> {
        let p = rows.n_columns();
        epistat_core::check::non_empty("x", rows.n_rows())?;
        // The first invalid weight, design row and response, the rows of
        // positive weight and the deviance at the starting means.
        let family = self.family;
        let mut bad_weight: Option<(usize, f64)> = None;
        let mut bad_x: Option<usize> = None;
        let mut bad_y: Option<usize> = None;
        let mut bad_response: Option<(usize, f64)> = None;
        let mut used = 0_usize;
        let mut old_deviance = empty_sum();
        let start = Source::Start;
        in_order(
            rows,
            |b| {
                let mut bw = None;
                let mut bx = None;
                let mut bf = None;
                let mut by = None;
                let mut used = 0;
                let mut terms = Vec::with_capacity(b.y.len());
                for k in 0..b.y.len() {
                    let (y, w) = (b.y[k], b.w[k]);
                    if bw.is_none() && !(w.is_finite() && w >= 0.0) {
                        bw = Some((b.start + k, w));
                    }
                    if bx.is_none() && (0..p).any(|j| !b.x[(k, j)].is_finite()) {
                        bx = Some(b.start + k);
                    }
                    if bf.is_none() && !y.is_finite() {
                        bf = Some(b.start + k);
                    }
                    if by.is_none() && !family.valid_response(y) {
                        by = Some((b.start + k, y));
                    }
                    used += usize::from(w > 0.0);
                    let r = self.at(&start, b.x.as_ref(), k, y, w);
                    terms.push(family.deviance(y, r.mu, w));
                }
                (bw, bx, bf, by, used, terms)
            },
            |(bw, bx, bf, by, u, terms)| {
                bad_weight = bad_weight.or(bw);
                bad_x = bad_x.or(bx);
                bad_y = bad_y.or(bf);
                bad_response = bad_response.or(by);
                used += u;
                for t in terms {
                    old_deviance += t;
                }
            },
        );
        if let Some((row, value)) = bad_weight {
            return Err(weight_error("weights", row, value));
        }
        if let Some(row) = bad_x {
            return Err(epistat_core::InputError::NotFinite {
                what: "x",
                row,
                column: None,
            }
            .into());
        }
        if let Some(row) = bad_y {
            return Err(epistat_core::InputError::NotFinite {
                what: "y",
                row,
                column: None,
            }
            .into());
        }
        crate::error::tolerance("tolerance", self.tolerance)?;
        family.check()?;
        if let Some((row, value)) = bad_response {
            return Err(Error::Response { row, value });
        }

        let mut previous: Option<Vec<f64>> = None;
        let mut beta = vec![0.0; p];
        let (mut converged, mut boundary, mut iterations) = (false, false, 0);
        let mut current_deviance = old_deviance;
        let mut last_gram = Mat::<f64>::zeros(p, p);
        let mut columns: Vec<usize> = (0..p).collect();
        let mut weights_from = Source::Start;
        // The products at the predictor the next iteration starts from, when
        // they were formed with its deviance.
        let mut ahead: Option<Summed> = None;

        for iteration in 1..=self.max_iterations {
            iterations = iteration;
            let source = match &previous {
                None if iteration == 1 => Source::Start,
                _ => Source::Beta(beta.clone()),
            };
            if iteration == 1 && self.ridge == 0.0 {
                // The products at the starting means over every column give
                // the Gram matrix the screen for aliased columns reads, and
                // are those of this iteration when no column is aliased.
                let all = self.summed(rows, &source, &columns);
                if let Some(row) = all.bad {
                    return Err(working_weight(row));
                }
                let keep = screened_columns(&all.gram, rows.n_rows(), self.tolerance, |m| {
                    independent_columns(self, rows, self.tolerance, m)
                });
                columns = (0..p).filter(|&j| keep[j]).collect();
                if columns.len() == p {
                    ahead = Some(all);
                }
            }
            let summed = match ahead.take() {
                Some(s) => s,
                None => self.summed(rows, &source, &columns),
            };
            if let Some(row) = summed.bad {
                return Err(working_weight(row));
            }
            let mut gram = summed.gram;
            for j in 1..gram.ncols() {
                gram[(j, j)] += self.ridge;
            }
            let solution =
                epistat_linalg::solve(gram.as_ref(), summed.cross.as_ref(), &Attempt::SYMMETRIC)?;
            last_gram = gram;
            beta = vec![0.0; p];
            for (a, &j) in columns.iter().enumerate() {
                beta[j] = solution.x[(a, 0)];
            }
            weights_from = source;
            let mut at = self.deviance_at(rows, &Source::Beta(beta.clone()), Some(&columns));
            current_deviance = at.deviance;
            boundary = false;
            if !current_deviance.is_finite() || !at.valid {
                let Some(old) = previous.as_ref() else {
                    return Err(Error::NoValidStart);
                };
                let mut halvings = 0;
                while !current_deviance.is_finite() || !at.valid {
                    halvings += 1;
                    if halvings > self.max_iterations {
                        return Err(Error::StepHalving);
                    }
                    for (b, o) in beta.iter_mut().zip(old) {
                        *b = (*b + o) / 2.0;
                    }
                    at = self.deviance_at(rows, &Source::Beta(beta.clone()), Some(&columns));
                    current_deviance = at.deviance;
                }
                boundary = true;
            }
            ahead = at.products;
            if (current_deviance - old_deviance).abs() / (0.1 + current_deviance.abs())
                < self.tolerance
            {
                converged = true;
                break;
            }
            old_deviance = current_deviance;
            previous = Some(beta.clone());
        }

        let rank = columns.len();
        let dispersion = if family.fixed_dispersion() {
            Some(1.0)
        } else if used > rank {
            let final_source = Source::Beta(beta.clone());
            let mut pearson = empty_sum();
            in_order(
                rows,
                |b| {
                    let mut terms = Vec::new();
                    for k in 0..b.y.len() {
                        let (y, w) = (b.y[k], b.w[k]);
                        let then = self.at(&weights_from, b.x.as_ref(), k, y, w);
                        let (_, w2) = self.working(then, y, w);
                        if w2 > 0.0 {
                            let now = self.at(&final_source, b.x.as_ref(), k, y, w);
                            let residual = (y - now.mu) / now.slope;
                            terms.push(w2 * residual * residual);
                        }
                    }
                    terms
                },
                |terms| {
                    for t in terms {
                        pearson += t;
                    }
                },
            );
            Some(pearson / (used as f64 - rank as f64))
        } else {
            None
        };
        let inverse = Factor::new(last_gram.as_ref(), &Attempt::SYMMETRIC)?
            .solve(Mat::<f64>::identity(rank, rank).as_ref())?;
        let mut values = vec![None; p * p];
        if let Some(dispersion) = dispersion {
            for (a, &i) in columns.iter().enumerate() {
                for (b, &j) in columns.iter().enumerate() {
                    values[i * p + j] = Some(dispersion * inverse[(a, b)]);
                }
            }
        }
        let covariance = covariance(p, values)?;
        let coefficients = (0..p)
            .map(|j| columns.contains(&j).then_some(beta[j]))
            .collect();
        Ok(RowsFit {
            coefficients,
            rank,
            residual_df: used.saturating_sub(rank),
            deviance: current_deviance,
            iterations,
            converged,
            boundary,
            dispersion,
            covariance,
            model: *self,
            columns,
            beta,
            weights_from,
        })
    }

    /// `X'WX` and `X'Wz` at `source` over `columns`.
    fn summed(&self, rows: &dyn Rows, source: &Source, columns: &[usize]) -> Summed {
        let (n, q) = (rows.n_rows(), columns.len());
        let mut gram = BlockSum::new(n, q, q);
        let mut cross = BlockSum::new(n, q, 1);
        let mut bad = None;
        in_order(
            rows,
            |b| self.products(b, source, columns),
            |t| {
                gram.add(&t.gram);
                cross.add(&t.cross);
                bad = bad.or(t.bad);
            },
        );
        Summed {
            gram: gram.finish_symmetric(),
            cross: cross.finish(),
            bad,
        }
    }

    /// The deviance at `source` and whether every mean is valid, with the
    /// products of the next iteration at it when `columns` is given.
    fn deviance_at(&self, rows: &dyn Rows, source: &Source, columns: Option<&[usize]>) -> AtSource {
        let (n, family) = (rows.n_rows(), self.family);
        let mut deviance = empty_sum();
        let mut valid = true;
        let q = columns.map_or(0, <[usize]>::len);
        let mut gram = BlockSum::new(n, q, q);
        let mut cross = BlockSum::new(n, q, 1);
        let mut bad = None;
        in_order(
            rows,
            |b| {
                let mut terms = Vec::with_capacity(b.y.len());
                let mut valid = true;
                for k in 0..b.y.len() {
                    let (y, w) = (b.y[k], b.w[k]);
                    let r = self.at(source, b.x.as_ref(), k, y, w);
                    terms.push(family.deviance(y, r.mu, w));
                    valid &= family.valid_mean(r.mu);
                }
                let products = columns.map(|c| self.products(b, source, c));
                (terms, valid, products)
            },
            |(terms, v, products)| {
                for t in terms {
                    deviance += t;
                }
                valid &= v;
                if let Some(t) = products {
                    gram.add(&t.gram);
                    cross.add(&t.cross);
                    bad = bad.or(t.bad);
                }
            },
        );
        let products = columns.map(|_| Summed {
            gram: gram.finish_symmetric(),
            cross: cross.finish(),
            bad,
        });
        AtSource {
            deviance,
            valid,
            products,
        }
    }
}

struct Products {
    gram: Mat<f64>,
    cross: Mat<f64>,
    bad: Option<usize>,
}

struct Summed {
    gram: Mat<f64>,
    cross: Mat<f64>,
    bad: Option<usize>,
}

struct AtSource {
    deviance: f64,
    valid: bool,
    products: Option<Summed>,
}

/// The first `limit` of the columns of `sqrt(W) X` at the starting means that
/// `Glm::fit` keeps, by its modified Gram-Schmidt with a second pass, in
/// column order.
///
/// The vector of a column after each projection is formed again for every row
/// from the inner products found so far, so each inner product and each norm
/// takes one pass over the rows, summed in row order as `Glm::fit` sums them.
/// The error of a working weight that is not finite at `row`, as the
/// products over a whole design report a bad weight.
const fn working_weight(row: usize) -> Error {
    Error::Design(epistat_linalg::Error::Input(
        epistat_core::InputError::NotFinite {
            what: "working weights",
            row,
            column: None,
        },
    ))
}

fn independent_columns(model: &Glm, rows: &dyn Rows, epsilon: f64, limit: usize) -> Vec<bool> {
    let tol = 1e-7_f64.min(epsilon / 1000.0);
    let p = limit;
    // The basis: each kept column with the inner products it was projected by,
    // two passes over the basis before it, and its norm after them.
    let mut basis: Vec<Kept> = Vec::new();
    let mut keep = Vec::with_capacity(p);
    // The first stage of the next column, formed with the last stage of this
    // one when it needs only basis vectors that are already known.
    let mut ahead: Option<(f64, Option<f64>)> = None;
    for j in 0..p {
        let mut dots: Vec<f64> = Vec::with_capacity(2 * basis.len());
        // Stage 0 gives the sum of squares of the column before any
        // projection and its inner product with the first basis vector.
        let first = ahead
            .take()
            .unwrap_or_else(|| sums(model, rows, &basis, &[(j, &[], basis.first().map(|_| 0))])[0]);
        let original = first.0.sqrt();
        if let Some(d) = first.1 {
            dots.push(d);
        }
        for stage in dots.len()..2 * basis.len() {
            let (_, d) = sums(
                model,
                rows,
                &basis,
                &[(j, &dots, Some(stage % basis.len()))],
            )[0];
            dots.push(d.unwrap_or(f64::NAN));
        }
        // With no projection the norm left is the norm the column started
        // with, summed alike.
        let left = if dots.is_empty() {
            original
        } else if j + 1 < p {
            let both = sums(
                model,
                rows,
                &basis,
                &[(j, &dots, None), (j + 1, &[], Some(0))],
            );
            ahead = Some(both[1]);
            both[0].0.sqrt()
        } else {
            sums(model, rows, &basis, &[(j, &dots, None)])[0].0.sqrt()
        };
        if left < tol * if original > 0.0 { original } else { 1.0 } {
            keep.push(false);
            continue;
        }
        basis.push(Kept {
            column: j,
            dots,
            norm: left,
        });
        keep.push(true);
    }
    keep
}

/// A column kept by the test for aliased columns.
struct Kept {
    column: usize,
    /// The inner products it was projected by: two passes over the basis
    /// vectors before it.
    dots: Vec<f64>,
    norm: f64,
}

/// For each request `(j, dots, with)`, column `j` projected by the inner
/// products `dots`: the sum of the squares of its values over the rows, and,
/// when `with` is given, its inner product with basis vector `with`, both
/// summed in row order. The requests share one pass over the rows.
fn sums(
    model: &Glm,
    rows: &dyn Rows,
    basis: &[Kept],
    requests: &[(usize, &[f64], Option<usize>)],
) -> Vec<(f64, Option<f64>)> {
    let mut squares = vec![empty_sum(); requests.len()];
    let mut inner = vec![empty_sum(); requests.len()];
    in_order(
        rows,
        |b| {
            let len = b.y.len();
            let mut sq = vec![Vec::with_capacity(len); requests.len()];
            let mut ip = vec![Vec::new(); requests.len()];
            let mut q = vec![0.0; basis.len()];
            for k in 0..len {
                let (y, w) = (b.y[k], b.w[k]);
                let r = model.at(&Source::Start, b.x.as_ref(), k, y, w);
                let (_, w2) = model.working(r, y, w);
                let s = w2.sqrt();
                for (a, kept) in basis.iter().enumerate() {
                    let mut u = s * b.x[(k, kept.column)];
                    for (stage, &d) in kept.dots.iter().enumerate() {
                        u -= d * q[stage % a.max(1)];
                    }
                    q[a] = u / kept.norm;
                }
                for (t, &(j, dots, with)) in requests.iter().enumerate() {
                    let mut u = s * b.x[(k, j)];
                    for (stage, &d) in dots.iter().enumerate() {
                        u -= d * q[stage % basis.len()];
                    }
                    sq[t].push(u * u);
                    if let Some(v) = with {
                        ip[t].push(q[v] * u);
                    }
                }
            }
            (sq, ip)
        },
        |(sq, ip)| {
            for (t, terms) in sq.into_iter().enumerate() {
                for v in terms {
                    squares[t] += v;
                }
            }
            for (t, terms) in ip.into_iter().enumerate() {
                for v in terms {
                    inner[t] += v;
                }
            }
        },
    );
    requests
        .iter()
        .enumerate()
        .map(|(t, &(_, _, with))| (squares[t], with.map(|_| inner[t])))
        .collect()
}

impl RowsFit {
    /// The fitted mean of each row, as [`Fit::fitted`](crate::Fit::fitted).
    #[must_use]
    pub fn fitted(&self, rows: &dyn Rows) -> Vec<f64> {
        let source = Source::Beta(self.beta.clone());
        let mut out = Vec::with_capacity(rows.n_rows());
        in_order(
            rows,
            |b| {
                (0..b.y.len())
                    .map(|k| self.model.at(&source, b.x.as_ref(), k, b.y[k], b.w[k]).mu)
                    .collect::<Vec<f64>>()
            },
            |part| out.extend(part),
        );
        out
    }

    /// The sandwich covariance of the coefficients, as
    /// [`robust_covariance`](crate::robust::robust_covariance) gives it for
    /// the fit, the design and the response the rows make, with the clusters
    /// of [`Block::cluster`] when `clustered`.
    ///
    /// # Errors
    ///
    /// As `robust_covariance`, and [`Error::Invalid`] for clusters that
    /// are not numbered in the order in which they first appear.
    pub fn robust_covariance(
        &self,
        rows: &dyn Rows,
        clustered: bool,
        kind: SandwichType,
    ) -> Result<Covariance, Error> {
        let p = self.coefficients.len();
        if kind == SandwichType::Hc3 && clustered {
            return Err(Error::Unsupported {
                what: "HC3 with clusters",
            });
        }
        let columns = &self.columns;
        let q = columns.len();
        let dispersion = self
            .dispersion
            .filter(|&d| d > 0.0)
            .ok_or(Error::Undefined("a positive dispersion"))?;
        let inverse = Mat::from_fn(q, q, |a, b| {
            self.covariance
                .get(columns[a], columns[b])
                .map_or(f64::NAN, |v| v / dispersion)
        });
        let n = rows.n_rows();
        let rank = self.rank;
        let model = self.model;
        let fitted = Source::Beta(self.beta.clone());
        // The score of each row of a block, divided by one less its leverage
        // for HC3, and its reduced design.
        let score = |b: &Filled| -> (Mat<f64>, Vec<f64>) {
            let len = b.y.len();
            let reduced = Mat::from_fn(len, q, |i, a| b.x[(i, columns[a])]);
            let values = (0..len)
                .map(|k| {
                    let (y, w) = (b.y[k], b.w[k]);
                    let then = model.at(&self.weights_from, b.x.as_ref(), k, y, w);
                    let (_, w2) = model.working(then, y, w);
                    let now = model.at(&fitted, b.x.as_ref(), k, y, w);
                    let r = if w2 > 0.0 {
                        w2 * (y - now.mu) / model.link.derivative(now.eta)
                    } else {
                        0.0
                    };
                    if kind == SandwichType::Hc3 {
                        let mut quad = 0.0;
                        for a in 0..q {
                            for c in 0..q {
                                quad += reduced[(k, a)] * inverse[(a, c)] * reduced[(k, c)];
                            }
                        }
                        r / (1.0 - w2 * quad)
                    } else {
                        r
                    }
                })
                .collect();
            (reduced, values)
        };
        let (meat, groups) = if clustered {
            let mut sums: Vec<f64> = Vec::new();
            let mut count = 0_usize;
            let mut disordered = false;
            in_order(
                rows,
                |b| {
                    let (reduced, r) = score(b);
                    (reduced, r, b.cluster.clone())
                },
                |(reduced, r, cluster)| {
                    for (k, &c) in cluster.iter().enumerate() {
                        if c > count {
                            disordered = true;
                            continue;
                        }
                        if c == count {
                            count += 1;
                            sums.extend(std::iter::repeat_n(0.0, q));
                        }
                        for a in 0..q {
                            sums[c * q + a] += r[k] * reduced[(k, a)];
                        }
                    }
                },
            );
            if disordered {
                return Err(Error::Invalid(
                    "the clusters must be numbered in the order in which they first appear",
                ));
            }
            let matrix = Mat::from_fn(count, q, |c, a| sums[c * q + a]);
            (row_scaled_gram(matrix.as_ref(), |_| 1.0), Some(count))
        } else {
            let mut gram = BlockSum::new(n, q, q);
            in_order(
                rows,
                |b| {
                    let (reduced, r) = score(b);
                    block_scaled_gram(reduced.as_ref(), &r)
                },
                |t| gram.add(&t),
            );
            (gram.finish_symmetric(), None)
        };
        let v = sandwich_with_inverse(inverse.as_ref(), meat.as_ref());
        place(&v, kind.factor(n, rank, groups), columns, p)
    }
}
