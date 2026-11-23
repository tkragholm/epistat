//! Solves of `A x = b` for a square matrix `A` that fall back through a
//! chosen sequence of factorisations, with optional ridge retries.
//!
//! Each factorisation runs on one thread, so a solution does not depend on
//! the number of threads available. The systems are the small `p x p` ones of
//! regression.
//!
//! # Algorithm
//!
//! An [`Attempt`] is a [`Method`] with an optional [`Ridge`]. [`solve`] and
//! [`Factor::new`] go through the attempts in order. For each attempt they
//! try the matrix `A` itself, and then, if the attempt has a ridge, the
//! matrices `A + r I` for
//!
//! ```text
//! r(0) = start,    r(k+1) = r(k) * growth,    k = 0, ..., tries - 1.
//! ```
//!
//! Each `r(k)` is formed by repeated multiplication and is added to the
//! original `A`, not to a matrix already ridged. The first candidate that
//! succeeds is used, and the [`Solution`] or [`Factor`] records the method and
//! the ridge `r` that produced it, with `r = 0` for the matrix itself.
//!
//! The methods are the dense factorisations of \[1\] and \[3\]:
//!
//! ```text
//! Cholesky      A = L L'       symmetric positive definite
//! Lblt          A = L B L'     symmetric indefinite, B block diagonal
//! PartialPivLu  P A = L U      row pivoting
//! FullPivLu     P A Q = L U    row and column pivoting
//! ```
//!
//! Cholesky and Lblt read only the lower triangle of `A`. Lblt is the
//! Bunch-Kaufman factorisation \[2\], which pivots symmetrically and so keeps
//! the symmetry of `A`. `B` has blocks of order 1 and 2. Full pivoting is the
//! most reliable of the four on rank-deficient or nearly singular matrices
//! \[1, 3\]. [`Attempt::SYMMETRIC`] runs the four in the order above, without
//! ridge.
//!
//! # Failure
//!
//! A candidate fails when its factorisation fails or when its solution has a
//! non-finite entry. Only Cholesky can fail in the first way, when it meets a
//! pivot that is not positive. The other three factorise any matrix, so for
//! them failure shows only as a non-finite solution, which [`Factor::solve`]
//! reports as [`SolveError::NotFinite`]. When every candidate fails, [`solve`]
//! returns [`SolveError::NotFinite`]. A shape error is returned before any
//! factorisation.
//!
//! # Factor and solve
//!
//! [`solve`] factorises and solves each candidate in turn, so it moves on to
//! the next candidate when a solution is not finite. [`Factor::new`] stops at
//! the first candidate that factorises, whether or not its solutions turn out
//! to be finite, and keeps the factorisation for several right-hand sides.
//! The two can therefore choose different methods for the same matrix.
//! [`inverse`] is [`solve`] against the identity.
//!
//! # Determinism
//!
//! Cholesky runs through faer's in-place routines with `Par::Seq`. The other
//! factorisations and their solves have no such argument, so they run inside a
//! rayon pool of one thread, where faer's default parallelism resolves to one
//! thread. No step splits a sum across threads, so the bits of a result are
//! the same for every thread count of the calling program. faer chooses its
//! SIMD kernels from the CPU at run time. The bits are therefore not promised
//! to be the same across CPUs.
//!
//! # References
//!
//! 1. Golub GH, Van Loan CF. Matrix Computations. 4th ed. Baltimore: Johns
//!    Hopkins University Press; 2013.
//! 2. Bunch JR, Kaufman L. Some stable methods for calculating inertia and
//!    solving symmetric linear systems. Math Comput. 1977;31(137):163-79.
//! 3. Higham NJ. Accuracy and Stability of Numerical Algorithms. 2nd ed.
//!    Philadelphia: SIAM; 2002.

use std::sync::OnceLock;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::solvers::{FullPivLu, Lblt, PartialPivLu};
use faer::prelude::*;
use faer::{Conj, Par, Side, Spec};

use crate::is_finite;

/// A factorisation to try.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    /// `L L'` of the lower triangle, for symmetric positive definite systems.
    Cholesky,
    /// Bunch-Kaufman `L B L'` of the lower triangle, for symmetric indefinite
    /// systems.
    Lblt,
    /// LU with partial pivoting, for general square systems.
    PartialPivLu,
    /// For rank-deficient or nearly singular systems.
    FullPivLu,
}

/// Retries after a failure with `ridge` added to the diagonal: `start`, then
/// `start * growth`, and so on, `tries` times in all. Each retry adds the
/// ridge to the original matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ridge {
    /// The first ridge.
    pub start: f64,
    /// The factor by which the ridge grows at each retry.
    pub growth: f64,
    /// The number of retries.
    pub tries: u32,
}

/// One step of a fallback sequence.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Attempt {
    /// The factorisation to use.
    pub method: Method,
    /// The ridge schedule to retry with, if any.
    pub ridge: Option<Ridge>,
    /// A factorisation whose smallest pivot is at most this in absolute
    /// value counts as failed: the diagonal of `U` for the LU methods, of `L`
    /// for Cholesky. 0 accepts any non-zero pivot.
    pub min_pivot: f64,
}

impl Attempt {
    /// An attempt with no ridge and no pivot threshold.
    #[must_use]
    pub const fn plain(method: Method) -> Self {
        Self {
            method,
            ridge: None,
            min_pivot: 0.0,
        }
    }

    /// An attempt that retries with a ridge, with no pivot threshold.
    #[must_use]
    pub const fn ridged(method: Method, ridge: Ridge) -> Self {
        Self {
            method,
            ridge: Some(ridge),
            min_pivot: 0.0,
        }
    }

    /// The same attempt, failing when a pivot is at most `threshold` in
    /// absolute value. Not checked for Bunch-Kaufman.
    #[must_use]
    pub const fn with_min_pivot(mut self, threshold: f64) -> Self {
        self.min_pivot = threshold;
        self
    }

    /// Cholesky, then Bunch-Kaufman, then LU with partial and then full
    /// pivoting: the sequence `did_methods` and `semicontinuous_models` use.
    pub const SYMMETRIC: [Self; 4] = [
        Self::plain(Method::Cholesky),
        Self::plain(Method::Lblt),
        Self::plain(Method::PartialPivLu),
        Self::plain(Method::FullPivLu),
    ];
}

/// The ways a solve can fail.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SolveError {
    /// The matrix is not square or the right-hand side has the wrong number of rows.
    #[error("the system is {rows} by {cols} with {rhs} right-hand-side rows")]
    Shape {
        /// The number of rows of the matrix.
        rows: usize,
        /// The number of columns of the matrix.
        cols: usize,
        /// The number of rows of the right-hand side.
        rhs: usize,
    },
    /// Every attempt gave a solution that is not finite.
    #[error("no factorisation gave a finite solution")]
    NotFinite,
}

/// A solution with the factorisation and ridge that produced it.
#[derive(Debug, Clone)]
pub struct Solution {
    /// The solution.
    pub x: Mat<f64>,
    /// The factorisation that produced it.
    pub method: Method,
    /// The ridge added to the diagonal, 0 when none was.
    pub ridge: f64,
}

/// Solves `a x = b`, trying each attempt in turn until one factorises `a`
/// and gives a finite solution.
///
/// # Errors
///
/// [`SolveError::Shape`] when `a` is not square or `b` has another number of
/// rows, and [`SolveError::NotFinite`] when every attempt fails.
pub fn solve(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    attempts: &[Attempt],
) -> Result<Solution, SolveError> {
    check_shape(a, b.nrows())?;
    for attempt in attempts {
        for (matrix, ridge) in candidates(a, attempt.ridge) {
            if let Some(factor) =
                Factor::with_method(matrix.as_ref(), attempt.method, attempt.min_pivot)
                && let Ok(x) = factor.solve(b)
            {
                return Ok(Solution {
                    x,
                    method: attempt.method,
                    ridge,
                });
            }
        }
    }
    Err(SolveError::NotFinite)
}

/// The inverse of `a`, by [`solve`] against the identity.
///
/// # Errors
///
/// As [`solve`].
pub fn inverse(a: MatRef<'_, f64>, attempts: &[Attempt]) -> Result<Solution, SolveError> {
    solve(
        a,
        Mat::<f64>::identity(a.nrows(), a.nrows()).as_ref(),
        attempts,
    )
}

/// A factorised matrix, for solving with several right-hand sides.
pub struct Factor {
    kind: Kind,
    method: Method,
    ridge: f64,
    n: usize,
}

enum Kind {
    Cholesky(Mat<f64>),
    Lblt(Lblt<f64>),
    PartialPivLu(PartialPivLu<f64>),
    FullPivLu(FullPivLu<f64>),
}

impl Factor {
    /// Factorises `a` with the first attempt that succeeds. Bunch-Kaufman and
    /// the LU methods always succeed, so attempts after the first of them are
    /// not reached; whether a solution is finite shows only in
    /// [`Factor::solve`], so this can choose another method than [`solve`]
    /// does for the same matrix.
    ///
    /// # Errors
    ///
    /// [`SolveError::Shape`] when `a` is not square, and
    /// [`SolveError::NotFinite`] when no attempt factorises it.
    pub fn new(a: MatRef<'_, f64>, attempts: &[Attempt]) -> Result<Self, SolveError> {
        check_shape(a, a.nrows())?;
        for attempt in attempts {
            for (matrix, ridge) in candidates(a, attempt.ridge) {
                if let Some(mut factor) =
                    Self::with_method(matrix.as_ref(), attempt.method, attempt.min_pivot)
                {
                    factor.ridge = ridge;
                    return Ok(factor);
                }
            }
        }
        Err(SolveError::NotFinite)
    }

    /// The factorisation that succeeded.
    #[must_use]
    pub const fn method(&self) -> Method {
        self.method
    }

    /// The ridge added to the diagonal before factorising, 0 when none was.
    #[must_use]
    pub const fn ridge(&self) -> f64 {
        self.ridge
    }

    /// The lower Cholesky factor `L`, with `L L'` the factorised matrix, when
    /// the method is Cholesky.
    #[must_use]
    pub fn cholesky_factor(&self) -> Option<MatRef<'_, f64>> {
        match &self.kind {
            Kind::Cholesky(lower) => Some(lower.as_ref()),
            _ => None,
        }
    }

    /// Solves for `b`.
    ///
    /// # Errors
    ///
    /// [`SolveError::Shape`] when `b` has the wrong number of rows, and
    /// [`SolveError::NotFinite`] when the solution is not finite.
    pub fn solve(&self, b: MatRef<'_, f64>) -> Result<Mat<f64>, SolveError> {
        let n = self.n;
        if b.nrows() != n {
            return Err(SolveError::Shape {
                rows: n,
                cols: n,
                rhs: b.nrows(),
            });
        }
        let x = match &self.kind {
            Kind::Cholesky(lower) => cholesky_solve(lower.as_ref(), b),
            Kind::Lblt(f) => on_one_thread(|| f.solve(b)),
            Kind::PartialPivLu(f) => on_one_thread(|| f.solve(b)),
            Kind::FullPivLu(f) => on_one_thread(|| f.solve(b)),
        };
        if is_finite(x.as_ref()) {
            Ok(x)
        } else {
            Err(SolveError::NotFinite)
        }
    }

    fn with_method(a: MatRef<'_, f64>, method: Method, min_pivot: f64) -> Option<Self> {
        let kind = match method {
            Method::Cholesky => Kind::Cholesky(cholesky(a)?),
            Method::Lblt => Kind::Lblt(on_one_thread(|| a.lblt(Side::Lower))),
            Method::PartialPivLu => Kind::PartialPivLu(on_one_thread(|| a.partial_piv_lu())),
            Method::FullPivLu => Kind::FullPivLu(on_one_thread(|| a.full_piv_lu())),
        };
        if min_pivot > 0.0 {
            let diagonal = match &kind {
                Kind::Cholesky(lower) => Some(lower.as_ref()),
                Kind::PartialPivLu(f) => Some(f.U()),
                Kind::FullPivLu(f) => Some(f.U()),
                Kind::Lblt(_) => None,
            };
            if let Some(factor) = diagonal {
                let n = factor.nrows().min(factor.ncols());
                if (0..n).any(|i| factor[(i, i)].abs() <= min_pivot) {
                    return None;
                }
            }
        }
        Some(Self {
            kind,
            method,
            ridge: 0.0,
            n: a.nrows(),
        })
    }
}

fn check_shape(a: MatRef<'_, f64>, rhs: usize) -> Result<(), SolveError> {
    if a.nrows() == a.ncols() && a.nrows() == rhs {
        Ok(())
    } else {
        Err(SolveError::Shape {
            rows: a.nrows(),
            cols: a.ncols(),
            rhs,
        })
    }
}

/// The matrix itself, then its ridged copies.
fn candidates(
    a: MatRef<'_, f64>,
    ridge: Option<Ridge>,
) -> impl Iterator<Item = (Mat<f64>, f64)> + '_ {
    let added = ridge.into_iter().flat_map(|ridge| {
        std::iter::successors(Some(ridge.start), move |&r| Some(r * ridge.growth))
            .take(ridge.tries as usize)
    });
    std::iter::once((a.to_owned(), 0.0)).chain(added.map(move |r| {
        let mut ridged = a.to_owned();
        for i in 0..ridged.nrows() {
            ridged[(i, i)] += r;
        }
        (ridged, r)
    }))
}

/// The lower Cholesky factor of `a`'s lower triangle, on this thread.
fn cholesky(a: MatRef<'_, f64>) -> Option<Mat<f64>> {
    cholesky_factor(a, Regularisation::default())
}

/// faer's dynamic regularisation of a Cholesky factorisation, its
/// `LltRegularization`: while `A` is factorised, a pivot at or below `epsilon`
/// is regularised to `delta`. It applies only when both are positive; the
/// default, both zero, factorises `A` as it is.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Regularisation {
    /// The value a regularised pivot takes.
    pub delta: f64,
    /// The threshold at or below which a pivot is regularised.
    pub epsilon: f64,
}

/// The lower Cholesky factor `L` of `A = L L'`, with zeros above the diagonal,
/// or `None` when a pivot is not positive after `regularisation`.
///
/// Only the lower triangle of `a` is read. The factorisation is faer's in-place
/// one on one thread, the one [`Method::Cholesky`] uses. With an active
/// regularisation, faer's factor is used when every diagonal entry is finite
/// and positive. faer's blocked kernels do not apply the regularisation in
/// every case, and can return a zero or negative diagonal entry. The factor is
/// then computed again entry by entry, with each pivot at or below `epsilon`
/// replaced by `delta`, so a regularised pivot always gives the diagonal entry
/// `sqrt(delta)`. `None` is returned when that factor has a non-finite entry.
///
/// # Panics
///
/// Panics when `a` is not square.
#[must_use]
pub fn cholesky_factor(a: MatRef<'_, f64>, regularisation: Regularisation) -> Option<Mat<f64>> {
    assert_eq!(
        a.nrows(),
        a.ncols(),
        "a Cholesky factor of a non-square matrix"
    );
    let n = a.nrows();
    let mut lower = Mat::<f64>::zeros(n, n);
    lower.copy_from_triangular_lower(a);
    let mut memory = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<f64>(
        n,
        Par::Seq,
        Spec::default(),
    ));
    llt::factor::cholesky_in_place(
        lower.as_mut(),
        llt::factor::LltRegularization {
            dynamic_regularization_delta: regularisation.delta,
            dynamic_regularization_epsilon: regularisation.epsilon,
        },
        Par::Seq,
        MemStack::new(&mut memory),
        Spec::default(),
    )
    .ok()?;
    for j in 1..n {
        for i in 0..j {
            lower[(i, j)] = 0.0;
        }
    }
    let active = regularisation.delta > 0.0 && regularisation.epsilon > 0.0;
    if active && !(0..n).all(|i| lower[(i, i)].is_finite() && lower[(i, i)] > 0.0) {
        return regularised_cholesky(a, regularisation);
    }
    Some(lower)
}

/// The lower Cholesky factor with each pivot at or below `epsilon` set to
/// `delta`, computed row by row from `a`'s lower triangle.
fn regularised_cholesky(a: MatRef<'_, f64>, regularisation: Regularisation) -> Option<Mat<f64>> {
    let n = a.nrows();
    let mut lower = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        let mut pivot = a[(j, j)];
        for k in 0..j {
            pivot -= lower[(j, k)] * lower[(j, k)];
        }
        if !pivot.is_finite() {
            return None;
        }
        if pivot <= regularisation.epsilon {
            pivot = regularisation.delta;
        }
        let diagonal = pivot.sqrt();
        lower[(j, j)] = diagonal;
        for i in j + 1..n {
            let mut value = a[(i, j)];
            for k in 0..j {
                value -= lower[(i, k)] * lower[(j, k)];
            }
            lower[(i, j)] = value / diagonal;
        }
    }
    (0..n)
        .all(|j| (0..n).all(|i| lower[(i, j)].is_finite()))
        .then_some(lower)
}

/// Repeated Cholesky solves of systems of one size, each `A x = b` with one
/// right-hand side, reusing the factor and the work memory between them.
///
/// The arithmetic is that of [`solve`] with [`Method::Cholesky`]: faer's
/// in-place factorisation and triangular solves on one thread.
pub struct CholeskySolver {
    dim: usize,
    factor: Mat<f64>,
    rhs: Mat<f64>,
    memory: MemBuffer,
}

impl CholeskySolver {
    /// A solver for systems of `dim` unknowns.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        let factor = llt::factor::cholesky_in_place_scratch::<f64>(dim, Par::Seq, Spec::default());
        let solve = llt::solve::solve_in_place_scratch::<f64>(dim, 1, Par::Seq);
        Self {
            dim,
            factor: Mat::zeros(dim, dim),
            rhs: Mat::zeros(dim, 1),
            memory: MemBuffer::new(factor.or(solve)),
        }
    }

    /// The number of unknowns.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Solves `A x = b` for `A` given row-major in `a`, reading its lower
    /// triangle, and writes `x` to `out`.
    ///
    /// # Errors
    ///
    /// [`SolveError::Shape`] when `a`, `b` or `out` does not have the solver's
    /// size, and [`SolveError::NotFinite`] when a pivot is not positive or the
    /// solution has a non-finite entry. `out` is then unspecified.
    pub fn solve_into(&mut self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), SolveError> {
        let n = self.dim;
        if a.len() != n * n || b.len() != n || out.len() != n {
            return Err(SolveError::Shape {
                rows: n,
                cols: a.len().checked_div(n).unwrap_or(0),
                rhs: b.len(),
            });
        }
        self.factor
            .as_mut()
            .copy_from(MatRef::from_row_major_slice(a, n, n));
        self.rhs
            .as_mut()
            .copy_from(MatRef::from_column_major_slice(b, n, 1));
        llt::factor::cholesky_in_place(
            self.factor.as_mut(),
            llt::factor::LltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut self.memory),
            Spec::default(),
        )
        .map_err(|_| SolveError::NotFinite)?;
        llt::solve::solve_in_place_with_conj(
            self.factor.as_ref(),
            Conj::No,
            self.rhs.as_mut(),
            Par::Seq,
            MemStack::new(&mut self.memory),
        );
        for (row, value) in out.iter_mut().enumerate() {
            let solved = self.rhs[(row, 0)];
            if !solved.is_finite() {
                return Err(SolveError::NotFinite);
            }
            *value = solved;
        }
        Ok(())
    }
}

fn cholesky_solve(lower: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let mut x = b.to_owned();
    let mut memory = MemBuffer::new(llt::solve::solve_in_place_scratch::<f64>(
        lower.nrows(),
        b.ncols(),
        Par::Seq,
    ));
    llt::solve::solve_in_place_with_conj(
        lower,
        Conj::No,
        x.as_mut(),
        Par::Seq,
        MemStack::new(&mut memory),
    );
    x
}

/// Runs `f` on a rayon pool of one thread, where faer's default parallelism
/// is one thread.
fn on_one_thread<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "epistat-linalg".to_owned())
            .build()
            .expect("a one-thread pool")
    })
    .install(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_testkit::{Tolerance, assert_close};

    fn column(values: &[f64]) -> Mat<f64> {
        Mat::from_fn(values.len(), 1, |i, _| values[i])
    }

    fn values(x: &Mat<f64>) -> Vec<f64> {
        (0..x.nrows()).map(|i| x[(i, 0)]).collect()
    }

    #[test]
    fn a_positive_definite_system_uses_cholesky() {
        let a = mat![[4.0, 1.0], [1.0, 3.0]];
        let solution = solve(
            a.as_ref(),
            column(&[1.0, 2.0]).as_ref(),
            &Attempt::SYMMETRIC,
        )
        .unwrap();
        assert_eq!(solution.method, Method::Cholesky);
        assert_close(
            &values(&solution.x),
            &[1.0 / 11.0, 7.0 / 11.0],
            Tolerance::exact().ulps(4),
        );
    }

    #[test]
    fn an_indefinite_system_falls_back() {
        let a = mat![[0.0, 1.0], [1.0, 0.0]];
        let solution = solve(
            a.as_ref(),
            column(&[1.0, 2.0]).as_ref(),
            &Attempt::SYMMETRIC,
        )
        .unwrap();
        assert_ne!(solution.method, Method::Cholesky);
        assert_close(
            &values(&solution.x),
            &[2.0, 1.0],
            Tolerance::exact().abs(1e-15),
        );
    }

    #[test]
    fn a_singular_system_fails_unless_ridged() {
        let a = mat![[1.0, 2.0], [2.0, 4.0]];
        let b = column(&[1.0, 2.0]);
        assert_eq!(
            solve(a.as_ref(), b.as_ref(), &Attempt::SYMMETRIC).unwrap_err(),
            SolveError::NotFinite
        );
        let ridge = Ridge {
            start: 1e-8,
            growth: 10.0,
            tries: 8,
        };
        let solution = solve(
            a.as_ref(),
            b.as_ref(),
            &[Attempt::ridged(Method::Cholesky, ridge)],
        )
        .unwrap();
        assert!(solution.ridge > 0.0);
    }

    #[test]
    fn the_reused_cholesky_solver_agrees_with_solve() {
        let a = mat![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let b = [1.0, -2.0, 0.5];
        let expected = solve(
            a.as_ref(),
            column(&b).as_ref(),
            &[Attempt::plain(Method::Cholesky)],
        )
        .unwrap();
        let flat: Vec<f64> = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| a[(i, j)])
            .collect();
        let mut solver = CholeskySolver::new(3);
        let mut out = [0.0; 3];
        for _ in 0..2 {
            solver.solve_into(&flat, &b, &mut out).unwrap();
            assert_eq!(
                out.map(f64::to_bits).to_vec(),
                values(&expected.x)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let singular = [1.0, 2.0, 2.0, 1.0];
        let mut two = CholeskySolver::new(2);
        assert_eq!(
            two.solve_into(&singular, &[1.0, 1.0], &mut [0.0; 2]),
            Err(SolveError::NotFinite)
        );
        assert!(matches!(
            two.solve_into(&flat, &b, &mut out),
            Err(SolveError::Shape { .. })
        ));
    }

    #[test]
    fn regularisation_factorises_a_singular_matrix() {
        let a = mat![[1.0, 1.0], [1.0, 1.0]];
        assert!(cholesky_factor(a.as_ref(), Regularisation::default()).is_none());
        let lower = cholesky_factor(
            a.as_ref(),
            Regularisation {
                delta: 1e-6,
                epsilon: 1e-12,
            },
        )
        .unwrap();
        assert_eq!(
            [lower[(0, 0)], lower[(1, 0)], lower[(0, 1)]].map(f64::to_bits),
            [1.0_f64, 1.0, 0.0].map(f64::to_bits)
        );
        assert_eq!(lower[(1, 1)].to_bits(), 1e-6_f64.sqrt().to_bits());
    }

    #[test]
    fn a_regularised_factor_has_a_positive_diagonal_and_is_usable() {
        let regularisation = Regularisation {
            delta: 1e-6,
            epsilon: 1e-12,
        };
        let cases = [
            mat![[1.0, 2.0], [2.0, 4.0]],
            mat![[0.0, 0.0], [0.0, 0.0]],
            mat![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            mat![[4.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
        ];
        for a in &cases {
            let lower = cholesky_factor(a.as_ref(), regularisation).unwrap();
            let n = a.nrows();
            for i in 0..n {
                assert!(lower[(i, i)] > 0.0);
            }
            // L L' reproduces A up to the regularised pivots, at most delta
            // per diagonal entry.
            let product = &lower * lower.transpose();
            for i in 0..n {
                for j in 0..n {
                    let tolerance = if i == j { 3e-6 } else { 2e-3 };
                    assert!((product[(i, j)] - a[(i, j)]).abs() <= tolerance);
                }
            }
            let solved = cholesky_solve(lower.as_ref(), column(&vec![1.0; n]).as_ref());
            assert!(solved.col(0).iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn an_indefinite_matrix_is_regularised_not_returned_with_a_negative_pivot() {
        let a = mat![[1.0, 2.0], [2.0, 1.0]];
        let lower = cholesky_factor(
            a.as_ref(),
            Regularisation {
                delta: 1e-6,
                epsilon: 1e-12,
            },
        )
        .unwrap();
        assert_eq!(lower[(1, 1)].to_bits(), 1e-6_f64.sqrt().to_bits());
    }

    #[test]
    fn the_inverse_times_the_matrix_is_the_identity() {
        let a = mat![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let inverse = inverse(a.as_ref(), &Attempt::SYMMETRIC).unwrap().x;
        let product = &a * &inverse;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product[(i, j)] - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn a_pivot_threshold_turns_a_tiny_pivot_into_a_failure() {
        // Rank one plus 1e-16: LU succeeds with a pivot near 1e-16.
        let a = mat![[1.0, 2.0], [2.0, 4.0 + 1e-15]];
        let b = column(&[1.0, 2.0]);
        let loose = solve(
            a.as_ref(),
            b.as_ref(),
            &[Attempt::plain(Method::PartialPivLu)],
        );
        assert!(loose.is_ok());
        let strict = [Attempt::plain(Method::PartialPivLu).with_min_pivot(1e-14)];
        assert_eq!(
            solve(a.as_ref(), b.as_ref(), &strict).unwrap_err(),
            SolveError::NotFinite
        );
        let ridge = Ridge {
            start: 1e-8,
            growth: 10.0,
            tries: 7,
        };
        let ridged = [Attempt::ridged(Method::PartialPivLu, ridge).with_min_pivot(1e-14)];
        assert!(solve(a.as_ref(), b.as_ref(), &ridged).unwrap().ridge > 0.0);
    }

    #[test]
    fn the_cholesky_factor_is_exposed() {
        let a = mat![[4.0, 2.0], [2.0, 3.0]];
        let factor = Factor::new(a.as_ref(), &Attempt::SYMMETRIC).unwrap();
        let lower = factor.cholesky_factor().unwrap();
        assert!((lower[(0, 0)] - 2.0).abs() < 1e-15 && lower[(0, 1)].abs() < 1e-300);
    }

    #[test]
    fn shapes_are_checked() {
        let a = mat![[1.0, 0.0], [0.0, 1.0]];
        let err = solve(a.as_ref(), column(&[1.0]).as_ref(), &Attempt::SYMMETRIC).unwrap_err();
        assert!(matches!(err, SolveError::Shape { .. }));
    }

    #[test]
    fn cholesky_matches_faer_on_one_thread() {
        let n = 12;
        let x = Mat::from_fn(40, n, |i, j| ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.4);
        let a = x.transpose() * &x + Mat::<f64>::identity(n, n);
        let b = Mat::from_fn(n, 2, |i, j| (i + j) as f64);
        let ours = Factor::new(a.as_ref(), &Attempt::SYMMETRIC)
            .unwrap()
            .solve(b.as_ref())
            .unwrap();
        let theirs = on_one_thread(|| a.llt(Side::Lower).unwrap().solve(&b));
        for j in 0..2 {
            for i in 0..n {
                assert_eq!(ours[(i, j)].to_bits(), theirs[(i, j)].to_bits());
            }
        }
    }
}
