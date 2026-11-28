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
//! Before any factorisation, [`solve`] and [`Factor::new`] refuse a shape that
//! does not fit, an empty list of attempts ([`Error::NoAttempts`]) and a
//! matrix or right-hand side with NaN or an infinity (`Error::Input` with
//! `InputError::NotFinite`, naming the first such entry).
//!
//! A candidate fails when its factorisation fails or when its solution has a
//! non-finite entry. Only Cholesky can fail in the first way, when it meets a
//! pivot that is not positive. The other three factorise any matrix, so for
//! them failure shows only as a non-finite solution, which [`Factor::solve`]
//! reports as [`Error::Singular`]. When every candidate fails, [`solve`]
//! returns [`Error::Singular`].
//!
//! # Condition
//!
//! A finite solution can still be meaningless: `[[1, 1], [1, 1 + 1e-12]]`
//! solves to entries of about `1e12`. Every [`Solution`] carries `rcond`,
//! the reciprocal condition number `1 / (|A|_1 |A^-1|_1)` of the matrix that
//! was factorised, with `|A^-1|_1` estimated as LAPACK's condition
//! estimators do: Hager's method \[4\] with Higham's refinements \[3\],
//! which needs a few solves with the factor and its transpose and is
//! usually within a factor of 3 of the exact value. [`solve_with`] refuses a
//! solution whose `rcond` is below `1 / max_condition` with
//! [`Error::IllConditioned`]; by default there is no limit, as in [`solve`].
//! R's `solve` refuses below an `rcond` of the machine epsilon.
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
//!    Philadelphia: SIAM; 2002. Chapter 15.
//! 4. Hager WW. Condition estimates. SIAM J Sci Stat Comput.
//!    1984;5(2):311-6.

use std::sync::OnceLock;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::solvers::{FullPivLu, Lblt, PartialPivLu};
use faer::prelude::*;
use faer::{Conj, Par, Side, Spec};

use epistat_core::InputError;

use crate::error::check_length;
use crate::{Error, is_finite};

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
    /// pivoting, without ridge: the fastest route for a symmetric positive
    /// definite matrix, falling back to methods that need less of the matrix
    /// when it is indefinite or nearly singular.
    pub const SYMMETRIC: [Self; 4] = [
        Self::plain(Method::Cholesky),
        Self::plain(Method::Lblt),
        Self::plain(Method::PartialPivLu),
        Self::plain(Method::FullPivLu),
    ];
}

/// A solution with the factorisation and ridge that produced it.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Solution {
    /// The solution.
    pub x: Mat<f64>,
    /// The factorisation that produced it.
    pub method: Method,
    /// The ridge added to the diagonal, 0 when none was.
    pub ridge: f64,
    /// The estimated reciprocal condition number `1 / (|A|_1 |A^-1|_1)` of
    /// the matrix factorised, `A` plus the ridge: 1 for a multiple of the
    /// identity, near 0 for a matrix close to singular. A solution with an
    /// `rcond` near the machine epsilon has lost about all its digits. See
    /// the module documentation.
    pub rcond: f64,
}

/// Limits on a solution that [`solve_with`] checks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolveOptions {
    /// The largest condition number accepted, `1 / rcond`. `None`, the
    /// default, accepts any finite solution. `Some(1.0 / f64::EPSILON)` is
    /// the rule of R's `solve`.
    pub max_condition: Option<f64>,
}

/// Solves `a x = b`, trying each attempt in turn until one factorises `a`
/// and gives a finite solution.
///
/// The [`Solution`] reports the reciprocal condition number of the matrix
/// factorised; [`solve_with`] refuses one that is too small.
///
/// # Errors
///
/// [`Error::Input`] when `a` is not square, `b` has another number of rows,
/// or either holds NaN or an infinity; [`Error::NoAttempts`] when `attempts`
/// is empty; and [`Error::Singular`] when every attempt fails.
pub fn solve(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    attempts: &[Attempt],
) -> Result<Solution, Error> {
    solve_with(a, b, attempts, &SolveOptions::default())
}

/// [`solve`] with the limits of `options`.
///
/// # Errors
///
/// As [`solve`], and [`Error::IllConditioned`] when the solution's `rcond`
/// is below `1 / max_condition`, or [`Error::Input`] when `max_condition` is
/// not a number of at least 1.
pub fn solve_with(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    attempts: &[Attempt],
    options: &SolveOptions,
) -> Result<Solution, Error> {
    check_shape(a, b.nrows())?;
    if let Some(limit) = options.max_condition
        && !(limit >= 1.0)
    {
        return Err(Error::Input(InputError::Domain {
            what: "max_condition",
            value: limit,
            domain: "[1, inf]",
            row: None,
        }));
    }
    check_attempts(attempts)?;
    check_finite("a", a)?;
    check_finite("b", b)?;
    for attempt in attempts {
        for (matrix, ridge) in candidates(a, attempt.ridge) {
            if let Some(factor) =
                Factor::with_method(matrix.as_ref(), attempt.method, attempt.min_pivot)
                && let Ok(x) = factor.solve(b)
            {
                let rcond = factor.rcond();
                if let Some(limit) = options.max_condition
                    && rcond < 1.0 / limit
                {
                    return Err(Error::IllConditioned {
                        rcond,
                        limit: 1.0 / limit,
                    });
                }
                return Ok(Solution {
                    x,
                    method: attempt.method,
                    ridge,
                    rcond,
                });
            }
        }
    }
    Err(Error::Singular)
}

/// The inverse of `a`, by [`solve`] against the identity.
///
/// # Errors
///
/// As [`solve`].
pub fn inverse(a: MatRef<'_, f64>, attempts: &[Attempt]) -> Result<Solution, Error> {
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
    /// `|A|_1` of the matrix factorised, as the method reads it.
    norm: f64,
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
    /// [`Error::Input`] when `a` is not square or holds NaN or an infinity,
    /// [`Error::NoAttempts`] when `attempts` is empty, and
    /// [`Error::Singular`] when no attempt factorises it.
    pub fn new(a: MatRef<'_, f64>, attempts: &[Attempt]) -> Result<Self, Error> {
        check_shape(a, a.nrows())?;
        check_attempts(attempts)?;
        check_finite("a", a)?;
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
        Err(Error::Singular)
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
    /// [`Error::Input`] when `b` has the wrong number of rows or holds NaN or
    /// an infinity, and [`Error::Singular`] when the solution is not finite.
    pub fn solve(&self, b: MatRef<'_, f64>) -> Result<Mat<f64>, Error> {
        check_length("rows of b", b.nrows(), self.n)?;
        check_finite("b", b)?;
        let x = self.solve_unchecked(b);
        if is_finite(x.as_ref()) {
            Ok(x)
        } else {
            Err(Error::Singular)
        }
    }

    /// The estimated reciprocal condition number `1 / (|A|_1 |A^-1|_1)` of
    /// the factorised matrix, as in [`Solution::rcond`], in `[0, 1]`: 0 when
    /// the estimate of `|A^-1|_1` is not finite. It costs a few solves with the factor.
    #[must_use]
    pub fn rcond(&self) -> f64 {
        if self.n == 0 {
            return 1.0;
        }
        let inverse_norm = inverse_norm_one(
            self.n,
            |x| self.solve_unchecked(x),
            |x| self.solve_transpose_unchecked(x),
        );
        let condition = self.norm * inverse_norm;
        if condition.is_finite() && condition > 0.0 {
            // The condition number is at least 1; rounding of the estimate
            // can take it just below.
            (1.0 / condition).min(1.0)
        } else {
            0.0
        }
    }

    fn solve_unchecked(&self, b: MatRef<'_, f64>) -> Mat<f64> {
        match &self.kind {
            Kind::Cholesky(lower) => cholesky_solve(lower.as_ref(), b),
            Kind::Lblt(f) => on_one_thread(|| f.solve(b)),
            Kind::PartialPivLu(f) => on_one_thread(|| f.solve(b)),
            Kind::FullPivLu(f) => on_one_thread(|| f.solve(b)),
        }
    }

    /// `x` with `A' x = b`; the symmetric factorisations are their own
    /// transposes.
    fn solve_transpose_unchecked(&self, b: MatRef<'_, f64>) -> Mat<f64> {
        match &self.kind {
            Kind::Cholesky(lower) => cholesky_solve(lower.as_ref(), b),
            Kind::Lblt(f) => on_one_thread(|| f.solve(b)),
            Kind::PartialPivLu(f) => on_one_thread(|| f.solve_transpose(b)),
            Kind::FullPivLu(f) => on_one_thread(|| f.solve_transpose(b)),
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
        let symmetric = matches!(method, Method::Cholesky | Method::Lblt);
        Some(Self {
            kind,
            method,
            ridge: 0.0,
            n: a.nrows(),
            norm: norm_one(a, symmetric),
        })
    }
}

/// `|A|_1`, the largest column sum of absolute values. With `symmetric`, the
/// matrix is the symmetric one whose lower triangle `a` holds, which is what
/// Cholesky and Bunch-Kaufman factorise.
fn norm_one(a: MatRef<'_, f64>, symmetric: bool) -> f64 {
    let n = a.ncols();
    (0..n)
        .map(|j| {
            (0..a.nrows())
                .map(|i| {
                    if symmetric && i < j {
                        a[(j, i)].abs()
                    } else {
                        a[(i, j)].abs()
                    }
                })
                .sum::<f64>()
        })
        .fold(0.0, f64::max)
}

/// The most iterations of Hager's method; LAPACK's estimators stop at 5.
const HAGER_ITERATIONS: usize = 5;

/// An estimate of `|A^-1|_1` from solves with `A` and with `A'`: Hager's
/// method, which is a lower bound of the norm and usually close to it, with
/// Higham's test vector of alternating signs as a second lower bound, the
/// larger of the two returned.
fn inverse_norm_one(
    n: usize,
    solve: impl Fn(MatRef<'_, f64>) -> Mat<f64>,
    solve_transpose: impl Fn(MatRef<'_, f64>) -> Mat<f64>,
) -> f64 {
    let one_norm = |y: &Mat<f64>| (0..n).map(|i| y[(i, 0)].abs()).sum::<f64>();
    let signs = |y: &Mat<f64>| Mat::from_fn(n, 1, |i, _| if y[(i, 0)] >= 0.0 { 1.0 } else { -1.0 });
    let largest = |z: &Mat<f64>| {
        (0..n).fold(0, |best, i| {
            if z[(i, 0)].abs() > z[(best, 0)].abs() {
                i
            } else {
                best
            }
        })
    };
    let mut x = Mat::from_fn(n, 1, |_, _| 1.0 / n as f64);
    let mut y = solve(x.as_ref());
    let mut estimate = one_norm(&y);
    if n > 1 && estimate.is_finite() {
        let mut xi = signs(&y);
        let mut z = solve_transpose(xi.as_ref());
        let mut j = largest(&z);
        for _ in 1..HAGER_ITERATIONS {
            x = Mat::from_fn(n, 1, |i, _| if i == j { 1.0 } else { 0.0 });
            y = solve(x.as_ref());
            let next = one_norm(&y);
            let next_signs = signs(&y);
            let repeated = (0..n).all(|i| next_signs[(i, 0)] == xi[(i, 0)]);
            if repeated || !(next > estimate) {
                estimate = estimate.max(next);
                break;
            }
            estimate = next;
            xi = next_signs;
            z = solve_transpose(xi.as_ref());
            let k = largest(&z);
            if z[(k, 0)].abs() <= z[(j, 0)] {
                break;
            }
            j = k;
        }
        let step = 1.0 / (n - 1) as f64;
        let alternating = Mat::from_fn(n, 1, |i, _| {
            let magnitude = 1.0 + i as f64 * step;
            if i % 2 == 0 { magnitude } else { -magnitude }
        });
        let higham = 2.0 * one_norm(&solve(alternating.as_ref())) / (3.0 * n as f64);
        estimate = estimate.max(higham);
    }
    estimate
}

/// `attempts` is not empty.
const fn check_attempts(attempts: &[Attempt]) -> Result<(), Error> {
    if attempts.is_empty() {
        Err(Error::NoAttempts)
    } else {
        Ok(())
    }
}

/// Every entry of `matrix` is finite, or the first that is not, by column.
fn check_finite(what: &'static str, matrix: MatRef<'_, f64>) -> Result<(), Error> {
    for j in 0..matrix.ncols() {
        if let Some(row) = (0..matrix.nrows()).find(|&i| !matrix[(i, j)].is_finite()) {
            return Err(Error::Input(InputError::NotFinite {
                what,
                row,
                column: Some(j),
            }));
        }
    }
    Ok(())
}

/// `a` is square and the right-hand side has `rhs` rows, one per row of `a`.
fn check_shape(a: MatRef<'_, f64>, rhs: usize) -> Result<(), Error> {
    check_length("columns of a", a.ncols(), a.nrows())?;
    check_length("rows of b", rhs, a.nrows())
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
/// `sqrt(delta)`. `None` is returned when that factor has a non-finite entry,
/// and when `a` is not square.
#[must_use]
pub fn cholesky_factor(a: MatRef<'_, f64>, regularisation: Regularisation) -> Option<Mat<f64>> {
    if a.nrows() != a.ncols() {
        return None;
    }
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
    /// [`Error::Input`] when `a`, `b` or `out` does not have the solver's
    /// size, and [`Error::Singular`] when a pivot is not positive or the
    /// solution has a non-finite entry. `out` is then unspecified.
    pub fn solve_into(&mut self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), Error> {
        let n = self.dim;
        check_length("a", a.len(), n * n)?;
        check_length("b", b.len(), n)?;
        check_length("out", out.len(), n)?;
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
        .map_err(|_| Error::Singular)?;
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
                return Err(Error::Singular);
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
pub(crate) fn on_one_thread<R: Send>(f: impl FnOnce() -> R + Send) -> R {
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
    use epistat_core::InputError;
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
            Error::Singular
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
            Err(Error::Singular)
        );
        assert!(matches!(
            two.solve_into(&flat, &b, &mut out),
            Err(Error::Input(InputError::Length { .. }))
        ));
    }

    #[test]
    fn a_non_square_matrix_has_no_cholesky_factor() {
        let a = Mat::<f64>::zeros(2, 3);
        assert!(cholesky_factor(a.as_ref(), Regularisation::default()).is_none());
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
            Error::Singular
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

    /// `1 / (|A|_1 |A^-1|_1)` with the inverse formed column by column.
    fn exact_rcond(a: &Mat<f64>) -> f64 {
        let inverse = inverse(a.as_ref(), &[Attempt::plain(Method::FullPivLu)])
            .unwrap()
            .x;
        1.0 / (norm_one(a.as_ref(), false) * norm_one(inverse.as_ref(), false))
    }

    #[test]
    fn the_reciprocal_condition_number_is_reported() {
        let b = column(&[1.0, 2.0]);
        let identity = Mat::<f64>::identity(2, 2) * 3.0;
        let solution = solve(identity.as_ref(), b.as_ref(), &Attempt::SYMMETRIC).unwrap();
        assert_eq!(solution.rcond.to_bits(), 1.0_f64.to_bits());
        let cases = [
            mat![[4.0, 1.0], [1.0, 3.0]],
            mat![[1.0, 1.0], [1.0, 1.0 + 1e-12]],
            mat![[0.0, 1.0], [1.0, 0.0]],
            mat![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]],
            mat![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]],
        ];
        let lu = [Attempt::plain(Method::PartialPivLu)];
        for (k, a) in cases.iter().enumerate() {
            let rhs = column(&vec![1.0; a.nrows()]);
            // The last matrix is not symmetric, so only LU factorises it.
            let sequences: &[&[Attempt]] = if k + 1 < cases.len() {
                &[&Attempt::SYMMETRIC, &lu]
            } else {
                &[&lu]
            };
            for attempts in sequences {
                let solution = solve(a.as_ref(), rhs.as_ref(), attempts).unwrap();
                let exact = exact_rcond(a);
                // The estimate of |A^-1| is a lower bound, so rcond is at
                // least the exact value, and usually within a factor of 3.
                assert!(
                    solution.rcond >= exact * (1.0 - 1e-6) && solution.rcond <= 3.0 * exact,
                    "{a:?} {:?}: {} against {exact}",
                    solution.method,
                    solution.rcond
                );
            }
        }
        let near = &cases[1];
        let solution = solve(near.as_ref(), b.as_ref(), &Attempt::SYMMETRIC).unwrap();
        assert!(solution.rcond < 1e-12);
    }

    #[test]
    fn an_ill_conditioned_matrix_is_refused_when_asked() {
        let near = mat![[1.0, 1.0], [1.0, 1.0 + 2.0 * f64::EPSILON]];
        let b = column(&[1.0, 2.0]);
        let r = SolveOptions {
            max_condition: Some(1.0 / f64::EPSILON),
        };
        assert!(solve(near.as_ref(), b.as_ref(), &Attempt::SYMMETRIC).is_ok());
        assert!(matches!(
            solve_with(near.as_ref(), b.as_ref(), &Attempt::SYMMETRIC, &r),
            Err(Error::IllConditioned { .. })
        ));
        let fine = mat![[4.0, 1.0], [1.0, 3.0]];
        let plain = solve(fine.as_ref(), b.as_ref(), &Attempt::SYMMETRIC).unwrap();
        let checked = solve_with(fine.as_ref(), b.as_ref(), &Attempt::SYMMETRIC, &r).unwrap();
        assert_eq!(values(&plain.x), values(&checked.x));
        let bad = SolveOptions {
            max_condition: Some(0.5),
        };
        assert!(matches!(
            solve_with(fine.as_ref(), b.as_ref(), &Attempt::SYMMETRIC, &bad),
            Err(Error::Input(InputError::Domain { .. }))
        ));
    }

    #[test]
    fn no_attempts_and_values_that_are_not_finite_are_their_own_errors() {
        let a = mat![[4.0, 1.0], [1.0, 3.0]];
        let b = column(&[1.0, 2.0]);
        assert_eq!(
            solve(a.as_ref(), b.as_ref(), &[]).unwrap_err(),
            Error::NoAttempts
        );
        assert!(matches!(
            Factor::new(a.as_ref(), &[]),
            Err(Error::NoAttempts)
        ));
        let nan = mat![[4.0, 1.0], [f64::NAN, 3.0]];
        assert_eq!(
            solve(nan.as_ref(), b.as_ref(), &Attempt::SYMMETRIC).unwrap_err(),
            Error::Input(InputError::NotFinite {
                what: "a",
                row: 1,
                column: Some(0),
            })
        );
        assert!(matches!(
            Factor::new(nan.as_ref(), &Attempt::SYMMETRIC),
            Err(Error::Input(InputError::NotFinite { what: "a", .. }))
        ));
        let infinite = column(&[1.0, f64::INFINITY]);
        assert!(matches!(
            solve(a.as_ref(), infinite.as_ref(), &Attempt::SYMMETRIC),
            Err(Error::Input(InputError::NotFinite { what: "b", .. }))
        ));
        let factor = Factor::new(a.as_ref(), &Attempt::SYMMETRIC).unwrap();
        assert!(matches!(
            factor.solve(infinite.as_ref()),
            Err(Error::Input(InputError::NotFinite { what: "b", .. }))
        ));
        assert!(factor.rcond() > 0.3);
    }

    #[test]
    fn shapes_are_checked() {
        let a = mat![[1.0, 0.0], [0.0, 1.0]];
        let err = solve(a.as_ref(), column(&[1.0]).as_ref(), &Attempt::SYMMETRIC).unwrap_err();
        assert!(matches!(err, Error::Input(InputError::Length { .. })));
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
