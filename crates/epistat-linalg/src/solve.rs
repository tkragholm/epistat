use std::sync::OnceLock;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::solvers::{FullPivLu, Lblt, PartialPivLu};
use faer::prelude::*;
use faer::{Conj, Par, Side, Spec};

use crate::is_finite;

/// A factorisation to try.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// `L L'` of the lower triangle, for symmetric positive definite systems.
    Cholesky,
    /// Bunch-Kaufman `L B L'` of the lower triangle, for symmetric indefinite
    /// systems.
    Lblt,
    PartialPivLu,
    /// For rank-deficient or nearly singular systems.
    FullPivLu,
}

/// Retries after a failure with `ridge` added to the diagonal: `start`, then
/// `start * growth`, and so on, `tries` times in all. Each retry adds the
/// ridge to the original matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ridge {
    pub start: f64,
    pub growth: f64,
    pub tries: u32,
}

/// One step of a fallback sequence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attempt {
    pub method: Method,
    pub ridge: Option<Ridge>,
}

impl Attempt {
    #[must_use]
    pub const fn plain(method: Method) -> Self {
        Self {
            method,
            ridge: None,
        }
    }

    #[must_use]
    pub const fn ridged(method: Method, ridge: Ridge) -> Self {
        Self {
            method,
            ridge: Some(ridge),
        }
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

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SolveError {
    #[error("the system is {rows} by {cols} with {rhs} right-hand-side rows")]
    Shape {
        rows: usize,
        cols: usize,
        rhs: usize,
    },
    #[error("no factorisation gave a finite solution")]
    NotFinite,
}

/// A solution with the factorisation and ridge that produced it.
#[derive(Debug, Clone)]
pub struct Solution {
    pub x: Mat<f64>,
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
            if let Some(factor) = Factor::with_method(matrix.as_ref(), attempt.method) {
                if let Ok(x) = factor.solve(b) {
                    return Ok(Solution {
                        x,
                        method: attempt.method,
                        ridge,
                    });
                }
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
    /// Factorises `a` with the first attempt that succeeds. The LU methods
    /// always succeed; whether their solutions are finite shows only in
    /// [`Factor::solve`].
    ///
    /// # Errors
    ///
    /// [`SolveError::Shape`] when `a` is not square, and
    /// [`SolveError::NotFinite`] when no attempt factorises it.
    pub fn new(a: MatRef<'_, f64>, attempts: &[Attempt]) -> Result<Self, SolveError> {
        check_shape(a, a.nrows())?;
        for attempt in attempts {
            for (matrix, ridge) in candidates(a, attempt.ridge) {
                if let Some(mut factor) = Self::with_method(matrix.as_ref(), attempt.method) {
                    factor.ridge = ridge;
                    return Ok(factor);
                }
            }
        }
        Err(SolveError::NotFinite)
    }

    #[must_use]
    pub const fn method(&self) -> Method {
        self.method
    }

    /// The ridge added to the diagonal before factorising, 0 when none was.
    #[must_use]
    pub const fn ridge(&self) -> f64 {
        self.ridge
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

    fn with_method(a: MatRef<'_, f64>, method: Method) -> Option<Self> {
        let kind = match method {
            Method::Cholesky => Kind::Cholesky(cholesky(a)?),
            Method::Lblt => Kind::Lblt(on_one_thread(|| a.lblt(Side::Lower))),
            Method::PartialPivLu => Kind::PartialPivLu(on_one_thread(|| a.partial_piv_lu())),
            Method::FullPivLu => Kind::FullPivLu(on_one_thread(|| a.full_piv_lu())),
        };
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
        llt::factor::LltRegularization::default(),
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
    Some(lower)
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
