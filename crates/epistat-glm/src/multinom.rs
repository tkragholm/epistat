//! Multinomial logistic regression, the generalised propensity score model.
//!
//! [`Multinom::fit`] fits the model of R's `nnet::multinom()` without a
//! penalty: with classes `0, ..., K - 1`, class 0 is the baseline, and the
//! probability of class `k` for a row `x` is
//!
//! ```text
//! P(k | x) = exp(x'b_k) / (1 + sum_{l=1}^{K-1} exp(x'b_l)),   k >= 1
//! P(0 | x) = 1 / (1 + sum_{l=1}^{K-1} exp(x'b_l))
//! ```
//!
//! with one coefficient vector `b_k` of length `p` for each class but the
//! baseline. The log-likelihood with case weights `w` is
//! `sum w (x'b_y - log(1 + sum exp(x'b_l)))`. It is concave, with gradient
//! `sum w (1[y = k] - P(k | x)) x` in block `k` and Hessian blocks
//! `-sum w P(k | x) (1[k = l] - P(l | x)) x x'`.
//!
//! # Algorithm
//!
//! Newton's method from `b = 0`, which solves the score equations with the
//! exact Hessian. A step that does not raise the log-likelihood is halved, at
//! most 30 times. The Hessian is factorised by Cholesky through
//! `epistat_linalg`, so a design that is collinear, or a class with no rows,
//! is an error and not a repaired fit. The fit has converged when the largest
//! step is below `tolerance * (1 + largest |b|)`, or when no step raises the
//! log-likelihood and the Newton step is below `sqrt(eps) * (1 + largest |b|)`. `nnet::multinom()` instead
//! runs BFGS to a relative change of `1e-8` in its criterion, which stops with
//! coefficients about `1e-4` from the maximum, so a comparison with its
//! default fit is limited by R's stopping rule and not by ours. The test runs
//! R with `reltol = 1e-14` to compare at the precision of the arithmetic.
//!
//! With complete or quasi-complete separation the likelihood has no maximum.
//! The Hessian then becomes singular and the fit is
//! [`MultinomError::Separation`], or the iteration stops at its limit with
//! [`MultinomFit::converged`] false and [`MultinomFit::fitted_at_boundary`]
//! true, which is the case when a fitted probability lies within ten machine
//! epsilons of 0 or 1.
//!
//! Sums run over the rows in order, so results do not depend on the number of
//! threads.
//!
//! The fit is checked against `nnet::multinom()` 7.3-21: coefficients,
//! fitted probabilities, residual deviance and the standard errors of
//! `summary()`.

use epistat_linalg::{Attempt, Factor, Mat, MatRef, Method, SolveError};
use rayon::prelude::*;

/// Fitting controls for the multinomial logit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Multinom {
    /// The largest step, relative to `1 + max |b|`, at which the fit has
    /// converged.
    pub tolerance: f64,
    /// The most Newton iterations.
    pub max_iterations: usize,
}

impl Default for Multinom {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 100,
        }
    }
}

/// Why a multinomial logit could not be fitted.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MultinomError {
    /// An input has the wrong length.
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        /// The name of the input.
        what: &'static str,
        /// The length it needed.
        expected: usize,
        /// The length it had.
        found: usize,
    },
    /// There are fewer than two classes, or a label is not below the number
    /// of classes.
    #[error(
        "class {value} of row {row} is outside 0..{classes}, or there are fewer than two classes"
    )]
    Class {
        /// The row.
        row: usize,
        /// Its label.
        value: usize,
        /// The number of classes.
        classes: usize,
    },
    /// A class has no row with positive weight.
    #[error("class {class} has no row with positive weight")]
    EmptyClass {
        /// The class.
        class: usize,
    },
    /// A weight is negative or not finite.
    #[error("weight {index} is {value}, which is negative or not finite")]
    Weight {
        /// The row.
        index: usize,
        /// Its value.
        value: f64,
    },
    /// The design has a missing or infinite value.
    #[error("the design matrix has a value that is not finite")]
    Design,
    /// The Hessian became singular with fitted probabilities at 0 or 1: the
    /// classes are separated by the covariates and the likelihood has no
    /// maximum.
    #[error("the classes are separated by the covariates")]
    Separation,
    /// The Hessian is not positive definite: the design is collinear.
    #[error("the Hessian is singular, the design is collinear")]
    Singular,
    /// The tolerance is not positive and finite.
    #[error("the tolerance must be positive and finite")]
    Tolerance,
    /// A linear solve failed.
    #[error(transparent)]
    Solve(#[from] SolveError),
}

/// A fitted multinomial logit.
#[derive(Clone, Debug, PartialEq)]
pub struct MultinomFit {
    /// The number of classes `K`.
    pub classes: usize,
    /// The number of columns of the design.
    pub columns: usize,
    /// The coefficients `b_1, ..., b_{K-1}` of the classes after the baseline,
    /// class by class, `(K - 1) * p` values.
    pub coefficients: Vec<f64>,
    /// The fitted probabilities, row by row, `n * K` values.
    pub probabilities: Vec<f64>,
    /// The log-likelihood at the fit.
    pub log_likelihood: f64,
    /// The residual deviance, `-2` times the log-likelihood, as `nnet` reports.
    pub deviance: f64,
    /// The covariance of the coefficients, the inverse of the information,
    /// row-major `(K - 1) p` square in the order of `coefficients`.
    pub covariance: Vec<f64>,
    /// The number of Newton iterations.
    pub iterations: usize,
    /// Whether the largest step fell below the tolerance.
    pub converged: bool,
    /// Whether a fitted probability lies within ten machine epsilons of 0 or
    /// 1, which marks separation.
    pub fitted_at_boundary: bool,
}

impl MultinomFit {
    /// The coefficient of column `j` for class `k`, with `k` in
    /// `1..classes`.
    #[must_use]
    pub fn coefficient(&self, class: usize, column: usize) -> f64 {
        self.coefficients[(class - 1) * self.columns + column]
    }

    /// The fitted probabilities of row `i`, one per class.
    #[must_use]
    pub fn row_probabilities(&self, row: usize) -> &[f64] {
        &self.probabilities[row * self.classes..(row + 1) * self.classes]
    }

    /// The standard errors of the coefficients, in the order of
    /// `coefficients`.
    #[must_use]
    pub fn standard_errors(&self) -> Vec<f64> {
        let q = self.coefficients.len();
        (0..q)
            .map(|i| self.covariance[i * q + i].max(0.0).sqrt())
            .collect()
    }

    /// The log of the odds of class `class` against the baseline for a row of
    /// covariates, `x'b_class`, with `class` in `1..classes`.
    #[must_use]
    pub fn linear_predictor(&self, class: usize, x: &[f64]) -> f64 {
        let b = &self.coefficients[(class - 1) * self.columns..class * self.columns];
        let mut s = 0.0;
        for (xi, bi) in x.iter().zip(b) {
            s += xi * bi;
        }
        s
    }
}

impl Multinom {
    /// Fits the model to the class labels `y` (each in `0..classes`) on the
    /// columns of `x`, which holds the intercept if the model has one.
    /// `weights` are case weights, 1 when `None`; rows of weight 0 do not
    /// enter.
    ///
    /// # Errors
    ///
    /// See [`MultinomError`].
    pub fn fit(
        &self,
        x: MatRef<'_, f64>,
        y: &[usize],
        classes: usize,
        weights: Option<&[f64]>,
    ) -> Result<MultinomFit, MultinomError> {
        let (n, p) = (x.nrows(), x.ncols());
        let ones = vec![1.0; n];
        let w = weights.unwrap_or(&ones);
        let check = |what, expected, found| {
            if expected == found {
                Ok(())
            } else {
                Err(MultinomError::Length {
                    what,
                    expected,
                    found,
                })
            }
        };
        check("y", n, y.len())?;
        check("weights", n, w.len())?;
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(MultinomError::Tolerance);
        }
        if let Some((index, &value)) = w
            .iter()
            .enumerate()
            .find(|&(_, &v)| !(v.is_finite() && v >= 0.0))
        {
            return Err(MultinomError::Weight { index, value });
        }
        if classes < 2 {
            return Err(MultinomError::Class {
                row: 0,
                value: classes,
                classes,
            });
        }
        if let Some((row, &value)) = y.iter().enumerate().find(|&(_, &c)| c >= classes) {
            return Err(MultinomError::Class {
                row,
                value,
                classes,
            });
        }
        for class in 0..classes {
            if !(0..n).any(|i| y[i] == class && w[i] > 0.0) {
                return Err(MultinomError::EmptyClass { class });
            }
        }
        if (0..n).any(|i| (0..p).any(|j| !x[(i, j)].is_finite())) {
            return Err(MultinomError::Design);
        }

        let q = (classes - 1) * p;
        let mut beta = vec![0.0; q];
        let (mut loglik, mut probs) = evaluate(x, y, w, classes, &beta);
        let mut converged = false;
        let mut iterations = 0;
        for iteration in 1..=self.max_iterations {
            iterations = iteration;
            let (gradient, hessian) = derivatives(x, y, w, classes, &probs);
            let factor = Factor::new(hessian.as_ref(), &[Attempt::plain(Method::Cholesky)])
                .map_err(|_| singular(&probs))?;
            let rhs = Mat::from_fn(q, 1, |i, _| gradient[i]);
            let step = factor.solve(rhs.as_ref())?;
            let mut scale = 1.0;
            let mut accepted = None;
            for _ in 0..30 {
                let trial: Vec<f64> = (0..q).map(|i| beta[i] + scale * step[(i, 0)]).collect();
                let (l, pr) = evaluate(x, y, w, classes, &trial);
                if l.is_finite() && l >= loglik {
                    accepted = Some((trial, l, pr));
                    break;
                }
                scale /= 2.0;
            }
            let Some((trial, l, pr)) = accepted else {
                // No ascent along the step, even halved 30 times. That is the
                // maximum to rounding only when the full Newton step is itself
                // negligible, at the square root of the machine epsilon
                // relative to the coefficients, where the gain in the
                // log-likelihood is below the rounding error of the sum.
                let largest_step = (0..q).map(|i| step[(i, 0)].abs()).fold(0.0_f64, f64::max);
                let largest_beta = beta.iter().map(|b| b.abs()).fold(0.0_f64, f64::max);
                converged = largest_step <= f64::EPSILON.sqrt() * (1.0 + largest_beta);
                break;
            };
            let mut largest_step = 0.0_f64;
            let mut largest_beta = 0.0_f64;
            for i in 0..q {
                largest_step = largest_step.max((trial[i] - beta[i]).abs());
                largest_beta = largest_beta.max(trial[i].abs());
            }
            beta = trial;
            loglik = l;
            probs = pr;
            if largest_step <= self.tolerance * (1.0 + largest_beta) {
                converged = true;
                break;
            }
        }
        // The information at the final coefficients.
        let (_, hessian) = derivatives(x, y, w, classes, &probs);
        let covariance = Factor::new(hessian.as_ref(), &[Attempt::plain(Method::Cholesky)])
            .map_err(|_| MultinomError::Singular)?
            .solve(Mat::<f64>::identity(q, q).as_ref())?;
        let boundary = at_boundary(&probs);
        Ok(MultinomFit {
            classes,
            columns: p,
            coefficients: beta,
            probabilities: probs,
            log_likelihood: loglik,
            deviance: -2.0 * loglik,
            covariance: (0..q * q).map(|k| covariance[(k / q, k % q)]).collect(),
            iterations,
            converged,
            fitted_at_boundary: boundary,
        })
    }
}

fn at_boundary(probs: &[f64]) -> bool {
    probs
        .iter()
        .any(|&v| v < 10.0 * f64::EPSILON || v > 1.0 - 10.0 * f64::EPSILON)
}

fn singular(probs: &[f64]) -> MultinomError {
    if at_boundary(probs) {
        MultinomError::Separation
    } else {
        MultinomError::Singular
    }
}

/// Rows from which the per-row work runs in parallel.
const PARALLEL_FROM: usize = 2048;

/// The log-likelihood and the fitted probabilities at `beta`.
///
/// The probabilities and each row's term are computed row by row, in
/// parallel for many rows, and the terms are added in the order of the rows.
fn evaluate(
    x: MatRef<'_, f64>,
    y: &[usize],
    w: &[f64],
    classes: usize,
    beta: &[f64],
) -> (f64, Vec<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let mut probs = vec![0.0; n * classes];
    let mut terms = vec![0.0; n];
    let row = |eta: &mut Vec<f64>, i: usize, probs: &mut [f64], term: &mut f64| {
        let mut top = 0.0_f64;
        for k in 1..classes {
            let mut s = 0.0;
            for j in 0..p {
                s += x[(i, j)] * beta[(k - 1) * p + j];
            }
            eta[k] = s;
            top = top.max(s);
        }
        eta[0] = 0.0;
        let mut sum = 0.0;
        for e in eta.iter() {
            sum += (e - top).exp();
        }
        let log_sum = top + sum.ln();
        for k in 0..classes {
            probs[k] = (eta[k] - log_sum).exp();
        }
        *term = w[i] * (eta[y[i]] - log_sum);
    };
    if n >= PARALLEL_FROM {
        probs
            .par_chunks_mut(classes)
            .zip(terms.par_iter_mut())
            .enumerate()
            .for_each_init(
                || vec![0.0; classes],
                |eta, (i, (probs, term))| row(eta, i, probs, term),
            );
    } else {
        let mut eta = vec![0.0; classes];
        for (i, (probs, term)) in probs.chunks_mut(classes).zip(&mut terms).enumerate() {
            row(&mut eta, i, probs, term);
        }
    }
    let mut loglik = 0.0;
    for term in terms {
        loglik += term;
    }
    (loglik, probs)
}

/// The gradient of the log-likelihood and the negative Hessian, which is the
/// information, at the fitted probabilities `probs`.
///
/// Every entry is a sum over the rows with nonzero weight, taken in the order
/// of the rows, and the entries are computed in parallel, so the result does
/// not depend on the number of threads.
fn derivatives(
    x: MatRef<'_, f64>,
    y: &[usize],
    w: &[f64],
    classes: usize,
    probs: &[f64],
) -> (Vec<f64>, Mat<f64>) {
    let (n, p) = (x.nrows(), x.ncols());
    let q = (classes - 1) * p;
    let active: Vec<usize> = (0..n).filter(|&i| w[i] != 0.0).collect();
    let in_parallel = n >= PARALLEL_FROM;

    let gradient_entry = |entry: usize| {
        let (k, j) = (entry / p + 1, entry % p);
        let mut sum = 0.0;
        for &i in &active {
            let r = w[i] * (f64::from(u8::from(y[i] == k)) - probs[i * classes + k]);
            sum += r * x[(i, j)];
        }
        sum
    };
    // The entry in row `(k - 1) p + a` and column `(l - 1) p + b` of the lower
    // block triangle, `l <= k`.
    let information_entry = |(k, l, a, b): (usize, usize, usize, usize)| {
        let mut sum = 0.0;
        for &i in &active {
            let pi = &probs[i * classes..(i + 1) * classes];
            let delta = if k == l { pi[k] } else { 0.0 };
            let c = w[i] * (delta - pi[k] * pi[l]);
            let ca = c * x[(i, a)];
            sum += ca * x[(i, b)];
        }
        sum
    };
    let entries: Vec<(usize, usize, usize, usize)> = (1..classes)
        .flat_map(|k| (1..=k).map(move |l| (k, l)))
        .flat_map(|(k, l)| (0..p).flat_map(move |a| (0..p).map(move |b| (k, l, a, b))))
        .collect();
    let (gradient, lower): (Vec<f64>, Vec<f64>) = if in_parallel {
        (
            (0..q).into_par_iter().map(gradient_entry).collect(),
            entries.par_iter().copied().map(information_entry).collect(),
        )
    } else {
        (
            (0..q).map(gradient_entry).collect(),
            entries.iter().copied().map(information_entry).collect(),
        )
    };
    let mut information = Mat::<f64>::zeros(q, q);
    for (&(k, l, a, b), value) in entries.iter().zip(lower) {
        information[((k - 1) * p + a, (l - 1) * p + b)] = value;
    }
    // Mirror the lower block triangle into the upper.
    for k in 1..classes {
        for l in 1..k {
            for a in 0..p {
                for b in 0..p {
                    information[((l - 1) * p + b, (k - 1) * p + a)] =
                        information[((k - 1) * p + a, (l - 1) * p + b)];
                }
            }
        }
    }
    (gradient, information)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate_reference(
        x: MatRef<'_, f64>,
        y: &[usize],
        w: &[f64],
        classes: usize,
        beta: &[f64],
    ) -> (f64, Vec<f64>) {
        let (n, p) = (x.nrows(), x.ncols());
        let mut probs = vec![0.0; n * classes];
        let mut loglik = 0.0;
        let mut eta = vec![0.0; classes];
        for i in 0..n {
            let mut top = 0.0_f64;
            for k in 1..classes {
                let mut s = 0.0;
                for j in 0..p {
                    s += x[(i, j)] * beta[(k - 1) * p + j];
                }
                eta[k] = s;
                top = top.max(s);
            }
            eta[0] = 0.0;
            let mut sum = 0.0;
            for e in &eta {
                sum += (e - top).exp();
            }
            let log_sum = top + sum.ln();
            for k in 0..classes {
                probs[i * classes + k] = (eta[k] - log_sum).exp();
            }
            loglik += w[i] * (eta[y[i]] - log_sum);
        }
        (loglik, probs)
    }

    fn derivatives_reference(
        x: MatRef<'_, f64>,
        y: &[usize],
        w: &[f64],
        classes: usize,
        probs: &[f64],
    ) -> (Vec<f64>, Mat<f64>) {
        let (n, p) = (x.nrows(), x.ncols());
        let q = (classes - 1) * p;
        let mut gradient = vec![0.0; q];
        let mut information = Mat::<f64>::zeros(q, q);
        for i in 0..n {
            if w[i] == 0.0 {
                continue;
            }
            let pi = &probs[i * classes..(i + 1) * classes];
            for k in 1..classes {
                let r = w[i] * (f64::from(u8::from(y[i] == k)) - pi[k]);
                for j in 0..p {
                    gradient[(k - 1) * p + j] += r * x[(i, j)];
                }
                for l in 1..=k {
                    let delta = if k == l { pi[k] } else { 0.0 };
                    let c = w[i] * (delta - pi[k] * pi[l]);
                    for a in 0..p {
                        let ca = c * x[(i, a)];
                        for b in 0..p {
                            information[((k - 1) * p + a, (l - 1) * p + b)] += ca * x[(i, b)];
                        }
                    }
                }
            }
        }
        // Mirror the lower block triangle into the upper.
        for k in 1..classes {
            for l in 1..k {
                for a in 0..p {
                    for b in 0..p {
                        information[((l - 1) * p + b, (k - 1) * p + a)] =
                            information[((k - 1) * p + a, (l - 1) * p + b)];
                    }
                }
            }
        }
        (gradient, information)
    }

    /// A small deterministic generator.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1_u64 << 53) as f64
        }

        fn below(&mut self, bound: usize) -> usize {
            self.unit();
            usize::try_from((self.0 >> 33) % u64::try_from(bound).unwrap()).unwrap()
        }
    }

    /// The log-likelihood, probabilities, gradient and information equal those
    /// of the sequential reference bit for bit, with zero weights, for row
    /// counts below and above the one where the work runs in parallel.
    #[test]
    fn parallel_evaluation_equals_the_sequential_reference() {
        for (n, classes, p) in [(50, 3, 3), (5000, 3, 3), (4096, 4, 2), (2500, 2, 4)] {
            let mut rng = Lcg(7 + n as u64);
            let x = Mat::from_fn(n, p, |i, j| {
                if j == 0 {
                    1.0
                } else {
                    rng.unit() * 2.0 - 1.0 + 0.01 * i as f64 % 1.0
                }
            });
            let y: Vec<usize> = (0..n).map(|_| rng.below(classes)).collect();
            let w: Vec<f64> = (0..n)
                .map(|i| if i % 11 == 0 { 0.0 } else { 0.3 + rng.unit() })
                .collect();
            let beta: Vec<f64> = (0..(classes - 1) * p).map(|_| rng.unit() - 0.5).collect();
            let (loglik, probs) = evaluate(x.as_ref(), &y, &w, classes, &beta);
            let (loglik_ref, probs_ref) = evaluate_reference(x.as_ref(), &y, &w, classes, &beta);
            assert_eq!(
                loglik.to_bits(),
                loglik_ref.to_bits(),
                "log-likelihood, n {n}"
            );
            assert!(
                probs
                    .iter()
                    .zip(&probs_ref)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            let (gradient, information) = derivatives(x.as_ref(), &y, &w, classes, &probs);
            let (gradient_ref, information_ref) =
                derivatives_reference(x.as_ref(), &y, &w, classes, &probs);
            assert!(
                gradient
                    .iter()
                    .zip(&gradient_ref)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            let q = (classes - 1) * p;
            for r in 0..q {
                for c in 0..q {
                    assert_eq!(
                        information[(r, c)].to_bits(),
                        information_ref[(r, c)].to_bits(),
                        "information ({r}, {c}), n {n}"
                    );
                }
            }
        }
    }

    /// A fit on 5000 rows is the same on pools of 1, 2, 4 and 12 threads.
    #[test]
    fn fit_is_the_same_on_every_pool_size() {
        let mut rng = Lcg(3);
        let n = 5000;
        let x = Mat::from_fn(
            n,
            3,
            |_, j| if j == 0 { 1.0 } else { rng.unit() * 2.0 - 1.0 },
        );
        let y: Vec<usize> = (0..n).map(|_| rng.below(3)).collect();
        epistat_testkit::assert_same_on_pools(&[1, 2, 4, 12], || {
            let fit = Multinom::default().fit(x.as_ref(), &y, 3, None).unwrap();
            fit.coefficients
                .into_iter()
                .chain(fit.probabilities)
                .collect()
        });
    }
}
