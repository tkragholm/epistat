//! What every fit shares: the [`Model`] trait, the fit metadata, and the
//! relaxed strategy that retries a failed fit with a growing ridge.

use epistat_linalg::MatRef;

use crate::Error;

/// Predictions and diagnostics of a fitted cost model.
pub trait Model {
    /// The predictions, such as [`TwoPartPrediction`](crate::two_part::TwoPartPrediction).
    type Prediction;
    /// The diagnostics, such as [`TwoPartReport`](crate::two_part::TwoPartReport).
    type Report;

    /// The predictions at the rows of the design `x`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `x` does not have one column per coefficient.
    fn predict(&self, x: MatRef<'_, f64>) -> Result<Self::Prediction, Error>;

    /// The diagnostics of the fit.
    fn report(&self) -> &Self::Report;
}

/// The algorithm that produced a fit.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    /// Iteratively reweighted least squares.
    Irls,
    /// Markov chain Monte Carlo.
    Mcmc,
}

/// What every fit reports about its iteration.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitMetadata {
    /// Iterations taken.
    pub iterations: usize,
    /// Whether the fit met its convergence rule.
    pub converged: bool,
    /// The algorithm used.
    pub solver: SolverKind,
    /// The attempt that succeeded under [`FitStrategy::Relaxed`], 0 for the first.
    pub fallback_attempts: usize,
}

impl Default for FitMetadata {
    fn default() -> Self {
        Self {
            iterations: 0,
            converged: false,
            solver: SolverKind::Irls,
            fallback_attempts: 0,
        }
    }
}

/// What a fit does when it fails to converge.
///
/// The default refuses: a fit that does not converge, ends at the bound of
/// the log link or meets a singular system returns the error. The relaxed
/// strategy is a repair and must be asked for; every attempt it makes is
/// reported in the fit's `attempts` and the attempt that succeeded in
/// [`FitMetadata::fallback_attempts`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FitStrategy {
    /// Fail on the first failure.
    #[default]
    Strict,
    /// Retry a failed fit with a ridge of `fallback_lambda`, then ten times
    /// that, and so on, up to `max_retries` retries.
    Relaxed {
        /// The ridge of the first retry.
        fallback_lambda: f64,
        /// The most retries. Each retry starts from the caller's starting
        /// values, not from the failed attempt.
        max_retries: usize,
    },
}

/// How one attempt of a retried fit ended.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// The attempt failed to converge or its solve failed.
    Diverged,
}

/// One failed attempt of a fit under [`FitStrategy::Relaxed`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct AttemptDiagnostics {
    /// The attempt's index, 0 for the first.
    pub attempt: usize,
    /// The ridge penalty of the attempt.
    pub lambda_used: f64,
    /// How the attempt ended.
    pub outcome: AttemptOutcome,
}

/// A successful fit from [`run_with_retries`]: the fitted value, the failed attempts
/// before it, and the index of the attempt that succeeded.
pub(crate) struct RetriedFit<T> {
    pub fit: T,
    pub attempts: Vec<AttemptDiagnostics>,
    pub attempt_idx: usize,
}

/// Run `fit` once under [`FitStrategy::Strict`], or up to `1 + max_retries` times under
/// [`FitStrategy::Relaxed`].
///
/// Attempt `k > 0` fits with `relax(base, fallback_lambda, 10^(k-1))`. A retryable
/// failure ([`Error::is_retryable`]) is recorded as `Diverged` with
/// `lambda_used(options)` and the next attempt runs. Any other error is returned at
/// once. If every attempt fails, the last retryable error is returned.
pub(crate) fn run_with_retries<O: Copy, T>(
    base: O,
    strategy: FitStrategy,
    relax: impl Fn(O, f64, f64) -> O,
    lambda_used: impl Fn(&O) -> f64,
    mut fit: impl FnMut(O) -> Result<T, Error>,
) -> Result<RetriedFit<T>, Error> {
    let max_attempts = match strategy {
        FitStrategy::Strict => 1,
        FitStrategy::Relaxed { max_retries, .. } => 1 + max_retries,
    };
    let mut current = base;
    let mut attempts = Vec::new();
    let mut last_err = None;

    for attempt_idx in 0..max_attempts {
        if attempt_idx > 0
            && let FitStrategy::Relaxed {
                fallback_lambda, ..
            } = strategy
        {
            let scale = 10.0f64.powi(i32::try_from(attempt_idx - 1).unwrap_or(0));
            current = relax(base, fallback_lambda, scale);
        }

        let lambda = lambda_used(&current);
        match fit(current) {
            Ok(fit) => {
                return Ok(RetriedFit {
                    fit,
                    attempts,
                    attempt_idx,
                });
            }
            Err(err) if err.is_retryable() => {
                attempts.push(AttemptDiagnostics {
                    attempt: attempt_idx,
                    lambda_used: lambda,
                    outcome: AttemptOutcome::Diverged,
                });
                last_err = Some(err);
            }
            Err(err) => return Err(err),
        }
    }

    Err(last_err.unwrap_or(Error::NotConverged { iterations: 0 }))
}
