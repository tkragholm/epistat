//! Binomial propensity models: the generalised linear model by IRLS, and the
//! elastic net by coordinate descent.

use super::EstimationError;
use super::config::{ElasticNetLogisticConfig, LogisticRegressionConfig, PropensityLink};
use super::cross_validation::CrossValidation;
use super::encoding::CovariateMatrix;
use epistat_glm::{ElasticNet, Family, Glm, Link};
use epistat_linalg::MatRef;

#[derive(Debug, Clone)]
pub(super) struct LogisticFit {
    /// `None` for a column the model could not identify (an aliased column).
    pub(super) coefficients: Vec<Option<f64>>,
    pub(super) linear_predictor: Vec<f64>,
    pub(super) probabilities: Vec<f64>,
    pub(super) converged: bool,
    pub(super) iterations: usize,
    /// Names of the aliased columns.
    pub(super) aliased_columns: Vec<String>,
    /// The penalty of an elastic net fit.
    pub(super) lambda: Option<f64>,
    pub(super) cross_validation: Option<CrossValidation>,
    /// The statistics of a covariate balancing fit.
    pub(super) cbps: Option<super::cbps::CbpsFit>,
    /// A score at the bound the fit applies, which fitted probabilities cannot
    /// show.
    pub(super) at_boundary: bool,
}

/// The binomial generalised linear model with the configured link, fitted by
/// [`epistat_glm::Glm`] with prior weights `weights` (1 when `None`).
///
/// A column that is a linear combination of the columns before it is aliased,
/// as in R, and its coefficient is `None`. Non-convergence is reported in
/// `converged`. A first step that leaves the model's domain, or a step that
/// halving cannot correct, is an error.
pub(super) fn fit_glm(
    matrix: &CovariateMatrix,
    labels: &[f64],
    weights: Option<&[f64]>,
    config: &LogisticRegressionConfig,
) -> Result<LogisticFit, EstimationError> {
    if labels.len() != matrix.n_rows || weights.is_some_and(|w| w.len() != matrix.n_rows) {
        return Err(EstimationError::LinearSystemSolveFailed);
    }
    if config.max_iter == 0 || !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EstimationError::InvalidGlmConfig);
    }
    let link = match config.link {
        PropensityLink::Logit => Link::Logit,
        PropensityLink::Probit => Link::Probit,
        PropensityLink::Cloglog => Link::Cloglog,
    };
    let mut model = Glm::new(Family::Binomial, link);
    model.tolerance = config.tolerance;
    model.max_iterations = config.max_iter;
    let design = MatRef::from_row_major_slice(&matrix.values, matrix.n_rows, matrix.n_cols);
    let fit = model
        .fit(design, labels, weights, None, None)
        .map_err(EstimationError::GlmFitFailed)?;
    let aliased_columns = fit
        .coefficients
        .iter()
        .zip(&matrix.column_names)
        .filter(|(coefficient, _)| coefficient.is_none())
        .map(|(_, name)| name.clone())
        .collect();
    Ok(LogisticFit {
        coefficients: fit.coefficients,
        linear_predictor: fit.linear_predictor,
        probabilities: fit.fitted,
        converged: fit.converged,
        iterations: fit.iterations,
        aliased_columns,
        lambda: None,
        cross_validation: None,
        cbps: None,
        at_boundary: false,
    })
}

/// Whether a fitted probability is as close to 0 or 1 as R's `glm()` flags
/// with "fitted probabilities numerically 0 or 1 occurred": within ten
/// machine epsilons. It is the sign of complete or quasi-complete separation,
/// where the likelihood has no finite maximum, or of a near-degenerate fit.
pub(super) fn any_fitted_at_boundary(probabilities: &[f64]) -> bool {
    let margin = 10.0 * f64::EPSILON;
    probabilities
        .iter()
        .any(|p| *p < margin || *p > 1.0 - margin)
}

/// The settings of one elastic-net fit that do not depend on `lambda`.
#[derive(Debug, Clone, Copy)]
pub(super) struct NetSettings {
    pub(super) alpha: f64,
    pub(super) tolerance: f64,
    pub(super) max_iter: usize,
    pub(super) probability_clip: f64,
    pub(super) standardize: bool,
}

impl NetSettings {
    pub(super) fn validate(&self) -> Result<(), EstimationError> {
        if !self.alpha.is_finite()
            || !self.tolerance.is_finite()
            || !self.probability_clip.is_finite()
            || self.max_iter == 0
            || self.tolerance <= 0.0
            || self.probability_clip <= 0.0
            || self.probability_clip >= 0.5
        {
            return Err(EstimationError::InvalidElasticNetConfig);
        }
        Ok(())
    }
}

/// The penalty factor of each column: 0 for the intercept, 1 for the others.
pub(super) fn penalty_factors(matrix: &CovariateMatrix) -> Vec<f64> {
    matrix
        .column_names
        .iter()
        .map(|name| if name == "intercept" { 0.0 } else { 1.0 })
        .collect()
}

/// A subset of rows of the design with its response and weights, the penalised
/// columns scaled to unit weighted standard deviation when asked.
///
/// The scaling uses the weights normalised to sum to 1 and divisor 1, as
/// `glmnet(standardize = TRUE)` does. A column with no weighted variation, and
/// every unpenalised column, keeps scale 1. The design is not centred: the
/// unpenalised intercept column absorbs the means, so a fit on the scaled
/// design gives the original coefficients by dividing by the scales.
pub(super) struct NetProblem {
    pub(super) values: Vec<f64>,
    pub(super) y: Vec<f64>,
    /// Prior weights; `None` when the caller gave none, which keeps the unit
    /// weights exact.
    pub(super) weights: Option<Vec<f64>>,
    pub(super) scale: Vec<f64>,
    pub(super) n_rows: usize,
    pub(super) n_cols: usize,
}

impl NetProblem {
    pub(super) fn new(
        matrix: &CovariateMatrix,
        rows: &[usize],
        labels: &[f64],
        weights: Option<&[f64]>,
        penalty: &[f64],
        standardize: bool,
    ) -> Self {
        let n_cols = matrix.n_cols;
        let weight = |row: usize| weights.map_or(1.0, |w| w[row]);
        let total: f64 = rows.iter().map(|&row| weight(row)).sum();
        let mut scale = vec![1.0; n_cols];
        if standardize && total > 0.0 {
            for col in 0..n_cols {
                if penalty[col] <= 0.0 {
                    continue;
                }
                let mean = rows
                    .iter()
                    .map(|&row| weight(row) * matrix.row(row)[col])
                    .sum::<f64>()
                    / total;
                let variance = rows
                    .iter()
                    .map(|&row| {
                        let d = matrix.row(row)[col] - mean;
                        weight(row) * d * d
                    })
                    .sum::<f64>()
                    / total;
                let sd = variance.sqrt();
                if sd.is_finite() && sd > 0.0 {
                    scale[col] = sd;
                }
            }
        }
        let mut values = Vec::with_capacity(rows.len() * n_cols);
        for &row in rows {
            for (x, s) in matrix.row(row).iter().zip(&scale) {
                values.push(x / s);
            }
        }
        Self {
            values,
            y: rows.iter().map(|&row| labels[row]).collect(),
            weights: weights.map(|w| rows.iter().map(|&row| w[row]).collect()),
            scale,
            n_rows: rows.len(),
            n_cols,
        }
    }

    /// Fits at `lambda`, from `start` (coefficients on the scaled design).
    pub(super) fn fit(
        &self,
        settings: &NetSettings,
        lambda: f64,
        penalty: &[f64],
        start: Option<&[f64]>,
    ) -> Result<epistat_glm::ElasticNetFit, EstimationError> {
        let design = MatRef::from_row_major_slice(&self.values, self.n_rows, self.n_cols);
        let mut model = ElasticNet::new(
            Family::Binomial,
            lambda.max(0.0),
            settings.alpha.clamp(0.0, 1.0),
        );
        model.tolerance = settings.tolerance;
        model.max_iterations = settings.max_iter;
        model.probability_clip = settings.probability_clip;
        model
            .fit(
                design,
                &self.y,
                self.weights.as_deref(),
                None,
                Some(penalty),
                start,
            )
            .map_err(|_| EstimationError::InvalidElasticNetConfig)
    }

    /// Coefficients on the scale of the original covariates.
    pub(super) fn unscale(&self, scaled: &[f64]) -> Vec<f64> {
        scaled.iter().zip(&self.scale).map(|(b, s)| b / s).collect()
    }
}

/// Elastic-net logistic regression by IRLS with cyclic coordinate descent,
/// fitted by [`epistat_glm::ElasticNet`].
///
/// The objective is the mean log-loss plus
/// `lambda * (alpha * |b|_1 + (1 - alpha) / 2 * |b|_2^2)`, with the intercept
/// unpenalised. The covariates are used as given, or scaled to unit weighted
/// standard deviation first with `standardize`. It is the objective of R's
/// `glmnet(family = "binomial")` with `standardize` set the same way. The
/// column named `intercept` has penalty factor 0 and every other column 1.
/// `alpha` is clamped to `[0, 1]` and a negative `lambda` is taken as 0.
/// `weights` are prior weights (1 when `None`).
///
/// The coordinate pass stops when `max_j v_j * (change in b_j)^2` is at most
/// `tolerance`, the rule of `glmnet`'s `thresh`, and the outer loop stops on the
/// same quantity measured over a whole IRLS step. `converged` is false when
/// `max_iter` steps or the pass limit were reached first.
///
/// Reference: Friedman, Hastie and Tibshirani (2010), Regularization paths for
/// generalized linear models via coordinate descent, Journal of Statistical
/// Software 33(1), doi:10.18637/jss.v033.i01.
pub(super) fn fit_elastic_net_logistic(
    matrix: &CovariateMatrix,
    labels: &[f64],
    weights: Option<&[f64]>,
    config: &ElasticNetLogisticConfig,
) -> Result<LogisticFit, EstimationError> {
    if labels.len() != matrix.n_rows || weights.is_some_and(|w| w.len() != matrix.n_rows) {
        return Err(EstimationError::LinearSystemSolveFailed);
    }
    let settings = NetSettings {
        alpha: config.alpha,
        tolerance: config.tolerance,
        max_iter: config.max_iter,
        probability_clip: config.probability_clip,
        standardize: config.standardize,
    };
    settings.validate()?;
    if !config.lambda.is_finite() || matrix.n_rows == 0 {
        return Err(EstimationError::InvalidElasticNetConfig);
    }
    let penalty = penalty_factors(matrix);
    let rows: Vec<usize> = (0..matrix.n_rows).collect();
    let problem = NetProblem::new(
        matrix,
        &rows,
        labels,
        weights,
        &penalty,
        settings.standardize,
    );
    let fit = problem.fit(&settings, config.lambda, &penalty, None)?;
    Ok(LogisticFit {
        coefficients: problem
            .unscale(&fit.coefficients)
            .into_iter()
            .map(Some)
            .collect(),
        linear_predictor: fit.linear_predictor,
        probabilities: fit.fitted,
        converged: fit.converged,
        iterations: fit.iterations,
        aliased_columns: Vec::new(),
        lambda: Some(config.lambda.max(0.0)),
        cross_validation: None,
        cbps: None,
        at_boundary: false,
    })
}
