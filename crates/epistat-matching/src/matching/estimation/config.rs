//! Configuration for covariate encoding, propensity estimation and Mahalanobis preparation.

use crate::types::DistanceCaliper;
use rapidhash::RapidHashMap;

/// Missing-value handling for in-crate covariate preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingValuePolicy {
    /// Reject missing values.
    #[default]
    Error,
    /// Impute numeric columns with observed means and map categorical missing values
    /// to a dedicated level.
    Impute,
}

/// Covariate encoding options for design-matrix construction.
#[derive(bon::Builder, Debug, Clone)]
pub struct CovariateEncodingConfig {
    /// Explicit covariate keys. Empty means all observed keys in deterministic order.
    #[builder(default)]
    pub covariate_keys: Vec<String>,
    /// Include an intercept column.
    #[builder(default = false)]
    pub include_intercept: bool,
    /// Drop the first categorical level for each encoded factor.
    #[builder(default = true)]
    pub drop_first_categorical_level: bool,
    /// Missing-value policy.
    #[builder(default)]
    pub missing_value_policy: MissingValuePolicy,
}

impl Default for CovariateEncodingConfig {
    fn default() -> Self {
        Self {
            covariate_keys: Vec::new(),
            include_intercept: false,
            drop_first_categorical_level: true,
            missing_value_policy: MissingValuePolicy::Error,
        }
    }
}

/// The link of the generalised linear propensity model.
///
/// These are the links of R's `binomial()` that MatchIt's
/// `distance = "glm"` accepts for a propensity score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PropensityLink {
    /// `eta = ln(p / (1 - p))`.
    #[default]
    Logit,
    /// `eta = Phi^-1(p)`, with `Phi` the standard normal distribution function.
    Probit,
    /// `eta = ln(-ln(1 - p))`.
    Cloglog,
}

/// Settings of the generalised linear propensity model, fitted by
/// [`epistat_glm::Glm`] as R's `glm()` does, with no penalty.
///
/// The defaults are those of R's `glm.control`: 25 iterations and a relative
/// deviance change of 1e-8.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct LogisticRegressionConfig {
    /// Maximum IRLS iterations.
    #[builder(default = 25_usize)]
    pub max_iter: usize,
    /// Convergence tolerance on the relative change of the deviance, R's
    /// `epsilon`.
    #[builder(default = 1e-8_f64)]
    pub tolerance: f64,
    /// The link between the propensity score and the linear predictor.
    #[builder(default)]
    pub link: PropensityLink,
}

impl Default for LogisticRegressionConfig {
    fn default() -> Self {
        Self {
            max_iter: 25,
            tolerance: 1e-8,
            link: PropensityLink::Logit,
        }
    }
}

/// Elastic-net logistic settings for in-crate propensity estimation.
///
/// The fit minimises the mean log-loss plus
/// `lambda * (alpha * |b|_1 + (1 - alpha) / 2 * |b|_2^2)`, with the intercept
/// unpenalised and the covariates used as given, not standardised. This is the
/// objective of R's `glmnet(family = "binomial", standardize = FALSE)`, and the
/// coefficients agree with it to about 1e-7
/// (`tests/r_glmnet.rs`). `glmnet`'s default `standardize = TRUE`
/// penalises the standardised coefficients and gives different estimates.
///
/// The optimiser is IRLS with cyclic coordinate descent on the penalised
/// weighted least-squares approximation, as in `glmnet` (Friedman, Hastie and
/// Tibshirani 2010, doi:10.18637/jss.v033.i01). It stops when the largest
/// change of a coefficient, squared and weighted by the weighted variance of its
/// column, is at most `tolerance`, so the rule does not depend on covariate
/// scale. With a column standardised to unit weighted variance, the default
/// 1e-20 corresponds to a coefficient change of about 1e-10. The defaults
/// converge on correlated designs and on covariates of large scale. [`PropensityScoreEstimate::converged`](crate::PropensityScoreEstimate::converged)
/// is `false` when `max_iter` IRLS steps are not enough.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct ElasticNetLogisticConfig {
    /// Maximum IRLS steps.
    #[builder(default = 100_usize)]
    pub max_iter: usize,
    /// Convergence tolerance on the largest variance-weighted squared coefficient change.
    #[builder(default = 1e-20_f64)]
    pub tolerance: f64,
    /// Elastic-net penalty strength.
    #[builder(default = 1e-2_f64)]
    pub lambda: f64,
    /// Elastic-net mixing parameter (0 = ridge, 1 = lasso).
    #[builder(default = 0.5_f64)]
    pub alpha: f64,
    /// Numeric stability clipping for predicted probabilities.
    #[builder(default = 1e-8_f64)]
    pub probability_clip: f64,
    /// Scale each penalised column to unit weighted standard deviation before
    /// fitting, as `glmnet(standardize = TRUE)` does. The coefficients are
    /// returned on the scale of the covariates.
    #[builder(default = false)]
    pub standardize: bool,
}

impl Default for ElasticNetLogisticConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-20,
            lambda: 1e-2,
            alpha: 0.5,
            probability_clip: 1e-8,
            standardize: false,
        }
    }
}

/// Which cross-validated penalty `lambda` is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LambdaRule {
    /// The `lambda` with the smallest mean cross-validated deviance, the
    /// largest of them if several tie (`lambda.min` of `cv.glmnet`).
    Min,
    /// The largest `lambda` whose mean deviance is within one standard error
    /// of the smallest (`lambda.1se` of `cv.glmnet`), which is the one MatchIt
    /// uses.
    #[default]
    OneStandardError,
}

/// Elastic-net logistic settings with the penalty chosen by cross-validation,
/// as MatchIt does for `distance = "lasso"`, `"ridge"` and `"elasticnet"`.
///
/// The folds come from `epistat-rng` seeded with `seed`: the row indices are
/// dealt to the `nfolds` folds in turn and shuffled, so the fold sizes differ
/// by at most one. R's `cv.glmnet` draws its folds the same way with R's
/// generator, so the folds, and the chosen `lambda`, differ from R's unless
/// R is given the same `foldid`. The `lambda` path is `glmnet`'s: `nlambda`
/// values from the smallest `lambda` that sets every penalised coefficient to
/// zero down to that value times `lambda_min_ratio`, equally spaced on the log
/// scale. The deviance is the binomial deviance with probabilities clipped to
/// `[1e-5, 1 - 1e-5]`. A column is standardised by default, as `glmnet` does.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct ElasticNetCvConfig {
    /// Seed of the random number stream that assigns the folds.
    pub seed: u64,
    /// Elastic-net mixing parameter (0 = ridge, 1 = lasso).
    #[builder(default = 0.5_f64)]
    pub alpha: f64,
    /// Number of folds, at least 3.
    #[builder(default = 10_usize)]
    pub nfolds: usize,
    /// Number of `lambda` values on the path.
    #[builder(default = 100_usize)]
    pub nlambda: usize,
    /// Smallest `lambda` as a fraction of the largest. `None` is `glmnet`'s
    /// default, 1e-4 when there are more rows than covariates and 1e-2
    /// otherwise.
    pub lambda_min_ratio: Option<f64>,
    /// The rule that picks `lambda` from the cross-validation curve.
    #[builder(default)]
    pub rule: LambdaRule,
    /// Scale each penalised column to unit weighted standard deviation.
    #[builder(default = true)]
    pub standardize: bool,
    /// Maximum IRLS steps per fit.
    #[builder(default = 100_usize)]
    pub max_iter: usize,
    /// Convergence tolerance of each fit, as in [`ElasticNetLogisticConfig`].
    #[builder(default = 1e-20_f64)]
    pub tolerance: f64,
    /// Numeric stability clipping for predicted probabilities.
    #[builder(default = 1e-8_f64)]
    pub probability_clip: f64,
}

/// The estimand whose balance conditions a covariate balancing propensity
/// score satisfies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CbpsEstimand {
    /// The average treatment effect in the treated: the controls, weighted by
    /// the odds of treatment, are balanced to the treated. MatchIt's default.
    #[default]
    Att,
    /// The average treatment effect: both groups, weighted by the inverse of
    /// the probability of their own treatment, are balanced to the sample.
    Ate,
}

/// The estimator of a covariate balancing propensity score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CbpsMethod {
    /// Score and balance conditions together, weighted by the inverse
    /// covariance of the moments: CBPS's `method = "over"`.
    #[default]
    Over,
    /// The balance conditions alone, which hold exactly at the fit: CBPS's
    /// `method = "exact"`.
    Exact,
}

/// Settings of the covariate balancing propensity score of Imai and Ratkovic
/// (2014), as MatchIt's `distance = "cbps"` fits it with `CBPS::CBPS()`.
///
/// The fit is the two-step generalised method of moments estimator, which is
/// the default of CBPS (`twostep = TRUE`). It is described in the module
/// documentation of the implementation. The design must have an `intercept`
/// column.
///
/// Each optimisation stops when the gradient norm of the loss is below
/// `gradient_tolerance`, or when the loss changes by less than four times its
/// estimated rounding error at the start of the search, `eps = f64::EPSILON`. The default
/// gradient tolerance is `eps^(2/3)`, about 3.7e-11. The loss is in the
/// orthogonalised coordinates, in which every column of the design has unit
/// length, and a point with gradient norm `g` is within `g / lambda` of the
/// minimum, with `lambda` the smallest eigenvalue of the Hessian of the loss.
/// The reference fits have `lambda` between 3e-4 and 3e-3, so the default
/// puts the linear predictor within about 1e-7 of the minimum.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct CbpsConfig {
    /// The estimand.
    #[builder(default)]
    pub estimand: CbpsEstimand,
    /// Over-identified or exact.
    #[builder(default)]
    pub method: CbpsMethod,
    /// Maximum BFGS iterations of each optimisation, CBPS's `iterations`.
    #[builder(default = 1000_usize)]
    pub max_iter: usize,
    /// Gradient norm at which an optimisation has converged.
    #[builder(default = f64::EPSILON.powf(2.0 / 3.0))]
    pub gradient_tolerance: f64,
}

impl Default for CbpsConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

/// In-crate propensity estimator selection.
#[derive(Debug, Clone, Default)]
pub enum PropensityEstimator {
    /// A binomial generalised linear model with the link of
    /// [`LogisticRegressionConfig::link`], logit by default, fitted as R's
    /// `glm()` does.
    #[default]
    GlmLogit,
    /// Elastic-net penalized logistic regression at a given `lambda`.
    ElasticNetLogit(ElasticNetLogisticConfig),
    /// Elastic-net penalized logistic regression with `lambda` chosen by
    /// cross-validation.
    ElasticNetCv(ElasticNetCvConfig),
    /// The covariate balancing propensity score.
    Cbps(CbpsConfig),
}

/// Output scale used for propensity-score distance values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PropensityScoreOutputScale {
    /// Use probability scale.
    #[default]
    Probability,
    /// Use the linear predictor, the scale of the link (`logit` for the logit
    /// link): MatchIt's `link = "linear.logit"` and its siblings.
    LinearPredictor,
}

/// Combined configuration for in-crate propensity score estimation.
#[derive(bon::Builder, Debug, Clone)]
pub struct PropensityScoreConfig {
    /// Covariate encoding options.
    #[builder(default = CovariateEncodingConfig::builder().include_intercept(true).build())]
    pub encoding: CovariateEncodingConfig,
    /// Estimator configuration.
    #[builder(default)]
    pub estimator: PropensityEstimator,
    /// Solver options.
    #[builder(default)]
    pub logistic: LogisticRegressionConfig,
    /// Output scale used for matching distance.
    #[builder(default)]
    pub output_scale: PropensityScoreOutputScale,
    /// Optional caliper on selected output scale.
    pub caliper: Option<DistanceCaliper>,
    /// Sampling weights by record id, entered into the fit as prior weights as
    /// MatchIt passes `s.weights` to `glm()`. Every record needs a finite,
    /// non-negative weight. `None` is a weight of 1 for every record.
    pub sampling_weights: Option<RapidHashMap<String, f64>>,
    /// Refit the model on the units that common-support trimming keeps, as
    /// `matchit(reestimate = TRUE)` does. It needs a trimming policy, which
    /// [`prepare_propensity_distance_config_with_discard`](super::prepare_propensity_distance_config_with_discard)
    /// and [`estimate_propensity_and_match`](super::estimate_propensity_and_match)
    /// take from the caller and the matching criteria.
    #[builder(default = false)]
    pub reestimate: bool,
}

impl Default for PropensityScoreConfig {
    fn default() -> Self {
        Self {
            encoding: CovariateEncodingConfig::builder()
                .include_intercept(true)
                .build(),
            estimator: PropensityEstimator::default(),
            logistic: LogisticRegressionConfig::default(),
            output_scale: PropensityScoreOutputScale::Probability,
            caliper: None,
            sampling_weights: None,
            reestimate: false,
        }
    }
}

/// Mahalanobis preprocessing configuration using in-crate covariate encoding.
#[derive(bon::Builder, Debug, Clone)]
pub struct MahalanobisPreparationConfig {
    /// Covariate encoding options.
    #[builder(default)]
    pub encoding: CovariateEncodingConfig,
    /// Covariance estimation strategy. The default is MatchIt's.
    #[builder(default = MahalanobisCovarianceStrategy::PooledByDegreesOfFreedom)]
    pub covariance_strategy: MahalanobisCovarianceStrategy,
    /// Optional covariate transformation before covariance estimation.
    #[builder(default = MahalanobisTransform::Raw)]
    pub transform: MahalanobisTransform,
    /// Initial ridge added to covariance diagonal before inversion.
    #[builder(default = 1e-8_f64)]
    pub ridge: f64,
    /// Number of ridge escalation attempts.
    #[builder(default = 8_usize)]
    pub max_ridge_attempts: usize,
}

impl Default for MahalanobisPreparationConfig {
    fn default() -> Self {
        Self {
            encoding: CovariateEncodingConfig::default(),
            covariance_strategy: MahalanobisCovarianceStrategy::PooledByDegreesOfFreedom,
            transform: MahalanobisTransform::Raw,
            ridge: 1e-8,
            max_ridge_attempts: 8,
        }
    }
}

/// Covariance strategy used for Mahalanobis preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MahalanobisCovarianceStrategy {
    /// Average of the anchor and candidate covariance matrices, each with the
    /// same weight, whatever the group sizes.
    PooledWithinGroups,
    /// Pooled within-group covariance: the sums of squares and cross-products
    /// about each group's own mean, over the number of units less two. This is
    /// the matrix MatchIt's `mahalanobis_dist()` uses, and it equals
    /// [`Self::PooledWithinGroups`] only when the groups are the same size.
    PooledByDegreesOfFreedom,
    /// Covariance over the full sample after encoding.
    FullSample,
}

/// Covariate transformation used before Mahalanobis covariance estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MahalanobisTransform {
    /// No transformation.
    Raw,
    /// Rank-transform each column (average ranks for ties).
    Rank,
    /// Rank-transform each column (average ranks for ties) and rescale the
    /// covariance of the ranks so that every diagonal entry is the variance of
    /// the untied ranks `1, ..., n`, `n (n + 1) / 12` (Rosenbaum 2010, Design of
    /// Observational Studies, section 8.3). Ties shrink the variance of a
    /// column's ranks, and the rescaling stops a tied column from counting more
    /// in the distance than an untied one. With the full-sample covariance this
    /// is MatchIt's `robust_mahalanobis_dist()`.
    ///
    /// Categorical covariates are encoded with a column for every level, as
    /// MatchIt does, so the covariance of the ranks is singular and the
    /// distance uses its generalised inverse (eigenvalues at most 1e-8 of the
    /// largest are dropped). [`MahalanobisDistancePreparation::covariance_rank`](super::MahalanobisDistancePreparation::covariance_rank)
    /// reports the rank kept, and `ridge` is not used.
    RankTieAdjusted,
}

/// How the columns are scaled before a Euclidean distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EuclideanScale {
    /// The columns as encoded. This is MatchIt's `euclidean_dist()`.
    #[default]
    None,
    /// Each column divided by its pooled within-group standard deviation, the
    /// two groups being the anchors and the candidates. This is MatchIt's
    /// `scaled_euclidean_dist()`: a column of only 0 and 1 has the variance
    /// `sum_g n_g p_g (1 - p_g) / n`, any other column the sum of squares about
    /// its group mean over `n - 2`.
    PooledWithinGroupSd,
}

/// Euclidean distance preprocessing configuration.
///
/// Categorical covariates get a column for every level, each scaled by
/// `1 / sqrt(2)`, so that two units in different levels are 1 apart, as in
/// MatchIt. `encoding.drop_first_categorical_level` is ignored.
#[derive(bon::Builder, Debug, Clone, Default)]
pub struct EuclideanPreparationConfig {
    /// Covariate encoding options.
    #[builder(default)]
    pub encoding: CovariateEncodingConfig,
    /// Scaling of the columns.
    #[builder(default)]
    pub scale: EuclideanScale,
}
