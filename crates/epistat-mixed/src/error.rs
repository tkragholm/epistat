/// Why a mixed model could not be fitted or evaluated.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MixedError {
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
    /// A value of the design, the response, the slope or the offset is not
    /// finite.
    #[error("{what} has a value that is not finite at row {row}")]
    NotFinite {
        /// The name of the input.
        what: &'static str,
        /// The row.
        row: usize,
    },
    /// A prior weight is negative or not finite, or, for the linear model, not
    /// positive.
    #[error("prior weight {row} is {value}, which the model does not accept")]
    Weight {
        /// The row.
        row: usize,
        /// The weight.
        value: f64,
    },
    /// A response lies outside the support of the family.
    #[error("response {row} is {value}, outside the family's support")]
    Response {
        /// The row.
        row: usize,
        /// The response.
        value: f64,
    },
    /// There are no rows, no columns, or no more rows than coefficients.
    #[error("{0}")]
    Size(&'static str),
    /// The columns of the design are linearly dependent, so the fixed
    /// effects are not identified.
    #[error("the design matrix does not have full column rank")]
    RankDeficient,
    /// The family and link are not a model this crate fits.
    #[error(
        "only the binomial family with the logit link and the Poisson family with the log link are fitted"
    )]
    Family,
    /// The number of quadrature nodes is outside `1..=100`.
    #[error("the number of quadrature nodes must lie in 1..=100, not {0}")]
    Nodes(usize),
    /// A tolerance or iteration limit is not positive.
    #[error("{0} must be positive and finite")]
    Control(&'static str),
    /// A parameter given for an evaluation has the wrong length or is not
    /// finite.
    #[error("the parameters must be {0} finite values")]
    Parameters(usize),
    /// The starting fit without random effects failed.
    #[error(transparent)]
    Start(#[from] epistat_glm::GlmError),
    /// The conditional mode of a cluster could not be found.
    #[error("the conditional mode of cluster {0} did not converge")]
    Mode(usize),
    /// The criterion is not finite at the start of the optimisation.
    #[error("the criterion is not finite at the starting values")]
    NotFiniteStart,
}
