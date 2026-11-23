use std::fmt;
use std::sync::Arc;

use crate::data::{Column, Incomplete};

/// A function that gives a passive column from the current completed data.
///
/// It receives every column as numbers (a categorical code as its number), with
/// all missing cells filled in, and returns a column with one value per row.
/// Only the values at the column's missing cells are used.
#[derive(Clone)]
pub struct PassiveFn(pub Arc<dyn Fn(&[Vec<f64>]) -> Vec<f64> + Send + Sync>);

impl PassiveFn {
    /// Wraps `f`.
    pub fn new(f: impl Fn(&[Vec<f64>]) -> Vec<f64> + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
}

impl fmt::Debug for PassiveFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PassiveFn")
    }
}

/// Which predicted values are compared in predictive mean matching, as
/// `matchtype` of `mice.impute.pmm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchType {
    /// Both the donors and the recipients are predicted from the fitted
    /// coefficients (`matchtype = 0`).
    Fitted,
    /// The donors are predicted from the fitted coefficients and the
    /// recipients from the drawn ones (`matchtype = 1`, the default of
    /// `mice`).
    Mixed,
    /// Both are predicted from the drawn coefficients (`matchtype = 2`).
    Drawn,
}

/// How one incomplete column is imputed.
#[derive(Clone, Debug)]
pub enum Method {
    /// Linear regression with a draw of the parameters, for a continuous
    /// column: `mice.impute.norm`.
    Norm,
    /// Predictive mean matching of a continuous column: `mice.impute.pmm`.
    Pmm {
        /// The number of donors, `donors` of `mice`, whose default is 5.
        donors: usize,
        /// Which predicted values are compared.
        match_type: MatchType,
    },
    /// Logistic regression with a draw of the parameters, for a binary
    /// column: `mice.impute.logreg`.
    Logreg,
    /// Multinomial logistic regression for a nominal column. With `proper`
    /// the coefficients are drawn from their sampling distribution before
    /// the class is drawn. `mice.impute.polyreg` does not draw them, so it
    /// ignores the uncertainty in the coefficients and is not a proper
    /// imputation in Rubin's sense; `proper: false` reproduces it.
    Polyreg {
        /// Whether to draw the coefficients.
        proper: bool,
    },
    /// A random draw of an observed value, `mice.impute.sample`. Works for
    /// every column type.
    Sample,
    /// A column computed from the others by a function, `mice`'s passive
    /// imputation. The column must be continuous.
    Passive(PassiveFn),
}

impl Method {
    /// Predictive mean matching with `mice`'s defaults: 5 donors and the mixed
    /// match type.
    #[must_use]
    pub const fn pmm() -> Self {
        Self::Pmm {
            donors: 5,
            match_type: MatchType::Mixed,
        }
    }
}

/// The models of an imputation: a method per column, the predictors of each
/// column and the order of the visits.
#[derive(Clone, Debug)]
pub struct ImputeSpec {
    /// The method of each column, `None` for a column without missing cells.
    pub methods: Vec<Option<Method>>,
    /// The columns that predict each column, the rows of `mice`'s predictor
    /// matrix. A column is not its own predictor.
    pub predictors: Vec<Vec<usize>>,
    /// The order in which the incomplete columns are visited in an
    /// iteration, each column once.
    pub visit_order: Vec<usize>,
}

impl ImputeSpec {
    /// `mice`'s defaults for `data`: predictive mean matching of continuous
    /// columns, logistic regression of binary ones and multinomial regression
    /// of nominal ones, every other column as a predictor, and the incomplete
    /// columns visited from left to right. An incomplete ordered column gets
    /// no method, which [`impute`](crate::impute) refuses.
    #[must_use]
    pub fn default_for(data: &Incomplete) -> Self {
        let n = data.n_columns();
        let methods = (0..n)
            .map(|j| {
                if data.missing(j).count() == 0 {
                    return None;
                }
                match &data.columns()[j] {
                    Column::Continuous(_) => Some(Method::pmm()),
                    Column::Binary(_) => Some(Method::Logreg),
                    Column::Nominal { .. } => Some(Method::Polyreg { proper: true }),
                    Column::Ordered { .. } => None,
                }
            })
            .collect();
        Self {
            methods,
            predictors: (0..n)
                .map(|j| (0..n).filter(|&k| k != j).collect())
                .collect(),
            visit_order: (0..n).filter(|&j| data.missing(j).count() > 0).collect(),
        }
    }
}

/// The controls of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImputeOptions {
    /// The number of imputed data sets, one chain each.
    pub m: usize,
    /// The number of iterations of each chain, `maxit` of `mice`, whose
    /// default is 5.
    pub max_iterations: usize,
    /// The seed. A draw depends only on the seed, the chain, the iteration,
    /// the column and the row.
    pub seed: u64,
}
