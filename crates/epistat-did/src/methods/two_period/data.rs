//! The data view of the two-period estimators.

use epistat_core::{InputError, check};
use epistat_linalg::MatRef;

use crate::error::Error;

/// How the outcome of a [`TwoPeriodData`] was observed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Design<'a> {
    /// Each row is a unit observed in both periods, with its outcome change
    /// `Y_post - Y_pre`, `DRDID`'s `panel = TRUE`.
    Panel {
        /// The change in outcome of each unit.
        change: &'a [f64],
    },
    /// Each row is one observation in one of the two periods, `DRDID`'s
    /// `panel = FALSE`.
    RepeatedCrossSection {
        /// Whether each row is from the post-treatment period.
        post: &'a [bool],
        /// The outcome of each row.
        outcome: &'a [f64],
    },
}

/// A validated two-period sample, one entry per row.
///
/// On a panel a row is a unit with its outcome change; on a repeated
/// cross-section a row is one observation with its period and outcome.
/// [`TwoPeriodData::panel`] and [`TwoPeriodData::repeated_cross_section`]
/// take the required columns, the builder's methods add the optional ones,
/// and [`TwoPeriodDataBuilder::build`] checks every column once:
///
/// | column | check |
/// |---|---|
/// | every column | one entry per row, at least one row, at most `u32::MAX` rows |
/// | `change`, `outcome` | finite |
/// | `weights` | finite and positive |
/// | `covariates` | finite |
///
/// The covariates are given without an intercept: the estimators add it as
/// the first column of their design, on a panel and on a repeated
/// cross-section alike. Cluster codes, when given, make the standard errors
/// those of `did` with `clustervars`.
///
/// ```
/// use epistat_did::two_period::{self, TwoPeriodData, TwoPeriodOptions};
/// use epistat_linalg::MatRef;
///
/// let treated = [true, true, true, false, false, false, false];
/// let change = [3.1, 2.9, 3.4, 1.0, 1.2, 0.8, 1.1];
/// let age = [41.0, 37.0, 45.0, 40.0, 38.0, 44.0, 39.0];
/// let data = TwoPeriodData::panel(&treated, &change)
///     .covariates(MatRef::from_column_major_slice(&age, 7, 1))
///     .build()?;
/// let fit = two_period::estimate(&data, &TwoPeriodOptions::default())?;
/// assert!(fit.estimate.estimate > 1.5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct TwoPeriodData<'a> {
    treated: &'a [bool],
    design: Design<'a>,
    weights: Option<&'a [f64]>,
    covariates: Option<MatRef<'a, f64>>,
    clusters: Option<&'a [u32]>,
}

/// The builder of a [`TwoPeriodData`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct TwoPeriodDataBuilder<'a> {
    treated: &'a [bool],
    design: Design<'a>,
    weights: Option<&'a [f64]>,
    covariates: Option<MatRef<'a, f64>>,
    clusters: Option<&'a [u32]>,
}

impl<'a> TwoPeriodData<'a> {
    /// A panel: whether each unit is treated and its outcome change between
    /// the two periods.
    pub const fn panel(treated: &'a [bool], change: &'a [f64]) -> TwoPeriodDataBuilder<'a> {
        TwoPeriodDataBuilder::new(treated, Design::Panel { change })
    }

    /// A repeated cross-section: whether each row is treated, whether it is
    /// from the post-treatment period, and its outcome.
    pub const fn repeated_cross_section(
        treated: &'a [bool],
        post: &'a [bool],
        outcome: &'a [f64],
    ) -> TwoPeriodDataBuilder<'a> {
        TwoPeriodDataBuilder::new(treated, Design::RepeatedCrossSection { post, outcome })
    }

    /// Whether each row is treated.
    #[must_use]
    pub const fn treated(&self) -> &'a [bool] {
        self.treated
    }

    /// The design and its outcome columns.
    #[must_use]
    pub const fn design(&self) -> Design<'a> {
        self.design
    }

    /// The prior weight of each row, `None` when every row weighs 1.
    #[must_use]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.weights
    }

    /// The covariates, without the intercept the estimators add; `None`
    /// without covariates.
    #[must_use]
    pub const fn covariates(&self) -> Option<MatRef<'a, f64>> {
        self.covariates
    }

    /// The cluster code of each row, `None` when each row is its own cluster.
    #[must_use]
    pub const fn clusters(&self) -> Option<&'a [u32]> {
        self.clusters
    }

    /// The number of rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.treated.len()
    }

    /// Whether there are no rows, which the builder refuses, so always false.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.treated.is_empty()
    }

    /// Whether the data are a panel.
    #[must_use]
    pub const fn is_panel(&self) -> bool {
        matches!(self.design, Design::Panel { .. })
    }

    /// The estimators' design row-major: the intercept, then the covariates
    /// of each row.
    pub(crate) fn design_flat(&self) -> (Vec<f64>, usize) {
        let n = self.len();
        let columns = 1 + self.covariates.map_or(0, |x| x.ncols());
        let mut flat = Vec::with_capacity(n * columns);
        for i in 0..n {
            flat.push(1.0);
            if let Some(x) = self.covariates {
                flat.extend((0..x.ncols()).map(|j| x[(i, j)]));
            }
        }
        (flat, columns)
    }

    /// The covariates of row `i`, without an intercept.
    pub(crate) fn covariate_row(&self, i: usize) -> Vec<f64> {
        self.covariates
            .map_or_else(Vec::new, |x| (0..x.ncols()).map(|j| x[(i, j)]).collect())
    }

    /// The weight of row `i`.
    pub(crate) fn weight(&self, i: usize) -> f64 {
        self.weights.map_or(1.0, |w| w[i])
    }

    /// The weight of every row.
    pub(crate) fn weight_column(&self) -> Vec<f64> {
        self.weights
            .map_or_else(|| vec![1.0; self.len()], <[f64]>::to_vec)
    }
}

impl<'a> TwoPeriodDataBuilder<'a> {
    const fn new(treated: &'a [bool], design: Design<'a>) -> Self {
        Self {
            treated,
            design,
            weights: None,
            covariates: None,
            clusters: None,
        }
    }

    /// Prior weights, one per row.
    pub const fn weights(mut self, weights: &'a [f64]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Covariates, one row per row, without an intercept: the estimators add
    /// it.
    pub const fn covariates(mut self, covariates: MatRef<'a, f64>) -> Self {
        self.covariates = Some(covariates);
        self
    }

    /// Cluster codes, one per row.
    pub const fn clusters(mut self, clusters: &'a [u32]) -> Self {
        self.clusters = Some(clusters);
        self
    }

    /// The validated data.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for the first check of [`TwoPeriodData`] that fails:
    /// [`InputError::Empty`] without rows, [`InputError::TooLarge`] for more
    /// rows than `u32` numbers, [`InputError::Length`] for a column of the
    /// wrong length, [`InputError::NotFinite`] for an outcome, weight or
    /// covariate that is not finite and [`InputError::Domain`] for a weight
    /// that is not positive.
    pub fn build(self) -> Result<TwoPeriodData<'a>, Error> {
        let n = self.treated.len();
        if n == 0 {
            return Err(InputError::Empty { what: "treated" }.into());
        }
        if u32::try_from(n).is_err() {
            return Err(InputError::TooLarge {
                what: "rows of a two-period sample",
                limit: u64::from(u32::MAX),
                found: u64::try_from(n).unwrap_or(u64::MAX),
            }
            .into());
        }
        match self.design {
            Design::Panel { change } => {
                check::same_length("change", n, change.len())?;
                check::finite("change", change)?;
            }
            Design::RepeatedCrossSection { post, outcome } => {
                check::same_length("post", n, post.len())?;
                check::same_length("outcome", n, outcome.len())?;
                check::finite("outcome", outcome)?;
            }
        }
        if let Some(weights) = self.weights {
            check::same_length("weights", n, weights.len())?;
            check::finite("weights", weights)?;
            if let Some(&value) = weights.iter().find(|&&w| w <= 0.0) {
                return Err(InputError::Domain {
                    what: "weights",
                    value,
                    domain: "(0, inf)",
                    row: None,
                }
                .into());
            }
        }
        if let Some(x) = self.covariates {
            check::same_length("covariates", n, x.nrows())?;
            for j in 0..x.ncols() {
                if let Some(row) = (0..n).find(|&i| !x[(i, j)].is_finite()) {
                    return Err(InputError::NotFinite {
                        what: "covariates",
                        row,
                        column: None,
                    }
                    .into());
                }
            }
        }
        if let Some(clusters) = self.clusters {
            check::same_length("clusters", n, clusters.len())?;
        }
        Ok(TwoPeriodData {
            treated: self.treated,
            design: self.design,
            weights: self.weights,
            covariates: self.covariates,
            clusters: self.clusters,
        })
    }
}

#[cfg(test)]
mod tests {
    use epistat_linalg::MatRef;
    use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

    use super::TwoPeriodData;

    const TREATED: [bool; 6] = [true, true, true, false, false, false];
    const CHANGE: [f64; 6] = [3.1, 2.9, 3.4, 1.0, 1.2, 0.8];
    const POST: [bool; 6] = [false, true, false, true, false, true];
    const AGE: [f64; 6] = [41.0, 37.0, 45.0, 40.0, 38.0, 44.0];

    /// An empty outcome column is shorter than the treatment column, and the
    /// rows of a two-period sample have no order.
    fn code(bad: BadInput) -> &'static str {
        match bad {
            BadInput::Empty => "input.length",
            other => other.code(),
        }
    }

    #[test]
    fn bad_columns_are_input_errors() {
        let len = |built: Result<TwoPeriodData<'_>, crate::Error>| built.map(|data| data.len());
        assert_rejects_bad_input_with_codes("TwoPeriodData::panel", code, |bad| {
            (bad != BadInput::Unsorted).then(|| {
                let change = bad.column(&CHANGE);
                len(TwoPeriodData::panel(&TREATED, &change).build())
            })
        });
        assert_rejects_bad_input_with_codes("TwoPeriodData::repeated_cross_section", code, |bad| {
            (bad != BadInput::Unsorted).then(|| {
                let outcome = bad.column(&CHANGE);
                len(TwoPeriodData::repeated_cross_section(&TREATED, &POST, &outcome).build())
            })
        });
        assert_rejects_bad_input_with_codes("TwoPeriodData weights", code, |bad| {
            (bad != BadInput::Unsorted).then(|| {
                let weights = bad.column(&[1.0; 6]);
                len(TwoPeriodData::panel(&TREATED, &CHANGE)
                    .weights(&weights)
                    .build())
            })
        });
        assert_rejects_bad_input_with_codes("TwoPeriodData covariates", code, |bad| {
            (bad != BadInput::Unsorted).then(|| {
                let age = bad.column(&AGE);
                let covariates = MatRef::from_column_major_slice(&age, age.len(), 1);
                len(TwoPeriodData::panel(&TREATED, &CHANGE)
                    .covariates(covariates)
                    .build())
            })
        });
    }
}
