//! The panel of staggered adoption, the data view of the estimators that
//! follow units over periods.

use epistat_core::{CaseWeights, Col, InputError, Panel, check};
use epistat_linalg::MatRef;

use crate::error::Error;
use crate::methods::att_gt::Design;

/// A validated panel with staggered adoption, one entry per row.
///
/// It holds one row per unit and period: the unit as a `u32` code, the
/// period as an `i64`, the period the unit is first treated (`None` for never
/// treated), the outcome, and optionally prior weights, covariates, cluster
/// codes and, for matched designs, the cohort a never-treated unit was
/// matched to. Its key is [`epistat_core::Panel`]: the rows are sorted by
/// unit and then period, with no pair repeated, so [`Panel::sort_order`]
/// gives the order for unsorted input. Labels become codes with
/// [`epistat_core::Codes::from_labels`], whose sorted order keeps the order
/// of the units that of their labels.
///
/// [`StaggeredPanel::builder`] takes the four required columns, the builder's
/// methods add the optional ones, and [`StaggeredPanelBuilder::build`] checks
/// every column once:
///
/// | column | check |
/// |---|---|
/// | `unit`, `period` | the same length, at least one row, sorted by unit then period, no repeated pair |
/// | every other column | one entry per row |
/// | `outcome` | finite |
/// | `weights` | a [`CaseWeights`]: finite, not negative and not all zero; a zero weight is allowed and its meaning documented by each estimator, which reads the column and not its kind |
/// | `covariates` | finite |
/// | `first_treated`, `clusters`, `comparison_cohort` | constant within a unit |
/// | `first_treated` | not `Some(0)` when every period is positive, R's `did` coding of never treated, which is `None` here |
///
/// [`StaggeredPanel::cross_sections`] builds a repeated cross-section
/// instead: each row is its own observation, in any order, with no unit
/// code, and the unit-level checks do not apply. Only
/// [`att_gt::estimate`](crate::att_gt::estimate) takes one; the estimators
/// that follow units over periods refuse it with [`Error::Unsupported`].
///
/// Each unit is first treated in one period, or never, and stays treated
/// from then on: row `i` is treated when `period[i] >= first_treated[i]`.
/// The estimators take the validated panel and do not repeat these checks.
///
/// ```
/// use epistat_core::Codes;
/// use epistat_did::StaggeredPanel;
///
/// // Two counties by their numbers, observed in 2004 and 2005; the first
/// // is treated from 2005.
/// let county = [1003_i64, 1003, 1001, 1001];
/// let year = [2004, 2005, 2004, 2005];
/// let first_treated = [Some(2005), Some(2005), None, None];
/// let y = [1.0, 2.0, 1.5, 1.6];
/// let state = [7, 7, 3, 3];
///
/// // Codes in sorted label order: 1001 is unit 0 and 1003 unit 1, so the
/// // rows of unit 0 come first.
/// let units = Codes::from_labels(&county)?;
/// let order = epistat_core::Panel::sort_order(units.codes(), &year)?;
/// let unit = order.apply(units.codes())?;
/// let period = order.apply(&year)?;
/// let cohort = order.apply(&first_treated)?;
/// let outcome = order.apply(&y)?;
/// let cluster = order.apply(&state)?;
///
/// let panel = StaggeredPanel::builder(&unit, &period, &cohort, &outcome)
///     .clusters(&cluster)
///     .build()?;
/// assert_eq!(panel.n_units(), 2);
/// assert_eq!(panel.first_treated()[0], None);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct StaggeredPanel<'a> {
    /// `None` for repeated cross-sections, which have no unit codes.
    key: Option<Panel<'a>>,
    period: &'a [i64],
    first_treated: Vec<Option<i64>>,
    outcome: &'a [f64],
    weights: Option<&'a [f64]>,
    covariates: Option<MatRef<'a, f64>>,
    clusters: Option<&'a [u32]>,
    comparison_cohort: Option<Vec<Option<i64>>>,
}

/// The builder of a [`StaggeredPanel`].
#[derive(Clone, Debug)]
#[must_use]
pub struct StaggeredPanelBuilder<'a> {
    unit: Option<&'a [u32]>,
    period: &'a [i64],
    first_treated: Vec<Option<i64>>,
    outcome: &'a [f64],
    weights: Option<&'a [f64]>,
    covariates: Option<MatRef<'a, f64>>,
    clusters: Option<&'a [u32]>,
    comparison_cohort: Option<Vec<Option<i64>>>,
}

/// The rows of a nullable column, a null row `None`.
fn rows<'a>(column: impl Into<Col<'a, i64>>) -> Vec<Option<i64>> {
    column.into().iter().collect()
}

impl<'a> StaggeredPanel<'a> {
    /// The panel of the unit codes `unit`, the periods `period`, the first
    /// treated period of each row's unit (null for never treated) and the
    /// outcome. The first treated periods are a nullable column: a
    /// `&[Option<i64>]`, or a [`Col`] over an Arrow column's values and
    /// validity.
    pub fn builder(
        unit: &'a [u32],
        period: &'a [i64],
        first_treated: impl Into<Col<'a, i64>>,
        outcome: &'a [f64],
    ) -> StaggeredPanelBuilder<'a> {
        StaggeredPanelBuilder {
            unit: Some(unit),
            period,
            first_treated: rows(first_treated),
            outcome,
            weights: None,
            covariates: None,
            clusters: None,
            comparison_cohort: None,
        }
    }

    /// Repeated cross-sections of the periods `period`, the first treated
    /// period of each row's group (`None` for never treated) and the
    /// outcome, `did`'s `panel = FALSE`: each row is its own observation, so
    /// the rows need no unit code and may come in any order.
    ///
    /// ```
    /// use epistat_did::StaggeredPanel;
    /// use epistat_did::att_gt::Design;
    ///
    /// let year = [2005, 2004, 2005, 2004];
    /// let first_treated = [Some(2005), Some(2005), None, None];
    /// let y = [2.0, 1.0, 1.6, 1.5];
    /// let rows = StaggeredPanel::cross_sections(&year, &first_treated[..], &y).build()?;
    /// assert_eq!(rows.design(), Design::RepeatedCrossSection);
    /// assert_eq!(rows.unit(), None);
    /// # Ok::<(), epistat_did::Error>(())
    /// ```
    pub fn cross_sections(
        period: &'a [i64],
        first_treated: impl Into<Col<'a, i64>>,
        outcome: &'a [f64],
    ) -> StaggeredPanelBuilder<'a> {
        StaggeredPanelBuilder {
            unit: None,
            period,
            first_treated: rows(first_treated),
            outcome,
            weights: None,
            covariates: None,
            clusters: None,
            comparison_cohort: None,
        }
    }

    /// Whether the rows are units followed over periods or repeated
    /// cross-sections.
    #[must_use]
    pub const fn design(&self) -> Design {
        match self.key {
            Some(_) => Design::Panel,
            None => Design::RepeatedCrossSection,
        }
    }

    /// The key: the unit code and period of each row; `None` for repeated
    /// cross-sections.
    #[must_use]
    pub const fn key(&self) -> Option<Panel<'a>> {
        self.key
    }

    /// The unit code of each row; `None` for repeated cross-sections.
    #[must_use]
    pub const fn unit(&self) -> Option<&'a [u32]> {
        match self.key {
            Some(key) => Some(key.unit()),
            None => None,
        }
    }

    /// The key of a panel, or [`Error::Unsupported`] for repeated
    /// cross-sections, for the estimators that follow units.
    pub(crate) const fn panel_key(&self) -> Result<Panel<'a>, Error> {
        match self.key {
            Some(key) => Ok(key),
            None => Err(Error::Unsupported {
                what: "repeated cross-sections in an estimator that follows units over \
                       periods; att_gt::estimate takes them",
            }),
        }
    }

    /// The unit codes of a panel, or [`Error::Unsupported`] as
    /// [`panel_key`](Self::panel_key).
    pub(crate) const fn panel_unit(&self) -> Result<&'a [u32], Error> {
        match self.panel_key() {
            Ok(key) => Ok(key.unit()),
            Err(error) => Err(error),
        }
    }

    /// The period of each row.
    #[must_use]
    pub const fn period(&self) -> &'a [i64] {
        self.period
    }

    /// The first treated period of each row's unit, `None` for never
    /// treated.
    #[must_use]
    pub fn first_treated(&self) -> &[Option<i64>] {
        &self.first_treated
    }

    /// The outcome of each row.
    #[must_use]
    pub const fn outcome(&self) -> &'a [f64] {
        self.outcome
    }

    /// The prior weight of each row, `None` when every row weighs 1.
    #[must_use]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.weights
    }

    /// The covariates, one row per row, `None` without covariates.
    #[must_use]
    pub const fn covariates(&self) -> Option<MatRef<'a, f64>> {
        self.covariates
    }

    /// The cluster code of each row, `None` when each unit is its own
    /// cluster.
    #[must_use]
    pub const fn clusters(&self) -> Option<&'a [u32]> {
        self.clusters
    }

    /// For each row of a never-treated unit, the cohort the unit was matched
    /// to as a comparator, `None` for a unit matched to none; `None` for the
    /// whole column when the design is not matched.
    #[must_use]
    pub fn comparison_cohort(&self) -> Option<&[Option<i64>]> {
        self.comparison_cohort.as_deref()
    }

    /// The number of rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.period.len()
    }

    /// Whether the panel has no rows, which the builder refuses, so always
    /// false.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.period.is_empty()
    }

    /// The number of distinct units; on repeated cross-sections the number
    /// of rows, each its own observation.
    #[must_use]
    pub fn n_units(&self) -> usize {
        self.key.map_or(self.period.len(), |key| key.n_units())
    }

    /// The periods as fixed-effect codes `period - min(period)`, `None` when
    /// they span more than 2^32 values.
    pub(crate) fn period_codes(&self) -> Option<Vec<u32>> {
        let period = self.period();
        let t_min = period.iter().copied().min().unwrap_or(0);
        period
            .iter()
            .map(|&t| u32::try_from(t.checked_sub(t_min)?).ok())
            .collect()
    }
}

impl<'a> StaggeredPanelBuilder<'a> {
    /// Case weights, one per row.
    pub const fn weights(mut self, weights: CaseWeights<'a>) -> Self {
        self.weights = Some(weights.as_slice());
        self
    }

    /// Covariates, one row per row.
    pub const fn covariates(mut self, covariates: MatRef<'a, f64>) -> Self {
        self.covariates = Some(covariates);
        self
    }

    /// Cluster codes, one per row and, on a panel, constant within a unit.
    pub const fn clusters(mut self, clusters: &'a [u32]) -> Self {
        self.clusters = Some(clusters);
        self
    }

    /// The cohort each never-treated unit was matched to as a comparator,
    /// one entry per row and constant within a unit.
    pub fn comparison_cohort(mut self, comparison_cohort: impl Into<Col<'a, i64>>) -> Self {
        self.comparison_cohort = Some(rows(comparison_cohort));
        self
    }

    /// The validated panel.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for the first check of [`StaggeredPanel`] that
    /// fails: [`InputError::Length`], [`InputError::Empty`],
    /// [`InputError::Unsorted`] or [`InputError::Duplicate`] for the key,
    /// [`InputError::Length`] for another column of the wrong length,
    /// [`InputError::NotFinite`] for an outcome or covariate that is not
    /// finite, [`InputError::Domain`] for a cohort coded 0 among positive
    /// periods, and [`InputError::NotConstant`] for a column that changes
    /// within a unit.
    pub fn build(self) -> Result<StaggeredPanel<'a>, Error> {
        let key = if let Some(unit) = self.unit {
            Some(Panel::new(unit, self.period)?)
        } else {
            check::non_empty("period", self.period.len())?;
            None
        };
        let n = self.period.len();
        check::same_length("first_treated", n, self.first_treated.len())?;
        check::same_length("outcome", n, self.outcome.len())?;
        check::finite("outcome", self.outcome)?;
        if let Some(weights) = self.weights {
            check::same_length("weights", n, weights.len())?;
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
        check_r_never_treated(self.period, &self.first_treated)?;
        if let Some(clusters) = self.clusters {
            check::same_length("clusters", n, clusters.len())?;
        }
        if let Some(cohort) = &self.comparison_cohort {
            check::same_length("comparison_cohort", n, cohort.len())?;
        }
        if let Some(key) = key {
            constant_by_unit(key, "first_treated", &self.first_treated)?;
            if let Some(clusters) = self.clusters {
                constant_by_unit(key, "clusters", clusters)?;
            }
            if let Some(cohort) = &self.comparison_cohort {
                constant_by_unit(key, "comparison_cohort", cohort)?;
            }
        }
        Ok(StaggeredPanel {
            key,
            period: self.period,
            first_treated: self.first_treated,
            outcome: self.outcome,
            weights: self.weights,
            covariates: self.covariates,
            clusters: self.clusters,
            comparison_cohort: self.comparison_cohort,
        })
    }
}

/// A cohort of 0 among periods that are all positive is R's `did` coding of
/// a never-treated unit, which here would be a cohort treated before the
/// panel and dropped; it is refused with the coding to use.
fn check_r_never_treated(period: &[i64], first_treated: &[Option<i64>]) -> Result<(), InputError> {
    if period.iter().all(|&t| t > 0)
        && let Some(row) = first_treated.iter().position(|&g| g == Some(0))
    {
        return Err(InputError::Domain {
            what: "first_treated",
            value: 0.0,
            domain: "a period, or None for never treated: R's did codes never treated as 0, \
                     which here is a cohort treated before the panel; pass None, or convert \
                     R's column with epistat_did::cohort::from_r",
            row: Some(row),
        });
    }
    Ok(())
}

/// `value` is the same on every row of a unit. The rows of a unit are
/// adjacent in a validated key, so each run is compared with its first row.
fn constant_by_unit<T: PartialEq>(
    key: Panel<'_>,
    what: &'static str,
    value: &[T],
) -> Result<(), InputError> {
    for (unit, rows) in key.units() {
        let first = &value[rows.start];
        if value[rows].iter().any(|v| v != first) {
            return Err(InputError::NotConstant { what, unit });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::Classify;
    use epistat_linalg::Mat;

    const UNIT: [u32; 6] = [0, 0, 0, 2, 2, 2];
    const PERIOD: [i64; 6] = [2000, 2001, 2002, 2000, 2001, 2002];
    const COHORT: [Option<i64>; 6] = [Some(2001), Some(2001), Some(2001), None, None, None];
    const Y: [f64; 6] = [1.0, 2.0, 3.0, 1.5, 1.5, 2.5];

    fn input(result: Result<StaggeredPanel<'_>, Error>) -> InputError {
        match result {
            Err(Error::Input(error)) => error,
            other => panic!("expected an input error, got {other:?}"),
        }
    }

    #[test]
    fn a_valid_panel_keeps_its_columns() {
        let w = [1.0, 1.0, 1.0, 0.0, 2.0, 2.0];
        let x = Mat::from_fn(6, 1, |i, _| f64::from(u32::try_from(i).unwrap()));
        let clusters = [7, 7, 7, 3, 3, 3];
        let matched = [None, None, None, Some(2001), Some(2001), Some(2001)];
        let panel = StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT[..], &Y)
            .weights(epistat_core::CaseWeights::new(&w).expect("weights"))
            .covariates(x.as_ref())
            .clusters(&clusters)
            .comparison_cohort(&matched[..])
            .build()
            .unwrap();
        assert_eq!(panel.len(), 6);
        assert!(!panel.is_empty());
        assert_eq!(panel.n_units(), 2);
        assert_eq!(panel.unit(), Some(&UNIT[..]));
        assert_eq!(panel.period(), &PERIOD);
        assert_eq!(panel.first_treated(), &COHORT);
        assert_eq!(panel.outcome(), &Y);
        assert_eq!(panel.weights(), Some(&w[..]));
        assert_eq!(panel.covariates().map(|x| x.ncols()), Some(1));
        assert_eq!(panel.clusters(), Some(&clusters[..]));
        assert_eq!(panel.comparison_cohort(), Some(&matched[..]));
    }

    #[test]
    fn the_key_must_be_sorted_and_unique() {
        let period = [2001, 2000, 2002, 2000, 2001, 2002];
        let error = input(StaggeredPanel::builder(&UNIT, &period, &COHORT[..], &Y).build());
        assert_eq!(
            error,
            InputError::Unsorted {
                what: "panel",
                row: 1
            }
        );
        let period = [2000, 2000, 2002, 2000, 2001, 2002];
        let error = input(StaggeredPanel::builder(&UNIT, &period, &COHORT[..], &Y).build());
        assert_eq!(
            error,
            InputError::Duplicate {
                what: "panel",
                row: 1
            }
        );
        let error = input(StaggeredPanel::builder(&[], &[], &[] as &[Option<i64>], &[]).build());
        assert_eq!(error.code(), "input.empty");
    }

    #[test]
    fn columns_of_the_wrong_length_are_refused() {
        let error = input(StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT[..5], &Y).build());
        assert_eq!(
            error,
            InputError::Length {
                what: "first_treated",
                expected: 6,
                found: 5
            }
        );
        let error = input(
            StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT[..], &Y)
                .clusters(&[0, 0])
                .build(),
        );
        assert_eq!(error.code(), "input.length");
        let x = Mat::<f64>::zeros(5, 1);
        let error = input(
            StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT[..], &Y)
                .covariates(x.as_ref())
                .build(),
        );
        assert_eq!(error.code(), "input.length");
    }

    #[test]
    fn values_outside_their_domain_are_refused() {
        let mut y = Y;
        y[4] = f64::NAN;
        let error = input(StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT[..], &y).build());
        assert_eq!(
            error,
            InputError::NotFinite {
                what: "outcome",
                row: 4,
                column: None
            }
        );
        let w = [1.0, 1.0, -1.0, 1.0, 1.0, 1.0];
        let error = CaseWeights::new(&w).unwrap_err();
        assert_eq!(error.code(), "input.domain");
        let error = input(StaggeredPanel::builder(&UNIT, &PERIOD, &[Some(0); 6][..], &Y).build());
        assert_eq!(error.code(), "input.domain");
        assert!(
            error
                .to_string()
                .contains("R's did codes never treated as 0"),
            "{error}"
        );
        // A cohort of 0 among periods that start at 0 is a real period.
        let period = [0, 1, 2, 0, 1, 2];
        let cohort = [Some(0), Some(0), Some(0), None, None, None];
        assert!(
            StaggeredPanel::builder(&UNIT, &period, &cohort[..], &Y)
                .build()
                .is_ok()
        );
        let x = Mat::from_fn(
            6,
            2,
            |i, j| if i == 3 && j == 1 { f64::INFINITY } else { 0.0 },
        );
        let error = input(
            StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT[..], &Y)
                .covariates(x.as_ref())
                .build(),
        );
        assert_eq!(
            error,
            InputError::NotFinite {
                what: "covariates",
                row: 3,
                column: None
            }
        );
    }

    #[test]
    fn bad_columns_are_input_errors() {
        use epistat_testkit::{BadInput, assert_rejects_bad_input};
        let reversed: Vec<i64> = PERIOD.iter().rev().copied().collect();
        assert_rejects_bad_input("StaggeredPanel::build", |bad| {
            let y = bad.column(&Y);
            let built = match bad {
                BadInput::Unsorted => {
                    StaggeredPanel::builder(&UNIT, &reversed, &COHORT[..], &Y).build()
                }
                BadInput::Empty => {
                    StaggeredPanel::builder(&[], &[], &[] as &[Option<i64>], &y).build()
                }
                _ => StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT[..], &y).build(),
            };
            Some(built.map(|panel| panel.len()))
        });
    }

    #[test]
    fn unit_level_columns_must_be_constant_within_a_unit() {
        let cohort = [Some(2001), Some(2002), Some(2001), None, None, None];
        let error = input(StaggeredPanel::builder(&UNIT, &PERIOD, &cohort[..], &Y).build());
        assert_eq!(
            error,
            InputError::NotConstant {
                what: "first_treated",
                unit: 0
            }
        );
        let clusters = [1, 1, 1, 4, 5, 4];
        let error = input(
            StaggeredPanel::builder(&UNIT, &PERIOD, &COHORT[..], &Y)
                .clusters(&clusters)
                .build(),
        );
        assert_eq!(
            error,
            InputError::NotConstant {
                what: "clusters",
                unit: 2
            }
        );
    }
}
