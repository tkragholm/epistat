//! The data views the models take.
//!
//! Each model takes its data as one borrowed view, built by a builder whose
//! `build` checks the columns once and returns a `Result`. A bootstrap builds
//! the view of the full sample once and refits it under many weight columns
//! with [`CostData::with_weights`], which checks only the new weights. The
//! views hold no copy of the data.
//!
//! | view | models | columns |
//! |---|---|---|
//! | [`CostData`] | [`two_part`](crate::two_part), [`tweedie`](crate::tweedie), [`lognormal`](crate::lognormal), [`compare`](crate::compare) | design, outcome, weights, clusters |
//! | [`LongitudinalData`] | [`mtp`](crate::mtp) | outcome, two designs, subject, time, family |
//!
//! # Checks
//!
//! Both views refuse, with [`Error::Input`]:
//!
//! - no rows, or a design without columns ([`InputError::Empty`]),
//! - a column whose length is not the number of rows of the design
//!   ([`InputError::Length`]),
//! - a value of a design, the outcome or the time that is NaN or infinite
//!   ([`InputError::NotFinite`], with the row; for a design the row of the
//!   first such entry, columns read in order),
//! - a negative outcome ([`InputError::Domain`]).
//!
//! The weights of [`CostData`] are a [`CaseWeights`] column, which refuses a
//! weight that is negative or not finite when it is built; the view refuses
//! weights whose length is not the number of rows ([`InputError::Length`]).
//! A weight of 0 is allowed and takes its row out of the fit: it adds nothing
//! to the fit's sums and is not counted among the rows that enter it, so a
//! resample-count bootstrap can pass the count of each row as its weight.
//!
//! Clusters and subjects are `u32` codes, as
//! [`Codes::from_labels`](epistat_core::Codes::from_labels) gives them. The
//! cluster-robust standard errors number the clusters in the order they first
//! appear, and the MTP sampler takes the subjects in the order of their codes,
//! so codes in sorted label order give the order of the labels.

#[cfg(doc)]
use epistat_core::InputError;
use epistat_core::{CaseWeights, InputError as Refusal, check};
use epistat_linalg::{Mat, MatRef};

use crate::Error;

/// The row of the first entry of `x` that is not finite, columns read in order.
fn first_not_finite(x: MatRef<'_, f64>) -> Option<usize> {
    let mut first: Option<usize> = None;
    for j in 0..x.ncols() {
        let limit = first.unwrap_or(x.nrows());
        let column = x.col(j);
        if let Some(row) = (0..limit).find(|&i| !column[i].is_finite()) {
            first = Some(row);
        }
    }
    first
}

fn finite_matrix(what: &'static str, x: MatRef<'_, f64>) -> Result<(), Error> {
    first_not_finite(x).map_or(Ok(()), |row| {
        Err(Refusal::NotFinite {
            what,
            row,
            column: None,
        }
        .into())
    })
}

fn non_negative(what: &'static str, values: &[f64]) -> Result<(), Error> {
    check::finite(what, values)?;
    values.iter().find(|&&v| v < 0.0).map_or(Ok(()), |&value| {
        Err(Refusal::Domain {
            what,
            value,
            domain: "[0, inf)",
            row: None,
        }
        .into())
    })
}

fn columns(what: &'static str, x: MatRef<'_, f64>) -> Result<(), Error> {
    if x.ncols() == 0 {
        return Err(Refusal::Empty { what }.into());
    }
    Ok(())
}

pub(crate) fn check_weights(n: usize, weights: &CaseWeights<'_>) -> Result<(), Error> {
    check::same_length("weights", n, weights.len())?;
    Ok(())
}

/// `x` has `p` columns and `weights`, when given, one value per row of `x`:
/// the check of a design passed to a fitted model.
pub(crate) fn check_columns(
    x: MatRef<'_, f64>,
    p: usize,
    weights: Option<&[f64]>,
) -> Result<(), Error> {
    check::same_length("columns of x", p, x.ncols())?;
    if let Some(weights) = weights {
        check::same_length("weights", x.nrows(), weights.len())?;
    }
    Ok(())
}

/// A column as an `n x 1` matrix view, without a copy.
pub(crate) fn column(values: &[f64]) -> MatRef<'_, f64> {
    MatRef::from_column_major_slice(values, values.len(), 1)
}

/// The data of a cost model: a design, a non-negative outcome, and optional
/// case weights and cluster codes.
///
/// The design holds the intercept column if the model is to have one. The
/// weights are R's prior `weights` unless their [`CaseWeights`] kind says
/// otherwise; the clusters, one code per row, make the sandwich the default
/// covariance and give its clustered form.
///
/// ```
/// use epistat_linalg::Mat;
/// use epistat_semicontinuous::{CostData, two_part};
///
/// let n = 40;
/// let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { i as f64 / 10.0 });
/// let y: Vec<f64> = (0..n).map(|i| if i % 4 == 0 { 0.0 } else { 1.0 + 0.1 * i as f64 }).collect();
/// let data = CostData::builder(x.as_ref(), &y).build()?;
/// let model = two_part::fit(&data, &two_part::TwoPartOptions::default(), None)?;
/// assert_eq!(model.logit.coefficients.len(), 2);
/// # Ok::<(), epistat_semicontinuous::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CostData<'a> {
    x: MatRef<'a, f64>,
    y: &'a [f64],
    weights: Option<CaseWeights<'a>>,
    clusters: Option<&'a [u32]>,
}

/// The builder of [`CostData`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct CostDataBuilder<'a> {
    data: CostData<'a>,
}

impl<'a> CostData<'a> {
    /// The outcome `y` on the columns of the design `x`, one row each.
    pub const fn builder(x: MatRef<'a, f64>, y: &'a [f64]) -> CostDataBuilder<'a> {
        CostDataBuilder {
            data: Self {
                x,
                y,
                weights: None,
                clusters: None,
            },
        }
    }

    /// The design.
    #[must_use]
    pub const fn x(&self) -> MatRef<'a, f64> {
        self.x
    }

    /// The outcome.
    #[must_use]
    pub const fn y(&self) -> &'a [f64] {
        self.y
    }

    /// The case weights, 1 for every row when `None`.
    #[must_use]
    pub const fn weights(&self) -> Option<CaseWeights<'a>> {
        self.weights
    }

    /// The values of the case weights.
    pub(crate) fn weight_values(&self) -> Option<&'a [f64]> {
        self.weights.map(|w| w.as_slice())
    }

    /// The cluster code of each row.
    #[must_use]
    pub const fn clusters(&self) -> Option<&'a [u32]> {
        self.clusters
    }

    /// The number of rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.x.nrows()
    }

    /// The number of columns of the design.
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.x.ncols()
    }

    /// The view with other case weights, checked as `build` checks them. The
    /// design and the outcome are not scanned again, so a bootstrap refits
    /// one sample under many weight columns at the cost of one scan of each.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for weights of the wrong length.
    pub fn with_weights<'b>(&self, weights: CaseWeights<'b>) -> Result<CostData<'b>, Error>
    where
        'a: 'b,
    {
        check_weights(self.n_rows(), &weights)?;
        Ok(CostData {
            x: self.x,
            y: self.y,
            weights: Some(weights),
            clusters: self.clusters,
        })
    }

    /// The view without weights.
    #[must_use]
    pub const fn without_weights(&self) -> Self {
        Self {
            weights: None,
            ..*self
        }
    }

    /// The view without clusters.
    #[must_use]
    pub const fn without_clusters(&self) -> Self {
        Self {
            clusters: None,
            ..*self
        }
    }

    /// A view of columns already checked, such as rows of a checked view.
    pub(crate) const fn from_checked(
        x: MatRef<'a, f64>,
        y: &'a [f64],
        weights: Option<CaseWeights<'a>>,
        clusters: Option<&'a [u32]>,
    ) -> Self {
        Self {
            x,
            y,
            weights,
            clusters,
        }
    }

    /// The outcome as an `n x 1` view.
    pub(crate) fn y_column(&self) -> MatRef<'a, f64> {
        column(self.y)
    }

    /// The weights as an `n x 1` view.
    pub(crate) fn weights_column(&self) -> Option<MatRef<'a, f64>> {
        self.weight_values().map(column)
    }
}

impl<'a> CostDataBuilder<'a> {
    /// Case weights, R's `weights` for prior weights: 1 for every row when
    /// not given.
    pub const fn weights(mut self, weights: CaseWeights<'a>) -> Self {
        self.data.weights = Some(weights);
        self
    }

    /// The cluster code of each row, for cluster-robust standard errors.
    pub const fn clusters(mut self, clusters: &'a [u32]) -> Self {
        self.data.clusters = Some(clusters);
        self
    }

    /// The checked view.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] as the [module](self) lists.
    pub fn build(self) -> Result<CostData<'a>, Error> {
        let data = self.data;
        let n = data.x.nrows();
        check::non_empty("x", n)?;
        columns("columns of x", data.x)?;
        check::same_length("y", n, data.y.len())?;
        finite_matrix("x", data.x)?;
        non_negative("y", data.y)?;
        if let Some(weights) = &data.weights {
            check_weights(n, weights)?;
        }
        if let Some(clusters) = data.clusters {
            check::same_length("clusters", n, clusters.len())?;
        }
        Ok(data)
    }
}

/// Person-period data for the marginalised two-part model: an outcome, the
/// designs of the binary and the mean part, the subject and the time of each
/// row, and optionally its family.
///
/// The time enters the random time slopes. Rows of one subject need not be
/// adjacent or sorted.
#[derive(Clone, Copy, Debug)]
pub struct LongitudinalData<'a> {
    y: &'a [f64],
    x_binary: MatRef<'a, f64>,
    x_mean: MatRef<'a, f64>,
    subject: &'a [u32],
    time: &'a [f64],
    family: Option<&'a [u32]>,
}

/// The builder of [`LongitudinalData`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct LongitudinalDataBuilder<'a> {
    data: LongitudinalData<'a>,
}

impl<'a> LongitudinalData<'a> {
    /// The outcome `y` with the binary-part design `x_binary`, the mean
    /// design `x_mean`, and the subject code and time of each row.
    pub const fn builder(
        y: &'a [f64],
        x_binary: MatRef<'a, f64>,
        x_mean: MatRef<'a, f64>,
        subject: &'a [u32],
        time: &'a [f64],
    ) -> LongitudinalDataBuilder<'a> {
        LongitudinalDataBuilder {
            data: Self {
                y,
                x_binary,
                x_mean,
                subject,
                time,
                family: None,
            },
        }
    }

    /// The outcome.
    #[must_use]
    pub const fn y(&self) -> &'a [f64] {
        self.y
    }

    /// The design of the binary part, `Pr(y > 0)`.
    #[must_use]
    pub const fn x_binary(&self) -> MatRef<'a, f64> {
        self.x_binary
    }

    /// The design of the marginal mean.
    #[must_use]
    pub const fn x_mean(&self) -> MatRef<'a, f64> {
        self.x_mean
    }

    /// The subject code of each row.
    #[must_use]
    pub const fn subject(&self) -> &'a [u32] {
        self.subject
    }

    /// The time of each row.
    #[must_use]
    pub const fn time(&self) -> &'a [f64] {
        self.time
    }

    /// The family code of each row.
    #[must_use]
    pub const fn family(&self) -> Option<&'a [u32]> {
        self.family
    }

    /// The number of rows.
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.y.len()
    }

    /// The owned copy the sampler reads.
    pub(crate) fn to_owned_input(self) -> crate::mtp::input::LongitudinalInput {
        crate::mtp::input::LongitudinalInput {
            outcome: Mat::from_fn(self.y.len(), 1, |i, _| self.y[i]),
            x_binary: self.x_binary.to_owned(),
            x_mean: self.x_mean.to_owned(),
            subject_ids: self.subject.iter().map(|&s| u64::from(s)).collect(),
            time: self.time.to_vec(),
            family_ids: self
                .family
                .map(|family| family.iter().map(|&f| u64::from(f)).collect()),
        }
    }
}

impl<'a> LongitudinalDataBuilder<'a> {
    /// The family code of each row, for a family-level random intercept.
    pub const fn families(mut self, family: &'a [u32]) -> Self {
        self.data.family = Some(family);
        self
    }

    /// The checked view.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] as the [module](self) lists.
    pub fn build(self) -> Result<LongitudinalData<'a>, Error> {
        let data = self.data;
        let n = data.y.len();
        check::non_empty("y", n)?;
        columns("columns of x_binary", data.x_binary)?;
        columns("columns of x_mean", data.x_mean)?;
        check::same_length("x_binary", n, data.x_binary.nrows())?;
        check::same_length("x_mean", n, data.x_mean.nrows())?;
        check::same_length("subject", n, data.subject.len())?;
        check::same_length("time", n, data.time.len())?;
        if let Some(family) = data.family {
            check::same_length("family", n, family.len())?;
        }
        finite_matrix("x_binary", data.x_binary)?;
        finite_matrix("x_mean", data.x_mean)?;
        check::finite("time", data.time)?;
        non_negative("y", data.y)?;
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(result: Result<CostData<'_>, Error>) -> Refusal {
        match result {
            Err(Error::Input(error)) => error,
            other => panic!("expected an input error, got {other:?}"),
        }
    }

    #[test]
    fn cost_data_refuses_each_kind_of_bad_column() {
        let x = Mat::from_fn(3, 1, |_, _| 1.0);
        let y = [0.0, 1.0, 2.0];
        assert!(CostData::builder(x.as_ref(), &y).build().is_ok());

        let empty = Mat::<f64>::zeros(3, 0);
        assert_eq!(
            refused(CostData::builder(empty.as_ref(), &y).build()),
            Refusal::Empty {
                what: "columns of x"
            }
        );
        assert_eq!(
            refused(CostData::builder(x.as_ref(), &y[..2]).build()),
            Refusal::Length {
                what: "y",
                expected: 3,
                found: 2
            }
        );
        let nan = Mat::from_fn(3, 1, |i, _| if i == 1 { f64::NAN } else { 1.0 });
        assert_eq!(
            refused(CostData::builder(nan.as_ref(), &y).build()),
            Refusal::NotFinite {
                what: "x",
                row: 1,
                column: None
            }
        );
        assert_eq!(
            refused(CostData::builder(x.as_ref(), &[0.0, -1.0, 2.0]).build()),
            Refusal::Domain {
                what: "y",
                value: -1.0,
                domain: "[0, inf)",
                row: None
            }
        );
        assert_eq!(
            refused(CostData::builder(x.as_ref(), &[0.0, f64::INFINITY, 2.0]).build()),
            Refusal::NotFinite {
                what: "y",
                row: 1,
                column: None
            }
        );
        let zero = [1.0, 0.0, 1.0];
        let zero = CaseWeights::new(&zero).expect("a zero weight is allowed");
        assert!(
            CostData::builder(x.as_ref(), &y)
                .weights(zero)
                .build()
                .is_ok()
        );
        let short = [1.0, 1.0];
        assert_eq!(
            refused(
                CostData::builder(x.as_ref(), &y)
                    .weights(CaseWeights::new(&short).expect("weights"))
                    .build()
            ),
            Refusal::Length {
                what: "weights",
                expected: 3,
                found: 2
            }
        );
        assert_eq!(
            refused(CostData::builder(x.as_ref(), &y).clusters(&[1, 2]).build()),
            Refusal::Length {
                what: "clusters",
                expected: 3,
                found: 2
            }
        );
    }

    #[test]
    fn with_weights_checks_only_the_weights() {
        let x = Mat::from_fn(3, 1, |_, _| 1.0);
        let y = [0.0, 1.0, 2.0];
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        let w = [1.0, 2.0, 0.5];
        let weighted = data
            .with_weights(CaseWeights::new(&w).expect("weights"))
            .expect("weights");
        assert_eq!(weighted.weight_values(), Some(&w[..]));
        let short = [1.0, 2.0];
        assert!(
            data.with_weights(CaseWeights::new(&short).expect("weights"))
                .is_err()
        );
        assert!(weighted.without_weights().weights().is_none());
    }

    #[test]
    fn longitudinal_data_refuses_a_subject_column_of_the_wrong_length() {
        let x = Mat::from_fn(2, 1, |_, _| 1.0);
        let y = [1.0, 0.0];
        let ok = LongitudinalData::builder(&y, x.as_ref(), x.as_ref(), &[1, 1], &[0.0, 1.0]);
        assert!(ok.build().is_ok());
        let bad = LongitudinalData::builder(&y, x.as_ref(), x.as_ref(), &[1], &[0.0, 1.0]);
        assert!(matches!(
            bad.build(),
            Err(Error::Input(Refusal::Length {
                what: "subject",
                ..
            }))
        ));
    }
}
