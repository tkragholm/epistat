//! The data a mixed model takes.

#[cfg(doc)]
use epistat_core::InputError;
use epistat_core::check;
use epistat_linalg::MatRef;

use crate::Error;

/// The data of a mixed model: a design, a response, the group of each row,
/// and optionally the variable of a random slope, prior weights and an
/// offset.
///
/// The rows of a group need not be adjacent. Groups are ordered by their
/// labels, which is the order of R's factor levels for integer labels, and
/// the conditional modes of a fit follow that order.
///
/// `build` refuses, with [`Error::Input`]:
///
/// - a design without rows ([`InputError::Empty`]),
/// - a column whose length is not the number of rows of the design
///   ([`InputError::Length`]),
/// - a value of the design, the response, the slope or the offset that is
///   NaN or infinite ([`InputError::NotFinite`], with the row, and for the
///   design the column),
/// - a weight that is NaN or infinite ([`InputError::NotFinite`]) or
///   negative ([`InputError::Domain`]).
///
/// What else the data must satisfy depends on the model: [`Lmm`](crate::Lmm)
/// needs positive weights, [`Glmm`](crate::Glmm) a response in the support
/// of its family and no slope. Each checks that when it fits.
///
/// ```
/// use epistat_linalg::Mat;
/// use epistat_mixed::{Lmm, MixedData};
///
/// let x = Mat::from_fn(8, 1, |_, _| 1.0);
/// let y = [1.0, 1.4, 0.8, 2.1, 2.5, 1.9, 0.2, 0.5];
/// let groups = [1, 1, 1, 2, 2, 2, 3, 3];
/// let data = MixedData::builder(x.as_ref(), &y, &groups).build()?;
/// let fit = Lmm::default().fit(&data)?;
/// assert_eq!(fit.groups, [1, 2, 3]);
/// # Ok::<(), epistat_mixed::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MixedData<'a> {
    x: MatRef<'a, f64>,
    y: &'a [f64],
    groups: &'a [u32],
    slope: Option<&'a [f64]>,
    weights: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
}

/// The builder of [`MixedData`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct MixedDataBuilder<'a> {
    data: MixedData<'a>,
}

impl<'a> MixedData<'a> {
    /// The response `y` on the columns of the design `x`, with the group
    /// label of each row.
    pub const fn builder(
        x: MatRef<'a, f64>,
        y: &'a [f64],
        groups: &'a [u32],
    ) -> MixedDataBuilder<'a> {
        MixedDataBuilder {
            data: Self {
                x,
                y,
                groups,
                slope: None,
                weights: None,
                offset: None,
            },
        }
    }

    /// The design of the fixed effects.
    #[must_use]
    pub const fn x(&self) -> MatRef<'a, f64> {
        self.x
    }

    /// The response.
    #[must_use]
    pub const fn y(&self) -> &'a [f64] {
        self.y
    }

    /// The group label of each row.
    #[must_use]
    pub const fn groups(&self) -> &'a [u32] {
        self.groups
    }

    /// The variable of the random slope, `None` for a random intercept only.
    #[must_use]
    pub const fn slope(&self) -> Option<&'a [f64]> {
        self.slope
    }

    /// The prior weights, 1 for every row when `None`.
    #[must_use]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.weights
    }

    /// The offset, 0 for every row when `None`.
    #[must_use]
    pub const fn offset(&self) -> Option<&'a [f64]> {
        self.offset
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
}

impl<'a> MixedDataBuilder<'a> {
    /// The variable `t` of a random slope, so that the random effects of a
    /// group are an intercept and a coefficient of `t`, as `(1 + t | g)` in
    /// `lme4`.
    pub const fn slope(mut self, slope: &'a [f64]) -> Self {
        self.data.slope = Some(slope);
        self
    }

    /// Prior weights, as R's `weights`: 1 when not given.
    pub const fn weights(mut self, weights: &'a [f64]) -> Self {
        self.data.weights = Some(weights);
        self
    }

    /// A known part of the linear predictor, as R's `offset`.
    pub const fn offset(mut self, offset: &'a [f64]) -> Self {
        self.data.offset = Some(offset);
        self
    }

    /// The view, checked as [`MixedData`] describes.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for a check that fails.
    pub fn build(self) -> Result<MixedData<'a>, Error> {
        let data = self.data;
        let n = data.x.nrows();
        check::non_empty("x", n)?;
        check::same_length("y", n, data.y.len())?;
        check::same_length("groups", n, data.groups.len())?;
        for (what, column) in [
            ("slope", data.slope),
            ("weights", data.weights),
            ("offset", data.offset),
        ] {
            if let Some(column) = column {
                check::same_length(what, n, column.len())?;
            }
        }
        for j in 0..data.x.ncols() {
            if let Some(row) = (0..n).find(|&i| !data.x[(i, j)].is_finite()) {
                return Err(epistat_core::InputError::NotFinite {
                    what: "x",
                    row,
                    column: Some(j),
                }
                .into());
            }
        }
        check::finite("y", data.y)?;
        if let Some(slope) = data.slope {
            check::finite("slope", slope)?;
        }
        if let Some(offset) = data.offset {
            check::finite("offset", offset)?;
        }
        if let Some(weights) = data.weights {
            check::weights("weights", weights)?;
        }
        Ok(data)
    }
}
