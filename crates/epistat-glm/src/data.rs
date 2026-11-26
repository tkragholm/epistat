//! The data views the models take.
//!
//! Each model takes its data as one borrowed view, built by a builder whose
//! `build` checks the columns once and returns a `Result`. A bootstrap or a
//! search over controls builds the view once and fits it many times. The
//! views hold no copy of the data.
//!
//! | view | models | columns |
//! |---|---|---|
//! | [`GlmData`] | [`Glm`](crate::Glm), [`NegBin`](crate::NegBin), [`ElasticNet`](crate::ElasticNet) | design, response, weights, offset, starting coefficients |
//! | [`GeeData`] | [`Gee`](crate::Gee) | design, response, clusters, waves, weights, offset |
//! | [`OrdinalData`] | [`Ordinal`](crate::Ordinal) | design, category-specific design, categories, weights, offset, clusters |
//! | [`MultinomData`] | [`Multinom`](crate::Multinom) | design, class labels, weights |
//!
//! # Checks
//!
//! Every view refuses, with [`Error::Input`](crate::Error::Input):
//!
//! - a design without rows ([`InputError::Empty`]),
//! - a column whose length is not the number of rows of the design, or
//!   starting coefficients whose length is not its number of columns
//!   ([`InputError::Length`]),
//! - a value of the design, the response, the offset or the starting
//!   coefficients that is NaN or infinite ([`InputError::NotFinite`], with the
//!   row; for the design the row of the first such entry, columns read in
//!   order),
//! - a weight that is NaN or infinite ([`InputError::NotFinite`]) or
//!   negative ([`InputError::Domain`]).
//!
//! A weight of 0 is allowed: its row does not enter the fit. What the
//! response must be depends on the model (a count, a proportion, a positive
//! cost), so the model checks it when it fits, with
//! [`Error::Response`](crate::Error::Response).

#[cfg(doc)]
use epistat_core::InputError;
use epistat_core::check;
use epistat_linalg::MatRef;

use crate::Error;
use crate::error::{finite, finite_matrix, length, weights as check_weights};

/// The design with the weights and offset that every view has, checked.
#[derive(Clone, Copy, Debug)]
struct Common<'a> {
    x: MatRef<'a, f64>,
    weights: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
}

impl Common<'_> {
    fn check(self) -> Result<Self, Error> {
        let n = self.x.nrows();
        check::non_empty("x", n)?;
        if let Some(w) = self.weights {
            length("weights", n, w.len())?;
            check_weights("weights", w)?;
        }
        if let Some(o) = self.offset {
            length("offset", n, o.len())?;
            finite("offset", o)?;
        }
        finite_matrix("x", self.x)?;
        Ok(self)
    }
}

/// The data of a generalised linear model: a design, a response and optional
/// prior weights, offset and starting coefficients.
///
/// ```
/// use epistat_glm::{Family, Glm, GlmData, Link};
/// use epistat_linalg::Mat;
///
/// let x = Mat::from_fn(6, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
/// let y = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
/// let w = [1.0, 2.0, 1.0, 1.0, 2.0, 1.0];
/// let data = GlmData::builder(x.as_ref(), &y).weights(&w).build()?;
/// let fit = Glm::new(Family::Binomial, Link::Logit).fit(&data)?;
/// assert!(fit.converged);
/// # Ok::<(), epistat_glm::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GlmData<'a> {
    common: Common<'a>,
    y: &'a [f64],
    start: Option<&'a [f64]>,
}

/// The builder of [`GlmData`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct GlmDataBuilder<'a> {
    data: GlmData<'a>,
}

impl<'a> GlmData<'a> {
    /// The response `y` on the columns of the design `x`, one row each.
    pub const fn builder(x: MatRef<'a, f64>, y: &'a [f64]) -> GlmDataBuilder<'a> {
        GlmDataBuilder {
            data: Self {
                common: Common {
                    x,
                    weights: None,
                    offset: None,
                },
                y,
                start: None,
            },
        }
    }

    /// The design.
    #[must_use]
    pub const fn x(&self) -> MatRef<'a, f64> {
        self.common.x
    }

    /// The response.
    #[must_use]
    pub const fn y(&self) -> &'a [f64] {
        self.y
    }

    /// The prior weights, 1 for every row when `None`.
    #[must_use]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.common.weights
    }

    /// The offset, 0 for every row when `None`.
    #[must_use]
    pub const fn offset(&self) -> Option<&'a [f64]> {
        self.common.offset
    }

    /// The starting coefficients, one per column of the design.
    #[must_use]
    pub const fn start(&self) -> Option<&'a [f64]> {
        self.start
    }

    /// The number of rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.common.x.nrows()
    }

    /// The number of columns of the design.
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.common.x.ncols()
    }

    /// The view with other starting coefficients, checked as `build` checks
    /// them.
    pub(crate) fn with_start(self, start: Option<&'a [f64]>) -> Result<Self, Error> {
        if let Some(s) = start {
            length("start", self.n_columns(), s.len())?;
            finite("start", s)?;
        }
        Ok(Self { start, ..self })
    }

    /// The view of the given parts, each checked as `build` checks it.
    pub(crate) fn from_parts(
        x: MatRef<'a, f64>,
        y: &'a [f64],
        weights: Option<&'a [f64]>,
        offset: Option<&'a [f64]>,
        start: Option<&'a [f64]>,
    ) -> Result<Self, Error> {
        let mut builder = Self::builder(x, y);
        builder.data.common.weights = weights;
        builder.data.common.offset = offset;
        builder.data.start = start;
        builder.build()
    }
}

impl<'a> GlmDataBuilder<'a> {
    /// Prior weights, as R's `weights`: 1 when not given, and a row of weight
    /// 0 does not enter the fit.
    pub const fn weights(mut self, weights: &'a [f64]) -> Self {
        self.data.common.weights = Some(weights);
        self
    }

    /// An offset added to the linear predictor, as R's `offset`.
    pub const fn offset(mut self, offset: &'a [f64]) -> Self {
        self.data.common.offset = Some(offset);
        self
    }

    /// Starting coefficients, one per column of the design, as R's `start`.
    pub const fn start(mut self, start: &'a [f64]) -> Self {
        self.data.start = Some(start);
        self
    }

    /// The view, checked as the module documentation describes.
    ///
    /// # Errors
    ///
    /// [`Error::Input`](crate::Error::Input) for a check that fails.
    pub fn build(self) -> Result<GlmData<'a>, Error> {
        let data = self.data;
        let common = data.common.check()?;
        let n = common.x.nrows();
        length("y", n, data.y.len())?;
        finite("y", data.y)?;
        let data = GlmData { common, ..data };
        data.with_start(data.start)
    }
}

/// The data of a GEE: a design, a response, the cluster of each row and
/// optional waves, prior weights and offset.
///
/// The rows of a cluster need not be adjacent. Without waves the rows of a
/// cluster are ordered as given.
#[derive(Clone, Copy, Debug)]
pub struct GeeData<'a> {
    common: Common<'a>,
    y: &'a [f64],
    clusters: &'a [u32],
    waves: Option<&'a [u32]>,
}

/// The builder of [`GeeData`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct GeeDataBuilder<'a> {
    data: GeeData<'a>,
}

impl<'a> GeeData<'a> {
    /// The response `y` on the columns of `x`, with the cluster of each row.
    pub const fn builder(
        x: MatRef<'a, f64>,
        y: &'a [f64],
        clusters: &'a [u32],
    ) -> GeeDataBuilder<'a> {
        GeeDataBuilder {
            data: Self {
                common: Common {
                    x,
                    weights: None,
                    offset: None,
                },
                y,
                clusters,
                waves: None,
            },
        }
    }

    /// The design.
    #[must_use]
    pub const fn x(&self) -> MatRef<'a, f64> {
        self.common.x
    }

    /// The response.
    #[must_use]
    pub const fn y(&self) -> &'a [f64] {
        self.y
    }

    /// The cluster of each row.
    #[must_use]
    pub const fn clusters(&self) -> &'a [u32] {
        self.clusters
    }

    /// The wave of each row, its time or position within its cluster.
    #[must_use]
    pub const fn waves(&self) -> Option<&'a [u32]> {
        self.waves
    }

    /// The prior weights, 1 for every row when `None`.
    #[must_use]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.common.weights
    }

    /// The offset, 0 for every row when `None`.
    #[must_use]
    pub const fn offset(&self) -> Option<&'a [f64]> {
        self.common.offset
    }
}

impl<'a> GeeDataBuilder<'a> {
    /// The wave of each row, which orders the rows of a cluster and indexes
    /// the unstructured correlation.
    pub const fn waves(mut self, waves: &'a [u32]) -> Self {
        self.data.waves = Some(waves);
        self
    }

    /// Prior weights, which scale the variance of a row by `1 / w`.
    pub const fn weights(mut self, weights: &'a [f64]) -> Self {
        self.data.common.weights = Some(weights);
        self
    }

    /// An offset added to the linear predictor.
    pub const fn offset(mut self, offset: &'a [f64]) -> Self {
        self.data.common.offset = Some(offset);
        self
    }

    /// The view, checked as the module documentation describes.
    ///
    /// # Errors
    ///
    /// [`Error::Input`](crate::Error::Input) for a check that fails.
    pub fn build(self) -> Result<GeeData<'a>, Error> {
        let data = self.data;
        let common = data.common.check()?;
        let n = common.x.nrows();
        length("y", n, data.y.len())?;
        finite("y", data.y)?;
        length("clusters", n, data.clusters.len())?;
        if let Some(w) = data.waves {
            length("waves", n, w.len())?;
        }
        Ok(GeeData { common, ..data })
    }
}

/// The data of an ordinal model: a design, the category of each row and
/// optional category-specific columns, case weights, offset and clusters.
///
/// The design must not hold an intercept: the thresholds are the intercepts.
/// The columns of `nominal`, if given, have one coefficient per threshold, the
/// partial proportional odds model.
#[derive(Clone, Copy, Debug)]
pub struct OrdinalData<'a> {
    common: Common<'a>,
    nominal: Option<MatRef<'a, f64>>,
    y: &'a [u32],
    clusters: Option<&'a [u32]>,
}

/// The builder of [`OrdinalData`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct OrdinalDataBuilder<'a> {
    data: OrdinalData<'a>,
}

impl<'a> OrdinalData<'a> {
    /// The categories `y`, numbered `0..K`, on the columns of `x`.
    pub const fn builder(x: MatRef<'a, f64>, y: &'a [u32]) -> OrdinalDataBuilder<'a> {
        OrdinalDataBuilder {
            data: Self {
                common: Common {
                    x,
                    weights: None,
                    offset: None,
                },
                nominal: None,
                y,
                clusters: None,
            },
        }
    }

    /// The design.
    #[must_use]
    pub const fn x(&self) -> MatRef<'a, f64> {
        self.common.x
    }

    /// The category-specific columns.
    #[must_use]
    pub const fn nominal(&self) -> Option<MatRef<'a, f64>> {
        self.nominal
    }

    /// The category of each row.
    #[must_use]
    pub const fn y(&self) -> &'a [u32] {
        self.y
    }

    /// The case weights, 1 for every row when `None`.
    #[must_use]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.common.weights
    }

    /// The offset, 0 for every row when `None`.
    #[must_use]
    pub const fn offset(&self) -> Option<&'a [f64]> {
        self.common.offset
    }

    /// The cluster of each row, for the robust covariance.
    #[must_use]
    pub const fn clusters(&self) -> Option<&'a [u32]> {
        self.clusters
    }
}

impl<'a> OrdinalDataBuilder<'a> {
    /// Category-specific columns, one row per row of the design.
    pub const fn nominal(mut self, nominal: MatRef<'a, f64>) -> Self {
        self.data.nominal = Some(nominal);
        self
    }

    /// Case weights.
    pub const fn weights(mut self, weights: &'a [f64]) -> Self {
        self.data.common.weights = Some(weights);
        self
    }

    /// An offset added to the linear predictor.
    pub const fn offset(mut self, offset: &'a [f64]) -> Self {
        self.data.common.offset = Some(offset);
        self
    }

    /// The cluster of each row, which sums the scores of the robust
    /// covariance within a cluster.
    pub const fn clusters(mut self, clusters: &'a [u32]) -> Self {
        self.data.clusters = Some(clusters);
        self
    }

    /// The view, checked as the module documentation describes.
    ///
    /// # Errors
    ///
    /// [`Error::Input`](crate::Error::Input) for a check that fails.
    pub fn build(self) -> Result<OrdinalData<'a>, Error> {
        let data = self.data;
        let common = data.common.check()?;
        let n = common.x.nrows();
        length("y", n, data.y.len())?;
        if let Some(nominal) = data.nominal {
            length("nominal", n, nominal.nrows())?;
            finite_matrix("nominal", nominal)?;
        }
        if let Some(c) = data.clusters {
            length("clusters", n, c.len())?;
        }
        Ok(OrdinalData { common, ..data })
    }
}

/// The data of a multinomial logit: a design, the class of each row and
/// optional case weights.
#[derive(Clone, Copy, Debug)]
pub struct MultinomData<'a> {
    common: Common<'a>,
    y: &'a [usize],
    classes: usize,
}

/// The builder of [`MultinomData`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct MultinomDataBuilder<'a> {
    data: MultinomData<'a>,
}

impl<'a> MultinomData<'a> {
    /// The class labels `y`, each in `0..classes`, on the columns of `x`,
    /// which holds the intercept if the model has one.
    pub const fn builder(
        x: MatRef<'a, f64>,
        y: &'a [usize],
        classes: usize,
    ) -> MultinomDataBuilder<'a> {
        MultinomDataBuilder {
            data: Self {
                common: Common {
                    x,
                    weights: None,
                    offset: None,
                },
                y,
                classes,
            },
        }
    }

    /// The design.
    #[must_use]
    pub const fn x(&self) -> MatRef<'a, f64> {
        self.common.x
    }

    /// The class of each row.
    #[must_use]
    pub const fn y(&self) -> &'a [usize] {
        self.y
    }

    /// The number of classes.
    #[must_use]
    pub const fn classes(&self) -> usize {
        self.classes
    }

    /// The case weights, 1 for every row when `None`.
    #[must_use]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.common.weights
    }

    /// The view of the given parts, checked as `build` checks them.
    pub(crate) fn from_parts(
        x: MatRef<'a, f64>,
        y: &'a [usize],
        classes: usize,
        weights: Option<&'a [f64]>,
    ) -> Result<Self, Error> {
        let mut builder = Self::builder(x, y, classes);
        builder.data.common.weights = weights;
        builder.build()
    }
}

impl<'a> MultinomDataBuilder<'a> {
    /// Case weights: a row of weight 0 does not enter.
    pub const fn weights(mut self, weights: &'a [f64]) -> Self {
        self.data.common.weights = Some(weights);
        self
    }

    /// The view, checked as the module documentation describes, and with at
    /// least two classes and every label below their number.
    ///
    /// # Errors
    ///
    /// [`Error::Input`](crate::Error::Input) for a check that fails,
    /// [`Error::Invalid`](crate::Error::Invalid) for fewer than two classes
    /// and [`Error::Response`](crate::Error::Response) for a label that is
    /// not below their number.
    pub fn build(self) -> Result<MultinomData<'a>, Error> {
        let data = self.data;
        let common = data.common.check()?;
        let n = common.x.nrows();
        length("y", n, data.y.len())?;
        if data.classes < 2 {
            return Err(Error::Invalid("a multinomial logit needs two classes"));
        }
        if let Some(row) = data.y.iter().position(|&c| c >= data.classes) {
            return Err(Error::Response {
                row,
                value: data.y[row] as f64,
            });
        }
        Ok(MultinomData { common, ..data })
    }
}
