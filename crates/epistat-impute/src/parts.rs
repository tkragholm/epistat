//! The elementary steps of the imputation methods, each a function of its
//! inputs and of the random draws it needs.
//!
//! [`impute`](crate::impute) calls these with draws from its streams. They are
//! public so that a test can inject the draws and compare an intermediate
//! result with `mice`'s, which is how the methods are checked, and hidden from
//! the documentation: they are not part of the API and may change in any
//! release.
//!
//! # Linear model
//!
//! [`linear_fit`] is least squares of `y` on the columns of `x` with `n`
//! rows and `p` columns, giving `b = (X'X)^-1 X'y`, `V = (X'X)^-1`, the
//! residual sum of squares `RSS` and `df = n - p`. A column that is a linear
//! combination of the columns before it is aliased: its coefficient is 0 and
//! it is left out of `V`, as `mice` sets it to 0 (`rank.adjust`).
//! [`draw_linear`] is `mice:::.norm.draw`:
//!
//! ```text
//! sigma* = sqrt(RSS / chi2),         chi2 ~ chi-square(df)
//! beta*  = b + chol(V) z sigma*,     z ~ N(0, I)
//! ```
//!
//! with `chol(V)` the lower Cholesky factor. `mice` adds a ridge to `V` only
//! when the inverse fails, which does not occur here because aliased columns
//! are removed first.
//!
//! # Predictive mean matching
//!
//! [`SortedDonors`] holds the donors' predicted values in order. For a
//! recipient with predicted value `t` and `k` donors, the eligible donors are
//! those whose distance `|yhat - t|` does not exceed the `k`-th smallest
//! distance, so donors tied at the cut are all eligible, as in
//! `mice:::.pmm.match`. The donor is the one at position `floor(u c)` among
//! the `c` eligible donors in sorted order, for a uniform `u`. The search is a
//! binary search and an expansion of a window, `O(log n + k)` per recipient
//! after an `O(n log n)` sort. `mice`'s default search (`matchindex`) takes
//! exactly `k` donors and breaks a tie at the cut by the position in its own
//! sort, so the two differ only where predicted values are tied.
//!
//! # Logistic and multinomial models
//!
//! [`augmentation`] is `mice:::augment`: `2 p k` rows are added to the
//! `p` predictor columns, for `k` observed outcome values. For each column
//! the rows are placed at the mean plus or minus half a standard deviation,
//! clamped to the observed range, one pair per outcome value, with weight
//! `(p + 1) / (2 p k)` each. They keep the likelihood finite when the observed
//! data separate the outcome. [`logistic_fit`] and [`multinomial_fit`] fit by
//! the Newton and IRLS routines of `epistat-glm`, [`draw_coefficients`] draws
//! `beta* = b + chol(V) z` for the logistic and multinomial models, and
//! [`class_probabilities`] and [`class_from_uniform`] turn the coefficients
//! into a class: the first class whose cumulative probability reaches the
//! uniform `u`.
//!
//! # References
//!
//! 1. van Buuren S, Groothuis-Oudshoorn K. mice: multivariate imputation by
//!    chained equations in R. J Stat Softw. 2011;45(3):1-67.
//! 2. van Buuren S. Flexible Imputation of Missing Data. 2nd ed. Boca Raton:
//!    Chapman and Hall/CRC; 2018. Sections 3.2 and 3.4.
//! 3. Rubin DB. Multiple Imputation for Nonresponse in Surveys. New York:
//!    Wiley; 1987.
//! 4. White IR, Daniel R, Royston P. Avoiding bias due to perfect prediction
//!    in multiple imputation of incomplete categorical variables. Comput Stat
//!    Data Anal. 2010;54(10):2267-75.

use std::ops::Range;

use epistat_glm::{Family, Glm, GlmData, Link, Multinom, MultinomData};
use epistat_linalg::cholesky::{Pivot, cholesky_lower};
use epistat_linalg::design::{independent_columns, select_columns};
use epistat_linalg::{Mat, MatRef};

use crate::error::{Error, Result, length};

/// Least squares of an outcome on a design, for the parameter draw.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearFit {
    /// The coefficients, 0 for an aliased column.
    pub coefficients: Vec<f64>,
    /// Whether each column is in the model, false for an aliased column.
    pub active: Vec<bool>,
    /// `(X'X)^-1` of the active columns, row-major, side `active.iter().filter(..).count()`.
    pub covariance: Vec<f64>,
    /// The residual sum of squares.
    pub rss: f64,
    /// The degrees of freedom, rows less columns of the design.
    pub df: f64,
}

impl LinearFit {
    /// The number of columns in the model.
    #[must_use]
    pub fn n_active(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }
}

/// Fits `y` on the columns of `x`, which holds the intercept if the model has
/// one. `column` names the imputed column in errors.
///
/// # Errors
///
/// [`Error::TooFewObserved`] when `x` has no more rows than columns, and
/// the regression's own errors.
pub fn linear_fit(x: MatRef<'_, f64>, y: &[f64], column: usize) -> Result<LinearFit> {
    let (n, p) = (x.nrows(), x.ncols());
    if n <= p {
        return Err(Error::TooFewObserved {
            column,
            observed: n,
            parameters: p,
        });
    }
    let fit = Glm::new(Family::Gaussian, Link::Identity).fit(&GlmData::builder(x, y).build()?)?;
    if !fit.converged {
        return Err(Error::NotConverged { column });
    }
    let active: Vec<bool> = fit.coefficients.iter().map(Option::is_some).collect();
    let covariance = unscaled_covariance(fit.covariance.as_slice(), &active, fit.dispersion);
    Ok(LinearFit {
        coefficients: fit.coefficients.iter().map(|c| c.unwrap_or(0.0)).collect(),
        active,
        covariance,
        rss: fit.deviance,
        df: (n - p) as f64,
    })
}

/// The covariance of the active coefficients divided by the dispersion, which
/// is `(X'X)^-1` for the Gaussian model. A zero dispersion gives zeros.
fn unscaled_covariance(full: &[Option<f64>], active: &[bool], dispersion: Option<f64>) -> Vec<f64> {
    let p = active.len();
    let scale = dispersion.unwrap_or(1.0);
    let keep: Vec<usize> = (0..p).filter(|&j| active[j]).collect();
    let mut out = Vec::with_capacity(keep.len() * keep.len());
    for &i in &keep {
        for &j in &keep {
            let v = full[i * p + j].unwrap_or(0.0);
            out.push(if scale == 0.0 { 0.0 } else { v / scale });
        }
    }
    out
}

/// A draw of the regression parameters: `mice:::.norm.draw`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterDraw {
    /// `beta*`, 0 for an aliased column.
    pub beta: Vec<f64>,
    /// `sigma*`.
    pub sigma: f64,
}

/// `sigma* = sqrt(RSS / chi_square)` and `beta* = b + chol(V) z sigma*`, from a
/// chi-square draw with `fit.df` degrees of freedom and the standard normals
/// `z`, one per active column.
///
/// # Errors
///
/// [`Error::Input`] when `z` is not as long as the active columns,
/// [`Error::Covariance`] when `V` has no Cholesky factor.
pub fn draw_linear(
    fit: &LinearFit,
    chi_square: f64,
    z: &[f64],
    column: usize,
) -> Result<ParameterDraw> {
    let sigma = (fit.rss / chi_square).sqrt();
    let beta = if sigma == 0.0 {
        fit.coefficients.clone()
    } else {
        draw_coefficients(
            &fit.coefficients,
            &fit.active,
            &fit.covariance,
            z,
            sigma,
            column,
        )?
    };
    Ok(ParameterDraw { beta, sigma })
}

/// `b + scale chol(V) z` over the active coefficients, with 0 for the others.
///
/// `covariance` is `V` of the active coefficients, row-major, and `z` has one
/// standard normal per active coefficient.
///
/// # Errors
///
/// [`Error::Input`] for a `z` of the wrong length and
/// [`Error::Covariance`] when `V` has no Cholesky factor.
pub fn draw_coefficients(
    coefficients: &[f64],
    active: &[bool],
    covariance: &[f64],
    z: &[f64],
    scale: f64,
    column: usize,
) -> Result<Vec<f64>> {
    let k = active.iter().filter(|&&a| a).count();
    length("z", k, z.len())?;
    let lower = cholesky_lower(covariance, k, Pivot::Strict).ok_or(Error::Covariance { column })?;
    let mut out = vec![0.0; coefficients.len()];
    let mut a = 0;
    for (j, slot) in out.iter_mut().enumerate() {
        if active[j] {
            let mut shift = 0.0;
            for (c, &zc) in z.iter().enumerate().take(a + 1) {
                shift += lower[a * k + c] * zc;
            }
            *slot = coefficients[j] + shift * scale;
            a += 1;
        }
    }
    Ok(out)
}

/// The donors of predictive mean matching, ordered by predicted value.
#[derive(Clone, Debug, PartialEq)]
pub struct SortedDonors {
    values: Vec<f64>,
    order: Vec<usize>,
}

impl SortedDonors {
    /// Sorts the donors' predicted values, ties by position.
    ///
    /// # Errors
    ///
    /// [`Error::NotFinite`] for a value that is not finite.
    pub fn new(predicted: &[f64], column: usize) -> Result<Self> {
        if predicted.iter().any(|v| !v.is_finite()) {
            return Err(Error::NotFinite { column });
        }
        let mut order: Vec<usize> = (0..predicted.len()).collect();
        order.sort_unstable_by(|&a, &b| predicted[a].total_cmp(&predicted[b]).then(a.cmp(&b)));
        let values = order.iter().map(|&i| predicted[i]).collect();
        Ok(Self { values, order })
    }

    /// The number of donors.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether there is no donor.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The positions, in sorted order, of the donors whose distance to
    /// `target` does not exceed that of the `donors`-th nearest.
    #[must_use]
    pub fn eligible(&self, target: f64, donors: usize) -> Range<usize> {
        let len = self.values.len();
        if len == 0 {
            return 0..0;
        }
        let k = donors.clamp(1, len);
        let distance = |v: f64| (v - target).abs();
        let split = self.values.partition_point(|&v| v < target);
        let (mut left, mut right) = (split, split);
        let mut cut = 0.0;
        for _ in 0..k {
            let take_left = left > 0
                && (right >= len
                    || distance(self.values[left - 1]) <= distance(self.values[right]));
            if take_left {
                left -= 1;
                cut = distance(self.values[left]);
            } else {
                cut = distance(self.values[right]);
                right += 1;
            }
        }
        while left > 0 && distance(self.values[left - 1]) <= cut {
            left -= 1;
        }
        while right < len && distance(self.values[right]) <= cut {
            right += 1;
        }
        left..right
    }

    /// The index, in the list given to [`new`](Self::new), of the donor at
    /// position `position` in sorted order.
    #[must_use]
    pub fn donor_at(&self, position: usize) -> usize {
        self.order[position]
    }

    /// The index, in the list given to [`new`](Self::new), of the donor chosen
    /// for `target` by the uniform `u` in `[0, 1)`.
    ///
    /// # Panics
    ///
    /// If there is no donor.
    #[must_use]
    pub fn pick(&self, target: f64, donors: usize, u: f64) -> usize {
        let range = self.eligible(target, donors);
        let count = range.len();
        let offset = position_from_uniform(u, count);
        self.order[range.start + offset]
    }
}

/// The position `floor(u * count)` among `count` items, for `u` in `[0, 1)`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "u is not negative and the result is clamped below count"
)]
fn position_from_uniform(u: f64, count: usize) -> usize {
    ((u * count as f64) as usize).min(count - 1)
}

/// The rows `mice:::augment` adds to the predictors of a logistic or
/// multinomial model.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Augmentation {
    /// The predictor values of the added rows, `rows` by `p`, row-major.
    pub x: Vec<f64>,
    /// The outcome of each added row.
    pub y: Vec<f64>,
    /// The weight of each added row, `(p + 1) / rows`.
    pub weight: f64,
    /// The number of added rows, `2 p k`.
    pub rows: usize,
}

/// The augmentation rows for the predictor columns `x` (one slice per column,
/// no intercept, all rows of the data) and the sorted distinct observed
/// outcome values `outcomes`. `None` when there are no predictors.
///
/// Where `mice` skips the augmentation when exactly one row is missing, this
/// does not.
#[must_use]
pub fn augmentation(x: &[Vec<f64>], outcomes: &[f64]) -> Option<Augmentation> {
    augmentation_of(x, outcomes)
}

/// [`augmentation`] for columns held in any form.
pub(crate) fn augmentation_of<C: AsRef<[f64]>>(x: &[C], outcomes: &[f64]) -> Option<Augmentation> {
    let p = x.len();
    let k = outcomes.len();
    if p == 0 || k == 0 {
        return None;
    }
    let nr = 2 * p * k;
    let stats: Vec<(f64, f64, f64, f64)> = x
        .iter()
        .map(|col| {
            let col = col.as_ref();
            let n = col.len() as f64;
            let mean = col.iter().sum::<f64>() / n;
            let ss: f64 = col.iter().map(|v| (v - mean) * (v - mean)).sum();
            let sd = (ss / (n - 1.0)).sqrt();
            let min = col.iter().copied().fold(f64::INFINITY, f64::min);
            let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (mean, sd, min, max)
        })
        .collect();
    // mice fills a matrix of nr rows by column from the vector
    // (0.5, -0.5) repeated k times followed by nr zeros, recycled.
    let cycle = 2 * k + nr;
    let mut values = Vec::with_capacity(nr * p);
    for (j, &(mean, sd, min, max)) in stats.iter().enumerate() {
        for i in 0..nr {
            let position = (j * nr + i) % cycle;
            let b = if position < 2 * k {
                if position.is_multiple_of(2) {
                    0.5
                } else {
                    -0.5
                }
            } else {
                0.0
            };
            values.push((mean + b * sd).max(min).min(max));
        }
    }
    let mut x_out = vec![0.0; nr * p];
    for j in 0..p {
        for i in 0..nr {
            x_out[i * p + j] = values[j * nr + i];
        }
    }
    // e = rep(rep(icod, each = 2), p), one entry per row.
    let y = (0..nr).map(|i| outcomes[(i / 2) % k]).collect();
    Some(Augmentation {
        x: x_out,
        y,
        weight: (p as f64 + 1.0) / nr as f64,
        rows: nr,
    })
}

/// A logistic regression, for the parameter draw.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticFit {
    /// The coefficients, 0 for an aliased column.
    pub coefficients: Vec<f64>,
    /// Whether each column is in the model.
    pub active: Vec<bool>,
    /// The covariance of the active coefficients, `(X'WX)^-1`.
    pub covariance: Vec<f64>,
}

/// Fits the logistic model of `y` (0 or 1) on `x` with weights `weights`.
///
/// # Errors
///
/// [`Error::NotConverged`] and the regression's own errors.
pub fn logistic_fit(
    x: MatRef<'_, f64>,
    y: &[f64],
    weights: &[f64],
    column: usize,
) -> Result<LogisticFit> {
    let data = GlmData::builder(x, y).weights(weights).build()?;
    let fit = Glm::new(Family::Binomial, Link::Logit).fit(&data)?;
    if !fit.converged {
        return Err(Error::NotConverged { column });
    }
    let p = x.ncols();
    let active: Vec<bool> = fit.coefficients.iter().map(Option::is_some).collect();
    let keep: Vec<usize> = (0..p).filter(|&j| active[j]).collect();
    let mut covariance = Vec::with_capacity(keep.len() * keep.len());
    for &i in &keep {
        for &j in &keep {
            covariance
                .push(fit.covariance.as_slice()[i * p + j].ok_or(Error::Covariance { column })?);
        }
    }
    Ok(LogisticFit {
        coefficients: fit.coefficients.iter().map(|c| c.unwrap_or(0.0)).collect(),
        active,
        covariance,
    })
}

/// A multinomial regression, for the parameter draw.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct MultinomialFit {
    /// The number of classes.
    pub classes: usize,
    /// The columns of the design, with those left out.
    pub columns: usize,
    /// The coefficients of classes `1..classes`, class by class, `columns`
    /// each, 0 for a column left out.
    pub coefficients: Vec<f64>,
    /// Whether each design column is in the model.
    pub active: Vec<bool>,
    /// The covariance of the active coefficients, class by class.
    pub covariance: Vec<f64>,
}

/// Fits the multinomial logit of `y` (classes `0..classes`) on `x` with
/// weights. A column that is a linear combination of the columns before it is
/// left out, so a collinear design is fitted without it.
///
/// # Errors
///
/// The regression's own errors.
pub fn multinomial_fit(
    x: MatRef<'_, f64>,
    y: &[usize],
    classes: usize,
    weights: &[f64],
) -> Result<MultinomialFit> {
    let p = x.ncols();
    let active = independent_columns(x, weights, None, 1e-8)
        .map_err(|e| Error::Glm(epistat_glm::Error::Design(e)))?;
    let reduced = select_columns(x, &active);
    let data = MultinomData::builder(reduced.as_ref(), y, classes)
        .weights(weights)
        .build()
        .map_err(Error::Multinom)?;
    let fit = Multinom::default().fit(&data).map_err(Error::Multinom)?;
    let pa = reduced.ncols();
    let mut coefficients = vec![0.0; (classes - 1) * p];
    for class in 0..classes - 1 {
        let mut a = 0;
        for j in 0..p {
            if active[j] {
                coefficients[class * p + j] = fit.coefficients[class * pa + a];
                a += 1;
            }
        }
    }
    Ok(MultinomialFit {
        classes,
        columns: p,
        coefficients,
        active,
        covariance: fit
            .covariance
            .as_slice()
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect(),
    })
}

impl MultinomialFit {
    /// `beta* = b + chol(V) z`, laid out as `coefficients`.
    ///
    /// # Errors
    ///
    /// As [`draw_coefficients`].
    pub fn draw(&self, z: &[f64], column: usize) -> Result<Vec<f64>> {
        let p = self.columns;
        let pa = self.active.iter().filter(|&&a| a).count();
        let q = (self.classes - 1) * pa;
        length("z", q, z.len())?;
        let compact: Vec<f64> = (0..self.classes - 1)
            .flat_map(|class| (0..p).filter(|&j| self.active[j]).map(move |j| (class, j)))
            .map(|(class, j)| self.coefficients[class * p + j])
            .collect();
        let all_active = vec![true; q];
        let drawn = draw_coefficients(&compact, &all_active, &self.covariance, z, 1.0, column)?;
        let mut out = vec![0.0; (self.classes - 1) * p];
        let mut next = 0;
        for class in 0..self.classes - 1 {
            for j in 0..p {
                if self.active[j] {
                    out[class * p + j] = drawn[next];
                    next += 1;
                }
            }
        }
        Ok(out)
    }

    /// The number of coefficients that are drawn, `(classes - 1)` times the
    /// active columns.
    #[must_use]
    pub fn n_free(&self) -> usize {
        (self.classes - 1) * self.active.iter().filter(|&&a| a).count()
    }
}

/// The class probabilities of a row `x` under multinomial coefficients laid
/// out as [`MultinomialFit::coefficients`]: class 0 has linear predictor 0.
#[must_use]
pub fn class_probabilities(coefficients: &[f64], classes: usize, x: &[f64]) -> Vec<f64> {
    let p = x.len();
    let mut eta = vec![0.0; classes];
    for class in 1..classes {
        let b = &coefficients[(class - 1) * p..class * p];
        let mut s = 0.0;
        for (xi, bi) in x.iter().zip(b) {
            s += xi * bi;
        }
        eta[class] = s;
    }
    let top = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut prob: Vec<f64> = eta.iter().map(|e| (e - top).exp()).collect();
    let total: f64 = prob.iter().sum();
    for q in &mut prob {
        *q /= total;
    }
    prob
}

/// The first class whose cumulative probability reaches `u`, the last class if
/// rounding leaves the total just below `u`.
#[must_use]
pub fn class_from_uniform(probabilities: &[f64], u: f64) -> usize {
    let mut cumulative = 0.0;
    for (class, q) in probabilities.iter().enumerate() {
        cumulative += q;
        if cumulative >= u {
            return class;
        }
    }
    probabilities.len() - 1
}

/// A design matrix from columns of equal length with a leading intercept,
/// restricted to `rows`.
pub(crate) fn design_matrix<C: AsRef<[f64]>>(columns: &[C], rows: &[usize]) -> Mat<f64> {
    Mat::from_fn(rows.len(), columns.len() + 1, |i, k| {
        if k == 0 {
            1.0
        } else {
            columns[k - 1].as_ref()[rows[i]]
        }
    })
}
