//! The average causal response to a continuous treatment, after Callaway,
//! Goodman-Bacon and Sant'Anna \[1\], as R's `contdid` estimates it.
//!
//! [`estimate`] takes a [`ContinuousData`], the dose and outcome change of
//! each unit, and [`AcrtOptions`], whose [`BasisSpec`] names the sieve, and
//! returns an [`Acrt`].
//!
//! The Average Causal Response on the Treated (ACRT) is estimated with a sieve approximation to the
//! dose-response function: the control trend is estimated, the treated units'
//! outcome changes are projected onto a basis in the dose, and the derivative
//! of the fitted function is averaged over the treated doses.
//!
//! # Algorithm
//!
//! With `ΔY_i` the outcome change, `D_i` the dose, `w_i` the weight and
//! `ψ(d)` a `K`-dimensional basis, the units with `D = 0` are the controls `C`
//! and those with `D > 0` the treated `T`. The estimator solves
//!
//! ```text
//! m0 = sum_C w_i ΔY_i / W_C,                    W_C = sum_C w_i,
//! θ  = H^-1 sum_T w_i ψ(D_i) (ΔY_i - m0),       H   = sum_T w_i ψ(D_i) ψ(D_i)',
//! ACRT = (1 / n_T) sum_T ψ'(D_i)' θ,
//! ```
//!
//! the last an unweighted mean over the `n_T` treated units.
//!
//! # Influence function
//!
//! The three equations are estimating equations of `(m0, θ, ACRT)` averaged
//! over all `n` units, so the influence function of a unit is the linearisation
//! of the whole procedure, scaled so that `ACRT - ACRT_0 ≈ (1/n) sum_i IF_i`,
//! the per-unit scale of [`epistat_inference::Influence`]. With `α = H^-1 dbar`,
//! `dbar = (1 / n_T) sum_T ψ'(D_i)`, `r_i = ΔY_i - m0 - ψ(D_i)' θ` and
//! `s = sum_T w_i ψ(D_i)`,
//!
//! ```text
//! treated:   IF_i = (n / n_T)(ψ'(D_i)' θ - ACRT) + n w_i α' ψ(D_i) r_i
//! control:   IF_i = - α' s · n w_i (ΔY_i - m0) / W_C
//! ```
//!
//! and 0 for a unit with a negative dose, which is in neither group. The control
//! term is the effect of the estimated trend through `θ`. It is zero, up to
//! rounding, when the basis spans the constants, as the polynomial basis and
//! the full B-spline basis do: the trend then moves only the constant of the
//! fit, whose derivative is zero.
//!
//! # References
//!
//! 1. Callaway B, Goodman-Bacon A, Sant'Anna PHC. Difference-in-differences
//!    with a continuous treatment. NBER Working Paper 32117, 2024; R package
//!    `contdid`.

mod sieve;

use epistat_core::{Estimate, InputError, Level, check};
use epistat_inference::Influence;
use epistat_linalg::basis::{BSpline, Basis, Polynomial};

use crate::error::Error;
use crate::inference::{estimate_from, normal_interval};
use crate::methods::two_period::row_influence;

/// The basis of the sieve in the dose.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum BasisSpec {
    /// The powers `1, d, ..., d^degree`, or without `constant` the powers
    /// from `d`.
    Polynomial {
        /// The highest power.
        degree: usize,
        /// Whether the constant function is in the basis.
        constant: bool,
    },
    /// B-splines of `degree` with sorted `internal_knots` strictly inside
    /// `[lower, upper]`, R's `splines::bs(d, knots, degree, intercept = TRUE,
    /// Boundary.knots = c(lower, upper))`. A dose outside the boundary has no
    /// basis values, and is refused.
    BSpline {
        /// The degree of the pieces.
        degree: usize,
        /// The interior knots.
        internal_knots: Vec<f64>,
        /// The lower boundary knot.
        lower: f64,
        /// The upper boundary knot.
        upper: f64,
    },
}

impl Default for BasisSpec {
    /// The cubic polynomials, which span the functions of `contdid`'s default
    /// cubic B-spline basis without interior knots, so the ACRT is the same
    /// up to the rounding of the normal equations.
    fn default() -> Self {
        Self::Polynomial {
            degree: 3,
            constant: true,
        }
    }
}

/// The powers of a polynomial without its constant.
struct WithoutConstant(Polynomial);

impl Basis for WithoutConstant {
    fn len(&self) -> usize {
        self.0.degree
    }

    fn values_into(&self, x: f64, out: &mut [f64]) {
        let all = self.0.values(x);
        out[..self.len()].copy_from_slice(&all[1..]);
    }

    fn derivatives_into(&self, x: f64, out: &mut [f64]) {
        let all = self.0.derivatives(x);
        out[..self.len()].copy_from_slice(&all[1..]);
    }
}

impl BasisSpec {
    fn build(&self) -> Result<Box<dyn Basis>, Error> {
        match self {
            &Self::Polynomial { degree, constant } => {
                if constant {
                    Ok(Box::new(Polynomial { degree }))
                } else if degree == 0 {
                    Err(InputError::Domain {
                        what: "degree of a basis without its constant",
                        value: 0.0,
                        domain: "[1, inf)",
                    }
                    .into())
                } else {
                    Ok(Box::new(WithoutConstant(Polynomial { degree })))
                }
            }
            Self::BSpline {
                degree,
                internal_knots,
                lower,
                upper,
            } => BSpline::new(*degree, internal_knots, *lower, *upper)
                .map(|spline| Box::new(spline) as Box<dyn Basis>)
                .map_err(|_| Error::Unsupported {
                    what: "B-spline knots that are not sorted and strictly inside a finite, increasing boundary",
                }),
        }
    }

    /// Whether `dose` lies where the basis is defined.
    fn covers(&self, dose: f64) -> bool {
        match self {
            Self::Polynomial { .. } => true,
            Self::BSpline { lower, upper, .. } => *lower <= dose && dose <= *upper,
        }
    }
}

/// A validated sample of a continuous treatment, one unit per row.
///
/// A unit with dose 0 is a control, a unit with a positive dose is treated,
/// and a unit with a negative dose is in neither group and has no influence.
/// [`ContinuousData::builder`] takes the dose and the outcome change of each
/// unit between the two periods, and [`ContinuousDataBuilder::build`] checks
/// that every column has one entry per unit, at least one unit, that the
/// doses and changes are finite and that the weights are finite and not
/// negative.
#[derive(Clone, Copy, Debug)]
pub struct ContinuousData<'a> {
    dose: &'a [f64],
    change: &'a [f64],
    weights: Option<&'a [f64]>,
}

/// The builder of a [`ContinuousData`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct ContinuousDataBuilder<'a> {
    dose: &'a [f64],
    change: &'a [f64],
    weights: Option<&'a [f64]>,
}

impl<'a> ContinuousData<'a> {
    /// The data of the doses `dose` and the outcome changes `change`.
    pub const fn builder(dose: &'a [f64], change: &'a [f64]) -> ContinuousDataBuilder<'a> {
        ContinuousDataBuilder {
            dose,
            change,
            weights: None,
        }
    }

    /// The dose of each unit.
    #[must_use]
    pub const fn dose(&self) -> &'a [f64] {
        self.dose
    }

    /// The outcome change of each unit.
    #[must_use]
    pub const fn change(&self) -> &'a [f64] {
        self.change
    }

    /// The prior weight of each unit, `None` when every unit weighs 1.
    #[must_use]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.weights
    }

    /// The number of units.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.dose.len()
    }

    /// Whether there are no units, which the builder refuses, so always false.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.dose.is_empty()
    }
}

impl<'a> ContinuousDataBuilder<'a> {
    /// Prior weights, one per unit; a unit of weight 0 counts in neither the
    /// trend nor the sieve, but in the means over units.
    pub const fn weights(mut self, weights: &'a [f64]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// The validated data.
    ///
    /// # Errors
    ///
    /// [`Error::Input`]: [`InputError::Empty`] without units,
    /// [`InputError::TooLarge`] for more units than `u32` numbers,
    /// [`InputError::Length`] for a column of the wrong length,
    /// [`InputError::NotFinite`] for a dose, change or weight that is not
    /// finite and [`InputError::Domain`] for a negative weight.
    pub fn build(self) -> Result<ContinuousData<'a>, Error> {
        let n = self.dose.len();
        if n == 0 {
            return Err(InputError::Empty { what: "dose" }.into());
        }
        if u32::try_from(n).is_err() {
            return Err(InputError::TooLarge {
                what: "units of a continuous treatment",
                limit: u64::from(u32::MAX),
                found: u64::try_from(n).unwrap_or(u64::MAX),
            }
            .into());
        }
        check::finite("dose", self.dose)?;
        check::same_length("change", n, self.change.len())?;
        check::finite("change", self.change)?;
        if let Some(weights) = self.weights {
            check::same_length("weights", n, weights.len())?;
            check::finite("weights", weights)?;
            if let Some(&value) = weights.iter().find(|&&w| w < 0.0) {
                return Err(InputError::Domain {
                    what: "weights",
                    value,
                    domain: "[0, inf)",
                }
                .into());
            }
        }
        Ok(ContinuousData {
            dose: self.dose,
            change: self.change,
            weights: self.weights,
        })
    }
}

/// The options of [`estimate`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AcrtOptions {
    /// The sieve's basis in the dose.
    pub basis: BasisSpec,
    /// The confidence level of the normal interval.
    pub level: Level,
}

/// The ACRT of a continuous treatment.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Acrt {
    /// The unweighted mean over the treated units of the derivative of the
    /// fitted dose-response function at their dose, with its standard error
    /// from the influence function and the normal interval.
    pub estimate: Estimate,
    /// The sieve coefficients, one per basis function in the basis's order.
    pub coefficients: Vec<f64>,
    /// Units with a positive dose.
    pub n_treated: usize,
    /// Units with dose 0.
    pub n_control: usize,
    /// The influence function, one value per unit and 0 for a negative dose,
    /// under [`epistat_inference::ClusterRule::NoSmallSample`].
    pub influence: Influence,
}

/// The ACRT of `data` with the sieve of `options.basis`.
///
/// ```
/// use epistat_did::continuous::{self, AcrtOptions, BasisSpec, ContinuousData};
///
/// // A response of slope 5 in the dose above a common trend of 2.
/// let dose: Vec<f64> = (0..40).map(|i| if i < 20 { 0.0 } else { f64::from(i - 19) }).collect();
/// let change: Vec<f64> = dose.iter().map(|d| 2.0 + 5.0 * d).collect();
/// let data = ContinuousData::builder(&dose, &change).build()?;
/// let fit = continuous::estimate(&data, &AcrtOptions {
///     basis: BasisSpec::Polynomial { degree: 1, constant: true },
///     ..AcrtOptions::default()
/// })?;
/// assert!((fit.estimate.estimate - 5.0).abs() < 1e-9);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// [`Error::Input`] for a polynomial without its constant of degree 0;
/// [`Error::Unsupported`] for B-spline knots that do not define a basis, or a
/// positive dose outside its boundary; [`Error::NoComparison`] when the units of dose 0
/// have no weight, [`Error::NoTreated`] without a positive dose and
/// [`Error::Singular`] for a sieve whose normal equations are singular.
pub fn estimate(data: &ContinuousData<'_>, options: &AcrtOptions) -> Result<Acrt, Error> {
    let basis = options.basis.build()?;
    if !data
        .dose
        .iter()
        .all(|&d| d <= 0.0 || options.basis.covers(d))
    {
        return Err(Error::Unsupported {
            what: "a dose outside the boundary of the B-spline basis",
        });
    }
    let weight = |i: usize| data.weights.map_or(1.0, |w| w[i]);
    let fit = sieve::estimate(data.dose, data.change, &weight, basis.as_ref())?;
    let ci = normal_interval(fit.acrt, fit.se, options.level.confidence_value());
    Ok(Acrt {
        estimate: estimate_from(fit.acrt, fit.se, ci, options.level),
        coefficients: fit.coefficients,
        n_treated: fit.n_treated,
        n_control: fit.n_control,
        influence: row_influence(fit.influence)?,
    })
}
