//! Wooldridge's extended two-way fixed-effects regression \[1\] and its
//! average effects, as `etwfe::etwfe` and `etwfe::emfx` 0.6.2 compute them
//! \[2\], for linear models and, with cohort and period dummies, for Poisson
//! and logit models.
//!
//! # Algorithm
//!
//! For unit `i` in period `t`, first treated in period `g_i`, with
//! covariates `x_it` and their deviations `x.._it = x_it - xbar_{g_i}` from the
//! mean of the unit's cohort (over every row of the input, unweighted, as
//! `etwfe` demeans), the regression is
//!
//! ```text
//! y_it = a_{g_i or i} + l_t + x_it' b + sum_{g != r} 1[g_i = g] x_it' b_g + sum_{s != t_1} 1[t = s] x_it' c_s
//!        + sum_{(g, s) in C} D_it 1[g_i = g, t = s] (tau_gs + x.._it' d_gs) + e_it
//! ```
//!
//! with cohort effects (`etwfe`'s default) or unit effects
//! ([`EtwfeEffects::Unit`], `ivar`), `r` the reference cohort and `t_1` the
//! first period. The cells `C` and the indicator `D_it` depend on the
//! controls:
//!
//! - [`EtwfeControl::NotYetTreated`] (`cgroup = "notyet"`): `D_it = 1[t >=
//!   g_i]`, so the cells are the treated ones and the untreated rows of every
//!   cohort are controls. Without never-treated units the last cohort is the
//!   reference and the rows from its first treated period on are removed.
//! - [`EtwfeControl::NeverTreated`] (`cgroup = "never"`): `D_it = 1[t != g_i -
//!   1]`, every cell of a treated cohort but the one before treatment, so
//!   only the never treated are controls and the cells before treatment are
//!   placebo effects.
//!
//! The reference cohort is the never treated; without them, the cohorts first
//! treated after the last period, which are never treated in the panel.
//! Cohorts first treated in or before the first period have no untreated row
//! and are removed.
//!
//! The unit or cohort and period effects are absorbed by
//! [`epistat_glm::absorb`] and the regression is fitted by
//! [`epistat_glm::feols`], so the cost is linear in the rows at a fixed
//! number of cells and covariates. The covariance is that of the caller's
//! [`Vcov`]; `fixest`'s default, which `etwfe` keeps, is iid.
//!
//! # Average effects
//!
//! `emfx` averages, over a set of rows, the contrast of the prediction with
//! `D = 1` and with `D = 0`,
//!
//! ```text
//! c_it = tau_{g_i t} + x.._it' d_{g_i t},       theta = mean_{rows in S} c_it = J beta
//! ```
//!
//! with zero for a cell without a coefficient. The rows are the treated
//! cells' (with never-treated controls, the rows of treated cohorts from
//! their first treated period on), and for [`EtwfeType::Event`] with
//! never-treated controls every row of a treated cohort; the averages are
//! overall, by cohort, by period and by event time `t - g`, and unweighted, as
//! `emfx` weights every row by one. `J` is the mean over the rows of the
//! derivative of `c_it` in the coefficients, and the delta-method variance is
//! `J V J'` with `V` the covariance of the coefficients; `marginaleffects`
//! finds `J` by forward differences, which for this linear map is exact but
//! for rounding. The variance takes the rows and the cohort means of the
//! covariates as fixed. A zero `J` (the cell before treatment, whose
//! coefficient the normalisation sets to zero) has no standard error.
//!
//! # Influence
//!
//! The influence function of a coefficient for unit `u` is
//! `psi_u = N_u e_k' B sum_{rows of u} w_it e_it x~_it` (`N_u` the units of the
//! fit, `x~` the demeaned regressors, `B = (X~' W X~)^-1`), and that of an
//! average effect is `J psi_u`. Summed within unit clusters by
//! [`did_cluster_sums`](crate::did_cluster_sums) they give the unit-clustered
//! covariance without small-sample factors.
//!
//! # Nonlinear models
//!
//! [`estimate_etwfe_glm`] fits the Poisson (log link) and logit models of
//! `etwfe(family = "poisson" / "logit")`. `etwfe` then calls `fixest::feglm`
//! with `fe = "none"` and cohort effects: the absorbed effects become an
//! intercept and the dummies of the cohorts other than the reference and of
//! the periods after the first,
//!
//! ```text
//! g(E y_it) = a + c_{g_i} + l_t + x_it' b + sum_{g != r} 1[g_i = g] x_it' b_g + sum_{s != t_1} 1[t = s] x_it' c_s
//!             + sum_{(g, s) in C} D_it 1[g_i = g, t = s] (tau_gs + x.._it' d_gs)
//! ```
//!
//! with the cells, the indicator and the covariates of the linear model and
//! `c_r = l_{t_1} = 0`. The columns are, in `etwfe`'s order, the intercept,
//! the covariates and their cohort and period slopes, the cohort and period
//! dummies, the cells and the cells' slopes ([`EtwfeGlmTerm`]). The rows of
//! weight zero are removed, as `fixest` removes them. The model is fitted by
//! [`Glm::fit_rows`](epistat_glm::Glm::fit_rows), the iteration of R's
//! `glm.fit` \[3\] on rows formed block by block, so the memory is that of the
//! panel and the cost linear in the rows; `feglm` iterates the same way, and
//! both stop when the deviance changes by less than `tol (|D| + 0.1)`.
//!
//! With `B = (X'WX)^-1` at the working weights of the last iteration, the
//! scores `s_i = w_i (y_i - mu_i) x_i`, `n` rows and `K` coefficients, the
//! covariances are those of `fixest` for a model without fixed effects:
//!
//! ```text
//! iid        (n - 1) / (n - K) B
//! hetero     n / (n - K) B (sum_i s_i s_i') B
//! cluster    G / (G - 1) (n - 1) / (n - K) B (sum_g s_g s_g') B
//! ```
//!
//! with the factors in `n` and `K` dropped without [`Ssc::k_adj`] and the
//! one in `G` without [`Ssc::g_adj`]. `fixest`'s default is iid.
//!
//! `emfx` averages over the rows of the linear case the contrast of the
//! predicted mean with `D = 1` and with `D = 0`, `mu = g^-1`:
//!
//! ```text
//! theta = mean_S [mu(eta1_it) - mu(eta0_it)],     eta1_it = eta0_it + tau_{g_i t} + x.._it' d_{g_i t}
//! J     = mean_S [mu'(eta1_it) x1_it - mu'(eta0_it) x0_it]
//! ```
//!
//! with `x0` the row's design without its cell and `x1` with it, and
//! `Var(theta) = J V J'`. `marginaleffects` finds `J` by forward differences
//! with steps `max(|beta_j| sqrt(eps), 1e-10)`, which err by about half the
//! step times the second derivative; here `J` is exact. The influence
//! function of a coefficient for unit `u` is `N_u B sum_{rows of u} s_i`, and
//! that of an average `J psi_u`; summed within unit clusters they give the
//! unit-clustered covariance without small-sample factors.
//!
//! `emfx` replaces the rows by their cohort and period means from 500 000
//! rows on (`compress = "auto"`), which for these models changes the
//! averages; here every row enters, as with `compress = FALSE`. `fixest`
//! forms the clustered and robust scores from the working residuals of the
//! last iteration, at the coefficients before the last update; here they are
//! taken at the estimates. The two differ at the size of the convergence
//! criterion. One cluster variable is supported.
//!
//! # Where this differs from `etwfe` 0.6.2
//!
//! `etwfe` takes as reference one cohort value: the smallest of those after
//! the last period, else the smallest before the first period, else the last
//! cohort. When never-treated units coded before the first period sit beside
//! a cohort treated after the last, that cohort becomes the reference and the
//! never treated are taken as treated in every period; cohorts treated before
//! the first period are kept as treated in every period. Here the never
//! treated and every cohort treated after the last period are controls, and
//! the always treated are removed.
//!
//! # References
//!
//! 1. Wooldridge JM. Two-way fixed effects, the two-way Mundlak regression,
//!    and difference-in-differences estimators. Empir Econ.
//!    2025;69:2545-87. doi:10.1007/s00181-025-02807-z (working paper 2021)
//! 2. McDermott G. The R package `etwfe` 0.6.2: `etwfe` and `emfx`; with
//!    `marginaleffects` 1.0.0 (`slopes`) and `fixest` 0.14.2 (`feols`,
//!    `feglm`).
//! 3. R Core Team. Fitting generalized linear models: `glm` and `glm.fit`.
//!    Help pages of the stats package of R.

mod glm;

use std::collections::BTreeSet;

pub use glm::{
    EtwfeFamily, EtwfeGlmCoefficient, EtwfeGlmConfig, EtwfeGlmResult, EtwfeGlmTerm,
    estimate_etwfe_glm,
};

use epistat_glm::{Absorb, Error as AbsorbError, Feols, FixedEffects, Singletons, Ssc, Vcov};
use epistat_linalg::Mat;

use crate::methods::imputation::check_constant_within_units;
use crate::methods::staggered::StaggeredPanel;
use crate::methods::sunab::period_codes;

/// The controls of the regression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EtwfeControl {
    /// The untreated rows of every cohort (`cgroup = "notyet"`).
    NotYetTreated,
    /// The never treated only (`cgroup = "never"`).
    NeverTreated,
}

/// The fixed effects besides the periods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EtwfeEffects {
    /// One per cohort, `etwfe`'s default.
    Cohort,
    /// One per unit (`ivar`).
    Unit,
}

/// The controls of [`estimate_etwfe`].
#[derive(Clone, Copy, Debug)]
pub struct EtwfeConfig<'a> {
    /// The controls.
    pub control: EtwfeControl,
    /// The fixed effects.
    pub effects: EtwfeEffects,
    /// The covariance of the coefficients, from which the standard errors of
    /// the average effects follow.
    pub vcov: Vcov<'a>,
    /// The demeaning.
    pub absorb: Absorb,
    /// The small-sample correction of the covariance.
    pub ssc: Ssc,
}

impl EtwfeConfig<'_> {
    /// The defaults of `etwfe`: not-yet-treated controls, cohort effects and
    /// `fixest`'s default iid covariance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            control: EtwfeControl::NotYetTreated,
            effects: EtwfeEffects::Cohort,
            vcov: Vcov::Iid,
            absorb: Absorb::new(),
            ssc: Ssc::new(),
        }
    }
}

impl Default for EtwfeConfig<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Why an extended two-way fixed-effects fit failed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EtwfeError {
    /// An input has the wrong length.
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        /// The input.
        what: &'static str,
        /// The length it needed.
        expected: usize,
        /// The length it had.
        found: usize,
    },
    /// A covariate is not finite.
    #[error("covariate {column} is not finite on row {row}")]
    NotFinite {
        /// The covariate.
        column: usize,
        /// The row.
        row: usize,
    },
    /// A unit has rows with different first treated periods.
    #[error("unit {unit} has rows with different first treated periods")]
    Cohort {
        /// The unit.
        unit: u32,
    },
    /// Never-treated controls were asked for and no cohort is never treated.
    #[error("no unit is never treated in the panel")]
    NoNeverTreated,
    /// No row carries a cell indicator.
    #[error("no row carries a cohort and period indicator")]
    NoTreatment,
    /// A period does not fit in 32 bits.
    #[error("the periods span more than 2^31 values")]
    Periods,
    /// The fit failed.
    #[error(transparent)]
    Fit(#[from] AbsorbError),
}

/// A column of the regression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EtwfeTerm {
    /// Covariate `k`.
    Covariate(usize),
    /// Covariate `k` on the rows of a cohort.
    CohortSlope {
        /// The cohort.
        cohort: i32,
        /// The covariate.
        covariate: usize,
    },
    /// Covariate `k` on the rows of a period.
    PeriodSlope {
        /// The period.
        period: i32,
        /// The covariate.
        covariate: usize,
    },
    /// The cell indicator `D 1[g, t]`, the effect `tau_gt` at the cohort's
    /// mean covariates.
    Cell {
        /// The cohort.
        cohort: i32,
        /// The period.
        period: i32,
    },
    /// The cell indicator times the cohort-demeaned covariate `k`.
    CellSlope {
        /// The cohort.
        cohort: i32,
        /// The period.
        period: i32,
        /// The covariate.
        covariate: usize,
    },
}

/// One coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct EtwfeCoefficient {
    /// Its column.
    pub term: EtwfeTerm,
    /// The coefficient, `None` when the column is collinear.
    pub estimate: Option<f64>,
    /// Its standard error.
    pub std_error: Option<f64>,
}

/// The averages of `emfx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EtwfeType {
    /// Over every row (`type = "simple"`).
    Simple,
    /// By cohort (`"group"`).
    Group(i32),
    /// By period (`"calendar"`).
    Calendar(i32),
    /// By event time `t - g` (`"event"`).
    Event(i32),
}

/// One average effect.
#[derive(Clone, Debug, PartialEq)]
pub struct EtwfeEffect {
    /// What it averages.
    pub term: EtwfeType,
    /// The mean contrast.
    pub estimate: f64,
    /// `sqrt(J V J')`, `None` when `J` is zero or the covariance undefined.
    pub std_error: Option<f64>,
    /// The rows it averages.
    pub n_rows: usize,
}

/// An extended two-way fixed-effects fit and its average effects.
#[derive(Clone, Debug)]
pub struct EtwfeResult {
    /// The coefficients in `etwfe`'s order: the covariates, their cohort and
    /// period slopes, the cells by cohort and period, and the cells' slopes
    /// by covariate.
    pub coefficients: Vec<EtwfeCoefficient>,
    /// The covariance of the coefficients, row-major, `None` for dropped
    /// columns.
    pub covariance: Vec<Option<f64>>,
    /// The average effects: overall, by cohort, by period and by event time,
    /// each in increasing key.
    pub effects: Vec<EtwfeEffect>,
    /// `J V J'` over the average effects, row-major.
    pub effect_covariance: Vec<Option<f64>>,
    /// The influence function of each average effect, one value per unit of
    /// [`EtwfeResult::units`].
    pub effect_influence: Vec<Vec<f64>>,
    /// The influence function of each coefficient, one value per unit;
    /// `None` for dropped columns.
    pub influence: Vec<Option<Vec<f64>>>,
    /// The `K` of the small-sample correction.
    pub df_k: usize,
    /// The units in the fit, in increasing code.
    pub units: Vec<u32>,
    /// The rows in the fit.
    pub n_rows: usize,
    /// The rows removed as singletons.
    pub removed_singletons: usize,
    /// The rows removed before the fit: those of always-treated cohorts and,
    /// without never-treated units under not-yet-treated controls, those
    /// from the last cohort's first treated period on.
    pub excluded_rows: usize,
}

/// The cohort of a row as a key: `None` for never treated.
type Key = Option<i32>;

/// The reference, the cells, the columns and the rows of the regression of
/// one panel under one choice of controls, shared by the linear and the
/// nonlinear fits.
struct Layout<'p> {
    panel: &'p StaggeredPanel<'p>,
    control: EtwfeControl,
    /// The number of covariates.
    p: usize,
    /// The period codes of the fixed effects, from 0.
    time: Vec<u32>,
    /// Every cohort, never treated left out.
    cohorts: BTreeSet<i32>,
    /// The cohorts with cell indicators.
    treated: Vec<i32>,
    excluded: Vec<bool>,
    excluded_rows: usize,
    /// The index of each row's cohort: 0 for never treated, then the
    /// cohorts in increasing order.
    row_key: Vec<usize>,
    /// The cohort means of the covariates, `row_key` by covariate.
    means: Vec<f64>,
    /// The cohorts other than the reference with rows in the fit.
    slope_cohorts: Vec<i32>,
    /// The periods after the first with rows in the fit.
    slope_periods: Vec<i32>,
    cells: Vec<(i32, i32)>,
    /// The cell of each row whose indicator is one.
    row_cell: Vec<Option<usize>>,
    /// The columns of the linear regression.
    terms: Vec<EtwfeTerm>,
    /// The position of the first cell among `terms`.
    first_cell: usize,
}

impl<'p> Layout<'p> {
    fn new(panel: &'p StaggeredPanel<'p>, control: EtwfeControl) -> Result<Self, EtwfeError> {
        let n = panel.unit.len();
        for (what, found) in [
            ("time", panel.time.len()),
            ("first_treated", panel.first_treated.len()),
            ("outcome", panel.outcome.len()),
            ("weights", panel.weights.map_or(n, <[f64]>::len)),
            ("covariates", panel.covariates.map_or(n, |x| x.nrows())),
        ] {
            if found != n {
                return Err(EtwfeError::Length {
                    what,
                    expected: n,
                    found,
                });
            }
        }
        check_constant_within_units(panel.unit, panel.first_treated)
            .map_err(|unit| EtwfeError::Cohort { unit })?;
        let time = period_codes(panel.time).ok_or(EtwfeError::Periods)?;
        let p = panel.covariates.map_or(0, |x| x.ncols());
        if let Some(x) = panel.covariates {
            for column in 0..p {
                if let Some(row) = (0..n).find(|&i| !x[(i, column)].is_finite()) {
                    return Err(EtwfeError::NotFinite { column, row });
                }
            }
        }
        let Some(t_first) = panel.time.iter().copied().min() else {
            return Err(EtwfeError::NoTreatment);
        };
        let t_last = panel.time.iter().copied().max().unwrap_or(t_first);

        // The reference, the treated cohorts and the rows removed.
        let cohorts: BTreeSet<i32> = panel.first_treated.iter().flatten().copied().collect();
        let has_never = panel.first_treated.iter().any(Option::is_none);
        let after: Vec<i32> = cohorts.iter().copied().filter(|&g| g > t_last).collect();
        let mut treated: Vec<i32> = cohorts
            .iter()
            .copied()
            .filter(|&g| g > t_first && g <= t_last)
            .collect();
        let (reference, cutoff): (Key, Option<i32>) = if has_never {
            (None, None)
        } else if let Some(&g) = after.first() {
            (Some(g), None)
        } else if control == EtwfeControl::NeverTreated {
            return Err(EtwfeError::NoNeverTreated);
        } else {
            let last = treated.pop().ok_or(EtwfeError::NoTreatment)?;
            (Some(last), Some(last))
        };
        let excluded: Vec<bool> = (0..n)
            .map(|i| {
                panel.first_treated[i].is_some_and(|g| g <= t_first)
                    || cutoff.is_some_and(|c| panel.time[i] >= c)
            })
            .collect();
        let excluded_rows = excluded.iter().filter(|&&b| b).count();

        // The cohort means of the covariates, over every row of the input.
        let key_index = |g: Key| -> usize {
            g.map_or(0, |g| 1 + cohorts.iter().position(|&c| c == g).unwrap_or(0))
        };
        let groups = 1 + cohorts.len();
        let mut means = vec![0.0; groups * p];
        let mut counts = vec![0usize; groups];
        let row_key: Vec<usize> = panel.first_treated.iter().map(|&g| key_index(g)).collect();
        for (i, &k) in row_key.iter().enumerate() {
            counts[k] += 1;
            if let Some(x) = panel.covariates {
                for j in 0..p {
                    means[k * p + j] += x[(i, j)];
                }
            }
        }
        for k in 0..groups {
            for j in 0..p {
                if counts[k] > 0 {
                    means[k * p + j] /= counts[k] as f64;
                }
            }
        }

        let mut layout = Self {
            panel,
            control,
            p,
            time,
            cohorts,
            treated,
            excluded,
            excluded_rows,
            row_key,
            means,
            slope_cohorts: Vec::new(),
            slope_periods: Vec::new(),
            cells: Vec::new(),
            row_cell: Vec::new(),
            terms: Vec::new(),
            first_cell: 0,
        };

        // The columns.
        let kept_rows = |i: usize| !layout.excluded[i];
        layout.slope_cohorts = layout
            .cohorts
            .iter()
            .copied()
            .filter(|&g| {
                Some(g) != reference
                    && (0..n).any(|i| kept_rows(i) && panel.first_treated[i] == Some(g))
            })
            .collect();
        layout.slope_periods = (0..n)
            .filter(|&i| kept_rows(i) && panel.time[i] != t_first)
            .map(|i| panel.time[i])
            .collect::<BTreeSet<i32>>()
            .into_iter()
            .collect();
        layout.cells = (0..n)
            .filter(|&i| layout.indicator(i))
            .map(|i| (panel.first_treated[i].unwrap_or(0), panel.time[i]))
            .collect::<BTreeSet<(i32, i32)>>()
            .into_iter()
            .collect();
        if layout.cells.is_empty() {
            return Err(EtwfeError::NoTreatment);
        }
        let mut terms: Vec<EtwfeTerm> = (0..p).map(EtwfeTerm::Covariate).collect();
        for k in 0..p {
            terms.extend(
                layout
                    .slope_cohorts
                    .iter()
                    .map(|&cohort| EtwfeTerm::CohortSlope {
                        cohort,
                        covariate: k,
                    }),
            );
        }
        for k in 0..p {
            terms.extend(
                layout
                    .slope_periods
                    .iter()
                    .map(|&period| EtwfeTerm::PeriodSlope {
                        period,
                        covariate: k,
                    }),
            );
        }
        layout.first_cell = terms.len();
        terms.extend(
            layout
                .cells
                .iter()
                .map(|&(cohort, period)| EtwfeTerm::Cell { cohort, period }),
        );
        for k in 0..p {
            terms.extend(
                layout
                    .cells
                    .iter()
                    .map(|&(cohort, period)| EtwfeTerm::CellSlope {
                        cohort,
                        period,
                        covariate: k,
                    }),
            );
        }
        layout.terms = terms;
        layout.row_cell = (0..n)
            .map(|i| {
                if layout.indicator(i) {
                    layout.cell_of(i)
                } else {
                    None
                }
            })
            .collect();
        Ok(layout)
    }

    fn n(&self) -> usize {
        self.panel.unit.len()
    }

    fn is_treated(&self, g: Key) -> bool {
        g.is_some_and(|g| self.treated.binary_search(&g).is_ok())
    }

    /// Whether the cell indicator `D_it 1[g_i = g, t = s]` of row `i` is one
    /// for its own cohort and period.
    fn indicator(&self, i: usize) -> bool {
        let g = self.panel.first_treated[i];
        if self.excluded[i] || !self.is_treated(g) {
            return false;
        }
        let g = g.unwrap_or(0);
        match self.control {
            EtwfeControl::NotYetTreated => self.panel.time[i] >= g,
            EtwfeControl::NeverTreated => self.panel.time[i] != g - 1,
        }
    }

    /// Covariate `j` of row `i` less its cohort mean.
    fn demeaned(&self, i: usize, j: usize) -> f64 {
        self.panel.covariates.map_or(0.0, |x| {
            x[(i, j)] - self.means[self.row_key[i] * self.p + j]
        })
    }

    /// The cell of row `i`'s cohort and period, whatever its indicator.
    fn cell_of(&self, i: usize) -> Option<usize> {
        let g = self.panel.first_treated[i]?;
        self.cells.binary_search(&(g, self.panel.time[i])).ok()
    }

    /// The prior weights of the fit: zero for the rows removed, `None` when
    /// every row has weight one.
    fn fit_weights(&self) -> Option<Vec<f64>> {
        (self.panel.weights.is_some() || self.excluded_rows > 0).then(|| {
            (0..self.n())
                .map(|i| {
                    if self.excluded[i] {
                        0.0
                    } else {
                        self.panel.weights.map_or(1.0, |w| w[i])
                    }
                })
                .collect()
        })
    }

    /// Whether row `i` enters the effects overall, by cohort and by period.
    fn effect_row(&self, i: usize) -> bool {
        match self.control {
            EtwfeControl::NotYetTreated => self.indicator(i),
            EtwfeControl::NeverTreated => {
                self.indicator(i) && self.panel.time[i] >= self.panel.first_treated[i].unwrap_or(0)
            }
        }
    }

    /// Whether row `i` enters the effects by event time.
    fn event_row(&self, i: usize) -> bool {
        match self.control {
            EtwfeControl::NotYetTreated => self.indicator(i),
            EtwfeControl::NeverTreated => {
                !self.excluded[i] && self.is_treated(self.panel.first_treated[i])
            }
        }
    }

    /// The averages of `emfx`.
    fn averages(&self) -> Averages {
        let panel = self.panel;
        let set = |f: &dyn Fn(usize) -> Option<i32>| -> Vec<i32> {
            (0..self.n())
                .filter_map(f)
                .collect::<BTreeSet<i32>>()
                .into_iter()
                .collect()
        };
        let group = set(&|i| {
            self.effect_row(i)
                .then(|| panel.first_treated[i].unwrap_or(0))
        });
        let calendar = set(&|i| self.effect_row(i).then_some(panel.time[i]));
        let event = set(&|i| {
            self.event_row(i)
                .then(|| panel.time[i] - panel.first_treated[i].unwrap_or(0))
        });
        let mut keys: Vec<EtwfeType> = vec![EtwfeType::Simple];
        keys.extend(group.iter().map(|&g| EtwfeType::Group(g)));
        keys.extend(calendar.iter().map(|&t| EtwfeType::Calendar(t)));
        keys.extend(event.iter().map(|&e| EtwfeType::Event(e)));
        Averages {
            keys,
            group,
            calendar,
            event,
        }
    }
}

/// The averages of `emfx`: overall, by cohort, by period and by event time,
/// each in increasing key.
struct Averages {
    keys: Vec<EtwfeType>,
    group: Vec<i32>,
    calendar: Vec<i32>,
    event: Vec<i32>,
}

impl Averages {
    /// The averages row `i` enters, `usize::MAX` in the slots of those it
    /// does not: overall, by cohort, by period and by event time. `None`
    /// when it enters none.
    fn targets(&self, layout: &Layout<'_>, i: usize) -> Option<[usize; 4]> {
        let in_effect = layout.effect_row(i);
        let in_event = layout.event_row(i);
        if !in_effect && !in_event {
            return None;
        }
        let panel = layout.panel;
        let g = panel.first_treated[i].unwrap_or(0);
        let g_offset = 1;
        let c_offset = g_offset + self.group.len();
        let e_offset = c_offset + self.calendar.len();
        let mut targets = [usize::MAX; 4];
        if in_effect {
            targets[0] = 0;
            targets[1] = g_offset + self.group.binary_search(&g).unwrap_or(0);
            targets[2] = c_offset + self.calendar.binary_search(&panel.time[i]).unwrap_or(0);
        }
        if in_event {
            targets[3] = e_offset + self.event.binary_search(&(panel.time[i] - g)).unwrap_or(0);
        }
        Some(targets)
    }
}

/// The units of the rows `rows`, in increasing code, and the position of
/// each row's unit among them.
fn unit_codes(unit: &[u32], rows: &[usize]) -> (Vec<u32>, Vec<u32>) {
    let unit_rows: Vec<u32> = rows.iter().map(|&i| unit[i]).collect();
    let mut units: Vec<u32> = unit_rows.clone();
    units.sort_unstable();
    units.dedup();
    let codes: Vec<u32> = unit_rows
        .iter()
        .map(|u| u32::try_from(units.binary_search(u).unwrap_or(0)).unwrap_or(u32::MAX))
        .collect();
    (units, codes)
}

/// Fits the extended two-way fixed-effects regression of `config` to `panel`
/// and its average effects. See the module documentation.
///
/// # Errors
///
/// [`EtwfeError`] for inputs of the wrong length or not finite, cohorts that
/// vary within a unit, never-treated controls without never-treated units,
/// no treated cell, periods out of range, and the errors of the fit.
#[expect(
    clippy::too_many_lines,
    reason = "one pass from the layout to the design, the fit and the averages"
)]
pub fn estimate_etwfe(
    panel: &StaggeredPanel<'_>,
    config: &EtwfeConfig<'_>,
) -> Result<EtwfeResult, EtwfeError> {
    let layout = Layout::new(panel, config.control)?;
    let n = layout.n();
    let p = layout.p;
    let terms = &layout.terms;
    let first_cell = layout.first_cell;
    let n_cells = layout.cells.len();
    let row_cell = &layout.row_cell;
    let excluded_rows = layout.excluded_rows;
    let slope_cohorts = &layout.slope_cohorts;
    let slope_periods = &layout.slope_periods;

    let q = terms.len();
    let mut x = Mat::<f64>::zeros(n, q);
    if let Some(cov) = panel.covariates {
        let gc = slope_cohorts.len();
        let tc = slope_periods.len();
        for i in 0..n {
            let cohort_slot =
                panel.first_treated[i].and_then(|g| slope_cohorts.binary_search(&g).ok());
            let period_slot = slope_periods.binary_search(&panel.time[i]).ok();
            for k in 0..p {
                let v = cov[(i, k)];
                x[(i, k)] = v;
                if let Some(a) = cohort_slot {
                    x[(i, p + k * gc + a)] = v;
                }
                if let Some(b) = period_slot {
                    x[(i, p + p * gc + k * tc + b)] = v;
                }
            }
        }
    }
    for (i, c) in row_cell.iter().enumerate() {
        if let Some(c) = *c {
            x[(i, first_cell + c)] = 1.0;
            for k in 0..p {
                x[(i, first_cell + n_cells * (k + 1) + c)] = layout.demeaned(i, k);
            }
        }
    }
    let fit_weights = layout.fit_weights();
    let cohort_code: Vec<u32>;
    let first_factor: &[u32] = match config.effects {
        EtwfeEffects::Unit => panel.unit,
        EtwfeEffects::Cohort => {
            cohort_code = layout
                .row_key
                .iter()
                .map(|&k| u32::try_from(k).unwrap_or(u32::MAX))
                .collect();
            &cohort_code
        }
    };
    let fe = FixedEffects::new(
        &[first_factor, &layout.time],
        fit_weights.as_deref(),
        Singletons::Remove,
    )?;
    let fit = Feols {
        absorb: config.absorb,
    }
    .fit_owned(x, panel.outcome, &fe)?;
    let cov = fit.vcov(&fe, config.vcov, config.ssc)?;
    let coefficients: Vec<EtwfeCoefficient> = terms
        .iter()
        .enumerate()
        .map(|(j, &term)| EtwfeCoefficient {
            term,
            estimate: fit.coefficients[j],
            std_error: cov.matrix.as_slice()[j * q + j].map(|v| v.max(0.0).sqrt()),
        })
        .collect();

    // Influence by unit: the units are the panel's units in either case.
    let (units, unit_codes) = unit_codes(panel.unit, fe.rows());
    let n_units = units.len();
    let sums = fit.influence_sums(&fe, &unit_codes, n_units)?;
    let mut influence = vec![None; q];
    for (a, &j) in fit.kept_columns().iter().enumerate() {
        influence[j] = Some(
            (0..n_units)
                .map(|u| n_units as f64 * sums[(u, a)])
                .collect::<Vec<f64>>(),
        );
    }

    // The averages: sums of the cell indicators and demeaned covariates over
    // each set of rows.
    let averages = layout.averages();
    let keys = &averages.keys;
    let m = keys.len();
    let width = n_cells * (1 + p);
    let mut sums_j = vec![0.0; m * width];
    let mut rows_in = vec![0usize; m];
    for i in 0..n {
        let Some(targets) = averages.targets(&layout, i) else {
            continue;
        };
        // The contrast's derivative: the row's cell and its slopes, zero
        // for a row whose cell has no column.
        let cell = layout.cell_of(i);
        for &a in targets.iter().filter(|&&a| a != usize::MAX) {
            rows_in[a] += 1;
            if let Some(c) = cell {
                sums_j[a * width + c] += 1.0;
                for k in 0..p {
                    sums_j[a * width + n_cells * (k + 1) + c] += layout.demeaned(i, k);
                }
            }
        }
    }
    // J over all columns, row by effect.
    let jacobian: Vec<Vec<f64>> = (0..m)
        .map(|a| {
            let mut row = vec![0.0; q];
            let rows = rows_in[a] as f64;
            for (c, value) in sums_j[a * width..(a + 1) * width].iter().enumerate() {
                let j = first_cell + c;
                if fit.coefficients[j].is_some() {
                    row[j] = value / rows;
                }
            }
            row
        })
        .collect();
    let mut effect_covariance = vec![None; m * m];
    for a in 0..m {
        for b in 0..m {
            let mut total = 0.0;
            let mut defined = true;
            for (i, &ja) in jacobian[a].iter().enumerate() {
                if ja == 0.0 {
                    continue;
                }
                for (j, &jb) in jacobian[b].iter().enumerate() {
                    if jb == 0.0 {
                        continue;
                    }
                    match cov.matrix.as_slice()[i * q + j] {
                        Some(v) => total += ja * jb * v,
                        None => defined = false,
                    }
                }
            }
            effect_covariance[a * m + b] = defined.then_some(total);
        }
    }
    let effects: Vec<EtwfeEffect> = keys
        .iter()
        .enumerate()
        .map(|(a, &term)| {
            let zero = jacobian[a].iter().all(|&v| v == 0.0);
            EtwfeEffect {
                term,
                estimate: jacobian[a]
                    .iter()
                    .zip(&fit.coefficients)
                    .map(|(j, b)| j * b.unwrap_or(0.0))
                    .sum(),
                std_error: if zero {
                    None
                } else {
                    effect_covariance[a * m + a].map(|v| v.max(0.0).sqrt())
                },
                n_rows: rows_in[a],
            }
        })
        .collect();
    let effect_influence: Vec<Vec<f64>> = jacobian
        .iter()
        .map(|row| {
            let mut out = vec![0.0; n_units];
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0
                    && let Some(psi) = &influence[j]
                {
                    for (o, s) in out.iter_mut().zip(psi) {
                        *o += v * s;
                    }
                }
            }
            out
        })
        .collect();

    Ok(EtwfeResult {
        coefficients,
        covariance: cov.matrix.into_values(),
        effects,
        effect_covariance,
        effect_influence,
        influence,
        df_k: cov.df_k,
        units,
        n_rows: fe.n_rows(),
        removed_singletons: fe.removed_singletons(),
        excluded_rows,
    })
}
