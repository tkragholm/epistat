//! The covariate balancing propensity score for a binary treatment.
//!
//! # Algorithm
//!
//! With sample weights `s_i`, `n = sum s_i`, treatment `t_i`, a design `x_i` and
//! `e_i = expit(x_i' beta)`, the propensity score model has the score
//! conditions `sum s_i x_i (t_i - e_i) = 0`. The weights for the average effect
//! in the population (ATE) are `1 / e_i` for the treated and `1 / (1 - e_i)` for
//! the controls, and they balance the design when
//!
//! ```text
//! sum_i s_i x_i  wt_i = 0,    wt_i = t_i / e_i - (1 - t_i) / (1 - e_i).
//! ```
//!
//! For the effect on the treated (ATT) the controls get the odds `e / (1 - e)`,
//! the treated 1, and the balance condition is `sum s_i x_i (n / n_t) (t_i - e_i)
//! / (1 - e_i) = 0`, with `n_t = sum s_i t_i`. The effect on the controls (ATC)
//! is the ATT of the flipped treatment, and its coefficients are reported for the
//! original treatment. Imai and Ratkovic (2014) choose `beta` so that the balance
//! condition holds, or, over-identified, so that both the score and the balance
//! conditions hold as well as possible.
//!
//! The design is rotated to `x~ = L^-1 x` with `L L' = n^-1 sum s_i x_i x_i'`, so
//! that the iterations work with columns of unit mean square that are
//! uncorrelated under the sample weights. The fit does not change under this
//! transformation, and the coefficients are returned for the design as given.
//!
//! ## Just identified
//!
//! The balance condition is the gradient of a strictly concave function of the
//! linear predictor `eta`:
//!
//! ```text
//! ATE  F = sum s [ (2t - 1) eta - t exp(-eta) - (1 - t) exp(eta) ]
//! ATT  F = sum s [ t eta - (1 - t) exp(eta) ]
//! ```
//!
//! (ATT without the constant factor `n / n_t`, which does not change the root).
//! `F` has a maximum if and only if the balance condition has a solution. It is
//! found by Newton's method from the logistic maximum likelihood fit, with the
//! exact Hessian `-sum s h x x'`, `h = t exp(-eta) + (1 - t) exp(eta)` for the ATE
//! and `(1 - t) exp(eta)` for the ATT, and a step halved until `F` rises by the
//! Armijo amount plus eight units of rounding. The iteration stops when the
//! largest entry of `n^-1 sum s x~ wt` is at most the tolerance. When `F` has no
//! maximum, the coefficients grow without bound and [`Error::Infeasible`] is
//! returned.
//!
//! ## Over-identified
//!
//! With `psi_i = s_i (x_i (t_i - e_i), x_i wt_i)` of dimension `2k`, the
//! moment vector is `g(beta) = n^-1 sum psi_i`, and the estimate minimises `J =
//! g' W g`. `W` is the inverse of the model's expectation of `n^-1 sum psi psi'`,
//!
//! ```text
//! ATE  V = n^-1 sum s [ e(1-e) xx'   xx'              ]
//!                     [ xx'          xx' / (e(1-e))   ]
//! ATT  V = n^-1 sum s [ e(1-e) xx'          (n/n_t) e xx'                ]
//!                     [ (n/n_t) e xx'       (n/n_t)^2 e/(1-e) xx'        ]
//! ```
//!
//! as in the `CBPS` package. With [`CbpsOptions::two_step`], `W = V(beta_0)^-1`
//! at the logistic maximum likelihood fit `beta_0`, which is the fit of
//! `epistat_glm` polished by Newton steps to a score of at most 1e-14, since the
//! fit stops at a deviance change of 1e-8 and would leave `W` imprecise. The package takes `beta_0` scaled by a factor
//! found by a coarse line search in `[0.8, 1.1]`, so its weighting matrix, and
//! with it the minimum, differs from this one in the fourth digit. Without
//! `two_step`, `W = V(beta)^-1` is updated at every `beta`, which is the
//! continuously updated estimator.
//!
//! `J` is not convex, so it is minimised from two starts, as the package does:
//! the logistic fit and the just identified solution (when it exists). The one
//! with the smaller `J` is returned, with its [`Start`]. With `two_step`, `J` is
//! minimised by Newton steps with the exact Hessian `2 (G'WG + C)`, `G = dg/dbeta`,
//! `C = sum_m v_m d^2 g_m / dbeta^2`, `v = W g`, and the Gauss-Newton matrix `G'WG`
//! where the Hessian is not positive definite. The step is halved until `J` falls
//! by the Armijo amount plus rounding, and the iteration stops when the largest
//! entry of the gradient `2 G'W g` is at most the tolerance. The continuously updated `J` is
//! minimised by BFGS from `epistat_optim` with its analytic gradient
//!
//! ```text
//! dJ/dbeta_j = 2 (dg/dbeta_j)' v - v' (dV/dbeta_j) v,    v = W g.
//! ```
//!
//! BFGS searches along a line on `J`, and near the minimum `J` changes by less
//! than its rounding over a range of `beta` of about `sqrt(2 u J / lambda)`, `u`
//! the unit roundoff and `lambda` the smallest curvature, which reaches 1e-8 for
//! the flat objectives of the tests. Where the search stops in that range depends
//! on the last bits of `exp` and so on the platform. The gradient keeps falling
//! there, so the BFGS result is finished by Newton steps on the gradient, with
//! the exact Hessian `2 H`, with `H` as defined under Variance. A step is taken
//! while it reduces the largest entry of the gradient, until that entry is at
//! most the tolerance, for at most 20 steps.
//!
//! ## Weights
//!
//! The weights are `1/e` and `1/(1-e)` (ATE), `n/n_t` and `(n/n_t) e/(1-e)`
//! (ATT), each divided by their sum `sum s wt` over its group when
//! [`CbpsOptions::standardise`] is set, which is the default of the package.
//! Standardised weights sum to 1 in each group and the person counts as `s w`.
//!
//! # Variance
//!
//! The coefficients have the influence `phi_i = -n^-1 A^-1 G'W psi_i` with `A =
//! G'WG`, `G` the derivative of `g`: the influence function of a GMM estimator
//! with weighting matrix `W`, valid for any positive definite `W`. For the just
//! identified estimator it is `-n^-1 G_b^-1 psi_bal,i`, with `G_b` the balance
//! block of `G`.
//!
//! For the two-step estimator the influence is instead the exact derivative of the
//! minimiser with respect to the case weight of the person (the infinitesimal
//! jackknife):
//!
//! ```text
//! phi_i = -n^-1 A^-1 [ G'W (psi_i - s_i kappa_i g) + s_i (E_i - a_1 - kappa_i a_2) ]
//! ```
//!
//! with `A = G'WG + C`, half the Hessian of `J`, where that is positive definite
//! and else `G'WG`. The terms differ from the first form by multiples of `g`, which
//! is of order `n^-1/2` at a correct model, so the two are asymptotically the same.
//! They arise because the normalisation of the moments by `n`, or by `n_t` for the
//! ATT balance block, and the derivative `G` change with the weight of a person.
//! `kappa_i` is 1 for the ATE and `n / n_t` for a treated person of the ATT in the
//! balance block, and 1 in the score block. `E_i = G_i' v` with `v = W g` and `G_i`
//! the derivative of the moments of person `i` alone, and `a_1` and `a_2` are the
//! contributions of the score and balance blocks to `G' v`.
//!
//! For the continuously updated estimator the influence is also the exact
//! derivative of the minimiser. With `v = W g`, `r_i = (x_i' v_1, x_i' v_2)` for
//! the score and balance halves of `v`, the score and balance terms `u = t - e`
//! and `wt`, the blocks `a, b, c` of `V` per person and primes for derivatives
//! in `eta`, half the gradient of `J` is `Q = n^-1 sum s_i x_i f_i` with
//!
//! ```text
//! f = u' r_1 + wt' r_2 - (a' r_1^2 + 2 b' r_1 r_2 + c' r_2^2) / 2,
//! ```
//!
//! and with `p = u' - a' r_1 - b' r_2`, `m = wt' - b' r_1 - c' r_2` and
//! `D = n^-1 sum s_i [x_i p_i; x_i m_i] x_i'`, half its Hessian is `H = n^-1
//! sum s_i (d f_i / d eta) x_i x_i' + D'WD`. The weighting matrix moves with
//! the case weight of person `i` as `v` does, by `n^-1 W (psi_i - M_i v)` with
//! `psi_i` the moments of the person at unit weight and `M_i` its block of `V`,
//! so the influence of the coefficients is
//!
//! ```text
//! phi_i = -n^-1 s_i H^-1 [ x_i f_i - Q + D'W (psi_i - M_i v) ].
//! ```
//!
//! For the ATT the balance moments and their rows of `V` carry the factor `n /
//! n_t`, which cancels in `J`, so it does not enter. Where `H` is not positive
//! definite the first form is used.
//!
//! The derivative of the log weight is `-(1-e) x` for a treated person and `e x`
//! for a control (ATE), 0 for the treated and `x` for the controls (ATT).
//! Standardising divides the weights of a group by their sum, which is estimated
//! too: it is added to the parameters, so the standardised weights of a group
//! `wt / (n lambda)` have the parameter `lambda` with the influence given in the
//! source.
//!
//! # Limits
//!
//! `J` can have several minima, which is why the result states the start it came
//! from. The variances of the just identified, two-step and continuously updated
//! estimators are checked against the infinitesimal jackknife of the whole
//! procedure. The treatment is binary. A continuous treatment, the non-parametric
//! CBPS and the multi-valued treatment are not implemented. A fitted probability
//! numerically 0 or 1 is an error.
//!
//! # References
//!
//! 1. Imai K, Ratkovic M. Covariate balancing propensity score. J R Stat Soc B.
//!    2014;76(1):243-63.
//! 2. Fong C, Hazlett C, Imai K. Covariate balancing propensity score for a
//!    continuous treatment. Ann Appl Stat. 2018;12(1):156-77.
//! 3. Hansen LP, Heaton J, Yaron A. Finite-sample properties of some alternative
//!    GMM estimators. J Bus Econ Stat. 1996;14(3):262-80.

use epistat_glm::{Family, Glm, Link};
use epistat_linalg::{
    Attempt, Mat, MatRef, Method, Regularisation, cholesky_factor, design::independent_columns,
    inverse, matvec_into, solve, weighted_gram, xtv,
};
use epistat_optim::bfgs::Bfgs;

use super::{BalanceWeights, Nuisance, compensated, resolve_weights, summarise};
use epistat_core::{CaseWeights, Estimand};
use epistat_stats::logistic::expit;

use epistat_core::check;

use crate::error::{Error, Model, Result, empty};
use crate::shared::{check_design, check_estimand, fit_glm, glm_data};

/// Whether the balance conditions alone or with the score conditions fix the
/// coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Identification {
    /// The coefficients solve the balance conditions, as many as coefficients.
    Exact,
    /// The balance and score conditions are both used, by generalised method of
    /// moments.
    #[default]
    Over,
}

/// Where the minimum of the over-identified objective was found from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    /// The logistic maximum likelihood fit.
    Logistic,
    /// The just identified solution.
    JustIdentified,
}

/// The data of [`cbps`].
#[derive(Debug, Clone, Copy)]
pub struct CbpsInput<'a> {
    /// Whether each person is treated.
    pub treatment: &'a [bool],
    /// The design of the propensity score model, with the intercept as a column.
    pub design: MatRef<'a, f64>,
    /// Sample weights, 1 for each person when `None`.
    pub sample_weight: Option<CaseWeights<'a>>,
}

/// What [`cbps`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CbpsOptions {
    /// The population the weights refer to: the ATE (the default), ATT or
    /// ATC, `CBPS`'s `ATT = 0, 1, 2`.
    pub estimand: Estimand,
    /// Just or over identified. The default is over identified.
    pub identification: Identification,
    /// Whether the weighting matrix is fixed at the logistic fit (the default)
    /// or updated at every coefficient.
    pub two_step: bool,
    /// Whether the weights sum to 1 in each group, as the `CBPS` package does by
    /// default.
    pub standardise: bool,
    /// The tolerance on the largest absolute entry of the gradient or balance
    /// condition. The default is `1e-10`.
    pub tolerance: f64,
    /// The most iterations of each solver. The default is 200.
    pub max_iterations: usize,
}

impl Default for CbpsOptions {
    fn default() -> Self {
        Self {
            estimand: Estimand::Ate,
            identification: Identification::Over,
            two_step: true,
            standardise: true,
            tolerance: 1e-10,
            max_iterations: 200,
        }
    }
}

/// The result of [`cbps`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Cbps {
    /// The weights and what the outcome estimates need.
    pub weights: BalanceWeights,
    /// The coefficients of the propensity model for the design as given.
    pub coefficients: Vec<f64>,
    /// The fitted probability of treatment of each person.
    pub propensity: Vec<f64>,
    /// The over-identified objective `J` at the estimate, without the factor `n`
    /// of the test statistic, as `CBPS` reports it. `None` when just identified.
    pub objective: Option<f64>,
    /// `J` at the logistic fit, with the same weighting matrix. `None` when just
    /// identified.
    pub objective_logistic: Option<f64>,
    /// The start of the minimum, `None` when just identified.
    pub start: Option<Start>,
    /// The largest standardised difference between the weighted means of the
    /// treated and the controls over the columns of the design that vary.
    pub max_imbalance: f64,
}

/// What the Hessian of the continuously updated `J` and the influence of its
/// minimiser share: the `f` of the module documentation per person as `phi`, the
/// factors `alpha` and `beta` of the weighting-matrix term, `W D` as `wd` and
/// half the Hessian of `J` as `half`.
struct CuSecond {
    phi: Vec<f64>,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    wd: Mat<f64>,
    half: Mat<f64>,
}

struct Problem {
    x: Mat<f64>,
    /// Flipped for the ATC.
    t: Vec<bool>,
    s: Vec<f64>,
    att: bool,
    n: f64,
    /// `n / n_t` for the ATT, else 1.
    c: f64,
}

impl Problem {
    fn k(&self) -> usize {
        self.x.ncols()
    }

    fn eta(&self, beta: &[f64]) -> Vec<f64> {
        let mut eta = vec![0.0; self.x.nrows()];
        matvec_into(
            &mut eta,
            self.x.as_ref(),
            MatRef::from_column_major_slice(beta, beta.len(), 1),
        );
        eta
    }

    fn probs(&self, beta: &[f64]) -> Vec<f64> {
        self.eta(beta).into_iter().map(expit).collect()
    }

    /// The balance weight `wt`.
    fn wbal(&self, e: &[f64]) -> Vec<f64> {
        (0..e.len())
            .map(|i| {
                if self.att {
                    self.c * (f64::from(u8::from(self.t[i])) - e[i]) / (1.0 - e[i])
                } else if self.t[i] {
                    1.0 / e[i]
                } else {
                    -1.0 / (1.0 - e[i])
                }
            })
            .collect()
    }

    /// `d wt / d eta`, never positive.
    fn dwbal(&self, e: &[f64]) -> Vec<f64> {
        (0..e.len())
            .map(|i| {
                if self.att {
                    if self.t[i] {
                        0.0
                    } else {
                        -self.c * e[i] / (1.0 - e[i])
                    }
                } else {
                    let r = f64::from(u8::from(self.t[i])) - e[i];
                    -r * r / (e[i] * (1.0 - e[i]))
                }
            })
            .collect()
    }

    /// `d^2 wt / d eta^2`.
    fn d2wbal(&self, e: &[f64]) -> Vec<f64> {
        (0..e.len())
            .map(|i| {
                if self.att {
                    if self.t[i] {
                        0.0
                    } else {
                        -self.c * e[i] / (1.0 - e[i])
                    }
                } else if self.t[i] {
                    (1.0 - e[i]) / e[i]
                } else {
                    -e[i] / (1.0 - e[i])
                }
            })
            .collect()
    }

    /// `sum_i h_i x_i x_i'` for weights `h` of either sign, divided by `n`.
    fn signed_gram(&self, h: &[f64]) -> Mat<f64> {
        let k = self.k();
        let pos: Vec<f64> = h.iter().map(|&v| v.max(0.0)).collect();
        let neg: Vec<f64> = h.iter().map(|&v| (-v).max(0.0)).collect();
        let gp = weighted_gram(self.x.as_ref(), &pos).unwrap_or_else(|_| Mat::zeros(k, k));
        let gn = weighted_gram(self.x.as_ref(), &neg).unwrap_or_else(|_| Mat::zeros(k, k));
        Mat::from_fn(k, k, |r, c| (gp[(r, c)] - gn[(r, c)]) / self.n)
    }

    /// `sum_m v_m d^2 g_m / dbeta^2`, the part of half the Hessian of `J` that
    /// the Gauss-Newton matrix leaves out, for `v = W g`.
    fn second_order(&self, e: &[f64], v: &[f64]) -> Mat<f64> {
        let k = self.k();
        let (r1, r2) = (self.eta(&v[..k]), self.eta(&v[k..]));
        let d2 = self.d2wbal(e);
        let h: Vec<f64> = (0..e.len())
            .map(|i| {
                self.s[i] * (-r1[i] * e[i] * (1.0 - e[i]) * (1.0 - 2.0 * e[i]) + r2[i] * d2[i])
            })
            .collect();
        self.signed_gram(&h)
    }

    fn sw(&self, v: &[f64]) -> Vec<f64> {
        v.iter().zip(&self.s).map(|(a, s)| a * s).collect()
    }

    /// The moments `g`, length `2k`.
    fn moments(&self, e: &[f64]) -> Vec<f64> {
        let k = self.k();
        let score: Vec<f64> = (0..e.len())
            .map(|i| self.s[i] * (f64::from(u8::from(self.t[i])) - e[i]))
            .collect();
        let bal = self.sw(&self.wbal(e));
        let (a, b) = (xtv(self.x.as_ref(), &score), xtv(self.x.as_ref(), &bal));
        (0..2 * k)
            .map(|j| if j < k { a[(j, 0)] } else { b[(j - k, 0)] } / self.n)
            .collect()
    }

    /// `dg/dbeta`, `2k x k`.
    fn jacobian(&self, e: &[f64]) -> Mat<f64> {
        let k = self.k();
        let top: Vec<f64> = (0..e.len())
            .map(|i| self.s[i] * e[i] * (1.0 - e[i]))
            .collect();
        let bot: Vec<f64> = self
            .dwbal(e)
            .iter()
            .zip(&self.s)
            .map(|(d, s)| -d * s)
            .collect();
        let gt = weighted_gram(self.x.as_ref(), &top).unwrap_or_else(|_| Mat::zeros(k, k));
        let gb = weighted_gram(self.x.as_ref(), &bot).unwrap_or_else(|_| Mat::zeros(k, k));
        Mat::from_fn(2 * k, k, |r, c| {
            if r < k {
                -gt[(r, c)] / self.n
            } else {
                -gb[(r - k, c)] / self.n
            }
        })
    }

    /// The three weights of the blocks of `V`, per person.
    fn vparts(&self, e: &[f64]) -> [Vec<f64>; 3] {
        let m = e.len();
        let a: Vec<f64> = (0..m).map(|i| e[i] * (1.0 - e[i])).collect();
        if self.att {
            [
                a,
                (0..m).map(|i| self.c * e[i]).collect(),
                (0..m)
                    .map(|i| self.c * self.c * e[i] / (1.0 - e[i]))
                    .collect(),
            ]
        } else {
            [
                a.clone(),
                vec![1.0; m],
                (0..m).map(|i| 1.0 / a[i]).collect(),
            ]
        }
    }

    /// The derivatives with respect to `eta` of [`Problem::vparts`].
    fn dvparts(&self, e: &[f64]) -> [Vec<f64>; 3] {
        let m = e.len();
        let da: Vec<f64> = (0..m)
            .map(|i| e[i] * (1.0 - e[i]) * (1.0 - 2.0 * e[i]))
            .collect();
        if self.att {
            [
                da,
                (0..m).map(|i| self.c * e[i] * (1.0 - e[i])).collect(),
                (0..m)
                    .map(|i| self.c * self.c * e[i] / (1.0 - e[i]))
                    .collect(),
            ]
        } else {
            [
                da,
                vec![0.0; m],
                (0..m)
                    .map(|i| -(1.0 - 2.0 * e[i]) / (e[i] * (1.0 - e[i])))
                    .collect(),
            ]
        }
    }

    /// The second derivatives with respect to `eta` of [`Problem::vparts`].
    fn d2vparts(&self, e: &[f64]) -> [Vec<f64>; 3] {
        let m = e.len();
        let d2a: Vec<f64> = (0..m)
            .map(|i| e[i] * (1.0 - e[i]) * (1.0 - 6.0 * e[i] + 6.0 * e[i] * e[i]))
            .collect();
        if self.att {
            [
                d2a,
                (0..m)
                    .map(|i| self.c * e[i] * (1.0 - e[i]) * (1.0 - 2.0 * e[i]))
                    .collect(),
                (0..m)
                    .map(|i| self.c * self.c * e[i] / (1.0 - e[i]))
                    .collect(),
            ]
        } else {
            [
                d2a,
                vec![0.0; m],
                (0..m)
                    .map(|i| {
                        let f = e[i] * (1.0 - e[i]);
                        let d = 1.0 - 2.0 * e[i];
                        2.0 + d * d / f
                    })
                    .collect(),
            ]
        }
    }

    /// The terms of the continuously updated objective at `e` that its Hessian
    /// and the influence of its minimiser share. See the module documentation
    /// for the formulas.
    fn cu_second(&self, e: &[f64]) -> Result<CuSecond> {
        let (k, m) = (self.k(), e.len());
        let w = self.v_inverse(e)?;
        let g = self.moments(e);
        let (_, v) = quad(&w, &g);
        let (r1, r2) = (self.eta(&v[..k]), self.eta(&v[k..]));
        let [a, b, cc] = self.vparts(e);
        let [da, db, dc] = self.dvparts(e);
        let [d2a, d2b, d2c] = self.d2vparts(e);
        let (wb, dw, d2w) = (self.wbal(e), self.dwbal(e), self.d2wbal(e));
        let mut phi = vec![0.0; m];
        let mut h_eta = vec![0.0; m];
        let mut hp = vec![0.0; m];
        let mut hm = vec![0.0; m];
        let mut alpha = vec![0.0; m];
        let mut beta = vec![0.0; m];
        for i in 0..m {
            let (p1, p2) = (r1[i], r2[i]);
            let f = e[i] * (1.0 - e[i]);
            let (du, d2u) = (-f, -f * (1.0 - 2.0 * e[i]));
            phi[i] = du * p1 + dw[i] * p2
                - 0.5 * (da[i] * p1 * p1 + 2.0 * db[i] * p1 * p2 + dc[i] * p2 * p2);
            let dphi = d2u * p1 + d2w[i] * p2
                - 0.5 * (d2a[i] * p1 * p1 + 2.0 * d2b[i] * p1 * p2 + d2c[i] * p2 * p2);
            h_eta[i] = self.s[i] * dphi;
            hp[i] = self.s[i] * (du - da[i] * p1 - db[i] * p2);
            hm[i] = self.s[i] * (dw[i] - db[i] * p1 - dc[i] * p2);
            let u = f64::from(u8::from(self.t[i])) - e[i];
            alpha[i] = u - a[i] * p1 - b[i] * p2;
            beta[i] = wb[i] - b[i] * p1 - cc[i] * p2;
        }
        // D = n^-1 sum s [x p; x m] x' and half the Hessian C_eta + D' W D.
        let (top, bottom) = (self.signed_gram(&hp), self.signed_gram(&hm));
        let d = Mat::from_fn(2 * k, k, |r, c| {
            if r < k {
                top[(r, c)]
            } else {
                bottom[(r - k, c)]
            }
        });
        let wd = Mat::from_fn(2 * k, k, |r, c| {
            compensated((0..2 * k).map(|j| w[(r, j)] * d[(j, c)]))
        });
        let c_eta = self.signed_gram(&h_eta);
        let half = Mat::from_fn(k, k, |r, c| {
            c_eta[(r, c)] + compensated((0..2 * k).map(|j| d[(j, r)] * wd[(j, c)]))
        });
        Ok(CuSecond {
            phi,
            alpha,
            beta,
            wd,
            half,
        })
    }

    /// The influence of the continuously updated coefficients: the exact
    /// derivative of the minimiser with respect to the case weight of each
    /// person, times its sample weight, as rows of an `n x k` matrix. `None`
    /// when the Hessian of `J` is not positive definite there. See the module
    /// documentation for the formulas.
    fn cu_influence(&self, e: &[f64]) -> Result<Option<Mat<f64>>> {
        let (k, m) = (self.k(), e.len());
        let CuSecond {
            phi,
            alpha,
            beta,
            wd,
            half,
        } = self.cu_second(e)?;
        // Q = n^-1 sum s x phi, half the gradient of J; zero at the minimum up to
        // the solver's tolerance.
        let xq = xtv(self.x.as_ref(), &self.sw(&phi));
        let q: Vec<f64> = (0..k).map(|j| xq[(j, 0)] / self.n).collect();
        if cholesky_factor(half.as_ref(), Regularisation::default()).is_none() {
            return Ok(None);
        }
        let hinv = inverse(half.as_ref(), &Attempt::SYMMETRIC)
            .map_err(|_| Error::Singular {
                model: Model::Balance,
            })?
            .x;
        let mut out = Mat::<f64>::zeros(m, k);
        let mut z = vec![0.0; k];
        for i in 0..m {
            for (j, zj) in z.iter_mut().enumerate() {
                let via: f64 = (0..k)
                    .map(|r| self.x[(i, r)] * (alpha[i] * wd[(r, j)] + beta[i] * wd[(k + r, j)]))
                    .sum();
                *zj = self.x[(i, j)] * phi[i] - q[j] + via;
            }
            for r in 0..k {
                let v: f64 = (0..k).map(|c| hinv[(r, c)] * z[c]).sum();
                out[(i, r)] = -self.s[i] * v / self.n;
            }
        }
        Ok(Some(out))
    }

    /// `V`, `2k x 2k`.
    fn v(&self, e: &[f64]) -> Result<Mat<f64>> {
        let k = self.k();
        let [a, b, c] = self.vparts(e);
        let g = |w: &[f64]| {
            weighted_gram(self.x.as_ref(), &self.sw(w))
                .map_err(|err| Error::linalg(Model::Balance, err))
        };
        let (ga, gb, gc) = (g(&a)?, g(&b)?, g(&c)?);
        Ok(Mat::from_fn(2 * k, 2 * k, |r, cc| {
            let v = match (r < k, cc < k) {
                (true, true) => ga[(r, cc)],
                (true, false) => gb[(r, cc - k)],
                (false, true) => gb[(r - k, cc)],
                (false, false) => gc[(r - k, cc - k)],
            };
            v / self.n
        }))
    }

    fn v_inverse(&self, e: &[f64]) -> Result<Mat<f64>> {
        let v = self.v(e)?;
        inverse(v.as_ref(), &Attempt::SYMMETRIC)
            .map(|s| s.x)
            .map_err(|_| Error::Singular {
                model: Model::Balance,
            })
    }
}

fn check_probs(e: &[f64]) -> Result<()> {
    let lim = 1e-12;
    if let Some(i) = e.iter().position(|&p| !(p > lim && p < 1.0 - lim)) {
        return Err(Error::Separation {
            model: Model::Propensity,
            row: i,
        });
    }
    Ok(())
}

/// The influence of the parameters of the weights and the derivative of the log
/// weights.
///
/// With raw weights the parameters are the coefficients. Standardised weights
/// `wt / D_g`, `D_g = sum_g s wt`, divide by a quantity that is itself estimated.
/// Writing `D_g = n lambda_g`, `lambda_g` solves `sum_i s_i (1[i in g] wt_i -
/// lambda_g) = 0`. A constant factor common to the groups leaves a weighted least
/// squares fit unchanged, so the weights are `wt / lambda_g` and `lambda_g` joins
/// the coefficients as a parameter, with influence
/// `n^-1 (s_i (1[i in g] wt_i - lambda_g) + M_g' phi_i)`, `M_g = sum_g s wt dlog wt`.
fn nuisance(
    standardise: bool,
    prob: &Problem,
    raw: &[f64],
    norm: &[f64; 2],
    beta_influence: &Mat<f64>,
    dlog_raw: &Mat<f64>,
) -> Nuisance {
    if !standardise {
        return Nuisance {
            influence: beta_influence.clone(),
            dlog: dlog_raw.clone(),
        };
    }
    let (n, k) = (raw.len(), prob.k());
    let mut influence = Mat::<f64>::zeros(n, k + 2);
    let mut dlog = Mat::<f64>::zeros(n, k + 2);
    for i in 0..n {
        for j in 0..k {
            influence[(i, j)] = beta_influence[(i, j)];
            dlog[(i, j)] = dlog_raw[(i, j)];
        }
    }
    for arm in [false, true] {
        let a = usize::from(arm);
        let lambda = norm[a] / prob.n;
        let m: Vec<f64> = (0..k)
            .map(|j| {
                compensated(
                    (0..n)
                        .filter(|&i| prob.t[i] == arm)
                        .map(|i| prob.s[i] * raw[i] * dlog_raw[(i, j)]),
                )
            })
            .collect();
        for i in 0..n {
            let member = prob.t[i] == arm;
            let own = prob.s[i] * (if member { raw[i] } else { 0.0 } - lambda);
            let via: f64 = (0..k).map(|j| m[j] * beta_influence[(i, j)]).sum();
            influence[(i, k + a)] = (own + via) / prob.n;
            if member {
                dlog[(i, k + a)] = -1.0 / lambda;
            }
        }
    }
    Nuisance { influence, dlog }
}

fn quad(w: &Mat<f64>, g: &[f64]) -> (f64, Vec<f64>) {
    let m = g.len();
    let v: Vec<f64> = (0..m)
        .map(|r| compensated((0..m).map(|c| w[(r, c)] * g[c])))
        .collect();
    (compensated((0..m).map(|r| g[r] * v[r])), v)
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
}

impl Problem {
    /// Newton steps on the logistic score until it is at most 1e-14, from the
    /// fit of `glm`, which stops when the deviance changes by 1e-8 and so leaves
    /// the coefficients, and the weighting matrix built from them, imprecise.
    fn polish_logistic(&self, beta: &mut [f64]) {
        let k = self.k();
        for _ in 0..30 {
            let e = self.probs(beta);
            let score: Vec<f64> = {
                let r: Vec<f64> = (0..e.len())
                    .map(|i| self.s[i] * (f64::from(u8::from(self.t[i])) - e[i]))
                    .collect();
                let x = xtv(self.x.as_ref(), &r);
                (0..k).map(|j| x[(j, 0)] / self.n).collect()
            };
            if max_abs(&score) <= 1e-14 {
                return;
            }
            let h: Vec<f64> = (0..e.len())
                .map(|i| self.s[i] * e[i] * (1.0 - e[i]))
                .collect();
            let Ok(gram) = weighted_gram(self.x.as_ref(), &h) else {
                return;
            };
            let hess = Mat::from_fn(k, k, |r, c| gram[(r, c)] / self.n);
            let rhs = Mat::from_fn(k, 1, |r, _| score[r]);
            let Ok(step) = solve(
                hess.as_ref(),
                rhs.as_ref(),
                &[Attempt::plain(Method::Cholesky)],
            ) else {
                return;
            };
            let moved: Vec<f64> = (0..k).map(|j| beta[j] + step.x[(j, 0)]).collect();
            if moved.iter().any(|b| !b.is_finite()) {
                return;
            }
            beta.copy_from_slice(&moved);
        }
    }

    /// The potential `F` of the balance condition.
    fn potential(&self, beta: &[f64]) -> f64 {
        let eta = self.eta(beta);
        compensated((0..eta.len()).map(|i| {
            let (t, et) = (f64::from(u8::from(self.t[i])), eta[i]);
            self.s[i]
                * if self.att {
                    t * et - (1.0 - t) * et.exp()
                } else {
                    (2.0 * t - 1.0) * et - t * (-et).exp() - (1.0 - t) * et.exp()
                }
        }))
    }

    /// The balance condition `n^-1 sum s x wt`, without the factor `c`.
    fn balance(&self, beta: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let e = self.probs(beta);
        let mut w = self.wbal(&e);
        if self.att {
            for v in &mut w {
                *v /= self.c;
            }
        }
        let b = xtv(self.x.as_ref(), &self.sw(&w));
        ((0..self.k()).map(|j| b[(j, 0)] / self.n).collect(), e)
    }

    fn just_identified(&self, start: &[f64], opts: &CbpsOptions) -> Result<(Vec<f64>, usize)> {
        let k = self.k();
        let mut beta = start.to_vec();
        let mut f = self.potential(&beta);
        for iteration in 0..=opts.max_iterations {
            let (g, _) = self.balance(&beta);
            if max_abs(&g) <= opts.tolerance {
                return Ok((beta, iteration));
            }
            if iteration == opts.max_iterations {
                break;
            }
            let eta = self.eta(&beta);
            let h: Vec<f64> = (0..eta.len())
                .map(|i| {
                    let t = f64::from(u8::from(self.t[i]));
                    self.s[i]
                        * if self.att {
                            (1.0 - t) * eta[i].exp()
                        } else {
                            t * (-eta[i]).exp() + (1.0 - t) * eta[i].exp()
                        }
                })
                .collect();
            let gram =
                weighted_gram(self.x.as_ref(), &h).map_err(|e| Error::linalg(Model::Balance, e))?;
            let hess = Mat::from_fn(k, k, |r, c| gram[(r, c)] / self.n);
            let rhs = Mat::from_fn(k, 1, |r, _| g[r]);
            let step = solve(hess.as_ref(), rhs.as_ref(), &[Attempt::plain(Method::Cholesky)])
                .map_err(|_| Error::Infeasible {
                    model: Model::Balance,
                    why: "the balance conditions have no solution: the curvature of the potential vanishes",
                })?
                .x;
            let slope: f64 = (0..k).map(|j| g[j] * step[(j, 0)]).sum();
            let mut t = 1.0;
            let mut accepted = false;
            for _ in 0..60 {
                let trial: Vec<f64> = (0..k).map(|j| beta[j] + t * step[(j, 0)]).collect();
                let f2 = self.potential(&trial);
                if f2.is_finite()
                    && f2 >= f + 1e-4 * t * slope * self.n - 8.0 * f64::EPSILON * (f.abs() + 1.0)
                {
                    beta = trial;
                    f = f2;
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }
            if !accepted {
                return Err(Error::NotConverged {
                    model: Model::Balance,
                    iterations: None,
                });
            }
            if beta.iter().any(|b| b.abs() > 1e4) {
                return Err(Error::Infeasible {
                    model: Model::Balance,
                    why: "the balance conditions have no solution: the coefficients grow without bound",
                });
            }
        }
        Err(Error::NotConverged {
            model: Model::Balance,
            iterations: Some(opts.max_iterations),
        })
    }

    /// `J` at `beta` for a fixed weighting matrix.
    fn j_fixed(&self, w: &Mat<f64>, beta: &[f64]) -> Result<f64> {
        let e = self.probs(beta);
        check_probs(&e)?;
        Ok(quad(w, &self.moments(&e)).0)
    }

    /// Newton minimisation of `J` for a fixed weighting matrix, with the
    /// Gauss-Newton matrix where the Hessian is not positive definite.
    fn newton_fixed(
        &self,
        w: &Mat<f64>,
        start: &[f64],
        opts: &CbpsOptions,
    ) -> Result<(Vec<f64>, f64, usize)> {
        let k = self.k();
        let mut beta = start.to_vec();
        let mut j = self.j_fixed(w, &beta)?;
        for iteration in 0..=opts.max_iterations {
            let e = self.probs(&beta);
            let g = self.moments(&e);
            let gj = self.jacobian(&e);
            let (_, v) = quad(w, &g);
            let grad: Vec<f64> = (0..k)
                .map(|c| 2.0 * compensated((0..2 * k).map(|r| gj[(r, c)] * v[r])))
                .collect();
            if max_abs(&grad) <= opts.tolerance {
                return Ok((beta, j, iteration));
            }
            if iteration == opts.max_iterations {
                break;
            }
            // G' W G
            let wg = Mat::from_fn(2 * k, k, |r, c| {
                compensated((0..2 * k).map(|m| w[(r, m)] * gj[(m, c)]))
            });
            let a = Mat::from_fn(k, k, |r, c| {
                compensated((0..2 * k).map(|m| gj[(m, r)] * wg[(m, c)]))
            });
            let curvature = self.second_order(&e, &v);
            let full = Mat::from_fn(k, k, |r, c| a[(r, c)] + curvature[(r, c)]);
            let rhs = Mat::from_fn(k, 1, |r, _| -0.5 * grad[r]);
            // Newton where the full Hessian is positive definite, else Gauss-Newton.
            let step = match solve(
                full.as_ref(),
                rhs.as_ref(),
                &[Attempt::plain(Method::Cholesky)],
            ) {
                Ok(sol) => sol.x,
                Err(_) => {
                    solve(a.as_ref(), rhs.as_ref(), &Attempt::SYMMETRIC)
                        .map_err(|_| Error::Singular {
                            model: Model::Balance,
                        })?
                        .x
                }
            };
            let slope: f64 = (0..k).map(|c| grad[c] * step[(c, 0)]).sum();
            let gmax = max_abs(&grad);
            let mut t = 1.0;
            let mut accepted = false;
            for attempt in 0..60 {
                let trial: Vec<f64> = (0..k).map(|c| beta[c] + t * step[(c, 0)]).collect();
                if let Ok(j2) = self.j_fixed(w, &trial)
                    && j2.is_finite()
                {
                    // Near the minimum a step changes J by less than its rounding,
                    // so a full step is also accepted when it halves the gradient.
                    let armijo = j2 <= j + 1e-4 * t * slope + 8.0 * f64::EPSILON * (j.abs() + 1e-3);
                    let shrinks = attempt == 0 && {
                        let e2 = self.probs(&trial);
                        let g2 = self.moments(&e2);
                        let gj2 = self.jacobian(&e2);
                        let (_, v2) = quad(w, &g2);
                        let grad2: Vec<f64> = (0..k)
                            .map(|c| 2.0 * compensated((0..2 * k).map(|r| gj2[(r, c)] * v2[r])))
                            .collect();
                        max_abs(&grad2) < 0.5 * gmax
                    };
                    if armijo || shrinks {
                        beta = trial;
                        j = j2;
                        accepted = true;
                        break;
                    }
                }
                t *= 0.5;
            }
            if !accepted {
                return Err(Error::NotConverged {
                    model: Model::Balance,
                    iterations: None,
                });
            }
        }
        Err(Error::NotConverged {
            model: Model::Balance,
            iterations: Some(opts.max_iterations),
        })
    }

    /// Newton steps on the gradient of the continuously updated `J` from a BFGS
    /// result, with its exact Hessian. Returns the point, `J`, the gradient and the number of steps.
    fn polish_cu(
        &self,
        mut beta: Vec<f64>,
        mut j: f64,
        mut grad: Vec<f64>,
        tolerance: f64,
    ) -> (Vec<f64>, f64, Vec<f64>, usize) {
        let k = self.k();
        let mut steps = 0;
        while steps < 20 && max_abs(&grad) > tolerance {
            let e = self.probs(&beta);
            if check_probs(&e).is_err() {
                break;
            }
            let Ok(second) = self.cu_second(&e) else {
                break;
            };
            let hess = Mat::from_fn(k, k, |r, c| 2.0 * second.half[(r, c)]);
            let rhs = Mat::from_fn(k, 1, |r, _| -grad[r]);
            let Ok(sol) = solve(
                hess.as_ref(),
                rhs.as_ref(),
                &[Attempt::plain(Method::Cholesky)],
            ) else {
                break;
            };
            let trial: Vec<f64> = (0..k).map(|c| beta[c] + sol.x[(c, 0)]).collect();
            let (j2, grad2) = self.cu(&trial);
            if !j2.is_finite() || max_abs(&grad2) >= max_abs(&grad) {
                break;
            }
            beta = trial;
            j = j2;
            grad = grad2;
            steps += 1;
        }
        (beta, j, grad, steps)
    }

    /// `J` and its gradient for the continuously updated weighting matrix.
    fn cu(&self, beta: &[f64]) -> (f64, Vec<f64>) {
        let k = self.k();
        let bad = || (f64::INFINITY, vec![0.0; k]);
        let e = self.probs(beta);
        if check_probs(&e).is_err() {
            return bad();
        }
        let Ok(w) = self.v_inverse(&e) else {
            return bad();
        };
        let g = self.moments(&e);
        let (j, v) = quad(&w, &g);
        let gj = self.jacobian(&e);
        let [da, db, dc] = self.dvparts(&e);
        let m = e.len();
        let r1 = self.eta(&v[..k]);
        let r2 = self.eta(&v[k..]);
        let q: Vec<f64> = (0..m)
            .map(|i| {
                self.s[i]
                    * (da[i] * r1[i] * r1[i] + 2.0 * db[i] * r1[i] * r2[i] + dc[i] * r2[i] * r2[i])
            })
            .collect();
        let xq = xtv(self.x.as_ref(), &q);
        let grad: Vec<f64> = (0..k)
            .map(|c| 2.0 * compensated((0..2 * k).map(|r| gj[(r, c)] * v[r])) - xq[(c, 0)] / self.n)
            .collect();
        (j, grad)
    }
}

/// Covariate balancing propensity score for a binary treatment.
///
/// # Errors
///
/// [`Error::Input`] for values of different lengths, a value that is not finite,
/// a negative sample weight and a tolerance that is not positive;
/// [`Error::Invalid`] for one group without persons of positive weight;
/// [`Error::Aliased`] or [`Error::Singular`] for a design with an aliased
/// column; [`Error::Separation`] for a fitted probability numerically 0 or 1;
/// [`Error::Infeasible`] when the balance conditions have no solution; and
/// [`Error::NotConverged`] when the logistic fit or an iteration does not
/// reach its tolerance.
#[expect(
    clippy::too_many_lines,
    reason = "one pass from the data to the weights and the influence functions"
)]
pub fn cbps(input: &CbpsInput<'_>, opts: &CbpsOptions) -> Result<Cbps> {
    let n = input.treatment.len();
    if n == 0 {
        return Err(empty("treatment"));
    }
    check_design("design", input.design, n)?;
    let k = input.design.ncols();
    check::positive("tolerance", opts.tolerance)?;
    check_estimand(opts.estimand)?;
    let s = resolve_weights("sample_weight", input.sample_weight, n)?;
    let flip = opts.estimand == Estimand::Atc;
    let t: Vec<bool> = input.treatment.iter().map(|&a| a != flip).collect();
    let (nt, nc) = (
        compensated((0..n).filter(|&i| t[i]).map(|i| s[i])),
        compensated((0..n).filter(|&i| !t[i]).map(|i| s[i])),
    );
    if nt <= 0.0 || nc <= 0.0 {
        return Err(Error::invalid(
            "treatment",
            "both groups need persons of positive weight",
        ));
    }
    let total = nt + nc;
    let keep = independent_columns(input.design, &s, None, 1e-8)
        .map_err(|e| Error::linalg(Model::Propensity, e))?;
    if let Some(column) = keep.iter().position(|&x| !x) {
        return Err(Error::Aliased {
            model: Model::Propensity,
            column,
        });
    }
    // Whiten: L L' = n^-1 X' S X, x~_i = L^-1 x_i.
    let gram = weighted_gram(input.design, &s).map_err(|e| Error::linalg(Model::Propensity, e))?;
    let scaled = Mat::from_fn(k, k, |r, c| gram[(r, c)] / total);
    let l = cholesky_factor(scaled.as_ref(), Regularisation::default()).ok_or(Error::Singular {
        model: Model::Propensity,
    })?;
    let mut x = Mat::<f64>::zeros(n, k);
    for i in 0..n {
        for r in 0..k {
            let mut v = input.design[(i, r)];
            for c in 0..r {
                v -= l[(r, c)] * x[(i, c)];
            }
            x[(i, r)] = v / l[(r, r)];
        }
    }
    let prob = Problem {
        x,
        t: t.clone(),
        s: s.clone(),
        att: opts.estimand != Estimand::Ate,
        n: total,
        c: if opts.estimand == Estimand::Ate {
            1.0
        } else {
            total / nt
        },
    };
    let y: Vec<f64> = t.iter().map(|&a| f64::from(u8::from(a))).collect();
    let what = Model::Propensity;
    let (_, mut ml) = fit_glm(
        &Glm::new(Family::Binomial, Link::Logit),
        &glm_data(prob.x.as_ref(), &y, Some(&s), None, what)?,
        what,
    )?;
    prob.polish_logistic(&mut ml);
    check_probs(&prob.probs(&ml))?;

    let mut iterations = 0;
    let (beta, objective, objective_logistic, start) = match opts.identification {
        Identification::Exact => {
            let (b, it) = prob.just_identified(&ml, opts)?;
            iterations += it;
            (b, None, None, None)
        }
        Identification::Over => {
            let exact = prob.just_identified(&ml, opts).ok();
            let e0 = prob.probs(&ml);
            let w0 = prob.v_inverse(&e0)?;
            let j_ml = quad(&w0, &prob.moments(&e0)).0;
            let mut starts = vec![(Start::Logistic, ml.clone())];
            if let Some((b, it)) = exact {
                iterations += it;
                starts.push((Start::JustIdentified, b));
            }
            let mut best: Option<(Start, Vec<f64>, f64)> = None;
            let mut last_err = None;
            for (which, s0) in starts {
                let run = if opts.two_step {
                    prob.newton_fixed(&w0, &s0, opts)
                } else {
                    let bfgs = Bfgs {
                        max_iterations: opts.max_iterations,
                        gradient_tolerance: opts.tolerance,
                        cost_tolerance: 1e-300,
                        ..Bfgs::default()
                    };
                    bfgs.minimize(&s0, |b| prob.cu(b), None)
                        .map_err(|_| Error::NotConverged {
                            model: Model::Balance,
                            iterations: None,
                        })
                        .and_then(|m| {
                            let (x, value, gradient, steps) =
                                prob.polish_cu(m.x, m.value, m.gradient, opts.tolerance);
                            let iterations = m.iterations + steps;
                            if m.converged
                                || max_abs(&gradient) <= opts.tolerance
                                || gradient.iter().all(|g| g.abs() <= 1e-7)
                            {
                                Ok((x, value, iterations))
                            } else {
                                Err(Error::NotConverged {
                                    model: Model::Balance,
                                    iterations: Some(iterations),
                                })
                            }
                        })
                };
                match run {
                    Ok((b, j, it)) => {
                        iterations += it;
                        if best.as_ref().is_none_or(|(_, _, jb)| j < *jb) {
                            best = Some((which, b, j));
                        }
                    }
                    Err(e) => last_err = Some(e),
                }
            }
            let (which, b, j) = best.ok_or_else(|| {
                last_err.unwrap_or(Error::NotConverged {
                    model: Model::Balance,
                    iterations: None,
                })
            })?;
            let jl = if opts.two_step { j_ml } else { prob.cu(&ml).0 };
            (b, Some(j), Some(jl), Some(which))
        }
    };

    let e = prob.probs(&beta);
    check_probs(&e)?;
    // Positive weights before standardising, and the derivative of their logs.
    let mut raw = vec![0.0; n];
    let mut dlog_raw = Mat::<f64>::zeros(n, k);
    for i in 0..n {
        let (r, dr): (f64, f64) = if opts.estimand == Estimand::Ate {
            if t[i] {
                (1.0 / e[i], -(1.0 - e[i]))
            } else {
                (1.0 / (1.0 - e[i]), e[i])
            }
        } else if t[i] {
            (prob.c, 0.0)
        } else {
            (prob.c * e[i] / (1.0 - e[i]), 1.0)
        };
        raw[i] = r;
        for j in 0..k {
            dlog_raw[(i, j)] = dr * prob.x[(i, j)];
        }
    }
    let mut weights = raw.clone();
    let mut norm = [1.0, 1.0];
    if opts.standardise {
        for arm in [false, true] {
            let d = compensated((0..n).filter(|&i| t[i] == arm).map(|i| s[i] * raw[i]));
            norm[usize::from(arm)] = d;
            for i in (0..n).filter(|&i| t[i] == arm) {
                weights[i] = raw[i] / d;
            }
        }
    }
    // The influence of the coefficients.
    let e_final = &e;
    let jac = prob.jacobian(e_final);
    let wbal = prob.wbal(e_final);
    let (kmat, kmat_extra): (Mat<f64>, Mat<f64>) = match opts.identification {
        Identification::Exact => {
            let gb = Mat::from_fn(k, k, |r, c| jac[(k + r, c)]);
            let inv = inverse(gb.as_ref(), &Attempt::SYMMETRIC)
                .map_err(|_| Error::Singular {
                    model: Model::Balance,
                })?
                .x;
            // K = -n^-1 Gb^-1 on the balance block, 0 on the score block.
            (
                Mat::from_fn(k, 2 * k, |r, c| {
                    if c >= k {
                        -inv[(r, c - k)] / total
                    } else {
                        0.0
                    }
                }),
                Mat::zeros(k, k),
            )
        }
        Identification::Over => {
            let w = if opts.two_step {
                prob.v_inverse(&prob.probs(&ml))?
            } else {
                prob.v_inverse(e_final)?
            };
            let wg = Mat::from_fn(2 * k, k, |r, c| {
                compensated((0..2 * k).map(|m| w[(r, m)] * jac[(m, c)]))
            });
            let mut a = Mat::from_fn(k, k, |r, c| {
                compensated((0..2 * k).map(|m| jac[(m, r)] * wg[(m, c)]))
            });
            if opts.two_step {
                // The exact derivative of the minimiser: add the second-order
                // term when that keeps the matrix positive definite, as it does
                // at a strict minimum.
                let g = prob.moments(e_final);
                let (_, v) = quad(&w, &g);
                let c = prob.second_order(e_final, &v);
                let full = Mat::from_fn(k, k, |r, cc| a[(r, cc)] + c[(r, cc)]);
                if cholesky_factor(full.as_ref(), Regularisation::default()).is_some() {
                    a = full;
                }
            }
            let ainv = inverse(a.as_ref(), &Attempt::SYMMETRIC)
                .map_err(|_| Error::Singular {
                    model: Model::Balance,
                })?
                .x;
            // K = -n^-1 A^-1 G'W
            (
                Mat::from_fn(k, 2 * k, |r, c| {
                    -compensated((0..k).map(|m| ainv[(r, m)] * wg[(c, m)])) / total
                }),
                Mat::from_fn(k, k, |r, c| -ainv[(r, c)] / total),
            )
        }
    };
    // The two-step over-identified fit: the exact derivative of the minimiser with
    // respect to the weight of each person. Its moments do not average to zero, so
    // the normalisation by n (by n_t for the ATT balance block) makes the
    // contribution of person i `psi_i - s_i kappa_i g`, which sums to zero, and the
    // derivative `G` of the moments depends on the weight too, through
    // `s_i (E_i - a_1 - kappa_i a_2)` below, with `E_i` the part of `G_i' v` for
    // person i, `v = W g`, and `a_1` and `a_2` the parts of `G' v` of the two blocks.
    // These terms are proportional to `g`.
    let exact = opts.identification == Identification::Over && opts.two_step;
    let gbar = prob.moments(e_final);
    let mut extra = vec![vec![0.0; k]; if exact { n } else { 0 }];
    let (mut a1, mut a2) = (vec![0.0; k], vec![0.0; k]);
    if exact {
        let w = prob.v_inverse(&prob.probs(&ml))?;
        let (_, v) = quad(&w, &gbar);
        for c in 0..k {
            a1[c] = compensated((0..k).map(|r| jac[(r, c)] * v[r]));
            a2[c] = compensated((0..k).map(|r| jac[(k + r, c)] * v[k + r]));
        }
        let (r1, r2) = (prob.eta(&v[..k]), prob.eta(&v[k..]));
        let dw = prob.dwbal(e_final);
        for i in 0..n {
            let kappa = if opts.estimand == Estimand::Ate {
                1.0
            } else {
                prob.c * f64::from(u8::from(t[i]))
            };
            for c in 0..k {
                let e_ic = prob.x[(i, c)] * (-r1[i] * e[i] * (1.0 - e[i]) + r2[i] * dw[i]);
                extra[i][c] = s[i] * (e_ic - a1[c] - kappa * a2[c]);
            }
        }
    }
    // The continuously updated fit: the exact derivative of its minimiser, where
    // the Hessian of `J` is positive definite, else the first-order form below.
    let updated = if opts.identification == Identification::Over && !opts.two_step {
        prob.cu_influence(e_final)?
    } else {
        None
    };
    let beta_influence = if let Some(exact_cu) = updated {
        exact_cu
    } else {
        let mut beta_influence = Mat::<f64>::zeros(n, k);
        for i in 0..n {
            let ti = f64::from(u8::from(t[i]));
            let kappa = if opts.estimand == Estimand::Ate {
                1.0
            } else {
                prob.c * ti
            };
            let mut psi = vec![0.0; 2 * k];
            for c in 0..k {
                psi[c] = s[i] * prob.x[(i, c)] * (ti - e[i]);
                psi[k + c] = s[i] * prob.x[(i, c)] * wbal[i];
                if exact {
                    psi[c] -= s[i] * gbar[c];
                    psi[k + c] -= s[i] * kappa * gbar[k + c];
                }
            }
            for r in 0..k {
                let mut v: f64 = (0..2 * k).map(|c| kmat[(r, c)] * psi[c]).sum();
                if exact {
                    v += (0..k)
                        .map(|c| kmat_extra[(r, c)] * extra[i][c])
                        .sum::<f64>();
                }
                beta_influence[(i, r)] = v;
            }
        }
        beta_influence
    };
    let nuisance = nuisance(
        opts.standardise,
        &prob,
        &raw,
        &norm,
        &beta_influence,
        &dlog_raw,
    );
    // Coefficients and probabilities for the design as given, original labels.
    let mut coefficients = vec![0.0; k];
    for r in (0..k).rev() {
        let mut v = beta[r];
        for c in (r + 1)..k {
            v -= l[(c, r)] * coefficients[c];
        }
        coefficients[r] = v / l[(r, r)];
    }
    let propensity: Vec<f64> = if flip {
        for b in &mut coefficients {
            *b = -*b;
        }
        e.iter().map(|p| 1.0 - p).collect()
    } else {
        e.clone()
    };
    // Imbalance between the weighted groups.
    let mut max_imbalance = 0.0_f64;
    let all: Vec<usize> = (0..n).filter(|&i| s[i] > 0.0).collect();
    for j in 0..k {
        let col = |i: usize| input.design[(i, j)];
        let m = compensated(all.iter().map(|&i| s[i] * col(i))) / total;
        let sd = (compensated(all.iter().map(|&i| s[i] * (col(i) - m).powi(2))) / total).sqrt();
        if sd == 0.0 {
            continue;
        }
        let wmean = |arm: bool| {
            let d = compensated(
                (0..n)
                    .filter(|&i| input.treatment[i] == arm)
                    .map(|i| s[i] * weights[i]),
            );
            compensated(
                (0..n)
                    .filter(|&i| input.treatment[i] == arm)
                    .map(|i| s[i] * weights[i] * col(i)),
            ) / d
        };
        max_imbalance = max_imbalance.max(((wmean(true) - wmean(false)) / sd).abs());
    }
    let ctrl = summarise(
        opts.estimand != Estimand::Atc,
        false,
        input.treatment,
        &s,
        &weights,
    );
    let trt = summarise(
        opts.estimand != Estimand::Att,
        true,
        input.treatment,
        &s,
        &weights,
    );
    Ok(Cbps {
        weights: BalanceWeights::new(
            opts.estimand,
            weights,
            s,
            input.treatment.to_vec(),
            ctrl,
            trt,
            iterations,
            nuisance,
        ),
        coefficients,
        propensity,
        objective,
        objective_logistic,
        start,
        max_imbalance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Twice `cu_second(..).half` is the Jacobian of the gradient of `cu`. A
    /// central difference of the gradient at step `h` has a truncation error of
    /// order `h^2 f'''/6` and a rounding error of order `u |grad| / h`, `u` the
    /// unit roundoff. Over `h` from 1e-3 to 1e-6 the deviation from the exact
    /// matrix falls as `h^2` to 2e-7 of the largest entry at `h = 1e-4` and then
    /// rises with rounding, so 1e-6 of the largest entry at `h = 1e-4` accepts
    /// the difference's error and rejects a wrong term, which is of order one.
    #[test]
    fn the_exact_hessian_is_the_jacobian_of_the_gradient() {
        let n = 200;
        let mut rng = epistat_rng::Stream::seeded(5);
        let x = Mat::from_fn(n, 3, |_, c| if c == 0 { 1.0 } else { rng.normal() });
        let t: Vec<bool> = (0..n)
            .map(|i| rng.uniform() < expit(0.4 * x[(i, 1)] - 0.3))
            .collect();
        for ate in [true, false] {
            let nt = t.iter().filter(|&&a| a).count() as f64;
            let prob = Problem {
                x: x.clone(),
                t: t.clone(),
                s: vec![1.0; n],
                att: !ate,
                n: n as f64,
                c: if ate { 1.0 } else { n as f64 / nt },
            };
            let beta = [0.1, -0.3, 0.2];
            let e = prob.probs(&beta);
            let exact = prob.cu_second(&e).unwrap().half;
            let h = 1e-4;
            let mut scale = 0.0_f64;
            let mut worst = 0.0_f64;
            for c in 0..3 {
                let (mut up, mut down) = (beta, beta);
                up[c] += h;
                down[c] -= h;
                let (gu, gd) = (prob.cu(&up).1, prob.cu(&down).1);
                for r in 0..3 {
                    let fd = (gu[r] - gd[r]) / (2.0 * h);
                    scale = scale.max(fd.abs());
                    worst = worst.max((fd - 2.0 * exact[(r, c)]).abs());
                }
            }
            assert!(
                worst < 1e-6 * scale,
                "ate {ate}: {worst:e} against {scale:e}"
            );
        }
    }
}
