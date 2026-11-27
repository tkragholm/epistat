//! The matching estimator of Abadie and Imbens, its bias correction and its
//! variance.
//!
//! The estimator matches every unit of the target group on the covariates,
//! with replacement, to the `M` nearest units of the other treatment group,
//! and averages the differences in outcome. It estimates the average effect
//! in the treated ([`Estimand::Att`]), the controls ([`Estimand::Atc`]) or
//! everyone ([`Estimand::Ate`]). Its variance is not the variance of a mean of
//! independent differences, since a unit used as a match for several others
//! enters several differences. Abadie and Imbens \[1\] give a variance that
//! allows for it. This module is a separate estimator from
//! [`crate::effects`]: it does the matching itself, on the covariates, and
//! gives the estimate and its error without an outcome model.
//!
//! # Matching
//!
//! Covariates are standardised by their sample standard deviation over all
//! units. The distance is Euclidean in the standardised covariates
//! ([`DistanceWeight::Diagonal`]) or Mahalanobis ([`DistanceWeight::Mahalanobis`]),
//! with the covariance of the standardised covariates taken with divisor `n`.
//! The `M` nearest units of the other group are the matches. All units
//! within `distance_tolerance` of the `M`-th nearest distance are matches
//! as well, and each match of unit `i` counts `1 / |J(i)|` where `J(i)` is
//! the set of its matches. The set `J(i)` depends on the other group only, so
//! the same unit can be a match for many others. `K_j` is the sum of these
//! counts over the units `j` is a match for, and `KK_j` the sum of their
//! squares.
//!
//! The neighbours, for the matches and for the conditional variance, are
//! found on a k-d tree over each group (Friedman, Bentley and Finkel, 1977)
//! when the group is large enough and the covariates few enough for one to
//! pay: the `M`-th nearest distance first, then every unit within the
//! tolerance of it. That is an expected `O(log n)` per unit in few
//! covariates where a scan of the other group costs `O(n)`, and it finds the
//! units the scan finds, ties included, since the box bounds of the tree are
//! never above the rounded distances they bound. Smaller groups are scanned.
//!
//! # Estimate
//!
//! With `Y_i(0)` replaced by the mean outcome over `J(i)` for a treated unit
//! `i`, and `Y_i(1)` replaced the same way for a control, the estimate is
//! the mean over the target units of `Y_i(1) - Y_i(0)`. For the ATC the roles of
//! the groups are swapped and the sign of the difference reversed.
//!
//! With `bias_adjust`, the outcome of a match is first corrected for the
//! difference in covariates to its anchor (Abadie and Imbens \[2\]). The
//! outcome of a control `j` matched to unit `i` becomes
//! `Y_j + (Z_i - Z_j)' a_0`, with `a_0` the coefficients of the weighted
//! least squares regression of the outcome on an intercept and `Z` among the
//! controls, each weighted by `K_j`. For a treated match `a_1` is
//! obtained the same way among the treated, which the ATT does not need. `Z`
//! is the matching covariates unless others are given.
//!
//! # Variance
//!
//! With `N` target units, `sigma_i^2` the conditional variance of the outcome
//! of unit `i` given its covariates, `tau` the estimate and `D_i` the
//! difference `Y_i - Y_i(other)` of an anchor (adjusted when the estimate
//! is), the estimated variance of the population effect is
//!
//! ```text
//! V = (1 / N^2) ( sum_i sigma_i^2 c_i (K_i^2 + e K_i - KK_i)
//!                 + sum_{i in anchors} (D_i - tau)^2 )
//! ```
//!
//! where for the ATT and ATC the sum over `i` is over the units of the other
//! group and `c_i = 1`, `e = 0`, and for the ATE it is over all units, with
//! `e = 2`. The estimated variance of the sample effect is
//!
//! ```text
//! V_sample = (1 / N^2) sum_i sigma_i^2 (a_i + c_i K_i)^2
//! ```
//!
//! with `a_i` 1 for an anchor and 0 for a unit of the other group when
//! estimating an effect in one group, and 1 for every unit for the ATE. The
//! conditional part is `max(V_sample, V) - V_sample`. With bias adjustment
//! `D_i` is the adjusted difference. The standard error is the square root
//! of `V`. [`AbadieImbensEstimate::se_standard`] is the one that treats the
//! `D_i` as independent.
//!
//! `sigma_i^2` is estimated with `var_calc` matches. With `var_calc = 0` it
//! is the same for every unit, half the mean square of `D_i - tau`, which
//! is right under homoskedasticity. With `var_calc = J` it is the sample
//! variance of the outcome of unit `i` and of its `J` nearest neighbours in
//! its own treatment group, excluding itself, with ties at the `J`-th nearest
//! distance all included.
//!
//! # What is not here
//!
//! Matching without replacement and the estimator that matches on a
//! propensity score estimated in the same sample, whose variance needs the
//! adjustment of Abadie and Imbens \[3\], are not done: `Matching::Match`
//! gives no such adjustment either. A constant covariate, a singular
//! covariance in the Mahalanobis distance or a singular regression for the
//! bias adjustment is an error, where `Match` adds a small ridge and goes on.
//! `Match`'s calipers, exact matching, sample weights and common support are
//! not offered.
//!
//! # Validation
//!
//! Checked against `Matching::Match` 4.10.15 in
//! `tests/r_abadie_imbens.rs`, which records the estimate and the
//! standard errors for each target, `M`, bias adjustment, `Var.calc` and
//! distance weight on `lalonde` (with ties) and on a simulated cohort.
//!
//! # References
//!
//! 1. Abadie A, Imbens GW. Large sample properties of matching estimators for
//!    average treatment effects. Econometrica. 2006;74(1):235-67.
//! 2. Abadie A, Imbens GW. Bias-corrected matching estimators for average
//!    treatment effects. J Bus Econ Stat. 2011;29(1):1-11.
//! 3. Abadie A, Imbens GW. Matching on the estimated propensity score.
//!    Econometrica. 2016;84(2):781-807.

use crate::Error;

use epistat_linalg::{Attempt, Factor, Mat, Method, solve, weighted_gram, weighted_xtz};
use rayon::prelude::*;

use crate::Estimand;
use crate::effects::UnitData;
use crate::kdtree::{Euclidean, KdTree, worth_a_tree};

/// The distance the matches are found with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceWeight {
    /// Euclidean distance in the standardised covariates, `Weight = 1` in
    /// `Match`.
    #[default]
    Diagonal,
    /// Mahalanobis distance, `Weight = 2` in `Match`.
    Mahalanobis,
}

/// How the estimate and its error are made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbadieImbensSpec {
    /// The group the effect is for.
    pub target: Estimand,
    /// The number of matches per unit, `M` in `Match`.
    pub matches: usize,
    /// Whether to correct the outcome of a match for the covariate
    /// difference, `BiasAdjust` in `Match`.
    pub bias_adjust: bool,
    /// The number of neighbours for the conditional variances, `Var.calc` in
    /// `Match`. 0 gives the homoskedastic estimate.
    pub var_calc: usize,
    /// The distance, `Weight` in `Match`.
    pub weight: DistanceWeight,
    /// Distances within this of the `M`-th nearest also match, `distance.tolerance`
    /// in `Match`.
    pub distance_tolerance: f64,
}

impl AbadieImbensSpec {
    /// The defaults of `Match`: the ATT with one match, no bias adjustment,
    /// the homoskedastic variance and the standardised Euclidean distance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target: Estimand::Att,
            matches: 1,
            bias_adjust: false,
            var_calc: 0,
            weight: DistanceWeight::Diagonal,
            distance_tolerance: 1e-5,
        }
    }
}

impl Default for AbadieImbensSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// The units of the estimator: treatment, outcome and covariates.
#[derive(Debug, Clone, PartialEq)]
pub struct AbadieImbensData {
    treated: Vec<bool>,
    outcome: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    bias_covariates: Option<Vec<Vec<f64>>>,
}

fn check_columns(what: &'static str, columns: &[Vec<f64>], n: usize) -> Result<(), Error> {
    for column in columns {
        if column.len() != n {
            return Err(Error::length(what, n, column.len()));
        }
        if let Some(row) = column.iter().position(|v| !v.is_finite()) {
            return Err(Error::not_finite(what, row));
        }
    }
    Ok(())
}

impl AbadieImbensData {
    /// The units, with `covariates` one column per covariate.
    ///
    /// # Errors
    ///
    /// [`Error::Length`], [`Error::NotFinite`],
    /// [`Error::NoCovariates`] and [`Error::OneGroup`].
    pub fn new(
        treated: Vec<bool>,
        outcome: Vec<f64>,
        covariates: Vec<Vec<f64>>,
    ) -> Result<Self, Error> {
        let n = treated.len();
        if outcome.len() != n {
            return Err(Error::length("outcome", n, outcome.len()));
        }
        if let Some(row) = outcome.iter().position(|v| !v.is_finite()) {
            return Err(Error::not_finite("outcome", row));
        }
        if covariates.is_empty() {
            return Err(Error::NoCovariates);
        }
        check_columns("covariate", &covariates, n)?;
        let n_treated = treated.iter().filter(|t| **t).count();
        if n_treated == 0 || n_treated == n {
            return Err(Error::OneGroup);
        }
        Ok(Self {
            treated,
            outcome,
            covariates,
            bias_covariates: None,
        })
    }

    /// The units of a [`UnitData`], matched on its covariates.
    ///
    /// # Errors
    ///
    /// As [`AbadieImbensData::new`].
    pub fn from_unit_data(units: &UnitData) -> Result<Self, Error> {
        Self::new(
            units.treated_flags().to_vec(),
            units.outcomes().to_vec(),
            units.covariate_columns().to_vec(),
        )
    }

    /// The covariates the bias adjustment regresses on, where they are not
    /// the matching covariates.
    ///
    /// # Errors
    ///
    /// [`Error::Length`] and [`Error::NotFinite`].
    pub fn with_bias_covariates(mut self, columns: Vec<Vec<f64>>) -> Result<Self, Error> {
        check_columns("bias covariate", &columns, self.treated.len())?;
        self.bias_covariates = Some(columns);
        Ok(self)
    }

    /// The number of units.
    #[must_use]
    pub fn len(&self) -> usize {
        self.treated.len()
    }

    /// Whether there are no units.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.treated.is_empty()
    }
}

/// The estimate and its standard errors.
#[derive(Debug, Clone, PartialEq)]
pub struct AbadieImbensEstimate {
    /// The estimate, bias adjusted when the spec asks for it.
    pub estimate: f64,
    /// The estimate without the bias adjustment.
    pub estimate_unadjusted: f64,
    /// The standard error of the effect in the population, `Match`'s `se`.
    pub se: f64,
    /// The standard error of the effect in the sample, from the variance
    /// `Match` gives with `sample = TRUE`.
    pub se_sample: f64,
    /// The square root of the part of the population variance that the
    /// conditional variance does not account for, `Match`'s `se.cond`.
    pub se_conditional: f64,
    /// The standard error that takes the differences as independent, `Match`'s
    /// `se.standard`, of the unadjusted differences.
    pub se_standard: f64,
    /// The number of units the effect averages over.
    pub units: usize,
    /// How often each unit is a match, in the weights of `K_j` (zero for a
    /// unit that is no one's match).
    pub times_matched: Vec<f64>,
}

/// The estimate, bias correction and variance of the matching estimator.
///
/// # Errors
///
/// [`Error`] for the data and the spec that cannot be used.
pub fn abadie_imbens(
    data: &AbadieImbensData,
    spec: &AbadieImbensSpec,
) -> Result<AbadieImbensEstimate, Error> {
    Estimator::new(data, spec, true)?.run()
}

struct Estimator<'a> {
    data: &'a AbadieImbensData,
    spec: &'a AbadieImbensSpec,
    /// Whether the unit has the role of the treated in the estimate: the
    /// treated, or the controls for the ATC.
    role: Vec<bool>,
    /// Coordinates in which Euclidean distance is the matching distance,
    /// `k` per unit.
    coords: Vec<f64>,
    k: usize,
    /// The k-d tree over the units without and with the role of the
    /// treated, keyed by unit, where the group is worth one.
    trees: [Option<KdTree>; 2],
}

/// One anchor with its matches.
struct Anchor {
    unit: usize,
    matches: Vec<usize>,
}

impl<'a> Estimator<'a> {
    /// The estimator, with the neighbours found on k-d trees when
    /// `use_trees` is set and by a scan of the pool otherwise.
    fn new(
        data: &'a AbadieImbensData,
        spec: &'a AbadieImbensSpec,
        use_trees: bool,
    ) -> Result<Self, Error> {
        if spec.matches == 0 {
            return Err(Error::ZeroMatches);
        }
        if !(spec.distance_tolerance.is_finite() && spec.distance_tolerance >= 0.0) {
            return Err(Error::InvalidTolerance);
        }
        let n = data.len();
        let n_treated = data.treated.iter().filter(|t| **t).count();
        let n_control = n - n_treated;
        let role: Vec<bool> = match spec.target {
            Estimand::Atc => data.treated.iter().map(|t| !t).collect(),
            _ => data.treated.clone(),
        };
        let (n_role, n_other) = if matches!(spec.target, Estimand::Atc) {
            (n_control, n_treated)
        } else {
            (n_treated, n_control)
        };
        let available = if matches!(spec.target, Estimand::Ate) {
            n_role.min(n_other)
        } else {
            n_other
        };
        if spec.matches > available {
            return Err(Error::TooManyMatches {
                asked: spec.matches,
                available,
            });
        }
        if spec.var_calc > 0 && spec.var_calc >= n_treated.min(n_control) {
            return Err(Error::TooManyVarianceMatches {
                asked: spec.var_calc,
                group: n_treated.min(n_control),
            });
        }
        let (coords, k) = standardised_coordinates(&data.covariates, spec.weight)?;
        let trees = if use_trees && sums_are_finite(&coords, k) {
            [false, true].map(|with_role| {
                let keys: Vec<usize> = (0..n).filter(|&i| role[i] == with_role).collect();
                worth_a_tree(keys.len(), k).then(|| {
                    let mut points = Vec::with_capacity(keys.len() * k);
                    for &i in &keys {
                        points.extend_from_slice(&coords[i * k..(i + 1) * k]);
                    }
                    KdTree::new(k, &points, &keys, n)
                })
            })
        } else {
            [None, None]
        };
        Ok(Self {
            data,
            spec,
            role,
            coords,
            k,
            trees,
        })
    }

    fn distance(&self, i: usize, j: usize) -> f64 {
        let a = &self.coords[i * self.k..(i + 1) * self.k];
        let b = &self.coords[j * self.k..(j + 1) * self.k];
        let mut sum = 0.0;
        for (x, y) in a.iter().zip(b) {
            let d = x - y;
            sum += d * d;
        }
        sum.sqrt()
    }

    /// The `count` nearest of `pool` to `unit`, and every unit within the
    /// tolerance of the `count`-th, in the order of `pool`.
    fn nearest(&self, unit: usize, pool: &[usize], count: usize) -> Vec<usize> {
        let distances: Vec<f64> = pool.iter().map(|&j| self.distance(unit, j)).collect();
        let mut sorted = distances.clone();
        sorted.sort_by(f64::total_cmp);
        let limit = sorted[count - 1] + self.spec.distance_tolerance;
        pool.iter()
            .zip(&distances)
            .filter(|(_, d)| **d <= limit)
            .map(|(&j, _)| j)
            .collect()
    }

    /// As [`Self::nearest`] for the pool of the units whose role is
    /// `with_role`, less `unit` itself, on that group's tree when it has one.
    /// The tree gives the `count`-th smallest distance and then every unit
    /// within the tolerance of it, which are the units the scan keeps.
    fn nearest_in(&self, unit: usize, with_role: bool, pool: &[usize], count: usize) -> Vec<usize> {
        if let Some(tree) = &self.trees[usize::from(with_role)] {
            let query = &self.coords[unit * self.k..(unit + 1) * self.k];
            let found = tree.nearest(&Euclidean, query, count, None, &mut |j| j != unit);
            if let Some(&(last, _)) = found.get(count - 1) {
                let limit = last + self.spec.distance_tolerance;
                let mut near = tree.within(&Euclidean, query, limit);
                near.retain(|&j| j != unit);
                near.sort_unstable();
                return near;
            }
        }
        self.nearest(unit, pool, count)
    }

    fn run(&self) -> Result<AbadieImbensEstimate, Error> {
        let n = self.data.len();
        let all = matches!(self.spec.target, Estimand::Ate);
        let role_units: Vec<usize> = (0..n).filter(|&i| self.role[i]).collect();
        let other_units: Vec<usize> = (0..n).filter(|&i| !self.role[i]).collect();
        let anchor_units: Vec<usize> = if all {
            (0..n).collect()
        } else {
            role_units.clone()
        };
        let m = self.spec.matches;
        let anchors: Vec<Anchor> = anchor_units
            .par_iter()
            .map(|&unit| {
                let pool = if self.role[unit] {
                    &other_units
                } else {
                    &role_units
                };
                Anchor {
                    unit,
                    matches: self.nearest_in(unit, !self.role[unit], pool, m),
                }
            })
            .collect();

        // K_j and KK_j.
        let mut count_sum = vec![0.0; n];
        let mut count_squares = vec![0.0; n];
        for a in &anchors {
            let w = 1.0 / a.matches.len() as f64;
            for &j in &a.matches {
                count_sum[j] += w;
                count_squares[j] += w * w;
            }
        }

        // The difference of each row, in the direction of the role: an anchor
        // with the role of the treated gives its outcome minus its match's,
        // the other gives the reverse. With bias adjustment the match's
        // outcome moves by the covariate difference times the coefficients of
        // the match's group.
        let y = &self.data.outcome;
        let adjustment = if self.spec.bias_adjust {
            let z = self
                .data
                .bias_covariates
                .as_ref()
                .unwrap_or(&self.data.covariates);
            let alpha_other = self.regression(z, &count_sum, false)?;
            let alpha_role = if all {
                self.regression(z, &count_sum, true)?
            } else {
                vec![0.0; z.len()]
            };
            Some((z, alpha_other, alpha_role))
        } else {
            None
        };
        let mut raw_rows: Vec<Vec<f64>> = Vec::with_capacity(anchors.len());
        let mut adj_rows: Vec<Vec<f64>> = Vec::with_capacity(anchors.len());
        for a in &anchors {
            let i = a.unit;
            let sign = if self.role[i] { 1.0 } else { -1.0 };
            let mut raw = Vec::with_capacity(a.matches.len());
            let mut adj = Vec::with_capacity(a.matches.len());
            for &j in &a.matches {
                let d = sign * (y[i] - y[j]);
                let mut shift = 0.0;
                if let Some((z, alpha_other, alpha_role)) = &adjustment {
                    let alpha = if self.role[i] {
                        alpha_other
                    } else {
                        alpha_role
                    };
                    for (c, col) in z.iter().enumerate() {
                        shift += (col[i] - col[j]) * alpha[c];
                    }
                }
                raw.push(d);
                // A match's outcome plus `shift` is its adjusted outcome.
                adj.push(d - sign * shift);
            }
            raw_rows.push(raw);
            adj_rows.push(adj);
        }
        let n_anchor = anchors.len() as f64;
        let mean_of = |rows: &[f64]| sum(rows) / rows.len() as f64;
        let raw_diff: Vec<f64> = raw_rows.iter().map(|r| mean_of(r)).collect();
        let adj_diff: Vec<f64> = adj_rows.iter().map(|r| mean_of(r)).collect();
        let est_raw = sum(&raw_diff) / n_anchor;
        let est = sum(&adj_diff) / n_anchor;

        // The mean square of the row differences about `centre`, each row
        // weighted by one over the matches of its anchor.
        let row_square = |rows: &[Vec<f64>], centre: f64| -> f64 {
            let mut total = 0.0;
            for r in rows {
                let w = 1.0 / r.len() as f64;
                for d in r {
                    total += w * (d - centre) * (d - centre);
                }
            }
            total
        };

        // Conditional variances.
        let sigma: Vec<f64> = if self.spec.var_calc == 0 {
            vec![0.5 * row_square(&adj_rows, est) / n_anchor; n]
        } else {
            self.neighbour_variances()
        };

        // Variances.
        let n2 = n_anchor * n_anchor;
        let mut v_sample = 0.0;
        let mut v_pop = 0.0;
        for i in 0..n {
            let is_anchor = all || self.role[i];
            let a_i = f64::from(u8::from(is_anchor));
            let c_i = if all {
                1.0
            } else {
                f64::from(u8::from(!self.role[i]))
            };
            let s = a_i + c_i * count_sum[i];
            v_sample += sigma[i] * s * s;
            let extra = if all { 2.0 * count_sum[i] } else { 0.0 };
            v_pop += sigma[i] * c_i * (count_sum[i] * count_sum[i] + extra - count_squares[i]);
        }
        let mut d_var = 0.0;
        for d in &adj_diff {
            d_var += (d - est) * (d - est);
        }
        let v_sample = v_sample / n2;
        let v_pop = (v_pop + d_var) / n2;
        let v_cond = v_pop.max(v_sample) - v_sample;

        let se_standard = (row_square(&raw_rows, est_raw) / n2).sqrt();
        let sign = if matches!(self.spec.target, Estimand::Atc) {
            -1.0
        } else {
            1.0
        };
        Ok(AbadieImbensEstimate {
            estimate: sign * est,
            estimate_unadjusted: sign * est_raw,
            se: v_pop.sqrt(),
            se_sample: v_sample.sqrt(),
            se_conditional: v_cond.sqrt(),
            se_standard,
            units: anchors.len(),
            times_matched: count_sum,
        })
    }

    /// The sample variance of each unit's outcome with its `var_calc`
    /// nearest neighbours in its own group, ties at the last all in.
    fn neighbour_variances(&self) -> Vec<f64> {
        let n = self.data.len();
        let treated = &self.data.treated;
        let groups: [Vec<usize>; 2] = [
            (0..n).filter(|&i| !treated[i]).collect(),
            (0..n).filter(|&i| treated[i]).collect(),
        ];
        let y = &self.data.outcome;
        let j = self.spec.var_calc;
        (0..n)
            .into_par_iter()
            .map(|i| {
                let group = &groups[usize::from(treated[i])];
                let near = if self.trees[usize::from(self.role[i])].is_some() {
                    self.nearest_in(i, self.role[i], &[], j)
                } else {
                    let pool: Vec<usize> = group.iter().copied().filter(|&u| u != i).collect();
                    self.nearest(i, &pool, j)
                };
                let count = near.len() + 1;
                let mut total = y[i];
                for &u in &near {
                    total += y[u];
                }
                let mean = total / count as f64;
                let mut ss = (y[i] - mean) * (y[i] - mean);
                for &u in &near {
                    ss += (y[u] - mean) * (y[u] - mean);
                }
                ss / (count - 1) as f64
            })
            .collect()
    }

    /// The coefficients of the regression of the outcome on `z` with an
    /// intercept, over the units with the given role, each weighted by its
    /// match count. The intercept is dropped.
    fn regression(&self, z: &[Vec<f64>], weight: &[f64], role: bool) -> Result<Vec<f64>, Error> {
        let rows: Vec<usize> = (0..self.data.len())
            .filter(|&i| self.role[i] == role && weight[i] > 0.0)
            .collect();
        let p = z.len() + 1;
        if rows.len() < p {
            return Err(Error::SingularRegression);
        }
        let x = Mat::from_fn(
            rows.len(),
            p,
            |r, c| if c == 0 { 1.0 } else { z[c - 1][rows[r]] },
        );
        let w: Vec<f64> = rows.iter().map(|&i| weight[i]).collect();
        let yv: Vec<f64> = rows.iter().map(|&i| self.data.outcome[i]).collect();
        let gram = weighted_gram(x.as_ref(), &w).map_err(|_| Error::SingularRegression)?;
        let rhs = weighted_xtz(x.as_ref(), &w, &yv).map_err(|_| Error::SingularRegression)?;
        let largest = (0..p).map(|c| gram[(c, c)]).fold(0.0_f64, f64::max);
        // The threshold applies to the diagonal of the Cholesky factor, whose
        // square is a pivot of the Gram matrix. A pivot at most machine
        // epsilon times the largest diagonal entry is taken as zero, so the
        // test does not depend on the scale of the covariates.
        let attempt =
            Attempt::plain(Method::Cholesky).with_min_pivot(f64::EPSILON.sqrt() * largest.sqrt());
        let solution = solve(gram.as_ref(), rhs.as_ref(), &[attempt])
            .map_err(|_| Error::SingularRegression)?;
        Ok((1..p).map(|c| solution.x[(c, 0)]).collect())
    }
}

/// Whether every sum of squared differences of two points of `coords`, `k`
/// coordinates each, is finite. None exceeds the sum over the coordinates of
/// the squared range, the largest value less the smallest, in floating point
/// too.
fn sums_are_finite(coords: &[f64], k: usize) -> bool {
    if coords.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let mut largest = 0.0_f64;
    for c in 0..k {
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for value in coords.iter().skip(c).step_by(k) {
            low = low.min(*value);
            high = high.max(*value);
        }
        let range = high - low;
        largest += range * range;
    }
    largest.is_finite()
}

fn sum(values: &[f64]) -> f64 {
    let mut s = 0.0;
    for v in values {
        s += v;
    }
    s
}

/// The covariates standardised by the sample standard deviation, and for the
/// Mahalanobis distance transformed so that Euclidean distance in the result
/// is the Mahalanobis distance.
fn standardised_coordinates(
    covariates: &[Vec<f64>],
    weight: DistanceWeight,
) -> Result<(Vec<f64>, usize), Error> {
    let k = covariates.len();
    let n = covariates[0].len();
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    for (c, column) in covariates.iter().enumerate() {
        let mean = sum(column) / n as f64;
        let mut ss = 0.0;
        for v in column {
            ss += (v - mean) * (v - mean);
        }
        let sd = (ss / (n - 1) as f64).sqrt();
        if !(sd > 0.0 && sd.is_finite()) {
            return Err(Error::ConstantCovariate(c));
        }
        cols.push(column.iter().map(|v| (v - mean) / sd).collect());
    }
    let mut coords = vec![0.0; n * k];
    match weight {
        DistanceWeight::Diagonal => {
            for (c, col) in cols.iter().enumerate() {
                for (i, v) in col.iter().enumerate() {
                    coords[i * k + c] = *v;
                }
            }
        }
        DistanceWeight::Mahalanobis => {
            // The inverse of G / n, with G = L L', is n G^-1. A point x has
            // distance `sqrt(n) |L^-1 x|` from the origin.
            let gram = Mat::from_fn(k, k, |a, b| {
                let mut s = 0.0;
                for (u, v) in cols[a].iter().zip(&cols[b]) {
                    s += u * v;
                }
                s
            });
            let largest = (0..k).map(|c| gram[(c, c)]).fold(0.0_f64, f64::max);
            // As for the regression: a pivot of the Gram matrix at most
            // machine epsilon times its largest diagonal entry is zero.
            let attempt = Attempt::plain(Method::Cholesky)
                .with_min_pivot(f64::EPSILON.sqrt() * largest.sqrt());
            let factor =
                Factor::new(gram.as_ref(), &[attempt]).map_err(|_| Error::SingularCovariance)?;
            let l = factor.cholesky_factor().ok_or(Error::SingularCovariance)?;
            let root_n = (n as f64).sqrt();
            for i in 0..n {
                // Forward substitution L u = x_i.
                let mut u = vec![0.0; k];
                for a in 0..k {
                    let mut s = cols[a][i];
                    for b in 0..a {
                        s -= l[(a, b)] * u[b];
                    }
                    u[a] = s / l[(a, a)];
                }
                for a in 0..k {
                    coords[i * k + a] = root_n * u[a];
                }
            }
        }
    }
    Ok((coords, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> AbadieImbensData {
        // One covariate. Treated at 0 (outcome 1) and 10 (outcome 6);
        // controls at -1 (0), 2 (3) and 9 (4).
        AbadieImbensData::new(
            vec![true, true, false, false, false],
            vec![1.0, 6.0, 0.0, 3.0, 4.0],
            vec![vec![0.0, 10.0, -1.0, 2.0, 9.0]],
        )
        .unwrap()
    }

    #[test]
    fn att_with_one_match_is_the_mean_difference_to_the_nearest_control() {
        // Treated 0 matches -1 (difference 1), treated 10 matches 9 (2).
        let e = abadie_imbens(&tiny(), &AbadieImbensSpec::new()).unwrap();
        assert!((e.estimate - 1.5).abs() < 1e-14);
        assert_eq!(e.units, 2);
        assert_eq!(e.times_matched, vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn atc_matches_the_controls_to_the_treated() {
        // Control -1 matches 0 (difference 0 - 1 = -1 for treated minus
        // control, so the effect is 1 - 0 = 1), 2 matches 0 (1 - 3 = -2) and
        // 9 matches 10 (6 - 4 = 2). The mean is 1/3.
        let spec = AbadieImbensSpec {
            target: Estimand::Atc,
            ..AbadieImbensSpec::new()
        };
        let e = abadie_imbens(&tiny(), &spec).unwrap();
        assert!((e.estimate - 1.0 / 3.0).abs() < 1e-14);
    }

    /// The k-d trees give the estimate of the scan of every unit, to the bit:
    /// on covariates with few values, so that many distances tie and the
    /// tolerance takes in several matches, for the three targets, one and
    /// three matches, the conditional variance from two neighbours, bias
    /// adjustment and both distances.
    #[test]
    fn the_trees_give_the_estimate_of_the_scan() {
        let n = 3_600;
        let mut rng = epistat_testkit::SplitMix64::new(7);
        let mut next = move || rng.next_u64();
        let treated: Vec<bool> = (0..n).map(|_| next() % 3 == 0).collect();
        let covariates: Vec<Vec<f64>> = (0..3)
            .map(|c| {
                (0..n)
                    .map(|_| (next() % if c == 2 { 2 } else { 6 }) as f64)
                    .collect()
            })
            .collect();
        let outcome: Vec<f64> = (0..n)
            .map(|i| (next() % 100) as f64 / 10.0 + covariates[0][i])
            .collect();
        let data = AbadieImbensData::new(treated, outcome, covariates).unwrap();
        for target in [Estimand::Att, Estimand::Atc, Estimand::Ate] {
            for (matches, var_calc, bias_adjust, weight) in [
                (1, 0, false, DistanceWeight::Diagonal),
                (3, 2, true, DistanceWeight::Diagonal),
                (1, 2, true, DistanceWeight::Mahalanobis),
            ] {
                let spec = AbadieImbensSpec {
                    target,
                    matches,
                    bias_adjust,
                    var_calc,
                    weight,
                    ..AbadieImbensSpec::new()
                };
                let on_trees = Estimator::new(&data, &spec, true).unwrap();
                assert!(on_trees.trees.iter().all(Option::is_some));
                let by_scan = Estimator::new(&data, &spec, false).unwrap();
                assert!(by_scan.trees.iter().all(Option::is_none));
                assert_eq!(
                    format!("{:?}", on_trees.run()),
                    format!("{:?}", by_scan.run()),
                    "{target:?}, {matches} matches, var_calc {var_calc}, {weight:?}"
                );
            }
        }
    }

    #[test]
    fn invalid_input_is_an_error() {
        let spec = AbadieImbensSpec {
            matches: 4,
            ..AbadieImbensSpec::new()
        };
        assert!(matches!(
            abadie_imbens(&tiny(), &spec),
            Err(Error::TooManyMatches { .. })
        ));
        let spec = AbadieImbensSpec {
            matches: 0,
            ..AbadieImbensSpec::new()
        };
        assert_eq!(abadie_imbens(&tiny(), &spec), Err(Error::ZeroMatches));
        let constant = AbadieImbensData::new(
            vec![true, false, false],
            vec![1.0, 2.0, 3.0],
            vec![vec![1.0, 1.0, 1.0]],
        )
        .unwrap();
        assert_eq!(
            abadie_imbens(&constant, &AbadieImbensSpec::new()),
            Err(Error::ConstantCovariate(0))
        );
        assert!(
            AbadieImbensData::new(vec![true, true], vec![1.0, 2.0], vec![vec![0.0, 1.0]]).is_err()
        );
    }
}
