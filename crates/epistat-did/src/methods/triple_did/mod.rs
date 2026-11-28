//! Doubly robust triple differences, after Ortiz-Villavicencio and
//! Sant'Anna \[1\], as R's `triplediff::ddd` estimates them with two periods.
//!
//! [`estimate`] takes a [`TripleDidData`], the group `S` (whether the unit is
//! in the group that is treated, such as an enabled region), the partition
//! `Q` (whether it is eligible, such as an eligible age group), the outcome
//! change and optional weights, covariates and clusters, and returns a
//! [`TripleDid`].
//!
//! # Algorithm
//!
//! The units fall in four subgroups, numbered as `triplediff` numbers them:
//! 4 is `S = 1, Q = 1`, the focal group; 3 is `S = 1, Q = 0`; 2 is
//! `S = 0, Q = 1`; 1 is `S = 0, Q = 0`. Then
//!
//! ```text
//! ATT_DDD = ATT(4 vs 3) + ATT(4 vs 2) - ATT(4 vs 1),
//! ```
//!
//! each term the panel doubly robust estimate of
//! [`two_period`](crate::two_period) on subgroup 4 against one comparator
//! subgroup, with the intercept added to the covariates. The influence
//! function is the signed sum of the three, each scaled by `n / n_k` for the
//! `n_k` units of its comparison, as `triplediff::att_dr` scales them. The
//! standard error, clustering and bootstrap interval are those of
//! [`two_period`](crate::two_period).
//!
//! # References
//!
//! 1. Ortiz-Villavicencio M, Sant'Anna PHC. Better understanding triple
//!    differences estimators. Working paper, 2025.

use epistat_core::{Bootstrap, CaseWeights, Estimate, InputError, Level, check};
use epistat_inference::{Clusters, Influence};
use epistat_linalg::MatRef;

use crate::error::Error;
use crate::inference::did_clusters::standard_error;
use crate::inference::{estimate_from, normal_interval, standard_error_from_influence};
use crate::methods::drdid::panel::estimate_drdid_panel;
use crate::methods::two_period::{NuisanceOptions, domain, row_influence};
use crate::types::DrDidObservation;

/// A validated triple-difference sample, one unit per row.
///
/// [`TripleDidData::builder`] takes the group, the partition and the outcome
/// change of each unit; [`TripleDidDataBuilder::build`] checks that every
/// column has one entry per unit, at least one unit, that the changes and
/// covariates are finite and that the weights are finite and positive. The
/// covariates are given without an intercept, which the estimator adds.
#[derive(Clone, Copy, Debug)]
pub struct TripleDidData<'a> {
    group: &'a [bool],
    partition: &'a [bool],
    change: &'a [f64],
    weights: Option<&'a [f64]>,
    covariates: Option<MatRef<'a, f64>>,
    clusters: Option<&'a [u32]>,
}

/// The builder of a [`TripleDidData`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct TripleDidDataBuilder<'a> {
    data: TripleDidData<'a>,
}

impl<'a> TripleDidData<'a> {
    /// The data of the group `S`, the partition `Q` and the outcome change of
    /// each unit.
    pub const fn builder(
        group: &'a [bool],
        partition: &'a [bool],
        change: &'a [f64],
    ) -> TripleDidDataBuilder<'a> {
        TripleDidDataBuilder {
            data: Self {
                group,
                partition,
                change,
                weights: None,
                covariates: None,
                clusters: None,
            },
        }
    }

    /// Whether each unit is in the group that is treated, `S`.
    #[must_use]
    pub const fn group(&self) -> &'a [bool] {
        self.group
    }

    /// Whether each unit is in the eligible partition, `Q`.
    #[must_use]
    pub const fn partition(&self) -> &'a [bool] {
        self.partition
    }

    /// The outcome change of each unit.
    #[must_use]
    pub const fn change(&self) -> &'a [f64] {
        self.change
    }

    /// The number of units.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.change.len()
    }

    /// Whether there are no units, which the builder refuses, so always false.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.change.is_empty()
    }

    /// The subgroup of unit `i`, numbered as `triplediff` numbers them.
    const fn subgroup(&self, i: usize) -> u8 {
        match (self.group[i], self.partition[i]) {
            (true, true) => 4,
            (true, false) => 3,
            (false, true) => 2,
            (false, false) => 1,
        }
    }

    /// The design row of unit `i`: the intercept, then the covariates.
    fn design_row(&self, i: usize) -> Vec<f64> {
        let mut row = vec![1.0];
        if let Some(x) = self.covariates {
            row.extend((0..x.ncols()).map(|j| x[(i, j)]));
        }
        row
    }
}

impl<'a> TripleDidDataBuilder<'a> {
    /// Prior weights, one per unit.
    pub const fn weights(mut self, weights: CaseWeights<'a>) -> Self {
        self.data.weights = Some(weights.as_slice());
        self
    }

    /// Covariates, one row per unit, without an intercept.
    pub const fn covariates(mut self, covariates: MatRef<'a, f64>) -> Self {
        self.data.covariates = Some(covariates);
        self
    }

    /// Cluster codes, one per unit.
    pub const fn clusters(mut self, clusters: &'a [u32]) -> Self {
        self.data.clusters = Some(clusters);
        self
    }

    /// The validated data.
    ///
    /// # Errors
    ///
    /// [`Error::Input`]: [`InputError::Empty`] without units,
    /// [`InputError::TooLarge`] for more units than `u32` numbers,
    /// [`InputError::Length`] for a column of the wrong length,
    /// [`InputError::NotFinite`] for a change, weight or covariate that is not
    /// finite and [`InputError::Domain`] for a weight that is not positive.
    pub fn build(self) -> Result<TripleDidData<'a>, Error> {
        let data = self.data;
        let n = data.change.len();
        if n == 0 {
            return Err(InputError::Empty { what: "change" }.into());
        }
        if u32::try_from(n).is_err() {
            return Err(InputError::TooLarge {
                what: "units of a triple difference",
                limit: u64::from(u32::MAX),
                found: u64::try_from(n).unwrap_or(u64::MAX),
            }
            .into());
        }
        check::finite("change", data.change)?;
        check::same_length("group", n, data.group.len())?;
        check::same_length("partition", n, data.partition.len())?;
        if let Some(weights) = data.weights {
            check::same_length("weights", n, weights.len())?;
            check::finite("weights", weights)?;
            if let Some(&value) = weights.iter().find(|&&w| w <= 0.0) {
                return Err(domain("weights", value, "(0, inf)"));
            }
        }
        if let Some(x) = data.covariates {
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
        if let Some(clusters) = data.clusters {
            check::same_length("clusters", n, clusters.len())?;
        }
        Ok(data)
    }
}

/// The options of [`estimate`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TripleDidOptions {
    /// The nuisance models of the three doubly robust comparisons.
    pub nuisance: NuisanceOptions,
    /// The confidence level of the interval.
    pub level: Level,
    /// A multiplier bootstrap for the interval, as
    /// [`TwoPeriodOptions::bootstrap`](crate::two_period::TwoPeriodOptions::bootstrap);
    /// `None`, the default, gives the normal interval. Not with clusters.
    pub bootstrap: Option<Bootstrap>,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

/// A triple-difference ATT.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct TripleDid {
    /// The ATT of the focal group `S = 1, Q = 1` against the three comparison
    /// groups, with its standard error, interval and normal p-value.
    pub estimate: Estimate,
    /// The influence function, one value per unit, under
    /// [`epistat_inference::ClusterRule::NoSmallSample`].
    pub influence: Influence,
}

/// The doubly robust triple-difference ATT of `data`.
///
/// # Errors
///
/// [`Error::Input`] for a nuisance option outside its domain or a bootstrap
/// without replicates; [`Error::Unsupported`] for a bootstrap with clusters;
/// and the errors of the three panel doubly robust fits, as
/// [`two_period::estimate`](crate::two_period::estimate): [`Error::NoTreated`]
/// without a focal unit, [`Error::NoComparison`] for an empty comparison
/// subgroup and [`Error::Singular`] for a nuisance model that cannot be
/// fitted.
pub fn estimate(data: &TripleDidData<'_>, options: &TripleDidOptions) -> Result<TripleDid, Error> {
    options.nuisance.validate()?;
    if let Some(bootstrap) = options.bootstrap {
        if bootstrap.replicates == 0 {
            return Err(domain("bootstrap replicates", 0.0, "[1, inf)"));
        }
        if data.clusters.is_some() {
            return Err(Error::Unsupported {
                what: "a triple-difference bootstrap with clusters",
            });
        }
    }
    let config = options.nuisance.config(options.level, options.bootstrap);
    let n = data.len();
    let full_sample_n = crate::util::usize_to_f64(n);

    let mut att_ddd = 0.0;
    let mut influence_function = vec![0.0; n];

    for (comparison_subgroup, sign) in [(3_u8, 1.0_f64), (2_u8, 1.0_f64), (1_u8, -1.0_f64)] {
        let selected_indices = (0..n)
            .filter(|&i| {
                let subgroup = data.subgroup(i);
                subgroup == 4 || subgroup == comparison_subgroup
            })
            .collect::<Vec<_>>();
        if selected_indices.is_empty() {
            return Err(Error::NoComparison);
        }

        let comparison_n = crate::util::usize_to_f64(selected_indices.len());
        let cell_obs = selected_indices
            .iter()
            .map(|&idx| DrDidObservation {
                treated: data.subgroup(idx) == 4,
                delta_outcome: data.change[idx],
                covariates: data.design_row(idx),
                weight: data.weights.map_or(1.0, |w| w[idx]),
            })
            .collect::<Vec<_>>();

        let res = estimate_drdid_panel(&cell_obs, config)?;

        let scale = full_sample_n / comparison_n;
        att_ddd = sign.mul_add(res.att, att_ddd);
        for (local_idx, &global_idx) in selected_indices.iter().enumerate() {
            influence_function[global_idx] = (sign * scale).mul_add(
                res.influence_function[local_idx],
                influence_function[global_idx],
            );
        }
    }

    let (se, ci) = if let Some(codes) = data.clusters {
        let se = standard_error(&influence_function, &Clusters::from_codes(codes));
        (
            se,
            normal_interval(att_ddd, se, options.level.confidence_value()),
        )
    } else {
        let se = standard_error_from_influence(&influence_function);
        let ci =
            crate::methods::drdid::confidence_interval(att_ddd, se, &influence_function, config);
        (se, ci)
    };
    Ok(TripleDid {
        estimate: estimate_from(att_ddd, se, ci, options.level),
        influence: row_influence(influence_function)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_comparison_subgroup_is_refused() {
        let group = [true, false, false];
        let partition = [true, true, false];
        let change = [10.0, 2.0, 3.0];
        let data = TripleDidData::builder(&group, &partition, &change)
            .build()
            .unwrap();
        assert_eq!(
            estimate(&data, &TripleDidOptions::default()).unwrap_err(),
            Error::NoComparison
        );
    }

    #[test]
    fn subgroups_follow_the_triplediff_layout() {
        let group = [true, true, false, false];
        let partition = [true, false, true, false];
        let change = [0.0; 4];
        let data = TripleDidData::builder(&group, &partition, &change)
            .build()
            .unwrap();
        let codes: Vec<u8> = (0..4).map(|i| data.subgroup(i)).collect();
        assert_eq!(codes, [4, 3, 2, 1]);
    }
}
