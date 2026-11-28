//! Matching after multiple imputation, and the pooled effect.
//!
//! The caller imputes the missing values and gives the `m` completed data
//! sets. Nothing here imputes. Each set is an [`ImputedSet`], the cases and
//! the controls with their imputed covariates, with the same ids in every
//! set. Two approaches to matching follow the documentation of
//! `MatchThem::matchthem()` \[1\] and Leyrat et al. \[2\]:
//!
//! * [`ImputationApproach::Within`]: every set is matched on a propensity
//!   score fitted in that set, as [`estimate_propensity_and_match`] does for
//!   one set.
//! * [`ImputationApproach::Across`]: the propensity score is fitted in every
//!   set, the `m` scores of a unit are averaged on the scale the matching uses
//!   (the probability by default, or the linear predictor as set in the
//!   [`PropensityScoreConfig`]), and every set is matched on the averaged
//!   score. The averaged scores are in the `distance_scores` of each returned
//!   estimate. The sets must hold the same units, and `reestimate` is not
//!   allowed.
//!
//! [`estimate_pooled`] then estimates the effect in every matched set with
//! [`estimate_effects`], with the outcome and covariates of that set, and
//! pools the effects over the sets by Rubin's rules with the Barnard-Rubin
//! degrees of freedom ([`epistat_inference::pooling`]). Each set's variance is
//! the squared standard error of its effect, cluster-robust as the
//! [`EffectSpec`] says. The caller chooses the complete-data degrees of freedom
//! with a [`CompleteDataDf`]. The usual rule, [`CompleteDataDf::Residual`], is
//! the smallest, over the sets, of the number of rows of the fit less the
//! number of coefficients, which is what `mice::pool()` takes from the first
//! fit of sets of equal size.
//!
//! A ratio or an odds ratio is pooled on the log scale, so [`PooledEffects`]
//! holds the pooled log ratio and log odds ratio. Rubin's rules assume that
//! the estimate is close to normal.
//!
//! The matching and the effects are checked against `MatchThem` 1.2.1 and
//! `mice` 3.19.0 on data sets imputed once in R. The pooling is checked
//! against `mice::pool()` in `epistat-inference`.
//!
//! # References
//!
//! 1. Pishgar F, Greifer N, Leyrat C, Stuart E. MatchThem: matching and
//!    weighting after multiple imputation. R J. 2021;13(2):292-305.
//! 2. Leyrat C, Seaman SR, White IR, Douglas I, Smeeth L, Kim J, et al.
//!    Propensity score analysis with partially observed covariates: how
//!    should multiple imputation be used? Stat Methods Med Res.
//!    2019;28(1):3-19.
//! 3. Rubin DB. Multiple Imputation for Nonresponse in Surveys. New York:
//!    Wiley; 1987.

use crate::Error;
use std::sync::Arc;

use rayon::prelude::*;

use epistat_inference::pooling::{Pooled, dfcom_residual, rubin};

use crate::constraints::ConstraintGroup;
use crate::effects::{
    Effect, EffectEstimates, EffectSpec, MatchedSample, UnitData, estimate_effects,
};
use crate::matching::{
    CovariateRecord, DistanceConfig, MatchingRecord, PropensityMatchedOutcome,
    PropensityScoreConfig, ScoreTable, SelectionStrategy, StandardMatchRequest,
    estimate_propensity_and_match, match_standard, prepare_propensity_distance_config,
};
use crate::output::{MatchDataOptions, Units, match_data};
use crate::types::{MatchRatio, MatchingCriteria};

/// How the imputed data sets are combined in matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImputationApproach {
    /// Match each data set on its own propensity score.
    Within,
    /// Match each data set on the propensity score averaged over the sets.
    Across,
}

/// One completed data set: the cases and the controls with imputed
/// covariates.
#[derive(Debug)]
pub struct ImputedSet<'a, R> {
    /// The cases, the treated units for the effect in the treated.
    pub cases: &'a [R],
    /// The controls.
    pub controls: &'a [R],
}

impl<R> Clone for ImputedSet<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> Copy for ImputedSet<'_, R> {}

/// The matches of the imputed data sets.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ImputedMatches {
    /// The approach used.
    pub approach: ImputationApproach,
    /// The match of each data set, in the order given.
    pub sets: Vec<PropensityMatchedOutcome>,
}

/// Matches every imputed data set.
///
/// `criteria`, `strategy`, `ratio_fallback`, `extra_constraints` and
/// `config` are those of [`estimate_propensity_and_match`] and apply to every
/// set. With [`ImputationApproach::Across`] the matching trims to common
/// support on the averaged score, as `criteria.common_support` says.
///
/// # Errors
///
/// See [`Error`].
pub fn match_imputed<R, S, G>(
    sets: &[ImputedSet<'_, R>],
    approach: ImputationApproach,
    criteria: &MatchingCriteria,
    strategy: &S,
    ratio_fallback: &[MatchRatio],
    extra_constraints: &G,
    config: &PropensityScoreConfig,
) -> Result<ImputedMatches, Error>
where
    R: MatchingRecord + CovariateRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
{
    if sets.len() < 2 {
        return Err(Error::TooFewSets { found: sets.len() });
    }
    match approach {
        ImputationApproach::Within => {
            let matched = in_set_order(
                sets.par_iter()
                    .map(|set| {
                        estimate_propensity_and_match(
                            set.cases,
                            set.controls,
                            criteria,
                            strategy.clone(),
                            ratio_fallback,
                            extra_constraints,
                            config,
                        )
                    })
                    .collect(),
            )?;
            Ok(ImputedMatches {
                approach,
                sets: matched,
            })
        }
        ImputationApproach::Across => {
            if config.reestimate {
                return Err(Error::ReestimateAcross);
            }
            let estimates = in_set_order(
                sets.par_iter()
                    .map(|set| {
                        prepare_propensity_distance_config(set.cases, set.controls, config, None)
                            .map(|prepared| prepared.estimate)
                    })
                    .collect(),
            )?;
            let averaged = average_scores(&estimates, sets)?;
            let distance_config =
                DistanceConfig::propensity_score_table(averaged.clone(), config.caliper);
            let matched = sets
                .par_iter()
                .zip(estimates)
                .map(|(set, mut estimate)| {
                    let request = StandardMatchRequest {
                        criteria,
                        strategy: strategy.clone(),
                        constraints: extra_constraints,
                        ratio_fallback,
                        distance_config: Some(&distance_config),
                    };
                    let outcome = match_standard(set.cases, set.controls, request);
                    estimate.set_distance_scores(&averaged);
                    PropensityMatchedOutcome {
                        outcome,
                        propensity: estimate,
                    }
                })
                .collect();
            Ok(ImputedMatches {
                approach,
                sets: matched,
            })
        }
    }
}

/// The mean over the sets of each unit's score, summed in the order of the
/// sets, for the units of the first set in its order.
fn average_scores<R: MatchingRecord>(
    estimates: &[crate::PropensityScoreEstimate],
    sets: &[ImputedSet<'_, R>],
) -> Result<ScoreTable, Error> {
    let first = &estimates[0];
    let mut total: Vec<f64> = first.distance_scores.to_vec();
    for (index, estimate) in estimates.iter().enumerate().skip(1) {
        let same_order = estimate.ids == first.ids;
        if estimate.distance_scores.len() != first.distance_scores.len() {
            let id = sets[index]
                .cases
                .iter()
                .chain(sets[index].controls)
                .map(MatchingRecord::id)
                .find(|id| first.position(id).is_none())
                .unwrap_or("")
                .to_string();
            return Err(Error::UnitsDiffer { set: index, id });
        }
        for (position, id) in estimate.ids.iter().enumerate() {
            let at = if same_order {
                Some(position)
            } else {
                first.position(id)
            };
            match at {
                Some(at) => total[at] += estimate.distance_scores[position],
                None => {
                    return Err(Error::UnitsDiffer {
                        set: index,
                        id: id.clone(),
                    });
                }
            }
        }
    }
    let m = estimates.len() as f64;
    for value in &mut total {
        *value /= m;
    }
    Ok(ScoreTable::new(Arc::clone(&first.ids), total.into()))
}

/// The complete-data degrees of freedom that the Barnard-Rubin correction
/// of the pooled degrees of freedom uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompleteDataDf {
    /// The smallest over the sets of the rows of the fit less the number of
    /// coefficients, at least 1 ([`dfcom_residual`]).
    Residual,
    /// The smallest over the sets of the rows of the fit less the number of
    /// coefficients, not limited below. A set with no more rows than
    /// coefficients makes the pooling fail with
    /// [`Error::Inference`].
    RowsLessCoefficients,
    /// A value of the caller's, for a model whose information is not in its
    /// rows (the events of a Cox model, see
    /// [`epistat_inference::pooling::dfcom_events`]), or `None` for infinite
    /// degrees of freedom.
    Given(Option<f64>),
}

impl CompleteDataDf {
    fn value(self, estimates: &[EffectEstimates]) -> Option<f64> {
        let smallest = |df: fn(&EffectEstimates) -> f64| {
            estimates.iter().map(df).fold(f64::INFINITY, f64::min)
        };
        match self {
            Self::Residual => Some(smallest(|e| dfcom_residual(e.rows, e.coefficients.len()))),
            Self::RowsLessCoefficients => {
                Some(smallest(|e| e.rows as f64 - e.coefficients.len() as f64))
            }
            Self::Given(df) => df,
        }
    }
}

/// The effects of a treatment pooled over the imputed data sets.
///
/// An effect is `None` when a data set gives no standard error for it, or the
/// effect does not exist for the model, as in [`EffectEstimates`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PooledEffects {
    /// The difference of the average predictions.
    pub difference: Option<Pooled>,
    /// The ratio of the average predictions. Prefer [`Self::log_ratio`].
    pub ratio: Option<Pooled>,
    /// The log of the ratio.
    pub log_ratio: Option<Pooled>,
    /// The odds ratio. Prefer [`Self::log_odds_ratio`].
    pub odds_ratio: Option<Pooled>,
    /// The log of the odds ratio.
    pub log_odds_ratio: Option<Pooled>,
    /// The complete-data degrees of freedom given to the pooling, `None` for
    /// infinite.
    pub complete_data_df: Option<f64>,
}

/// The effects of every data set and their pooled value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ImputedEffects {
    /// The estimates of each data set, in the order given.
    pub per_set: Vec<EffectEstimates>,
    /// The pooled effects.
    pub pooled: PooledEffects,
}

/// Pools the effects of the data sets by Rubin's rules, with the
/// complete-data degrees of freedom `df`.
///
/// # Errors
///
/// [`Error::TooFewSets`] for fewer than two sets and
/// [`Error::Inference`] when the pooling fails, for instance when
/// `df` is [`CompleteDataDf::RowsLessCoefficients`] and a data set has no more
/// rows than coefficients.
pub fn pool_effects(
    estimates: &[EffectEstimates],
    df: CompleteDataDf,
) -> Result<PooledEffects, Error> {
    if estimates.len() < 2 {
        return Err(Error::TooFewSets {
            found: estimates.len(),
        });
    }
    let dfcom = df.value(estimates);
    let pool = |pick: fn(&EffectEstimates) -> Option<Effect>| -> Result<Option<Pooled>, Error> {
        let mut q = Vec::with_capacity(estimates.len());
        let mut u = Vec::with_capacity(estimates.len());
        for e in estimates {
            let Some(effect) = pick(e) else {
                return Ok(None);
            };
            let Some(se) = effect.se else {
                return Ok(None);
            };
            q.push(effect.estimate);
            u.push(se * se);
        }
        Ok(Some(rubin(&q, &u, dfcom)?))
    };
    Ok(PooledEffects {
        difference: pool(|e| Some(e.difference))?,
        ratio: pool(|e| e.ratio)?,
        log_ratio: pool(|e| e.log_ratio)?,
        odds_ratio: pool(|e| e.odds_ratio)?,
        log_odds_ratio: pool(|e| e.log_odds_ratio)?,
        complete_data_df: dfcom,
    })
}

/// Estimates the effect in every matched data set and pools it.
///
/// `data` holds the treatment, outcome and covariates of each set, which may
/// differ between sets when the outcome or the covariates were imputed. `df`
/// is the rule for the complete-data degrees of freedom of the pooling.
///
/// # Errors
///
/// See [`Error`].
pub fn estimate_pooled<R: MatchingRecord>(
    sets: &[ImputedSet<'_, R>],
    matches: &ImputedMatches,
    data: &[UnitData],
    options: MatchDataOptions,
    spec: &EffectSpec,
    df: CompleteDataDf,
) -> Result<ImputedEffects, Error> {
    let check = |what, found| {
        if found == sets.len() {
            Ok(())
        } else {
            Err(Error::length(what, sets.len(), found))
        }
    };
    check("matches", matches.sets.len())?;
    check("data", data.len())?;
    let per_set = in_set_order(
        sets.par_iter()
            .zip(&matches.sets)
            .zip(data)
            .map(|((set, matched), units_data)| {
                estimate_set(set, matched, units_data, options, spec)
            })
            .collect(),
    )?;
    let pooled = pool_effects(&per_set, df)?;
    Ok(ImputedEffects { per_set, pooled })
}

/// The effects in one matched data set.
fn estimate_set<R: MatchingRecord>(
    set: &ImputedSet<'_, R>,
    matched: &PropensityMatchedOutcome,
    units_data: &UnitData,
    options: MatchDataOptions,
    spec: &EffectSpec,
) -> Result<EffectEstimates, Error> {
    let ids = set
        .cases
        .iter()
        .chain(set.controls)
        .map(|r| r.id().to_string())
        .collect();
    let anchor = (0..set.cases.len() + set.controls.len())
        .map(|i| i < set.cases.len())
        .collect();
    let units = Units::new(ids, anchor)?;
    let md = match_data(&matched.outcome, &units, options)?;
    let sample = MatchedSample::from_match_data(&md, units_data)?;
    estimate_effects(&sample, spec)
}

/// The results of the sets in the order of the sets, or the error of the
/// first set that failed, whichever thread finished first.
fn in_set_order<T>(results: Vec<Result<T, Error>>) -> Result<Vec<T>, Error> {
    results.into_iter().collect()
}
