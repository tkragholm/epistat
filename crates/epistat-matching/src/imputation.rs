//! Matching after multiple imputation, and the pooled effect.
//!
//! The caller imputes the missing values and gives the `m` completed data
//! sets. Nothing here imputes. Each set is an [`ImputedSet`], the anchors and
//! the candidates with their imputed covariates, with the same ids in every
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
//! [`EffectSpec`] says. The complete-data degrees of freedom are the smallest,
//! over the sets, of the number of rows of the fit less the number of
//! coefficients, which is what `mice::pool()` takes from the first fit of
//! sets of equal size.
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

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use epistat_inference::InferenceError;
use epistat_inference::pooling::{Pooled, rubin};

use crate::effects::{
    Effect, EffectError, EffectEstimates, EffectSpec, MatchedSample, UnitData, estimate_effects,
};
use crate::matching::{
    ConstraintGroup, CovariateRecord, DistanceConfig, EstimationError, MatchingRecord,
    PropensityMatchedOutcome, PropensityScoreConfig, ScoreTable, SelectionStrategy,
    StandardMatchRequest, estimate_propensity_and_match, match_standard,
    prepare_propensity_distance_config,
};
use crate::output::{MatchDataOptions, OutputError, Units, match_data};
use crate::types::{MatchRatio, MatchingCriteria};

/// How the imputed data sets are combined in matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImputationApproach {
    /// Match each data set on its own propensity score.
    Within,
    /// Match each data set on the propensity score averaged over the sets.
    Across,
}

/// One completed data set: the anchors and the candidates with imputed
/// covariates.
#[derive(Debug)]
pub struct ImputedSet<'a, R> {
    /// The anchors, the treated units for the effect in the treated.
    pub anchors: &'a [R],
    /// The candidates.
    pub candidates: &'a [R],
}

impl<R> Clone for ImputedSet<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> Copy for ImputedSet<'_, R> {}

/// Why a match or a pooled effect could not be made.
#[derive(Debug, Clone)]
pub enum ImputationError {
    /// No data set was given, or one: Rubin's rules need at least two.
    TooFewSets {
        /// The number given.
        found: usize,
    },
    /// An input list does not have one entry per data set.
    SetCount {
        /// The input.
        what: &'static str,
        /// The number of data sets.
        expected: usize,
        /// The number of entries.
        found: usize,
    },
    /// The across approach found a unit in one set that another set lacks.
    UnitsDiffer {
        /// The data set, counted from 0.
        set: usize,
        /// The unit.
        id: String,
    },
    /// `reestimate` was set with the across approach, which fits the score
    /// before it is averaged.
    ReestimateAcross,
    /// A propensity score could not be estimated.
    Estimation(EstimationError),
    /// The matched data of a set could not be made.
    Output(OutputError),
    /// The effect could not be estimated in a set.
    Effect(EffectError),
    /// The effects could not be pooled.
    Pooling(InferenceError),
}

impl Display for ImputationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewSets { found } => write!(
                f,
                "{found} imputed data sets were given; pooling needs at least 2"
            ),
            Self::SetCount {
                what,
                expected,
                found,
            } => write!(
                f,
                "{what} has {found} entries for {expected} data sets; give one for each"
            ),
            Self::UnitsDiffer { set, id } => write!(
                f,
                "unit {id} of data set {set} is not in the first data set; \
                 the across approach needs the same units in every set"
            ),
            Self::ReestimateAcross => f.write_str(
                "reestimate is not available with the across approach; \
                 the score is fitted in each set before it is averaged",
            ),
            Self::Estimation(e) => write!(f, "{e}"),
            Self::Output(e) => write!(f, "{e}"),
            Self::Effect(e) => write!(f, "{e}"),
            Self::Pooling(e) => write!(f, "{e}"),
        }
    }
}

impl Error for ImputationError {}

impl From<EstimationError> for ImputationError {
    fn from(e: EstimationError) -> Self {
        Self::Estimation(e)
    }
}

impl From<OutputError> for ImputationError {
    fn from(e: OutputError) -> Self {
        Self::Output(e)
    }
}

impl From<EffectError> for ImputationError {
    fn from(e: EffectError) -> Self {
        Self::Effect(e)
    }
}

impl From<InferenceError> for ImputationError {
    fn from(e: InferenceError) -> Self {
        Self::Pooling(e)
    }
}

/// The matches of the imputed data sets.
#[derive(Debug, Clone)]
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
/// See [`ImputationError`].
pub fn match_imputed<R, S, G>(
    sets: &[ImputedSet<'_, R>],
    approach: ImputationApproach,
    criteria: &MatchingCriteria,
    strategy: &S,
    ratio_fallback: &[MatchRatio],
    extra_constraints: &G,
    config: &PropensityScoreConfig,
) -> Result<ImputedMatches, ImputationError>
where
    R: MatchingRecord + CovariateRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
{
    if sets.len() < 2 {
        return Err(ImputationError::TooFewSets { found: sets.len() });
    }
    match approach {
        ImputationApproach::Within => {
            let mut matched = Vec::with_capacity(sets.len());
            for set in sets {
                matched.push(estimate_propensity_and_match(
                    set.anchors,
                    set.candidates,
                    criteria,
                    strategy.clone(),
                    ratio_fallback,
                    extra_constraints,
                    config,
                )?);
            }
            Ok(ImputedMatches {
                approach,
                sets: matched,
            })
        }
        ImputationApproach::Across => {
            if config.reestimate {
                return Err(ImputationError::ReestimateAcross);
            }
            let mut estimates = Vec::with_capacity(sets.len());
            for set in sets {
                estimates.push(
                    prepare_propensity_distance_config(set.anchors, set.candidates, config)?
                        .estimate,
                );
            }
            let averaged = average_scores(&estimates, sets)?;
            let distance_config =
                DistanceConfig::propensity_score_table(averaged.clone(), config.caliper);
            let mut matched = Vec::with_capacity(sets.len());
            for (set, mut estimate) in sets.iter().zip(estimates) {
                let request = StandardMatchRequest {
                    criteria,
                    strategy: strategy.clone(),
                    constraints: extra_constraints,
                    ratio_fallback,
                    distance_config: Some(&distance_config),
                };
                let outcome = match_standard(set.anchors, set.candidates, request);
                estimate.set_distance_scores(&averaged);
                matched.push(PropensityMatchedOutcome {
                    outcome,
                    propensity: estimate,
                });
            }
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
) -> Result<ScoreTable, ImputationError> {
    let first = &estimates[0];
    let mut total: Vec<f64> = first.distance_scores.to_vec();
    for (index, estimate) in estimates.iter().enumerate().skip(1) {
        let same_order = estimate.ids == first.ids;
        if estimate.distance_scores.len() != first.distance_scores.len() {
            let id = sets[index]
                .anchors
                .iter()
                .chain(sets[index].candidates)
                .map(MatchingRecord::id)
                .find(|id| first.position(id).is_none())
                .unwrap_or("")
                .to_string();
            return Err(ImputationError::UnitsDiffer { set: index, id });
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
                    return Err(ImputationError::UnitsDiffer {
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

/// The effects of a treatment pooled over the imputed data sets.
///
/// An effect is `None` when a data set gives no standard error for it, or the
/// effect does not exist for the model, as in [`EffectEstimates`].
#[derive(Debug, Clone, PartialEq)]
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
    /// The complete-data degrees of freedom given to the pooling.
    pub complete_data_df: f64,
}

/// The effects of every data set and their pooled value.
#[derive(Debug, Clone, PartialEq)]
pub struct ImputedEffects {
    /// The estimates of each data set, in the order given.
    pub per_set: Vec<EffectEstimates>,
    /// The pooled effects.
    pub pooled: PooledEffects,
}

/// Pools the effects of the data sets by Rubin's rules.
///
/// # Errors
///
/// [`ImputationError::TooFewSets`] for fewer than two sets and
/// [`ImputationError::Pooling`] when the pooling fails, for instance when a
/// data set has no more rows than coefficients.
pub fn pool_effects(estimates: &[EffectEstimates]) -> Result<PooledEffects, ImputationError> {
    if estimates.len() < 2 {
        return Err(ImputationError::TooFewSets {
            found: estimates.len(),
        });
    }
    let dfcom = estimates
        .iter()
        .map(|e| e.rows as f64 - e.coefficients.len() as f64)
        .fold(f64::INFINITY, f64::min);
    let pool =
        |pick: fn(&EffectEstimates) -> Option<Effect>| -> Result<Option<Pooled>, ImputationError> {
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
            Ok(Some(rubin(&q, &u, Some(dfcom))?))
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
/// differ between sets when the outcome or the covariates were imputed.
///
/// # Errors
///
/// See [`ImputationError`].
pub fn estimate_pooled<R: MatchingRecord>(
    sets: &[ImputedSet<'_, R>],
    matches: &ImputedMatches,
    data: &[UnitData],
    options: MatchDataOptions,
    spec: &EffectSpec,
) -> Result<ImputedEffects, ImputationError> {
    let check = |what, found| {
        if found == sets.len() {
            Ok(())
        } else {
            Err(ImputationError::SetCount {
                what,
                expected: sets.len(),
                found,
            })
        }
    };
    check("matches", matches.sets.len())?;
    check("data", data.len())?;
    let mut per_set = Vec::with_capacity(sets.len());
    for ((set, matched), units_data) in sets.iter().zip(&matches.sets).zip(data) {
        let ids = set
            .anchors
            .iter()
            .chain(set.candidates)
            .map(|r| r.id().to_string())
            .collect();
        let anchor = (0..set.anchors.len() + set.candidates.len())
            .map(|i| i < set.anchors.len())
            .collect();
        let units = Units::new(ids, anchor)?;
        let md = match_data(&matched.outcome, &units, options)?;
        let sample = MatchedSample::from_match_data(&md, units_data)?;
        per_set.push(estimate_effects(&sample, spec)?);
    }
    let pooled = pool_effects(&per_set)?;
    Ok(ImputedEffects { per_set, pooled })
}
