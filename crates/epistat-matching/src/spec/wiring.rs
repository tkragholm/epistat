//! Genetic and cardinality matching as options of a [`MatchSpec`].
//!
//! The methods themselves are [`GeneticProblem`] and [`match_cardinality`].
//! This module reads the options, builds their inputs from the units and turns
//! their results into a [`SpecOutput`].

use super::{
    CovariatesSpec, DistanceSpec, MatchMode, MatchSpec, OrderSpec, SpecOutput, build_distance,
    scores_of,
};
use crate::Error;
use crate::matching::{
    BalanceMeasure, CardinalityLimits, CardinalityOutcome, CardinalityRequest, CovariateRecord,
    GeneticFit, GeneticMatchOptions, GeneticProblem, GeneticSearch, LossRule, MatchingRecord,
    encode_covariates_pooled, match_cardinality,
};
use crate::output::{
    LongMatches, MatchData, MatchDataOptions, Units, get_matches_from_pairs,
    match_data_from_weights,
};
use crate::types::{Estimand, MatchDiagnostics, MatchOutcome, MatchedPair, MatchingCriteria};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::Duration;

/// The balance measure that genetic matching tries to improve, `fit.func` of
/// `Matching::GenMatch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FitFuncSpec {
    /// The p-values of tests on every covariate, to be made large.
    #[default]
    #[serde(rename = "pvals")]
    Pvals,
    /// The mean over covariates of the mean difference between the empirical
    /// quantile functions.
    #[serde(rename = "qqmean.mean")]
    QqMeanMean,
    /// The mean differences of every covariate, largest first.
    #[serde(rename = "qqmean.max")]
    QqMeanMax,
    /// The mean over covariates of the largest difference.
    #[serde(rename = "qqmax.mean")]
    QqMaxMean,
    /// The largest differences of every covariate, largest first.
    #[serde(rename = "qqmax.max")]
    QqMaxMax,
    /// The median over covariates of the median difference.
    #[serde(rename = "qqmedian.median")]
    QqMedianMedian,
    /// The median differences of every covariate, largest first.
    #[serde(rename = "qqmedian.max")]
    QqMedianMax,
}

impl From<FitFuncSpec> for BalanceMeasure {
    fn from(spec: FitFuncSpec) -> Self {
        match spec {
            FitFuncSpec::Pvals => Self::PValues,
            FitFuncSpec::QqMeanMean => Self::QqMeanMean,
            FitFuncSpec::QqMeanMax => Self::QqMeanMax,
            FitFuncSpec::QqMaxMean => Self::QqMaxMean,
            FitFuncSpec::QqMaxMax => Self::QqMaxMax,
            FitFuncSpec::QqMedianMedian => Self::QqMedianMedian,
            FitFuncSpec::QqMedianMax => Self::QqMedianMax,
        }
    }
}

/// What the p-values are reduced to, `loss` of `Matching::GenMatch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossSpec {
    /// All p-values, the smallest first.
    #[default]
    Sorted,
    /// Only the smallest p-value.
    Minimum,
}

/// Options of the genetic method, those of `matchit(method = "genetic")` and
/// of the `GenMatch` function it calls.
///
/// The matching variables are the covariates and, when the distance is a
/// score, the score, as `matchit()` takes them. Strata keys of `exact` and the
/// sex key of `match_sex` are matched on equal values. The anchors are the
/// focal group, which takes its controls in the order of `m_order`. The search
/// is seeded by the spec's `seed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneticSpec {
    /// The covariates that are matched on and whose balance is measured.
    pub covariates: CovariatesSpec,
    /// Individuals per generation, `pop.size`. 0 evaluates the starting
    /// weights only. 1 is an error.
    pub pop_size: usize,
    /// Generations to run, `max.generations`. It binds only with
    /// `hard_generation_limit`.
    pub max_generations: usize,
    /// Generations without an improvement after which the search stops,
    /// `wait.generations`.
    pub wait_generations: usize,
    /// Stop at `max_generations`, `hard.generation.limit`.
    pub hard_generation_limit: bool,
    /// The balance measure, `fit.func`.
    pub fit_func: FitFuncSpec,
    /// The reduction of the p-values, `loss`. Read with `fit_func = "pvals"`
    /// only.
    pub loss: LossSpec,
    /// Include a Kolmogorov-Smirnov test for each covariate, `ks`.
    pub ks: bool,
    /// A paired t-test, `paired`. Welch's test when false.
    pub paired: bool,
    /// Bootstrap replicates of the Kolmogorov-Smirnov p-value, `nboots`.
    pub nboots: usize,
    /// The smallest standard deviation of a matching variable, and the amount
    /// added to every weight when the smallest weight is below it. This is the
    /// `tolerance` of `Matching::Match`. It is not the `distance.tolerance` of
    /// `matchit()`, which sets the width of ties in the distance and has no
    /// counterpart here, where ties go to the earlier unit.
    pub tolerance: f64,
    /// The order in which anchors take their controls, `m.order`. Largest and
    /// smallest order by the score, which the distance must give. The default
    /// is largest with a score and the order of the data without one.
    pub m_order: Option<OrderSpec>,
}

impl Default for GeneticSpec {
    fn default() -> Self {
        let search = GeneticSearch::default();
        let fit = GeneticFit::default();
        Self {
            covariates: CovariatesSpec::default(),
            pop_size: search.population,
            max_generations: search.max_generations,
            wait_generations: search.wait_generations,
            hard_generation_limit: search.hard_generation_limit,
            fit_func: FitFuncSpec::Pvals,
            loss: LossSpec::Sorted,
            ks: fit.ks,
            paired: fit.paired,
            nboots: fit.bootstrap,
            tolerance: f64::EPSILON.sqrt(),
            m_order: None,
        }
    }
}

/// Options of the cardinality method, those of `matchit(method =
/// "cardinality")`.
///
/// The covariates are balanced as indicator columns, one per level of a
/// categorical covariate. The anchors are the treated group. With a `ratio`
/// the largest subset is selected in which the groups are in that ratio and
/// the covariate means differ by at most the tolerances. Without one (profile
/// matching) the target is the mean of the focal group for the effect in the
/// treated and the controls, and of the sample for the effect in the whole
/// sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CardinalitySpec {
    /// The covariates to balance.
    pub covariates: CovariatesSpec,
    /// Others per focal unit. `None` is profile matching. The `ratio` of the
    /// spec is not read by this method.
    pub ratio: Option<f64>,
    /// The tolerance of each covariate column, or one value for all, `tols`.
    pub tols: Vec<f64>,
    /// Whether each tolerance is in standard deviations, or one value for
    /// all, `std.tols`.
    pub std_tols: Vec<bool>,
    /// Covariates to pair the selected units on, by optimal pair matching on
    /// their Mahalanobis distance, `mahvars`. Empty for no pairing.
    pub mahvars: Vec<String>,
    /// Seconds the search of each stratum may take, `time`. `None` for no
    /// limit.
    pub time: Option<f64>,
    /// Relaxations the search of each stratum may solve. `None` for no limit.
    pub nodes: Option<u64>,
    /// Sampling weights by unit id, for profile matching.
    pub sampling_weights: Option<BTreeMap<String, f64>>,
}

impl Default for CardinalitySpec {
    fn default() -> Self {
        let limits = CardinalityLimits::default();
        Self {
            covariates: CovariatesSpec::default(),
            ratio: Some(1.0),
            tols: vec![0.05],
            std_tols: vec![true],
            mahvars: Vec::new(),
            time: limits.time.map(|time| time.as_secs_f64()),
            nodes: limits.nodes,
            sampling_weights: None,
        }
    }
}

/// The outcome of cardinality or profile matching with the units it was run
/// on.
#[derive(Debug, Clone)]
pub struct CardinalityMatch {
    /// The weights, selection and pairs, by position in `ids`.
    pub outcome: CardinalityOutcome,
    /// The ids of the units, the anchors first and then the candidates, each in
    /// the order given.
    pub ids: Vec<String>,
    /// Whether each unit is an anchor.
    pub anchor: Vec<bool>,
}

impl CardinalityMatch {
    /// The units, for [`get_matches`](crate::get_matches) and the other
    /// outputs.
    ///
    /// # Errors
    ///
    /// [`Error::DuplicateId`] when an id occurs twice.
    pub fn units(&self) -> Result<Units, Error> {
        Units::new(self.ids.clone(), self.anchor.clone())
    }

    /// The weight of each selected unit by id, which is the form the balance
    /// report takes through
    /// [`BalanceWeights::from_unit_weights`](crate::BalanceWeights::from_unit_weights).
    #[must_use]
    pub fn unit_weights(&self) -> HashMap<String, f64> {
        self.ids
            .iter()
            .zip(&self.outcome.weights)
            .filter(|(_, weight)| **weight > 0.0)
            .map(|(id, weight)| (id.clone(), *weight))
            .collect()
    }

    /// The matched data: the units with their weight and, when the selection
    /// was paired, their pair.
    ///
    /// # Errors
    ///
    /// [`Error::DuplicateId`] when an id occurs twice.
    pub fn match_data(&self, options: MatchDataOptions) -> Result<MatchData, Error> {
        let pair = self
            .outcome
            .pairs
            .clone()
            .unwrap_or_else(|| vec![None; self.ids.len()]);
        match_data_from_weights(&self.outcome.weights, &pair, &self.units()?, options)
    }

    /// The long form of the pairs, which exist when `mahvars` was given.
    ///
    /// # Errors
    ///
    /// [`Error::NotPaired`] when the selection was not paired and
    /// [`Error::DuplicateId`] when an id occurs twice.
    pub fn get_matches(&self, include_unmatched: bool) -> Result<LongMatches, Error> {
        let pair = self.outcome.pairs.as_ref().ok_or(Error::NotPaired)?;
        get_matches_from_pairs(
            pair,
            &self.outcome.weights,
            &self.units()?,
            include_unmatched,
        )
    }
}

impl MatchSpec {
    fn reject_unused(&self, method: &str, extra: &[(&'static str, bool)]) -> Result<(), Error> {
        let mut conflicts: Vec<(&'static str, bool)> = vec![
            ("ratio_fallback", !self.ratio_fallback.is_empty()),
            ("unique_by_key", self.unique_by_key.is_some()),
            ("assignment", self.assignment.is_some()),
            (
                "birth_window_days",
                self.birth_window_days != MatchingCriteria::default().birth_date_window_days,
            ),
        ];
        conflicts.extend_from_slice(extra);
        match conflicts.iter().find(|(_, set)| *set) {
            Some((option, _)) => Err(Error::option(
                option,
                format!("the {method} method does not use it; drop it"),
            )),
            None => Ok(()),
        }
    }

    pub(super) fn validate_genetic(&self) -> Result<(), Error> {
        if self.mode != MatchMode::Standard {
            return Err(Error::option(
                "method",
                "genetic matching needs anchors and candidates in standard mode",
            ));
        }
        self.reject_unused("genetic", &[])?;
        let genetic = &self.genetic;
        if genetic.pop_size == 1 {
            return Err(Error::option(
                "genetic.pop_size",
                "a population of 1 cannot recombine; use 0 or at least 2",
            ));
        }
        for (name, value) in [
            ("genetic.max_generations", genetic.max_generations),
            ("genetic.wait_generations", genetic.wait_generations),
        ] {
            if value == 0 {
                return Err(Error::option(name, "this is 0; use 1 or more"));
            }
        }
        if !(genetic.tolerance.is_finite() && genetic.tolerance > 0.0) {
            return Err(Error::option(
                "genetic.tolerance",
                "the tolerance must be positive and finite",
            ));
        }
        if self.estimand != Estimand::Att {
            return Err(Error::option(
                "estimand",
                "genetic matching takes its focal group from the anchors; \
                 give the focal group as the anchors and use att",
            ));
        }
        let score = matches!(
            self.distance,
            DistanceSpec::Propensity { .. } | DistanceSpec::Scores { .. }
        );
        let plain = matches!(self.distance, DistanceSpec::Date { caliper: None });
        if !score && !plain {
            return Err(Error::option(
                "distance",
                "genetic matching adds a propensity score or a score of your own to the \
                 covariates it matches on; set the distance to one of them, or drop it",
            ));
        }
        match genetic.m_order {
            Some(OrderSpec::Largest | OrderSpec::Smallest) if !score => {
                return Err(Error::option(
                    "genetic.m_order",
                    "largest and smallest order the anchors by a score; \
                     set the distance to propensity or scores, or use data",
                ));
            }
            Some(OrderSpec::Random | OrderSpec::Closest | OrderSpec::Farthest) => {
                return Err(Error::option(
                    "genetic.m_order",
                    "genetic matching orders anchors by largest, smallest or data",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn validate_cardinality(&self) -> Result<(), Error> {
        if self.mode != MatchMode::Standard {
            return Err(Error::option(
                "method",
                "cardinality matching needs anchors and candidates in standard mode",
            ));
        }
        self.reject_unused(
            "cardinality",
            &[
                ("ratio", self.ratio != 1),
                ("replace", self.replace),
                ("calipers", !self.calipers.is_empty()),
                ("antiexact", !self.antiexact.is_empty()),
                ("match_sex", self.match_sex),
                ("distance", !self.distance.is_default()),
                ("discard", self.discard.is_some()),
            ],
        )?;
        let cardinality = &self.cardinality;
        if self.estimand == Estimand::Atm {
            return Err(Error::option(
                "estimand",
                "cardinality matching gives the effect in the treated, the controls or the \
                 whole sample; use att, atc or ate",
            ));
        }
        if cardinality
            .ratio
            .is_some_and(|ratio| !(ratio.is_finite() && ratio > 0.0))
        {
            return Err(Error::option(
                "cardinality.ratio",
                "the ratio must be positive and finite, or absent for profile matching",
            ));
        }
        if cardinality.tols.is_empty()
            || cardinality
                .tols
                .iter()
                .any(|tol| !(tol.is_finite() && *tol >= 0.0))
        {
            return Err(Error::option(
                "cardinality.tols",
                "give one tolerance, or one for each covariate column, each 0 or more",
            ));
        }
        if cardinality.std_tols.is_empty() {
            return Err(Error::option(
                "cardinality.std_tols",
                "give one flag, or one for each covariate column",
            ));
        }
        if cardinality
            .time
            .is_some_and(|time| !(time > 0.0 && Duration::try_from_secs_f64(time).is_ok()))
        {
            return Err(Error::option(
                "cardinality.time",
                "the time limit must be a positive number of seconds that a duration can \
                 hold, or absent for none",
            ));
        }
        if cardinality.nodes == Some(0) {
            return Err(Error::option(
                "cardinality.nodes",
                "the node limit is 0; use 1 or more, or leave it out",
            ));
        }
        let profile = cardinality.ratio.is_none() || self.estimand == Estimand::Ate;
        if let Some(weights) = &cardinality.sampling_weights {
            if !profile {
                return Err(Error::option(
                    "cardinality.sampling_weights",
                    "sampling weights need profile matching: leave the ratio out, or use ate",
                ));
            }
            if weights.values().any(|w| !(w.is_finite() && *w >= 0.0)) {
                return Err(Error::option(
                    "cardinality.sampling_weights",
                    "a sampling weight is negative or not finite",
                ));
            }
        }
        if !cardinality.mahvars.is_empty() {
            let whole = cardinality.ratio.is_some_and(|r| r.fract() == 0.0);
            if !whole || self.estimand == Estimand::Atc {
                return Err(Error::option(
                    "cardinality.mahvars",
                    "pairing needs a whole-number ratio and att or ate",
                ));
            }
        }
        Ok(())
    }
}

/// The value of a strata key of each unit, anchors first.
fn strata_values<R: MatchingRecord>(
    option: &'static str,
    key: &str,
    anchors: &[R],
    candidates: &[R],
) -> Result<Vec<String>, Error> {
    anchors
        .iter()
        .chain(candidates)
        .map(|record| {
            record.stratum(key).map(str::to_owned).ok_or_else(|| {
                Error::option(
                    option,
                    format!("the unit {} has no value for the key {key}", record.id()),
                )
            })
        })
        .collect()
}

/// The columns of an encoded covariate matrix.
fn columns_of(matrix: &crate::matching::CovariateMatrix) -> Vec<Vec<f64>> {
    (0..matrix.n_cols)
        .map(|column| {
            (0..matrix.n_rows)
                .map(|row| matrix.values[row * matrix.n_cols + column])
                .collect()
        })
        .collect()
}

pub(super) fn run_genetic<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    anchors: &[R],
    candidates: &[R],
) -> Result<SpecOutput, Error> {
    let genetic = &spec.genetic;
    let ids: Vec<&str> = anchors
        .iter()
        .chain(candidates)
        .map(MatchingRecord::id)
        .collect();
    let n = ids.len();
    let focal: Vec<bool> = (0..n).map(|i| i < anchors.len()).collect();

    let encoded =
        encode_covariates_pooled(anchors, candidates, &genetic.covariates.encoding(false))?;
    let balance = columns_of(&encoded);
    let mut variables = balance.clone();
    let mut calipers: Vec<Option<f64>> = encoded
        .column_names
        .iter()
        .map(|name| spec.calipers.get(name).copied())
        .collect();
    if let Some(field) = spec
        .calipers
        .keys()
        .find(|field| !encoded.column_names.contains(field))
    {
        return Err(Error::option(
            "calipers",
            format!("{field} is not one of the covariate columns the method matches on"),
        ));
    }
    let mut exact = vec![false; variables.len()];

    let built = build_distance(
        &spec.distance,
        anchors,
        candidates,
        spec.estimand,
        spec.seed,
        false,
    )?;
    let score: Option<Vec<f64>> = built
        .config
        .as_ref()
        .and_then(scores_of)
        .map(|scores| {
            ids.iter()
                .map(|id| {
                    scores.get(id).copied().ok_or_else(|| {
                        Error::option("distance", format!("the unit {id} has no score"))
                    })
                })
                .collect::<Result<Vec<f64>, Error>>()
        })
        .transpose()?;
    if let Some(score) = &score {
        variables.push(score.clone());
        calipers.push(spec.distance.caliper());
        exact.push(false);
    }

    let mut keys: Vec<(&'static str, &str)> = spec
        .exact
        .iter()
        .map(|key| ("exact", key.as_str()))
        .collect();
    if spec.match_sex {
        keys.push(("match_sex", spec.sex_key.as_str()));
    }
    for (option, key) in keys {
        let values = strata_values(option, key, anchors, candidates)?;
        let levels: BTreeSet<&String> = values.iter().collect();
        let code = |value: &String| {
            levels.iter().position(|level| *level == value).unwrap_or(0) as f64 + 1.0
        };
        variables.push(values.iter().map(code).collect());
        calipers.push(None);
        exact.push(true);
    }
    let calipers = if calipers.iter().any(Option::is_some) {
        calipers
    } else {
        Vec::new()
    };

    let mut forbidden = Vec::new();
    for key in &spec.antiexact {
        let values = strata_values("antiexact", key, anchors, candidates)?;
        for i in (0..n).filter(|&i| focal[i]) {
            for j in (0..n).filter(|&j| !focal[j]) {
                if values[i] == values[j] {
                    forbidden.push((i, j));
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    let by_score = |descending: bool| {
        let score = score.as_deref().unwrap_or(&[]);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let ordering = if descending {
                score[b].total_cmp(&score[a])
            } else {
                score[a].total_cmp(&score[b])
            };
            ordering.then(a.cmp(&b))
        });
        order
    };
    match genetic.m_order.unwrap_or(if score.is_some() {
        OrderSpec::Largest
    } else {
        OrderSpec::Data
    }) {
        OrderSpec::Largest => order = by_score(true),
        OrderSpec::Smallest => order = by_score(false),
        _ => {}
    }

    let problem = GeneticProblem::new(
        &focal,
        &variables,
        &balance,
        &order,
        GeneticMatchOptions {
            ratio: spec.ratio,
            replace: spec.replace,
            calipers,
            exact,
            forbidden,
        },
        genetic.tolerance,
    )?;
    let fit = GeneticFit {
        measure: genetic.fit_func.into(),
        loss: match genetic.loss {
            LossSpec::Sorted => LossRule::Sorted,
            LossSpec::Minimum => LossRule::Minimum,
        },
        ks: genetic.ks,
        paired: genetic.paired,
        bootstrap: genetic.nboots,
    };
    let search = GeneticSearch {
        population: genetic.pop_size,
        max_generations: genetic.max_generations,
        wait_generations: genetic.wait_generations,
        hard_generation_limit: genetic.hard_generation_limit,
        seed: spec.seed,
        ..GeneticSearch::default()
    };
    let found = problem.search(&fit, &search)?;

    let rows = &found.matches;
    let pairs: Vec<MatchedPair> = rows
        .focal
        .iter()
        .zip(&rows.control)
        .map(|(f, c)| MatchedPair::new(ids[*f], ids[*c]))
        .collect();
    let matched: HashSet<usize> = rows.focal.iter().copied().collect();
    let used: HashSet<usize> = rows.control.iter().copied().collect();
    let unmatched = anchors.len() - matched.len();
    let outcome = MatchOutcome {
        unmatched_cases: unmatched,
        used_controls: if spec.replace { 0 } else { used.len() },
        matched_cases: matched.len(),
        avg_controls_per_case: (!matched.is_empty())
            .then(|| rows.len() as f64 / matched.len() as f64),
        diagnostics: MatchDiagnostics {
            total_cases_evaluated: anchors.len(),
            cases_below_required_ratio: unmatched,
            matched_cases: matched.len(),
            pairs_selected: rows.len(),
            requested_estimand: spec.estimand,
            realized_estimand: spec.estimand,
            ..MatchDiagnostics::default()
        },
        pairs,
    };
    Ok(SpecOutput::Genetic {
        outcome: Box::new(outcome),
        search: Box::new(found),
    })
}

pub(super) fn run_cardinality<R: MatchingRecord + CovariateRecord>(
    spec: &MatchSpec,
    anchors: &[R],
    candidates: &[R],
) -> Result<SpecOutput, Error> {
    let options = &spec.cardinality;
    let ids: Vec<String> = anchors
        .iter()
        .chain(candidates)
        .map(|record| record.id().to_string())
        .collect();
    let treated: Vec<bool> = (0..ids.len()).map(|i| i < anchors.len()).collect();

    let mut encoding = options.covariates.encoding(false);
    encoding.drop_first_categorical_level = false;
    let covariates = columns_of(&encode_covariates_pooled(anchors, candidates, &encoding)?);

    let pair_on = if options.mahvars.is_empty() {
        Vec::new()
    } else {
        let mut encoding = CovariatesSpec {
            names: options.mahvars.clone(),
            missing: options.covariates.missing,
        }
        .encoding(false);
        encoding.drop_first_categorical_level = false;
        columns_of(&encode_covariates_pooled(anchors, candidates, &encoding)?)
    };

    let exact: Option<Vec<u64>> = if spec.exact.is_empty() {
        None
    } else {
        let mut per_key = Vec::new();
        for key in &spec.exact {
            per_key.push(strata_values("exact", key, anchors, candidates)?);
        }
        let tuples: Vec<Vec<&String>> = (0..ids.len())
            .map(|unit| per_key.iter().map(|values| &values[unit]).collect())
            .collect();
        let levels: BTreeSet<&Vec<&String>> = tuples.iter().collect();
        Some(
            tuples
                .iter()
                .map(|tuple| levels.iter().position(|level| *level == tuple).unwrap_or(0) as u64)
                .collect(),
        )
    };

    let sampling: Option<Vec<f64>> = options
        .sampling_weights
        .as_ref()
        .map(|weights| {
            ids.iter()
                .map(|id| {
                    weights.get(id).copied().ok_or_else(|| {
                        Error::option(
                            "cardinality.sampling_weights",
                            format!("the unit {id} has no sampling weight"),
                        )
                    })
                })
                .collect::<Result<Vec<f64>, Error>>()
        })
        .transpose()?;

    let mut request = CardinalityRequest::new(&treated, &covariates);
    request.estimand = spec.estimand;
    request.ratio = options.ratio;
    request.tols.clone_from(&options.tols);
    request.std_tols.clone_from(&options.std_tols);
    request.exact = exact.as_deref();
    request.sampling_weights = sampling.as_deref();
    request.pair_on = &pair_on;
    request.limits = CardinalityLimits {
        time: options
            .time
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(|_| {
                Error::option(
                    "cardinality.time",
                    "the time limit is not a number of seconds that a duration can hold",
                )
            })?,
        nodes: options.nodes,
    };
    let outcome = match_cardinality(&request)?;
    Ok(SpecOutput::Cardinality(Box::new(CardinalityMatch {
        outcome,
        ids,
        anchor: treated,
    })))
}
