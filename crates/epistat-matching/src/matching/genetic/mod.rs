//! Genetic matching, MatchIt's `method = "genetic"`.
//!
//! Genetic matching (Diamond and Sekhon 2013) chooses the weights of a
//! distance so that the matched sample is balanced, then matches nearest
//! neighbour on that distance. This module has three parts.
//!
//! * [`GeneticProblem::match_pairs`] matches for given weights. It is the
//!   matching `Matching::Match(Weight = 3, Weight.matrix = W)` does, and
//!   MatchIt's genetic method with those weights.
//! * [`GeneticProblem::loss`] measures the balance of such a match with
//!   GenMatch's fitness functions, see [`GeneticFit`].
//! * [`GeneticProblem::search`] looks for the weights with the best loss with
//!   a seeded genetic algorithm.
//!
//! # The distance
//!
//! Every matching variable is centred and divided by its standard deviation
//! over all units, with divisor `n - 1`. The distance between two units is the
//! square root of the sum over variables of the weight times the squared
//! difference of the standardised values. With all weights equal this is the
//! Euclidean distance between standardised variables. The variables are not
//! decorrelated: the weights of `GenMatch` are those of a diagonal matrix
//! applied to the standardised variables, and a Mahalanobis distance appears
//! only through the choice of variables. When the minimum of the weights is
//! below `tolerance` (the square root of the machine epsilon by default),
//! `tolerance` is added to every weight, as `GenMatch` does, so that the
//! weight matrix stays positive definite.
//!
//! # Matching for given weights
//!
//! The units are taken in the order given. Each focal unit in turn takes the
//! `ratio` nearest controls that pass the constraints and, without
//! replacement, are not taken: a focal unit gets all its controls before the
//! next one is served. Ties between controls at the same distance go to the
//! earlier unit in the order given. MatchIt breaks them at random from R's
//! random stream, which cannot be reproduced, so the pairs are compared with
//! MatchIt's only where no two controls tie. A focal unit that has fewer than
//! `ratio` admissible controls left at its turn is unmatched and takes none,
//! which is what MatchIt's genetic method does: with a caliper or an exact
//! variable a unit gets all its controls or none.
//!
//! The constraints are the calipers, which bound the absolute difference of
//! the raw values of a variable, the exact variables, which must be equal, and
//! a list of forbidden pairs. A unit's weight in a pair is the reciprocal of
//! the number of controls it was matched to.
//!
//! The nearest controls are found on a k-d tree over the standardised
//! controls ([`crate::kdtree`]), built once for the problem and searched with
//! the weighted sum of squares of each weight vector, the constraints as its
//! admission test. Without replacement each match works on its own copy of
//! the tree, from which the controls are removed as they are taken. The tree
//! gives the controls a scan of every control gives, ties included. Few
//! controls, many variables, or weights so large that a sum of squares could
//! overflow are left to the scan.
//!
//! # The search
//!
//! The search is not checked against R step by step. `rgenoud` draws its
//! random numbers from R's stream and its operators have details that are not
//! documented well enough to reproduce, so this module implements the
//! algorithm described in its documentation: a population of weight vectors
//! in `[min_weight, max_weight]`, a fixed share of each generation made by each
//! of eight operators (cloning, uniform mutation, boundary mutation,
//! non-uniform mutation, polytope crossover, simple crossover, whole
//! non-uniform mutation and heuristic crossover), parents chosen by binary
//! tournament, the best individual always kept, and a stop after
//! `wait_generations` generations without an improvement. It is checked by the
//! quality of what it finds against `GenMatch`'s, see the tests. The best
//! individual and the whole run depend only on the seed, not on the number of
//! threads: every random draw is made on one thread in a fixed order, and the
//! fitness of each individual is computed independently with a seed from its
//! generation and position.
//!
//! # References
//!
//! 1. Diamond A, Sekhon JS. Genetic matching for estimating causal effects: a
//!    general multivariate matching method for achieving balance in
//!    observational studies. Rev Econ Stat. 2013;95(3):932-45.
//!    doi:10.1162/REST_a_00318
//! 2. Sekhon JS. Multivariate and propensity score matching software with
//!    automated balance optimization: the Matching package for R. J Stat
//!    Softw. 2011;42(7). doi:10.18637/jss.v042.i07
//! 3. Mebane WR, Sekhon JS. Genetic optimization using derivatives: the
//!    rgenoud package for R. J Stat Softw. 2011;42(11).
//!    doi:10.18637/jss.v042.i11

mod balance;
mod ks;
mod search;

use std::fmt;

use epistat_rng::Stream;
use rayon::prelude::*;
use rustc_hash::FxHashSet;

use crate::kdtree::{KdTree, WeightedSquares, worth_a_tree};

pub use balance::{BalanceMeasure, GeneticFit, GeneticLoss, LossRule};
pub use search::{GeneticOutcome, GeneticSearch};

/// A two-sample Kolmogorov-Smirnov test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KsTest {
    /// The largest absolute difference of the empirical distribution
    /// functions.
    pub statistic: f64,
    /// The p-value of the two-sided test.
    pub p_value: f64,
    /// Whether the p-value is exact, which it is when the product of the
    /// sample sizes is below 10000, ties included. Otherwise it is
    /// asymptotic.
    pub exact: bool,
}

/// The two-sample Kolmogorov-Smirnov test of `x` against `y`, as `ks.test`
/// does it: exact below a product of sample sizes of 10000, asymptotic from
/// there, and with the exact distribution given the ties in the pooled
/// sample. `None` when either sample is empty or a value is not finite.
#[must_use]
pub fn ks_two_sample(x: &[f64], y: &[f64]) -> Option<KsTest> {
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return None;
    }
    let (statistic, p_value, exact) = ks::ks_test(x, y)?;
    Some(KsTest {
        statistic,
        p_value,
        exact,
    })
}

/// `1 - K(x)`, the upper tail of the Kolmogorov distribution at `x`, summed
/// to rounding. The p-value of the asymptotic two-sample test is this at
/// `sqrt(m n / (m + n))` times the statistic.
#[must_use]
pub fn ks_limit_upper(x: f64) -> f64 {
    ks::kolmogorov_upper(x).clamp(0.0, 1.0)
}

/// What can go wrong in genetic matching.
#[derive(Debug, Clone, PartialEq)]
pub enum GeneticError {
    /// An input has the wrong length or is empty.
    Shape(String),
    /// A value that must be finite is not.
    NotFinite(String),
    /// An argument is outside its range.
    Invalid(String),
}

impl fmt::Display for GeneticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(message) | Self::NotFinite(message) | Self::Invalid(message) => {
                f.write_str(message)
            }
        }
    }
}

impl std::error::Error for GeneticError {}

/// The constraints and size of a match.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneticMatchOptions {
    /// Controls per focal unit.
    pub ratio: usize,
    /// Whether a control can be matched more than once.
    pub replace: bool,
    /// Per matching variable, the largest absolute difference of the raw
    /// values allowed in a pair. Empty for none, or one entry per variable.
    pub calipers: Vec<Option<f64>>,
    /// Per matching variable, whether the values must be equal in a pair.
    /// Empty for none, or one entry per variable.
    pub exact: Vec<bool>,
    /// Pairs `(focal, control)`, by position in the input, that must not be
    /// matched.
    pub forbidden: Vec<(usize, usize)>,
}

impl Default for GeneticMatchOptions {
    fn default() -> Self {
        Self {
            ratio: 1,
            replace: false,
            calipers: Vec::new(),
            exact: Vec::new(),
            forbidden: Vec::new(),
        }
    }
}

/// The pairs of a match.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneticMatches {
    /// The focal unit of each row, by position in the input.
    pub focal: Vec<usize>,
    /// The control of each row, by position in the input.
    pub control: Vec<usize>,
    /// The weight of each row, the reciprocal of the number of controls of
    /// its focal unit.
    pub weight: Vec<f64>,
}

impl GeneticMatches {
    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.focal.len()
    }

    /// Whether no unit was matched.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.focal.is_empty()
    }
}

/// The units, variables and constraints of a genetic match.
///
/// Built once, then used for any number of weight vectors.
#[derive(Debug, Clone)]
pub struct GeneticProblem {
    /// Matching variables, standardised, row-major over the used units.
    z_focal: Vec<f64>,
    z_control: Vec<f64>,
    /// Raw values for calipers and exact variables, row-major, kept only
    /// when a constraint needs them.
    raw_focal: Vec<f64>,
    raw_control: Vec<f64>,
    limits: Vec<f64>,
    constrained: bool,
    /// Position in the input of each focal and control unit.
    focal_position: Vec<usize>,
    control_position: Vec<usize>,
    /// Per focal unit, the sorted indices of controls it must not match.
    forbidden: Vec<Vec<usize>>,
    /// Balance covariates of the focal and control units, one vector per
    /// covariate.
    balance_focal: Vec<Vec<f64>>,
    balance_control: Vec<Vec<f64>>,
    few_values: Vec<bool>,
    variables: usize,
    options: GeneticMatchOptions,
    tolerance: f64,
    /// Per matching variable, the largest standardised value minus the
    /// smallest over the used units. No difference between a focal unit and
    /// a control is larger in floating point.
    z_range: Vec<f64>,
    /// The k-d tree over `z_control`, keyed by control row, when the problem
    /// is worth one.
    tree: Option<KdTree>,
}

/// The standard deviation `GenMatch` standardises by: divisor `n - 1`, and
/// never below `tolerance`.
fn standard_deviation(values: &[f64], tolerance: f64) -> (f64, f64) {
    let n = values.len() as f64;
    let mut sum = 0.0;
    for v in values {
        sum += v;
    }
    let mean = sum / n;
    let mut squares = 0.0;
    for v in values {
        squares += (v - mean) * (v - mean);
    }
    let sd = (squares / (n - 1.0)).sqrt();
    (mean, if sd < tolerance { tolerance } else { sd })
}

impl GeneticProblem {
    /// Prepares a match.
    ///
    /// `focal[i]` says whether unit `i` is in the focal group, the treated
    /// units for the ATT and the controls for the ATC. `variables` holds the
    /// matching variables as one vector per variable, each as long as `focal`.
    /// `balance` holds the covariates whose balance is measured, in the same
    /// way. `order` lists the positions of the units to use, in the order in
    /// which the focal units are served; a unit that is not listed takes no
    /// part, not even in the standardisation. `tolerance` is the smallest
    /// standard deviation and the ridge of the weight matrix; `GenMatch` uses
    /// the square root of the machine epsilon.
    ///
    /// # Errors
    ///
    /// When a vector has the wrong length, a value is not finite, `order`
    /// repeats a position or leaves one out of range, or one group is empty.
    pub fn new(
        focal: &[bool],
        variables: &[Vec<f64>],
        balance: &[Vec<f64>],
        order: &[usize],
        options: GeneticMatchOptions,
        tolerance: f64,
    ) -> Result<Self, GeneticError> {
        let n = focal.len();
        if variables.is_empty() {
            return Err(GeneticError::Shape(
                "genetic matching needs at least one matching variable".into(),
            ));
        }
        if balance.is_empty() {
            return Err(GeneticError::Shape(
                "genetic matching needs at least one balance covariate".into(),
            ));
        }
        for (set, name) in [
            (variables, "matching variable"),
            (balance, "balance covariate"),
        ] {
            for (k, column) in set.iter().enumerate() {
                if column.len() != n {
                    return Err(GeneticError::Shape(format!(
                        "{name} {k} has {} values for {n} units",
                        column.len()
                    )));
                }
                if let Some(at) = column.iter().position(|v| !v.is_finite()) {
                    return Err(GeneticError::NotFinite(format!(
                        "{name} {k} is not finite at unit {at}"
                    )));
                }
            }
        }
        if !(tolerance.is_finite() && tolerance > 0.0) {
            return Err(GeneticError::Invalid(
                "the tolerance must be positive and finite".into(),
            ));
        }
        if options.ratio == 0 {
            return Err(GeneticError::Invalid("the ratio must be at least 1".into()));
        }
        let p = variables.len();
        if !options.calipers.is_empty() && options.calipers.len() != p {
            return Err(GeneticError::Shape(format!(
                "calipers has {} entries for {p} matching variables",
                options.calipers.len()
            )));
        }
        if !options.exact.is_empty() && options.exact.len() != p {
            return Err(GeneticError::Shape(format!(
                "exact has {} entries for {p} matching variables",
                options.exact.len()
            )));
        }
        if let Some(bad) = options
            .calipers
            .iter()
            .flatten()
            .find(|c| !(c.is_finite() && **c >= 0.0))
        {
            return Err(GeneticError::Invalid(format!(
                "a caliper must be finite and not negative, got {bad}"
            )));
        }
        let mut seen = vec![false; n];
        for &position in order {
            if position >= n || std::mem::replace(&mut seen[position], true) {
                return Err(GeneticError::Invalid(format!(
                    "the order repeats or leaves out of range the position {position}"
                )));
            }
        }
        let used = order.len();
        let n_focal = order.iter().filter(|&&i| focal[i]).count();
        let n_control = used - n_focal;
        if n_focal == 0 || n_control == 0 {
            return Err(GeneticError::Invalid(
                "both the focal and the control group need at least one unit".into(),
            ));
        }
        if used < 2 {
            return Err(GeneticError::Invalid(
                "at least two units are needed".into(),
            ));
        }

        let mut limits = vec![f64::INFINITY; p];
        for (k, caliper) in options.calipers.iter().enumerate() {
            if let Some(width) = caliper {
                limits[k] = *width;
            }
        }
        for (k, &flag) in options.exact.iter().enumerate() {
            if flag {
                limits[k] = 0.0;
            }
        }
        let constrained = limits.iter().any(|l| l.is_finite());

        let focal_position: Vec<usize> = order.iter().copied().filter(|&i| focal[i]).collect();
        let control_position: Vec<usize> = order.iter().copied().filter(|&i| !focal[i]).collect();

        let mut z_focal = vec![0.0; n_focal * p];
        let mut z_control = vec![0.0; n_control * p];
        let mut raw_focal = Vec::new();
        let mut raw_control = Vec::new();
        if constrained {
            raw_focal = vec![0.0; n_focal * p];
            raw_control = vec![0.0; n_control * p];
        }
        for (k, column) in variables.iter().enumerate() {
            let used_values: Vec<f64> = order.iter().map(|&i| column[i]).collect();
            let (mean, sd) = standard_deviation(&used_values, tolerance);
            for (row, &i) in focal_position.iter().enumerate() {
                z_focal[row * p + k] = (column[i] - mean) / sd;
                if constrained {
                    raw_focal[row * p + k] = column[i];
                }
            }
            for (row, &i) in control_position.iter().enumerate() {
                z_control[row * p + k] = (column[i] - mean) / sd;
                if constrained {
                    raw_control[row * p + k] = column[i];
                }
            }
        }

        let focal_index: rustc_hash::FxHashMap<usize, usize> = focal_position
            .iter()
            .enumerate()
            .map(|(row, &i)| (i, row))
            .collect();
        let control_index: rustc_hash::FxHashMap<usize, usize> = control_position
            .iter()
            .enumerate()
            .map(|(row, &i)| (i, row))
            .collect();
        let mut forbidden = vec![Vec::new(); n_focal];
        for &(f, c) in &options.forbidden {
            if let (Some(&fr), Some(&cr)) = (focal_index.get(&f), control_index.get(&c)) {
                forbidden[fr].push(cr);
            }
        }
        for list in &mut forbidden {
            list.sort_unstable();
            list.dedup();
        }

        let balance_focal: Vec<Vec<f64>> = balance
            .iter()
            .map(|c| focal_position.iter().map(|&i| c[i]).collect())
            .collect();
        let balance_control: Vec<Vec<f64>> = balance
            .iter()
            .map(|c| control_position.iter().map(|&i| c[i]).collect())
            .collect();
        let few_values: Vec<bool> = balance
            .iter()
            .map(|c| {
                let mut distinct: FxHashSet<u64> = FxHashSet::default();
                for &i in order {
                    distinct.insert(c[i].to_bits());
                    if distinct.len() >= 3 {
                        return false;
                    }
                }
                true
            })
            .collect();

        let z_range: Vec<f64> = (0..p)
            .map(|k| {
                let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
                for value in z_focal.iter().chain(&z_control).skip(k).step_by(p) {
                    low = low.min(*value);
                    high = high.max(*value);
                }
                high - low
            })
            .collect();
        let tree =
            (worth_a_tree(n_control, p) && z_control.iter().all(|z| z.is_finite())).then(|| {
                let keys: Vec<usize> = (0..n_control).collect();
                KdTree::new(p, &z_control, &keys, n_control)
            });

        Ok(Self {
            z_focal,
            z_control,
            raw_focal,
            raw_control,
            limits,
            constrained,
            focal_position,
            control_position,
            forbidden,
            balance_focal,
            balance_control,
            few_values,
            variables: p,
            options,
            tolerance,
            z_range,
            tree,
        })
    }

    /// The number of matching variables, which is the length of a weight
    /// vector.
    #[must_use]
    pub const fn variables(&self) -> usize {
        self.variables
    }

    /// The number of balance covariates.
    #[must_use]
    pub fn balance_covariates(&self) -> usize {
        self.balance_focal.len()
    }

    /// The weights as the distance uses them: with `tolerance` added to every
    /// one when the smallest is below it.
    fn effective_weights(&self, weights: &[f64]) -> Vec<f64> {
        let smallest = weights.iter().copied().fold(f64::INFINITY, f64::min);
        if smallest < self.tolerance {
            weights.iter().map(|w| w + self.tolerance).collect()
        } else {
            weights.to_vec()
        }
    }

    fn check_weights(&self, weights: &[f64]) -> Result<(), GeneticError> {
        if weights.len() != self.variables {
            return Err(GeneticError::Shape(format!(
                "{} weights for {} matching variables",
                weights.len(),
                self.variables
            )));
        }
        if let Some(bad) = weights.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
            return Err(GeneticError::Invalid(format!(
                "a weight must be finite and not negative, got {bad}"
            )));
        }
        Ok(())
    }

    /// Matches for `weights`, one per matching variable.
    ///
    /// # Errors
    ///
    /// When the weights have the wrong length or a weight is negative or not
    /// finite.
    pub fn match_pairs(&self, weights: &[f64]) -> Result<GeneticMatches, GeneticError> {
        self.check_weights(weights)?;
        let pairs = self.pairs_by_row(&self.effective_weights(weights));
        let mut focal = Vec::with_capacity(pairs.len());
        let mut control = Vec::with_capacity(pairs.len());
        let mut weight = Vec::with_capacity(pairs.len());
        let mut start = 0;
        while start < pairs.len() {
            let mut end = start;
            while end < pairs.len() && pairs[end].0 == pairs[start].0 {
                end += 1;
            }
            let share = 1.0 / (end - start) as f64;
            for &(f, c) in &pairs[start..end] {
                focal.push(self.focal_position[f]);
                control.push(self.control_position[c]);
                weight.push(share);
            }
            start = end;
        }
        Ok(GeneticMatches {
            focal,
            control,
            weight,
        })
    }

    /// Adds to `best`, nearest first and ties by position, the admissible
    /// controls of focal unit `row` among `start..end` that are nearer than
    /// what it holds, keeping at most `ratio`.
    fn scan(
        &self,
        row: usize,
        w: &[f64],
        taken: Option<&[bool]>,
        start: usize,
        end: usize,
        best: &mut Vec<(f64, usize)>,
    ) {
        let p = self.variables;
        let ratio = self.options.ratio;
        let zf = &self.z_focal[row * p..(row + 1) * p];
        for c in start..end {
            if let Some(taken) = taken {
                if taken[c] {
                    continue;
                }
            }
            if !self.admits(row, c) {
                continue;
            }
            let zc = &self.z_control[c * p..(c + 1) * p];
            let mut d = 0.0;
            for k in 0..p {
                let diff = zf[k] - zc[k];
                d += w[k] * (diff * diff);
            }
            if best.len() == ratio && d >= best[ratio - 1].0 {
                continue;
            }
            let at = best.partition_point(|&(other, _)| other <= d);
            if best.len() == ratio {
                best.pop();
            }
            best.insert(at, (d, c));
        }
    }

    /// The `ratio` nearest admissible controls of focal unit `row`, nearest
    /// first, ties by position. `taken` marks the controls that are gone.
    /// With `split`, a long list of controls is scanned in chunks of a fixed
    /// size in parallel and the chunks are merged in order, which gives the
    /// list a single scan would.
    fn nearest(
        &self,
        row: usize,
        w: &[f64],
        taken: Option<&[bool]>,
        split: bool,
        best: &mut Vec<(f64, usize)>,
    ) {
        const CHUNK: usize = 8192;
        let ratio = self.options.ratio;
        let n_control = self.control_position.len();
        best.clear();
        if split && n_control >= 2 * CHUNK {
            let chunks: Vec<Vec<(f64, usize)>> = (0..n_control.div_ceil(CHUNK))
                .into_par_iter()
                .map(|chunk| {
                    let mut local = Vec::with_capacity(ratio);
                    self.scan(
                        row,
                        w,
                        taken,
                        chunk * CHUNK,
                        ((chunk + 1) * CHUNK).min(n_control),
                        &mut local,
                    );
                    local
                })
                .collect();
            for list in chunks {
                for (d, c) in list {
                    if best.len() == ratio && d >= best[ratio - 1].0 {
                        continue;
                    }
                    let at = best.partition_point(|&(other, _)| other <= d);
                    if best.len() == ratio {
                        best.pop();
                    }
                    best.insert(at, (d, c));
                }
            }
        } else {
            self.scan(row, w, taken, 0, n_control, best);
        }
        // A focal unit that cannot get all its controls gets none.
        if best.len() < ratio {
            best.clear();
        }
    }

    /// Whether control `c` passes the calipers, the exact variables and the
    /// forbidden pairs of focal unit `row`.
    fn admits(&self, row: usize, c: usize) -> bool {
        let p = self.variables;
        if self.constrained {
            let rf = &self.raw_focal[row * p..(row + 1) * p];
            let rc = &self.raw_control[c * p..(c + 1) * p];
            if rf
                .iter()
                .zip(rc)
                .zip(&self.limits)
                .any(|((a, b), limit)| (a - b).abs() > *limit)
            {
                return false;
            }
        }
        let forbidden = &self.forbidden[row];
        forbidden.is_empty() || forbidden.binary_search(&c).is_err()
    }

    /// The tree, when it gives the scan's result for the weights `w`: every
    /// weighted sum of squares is then finite, since none exceeds the sum
    /// over the ranges of the variables.
    fn tree_for(&self, w: &[f64]) -> Option<&KdTree> {
        let tree = self.tree.as_ref()?;
        let mut largest = 0.0_f64;
        for (range, weight) in self.z_range.iter().zip(w) {
            largest += weight * (range * range);
        }
        largest.is_finite().then_some(tree)
    }

    /// As [`Self::nearest`], searched on `tree`, which holds the controls
    /// that are not taken.
    fn nearest_on(&self, tree: &KdTree, row: usize, w: &[f64], best: &mut Vec<(f64, usize)>) {
        let p = self.variables;
        let ratio = self.options.ratio;
        let zf = &self.z_focal[row * p..(row + 1) * p];
        *best = tree.nearest(&WeightedSquares(w), zf, ratio, None, &mut |c| {
            self.admits(row, c)
        });
        if best.len() < ratio {
            best.clear();
        }
    }

    /// The pairs as `(focal row, control row)`, in order of service and
    /// nearest first within a focal unit.
    fn pairs_by_row(&self, w: &[f64]) -> Vec<(usize, usize)> {
        match self.tree_for(w) {
            Some(tree) => self.pairs_on_tree(tree, w),
            None => self.pairs_by_scan(w),
        }
    }

    /// [`Self::pairs_by_row`] on the tree.
    fn pairs_on_tree(&self, tree: &KdTree, w: &[f64]) -> Vec<(usize, usize)> {
        let n_focal = self.focal_position.len();
        if self.options.replace {
            let per_focal: Vec<Vec<(f64, usize)>> = (0..n_focal)
                .into_par_iter()
                .map_init(Vec::new, |best, row| {
                    self.nearest_on(tree, row, w, best);
                    best.clone()
                })
                .collect();
            let mut pairs = Vec::new();
            for (row, list) in per_focal.iter().enumerate() {
                for &(_, c) in list {
                    pairs.push((row, c));
                }
            }
            pairs
        } else {
            let mut tree = tree.clone();
            let mut best = Vec::new();
            let mut pairs = Vec::new();
            for row in 0..n_focal {
                self.nearest_on(&tree, row, w, &mut best);
                for &(_, c) in &best {
                    tree.remove(c);
                    pairs.push((row, c));
                }
            }
            pairs
        }
    }

    /// [`Self::pairs_by_row`] by a scan of every control.
    fn pairs_by_scan(&self, w: &[f64]) -> Vec<(usize, usize)> {
        let n_focal = self.focal_position.len();
        if self.options.replace {
            let per_focal: Vec<Vec<(f64, usize)>> = (0..n_focal)
                .into_par_iter()
                .map_init(Vec::new, |best, row| {
                    self.nearest(row, w, None, false, best);
                    best.clone()
                })
                .collect();
            let mut pairs = Vec::new();
            for (row, list) in per_focal.iter().enumerate() {
                for &(_, c) in list {
                    pairs.push((row, c));
                }
            }
            pairs
        } else {
            let mut taken = vec![false; self.control_position.len()];
            let mut best = Vec::new();
            let mut pairs = Vec::new();
            for row in 0..n_focal {
                self.nearest(row, w, Some(&taken), true, &mut best);
                for &(_, c) in &best {
                    taken[c] = true;
                    pairs.push((row, c));
                }
            }
            pairs
        }
    }

    /// The loss of the match for `weights`. `seed` drives the bootstrap
    /// Kolmogorov-Smirnov p-values when `fit` asks for them and is otherwise
    /// unused. A weight vector that matches nothing, or too little for a
    /// t-test, has the worst loss there is.
    ///
    /// # Errors
    ///
    /// When the weights have the wrong length or a weight is negative or not
    /// finite.
    pub fn loss(
        &self,
        weights: &[f64],
        fit: &GeneticFit,
        seed: u64,
    ) -> Result<GeneticLoss, GeneticError> {
        self.check_weights(weights)?;
        Ok(self.loss_unchecked(weights, fit, seed))
    }

    fn loss_unchecked(&self, weights: &[f64], fit: &GeneticFit, seed: u64) -> GeneticLoss {
        let pairs = self.pairs_by_row(&self.effective_weights(weights));
        self.loss_of_pairs(&pairs, fit, seed)
    }

    fn loss_of_pairs(&self, pairs: &[(usize, usize)], fit: &GeneticFit, seed: u64) -> GeneticLoss {
        let q = self.balance_focal.len();
        if pairs.is_empty() {
            return balance::worst_loss(fit, q);
        }
        let mut weight = Vec::with_capacity(pairs.len());
        let mut start = 0;
        while start < pairs.len() {
            let mut end = start;
            while end < pairs.len() && pairs[end].0 == pairs[start].0 {
                end += 1;
            }
            let share = 1.0 / (end - start) as f64;
            weight.extend(std::iter::repeat_n(share, end - start));
            start = end;
        }
        let columns: Vec<(Vec<f64>, Vec<f64>)> = (0..q)
            .map(|k| {
                let treated = pairs
                    .iter()
                    .map(|&(f, _)| self.balance_focal[k][f])
                    .collect();
                let control = pairs
                    .iter()
                    .map(|&(_, c)| self.balance_control[k][c])
                    .collect();
                (treated, control)
            })
            .collect();
        let mut stream = Stream::seeded(seed);
        balance::loss_of_rows(fit, &columns, &weight, &self.few_values, &mut stream)
            .unwrap_or_else(|| balance::worst_loss(fit, q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize) -> (Vec<bool>, Vec<Vec<f64>>) {
        let mut stream = Stream::seeded(5);
        let focal: Vec<bool> = (0..n).map(|i| i % 4 == 0).collect();
        let columns = (0..3)
            .map(|_| (0..n).map(|_| stream.uniform() * 10.0).collect())
            .collect();
        (focal, columns)
    }

    fn problem(options: GeneticMatchOptions) -> (GeneticProblem, Vec<Vec<f64>>, Vec<bool>) {
        let (focal, columns) = data(60);
        let order: Vec<usize> = (0..focal.len()).collect();
        let problem = GeneticProblem::new(
            &focal,
            &columns,
            &columns,
            &order,
            options,
            f64::EPSILON.sqrt(),
        )
        .unwrap();
        (problem, columns, focal)
    }

    /// Distances written out from the definition, over the units in order.
    fn brute(
        columns: &[Vec<f64>],
        focal: &[bool],
        w: &[f64],
        replace: bool,
    ) -> Vec<(usize, usize)> {
        let n = focal.len();
        let z: Vec<Vec<f64>> = columns
            .iter()
            .map(|c| {
                let mean = c.iter().sum::<f64>() / n as f64;
                let sd =
                    (c.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
                c.iter().map(|v| (v - mean) / sd).collect()
            })
            .collect();
        let mut taken = vec![false; n];
        let mut pairs = Vec::new();
        for i in (0..n).filter(|&i| focal[i]) {
            let best = (0..n)
                .filter(|&j| !focal[j] && (replace || !taken[j]))
                .min_by(|&a, &b| {
                    let d = |j: usize| -> f64 {
                        (0..z.len())
                            .map(|k| w[k] * (z[k][i] - z[k][j]).powi(2))
                            .sum()
                    };
                    d(a).total_cmp(&d(b)).then(a.cmp(&b))
                })
                .unwrap();
            taken[best] = true;
            pairs.push((i, best));
        }
        pairs
    }

    #[test]
    fn one_to_one_pairs_follow_the_definition() {
        for replace in [false, true] {
            let (problem, columns, focal) = problem(GeneticMatchOptions {
                replace,
                ..GeneticMatchOptions::default()
            });
            let w = [1.0, 2.5, 0.4];
            let got = problem.match_pairs(&w).unwrap();
            let pairs: Vec<(usize, usize)> = got
                .focal
                .iter()
                .copied()
                .zip(got.control.iter().copied())
                .collect();
            assert_eq!(pairs, brute(&columns, &focal, &w, replace));
        }
    }

    #[test]
    fn calipers_exact_and_forbidden_pairs_are_held() {
        let (focal, columns) = data(60);
        let order: Vec<usize> = (0..60).collect();
        let is_focal = &focal;
        let forbidden: Vec<(usize, usize)> = (0..60)
            .filter(|&i| focal[i])
            .flat_map(|i| {
                (0..60)
                    .filter(move |&j| !is_focal[j] && (i + j) % 3 == 0)
                    .map(move |j| (i, j))
            })
            .collect();
        let options = GeneticMatchOptions {
            ratio: 1,
            replace: true,
            calipers: vec![Some(2.0), None, None],
            exact: vec![false, false, false],
            forbidden: forbidden.clone(),
        };
        let problem =
            GeneticProblem::new(&focal, &columns, &columns, &order, options, 1e-8).unwrap();
        let got = problem.match_pairs(&[1.0, 1.0, 1.0]).unwrap();
        assert!(!got.is_empty());
        for (f, c) in got.focal.iter().zip(&got.control) {
            assert!((columns[0][*f] - columns[0][*c]).abs() <= 2.0);
            assert!(!forbidden.contains(&(*f, *c)));
        }
    }

    #[test]
    fn a_unit_without_all_its_controls_gets_none() {
        let (problem, ..) = problem(GeneticMatchOptions {
            ratio: 3,
            replace: false,
            ..GeneticMatchOptions::default()
        });
        // 15 focal units and 45 controls: every focal unit gets three.
        let got = problem.match_pairs(&[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(got.len(), 45);
        let (problem, ..) = problem_with_ratio(4);
        // 15 focal units want 60 controls of 45: the first eleven take 44.
        let got = problem.match_pairs(&[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(got.len(), 44);
    }

    fn problem_with_ratio(ratio: usize) -> (GeneticProblem, Vec<Vec<f64>>, Vec<bool>) {
        problem(GeneticMatchOptions {
            ratio,
            ..GeneticMatchOptions::default()
        })
    }

    #[test]
    fn a_split_scan_gives_the_list_of_a_single_scan() {
        // Many controls on a coarse grid, so that distances tie across the
        // chunk boundaries and the order among ties is tested.
        let n = 20_000;
        let mut stream = Stream::seeded(11);
        let focal: Vec<bool> = (0..n).map(|i| i < 40).collect();
        let columns: Vec<Vec<f64>> = (0..2)
            .map(|_| {
                (0..n)
                    .map(|_| f64::from(u8::try_from(stream.below(6)).unwrap()))
                    .collect()
            })
            .collect();
        let order: Vec<usize> = (0..n).collect();
        let problem = GeneticProblem::new(
            &focal,
            &columns,
            &columns,
            &order,
            GeneticMatchOptions {
                ratio: 3,
                ..GeneticMatchOptions::default()
            },
            1e-8,
        )
        .unwrap();
        let w = [1.0, 2.0];
        let mut taken = vec![false; n - 40];
        for c in (0..taken.len()).step_by(7) {
            taken[c] = true;
        }
        for row in 0..40 {
            let (mut one, mut many) = (Vec::new(), Vec::new());
            problem.nearest(row, &w, Some(&taken), false, &mut one);
            problem.nearest(row, &w, Some(&taken), true, &mut many);
            assert_eq!(one, many, "focal unit {row}");
            assert_eq!(one.len(), 3);
        }
    }

    /// The k-d tree gives the pairs of the scan of every control: on a coarse
    /// grid where many distances tie, in two to four variables, with ratios 1
    /// and 3, with and without replacement, with a caliper and forbidden
    /// pairs, and for weights that include a zero.
    #[test]
    fn the_tree_gives_the_pairs_of_the_scan() {
        let n = 3_000;
        let mut stream = Stream::seeded(17);
        for p in 2..=4 {
            let focal: Vec<bool> = (0..n).map(|i| i % 5 == 0).collect();
            let columns: Vec<Vec<f64>> = (0..p)
                .map(|_| {
                    (0..n)
                        .map(|_| f64::from(u8::try_from(stream.below(5)).unwrap()))
                        .collect()
                })
                .collect();
            let order: Vec<usize> = (0..n).rev().collect();
            let forbidden: Vec<(usize, usize)> = (0..n)
                .step_by(5)
                .flat_map(|i| (1..n).step_by(13).map(move |j| (i, j + (i % 3))))
                .filter(|&(_, j)| j < n && j % 5 != 0)
                .collect();
            for (ratio, replace, constrained) in [
                (1, false, false),
                (3, false, true),
                (1, true, true),
                (3, true, false),
            ] {
                let options = GeneticMatchOptions {
                    ratio,
                    replace,
                    calipers: if constrained {
                        (0..p).map(|k| (k == 0).then_some(1.0)).collect()
                    } else {
                        Vec::new()
                    },
                    exact: Vec::new(),
                    forbidden: if constrained {
                        forbidden.clone()
                    } else {
                        Vec::new()
                    },
                };
                let problem =
                    GeneticProblem::new(&focal, &columns, &columns, &order, options, 1e-8).unwrap();
                for w in [
                    vec![1.0; p],
                    (0..p).map(|k| [0.0, 2.5, 7.0][k % 3]).collect(),
                ] {
                    let tree = problem.tree_for(&w).expect("a tree");
                    let scan = problem.pairs_by_scan(&w);
                    assert!(scan.len() > 100);
                    assert_eq!(
                        problem.pairs_on_tree(tree, &w),
                        scan,
                        "{p} variables, ratio {ratio}, replace {replace}, constrained {constrained}"
                    );
                }
            }
        }
    }

    #[test]
    fn weights_that_could_overflow_are_left_to_the_scan() {
        let (problem, ..) = problem(GeneticMatchOptions::default());
        assert!(
            problem.tree_for(&[1.0, 1.0, 1.0]).is_none(),
            "too few controls"
        );
        let (focal, columns) = data(2_000);
        let order: Vec<usize> = (0..2_000).collect();
        let problem = GeneticProblem::new(
            &focal,
            &columns,
            &columns,
            &order,
            GeneticMatchOptions::default(),
            1e-8,
        )
        .unwrap();
        assert!(problem.tree_for(&[1.0, 1.0, 1.0]).is_some());
        assert!(problem.tree_for(&[1.0, f64::MAX, 1.0]).is_none());
    }

    #[test]
    fn invalid_input_is_an_error() {
        let (focal, columns) = data(20);
        let order: Vec<usize> = (0..20).collect();
        let make = |options: GeneticMatchOptions, order: &[usize]| {
            GeneticProblem::new(&focal, &columns, &columns, order, options, 1e-8)
        };
        assert!(
            make(
                GeneticMatchOptions {
                    ratio: 0,
                    ..Default::default()
                },
                &order
            )
            .is_err()
        );
        assert!(make(GeneticMatchOptions::default(), &[0, 0, 1]).is_err());
        assert!(make(GeneticMatchOptions::default(), &[1, 2, 3]).is_err());
        let (problem, ..) = problem(GeneticMatchOptions::default());
        assert!(problem.match_pairs(&[1.0, 1.0]).is_err());
        assert!(problem.match_pairs(&[1.0, -1.0, 1.0]).is_err());
        assert!(problem.match_pairs(&[1.0, f64::NAN, 1.0]).is_err());
    }

    #[test]
    fn search_never_does_worse_than_its_start_and_is_reproducible() {
        let (problem, ..) = problem(GeneticMatchOptions::default());
        let fit = GeneticFit::default();
        let search = GeneticSearch {
            population: 20,
            max_generations: 5,
            hard_generation_limit: true,
            seed: 3,
            ..GeneticSearch::default()
        };
        let first = problem.search(&fit, &search).unwrap();
        let second = problem.search(&fit, &search).unwrap();
        assert_eq!(first, second);
        let start = problem.loss(&[1.0, 1.0, 1.0], &fit, 0).unwrap();
        assert_ne!(first.loss.compare(&start, 0.0), std::cmp::Ordering::Less);
        for pair in first.history.windows(2) {
            assert_ne!(pair[1].compare(&pair[0], 0.0), std::cmp::Ordering::Less);
        }
        assert!(
            problem
                .search(
                    &fit,
                    &GeneticSearch {
                        population: 1,
                        ..search.clone()
                    }
                )
                .is_err()
        );
    }
}
