//! The genetic search for weights.

use epistat_rng::{Stream, replicate_seed};
use rayon::prelude::*;

use super::{GeneticError, GeneticFit, GeneticLoss, GeneticMatches, GeneticProblem};

/// The settings of the search, with `GenMatch`'s defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneticSearch {
    /// Individuals per generation. `pop.size`, 100 by default. 0 leaves only
    /// the starting values, evaluated once.
    pub population: usize,
    /// Generations to run. `max.generations`, 100 by default. The limit binds
    /// only with `hard_generation_limit`; without it the search runs until
    /// `wait_generations` pass without an improvement.
    pub max_generations: usize,
    /// Generations without an improvement after which the search stops.
    /// `wait.generations`, 4 by default.
    pub wait_generations: usize,
    /// Whether `max_generations` binds. `hard.generation.limit`.
    pub hard_generation_limit: bool,
    /// Smallest weight. `min.weight`, 0 by default.
    pub min_weight: f64,
    /// Largest weight. `max.weight`, 1000 by default.
    pub max_weight: f64,
    /// The weights of the first individual. All ones by default.
    pub starting_values: Option<Vec<f64>>,
    /// Two losses are equal when they differ by at most this in every entry,
    /// which is when a generation counts as no improvement. 1e-6, as
    /// `solution.tolerance` of `genoud`.
    pub solution_tolerance: f64,
    /// Seed of the random draws.
    pub seed: u64,
}

impl Default for GeneticSearch {
    fn default() -> Self {
        Self {
            population: 100,
            max_generations: 100,
            wait_generations: 4,
            hard_generation_limit: false,
            min_weight: 0.0,
            max_weight: 1000.0,
            starting_values: None,
            solution_tolerance: 1e-6,
            seed: 0,
        }
    }
}

/// What a search found.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneticOutcome {
    /// The best weights, one per matching variable.
    pub weights: Vec<f64>,
    /// Their loss.
    pub loss: GeneticLoss,
    /// The match for these weights.
    pub matches: GeneticMatches,
    /// Generations run, not counting the initial population.
    pub generations: usize,
    /// Losses computed.
    pub evaluations: usize,
    /// The best loss after the initial population and after each generation.
    pub history: Vec<GeneticLoss>,
}

#[derive(Clone)]
struct Individual {
    weights: Vec<f64>,
    loss: GeneticLoss,
}

/// The eight operators of the search. Each makes the same share of a
/// generation, as `genoud`'s defaults give them equal weights.
#[derive(Clone, Copy)]
enum Operator {
    Cloning,
    UniformMutation,
    BoundaryMutation,
    NonUniformMutation,
    PolytopeCrossover,
    SimpleCrossover,
    WholeNonUniformMutation,
    HeuristicCrossover,
}

const OPERATORS: [Operator; 8] = [
    Operator::Cloning,
    Operator::UniformMutation,
    Operator::BoundaryMutation,
    Operator::NonUniformMutation,
    Operator::PolytopeCrossover,
    Operator::SimpleCrossover,
    Operator::WholeNonUniformMutation,
    Operator::HeuristicCrossover,
];

/// The shape parameter of the non-uniform mutations, 6 in `genoud`.
const NON_UNIFORM_SHAPE: i32 = 6;

struct Operators<'a> {
    low: f64,
    high: f64,
    stream: &'a mut Stream,
    generation: usize,
    limit: usize,
}

impl Operators<'_> {
    fn uniform_in(&mut self) -> f64 {
        self.low + (self.high - self.low) * self.stream.uniform()
    }

    /// The decrease with the generation of the non-uniform step, in `[0, 1]`.
    fn shrink(&mut self, range: f64) -> f64 {
        let progress = if self.limit == 0 {
            1.0
        } else {
            (self.generation as f64 / self.limit as f64).min(1.0)
        };
        let r = self.stream.uniform();
        range * (1.0 - r.powf((1.0 - progress).powi(NON_UNIFORM_SHAPE)))
    }

    fn non_uniform_step(&mut self, value: f64) -> f64 {
        if self.stream.uniform() < 0.5 {
            value + self.shrink(self.high - value)
        } else {
            value - self.shrink(value - self.low)
        }
    }

    fn apply(&mut self, operator: Operator, parents: &[&Vec<f64>]) -> Vec<Vec<f64>> {
        let p = parents[0].len();
        match operator {
            Operator::Cloning => vec![parents[0].clone()],
            Operator::UniformMutation => {
                let mut child = parents[0].clone();
                let k = self.stream.below(p);
                child[k] = self.uniform_in();
                vec![child]
            }
            Operator::BoundaryMutation => {
                let mut child = parents[0].clone();
                let k = self.stream.below(p);
                child[k] = if self.stream.uniform() < 0.5 {
                    self.low
                } else {
                    self.high
                };
                vec![child]
            }
            Operator::NonUniformMutation => {
                let mut child = parents[0].clone();
                let k = self.stream.below(p);
                child[k] = self.non_uniform_step(child[k]);
                vec![child]
            }
            Operator::WholeNonUniformMutation => {
                let mut child = parents[0].clone();
                for value in &mut child {
                    *value = self.non_uniform_step(*value);
                }
                vec![child]
            }
            Operator::PolytopeCrossover => {
                let draws: Vec<f64> = parents.iter().map(|_| self.stream.uniform()).collect();
                let mut total = 0.0;
                for d in &draws {
                    total += d;
                }
                if total <= 0.0 {
                    return vec![parents[0].clone()];
                }
                let mut child = vec![0.0; p];
                for (parent, d) in parents.iter().zip(&draws) {
                    for k in 0..p {
                        child[k] += parent[k] * (d / total);
                    }
                }
                vec![child]
            }
            Operator::SimpleCrossover => {
                let (a, b) = (parents[0], parents[1]);
                if p < 2 {
                    return vec![a.clone(), b.clone()];
                }
                let cut = 1 + self.stream.below(p - 1);
                let mut first = a.clone();
                let mut second = b.clone();
                first[cut..].copy_from_slice(&b[cut..]);
                second[cut..].copy_from_slice(&a[cut..]);
                vec![first, second]
            }
            Operator::HeuristicCrossover => {
                // `parents[0]` is the better of the two.
                let (better, worse) = (parents[0], parents[1]);
                for _ in 0..10 {
                    let r = self.stream.uniform();
                    let child: Vec<f64> = better
                        .iter()
                        .zip(worse)
                        .map(|(a, b)| a + r * (a - b))
                        .collect();
                    if child.iter().all(|v| *v >= self.low && *v <= self.high) {
                        return vec![child];
                    }
                }
                vec![better.clone()]
            }
        }
    }
}

impl GeneticProblem {
    /// Searches for the weights with the best loss.
    ///
    /// The best of the starting values and the individuals of every generation
    /// is kept, so the result is never worse than the starting values. The
    /// run, including the order of the draws, is a function of `search.seed`
    /// alone. The search is not checked against R step by step, see the module
    /// documentation.
    ///
    /// # Errors
    ///
    /// When the settings are invalid: a population of 1, an empty or
    /// reversed weight range, or starting values of the wrong length or
    /// outside the range.
    pub fn search(
        &self,
        fit: &GeneticFit,
        search: &GeneticSearch,
    ) -> Result<GeneticOutcome, GeneticError> {
        let p = self.variables;
        if search.population == 1 {
            return Err(GeneticError::Invalid(
                "a population of 1 cannot recombine; use 0 or at least 2".into(),
            ));
        }
        if !(search.min_weight.is_finite()
            && search.max_weight.is_finite()
            && search.min_weight >= 0.0
            && search.min_weight < search.max_weight)
        {
            return Err(GeneticError::Invalid(
                "the weights need 0 <= min_weight < max_weight, both finite".into(),
            ));
        }
        let start = match &search.starting_values {
            Some(values) => {
                self.check_weights(values)?;
                if values
                    .iter()
                    .any(|v| *v < search.min_weight || *v > search.max_weight)
                {
                    return Err(GeneticError::Invalid(
                        "the starting values must lie in [min_weight, max_weight]".into(),
                    ));
                }
                values.clone()
            }
            None => vec![1.0_f64.clamp(search.min_weight, search.max_weight); p],
        };
        let mut stream = Stream::seeded(search.seed);
        let seed_of = |generation: usize, index: usize| {
            replicate_seed(replicate_seed(search.seed, generation as u64), index as u64)
        };
        let evaluate = |weights: Vec<f64>, generation: usize, index: usize| Individual {
            loss: self.loss_unchecked(&weights, fit, seed_of(generation, index)),
            weights,
        };
        let tol = search.solution_tolerance;

        // The initial population: the starting values, then draws.
        let mut initial = vec![start];
        for _ in 1..search.population {
            let draw: Vec<f64> = (0..p)
                .map(|_| {
                    search.min_weight + (search.max_weight - search.min_weight) * stream.uniform()
                })
                .collect();
            initial.push(draw);
        }
        let mut population: Vec<Individual> = initial
            .into_par_iter()
            .enumerate()
            .map(|(index, weights)| evaluate(weights, 0, index))
            .collect();
        let mut evaluations = population.len();
        sort_best_first(&mut population);
        let mut best = population[0].clone();
        let mut history = vec![best.loss.clone()];
        let mut idle = 0_usize;
        let mut generation = 0_usize;

        loop {
            if search.population < 2
                || idle >= search.wait_generations
                || (search.hard_generation_limit && generation >= search.max_generations)
                || (search.max_generations == 0 && generation >= 1)
            {
                break;
            }
            generation += 1;
            let size = search.population;
            let children_wanted = size - 1;
            let mut ops = Operators {
                low: search.min_weight,
                high: search.max_weight,
                stream: &mut stream,
                generation,
                limit: search.max_generations,
            };
            // Equal shares; the remainder goes to cloning.
            let share = children_wanted / OPERATORS.len();
            let mut counts = [share; 8];
            counts[0] += children_wanted - share * OPERATORS.len();

            let mut children: Vec<(Vec<f64>, Option<GeneticLoss>)> =
                Vec::with_capacity(children_wanted);
            for (operator, &count) in OPERATORS.iter().zip(&counts) {
                let mut made = 0;
                while made < count {
                    let need = match operator {
                        Operator::PolytopeCrossover => (p + 1).min(size),
                        Operator::SimpleCrossover | Operator::HeuristicCrossover => 2,
                        _ => 1,
                    };
                    let mut chosen: Vec<usize> =
                        (0..need).map(|_| tournament(ops.stream, size)).collect();
                    if matches!(operator, Operator::HeuristicCrossover) {
                        chosen.sort_unstable();
                    }
                    let parents: Vec<&Vec<f64>> =
                        chosen.iter().map(|&i| &population[i].weights).collect();
                    for child in ops.apply(*operator, &parents) {
                        if made == count {
                            break;
                        }
                        let known = matches!(operator, Operator::Cloning)
                            .then(|| population[chosen[0]].loss.clone());
                        children.push((child, known));
                        made += 1;
                    }
                }
            }

            let scored: Vec<Individual> = children
                .into_par_iter()
                .enumerate()
                .map(|(index, (weights, known))| match known {
                    Some(loss) => Individual { weights, loss },
                    None => evaluate(weights, generation, index + 1),
                })
                .collect();
            evaluations += scored.len().saturating_sub(counts[0]);
            let mut next = Vec::with_capacity(size);
            next.push(population[0].clone());
            next.extend(scored);
            population = next;
            sort_best_first(&mut population);

            if population[0].loss.compare(&best.loss, tol) == std::cmp::Ordering::Greater {
                idle = 0;
            } else {
                idle += 1;
            }
            if population[0].loss.compare(&best.loss, 0.0) == std::cmp::Ordering::Greater {
                best = population[0].clone();
            }
            history.push(best.loss.clone());
        }

        let matches = self.match_pairs(&best.weights)?;
        Ok(GeneticOutcome {
            weights: best.weights,
            loss: best.loss,
            matches,
            generations: generation,
            evaluations,
            history,
        })
    }
}

/// Binary tournament on a population sorted best first: the better of two
/// uniform draws, which is the smaller index.
fn tournament(stream: &mut Stream, size: usize) -> usize {
    let a = stream.below(size);
    let b = stream.below(size);
    a.min(b)
}

fn sort_best_first(population: &mut [Individual]) {
    // Stable, so equal losses keep their order and the run is reproducible.
    population.sort_by(|a, b| b.loss.compare(&a.loss, 0.0));
}
