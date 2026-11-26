// The algorithm follows argmin's NelderMead (MIT OR Apache-2.0).
//! Nelder-Mead simplex minimiser.
//!
//! The caller supplies the starting simplex of `n + 1` vertices in `n`
//! dimensions. Each iteration replaces the worst vertex by a reflected, expanded
//! or contracted point, or shrinks the whole simplex toward the best vertex.
//! The method needs only function values \[1\]. Its convergence to a minimiser is
//! proved only for strictly convex functions in one and two dimensions \[2\].
//!
//! # Algorithm
//!
//! Order the vertices so that `f(x_0) <= f(x_1) <= ... <= f(x_n)`. Let `c` be the
//! centroid of every vertex but the worst:
//!
//! ```text
//! c = (x_0 + ... + x_{n-1}) / n
//! ```
//!
//! The coefficients are `α` for reflection, `γ` for expansion, `ρ` for
//! contraction and `σ` for shrinkage, with defaults 1, 2, 0.5 and 0.5, the
//! standard values \[1\] \[2\]. The reflected point is
//!
//! ```text
//! x_r = c + α (c - x_n)
//! ```
//!
//! The tests are made in this order, with `f_r = f(x_r)`:
//!
//! 1. `f_0 <= f_r < f_{n-1}`: `x_r` replaces `x_n`.
//! 2. `f_r < f_0`: the expanded point `x_e = c + γ (x_r - c)` is evaluated. It
//!    replaces `x_n` if `f_e < f_r`, and otherwise `x_r` does.
//! 3. `f_r >= f_{n-1}`: a contraction is tried.
//!    - If `f_r < f_n`, the outside contraction `x_c = c + ρ (x_r - c)` replaces
//!      `x_n` when `f_c <= f_r`.
//!    - Otherwise the inside contraction `x_c = c + ρ (x_n - c)` replaces `x_n`
//!      when `f_c < f_n`.
//!    - If the contraction is not accepted, the simplex shrinks: every vertex but
//!      the best moves to `x_i = x_0 + σ (x_i - x_0)` and is evaluated again.
//! 4. Otherwise `f_r` cannot be ordered, as when it is NaN, and the search fails
//!    with [`NelderMeadError::Unordered`].
//!
//! The vertices are sorted again after every step. The sort treats NaN costs of
//! the starting vertices as equal to every other value.
//!
//! # Stopping rules
//!
//! Before each iteration the sample standard deviation of the `n + 1` vertex
//! costs, with divisor `n`, is compared with `sd_tolerance`, by default
//! `f64::EPSILON`. Below it the search stops and is converged. It also stops, not
//! converged, after `max_iterations` iterations, by default 1000, or when the best
//! cost is `-∞`. The test is made before the first iteration, so a simplex whose
//! vertices have equal costs returns at once with 0 iterations.
//!
//! The standard deviation measures the spread of the costs and not of the
//! vertices, so a simplex on a flat ridge can stop while the vertices are far
//! apart.
//!
//! # Result
//!
//! [`Minimum`] holds the best point seen. It is replaced only when a vertex has a
//! strictly lower cost, so on ties the earlier point is kept. Infinite costs are
//! handled so that the first vertex is accepted when every cost is `+∞`.
//!
//! # References
//!
//! 1. Nelder JA, Mead R. A simplex method for function minimization. Comput J.
//!    1965;7(4):308-13. doi:10.1093/comjnl/7.4.308
//! 2. Lagarias JC, Reeds JA, Wright MH, Wright PE. Convergence properties of the
//!    Nelder-Mead simplex method in low dimensions. SIAM J Optim.
//!    1998;9(1):112-47. doi:10.1137/S1052623496303470

use std::fmt;

/// Settings for the Nelder-Mead method.
#[derive(Debug, Clone, PartialEq)]
pub struct NelderMead {
    /// Stop when the sample standard deviation of the vertex costs is below
    /// this value. Default `f64::EPSILON`.
    pub sd_tolerance: f64,
    /// Maximum number of iterations. Default 1000.
    pub max_iterations: usize,
    /// Reflection coefficient. Default 1.
    pub alpha: f64,
    /// Expansion coefficient. Default 2.
    pub gamma: f64,
    /// Contraction coefficient. Default 0.5.
    pub rho: f64,
    /// Shrink coefficient. Default 0.5.
    pub sigma: f64,
}

impl Default for NelderMead {
    fn default() -> Self {
        Self {
            sd_tolerance: f64::EPSILON,
            max_iterations: 1000,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

/// Result of a minimisation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Minimum {
    /// Best point found.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub value: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of evaluations of the objective, the starting vertices
    /// included.
    pub evaluations: usize,
    /// True if the standard deviation criterion was met, false if the
    /// iteration cap stopped the search.
    pub converged: bool,
}

/// Errors from [`NelderMead::minimize`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NelderMeadError {
    /// The simplex has no vertices, or its vertices have no coordinates.
    EmptySimplex,
    /// The vertices do not all have the same length.
    RaggedSimplex,
    /// The number of vertices is not the dimension plus one.
    WrongVertexCount {
        /// Number of vertices supplied.
        vertices: usize,
        /// Dimension of the vertices.
        dimension: usize,
    },
    /// A trial point had an objective value that cannot be ordered, such as
    /// NaN.
    Unordered,
}

impl fmt::Display for NelderMeadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySimplex => write!(f, "the simplex is empty"),
            Self::RaggedSimplex => write!(f, "the simplex vertices differ in length"),
            Self::WrongVertexCount {
                vertices,
                dimension,
            } => write!(
                f,
                "a simplex in {dimension} dimensions needs {} vertices, got {vertices}",
                dimension + 1
            ),
            Self::Unordered => write!(f, "the objective returned a value that cannot be ordered"),
        }
    }
}

impl std::error::Error for NelderMeadError {}

struct Vertex {
    x: Vec<f64>,
    cost: f64,
}

/// Point `a + (b - a) * k`.
fn toward(a: &[f64], b: &[f64], k: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(&a, &b)| a + (b - a) * k).collect()
}

fn sort_by_cost(vertices: &mut [Vertex]) {
    vertices.sort_by(|a, b| {
        a.cost
            .partial_cmp(&b.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Sample standard deviation of the vertex costs.
fn cost_sd(vertices: &[Vertex]) -> f64 {
    let n = vertices.len() as f64;
    let mean = vertices.iter().map(|v| v.cost).sum::<f64>() / n;
    let squares = vertices
        .iter()
        .map(|v| (v.cost - mean) * (v.cost - mean))
        .sum::<f64>();
    ((1.0 / (n - 1.0)) * squares).sqrt()
}

/// Best point seen so far, replaced only by a strictly lower cost.
struct Best {
    x: Vec<f64>,
    cost: f64,
}

impl Best {
    fn offer(&mut self, vertex: &Vertex) {
        if vertex.cost < self.cost
            || (self.cost.is_infinite()
                && vertex.cost.is_infinite()
                && self.cost.is_sign_positive() == vertex.cost.is_sign_positive())
        {
            self.x.clone_from(&vertex.x);
            self.cost = vertex.cost;
        }
    }
}

impl NelderMead {
    /// Minimises `f` starting from `simplex`, which must hold `n + 1`
    /// vertices of length `n`.
    ///
    /// # Errors
    ///
    /// Returns an error if the simplex is empty, ragged or has the wrong
    /// number of vertices, or if the objective returns NaN at a trial point.
    pub fn minimize(
        &self,
        simplex: Vec<Vec<f64>>,
        mut f: impl FnMut(&[f64]) -> f64,
    ) -> Result<Minimum, NelderMeadError> {
        let mut evaluations = 0;
        let mut minimum = self.search_from(simplex, |x| {
            evaluations += 1;
            f(x)
        })?;
        minimum.evaluations = evaluations;
        Ok(minimum)
    }

    fn search_from(
        &self,
        simplex: Vec<Vec<f64>>,
        mut f: impl FnMut(&[f64]) -> f64,
    ) -> Result<Minimum, NelderMeadError> {
        let dimension = simplex.first().map_or(0, Vec::len);
        if dimension == 0 {
            return Err(NelderMeadError::EmptySimplex);
        }
        if simplex.iter().any(|v| v.len() != dimension) {
            return Err(NelderMeadError::RaggedSimplex);
        }
        if simplex.len() != dimension + 1 {
            return Err(NelderMeadError::WrongVertexCount {
                vertices: simplex.len(),
                dimension,
            });
        }

        let mut vertices: Vec<Vertex> = simplex
            .into_iter()
            .map(|x| {
                let cost = f(&x);
                Vertex { x, cost }
            })
            .collect();
        sort_by_cost(&mut vertices);
        let mut best = Best {
            x: vertices[0].x.clone(),
            cost: f64::INFINITY,
        };
        best.offer(&vertices[0]);

        let mut iterations = 0;
        let mut converged = false;
        loop {
            if cost_sd(&vertices) < self.sd_tolerance {
                converged = true;
                break;
            }
            if iterations >= self.max_iterations || best.cost <= f64::NEG_INFINITY {
                break;
            }

            self.step(&mut vertices, &mut f)?;
            sort_by_cost(&mut vertices);
            best.offer(&vertices[0]);
            iterations += 1;
        }

        Ok(Minimum {
            x: best.x,
            value: best.cost,
            iterations,
            evaluations: 0,
            converged,
        })
    }

    /// One iteration: replaces the worst vertex, or shrinks toward the best.
    /// The vertices must be sorted by cost on entry.
    fn step(
        &self,
        vertices: &mut [Vertex],
        f: &mut impl FnMut(&[f64]) -> f64,
    ) -> Result<(), NelderMeadError> {
        // Centroid of every vertex but the worst.
        let last = vertices.len() - 1;
        let scale = 1.0 / (last as f64);
        let mut centroid = vertices[0].x.clone();
        for v in &vertices[1..last] {
            for (c, &x) in centroid.iter_mut().zip(&v.x) {
                *c += x;
            }
        }
        for c in &mut centroid {
            *c *= scale;
        }

        let best_cost = vertices[0].cost;
        let second_worst_cost = vertices[last - 1].cost;
        let worst_cost = vertices[last].cost;

        let reflected: Vec<f64> = centroid
            .iter()
            .zip(&vertices[last].x)
            .map(|(&c, &w)| c + (c - w) * self.alpha)
            .collect();
        let reflected_cost = f(&reflected);

        if reflected_cost < second_worst_cost && reflected_cost >= best_cost {
            vertices[last] = Vertex {
                x: reflected,
                cost: reflected_cost,
            };
        } else if reflected_cost < best_cost {
            let expanded = toward(&centroid, &reflected, self.gamma);
            let expanded_cost = f(&expanded);
            vertices[last] = if expanded_cost < reflected_cost {
                Vertex {
                    x: expanded,
                    cost: expanded_cost,
                }
            } else {
                Vertex {
                    x: reflected,
                    cost: reflected_cost,
                }
            };
        } else if reflected_cost >= second_worst_cost {
            // Outside contraction toward the reflected point, or inside
            // contraction toward the worst vertex.
            let outside = reflected_cost < worst_cost;
            let (from, reference) = if outside {
                (&reflected, reflected_cost)
            } else {
                (&vertices[last].x, worst_cost)
            };
            let contracted = toward(&centroid, from, self.rho);
            let contracted_cost = f(&contracted);
            let accepted = if outside {
                contracted_cost <= reference
            } else {
                contracted_cost < reference
            };
            if accepted {
                vertices[last] = Vertex {
                    x: contracted,
                    cost: contracted_cost,
                };
            } else {
                let origin = vertices[0].x.clone();
                for v in &mut vertices[1..] {
                    v.x = toward(&origin, &v.x, self.sigma);
                    v.cost = f(&v.x);
                }
            }
        } else {
            return Err(NelderMeadError::Unordered);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(p: &[f64]) -> f64 {
        let (x, y) = (p[0], p[1]);
        (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)
    }

    #[test]
    fn rosenbrock_converges_near_one() {
        let nm = NelderMead {
            sd_tolerance: 1e-12,
            max_iterations: 2000,
            ..NelderMead::default()
        };
        let simplex = vec![vec![-1.2, 1.0], vec![-1.0, 1.0], vec![-1.2, 1.2]];
        let m = nm.minimize(simplex, rosenbrock).unwrap();
        assert!(m.converged);
        assert!((m.x[0] - 1.0).abs() < 1e-4, "{:?}", m.x);
        assert!((m.x[1] - 1.0).abs() < 1e-4, "{:?}", m.x);
    }

    #[test]
    fn quadratic_in_three_dimensions() {
        let target = [1.0, -2.0, 3.0];
        let simplex = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let nm = NelderMead {
            sd_tolerance: 1e-14,
            ..NelderMead::default()
        };
        let m = nm
            .minimize(simplex, |p| {
                p.iter().zip(&target).map(|(a, b)| (a - b) * (a - b)).sum()
            })
            .unwrap();
        assert!(m.converged);
        for (a, b) in m.x.iter().zip(&target) {
            assert!((a - b).abs() < 1e-5, "{:?}", m.x);
        }
    }

    #[test]
    fn iteration_cap_stops_early() {
        let nm = NelderMead {
            max_iterations: 3,
            sd_tolerance: 1e-12,
            ..NelderMead::default()
        };
        let simplex = vec![vec![-1.2, 1.0], vec![-1.0, 1.0], vec![-1.2, 1.2]];
        let mut calls = 0;
        let m = nm
            .minimize(simplex, |p| {
                calls += 1;
                rosenbrock(p)
            })
            .unwrap();
        assert!(!m.converged);
        assert_eq!(m.iterations, 3);
        assert_eq!(m.evaluations, calls);
        assert!(m.evaluations >= 3 + 3);
    }

    #[test]
    fn rejects_bad_simplices() {
        let nm = NelderMead::default();
        let f = |p: &[f64]| p[0];
        assert_eq!(nm.minimize(vec![], f), Err(NelderMeadError::EmptySimplex));
        assert_eq!(
            nm.minimize(vec![vec![], vec![]], f),
            Err(NelderMeadError::EmptySimplex)
        );
        assert_eq!(
            nm.minimize(vec![vec![0.0, 1.0], vec![1.0], vec![2.0, 2.0]], f),
            Err(NelderMeadError::RaggedSimplex)
        );
        assert_eq!(
            nm.minimize(vec![vec![0.0, 1.0], vec![1.0, 0.0]], f),
            Err(NelderMeadError::WrongVertexCount {
                vertices: 2,
                dimension: 2
            })
        );
        assert!(NelderMeadError::EmptySimplex.to_string().contains("empty"));
    }
}
