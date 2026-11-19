use epistat_linalg::{Attempt, Factor, SolveError, weighted_gram, weighted_xtz};
use faer::{Mat, MatRef};

use crate::{Family, Link};

/// A model and its fitting controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm {
    pub family: Family,
    pub link: Link,
    /// R's `epsilon`.
    pub tolerance: f64,
    /// R's `maxit`.
    pub max_iterations: usize,
    /// Added to the diagonal of `X'WX` for every coefficient but the first,
    /// which is 0 in R.
    pub ridge: f64,
}

impl Glm {
    #[must_use]
    pub const fn new(family: Family, link: Link) -> Self {
        Self {
            family,
            link,
            tolerance: 1e-8,
            max_iterations: 25,
            ridge: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GlmError {
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    #[error("response {index} is {value}, outside the family's support")]
    Response { index: usize, value: f64 },
    #[error("no valid coefficients were found; supply starting values")]
    NoValidStart,
    #[error("the step could not be corrected by halving")]
    StepHalving,
    #[error(transparent)]
    Solve(#[from] SolveError),
}

/// A fitted model.
#[derive(Clone, Debug)]
pub struct Fit {
    pub coefficients: Vec<f64>,
    pub fitted: Vec<f64>,
    pub linear_predictor: Vec<f64>,
    pub deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    /// Whether a step was halved in the last iteration.
    pub boundary: bool,
    /// The IRLS weights `w mu'(eta)^2 / V(mu)` of the last iteration, which
    /// R reports as the fit's weights.
    pub working_weights: Vec<f64>,
    /// The Pearson dispersion, or 1 for the binomial and Poisson families.
    pub dispersion: f64,
    /// `dispersion * (X'WX)^-1` with the last iteration's weights, as R's
    /// `summary.glm` computes it, row-major.
    pub covariance: Vec<f64>,
}

impl Fit {
    /// The square roots of the covariance diagonal.
    #[must_use]
    pub fn standard_errors(&self) -> Vec<f64> {
        let p = self.coefficients.len();
        (0..p)
            .map(|i| self.covariance[i * p + i].max(0.0).sqrt())
            .collect()
    }
}

impl Glm {
    /// Fits the model to `y` on the columns of `x`.
    ///
    /// `weights` are prior weights (1 when `None`), rows with weight 0 do not
    /// enter the fit. `offset` is added to the linear predictor. `start`
    /// replaces the family's starting means with starting coefficients.
    ///
    /// # Errors
    ///
    /// See [`GlmError`].
    pub fn fit(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<Fit, GlmError> {
        let (n, p) = (x.nrows(), x.ncols());
        let ones = vec![1.0; n];
        let zeros = vec![0.0; n];
        let weights = weights.unwrap_or(&ones);
        let offset = offset.unwrap_or(&zeros);
        check("y", n, y.len())?;
        check("weights", n, weights.len())?;
        check("offset", n, offset.len())?;
        if let Some(start) = start {
            check("start", p, start.len())?;
        }
        if let Some((index, &value)) = y
            .iter()
            .enumerate()
            .find(|&(_, &v)| !self.family.valid_response(v))
        {
            return Err(GlmError::Response { index, value });
        }
        let (family, link) = (self.family, self.link);
        let deviance = |mu: &[f64]| -> f64 {
            (0..n)
                .map(|i| family.deviance(y[i], mu[i], weights[i]))
                .sum()
        };
        let predict = |beta: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let eta: Vec<f64> = (0..n)
                .map(|i| (0..p).map(|j| x[(i, j)] * beta[j]).sum::<f64>() + offset[i])
                .collect();
            let mu = eta.iter().map(|&e| link.inverse(e)).collect();
            (eta, mu)
        };
        let valid = |mu: &[f64]| mu.iter().all(|&m| family.valid_mean(m));

        // R starts from the family's means, mapped to the linear predictor and
        // back, unless coefficients are given.
        let (mut eta, mut mu) = start.map_or_else(
            || {
                let eta: Vec<f64> = (0..n)
                    .map(|i| link.link(family.start(y[i], weights[i])))
                    .collect();
                let mu = eta.iter().map(|&e| link.inverse(e)).collect();
                (eta, mu)
            },
            predict,
        );
        let mut old_deviance = deviance(&mu);
        let mut previous: Option<Vec<f64>> = start.map(<[f64]>::to_vec);
        let mut beta = previous.clone().unwrap_or_else(|| vec![0.0; p]);
        let (mut converged, mut boundary, mut iterations) = (false, false, 0);
        let mut current_deviance = old_deviance;
        let mut last_weights = vec![0.0; n];
        let mut last_gram = Mat::<f64>::zeros(p, p);

        for iteration in 1..=self.max_iterations {
            iterations = iteration;
            let mut z = vec![0.0; n];
            let mut w2 = vec![0.0; n];
            for i in 0..n {
                let derivative = link.derivative(eta[i]);
                if weights[i] > 0.0 && derivative != 0.0 {
                    z[i] = eta[i] - offset[i] + (y[i] - mu[i]) / derivative;
                    w2[i] = weights[i] * derivative * derivative / family.variance(mu[i]);
                }
            }
            (beta, last_gram) = self.least_squares(x, &w2, &z)?;
            last_weights.clone_from(&w2);
            (eta, mu) = predict(&beta);
            current_deviance = deviance(&mu);
            boundary = false;
            if !current_deviance.is_finite() || !valid(&mu) {
                let Some(old) = previous.as_ref() else {
                    return Err(GlmError::NoValidStart);
                };
                let mut halvings = 0;
                while !current_deviance.is_finite() || !valid(&mu) {
                    halvings += 1;
                    if halvings > self.max_iterations {
                        return Err(GlmError::StepHalving);
                    }
                    for (b, o) in beta.iter_mut().zip(old) {
                        *b = (*b + o) / 2.0;
                    }
                    (eta, mu) = predict(&beta);
                    current_deviance = deviance(&mu);
                }
                boundary = true;
            }
            if (current_deviance - old_deviance).abs() / (0.1 + current_deviance.abs())
                < self.tolerance
            {
                converged = true;
                break;
            }
            old_deviance = current_deviance;
            previous = Some(beta.clone());
        }

        let used = weights.iter().filter(|&&w| w > 0.0).count();
        let dispersion = if family.fixed_dispersion() {
            1.0
        } else {
            let pearson: f64 = (0..n)
                .filter(|&i| weights[i] > 0.0)
                .map(|i| weights[i] * (y[i] - mu[i]) * (y[i] - mu[i]) / family.variance(mu[i]))
                .sum();
            pearson / (used as f64 - p as f64)
        };
        let inverse = Factor::new(last_gram.as_ref(), &Attempt::SYMMETRIC)?
            .solve(Mat::<f64>::identity(p, p).as_ref())?;
        let covariance = (0..p * p)
            .map(|k| dispersion * inverse[(k / p, k % p)])
            .collect();
        Ok(Fit {
            coefficients: beta,
            fitted: mu,
            linear_predictor: eta,
            deviance: current_deviance,
            iterations,
            converged,
            boundary,
            working_weights: last_weights,
            dispersion,
            covariance,
        })
    }

    fn information(&self, x: MatRef<'_, f64>, w2: &[f64]) -> Mat<f64> {
        let mut gram = weighted_gram(x, w2);
        for j in 1..gram.ncols() {
            gram[(j, j)] += self.ridge;
        }
        gram
    }

    fn least_squares(
        &self,
        x: MatRef<'_, f64>,
        w2: &[f64],
        z: &[f64],
    ) -> Result<(Vec<f64>, Mat<f64>), GlmError> {
        let gram = self.information(x, w2);
        let rhs = weighted_xtz(x, w2, z);
        let solution = epistat_linalg::solve(gram.as_ref(), rhs.as_ref(), &Attempt::SYMMETRIC)?;
        Ok(((0..x.ncols()).map(|j| solution.x[(j, 0)]).collect(), gram))
    }
}

fn check(what: &'static str, expected: usize, found: usize) -> Result<(), GlmError> {
    if expected == found {
        Ok(())
    } else {
        Err(GlmError::Length {
            what,
            expected,
            found,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Mat<f64>, Vec<f64>) {
        let x = Mat::from_fn(40, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                f64::from(u32::try_from(i).unwrap()) / 10.0
            }
        });
        let y = (0..40)
            .map(|i| {
                0.5 + f64::from(u32::try_from(i % 7).unwrap()) * 0.3
                    + f64::from(u32::try_from(i).unwrap()) / 20.0
            })
            .collect();
        (x, y)
    }

    #[test]
    fn tweedie_at_power_two_is_the_gamma_fit() {
        let (x, y) = data();
        let gamma = Glm::new(Family::Gamma, Link::Log)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        let tweedie = Glm::new(Family::Tweedie { power: 2.0 }, Link::Log)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        assert_eq!(gamma.coefficients, tweedie.coefficients);
    }

    #[test]
    fn tweedie_between_one_and_two_fits() {
        let (x, mut y) = data();
        y[3] = 0.0;
        let fit = Glm::new(Family::Tweedie { power: 1.5 }, Link::Log)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        assert!(fit.converged && fit.deviance.is_finite() && fit.dispersion > 0.0);
    }

    #[test]
    fn a_ridge_shrinks_the_slope() {
        let (x, y) = data();
        let plain = Glm::new(Family::Gaussian, Link::Identity)
            .fit(x.as_ref(), &y, None, None, None)
            .unwrap();
        let ridged = Glm {
            ridge: 50.0,
            ..Glm::new(Family::Gaussian, Link::Identity)
        }
        .fit(x.as_ref(), &y, None, None, None)
        .unwrap();
        assert!(ridged.coefficients[1].abs() < plain.coefficients[1].abs());
    }

    #[test]
    fn errors() {
        let (x, y) = data();
        let glm = Glm::new(Family::Binomial, Link::Logit);
        assert!(matches!(
            glm.fit(x.as_ref(), &y, None, None, None),
            Err(GlmError::Response { .. })
        ));
        assert!(matches!(
            glm.fit(x.as_ref(), &y[..3], None, None, None),
            Err(GlmError::Length { .. })
        ));
    }
}
