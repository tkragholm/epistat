# epistat-glm

Generalised linear models (Gaussian, binomial, Poisson, gamma and Tweedie,
with identity, log and logit links) fitted by IRLS as R's `glm.fit` does:
the same starting values, step halving and convergence rule, with prior
weights, offsets, starting coefficients and an optional ridge. Coefficients,
deviances, iteration counts and standard errors agree with R's `glm()` on
the test fits, and results do not depend on the number of threads.
