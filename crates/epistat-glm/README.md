# epistat-glm

Generalised linear models (Gaussian, binomial, Poisson, gamma and Tweedie,
with identity, log, logit, probit and complementary log-log links) fitted by IRLS as
R's `glm.fit` does: the same starting values, step halving and convergence rule, with prior
weights, offsets, starting coefficients and an optional ridge. Coefficients,
deviances, iteration counts and standard errors agree with R's `glm()` on
the test fits, and results do not depend on the number of threads.

The `multinom` module fits the multinomial logit, the model of the generalised
propensity score, as `nnet::multinom()` does. The `elastic_net` module fits Gaussian and binomial elastic nets by coordinate
descent as glmnet does, with weights, penalty factors and warm starts, and the
`park` module has the modified Park test of Manning and Mullahy and the classic
Park test. Both are checked against R.
