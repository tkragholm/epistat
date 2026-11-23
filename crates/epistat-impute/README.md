# epistat-impute

Multiple imputation by chained equations, as R's `mice` does it, for data with
continuous, binary and nominal variables.

- Methods: linear regression with a parameter draw (`norm`), predictive mean
  matching with a sorted donor search (`pmm`), logistic regression with
  `mice`'s augmentation rows (`logreg`), multinomial regression with the
  coefficients drawn from their sampling distribution (`polyreg`, which in
  `mice` ignores the uncertainty in them), random draws of observed values
  (`sample`) and passive columns computed from the others.
- A predictor list per column, a visit order, and counts of the repairs made:
  predictors dropped for collinearity and fits that needed augmentation.
- Survival outcomes: the Nelson-Aalen cumulative hazard at each person's own
  time, with `mice`'s tie rule or the textbook one, for use together with the
  event indicator as predictors of an incomplete covariate (White and Royston).
- Trace of the mean and variance of the imputed cells by iteration.
- Hand-off to Rubin's rules: the completed data sets are analysed by a closure
  and pooled with `epistat-inference::pooling`, for a scalar or a vector.

Every draw comes from an `epistat-rng` stream seeded by the chain, iteration,
column and row, so the imputed values are the same on any number of threads.
The methods are checked against `mice` and `nnet`: the deterministic steps with
the random draws injected, the distributions of the imputed values and of the
pooled estimates within Monte Carlo error, and the coverage of the pooled
intervals over simulated studies.

Not included: ordered logistic imputation, post-processing, multivariate
blocks and substantive-model-compatible imputation.
