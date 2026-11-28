# Changelog

## [0.1.0] - unreleased

First release of `epistat-impute`: multiple imputation by chained equations as R's `mice` does it, for continuous, binary and nominal variables.

### Added

- `impute` with `ImputeSpec` and `ImputeOptions`. The methods are `norm` (linear regression with a parameter draw), `pmm` (predictive mean matching with a sorted donor search), `logreg` (logistic regression with `mice`'s augmentation rows), `polyreg` (multinomial regression with coefficients drawn from their sampling distribution, where `mice` ignores that uncertainty), `sample` (random draws of observed values) and passive columns computed from the others.
- A predictor list per column, a visit order, and counts of the repairs made: predictors dropped for collinearity and fits that needed augmentation.
- `nelson_aalen` with `Ties`: the Nelson-Aalen cumulative hazard at each person's own time, with `mice`'s tie rule or the textbook one, for use with the event indicator as predictors of an incomplete covariate (White and Royston).
- A trace of the mean and variance of the imputed cells by iteration.
- `Imputed::pool` and `pool_vector`: the completed data sets are analysed by a closure and pooled by Rubin's rules through `epistat-inference::pooling`, for a scalar or a vector.
- Every draw comes from an `epistat-rng` stream seeded by chain, iteration, column and row, so the imputed values are the same on any number of threads.
- Not included: ordered logistic imputation, post-processing, multivariate blocks and substantive-model-compatible imputation.

### Results

No released result changes.
