# epistat-semicontinuous

Models for semicontinuous outcomes, non-negative with a mass at zero, such as
costs: the two-part model (a logit for any positive outcome and a gamma
regression with log link for the positive ones), the Tweedie GLM with log link
by quasi-likelihood, log-normal regression with Duan's smearing
retransformation, and the marginalised two-part model for panels, fitted by
MCMC with correlated subject effects.

Each method has one entry point, `two_part::fit`, `tweedie::fit`,
`lognormal::fit`, `mtp::fit` and `compare::{compare, cross_validate,
select}`, on a checked view of the data, `CostData` or `LongitudinalData`,
and every function returns the crate's one `Error`. The regression fits are
IRLS with `glm.fit`'s convergence rule and step halving, with prior weights,
ridge and elastic-net penalties, warm starts and, when asked for, a fallback
ladder of ridges for fits that fail, reported in the fit. Standard errors are
model-based, robust or cluster-robust. Compact designs refit one design
under many weight vectors, as a fractional-random-weight bootstrap does,
without forming the dense products again. Model comparison gives in-sample
metrics, AIC and BIC, K-fold cross-validation and the Park test.

The Tweedie fit agrees with R's `glm` with `statmod::tweedie`, the elastic
net with glmnet, and the robust standard errors with HC0 sandwiches from R's
`glm`. The R scripts that write the references are in `tests/r`. Results do
not depend on the number of threads, and golden hashes pin the fits.

```rust
use epistat_linalg::Mat;
use epistat_semicontinuous::{CostData, Model, two_part};

let n = 40;
let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { i as f64 / 10.0 });
let y: Vec<f64> = (0..n)
    .map(|i| if i % 4 == 0 { 0.0 } else { 1.0 + 0.1 * i as f64 })
    .collect();
let data = CostData::builder(x.as_ref(), &y).build()?;

let model = two_part::fit(&data, &two_part::TwoPartOptions::default(), None)?;
let prediction = model.predict(x.as_ref())?;
assert_eq!(prediction.expected_outcome.nrows(), n);
assert!(model.report.iterations_logit > 0);
# Ok::<(), epistat_semicontinuous::Error>(())
```

The `bench-internals` feature exposes the products the benchmarks time. The
`nightly` feature enables faer's AVX-512 kernels through `epistat-linalg`.

Unlike the other epistat crates, the fits use fused multiply-add, so results
are the same on any number of threads but can differ between platforms in
the last bits.
