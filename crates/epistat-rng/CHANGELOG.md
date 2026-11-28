# Changelog

## [0.2.0] - unreleased

The draws that take parameters now return one error type that carries a stable code, and the crate gains seed derivation and a binomial draw. Every stream and every draw returns the same numbers as 0.1.3.

### Breaking changes

- `DrawError` is replaced by `epistat_rng::Error`, a non-exhaustive enum with one variant, `Input`, that wraps `epistat_core::InputError`. It is no longer `Copy` or `Eq`. Code that names the type writes `epistat_rng::Error`, and code that matched on the message text reads `error.code()` through `epistat_core::Classify` instead.
- `Stream::wishart` and `Stream::inverse_wishart` refuse a non-finite entry below the diagonal of `scale_cholesky`. Before, only the diagonal was checked.
- The crate depends on `epistat-core`.

### Added

- `Error`, which implements `Classify` and converts from `InputError`.
- `derive(base, path)`, the seed at the end of a path of indices, equal to `replicate_seed` folded over the path.
- `replicate_seeds(bootstrap)`, the seed of every replicate of a `Bootstrap`.
- `Stream::binomial(n, p)`, a binomial draw that stays fast for large `n`.
- `mix64`, the SplitMix64 output function without the increment. `splitmix64(x)` is `mix64(x + GAMMA)`.
- `Stream::wishart_from_scale` and `Stream::inverse_wishart_from_scale`, which take the scale matrix itself, read its dimension from its length, refuse one that is not symmetric or not positive definite, and factorise it, so the draws are those of `wishart` and `inverse_wishart` with its Cholesky factor. The factor forms stay; a scale passed to them where the factor belongs drew from another distribution without an error.
- `rademacher_sums`, which draws `rademacher_sum` for several streams at once and returns, per stream, the same value and final state as the single-stream call.

### Deprecated

- `DrawError`, use `Error`. Deprecated items are removed in the next release that breaks the API.

### Results

No released result changes.

## [0.1.3] - 2026-11-19

- Added the distribution draws on `Stream`: normal, gamma, chi-square, beta, Wishart and inverse Wishart.

## [0.1.2] - 2026-11-19

- Published the Python reference of the stream with the crate tests. No change to the public items.

## [0.1.1] - 2026-11-19

- `Stream` can drive `rand` and `rand_distr` distributions.

## [0.1.0] - 2026-11-19

- First release: `Stream`, `splitmix64`, `replicate_seed`, `seeded_id_hash`, bootstrap resampling counts, Dirichlet weights and `rademacher_sum`.
