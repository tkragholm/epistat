# Changelog

## [0.4.0] - unreleased

The crate has one error type with stable codes, adds restricted and natural cubic splines and a set of small-matrix routines, and no longer offers a `nightly` feature. Results are unchanged from 0.3.1.

### Breaking changes

- `DesignError`, `SolveError` and `basis::KnotError` are replaced by `epistat_linalg::Error`, a non-exhaustive enum with the variants `Input` (wrapping `epistat_core::InputError`), `Singular`, `Boundary`, `OutsideBoundary`, `TooFewKnots`, `TooFewObservations` and `TooFewDistinct`. The old `Length`, `Weight`, `Tolerance`, `Shape`, `NotFinite` and `Unsorted` variants are now `Error::Input`. A matrix that is not finite is reported as `Error::Singular` by the solves. Match on `Error` or read `code()` through `epistat_core::Classify`.
- `Solution` is `#[non_exhaustive]`. Read its fields and do not build it with a struct literal.
- The `nightly` feature is removed. It enabled faer's AVX-512 kernels. A program that wants them can turn on faer's `nightly` feature in its own manifest, and results then differ in the last bits on a machine with AVX-512. On x86, last bits also depend on which vector instructions the processor offers.
- The crate depends on `epistat-core` and `epistat-stats`, and its minimum supported Rust version is 1.89, up from 1.85.

### Added

- `basis::RestrictedCubic`, a restricted cubic spline in Harrell's form, with `RcsScale` for the scaling, as `Hmisc::rcspline.eval`.
- `basis::NaturalSpline`, a natural cubic spline, as `splines::ns`, with prediction at new values and `count_outside` for values beyond the boundary.
- `basis::harrell_knots` and `basis::natural_spline_knots`, the knot rules of those splines.
- `basis::basis_matrix`, the basis evaluated at a vector of values, and `BSpline::nonzero_values_into`, the nonzero basis functions at one value.
- `solve::lu_partial`, `solve::llt` and `solve::least_squares`, dense solves with one factorisation.
- `small`, solves with a small Cholesky factor: `forward_solve`, `backward_solve`, `cholesky_solve`, `cholesky_inverse`, `log_det_from_cholesky` and `quadratic_form_inverse`.
- `cholesky::cholesky_lower` with `Pivot`, and `cholesky_factor` with `Regularisation`, `CholeskySolver` for repeated solves of one dimension, and `eigen::pinv`, the generalised inverse of a symmetric matrix.
- `gram`, `congruence`, `block_scaled_gram`, `block_xtv` and `BlockSum`, products over fixed blocks of rows.
- `design::from_columns`, a design matrix from columns.
- `weight_error`, which says what is wrong with a weight vector.
- Re-exports of faer's `Col` and `Par`.

### Deprecated

- `DesignError`, `SolveError` and `basis::KnotError`, aliases of `Error`, use `Error`. Deprecated items are removed in the next release that breaks the API.

### Results

No released result changes.

## [0.3.1] - 2026-11-22

- Added a Jacobi eigendecomposition of a small symmetric matrix.
- Re-exported faer's `Mat`, `MatRef` and `Side`, so that dependants reach faer only through this crate.

## [0.3.0] - 2026-11-19

- Undefined results are returned as `None`, and invalid weights are refused.

## [0.2.0] - 2026-11-19

- A factorisation can refuse pivots below a threshold. `Method` and `Attempt` are non-exhaustive.

## [0.1.0] - 2026-11-19

- First release: solves with a sequence of fallback factorisations, row-blocked regression products, B-spline and polynomial bases, design-column pruning, selections and an inverse.
