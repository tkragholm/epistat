# Changelog

## [0.2.0] - unreleased

Every solver now returns one error type with stable codes and refuses non-finite or mismatched input before it computes. The crate gains a one-dimensional root finder, Brent's minimiser and a Frank-Wolfe solver.

### Breaking changes

- The per-solver error types `BfgsError`, `NelderMeadError`, `LpError`, `BoundedLpError`, `MipError` and `FlowError` are replaced by `epistat_optim::Error`, a non-exhaustive enum. It wraps `epistat_core::InputError` for bad input and is not `Copy`. The variants that carried a count (for example `NelderMeadError` with vertices and dimension) are now `Error::Input` with the name of the argument, the expected length and the length found. Match on `Error` or read `code()` through `epistat_core::Classify`.
- `DenseSimplex::new(columns, g)` returns `Result<DenseSimplex, Error>` and refuses no columns, an empty `g`, ragged columns and non-finite values. Write `DenseSimplex::new(&columns, &g)?`.
- `branch_and_bound::solve` checks the lengths and finiteness of its input before indexing, so a column bound that is too short returns an error instead of panicking.
- `NelderMead::minimize` refuses a starting simplex with a non-finite vertex, and `Bfgs::minimize` refuses a non-finite starting point, with `Error::Input`.
- `bfgs::Minimum`, `nelder_mead::Minimum`, `min_cost_flow::FlowSolution`, `branch_and_bound::Solution` and `branch_and_bound::Status` are `#[non_exhaustive]`. Build them only inside the crate and read their fields by name.
- `bfgs::Minimum` and `nelder_mead::Minimum` have a new field `evaluations`.
- Items of `bounded_simplex` that exposed the internals of the branch and bound search are no longer public: `DualBound`, `TableauRow`, `BoundedSimplex::basic_columns`, `relaxation_bound`, `perturbation_total`, `perturb_costs`, `scale_perturbation`, `reduced_costs`, `tableau_row`, `dual_bound`, `row_activity`, `row_tolerance` and `solve_within`. Use `BoundedSimplex::value`, `values`, `objective`, `slack` and `at_bound` to read a solution.
- The crate depends on `epistat-core` and `epistat-rng`.
- `bfgs::Minimum`, `nelder_mead::Minimum`, `brent::Minimum`, `root::Root` and `frank_wolfe::FrankWolfeSolution` are `#[must_use]` and have a new field `termination`. Code that drops one of them now gets a warning.

### Added

- `root::itp`, the ITP root finder of Oliveira and Takahashi, with `Itp` for its options and `Root` for the result. It stops within the number of steps bisection needs plus `n0`.
- `root::bisect`, bisection on a bracket, with `Bisection` and `Root`.
- `brent::minimize`, Brent's one-dimensional minimiser with the tolerance rule of R's `optimize`, with `Brent` and `Minimum`.
- `frank_wolfe`, Frank-Wolfe least squares on the probability simplex with an exact line search: `SimplexLeastSquares`, `FrankWolfeOptions` (with `tolerance` and `max_iterations`, the name the other solvers use), `FrankWolfeState` and `FrankWolfeSolution`.
- `Error`, which implements `Classify` and converts from `InputError`.
- `evaluations`, the number of calls of the objective, on the minimum returned by `Bfgs`, `NelderMead` and `brent::minimize`.
- `Termination`, the rule that ended an iterative search (a gradient, cost, spread or bracket tolerance, an exact root, a rejected step, a cost unbounded below or the iteration limit), on the results of BFGS, Nelder-Mead, Brent, bisection, ITP and Frank-Wolfe, and `require_converged()` on each, which returns the new `Error::NotConverged` when the search stopped without converging. An unconverged search is still `Ok` with `converged: false`, as R's `optim` and `optimize` return.
- `Default` for `FrankWolfeOptions`: a tolerance of `1e-6` and 1000 steps, the defaults of `synthdid`'s `sc.weight.fw`.

### Deprecated

- `BfgsError`, `NelderMeadError`, `LpError`, `BoundedLpError`, `MipError` and `FlowError`, all aliases of `Error`, use `Error`. Deprecated items are removed in the next release that breaks the API.

### Results

No released result changes.

## [0.1.1] - 2026-11-22

- Added `branch_and_bound`, a mixed binary program solver on a new bounded-variable dual simplex in `bounded_simplex`, with a repair search for binary programs.
- Added `min_cost_flow`, a network simplex solver for minimum-cost flow, which can start from a basis or from a feasible flow.
- BFGS counts a cost rise within the tolerance as convergence.
- The dense simplex scales its tolerances by the data.

## [0.1.0] - 2026-11-19

- First release: `bfgs`, `nelder_mead` and the dense `simplex`.
- Multi-start helpers in `multistart`, with a selection that does not depend on the number of threads.
- BFGS stops at a bad step.
