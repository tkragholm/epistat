# epistat-optim

Optimisers for the small problems of statistical estimation. The only
dependency is `rayon`, and it is optional. The crate holds:

- BFGS with the line search of Moré and Thuente,
- Brent's one-dimensional minimiser, as R's `optimize`,
- the Nelder-Mead simplex method,
- a dense primal simplex method for linear programs with warm starts,
- a bounded-variable dual simplex method for programs with few rows and up to a
  million bounded columns,
- branch and bound for programs with binary columns, on the bounded simplex,
- minimum-cost flow by the network simplex method, and
- independent restarts of a fit, run in parallel under the `rayon` feature.
