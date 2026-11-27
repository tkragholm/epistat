# epistat-optim

Optimisers for the small problems of statistical estimation. The only
dependency is `rayon`, and it is optional. The crate holds:

- BFGS with the line search of Moré and Thuente,
- Brent's one-dimensional minimiser, with the tolerance of R's `optimize`,
- roots in a bracket by bisection and by the ITP method,
- the Nelder-Mead simplex method,
- a dense primal simplex method for linear programs with warm starts,
- a bounded-variable dual simplex method for programs with few rows and up to a
  million bounded columns,
- branch and bound for programs with binary columns, on the bounded simplex,
- the Frank-Wolfe method with an exact line search for ridge-penalised least
  squares on the probability simplex, as `synthdid` computes its weights,
- minimum-cost flow by the network simplex method, and
- independent restarts of a fit, run in parallel under the `rayon` feature.
