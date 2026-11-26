# The points R's uniroot evaluates, the root it returns and its iteration
# count, for the functions of tests/r_root.rs. The output is pasted into that
# test. Run: Rscript root.R
cases <- list(
  list(name = "cubic", f = function(x) x^3 - x - 2, lower = 1, upper = 2, tol = .Machine$double.eps^0.25),
  list(name = "cosine", f = function(x) cos(x) - x, lower = 0, upper = 1, tol = 1e-12),
  list(name = "exponential", f = function(x) exp(x) - 5, lower = 0, upper = 3, tol = 1e-9),
  list(name = "flat cube", f = function(x) (x - 1 / 3)^3, lower = 0, upper = 1, tol = 1e-10),
  list(name = "decreasing", f = function(x) 1 / (1 + x) - 0.3, lower = 0, upper = 10, tol = 1e-8),
  list(name = "step", f = function(x) if (x < 0.6) -1 else 1, lower = 0, upper = 1, tol = 1e-6)
)
cat(sprintf("// %s, %s\n", R.version.string, "stats::uniroot"))
for (case in cases) {
  seen <- numeric(0)
  g <- function(x) { seen <<- c(seen, x); case$f(x) }
  r <- uniroot(g, c(case$lower, case$upper), tol = case$tol, maxiter = 1000)
  # uniroot evaluates the two ends first and the root once more at the end.
  search <- seen[3:(length(seen) - 1)]
  cat(sprintf("(\"%s\", %.17g, %.17g, %.17g, %.17g, %d, &[%s]),\n", case$name, case$lower,
              case$upper, case$tol, r$root, r$iter, paste(sprintf("%.17g", search), collapse = ", ")))
}
