# The points R's optimize evaluates, and the minimum it returns, for the
# functions of tests/r_brent.rs. The output is pasted into that test.
# Run: Rscript brent.R
cases <- list(
  list(name = "parabola", f = function(x) (x - 1 / 3)^2, lower = 0, upper = 1, tol = .Machine$double.eps^0.25),
  list(name = "sine", f = function(x) sin(x), lower = 0, upper = 2 * pi, tol = 1e-10),
  list(name = "kink", f = function(x) abs(x - 0.7), lower = -1, upper = 3, tol = 1e-8),
  list(name = "quartic", f = function(x) x^4 - 3 * x^3 + 2, lower = -1, upper = 4, tol = 1e-6),
  list(name = "flat edge", f = function(x) exp(x), lower = -2, upper = 1, tol = 1e-7)
)
cat(sprintf("// %s, %s\n", R.version.string, "stats::optimize"))
for (case in cases) {
  seen <- numeric(0)
  g <- function(x) { seen <<- c(seen, x); case$f(x) }
  m <- optimize(g, c(case$lower, case$upper), tol = case$tol)
  cat(sprintf("(\"%s\", %.17g, %.17g, %.17g, %.17g, &[%s]),\n", case$name, case$lower, case$upper,
              case$tol, m$minimum, paste(sprintf("%.17g", seen), collapse = ", ")))
}
