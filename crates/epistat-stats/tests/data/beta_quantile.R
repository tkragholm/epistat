# Reference values for epistat_stats::student::beta_quantile: qbeta(p, a, b)
# on a grid of shapes from 1e-3 to 1e4 and probabilities from 1e-12 to
# 1 - 1e-12.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/beta_quantile.R
suppressMessages(library(jsonlite))
shapes <- c(1e-3, 0.01, 0.1, 0.5, 1, 2.5, 10, 100, 1000, 1e4)
ps <- c(1e-12, 1e-6, 1e-3, 0.025, 0.3, 0.5, 0.7, 0.975, 0.999, 1 - 1e-6, 1 - 1e-12)
cases <- list()
for (a in shapes) for (b in shapes) for (p in ps) {
  warned <- FALSE
  q <- withCallingHandlers(qbeta(p, a, b),
                           warning = function(w) { warned <<- TRUE; invokeRestart("muffleWarning") })
  cases[[length(cases) + 1]] <- list(p = p, a = a, b = b, q = q, warned = warned)
}
write(toJSON(list(versions = list(r = R.version.string), cases = cases),
             auto_unbox = TRUE, digits = 17), "tests/data/beta_quantile.json")
