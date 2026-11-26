# Reference values for epistat_stats::rates::rate_ratio_exact: the conditional
# exact confidence interval, estimate and p-value of stats::poisson.test with
# two counts, on a grid of counts, person-times and levels.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/rate_ratio_exact.R
suppressMessages(library(jsonlite))
cases <- list()
counts <- list(c(1, 1), c(0, 4), c(4, 0), c(2, 15), c(15, 2), c(10, 10), c(37, 52),
               c(250, 310), c(1200, 1000), c(3, 80000), c(60000, 4), c(100000, 99000))
times <- list(c(1, 1), c(1, 3), c(7, 2.5), c(1000, 1300), c(12.5, 4321.75))
for (e in counts) for (t in times) for (level in c(0.8, 0.95, 0.99, 0.999)) {
  pt <- poisson.test(e, t, conf.level = level)
  cases[[length(cases) + 1]] <- list(
    e1 = e[1], e0 = e[2], t1 = t[1], t0 = t[2], level = level,
    lower = pt$conf.int[1], upper = pt$conf.int[2], estimate = unname(pt$estimate),
    p = pt$p.value)
}
write(toJSON(list(versions = list(r = R.version.string), cases = cases),
             auto_unbox = TRUE, digits = 17, null = "null"),
      "tests/data/rate_ratio_exact.json")
