# Reference values for the distribution objects of epistat_stats::dist and
# the free functions they call: densities and their logs, both tails and
# their logs, and quantiles of both tails, from R's d, p and q functions.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/distributions.R
suppressMessages(library(jsonlite))

probabilities <- c(1e-300, 1e-17, 1e-10, 1e-4, 0.025, 0.3, 0.5, 0.7, 0.975, 1 - 1e-4, 1 - 1e-10)

# One case per distribution and parameter set: points x with d, log d,
# lower and upper tails and their logs; probabilities p with the lower and
# upper quantiles.
continuous <- function(name, params, xs, d, p, q) {
  args <- as.list(params)
  points <- lapply(xs, function(x) {
    list(
      x = x,
      d = do.call(d, c(list(x), args)),
      log_d = do.call(d, c(list(x), args, log = TRUE)),
      lower = do.call(p, c(list(x), args)),
      upper = do.call(p, c(list(x), args, lower.tail = FALSE)),
      log_lower = do.call(p, c(list(x), args, log.p = TRUE)),
      log_upper = do.call(p, c(list(x), args, lower.tail = FALSE, log.p = TRUE))
    )
  })
  quantiles <- lapply(probabilities, function(pr) {
    list(
      p = pr,
      lower = do.call(q, c(list(pr), args)),
      upper = do.call(q, c(list(pr), args, lower.tail = FALSE))
    )
  })
  list(name = name, params = as.list(params), points = points, quantiles = quantiles)
}

cases <- list()
add <- function(case) cases[[length(cases) + 1]] <<- case

for (ms in list(c(0, 1), c(-3, 0.5), c(100, 15))) {
  add(continuous("normal", c(mean = ms[1], sd = ms[2]),
                 ms[1] + ms[2] * c(-40, -8, -2, -0.3, 0, 0.4, 1.5, 6, 30),
                 dnorm, pnorm, qnorm))
}
for (df in c(0.7, 1, 3, 12.5, 200, 5e4)) {
  add(continuous("t", c(df = df), c(-50, -4, -1, -0.1, 0, 0.6, 2, 7, 80), dt, pt, qt))
}
for (df in c(0.4, 1, 2, 7, 60, 1e4)) {
  add(continuous("chisq", c(df = df), c(1e-8, 0.01, 0.5, 1, 3, 9, 40, 300, 2e4), dchisq, pchisq, qchisq))
}
for (dfs in list(c(1, 1), c(2, 9), c(5, 30), c(0.8, 3), c(40, 1e3), c(4, 2e7))) {
  add(continuous("f", c(df1 = dfs[1], df2 = dfs[2]),
                 c(1e-6, 0.05, 0.4, 1, 2.5, 8, 60, 1e3), stats::df, pf, qf))
}
for (sr in list(c(0.5, 1), c(1, 2), c(3, 0.25), c(20, 4), c(400, 1))) {
  add(continuous("gamma", c(shape = sr[1], rate = sr[2]),
                 c(1e-9, 0.01, 0.3, 1, 2.5, 7, 30, 120, 600), dgamma, pgamma, qgamma))
}
for (ab in list(c(0.5, 0.5), c(1, 1), c(2, 5), c(0.3, 4), c(30, 12), c(800, 900))) {
  add(continuous("beta", c(shape1 = ab[1], shape2 = ab[2]),
                 c(1e-8, 0.001, 0.1, 0.35, 0.5, 0.47, 0.8, 0.999, 1 - 1e-8), dbeta, pbeta, qbeta))
}

discrete <- function(name, params, xs, d, p, q) {
  args <- as.list(params)
  points <- lapply(xs, function(x) {
    list(
      x = x,
      d = do.call(d, c(list(x), args)),
      log_d = do.call(d, c(list(x), args, log = TRUE)),
      lower = do.call(p, c(list(x), args)),
      upper = do.call(p, c(list(x), args, lower.tail = FALSE))
    )
  })
  quantiles <- lapply(probabilities, function(pr) {
    list(
      p = pr,
      lower = do.call(q, c(list(pr), args)),
      upper = do.call(q, c(list(pr), args, lower.tail = FALSE))
    )
  })
  list(name = name, params = as.list(params), points = points, quantiles = quantiles)
}
for (lambda in c(0.02, 1, 3.5, 40, 2500, 1e6)) {
  xs <- unique(pmax(0, round(lambda + sqrt(lambda) * c(-30, -4, -1, 0, 1, 3, 10, 40))))
  add(discrete("poisson", c(lambda = lambda), xs, dpois, ppois, qpois))
}
for (np in list(c(1, 0.5), c(10, 0.3), c(25, 0.02), c(400, 0.9), c(1e5, 0.4))) {
  n <- np[1]; pr <- np[2]
  xs <- unique(pmin(n, pmax(0, round(n * pr + sqrt(n * pr * (1 - pr)) * c(-20, -3, -1, 0, 1, 4, 20)))))
  add(discrete("binomial", c(size = n, prob = pr), xs, dbinom, pbinom, qbinom))
}

write(toJSON(list(versions = list(r = R.version.string), cases = cases),
             auto_unbox = TRUE, digits = 17, na = "string"), "tests/data/distributions.json")
