# Bias and coverage of the pooled coefficient of a covariate that is missing at
# random, imputed by mice's norm and pmm.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/impute_linear.R
#
# 500 studies of n = 200: x and z standard normal, y = 1 + 0.5 x + 0.5 z + e
# with e standard normal, and x missing with probability
# plogis(-0.8 + 0.7 z + 0.5 y), which depends on the observed y and z. x is
# imputed from z and y with m = 10 and 5 iterations, the model y ~ x + z is fitted
# to each completed set and pooled by mice::pool, and the 95 per cent interval
# of the coefficient of x is checked against 0.5. The file records the pooled
# estimate and whether the interval covers, for each study.

suppressMessages({
  library(mice)
  library(jsonlite)
})

studies <- 500
n <- 200
set.seed(20261201)
one <- function(method) {
  x <- rnorm(n); z <- rnorm(n); y <- 1 + 0.5 * x + 0.5 * z + rnorm(n)
  xo <- x; xo[runif(n) < plogis(-0.8 + 0.7 * z + 0.5 * y)] <- NA
  d <- data.frame(x = xo, z = z, y = y)
  imp <- mice(d, m = 10, maxit = 5, method = c(method, "", ""), printFlag = FALSE)
  p <- summary(pool(with(imp, lm(y ~ x + z))), conf.int = TRUE)
  p <- p[p$term == "x", ]
  c(estimate = p$estimate, covered = as.numeric(p[["2.5 %"]] <= 0.5 && 0.5 <= p[["97.5 %"]]))
}
res <- list()
for (method in c("norm", "pmm")) {
  m <- t(replicate(studies, one(method)))
  res[[method]] <- list(estimate = m[, "estimate"], covered = m[, "covered"])
}
write(toJSON(list(versions = list(r = R.version.string, mice = as.character(packageVersion("mice"))),
                  truth = 0.5, n = n, studies = studies, results = res),
             auto_unbox = TRUE, digits = NA), "tests/data/impute_linear.json")
