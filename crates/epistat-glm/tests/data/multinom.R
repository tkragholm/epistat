# Reference multinomial logit fits from nnet::multinom.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/multinom.R
#
# Simulated data with a fixed seed: 500 rows, a standard normal covariate, a
# skewed covariate and a binary covariate, and a response of three or four
# classes drawn from a multinomial logit. Every fit is made unweighted and with
# case weights, and records the coefficients (a row per class after the first,
# a column per term, intercept first), the fitted probabilities, the residual
# deviance and the standard errors of summary(). multinom() is run with
# reltol = 1e-14 and maxit = 10000 so that R's stopping rule, a relative change
# of 1e-8 in BFGS, does not set the precision of the comparison.

suppressMessages({
  library(nnet)
  library(jsonlite)
})

set.seed(20261121)
n <- 500
x1 <- rnorm(n)
x2 <- rexp(n)
x3 <- rbinom(n, 1, 0.4)
weights <- round(runif(n, 0.5, 3), 3)

draw <- function(k) {
  coefs <- matrix(c(0.3, 0.8, -0.5, 0.6,
                    -0.4, -0.7, 0.4, 0.2,
                    0.1, 0.5, 0.3, -0.9)[seq_len(4 * (k - 1))], nrow = k - 1, byrow = TRUE)
  eta <- cbind(0, cbind(1, x1, x2, x3) %*% t(coefs))
  p <- exp(eta) / rowSums(exp(eta))
  apply(p, 1, function(pi) sample.int(k, 1, prob = pi))
}

fits <- list()
for (k in c(3, 4)) {
  y <- factor(draw(k))
  d <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
  for (weighted in c(FALSE, TRUE)) {
    m <- multinom(y ~ x1 + x2 + x3, data = d, weights = if (weighted) weights else rep(1, n),
                  trace = FALSE, reltol = 1e-14, maxit = 10000, Hess = TRUE)
    s <- summary(m)
    fits[[length(fits) + 1]] <- list(
      classes = k, weighted = weighted,
      y = as.integer(y) - 1L,
      coefficients = unname(coef(m)),
      se = unname(s$standard.errors),
      fitted = unname(fitted(m)),
      deviance = m$deviance
    )
  }
}

out <- list(versions = list(r = R.version.string, nnet = as.character(packageVersion("nnet"))),
            x = cbind(x1, x2, x3), weights = weights, fits = fits)
write(toJSON(out, auto_unbox = TRUE, digits = NA, matrix = "rowmajor"), "tests/data/multinom.json")
