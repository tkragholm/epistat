# Complementary log-log fits: binomial(link = "cloglog") and
# quasi(link = "cloglog", variance = "constant"), the Gaussian family with the
# cloglog link here.
# Run: Rscript glm_cloglog.R > glm_cloglog.json
suppressMessages(library(jsonlite))
set.seed(20261120)
n <- 150
x <- cbind(1, rnorm(n), runif(n), rbinom(n, 1, 0.4))
eta <- -1.2 + 0.5 * x[, 2] - 0.6 * x[, 3] + 0.4 * x[, 4]
mu <- -expm1(-exp(eta))
trials <- sample(1:8, n, replace = TRUE)
yb <- rbinom(n, trials, mu) / trials
weights <- as.numeric(trials)
offset <- rnorm(n, 0, 0.2)
# A response in (0, 1), as the start of the constant-variance fit needs.
yq <- pmin(pmax(mu + rnorm(n, 0, 0.1), 0.02), 0.98)
wq <- runif(n, 0.5, 2)

one <- function(name, fit, family) {
  list(name = name, family = family,
       coefficients = unname(coef(fit)),
       se = unname(sqrt(diag(vcov(fit)))),
       deviance = fit$deviance,
       iterations = fit$iter,
       dispersion = summary(fit)$dispersion,
       converged = fit$converged)
}
fits <- list(
  one("binomial", glm(yb ~ x - 1, family = binomial(link = "cloglog"),
                      weights = weights), "binomial"),
  one("binomial_offset", glm(yb ~ x - 1, family = binomial(link = "cloglog"),
                             weights = weights, offset = offset), "binomial"),
  one("constant", glm(yq ~ x - 1, family = quasi(link = "cloglog", variance = "constant")),
      "gaussian"),
  one("constant_weighted_offset",
      glm(yq ~ x - 1, family = quasi(link = "cloglog", variance = "constant"),
          weights = wq, offset = offset), "gaussian")
)
cat(toJSON(list(versions = list(R = R.version.string),
                x = unname(x), yb = yb, yq = yq, binomial_weights = weights,
                weights = wq, offset = offset, fits = fits),
           digits = I(17), auto_unbox = TRUE, na = "null"))
