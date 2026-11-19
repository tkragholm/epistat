# Tweedie fits with a log link and zero responses, by R's glm() with
# statmod::tweedie(var.power = p, link.power = 0).
# Run: Rscript glm_tweedie.R > glm_tweedie.json
suppressMessages({
  library(statmod)
  library(jsonlite)
})
set.seed(20261119)
n <- 120
x <- cbind(1, rnorm(n), runif(n), rbinom(n, 1, 0.4))
mu <- exp(0.4 + 0.5 * x[, 2] - 0.8 * x[, 3] + 0.3 * x[, 4])
y <- numeric(n)
for (i in seq_len(n)) {
  # Compound Poisson-gamma: zero with probability exp(-lambda).
  k <- rpois(1, 0.9)
  y[i] <- if (k == 0) 0 else sum(rgamma(k, shape = 1.5, scale = mu[i] / 1.35))
}
weights <- runif(n, 0.5, 2)
stopifnot(sum(y == 0) >= 10)

fits <- lapply(c(1.2, 1.5, 1.8), function(p) {
  fit <- glm(y ~ x - 1, family = tweedie(var.power = p, link.power = 0),
             weights = weights)
  list(power = p,
       coefficients = unname(coef(fit)),
       se = unname(sqrt(diag(vcov(fit)))),
       deviance = fit$deviance,
       iterations = fit$iter,
       dispersion = summary(fit)$dispersion,
       converged = fit$converged)
})
cat(toJSON(list(x = unname(x), y = y, weights = weights, fits = fits),
           digits = NA, auto_unbox = TRUE, na = "null"))
