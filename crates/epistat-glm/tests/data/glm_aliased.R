# Fits with an exactly aliased column and with nearly aliased columns, by R's
# glm(). R reports NA for the coefficient and the standard error of an aliased
# column, which it chooses by a pivoted QR with tolerance
# min(1e-7, epsilon / 1000), and fits the others.
# Run: Rscript glm_aliased.R > glm_aliased.json
suppressMessages(library(jsonlite))
set.seed(20261119)
n <- 80
x1 <- rnorm(n)
x2 <- runif(n)
group <- sample(0:2, n, replace = TRUE)
dummies <- cbind(group == 0, group == 1, group == 2) + 0
noise <- rnorm(n)

designs <- list(
  # The third column is the sum of the first two after the intercept.
  exact_sum = cbind(1, x1, x2, x1 + x2),
  # Three dummies that add up to the intercept.
  exact_dummies = cbind(1, x1, dummies),
  # A column that is 2 * x1 - x2 and an all-zero column after it.
  exact_two = cbind(1, x1, 2 * x1 - x2, x2, 0),
  near_1e_4 = cbind(1, x1, x2, x1 + x2 + 1e-4 * noise),
  near_1e_6 = cbind(1, x1, x2, x1 + x2 + 1e-6 * noise)
)
eta <- 0.3 + 0.5 * x1 - 0.4 * x2
y_gauss <- eta + rnorm(n, sd = 0.3)
y_pois <- rpois(n, exp(eta))

fits <- list()
for (name in names(designs)) {
  x <- designs[[name]]
  for (fam in c("gaussian", "poisson")) {
    family <- if (fam == "gaussian") gaussian() else poisson()
    y <- if (fam == "gaussian") y_gauss else y_pois
    fit <- glm(y ~ x - 1, family = family)
    s <- summary(fit)
    se <- rep(NA_real_, ncol(x))
    se[!is.na(coef(fit))] <- sqrt(diag(s$cov.unscaled)) * sqrt(s$dispersion)
    fits[[length(fits) + 1]] <- list(
      name = paste(name, fam, sep = "_"), family = fam, x = unname(x), y = y,
      coefficients = unname(coef(fit)), se = se, rank = fit$rank,
      deviance = fit$deviance, iterations = fit$iter)
  }
}
cat(toJSON(list(fits = fits), digits = NA, auto_unbox = TRUE, na = "null"))
