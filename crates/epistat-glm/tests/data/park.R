# The modified Park test of Manning and Mullahy (2001) and the classic Park
# test, by R. The mean model is a log-link GLM; the modified test regresses the
# squared raw-scale residuals on the log of the fitted means with
# glm(family = Gamma(link = "log")), with the mean model's prior weights when
# there are any. The classic test is lm(log(r2) ~ log(mu)).
# Run: Rscript park.R > park.json
suppressMessages(library(jsonlite))
set.seed(20261119)
n <- 200
x <- cbind(1, rnorm(n), runif(n))
mu <- exp(6 + 0.6 * x[, 2] + 0.8 * x[, 3])
weights <- runif(n, 0.5, 2)
datasets <- list(
  gamma_variance = rgamma(n, shape = 2, rate = 2 / mu),
  poisson_like = rpois(n, mu / 20) * 20 + 0.5
)
cases <- list()
for (name in names(datasets)) {
  y <- datasets[[name]]
  for (weighted in c(FALSE, TRUE)) {
    w <- if (weighted) weights else rep(1, n)
    mean_fit <- glm(y ~ x - 1, family = Gamma(link = "log"), weights = w)
    fitted <- unname(fitted(mean_fit))
    r2 <- (y - fitted)^2
    park <- glm(r2 ~ log(fitted), family = Gamma(link = "log"), weights = w)
    tight <- glm(r2 ~ log(fitted), family = Gamma(link = "log"), weights = w,
                 control = glm.control(epsilon = 1e-14, maxit = 200))
    classic <- summary(lm(log(r2) ~ log(fitted)))
    cases[[length(cases) + 1]] <- list(
      name = name, weighted = weighted, y = y, fitted = fitted,
      coefficients = unname(coef(park)),
      se = unname(coef(summary(park))[, "Std. Error"]),
      iterations = park$iter, converged = park$converged,
      tight_coefficients = unname(coef(tight)),
      tight_se = unname(coef(summary(tight))[, "Std. Error"]),
      classic_coefficients = unname(coef(classic)[, "Estimate"]),
      classic_se = unname(coef(classic)[, "Std. Error"]))
  }
}
cat(toJSON(list(weights = weights, cases = cases), digits = NA, auto_unbox = TRUE))
