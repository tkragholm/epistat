# Probit fits: binomial(link = "probit") with prior weights (trials) and an
# offset, and a response on the 0/1 scale with no weights.
# Run: Rscript glm_probit.R > glm_probit.json
suppressMessages(library(jsonlite))
set.seed(20261121)
n <- 150
x <- cbind(1, rnorm(n), runif(n), rbinom(n, 1, 0.4))
eta <- -0.3 + 0.7 * x[, 2] - 0.9 * x[, 3] + 0.5 * x[, 4]
mu <- pnorm(eta)
trials <- sample(1:8, n, replace = TRUE)
yb <- rbinom(n, trials, mu) / trials
weights <- as.numeric(trials)
offset <- rnorm(n, 0, 0.2)
y01 <- rbinom(n, 1, mu)

one <- function(name, fit) {
  list(name = name,
       coefficients = unname(coef(fit)),
       se = unname(sqrt(diag(vcov(fit)))),
       deviance = fit$deviance,
       iterations = fit$iter,
       converged = fit$converged)
}
ctl <- glm.control(epsilon = 1e-8, maxit = 25)
fits <- list(
  one("binomial", glm(yb ~ x - 1, family = binomial(link = "probit"),
                      weights = weights, control = ctl)),
  one("binomial_offset", glm(yb ~ x - 1, family = binomial(link = "probit"),
                             weights = weights, offset = offset, control = ctl)),
  one("bernoulli", glm(y01 ~ x - 1, family = binomial(link = "probit"),
                       control = ctl))
)
cat(toJSON(list(versions = list(R = R.version.string, jsonlite = as.character(packageVersion("jsonlite"))),
                x = unname(x), yb = yb, y01 = y01,
                binomial_weights = weights, offset = offset, fits = fits),
           digits = I(17), auto_unbox = TRUE, na = "null"))
