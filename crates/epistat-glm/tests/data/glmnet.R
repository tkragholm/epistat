# Elastic-net fits by glmnet, with standardize = FALSE, per-observation
# weights, a penalty factor that has a zero, correlated and badly scaled
# columns, and alpha 0, 0.5 and 1.
# glmnet rescales penalty.factor to sum to the number of columns, so the
# factors below already do, and the fit uses them as given.
# For the Gaussian family glmnet divides the ridge term by the weighted
# standard deviation sy of the response (about its weighted mean, weights
# summing to 1), which is stored as sy.
# Run: Rscript glmnet.R > glmnet.json
suppressMessages({
  library(glmnet)
  library(jsonlite)
})
set.seed(20261119)
n <- 150
x1 <- rnorm(n)
x2 <- 0.9 * x1 + 0.3 * rnorm(n)
x3 <- 1000 * rnorm(n)
x4 <- rbinom(n, 1, 0.4)
x5 <- 1e-3 * runif(n)
x6 <- 50 + rnorm(n)
x <- cbind(x1, x2, x3, x4, x5, x6)
weights <- runif(n, 0.5, 2)
offset <- 0.2 * rnorm(n)
beta <- c(0.8, -0.5, 0.0004, 0.6, 300, 0.05)
eta <- drop(x %*% beta) - 2.5
y_gaussian <- eta + rnorm(n, sd = 1.5)
y_binomial <- rbinom(n, 1, plogis(eta - mean(eta)))
penalty <- c(1, 0, 2, 1, 0.5, 1.5)
stopifnot(sum(penalty) == ncol(x))

fits <- list()
for (family in c("gaussian", "binomial")) {
  y <- if (family == "gaussian") y_gaussian else y_binomial
  for (alpha in c(0, 0.5, 1)) {
    for (variant in c("plain", "weighted", "weighted_factors", "weighted_factors_offset")) {
      w <- if (variant == "plain") rep(1, n) else weights
      pf <- if (grepl("factors", variant)) penalty else rep(1, ncol(x))
      o <- if (grepl("offset", variant)) offset else rep(0, n)
      # lambdas as fractions of the smallest lambda that zeroes every
      # penalised coefficient, read from glmnet's own path
      path <- glmnet(x, y, family = family, weights = w, alpha = max(alpha, 0.05),
                     standardize = FALSE, penalty.factor = pf, offset = o)
      for (fraction in c(0.0005, 0.00005, 0.000005)) {
        lambda <- max(path$lambda) * fraction
        fit <- glmnet(x, y, family = family, weights = w, alpha = alpha,
                      lambda = lambda, standardize = FALSE,
                      penalty.factor = pf, offset = o,
                      control = list(thresh = 1e-20, maxit = 1e7))
        wn <- w / sum(w)
        z <- y - o
        sy <- sqrt(sum(wn * (z - sum(wn * z))^2))
        fits[[length(fits) + 1]] <- list(
          family = family, alpha = alpha, variant = variant, lambda = lambda,
          weighted = variant != "plain", factors = grepl("factors", variant),
          offset = grepl("offset", variant),
          coefficients = as.numeric(coef(fit)),
          sy = sy, thresh = 1e-20)
      }
    }
  }
}
cat(toJSON(list(x = unname(x), weights = weights, offset = offset,
                penalty = penalty, y_gaussian = y_gaussian,
                y_binomial = y_binomial, fits = fits),
           digits = NA, auto_unbox = TRUE, na = "null"))
