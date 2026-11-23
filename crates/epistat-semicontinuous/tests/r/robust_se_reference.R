# Reference standard errors for tests/robust_se.rs.
# Run: Rscript tests/r/robust_se_reference.R fixture1.csv fixture2.csv
# The CSV files hold the rows of the two fixtures the test generates, columns
# x1, x2, y, written with the shortest decimal that reads back exactly.
# Needs statmod. Prints, for each fit, the model-based standard errors with
# dispersion fixed at 1 and the HC0 sandwich standard errors.
suppressMessages(library(statmod))
args <- commandArgs(trailingOnly = TRUE)
cat("R", as.character(getRversion()), "statmod", as.character(packageVersion("statmod")), "\n")
options(digits = 12)

report <- function(label, fit, X, y, p) {
  mu <- unname(fitted(fit))
  bread <- solve(crossprod(X, X * mu^(2 - p)))
  score <- X * ((y - mu) * mu^(1 - p))
  hc0 <- sqrt(diag(bread %*% crossprod(score) %*% bread))
  cat(label, "model-based:", sqrt(diag(bread)), "\n")
  cat(label, "HC0:", unname(hc0), "\n")
}

d <- read.csv(args[1])
X <- cbind(1, d$x1, d$x2)
pos <- as.numeric(d$y > 0)
fit <- glm(pos ~ d$x1 + d$x2, family = binomial)
mu <- fitted(fit)
bread <- solve(crossprod(X, X * (mu * (1 - mu))))
score <- X * (pos - mu)
cat("logit model-based:", sqrt(diag(bread)), "\n")
cat("logit HC0:", sqrt(diag(bread %*% crossprod(score) %*% bread)), "\n")
dp <- d[d$y > 0, ]
Xp <- cbind(1, dp$x1, dp$x2)
fit <- glm(y ~ x1 + x2, data = dp, family = Gamma(link = "log"),
           control = glm.control(epsilon = 1e-14, maxit = 200))
report("gamma", fit, Xp, dp$y, 2)

d <- read.csv(args[2])
X <- cbind(1, d$x1, d$x2)
fit <- glm(y ~ x1 + x2, data = d,
           family = statmod::tweedie(var.power = 1.5, link.power = 0),
           control = glm.control(epsilon = 1e-14, maxit = 200))
report("tweedie 1.5", fit, X, d$y, 1.5)
