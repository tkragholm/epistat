# Bias of the pooled Cox log hazard ratio with and without the outcome in the
# imputation model, from simulated studies imputed with mice.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/impute_cox.R
#
# Each study has n = 300 people with a covariate x ~ N(0, 1) and a complete
# covariate z ~ N(0, 1). The event time is exponential with rate
# 0.1 exp(0.3 x + 0.3 z) and the censoring time exponential with rate 0.05, so
# about a third are censored. x is missing with probability
# plogis(-1.6 + status + 0.5 z), which depends on the observed outcome
# and covariate and so is missing at random, for about 30 per cent of the
# people. The log hazard ratio is small because the White-Royston linear
# approximation is accurate for small effects (at 0.5 the pooled estimate is
# about 0.03 low). x is imputed by mice's norm with
# m = 10 and 5 iterations, in two ways: from z, the event indicator and the
# Nelson-Aalen cumulative hazard (White and Royston 2009), and from z alone.
# The Cox model coxph(Surv(time, status) ~ x + z) is fitted to each completed
# set and pooled by mice::pool. The file records, over the studies, the
# estimate of the log hazard ratio of x and its variance. The test generates
# its own studies in Rust with its own random numbers, so the two sets of
# studies share the design and not the draws.

suppressMessages({
  library(mice)
  library(survival)
  library(jsonlite)
})

studies <- 300
n <- 300
beta <- 0.3
set.seed(20261130)
one <- function(with_outcome) {
  x <- rnorm(n); z <- rnorm(n)
  t_event <- rexp(n, 0.1 * exp(beta * x + 0.3 * z))
  t_cens <- rexp(n, 0.05)
  time <- pmin(t_event, t_cens); status <- as.integer(t_event <= t_cens)
  pm <- plogis(-1.6 + status + 0.5 * z)
  x_obs <- x; x_obs[runif(n) < pm] <- NA
  d <- data.frame(time = time, status = status, z = z, x = x_obs)
  d$H <- nelsonaalen(d, time, status)
  d <- d[, c("x", "z", "status", "H", "time")]
  pred <- matrix(0, 5, 5, dimnames = list(names(d), names(d)))
  pred["x", c("z", if (with_outcome) c("status", "H"))] <- 1
  imp <- mice(d, m = 10, maxit = 5, method = c("norm", "", "", "", ""), predictorMatrix = pred,
              printFlag = FALSE)
  fit <- with(imp, coxph(Surv(time, status) ~ x + z))
  p <- summary(pool(fit))
  c(estimate = p$estimate[p$term == "x"], se = p$std.error[p$term == "x"])
}
res <- list()
for (w in c(TRUE, FALSE)) {
  m <- t(replicate(studies, one(w)))
  res[[if (w) "with" else "without"]] <- list(estimate = m[, "estimate"], se = m[, "se"])
}
write(toJSON(list(versions = list(r = R.version.string, mice = as.character(packageVersion("mice")),
                                  survival = as.character(packageVersion("survival"))),
                  beta = beta, n = n, studies = studies, results = res),
             auto_unbox = TRUE, digits = NA), "tests/data/impute_cox.json")
