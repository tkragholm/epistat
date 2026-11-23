# Clone-censor-weight analysis of "start treatment within 3 periods and stay on
# it" against "never start", written out with base R (no package implements it),
# following Maringe et al. (2020).
#
# Person-period data are simulated with a time-varying covariate l that depends
# on the earlier treatment and drives both starting and stopping treatment and
# the outcome. The reference clones every person, censors each clone at its first
# deviation (the row of the deviation is dropped from the outcome), fits for
# each strategy a logistic model of the deviation on the rows at risk of it,
# forms the cumulative product of the stabilised inverse probability of staying
# uncensored, and fits the weighted pooled logistic outcome model with the
# covariance of sandwich::vcovCL(type = NULL) clustered by person. It also gives
# the cumulative incidence of each strategy.
# The rows of the clones, their weights and the fits are written out; Rust forms
# them from the person-period data.
# Numbers are written with 17 digits.
# Run: Rscript clone_censor_weight.R > clone_censor_weight.json
suppressMessages({
  library(jsonlite)
  library(sandwich)
})
set.seed(20261127)
n <- 1500
tmax <- 9
grace <- 3
rows <- vector("list", n)
for (i in seq_len(n)) {
  l0 <- rnorm(1)
  l <- l0
  a_prev <- 0
  out <- list()
  for (p in 0:tmax) {
    l <- 0.5 * l + 0.4 * a_prev + rnorm(1, 0, 0.7)
    pa <- if (a_prev == 1) plogis(2.4 - 0.5 * l) else plogis(-2.2 + 0.7 * l + 0.1 * p)
    a <- rbinom(1, 1, pa)
    y <- rbinom(1, 1, plogis(-4 - 0.6 * a + 0.4 * l + 0.05 * p))
    out[[length(out) + 1]] <- data.frame(id = i - 1, period = p, treatment = a,
      outcome = y, l = round(l, 4), l0 = round(l0, 4))
    a_prev <- a
    if (y == 1) break
  }
  rows[[i]] <- do.call(rbind, out)
}
d <- do.call(rbind, rows)
d$row <- seq_len(nrow(d)) - 1

# Cloning and censoring.
clones <- list()
for (id in unique(d$id)) {
  di <- d[d$id == id, ]
  for (arm in 1:2) {
    started <- FALSE
    for (k in seq_len(nrow(di))) {
      r <- di[k, ]
      if (arm == 1) {  # initiate within grace, stay on treatment
        if (started) {
          risk <- TRUE; dev <- r$treatment == 0
        } else if (r$period == grace) {
          risk <- TRUE; dev <- r$treatment == 0
        } else {
          risk <- FALSE; dev <- FALSE
        }
      } else {  # never initiate
        risk <- TRUE; dev <- r$treatment == 1
      }
      ev <- !dev && r$outcome == 1
      clones[[length(clones) + 1]] <- data.frame(row = r$row, id = id, arm = arm - 1,
        period = r$period, dev = as.integer(dev), risk = as.integer(risk),
        outcome = as.integer(ev), l = r$l, l0 = r$l0)
      if (dev) break
      started <- started || r$treatment == 1
      if (ev) break
    }
  }
}
cl <- do.call(rbind, clones)
cl$weight <- NA_real_
cl$pd <- NA_real_
cl$pn <- NA_real_
fits <- list()
for (arm in 0:1) {
  sel <- cl$arm == arm
  at_risk <- sel & cl$risk == 1
  md <- glm(dev ~ l + l0 + period + I(period^2), data = cl[at_risk, ], family = binomial)
  mn <- glm(dev ~ period, data = cl[at_risk, ], family = binomial)
  cl$pd[sel] <- predict(md, newdata = cl[sel, ], type = "response")
  cl$pn[sel] <- predict(mn, newdata = cl[sel, ], type = "response")
  fits[[arm + 1]] <- list(denominator = as.numeric(coef(md)), numerator = as.numeric(coef(mn)))
}
cl$factor <- ifelse(cl$risk == 1, (1 - cl$pn) / (1 - cl$pd), 1)
cl$weight <- ave(cl$factor, interaction(cl$id, cl$arm), FUN = cumprod)

obs <- cl[cl$dev == 0, ]
m <- suppressWarnings(glm(outcome ~ arm + period + I(period^2) + arm:period, data = obs,
  weights = weight, family = binomial))
v <- vcovCL(m, cluster = obs$id, type = NULL, sandwich = TRUE, fix = FALSE)
times <- 0:tmax
cum <- function(arm) {
  nd <- data.frame(arm = arm, period = times)
  p <- predict(m, newdata = nd, type = "response")
  1 - cumprod(1 - p)
}
out <- list(
  versions = list(R = as.character(getRversion()),
    sandwich = as.character(packageVersion("sandwich")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  grace = grace,
  input = list(id = d$id, period = d$period, treatment = d$treatment,
    outcome = d$outcome, l = d$l, l0 = d$l0),
  clones = list(row = cl$row, arm = cl$arm, dev = cl$dev, risk = cl$risk,
    outcome = cl$outcome, weight = cl$weight, pd = cl$pd, pn = cl$pn),
  fits = fits,
  msm_coef = as.numeric(coef(m)),
  msm_vcov = as.numeric(v),
  times = times,
  cum_inc_0 = cum(0), cum_inc_1 = cum(1))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null"))
