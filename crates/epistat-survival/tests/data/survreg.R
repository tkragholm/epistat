# Accelerated failure time fits by survival::survreg: the lung data and
# simulated right-, left- and interval-censored data, every distribution,
# strata on the scale, case weights, a fixed scale, an intercept-only model and
# an aliased column.
# For each fit: the design matrix, the coefficients (NA, written null, for an
# aliased column), the log scale of each stratum, the variance matrix over the
# coefficients and the log scales, the log likelihood at the fit and for the
# intercept-only model, the iteration count, the linear predictors, predictions
# on the first 25 rows (linear predictor, response and quantiles with their
# standard errors) and the residuals of each type for every row. The condition
# number of the variance on the scale on which survreg iterates (covariates
# centred and scaled as survreg.fit does when the first column is an intercept)
# is written too, since it bounds how far two implementations of the same
# Newton iteration can differ.
# The versions of R, survival and jsonlite that wrote the file are in it.
# Numbers are written with 17 significant digits, so every double reads back
# as written.
# Run: Rscript survreg.R > survreg.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)

# ---- data ----
lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
lung2$event <- as.numeric(lung2$status == 2)
lung2$stratum <- lung2$sex - 1
lung2$weight <- round(runif(nrow(lung2), 0.5, 3), 2)
rownames(lung2) <- NULL

sim_n <- 240
sim_cov <- function() {
  data.frame(
    x1 = round(rnorm(sim_n), 3), x2 = rbinom(sim_n, 1, 0.4),
    stratum = sample(0:2, sim_n, replace = TRUE),
    weight = round(runif(sim_n, 0.5, 3), 2)
  )
}
# log T = 1 + 0.5 x1 - 0.4 x2 + sigma W, W standard minimum extreme value,
# with the scale of the stratum.
sim_time <- function(d) {
  sigma <- c(0.5, 0.8, 1.2)[d$stratum + 1]
  round(exp(1 + 0.5 * d$x1 - 0.4 * d$x2 + sigma * log(rexp(sim_n))), 3) + 0.001
}
right <- sim_cov()
tt <- sim_time(right)
cc <- round(rexp(sim_n, 0.15), 3) + 0.001
right$time <- pmin(tt, cc)
right$event <- as.numeric(tt <= cc)

# Left censored: a detection limit, below which only "at most the limit" is known.
left <- sim_cov()
tt <- sim_time(left)
lim <- round(runif(sim_n, 0.3, 2), 3)
left$time <- pmax(tt, lim)
left$event <- as.numeric(tt >= lim)

# Interval censored: visits at 1, 2, ..., 9; some times are seen exactly.
make_interval <- function(zero_lower) {
  d <- sim_cov()
  tt <- sim_time(d)
  k <- pmin(ceiling(tt), 10)
  lower <- k - 1
  upper <- k
  status <- rep(3, sim_n)
  status[k == 1] <- 2                 # before the first visit: left censored
  status[tt > 9] <- 0                 # after the last visit: right censored
  exact <- runif(sim_n) < 0.1
  status[exact] <- 1
  time <- ifelse(status == 0, 9, ifelse(status == 2, upper, ifelse(status == 1, tt, lower)))
  time2 <- ifelse(status == 3, upper, NA)
  if (zero_lower) {
    # Interval rows from time 0 are left censored in disguise.
    z <- which(status == 2)
    status[z] <- 3
    time[z] <- 0
    time2[z] <- upper[z]
  }
  d$time <- time
  d$time2 <- time2
  d$status <- status
  d
}
interval <- make_interval(FALSE)
interval0 <- make_interval(TRUE)

datasets <- list(lung = lung2, right = right, left = left, interval = interval, interval0 = interval0)
surv_of <- list(
  lung = "Surv(time, event)", right = "Surv(time, event)",
  left = "Surv(time, event, type = 'left')",
  interval = "Surv(time, time2, status, type = 'interval')",
  interval0 = "Surv(time, time2, status, type = 'interval')"
)

# ---- fits ----
# name, dataset, distribution, right-hand side, strata, weights, scale
specs <- list()
add <- function(name, data, dist, rhs, strata = FALSE, weights = FALSE, scale = 0) {
  specs[[length(specs) + 1]] <<- list(name, data, dist, rhs, strata, weights, scale)
}
for (dist in c("weibull", "exponential", "rayleigh", "extreme", "gaussian", "logistic", "lognormal", "loglogistic")) {
  add(paste0("lung_", dist), "lung", dist, "age + sex + ph.ecog")
}
add("lung_weibull_strata", "lung", "weibull", "age + ph.ecog", strata = TRUE)
add("lung_lognormal_strata", "lung", "lognormal", "age + ph.ecog", strata = TRUE)
add("lung_weibull_weights", "lung", "weibull", "age + sex + ph.ecog", weights = TRUE)
add("lung_weibull_fixed", "lung", "weibull", "age + sex + ph.ecog", scale = 1.5)
add("lung_gaussian_fixed", "lung", "gaussian", "age + sex", scale = 300)
add("lung_weibull_null", "lung", "weibull", "1")
add("lung_gaussian_null", "lung", "gaussian", "1")
add("lung_weibull_alias", "lung", "weibull", "age + sex + ph.ecog + I(2 * age)")
for (dist in c("weibull", "exponential", "gaussian", "logistic", "lognormal", "loglogistic")) {
  add(paste0("right_", dist), "right", dist, "x1 + x2")
  add(paste0("left_", dist), "left", dist, "x1 + x2")
  add(paste0("interval_", dist), "interval", dist, "x1 + x2")
}
add("right_weibull_all", "right", "weibull", "x1 + x2", strata = TRUE, weights = TRUE)
add("left_weibull_all", "left", "weibull", "x1 + x2", strata = TRUE, weights = TRUE)
add("interval_weibull_all", "interval", "weibull", "x1 + x2", strata = TRUE, weights = TRUE)
add("interval_gaussian_all", "interval", "gaussian", "x1 + x2", strata = TRUE, weights = TRUE)
add("interval_logistic_fixed", "interval", "logistic", "x1 + x2", scale = 1.1)
add("interval_extreme", "interval", "extreme", "x1 + x2")
add("interval_rayleigh", "interval", "rayleigh", "x1 + x2")
add("interval_weibull_null", "interval", "weibull", "1")
for (dist in c("weibull", "exponential", "lognormal")) {
  add(paste0("interval0_", dist), "interval0", dist, "x1 + x2")
}

residual_types <- c("response", "deviance", "working", "ldcase", "ldresp", "ldshape")
na_matrix <- function(m) unname(as.matrix(m))
fits <- list()
for (s in specs) {
  name <- s[[1]]; dname <- s[[2]]; dist <- s[[3]]; rhs <- s[[4]]
  use_strata <- s[[5]]; use_weights <- s[[6]]; fixed <- s[[7]]
  d <- datasets[[dname]]
  rhs_full <- if (use_strata) paste(rhs, "+ strata(stratum)") else rhs
  f <- as.formula(paste(surv_of[[dname]], "~", rhs_full))
  args <- list(formula = f, data = quote(d), dist = dist, x = TRUE, model = TRUE)
  if (use_weights) args$weights <- quote(weight)
  if (fixed > 0) args$scale <- fixed
  fit <- do.call(survreg, args)
  p <- length(coef(fit))
  ok <- !is.na(coef(fit))
  nscale <- ncol(fit$var) - p
  # The covariates as survreg.fit scales them.
  X <- model.matrix(fit)
  tmat <- diag(1, p + nscale)
  if (all(X[, 1] == 1) && ncol(X) > 1) {
    okay <- apply(X, 2, function(z) all(z == 0 | z == 1))
    if (!all(okay)) {
      center <- ifelse(okay, 0, colMeans(X))
      stdev <- ifelse(okay, 1, apply(X, 2, sd))
      tmat <- diag(c(1 / stdev, rep(1, nscale)))
      tmat[1, 2:p] <- -center[2:p] / stdev[2:p]
    }
  }
  keep <- c(ok, rep(TRUE, nscale))
  vs <- solve(tmat) %*% fit$var %*% t(solve(tmat))
  condition <- kappa(vs[keep, keep, drop = FALSE], exact = TRUE)
  idx <- 1:25
  nd <- d[idx, ]
  pr <- function(type, ...) predict(fit, newdata = nd, type = type, se.fit = TRUE, ...)
  p_lp <- pr("lp")
  p_resp <- pr("response")
  p_q <- pr("quantile", p = c(0.1, 0.5, 0.9))
  res <- lapply(residual_types, function(t) as.numeric(residuals(fit, type = t)))
  names(res) <- residual_types
  fits[[length(fits) + 1]] <- list(
    name = name, dataset = dname, distribution = dist,
    use_strata = use_strata, use_weights = use_weights, fixed_scale = fixed,
    x = na_matrix(X),
    coefficients = I(unname(coef(fit))),
    scale = I(unname(fit$scale)),
    log_scale = I(log(unname(fit$scale))),
    variance = na_matrix(fit$var),
    loglik = fit$loglik[2], loglik_null = fit$loglik[1],
    iterations = fit$iter, df = fit$df,
    linear_predictors = as.numeric(fit$linear.predictors),
    information_condition = condition,
    predict_rows = length(idx),
    lp = as.numeric(p_lp$fit), lp_se = as.numeric(p_lp$se.fit),
    response = as.numeric(p_resp$fit), response_se = as.numeric(p_resp$se.fit),
    quantile = na_matrix(p_q$fit), quantile_se = na_matrix(p_q$se.fit),
    residuals = res
  )
}
col <- function(d, v) if (v %in% names(d)) d[[v]] else rep(NA_real_, nrow(d))
data_out <- lapply(datasets, function(d) list(
  time = col(d, "time"), event = col(d, "event"), time2 = col(d, "time2"),
  status = col(d, "status"), stratum = col(d, "stratum"), weight = col(d, "weight")
))
cat(toJSON(list(
  versions = list(
    r = R.version.string,
    survival = as.character(packageVersion("survival")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  datasets = data_out, fits = fits
), digits = I(17), auto_unbox = TRUE, na = "null"))
