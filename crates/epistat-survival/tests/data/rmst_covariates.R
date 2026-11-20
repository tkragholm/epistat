# survRM2::rmst2 with covariates: the regression of the restricted mean on the arm
# and covariates by inverse probability of censoring weighting (Tian et al. 2014),
# for the difference, the ratio and the ratio of times lost.
#
# Data: survival::lung (sex as the arm, age and ph.ecog as covariates) and
# simulated sets with tied integer times, one and two covariates, and an arm whose
# last time is a censoring. `r_stop_error` is how far the coefficients of the
# default glm (stopping on a change of the deviance of 1e-8) are from those of
# the same weights and equation solved with epsilon 1e-14, which is R's own error
# in the ratio and loss ratio fits; the weights are formed as rmst2reg forms them.
# Run: Rscript rmst_covariates.R > rmst_covariates.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(survRM2)
})
set.seed(20261123)
v <- function(x) I(as.numeric(x))

sim <- function(n, last_censored = FALSE) {
  arm <- rbinom(n, 1, 0.5)
  x1 <- round(rnorm(n), 2)
  x2 <- rbinom(n, 1, 0.4)
  time <- pmin(pmax(1, ceiling(rexp(n, 0.15 * exp(0.3 * x1 - 0.5 * arm)))), 24)
  status <- rbinom(n, 1, 0.6)
  if (last_censored) {
    i <- which(arm == 1)[1]
    time[i] <- 30
    status[i] <- 0
    j <- which(arm == 0)[1]
    time[j] <- 30
    status[j] <- 1
  }
  data.frame(time = time, status = status, arm = arm, x1 = x1, x2 = x2)
}
lungd <- na.omit(with(lung, data.frame(time = time, status = status - 1, arm = sex - 1,
                                       x1 = age, x2 = ph.ecog)))
datasets <- list(
  lung = lungd,
  sim_a = sim(150),
  sim_b = sim(90, TRUE),
  sim_c = sim(60)
)

# The weights as rmst2reg forms them, and the fit of one type to tolerance eps.
tight <- function(d, covs, tau, type) {
  x <- cbind(1, d$arm, as.matrix(d[covs]))
  y0 <- pmin(d$time, tau)
  d0 <- d$status
  d0[y0 == tau] <- 1
  w <- numeric(nrow(d))
  for (a in 0:1) {
    i <- which(d$arm == a)
    i <- i[order(y0[i])]
    f <- survRM2:::func_surv(y0[i], 1 - d0[i])
    w[i] <- d0[i] / rep(f$surv, table(y0[i]))
  }
  ctrl <- glm.control(epsilon = 1e-14, maxit = 200)
  if (type == "difference") {
    b <- coef(lm(y0 ~ x - 1, weights = w))
    mu <- as.vector(x %*% b)
  } else {
    yy <- if (type == "ratio") y0 else tau - y0
    b <- coef(glm(yy ~ x - 1, family = "quasipoisson", weights = w, control = ctrl))
    mu <- exp(as.vector(x %*% b))
  }
  # The 2-norm condition number of the matrix the fit solves with: X' W X for the
  # identity link and X' W diag(mu) X for the log link.
  m <- t(x * w * (if (type == "difference") 1 else mu)) %*% x
  list(beta = b, cond = kappa(m, exact = TRUE))
}

run <- function(d, covs, tau, alpha) {
  cv <- if (length(covs)) as.matrix(d[covs]) else NULL
  out <- rmst2(d$time, d$status, d$arm, tau = tau, covariates = cv, alpha = alpha)
  one <- function(m, type) {
    tt <- tight(d, covs, out$tau, type)
    r <- list(coef = v(m[, "coef"]), se = v(m[, "se(coef)"]), z = v(m[, "z"]), p = v(m[, "p"]),
              lower = v(m[, ncol(m) - 1]), upper = v(m[, ncol(m)]),
              stop_error = max(abs(m[, "coef"] - tt$beta)), cond = tt$cond)
    if (type != "difference") r$ratio <- v(m[, "exp(coef)"])
    r
  }
  list(covs = I(covs), tau_given = if (is.null(tau)) NA else tau, tau = out$tau, alpha = alpha,
       difference = one(out$RMST.difference.adjusted, "difference"),
       ratio = one(out$RMST.ratio.adjusted, "ratio"),
       loss_ratio = one(out$RMTL.ratio.adjusted, "lossratio"))
}

cases <- list()
for (nm in names(datasets)) {
  d <- datasets[[nm]]
  mx <- max(d$time)
  taus <- if (nm == "lung") list(NULL, 300, 600, 800) else list(NULL, 0.5 * mx, 0.8 * mx)
  covsets <- if (nm == "sim_c") list("x1") else list(c("x1"), c("x1", "x2"))
  fits <- list()
  for (cs in covsets) for (tau in taus) fits[[length(fits) + 1]] <- run(d, cs, tau, 0.05)
  fits[[length(fits) + 1]] <- run(d, covsets[[1]], taus[[2]], 0.1)
  cases[[length(cases) + 1]] <- list(name = nm, fits = fits)
}
ds <- lapply(datasets, function(d) c(lapply(d, v)))
versions <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
                 survRM2 = as.character(packageVersion("survRM2")),
                 jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = versions, datasets = ds, cases = cases),
           digits = I(17), auto_unbox = TRUE, na = "null"))
