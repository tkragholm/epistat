# Flexible parametric survival models (Royston-Parmar) against R references.
#
# Four sources:
#  * survreg: with no interior knot the three scales are the Weibull, log-logistic
#    and log-normal accelerated failure time models, gamma_1 = 1 / scale,
#    beta = -coef / scale, with survreg's log likelihood, which is that of the
#    density of the time and so includes the Jacobian of the log time.
#  * flexsurv::flexsurvspline (scale hazard, odds, normal; weights; delayed entry;
#    time-varying effects through anc, which share the baseline knots).
#  * rstpm2::stpm2 (the same scales, weights, delayed entry or counting-process
#    rows, time-varying effects with their own degrees of freedom) with its
#    predictions of the survivor function, hazard, cumulative hazard and hazard
#    ratio and their limits.
#  * a transcription of the likelihood in this script, maximised by BFGS from a
#    crude start and polished by Newton steps with numDeriv derivatives, which is
#    checked against the packages above, and gives the covariance, the robust
#    covariance and the delta-method standard errors of predictions by the
#    Jacobian of numDeriv.
# For each fit: the data set, the settings, the log likelihood, the coefficients
# of the covariates (only where they do not depend on the basis), the survivor
# function at ten times for two covariate profiles and what the source gives
# for the hazard, cumulative hazard and hazard ratio.
# The versions of R and the packages are in the output. Numbers are written with
# 17 significant digits.
# Run: Rscript flexpar.R > flexpar.json
suppressMessages({
  library(jsonlite); library(survival); library(flexsurv); library(rstpm2)
  library(numDeriv); library(Hmisc); library(bbmle)
})
set.seed(20261123)

make_data <- function(n, entry, seed_shift) {
  d <- data.frame(x1 = round(rnorm(n), 3), x2 = rbinom(n, 1, 0.4))
  d$x3 <- round(2 * d$x1 - 0.5, 3)
  tt <- round(rweibull(n, 1.3, exp(-(0.5 * d$x1 - 0.4 * d$x2) / 1.3) * 2), 2) + 0.01
  cc <- round(rexp(n, 0.2), 2) + 0.01
  d$time <- pmin(tt, cc)
  d$event <- as.numeric(tt <= cc)
  d$entry <- 0
  if (entry) {
    d$entry <- ifelse(runif(n) < 0.4, round(runif(n, 0, 0.6) * d$time, 2), 0)
    d <- d[d$entry < d$time, ]
  }
  d$weight <- round(runif(nrow(d), 0.5, 2.5), 2)
  d$cluster <- rep(seq_len(ceiling(nrow(d) / 4)), each = 4)[seq_len(nrow(d))]
  rownames(d) <- NULL
  d
}
dA <- make_data(400, FALSE)
dB <- make_data(400, TRUE)
# Counting-process rows: each subject of dA is cut at 1 and 2.5, and a covariate
# switches on at 1 for the subjects with x2 = 1.
split_rows <- function(d) {
  d$id <- seq_len(nrow(d))
  cuts <- c(1, 2.5)
  out <- list()
  for (i in seq_len(nrow(d))) {
    r <- d[i, ]
    edges <- c(0, cuts[cuts < r$time], r$time)
    for (k in seq_len(length(edges) - 1)) {
      row <- r
      row$entry <- edges[k]; row$time <- edges[k + 1]
      row$event <- if (k == length(edges) - 1) r$event else 0
      row$x4 <- as.numeric(r$x2 == 1 && edges[k] >= 1)
      out[[length(out) + 1]] <- row
    }
  }
  o <- do.call(rbind, out); rownames(o) <- NULL; o
}
dC <- split_rows(dA)
dC$weight <- 1  # the counting-process rows are unweighted
datasets <- list(A = dA, B = dB, C = dC)

times10 <- c(0.1, 0.3, 0.6, 1, 1.5, 2, 3, 4, 5, 6)
profiles <- list(c(0.5, 1), c(-0.7, 0))
scale_names <- c(hazard = "PH", odds = "PO", probit = "probit")
flex_scales <- c(hazard = "hazard", odds = "odds", probit = "normal")

# ---- transcription ----
rcs <- function(u, knots) {
  k <- length(knots)
  if (k == 2) return(list(v = matrix(u, ncol = 1), d = matrix(1, length(u), 1)))
  tk <- knots; D <- (tk[k] - tk[1])^2; gap <- tk[k] - tk[k - 1]
  v <- matrix(u, ncol = 1); d <- matrix(1, length(u), 1)
  for (j in seq_len(k - 2)) {
    a <- (tk[k] - tk[j]) / gap; b <- (tk[k - 1] - tk[j]) / gap
    v <- cbind(v, (pmax(u - tk[j], 0)^3 - pmax(u - tk[k - 1], 0)^3 * a + pmax(u - tk[k], 0)^3 * b) / D)
    d <- cbind(d, 3 * (pmax(u - tk[j], 0)^2 - pmax(u - tk[k - 1], 0)^2 * a + pmax(u - tk[k], 0)^2 * b) / D)
  }
  list(v = v, d = d)
}
# the basis agrees with Hmisc's
chk <- seq(-2, 3, length.out = 9)
kk <- c(-1.5, -0.2, 0.7, 1.8)
stopifnot(max(abs(rcs(chk, kk)$v[, -1] - rcspline.eval(chk, kk, inclx = FALSE))) < 1e-12)

qknots <- function(logev, df) as.numeric(quantile(logev, seq(0, 1, length.out = df + 1)))
design_rows <- function(u, X, bk, tvc) {
  b <- rcs(u, bk)
  z <- cbind(1, b$v, X); w <- cbind(0, b$d, matrix(0, length(u), ncol(X)))
  for (t in tvc) {
    tb <- rcs(u, t$knots)
    z <- cbind(z, X[, t$col] * tb$v); w <- cbind(w, X[, t$col] * tb$d)
  }
  list(z = z, w = w)
}
Hfun <- function(scale, eta) switch(scale, hazard = exp(eta), odds = log1p(exp(eta)), probit = -pnorm(-eta, log.p = TRUE))
logm <- function(scale, eta) switch(scale, hazard = eta, odds = eta - log1p(exp(eta)),
  probit = dnorm(eta, log = TRUE) - pnorm(-eta, log.p = TRUE))
loglik_fn <- function(theta, d, X, scale, bk, tvc, offset = 0) {
  u <- log(d$time); r <- design_rows(u, X, bk, tvc)
  eta <- drop(r$z %*% theta) + offset; ws <- drop(r$w %*% theta)
  if (any(ws[d$event == 1 & d$weight > 0] <= 0)) return(-1e100)
  ll <- d$weight * (ifelse(d$event == 1, logm(scale, eta) + log(pmax(ws, 1e-300)) - u, 0) - Hfun(scale, eta))
  ent <- d$entry > 0
  if (any(ent)) {
    r0 <- design_rows(log(d$entry[ent]), X[ent, , drop = FALSE], bk, tvc)
    eta0 <- drop(r0$z %*% theta) + (if (length(offset) > 1) offset[ent] else offset)
    ll[ent] <- ll[ent] + d$weight[ent] * Hfun(scale, eta0)
  }
  sum(ll)
}
# Analytic score and information of the log likelihood, written for matrices.
link_derivs <- function(scale, eta) {
  switch(scale,
    hazard = list(H1 = exp(eta), H2 = exp(eta), g1 = rep(1, length(eta)), g2 = rep(0, length(eta))),
    odds = { p <- plogis(eta); list(H1 = p, H2 = p * (1 - p), g1 = 1 - p, g2 = -p * (1 - p)) },
    probit = { l <- dnorm(eta) / pnorm(-eta); list(H1 = l, H2 = l * (l - eta), g1 = -eta + l, g2 = -1 + l * (l - eta)) })
}
score_info <- function(theta, d, X, scale, bk, tvc, per_row = FALSE) {
  u <- log(d$time); r <- design_rows(u, X, bk, tvc)
  eta <- drop(r$z %*% theta); ws <- drop(r$w %*% theta)
  ev <- d$event == 1; c <- d$weight
  L <- link_derivs(scale, eta)
  wsafe <- ifelse(ev, ws, 1)
  sz <- c * (ifelse(ev, L$g1, 0) - L$H1); sw <- c * ifelse(ev, 1 / wsafe, 0)
  a <- c * (L$H2 - ifelse(ev, L$g2, 0)); b <- c * ifelse(ev, 1 / wsafe^2, 0)
  ent <- d$entry > 0
  S <- sz * r$z + sw * r$w
  info <- crossprod(r$z, a * r$z) + crossprod(r$w, b * r$w)
  if (any(ent)) {
    r0 <- design_rows(log(d$entry[ent]), X[ent, , drop = FALSE], bk, tvc)
    L0 <- link_derivs(scale, drop(r0$z %*% theta))
    S[ent, ] <- S[ent, ] + (c[ent] * L0$H1) * r0$z
    info <- info - crossprod(r0$z, (c[ent] * L0$H2) * r0$z)
  }
  list(score = colSums(S), info = info, rows = S)
}
fit_transcription <- function(d, covs, scale, df, tvc_spec) {
  X <- as.matrix(d[, covs, drop = FALSE])
  logev <- sort(log(d$time[d$event == 1 & d$weight > 0]))
  bk <- qknots(logev, df)
  tvc <- lapply(tvc_spec, function(t) list(col = t$col, knots = qknots(logev, t$df)))
  q <- 1 + df + ncol(X) + sum(unlist(lapply(tvc, function(t) length(t$knots) - 1)))
  # start from the straight-line Nelson-Aalen fit
  na <- survfit(Surv(entry, time, event) ~ 1, data = d, weights = weight, ctype = 1)
  H <- na$cumhaz; tm <- na$time; ok <- na$n.event > 0 & H > 0
  y <- switch(scale, hazard = log(H), odds = log(expm1(H)), probit = -qnorm(exp(-H)))
  fitl <- lm(y[ok] ~ log(tm[ok]), weights = na$n.event[ok])
  theta <- numeric(q); theta[1] <- coef(fitl)[1]; theta[2] <- max(coef(fitl)[2], 0.5)
  f <- function(th) loglik_fn(th, d, X, scale, bk, tvc)
  # the numerical optimiser finds the neighbourhood, then Newton with the analytic derivatives
  o <- optim(theta, function(th) -f(th), method = "BFGS", control = list(maxit = 5000, reltol = 1e-15))
  theta <- o$par
  for (it in 1:100) {
    si <- score_info(theta, d, X, scale, bk, tvc)
    step <- solve(si$info, si$score)
    alpha <- 1
    while (alpha > 1e-10 && !(f(theta + alpha * step) >= f(theta) - 1e-13)) alpha <- alpha / 2
    theta <- theta + alpha * step
    if (max(abs(step)) < 1e-12) break
  }
  si <- score_info(theta, d, X, scale, bk, tvc)
  list(theta = theta, V = solve(si$info), ll = f(theta), grad_norm = sqrt(sum(si$score^2)), X = X, bk = bk, tvc = tvc, df = df, scale = scale,
    rows = si$rows)
}
robust_cov <- function(fit, d) {
  S <- rowsum(fit$rows, d$cluster)
  fit$V %*% crossprod(S) %*% fit$V
}
quantity <- function(fit, x, t, what) {
  # function of theta giving the quantity at covariate row x and time t
  function(th) {
    u <- log(t)
    r <- design_rows(u, matrix(x, nrow = 1), fit$bk, fit$tvc)
    eta <- sum(r$z * th); ws <- sum(r$w * th)
    m <- (function(e) switch(fit$scale, hazard = exp(e), odds = plogis(e), probit = dnorm(e) / pnorm(-e)))(eta)
    switch(what,
      surv = exp(-Hfun(fit$scale, eta)),
      cumhaz = Hfun(fit$scale, eta),
      hazard = m * ws / t,
      loghazard = log(m * ws / t))
  }
}
predictions <- function(fit, profiles, times, cov) {
  out <- list()
  for (p in seq_along(profiles)) for (what in c("surv", "cumhaz", "hazard")) {
    est <- se <- numeric(length(times))
    for (i in seq_along(times)) {
      f <- quantity(fit, profiles[[p]], times[i], what)
      est[i] <- f(fit$theta)
      g <- jacobian(f, fit$theta, method.args = list(eps = 1e-5, d = 1e-4, r = 6))
      se[i] <- sqrt(drop(g %*% cov %*% t(g)))
    }
    out[[paste0(what, "_", p)]] <- list(estimate = est, se = se)
  }
  # contrasts of profile 1 against profile 2
  hr_est <- hr_se <- sd_est <- sd_se <- hd_est <- hd_se <- numeric(length(times))
  for (i in seq_along(times)) {
    lh <- function(th) quantity(fit, profiles[[1]], times[i], "loghazard")(th) - quantity(fit, profiles[[2]], times[i], "loghazard")(th)
    sf <- function(th) quantity(fit, profiles[[1]], times[i], "surv")(th) - quantity(fit, profiles[[2]], times[i], "surv")(th)
    hf <- function(th) quantity(fit, profiles[[1]], times[i], "hazard")(th) - quantity(fit, profiles[[2]], times[i], "hazard")(th)
    for (nm in c("lh", "sf", "hf")) {
      f <- get(nm); g <- jacobian(f, fit$theta, method.args = list(eps = 1e-5, d = 1e-4, r = 6))
      s <- sqrt(drop(g %*% cov %*% t(g))); v <- f(fit$theta)
      if (nm == "lh") { hr_est[i] <- exp(v); hr_se[i] <- s }
      if (nm == "sf") { sd_est[i] <- v; sd_se[i] <- s }
      if (nm == "hf") { hd_est[i] <- v; hd_se[i] <- s }
    }
  }
  out$hazard_ratio <- list(estimate = hr_est, se_log = hr_se)
  out$survival_difference <- list(estimate = sd_est, se = sd_se)
  out$hazard_difference <- list(estimate = hd_est, se = hd_se)
  out
}

fits <- list()
add <- function(x) fits[[length(fits) + 1]] <<- x
cov_names <- c("x1", "x2")

# ---- 1. survreg parity, df = 1 ----
for (sc in names(flex_scales)) {
  d <- dA
  dist <- c(hazard = "weibull", odds = "loglogistic", probit = "lognormal")[[sc]]
  sr <- survreg(Surv(time, event) ~ x1 + x2, data = d, weights = weight, dist = dist)
  sigma <- sr$scale
  add(list(name = paste0("survreg_", sc), source = "survreg", dataset = "A", scale = sc, df = 1, weighted = TRUE, knots = NULL,
    covariates = cov_names, tvc = list(),
    loglik = sr$loglik[2],
    gamma = c(-coef(sr)[1] / sigma, 1 / sigma),
    beta = unname(-coef(sr)[2:3] / sigma),
    theta_cov = NULL))
}

# ---- 2. flexsurv ----
# Unweighted fits use the default knots, which both packages and this crate place
# at the type 7 quantiles of the log event times with the extremes as boundary
# knots. flexsurv places them at weighted quantiles when there are weights, so
# the weighted fits are given the unweighted knots.
default_knots <- function(d, df) {
  le <- log(d$time[d$event == 1])
  as.numeric(quantile(le, seq(0, 1, length.out = df + 1)))
}
flexfit <- function(d, sc, df, entry, weighted, anc = NULL) {
  fm <- if (entry) Surv(entry, time, event) ~ x1 + x2 else Surv(time, event) ~ x1 + x2
  kn <- default_knots(d, df)
  args <- list(fm, data = d, scale = flex_scales[[sc]], anc = anc, method = "BFGS",
    control = list(reltol = 1e-15, maxit = 2000))
  if (weighted) args$weights <- d$weight
  args$knots <- kn[2:df]; args$bknots <- kn[c(1, df + 1)]
  if (df == 1) { args$knots <- NULL; args$k <- 0 }
  do.call(flexsurvspline, args)
}
nd_profiles <- data.frame(x1 = sapply(profiles, `[`, 1), x2 = sapply(profiles, `[`, 2))
for (spec in list(c("A", FALSE), c("B", TRUE))) for (sc in names(flex_scales)) for (df in c(3, 4)) {
  d <- datasets[[spec[1]]]
  weighted <- spec[1] == "B"
  f <- flexfit(d, sc, df, as.logical(spec[2]), weighted)
  nd <- nd_profiles
  surv <- lapply(1:2, function(p) summary(f, newdata = nd[p, ], t = times10, type = "survival", ci = FALSE, tidy = TRUE)$est)
  haz <- lapply(1:2, function(p) summary(f, newdata = nd[p, ], t = times10, type = "hazard", ci = FALSE, tidy = TRUE)$est)
  cum <- lapply(1:2, function(p) summary(f, newdata = nd[p, ], t = times10, type = "cumhaz", ci = FALSE, tidy = TRUE)$est)
  add(list(name = paste("flexsurv", spec[1], sc, df, sep = "_"), source = "flexsurv", dataset = spec[1], scale = sc, df = df,
    weighted = weighted, knots = if (weighted) default_knots(d, df) else NULL,
    covariates = cov_names, tvc = list(), loglik = as.numeric(logLik(f)),
    beta = unname(f$res[c("x1", "x2"), "est"]), beta_se = unname(f$res[c("x1", "x2"), "se"]),
    surv = surv, hazard = haz, cumhaz = cum))
}
# flexsurv time-varying effect of x2 sharing the baseline knots (df 3)
for (sc in names(flex_scales)) {
  d <- dB
  f <- flexfit(d, sc, 3, TRUE, TRUE, anc = list(gamma1 = ~x2, gamma2 = ~x2, gamma3 = ~x2))
  surv <- lapply(1:2, function(p) summary(f, newdata = nd_profiles[p, ], t = times10, type = "survival", ci = FALSE, tidy = TRUE)$est)
  add(list(name = paste("flexsurv_tvc_B", sc, 3, sep = "_"), source = "flexsurv", dataset = "B", scale = sc, df = 3,
    weighted = TRUE, knots = default_knots(d, 3),
    covariates = cov_names, tvc = list(list(column = 1, df = 3)), loglik = as.numeric(logLik(f)),
    surv = surv))
}

# ---- 3. rstpm2 ----
# rstpm2 places the knots unweighted, so its fits can be weighted with the default knots.
rstfit <- function(d, sc, df, entry, weighted, tvc = NULL, covs = "x1 + x2") {
  lhs <- if (entry) "Surv(entry, time, event)" else "Surv(time, event)"
  fm <- as.formula(paste(lhs, "~", covs))
  if (weighted) stpm2(fm, data = d, df = df, link.type = scale_names[[sc]], weights = weight, tvc = tvc, control = list(reltol.search = 1e-14, reltol.final = 1e-14))
  else stpm2(fm, data = d, df = df, link.type = scale_names[[sc]], tvc = tvc, control = list(reltol.search = 1e-14, reltol.final = 1e-14))
}
rst_preds <- function(f, nd) {
  pr <- function(type, ...) lapply(1:2, function(p) {
    nn <- nd[p, , drop = FALSE][rep(1, length(times10)), ]; nn$time <- times10
    as.data.frame(predict(f, newdata = nn, type = type, se.fit = TRUE, ...))
  })
  list(surv = pr("surv"), hazard = pr("hazard"), cumhaz = pr("cumhaz"))
}
for (spec in list(c("A", FALSE), c("B", TRUE))) for (sc in names(flex_scales)) for (df in c(3, 4)) {
  d <- datasets[[spec[1]]]
  weighted <- spec[1] == "B"
  f <- rstfit(d, sc, df, as.logical(spec[2]), weighted)
  P <- rst_preds(f, nd_profiles)
  add(list(name = paste("rstpm2", spec[1], sc, df, sep = "_"), source = "rstpm2", dataset = spec[1], scale = sc, df = df,
    weighted = weighted, knots = NULL,
    covariates = cov_names, tvc = list(), loglik = as.numeric(logLik(f)),
    beta = unname(coef(f)[c("x1", "x2")]), beta_se = unname(sqrt(diag(vcov(f)))[c("x1", "x2")]),
    surv = lapply(P$surv, as.list), hazard = lapply(P$hazard, as.list), cumhaz = lapply(P$cumhaz, as.list)))
}
# rstpm2 time-varying effects with their own degrees of freedom, with entry
for (sc in names(flex_scales)) for (cfg in list(c(3, 2), c(4, 3))) {
  d <- dB
  f <- rstfit(d, sc, cfg[1], TRUE, FALSE, tvc = list(x2 = cfg[2]))
  P <- rst_preds(f, nd_profiles)
  hr <- as.data.frame(predict(f, newdata = data.frame(x1 = 0.5, x2 = 0, time = times10), type = "hr", var = "x2",
    exposed = function(data) transform(data, x2 = 1), se.fit = TRUE))
  add(list(name = paste("rstpm2_tvc_B", sc, cfg[1], cfg[2], sep = "_"), source = "rstpm2", dataset = "B", scale = sc,
    df = cfg[1], weighted = FALSE, knots = NULL, covariates = cov_names, tvc = list(list(column = 1, df = cfg[2])),
    loglik = as.numeric(logLik(f)),
    surv = lapply(P$surv, as.list), hazard = lapply(P$hazard, as.list), cumhaz = lapply(P$cumhaz, as.list),
    hr_profile = c(0.5, 0), hr = as.list(hr)))
}
# rstpm2 on counting-process rows with a covariate that changes at 1, hazard scale
{
  f <- stpm2(Surv(entry, time, event) ~ x1 + x4, data = dC, df = 3, control = list(reltol.search = 1e-14, reltol.final = 1e-14))
  add(list(name = "rstpm2_C_hazard_3", source = "rstpm2", dataset = "C", scale = "hazard", df = 3, weighted = FALSE, knots = NULL,
    covariates = c("x1", "x4"), tvc = list(), loglik = as.numeric(logLik(f)),
    beta = unname(coef(f)[c("x1", "x4")]), beta_se = unname(sqrt(diag(vcov(f)))[c("x1", "x4")])))
}

# ---- 4. transcription ----
trans_cases <- list(
  list(ds = "B", sc = "hazard", df = 2, tvc = list()),
  list(ds = "B", sc = "hazard", df = 4, tvc = list(list(col = 2, df = 2))),
  list(ds = "B", sc = "hazard", df = 6, tvc = list()),
  list(ds = "B", sc = "odds", df = 3, tvc = list(list(col = 2, df = 3))),
  list(ds = "B", sc = "odds", df = 5, tvc = list()),
  list(ds = "B", sc = "probit", df = 3, tvc = list(list(col = 2, df = 2))),
  list(ds = "B", sc = "probit", df = 5, tvc = list()),
  list(ds = "A", sc = "hazard", df = 3, tvc = list(list(col = 1, df = 2), list(col = 2, df = 1))),
  list(ds = "C", sc = "hazard", df = 3, tvc = list(), covs = c("x1", "x4"))
)
for (tc in trans_cases) {
  d <- datasets[[tc$ds]]
  covs <- if (is.null(tc$covs)) cov_names else tc$covs
  fit <- fit_transcription(d, covs, tc$sc, tc$df, tc$tvc)
  prof <- if (tc$ds == "C") list(c(0.5, 1), c(-0.7, 0)) else profiles
  # the tvc columns refer to the columns of X
  pr <- predictions(fit, prof, times10, fit$V)
  rob <- robust_cov(fit, d)
  tvcs <- lapply(tc$tvc, function(t) list(column = t$col - 1, df = t$df))
  add(list(name = paste("transcription", tc$ds, tc$sc, tc$df, length(tc$tvc), sep = "_"), source = "transcription",
    dataset = tc$ds, scale = tc$sc, df = tc$df, weighted = TRUE, covariates = covs, tvc = tvcs,
    loglik = fit$ll, grad_norm = fit$grad_norm, theta = fit$theta, cov = as.vector(t(fit$V)), robust_cov = as.vector(t(rob)),
    knots = fit$bk, profiles = prof, predictions = pr))
}

data_out <- lapply(datasets, function(d) {
  l <- list(time = d$time, start = d$entry, event = d$event, weight = d$weight, cluster = d$cluster - 1,
    x1 = d$x1, x2 = d$x2, x3 = d$x3)
  if (!is.null(d$x4)) { l$x4 <- d$x4; l$id <- d$id - 1 }
  l
})
cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
    flexsurv = as.character(packageVersion("flexsurv")), rstpm2 = as.character(packageVersion("rstpm2")),
    numDeriv = as.character(packageVersion("numDeriv")), Hmisc = as.character(packageVersion("Hmisc")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  times = times10, profiles = profiles, data = data_out, fits = fits),
  digits = 17, auto_unbox = TRUE, null = "null", na = "null"))
