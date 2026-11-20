# The curves of survfit.coxph that cox_predict.R does not cover, from survival
# 3.8-12: the Kalbfleisch-Prentice curve (stype = 1), start.time, the counts of
# censored rows at every exit time (censor = TRUE), and curves for covariates
# that change over time (newdata in counting-process form with id).
# Fits: lung with strata and case weights, lung as counting-process data with
# start 0 and Efron ties, and simulated counting-process data with strata and
# case weights, with an offset, and with Breslow and Efron ties.
# For each fit: the design matrix, the coefficients, the variance coxph keeps,
# the data, the new rows with their relative risk, and for each combination of
# stype, start.time and censor the curve of every new row, with log and log-log
# limits. For fits on counting-process data, the curves of four subjects with
# changing covariates; the fourth has a gap between its two intervals, where
# survfit.coxph shifts the times after the gap back by its length, which is
# written down as gap_start and gap_length.
# The Kalbfleisch-Prentice increment at a time with tied events is the root of
# sum w_j r_j / (1 - a^r_j) = nrisk, which survival finds by 35 bisection steps
# from 0.5 with step 0.25. Its error is therefore up to 2^-36 in a, and the
# test allows for it.
# Run: Rscript cox_curve.R > cox_curve.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)

# ---- data, as in cox_predict.R
n <- 260
x1 <- round(rnorm(n), 3)
x2 <- round(runif(n, 0, 2), 3)
x3 <- rbinom(n, 1, 0.45)
g <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
stratum <- sample(0:2, n, replace = TRUE)
stop <- pmax(ceiling(rexp(n, 0.08 * exp(0.4 * x1 - 0.5 * x3))), 2)
start <- pmin(sample(0:6, n, replace = TRUE), stop - 1)
event <- rbinom(n, 1, 0.75)
w <- round(runif(n, 0.5, 3), 2)
id <- sample(rep(1:60, length.out = n))
off <- round(rnorm(n, 0.5, 0.6), 3)
sim <- data.frame(start, stop, event, x1, x2, x3, g, stratum, w, id, off)

l <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog", "wt.loss")])
lw <- round(runif(nrow(l), 0.5, 2), 2)
lg <- factor(c("low", "mid", "high")[1 + (l$ph.ecog >= 1) + (l$ph.ecog >= 2)],
             levels = c("low", "mid", "high"))
lungd <- data.frame(start = 0, stop = l$time, event = as.integer(l$status == 2),
                    x1 = l$age, x2 = l$wt.loss, x3 = as.integer(l$sex == 2),
                    g = lg, stratum = l$sex, w = lw, id = seq_len(nrow(l)),
                    off = 0)

sim_new <- data.frame(
  x1 = c(-0.5, 0.3, 1.2, 0, 0.8), x2 = c(0.4, 1.5, 0.9, 1, 0.2),
  x3 = c(0, 1, 1, 0, 1), stratum = c(0, 1, 2, 1, 0), off = c(0.2, -0.3, 0, 0.5, 0.1))
lung_new <- data.frame(
  x1 = c(50, 62, 70, 58), x2 = c(0, 10, -5, 20), x3 = c(0, 1, 1, 0),
  stratum = c(1, 2, 1, 2), off = 0)

# four subjects with changing covariates, in counting-process form
sim_path <- data.frame(
  id = c(1, 1, 1, 2, 2, 3, 4, 4),
  start = c(0, 5, 12, 2, 8, 0, 0, 9),
  stop = c(5, 12, 30, 8, 20, 40, 6, 25),
  x1 = c(-0.5, 0.2, 1.0, 0.3, 0.9, 0.1, -1.0, 0.5),
  x2 = c(0.4, 0.8, 1.6, 1.5, 0.3, 1.0, 0.2, 1.9),
  x3 = c(0, 0, 1, 1, 0, 1, 0, 1),
  stratum = c(0, 0, 0, 1, 2, 1, 2, 2),
  off = c(0.2, 0.1, 0, -0.3, 0.2, 0.5, 0, 0.1), event = 1)
lung_path <- data.frame(
  id = c(1, 1, 1, 2, 2, 3, 4, 4),
  start = c(0, 100, 300, 50, 250, 0, 0, 400),
  stop = c(100, 300, 700, 250, 600, 900, 350, 800),
  x1 = c(55, 56, 57, 65, 66, 60, 70, 71),
  x2 = c(0, 5, 12, 10, -4, 8, 3, 15),
  x3 = c(0, 0, 1, 1, 1, 0, 1, 0),
  stratum = 1, off = 0, event = 1)
sim_gap <- list(start = 6, length = 3)
lung_gap <- list(start = 350, length = 50)

specs <- list(
  list(name = "lung_breslow_strata_weights", d = "lung", terms = "x1 + x2", ties = "breslow", strata = TRUE, weights = TRUE, st = 150),
  list(name = "lung_efron_counting_weights", d = "lung", terms = "x1 + x2 + x3", ties = "efron", entry = TRUE, weights = TRUE, st = 150, paths = TRUE),
  list(name = "sim_breslow_counting_strata_weights", d = "sim", terms = "x1 + x2 + x3", ties = "breslow", entry = TRUE, strata = TRUE, weights = TRUE, st = 8, paths = TRUE),
  list(name = "sim_efron_counting_strata_weights", d = "sim", terms = "x1 + x2 + x3", ties = "efron", entry = TRUE, strata = TRUE, weights = TRUE, st = 8, paths = TRUE),
  list(name = "sim_efron_counting_strata_offset", d = "sim", terms = "x1 + x3", ties = "efron", entry = TRUE, strata = TRUE, offset = TRUE, st = 8, paths = TRUE),
  list(name = "sim_breslow_nostart_weights", d = "sim", terms = "x1 + x2", ties = "breslow", weights = TRUE, st = 6)
)

limits_of <- function(sf) {
  out <- list()
  for (ct in c("log", "log-log")) {
    ci <- survfit_confint(sf$surv, sf$std.err, logse = TRUE, ct, 0.95)
    out[[ct]] <- list(lower = ci$lower, upper = ci$upper)
  }
  out
}
# survfit_confint is survival's own, used so the limits of a curve with
# surv = 0 follow it
survfit_confint <- survival:::survfit_confint

one_curve <- function(sf) {
  list(time = sf$time, n_risk = sf$n.risk, n_event = sf$n.event,
       n_censor = sf$n.censor, surv = sf$surv, cumhaz = sf$cumhaz,
       std_err = sf$std.err, limits = limits_of(sf))
}

run <- function(s) {
  d <- if (s$d == "lung") lungd else sim
  nd <- if (s$d == "lung") lung_new else sim_new
  resp <- if (isTRUE(s$entry)) "Surv(start, stop, event)" else "Surv(stop, event)"
  rhs <- s$terms
  if (isTRUE(s$strata)) rhs <- paste(rhs, "+ strata(stratum)")
  if (isTRUE(s$offset)) rhs <- paste(rhs, "+ offset(off)")
  f <- as.formula(paste(resp, "~", rhs))
  args <- list(formula = f, data = d, ties = s$ties)
  if (isTRUE(s$weights)) args$weights <- d$w
  fit <- do.call(coxph, args)
  X <- model.matrix(fit)
  cn <- colnames(X)
  beta <- coef(fit)
  newX <- model.matrix(as.formula(paste("~", s$terms)), nd)[, -1, drop = FALSE]
  stopifnot(identical(colnames(newX), cn))
  wts <- if (isTRUE(s$weights)) d$w else rep(1, nrow(d))
  offset_mean <- if (isTRUE(s$offset)) sum(d$off * wts / sum(wts)) else 0
  newoff <- if (isTRUE(s$offset)) nd$off else rep(0, nrow(nd))
  theta <- exp(c((newX - rep(fit$means, each = nrow(newX))) %*% beta) + newoff - offset_mean)
  out <- list(
    name = s$name, ties = s$ties, entry = isTRUE(s$entry), strata = isTRUE(s$strata),
    weights = isTRUE(s$weights), offset = isTRUE(s$offset),
    columns = cn, x = X, coefficients = unname(beta), var = fit$var,
    means = unname(fit$means), start = d$start, stop = d$stop, event = d$event,
    stratum = d$stratum, weight = d$w, offset_values = d$off,
    new_x = newX, new_stratum = nd$stratum, new_offset = nd$off, new_theta = theta,
    start_time = s$st
  )
  scen <- list()
  for (stype in 1:2) {
    for (st in list(NULL, s$st)) {
      for (cens in c(TRUE, FALSE)) {
        curves <- list()
        for (i in seq_len(nrow(nd))) {
          a <- list(fit, newdata = nd[i, , drop = FALSE], stype = stype, censor = cens)
          if (!is.null(st)) a$start.time <- st
          curves[[i]] <- one_curve(do.call(survfit, a))
        }
        scen[[length(scen) + 1]] <- list(stype = stype, start_time = st,
                                        censor = cens, curves = curves)
      }
    }
  }
  out$scenarios <- scen
  if (isTRUE(s$paths)) {
    pd <- if (s$d == "lung") lung_path else sim_path
    gap <- if (s$d == "lung") lung_gap else sim_gap
    pnew <- model.matrix(as.formula(paste("~", s$terms)), pd)[, -1, drop = FALSE]
    out$path <- list(id = pd$id, start = pd$start, stop = pd$stop, x = pnew,
                     stratum = pd$stratum, offset = pd$off,
                     gap_start = gap$start, gap_length = gap$length)
    ps <- list()
    for (stype in 1:2) {
      for (st in list(NULL, s$st)) {
        for (cens in c(TRUE, FALSE)) {
          a <- list(fit, newdata = pd, id = pd$id, stype = stype, censor = cens)
          a$id <- NULL
          a$id <- quote(id)
          if (!is.null(st)) a$start.time <- st
          sf <- do.call(survfit, a)
          ci <- limits_of(sf)
          len <- sf$strata
          ends <- cumsum(len)
          begins <- ends - len + 1
          per <- list()
          for (k in seq_along(len)) {
            ix <- begins[k]:ends[k]
            per[[k]] <- list(id = as.integer(names(len)[k]), time = sf$time[ix],
                             n_risk = sf$n.risk[ix], n_event = sf$n.event[ix],
                             n_censor = sf$n.censor[ix], surv = sf$surv[ix],
                             cumhaz = sf$cumhaz[ix], std_err = sf$std.err[ix],
                             limits = list(log = list(lower = ci$log$lower[ix], upper = ci$log$upper[ix]),
                                           `log-log` = list(lower = ci[["log-log"]]$lower[ix], upper = ci[["log-log"]]$upper[ix])))
          }
          ps[[length(ps) + 1]] <- list(stype = stype, start_time = st, censor = cens, ids = per)
        }
      }
    }
    out$path_scenarios <- ps
  }
  out
}

res <- list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  fits = lapply(specs, run)
)
cat(toJSON(res, digits = I(17), na = "null", auto_unbox = TRUE, null = "null"))
