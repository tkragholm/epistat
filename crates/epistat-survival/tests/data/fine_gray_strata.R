# The Fine-Gray model with a censoring distribution per stratum, by
# cmprsk::crr(cengroup =) and by coxph on rows built here.
#
# Two data sets, each with cause 0 for censoring, two causes and a stratum of
# three levels:
#   tied        integer times and integer case weights, so failures tie with
#               each other, with censorings and with competing events
#   continuous  no tied times, unit weights
#
# survival::finegray estimates the censoring distribution per stratum only
# for a model stratified by the same variable, and writes each stratum's rows
# up to its own last time and event, so it does not give the rows of a model
# with one baseline. The rows are therefore built here: an individual with
# the cause of interest or censored has the row (0, x]; one with a competing
# event at x has the row (0, c] at its case weight, with c the first censoring
# time of its stratum at or after x or the largest time, and the row
# (c_j, c_j+1] at weight w G(c_j) / G(x-) for each later censoring time c_j of
# its stratum, the last ending at the largest time. G is the weighted
# Kaplan-Meier estimate of the censoring distribution within the stratum, with
# a failure at a censoring time taken out of the risk set first, as finegray
# does, and 0 past the stratum's largest time, as crr has it. coxph fits bin + x + factor(fac) on the rows with Breslow and Efron
# ties, with no cluster and with the subject as cluster, with its default
# convergence rule, and the reference holds what fine_gray.R holds for such a
# fit.
#
# crr(cengroup = stratum) on the continuous set, where no failure ties a
# censoring and crr's censoring distribution is the one above: coefficients,
# information and log likelihood, for bin + x + the indicators of fac.
#
# Run: Rscript fine_gray_strata.R > fine_gray_strata.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(cmprsk)
})
set.seed(20261122)
n <- 150

make <- function(name, time, event, weight) {
  n <- length(time)
  list(
    name = name, time = time, event = event, weight = weight,
    x = round(rnorm(n) + 0.3 * (event == 1), 2),
    bin = rbinom(n, 1, 0.45),
    fac = (seq_len(n) + event) %% 3L,
    stratum = sample(0:2, n, TRUE),
    id = seq_len(n),
    causes = 1:2
  )
}
sets <- list(
  make("tied", sample(1:12, n, TRUE), sample(0:2, n, TRUE, prob = c(.35, .4, .25)),
       sample(1:3, n, TRUE)),
  make("continuous", rexp(n, 0.2), sample(0:2, n, TRUE, prob = c(.35, .4, .25)), rep(1L, n))
)

multi_control <- coxph.control(timefix = FALSE, iter.max = 50)

# G within one stratum: a step function as (times, values), 1 before the
# first censoring time. Past the stratum's largest time G is 0, as crr has
# it, so a competing event of the stratum leaves the risk set there.
censoring_curve <- function(time, event, w, horizon) {
  ctimes <- sort(unique(time[event == 0]))
  g <- 1
  values <- numeric(length(ctimes))
  for (j in seq_along(ctimes)) {
    c <- ctimes[j]
    d <- sum(w[time == c & event == 0])
    at_risk <- sum(w[time > c]) + d
    g <- g * (1 - d / at_risk)
    values[j] <- g
  }
  last <- max(time)
  if (last < horizon && (length(ctimes) == 0 || max(ctimes) < last)) {
    ctimes <- c(ctimes, last)
    values <- c(values, 0)
  }
  list(time = ctimes, value = values)
}
before <- function(curve, t) {
  k <- sum(curve$time < t)
  if (k == 0) 1 else curve$value[k]
}
at <- function(curve, t) {
  k <- sum(curve$time <= t)
  if (k == 0) 1 else curve$value[k]
}

strata_rows <- function(s, cause) {
  horizon <- max(s$time)
  curves <- lapply(0:2, function(g) {
    keep <- s$stratum == g
    censoring_curve(s$time[keep], s$event[keep], s$weight[keep], horizon)
  })
  out <- list()
  for (i in seq_along(s$time)) {
    t <- s$time[i]
    w <- s$weight[i]
    e <- s$event[i]
    if (e == 0 || e == cause) {
      out[[length(out) + 1]] <- c(i, 0, t, as.numeric(e == cause), w)
      next
    }
    curve <- curves[[s$stratum[i] + 1]]
    later <- curve$time[curve$time >= t]
    gb <- before(curve, t)
    bounds <- c(0, later)
    if (length(later) == 0 || max(later) < horizon) bounds <- c(bounds, horizon)
    for (j in seq_len(length(bounds) - 1)) {
      start <- bounds[j]
      weight <- if (start < t) w else if (gb > 0) w * at(curve, start) / gb else 0
      if (weight > 0) out[[length(out) + 1]] <- c(i, start, bounds[j + 1], 0, weight)
    }
  }
  m <- do.call(rbind, out)
  data.frame(orig = m[, 1], start = m[, 2], stop = m[, 3], status = m[, 4], weight = m[, 5])
}

kappa_scaled <- function(v, xs, w) {
  centred <- sweep(xs, 2, colSums(w * xs) / sum(w))
  sc <- colSums(w * abs(centred)) / sum(w)
  sc[sc == 0] <- 1
  d <- diag(sc, length(sc))
  kappa(d %*% v %*% d, exact = TRUE)
}

fit_multi <- function(s, rows, ties, cl) {
  rows$bin <- s$bin[rows$orig]
  rows$x <- s$x[rows$orig]
  rows$fac <- factor(s$fac[rows$orig], 0:2)
  args <- list(formula = Surv(start, stop, status) ~ bin + x + fac, data = rows,
               weights = rows$weight, ties = ties, control = multi_control,
               robust = cl != "none")
  if (cl != "none") args$cluster <- s[[cl]][rows$orig]
  fit <- do.call(coxph, args)
  stopifnot(fit$iter < 50)
  args0 <- args
  args0$control <- coxph.control(timefix = FALSE, iter.max = 0)
  zero <- do.call(coxph, c(args0, list(init = rep(0, 4))))
  naive <- function(f) as.matrix(if (cl != "none") f$naive.var else f$var)
  vn <- naive(fit)
  b <- coef(fit)
  xs <- model.matrix(~ bin + x + fac, rows)[, -1]
  w <- rows$weight
  list(ties = ties, cluster = cl, coefficients = unname(b),
       vcov = unname(vn), info = unname(solve(vn)),
       vcov_robust = if (cl != "none") unname(as.matrix(fit$var)) else numeric(),
       loglik = fit$loglik[2], loglik_null = fit$loglik[1], iterations = fit$iter,
       wald_model = sum(b * solve(vn, b)), wald = fit$wald.test, score = fit$score,
       kappa = kappa_scaled(solve(vn), xs, w),
       kappa_init = kappa_scaled(solve(naive(zero)), xs, w),
       kappa_robust = if (cl != "none") kappa_scaled(as.matrix(fit$var), xs, w) else NA_real_,
       n_events = sum(rows$weight[rows$status == 1]))
}

datasets <- list()
for (s in sets) {
  causes <- list()
  for (cause in s$causes) {
    rows <- strata_rows(s, cause)
    multi <- list()
    for (ties in c("breslow", "efron")) for (cl in c("none", "id")) {
      multi[[length(multi) + 1]] <- fit_multi(s, rows, ties, cl)
    }
    entry <- list(cause = cause, n_rows = nrow(rows), multi = multi)
    if (s$name == "continuous") {
      covs <- cbind(s$bin, s$x, s$fac == 1, s$fac == 2) + 0
      cr <- crr(s$time, s$event, covs, cengroup = s$stratum, failcode = cause,
                cencode = 0, gtol = 1e-8, maxiter = 100)
      stopifnot(cr$converged)
      entry$crr_multi <- list(
        coefficients = unname(cr$coef), information = unname(cr$inf),
        loglik = as.numeric(cr$loglik)
      )
    }
    causes[[length(causes) + 1]] <- entry
  }
  datasets[[length(datasets) + 1]] <- c(
    s[c("name", "time", "event", "weight", "x", "bin", "fac", "stratum", "id")],
    list(results = causes)
  )
}

pkgs <- c("survival", "cmprsk", "jsonlite")
cat(toJSON(list(
  r_version = R.version.string,
  packages = setNames(lapply(pkgs, function(p) as.character(packageVersion(p))), pkgs),
  datasets = datasets
), digits = I(17), auto_unbox = TRUE, na = "null", dataframe = "columns"))
