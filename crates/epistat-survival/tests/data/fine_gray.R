# The Fine-Gray model by survival::finegray + coxph and by cmprsk::crr.
#
# Four data sets, each with cause 0 for censoring and two or three causes:
#   tied        integer times, so failures tie with each other, with
#               censorings and with competing events; three causes
#   continuous  no tied times
#   tied_apart  failures at integer times, censorings at half-integers, so
#               times tie within a kind and never across kinds
#   weighted    integer times, integer case weights, two causes
# For every data set and cause: the rows survival::finegray writes (original
# row, start, stop, status, weight) and coxph fits on them with Breslow and
# Efron ties, with no cluster, with the subject as cluster and with groups of
# four subjects as cluster. coxph runs with its default convergence rule (eps
# 1e-9, no polishing), so its Newton path is that of the Rust fit, and the
# iteration count is recorded.
#
# Several covariates: coxph on the same rows with a binary covariate, a
# continuous one and a factor of three levels (fac), as bin + x + factor(fac),
# whose columns are bin, x, fac1 and fac2. coxph runs with its default
# convergence rule (eps 1e-9, no polishing), so the Newton path is that of the
# Rust fit, and the iteration count is recorded. The reference holds the
# coefficients, the model-based covariance, the robust covariance with the
# subject or a group of four as cluster, the log partial likelihoods, the Wald
# test (model-based, and robust with a cluster), the score test, and the
# condition numbers of the information at the fit, at zero and of the robust
# covariance, on covariates scaled by their weighted mean absolute deviation.
#
# Case weights. finegray(weights =) estimates the censoring distribution
# without them. The reference for a weighted analysis is therefore finegray on
# data with each subject repeated weight times, keeping one copy of each
# subject's rows with the weight put back on the row. That is the weighted
# Kaplan-Meier estimate of the censoring distribution. The rows of
# finegray(weights =) are kept as well, as `rows_weights_argument`.
#
# cmprsk::crr (gtol 1e-8, the tightest at which its line search converges on
# every set) on the data sets without weights: coefficient, its standard
# error and the information, with the convergence tolerance tightened.
# `replica` is the Breslow estimator written out with the censoring
# distribution computed as crr computes it; the script stops unless it agrees
# with crr, which shows what crr estimates.
#
# Run: Rscript fine_gray.R > fine_gray.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(cmprsk)
})
set.seed(20261120)
n <- 120

make <- function(name, time, event, weight = rep(1L, length(time)), kmax) {
  n <- length(time)
  list(
    name = name, time = time, event = event, weight = weight,
    x = round(rnorm(n) + 0.3 * (event == 1), 2),
    bin = rbinom(n, 1, 0.45),
    fac = (seq_len(n) + event) %% 3L,
    id = seq_len(n),
    group = rep(seq_len(ceiling(n / 4)), each = 4)[seq_len(n)],
    causes = 1:kmax
  )
}
sets <- list(
  make("tied", sample(1:12, n, TRUE), sample(0:3, n, TRUE, prob = c(.3, .4, .2, .1)), kmax = 3),
  make("continuous", rexp(n, 0.2), sample(0:2, n, TRUE, prob = c(.3, .45, .25)), kmax = 2)
)
ta <- sample(1:12, n, TRUE)
ea <- sample(0:2, n, TRUE, prob = c(.3, .45, .25))
sets[[3]] <- make("tied_apart", ifelse(ea == 0, ta + 0.5, ta), ea, kmax = 2)
nw <- 80
sets[[4]] <- make("weighted", sample(1:12, nw, TRUE), sample(0:2, nw, TRUE, prob = c(.3, .45, .25)),
                  weight = sample(1:3, nw, TRUE), kmax = 2)

multi_control <- coxph.control(timefix = FALSE, iter.max = 50)

fg_rows <- function(s, cause, use_weight_argument = FALSE) {
  n <- length(s$time)
  d <- data.frame(time = s$time,
                  st = factor(s$event, 0:max(s$causes), c("cens", paste0("c", s$causes))),
                  orig = seq_len(n), w = s$weight)
  if (use_weight_argument) {
    fg <- finegray(Surv(time, st) ~ orig, data = d, etype = paste0("c", cause),
                   weights = w, timefix = FALSE)
    return(data.frame(orig = fg$orig, start = fg$fgstart, stop = fg$fgstop,
                      status = fg$fgstatus, weight = fg$fgwt))
  }
  idx <- rep(seq_len(n), s$weight)
  copy <- ave(seq_along(idx), idx, FUN = seq_along)
  dr <- d[idx, ]
  dr$copy <- copy
  fg <- finegray(Surv(time, st) ~ orig + copy, data = dr, etype = paste0("c", cause),
                 timefix = FALSE)
  fg <- fg[fg$copy == 1, ]
  data.frame(orig = fg$orig, start = fg$fgstart, stop = fg$fgstop,
             status = fg$fgstatus, weight = fg$fgwt * s$weight[fg$orig])
}

fit_one <- function(s, rows, cov, ties, cl) {
  rows$X <- s[[cov]][rows$orig]
  args <- list(formula = Surv(start, stop, status) ~ X, data = rows, weights = rows$weight,
               ties = ties, control = multi_control, robust = cl != "none")
  if (cl != "none") args$cluster <- s[[cl]][rows$orig]
  fit <- do.call(coxph, args)
  stopifnot(fit$iter < 50)
  list(covariate = cov, ties = ties, cluster = cl,
       coefficient = unname(coef(fit)),
       se = as.numeric(sqrt(if (cl != "none") fit$naive.var else fit$var)),
       robust_se = if (cl != "none") as.numeric(sqrt(fit$var)) else NA_real_,
       iterations = fit$iter,
       n_events = sum(rows$weight[rows$status == 1]))
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
       vcov = unname(vn), info = unname(solve(vn)), vcov_robust = if (cl != "none") unname(as.matrix(fit$var)) else numeric(),
       loglik = fit$loglik[2], loglik_null = fit$loglik[1], iterations = fit$iter,
       wald_model = sum(b * solve(vn, b)), wald = fit$wald.test, score = fit$score,
       kappa = kappa_scaled(solve(vn), xs, w),
       kappa_init = kappa_scaled(solve(naive(zero)), xs, w),
       kappa_robust = if (cl != "none") kappa_scaled(as.matrix(fit$var), xs, w) else NA_real_,
       n_events = sum(rows$weight[rows$status == 1]))
}

# crr's censoring distribution, by its own code.
crr_g <- function(time, event) {
  o <- order(time)
  ftime <- time[o]
  cenind <- as.numeric(event[o] == 0)
  u <- survfit(Surv(ftime, cenind) ~ 1)
  u <- approx(c(min(0, u$time) - 10 * .Machine$double.eps,
                c(u$time, max(u$time) * (1 + 10 * .Machine$double.eps))),
              c(1, u$surv, 0), xout = ftime * (1 - 100 * .Machine$double.eps),
              method = "constant", f = 0, rule = 2)$y
  out <- numeric(length(time))
  out[o] <- u
  out
}

# Breslow estimator of the subdistribution hazard with weight G(t-) / G(x_i-)
# for a competing event at x_i < t, by Newton-Raphson on the log partial
# likelihood.
replica <- function(time, event, x, cause) {
  g <- crr_g(time, event)
  uft <- sort(unique(time[event == cause]))
  gt <- sapply(uft, function(t) { i <- which(time == t)[1]; g[i] })
  comp <- event != 0 & event != cause
  beta <- 0
  for (it in 1:50) {
    u <- 0; info <- 0
    for (j in seq_along(uft)) {
      t <- uft[j]
      w <- ifelse(time >= t, 1, ifelse(comp & time < t, gt[j] / g, 0))
      w[!is.finite(w)] <- 0
      th <- w * exp(beta * x)
      s0 <- sum(th); s1 <- sum(th * x); s2 <- sum(th * x^2)
      f <- event == cause & time == t
      d <- sum(f)
      u <- u + sum(x[f]) - d * s1 / s0
      info <- info + d * (s2 / s0 - (s1 / s0)^2)
    }
    step <- u / info
    beta <- beta + step
    if (abs(step) < 1e-14) break
  }
  beta
}

datasets <- list()
for (s in sets) {
  causes <- list()
  for (cause in s$causes) {
    rows <- fg_rows(s, cause)
    fits <- list()
    for (cov in c("x", "bin")) for (ties in c("breslow", "efron")) {
      for (cl in c("none", "id", "group")) {
        fits[[length(fits) + 1]] <- fit_one(s, rows, cov, ties, cl)
      }
    }
    multi <- list()
    for (ties in c("breslow", "efron")) for (cl in c("none", "id", "group")) {
      multi[[length(multi) + 1]] <- fit_multi(s, rows, ties, cl)
    }
    entry <- list(cause = cause, rows = rows, fits = fits, multi = multi)
    if (s$name == "weighted") {
      entry$rows_weights_argument <- fg_rows(s, cause, TRUE)
    } else {
      crr_fits <- list()
      for (cov in c("x", "bin")) {
        cr <- crr(s$time, s$event, cbind(s[[cov]]), failcode = cause, cencode = 0,
                  gtol = 1e-8, maxiter = 100)
        rep_beta <- replica(s$time, s$event, s[[cov]], cause)
        # crr stops when its score is below gtol * max(|loglik|, 1) / max(|b|, 1),
        # so its coefficient is within that over the information of the root.
        bound <- 2 * 1e-8 * max(abs(cr$loglik), 1) / cr$inf[1, 1]
        stopifnot(cr$converged, abs(rep_beta - cr$coef) < bound)
        crr_fits[[length(crr_fits) + 1]] <- list(
          covariate = cov, coefficient = unname(cr$coef), se = sqrt(cr$var[1, 1]),
          information = cr$inf[1, 1], loglik = as.numeric(cr$loglik), replica = rep_beta
        )
      }
      entry$crr <- crr_fits
      # Several covariates: bin, x and the two indicators of fac.
      covs <- cbind(s$bin, s$x, s$fac == 1, s$fac == 2) + 0
      cr <- crr(s$time, s$event, covs, failcode = cause, cencode = 0,
                gtol = 1e-8, maxiter = 100)
      stopifnot(cr$converged)
      entry$crr_multi <- list(
        coefficients = unname(cr$coef), information = unname(cr$inf),
        loglik = as.numeric(cr$loglik)
      )
    }
    causes[[length(causes) + 1]] <- entry
  }
  datasets[[length(datasets) + 1]] <- c(
    s[c("name", "time", "event", "weight", "x", "bin", "fac", "id", "group")],
    list(results = causes)
  )
}

pkgs <- c("survival", "cmprsk", "jsonlite")
cat(toJSON(list(
  r_version = R.version.string,
  packages = setNames(lapply(pkgs, function(p) as.character(packageVersion(p))), pkgs),
  datasets = datasets
), digits = I(17), auto_unbox = TRUE, na = "null", dataframe = "columns"))
