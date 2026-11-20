# Cox fits by survival::coxph with offsets, starting values and iteration
# control, and the tests of summary.coxph, on the data of cox_matrix.R (same
# seed, same columns): every fit of cox_matrix.R again with its tests, then
# fits with offset(), init =, iter.max (also 0, which coxph uses for score
# tests) and eps.
# For each fit: the coefficients and standard errors (NA, written null, for an
# aliased column), the log partial likelihood at init and at the fit, the
# iteration count and whether coxph ran out of iterations, the rank, the score
# (sctest), likelihood-ratio and Wald tests as coxph stores them (the Wald test
# is the robust one when a cluster is given, and wald_model is the model-based
# one then), the robust score test (rscore) with a cluster, coxph's warnings,
# and the condition numbers, on the scaled covariates, of the information
# matrix at the fit, at init, of the robust covariance matrix and of the cluster
# score cross-products at init, which bound the error of the quantities that
# depend on them, and the condition numbers at the points of the iteration
# (kappa_path). When coxph.fit runs out of iterations its variance is wrong,
# and the variance fields hold that of a fit started at the same coefficients
# (variance_refit). coxph runs coxph.fit for data without entry times and
# agreg.fit for data with them; the two stop differently when they halve or run
# out of iterations, and coxph.fit reports iter.max + 1 iterations when it runs
# out.
# The versions of R, survival and jsonlite are in the file. Numbers are written
# with 17 significant digits.
# Run: Rscript cox_control.R > cox_control.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261119)
n <- 150
x1 <- round(rnorm(n), 3)
x2 <- round(runif(n, 0, 2), 3)
x3 <- rbinom(n, 1, 0.45)
rate <- 0.08 * exp(0.4 * x1 - 0.5 * x3 + 0.3 * x2)
stop <- pmax(ceiling(rexp(n, rate)), 2)
start <- pmin(sample(0:5, n, replace = TRUE), stop - 1)
event <- rbinom(n, 1, 0.75)
stratum <- sample(0:2, n, replace = TRUE)
weight <- round(runif(n, 0.5, 3), 2)
id <- sample(rep(1:50, length.out = n))
sep <- as.numeric(seq_len(n) %in% sample(which(event == 0), 8))
columns <- cbind(
  x1 = x1, x2 = x2, x3 = x3,
  sum12 = x1 + x2, zero = 0, constant = 1,
  near = x1 + x2 + 1e-4 * rnorm(n), sep = sep
)
# The offset is drawn after the columns, so they are those of cox_matrix.R.
offset <- round(rnorm(n, 0.5, 0.6), 3)
d <- data.frame(start, stop, event, stratum, weight, id, offset)

C3 <- c("x1", "x2", "x3")
# name, columns, ties, entry, strata, weights, cluster, offset, init, iter.max, eps
spec <- function(name, cols, ties, entry = FALSE, strata = FALSE, weights = FALSE,
                 cluster = FALSE, offset = FALSE, init = NULL, iter.max = NULL,
                 eps = NULL) {
  list(name = name, cols = cols, ties = ties, entry = entry, strata = strata,
       weights = weights, cluster = cluster, offset = offset, init = init,
       iter.max = iter.max, eps = eps)
}
specs <- list(
  spec("breslow_plain", C3, "breslow"),
  spec("efron_plain", C3, "efron"),
  spec("breslow_entry", C3, "breslow", entry = TRUE),
  spec("efron_entry", C3, "efron", entry = TRUE),
  spec("breslow_strata", C3, "breslow", strata = TRUE),
  spec("efron_strata", C3, "efron", strata = TRUE),
  spec("breslow_weights", C3, "breslow", weights = TRUE),
  spec("efron_weights", C3, "efron", weights = TRUE),
  spec("breslow_cluster", C3, "breslow", cluster = TRUE),
  spec("efron_cluster", C3, "efron", cluster = TRUE),
  spec("breslow_all", C3, "breslow", TRUE, TRUE, TRUE, TRUE),
  spec("efron_all", C3, "efron", TRUE, TRUE, TRUE, TRUE),
  spec("alias_sum", c("x1", "x2", "x3", "sum12"), "efron", TRUE, FALSE, TRUE, TRUE),
  spec("alias_zero_constant", c("x1", "zero", "x3", "constant"), "breslow", strata = TRUE),
  spec("near_alias", c("x1", "x2", "near"), "efron", cluster = TRUE),
  spec("monotone_efron", c("x1", "x3", "sep"), "efron"),
  spec("monotone_breslow_entry", c("x1", "x3", "sep"), "breslow", entry = TRUE, strata = TRUE),
  # offsets
  spec("offset_efron", C3, "efron", offset = TRUE),
  spec("offset_breslow_all", C3, "breslow", TRUE, TRUE, TRUE, TRUE, offset = TRUE),
  spec("offset_efron_cluster", C3, "efron", cluster = TRUE, offset = TRUE),
  spec("offset_alias", c("x1", "x2", "x3", "sum12"), "efron", strata = TRUE, offset = TRUE),
  # starting values
  spec("init_near", C3, "efron", TRUE, TRUE, TRUE, TRUE, init = c(0.35, 0.2, -0.45)),
  spec("init_far", C3, "efron", init = c(3, -3, 3)),
  spec("init_far_breslow", C3, "breslow", strata = TRUE, init = c(-4, 4, -4)),
  spec("init_alias", c("x1", "x2", "x3", "sum12"), "efron", TRUE, FALSE, TRUE, TRUE,
       init = c(0.2, 0.1, -0.3, 0.4)),
  spec("init_offset", C3, "breslow", entry = TRUE, cluster = TRUE, offset = TRUE,
       init = c(0.5, -0.5, 1)),
  # iteration control
  spec("iter0_default", C3, "efron", iter.max = 0),
  spec("iter0_init", C3, "efron", TRUE, TRUE, TRUE, TRUE, init = c(0.3, 0.1, -0.4), iter.max = 0),
  spec("iter0_alias", c("x1", "x2", "x3", "sum12"), "breslow", TRUE, FALSE, TRUE, TRUE,
       init = c(0.2, 0.1, -0.3, 0.4), iter.max = 0),
  spec("iter0_alias_default", c("x1", "zero", "x3", "constant"), "efron", iter.max = 0),
  spec("iter0_offset_cluster", C3, "efron", cluster = TRUE, offset = TRUE,
       init = c(0.1, 0.2, 0.3), iter.max = 0),
  spec("iter0_monotone", c("x1", "x3", "sep"), "efron", iter.max = 0),
  spec("iter1", C3, "breslow", iter.max = 1),
  spec("iter2", C3, "efron", TRUE, FALSE, FALSE, FALSE, iter.max = 2),
  spec("iter3_cluster", C3, "efron", cluster = TRUE, iter.max = 3),
  spec("iter3_far", C3, "breslow", strata = TRUE, init = c(-4, 4, -4), iter.max = 3),
  spec("eps_loose", C3, "efron", strata = TRUE, eps = 1e-4),
  spec("eps_tight", C3, "efron", TRUE, TRUE, TRUE, FALSE, eps = 1e-11),
  spec("eps_loose_init", C3, "breslow", init = c(1, -1, 1), eps = 1e-3)
)

run <- function(s, iter.max = s$iter.max, init = s$init) {
  x <- columns[, s$cols, drop = FALSE]
  d$X <- x
  lhs <- if (s$entry) "Surv(start, stop, event)" else "Surv(stop, event)"
  rhs <- "X"
  if (s$strata) rhs <- paste(rhs, "+ strata(stratum)")
  if (s$offset) rhs <- paste(rhs, "+ offset(offset)")
  ctl <- list(timefix = FALSE)
  if (!is.null(iter.max)) ctl$iter.max <- iter.max
  if (!is.null(s$eps)) ctl$eps <- s$eps
  args <- list(formula = as.formula(paste(lhs, "~", rhs)), data = d, ties = s$ties,
               control = do.call(coxph.control, ctl), robust = s$cluster)
  if (s$weights) args$weights <- d$weight
  if (s$cluster) args$cluster <- d$id
  if (!is.null(init)) args$init <- init
  warns <- character()
  fit <- withCallingHandlers(do.call(coxph, args), warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(fit = fit, warns = warns, x = x)
}

fits <- list()
for (s in specs) {
  r <- run(s)
  fit <- r$fit
  k <- length(s$cols)
  # coxph.fit (data without entry times) inverts an information matrix that it
  # has just recomputed and not factorised when it runs out of iterations, so
  # the variance of such a fit is wrong. The variance at the same coefficients
  # is that of the fit started there and not iterated, and the Wald tests are
  # recomputed from it.
  bad_var <- !s$entry && any(grepl("Ran out of iterations", r$warns))
  vfit <- fit
  if (bad_var) {
    at <- coef(fit)
    at[is.na(at)] <- 0
    vfit <- run(s, iter.max = 0, init = at)$fit
  }
  vn <- as.matrix(if (s$cluster) vfit$naive.var else vfit$var)
  iter0 <- !is.null(s$iter.max) && s$iter.max == 0
  # With iter.max = 0 coxph leaves the starting values in place, aliased
  # columns included, and marks an aliased column by a zero variance.
  ok <- if (iter0) diag(vn) != 0 else !is.na(coef(fit))
  full <- function(values) {
    out <- rep(NA_real_, k)
    out[ok] <- values
    out
  }
  keep <- function(v) if (nrow(v) == k) v[ok, ok, drop = FALSE] else v
  vn <- keep(vn)
  w <- if (s$weights) d$weight else rep(1, n)
  scale_of <- function(xs) {
    centred <- sweep(xs, 2, colSums(w * xs) / sum(w))
    sc <- colSums(w * abs(centred)) / sum(w)
    sc[sc == 0] <- 1
    sc
  }
  xs <- r$x[, ok, drop = FALSE]
  sc <- scale_of(xs)
  kappa_of <- function(v) kappa(solve(diag(sc, length(sc)) %*% v %*% diag(sc, length(sc))), exact = TRUE)
  # The fit at its starting values, for the information and the cluster
  # scores there.
  init <- if (is.null(s$init)) rep(0, k) else s$init
  r0 <- run(s, iter.max = 0, init = init)
  v0 <- as.matrix(if (s$cluster) r0$fit$naive.var else r0$fit$var)
  ok0 <- diag(v0) != 0
  v0 <- v0[ok0, ok0, drop = FALSE]
  sc0 <- scale_of(r$x[, ok0, drop = FALSE])
  kappa0 <- kappa(solve(diag(sc0, length(sc0)) %*% v0 %*% diag(sc0, length(sc0))), exact = TRUE)
  kappa_meat <- NA_real_
  rscore_at_init <- NA_real_
  if (s$cluster) {
    u0 <- as.matrix(residuals(r0$fit, type = "score", collapse = d$id, weighted = TRUE))[, ok0, drop = FALSE]
    # coxph computes rscore from score residuals at a linear predictor of
    # init without the offset (it sets the linear predictors to zero, or to
    # X %*% init). The robust score test of a model with an offset is at init
    # with the offset in, which is what the residuals of the fit at init give.
    rscore_at_init <- coxph.wtest(crossprod(u0), colSums(u0), coxph.control()$toler.chol)$test
    # The cluster score sums are on the scale of the columns given, so the
    # condition number is that of the meat on the scaled covariates:
    # a column scaled by 1 / sc0 has a score scaled by 1 / sc0.
    kappa_meat <- kappa(diag(1 / sc0, length(sc0)) %*% crossprod(u0) %*% diag(1 / sc0, length(sc0)), exact = TRUE)
  }
  dd <- (coef(fit) - init)[ok]
  wald_model <- sum(dd * solve(vn, dd))
  wald_main <- if (bad_var) {
    if (s$cluster) sum(dd * solve(keep(as.matrix(vfit$var)), dd)) else wald_model
  } else fit$wald.test
  # The condition number of the information matrix at the coefficients after
  # each trial point up to the last: the points at which the iteration solves.
  kappa_path <- numeric()
  last_trial <- if (is.null(s$iter.max)) fit$iter else min(fit$iter, s$iter.max)
  for (j in seq_len(max(last_trial, 1)) - 1) {
    at <- if (j == 0) init else coef(run(s, iter.max = j)$fit)
    at[is.na(at)] <- 0
    vj <- as.matrix(run(s, iter.max = 0, init = at)$fit[[if (s$cluster) "naive.var" else "var"]])
    okj <- diag(vj) != 0
    scj <- scale_of(r$x[, okj, drop = FALSE])
    vj <- vj[okj, okj, drop = FALSE]
    kappa_path <- c(kappa_path, kappa(solve(diag(scj, length(scj)) %*% vj %*% diag(scj, length(scj))), exact = TRUE))
  }
  kappa_robust <- NA_real_
  if (s$cluster) kappa_robust <- kappa_of(keep(as.matrix(vfit$var)))
  # kappa of the information at the fit uses the model-based variance, scaled
  # in the same way (vn over the columns that are not aliased).
  monotone <- integer()
  for (m in r$warns) {
    if (grepl("may be infinite", m)) {
      monotone <- c(monotone, as.integer(strsplit(trimws(sub(";.*", "", sub(".*variable", "", m))), "[ ,]+")[[1]]))
    }
  }
  embed <- function(v) {
    out <- matrix(NA_real_, k, k)
    out[ok, ok] <- v
    unname(out)
  }
  fits[[length(fits) + 1]] <- list(
    name = s$name, columns = s$cols, ties = s$ties,
    use_start = s$entry, use_strata = s$strata, use_weights = s$weights,
    use_cluster = s$cluster,
    offset = if (s$offset) offset else numeric(),
    init = if (is.null(s$init)) numeric() else s$init,
    iter_max = if (is.null(s$iter.max)) 20L else as.integer(s$iter.max),
    eps = if (is.null(s$eps)) 1e-9 else s$eps,
    coefficients = full(unname(coef(fit))[ok]),
    se = full(sqrt(diag(vn))),
    robust_se = if (s$cluster) full(sqrt(diag(keep(as.matrix(vfit$var))))) else numeric(),
    vcov = embed(vn),
    loglik_init = fit$loglik[1], loglik = fit$loglik[2],
    iterations = fit$iter, rank = sum(ok),
    ran_out = any(grepl("Ran out of iterations", r$warns)),
    warnings = r$warns, monotone_columns = as.list(monotone),
    score = fit$score, wald = wald_main, variance_refit = bad_var, wald_model = wald_model,
    robust_score = if (is.null(fit$rscore)) NA_real_ else fit$rscore,
    robust_score_at_init = rscore_at_init,
    kappa_fit = kappa_of(vn), kappa_init = kappa0, kappa_path = as.list(kappa_path), kappa_meat = kappa_meat,
    kappa_robust = kappa_robust
  )
}
cat(toJSON(list(
  versions = list(
    r = R.version.string,
    survival = as.character(packageVersion("survival")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  columns = unname(columns), column_names = colnames(columns),
  start = start, stop = stop, event = event, stratum = stratum,
  weight = weight, id = id, fits = fits
), digits = I(17), auto_unbox = TRUE, na = "null"))
