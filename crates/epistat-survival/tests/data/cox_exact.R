# Cox fits with exact ties by survival::coxph(ties = "exact") and conditional
# logistic regression by survival::clogit.
#
# Four data sets: "tied", 120 rows with integer times, so that events tie in
# sets of up to eight, with strata and an offset; "small", 45 rows of the same
# kind with delayed entry and risk sets of at most 20, because coxph enumerates
# the subsets of the risk set for data with entry times; "infert", the
# matched case-control data of the survival package, with one case and one
# to two controls in each set; and "matched", a simulated matched
# case-control set of 1:1 sets, 1:3 sets, 2:4 sets with two cases and one set
# without a case.
# For each fit: the coefficients and standard errors (NA, written null, for an
# aliased column), the covariance matrix, the log partial likelihood at init and
# at the fit, the iteration count, whether the fit ran out of iterations, the
# rank, the score (sctest), likelihood-ratio and Wald tests, n and nevent as the
# fit reports them, the warnings, and the condition numbers, on the scaled
# covariates, of the information matrix at the fit and at init, and at the points
# of the iteration (kappa_path), which bound the error of the quantities that
# depend on solves with it. d_max is the largest number of tied events at a time
# and n_risk_max the largest risk set, which bound the error of the recursion
# that coxexact.fit uses for data without entry times, and subsets_max the
# largest number of subsets of a risk set that agexact.fit enumerates and sums
# one by one for data with entry times.
# coxph is called with timefix = FALSE, since the times are integers. When
# coxexact.fit runs out of iterations its variance is wrong, and the variance
# fields hold that of a fit started at the same coefficients (variance_refit).
# The fits with entry times start at most one unit from zero. agexact.fit does not
# scale the covariates, and at a start of 3 its score test (sctest) differs from
# the value of an independent recursion, which agrees with epistat to 2e-15,
# by 5e-10, relative, more than the error that its condition number allows.
# The versions of R, survival and jsonlite are in the file. Numbers are written
# with 17 significant digits.
# Run: Rscript cox_exact.R > cox_exact.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)

n <- 120
x1 <- round(rnorm(n), 3)
x2 <- round(runif(n, 0, 2), 3)
x3 <- rbinom(n, 1, 0.45)
rate <- 0.05 * exp(0.4 * x1 - 0.5 * x3 + 0.3 * x2)
stop <- pmax(ceiling(rexp(n, rate)), 2)
start <- pmin(sample(0:6, n, replace = TRUE), stop - 1)
event <- rbinom(n, 1, 0.8)
stratum <- sample(0:2, n, replace = TRUE)
sep <- as.numeric(seq_len(n) %in% sample(which(event == 0), 6))
tied <- list(
  columns = cbind(x1 = x1, x2 = x2, x3 = x3, sum12 = x1 + x2, constant = 1, sep = sep),
  start = start, stop = stop, event = event, stratum = stratum,
  offset = round(rnorm(n, 0.4, 0.5), 3)
)

m <- 45
y1 <- round(rnorm(m), 3)
y2 <- round(runif(m, 0, 2), 3)
y3 <- rbinom(m, 1, 0.45)
srate <- 0.15 * exp(0.4 * y1 - 0.5 * y3 + 0.3 * y2)
sstop <- pmax(ceiling(rexp(m, srate)), 2)
sevent <- rbinom(m, 1, 0.8)
small <- list(
  columns = cbind(x1 = y1, x2 = y2, x3 = y3, sum12 = y1 + y2, constant = 1,
                  sep = as.numeric(seq_len(m) %in% sample(which(sevent == 0), 3))),
  start = pmin(sample(0:3, m, replace = TRUE), sstop - 1), stop = sstop,
  event = sevent, stratum = sample(0:2, m, replace = TRUE),
  offset = round(rnorm(m, 0.4, 0.5), 3)
)

data(infert)
infert <- infert[order(infert$stratum), ]
infert_cols <- cbind(spontaneous = infert$spontaneous, induced = infert$induced,
                     age = infert$age, parity = infert$parity)
infert_data <- list(
  columns = infert_cols, start = rep(0, nrow(infert)), stop = rep(1, nrow(infert)),
  event = infert$case, stratum = infert$stratum, offset = rep(0, nrow(infert))
)

# Matched sets of 1:1, 1:3 and 2:4 and one set with no case.
sets <- c(rep(2, 14), rep(4, 14), rep(6, 12), 3)
nm <- sum(sets)
set_id <- rep(seq_along(sets), sets)
m1 <- round(rnorm(nm), 3)
m2 <- rbinom(nm, 1, 0.4)
m3 <- round(runif(nm, 0, 3), 3)
case <- integer(nm)
for (s in seq_along(sets)) {
  rows <- which(set_id == s)
  d <- if (sets[s] == 6) 2 else if (sets[s] == 3) 0 else 1
  if (d > 0) {
    w <- exp(0.7 * m1[rows] - 0.5 * m2[rows] + 0.2 * m3[rows])
    case[sample(rows, d, prob = w)] <- 1
  }
}
matched <- list(
  columns = cbind(m1 = m1, m2 = m2, m3 = m3), start = rep(0, nm), stop = rep(1, nm),
  event = case, stratum = set_id, offset = round(rnorm(nm, 0, 0.3), 3)
)

datasets <- list(tied = tied, small = small, infert = infert_data, matched = matched)

# name, data, kind (cox or clogit), columns, method, entry, strata, offset,
# init, iter.max, eps
spec <- function(name, data, kind, cols, method, entry = FALSE, strata = FALSE,
                 offset = FALSE, init = NULL, iter.max = NULL, eps = NULL) {
  list(name = name, data = data, kind = kind, cols = cols, method = method,
       entry = entry, strata = strata, offset = offset, init = init,
       iter.max = iter.max, eps = eps)
}
C3 <- c("x1", "x2", "x3")
specs <- list(
  spec("exact_plain", "tied", "cox", C3, "exact"),
  spec("exact_strata", "tied", "cox", C3, "exact", strata = TRUE),
  spec("exact_entry", "small", "cox", C3, "exact", entry = TRUE),
  spec("exact_entry_strata", "small", "cox", C3, "exact", entry = TRUE, strata = TRUE),
  spec("exact_offset", "tied", "cox", C3, "exact", offset = TRUE),
  spec("exact_entry_strata_offset", "small", "cox", C3, "exact", TRUE, TRUE, TRUE),
  spec("exact_one_column", "tied", "cox", "x1", "exact", strata = TRUE),
  spec("exact_alias", "tied", "cox", c("x1", "x2", "x3", "sum12"), "exact", strata = TRUE),
  spec("exact_alias_constant", "small", "cox", c("x1", "constant", "x3"), "exact", entry = TRUE),
  spec("exact_monotone", "tied", "cox", c("x1", "x3", "sep"), "exact"),
  spec("exact_monotone_entry", "small", "cox", c("x1", "x3", "sep"), "exact", entry = TRUE),
  spec("exact_init_near", "tied", "cox", C3, "exact", strata = TRUE, init = c(0.3, 0.2, -0.4)),
  spec("exact_init_far", "tied", "cox", C3, "exact", init = c(3, -3, 3)),
  spec("exact_init_far_entry", "small", "cox", C3, "exact", entry = TRUE, init = c(-1, 1, -1)),
  spec("exact_iter0", "tied", "cox", C3, "exact", iter.max = 0),
  spec("exact_iter0_init", "small", "cox", C3, "exact", entry = TRUE, strata = TRUE,
       init = c(0.3, 0.1, -0.4), iter.max = 0),
  spec("exact_iter1", "tied", "cox", C3, "exact", iter.max = 1),
  spec("exact_iter1_entry", "small", "cox", C3, "exact", entry = TRUE, iter.max = 1),
  spec("exact_iter2", "tied", "cox", C3, "exact", iter.max = 2),
  spec("exact_iter2_entry", "small", "cox", C3, "exact", entry = TRUE, iter.max = 2),
  spec("exact_iter3_far", "tied", "cox", C3, "exact", strata = TRUE, init = c(-2, 2, -2),
       iter.max = 3),
  spec("exact_iter3_far_entry", "small", "cox", C3, "exact", entry = TRUE, init = c(-1, 1, -1),
       iter.max = 3),
  spec("exact_eps_loose", "tied", "cox", C3, "exact", strata = TRUE, eps = 1e-4),
  spec("exact_eps_tight", "tied", "cox", C3, "exact", eps = 1e-11),
  # The same data with the approximations, for the difference between methods.
  spec("breslow_tied", "tied", "cox", C3, "breslow", strata = TRUE),
  spec("efron_tied", "tied", "cox", C3, "efron", strata = TRUE),
  spec("infert_exact", "infert", "clogit", c("spontaneous", "induced"), "exact", strata = TRUE),
  spec("infert_approximate", "infert", "clogit", c("spontaneous", "induced"), "approximate",
       strata = TRUE),
  spec("infert_efron", "infert", "clogit", c("spontaneous", "induced"), "efron", strata = TRUE),
  spec("infert_breslow", "infert", "clogit", c("spontaneous", "induced"), "breslow", strata = TRUE),
  spec("infert_exact_four", "infert", "clogit", c("spontaneous", "induced", "age", "parity"),
       "exact", strata = TRUE),
  spec("infert_exact_init", "infert", "clogit", c("spontaneous", "induced"), "exact",
       strata = TRUE, init = c(2, -1)),
  spec("matched_exact", "matched", "clogit", c("m1", "m2", "m3"), "exact", strata = TRUE),
  spec("matched_approximate", "matched", "clogit", c("m1", "m2", "m3"), "approximate",
       strata = TRUE),
  spec("matched_efron", "matched", "clogit", c("m1", "m2", "m3"), "efron", strata = TRUE),
  spec("matched_exact_offset", "matched", "clogit", c("m1", "m2"), "exact", strata = TRUE,
       offset = TRUE),
  spec("matched_exact_iter2", "matched", "clogit", c("m1", "m2", "m3"), "exact", strata = TRUE,
       iter.max = 2)
)

run <- function(s, iter.max = s$iter.max, init = s$init) {
  dat <- datasets[[s$data]]
  d <- data.frame(start = dat$start, stop = dat$stop, event = dat$event,
                  stratum = dat$stratum, offset = dat$offset)
  d$X <- dat$columns[, s$cols, drop = FALSE]
  ctl <- list(timefix = FALSE)
  if (!is.null(iter.max)) ctl$iter.max <- iter.max
  if (!is.null(s$eps)) ctl$eps <- s$eps
  rhs <- "X"
  if (s$strata) rhs <- paste(rhs, "+ strata(stratum)")
  if (s$offset) rhs <- paste(rhs, "+ offset(offset)")
  warns <- character()
  catch <- function(expr) withCallingHandlers(expr, warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  if (s$kind == "cox") {
    lhs <- if (s$entry) "Surv(start, stop, event)" else "Surv(stop, event)"
    args <- list(formula = as.formula(paste(lhs, "~", rhs)), data = d, ties = s$method,
                 control = do.call(coxph.control, ctl))
    if (!is.null(init)) args$init <- init
    fit <- catch(do.call(coxph, args))
  } else {
    args <- list(formula = as.formula(paste("event ~", rhs)), data = d, method = s$method,
                 control = do.call(coxph.control, ctl))
    if (!is.null(init)) args$init <- init
    fit <- catch(do.call(clogit, args))
  }
  list(fit = fit, warns = warns, x = dat$columns[, s$cols, drop = FALSE])
}

fits <- list()
for (s in specs) {
  r <- run(s)
  fit <- r$fit
  k <- length(s$cols)
  dat <- datasets[[s$data]]
  nr <- nrow(dat$columns)
  # coxexact.fit, like coxph.fit, inverts an information matrix that it has
  # just recomputed and not factorised when it runs out of iterations, so the
  # variance of such a fit is wrong. The variance at the same coefficients is
  # that of a fit started there and not iterated.
  ran_out <- any(grepl("Ran out of iterations", r$warns))
  vfit <- fit
  if (ran_out && !s$entry) {
    at <- coef(fit)
    at[is.na(at)] <- 0
    vfit <- run(s, iter.max = 0, init = at)$fit
  }
  vn <- as.matrix(vfit$var)
  iter0 <- !is.null(s$iter.max) && s$iter.max == 0
  # With iter.max = 0 coxph leaves the starting values in place, aliased
  # columns included, and marks an aliased column by a zero variance.
  ok <- if (iter0) diag(vn) != 0 else !is.na(coef(fit))
  full <- function(values) {
    out <- rep(NA_real_, k)
    out[ok] <- values
    out
  }
  vn <- vn[ok, ok, drop = FALSE]
  scale_of <- function(xs) {
    centred <- sweep(xs, 2, colMeans(xs))
    sc <- colMeans(abs(centred))
    sc[sc == 0] <- 1
    sc
  }
  kappa_at <- function(v, cols) {
    if (!any(cols)) return(NA_real_)
    sc <- scale_of(r$x[, cols, drop = FALSE])
    kappa(solve(diag(sc, length(sc)) %*% v %*% diag(sc, length(sc))), exact = TRUE)
  }
  init <- if (is.null(s$init)) rep(0, k) else s$init
  r0 <- run(s, iter.max = 0, init = init)
  v0 <- as.matrix(r0$fit$var)
  ok0 <- diag(v0) != 0
  kappa0 <- kappa_at(v0[ok0, ok0, drop = FALSE], ok0)
  dd <- (coef(fit) - init)[ok]
  wald <- sum(dd * solve(vn, dd))
  kappa_path <- numeric()
  last_trial <- if (is.null(s$iter.max)) fit$iter else min(fit$iter, s$iter.max)
  for (j in seq_len(max(last_trial, 1)) - 1) {
    at <- if (j == 0) init else coef(run(s, iter.max = j)$fit)
    at[is.na(at)] <- 0
    run0 <- run(s, iter.max = 0, init = at)
    vj <- as.matrix(run0$fit$var)
    okj <- diag(vj) != 0
    kappa_path <- c(kappa_path, kappa_at(vj[okj, okj, drop = FALSE], okj))
  }
  monotone <- integer()
  for (m in r$warns) {
    if (grepl("may be infinite", m)) {
      monotone <- c(monotone, as.integer(strsplit(trimws(sub(";.*", "", sub(".*variable", "", m))), "[ ,]+")[[1]]))
    }
  }
  # Largest number of tied events, and of rows at risk, over the event times.
  d_max <- 0
  n_risk_max <- 0
  subsets_max <- 0
  for (st in unique(if (s$strata) dat$stratum else 0)) {
    rows <- if (s$strata) which(dat$stratum == st) else seq_len(nr)
    for (t in unique(dat$stop[rows][dat$event[rows] == 1])) {
      dying <- sum(dat$stop[rows] == t & dat$event[rows] == 1)
      risk <- sum(dat$stop[rows] >= t & (if (s$entry) dat$start[rows] < t else TRUE))
      d_max <- max(d_max, dying)
      n_risk_max <- max(n_risk_max, risk)
      subsets_max <- max(subsets_max, choose(risk, dying))
    }
  }
  embed <- function(v) {
    out <- matrix(NA_real_, k, k)
    out[ok, ok] <- v
    unname(out)
  }
  fits[[length(fits) + 1]] <- list(
    name = s$name, data = s$data, kind = s$kind, columns = I(s$cols), method = s$method,
    use_start = s$entry, use_strata = s$strata,
    init = I(if (is.null(s$init)) numeric() else s$init),
    use_offset = s$offset,
    iter_max = if (is.null(s$iter.max)) 20L else as.integer(s$iter.max),
    eps = if (is.null(s$eps)) 1e-9 else s$eps,
    coefficients = I(full(unname(coef(fit))[ok])),
    se = I(full(sqrt(diag(vn)))),
    vcov = I(embed(vn)),
    loglik_init = fit$loglik[1], loglik = fit$loglik[2],
    iterations = fit$iter, rank = sum(ok),
    ran_out = ran_out, variance_refit = ran_out && !s$entry,
    warnings = I(r$warns), monotone_columns = I(monotone),
    score = fit$score, wald = wald, n = fit$n, nevent = fit$nevent,
    d_max = d_max, n_risk_max = n_risk_max, subsets_max = subsets_max,
    kappa_fit = kappa_at(vn, ok), kappa_init = kappa0, kappa_path = I(kappa_path)
  )
}
datasets_out <- lapply(datasets, function(dat) {
  list(columns = unname(dat$columns), column_names = colnames(dat$columns),
       start = dat$start, stop = dat$stop, event = dat$event, stratum = dat$stratum,
       offset = dat$offset)
})
cat(toJSON(list(
  versions = list(
    r = R.version.string,
    survival = as.character(packageVersion("survival")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  datasets = datasets_out, fits = fits
), digits = I(17), auto_unbox = TRUE, na = "null"))
