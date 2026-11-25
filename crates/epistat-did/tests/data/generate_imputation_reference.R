# Reference values for the imputation estimator of Borusyak, Jaravel and
# Spiess and the two-stage estimator of Gardner
# (`epistat_did::estimate_did_imputation`, `epistat_did::estimate_did2s`),
# read by tests/parity_imputation.rs.
#
# Run from the crate root:  Rscript tests/data/generate_imputation_reference.R
#
# The references are didimputation::did_imputation 0.5.1 and did2s::did2s
# 1.2.1 from CRAN, with fixest 0.14.2. Every case records
#
#   package   what the package returns: estimates and standard errors, and for
#             did2s the corrected clustered covariance;
#   exact     the same quantities computed here from the formulas with dense
#             sparse-matrix solves (no iteration): the weights of the
#             estimator, the cluster sums, the standard errors and the
#             influence function of every unit, N times the unit's sum of
#             v_it (tau_it - tau_bar), with N the units in the estimation;
#   lambda    the smallest positive eigenvalues of D0' W D0 and of
#             [X0 D0]' W [X0 D0] with every unit and period dummy, from which
#             the tests derive their tolerances.
#
# The script checks its own formulas against the packages: did2s's covariance
# and the standard errors of did_imputation where the package is right. With
# weights that are not constant, did_imputation 0.5.1 solves the variance's
# auxiliary system with the squared weights: it forms
# v = -w Z0 (Z0' W^2 Z0)^-1 Z1' W wtr instead of -w Z0 (Z0' W Z0)^-1 Z1' wtr.
# The script reproduces the package's value with that formula
# (`package_formula_se`) and records the correct one in `exact`.
#
# Cases: the first 250 units of didimputation's df_hom, did2s's castle with its
# population weights, and three simulated staggered panels: unbalanced with
# row weights, zero weights, two covariates, always-treated units, a cohort
# treated after the panel and state clusters; no never-treated units, so that
# treated rows in the last periods cannot be imputed; and periods two years
# apart with unit weights.

suppressMessages({
  library(didimputation)
  library(did2s)
  library(fixest)
  library(data.table)
  library(Matrix)
  library(jsonlite)
})
stopifnot(
  packageVersion("didimputation") == "0.5.1",
  packageVersion("did2s") == "1.2.1",
  packageVersion("fixest") == "0.14.2",
  packageVersion("jsonlite") == "2.0.0"
)

sc <- jsonlite::unbox

# ---------------------------------------------------------------------------
# Exact computations

# The estimation sample: rows whose unit and period have untreated rows of
# positive weight. `d` has columns unit, time, g (NA for never treated), y, w,
# cl and the covariates named in `xs`.
prepare <- function(d, xs) {
  d$treat <- as.numeric(!is.na(d$g) & d$time >= d$g)
  d$e <- ifelse(is.na(d$g), -Inf, d$time - d$g)
  fit0 <- d$treat == 0 & d$w > 0
  u0 <- sort(unique(d$unit[fit0]))
  t0 <- sort(unique(d$time[fit0]))
  keep <- d$unit %in% u0 & d$time %in% t0
  E <- d[keep, ]
  E$fit0 <- E$treat == 0 & E$w > 0
  rhs <- c(xs, "factor(unit, levels = u0)", "factor(time, levels = t0)")
  Z <- sparse.model.matrix(stats::as.formula(paste("~", paste(rhs, collapse = " + "))), E)
  Z0 <- Z[E$fit0, , drop = FALSE]
  w0 <- E$w[E$fit0]
  M0 <- crossprod(Z0, w0 * Z0)
  stopifnot(qr(as.matrix(M0))$rank == ncol(M0))
  beta <- solve(M0, crossprod(Z0, w0 * E$y[E$fit0]))
  E$adj <- E$y - as.numeric(Z %*% beta)
  # Full dummies, for the eigenvalues of the tolerances.
  D <- cbind(sparse.model.matrix(~ 0 + factor(unit, levels = u0), E),
             sparse.model.matrix(~ 0 + factor(time, levels = t0), E))
  XD <- if (length(xs)) cbind(as.matrix(E[, xs, drop = FALSE]), as.matrix(D)) else as.matrix(D)
  pos <- function(A) {
    ev <- eigen(as.matrix(A), symmetric = TRUE, only.values = TRUE)$values
    c(min = min(ev[ev > 1e-9 * max(ev)]), max = max(ev))
  }
  lD <- pos(crossprod(D[E$fit0, ], w0 * D[E$fit0, ]))
  lF <- pos(crossprod(XD[E$fit0, , drop = FALSE], w0 * XD[E$fit0, , drop = FALSE]))
  list(E = E, Z = Z, M0 = M0, beta = beta, dropped = nrow(d) - nrow(E),
       lambda = list(d_plus = sc(lD[["min"]]), full_plus = sc(lF[["min"]]), full_max = sc(lF[["max"]])))
}

# The imputation estimates for the treatment weights `wtr` (a matrix over the
# rows of the estimation sample, zero on untreated rows), with the variance of
# Borusyak et al. clustered by `cl`. `squared` gives the package's form.
imputation_exact <- function(p, wtr, squared = FALSE) {
  E <- p$E
  Z <- p$Z
  wtr <- wtr * E$w
  wtr <- sweep(wtr, 2, colSums(wtr), "/")
  est <- colSums(wtr * E$adj)
  treated <- E$treat == 1
  Z1 <- Z[treated, , drop = FALSE]
  Z0 <- Z[E$fit0, , drop = FALSE]
  w0 <- E$w[E$fit0]
  v <- matrix(0, nrow(E), ncol(wtr))
  v[treated, ] <- wtr[treated, ]
  if (squared) {
    g <- solve(crossprod(Z0, w0^2 * Z0), crossprod(Z1, E$w[treated] * wtr[treated, , drop = FALSE]))
  } else {
    g <- solve(p$M0, crossprod(Z1, wtr[treated, , drop = FALSE]))
  }
  v[E$fit0, ] <- -w0 * as.matrix(Z0 %*% g)
  cell <- paste(E$g, E$e)
  tcol <- matrix(E$adj, nrow(E), ncol(wtr))
  for (k in seq_len(ncol(wtr))) {
    num <- tapply(v[, k]^2 * E$adj, cell, sum)
    den <- tapply(v[, k]^2, cell, sum)
    bar <- ifelse(den > 0, num / den, 0)[cell]
    tcol[treated, k] <- E$adj[treated] - bar[treated]
  }
  contrib <- v * tcol
  S <- rowsum(contrib, E$cl)
  units <- sort(unique(E$unit))
  inf <- length(units) * rowsum(contrib, E$unit)[as.character(units), , drop = FALSE]
  list(estimate = unname(est), std_error = sqrt(colSums(S^2)),
       covariance = unname(crossprod(S)), influence = unname(inf), units = units)
}

# The pre-trend regression on the untreated rows of positive weight: the
# indicators of the relative periods `periods` (the smallest relative period,
# the never treated when there are any, is the reference of fixest::i), the
# covariates and the fixed effects. `exact` comes from lm's QR, NA for
# aliased indicators; `kappa` is the condition number of the demeaned design
# scaled to unit columns; fixest at fixef.tol = 1e-11, its smallest, gives the clustered
# standard errors with its small-sample correction, and `tight_error` is the
# largest distance of its residuals from lm's over their root mean square.
pretrend_exact <- function(p, xs, spec) {
  E <- p$E[p$E$fit0, ]
  levels <- sort(unique(E$e))
  free <- levels[-1]
  periods <- if (isTRUE(spec)) free else sort(spec)
  fitted <- periods[periods %in% free]
  P <- sapply(fitted, function(k) as.numeric(E$e == k))
  P <- matrix(P, ncol = length(fitted))
  colnames(P) <- paste0("p", seq_along(fitted))
  X <- cbind(P, as.matrix(E[, xs, drop = FALSE]))
  D <- model.matrix(~ factor(unit) + factor(time), E)
  # Fixed effects first, so that the QR drops the later, collinear indicators
  # as fixest does.
  m <- lm.wfit(cbind(D, X), E$y, E$w)
  est <- m$coefficients[ncol(D) + seq_len(ncol(P))]
  # Demeaned design, exactly, and its condition number over kept columns.
  Xt <- qr.resid(qr(sqrt(E$w) * D), sqrt(E$w) * X)
  keep <- !is.na(m$coefficients[ncol(D) + seq_len(ncol(X))])
  G <- crossprod(Xt[, keep, drop = FALSE])
  sdv <- sqrt(diag(G))
  ev <- eigen(G / outer(sdv, sdv), symmetric = TRUE, only.values = TRUE)$values
  E$pre <- factor(ifelse(E$e %in% fitted, E$e, NA))
  for (j in seq_along(fitted)) E[[paste0("p", j)]] <- P[, j]
  f <- stats::as.formula(paste("y ~", paste(c(colnames(P), xs), collapse = " + "), "| unit + time"))
  tight <- feols(f, E, weights = ~w, cluster = ~cl, fixef.rm = "none", fixef.tol = 1e-11,
                 fixef.iter = 100000, warn = FALSE, notes = FALSE)
  se <- rep(NA_real_, length(periods))
  ct <- coeftable(tight)
  for (j in seq_along(fitted)) {
    name <- paste0("p", j)
    if (name %in% rownames(ct)) se[match(fitted[j], periods)] <- ct[name, "Std. Error"]
  }
  e_exact <- rep(NA_real_, length(periods))
  e_exact[match(fitted, periods)] <- est
  r_exact <- m$residuals
  r_tight <- stats::residuals(tight)
  list(periods = periods, estimate = e_exact, std_error = se,
       kappa = sc(max(ev) / min(ev)),
       tight_error = sc(max(abs(r_tight - r_exact)) / sqrt(mean(r_exact^2))))
}

# The target columns of did_imputation for the static effect or horizons.
targets <- function(E, horizons) {
  if (is.null(horizons)) {
    m <- matrix(E$treat, ncol = 1)
    colnames(m) <- "treat"
    return(m)
  }
  m <- sapply(horizons, function(h) as.numeric(E$e == h))
  m <- matrix(m, ncol = length(horizons))
  colnames(m) <- horizons
  m
}

# did2s with indicator second stages: the coefficients, the corrected
# covariance of Gardner and the influence of every unit.
did2s_exact <- function(p, spec, refs) {
  E <- p$E
  Z <- p$Z
  if (spec == "static") {
    X2 <- matrix(E$treat, ncol = 1)
    terms <- "static"
  } else {
    rel <- sort(unique(E$e[is.finite(E$e) & !(E$e %in% refs)]))
    X2 <- sapply(rel, function(r) as.numeric(E$e == r))
    X2 <- matrix(X2, ncol = length(rel))
    present <- colSums(E$w * X2) > 0
    X2 <- X2[, present, drop = FALSE]
    terms <- rel[present]
  }
  w <- E$w
  B2 <- solve(crossprod(X2, w * X2))
  beta2 <- as.numeric(B2 %*% crossprod(X2, w * E$adj))
  e2 <- E$adj - as.numeric(X2 %*% beta2)
  Gam <- solve(p$M0, crossprod(Z, w * X2))
  h <- as.matrix(Z %*% Gam)
  u <- ifelse(E$fit0, E$adj, 0)
  IF <- (w * e2 * X2 - w * u * h) %*% B2
  S <- rowsum(IF, E$cl)
  units <- sort(unique(E$unit))
  inf <- length(units) * rowsum(IF, E$unit)[as.character(units), , drop = FALSE]
  list(terms = terms, estimate = beta2, covariance = unname(crossprod(S)),
       std_error = sqrt(diag(crossprod(S))), influence = unname(inf))
}

# ---------------------------------------------------------------------------
# Package calls

pkg_frame <- function(d) {
  f <- as.data.frame(d)
  f$gz <- ifelse(is.na(f$g), 0, f$g)
  f$rel <- ifelse(is.na(f$g), Inf, f$time - f$g)
  f$treat <- as.numeric(!is.na(f$g) & f$time >= f$g)
  f
}

first_stage_of <- function(xs) {
  stats::as.formula(paste("~", if (length(xs)) paste(xs, collapse = " + ") else "0", "| unit + time"))
}

run_imputation <- function(d, xs, p, spec) {
  f <- pkg_frame(d)
  args <- list(data = f, yname = "y", gname = "gz", tname = "time", idname = "unit",
               first_stage = first_stage_of(xs), cluster_var = "cl")
  if (spec$weighted) args$wname <- "w"
  wt_cols <- NULL
  if (identical(spec$target, "horizons")) {
    args$horizon <- if (is.null(spec$horizons)) TRUE else spec$horizons
  } else if (identical(spec$target, "weights")) {
    f$custom <- spec$custom
    args$data <- f
    args$wtr <- "custom"
  }
  if (!is.null(spec$pretrends)) args$pretrends <- spec$pretrends
  out <- suppressWarnings(do.call(did_imputation, args))
  out <- as.data.frame(out)
  num <- suppressWarnings(as.numeric(out$term))
  is_pre <- !is.na(num) & num < 0
  pre <- out[is_pre, ]
  post <- out[!is_pre, ]

  E <- p$E
  if (identical(spec$target, "weights")) {
    wtr <- matrix(spec$custom[match(paste(E$unit, E$time), paste(d$unit, d$time))] * E$treat, ncol = 1)
    colnames(wtr) <- "custom"
  } else if (identical(spec$target, "horizons")) {
    h <- if (is.null(spec$horizons)) sort(unique(d$time - d$g)[is.finite(unique(d$time - d$g))]) else spec$horizons
    h <- h[h >= 0]
    wtr <- targets(E, h)
  } else {
    wtr <- targets(E, NULL)
  }
  exact <- imputation_exact(p, wtr)
  pkg_form <- imputation_exact(p, wtr, squared = TRUE)
  # The package's terms in our order.
  pkg_se <- post$std.error[match(colnames(wtr), post$term)]
  pkg_est <- post$estimate[match(colnames(wtr), post$term)]
  ok <- !is.na(pkg_est)
  # fixest stops its demeaning at fixef.tol = 1e-6, so the package is
  # accurate to about 1e-8 here; the distance is recorded for the tests.
  est_error <- max(abs(pkg_est[ok] - exact$estimate[ok]))
  se_error <- max(abs(pkg_se[ok] - pkg_form$std_error[ok]))
  stopifnot(est_error < 1e-6, se_error < 1e-6 * max(pkg_se[ok]))
  list(
    name = sc(spec$name),
    target = sc(spec$target),
    weighted = sc(spec$weighted),
    horizons = if (identical(spec$target, "horizons")) as.numeric(colnames(wtr)) else NULL,
    all_horizons = sc(identical(spec$target, "horizons") && is.null(spec$horizons)),
    custom = if (identical(spec$target, "weights")) spec$custom else NULL,
    pretrends = if (isTRUE(spec$pretrends)) sc("all") else spec$pretrends,
    package = list(estimate = pkg_est, std_error = pkg_se,
                   pre_terms = as.numeric(pre$term), pre_estimate = pre$estimate,
                   pre_std_error = pre$std.error),
    pretrend = if (is.null(spec$pretrends)) NULL else pretrend_exact(p, xs, spec$pretrends),
    package_formula_se = pkg_form$std_error,
    package_error = list(estimate = sc(est_error), std_error = sc(se_error)),
    exact = exact[c("estimate", "std_error", "covariance", "influence")]
  )
}

run_did2s <- function(d, xs, p, spec) {
  f <- pkg_frame(d)
  second <- if (spec$spec == "static") ~ i(treat, ref = 0) else
    stats::as.formula(paste0("~ i(rel, ref = c(", paste(c(spec$references, "Inf"), collapse = ", "), "))"))
  args <- list(data = f, yname = "y", first_stage = first_stage_of(xs), second_stage = second,
               treatment = "treat", cluster_var = "cl")
  if (spec$weighted) args$weights <- "w"
  out <- suppressWarnings(do.call(did2s, args))
  exact <- did2s_exact(p, spec$spec, spec$references)
  V <- unname(as.matrix(stats::vcov(out)))
  est_error <- max(abs(stats::coef(out) - exact$estimate))
  se_error <- max(abs(sqrt(diag(V)) - exact$std_error))
  stopifnot(est_error < 1e-6, se_error < 1e-6 * max(sqrt(diag(V))))
  list(name = sc(spec$name), spec = sc(spec$spec), references = spec$references,
       weighted = sc(spec$weighted),
       package = list(terms = names(stats::coef(out)), estimate = unname(stats::coef(out)),
                      std_error = sqrt(diag(V)), covariance = V),
       package_error = list(estimate = sc(est_error), std_error = sc(se_error)),
       exact = exact[c("estimate", "std_error", "covariance", "influence")],
       terms = if (spec$spec == "static") NULL else exact$terms)
}

# ---------------------------------------------------------------------------
# Data

simulate <- function(seed, n_units, times, cohorts, p_never, drop = 0, weights = "none",
                     n_x = 0, n_states = NULL, n_zero = 0) {
  set.seed(seed)
  ids <- sort(sample.int(5L * n_units, n_units))
  g <- sample(c(cohorts, NA), n_units, replace = TRUE,
              prob = c(rep((1 - p_never) / length(cohorts), length(cohorts)), p_never))
  d <- expand.grid(time = times, k = seq_len(n_units))
  d$unit <- ids[d$k]
  d$g <- g[d$k]
  if (drop > 0) d <- d[stats::runif(nrow(d)) > drop, ]
  alpha <- stats::rnorm(n_units)
  lambda <- stats::rnorm(length(times))
  d$treat <- !is.na(d$g) & d$time >= d$g
  e <- ifelse(d$treat, d$time - d$g, 0)
  tau <- ifelse(d$treat, 1 + 0.15 * e + 0.2 * match(d$g, cohorts) + 0.5 * stats::rnorm(nrow(d)), 0)
  y <- alpha[d$k] + lambda[match(d$time, times)] + tau + stats::rnorm(nrow(d))
  xs <- character(0)
  for (j in seq_len(n_x)) {
    x <- stats::rnorm(nrow(d)) + 0.3 * d$treat + 0.1 * alpha[d$k]
    d[[paste0("x", j)]] <- x
    y <- y + c(0.7, -0.4)[j] * x
    xs <- c(xs, paste0("x", j))
  }
  d$y <- y
  d$w <- switch(weights,
    none = rep(1, nrow(d)),
    row = stats::runif(nrow(d), 0.5, 2),
    unit = stats::runif(n_units, 0.5, 2)[d$k])
  if (n_zero > 0) {
    # Zero weights on untreated rows of units that keep other untreated rows.
    cand <- which(!d$treat & ave(as.numeric(!d$treat), d$unit, FUN = sum) >= 4)
    d$w[sample(cand, n_zero)] <- 0
  }
  d$cl <- if (is.null(n_states)) d$unit else (match(d$unit, ids) %% n_states) + 1
  d$k <- NULL
  d$treat <- NULL
  d <- d[order(d$unit, d$time), ]
  rownames(d) <- NULL
  list(d = d, xs = xs)
}

data("df_hom", package = "didimputation")
hom <- as.data.frame(df_hom)
hom <- hom[hom$unit <= 250, ]
hom <- data.frame(unit = hom$unit, time = hom$year, g = ifelse(hom$g == 0, NA, hom$g),
                  y = hom$dep_var, w = 1, cl = hom$state)

data("castle", package = "did2s")
castle <- as.data.frame(castle)
first_post <- tapply(ifelse(castle$post == 1, castle$year, NA), castle$state,
                     function(z) if (all(is.na(z))) NA else min(z, na.rm = TRUE))
castle <- data.frame(unit = castle$state, time = castle$year,
                     g = as.numeric(first_post[as.character(castle$state)]),
                     y = castle$l_homicide, w = castle$popwt, cl = castle$state)
stopifnot(all((!is.na(castle$g) & castle$time >= castle$g) ==
              (as.data.frame(did2s::castle)$post == 1)))

sim_full <- simulate(11, 120, 2001:2012, c(1999, 2004, 2006, 2009, 2015), 0.25, drop = 0.15,
                     weights = "row", n_x = 2, n_states = 15, n_zero = 6)
sim_nonever <- simulate(12, 80, 1:10, c(4, 6, 8), 0, weights = "none", n_x = 1)
sim_gaps <- simulate(13, 100, seq(2, 20, by = 2), c(8, 12), 0.3, weights = "unit", n_states = 20)

custom <- with(sim_full$d, ifelse(!is.na(g) & g == 2006 & time >= g, time - 2005, 0))

cases <- list(
  list(name = "df_hom", d = hom, xs = character(0),
       imputation = list(
         list(name = "static", target = "static", weighted = FALSE),
         list(name = "all_horizons_pretrends", target = "horizons", weighted = FALSE, pretrends = TRUE),
         list(name = "horizons_pretrends_3", target = "horizons", horizons = c(0, 1, 5, 10),
              weighted = FALSE, pretrends = -3:-1)),
       did2s = list(
         list(name = "static", spec = "static", references = numeric(0), weighted = FALSE),
         list(name = "event", spec = "event", references = -1, weighted = FALSE))),
  list(name = "castle", d = castle, xs = character(0),
       imputation = list(
         list(name = "static_weighted", target = "static", weighted = TRUE),
         list(name = "horizons_weighted", target = "horizons", weighted = TRUE),
         list(name = "static", target = "static", weighted = FALSE)),
       did2s = list(
         list(name = "static_weighted", spec = "static", references = numeric(0), weighted = TRUE),
         list(name = "event_weighted", spec = "event", references = -1, weighted = TRUE))),
  list(name = "sim_full", d = sim_full$d, xs = sim_full$xs,
       imputation = list(
         list(name = "static", target = "static", weighted = TRUE),
         list(name = "horizons", target = "horizons", horizons = c(0, 1, 2, 5), weighted = TRUE,
              pretrends = c(-1, -3)),
         list(name = "all_horizons_pretrends", target = "horizons", weighted = TRUE, pretrends = TRUE),
         list(name = "custom", target = "weights", custom = custom, weighted = TRUE),
         list(name = "unweighted", target = "horizons", horizons = c(0, 3), weighted = FALSE)),
       did2s = list(
         list(name = "static", spec = "static", references = numeric(0), weighted = TRUE),
         list(name = "event", spec = "event", references = -1, weighted = TRUE),
         list(name = "event_two_refs", spec = "event", references = c(-1, -4), weighted = TRUE))),
  list(name = "sim_nonever", d = sim_nonever$d, xs = sim_nonever$xs,
       imputation = list(
         list(name = "all_horizons_pretrends", target = "horizons", weighted = FALSE, pretrends = TRUE),
         list(name = "static", target = "static", weighted = FALSE)),
       did2s = list(
         list(name = "event", spec = "event", references = -1, weighted = FALSE))),
  list(name = "sim_gaps", d = sim_gaps$d, xs = sim_gaps$xs,
       imputation = list(
         list(name = "all_horizons", target = "horizons", weighted = TRUE, pretrends = c(-2, -4)),
         list(name = "static", target = "static", weighted = TRUE)),
       did2s = list(
         list(name = "event", spec = "event", references = -2, weighted = TRUE)))
)

out <- lapply(cases, function(cs) {
  d <- cs$d
  p_w <- prepare(d, cs$xs)
  d1 <- d
  d1$w <- 1
  p_1 <- prepare(d1, cs$xs)
  imp <- lapply(cs$imputation, function(s) run_imputation(if (s$weighted) d else d1, cs$xs,
                                                          if (s$weighted) p_w else p_1, s))
  two <- lapply(cs$did2s, function(s) run_did2s(if (s$weighted) d else d1, cs$xs,
                                                if (s$weighted) p_w else p_1, s))
  cat(sprintf("%-12s rows %5d  dropped %4d (weighted) %4d (unweighted)\n", cs$name, nrow(d),
              p_w$dropped, p_1$dropped))
  list(
    name = sc(cs$name),
    unit = d$unit, time = d$time, g = d$g, y = d$y, w = d$w, cl = d$cl,
    x = if (length(cs$xs)) lapply(cs$xs, function(x) d[[x]]) else list(),
    units = sort(unique(p_w$E$unit)), units_unweighted = sort(unique(p_1$E$unit)),
    dropped = sc(p_w$dropped), dropped_unweighted = sc(p_1$dropped),
    lambda = p_w$lambda, lambda_unweighted = p_1$lambda,
    imputation = imp, did2s = two
  )
})

ref <- list(
  versions = list(R = sc(paste(R.version$major, R.version$minor, sep = ".")),
                  didimputation = sc(as.character(packageVersion("didimputation"))),
                  did2s = sc(as.character(packageVersion("did2s"))),
                  fixest = sc(as.character(packageVersion("fixest")))),
  cases = out
)
writeLines(toJSON(ref, auto_unbox = FALSE, digits = NA, na = "null", null = "null"),
           "tests/data/imputation_ref.json")
