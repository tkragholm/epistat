# Reference values for synthetic difference-in-differences, synthetic control
# and difference-in-differences from the R package synthdid.
#
# Run from the crate root:  Rscript tests/data/generate_synthdid_reference.R
#
# Written for tests/parity_synthdid.rs. synthdid is not on CRAN. The version
# used is 0.0.9 from the authors' repository, github.com/synth-inference/synthdid,
# commit 70c1ce3eac58e28c30b67435ca377bb48baa9b8a, which the script checks.
#
# Scenarios
#   california   the package's California Proposition 99 panel (38 controls,
#                1 treated, 19 of 31 years before), the three estimators, and
#                the placebo with R's draws; the jackknife and the bootstrap
#                are NA with one treated unit
#   sim          a simulated factor-model panel, 50 controls and 10 treated,
#                20 of 30 periods before; the three estimators with the
#                jackknife and with the placebo and the bootstrap on R's
#                draws, and options of synthdid_estimate changed one at a time
#   covariates   a smaller panel with two covariates, the three estimators,
#                and the synthdid estimator's jackknife, placebo and bootstrap
#   staggered    three adoption cohorts and never-treated units, estimated
#                cohort by cohort with synthdid_estimate and aggregated by
#                treated cells, written out here because synthdid takes one
#                adoption date; the jackknife, placebo and bootstrap of the
#                whole procedure on R's draws
#
# The placebo and bootstrap draws are R's own: the script draws them as
# placebo_se and bootstrap_sample do, refits each as their inner function
# theta does, and checks that the standard error equals synthdid_se's under
# the same seed. Draws are stored 0-based. Each fit stores its weights, the
# effect curve and the number of Frank-Wolfe steps of the last runs.
#
# Versions: R 4.6.2, synthdid 0.0.9 (commit above), jsonlite 2.0.0.

suppressMessages({
  library(synthdid)
  library(jsonlite)
  stopifnot(
    packageVersion("synthdid") == "0.0.9",
    packageDescription("synthdid")$RemoteSha == "70c1ce3eac58e28c30b67435ca377bb48baa9b8a",
    packageVersion("jsonlite") == "2.0.0"
  )
})
sum_normalize <- synthdid:::sum_normalize

steps <- function(v) if (is.null(v)) 0L else sum(!is.na(v))
last_value <- function(v) if (is.null(v) || all(is.na(v))) NA else v[max(which(!is.na(v)))]

fit_info <- function(est) {
  w <- attr(est, "weights")
  o <- attr(est, "opts")
  s <- attr(est, "setup")
  covariates <- dim(s$X)[3] > 0
  list(
    estimate = c(est),
    omega = w$omega,
    lambda = w$lambda,
    beta = if (is.null(w$beta)) numeric(0) else w$beta,
    curve = c(synthdid_effect_curve(est)),
    zeta_omega = o$zeta.omega,
    zeta_lambda = o$zeta.lambda,
    min_decrease = o$min.decrease,
    lambda_steps = if (covariates) 0L else steps(w$lambda.vals),
    omega_steps = if (covariates) 0L else steps(w$omega.vals),
    joint_steps = if (covariates) steps(w$vals) else 0L,
    lambda_last = if (covariates) NA else last_value(w$lambda.vals),
    omega_last = if (covariates) NA else last_value(w$omega.vals),
    joint_last = if (covariates) last_value(w$vals) else NA
  )
}

# The inner functions of placebo_se and bootstrap_sample, applied to given
# draws; each returns the refit.
placebo_refit <- function(estimate, ind) {
  setup <- attr(estimate, "setup"); opts <- attr(estimate, "opts")
  weights <- attr(estimate, "weights")
  N1 <- nrow(setup$Y) - setup$N0
  N0 <- length(ind) - N1
  weights.boot <- weights
  weights.boot$omega <- sum_normalize(weights$omega[ind[1:N0]])
  do.call(synthdid_estimate, c(list(Y = setup$Y[ind, ], N0 = N0, T0 = setup$T0,
                                    X = setup$X[ind, , , drop = FALSE], weights = weights.boot), opts))
}
bootstrap_refit <- function(estimate, ind) {
  setup <- attr(estimate, "setup"); opts <- attr(estimate, "opts")
  weights <- attr(estimate, "weights")
  weights.boot <- weights
  weights.boot$omega <- sum_normalize(weights$omega[sort(ind[ind <= setup$N0])])
  do.call(synthdid_estimate, c(list(Y = setup$Y[sort(ind), ], N0 = sum(ind <= setup$N0),
                                    T0 = setup$T0, X = setup$X[sort(ind), , , drop = FALSE],
                                    weights = weights.boot), opts))
}
jackknife_refit <- function(estimate, ind) {
  setup <- attr(estimate, "setup"); opts <- attr(estimate, "opts")
  weights <- attr(estimate, "weights")
  opts$update.omega <- opts$update.lambda <- FALSE
  weights.jk <- weights
  weights.jk$omega <- sum_normalize(weights$omega[ind[ind <= setup$N0]])
  do.call(synthdid_estimate, c(list(Y = setup$Y[ind, ], N0 = sum(ind <= setup$N0),
                                    T0 = setup$T0, X = setup$X[ind, , , drop = FALSE],
                                    weights = weights.jk), opts))
}

placebo_draws <- function(N0, B, seed) {
  set.seed(seed)
  replicate(B, sample(1:N0), simplify = FALSE)
}
bootstrap_draws <- function(N, N0, B, seed) {
  set.seed(seed)
  out <- list()
  while (length(out) < B) {
    ind <- sample(1:N, replace = TRUE)
    if (!(all(ind <= N0) || all(ind > N0))) out[[length(out) + 1]] <- ind
  }
  out
}
replicate_se <- function(u) { u <- u[is.finite(u)]; B <- length(u); sqrt((B - 1) / B) * sd(u) }
jack_se <- function(u) { n <- length(u); sqrt(((n - 1) / n) * (n - 1) * var(u)) }

replicates <- function(est, draws, refit, se) {
  fits <- lapply(draws, function(ind) refit(est, ind))
  u <- sapply(fits, c)
  curves <- t(sapply(fits, function(f) c(synthdid_effect_curve(f))))
  if (nrow(curves) == 1) curves <- t(curves)
  list(draws = lapply(draws, function(d) d - 1L), replicates = u, se = se(u),
       curve_se = apply(curves, 2, se))
}

placebo_info <- function(est, B, seed) {
  N0 <- attr(est, "setup")$N0
  r <- replicates(est, placebo_draws(N0, B, seed), placebo_refit, replicate_se)
  set.seed(seed)
  stopifnot(isTRUE(all.equal(r$se, c(synthdid_se(est, method = "placebo", replications = B)),
                             tolerance = 1e-14)))
  r
}
bootstrap_info <- function(est, B, seed) {
  s <- attr(est, "setup")
  r <- replicates(est, bootstrap_draws(nrow(s$Y), s$N0, B, seed), bootstrap_refit, replicate_se)
  set.seed(seed)
  stopifnot(isTRUE(all.equal(r$se, c(synthdid_se(est, method = "bootstrap", replications = B)),
                             tolerance = 1e-14)))
  r
}
jackknife_info <- function(est) {
  n <- nrow(attr(est, "setup")$Y)
  fits <- lapply(1:n, function(i) jackknife_refit(est, (1:n)[-i]))
  u <- sapply(fits, c)
  curves <- t(sapply(fits, function(f) c(synthdid_effect_curve(f))))
  se <- jack_se(u)
  stopifnot(isTRUE(all.equal(se, c(synthdid_se(est, method = "jackknife")), tolerance = 1e-14)))
  list(replicates = u, se = se, curve_se = apply(curves, 2, jack_se))
}

matrix_rows <- function(Y) lapply(seq_len(nrow(Y)), function(i) unname(Y[i, ]))
covariate_blocks <- function(X) lapply(seq_len(dim(X)[3]), function(k) matrix_rows(X[, , k]))

ref <- list(
  r_version = R.version.string,
  synthdid_version = as.character(packageVersion("synthdid")),
  synthdid_commit = packageDescription("synthdid")$RemoteSha
)

## California Proposition 99 --------------------------------------------------
data("california_prop99")
setup <- panel.matrices(california_prop99)
Y <- setup$Y
cal <- list(Y = matrix_rows(Y), N0 = setup$N0, T0 = setup$T0,
            noise_level = sd(apply(Y[1:setup$N0, 1:setup$T0], 1, diff)))
# The long data with each state numbered in panel.matrices' order of names.
states <- sort(unique(california_prop99$State))
cal$long <- list(unit = match(california_prop99$State, states), time = california_prop99$Year,
                 outcome = california_prop99$PacksPerCapita, treated = california_prop99$treated)
cal$row_units <- match(rownames(Y), states)
for (name in c("synthdid", "sc", "did")) {
  estimator <- switch(name, synthdid = synthdid_estimate, sc = sc_estimate, did = did_estimate)
  est <- estimator(Y, setup$N0, setup$T0)
  stopifnot(is.na(synthdid_se(est, method = "jackknife")), is.na(synthdid_se(est, method = "bootstrap")))
  cal[[name]] <- fit_info(est)
  cal[[name]]$placebo <- placebo_info(est, 100, 1)
}
ref$california <- cal

## Simulated panel ---------------------------------------------------------------
set.seed(20261125)
N <- 60; N0 <- 50; Tn <- 30; T0 <- 20
L <- 3
U <- matrix(rnorm(N * L), N, L); V <- matrix(rnorm(Tn * L), Tn, L)
U[(N0 + 1):N, 1] <- U[(N0 + 1):N, 1] + 1
Y <- U %*% t(V) + outer(rnorm(N, sd = 2), rep(1, Tn)) + outer(rep(1, N), cumsum(rnorm(Tn, sd = 0.5))) +
  matrix(rnorm(N * Tn, sd = 0.5), N, Tn)
Y[(N0 + 1):N, (T0 + 1):Tn] <- Y[(N0 + 1):N, (T0 + 1):Tn] + 1.5
sim <- list(Y = matrix_rows(Y), N0 = N0, T0 = T0,
            noise_level = sd(apply(Y[1:N0, 1:T0], 1, diff)))
for (name in c("synthdid", "sc", "did")) {
  estimator <- switch(name, synthdid = synthdid_estimate, sc = sc_estimate, did = did_estimate)
  est <- estimator(Y, N0, T0)
  B <- if (name == "synthdid") 200 else 50
  sim[[name]] <- fit_info(est)
  sim[[name]]$jackknife <- jackknife_info(est)
  sim[[name]]$placebo <- placebo_info(est, B, 2)
  sim[[name]]$bootstrap <- bootstrap_info(est, B, 3)
}
lambda_given <- sum_normalize(c(rep(0, T0 - 3), 1, 2, 3))
sim$options <- list(
  no_sparsify = fit_info(synthdid_estimate(Y, N0, T0, sparsify = NULL)),
  no_intercepts = fit_info(synthdid_estimate(Y, N0, T0, omega.intercept = FALSE, lambda.intercept = FALSE)),
  eta_omega = fit_info(synthdid_estimate(Y, N0, T0, eta.omega = 0.3)),
  noise_level = fit_info(synthdid_estimate(Y, N0, T0, noise.level = 2)),
  max_iter = fit_info(synthdid_estimate(Y, N0, T0, max.iter = 50, max.iter.pre.sparsify = 10)),
  lambda_fixed = fit_info(synthdid_estimate(Y, N0, T0, weights = list(lambda = lambda_given))),
  lambda_start = fit_info(synthdid_estimate(Y, N0, T0, weights = list(lambda = lambda_given),
                                            update.lambda = TRUE))
)
sim$options$lambda_given <- lambda_given

## Covariates ------------------------------------------------------------------
set.seed(20261126)
N <- 30; N0 <- 24; Tn <- 15; T0 <- 10; K <- 2
X <- array(rnorm(N * Tn * K), c(N, Tn, K))
X[(N0 + 1):N, , 1] <- X[(N0 + 1):N, , 1] + 0.5
Y <- outer(rnorm(N, sd = 2), rep(1, Tn)) + outer(rep(1, N), cumsum(rnorm(Tn, sd = 0.5))) +
  0.8 * X[, , 1] - 0.4 * X[, , 2] + matrix(rnorm(N * Tn, sd = 0.5), N, Tn)
Y[(N0 + 1):N, (T0 + 1):Tn] <- Y[(N0 + 1):N, (T0 + 1):Tn] + 1
cov <- list(Y = matrix_rows(Y), X = covariate_blocks(X), N0 = N0, T0 = T0)
for (name in c("synthdid", "sc", "did")) {
  estimator <- switch(name, synthdid = synthdid_estimate, sc = sc_estimate, did = did_estimate)
  est <- estimator(Y, N0, T0, X = X)
  cov[[name]] <- fit_info(est)
  if (name == "synthdid") {
    cov[[name]]$jackknife <- jackknife_info(est)
    cov[[name]]$placebo <- placebo_info(est, 10, 4)
    cov[[name]]$bootstrap <- bootstrap_info(est, 10, 5)
  }
}
ref$sim <- sim
ref$covariates <- cov

## Staggered adoption ----------------------------------------------------------
# adoption: 0-based first treated period, NA for never treated.
set.seed(20261127)
Tn <- 18
adoption <- c(rep(NA, 30), rep(8, 5), rep(11, 6), rep(14, 4))
N <- length(adoption)
U <- matrix(rnorm(N * 2), N, 2); V <- matrix(rnorm(Tn * 2), Tn, 2)
Y <- U %*% t(V) + outer(rnorm(N, sd = 2), rep(1, Tn)) + outer(rep(1, N), cumsum(rnorm(Tn, sd = 0.5))) +
  matrix(rnorm(N * Tn, sd = 0.5), N, Tn)
for (i in which(!is.na(adoption))) {
  post <- (adoption[i] + 1):Tn
  Y[i, post] <- Y[i, post] + 1 + 0.1 * seq_along(post)
}
# Shuffle the rows so that the cohorts are not in blocks.
perm <- sample(N)
Y <- Y[perm, ]; adoption <- adoption[perm]
controls <- which(is.na(adoption))
cohorts <- sort(unique(adoption[!is.na(adoption)]))
treated_all <- sort(which(!is.na(adoption)))

aggregate_cohorts <- function(values) {
  # values: list of list(n, post, estimate, curve)
  cells <- sapply(values, function(v) v$n * v$post)
  estimate <- 0
  for (i in seq_along(values)) estimate <- estimate + (cells[i] / sum(cells)) * values[[i]]$estimate
  len <- max(Tn - cohorts)
  curve <- sapply(0:(len - 1), function(e) {
    present <- Filter(function(v) v$post > e, values)
    if (length(present) == 0) return(NA)
    size <- sum(sapply(present, function(v) v$n))
    s <- 0
    for (v in present) s <- s + (v$n / size) * v$curve[e + 1]
    s
  })
  list(estimate = estimate, curve = curve)
}
cohort_value <- function(f, n, a) list(n = n, post = Tn - a, estimate = c(f), curve = c(synthdid_effect_curve(f)))

stag <- list(Y = matrix_rows(Y), adoption = adoption)
for (name in c("synthdid", "did")) {
  estimator <- switch(name, synthdid = synthdid_estimate, did = did_estimate)
  fits <- lapply(cohorts, function(a) {
    tr <- which(adoption == a)
    estimator(Y[c(controls, tr), ], length(controls), a)
  })
  values <- mapply(function(f, a) cohort_value(f, sum(adoption == a, na.rm = TRUE), a), fits, cohorts,
                   SIMPLIFY = FALSE)
  agg <- aggregate_cohorts(values)
  refit <- function(a_index, ctrl_units, ctrl_pos, tr_units, fixed) {
    f <- fits[[a_index]]
    opts <- attr(f, "opts"); w <- attr(f, "weights")
    if (fixed) opts$update.omega <- opts$update.lambda <- FALSE
    w$omega <- sum_normalize(w$omega[ctrl_pos])
    do.call(synthdid_estimate, c(list(Y = Y[c(ctrl_units, tr_units), ], N0 = length(ctrl_units),
                                      T0 = cohorts[a_index], weights = w), opts))
  }
  # Jackknife: leave each unit out of every cohort, weights fixed.
  jk <- lapply(1:N, function(i) {
    keep_pos <- which(controls != i)
    vals <- lapply(seq_along(cohorts), function(k) {
      tr <- setdiff(which(adoption == cohorts[k]), i)
      cohort_value(refit(k, controls[keep_pos], keep_pos, tr, TRUE), length(tr), cohorts[k])
    })
    aggregate_cohorts(vals)
  })
  u <- sapply(jk, function(r) r$estimate)
  jk_curves <- t(sapply(jk, function(r) r$curve))
  # Placebo: permute the controls; the last ones take the treated units'
  # cohorts in panel order.
  B <- 20
  set.seed(6)
  pdraws <- replicate(B, sample(seq_along(controls)), simplify = FALSE)
  pl <- lapply(pdraws, function(p) {
    kept <- length(controls) - length(treated_all)
    pseudo <- controls[p[(kept + 1):length(p)]]
    pseudo_cohort <- adoption[treated_all]
    vals <- lapply(seq_along(cohorts), function(k) {
      tr <- pseudo[pseudo_cohort == cohorts[k]]
      cohort_value(refit(k, controls[p[1:kept]], p[1:kept], tr, FALSE), length(tr), cohorts[k])
    })
    aggregate_cohorts(vals)
  })
  # Bootstrap: draw units from the whole panel; cohorts without a draw drop out.
  set.seed(7)
  bdraws <- list()
  while (length(bdraws) < B) {
    ind <- sample(1:N, replace = TRUE)
    if (any(is.na(adoption[ind])) && any(!is.na(adoption[ind]))) bdraws[[length(bdraws) + 1]] <- ind
  }
  bs <- lapply(bdraws, function(ind) {
    ind <- sort(ind)
    ctrl <- ind[is.na(adoption[ind])]
    pos <- match(ctrl, controls)
    vals <- list()
    for (k in seq_along(cohorts)) {
      tr <- ind[!is.na(adoption[ind]) & adoption[ind] == cohorts[k]]
      if (length(tr) == 0) next
      vals[[length(vals) + 1]] <- cohort_value(refit(k, ctrl, pos, tr, FALSE), length(tr), cohorts[k])
    }
    aggregate_cohorts(vals)
  })
  curve_se <- function(rs, se) {
    m <- t(sapply(rs, function(r) r$curve))
    apply(m, 2, function(col) se(col[is.finite(col)]))
  }
  stag[[name]] <- list(
    estimate = agg$estimate,
    event_time = agg$curve,
    cohorts = lapply(seq_along(cohorts), function(k) c(list(adoption = cohorts[k]), fit_info(fits[[k]]))),
    jackknife = list(replicates = u, se = jack_se(u), curve_se = apply(jk_curves, 2, jack_se)),
    placebo = list(draws = lapply(pdraws, function(d) d - 1L), replicates = sapply(pl, function(r) r$estimate),
                   se = replicate_se(sapply(pl, function(r) r$estimate)), curve_se = curve_se(pl, replicate_se)),
    bootstrap = list(draws = lapply(bdraws, function(d) d - 1L), replicates = sapply(bs, function(r) r$estimate),
                     se = replicate_se(sapply(bs, function(r) r$estimate)), curve_se = curve_se(bs, replicate_se))
  )
}
ref$staggered <- stag

write_json(ref, "tests/data/synthdid_ref.json", digits = 17, auto_unbox = TRUE, na = "null",
           pretty = FALSE)
