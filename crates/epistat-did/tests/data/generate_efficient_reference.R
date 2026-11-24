# Reference values for epistat_did::estimate_att_gt_efficient and the
# aggregations of efficient ATT(g,t) estimates.
#
# The estimator is written out here from its formulas, those of the module
# docs of src/methods/efficient/mod.rs. For each cell (g, t) and baseline b < g
# it forms the weighted DiD theta_b on the units observed at t and at every
# baseline, its influence function psi_b over all N units (zero outside the
# cell), the weights a = Omega^{-1} 1 / sum(Omega^{-1} 1) with
# Omega = crossprod(psi) / N, and the cell's influence function psi a. An
# aggregate's standard error is that of the combined influence function. The
# root of the sum of squared weighted standard errors, which leaves out the
# covariance between cells, is written as se_naive for contrast.
#
# did::att_gt with base_period = "universal" estimates theta_{g-1}(g, t) and
# returns its influence function in the full sample, so on the balanced panel
# it checks theta_b at b = g - 1 for every cohort, and for cohort 2, whose only
# baseline is period 1, the efficient cell and its influence function too. The
# standard error of an equally weighted combination of cohort 2's cells is
# also formed from did's own influence functions.
#
# Cases:
#   balanced    300 units, periods 1 to 5, cohorts 2, 3, 4 and never treated,
#               sampling weights uniform on (0.5, 2), every unit observed in
#               every period. A unit-level shock in period 1 enters every
#               change Y_t - Y_1, so cells that share that baseline are
#               strongly correlated.
#   unbalanced  the same design with each (unit, period) row dropped with
#               probability 0.15, so the cells use different units.
#
# Run from the crate root:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/generate_efficient_reference.R
suppressMessages({
  library(did)
  library(jsonlite)
})
stopifnot(packageVersion("did") == "2.5.1", packageVersion("jsonlite") == "2.0.0")

simulate <- function(seed, missing_prob) {
  set.seed(seed)
  n <- 300
  periods <- 1:5
  g <- sample(c(2, 3, 4, 0), n, replace = TRUE, prob = c(0.2, 0.2, 0.2, 0.4))
  w <- runif(n, 0.5, 2)
  alpha <- rnorm(n, 0, 2)
  shock <- rnorm(n, 0, 1.5)
  lambda <- c(0, 0.3, 0.5, 0.9, 1.2)
  rows <- expand.grid(t = periods, id = seq_len(n))[, c("id", "t")]
  rows$g <- g[rows$id]
  rows$w <- w[rows$id]
  treated <- rows$g > 0 & rows$t >= rows$g
  effect <- ifelse(treated, 1 + 0.5 * (rows$t - rows$g) + 0.25 * (rows$g - 2), 0)
  rows$y <- alpha[rows$id] + lambda[rows$t] + effect + rnorm(nrow(rows)) +
    ifelse(rows$t == 1, shock[rows$id], 0)
  rows <- rows[runif(nrow(rows)) >= missing_prob, ]
  rows[order(rows$id, rows$t), ]
}

efficient <- function(rows) {
  ids <- sort(unique(rows$id))
  n <- length(ids)
  pos <- match(rows$id, ids)
  y <- matrix(NA_real_, n, max(rows$t))
  y[cbind(pos, rows$t)] <- rows$y
  cohort_of <- numeric(n)
  cohort_of[pos] <- rows$g
  w <- numeric(n)
  w[pos] <- rows$w
  cells <- list()
  for (g in sort(unique(cohort_of[cohort_of > 0]))) {
    cohort <- cohort_of == g
    seen <- which(colSums(!is.na(y[cohort, , drop = FALSE])) > 0)
    pre <- seen[seen < g]
    for (t in seen[seen >= g]) {
      complete <- !is.na(y[, t]) & rowSums(is.na(y[, pre, drop = FALSE])) == 0
      tr <- complete & cohort
      co <- complete & cohort_of == 0
      psi <- matrix(0, n, length(pre))
      theta <- numeric(length(pre))
      for (j in seq_along(pre)) {
        d <- y[, t] - y[, pre[j]]
        wt <- sum(w[tr])
        wc <- sum(w[co])
        mt <- sum(w[tr] * d[tr]) / wt
        mc <- sum(w[co] * d[co]) / wc
        theta[j] <- mt - mc
        psi[tr, j] <- n * w[tr] * (d[tr] - mt) / wt
        psi[co, j] <- -n * w[co] * (d[co] - mc) / wc
      }
      omega <- crossprod(psi) / n
      v <- solve(omega, rep(1, length(pre)))
      a <- v / sum(v)
      inf <- drop(psi %*% a)
      cells[[length(cells) + 1]] <- list(
        group = g, time = t, att = sum(a * theta),
        se = sqrt(sum(inf^2) / n) / sqrt(n),
        treated_n = sum(tr), control_n = sum(co),
        total_weight = sum(w[tr]) + sum(w[co]),
        baselines = pre, baseline_att = theta, baseline_weight = a,
        kappa = kappa(omega, exact = TRUE),
        influence = inf
      )
    }
  }
  list(ids = ids, n = n, cells = cells)
}

aggregate_cells <- function(est, kind, weighting, keep_influence) {
  key_of <- switch(kind,
    event_time = function(cell) cell$time - cell$group,
    cohort = function(cell) cell$group,
    calendar = function(cell) cell$time,
    overall = function(cell) 0
  )
  keys <- vapply(est$cells, key_of, numeric(1))
  lapply(sort(unique(keys)), function(key) {
    members <- est$cells[keys == key]
    raw <- vapply(members, function(cell) switch(weighting,
      equal = 1,
      treated_count = cell$treated_n,
      total_weight = cell$total_weight
    ), numeric(1))
    c_k <- raw / sum(raw)
    inf <- Reduce(`+`, Map(function(cell, ck) ck * cell$influence, members, c_k))
    se_k <- vapply(members, function(cell) cell$se, numeric(1))
    out <- list(
      kind = kind, weighting = weighting, key = key,
      estimate = sum(c_k * vapply(members, function(cell) cell$att, numeric(1))),
      se = sqrt(sum(inf^2) / est$n) / sqrt(est$n),
      se_naive = sqrt(sum(c_k^2 * se_k^2)),
      components = length(members)
    )
    if (keep_influence) out$influence <- inf
    out
  })
}

aggregates <- function(est) {
  out <- list()
  for (kind in c("event_time", "cohort", "calendar", "overall")) {
    for (weighting in c("equal", "treated_count", "total_weight")) {
      keep <- kind == "event_time" && weighting == "equal"
      out <- c(out, aggregate_cells(est, kind, weighting, keep))
    }
  }
  out
}

# did reorders units by cohort, so the rows of inffunc are matched to ids
# through the data did kept (see generate_cs_panel_reference.R).
did_reference <- function(rows, est) {
  out <- att_gt(
    yname = "y", tname = "t", idname = "id", gname = "g", data = rows,
    panel = TRUE, control_group = "nevertreated", base_period = "universal",
    est_method = "dr", weightsname = "w", bstrap = FALSE, cband = FALSE
  )
  dp <- out$DIDparams
  order <- as.numeric(unique(dp$data[[dp$idname]]))
  inffunc <- as.matrix(out$inffunc)[match(est$ids, order), , drop = FALSE]
  post <- which(out$t >= out$group)
  cells <- lapply(post, function(k) list(
    group = out$group[k], time = out$t[k], att = out$att[k], se = out$se[k],
    influence = inffunc[, k]
  ))
  cohort2 <- post[out$group[post] == 2]
  combined <- drop(inffunc[, cohort2, drop = FALSE] %*% rep(1 / length(cohort2), length(cohort2)))
  list(
    cells = cells,
    cohort2_equal_estimate = mean(out$att[cohort2]),
    cohort2_equal_se = sqrt(sum(combined^2) / est$n) / sqrt(est$n)
  )
}

make_case <- function(name, rows, with_did) {
  est <- efficient(rows)
  case <- list(
    name = name,
    rows = list(id = rows$id, g = rows$g, t = rows$t, y = rows$y, w = rows$w),
    ids = est$ids,
    cells = lapply(est$cells, function(cell) cell),
    aggregates = aggregates(est)
  )
  if (with_did) case$did <- did_reference(rows, est)
  case
}

ref <- list(
  versions = list(
    r = R.version.string,
    did = as.character(packageVersion("did")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  cases = list(
    make_case("balanced", simulate(20261124, 0), TRUE),
    make_case("unbalanced", simulate(20261125, 0.15), FALSE)
  )
)
write_json(ref, "tests/data/efficient_ref.json", digits = I(17), auto_unbox = TRUE)
cat("tests/data/efficient_ref.json written\n")
