# Residuals and the test of proportional hazards of Cox fits by
# survival::coxph, residuals.coxph and cox.zph, on the lung data and on the
# counting-process data of cox_matrix.R (same seed, same columns). Times are
# integers, so events tie. Both tie rules, delayed entry, strata, case weights,
# a cluster, an aliased column and a factor with two columns as one term.
#
# For each fit: the coefficients, the model-based covariance matrix (NA written
# null for an aliased column), the condition number of the information matrix on
# the scaled covariates, and residuals() of every type, weighted as the default
# and the other way, with collapse by the cluster (when there is one) and by a
# vector of codes. The Schoenfeld residuals come with the row of each event,
# which is the order of residuals.coxph. For each fit and each transform
# of cox.zph: the table, the transformed times, the times, the strata, the scaled
# residuals y, the variance, the row of each event (cox.zph orders tied events of
# data with delayed entry by descending row, which the row index lets the test
# undo), and the condition number of the information matrix of the extended
# model that the test inverts, on its scaled form. For the factor term also
# singledf = TRUE.
# The versions of R, survival and jsonlite are in the file. Numbers are written
# with 17 significant digits.
# Run: Rscript cox_residuals.R > cox_residuals.json
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
# Drawn after everything of cox_matrix.R, so that the columns above are its.
fac <- factor(sample(1:3, n, replace = TRUE))
fdummy <- model.matrix(~ fac)[, -1]
colnames(fdummy) <- c("f2", "f3")
syn <- data.frame(start, stop, event, stratum, weight, id, columns, fdummy, fac)
syn$grp <- (seq_len(n) - 1) %% 7

lc <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
lc$event <- as.numeric(lc$status == 2)
lc$stratum <- lc$sex
lc$sexf <- lc$sex - 1
lc$weight <- 1
lc$grp <- (seq_len(nrow(lc)) - 1) %% 7

nan_null <- function(m) unname(m)
mat <- function(r) {
  m <- as.matrix(r)
  m <- unname(m)
  m
}

# name, data, rhs, columns (names of the matrix columns given to the Rust fit),
# ties, entry, strata, weights, cluster, terms (zero based column groups or NULL)
specs <- list(
  list("lung_efron", "lung", "age + sexf + ph.ecog", c("age", "sexf", "ph.ecog"), "efron", FALSE, FALSE, FALSE, FALSE, NULL),
  list("lung_breslow", "lung", "age + sexf + ph.ecog", c("age", "sexf", "ph.ecog"), "breslow", FALSE, FALSE, FALSE, FALSE, NULL),
  list("lung_strata_efron", "lung", "age + ph.ecog + strata(stratum)", c("age", "ph.ecog"), "efron", FALSE, TRUE, FALSE, FALSE, NULL),
  list("lung_strata_breslow", "lung", "age + ph.ecog + strata(stratum)", c("age", "ph.ecog"), "breslow", FALSE, TRUE, FALSE, FALSE, NULL),
  list("syn_efron_plain", "syn", "x1 + x2 + x3", c("x1", "x2", "x3"), "efron", FALSE, FALSE, FALSE, FALSE, NULL),
  list("syn_breslow_plain", "syn", "x1 + x2 + x3", c("x1", "x2", "x3"), "breslow", FALSE, FALSE, FALSE, FALSE, NULL),
  list("syn_efron_entry", "syn", "x1 + x2 + x3", c("x1", "x2", "x3"), "efron", TRUE, FALSE, FALSE, FALSE, NULL),
  list("syn_breslow_entry", "syn", "x1 + x2 + x3", c("x1", "x2", "x3"), "breslow", TRUE, FALSE, FALSE, FALSE, NULL),
  list("syn_efron_strata", "syn", "x1 + x2 + x3 + strata(stratum)", c("x1", "x2", "x3"), "efron", FALSE, TRUE, FALSE, FALSE, NULL),
  list("syn_breslow_weights", "syn", "x1 + x2 + x3", c("x1", "x2", "x3"), "breslow", FALSE, FALSE, TRUE, FALSE, NULL),
  list("syn_efron_weights", "syn", "x1 + x2 + x3", c("x1", "x2", "x3"), "efron", FALSE, FALSE, TRUE, FALSE, NULL),
  list("syn_efron_cluster", "syn", "x1 + x2 + x3", c("x1", "x2", "x3"), "efron", FALSE, FALSE, FALSE, TRUE, NULL),
  list("syn_efron_all", "syn", "x1 + x2 + x3 + strata(stratum)", c("x1", "x2", "x3"), "efron", TRUE, TRUE, TRUE, TRUE, NULL),
  list("syn_breslow_all", "syn", "x1 + x2 + x3 + strata(stratum)", c("x1", "x2", "x3"), "breslow", TRUE, TRUE, TRUE, TRUE, NULL),
  list("syn_efron_factor", "syn", "x1 + fac", c("x1", "f2", "f3"), "efron", FALSE, FALSE, FALSE, FALSE, list(I(0), I(c(1, 2)))),
  list("syn_breslow_factor_all", "syn", "x1 + fac + strata(stratum)", c("x1", "f2", "f3"), "breslow", TRUE, TRUE, TRUE, FALSE, list(I(0), I(c(1, 2)))),
  list("syn_alias", "syn", "x1 + x2 + x3 + sum12", c("x1", "x2", "x3", "sum12"), "efron", TRUE, FALSE, TRUE, TRUE, NULL)
)

# Capture the information matrix that cox.zph passes back from C.
zph_probe <- cox.zph
captured <- new.env()
environment(zph_probe) <- list2env(
  list(.Call = function(...) {
    r <- base::.Call(...)
    if (is.list(r) && !is.null(r$imat)) assign("imat", r$imat, captured)
    r
  }),
  parent = environment(cox.zph)
)

fits <- list()
for (s in specs) {
  name <- s[[1]]; which_data <- s[[2]]; rhs <- s[[3]]; cols <- s[[4]]
  ties <- s[[5]]; use_start <- s[[6]]; use_strata <- s[[7]]
  use_weights <- s[[8]]; use_cluster <- s[[9]]; terms_zero <- s[[10]]
  d <- if (which_data == "lung") lc else syn
  m <- nrow(d)
  lhs <- if (which_data == "lung") "Surv(time, event)" else if (use_start) "Surv(start, stop, event)" else "Surv(stop, event)"
  args <- list(formula = as.formula(paste(lhs, "~", rhs)), data = d, ties = ties)
  if (use_weights) args$weights <- d$weight
  if (use_cluster) args$cluster <- d$id
  args$robust <- use_cluster
  fit <- do.call(coxph, args)
  ok <- !is.na(coef(fit))
  vn <- if (use_cluster) fit$naive.var else fit$var
  vn <- as.matrix(vn)
  embed <- function(v) {
    out <- matrix(NA_real_, length(ok), length(ok))
    out[ok, ok] <- if (nrow(v) == length(ok)) v[ok, ok, drop = FALSE] else v
    unname(out)
  }
  xm <- as.matrix(d[, cols, drop = FALSE])
  w <- if (use_weights) d$weight else rep(1, m)
  xs <- xm[, ok, drop = FALSE]
  centred <- sweep(xs, 2, colSums(w * xs) / sum(w))
  scl <- colSums(w * abs(centred)) / sum(w)
  scl[scl == 0] <- 1
  vnk <- if (nrow(vn) == length(ok)) vn[ok, ok, drop = FALSE] else vn
  vs <- diag(scl, length(scl)) %*% vnk %*% diag(scl, length(scl))
  kappa_info <- kappa(solve(vs), exact = TRUE)

  # Columns of residuals over all columns of the design, NA for aliased ones.
  full <- function(r) {
    r <- as.matrix(r)
    if (ncol(r) == length(ok)) return(unname(r))
    out <- matrix(NA_real_, nrow(r), length(ok))
    out[, ok] <- r
    unname(out)
  }
  res <- list()
  res$martingale <- as.vector(residuals(fit, "martingale"))
  res$martingale_weighted <- as.vector(residuals(fit, "martingale", weighted = TRUE))
  res$deviance <- as.vector(residuals(fit, "deviance"))
  res$score <- full(residuals(fit, "score"))
  res$score_weighted <- full(residuals(fit, "score", weighted = TRUE))
  res$dfbeta <- full(residuals(fit, "dfbeta"))
  res$dfbeta_unweighted <- full(residuals(fit, "dfbeta", weighted = FALSE))
  res$dfbetas <- full(residuals(fit, "dfbetas"))
  sch <- residuals(fit, "schoenfeld")
  res$schoenfeld <- full(sch)
  res$schoenfeld_weighted <- full(residuals(fit, "schoenfeld", weighted = TRUE))
  res$schoenfeld_time <- as.numeric(rownames(as.matrix(sch)))
  if (is.null(rownames(as.matrix(sch)))) res$schoenfeld_time <- as.numeric(names(sch))
  if (!any(!ok)) res$scaledsch <- full(residuals(fit, "scaledsch"))
  # The row of each event in the order of residuals.coxph.
  y <- if (which_data == "lung") cbind(d$time, d$event) else if (use_start) cbind(d$start, d$stop, d$event) else cbind(d$stop, d$event)
  ny <- ncol(y)
  status <- y[, ny]
  istrat <- if (use_strata) as.integer(factor(d$stratum)) else integer(m)
  ord <- order(istrat, y[, ny - 1], -status)
  res$schoenfeld_row <- ord[status[ord] == 1] - 1L
  res$schoenfeld_stratum <- if (use_strata) d$stratum[ord[status[ord] == 1]] else integer()
  # Collapse by a vector of codes, and by the cluster.
  for (type in c("martingale", "deviance", "score", "dfbeta", "dfbetas")) {
    res[[paste0(type, "_collapse_codes")]] <- full(as.matrix(residuals(fit, type, collapse = d$grp)))
    if (use_cluster) {
      # collapse = TRUE fails for type = "martingale", which never fetches the
      # cluster and finds the function cluster() instead, so the cluster is
      # passed for it.
      by <- if (type == "martingale") d$id else TRUE
      res[[paste0(type, "_collapse_cluster")]] <- full(as.matrix(residuals(fit, type, collapse = by)))
    }
  }
  # Deviance residuals of groups use the status summed over the group, and
  # R's formula gives NaN, written null, where that is 2 or more.
  res$status_collapse_codes <- as.vector(rowsum(status, d$grp))
  if (use_cluster) res$status_collapse_cluster <- as.vector(rowsum(status, d$id))
  res$martingale_weighted_collapse_codes <- full(as.matrix(residuals(fit, "martingale", weighted = TRUE, collapse = d$grp)))
  res$score_weighted_collapse_codes <- full(as.matrix(residuals(fit, "score", weighted = TRUE, collapse = d$grp)))

  zph <- list()
  for (transform in c("km", "rank", "identity", "log")) {
    for (single in c(FALSE, TRUE)) {
      if (single && is.null(terms_zero)) next
      z <- zph_probe(fit, transform = transform, singledf = single)
      imat <- captured$imat
      sc <- sqrt(diag(imat))
      ev <- if (ny == 2) status == 1 else status == 1
      indx <- if (ny == 2) order(istrat, y[, 1]) else rev(order(-istrat, -y[, 2]))
      indx <- indx[ev[indx]]
      zph[[length(zph) + 1]] <- list(
        transform = transform, singledf = single,
        table = unname(as.matrix(z$table)), x = z$x, time = z$time,
        stratum = if (use_strata) as.integer(z$strata) else integer(),
        row = indx - 1L,
        y = unname(as.matrix(z$y)), var = unname(as.matrix(z$var)),
        kappa = kappa(diag(1 / sc, length(sc)) %*% imat %*% diag(1 / sc, length(sc)), exact = TRUE)
      )
    }
  }
  fits[[length(fits) + 1]] <- list(
    name = name, data = which_data, columns = cols, ties = ties,
    use_start = use_start, use_strata = use_strata,
    use_weights = use_weights, use_cluster = use_cluster,
    terms = terms_zero,
    coefficients = unname(coef(fit)), vcov = embed(vn),
    information_condition = kappa_info,
    residuals = res, zph = zph
  )
}
datasets <- list(
  lung = list(
    stop = lc$time, event = lc$event, stratum = lc$stratum, weight = lc$weight,
    id = seq_len(nrow(lc)), grp = lc$grp,
    columns = list(age = lc$age, sexf = lc$sexf, ph.ecog = lc$ph.ecog)
  ),
  syn = list(
    start = start, stop = stop, event = event, stratum = stratum, weight = weight,
    id = id, grp = syn$grp,
    columns = as.list(as.data.frame(cbind(columns, fdummy)))
  )
)
cat(toJSON(list(
  versions = list(
    r = R.version.string,
    survival = as.character(packageVersion("survival")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  datasets = datasets, fits = fits
), digits = I(17), auto_unbox = TRUE, na = "null", null = "null"))
