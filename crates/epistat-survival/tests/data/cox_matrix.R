# Cox fits on several covariates by survival::coxph: Breslow and Efron ties,
# delayed entry (counting-process Surv(start, stop, event)), strata, case
# weights, cluster-robust variance, exactly aliased columns and a coefficient
# that diverges. Times are integers, so events tie.
# For each fit: coefficients and standard errors (NA, written null, for an
# aliased column), the log partial likelihood at the fit and at zero, the
# iteration count, coxph's warnings, the model-based covariance matrix, the
# cluster-robust one where a cluster is used, the score residuals of each row
# with and without the case weight (all on the scale of the columns given, NA,
# written null, in the rows and columns of an aliased coefficient), and the
# condition number of the
# information matrix on the covariates scaled as coxph scales them (centred,
# divided by their weighted mean absolute deviation), which bounds how far two
# implementations of the same Newton iteration can differ.
# The versions of R, survival and jsonlite that wrote the file are in it.
# Numbers are written with 17 significant digits, so every double reads back
# as written.
# Run: Rscript cox_matrix.R > cox_matrix.json
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
# Eight censored rows share a covariate no event row has.
sep <- as.numeric(seq_len(n) %in% sample(which(event == 0), 8))
columns <- cbind(
  x1 = x1, x2 = x2, x3 = x3,
  sum12 = x1 + x2, zero = 0, constant = 1,
  near = x1 + x2 + 1e-4 * rnorm(n), sep = sep
)
d <- data.frame(start, stop, event, stratum, weight, id)

specs <- list(
  list("breslow_plain", c("x1", "x2", "x3"), "breslow", FALSE, FALSE, FALSE, FALSE),
  list("efron_plain", c("x1", "x2", "x3"), "efron", FALSE, FALSE, FALSE, FALSE),
  list("breslow_entry", c("x1", "x2", "x3"), "breslow", TRUE, FALSE, FALSE, FALSE),
  list("efron_entry", c("x1", "x2", "x3"), "efron", TRUE, FALSE, FALSE, FALSE),
  list("breslow_strata", c("x1", "x2", "x3"), "breslow", FALSE, TRUE, FALSE, FALSE),
  list("efron_strata", c("x1", "x2", "x3"), "efron", FALSE, TRUE, FALSE, FALSE),
  list("breslow_weights", c("x1", "x2", "x3"), "breslow", FALSE, FALSE, TRUE, FALSE),
  list("efron_weights", c("x1", "x2", "x3"), "efron", FALSE, FALSE, TRUE, FALSE),
  list("breslow_cluster", c("x1", "x2", "x3"), "breslow", FALSE, FALSE, FALSE, TRUE),
  list("efron_cluster", c("x1", "x2", "x3"), "efron", FALSE, FALSE, FALSE, TRUE),
  list("breslow_all", c("x1", "x2", "x3"), "breslow", TRUE, TRUE, TRUE, TRUE),
  list("efron_all", c("x1", "x2", "x3"), "efron", TRUE, TRUE, TRUE, TRUE),
  list("alias_sum", c("x1", "x2", "x3", "sum12"), "efron", TRUE, FALSE, TRUE, TRUE),
  list("alias_zero_constant", c("x1", "zero", "x3", "constant"), "breslow", FALSE, TRUE, FALSE, FALSE),
  list("near_alias", c("x1", "x2", "near"), "efron", FALSE, FALSE, FALSE, TRUE),
  list("monotone_efron", c("x1", "x3", "sep"), "efron", FALSE, FALSE, FALSE, FALSE),
  list("monotone_breslow_entry", c("x1", "x3", "sep"), "breslow", TRUE, TRUE, FALSE, FALSE)
)

fits <- list()
for (s in specs) {
  name <- s[[1]]; cols <- s[[2]]; ties <- s[[3]]
  use_start <- s[[4]]; use_strata <- s[[5]]; use_weights <- s[[6]]; use_cluster <- s[[7]]
  x <- columns[, cols, drop = FALSE]
  d$X <- x
  lhs <- if (use_start) "Surv(start, stop, event)" else "Surv(stop, event)"
  rhs <- if (use_strata) "X + strata(stratum)" else "X"
  args <- list(formula = as.formula(paste(lhs, "~", rhs)), data = d, ties = ties,
               control = coxph.control(timefix = FALSE))
  if (use_weights) args$weights <- d$weight
  if (use_cluster) args$cluster <- d$id
  # With case weights that are not integers coxph switches to the robust
  # variance on its own, so the model-based one is asked for.
  args$robust <- use_cluster
  warns <- character()
  fit <- withCallingHandlers(do.call(coxph, args), warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  ok <- !is.na(coef(fit))
  full <- function(values) {
    out <- rep(NA_real_, length(cols))
    out[ok] <- values
    out
  }
  vn <- if (use_cluster) fit$naive.var else fit$var
  vn <- as.matrix(vn)
  vr <- as.matrix(fit$var)
  # The covariance matrices may or may not carry rows for aliased columns.
  keep <- function(v) if (nrow(v) == length(ok)) v[ok, ok, drop = FALSE] else v
  # A covariance matrix over all columns, NA in the rows and columns of an
  # aliased one.
  embed <- function(v) {
    out <- matrix(NA_real_, length(ok), length(ok))
    out[ok, ok] <- v
    unname(out)
  }
  vn <- keep(vn)
  naive_se <- full(sqrt(diag(vn)))
  robust_se <- if (use_cluster) full(sqrt(diag(keep(vr)))) else numeric()
  w <- if (use_weights) d$weight else rep(1, n)
  xs <- x[, ok, drop = FALSE]
  centred <- sweep(xs, 2, colSums(w * xs) / sum(w))
  scale <- colSums(w * abs(centred)) / sum(w)
  scale[scale == 0] <- 1
  vs <- diag(scale, length(scale)) %*% vn %*% diag(scale, length(scale))
  monotone <- integer()
  for (m in warns) {
    if (grepl("may be infinite", m)) {
      monotone <- c(monotone, as.integer(strsplit(trimws(sub(";.*", "", sub(".*variable", "", m))), "[ ,]+")[[1]]))
    }
  }
  vcov_model <- embed(keep(as.matrix(if (use_cluster) fit$naive.var else vcov(fit))))
  vcov_robust <- if (use_cluster) embed(keep(as.matrix(vcov(fit)))) else list()
  # Over all columns when a coefficient is aliased, the aliased ones not
  # meaningful.
  # residuals() leaves the case weight out of a score residual unless asked;
  # the weighted ones add up to the score and are what the robust variance sums.
  score_columns <- function(weighted) {
    sc <- as.matrix(residuals(fit, type = "score", weighted = weighted))
    out <- matrix(NA_real_, n, length(ok))
    out[, ok] <- if (ncol(sc) == length(ok)) sc[, ok, drop = FALSE] else sc
    unname(out)
  }
  fits[[length(fits) + 1]] <- list(
    name = name, columns = cols, ties = ties,
    use_start = use_start, use_strata = use_strata,
    use_weights = use_weights, use_cluster = use_cluster,
    coefficients = unname(coef(fit)), se = naive_se, robust_se = robust_se,
    loglik = fit$loglik[2], loglik_null = fit$loglik[1], iterations = fit$iter,
    rank = sum(ok), warnings = warns, monotone_columns = as.list(monotone),
    information_condition = kappa(solve(vs), exact = TRUE),
    vcov = vcov_model, vcov_robust = vcov_robust, score_residuals = score_columns(FALSE),
    score_residuals_weighted = score_columns(TRUE)
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
