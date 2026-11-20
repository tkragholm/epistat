# Reference pair distances from MatchIt's distance functions.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_distances.R
#
# For each data set the script writes the covariates and, for each distance,
# the matrix of distances between every treated unit (rows, in data order) and
# every control (columns, in data order) as returned by `mahalanobis_dist()`,
# `robust_mahalanobis_dist()`, `euclidean_dist()` and `scaled_euclidean_dist()`.
#
# Data:
#   lalonde_sub   30 treated and 60 control units from `lalonde`, with the
#                 factor `race` (MatchIt codes a factor as one column for
#                 every level, each scaled by 1/sqrt(2)), two binary
#                 covariates and two with many ties and a long tail.
#   sim_ties      15 treated and 45 control units with a constant column, a
#                 column of integers with many ties, a binary column, a
#                 three-level factor and a normal column.
#
# Distances:
#   mahalanobis           the default: pooled within-group covariance
#   mahalanobis_full      `var` set to the full-sample covariance, with a
#                         factor given as dummy columns without its first level
#   *_discarded           the same with `discarded` set; the covariance and
#                         the ranks use the kept units only, so the matrix
#                         holds distances between kept units
#   robust, euclidean, scaled_euclidean
#
# `cond` is the condition number of the correlation matrix belonging to the
# covariance matrix the distance inverts as this
# crate builds it (levels of a factor but the first as columns for the
# Mahalanobis distance, the eigenvalues MatchIt's generalised inverse keeps
# for the rank distance). Rounding in a quadratic form in the inverse scales
# with the condition number after the columns are equilibrated, not with the
# covariance's, which carries the units of the columns. The tests derive their
# tolerance from it.

suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

cond_of <- function(m, tol = 1e-8) {
  ev <- eigen(cov2cor(m), symmetric = TRUE, only.values = TRUE)$values
  ev <- ev[ev > tol * max(ev)]
  max(ev) / min(ev)
}

drop_constant <- function(X) {
  X[, apply(X, 2, function(x) diff(range(x)) > sqrt(.Machine$double.eps)),
    drop = FALSE]
}

make_case <- function(name, f, df, discarded = NULL) {
  treat <- df$treat
  nd <- is.null(discarded)
  if (nd) discarded <- rep(FALSE, nrow(df))
  keep <- !discarded
  # Design as this crate builds it: dummy columns without the first level.
  Xd <- drop_constant(model.matrix(f, df)[, -1, drop = FALSE])
  Xk <- Xd[keep, , drop = FALSE]
  tk <- treat[keep]
  # Pooled within-group covariance, sums of squares over n - 2.
  Xc <- Xk
  for (g in 0:1) Xc[tk == g, ] <- sweep(Xk[tk == g, , drop = FALSE], 2,
                                         colMeans(Xk[tk == g, , drop = FALSE]))
  pooled <- crossprod(Xc) / (nrow(Xk) - 2)
  full <- cov(Xk)
  Xfull <- MatchIt:::get_covs_matrix_for_dist(f, df)
  Xfull <- drop_constant(Xfull)[keep, , drop = FALSE]
  Rk <- apply(Xfull, 2, rank)
  vr <- cov(Rk)
  mult <- sd(seq_len(nrow(Rk))) / sqrt(diag(vr))
  vr <- vr * outer(mult, mult)

  sel <- function(m) unname(m[keep[treat == 1], keep[treat == 0], drop = FALSE])
  d <- list()
  d$mahalanobis <- sel(mahalanobis_dist(f, df, discarded = discarded))
  # With `var` given MatchIt does not rescale the columns, and the covariance
  # of a factor's full set of level columns is singular, which its determinant
  # test does not catch at lalonde's scale. The factor is therefore given as
  # its dummy columns without the first level, and the distance does not depend
  # on which levels are left out.
  dfd <- data.frame(treat = treat, Xd)
  d$mahalanobis_full <- sel(mahalanobis_dist(
    treat ~ ., dfd, var = cov(Xd[keep, , drop = FALSE]), discarded = discarded))
  d$robust <- sel(robust_mahalanobis_dist(f, df, discarded = discarded))
  d$scaled_euclidean <- sel(scaled_euclidean_dist(f, df, discarded = discarded))
  if (nd) d$euclidean <- sel(euclidean_dist(f, df))
  num <- Filter(is.numeric, df[setdiff(names(df), "treat")])
  list(
    name = name,
    treat = treat[keep],
    numerics = lapply(num, function(x) x[keep]),
    factors = lapply(Filter(function(x) is.factor(x) || is.character(x),
                            df[setdiff(names(df), "treat")]),
                     function(x) as.character(x[keep])),
    distances = d,
    cond = list(mahalanobis = cond_of(pooled), mahalanobis_full = cond_of(full),
                robust = cond_of(vr)),
    n_treated = sum(tk == 1), n_control = sum(tk == 0))
}

data(lalonde)
set.seed(20261120)
rows <- c(sample(which(lalonde$treat == 1), 30), sample(which(lalonde$treat == 0), 60))
rows <- sort(rows)
lal <- lalonde[rows, c("treat", "age", "educ", "race", "married", "nodegree", "re74", "re75")]
rownames(lal) <- NULL
f_lal <- treat ~ age + educ + race + married + nodegree + re74 + re75

set.seed(7)
n <- 60
sim <- data.frame(
  treat = c(rep(1, 15), rep(0, 45)),
  x1 = round(rnorm(n), 6),
  x2 = sample(0:3, n, replace = TRUE),
  x3 = rbinom(n, 1, 0.3),
  k = 4,
  grp = factor(sample(c("a", "b", "c"), n, replace = TRUE, prob = c(0.2, 0.5, 0.3)))
)
sim$x1[1:15] <- sim$x1[1:15] + 0.5
f_sim <- treat ~ x1 + x2 + x3 + k + grp

disc <- rep(FALSE, nrow(lal)); disc[c(2, 5, 9, 33, 40, 51, 70, 85)] <- TRUE

cases <- list(
  make_case("lalonde_sub", f_lal, lal),
  make_case("lalonde_sub_discarded", f_lal, lal, discarded = disc),
  make_case("sim_ties", f_sim, sim)
)

# What MatchIt does with a constant column and with only constant columns.
const_only <- data.frame(treat = c(1, 1, 0, 0, 0), k = 3)
const_note <- tryCatch({
  m <- mahalanobis_dist(treat ~ k, const_only)
  list(error = FALSE, distances = as.vector(unname(m)))
}, error = function(e) list(error = TRUE, message = conditionMessage(e)))

# A column that is constant within each group but differs between them has no
# pooled within-group standard deviation.
between <- data.frame(treat = c(1, 1, 0, 0, 0), a = c(1, 2, 3, 1, 2),
                      b = c(5, 5, 9, 9, 9))
between_note <- suppressWarnings(list(
  scaled_euclidean = as.vector(unname(scaled_euclidean_dist(treat ~ a + b, between)))))

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  cases = cases,
  constant_only = const_note,
  constant_within_groups = between_note
)
write_json(out, "tests/data/matchit_distances.json", digits = NA, auto_unbox = TRUE,
           na = "string", pretty = FALSE)
cat(sprintf("%.0f KB\n", file.size("tests/data/matchit_distances.json") / 1024))
