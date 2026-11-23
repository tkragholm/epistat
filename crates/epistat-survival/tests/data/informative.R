# Reference values for epistat_survival::informative: the informative sets of
# matched case-control data and of a stratified Cox model, counted in base R
# with tapply.
#
# 150 sets of 2 to 6 rows with 0 to 2 cases each (so some have no case, some
# only cases), a binary exposure that is constant in some sets, and a
# continuous covariate that is constant within a third of the sets. The set
# labels are not 0..n-1. A second data set has weights, some of them 0, and
# survival times with events, for the stratified Cox count.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/informative.R
suppressMessages(library(jsonlite))
set.seed(20261123)

make <- function(n_sets, weights = FALSE) {
  size <- sample(2:6, n_sets, replace = TRUE)
  set <- rep(seq_len(n_sets) * 7L + 11L, size)
  n <- length(set)
  case <- unlist(lapply(size, function(s) {
    k <- sample(0:min(2, s), 1)
    sample(c(rep(TRUE, k), rep(FALSE, s - k)))
  }))
  flat <- seq_len(n_sets) %% 5 == 0
  x1 <- unlist(lapply(seq_len(n_sets), function(g) {
    s <- size[g]
    if (flat[g]) rep(rbinom(1, 1, 0.5), s) else rbinom(s, 1, 0.5)
  }))
  const_x2 <- seq_len(n_sets) %% 3 == 0
  x2 <- unlist(lapply(seq_len(n_sets), function(g) {
    s <- size[g]
    if (const_x2[g]) rep(round(rnorm(1), 2), s) else round(rnorm(s), 2)
  }))
  d <- list(set = as.integer(set), case = case, x = cbind(x1, x2))
  if (weights) d$weight <- ifelse(runif(n) < 0.15, 0, 1 + rbinom(n, 2, 0.5))
  d
}

count <- function(d, case) {
  use <- if (is.null(d$weight)) rep(TRUE, length(d$set)) else d$weight > 0
  set <- d$set[use]; cs <- case[use]; x <- d$x[use, , drop = FALSE]
  labels <- unique(set)   # order of first appearance
  rows <- vapply(labels, function(l) sum(set == l), 1L)
  cases <- vapply(labels, function(l) sum(cs[set == l]), 1L)
  varies_col <- sapply(seq_len(ncol(x)), function(j)
    vapply(labels, function(l) length(unique(x[set == l, j])) > 1, TRUE))
  varies <- apply(varies_col, 1, any)
  both <- cases > 0 & cases < rows
  inf <- both & varies
  list(labels = labels, rows = rows, cases = cases, varies = varies,
       n_sets = length(labels), n_informative_sets = sum(inf),
       n_rows = sum(rows), n_informative_rows = sum(rows[inf]),
       n_informative_cases = sum(cases[inf]),
       by_column = colSums(varies_col & both))
}

a <- make(150)
b <- make(120, weights = TRUE)
b$time <- round(rexp(length(b$set)), 3)
b$status <- as.integer(b$case)
out <- list(
  versions = list(r = R.version.string),
  clogit = c(a, list(expected = count(a, a$case))),
  cox = c(b, list(expected = count(b, b$status > 0)))
)
write(toJSON(out, auto_unbox = TRUE, digits = 17, null = "null"),
      "tests/data/informative.json")
