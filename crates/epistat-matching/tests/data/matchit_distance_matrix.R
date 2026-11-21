# Pairs of MatchIt 4.8.2's method = "nearest" when `distance` is a matrix of
# treated-by-control distances.
#
# The data are simulated with a fixed seed, in random row order. The matrix is
# |dx| + 0.3 |dy| over two covariates, which are continuous, so no two pairs of
# one treated unit share a distance and MatchIt's tie rule does not matter. A
# second matrix has entries set to Inf, which MatchIt takes as forbidden pairs,
# for some treated units every entry.
#
# For every run the file holds, per focal unit in row order, the rows of the
# units matched to it in order of assignment (match.matrix). For the ATC runs
# the focal units are the controls and MatchIt reads the same matrix
# transposed; the test builds the transpose itself.
#
# `caliper_exact` is the distance of the fifth nearest pair, used as a caliper
# in the runs that have one, so a pair sits exactly on the caliper and the test
# shows that a pair at the caliper is kept.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_distance_matrix.R
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

set.seed(20261121)
n <- 90
d <- data.frame(treat = rbinom(n, 1, 0.3), x = rnorm(n), y = rnorm(n))
tr <- which(d$treat == 1)
co <- which(d$treat == 0)
dm <- abs(outer(d$x[tr], d$x[co], "-")) + 0.3 * abs(outer(d$y[tr], d$y[co], "-"))
dimnames(dm) <- list(tr, co)
dm_inf <- dm
dm_inf[sample(length(dm_inf), 400)] <- Inf
dm_inf[c(2, 5), ] <- Inf
caliper_exact <- unname(sort(apply(dm, 1, min))[5])

mm_rows <- function(m) {
  mm <- m$match.matrix
  lapply(seq_len(nrow(mm)), function(i) as.integer(mm[i, !is.na(mm[i, ])]))
}

runs <- list()
add <- function(name, matrix, estimand = "ATT", ratio = 1, replace = FALSE,
                order = NULL, caliper = NULL) {
  args <- list(treat ~ x + y, data = d, distance = matrix, estimand = estimand,
               ratio = ratio, replace = replace)
  if (!is.null(order)) args$m.order <- order
  if (!is.null(caliper)) {
    args$caliper <- caliper
    args$std.caliper <- FALSE
  }
  m <- suppressWarnings(do.call(matchit, args))
  runs[[length(runs) + 1]] <<- list(
    name = name, matrix = matrix_name(matrix), estimand = estimand,
    ratio = ratio, replace = replace,
    order = if (is.null(order)) NA else order,
    caliper = if (is.null(caliper)) NA else caliper,
    matchit = mm_rows(m))
}
matrix_name <- function(m) if (identical(m, dm)) "dist" else "inf"

add("att_default", dm)
add("att_ratio2", dm, ratio = 2)
add("att_ratio3_replace", dm, ratio = 3, replace = TRUE)
add("att_closest", dm, order = "closest")
add("att_farthest", dm, order = "farthest")
add("att_closest_ratio2", dm, order = "closest", ratio = 2)
add("att_farthest_ratio2", dm, order = "farthest", ratio = 2)
add("att_caliper", dm, caliper = caliper_exact)
add("att_caliper_ratio2", dm, caliper = caliper_exact * 3, ratio = 2)
add("atc_default", dm, estimand = "ATC")
add("atc_ratio2", dm, estimand = "ATC", ratio = 2)
add("inf_default", dm_inf)
add("inf_ratio2", dm_inf, ratio = 2)
add("inf_closest", dm_inf, order = "closest")

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  treat = d$treat,
  rows_treated = tr, rows_control = co,
  dist = lapply(seq_len(nrow(dm)), function(i) unname(dm[i, ])),
  inf = lapply(seq_len(nrow(dm_inf)), function(i) {
    v <- unname(dm_inf[i, ]); ifelse(is.infinite(v), NA, v)
  }),
  caliper_exact = caliper_exact,
  runs = runs)
write_json(out, "tests/data/matchit_distance_matrix.json", digits = I(17),
           auto_unbox = TRUE, na = "null")
