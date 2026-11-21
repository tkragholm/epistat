# Coarsened exact matching of MatchIt 4.8.2's matchit(method = "cem"), with and
# without k2k.
#
# Data sets (the units are in random order, so MatchIt's numbering of
# subclasses follows the group of the first row):
#   lalonde      a sample of 400 rows of lalonde with the numeric variables
#                age, educ, re74, re75 and the text variables race, married,
#                nodegree. educ_text is educ as text, for grouping.
#   lalonde_u    the same with age and re74 moved by 1e-7 and 1e-6 times the
#                row number, so that no two units of a stratum are at the same
#                distance and the k2k pairs do not depend on a tie rule.
#   tenths       one numeric variable x of the multiples of 0.1 from 0 to 1 and
#                a binary text variable yt. With 10 bins the cut points are
#                k * 0.1, and 3 * 0.1 is above 0.3 in floating point, so the
#                values 0.3 and 0.6 lie below the cut point that looks equal.
#   integers     the integers 0 to 8, which lie on equal-width cut points for
#                2, 4 and 8 bins.
#
# Every run holds the arguments given to matchit() in a form the Rust test
# reads, and what matchit() returned: the subclass of every unit (null for a
# unit left out), the weights, and for k2k the match matrix (the row of the
# control of each treated unit in row order, null where there is none).
#
# `cuts`, `groups` and `k2k` are written as follows. A binning is one of
#   {"kind": "rule", "rule": "sturges" | "fd" | "scott"}, {"kind": "bins",
#   "n": n}, {"kind": "cuts", "values": [...]}, {"kind": "quantiles", "n": q},
#   {"kind": "exact"}.
# `cut_all` is passed to matchit() as `cutpoints` for every numeric variable;
# without it the per-variable `cut` goes in a list and a numeric variable with
# no entry takes MatchIt's default, Sturges.
#
# The "q2" run reproduces a defect, see `q2_defect` in the output: MatchIt reads
# the one quantile that "q2" gives as a number of bins.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_cem.R
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

data(lalonde, package = "MatchIt")
set.seed(20261121)
lal <- lalonde[sample(nrow(lalonde), 400), ]
lal$race <- as.character(lal$race)
lal$married <- as.character(lal$married)
lal$nodegree <- as.character(lal$nodegree)
lal$educ_text <- as.character(lal$educ)
rownames(lal) <- NULL
lal_u <- lal
lal_u$age <- lal_u$age + 1e-7 * seq_len(nrow(lal_u))
lal_u$re74 <- lal_u$re74 + 1e-6 * seq_len(nrow(lal_u))

# Multiples of 0.1 written as k / 10, the double nearest each, and integers.
set.seed(20261122)
boundary_set <- function(x) {
  d <- data.frame(treat = rbinom(length(x), 1, 0.45), x = x)
  d$yt <- as.character(rep(0:1, length.out = length(x)))
  d <- d[sample(nrow(d)), ]
  rownames(d) <- NULL
  d
}
tenths <- boundary_set(rep(0:10 / 10, 36))
integers <- boundary_set(rep(0:8, 44))

datasets <- list(lalonde = lal, lalonde_u = lal_u, tenths = tenths, integers = integers)

rule <- function(r) list(kind = "rule", rule = r)
bins <- function(n) list(kind = "bins", n = n)
cuts <- function(...) list(kind = "cuts", values = I(c(...)))
quant <- function(n) list(kind = "quantiles", n = n)
exact <- list(kind = "exact")
num <- function(name, cut = NULL) list(name = name, type = "numeric", cut = cut)
cat_var <- function(name, groups = NULL) {
  list(name = name, type = "categorical",
       groups = if (is.null(groups)) NULL else lapply(groups, function(g) I(as.character(g))))
}

to_r <- function(cut) {
  switch(cut$kind,
    rule = cut$rule, bins = cut$n, cuts = unclass(cut$values),
    quantiles = paste0("q", cut$n), exact = 0)
}

run_one <- function(name, dataset, vars, estimand = "ATT", cut_all = NULL,
                    k2k = NULL) {
  d <- datasets[[dataset]]
  f <- as.formula(paste("treat ~", paste(vapply(vars, `[[`, "", "name"), collapse = " + ")))
  args <- list(f, data = d, method = "cem", estimand = estimand)
  numeric_vars <- Filter(function(v) v$type == "numeric", vars)
  if (!is.null(cut_all)) {
    args$cutpoints <- to_r(cut_all)
  } else if (length(numeric_vars) > 0) {
    cp <- list()
    for (v in numeric_vars) if (!is.null(v$cut)) cp[[v$name]] <- to_r(v$cut)
    if (length(cp) > 0) args$cutpoints <- cp
  }
  grouped <- Filter(function(v) v$type == "categorical" && !is.null(v$groups), vars)
  if (length(grouped) > 0) {
    args$grouping <- setNames(
      lapply(grouped, function(v) lapply(v$groups, as.character)),
      vapply(grouped, `[[`, "", "name"))
  }
  if (!is.null(k2k)) {
    args$k2k <- TRUE
    args$k2k.method <- k2k$method
    if (!is.null(k2k$order)) args$m.order <- k2k$order
  }
  m <- tryCatch(suppressWarnings(do.call(matchit, args)),
                error = function(e) conditionMessage(e))
  res <- list(name = name, dataset = dataset, estimand = estimand,
              vars = vars, cut_all = cut_all, k2k = k2k)
  if (is.character(m)) {
    res$error <- m
    return(res)
  }
  res$subclass <- ifelse(is.na(m$subclass), NA, as.integer(m$subclass))
  res$weights <- unname(m$weights)
  if (!is.null(k2k)) {
    mm <- m$match.matrix
    res$match_rows <- lapply(seq_len(nrow(mm)), function(i)
      if (is.na(mm[i, 1])) NULL else as.integer(mm[i, 1]))
    res$focal_rows <- as.integer(rownames(mm))
  }
  res
}

lalonde_vars <- list(num("age"), num("educ"), cat_var("race"), cat_var("married"), num("re74"))
simple <- list(num("age"), num("educ"), cat_var("race"), cat_var("married"))
numeric_only <- list(num("age"), num("educ"), num("re74"), num("re75"))

runs <- list(
  run_one("default", "lalonde", lalonde_vars),
  run_one("default_atc", "lalonde", lalonde_vars, "ATC"),
  run_one("default_ate", "lalonde", lalonde_vars, "ATE"),
  run_one("fd", "lalonde", lalonde_vars, cut_all = rule("fd")),
  run_one("scott", "lalonde", lalonde_vars, cut_all = rule("scott")),
  run_one("bins5", "lalonde", lalonde_vars, cut_all = bins(5)),
  run_one("bins1_drops", "lalonde", simple, cut_all = bins(1)),
  run_one("explicit", "lalonde",
          list(num("age", cuts(25, 30, 35)), num("educ", bins(4)),
               cat_var("race"), cat_var("married"), num("re74", cuts(0, 5000, 10000))),
          "ATT"),
  run_one("explicit_unsorted", "lalonde",
          list(num("age", cuts(35, 25, 30, 25)), num("educ", cuts(12, 11)),
               cat_var("race")), "ATT"),
  run_one("exact_numeric", "lalonde",
          list(num("age", exact), num("educ", exact), cat_var("married")), "ATT"),
  run_one("q3", "lalonde", simple, cut_all = quant(3)),
  run_one("q4", "lalonde", simple, cut_all = quant(4)),
  run_one("q5_ate", "lalonde", simple, "ATE", cut_all = quant(5)),
  run_one("q2", "lalonde", simple, cut_all = quant(2)),
  run_one("grouping", "lalonde",
          list(num("age", bins(4)), cat_var("race", list(c("black", "hispan"))),
               cat_var("educ_text", list(c("9", "10", "11"), c("12", "13", "14"))),
               cat_var("married")), "ATT"),
  run_one("grouping_chain", "lalonde",
          list(num("age", bins(3)),
               cat_var("educ_text", list(c("9", "10"), c("11", "9", "12")))), "ATT"),
  run_one("boundary_tenths_bins10", "tenths", list(num("x", bins(10)), cat_var("yt"))),
  run_one("boundary_tenths_bins5", "tenths", list(num("x", bins(5)), cat_var("yt"))),
  run_one("boundary_tenths_cuts", "tenths", list(num("x", cuts(0.3, 0.6)), cat_var("yt"))),
  run_one("boundary_ints_bins4", "integers", list(num("x", bins(4)), cat_var("yt"))),
  run_one("boundary_ints_bins8", "integers", list(num("x", bins(8)))),
  run_one("boundary_ints_bins3", "integers", list(num("x", bins(3)), cat_var("yt"))),
  run_one("boundary_ints_cuts", "integers", list(num("x", cuts(2, 5)), cat_var("yt"))),
  run_one("boundary_ints_q4", "integers", list(num("x", quant(4)), cat_var("yt"))),
  run_one("boundary_ints_sturges", "integers", list(num("x"), cat_var("yt"))),
  run_one("boundary_ints_fd", "integers", list(num("x", rule("fd")), cat_var("yt"))),
  # k2k
  run_one("k2k_mahalanobis", "lalonde_u", numeric_only, k2k = list(method = "mahalanobis")),
  run_one("k2k_euclidean", "lalonde_u", numeric_only, k2k = list(method = "euclidean")),
  run_one("k2k_scaled_euclidean", "lalonde_u", numeric_only,
          k2k = list(method = "scaled_euclidean")),
  run_one("k2k_mahalanobis_race", "lalonde_u", list(num("age"), num("educ"), cat_var("race"), num("re74")),
          k2k = list(method = "mahalanobis")),
  run_one("k2k_mahalanobis_closest", "lalonde_u", numeric_only,
          k2k = list(method = "mahalanobis", order = "closest")),
  run_one("k2k_mahalanobis_farthest", "lalonde_u", numeric_only,
          k2k = list(method = "mahalanobis", order = "farthest")),
  run_one("k2k_euclidean_closest", "lalonde_u", numeric_only,
          k2k = list(method = "euclidean", order = "closest")),
  run_one("k2k_mahalanobis_atc", "lalonde_u", numeric_only, "ATC",
          k2k = list(method = "mahalanobis")))

# The defect: "q2" gives one quantile, the median, and matchit() reads a
# single number as a number of bins. The same strata come from asking for
# "median" bins, and differ from those of the median as the only cut point,
# which matchit() cannot be asked for.
med <- median(lal$age)
as_bins <- matchit(treat ~ age + educ + race + married, lal, method = "cem",
                   cutpoints = list(age = med, educ = "q2"))
q2_run <- matchit(treat ~ age + educ + race + married, lal, method = "cem",
                  cutpoints = list(age = "q2", educ = "q2"))
q2_defect <- list(
  median_age = med,
  subclass_of_q2 = ifelse(is.na(q2_run$subclass), NA, as.integer(q2_run$subclass)),
  subclass_of_median_as_bins = ifelse(is.na(as_bins$subclass), NA, as.integer(as_bins$subclass)))
q2_defect$agree <- identical(q2_defect$subclass_of_q2, q2_defect$subclass_of_median_as_bins)

datasets_out <- lapply(datasets, function(d) {
  out <- lapply(d, function(col) if (is.numeric(col)) col else as.character(col))
  out
})

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  datasets = datasets_out, runs = runs, q2_defect = q2_defect)
write_json(out, "tests/data/matchit_cem.json", digits = I(17),
           auto_unbox = TRUE, na = "null", null = "null")
