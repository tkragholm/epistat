# Reference optimal full matching from MatchIt's matchit(method = "full"),
# which calls optmatch::fullmatch(). Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_full.R
#
# Data sets. The distance of two units is the Euclidean distance of two
# covariates (x, x2), given to MatchIt as a distance matrix, or the absolute
# difference of the score x given as a vector.
#   euclid        20 treated, 60 controls
#   euclid_strata 30 treated, 80 controls, exact strata g with 3 levels
#   euclid_few    30 treated, 18 controls
#   score         20 treated, 60 controls, scores on a line
#   ties          scores rounded to one digit, so optimal matches tie
#
# Runs combine the limits of fullmatch() (default, min.controls,
# max.controls, both, many-to-one limits, omit.fraction with either sign,
# mean.controls), the estimands ATT, ATC and ATE, exact strata, and calipers
# on the distance (raw units, which MatchIt allows with full matching only
# while min.controls is 0). tol is 1e-9. optmatch is run with
# options(fullmatch_try_recovery = FALSE), so a problem with no feasible
# matching fails instead of being repaired by omitting more controls.
#
# Runs with a negative omit.fraction or a max.controls of at most 1/2 meet
# the defect described near the end of this file, so optmatch's total can be
# above the optimum there.
#
# A run where MatchIt stops with "No matches were found" is recorded as an
# error. A run where one exact stratum with admissible pairs gets no matches
# while another does is left out, since the crate treats that as an error for
# the whole match.
#
# For ATC MatchIt takes the controls as the focal group, so the crate is given
# the controls as anchors. For each run the file holds the subclass of every
# unit (null where MatchIt leaves a unit unmatched), MatchIt's weights, the
# total distance (the sum of the distances of the treated-control pairs within
# each subclass), and whether the partition is unique, meaning that six
# perturbations of every entry of the distance matrix by about 1e-5 standard
# deviations of the score give the same partition, on a Euclidean data set. `r_epsilon` is optmatch's
# resolution: its total is at most 1.5 * epsilon per pair above the optimum.

suppressMessages({
  library(MatchIt)
  library(optmatch)
  library(jsonlite)
})
options(fullmatch_try_recovery = FALSE)
TOL <- 1e-9
PERTURBATIONS <- 4

make_data <- function(seed, n_t, n_c, levels = NULL, p_t = NULL, p_c = NULL,
                      digits = NULL, shift = 0.6) {
  set.seed(seed)
  x <- c(rnorm(n_t, shift), rnorm(n_c, 0))
  if (!is.null(digits)) x <- round(x, digits)
  g <- if (is.null(levels)) rep("a", n_t + n_c) else
    c(sample(levels, n_t, TRUE, p_t), sample(levels, n_c, TRUE, p_c))
  d <- data.frame(treat = c(rep(1L, n_t), rep(0L, n_c)), x = x, g = g)
  d$x2 <- rnorm(nrow(d), 0.3 * d$treat)
  d <- d[sample(nrow(d)), ]
  rownames(d) <- NULL
  d
}

scenarios <- list(
  euclid = make_data(201, 20, 60),
  euclid_strata = make_data(202, 30, 80, c("a", "b", "c"),
                            p_t = c(0.3, 0.3, 0.4), p_c = c(0.45, 0.4, 0.15)),
  euclid_few = make_data(203, 30, 18),
  score = make_data(204, 20, 60),
  ties = make_data(205, 20, 60, digits = 1)
)
kind_of <- function(name) if (startsWith(name, "euclid")) "euclid" else "score"
has_strata <- function(name) name == "euclid_strata"

# Treated x controls distance matrix, rows and columns named by data row.
dist_matrix <- function(d, kind) {
  f <- which(d$treat == 1); o <- which(d$treat == 0)
  D <- if (kind == "euclid") {
    sqrt(outer(d$x[f], d$x[o], "-")^2 + outer(d$x2[f], d$x2[o], "-")^2)
  } else abs(outer(d$x[f], d$x[o], "-"))
  dimnames(D) <- list(as.character(f), as.character(o))
  D
}

call_matchit <- function(d, kind, estimand, limits, exact, caliper, noise = NULL) {
  D <- dist_matrix(d, kind)
  distance <- if (kind == "euclid" || !is.null(noise)) {
    if (is.null(noise)) D else D + noise
  } else d$x
  args <- c(list(formula = treat ~ x, data = d, method = "full",
                 distance = distance, estimand = estimand,
                 exact = if (exact) ~g else NULL, tol = TOL), limits)
  if (!is.null(caliper)) {
    args$caliper <- caliper
    args$std.caliper <- FALSE
  }
  suppressWarnings(do.call(matchit, args))
}

# The partition of the units: for each unit the sorted members of its subclass,
# NA where unmatched.
signature <- function(m) {
  s <- m$subclass
  vapply(seq_along(s), function(i) {
    if (is.na(s[i])) NA_character_ else paste(which(s == s[i]), collapse = ",")
  }, character(1))
}

total_distance <- function(m, D) {
  s <- m$subclass
  tot <- 0
  for (grp in levels(s)) {
    members <- which(s == grp)
    t <- members[members %in% as.integer(rownames(D))]
    cc <- members[members %in% as.integer(colnames(D))]
    if (length(t) && length(cc)) {
      tot <- tot + sum(D[as.character(t), as.character(cc), drop = FALSE])
    }
  }
  tot
}

limit_sets <- list(
  default = list(),
  min1 = list(min.controls = 1),
  max3 = list(max.controls = 3),
  min1_max4 = list(min.controls = 1, max.controls = 4),
  min2 = list(min.controls = 2),
  many_one = list(min.controls = 0.5, max.controls = 2),
  max_half = list(max.controls = 0.5),
  omit03 = list(omit.fraction = 0.3),
  omit_neg = list(omit.fraction = -0.3),
  mean15 = list(mean.controls = 1.5),
  mean2_min1 = list(min.controls = 1, mean.controls = 2),
  mean2_max3 = list(max.controls = 3, mean.controls = 2)
)

runs <- list()
set.seed(20261122)
for (name in names(scenarios)) {
  d <- scenarios[[name]]
  kind <- kind_of(name)
  D <- dist_matrix(d, kind)
  # Calipers are in raw units of the distance.
  calipers <- list(NULL, as.numeric(quantile(D, 0.35)), as.numeric(quantile(D, 0.6)))
  cat(name, format(Sys.time()), "\n", file = stderr())
  for (estimand in c("ATT", "ATC", "ATE")) {
    for (lname in names(limit_sets)) {
      limits <- limit_sets[[lname]]
      for (exact in c(FALSE, TRUE)) {
        if (exact && !has_strata(name)) next
        for (ci in seq_along(calipers)) {
          caliper <- calipers[[ci]]
          if (!is.null(caliper) && !is.null(limits$min.controls) && limits$min.controls != 0) next
          # Keep the number of runs down: calipers only with a few limit sets.
          if (!is.null(caliper) && !lname %in% c("default", "max3", "many_one", "omit03")) next
          m <- tryCatch(call_matchit(d, kind, estimand, limits, exact, caliper),
                        error = function(e) NULL)
          run <- list(scenario = name, estimand = estimand, limits = lname,
                      exact = exact, caliper = caliper)
          if (is.null(m)) {
            run$error <- TRUE
            runs[[length(runs) + 1]] <- run
            next
          }
          # A stratum with admissible pairs but no matches is a partial failure.
          if (exact) {
            ok <- TRUE
            for (lev in unique(d$g)) {
              rows <- which(d$g == lev)
              adm <- if (is.null(caliper)) TRUE else {
                Dl <- D[as.character(intersect(rows, which(d$treat == 1))),
                        as.character(intersect(rows, which(d$treat == 0))), drop = FALSE]
                length(Dl) > 0 && any(Dl <= caliper)
              }
              both <- any(d$treat[rows] == 1) && any(d$treat[rows] == 0)
              if (both && adm && all(is.na(m$subclass[rows]))) ok <- FALSE
            }
            if (!ok) next
          }
          sig <- signature(m)
          unique_partition <- all(vapply(seq_len(PERTURBATIONS), function(k) {
            sc <- 1e-5 * sd(d$x)
            noise <- matrix(abs(rnorm(length(D), 0, sc)), nrow(D), ncol(D))
            pm <- tryCatch(call_matchit(d, kind, estimand, limits, exact, caliper, noise),
                           error = function(e) NULL)
            !is.null(pm) && identical(signature(pm), sig)
          }, logical(1)))
          run$error <- FALSE
          run$subclass <- ifelse(is.na(m$subclass), NA, as.integer(m$subclass))
          run$weights <- unname(m$weights)
          run$total <- total_distance(m, D)
          run$r_epsilon <- max(D) / (.Machine$integer.max / 64 - 2)
          # Distances on a line tie, and a few perturbations can miss a tie, so only
          # the Euclidean data sets claim a unique partition.
          run$unique <- unique_partition && kind == "euclid"
          runs[[length(runs) + 1]] <- run
        }
      }
    }
  }
}

# A defect in optmatch::fullmatch(): a negative omit.fraction (or a max.controls
# of at most 1/2) solves the problem with treated and controls exchanged, but
# limits the group size by the dimensions of the problem before the exchange.
# With fewer treated than controls the largest group is capped at the number
# of treated plus one controls per treated (11 here), where the transposed call
# with a positive omit.fraction, which the documentation says is the same
# problem, allows 15 and has a lower total distance.
set.seed(7)
nt <- 10; nc <- 40
xs <- matrix(rnorm((nt + nc) * 2), ncol = 2)
DD <- as.matrix(dist(xs))[1:nt, (nt + 1):(nt + nc)]
dimnames(DD) <- list(paste0("t", 1:nt), paste0("c", 1:nc))
nm <- c(rownames(DD), colnames(DD))
tot_of <- function(p, D) {
  s <- 0
  for (g in levels(p)) {
    m <- names(p)[which(p == g)]
    tt <- m[m %in% rownames(D)]; cc <- m[m %in% colnames(D)]
    if (length(tt) && length(cc)) s <- s + sum(D[tt, cc, drop = FALSE])
  }
  s
}
neg <- fullmatch(DD, omit.fraction = -0.3, tol = TOL, data = data.frame(row.names = nm))
pos <- fullmatch(t(DD), omit.fraction = 0.3, tol = TOL, data = data.frame(row.names = nm))
defect <- list(
  points = unname(xs), n_treated = nt, n_controls = nc,
  total_negative = tot_of(neg, DD), total_transposed = tot_of(pos, DD),
  largest_group_negative = max(table(neg[!is.na(neg)])),
  largest_group_transposed = max(table(pos[!is.na(pos)])),
  r_epsilon = max(DD) / (.Machine$integer.max / 64 - 2))
stopifnot(defect$total_negative > defect$total_transposed)

out <- list(
  defect = defect,
  versions = list(
    r = R.version.string,
    matchit = as.character(packageVersion("MatchIt")),
    optmatch = as.character(packageVersion("optmatch")),
    jsonlite = as.character(packageVersion("jsonlite")),
    solver = optmatch:::handleSolver("")),
  scenarios = lapply(scenarios, function(d) as.list(d)),
  limits = limit_sets,
  runs = runs)
write_json(out, "tests/data/matchit_full.json", digits = I(17),
           auto_unbox = TRUE, na = "null", null = "null")
n_ok <- sum(vapply(runs, function(r) !isTRUE(r$error), logical(1)))
cat(length(runs), "runs,", n_ok, "matched,",
    sum(vapply(runs, function(r) isTRUE(r$unique), logical(1))), "unique,",
    sum(vapply(runs, function(r) isTRUE(r$error), logical(1))), "errors\n")
