# Reference optimal pair matching from MatchIt's matchit(method = "optimal"),
# which calls optmatch::fullmatch(). Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_optimal.R
#
# Data sets (the distance of two units is the absolute difference of a score x,
# or the Euclidean distance of two covariates for "euclid"):
#   wide      25 treated, 90 controls, no exact strata
#   strata    40 treated, 110 controls, exact strata g with 3 levels, one with
#             fewer controls than treated
#   tight     30 treated, 36 controls: fewer than ratio controls per treated
#   few       40 treated, 25 controls: fewer controls than treated
#   ties      scores rounded to one digit, so optimal matches are not unique
#   euclid    20 treated, 60 controls, Euclidean distance of two covariates,
#             given to MatchIt as a distance matrix
#   euclid_strata, euclid_tight, euclid_few
#             the strata, tight and few designs with the Euclidean distance,
#             whose optima are unique
#
# Every run is a call to matchit(method = "optimal") with estimand ATT or ATC,
# ratio 1, 2 or 3, with and without the exact strata. For ATC MatchIt matches
# the controls to treated units, so the focal group is the controls. For each
# run the file holds the pairs (focal unit, matched unit, rows of the data set,
# from 1), MatchIt's weights, the total distance of the pairs and whether the
# optimum is unique, meaning that six perturbations of every entry of the
# distance matrix by about 1e-5 standard deviations of the score give the same
# pairs, on a Euclidean data set. (Absolute differences of a score on a line tie often: two treated units
# on one side of two controls can swap them at the same cost.)
#
# `matchit()` ignores calipers with method = "optimal". The caliper runs call
# optmatch::fullmatch() directly on the distances with forbidden pairs set to
# Inf, in the way matchit2optimal() arranges the call (mean.controls = ratio,
# fewer controls per focal unit than the ratio handled by fractional mean and a
# smaller minimum, fewer controls than focal units by exchanging the groups).
# The script checks that this arrangement gives the same pairs as matchit() on
# every run without a caliper that is unique.
#
# optmatch's solver is LEMON's cycle cancelling here (rrelaxiv is not
# installed), on distances rounded to integers: with tol set to 1e-9 the
# resolution is epsilon = max(distance) / (.Machine$integer.max / 64 - 2), and
# the total distance of the result exceeds the optimum by at most
# 1.5 * epsilon per pair. `r_epsilon` holds epsilon for each run.

suppressMessages({
  library(MatchIt)
  library(optmatch)
  library(jsonlite)
})
options(fullmatch_try_recovery = FALSE)
TOL <- 1e-9
PERTURBATIONS <- 6

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
  wide = make_data(101, 25, 90),
  strata = make_data(102, 40, 110, c("a", "b", "c"),
                     p_t = c(0.3, 0.3, 0.4), p_c = c(0.45, 0.4, 0.15)),
  tight = make_data(103, 30, 36),
  few = make_data(104, 40, 25),
  ties = make_data(105, 30, 80, digits = 1),
  euclid = make_data(106, 20, 60),
  euclid_strata = make_data(107, 40, 110, c("a", "b", "c"),
                            p_t = c(0.3, 0.3, 0.4), p_c = c(0.45, 0.4, 0.15)),
  euclid_tight = make_data(108, 30, 36),
  euclid_few = make_data(109, 40, 25)
)
has_strata <- vapply(names(scenarios), function(n) n %in% c("strata", "euclid_strata"),
                     logical(1))
kind_of <- function(name) if (startsWith(name, "euclid")) "euclid" else "score"

# The focal x other distance matrix of a data set for a focal group (1 for
# the treated, 0 for the controls). Row and column names are the data rows.
dist_matrix <- function(d, kind, focal) {
  f <- which(d$treat == focal)
  o <- which(d$treat != focal)
  D <- if (kind == "euclid") {
    sqrt(outer(d$x[f], d$x[o], "-")^2 + outer(d$x2[f], d$x2[o], "-")^2)
  } else abs(outer(d$x[f], d$x[o], "-"))
  dimnames(D) <- list(as.character(f), as.character(o))
  D
}

# matchit() on the score vector, or on the treated x controls distance matrix
# plus `noise` (a matrix) when there is noise or the distance is a matrix.
# Perturbing the score itself would keep the ties of absolute differences on a
# line, so the uniqueness check perturbs every entry of the matrix instead.
call_matchit <- function(d, kind, estimand, ratio, exact, noise = NULL) {
  ex <- if (exact) ~g else NULL
  if (kind == "euclid" || !is.null(noise)) {
    D <- dist_matrix(d, kind, 1)
    if (!is.null(noise)) D <- D + noise
    distance <- D
  } else {
    distance <- d$x
  }
  suppressWarnings(matchit(treat ~ x, data = d, method = "optimal",
                           distance = distance, estimand = estimand,
                           ratio = ratio, exact = ex, tol = TOL))
}

pairs_of <- function(m) {
  mm <- m$match.matrix
  out <- list(focal = integer(), other = integer())
  for (i in seq_len(nrow(mm))) {
    o <- mm[i, ]
    o <- o[!is.na(o)]
    out$focal <- c(out$focal, rep(as.integer(rownames(mm)[i]), length(o)))
    out$other <- c(out$other, as.integer(o))
  }
  ord <- order(out$focal, out$other)
  list(focal = out$focal[ord], other = out$other[ord])
}

same_pairs <- function(a, b) identical(a$focal, b$focal) && identical(a$other, b$other)

total_distance <- function(D, p) {
  sum(D[cbind(match(as.character(p$focal), rownames(D)),
              match(as.character(p$other), colnames(D)))])
}

r_epsilon <- function(D) {
  max(D[is.finite(D)]) / (.Machine$integer.max / 64 - 2)
}

# matchit2optimal()'s arrangement of fullmatch() for one stratum.
opt_stratum <- function(D, ratio) {
  nr <- nrow(D); nc <- ncol(D)
  e <- nc / nr
  flipped <- FALSE
  if (e < 1) {
    D <- t(D); flipped <- TRUE
    mn <- mx <- 1; mean_c <- 1
  } else if (e < ratio) {
    mn <- floor(e); mx <- ratio; mean_c <- e
  } else {
    mn <- mx <- mean_c <- ratio
  }
  nm <- c(rownames(D), colnames(D))
  p <- suppressWarnings(fullmatch(as.InfinitySparseMatrix(D), min.controls = mn,
                                  max.controls = mx, mean.controls = mean_c,
                                  tol = TOL, data = data.frame(row.names = nm)))
  if (flipped) D <- t(D)
  matched <- !is.na(p)
  if (!any(matched)) return(NULL)
  focal_names <- rownames(D)
  out <- list(focal = integer(), other = integer())
  for (grp in unique(as.character(p[matched]))) {
    members <- names(p)[matched & as.character(p) == grp]
    f <- as.integer(members[members %in% focal_names])
    o <- as.integer(members[!members %in% focal_names])
    out$focal <- c(out$focal, rep(f, each = length(o)))
    out$other <- c(out$other, rep(o, times = length(f)))
  }
  ord <- order(out$focal, out$other)
  list(focal = out$focal[ord], other = out$other[ord])
}

weights_of <- function(m) unname(m$weights)

runs <- list()
set.seed(20261121)
for (name in names(scenarios)) {
  d <- scenarios[[name]]
  kind <- kind_of(name)
  for (estimand in c("ATT", "ATC")) {
    focal <- if (estimand == "ATT") 1 else 0
    D0 <- dist_matrix(d, kind, focal)
    for (ratio in c(1, 2, 3)) {
      for (exact in c(FALSE, TRUE)) {
        if (exact && !has_strata[[name]]) next
        run <- list(scenario = name, estimand = estimand, ratio = ratio,
                    exact = exact)
        m <- tryCatch(call_matchit(d, kind, estimand, ratio, exact),
                      error = function(e) NULL)
        if (is.null(m)) {
          run$error <- TRUE
          runs[[length(runs) + 1]] <- run
          next
        }
        p <- pairs_of(m)
        # Perturbed distances for the uniqueness check.
        perturbed <- lapply(1:PERTURBATIONS, function(k) {
          sc <- 1e-5 * sd(d$x)
          D <- dist_matrix(d, kind, 1)
          noise <- matrix(abs(rnorm(length(D), 0, sc)), nrow(D), ncol(D))
          pm <- tryCatch(call_matchit(d, kind, estimand, ratio, exact, noise),
                         error = function(e) NULL)
          if (is.null(pm)) NULL else pairs_of(pm)
        })
        unique_optimum <- all(vapply(perturbed, function(q) {
          !is.null(q) && same_pairs(p, q)
        }, logical(1)))
        if (!exact) {
          mine <- opt_stratum(D0, ratio)
          # Where the optimum is unique, the direct call must agree with
          # matchit(). On a line the optimum is rarely unique and matchit()
          # also thins the problem before solving, so only the Euclidean data
          # are required to agree whatever the flag.
          if (unique_optimum || kind == "euclid") {
            stopifnot(!is.null(mine), same_pairs(mine, p))
          }
        }
        run$error <- FALSE
        run$focal <- p$focal
        run$other <- p$other
        run$weights <- weights_of(m)
        run$total <- total_distance(D0, p)
        run$r_epsilon <- r_epsilon(D0)
        # Distances on a line tie, and a few perturbations can miss a tie, so
        # only the Euclidean data sets claim a unique optimum.
        run$unique <- unique_optimum && kind == "euclid"
        runs[[length(runs) + 1]] <- run
      }
    }
  }
}

# Calipers, through optmatch directly (ATT, one stratum).
caliper_runs <- list()
for (name in c("wide", "tight", "few", "euclid", "euclid_tight")) {
  d <- scenarios[[name]]
  D0 <- dist_matrix(d, kind_of(name), 1)
  for (width in c(0.25, 0.5, 1, 1.5)) {
    for (ratio in c(1, 2)) {
      D <- D0
      D[D > width] <- Inf
      run <- list(scenario = name, caliper = width, ratio = ratio)
      if (any(rowSums(is.finite(D)) == 0)) {
        # A treated unit with no control in the caliper is dropped by the
        # crate and counted in the number of rows by optmatch; no reference.
        next
      }
      p <- tryCatch(opt_stratum(D, ratio), error = function(e) NULL)
      if (is.null(p)) {
        run$error <- TRUE
        caliper_runs[[length(caliper_runs) + 1]] <- run
        next
      }
      sc <- 1e-5 * sd(d$x)
      unique_optimum <- all(vapply(1:PERTURBATIONS, function(k) {
        Dp <- D
        fin <- is.finite(Dp)
        Dp[fin] <- Dp[fin] + abs(rnorm(sum(fin), 0, sc))
        q <- tryCatch(opt_stratum(Dp, ratio), error = function(e) NULL)
        !is.null(q) && same_pairs(p, q)
      }, logical(1)))
      run$error <- FALSE
      run$focal <- p$focal
      run$other <- p$other
      run$total <- total_distance(D0, p)
      run$r_epsilon <- r_epsilon(D)
      run$unique <- unique_optimum && startsWith(name, "euclid")
      caliper_runs[[length(caliper_runs) + 1]] <- run
    }
  }
}

out <- list(
  versions = list(
    r = R.version.string,
    matchit = as.character(packageVersion("MatchIt")),
    optmatch = as.character(packageVersion("optmatch")),
    jsonlite = as.character(packageVersion("jsonlite")),
    solver = optmatch:::handleSolver("")),
  scenarios = lapply(scenarios, function(d) as.list(d)),
  runs = runs,
  caliper_runs = caliper_runs)
write_json(out, "tests/data/matchit_optimal.json", digits = I(17),
           auto_unbox = TRUE, na = "null")
cat(length(runs), "runs,", length(caliper_runs), "caliper runs;",
    sum(vapply(runs, function(r) isTRUE(r$unique), logical(1))), "unique\n")
