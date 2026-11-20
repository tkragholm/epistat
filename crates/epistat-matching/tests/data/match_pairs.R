# Pairs, weights and output shapes of MatchIt 4.8.2 for the rows that do not
# depend on how candidates are ranked: estimand, exact, caliper on named
# covariates (raw scale), replace, ratio, match.matrix, subclass, weights and
# get_matches().
#
# Making MatchIt's nearest neighbour comparable with a lowest-index choice:
# `distance` is 0 for every unit of the focal group (the treated units for
# estimand = "ATT", the controls for "ATC") and 1, 2, 3, ... in row order for
# the other group. The nearest available unit is then the first in row order,
# with no ties, and `m.order = "data"` takes the focal units in row order.
#
# For every run the file holds:
#   matchit      what matchit() returned: match.matrix, subclass, weights with
#                the default normalize = TRUE and with normalize = FALSE, and
#                get_matches() for a subset of the runs
#   other        the weights MatchIt's own routine gives the same pairs under
#                the opposite estimand, with the roles of the groups swapped
#   ate          for replace = FALSE, the stratum weights for estimand = "ATE"
#                on the same pairs (nearest neighbour matching itself refuses
#                ATE)
#   sequential   for replace = FALSE and ratio > 1, where MatchIt hands out
#                controls in rounds (every focal unit gets its first control
#                before any gets a second), the same fields for the pairs that
#                result when each focal unit takes all its controls before the
#                next focal unit starts, which is how epistat-matching works
# Units are numbered by row of the data set, from 1. A weight list is given
# for the units with non-zero weight; the others have weight 0.
#
# The two models of the order of assignment are small loops written here from
# the documented behaviour, and the script stops if the round model disagrees
# with matchit() in any run.
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/match_pairs.R
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

args <- commandArgs(FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))

# ---- data -----------------------------------------------------------------

set.seed(20261120)
strata_counts <- data.frame(
  s1 = rep(c("a", "b", "c", "d"), each = 2), s2 = rep(0:1, 4),
  treated = c(8, 4, 6, 3, 5, 4, 3, 3),
  controls = c(30, 12, 25, 6, 20, 3, 14, 0)
)
sim <- do.call(rbind, lapply(seq_len(nrow(strata_counts)), function(i) {
  r <- strata_counts[i, ]
  data.frame(
    treat = c(rep(1L, r$treated), rep(0L, r$controls)),
    s1 = r$s1, s2 = r$s2
  )
}))
sim <- sim[sample(nrow(sim)), ]
sim$x <- round(rnorm(nrow(sim), 10, 2), 1)
sim$age <- round(runif(nrow(sim), 20, 40))
rownames(sim) <- NULL

data(lalonde, package = "MatchIt")
set.seed(20261121)
tr <- sample(which(lalonde$treat == 1), 80)
co <- sample(which(lalonde$treat == 0), 240)
lal <- lalonde[sample(c(tr, co)), ]
lal$race <- as.character(lal$race)
rownames(lal) <- NULL

scenarios <- list(
  sim = list(
    data = sim, strata = c("s1", "s2"),
    calipers = list(list(), list(age = 3), list(x = 1), list(age = 3, x = 1.5)),
    ratios = c(1, 2, 3)
  ),
  lalonde = list(
    data = lal, strata = c("race", "married", "nodegree"),
    calipers = list(list(), list(age = 3), list(age = 3, re74 = 2000)),
    ratios = c(1, 3)
  )
)

# ---- helpers --------------------------------------------------------------

normalize_weights <- function(w, treat) {
  for (g in 0:1) {
    nz <- treat == g & w > 0
    if (any(nz)) w[nz] <- w[nz] / mean(w[nz])
  }
  w
}

# MatchIt's own routines for raw weights from a match matrix and from strata.
mm_weights <- function(mm, treat, focal) {
  names(treat) <- seq_along(treat)
  tryCatch(unname(MatchIt:::get_weights_from_mm(mm, treat, focal)),
           error = function(e) rep(0, length(treat)))
}
sub_weights <- function(subclass, treat, estimand) {
  tryCatch(MatchIt:::get_weights_from_subclass(subclass, treat, estimand),
           error = function(e) rep(0, length(treat)))
}

# Match matrix of character unit names from a list of index vectors.
as_mm <- function(rows, focal_idx, ratio) {
  m <- matrix(NA_character_, length(focal_idx), ratio,
              dimnames = list(as.character(focal_idx), NULL))
  for (i in seq_along(focal_idx)) {
    v <- rows[[i]]
    if (length(v)) m[i, seq_along(v)] <- as.character(v)
  }
  m
}

# Pairs from rows (list over focal units) as a match matrix with the roles
# swapped: one row per unit of the other group, in row order of the data, as
# MatchIt's weight routine reads a match matrix by position, holding the focal
# units matched to it. Units with no match are rows of NA.
mirror_mm <- function(rows, focal_idx, pool_idx) {
  pool_of <- unlist(rows)
  if (!length(pool_of)) return(NULL)
  focal_of <- rep(focal_idx, lengths(rows))
  lst <- lapply(pool_idx, function(p) focal_of[pool_of == p])
  k <- max(lengths(lst))
  m <- matrix(NA_character_, length(pool_idx), k, dimnames = list(as.character(pool_idx), NULL))
  for (i in seq_along(pool_idx)) if (length(lst[[i]])) m[i, seq_along(lst[[i]])] <- as.character(lst[[i]])
  m
}

# Everything stored for one set of pairs.
describe_pairs <- function(rows, focal_idx, pool_idx, treat, focal_value, ratio, replace) {
  n <- length(treat)
  mm <- as_mm(rows, focal_idx, ratio)
  raw <- if (any(lengths(rows) > 0)) mm_weights(mm, treat, focal_value) else rep(0, n)
  mir <- mirror_mm(rows, focal_idx, pool_idx)
  oth <- if (is.null(mir)) rep(0, n) else mm_weights(mir, treat, 1L - focal_value)
  out <- list(
    rows = lapply(rows, function(v) as.integer(v)),
    raw = raw, norm = normalize_weights(raw, treat),
    other_raw = oth, other_norm = normalize_weights(oth, treat)
  )
  if (!replace) {
    sub <- rep(NA_integer_, n)
    matched <- which(lengths(rows) > 0)
    for (j in seq_along(matched)) sub[c(focal_idx[matched[j]], rows[[matched[j]]])] <- j
    out$subclass <- sub
    ate <- sub_weights(sub, treat, "ATE")
    out$ate_raw <- ate
    out$ate_norm <- normalize_weights(ate, treat)
  }
  out
}

sparse <- function(p, n) {
  keep <- which(p$raw > 0 | p$other_raw > 0)
  sp <- function(w) round(w[keep], 12)
  out <- list(idx = keep, raw = sp(p$raw), norm = sp(p$norm),
              other_raw = sp(p$other_raw))
  if (!is.null(p$subclass)) {
    s <- which(!is.na(p$subclass))
    out$subclass_idx <- s
    out$subclass <- p$subclass[s]
    out$ate_raw <- round(p$ate_raw[keep], 12)
  }
  out$rows <- p$rows
  out
}

# The two models of the order of assignment.
assign_pairs <- function(elig, ratio, replace, rounds) {
  nf <- nrow(elig)
  used <- logical(ncol(elig))
  chosen <- vector("list", nf)
  for (i in seq_len(nf)) chosen[[i]] <- integer(0)
  if (rounds) {
    for (r in seq_len(ratio)) for (i in seq_len(nf)) {
      ok <- elig[i, ] & (replace | !used)
      ok[chosen[[i]]] <- FALSE
      pick <- which(ok)[1]
      if (!is.na(pick)) {
        chosen[[i]] <- c(chosen[[i]], pick)
        if (!replace) used[pick] <- TRUE
      }
    }
  } else {
    for (i in seq_len(nf)) {
      ok <- elig[i, ] & (replace | !used)
      pick <- head(which(ok), ratio)
      chosen[[i]] <- pick
      if (!replace) used[pick] <- TRUE
    }
  }
  chosen
}

# ---- runs -----------------------------------------------------------------

runs <- list()
for (sname in names(scenarios)) {
  sc <- scenarios[[sname]]
  d <- sc$data
  n <- nrow(d)
  idn <- as.character(seq_len(n))
  rownames(d) <- idn
  for (est in c("ATT", "ATC")) {
    focal_value <- if (est == "ATT") 1L else 0L
    focal_idx <- which(d$treat == focal_value)
    pool_idx <- which(d$treat != focal_value)
    dist <- numeric(n)
    dist[pool_idx] <- seq_along(pool_idx)
    d$dist <- dist
    for (use_exact in c(FALSE, TRUE)) for (cal in sc$calipers) {
      for (replace in c(FALSE, TRUE)) for (ratio in sc$ratios) {
        key <- do.call(paste, c(lapply(d[sc$strata], as.character), sep = "|"))
        elig <- outer(focal_idx, pool_idx, function(f, p) {
          ok <- rep(TRUE, length(f))
          if (use_exact) ok <- ok & key[f] == key[p]
          for (nm in names(cal)) ok <- ok & abs(d[[nm]][f] - d[[nm]][p]) <= cal[[nm]]
          ok
        })
        warns <- character(0)
        fit <- withCallingHandlers(
          tryCatch(
            matchit(
              as.formula("treat ~ age"), data = d, method = "nearest",
              distance = d$dist, estimand = est, m.order = "data",
              ratio = ratio, replace = replace,
              exact = if (use_exact) as.formula(paste("~", paste(sc$strata, collapse = " + "))) else NULL,
              caliper = if (length(cal)) unlist(cal) else NULL, std.caliper = FALSE
            ),
            error = function(e) e
          ),
          warning = function(w) {
            warns <<- c(warns, conditionMessage(w))
            invokeRestart("muffleWarning")
          }
        )
        if (inherits(fit, "error")) {
          stop(sprintf("%s %s exact=%s ratio=%d replace=%s: %s", sname, est,
                       use_exact, ratio, replace, conditionMessage(fit)))
        }
        mm <- fit$match.matrix
        rows <- lapply(seq_along(focal_idx), function(i) {
          v <- mm[i, ]
          as.integer(v[!is.na(v)])
        })
        # The round model must reproduce matchit(); the other model is ours.
        round_model <- assign_pairs(elig, ratio, replace, TRUE)
        round_model <- lapply(round_model, function(v) pool_idx[v])
        stopifnot(identical(lapply(rows, as.integer), lapply(round_model, as.integer)))
        treat <- d$treat
        p <- describe_pairs(rows, focal_idx, pool_idx, treat, focal_value, ratio, replace)
        # MatchIt's weights are what its own routine gives these pairs.
        stopifnot(isTRUE(all.equal(unname(fit$weights), p$norm, tolerance = 1e-12)))
        stopifnot(isTRUE(all.equal(
          unname(suppressWarnings(matchit(as.formula("treat ~ age"), data = d, method = "nearest",
                         distance = d$dist, estimand = est, m.order = "data",
                         ratio = ratio, replace = replace, normalize = FALSE,
                         exact = if (use_exact) as.formula(paste("~", paste(sc$strata, collapse = " + "))) else NULL,
                         caliper = if (length(cal)) unlist(cal) else NULL,
                         std.caliper = FALSE))$weights),
          p$raw, tolerance = 1e-12)))
        if (!replace) {
          sub <- as.integer(fit$subclass)
          stopifnot(identical(sub, p$subclass))
        }
        run <- list(
          scenario = sname, estimand = est, exact = if (use_exact) sc$strata else list(),
          caliper = cal, replace = replace, ratio = ratio,
          warnings = unique(warns),
          matchit = sparse(p, n)
        )
        if (!replace && ratio > 1) {
          seq_rows <- lapply(assign_pairs(elig, ratio, replace, FALSE), function(v) pool_idx[v])
          ps <- describe_pairs(seq_rows, focal_idx, pool_idx, treat, focal_value, ratio, replace)
          run$sequential <- sparse(ps, n)
        }
        # get_matches() for the runs of the main grid without a caliper on x.
        if (is.null(cal$x) && length(cal) <= 1) {
          gm <- suppressWarnings(get_matches(fit, id = "id", data = d))
          run$get_matches <- list(
            id = as.integer(gm$id), subclass = as.integer(as.character(gm$subclass)),
            weights = round(gm$weights, 12)
          )
        }
        runs[[length(runs) + 1]] <- run
      }
    }
  }
}

scenario_data <- lapply(scenarios, function(sc) {
  d <- sc$data
  list(
    treat = d$treat,
    strata = setNames(lapply(sc$strata, function(s) as.character(d[[s]])), sc$strata),
    numerics = setNames(lapply(c("age", if ("x" %in% names(d)) "x", if ("re74" %in% names(d)) "re74"),
                               function(s) as.numeric(d[[s]])),
                        c("age", if ("x" %in% names(d)) "x", if ("re74" %in% names(d)) "re74"))
  )
})

out <- list(
  versions = list(r = R.version.string, matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  scenarios = scenario_data,
  runs = runs
)
write(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"),
      file.path(here, "match_pairs.json"))
cat(length(runs), "runs\n")
