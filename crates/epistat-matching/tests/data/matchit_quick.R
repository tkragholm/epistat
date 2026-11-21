# Reference generalised full matching from quickmatch 0.2.3 and MatchIt's
# matchit(method = "quick").
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_quick.R
#
# quickmatch's subclasses depend on how its seeds are found, and on scclust's
# choices among units at equal distance. The data here have no equal
# distances, so each seed method gives one partition. The file holds, for
# several data sets, the points (one row per unit, row-major), the treatment,
# and the subclass quickmatch() gives each unit (null for a unit it leaves
# unassigned) under each seed method and each set of constraints, calling
# quickmatch() directly with the point matrix as matchit() does.
#
# For data sets where MatchIt can be given the same distance, the file also
# holds matchit(method = "quick") for each estimand: the subclass, the
# weights MatchIt returns (rescaled so that the weights of each group sum to
# the number of its units that keep a weight) and the unscaled weights
# computed here from the subclass counts, checked against MatchIt's after
# the same rescaling.
#
# Data:
#   line_80, line_300   a score on a line, 80 and 300 units, treated first
#   plane_150           two coordinates
#   space_100           three coordinates
#   lalonde_ps          (treated first) the linear predictor of the propensity score of
#                       lalonde, perturbed so that no two units share a score
#   mahal_120           (treated first) two covariates and matchit(distance = "mahalanobis")
#   small_*             sets of 9 to 11 units for the exact optimum
suppressMessages({
  library(MatchIt)
  library(quickmatch)
  library(jsonlite)
})

as_labels <- function(x) {
  x <- as.integer(unclass(x))
  ifelse(is.na(x), NA, x)
}

qm_run <- function(points, treat, min_anchors = 1, min_candidates = 1,
                   size = NULL, caliper = NULL, seed_method = "lexical") {
  tf <- factor(ifelse(treat == 1, "treated", "control"),
               levels = c("control", "treated"))
  tc <- c(control = min_candidates, treated = min_anchors)
  args <- list(distances = points, treatments = tf,
               treatment_constraints = tc, seed_method = seed_method)
  if (!is.null(size)) args$size_constraint <- size
  if (!is.null(caliper)) args$caliper <- caliper
  cl <- tryCatch(do.call(quickmatch::quickmatch, args), error = function(e) NULL)
  list(min_anchors = min_anchors, min_candidates = min_candidates,
       size = if (is.null(size)) NA else size,
       caliper = if (is.null(caliper)) NA else caliper,
       seed_method = seed_method,
       error = is.null(cl),
       cluster = if (is.null(cl)) NULL else as_labels(cl))
}

unscaled <- function(subclass, treat, estimand) {
  w <- rep(0, length(treat))
  for (s in sort(unique(subclass[!is.na(subclass)]))) {
    i <- which(subclass == s)
    n1 <- sum(treat[i] == 1)
    n0 <- sum(treat[i] == 0)
    n <- n1 + n0
    w[i] <- switch(estimand,
      ATT = ifelse(treat[i] == 1, 1, n1 / n0),
      ATC = ifelse(treat[i] == 1, n0 / n1, 1),
      ATE = ifelse(treat[i] == 1, n / n1, n / n0))
  }
  w
}

matchit_run <- function(df, f, distance, seed_method, estimand, caliper = NULL) {
  args <- list(formula = f, data = df, method = "quick", distance = distance,
               estimand = estimand, seed_method = seed_method)
  if (!is.null(caliper)) {
    args$caliper <- caliper
    args$std.caliper <- FALSE
  }
  summarise_matchit(do.call(matchit, args), df$treat, estimand, seed_method,
                    caliper)
}

summarise_matchit <- function(m, treat, estimand, seed_method, caliper = NULL) {
  sub <- ifelse(is.na(m$subclass), NA, as.integer(m$subclass))
  raw <- unscaled(sub, treat, estimand)
  scaled <- raw
  for (g in 0:1) {
    i <- which(treat == g & !is.na(sub))
    if (length(i)) scaled[i] <- raw[i] * length(i) / sum(raw[i])
  }
  stopifnot(all(abs(scaled - unname(m$weights)) <=
                1e-12 * pmax(1, abs(unname(m$weights)))))
  list(estimand = estimand, seed_method = seed_method,
       caliper = if (is.null(caliper)) NA else caliper,
       subclass = sub, weights = unname(m$weights), unscaled_weights = raw)
}

constraint_sets <- list(
  list(1, 1, NULL), list(1, 2, NULL), list(2, 1, NULL), list(2, 2, 7),
  list(1, 1, 5))
seed_methods <- c("lexical", "inwards_order", "inwards_updating")

make_set <- function(name, points, treat, matchit_frame = NULL, f = NULL,
                     distance = NULL, calipers = NULL) {
  points <- as.matrix(points)
  runs <- list()
  for (sm in seed_methods) for (cs in constraint_sets) {
    runs[[length(runs) + 1]] <- qm_run(points, treat, cs[[1]], cs[[2]], cs[[3]],
                                       NULL, sm)
  }
  for (cal in calipers) for (sm in seed_methods) {
    runs[[length(runs) + 1]] <- qm_run(points, treat, 1, 1, NULL, cal, sm)
  }
  mi <- list()
  if (!is.null(matchit_frame)) {
    for (sm in seed_methods) for (est in c("ATT", "ATC", "ATE")) {
      mi[[length(mi) + 1]] <- matchit_run(matchit_frame, f, distance, sm, est)
    }
    for (cal in calipers[1]) for (est in c("ATT", "ATE")) {
      mi[[length(mi) + 1]] <- matchit_run(matchit_frame, f, distance, "lexical",
                                          est, cal)
    }
  }
  list(name = name, dim = ncol(points), treat = treat,
       points = as.vector(t(points)), runs = runs, matchit = mi)
}

set.seed(20261121)

sim_line <- function(n, shift) {
  x <- rnorm(n)
  treat <- rbinom(n, 1, plogis(-0.4 + shift * x))
  data.frame(treat = treat, x = x)
}
# The crate takes the anchors first and then the candidates, and the seed
# order follows the order of the units, so the data MatchIt sees put the
# treated units first, in the order they were drawn.
treated_first <- function(d) {
  d <- d[order(-d$treat), , drop = FALSE]
  rownames(d) <- NULL
  d
}
l80 <- treated_first(sim_line(80, 0.9))
l300 <- treated_first(sim_line(300, 0.7))

sim_plane <- function(n, d) {
  m <- matrix(rnorm(n * d), n, d)
  treat <- rbinom(n, 1, plogis(-0.5 + 0.8 * m[, 1]))
  list(treat = treat, m = m)
}
p150 <- sim_plane(150, 2)
s100 <- sim_plane(100, 3)

data(lalonde)
ps <- fitted(glm(treat ~ age + educ + race + married + nodegree + re74 + re75,
                 data = lalonde, family = binomial))
lal <- treated_first(data.frame(treat = lalonde$treat,
                  ps = qlogis(ps) +
                    1e-7 * (seq_len(nrow(lalonde)) %% 97) / 97 +
                    1e-9 * seq_len(nrow(lalonde))))
stopifnot(!anyDuplicated(lal$ps))

mh <- data.frame(z1 = rnorm(120), z2 = rnorm(120))
mh$treat <- rbinom(120, 1, plogis(-0.4 + 0.7 * mh$z1))
mh <- treated_first(mh)

small <- function(name, n, d) {
  m <- matrix(rnorm(n * d), n, d)
  treat <- c(1, 0, rbinom(n - 2, 1, 0.45))
  make_set(name, m, treat)
}

datasets <- list(
  make_set("line_80", l80$x, l80$treat, l80, treat ~ x, l80$x,
           calipers = c(0.15, 0.4)),
  make_set("line_300", l300$x, l300$treat, l300, treat ~ x, l300$x,
           calipers = c(0.05)),
  make_set("plane_150", p150$m, p150$treat, calipers = c(0.6)),
  make_set("space_100", s100$m, s100$treat),
  make_set("lalonde_ps", lal$ps, lal$treat, lal, treat ~ ps, lal$ps,
           calipers = c(0.3)),
  make_set("mahal_120", as.matrix(mh[c("z1", "z2")]), mh$treat))

mh_runs <- list()
for (sm in seed_methods) for (est in c("ATT", "ATC", "ATE")) {
  m <- matchit(treat ~ z1 + z2, data = mh, method = "quick",
               distance = "mahalanobis", estimand = est, seed_method = sm)
  mh_runs[[length(mh_runs) + 1]] <- summarise_matchit(m, mh$treat, est, sm)
}
datasets[[6]]$matchit <- mh_runs
datasets[[6]]$z1 <- mh$z1
datasets[[6]]$z2 <- mh$z2

smalls <- lapply(seq_len(8), function(i) {
  s <- small(paste0("small_", i), 9 + (i %% 3), if (i %% 2) 1 else 2)
  s$runs <- s$runs[sapply(s$runs, function(r)
    is.null(r$caliper) && r$min_anchors == 1 && r$min_candidates == 1 &&
      is.null(r$size))]
  s
})

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  quickmatch = as.character(packageVersion("quickmatch")),
                  scclust = as.character(packageVersion("scclust")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  datasets = c(datasets, smalls))
write_json(out, "tests/data/matchit_quick.json", digits = I(17),
           auto_unbox = TRUE, na = "null")
