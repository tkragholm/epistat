# Reference for the matching step of vector matching, from Matching::Match.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/vector_matching.R
#
# Eight simulated data sets of 300 to 500 units in three groups with three
# covariates. For each, nnet::multinom() gives the generalised propensity
# score, the units are clustered by stats::kmeans() on the logit scores of
# the group that is neither the focal group (0) nor the compared one, and,
# within each cluster, Matching::Match(M = 1, replace = TRUE,
# caliper = 0.25 or none) matches the units of the focal group to those of the
# compared group on the logit score of the focal group, as Lopez and Gutman's
# description of vector matching says. The file records the ids, scores,
# focal flags and clusters that went in, and the pairs that came out, so that
# the Rust test gives the matching step the same input. Match() treats two
# candidates whose squared scaled distances differ by less than
# distance.tolerance (1e-5 by default) as tied and then picks one at random, so
# the script sets distance.tolerance = 0 to get the nearest candidate. A cluster in which
# Match() finds no valid match (it returns -99999) has no pairs. Ties in the scores
# do not occur because the covariates are continuous.

suppressMessages({
  library(nnet)
  library(Matching)
  library(jsonlite)
})

set.seed(20261121)
runs <- list()
for (r in 1:8) {
  n <- sample(300:500, 1)
  x <- matrix(rnorm(n * 3), n, 3)
  eta <- cbind(0, x %*% matrix(rnorm(6, sd = 0.7), 3, 2))
  p <- exp(eta) / rowSums(exp(eta))
  g <- apply(p, 1, function(pi) sample.int(3, 1, prob = pi))
  d <- data.frame(g = factor(g), x1 = x[, 1], x2 = x[, 2], x3 = x[, 3])
  m <- multinom(g ~ x1 + x2 + x3, data = d, trace = FALSE, reltol = 1e-12, maxit = 5000)
  gps <- fitted(m)
  lg <- log(gps / (1 - gps))
  other <- if (r %% 2 == 0) 2 else 3
  third <- setdiff(1:3, c(1, other))
  k <- sample(2:4, 1)
  km <- kmeans(lg[, third], centers = k, nstart = 5)
  caliper <- if (r %% 4 == 0) NULL else 0.25
  keep <- which(g %in% c(1, other))
  ids <- keep
  score <- lg[keep, 1]
  focal <- g[keep] == 1
  cluster <- km$cluster[keep]
  pairs <- list()
  for (cl in sort(unique(cluster))) {
    idx <- which(cluster == cl)
    if (sum(focal[idx]) == 0 || sum(!focal[idx]) == 0) next
    res <- tryCatch(
      Match(Tr = focal[idx], X = score[idx], M = 1, replace = TRUE,
            caliper = caliper, ties = FALSE, Weight = 1,
            distance.tolerance = 0),
      error = function(e) NULL)
    if (!is.list(res)) next
    pairs[[length(pairs) + 1]] <- list(
      focal = ids[idx][res$index.treated],
      control = ids[idx][res$index.control])
  }
  runs[[length(runs) + 1]] <- list(
    other = other - 1L, caliper = caliper, ids = ids, score = score,
    is_focal = focal, cluster = cluster - 1L,
    pairs_focal = unlist(lapply(pairs, function(p) p$focal)),
    pairs_control = unlist(lapply(pairs, function(p) p$control)))
}
out <- list(versions = list(r = R.version.string,
                            Matching = as.character(packageVersion("Matching")),
                            nnet = as.character(packageVersion("nnet"))),
            runs = runs)
write(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), "tests/data/vector_matching.json")
