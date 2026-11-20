# Reference subclassification from MatchIt's matchit(method = "subclass").
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_subclass.R
#
# Every run supplies the distance as a numeric vector, so MatchIt uses the
# scores as given. For each run the script writes MatchIt's `q.cut`, the
# subclass of every unit (null where MatchIt leaves a unit without one), the
# weights MatchIt returns, and whether MatchIt stopped with an error, which it
# does when a group has fewer than `subclass * min.n` units.
#
# Data:
#   lalonde, sim_discrete  glm distance; the second has 24 distinct scores, so
#                          quantiles fall on tied scores
#   random_*               30 small random data sets, scores rounded to 1 or 2
#                          digits so that units tie at the cutpoints, with the
#                          groups shifted so that some subclasses start with
#                          too few units of one group
# Settings: estimand ATT, ATC and ATE; subclass 2 to 10; min.n 0, 1, 2 (the random sets) and 0, 1, 4 (the others).

suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

run_one <- function(treat, distance, estimand, subclass, min_n) {
  df <- data.frame(treat = treat, z = seq_along(treat))
  m <- tryCatch(
    suppressWarnings(matchit(treat ~ z, data = df, method = "subclass",
                             distance = distance, estimand = estimand,
                             subclass = subclass, min.n = min_n)),
    error = function(e) NULL)
  if (is.null(m)) {
    return(list(estimand = estimand, subclass = subclass, min_n = min_n,
                error = TRUE))
  }
  list(estimand = estimand, subclass = subclass, min_n = min_n, error = FALSE,
       q_cut = unname(m$q.cut),
       assignment = ifelse(is.na(m$subclass), NA, as.integer(m$subclass)),
       weights = unname(m$weights))
}

runs_for <- function(name, treat, distance, subclasses, min_ns) {
  runs <- list()
  for (estimand in c("ATT", "ATC", "ATE")) {
    for (k in subclasses) {
      for (min_n in min_ns) {
        runs[[length(runs) + 1]] <- run_one(treat, distance, estimand, k, min_n)
      }
    }
  }
  list(name = name, treat = treat, distance = distance, runs = runs)
}

data(lalonde)
lal <- data.frame(
  treat = lalonde$treat, age = lalonde$age, educ = lalonde$educ,
  black = as.numeric(lalonde$race == "black"),
  hispan = as.numeric(lalonde$race == "hispan"),
  married = lalonde$married, nodegree = lalonde$nodegree,
  re74 = lalonde$re74, re75 = lalonde$re75)
glm_distance <- function(df) {
  f <- as.formula(paste("treat ~", paste(setdiff(names(df), "treat"),
                                         collapse = " + ")))
  unname(matchit(f, data = df, method = NULL, distance = "glm")$distance)
}

set.seed(20261120)
n <- 400
d1 <- rbinom(n, 1, 0.5)
d2 <- sample(0:3, n, replace = TRUE)
d3 <- sample(0:2, n, replace = TRUE)
sim <- data.frame(
  treat = rbinom(n, 1, plogis(-0.5 + 0.9 * d1 + 0.5 * d2 - 0.4 * d3)),
  d1 = d1, d2 = d2, d3 = d3)

datasets <- list(
  runs_for("lalonde", lal$treat, glm_distance(lal), c(3, 6, 10), c(0, 1, 4)),
  runs_for("sim_discrete", sim$treat, glm_distance(sim), c(3, 6, 10), c(0, 1, 4)))

for (i in 1:30) {
  n_t <- sample(8:30, 1)
  n_c <- sample(8:40, 1)
  digits <- sample(1:2, 1)
  shift <- runif(1, 0, 0.5)
  zt <- pmin(1, pmax(0, round(rbeta(n_t, 2 + 3 * shift, 2), digits)))
  zc <- pmin(1, pmax(0, round(rbeta(n_c, 2, 2 + 3 * shift), digits)))
  # Interleave the groups so that the order of units within a group is the data
  # order, as in the Rust test.
  treat <- c(rep(1, n_t), rep(0, n_c))
  distance <- c(zt, zc)
  o <- sample(length(treat))
  datasets[[length(datasets) + 1]] <- runs_for(
    paste0("random_", i), treat[o], distance[o], sample(2:10, 2), c(0, 1, 2))
}

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  datasets = datasets)
write_json(out, "tests/data/matchit_subclass.json", digits = I(17),
           auto_unbox = TRUE, na = "null")
