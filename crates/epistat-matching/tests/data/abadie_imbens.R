# Reference Abadie-Imbens matching estimates from Matching::Match.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/abadie_imbens.R
#
# Two data sets: `lalonde` from MatchIt (age, educ, married, nodegree, re74;
# outcome re78; many tied distances) and a simulated cohort of 300 units with
# three continuous covariates and one binary (no ties). For each data set the
# script calls Match() with replacement and ties for every combination of
#   estimand  ATT, ATE, ATC
#   M         1, 3
#   BiasAdjust FALSE, TRUE
#   Var.calc  0, 2
#   Weight    1 (standardised Euclidean), 2 (Mahalanobis)
# and records the estimate, est.noadj, se (the population variance), se.cond
# and se.standard. A second call with sample = TRUE gives the standard error of
# the sample effect. The simulated cohort also runs the bias adjustment on a
# set of regressors Z that differs from the matching covariates.
library(Matching)
library(MatchIt)
library(jsonlite)

data(lalonde)
set.seed(20261121)
n <- 300
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rexp(n); x4 <- rbinom(n, 1, 0.4)
tr <- rbinom(n, 1, plogis(0.6 * x1 - 0.4 * x2 + 0.3 * x4 - 0.4))
y <- 1 + x1 + 0.5 * x2^2 + 0.3 * x3 + x4 + 2 * tr + rnorm(n)
extra <- x1 * x2

sets <- list(
  lalonde = list(
    X = as.matrix(lalonde[, c("age", "educ", "married", "nodegree", "re74")]),
    Tr = lalonde$treat, Y = lalonde$re78, Z = NULL),
  sim = list(X = cbind(x1, x2, x3, x4), Tr = tr, Y = y, Z = NULL))

run <- function(set, estimand, M, bias, vc, weight, Z = NULL) {
  args <- list(Y = set$Y, Tr = set$Tr, X = set$X, estimand = estimand, M = M,
               BiasAdjust = bias, Var.calc = vc, Weight = weight,
               replace = TRUE, ties = TRUE)
  if (!is.null(Z)) args$Z <- Z
  m <- suppressWarnings(do.call(Match, args))
  m2 <- suppressWarnings(do.call(Match, c(args, list(sample = TRUE))))
  list(estimand = estimand, M = M, bias = bias, var_calc = vc,
       weight = weight, z = !is.null(Z),
       est = as.numeric(m$est), est_noadj = as.numeric(m$est.noadj),
       se = as.numeric(m$se), se_sample = as.numeric(m2$se),
       se_cond = as.numeric(m$se.cond), se_standard = as.numeric(m$se.standard))
}

results <- list()
for (name in names(sets)) {
  set <- sets[[name]]
  for (estimand in c("ATT", "ATE", "ATC")) for (M in c(1, 3))
    for (bias in c(FALSE, TRUE)) for (vc in c(0, 2)) for (weight in c(1, 2)) {
      r <- run(set, estimand, M, bias, vc, weight)
      r$data <- name
      results[[length(results) + 1]] <- r
    }
}
for (estimand in c("ATT", "ATE")) for (vc in c(0, 2)) {
  r <- run(sets$sim, estimand, 2, TRUE, vc, 1, Z = cbind(x1, x2, x3, x4, extra))
  r$data <- "sim"
  results[[length(results) + 1]] <- r
}

out <- list(
  versions = list(r = R.version.string,
                  matching = as.character(packageVersion("Matching")),
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  lalonde = list(treat = lalonde$treat, re78 = lalonde$re78,
                 age = lalonde$age, educ = lalonde$educ,
                 married = lalonde$married, nodegree = lalonde$nodegree,
                 re74 = lalonde$re74),
  sim = list(treat = tr, y = y, x1 = x1, x2 = x2, x3 = x3, x4 = x4,
             extra = extra),
  results = results)
write_json(out, "tests/data/abadie_imbens.json", digits = I(17),
           auto_unbox = TRUE, na = "null")
