# Reference propensity scores from MatchIt's matchit(distance = "glm"): the
# logit, probit and cloglog links on the probability scale (link = "logit",
# "probit", "cloglog") and on the linear predictor scale (link = "linear.logit",
# "linear.probit", "linear.cloglog"), with sampling weights (s.weights), and
# with common-support trimming and re-estimation (discard, reestimate = TRUE).
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_propensity.R
#
# Four data sets: MatchIt's lalonde (raw scale, covariates up to 6e4), a
# simulated data set with continuous and binary covariates, a simulated data
# set of three discrete covariates, so that many units share one score, and a
# simulated data set whose groups barely overlap, so that trimming removes many
# units.
#
# MatchIt's default glm control is used, as MatchIt calls it: epsilon 1e-8 and
# 25 iterations. A re-estimated fit keeps the first fit's distance for the
# units it discards, which the script checks. The Rust fit follows the same iteration, so the two agree to
# the rounding of the linear solves. For each fit the script writes the
# condition number of sqrt(W) X at the solution with its columns scaled to unit
# length, which sets the size of that rounding: the normal equations are solved
# by a Cholesky factorisation, whose rounding does not depend on the scale of
# a column. It also writes the length of each column of sqrt(W) X, which scales
# the rounding of the matching coefficient.

suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

# Condition number of sqrt(w) X with every column scaled to unit length.
scaled_condition <- function(x, w) {
  u <- sqrt(w) * x
  kappa(sweep(u, 2, sqrt(colSums(u^2)), "/"), exact = TRUE)
}

data(lalonde)
lal <- data.frame(
  treat = lalonde$treat, age = lalonde$age, educ = lalonde$educ,
  black = as.numeric(lalonde$race == "black"),
  hispan = as.numeric(lalonde$race == "hispan"),
  married = lalonde$married, nodegree = lalonde$nodegree,
  re74 = lalonde$re74, re75 = lalonde$re75)

set.seed(20261120)
n <- 300
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.4)
x3 <- round(runif(n, 18, 60))
sim_cont <- data.frame(
  treat = rbinom(n, 1, plogis(-0.4 + 0.8 * x1 + 0.6 * x2 - 0.03 * (x3 - 40))),
  x1 = x1, x2 = x2, x3 = x3)

n <- 400
d1 <- rbinom(n, 1, 0.5)
d2 <- sample(0:3, n, replace = TRUE)
d3 <- sample(0:2, n, replace = TRUE)
sim_discrete <- data.frame(
  treat = rbinom(n, 1, plogis(-0.5 + 0.9 * d1 + 0.5 * d2 - 0.4 * d3)),
  d1 = d1, d2 = d2, d3 = d3)

n <- 400
z1 <- rnorm(n)
z2 <- rbinom(n, 1, 0.5)
sim_overlap <- data.frame(
  treat = rbinom(n, 1, plogis(-1.2 + 1.6 * z1 + 0.5 * z2)),
  z1 = z1, z2 = z2)

links <- c("logit", "probit", "cloglog")

distance_of <- function(f, df, link, s.weights = NULL, ...) {
  m <- matchit(f, data = df, method = NULL, distance = "glm", link = link,
               s.weights = s.weights, ...)
  list(distance = unname(m$distance), model = m$model, discarded = m$discarded)
}

fit_one <- function(name, df) {
  vars <- setdiff(names(df), "treat")
  f <- as.formula(paste("treat ~", paste(vars, collapse = " + ")))
  set.seed(1)
  w <- round(runif(nrow(df), 0.4, 2.5), 3)
  x <- model.matrix(f, df)

  by_link <- lapply(links, function(link) {
    p <- distance_of(f, df, link)
    l <- distance_of(f, df, paste0("linear.", link))
    pw <- distance_of(f, df, link, s.weights = w)
    lw <- distance_of(f, df, paste0("linear.", link), s.weights = w)
    # Condition number of sqrt(W) X at the solution, from the working weights.
    cond <- function(model) scaled_condition(x, model$weights)
    norms <- function(model) unname(sqrt(colSums((sqrt(model$weights) * x)^2)))
    list(link = link,
         probability = p$distance, linear = l$distance,
         coefficient_names = names(coef(p$model)),
         coefficients = unname(coef(p$model)),
         iterations = p$model$iter, converged = p$model$converged,
         condition = cond(p$model), column_norms = norms(p$model),
         weighted_probability = pw$distance, weighted_linear = lw$distance,
         weighted_coefficients = unname(coef(pw$model)),
         weighted_iterations = pw$model$iter,
         weighted_condition = cond(pw$model),
         weighted_column_norms = norms(pw$model))
  })

  trimmed <- lapply(c("treated", "control", "both"), function(policy) {
    one <- function(link, s.weights = NULL) {
      r <- distance_of(f, df, link, s.weights = s.weights, discard = policy,
                       reestimate = TRUE)
      first <- distance_of(f, df, link, s.weights = s.weights, discard = policy)
      stopifnot(all(abs(r$distance - first$distance)[r$discarded] == 0))
      list(distance = r$distance,
           condition = scaled_condition(x[!r$discarded, ], r$model$weights),
           discarded = as.logical(r$discarded),
           same_discarded = identical(r$discarded, first$discarded))
    }
    list(policy = policy,
         probability = one("logit"),
         linear_probit = one("linear.probit"),
         weighted = one("logit", w))
  })

  list(name = name, treat = df$treat,
       covariates = lapply(vars, function(v) df[[v]]),
       covariate_names = vars, weights = w,
       links = by_link, trimmed = trimmed)
}

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  fits = list(fit_one("lalonde", lal), fit_one("sim_continuous", sim_cont),
              fit_one("sim_discrete", sim_discrete),
              fit_one("sim_overlap", sim_overlap)))
write_json(out, "tests/data/matchit_propensity.json", digits = I(17),
           auto_unbox = TRUE)
