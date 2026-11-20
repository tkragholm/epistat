# Reference propensity scores from MatchIt's matchit(distance = "glm"), on the
# probability scale (link = "logit") and on the linear predictor scale
# (link = "linear.logit").
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_propensity.R
#
# Three data sets: MatchIt's lalonde (raw scale, covariates up to 6e4), a
# simulated data set with continuous and binary covariates, and a simulated
# data set of three discrete covariates, so that many units share one score.
#
# The fit uses glm's convergence epsilon of 1e-14 so that R's own error is far
# below the comparison tolerance. For each fit the script also writes the first
# order shift of the linear predictor that the Rust fit's L2 penalty causes:
# the penalised solution minimises -loglik + (l2 / 2) * |b|^2 over the
# non-intercept coefficients, so it differs from the maximum likelihood
# solution by d = -(X'WX + l2 P)^-1 (l2 P b), with P the identity except for a
# zero at the intercept, and the linear predictor by X d.

suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

L2 <- 1e-6  # LogisticRegressionConfig::default().l2_penalty

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

fit_one <- function(name, df) {
  vars <- setdiff(names(df), "treat")
  f <- as.formula(paste("treat ~", paste(vars, collapse = " + ")))
  opts <- list(control = glm.control(epsilon = 1e-14, maxit = 100))
  p <- matchit(f, data = df, method = NULL, distance = "glm", link = "logit",
               distance.options = opts)
  l <- matchit(f, data = df, method = NULL, distance = "glm",
               link = "linear.logit", distance.options = opts)
  coef <- coef(p$model)
  x <- model.matrix(f, df)
  pr <- unname(p$distance)
  w <- pr * (1 - pr)
  pen <- diag(c(0, rep(1, ncol(x) - 1)))
  hess <- crossprod(x, x * w) + L2 * pen
  delta <- -solve(hess, L2 * pen %*% coef)
  list(name = name, treat = df$treat,
       covariates = lapply(vars, function(v) df[[v]]),
       covariate_names = vars,
       probability = pr, linear = unname(l$distance),
       coefficient_names = names(coef), coefficients = unname(coef),
       linear_link_coefficients = unname(coef(l$model)),
       penalty_shift_linear = abs(as.vector(x %*% delta)),
       penalty_shift_coefficients = abs(as.vector(delta)))
}

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  l2_penalty = L2,
  fits = list(fit_one("lalonde", lal), fit_one("sim_continuous", sim_cont),
              fit_one("sim_discrete", sim_discrete)))
write_json(out, "tests/data/matchit_propensity.json", digits = NA,
           auto_unbox = TRUE)
