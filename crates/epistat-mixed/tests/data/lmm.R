# Reference linear mixed models from lme4::lmer.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript lmm.R > lmm.json
#
# Cases: sleepstudy with a random intercept and slope on Days and with a
# random intercept only; a simulated random-intercept model with 1000
# unbalanced clusters, prior weights and an offset; a simulated model with a
# correlated random slope, 300 unbalanced clusters and prior weights; and a
# simulated model with no cluster variation, where the fit is singular. Each
# is fitted by REML and ML.
#
# lmer is run with BOBYQA (nloptwrap) at xtol_abs 1e-14 and no relative
# tolerances, so that its optimiser stops much closer to the optimum than at
# its defaults. At the optimum and at two other values of theta the script
# records what lme4 reports for a model whose optimiser returned that theta
# (mkMerMod): the criterion, fixed effects, sigma, vcov, VarCorr, the
# conditional modes and their conditional covariances.
suppressMessages({
  library(lme4)
  library(jsonlite)
})

vec <- function(m) as.vector(t(as.matrix(m)))
ctl <- lmerControl(optimizer = "nloptwrap",
                   optCtrl = list(xtol_abs = 1e-14, ftol_abs = 1e-15, xtol_rel = 0,
                                  ftol_rel = 0, maxeval = 1e5),
                   check.conv.singular = "ignore")

describe <- function(m) {
  re <- ranef(m, condVar = TRUE)[[1]]
  pv <- attr(re, "postVar")
  q <- ncol(re)
  list(theta = unname(getME(m, "theta")),
       criterion = if (isREML(m)) REMLcrit(m) else deviance(m),
       log_lik = as.numeric(logLik(m)),
       beta = unname(fixef(m)),
       sigma = sigma(m),
       vcov = vec(vcov(m)),
       random_cov = vec(VarCorr(m)[[1]]),
       modes = vec(as.matrix(re)),
       mode_cov = as.vector(unlist(lapply(seq_len(nrow(re)), function(i) vec(matrix(pv[, , i], q, q))))))
}

at_theta <- function(lf, theta) {
  dfun <- do.call(mkLmerDevfun, lf)
  val <- dfun(theta)
  m <- mkMerMod(environment(dfun), list(par = theta, fval = val, conv = 0, message = ""),
                lf$reTrms, fr = lf$fr)
  describe(m)
}

run <- function(name, formula, d, slope) {
  out <- list()
  for (reml in c(TRUE, FALSE)) {
    fit <- lmer(formula, d, REML = reml, weights = w, offset = o, control = ctl)
    th <- unname(getME(fit, "theta"))
    lf <- lFormula(formula, d, REML = reml, weights = w, offset = o)
    others <- list(th * 0.7 + 0.05, th * 1.3 + 0.1)
    out[[if (reml) "reml" else "ml"]] <- list(
      fit = describe(fit),
      singular = isSingular(fit),
      at = lapply(others, function(t) at_theta(lf, t)))
  }
  X <- model.matrix(reformulas::nobars(formula), d)
  list(name = name, x = unname(lapply(seq_len(nrow(X)), function(i) unname(X[i, ]))),
       y = d$y, cluster = as.integer(as.character(d$g)),
       slope = if (is.null(slope)) NULL else d[[slope]],
       weights = d$w, offset = d$o, fits = out)
}

cases <- list()

data(sleepstudy)
ss <- data.frame(y = sleepstudy$Reaction, days = sleepstudy$Days,
                 g = factor(as.integer(as.character(sleepstudy$Subject))), w = 1, o = 0)
cases[[1]] <- run("sleepstudy slope", y ~ days + (days | g), ss, "days")
cases[[2]] <- run("sleepstudy intercept", y ~ days + (1 | g), ss, NULL)

set.seed(20261123)
G <- 1000
size <- sample(1:6, G, replace = TRUE)
g <- rep(seq_len(G), size)
n <- length(g)
x1 <- signif(rnorm(n), 10)
x2 <- rbinom(n, 1, 0.4)
x3 <- signif(rep(rnorm(G), size), 10)
w <- signif(runif(n, 0.5, 2), 10)
o <- signif(0.3 * rnorm(n), 10)
y <- signif(1 + 0.5 * x1 - 0.4 * x2 + 0.3 * x3 + o + rep(0.8 * rnorm(G), size) +
            1.2 * rnorm(n) / sqrt(w), 10)
# Rows in a scrambled order, so the fit has to group them.
perm <- sample(n)
sim1 <- data.frame(y, x1, x2, x3, g = factor(g), w, o)[perm, ]
cases[[3]] <- run("simulated intercept", y ~ x1 + x2 + x3 + (1 | g), sim1, NULL)

G <- 300
size <- sample(2:10, G, replace = TRUE)
g <- rep(seq_len(G), size)
n <- length(g)
t <- signif(unlist(lapply(size, function(m) sort(runif(m, 0, 5)))), 10)
x1 <- signif(rnorm(n), 10)
b0 <- rnorm(G)
b1 <- 0.4 * b0 + 0.3 * rnorm(G)
w <- signif(runif(n, 0.5, 2), 10)
y <- signif(2 + 0.3 * t - 0.5 * x1 + rep(b0, size) + rep(b1, size) * t +
            0.7 * rnorm(n) / sqrt(w), 10)
sim2 <- data.frame(y, t, x1, g = factor(g), w, o = 0)
cases[[4]] <- run("simulated slope", y ~ t + x1 + (t | g), sim2, "t")

G <- 40
size <- sample(3:6, G, replace = TRUE)
g <- rep(seq_len(G), size)
n <- length(g)
x1 <- signif(rnorm(n), 10)
# Cluster means pulled toward the grand mean, so the between-cluster
# variance is below what the residual variance implies and REML and ML put
# the random intercept at 0.
e <- rnorm(n)
e <- e - 0.8 * ave(e, g)
y <- signif(0.5 + 0.2 * x1 + e, 10)
sim3 <- data.frame(y, x1, g = factor(g), w = 1, o = 0)
cases[[5]] <- run("singular", y ~ x1 + (1 | g), sim3, NULL)

versions <- list(r = R.version.string, lme4 = as.character(packageVersion("lme4")),
                 matrix = as.character(packageVersion("Matrix")),
                 nloptr = as.character(packageVersion("nloptr")))
cat(toJSON(list(versions = versions, cases = cases), digits = NA, auto_unbox = TRUE,
           null = "null", na = "null"))
