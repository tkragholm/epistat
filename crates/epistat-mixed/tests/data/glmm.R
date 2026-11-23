# Reference generalised linear mixed models with a random intercept from
# lme4::glmer, and the Gauss-Hermite rules of lme4::GHrule.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript glmm.R > glmm.json
#
# Cases: cbpp (binomial with trials, 15 herds); simulated sibling pairs (1000
# clusters of 2 with a binary outcome and a large cluster variance); a
# simulated Poisson model with 300 unbalanced clusters in scrambled row order,
# an offset and prior weights; and a simulated binary model with no cluster
# variation, whose fit is singular. Each is fitted with nAGQ = 1 (Laplace) and
# with more nodes.
#
# glmer is run with bobyqa at rhoend 1e-12 and with tolPwrss 1e-13, so that
# neither the optimiser nor the inner PIRLS iteration sets the comparison.
# At the optimum and at two other points the script records the deviance
# function of lme4 (its stage 2, a function of theta and beta), the
# quadrature transcribed at lme4's modes, the conditional modes and their
# conditional variances theta^2 / (1 + theta^2 sum w b''), and the Hessian of
# the transcribed quadrature by numDeriv with Richardson extrapolation. Its
# first step is 3% of each parameter, with four halvings: smaller first steps
# amplify the rounding of the deviance, and these settings agree with steps
# of 1% to about 1e-10. At the optimum it also records what glmer reports
# (logLik, ranef with condVar, vcov) and the marginal log likelihood with each
# cluster's integral computed by integrate() at a relative tolerance of
# 1e-13, in pieces around the cluster's mode.
suppressMessages({
  library(lme4)
  library(numDeriv)
  library(jsonlite)
})

vec <- function(m) as.vector(t(as.matrix(m)))
ctl <- glmerControl(optimizer = "bobyqa", tolPwrss = 1e-13,
                    optCtrl = list(rhoend = 1e-12, maxfun = 1e5),
                    check.conv.grad = "ignore", check.conv.singular = "ignore",
                    check.conv.hess = "ignore")

# The log density of each row with all constants, at the linear predictor.
logf <- function(family, y, w, eta) {
  if (family == "binomial") {
    lchoose(w, round(w * y)) + w * (y * plogis(eta, log.p = TRUE) +
                                    (1 - y) * plogis(-eta, log.p = TRUE))
  } else {
    w * dpois(y, exp(eta), log = TRUE)
  }
}

saturated <- function(family, y, w) {
  if (family == "binomial") sum(dbinom(round(w * y), w, y, log = TRUE))
  else sum(w * dpois(y, y, log = TRUE))
}

exact_loglik <- function(family, X, y, w, o, g, theta, beta) {
  eta0 <- drop(X %*% beta) + o
  total <- 0
  for (cl in unique(g)) {
    ix <- which(g == cl)
    h <- function(u) sapply(u, function(v) sum(logf(family, y[ix], w[ix], eta0[ix] + theta * v)) + dnorm(v, log = TRUE))
    top <- optimize(h, c(-30, 30), maximum = TRUE, tol = 1e-10)
    m <- top$maximum
    cuts <- c(m - 40, m - 2, m - 0.5, m, m + 0.5, m + 2, m + 40)
    val <- 0
    for (j in seq_len(length(cuts) - 1)) {
      val <- val + integrate(function(u) exp(h(u) - top$objective), cuts[j], cuts[j + 1],
                             rel.tol = 1e-13, abs.tol = 0, subdivisions = 1000L)$value
    }
    total <- total + log(val) + top$objective
  }
  total
}

# The adaptive quadrature (Laplace for k = 1) at lme4's modes with the
# curvature at those modes, as -2 log L with the constants. lme4's own
# deviance takes the curvature, and so the scale of the nodes, from the PIRLS
# iterate before the last one, which makes it depend on where PIRLS started
# and rough at the scale of numDeriv's steps.
quad_deviance <- function(devfun, par, k, family, X, y, w, o, g) {
  devfun(par)
  u <- environment(devfun)$pp$u(1)
  ids <- sort(unique(g))
  at <- match(g, ids)
  eta0 <- drop(X %*% par[-1]) + o
  eta <- eta0 + par[1] * u[at]
  mu <- if (family == "binomial") plogis(eta) else exp(eta)
  b2 <- if (family == "binomial") mu * (1 - mu) else mu
  a <- 1 + par[1]^2 * tapply(w * b2, at, sum)
  s <- 1 / sqrt(a)
  rule <- GHrule(k)
  total <- 0
  for (i in seq_along(ids)) {
    ix <- which(at == i)
    terms <- sapply(seq_len(nrow(rule)), function(j) {
      v <- u[i] + s[i] * rule[j, "z"]
      log(rule[j, "w"]) + rule[j, "z"]^2 / 2 +
        sum(logf(family, y[ix], w[ix], eta0[ix] + par[1] * v)) - v^2 / 2
    })
    top <- max(terms)
    total <- total + log(s[i]) + top + log(sum(exp(terms - top)))
  }
  -2 * total
}

state_at <- function(devfun, par, family, X, y, w, o, g, k) {
  dev <- devfun(par)
  e <- environment(devfun)
  u <- e$pp$u(1)
  mu <- e$resp$mu
  b2 <- if (family == "binomial") mu * (1 - mu) else mu
  a <- 1 + par[1]^2 * tapply(w * b2, g, sum)
  f <- function(p) quad_deviance(devfun, p, k, family, X, y, w, o, g)
  transcribed <- -f(par) / 2
  H <- hessian(f, par, method.args = list(d = 0.03, r = 4))
  list(par = par, deviance = dev, transcribed = transcribed, modes = par[1] * u,
       mode_var = unname(par[1]^2 / a), hessian = vec(H))
}

run <- function(name, formula, d, family, nodes, X, y, w, o, g) {
  fam <- if (family == "binomial") binomial() else poisson()
  fits <- list()
  for (k in nodes) {
    fit <- glmer(formula, d, family = fam, weights = w, offset = o, nAGQ = k, control = ctl)
    devfun <- glmer(formula, d, family = fam, weights = w, offset = o, nAGQ = k,
                    control = ctl, devFunOnly = TRUE)
    th <- unname(getME(fit, "theta"))
    be <- unname(fixef(fit))
    par <- c(th, be)
    points <- list(par, c(th * 0.8 + 0.05, be + 0.05), c(th * 1.25 + 0.1, be - 0.05))
    re <- ranef(fit, condVar = TRUE)[[1]]
    fits[[length(fits) + 1]] <- list(
      nodes = k,
      theta = th, beta = be,
      log_lik = as.numeric(logLik(fit)),
      singular = isSingular(fit),
      vcov = vec(vcov(fit)),
      modes = unname(re[, 1]),
      mode_var = as.vector(attr(re, "postVar")),
      exact_log_lik = exact_loglik(family, X, y, w, o, g, th, be),
      at = lapply(points, function(p) state_at(devfun, p, family, X, y, w, o, g, k)))
  }
  list(name = name, family = family, saturated = saturated(family, y, w),
       x = unname(lapply(seq_len(nrow(X)), function(i) unname(X[i, ]))),
       y = y, cluster = g, weights = w, offset = o, fits = fits)
}

cases <- list()

data(cbpp)
d <- data.frame(y = cbpp$incidence / cbpp$size, w = cbpp$size, o = 0,
                period = cbpp$period, g = factor(as.integer(as.character(cbpp$herd))))
X <- model.matrix(~ period, d)
cases[[1]] <- run("cbpp", y ~ period + (1 | g), d, "binomial", c(1, 5, 15),
                  X, d$y, d$w, d$o, as.integer(as.character(d$g)))

set.seed(20261124)
G <- 1000
g <- rep(seq_len(G), each = 2)
n <- length(g)
x1 <- signif(rnorm(n), 10)
x2 <- rbinom(n, 1, 0.5)
b <- rep(1.5 * rnorm(G), each = 2)
y <- rbinom(n, 1, plogis(-0.5 + 0.6 * x1 - 0.4 * x2 + b))
d <- data.frame(y, x1, x2, w = 1, o = 0, g = factor(g))
X <- model.matrix(~ x1 + x2, d)
cases[[2]] <- run("sibling pairs", y ~ x1 + x2 + (1 | g), d, "binomial", c(1, 5, 15, 25),
                  X, d$y, d$w, d$o, g)

G <- 300
size <- sample(1:8, G, replace = TRUE)
g <- rep(seq_len(G), size)
n <- length(g)
x1 <- signif(rnorm(n), 10)
exposure <- signif(runif(n, 0.5, 3), 10)
o <- log(exposure)
w <- sample(1:3, n, replace = TRUE)
y <- rpois(n, exp(0.2 + 0.4 * x1 + o + rep(0.6 * rnorm(G), size)))
perm <- sample(n)
d <- data.frame(y, x1, w, o, g = factor(g))[perm, ]
X <- model.matrix(~ x1, d)
cases[[3]] <- run("poisson", y ~ x1 + (1 | g), d, "poisson", c(1, 9),
                  X, d$y, d$w, d$o, as.integer(as.character(d$g)))

# Independent outcomes, with seeds tried until glmer puts theta at 0.
for (seed in 1:200) {
  set.seed(seed)
  G <- 60
  g <- rep(seq_len(G), each = 4)
  n <- length(g)
  x1 <- signif(rnorm(n), 10)
  y <- rbinom(n, 1, plogis(-0.3 + 0.5 * x1))
  d <- data.frame(y, x1, w = 1, o = 0, g = factor(g))
  fit <- glmer(y ~ x1 + (1 | g), d, family = binomial, control = ctl)
  if (getME(fit, "theta") == 0) break
}
X <- model.matrix(~ x1, d)
cases[[4]] <- run("singular", y ~ x1 + (1 | g), d, "binomial", c(1),
                  X, d$y, d$w, d$o, g)

rules <- lapply(c(1, 2, 5, 15, 25), function(k) {
  r <- GHrule(k)
  list(nodes = k, z = r[, "z"], w = r[, "w"])
})

versions <- list(r = R.version.string, lme4 = as.character(packageVersion("lme4")),
                 numderiv = as.character(packageVersion("numDeriv")))
cat(toJSON(list(versions = versions, rules = rules, cases = cases), digits = NA,
           auto_unbox = TRUE, null = "null", na = "null"))
