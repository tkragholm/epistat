# Augmented inverse probability weighting, checked against hand-written R.
#
# Every scenario fits glm models of the outcome in each arm and of the treatment,
# forms the augmented means, their difference and ratio for the estimand (ATE,
# ATT or ATC), with case weights and clusters, and gives:
# - the efficient variance, the sum of the squared influences of the formulas
#   written out here (clusters summed first, times G / (G - 1)), for the
#   predictions as given;
# - the stacked variance, the infinitesimal jackknife of the whole procedure (refit
#   the three glm models with case weights, form the means), found by central
#   differences of the case weight of each unit with Richardson extrapolation
#   (steps h and h / 2). No formula of the Rust code is shared with it.
# - the Lipschitz constants of the means with respect to the coefficients of the
#   models by numDeriv, and the condition numbers of the information matrices, for
#   the tolerances.
# riskRegression::ate(estimator = "AIPTW") is run for the binary outcome, ATE,
# unweighted, with and without known.nuisance.
# The versions are in the file. Numbers are written with 17 digits.
# Run: Rscript aipw.R > aipw.json
suppressMessages({
  library(jsonlite)
  library(riskRegression)
  library(numDeriv)
})
set.seed(20261124)
n <- 300
x1 <- round(rnorm(n), 3)
x2 <- rbinom(n, 1, 0.4)
a <- rbinom(n, 1, plogis(0.4 * x1 - 0.5 * x2 - 0.1))
yb <- rbinom(n, 1, plogis(-0.5 + 0.8 * x1 + 0.6 * x2 + 0.7 * a))
yc <- round(1 + 0.5 * x1 - 0.8 * x2 + 1.2 * a + rnorm(n), 4)
yp <- rpois(n, exp(0.2 + 0.3 * x1 - 0.2 * x2 + 0.4 * a))
w_user <- round(runif(n, 0.5, 2), 3)
cl <- (seq_len(n) - 1) %% 60
dat <- data.frame(a = a, x1 = x1, x2 = x2, yb = yb, yc = yc, yp = yp)
X <- unname(model.matrix(~ x1 + x2, dat))
Z <- X
gctl <- glm.control(epsilon = 1e-13, maxit = 100)  # the refits of the jackknife
H <- 1e-4

fams <- list(binary = binomial(), gaussian = gaussian(), poisson = poisson())
ycol <- list(binary = "yb", gaussian = "yc", poisson = "yp")

# The terms D, U1, U0 of the ratio estimators.
terms <- function(y, a, m1, m0, e, target) {
  switch(target,
    ATE = list(d = rep(1, length(y)),
               u1 = m1 + a * (y - m1) / e, u0 = m0 + (1 - a) * (y - m0) / (1 - e)),
    ATT = list(d = a, u1 = a * y, u0 = a * m0 + (1 - a) * e / (1 - e) * (y - m0)),
    ATC = list(d = 1 - a, u1 = (1 - a) * m1 + a * (1 - e) / e * (y - m1), u0 = (1 - a) * y))
}
means <- function(y, a, m1, m0, e, w, target) {
  t <- terms(y, a, m1, m0, e, target)
  s <- sum(w * t$d)
  c(sum(w * t$u1) / s, sum(w * t$u0) / s)
}
# Influences of the means with the predictions fixed.
efficient_if <- function(y, a, m1, m0, e, w, target) {
  t <- terms(y, a, m1, m0, e, target)
  s <- sum(w * t$d)
  th <- c(sum(w * t$u1) / s, sum(w * t$u0) / s)
  cbind(w * (t$u1 - th[1] * t$d) / s, w * (t$u0 - th[2] * t$d) / s)
}
unit_cov <- function(inf, cluster) {
  s <- if (is.null(cluster)) inf else rowsum(inf, cluster)
  g <- nrow(s)
  crossprod(s) * (if (is.null(cluster)) 1 else g / (g - 1))
}
summarise <- function(th, V) {
  # V is the covariance of (mean1, mean0)
  r <- th[1] / th[2]
  gr <- c(1 / th[2], -th[1] / th[2]^2)
  list(mean1 = th[1], mean0 = th[2], diff = th[1] - th[2], ratio = r,
       se_mean1 = sqrt(V[1, 1]), se_mean0 = sqrt(V[2, 2]),
       se_diff = sqrt(V[1, 1] + V[2, 2] - 2 * V[1, 2]),
       se_ratio = sqrt(drop(t(gr) %*% V %*% gr)))
}

scenario <- function(name, outcome, target, weighted, clustered, trunc = NULL) {
  t0 <- proc.time()[3]
  fam <- fams[[outcome]]
  y <- dat[[ycol[[outcome]]]]
  w <- if (weighted) w_user else rep(1, n)
  cluster <- if (clustered) cl else NULL
  # The procedure: models, predictions, means.
  # The fits stop at glm's default tolerance, as the fits of the Rust code do, or at
  # 1e-13 for the refits of the jackknife.
  fit_all <- function(om, tight = FALSE) {
    gc <- if (tight) gctl else glm.control()
    ww <- w * om
    dd <- cbind(dat, y = y, ww = ww)
    fp <- glm(a ~ x1 + x2, data = dd, family = binomial(), weights = ww, control = gc)
    f1 <- glm(y ~ x1 + x2, data = dd[a == 1, ], family = fam, weights = ww, control = gc)
    f0 <- glm(y ~ x1 + x2, data = dd[a == 0, ], family = fam, weights = ww, control = gc)
    list(fp = fp, f1 = f1, f0 = f0)
  }
  preds <- function(co) {
    e <- plogis(drop(Z %*% co$e))
    raw <- e
    clip <- rep(FALSE, n)
    if (!is.null(trunc)) { e0 <- e; e <- pmin(pmax(e, trunc[1]), trunc[2]); clip <- e != e0 }
    list(m1 = fam$linkinv(drop(X %*% co$b1)), m0 = fam$linkinv(drop(X %*% co$b0)), e = e, raw = raw, clip = clip)
  }
  coefs <- function(m) list(e = unname(coef(m$fp)), b1 = unname(coef(m$f1)), b0 = unname(coef(m$f0)))
  procedure <- function(om) {
    pr <- preds(coefs(fit_all(om, tight = TRUE)))
    means(y, a, pr$m1, pr$m0, pr$e, w * om, target)
  }
  # Unit case weights, one per person or per cluster.
  units <- if (clustered) cl else seq_len(n)
  ids <- sort(unique(units))
  base <- fit_all(rep(1, n))
  co <- coefs(base)
  pr <- preds(co)
  th <- means(y, a, pr$m1, pr$m0, pr$e, w, target)
  inf <- efficient_if(y, a, pr$m1, pr$m0, pr$e, w, target)
  Veff <- unit_cov(inf, cluster)
  # Lipschitz constants: the means as functions of the coefficients.
  theta_of <- function(p) {
    k <- length(co$e)
    cc <- list(e = p[1:k], b1 = p[k + 1:k], b0 = p[2 * k + 1:k])
    q <- preds(cc)
    means(y, a, q$m1, q$m0, q$e, w, target)
  }
  p0 <- c(co$e, co$b1, co$b0)
  jac <- jacobian(theta_of, p0)
  lip <- apply(abs(jac) %*% diag(pmax(abs(p0), 1)), 1, sum)
  info_kappa <- function(Xm, wt) {
    g <- crossprod(Xm * sqrt(wt)); s <- sqrt(diag(g))
    kappa(diag(1 / s) %*% g %*% diag(1 / s), exact = TRUE)
  }
  vf <- fam$variance
  kap <- max(info_kappa(Z, w * pr$e * (1 - pr$e)),
             info_kappa(X[a == 1, ], (w * vf(pr$m1))[a == 1]),
             info_kappa(X[a == 0, ], (w * vf(pr$m0))[a == 0]))
  fd <- function(g) {
    d <- function(hh) (g(1 + hh) - g(1 - hh)) / (2 * hh)
    d1 <- d(H); d2 <- d(H / 2)
    list(rich = (4 * d2 - d1) / 3, plain = d1)
  }
  ir <- matrix(0, length(ids), 2); ip <- ir
  for (u in seq_along(ids)) {
    mem <- units == ids[u]
    r <- fd(function(x) { om <- rep(1, n); om[mem] <- x; procedure(om) })
    ir[u, ] <- r$rich; ip[u, ] <- r$plain
  }
  gfac <- if (clustered) length(ids) / (length(ids) - 1) else 1
  Vst <- crossprod(ir) * gfac
  Vst_plain <- crossprod(ip) * gfac
  cat(name, "done in", proc.time()[3] - t0, "s\n", file = stderr())
  list(name = name, outcome = outcome, target = target, weighted = weighted,
       clustered = clustered, trunc = trunc, n_clipped = sum(pr$clip),
       y = y, weights = w, cluster = cluster, mu1 = pr$m1, mu0 = pr$m0, propensity = pr$raw,
       coefficients = list(e = co$e, b1 = co$b1, b0 = co$b0),
       efficient = summarise(th, Veff), stacked = summarise(th, Vst),
       stacked_plain = summarise(th, Vst_plain),
       n_units = length(ids), lipschitz = lip, kappa = kap)
}

scenarios <- list(
  scenario("binary_ate", "binary", "ATE", FALSE, FALSE),
  scenario("binary_att_weighted", "binary", "ATT", TRUE, FALSE),
  scenario("binary_atc_weighted", "binary", "ATC", TRUE, FALSE),
  scenario("gaussian_ate_clustered", "gaussian", "ATE", FALSE, TRUE),
  scenario("gaussian_att_clustered_weighted", "gaussian", "ATT", TRUE, TRUE),
  scenario("poisson_ate_truncated", "poisson", "ATE", FALSE, FALSE, trunc = c(0.25, 0.75)),
  scenario("gaussian_atc", "gaussian", "ATC", FALSE, FALSE)
)

# riskRegression::ate, binary outcome, ATE, unweighted.
dd <- data.frame(y = yb, a = factor(a), x1 = x1, x2 = x2)
fo <- glm(y ~ a + x1 + x2, data = dd, family = binomial)
fp <- glm(a ~ x1 + x2, data = dd, family = binomial)
rr <- function(known) {
  r <- ate(fo, treatment = fp, data = dd, estimator = "AIPTW", se = TRUE, verbose = FALSE,
           times = NA, known.nuisance = known)
  list(mean0 = r$meanRisk$estimate[1], mean1 = r$meanRisk$estimate[2],
       se_mean0 = r$meanRisk$se[1], se_mean1 = r$meanRisk$se[2],
       diff = r$diffRisk$estimate, se_diff = r$diffRisk$se,
       ratio = r$ratioRisk$estimate, se_ratio = r$ratioRisk$se)
}
# The outcome model of ate has the treatment as a main effect, one slope for both
# arms, so its predictions are given to the core as they are, and the efficient
# variance is compared with known.nuisance = TRUE.
dd1 <- dd; dd1$a <- factor(1, levels = c("0", "1"))
dd0 <- dd; dd0$a <- factor(0, levels = c("0", "1"))
out <- list(
  versions = list(r = R.version.string, riskRegression = as.character(packageVersion("riskRegression")),
                  numDeriv = as.character(packageVersion("numDeriv")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  x = X, scenarios = scenarios,
  riskregression = list(
    y = yb, a = a, mu1 = unname(predict(fo, newdata = dd1, type = "response")),
    mu0 = unname(predict(fo, newdata = dd0, type = "response")),
    propensity = unname(fitted(fp)),
    stacked = rr(FALSE), known = rr(TRUE)),
  treatment = a
)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
