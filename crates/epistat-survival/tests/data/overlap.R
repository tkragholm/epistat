# Overlap, matching and optimal-subset weights, checked against WeightIt
# (weightit with estimand ATO, ATM, ATOS) and PSweight (SumStat, overlap and
# matching weights, which it normalises to sum to one within each group), on the
# lalonde data of cobalt with simulated survival times.
#
# weights: for each estimand the weights, fitted probabilities and coefficients of
# WeightIt, the summary of the weights, the exact balance of the weighted
# covariate sums, and for ATOS the cut-off alpha of get_w_from_ps.
# cox: the covariance of a weighted Cox fit whose weights are estimated, from
# derivatives found by central differences with Richardson extrapolation of
# refits of glm and coxph, as in weighting.R, so that no formula of the Rust code
# is shared with the reference. The cut-off alpha of ATOS and the kink of ATM
# at one half are held fixed.
# The versions are in the file. Numbers are written with 17 digits.
# Run: Rscript overlap.R > overlap.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(WeightIt)
  library(PSweight)
})
data("lalonde", package = "cobalt")
set.seed(20261122)
n <- nrow(lalonde)
d <- lalonde
d$time <- ceiling(rexp(n, 0.02 * exp(0.5 * (d$re74 > 0) - 0.3 * d$treat + 0.03 * (d$age - 27))))
cens <- ceiling(runif(n, 20, 120))
d$status <- as.integer(d$time <= cens)
d$t <- pmin(d$time, cens)
f <- treat ~ age + educ + race + married + nodegree + re74 + re75
Z <- unname(model.matrix(f, d))
a <- d$treat

summ <- function(w) {
  list(n = length(w), sum = sum(w), min = min(w), max = max(w), mean = mean(w),
       cv = sd(w) / mean(w), ess = sum(w)^2 / sum(w^2))
}
kappa_ps <- function(p) {
  g <- crossprod(Z * sqrt(p * (1 - p)))
  s <- sqrt(diag(g))
  kappa(diag(1 / s) %*% g %*% diag(1 / s), exact = TRUE)
}
ss <- SumStat(f, data = d, weight = c("overlap", "matching"))
weights <- list()
wobj <- list()
for (e in c("ATO", "ATM", "ATOS")) {
  W <- weightit(f, data = d, method = "glm", estimand = e)
  wobj[[e]] <- W
  p <- W$ps
  gw <- get_w_from_ps(p, a, estimand = e)
  entry <- list(
    name = tolower(e), estimand = e,
    weights = unname(W$weights), propensity = unname(p),
    coefficients = unname(coef(glm(f, data = d, family = binomial))),
    treated = summ(W$weights[a == 1]), control = summ(W$weights[a == 0]),
    kappa = kappa_ps(p), alpha = attr(gw, "alpha"),
    balance_gap = max(abs(colSums(W$weights * a * Z) - colSums(W$weights * (1 - a) * Z)))
  )
  stopifnot(max(abs(gw - W$weights)) == 0)
  if (e == "ATO") entry$psweight <- unname(ss$ps.weights$overlap)
  if (e == "ATM") entry$psweight <- unname(ss$ps.weights$matching)
  weights[[length(weights) + 1]] <- entry
}

fd <- function(f, theta, i, h) {
  at <- function(hh) {
    up <- theta; dn <- theta; up[i] <- up[i] + hh; dn[i] <- dn[i] - hh
    (f(up) - f(dn)) / (2 * hh)
  }
  d1 <- at(h); d2 <- at(h / 2)
  list(rich = (4 * d2 - d1) / 3, plain = d1)
}
alpha0 <- weights[[3]]$alpha
weights_from <- function(theta, estimand) {
  p <- plogis(drop(Z %*% theta))
  switch(estimand,
    ATO = ifelse(a == 1, 1 - p, p),
    ATM = ifelse(a == 1, ifelse(p > 0.5, (1 - p) / p, 1), ifelse(p < 0.5, p / (1 - p), 1)),
    ATOS = ifelse(p < alpha0 | p > 1 - alpha0, 0, ifelse(a == 1, 1 / p, 1 / (1 - p))))
}
hscale <- apply(abs(Z), 2, max)
ctl_tight <- coxph.control(eps = 1e-13, toler.chol = 1e-14, iter.max = 100)
rows <- data.frame(subject = seq_len(n), stop = d$t, status = d$status, treat = a, age = d$age)
covs <- c("treat", "age")
fml <- as.formula("Surv(stop, status) ~ treat + age")
ps_if <- function(p) {
  A <- -crossprod(Z * sqrt(p * (1 - p)))
  -t(solve(A, t(Z * (a - p))))
}
cox_fd <- function(estimand, h = 1e-4) {
  theta0 <- unname(coef(glm(f, data = d, family = binomial)))
  # Rows of weight 0 are left out, as they do not enter the fit.
  fitw <- function(w) {
    keep <- w > 0
    do.call(coxph, list(fml, data = rows[keep, ], weights = w[keep], ties = "efron",
                        control = ctl_tight))
  }
  w0 <- weights_from(theta0, estimand)
  beta0 <- coef(fitw(w0))
  p <- plogis(drop(Z %*% theta0))
  ifs <- ps_if(p)
  k <- length(theta0)
  db <- function(extrap) {
    sapply(seq_len(n), function(u) {
      # the case weight of subject u multiplies its weight
      g <- function(om) { w <- w0; w[u] <- w0[u] * om; coef(fitw(w)) }
      up <- function(x) (g(1 + x) - g(1 - x)) / (2 * x)
      d1 <- up(h); d2 <- up(h / 2)
      if (extrap) (4 * d2 - d1) / 3 else d1
    })
  }
  dba <- function(extrap) {
    sapply(seq_len(k), function(i) {
      r <- fd(function(th) coef(fitw(weights_from(th, estimand))), theta0, i, h / hscale[i])
      if (extrap) r$rich else r$plain
    })
  }
  total <- function(extrap) {
    dbw <- t(matrix(db(extrap), nrow = length(covs)))
    dbt <- matrix(dba(extrap), nrow = length(covs))
    inf <- dbw + ifs %*% t(dbt)
    list(adj = crossprod(inf), known = crossprod(dbw))
  }
  r <- total(TRUE); q <- total(FALSE)
  list(estimand = estimand, coefficients = I(unname(beta0)),
       covariance = unname(r$adj), robust_covariance = unname(r$known),
       covariance_plain = unname(q$adj), robust_covariance_plain = unname(q$known))
}
cox <- lapply(c("ATO", "ATM", "ATOS"), cox_fd)
out <- list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  WeightIt = as.character(packageVersion("WeightIt")),
                  PSweight = as.character(packageVersion("PSweight")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  design = Z, treat = a, stop = d$t, status = d$status, age = d$age,
  weights = weights, cox = cox
)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
