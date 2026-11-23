# Inverse probability weights for a treatment that changes over time, from a pooled
# logistic model of treatment on person-period data, checked against
# ipw::ipwtm(type = "first" and "all", family = "binomial") and against weights
# built from glm and cumprod directly, with the variance of a weighted Cox fit that
# accounts for the estimated models, checked against the infinitesimal jackknife
# of the whole procedure found by numerical derivatives.
#
# Data: one row per subject and interval, a baseline covariate x1, a confounder l
# that is affected by the earlier treatment and affects the next one, and an
# event indicator. "all": the treatment is a Bernoulli variable in each interval.
# "first": the treatment is absorbing and rows continue after it starts.
# The procedure of the variance is: fit the models of treatment with case weights,
# form the weights, truncate them at their own quantiles (type 7, over every row,
# not weighted by the case weights, as ipwtm), so that the limits move with the
# fits, and fit the Cox model with the weights times the case weights. The influence of a subject is
# the derivative of the coefficients with respect to its case weight at 1, by
# central differences with Richardson extrapolation (steps h and h / 2).
# Run: Rscript weighting_treatment.R > weighting_treatment.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(ipw)
})
set.seed(20261123)
sim <- function(n, absorbing) {
  rows <- list()
  for (i in seq_len(n)) {
    x1 <- round(rnorm(1), 3)
    l <- round(rnorm(1), 3)
    aprev <- 0
    for (k in 1:8) {
      l <- round(0.5 * l + rnorm(1, 0, 0.8) + 0.3 * aprev, 3)
      pa <- plogis(-1.4 + 0.7 * l + 0.3 * x1 + (if (absorbing) 0 else 1.2 * aprev))
      a <- if (absorbing && aprev == 1) 1 else rbinom(1, 1, pa)
      ev <- rbinom(1, 1, plogis(-3.3 + 0.4 * l - 0.6 * a + 0.3 * x1))
      rows[[length(rows) + 1]] <- data.frame(id = i, k = k, x1 = x1, l = l, a = a, event = ev,
                                             start = k - 1, stop = k)
      aprev <- a
      if (ev == 1) break
    }
  }
  do.call(rbind, rows)
}
fden <- ~ x1 + l + k
fnum <- ~ x1 + k
ctl <- coxph.control(eps = 1e-13, toler.chol = 1e-14, iter.max = 100)
gctl <- glm.control(epsilon = 1e-13, maxit = 100)

# Weights by hand from glm, for case weights om per row (1 for the plain weights);
# the fits stop at glm's default tolerance as ipwtm's do, or at 1e-13 when tight.
hand <- function(dat, type, stab, om = rep(1, nrow(dat)), lim = NULL, trunc = NULL, tight = FALSE) {
  gc <- if (tight) gctl else glm.control()
  N <- nrow(dat)
  sel <- if (type == "all") rep(TRUE, N) else
    ave(dat$a, dat$id, FUN = function(x) c(0, cumsum(x)[-length(x)])) == 0
  dd <- dat; dd$om <- om
  md <- glm(update(fden, a ~ .), data = dd[sel, ], family = binomial, weights = om, control = gc)
  pd <- predict(md, newdata = dat, type = "response")
  fd <- ifelse(sel, ifelse(dat$a == 1, pd, 1 - pd), 1)
  fn <- rep(1, N); pn <- NA; cn <- NULL
  if (stab) {
    mn <- glm(update(fnum, a ~ .), data = dd[sel, ], family = binomial, weights = om, control = gc)
    pn <- predict(mn, newdata = dat, type = "response")
    fn <- ifelse(sel, ifelse(dat$a == 1, pn, 1 - pn), 1)
    cn <- unname(coef(mn))
  }
  f <- fn / fd
  w <- ave(f, dat$id, FUN = cumprod)
  wb <- w / f
  if (!is.null(trunc) && is.null(lim)) lim <- quantile(w, c(trunc, 1 - trunc), names = FALSE)
  wt <- if (is.null(lim)) w else pmin(pmax(w, lim[1]), lim[2])
  wbt <- if (is.null(lim)) wb else pmin(pmax(wb, lim[1]), lim[2])
  kap <- function(f0, p) {
    X <- unname(model.matrix(f0, dat[sel, ]))
    g <- crossprod(X * sqrt((p * (1 - p))[sel]))
    s <- sqrt(diag(g))
    kappa(diag(1 / s) %*% g %*% diag(1 / s), exact = TRUE)
  }
  list(weights = unname(w), weights_before = unname(wb), truncated = unname(wt),
       truncated_before = unname(wbt), limits = lim, active = as.integer(sel),
       probability = unname(pd), coefficients = unname(coef(md)),
       numerator_probability = if (stab) unname(pn) else NULL, numerator_coefficients = cn,
       kappa_den = kap(fden, pd), kappa_num = if (stab) kap(fnum, pn) else NULL)
}

case <- function(name, dat, type, stab, trunc = NULL, h = 1e-4) {
  t0 <- proc.time()[3]
  N <- nrow(dat)
  ip <- ipwtm(exposure = a, family = "binomial", link = "logit",
              numerator = if (stab) fnum else NULL, denominator = fden,
              id = id, timevar = k, type = type, data = dat, trunc = trunc)
  h0 <- hand(dat, type, stab, trunc = trunc)
  n <- max(dat$id)
  one <- rep(1, n)
  outcome <- function(w, om_sub) {
    dd <- dat; dd$ww <- w * om_sub[dat$id]
    unname(coef(do.call(coxph, list(Surv(start, stop, event) ~ a + x1, data = dd, weights = quote(ww),
                                    ties = "efron", control = ctl))))
  }
  wfun <- function(om_sub) {
    hh <- hand(dat, type, stab, om = om_sub[dat$id], trunc = trunc, tight = TRUE)
    if (is.null(trunc)) hh$weights else hh$truncated
  }
  w0 <- wfun(one)
  est0 <- outcome(w0, one)
  fdiff <- function(g) {
    d <- function(hh) (g(1 + hh) - g(1 - hh)) / (2 * hh)
    d1 <- d(h); d2 <- d(h / 2)
    list(rich = (4 * d2 - d1) / 3, plain = d1)
  }
  m <- length(est0)
  ia <- list(rich = matrix(0, n, m), plain = matrix(0, n, m))
  ik <- list(rich = matrix(0, n, m), plain = matrix(0, n, m))
  for (u in seq_len(n)) {
    pert <- function(x) { o <- one; o[u] <- x; o }
    ra <- fdiff(function(x) outcome(wfun(pert(x)), pert(x)))
    rk <- fdiff(function(x) outcome(w0, pert(x)))
    ia$rich[u, ] <- ra$rich; ia$plain[u, ] <- ra$plain
    ik$rich[u, ] <- rk$rich; ik$plain[u, ] <- rk$plain
  }
  cat(name, "done in", proc.time()[3] - t0, "s\n", file = stderr())
  c(list(name = name, type = type, stabilise = stab, trunc = trunc,
         id = dat$id - 1, start = dat$start, stop = dat$stop, status = dat$event,
         treated = dat$a, x1 = dat$x1,
         design = unname(model.matrix(fden, dat)),
         numerator_design = if (stab) unname(model.matrix(fnum, dat)) else NULL,
         ipw_weights = unname(ip$ipw.weights),
         ipw_truncated = if (is.null(trunc)) NULL else unname(ip$weights.trunc),
         estimate = est0,
         covariance = crossprod(ia$rich), covariance_plain = crossprod(ia$plain),
         known = crossprod(ik$rich), known_plain = crossprod(ik$plain)), h0)
}
d_all <- sim(150, FALSE)
d_first <- sim(150, TRUE)
out <- list(
  versions = list(r = R.version.string, ipw = as.character(packageVersion("ipw")),
                  survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  cases = list(
    case("first_stab", d_first, "first", TRUE),
    case("first_unstab_trunc", d_first, "first", FALSE, trunc = 0.02),
    case("all_stab_trunc", d_all, "all", TRUE, trunc = 0.05),
    case("all_unstab", d_all, "all", FALSE)
  )
)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
