# Targeted maximum likelihood estimation of the risk by a time with right
# censoring, in discrete time, written out by hand (survtmle, the package of this
# method, is no longer on CRAN).
#
# Data: n = 400 persons, two covariates, a binary treatment, follow-up in
# intervals 1 to 6, event and censoring hazards logistic in the interval, the
# covariates and the treatment. The nuisance models are glm fits: logistic
# regressions of the event and of censoring on the person-intervals at risk
# (interval as a factor, covariates, treatment) and of the treatment. Their
# predictions under each arm are the input of the estimator, which targets each
# arm's hazards along the clever covariate
#   H_k = -S(t) / (S(k) max(g_a G(k - 1), b))
# by logistic regressions with offset, iterated until the mean of the efficient
# influence function is at most se / (sqrt(n) ln n).
#
# Scenarios: horizons 4 and 6, with and without a bound, case weights and
# clusters. For the tolerances: the Lipschitz constants of every reported number
# with respect to the fluctuation coefficients by numDeriv (the iterations
# fixed), and the condition numbers of the fluctuation fits.
# The versions are in the file. Numbers are written with 17 digits.
# Run: Rscript tmle_survival.R > tmle_survival.json
suppressMessages({
  library(jsonlite)
  library(numDeriv)
})
set.seed(20261126)
n <- 400
K <- 6
x1 <- round(rnorm(n), 3)
x2 <- rbinom(n, 1, 0.4)
a <- rbinom(n, 1, plogis(0.5 * x1 - 0.4 * x2))
time <- integer(n); event <- logical(n)
for (i in seq_len(n)) {
  for (k in seq_len(K)) {
    if (runif(1) < plogis(-2.2 + 0.15 * k + 0.6 * x1[i] + 0.4 * x2[i] - 0.5 * a[i])) {
      time[i] <- k; event[i] <- TRUE; break
    }
    if (k == K || runif(1) < plogis(-2.6 + 0.3 * x1[i] + 0.3 * a[i])) {
      time[i] <- k; event[i] <- FALSE; break
    }
  }
}
w_user <- round(runif(n, 0.5, 2), 3)
cl <- sort(sample(0:49, n, replace = TRUE))

# Person-intervals: event rows for k <= time, censoring rows for k < time or
# k = time without the event.
pp <- do.call(rbind, lapply(seq_len(n), function(i) {
  k <- seq_len(time[i])
  data.frame(id = i, k = k, x1 = x1[i], x2 = x2[i], a = a[i],
             dn = as.integer(event[i] & k == time[i]),
             dc = as.integer(!event[i] & k == time[i] & time[i] < K),
             at_c = !(event[i] & k == time[i]) & k < K)
}))

predictions <- function(weights) {
  wpp <- weights[pp$id]
  fe <- glm(dn ~ factor(k, levels = 1:K) + x1 + x2 + a, family = binomial, data = pp, weights = wpp)
  fc <- glm(dc ~ factor(k, levels = 1:(K - 1)) + x1 + x2 + a, family = binomial,
            data = pp[pp$at_c, ], weights = wpp[pp$at_c])
  fg <- glm(a ~ x1 + x2, family = binomial, weights = weights)
  grid <- function(arm, fit, kmax) {
    m <- matrix(0, n, K)
    for (k in seq_len(kmax)) {
      nd <- data.frame(k = k, x1 = x1, x2 = x2, a = arm)
      m[, k] <- predict(fit, newdata = nd, type = "response")
    }
    m
  }
  list(h1 = grid(1, fe, K), h0 = grid(0, fe, K), c1 = grid(1, fc, K - 1),
       c0 = grid(0, fc, K - 1), g = unname(fitted(fg)))
}

our_var <- function(inf, cluster) {
  if (is.null(cluster)) return(sum(inf^2))
  s <- rowsum(inf, cluster); g <- nrow(s)
  sum(s^2) * g / (g - 1)
}

# The targeted estimator of S_a(t) for one arm. With `eps` the coefficients of
# the iterations are given instead of fitted.
target_arm <- function(arm, haz, cens, g, w, t, b, cluster, eps = NULL, max_iter = 50) {
  ga <- if (arm == 1) g else 1 - g
  lam <- haz[, 1:t, drop = FALSE]
  # G(k - 1) for k = 1..t
  G <- matrix(1, n, t)
  if (t > 1) for (k in 2:t) G[, k] <- G[, k - 1] * (1 - cens[, k - 1])
  den <- pmax(ga * G, b)
  at <- outer(time, 1:t, ">=") & (a == arm)
  dn <- outer(time, 1:t, "==") & event
  surv <- function(lam) t(apply(1 - lam, 1, cumprod))
  clever <- function(lam) { S <- surv(lam); -S[, t] / (S * den) }
  eif <- function(lam) {
    S <- surv(lam); H <- clever(lam)
    score <- rowSums(ifelse(at, H * (dn - lam), 0))
    psi <- sum(w * S[, t]) / sum(w)
    w * (score + S[, t] - psi) / sum(w)
  }
  iters <- 0; coefs <- c(); infos <- c()
  repeat {
    inf <- eif(lam)
    se <- sqrt(our_var(inf, cluster))
    score <- sum(w * rowSums(ifelse(at, clever(lam) * (dn - lam), 0))) / sum(w)
    done <- if (is.null(eps)) abs(score) <= se / (sqrt(n) * log(n)) else iters == length(eps)
    if (done) break
    if (iters == max_iter) stop("no convergence")
    H <- clever(lam)
    rows <- which(at)
    if (is.null(eps)) {
      f <- suppressWarnings(glm(as.numeric(dn[rows]) ~ -1 + H[rows] + offset(qlogis(lam[rows])),
                                family = binomial, weights = w[row(at)[rows]]))
      e <- unname(coef(f))
      infos <- c(infos, sum(w[row(at)[rows]] * H[rows]^2 * fitted(f) * (1 - fitted(f))))
    } else e <- eps[iters + 1]
    coefs <- c(coefs, e)
    lam <- plogis(qlogis(lam) + e * H)
    iters <- iters + 1
  }
  S <- surv(lam)
  list(psi = sum(w * S[, t]) / sum(w), inf = eif(lam), iters = iters, eps = coefs,
       score = score, infos = infos)
}

# The one-step (augmented) estimator on the initial hazards.
one_step <- function(arm, haz, cens, g, w, t, b) {
  ga <- if (arm == 1) g else 1 - g
  lam <- haz[, 1:t, drop = FALSE]
  G <- matrix(1, n, t)
  if (t > 1) for (k in 2:t) G[, k] <- G[, k - 1] * (1 - cens[, k - 1])
  den <- pmax(ga * G, b)
  at <- outer(time, 1:t, ">=") & (a == arm)
  dn <- outer(time, 1:t, "==") & event
  S <- t(apply(1 - lam, 1, cumprod))
  H <- -S[, t] / (S * den)
  sum(w * (S[, t] + rowSums(ifelse(at, H * (dn - lam), 0)))) / sum(w)
}

summ <- function(r1, r0, if1, if0, cluster) {
  r <- r1 / r0
  c(risk1 = r1, risk0 = r0, diff = r1 - r0, ratio = r,
    se_risk1 = sqrt(our_var(if1, cluster)), se_risk0 = sqrt(our_var(if0, cluster)),
    se_diff = sqrt(our_var(if1 - if0, cluster)),
    se_ratio = sqrt(our_var((if1 - r * if0) / r0, cluster)))
}

scenario <- function(name, t, b, weighted, clustered) {
  w0 <- if (weighted) w_user else rep(1, n)
  w <- w0 / sum(w0) * n
  cluster <- if (clustered) cl else NULL
  p <- suppressWarnings(predictions(w))
  run <- function(e1 = NULL, e0 = NULL) {
    t1 <- target_arm(1, p$h1, p$c1, p$g, w, t, b, cluster, eps = e1)
    t0 <- target_arm(0, p$h0, p$c0, p$g, w, t, b, cluster, eps = e0)
    list(t1 = t1, t0 = t0, numbers = summ(1 - t1$psi, 1 - t0$psi, -t1$inf, -t0$inf, cluster))
  }
  base <- run()
  e1 <- base$t1$eps; e0 <- base$t0$eps
  f <- function(e) run(e[seq_along(e1)], e[length(e1) + seq_along(e0)])$numbers
  e <- c(e1, e0)
  lip <- if (length(e)) drop(abs(jacobian(f, e)) %*% pmax(abs(e), 1)) else rep(0, 8)
  cat(name, "iterations", base$t1$iters, base$t0$iters, "\n", file = stderr())
  list(name = name, horizon = t, bound = b, weighted = weighted, clustered = clustered,
       weights = w0, cluster = cluster,
       hazard1 = p$h1, hazard0 = p$h0, censoring1 = p$c1, censoring0 = p$c0, propensity = p$g,
       numbers = unname(base$numbers), iterations = c(base$t1$iters, base$t0$iters),
       epsilon1 = I(e1), epsilon0 = I(e0),
       scores = c(-base$t1$score, -base$t0$score),
       one_step = c(1 - one_step(1, p$h1, p$c1, p$g, w, t, b), 1 - one_step(0, p$h0, p$c0, p$g, w, t, b)),
       lipschitz = lip, info = c(base$t1$infos, base$t0$infos))
}

scenarios <- list(
  scenario("horizon4", 4, 0, FALSE, FALSE),
  scenario("horizon6_bound", 6, 0.05, FALSE, FALSE),
  scenario("horizon4_weighted_clustered", 4, 0, TRUE, TRUE)
)
out <- list(
  versions = list(r = R.version.string, numDeriv = as.character(packageVersion("numDeriv")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  time = time, event = event, treatment = a, x1 = x1, x2 = x2, scenarios = scenarios)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", na = "null", matrix = "rowmajor"))
