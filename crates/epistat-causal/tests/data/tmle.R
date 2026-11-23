# Targeted maximum likelihood estimation, checked against the tmle package and
# against the estimator written out by hand.
#
# Data: two covariates, a binary treatment, a binary and a continuous outcome,
# case weights, clusters of equal and of unequal size, n = 400; and a second
# treatment with poor overlap, for which the support rule of the effect on the
# treated and on the controls leaves out persons.
#
# tmle scenarios. tmle::tmle with Q.SL.library = "SL.glm", g.SL.library =
# "SL.glm", cvQinit = FALSE and prescreenW.g = FALSE, which fits the main-terms
# glm of the outcome on the treatment and the covariates and the logistic glm of
# the treatment, so that the comparison is exact; and tmle::tmle given those
# predictions (Q and g1W), for the estimator on given predictions. The
# hand-written estimator below reproduces both runs to the last bits before they
# are written, which checks that it is tmle's algorithm.
#
# Hand-written scenarios. The estimator with one outcome model in each arm, a
# known outcome range, no propensity bound, clusters of unequal size and the
# support rule switched off, with the variance of this crate (sum of squared
# influences, cluster sums, G / (G - 1)).
#
# Tolerances: the Lipschitz constants of every reported number with respect to
# the coefficients of the nuisance models (the whole procedure, the
# fluctuation refitted, the one-step path at its number of steps) and with
# respect to the coefficients of the fluctuation or of the recalibration, by
# numDeriv, and the condition numbers of the information matrices.
# The versions are in the file. Numbers are written with 17 digits.
# Run: Rscript tmle.R > tmle.json
suppressMessages({
  library(jsonlite)
  library(tmle)
  library(numDeriv)
})
set.seed(20261125)
n <- 400
x1 <- round(rnorm(n), 3)
x2 <- rbinom(n, 1, 0.4)
a <- rbinom(n, 1, plogis(0.6 * x1 - 0.5 * x2 - 0.1))
yb <- rbinom(n, 1, plogis(-0.5 + 0.8 * x1 + 0.6 * x2 + 0.7 * a))
yc <- round(1 + 0.5 * x1 - 0.8 * x2 + 1.2 * a + rnorm(n), 4)
w_user <- round(runif(n, 0.5, 2), 3)
cl_equal <- (seq_len(n) - 1) %% 80
cl_unequal <- sort(sample(0:59, n, replace = TRUE))
set.seed(7)
z1 <- round(rnorm(n), 3)
z2 <- rbinom(n, 1, 0.4)
a_ov <- rbinom(n, 1, plogis(-0.3 + 2.5 * z1 - 0.4 * z2))
y_ov <- rbinom(n, 1, plogis(-0.4 + 0.7 * z1 + 0.5 * z2 + 0.6 * a_ov))
X <- unname(cbind(1, x1, x2))
X_ov <- unname(cbind(1, z1, z2))
EPS_TIGHT <- glm.control(epsilon = 1e-14, maxit = 100)

clamp <- function(x, lo, hi) pmin(pmax(x, lo), hi)

# The one-step path of the effect on the arm a = 1, as tmle's oneStepATT. With
# `fixed`, exactly that many steps and no test of the loss.
one_step <- function(ys, a, Q, g1, w, b, qb, step, fixed = NULL) {
  qq <- mean(a * w)
  g1 <- clamp(g1, b, 1); g0 <- clamp(1 - g1, b, 1)
  loss <- function(Q, g1, g0) -mean(w * (ys * log(Q[, 1]) + (1 - ys) * log(1 - Q[, 1]) +
                                           a * log(g1) + (1 - a) * log(g0)))
  ic <- function(Q, g1, g0, psi) w * ((a - (1 - a) * g1 / g0) * (ys - Q[, 1]) + a * (Q[, 3] - Q[, 2] - psi)) / qq
  psi <- mean(w * (Q[, 3] - Q[, 2]) * g1 / qq)
  IC <- ic(Q, g1, g0, psi)
  h <- if (mean(qq * IC) > 0) -step else step
  lag <- psi
  L <- loss(Q, g1, g0); Lprev <- Inf
  k <- 0
  state <- list(psi = psi, IC = IC, Q = Q)
  limit <- max(1000, 2 / step)
  go <- function() if (is.null(fixed)) Lprev > L && k < limit else k < fixed + 1
  while (go()) {
    state <- list(psi = psi, IC = IC, Q = Q)
    gn <- plogis(qlogis(g1) - h * (Q[, 3] - Q[, 2] - lag))
    H <- cbind(a - (1 - a) * g1 / (1 - g1), -g1 / (1 - g1), 1)
    Q <- clamp(plogis(qlogis(Q) - h * H), qb, 1 - qb)
    g0 <- clamp(1 - gn, b, 1); g1 <- clamp(gn, b, 1)
    lag <- psi
    psi <- mean(w * (Q[, 3] - Q[, 2]) * g1 / qq)
    Lprev <- L; L <- loss(Q, g1, g0); if (!is.finite(L)) L <- Inf
    IC <- ic(Q, g1, g0, psi)
    k <- k + 1
  }
  list(psi = state$psi, IC = state$IC, Q = state$Q, steps = k - 1)
}

# Our variance of influences IF (sum of squares, cluster sums, G / (G - 1)).
our_var <- function(inf, cluster) {
  if (is.null(cluster)) return(sum(inf^2))
  s <- rowsum(inf, cluster); g <- nrow(s)
  sum(s^2) * g / (g - 1)
}
# Summary of means, difference and ratio from the influences of the two means.
summ <- function(m1, m0, if1, if0, cluster) {
  r <- m1 / m0
  c(mean1 = m1, mean0 = m0, diff = m1 - m0, ratio = r,
    se_mean1 = sqrt(our_var(if1, cluster)), se_mean0 = sqrt(our_var(if0, cluster)),
    se_diff = sqrt(our_var(if1 - if0, cluster)),
    se_ratio = sqrt(our_var((if1 - r * if0) / m0, cluster)))
}

# The estimator. q1, q0: scaled bounded predictions; g: the propensity; ys: the
# scaled outcome; y: the outcome; w: case weights rescaled to mean 1; lower and
# width: the range; b: the propensity bound; qb: the outcome bound. `refit`: a
# function(rows, g) giving the propensity on rows when fewer than 90 percent of an
# arm remain. `eps`, `cal` and `steps` fix the fluctuation, the recalibration
# intercepts and the numbers of steps, for the derivatives.
estimator <- function(y, ys, a, q1, q0, g, w, lower, width, b, qb, cluster, restrict,
                      refit, ctl = glm.control(), eps = NULL, cal = NULL, steps = NULL,
                      step = 1e-3) {
  n <- length(y)
  out <- list()
  # Effect in the population.
  g1 <- clamp(g, b, 1); g0 <- clamp(1 - g, b, 1)
  l1 <- qlogis(q1); l0 <- qlogis(q0); lA <- ifelse(a == 1, l1, l0)
  H0 <- (1 - a) / g0; H1 <- a / g1
  if (is.null(eps)) {
    f <- glm(ys ~ -1 + offset(lA) + H0 + H1, family = binomial, weights = w, control = ctl)
    e <- unname(coef(f)); e[is.na(e)] <- 0
    out$fluct_iter <- f$iter
    out$fluct_info <- crossprod(cbind(H0, H1) * sqrt(w * fitted(f) * (1 - fitted(f))))
  } else e <- eps
  out$eps <- e
  s1 <- plogis(l1 + e[2] / g1) * width + lower
  s0 <- plogis(l0 + e[1] / g0) * width + lower
  m1 <- sum(w * s1) / sum(w); m0 <- sum(w * s0) / sum(w)
  if1 <- w * (a / g1 * (y - s1) + s1 - m1) / sum(w)
  if0 <- w * ((1 - a) / g0 * (y - s0) + s0 - m0) / sum(w)
  out$ate <- summ(m1, m0, if1, if0, cluster)
  out$ate_if <- list(if1 = if1, if0 = if0)
  out$ate_trimmed <- sum(g < b | 1 - g < b)
  # Effects on the treated and on the controls.
  att_rows <- g >= min(g[a == 1]); atc_rows <- g <= max(g[a == 0])
  apply_rule <- restrict && sum(1 - a[att_rows]) > 10 && sum(a[atc_rows]) > 10
  out$cal <- c(NA, NA)
  out$steps <- c(NA, NA)
  for (t in c("att", "atc")) {
    k <- if (t == "att") 1 else 2
    if (apply_rule) {
      rows <- which(if (t == "att") att_rows else atc_rows)
      arm_total <- if (t == "att") sum(1 - a) else sum(a)
      kept <- if (t == "att") sum(1 - a[rows]) else sum(a[rows])
      if (kept >= 0.9 * arm_total) {
        cc <- if (is.null(cal)) unname(coef(glm(a[rows] ~ 1 + offset(qlogis(g[rows])),
                                              family = binomial, control = ctl))) else cal[k]
        out$cal[k] <- cc
        gt <- plogis(cc + qlogis(g[rows]))
        out[[paste0(t, "_branch")]] <- "intercept"
      } else {
        gt <- refit(rows, g)
        out[[paste0(t, "_branch")]] <- "refit"
      }
    } else {
      rows <- seq_len(n); gt <- g[rows]
      out[[paste0(t, "_branch")]] <- "none"
    }
    m <- length(rows)
    wr <- w[rows] / sum(w[rows]) * m
    Q <- plogis(qlogis(cbind(ifelse(a == 1, q1, q0), q0, q1)))[rows, , drop = FALSE]
    if (t == "att") {
      aa <- a[rows]; QQ <- Q; gg <- gt
    } else {
      aa <- 1 - a[rows]; QQ <- Q[, c(1, 3, 2)]; gg <- 1 - gt
    }
    o <- one_step(ys[rows], aa, QQ, gg, wr, b, qb, step,
                  fixed = if (is.null(steps)) NULL else steps[k])
    out$steps[k] <- o$steps
    sign <- if (t == "att") 1 else -1
    effect <- sign * o$psi * width
    infl <- numeric(n); infl[rows] <- sign * o$IC * width / m
    size <- sum(wr * aa)
    own <- sum(wr * aa * y[rows]) / size
    own_if <- numeric(n); own_if[rows] <- wr * aa * (y[rows] - own) / size
    cl <- if (is.null(cluster)) NULL else cluster[rows]
    if (t == "att") {
      s <- summ(own, own - effect, own_if[rows], (own_if - infl)[rows], cl)
    } else {
      s <- summ(own + effect, own, (own_if + infl)[rows], own_if[rows], cl)
    }
    out[[t]] <- s
    out[[paste0(t, "_ic")]] <- o$IC * width   # tmle's IC on the rows
    out[[paste0(t, "_rows")]] <- rows
    out[[paste0(t, "_score")]] <- sum(infl)
  }
  out
}

# tmle's variance of an influence vector IC (per row) with clusters id.
tmle_var <- function(ic, id) {
  if (is.null(id)) return(var(ic) / length(ic))
  m <- as.vector(by(ic, id, mean)); var(m) / length(m)
}

numbers <- function(o) c(o$ate, o$att, o$atc)

scenario <- function(name, data, outcome, weighted, cluster_kind, gbound, model,
                     range_kind = "observed", restrict = TRUE, run_tmle = TRUE) {
  t0 <- proc.time()[3]
  d <- if (data == "main") list(a = a, x = X, y = if (outcome == "binary") yb else yc)
       else list(a = a_ov, x = X_ov, y = y_ov)
  y <- d$y; aa <- d$a; xx <- d$x
  wraw <- if (weighted) w_user else rep(1, n)
  w <- wraw / sum(wraw) * n
  cluster <- switch(cluster_kind, none = NULL, equal = cl_equal, unequal = cl_unequal)
  b <- if (is.null(gbound)) 5 / sqrt(n) / log(n) else if (gbound == 0) 0 else gbound
  qb <- 5e-4
  rng <- if (range_kind == "observed") range(y) else c(min(y) - 1, max(y) + 2)
  lower <- rng[1]; width <- rng[2] - rng[1]
  ys <- (y - lower) / width
  fam <- if (outcome == "binary") binomial() else gaussian()
  # The nuisance fits as functions of their coefficients.
  fit_g <- function(ctl) glm(aa ~ xx - 1, family = binomial, weights = w, control = ctl)
  fit_q <- function(ctl) {
    if (model == "pooled") {
      list(glm(ys ~ cbind(xx, aa) - 1, family = fam, weights = w, control = ctl))
    } else {
      list(glm(ys ~ xx - 1, family = fam, weights = w * aa, control = ctl),
           glm(ys ~ xx - 1, family = fam, weights = w * (1 - aa), control = ctl))
    }
  }
  k <- ncol(xx)
  preds <- function(p) {
    ge <- plogis(drop(xx %*% p[1:k]))
    if (model == "pooled") {
      bq <- p[k + 1:(k + 1)]
      q1 <- fam$linkinv(drop(xx %*% bq[1:k]) + bq[k + 1])
      q0 <- fam$linkinv(drop(xx %*% bq[1:k]))
    } else {
      q1 <- fam$linkinv(drop(xx %*% p[k + 1:k]))
      q0 <- fam$linkinv(drop(xx %*% p[2 * k + 1:k]))
    }
    list(g = ge, q1 = clamp(q1, qb, 1 - qb), q0 = clamp(q0, qb, 1 - qb))
  }
  refit_with <- function(ctl) function(rows, g) {
    dd <- data.frame(A = aa, gg = g, xx)[rows, ]
    m <- glm(A ~ gg + . - 1, data = dd, family = binomial, weights = w[rows], control = ctl)
    unname(fitted(m))
  }
  gf <- fit_g(glm.control()); qf <- fit_q(glm.control())
  p0 <- c(unname(coef(gf)), unlist(lapply(qf, function(m) unname(coef(m)))))
  pr <- preds(p0)
  run <- function(pr, ctl = glm.control(), ...) {
    estimator(y, ys, aa, pr$q1, pr$q0, pr$g, w, lower, width, b, qb, cluster, restrict,
              refit_with(ctl), ctl = ctl, ...)
  }
  base <- run(pr)
  # Lipschitz constants with respect to the nuisance coefficients, the whole
  # procedure with tight refits and the steps fixed.
  of_nuis <- function(p) numbers(run(preds(p), ctl = EPS_TIGHT, steps = base$steps))
  jn <- jacobian(of_nuis, p0)
  lip_nuis <- drop(abs(jn) %*% pmax(abs(p0), 1))
  # With respect to the fluctuation (effect in the population) and the
  # recalibration intercepts (on the arms), the predictions fixed.
  of_eps <- function(e) numbers(run(pr, eps = e[1:2], cal = e[3:4], steps = base$steps))
  e0 <- c(base$eps, ifelse(is.na(base$cal), 0, base$cal))
  je <- jacobian(of_eps, e0)
  lip_eps <- drop(abs(je) %*% pmax(abs(e0), 1))
  info_kappa <- function(g) { s <- sqrt(diag(g)); kappa(diag(1 / s) %*% g %*% diag(1 / s), exact = TRUE) }
  kap <- max(info_kappa(crossprod(xx * sqrt(w * pr$g * (1 - pr$g)))),
             sapply(qf, function(m) {
               keep <- m$prior.weights > 0
               info_kappa(crossprod(model.matrix(m)[keep, , drop = FALSE] * sqrt(m$weights[keep])))
             }),
             info_kappa(base$fluct_info))
  res <- list(name = name, data = data, outcome = outcome, weighted = weighted,
              cluster_kind = cluster_kind, gbound = gbound, model = model,
              range = rng, range_kind = range_kind, restrict = restrict,
              y = y, weights = wraw, cluster = cluster,
              mu1 = pr$q1 * width + lower, mu0 = pr$q0 * width + lower, propensity = pr$g,
              bound = b, numbers = unname(numbers(base)), steps = base$steps,
              eps = base$eps, fluct_iter = base$fluct_iter,
              branches = c(base$att_branch, base$atc_branch),
              n_used = c(length(base$att_rows), length(base$atc_rows)),
              n_trimmed = base$ate_trimmed,
              scores = c(base$att_score, base$atc_score),
              lip_nuis = lip_nuis, lip_eps = lip_eps, kappa = kap)
  if (run_tmle) {
    W <- data.frame(xx[, -1])
    common <- list(Y = y, A = aa, W = W, family = if (outcome == "binary") "binomial" else "gaussian",
                   prescreenW.g = FALSE, obsWeights = if (weighted) wraw else NULL,
                   id = if (is.null(cluster)) seq_len(n) else cluster, gbound = gbound)
    r_fit <- do.call(tmle, c(common, list(Q.SL.library = "SL.glm", g.SL.library = "SL.glm",
                                           cvQinit = FALSE)))
    r_core <- do.call(tmle, c(common, list(Q = cbind(res$mu0, res$mu1), g1W = pr$g,
                                            g.SL.library = "SL.glm")))
    pick <- function(r) {
      e <- r$estimates
      list(ate = e$ATE$psi, var_ate = e$ATE$var.psi, ey1 = e$EY1$psi, var_ey1 = e$EY1$var.psi,
           ey0 = e$EY0$psi, var_ey0 = e$EY0$var.psi,
           rr = if (is.null(e$RR)) NULL else e$RR$psi,
           var_log_rr = if (is.null(e$RR)) NULL else e$RR$var.log.psi,
           att = e$ATT$psi, var_att = e$ATT$var.psi, atc = e$ATC$psi, var_atc = e$ATC$var.psi,
           epsilon = unname(r$epsilon))
    }
    res$tmle_fit <- pick(r_fit)
    res$tmle_core <- pick(r_core)
    # The hand-written estimator is tmle's: its numbers equal both runs.
    id <- if (is.null(cluster)) NULL else cluster
    hand <- list(ate = base$ate[["diff"]], ey1 = base$ate[["mean1"]], ey0 = base$ate[["mean0"]],
                 att = base$att[["diff"]], atc = base$atc[["diff"]],
                 var_ate = tmle_var(n * (base$ate_if$if1 - base$ate_if$if0), id),
                 var_att = tmle_var(base$att_ic, id[base$att_rows]),
                 var_atc = tmle_var(base$atc_ic, id[base$atc_rows]))
    for (r in list(res$tmle_fit, res$tmle_core)) {
      for (nm in names(hand)) {
        dev <- abs(hand[[nm]] - r[[nm]]) / max(abs(r[[nm]]), 1e-300)
        if (dev > 1e-12) stop(sprintf("%s: hand-written %s differs from tmle by %g", name, nm, dev))
      }
    }
  }
  cat(name, "done in", proc.time()[3] - t0, "s\n", file = stderr())
  res
}

scenarios <- list(
  scenario("binary_default", "main", "binary", FALSE, "none", NULL, "pooled"),
  scenario("gaussian_default", "main", "gaussian", FALSE, "none", NULL, "pooled"),
  scenario("binary_weighted", "main", "binary", TRUE, "none", NULL, "pooled"),
  scenario("gaussian_clustered_weighted", "main", "gaussian", TRUE, "equal", NULL, "pooled"),
  scenario("binary_gbound", "main", "binary", FALSE, "none", 0.2, "pooled"),
  scenario("overlap_refit", "overlap", "binary", FALSE, "none", NULL, "pooled"),
  scenario("gaussian_byarm_known_unequal", "main", "gaussian", TRUE, "unequal", 0, "byarm",
           range_kind = "known", run_tmle = FALSE),
  scenario("binary_byarm_unrestricted", "main", "binary", FALSE, "none", 0, "byarm",
           restrict = FALSE, run_tmle = FALSE)
)

out <- list(
  versions = list(r = R.version.string, tmle = as.character(packageVersion("tmle")),
                  SuperLearner = as.character(packageVersion("SuperLearner")),
                  numDeriv = as.character(packageVersion("numDeriv")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  x = X, x_overlap = X_ov, treatment = a, treatment_overlap = a_ov, scenarios = scenarios)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
