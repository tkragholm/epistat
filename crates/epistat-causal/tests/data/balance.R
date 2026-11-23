# Entropy balancing and the covariate balancing propensity score, checked against
# WeightIt (method = "ebal"), ebal and CBPS, and against estimators written out
# here with numerical derivatives that share no code with the Rust implementation.
#
# Entropy balancing (data `main`, n = 400):
#   - WeightIt::weightit(method = "ebal") weights for ATT, ATC and ATE, with
#     moments 1 to 3, interactions, sample weights and base weights; the largest
#     standardised imbalance of those weights, the smallest eigenvalue of the
#     weighted covariance of the standardised columns, and the norms of the
#     standardised columns, which bound the distance of two solutions to the same
#     balance problem;
#   - ebal::ebalance for the ATT with base weights (its constraint tolerance is 1e-6);
#   - the weights of an own Newton solver, to 1e-14.
# Infeasible: a covariate whose target mean is outside the range of the controls.
# The infinitesimal jackknife: the derivative of the whole procedure with respect to
# the case weight of each person (central differences, Richardson extrapolation of
# steps h and h / 2), for a weighted regression of the outcome on a design, which
# is the variance that the stacked equations must reproduce.
# The bootstrap that refits the weights, for the standard error of the effect.
#
# CBPS (data `main`):
#   - CBPS::CBPS for ATE, ATT and ATC, exact and over, with sample weights, with and
#     without twostep and standardize;
#   - own solvers for the same estimators: the balance condition by Newton, the
#     objective by Gauss-Newton with a weighting matrix at the logistic fit (Jacobians
#     by numDeriv with a large step and six Richardson extrapolations), the
#     continuously updated objective by BFGS with derivatives by numDeriv;
#   - the weighting matrix the package uses (it scales the logistic fit by a
#     factor from optimize on [0.8, 1.1]), and the minimum with it, to measure the
#     effect of that choice;
#   - the sizes that bound the distance between the minimum and the package's
#     result, from the gradient left at its result.
# Numbers are written with 17 digits.
# Run: Rscript balance.R > balance.json
suppressMessages({
  library(jsonlite)
  library(WeightIt)
  library(CBPS)
  library(ebal)
  library(numDeriv)
})
set.seed(20261123)
gen <- function(n) {
  x1 <- round(rnorm(n), 3)
  x2 <- rbinom(n, 1, 0.4)
  x3 <- round(rgamma(n, 2, 1), 3)
  a <- rbinom(n, 1, plogis(-0.3 + 0.6 * x1 - 0.5 * x2 + 0.25 * x3))
  y <- round(1 + 0.5 * x1 - 0.8 * x2 + 0.3 * x3 + 1.2 * a + rnorm(n), 4)
  list(a = a, x1 = x1, x2 = x2, x3 = x3, y = y,
       s = round(runif(n, 0.5, 2), 3), q = round(runif(n, 0.5, 1.5), 3),
       cl = (seq_len(n) - 1) %% (n %/% 5))
}
dm <- gen(400)
dat <- data.frame(a = dm$a, x1 = dm$x1, x2 = dm$x2, x3 = dm$x3)

# ---- own entropy balancing (the convention of WeightIt: sum s w = sum of s in the target)
expand <- function(C, moments, int) {
  bin <- apply(C, 2, function(v) all(v %in% c(0, 1)))
  out <- C
  Cc <- sweep(C, 2, colMeans(C))
  if (moments >= 2) for (k in 2:moments) for (j in which(!bin)) out <- cbind(out, Cc[, j]^k)
  if (int) for (i in 1:(ncol(C) - 1)) for (j in (i + 1):ncol(C)) {
    ci <- if (bin[i]) C[, i] else Cc[, i]; cj <- if (bin[j]) C[, j] else Cc[, j]
    pr <- ci * cj
    if (diff(range(pr)) > 0) out <- cbind(out, pr)
  }
  unname(out)
}
ebal_arm <- function(U, av) {
  p <- ncol(U); lam <- rep(0, p)
  F <- function(l) { e <- U %*% l; m <- max(e); m + log(sum(av * exp(e - m))) }
  for (it in 1:200) {
    e <- drop(U %*% lam); m <- max(e); pw <- av * exp(e - m); pw <- pw / sum(pw)
    g <- colSums(pw * U)
    if (max(abs(g)) < 1e-15) break
    H <- crossprod(U, pw * U) - tcrossprod(g)
    st <- -solve(H, g); t <- 1; f0 <- F(lam)
    while (F(lam + t * st) > f0 + 1e-4 * t * sum(g * st) + 1e-14 && t > 1e-12) t <- t / 2
    lam <- lam + t * st
  }
  lam
}
ebal_w <- function(a, C, s, q, estimand) {
  n <- length(a)
  tg <- switch(estimand, ATT = a == 1, ATC = a == 0, ATE = rep(TRUE, n)) & s > 0
  arms <- switch(estimand, ATT = 0, ATC = 1, ATE = c(0, 1))
  size <- sum(s[tg]); m <- colSums(s[tg] * C[tg, , drop = FALSE]) / size
  sdv <- sqrt(colSums(s[tg] * sweep(C[tg, , drop = FALSE], 2, m)^2) / size)
  U <- sweep(sweep(C, 2, m), 2, sdv, "/")
  w <- rep(1, n)
  for (g in arms) {
    r <- which(a == g & s * q > 0)
    lam <- ebal_arm(U[r, , drop = FALSE], s[r] * q[r])
    e <- drop(U %*% lam); mx <- max(e[r])
    den <- sum(s[r] * q[r] * exp(e[r] - mx))
    w[a == g] <- size / den * q[a == g] * exp(e[a == g] - mx)
  }
  w
}
wls <- function(y, Z, ww) { solve(crossprod(Z, ww * Z), crossprod(Z, ww * y)) }
imbalance <- function(a, C, s, w, estimand) {
  n <- length(a)
  tg <- switch(estimand, ATT = a == 1, ATC = a == 0, ATE = rep(TRUE, n)) & s > 0
  arms <- switch(estimand, ATT = 0, ATC = 1, ATE = c(0, 1))
  size <- sum(s[tg]); m <- colSums(s[tg] * C[tg, , drop = FALSE]) / size
  sdv <- sqrt(colSums(s[tg] * sweep(C[tg, , drop = FALSE], 2, m)^2) / size)
  max(sapply(arms, function(g) {
    r <- a == g & s > 0
    max(abs((colSums(s[r] * w[r] * C[r, , drop = FALSE]) / size - m) / sdv))
  }))
}

ebal_cases <- list(
  list(name = "att_m1", estimand = "ATT", moments = 1, int = FALSE, use_s = FALSE, use_q = FALSE),
  list(name = "ate_m1", estimand = "ATE", moments = 1, int = FALSE, use_s = FALSE, use_q = FALSE),
  list(name = "atc_m1", estimand = "ATC", moments = 1, int = FALSE, use_s = FALSE, use_q = FALSE),
  list(name = "att_m1_q", estimand = "ATT", moments = 1, int = FALSE, use_s = FALSE, use_q = TRUE),
  list(name = "att_m2", estimand = "ATT", moments = 2, int = FALSE, use_s = FALSE, use_q = FALSE),
  list(name = "ate_m3_int", estimand = "ATE", moments = 3, int = TRUE, use_s = FALSE, use_q = FALSE),
  list(name = "att_m2_int_s_q", estimand = "ATT", moments = 2, int = TRUE, use_s = TRUE, use_q = TRUE),
  list(name = "ate_m1_s_q", estimand = "ATE", moments = 1, int = FALSE, use_s = TRUE, use_q = TRUE),
  list(name = "atc_m2_s", estimand = "ATC", moments = 2, int = FALSE, use_s = TRUE, use_q = FALSE)
)
ebal_out <- lapply(ebal_cases, function(cs) {
  s <- if (cs$use_s) dm$s else rep(1, 400); q <- if (cs$use_q) dm$q else rep(1, 400)
  args <- list(a ~ x1 + x2 + x3, data = dat, method = "ebal", estimand = cs$estimand,
               moments = cs$moments, int = cs$int, reltol = 1e-14)
  if (cs$use_s) args$s.weights <- s
  if (cs$use_q) args$base.weights <- q
  fit <- suppressWarnings(do.call(weightit, args))
  w <- fit$weights
  C <- expand(as.matrix(dat[, -1]), cs$moments, cs$int)
  w_own <- ebal_w(dm$a, C, s, q, cs$estimand)
  # sizes for the bound on the distance of two solutions of the balance problem
  tg <- switch(cs$estimand, ATT = dm$a == 1, ATC = dm$a == 0, ATE = rep(TRUE, 400))
  size <- sum(s[tg]); m <- colSums(s[tg] * C[tg, , drop = FALSE]) / size
  sdv <- sqrt(colSums(s[tg] * sweep(C[tg, , drop = FALSE], 2, m)^2) / size)
  U <- sweep(sweep(C, 2, m), 2, sdv, "/")
  arms <- switch(cs$estimand, ATT = 0, ATC = 1, ATE = c(0, 1))
  lmin <- min(sapply(arms, function(g) {
    r <- dm$a == g; pw <- s[r] * w_own[r] / sum(s[r] * w_own[r])
    g1 <- colSums(pw * U[r, , drop = FALSE])
    min(eigen(crossprod(U[r, , drop = FALSE], pw * U[r, , drop = FALSE]) - tcrossprod(g1), symmetric = TRUE)$values)
  }))
  list(name = cs$name, estimand = cs$estimand, moments = cs$moments, int = cs$int,
       use_s = cs$use_s, use_q = cs$use_q, weights = unname(w), weights_own = w_own,
       imbalance = imbalance(dm$a, C, s, w, cs$estimand),
       imbalance_own = imbalance(dm$a, C, s, w_own, cs$estimand),
       lambda_min = lmin, unorm = unname(sqrt(rowSums(U^2))),
       ncol = ncol(C))
})

# ebal package, ATT with base weights
Xm <- as.matrix(dat[, -1])
eb <- ebalance(Treatment = dm$a, X = Xm, base.weight = dm$q[dm$a == 0], constraint.tolerance = 1e-6, print.level = -1)
ebal_pkg <- list(
  weights_controls = eb$w,            # sums to the number of treated
  tol = 1e-6,
  sd_treated = unname(apply(Xm[dm$a == 1, ], 2, sd) * sqrt((sum(dm$a) - 1) / sum(dm$a))))

# infeasible: a covariate that never exceeds the target mean among the controls
inf_x <- dm$x3
inf_x[dm$a == 0] <- pmin(inf_x[dm$a == 0], quantile(inf_x[dm$a == 1], 0.5))
infeasible <- list(x3 = inf_x,
  target_mean = mean(inf_x[dm$a == 1]), max_controls = max(inf_x[dm$a == 0]))
inf_fit <- tryCatch(suppressWarnings(weightit(a ~ x1 + x2 + I(inf_x), data = dat, method = "ebal",
  estimand = "ATT")), error = function(e) e)
infeasible$weightit_imbalance <- if (inherits(inf_fit, "error")) NA else
  imbalance(dm$a, cbind(dm$x1, dm$x2, inf_x), rep(1, 400), inf_fit$weights, "ATT")

# ---- the infinitesimal jackknife and the bootstrap, entropy balancing
ij_deriv <- function(f, s, h = 1e-4) {
  n <- length(s); k <- length(f(s)); out <- matrix(0, n, k)
  for (i in 1:n) {
    d <- function(step) { sp <- s; sm <- s; sp[i] <- s[i] + step; sm[i] <- s[i] - step
      (f(sp) - f(sm)) / (2 * step) }
    out[i, ] <- (4 * d(h / 2) - d(h)) / 3
  }
  out
}
di <- gen(150)
dfi <- data.frame(a = di$a, x1 = di$x1, x2 = di$x2, x3 = di$x3)
Ci <- as.matrix(dfi[, -1])
Zi <- cbind(1, di$a, di$x1)
ij_cases <- lapply(c("ATT", "ATE"), function(est) {
  f <- function(sv) { w <- ebal_w(di$a, Ci, sv, di$q, est); wls(di$y, Zi, sv * w) }
  s0 <- di$s
  IF <- ij_deriv(f, s0)
  list(estimand = est, theta = drop(f(s0)), influence = as.vector(IF), k = 3)
})
ij_means <- lapply(c("ATT", "ATE"), function(est) {
  Zm <- cbind(1 - di$a, di$a)
  f <- function(sv) { w <- ebal_w(di$a, Ci, sv, di$q, est); wls(di$y, Zm, sv * w) }
  IF <- ij_deriv(f, di$s)
  list(estimand = est, theta = drop(f(di$s)), influence = as.vector(IF), k = 2)
})
db <- gen(1500); Cb <- cbind(db$x1, db$x2, db$x3)
boot_ebal <- function(est, B = 400) {
  Zm <- cbind(1 - db$a, db$a)
  one <- function(idx) {
    w <- ebal_w(db$a[idx], Cb[idx, ], rep(1, length(idx)), rep(1, length(idx)), est)
    b <- wls(db$y[idx], Zm[idx, ], w); b[2] - b[1]
  }
  all <- seq_along(db$a)
  est0 <- one(all)
  d <- replicate(B, one(sample(all, replace = TRUE)))
  list(estimand = est, estimate = est0, se_boot = sd(d), B = B)
}
boot_e <- lapply(c("ATT", "ATE"), boot_ebal)

# ---- CBPS, own solvers
# The numerical derivatives use a larger relative step and more Richardson
# extrapolations than numDeriv's default, which keeps the truncation error below
# 1e-12 and the rounding error near 1e-13.
ma <- list(eps = 1e-4, d = 1e-3, r = 6, v = 2)
jac <- function(f, x) jacobian(f, x, method.args = ma)
# The logistic maximum likelihood fit, polished by Newton steps to a score of 1e-15:
# glm stops when the deviance changes by 1e-8, which leaves the coefficients and so
# the weighting matrix of the over-identified objective only accurate to 1e-5.
ml_fit <- function(X, t, s) {
  b <- coef(suppressWarnings(glm(t ~ X - 1, family = binomial, weights = s)))
  for (it in 1:30) {
    e <- plogis(drop(X %*% b)); sc <- drop(crossprod(X, s * (t - e))) / sum(s)
    if (max(abs(sc)) < 1e-15) break
    b <- b + solve(crossprod(X, s * e * (1 - e) * X) / sum(s), sc)
  }
  unname(b)
}
prob <- function(X, t, s, att) {
  n <- sum(s); nt <- sum(s[t == 1])
  list(X = X, t = t, s = s, att = att, n = n, cc = if (att) n / nt else 1)
}
wbal <- function(P, e) if (P$att) P$cc * (P$t - e) / (1 - e) else ifelse(P$t == 1, 1 / e, -1 / (1 - e))
gmom <- function(P, beta) {
  e <- plogis(drop(P$X %*% beta))
  c(crossprod(P$X, P$s * (P$t - e)), crossprod(P$X, P$s * wbal(P, e))) / P$n
}
Vmat <- function(P, beta) {
  e <- plogis(drop(P$X %*% beta)); X <- P$X; s <- P$s
  if (P$att) { a <- e * (1 - e); b <- P$cc * e; c3 <- P$cc^2 * e / (1 - e) }
  else { a <- e * (1 - e); b <- rep(1, length(e)); c3 <- 1 / (e * (1 - e)) }
  G <- function(w) crossprod(X, s * w * X)
  rbind(cbind(G(a), G(b)), cbind(G(b), G(c3))) / P$n
}
bal_root <- function(P, start) {
  k <- ncol(P$X); beta <- start
  gb <- function(b) gmom(P, b)[-(1:k)]
  for (it in 1:200) {
    g <- gb(beta); if (max(abs(g)) < 1e-15) break
    Jm <- jac(gb, beta); st <- -solve(Jm, g); t <- 1
    while (sum(gb(beta + t * st)^2) > sum(g^2) && t > 1e-10) t <- t / 2
    beta <- beta + t * st
  }
  beta
}
# Newton steps with numerical gradient and Hessian, which finish a minimisation that
# a line search on the objective cannot finish below a gradient of about 1e-8.
polish <- function(f, b) {
  for (it in 1:20) {
    g <- grad(f, b, method.args = ma)
    if (max(abs(g)) < 1e-12) break
    b <- b - solve(hessian(f, b, method.args = ma), g)
  }
  b
}
J_fixed <- function(P, W, beta) { g <- gmom(P, beta); drop(t(g) %*% W %*% g) }
gn_min <- function(P, W, start) {
  beta <- start
  for (it in 1:500) {
    g <- gmom(P, beta); Gm <- jac(function(b) gmom(P, b), beta)
    grad <- 2 * drop(t(Gm) %*% W %*% g)
    if (max(abs(grad)) < 1e-14) break
    st <- -solve(t(Gm) %*% W %*% Gm, grad / 2); t <- 1
    J0 <- J_fixed(P, W, beta)
    while (J_fixed(P, W, beta + t * st) > J0 + 1e-4 * t * sum(grad * st) + 1e-15 && t > 1e-10) t <- t / 2
    beta <- beta + t * st
  }
  polish(function(z) J_fixed(P, W, z), beta)
}
J_cu <- function(P, beta) { g <- gmom(P, beta); drop(t(g) %*% solve(Vmat(P, beta), g)) }
cu_min <- function(P, start) {
  o <- optim(start, function(b) J_cu(P, b), function(b) grad(function(z) J_cu(P, z), b, method.args = ma),
             method = "BFGS", control = list(maxit = 5000, reltol = 1e-16, abstol = 0))
  polish(function(z) J_cu(P, z), o$par)
}
cbps_own <- function(X, t, s, att, ident, twostep) {
  P <- prob(X, t, s, att)
  ml <- ml_fit(X, t, s)
  b_exact <- bal_root(P, ml)
  if (ident == "exact") return(list(beta = b_exact, P = P, ml = ml))
  if (twostep) {
    W <- solve(Vmat(P, ml))
    b1 <- gn_min(P, W, ml); b2 <- gn_min(P, W, b_exact)
    best <- if (J_fixed(P, W, b1) <= J_fixed(P, W, b2)) b1 else b2
    list(beta = best, P = P, ml = ml, W = W, b1 = b1, b2 = b2, b_exact = b_exact)
  } else {
    b1 <- cu_min(P, ml); b2 <- cu_min(P, b_exact)
    best <- if (J_cu(P, b1) <= J_cu(P, b2)) b1 else b2
    list(beta = best, P = P, ml = ml, b1 = b1, b2 = b2, b_exact = b_exact)
  }
}
cbps_weights <- function(P, beta, std) {
  e <- plogis(drop(P$X %*% beta)); t <- P$t
  r <- if (P$att) ifelse(t == 1, P$cc, P$cc * e / (1 - e)) else ifelse(t == 1, 1 / e, 1 / (1 - e))
  if (std) for (g in 0:1) r[t == g] <- r[t == g] / sum(P$s[t == g] * r[t == g])
  r
}
Xd <- cbind(1, dm$x1, dm$x2, dm$x3)
cbps_cases <- list(
  list(name = "ate_exact", att = "ATE", ident = "exact", twostep = TRUE, std = TRUE, use_s = FALSE),
  list(name = "att_exact", att = "ATT", ident = "exact", twostep = TRUE, std = TRUE, use_s = FALSE),
  list(name = "atc_exact", att = "ATC", ident = "exact", twostep = TRUE, std = TRUE, use_s = FALSE),
  list(name = "ate_exact_s", att = "ATE", ident = "exact", twostep = TRUE, std = TRUE, use_s = TRUE),
  list(name = "ate_over", att = "ATE", ident = "over", twostep = TRUE, std = TRUE, use_s = FALSE),
  list(name = "att_over", att = "ATT", ident = "over", twostep = TRUE, std = TRUE, use_s = FALSE),
  list(name = "atc_over", att = "ATC", ident = "over", twostep = TRUE, std = TRUE, use_s = FALSE),
  list(name = "ate_over_raw", att = "ATE", ident = "over", twostep = TRUE, std = FALSE, use_s = FALSE),
  list(name = "ate_over_cu", att = "ATE", ident = "over", twostep = FALSE, std = TRUE, use_s = FALSE),
  list(name = "att_over_cu", att = "ATT", ident = "over", twostep = FALSE, std = TRUE, use_s = FALSE)
)
cbps_out <- lapply(cbps_cases, function(cs) {
  s <- if (cs$use_s) dm$s else rep(1, 400)
  attc <- switch(cs$att, ATE = 0, ATT = 1, ATC = 2)
  cap <- capture.output(pk <- suppressWarnings(CBPS(a ~ x1 + x2 + x3, data = dat, ATT = attc,
    method = cs$ident, twostep = cs$twostep, standardize = cs$std,
    sample.weights = s, iterations = 5000)))
  tt <- if (cs$att == "ATC") 1 - dm$a else dm$a
  own <- cbps_own(Xd, tt, s, cs$att != "ATE", cs$ident, cs$twostep)
  P <- own$P; beta <- own$beta; k <- 4
  e_own <- plogis(drop(Xd %*% beta))
  w_own <- cbps_weights(P, beta, cs$std)
  # the package's solution in the flipped labels for the ATC: fitted values are of t
  e_pk <- pk$fitted.values
  beta_pk <- coef(pk)[, 1]
  # bound on the distance of the package's result from the minimum of the same problem
  if (cs$ident == "exact") {
    g_pk <- gmom(P, beta_pk)[-(1:k)]
    Gb <- jac(function(b) gmom(P, b)[-(1:k)], beta)
    hx <- sqrt(rowSums((Xd %*% solve(Gb))^2))
    resid <- sqrt(sum(g_pk^2)); w_effect <- rep(0, 400)
  } else if (cs$twostep) {
    # the package's weighting matrix: V at the scaled logistic fit
    al <- optimize(function(al) J_cu(P, own$ml * al), interval = c(0.8, 1.1))$minimum
    Wa <- solve(Vmat(P, own$ml * al))
    b_a <- gn_min(P, Wa, own$ml); b_a2 <- gn_min(P, Wa, own$b_exact)
    b_a <- if (J_fixed(P, Wa, b_a) <= J_fixed(P, Wa, b_a2)) b_a else b_a2
    Gm <- jac(function(b) gmom(P, b), beta_pk)
    grad_pk <- drop(t(Gm) %*% Wa %*% gmom(P, beta_pk))      # half the gradient of J
    A <- t(Gm) %*% Wa %*% Gm
    hx <- sqrt(rowSums((Xd %*% solve(A))^2)); resid <- sqrt(sum(grad_pk^2))
    w_effect <- abs(plogis(drop(Xd %*% b_a)) - e_own)
  } else {
    gr <- function(b) grad(function(z) J_cu(P, z), b, method.args = ma)
    Hs <- hessian(function(z) J_cu(P, z), beta_pk, method.args = ma)
    hx <- sqrt(rowSums((Xd %*% solve(Hs))^2)); resid <- sqrt(sum(gr(beta_pk)^2)); w_effect <- rep(0, 400)
  }
  Jf <- if (cs$ident == "over") (if (cs$twostep) J_fixed(P, own$W, beta) else J_cu(P, beta)) else NA
  list(name = cs$name, att = cs$att, ident = cs$ident, twostep = cs$twostep, std = cs$std,
       use_s = cs$use_s,
       coef_pkg = beta_pk, fitted_pkg = e_pk, weights_pkg = pk$weights / s, J_pkg = pk$J,
       coef_own = beta, fitted_own = if (cs$att == "ATC") 1 - e_own else e_own, weights_own = w_own, J_own = Jf,
       hx = hx, resid = resid, w_effect = w_effect)
})

# ---- CBPS, the infinitesimal jackknife of the estimator with the weighting matrix held at
#      the value for the case weights of the data, and the bootstrap
Xi <- cbind(1, di$x1, di$x2, di$x3)
ij_cbps <- lapply(list(list("ATE", "exact", TRUE), list("ATT", "exact", FALSE),
                       list("ATE", "over", TRUE), list("ATT", "over", FALSE)), function(cs) {
  est <- cs[[1]]; ident <- cs[[2]]; std <- cs[[3]]
  att <- est != "ATE"
  P0 <- prob(Xi, di$a, di$s, att)
  ml0 <- ml_fit(Xi, di$a, di$s)
  W0 <- if (ident == "over") solve(Vmat(P0, ml0)) else NULL
  Zr <- cbind(1, di$a, di$x1)
  f <- function(sv) {
    P <- prob(Xi, di$a, sv, att)
    ml <- ml_fit(Xi, di$a, sv)
    be <- bal_root(P, ml)
    if (ident == "over") be <- gn_min(P, W0, be)
    w <- cbps_weights(P, be, std)
    wls(di$y, Zr, sv * w)
  }
  IF <- ij_deriv(f, di$s, h = 1e-4)
  list(estimand = est, ident = ident, std = std, theta = drop(f(di$s)), influence = as.vector(IF), k = 3)
})
Xb <- cbind(1, db$x1, db$x2, db$x3)
boot_cbps <- function(est, ident, B = 300) {
  att <- est != "ATE"; Zm <- cbind(1 - db$a, db$a)
  one <- function(idx) {
    P <- prob(Xb[idx, ], db$a[idx], rep(1, length(idx)), att)
    ml <- ml_fit(Xb[idx, ], db$a[idx], rep(1, length(idx)))
    be <- bal_root(P, ml)
    if (ident == "over") be <- gn_min(P, solve(Vmat(P, ml)), be)
    w <- cbps_weights(P, be, TRUE)
    b <- wls(db$y[idx], Zm[idx, ], w); b[2] - b[1]
  }
  all <- seq_along(db$a); est0 <- one(all)
  d <- replicate(B, one(sample(all, replace = TRUE)))
  list(estimand = est, ident = ident, estimate = est0, se_boot = sd(d), B = B)
}
boot_c <- list(boot_cbps("ATE", "exact"), boot_cbps("ATT", "over"))

out <- list(
  versions = list(r = R.version.string, WeightIt = as.character(packageVersion("WeightIt")),
                  CBPS = as.character(packageVersion("CBPS")), ebal = as.character(packageVersion("ebal")),
                  numDeriv = as.character(packageVersion("numDeriv"))),
  main = dm, ebal = ebal_out, ebal_pkg = ebal_pkg, infeasible = infeasible,
  ij_data = di, ij_ebal = ij_cases, ij_ebal_means = ij_means, ij_cbps = ij_cbps,
  boot_data = db, boot_ebal = boot_e, boot_cbps = boot_c, cbps = cbps_out)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null"))
