# The infinitesimal jackknife of the continuously updated CBPS, on the data
# `ij_data` of balance.json (n = 150, with sample weights).
#
# The procedure: with case weights sv, the moments g and their covariance V are
# formed with weights sv, the coefficients minimise J = g' V^-1 g, with V updated
# at every coefficient, the weights are the CBPS weights at the minimum
# (standardised or not), and the outcome is a weighted least squares fit with
# weights sv times the CBPS weights. The influence of a person is the derivative
# of the fit's coefficients with respect to its case weight, by central
# differences with Richardson extrapolation (steps h and h / 2, h = 1e-4).
#
# Each refit solves the first-order condition by Newton steps from the
# unperturbed minimum with the Hessian there, to a gradient of 1e-15. The
# gradient 2 G' v - (v' dV/dbeta_j v)_j, v = V^-1 g, is written out with the
# derivatives of g and V in closed form, and checked against numDeriv's gradient
# of J at the logistic fit.
# Run: Rscript balance_cu.R > balance_cu.json
suppressMessages({
  library(jsonlite)
  library(numDeriv)
})
ref <- fromJSON("balance.json")
di <- ref$ij_data

ma <- list(eps = 1e-4, d = 1e-3, r = 6, v = 2)
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
dwbal <- function(P, e) {
  if (P$att) ifelse(P$t == 1, 0, -P$cc * e / (1 - e))
  else ifelse(P$t == 1, -(1 - e) / e, -e / (1 - e))
}
gmom <- function(P, beta) {
  e <- plogis(drop(P$X %*% beta))
  c(crossprod(P$X, P$s * (P$t - e)), crossprod(P$X, P$s * wbal(P, e))) / P$n
}
# The blocks of V per person and their derivatives in the linear predictor.
vparts <- function(P, e) {
  if (P$att) list(a = e * (1 - e), b = P$cc * e, c = P$cc^2 * e / (1 - e),
                  da = e * (1 - e) * (1 - 2 * e), db = P$cc * e * (1 - e), dc = P$cc^2 * e / (1 - e))
  else list(a = e * (1 - e), b = rep(1, length(e)), c = 1 / (e * (1 - e)),
            da = e * (1 - e) * (1 - 2 * e), db = rep(0, length(e)), dc = -(1 - 2 * e) / (e * (1 - e)))
}
Vmat <- function(P, beta) {
  e <- plogis(drop(P$X %*% beta)); X <- P$X; s <- P$s; q <- vparts(P, e)
  G <- function(w) crossprod(X, s * w * X)
  rbind(cbind(G(q$a), G(q$b)), cbind(G(q$b), G(q$c))) / P$n
}
J_cu <- function(P, beta) { g <- gmom(P, beta); drop(t(g) %*% solve(Vmat(P, beta), g)) }
grad_cu <- function(P, beta) {
  X <- P$X; s <- P$s; k <- ncol(X)
  e <- plogis(drop(X %*% beta)); q <- vparts(P, e)
  G <- rbind(crossprod(X, s * (-e * (1 - e)) * X), crossprod(X, s * dwbal(P, e) * X)) / P$n
  v <- solve(Vmat(P, beta), gmom(P, beta))
  r1 <- drop(X %*% v[1:k]); r2 <- drop(X %*% v[(k + 1):(2 * k)])
  quadv <- drop(crossprod(X, s * (q$da * r1^2 + 2 * q$db * r1 * r2 + q$dc * r2^2))) / P$n
  drop(2 * t(G) %*% v) - quadv
}
cu_min <- function(P, start) {
  o <- optim(start, function(b) J_cu(P, b), function(b) grad_cu(P, b),
             method = "BFGS", control = list(maxit = 5000, reltol = 1e-16, abstol = 0))
  b <- o$par
  for (it in 1:30) {
    g <- grad_cu(P, b)
    if (max(abs(g)) < 1e-15) break
    b <- b - solve(jacobian(function(z) grad_cu(P, z), b, method.args = ma), g)
  }
  b
}
newton_from <- function(P, b, H) {
  for (it in 1:50) {
    g <- grad_cu(P, b)
    if (max(abs(g)) < 1e-15) break
    b <- b - solve(H, g)
  }
  b
}
cbps_weights <- function(P, beta, std) {
  e <- plogis(drop(P$X %*% beta)); t <- P$t
  r <- if (P$att) ifelse(t == 1, P$cc, P$cc * e / (1 - e)) else ifelse(t == 1, 1 / e, 1 / (1 - e))
  if (std) for (g in 0:1) r[t == g] <- r[t == g] / sum(P$s[t == g] * r[t == g])
  r
}
wls <- function(y, Z, ww) { solve(crossprod(Z, ww * Z), crossprod(Z, ww * y)) }
ij_deriv <- function(f, s, h = 1e-4) {
  n <- length(s); k <- length(f(s)); out <- matrix(0, n, k)
  for (i in 1:n) {
    d <- function(step) { sp <- s; sm <- s; sp[i] <- s[i] + step; sm[i] <- s[i] - step
      (f(sp) - f(sm)) / (2 * step) }
    out[i, ] <- (4 * d(h / 2) - d(h)) / 3
  }
  out
}

Xi <- cbind(1, di$x1, di$x2, di$x3)
Zr <- cbind(1, di$a, di$x1)
cases <- lapply(list(list("ATE", TRUE), list("ATT", FALSE)), function(cs) {
  est <- cs[[1]]; std <- cs[[2]]; att <- est != "ATE"
  P0 <- prob(Xi, di$a, di$s, att)
  ml0 <- ml_fit(Xi, di$a, di$s)
  b0 <- cu_min(P0, ml0)
  # The closed-form gradient against numDeriv's, away from the minimum, where
  # the gradient is of order 1e-2 and numDeriv's noise about 1e-10.
  gn <- grad(function(z) J_cu(P0, z), ml0, method.args = ma)
  gap <- max(abs(grad_cu(P0, ml0) - gn)) / max(abs(gn))
  stopifnot(gap < 1e-7)
  H0 <- jacobian(function(z) grad_cu(P0, z), b0, method.args = ma)
  f <- function(sv) {
    P <- prob(Xi, di$a, sv, att)
    be <- newton_from(P, b0, H0)
    wls(di$y, Zr, sv * cbps_weights(P, be, std))
  }
  IF <- ij_deriv(f, di$s, h = 1e-4)
  list(estimand = est, ident = "over", std = std, theta = drop(f(di$s)),
       influence = as.vector(IF), k = 3, coefficients = b0, J = J_cu(P0, b0),
       gradient_check = gap)
})
out <- list(
  versions = list(r = R.version.string, numDeriv = as.character(packageVersion("numDeriv")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  ij_cbps_cu = cases)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null"))
