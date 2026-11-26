# The outcome model of stacked trials with a competing event, written out in R
# from its definition, as the reference of
# epistat_survival::weighting::trials::{fit_msm_competing,
# cumulative_incidence_competing}.
#
# Person-period rows are simulated: a baseline covariate x, an assigned arm a,
# follow-up times 0 to 5, a weight per row, an event y and a competing event d
# that ends the follow-up. In a period the competing event comes first. The
# script fits
#   the competing event: glm(d ~ a + t + t^2 + x, weights = w), all rows;
#   the event:           glm(y ~ a + t + t^2 + x, weights = w), rows with d = 0;
# and the stacked sandwich covariance of the two clustered by person, with the
# cross block A^-1 (sum_g U_g V_g') B^-1 G / (G - 1) written out and the two
# diagonal blocks checked against sandwich::vcovCL(type = "HC0"), each with its
# own persons as clusters; the cross block has G, all persons. The risk of
# the event by t, F_t = sum_{s <= t} p_s (1 - q_s) S_{s-1} with S the product
# of (1 - p)(1 - q), and of the competing event, sum q_s S_{s-1}, are averaged
# over the persons at time 0 with the arm set, and their standard errors are
# sqrt(g' V g) with g the gradient by central differences (step 1e-6).
# Numbers are written with 17 digits.
#
# Run from the crate directory:
#   Rscript tests/data/trial_competing.R > tests/data/trial_competing.json
suppressMessages({
  library(jsonlite)
  library(sandwich)
})
set.seed(20261126)
n <- 600
rows <- list()
for (i in seq_len(n)) {
  x <- rnorm(1)
  a <- rbinom(1, 1, plogis(0.3 * x))
  w <- round(runif(1, 0.5, 2), 3)
  for (t in 0:5) {
    d <- rbinom(1, 1, plogis(-3 + 0.5 * x + 0.1 * t - 0.2 * a))
    y <- if (d == 1) 0 else rbinom(1, 1, plogis(-2.6 - 0.5 * a + 0.4 * x + 0.15 * t))
    rows[[length(rows) + 1]] <- data.frame(id = i, a = a, t = t, x = round(x, 4),
      w = w * (1 + 0.05 * t), y = y, d = d)
    if (y == 1 || d == 1) break
  }
}
dat <- do.call(rbind, rows)
X <- cbind(1, dat$a, dat$t, dat$t^2, dat$x)
fit_d <- suppressWarnings(glm(d ~ a + t + I(t^2) + x, family = binomial, weights = w, data = dat))
keep <- dat$d == 0
fit_y <- suppressWarnings(glm(y ~ a + t + I(t^2) + x, family = binomial, weights = w,
  data = dat[keep, ]))
by <- coef(fit_y); gam <- coef(fit_d)
p_y <- plogis(drop(X %*% by)); p_d <- plogis(drop(X %*% gam))
U <- (keep * dat$w * (dat$y - p_y)) * X
V <- (dat$w * (dat$d - p_d)) * X
A <- crossprod(X[keep, ], (dat$w * p_y * (1 - p_y))[keep] * X[keep, ])
B <- crossprod(X, dat$w * p_d * (1 - p_d) * X)
Ug <- rowsum(U, dat$id); Vg <- rowsum(V, dat$id)
G <- nrow(Ug)
adj <- G / (G - 1)
cross <- solve(A) %*% crossprod(Ug, Vg) %*% solve(B) * adj
# The event model alone has the persons with a row without d as its clusters.
Gy <- length(unique(dat$id[keep]))
vy <- solve(A) %*% crossprod(Ug) %*% solve(A) * Gy / (Gy - 1)
vd <- solve(B) %*% crossprod(Vg) %*% solve(B) * adj
vcl_y <- vcovCL(fit_y, cluster = dat$id[keep], type = "HC0")
vcl_d <- vcovCL(fit_d, cluster = dat$id, type = "HC0")
# glm's unscaled covariance is formed with the IRLS weights of its last
# iteration, so the blocks agree to the convergence of the fit.
stopifnot(max(abs(vy - vcl_y)) < 1e-5 * max(abs(vcl_y)),
  max(abs(vd - vcl_d)) < 1e-5 * max(abs(vcl_d)))
stacked <- rbind(cbind(vcl_y, cross), cbind(t(cross), vcl_d))

base <- dat[dat$t == 0, ]
risks <- function(theta, arm) {
  b <- theta[1:5]; g <- theta[6:10]
  m <- nrow(base)
  s <- rep(1, m); f <- rep(0, m); cc <- rep(0, m)
  out_f <- numeric(6); out_c <- numeric(6)
  for (t in 0:5) {
    x <- cbind(1, arm, t, t^2, base$x)
    p <- plogis(drop(x %*% b)); q <- plogis(drop(x %*% g))
    f <- f + p * (1 - q) * s
    cc <- cc + q * s
    s <- s * (1 - p) * (1 - q)
    out_f[t + 1] <- mean(f); out_c[t + 1] <- mean(cc)
  }
  c(out_f, out_c)
}
theta <- c(by, gam)
gradient <- function(arm) {
  h <- 1e-6
  sapply(seq_along(theta), function(k) {
    e <- replace(numeric(length(theta)), k, h)
    (risks(theta + e, arm) - risks(theta - e, arm)) / (2 * h)
  })
}
arm <- function(a) {
  r <- risks(theta, a)
  J <- gradient(a)
  se <- sqrt(rowSums((J %*% stacked) * J))
  list(risk = r[1:6], se = se[1:6], competing_risk = r[7:12], competing_se = se[7:12])
}
out <- list(
  versions = list(R = as.character(getRversion()),
    sandwich = as.character(packageVersion("sandwich")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  rows = list(id = dat$id, a = dat$a, t = dat$t, x = dat$x, w = dat$w, y = dat$y, d = dat$d),
  event_coef = as.numeric(by), competing_coef = as.numeric(gam),
  event_vcov = as.numeric(t(vcl_y)), competing_vcov = as.numeric(t(vcl_d)),
  cross = as.numeric(t(cross)),
  arm0 = arm(0), arm1 = arm(1))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE))
