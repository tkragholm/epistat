# The influence function of the Brier score with a Cox censoring model (Breslow
# ties), by numerical differentiation. Times are rounded so that deaths and
# censorings tie. The score is
#   B(w) = sum_i w_i r_i(w) / sum_i w_i,
# with r_i the weighted squared error of row i and the censoring model refitted
# with the case weights w. The influence function of row k is
#   IF_k = N dB / d w_k at w = 1,
# taken as the central difference at two step sizes, and Richardson's
# combination of the two. riskRegression's influence function counts a tied time
# once, so it cannot be the reference here.
# Run: Rscript metrics_censoring_ties.R > metrics_censoring_ties.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261122)
n <- 90
x1 <- round(rnorm(n), 1)
x2 <- rbinom(n, 1, 0.5)
t1 <- rexp(n, 0.4 * exp(0.5 * x1))
cc <- rexp(n, 0.3 * exp(0.4 * x2 + 0.3 * x1))
time <- round(pmin(t1, cc), 1)
time[time == 0] <- 0.1
status <- as.integer(t1 <= cc)
d <- data.frame(time = time, status = status, x1 = x1, x2 = x2)
cat_ties <- sum(duplicated(d$time))
fit_model <- coxph(Surv(time, status) ~ x1, data = d)
times <- c(0.6, 1.2, 2)
stopifnot(max(times) < max(d$time))
# risks from the Cox model for the event, at each row
risk <- sapply(times, function(t) {
  H <- basehaz(fit_model, centered = FALSE)
  h <- approx(H$time, H$hazard, xout = t, method = "constant", f = 0, yleft = 0, rule = 2)$y
  1 - exp(-h * exp(coef(fit_model) * d$x1))
})

cens_x <- cbind(x1 = d$x1, x2 = d$x2)
brier_w <- function(w, eps = 1e-14) {
  f <- suppressWarnings(coxph(Surv(time, status == 0) ~ x1 + x2, data = d, weights = w, ties = "breslow",
                              control = coxph.control(eps = eps, toler.chol = 1e-14, iter.max = 100)))
  b <- coef(f)
  q <- as.numeric(exp(cens_x %*% b))
  H <- basehaz(f, centered = FALSE)
  lam_before <- c(0, H$hazard)[1 + findInterval(d$time, H$time, left.open = TRUE)]
  lam_through <- function(t) c(0, H$hazard)[1 + findInterval(t, H$time)]
  sapply(seq_along(times), function(j) {
    t <- times[j]
    r <- ifelse(d$time <= t,
                ifelse(d$status == 1, (1 - risk[, j])^2 / exp(-q * lam_before), 0),
                risk[, j]^2 / exp(-q * lam_through(t)))
    sum(w * r) / sum(w)
  })
}
w0 <- rep(1, n)
wmax_fit <- suppressWarnings(coxph(Surv(time, status == 0) ~ x1 + x2, data = d, ties = "breslow",
                                   control = coxph.control(eps = 1e-14, toler.chol = 1e-14, iter.max = 100)))
Hm <- basehaz(wmax_fit, centered = FALSE)
qm <- as.numeric(exp(cens_x %*% coef(wmax_fit)))
wmax <- 1 / exp(-max(qm) * max(Hm$hazard))
b0 <- brier_w(w0)
b0_loose <- brier_w(w0, eps = 1e-12)
fit_error <- max(abs(b0 - b0_loose))

fd <- function(k, h) {
  wp <- w0; wp[k] <- 1 + h
  wm <- w0; wm[k] <- 1 - h
  n * (brier_w(wp) - brier_w(wm)) / (2 * h)
}
h1 <- 1e-3
h2 <- 2e-3
d1 <- t(sapply(seq_len(n), fd, h = h1))
d2 <- t(sapply(seq_len(n), fd, h = h2))
rows <- function(m) lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))

cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  time = d$time, status = d$status, x1 = d$x1, x2 = d$x2, times = times,
  n_tied_times = cat_ties, risk = rows(risk), brier = b0, wmax = wmax, fit_error = fit_error,
  h1 = h1, h2 = h2, d1 = rows(d1), d2 = rows(d2)
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
