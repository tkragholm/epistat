# Reference fits of the parametric waiting time distribution of Støvring,
# Pottegård and Hallas (Pharmacoepidemiol Drug Saf 2016;25:1451-9), written
# out in base R from the paper, since no CRAN package fits it.
#
# The likelihood of a waiting time t in a window of delta days is
#   gamma * g(t) + (1 - gamma) / delta,
# with the forward recurrence density g(t) = (1 - F(t)) / M of a log-normal or
# a Weibull inter-arrival distribution F with mean M, estimated on the scale
# logit(gamma), mu, log(sigma) or logit(gamma), log(alpha), log(beta), by
# optim(method = "BFGS") with the analytic gradient. The duration is the 80th
# percentile of F.
#
# The data are simulated: prevalent users follow a renewal process with
# log-normal gaps that started long before the window, incident users start
# at a uniform day in it, and every user stops for good after a
# redemption with probability 0.1, which gives the reverse distribution its
# uniform part. Each person's dispensing days are written, and the
# waiting time of a day d in the window [start, end) is d - start + 1/2
# forward and end - d - 1/2 reverse.
#
# For the comparison the script also writes, at its optimum, the gradient
# norm and the smallest eigenvalue of the Hessian of the mean negative
# log-likelihood (optimHess), from which the test derives its tolerance.
#
# Run from the crate directory:
#   Rscript tests/data/wtd_reference.R > tests/data/wtd_reference.json
suppressMessages(library(jsonlite))
set.seed(20261126)

start <- 0; end <- 365; delta <- end - start
n_prev <- 1500; n_inc <- 600
meanlog <- log(90) - 0.5 * log(1 + (30 / 90)^2); sdlog <- sqrt(log(1 + (30 / 90)^2))
person <- integer(0); day <- integer(0)
for (p in seq_len(n_prev)) {
  # Started long ago and still in treatment 400 days before the window.
  t <- -400 - runif(1, 0, 200)
  days <- integer(0)
  while (t < end + 200) {
    t <- t + rlnorm(1, meanlog, sdlog)
    days <- c(days, floor(t))
    if (t > -400 && runif(1) < 0.1) break
  }
  days <- days[days >= -400 & days < end + 100]
  person <- c(person, rep(p - 1L, length(days))); day <- c(day, days)
}
for (p in seq_len(n_inc)) {
  t <- runif(1, start, end)
  days <- integer(0)
  while (t < end + 100) {
    days <- c(days, floor(t))
    if (runif(1) < 0.1) break
    t <- t + rlnorm(1, meanlog, sdlog)
  }
  person <- c(person, rep(n_prev + p - 1L, length(days))); day <- c(day, days)
}

times_of <- function(direction) {
  inside <- day >= start & day < end
  if (direction == "forward") {
    d <- tapply(day[inside], person[inside], min)
    as.numeric(d) - start + 0.5
  } else {
    d <- tapply(day[inside], person[inside], max)
    end - as.numeric(d) - 0.5
  }
}

natural <- function(family, x) {
  g <- 1 / (1 + exp(-x[1]))
  if (family == "lognormal") c(g, x[2], exp(x[3])) else c(g, exp(x[2]), exp(x[3]))
}
dens <- function(family, a, b, t) {
  if (family == "lognormal") {
    c0 <- exp(-a - b^2 / 2); z <- (log(t) - a) / b
    g <- c0 * pnorm(-z)
    list(g = g, d1 = c0 * dnorm(z) / b - g, d2 = c0 * dnorm(z) * z - b^2 * g)
  } else {
    bt <- b * t; pw <- bt^a
    g <- b * exp(-lgamma(1 + 1 / a) - pw)
    list(g = g, d1 = g * (digamma(1 + 1 / a) / a - a * pw * log(bt)), d2 = g * (1 - a * pw))
  }
}
objective <- function(family, t) function(x) {
  p <- natural(family, x); d <- dens(family, p[2], p[3], t)
  -mean(log(p[1] * d$g + (1 - p[1]) / delta))
}
gradient <- function(family, t) function(x) {
  p <- natural(family, x); d <- dens(family, p[2], p[3], t)
  m <- p[1] * d$g + (1 - p[1]) / delta
  -c(mean(p[1] * (1 - p[1]) * (d$g - 1 / delta) / m),
     mean(p[1] * d$d1 / m), mean(p[1] * d$d2 / m))
}
fit <- function(family, t) {
  x0 <- if (family == "lognormal") c(0, mean(log(t)), log(sd(log(t)))) else c(0, 0, -log(mean(t)))
  f <- objective(family, t); g <- gradient(family, t)
  o <- optim(x0, f, g, method = "BFGS", control = list(reltol = 1e-16, maxit = 5000))
  # Polish with Newton steps on the numerical Hessian of the analytic gradient.
  x <- o$par
  for (i in 1:20) {
    h <- optimHess(x, f, g)
    x <- x - solve(h, g(x))
  }
  h <- optimHess(x, f, g)
  p <- natural(family, x)
  tau <- if (family == "lognormal") exp(p[2] + p[3] * qnorm(0.8)) else (-log(0.2))^(1 / p[2]) / p[3]
  list(family = family, x = x, prevalent = p[1], a = p[2], b = p[3],
       loglik = -f(x) * length(t), duration = tau,
       gradient_norm = sqrt(sum(g(x)^2)), min_eigen = min(eigen(h, only.values = TRUE)$values))
}

fits <- list()
for (direction in c("forward", "reverse")) {
  t <- times_of(direction)
  for (family in c("lognormal", "weibull")) {
    r <- fit(family, t); r$direction <- direction; r$n <- length(t)
    fits[[length(fits) + 1]] <- r
  }
}

cat(toJSON(list(
  versions = list(R = as.character(getRversion())),
  window = list(start = start, end = end),
  person = person, day = day,
  fits = fits
), digits = NA, auto_unbox = TRUE))
