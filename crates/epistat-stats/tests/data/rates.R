# Reference values for epistat_stats::rates and the Byar interval of
# epistat_stats::poisson.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/rates.R
#
# Crude rates come from stats::poisson.test, rate ratios from the closed
# forms (Wald on the log scale) and from poisson.test (exact p-value and the
# estimate). The standardised rate and its gamma interval are written out from
# Fay and Feuer (1997) with qchisq, and popEpi::rate gives the point estimate
# of the standardised rate. Byar is written out from its formula; epitools is
# not installed.
suppressMessages({
  library(jsonlite)
  library(popEpi)
})
set.seed(20261122)

crude <- list()
for (d in c(0, 1, 2, 7, 30, 250, 4000)) {
  for (y in c(1, 12.5, 4321.75)) {
    for (level in c(0.9, 0.95, 0.99)) {
      ci <- poisson.test(d, y, conf.level = level)$conf.int
      a <- (1 - level) / 2
      z <- qnorm(1 - a)
      byar_lo <- if (d == 0) 0 else { b <- 1 - 1 / (9 * d) - z / (3 * sqrt(d)); if (b > 0) d * b^3 / y else NA }
      byar_hi <- (d + 1) * (1 - 1 / (9 * (d + 1)) + z / (3 * sqrt(d + 1)))^3 / y
      crude[[length(crude) + 1]] <- list(events = d, time = y, level = level, scale = 1000,
        rate = d / y * 1000, lower = ci[1] * 1000, upper = ci[2] * 1000,
        byar_lower = byar_lo, byar_upper = byar_hi)
    }
  }
}

std_case <- function(name, d, y, w, level = 0.95) {
  keep <- w > 0
  w1 <- w / sum(w)
  w1k <- w1[keep]; dk <- d[keep]; yk <- y[keep]
  r <- sum(w1k * dk / yk)
  v <- sum(w1k^2 * dk / yk^2)
  wm <- max(w1k / yk)
  a <- (1 - level) / 2
  lower <- if (r == 0) 0 else v / (2 * r) * qchisq(a, 2 * r^2 / v)
  upper <- (v + wm^2) / (2 * (r + wm)) * qchisq(1 - a, 2 * (r + wm)^2 / (v + wm^2))
  dt <- data.frame(d = dk, y = yk, a = seq_along(dk))
  pe <- as.data.frame(rate(dt, obs = "d", pyrs = "y", adjust = "a", weights = w1k))
  list(name = name, events = d, time = y, weights = w, level = level,
       rate = r, variance = v, lower = lower, upper = upper,
       popepi_rate = pe$rate.adj)
}
std <- list()
k <- 60
y <- round(runif(k, 50, 20000), 1)
d <- rpois(k, y * runif(k, 0.0005, 0.02))
d[c(3, 17, 40)] <- 0
w <- runif(k, 0.1, 2)
std[[1]] <- std_case("sixty strata with zero counts", d, y, w)
std[[2]] <- std_case("sixty strata, level 0.99", d, y, w, 0.99)
esp <- c(1000, 4000, 5500, 5500, 5500, 6000, 6000, 6500, 7000, 7000, 7000, 7000, 6500, 6000, 5500, 5000, 4000, 2500, 1500, 800, 200)
y2 <- round(runif(21, 500, 8000), 1); d2 <- rpois(21, y2 * seq(0.0002, 0.03, length.out = 21))
std[[3]] <- std_case("European standard population 1976, 21 age groups", d2, y2, esp)
std[[4]] <- std_case("all counts zero", rep(0, 5), c(10, 20, 30, 40, 50), rep(1, 5))
std[[5]] <- std_case("a weight of zero on an empty stratum", c(3, 4, 0), c(100, 200, 0), c(1, 2, 0))

pairs <- list()
for (e in list(c(5, 100, 9, 120), c(1, 10, 1, 10), c(40, 3e3, 25, 2e3), c(300, 1e5, 250, 9e4), c(2, 50, 90, 100), c(0, 5, 3, 7))) {
  e1 <- e[1]; t1 <- e[2]; e0 <- e[3]; t0 <- e[4]
  pt <- poisson.test(c(e1, e0), c(t1, t0))
  out <- list(e1 = e1, t1 = t1, e0 = e0, t0 = t0, exact_p = pt$p.value, estimate = unname(pt$estimate))
  for (level in c(0.95, 0.99)) {
    z <- qnorm(1 - (1 - level) / 2)
    if (e1 > 0 && e0 > 0) {
      lr <- log(e1 / t1) - log(e0 / t0); se <- sqrt(1 / e1 + 1 / e0)
      out[[paste0("rr_", level)]] <- c(exp(lr), exp(lr - z * se), exp(lr + z * se), 2 * pnorm(-abs(lr / se)))
    }
    dd <- e1 / t1 - e0 / t0; sd <- sqrt(e1 / t1^2 + e0 / t0^2)
    out[[paste0("rd_", level)]] <- c(dd, dd - z * sd, dd + z * sd, 2 * pnorm(-abs(dd / sd)))
  }
  pairs[[length(pairs) + 1]] <- out
}
# Exact p-values on a grid, including large counts.
exact <- list()
for (t in list(c(1, 1), c(1, 3), c(7, 2.5), c(1000, 1300))) {
  for (e in list(c(0, 4), c(3, 3), c(10, 2), c(2, 15), c(100, 140), c(400, 520), c(1200, 1000))) {
    exact[[length(exact) + 1]] <- list(e1 = e[1], t1 = t[1], e0 = e[2], t0 = t[2],
      p = poisson.test(e, t)$p.value)
  }
}

write(toJSON(list(versions = list(r = R.version.string, popEpi = as.character(packageVersion("popEpi"))),
                  crude = crude, standardised = std, pairs = pairs, exact = exact),
             auto_unbox = TRUE, digits = NA, null = "null", na = "null"),
      "tests/data/rates.json")
