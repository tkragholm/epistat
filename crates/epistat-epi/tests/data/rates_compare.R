# Reference values for the comparison of two directly standardised rates
# (epistat_stats::rates::standardised_ratio and standardised_difference) and
# for the tests of a common rate ratio over strata
# (epistat_stats::mantel_haenszel::rate_ratio_breslow_day and
# rate_ratio_homogeneity).
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/rates_compare.R
#
# No CRAN package gives an interval for the ratio of two standardised rates
# (dsr, which had the log-normal one, is archived), so the F interval of Fay
# (1999) and the modified F interval of Tiwari, Clegg and Zou (2006) are
# written out from the papers with qf. The shift of the modified interval is
# checked inside this script against asht::wspoissonTest(wmtype = "tcz"),
# Fay's own implementation of the modified gamma interval of one rate, which
# uses the same mean and mean square of the weights. The tests of a common
# rate ratio come from glm: the deviance and the Pearson statistic of the
# binomial model with offset log(t1 / t0), and the likelihood ratio test of
# the interaction in a Poisson model, which equals the deviance.
suppressMessages({
  library(jsonlite)
  library(asht)
})
set.seed(20261126)

parts <- function(d, y, w) {
  keep <- w > 0
  s <- w[keep] / sum(w)
  W <- s / y[keep]
  dk <- d[keep]
  list(r = sum(W * dk), v = sum(W^2 * dk), wmax = max(W), wmean = mean(W),
       wsq = mean(W^2), W = W, d = dk)
}

ratio_case <- function(name, d1, y1, d0, y0, w, level = 0.95) {
  a <- (1 - level) / 2
  z <- qnorm(1 - a)
  p1 <- parts(d1, y1, w)
  p0 <- parts(d0, y0, w)
  df <- function(r, v) 2 * r^2 / v
  fint <- function(s1, u1, s0, u0) {
    lower <- if (p1$r == 0) 0 else
      p1$r / (p0$r + s0) * qf(a, df(p1$r, p1$v), df(p0$r + s0, p0$v + u0))
    upper <- (p1$r + s1) / p0$r * qf(1 - a, df(p1$r + s1, p1$v + u1), df(p0$r, p0$v))
    c(lower, upper)
  }
  fay <- fint(p1$wmax, p1$wmax^2, p0$wmax, p0$wmax^2)
  tcz <- fint(p1$wmean, p1$wsq, p0$wmean, p0$wsq)
  # The shift of the modified interval is the one of asht's "tcz" upper limit
  # for one rate.
  for (p in list(p1, p0)) {
    ref <- wspoissonTest(p$d, p$W, wmtype = "tcz", conf.level = level)$conf.int[2]
    mine <- (p$v + p$wsq) / (2 * (p$r + p$wmean)) *
      qchisq(1 - a, 2 * (p$r + p$wmean)^2 / (p$v + p$wsq))
    stopifnot(abs(ref - mine) <= 1e-12 * ref)
  }
  out <- list(name = name, events1 = d1, time1 = y1, events0 = d0, time0 = y0,
              weights = w, level = level, ratio = p1$r / p0$r,
              fay = fay, tcz = tcz, difference = p1$r - p0$r)
  sd <- sqrt(p1$v + p0$v)
  out$difference_limits <- c(p1$r - p0$r - z * sd, p1$r - p0$r + z * sd,
                             2 * pnorm(-abs((p1$r - p0$r) / sd)))
  if (p1$r > 0) {
    lr <- log(p1$r / p0$r)
    se <- sqrt(p1$v / p1$r^2 + p0$v / p0$r^2)
    out$log_limits <- c(exp(lr - z * se), exp(lr + z * se), 2 * pnorm(-abs(lr / se)))
  }
  out
}

ratios <- list()
k <- 18
y1 <- round(runif(k, 200, 9000), 1); y0 <- round(runif(k, 200, 9000), 1)
lam <- seq(0.0002, 0.02, length.out = k)
d1 <- rpois(k, 1.4 * y1 * lam); d0 <- rpois(k, y0 * lam)
esp <- c(5000, 5500, 5500, 5500, 6000, 6000, 6500, 7000, 7000, 7000, 7000, 6500,
         5500, 4000, 2500, 1500, 800, 200)
ratios[[1]] <- ratio_case("eighteen age groups, ESP 2013 above 15", d1, y1, d0, y0, esp)
ratios[[2]] <- ratio_case("eighteen age groups, level 0.99", d1, y1, d0, y0, esp, 0.99)
d1s <- rpois(k, 0.05 * y1 * lam); d0s <- rpois(k, 0.05 * y0 * lam)
ratios[[3]] <- ratio_case("sparse counts", d1s, y1, d0s, y0, esp)
ratios[[4]] <- ratio_case("no events in the group", rep(0, 4), c(100, 200, 300, 400),
                          c(2, 3, 1, 5), c(150, 250, 100, 300), c(1, 2, 3, 4))
ratios[[5]] <- ratio_case("the two regions of the rates guide",
                          c(3, 5, 6, 5), c(6000, 5000, 3000, 1500),
                          c(1, 2, 6, 11), c(2000, 3000, 4500, 5500),
                          c(8000, 8000, 7500, 7000))
ratios[[6]] <- ratio_case("a stratum without weight", c(3, 5, 6, 0), c(600, 500, 300, 0),
                          c(1, 2, 6, 9), c(200, 300, 450, 10), c(2, 3, 1, 0))
ratios[[7]] <- ratio_case("large counts", c(4000, 9000, 15000), c(2e6, 3e6, 2.5e6),
                          c(3500, 8000, 16000), c(1.9e6, 3.1e6, 2.6e6), c(1, 1, 1))

homog_case <- function(name, x1, t1, x0, t0) {
  keep <- t1 > 0 & t0 > 0 & (x1 + x0) > 0
  n <- x1 + x0
  o <- log(t1 / t0)
  fit <- glm(cbind(x1, x0)[keep, , drop = FALSE] ~ 1, offset = o[keep],
             family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100))
  ratio <- exp(unname(coef(fit)))
  pearson <- sum(residuals(fit, type = "pearson")^2)
  # The same likelihood ratio from Poisson models with and without the
  # interaction.
  k <- length(x1)
  long <- data.frame(d = c(x1, x0), y = c(t1, t0), exposed = rep(c(1, 0), each = k),
                     stratum = factor(rep(seq_len(k), 2)))
  long <- long[rep(keep, 2), ]
  main <- glm(d ~ exposed + stratum, offset = log(y), family = poisson, data = long,
              control = glm.control(epsilon = 1e-14, maxit = 100))
  full <- glm(d ~ exposed * stratum, offset = log(y), family = poisson, data = long,
              control = glm.control(epsilon = 1e-14, maxit = 100))
  # Breslow-Day for rates with the Mantel-Haenszel ratio, written out.
  tt <- t1 + t0
  mh <- sum(x1 * t0 / tt) / sum(x0 * t1 / tt)
  p <- mh * t1 / (mh * t1 + t0)
  e <- (x1 - n * p)[keep]; v <- (n * p * (1 - p))[keep]
  bd <- sum(e^2 / v)
  list(name = name, x1 = x1, t1 = t1, x0 = x0, t0 = t0, df = sum(keep) - 1,
       ratio = ratio, deviance = deviance(fit), pearson = pearson,
       poisson_lr = deviance(main) - deviance(full), mh = mh,
       breslow_day = bd, tarone = bd - sum(e)^2 / sum(v))
}

homog <- list()
homog[[1]] <- homog_case("the four age bands of the rates guide",
                         c(3, 5, 6, 5), c(6000, 5000, 3000, 1500),
                         c(1, 2, 6, 11), c(2000, 3000, 4500, 5500))
ks <- 12
t1 <- round(runif(ks, 100, 5000), 2); t0 <- round(runif(ks, 100, 5000), 2)
base <- runif(ks, 0.002, 0.03)
homog[[2]] <- homog_case("twelve strata with a common ratio",
                         rpois(ks, 1.8 * t1 * base), t1, rpois(ks, t0 * base), t0)
homog[[3]] <- homog_case("twelve strata with ratios rising",
                         rpois(ks, seq(0.5, 4, length.out = ks) * t1 * base), t1,
                         rpois(ks, t0 * base), t0)
homog[[4]] <- homog_case("an empty stratum and one without unexposed time",
                         c(4, 0, 7, 2, 3), c(100, 50, 300, 80, 40),
                         c(6, 0, 5, 9, 0), c(200, 60, 250, 90, 0))

write(toJSON(list(versions = list(r = R.version.string,
                                  asht = as.character(packageVersion("asht"))),
                  ratios = ratios, homogeneity = homog),
             auto_unbox = TRUE, digits = 17, null = "null", na = "null"),
      "tests/data/rates_compare.json")
