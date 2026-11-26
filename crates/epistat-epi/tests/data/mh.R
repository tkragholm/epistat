# Reference values for epistat_stats::mantel_haenszel from metafor::rma.mh and
# stats::mantelhaen.test.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/mh.R
#
# rma.mh runs with add = 0 and to = "none", so no continuity constant is added
# to a zero cell, and with correct = FALSE for the uncorrected statistic. Its
# "MH" statistic is the Mantel-Haenszel chi-square on n - 1, "BD" and "TA" the
# Breslow-Day and Tarone statistics with k.pos strata. stats::mantelhaen.test
# gives the odds ratio, its interval and the corrected and uncorrected
# statistic. epiR::epi.2by2 would be the epidemiologist's front end, but epiR
# needs the sf package, which does not build on this machine, so it is not
# checked. Values are written with 17 significant digits.
suppressMessages({
  library(jsonlite)
  library(metafor)
})
set.seed(20261122)

mh <- function(ai, bi, ci, di, measure, correct = FALSE) {
  suppressWarnings(rma.mh(ai = ai, bi = bi, ci = ci, di = di, measure = measure,
                          add = 0, to = "none", drop00 = FALSE, correct = correct))
}
num <- function(x) if (length(x) == 0 || is.na(x)) NULL else unname(as.numeric(x))
pooled <- function(r, exp_scale) {
  list(estimate = num(if (exp_scale) exp(r$beta) else r$beta), se = num(r$se),
       lower = num(if (exp_scale) exp(r$ci.lb) else r$ci.lb),
       upper = num(if (exp_scale) exp(r$ci.ub) else r$ci.ub), p = num(r$pval))
}

table_case <- function(name, a, b, c, d) {
  out <- list(name = name, a = I(a), b = I(b), c = I(c), d = I(d))
  for (m in c("OR", "RR", "RD")) {
    r <- mh(a, b, c, d, m)
    out[[m]] <- pooled(r, m != "RD")
  }
  r <- mh(a, b, c, d, "OR")
  # A stratum with an empty arm carries no information on homogeneity and
  # divides by zero in rma.mh's Breslow-Day statistic, so that statistic is
  # computed without such strata, which leave the pooled odds ratio unchanged.
  keep <- (a + b) > 0 & (c + d) > 0
  rbd <- mh(a[keep], b[keep], c[keep], d[keep], "OR")
  out$test_uncorrected <- list(stat = num(r$MH), p = num(r$MHp))
  rc <- mh(a, b, c, d, "OR", correct = TRUE)
  out$test_metafor_corrected <- list(stat = num(rc$MH), p = num(rc$MHp))
  out$bd <- list(stat = num(rbd$BD), p = num(rbd$BDp), k_pos = num(rbd$k.pos))
  out$ta <- list(stat = num(rbd$TA), p = num(rbd$TAp))
  if (length(a) > 1 || TRUE) {
    arr <- array(0, c(2, 2, length(a)))
    for (i in seq_along(a)) arr[, , i] <- matrix(c(a[i], c[i], b[i], d[i]), 2)
    mt <- tryCatch(suppressWarnings(mantelhaen.test(arr, correct = FALSE)), error = function(e) NULL)
    mtc <- tryCatch(suppressWarnings(mantelhaen.test(arr, correct = TRUE)), error = function(e) NULL)
    if (!is.null(mt)) {
      out$mantelhaen <- list(stat = num(mt$statistic), p = num(mt$p.value), or = num(mt$estimate),
                             lower = num(mt$conf.int[1]), upper = num(mt$conf.int[2]),
                             stat_corrected = num(mtc$statistic), p_corrected = num(mtc$p.value))
    }
  }
  out
}

tables <- list()
bcg <- dat.bcg
tables[[1]] <- table_case("BCG, 13 trials", bcg$tpos, bcg$tneg, bcg$cpos, bcg$cneg)
sim <- function(k, p1 = 0.05, p0 = 0.03, nmax = 400) {
  n1 <- sample(20:nmax, k, TRUE); n0 <- sample(20:nmax, k, TRUE)
  a <- rbinom(k, n1, p1 * exp(rnorm(k, 0, .3))); c <- rbinom(k, n0, p0 * exp(rnorm(k, 0, .3)))
  list(a = a, b = n1 - a, c = c, d = n0 - c)
}
s <- sim(8, 0.01, 0.01, 60); s$a[2] <- 0; s$c[5] <- 0; s$b[7] <- 0
tables[[2]] <- table_case("eight strata with zero cells", s$a, s$b, s$c, s$d)
s <- sim(6); s$a[3] <- 0; s$b[3] <- 0; s$c[4] <- 0; s$d[4] <- 0
tables[[3]] <- table_case("empty exposed and unexposed arms in two strata", s$a, s$b, s$c, s$d)
s <- sim(5); s$a[1] <- 0; s$c[1] <- 0
tables[[4]] <- table_case("a stratum without cases", s$a, s$b, s$c, s$d)
tables[[5]] <- table_case("one stratum", 12, 30, 5, 41)
tables[[6]] <- table_case("two strata", c(12, 7), c(30, 25), c(5, 9), c(41, 33))
s <- sim(500, 0.08, 0.05, 150)
tables[[7]] <- table_case("500 strata", s$a, s$b, s$c, s$d)
tables[[8]] <- table_case("counts of 10^7", c(2.1e6, 3.3e6, 1.2e6, 4.0e6), c(9.2e6, 8.7e6, 9.9e6, 7.4e6),
                          c(1.6e6, 2.9e6, 1.0e6, 3.1e6), c(9.9e6, 9.1e6, 9.8e6, 8.5e6))
tables[[9]] <- table_case("pooled odds ratio of 1", c(10, 20), c(20, 40), c(10, 20), c(20, 40))
s <- sim(6); s$a[1] <- 0; s$b[1] <- 0
tables[[10]] <- table_case("no exposed in one stratum", s$a, s$b, s$c, s$d)
tables[[11]] <- table_case("no exposed cases at all", c(0, 0, 0), c(10, 12, 9), c(3, 4, 2), c(20, 30, 25))

rate_case <- function(name, x1, t1, x0, t0) {
  out <- list(name = name, x1 = I(x1), t1 = I(t1), x0 = I(x0), t0 = I(t0))
  r <- suppressWarnings(rma.mh(x1i = x1, x2i = x0, t1i = t1, t2i = t0, measure = "IRR", add = 0, to = "none", drop00 = FALSE, correct = FALSE))
  out$IRR <- pooled(r, TRUE)
  out$test_uncorrected <- list(stat = num(r$MH), p = num(r$MHp))
  rc <- suppressWarnings(rma.mh(x1i = x1, x2i = x0, t1i = t1, t2i = t0, measure = "IRR", add = 0, to = "none", drop00 = FALSE, correct = TRUE))
  out$test_corrected <- list(stat = num(rc$MH), p = num(rc$MHp))
  r <- suppressWarnings(rma.mh(x1i = x1, x2i = x0, t1i = t1, t2i = t0, measure = "IRD", add = 0, to = "none", drop00 = FALSE))
  out$IRD <- pooled(r, FALSE)
  out
}
rates <- list()
k <- 12
t1 <- round(runif(k, 50, 3000), 1); t0 <- round(runif(k, 50, 3000), 1)
x1 <- rpois(k, t1 * 0.01); x0 <- rpois(k, t0 * 0.006)
rates[[1]] <- rate_case("twelve strata", x1, t1, x0, t0)
x1b <- x1; x1b[c(2, 9)] <- 0
rates[[2]] <- rate_case("twelve strata with zero counts", x1b, t1, x0, t0)
rates[[3]] <- rate_case("one stratum", 14, 900, 9, 1100)
rates[[4]] <- rate_case("no exposed events", rep(0, 3), c(100, 200, 300), c(2, 3, 4), c(150, 250, 350))
rates[[5]] <- rate_case("large counts", c(2e6, 3e6), c(1e8, 2e8), c(1.5e6, 2.5e6), c(1.1e8, 1.9e8))

write(toJSON(list(versions = list(r = R.version.string, metafor = as.character(packageVersion("metafor"))),
                  tables = tables, rates = rates),
             auto_unbox = TRUE, digits = 17, null = "null", na = "null"),
      "tests/data/mh.json")
