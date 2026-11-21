# Reference sensitivity analyses: Rosenbaum bounds and E-values.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/sensitivity.R
#
# Matched pairs: 60 simulated pairs with responses rounded to one decimal, so
# that there are ties among the absolute differences and zero differences.
#   rbounds::psens   bounds of the Wilcoxon signed rank p-value. psens rounds
#                    its p-values to four digits, so the script also records
#                    the same formula without rounding ("psens_exact"), with
#                    the upper normal tail taken directly, since psens's
#                    1 - pnorm loses digits for p-values below 1e-7.
#   rbounds::hlsens  bounds of the Hodges-Lehmann estimate, found by steps of
#                    `pr`.
# Exact signed rank tail: 25 pairs without ties, where psignrank gives the
# distribution at gamma = 1; and 12 pairs with ties, where the bound at other
# gamma is the sum over all 2^12 sign patterns.
# Matched sets: 40 sets of one treated unit and one to three controls, and 40
# sets of one treated unit and three controls.
#   sensitivitymv::senmv  for several choices of the m-statistic, with sets of
#                         different sizes and for pair differences.
#   sensitivitymw::senmw  for sets of equal size.
# E-values: EValue::evalues.RR, .OR, .HR and .MD on a grid of estimates and
# limits.
library(rbounds)
library(sensitivitymv)
library(sensitivitymw)
library(EValue)
library(jsonlite)

set.seed(20261121)
na_null <- function(x) ifelse(is.na(x), NA_real_, x)

# ---- Matched pairs -------------------------------------------------------
n <- 60
ctrl <- round(rnorm(n, 10, 2), 1)
trt <- round(ctrl + rnorm(n, 0.8, 1.5), 1)
gammas <- seq(1, 4, by = 0.5)
ps <- psens(trt, ctrl, Gamma = 4, GammaInc = 0.5)$bounds

psens_exact <- function(trt, ctrl, gamma) {
  diff <- trt - ctrl
  diff <- diff[diff != 0]
  ranks <- rank(abs(diff), ties.method = "average")
  T <- sum((diff > 0) * ranks)
  pp <- gamma / (1 + gamma); pm <- 1 / (1 + gamma)
  V <- sum(ranks^2 * pp * (1 - pp))
  c(low = pnorm((T - sum(ranks * pm)) / sqrt(V), lower.tail = FALSE),
    up = pnorm((T - sum(ranks * pp)) / sqrt(V), lower.tail = FALSE),
    stat = T, n = length(diff))
}
pe <- t(sapply(gammas, function(g) psens_exact(trt, ctrl, g)))

pr <- 0.0005
hl <- hlsens(trt, ctrl, pr = pr, Gamma = 4, GammaInc = 0.5)$bounds

# Exact signed rank tails.
n1 <- 25
d1 <- rnorm(n1, 0.5, 1)
r1 <- rank(abs(d1)); t1 <- sum(r1[d1 > 0])
exact1 <- list(diffs = d1, stat = t1, p = 1 - psignrank(t1 - 1, n1))

d2 <- c(1.5, -0.5, 2, 0.5, -1, 1, 2.5, 3, -2, 0.5, 1.5, 4)
r2 <- rank(abs(d2), ties.method = "average"); t2 <- sum(r2[d2 > 0])
pats <- as.matrix(expand.grid(rep(list(c(0, 1)), length(d2))))
stats2 <- as.vector(pats %*% r2)
tail_p <- function(p) {
  w <- apply(pats, 1, function(b) prod(ifelse(b == 1, p, 1 - p)))
  sum(w[stats2 >= t2 - 1e-9])
}
exact2 <- list(diffs = d2, gammas = c(1, 1.5, 2, 3),
               p_lower = sapply(c(1, 1.5, 2, 3), function(g) tail_p(1 / (1 + g))),
               p_upper = sapply(c(1, 1.5, 2, 3), function(g) tail_p(g / (1 + g))))

# ---- Matched sets ---------------------------------------------------------
make_sets <- function(sizes) {
  m <- max(sizes)
  y <- matrix(NA_real_, length(sizes), m)
  for (i in seq_along(sizes)) {
    base <- rnorm(1, 0, 1)
    y[i, 1:sizes[i]] <- round(base + c(0.6, rep(0, sizes[i] - 1)) +
                                rt(sizes[i], 4) * 0.8, 2)
  }
  y
}
ymv <- make_sets(sample(2:4, 40, replace = TRUE))
ymw <- make_sets(rep(4, 40))
ydiff <- round(rnorm(40, 0.3, 1), 2)

mv_cases <- list(
  list(name = "default", args = list()),
  list(name = "inner", args = list(inner = 0.5, trim = 2.5)),
  list(name = "lambda", args = list(inner = 0.25, trim = 2, lambda = 0.75)),
  list(name = "trimmean", args = list(trim = Inf, TonT = TRUE)),
  list(name = "tau", args = list(tau = 0.2)))
mv_gammas <- c(1, 1.5, 2, 3)
mv <- list()
for (cs in mv_cases) for (g in mv_gammas) {
  r <- do.call(senmv, c(list(y = ymv, gamma = g), cs$args))
  mv[[length(mv) + 1]] <- list(case = cs$name, input = "sets", gamma = g,
    pval = r$pval, deviate = r$deviate, statistic = r$statistic,
    expectation = r$expectation, variance = r$variance)
}
for (g in mv_gammas) {
  r <- senmv(ydiff, gamma = g)
  mv[[length(mv) + 1]] <- list(case = "default", input = "diff", gamma = g,
    pval = r$pval, deviate = r$deviate, statistic = r$statistic,
    expectation = r$expectation, variance = r$variance)
}
mw <- list()
for (g in mv_gammas) {
  r <- senmw(ymw, gamma = g)
  mw[[length(mw) + 1]] <- list(gamma = g, pval = r$pval,
    deviate = r$deviate, statistic = r$statistic,
    expectation = r$expectation, variance = r$variance)
}

# ---- E-values --------------------------------------------------------------
ev <- function(r) {
  e <- r["E-values", ]
  list(point = unname(na_null(e[1])), lower = unname(na_null(e[2])),
       upper = unname(na_null(e[3])))
}
ev_cases <- list()
add <- function(kind, args, res) {
  ev_cases[[length(ev_cases) + 1]] <<- c(list(kind = kind), args, ev(res))
}
quiet <- function(expr) suppressMessages(suppressWarnings(expr))
rr_grid <- list(
  list(est = 2, lo = 1.4, hi = 2.9, true = 1),
  list(est = 2, lo = 0.8, hi = 5, true = 1),
  list(est = 0.5, lo = 0.3, hi = 0.8, true = 1),
  list(est = 0.5, lo = 0.2, hi = 1.2, true = 1),
  list(est = 1.7, lo = NA, hi = NA, true = 1),
  list(est = 0.7, lo = NA, hi = NA, true = 1),
  list(est = 3, lo = 2, hi = 4.5, true = 1.5),
  list(est = 3, lo = 1.2, hi = 4.5, true = 1.5),
  list(est = 1.4, lo = 1.1, hi = 1.8, true = 2),
  list(est = 0.4, lo = 0.2, hi = 0.6, true = 0.7),
  list(est = 1, lo = 0.8, hi = 1.3, true = 1),
  list(est = 2.5, lo = 1.5, hi = NA, true = 1),
  list(est = 0.6, lo = NA, hi = 0.9, true = 1))
for (a in rr_grid) add("RR", a, quiet(evalues.RR(a$est, a$lo, a$hi, a$true)))
for (a in rr_grid) for (rare in c(TRUE, FALSE)) {
  b <- c(a, list(rare = rare))
  add("OR", b, quiet(evalues.OR(a$est, a$lo, a$hi, rare = rare, true = a$true)))
  add("HR", b, quiet(evalues.HR(a$est, a$lo, a$hi, rare = rare, true = a$true)))
}
md_grid <- list(
  list(est = 0.4, se = 0.1, true = 0), list(est = 0.4, se = 0.3, true = 0),
  list(est = -0.5, se = 0.2, true = 0), list(est = 0.3, se = NA, true = 0),
  list(est = 0.8, se = 0.2, true = 0.2), list(est = 0, se = 0.2, true = 0))
for (a in md_grid) add("MD", a, quiet(evalues.MD(a$est, a$se, a$true)))

out <- list(
  versions = list(r = R.version.string,
    rbounds = as.character(packageVersion("rbounds")),
    sensitivitymv = as.character(packageVersion("sensitivitymv")),
    sensitivitymw = as.character(packageVersion("sensitivitymw")),
    evalue = as.character(packageVersion("EValue")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  pairs = list(trt = trt, ctrl = ctrl, gammas = gammas,
               psens_lower = ps[[2]], psens_upper = ps[[3]],
               psens_exact_lower = pe[, "low"], psens_exact_upper = pe[, "up"],
               statistic = pe[1, "stat"], n_used = pe[1, "n"],
               pr = pr, hl_lower = hl[[2]], hl_upper = hl[[3]]),
  exact1 = exact1, exact2 = exact2,
  sets_mv = lapply(seq_len(nrow(ymv)), function(i) as.numeric(na.omit(ymv[i, ]))),
  sets_mw = lapply(seq_len(nrow(ymw)), function(i) as.numeric(ymw[i, ])),
  pair_diffs = ydiff,
  senmv = mv, senmw = mw, evalues = ev_cases)
write_json(out, "tests/data/sensitivity.json", digits = I(17),
           auto_unbox = TRUE, na = "null")
