# Reference values for the sensitivity values, amplification, binary and
# censored outcomes in matched sets, and the interval of an m-statistic.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/rosenbaum_methods.R
#
# Sensitivity values: uniroot on the upper p-value bound minus 0.05, with a
# tolerance of 1e-12 in gamma, for
#   the signed rank test of 80 pairs, by the normal formula of psens written
#   out with the upper tail taken directly;
#   sensitivitymv::senmv on 60 sets of one to three controls;
#   the binomial tail of 45 discordant pairs;
#   sensitivity2x2xk::mh on 200 sets of one treated and three controls with a
#   binary outcome.
# Amplification: sensitivitymv::amplify.
# Binary outcomes: sensitivity2x2xk::mh (exact, by convolution with R's
#   convolve, which is an FFT) and mhLS (normal) on the 1:3 sets and on 40
#   strata of two to four treated and two to six controls, and pbinom for
#   pairs, at gamma 1 to 3. The lower bound is mh with 1 / gamma, which is
#   the bound at odds ratio 1 / gamma.
# Censored times: 80 sets of one treated and one to three controls with
#   exponential times rounded to one decimal, censored at a time shared by
#   the set or earlier at random. Gehan's scores are computed below from
#   their definition and passed to sensitivitymv::separable1v.
# Interval: sensitivitymw::senmwCI on 50 sets of one treated and three
#   controls, Huber's score with trim 2.5, with detail = TRUE and tol = 1e-9.
library(sensitivitymv)
library(sensitivitymw)
library(sensitivity2x2xk)
library(jsonlite)

set.seed(20261126)
alpha <- 0.05
root <- function(f) uniroot(f, c(1, 20), tol = 1e-12)$root

# ---- Signed rank test of pairs -------------------------------------------
n_pairs <- 80
diffs <- round(rnorm(n_pairs, 0.45, 1), 2)
signed_upper <- function(d, gamma) {
  d <- d[d != 0]
  r <- rank(abs(d))
  t <- sum(r[d > 0])
  p <- gamma / (1 + gamma)
  pnorm((t - p * sum(r)) / sqrt(p * (1 - p) * sum(r^2)), lower.tail = FALSE)
}
signed_value <- root(function(g) signed_upper(diffs, g) - alpha)

# ---- senmv on sets of different sizes ------------------------------------
n_sets <- 60
sizes <- sample(1:3, n_sets, replace = TRUE)
ymv <- matrix(NA_real_, n_sets, 4)
for (i in 1:n_sets) {
  shared <- rnorm(1)
  ymv[i, 1] <- shared + 0.6 + rnorm(1, 0, 0.8)
  ymv[i, 2:(1 + sizes[i])] <- shared + rnorm(sizes[i], 0, 0.8)
}
mv_value <- root(function(g) senmv(ymv, gamma = g, trim = 2.5)$pval - alpha)

# ---- Binary pairs ----------------------------------------------------------
binom_upper <- function(g) pbinom(29, 45, g / (1 + g), lower.tail = FALSE)
binom_value <- root(function(g) binom_upper(g) - alpha)
gammas <- c(1, 1.25, 1.5, 2, 3)
binom_tail <- list(upper = sapply(gammas, binom_upper),
                   lower = sapply(gammas, function(g) pbinom(29, 45, 1 / (1 + g), lower.tail = FALSE)))

# ---- Binary 1:3 sets -------------------------------------------------------
n_bin <- 200
bt <- rbinom(n_bin, 1, 0.16)
bc <- matrix(rbinom(3 * n_bin, 1, 0.08), n_bin, 3)
tab13 <- array(0, c(2, 2, n_bin))
for (i in 1:n_bin) {
  cc <- sum(bc[i, ])
  tab13[, , i] <- matrix(c(bt[i], cc, 1 - bt[i], 3 - cc), 2, 2)
}
bounds_of <- function(tab) {
  lapply(gammas, function(g) {
    up <- mh(tab, Gamma = g)
    lo <- mh(tab, Gamma = 1 / g)
    ls <- mhLS(tab, Gamma = g)
    list(gamma = g, statistic = up$A, p_upper = up$pval, p_lower = lo$pval,
         ls_p_upper = ls$pval, ls_expectation = ls$Expectation,
         ls_variance = ls$Variance)
  })
}
b13 <- bounds_of(tab13)
mh_value <- root(function(g) mh(tab13, Gamma = g)$pval - alpha)

# ---- Binary strata with several treated ------------------------------------
n_str <- 40
strata <- list()
tabk <- array(0, c(2, 2, n_str))
for (i in 1:n_str) {
  nt <- sample(2:4, 1); nc <- sample(2:6, 1)
  tc <- rbinom(1, nt, 0.35); ccs <- rbinom(1, nc, 0.2)
  tabk[, , i] <- matrix(c(tc, ccs, nt - tc, nc - ccs), 2, 2)
  strata[[i]] <- c(tc, nt - tc, ccs, nc - ccs)
}
bk <- bounds_of(tabk)

# ---- Censored times --------------------------------------------------------
n_cens <- 80
csizes <- sample(1:3, n_cens, replace = TRUE)
precedes <- function(tj, ej, tk, ek) ej && (tj < tk || (tj == tk && !ek))
gehan <- function(time, event) {
  m <- length(time)
  sapply(1:m, function(j) {
    s <- 0
    for (k in (1:m)[-j]) {
      s <- s + precedes(time[j], event[j], time[k], event[k]) -
        precedes(time[k], event[k], time[j], event[j])
    }
    s
  })
}
cens_sets <- list()
scores <- matrix(NA_real_, n_cens, 4)
for (i in 1:n_cens) {
  m <- 1 + csizes[i]
  rate <- c(1.6, rep(1, csizes[i]))
  event_time <- round(rexp(m, rate), 1)
  admin <- round(runif(1, 0.8, 2), 1)
  loss <- ifelse(runif(m) < 0.15, round(runif(m, 0, admin), 1), Inf)
  censor <- pmin(admin, loss)
  time <- pmin(event_time, censor)
  event <- event_time <= censor
  cens_sets[[i]] <- list(time = time, event = event)
  scores[i, 1:m] <- gehan(time, event)
}
cens <- lapply(gammas, function(g) {
  s <- separable1v(scores, gamma = g)
  list(gamma = g, pval = s$pval, deviate = s$deviate, statistic = s$statistic,
       expectation = s$expectation, variance = s$variance)
})
cens_value <- root(function(g) separable1v(scores, gamma = g)$pval - alpha)

# ---- Amplification ---------------------------------------------------------
amp <- list(
  list(gamma = 2.08, lambda = c(2.5, 3, 5, 10), delta = as.numeric(amplify(2.08, c(2.5, 3, 5, 10)))),
  list(gamma = 1.5, lambda = c(1.6, 2, 4, 50), delta = as.numeric(amplify(1.5, c(1.6, 2, 4, 50)))))

# ---- senmwCI -----------------------------------------------------------------
n_mw <- 50
ymw <- matrix(NA_real_, n_mw, 4)
for (i in 1:n_mw) {
  shared <- rnorm(1)
  ymw[i, 1] <- shared + 0.5 + rnorm(1, 0, 0.7)
  ymw[i, 2:4] <- shared + rnorm(3, 0, 0.7)
}
ci <- list()
for (g in c(1, 1.5, 2)) for (one in c(TRUE, FALSE)) {
  x <- senmwCI(ymw, gamma = g, trim = 2.5, alpha = alpha, one.sided = one,
               tol = 1e-9, detail = TRUE)
  ci[[length(ci) + 1]] <- list(gamma = g, one_sided = one,
    estimate = as.numeric(x$PointEstimate), interval = as.numeric(x$Confidence.Interval),
    tol = x$tolerance)
}

rows <- function(m) lapply(seq_len(nrow(m)), function(i) as.numeric(na.omit(m[i, ])))
out <- list(
  versions = list(r = R.version.string,
    sensitivitymv = as.character(packageVersion("sensitivitymv")),
    sensitivitymw = as.character(packageVersion("sensitivitymw")),
    sensitivity2x2xk = as.character(packageVersion("sensitivity2x2xk")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  alpha = alpha, gammas = gammas,
  pair_diffs = diffs, signed_value = signed_value,
  sets_mv = rows(ymv), mv_value = mv_value,
  binom_value = binom_value, binom_tail = binom_tail,
  binary_treated = bt, binary_controls = lapply(1:n_bin, function(i) bc[i, ]),
  binary_13 = b13, mh_value = mh_value,
  strata = strata, binary_strata = bk,
  censored = lapply(cens_sets, function(s) list(time = s$time, event = s$event)),
  censored_bounds = cens, censored_value = cens_value,
  amplify = amp,
  sets_mw = rows(ymw), senmwci = ci)
write_json(out, "tests/data/rosenbaum_methods.json", digits = I(17),
           auto_unbox = TRUE, na = "null")
