# The attributable fraction function of a Cox model, epistat-survival's
# standardise::attributable, against AFcoxph of the R package AF 0.1.5, run from
# its source (AF was archived from CRAN on 2026-05-08 and depends on packages
# that are not on CRAN for this R; AFcoxph needs only survival and data.table).
# Set AF_SRC to the directory AF/R of the unpacked tarball
# https://cran.r-project.org/src/contrib/Archive/AF/AF_0.1.5.tar.gz.
# Cases: continuous times, and times rounded to tenths, which tie.
#
# AFcoxph's variance is not used as the reference for the variance: its score
# for the Breslow baseline, H0res * (T <= t), has only the event part of the
# influence function of the cumulative hazard and leaves out the compensator,
# the term of everybody at risk, so its variance is too small and increasingly
# so with time. The reference for the variance is the delete-one jackknife of
# the standardised risks and of AF(t), refitting the model without each person
# and averaging over the persons that remain, which has no formula in common
# with the influence function used in Rust. The AFcoxph variances are recorded
# for the report.
# Run: AF_SRC=/path/to/AF/R Rscript af_cox.R > af_cox.json
suppressMessages({
  library(jsonlite); library(survival); library(data.table)
})
src <- Sys.getenv("AF_SRC")
for (f in c("AFfunctions.R", "AFcoxph.R")) source(file.path(src, f))
set.seed(20261126)
expit <- function(x) 1 / (1 + exp(-x))
rows <- function(M) lapply(seq_len(nrow(M)), function(i) I(unname(M[i, ])))

cox_case <- function(label, ties) {
  m <- 800
  Z <- rnorm(m); X <- rbinom(m, 1, expit(0.5 * Z - 0.2))
  Tt <- rexp(m, exp(-1.2 + 0.7 * X + 0.5 * Z + 0.3 * X * Z)); C <- runif(m, 0.5, 4)
  time <- pmin(Tt, C); status <- as.numeric(Tt <= C)
  if (ties) time <- round(time, 1) + 0.1 * (time < 0.05)
  dat <- data.frame(time, status, X, Z)
  fit <- coxph(Surv(time, status) ~ X + Z + X:Z, data = dat, ties = "breslow",
               control = coxph.control(eps = 1e-12, toler.chol = 1e-14, iter.max = 100))
  ev <- sort(unique(time[status == 1]))
  times <- unname(quantile(ev, c(.05, .15, .3, .5, .7, .85, .95), type = 1))
  a <- AFcoxph(fit, data = dat, exposure = "X", times = times)
  dat0 <- transform(dat, X = 0)
  at <- function(idx) {
    f <- coxph(Surv(time, status) ~ X + Z + X:Z, data = dat[idx, ], ties = "breslow")
    bh <- basehaz(f, centered = FALSE)
    H <- sapply(times, function(t) { i <- which(bh$time <= t); if (length(i) == 0) 0 else bh$hazard[max(i)] })
    lp <- drop(model.matrix(~ X + Z + X:Z, dat[idx, ])[, -1, drop = FALSE] %*% coef(f))
    lp0 <- drop(model.matrix(~ X + Z + X:Z, dat0[idx, ])[, -1, drop = FALSE] %*% coef(f))
    S <- sapply(H, function(h) mean(exp(-h * exp(lp))))
    S0 <- sapply(H, function(h) mean(exp(-h * exp(lp0))))
    rbind(S = S, S0 = S0, AF = 1 - (1 - S0) / (1 - S))
  }
  jk <- lapply(seq_len(m), function(i) at(setdiff(seq_len(m), i)))
  jkv <- function(row) {
    M <- t(sapply(jk, function(r) r[row, ]))
    (m - 1) / m * colSums((M - matrix(colMeans(M), m, length(times), byrow = TRUE))^2)
  }
  d1 <- model.matrix(~ X + Z + X:Z, dat)[, -1]
  d0 <- model.matrix(~ X + Z + X:Z, transform(dat, X = 0))[, -1]
  list(label = label, time = time, status = status, design = rows(d1), design0 = rows(d0),
       times = times, coef = I(unname(coef(fit))), vcov = rows(vcov(fit)),
       af = I(a$AF.est), af_var = I(drop(a$AF.var)), s = I(a$S.est), s0 = I(a$S0.est),
       s_var = I(a$S.var), s0_var = I(a$S0.var),
       jk_s_var = I(jkv("S")), jk_s0_var = I(jkv("S0")), jk_af_var = I(jkv("AF")))
}
cox <- list(cox_case("continuous times", FALSE), cox_case("tied times", TRUE))


out <- list(versions = list(R = R.version.string, AF = "0.1.5 (source)",
  survival = as.character(packageVersion("survival"))), cases = cox)
cat(toJSON(out, auto_unbox = TRUE, digits = NA, na = "string", null = "null"))
