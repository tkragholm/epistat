# Reference values for the pooling extensions of epistat_inference::pooling:
# the pooled vector, D1, D2 and D3 tests and pooling on a transformed scale.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/pooling_ext.R
#
# mice 3.19.0 and mitml 0.4.5 on mice::nhanes2 (logistic and linear models,
# m = 5 and m = 50) and on survival::lung (Cox models, m = 5), imputed with a
# fixed seed. The Rust tests read the per-data-set estimates, covariances and
# fit statistics that this script writes, so nothing is imputed there.
#   - Pooled vector: Qbar, Ubar, B and T computed from the stored fits, and the
#     average relative increase of D1; the scalar rules per coefficient come
#     from mice::pool.
#   - D1: mitml::testModels(method = "D1") without and with the complete-data
#     degrees of freedom, as mice::D1 calls it.
#   - D2: mitml::testModels(method = "D2", use = "wald") from the Wald statistics
#     of the stored fits, and use = "likelihood" from their log-likelihoods.
#   - D3: mice::D3, whose deviances at the own and at the pooled estimates are
#     stored with it.
#   - Transformed pooling: the delta-method variance written out here, pooled
#     with mice::pool.scalar, the transformation undone by hand.
# dfcom for a Cox model is the number of events less the number of
# coefficients, as mice::get.dfcom. Values are written with 17 significant
# digits.
suppressMessages({
  library(jsonlite)
  library(mice)
  library(mitml)
  library(survival)
})
set.seed(20261122)
num <- function(x) as.numeric(unname(x))

per_set <- function(fits) lapply(fits, function(f) list(coef = I(num(coef(f))), vcov = I(as.vector(vcov(f)))))

tested <- function(fit1, fit0) {
  n1 <- names(coef(fit1)); n0 <- names(coef(fit0))
  which(!(n1 %in% n0))
}

case <- function(name, imp, f1, f0, fit_fn, dfcom = NULL, d3 = TRUE) {
  fits1 <- lapply(seq_len(imp$m), function(i) fit_fn(f1, complete(imp, i)))
  fits0 <- lapply(seq_len(imp$m), function(i) fit_fn(f0, complete(imp, i)))
  idx <- tested(fits1[[1]], fits0[[1]])
  k <- length(idx)
  m <- imp$m
  mira1 <- as.mira(fits1); mira0 <- as.mira(fits0)
  est <- sapply(fits1, coef)                        # p x m
  est <- matrix(est, ncol = m)
  pooled <- summary(pool(mira1), "all", conf.int = FALSE)
  dfc <- if (is.null(dfcom)) mice:::get.dfcom(fits1[[1]], NULL) else dfcom
  tm <- function(method, ...) {
    x <- testModels(fits1, fits0, method = method, ...)$test
    list(f = num(x[1, "F.value"]), df1 = num(x[1, "df1"]), df2 = num(x[1, "df2"]), p = num(x[1, "P(>F)"]), riv = num(x[1, "RIV"]))
  }
  out <- list(name = name, m = m, k = k, tested = I(idx), dfcom = dfc,
              fits1 = per_set(fits1),
              pooled = list(term = I(names(coef(fits1[[1]]))), estimate = I(num(pooled$estimate)),
                            ubar = I(num(pooled$ubar)), b = I(num(pooled$b)), t = I(num(pooled$t)),
                            df = I(num(pooled$df)), riv = I(num(pooled$riv)), p = I(num(pooled$p.value)),
                            dfcom = I(num(pooled$dfcom))),
              d1 = tm("D1"), d1_dfcom = tm("D1", df.com = dfc),
              d2_wald = tm("D2", use = "wald"))
  if (inherits(fits1[[1]], "glm") || inherits(fits1[[1]], "lm")) {
    out$d2_lik <- tm("D2", use = "likelihood")
    out$lik_stats <- I(num(sapply(seq_len(m), function(i) -2 * (as.numeric(logLik(fits0[[i]])) - as.numeric(logLik(fits1[[i]]))))))
  }
  out$wald_stats <- I(num(sapply(seq_len(m), function(i) {
    q <- coef(fits1[[i]])[idx]; u <- vcov(fits1[[i]])[idx, idx, drop = FALSE]
    drop(t(q) %*% solve(u) %*% q)
  })))
  if (d3) {
    r <- tryCatch(mice::D3(mira1, mira0), error = function(e) NULL)
    if (!is.null(r)) {
      x <- r$result
      out$d3 <- list(f = num(x[1]), df1 = num(x[2]), df2 = num(x[3]), p = num(x[4]), riv = num(x[5]),
                     lr_own = I(num(r$deviances$dev0.M - r$deviances$dev1.M)),
                     lr_pooled = I(num(r$deviances$dev0.L - r$deviances$dev1.L)))
    }
  }
  out
}

glm_fit <- function(f, d) glm(f, data = d, family = binomial)
lm_fit <- function(f, d) lm(f, data = d)
cox_fit <- function(f, d) coxph(f, data = d)

cases <- list()
imp5 <- mice(nhanes2, m = 5, printFlag = FALSE, seed = 11)
imp50 <- mice(nhanes2, m = 50, printFlag = FALSE, seed = 12)
for (spec in list(list(5, imp5), list(50, imp50))) {
  m <- spec[[1]]; imp <- spec[[2]]
  cases[[length(cases) + 1]] <- case(sprintf("logistic, m = %d, k = 4", m), imp,
    hyp ~ bmi + chl + age, hyp ~ 1, glm_fit)
  cases[[length(cases) + 1]] <- case(sprintf("logistic, m = %d, k = 2", m), imp,
    hyp ~ bmi + chl + age, hyp ~ bmi + chl, glm_fit)
  cases[[length(cases) + 1]] <- case(sprintf("logistic, m = %d, k = 1", m), imp,
    hyp ~ bmi + chl + age, hyp ~ chl + age, glm_fit)
  cases[[length(cases) + 1]] <- case(sprintf("linear, m = %d, k = 3", m), imp,
    chl ~ bmi + age + hyp, chl ~ bmi, lm_fit)
}
boysd <- boys[, c("age", "hgt", "wgt", "hc", "reg")]
impboys <- mice(boysd, m = 5, maxit = 2, printFlag = FALSE, seed = 14)
impboys20 <- mice(boysd, m = 20, maxit = 2, printFlag = FALSE, seed = 15)
for (spec in list(list(5, impboys), list(20, impboys20))) {
  m <- spec[[1]]; imp <- spec[[2]]
  cases[[length(cases) + 1]] <- case(sprintf("boys linear, m = %d, drop reg", m), imp,
    hgt ~ age + wgt + hc + reg, hgt ~ age + wgt + hc, lm_fit)
  cases[[length(cases) + 1]] <- case(sprintf("boys linear, m = %d, drop wgt and hc", m), imp,
    hgt ~ age + wgt + hc + reg, hgt ~ age + reg, lm_fit)
  cases[[length(cases) + 1]] <- case(sprintf("boys logistic, m = %d, drop reg", m), imp,
    I(hc > 50) ~ age + wgt + hgt + reg, I(hc > 50) ~ age + wgt + hgt, glm_fit)
  cases[[length(cases) + 1]] <- case(sprintf("boys logistic, m = %d, drop hgt", m), imp,
    I(hc > 50) ~ age + wgt + hgt + reg, I(hc > 50) ~ age + wgt + reg, glm_fit)
  cases[[length(cases) + 1]] <- case(sprintf("boys logistic, m = %d, drop wgt and hgt", m), imp,
    I(hc > 50) ~ age + wgt + hgt + reg, I(hc > 50) ~ age + reg, glm_fit)
}
lungd <- lung[, c("time", "status", "age", "sex", "ph.ecog", "ph.karno", "wt.loss")]
implung <- mice(lungd, m = 5, printFlag = FALSE, seed = 13)
nev <- sum(lung$status == 2)
cases[[length(cases) + 1]] <- case("Cox, m = 5, two coefficients", implung,
  Surv(time, status) ~ age + sex + ph.ecog + ph.karno, Surv(time, status) ~ age + sex,
  cox_fit, dfcom = nev - 4, d3 = FALSE)
cases[[length(cases) + 1]] <- case("Cox, m = 5, one coefficient", implung,
  Surv(time, status) ~ age + sex + ph.ecog, Surv(time, status) ~ age + sex,
  cox_fit, dfcom = nev - 3, d3 = FALSE)

# Transformed pooling: original-scale estimates and standard errors.
tr_case <- function(name, q, se, kind, dfcom = NULL) {
  g <- switch(kind, log = log, logit = qlogis, fisherz = atanh, cloglog = function(x) log(-log(1 - x)), sqrt = sqrt)
  ginv <- switch(kind, log = exp, logit = plogis, fisherz = tanh, cloglog = function(x) 1 - exp(-exp(x)), sqrt = function(x) x^2)
  dg <- switch(kind, log = 1 / q, logit = 1 / (q * (1 - q)), fisherz = 1 / (1 - q^2),
               cloglog = 1 / ((1 - q) * (-log(1 - q))), sqrt = 1 / (2 * sqrt(q)))
  qt <- g(q); ut <- se^2 * dg^2
  p <- pool.scalar(qt, ut, n = if (is.null(dfcom)) Inf else dfcom + 1)
  # pool.scalar takes the sample size n and uses n - 1 as the complete-data df
  list(name = name, kind = kind, q = I(q), se = I(se), dfcom = dfcom,
       qbar = p$qbar, ubar = p$ubar, b = p$b, t = p$t, df = p$df, r = p$r,
       estimate = ginv(p$qbar),
       lower = ginv(p$qbar - qt(0.975, p$df) * sqrt(p$t)), upper = ginv(p$qbar + qt(0.975, p$df) * sqrt(p$t)),
       pvalue_null = 2 * pt(abs((p$qbar - g(if (kind == "fisherz") 0.2 else if (kind %in% c("log", "sqrt")) 1 else 0.4)) / sqrt(p$t)), p$df, lower.tail = FALSE))
}
tcases <- list(
  tr_case("hazard ratios", c(1.31, 1.52, 1.18, 1.44, 1.27, 1.39), c(0.12, 0.14, 0.11, 0.13, 0.12, 0.15), "log", 90),
  tr_case("proportions", c(0.31, 0.27, 0.36, 0.29, 0.33), c(0.04, 0.045, 0.04, 0.05, 0.04), "logit", 150),
  tr_case("correlations", c(0.41, 0.35, 0.47, 0.38, 0.44, 0.40, 0.36), c(0.09, 0.1, 0.08, 0.09, 0.09, 0.1, 0.11), "fisherz", 40),
  tr_case("cumulative incidences", c(0.12, 0.15, 0.11, 0.14), c(0.02, 0.025, 0.02, 0.022), "cloglog", NULL),
  tr_case("Brier scores", c(0.081, 0.093, 0.087, 0.09, 0.085), c(0.006, 0.007, 0.006, 0.0065, 0.006), "sqrt", 200)
)

# Upper tails of F and two-sided tails of t, for the p-values of the tests and of
# Pooled::p_value, including very large degrees of freedom.
pf_grid <- list()
for (k in c(1, 2, 3, 5, 10)) for (nu in c(4, 5, 12.5, 30, 100, 1e3, 1e4, 1e5, 1e6, 1e8, 1e10, 1e12)) for (f in c(0.01, 0.5, 1, 2, 5, 20, 100))
  pf_grid[[length(pf_grid) + 1]] <- list(k = k, nu = nu, f = f, p = pf(f, k, nu, lower.tail = FALSE))
pt_grid <- list()
for (nu in c(3, 7.5, 30, 1e3, 1e5, 1e7, 1e9, 1e12)) for (z in c(0.1, 1, 2, 3.5, 8))
  pt_grid[[length(pt_grid) + 1]] <- list(nu = nu, z = z, p = 2 * pt(z, nu, lower.tail = FALSE))

write(toJSON(list(versions = list(r = R.version.string, mice = as.character(packageVersion("mice")),
                                  mitml = as.character(packageVersion("mitml"))),
                  cases = cases, transformed = tcases, pf = pf_grid, pt = pt_grid),
             auto_unbox = TRUE, digits = 17, null = "null", na = "null"),
      "tests/data/pooling_ext.json")
