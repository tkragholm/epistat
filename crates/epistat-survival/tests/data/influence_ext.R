# Standardised risks by the g-formula, riskRegression::ate, for data with entry
# times (counting-process data, left truncation) and for cause-specific models
# with surv.type = "survival". The cases are those of standardise.R with the
# entry times added, and the survival-type models, with and without entry
# times, for both values of product.limit. Times are continuous, so that no
# event time is tied (riskRegression's influence function counts a tied time
# once, see absrisk.R). riskRegression refuses case weights; the standard
# errors with case weights and with an offset are checked against numerical
# derivatives in tests/influence_extensions.rs.
# riskRegression's standard errors with entry times differ from the derivative
# of the estimate with respect to the case weight of each person (up to 25 per
# cent for the cases here, mostly at early times), which is what an influence
# function is. The Cox cases therefore carry that derivative, computed here by
# refitting coxph with weights (numeric_se), and the test asserts both the
# agreement with it and the difference from ate.
# Versions of R, riskRegression, prodlim, survival and jsonlite are in the file.
# Numbers are written with 17 significant digits.
# Run: Rscript influence_ext.R > influence_ext.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(riskRegression)
  library(prodlim)
})
set.seed(20261122)

num <- function(m) {
  m <- as.matrix(m)
  lapply(seq_len(nrow(m)), function(i) I(as.numeric(m[i, ])))
}

make_data <- function(n, entry = TRUE) {
  d <- data.frame(x1 = rnorm(n), x2 = rbinom(n, 1, 0.4))
  d$A <- factor(rbinom(n, 1, plogis(0.5 * d$x1)))
  a1 <- as.numeric(d$A == 1)
  t1 <- rexp(n, 0.1 * exp(0.5 * d$x1 - 0.4 * a1 + 0.3 * d$x2))
  t2 <- rexp(n, 0.06 * exp(-0.4 * a1))
  cc <- runif(n, 2, 16)
  d$time <- pmin(t1, t2, cc)
  d$st <- ifelse(cc < pmin(t1, t2), 0, ifelse(t1 < t2, 1, 2))
  d$start <- if (entry) ifelse(runif(n) < 0.5, 0.35 * d$time * runif(n), 0) else 0
  d$s <- sample(1:2, n, replace = TRUE)
  d
}

ate_tables <- function(a) {
  tab <- function(df, cols) lapply(seq_len(nrow(df)), function(i) {
    out <- as.list(df[i, cols, with = FALSE])
    lapply(out, function(v) if (is.character(v) || is.factor(v)) as.character(v) else v)
  })
  list(mean = tab(a$meanRisk, c("time", "treatment", "estimate", "se", "lower", "upper")),
       diff = tab(a$diffRisk, c("time", "A", "B", "estimate.A", "estimate.B", "estimate", "se", "lower", "upper", "p.value")),
       ratio = tab(a$ratioRisk, c("time", "A", "B", "estimate", "se", "lower", "upper", "p.value")))
}

# The largest hazard increment and the largest cumulative hazard at the last
# event time over the persons, summed over the models that form the risk.
scales <- function(models, d) {
  etimes <- sort(unique(d$time[d$st != 0]))
  tot_h <- 0
  tot_H <- 0
  for (m in models) {
    pc <- predictCox(m, newdata = d, times = etimes, type = c("hazard", "cumhazard"), centered = FALSE)
    h <- pc$hazard
    h[is.na(h)] <- 0
    tot_h <- tot_h + h
    tot_H <- tot_H + apply(pc$cumhazard, 1, function(v) max(v, na.rm = TRUE))
  }
  list(max_hazard = max(apply(tot_h, 1, max)), total_cumhazard = max(tot_H),
       n_event_times = length(etimes))
}

common <- function(name, kind, d, fit_models, a, times, extra) {
  c(list(name = name, kind = kind, time = d$time, start = d$start, status = as.integer(d$st),
         designs = unname(lapply(fit_models, function(m) num(m$x))),
         coefficients = unname(lapply(fit_models, function(m) I(as.numeric(coef(m))))),
         var = unname(lapply(fit_models, function(m) num(m$var))),
         times = times, nlevels = 2, ate = ate_tables(a), scales = scales(fit_models, d)),
    extra)
}

# The standardised risks of a Breslow Cox fit with case weights w, levels by
# times, from the fit and its baseline hazard (riskRegression refuses weights).
cox_estimate <- function(d, times, strata, w) {
  rhs <- "A + x1 + x2"
  if (strata) rhs <- paste(rhs, "+ strata(s)")
  f <- coxph(as.formula(paste("Surv(start, time, st == 1) ~", rhs)), data = d, weights = w,
             ties = "breslow", control = coxph.control(eps = 1e-13, iter.max = 100))
  bh <- basehaz(f, centered = FALSE)
  out <- matrix(NA_real_, 2, length(times))
  for (l in 0:1) {
    x <- cbind(A1 = l, x1 = d$x1, x2 = d$x2)
    theta <- exp(as.numeric(x %*% coef(f)))
    for (k in seq_along(times)) {
      cum <- numeric(nrow(d))
      for (s in if (strata) 1:2 else 1) {
        b <- if (strata) bh[bh$strata == paste0("s=", s), ] else bh
        h <- b$hazard[max(which(b$time <= times[k]))]
        cum[if (strata) d$s == s else TRUE] <- h
      }
      out[l + 1, k] <- sum(w * (1 - exp(-theta * cum))) / sum(w)
    }
  }
  out
}

# The influence function of that estimate for each person by the central
# difference in the log of the person's weight, with Richardson extrapolation
# from the steps h and h / 2, the standard error that follows, and its
# tolerance: the root of the sum of squares of the truncation error
# 2 |D(h) - D(h / 2)| and the rounding error 4 c eps F / (h / 2) of each
# person, with c = 4 (n + p + J).
numeric_se <- function(d, times, strata) {
  n <- nrow(d)
  h <- 1e-4
  est0 <- cox_estimate(d, times, strata, rep(1, n))
  J <- sum(d$st == 1)
  inf <- array(0, c(n, 2, length(times)))
  tol <- array(0, c(n, 2, length(times)))
  for (i in seq_len(n)) {
    dd <- list()
    for (k in 1:2) {
      step <- if (k == 1) h else h / 2
      w <- rep(1, n); w[i] <- 1 + step
      up <- cox_estimate(d, times, strata, w)
      w[i] <- 1 - step
      dn <- cox_estimate(d, times, strata, w)
      dd[[k]] <- (up - dn) / (2 * step)
    }
    inf[i, , ] <- (4 * dd[[2]] - dd[[1]]) / 3
    tol[i, , ] <- 2 * abs(dd[[1]] - dd[[2]]) + 4 * 4 * (n + 3 + J) * .Machine$double.eps * est0 / (h / 2)
  }
  list(se = num(apply(inf, c(2, 3), function(v) sqrt(sum(v^2)))),
       tol = num(apply(tol, c(2, 3), function(v) sqrt(sum(v^2)))))
}

cox_case <- function(name, d, times, strata = FALSE) {
  rhs <- "A + x1 + x2"
  if (strata) rhs <- paste(rhs, "+ strata(s)")
  f <- coxph(as.formula(paste("Surv(start, time, st == 1) ~", rhs)), data = d, x = TRUE, y = TRUE)
  a <- ate(f, data = d, treatment = "A", times = times, se = TRUE, verbose = FALSE, B = 0,
           contrasts = levels(d$A))
  common(name, "cox", d, list(f), a, times,
         list(stratum = if (strata) as.integer(d$s) else NULL, cause = NULL, product_limit = NULL,
              numeric = numeric_se(d, times, strata)))
}

csc_case <- function(name, d, times, cause, pl, surv_type = "hazard") {
  ent <- if (any(d$start > 0)) ", entry = start" else ""
  # ate re-evaluates the call of CSC, so the formulas live in the global scope.
  assign("f1", as.formula(paste0("Hist(time, st", ent, ") ~ A + x1 + x2")), envir = globalenv())
  assign("f2", as.formula(paste0("Hist(time, st", ent, ") ~ A + x1")), envir = globalenv())
  # `cause` of CSC puts the cause of interest first, which only the survival
  # type needs.
  fit <- if (surv_type == "hazard") CSC(list(f1, f2), data = d) else
    CSC(list(f1, f2), data = d, cause = cause, surv.type = surv_type)
  a <- ate(fit, data = d, treatment = "A", times = times, cause = cause, se = TRUE,
           verbose = FALSE, B = 0, product.limit = pl, contrasts = levels(d$A))
  common(name, if (surv_type == "hazard") "csc" else "csc_survival", d, fit$models, a, times,
         list(stratum = NULL, cause = cause, product_limit = pl))
}

cases <- list()
d1 <- make_data(300)
cases[[1]] <- cox_case("cox_entry", d1, c(1, 3, 6, 10, 14))
d2 <- make_data(350)
cases[[2]] <- cox_case("cox_entry_strata", d2, c(1.5, 4, 8), strata = TRUE)
d3 <- make_data(300)
k <- 3
for (cause in 1:2) for (pl in c(TRUE, FALSE)) {
  cases[[k]] <- csc_case(paste0("csc_entry_cause", cause, "_pl", pl), d3, c(1, 2.5, 4, 6), cause, pl)
  k <- k + 1
}
for (entry in c(FALSE, TRUE)) {
  d4 <- make_data(600, entry = entry)
  for (pl in c(TRUE, FALSE)) {
    cases[[k]] <- csc_case(paste0("csc_survival_", if (entry) "entry_" else "", "pl", pl), d4,
                           c(1, 2.5, 4, 6), 1, pl, surv_type = "survival")
    k <- k + 1
  }
}

vers <- list(r = R.version.string,
             riskRegression = as.character(packageVersion("riskRegression")),
             prodlim = as.character(packageVersion("prodlim")),
             survival = as.character(packageVersion("survival")),
             jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = vers, cases = cases),
           digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
