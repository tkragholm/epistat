# Standardised risks and average treatment effects by the g-formula, as
# riskRegression::ate (estimator "GFORMULA") with Cox and cause-specific Cox
# models, and stdReg::stdCoxph for a Breslow Cox fit.
# For each case: the data (time, status, optional stratum), the design the
# model was fitted on (the treatment dummies first), the coefficients and
# covariance coxph returned, the times, and the ate output: the risk of every
# treatment level at every time with se and limits, and the difference and
# ratio of every level against the first with se, limits and p-values. The
# cause-specific cases are for both causes and both values of product.limit.
# A case with tied event times (Efron): its point estimates must agree with
# ours, its standard errors do not (riskRegression counts a tied time once in
# the influence function of the coefficients, see absrisk.R).
# A last block gives stdCoxph for the Breslow fit of the first case: the
# survival under each level and the covariance of the survivals at each time.
# Versions of R, riskRegression, stdReg, prodlim, survival and jsonlite are in
# the file. Numbers are written with 17 significant digits.
# Run: Rscript standardise.R > standardise.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(riskRegression)
  library(prodlim)
  library(stdReg)
})
set.seed(20261120)

num <- function(m) {
  m <- as.matrix(m)
  lapply(seq_len(nrow(m)), function(i) I(as.numeric(m[i, ])))
}

make_data <- function(n, levels3 = FALSE, tied = FALSE, competing = FALSE) {
  d <- data.frame(x1 = rnorm(n), x2 = rbinom(n, 1, 0.4))
  if (levels3) {
    u <- runif(n)
    d$A <- factor(ifelse(u < plogis(0.5 * d$x1) / 2, 0, ifelse(u < 0.75, 1, 2)))
  } else {
    d$A <- factor(rbinom(n, 1, plogis(0.5 * d$x1)))
  }
  a1 <- as.numeric(d$A == 1)
  a2 <- as.numeric(d$A == 2)
  t1 <- rexp(n, 0.1 * exp(0.5 * d$x1 - 0.4 * a1 + 0.2 * a2 + 0.3 * d$x2))
  t2 <- if (competing) rexp(n, 0.06 * exp(-0.4 * a1 + 0.2 * a2)) else Inf
  cc <- runif(n, 2, 16)
  t <- pmin(t1, t2, cc)
  d$time <- if (tied) ceiling(t * 4) / 4 else t
  d$st <- ifelse(cc < pmin(t1, t2), 0, ifelse(t1 < t2, 1, 2))
  d$s <- sample(1:3, n, replace = TRUE)
  d
}

ate_tables <- function(a, nlev) {
  m <- a$meanRisk
  d <- a$diffRisk
  r <- a$ratioRisk
  tab <- function(df, cols) lapply(seq_len(nrow(df)), function(i) {
    out <- as.list(df[i, cols, with = FALSE])
    lapply(out, function(v) if (is.character(v) || is.factor(v)) as.character(v) else v)
  })
  list(mean = tab(m, c("time", "treatment", "estimate", "se", "lower", "upper")),
       diff = tab(d, c("time", "A", "B", "estimate.A", "estimate.B", "estimate", "se", "lower", "upper", "p.value")),
       ratio = tab(r, c("time", "A", "B", "estimate", "se", "lower", "upper", "p.value")))
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

cox_case <- function(name, d, times, strata = FALSE, ties = "efron", nlev = 2) {
  rhs <- "A + x1 + x2"
  if (strata) rhs <- paste(rhs, "+ strata(s)")
  f <- coxph(as.formula(paste("Surv(time, st) ~", rhs)), data = d, x = TRUE, y = TRUE, ties = ties)
  a <- ate(f, data = d, treatment = "A", times = times, se = TRUE, verbose = FALSE, B = 0,
           contrasts = levels(d$A))
  list(name = name, kind = "cox", ties = ties, time = d$time, status = as.integer(d$st),
       stratum = if (strata) as.integer(d$s) else NULL,
       designs = list(num(f$x)), coefficients = list(I(as.numeric(coef(f)))), var = list(num(f$var)),
       times = times, nlevels = nlev, ate = ate_tables(a, nlev), scales = scales(list(f), d))
}

csc_case <- function(name, d, times, cause, pl) {
  fit <- CSC(list(Hist(time, st) ~ A + x1 + x2, Hist(time, st) ~ A + x1), data = d)
  a <- ate(fit, data = d, treatment = "A", times = times, cause = cause, se = TRUE,
           verbose = FALSE, B = 0, product.limit = pl, contrasts = levels(d$A))
  list(name = name, kind = "csc", cause = cause, product_limit = pl,
       time = d$time, status = as.integer(d$st),
       designs = unname(lapply(fit$models, function(m) num(m$x))),
       coefficients = unname(lapply(fit$models, function(m) I(as.numeric(coef(m))))),
       var = unname(lapply(fit$models, function(m) num(m$var))),
       times = times, nlevels = 2, ate = ate_tables(a, 2), scales = scales(fit$models, d))
}

cases <- list()
d1 <- make_data(300)
cases[[1]] <- cox_case("cox_binary", d1, c(1, 3, 6, 10, 14))
d2 <- make_data(400, levels3 = TRUE)
cases[[2]] <- cox_case("cox_three_levels", d2, c(2, 5, 9), nlev = 3)
d3 <- make_data(350)
cases[[3]] <- cox_case("cox_strata", d3, c(1.5, 4, 8), strata = TRUE)
d4 <- make_data(300, tied = TRUE)
cases[[4]] <- cox_case("cox_tied", d4, c(1, 3, 6, 10))
d5 <- make_data(300, competing = TRUE)
k <- 5
for (cause in 1:2) for (pl in c(TRUE, FALSE)) {
  cases[[k]] <- csc_case(paste0("csc_cause", cause, "_pl", pl), d5, c(1, 2.5, 4, 6), cause, pl)
  k <- k + 1
}

# stdReg::stdCoxph for a Breslow fit of d1
d1$Ab <- as.numeric(d1$A == 1)
fb <- coxph(Surv(time, st) ~ Ab + x1 + x2, data = d1, x = TRUE, y = TRUE, ties = "breslow")
tt <- c(1, 3, 6, 10)
s <- stdCoxph(fb, data = d1, X = "Ab", x = c(0, 1), t = tt)
stdreg <- list(times = tt, coefficients = I(as.numeric(coef(fb))), var = num(fb$var),
               survival = num(s$est), vcov = lapply(s$vcov, num))

vers <- list(r = R.version.string,
             riskRegression = as.character(packageVersion("riskRegression")),
             stdReg = as.character(packageVersion("stdReg")),
             prodlim = as.character(packageVersion("prodlim")),
             survival = as.character(packageVersion("survival")),
             jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = vers, cases = cases, stdreg = stdreg),
           digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
