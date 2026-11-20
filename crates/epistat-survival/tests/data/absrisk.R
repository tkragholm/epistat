# Absolute risk (cumulative incidence) from cause-specific Cox models, as
# riskRegression::CSC and predict.CauseSpecificCox (type "absRisk").
# For each case: the data (time, status 0..K, optional stratum), the design of
# each cause-specific model, the coefficients and variance coxph returned for
# it, new rows with their design per model, prediction times, and
#  - absRisk, absRisk.se (se = TRUE) and the loglog limits (confint default)
#    for the cause asked, plus the limits for the other transforms;
#  - the same with product.limit = FALSE.
# Cases: the Melanoma data of riskRegression (one event time shared by two
# causes), simulated data with different covariates per cause, a stratum and
# no tied times, and simulated data with tied times, three causes, Breslow
# and Efron ties. Both surv.type values.
# A last block compares riskRegression's influence function of the
# coefficients (iidCox) with dfbeta of coxph, with and without tied events.
# Versions of R, riskRegression, prodlim, survival and jsonlite are in the
# file. Numbers are written with 17 significant digits.
# Run: Rscript absrisk.R > absrisk.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(riskRegression)
  library(prodlim)
})
set.seed(20261121)

num <- function(m) {
  m <- as.matrix(m)
  lapply(seq_len(nrow(m)), function(i) I(as.numeric(m[i, ])))
}
nn <- function(a) {
  # a matrix with NA to a list of rows, NA as null
  lapply(seq_len(nrow(a)), function(i) unname(lapply(unname(a[i, ]), function(v) if (is.na(v)) NULL else v)))
}

run_case <- function(name, dat, vars, causes_vars, surv_type, ties, cause,
                     newdata, times, strata_var = NULL) {
  # dat has time, status; vars: names of numeric covariates; causes_vars: list
  # of character vectors, the covariates of each model (cause order of the
  # data for hazard; cause model then overall model for survival)
  fl <- lapply(causes_vars, function(v) {
    rhs <- paste(c(v, if (!is.null(strata_var)) paste0("strata(", strata_var, ")")), collapse = " + ")
    as.formula(paste0("Hist(time, status) ~ ", rhs))
  })
  fit <- CSC(fl, data = dat, cause = cause, surv.type = surv_type, ties = ties)
  designs <- lapply(seq_along(fit$models), function(k) fit$models[[k]]$x)
  # the models are listed with the cause of interest first
  new_designs <- lapply(seq_along(fit$models), function(k) {
    as.matrix(newdata[, causes_vars[[k]], drop = FALSE])
  })
  out <- list(name = name, surv_type = surv_type, ties = ties, cause = cause,
              causes_order = as.integer(as.character(fit$causes)),
              time = dat$time, status = dat$status,
              stratum = if (is.null(strata_var)) NULL else as.integer(dat[[strata_var]]),
              designs = lapply(designs, num),
              coefficients = unname(lapply(fit$models, function(m) I(as.numeric(coef(m))))),
              var = unname(lapply(fit$models, function(m) num(m$var))),
              converged = unname(sapply(fit$models, function(m) m$iter < 20)),
              new_designs = lapply(new_designs, num),
              new_stratum = if (is.null(strata_var)) NULL else as.integer(newdata[[strata_var]]),
              times = times, results = list())
  # the hazards the survival is formed from: the largest total hazard
  # increment and the total cumulative hazard at the last event time, per row
  etimes <- sort(unique(dat$time[dat$status != 0]))
  used <- if (surv_type == "hazard") seq_along(fit$models) else 2
  tot_h <- 0
  tot_H <- 0
  for (k in used) {
    pc <- predictCox(fit$models[[k]], newdata = newdata, times = etimes,
                     type = c("hazard", "cumhazard"), centered = FALSE)
    # beyond the last time of a stratum the values are NA
    h <- pc$hazard
    h[is.na(h)] <- 0
    H <- apply(pc$cumhazard, 1, function(v) max(v, na.rm = TRUE))
    tot_h <- tot_h + h
    tot_H <- tot_H + H
  }
  out$max_hazard <- apply(tot_h, 1, max)
  out$total_cumhazard <- as.numeric(tot_H)
  for (pl in c(TRUE, FALSE)) {
    p <- predict(fit, newdata = newdata, times = times, cause = cause,
                 se = TRUE, product.limit = pl)
    res <- list(product_limit = pl,
                absrisk = nn(p$absRisk), se = nn(p$absRisk.se),
                lower = nn(p$absRisk.lower), upper = nn(p$absRisk.upper))
    for (tr in c("none", "log", "cloglog")) {
      q <- confint(p, absRisk.transform = tr)
      res[[paste0("lower_", tr)]] <- nn(q$absRisk.lower)
      res[[paste0("upper_", tr)]] <- nn(q$absRisk.upper)
    }
    p0 <- predict(fit, newdata = newdata, times = times, cause = cause, product.limit = pl)
    res$absrisk_nose <- nn(p0$absRisk)
    out$results[[length(out$results) + 1]] <- res
  }
  out
}

cases <- list()

# ---- Melanoma
data(Melanoma)
mel <- data.frame(
  time = Melanoma$time, status = as.integer(Melanoma$status),
  sex = as.numeric(Melanoma$sex == "Male"), age = Melanoma$age,
  ulcer = as.numeric(Melanoma$ulcer == "present"), logthick = Melanoma$logthick,
  epicel = as.numeric(Melanoma$epicel == "present"),
  inv1 = as.numeric(Melanoma$invasion == "level.1"),
  inv2 = as.numeric(Melanoma$invasion == "level.2")
)
mel_new <- data.frame(
  sex = c(0, 1, 1, 0, 1), age = c(40, 55, 70, 62, 30), ulcer = c(0, 1, 1, 0, 1),
  logthick = c(0.2, 1.1, 1.9, 0.7, 0.5), epicel = c(0, 0, 1, 1, 0),
  inv1 = c(0, 1, 0, 1, 0), inv2 = c(1, 0, 0, 0, 0)
)
mel_times <- c(3000, 365, 1825, 800, 4500, 5, 5565, 6000)
cases[[1]] <- run_case(
  "melanoma_hazard", mel, NULL,
  list(c("sex", "age", "ulcer", "logthick"), c("age", "sex", "inv1", "inv2")),
  "hazard", "efron", 1, mel_new, mel_times)
cases[[2]] <- run_case(
  "melanoma_hazard_cause2", mel, NULL,
  list(c("sex", "age", "ulcer", "logthick"), c("age", "sex", "inv1", "inv2")),
  "hazard", "efron", 2, mel_new, mel_times)
mel_any <- mel
cases[[3]] <- run_case(
  "melanoma_survival", mel, NULL,
  list(c("sex", "age", "ulcer", "logthick"), c("age", "sex", "ulcer", "inv1", "inv2")),
  "survival", "efron", 1, mel_new, mel_times)
cases[[4]] <- run_case(
  "melanoma_breslow", mel, NULL,
  list(c("sex", "age", "ulcer", "logthick"), c("age", "sex", "inv1", "inv2")),
  "hazard", "breslow", 1, mel_new, mel_times)

# ---- simulated, a stratum, no tied times, different covariates per cause
n <- 220
sim <- data.frame(
  x1 = rnorm(n), x2 = runif(n), x3 = rbinom(n, 1, 0.4), s = sample(1:3, n, replace = TRUE)
)
h1 <- 0.10 * exp(0.5 * sim$x1 - 0.6 * sim$x3)
h2 <- 0.06 * exp(-0.4 * sim$x2 + 0.3 * sim$x1)
t1 <- rexp(n, h1)
t2 <- rexp(n, h2)
cens <- runif(n, 2, 18)
sim$time <- pmin(t1, t2, cens)
sim$status <- ifelse(sim$time == cens, 0L, ifelse(t1 < t2, 1L, 2L))
sim_new <- data.frame(x1 = c(-0.5, 0, 1.2, 0.3), x2 = c(0.1, 0.5, 0.9, 0.3),
                      x3 = c(0, 1, 1, 0), s = c(1, 2, 3, 2))
sim_times <- c(0.2, 1, 2.5, 5, 8, 12, 17, 25)
cases[[5]] <- run_case(
  "sim_strata_hazard", sim, NULL,
  list(c("x1", "x3"), c("x1", "x2")),
  "hazard", "efron", 1, sim_new, sim_times, strata_var = "s")
cases[[6]] <- run_case(
  "sim_strata_survival", sim, NULL,
  list(c("x1", "x3"), c("x1", "x2", "x3")),
  "survival", "efron", 1, sim_new, sim_times, strata_var = "s")
cases[[7]] <- run_case(
  "sim_nostrata_cause2", sim, NULL,
  list(c("x1", "x2"), c("x1", "x3")),
  "hazard", "efron", 2, sim_new, sim_times)

# ---- simulated, tied times, three causes
n <- 400
tie <- data.frame(x1 = round(rnorm(n), 2), x2 = rbinom(n, 1, 0.5))
r1 <- 0.07 * exp(0.4 * tie$x1)
r2 <- 0.05 * exp(-0.5 * tie$x2)
r3 <- 0.03 * exp(0.3 * tie$x1 + 0.4 * tie$x2)
tt <- cbind(rexp(n, r1), rexp(n, r2), rexp(n, r3))
cz <- pmin(runif(n, 3, 25), 12)
tm <- pmin(apply(tt, 1, min), cz)
tie$time <- pmax(ceiling(tm), 1)
tie$status <- ifelse(tm == cz, 0L, apply(tt, 1, which.min))
tie_new <- data.frame(x1 = c(-1, 0, 0.8), x2 = c(0, 1, 1))
tie_times <- c(1, 3, 6, 10, 12, 15)
for (ty in c("efron", "breslow")) {
  cases[[length(cases) + 1]] <- run_case(
    paste0("ties_", ty), tie, NULL,
    list(c("x1"), c("x2"), c("x1", "x2")),
    "hazard", ty, 1, tie_new, tie_times)
}

# ---- influence function of the coefficients with tied events (Breslow fit)
# riskRegression's IFbeta against dfbeta (the score residual times the
# variance) of coxph. Without ties they agree; with tied events IFbeta counts
# a tied event time once in the risk-set sum, where dfbeta counts it d times.
ifdiff <- function(d, fo) {
  f <- coxph(fo, data = d, ties = "breslow", x = TRUE, y = TRUE)
  ib <- iidCox(f, return.object = FALSE)$IFbeta
  db <- residuals(f, type = "dfbeta")
  list(max_diff = max(abs(as.matrix(ib) - as.matrix(db))),
       max_abs = max(abs(as.matrix(db))))
}
ties_note <- list(
  tied = ifdiff(tie, Surv(time, status == 1) ~ x1),
  untied = ifdiff(sim, Surv(time, status == 1) ~ x1 + x3)
)

vers <- list(r = R.version.string,
             riskRegression = as.character(packageVersion("riskRegression")),
             prodlim = as.character(packageVersion("prodlim")),
             survival = as.character(packageVersion("survival")),
             jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = vers, cases = cases,
                ties_note = ties_note),
           digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
