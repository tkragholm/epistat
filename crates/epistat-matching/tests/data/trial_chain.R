# The whole chain of a sequential trial emulation, against
# TrialEmulation::initiators: the expansion of epistat-matching, the weights
# and the outcome model of epistat_survival::weighting::trials, and the
# cumulative incidence, run on the same person-period data.
#
# A panel is simulated: a fixed covariate x1, a time-varying covariate x2 that
# depends on the previous treatment, a treatment that persists and switches
# with a probability that depends on x2, an outcome, and censoring that ends a
# person's rows. A person is eligible in a period while never treated before
# it, so a trial enrols new users and non-users, and per-protocol follow-up of
# an initiator ends when the person stops.
#
# Three analyses: intention to treat (no weights), per protocol with switch
# weights, and per protocol with switch and censoring weights. For each the file
# holds the coefficients of the outcome model, its cluster-robust covariance
# (sandwich::vcovCL as TrialEmulation calls it), the number of rows of the
# expanded data and its weights, the coefficients of the weight models (from
# data_preparation with the same arguments), and the cumulative
# incidence under each assigned treatment by follow-up times 0 to 5 over the
# trial starts of the expanded data (predict with its default newdata).
# Numbers are written with 17 digits.
#
# Run from the crate directory:
#   Rscript tests/data/trial_chain.R > tests/data/trial_chain.json
suppressMessages({
  library(jsonlite)
  library(data.table)
  library(TrialEmulation)
  library(sandwich)
})
set.seed(20261126)
n <- 400
periods <- 0:7
rows <- vector("list", n)
for (i in seq_len(n)) {
  x1 <- rbinom(1, 1, 0.5)
  a_prev <- 0
  x2 <- rnorm(1)
  ever <- 0
  out <- list()
  for (p in periods) {
    x2 <- 0.6 * x2 + 0.3 * a_prev + rnorm(1, 0, 0.8)
    pa <- plogis(if (a_prev == 1) 1.4 + 0.5 * x2 else -1.5 + 0.6 * x2 + 0.4 * x1)
    a <- rbinom(1, 1, pa)
    y <- rbinom(1, 1, plogis(-3.2 - 0.6 * a + 0.35 * x2 + 0.3 * x1 + 0.08 * p))
    cens <- rbinom(1, 1, plogis(-3.2 + 0.4 * x1 + 0.3 * x2))
    out[[length(out) + 1]] <- data.frame(id = i, period = p, treatment = a, x1 = x1,
      x2 = round(x2, 4), outcome = y, censored = cens, eligible = as.integer(ever == 0))
    ever <- ever + a
    a_prev <- a
    if (y == 1 || cens == 1) break
  }
  rows[[i]] <- do.call(rbind, out)
}
d <- do.call(rbind, rows)

# The coefficients of the four weight models, numerator and denominator for
# the previous treatment 0 and 1.
coefs <- function(models, prefix) {
  one <- function(name) as.numeric(models[[paste0(prefix, "_", name)]]$summary$estimate)
  list(d0 = one("d0"), n0 = one("n0"), d1 = one("d1"), n1 = one("n1"))
}

analysis <- function(estimand, use_censor) {
  dir <- tempfile(); dir.create(dir)
  common <- list(id = "id", period = "period", treatment = "treatment",
    outcome = "outcome", eligible = "eligible", estimand_type = estimand,
    outcome_cov = ~x1 + x2)
  if (estimand == "PP") {
    common <- c(common, list(switch_d_cov = ~x1 + x2 + period, switch_n_cov = ~x1))
  }
  if (use_censor) {
    common <- c(common, list(use_censor_weights = TRUE, cense = "censored",
      cense_d_cov = ~x1 + x2, cense_n_cov = ~x1, pool_cense = "none"))
  }
  prep <- do.call(data_preparation, c(list(d), common, list(data_dir = dir, quiet = TRUE)))
  fit <- do.call(initiators, c(list(d), common, list(model_var = "assigned_treatment",
    include_followup_time = ~followup_time + I(followup_time^2),
    include_trial_period = ~trial_period, data_dir = dir, quiet = TRUE)))
  pr <- predict(fit, predict_times = 0:5, conf_int = FALSE)
  m <- fit$model$data
  list(
    estimand = estimand,
    use_censor = use_censor,
    rows = nrow(m),
    weight_sum = sum(m$weight),
    weight_max = max(m$weight),
    switch_coef = if (estimand == "PP") coefs(prep$switch_models, "switch") else NULL,
    censor_coef = if (use_censor) coefs(prep$censor_models, "cens") else NULL,
    msm_names = names(coef(fit$model)),
    msm_coef = as.numeric(coef(fit$model)),
    msm_vcov = as.numeric(fit$robust$matrix),
    cum_inc_0 = pr$assigned_treatment_0$cum_inc,
    cum_inc_1 = pr$assigned_treatment_1$cum_inc
  )
}
out <- list(
  versions = list(R = as.character(getRversion()),
    TrialEmulation = as.character(packageVersion("TrialEmulation")),
    sandwich = as.character(packageVersion("sandwich")),
    data.table = as.character(packageVersion("data.table")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  panel = list(id = d$id, period = d$period, treatment = d$treatment,
    eligible = d$eligible, outcome = d$outcome, censored = d$censored,
    x1 = d$x1, x2 = d$x2),
  analyses = list(analysis("ITT", FALSE), analysis("PP", FALSE), analysis("PP", TRUE)))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null", null = "null"))
