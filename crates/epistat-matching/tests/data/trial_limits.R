# Truncated weights in the outcome model of sequential trials, against
# TrialEmulation::initiators with analysis_weights = "p99" and
# "weight_limits", on the panel of trial_chain.R (read from its output, so the
# two references share their data), per protocol with switch weights.
#
# For each the file holds the coefficients of the outcome model, its
# cluster-robust covariance and the cumulative incidence under each assigned
# treatment by follow-up times 0 to 5. Numbers are written with 17 digits.
#
# Run from the crate directory, after trial_chain.R:
#   Rscript tests/data/trial_limits.R > tests/data/trial_limits.json
suppressMessages({
  library(jsonlite)
  library(data.table)
  library(TrialEmulation)
  library(sandwich)
})
d <- as.data.frame(fromJSON("tests/data/trial_chain.json")$panel)
analysis <- function(weights, limits) {
  dir <- tempfile(); dir.create(dir)
  fit <- initiators(d, id = "id", period = "period", treatment = "treatment",
    outcome = "outcome", eligible = "eligible", estimand_type = "PP",
    outcome_cov = ~x1 + x2, switch_d_cov = ~x1 + x2 + period, switch_n_cov = ~x1,
    model_var = "assigned_treatment",
    include_followup_time = ~followup_time + I(followup_time^2),
    include_trial_period = ~trial_period, analysis_weights = weights,
    weight_limits = limits, data_dir = dir, quiet = TRUE)
  pr <- predict(fit, predict_times = 0:5, conf_int = FALSE)
  list(analysis_weights = weights, weight_limits = limits,
    msm_coef = as.numeric(coef(fit$model)),
    msm_vcov = as.numeric(fit$robust$matrix),
    cum_inc_0 = pr$assigned_treatment_0$cum_inc,
    cum_inc_1 = pr$assigned_treatment_1$cum_inc)
}
out <- list(
  versions = list(R = as.character(getRversion()),
    TrialEmulation = as.character(packageVersion("TrialEmulation")),
    sandwich = as.character(packageVersion("sandwich")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  analyses = list(analysis("p99", c(0, Inf)), analysis("weight_limits", c(0.5, 3))))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null"))
