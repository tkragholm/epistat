# Per-protocol weights and the marginal structural model of sequential trials,
# against TrialEmulation::initiators(estimand_type = "PP").
#
# A person-period data set is simulated: a time-varying covariate x2 that
# depends on the previous treatment, a treatment that persists and switches
# with a probability that depends on x2, an outcome and censoring. Two analyses
# are run, with the weights of switching alone and with the weights of censoring
# too. For each the file holds
# - the rows the weight models are fitted on (the data after
#   TrialEmulation:::data_manipulation), the probabilities of the four switch
#   models and the censoring models, and the weight of each row;
# - the expanded trials with their weights, as data_preparation gives them;
# - the coefficients and the cluster-robust covariance of the outcome model and
#   its predicted cumulative incidence under each assigned treatment.
# Numbers are written with 17 digits.
# Run: Rscript trial_msm.R > trial_msm.json
suppressMessages({
  library(jsonlite)
  library(data.table)
  library(TrialEmulation)
  library(sandwich)
})
set.seed(20261126)
n <- 500
periods <- 0:6
rows <- vector("list", n)
for (i in seq_len(n)) {
  x1 <- rbinom(1, 1, 0.5)
  age <- runif(1, 30, 60)
  a_prev <- 0
  x2 <- rnorm(1)
  ever <- 0
  out <- list()
  for (p in periods) {
    x2 <- 0.6 * x2 + 0.3 * a_prev + rnorm(1, 0, 0.8)
    pa <- plogis(if (a_prev == 1) 1.2 + 0.5 * x2 else -1.3 + 0.6 * x2 + 0.4 * x1)
    a <- rbinom(1, 1, pa)
    y <- rbinom(1, 1, plogis(-3.6 - 0.5 * a + 0.35 * x2 + 0.3 * x1 + 0.1 * p))
    cens <- rbinom(1, 1, plogis(-3 + 0.4 * x1 + 0.3 * x2))
    out[[length(out) + 1]] <- data.frame(id = i, period = p, treatment = a, x1 = x1,
      x2 = round(x2, 4), age_s = round((age + p) / 60, 4), outcome = y,
      censored = cens, eligible = as.integer(ever == 0))
    ever <- ever + a
    a_prev <- a
    if (y == 1 || cens == 1) break
  }
  rows[[i]] <- do.call(rbind, out)
}
d <- do.call(rbind, rows)

analysis <- function(use_censor) {
  dir <- tempfile(); dir.create(dir)
  common <- list(id = "id", period = "period", treatment = "treatment",
    outcome = "outcome", eligible = "eligible", estimand_type = "PP",
    outcome_cov = ~x1 + x2 + age_s, switch_d_cov = ~x1 + x2 + age_s,
    switch_n_cov = ~x1)
  if (use_censor) {
    common <- c(common, list(use_censor_weights = TRUE, cense = "censored",
      cense_d_cov = ~x1 + x2 + age_s, cense_n_cov = ~x1, pool_cense = "none"))
  }
  prep <- do.call(data_preparation, c(list(d), common, list(data_dir = dir, quiet = TRUE)))
  args <- prep$parameters
  sel <- TrialEmulation:::select_data_cols(as.data.table(d), args)
  sw <- TrialEmulation:::data_manipulation(sel, use_censor = args$censor_at_switch)
  wr <- do.call(TrialEmulation:::weight_func,
    c(list(sw_data = sw), TrialEmulation:::args_for_fun(args, "weight_func")))
  w <- as.data.frame(wr$data)
  fit <- do.call(initiators, c(list(d), common, list(model_var = "assigned_treatment",
    include_followup_time = ~followup_time + I(followup_time^2),
    include_trial_period = ~trial_period, data_dir = dir, quiet = TRUE)))
  newdata <- as.data.frame(prep$data[trial_period == 0 & followup_time == 0, ])
  pr <- predict(fit, newdata = newdata, predict_times = 0:5, conf_int = FALSE)
  rows <- list(id = w$id, period = w$period, treatment = w$treatment, am_1 = w$am_1,
    censored = if (use_censor) w$censored else NULL, x1 = w$x1, x2 = w$x2, age_s = w$age_s,
    p0_d = w$p0_d, p0_n = w$p0_n, p1_d = w$p1_d, p1_n = w$p1_n,
    pC_d = if (use_censor) w$pC_d else NULL, pC_n = if (use_censor) w$pC_n else NULL,
    wt = w$wt)
  rows <- rows[!vapply(rows, is.null, logical(1))]
  list(
    use_censor = use_censor,
    rows = rows,
    switch_coef = lapply(prep$switch_models, function(m) as.numeric(coef(m$model))),
    cens_coef = if (use_censor) lapply(prep$censor_models, function(m) as.numeric(coef(m$model))) else NULL,
    trials = list(id = prep$data$id, trial_period = prep$data$trial_period,
      followup_time = prep$data$followup_time, outcome = prep$data$outcome,
      weight = prep$data$weight, assigned = prep$data$assigned_treatment,
      x1 = prep$data$x1, x2 = prep$data$x2, age_s = prep$data$age_s),
    msm_names = names(coef(fit$model)),
    msm_coef = as.numeric(coef(fit$model)),
    msm_vcov = as.numeric(fit$robust$matrix),
    msm_se_model = as.numeric(sqrt(diag(vcov(fit$model)))),
    predict_times = pr[[1]]$followup_time,
    cum_inc_0 = pr$assigned_treatment_0$cum_inc,
    cum_inc_1 = pr$assigned_treatment_1$cum_inc
  )
}
out <- list(
  versions = list(R = as.character(getRversion()),
    TrialEmulation = as.character(packageVersion("TrialEmulation")),
    sandwich = as.character(packageVersion("sandwich")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  analyses = list(analysis(FALSE), analysis(TRUE)))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null"))
