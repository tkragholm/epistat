# Augmented inverse probability weighting for the risk at a time with right
# censoring, against riskRegression::ate(estimator = c("AIPTW", "IPTW")).
#
# Scenarios: a Kaplan-Meier censoring model, a Cox censoring model with
# covariates and the treatment, and tied event times (Breslow outcome model).
# The outcome model is a Cox model with the treatment among the covariates, the
# treatment model a logistic regression. ate is run with known.nuisance = TRUE,
# so its standard error is that of the influence function of the estimator with
# the nuisance estimates fixed, and with FALSE for the record. The predictions
# the Rust code uses are written too (the estimates of ate are the reference).
# Numbers are written with 17 digits.
# Run: Rscript aipw_surv.R > aipw_surv.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(prodlim)
  library(riskRegression)
})
set.seed(20261125)
make <- function(n, ties) {
  x1 <- round(rnorm(n), 3)
  x2 <- rbinom(n, 1, 0.4)
  a <- rbinom(n, 1, plogis(0.5 * x1 - 0.4 * x2))
  te <- rexp(n, 0.12 * exp(0.5 * x1 - 0.4 * x2 - 0.6 * a))
  tc <- pmin(rexp(n, 0.06 * exp(0.3 * x1 + 0.3 * a)), 12)
  time <- pmin(te, tc)
  status <- as.integer(te <= tc)
  if (ties) time <- ceiling(time * 4) / 4 else time <- round(time, 6)
  data.frame(time = time, status = status, a = a, x1 = x1, x2 = x2)
}
scenario <- function(name, d, censor_formula, times, ties_method) {
  a_num <- d$a
  d$a <- factor(d$a, levels = c(0, 1))
  fit_y <- coxph(Surv(time, status) ~ a + x1 + x2, data = d, x = TRUE, y = TRUE,
                 ties = ties_method)
  fit_a <- glm(a ~ x1 + x2, data = d, family = binomial)
  fit_c <- if (censor_formula == "km") {
    prodlim(Hist(time, status) ~ a, data = transform(d, status = 1L - status))
  } else {
    coxph(Surv(time, status == 0) ~ a + x1 + x2, data = d, x = TRUE, y = TRUE,
          ties = ties_method)
  }
  run <- function(known) {
    suppressWarnings(ate(fit_y, treatment = fit_a, censor = fit_c, data = d,
                         times = times, estimator = c("AIPTW", "IPTW"),
                         known.nuisance = known, verbose = FALSE, se = TRUE))
  }
  r1 <- run(TRUE)
  r0 <- run(FALSE)
  tab <- function(r, est) {
    m <- as.data.frame(r$meanRisk)
    m <- m[m$estimator == est, ]
    dd <- as.data.frame(r$diffRisk)
    dd <- dd[dd$estimator == est, ]
    list(mean1 = m$estimate[m$treatment == "1"], se1 = m$se[m$treatment == "1"],
         mean0 = m$estimate[m$treatment == "0"], se0 = m$se[m$treatment == "0"],
         diff = dd$estimate, diff_se = dd$se,
         ratio = as.data.frame(r$ratioRisk)[as.data.frame(r$ratioRisk)$estimator == est, ]$estimate)
  }
  list(name = name, times = times, ties = ties_method,
       censor = censor_formula,
       time = d$time, status = d$status, a = a_num, x1 = d$x1, x2 = d$x2,
       propensity = as.numeric(fitted(fit_a)),
       aiptw = tab(r1, "AIPTW"), iptw = tab(r1, "IPTW"),
       aiptw_full = tab(r0, "AIPTW"), iptw_full = tab(r0, "IPTW"))
}
d1 <- make(500, FALSE)
d2 <- make(500, TRUE)
out <- list(
  versions = list(R = as.character(getRversion()),
                  riskRegression = as.character(packageVersion("riskRegression")),
                  survival = as.character(packageVersion("survival")),
                  prodlim = as.character(packageVersion("prodlim")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  scenarios = list(
    scenario("km", d1, "km", c(2, 5, 8), "breslow"),
    scenario("cox", d1, "cox", c(2, 5, 8), "breslow"),
    scenario("ties_km", d2, "km", c(2, 5, 8), "breslow"),
    scenario("ties_cox", d2, "cox", c(2, 5, 8), "breslow")))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null"))
