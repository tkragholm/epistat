# survival::rttright and survival::brier.
#
# rttright: the lung data with and without reporting times, case weights, strata,
# renorm off, a competing-risks status (a factor, so an mright Surv), and data
# with many ties between deaths and censorings, with and without timefix. For
# each case: the data, the call and the weight matrix (one row per row, one
# column per time).
# brier: a Cox model on the lung data, without and with case weights (some of
# them 0), with ties on and off, at the default times (the event times) and at
# chosen times, and on data with ties between deaths and censorings. For each
# case: the data and the design, the times, the Brier score, the R-squared and
# the detail (p0, phat, eff.n).
# Seventeen digits are written so that every double reads back exactly. The
# versions of R, survival and jsonlite are in the file.
# Run: Rscript metrics_survival.R > metrics_survival.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)

lung2 <- lung[complete.cases(lung[, c("time", "status", "age", "sex", "ph.ecog")]), ]
lung2$event <- lung2$status - 1L
n <- nrow(lung2)
lung2$w <- sample(c(0, 1, 1, 2, 3, 0.5), n, replace = TRUE)
# coxph refuses weights of 0, so the weights of the brier cases have none.
lung2$wb <- ifelse(lung2$w == 0, 1.5, lung2$w)
lung2$cause <- factor(ifelse(lung2$event == 0, "cens", ifelse(runif(n) < 0.6, "a", "b")),
                      levels = c("cens", "a", "b"))
lung2$sexf <- lung2$sex

# Ties between deaths and censorings, and times that differ by rounding error.
m <- 400
tied <- data.frame(
  time = sample(1:30, m, replace = TRUE),
  event = rbinom(m, 1, 0.5),
  x = round(rnorm(m), 1)
)
tied$w <- sample(c(1, 1, 2), m, replace = TRUE)
tied$wb <- tied$w
fuzzy <- tied
fuzzy$time <- fuzzy$time + sample(c(0, 1e-10, 3e-9), m, replace = TRUE)

mat <- function(x) if (is.matrix(x)) unname(x) else as.numeric(x)

rtt <- list()
add_rtt <- function(name, df, resp, rhs, times = NULL, weights = NULL, ...) {
  f <- as.formula(paste(resp, "~", rhs))
  args <- list(formula = f, data = df, ...)
  if (!is.null(times)) args$times <- times
  if (!is.null(weights)) args$weights <- as.name(weights)
  w <- do.call(rttright, args)
  rtt[[length(rtt) + 1]] <<- list(
    name = name, data = deparse(substitute(df)), resp = resp, rhs = rhs,
    times = times, weights = weights, renorm = list(...)$renorm,
    timefix = list(...)$timefix, w = mat(w), ncol = if (is.null(times)) 1L else length(times)
  )
}
add_rtt("lung, end of follow-up", lung2, "Surv(time, event)", "1")
add_rtt("lung, times", lung2, "Surv(time, event)", "1", times = c(100, 200, 365, 500))
add_rtt("lung, times, no renorm", lung2, "Surv(time, event)", "1", times = c(100, 365), renorm = FALSE)
add_rtt("lung, weights, times", lung2, "Surv(time, event)", "1", times = c(100, 200, 365, 500), weights = "w")
add_rtt("lung, weights, no renorm", lung2, "Surv(time, event)", "1", times = c(150, 450), weights = "w", renorm = FALSE)
add_rtt("lung, strata, times", lung2, "Surv(time, event)", "strata(sexf)", times = c(100, 200, 365, 500))
add_rtt("lung, strata, weights, end of follow-up", lung2, "Surv(time, event)", "strata(sexf)", weights = "w")
add_rtt("lung, competing causes, times", lung2, "Surv(time, cause)", "1", times = c(100, 200, 365, 500))
add_rtt("lung, competing causes, end of follow-up", lung2, "Surv(time, cause)", "1")
add_rtt("ties, end of follow-up", tied, "Surv(time, event)", "1")
add_rtt("ties, times at observed times", tied, "Surv(time, event)", "1", times = c(1, 5, 10, 17, 30, 31))
add_rtt("ties, weights, times before and after", tied, "Surv(time, event)", "1", times = c(0.5, 12, 29.5, 40), weights = "w")
add_rtt("near ties, timefix", fuzzy, "Surv(time, event)", "1", times = c(5, 10, 20))
add_rtt("near ties, timefix off", fuzzy, "Surv(time, event)", "1", times = c(5, 10, 20), timefix = FALSE)

# brier
dd <- function(df) as.list(df)
bri <- list()
add_bri <- function(name, df, formula, times = NULL, weights = NULL, ties = TRUE, timefix = TRUE) {
  # brier reads the data through fit$call$data, so the data frame is in scope.
  assign("bdata", df, envir = globalenv())
  fit <- if (is.null(weights)) coxph(formula, data = bdata) else coxph(formula, data = bdata, weights = wb)
  args <- list(fit = fit, ties = ties, detail = TRUE, timefix = timefix)
  if (!is.null(times)) args$times <- times
  b <- do.call(brier, args)
  x <- model.matrix(fit)
  # brier estimates the baseline with survfit, which applies aeqSurv whatever the
  # timefix argument of brier says. The same Kaplan-Meier estimate with the
  # argument honoured is p0_asked.
  cw <- if (is.null(weights)) rep(1, nrow(df)) else df[[weights]]
  s0 <- survfit(Surv(time, event) ~ 1, data = df, weights = cw, timefix = timefix, stype = 1)
  p0_asked <- 1 - summary(s0, b$times, extend = TRUE)$surv
  bri[[length(bri) + 1]] <<- list(
    name = name, data = deparse(substitute(df)), weights = weights, ties = ties, timefix = timefix,
    times_given = !is.null(times), times = b$times, brier = b$brier, rsquared = b$rsquared,
    p0 = b$p0, p0_asked = p0_asked, phat = mat(b$phat), eff_n = b$eff.n, x = mat(x),
    coefficients = I(unname(coef(fit)))
  )
}
add_bri("lung", lung2, Surv(time, event) ~ age + sex + ph.ecog, times = c(100, 200, 300, 400, 500))
add_bri("lung, default times", lung2, Surv(time, event) ~ age + sex + ph.ecog)
add_bri("lung, weights", lung2, Surv(time, event) ~ age + sex + ph.ecog, times = c(100, 200, 300, 400, 500), weights = "wb")
add_bri("lung, ties off", lung2, Surv(time, event) ~ age + sex + ph.ecog, times = c(100, 200, 300, 400, 500), ties = FALSE)
add_bri("tied times", tied, Surv(time, event) ~ x, times = c(3, 7, 10, 20, 29))
add_bri("tied times, ties off", tied, Surv(time, event) ~ x, times = c(3, 7, 10, 20, 29), ties = FALSE)
add_bri("tied times, weights", tied, Surv(time, event) ~ x, times = c(3, 7, 10, 20, 30), weights = "wb")
add_bri("tied times, timefix off", tied, Surv(time, event) ~ x, times = c(3, 7, 10, 20, 29), timefix = FALSE)
add_bri("near ties", fuzzy, Surv(time, event) ~ x, times = c(5, 10, 20))
add_bri("near ties, timefix off", fuzzy, Surv(time, event) ~ x, times = c(5, 10, 20), timefix = FALSE)

# Dataset columns in the order the rttright cases name them.
datasets <- list(
  lung = list(time = lung2$time, event = lung2$event, status_cause = as.integer(lung2$cause) - 1L,
              w = lung2$w, wb = lung2$wb, sex = lung2$sex, age = lung2$age, ph_ecog = lung2$ph.ecog),
  tied = list(time = tied$time, event = tied$event, w = tied$w, wb = tied$wb, x = tied$x),
  fuzzy = list(time = fuzzy$time, event = fuzzy$event, w = fuzzy$w, wb = fuzzy$w, x = fuzzy$x)
)

cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  datasets = datasets, rttright = rtt, brier = bri
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
