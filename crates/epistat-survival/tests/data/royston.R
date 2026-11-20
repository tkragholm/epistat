# survival::royston: the D measure of Royston and Sauerbrei with R.D, R.KO, R.N and C.GH.
# Cases: the lung data with a linear predictor that hardly ties, with a one-variable
# predictor that ties heavily (ph.ecog, four values), with weights and a stratified fit
# (royston refits on fit$y and ignores both), simulated data with tied times and a
# predictor on a coarse grid, and counting-process data with delayed entry. Each case is
# run with ties = TRUE and FALSE and with adjust = FALSE and TRUE. The predictor is
# written out so that a test can feed it to the Rust function, and the coefficients so
# that a test can compare its own fit with R's, with the log likelihoods, the number of
# rows and the condition number of the information. Seventeen digits are written.
# Run: Rscript royston.R > royston.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261121)

lung2 <- lung[complete.cases(lung[, c("time", "status", "age", "sex", "ph.ecog", "wt.loss")]), ]
n1 <- nrow(lung2)

n3 <- 800
x1 <- sample(0:3, n3, replace = TRUE)
x2 <- round(rnorm(n3), 1)
t3 <- round(rexp(n3, 0.1 * exp(0.4 * x1 + 0.5 * x2)), 1) + 0.1
c3 <- round(runif(n3, 0, 25), 1) + 0.1
sim <- data.frame(time = pmin(t3, c3), status = as.integer(t3 <= c3), x1 = x1, x2 = x2)

cp <- sim
cp$start <- round(runif(n3, 0, 0.6) * cp$time, 1)
cp$start <- pmin(cp$start, cp$time - 0.1)

cases <- list(
  list(name = "lung_lp", fit = function() coxph(Surv(time, status) ~ age + sex + ph.ecog + wt.loss, lung2),
       data = lung2, x = c("age", "sex", "ph.ecog", "wt.loss")),
  list(name = "lung_ecog", fit = function() coxph(Surv(time, status) ~ ph.ecog, lung2),
       data = lung2, x = c("ph.ecog")),
  list(name = "lung_weights_strata", fit = function() {
         lung2$w <- round(runif(n1, 0.5, 3), 2); lung2 <<- lung2
         coxph(Surv(time, status) ~ age + ph.ecog + strata(sex), lung2, weights = w)
       }, data = NULL, x = c("age", "ph.ecog")),
  list(name = "sim_ties", fit = function() coxph(Surv(time, status) ~ x1 + x2, sim),
       data = sim, x = c("x1", "x2")),
  list(name = "sim_x1only", fit = function() coxph(Surv(time, status) ~ x1, sim),
       data = sim, x = c("x1")),
  list(name = "sim_counting", fit = function() coxph(Surv(start, time, status) ~ x1 + x2, cp),
       data = cp, x = c("x1", "x2"))
)

out <- list(versions = list(r = R.version.string,
                            survival = as.character(packageVersion("survival")),
                            jsonlite = as.character(packageVersion("jsonlite"))),
            cases = list())
for (cs in cases) {
  fit <- cs$fit()
  d <- if (is.null(cs$data)) lung2 else cs$data
  y <- fit$y
  res <- list()
  for (ties in c(TRUE, FALSE)) for (adjust in c(FALSE, TRUE)) {
    r <- royston(fit, ties = ties, adjust = adjust)
    res[[length(res) + 1]] <- list(ties = ties, adjust = adjust, values = as.list(r))
  }
  rec <- list(name = cs$name,
              stop = unname(y[, ncol(y) - 1]), status = unname(y[, ncol(y)]),
              start = if (ncol(y) == 3) unname(y[, 1]) else NULL,
              weight = if (cs$name == "lung_weights_strata") d$w else NULL,
              stratum = if (cs$name == "lung_weights_strata") as.integer(d$sex) else NULL,
              x = lapply(cs$x, function(v) unname(d[[v]])),
              coefficients = I(unname(coef(fit))),
              eta = unname(predict(fit)),
              nevent = fit$nevent, loglik = fit$loglik, n = fit$n,
              information_condition = kappa(solve(vcov(fit)), exact = TRUE), results = res)
  out$cases[[length(out$cases) + 1]] <- rec
}
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", na = "string"))
