# Calibration by pseudo-values: the nearest-neighbour curve of
# riskRegression::plotCalibration(method = "nne") with cens.method other than
# "local" (the mean of the pseudo-values of Score(cens.method = "pseudo") in a
# neighbourhood), and the logistic and complementary log-log recalibration of
# the pseudo-values on the predictions by geepack::geese.fit. Times are rounded
# so that deaths and censorings tie.
# Run: Rscript metrics_recalibration.R > metrics_recalibration.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(riskRegression)
  library(prodlim)
  library(geepack)
})
set.seed(20261123)
n <- 260
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.5)
t1 <- rexp(n, 0.3 * exp(0.6 * x1))
t2 <- rexp(n, 0.12 * exp(0.3 * x2))
cc <- rexp(n, 0.2 * exp(0.3 * x2))
time <- round(pmin(t1, t2, cc), 2)
time[time == 0] <- 0.01
cr <- ifelse(pmin(t1, t2) > cc, 0L, ifelse(t1 <= t2, 1L, 2L))
d <- data.frame(time = time, cr = cr, any = as.integer(cr != 0L), x1 = x1, x2 = x2)
d <- d[order(d$time, -d$any), ]
times <- c(1, 2.5, 4)
stopifnot(max(times) < max(d$time))
rows <- function(m) lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))

csc <- CSC(Hist(time, cr) ~ x1 + x2, data = d)
risk_cr <- predictRisk(csc, newdata = d, times = times, cause = 1)
cox_any <- coxph(Surv(time, any) ~ x1 + x2, data = d, x = TRUE)
risk_s <- predictRisk(cox_any, newdata = d, times = times)
# A prediction with few distinct values, to see tied neighbourhoods.
few <- as.numeric(cut(d$x1, c(-Inf, -0.7, 0, 0.7, Inf)))
risk_few <- sapply(times, function(t) c(0.1, 0.25, 0.4, 0.6)[few] * (0.5 + 0.2 * t))

jack <- function(kind, tp) {
  if (kind == "survival") {
    f <- prodlim(Hist(time, any) ~ 1, data = d)
    1 - as.numeric(jackknife(f, times = tp))
  } else {
    f <- prodlim(Hist(time, cr, cens.code = 0) ~ 1, data = d)
    as.numeric(jackknife(f, times = tp, cause = 1))
  }
}

nne_case <- function(kind, risk, j, bw, round, label) {
  tp <- times[j]
  formula <- if (kind == "survival") Surv(time, any) ~ 1 else Hist(time, cr) ~ 1
  sc <- suppressWarnings(Score(list(m = risk), formula = formula, data = d, times = times,
                               metrics = "brier", plots = "calibration", cens.method = "pseudo",
                               null.model = FALSE, se.fit = FALSE,
                               cause = if (kind == "survival") NULL else 1))
  pf <- plotCalibration(sc, times = tp, method = "nne", cens.method = "jackknife",
                        bandwidth = bw, round = round, plot = FALSE)$plotFrames[[1]]
  list(label = label, kind = kind, j = j - 1, bandwidth = bw, round = round,
       pred = I(as.numeric(pf$Pred)), obs = I(as.numeric(pf$Obs)),
       r_bandwidth = as.numeric(attr(pf, "bandwidth")))
}
nne <- list(
  nne_case("survival", risk_s, 1, 0.1, 2, "survival bw 0.1 round 2"),
  nne_case("survival", risk_s, 2, n^(-1/4), 2, "survival smooth round 2"),
  nne_case("survival", risk_s, 3, 0.3, 0, "survival bw 0.3 no rounding"),
  nne_case("competing", risk_cr, 1, 0.1, 2, "competing bw 0.1 round 2"),
  nne_case("competing", risk_cr, 2, 0.05, 3, "competing bw 0.05 round 3"),
  nne_case("competing", risk_cr, 3, 0.2, 2, "competing bw 0.2 round 2"),
  nne_case("competing", risk_few, 2, 0.1, 2, "competing few distinct, bw 0.1"),
  nne_case("competing", risk_few, 3, 0.4, 2, "competing few distinct, bw 0.4"),
  nne_case("survival", risk_s, 2, 1, 2, "survival bw 1")
)

# R's round, for the rounding rule.
set.seed(1)
rv <- c(0.125, 0.375, 0.625, 0.005, 0.015, 0.025, 0.045, 0.255, 0.115, 0.995, runif(60))
rounding <- list(x = rv, d2 = round(rv, 2), d3 = round(rv, 3), d0 = round(rv, 0), d1 = round(rv, 1))

ctrl <- geese.control(epsilon = 1e-13, maxit = 200)
slope_case <- function(kind, risk, j, link) {
  tp <- times[j]
  y <- jack(kind, tp)
  lp <- if (link == "logit") qlogis(risk[, j]) else log(-log(1 - risk[, j]))
  X <- cbind(1, lp)
  g <- geese.fit(x = X, y = y, id = seq_len(n), mean.link = link, family = gaussian(),
                 variance = "gaussian", corstr = "independence", control = ctrl, b = NULL)
  list(kind = kind, j = j - 1, link = link, coef = as.numeric(g$beta),
       vcov = as.numeric(t(g$vbeta)), converged = !as.logical(g$error), y = y)
}
slopes <- list()
for (kind in c("survival", "competing")) {
  risk <- if (kind == "survival") risk_s else risk_cr
  for (j in 1:3) for (link in c("logit", "cloglog")) {
    slopes[[length(slopes) + 1]] <- slope_case(kind, risk, j, link)
  }
}

cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  riskRegression = as.character(packageVersion("riskRegression")),
                  prodlim = as.character(packageVersion("prodlim")),
                  geepack = as.character(packageVersion("geepack")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  data = list(time = d$time, cr = d$cr, any = d$any, times = times,
              risk_s = rows(risk_s), risk_cr = rows(risk_cr), risk_few = rows(risk_few)),
  nne = nne, slopes = slopes, rounding = rounding
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
