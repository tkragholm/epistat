# riskRegression::Score with a Cox censoring model (conservative = FALSE),
# the contrasts of Score between models, and the calibration slope, intercept
# and pseudo-value curves of riskRegression, on data that is written out.
# Run: Rscript metrics_gaps.R > metrics_gaps.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(riskRegression)
  library(prodlim)
  library(rms)
  library(data.table)
})
set.seed(20261121)
rows <- function(m) lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))

# ---- data: continuous times (no ties), covariate dependent censoring
n <- 300
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.5)
t1 <- rexp(n, 0.35 * exp(0.6 * x1))
t2 <- rexp(n, 0.15 * exp(0.3 * x2))
cc <- rexp(n, 0.2 * exp(0.5 * x2 + 0.3 * x1))
time <- pmin(t1, t2, cc)
cr <- ifelse(time == cc, 0L, ifelse(time == t1, 1L, 2L))
d <- data.frame(time = time, cr = cr, any = as.integer(cr != 0L),
                x1 = x1, x2 = x2)
d <- d[order(d$time), ]
stopifnot(!any(duplicated(d$time)))
times <- c(1, 2, 3, 4)
times_late <- times
stopifnot(max(times) < max(d$time))

# models: a (x1 only) and b (x1 and x2), cause-specific Cox for competing risks
csc_a <- CSC(Hist(time, cr) ~ x1, data = d)
csc_b <- CSC(Hist(time, cr) ~ x1 + x2, data = d)
risk_a <- predictRisk(csc_a, newdata = d, times = times, cause = 1)
risk_b <- predictRisk(csc_b, newdata = d, times = times, cause = 1)
cox_a <- coxph(Surv(time, any) ~ x1, data = d, x = TRUE)
cox_b <- coxph(Surv(time, any) ~ x1 + x2, data = d, x = TRUE)
srisk_a <- predictRisk(cox_a, newdata = d, times = times)
srisk_b <- predictRisk(cox_b, newdata = d, times = times)

# The censoring model as Score fits it: rms::cph with eps = 1e-6.
cph_fit <- function(formula, d, ev) {
  dd <- copy(as.data.table(d))
  dd[, riskRegression_time := time]
  dd[, riskRegression_status := ev]
  f <- suppressWarnings(rms::cph(Surv(riskRegression_time, riskRegression_status == 0) ~ x1 + x2,
                                 data = dd, x = TRUE, y = TRUE, eps = 1e-6, surv = TRUE))
  as.numeric(coef(f))
}

score_run <- function(preds, formula, data, cause = NULL, cens, conservative, contrasts = FALSE) {
  args <- list(object = preds, formula = formula, data = data, times = times,
               metrics = c("auc", "brier"), summary = "ipa", null.model = TRUE,
               se.fit = TRUE, conservative = conservative, cens.model = cens,
               contrasts = contrasts)
  if (!is.null(cause)) args$cause <- cause
  suppressWarnings(do.call(Score, args))
}
extract <- function(sc, models) {
  b <- sc$Brier$score
  a <- sc$AUC$score
  ip <- sc$IPA$score
  out <- list()
  for (m in models) {
    out[[m]] <- list(
      brier = b[b$model == m, ]$Brier, brier_se = b[b$model == m, ]$se,
      auc = if (m != "Null model") a[a$model == m, ]$AUC else NULL,
      auc_se = if (m != "Null model") a[a$model == m, ]$se else NULL,
      ipa = if (m != "Null model") ip[ip$model == m, ]$IPA else NULL,
      ipa_se = if (m != "Null model") ip[ip$model == m, ]$se else NULL)
  }
  out
}
contr <- function(tab) {
  tab <- as.data.frame(tab)
  lapply(seq_len(nrow(tab)), function(i) {
    r <- tab[i, ]
    c(list(times = r$times, model = as.character(r$model), reference = as.character(r$reference)),
      lapply(r[setdiff(names(r), c("times", "model", "reference"))], as.numeric))
  })
}

cases <- list()
# Survival (all-cause death) and competing risks, each with the Cox censoring model
# on x1 + x2 and on x2 alone, non-conservative, and the contrasts of the two models.
for (spec in list(
  list(name = "survival", formula = Surv(time, any) ~ x1 + x2, preds = list(a = srisk_a, b = srisk_b), cause = NULL, cens_cols = c("x1", "x2")),
  list(name = "survival x2", formula = Surv(time, any) ~ x2, preds = list(a = srisk_a, b = srisk_b), cause = NULL, cens_cols = "x2"),
  list(name = "competing", formula = Hist(time, cr) ~ x1 + x2, preds = list(a = risk_a, b = risk_b), cause = 1, cens_cols = c("x1", "x2")),
  list(name = "survival km", formula = Surv(time, any) ~ 1, preds = list(a = srisk_a, b = srisk_b), cause = NULL, cens_cols = NULL),
  list(name = "competing km", formula = Hist(time, cr) ~ 1, preds = list(a = risk_a, b = risk_b), cause = 1, cens_cols = NULL)
)) {
  runs <- list()
  ddw <- copy(as.data.table(d)); ddw[, riskRegression_time := time]; ddw[, riskRegression_status := any]
  if (!is.null(spec$cause)) ddw[, riskRegression_event := cr]
  wts <- riskRegression:::getCensoringWeights(formula = spec$formula, data = ddw, times = times,
                                              cens.model = if (is.null(spec$cens_cols)) "KaplanMeier" else "cox",
                                              influence.curve = FALSE, censoring.save.memory = FALSE, verbose = 0)
  wmax <- max(1 / c(as.numeric(wts$IPCW.times), as.numeric(wts$IPCW.subject.times)))
  for (cons in c(FALSE, TRUE)) {
    cens <- if (is.null(spec$cens_cols)) "km" else "cox"
    sc <- score_run(spec$preds, spec$formula, d, spec$cause, cens, cons, contrasts = TRUE)
    runs[[length(runs) + 1]] <- list(
      conservative = cons, cens = cens,
      models = extract(sc, c("a", "b", "Null model")),
      brier_contrasts = contr(sc$Brier$contrasts),
      auc_contrasts = contr(sc$AUC$contrasts))
  }
  cases[[length(cases) + 1]] <- list(name = spec$name, wmax = wmax, kind = if (is.null(spec$cause)) "survival" else "competing",
                                    cens_cols = if (is.null(spec$cens_cols)) NULL else I(spec$cens_cols), runs = runs)
}

# Censoring weights and coefficients of cph, for the coefficients check.
ev <- d$any
dd <- copy(as.data.table(d)); dd[, riskRegression_time := time]; dd[, riskRegression_status := any]
w <- riskRegression:::getCensoringWeights(formula = Surv(time, any) ~ x1 + x2, data = dd, times = times,
                                          cens.model = "cox", influence.curve = FALSE,
                                          censoring.save.memory = FALSE, verbose = 0)
cens_weights <- list(coef = cph_fit(NULL, d, d$any), at_times = rows(as.matrix(w$IPCW.times)),
                     before_own = as.numeric(w$IPCW.subject.times))

cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  riskRegression = as.character(packageVersion("riskRegression")),
                  prodlim = as.character(packageVersion("prodlim")),
                  rms = as.character(packageVersion("rms")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  data = list(time = d$time, cr = d$cr, any = d$any, x1 = d$x1, x2 = d$x2, times = times,
              risk_a = rows(risk_a), risk_b = rows(risk_b),
              srisk_a = rows(srisk_a), srisk_b = rows(srisk_b)),
  cens_weights = cens_weights,
  cases = cases
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
