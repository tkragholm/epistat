# riskRegression::Score (Brier, IPA, AUC, calibration by risk group) and
# timeROC::timeROC (AUC) on predicted risks that are given.
#
# Three data sets:
#   sim      right-censored times from an exponential model with known risks, rounded
#            to two decimals so that deaths and censorings tie; the risks are the
#            true risks 1 - exp(-rate t).
#   melanoma survival: time to death from any cause in riskRegression's Melanoma
#            data, with risks from a Cox model.
#   melanoma_cr competing risks: death from melanoma against death from other
#            causes, with absolute risks from cause-specific Cox models (CSC).
# For each data set and each set of options: the Score output (Brier, its se, the
# null model's, the IPA and its se, the AUC and its se), with the Kaplan-Meier
# censoring model, with conservative = TRUE, and with a Cox censoring model whose
# weights are written so that the Rust code can be given them. timeROC's AUC and
# its iid-based se are written for the same markers and times, for both
# definitions with competing risks. Calibration is plotCalibration's quantile
# frame at one time.
# Seventeen digits are written so that every double reads back exactly. The
# versions of R and of every package are in the file.
# Run: Rscript metrics_score.R > metrics_score.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(riskRegression)
  library(timeROC)
  library(prodlim)
  library(data.table)
})
set.seed(20261120)

mat <- function(x) unname(as.matrix(x))
rows <- function(m) lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))

# ---- data
n <- 350
x <- round(rnorm(n), 1)
rate <- 0.5 * exp(0.7 * x)
Tev <- rexp(n, rate)
Cen <- rexp(n, 0.45)
sim <- data.frame(time = round(pmin(Tev, Cen), 2), status = as.integer(Tev <= Cen), x = x)
sim <- sim[order(sim$time, -sim$status), ]
sim$time[sim$time == 0] <- 0.01
sim <- sim[order(sim$time, -sim$status), ]
sim_times <- c(0.5, 1, 1.5, 2)
# Times that are observed values, to see how a death at the horizon is counted.
sim_times_obs <- sort(c(sim$time[which(sim$status == 1)[c(20, 60, 100)]],
                        sim$time[which(sim$status == 0)[c(30, 70)]]))
sim_rate <- 0.5 * exp(0.7 * sim$x)
risk_at <- function(t) 1 - exp(-sim_rate * t)

data(Melanoma, package = "riskRegression")
mel <- Melanoma
mel$status <- NULL
mel$any <- as.integer(mel$event != "censored")
mel$cr <- as.integer(factor(mel$event, levels = c("censored", "death.malignant.melanoma", "death.other.causes"))) - 1L
mel <- mel[order(mel$time, -mel$any), ]
mel_times <- c(1000, 1800, 2500, 3000)

csc <- CSC(Hist(time, cr) ~ age + sex + invasion + ulcer, data = mel)
mel_cr_risk <- predictRisk(csc, newdata = mel, times = mel_times, cause = 1)
cox_any <- coxph(Surv(time, any) ~ age + sex + invasion + ulcer, data = mel, x = TRUE)
mel_s_risk <- predictRisk(cox_any, newdata = mel, times = mel_times)

# ---- Score wrappers
score_out <- function(sc, cr) {
  b <- sc$Brier$score
  a <- sc$AUC$score
  ip <- sc$IPA$score
  keep <- function(d, m) d[d$model == m, ]
  list(
    brier = keep(b, "m")$Brier, brier_se = keep(b, "m")$se,
    null_brier = keep(b, "Null model")$Brier, null_se = keep(b, "Null model")$se,
    ipa = keep(ip, "m")$IPA, ipa_se = keep(ip, "m")$se,
    auc = a$AUC, auc_se = a$se
  )
}

run_score <- function(label, d, formula, pred, times, cause = NULL, cens = "km",
                      conservative = FALSE, cens_formula = NULL) {
  args <- list(object = list(m = pred), formula = formula, data = d, times = times,
               metrics = c("auc", "brier"), summary = "ipa", null.model = TRUE,
               se.fit = TRUE, conservative = conservative, cens.model = cens,
               contrasts = FALSE)
  if (!is.null(cause)) args$cause <- cause
  sc <- do.call(Score, args)
  out <- score_out(sc)
  out$label <- label
  out$cens <- cens
  out$conservative <- conservative
  out$times <- times
  out
}

# Weights of a Cox model for the censoring times, as Score forms them, in the
# order of the (already sorted) data.
cox_weights <- function(d, formula, times, event_col) {
  dd <- as.data.table(d)
  dd[, riskRegression_time := dd$time]
  dd[, riskRegression_status := as.integer(dd[[event_col]] != 0)]
  w <- riskRegression:::getCensoringWeights(formula = formula, data = dd, times = times,
                                            cens.model = "cox", influence.curve = FALSE,
                                            censoring.save.memory = FALSE, verbose = 0)
  list(at_times = rows(as.matrix(w$IPCW.times)), before_own = as.numeric(w$IPCW.subject.times))
}

cases <- list()
add <- function(...) cases[[length(cases) + 1]] <<- list(...)

# sim: survival, known risks
pred_sim <- sapply(sim_times, risk_at)
add(name = "sim", dataset = "sim", kind = "survival", times = sim_times,
    score = list(
      run_score("km", sim, Surv(time, status) ~ 1, pred_sim, sim_times),
      run_score("km conservative", sim, Surv(time, status) ~ 1, pred_sim, sim_times, conservative = TRUE),
      c(run_score("cox conservative", sim, Surv(time, status) ~ x, pred_sim, sim_times,
                  cens = "cox", conservative = TRUE),
        list(weights = cox_weights(sim, Surv(time, status) ~ x, sim_times, "status")))
    ))
pred_sim_obs <- sapply(sim_times_obs, risk_at)
add(name = "sim at observed times", dataset = "sim", kind = "survival", times = sim_times_obs,
    score = list(run_score("km", sim, Surv(time, status) ~ 1, pred_sim_obs, sim_times_obs)))

# melanoma: all-cause death
add(name = "melanoma", dataset = "melanoma", kind = "survival", times = mel_times,
    score = list(
      run_score("km", mel, Surv(time, any) ~ 1, mel_s_risk, mel_times),
      run_score("km conservative", mel, Surv(time, any) ~ 1, mel_s_risk, mel_times, conservative = TRUE),
      c(run_score("cox conservative", mel, Surv(time, any) ~ age + sex, mel_s_risk, mel_times,
                  cens = "cox", conservative = TRUE),
        list(weights = cox_weights(mel, Surv(time, any) ~ age + sex, mel_times, "any")))
    ))

# melanoma: competing risks
add(name = "melanoma competing risks", dataset = "melanoma", kind = "competing", times = mel_times,
    score = list(
      run_score("km", mel, Hist(time, cr) ~ 1, mel_cr_risk, mel_times, cause = 1),
      run_score("km conservative", mel, Hist(time, cr) ~ 1, mel_cr_risk, mel_times, cause = 1,
                conservative = TRUE)
    ))

# ---- timeROC
troc <- function(d, delta, marker, times, cause = 1) {
  r <- timeROC(T = d$time, delta = delta, marker = marker, cause = cause, times = times, iid = TRUE)
  has2 <- !is.null(r$AUC_2)
  list(times = r$times,
       auc1 = if (has2) as.numeric(r$AUC_1) else as.numeric(r$AUC),
       auc2 = if (has2) as.numeric(r$AUC_2) else NULL,
       se1 = as.numeric(r$inference$vect_sd_1), se2 = as.numeric(r$inference$vect_sd_2),
       stats = rows(r$Stats))
}
tr <- list()
tr$sim <- troc(sim, sim$status, sim$x, c(0.5, 1, 1.5, 2))
tr$sim_obs <- troc(sim, sim$status, sim$x, sim_times_obs)
mk <- mel_cr_risk[, 2]
tr$melanoma <- troc(mel, mel$any, mk, c(1000, 1800, 2500, 3000))
tr$melanoma_cr <- troc(mel, mel$cr, mk, c(1000, 1800, 2500, 3000), cause = 1)
# Times that are observed in the melanoma data (ties at the horizon).
mel_obs_times <- sort(unique(mel$time))[c(60, 120, 160)]
tr$melanoma_cr_obs <- troc(mel, mel$cr, mk, mel_obs_times, cause = 1)
mel_obs_risk <- predictRisk(csc, newdata = mel, times = mel_obs_times, cause = 1)
add(name = "melanoma competing risks at observed times", dataset = "melanoma", kind = "competing",
    times = mel_obs_times,
    score = list(run_score("km", mel, Hist(time, cr) ~ 1, mel_obs_risk, mel_obs_times, cause = 1)))
# Marker of timeROC: the risk at the middle time, identical for every time; Score
# gets a matrix with that column repeated.
mk_mat <- function(m, k) matrix(rep(m, k), ncol = k)

# ---- calibration
cal_one <- function(d, formula, pred, times, tp, q = 5, cause = NULL) {
  args <- list(object = list(m = pred), formula = formula, data = d, times = times,
               metrics = "brier", plots = "calibration", null.model = FALSE, se.fit = FALSE)
  if (!is.null(cause)) args$cause <- cause
  sc <- do.call(Score, args)
  pc <- plotCalibration(sc, times = tp, method = "quantile", q = q, plot = FALSE)
  f <- pc$plotFrames[[1]]
  list(time = tp, q = q, pred = as.numeric(f$Pred), obs = as.numeric(f$Obs),
       groups = as.numeric(attr(f, "quantiles")))
}
cal <- list(
  sim = lapply(c(0.5, 1, 2), function(tp) cal_one(sim, Surv(time, status) ~ 1, pred_sim, sim_times, tp)),
  sim_10 = list(cal_one(sim, Surv(time, status) ~ 1, pred_sim, sim_times, 1, q = 10)),
  melanoma = lapply(c(1000, 2500, 3000), function(tp)
    cal_one(mel, Surv(time, any) ~ 1, mel_s_risk, mel_times, tp, q = 4)),
  melanoma_cr = lapply(c(1000, 2500, 3000), function(tp)
    cal_one(mel, Hist(time, cr) ~ 1, mel_cr_risk, mel_times, tp, q = 4, cause = 1))
)

# Few distinct predictions: plotCalibration orders the predictions apart from the
# rows (factor(sort(p))), so its observed risks belong to other rows. The groups
# by value, with each row in its own group, are computed by hand with prodlim.
few <- as.numeric(cut(sim$x, c(-Inf, -0.5, 0.5, Inf)))
risk_few <- c(0.2, 0.45, 0.7)[few]
pred_few <- sapply(sim_times, function(t) risk_few)
sc_few <- Score(list(m = pred_few), Surv(time, status) ~ 1, data = sim, times = sim_times,
                metrics = "brier", plots = "calibration", null.model = FALSE, se.fit = FALSE)
pc_few <- plotCalibration(sc_few, times = 1, method = "quantile", q = 5, plot = FALSE)$plotFrames[[1]]
g_few <- factor(risk_few)
fit_few <- prodlim(Hist(time, status) ~ g_few, data = sim)
obs_aligned <- as.numeric(predict(fit_few, newdata = data.frame(g_few = levels(g_few)), cause = 1,
                                  mode = "matrix", times = 1, type = "cuminc"))
cal_few <- list(time = 1, risk = risk_few, r_pred = as.numeric(pc_few$Pred), r_obs = as.numeric(pc_few$Obs),
                values = as.numeric(levels(g_few)), obs_aligned = obs_aligned)

datasets <- list(
  sim = list(time = sim$time, status = sim$status, x = sim$x, risk = rows(pred_sim),
             risk_obs = rows(pred_sim_obs)),
  melanoma = list(time = mel$time, status = mel$any, cr = mel$cr, risk_s = rows(mel_s_risk),
                  risk_cr = rows(mel_cr_risk), risk_cr_obs = rows(mel_obs_risk))
)

cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  riskRegression = as.character(packageVersion("riskRegression")),
                  timeROC = as.character(packageVersion("timeROC")),
                  prodlim = as.character(packageVersion("prodlim")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  datasets = datasets, cases = cases, timeroc = tr, calibration = cal, calibration_few = cal_few
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
