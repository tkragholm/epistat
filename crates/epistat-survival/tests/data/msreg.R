# Multi-state Cox regression and its predictions, by survival 3.8-12 and
# mstate 0.3.3, on the ebmt3 data of mstate: a transplant (state 1), platelet
# recovery (state 2) and relapse or death (state 3) in an illness-death
# process, for a random 700 of the 2204 patients, with three covariates
# (age 20-40, age over 40, T-cell depletion). Times are in days and tie heavily.
#
# fits: coxph with a multi-state outcome, Surv(t1, t2, state) with id and
# istate, for separate coefficients in every transition, common coefficients,
# a mix of common, separate and omitted coefficients, and shared baselines
# with proportional baselines (coxph's "shared"), with Efron and Breslow ties.
# For each: the coefficients, the robust covariance coxph returns, the
# model-based covariance, the log likelihoods, the iteration count, the
# condition number of the information matrix, and the terms read from coxph's
# cmap, for the Rust test to build the same model: for each coefficient in
# order the covariate (0-based, -1 for the constant of a proportional
# baseline) and the transitions (0-based).
#
# msfit: the stacked data of msprep fitted with coxph and a strata(trans) term,
# for separate coefficients (expand.covs) and for common coefficients, and a
# shared-baseline model with indicators of transitions, with Efron and Breslow
# ties; for a covariate profile the cumulative hazards, the covariance of
# them (mstate's varHaz), and probtrans at predt = 0 and 200 with its
# standard errors. mstate's stacked fit has no cluster, so its covariance is the
# model-based one.
#
# Run: Rscript msreg.R > msreg.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(mstate)
})
set.seed(20261120)
data(ebmt3)
d <- ebmt3[sort(sample(nrow(ebmt3), 700)), ]
d$a1 <- as.numeric(d$age == "20-40")
d$a2 <- as.numeric(d$age == ">40")
d$tcd1 <- as.numeric(d$tcd == "TCD")
covs <- c("a1", "a2", "tcd1")

rows <- list()
for (i in seq_len(nrow(d))) {
  r <- d[i, ]
  if (r$prstat == 1) {
    rows[[length(rows) + 1]] <- data.frame(id = r$id, t1 = 0, t2 = r$prtime, st = 2, is = 1)
    rows[[length(rows) + 1]] <- data.frame(id = r$id, t1 = r$prtime, t2 = r$rfstime,
                                           st = ifelse(r$rfsstat == 1, 3, 0), is = 2)
  } else {
    rows[[length(rows) + 1]] <- data.frame(id = r$id, t1 = 0, t2 = r$rfstime,
                                           st = ifelse(r$rfsstat == 1, 3, 0), is = 1)
  }
}
ms <- do.call(rbind, rows)
ms <- merge(ms, d[, c("id", covs)], by = "id", sort = FALSE)
ms <- ms[order(ms$id, ms$t2), ]
ms$stf <- factor(ms$st, 0:3, c("censor", "1", "2", "3"))
rownames(ms) <- NULL

# The condition number of the information matrix on the stacked covariates
# scaled as coxph scales them: centred, divided by the mean absolute deviation.
# The stacked design is built from the terms: one row per row of a state and
# transition that leaves it.
trs <- unique(ms[ms$st > 0, c("is", "st")])
trs <- trs[order(trs$is, trs$st), ]
cond <- function(fit, terms) {
  x <- do.call(rbind, lapply(seq_len(nrow(trs)), function(k) {
    rws <- ms[ms$is == trs$is[k], ]
    sapply(terms, function(t) {
      if (!((k - 1L) %in% unlist(t$transitions))) return(rep(0, nrow(rws)))
      if (t$column < 0) rep(1, nrow(rws)) else rws[[covs[t$column + 1L]]]
    })
  }))
  scale <- colMeans(abs(sweep(x, 2, colMeans(x))))
  D <- diag(scale, length(scale))
  kappa(D %*% solve(fit$naive.var) %*% D, exact = TRUE)
}
terms_of <- function(fit) {
  cm <- fit$cmap
  vars <- rownames(cm)
  codes <- sort(unique(cm[cm > 0]))
  lapply(codes, function(code) {
    w <- which(cm == code, arr.ind = TRUE)
    v <- vars[w[1, 1]]
    list(column = if (startsWith(v, "ph(")) -1L else match(v, covs) - 1L,
         transitions = as.list(unname(w[, 2] - 1L)))
  })
}
one_fit <- function(name, formula, ties, baseline) {
  f <- coxph(formula, id = id, istate = is, data = ms, ties = ties)
  list(name = name, ties = ties, baseline = baseline, terms = terms_of(f),
       coefficients = unname(coef(f)), vcov_robust = unname(f$var),
       vcov_model = unname(f$naive.var), loglik = f$loglik[2], loglik_null = f$loglik[1],
       iterations = f$iter, information_condition = cond(f, terms_of(f)), n_events = f$nevent)
}
fits <- list(
  one_fit("separate_efron", Surv(t1, t2, stf) ~ a1 + a2 + tcd1, "efron", "separate"),
  one_fit("separate_breslow", Surv(t1, t2, stf) ~ a1 + a2 + tcd1, "breslow", "separate"),
  one_fit("common", list(Surv(t1, t2, stf) ~ 1, 1:2 + 1:3 + 2:3 ~ (a1 + a2 + tcd1) / common), "efron", "separate"),
  one_fit("mixed", list(Surv(t1, t2, stf) ~ 1, 1:2 + 1:3 ~ a1 / common, 2:3 ~ tcd1 + a2), "efron", "separate"),
  one_fit("shared", list(Surv(t1, t2, stf) ~ 1, 1:2 + 1:3 + 2:3 ~ (a1 + tcd1) / shared), "efron", "shared")
)

# ---- msfit and probtrans
tmat <- trans.illdeath()
msd <- msprep(time = c(NA, "prtime", "rfstime"), status = c(NA, "prstat", "rfsstat"),
              data = d, trans = tmat, keep = covs)
msexp <- expand.covs(msd, covs, longnames = FALSE)
profile <- c(a1 = 1, a2 = 0, tcd1 = 1)
newbase <- function() {
  n <- data.frame(a1 = 1, a2 = 0, tcd1 = 1, trans = 1:3)
  attr(n, "trans") <- tmat
  class(n) <- c("msdata", "data.frame")
  n
}
pt_list <- function(pt, S) {
  lapply(1:S, function(s) {
    x <- pt[[s]]
    list(time = x$time, p = unname(as.matrix(x[, paste0("pstate", 1:S)])),
         se = unname(as.matrix(x[, paste0("se", 1:S)])))
  })
}
msf_case <- function(name, cox, newdata, ties, shared = FALSE, extra = NULL) {
  msf <- msfit(cox, newdata, trans = tmat)
  H <- msf$Haz
  vh <- msf$varHaz
  list(name = name, ties = ties, profile = unname(profile),
       coefficients = unname(coef(cox)), vcov = unname(cox$var),
       time = sort(unique(H$time)),
       haz = lapply(1:3, function(k) H$Haz[H$trans == k]),
       varhaz = lapply(seq_len(nrow(unique(vh[, c("trans1", "trans2")]))), function(j) {
         u <- unique(vh[, c("trans1", "trans2")])[j, ]
         list(trans1 = u$trans1, trans2 = u$trans2,
              value = vh$varHaz[vh$trans1 == u$trans1 & vh$trans2 == u$trans2])
       }),
       probtrans0 = pt_list(probtrans(msf, predt = 0), 3),
       probtrans200 = pt_list(probtrans(msf, predt = 200), 3),
       extra = extra)
}
cases <- list()
for (ties in c("efron", "breslow")) {
  f <- as.formula(paste("Surv(Tstart, Tstop, status) ~",
                        paste(c(outer(covs, 1:3, paste, sep = ".")), collapse = " + "),
                        "+ strata(trans)"))
  cox <- coxph(f, data = msexp, method = ties)
  nd <- expand.covs(newbase(), covs, longnames = FALSE)
  nd$strata <- 1:3
  cases[[length(cases) + 1]] <- msf_case(paste0("separate_", ties), cox, nd, ties)
  f2 <- Surv(Tstart, Tstop, status) ~ a1 + a2 + tcd1 + strata(trans)
  cox2 <- coxph(f2, data = msexp, method = ties)
  nd2 <- newbase()
  nd2$strata <- 1:3
  cases[[length(cases) + 1]] <- msf_case(paste0("common_", ties), cox2, nd2, ties)
}
# Common coefficients on one shared baseline: the three hazards are one.
msexp$one <- 1
cox3 <- coxph(Surv(Tstart, Tstop, status) ~ a1 + a2 + tcd1 + strata(one), data = msexp,
              method = "efron")
nd3 <- newbase()
nd3$strata <- 1
shared_msf <- tryCatch(msf_case("common_shared", cox3, nd3, "efron"), error = function(e) NULL)
if (!is.null(shared_msf)) cases[[length(cases) + 1]] <- shared_msf

cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  mstate = as.character(packageVersion("mstate")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  covariates = covs,
  data = list(id = ms$id, start = ms$t1, stop = ms$t2, status = ms$st, istate = ms$is,
              x = unname(as.matrix(ms[, covs]))),
  fits = fits, msfit = cases
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
