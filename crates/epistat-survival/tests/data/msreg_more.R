# Further multi-state references: Greenwood variances, probabilities at a fixed
# horizon, covariances of the probabilities, several covariate profiles and
# profiles that differ between transitions, models with time-varying
# covariates, and weights, by survival 3.8-12 and mstate 0.3.3 on a random 300
# of the patients of mstate's ebmt3 (transplant, platelet recovery, relapse or
# death in an illness-death process; three covariates, times in days).
#
# msfit cases: the null model (coxph with strata(trans) only) with mstate's
# default (Aalen) and vartype = "greenwood" variances; a stacked fit with
# separate coefficients (expand.covs) and one with common coefficients; and a
# stacked fit with separate coefficients and two time-varying covariates, built
# by splitting each row at days 20 and 30 and taking the covariate values at the
# start of each row (a covariate that switches on at day 20 or 30, and the start time times a covariate). Each has probtrans with method "aalen" and "greenwood", forward
# and with direction = "fixedhorizon" (at an event time and between event
# times), and the covariance of the probabilities for some. The covariate
# models have several profiles, one of which differs between transitions
# (newdata with one row for each transition, as in the mstate documentation).
#
# weights: mstate's msfit refuses a weighted model; its message is recorded.
# survfit.coxph supports weights, so the cumulative hazard of each transition and
# its variance are taken from it for a weighted stacked fit.
#
# mstate's probtrans stores the covariance array of a fixed horizon one slice
# off the rows of its result: the slice k + 1 holds the covariance of the matrix
# of row k. The arrays are recorded as returned, with the standard errors of the
# rows, so that the test asserts that shift.
#
# Run: Rscript msreg_more.R > msreg_more.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(mstate)
})
set.seed(20261121)
data(ebmt3)
d <- ebmt3[sort(sample(nrow(ebmt3), 300)), ]
d$a1 <- as.numeric(d$age == "20-40")
d$a2 <- as.numeric(d$age == ">40")
d$tcd1 <- as.numeric(d$tcd == "TCD")
covs <- c("a1", "a2", "tcd1")
tmat <- trans.illdeath()

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
rownames(ms) <- NULL

msd <- msprep(time = c(NA, "prtime", "rfstime"), status = c(NA, "prstat", "rfsstat"),
              data = d, trans = tmat, keep = covs)
msexp <- expand.covs(msd, covs, longnames = FALSE)

# ---- probabilities
pt_out <- function(pt, S, cov = FALSE) {
  out <- list(time = pt[[1]]$time,
              p = lapply(1:S, function(s) unname(as.matrix(pt[[s]][, paste0("pstate", 1:S)]))),
              se = lapply(1:S, function(s) unname(as.matrix(pt[[s]][, paste0("se", 1:S)]))))
  if (cov) {
    a <- pt$varMatrix
    out$varmatrix <- lapply(seq_len(dim(a)[3]), function(i) unname(a[, , i]))
  }
  out
}
configs <- list(
  list(predt = 0, direction = "forward", method = "aalen"),
  list(predt = 200, direction = "forward", method = "greenwood"),
  list(predt = 500, direction = "fixedhorizon", method = "aalen"),
  list(predt = 500.5, direction = "fixedhorizon", method = "greenwood")
)
cov_configs <- list(
  list(predt = 200, direction = "forward", method = "aalen"),
  list(predt = 500, direction = "fixedhorizon", method = "greenwood")
)
all_pt <- function(msf, with_cov, only_first = FALSE) {
  cfg <- if (only_first) configs[c(1, 3)] else configs
  res <- lapply(cfg, function(c) {
    pt <- probtrans(msf, predt = c$predt, direction = c$direction, method = c$method)
    c(c, list(covariance = FALSE), pt_out(pt, 3))
  })
  if (with_cov) {
    res <- c(res, lapply(cov_configs, function(c) {
      pt <- probtrans(msf, predt = c$predt, direction = c$direction, method = c$method,
                      covariance = TRUE)
      c(c, list(covariance = TRUE), pt_out(pt, 3, TRUE))
    }))
  }
  res
}
msf_out <- function(msf) {
  H <- msf$Haz
  vh <- msf$varHaz
  u <- unique(vh[, c("trans1", "trans2")])
  list(time = sort(unique(H$time)),
       haz = lapply(1:3, function(k) H$Haz[H$trans == k]),
       varhaz = lapply(seq_len(nrow(u)), function(j) {
         list(trans1 = u$trans1[j], trans2 = u$trans2[j],
              value = vh$varHaz[vh$trans1 == u$trans1[j] & vh$trans2 == u$trans2[j]])
       }))
}

cases <- list()

# ---- the null model: Aalen and Greenwood variances
for (ties in c("efron", "breslow")) {
  cox0 <- coxph(Surv(Tstart, Tstop, status) ~ strata(trans), data = msd, method = ties)
  for (vt in c("aalen", "greenwood")) {
    msf <- msfit(cox0, trans = tmat, vartype = vt)
    cases[[length(cases) + 1]] <- list(
      name = paste0("null_", vt, "_", ties), kind = "null", ties = ties, vartype = vt,
      profiles = list(c(list(name = "null", values = list()), msf_out(msf),
                        list(probtrans = all_pt(msf, with_cov = (ties == "efron"))))))
  }
}

# ---- covariate models: several profiles, one that differs by transition
newdata_of <- function(rows) {
  # rows: a list of three named vectors, one for each transition
  nd <- data.frame(a1 = sapply(rows, `[[`, "a1"), a2 = sapply(rows, `[[`, "a2"),
                   tcd1 = sapply(rows, `[[`, "tcd1"), trans = 1:3)
  attr(nd, "trans") <- tmat
  class(nd) <- c("msdata", "data.frame")
  nd
}
p1 <- c(a1 = 1, a2 = 0, tcd1 = 1)
p2 <- c(a1 = 0, a2 = 1, tcd1 = 0)
p3 <- c(a1 = 0, a2 = 0, tcd1 = 1)
profile_sets <- list(
  list(name = "p1", rows = list(p1, p1, p1)),
  list(name = "p2", rows = list(p2, p2, p2)),
  list(name = "by_transition", rows = list(p1, p2, p3))
)
for (ties in c("efron", "breslow")) {
  for (kind in c("separate", "common")) {
    if (kind == "separate") {
      f <- as.formula(paste("Surv(Tstart, Tstop, status) ~",
                            paste(c(outer(covs, 1:3, paste, sep = ".")), collapse = " + "),
                            "+ strata(trans)"))
      cox <- coxph(f, data = msexp, method = ties)
    } else {
      cox <- coxph(Surv(Tstart, Tstop, status) ~ a1 + a2 + tcd1 + strata(trans),
                   data = msexp, method = ties)
    }
    profs <- lapply(profile_sets, function(ps) {
      nd <- newdata_of(ps$rows)
      if (kind == "separate") nd <- expand.covs(nd, covs, longnames = FALSE)
      nd$strata <- 1:3
      msf <- msfit(cox, nd, trans = tmat)
      c(list(name = ps$name, values = lapply(ps$rows, unname)), msf_out(msf),
        list(probtrans = all_pt(msf, with_cov = (ps$name == "by_transition" && ties == "efron"),
                                only_first = (ps$name == "p2"))))
    })
    cases[[length(cases) + 1]] <- list(
      name = paste0(kind, "_", ties), kind = kind, ties = ties,
      coefficients = unname(coef(cox)), vcov = unname(cox$var), profiles = profs)
  }
}

# ---- time-varying covariates
cuts <- c(20, 60, 200)
split <- do.call(rbind, lapply(seq_len(nrow(ms)), function(i) {
  r <- ms[i, ]
  inside <- cuts[cuts > r$t1 & cuts < r$t2]
  if (length(inside) == 0) return(r)
  edges <- c(r$t1, inside, r$t2)
  pieces <- r[rep(1, length(edges) - 1), ]
  pieces$t1 <- edges[-length(edges)]
  pieces$t2 <- edges[-1]
  pieces$st[-nrow(pieces)] <- 0
  pieces
}))
# a1tm changes at every piece; slow is 0 before platelet recovery and 1 or 0
# after it, by whether recovery took 30 days or more, so it changes at the
# transition to state 2 and is only in the model of the transition out of it.
split$a1tm <- split$a1 * split$t1 / 365
split$slow <- as.numeric(split$is == 2 & split$t1 >= 30)
tvc <- c("a1", "tcd1", "a1tm", "slow")
trs <- unique(ms[ms$st > 0, c("is", "st")])
trs <- trs[order(trs$is, trs$st), ]
# the terms: covariate (0-based) and the transition (1-based) of each coefficient
tv_terms <- c(lapply(0:2, function(c) lapply(1:3, function(k) list(c, k))), list(list(list(3, 3))))
tv_terms <- unlist(tv_terms, recursive = FALSE)
tv_names <- sapply(tv_terms, function(t) paste(tvc[t[[1]] + 1], t[[2]], sep = "."))
stk <- do.call(rbind, lapply(seq_len(nrow(trs)), function(k) {
  r <- split[split$is == trs$is[k], ]
  out <- data.frame(id = r$id, Tstart = r$t1, Tstop = r$t2, status = as.numeric(r$st == trs$st[k]),
                    trans = k)
  for (nm in tv_names) {
    cv <- sub("\\..*", "", nm)
    kk <- as.integer(sub(".*\\.", "", nm))
    out[[nm]] <- if (kk == k) r[[cv]] else 0
  }
  out
}))
for (ties in c("efron", "breslow")) {
  f <- as.formula(paste("Surv(Tstart, Tstop, status) ~", paste(tv_names, collapse = " + "),
                        "+ strata(trans)"))
  cox <- coxph(f, data = stk, method = ties)
  fm <- model.matrix(cox)
  sc <- colMeans(abs(sweep(fm, 2, colMeans(fm))))
  Dm <- diag(sc, length(sc))
  kap <- kappa(Dm %*% solve(cox$var) %*% Dm, exact = TRUE)
  pv <- list(name = "steps", rows = list(c(a1 = 1, tcd1 = 1, a1tm = 0.4, slow = 1),
                                         c(a1 = 0, tcd1 = 1, a1tm = 0, slow = 1),
                                         c(a1 = 1, tcd1 = 0, a1tm = 0.1, slow = 0)))
  profs <- lapply(list(pv), function(ps) {
    nd <- data.frame(trans = 1:3)
    for (nm in tv_names) {
      cv <- sub("\\..*", "", nm)
      kk <- as.integer(sub(".*\\.", "", nm))
      nd[[nm]] <- sapply(1:3, function(k) if (k == kk) ps$rows[[k]][[cv]] else 0)
    }
    nd$strata <- 1:3
    msf <- msfit(cox, nd, trans = tmat)
    c(list(name = ps$name, values = lapply(ps$rows, unname)), msf_out(msf),
      list(probtrans = all_pt(msf, with_cov = (ties == "efron"))))
  })
  cases[[length(cases) + 1]] <- list(
    name = paste0("tv_", ties), kind = "tv", ties = ties,
    terms = lapply(tv_terms, function(t) list(column = t[[1]], transitions = list(t[[2]] - 1L))),
    coefficients = unname(coef(cox)), vcov = unname(cox$var),
    loglik = cox$loglik[2], iterations = cox$iter, information_condition = kap,
    n_events = cox$nevent, profiles = profs)
}
tvdata <- list(id = split$id, start = split$t1, stop = split$t2, status = split$st,
               istate = split$is, x = unname(as.matrix(split[, tvc])))

# ---- weights
# msprep numbers the patients 1 to n in the order of the data; the weight goes by
# the patient's own id.
msexp$w <- 0.5 + (d$id[msexp$id] %% 4) / 2
f <- Surv(Tstart, Tstop, status) ~ a1.1 + a2.1 + tcd1.1 + a1.2 + a2.2 + tcd1.2 + a1.3 + a2.3 +
  tcd1.3 + strata(trans)
nd <- expand.covs(newdata_of(list(p1, p1, p1)), covs, longnames = FALSE)
nd$strata <- 1:3
refusal <- NULL
weights <- lapply(c("efron", "breslow"), function(ties) {
  wcox <- coxph(f, data = msexp, weights = w, robust = FALSE, method = ties)
  if (is.null(refusal)) {
    refusal <<- tryCatch({ msfit(wcox, nd, trans = tmat); "" },
                         error = function(e) conditionMessage(e))
  }
  wh <- lapply(1:3, function(k) {
    s <- survfit(wcox, newdata = nd[k, ])
    list(time = s$time, cumhaz = s$cumhaz, std_err = s$std.err)
  })
  list(ties = ties, coefficients = unname(coef(wcox)), vcov = unname(wcox$var),
       profile = unname(p1), hazards = wh)
})

cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  mstate = as.character(packageVersion("mstate")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  covariates = covs,
  data = list(id = ms$id, start = ms$t1, stop = ms$t2, status = ms$st, istate = ms$is,
              x = unname(as.matrix(ms[, covs]))),
  tvdata = tvdata, cases = cases, refusal = refusal, weights = weights
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
