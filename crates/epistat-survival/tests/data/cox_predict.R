# Predictions from Cox fits by survival 3.8-12: basehaz(), predict.coxph() and
# survfit.coxph(), on the lung data and on a simulated counting-process data
# set with strata, case weights, an offset and clusters, with Breslow and
# Efron ties.
# For each fit: the design matrix, the coefficients, the variance coxph keeps
# (the robust one with a cluster), the means coxph reports, and
#  - basehaz: the cumulative hazard at the event times, centred and not;
#  - predict: lp and risk with se.fit, for reference "strata", "sample" and
#    "zero", on the fitted data and on newdata; expected with se.fit; terms;
#  - survfit: for each row of newdata the curve at the event times with its
#    standard error and confidence limits, for every conf.type.
# predict.coxph has three results that survival does not get right, and the
# file records both what survival returns and the value that follows from the
# definition:
#  - se.fit of type "risk" is se(lp) * sqrt(risk); the delta method gives
#    se(lp) * risk. Both are written.
#  - se.fit of type "expected" for (start, stop] data is sqrt(v2 - v1) with
#    v = varh + dt V dt' at stop and at start, which is the difference of two
#    variances. The variance of the increment is
#    varh2 - varh1 + (dt2 - dt1) V (dt2 - dt1)'. Both are written; the second
#    is computed here from survival:::agsurv.
#  - for the fitted data, predict(type = "expected") is the event indicator
#    less the martingale residual. For a row with an event at a time shared
#    with other events, under Efron ties, that is the hazard up to the time
#    plus the part (1 - j / d) of the hazard at the time, where the newdata
#    form is the cumulative hazard at the time times the risk. fit_definition
#    is the second for every row, from survival:::agsurv; fit is the first.
#  - without se.fit and newdata, lp keeps the raw offset where with se.fit it
#    is offset - mean(offset). Only the second is compared.
# The versions of R, survival and jsonlite are in the file. Numbers are written
# with 17 significant digits.
# Run: Rscript cox_predict.R > cox_predict.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)

# ---- data
n <- 260
x1 <- round(rnorm(n), 3)
x2 <- round(runif(n, 0, 2), 3)
x3 <- rbinom(n, 1, 0.45)
g <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
stratum <- sample(0:2, n, replace = TRUE)
stop <- pmax(ceiling(rexp(n, 0.08 * exp(0.4 * x1 - 0.5 * x3))), 2)
start <- pmin(sample(0:6, n, replace = TRUE), stop - 1)
event <- rbinom(n, 1, 0.75)
w <- round(runif(n, 0.5, 3), 2)
id <- sample(rep(1:60, length.out = n))
off <- round(rnorm(n, 0.5, 0.6), 3)
sim <- data.frame(start, stop, event, x1, x2, x3, g, stratum, w, id, off)

l <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog", "wt.loss")])
lw <- round(runif(nrow(l), 0.5, 2), 2)
lg <- factor(c("low", "mid", "high")[1 + (l$ph.ecog >= 1) + (l$ph.ecog >= 2)],
             levels = c("low", "mid", "high"))
lungd <- data.frame(start = 0, stop = l$time, event = as.integer(l$status == 2),
                    x1 = l$age, x2 = l$wt.loss, x3 = as.integer(l$sex == 2),
                    g = lg, stratum = l$sex, w = lw, id = seq_len(nrow(l)),
                    off = 0)

sim_new <- data.frame(
  x1 = c(-0.5, 0.3, 1.2, 0, 0.8), x2 = c(0.4, 1.5, 0.9, 1, 0.2),
  x3 = c(0, 1, 1, 0, 1), g = factor(c("a", "b", "c", "b", "a"), levels = levels(g)),
  stratum = c(0, 1, 2, 1, 0), start = c(0, 3, 6, 2, 9),
  stop = c(5, 10, 20, 8, 30), event = 1, off = c(0.2, -0.3, 0, 0.5, 0.1))
lung_new <- data.frame(
  x1 = c(50, 62, 70, 58), x2 = c(0, 10, -5, 20), x3 = c(0, 1, 1, 0),
  g = factor(c("low", "mid", "high", "mid"), levels = levels(lg)),
  stratum = c(1, 2, 1, 2), start = 0, stop = c(100, 300, 500, 800), event = 1,
  off = 0)

# ---- helpers
# expected with the variance of the increment, from survival's own pieces
expected_fix <- function(fit, d, nd, strata_name, has_strata) {
  y <- fit$y
  X <- model.matrix(fit)
  wt <- if (is.null(fit$weights)) rep(1, nrow(X)) else fit$weights
  # coxph keeps no strata in the fit, so they come from the data
  ostrat <- if (!has_strata) rep(1L, nrow(X)) else as.integer(factor(d[[strata_name]]))
  nstrat <- if (!has_strata) rep(1L, nrow(nd)) else as.integer(factor(nd[[strata_name]], levels = sort(unique(d[[strata_name]]))))
  risk <- exp(fit$linear.predictors)
  Xc <- X - rep(fit$means, each = nrow(X))
  newXc <- model.matrix(delete.response(terms(fit)), nd)
  newXc <- newXc[, names(coef(fit)), drop = FALSE] - rep(fit$means, each = nrow(nd))
  newrisk <- exp(c(newXc %*% fit$coefficients))
  st <- survtype <- ifelse(fit$method == "efron", 3, 2)
  se <- double(nrow(nd))
  se_fit <- double(nrow(X))
  exp_fit <- double(nrow(X))
  one <- function(afit, xc, h, t1, t2, cnt) {
    xbar <- apply(afit$xbar, 2, cumsum)
    j1 <- findInterval(t1, afit$time); j2 <- findInterval(t2, afit$time)
    H1 <- c(0, afit$cumhaz)[j1 + 1]; H2 <- c(0, afit$cumhaz)[j2 + 1]
    v1 <- c(0, cumsum(afit$varhaz))[j1 + 1]; v2 <- c(0, cumsum(afit$varhaz))[j2 + 1]
    X1 <- rbind(0, xbar)[j1 + 1, , drop = FALSE]; X2 <- rbind(0, xbar)[j2 + 1, , drop = FALSE]
    d1 <- H1 * xc - X1; d2 <- H2 * xc - X2
    dd <- d2 - d1
    sqrt((v2 - v1) + rowSums((dd %*% fit$var) * dd)) * h
  }
  for (i in unique(ostrat)) {
    indx <- which(ostrat == i)
    afit <- survival:::agsurv(y[indx, , drop = FALSE], Xc[indx, , drop = FALSE], wt[indx], risk[indx], st, st)
    t1 <- if (ncol(y) == 3) y[indx, 1] else rep(0, length(indx))
    jj1 <- findInterval(if (ncol(y) == 3) y[indx, 1] else rep(0, length(indx)), afit$time)
    jj2 <- findInterval(y[indx, ncol(y) - 1], afit$time)
    exp_fit[indx] <- (c(0, afit$cumhaz)[jj2 + 1] - c(0, afit$cumhaz)[jj1 + 1]) * risk[indx]
    se_fit[indx] <- one(afit, Xc[indx, , drop = FALSE], risk[indx], t1, y[indx, ncol(y) - 1])
    k <- which(nstrat == match(i, sort(unique(ostrat))))
    if (length(k)) {
      t1n <- if (ncol(y) == 3) nd$start[k] else rep(0, length(k))
      se[k] <- one(afit, newXc[k, , drop = FALSE], newrisk[k], t1n, nd$stop[k])
    }
  }
  list(fitted = se_fit, new = se, expected = exp_fit)
}

specs <- list(
  list(name = "lung_breslow", d = "lung", terms = "x1 + x2 + x3", ties = "breslow"),
  list(name = "lung_efron_weights", d = "lung", terms = "x1 + x2 + x3", ties = "efron", weights = TRUE),
  list(name = "lung_efron_strata_factor", d = "lung", terms = "x1 + x2 + g", ties = "efron", strata = TRUE),
  list(name = "sim_efron_counting", d = "sim", terms = "x1 + x2 + x3", ties = "efron", entry = TRUE),
  list(name = "sim_breslow_counting_strata_weights", d = "sim", terms = "x1 + x2 + x3", ties = "breslow", entry = TRUE, strata = TRUE, weights = TRUE),
  list(name = "sim_efron_counting_strata_weights_factor", d = "sim", terms = "x1 + x2 + g + x3", ties = "efron", entry = TRUE, strata = TRUE, weights = TRUE, cluster = TRUE),
  list(name = "sim_efron_strata_weights_nostart", d = "sim", terms = "x1 + x2 + x3", ties = "efron", strata = TRUE, weights = TRUE),
  list(name = "sim_efron_counting_strata_offset", d = "sim", terms = "x1 + x3", ties = "efron", entry = TRUE, strata = TRUE, offset = TRUE),
  list(name = "sim_breslow_strata_offset_weights", d = "sim", terms = "x1 + x2", ties = "breslow", strata = TRUE, offset = TRUE, weights = TRUE)
)

run <- function(s) {
  d <- if (s$d == "lung") lungd else sim
  nd <- if (s$d == "lung") lung_new else sim_new
  resp <- if (isTRUE(s$entry)) "Surv(start, stop, event)" else "Surv(stop, event)"
  rhs <- s$terms
  if (isTRUE(s$strata)) rhs <- paste(rhs, "+ strata(stratum)")
  if (isTRUE(s$cluster)) rhs <- paste(rhs, "+ cluster(id)")
  if (isTRUE(s$offset)) rhs <- paste(rhs, "+ offset(off)")
  f <- as.formula(paste(resp, "~", rhs))
  args <- list(formula = f, data = d, ties = s$ties)
  if (isTRUE(s$weights)) args$weights <- d$w
  fit <- do.call(coxph, args)
  X <- model.matrix(fit)
  p <- ncol(X)
  cn <- colnames(X)
  newX <- model.matrix(as.formula(paste("~", s$terms)), nd)[, -1, drop = FALSE]
  stopifnot(identical(colnames(newX), cn))
  has_strata <- isTRUE(s$strata)
  has_off <- isTRUE(s$offset)
  nobs <- nrow(d)
  # the term groups: the columns of each term of the formula
  assign <- fit$assign
  assign <- assign[!grepl("strata|cluster", names(assign))]
  out <- list(
    name = s$name, ties = s$ties, entry = isTRUE(s$entry), strata = has_strata,
    weights = isTRUE(s$weights), cluster = isTRUE(s$cluster), offset = has_off,
    columns = cn, x = X, coefficients = unname(coef(fit)), var = fit$var,
    naive_var = if (isTRUE(s$cluster)) fit$naive.var else fit$var,
    robust = isTRUE(s$cluster),
    means = unname(fit$means), groups = unname(lapply(assign, function(a) I(a - 1L))),
    start = d$start, stop = d$stop, event = d$event, stratum = d$stratum,
    weight = d$w, offset_values = d$off,
    new_x = newX, new_stratum = nd$stratum, new_start = nd$start,
    new_stop = nd$stop, new_offset = nd$off
  )
  # basehaz
  sf <- survfit(fit, se.fit = FALSE, censor = FALSE)
  zc <- ifelse(is.na(coef(fit)), 0, coef(fit))
  out$basehaz <- list(
    time = sf$time, n_event = sf$n.event, n_risk = sf$n.risk,
    strata = I(if (is.null(sf$strata)) length(sf$time) else unname(sf$strata)),
    centered = sf$cumhaz, uncentered = sf$cumhaz * exp(-sum(fit$means * zc)))
  # predict
  pr <- list()
  for (ref in c("strata", "sample", "zero")) {
    for (ty in c("lp", "risk")) {
      a <- predict(fit, type = ty, se.fit = TRUE, reference = ref)
      b <- predict(fit, newdata = nd, type = ty, se.fit = TRUE, reference = ref)
      pr[[paste(ty, ref, sep = "_")]] <- list(fit = a$fit, se = a$se.fit,
                                              new_fit = b$fit, new_se = b$se.fit)
    }
  }
  out$predict <- pr
  # expected: survival's answer and the one from the definition
  ex <- predict(fit, type = "expected", se.fit = TRUE)
  ex_plain <- predict(fit, type = "expected")
  fix <- expected_fix(fit, d, nd, "stratum", has_strata)
  ex_new <- list(fit = NULL, se = NULL)
  if (!has_off) {
    en <- predict(fit, newdata = nd, type = "expected", se.fit = TRUE)
    ex_new <- list(fit = en$fit, se = en$se.fit)
  } else {
    # survival adds the offset of newdata to the risk where it should multiply
    ex_new <- list(fit = NULL, se = NULL)
  }
  sv <- predict(fit, type = "survival", se.fit = TRUE)
  out$expected <- list(fit = ex$fit, se_r = ex$se.fit, fit_plain = ex_plain,
                       se_definition = fix$fitted, fit_definition = fix$expected, new_fit = ex_new$fit,
                       new_se_r = ex_new$se, new_se_definition = if (has_off) NULL else fix$new,
                       survival = sv$fit, survival_se = sv$se.fit)
  # terms
  tm <- predict(fit, type = "terms", se.fit = TRUE)
  tmn <- predict(fit, newdata = nd, type = "terms", se.fit = TRUE)
  out$terms <- list(fit = unname(tm$fit), se = unname(tm$se.fit),
                    constant = attr(tm$fit, "constant"),
                    new_fit = unname(tmn$fit), new_se = unname(tmn$se.fit),
                    term_names = colnames(tm$fit))
  # survfit
  curves <- list()
  cts <- c("log", "log-log", "plain", "logit", "arcsin")
  for (i in seq_len(nrow(nd))) {
    ci <- list()
    for (ct in cts) {
      sfi <- survfit(fit, newdata = nd[i, , drop = FALSE], censor = FALSE, conf.type = ct)
      ci[[ct]] <- list(lower = sfi$lower, upper = sfi$upper)
      if (ct == "log") {
        base <- list(time = sfi$time, n_risk = sfi$n.risk, n_event = sfi$n.event,
                     surv = sfi$surv, cumhaz = sfi$cumhaz, std_err = sfi$std.err)
      }
    }
    sfi2 <- survfit(fit, newdata = nd[i, , drop = FALSE], censor = FALSE,
                    conf.type = "log", conf.int = 0.9)
    base$ci90 <- list(lower = sfi2$lower, upper = sfi2$upper)
    base$ci <- ci
    curves[[i]] <- base
  }
  out$survfit <- curves
  # the other cumulative hazard type
  alt <- if (s$ties == "efron") 1 else 2
  alt_curves <- list()
  for (i in seq_len(nrow(nd))) {
    sfi <- survfit(fit, newdata = nd[i, , drop = FALSE], censor = FALSE, ctype = alt)
    alt_curves[[i]] <- list(time = sfi$time, surv = sfi$surv, cumhaz = sfi$cumhaz,
                            std_err = sfi$std.err, lower = sfi$lower, upper = sfi$upper)
  }
  out$survfit_alt <- list(ctype = alt, curves = alt_curves)
  # the default curve, at the means
  sfm <- survfit(fit, censor = FALSE)
  out$survfit_means <- list(time = sfm$time, surv = sfm$surv, cumhaz = sfm$cumhaz,
                            std_err = sfm$std.err, lower = sfm$lower, upper = sfm$upper,
                            strata = I(if (is.null(sfm$strata)) length(sfm$time) else unname(sfm$strata)))
  out
}

res <- list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  fits = lapply(specs, run)
)
cat(toJSON(res, digits = I(17), na = "null", auto_unbox = TRUE, null = "null"))
