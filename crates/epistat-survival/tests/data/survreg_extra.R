# survreg options added after the first set of fits: offset() terms, the t
# distribution, robust and clustered variance, the dfbeta and dfbetas
# residuals (also collapsed by cluster and weighted) and predict(type =
# "terms"), all on the lung data and on simulated right-, left- and
# interval-censored data.
#
# For each fit the file holds what survreg.R holds (design, coefficients, log
# scales, model-based variance, log likelihoods, iterations, linear
# predictors, predictions on the first 25 rows, residuals) and, where they
# apply, the robust variance, the dfbeta and dfbetas residuals and the terms.
#
# Three results of R 4.6.2 with survival 3.8-12 are wrong, and the file holds
# the value computed from R's own pieces next to R's value, so that the tests
# show both:
# 1. residuals.survreg has the wrong sign for the derivative in the log scale
#    of an interval-censored row (ds is tdenom * (z2 f(z2) - z f(z)), where it
#    must be the negative of that). It reaches dfbeta, and so the robust
#    variance, through the log scale whenever the scale is estimated. The
#    "corrected" values are score %*% vv computed here from
#    residuals(type = "matrix") with that sign put right, for the rows of
#    status 3.
# 2. predict.survreg with newdata takes the offset as 0 (and leaves the offset
#    out of a quantile), where the fitted data have it in the linear
#    predictor. "lp_new" and "quantile_new" are R's newdata results, "lp" and
#    "quantile" those of the fitted data.
#    Where the robust variance is affected, predictions from it are too:
#    "lp_se_corrected" is the standard error of the linear predictor from the
#    corrected robust variance, and the other standard errors are not compared.
# 3. predict.survreg(type = "terms", se.fit = TRUE) computes the variance of a
#    term as (x * beta) V (x * beta)', with the coefficients inside, where the
#    delta method gives x V x'. "terms_se_delta" is x V x' from R's x (centred
#    on fit$means) and fit$var.
# 4. The deviance function of the t distribution is wrong for an
#    interval-censored row twice: its saturated log likelihood is
#    log(1 - 2 pt(width / 2)), the log of a negative number, so deviance
#    residuals are NaN, and its centre is rowMeans(y) with the status column
#    in y, which is (lower + upper + 3) / 3 where the other distributions use
#    (lower + upper) / 2, so response residuals are wrong. The tests leave
#    those rows out and show the difference.
#
# The versions of R, survival and jsonlite are in the file. Numbers are
# written with 17 significant digits.
# Run: Rscript survreg_extra.R > survreg_extra.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261121)

lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
lung2$event <- as.numeric(lung2$status == 2)
lung2$stratum <- lung2$sex - 1
lung2$weight <- round(runif(nrow(lung2), 0.5, 3), 2)
lung2$off <- round(rnorm(nrow(lung2), 0, 0.25), 3)
lung2$cluster <- sample(seq(10, 400, by = 10), nrow(lung2), replace = TRUE)
rownames(lung2) <- NULL

sim_n <- 240
sim_cov <- function() {
  data.frame(
    x1 = round(rnorm(sim_n), 3), x2 = rbinom(sim_n, 1, 0.4),
    stratum = sample(0:2, sim_n, replace = TRUE),
    weight = round(runif(sim_n, 0.5, 3), 2),
    off = round(rnorm(sim_n, 0, 0.25), 3),
    cluster = sample(seq(5, 300, by = 5), sim_n, replace = TRUE)
  )
}
sim_time <- function(d) {
  sigma <- c(0.5, 0.8, 1.2)[d$stratum + 1]
  round(exp(1 + 0.5 * d$x1 - 0.4 * d$x2 + d$off + sigma * log(rexp(sim_n))), 3) + 0.001
}
right <- sim_cov()
tt <- sim_time(right)
cc <- round(rexp(sim_n, 0.15), 3) + 0.001
right$time <- pmin(tt, cc)
right$event <- as.numeric(tt <= cc)

left <- sim_cov()
tt <- sim_time(left)
lim <- round(runif(sim_n, 0.3, 2), 3)
left$time <- pmax(tt, lim)
left$event <- as.numeric(tt >= lim)

interval <- sim_cov()
tt <- sim_time(interval)
k <- pmin(ceiling(tt), 10)
lower <- k - 1
upper <- k
status <- rep(3, sim_n)
status[k == 1] <- 2
status[tt > 9] <- 0
exact <- runif(sim_n) < 0.1
status[exact] <- 1
interval$time <- ifelse(status == 0, 9, ifelse(status == 2, upper, ifelse(status == 1, tt, lower)))
interval$time2 <- ifelse(status == 3, upper, NA)
interval$status <- status

datasets <- list(lung = lung2, right = right, left = left, interval = interval)
surv_of <- list(
  lung = "Surv(time, event)", right = "Surv(time, event)",
  left = "Surv(time, event, type = 'left')",
  interval = "Surv(time, time2, status, type = 'interval')"
)

specs <- list()
add <- function(name, data, dist, rhs, strata = FALSE, weights = FALSE, scale = 0,
                offset = FALSE, robust = "none", parms = 0, terms = FALSE) {
  specs[[length(specs) + 1]] <<- list(name = name, data = data, dist = dist, rhs = rhs,
    strata = strata, weights = weights, scale = scale, offset = offset,
    robust = robust, parms = parms, terms = terms)
}
# offsets
add("lung_weibull_offset", "lung", "weibull", "age + sex + factor(ph.ecog)", offset = TRUE, terms = TRUE)
add("lung_lognormal_offset_strata", "lung", "lognormal", "age + ph.ecog", strata = TRUE, offset = TRUE)
add("lung_weibull_offset_null", "lung", "weibull", "1", offset = TRUE)
add("lung_weibull_offset_alias", "lung", "weibull", "age + sex + ph.ecog + I(2 * age)", offset = TRUE)
add("right_gaussian_offset", "right", "gaussian", "x1 + x2", offset = TRUE)
add("right_weibull_offset_all", "right", "weibull", "x1 + x2", strata = TRUE, weights = TRUE, offset = TRUE)
add("left_lognormal_offset", "left", "lognormal", "x1 + x2", offset = TRUE)
add("interval_weibull_offset", "interval", "weibull", "x1 + x2", offset = TRUE)
add("interval_logistic_offset_fixed", "interval", "logistic", "x1 + x2", scale = 1.1, offset = TRUE)
# the t distribution
add("lung_t4", "lung", "t", "age + sex + ph.ecog", parms = 4)
add("lung_t6_strata", "lung", "t", "age + ph.ecog", strata = TRUE, parms = 6)
add("lung_t3_weights", "lung", "t", "age + sex + ph.ecog", weights = TRUE, parms = 3)
add("lung_t4_fixed", "lung", "t", "age + sex", scale = 250, parms = 4)
add("right_t3", "right", "t", "x1 + x2", parms = 3)
add("right_t4_offset", "right", "t", "x1 + x2", offset = TRUE, parms = 4)
add("left_t5", "left", "t", "x1 + x2", parms = 5)
add("interval_t4", "interval", "t", "x1 + x2", parms = 4)
add("interval_t2.5", "interval", "t", "x1 + x2", parms = 2.5)
# robust and clustered variance
add("lung_weibull_robust", "lung", "weibull", "age + sex + factor(ph.ecog)", robust = "robust", terms = TRUE)
add("right_weibull_robust_weights", "right", "weibull", "x1 + x2", weights = TRUE, robust = "robust")
add("lung_lognormal_cluster", "lung", "lognormal", "age + sex + ph.ecog", robust = "cluster")
add("left_weibull_cluster", "left", "weibull", "x1 + x2", robust = "cluster")
add("interval_weibull_robust", "interval", "weibull", "x1 + x2", robust = "robust")
add("interval_logistic_fixed_robust", "interval", "logistic", "x1 + x2", scale = 1.1, robust = "robust")
add("interval_weibull_cluster_strata", "interval", "weibull", "x1 + x2", strata = TRUE, weights = TRUE, robust = "cluster")
add("right_weibull_cluster_all", "right", "weibull", "x1 + x2", strata = TRUE, weights = TRUE, robust = "cluster")
add("lung_t4_robust", "lung", "t", "age + sex + ph.ecog", robust = "robust", parms = 4)
add("lung_weibull_offset_cluster", "lung", "weibull", "age + sex + ph.ecog", offset = TRUE, robust = "cluster")
add("lung_weibull_alias_robust", "lung", "weibull", "age + sex + ph.ecog + I(2 * age)", robust = "robust")
# plain fits to carry dfbeta, dfbetas and terms
add("lung_weibull_plain", "lung", "weibull", "age + sex + ph.ecog", weights = TRUE, terms = TRUE)
add("interval_gaussian_plain", "interval", "gaussian", "x1 + x2", strata = TRUE)

residual_types <- c("response", "deviance", "working", "ldcase", "ldresp", "ldshape")
na_matrix <- function(m) unname(as.matrix(m))
fits <- list()
for (s in specs) {
  d <- datasets[[s$data]]
  rhs_full <- s$rhs
  if (s$strata) rhs_full <- paste(rhs_full, "+ strata(stratum)")
  if (s$offset) rhs_full <- paste(rhs_full, "+ offset(off)")
  if (s$robust == "cluster") rhs_full <- paste(rhs_full, "+ cluster(cluster)")
  f <- as.formula(paste(surv_of[[s$data]], "~", rhs_full))
  args <- list(formula = f, data = quote(d), dist = s$dist, x = TRUE, model = TRUE)
  if (s$weights) args$weights <- quote(weight)
  if (s$scale > 0) args$scale <- s$scale
  if (s$parms > 0) args$parms <- s$parms
  if (s$robust == "robust") args$robust <- TRUE
  fit <- do.call(survreg, args)
  p <- length(coef(fit))
  ok <- !is.na(coef(fit))
  vv <- if (is.null(fit$naive.var)) fit$var else fit$naive.var
  nscale <- ncol(vv) - p
  X <- model.matrix(fit)
  tmat <- diag(1, p + nscale)
  if (all(X[, 1] == 1) && ncol(X) > 1) {
    okay <- apply(X, 2, function(z) all(z == 0 | z == 1))
    if (!all(okay)) {
      center <- ifelse(okay, 0, colMeans(X))
      stdev <- ifelse(okay, 1, apply(X, 2, sd))
      tmat <- diag(c(1 / stdev, rep(1, nscale)))
      tmat[1, 2:p] <- -center[2:p] / stdev[2:p]
    }
  }
  keep <- c(ok, rep(TRUE, nscale))
  vs <- solve(tmat) %*% vv %*% t(solve(tmat))
  condition <- kappa(vs[keep, keep, drop = FALSE], exact = TRUE)
  idx <- 1:25
  nd <- d[idx, ]
  sel <- function(r) {
    if (is.list(r)) list(fit = r$fit, se.fit = r$se.fit) else r
  }
  # Predictions of the fitted data (offset included), first 25 rows.
  pr <- function(type, ...) {
    r <- predict(fit, type = type, se.fit = TRUE, ...)
    list(fit = if (is.matrix(r$fit)) r$fit[idx, , drop = FALSE] else r$fit[idx],
         se.fit = if (is.matrix(r$se.fit)) r$se.fit[idx, , drop = FALSE] else r$se.fit[idx])
  }
  p_lp <- pr("lp")
  p_resp <- pr("response")
  p_q <- pr("quantile", p = c(0.1, 0.5, 0.9))
  res <- lapply(residual_types, function(t) as.numeric(residuals(fit, type = t)))
  names(res) <- residual_types

  # dfbeta
  ytype <- attr(fit$y, "type")
  status3 <- if (ytype == "interval") fit$y[, 3] == 3 else rep(FALSE, nrow(X))
  defect <- any(status3) && nscale > 0
  strata_index <- if (s$strata) d$stratum + 1 else rep(1, nrow(X))
  hand <- function() {
    deriv <- residuals(fit, type = "matrix")
    if (defect) deriv[status3, 4] <- -deriv[status3, 4]
    score <- deriv[, 2] * X
    if (nscale > 1) {
      d4 <- matrix(0, nrow(X), nscale)
      d4[cbind(1:nrow(X), strata_index)] <- deriv[, 4]
      score <- cbind(score, d4)
    } else if (nscale == 1) score <- cbind(score, deriv[, 4])
    score %*% vv
  }
  corrected <- hand()
  w <- if (s$weights) d$weight else rep(1, nrow(X))
  clusters <- d$cluster
  native <- list(
    dfbeta = residuals(fit, type = "dfbeta"),
    dfbetas = residuals(fit, type = "dfbetas"),
    weighted = residuals(fit, type = "dfbeta", weighted = TRUE),
    collapsed = residuals(fit, type = "dfbeta", weighted = TRUE, collapse = clusters),
    collapsed_s = residuals(fit, type = "dfbetas", collapse = clusters)
  )
  cor <- list(
    dfbeta = corrected,
    dfbetas = corrected * rep(1 / sqrt(diag(vv)), each = nrow(corrected)),
    weighted = corrected * w
  )
  cor$collapsed <- drop(rowsum(cor$weighted, clusters))
  cor$collapsed_s <- drop(rowsum(cor$dfbetas, clusters))
  # Without the defect R's own values must be the corrected ones.
  if (!defect) {
    for (nm in names(native)) {
      stopifnot(isTRUE(all.equal(unname(as.matrix(native[[nm]])), unname(as.matrix(cor[[nm]])),
                                 tolerance = 1e-8)))
    }
  }
  robust_native <- NULL
  robust_corrected <- NULL
  n_clusters <- NULL
  if (s$robust != "none") {
    robust_native <- na_matrix(fit$var)
    grp <- if (s$robust == "cluster") clusters else seq_len(nrow(X))
    robust_corrected <- crossprod(rowsum(cor$weighted, grp))
    n_clusters <- length(unique(grp))
    if (!defect) stopifnot(isTRUE(all.equal(robust_native, unname(robust_corrected), tolerance = 1e-8)))
    robust_corrected <- unname(robust_corrected)
  }

  # predictions from newdata, where R drops the offset
  lp_new <- NULL
  quantile_new <- NULL
  if (s$offset) {
    lp_new <- as.numeric(predict(fit, newdata = nd, type = "lp"))
    quantile_new <- predict(fit, newdata = nd, type = "quantile", p = c(0.1, 0.5, 0.9))
    if (is.matrix(quantile_new)) quantile_new <- na_matrix(quantile_new) else quantile_new <- na_matrix(matrix(quantile_new, ncol = 3))
  }

  # Standard errors of the linear predictor from the corrected robust
  # variance, where R's own come from the wrong one.
  lp_se_corrected <- NULL
  if (!is.null(robust_corrected) && defect) {
    Xs <- X[idx, , drop = FALSE]
    Vc <- robust_corrected[seq_len(p), seq_len(p), drop = FALSE]
    lp_se_corrected <- sqrt(rowSums((Xs %*% Vc) * Xs))
  }

  # terms
  terms_out <- NULL
  if (s$terms) {
    asg <- attr(X, "assign")
    nterms <- max(asg)
    groups <- lapply(seq_len(nterms), function(i) I(which(asg == i) - 1))
    tm <- predict(fit, newdata = nd, type = "terms", se.fit = TRUE)
    Xnd <- model.matrix(delete.response(fit$terms), nd, xlev = fit$xlevels)
    Xc <- sweep(Xnd, 2, fit$means)
    V <- fit$var
    se_delta <- matrix(NA_real_, nrow(nd), nterms)
    for (i in seq_len(nterms)) {
      ii <- which(asg == i)
      for (j in seq_len(nrow(nd))) {
        xi <- Xc[j, ii, drop = FALSE]
        se_delta[j, i] <- sqrt(sum(xi %*% V[ii, ii, drop = FALSE] %*% t(xi)))
      }
    }
    terms_out <- list(
      groups = groups, means = as.numeric(fit$means), fit = na_matrix(tm$fit),
      se_r = na_matrix(tm$se.fit), se_delta = se_delta,
      x_new = na_matrix(Xnd)
    )
  }

  fits[[length(fits) + 1]] <- list(
    name = s$name, dataset = s$data, distribution = s$dist, parms = s$parms,
    use_strata = s$strata, use_weights = s$weights, fixed_scale = s$scale,
    use_offset = s$offset, robust = s$robust,
    x = na_matrix(X),
    coefficients = I(unname(coef(fit))),
    log_scale = I(log(unname(fit$scale))),
    variance = na_matrix(vv),
    loglik = fit$loglik[2], loglik_null = fit$loglik[1],
    iterations = fit$iter, df = fit$df,
    linear_predictors = as.numeric(fit$linear.predictors),
    information_condition = condition,
    predict_rows = length(idx),
    lp = as.numeric(p_lp$fit), lp_se = as.numeric(p_lp$se.fit),
    response = as.numeric(p_resp$fit), response_se = as.numeric(p_resp$se.fit),
    quantile = na_matrix(p_q$fit), quantile_se = na_matrix(p_q$se.fit),
    lp_new = lp_new, quantile_new = quantile_new, lp_se_corrected = lp_se_corrected,
    residuals = res,
    interval_scale_defect = defect,
    dfbeta = lapply(native, na_matrix),
    dfbeta_corrected = lapply(cor, na_matrix),
    robust_variance = robust_native,
    robust_variance_corrected = robust_corrected,
    n_clusters = n_clusters,
    terms = terms_out
  )
}
col <- function(d, v) if (v %in% names(d)) d[[v]] else rep(NA_real_, nrow(d))
data_out <- lapply(datasets, function(d) list(
  time = col(d, "time"), event = col(d, "event"), time2 = col(d, "time2"),
  status = col(d, "status"), stratum = col(d, "stratum"), weight = col(d, "weight"),
  off = col(d, "off"), cluster = col(d, "cluster")
))
cat(toJSON(list(
  versions = list(
    r = R.version.string,
    survival = as.character(packageVersion("survival")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  datasets = data_out, fits = fits
), digits = I(17), auto_unbox = TRUE, na = "null", null = "null"))
