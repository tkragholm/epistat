# Reference values for epistat_did::estimate_etwfe: Wooldridge's extended
# two-way fixed-effects regression as etwfe 0.6.2 fits it (on fixest 0.14.2)
# and the average effects of emfx(type = "simple", "group", "calendar",
# "event") with their delta-method standard errors (marginaleffects 1.0.0).
# Cases: not-yet-treated controls with a never-treated cohort, a covariate
# and cohort and period effects, with fixest's default iid covariance;
# never-treated controls with two covariates, unit effects and unit clusters;
# and no never-treated cohort, so that the last cohort is the control and
# the periods from its treatment on are removed, with prior weights and the
# heteroskedasticity-robust covariance. The never treated are NA for the Rust
# side and 0 for etwfe.
#
# For each case the script records, besides the packages' numbers, the
# fixed-effects demeaning error of fixest (from exact projections, lm.fit),
# the smallest positive eigenvalue of D'WD, and for each emfx table the
# analytic Jacobian J of the averages, J V J', and the largest sum of the
# absolute terms of one prediction, from which the tests bound the error of
# marginaleffects' forward differences.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript generate_etwfe_reference.R > etwfe_ref.json
suppressMessages({
  library(fixest)
  library(etwfe)
  library(marginaleffects)
  library(jsonlite)
})
stopifnot(packageVersion("fixest") == "0.14.2", packageVersion("etwfe") == "0.6.2",
          packageVersion("marginaleffects") == "1.0.0", packageVersion("jsonlite") == "2.0.0")
set.seed(20261127)
setFixest_nthreads(1)
TOL <- 1e-11
vec <- function(m) as.vector(t(m))

panel <- function(G = 150, T = 8, never = TRUE, drop = 0.1) {
  d <- expand.grid(t = 1:T, id = 0:(G - 1))
  d <- d[runif(nrow(d)) > drop, ]
  g <- sample(c(3, 5, 6, if (never) 0), G, TRUE)
  d$g <- g[d$id + 1]
  u <- rnorm(G); tt <- rnorm(T)
  d$x1 <- rnorm(nrow(d)) + 0.4 * u[d$id + 1]
  d$x2 <- runif(nrow(d)) + 0.2 * d$t
  treat <- d$g > 0 & d$t >= d$g
  eff <- ifelse(treat, 1 + 0.3 * (d$t - d$g) + 0.5 * d$x1 * (d$g == 3), 0)
  d$y <- u[d$id + 1] + tt[d$t] + 0.5 * d$x1 - 0.3 * d$x2 + eff + rnorm(nrow(d))
  d$w <- runif(nrow(d), 0.5, 2)
  d
}

exact_error <- function(m, data, fes) {
  kept <- obs(m)
  V <- cbind(model.matrix(m), y = data$y[kept])
  D <- do.call(cbind, lapply(fes, function(f) model.matrix(as.formula(paste("~ factor(", f, ") - 1")), data[kept, ])))
  ww <- if (is.null(m$weights)) rep(1, length(kept)) else m$weights
  ex <- apply(V, 2, function(v) lm.fit(D * sqrt(ww), v * sqrt(ww))$residuals / sqrt(ww))
  dm <- demean(V, data[kept, fes], weights = ww, tol = TOL, iter = 100000, notes = FALSE)
  scale <- apply(V, 2, function(v) { vc <- v - weighted.mean(v, ww); sqrt(sum(ww * vc^2) / sum(ww)) })
  ev <- eigen(crossprod(D * sqrt(ww)), symmetric = TRUE, only.values = TRUE)$values
  list(error = max(sweep(abs(dm - ex), 2, scale, "/")), lambda_plus = min(ev[ev > 1e-9 * max(ev)]))
}

# The analytic Jacobian of an emfx table: the mean over the rows of each
# group of the model matrix with the treatment on minus with it off.
emfx_table <- function(m, type) {
  e <- emfx(m, type)
  key <- switch(type, simple = NULL, group = "g", calendar = "t", event = "event")
  out <- list(key = if (is.null(key)) NULL else I(e[[key]]),
              estimate = I(e$estimate), se = I(e$std.error))
  dat <- as.data.frame(eval(m$call$data, m$call_env))
  a <- attr(m, "etwfe")
  if (a$cgroup == "never") {
    dat <- dat[dat$g != a$gref, ]
    if (type != "event") dat <- dat[dat$.Dtreat & dat$t >= dat$g, ]
  } else {
    dat <- dat[!is.na(dat$.Dtreat) & dat$.Dtreat, ]
  }
  hi <- dat; hi$.Dtreat <- TRUE
  lo <- dat; lo$.Dtreat <- FALSE
  b <- coef(m)
  # model.matrix on new data leaves out the levels that are not in it.
  full <- function(nd) {
    X <- model.matrix(m, data = nd, type = "rhs")
    out <- matrix(0, nrow(X), length(b), dimnames = list(NULL, names(b)))
    common <- intersect(colnames(X), names(b))
    out[, common] <- X[, common]
    out
  }
  X1 <- full(hi)
  X0 <- full(lo)
  fe <- fixef(m)
  fes <- names(fe)
  fe_abs <- Reduce(`+`, lapply(fes, function(f) abs(fe[[f]][as.character(dat[[f]])])))
  terms <- pmax(rowSums(abs(sweep(X1, 2, b, "*"))), rowSums(abs(sweep(X0, 2, b, "*")))) + fe_abs
  groups <- if (is.null(key)) rep(1, nrow(dat)) else if (key == "event") dat$t - dat$g else dat[[key]]
  levels <- if (is.null(key)) 1 else e[[key]]
  J <- t(sapply(levels, function(l) colMeans((X1 - X0)[groups == l, , drop = FALSE])))
  if (length(levels) == 1) J <- matrix(J, nrow = 1)
  V <- vcov(m)
  out$jacobian <- I(vec(J))
  out$se_analytic <- I(sqrt(pmax(rowSums((J %*% V) * J), 0)))
  out$terms <- max(terms)
  out
}

case <- function(name, d, spec, fml, ...) {
  m <- etwfe(fml, tvar = t, gvar = g, data = d, fixef.tol = TOL, fixef.iter = 100000, ...)
  a <- attr(m, "etwfe")
  data <- eval(m$call$data, m$call_env)
  fes <- if (is.null(a$ivar)) c("g", "t") else c("id", "t")
  V <- vcov(m, attr = TRUE)
  Vu <- vcov(m, vcov = ~id, ssc = ssc(K.adj = FALSE, G.adj = FALSE))
  cov_names <- if (is.null(a$ctrls)) NULL else strsplit(a$ctrls, " \\+ ")[[1]]
  list(name = name, spec = spec,
       unit = d$id, time = d$t, first_treated = ifelse(d$g == 0, NA, d$g), outcome = d$y,
       weights = if (isTRUE(spec$weighted)) d$w else NULL,
       covariates = if (is.null(cov_names)) NULL else I(vec(as.matrix(d[, cov_names]))),
       n_covariates = length(cov_names),
       names = I(names(coef(m))), coef = I(unname(coef(m))), vcov = I(vec(unname(V))),
       vcov_unadjusted = I(vec(unname(Vu))), df_k = attr(V, "df.K"), nobs = m$nobs,
       collinear = I(if (is.null(m$collin.var)) character(0) else m$collin.var),
       simple = emfx_table(m, "simple"), group = emfx_table(m, "group"),
       calendar = emfx_table(m, "calendar"), event = emfx_table(m, "event"),
       exact = exact_error(m, data, fes))
}

d1 <- panel()
d2 <- panel()
d3 <- panel(never = FALSE)
cases <- list(
  case("notyet_cohort_iid", d1, list(control = "notyet", effects = "cohort", vcov = "iid"), y ~ x1),
  case("never_unit_cluster", d2, list(control = "never", effects = "unit", vcov = "cluster"),
       y ~ x1 + x2, cgroup = "never", ivar = id, vcov = ~id),
  case("notyet_last_hetero", d3, list(control = "notyet", effects = "cohort", vcov = "hetero",
                                      weighted = TRUE),
       y ~ 1, weights = ~w, vcov = "hetero")
)
cat(toJSON(list(r = R.version.string, fixest = as.character(packageVersion("fixest")),
                etwfe = as.character(packageVersion("etwfe")),
                marginaleffects = as.character(packageVersion("marginaleffects")),
                tolerance = TOL, cases = cases),
           digits = NA, auto_unbox = TRUE, null = "null", na = "null"))
