# Reference values for epistat_did::estimate_etwfe_glm: the Poisson and
# logit extended two-way regressions as etwfe 0.6.2 fits them
# (etwfe(family = "poisson" / "logit"), which calls fixest 0.14.2's feglm
# with cohort and period dummies and no fixed effects), and the average
# effects of emfx(type = "simple", "group", "calendar", "event") with the
# delta-method standard errors of marginaleffects 1.0.0, whose Jacobian is
# a forward difference in each coefficient.
# Cases: Poisson with not-yet-treated controls, a never-treated cohort, one
# covariate and the default iid covariance; Poisson with never-treated
# controls, two covariates, prior weights and unit clusters; logit with no
# never-treated cohort, so that the last cohort is the control and the
# periods from its treatment on are removed, one covariate, prior weights
# and unit clusters; logit with never-treated controls, one covariate and
# the heteroskedasticity-robust covariance. The never treated are NA for the
# Rust side and 0 for etwfe.
#
# For each case the script records, besides the packages' numbers, what the
# tests need to bound the distance between two fits that each stop at the
# deviance criterion glm.tol: the deviance, the condition number of X'WX, the
# largest x' (X'WX)^-1 x over the rows of the fit and the rows of the
# effects, the diagonal of (X'WX)^-1, and for each coefficient and each emfx
# row the first-order sensitivity of its clustered or robust variance to the
# coefficients (see the test). For each emfx table it records the analytic
# Jacobian J of the averages at R's coefficients, sqrt(J V J'), J (X'WX)^-1
# J', and the sums that bound the error of the forward differences.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript generate_etwfe_glm_reference.R > etwfe_glm_ref.json
suppressMessages({
  library(fixest)
  library(etwfe)
  library(marginaleffects)
  library(jsonlite)
})
stopifnot(packageVersion("fixest") == "0.14.2", packageVersion("etwfe") == "0.6.2",
          packageVersion("marginaleffects") == "1.0.0", packageVersion("jsonlite") == "2.0.0")
set.seed(20261125)
setFixest_nthreads(1)
TOL <- 1e-12
vec <- function(m) as.vector(t(m))

panel <- function(family, G = 300, T = 8, never = TRUE, drop = 0.1) {
  d <- expand.grid(t = 1:T, id = 0:(G - 1))
  d <- d[runif(nrow(d)) > drop, ]
  g <- sample(c(3, 5, 6, if (never) 0), G, TRUE)
  d$g <- g[d$id + 1]
  u <- rnorm(G, sd = 0.4); tt <- rnorm(T, sd = 0.2)
  d$x1 <- rnorm(nrow(d)) + 0.4 * u[d$id + 1]
  d$x2 <- runif(nrow(d)) + 0.1 * d$t
  treat <- d$g > 0 & d$t >= d$g
  eff <- ifelse(treat, 0.3 + 0.05 * (d$t - d$g) + 0.2 * d$x1 * (d$g == 3), 0)
  eta <- u[d$id + 1] + tt[d$t] + 0.3 * d$x1 - 0.2 * d$x2 + eff
  d$y <- if (family == "poisson") rpois(nrow(d), exp(0.5 + eta)) else rbinom(nrow(d), 1, plogis(-0.2 + eta))
  d$w <- runif(nrow(d), 0.5, 2)
  d
}

# The rows of an emfx table, the design with the treatment on and off, and
# the averages' Jacobian and the bounds of its forward differences.
emfx_table <- function(m, type, B, V, groups_of, sens) {
  e <- emfx(m, type, compress = FALSE)
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
  full <- function(nd) {
    X <- model.matrix(m, data = nd, type = "rhs")
    out <- matrix(0, nrow(X), length(b), dimnames = list(NULL, names(b)))
    common <- intersect(colnames(X), names(b))
    out[, common] <- X[, common]
    out
  }
  X1 <- full(hi); X0 <- full(lo)
  e1 <- drop(X1 %*% b); e0 <- drop(X0 %*% b)
  fam <- m$family
  mu1 <- fam$linkinv(e1); mu0 <- fam$linkinv(e0)
  d1 <- fam$mu.eta(e1); d0 <- fam$mu.eta(e0)
  K <- length(b)
  S1 <- rowSums(abs(sweep(X1, 2, b, "*"))); S0 <- rowSums(abs(sweep(X0, 2, b, "*")))
  grp <- if (is.null(key)) rep(1, nrow(dat)) else if (key == "event") dat$t - dat$g else dat[[key]]
  levels <- if (is.null(key)) 1 else e[[key]]
  rows <- lapply(levels, function(l) grp == l)
  colmean <- function(M, r) colMeans(M[r, , drop = FALSE])
  J <- t(sapply(rows, function(r) colmean(d1 * X1 - d0 * X0, r)))
  if (length(levels) == 1) J <- matrix(J, nrow = 1)
  mat <- function(f) { M <- t(sapply(rows, f)); if (length(levels) == 1) matrix(M, nrow = 1) else M }
  out$jacobian <- I(vec(J))
  out$se_analytic <- I(sqrt(pmax(rowSums((J %*% V) * J), 0)))
  out$jbj <- I(rowSums((J %*% B) * J))
  # mean(mu'1 |x1| + mu'0 |x0|), mean(mu'1 x1^2 + mu'0 x0^2) and the largest
  # |x| by column, and mean(mu'1 + mu'0), over each table row's rows.
  out$jabs <- I(vec(mat(function(r) colmean(d1 * abs(X1) + d0 * abs(X0), r))))
  out$curv <- I(vec(mat(function(r) colmean(d1 * X1^2 + d0 * X0^2, r))))
  out$xmax <- I(vec(mat(function(r) apply(pmax(abs(X1[r, , drop = FALSE]), abs(X0[r, , drop = FALSE])), 2, max))))
  out$slope <- I(sapply(rows, function(r) mean(d1[r] + d0[r])))
  # The rounding of one evaluation of an average: each mean from a linear
  # predictor of K terms rounded to (K + 2) eps S and the inverse link, and
  # the sum over the rows.
  out$round <- I(sapply(rows, function(r) mean(mu1[r] * ((K + 2) * S1[r] + 2) + mu0[r] * ((K + 2) * S0[r] + 2)) + sum(r) * mean(mu1[r] + mu0[r])))
  out$n <- I(sapply(rows, sum))
  out$sens <- I(sapply(seq_along(levels), function(l) sens(J[l, ], groups_of$vcov)))
  out$sens_unit <- I(sapply(seq_along(levels), function(l) sens(J[l, ], groups_of$unit)))
  out$h <- max(rowSums((X1 %*% B) * X1), rowSums((X0 %*% B) * X0))
  out
}

case <- function(name, d, spec, fml, ...) {
  m <- etwfe(fml, tvar = t, gvar = g, data = d, family = spec$family, glm.tol = TOL, glm.iter = 100, ...)
  stopifnot(m$convStatus)
  a <- attr(m, "etwfe")
  V <- vcov(m, attr = TRUE)
  Vu <- vcov(m, vcov = ~id, ssc = ssc(K.adj = FALSE, G.adj = FALSE))
  cov_names <- if (is.null(a$ctrls)) NULL else strsplit(a$ctrls, " \\+ ")[[1]]
  kept <- obs(m)
  X <- model.matrix(m)
  B <- solve(m$hessian)
  W <- m$irls_weights
  S <- m$scores
  # The first-order sensitivity of sum_g (c' B S_g)^2 to the coefficients,
  # per unit of max |x' dbeta|: 2 sum_g |a' S_g| (sqrt(c'Bc S_g'BS_g) + 2
  # sum_{i in g} W_i |a' x_i|) with a = B c.
  sens <- function(c, groups) {
    if (is.null(groups)) return(0)
    a <- drop(B %*% c)
    Sg <- rowsum(S, groups, reorder = FALSE)
    Tg <- rowsum(W * abs(drop(X %*% a)), groups, reorder = FALSE)
    Qg <- rowSums((Sg %*% B) * Sg)
    2 * sum(abs(drop(Sg %*% a)) * (sqrt(drop(t(c) %*% B %*% c) * Qg) + 2 * Tg))
  }
  groups_of <- list(vcov = switch(spec$vcov, iid = NULL, hetero = seq_along(kept), cluster = d$id[kept]),
                    unit = d$id[kept])
  K <- length(coef(m))
  tables <- lapply(c(simple = "simple", group = "group", calendar = "calendar", event = "event"),
                   function(type) emfx_table(m, type, B, V, groups_of, sens))
  h_fit <- max(rowSums((X %*% B) * X))
  list(name = name, spec = spec,
       unit = d$id, time = d$t, first_treated = ifelse(d$g == 0, NA, d$g), outcome = d$y,
       weights = if (isTRUE(spec$weighted)) d$w else NULL,
       covariates = if (is.null(cov_names)) NULL else I(vec(as.matrix(d[, cov_names]))),
       n_covariates = length(cov_names),
       names = I(names(coef(m))), coef = I(unname(coef(m))), vcov = I(vec(unname(V))),
       vcov_unadjusted = I(vec(unname(Vu))), df_k = attr(V, "df.K"), nobs = m$nobs,
       deviance = m$deviance, iterations = m$iterations, tolerance = TOL,
       collinear = I(if (is.null(m$collin.var)) character(0) else m$collin.var),
       kappa = kappa(m$hessian, exact = TRUE), b_diag = I(diag(B)),
       sens = I(sapply(seq_len(K), function(j) sens(diag(K)[, j], groups_of$vcov))),
       sens_unit = I(sapply(seq_len(K), function(j) sens(diag(K)[, j], groups_of$unit))),
       h = max(h_fit, sapply(tables, function(t) t$h)),
       simple = tables$simple, group = tables$group, calendar = tables$calendar,
       event = tables$event)
}

d1 <- panel("poisson")
d2 <- panel("poisson")
d3 <- panel("logit", never = FALSE)
d4 <- panel("logit")
cases <- list(
  case("poisson_notyet_iid", d1, list(family = "poisson", control = "notyet", vcov = "iid"), y ~ x1),
  case("poisson_never_cluster", d2, list(family = "poisson", control = "never", vcov = "cluster",
                                         weighted = TRUE),
       y ~ x1 + x2, cgroup = "never", weights = ~w, vcov = ~id),
  case("logit_last_cluster", d3, list(family = "logit", control = "notyet", vcov = "cluster",
                                      weighted = TRUE),
       y ~ x1, weights = ~w, vcov = ~id),
  case("logit_never_hetero", d4, list(family = "logit", control = "never", vcov = "hetero"),
       y ~ x1, cgroup = "never", vcov = "hetero")
)
cat(toJSON(list(r = R.version.string, fixest = as.character(packageVersion("fixest")),
                etwfe = as.character(packageVersion("etwfe")),
                marginaleffects = as.character(packageVersion("marginaleffects")),
                tolerance = TOL, cases = cases),
           digits = NA, auto_unbox = TRUE, null = "null", na = "null"))
