# Inverse probability weights and weighted outcome models, checked against
# WeightIt (weights, summary, M-estimation variance of coxph_weightit), ipw
# (ipwpoint) and survival (coxph, survfit), on the lalonde data of cobalt with
# simulated survival times.
#
# iptw: for each estimand the weights, fitted probabilities and coefficients of
# WeightIt, the summary of the weights it prints, and the weights of
# ipw::ipwpoint for the stabilised case.
# cox: the coefficients and the covariance of coxph_weightit (which accounts for
# the estimation of the weights) and of coxph(weights, robust = TRUE) (which does
# not). Where WeightIt cannot be used (several rows per subject) the covariance
# is built from derivatives found by central differences with Richardson
# extrapolation, refitting coxph with perturbed subject weights, so that no
# formula of the Rust code is shared with the reference.
# km: the same finite-difference construction for the weighted Kaplan-Meier
# curves of survfit at chosen times, with and without delayed entry.
# ipcw: weights of the censoring distribution built from survfit and coxph.
# The versions are in the file. Numbers are written with 17 digits.
# Run: Rscript weighting.R > weighting.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(WeightIt)
  library(ipw)
})
data("lalonde", package = "cobalt")
set.seed(20261120)
n <- nrow(lalonde)
d <- lalonde
d$time <- ceiling(rexp(n, 0.02 * exp(0.5 * (d$re74 > 0) - 0.3 * d$treat + 0.03 * (d$age - 27))))
cens <- ceiling(runif(n, 20, 120))
d$status <- as.integer(d$time <= cens)
d$t <- pmin(d$time, cens)
d$id <- seq_len(n)
f <- treat ~ age + educ + race + married + nodegree + re74 + re75
Z <- unname(model.matrix(f, d))
a <- d$treat
eps <- .Machine$double.eps

summ <- function(w) {
  list(n = length(w), sum = sum(w), min = min(w), max = max(w), mean = mean(w),
       cv = sd(w) / mean(w), ess = sum(w)^2 / sum(w^2))
}
kappa_ps <- function(p) {
  g <- crossprod(Z * sqrt(p * (1 - p)))
  s <- sqrt(diag(g))
  kappa(diag(1 / s) %*% g %*% diag(1 / s), exact = TRUE)
}

iptw <- list()
wobj <- list()
for (s in list(c("ate", "ATE", FALSE), c("att", "ATT", FALSE), c("atc", "ATC", FALSE),
               c("ate_stab", "ATE", TRUE))) {
  W <- weightit(f, data = d, method = "glm", estimand = s[2], stabilize = as.logical(s[3]))
  wobj[[s[1]]] <- W
  sm <- summary(W)
  p <- W$ps
  entry <- list(
    name = s[1], estimand = s[2], stabilise = as.logical(s[3]),
    weights = unname(W$weights), propensity = unname(p),
    coefficients = unname(coef(glm(f, data = d, family = binomial))),
    treated = summ(W$weights[a == 1]), control = summ(W$weights[a == 0]),
    weightit_ess = as.vector(unname(sm$effective.sample.size[2, ])),
    weightit_cv = unname(unlist(sm$coef.of.var)),
    kappa = kappa_ps(p)
  )
  if (s[1] == "ate_stab") {
    ip <- ipwpoint(exposure = treat, family = "binomial", link = "logit",
                   numerator = ~1, denominator = ~ age + educ + race + married + nodegree + re74 + re75,
                   data = d)
    entry$ipw_weights <- unname(ip$ipw.weights)
  }
  iptw[[length(iptw) + 1]] <- entry
}

# Central differences, with Richardson extrapolation, of f at theta in the
# direction i: returns the extrapolated derivative and the plain one with the
# larger step.
fd <- function(f, theta, i, h) {
  at <- function(hh) {
    up <- theta; dn <- theta; up[i] <- up[i] + hh; dn[i] <- dn[i] - hh
    (f(up) - f(dn)) / (2 * hh)
  }
  d1 <- at(h); d2 <- at(h / 2)
  list(rich = (4 * d2 - d1) / 3, plain = d1)
}

weights_from <- function(theta, Zm, aa, estimand, stab) {
  k <- ncol(Zm)
  p <- plogis(drop(Zm %*% theta[seq_len(k)]))
  base <- switch(estimand,
    ATE = ifelse(aa == 1, 1 / p, 1 / (1 - p)),
    ATT = ifelse(aa == 1, 1, p / (1 - p)),
    ATC = ifelse(aa == 1, (1 - p) / p, 1))
  if (stab) base <- base * ifelse(aa == 1, theta[k + 1], 1 - theta[k + 1])
  base
}
# A step in a coefficient changes the linear predictor by the step times the
# largest entry of its column, so the steps are scaled by it.
hscale <- c(apply(abs(Z), 2, max), 1)
ps_if <- function(Zm, aa, p, stab, pi0) {
  k <- ncol(Zm)
  A <- matrix(0, k + stab, k + stab)
  A[1:k, 1:k] <- -crossprod(Zm * sqrt(p * (1 - p)))
  psi <- Zm * (aa - p)
  if (stab) { A[k + 1, k + 1] <- -nrow(Zm); psi <- cbind(psi, aa - pi0) }
  -t(solve(A, t(psi)))
}

# Cox outcome models ----------------------------------------------------------
ctl_tight <- coxph.control(eps = 1e-13, toler.chol = 1e-14, iter.max = 100)

estimand_of <- list(ate = "ATE", att = "ATT", atc = "ATC", ate_stab = "ATE")
# The stacked estimating equations of the propensity model and of the Breslow
# Cox score of WeightIt (its own psi function), differentiated numerically as a
# whole, which gives the exact derivative of the Cox score with respect to the
# weights. `shortcut` replaces that block by what WeightIt computes: the Cox score
# at weights of one times the derivative of the weights.
stacked_cox <- function(fit, data, estimand, stab, cluster) {
  psi_out <- WeightIt:::.get_coxph_psi(fit)
  X <- unname(as.matrix(data[, names(coef(fit)), drop = FALSE]))
  y <- cbind(time = data$t, status = data$status)
  k <- ncol(Z)
  theta0 <- c(unname(coef(glm(f, data = data, family = binomial))), if (stab) mean(a))
  B <- coef(fit)
  psi_ps <- function(th) {
    p <- plogis(drop(Z %*% th[1:k]))
    out <- Z * (a - p)
    if (stab) out <- cbind(out, a - th[k + 1])
    out
  }
  Psi_sum <- function(th) {
    w <- weights_from(th[seq_len(k + stab)], Z, a, estimand, stab)
    c(colSums(psi_ps(th)), colSums(psi_out(th[k + stab + seq_along(B)], X, y, w, 0)))
  }
  th <- c(theta0, B)
  w <- weights_from(theta0, Z, a, estimand, stab)
  A <- numDeriv::jacobian(Psi_sum, th)
  Pm <- cbind(psi_ps(theta0), psi_out(B, X, y, w, 0))
  units <- if (is.null(cluster)) seq_len(n) else data[[cluster]]
  Pu <- rowsum(Pm, units)
  M <- crossprod(Pu)
  V <- function(A) { Ai <- solve(A); Ai %*% M %*% t(Ai) }
  idx <- k + stab + seq_along(B)
  # WeightIt's block: unit-weight score times dw/dtheta
  dw <- t(sapply(seq_len(n), function(i) {
    numDeriv::grad(function(th) weights_from(th, Z, a, estimand, stab)[i], theta0)
  }))
  As <- A
  As[idx, seq_len(k + stab)] <- crossprod(psi_out(B, X, y, 1, 0), dw)
  list(exact = V(A)[idx, idx, drop = FALSE], shortcut = V(As)[idx, idx, drop = FALSE])
}

cox <- list()
cox_case <- function(name, iptw_name, ties, rhs, xcols, data, W, cluster = NULL) {
  args <- list(formula = as.formula(paste("Surv(t, status) ~", rhs)), data = data,
               weightit = W, ties = ties)
  if (!is.null(cluster)) args$cluster <- as.formula(paste("~", cluster))
  fit <- do.call(coxph_weightit, args)
  cargs <- list(formula = as.formula(paste("Surv(t, status) ~", rhs)), data = data,
                weights = W$weights, ties = ties, robust = TRUE)
  if (!is.null(cluster)) cargs$cluster <- data[[cluster]]
  naive <- do.call(coxph, cargs)
  x <- unname(as.matrix(data[, xcols, drop = FALSE]))
  vm <- naive$naive.var
  sc <- colSums(W$weights * abs(sweep(x, 2, colSums(W$weights * x) / sum(W$weights)))) / sum(W$weights)
  st <- stacked_cox(fit, data, estimand_of[[iptw_name]], iptw_name == "ate_stab", cluster)
  list(name = name, iptw = iptw_name, ties = ties, columns = I(xcols),
       exact_covariance = unname(st$exact), shortcut_covariance = unname(st$shortcut),
       stop = data$t, status = data$status,
       cluster = if (is.null(cluster)) NULL else data[[cluster]],
       x = x, coefficients = I(unname(coef(fit))),
       covariance = unname(vcov(fit)), robust_covariance = unname(vcov(naive)),
       kappa = kappa(diag(1 / sc, length(sc)) %*% solve(vm) %*% diag(1 / sc, length(sc)), exact = TRUE))
}
d$sexish <- as.numeric(d$re74 > 0)
d$cl <- (seq_len(n) - 1) %% 150
# coxph_weightit's estimating functions are those of the Breslow likelihood, so
# these cases are Breslow. The Efron cases below use finite differences.
cox[[1]] <- cox_case("ate_breslow", "ate", "breslow", "treat + age", c("treat", "age"), d, wobj$ate)
cox[[2]] <- cox_case("att_breslow", "att", "breslow", "treat + age + sexish", c("treat", "age", "sexish"), d, wobj$att)
cox[[3]] <- cox_case("atc_breslow", "atc", "breslow", "treat", c("treat"), d, wobj$atc)
cox[[4]] <- cox_case("stab_breslow", "ate_stab", "breslow", "treat + age", c("treat", "age"), d, wobj$ate_stab)
cox[[5]] <- cox_case("ate_breslow_cluster", "ate", "breslow", "treat + age", c("treat", "age"), d, wobj$ate, "cl")

# Counting-process data: every subject with follow-up beyond a cut is split in
# two rows, the second with a time-varying covariate.
cut <- ceiling(runif(n, 1, 40))
cut[d$t <= cut] <- NA
sp <- do.call(rbind, lapply(seq_len(n), function(i) {
  if (is.na(cut[i])) {
    data.frame(subject = i, start = 0, stop = d$t[i], status = d$status[i], tv = 0)
  } else {
    data.frame(subject = c(i, i), start = c(0, cut[i]), stop = c(cut[i], d$t[i]),
               status = c(0, d$status[i]), tv = c(0, 1))
  }
}))
sp$treat <- a[sp$subject]
sp$age <- d$age[sp$subject]
sp$sexish <- d$sexish[sp$subject]
one <- data.frame(subject = seq_len(n), start = 0, stop = d$t, status = d$status,
                  treat = a, age = d$age, sexish = d$sexish)

cox_fd <- function(name, estimand, stab, ties, rows, covs, entry, h) {
  theta0 <- c(unname(coef(glm(f, data = d, family = binomial))), if (stab) mean(a))
  lhs <- if (entry) "Surv(start, stop, status)" else "Surv(stop, status)"
  fml <- as.formula(paste(lhs, "~", paste(covs, collapse = " + ")))
  fitw <- function(wsubj) {
    do.call(coxph, list(fml, data = rows, weights = wsubj[rows$subject], ties = ties,
                        control = ctl_tight))
  }
  w0 <- weights_from(theta0, Z, a, estimand, stab)
  beta0 <- coef(fitw(w0))
  p <- plogis(drop(Z %*% theta0[seq_len(ncol(Z))]))
  ifs <- ps_if(Z, a, p, stab, mean(a))
  k <- length(theta0)
  db <- function(extrap) {
    sapply(seq_len(n), function(u) {
      g <- function(wu) { w <- w0; w[u] <- wu; coef(fitw(w)) }
      up <- function(hh) (g(w0[u] + hh) - g(w0[u] - hh)) / (2 * hh)
      hh <- h * w0[u]
      d1 <- up(hh); d2 <- up(hh / 2)
      if (extrap) (4 * d2 - d1) / 3 else d1
    })
  }
  dba <- function(extrap) {
    sapply(seq_len(k), function(i) {
      r <- fd(function(th) coef(fitw(weights_from(th, Z, a, estimand, stab))), theta0, i, h / hscale[i])
      if (extrap) r$rich else r$plain
    })
  }
  total <- function(extrap) {
    dbw <- t(matrix(db(extrap), nrow = length(covs))) * w0
    dbt <- matrix(dba(extrap), nrow = length(covs))
    inf <- dbw + ifs %*% t(dbt)
    list(adj = crossprod(inf), known = crossprod(dbw))
  }
  r <- total(TRUE); q <- total(FALSE)
  x <- unname(as.matrix(rows[, covs, drop = FALSE]))
  list(name = name, iptw = switch(paste(estimand, stab), "ATE FALSE" = "ate", "ATT FALSE" = "att",
                                  "ATC FALSE" = "atc", "ATE TRUE" = "ate_stab"),
       ties = ties,
       start = if (entry) rows$start else NULL, stop = rows$stop, status = rows$status,
       subject = if (entry) rows$subject - 1 else NULL,
       x = x, coefficients = I(unname(beta0)),
       covariance = unname(r$adj), robust_covariance = unname(r$known),
       covariance_plain = unname(q$adj), robust_covariance_plain = unname(q$known))
}
cox_fdcases <- list(
  cox_fd("fd_ate_breslow", "ATE", FALSE, "breslow", one, c("treat", "age"), FALSE, 1e-4),
  cox_fd("fd_att_efron", "ATT", FALSE, "efron", one, c("treat", "age"), FALSE, 1e-4),
  cox_fd("fd_stab_efron", "ATE", TRUE, "efron", one, c("treat", "age"), FALSE, 1e-4),
  cox_fd("cp_ate_efron", "ATE", FALSE, "efron", sp, c("treat", "age", "tv"), TRUE, 1e-4),
  cox_fd("cp_att_breslow", "ATT", FALSE, "breslow", sp, c("treat", "age", "tv"), TRUE, 1e-4)
)

# Kaplan-Meier -----------------------------------------------------------------
km_fd <- function(name, estimand, stab, entry, times, h) {
  dd <- d
  if (entry) dd$start <- pmin(floor(runif(n, 0, 1) * dd$t * 0.8), dd$t - 1) else dd$start <- 0
  theta0 <- c(unname(coef(glm(f, data = dd, family = binomial))), if (stab) mean(a))
  k <- length(theta0)
  curve <- function(w) {
    sf <- survfit(Surv(start, t, status) ~ treat, data = dd, weights = w)
    s <- summary(sf, times = times, extend = TRUE)
    matrix(s$surv, nrow = length(times))        # times x groups (control, treated)
  }
  w0 <- weights_from(theta0, Z, a, estimand, stab)
  S0 <- curve(w0)
  sf_rob <- survfit(Surv(start, t, status) ~ treat, data = dd, weights = w0, robust = TRUE, id = id)
  se_rob <- summary(sf_rob, times = times, extend = TRUE)$std.err
  p <- plogis(drop(Z %*% theta0[seq_len(ncol(Z))]))
  ifs <- ps_if(Z, a, p, stab, mean(a))
  dS <- function(extrap) {
    # array n x times x groups
    out <- array(0, c(n, length(times), 2))
    for (u in seq_len(n)) {
      g <- function(wu) { w <- w0; w[u] <- wu; curve(w) }
      hh <- h * w0[u]
      d1 <- (g(w0[u] + hh) - g(w0[u] - hh)) / (2 * hh)
      d2 <- (g(w0[u] + hh / 2) - g(w0[u] - hh / 2)) / hh
      out[u, , ] <- if (extrap) (4 * d2 - d1) / 3 else d1
    }
    out
  }
  dSa <- function(extrap) {
    out <- array(0, c(k, length(times), 2))
    for (i in seq_len(k)) {
      r <- fd(function(th) curve(weights_from(th, Z, a, estimand, stab)), theta0, i, h / hscale[i])
      out[i, , ] <- if (extrap) r$rich else r$plain
    }
    out
  }
  total <- function(extrap) {
    du <- dS(extrap); da <- dSa(extrap)
    adj <- array(0, c(length(times), 2, 2)); known <- adj
    for (ti in seq_along(times)) {
      inf <- sapply(1:2, function(g) w0 * du[, ti, g] + drop(ifs %*% da[, ti, g]))
      kn <- sapply(1:2, function(g) w0 * du[, ti, g])
      adj[ti, , ] <- crossprod(inf); known[ti, , ] <- crossprod(kn)
    }
    list(adj = adj, known = known)
  }
  r <- total(TRUE); q <- total(FALSE)
  list(name = name, estimand = estimand, stabilise = stab, entry = entry,
       start = dd$start, stop = dd$t, status = dd$status, times = times,
       iptw_weights = w0,
       surv = S0, se_robust_survfit = se_rob,
       covariance = r$adj, known = r$known, covariance_plain = q$adj, known_plain = q$known)
}
tms <- c(10, 30, 60, 100)
km <- list(
  km_fd("km_ate", "ATE", FALSE, FALSE, tms, 1e-4),
  km_fd("km_att_entry", "ATT", FALSE, TRUE, tms, 1e-4),
  km_fd("km_ate_stab", "ATE", TRUE, FALSE, tms, 1e-4)
)

# Censoring weights ------------------------------------------------------------
# A KM or Cox model of the censoring process; the weight of a row at its stop
# is 1 / G(stop-), at its start 1 / G(start), with numerators the marginal KM.
set.seed(777)
m <- 400
x1 <- round(rnorm(m), 2)
x2 <- rbinom(m, 1, 0.4)
grp <- sample(0:1, m, replace = TRUE)
Tev <- ceiling(rexp(m, 0.05 * exp(0.4 * x1)))
Cen <- ceiling(rexp(m, 0.04 * exp(0.5 * x1 + 0.6 * x2)))
tt <- pmin(Tev, Cen); stt <- as.integer(Tev <= Cen)
cutp <- ceiling(runif(m, 1, 12)); cutp[tt <= cutp] <- NA
# delayed entry for about a third of the subjects
enter <- ifelse(runif(m) < 0.3, sample(1:3, m, replace = TRUE), 0)
enter <- pmin(enter, ifelse(is.na(cutp), tt - 1, cutp - 1))
rows <- do.call(rbind, lapply(seq_len(m), function(i) {
  if (is.na(cutp[i])) data.frame(id = i, start = enter[i], stop = tt[i], status = stt[i], x1 = x1[i], tv = 0, grp = grp[i])
  else data.frame(id = rep(i, 2), start = c(enter[i], cutp[i]), stop = c(cutp[i], tt[i]), status = c(0, stt[i]),
                  x1 = x1[i], tv = c(0, 1), grp = grp[i])
}))
# the censoring event: a status 0 on the last row of the subject
last <- ave(rows$stop, rows$id, FUN = max) == rows$stop
rows$cens <- as.integer(rows$status == 0 & last)
# weights for a model: function giving G(tau-) or G(tau) for every row
stepval <- function(times, haz, tau, left) {
  # cumulative hazard at tau, with or without the jump at tau
  cs <- c(0, cumsum(haz))
  idx <- if (left) findInterval(tau, times, left.open = TRUE) else findInterval(tau, times)
  cs[idx + 1]
}
ipcw_case <- function(name, model, by_group, stab, ties = "efron") {
  rw <- rows
  gs <- if (by_group) rw$grp else rep(0, nrow(rw))
  Gstep <- function(sub, left, tau) {
    # log of the product-limit survival of the censoring process in sub
    sf <- survfit(Surv(start, stop, cens) ~ 1, data = sub, ctype = 1, stype = 1)
    ev <- sf$time[sf$n.event > 0]
    lg <- cumsum(log(1 - sf$n.event[sf$n.event > 0] / sf$n.risk[sf$n.event > 0]))
    idx <- if (left) findInterval(tau, ev, left.open = TRUE) else findInterval(tau, ev)
    c(0, lg)[idx + 1]
  }
  kap <- NA
  if (model == "cox") {
    fml <- if (by_group) Surv(start, stop, cens) ~ x1 + tv + strata(grp) else Surv(start, stop, cens) ~ x1 + tv
    fit <- coxph(fml, data = rw, ties = ties)
    bh <- basehaz(fit, centered = FALSE)
    th_all <- exp(drop(as.matrix(rw[, c("x1", "tv")]) %*% coef(fit)))
    vm <- fit$var
    sc <- apply(as.matrix(rw[, c("x1", "tv")]), 2, function(x) mean(abs(x - mean(x))))
    kap <- kappa(diag(1 / sc, 2) %*% solve(vm) %*% diag(1 / sc, 2), exact = TRUE)
  }
  out_stop <- numeric(nrow(rw)); out_start <- numeric(nrow(rw))
  for (g in unique(gs)) {
    sel <- gs == g
    sub <- rw[sel, ]
    s0 <- ave(sub$start, sub$id, FUN = min)
    g0 <- Gstep(sub, FALSE, s0)
    if (model == "km") {
      den_stop <- Gstep(sub, TRUE, sub$stop) - g0; den_start <- Gstep(sub, FALSE, sub$start) - g0
    } else {
      bhg <- if (by_group) bh[bh$strata == paste0("grp=", g), ] else bh
      th <- th_all[sel]
      Hleft <- function(tau) stepval(bhg$time, diff(c(0, bhg$hazard)), tau, TRUE)
      Hincl <- function(tau) stepval(bhg$time, diff(c(0, bhg$hazard)), tau, FALSE)
      # cumulative hazard over the earlier rows of the subject, inclusive
      prior <- numeric(nrow(sub))
      for (i in seq_len(nrow(sub))) {
        ks <- which(sub$id == sub$id[i] & sub$stop <= sub$start[i])
        prior[i] <- sum(th[ks] * (Hincl(sub$stop[ks]) - Hincl(sub$start[ks])))
      }
      den_stop <- -(prior + th * (Hleft(sub$stop) - Hincl(sub$start)))
      den_start <- -prior
    }
    num_stop <- 0; num_start <- 0
    if (stab) {
      num_stop <- Gstep(sub, TRUE, sub$stop) - g0; num_start <- Gstep(sub, FALSE, sub$start) - g0
    }
    out_stop[sel] <- exp(num_stop - den_stop); out_start[sel] <- exp(num_start - den_start)
  }
  list(name = name, model = model, by_group = by_group, stabilise = stab, ties = ties,
       id = rw$id, start = rw$start, stop = rw$stop, status = rw$status,
       x = unname(as.matrix(rw[, c("x1", "tv")])), group = rw$grp, kappa = kap,
       weight_stop = out_stop, weight_start = out_start)
}
ipcw <- list(
  ipcw_case("km", "km", FALSE, FALSE),
  ipcw_case("km_group_stab", "km", TRUE, TRUE),
  ipcw_case("cox_efron", "cox", FALSE, FALSE),
  ipcw_case("cox_stab", "cox", FALSE, TRUE),
  ipcw_case("cox_breslow_group", "cox", TRUE, TRUE, "breslow")
)

out <- list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  WeightIt = as.character(packageVersion("WeightIt")),
                  ipw = as.character(packageVersion("ipw")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  treat = a, design = Z, iptw = iptw, cox = cox, cox_fd = cox_fdcases, km = km, ipcw = ipcw
)
cat(toJSON(out, auto_unbox = TRUE, digits = I(17), na = "null", null = "null", matrix = "rowmajor"))
