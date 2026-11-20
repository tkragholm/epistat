# Variance of weighted Cox fits and Kaplan-Meier curves whose weights come from
# estimated models of censoring, of treatment, or of both, checked against the
# infinitesimal jackknife of the whole procedure, found by numerical derivatives.
#
# The procedure is: fit the model(s) of the weights, form the weights, fit the
# outcome model. The case weight omega_u of a subject multiplies that subject's
# contribution to every one of these fits, and the influence of the subject on the
# outcome estimate is the derivative of the estimate with respect to omega_u at 1.
# It is found by central differences of the whole procedure refitted with
# perturbed weights, with Richardson extrapolation (steps h and h / 2). The
# covariance is the sum over subjects of the products of the influences. The
# covariance that takes the weights as known perturbs the outcome fit only. No
# formula of the Rust code is shared with this construction: the models are
# fitted by glm, survfit and coxph with case weights.
#
# Settings (one per case): the treatment model (glm of the baseline treatment on
# covariates, ATE weights, stabilised or not), the censoring model (pooled
# logistic as ipw::ipwtm(type = "cens"), Kaplan-Meier or Cox model of the censoring
# hazard), the weight of the row that the outcome uses (through the row / at the
# stop, or before the row / at the start), and the outcome (Cox, Kaplan-Meier).
# Run: Rscript weighting_estimated.R > weighting_estimated.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261122)
n <- 200
sim_pp <- function(n) {
  rows <- list()
  for (i in seq_len(n)) {
    x1 <- round(rnorm(1), 3)
    x2 <- rbinom(1, 1, 0.4)
    a <- rbinom(1, 1, plogis(0.3 * x1 - 0.2 * x2))
    l <- round(rnorm(1), 3)
    k <- 0
    repeat {
      k <- k + 1
      l <- round(0.6 * l + rnorm(1, 0, 0.7) + 0.2 * a, 3)
      ev <- rbinom(1, 1, plogis(-3 + 0.4 * l - 0.5 * a + 0.3 * x1))
      ce <- if (ev == 1) 0 else rbinom(1, 1, plogis(-2.2 + 0.6 * l + 0.4 * a - 0.3 * x2 + 0.05 * k))
      rows[[length(rows) + 1]] <- data.frame(id = i, k = k, a = a, x1 = x1, x2 = x2, l = l,
                                             event = ev, cens = ce)
      if (ev == 1 || ce == 1) break
    }
  }
  d <- do.call(rbind, rows)
  d$start <- d$k - 1
  d$stop <- d$k
  d
}
dat <- sim_pp(n)
N <- nrow(dat)
sub <- dat[!duplicated(dat$id), ]
ctl <- coxph.control(eps = 1e-13, toler.chol = 1e-14, iter.max = 100)
gctl <- glm.control(epsilon = 1e-13, maxit = 100)
tms <- c(2, 4, 7, 10)

stepval <- function(times, cum, tau, left) {
  idx <- if (left) findInterval(tau, times, left.open = TRUE) else findInterval(tau, times)
  c(0, cum)[idx + 1]
}

# Censoring weights from a Kaplan-Meier or Cox model of the censoring hazard,
# with case weights om per row (the construction of weighting.R).
hazard_weights <- function(om, model, stab, ties, xcols, by_group) {
  rw <- dat
  rw$om <- om
  rw$cn <- as.integer(rw$cens == 1)    # last row with status 0
  gs <- if (by_group) rw$a else rep(0, N)
  Gstep <- function(s, left, tau) {
    sf <- survfit(Surv(start, stop, cn) ~ 1, data = s, weights = om, ctype = 1, stype = 1)
    ok <- sf$n.event > 0
    ev <- sf$time[ok]
    lg <- cumsum(log(1 - sf$n.event[ok] / sf$n.risk[ok]))
    idx <- if (left) findInterval(tau, ev, left.open = TRUE) else findInterval(tau, ev)
    c(0, lg)[idx + 1]
  }
  if (model == "cox") {
    rhs <- paste(c(xcols, if (by_group) "strata(a)"), collapse = " + ")
    fit <- coxph(as.formula(paste("Surv(start, stop, cn) ~", rhs)), data = rw, weights = om,
                 ties = ties, control = ctl)
    bh <- basehaz(fit, centered = FALSE)
    th_all <- exp(drop(as.matrix(rw[, xcols, drop = FALSE]) %*% coef(fit)))
  }
  out_stop <- numeric(N); out_start <- numeric(N)
  for (g in unique(gs)) {
    sel <- gs == g
    s <- rw[sel, ]
    s0 <- ave(s$start, s$id, FUN = min)
    g0 <- Gstep(s, FALSE, s0)
    km_stop <- Gstep(s, TRUE, s$stop) - g0
    km_start <- Gstep(s, FALSE, s$start) - g0
    if (model == "km") {
      den_stop <- km_stop; den_start <- km_start
    } else {
      bhg <- if (by_group) bh[bh$strata == paste0("a=", g), ] else bh
      th <- th_all[sel]
      Hl <- function(tau) stepval(bhg$time, bhg$hazard, tau, TRUE)
      Hi <- function(tau) stepval(bhg$time, bhg$hazard, tau, FALSE)
      contrib <- th * (Hi(s$stop) - Hi(s$start))
      prior <- ave(contrib, s$id, FUN = function(x) c(0, cumsum(x)[-length(x)]))
      den_stop <- -(prior + th * (Hl(s$stop) - Hi(s$start)))
      den_start <- -prior
    }
    num_stop <- 0; num_start <- 0
    if (stab) { num_stop <- km_stop; num_start <- km_start }
    out_stop[sel] <- exp(num_stop - den_stop)
    out_start[sel] <- exp(num_start - den_start)
  }
  list(stop = out_stop, start = out_start)
}

fden <- cens ~ a + x1 + x2 + l + k
fnum <- cens ~ a + x1 + k
pooled_weights <- function(om, stab) {
  sel <- ave(dat$cens, dat$id, FUN = function(x) c(0, cumsum(x)[-length(x)])) == 0
  dd <- dat; dd$om <- om; dd <- dd[sel, ]
  md <- glm(fden, data = dd, family = binomial, weights = om, control = gctl)
  fd <- 1 - predict(md, newdata = dat, type = "response")
  fn <- rep(1, N)
  if (stab) {
    mn <- glm(fnum, data = dd, family = binomial, weights = om, control = gctl)
    fn <- 1 - predict(mn, newdata = dat, type = "response")
  }
  f <- fn / fd
  w <- ave(f, dat$id, FUN = cumprod)
  list(stop = w, start = w / f)
}

iptw_weights <- function(om_sub, stab) {
  sb <- sub; sb$om <- om_sub
  m <- glm(a ~ x1 + x2, data = sb, family = binomial, weights = om, control = gctl)
  p <- fitted(m)
  w <- ifelse(sub$a == 1, 1 / p, 1 / (1 - p))
  if (stab) {
    pi0 <- sum(om_sub * sub$a) / sum(om_sub)
    w <- w * ifelse(sub$a == 1, pi0, 1 - pi0)
  }
  w[dat$id]
}

row_weights <- function(spec, om_sub) {
  om <- om_sub[dat$id]
  w <- rep(1, N)
  if (!is.null(spec$iptw)) w <- w * iptw_weights(om_sub, spec$iptw$stab)
  if (!is.null(spec$cens)) {
    c <- spec$cens
    cw <- if (c$model == "pooled") pooled_weights(om, c$stab)
          else hazard_weights(om, c$model, c$stab, c$ties, c$xcols, c$by_group)
    w <- w * (if (c$at == "stop") cw$stop else cw$start)
  }
  w
}

outcome <- function(spec, w, om_sub) {
  dd <- dat
  dd$ww <- w * om_sub[dat$id]
  if (spec$outcome == "cox") {
    unname(coef(coxph(Surv(start, stop, event) ~ a + x1 + l, data = dd, weights = ww,
                      ties = "efron", control = ctl)))
  } else {
    sf <- survfit(Surv(start, stop, event) ~ a, data = dd, weights = ww)
    s <- summary(sf, times = tms, extend = TRUE)
    c(matrix(s$surv, nrow = length(tms)))     # times x groups, column-major
  }
}

fdiff <- function(f, h) {
  d <- function(hh) (f(1 + hh) - f(1 - hh)) / (2 * hh)
  d1 <- d(h); d2 <- d(h / 2)
  list(rich = (4 * d2 - d1) / 3, plain = d1)
}

# cl: the cluster of each subject, or NULL; the units of the perturbation are the
# clusters, or the subjects.
run_case <- function(name, spec, cl = NULL, h = 1e-4) {
  t0 <- proc.time()[3]
  one <- rep(1, n)
  w0 <- row_weights(spec, one)
  est0 <- outcome(spec, w0, one)
  m <- length(est0)
  units <- if (is.null(cl)) seq_len(n) else sort(unique(cl))
  nu <- length(units)
  inf_adj <- list(rich = matrix(0, nu, m), plain = matrix(0, nu, m))
  inf_kn <- list(rich = matrix(0, nu, m), plain = matrix(0, nu, m))
  for (ui in seq_len(nu)) {
    members <- if (is.null(cl)) units[ui] else which(cl == units[ui])
    pert <- function(x) { o <- one; o[members] <- x; o }
    ra <- fdiff(function(x) outcome(spec, row_weights(spec, pert(x)), pert(x)), h)
    rk <- fdiff(function(x) outcome(spec, w0, pert(x)), h)
    inf_adj$rich[ui, ] <- ra$rich; inf_adj$plain[ui, ] <- ra$plain
    inf_kn$rich[ui, ] <- rk$rich; inf_kn$plain[ui, ] <- rk$plain
  }
  cat(name, "done in", proc.time()[3] - t0, "s\n", file = stderr())
  list(name = name, spec = spec,
       n_units = nu, cluster = if (is.null(cl)) NULL else cl[dat$id],
       id = dat$id - 1, start = dat$start, stop = dat$stop, status = dat$event,
       censored = dat$cens, a = dat$a, x1 = dat$x1, x2 = dat$x2, l = dat$l, k = dat$k,
       weights = w0, estimate = est0,
       covariance = crossprod(inf_adj$rich), covariance_plain = crossprod(inf_adj$plain),
       known = crossprod(inf_kn$rich), known_plain = crossprod(inf_kn$plain))
}

cases <- list(
  run_case("pooled_cox", list(cens = list(model = "pooled", stab = TRUE, at = "stop"), outcome = "cox")),
  run_case("pooled_km", list(cens = list(model = "pooled", stab = FALSE, at = "start"), outcome = "km")),
  run_case("iptw_pooled_cox", list(iptw = list(stab = FALSE),
           cens = list(model = "pooled", stab = TRUE, at = "stop"), outcome = "cox")),
  run_case("iptw_pooled_km", list(iptw = list(stab = TRUE),
           cens = list(model = "pooled", stab = FALSE, at = "stop"), outcome = "km")),
  run_case("km_cox", list(cens = list(model = "km", stab = FALSE, at = "stop", by_group = FALSE,
           xcols = NULL, ties = "efron"), outcome = "cox")),
  run_case("km_km", list(cens = list(model = "km", stab = FALSE, at = "start", by_group = TRUE,
           xcols = NULL, ties = "efron"), outcome = "km")),
  run_case("cox_cox_efron_stab", list(cens = list(model = "cox", stab = TRUE, at = "stop",
           by_group = FALSE, xcols = c("a", "x1", "l"), ties = "efron"), outcome = "cox")),
  run_case("cox_km_breslow_strata", list(cens = list(model = "cox", stab = FALSE, at = "start",
           by_group = TRUE, xcols = c("x1", "l"), ties = "breslow"), outcome = "km")),
  run_case("iptw_cox_cox", list(iptw = list(stab = FALSE),
           cens = list(model = "cox", stab = TRUE, at = "start", by_group = FALSE,
                       xcols = c("a", "x1", "l"), ties = "efron"), outcome = "cox")),
  run_case("iptw_km_km", list(iptw = list(stab = FALSE),
           cens = list(model = "km", stab = FALSE, at = "stop", by_group = TRUE,
                       xcols = NULL, ties = "efron"), outcome = "km")),
  run_case("cluster_iptw_pooled_cox", list(iptw = list(stab = FALSE),
           cens = list(model = "pooled", stab = TRUE, at = "stop"), outcome = "cox"),
           cl = (seq_len(n) - 1) %% 40),
  run_case("cluster_cox_km", list(cens = list(model = "cox", stab = FALSE, at = "stop",
           by_group = TRUE, xcols = c("x1", "l"), ties = "efron"), outcome = "km"),
           cl = (seq_len(n) - 1) %% 40)
)
design <- function(f, d) unname(model.matrix(f, d))
out <- list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  times = tms,
  z_den = design(~ a + x1 + x2 + l + k, dat), z_num = design(~ a + x1 + k, dat),
  z_iptw = design(~ x1 + x2, sub),
  treat = sub$a,
  cases = cases
)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
