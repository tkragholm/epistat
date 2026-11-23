# Net survival and excess hazard models against R references.
#
# The reference hazards are those of a synthetic daily rate table by age in days,
# sex and calendar date, with whole-day cut points: Gompertz in age, a factor for
# sex and a trend in the calendar year. Deaths fall on whole days, censorings and
# entries on half days. Five sources:
#  * relsurv::rs.surv (Pohar Perme and Ederer II, type "fleming-harrington" so
#    that the survival is exp(-Lambda), the three conf.type) on a cohort of three
#    strata, one of them without events, without delayed entry, with
#    precision = 0.5 so that each step of its integral stays in one cell of the
#    table. Its Ederer II is then exact. Its Pohar Perme integral over the time
#    between two exit times is not: the sums of its steps are not reset from one
#    step to the next, so the integrand of a step is the mean over the steps so
#    far. A transcription of that rule is checked against it here, and the
#    deviation is written out.
#  * the continuous estimators written here from their definitions, evaluated at
#    every entry and exit time, on the same cohort with delayed entry. Between
#    two such times the risk set is fixed and the integral of
#    sum w lambda / sum w is ln W(b) - ln W(a); this identity is checked against
#    stats::integrate on a few intervals.
#  * the interval estimators written here from their definition, for two widths.
#  * popEpi::relpois (the Poisson excess hazard model with the link
#    log(mu - d*)) on a Lexis object split by popEpi::splitMulti at the
#    intervals and at the cuts of the table, with delayed entry, and the same
#    link by glm on the records aggregated by interval and covariate pattern
#    (Dickman et al. 2004).
#  * rstpm2::stpm2 with bhazard (the flexible parametric excess hazard model of
#    Nelson et al.), with and without delayed entry, and its predictions of the
#    net survival and the excess hazard; with bhazard, rstpm2 gives limits of
#    the hazard on its own scale, not on the log scale.
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/relsurv.R
# The versions of R and the packages are in the output. Numbers are written with
# 17 significant digits.
suppressMessages({
  library(jsonlite); library(survival); library(relsurv); library(popEpi)
  library(Epi); library(rstpm2); library(bbmle); library(data.table)
})
set.seed(20261123)

# ---- the rate table ----
day0 <- as.numeric(as.Date("1980-01-01"))
age_cuts <- (0:110) * 365
year_cuts <- day0 + (0:45) * 365
na <- length(age_cuts); ny <- length(year_cuts)
haz <- array(0, dim = c(na, 2, ny))
for (a in 1:na) for (s in 1:2) for (y in 1:ny) {
  haz[a, s, y] <- exp(-10.5 + 0.09 * (a - 1) + 0.35 * (s == 1) - 0.012 * (y - 1)) / 365
}
rt <- haz
attributes(rt) <- list(
  dim = c(na, 2L, ny),
  dimnames = list(as.character(0:(na - 1)), c("male", "female"), as.character(1980 + 0:(ny - 1))),
  dimid = c("age", "sex", "year"), type = c(2L, 1L, 3L),
  cutpoints = list(age_cuts, NULL, as.Date(year_cuts, origin = "1970-01-01")),
  class = "ratetable")
stopifnot(is.ratetable(rt))

# The piecewise constant reference hazard of one person over time since the
# origin: the ends of the pieces, the hazards per day and the cumulative hazard
# at the ends. The last cell of each time dimension reaches to infinity, as in R.
pieces <- function(age0, date0, sex, tmax) {
  ta <- age_cuts[age_cuts > age0] - age0
  ty <- year_cuts[year_cuts > date0] - date0
  b <- sort(unique(c(ta, ty))); b <- b[b < tmax]
  ends <- c(b, tmax); starts <- c(0, b); mid <- (starts + ends) / 2
  ai <- findInterval(age0 + mid, age_cuts); yi <- findInterval(date0 + mid, year_cuts)
  h <- haz[cbind(ai, sex, yi)]
  list(ends = ends, haz = h, cum = cumsum((ends - starts) * h))
}
Hf <- function(p, t) approx(c(0, p$ends), c(0, p$cum), xout = t)$y
hf <- function(p, u) p$haz[pmin(findInterval(u, c(0, p$ends), left.open = TRUE), length(p$haz))]
pcs_of <- function(d) lapply(seq_len(nrow(d)), function(i) pieces(d$age0[i], d$date0[i], d$sex[i], d$time[i]))

# Deaths on whole days, censorings on half days, entries at censoring times.
make_cohort <- function(n, scale, shape, entry, cens_hi) {
  d <- data.frame(
    age0 = floor(runif(n, 40, 80) * 365),
    date0 = floor(runif(n, day0 + 5 * 365, day0 + 20 * 365)),
    sex = sample(1:2, n, TRUE),
    x1 = round(rnorm(n), 3), x2 = rbinom(n, 1, 0.4), x3 = sample(0:2, n, TRUE))
  d$grp <- sample(1:2, n, TRUE)
  tt <- 1 + floor(scale * exp(-(0.4 * d$x1 + 0.3 * d$x2 - 0.2 * d$x3) / shape) * (-log(runif(n)))^(1 / shape))
  cc <- 150.5 + floor(runif(n, 0, cens_hi))
  d$time <- pmin(tt, cc); d$status <- as.numeric(tt <= cc)
  d$start <- 0
  if (entry) {
    censor_times <- sort(unique(d$time[d$status == 0]))
    late <- which(runif(n) < 0.3)
    for (i in late) {
      cand <- censor_times[censor_times < 0.6 * d$time[i]]
      if (length(cand) > 0) d$start[i] <- cand[sample.int(length(cand), 1)]
    }
  }
  rownames(d) <- NULL
  d
}
pack <- function(d) as.list(d[, c("age0", "date0", "sex", "x1", "x2", "x3", "grp", "time", "status", "start")])

# ---- 1. relsurv::rs.surv, three strata ----
dA <- make_cohort(400, 2200, 1.2, FALSE, 2500)
# a third stratum of ten persons, none of whom dies
extra <- make_cohort(10, 2200, 1.2, FALSE, 2500)
extra$status <- 0; extra$time <- floor(extra$time) + 0.5; extra$grp <- 3
dA <- rbind(dA, extra)
dA$dt <- as.Date(dA$date0, origin = "1970-01-01")
rs_run <- function(method, conf) {
  f <- rs.surv(Surv(time, status) ~ grp, data = dA, ratetable = rt,
               rmap = list(age = age0, sex = sex, year = dt),
               method = method, type = "fleming-harrington", conf.type = conf, precision = 0.5)
  st <- rep(seq_along(f$strata), f$strata)
  lapply(seq_along(f$strata), function(k) {
    w <- st == k
    list(time = f$time[w], n_risk = f$n.risk[w], events = f$n.event[w], surv = f$surv[w],
         std_err = f$std.err[w], lower = f$lower[w], upper = f$upper[w])
  })
}
relsurv_runs <- list()
for (m in c("pohar-perme", "ederer2")) for (cf in c("log-log", "log", "plain")) {
  relsurv_runs[[length(relsurv_runs) + 1]] <- list(method = m, conf = cf, strata = rs_run(m, cf))
}
pA <- pcs_of(dA)
# relsurv's Pohar Perme integral by its own rule, for the first stratum.
relsurv_rule <- function(d, pc, h) {
  tis <- sort(unique(d$time)); lam <- 0; prev <- 0; out <- c()
  for (b in tis) {
    act <- which(d$time >= b)
    fy <- 0; fyl <- 0; fy2 <- 0; fyl2 <- 0; fint <- 0; u <- prev
    while (u < b) {
      st <- min(h, b - u)
      lu <- sapply(act, function(i) hf(pc[[i]], u + st / 2))
      s0 <- exp(-sapply(act, function(i) Hf(pc[[i]], u))); s1 <- s0 * exp(-st * lu)
      fyl <- fyl + sum(lu / s0); fy <- fy + sum(1 / s0); fyl2 <- fyl2 + sum(lu / s1); fy2 <- fy2 + sum(1 / s1)
      fint <- fint + (fyl / fy / 2 + fyl2 / fy2 / 2) * st; u <- u + st
    }
    Hb <- sapply(act, function(i) Hf(pc[[i]], b)); dead <- d$time[act] == b & d$status[act] == 1
    lam <- lam + sum(exp(Hb)[dead]) / sum(exp(Hb)) - fint; out <- c(out, lam); prev <- b
  }
  exp(-out)
}
g1 <- dA$grp == 1
rule_deviation <- max(abs(relsurv_rule(dA[g1, ], pA[g1], 0.5) / relsurv_runs[[1]]$strata[[1]]$surv - 1))

# ---- 1b. the continuous estimators by definition, with delayed entry ----
exact_ref <- function(d, pc, method) {
  points <- sort(unique(c(d$time, d$start[d$start > 0])))
  stops <- sort(unique(d$time))
  lam <- 0; vr <- 0; prev <- 0
  out <- list(time = c(), lambda = c(), variance = c(), n = c(), events = c())
  for (b in points) {
    act <- which(d$start < b & d$time >= b)
    if (length(act) == 0) { prev <- b; next }
    n <- length(act)
    dead <- d$time[act] == b & d$status[act] == 1
    Ha <- sapply(act, function(i) Hf(pc[[i]], prev)); Hb <- sapply(act, function(i) Hf(pc[[i]], b))
    if (method == "pp") {
      lam <- lam + sum(exp(Hb)[dead]) / sum(exp(Hb)) - log(sum(exp(Hb)) / sum(exp(Ha)))
      vr <- vr + sum(exp(2 * Hb)[dead]) / sum(exp(Hb))^2
    } else {
      lam <- lam + sum(dead) / n - sum(Hb - Ha) / n
      vr <- vr + sum(dead) / n^2
    }
    if (b %in% stops) {
      out$time <- c(out$time, b); out$lambda <- c(out$lambda, lam); out$variance <- c(out$variance, vr)
      out$n <- c(out$n, n); out$events <- c(out$events, sum(dead))
    }
    prev <- b
  }
  out
}
# The identity for the integral, against numerical quadrature on intervals of
# the first stratum.
identity_check <- function(d, pc) {
  points <- sort(unique(c(d$time, d$start[d$start > 0])))
  worst <- 0; prev <- 0
  for (b in points[seq(5, length(points), length.out = 6)]) {
    a <- max(points[points < b])
    act <- which(d$start < b & d$time >= b)
    fu <- Vectorize(function(u) {
      w <- exp(sapply(act, function(i) Hf(pc[[i]], u)))
      sum(w * sapply(act, function(i) hf(pc[[i]], u))) / sum(w)
    })
    brk <- sort(unique(c(a, b, unlist(lapply(act, function(i) pc[[i]]$ends)))))
    brk <- brk[brk >= a & brk <= b]
    q <- sum(sapply(seq_len(length(brk) - 1), function(k)
      integrate(fu, brk[k], brk[k + 1], rel.tol = 1e-13, abs.tol = 0)$value))
    Wa <- sum(exp(sapply(act, function(i) Hf(pc[[i]], a)))); Wb <- sum(exp(sapply(act, function(i) Hf(pc[[i]], b))))
    worst <- max(worst, abs(q - log(Wb / Wa)) / abs(q))
  }
  worst
}
dAe <- make_cohort(400, 2200, 1.2, TRUE, 2500)
dAe <- rbind(dAe, extra)
pAe <- pcs_of(dAe)
exact_runs <- list()
for (m in c("pp", "e2")) {
  exact_runs[[m]] <- lapply(1:3, function(k) { w <- dAe$grp == k; exact_ref(dAe[w, ], pAe[w], m) })
}
g1e <- dAe$grp == 1
identity_deviation <- identity_check(dAe[g1e, ], pAe[g1e])
stopifnot(identity_deviation < 1e-10)

# ---- 2. the interval estimators from their definition ----
interval_ref <- function(d, pc, method, width) {
  tmax <- max(d$time); g <- ceiling(tmax / width)
  S <- matrix(0, g, 6, dimnames = list(NULL, c("wy", "wd", "w2d", "wh", "y", "d")))
  for (i in seq_len(nrow(d))) for (k in seq_len(g)) {
    a <- (k - 1) * width; b <- k * width
    lo <- max(d$start[i], a); hi <- min(d$time[i], b)
    if (hi <= lo) next
    w <- if (method == "pp") exp(Hf(pc[[i]], (lo + hi) / 2)) else 1
    dd <- as.numeric(d$status[i] == 1 && hi == d$time[i])
    hh <- Hf(pc[[i]], hi) - Hf(pc[[i]], lo)
    S[k, ] <- S[k, ] + c(w * (hi - lo), w * dd, w * w * dd, w * hh, hi - lo, dd)
  }
  lam <- 0; vr <- 0; o <- list(time = c(), lambda = c(), variance = c(), n = c(), events = c())
  for (k in seq_len(g)) {
    if (S[k, "wy"] <= 0) next
    a <- (k - 1) * width; e <- min(k * width, tmax); w <- e - a
    lam <- lam + (S[k, "wd"] - S[k, "wh"]) / S[k, "wy"] * w
    vr <- vr + w^2 * S[k, "w2d"] / S[k, "wy"]^2
    o$time <- c(o$time, e); o$lambda <- c(o$lambda, lam); o$variance <- c(o$variance, vr)
    o$n <- c(o$n, S[k, "y"] / w); o$events <- c(o$events, S[k, "d"])
  }
  o
}
dB <- dAe[g1e, ]
pB <- pAe[g1e]
interval_runs <- list()
for (w in c(30, 91.25)) for (m in c("pp", "e2")) {
  interval_runs[[length(interval_runs) + 1]] <- list(method = m, width = w, res = interval_ref(dB, pB, m, w))
}

# ---- 3. the piecewise constant excess hazard model ----
cuts <- c(0, 90, 180, 365, 730, 1095, 1825, 2600)
dC <- make_cohort(700, 1800, 1.1, TRUE, 3200)
# Split at the intervals and at the cuts of the table, so that each record has
# one reference hazard, as popEpi::lexpand does to merge a table of hazards.
L <- Lexis(entry = list(fot = dC$start, age = dC$age0 + dC$start, per = dC$date0 + dC$start),
           exit = list(fot = dC$time), exit.status = factor(dC$status, 0:1, c("alive", "dead")),
           data = dC)
sl <- splitMulti(L, breaks = list(fot = cuts, age = age_cuts, per = year_cuts))
invisible(sl[, pop.haz := haz[cbind(findInterval(age, age_cuts), sex, findInterval(per, year_cuts))]])
invisible(sl[, d.exp := pop.haz * lex.dur])
# glm stops at a relative change of epsilon in the deviance; the fit is then
# continued from its estimate for a fixed number of iterations, which glm
# reports as not converged, so that the reference is at the maximum to rounding.
polish <- glm.control(epsilon = 1e-300, maxit = 8)
rp <- relpois(sl, lex.Xst %in% "dead" ~ -1 + FOT + x1 + x2, check = FALSE,
              control = glm.control(epsilon = 1e-14, maxit = 100))
rp <- suppressWarnings(relpois(sl, lex.Xst %in% "dead" ~ -1 + FOT + x1 + x2, check = FALSE,
                               start = coef(rp), control = polish))
used <- !is.na(cut(sl$fot, cuts, right = FALSE))
rpl <- list(coef = unname(coef(rp)), vcov = unname(vcov(rp)), deviance = rp$deviance,
            n_records = sum(used), iter = rp$iter, coef_names = names(coef(rp)))
# The same link on records aggregated by interval and covariate pattern.
ag <- as.data.frame(sl[used, .(d = sum(lex.Xst == "dead"), dstar = sum(d.exp), y = sum(lex.dur)),
                       by = .(k = findInterval(fot, cuts), x2, x3)])
ag <- ag[order(ag$k, ag$x2, ag$x3), ]
ag$FOT <- factor(ag$k)
fam <- poisson()
fam$link <- "log(mu - d*)"
fam$linkfun <- function(mu) log(mu - ag$dstar)
fam$linkinv <- function(eta) ag$dstar + exp(eta)
fam$initialize <- expression({ n <- rep.int(1, nobs); mustart <- pmax(y, ag$dstar) + 0.1 })
ga <- glm(d ~ -1 + FOT + x2 + factor(x3), data = ag, offset = log(y), family = fam,
          control = glm.control(epsilon = 1e-14, maxit = 100))
ga <- suppressWarnings(glm(d ~ -1 + FOT + x2 + factor(x3), data = ag, offset = log(y), family = fam,
                           start = coef(ga), control = polish))
gal <- list(coef = unname(coef(ga)), vcov = unname(vcov(ga)), deviance = ga$deviance,
            n_cells = nrow(ag), coef_names = names(coef(ga)))

# ---- 4. rstpm2::stpm2 with bhazard ----
yr <- 365.25
make_bh <- function(d, pc) {
  d$time_y <- d$time / yr; d$start_y <- d$start / yr
  d$bh <- sapply(seq_len(nrow(d)), function(i) hf(pc[[i]], d$time[i]) * yr)
  d
}
dD <- make_cohort(600, 1800, 1.1, FALSE, 3000)
dD <- make_bh(dD, pcs_of(dD))
dE <- make_cohort(600, 1800, 1.1, TRUE, 3000)
dE <- make_bh(dE, pcs_of(dE))
ctl <- list(reltol.search = 1e-14, reltol.final = 1e-14)
times10 <- c(0.1, 0.3, 0.6, 1, 1.5, 2, 3, 4, 5, 6)
nd <- data.frame(x1 = c(0.5, -0.7), x2 = c(1, 0))
rst_preds <- function(f) {
  pr <- function(type) lapply(1:2, function(p) {
    nn <- nd[p, , drop = FALSE][rep(1, length(times10)), ]; nn$time_y <- times10
    q <- predict(f, newdata = nn, type = type, se.fit = TRUE)
    list(Estimate = q[["Estimate"]], lower = q[["lower"]], upper = q[["upper"]])
  })
  list(surv = pr("surv"), hazard = pr("hazard"))
}
stpm_run <- function(d, entry, df) {
  fm <- as.formula(paste(if (entry) "Surv(start_y, time_y, status)" else "Surv(time_y, status)", "~ x1 + x2"))
  f <- stpm2(fm, data = d, df = df, bhazard = d$bh, control = ctl)
  list(df = df, entry = entry, cohort = pack(d), bh = d$bh,
       loglik = as.numeric(logLik(f)), coef = unname(coef(f)), coef_names = names(coef(f)),
       se = unname(sqrt(diag(vcov(f)))), preds = rst_preds(f))
}
stpm_runs <- list(stpm_run(dD, FALSE, 3), stpm_run(dD, FALSE, 4), stpm_run(dE, TRUE, 3))

write(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  relsurv = as.character(packageVersion("relsurv")), popEpi = as.character(packageVersion("popEpi")),
                  Epi = as.character(packageVersion("Epi")), rstpm2 = as.character(packageVersion("rstpm2")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  age_cuts = I(age_cuts), year_cuts = I(year_cuts), rates = I(as.vector(haz)),
  cohort_a = pack(dA), relsurv = relsurv_runs, relsurv_rule_deviation = rule_deviation,
  cohort_e = pack(dAe), exact = exact_runs, identity_deviation = identity_deviation,
  interval = interval_runs,
  cohort_c = pack(dC), cuts = I(cuts), relpois = rpl, aggregated = gal,
  stpm = stpm_runs),
  auto_unbox = TRUE, digits = 17, null = "null", na = "null"),
  "tests/data/relsurv.json")
