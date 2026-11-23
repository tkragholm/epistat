# Nested case-control analysis with Samuelsen's inverse probability weights,
# checked against multipleNCC (Stoer and Samuelsen) and survival::coxph.
# Four simulated cohorts: no entry times, delayed entry, matching strata, and
# times with ties. At each case time m controls are sampled without
# replacement from the risk set (entry < t <= exit, in the case's stratum, the
# case excluded) by the script. For each cohort:
#  - the number at risk at each case time, and the inclusion probability p of
#    every subject by the formula with an explicit loop over cases
#    (p = 1 - prod over cases j whose risk set holds i, j != i, of
#    (1 - m / (n_j - 1)), 1 for a case), checked against multipleNCC::KMprob;
#  - the weighted Cox fit of the sampled controls and the cases, with weights
#    1/p and a cluster per subject, as multipleNCC::wpl fits it, checked
#    against wpl (variance = "robust"): coefficients, model-based and robust
#    covariance, log partial likelihood, iterations, and the condition number
#    of the information on the scale of the columns;
#  - the baseline hazard of that fit (basehaz, centered = FALSE) and the
#    absolute risk 1 - exp(-H0(t) exp(x' beta)) of three covariate profiles at
#    five times.
# multipleNCC's matching and left-time branches use strict inequalities between
# a control's exit and a failure time, so those cohorts have no ties between
# them; the cohort with ties has neither.
# Numbers are written with 17 significant digits; the versions of R, survival,
# multipleNCC and jsonlite are in the file.
# Run: Rscript ncc.R > ncc.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(multipleNCC)
})
set.seed(20261124)

simulate <- function(n, entry_rule, strata, round_to = NULL) {
  x1 <- round(rnorm(n), 4)
  x2 <- rbinom(n, 1, 0.4)
  grp <- if (strata) sample(1:3, n, replace = TRUE) else rep(0L, n)
  t <- rexp(n, 0.12 * exp(0.5 * x1 - 0.4 * x2 + 0.2 * (grp == 2)))
  cens <- rexp(n, 0.05)
  tau <- 5
  exit <- pmin(t, cens, tau)
  case <- as.integer(t <= pmin(cens, tau))
  if (!is.null(round_to)) exit <- pmax(round(exit / round_to) * round_to, round_to)
  else exit <- round(exit, 6)
  entry <- if (entry_rule) ifelse(runif(n) < 0.4, round(runif(n) * exit * 0.8, 6), 0) else rep(0, n)
  data.frame(entry = entry, exit = exit, case = case, grp = grp, x1 = x1, x2 = x2, id = seq_len(n))
}

sample_controls <- function(d, m) {
  state <- rep(0L, nrow(d))
  ord <- order(d$exit[d$case == 1])
  cases <- which(d$case == 1)[ord]
  for (j in cases) {
    risk <- which(d$entry < d$exit[j] & d$exit >= d$exit[j] & d$grp == d$grp[j] & seq_len(nrow(d)) != j)
    pick <- risk[sample.int(length(risk), m)]
    state[pick] <- pmax(state[pick], 1L)
  }
  state[d$case == 1] <- 2L
  state
}

# n_j and p by the formula, with an explicit loop over the cases.
inclusion <- function(d, m, matched) {
  cases <- which(d$case == 1)
  nj <- sapply(cases, function(j) sum(d$entry < d$exit[j] & d$exit >= d$exit[j] & d$grp == d$grp[j]))
  stopifnot(all(nj - 1 >= m))
  q <- 1 - m / (nj - 1)
  p <- sapply(seq_len(nrow(d)), function(i) {
    inr <- d$entry[i] < d$exit[cases] & d$exit[i] >= d$exit[cases] & d$grp[i] == d$grp[cases] & cases != i
    1 - prod(q[inr])
  })
  p[d$case == 1] <- 1
  list(n_risk = nj, p = p)
}

kappa_of <- function(v, x) {
  sc <- colMeans(abs(sweep(x, 2, colMeans(x))))
  D <- diag(sc, length(sc))
  kappa(solve(D %*% v %*% D), exact = TRUE)
}

make <- function(name, n, m, entry_rule, strata, round_to = NULL, check_kmprob = TRUE) {
  d <- simulate(n, entry_rule, strata, round_to)
  state <- sample_controls(d, m)
  inc <- inclusion(d, m, strata)
  inc_idx <- which(state > 0)
  if (check_kmprob) {
    args <- list(survtime = d$exit, samplestat = state, m = m)
    if (entry_rule) args$left.time <- d$entry
    if (strata) { args$match.var <- d$grp; args$match.int <- c(0, 0) }
    pk <- invisible(do.call(KMprob, args))
    stopifnot(max(abs(pk - inc$p)) < 1e-12)
  }
  s <- d[inc_idx, ]
  w <- 1 / inc$p[inc_idx]
  X <- as.matrix(s[, c("x1", "x2")])
  fit <- coxph(Surv(entry, exit, case) ~ x1 + x2, data = s, weights = w, cluster = id)
  # multipleNCC's own fit, for the coefficients and the robust variance.
  f <- if (entry_rule) Surv(entry, exit, case) ~ x1 + x2 else Surv(exit, case) ~ x1 + x2
  wargs <- list(f, data = d, samplestat = state, m = m, weight.method = "KM", variance = "robust")
  if (strata) { wargs$match.var <- cbind(d$grp); wargs$match.int <- c(0, 0) }
  wfit <- suppressWarnings(do.call(wpl, wargs))
  stopifnot(isTRUE(all.equal(unname(wfit$coefficients), unname(fit$coefficients), tolerance = 1e-9)))
  stopifnot(isTRUE(all.equal(unname(as.matrix(wfit$var)), unname(as.matrix(fit$var)), tolerance = 1e-8)))
  bh <- basehaz(fit, centered = FALSE)
  profiles <- rbind(c(0, 0), c(1, 0), c(-0.5, 1))
  times <- c(0.5, 1.2, 2.5, 3.8, 4.9)
  H0 <- sapply(times, function(t) { k <- which(bh$time <= t); if (length(k)) bh$hazard[max(k)] else 0 })
  risk <- t(sapply(seq_len(nrow(profiles)), function(i) 1 - exp(-H0 * exp(sum(profiles[i, ] * coef(fit))))))
  list(
    name = name, m = m, entry_rule = entry_rule, strata = strata,
    entry = d$entry, exit = d$exit, case = d$case, grp = d$grp, x1 = d$x1, x2 = d$x2,
    state = state, n_risk = inc$n_risk, p = inc$p,
    coefficients = unname(fit$coefficients), var_model = unname(fit$naive.var), var_robust = unname(fit$var),
    loglik = fit$loglik[2], loglik_null = fit$loglik[1], iter = fit$iter,
    kappa = kappa_of(unname(fit$naive.var), X),
    base_time = bh$time, base_hazard = bh$hazard,
    profiles = unname(profiles), times = times, risk = unname(risk)
  )
}

sets <- list(
  make("plain", 2500, 2, FALSE, FALSE),
  make("entry", 2500, 3, TRUE, FALSE),
  make("matched", 2500, 2, FALSE, TRUE),
  make("ties", 2500, 2, FALSE, FALSE, round_to = 0.05)
)
cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  multipleNCC = as.character(packageVersion("multipleNCC")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  sets = sets
), digits = I(17), auto_unbox = TRUE, na = "null", null = "null"))
