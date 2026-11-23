# Shared gamma frailty Cox models against survival::coxph with
# frailty(cluster, distribution = "gamma").
#
# Each fit is coxph's penalised partial likelihood, either at a fixed theta
# (frailty(..., theta = value)) or with theta chosen by coxph's default search
# (method = "em", which maximises the corrected likelihood c.loglik). Fixed-theta
# fits are run with sparse = TRUE, whose variance treats the frailty block of the
# information as diagonal, and with sparse = FALSE, whose variance uses the whole
# information. The inner Newton iteration runs to eps = 1e-12 with a large
# iteration limit, so that the coefficients are the solution of the score
# equations to far below their standard errors; the "default" fits keep
# coxph.control() as it is, for the search path only.
#
# Two defects of coxpenal's C code (survival 3.8.12, coxfit5.c and agfit5.c)
# shape the cases. With case weights the sparse variance is wrong: the diagonal
# of the frailty block omits the weight of the events (fdiag[i] += temp2 *
# (1 - temp2), without wtave or meanwt), so weighted fits are compared with
# sparse = FALSE only. With case weights, Efron ties and (start, stop] data the
# fit does not converge: agfit5 sums risk * weights[p] into efron_wt, where risk
# already holds the weight, so its likelihood and its score disagree. That case is
# solved here instead by maximising the penalised partial likelihood written with
# coxph's own unpenalised fit (agreg.fit at iter.max = 0 for the likelihood, the
# score and martingale residuals for the gradient), by BFGS and then Newton steps
# on a numerical Hessian of the gradient, to a gradient below 1e-9.
#
# Profile fits: theta maximising c.loglik by optimize() on log theta with the
# fits run to eps = 1e-12.
#
# For each fit: the data set, ties, theta, the coefficients, var and var2 (the
# coefficient block), the frailties (log scale), fvar (sparse fits), the partial
# log likelihood at the estimate, c.loglik, and the search history (theta,
# loglik, c.loglik). The versions of R and survival are in the output. Numbers are
# written with 17 significant digits.
# Run: Rscript frailty.R > frailty.json
suppressMessages({ library(jsonlite); library(survival) })
set.seed(20261123)

simulate <- function(G, theta, entry = FALSE, strata = FALSE, weights = FALSE,
                     offset = FALSE, digits = 1) {
  size <- sample(2:6, G, replace = TRUE)
  cluster <- rep(seq_len(G), size)
  n <- length(cluster)
  z <- if (theta > 0) rgamma(G, 1 / theta, 1 / theta) else rep(1, G)
  x1 <- round(rnorm(n), 3)
  x2 <- rbinom(G, 1, 0.4)[cluster]
  st <- if (strata) rbinom(n, 1, 0.5) else rep(0, n)
  rate <- 0.2 * z[cluster] * exp(0.5 * x1 - 0.4 * x2) * ifelse(st == 1, 1.5, 1)
  tt <- rexp(n, rate)
  cc <- rexp(n, 0.1)
  time <- pmax(round(pmin(tt, cc), digits), 10^-digits)
  status <- as.numeric(tt <= cc)
  start <- rep(0, n)
  if (entry) {
    late <- runif(n) < 0.4
    start[late] <- round(runif(sum(late), 0, 0.6) * time[late], digits)
    start[start >= time] <- 0
  }
  w <- if (weights) round(runif(n, 0.5, 2.5), 2) else rep(1, n)
  o <- if (offset) round(rnorm(n, 0, 0.3), 3) else rep(0, n)
  data.frame(time, start, status, x1, x2, cluster, stratum = st, weight = w, offset = o)
}

kid <- data.frame(time = kidney$time, start = 0, status = kidney$status,
                  x1 = kidney$age, x2 = kidney$sex - 1, cluster = kidney$id,
                  stratum = 0, weight = 1, offset = 0)
datasets <- list(
  kidney = kid,
  sim50 = simulate(50, 0.6),
  sim50_entry = simulate(50, 0.6, entry = TRUE, strata = TRUE, weights = TRUE, offset = TRUE),
  sim50_weighted = simulate(50, 0.6, weights = TRUE),
  sim50_none = simulate(50, 0),
  sim500 = simulate(500, 0.4),
  sim5000 = simulate(5000, 0.4, digits = 2)
)

datasets$sim50_entry_unit <- transform(datasets$sim50_entry, weight = 1)

tight <- coxph.control(eps = 1e-12, iter.max = 500, outer.max = 30)

formula_for <- function(d, frail) {
  lhs <- if (any(d$start > 0)) "Surv(start, time, status)" else "Surv(time, status)"
  rhs <- "x1 + x2"
  if (any(d$stratum != 0)) rhs <- paste(rhs, "+ strata(stratum)")
  if (any(d$offset != 0)) rhs <- paste(rhs, "+ offset(offset)")
  as.formula(paste(lhs, "~", rhs, "+", frail))
}

run <- function(name, dname, ties, theta = NULL, sparse = TRUE, control = tight) {
  d <- datasets[[dname]]
  frail <- if (is.null(theta)) {
    sprintf("frailty(cluster, sparse = %s)", sparse)
  } else {
    sprintf("frailty(cluster, theta = %.17g, sparse = %s)", theta, sparse)
  }
  fit <- suppressWarnings(coxph(formula_for(d, frail), data = d, ties = ties,
                                weights = weight, control = control))
  p <- 2
  coef <- unname(coef(fit))[1:p]
  frailties <- if (sparse) fit$frail else unname(coef(fit))[-(1:p)]
  h <- fit$history[[1]]
  hist <- if (!is.null(h$history)) unname(as.matrix(h$history)) else NULL
  list(name = name, dataset = dname, ties = ties, sparse = sparse,
       fixed = !is.null(theta),
       theta = if (is.null(theta)) unname(hist[nrow(hist), 1]) else theta,
       coef = coef,
       var = c(fit$var[1:p, 1:p]), var2 = c(fit$var2[1:p, 1:p]),
       frail = frailties,
       fvar = if (sparse) fit$fvar else NULL,
       loglik = fit$loglik,
       c_loglik = h$c.loglik,
       history = if (is.null(hist)) NULL else lapply(seq_len(nrow(hist)), function(i) hist[i, ]),
       iter = fit$iter)
}

transcribe <- function(name, dname, ties, theta) {
  d <- datasets[[dname]]
  G <- max(d$cluster); nu <- 1 / theta; p <- 2
  fm <- Surv(start, time, status) ~ x1 + x2 + strata(stratum) + offset(off)
  pieces <- function(par) {
    d$off <- d$offset + par[-(1:p)][d$cluster]
    g <- coxph(fm, data = d, weights = weight, ties = ties, init = par[1:p],
               control = coxph.control(iter.max = 0), model = TRUE, x = TRUE, y = TRUE)
    w <- par[-(1:p)]
    list(value = g$loglik[2] - nu * sum(exp(w) - w), loglik = g$loglik[2],
         grad = c(colSums(residuals(g, "score", weighted = TRUE)),
                  rowsum(residuals(g, "martingale", weighted = TRUE), d$cluster)[, 1] -
                    nu * (exp(w) - 1)))
  }
  fn <- function(par) -pieces(par)$value
  gr <- function(par) -pieces(par)$grad
  par <- optim(rep(0, p + G), fn, gr, method = "BFGS",
               control = list(maxit = 5000, reltol = 1e-15))$par
  for (k in 1:5) {
    grad <- pieces(par)$grad
    if (max(abs(grad)) < 1e-10) break
    H <- optimHess(par, fn, gr)
    par <- par - solve(H, -grad)
  }
  last <- pieces(par)
  H <- optimHess(par, fn, gr)
  V <- solve(H)
  list(name = name, dataset = dname, ties = ties, sparse = FALSE, fixed = TRUE,
       transcribed = TRUE, theta = theta, coef = par[1:p],
       var = c(V[1:p, 1:p]), frail = par[-(1:p)], loglik = c(NA, last$loglik),
       gradient = max(abs(last$grad)))
}

profile <- function(name, dname, ties) {
  d <- datasets[[dname]]
  cll <- function(lt) {
    frail <- sprintf("frailty(cluster, theta = %.17g)", exp(lt))
    fit <- suppressWarnings(coxph(formula_for(d, frail), data = d, ties = ties,
                                  weights = weight, control = tight))
    fit$history[[1]]$c.loglik
  }
  o <- optimize(function(lt) -cll(lt), c(log(1e-3), log(10)), tol = 1e-9)
  list(name = name, dataset = dname, ties = ties, profile = TRUE,
       theta = exp(o$minimum), c_loglik = -o$objective)
}

fits <- list(
  run("kidney breslow sparse", "kidney", "breslow", 0.5),
  run("kidney breslow dense", "kidney", "breslow", 0.5, FALSE),
  run("kidney efron sparse", "kidney", "efron", 0.5),
  run("kidney efron dense", "kidney", "efron", 0.5, FALSE),
  run("kidney breslow search", "kidney", "breslow"),
  run("kidney efron search", "kidney", "efron"),
  run("sim50 breslow sparse", "sim50", "breslow", 0.6),
  run("sim50 breslow dense", "sim50", "breslow", 0.6, FALSE),
  run("sim50 efron sparse", "sim50", "efron", 0.6),
  run("sim50 efron dense", "sim50", "efron", 0.6, FALSE),
  run("sim50 breslow search", "sim50", "breslow"),
  run("sim50 efron search", "sim50", "efron"),
  run("sim50_weighted breslow dense", "sim50_weighted", "breslow", 0.6, FALSE),
  run("sim50_weighted efron dense", "sim50_weighted", "efron", 0.6, FALSE),
  run("sim50_entry breslow dense", "sim50_entry", "breslow", 0.6, FALSE),
  transcribe("sim50_entry efron transcribed", "sim50_entry", "efron", 0.6),
  run("sim50_entry_unit breslow sparse", "sim50_entry_unit", "breslow", 0.6),
  run("sim50_entry_unit breslow dense", "sim50_entry_unit", "breslow", 0.6, FALSE),
  run("sim50_entry_unit efron sparse", "sim50_entry_unit", "efron", 0.6),
  run("sim50_entry_unit efron dense", "sim50_entry_unit", "efron", 0.6, FALSE),
  run("sim50_entry_unit efron search", "sim50_entry_unit", "efron"),
  run("sim50_none breslow search", "sim50_none", "breslow"),
  run("sim50_none breslow small", "sim50_none", "breslow", 0.01, FALSE),
  run("sim500 efron sparse", "sim500", "efron", 0.4),
  run("sim500 efron dense", "sim500", "efron", 0.4, FALSE),
  run("sim500 breslow search", "sim500", "breslow"),
  run("sim5000 breslow sparse", "sim5000", "breslow", 0.4)
)
profiles <- list(
  profile("kidney breslow profile", "kidney", "breslow"),
  profile("kidney efron profile", "kidney", "efron"),
  profile("sim50_entry_unit efron profile", "sim50_entry_unit", "efron"),
  profile("sim500 breslow profile", "sim500", "breslow")
)

out <- list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  datasets = datasets,
  fits = fits,
  profiles = profiles
)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", dataframe = "columns"))
