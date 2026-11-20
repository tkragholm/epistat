# Cox models without covariates by survival 3.8-12: coxph(y ~ 1) and
# coxph(y ~ offset(o)), with strata, case weights, entry times, clusters and
# Breslow and Efron ties, on simulated data with tied times.
# For each fit: loglik (null and fitted, one value for a model without
# coefficients), n, nevent; basehaz at the event times; predict lp, risk and
# expected on the fitted data; survfit at the event times with its standard
# error and limits, for ctype 1 and 2.
# predict.coxph with se.fit fails for a model without coefficients in R, so the
# standard errors of lp and expected are not compared with R.
# Run: Rscript cox_null.R > cox_null.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261121)
n <- 240
stop <- pmax(ceiling(rexp(n, 0.15)), 2)
start <- pmin(sample(0:5, n, replace = TRUE), stop - 1)
event <- rbinom(n, 1, 0.7)
w <- round(runif(n, 0.5, 3), 2)
stratum <- sample(0:2, n, replace = TRUE)
id <- sample(rep(1:50, length.out = n))
off <- round(rnorm(n, 0.4, 0.5), 3)
d <- data.frame(start, stop, event, w, stratum, id, off)

specs <- list(
  list(name = "efron_plain", ties = "efron"),
  list(name = "breslow_plain", ties = "breslow"),
  list(name = "efron_strata_weights_offset", ties = "efron", strata = TRUE, weights = TRUE, offset = TRUE),
  list(name = "breslow_entry_offset", ties = "breslow", entry = TRUE, offset = TRUE),
  list(name = "efron_entry_strata_weights_cluster", ties = "efron", entry = TRUE, strata = TRUE, weights = TRUE, cluster = TRUE),
  list(name = "breslow_entry_strata_weights_offset_cluster", ties = "breslow", entry = TRUE, strata = TRUE, weights = TRUE, offset = TRUE, cluster = TRUE)
)

run <- function(s) {
  resp <- if (isTRUE(s$entry)) "Surv(start, stop, event)" else "Surv(stop, event)"
  rhs <- "1"
  if (isTRUE(s$offset)) rhs <- paste(rhs, "+ offset(off)")
  if (isTRUE(s$strata)) rhs <- paste(rhs, "+ strata(stratum)")
  if (isTRUE(s$cluster)) rhs <- paste(rhs, "+ cluster(id)")
  args <- list(formula = as.formula(paste(resp, "~", rhs)), data = d, ties = s$ties)
  if (isTRUE(s$weights)) args$weights <- d$w
  fit <- do.call(coxph, args)
  sf <- survfit(fit, censor = FALSE)
  strata_counts <- if (is.null(sf$strata)) length(sf$time) else unname(sf$strata)
  alt <- if (s$ties == "efron") 1 else 2
  sf2 <- survfit(fit, censor = FALSE, ctype = alt)
  bh <- basehaz(fit, centered = TRUE)
  # basehaz lists the censoring times as well, with the cumulative hazard
  # carried forward; the event times are compared.
  stopifnot(all(abs(bh$hazard[match(paste(sf$time, rep(seq_along(strata_counts), strata_counts)), paste(bh$time, if (is.null(bh$strata)) 1 else as.integer(bh$strata)))] - sf$cumhaz) < 1e-12))
  list(
    name = s$name, ties = s$ties, entry = isTRUE(s$entry), strata = isTRUE(s$strata),
    weights = isTRUE(s$weights), offset = isTRUE(s$offset), cluster = isTRUE(s$cluster),
    loglik = fit$loglik, n = fit$n, nevent = fit$nevent,
    basehaz = list(time = sf$time, hazard = sf$cumhaz),
    lp = predict(fit, type = "lp"), risk = predict(fit, type = "risk"),
    expected = predict(fit, type = "expected"),
    survfit = list(time = sf$time, n_risk = sf$n.risk, n_event = sf$n.event, surv = sf$surv,
                   cumhaz = sf$cumhaz, std_err = sf$std.err, lower = sf$lower, upper = sf$upper,
                   strata = I(strata_counts)),
    survfit_alt = list(ctype = alt, surv = sf2$surv, cumhaz = sf2$cumhaz,
                       std_err = sf2$std.err, lower = sf2$lower, upper = sf2$upper),
    start = d$start, stop = d$stop, event = d$event, stratum = d$stratum,
    weight = d$w, offset_values = d$off, id = d$id
  )
}
res <- list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  fits = lapply(specs, run)
)
cat(toJSON(res, digits = I(17), na = "null", auto_unbox = TRUE, null = "null"))
