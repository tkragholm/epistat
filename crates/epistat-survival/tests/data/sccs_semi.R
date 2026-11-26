# The sample size and the semiparametric fit of epistat-survival::sccs against
# SCCS 1.8: samplesize() on the four examples of its help page and two more,
# with the same formula evaluated with unrounded normal quantiles (samplesize
# rounds them to four decimals), and semisccs() on simulated cases with two
# risk windows, one event for most cases and two for some.
#
# Mapping of the conventions, as in sccs.R: start = astart, end = aend + 1, and
# the windows of expogrp = c(0, 10) with aedrug = adrug + 27 are [0, 10) and
# [10, 28) after adrug.
# Run: Rscript sccs_semi.R > sccs_semi.json
suppressMessages({
  library(jsonlite)
  library(SCCS)
})
set.seed(20261126)

rows <- function(M) lapply(seq_len(nrow(M)), function(i) I(unname(M[i, ])))

unrounded <- function(eexpo, risk, astart, aend, p, alpha, power, eage = NULL, agegrp = NULL) {
  eage <- c(1, eage)
  es <- diff(c(astart, agegrp, aend))
  es[length(es)] <- es[length(es)] + 1
  r <- eage * risk / sum(eage * es)
  pi <- r * eexpo / (r * eexpo + 1 - r)
  v <- p * (r * eexpo + 1 - r) / ((1 - sum(p)) + sum(p * (r * eexpo + 1 - r)))
  A <- 2 * sum(v * (pi * log(eexpo) - log(r * eexpo + 1 - r)))
  B <- log(eexpo)^2 / A * sum(v * pi * (1 - pi))
  (qnorm(1 - alpha / 2) + qnorm(power) * sqrt(B))^2 / A
}
sizes <- list()
size <- function(eexpo, risk, astart, aend, p, alpha, power, eage = NULL, agegrp = NULL) {
  s <- samplesize(eexpo = eexpo, risk = risk, astart = astart, aend = aend, p = p,
                  alpha = alpha, power = power, eage = eage, agegrp = agegrp)
  sizes[[length(sizes) + 1]] <<- list(
    relative_incidence = eexpo, risk_length = risk, start = astart, end = aend + 1,
    age_cuts = I(if (is.null(agegrp)) numeric(0) else agegrp),
    age_relative_incidence = I(if (is.null(eage)) numeric(0) else eage),
    exposed = I(p), alpha = alpha, power = power,
    samplesize = s, unrounded = unrounded(eexpo, risk, astart, aend, p, alpha, power, eage, agegrp))
}
size(2.5, 21, 366, 730, 1, 0.05, 0.8)
size(2.5, 21, 366, 730, 0.75, 0.05, 0.8)
size(2.5, 21, 366, 730, c(0.50, 0.35, 0.1, 0.05), 0.05, 0.8, c(1.2, 1.6, 2.0), c(457, 548, 639))
size(2.5, 21, 366, 730, 0.75 * c(0.50, 0.35, 0.1, 0.05), 0.05, 0.8, c(1.2, 1.6, 2.0),
     c(457, 548, 639))
size(0.5, 42, 0, 3650, c(0.2, 0.3), 0.01, 0.9, 1.4, 1800)
size(1.3, 14, 100, 465, 0.6, 0.05, 0.9)

# semisccs: 200 cases observed for about a year, events on 60 distinct days.
n <- 200
astart <- sample(0:30, n, TRUE)
aend <- astart + sample(250:330, n, TRUE)
adrug <- ifelse(runif(n) < 0.8, astart + sample(0:200, n, TRUE), NA)
pool <- sort(sample(0:360, 60))
ev <- lapply(seq_len(n), function(i) {
  days <- pool[pool >= astart[i] & pool <= aend[i]]
  w <- exp(days / 200)
  if (!is.na(adrug[i])) w <- w * ifelse(days >= adrug[i] & days < adrug[i] + 10, 3,
                                        ifelse(days >= adrug[i] + 10 & days <= adrug[i] + 27, 1.5, 1))
  k <- if (runif(1) < 0.15) 2 else 1
  sort(sample(days, k, prob = w))
})
ne <- sapply(ev, length)
ds <- data.frame(indiv = rep(seq_len(n), ne), astart = rep(astart, ne), aend = rep(aend, ne),
                 aevent = unlist(ev), adrug = rep(adrug, ne), aedrug = rep(adrug + 27, ne))
sf <- semisccs(event ~ adrug, indiv = indiv, astart = astart, aend = aend, aevent = aevent,
               adrug = adrug, aedrug = aedrug, expogrp = c(0, 10), data = ds)
semi <- list(
  start = astart, end = aend + 1, event_offsets = c(0, cumsum(ne)), events = unlist(ev),
  exposure_offsets = c(0, cumsum(!is.na(adrug))), exposures = adrug[!is.na(adrug)],
  windows = list(list(from = 0, to = 10), list(from = 10, to = 28)),
  n_event_times = length(unique(unlist(ev))),
  coef = I(unname(sf$coefficients[, "coef"])), vcov = rows(sf$vcov),
  loglik = I(unname(sf$loglik)))

out <- list(versions = list(R = R.version.string, SCCS = as.character(packageVersion("SCCS"))),
            sizes = sizes, semi = semi)
cat(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"))
