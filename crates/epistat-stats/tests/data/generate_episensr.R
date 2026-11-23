# Reference results for epistat-stats::bias from episensr 2.2.0.
#
# Deterministic: the corrected tables and measures of misclass (exposure,
# exposure_pv and outcome), selection and confounders (RR, OR and RD), to
# machine precision.
#
# Probabilistic: R's draws cannot be reproduced, so each scenario is run K
# times with different seeds and the median, 2.5th and 97.5th percentiles of
# each replicate are saved; the test compares their means to the means of K
# runs of the Rust code, in standard errors of the difference. Selection is
# compared on the systematic error only: episensr::probsens_sel multiplies the
# standard error by a uniform draw where the other functions use a standard
# normal. The confounding scenario is a direct simulation in R, since
# probsens_conf also resamples the cell counts by binomial draws, which the
# Rust code does not do.
#
# Run from the crate directory:
#   Rscript tests/data/generate_episensr.R > tests/data/episensr_reference.json
suppressMessages({library(jsonlite); library(episensr)})
set.seed(20261123)
tables <- list(
  matrix(c(215, 83, 1021, 1202), 2, byrow = TRUE),
  matrix(c(118, 78, 1000, 1000), 2, byrow = TRUE),
  matrix(c(940, 410, 5200, 4400), 2, byrow = TRUE)
)
cells <- function(m) as.numeric(t(m))  # a, b, c, d
quiet <- function(expr) suppressMessages(suppressWarnings(expr))
# Parameters that give a cell below 1 stop episensr; those are left out.
attempt <- function(expr) tryCatch(quiet(expr), error = function(e) NULL)

det <- list()
add <- function(kind, tab, parms, res, extra = list()) {
  det[[length(det) + 1]] <<- c(list(kind = kind, table = cells(tab), parms = parms,
    corrected = cells(res$corr_data %||% res$cfder_data)), extra)
}
`%||%` <- function(a, b) if (is.null(a)) b else a
for (tab in tables) {
  for (bp in list(c(.9, .85, .95, .9), c(.85, .85, .97, .97), c(.95, .8, .99, .93))) {
    for (type in c("exposure", "outcome")) {
      r <- attempt(misclass(tab, type = type, bias_parms = bp))
      if (!is.null(r)) add(type, tab, bp, r, list(rr = r$adj_measures[1, 1], or = r$adj_measures[2, 1]))
    }
  }
  r <- quiet(misclass(tab, type = "exposure_pv", bias_parms = c(.9, .85, .95, .92)))
  add("exposure_pv", tab, c(.9, .85, .95, .92), r, list(rr = r$adj_measures[1, 1], or = r$adj_measures[2, 1]))
  r <- quiet(selection(tab, bias_parms = c(.9, .7, .8, .95)))
  add("selection", tab, c(.9, .7, .8, .95), r, list(rr = r$adj_measures[1, 1], or = r$adj_measures[2, 1]))
  for (type in c("RR", "OR")) {
    r <- attempt(confounders(tab, type = type, bias_parms = c(2, .4, .2)))
    if (!is.null(r)) det[[length(det) + 1]] <- list(kind = paste0("confounding_", type), table = cells(tab),
      parms = c(2, .4, .2), corrected = c(cells(r$cfder_data), cells(r$nocfder_data)),
      smr = r$adj_measures[1, 1], mh = r$adj_measures[2, 1], crude = r$obs_measures[1, 1])
  }
  r <- attempt(confounders(tab, type = "RD", bias_parms = c(.03, .35, .1)))
  if (!is.null(r)) det[[length(det) + 1]] <- list(kind = "confounding_RD", table = cells(tab), parms = c(.03, .35, .1),
    corrected = c(cells(r$cfder_data), cells(r$nocfder_data)), mh = r$adj_measures[1, 1],
    crude = r$obs_measures[1, 1])
}

K <- 10; reps <- 50000
summ <- function(x) c(median(x), quantile(x, .025, names = FALSE), quantile(x, .975, names = FALSE))
rep_run <- function(f) {
  out <- lapply(seq_len(K), function(k) { set.seed(1000 + k); f() })
  out
}
tab <- tables[[1]]
prob <- list()
# 1. exposure, non-differential, trapezoidal Se and triangular Sp
r1 <- rep_run(function() { x <- quiet(probsens(tab, type = "exposure", reps = reps,
  seca = list("trapezoidal", c(.75, .85, .95, 1)), spca = list("triangular", c(.9, 1, .97))))
  list(syst_rr = x$adj_measures[1, ], tot_rr = x$adj_measures[2, ], syst_or = x$adj_measures[3, ],
       tot_or = x$adj_measures[4, ], draws = nrow(x$sim_df)) })
# 2. exposure, differential with correlated draws
r2 <- rep_run(function() { x <- quiet(probsens(tab, type = "exposure", reps = reps,
  seca = list("uniform", c(.7, .9)), seexp = list("uniform", c(.65, .85)),
  spca = list("beta", c(200, 12)), spexp = list("beta", c(180, 10)), corr_se = .8, corr_sp = .8))
  list(syst_rr = x$adj_measures[1, ], tot_rr = x$adj_measures[2, ], syst_or = x$adj_measures[3, ],
       tot_or = x$adj_measures[4, ], draws = nrow(x$sim_df)) })
# 3. outcome, non-differential, truncated normal Se and uniform Sp
r3 <- rep_run(function() { x <- quiet(probsens(tab, type = "outcome", reps = reps,
  seca = list("normal", c(.7, .95, .85, .05)), spca = list("uniform", c(.95, 1))))
  list(syst_rr = x$adj_measures[1, ], tot_rr = x$adj_measures[2, ], syst_or = x$adj_measures[3, ],
       tot_or = x$adj_measures[4, ], draws = nrow(x$sim_df)) })
# 4. selection, systematic error only
r4 <- rep_run(function() { x <- quiet(probsens_sel(tab, reps = reps,
  case_exp = list("uniform", c(.8, .95)), case_nexp = list("uniform", c(.7, .9)),
  ncase_exp = list("trapezoidal", c(.4, .5, .6, .7)), ncase_nexp = list("beta", c(30, 30))))
  list(syst_rr = x$adj_measures[1, ], syst_or = x$adj_measures[3, ], draws = nrow(x$sim_df)) })
# 5. confounding by direct simulation: RR adjusted = crude / bias factor
crude <- (tab[1, 1] / (tab[1, 1] + tab[2, 1])) / (tab[1, 2] / (tab[1, 2] + tab[2, 2]))
r5 <- rep_run(function() {
  p1 <- rbeta(reps, 20, 30); p0 <- rbeta(reps, 10, 40); R <- rlnorm(reps, log(2), .25)
  adj <- crude / ((p1 * (R - 1) + 1) / (p0 * (R - 1) + 1))
  list(syst = summ(adj))
})
cat(toJSON(list(
  versions = list(R = as.character(getRversion()), episensr = as.character(packageVersion("episensr"))),
  K = K, reps = reps, table = cells(tab), deterministic = det,
  probabilistic = list(exposure_nondiff = r1, exposure_diff = r2, outcome = r3, selection = r4, confounding = r5)
), digits = NA, auto_unbox = TRUE, na = "null"))
