# Reference E-values for a risk difference and for selection.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/evalue.R
#
# EValue::evalues.RD on tables of different size and risk, with true values
# 0 and above, at alpha 0.05 and 0.1, and one table whose interval contains
# the true value. evalues.RD finds the bias factor of the lower limit on a
# grid of step `grid` (1e-4); the script records the grid.
# EValue::svalues.RR on a grid of estimates, limits and true values under
# every combination of assumptions it accepts.
library(EValue)
library(jsonlite)

quiet <- function(x) suppressMessages(suppressWarnings(x))
na_null <- function(x) if (is.null(x) || is.na(x)) NA_real_ else x

rd_cases <- list(
  list(t = c(150, 900, 9850, 89100), true = 0, alpha = 0.05),
  list(t = c(150, 900, 9850, 89100), true = 0.002, alpha = 0.05),
  list(t = c(150, 900, 9850, 89100), true = 0, alpha = 0.1),
  list(t = c(60, 40, 140, 160), true = 0, alpha = 0.05),
  list(t = c(60, 40, 140, 160), true = 0.05, alpha = 0.05),
  list(t = c(300, 120, 700, 880), true = 0, alpha = 0.05),
  list(t = c(30, 20, 70, 80), true = 0, alpha = 0.05))
rd <- lapply(rd_cases, function(x) {
  # Cell order (episensr's): a exposed cases, b unexposed cases, c exposed non-cases,
  # d unexposed non-cases.
  t <- x$t
  e <- quiet(evalues.RD(n11 = t[1], n10 = t[3], n01 = t[2], n00 = t[4],
                        true = x$true, alpha = x$alpha))
  list(table = t, true = x$true, alpha = x$alpha, grid = 1e-4,
       estimate = e$est.Evalue, lower = e$lower.Evalue)
})

assumptions <- list(
  general = list(),
  selected = list(sel_pop = TRUE),
  increased = list(risk_inc = TRUE),
  decreased = list(risk_dec = TRUE),
  s_eq_u = list(S_eq_U = TRUE),
  s_eq_u_increased = list(S_eq_U = TRUE, risk_inc = TRUE),
  s_eq_u_decreased = list(S_eq_U = TRUE, risk_dec = TRUE))
grid <- list(
  list(est = 1.5, lo = 1.26, hi = 1.78, true = 1),
  list(est = 2.4, lo = 1.1, hi = 5.2, true = 1),
  list(est = 0.6, lo = 0.45, hi = 0.8, true = 1),
  list(est = 1.3, lo = 0.9, hi = 1.9, true = 1),
  list(est = 2.0, lo = NA, hi = NA, true = 1),
  list(est = 1.8, lo = 1.4, hi = 2.3, true = 1.2),
  list(est = 1.1, lo = 1.02, hi = 1.2, true = 1.3))
sv <- list()
for (name in names(assumptions)) for (g in grid) {
  e <- quiet(do.call(svalues.RR, c(list(est = g$est, lo = g$lo, hi = g$hi, true = g$true),
                                   assumptions[[name]])))
  sv[[length(sv) + 1]] <- list(assumption = name, est = g$est, lo = na_null(g$lo),
    hi = na_null(g$hi), true = g$true,
    point = na_null(e[2, 1]), lower = na_null(e[2, 2]), upper = na_null(e[2, 3]))
}

out <- list(
  versions = list(r = R.version.string,
    evalue = as.character(packageVersion("EValue")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  rd = rd, selection = sv)
write_json(out, "tests/data/evalue.json", digits = I(17), auto_unbox = TRUE, na = "null")
