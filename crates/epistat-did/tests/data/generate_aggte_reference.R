# Reference values for the aggregation of ATT(g,t) with did::aggte.
#
# Run from the crate root:  Rscript tests/data/generate_aggte_reference.R
#
# Written for tests/parity_aggte.rs. Every scenario stores the ATT(g,t) cells and
# their influence functions as att_gt returned them, so that the Rust side
# aggregates R's own cells and what is compared is the aggregation alone. Influence
# rows are put in the order of the sorted unit ids, which is the order of
# epistat_did::unit_panel; did itself orders units by cohort.
#
# Scenarios
#   universal, varying   att_gt cells and the Wald pre-test (W, Wpval) as att_gt
#                        reports it, with and without clustering
#   max_e                aggte(max_e =) for type "simple" and "group", and
#                        "dynamic" together with balance_e
#   clustered            aggte on cells from att_gt(clustervars = "state"),
#                        analytic standard errors, and the aggregate influence
#                        functions
#   bootstrap            aggte(bstrap = TRUE, cband = TRUE, biters = 100000),
#                        unclustered and clustered, with the Monte Carlo error of
#                        the critical value
#   gap                  mpdta without 2005, where did's max_e for "simple" and
#                        "group" counts periods of the sorted period list and not
#                        years
#
# Versions: reproduces the stored fixture with R 4.6.2, did 2.5.1 and jsonlite
# 2.0.0. The bootstrap entries depend on the seed and on R's generator.

suppressMessages({
  library(did)
  library(jsonlite)
  stopifnot(packageVersion("did") == "2.5.1", packageVersion("jsonlite") == "2.0.0")
})

data(mpdta)
base_path <- "tests/data/"
biters <- 100000

units <- sort(unique(mpdta$countyreal))
unit_group <- as.numeric(tapply(mpdta$first.treat, mpdta$countyreal, function(z) z[1])[as.character(units)])
unit_state <- floor(units / 1000)

fit <- function(data, base_period, clustervars = NULL) {
  att_gt(yname = "lemp", tname = "year", idname = "countyreal", gname = "first.treat",
         xformla = ~ lpop, data = data, panel = TRUE, control_group = "nevertreated",
         base_period = base_period, est_method = "dr", bstrap = FALSE, cband = FALSE,
         clustervars = clustervars)
}

# Rows of an influence matrix in sorted unit-id order. `out` is an att_gt result.
sorted_rows <- function(m, out, ids) {
  dp <- out$DIDparams
  order_ids <- as.numeric(unique(dp$data[[dp$idname]]))
  as.matrix(m)[match(ids, order_ids), , drop = FALSE]
}

# The cells with a standard error. The (g, g - 1) cell of a universal base has
# att 0 and se NA and is left out, as epistat-did does.
cells <- function(out, ids) {
  keep <- !is.na(out$se)
  list(group = out$group[keep], t = out$t[keep], att = out$att[keep],
       inffunc = sorted_rows(out$inffunc, out, ids)[, keep, drop = FALSE])
}

pretest <- function(out) list(W = as.numeric(out$W), Wpval = as.numeric(out$Wpval))

# One aggregate: estimates, standard errors and influence functions, with the
# entries that have no standard error (the base event time) dropped.
agg <- function(a, out, ids) {
  inf <- a$inf.function
  path <- if (a$type == "simple") NULL else inf[[1]]
  overall <- if (a$type == "simple") inf[[1]] else inf[[2]]
  r <- list(overall_att = a$overall.att, overall_se = a$overall.se,
            inf_overall = as.numeric(sorted_rows(matrix(overall, ncol = 1), out, ids)))
  if (!is.null(path)) {
    keep <- !is.na(a$se.egt)
    r$egt <- a$egt[keep]; r$att_egt <- a$att.egt[keep]; r$se_egt <- a$se.egt[keep]
    r$crit_val <- a$crit.val.egt
    r$inf_path <- sorted_rows(path, out, ids)[, keep, drop = FALSE]
  }
  r
}

ag <- function(out, ids, ...) agg(aggte(out, bstrap = FALSE, cband = FALSE, ...), out, ids)

# --- universal and varying base, with the Wald pre-test ---
universal <- fit(mpdta, "universal")
varying <- fit(mpdta, "varying")
d_state <- mpdta
d_state$state <- floor(d_state$countyreal / 1000)
universal_cl <- fit(d_state, "universal", "state")
varying_cl <- fit(d_state, "varying", "state")

ref <- list(
  units = units, unit_group = unit_group, unit_state = unit_state,
  universal = c(cells(universal, units), pretest(universal),
                list(clustered_pretest = pretest(universal_cl))),
  varying = c(cells(varying, units), pretest(varying),
              list(clustered_pretest = pretest(varying_cl)))
)

# --- max_e on every type ---
max_e <- list()
for (m in c(0, 1)) {
  max_e[[length(max_e) + 1]] <- list(
    max_e = m,
    simple = ag(universal, units, type = "simple", max_e = m),
    group = ag(universal, units, type = "group", max_e = m),
    dynamic_balanced = ag(universal, units, type = "dynamic", balance_e = 1, max_e = m),
    # min_e and balance_e are ignored by "simple" and "group"
    simple_min_e = ag(universal, units, type = "simple", max_e = m, min_e = 1),
    group_balanced = ag(universal, units, type = "group", max_e = m, balance_e = 1)
  )
}
ref$max_e <- max_e

# --- clustered aggregates, analytic ---
ref$clustered <- list(
  simple = ag(universal_cl, units, type = "simple"),
  dynamic = ag(universal_cl, units, type = "dynamic"),
  group = ag(universal_cl, units, type = "group"),
  calendar = ag(universal_cl, units, type = "calendar")
)

# --- the bootstrap ---
# Both the standard error and the critical value of did:::mboot are order
# statistics of B draws, so each has a Monte Carlo error that is known from the
# draws. The standard error is (q75 - q25) / 1.349 times sqrt(G) / n. With
# densities f25 and f75 at the quartiles, Var(q_p) = p (1 - p) / (B f_p^2) and
# Cov(q25, q75) = 0.25 * 0.25 / (B f25 f75), so
#   Var(q75 - q25) = (3/16 / f25^2 + 3/16 / f75^2 - 2/16 / (f25 f75)) / B.
# The critical value is the type 1 quantile of the maximum of the studentised
# draws, with error sqrt(p (1 - p) / B) / f for the density f at the quantile.
# The densities are estimated from the quantiles at p +- 0.01 of fresh draws.
dens <- function(x, p) .02 / diff(quantile(x, c(p - .01, p + .01), type = 7))
mboot_diag <- function(inf, dp) {
  m <- did:::mboot(inf, dp, return_V = FALSE)
  b <- as.matrix(m$bres)
  nb <- nrow(b)
  iqr <- qnorm(.75) - qnorm(.25)
  sig <- apply(b, 2, function(x) { x <- sort(x); (x[ceiling(.75 * nb)] - x[ceiling(.25 * nb)]) / iqr })
  rel <- sapply(seq_len(ncol(b)), function(j) {
    f25 <- dens(b[, j], .25); f75 <- dens(b[, j], .75)
    v <- (3 / 16 / f25^2 + 3 / 16 / f75^2 - 2 / 16 / (f25 * f75)) / nb
    sqrt(v) / iqr / sig[j]
  })
  mx <- apply(abs(sweep(b, 2, sig, "/")), 1, max)
  pp <- 1 - dp$alp
  list(se_rel_mc = as.numeric(rel),
       crit_mc_se = sqrt(pp * (1 - pp) / nb) / as.numeric(dens(mx, pp)))
}
boot_one <- function(out, ids, type) {
  a <- aggte(out, type = type, bstrap = TRUE, cband = TRUE, biters = biters)
  r <- agg(a, out, ids)
  an <- aggte(out, type = type, bstrap = FALSE, cband = FALSE)
  r$analytic_overall_se <- an$overall.se
  overall <- mboot_diag(matrix(a$inf.function[[length(a$inf.function)]], ncol = 1), a$DIDparams)
  r$overall_se_rel_mc <- overall$se_rel_mc
  if (type != "simple") {
    keep <- !is.na(a$se.egt)
    r$analytic_se_egt <- an$se.egt[keep]
    d <- mboot_diag(a$inf.function[[1]][, keep, drop = FALSE], a$DIDparams)
    r$se_rel_mc <- d$se_rel_mc
    r$crit_mc_se <- d$crit_mc_se
  }
  r
}
set.seed(20261124)
ref$bootstrap <- list(
  biters = biters, alp = 0.05,
  unclustered = list(simple = boot_one(universal, units, "simple"),
                     dynamic = boot_one(universal, units, "dynamic"),
                     group = boot_one(universal, units, "group"),
                     calendar = boot_one(universal, units, "calendar")),
  clustered = list(dynamic = boot_one(universal_cl, units, "dynamic"))
)

# --- a panel with a missing period ---
gap <- subset(mpdta, year != 2005)
gap_fit <- fit(gap, "universal")
gap_units <- sort(unique(gap$countyreal))
ref$gap <- list(
  units = gap_units,
  unit_group = as.numeric(tapply(gap$first.treat, gap$countyreal, function(z) z[1])[as.character(gap_units)]),
  cells = cells(gap_fit, gap_units),
  max_e = lapply(c(0, 1, 2), function(m)
    list(max_e = m, simple = ag(gap_fit, gap_units, type = "simple", max_e = m),
         group = ag(gap_fit, gap_units, type = "group", max_e = m)))
)

write_json(ref, paste0(base_path, "aggte_ref.json"), digits = 17, auto_unbox = TRUE,
           matrix = "rowmajor", na = "null", null = "null")
cat("aggte_ref.json written\n")
