# Reference quantiles of the confidence intervals of the fixed-effects family:
# fixest 0.14.2 (feols with i() and sunab(), aggregate), etwfe 0.6.2 with
# marginaleffects 1.0.0 (linear and Poisson, coefficients and emfx), did2s
# 1.2.1 and didimputation 0.5.1, on one simulated staggered panel with a
# covariate and clusters coarser than the unit. For each model the script
# records q = (upper - lower) / (2 se) of its 95% intervals, which names the
# reference distribution and its degrees of freedom, and for fixest's
# aggregate, which reports p-values only, the t quantile at its degrees of
# freedom.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript generate_fe_intervals_reference.R > fe_intervals_ref.json
suppressMessages({
  library(fixest)
  library(etwfe)
  library(marginaleffects)
  library(did2s)
  library(didimputation)
  library(jsonlite)
})
stopifnot(packageVersion("fixest") == "0.14.2", packageVersion("etwfe") == "0.6.2",
          packageVersion("marginaleffects") == "1.0.0", packageVersion("did2s") == "1.2.1",
          packageVersion("didimputation") == "0.5.1")
set.seed(20261127)
setFixest_nthreads(1)

G <- 120
T <- 8
d <- expand.grid(t = 1:T, id = 0:(G - 1))
g <- sample(c(3, 5, 7, NA), G, TRUE)
d$g <- g[d$id + 1]
d$cl <- d$id %% 9
u <- rnorm(G); tt <- rnorm(T)
d$x1 <- rnorm(nrow(d))
d$treat <- !is.na(d$g) & d$t >= d$g
eff <- ifelse(d$treat, 1 + 0.2 * (d$t - d$g), 0)
d$y <- u[d$id + 1] + tt[d$t] + 0.5 * d$x1 + eff + rnorm(nrow(d))
d$yc <- rpois(nrow(d), exp(0.2 + 0.1 * u[d$id + 1] + 0.05 * d$t + 0.3 * d$treat))
d$rel <- ifelse(is.na(d$g), -1000, d$t - d$g)
d$g0 <- ifelse(is.na(d$g), 0, d$g)
d$gs <- ifelse(is.na(d$g), 10000, d$g)

# q of every row of a table with estimates, standard errors and bounds; the
# rows must agree.
quantile_of <- function(lo, hi, se) {
  q <- (hi - lo) / (2 * se)
  stopifnot(max(abs(q - q[1])) < 1e-9 * q[1])
  q[1]
}
fixest_q <- function(m) {
  ci <- confint(m)
  quantile_of(ci[, 1], ci[, 2], se(m))
}
mfx_q <- function(e) quantile_of(e$conf.low, e$conf.high, e$std.error)

twfe <- feols(y ~ i(rel, ref = c(-1, -1000)) | id + t, d, cluster = ~cl, notes = FALSE)
sunab_m <- feols(y ~ sunab(gs, t) | id + t, d, cluster = ~cl, notes = FALSE)
sunab_df <- degrees_freedom(sunab_m, "t")
et_iid <- etwfe(y ~ 0, tvar = t, gvar = g0, data = d, vcov = "iid")
et_cl <- etwfe(y ~ 0, tvar = t, gvar = g0, data = d, vcov = ~cl)
et_pois <- etwfe(yc ~ 0, tvar = t, gvar = g0, data = d, family = "poisson", vcov = "iid")
d2_static <- did2s(d, yname = "y", first_stage = ~ 0 | id + t,
                   second_stage = ~ i(treat, ref = FALSE), treatment = "treat",
                   cluster_var = "cl", verbose = FALSE)
d2_event <- did2s(d, yname = "y", first_stage = ~ 0 | id + t,
                  second_stage = ~ i(rel, ref = c(-1, -1000)), treatment = "treat",
                  cluster_var = "cl", verbose = FALSE)
imp <- did_imputation(d, yname = "y", gname = "g0", tname = "t", idname = "id",
                      cluster_var = "cl")
imp_h <- did_imputation(d, yname = "y", gname = "g0", tname = "t", idname = "id",
                        cluster_var = "cl", horizon = TRUE)

out <- list(
  r = R.version.string,
  unit = d$id, time = d$t, first_treated = d$g, clusters = d$cl, outcome = d$y,
  count = d$yc,
  twfe = fixest_q(twfe),
  sunab = fixest_q(sunab_m),
  sunab_aggregate = qt(0.975, sunab_df),
  etwfe_iid = fixest_q(et_iid),
  etwfe_iid_effects = mfx_q(emfx(et_iid, "event")),
  etwfe_cluster = fixest_q(et_cl),
  etwfe_cluster_effects = mfx_q(emfx(et_cl, "event")),
  etwfe_poisson = fixest_q(et_pois),
  etwfe_poisson_effects = mfx_q(emfx(et_pois, "event", compress = FALSE)),
  did2s_static = fixest_q(d2_static),
  did2s_event = fixest_q(d2_event),
  imputation = quantile_of(imp$conf.low, imp$conf.high, imp$std.error),
  imputation_horizons = quantile_of(imp_h$conf.low, imp_h$conf.high, imp_h$std.error),
  normal = qnorm(0.975)
)
cat(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"))
