# Reference values for epistat_did::sunab::fit: the interaction-weighted
# event study of Sun and Abraham as fixest 0.14.2 fits it,
# feols(y ~ sunab(g, t) | id + t), with the aggregations of
# aggregate(m, "period"), aggregate(m, "att") and aggregate(m, "cohort").
# Cases: an unbalanced weighted panel with a never-treated cohort and a unit
# seen once; a covariate with clusters coarser than the unit; no never-treated
# cohort, with the last-treated cohort as control after the periods from its
# treatment on are removed, and with it as control without that (ref.c);
# relative periods binned at both ends with the first period as a second
# reference; and bins of width two. The never treated are NA for the Rust
# side and 1000 for fixest, which takes a cohort that is not a period as never
# treated. The exact projections on the dummies (lm.fit) measure fixest's
# demeaning error.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript generate_sunab_reference.R > sunab_ref.json
suppressMessages({
  library(fixest)
  library(jsonlite)
})
stopifnot(packageVersion("fixest") == "0.14.2", packageVersion("jsonlite") == "2.0.0")
set.seed(20261125)
setFixest_nthreads(1)
TOL <- 1e-11
NEVER <- 1000
vec <- function(m) as.vector(t(m))

panel <- function(G = 160, T = 9, never = TRUE, drop = 0.15) {
  d <- expand.grid(t = 1:T, id = 0:(G - 1))
  d <- d[runif(nrow(d)) > drop, ]
  cohorts <- c(3, 5, 7, if (never) NEVER)
  g <- sample(cohorts, G, TRUE)
  d$g <- g[d$id + 1]
  d$state <- d$id %% 12
  u <- rnorm(G); tt <- rnorm(T)
  d$x1 <- rnorm(nrow(d)) + 0.3 * tt[d$t]
  treat <- d$g != NEVER & d$t >= d$g
  eff <- ifelse(treat, 1 + 0.3 * (d$t - d$g) + 0.2 * (d$g == 5), 0)
  d$y <- u[d$id + 1] + tt[d$t] + 0.5 * d$x1 + eff + rnorm(nrow(d))
  # A unit seen once, which fixest removes.
  if (never) d <- rbind(d, data.frame(t = 4, id = G, g = NEVER, state = 0, x1 = 0.1, y = 0.3))
  d
}

exact_error <- function(m, d, w) {
  kept <- obs(m)
  V <- cbind(model.matrix(m), y = d$y[kept])
  D <- cbind(model.matrix(~ factor(id) - 1, d[kept, ]), model.matrix(~ factor(t) - 1, d[kept, ]))
  ww <- if (is.null(w)) rep(1, length(kept)) else w[kept]
  ex <- apply(V, 2, function(v) lm.fit(D * sqrt(ww), v * sqrt(ww))$residuals / sqrt(ww))
  dm <- demean(V, d[kept, c("id", "t")], weights = ww, tol = TOL, iter = 100000, notes = FALSE)
  scale <- apply(V, 2, function(v) { vc <- v - weighted.mean(v, ww); sqrt(sum(ww * vc^2) / sum(ww)) })
  ev <- eigen(crossprod(D * sqrt(ww)), symmetric = TRUE, only.values = TRUE)$values
  list(error = max(sweep(abs(dm - ex), 2, scale, "/")), lambda_plus = min(ev[ev > 1e-9 * max(ev)]))
}

agg_table <- function(m, kind) {
  a <- aggregate(m, kind)
  list(names = I(rownames(a)), estimate = I(unname(a[, 1])), se = I(unname(a[, 2])))
}

# spec: control ("never", "last", "cohorts"), references (numbers, or "first"
# and "last" for fixest's .F and .L), bins (NULL, list(kind = "clamp", lo, hi)
# or list(kind = "width", width)).
# `fit` selects the rows fixest fits; the Rust side gets all of `d`.
case <- function(name, d, spec, w = NULL, cluster = ~id, covariate = FALSE, sunab_args = "",
                 fit = rep(TRUE, nrow(d))) {
  full <- d
  d <- d[fit, ]
  rhs <- paste0("sunab(g, t", sunab_args, ")", if (covariate) " + x1" else "")
  fml <- as.formula(paste("y ~", rhs, "| id + t"))
  m <- feols(fml, d, weights = w, cluster = cluster, fixef.tol = TOL, fixef.iter = 100000,
             notes = FALSE, warn = FALSE)
  V <- vcov(m, attr = TRUE)
  Vraw <- vcov(m, ssc = ssc(K.adj = FALSE, G.adj = FALSE))
  nm <- names(m$coefficients)
  mm <- model.matrix(m)
  wts <- if (is.null(m$weights)) rep(1, nrow(mm)) else m$weights
  list(name = name, spec = spec,
       unit = full$id, time = full$t, first_treated = ifelse(full$g == NEVER, NA, full$g),
       outcome = full$y, weights = w, covariate = if (covariate) full$x1 else NULL,
       clusters = if (identical(cluster, ~id)) NULL else full[[all.vars(cluster)]],
       names = I(nm), coef = I(unname(m$coefficients)), vcov = I(vec(unname(V))),
       vcov_unadjusted = I(vec(unname(Vraw))), df_k = attr(V, "df.K"), nobs = m$nobs,
       shares = I(unname(colSums(wts * sign(mm))[nm])),
       collinear = I(if (is.null(m$collin.var)) character(0) else m$collin.var),
       period = agg_table(m, "period"), att = agg_table(m, "att"), cohort = agg_table(m, "cohort"),
       exact = exact_error(m, d, w))
}

d1 <- panel()
w1 <- runif(nrow(d1), 0.5, 2)
d2 <- panel(never = FALSE)
d3 <- panel()
never <- list(control = "never", references = I(-1))
cases <- list(
  case("never_weighted", d1, never, w = w1),
  case("covariate_state", d1, never, cluster = ~state, covariate = TRUE),
  # Sun and Abraham's control when no unit is never treated: the last cohort,
  # in the periods before it is treated.
  case("last_treated", d2, list(control = "last", references = I(-1)),
       sunab_args = ", ref.c = 7", fit = d2$t < 7),
  case("cohort_reference", d2, list(control = "cohorts", cohorts = I(7), references = I(-1)),
       sunab_args = ", ref.c = 7"),
  case("clamped_first", d3, list(control = "never", references = I(c("-1", "first")),
                                 bins = list(kind = "clamp", lo = -4, hi = 3)),
       sunab_args = ', ref.p = c(-1, .F), bin.rel = list("-4" = -5:-4, "3" = 3:6)'),
  case("width_two", d3, list(control = "never", references = I(-1), bins = list(kind = "width", width = 2)),
       sunab_args = ', bin.rel = "bin::2"')
)
cat(toJSON(list(r = R.version.string, fixest = as.character(packageVersion("fixest")),
                tolerance = TOL, cases = cases),
           digits = NA, auto_unbox = TRUE, null = "null", na = "null"))
