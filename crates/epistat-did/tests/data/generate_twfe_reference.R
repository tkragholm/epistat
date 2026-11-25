# Reference values for epistat_did::estimate_twfe: two-way fixed-effects
# regressions with fixest 0.14.2, static (y ~ treat | id + t) and event study
# (y ~ i(rel, ref = c(-1, -1000)) | id + t, the never treated at -1000), on
# unbalanced weighted panels with singletons, with binned end points, a
# covariate, clusters coarser than the unit and no never-treated group. The
# exact projections on the dummies (lm.fit) measure fixest's demeaning error.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript generate_twfe_reference.R > twfe_ref.json
suppressMessages({
  library(fixest)
  library(jsonlite)
})
stopifnot(packageVersion("fixest") == "0.14.2")
set.seed(20261126)
setFixest_nthreads(1)
TOL <- 1e-11
vec <- function(m) as.vector(t(m))

panel <- function(G = 160, T = 9, never = TRUE, drop = 0.15) {
  d <- expand.grid(t = 1:T, id = 0:(G - 1))
  d <- d[runif(nrow(d)) > drop, ]
  cohorts <- c(3, 5, 7, if (never) NA)
  g <- sample(cohorts, G, TRUE)
  d$g <- g[d$id + 1]
  d$state <- d$id %% 12
  u <- rnorm(G); tt <- rnorm(T)
  d$x1 <- rnorm(nrow(d)) + 0.3 * tt[d$t]
  d$treat <- as.integer(!is.na(d$g) & d$t >= d$g)
  eff <- ifelse(d$treat == 1, 1 + 0.3 * (d$t - d$g), 0)
  d$y <- u[d$id + 1] + tt[d$t] + 0.5 * d$x1 + eff + rnorm(nrow(d))
  # A unit seen once, which fixest removes.
  d <- rbind(d, data.frame(t = 4, id = G, g = NA, state = 0, x1 = 0.1, treat = 0, y = 0.3))
  d$rel <- ifelse(is.na(d$g), -1000, d$t - d$g)
  d
}

exact_error <- function(m, d, cols, w) {
  kept <- obs(m)
  V <- as.matrix(cbind(model.matrix(m)[, cols, drop = FALSE], y = d$y[kept]))
  D <- cbind(model.matrix(~ factor(id) - 1, d[kept, ]), model.matrix(~ factor(t) - 1, d[kept, ]))
  ww <- if (is.null(w)) rep(1, length(kept)) else w[kept]
  ex <- apply(V, 2, function(v) lm.fit(D * sqrt(ww), v * sqrt(ww))$residuals / sqrt(ww))
  dm <- demean(V, d[kept, c("id", "t")], weights = ww, tol = TOL, iter = 100000, notes = FALSE)
  scale <- apply(V, 2, function(v) { vc <- v - weighted.mean(v, ww); sqrt(sum(ww * vc^2) / sum(ww)) })
  ev <- eigen(crossprod(D * sqrt(ww)), symmetric = TRUE, only.values = TRUE)$values
  list(error = max(sweep(abs(dm - ex), 2, scale, "/")), lambda_plus = min(ev[ev > 1e-9 * max(ev)]))
}

case <- function(name, d, rhs, spec, w = NULL, cluster = ~id, covariate = FALSE) {
  fml <- as.formula(paste("y ~", rhs, "| id + t"))
  m <- feols(fml, d, weights = w, cluster = cluster, fixef.tol = TOL, fixef.iter = 100000,
             notes = FALSE, warn = FALSE)
  V <- vcov(m, attr = TRUE)
  Vraw <- vcov(m, ssc = ssc(K.adj = FALSE, G.adj = FALSE))
  nm <- names(coef(m))
  list(name = name, spec = spec,
       unit = d$id, time = d$t, first_treated = d$g, outcome = d$y, weights = w,
       covariate = if (covariate) d$x1 else NULL,
       clusters = if (identical(cluster, ~id)) NULL else d[[all.vars(cluster)]],
       names = I(nm), coef = I(unname(coef(m))), se = I(unname(se(m))), vcov = I(vec(unname(V))),
       vcov_unadjusted = I(vec(unname(Vraw))), df_k = attr(V, "df.K"), nobs = m$nobs,
       collinear = I(if (is.null(m$collin.var)) character(0) else m$collin.var), exact = exact_error(m, d, nm, w))
}

d1 <- panel()
w1 <- runif(nrow(d1), 0.5, 2)
d2 <- panel(never = FALSE)
d3 <- panel()
d3$relb <- ifelse(d3$rel == -1000, -1000, pmin(pmax(d3$rel, -3), 2))
cases <- list(
  case("static_weighted", d1, "treat", list(kind = "static"), w = w1),
  case("event_weighted", d1, "i(rel, ref = c(-1, -1000))", list(kind = "event", reference = -1), w = w1),
  case("event_covariate_state", d1, "i(rel, ref = c(-1, -1000)) + x1",
       list(kind = "event", reference = -1), cluster = ~state, covariate = TRUE),
  case("event_no_never", d2, "i(rel, ref = c(-1, -1000))", list(kind = "event", reference = -1)),
  case("event_binned_ref2", d3, "i(relb, ref = c(-2, -1000))",
       list(kind = "event", reference = -2, lo = -3, hi = 2))
)
cat(toJSON(list(r = R.version.string, fixest = as.character(packageVersion("fixest")),
                tolerance = TOL, cases = cases),
           digits = NA, auto_unbox = TRUE, null = "null", na = "null"))
