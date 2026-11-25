# Reference values for epistat_glm::absorb and epistat_glm::feols:
# fixest::feols with two and three fixed-effect factors, weights, singletons
# and clusters (fixest 0.14.2), and the exact projections on the dummies from
# lm.fit, against which the demeaning error of both sides is measured.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript feols.R > feols.json
suppressMessages({
  library(fixest)
  library(jsonlite)
})
stopifnot(packageVersion("fixest") == "0.14.2")
set.seed(20261125)
setFixest_nthreads(1)

vec <- function(m) as.vector(t(m))
TOL <- 1e-11

# The exact residuals of the columns of V on the dummies of every factor, by
# the QR of lm.fit, and the smallest positive eigenvalue of D'WD.
exact_projection <- function(V, fes, w) {
  D <- do.call(cbind, lapply(fes, function(f) model.matrix(~ factor(f) - 1)))
  ww <- if (is.null(w)) rep(1, nrow(V)) else w
  res <- apply(V, 2, function(v) lm.fit(D * sqrt(ww), v * sqrt(ww))$residuals / sqrt(ww))
  ev <- eigen(crossprod(D * sqrt(ww)), symmetric = TRUE, only.values = TRUE)$values
  positive <- ev[ev > 1e-9 * max(ev)]
  list(residuals = res, lambda_plus = min(positive), deficiency = sum(ev <= 1e-9 * max(ev)))
}

make <- function(name, d, fe_names, x_names, weights = NULL, clusters, ssc_cases = list()) {
  fml <- as.formula(paste("y ~", paste(x_names, collapse = " + "), "|", paste(fe_names, collapse = " + ")))
  m <- feols(fml, d, weights = weights, vcov = "iid", fixef.tol = TOL, fixef.iter = 100000,
             notes = FALSE, warn = FALSE)
  kept <- obs(m)
  dk <- d[kept, ]
  wk <- if (is.null(weights)) NULL else weights[kept]
  p <- length(x_names)
  keep <- x_names %in% names(coef(m))
  expand_vec <- function(v) { out <- rep(NA_real_, p); out[keep] <- v; out }
  expand_mat <- function(M) { out <- matrix(NA_real_, p, p); out[keep, keep] <- M; vec(out) }
  vc <- function(v, s = ssc()) {
    V <- vcov(m, vcov = v, ssc = s, attr = TRUE, vcov_fix = FALSE)
    Vfix <- vcov(m, vcov = v, ssc = s, attr = TRUE, vcov_fix = TRUE)
    list(matrix = expand_mat(V), fixed = expand_mat(Vfix), df_k = attr(V, "df.K"),
         type = if (is.character(v)) v else "cluster",
         vars = I(if (is.character(v)) character(0) else all.vars(v)),
         ssc = list(k_adj = s$K.adj, k_fixef = s$K.fixef, k_exact = s$K.exact,
                    g_adj = s$G.adj, g_df = s$G.df))
  }
  vcovs <- list(iid = vc("iid"), hetero = vc("hetero"))
  for (cl in names(clusters)) vcovs[[cl]] <- vc(clusters[[cl]])
  for (sc in names(ssc_cases)) vcovs[[sc]] <- vc(ssc_cases[[sc]]$vcov, ssc_cases[[sc]]$ssc)
  fx <- fixef(m, notes = FALSE, fixef.tol = TOL)
  fixef_values <- lapply(fe_names, function(f) {
    out <- rep(NA_real_, max(d[[f]]) + 1)
    out[as.integer(names(fx[[f]])) + 1] <- unname(fx[[f]])
    out
  })
  V <- as.matrix(dk[, c(x_names, "y")])
  fes <- lapply(fe_names, function(f) dk[[f]])
  ex <- exact_projection(V, fes, wk)
  dm <- if (is.null(wk)) demean(V, dk[, fe_names, drop = FALSE], tol = TOL, iter = 100000, notes = FALSE)
        else demean(V, dk[, fe_names, drop = FALSE], weights = wk, tol = TOL, iter = 100000, notes = FALSE)
  ww <- if (is.null(wk)) rep(1, nrow(V)) else wk
  scale <- apply(V, 2, function(v) { vc <- v - weighted.mean(v, ww); sqrt(sum(ww * vc^2) / sum(ww)) })
  fixest_error <- max(sweep(abs(dm - ex$residuals), 2, scale, "/"))
  # The coefficients of the exact projection, by least squares on the exact
  # residuals of the covariates that fixest keeps.
  xr <- ex$residuals[, seq_len(p)][, keep, drop = FALSE]
  exact_coef <- expand_vec(lm.fit(xr * sqrt(ww), ex$residuals[, p + 1] * sqrt(ww))$coefficients)
  list(name = name, factors = lapply(fe_names, function(f) d[[f]]),
       weights = weights, x = unname(as.matrix(d[, x_names])), y = d$y,
       variables = sapply(unique(unlist(lapply(clusters, all.vars))), function(v) d[[v]],
                          simplify = FALSE),
       kept = kept - 1, coef = expand_vec(coef(m)), exact_coef = exact_coef,
       nparams = m$nparams, r2_within = unname(r2(m, "wr2")), vcov = vcovs,
       fixef = fixef_values, references = unname(attr(fx, "references")),
       exact_residuals = unname(ex$residuals), lambda_plus = ex$lambda_plus,
       deficiency = ex$deficiency, fixest_error = fixest_error)
}

# An unbalanced weighted panel: units in states, periods, a covariate
# constant within units (aliased), units with one row, zero weights.
panel <- function() {
  G <- 120; T <- 8
  d <- expand.grid(t = 0:(T - 1), id = 0:(G - 1))
  d <- d[runif(nrow(d)) < 0.8, ]
  d <- rbind(d, data.frame(t = c(2, 5, 7), id = G:(G + 2)))
  d$state <- d$id %% 15
  ui <- rnorm(G + 3); tt <- rnorm(T)
  d$x1 <- rnorm(nrow(d)) + 0.5 * ui[d$id + 1]
  d$x2 <- rbinom(nrow(d), 1, plogis(ui[d$id + 1] + tt[d$t + 1]))
  d$x3 <- ui[d$id + 1]
  d$y <- 1 + 0.4 * d$x1 - 0.7 * d$x2 + ui[d$id + 1] + tt[d$t + 1] + rnorm(nrow(d)) * (1 + 0.5 * d$x2)
  w <- runif(nrow(d), 0.5, 2)
  w[sample(nrow(d), 6)] <- 0
  list(d = d, w = w)
}

# Workers who move between firms over years: three crossed factors.
three_way <- function() {
  G <- 150; T <- 6; F <- 30
  d <- expand.grid(t = 0:(T - 1), id = 0:(G - 1))
  d <- d[runif(nrow(d)) < 0.85, ]
  home <- sample(0:(F - 1), G, TRUE)
  move <- runif(nrow(d)) < 0.3
  d$firm <- ifelse(move, sample(0:(F - 1), nrow(d), TRUE), home[d$id + 1])
  ui <- rnorm(G); tt <- rnorm(T); ff <- rnorm(F)
  d$x1 <- rnorm(nrow(d)) + ff[d$firm + 1]
  d$x2 <- rnorm(nrow(d))
  d$y <- 0.3 * d$x1 + 0.2 * d$x2 + ui[d$id + 1] + tt[d$t + 1] + ff[d$firm + 1] + rnorm(nrow(d))
  d
}

# Two groups of units observed in disjoint periods: two connected components.
disconnected <- function() {
  a <- expand.grid(t = 0:3, id = 0:39)
  b <- expand.grid(t = 4:8, id = 40:89)
  d <- rbind(a, b)
  d <- d[runif(nrow(d)) < 0.9, ]
  ui <- rnorm(90); tt <- rnorm(9)
  d$x1 <- rnorm(nrow(d))
  d$x2 <- rnorm(nrow(d)) + tt[d$t + 1]
  d$y <- 0.5 * d$x1 - 0.25 * d$x2 + ui[d$id + 1] + tt[d$t + 1] + rnorm(nrow(d))
  d
}

# Few clusters in two crossed dimensions with an outcome whose errors are
# shared within each cell, where the two-way covariance is not positive
# definite and fixest repairs it.
few_clusters <- function() {
  d <- expand.grid(t = 0:5, id = 0:59)
  d$a <- d$id %% 4
  d$b <- (d$id %/% 4 + d$t) %% 3
  d$x1 <- rnorm(nrow(d)); d$x2 <- rnorm(nrow(d)); d$x3 <- rnorm(nrow(d)) + d$a
  cell <- rnorm(12)
  d$y <- 0.2 * d$x1 + 0.1 * d$x3 + 3 * cell[d$a * 3 + d$b + 1] * d$x2 + rnorm(nrow(d), sd = 0.1)
  d
}

pw <- panel()
tw <- three_way()
dc <- disconnected()
fc <- few_clusters()
cases <- list(
  make("panel_weighted", pw$d, c("id", "t"), c("x1", "x2", "x3"), weights = pw$w,
       clusters = list(id = ~id, state = ~state, id_t = ~id + t, state_t = ~state + t),
       ssc_cases = list(
         id_exact = list(vcov = ~id, ssc = ssc(K.exact = TRUE)),
         iid_exact = list(vcov = "iid", ssc = ssc(K.exact = TRUE)),
         id_t_conventional = list(vcov = ~id + t, ssc = ssc(G.df = "conventional")),
         id_none = list(vcov = ~id, ssc = ssc(K.fixef = "none")),
         id_full = list(vcov = ~id, ssc = ssc(K.fixef = "full")),
         id_noadj = list(vcov = ~id, ssc = ssc(K.adj = FALSE, G.adj = FALSE)),
         iid_noadj = list(vcov = "iid", ssc = ssc(K.adj = FALSE)),
         state_exact = list(vcov = ~state, ssc = ssc(K.exact = TRUE)))),
  make("one_way", pw$d, c("id"), c("x1", "x2"), weights = pw$w,
       clusters = list(id = ~id, state_t = ~state + t)),
  make("three_way", tw, c("id", "t", "firm"), c("x1", "x2"),
       clusters = list(id = ~id, firm = ~firm, id_firm = ~id + firm),
       ssc_cases = list(
         id_exact = list(vcov = ~id, ssc = ssc(K.exact = TRUE)),
         hetero_exact = list(vcov = "hetero", ssc = ssc(K.exact = TRUE)))),
  make("disconnected", dc, c("id", "t"), c("x1", "x2"),
       clusters = list(id = ~id, t = ~t),
       ssc_cases = list(
         iid_exact = list(vcov = "iid", ssc = ssc(K.exact = TRUE)),
         t_exact = list(vcov = ~t, ssc = ssc(K.exact = TRUE)))),
  make("few_clusters", fc, c("id", "t"), c("x1", "x2", "x3"),
       clusters = list(a_b = ~a + b),
       ssc_cases = list(a_b_conventional = list(vcov = ~a + b, ssc = ssc(G.df = "conventional"))))
)
cat(toJSON(list(r = R.version.string, fixest = as.character(packageVersion("fixest")),
                tolerance = TOL, cases = cases),
           digits = NA, auto_unbox = TRUE, null = "null", na = "null", matrix = "rowmajor"))
