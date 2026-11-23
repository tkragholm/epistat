# Reference generalised estimating equations from geepack::geeglm and gee::gee.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript gee.R > gee.json
#
# Cases: ohio (binary, 537 children, 4 waves), seizure (Poisson, 59 patients,
# 4 visits), dietox (Gaussian, 72 pigs, 12 weeks with 3 missing), and two
# simulated unbalanced panels with missing waves, prior weights and an offset.
# geeglm is run at epsilon 1e-12 so that its stopping rule does not set the
# comparison. gee is run on the balanced cases and on unbalanced data without
# waves (rows are its time order). For the Mancl-DeRouen variance the script
# transcribes the formula from the converged geeglm state.
suppressMessages({
  library(geepack)
  library(gee)
  library(MASS)
  library(sandwich)
  library(jsonlite)
})

vec <- function(m) as.vector(t(m))
ctl <- geese.control(epsilon = 1e-12, maxit = 500)

make_x <- function(X) {
  d <- as.data.frame(X)
  names(d) <- paste0("x", seq_len(ncol(X)))
  d
}

# Mancl-DeRouen robust variance at the state of a geeglm fit.
mancl <- function(fit, X, y, id, wave, w, off, family, corstr) {
  beta <- unname(coef(fit))
  eta <- drop(X %*% beta) + off
  mu <- family$linkinv(eta)
  dmu <- family$mu.eta(eta)
  v <- family$variance(mu) / w
  phi <- fit$geese$gamma
  alpha <- fit$geese$alpha
  ids <- unique(id)
  Rm <- function(wv) {
    m <- length(wv)
    if (corstr == "exchangeable") {
      R <- matrix(alpha, m, m); diag(R) <- 1; R
    } else if (corstr == "ar1") {
      alpha^abs(outer(wv, wv, "-"))
    } else diag(m)
  }
  B <- 0
  parts <- list()
  for (g in ids) {
    ix <- which(id == g)
    D <- dmu[ix] * X[ix, , drop = FALSE]
    A <- sqrt(v[ix])
    Vi <- phi * outer(A, A) * Rm(wave[ix])
    Vinv <- solve(Vi)
    B <- B + t(D) %*% Vinv %*% D
    parts[[as.character(g)]] <- list(ix = ix, D = D, Vinv = Vinv)
  }
  Binv <- solve(B)
  M <- 0
  for (pt in parts) {
    ix <- pt$ix
    e <- y[ix] - mu[ix]
    H <- pt$D %*% Binv %*% t(pt$D) %*% pt$Vinv
    ec <- solve(diag(length(ix)) - H, e)
    s <- t(pt$D) %*% pt$Vinv %*% ec
    M <- M + s %*% t(s)
  }
  Binv %*% M %*% Binv
}


# The Yan-Fine estimators of geepack written out with dense matrices, used where
# geeglm cannot run (unstructured correlation with gaps in the waves crashes it)
# and checked against geeglm on ohio below.
transcribe <- function(X, y, id, wave, w, off, family, corstr, iterations = 300) {
  n <- nrow(X); p <- ncol(X)
  beta <- unname(coef(glm.fit(X, y, weights = w, offset = off, family = family)))
  ids <- unique(id); ix <- lapply(ids, function(g) which(id == g))
  M <- max(wave)
  state <- function(beta) {
    eta <- drop(X %*% beta) + off; mu <- family$linkinv(eta)
    sres <- (y - mu) / sqrt(family$variance(mu))
    r <- sres * sqrt(w)
    phi <- sum(r^2) / sum(w)
    alpha <- numeric(0)
    # The pair (j, k) of a cluster counts with weight w_j w_k, as geeglm does.
    pairs <- function(f) {
      for (i in ix) for (a in seq_along(i)) for (b in seq_len(a - 1)) f(i[a], i[b])
    }
    if (corstr == "exchangeable") {
      S <- 0; C <- 0
      pairs(function(j, k) { S <<- S + w[j] * w[k] * sres[j] * sres[k]; C <<- C + w[j] * w[k] })
      alpha <- S / (phi * C)
    } else if (corstr == "ar1") {
      z <- numeric(M); cnt <- numeric(M)
      pairs(function(j, k) {
        d <- abs(wave[j] - wave[k])
        z[d] <<- z[d] + w[j] * w[k] * sres[j] * sres[k]; cnt[d] <<- cnt[d] + w[j] * w[k]
      })
      z <- z / phi
      g <- function(a) sum(sapply(seq_len(M), function(d) if (cnt[d] > 0) d * a^(d - 1) * (z[d] - cnt[d] * a^d) else 0))
      alpha <- uniroot(g, c(-0.999, 0.999), tol = 1e-15)$root
    } else if (corstr == "unstructured") {
      S <- matrix(0, M, M); cnt <- matrix(0, M, M)
      pairs(function(j, k) {
        lo <- min(wave[j], wave[k]); hi <- max(wave[j], wave[k])
        S[lo, hi] <<- S[lo, hi] + w[j] * w[k] * sres[j] * sres[k]; cnt[lo, hi] <<- cnt[lo, hi] + w[j] * w[k]
      })
      Rm <- S / (phi * cnt); Rm[lower.tri(Rm, TRUE)] <- 0
      alpha <- t(Rm)[lower.tri(Rm)]
    }
    list(eta = eta, mu = mu, r = r, phi = phi, alpha = alpha)
  }
  Rof <- function(wv, alpha) {
    m <- length(wv)
    if (corstr == "independence") return(diag(m))
    if (corstr == "exchangeable") { R <- matrix(alpha, m, m); diag(R) <- 1; return(R) }
    if (corstr == "ar1") return(alpha^abs(outer(wv, wv, "-")))
    Rf <- matrix(0, M, M); Rf[lower.tri(Rf)] <- alpha; Rf <- Rf + t(Rf); diag(Rf) <- 1
    Rf[wv, wv, drop = FALSE]
  }
  moments <- function(beta, st) {
    dmu <- family$mu.eta(st$eta); a <- sqrt(family$variance(st$mu) / w)
    B <- matrix(0, p, p); u <- numeric(p); Mt <- matrix(0, p, p)
    for (i in ix) {
      D <- (dmu[i] / a[i]) * X[i, , drop = FALSE]
      Ri <- solve(Rof(wave[i], st$alpha))
      s <- t(D) %*% Ri %*% st$r[i]
      B <- B + t(D) %*% Ri %*% D; u <- u + s; Mt <- Mt + s %*% t(s)
    }
    list(B = B, u = drop(u), M = Mt)
  }
  for (it in seq_len(iterations)) {
    st <- state(beta); mo <- moments(beta, st)
    step <- solve(mo$B, mo$u); beta <- beta + step
    if (max(abs(step)) < 1e-14) break
  }
  st <- state(beta); mo <- moments(beta, st); Binv <- solve(mo$B)
  list(coef = beta, robust = vec(Binv %*% mo$M %*% Binv), naive = vec(st$phi * Binv),
       scale = st$phi, alpha = st$alpha)
}

record <- function(name, family, X, y, id, wave, w = NULL, off = NULL, balanced = FALSE,
                   structures = c("independence", "exchangeable", "ar1", "unstructured"),
                   md = c("exchangeable", "ar1")) {
  n <- nrow(X)
  wv <- if (is.null(w)) rep(1, n) else w
  ov <- if (is.null(off)) rep(0, n) else off
  d <- make_x(X)
  d$y <- y; d$id <- id; d$wave <- wave; d$w <- wv; d$off <- ov
  fm <- as.formula(paste("y ~", paste(names(make_x(X)), collapse = " + "), "- 1"))
  out <- list(name = name, family = family$family, link = family$link, x = unname(X), y = y,
              cluster = id, wave = wave, weights = w, offset = off, fits = list())
  for (cs in structures) {
    message("case ", name, " ", cs)
    args <- list(formula = fm, family = family, data = d, id = d$id, waves = d$wave,
                 corstr = cs, control = ctl)
    if (!is.null(w)) args$weights <- d$w
    if (!is.null(off)) args$offset <- d$off
    gl <- if (name == "sim_unstructured") {
      tr <- transcribe(X, y, id, wave, wv, ov, family, cs)
      list(coef = tr$coef, robust = tr$robust, naive = tr$naive, scale = tr$scale, alpha = tr$alpha)
    } else NULL
    if (name != "sim_unstructured") {
      f <- do.call(geeglm, args); g <- f$geese
    }
    entry <- list(structure = cs, transcription = if (name == "ohio" || name == "sim_gauss") {
      tr <- transcribe(X, y, id, wave, wv, ov, family, cs)
      list(coef = tr$coef, robust = tr$robust, naive = tr$naive, scale = tr$scale, alpha = tr$alpha)
    } else NULL,
      geeglm = if (!is.null(gl)) gl else list(coef = unname(coef(f)), robust = vec(g$vbeta), naive = vec(g$vbeta.naiv),
                    scale = unname(g$gamma), alpha = if (cs == "independence") numeric(0) else unname(g$alpha)))
    gs <- c(independence = "independence", exchangeable = "exchangeable", ar1 = "AR-M",
            unstructured = "unstructured")[[cs]]
    if (is.null(w) && is.null(off) && (balanced || cs %in% c("independence", "exchangeable"))) {
      invisible(capture.output(gf <- gee(fm, id = id, data = d, family = family, corstr = gs, Mv = 1,
                                         tol = 1e-12, maxiter = 500, silent = TRUE)))
      wc <- gf$working.correlation
      m <- nrow(wc)
      al <- if (cs == "exchangeable") wc[1, 2] else if (cs == "ar1") wc[1, 2] else
        if (cs == "unstructured") t(wc)[lower.tri(wc)] else numeric(0)
      entry$gee <- list(coef = unname(coef(gf)), robust = vec(gf$robust.variance),
                        naive = vec(gf$naive.variance), scale = gf$scale, alpha = unname(al))
    }
    if (cs %in% md)
      entry$mancl <- vec(mancl(f, X, y, id, wave, wv, ov, family, cs))
    if (!is.null(entry$transcription)) {
      tr <- entry$transcription
      stopifnot(max(abs(tr$coef - entry$geeglm$coef)) < 1e-7,
                max(abs(tr$robust - entry$geeglm$robust)) < 1e-7 * max(abs(entry$geeglm$robust)))
    }
    out$fits[[cs]] <- entry[!sapply(entry, is.null)]
  }
  out[!sapply(out, is.null)]
}

cases <- list()

data(ohio)
X <- cbind(1, ohio$age, ohio$smoke)
cases$ohio <- record("ohio", binomial(), X, ohio$resp, ohio$id, rep(0:3, 537) + 1L, balanced = TRUE)

data(seizure)
G0 <- nrow(seizure)
s2 <- data.frame(id = rep(seq_len(G0), each = 4), visit = rep(1:4, G0),
                 y = as.vector(t(as.matrix(seizure[, c("y1", "y2", "y3", "y4")]))),
                 trt = rep(as.numeric(seizure$trt == 1), each = 4),
                 base8 = rep(seizure$base, each = 4), age = rep(seizure$age, each = 4))
Xs <- cbind(1, s2$trt, log(s2$base8 / 4), log(s2$age))
cases$seizure <- record("seizure", poisson(), Xs, s2$y, s2$id, s2$visit, balanced = TRUE)

data(dietox)
dx <- dietox[order(dietox$Pig, dietox$Time), ]
dx$id <- as.integer(factor(dx$Pig))
Xd <- cbind(1, dx$Time, dx$Cu == "Cu035", dx$Cu == "Cu175") * 1
cases$dietox <- record("dietox", gaussian(), Xd, dx$Weight, dx$id, dx$Time, structures = c("independence", "exchangeable", "ar1"))

# Unbalanced panels with missing waves.
set.seed(20261123)
G <- 300
rows <- list()
for (g in seq_len(G)) {
  keep <- which(runif(5) < 0.75)
  if (length(keep) == 0) keep <- sample(5, 1)
  rows[[g]] <- data.frame(id = g, wave = keep)
}
panel <- do.call(rbind, rows)
n <- nrow(panel)
u <- rnorm(G, 0, 0.5)[panel$id]
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.4)
Xp <- cbind(1, x1, x2)
w <- sample(1:3, n, replace = TRUE) / 2
off <- rnorm(n, 0, 0.2)
ec <- rnorm(n) + u
yg <- 1 + 0.5 * x1 - 0.7 * x2 + 0.1 * panel$wave + ec
cases$sim_gauss <- record("sim_gauss", gaussian(), Xp, yg, panel$id, panel$wave, w = w, off = off,
                          structures = c("independence", "exchangeable", "ar1"),
                          md = c("exchangeable", "ar1"))
cases$sim_pois_wide <- NULL
lam <- exp(0.2 + 0.4 * x1 - 0.3 * x2 + off + u)
yp <- rpois(n, lam)
cases$sim_pois <- record("sim_pois", poisson(), Xp, yp, panel$id, panel$wave, off = off,
                         structures = c("independence", "exchangeable", "ar1"), md = c("exchangeable"))

# An unbalanced panel of clusters with at least three of five waves, where
# geeglm's unstructured fit runs.
set.seed(20261124)
rows <- list()
for (g in seq_len(250)) rows[[g]] <- data.frame(id = g, wave = sort(sample(5, sample(3:5, 1, prob = c(0.4, 0.4, 0.2)))))
panel2 <- do.call(rbind, rows)
n2 <- nrow(panel2)
u2 <- rnorm(250, 0, 0.4)[panel2$id]
z1 <- rnorm(n2); z2 <- rbinom(n2, 1, 0.5)
X2 <- cbind(1, z1, z2)
y2 <- rpois(n2, exp(0.3 + 0.3 * z1 - 0.2 * z2 + u2))
cases$sim_unstructured <- record("sim_unstructured", poisson(), X2, y2, panel2$id, panel2$wave,
                                 structures = c("unstructured"), md = character(0))

# Independence against glm with the HC0 cluster sandwich, ohio.
fit <- glm(resp ~ age + smoke, family = binomial, data = ohio)
indep <- list(coef = unname(coef(fit)), robust = vec(vcovCL(fit, cluster = ohio$id, type = "HC0", cadjust = FALSE)))

versions <- list(r = R.version.string, geepack = as.character(packageVersion("geepack")),
                 gee = as.character(packageVersion("gee")),
                 sandwich = as.character(packageVersion("sandwich")))
cat(toJSON(list(versions = versions, cases = unname(cases), independence_glm = indep),
           digits = NA, auto_unbox = TRUE, na = "null"))
