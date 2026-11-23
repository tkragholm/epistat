# Reference ordinal regression fits from ordinal::clm, MASS::polr, rms::orm and
# VGAM::vglm, and a transcription of the Brant test.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript ordinal.R > ordinal.json
#
# Parametrisation: P(Y <= k) = F(zeta_k - x'beta - w'tau_k), as clm and polr.
# clm's nominal terms enter as P(Y <= k) = F(theta_k + w'a_k), so tau = -a.
# Cases: housing (MASS, three categories, frequency weights) with four links; a
# simulated five-category outcome with ten covariates, weights, an offset and
# clusters; a simulated four-category outcome with two category-specific
# covariates (partial proportional odds, checked against clm's nominal and
# VGAM's parallel = FALSE); and the same data for the Brant test.
suppressMessages({
  library(MASS); library(ordinal); library(sandwich); library(rms); library(VGAM)
  library(jsonlite)
})
vec <- function(m) as.vector(t(m))
ctl <- clm.control(gradTol = 1e-10, maxIter = 500)

probs <- function(fit, d, resp) {
  # Without the response column predict returns all categories.
  unname(vec(predict(fit, newdata = d[, setdiff(names(d), resp), drop = FALSE], type = "prob")$fit))
}

# housing
data(housing, package = "MASS")
X <- model.matrix(~ Infl + Type + Cont, housing)[, -1]
housing_fits <- list()
for (lk in c("logit", "probit", "cloglog", "loglog")) {
  f <- clm(Sat ~ Infl + Type + Cont, data = housing, weights = Freq, link = lk, control = ctl)
  null <- clm(Sat ~ 1, data = housing, weights = Freq, link = lk, control = ctl)
  entry <- list(link = lk, theta = unname(f$alpha), beta = unname(f$beta), loglik = as.numeric(logLik(f)),
                null_loglik = as.numeric(logLik(null)), vcov = vec(vcov(f)),
                probs = probs(f, housing, "Sat"))
  pm <- c(logit = "logistic", probit = "probit", cloglog = "cloglog", loglog = "loglog")[[lk]]
  p <- polr(Sat ~ Infl + Type + Cont, data = housing, weights = Freq, method = pm, Hess = TRUE,
            control = list(reltol = 1e-14, maxit = 1000))
  entry$polr <- list(theta = unname(p$zeta), beta = unname(p$coefficients), deviance = p$deviance,
                     vcov = vec(vcov(p)))
  housing_fits[[lk]] <- entry
}
housing_case <- list(x = unname(X), y = as.integer(housing$Sat) - 1L, weights = housing$Freq,
                     fits = unname(housing_fits))

# simulated five categories
set.seed(20261123)
n <- 2000; p <- 10
Xs <- matrix(rnorm(n * p), n, p)
Xs[, 3] <- rbinom(n, 1, 0.4); Xs[, 7] <- rbinom(n, 1, 0.2)
beta <- c(0.5, -0.4, 0.8, 0.3, 0, -0.6, 0.5, 0.2, -0.3, 0.1)
off <- rnorm(n, 0, 0.3)
G <- 500
cluster <- rep(seq_len(G), length.out = n)
frail <- rnorm(G, 0, 0.5)[cluster]
lat <- drop(Xs %*% beta) + off + frail + rlogis(n)
y <- cut(lat, c(-Inf, -1.5, -0.3, 0.8, 2, Inf), labels = FALSE)
w <- sample(1:3, n, replace = TRUE)
ds <- data.frame(y = factor(y, ordered = TRUE), Xs, w = w, off = off)
names(ds)[2:11] <- paste0("x", 1:10)
fm <- as.formula(paste("y ~", paste0("x", 1:10, collapse = " + "), "+ offset(off)"))
sim_fits <- list()
for (lk in c("logit", "probit", "cloglog")) {
  f <- clm(fm, data = ds, weights = w, link = lk, control = ctl)
  null <- clm(y ~ 1, data = ds, weights = w, link = lk, control = ctl)
  entry <- list(link = lk, theta = unname(f$alpha), beta = unname(f$beta), loglik = as.numeric(logLik(f)),
                null_loglik = as.numeric(logLik(null)), vcov = vec(vcov(f)), probs = probs(f, ds, "y"))
  if (lk == "logit") {
    # orm has no offset here, so refit without it for its reference.
    fo <- as.formula(paste("y ~", paste0("x", 1:10, collapse = " + ")))
    co <- clm(fo, data = ds, weights = w, link = lk, control = ctl)
    o <- orm(fo, data = ds, weights = w, x = TRUE, y = TRUE, maxit = 100, eps = 1e-12)
    entry$orm <- list(theta_clm = unname(co$alpha), beta_clm = unname(co$beta),
                      theta = unname(-coef(o)[1:4]), beta = unname(coef(o)[5:14]))
    # sandwich::estfun.clm leaves the offset out of the scores, so the scores
    # are written out: row i in category k has u = theta_k - eta, v = theta_(k-1) - eta.
    K <- 5; eta <- drop(Xs %*% f$beta) + off; th <- f$alpha
    sc <- matrix(0, n, (K - 1) + p)
    for (i in seq_len(n)) {
      k <- y[i]
      u <- if (k < K) th[k] - eta[i] else Inf
      v <- if (k > 1) th[k - 1] - eta[i] else -Inf
      P <- plogis(u) - plogis(v)
      lu <- dlogis(u) / P; lv <- -dlogis(v) / P
      if (k < K) sc[i, k] <- lu
      if (k > 1) sc[i, k - 1] <- lv
      sc[i, (K - 1) + seq_len(p)] <- -(lu + lv) * Xs[i, ]
    }
    sc <- sc * w
    Vc <- vcov(f); Mc <- crossprod(rowsum(sc, cluster))
    entry$cluster <- vec(Vc %*% Mc %*% Vc)
  }
  sim_fits[[lk]] <- entry
}
sim_case <- list(x = unname(Xs), y = as.integer(y) - 1L, weights = w, offset = off, cluster = cluster,
                 fits = unname(sim_fits))

# partial proportional odds, four categories
set.seed(20261124)
n <- 800
x <- rnorm(n); z1 <- rnorm(n); z2 <- rbinom(n, 1, 0.5)
lat <- 0.8 * x + 0.5 * z1 + 1.0 * z2 + rlogis(n)
y4 <- cut(lat, c(-Inf, -0.5, 0.7, 2, Inf), labels = FALSE)
d4 <- data.frame(y = factor(y4, ordered = TRUE), x, z1, z2)
pp <- clm(y ~ x, nominal = ~ z1 + z2, data = d4, control = ctl)
am <- pp$alpha.mat   # rows: (Intercept), z1, z2; columns: thresholds
vg <- vglm(y ~ x + z1 + z2, cumulative(link = "logitlink", parallel = FALSE ~ z1 + z2), data = d4)
cv <- coef(vg)
ppo_case <- list(x = cbind(x), nominal = cbind(z1, z2), y = y4 - 1L,
                 theta = unname(am[1, ]), beta = unname(pp$beta),
                 tau = unname(-vec(t(am[2:3, ]))),   # row-major (K-1) x q: tau[k, l]
                 loglik = as.numeric(logLik(pp)),
                 vglm_loglik = as.numeric(logLik(vg)),
                 vglm_theta = unname(cv[1:3]), vglm_beta = unname(-cv[4]),
                 vglm_tau = unname(-cv[5:10]))

# Brant test, transcribed: binary logistic fits of y > k, covariance of the
# stacked estimates from Brant (1990), Wald tests of equal slopes.
brant <- function(X, y, w, K) {
  n <- nrow(X); p <- ncol(X)
  fits <- lapply(seq_len(K - 1), function(k)
    glm(I(y > k) ~ X, family = binomial, weights = w, control = glm.control(epsilon = 1e-14, maxit = 100)))
  Z <- cbind(1, X)
  pi_k <- lapply(fits, fitted)
  b <- lapply(fits, function(f) coef(f)[-1])
  info <- lapply(seq_len(K - 1), function(k) t(Z) %*% (Z * (w * pi_k[[k]] * (1 - pi_k[[k]]))))
  S <- matrix(0, (K - 1) * p, (K - 1) * p)
  for (j in seq_len(K - 1)) for (k in seq_len(K - 1)) {
    lo <- min(j, k); hi <- max(j, k)
    wjk <- w * (pi_k[[hi]] - pi_k[[j]] * pi_k[[k]])
    cvjk <- solve(info[[j]]) %*% (t(Z) %*% (Z * wjk)) %*% solve(info[[k]])
    S[((j - 1) * p + 1):(j * p), ((k - 1) * p + 1):(k * p)] <- cvjk[-1, -1, drop = FALSE]
  }
  bb <- unlist(b)
  D <- NULL
  for (k in seq_len(K - 2)) {
    row <- matrix(0, p, (K - 1) * p)
    row[, ((k - 1) * p + 1):(k * p)] <- diag(p); row[, (k * p + 1):((k + 1) * p)] <- -diag(p)
    D <- rbind(D, row)
  }
  # D rows are ordered by difference k, then covariate; reorder by covariate.
  idx <- as.vector(t(matrix(seq_len((K - 2) * p), K - 2, byrow = TRUE)))
  wald <- function(rows) {
    Dr <- D[rows, , drop = FALSE]; d <- drop(Dr %*% bb); V <- Dr %*% S %*% t(Dr)
    sum(d * solve(V, d))
  }
  per <- sapply(seq_len(p), function(j) wald(((seq_len(K - 2) - 1) * p) + j))
  list(omnibus = wald(seq_len((K - 2) * p)), omnibus_df = (K - 2) * p, per_covariate = per,
       per_df = K - 2, coefficients = vec(do.call(rbind, b)))
}
brant_case <- c(list(x = cbind(x, z1, z2), y = y4 - 1L, weights = rep(1, n), classes = 4L),
                brant(cbind(x, z1, z2), y4, rep(1, n), 4))
w5 <- sample(1:3, n, replace = TRUE)
brant_weighted <- c(list(x = cbind(x, z1, z2), y = y4 - 1L, weights = w5, classes = 4L),
                    brant(cbind(x, z1, z2), y4, w5, 4))

versions <- list(r = R.version.string, ordinal = as.character(packageVersion("ordinal")),
                 MASS = as.character(packageVersion("MASS")), rms = as.character(packageVersion("rms")),
                 VGAM = as.character(packageVersion("VGAM")), sandwich = as.character(packageVersion("sandwich")))
cat(toJSON(list(versions = versions, housing = housing_case, sim = sim_case, ppo = ppo_case,
                brant = list(brant_case, brant_weighted)),
           digits = NA, auto_unbox = TRUE, na = "null"))
