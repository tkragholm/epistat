# Attributable fractions of epistat-glm::attributable (logistic models) and of
# epistat-survival::standardise::attributable (Cox model), against the R
# package AF (Dahlqwist, Zetterqvist, Pawitan and Sjolander 2016), version
# 0.1.5, whose functions AFglm and AFcoxph are run from the package source.
#
# AF was archived on CRAN on 2026-05-08 and depends on packages that are not
# on CRAN for this R (ivtools), so it is not installed. Its sources, from
# https://cran.r-project.org/src/contrib/Archive/AF/AF_0.1.5.tar.gz, are read
# with source(): AFfunctions.R, AFglm.R and AFcoxph.R need only survival and
# data.table. Set AF_SRC to the directory AF/R of the unpacked tarball.
#
# Cases for the logistic estimator: cross-sectional with a confounder and an
# interaction, the same clustered, and a case-control sample, each against
# AFglm; weighted fits and a weighted clustered fit against a sandwich
# computed from the stacked estimating equations with numDeriv::jacobian, which
# shares nothing with the analytic gradient (and which reproduces AFglm
# where AFglm applies: the script stops if it does not, to 1e-6). The Cox model
# is in the survival crate, tests/data/af_cox.R.
#
# Run: AF_SRC=/path/to/AF/R Rscript af.R > af.json
suppressMessages({
  library(jsonlite); library(numDeriv); library(data.table)
})
src <- Sys.getenv("AF_SRC")
for (f in c("AFfunctions.R", "AFglm.R", "AFcoxph.R")) source(file.path(src, f))
set.seed(20261125)
expit <- function(x) 1 / (1 + exp(-x))
rows <- function(M) lapply(seq_len(nrow(M)), function(i) I(unname(M[i, ])))

# The hand sandwich of a function of the stacked estimating equations.
sandwich_var <- function(F, theta, cluster = NULL) {
  n <- nrow(F(theta))
  A <- jacobian(function(th) colMeans(F(th)), theta, method.args = list(eps = 1e-6, d = 1e-4, r = 6))
  Fm <- F(theta)
  if (!is.null(cluster)) {
    Fm <- rowsum(Fm, cluster)
    C <- nrow(Fm)
    meat <- var(Fm)
    solve(A) %*% meat %*% t(solve(A)) * C / n^2
  } else {
    solve(A) %*% var(Fm) %*% t(solve(A)) / n
  }
}

cases <- list()
n <- 1500
make <- function(n, intercept = -0.4) {
  Z <- rnorm(n); X <- rbinom(n, 1, expit(0.6 * Z - 0.3))
  Y <- rbinom(n, 1, expit(intercept + 0.9 * X + 0.5 * Z + 0.4 * X * Z))
  data.frame(Y, X, Z)
}
design <- function(d, x = d$X) { d$X <- x; model.matrix(~ X * Z, data = d) }

af_glm_case <- function(label, d, case_control, w = NULL, cluster = NULL, ref = TRUE) {
  X1 <- design(d); X0 <- design(d, 0)
  tight <- glm.control(epsilon = 1e-14, maxit = 100)
  fit <- if (is.null(w)) glm(Y ~ X * Z, family = binomial, data = d, control = tight)
         else glm(Y ~ X * Z, family = quasibinomial, data = d, weights = w, control = tight)
  beta <- coef(fit)
  V <- if (is.null(w)) vcov(fit) else summary(fit)$cov.unscaled
  y <- d$Y; ww <- if (is.null(w)) rep(1, nrow(d)) else w
  if (case_control) {
    delta <- function(b) drop((X1 - X0) %*% b)
    AFhat <- 1 - sum(ww * y * exp(-delta(beta))) / sum(ww * y)
    F <- function(th) cbind(ww * y * (exp(-delta(th[-1])) - th[1]),
                            ww * X1 * (y - expit(drop(X1 %*% th[-1]))))
    S <- sandwich_var(F, c(AFhat, beta), cluster)
    va <- S[1, 1]
  } else {
    p0 <- function(b) expit(drop(X0 %*% b))
    P <- sum(ww * y) / sum(ww); P0 <- sum(ww * p0(beta)) / sum(ww)
    AFhat <- 1 - P0 / P
    F <- function(th) cbind(ww * (y - th[1]), ww * (p0(th[-(1:2)]) - th[2]),
                            ww * X1 * (y - expit(drop(X1 %*% th[-(1:2)]))))
    S <- sandwich_var(F, c(P, P0, beta), cluster)
    g <- c(P0 / P^2, -1 / P)
    va <- drop(t(g) %*% S[1:2, 1:2] %*% g)
  }
  afpkg <- NULL
  if (ref && is.null(w)) {
    dd <- d; if (!is.null(cluster)) dd$id <- cluster
    a <- if (is.null(cluster)) AFglm(fit, data = dd, exposure = "X", case.control = case_control)
         else AFglm(fit, data = dd, exposure = "X", clusterid = "id", case.control = case_control)
    afpkg <- list(af = a$AF.est, var = drop(a$AF.var))
    cat(label, afpkg$af, AFhat, afpkg$var, va, "\n", file = stderr())
    stopifnot(abs(afpkg$af - AFhat) < 1e-12, abs(afpkg$var - va) < 1e-6 * va)
  }
  z <- qnorm(0.975)
  cases[[length(cases) + 1]] <<- list(label = label, case_control = case_control,
    y = d$Y, x = rows(X1), x0 = rows(X0), weights = w, cluster = cluster,
    af = AFhat, var = va, afpkg = afpkg, z = z)
}

d <- make(n)
af_glm_case("cross-sectional, interaction, against AFglm", d, FALSE)
id <- rep(seq_len(n / 2), each = 2)
af_glm_case("cross-sectional, clustered, against AFglm", d, FALSE, cluster = id)
# a case-control sample from a larger population
pop <- make(200000, intercept = -6)
cs <- pop[sample(which(pop$Y == 1), 400), ]; ct <- pop[sample(which(pop$Y == 0), 400), ]
cc <- rbind(cs, ct)
af_glm_case("case-control, against AFglm", cc, TRUE)
w <- runif(n, 0.4, 2.5)
af_glm_case("cross-sectional, weighted", d, FALSE, w = w)
af_glm_case("cross-sectional, weighted and clustered", d, FALSE, w = w, cluster = id)
wc <- runif(nrow(cc), 0.4, 2.5)
af_glm_case("case-control, weighted", cc, TRUE, w = wc)

# A sample with a protective exposure: a negative fraction.
pr <- make(1000); pr$Y <- rbinom(1000, 1, expit(-0.3 - 0.8 * pr$X + 0.4 * pr$Z))
af_glm_case("protective exposure", pr, FALSE)

out <- list(versions = list(R = R.version.string, AF = "0.1.5 (source)",
  numDeriv = as.character(packageVersion("numDeriv"))),
  glm = cases)
cat(toJSON(out, auto_unbox = TRUE, digits = NA, na = "string", null = "null"))
