# Deterministic parts of mice's imputation methods, evaluated with the random
# inputs fixed so that epistat-impute can be compared to them number by number.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/impute_parts.R
#
# What is recorded:
#   norm       estimice's coefficients and (X'X)^-1, the residual sum of squares
#              and degrees of freedom, and .norm.draw's sigma* and beta* for a
#              fixed chi-square draw and fixed standard normals z
#   augment    mice:::augment's added rows, outcomes and weights, for a binary
#              and a three-level outcome
#   logreg     glm.fit on the augmented data with quasibinomial weights, its
#              cov.unscaled, and beta* = beta + chol(V) z for fixed z
#   polyreg    nnet::multinom on the augmented data, reltol 1e-14, coefficients
#              and vcov
#   pmm        the donors that .pmm.match allows (distance at most the
#              donors-th smallest) for recipients with predicted values, with
#              ties among the donors' predicted values
#   nelson_aalen  mice::nelsonaalen on data with tied event and censoring times

suppressMessages({
  library(mice)
  library(nnet)
  library(jsonlite)
})

set.seed(20261123)
out <- list(versions = list(r = R.version.string, mice = as.character(packageVersion("mice")),
                            nnet = as.character(packageVersion("nnet")),
                            survival = as.character(packageVersion("survival"))))

# ---- norm ------------------------------------------------------------------
n <- 60
x <- cbind(rnorm(n), rexp(n), rbinom(n, 1, 0.4))
y <- 1 + x %*% c(0.5, -0.3, 0.8) + rnorm(n)
xd <- cbind(1, x)
est <- mice:::estimice(xd, as.vector(y))
chi <- 52.7
z <- rnorm(4)
sigma_star <- sqrt(sum(est$r^2) / chi)
beta_star <- est$c + (t(chol(mice:::sym(est$v))) %*% z) * sigma_star
out$norm <- list(x = x, y = as.vector(y), coef = as.vector(est$c), v = est$v,
                 rss = sum(est$r^2), df = est$df, chi = chi, z = z,
                 sigma = sigma_star, beta = as.vector(beta_star))

# ---- augment ---------------------------------------------------------------
m <- 80
xa <- cbind(rnorm(m, 2, 1.5), rexp(m), rbinom(m, 1, 0.3))
dimnames(xa) <- list(NULL, c("a", "b", "c"))
aug_case <- function(yy) {
  ry <- rep(TRUE, m); ry[c(3, 9, 40)] <- FALSE
  wy <- !ry
  yy[!ry] <- NA
  a <- mice:::augment(yy, ry, xa, wy)
  list(x_rows = as.matrix(a$x[(m + 1):nrow(a$x), ]), y_aug = as.numeric(a$y[(m + 1):length(a$y)]) ,
       weight = unique(a$w[(m + 1):length(a$w)]))
}
yb <- rbinom(m, 1, plogis(0.3 * xa[, 1] - 0.5))
yn <- sample(1:3, m, replace = TRUE)
ab <- aug_case(yb)
yn_f <- factor(yn)
an <- aug_case(yn_f)
an$y_aug <- as.numeric(as.character(an$y_aug))
out$augment <- list(x = xa, binary = list(y = yb, aug = ab), nominal = list(y = yn - 1L, aug = an))

# ---- logreg ----------------------------------------------------------------
ry <- rep(TRUE, m); ry[c(3, 9, 40)] <- FALSE
yobs <- yb; yobs[!ry] <- NA
a <- mice:::augment(yobs, ry, xa, !ry)
xx <- cbind(1, as.matrix(a$x))
fit <- glm.fit(x = xx[a$ry, , drop = FALSE], y = a$y[a$ry], family = quasibinomial(link = logit), weights = a$w[a$ry])
fs <- summary.glm(fit)
v <- fs$cov.unscaled
zl <- rnorm(4)
bstar <- coef(fit) + t(chol(mice:::sym(v))) %*% zl
out$logreg <- list(x = xa, y = yb, missing = which(!ry) - 1L, coef = as.vector(coef(fit)), v = v,
                   z = zl, beta = as.vector(bstar))

# ---- polyreg ---------------------------------------------------------------
yf <- factor(yn)
yobs <- yf; yobs[!ry] <- NA
a <- mice:::augment(yobs, ry, xa, !ry)
xy <- cbind.data.frame(y = a$y, x = a$x)
mf <- multinom(y ~ ., data = xy[a$ry, , drop = FALSE], weights = a$w[a$ry], trace = FALSE,
               reltol = 1e-14, maxit = 10000, Hess = TRUE)
out$polyreg <- list(x = xa, y = yn - 1L, missing = which(!ry) - 1L, coef = unname(coef(mf)),
                    vcov = unname(vcov(mf)), classes = 3)

# ---- pmm -------------------------------------------------------------------
donor_yhat <- round(rnorm(40), 1)         # ties among the donors
targets <- c(-3, -0.97, -0.05, 0, 0.33, 0.35, 1.2, 4)
eligible <- function(t, k) {
  d <- abs(donor_yhat - t)
  ds <- sort.int(d, partial = k)
  I(which(d <= ds[k]) - 1L)
}
out$pmm <- list(donor_yhat = donor_yhat, targets = targets,
                cases = lapply(c(1, 3, 5, 12, 40), function(k)
                  list(donors = k, eligible = lapply(targets, eligible, k = k))))

# ---- nelson_aalen ----------------------------------------------------------
nn <- 200
tm <- round(rexp(nn, 0.2), 1) + 0.1       # rounding makes ties
st <- rbinom(nn, 1, 0.6)
df <- data.frame(time = tm, status = st)
bh <- survival::basehaz(survival::coxph(survival::Surv(time, status) ~ 1, data = df, ties = "breslow"))
out$nelson_aalen <- list(time = tm, status = st, hazard = mice::nelsonaalen(df, time, status),
                         breslow = bh[match(tm, bh[, "time"]), "hazard"])

write(toJSON(out, auto_unbox = TRUE, digits = NA, matrix = "rowmajor"), "tests/data/impute_parts.json")
