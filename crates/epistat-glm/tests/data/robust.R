# Reference values for epistat_glm::robust: modified Poisson regression and
# binomial identity regression with sandwich::vcovHC and vcovCL.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript robust.R > robust.json
suppressMessages({
  library(sandwich)
  library(jsonlite)
})
set.seed(20261122)
G <- 700
sizes <- sample(1:4, G, replace = TRUE, prob = c(0.2, 0.4, 0.3, 0.1))
fam <- rep(seq_len(G), sizes)
n <- length(fam)
u <- rep(rnorm(G, 0, 0.3), sizes)
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.45)
f <- sample(0:2, n, replace = TRUE)
x <- cbind(1, x1, x2, f == 1, f == 2) * 1
p <- pmin(exp(-2.2 + 0.35 * x1 + 0.4 * x2 + 0.3 * (f == 1) - 0.2 * (f == 2) + u), 0.95)
y <- rbinom(n, 1, p)
w <- sample(1:3, n, replace = TRUE)
off <- rnorm(n, 0, 0.1)

vec <- function(m) as.vector(t(m))  # row-major

one <- function(name, X, y, weights = NULL, offset = NULL, cluster = NULL) {
  fit <- glm(y ~ X - 1, family = poisson(link = "log"), weights = weights, offset = offset)
  cf <- coef(fit)
  ok <- !is.na(cf)
  expand <- function(v) {  # v: covariance of the non-aliased coefficients
    k <- ncol(X)
    out <- matrix(NA_real_, k, k)
    out[ok, ok] <- v
    vec(out)
  }
  res <- list(name = name, x = unname(X), y = y, weights = weights, offset = offset,
              cluster = cluster, coef = unname(cf), n = nrow(X), rank = fit$rank,
              hc0 = expand(vcovHC(fit, type = "HC0")),
              hc1 = expand(vcovHC(fit, type = "HC1")),
              hc3 = expand(vcovHC(fit, type = "HC3")))
  if (!is.null(cluster)) {
    res$cl_hc1 <- expand(vcovCL(fit, cluster = cluster, type = "HC1", cadjust = TRUE))
    res$cl_hc0 <- expand(vcovCL(fit, cluster = cluster, type = "HC0", cadjust = FALSE))
  }
  res
}

cases <- list(
  one("plain", x, y),
  one("clustered", x, y, cluster = fam),
  one("weights_offset", x, y, weights = w, offset = off, cluster = fam),
  one("aliased", cbind(x, 0), y, cluster = fam)
)

# Binomial identity: a risk difference, from a start inside the unit interval.
xd <- x[, 1:3]
pd <- 0.08 + 0.05 * x2 + 0.02 * pmin(pmax(x1, -2), 2)
yd <- rbinom(n, 1, pd)
fd <- glm(yd ~ xd - 1, family = binomial(link = "identity"), start = c(0.1, 0, 0))
dif <- list(x = unname(xd), y = yd, cluster = fam, coef = unname(coef(fd)),
            iter = fd$iter,
            hc0 = vec(vcovHC(fd, type = "HC0")),
            hc1 = vec(vcovHC(fd, type = "HC1")),
            hc3 = vec(vcovHC(fd, type = "HC3")),
            cl_hc1 = vec(vcovCL(fd, cluster = fam, type = "HC1", cadjust = TRUE)))

cat(toJSON(list(r = R.version.string, sandwich = as.character(packageVersion("sandwich")),
                cases = cases, difference = dif),
           digits = NA, auto_unbox = TRUE, null = "null", na = "null"))
