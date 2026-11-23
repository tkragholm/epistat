# Reference values for epistat_inference::curve::contrast_curve: the contrast
# of a spline term against a reference value, from rms::Predict(ref.zero =
# TRUE) for a logistic model and a Cox model with a restricted cubic spline,
# and from Epi::ci.lin(ctr.mat =) for a Poisson model with a natural spline.
#
# The data are simulated here and the fitted coefficients, the covariance, the
# knots and the grid are written out, so the Rust side computes the curve from
# the same numbers. The reference is the curve R reports.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/curve.R
suppressMessages({
  library(rms)
  library(Epi)
  library(splines)
  library(survival)
  library(jsonlite)
})
set.seed(20261123)
n <- 600
x <- runif(n, 20, 80)
z <- rnorm(n)
grid <- seq(22, 78, length.out = 15)

# Predict reports the contrast on the scale of the linear predictor, with
# pointwise limits from the covariance of the fit. The reference value is the
# "Adjust to" value of the datadist the model was fitted under.
rcs_case <- function(name, fitter, knots, x0, level) {
  dd <- datadist(d); dd$limits["Adjust to", "x"] <- x0
  assign("dd", dd, envir = globalenv())
  options(datadist = "dd")
  fit <- fitter()
  p <- Predict(fit, x = grid, ref.zero = TRUE, conf.int = level)
  keep <- names(coef(fit)) %in% c("x", "x'", "x''", "x'''", "x''''")
  vc <- vcov(fit)
  list(name = name, knots = knots, reference = x0, level = level, grid = grid,
       coef = unname(coef(fit)[keep]), cov = unname(vc[keep, keep]),
       estimate = p$yhat, lower = p$lower, upper = p$upper)
}

y <- rbinom(n, 1, plogis(-0.5 + 0.04 * (x - 50) - 0.0015 * (x - 50)^2 + 0.3 * z))
d <- data.frame(y = y, x = x, z = z)
k4 <- unname(quantile(x, c(0.05, 0.35, 0.65, 0.95)))
k5 <- unname(quantile(x, c(0.05, 0.275, 0.5, 0.725, 0.95)))
tm <- rexp(n, exp(-3 + 0.03 * (x - 50) + 0.2 * z))
d$ev <- as.numeric(tm < 40)
d$tm <- pmin(tm, 40)
lrm_fit <- function() lrm(y ~ rcs(x, k4) + z, data = d, x = TRUE, y = TRUE)
cph_fit <- function() cph(Surv(tm, ev) ~ rcs(x, k5) + z, data = d)
cases <- list(
  rcs_case("lrm, 4 knots, reference at the median", lrm_fit, k4, median(x), 0.95),
  rcs_case("lrm, 4 knots, reference at 30, level 0.99", lrm_fit, k4, 30, 0.99),
  rcs_case("cph, 5 knots, reference at 40", cph_fit, k5, 40, 0.95)
)

# Natural spline in a Poisson model, with Epi::ci.lin and a contrast matrix.
cnt <- rpois(n, exp(0.5 + 0.02 * (x - 50) - 0.001 * (x - 50)^2 + 0.2 * z))
nsb <- ns(x, df = 4)
pf <- glm(cnt ~ nsb + z, family = poisson)
bmat <- function(v) unclass(predict(nsb, v))
x0 <- 35
ctr <- cbind(0, sweep(bmat(grid), 2, drop(bmat(x0))), 0)
ci <- ci.lin(pf, ctr.mat = ctr, alpha = 0.05)
nsc <- list(name = "ns, 4 df, Poisson, reference at 35", level = 0.95, reference = x0,
            grid = grid, interior = as.vector(attr(nsb, "knots")),
            boundary = as.vector(attr(nsb, "Boundary.knots")),
            coef = unname(coef(pf)[2:5]), cov = unname(vcov(pf)[2:5, 2:5]),
            estimate = unname(ci[, "Estimate"]), se = unname(ci[, "StdErr"]),
            lower = unname(ci[, "2.5%"]), upper = unname(ci[, "97.5%"]))

out <- list(versions = list(r = R.version.string, rms = as.character(packageVersion("rms")),
                            Epi = as.character(packageVersion("Epi"))),
            rcs = cases, ns = nsc)
write(toJSON(out, auto_unbox = TRUE, digits = 17, null = "null"), "tests/data/curve.json")
