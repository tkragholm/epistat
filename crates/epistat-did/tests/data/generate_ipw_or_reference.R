# Two-period IPW and outcome-regression DiD references, DRDID 1.3.0.
#
# Run from the crate root:  Rscript tests/data/generate_ipw_or_reference.R
#
# Writes ipw_or_two_period_ref.json: a panel and a repeated cross-section
# drawn here, each with a continuous and a binary covariate and sampling
# weights, and the results of std_ipw_did_panel, reg_did_panel,
# std_ipw_did_rc and reg_did_rc on them with the defaults (boot = FALSE,
# trim.level = 0.995), influence functions included.

suppressMessages({
  library(DRDID)
  library(jsonlite)
})
stopifnot(packageVersion("DRDID") == "1.3.0",
          packageVersion("jsonlite") == "2.0.0")
cat(R.version.string, "\n")

set.seed(20261124)

# Panel: 300 units, treatment depending on both covariates.
n <- 300
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.4)
d <- rbinom(n, 1, plogis(-0.3 + 0.6 * x1 - 0.5 * x2))
w <- runif(n, 0.5, 2)
y0 <- 1 + 0.5 * x1 + 0.8 * x2 + rnorm(n)
y1 <- y0 + 0.4 + 0.3 * x1 - 0.2 * x2 + 1.0 * d + rnorm(n)
X <- cbind(1, x1, x2)

ipw_p <- std_ipw_did_panel(y1, y0, d, X, i.weights = w, inffunc = TRUE)
reg_p <- reg_did_panel(y1, y0, d, X, i.weights = w, inffunc = TRUE)

# Repeated cross-section: 600 rows, half in each period.
m <- 600
z1 <- rnorm(m)
z2 <- rbinom(m, 1, 0.5)
post <- rep(c(0, 1), each = m / 2)
dd <- rbinom(m, 1, plogis(0.2 + 0.5 * z1 - 0.4 * z2))
ww <- runif(m, 0.5, 2)
y <- 2 + 0.6 * z1 - 0.3 * z2 + 0.5 * post + 0.4 * post * z1 + dd * (0.3 + 1.2 * post) + rnorm(m)
Z <- cbind(1, z1, z2)

ipw_r <- std_ipw_did_rc(y, post, dd, Z, i.weights = ww, inffunc = TRUE)
reg_r <- reg_did_rc(y, post, dd, Z, i.weights = ww, inffunc = TRUE)

grab <- function(fit) list(att = fit$ATT, se = fit$se, inffunc = as.vector(fit$att.inf.func))
ref <- list(
  panel = list(d = d, y0 = y0, y1 = y1, x1 = x1, x2 = x2, w = w,
               ipw = grab(ipw_p), reg = grab(reg_p)),
  rc = list(d = dd, post = post, y = y, x1 = z1, x2 = z2, w = ww,
            ipw = grab(ipw_r), reg = grab(reg_r))
)
write_json(ref, "tests/data/ipw_or_two_period_ref.json", digits = 17,
           auto_unbox = TRUE, na = "null", null = "null")
cat("ipw_or_two_period_ref.json written\n")
