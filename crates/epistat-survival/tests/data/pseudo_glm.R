# Regression on pseudo-observations: geepack::geese.fit with independence working
# correlation and a gaussian variance on the pseudo-values of survival::pseudo
# (the infinitesimal jackknife), and eventglm's cumincglm and rmeanglm, whose
# pseudo-values (rawPO) and design the Rust fit is given as they are.
#
# Data: simulated, two competing causes, a binary and a continuous covariate,
# tied integer times. Every fit is made to a tolerance of 1e-13 so that R's
# stopping rule leaves less than that in a coefficient.
# Run: Rscript pseudo_glm.R > pseudo_glm.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(geepack)
  library(eventglm)
})
set.seed(20261122)
v <- function(x) I(as.numeric(x))
n <- 160
x1 <- rbinom(n, 1, 0.5)
x2 <- round(rnorm(n), 3)
time <- pmin(pmax(1, ceiling(rexp(n, 0.1 * exp(0.4 * x1 - 0.2 * x2)))), 20)
status <- sample(0:2, n, replace = TRUE, prob = c(0.3, 0.4, 0.3))
d <- data.frame(time = time, status = status, x1 = x1, x2 = x2)
ctrl <- geese.control(epsilon = 1e-13, maxit = 100)

# geese on survival::pseudo values.
geese_case <- function(name, link, target, times) {
  multi <- target == "cif1"
  fit <- if (multi) survfit(Surv(time, factor(status, 0:2)) ~ 1, data = d)
         else survfit(Surv(time, status > 0) ~ 1, data = d)
  type <- if (target == "rmst") "auc" else "pstate"
  rf <- pseudo(fit, times, type = type, data.frame = TRUE)
  if (multi) rf <- rf[rf$state == "1", ]
  rf <- rf[order(rf[["(id)"]], rf$time), ]
  p <- matrix(rf$pseudo, nrow = n, byrow = TRUE)
  ntime <- length(times)
  # rows by subject, then time
  y <- as.numeric(t(p))
  X <- do.call(rbind, lapply(1:n, function(i) {
    do.call(rbind, lapply(seq_len(ntime), function(j) {
      c(1, if (ntime > 1) as.numeric(seq_len(ntime - 1) == j - 1) else NULL, x1[i], x2[i])
    }))
  }))
  colnames(X) <- paste0("c", seq_len(ncol(X)))
  id <- rep(seq_len(n), each = ntime)
  g <- geese.fit(x = X, y = y, id = id, mean.link = link, family = gaussian(),
                 variance = "gaussian", corstr = "independence", control = ctrl, b = NULL)
  list(name = name, link = link, target = target, times = v(times), y = v(y), x = v(t(X)),
       p = ncol(X), id = v(id), coef = v(g$beta), vcov = v(t(g$vbeta)),
       converged = !as.logical(g$error))
}

cases <- list(
  geese_case("surv_identity", "identity", "km", c(6)),
  geese_case("surv_cloglog", "cloglog", "km", c(6, 12)),
  geese_case("surv_identity_3", "identity", "km", c(4, 8, 12)),
  geese_case("cif_log", "log", "cif1", c(6, 12)),
  geese_case("cif_identity", "identity", "cif1", c(8)),
  geese_case("cif_cloglog", "cloglog", "cif1", c(8)),
  geese_case("rmst_identity", "identity", "rmst", c(12)),
  geese_case("rmst_log", "log", "rmst", c(12))
)

# eventglm: the pseudo-values it made and the fit it got from them.
eg_case <- function(name, fn, link, times, ...) {
  f <- fn(Surv(time, factor(status)) ~ x1 + x2, time = times, cause = "1", link = link,
          data = d, epsilon = 1e-13, maxit = 100, ...)
  ntime <- length(times)
  X <- f$x
  id <- if (is.null(f$cluster.id)) seq_len(nrow(X)) else f$cluster.id
  # eventglm's own vcov for one time is sandwich::vcovHC at its default, HC3,
  # which scales each row by its leverage; the covariance compared is HC0, the
  # sandwich without the scaling, which is what geese gives for several times.
  g <- f
  class(g) <- c("glm", "lm")
  V <- if (f$method == "geese") vcov(f) else sandwich::vcovHC(g, type = "HC0")
  list(name = name, link = link, method = f$method, times = v(times), y = v(f$rawPO),
       x = v(t(X)), p = ncol(X), id = v(id), coef = v(coef(f)), vcov = v(t(V)),
       vcov_hc3 = v(t(vcov(f))), converged = f$converged)
}
eg <- list(
  eg_case("cuminc_identity", cumincglm, "identity", c(8)),
  eg_case("cuminc_log", cumincglm, "log", c(8)),
  eg_case("cuminc_cloglog", cumincglm, "cloglog", c(8)),
  eg_case("cuminc_log_2", cumincglm, "log", c(5, 10)),
  eg_case("cuminc_cloglog_3", cumincglm, "cloglog", c(4, 8, 12)),
  eg_case("rmean_identity", rmeanglm, "identity", c(8)),
  eg_case("rmean_log", rmeanglm, "log", c(8))
)

versions <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
                 geepack = as.character(packageVersion("geepack")),
                 eventglm = as.character(packageVersion("eventglm")),
                 jsonlite = as.character(packageVersion("jsonlite")))
data <- list(time = v(time), status = v(status), x1 = v(x1), x2 = v(x2))
cat(toJSON(list(versions = versions, data = data, geese = cases, eventglm = eg),
           digits = I(17), auto_unbox = TRUE, na = "null"))
