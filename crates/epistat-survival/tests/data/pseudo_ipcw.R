# eventglm's pseudo-observations: inverse probability of censoring weighting with
# a Cox model of the censoring (model.censoring = "coxph", ipcw.method "binder"
# and "hajek"), and the two unweighted constructions, "independent" (the exact
# jackknife of the whole sample) and "stratified" (the exact jackknife within the
# strata of the censoring covariate). The values are those of eventglm's own
# pseudo-observation functions, which cumincglm and rmeanglm call for each time.
#
# Data: simulated, two competing causes, a binary and a continuous covariate,
# tied integer times, with the censoring depending on the binary covariate.
# Targets: the cumulative incidence of both causes, the restricted mean time lost
# to a cause, and, on the single-event version of the same data, the survival
# probability and the restricted mean.
# Run: Rscript pseudo_ipcw.R > pseudo_ipcw.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(eventglm)
})
set.seed(20261126)
v <- function(x) I(as.numeric(x))
n <- 140
x1 <- rbinom(n, 1, 0.5)
x2 <- round(rnorm(n), 3)
t_event <- ceiling(rexp(n, 0.1 * exp(0.4 * x1 - 0.2 * x2)))
t_cens <- ceiling(rexp(n, 0.06 * exp(0.9 * x1 + 0.3 * x2)))
time <- pmin(pmax(1, pmin(t_event, t_cens)), 24)
cause <- sample(1:2, n, replace = TRUE, prob = c(0.65, 0.35))
status <- ifelse(t_event <= t_cens & t_event <= 24, cause, 0)
d <- data.frame(time = time, status = status, ev = as.integer(status > 0), x1 = x1, x2 = x2)

f_cr <- Surv(time, factor(status)) ~ x1 + x2
f_km <- Surv(time, ev) ~ x1 + x2

po <- function(model, target, times, cens = NULL, method = "binder") {
  fn <- get(paste0("pseudo_", model), asNamespace("eventglm"))
  formula <- if (target %in% c("surv", "rmean")) f_km else f_cr
  type <- c(cif1 = "cuminc", cif2 = "cuminc", surv = "survival", rmtl1 = "rmean", rmean = "rmean")[[target]]
  cs <- c(cif1 = 1, cif2 = 2, rmtl1 = 1, surv = 1, rmean = 1)[[target]]
  fc <- if (is.null(cens)) NULL else as.formula(cens)
  gmin <- Inf
  m <- sapply(times, function(t) {
    r <- fn(formula = formula, time = t, cause = cs, data = d, type = type,
            formula.censoring = fc, ipcw.method = method)
    if (!is.null(attr(r, "ipcw.weights"))) gmin <<- min(gmin, attr(r, "ipcw.weights"))
    as.numeric(r)
  })
  out <- list(model = model, target = target, cens = if (is.null(cens)) "default" else cens,
              method = method, times = v(times), po = v(t(m)))
  if (model == "coxph") {
    # What the bounds of the test need from the censoring model: its smallest
    # probability of not being censored, its coefficients and the 2-norm condition
    # number of its information matrix.
    rhs <- if (is.null(cens)) "~ x1 + x2" else cens
    cf <- coxph(as.formula(paste("Surv(time, status == 0)", rhs)), data = d)
    out$cens_columns <- I(all.vars(as.formula(rhs)))
    out$gmin <- gmin
    out$beta <- v(coef(cf))
    out$condition <- kappa(solve(vcov(cf)), exact = TRUE)
  }
  out
}

times <- c(4, 8, 12, 30)
cases <- list()
add <- function(...) cases[[length(cases) + 1]] <<- po(...)
for (method in c("binder", "hajek")) {
  for (target in c("cif1", "cif2", "rmtl1", "surv", "rmean")) {
    add("coxph", target, times, NULL, method)
    add("coxph", target, times, "~ x1", method)
  }
}
# A time before every exit time: eventglm falls back to the first time of the
# censoring curve, where the probability of not being censored is below 1.
add("coxph", "surv", c(0.5, 4), "~ x1", "binder")
add("coxph", "rmean", c(0.5, 4), "~ x1", "binder")
for (target in c("cif1", "cif2", "rmtl1", "surv", "rmean")) {
  add("independent", target, times)
  add("stratified", target, times, "~ x1")
}
ds <- list(time = v(d$time), status = v(d$status), x1 = v(d$x1), x2 = v(d$x2))
versions <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
                 eventglm = as.character(packageVersion("eventglm")),
                 jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = versions, data = ds, cases = cases), digits = I(17), auto_unbox = TRUE,
           na = "null"))
