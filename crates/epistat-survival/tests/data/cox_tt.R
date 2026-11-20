# Cox fits with time-transform covariates, coxph(tt =), on the lung and
# veteran data of the survival package, some with a delayed entry time
# (drawn here), case weights, strata and a cluster.
# For each fit: the coefficients and standard errors, the robust standard
# errors with a cluster, the log partial likelihood at 0 and at the fit, the
# iteration count, the number of rows of the expanded data (coxph's own,
# nrow of x = TRUE), the number of risk sets (unique event times within a
# stratum, counted here from the data), the events, and the condition number
# of the information matrix scaled to unit diagonal, which bounds the error of
# the solves.
# The transforms are written as names: "none", "log" (x * log(t)), "time"
# (x * t), "step100" (x * (t > 100)) and "obrien" (coxph's default tt, the
# logit of the rank in the risk set). A column with a transform is the
# argument of tt() in the formula.
# The versions of R, survival and jsonlite are in the file. Numbers are written
# with 17 significant digits.
# Run: Rscript cox_tt.R > cox_tt.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)

ttfun <- list(
  log = function(x, t, ...) x * log(t),
  time = function(x, t, ...) x * t,
  step100 = function(x, t, ...) x * (t > 100)
)

lung2 <- na.omit(lung[, c("time", "status", "sex", "age", "ph.ecog", "wt.loss", "inst")])
lung2$event <- as.numeric(lung2$status == 2)
lung2$start <- pmin(sample(0:60, nrow(lung2), replace = TRUE), lung2$time - 1)
lung2$weight <- round(runif(nrow(lung2), 0.5, 3), 2)
lung2$sx <- as.numeric(lung2$sex)
vet <- veteran
vet$event <- as.numeric(vet$status == 1)
vet$stratum <- as.integer(vet$celltype) - 1L
vet$weight <- round(runif(nrow(vet), 0.5, 3), 2)

# name, data, covariates, transform per covariate, ties, start, strata, weights, cluster
spec <- function(name, d, vars, tts, ties, start = FALSE, strata = NULL,
                 weights = FALSE, cluster = NULL) {
  list(name = name, d = d, vars = vars, tts = tts, ties = ties, start = start,
       strata = strata, weights = weights, cluster = cluster)
}
specs <- list(
  spec("lung_log_efron", lung2, c("sex", "age"), c("none", "log"), "efron"),
  spec("lung_time_breslow", lung2, c("sex", "age"), c("none", "time"), "breslow"),
  spec("lung_strata_log", lung2, c("age", "ph.ecog"), c("log", "none"), "efron", strata = "sx"),
  spec("veteran_strata_time", vet, c("trt", "karno"), c("none", "time"), "efron", strata = "stratum"),
  spec("veteran_strata_time_breslow", vet, c("trt", "karno"), c("none", "time"), "breslow", strata = "stratum"),
  spec("veteran_step_weights", vet, c("karno", "age"), c("step100", "none"), "efron", weights = TRUE),
  spec("lung_counting_log_weights", lung2, c("sex", "age"), c("none", "log"), "efron",
       start = TRUE, weights = TRUE),
  spec("lung_counting_strata_time", lung2, c("age", "wt.loss"), c("time", "none"), "breslow",
       start = TRUE, strata = "sx"),
  spec("lung_obrien", lung2, c("sex", "age"), c("none", "obrien"), "efron"),
  spec("lung_two_transforms", lung2, c("age", "wt.loss"), c("log", "time"), "efron"),
  spec("lung_cluster", lung2, c("sex", "age"), c("none", "log"), "efron", cluster = "inst")
)

obrien <- function(x, time, riskset, weights) {
  f <- function(x) {
    r <- rank(x)
    (r - .5) / (.5 + length(r) - r)
  }
  unlist(tapply(x, riskset, f))
}

n_risk_sets <- function(d, s) {
  strat <- if (is.null(s$strata)) rep(0, nrow(d)) else d[[s$strata]]
  total <- 0
  for (g in unique(strat)) {
    dd <- d[strat == g, ]
    total <- total + length(unique(dd$time[dd$event == 1]))
  }
  total
}

run <- function(s) {
  d <- s$d
  terms <- ifelse(s$tts == "none", s$vars, paste0("tt(", s$vars, ")"))
  lhs <- if (s$start) "Surv(start, time, event)" else "Surv(time, event)"
  rhs <- paste(c(terms, if (!is.null(s$strata)) paste0("strata(", s$strata, ")")), collapse = " + ")
  fns <- lapply(s$tts[s$tts != "none"], function(k) if (k == "obrien") obrien else ttfun[[k]])
  args <- list(formula = as.formula(paste(lhs, "~", rhs)), data = d, ties = s$ties, x = TRUE)
  if (length(fns)) args$tt <- fns
  if (s$weights) args$weights <- d$weight
  if (!is.null(s$cluster)) args$cluster <- d[[s$cluster]]
  f <- do.call(coxph, args)
  v <- f$naive.var
  if (is.null(v)) v <- f$var
  vn <- as.matrix(v)
  dd <- diag(1 / sqrt(diag(solve(vn))))
  kappa <- kappa(dd %*% solve(vn) %*% dd, exact = TRUE)
  list(
    name = s$name, vars = s$vars, tts = s$tts, ties = s$ties,
    use_start = s$start, use_weights = s$weights,
    strata = if (is.null(s$strata)) NULL else d[[s$strata]],
    cluster = if (is.null(s$cluster)) NULL else d[[s$cluster]],
    coefficients = as.numeric(f$coefficients),
    se = sqrt(diag(vn)),
    robust_se = if (is.null(s$cluster)) NULL else sqrt(diag(as.matrix(f$var))),
    loglik = as.numeric(f$loglik), iterations = f$iter,
    n_expanded = nrow(f$x), n_risk_sets = n_risk_sets(d, s),
    nevent = f$nevent, kappa = kappa
  )
}

# The data are written once per source and a fit names its source.
datasets <- list(
  lung = list(
    stop = lung2$time, event = lung2$event, start = lung2$start, weight = lung2$weight,
    columns = lung2[, c("sex", "age", "ph.ecog", "wt.loss")]
  ),
  veteran = list(
    stop = vet$time, event = vet$event, start = rep(0, nrow(vet)), weight = vet$weight,
    columns = vet[, c("trt", "karno", "age")]
  )
)
fits <- lapply(specs, function(s) {
  r <- run(s)
  r$data <- if (identical(s$d, lung2)) "lung" else "veteran"
  r
})
cat(toJSON(
  list(
    versions = list(
      r = R.version.string,
      survival = as.character(packageVersion("survival")),
      jsonlite = as.character(packageVersion("jsonlite"))
    ),
    datasets = datasets, fits = fits
  ),
  digits = I(17), auto_unbox = TRUE, null = "null", na = "null", dataframe = "columns"
))
