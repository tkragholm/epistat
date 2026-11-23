# Case-cohort fits by survival::cch on the nwtco data (the help page's
# example): the five methods, a stratified subcohort, counting-process data
# with delayed entry, failure times without ties, and Breslow ties.
# cch runs coxph with Efron ties only. For Breslow ties the script runs the
# five fitters of cch with a coxph that has ties = "breslow", by changing the
# environment of each fitter, and checks that the same route reproduces cch
# for Efron ties to the last bit.
# For each fit: the coefficients, var (cch's `var`), phase2var, the model-based
# covariance before phase2var was added (cch stores the sum in naive.var), the
# iteration count of the fit that gives the coefficients, opt for the Borgan
# methods, and the condition number of the model-based information matrix on
# the scale of the columns, which bounds the error of every quantity computed
# from it.
# Numbers are written with 17 significant digits; the versions of R, survival
# and jsonlite are in the file.
# Run: Rscript casecohort.R > casecohort.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
data(nwtco)
set.seed(20261123)
full <- nwtco
full$sub <- as.numeric(full$in.subcohort)
keep <- full$rel == 1 | full$sub == 1
n_full <- nrow(full)

# Delayed entry for a third of the rows, failure times without ties by jitter,
# and a post-stratification into three groups of the whole cohort.
entry <- ifelse(runif(n_full) < 1 / 3, floor(runif(n_full) * 0.4 * full$edrel), 0)
entry <- pmin(entry, full$edrel - 1)
jit <- full$edrel + runif(n_full, 0, 0.9)
strat3 <- sample(1:3, n_full, replace = TRUE, prob = c(0.5, 0.3, 0.2))

datasets <- list(
  right = list(stop = full$edrel, start = rep(0, n_full), counting = FALSE, stratum = full$instit),
  entry = list(stop = full$edrel, start = entry, counting = TRUE, stratum = strat3),
  untied = list(stop = jit, start = rep(0, n_full), counting = FALSE, stratum = strat3)
)

xmat <- function(d) {
  X <- model.matrix(~ factor(stage) + factor(histol) + I(age / 12), data = d)[, -1, drop = FALSE]
  colnames(X) <- c("stage2", "stage3", "stage4", "histol2", "age")
  X
}
X_full <- xmat(full)

# The body of a fitter with `ties = <ties>` added to every call of coxph.
add_ties <- function(e, ties) {
  if (is.call(e)) {
    e <- as.call(lapply(as.list(e), add_ties, ties = ties))
    if (identical(e[[1]], as.name("coxph"))) e$ties <- ties
  }
  e
}

# One run of a fitter of cch: the preparation of cch() and the fitter, with a
# coxph that takes the ties rule.
run <- function(ds, method, ties, robust) {
  stratified <- method %in% c("I.Borgan", "II.Borgan")
  idx <- which(keep)
  texit <- ds$stop[idx]
  tenter <- ds$start[idx]
  subcoh <- full$sub[idx]
  cens <- full$rel[idx]
  cc <- cens + 1 - subcoh
  X <- X_full[idx, , drop = FALSE]
  id <- full$seqno[idx]
  dtime <- unique(texit[cens == 1])
  delta <- if (length(dtime) > 1) min(diff(sort(dtime))) / 2 else 1
  fitter <- getFromNamespace(method, "survival")
  body(fitter) <- add_ties(body(fitter), ties)
  if (stratified) {
    stratum <- factor(ds$stratum[idx])
    sizes <- table(factor(ds$stratum))
    out <- fitter(tenter = tenter, texit = texit, cc = cc, id = id, X = X,
                  stratum = as.numeric(stratum), stratum.sizes = sizes, delta)
    ntot <- NULL
  } else {
    ntot <- n_full
    out <- fitter(tenter = tenter, texit = texit, cc = cc, id = id, X = X, ntot = ntot,
                  robust = robust, delta)
  }
  # The fit that gives the coefficients: for Prentice the first one.
  iter <- out$iter
  if (method == "Prentice") {
    ent2 <- tenter
    ent2[cc == 2] <- texit[cc == 2] - delta
    iter <- survival::coxph(Surv(ent2, texit, as.numeric(cc > 0)) ~ X, timefix = FALSE, ties = ties)$iter
  }
  p2 <- as.matrix(out$phase2var)
  model_var <- as.matrix(out$naive.var) - p2
  sc <- colMeans(abs(sweep(X, 2, colMeans(X))))
  D <- diag(sc, length(sc))
  kappa_model <- kappa(solve(D %*% model_var %*% D), exact = TRUE)
  list(coefficients = unname(out$coefficients), var = unname(as.matrix(out$var)),
       phase2var = unname(p2), model_var = unname(model_var), iter = iter,
       opt = if (is.null(out$opt)) NULL else unname(out$opt), kappa = kappa_model,
       cch_ref = list(idx = idx, delta = delta, ntot = ntot))
}

cch_call <- function(ds, method, robust) {
  idx <- which(keep)
  d <- data.frame(stop = ds$stop[idx], start = ds$start[idx], rel = full$rel[idx],
                  sub = full$sub[idx], id = full$seqno[idx], stage = full$stage[idx],
                  histol = full$histol[idx], age = full$age[idx], stratum = ds$stratum[idx])
  sizes <- if (method %in% c("I.Borgan", "II.Borgan")) table(factor(ds$stratum)) else n_full
  f <- if (ds$counting) Surv(start, stop, rel) ~ factor(stage) + factor(histol) + I(age / 12)
       else Surv(stop, rel) ~ factor(stage) + factor(histol) + I(age / 12)
  suppressWarnings(cch(f, data = d, subcoh = ~sub, id = ~id, stratum = ~stratum,
                       cohort.size = sizes, method = method, robust = robust))
}

specs <- list()
add <- function(ds, name, method, ties = "efron", robust = FALSE) {
  specs[[length(specs) + 1]] <<- list(ds = ds, name = name, method = method, ties = ties, robust = robust)
}
for (m in c("Prentice", "SelfPrentice", "LinYing", "I.Borgan", "II.Borgan")) {
  add("right", paste0("right_", m), m)
  add("entry", paste0("entry_", m), m)
}
add("right", "right_LinYing_robust", "LinYing", robust = TRUE)
add("entry", "entry_LinYing_robust", "LinYing", robust = TRUE)
for (m in c("Prentice", "LinYing", "I.Borgan")) add("untied", paste0("untied_", m), m)
for (m in c("Prentice", "SelfPrentice", "LinYing", "I.Borgan", "II.Borgan")) {
  add("right", paste0("breslow_", m), m, ties = "breslow")
}
add("entry", "breslow_entry_Prentice", "Prentice", ties = "breslow")

fits <- lapply(specs, function(s) {
  ds <- datasets[[s$ds]]
  r <- run(ds, s$method, s$ties, s$robust)
  if (s$ties == "efron") {
    ref <- cch_call(ds, s$method, s$robust)
    # The route reproduces cch itself.
    stopifnot(isTRUE(all.equal(unname(ref$coefficients), r$coefficients, tolerance = 1e-12)))
    stopifnot(isTRUE(all.equal(unname(as.matrix(ref$var)), r$var, tolerance = 1e-12)))
    stopifnot(isTRUE(all.equal(unname(as.matrix(ref$phase2var)), r$phase2var, tolerance = 1e-12)))
  }
  c(list(name = s$name, dataset = s$ds, method = s$method, ties = s$ties, robust = s$robust),
    r[c("coefficients", "var", "phase2var", "model_var", "iter", "opt", "kappa")])
})

idx <- which(keep)
sizes_for <- function(name) as.numeric(table(factor(datasets[[name]]$stratum)))
out <- list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  n_cohort = n_full,
  x = unname(X_full[idx, ]),
  status = full$rel[idx],
  subcohort = full$sub[idx],
  datasets = lapply(names(datasets), function(nm) {
    ds <- datasets[[nm]]
    list(name = nm, stop = ds$stop[idx], start = ds$start[idx], counting = ds$counting,
         stratum = ds$stratum[idx], cohort_sizes = sizes_for(nm))
  }),
  fits = fits
)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null", null = "null"))
