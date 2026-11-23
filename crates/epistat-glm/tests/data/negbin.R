# Reference values for epistat_glm::negbin and the quasi families: MASS::glm.nb
# and glm(family = quasipoisson / quasibinomial).
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript negbin.R > negbin.json
suppressMessages({
  library(MASS)
  library(jsonlite)
})
set.seed(20261122)

sim <- function(n, theta, beta = c(0.4, 0.5, -0.6, 0.3, -0.2), offset = NULL,
                weights = NULL, zero_extra = 0) {
  x1 <- rbinom(n, 1, 0.4)
  x2 <- rnorm(n)
  f <- sample(0:2, n, replace = TRUE)
  x <- cbind(1, x1, x2, f == 1, f == 2) * 1
  eta <- drop(x %*% beta) + (if (is.null(offset)) 0 else offset)
  mu <- exp(eta)
  y <- if (is.infinite(theta)) rpois(n, mu) else rnbinom(n, mu = mu, size = theta)
  if (zero_extra > 0) y[runif(n) < zero_extra] <- 0
  list(x = x, y = y, offset = offset, weights = weights)
}

# One glm.nb run, returning what the Rust side compares.
run_nb <- function(sc, tight, link = "log") {
  ctrl <- if (tight) glm.control(epsilon = 1e-14, maxit = 200) else glm.control(trace = 1)
  steps <- 0
  warns <- character()
  fit <- tryCatch(
    withCallingHandlers(
      eval(bquote(glm.nb(sc$y ~ sc$x - 1, weights = sc$weights, offset = sc$offset,
                         control = ctrl, link = .(as.name(link))))),
      message = function(m) {
        if (grepl("^Theta\\(", conditionMessage(m))) steps <<- steps + 1
        invokeRestart("muffleMessage")
      },
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) e)
  if (inherits(fit, "error")) {
    return(list(tight = tight, error = conditionMessage(fit), warnings = warns))
  }
  s <- summary(fit)
  se <- rep(NA_real_, length(coef(fit)))
  se[!is.na(coef(fit))] <- s$coefficients[, 2]
  list(tight = tight, coef = unname(coef(fit)), se = se, theta = fit$theta,
       se_theta = fit$SE.theta, twologlik = fit$twologlik, deviance = fit$deviance,
       df_residual = fit$df.residual, alternations = if (tight) NA else steps,
       iter = fit$iter, warnings = warns)
}

scenarios <- list()
add <- function(name, sc, link = "log") {
  scenarios[[length(scenarios) + 1]] <<- list(
    name = name, link = link, x = unname(sc$x), y = sc$y,
    weights = sc$weights, offset = sc$offset,
    runs = list(run_nb(sc, FALSE, link), run_nb(sc, TRUE, link)))
}

# quine, the classic formula
x <- unname(model.matrix(Days ~ Sex / (Age + Eth * Lrn), quine))
add("quine", list(x = x, y = quine$Days))

n <- 400
for (theta in c(0.5, 2, 20)) add(sprintf("theta_%g", theta), sim(n, theta))
py <- exp(rnorm(n, 0, 0.4))
add("offset", sim(n, 2, offset = log(py)))
add("weights", sim(n, 2, weights = sample(1:4, n, replace = TRUE)))
s <- sim(n, 1.5); s$x <- cbind(s$x, 0)
add("aliased", s)
add("zero_inflated", sim(n, 3, zero_extra = 0.35))
add("poisson_limit", sim(n, Inf))
# Quasi families
quasi <- list()
for (nm in c("quasipoisson", "quasibinomial")) {
  sc <- sim(n, 2, weights = NULL)
  y <- if (nm == "quasipoisson") sc$y else as.numeric(sc$y > 1)
  wts <- sample(1:3, n, replace = TRUE)
  fit <- glm(y ~ sc$x - 1, family = get(nm)(), weights = wts)
  sm <- summary(fit)
  quasi[[length(quasi) + 1]] <- list(
    family = nm, x = unname(sc$x), y = y, weights = wts,
    coef = unname(coef(fit)), se = unname(sm$coefficients[, 2]),
    dispersion = sm$dispersion, df_residual = fit$df.residual,
    deviance = fit$deviance, iter = fit$iter)
}

cat(toJSON(list(r = R.version.string, mass = as.character(packageVersion("MASS")),
                scenarios = scenarios, quasi = quasi),
           digits = NA, auto_unbox = TRUE, null = "null", na = "null"))
