# Reference values for epistat_did::bacon::decompose: bacondecomp 0.1.1
# (CRAN) on a simulated balanced panel with an always-treated group and on
# the package's castle data, without and with time-varying covariates, and
# the static TWFE coefficient of fixest 0.14.2 that the weights reproduce.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript generate_bacon_reference.R > bacon_ref.json
suppressMessages({
  library(bacondecomp)
  library(fixest)
  library(jsonlite)
})
stopifnot(packageVersion("bacondecomp") == "0.1.1", packageVersion("fixest") == "0.14.2")
set.seed(20261127)

simulated <- function() {
  G <- 60; T <- 10
  d <- expand.grid(time = 1:T, id = 0:(G - 1))
  g <- sample(c(1, 4, 6, 8, NA), G, TRUE, prob = c(0.1, 0.2, 0.2, 0.2, 0.3))
  d$g <- g[d$id + 1]
  d$treated <- as.integer(!is.na(d$g) & d$time >= d$g)
  u <- rnorm(G); tt <- rnorm(T)
  d$x1 <- rnorm(nrow(d)) + 0.5 * d$treated
  d$x2 <- 0.2 * d$time + rnorm(nrow(d))
  eff <- ifelse(d$treated == 1, 1 + 0.4 * (d$time - d$g) + 0.3 * (d$g == 4), 0)
  d$y <- u[d$id + 1] + tt[d$time] + 0.5 * d$x1 - 0.3 * d$x2 + eff + rnorm(nrow(d))
  d
}

castle_data <- function() {
  d <- bacondecomp::castle
  data.frame(id = as.integer(factor(d$state)) - 1, time = d$year, treated = d$post,
             y = d$l_homicide, x1 = d$l_pop, x2 = d$l_income)
}

two_by_twos <- function(b) {
  b <- b[order(b$treated, b$untreated), ]
  list(treated = b$treated, untreated = ifelse(b$untreated == 99999, NA, b$untreated),
       estimate = unname(b$estimate), weight = b$weight, type = b$type)
}

case <- function(name, d) {
  plain <- bacon(y ~ treated, data = d, id_var = "id", time_var = "time", quietly = TRUE)
  controlled <- bacon(y ~ treated + x1 + x2, data = d, id_var = "id", time_var = "time", quietly = TRUE)
  twfe <- coef(feols(y ~ treated | id + time, d))[["treated"]]
  twfe_x <- coef(feols(y ~ treated + x1 + x2 | id + time, d))[["treated"]]
  list(name = name, unit = d$id, time = d$time, treated = d$treated, outcome = d$y,
       x1 = d$x1, x2 = d$x2, plain = two_by_twos(plain), controlled = two_by_twos(controlled$two_by_twos),
       omega = controlled$Omega, beta_within = unname(controlled$beta_hat_w),
       twfe = twfe, twfe_controlled = twfe_x)
}

cases <- list(case("simulated", simulated()), case("castle", castle_data()))
cat(toJSON(list(r = R.version.string, bacondecomp = as.character(packageVersion("bacondecomp")),
                fixest = as.character(packageVersion("fixest")), cases = cases),
           digits = NA, auto_unbox = TRUE, null = "null", na = "null"))
