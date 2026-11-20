# survival::pyears without a rate table: no dimension, a factor, one tcut,
# two tcut with a factor, delayed entry (start and stop), case weights, a
# scale, the order of dimensions, follow-up that starts below the first cut and
# ends beyond the last, cut points hit exactly by integer data, cut points hit
# up to rounding by decimal data (0.1 steps), and random data from a fixed seed.
# The random data are multiples of 1/4, which are exact in binary, so that two
# cuts met at the same instant are met exactly. With decimal data R sometimes
# sees them a rounding error apart and counts a piece of 1e-16 in a cell in
# between, which changes n; the case decimal_boundaries keeps decimals where
# that does not happen.
# Each dimension is recorded as kind "factor" (codes from 0, levels) or "time"
# (the value at time 0 and the cut points), in the order of the formula.
# For each case: the input, then pyears, n, event and offtable, with the arrays
# flattened in R's order (first dimension fastest).
# Seventeen digits are written so that every double reads back exactly.
# Run: Rscript pyears.R > pyears.json
suppressMessages({
  library(jsonlite)
  library(survival)
})

run <- function(name, d, dims, scale = 1, weights = NULL, counting = FALSE) {
  n <- nrow(d)
  terms <- character(0)
  spec <- list()
  for (k in seq_along(dims)) {
    dm <- dims[[k]]
    v <- paste0("v", k)
    if (dm$kind == "factor") {
      d[[v]] <- factor(dm$codes, levels = seq_len(dm$levels) - 1L)
      terms <- c(terms, v)
      spec[[k]] <- list(kind = "factor", codes = I(as.integer(dm$codes)), levels = dm$levels)
    } else {
      d[[v]] <- tcut(dm$origin, dm$cuts, labels = seq_len(length(dm$cuts) - 1L))
      terms <- c(terms, v)
      spec[[k]] <- list(kind = "time", origin = dm$origin, cuts = dm$cuts)
    }
  }
  lhs <- if (counting) "Surv(start, stop, status)" else "Surv(stop, status)"
  f <- as.formula(paste(lhs, "~", if (length(terms)) paste(terms, collapse = " + ") else "1"))
  d$w <- if (is.null(weights)) rep(1, n) else weights
  fit <- pyears(f, data = d, weights = w, scale = scale)
  list(
    name = name, scale = scale,
    start = if (counting) d$start else NULL,
    stop = d$stop, status = I(as.integer(d$status)),
    weight = if (is.null(weights)) NULL else weights,
    dims = spec,
    pyears = I(as.vector(fit$pyears)), n = I(as.vector(fit$n)),
    event = I(as.vector(fit$event)), offtable = fit$offtable
  )
}

cases <- list()

# integer data on integer cuts: every boundary is hit exactly
d <- data.frame(stop = c(5, 10, 3, 8, 12, 1, 7), status = c(1, 0, 1, 1, 0, 1, 0))
cases[[1]] <- run("integer_age", d, list(
  list(kind = "time", origin = c(0, 2, 5, 9, 1, 4, 3), cuts = c(0, 3, 6, 10, 20))))
cases[[2]] <- run("no_dimension", d, list())
cases[[3]] <- run("factor_only", d, list(
  list(kind = "factor", codes = c(0, 1, 2, 0, 1, 2, 2), levels = 4L)))

# decimal data: boundaries reached up to rounding
d <- data.frame(stop = c(0.2, 0.7, 0.1, 0.5, 0.3), status = c(1, 1, 0, 1, 1))
cases[[4]] <- run("decimal_boundaries", d, list(
  list(kind = "time", origin = c(0.1, 0.1, 0.2, 0.3, 0), cuts = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 1))))

# two time dimensions and a factor, off table at both ends
set.seed(2026)
n <- 400
d <- data.frame(stop = round(runif(n, 0.25, 9) * 4) / 4, status = rbinom(n, 1, 0.4))
age0 <- round(runif(n, 0, 70) * 4) / 4
yr0 <- round(runif(n, 1985, 2012) * 4) / 4
sex <- sample(0:1, n, replace = TRUE)
cases[[5]] <- run("two_time_and_factor", d, list(
  list(kind = "factor", codes = sex, levels = 2L),
  list(kind = "time", origin = age0, cuts = c(5, 20, 40, 60, 75)),
  list(kind = "time", origin = yr0, cuts = c(1990, 1995, 2000, 2005, 2010, 2015))))
cases[[6]] <- run("time_before_factor", d, list(
  list(kind = "time", origin = age0, cuts = c(0, 18, 45, 100)),
  list(kind = "factor", codes = sex, levels = 2L)))
cases[[7]] <- run("one_time", d, list(
  list(kind = "time", origin = age0, cuts = c(0, 10, 20, 30, 40, 50, 60, 70, 100))))

# delayed entry
st <- round(runif(n, 0, 3) * 4) / 4
d2 <- data.frame(start = st, stop = st + round(runif(n, 0.25, 6) * 4) / 4, status = rbinom(n, 1, 0.3))
cases[[8]] <- run("delayed_entry", d2, list(
  list(kind = "factor", codes = sex, levels = 2L),
  list(kind = "time", origin = age0, cuts = c(0, 10, 20, 30, 40, 50, 60, 70, 100)),
  list(kind = "time", origin = yr0, cuts = c(1980, 1990, 2000, 2010, 2020))),
  counting = TRUE)

# case weights
w <- round(runif(n, 0.2, 3) * 8) / 8
cases[[9]] <- run("weights", d2, list(
  list(kind = "time", origin = age0, cuts = c(0, 10, 20, 30, 40, 50, 60, 70, 100)),
  list(kind = "factor", codes = sex, levels = 2L)),
  weights = w, counting = TRUE)

# days and years
dd <- data.frame(stop = round(runif(n, 30, 3000)), status = rbinom(n, 1, 0.5))
a0 <- round(runif(n, 0, 60 * 365.25))
y0 <- round(runif(n, 10000, 15000))
cases[[10]] <- run("days_to_years", dd, list(
  list(kind = "time", origin = a0, cuts = 365.25 * c(0, 18, 40, 65, 100)),
  list(kind = "time", origin = y0, cuts = c(9000, 11000, 13000, 15000, 17000))),
  scale = 365.25)

# repeated and zero-width cuts
cases[[11]] <- run("repeated_cuts", d, list(
  list(kind = "time", origin = age0, cuts = c(0, 20, 20, 40, 40, 40, 100))))

cat(toJSON(list(
  r_version = R.version.string,
  survival_version = as.character(packageVersion("survival")),
  cases = cases
), digits = I(17), auto_unbox = TRUE, null = "null"))
