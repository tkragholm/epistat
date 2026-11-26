# Reference results for the bias analysis of rates, from episensr 2.2.0 and a
# direct simulation.
#
# Deterministic: episensr::probsens_irr with constant sensitivity and
# specificity corrects the cases and person-time the same way in every draw;
# its first draw gives the corrected cells. Its corrected rate ratio is
# (A / (A + C)) / (B / (B + D)), which treats the person-time as non-cases;
# the script records it beside the rate ratio (A / C) / (B / D).
#
# Probabilistic: a direct simulation in R of the same correction with the
# rate ratio (A / C) / (B / D), run K times with different seeds, saving the
# median, 2.5th and 97.5th percentiles of each run, with systematic error and
# with the log rate ratio shifted by a standard normal draw times
# sqrt(1 / a + 1 / b).
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/bias_rates.R
suppressMessages({library(jsonlite); library(episensr); library(trapezoid)})
set.seed(20261126)
counts <- c(120, 300)
pt <- c(20000, 90000)
quiet <- function(expr) suppressMessages(suppressWarnings(expr))

det <- lapply(list(c(0.9, 0.97), c(0.8, 0.99), c(0.95, 0.995)), function(p) {
  x <- quiet(probsens_irr(matrix(c(counts, pt), 2, byrow = TRUE), reps = 3,
    seca = list("constant", p[1]), spca = list("constant", p[2])))
  d <- x$sim_df[1, ]
  list(se = p[1], sp = p[2], cells = as.numeric(d[c("A1", "B1", "C1", "D1")]),
       episensr_irr = d$corr_IRR,
       irr = (d$A1 / d$C1) / (d$B1 / d$D1))
})

K <- 10
reps <- 20000
summ <- function(x) c(median(x), quantile(x, .025, names = FALSE), quantile(x, .975, names = FALSE))
a <- counts[1]; b <- counts[2]; c <- pt[1]; d <- pt[2]
se_log <- sqrt(1 / a + 1 / b)
prob <- lapply(seq_len(K), function(k) {
  set.seed(2000 + k)
  se <- rtrapezoid(reps, 0.75, 0.8, 0.9, 0.95)
  sp <- runif(reps, 0.97, 1)
  A <- (a - (1 - sp) * (a + b)) / (se - (1 - sp))
  B <- (a + b) - A
  C <- (c - (1 - sp) * (c + d)) / (se - (1 - sp))
  D <- (c + d) - C
  ok <- A > 0 & B > 0 & C > 0 & D > 0
  irr <- ((A / C) / (B / D))[ok]
  total <- exp(log(irr) - rnorm(length(irr)) * se_log)
  list(syst = summ(irr), tot = summ(total), kept = sum(ok))
})

out <- list(
  versions = list(r = R.version.string,
    episensr = as.character(packageVersion("episensr")),
    trapezoid = as.character(packageVersion("trapezoid")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  counts = counts, pt = pt, deterministic = det, K = K, reps = reps,
  probabilistic = prob)
write_json(out, "tests/data/bias_rates.json", digits = I(17), auto_unbox = TRUE, na = "null")
