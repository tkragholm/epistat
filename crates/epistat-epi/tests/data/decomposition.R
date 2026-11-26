# Reference values for epistat_stats::decomposition and
# epistat_stats::standardisation.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/decomposition.R
#
# The symmetric Kitagawa decomposition is Das Gupta's decomposition of a rate
# into an age structure and a rate factor, which DasGuptR::dgnpop gives with
# crossclassified = "size": the composition part is the difference of the
# two age_struct rates and the rate part that of the two rate rows. The
# decomposition with a reference and the variances of the parts are written
# out from their formulas. The standardisation weights of the comparators
# are those of MatchIt::matchit(method = "exact", estimand = "ATT") rescaled
# by MatchIt's own factor, and the weighted counts come from tapply.
suppressMessages({
  library(jsonlite)
  library(DasGuptR)
  library(MatchIt)
})
set.seed(20261126)

kitagawa_case <- function(name, d1, y1, d0, y0, scale = 1000) {
  a <- y1 / sum(y1); b <- y0 / sum(y0)
  r <- scale * d1 / y1; q <- scale * d0 / y0
  vr <- scale^2 * d1 / y1^2; vq <- scale^2 * d0 / y0^2
  k <- length(d1)
  df <- data.frame(pop = rep(c("A", "B"), each = k), age = rep(seq_len(k), 2),
                   size = c(y1, y0), rate = c(r, q))
  dg <- as.data.frame(dgnpop(df, pop = "pop", factors = "rate", id_vars = "age",
                             crossclassified = "size"))
  pick <- function(factor, p) dg$rate[dg$factor == factor & dg$pop == p]
  list(name = name, events1 = d1, time1 = y1, events0 = d0, time0 = y0, scale = scale,
       gap = sum(a * r) - sum(b * q),
       reference = c(sum((a - b) * q), sum(a * (r - q))),
       symmetric = c(sum((a - b) * (r + q) / 2), sum((r - q) * (a + b) / 2)),
       dasgupta = c(pick("age_struct", "A") - pick("age_struct", "B"),
                    pick("rate", "A") - pick("rate", "B")),
       var_gap = sum(a^2 * vr) + sum(b^2 * vq),
       var_reference = c(sum((a - b)^2 * vq), sum(a^2 * (vr + vq)),
                         -sum((a - b) * a * vq)),
       var_symmetric = c(sum(((a - b) / 2)^2 * (vr + vq)), sum(((a + b) / 2)^2 * (vr + vq)),
                         sum((a - b) / 2 * (a + b) / 2 * (vr - vq))))
}

kit <- list()
kit[[1]] <- kitagawa_case("the two regions of the rates guide",
                          c(3, 5, 6, 5), c(6000, 5000, 3000, 1500),
                          c(1, 2, 6, 11), c(2000, 3000, 4500, 5500))
k <- 15
y1 <- round(runif(k, 100, 20000), 1); y0 <- round(runif(k, 100, 20000), 1)
lam <- exp(seq(-8, -3, length.out = k))
kit[[2]] <- kitagawa_case("fifteen strata", rpois(k, 1.3 * y1 * lam), y1,
                          rpois(k, y0 * lam), y0)
d1 <- rpois(k, 0.2 * y1 * lam); d1[c(2, 9)] <- 0
kit[[3]] <- kitagawa_case("fifteen strata with empty counts, per person-year",
                          d1, y1, rpois(k, 0.2 * y0 * lam), y0, scale = 1)

# Standardisation to the exposed.
n <- 400
stratum <- sample(0:7, n, replace = TRUE)
exposed <- runif(n) < (0.15 + 0.08 * stratum)
m <- matchit(exposed ~ factor(stratum), data = data.frame(exposed, stratum),
             method = "exact", estimand = "ATT")
ne <- tapply(exposed, stratum, sum); nc <- tapply(!exposed, stratum, sum)
mine <- ifelse(exposed, 1, (ne / nc)[as.character(stratum)])
# MatchIt rescales the control weights so that they sum to the number of
# matched controls.
factor <- sum(m$weights[!exposed]) / sum(mine[!exposed])
stopifnot(max(abs(m$weights[!exposed] - factor * mine[!exposed])) < 1e-12)
g <- round(rgamma(n, 4, 4), 6)
gc <- tapply(g * !exposed, stratum, sum); ge <- tapply(g * exposed, stratum, sum)
std <- list(stratum = stratum, exposed = exposed, matchit = unname(m$weights),
            matchit_factor = factor, g = g, comparators = unname(gc),
            exposed_counts = unname(ge),
            weighted = unname(ifelse(exposed, g, g * (ge / gc)[as.character(stratum)])))

write(toJSON(list(versions = list(r = R.version.string,
                                  DasGuptR = as.character(packageVersion("DasGuptR")),
                                  MatchIt = as.character(packageVersion("MatchIt"))),
                  kitagawa = kit, standardisation = std),
             auto_unbox = TRUE, digits = 17, null = "null", na = "null"),
      "tests/data/decomposition.json")
