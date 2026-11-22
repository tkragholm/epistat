# Reference values for epistat_inference::multiplicity from stats::p.adjust.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/padjust.R
#
# Five cases: 50 uniform p-values, a vector with many ties and exact 0 and 1,
# the same 50 with more comparisons than p-values, a single p-value, and 3000
# p-values from a mixture of small and uniform values. Every method of
# p.adjust is run on each. The values are written with 17 significant digits.
suppressMessages(library(jsonlite))
set.seed(20261122)
methods <- c("bonferroni", "holm", "hochberg", "hommel", "BH", "BY")
cases <- list()
add <- function(name, p, n = length(p)) {
  adj <- lapply(methods, function(m) I(p.adjust(p, m, n)))
  names(adj) <- methods
  cases[[length(cases) + 1]] <<- list(name = name, p = I(p), n = n, adjusted = adj)
}
u <- runif(50)
add("fifty uniform", u)
add("ties, zero and one", c(0, 0.01, 0.01, 0.01, 0.2, 0.2, 0.5, 0.5, 0.99, 1, 1, 0.03))
add("more comparisons than p-values", u[1:20], 35)
add("two comparisons", c(0.03, 0.04))
add("one p-value", 0.3)
big <- c(runif(300, 0, 1e-4), runif(2700))
add("3000 mixture", big)
write(toJSON(list(versions = list(r = R.version.string), cases = cases),
             auto_unbox = TRUE, digits = 17, null = "null"), "tests/data/padjust.json")
