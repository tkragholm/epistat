# Reference results for epistat-register::intervals by logical day arrays in
# base R: intersection, difference, complement within a window and clipping of
# random half-open interval sets, on 1000 keys over days 0 to 59. Run from the
# crate directory:
#   Rscript tests/data/intervals_reference.R > tests/data/intervals_reference.json
suppressMessages(library(jsonlite))
set.seed(20261123)
days <- 60L
random_set <- function() {
  x <- rep(FALSE, days)
  pos <- sample(0:6, 1)
  while (pos < days) {
    len <- sample(1:12, 1)
    x[(pos + 1):min(pos + len, days)] <- TRUE
    pos <- pos + len + sample(1:8, 1)
  }
  x
}
# Intervals [start, end) of the runs of TRUE; day d is element d + 1.
runs <- function(x) {
  r <- rle(x)
  ends <- cumsum(r$lengths)
  starts <- ends - r$lengths
  keep <- r$values
  list(start = starts[keep], end = ends[keep])
}
n <- 1000
a <- b <- vector("list", n)
res <- list(inter = vector("list", n), minus = vector("list", n),
            compl = vector("list", n), clip = vector("list", n))
lo <- sample(0:30, n, replace = TRUE)
hi <- lo + sample(0:30, n, replace = TRUE)
for (k in seq_len(n)) {
  xa <- random_set(); xb <- random_set()
  a[[k]] <- runs(xa); b[[k]] <- runs(xb)
  res$inter[[k]] <- runs(xa & xb)
  res$minus[[k]] <- runs(xa & !xb)
  res$compl[[k]] <- runs(!xa)
  w <- rep(FALSE, days); if (hi[k] > lo[k]) w[(lo[k] + 1):hi[k]] <- TRUE
  res$clip[[k]] <- runs(xa & w)
}
flat <- function(lst) {
  list(start = unlist(lapply(lst, `[[`, "start")), end = unlist(lapply(lst, `[[`, "end")),
       n = sapply(lst, function(z) length(z$start)))
}
cat(toJSON(list(
  versions = list(R = as.character(getRversion())), days = days, lo = lo, hi = hi,
  a = flat(a), b = flat(b), intersect = flat(res$inter), subtract = flat(res$minus),
  complement = flat(res$compl), clip = flat(res$clip)
), digits = NA, auto_unbox = TRUE))
