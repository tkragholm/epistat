# Reference values for epistat_stats::student where a shape of the incomplete
# beta function is 1000 or more: pt(z, nu) for nu from 2000 to 1e7, and
# pbeta(x, a, b) with one or both shapes large, as f64 bit patterns. For pbeta
# the smaller of x and y = 1 - x is a multiple of 2^-30, so both are exact.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript student_large.R > student_large.rs
bits <- function(x) {
  vapply(x, function(v) {
    h <- toupper(paste(rev(as.character(writeBin(v, raw(), endian = "little"))), collapse = ""))
    paste0("0x", paste(substring(h, c(1, 5, 9, 13), c(4, 8, 12, 16)), collapse = "_"))
  }, "")
}
dyadic <- function(v) round(v * 2^30) / 2^30
cat(sprintf("// Generated with %s: pt(z, nu) and pbeta(x, a, b) at large shapes, as f64 bit patterns.\n\n",
            R.version.string))
cat("pub const PT_LARGE: &[(u64, u64, u64, u64)] = &[\n")
for (nu in c(2000, 1e4, 1e5, 1e6, 1e7)) {
  for (z in c(-9, -4, -1.5, 0.3, 2, 5, 12)) {
    cat(sprintf("    (%s, %s, %s, %s),\n", bits(z), bits(nu),
                bits(pt(z, nu)), bits(pt(z, nu, lower.tail = FALSE))))
  }
}
cat("];\n\n")
# (x, a, b, I_x(a, b)): a large and b small with y near b / a, the mirror image,
# and both large with x near a / (a + b).
cat("pub const PBETA_LARGE: &[(u64, u64, u64, u64)] = &[\n")
for (a in c(1000, 5000, 3e4, 2e5)) {
  for (b in c(0.5, 1, 3, 20)) {
    for (m in c(0.1, 0.5, 1, 2, 6)) {
      y <- dyadic(m * b / a)
      x <- 1 - y
      cat(sprintf("    (%s, %s, %s, %s),\n", bits(x), bits(a), bits(b), bits(pbeta(x, a, b))))
      cat(sprintf("    (%s, %s, %s, %s),\n", bits(y), bits(b), bits(a), bits(pbeta(y, b, a))))
    }
  }
}
for (a in c(2000, 8000, 5e4)) {
  for (r in c(0.5, 1, 3)) {
    b <- r * a
    mu <- a / (a + b)
    sd <- sqrt(a * b / ((a + b)^2 * (a + b + 1)))
    for (k in c(-4, -1, 0.5, 2, 5)) {
      x <- dyadic(mu + k * sd)
      cat(sprintf("    (%s, %s, %s, %s),\n", bits(x), bits(a), bits(b), bits(pbeta(x, a, b))))
    }
  }
}
cat("];\n")
