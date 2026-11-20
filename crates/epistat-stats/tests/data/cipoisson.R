# Reference values for epistat_stats::poisson: qgamma(p, shape) and
# survival::cipoisson, as f64 bit patterns. A limit that R gives as NaN is the
# bit pattern of NaN, which the Rust side reads as None.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript cipoisson.R > cipoisson.rs
suppressMessages(library(survival))
bits <- function(x) {
  vapply(x, function(v) {
    h <- toupper(paste(rev(as.character(writeBin(v, raw(), endian = "little"))), collapse = ""))
    paste0("0x", paste(substring(h, c(1, 5, 9, 13), c(4, 8, 12, 16)), collapse = "_"))
  }, "")
}
cat(sprintf("// Generated with %s and survival %s: qgamma(p, shape) and\n// cipoisson(k, time, p, method), as f64 bit patterns.\n\n",
            R.version.string, as.character(packageVersion("survival"))))

cat("pub const QGAMMA: &[(u64, u64, u64)] = &[\n")
for (shape in c(0.3, 0.5, 1, 2, 3, 5, 10, 25, 100, 1000, 1e5)) {
  for (p in c(1e-6, 0.025, 0.5, 0.975, 1 - 1e-6)) {
    cat(sprintf("    (%s, %s, %s),\n", bits(p), bits(shape), bits(qgamma(p, shape))))
  }
}
cat("];\n\n")

for (method in c("exact", "anscombe")) {
  cat(sprintf("pub const CIPOISSON_%s: &[(u64, u64, u64, u64, u64)] = &[\n", toupper(method)))
  for (k in c(0, 1, 2, 3, 5, 10, 50, 200, 1000, 2.5, 0.2)) {
    for (time in c(1, 3.7, 12345.6)) {
      for (level in c(0.9, 0.95, 0.99)) {
        ci <- suppressWarnings(cipoisson(k, time, level, method))
        cat(sprintf("    (%s, %s, %s, %s, %s),\n", bits(k), bits(time), bits(level),
                    bits(ci[["lower"]]), bits(ci[["upper"]])))
      }
    }
  }
  cat("];\n\n")
}
