# Reference values for epistat_stats::gamma::digamma and trigamma, as f64 bit
# patterns, from R's digamma() and trigamma().
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript digamma.R > digamma.rs
bits <- function(x) {
  vapply(x, function(v) {
    h <- toupper(paste(rev(as.character(writeBin(v, raw(), endian = "little"))), collapse = ""))
    paste0("0x", paste(substring(h, c(1, 5, 9, 13), c(4, 8, 12, 16)), collapse = "_"))
  }, "")
}
cat(sprintf("// Generated with %s: digamma(x) and trigamma(x), as f64 bit patterns.\n\n",
            R.version.string))
xs <- c(1e-8, 1e-3, 0.1, 0.5, 0.9, 1, 1.4616321449683623, 1.5, 2, 3.7, 5, 9.99, 10, 10.01,
        25, 100, 1e3, 1e5, 1e8, -0.3, -1.5, -2.7, -10.2)
cat("pub const DIGAMMA: &[(u64, u64, u64)] = &[\n")
for (x in xs) cat(sprintf("    (%s, %s, %s),\n", bits(x), bits(digamma(x)), bits(trigamma(x))))
cat("];\n")
