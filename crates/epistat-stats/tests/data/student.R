# Reference values for epistat_stats::student: pt(z, nu), qt(p, nu) and
# pbeta(x, a, b), as f64 bit patterns.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript student.R > student.rs
bits <- function(x) {
  vapply(x, function(v) {
    h <- toupper(paste(rev(as.character(writeBin(v, raw(), endian = "little"))), collapse = ""))
    paste0("0x", paste(substring(h, c(1, 5, 9, 13), c(4, 8, 12, 16)), collapse = "_"))
  }, "")
}
cat(sprintf("// Generated with %s: pt(z, nu), qt(p, nu) and pbeta(x, a, b), as f64 bit patterns.\n\n",
            R.version.string))
nus <- c(0.5, 1, 2, 3.5, 5, 10, 30, 100, 1000)
cat("pub const PT: &[(u64, u64, u64, u64)] = &[\n")
for (nu in nus) {
  for (z in c(-40, -8, -3, -1, -0.25, 0, 0.1, 0.9, 2.5, 6, 30)) {
    cat(sprintf("    (%s, %s, %s, %s),\n", bits(z), bits(nu),
                bits(pt(z, nu)), bits(pt(z, nu, lower.tail = FALSE))))
  }
}
cat("];\n\n")
cat("pub const QT: &[(u64, u64, u64)] = &[\n")
for (nu in nus) {
  for (p in c(1e-10, 1e-4, 0.025, 0.2, 0.5, 0.8, 0.975, 1 - 1e-4)) {
    cat(sprintf("    (%s, %s, %s),\n", bits(p), bits(nu), bits(qt(p, nu))))
  }
}
cat("];\n\n")
cat("pub const PBETA: &[(u64, u64, u64, u64)] = &[\n")
for (a in c(0.25, 0.5, 1, 2.5, 10, 60)) {
  for (b in c(0.5, 1, 3, 20)) {
    for (x in c(1e-6, 0.01, 0.2, 0.5, 0.8, 0.999)) {
      cat(sprintf("    (%s, %s, %s, %s),\n", bits(x), bits(a), bits(b), bits(pbeta(x, a, b))))
    }
  }
}
cat("];\n")
