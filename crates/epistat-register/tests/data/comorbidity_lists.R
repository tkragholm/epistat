# Writes src/comorbidity/lists.rs from the maps and weights that the R package
# comorbidity holds: the Quan et al. (2005) ICD-10 coding algorithms for the
# Charlson and Elixhauser conditions, the Charlson (1987) and Quan et al.
# (2011) weights and the van Walraven et al. (2009) weights.
# Run from the crate directory: Rscript tests/data/comorbidity_lists.R
suppressMessages(library(comorbidity))
maps <- comorbidity:::.maps
weights <- comorbidity:::.weights
labels_of <- function(map) {
  x <- data.frame(id = 1L)
  for (n in names(maps[[map]])) x[[n]] <- 0L
  l <- attr(comorbidity:::.labelled(x, map), "variable.labels")
  l[-1]
}
q <- function(s) paste0('"', gsub('"', '\\\\"', s), '"')
out <- c(
  "//! The code lists and weights of the comorbidity indices, written by",
  "//! `tests/data/comorbidity_lists.R` from the maps of the R package",
  sprintf("//! comorbidity %s, %s. Formatted by rustfmt. Do not edit by hand.",
          as.character(packageVersion("comorbidity")), R.version.string),
  "",
  "/// A condition as the package holds it: key, label and ICD-10 prefixes.",
  "pub(super) struct Spec {",
  "    pub key: &'static str,",
  "    pub label: &'static str,",
  "    pub prefixes: &'static [&'static str],",
  "}",
  ""
)
emit_map <- function(const, map) {
  m <- maps[[map]]
  lab <- labels_of(map)
  out <<- c(out, sprintf("pub(super) static %s: [Spec; %d] = [", const, length(m)))
  for (i in seq_along(m)) {
    out <<- c(out, "    Spec {",
              sprintf("        key: %s,", q(names(m)[i])),
              sprintf("        label: %s,", q(lab[i])),
              "        prefixes: &[")
    p <- m[[i]]
    chunks <- split(p, ceiling(seq_along(p) / 8))
    for (ch in chunks) out <<- c(out, paste0("            ", paste0(q(ch), ",", collapse = " ")))
    out <<- c(out, "        ],", "    },")
  }
  out <<- c(out, "];", "")
}
emit_w <- function(const, w) {
  stopifnot(all(w == round(w)))
  out <<- c(out, sprintf("pub(super) const %s: [f64; %d] = [", const, length(w)),
            paste0("    ", paste0(sprintf("%d.0", as.integer(w)), ",", collapse = " ")), "];", "")
}
emit_map("CHARLSON_QUAN_ICD10", "charlson_icd10_quan")
emit_map("ELIXHAUSER_QUAN_ICD10", "elixhauser_icd10_quan")
emit_w("CHARLSON_1987", weights$charlson_icd10_quan$charlson)
emit_w("CHARLSON_QUAN_2011", weights$charlson_icd10_quan$quan)
emit_w("ELIXHAUSER_VAN_WALRAVEN_2009", weights$elixhauser_icd10_quan$vw)
writeLines(out, "src/comorbidity/lists.rs")
