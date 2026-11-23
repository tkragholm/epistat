# Reference results for epistat-register::pedigree.
#
# 1. kinship: 2 * kinship2::kinship on a simulated pedigree of 5 generations
#    with a small mating pool per generation (so full and half siblings,
#    cousins and inbred persons occur), for every pair of persons.
# 2. pair kinds, family clusters, multiple births and birth order by direct
#    loops in base R on a pedigree of 200 persons with unknown and external
#    parents.
#
# Run from the crate directory:
#   Rscript tests/data/pedigree_reference.R > tests/data/pedigree_reference.json
suppressMessages({library(jsonlite); library(kinship2)})
set.seed(20261123)

# Persons in generations. Each generation has a pool of mothers and a pool of
# fathers drawn from the previous generation, and couples drawn from the pools.
simulate <- function(n_gen, per_gen, pool, p_missing, p_external) {
  id <- integer(0); sex <- character(0); mom <- integer(0); dad <- integer(0)
  gen <- integer(0); next_id <- 1000L; next_ext <- 900000L
  add <- function(sx, m, d, g) {
    id <<- c(id, next_id); sex <<- c(sex, sx); mom <<- c(mom, m); dad <<- c(dad, d)
    gen <<- c(gen, g); next_id <<- next_id + 1L
  }
  for (k in seq_len(per_gen)) add(sample(c("f", "m"), 1), NA, NA, 0L)
  for (g in seq_len(n_gen)) {
    prev <- which(gen == g - 1L)
    fem <- prev[sex[prev] == "f"]; mal <- prev[sex[prev] == "m"]
    fem <- fem[seq_len(min(pool, length(fem)))]; mal <- mal[seq_len(min(pool, length(mal)))]
    if (length(fem) == 0 || length(mal) == 0) stop("empty pool")
    couples <- cbind(sample(fem, 6, replace = TRUE), sample(mal, 6, replace = TRUE))
    for (k in seq_len(per_gen)) {
      cp <- couples[sample(nrow(couples), 1), ]
      m <- id[cp[1]]; d <- id[cp[2]]
      if (runif(1) < p_missing) m <- NA
      if (runif(1) < p_missing) d <- NA
      if (!is.na(m) && runif(1) < p_external) { m <- next_ext; next_ext <- next_ext + 1L }
      if (!is.na(d) && runif(1) < p_external) { d <- next_ext; next_ext <- next_ext + 1L }
      add(sample(c("f", "m"), 1), m, d, g)
    }
  }
  data.frame(id = id, sex = sex, mom = mom, dad = dad, gen = gen)
}

# --- 1. kinship2, complete pedigree: founders have both parents unknown ---
p1 <- simulate(5, 14, 4, 0, 0)
ped <- with(p1, pedigree(id = id, dadid = ifelse(is.na(dad), 0, dad),
                         momid = ifelse(is.na(mom), 0, mom),
                         sex = ifelse(sex == "m", 1, 2), missid = 0))
K <- kinship(ped)
ids <- rownames(K)
ord <- sample(nrow(p1))  # rows in a shuffled order
p1 <- p1[ord, ]
pos <- match(as.character(p1$id), ids)
R2 <- 2 * K[pos, pos]
# Founders: the generation 0 rows, to drop in the test of external parents.
founder <- p1$gen == 0
inbred <- diag(R2) > 1 + 1e-12

# --- 2. pair kinds by loops ---
p2 <- simulate(5, 40, 5, 0.12, 0.15)
p2 <- p2[sample(nrow(p2)), ]
n <- nrow(p2)
mom <- p2$mom; dad <- p2$dad; pid <- p2$id
same <- function(x, y) !is.na(x) && !is.na(y) && x == y
sib_kind <- function(i, j) {
  sm <- same(mom[i], mom[j]); sf <- same(dad[i], dad[j])
  if (sm && sf) return("Full")
  if (sm) return(if (!is.na(dad[i]) && !is.na(dad[j])) "MaternalHalf" else "MotherOnly")
  if (sf) return(if (!is.na(mom[i]) && !is.na(mom[j])) "PaternalHalf" else "FatherOnly")
  NA_character_
}
parents_of <- function(x) {  # ids of the parents of an id; none for an unknown id
  r <- match(x, pid)
  if (is.na(r)) return(integer(0))
  c(mom[r], dad[r])[!is.na(c(mom[r], dad[r]))]
}
anc_at <- function(r, k) {  # distinct ancestor ids exactly k generations up
  cur <- pid[r]
  for (s in seq_len(k)) cur <- unique(unlist(lapply(cur, parents_of)))
  if (is.null(cur)) integer(0) else cur
}
sib <- list(a = integer(0), b = integer(0), kind = character(0))
for (i in 1:(n - 1)) for (j in (i + 1):n) {
  k <- sib_kind(i, j)
  if (!is.na(k)) { sib$a <- c(sib$a, i - 1); sib$b <- c(sib$b, j - 1); sib$kind <- c(sib$kind, k) }
}
anc <- lapply(1:4, function(k) lapply(seq_len(n), function(r) anc_at(r, k)))
cousins <- function(d) {
  out <- list(a = integer(0), b = integer(0), kind = character(0))
  for (i in 1:(n - 1)) for (j in (i + 1):n) {
    sh <- length(intersect(anc[[d + 1]][[i]], anc[[d + 1]][[j]]))
    if (sh == 0) next
    closer <- any(sapply(seq_len(d), function(k) length(intersect(anc[[k]][[i]], anc[[k]][[j]])) > 0))
    if (closer) next
    out$a <- c(out$a, i - 1); out$b <- c(out$b, j - 1)
    out$kind <- c(out$kind, if (sh >= 2) "Full" else "Half")
  }
  out
}
cous1 <- cousins(1); cous2 <- cousins(2)

# Family clusters: label propagation over shared mother or father, numbered
# by first row.
lab <- seq_len(n)
repeat {
  old <- lab
  for (col in list(mom, dad)) {
    for (v in unique(col[!is.na(col)])) {
      rows <- which(col == v); lab[rows] <- min(lab[rows])
    }
  }
  if (all(old == lab)) break
}
clusters <- match(lab, unique(lab)) - 1

# Multiple births and birth order, by mother and a random birth day.
birth <- 10000 + sample(0:60, n, replace = TRUE) * 100
bmom <- mom
bmom[sample(n, 8)] <- NA
bday <- birth; bday[sample(n, 5)] <- NA
grp <- rep(NA_real_, n); size <- rep(NA_real_, n); order_ <- rep(NA_real_, n)
ok <- !is.na(bmom) & !is.na(bday)
key <- paste(bmom, bday)
first_of <- sapply(seq_len(n), function(r) if (ok[r]) min(which(ok & key == key[r])) else NA)
grp[ok] <- match(first_of[ok], sort(unique(first_of[ok]))) - 1
size[ok] <- sapply(which(ok), function(r) sum(ok & key == key[r]))
for (r in which(ok)) order_[r] <- length(unique(bday[ok & bmom == bmom[r] & bday < bday[r]]))

cat(toJSON(list(
  versions = list(R = as.character(getRversion()), kinship2 = as.character(packageVersion("kinship2"))),
  kinship = list(person = p1$id, mother = p1$mom, father = p1$dad, founder = founder,
                 relatedness = R2, inbred_count = sum(inbred)),
  kinds = list(person = pid, mother = mom, father = dad, birth_mother = bmom, birth_day = bday,
               siblings = sib, cousins1 = cous1, cousins2 = cous2, clusters = clusters,
               group = grp, size = size, order = order_)
), digits = NA, auto_unbox = TRUE, na = "null"))
