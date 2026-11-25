#!/usr/bin/env bash
# Runs the Rust programs and the R scripts of the benchmarks of all tiers on
# the same simulated data, one process per method, size and implementation,
# and records wall time, peak memory and whether the process finished within
# the time limit. Each process runs under a memory cap.
#
#   SCRATCH=/some/dir/outside/the/repo bench/run.sh
#
# Settings, all optional:
#   SIZES     default "10000 100000 1000000 10000000"; a method runs up to its
#             own largest size (persons, for the g-formula; units, for the
#             synthdid methods, which are run with SIZES="100 1000 10000")
#   METHODS   default every method in the table below
#   TIERS     default "1 2 3"; the methods of those tiers
#   LIMIT     seconds allowed to one process; default the method's own limit
#             from the table
#   MEMORY    memory cap of one process, default 11G (systemd-run when it
#             works, ulimit otherwise; macOS enforces neither, the peak is
#             still recorded)
#   BIN_<bench>  path of an already built program, instead of cargo bench
#   CARGO     the cargo command, default cargo
#   IMPLS     "rust r" (default), or just one of them
#   FORCE_REPS  repetitions of the method inside one process; default 5 below
#             10^5 rows, 3 below 10^6 and 1 above
#   KEEP_DATA   keep the data files of a size after it has been run
#   OUT       results directory, default SCRATCH/out
#   AF_SRC    the directory AF/R of the unpacked tarball of the R package AF
#             0.1.5 (archived from CRAN), which the R side of af_cross,
#             af_case_control and af_cox sources; those R runs fail without it
#
# Results, in OUT:
#   runs.csv        impl,n,method,status,process_wall_s,peak_rss_kb
#   rust_times.csv  n,method,reps,median_s,min_s   (the method only)
#   r_times.csv     the same for R
#   rust_METHOD_N.csv, r_METHOD_N.csv  the numbers each side produced
#   compare_N.csv   what bench/tier1/compare.R prints for each size: the
#                   largest difference between the two sides' numbers. It is
#                   written before any time is read, and a name that R writes
#                   and Rust does not is reported on stderr.
#
# A method is not run at a size when it timed out, failed or was skipped at
# the previous size. Peak memory is the maximum resident set size of the
# process tree, from bench/tier1/measure.py, and includes reading the data
# file. The programs and the data format are described in bench/tier1/common.rs.
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SIZES="${SIZES:-10000 100000 1000000 10000000}"
TIERS="${TIERS:-1 2 3}"
OUT="${OUT:-$SCRATCH/out}"
MEMORY="${MEMORY:-11G}"
IMPLS="${IMPLS:-rust r}"
CARGO="${CARGO:-cargo}"
mkdir -p "$OUT"
[ -f "$OUT/runs.csv" ] || echo "impl,n,method,status,process_wall_s,peak_rss_kb" > "$OUT/runs.csv"

# method:crate:bench target:data kind:largest size:R script:limit (s)
# The data kind names the file the program writes (METHOD=gen_KIND) and both
# sides read; "none" when the program makes its data itself. The R script is
# relative to bench/, "-" where R has no counterpart (the Rust program then
# runs alone). The kind "pool" is written by the R script.
TABLE="
rcs:epistat-linalg:tier1_linalg:spl:10000000:tier1/bench.R:120
ns:epistat-linalg:tier1_linalg:spl:10000000:tier1/bench.R:120
negbin:epistat-glm:tier1_glm:reg:1000000:tier1/bench.R:120
risk_robust:epistat-glm:tier1_glm:reg:10000000:tier1/bench.R:120
risk_cluster:epistat-glm:tier1_glm:reg:10000000:tier1/bench.R:120
within:epistat-glm:tier1_glm:reg:10000000:tier1/bench.R:120
glm_logit_p5:epistat-glm:tier1_glm_scale:none:10000000:-:300
glm_logit_p20:epistat-glm:tier1_glm_scale:none:10000000:-:300
mh_or:epistat-stats:tier1_stats:mh:10000000:tier1/bench.R:120
mh_rr:epistat-stats:tier1_stats:mhr:10000000:tier1/bench.R:120
padj_holm:epistat-inference:tier1_inference:pv:10000000:tier1/bench.R:120
padj_hochberg:epistat-inference:tier1_inference:pv:10000000:tier1/bench.R:120
padj_bh:epistat-inference:tier1_inference:pv:10000000:tier1/bench.R:120
padj_by:epistat-inference:tier1_inference:pv:10000000:tier1/bench.R:120
padj_hommel:epistat-inference:tier1_inference:pv:100000:tier1/bench.R:120
pool:epistat-inference:tier1_inference:pool:50:tier1/bench.R:120
events:epistat-register:tier1_register:evt:10000000:tier1/bench.R:120
comorbidity:epistat-register:tier1_register:cmb:10000000:tier1/bench.R:120
exposure:epistat-register:tier1_register:exp:10000000:tier1/bench.R:120
pyears_rates:epistat-survival:tier1_survival:pyr:10000000:tier1/bench.R:120
sir_ratetable:epistat-survival:tier1_survival:pyr:10000000:tier1/bench.R:120
overlap:epistat-survival:tier1_survival:ps:10000000:tier1/bench.R:120
ipwtm:epistat-survival:tier1_survival:ppw:10000000:tier1/bench.R:120
cox_p5:epistat-survival:tier1_cox_scale:none:10000000:-:300
cox_p20:epistat-survival:tier1_cox_scale:none:10000000:-:300
cox_entry_p5:epistat-survival:tier1_cox_scale:none:10000000:-:300
balance_weights:epistat-matching:tier1_balance_scale:none:10000000:-:300
aipw:epistat-causal:tier1_causal:aipw:10000000:tier1/bench.R:120
att_gt:epistat-did:tier1_did:did:1000000:tier1/bench.R:120
synthdid_t20:epistat-did:tier1_synthdid:sdid20:10000:tier1/synthdid.R:600
synthdid_t100:epistat-did:tier1_synthdid:sdid100:10000:tier1/synthdid.R:600
sc_t100:epistat-did:tier1_synthdid:sdid100:10000:tier1/synthdid.R:600
synthdid_jk_t100:epistat-did:tier1_synthdid:sdid100:10000:tier1/synthdid.R:900
synthdid_placebo_t20:epistat-did:tier1_synthdid:sdid20:10000:tier1/synthdid.R:900
two_part:epistat-semicontinuous:tier1_semicontinuous:sc:10000000:tier1/bench.R:120
lexis:epistat-register:tier2_register:lx:10000000:tier2/bench.R:120
intervals:epistat-register:tier2_register:iv:10000000:tier2/bench.R:120
residence:epistat-register:tier2_register:rs:10000000:tier2/bench.R:120
pedigree:epistat-register:tier2_register:pd:10000000:tier2/bench.R:120
bias_exposure:epistat-stats:tier2_bias:none:10000000:tier2/bench.R:120
bias_outcome:epistat-stats:tier2_bias:none:10000000:tier2/bench.R:120
gee_exch:epistat-glm:tier2_glm:gpanel:10000000:tier2/glm_bench.R:120
gee_ar1:epistat-glm:tier2_glm:gpanel:10000000:tier2/glm_bench.R:120
gee_unstr:epistat-glm:tier2_glm:gpanel:10000000:-:120
ordinal:epistat-glm:tier2_glm:gord:10000000:tier2/glm_bench.R:120
ordinal_ppo:epistat-glm:tier2_glm:gord:10000000:tier2/glm_bench.R:120
brant:epistat-glm:tier2_glm:gord:10000000:tier2/glm_bench.R:120
aipw_km:epistat-survival:tier2_survival:surv:10000000:tier2/survival.R:120
aipw_cox:epistat-survival:tier2_survival:surv:10000000:tier2/survival.R:120
trial_msm:epistat-survival:tier2_survival:trial:1000000:tier2/survival.R:120
ccw:epistat-survival:tier2_survival:ccw:1000000:tier2/survival.R:120
trial_msm_rows:epistat-survival:tier2_survival:trial:10000000:-:1200
ccw_rows:epistat-survival:tier2_survival:ccw:10000000:-:1200
sccs:epistat-survival:tier2_designs:scc:3000000:tier2/designs.R:300
crossover:epistat-survival:tier2_designs:cxo:10000000:tier2/designs.R:300
af_cox:epistat-survival:tier2_designs:afc:1000000:tier2/designs.R:300
af_cross:epistat-glm:tier2_af:afd:10000000:tier2/af_bench.R:300
af_case_control:epistat-glm:tier2_af:afd:10000000:tier2/af_bench.R:300
ebal_att:epistat-causal:tier3_balance:bal:10000000:tier3/balance.R:120
ebal_ate_m2:epistat-causal:tier3_balance:bal:10000000:tier3/balance.R:120
cbps_exact:epistat-causal:tier3_balance:bal:1000000:tier3/balance.R:120
cbps_over:epistat-causal:tier3_balance:bal:100000:tier3/balance.R:120
cch_prentice:epistat-survival:tier3_designs:cc:1000000:tier3/designs.R:600
cch_selfprentice:epistat-survival:tier3_designs:cc:1000000:tier3/designs.R:600
cch_linying:epistat-survival:tier3_designs:cc:1000000:tier3/designs.R:600
cch_borgan1:epistat-survival:tier3_designs:cc:1000000:tier3/designs.R:600
cch_borgan2:epistat-survival:tier3_designs:cc:1000000:tier3/designs.R:600
ncc_probabilities:epistat-survival:tier3_designs:ncc:1000000:tier3/designs.R:600
ncc_fit:epistat-survival:tier3_designs:ncc:1000000:tier3/designs.R:600
frailty_fixed:epistat-survival:tier3_frailty:frailty:1000000:tier3/frailty.R:600
frailty_dense:epistat-survival:tier3_frailty:frailty:1000000:tier3/frailty.R:600
frailty_search:epistat-survival:tier3_frailty:frailty:1000000:tier3/frailty.R:600
frailty_profile:epistat-survival:tier3_frailty:frailty:1000000:tier3/frailty.R:600
frailty_entry:epistat-survival:tier3_frailty:frailty:1000000:tier3/frailty.R:600
flexpar_hazard:epistat-survival:tier3_survival:flex:1000000:tier3/survival.R:600
flexpar_odds:epistat-survival:tier3_survival:flex:1000000:tier3/survival.R:600
flexpar_probit:epistat-survival:tier3_survival:flex:1000000:tier3/survival.R:600
flexpar_entry:epistat-survival:tier3_survival:flex:1000000:tier3/survival.R:600
flexpar_tvc:epistat-survival:tier3_survival:flex:1000000:tier3/survival.R:600
net_pp:epistat-survival:tier3_relsurv:rel:1000000:tier3/relsurv.R:600
net_e2:epistat-survival:tier3_relsurv:rel:1000000:tier3/relsurv.R:600
net_pp_monthly:epistat-survival:tier3_relsurv:rel:10000000:tier3/relsurv.R:600
excess_poisson:epistat-survival:tier3_relsurv:rel:10000000:tier3/relsurv.R:600
excess_flexpar:epistat-survival:tier3_relsurv:rel:10000000:tier3/relsurv.R:600
tmle_ate_binary:epistat-causal:tier3_tmle:tml:10000000:tier3/tmle.R:600
tmle_ate_gaussian:epistat-causal:tier3_tmle:tml:10000000:tier3/tmle.R:600
tmle_all_binary:epistat-causal:tier3_tmle:tml:10000000:tier3/tmle.R:600
tmle_survival:epistat-causal:tier3_tmle:tml:10000000:-:600
gf_static:epistat-causal:tier3_gformula:gf:1000000:tier3/gformula.R:600
gf_dynamic:epistat-causal:tier3_gformula:gf:1000000:tier3/gformula.R:600
gf_boot:epistat-causal:tier3_gformula:gf:1000000:tier3/gformula.R:600
pmm:epistat-impute:tier3_impute:imp:1000000:tier3/impute.R:600
norm:epistat-impute:tier3_impute:imp:1000000:tier3/impute.R:600
logreg:epistat-impute:tier3_impute:imp:1000000:tier3/impute.R:600
polyreg:epistat-impute:tier3_impute:imp:1000000:tier3/impute.R:600
pmm_chains:epistat-impute:tier3_impute:imp:1000000:-:600
lmm_intercept:epistat-mixed:tier3_mixed:mix:1000000:tier3/mixed.R:600
lmm_slope:epistat-mixed:tier3_mixed:mix:1000000:tier3/mixed.R:600
glmm_laplace:epistat-mixed:tier3_mixed:mix:1000000:tier3/mixed.R:600
glmm_agq:epistat-mixed:tier3_mixed:mix:1000000:tier3/mixed.R:600
poisson_laplace:epistat-mixed:tier3_mixed:mix:1000000:tier3/mixed.R:600
pairs_agq:epistat-mixed:tier3_mixed:pair:1000000:tier3/mixed.R:600
"
TABLE="$(echo "$TABLE" | sed '/^$/d')"
ALL="$(echo "$TABLE" | cut -d: -f1 | tr '\n' ' ')"
if [ -z "${METHODS:-}" ]; then
  METHODS=""
  for t in $TIERS; do
    METHODS="$METHODS $(echo "$TABLE" | awk -F: -v t="tier$t" 'index($3, t "_") == 1 {print $1}' | tr '\n' ' ')"
  done
fi
for m in $METHODS; do
  case " $ALL " in *" $m "*) ;; *) echo "unknown method $m; known: $ALL" >&2; exit 2 ;; esac
done
field() { echo "$TABLE" | awk -F: -v m="$1" -v f="$2" '$1 == m {print $f}'; }

declare -A BIN
build_all() { # builds the programs of METHODS once, in one cargo run so features agree
  local crates="" benches="" m c b given pkgs="" flags="" missing=""
  for m in $METHODS; do
    b=$(field "$m" 3); given="BIN_$b"
    if [ -n "${!given:-}" ]; then BIN[$b]="${!given}"; continue; fi
    c=$(field "$m" 2)
    case " $crates " in *" $c "*) ;; *) crates="$crates $c" ;; esac
    case " $benches " in *" $b "*) ;; *) benches="$benches $b" ;; esac
    missing=1
  done
  [ -z "$missing" ] && return
  for c in $crates; do pkgs="$pkgs -p $c"; done
  for b in $benches; do flags="$flags --bench $b"; done
  local found
  found="$(cd "$ROOT" && $CARGO bench $pkgs $flags --no-run --message-format=json 2>"$OUT/build.log" |
    python3 -c "
import json, sys
for line in sys.stdin:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'].startswith('tier'):
        print(j['target']['name'], j['executable'])")"
  while read -r name path; do
    [ -n "${name:-}" ] && [ -z "${BIN[$name]:-}" ] && BIN[$name]=$path
  done <<< "$found"
  for b in $benches; do
    echo "built $b: ${BIN[$b]:-MISSING}" >&2
    [ -z "${BIN[$b]:-}" ] && { echo "build failed, see $OUT/build.log" >&2; exit 1; }
  done
}
bin_of() { echo "${BIN[$1]}"; }
build_all

reps_for() { # n
  if [ -n "${FORCE_REPS:-}" ]; then echo "$FORCE_REPS"
  elif [ "$1" -ge 1000000 ]; then echo 1
  elif [ "$1" -ge 100000 ]; then echo 3
  else echo 5; fi
}

# Runs a command under the memory cap and the time limit.
guarded() { # timefile limit command...
  local tf=$1 limit=$2; shift 2
  if systemd-run --user --scope -q -p MemoryMax="$MEMORY" -p MemorySwapMax=0 true 2>/dev/null; then
    systemd-run --user --scope -q -p MemoryMax="$MEMORY" -p MemorySwapMax=0 -- \
      python3 "$HERE/tier1/measure.py" "$tf" "$limit" "$@"
  else
    ( ulimit -v 12000000 2>/dev/null; python3 "$HERE/tier1/measure.py" "$tf" "$limit" "$@" )
  fi
}

record() { # impl n method status timefile
  local rss wall
  rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$5")
  wall=$(awk -F': ' '/Elapsed/ {print $2}' "$5")
  echo "$1,$2,$3,$4,${wall:-NA},${rss:-NA}" >> "$OUT/runs.csv"
}

run_one() { # impl n method
  local impl=$1 n=$2 m=$3 st status tf rs limit
  rs=$(field "$m" 6)
  [ "$impl" = r ] && [ "$rs" = "-" ] && return
  if [ -n "$prev" ] && grep -q "^$impl,$prev,$m,\(timeout\|skipped\|error[0-9]*\)" "$OUT/runs.csv"; then
    echo "$impl,$n,$m,skipped,NA,NA" >> "$OUT/runs.csv"
    return
  fi
  limit="${LIMIT:-$(field "$m" 7)}"
  tf="$OUT/time_${impl}_${m}_${n}.txt"
  if [ "$impl" = rust ]; then
    N=$n METHOD=$m REPS="$(reps_for "$n")" SCRATCH="$SCRATCH" OUT="$OUT" \
      guarded "$tf" "$limit" "$(bin_of "$(field "$m" 3)")" > "$OUT/log_rust_${m}_${n}.txt" 2>&1
  else
    REPS="$(reps_for "$n")" OUT="$OUT" \
      guarded "$tf" "$limit" Rscript "$HERE/$rs" "$n" "$m" "$SCRATCH" > "$OUT/log_r_${m}_${n}.txt" 2>&1
  fi
  st=$?
  status=ok; [ $st -eq 124 ] && status=timeout; [ $st -ne 0 ] && [ $st -ne 124 ] && status="error$st"
  record "$impl" "$n" "$m" "$status" "$tf"
}

generate() { # kind n
  local kind=$1 n=$2 m
  [ "$kind" = none ] && return
  [ -f "$SCRATCH/${kind}_$n.csv" ] && return
  m=""
  for cand in $METHODS; do
    [ "$(field "$cand" 4)" = "$kind" ] && { m=$cand; break; }
  done
  if [ "$kind" = pool ]; then
    OUT="$OUT" Rscript "$HERE/$(field "$m" 6)" "$n" gen_pool "$SCRATCH" > "$OUT/log_gen_pool.txt" 2>&1
    return
  fi
  N=$n METHOD="gen_$kind" SCRATCH="$SCRATCH" OUT="$OUT" \
    "$(bin_of "$(field "$m" 3)")" > "$OUT/log_gen_${kind}_$n.txt" 2>&1
}

size_of() { # method n: the size the method runs at, empty when it does not run
  local m=$1 n=$2 kind maxn
  maxn=$(field "$m" 5); kind=$(field "$m" 4)
  if [ "$kind" = pool ]; then
    [ "$n" = "${SIZES%% *}" ] || return
    n=50
  fi
  [ "$n" -gt "$maxn" ] && return
  echo "$n"
}

prev_size=""
for n in $SIZES; do
  for m in $METHODS; do
    size=$(size_of "$m" "$n"); [ -z "$size" ] && continue
    generate "$(field "$m" 4)" "$size"
  done
  for m in $METHODS; do
    size=$(size_of "$m" "$n"); [ -z "$size" ] && continue
    prev=""
    [ -n "$prev_size" ] && [ "$(field "$m" 4)" != pool ] && prev=$prev_size
    for impl in $IMPLS; do run_one "$impl" "$size" "$m"; done
  done
  for cn in $n 50; do
    [ "$cn" = 50 ] && [ "$n" != "${SIZES%% *}" ] && continue
    Rscript "$HERE/tier1/compare.R" "$cn" "$SCRATCH" > "$OUT/compare_$cn.csv" 2>&1
    awk -F, 'NR > 1 && $9 > 0 {print "compare: " $2 " at " $1 ": R writes " $9 " numbers that Rust does not"}' \
      "$OUT/compare_$cn.csv" >&2
  done
  # Data files of this size are not needed again.
  [ -n "${KEEP_DATA:-}" ] || find "$SCRATCH" -maxdepth 1 -name "*_$n.csv" -delete
  prev_size=$n
done
