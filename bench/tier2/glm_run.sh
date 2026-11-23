#!/usr/bin/env bash
# Times the Rust programs and the R script of epistat-glm's tier 2 benchmarks
# (GEE and ordinal regression) on the same simulated data, one process per
# method and size. Meant for the Linux desktop: do not run above 10^5 rows on
# a laptop.
#
#   SCRATCH=/some/dir/outside/the/repo bench/tier2/glm_run.sh
#
# Settings, all optional: SIZES (default "10000 100000 1000000"), METHODS,
# IMPLS ("rust r"), REPS (default 3) and OUT (default SCRATCH/out). Results
# are those of bench/tier1/common.rs (rust_times.csv and r_times.csv, and the
# numbers compared by bench/tier1/compare.R).
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SIZES="${SIZES:-10000 100000 1000000}"
METHODS="${METHODS:-gee_exch gee_ar1 gee_unstr ordinal ordinal_ppo brant}"
IMPLS="${IMPLS:-rust r}"
export OUT="${OUT:-$SCRATCH/out}" REPS="${REPS:-3}" SCRATCH
mkdir -p "$OUT"
BIN="$(cd "$ROOT" && cargo bench -p epistat-glm --bench tier2_glm --no-run --message-format=json 2>/dev/null |
  python3 -c "
import json, sys
for line in sys.stdin:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'] == 'tier2_glm':
        print(j['executable'])" | tail -1)"
kind_of() { case "$1" in gee_*) echo gpanel ;; *) echo gord ;; esac; }
for n in $SIZES; do
  for kind in gpanel gord; do
    [ -f "$SCRATCH/${kind}_$n.csv" ] || N=$n METHOD="gen_$kind" "$BIN" > "$OUT/log_gen_${kind}_$n.txt" 2>&1
  done
  for m in $METHODS; do
    for impl in $IMPLS; do
      if [ "$impl" = rust ]; then
        N=$n METHOD=$m "$BIN" > "$OUT/log_rust_${m}_$n.txt" 2>&1
      else
        Rscript "$HERE/glm_bench.R" "$n" "$m" "$SCRATCH" > "$OUT/log_r_${m}_$n.txt" 2>&1
      fi
    done
  done
  Rscript "$ROOT/bench/tier1/compare.R" "$n" "$SCRATCH" > "$OUT/compare_$n.csv" 2>&1
  find "$SCRATCH" -maxdepth 1 -name "*_$n.csv" -delete
done
