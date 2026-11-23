#!/usr/bin/env bash
# Times linear and generalised linear mixed models in Rust and in lme4 on the
# same simulated data, one process per method and size, with the memory cap
# and time limit of bench/tier2/run.sh.
#
#   SCRATCH=/some/dir/outside/the/repo bench/tier3/mixed_run.sh
#
# Settings, all optional: SIZES (default "10000 100000 1000000"), METHODS
# (default every method of crates/epistat-mixed/benches/tier3_mixed.rs),
# LIMIT (seconds per process, default 600), MEMORY (default 11G), REPS
# (default 1 at 10^6 rows, otherwise 3), OUT, BIN (a built tier3_mixed program
# instead of cargo bench). Not for the Mac above 10^5 rows.
#
# Results in OUT: runs.csv (impl,n,method,status,process_wall_s,peak_rss_kb),
# rust_times.csv and r_times.csv (n,method,reps,median_s,min_s) and the numbers
# each side wrote for the comparison (theta, beta and the log likelihood),
# compared by bench/tier1/compare.R.
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SIZES="${SIZES:-10000 100000 1000000}"
METHODS="${METHODS:-lmm_intercept lmm_slope glmm_laplace glmm_agq poisson_laplace pairs_agq}"
LIMIT="${LIMIT:-600}"
MEMORY="${MEMORY:-11G}"
OUT="${OUT:-$SCRATCH/out}"
mkdir -p "$OUT"
[ -f "$OUT/runs.csv" ] || echo "impl,n,method,status,process_wall_s,peak_rss_kb" > "$OUT/runs.csv"
BIN="${BIN:-$(cd "$ROOT" && cargo bench -p epistat-mixed --bench tier3_mixed --no-run --message-format=json 2>/dev/null |
  python3 -c "
import json, sys
for line in sys.stdin:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'] == 'tier3_mixed':
        print(j['executable'])")}"
guarded() { # timefile command...
  local tf=$1; shift
  if systemd-run --user --scope -q -p MemoryMax="$MEMORY" -p MemorySwapMax=0 true 2>/dev/null; then
    systemd-run --user --scope -q -p MemoryMax="$MEMORY" -p MemorySwapMax=0 -- \
      python3 "$HERE/../tier1/measure.py" "$tf" "$LIMIT" "$@"
  else
    ( ulimit -v 12000000 2>/dev/null; python3 "$HERE/../tier1/measure.py" "$tf" "$LIMIT" "$@" )
  fi
}
for n in $SIZES; do
  reps=3; [ "$n" -ge 1000000 ] && reps=1
  for kind in mix pair; do
    [ -f "$SCRATCH/${kind}_$n.csv" ] || N=$n METHOD=gen_$kind SCRATCH="$SCRATCH" OUT="$OUT" "$BIN"
  done
  for m in $METHODS; do
    for impl in rust r; do
      tf="$OUT/time_${impl}_${m}_${n}.txt"
      if [ "$impl" = rust ]; then
        N=$n METHOD=$m REPS=$reps SCRATCH="$SCRATCH" OUT="$OUT" \
          guarded "$tf" "$BIN" > "$OUT/log_rust_${m}_${n}.txt" 2>&1
      else
        REPS=$reps OUT="$OUT" \
          guarded "$tf" Rscript "$HERE/mixed.R" "$n" "$m" "$SCRATCH" > "$OUT/log_r_${m}_${n}.txt" 2>&1
      fi
      st=$?; status=ok; [ $st -eq 124 ] && status=timeout; [ $st -ne 0 ] && [ $st -ne 124 ] && status="error$st"
      rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$tf")
      wall=$(awk -F': ' '/Elapsed/ {print $2}' "$tf")
      echo "$impl,$n,$m,$status,${wall:-NA},${rss:-NA}" >> "$OUT/runs.csv"
    done
  done
  Rscript "$HERE/../tier1/compare.R" "$n" "$SCRATCH" > "$OUT/compare_$n.csv" 2>&1
  [ -n "${KEEP_DATA:-}" ] || rm -f "$SCRATCH/mix_$n.csv" "$SCRATCH/pair_$n.csv"
done
