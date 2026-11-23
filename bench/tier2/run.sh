#!/usr/bin/env bash
# Runs the Rust programs of the tier 2 benchmarks, one process per method and
# size, and records wall time, peak memory and whether the process finished
# within the time limit. The data are generated inside each process from a
# fixed seed, so there is no R side; the programs and the files they write are
# those of bench/tier1 (common.rs, measure.py).
#
#   SCRATCH=/some/dir/outside/the/repo bench/tier2/run.sh
#
# Settings, all optional:
#   SIZES     default "10000 100000 1000000 10000000"; a method runs up to its
#             own largest size
#   METHODS   default every method in the table below
#   LIMIT     seconds allowed to one process, default 300
#   MEMORY    memory cap of one process, default 11G (systemd-run when it
#             works, ulimit otherwise)
#   OUT       results directory, default SCRATCH/out
#
# Results, in OUT: runs.csv (impl,n,method,status,process_wall_s,peak_rss_kb),
# rust_times.csv (n,method,reps,median_s,min_s) and rust_METHOD_N.csv with the
# numbers each run produced.
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TIER1="$ROOT/bench/tier1"
SIZES="${SIZES:-10000 100000 1000000 10000000}"
OUT="${OUT:-$SCRATCH/out}"
LIMIT="${LIMIT:-300}"
MEMORY="${MEMORY:-11G}"
mkdir -p "$OUT"
[ -f "$OUT/runs.csv" ] || echo "impl,n,method,status,process_wall_s,peak_rss_kb" > "$OUT/runs.csv"

# method:crate:bench:largest size
TABLE="
sccs:epistat-survival:tier2_survival:3000000
crossover:epistat-survival:tier2_survival:10000000
af_cox:epistat-survival:tier2_survival:1000000
af_cross:epistat-glm:tier2_glm:10000000
af_case_control:epistat-glm:tier2_glm:10000000
"
METHODS="${METHODS:-$(echo "$TABLE" | cut -d: -f1 | tr '\n' ' ')}"
field() { echo "$TABLE" | awk -F: -v m="$1" -v f="$2" '$1 == m {print $f}'; }

declare -A BIN
build_all() {
  local found
  found="$(cd "$ROOT" && cargo bench -p epistat-survival -p epistat-glm --benches --no-run --message-format=json 2>/dev/null |
    python3 -c "
import json, sys
for line in sys.stdin:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'].startswith('tier2_'):
        print(j['target']['name'], j['executable'])")"
  while read -r name path; do
    [ -n "${name:-}" ] && BIN[$name]=$path
  done <<< "$found"
}
build_all

reps_for() { if [ "$1" -ge 1000000 ]; then echo 1; elif [ "$1" -ge 100000 ]; then echo 3; else echo 5; fi; }

guarded() { # timefile command...
  local tf=$1; shift
  if systemd-run --user --scope -q -p MemoryMax="$MEMORY" -p MemorySwapMax=0 true 2>/dev/null; then
    systemd-run --user --scope -q -p MemoryMax="$MEMORY" -p MemorySwapMax=0 -- \
      python3 "$TIER1/measure.py" "$tf" "$LIMIT" "$@"
  else
    ( ulimit -v 12000000 2>/dev/null; python3 "$TIER1/measure.py" "$tf" "$LIMIT" "$@" )
  fi
}

prev=""
for n in $SIZES; do
  for m in $METHODS; do
    [ "$n" -gt "$(field "$m" 4)" ] && continue
    if [ -n "$prev" ] && grep -q "^rust,$prev,$m,\(timeout\|skipped\|error[0-9]*\)" "$OUT/runs.csv"; then
      echo "rust,$n,$m,skipped,NA,NA" >> "$OUT/runs.csv"; continue
    fi
    tf="$OUT/time_rust_${m}_${n}.txt"
    N=$n METHOD=$m REPS="$(reps_for "$n")" SCRATCH="$SCRATCH" OUT="$OUT" \
      guarded "$tf" "${BIN[$(field "$m" 3)]}" > "$OUT/log_rust_${m}_${n}.txt" 2>&1
    st=$?
    status=ok; [ $st -eq 124 ] && status=timeout; [ $st -ne 0 ] && [ $st -ne 124 ] && status="error$st"
    rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$tf")
    wall=$(awk -F': ' '/Elapsed/ {print $2}' "$tf")
    echo "rust,$n,$m,$status,${wall:-NA},${rss:-NA}" >> "$OUT/runs.csv"
  done
  prev=$n
done
