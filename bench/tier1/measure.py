#!/usr/bin/env python3
"""Runs a command under a time limit and writes its wall time and peak memory.

Usage: measure.py TIMEFILE LIMIT_SECONDS COMMAND...

Writes TIMEFILE with the lines "Maximum resident set size (kbytes): N" and
"Elapsed (wall clock) time (s): S", where the peak covers the command and its
children (the memory of the process tree that was waited for). Exits with the
status of the command, or 124 when the limit was reached and the process group
was killed. It stands in for `/usr/bin/time -v` and `timeout` where those are
not installed.
"""
import os
import resource
import signal
import subprocess
import sys
import time

timefile, limit, command = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
start = time.monotonic()
proc = subprocess.Popen(command, start_new_session=True)
try:
    status = proc.wait(timeout=limit)
except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    status = 124
wall = time.monotonic() - start
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if sys.platform == "darwin":
    peak //= 1024  # macOS reports bytes, Linux kilobytes
with open(timefile, "w") as f:
    f.write(f"Maximum resident set size (kbytes): {peak}\n")
    f.write(f"Elapsed (wall clock) time (s): {wall:.3f}\n")
sys.exit(status if status >= 0 else 128 - status)
