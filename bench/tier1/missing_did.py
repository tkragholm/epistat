# The Python side of the benchmark of epistat_did::att_missing_pre: the
# authors' replication code of Testa, Kennedy and Reimherr, `fit_missingdid`
# of `dgp.py` in github.com/testalorenzo/MissingDiD, on the data file the
# Rust program writes (`mdid`, see crates/epistat-did/benches/tier1_missing.rs).
# It writes the numbers the Rust side writes, as bench.R does, under the
# names r_METHOD_N.csv and r_times.csv that the driver reads.
#
# Written for epistat and licensed as epistat is (MIT OR Apache-2.0). The
# replication code is not part of epistat: it is loaded from MISSINGDID_SRC,
# a clone of that repository, and called unchanged.
#
# Usage: MISSINGDID_SRC=/path/to/MissingDiD python -I missing_did.py N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3). The Python environment needs numpy, pandas, statsmodels and tqdm.
import importlib.util
import io
import os
import statistics
import sys
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd

n, method, scratch = int(sys.argv[1]), sys.argv[2], sys.argv[3]
out = os.environ.get("OUT", os.path.join(scratch, "out"))
reps = int(os.environ.get("REPS", "3"))
os.makedirs(out, exist_ok=True)
src = os.environ["MISSINGDID_SRC"]
spec = importlib.util.spec_from_file_location("missingdid_dgp", os.path.join(src, "dgp.py"))
dgp = importlib.util.module_from_spec(spec)
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    spec.loader.exec_module(dgp)
warnings.filterwarnings("ignore")

ass, observed = {"missing_marx": ("2.3", "rx"), "missing_marxy1": ("2.4", "ry")}[method]
raw = pd.read_csv(os.path.join(scratch, f"mdid_{n}.csv"))
data = pd.DataFrame(
    {
        "z1": raw["z1"],
        "z2": raw["z2"],
        "z3": raw["z3"],
        "z4": raw["z4"],
        "y0": np.where(raw[observed] == 1, raw["y0"], np.nan),
        "y1": raw["y1"],
        "t": raw["t"].astype(int),
        "r": raw[observed].astype(int),
    }
)
del raw

times = []
for _ in range(reps):
    start = time.perf_counter()
    estimate, std = dgp.fit_missingdid(
        data,
        ass=ass,
        use_correct_model_mu=True,
        use_correct_model_gamma=True,
        use_correct_model_pi=True,
        use_correct_model_eta=True,
        augmentation=True,
    )
    times.append(time.perf_counter() - start)

prefix = "" if ass == "2.3" else "info_"
with open(os.path.join(out, f"r_{method}_{n}.csv"), "w") as f:
    f.write("name,value\n")
    f.write(f"{prefix}att,{float(estimate)!r}\n")
    f.write(f"{prefix}se,{float(std)!r}\n")
path = os.path.join(out, "r_times.csv")
fresh = not os.path.exists(path)
with open(path, "a") as f:
    if fresh:
        f.write("n,method,reps,median_s,min_s\n")
    f.write(f"{n},{method},{len(times)},{statistics.median(times)!r},{min(times)!r}\n")
