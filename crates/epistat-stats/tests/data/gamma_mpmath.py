# Reference values of the regularised incomplete gamma functions P(a, x) and
# Q(a, x) = 1 - P(a, x) at 50 significant digits or more, for
# epistat_stats::dist::gamma::{cdf, sf}.
#
# The shapes run from the smallest subnormal to 1e4 with x = a lambda for
# lambda from 1e-3 to 10, plus fixed x from 1e-8 to 300, and from 1e5 to 1e7
# near the centre (lambda from 0.99 to 1.01). The shape and x are doubles and
# the values are taken at those doubles exactly.
#
# Both tails are mpmath's gammainc(a, 0, x, regularized=True) and
# gammainc(a, x, inf, regularized=True) at 60 digits. Below the diagonal
# (x < a) P is also the positive series DLMF 8.11.4,
#     P(a, x) = x^a e^-x / Gamma(a + 1) * sum_{n>=0} x^n / ((a + 1) ... (a + n)),
# summed at 60 digits, and the two must agree to 1e-50 relative, which the
# script checks. Where mpmath's hypergeometric summation gives up (shapes
# from 1e6 near the centre), both tails come from the series at a precision
# that leaves 60 digits for the smaller one after Q = 1 - P.
#
# The P(a, a) of a = 1e15 and 1e300 is not on this grid: its reference is
# the expansion DLMF 8.12.15 with 8.12.17 in tests/gamma_mpmath.rs.
#
# Cases whose smaller tail is below 1e-300 are left out.
#
# Two further lists need no incomplete gamma function. "closed" holds
# Q(n + 1, x) = e^-x e_n(x), with the truncated exponential series e_n(x)
# summed in exact rationals (DLMF 8.4.10), and
# Q(n + 1/2, x) = erfc(sqrt x) + e^-x / sqrt(pi) sum_{k=1}^{n} x^(k - 1/2) / (1/2)_k
# (DLMF 8.4.14), for dyadic x, with P = 1 - Q at 360 digits. "recurrence" holds random (a, x) in (0, 1]^2
# and (0, 500]^2, from a fixed seed, with
# D(a, x) = x^a e^-x / Gamma(a + 1), the step of P(a + 1, x) = P(a, x) - D(a, x)
# (DLMF 8.8.5).
#
# Run: python3 gamma_mpmath.py > gamma_mpmath.json   (mpmath from PyPI)
import json
import random
import sys
from fractions import Fraction

import mpmath as mp

DIGITS = 60

SHAPES = [5e-324, 1e-300, 1e-3, 0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.5, 5.0,
          10.0, 12.0, 20.0, 50.0, 100.0, 1000.0, 1e4]
RATIOS = [1e-3, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 2.0, 2.35, 3.0, 10.0]
FIXED_X = [1e-8, 0.01, 0.5, 1.5, 5.0, 50.0, 300.0]
LARGE_SHAPES = [1e5, 1e6, 1e7]
CENTRAL_RATIOS = [0.99, 0.999, 1.0, 1.001, 1.01]


def series(a, x, dps):
    """P(a, x) by DLMF 8.11.4 at dps digits, summed until a term is below
    10^-dps of the sum once the terms decrease."""
    with mp.workdps(dps):
        a, x = mp.mpf(a), mp.mpf(x)
        front = mp.exp(a * mp.log(x) - x - mp.loggamma(a + 1))
        tol = mp.mpf(10) ** -dps
        term, total, n = mp.mpf(1), mp.mpf(1), 0
        while True:
            n += 1
            term *= x / (a + n)
            total += term
            if a + n > x and term < tol * total:
                return front * total


def by_gammainc(a, x):
    with mp.workdps(DIGITS):
        a_, x_ = mp.mpf(a), mp.mpf(x)
        return (+mp.gammainc(a_, 0, x_, regularized=True),
                +mp.gammainc(a_, x_, mp.inf, regularized=True))


def reference(a, x):
    try:
        lower, upper = by_gammainc(a, x)
        method = "gammainc"
    except mp.libmp.NoConvergence:
        lower = None
    if lower is not None and x < a:
        check = series(a, x, DIGITS)
        if abs(check - lower) > mp.mpf(10) ** -50 * lower:
            sys.exit(f"gammainc and the series disagree at a = {a}, x = {x}")
    if lower is None:
        # The smaller tail is about exp(-a (lambda - 1 - ln lambda)); Q = 1 - P
        # needs that many more digits.
        lam = mp.mpf(x) / a
        exponent = a * (lam - 1 - mp.log(lam)) / mp.log(10)
        dps = DIGITS + int(mp.ceil(exponent)) + 10
        lower = series(a, x, dps)
        with mp.workdps(dps):
            upper = 1 - lower
        method = "series"
    return lower, upper, method


def points():
    for a in SHAPES:
        xs = sorted(set([a * r for r in RATIOS] + FIXED_X))
        for x in xs:
            yield a, x
    for a in LARGE_SHAPES:
        for r in CENTRAL_RATIOS:
            yield a, a * r


cases = []
for a, x in points():
    if not 0 < x < float("inf"):
        continue
    lower, upper, method = reference(a, x)
    if min(lower, upper) < mp.mpf(10) ** -300:
        continue
    cases.append({"a": a, "x": x, "lower": float(lower), "upper": float(upper),
                  "method": method})


def truncated_exponential(n, x):
    x = Fraction(x)
    term, total = Fraction(1), Fraction(1)
    for k in range(1, n + 1):
        term = term * x / k
        total += term
    return total


closed = []
# P = 1 - Q is formed at a precision that leaves 60 digits for P down to 1e-300.
with mp.workdps(DIGITS + 300):
    for x in [0.25, 1.0, 2.5, 7.0, 20.0, 60.0, 150.0]:
        for n in [0, 1, 2, 4, 9, 19, 49, 99]:
            e = truncated_exponential(n, x)
            upper = mp.exp(-mp.mpf(x)) * mp.mpf(e.numerator) / mp.mpf(e.denominator)
            closed.append((n + 1.0, x, upper, "integer"))
            total, pochhammer = mp.mpf(0), mp.mpf(1)
            for k in range(1, n + 1):
                pochhammer *= mp.mpf(k) - mp.mpf(1) / 2
                total += mp.mpf(x) ** (k - mp.mpf(1) / 2) / pochhammer
            upper = mp.erfc(mp.sqrt(x)) + mp.exp(-mp.mpf(x)) / mp.sqrt(mp.pi) * total
            closed.append((n + 0.5, x, upper, "half-integer"))
    closed = [{"a": a, "x": x, "lower": float(1 - q), "upper": float(q), "method": m}
              for a, x, q, m in closed if min(q, 1 - q) >= mp.mpf(10) ** -300]

rng = random.Random(20261128)
recurrence = []
with mp.workdps(DIGITS):
    for scale in [1.0, 500.0]:
        for _ in range(100):
            a = (1.0 - rng.random()) * scale
            x = (1.0 - rng.random()) * scale
            front = mp.exp(mp.mpf(a) * mp.log(x) - x - mp.loggamma(mp.mpf(a) + 1))
            if front >= mp.mpf(10) ** -300:
                recurrence.append({"a": a, "x": x, "prefactor": float(front)})

json.dump({"versions": {"mpmath": mp.__version__, "digits": DIGITS},
           "cases": cases, "closed": closed, "recurrence": recurrence},
          sys.stdout, separators=(",", ":"))
sys.stdout.write("\n")
