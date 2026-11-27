# Reference values of the regularised incomplete beta function I_x(a, b) and
# of its complement 1 - I_x(a, b) = I_y(b, a) at 50 significant digits, for
# epistat_stats::dist::beta::regularised.
#
# Shapes run from 1e-3 to 1e7 and x from the far tails to the centre of each
# distribution. The smaller of x and y = 1 - x is a double; where the larger
# is not exactly 1 minus it, the reference is the value at the smaller, which
# is the one the function reads (it forms both logarithms from it).
#
# The values are mpmath's betainc(a, b, 0, x, regularized=True) and
# betainc(b, a, 0, y, regularized=True), at a precision that holds both x and
# y exactly, where that finishes quickly.
# With both shapes above 2000 mpmath's general algorithm takes minutes, and
# the value is the continued fraction DLMF 8.17.22 at 70 digits by forward
# recurrence, run until successive convergents agree to 1e-60 twice in a row,
# times x^a y^b / (a B(a, b)) from mpmath's loggamma, on the side of
# (a + 1) / (a + b + 2) where the fraction converges fast; the other tail is
# one minus it and is kept where it is above 1e-40. The two methods agree to
# 1e-47 on every case of this grid with min(a, b) <= 2000, which the script
# checks.
#
# Run: python3 beta_mpmath.py > beta_mpmath.json   (mpmath from PyPI)
import json
import sys

import mpmath as mp

mp.mp.dps = 50

SHAPES = [1e-3, 0.02, 0.3, 1.0, 2.5, 8.0, 30.0, 120.0, 500.0, 2000.0,
          1e4, 1e5, 1e6, 1e7]
CROSS_CHECK = 2000.0


def coefficient(n, a, b, x):
    m = n // 2
    if n % 2 == 0:
        return m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
    return -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))


def fraction(a, b, x):
    """1 + d1 / (1 + d2 / (1 + ...)) by forward recurrence, until three
    successive convergents agree: one step changes the value very little where
    d_2m is small, and two steps do not both do so before convergence."""
    tol = mp.mpf(10) ** -60
    a_prev, a_cur = mp.mpf(1), mp.mpf(1)
    b_prev, b_cur = mp.mpf(0), mp.mpf(1)
    last = before = mp.mpf(1)
    n = 0
    while True:
        n += 1
        d = coefficient(n, a, b, x)
        a_prev, a_cur = a_cur, a_cur + d * a_prev
        b_prev, b_cur = b_cur, b_cur + d * b_prev
        a_prev, a_cur, b_prev = a_prev / b_cur, a_cur / b_cur, b_prev / b_cur
        b_cur = mp.mpf(1)
        if (n > 2 and abs(a_cur - last) <= tol * abs(a_cur)
                and abs(last - before) <= tol * abs(a_cur)):
            return a_cur
        before, last = last, a_cur


def by_fraction(a, b, s, small_is_x):
    """(I_x(a, b), 1 - I_x(a, b)) by the fraction; s is the smaller of x, y."""
    with mp.workdps(70):
        a, b, s = mp.mpf(a), mp.mpf(b), mp.mpf(s)
        x, y = (s, 1 - s) if small_is_x else (1 - s, s)
        ln_x, ln_y = (mp.log(s), mp.log1p(-s)) if small_is_x else (mp.log1p(-s), mp.log(s))
        ln_front = (a * ln_x + b * ln_y
                    - mp.loggamma(a) - mp.loggamma(b) + mp.loggamma(a + b))
        if x < (a + 1) / (a + b + 2):
            value = mp.exp(ln_front) / (a * fraction(a, b, x))
            return value, 1 - value
        complement = mp.exp(ln_front) / (b * fraction(b, a, y))
        return 1 - complement, complement


def by_betainc(a, b, s, small_is_x):
    """(I_x(a, b), I_y(b, a)), each an integral from 0, at a precision that
    holds 1 - s exactly and leaves 50 digits for the result."""
    bits = 53 + 200 - mp.floor(mp.log(s, 2))
    with mp.workprec(int(bits)):
        s = mp.mpf(s)
        x, y = (s, 1 - s) if small_is_x else (1 - s, s)
        return (+mp.betainc(a, b, 0, x, regularized=True),
                +mp.betainc(b, a, 0, y, regularized=True))


def reference(a, b, s, small_is_x):
    """The two tails, and the method; the fraction forms one of them as a
    difference, which is kept only while it is above 1e-40."""
    if min(a, b) <= CROSS_CHECK:
        try:
            value, complement = by_betainc(a, b, s, small_is_x)
        except ValueError:
            # mpmath gives up on a tail far below 1e-300, which is not kept.
            return None
        check, check_complement = by_fraction(a, b, s, small_is_x)
        tol = mp.mpf(10) ** -47
        for v, c in ((value, check), (complement, check_complement)):
            if v > mp.mpf(10) ** -40 and abs(v - c) > tol * v:
                sys.exit(f"betainc and the fraction disagree at {a} {b} {s} {small_is_x}")
        return value, complement, "betainc"
    value, complement = by_fraction(a, b, s, small_is_x)
    if min(value, complement) < mp.mpf(10) ** -40:
        return None
    return value, complement, "fraction"


def points(a, b):
    """The smaller of x and y, and which one it is, for a grid of x."""
    mean = a / (a + b)
    sd = (a * b / ((a + b) ** 2 * (a + b + 1))) ** 0.5
    xs = [mean + k * sd for k in (-6, -3, -1, -0.3, 0, 0.3, 1, 3, 6)]
    xs += [(a + 1) / (a + b + 2) * f for f in (0.999, 1.0, 1.001)]
    out = []
    for x in xs:
        if 0 < x < 1:
            out.append((x, True) if x <= 0.5 else (1 - x, False))
    for j in (1, 3, 8, 30, 100):
        out.append((mean * 10.0 ** -j, True))
        out.append(((1 - mean) * 10.0 ** -j, False))
    return out


def dyadic(s):
    # A multiple of 2^-53 at most 1/2 has an exact complement.
    return round(s * 2.0 ** 53) / 2.0 ** 53 if s > 2.0 ** -20 else s


cases = []
for a in SHAPES:
    for b in SHAPES:
        seen = set()
        for s, small_is_x in points(a, b):
            s = dyadic(s)
            if not 0 < s <= 0.5 or (s, small_is_x) in seen:
                continue
            seen.add((s, small_is_x))
            found = reference(a, b, s, small_is_x)
            if found is None:
                continue
            value, complement, method = found
            if min(value, complement) < mp.mpf(10) ** -300:
                continue
            x, y = (s, 1.0 - s) if small_is_x else (1.0 - s, s)
            cases.append({
                "a": a, "b": b, "x": x, "y": y,
                "value": float(value), "complement": float(complement),
                "method": method,
            })

json.dump({"versions": {"mpmath": mp.__version__, "digits": 50},
           "cases": cases}, sys.stdout, separators=(",", ":"))
sys.stdout.write("\n")
