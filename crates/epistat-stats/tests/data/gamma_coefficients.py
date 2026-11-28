# The coefficient tables of epistat_stats::dist::gamma.
#
# ETA_RATIO: d_n, n = 0 ... 26, of eta / (lambda - 1) = sum d_n eta^n, where
# eta^2 / 2 = lambda - 1 - ln(lambda) with the sign of lambda - 1 (DLMF
# 8.12.1, 8.12.2). With mu = lambda - 1, eta^2 / 2 = mu^2/2 - mu^3/3 + ...,
# so eta = mu sqrt(1 + w(mu)) with w = sum_{k>=1} 2 (-1)^k mu^k / (k + 2).
# The series of eta in mu is reverted to mu = eta + eta^2/3 + eta^3/36 - ...
# (DLMF 8.12.13, 8.12.14) and d = eta / mu is its reciprocal, all in exact
# rational arithmetic. The script checks the first coefficients against
# DLMF 8.12.14 and Gil, Segura and Temme (2012, eq. 2.37), and the values
# c_k(0) = -1/3, -1/540, 25/6048, 101/155520 (DLMF 8.12.17) that the
# recursion of their eqs. 2.33-2.40 gives at eta = 0.
#
# RECIPROCAL_GAMMA: g_k, k = 1 ... 32, of 1 / Gamma(1 + a) - 1 = sum g_k a^k
# (DLMF 5.7.1), the Taylor coefficients of mpmath's rgamma at 40 digits.
#
# Run: python3 gamma_coefficients.py   (mpmath from PyPI); it prints both
# tables as Rust literals, each the double nearest the exact value.
from fractions import Fraction as F

import mpmath as mp

TERMS = 27


def mul(p, q, n):
    r = [F(0)] * n
    for i, a in enumerate(p[:n]):
        if a:
            for j, b in enumerate(q[:n - i]):
                r[i + j] += a * b
    return r


def eta_ratio(n_terms):
    m = n_terms + 2
    w = [F(0)] + [F(2 * (-1) ** k, k + 2) for k in range(1, m)]
    root = [F(1)] + [F(0)] * (m - 1)       # sqrt(1 + w), binomial series
    power, binomial = [F(1)] + [F(0)] * (m - 1), F(1)
    for j in range(1, m):
        binomial = binomial * (F(1, 2) - (j - 1)) / j
        power = mul(power, w, m)
        root = [r + binomial * p for r, p in zip(root, power)]
    eta = [F(0)] + root[:m - 1]            # eta as a series in mu
    mu = [F(0), F(1)] + [F(0)] * (m - 2)   # mu as a series in eta
    for n in range(2, m):
        composed, power = [F(0)] * m, [F(1)] + [F(0)] * (m - 1)
        for k in range(1, m):
            power = mul(power, mu, m)
            composed = [c + eta[k] * p for c, p in zip(composed, power)]
        mu[n] -= composed[n]
    ratio = mu[1:] + [F(0)]                # mu / eta
    d = [F(1)] + [F(0)] * (m - 1)
    for n in range(1, m):
        d[n] = -sum(ratio[k] * d[n - k] for k in range(1, n + 1))
    return mu, d[:n_terms]


mu, d = eta_ratio(TERMS)
assert mu[1:9] == [F(1), F(1, 3), F(1, 36), F(-1, 270), F(1, 4320),
                   F(1, 17010), F(-139, 5443200), F(1, 204120)]
assert d[:4] == [F(1), F(-1, 3), F(1, 12), F(-2, 135)]


def c_at_zero(k_max):
    """c_k(0) from the backward recursion beta_n = (n + 2) beta_{n+2} / a +
    d_{n+1} as a series in 1/a: S_a(0) = a / (a + beta_1) beta_0."""
    # Work with polynomials in u = 1/a, truncated at degree k_max.
    size = k_max + 1
    beta = [[F(0)] * size for _ in range(TERMS + 1)]
    for n in range(TERMS - 2, -1, -1):
        shifted = [F(0)] + [(n + 2) * c for c in beta[n + 2][:size - 1]]
        beta[n] = [s + (d[n + 1] if i == 0 else 0) for i, s in enumerate(shifted)]
    # a / (a + beta_1) = 1 / (1 + u beta_1)
    denom = [F(1)] + beta[1][:size - 1]
    inverse = [F(1)] + [F(0)] * (size - 1)
    for i in range(1, size):
        inverse[i] = -sum(denom[k] * inverse[i - k] for k in range(1, i + 1))
    return mul(inverse, beta[0], size)


assert c_at_zero(3) == [F(-1, 3), F(-1, 540), F(25, 6048), F(101, 155520)]

print("ETA_RATIO")
for v in d:
    print(f"    {float(v)!r},")

mp.mp.dps = 40
print("RECIPROCAL_GAMMA")
for v in mp.taylor(lambda z: mp.rgamma(1 + z), 0, 32)[1:]:
    print(f"    {float(v)!r},")
