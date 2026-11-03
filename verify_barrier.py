#!/usr/bin/env python3
"""
Verification suite for the reflection-principle CRR knock-out formulas.

  1. Brute-force path enumeration vs closed-form count V(n,j)  (n <= 16)
  2. Where does the Pascal-type recurrence V(n,j)=V(n-1,j)+V(n-1,j-1) fail?
  3. Closed-form price vs CRR backward induction (same barrier rule)
  4. Log-space implementation (stable for n up to 10^6) + runtime table
  5. Paper scenarios 1 & 2 reproduced
"""
import itertools, math, time
import numpy as np
from scipy.special import gammaln, comb

# ------------------------------------------------------------------ counts
def V_naive(n, j, sk):
    """Paper formula: V(n,j) = <<C(n,j) - C(n,j-sk)>>, sk = sigma*k_n (sigma=+1 up, -1 down)."""
    def C(a, b): return math.comb(a, b) if 0 <= b <= a else 0
    return max(C(n, j) - C(n, j - sk), 0)

def brute_counts(n, level):
    """Survivor counts per j. level>0: up barrier at +level; level<0: down at level."""
    cnt = [0] * (n + 1)
    for steps in itertools.product((1, -1), repeat=n):
        x, ok = 0, True
        for s in steps:
            x += s
            if (level > 0 and x >= level) or (level < 0 and x <= level):
                ok = False; break
        if ok: cnt[steps.count(1)] += 1
    return cnt

def test_counts(nmax=14):
    bad = 0; total = 0
    for n in range(1, nmax + 1):
        for lvl in list(range(1, n + 3)) + list(range(-1, -(n + 3), -1)):
            bf = brute_counts(n, lvl)
            for j in range(n + 1):
                total += 1
                # down case: <<C(n,j)-C(n,j+k)>> == V_naive(n,j,sk=-k)
                if V_naive(n, j, lvl) != bf[j]:
                    bad += 1
                    print("  MISMATCH", n, lvl, j, V_naive(n, j, lvl), bf[j])
    print(f"[1] brute force vs closed-form count: {total} cases, {bad} mismatches")

# --------------------------------------------------------------- recurrence
def test_recurrence(nmax=12):
    fails = []
    for kn in range(1, 6):
        for n in range(1, nmax + 1):
            for j in range(0, n + 1):
                lhs = V_naive(n, j, kn)
                rhs = V_naive(n - 1, j, kn) + V_naive(n - 1, j - 1, kn)
                if lhs != rhs: fails.append((kn, n, j, 2 * j - n, lhs, rhs))
    print(f"[2] fixed-k Pascal recurrence failures (k=1..5, n<=12): {len(fails)}")
    on_row = sum(1 for f in fails if f[3] == f[0])
    print(f"    failures with 2j-n == k (absorbing row): {on_row} of {len(fails)}")
    for f in fails[:6]:
        print(f"    k={f[0]} n={f[1]} j={f[2]} m={f[3]}  V(n,j)={f[4]}  V(n-1,j)+V(n-1,j-1)={f[5]}")
    # also: with k_n varying with n
    print("    (note: the paper's k_n changes with n, so n-1 and n use different barriers)")

# ------------------------------------------------------------------ pricing
def params(S, T, v, r, n):
    dt = T / n
    u = math.exp(v * math.sqrt(dt)); d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    return u, d, p

def _ceil_tol(a, tol=1e-12):
    """ceil(a), but a within tol of an integer is treated as that integer."""
    i = round(a)
    return i if abs(a - i) <= tol * max(1.0, abs(a)) else math.ceil(a)

def barrier_index(S, Hbar, u, kind):
    """k_n >= 1: lattice steps from S to the first node AT or beyond the barrier (paper, Section 2).
    A node exactly on the barrier kills (QuantLib/MATLAB/Lin-Palmer). Effective barrier: Ht = S*u**(sigma*k_n)."""
    if kind == "up":   return _ceil_tol(math.log(Hbar / S) / math.log(u))
    else:              return _ceil_tol(math.log(S / Hbar) / math.log(u))

def signed_index(kn, kind):
    """sigma*k_n with sigma=+1 (up), -1 (down)."""
    return kn if kind == "up" else -kn

def price_backward(S, K, T, v, r, n, Hbar, kind, opt):
    """CRR backward induction; node killed when its level index reaches the same barrier index."""
    u, d, p = params(S, T, v, r, n)
    kn = barrier_index(S, Hbar, u, kind); sk = signed_index(kn, kind)
    disc = math.exp(-r * T / n)
    lv = np.arange(-n, n + 1, 2)                       # level at step n (X = 2j-n)
    ST = S * u ** lv
    pay = np.maximum(ST - K, 0) if opt == "call" else np.maximum(K - ST, 0)
    dead = (lv >= sk) if kind == "up" else (lv <= sk)
    val = np.where(dead, 0.0, pay)
    for t in range(n - 1, -1, -1):
        lv = np.arange(-t, t + 1, 2)
        val = disc * (p * val[1:] + (1 - p) * val[:-1])
        dead = (lv >= sk) if kind == "up" else (lv <= sk)
        val = np.where(dead, 0.0, val)
    return float(val[0])

def price_closed_naive(S, K, T, v, r, n, Hbar, kind, opt):
    u, d, p = params(S, T, v, r, n)
    kn = barrier_index(S, Hbar, u, kind); sk = signed_index(kn, kind)
    tot = 0.0
    for j in range(n + 1):
        w = V_naive(n, j, sk) * p ** j * (1 - p) ** (n - j)
        ST = S * u ** j * d ** (n - j)
        pay = max(ST - K, 0) if opt == "call" else max(K - ST, 0)
        tot += w * pay
    return math.exp(-r * T) * tot

def price_closed_log(S, K, T, v, r, n, Hbar, kind, opt):
    """Vectorised, log-space. Survival ratio = C(n,j-sigma*k_n)/C(n,j) in log form."""
    u, d, p = params(S, T, v, r, n)
    kn = barrier_index(S, Hbar, u, kind); sk = signed_index(kn, kind)
    j = np.arange(n + 1, dtype=float)
    logC = gammaln(n + 1) - gammaln(j + 1) - gammaln(n - j + 1)
    logw = logC + j * math.log(p) + (n - j) * math.log(1 - p)
    m = 2 * j - n
    ir = j - sk                                         # index of the reflected binomial
    valid_k = (ir >= 0) & (ir <= n)
    kk = np.where(valid_k, ir, 0)
    logratio = (gammaln(j + 1) + gammaln(n - j + 1)
                - gammaln(kk + 1) - gammaln(n - kk + 1))  # log[C(n,ir)/C(n,j)] without logC cancel
    logratio = np.where(valid_k, logratio, -np.inf)
    alive = (m < kn) if kind == "up" else (m > -kn)
    # on alive nodes C(n,ir) <= C(n,j), so logratio <= 0; clip avoids overflow on dead nodes
    surv = np.where(alive, -np.expm1(np.minimum(logratio, 0.0)), 0.0)   # 1 - ratio, stable
    surv = np.where(alive, np.maximum(surv, 0.0), 0.0)
    with np.errstate(divide="ignore"):
        logsurv = np.log(surv)
    ST = S * np.exp(m * math.log(u))
    pay = np.maximum(ST - K, 0) if opt == "call" else np.maximum(K - ST, 0)
    return math.exp(-r * T) * float(np.sum(np.exp(logw + logsurv) * pay))

def test_prices():
    cases = [(130, 140, 36 / 365, 0.45, 0.04, 170, "up"), (130, 140, 36 / 365, 0.45, 0.04, 100, "down"),
             (100, 95, 0.5, 0.3, 0.03, 130, "up"), (100, 105, 0.5, 0.3, 0.03, 80, "down")]
    worst = 0
    print("[3] closed form vs backward induction (same barrier rule):")
    for (S, K, T, v, r, Hb, kind) in cases:
        for opt in ("call", "put"):
            for n in (50, 200, 800):
                a = price_backward(S, K, T, v, r, n, Hb, kind, opt)
                b = price_closed_naive(S, K, T, v, r, n, Hb, kind, opt)
                c = price_closed_log(S, K, T, v, r, n, Hb, kind, opt)
                worst = max(worst, abs(a - b), abs(a - c))
    print(f"    max |backward - closed| over all cases = {worst:.3e}")

def test_scenarios():
    print("[5] paper scenarios (closed form, log-space):")
    S, K, T, v, r = 130, 140, 36 / 365, 0.45, 0.04      # same T as the paper's Tables 1-2
    for lab, Hu, Hd in (("Scenario 1", 170, 100), ("Scenario 2", 155, 120)):
        for n in (1000, 10000):
            row = [price_closed_log(S, K, T, v, r, n, Hu, "up", "call"),
                   price_closed_log(S, K, T, v, r, n, Hd, "down", "call"),
                   price_closed_log(S, K, T, v, r, n, Hd, "down", "put"),
                   price_closed_log(S, K, T, v, r, n, Hu, "up", "put")]
            print(f"    {lab} n={n:<6} Cupout={row[0]:.6f} Cdownout={row[1]:.6f} Pdownout={row[2]:.6f} Pupout={row[3]:.6f}")

def test_runtime():
    print("[4] runtime, scenario 1 Cupout (seconds):")
    S, K, T, v, r, Hu = 130, 140, 36 / 365, 0.45, 0.04, 170
    print(f"    {'n':>8} {'closed(log)':>12} {'backward':>12} {'price diff':>12}")
    for n in (1000, 4000, 16000, 64000, 1000000):
        t = time.time(); a = price_closed_log(S, K, T, v, r, n, Hu, "up", "call"); ta = time.time() - t
        if n <= 16000:
            t = time.time(); b = price_backward(S, K, T, v, r, n, Hu, "up", "call"); tb = time.time() - t
            print(f"    {n:>8} {ta:>12.4f} {tb:>12.4f} {abs(a-b):>12.2e}")
        else:
            print(f"    {n:>8} {ta:>12.4f} {'(skipped)':>12} {'-':>12}   price={a:.6f}")

if __name__ == "__main__":
    test_counts(12)
    test_recurrence(12)
    test_prices()
    test_runtime()
    test_scenarios()
