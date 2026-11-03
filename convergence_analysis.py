#!/usr/bin/env python3
"""
Convergence analysis of the reflection-principle CRR knock-out price.

Lattice barrier sits on node S*u^(+-k_n) (k_n = ceil(a_n), a_n = |log(H/S)|/(v*sqrt(dt))).
Offset Delta_n = k_n - a_n in [0,1)  is the lattice-vs-barrier misalignment.
Prediction: price error ~ (delta - beta) * v*sqrt(dt) * dPrice/dlogH,
with beta = 0.5826 (Broadie-Glasserman-Kou).  We test that numerically.
"""
import math, numpy as np
from scipy.stats import norm
from scipy.special import gammaln
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from verify_barrier import price_closed_log

BETA = 0.5826

def rr_price(S, K, T, v, r, H, kind, opt):
    """Reiner-Rubinstein / Haug continuous-monitoring knock-out, b=r, no rebate."""
    b = r; sg = v * math.sqrt(T)
    mu = (b - v * v / 2) / v**2
    phi = 1 if opt == "call" else -1
    eta = 1 if kind == "down" else -1
    x1 = math.log(S / K) / sg + (1 + mu) * sg
    x2 = math.log(S / H) / sg + (1 + mu) * sg
    y1 = math.log(H * H / (S * K)) / sg + (1 + mu) * sg
    y2 = math.log(H / S) / sg + (1 + mu) * sg
    N = norm.cdf; df = math.exp(-r * T); hs = H / S
    A = phi*S*N(phi*x1) - phi*K*df*N(phi*x1 - phi*sg)
    B = phi*S*N(phi*x2) - phi*K*df*N(phi*x2 - phi*sg)
    C = phi*S*hs**(2*(mu+1))*N(eta*y1) - phi*K*df*hs**(2*mu)*N(eta*y1 - eta*sg)
    D = phi*S*hs**(2*(mu+1))*N(eta*y2) - phi*K*df*hs**(2*mu)*N(eta*y2 - eta*sg)
    if opt == "call" and kind == "down": return A - C if K > H else B - D
    if opt == "call" and kind == "up":   return 0.0 if K >= H else A - B + C - D
    if opt == "put"  and kind == "down": return 0.0 if K <= H else A - B + C - D
    if opt == "put"  and kind == "up":   return B - D if K > H else A - C

def sweep(S, K, T, v, r, Hb, kind, opt, ns):
    cont = rr_price(S, K, T, v, r, Hb, kind, opt)
    rows = []
    for n in ns:
        dt = T / n; sd = v * math.sqrt(dt)
        a = abs(math.log(Hb / S)) / sd
        kn = math.ceil(a - 1e-12)         # k_n: first node at or beyond the barrier
        delta = kn - a
        lat = price_closed_log(S, K, T, v, r, n, Hb, kind, opt)
        # effective lattice barrier and BGK-equivalent continuous barrier
        Ht = S * math.exp(kn * sd) if kind == "up" else S * math.exp(-kn * sd)   # effective barrier H~_n
        Hbgk = Ht * math.exp(-BETA * sd) if kind == "up" else Ht * math.exp(BETA * sd)
        cont_bgk = rr_price(S, K, T, v, r, Hbgk, kind, opt)
        cont_eff = rr_price(S, K, T, v, r, Ht, kind, opt)
        rows.append((n, delta, lat, lat - cont, lat - cont_bgk, lat - cont_eff, sd))
    return cont, np.array(rows)

def analyse(name, S, K, T, v, r, Hb, kind, opt, ns, ax_row):
    cont, R = sweep(S, K, T, v, r, Hb, kind, opt, ns)
    n, delta, lat, e_raw, e_bgk, e_eff, sd = R.T
    # regress sqrt(n)*e_raw on delta : zero crossing -> effective beta
    A = np.vstack([delta, np.ones_like(delta)]).T
    slope, icpt = np.linalg.lstsq(A, e_raw / sd, rcond=None)[0]   # error / (v sqrt(dt)) vs delta
    beta_hat = -icpt / slope
    # envelope decay exponent: fit log|e| ~ -alpha log n on upper envelope (max in blocks)
    def alpha(e):
        blocks = np.array_split(np.arange(len(n)), 12)
        x = np.log([n[b].mean() for b in blocks]); y = np.log([np.abs(e[b]).max() for b in blocks])
        return -np.polyfit(x, y, 1)[0]
    print(f"{name:<22} continuous={cont:9.6f}  beta_hat={beta_hat:6.3f}  "
          f"max|err|: vs H={np.abs(e_raw).max():.4f}, BGK-shift={np.abs(e_bgk).max():.4f}, "
          f"Ht(beta=0)={np.abs(e_eff).max():.5f} | decay exp: H={alpha(e_raw):.2f}, Ht={alpha(e_eff):.2f}")
    ax_row[0].plot(n, lat, lw=.7, label="lattice"); ax_row[0].axhline(cont, c="k", ls="--", lw=.8, label="continuous")
    ax_row[0].set_title(name); ax_row[0].set_xlabel("n"); ax_row[0].set_ylabel("price"); ax_row[0].legend(fontsize=7)
    ax_row[1].plot(n, e_raw, lw=.7, label="vs continuous H"); ax_row[1].plot(n, e_bgk, lw=.7, label="vs BGK-shifted H"); ax_row[1].plot(n, e_eff, lw=.9, c='k', label="vs Ht = H~_n (beta=0)")
    ax_row[1].axhline(0, c="k", lw=.5); ax_row[1].set_xlabel("n"); ax_row[1].set_ylabel("error"); ax_row[1].legend(fontsize=7)
    ax_row[2].scatter(delta, e_raw/sd, s=3); xx = np.linspace(0, 1, 50)
    ax_row[2].plot(xx, slope*xx + icpt, "r", lw=.8); ax_row[2].axvline(BETA, c="g", ls=":", lw=.8)
    ax_row[2].axhline(0, c="k", lw=.5); ax_row[2].set_xlabel(r"$\Delta_n = k_n - a_n$"); ax_row[2].set_ylabel("error / (v sqrt(dt))")
    return beta_hat

if __name__ == "__main__":
    S, K, T, v, r = 130, 140, 36 / 365, 0.45, 0.04   # same T as the paper's tables
    ns = np.arange(100, 3001, 7)
    cases = [("Up-out call, Hu=170", 170, "up", "call"), ("Down-out call, Hd=100", 100, "down", "call"),
             ("Down-out put, Hd=100", 100, "down", "put"), ("Up-out put, Hu=170", 170, "up", "put"),
             ("Up-out call, Hu=155", 155, "up", "call"), ("Down-out call, Hd=120", 120, "down", "call")]
    fig, axs = plt.subplots(len(cases), 3, figsize=(13, 3*len(cases)))
    print("Case                   continuous price / estimated beta / error envelopes")
    betas = [analyse(nm, S, K, T, v, r, Hb, kd, op, ns, axs[i]) for i, (nm, Hb, kd, op) in enumerate(cases)]
    plt.tight_layout(); plt.savefig("convergence.png", dpi=110)
    print(f"\nmean beta_hat = {np.mean(betas):.3f}  (theory: 0.5826)")
