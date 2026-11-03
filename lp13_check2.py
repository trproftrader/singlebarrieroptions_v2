import math
from verify_barrier import price_closed_log
from convergence_analysis import rr_price
print("(1) Reproduce Lin-Palmer (2013) Section 2 tables with our closed-form lattice code")
print("    down-and-out call: S0=100 K=100 H=95 sigma=.25 r=.10 T=1   (paper: 7.467, 7.231, 7.176, 7.064, 7.060; BS 7.050)")
for n in (1000,5000,10000,50000,100000):
    print("   n=%6d  ours = %.3f" % (n, price_closed_log(100,100,1,.25,.10,n,95,"down","call")))
print("    up-and-out call:   S0=100 K=95 H=105                  (paper: 0.047, 0.035, 0.036, 0.035, 0.034; BS 0.034)")
for n in (1000,5000,10000,50000,100000):
    print("   n=%6d  ours = %.3f" % (n, price_closed_log(100,95,1,.25,.10,n,105,"up","call")))
print("   BS ours: do=%.3f uo=%.3f" % (rr_price(100,100,1,.25,.10,95,"down","call"), rr_price(100,95,1,.25,.10,105,"up","call")))
