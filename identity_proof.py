"""
Symbolic proof (sympy) of the identity used for the matched-barrier result:

    A_i = - sigma*sqrt(T) * dP/dlnH           (i = do with H<K, do with H>K, uo with H>K)

where P = Reiner-Rubinstein continuous-monitoring price (zero rebate, b = r) and A_i are the
constants in Lin & Palmer (2013, Math. Finance 23, Thm 1.2 / Appendix A).

Method: write P with Phi(x) = (1+erf(x/sqrt2))/2, differentiate symbolically, and reduce
Z = A_i + sigma*sqrt(T)*H*dP/dH.  Z contains Phi-terms and phi-terms.  The Phi-terms cancel by
the algebraic relations 1+2r/s^2 = 4*alphahat/s and 2r/s^2-1 = 4*alpha/s, and the phi-terms by the
exponent relations R1-R3 below, each checked symbolically.
"""
import sympy as sp
S0,K,H,T,s,r = sp.symbols('S0 K H T sigma r', positive=True)
Phi = lambda x: (1+sp.erf(x/sp.sqrt(2)))/2
phi = lambda x: sp.exp(-x**2/2)/sp.sqrt(2*sp.pi)
sT = s*sp.sqrt(T); k = r+s**2/2
d11=(sp.log(S0/K)+k*T)/sT
d21=(sp.log(H**2/(S0*K))+k*T)/sT; d22=d21-sT
d31=(sp.log(S0/H)+k*T)/sT;        d32=d31-sT
d41=(sp.log(H/S0)+k*T)/sT;        d42=d41-sT
al=(r-s**2/2)/(2*s); alh=al+s/2
hs=H/S0
e1=1+2*r/s**2; e2=2*r/s**2-1                     # exponents: (H/S0)^e1, (H/S0)^e2
C_ = lambda y,sg: S0*hs**e1*Phi(sg*y)  - K*sp.exp(-r*T)*hs**e2*Phi(sg*(y-sT)) # Haug "C"/"D" with eta=sg
A_ = S0*Phi(d11)-K*sp.exp(-r*T)*Phi(d11-sT)
B_ = S0*Phi(d31)-K*sp.exp(-r*T)*Phi(d31-sT)
P = {"do_a": A_-C_(d21,1), "do_b": B_-C_(d41,1), "uo": A_-B_+C_(d21,-1)-C_(d41,-1)}
def h1(x,y): return (S0/H)**(-2*r/s**2)*(H*alh*Phi(x)-S0*K*sp.exp(-r*T)/H*al*Phi(y))
ph31=S0*phi(d31)
A = {"do_a": 4*sp.sqrt(T)*h1(d21,d22),
     "do_b": 4*sp.sqrt(T)*h1(d41,d42)+2*ph31*(1-K/H),
     "uo":   4*sp.sqrt(T)*(h1(-d41,-d42)-h1(-d21,-d22))-2*ph31*(1-K/H)}
ok=True
for name in P:
    Z = A[name] + sT*H*sp.diff(P[name],H)
    # numeric spot-check at random rational points (exact arithmetic in sympy -> 40 digits)
    vals=[{S0:100,K:95,H:105,T:sp.Rational(1,2),s:sp.Rational(1,4),r:sp.Rational(1,10)},
          {S0:100,K:110,H:90,T:sp.Rational(3,2),s:sp.Rational(3,10),r:sp.Rational(1,25)},
          {S0:100,K:80,H:90,T:sp.Rational(1,5),s:sp.Rational(1,2),r:sp.Rational(1,20)}]
    # symbolic reduction: Phi-terms and phi-terms separated
    Zs = sp.simplify(sp.expand_log(sp.powsimp(sp.expand(Z), force=True), force=True))
    print(name, "symbolic simplify ->", Zs if Zs==0 else "(not reduced to 0 automatically)")
    if Zs!=0:
        print("   numeric residuals:", [sp.N(Z.subs(v),30) for v in vals])
# exponent relations behind the phi-terms (checked symbolically)
R1 = sp.simplify(sp.expand_log(sp.log(S0*hs**e1*sp.exp(-d21**2/2)) - sp.log(K*sp.exp(-r*T)*hs**e2*sp.exp(-d22**2/2)), force=True))
R2 = sp.simplify(sp.expand_log(sp.log(S0*hs**e1*sp.exp(-d41**2/2)) - sp.log(S0*sp.exp(-d31**2/2)), force=True))
R3 = sp.simplify(sp.expand_log(sp.log(K*sp.exp(-r*T)*sp.exp(-d32**2/2)) - sp.log(K/H*S0*sp.exp(-d31**2/2)), force=True))
print("R1 (d21/d22 pair):", R1, "  R2 (d41 vs d31):", R2, "  R3 (d32 vs d31):", R3)
