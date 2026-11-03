"""
Up-and-out forward: derived leading coefficient and its identity with the barrier sensitivity.

Lattice (reflection, Lin-Palmer Lemma B.1(2)):  F_ui = J5 + J4F  with two tail sums whose thresholds depend on H only.
Lemma 4.1 of Lin-Palmer then gives (eps_n cancels):
   F_ui,n(K) = F_ui^cont(H;K) + c_ui * Delta_n/sqrt(n) + O(1/n),
   c_ui = 4 sqrt(T) h1(-d41,-d42) - 2 S phi(d31) (1 - K/H),
   F_ui^cont = S Phi(d31) - K e^{-rT} Phi(d32) + R [ H Phi(-d41) - (S K e^{-rT}/H) Phi(-d42) ],   R=(S/H)^(-2r/sigma^2)
and F_uo = (S - K e^{-rT}) - F_ui  (exact in the lattice: discounted S is a martingale).
Claim to prove here:   c_uo := -c_ui  =  + sigma sqrt(T) dF_uo^cont/dlogH.
"""
import sympy as sp
S0,K,H,T,s,r = sp.symbols('S0 K H T sigma r', positive=True)
Phi=lambda x:(1+sp.erf(x/sp.sqrt(2)))/2
phi=lambda x:sp.exp(-x**2/2)/sp.sqrt(2*sp.pi)
sT=s*sp.sqrt(T); k=r+s**2/2
d21=(sp.log(H**2/(S0*K))+k*T)/sT
d31=(sp.log(S0/H)+k*T)/sT; d32=d31-sT
d41=(sp.log(H/S0)+k*T)/sT; d42=d41-sT
al=(r-s**2/2)/(2*s); alh=al+s/2
R=(S0/H)**(-2*r/s**2)
h1=lambda x,y: R*(H*alh*Phi(x)-S0*K*sp.exp(-r*T)/H*al*Phi(y))
Fui = S0*Phi(d31)-K*sp.exp(-r*T)*Phi(d32)+R*(H*Phi(-d41)-S0*K*sp.exp(-r*T)/H*Phi(-d42))
Ftot= S0-K*sp.exp(-r*T)
Fuo = Ftot-Fui
c_ui = 4*sp.sqrt(T)*h1(-d41,-d42)-2*S0*phi(d31)*(1-K/H)
Z = (-c_ui) - sT*H*sp.diff(Fuo,H)          # want 0
Zs = sp.simplify(sp.expand_log(sp.powsimp(sp.expand(Z),force=True),force=True))
print("symbolic:  -c_ui - sigma sqrt(T) dF_uo/dlogH  ->", Zs)
