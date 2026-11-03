"""
Check Lin-Palmer (2013) Theorem 1.2 constants (Appendix A) against
(1) the identity  A = -/+ sigma*sqrt(T) * dP_cont/dlogH   (the claim that the O(1/sqrt n) term
    is exactly the first-order effect of moving the barrier to the lattice barrier H~), and
(2) the exact lattice price.
"""
import math, numpy as np
from scipy.stats import norm
from convergence_analysis import rr_price
from verify_barrier import price_closed_log

Phi = norm.cdf
def consts(S0,K,T,s,r,H):
    sT = s*math.sqrt(T); R=math.sqrt(T)
    d11=(math.log(S0/K)+(r+s*s/2)*T)/sT
    d21=(math.log(H*H/(S0*K))+(r+s*s/2)*T)/sT
    d31=(math.log(S0/H)+(r+s*s/2)*T)/sT
    d41=(math.log(H/S0)+(r+s*s/2)*T)/sT
    d12,d22,d32,d42=[x-sT for x in (d11,d21,d31,d41)]
    al=(r-s*s/2)/(2*s); alh=al+s/2
    be=(s**4-4*s*s*r+12*r*r)/(48*s); beh=-be-s*r/6
    I=(4*be+16*al**3/3)*math.log(S0/H)*T/s
    def h(i,x,y): return (S0/H)**(-2*r/s**2)*(H*alh**i*Phi(x)-S0*K*math.exp(-r*T)/H*al**i*Phi(y))
    phi31=S0/math.sqrt(2*math.pi)*math.exp(-d31**2/2)
    A1=4*R*h(1,d21,d22)
    A2=4*R*h(1,d41,d42)+2*phi31*(1-K/H)
    A3=4*R*(h(1,-d41,-d42)-h(1,-d21,-d22))-2*phi31*(1-K/H)
    return dict(A1=A1,A2=A2,A3=A3)

def dP_dlogH(S0,K,T,s,r,H,kind,opt,eps=1e-5):
    f=lambda x: rr_price(S0,K,T,s,r,H*math.exp(x),kind,opt)
    return (f(eps)-f(-eps))/(2*eps)

if __name__=="__main__":
    print("Identity check:  A_i  vs  -sigma*sqrt(T)*dP/dlogH")
    tests=[("do (H<K)","A1","down",(100,100,1,.25,.10,95)),
           ("do (H>K)","A2","down",(100,95,1,.25,.10,98)),
           ("uo (H>K)","A3","up",  (100,95,1,.25,.10,105)),
           ("do (H<K)","A1","down",(130,140,.2,.45,.04,100)),
           ("do (H>K)","A2","down",(130,100,.2,.45,.04,115)),
           ("uo (H>K)","A3","up",  (130,140,.2,.45,.04,170))]
    for nm,key,kind,(S0,K,T,s,r,H) in tests:
        c=consts(S0,K,T,s,r,H)[key]
        dp=dP_dlogH(S0,K,T,s,r,H,kind,"call")
        print(f"{nm:9} {key}: A={c:12.6f}   -sigma*sqrtT*dP/dlogH={-s*math.sqrt(T)*dp:12.6f}   ratio={c/(-s*math.sqrt(T)*dp):.6f}")
