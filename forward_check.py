"""
Down-out puts via the knock-out forward.
 (1) forward expansion for ARBITRARY K:  F_n(K) = F(H) + A2(K)*Delta_n/sqrt(n) + O(1/n)
 (2) put expansion:                      P_n   = P(H) + (A1-A2)*Delta_n/sqrt(n) + O(1/n)
Lattice quantities are exact closed-form sums; A1, A2 are the Lin-Palmer constants.
"""
import math, numpy as np
from scipy.special import gammaln
from mpmath import mp
import hp_identity as hp          # reuses consts() and prices() (mpmath)  -- runs its own check on import
mp.dps = 30

def lattice(S,K,T,v,r,n,Hbar,opt):
    """down-out lattice price, touch kills at k_n = ceil(a_n) (Lin-Palmer convention); opt in call/put/fwd"""
    dt=T/n; sd=v*math.sqrt(dt); u=math.exp(sd); p=(math.exp(r*dt)-1/u)/(u-1/u)
    a=math.log(S/Hbar)/sd; kn=math.ceil(a); Delta=kn-a
    j=np.arange(n+1,dtype=float); m=2*j-n
    logw=gammaln(n+1)-gammaln(j+1)-gammaln(n-j+1)+j*math.log(p)+(n-j)*math.log(1-p)
    kk=j+kn; valid=kk<=n; kc=np.where(valid,kk,0)
    lr=np.where(valid,gammaln(j+1)+gammaln(n-j+1)-gammaln(kc+1)-gammaln(n-kc+1),-np.inf)
    alive=m>-kn
    surv=np.where(alive,np.maximum(-np.expm1(np.minimum(lr,0)),0),0)
    ST=S*np.exp(m*sd)
    pay={"call":np.maximum(ST-K,0),"put":np.maximum(K-ST,0),"fwd":ST-K}[opt]
    return math.exp(-r*T)*float(np.sum(np.exp(logw)*surv*pay)), Delta

def run(S,K,T,v,r,H):
    c=hp.consts(S,K,T,v,r,H)                      # mp: do_a (A1), do_b (A2)
    A1=float(c["do_a"]); A2=float(c["do_b"])
    pr=hp.prices(S,K,T,v,r,H)
    Fc=float(pr["do_b"]); Cc=float(pr["do_a"]); Pc=Cc-Fc   # forward = B-D (formal), put = call - forward
    ns=np.arange(2000,12001,37)
    ry=[];rf=[];rp=[];dl=[]
    for n in ns:
        F,Dl=lattice(S,K,T,v,r,n,H,"fwd"); P,_=lattice(S,K,T,v,r,n,H,"put")
        dl.append(Dl); rf.append(math.sqrt(n)*(F-Fc)-A2*Dl); rp.append(math.sqrt(n)*(P-Pc)-(A1-A2)*Dl)
    rf=np.array(rf); rp=np.array(rp); ns=np.array(ns,float)
    print(f"K={K} H={H} (K>H):  A1={A1:.4f} A2={A2:.4f}  F^cont={Fc:.6f}  P^cont={Pc:.6f}")
    print(f"   forward: max|sqrt(n)(F_n-F)-A2*Delta| = {np.abs(rf).max():.4f}   (should be O(1/sqrt n): times sqrt(n) -> {np.max(np.abs(rf)*np.sqrt(ns)):.3f})")
    print(f"   put    : max|sqrt(n)(P_n-P)-(A1-A2)*Delta| = {np.abs(rp).max():.4f}  (times sqrt(n) -> {np.max(np.abs(rp)*np.sqrt(ns)):.3f})")
    # growth check across sub-ranges
    for lo,hi in ((2000,4000),(4000,8000),(8000,12001)):
        mk=(ns>=lo)&(ns<hi); print(f"      n in [{lo},{hi}): max|resid_put|={np.abs(rp[mk]).max():.4f}  max|resid_fwd|={np.abs(rf[mk]).max():.4f}")
run(130,140,36/365,.45,.04,100)
run(130,140,36/365,.45,.04,120)
run(100,105,1.0,.25,.10,95)
