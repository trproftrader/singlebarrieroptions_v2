import math, numpy as np
from scipy.special import gammaln
from convergence_analysis import rr_price

def lattice_level(S,K,T,v,r,n,kl,kind,opt):
    """Closed-form CRR knock-out price with kill level kl (>=1 steps): up kills at X>=kl, down kills at X<=-kl.
    With kl = k_n (paper, Section 2) this is the price at the effective barrier H~_n."""
    dt=T/n; u=math.exp(v*math.sqrt(dt)); p=(math.exp(r*dt)-1/u)/(u-1/u)
    j=np.arange(n+1,dtype=float); m=2*j-n
    logw=gammaln(n+1)-gammaln(j+1)-gammaln(n-j+1)+j*math.log(p)+(n-j)*math.log(1-p)
    kk=(j-kl) if kind=="up" else (j+kl)
    valid=(kk>=0)&(kk<=n); kc=np.where(valid,kk,0)
    lr=np.where(valid,gammaln(j+1)+gammaln(n-j+1)-gammaln(kc+1)-gammaln(n-kc+1),-np.inf)
    alive=(m<kl) if kind=="up" else (m>-kl)
    surv=np.where(alive,np.maximum(-np.expm1(np.minimum(lr,0)),0),0)
    ST=S*np.exp(m*math.log(u)); pay=np.maximum(ST-K,0) if opt=="call" else np.maximum(K-ST,0)
    return math.exp(-r*T)*float(np.sum(np.exp(logw)*surv*pay))

def study(name,S,K,T,v,r,H,kind,opt,ns):
    cont=rr_price(S,K,T,v,r,H,kind,opt)
    out=[]
    for n in ns:
        sd=v*math.sqrt(T/n); a=abs(math.log(H/S))/sd; k0=math.floor(a)
        if k0<1: continue
        Ht=lambda kl: S*math.exp(kl*sd) if kind=="up" else S*math.exp(-kl*sd)
        # raw: kill at k_n = ceil(a_n), the effective barrier H~_n
        kn=math.ceil(a-1e-12); raw=lattice_level(S,K,T,v,r,n,kn,kind,opt)
        # error vs continuous at Ht = H~_n
        e_eff=raw-rr_price(S,K,T,v,r,Ht(kn),kind,opt)
        # interpolate (log-linear in barrier) between layers k0 and k0+1
        w=a-k0
        interp=(1-w)*lattice_level(S,K,T,v,r,n,k0,kind,opt)+w*lattice_level(S,K,T,v,r,n,k0+1,kind,opt)
        out.append((n,raw-cont,e_eff,interp-cont))
    o=np.array(out)
    print(f"\n{name}   continuous = {cont:.6f}")
    print(f"  max|err| over n in [{ns[0]},{ns[-1]}]:  raw vs H = {np.abs(o[:,1]).max():.2e}   raw vs Ht = {np.abs(o[:,2]).max():.2e}   interpolated vs H = {np.abs(o[:,3]).max():.2e}")
    # scaling: n * max error in last third
    t=o[len(o)*2//3:]
    print(f"  n*|err| (last third, max):  raw vs H = {np.max(t[:,0]*np.abs(t[:,1])):.3f}   vs Ht = {np.max(t[:,0]*np.abs(t[:,2])):.3f}   interp = {np.max(t[:,0]*np.abs(t[:,3])):.3f}")
    # fitted decay exponent of envelope
    blocks=np.array_split(np.arange(len(o)),10)
    x=np.log([o[b,0].mean() for b in blocks])
    for lab,c in (("raw vs H",1),("vs Ht",2),("interp",3)):
        y=np.log([np.abs(o[b,c]).max() for b in blocks]); print(f"     decay exponent {lab:9}: {-np.polyfit(x,y,1)[0]:.2f}")

ns=np.arange(200,4001,13)
S,K,T,v,r=130,140,36/365,.45,.04
study("Cupout  Hu=170",S,K,T,v,r,170,"up","call",ns)
study("Pupout  Hu=155",S,K,T,v,r,155,"up","put",ns)
study("Cdownout Hd=120",S,K,T,v,r,120,"down","call",ns)
study("Pdownout Hd=100",S,K,T,v,r,100,"down","put",ns)
study("Pdownout Hd=120",S,K,T,v,r,120,"down","put",ns)
study("LP13: do call K=100 H=95",100,100,1,.25,.10,95,"down","call",ns)
