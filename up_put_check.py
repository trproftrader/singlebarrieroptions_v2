import math, numpy as np
from scipy.special import gammaln
from scipy.stats import norm
from convergence_analysis import rr_price
Phi=norm.cdf; phi=norm.pdf
def pieces(S,K,T,v,r,H):
    sT=v*math.sqrt(T); k=r+v*v/2
    d21=(math.log(H*H/(S*K))+k*T)/sT; d22=d21-sT
    d31=(math.log(S/H)+k*T)/sT; d32=d31-sT
    d41=(math.log(H/S)+k*T)/sT; d42=d41-sT
    al=(r-v*v/2)/(2*v); alh=al+v/2; R=(S/H)**(-2*r/v**2)
    h1=lambda x,y: R*(H*alh*Phi(x)-S*K*math.exp(-r*T)/H*al*Phi(y))
    Fui=S*Phi(d31)-K*math.exp(-r*T)*Phi(d32)+R*(H*Phi(-d41)-S*K*math.exp(-r*T)/H*Phi(-d42))
    Fuo=(S-K*math.exp(-r*T))-Fui
    c_uo=-(4*math.sqrt(T)*h1(-d41,-d42)-2*S*phi(d31)*(1-K/H))
    A3=4*math.sqrt(T)*(h1(-d41,-d42)-h1(-d21,-d22))-2*S*phi(d31)*(1-K/H)
    return Fuo,c_uo,-A3          # forward, its coefficient, call coefficient
def lattice(S,K,T,v,r,n,Hb,opt):
    dt=T/n; sd=v*math.sqrt(dt); p=(math.exp(r*dt)-math.exp(-sd))/(math.exp(sd)-math.exp(-sd))
    a=math.log(Hb/S)/sd; kn=math.ceil(a); Delta=kn-a
    j=np.arange(n+1,dtype=float); m=2*j-n
    logw=gammaln(n+1)-gammaln(j+1)-gammaln(n-j+1)+j*math.log(p)+(n-j)*math.log(1-p)
    kk=j-kn; valid=(kk>=0)&(kk<=n); kc=np.where(valid,kk,0)
    lr=np.where(valid,gammaln(j+1)+gammaln(n-j+1)-gammaln(kc+1)-gammaln(n-kc+1),-np.inf)
    alive=m<kn; surv=np.where(alive,np.maximum(-np.expm1(np.minimum(lr,0)),0),0)
    ST=S*np.exp(m*sd); pay={"call":np.maximum(ST-K,0),"put":np.maximum(K-ST,0),"fwd":ST-K}[opt]
    return math.exp(-r*T)*float(np.sum(np.exp(logw)*surv*pay)), Delta
def run(S,K,T,v,r,H):
    Fc,cF,cC=pieces(S,K,T,v,r,H)
    Cc=rr_price(S,K,T,v,r,H,"up","call"); Pc=rr_price(S,K,T,v,r,H,"up","put")
    print(f"K={K} H={H}: forward^cont={Fc:.6f}  vs  call-put (Haug) = {Cc-Pc:.6f}   (must agree)")
    ns=np.arange(2000,12001,37); rf=[];rp=[]
    for n in ns:
        F,D=lattice(S,K,T,v,r,n,H,"fwd"); P,_=lattice(S,K,T,v,r,n,H,"put")
        rf.append(math.sqrt(n)*(F-Fc)-cF*D); rp.append(math.sqrt(n)*(P-Pc)-(cC-cF)*D)
    ns=ns.astype(float); rf=np.array(rf); rp=np.array(rp)
    for lo,hi in ((2000,4000),(4000,8000),(8000,12001)):
        mk=(ns>=lo)&(ns<hi); print(f"   n in [{lo},{hi}): max|resid_fwd|={np.abs(rf[mk]).max():.4f}  max|resid_put|={np.abs(rp[mk]).max():.4f}")
if __name__=="__main__":
    run(130,140,36/365,.45,.04,170)
    run(130,140,36/365,.45,.04,155)
    run(100,95,1.0,.25,.10,105)
