# High-precision check of  A_i = -sigma sqrt(T) dP/dlogH  with mpmath (30 digits), random parameters.
from mpmath import mp, mpf, sqrt, log, exp, ncdf, pi, diff
import random
mp.dps = 30
def prices(S0,K,T,s,r,H):
    sT=s*sqrt(T); mu=(r-s*s/2)/s**2   # b=r
    x1=log(S0/K)/sT+(1+mu)*sT; x2=log(S0/H)/sT+(1+mu)*sT
    y1=log(H*H/(S0*K))/sT+(1+mu)*sT; y2=log(H/S0)/sT+(1+mu)*sT
    df=exp(-r*T); hs=H/S0
    A=S0*ncdf(x1)-K*df*ncdf(x1-sT); B=S0*ncdf(x2)-K*df*ncdf(x2-sT)
    C=S0*hs**(2*(mu+1))*ncdf(y1)-K*df*hs**(2*mu)*ncdf(y1-sT)       # eta=+1 (down)
    D=S0*hs**(2*(mu+1))*ncdf(y2)-K*df*hs**(2*mu)*ncdf(y2-sT)
    Cu=S0*hs**(2*(mu+1))*ncdf(-y1)-K*df*hs**(2*mu)*ncdf(-y1+sT)    # eta=-1 (up)
    Du=S0*hs**(2*(mu+1))*ncdf(-y2)-K*df*hs**(2*mu)*ncdf(-y2+sT)
    return dict(do_a=A-C, do_b=B-D, uo=A-B+Cu-Du)
def consts(S0,K,T,s,r,H):
    sT=s*sqrt(T); R=sqrt(T)
    d11=(log(S0/K)+(r+s*s/2)*T)/sT; d21=(log(H*H/(S0*K))+(r+s*s/2)*T)/sT
    d31=(log(S0/H)+(r+s*s/2)*T)/sT; d41=(log(H/S0)+(r+s*s/2)*T)/sT
    d22=d21-sT; d42=d41-sT
    al=(r-s*s/2)/(2*s); alh=al+s/2
    def h1(x,y): return (S0/H)**(-2*r/s**2)*(H*alh*ncdf(x)-S0*K*exp(-r*T)/H*al*ncdf(y))
    ph=S0/sqrt(2*pi)*exp(-d31**2/2)
    return dict(do_a=4*R*h1(d21,d22), do_b=4*R*h1(d41,d42)+2*ph*(1-K/H),
                uo=4*R*(h1(-d41,-d42)-h1(-d21,-d22))-2*ph*(1-K/H))
if __name__=='__main__':
    random.seed(1); worst=mpf(0)
    for _ in range(200):
        s=mpf(random.uniform(.1,.6)); r=mpf(random.uniform(0,.1)); T=mpf(random.uniform(.05,2)); S0=mpf(100)
        for case in ("do_a","do_b","uo"):
            if case=="do_a": K=S0*mpf(random.uniform(1.0,1.3)); H=S0*mpf(random.uniform(.6,.98)); H=min(H,K*mpf(.999))
            if case=="do_b": H=S0*mpf(random.uniform(.6,.98)); K=H*mpf(random.uniform(.7,.999))
            if case=="uo":   H=S0*mpf(random.uniform(1.02,1.5)); K=min(S0*mpf(random.uniform(.8,1.3)),H*mpf(.999))
            f=lambda lh: prices(S0,K,T,s,r,exp(lh))[case]
            dP=diff(f,log(H))
            A=consts(S0,K,T,s,r,H)[case]
            worst=max(worst,abs(A+s*sqrt(T)*dP))
    print("max |A + sigma sqrt(T) dP/dlogH| over 600 random cases (30 digits):", mp.nstr(worst,5))
