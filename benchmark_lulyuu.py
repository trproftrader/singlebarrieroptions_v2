"""Benchmark of Lu & Lyuu (2023, Math. Comput. Simul. 205, Fig. 11): DOC with S=95, K=100, H=90, r=10%, sigma=25%, T=1.
Their continuous (Reiner-Rubinstein/Merton) price is 5.9968; their BTT error is reported as <= 0.001 for n=201,203,...,1401."""
import math, numpy as np
from convergence_analysis import rr_price
from interp_test import lattice_level
S,K,T,v,r,H=95,100,1.0,.25,.10,90
cont=rr_price(S,K,T,v,r,H,"down","call"); print("continuous price: %.4f" % cont)
raw=[];eff=[];itp=[]
for n in range(201,1402,2):
    sd=v*math.sqrt(T/n); a=math.log(S/H)/sd; kn=math.ceil(a); k0=math.floor(a); w=a-k0
    P=lattice_level(S,K,T,v,r,n,kn,"down","call")
    raw.append(P-cont); eff.append(P-rr_price(S,K,T,v,r,S*math.exp(-kn*sd),"down","call"))
    itp.append((1-w)*lattice_level(S,K,T,v,r,n,k0,"down","call")+w*lattice_level(S,K,T,v,r,n,k0+1,"down","call")-cont)
for lab,e in (("plain CRR vs P(H)",raw),("plain CRR vs P(Ht)",eff),("two-layer interpolation vs P(H)",itp)):
    print("%-34s max|err| = %.5f" % (lab, np.max(np.abs(e))))
