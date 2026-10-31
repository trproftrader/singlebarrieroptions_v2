% matlab_compare.m
% Compare: (a) barrierbybls (Financial Instruments Toolbox, continuous monitoring)
%          (b) CRR closed-form lattice price (reflection principle), same T
%          (c) barrierbybls with the lattice's effective barrier H_eff
% Run:  matlab_compare            (or:  matlab_compare(36) ; days = 36)
% NOT run by the author of this script: if a call errors, paste the message.

function matlab_compare(days)
if nargin < 1, days = 36; end

S = 130; K = 140; r = 0.04; sigma = 0.45;
Hu = 155; Hd = 120;
T = days/365;                       % same T as C++ v2 (Actual/365)

Settle   = datenum(2026,10,31);
Maturity = Settle + days;

% Basis = 3 -> Actual/365.  Compounding = -1 -> continuous.
RateSpec  = intenvset('ValuationDate',Settle,'StartDates',Settle, ...
                      'EndDates',Maturity,'Rates',r,'Compounding',-1,'Basis',3);
StockSpec = stockspec(sigma,S);

%% ---------------- Diagnostic: is the maturity what we think? ------------
fprintf('--- Diagnostic (days = %d, T = %.8f) ---\n', days, T);
vanillaToolbox = optstockbybls(RateSpec,StockSpec,Settle,Maturity,'call',K);
vanillaBLS     = blsprice(S,K,r,T,sigma);
fprintf('Vanilla call: toolbox = %.6f | blsprice(T=days/365) = %.6f | diff = %+.2e\n', ...
        vanillaToolbox, vanillaBLS, vanillaToolbox - vanillaBLS);
fprintf('(diff ~ 0 means the toolbox uses T = days/365. If not, check Basis / dates.)\n\n');

%% ---------------- Table -------------------------------------------------
cases = { 'Cupout',   'call', 'uo', Hu, 'u';
          'Pupout',   'put',  'uo', Hu, 'u';
          'Cdownout', 'call', 'do', Hd, 'd';
          'Pdownout', 'put',  'do', Hd, 'd' };

ns = [1000 2000 4000 8000 10000 16000];

for c = 1:size(cases,1)
    nm = cases{c,1}; opt = cases{c,2}; bs = cases{c,3}; B = cases{c,4}; kd = cases{c,5};
    ml = barrierbybls(RateSpec,StockSpec,opt,K,Settle,Maturity,bs,B);
    fprintf('== %s   barrierbybls(H=%g) = %.6f\n', nm, B, ml);
    fprintf('%8s %5s %9s %10s %12s %12s %14s\n', ...
            'n','h','H_eff','lattice','lat-ML(H)','ML(H_eff)','lat-ML(H_eff)');
    for n = ns
        [lat,h,Heff] = crr_knockout(n,kd,strcmp(opt,'call'),B,S,K,r,sigma,T);
        mlE = barrierbybls(RateSpec,StockSpec,opt,K,Settle,Maturity,bs,Heff);
        fprintf('%8d %5d %9.4f %10.6f %+12.6f %12.6f %+14.6f\n', ...
                n,h,Heff,lat,lat-ml,mlE,lat-mlE);
    end
    fprintf('\n');
end
end

%% ---------------- CRR closed-form knock-out price (log-space) -------------
function [P,h,Heff] = crr_knockout(n,kind,isCall,B,S,K,r,v,T)
dt = T/n; u = exp(v*sqrt(dt)); d = 1/u;
p  = (exp(r*dt)-d)/(u-d);
if kind == 'u'
    h = floor(log(B/S)/log(u)) + 1;  Heff = S*u^h;
else
    h = floor(log(S/B)/log(u)) + 1;  Heff = S*u^(-h);
end
j  = 0:n;  m = 2*j - n;
logC = gammaln(n+1) - gammaln(j+1) - gammaln(n-j+1);
logw = logC + j*log(p) + (n-j)*log(1-p);

if kind == 'u'
    k = j - h;   alive = (m < h);
else
    k = j + h;   alive = (m > -h);
end
valid = (k >= 0) & (k <= n);
logratio = -inf(size(j));                 % log[ C(n,k)/C(n,j) ]
jj = j(valid); kk = k(valid);
logratio(valid) = gammaln(jj+1) + gammaln(n-jj+1) - gammaln(kk+1) - gammaln(n-kk+1);
ratio = exp(min(logratio,0));
surv  = zeros(size(j));
surv(alive) = max(1 - ratio(alive), 0);

ST  = S*exp(m*log(u));
if isCall, pay = max(ST-K,0); else, pay = max(K-ST,0); end
P = exp(-r*T) * sum( exp(logw) .* surv .* pay );
end
