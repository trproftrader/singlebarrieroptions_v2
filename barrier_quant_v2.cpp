// barrier_quant_v2.cpp
// ------------------------------------------------------------------
// Binomial (CRR) summation formulas for knock-out barrier options,
// compared with QuantLib's AnalyticBarrierEngine (continuous monitoring).
//
// Changes w.r.t. barrier_quant_revised.cpp
//  1. MATURITY: lattice and QuantLib now use the SAME time to maturity.
//     The maturity is an integer number of days, T = days/365 (Actual365Fixed).
//     (Before: QuantLib got int(0.1*365) = 36 days = 0.09863 y, the lattice got
//      T = 0.1, so the two sides priced different options.)
//  2. The four duplicated functions are one function: lattice_price().
//  3. Prices are returned (not printed), so they can be tabulated.
//  4. Extra column: QuantLib analytic price with the barrier moved to the
//     lattice's effective barrier Ht = H~_n = S*u^{+-k_n}. This is the barrier the
//     lattice really enforces; error vs this column decays ~ O(1/n) without
//     the Boyle-Lau oscillation.
//  5. Options already knocked out at t=0 (S >= Hu or S <= Hd) return 0.
//  6. ext::shared_ptr / ext::make_shared: works with QuantLib built with
//     boost or std shared_ptr.
//
// Compile:
//   g++ barrier_quant_v2.cpp -o barrier_quant_v2 -std=c++17 \
//       -I/opt/homebrew/include -L/opt/homebrew/lib -lQuantLib -lgmp -lm
// Run:   ./barrier_quant_v2 <n_max> [days=36] [step=1000] [Hu=170] [Hd=100]
//   e.g. ./barrier_quant_v2 10000          (T = 36/365, Scenario 1)
//        ./barrier_quant_v2 10000 36 1000 155 120   (Scenario 2)
//        ./barrier_quant_v2 10000 73       (T = 73/365 = 0.2)
// ------------------------------------------------------------------
#include <gmp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <ql/quantlib.hpp>

using namespace QuantLib;

enum Kind { UP, DOWN };

struct Market { double S, K, r, v, q; };

// Barrier index k_n >= 1: number of lattice steps from S to the first node AT or beyond the barrier
// (paper, Section 2: k_n = ceil(log(Hu/S)/log u); down: ceil(log(S/Hd)/log u)); a node exactly on the barrier kills.
// A ratio within 1e-12 of an integer is treated as that integer.
static int barrier_index(double S, double Barrier, double u, Kind kind) {
    double a = (kind == UP) ? std::log(Barrier / S) / std::log(u)
                            : std::log(S / Barrier) / std::log(u);
    const double k = std::round(a);
    if (std::fabs(a - k) <= 1e-12 * std::max(1.0, std::fabs(a))) return (int)k;
    return (int)std::ceil(a);
}

// Effective barrier enforced by the lattice (first killing node).
static double effective_barrier(double S, double u, int kn, Kind kind) {   // Ht = H~_n
    return (kind == UP) ? S * std::pow(u, kn) : S * std::pow(u, -kn);
}

// Closed-form lattice price:  e^{-rT} sum_j V(n,j) p^j q^{n-j} payoff
//   up  : V = <<C(n,j) - C(n,j-k_n)>>
//   down: V = <<C(n,j) - C(n,j+k_n)>>
static double lattice_price(int n, Kind kind, bool isCall, double Barrier,
                            const Market& m, double T, int* kn_out = nullptr,
                            double* Ht_out = nullptr) {
    const double S = m.S, K = m.K, r = m.r, v = m.v;
    const double u = std::exp(v * std::sqrt(T / n)), d = 1.0 / u;
    const double p = (std::exp(r * T / n) - d) / (u - d);
    const double q = 1.0 - p;
    const double lu = std::log(u);

    if ((kind == UP && S >= Barrier) || (kind == DOWN && S <= Barrier)) {
        if (kn_out) *kn_out = 0;
        if (Ht_out) *Ht_out = Barrier;
        return 0.0;                          // knocked out at inception
    }
    const int kn = barrier_index(S, Barrier, u, kind);
    if (kn_out) *kn_out = kn;
    if (Ht_out) *Ht_out = effective_barrier(S, u, kn, kind);

    mpf_set_default_prec(256);
    mpz_t b1, b2; mpz_init(b1); mpz_init(b2);
    mpf_t pj, qj, w, tmp; mpf_init(pj); mpf_init(qj); mpf_init(w); mpf_init(tmp);
    mpf_t pp, qq; mpf_init(pp); mpf_init(qq);
    mpf_set_d(pp, p); mpf_set_d(qq, q);

    double sum = 0.0;
    for (int j = 0; j <= n; ++j) {
        const double ST = S * std::exp((2.0 * j - n) * lu);
        const double payoff = isCall ? ST - K : K - ST;
        if (payoff <= 0.0) continue;

        mpz_bin_uiui(b1, n, j);
        long k = (kind == UP) ? (long)j - kn : (long)j + kn;   // reflected index
        if (k >= 0 && k <= n) {
            mpz_bin_uiui(b2, n, (unsigned long)k);
            if (mpz_cmp(b1, b2) > 0) mpz_sub(b1, b1, b2);
            else continue;                                   // <<.>> = 0
        }
        mpf_pow_ui(pj, pp, j);
        mpf_pow_ui(qj, qq, n - j);
        mpf_set_z(w, b1);
        mpf_mul(w, w, pj);
        mpf_mul(w, w, qj);
        sum += mpf_get_d(w) * payoff;
    }
    mpz_clear(b1); mpz_clear(b2);
    mpf_clear(pj); mpf_clear(qj); mpf_clear(w); mpf_clear(tmp); mpf_clear(pp); mpf_clear(qq);
    return std::exp(-r * T) * sum;
}

// QuantLib continuous-monitoring analytic price, maturity = today + days.
static double ql_price(Kind kind, bool isCall, double Barrier, const Market& m, int days) {
    Calendar cal = TARGET();
    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;
    Handle<Quote> spot(ext::make_shared<SimpleQuote>(m.S));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, m.r, Actual365Fixed()));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, m.q, Actual365Fixed()));
    Handle<BlackVolTermStructure> vTS(ext::make_shared<BlackConstantVol>(today, cal, m.v, Actual365Fixed()));
    auto process = ext::make_shared<GeneralizedBlackScholesProcess>(spot, qTS, rTS, vTS);
    ext::shared_ptr<StrikedTypePayoff> pay(
        ext::make_shared<PlainVanillaPayoff>(isCall ? Option::Call : Option::Put, m.K));
    ext::shared_ptr<Exercise> ex(ext::make_shared<EuropeanExercise>(today + Period(days, Days)));
    BarrierOption opt(kind == UP ? Barrier::UpOut : Barrier::DownOut, Barrier, 0.0, pay, ex);
    opt.setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(process));
    return opt.NPV();
}

int main(int argc, char* argv[]) {
    if (argc <= 1) { std::printf("Usage: %s <n_max> [days=36] [step=1000] [Hu=170] [Hd=100]\n", argv[0]); return 2; }
    const int nmax  = std::atoi(argv[1]);
    const int days  = (argc > 2) ? std::atoi(argv[2]) : 36;
    const int step  = (argc > 3) ? std::atoi(argv[3]) : 1000;
    const double T  = days / 365.0;                 // SAME T for lattice and QuantLib

    Market m{130.0, 140.0, 0.04, 0.45, 0.0};
    const double Bup   = (argc > 4) ? std::atof(argv[4]) : 170.0;
    const double Bdown = (argc > 5) ? std::atof(argv[5]) : 100.0;

    std::cout << "----------- Parameters -----------\n"
              << "S=" << m.S << " K=" << m.K << " Hu=" << Bup << " Hd=" << Bdown
              << " r=" << m.r << " sigma=" << m.v
              << " days=" << days << " T=days/365=" << std::setprecision(8) << T << "\n";

    struct Row { const char* name; Kind kind; bool call; double B; };
    const Row rows[] = { {"Cupout",   UP,   true,  Bup},   {"Pupout",   UP,   false, Bup},
                         {"Cdownout", DOWN, true,  Bdown}, {"Pdownout", DOWN, false, Bdown} };

    for (const Row& R : rows) {
        const double qlH = ql_price(R.kind, R.call, R.B, m, days);
        std::cout << "\n== " << R.name << "   QuantLib(H=" << R.B << ") = "
                  << std::fixed << std::setprecision(6) << qlH << "\n"
                  << "      n   k_n       Ht    lattice   lat-QL(H)    QL(Ht)   lat-QL(Ht)\n";
        for (int n = step; n <= nmax; n += step) {
            int kn; double Ht;
            const double lat = lattice_price(n, R.kind, R.call, R.B, m, T, &kn, &Ht);
            const double qlE = ql_price(R.kind, R.call, Ht, m, days);
            std::printf("%7d %4d %9.4f %10.6f %+11.6f %11.6f %+13.6f\n",
                        n, kn, Ht, lat, lat - qlH, qlE, lat - qlE);
        }
    }
    return 0;
}
