import numpy as np, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import p2_common as C
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "figs"); REP = os.path.join(HERE, "..", "report")
K = 3000; TOLS = (1e-3, 1e-6, 1e-9)
best = json.load(open(os.path.join(HERE, "best_params.json")))
P = C.make_problem(2026); h0 = P.hash

Fls, Fst = C.F(P, P.theta_ls), C.F(P, P.theta_star)
dls = np.linalg.norm(P.theta_ls - P.theta_star)
Atot = P.A.reshape(-1, C.D); pred = C.SIGMA * np.sqrt(np.trace(np.linalg.inv(Atot.T @ Atot)))
print("lambda2(W)=%.4f L=%.4f mu=%.4f | ||theta_LS-theta*||=%.5f predicted %.5f | F(theta_LS)=%.6f F(theta*)=%.6f | sigma^2/2=%.5f" % (P.lam2, P.L, P.mu, dls, pred, Fls, Fst, C.SIGMA ** 2 / 2))

runs = {f"DGD ($\\alpha$={best['DGD']:g}, tuned)": C.run_dgd(P, best['DGD'], K),
        "DGD ($\\alpha$=0.2)": C.run_dgd(P, 0.2, K),
        f"GT ($\\alpha$={best['GT']:g}, tuned)": C.run_gt(P, best['GT'], K),
        f"ADMM ($\\rho$={best['ADMM']:g}, tuned)": C.run_admm(P, best['ADMM'], K, check=True)}
assert P.hash == h0, "problem data changed during runs"
sty = list(zip(["tab:blue", "tab:cyan", "tab:red", "tab:green"], ["-", "--", "-", "-"]))

fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
panels = [(0, r"Consensus error $E_{con}[k]$", None), (1, r"$E_{true}[k]=\|\bar\theta[k]-\theta^\star\|_2$", dls),
          (2, r"Global objective $F(\bar\theta[k])$", Fls), (3, r"$\|\bar\theta[k]-\theta_{LS}\|_2$ (extra)", None)]
for a, (col, title, ref) in zip(ax.ravel(), panels):
    for (lab, h), (c, ls) in zip(runs.items(), sty): a.semilogy(np.maximum(h[:, col], 1e-17), label=lab, color=c, ls=ls)
    if ref is not None: a.axhline(ref, color='k', ls=':', lw=1, label=(r"$\|\theta_{LS}-\theta^\star\|$" if col == 1 else r"$F^\ast=F(\theta_{LS})$"))
    a.set_title(title, fontsize=10); a.set_xlabel("iteration $k$"); a.grid(alpha=.3, which="both")
ax[0, 0].legend(fontsize=6.5); ax[1, 0].set_ylim(Fls * 0.99, Fls * 1.05) if False else None
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_single.pdf")); plt.close()

plt.figure(figsize=(4.8, 3.3))
for (lab, h), (c, ls) in zip(runs.items(), sty): plt.semilogy(np.maximum(h[:, 2] - Fls, 1e-16), label=lab, color=c, ls=ls)
plt.xlabel("iteration $k$"); plt.ylabel(r"$F(\bar\theta[k])-F^\ast$ (clipped at $10^{-14}$: rounding level of $F$)"); plt.grid(alpha=.3, which="both"); plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_single_gap.pdf")); plt.close()

def it(h, t):
    i = C.first_below(h[:, 3], t); return str(i) if i is not None else "never"
L = [r"\begin{tabular}{lccccccc}\toprule Algorithm & it.\ to $10^{-3}$ & it.\ to $10^{-6}$ & it.\ to $10^{-9}$ & final $E_{con}$ & final $E_{true}$ & final $F-F^\ast$ & final $\|\bar\theta-\theta_{LS}\|$\\\midrule"]
for lab, h in runs.items():
    L.append(f"{lab} & {it(h,1e-3)} & {it(h,1e-6)} & {it(h,1e-9)} & {h[-1,0]:.1e} & {h[-1,1]:.5f} & {('$<10^{-14}$' if h[-1,2]-Fls<1e-14 else format(h[-1,2]-Fls,'.1e'))} & {h[-1,3]:.1e}\\\\")
L.append(r"\bottomrule\end{tabular}")
open(os.path.join(REP, "gen_p2_single_table.tex"), "w").write("\n".join(L))

m = {"pTwoLam": f"{P.lam2:.3f}", "pTwoL": f"{P.L:.2f}", "pTwoMu": f"{P.mu:.3f}", "pTwoDls": f"{dls:.5f}", "pTwoPred": f"{pred:.5f}",
     "pTwoFls": f"{Fls:.6f}", "pTwoFst": f"{Fst:.6f}", "pTwoSigHalf": f"{C.SIGMA**2/2:.5f}", "pTwoGap": f"{Fst-Fls:.1e}"}
open(os.path.join(REP, "gen_p2_single_numbers.tex"), "w").write("\n".join(r"\newcommand{\%s}{%s}" % kv for kv in m.items()))
for lab, h in runs.items(): print(lab, it(h,1e-3), it(h,1e-6), it(h,1e-9), "econ %.1e Etrue %.5f dist %.1e" % (h[-1,0], h[-1,1], h[-1,3]))
print("asserts passed: connected graph, W symmetric/doubly stochastic/pattern (4), identical data for all runs, ADMM closed form = local argmin")
