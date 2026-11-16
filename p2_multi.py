import numpy as np, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import p2_common as C
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "figs"); REP = os.path.join(HERE, "..", "report")
K = 3000; SEEDS = list(range(10)); TOLS = (1e-3, 1e-6, 1e-9)
GRIDS = {"DGD": [0.02, 0.05, 0.1, 0.2, 0.4], "GT": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], "ADMM": [0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 10.0]}
RUN = {"DGD": lambda P, p: C.run_dgd(P, p, K), "GT": lambda P, p: C.run_gt(P, p, K), "ADMM": lambda P, p: C.run_admm(P, p, K)}
PARAM = {"DGD": r"$\alpha$", "GT": r"$\alpha$", "ADMM": r"$\rho$"}
E2 = 2 * 40 * C.D
PERIT = {"DGD": 1 * E2, "GT": 2 * E2, "ADMM": 1 * E2}

problems = [C.make_problem(s) for s in SEEDS]

h0 = [P.hash for P in problems]

def med_iters(curves, tol):
    it = np.array([np.inf if C.first_below(c[:, 3], tol) is None else C.first_below(c[:, 3], tol) for c in curves], float)
    reached = int(np.isfinite(it).sum()); return np.median(it), reached

def fmtv(x): return 'diverged' if not np.isfinite(x) else f'{x:.1e}'

def fmt(med, reached, n=len(SEEDS)):
    return (f"{int(med)}" if np.isfinite(med) else "never") + (f" ({reached}/{n})" if reached < n else "")

tuning = {}; curves_all = {}
for alg, grid in GRIDS.items():
    tuning[alg] = []
    for p in grid:
        cs = [RUN[alg](P, p) for P in problems]; curves_all[(alg, p)] = cs
        row = {"p": p, "final_dist": float(np.median([c[-1, 3] for c in cs])), "final_econ": float(np.median([c[-1, 0] for c in cs]))}
        for t in TOLS: row[t] = med_iters(cs, t)
        tuning[alg].append(row)
assert [P.hash for P in problems] == h0, "data/graph/initial conditions changed during the experiment"

best = {}
for alg, rows in tuning.items():
    reach = [r for r in rows if np.isfinite(r[1e-6][0])]
    best[alg] = (min(reach, key=lambda r: r[1e-6][0]) if reach else min(rows, key=lambda r: r["final_dist"]))["p"]
json.dump(best, open(os.path.join(HERE, "best_params.json"), "w"))

L = [r"\begin{tabular}{llcccc}\toprule Alg. & param & iters to $10^{-3}$ & iters to $10^{-6}$ & final $\|\bar\theta-\theta_{LS}\|$ & final $E_{con}$\\\midrule"]
for alg, rows in tuning.items():
    for r in rows:
        star = r"$^\ast$" if r["p"] == best[alg] else ""
        L.append(f"{alg} & {PARAM[alg]}={r['p']:g}{star} & {fmt(*r[1e-3])} & {fmt(*r[1e-6])} & {fmtv(r['final_dist'])} & {fmtv(r['final_econ'])}\\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}"
open(os.path.join(REP, "gen_p2_tuning.tex"), "w").write("\n".join(L))

L = [r"\begin{tabular}{lcccccccc}\toprule Algorithm (best param.) & iters to $10^{-3}$ & iters to $10^{-6}$ & iters to $10^{-9}$ & final $E_{con}$ & final $\|\bar\theta-\theta_{LS}\|$ & scalars / iter. & scalars to $10^{-9}$\\\midrule"]
stats = {}
for alg in GRIDS:
    cs = curves_all[(alg, best[alg])]; row = {t: med_iters(cs, t) for t in TOLS}
    fe, fd = float(np.median([c[-1, 0] for c in cs])), float(np.median([c[-1, 3] for c in cs]))
    cost = row[1e-9][0] * PERIT[alg] if np.isfinite(row[1e-9][0]) else None
    stats[alg] = dict(it=row, econ=fe, dist=fd, cost=cost)
    L.append(f"{alg} ({PARAM[alg]}={best[alg]:g}) & {fmt(*row[1e-3])} & {fmt(*row[1e-6])} & {fmt(*row[1e-9])} & {fe:.1e} & {fd:.1e} & {PERIT[alg]} & " + (f"{cost:.2e}" if cost else "not reached") + r"\\")
L.append(r"\bottomrule\end{tabular}")
open(os.path.join(REP, "gen_p2_table.tex"), "w").write("\n".join(L))

mac = []
def add(name, val): mac.append(r"\newcommand{\%s}{%s}" % (name, val))
add("pTwoBestDGD", f"{best['DGD']:g}"); add("pTwoBestGT", f"{best['GT']:g}"); add("pTwoBestADMM", f"{best['ADMM']:g}")
add("pTwoPerIter", str(E2)); add("pTwoK", str(K))
for alg in GRIDS:
    for t, nm in zip(TOLS, ("a", "b", "c")):
        m_, r_ = stats[alg]["it"][t]; add(f"pTwoIt{alg}{nm}", (f"{int(m_)}" if np.isfinite(m_) else "never") + (f" ({r_}/10 reach)" if r_ < 10 else ""))
if stats["ADMM"]["cost"] and stats["GT"]["cost"]:
    add("pTwoTrafficRatio", f"{stats['GT']['cost'] / stats['ADMM']['cost']:.1f}")
    add("pTwoIterRatio", f"{stats['GT']['it'][1e-9][0] / stats['ADMM']['it'][1e-9][0]:.1f}")
    add("pTwoAdmmCost", f"{stats['ADMM']['cost']:.2e}"); add("pTwoGtCost", f"{stats['GT']['cost']:.2e}")
    add("pTwoAdmmItNine", f"{int(stats['ADMM']['it'][1e-9][0])}"); add("pTwoGtItNine", f"{int(stats['GT']['it'][1e-9][0])}")

gt_ok = [r["p"] for r in tuning["GT"] if np.isfinite(r["final_dist"]) and r["final_dist"] < 1e-3]
gt_bad = [r["p"] for r in tuning["GT"] if not (np.isfinite(r["final_dist"]) and r["final_dist"] < 1e-3)]
r1=[r for r in tuning["ADMM"] if r["p"]==1.0][0][1e-6][0]; rb=[r for r in tuning["ADMM"] if r["p"]==best["ADMM"]][0][1e-6][0]; add("pTwoRhoOneRatio", f"{r1/rb:.1f}")
add("pTwoGTok", f"{max(gt_ok):g}"); add("pTwoGTbad", f"{min(gt_bad):g}" if gt_bad else "none")
add("pTwoDGDfloorLo", f"{tuning['DGD'][0]['final_dist']:.1e}"); add("pTwoDGDfloorHi", f"{tuning['DGD'][-1]['final_dist']:.1e}")
open(os.path.join(REP, "gen_p2_multi_numbers.tex"), "w").write("\n".join(mac))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for alg in GRIDS:
    arr = np.array([c[:, 3] for c in curves_all[(alg, best[alg])]]); arr = np.where(np.isfinite(arr), arr, np.nan)
    med, lo, hi = np.nanmedian(arr, 0), np.nanpercentile(arr, 10, 0), np.nanpercentile(arr, 90, 0)
    for j, x in enumerate((np.arange(K + 1), np.arange(K + 1) * PERIT[alg] / 1e4)):
        ax[j].semilogy(x, np.maximum(med, 1e-17), label=f"{alg} ({PARAM[alg].strip('$').strip(chr(92))}={best[alg]:g})"); ax[j].fill_between(x, np.maximum(lo, 1e-17), np.maximum(hi, 1e-17), alpha=.2)
for a, xl in zip(ax, ("iteration $k$", r"scalars communicated in the whole network ($\times10^4$)")):
    a.set_xlabel(xl); a.set_ylabel(r"$\|\bar\theta[k]-\theta_{LS}\|_2$"); a.grid(alpha=.3, which="both"); a.set_ylim(1e-16, 10)
ax[0].legend(fontsize=7); ax[1].set_xlim(0, 60); ax[0].set_title("10 instances: median, 10-90% band", fontsize=9); ax[1].set_title("same, versus communication", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_multi.pdf")); plt.close()

print("best:", best)
for alg, rows in tuning.items():
    for r in rows: print(f"{alg:5s} {r['p']:<5g} 1e-3:{fmt(*r[1e-3]):<12s} 1e-6:{fmt(*r[1e-6]):<12s} final dist {r['final_dist']:.1e} econ {r['final_econ']:.1e}")
for alg, s in stats.items(): print(alg, {t: fmt(*s['it'][t]) for t in TOLS}, "econ %.1e dist %.1e cost" % (s['econ'], s['dist']), s['cost'])
