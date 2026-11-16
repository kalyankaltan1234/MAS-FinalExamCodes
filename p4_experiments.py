import os, sys, numpy as np
from multiprocessing import Pool
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from p4_covid import POLICIES, run_seed, Imax, Isafe, N, T
from p4_epi import run_unit_tests
from scipy.stats import beta as _beta
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "figs"); REP = os.path.join(HERE, "..", "report")
NSEED = 300; BASE = {p.name: p for p in POLICIES}
NOC, MAIN, TWO, UNIF = POLICIES[0].name, POLICIES[1].name, POLICIES[2].name, POLICIES[3].name
ABL, EXT = POLICIES[13].name, POLICIES[14].name
SA, SB, SC = POLICIES[10].name, POLICIES[11].name, POLICIES[12].name
def f1(x): return "%.1f" % x
def esc(s):
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("^", r"\^{}"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\~{}")):
        s = s.replace(a, b)
    return s
def tshort(n):
    for a, b in (("main: same-day test, screen 2%, L=5", "main (same-day, 2%, L=5)"), ("stronger A: two-day + quarantine pending test", "str. A: two-day + quarantine"), ("stronger B: screen 10%, L=7", "str. B: screen 10%, L=7"),
                 ("stronger C: B + two-day + quarantine", "str. C: B + two-day + quar."), ("ablation: isolate 10 d after first positive only", "ablation: 10 d after 1st positive"), ("extension: test sensitivity 0.9", "extension: sensitivity 0.9"),
                 ("two-day test (exactly day 2)", "two-day (exactly day 2)"), ("two-day test (uniform 0,1,2)", "two-day (uniform 0,1,2)")):
        if n == a: return b
    return n
macros = {}
def mac(name, val): macros[name] = val

if __name__ == "__main__":

    ut = run_unit_tests(verbose=True)
    mac("nUnit", len(ut)); mac("nUnitPass", sum(c for _, c in ut))
    with open(os.path.join(REP, "gen_p4_tests.tex"), "w") as f:
        f.write("\\begin{enumerate}[nosep,leftmargin=*,label=T\\arabic*.]\n")
        for nme, c in ut: f.write("\\item {\\footnotesize %s} \\hfill \\textbf{%s}\n" % (esc(nme), "PASS" if c else "FAIL"))
        f.write("\\end{enumerate}\n")

    with Pool() as pool: out = pool.map(run_seed, [(s, POLICIES) for s in range(NSEED)])
    R = {p.name: [dict(o[1][p.name], seed=o[0]) for o in out] for p in POLICIES}
    def agg(name, seeds=None):
        rs = R[name] if seeds is None else [R[name][s] for s in seeds]
        pk = np.array([r["peak"] for r in rs])
        return dict(n=len(rs), pk=pk.mean(), sd=pk.std(), worst=pk.max(), att=np.mean([r["attack"] for r in rs]),
                    p1=100 * np.mean([r["sat1"] for r in rs]), p2=100 * np.mean([r["sat2"] for r in rs]),
                    n1=sum(r["sat1"] for r in rs), n2=sum(r["sat2"] for r in rs), rho1=np.mean([r["rho1"] for r in rs]),
                    tests=np.mean([r["tests"] for r in rs]), ok3=sum(r["ok3"] for r in rs), tot3=sum(r["tot3"] for r in rs),
                    ok4=sum(r["ok4"] for r in rs), tot4=sum(r["tot4"] for r in rs))

    rowsA = [(NOC, "no control"), (MAIN, "\\textbf{main}: same-day test"), (TWO, "two-day test (exactly day 2)"), (ABL, "\\emph{ablation:} 10 days after first positive only"), (EXT, "\\emph{extension:} test sensitivity 0.9")]
    with open(os.path.join(REP, "gen_p4_tab20.tex"), "w") as f:
        f.write("\\begin{tabular}{lccccc}\\toprule & peak $N_I$ & attack rate & $\\varphi_1$ satisfied & mean $\\rho_1$ & $\\varphi_2$ satisfied\\\\\\midrule\n")
        for nm, lab in rowsA:
            a = agg(nm, range(20))
            f.write("%s & $%s\\pm%s$ & %.3f & %d/20 & $%+.1f$ & %d/20\\\\\n" % (lab, f1(a["pk"]), f1(a["sd"]), a["att"], a["n1"], a["rho1"], a["n2"]))
        f.write("\\bottomrule\\end{tabular}\n")

    with open(os.path.join(REP, "gen_p4_tabdev.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccccccc}\\toprule policy & peak $N_I$ mean$\\pm$sd & worst peak & attack & $\\varphi_1$ \\% & $\\varphi_2$ \\% & mean $\\rho_1$ & tests/run & $\\varphi_3/\\varphi_4$ compl.\\ (\\%)\\\\\\midrule\n")
        for p in POLICIES:
            a = agg(p.name)
            comp = ("%.0f/%.0f" % (100 * a["ok3"] / max(1, a["tot3"]), 100 * a["ok4"] / max(1, a["tot4"]))) if p.control else "--"
            f.write("%s & $%s\\pm%s$ & %d & %.2f & %.0f & %.0f & $%+.1f$ & %.0f & %s\\\\\n" % (esc(tshort(p.name)), f1(a["pk"]), f1(a["sd"]), a["worst"], a["att"], a["p1"], a["p2"], a["rho1"], a["tests"], comp))
            if p.name in (UNIF, "screen 10%", "tracing L=7", SC): f.write("\\midrule\n")
        f.write("\\bottomrule\\end{tabular}\n")

    a = agg(MAIN); n = agg(NOC); t = agg(TWO); u = agg(UNIF)
    mac("pctMainOne", "%.0f" % a["p1"]); mac("pctMainTwo", "%.0f" % a["p2"]); mac("pkMainMean", f1(a["pk"])); mac("pkMainSd", f1(a["sd"])); mac("pkMainWorst", a["worst"])
    mac("pkNoMean", f1(n["pk"])); mac("pkNoSd", f1(n["sd"])); mac("pkNoWorst", n["worst"]); mac("pctNoOne", "%.0f" % n["p1"]); mac("attNo", "%.2f" % n["att"]); mac("attMain", "%.2f" % a["att"])
    mac("pctTwoOne", "%.0f" % t["p1"]); mac("pkTwoMean", f1(t["pk"])); mac("pkTwoWorst", t["worst"]); mac("pctUnifOne", "%.0f" % u["p1"]); mac("pkUnifMean", f1(u["pk"]))
    d = np.array([R[TWO][s]["peak"] - R[MAIN][s]["peak"] for s in range(NSEED)])
    mac("dPeakTwo", "%+.1f" % d.mean()); mac("sePeakTwo", "%.1f" % (d.std(ddof=1) / np.sqrt(NSEED)))
    for tag, nm in (("A", SA), ("B", SB), ("C", SC)):
        s_ = agg(nm); mac("pctStrong" + tag, "%.0f" % s_["p1"]); mac("pkStrong" + tag, f1(s_["pk"])); mac("worstStrong" + tag, s_["worst"]); mac("attStrong" + tag, "%.2f" % s_["att"]); mac("testsStrong" + tag, "%.0f" % s_["tests"])
    mac("testsMain", "%.0f" % a["tests"]); mac("testsNo", "%.0f" % n["tests"])
    for tag, nm in (("Zero", "screen 0%"), ("One", "screen 1%"), ("Five", "screen 5%"), ("Ten", "screen 10%")): mac("pctScr" + tag, "%.0f" % agg(nm)["p1"]); mac("pkScr" + tag, f1(agg(nm)["pk"]))
    for tag, nm in (("Three", "tracing L=3"), ("Seven", "tracing L=7")): mac("pctL" + tag, "%.0f" % agg(nm)["p1"]); mac("pkL" + tag, f1(agg(nm)["pk"]))
    mac("pctAbl", "%.0f" % agg(ABL)["p1"]); mac("pkAbl", f1(agg(ABL)["pk"])); mac("pctExt", "%.0f" % agg(EXT)["p1"]); mac("pkExt", f1(agg(EXT)["pk"]))
    mac("okThreeMain", a["ok3"]); mac("totThreeMain", a["tot3"]); mac("okFourMain", a["ok4"]); mac("totFourMain", a["tot4"])
    mac("okFourTwo", t["ok4"]); mac("totFourTwo", t["tot4"]); mac("okThreeTwo", t["ok3"]); mac("totThreeTwo", t["tot3"])
    allruns = [r for nm in R for r in R[nm]]
    mac("nRuns", len(allruns)); mac("nFact", sum(r["n_fact"] for r in allruns)); mac("nGraph", sum(r["n_graph"] for r in allruns)); mac("nPform", sum(r["n_pform"] for r in allruns))
    allcomp = [r for nm in R if nm not in (NOC, ABL) for r in R[nm]]
    mac("okThreeAll", sum(r["ok3"] for r in allcomp)); mac("totThreeAll", sum(r["tot3"] for r in allcomp))
    ab = agg(ABL); mac("okThreeAbl", ab["ok3"]); mac("totThreeAbl", ab["tot3"]); mac("pctThreeAbl", "%.1f" % (100 * ab["ok3"] / ab["tot3"]))
    mac("okFourAll", sum(r["ok4"] for r in allcomp)); mac("totFourAll", sum(r["tot4"] for r in allcomp))

    M = R[MAIN]; Nn = R[NOC]
    fail = [r for r in M if not r["sat1"]]; ok = [r for r in M if r["sat1"]]
    mac("nFailBig", len(fail)); mac("nFailTwenty", sum(not r["sat1"] for r in M[:20]))
    mean = lambda rs, k: np.mean([r[k] for r in rs]) if rs else float("nan")
    mac("fdFail", f1(mean(fail, "first_detect"))); mac("fdOk", f1(mean(ok, "first_detect")))
    mac("infFail", f1(mean(fail, "inf_at_detect"))); mac("infOk", f1(mean(ok, "inf_at_detect")))
    mac("pdFail", f1(mean(fail, "peak_day"))); mac("pdOk", f1(mean(ok, "peak_day")))
    mac("excFail", f1(np.mean([r["peak"] - Imax for r in fail]))); mac("excWorst", max(r["peak"] for r in fail) - Imax)
    def hid_peak(r): d_ = r["peak_day"]; return r["ts"]["INF"][d_] - r["ts"]["CKI"][d_]
    def e_share(r): d_ = r["peak_day"]; return r["ts"]["NE"][d_] / max(1, r["ts"]["NE"][d_] + r["ts"]["NI"][d_])
    mac("hidFail", f1(np.mean([hid_peak(r) for r in fail]))); mac("hidOk", f1(np.mean([hid_peak(r) for r in ok])))
    mac("eShareFail", "%.0f" % (100 * np.mean([e_share(r) for r in fail]))); mac("eShareOk", "%.0f" % (100 * np.mean([e_share(r) for r in ok])))
    mac("nEarlyFail", sum(1 for r in fail if r["first_detect"] <= 2))
    fd = np.array([r["first_detect"] for r in M]); pk = np.array([r["peak"] for r in M]); pkn = np.array([r["peak"] for r in Nn])
    mac("corrDet", "%.2f" % np.corrcoef(fd, pk)[0, 1]); mac("corrNoMain", "%.2f" % np.corrcoef(pkn, pk)[0, 1])
    mac("fdMax", int(fd.max())); mac("fdMedian", "%.0f" % np.median(fd))
    inf_d_ = np.array([r["inf_at_detect"] for r in M]); mac("corrInfDet", "%.2f" % np.corrcoef(inf_d_, pk)[0, 1])
    late = fd >= 5; mac("nLate", int(late.sum())); mac("nEarly", int((~late).sum()))
    mac("pctFailLate", "%.0f" % (100 * np.mean(pk[late] >= Imax)) if late.any() else "n/a")
    mac("pctFailEarly", "%.0f" % (100 * np.mean(pk[~late] >= Imax)) if (~late).any() else "n/a")
    big = inf_d_ >= 25; mac("nBigDet", int(big.sum())); mac("pctFailBigDet", "%.0f" % (100 * np.mean(pk[big] >= Imax)) if big.any() else "n/a")
    mac("pctFailSmallDet", "%.0f" % (100 * np.mean(pk[~big] >= Imax)) if (~big).any() else "n/a")
    mac("nStrongAFail", sum(not r["sat1"] for r in R[SA])); mac("nStrongCFail", sum(not r["sat1"] for r in R[SC]))
    with open(os.path.join(REP, "gen_p4_fail.tex"), "w") as f:
        f.write("\\begin{tabular}{rccccccc}\\toprule seed & peak $N_I$ (main) & day of peak & first positive on day & $N_E{+}N_I$ then & undetected at peak & peak $N_I$ (no control) & excess over $I_{max}$\\\\\\midrule\n")
        for r in M[:20]:
            if not r["sat1"]: f.write("%d & %d & %d & %d & %d & %d & %d & %d\\\\\n" % (r["seed"], r["peak"], r["peak_day"], r["first_detect"], r["inf_at_detect"], hid_peak(r), Nn[r["seed"]]["peak"], r["peak"] - Imax))
        f.write("\\midrule\n\\multicolumn{8}{l}{\\emph{all %d failing seeds of the 300 development seeds:} mean peak %s, mean first-positive day %s, mean $N_E{+}N_I$ at that day %s}\\\\\n" % (len(fail), f1(mean(fail, "peak")), f1(mean(fail, "first_detect")), f1(mean(fail, "inf_at_detect"))))
        f.write("\\multicolumn{8}{l}{\\emph{all %d passing seeds:} mean peak %s, mean first-positive day %s, mean $N_E{+}N_I$ at that day %s}\\\\\n" % (len(ok), f1(mean(ok, "peak")), f1(mean(ok, "first_detect")), f1(mean(ok, "inf_at_detect"))))
        f.write("\\bottomrule\\end{tabular}\n")

    cands = [p for p in POLICIES if p.control and not p.iso_exposed and p.sens == 1.0 and p.refresh]
    crow = [(p.name, agg(p.name), np.mean([r["iso_days"] for r in R[p.name]])) for p in cands]
    full = [c for c in crow if c[1]["n1"] == NSEED]
    CHOSEN = min(full, key=lambda c: c[1]["tests"])[0] if full else max(crow, key=lambda c: c[1]["n1"])[0]
    mac("chosenName", esc(tshort(CHOSEN))); mac("nDev", NSEED); mac("nCand", len(cands)); mac("nCandFull", len(full))
    with open(os.path.join(REP, "gen_p4_sel.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccc}\\toprule policy (no quarantine of untested agents) & $\\varphi_1$ on dev seeds & worst peak & tests/run & isolation person-days/run\\\\\\midrule\n")
        for nm, a_, iso_ in crow:
            f.write("%s%s & %d/%d & %d & %.0f & %.0f\\\\\n" % ("\\textbf{" if nm == CHOSEN else "", esc(tshort(nm)) + ("} (chosen)" if nm == CHOSEN else ""), a_["n1"], NSEED, a_["worst"], a_["tests"], iso_))
        f.write("\\bottomrule\\end{tabular}\n")

    def cp(k, n, al=0.05):
        lo = 0.0 if k == 0 else _beta.ppf(al / 2, k, n - k + 1); hi = 1.0 if k == n else _beta.ppf(1 - al / 2, k + 1, n - k)
        return 100 * lo, 100 * hi
    FRESH = list(range(1000, 1300)); FN = len(FRESH)
    FNAMES = list(dict.fromkeys([NOC, MAIN, CHOSEN, SB, SC]))
    FPOL = [BASE[nm] for nm in FNAMES]
    with Pool() as pool: outf = pool.map(run_seed, [(sd, FPOL) for sd in FRESH])
    RF = {nm: [o[1][nm] for o in outf] for nm in FNAMES}
    mac("fN", FN); mac("fSeedLo", FRESH[0]); mac("fSeedHi", FRESH[-1])
    tagof = {NOC: "No", MAIN: "Main", SB: "B", SC: "C"}; tagof[CHOSEN] = "Ch"
    def fagg(nm):
        rs = RF[nm]; pk_ = np.array([r["peak"] for r in rs]); at = np.array([r["attack"] for r in rs])
        n1 = sum(r["sat1"] for r in rs); n2 = sum(r["sat2"] for r in rs)
        return dict(pk=pk_.mean(), sd=pk_.std(ddof=1), worst=int(pk_.max()), att=at.mean(), atsd=at.std(ddof=1), n1=n1, n2=n2, cp1=cp(n1, FN), cp2=cp(n2, FN),
                    ok3=sum(r["ok3"] for r in rs), tot3=sum(r["tot3"] for r in rs), ok4=sum(r["ok4"] for r in rs), tot4=sum(r["tot4"] for r in rs),
                    tests=np.mean([r["tests"] for r in rs]), isod=np.mean([r["iso_days"] for r in rs]), isomax=int(max(r["iso_max"] for r in rs)),
                    isomaxmean=np.mean([r["iso_max"] for r in rs]), notif=np.mean([r["notif"] for r in rs]))
    with open(os.path.join(REP, "gen_p4_fresh.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccccccc}\\toprule policy & peak $N_I$ mean$\\pm$sd & worst peak & attack rate & $\\varphi_1$ satisfied (95\\% CP interval) & $\\varphi_2$ satisfied & $\\varphi_3$ compl. & $\\varphi_4$ compl.\\\\\\midrule\n")
        for nm in FNAMES:
            a_ = fagg(nm); t_ = tagof[nm]
            c3 = "%d/%d" % (a_["ok3"], a_["tot3"]) if nm != NOC else "--"; c4 = "%d/%d" % (a_["ok4"], a_["tot4"]) if nm != NOC else "--"
            f.write("%s & $%s\\pm%s$ & %d & $%.3f\\pm%.3f$ & %d/%d = %.1f\\%% [%.1f, %.1f] & %d/%d & %s & %s\\\\\n" % (esc(tshort(nm)), f1(a_["pk"]), f1(a_["sd"]), a_["worst"], a_["att"], a_["atsd"], a_["n1"], FN, 100 * a_["n1"] / FN, a_["cp1"][0], a_["cp1"][1], a_["n2"], FN, c3, c4))
            if nm == MAIN: f.write("\\midrule\n")
            mac("f" + t_ + "Pk", f1(a_["pk"])); mac("f" + t_ + "Sd", f1(a_["sd"])); mac("f" + t_ + "Worst", a_["worst"]); mac("f" + t_ + "Att", "%.3f" % a_["att"])
            mac("f" + t_ + "One", a_["n1"]); mac("f" + t_ + "OnePct", "%.1f" % (100 * a_["n1"] / FN)); mac("f" + t_ + "OneLo", "%.1f" % a_["cp1"][0]); mac("f" + t_ + "OneHi", "%.1f" % a_["cp1"][1])
            mac("f" + t_ + "Two", a_["n2"]); mac("f" + t_ + "Tests", "%.0f" % a_["tests"]); mac("f" + t_ + "TestsPD", "%.3f" % (a_["tests"] / (N * T))); mac("f" + t_ + "IsoD", "%.0f" % a_["isod"])
            mac("f" + t_ + "IsoMean", "%.1f" % (a_["isod"] / T)); mac("f" + t_ + "IsoMax", a_["isomax"]); mac("f" + t_ + "Notif", "%.0f" % a_["notif"])
            mac("f" + t_ + "OkThree", a_["ok3"]); mac("f" + t_ + "TotThree", a_["tot3"]); mac("f" + t_ + "OkFour", a_["ok4"]); mac("f" + t_ + "TotFour", a_["tot4"])
        f.write("\\bottomrule\\end{tabular}\n")
    with open(os.path.join(REP, "gen_p4_cost.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccccc}\\toprule policy & tests per run & tests per person-day & isolation person-days per run & mean simultaneously isolated & max simultaneously isolated & exposure notifications per run\\\\\\midrule\n")
        for nm in FNAMES:
            a_ = fagg(nm)
            f.write("%s & %.0f & %.3f & %.0f & %.1f & %d & %.0f\\\\\n" % (esc(tshort(nm)), a_["tests"], a_["tests"] / (N * T), a_["isod"], a_["isod"] / T, a_["isomax"], a_["notif"]))
            if nm == MAIN: f.write("\\midrule\n")
        f.write("\\bottomrule\\end{tabular}\n")
    dpk = np.array([RF[MAIN][i]["peak"] - RF[CHOSEN][i]["peak"] for i in range(FN)]); mac("fDeltaPk", "%.1f" % dpk.mean()); mac("fDeltaPkSe", "%.1f" % (dpk.std(ddof=1) / np.sqrt(FN)))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bp = ax.boxplot([[r["peak"] for r in RF[nm]] for nm in FNAMES][::-1], orientation="horizontal", tick_labels=[tshort(nm) for nm in FNAMES][::-1], patch_artist=True)
    for b_ in bp["boxes"]: b_.set_facecolor("#cfe3f5")
    ax.axvline(Imax, c="k", ls="--", lw=1); ax.text(Imax + 1, len(FNAMES) + 0.4, r"$I_{max}=%d$" % Imax, fontsize=8)
    ax.set(xlabel=r"peak $\max_kN_I[k]$ over %d fresh seeds (%d--%d)" % (FN, FRESH[0], FRESH[-1])); ax.tick_params(axis="y", labelsize=7); plt.tight_layout(); plt.savefig(os.path.join(FIG, "p4_fresh.pdf")); plt.close()

    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3})
    def band(name, key):
        A_ = np.array([r["ts"][key] for r in R[name]], float); return A_.mean(0), np.percentile(A_, 10, 0), np.percentile(A_, 90, 0)
    t_ = np.arange(T)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.2))
    for nm, c, lab in ((NOC, "tab:gray", "no control"), (MAIN, "tab:blue", "main policy")):
        NEI = np.array([r["ts"]["NE"] + r["ts"]["NI"] for r in R[nm]], float); NI = np.array([r["ts"]["NI"] for r in R[nm]], float)
        ax[0, 0].plot(t_, NEI.mean(0), c=c, ls="--", label=r"true infections $N_E+N_I$, " + lab); ax[0, 0].plot(t_, NI.mean(0), c=c, label=r"symptomatic $N_I$, " + lab)
        ax[0, 0].fill_between(t_, np.percentile(NI, 10, 0), np.percentile(NI, 90, 0), color=c, alpha=.15)
    ax[0, 0].axhline(Imax, c="k", lw=.8, ls=":"); ax[0, 0].text(60, Imax + 3, r"$I_{max}$", fontsize=8)
    ax[0, 0].set(xlabel="day $k$", ylabel="individuals (mean over %d seeds)" % NSEED, title="(a) true infections: control vs no control"); ax[0, 0].legend(fontsize=6)
    for key, lab in (("CKI", r"known infected $C_{KI}$"), ("CKE", r"known exposed $C_{KE}$"), ("CKNI", r"known not infected $C_{K\neg I}$ (day of negative test)")):
        m, lo, hi = band(MAIN, key); ax[0, 1].plot(t_, m, label=lab)
    INF = np.array([r["ts"]["INF"] for r in R[MAIN]], float); ax[0, 1].plot(t_, INF.mean(0), "k--", label=r"true infected $N_E+N_I$ (same instant)")
    ax[0, 1].set(xlabel="day $k$", ylabel="individuals (mean)", title="(b) what agents know (main policy)"); ax[0, 1].legend(fontsize=6)
    for key, lab in (("TEST", "tested per day"), ("CKET", r"tested while knowing exposure $C_{KE\wedge T}$"), ("ISO", "isolating")):
        m, lo, hi = band(MAIN, key); ax[1, 0].plot(t_, m, label=lab); ax[1, 0].fill_between(t_, lo, hi, alpha=.12)
    ax[1, 0].set(xlabel="day $k$", ylabel="individuals per day (mean, band 10-90%)", title="(c) testing and isolation activity (main policy)"); ax[1, 0].legend(fontsize=6)
    hid = INF - np.array([r["ts"]["CKI"] for r in R[MAIN]], float)
    ax[1, 1].plot(t_, hid.mean(0), c="tab:red", label=r"undetected infections $N_E+N_I-C_{KI}$"); ax[1, 1].fill_between(t_, np.percentile(hid, 10, 0), np.percentile(hid, 90, 0), color="tab:red", alpha=.15)
    NIs = [r["ts"]["NI"] for r in R[MAIN]]; ax[1, 1].plot(t_, np.mean(NIs, 0), c="tab:blue", label=r"symptomatic $N_I$")
    ax[1, 1].set(xlabel="day $k$", ylabel="individuals (mean)", title="(d) hidden infections (main policy)"); ax[1, 1].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "p4_traj.pdf")); plt.close()

    names = [p.name for p in POLICIES]; short = [nm.replace("stronger ", "str. ").replace("two-day test", "two-day") for nm in names]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    data = [[r["peak"] for r in R[nm]] for nm in names]
    bp = ax.boxplot(data[::-1], orientation="horizontal", tick_labels=short[::-1], showfliers=True, patch_artist=True)
    for b_, nm in zip(bp["boxes"], names[::-1]): b_.set_facecolor("#cfe3f5" if nm != NOC else "#dddddd")
    ax.axvline(Imax, c="k", ls="--", lw=1); ax.text(Imax + 1, len(names) + 0.55, r"$I_{max}=%d$" % Imax, fontsize=8)
    ax.set(xlabel=r"peak number of symptomatic infected $\max_k N_I[k]$ (%d seeds)" % NSEED, title="Peak-infection distribution across seeds"); ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "p4_peaks.pdf")); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].scatter(fd[pk < Imax], pk[pk < Imax], s=14, label=r"$\varphi_1$ satisfied", c="tab:green"); ax[0].scatter(fd[pk >= Imax], pk[pk >= Imax], s=14, label=r"$\varphi_1$ violated", c="tab:red")
    ax[0].axhline(Imax, c="k", ls="--", lw=.8); ax[0].set(xlabel="day of first positive test result", ylabel=r"peak $N_I$", title="main policy, %d seeds" % NSEED); ax[0].legend(fontsize=7)
    inf_d = np.array([r["inf_at_detect"] for r in M])
    ax[1].scatter(inf_d[pk < Imax], pk[pk < Imax], s=14, c="tab:green"); ax[1].scatter(inf_d[pk >= Imax], pk[pk >= Imax], s=14, c="tab:red")
    ax[1].axhline(Imax, c="k", ls="--", lw=.8); ax[1].set(xlabel=r"true infections $N_E+N_I$ on the day of first positive result", ylabel=r"peak $N_I$", title="detection delay matters because the epidemic grows")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "p4_fail.pdf")); plt.close()

    with open(os.path.join(REP, "gen_p4_macros.tex"), "w") as f:
        for k_, v in macros.items(): f.write("\\newcommand{\\%s}{%s}\n" % ("pf" + k_, v))
    print("macros:", {k: v for k, v in macros.items()})
    for p in POLICIES:
        a_ = agg(p.name); print("%-52s peak %.1f+-%.1f worst %3d  attack %.2f  phi1 %3.0f%%  phi2 %3.0f%%  tests %.0f  ok3 %d/%d ok4 %d/%d" % (p.name, a_["pk"], a_["sd"], a_["worst"], a_["att"], a_["p1"], a_["p2"], a_["tests"], a_["ok3"], a_["tot3"], a_["ok4"], a_["tot4"]))
