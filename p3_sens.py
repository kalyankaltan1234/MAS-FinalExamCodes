import numpy as np, math, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import p3_ganga as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs"); REP = os.path.join(HERE, "..", "report")
BUFFER = g.DSAFE
BURSTS = {"v01": g.TSTAR, "v02": g.TSTAR}
ds = g.CITIES

def margins(vf=4.0, dist_scale=1.0, comm=g.DELTA_C, evac_scale=1.0, dists=None):
    dd = g.DIST if dists is None else dists
    tau = {e: g.tau_of(d * dist_scale, vf) for e, d in dd.items()}
    fa = g.ffr_generic(tau, BURSTS)

    return {n: fa[n] - (g.TSTAR + comm + evac_scale * g.evac(n) + BUFFER) for n in ds}

MAC = []

def mac(name, val): MAC.append(r"\newcommand{\%s}{%s}" % (name, val))

KEY = {"v4": "Four", "v5": "Five", "v3": "Three", "v2": "Two", "v1": "One", "v6": "Six", "v7": "Seven", "v8": "Eight"}
base = margins(); jb = min(base, key=base.get)
print("baseline margins after buffer:", {k: round(v) for k, v in base.items()})
assert jb == "v3"
raw_j = base["v3"] + BUFFER
mac("PsensBaseJoshMargin", "%d" % base["v3"]); mac("PsensBaseJoshRaw", "%d" % raw_j)
mac("PsensBaseOtherMin", "%d" % min(v for k, v in base.items() if k != "v3")); mac("PsensBaseMax", "%d" % max(base.values()))

SW = {"speed": ("flood speed (m/s)", np.arange(2, 12.0001, 0.01), lambda x: margins(vf=x)),
      "dist": ("distance scale (x my edge lengths)", np.arange(0.3, 1.6001, 0.002), lambda x: margins(dist_scale=x)),
      "lat": ("alarm latency (min)", np.arange(0, 600.1, 1.0), lambda x: margins(comm=x)),
      "evac": ("evacuation-time scale (x)", np.arange(0.5, 8.0001, 0.005), lambda x: margins(evac_scale=x))}
fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.3)); crit = {}; first = {}
for a, (key, (lab, xs, f)) in zip(axs, SW.items()):
    M = np.array([[f(x)[n] for n in ds] for x in xs]) / 60.0
    for j, n in enumerate(ds): a.plot(xs, M[:, j], label=n)
    a.axhline(0, color='k', lw=.8); a.set_xlabel(lab); a.set_ylabel("margin after 60-min buffer (h)"); a.grid(alpha=.3)
    a.set_ylim(-3, 12) if key in ("speed", "evac", "lat") else None
    worst = M.min(1); neg = np.where(worst < 0)[0]

    def boundary(col):
        ng = np.where(col < 0)[0]
        if len(ng) == 0: return None
        if key == "dist": return float(xs[ng[-1]])
        return float(xs[ng[0]])
    crit[key] = {n: boundary(M[:, j]) for j, n in enumerate(ds)}
    firstcity = ds[int(np.argmin(M[neg[0]] if key != "dist" else M[neg[-1]]))] if len(neg) else "-"
    first[key] = (boundary(worst), firstcity)
    print(lab, "-> system fails", "below" if key == "dist" else "from", first[key])
axs[0].legend(fontsize=6, ncol=2); plt.tight_layout(); plt.savefig(os.path.join(OUT, "p3_sens.pdf")); plt.close()
mac("PsensSpeedThr", "%.2f" % first["speed"][0]); mac("PsensDistThr", "%.3f" % first["dist"][0])
mac("PsensLatThr", "%d" % first["lat"][0]); mac("PsensEvacThr", "%.3f" % first["evac"][0])
mac("PsensEvacMin", "%d" % round(first["evac"][0] * g.evac("v3")))
mac("PsensLatExtra", "%d" % (first["lat"][0] - g.DELTA_C)); mac("PsensRawLat", "%d" % (raw_j + g.DELTA_C))
mac("PsensFirstCitySpeed", g.NAMES[first["speed"][1]]); mac("PsensFirstCityLat", g.NAMES[first["lat"][1]])
assert all(first[k][1] == "v3" for k in first), first

rows = []
for n in ds:
    c = crit
    def f(v, fmt, none=r"$>$ range"):
        return none if v is None else fmt % v
    rows.append(r"%s %s & %s & %s & %s & %s\\" % (g.LAB[n], g.NAMES[n], f(c["speed"][n], "%.2f", "none in [2,12]"),
                f(c["dist"][n], "%.2f", "none in [0.3,1.6]"), f(c["lat"][n], "%d", "none in [0,600]"), f(c["evac"][n], "%.2f", "none in [0.5,8]")))
g.write_table("gen_p3_sens_crit.tex", "lcccc", r"city & fails above speed (m/s) & fails below distance scale & fails from latency (min) & fails above evacuation scale", rows)

vf = np.linspace(2, 12, 60); cm = np.linspace(0, 240, 60)
Z = np.array([[margins(vf=a, comm=b)["v3"] / 60 for a in vf] for b in cm])
plt.figure(figsize=(4.8, 3.5)); cs = plt.contourf(vf, cm, Z, levels=np.linspace(-4, 6, 21), cmap="RdYlGn"); plt.colorbar(cs, label="Joshimath margin after buffer (h)")
plt.contour(vf, cm, Z, levels=[0], colors='k'); plt.scatter([4], [g.DELTA_C], c='b', marker='*', s=80, label="design point")
plt.xlabel("flood speed (m/s)"); plt.ylabel("alarm latency (min)"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(OUT, "p3_sens2d.pdf")); plt.close()

def mc(sampler, N=5000, seed=1):
    r = np.random.default_rng(seed); fails = {n: 0 for n in ds}; anyf = 0; mj = []
    for _ in range(N):
        mm = margins(**sampler(r))
        for n, v in mm.items(): fails[n] += v < 0
        anyf += min(mm.values()) < 0; mj.append(mm["v3"])
    return {n: fails[n] / N for n in ds}, anyf / N, np.array(mj)

def s1(r):
    return dict(vf=r.uniform(3, 8), dist_scale=r.uniform(0.8, 1.2), comm=r.choice([5, 15, 30, 60]), evac_scale=r.uniform(0.8, 2.0))

def s2(r):
    return dict(dists={e: r.uniform(lo, hi) for e, (d, lo, hi, art) in g.EDGES.items()})

p1, any1, mj1 = mc(s1); p2, any2, mj2 = mc(s2)
print("MC1 (speed U[3,8], dist x U[.8,1.2], latency {5,15,30,60}, evac x U[.8,2]):", {n: round(v, 4) for n, v in p1.items()}, "any", any1)
print("MC2 (edge lengths U[source lo, source hi]):", {n: round(v, 4) for n, v in p2.items()}, "any", any2, "Joshimath margin range (min) %.0f..%.0f median %.0f" % (mj2.min(), mj2.max(), np.median(mj2)))
rows = [r"%s %s & %.4f & %.4f\\" % (g.LAB[n], g.NAMES[n], p1[n], p2[n]) for n in ds]
rows.append(r"\midrule any city & %.4f & %.4f\\" % (any1, any2))
g.write_table("gen_p3_sens_mc.tex", "lcc", r"city & P(fail), Monte Carlo 1 & P(fail), Monte Carlo 2 (distance ranges)", rows)
mac("PsensMCJosh", "%.3f" % p1["v3"]); mac("PsensMCAny", "%.3f" % any1)
mac("PsensMCTwoJosh", "%.3f" % p2["v3"]); mac("PsensMCTwoAny", "%.3f" % any2)
mac("PsensMCTwoMin", "%d" % round(mj2.min())); mac("PsensMCTwoMax", "%d" % round(mj2.max())); mac("PsensMCTwoMed", "%d" % round(np.median(mj2)))
mac("PsensSpeedPct", "%.0f" % (100 * (8 - first["speed"][0]) / 5))
sec = sorted(((v, n) for n, v in p1.items() if n != "v3"), reverse=True)
mac("PsensMCSecondCity", g.NAMES[sec[0][1]]); mac("PsensMCSecond", "%.3f" % sec[0][0])
open(os.path.join(REP, "gen_p3_sens_macros.tex"), "w").write("\n".join(MAC) + "\n")
