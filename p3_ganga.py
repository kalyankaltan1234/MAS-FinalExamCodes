import numpy as np, heapq, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figs"); REP = os.path.join(HERE, "..", "report")

NAMES = {"v01": "Gomukh glacier", "v02": "Satopanth glacier", "v4": "Uttarkashi", "v5": "New Tehri",
         "v3": "Joshimath", "v2": "Rudraprayag", "v1": "Devprayag", "v6": "Rishikesh",
         "v7": "Haridwar", "v8": "Bijnor"}
LAB = {"v01": r"$v_{01}$", "v02": r"$v_{02}$", "v4": r"$v_4$", "v5": r"$v_5$", "v3": r"$v_3$", "v2": r"$v_2$",
       "v1": r"$v_1$", "v6": r"$v_6$", "v7": r"$v_7$", "v8": r"$v_8$"}

EDGES = {("v01", "v4"): (118, 108.0, 123.0, False),
         ("v02", "v3"): (58, 53.0, 71.0, False),
         ("v4", "v5"): (116, 76.0, 119.0, False),
         ("v4", "v2"): (176, 164.0, 178.0, True),
         ("v4", "v1"): (160, 136.0, 173.0, False),
         ("v5", "v3"): (254, 237.0, 271.0, True),
         ("v3", "v2"): (113, 111.0, 122.0, False),
         ("v5", "v1"): (74, 61.0, 79.0, False),
         ("v2", "v1"): (67, 65.0, 70.0, False),
         ("v1", "v6"): (73, 70.0, 75.0, False),
         ("v1", "v7"): (96, 93.1, 99.0, False),
         ("v1", "v8"): (175, 169.1, 181.0, False)}
GMAPS_CSV = os.environ.get("P3_GMAPS_CSV", os.path.join(HERE, "gmaps_distances.csv"))
GMAPS = {}; GMAPS_ROWS = []
if os.path.exists(GMAPS_CSV):
    import csv
    with open(GMAPS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            e = tuple(row["edge"].strip().split("->")); GMAPS_ROWS.append((e, row))
            if row["km"].strip():
                GMAPS[e] = float(row["km"])
for e, km in GMAPS.items():
    assert e in EDGES and km > 0, e
    EDGES[e] = (km, km, km, EDGES[e][3])
DIST = {e: v[0] for e, v in EDGES.items()}
POP = {"v4": 0.25, "v5": 0.50, "v3": 0.75, "v2": 1.00, "v1": 0.80, "v6": 1.20, "v7": 2.00, "v8": 1.50}
order = ["v01", "v02", "v4", "v5", "v3", "v2", "v1", "v6", "v7", "v8"]
nodes = order; SOURCES = ("v01", "v02"); CITIES = [n for n in order if n not in SOURCES]
VN, VF, DT = 3.0, 4.0, 60.0
QN, QF = 18.0, 70.0
DELTA_C, DSAFE, T_RELAY, TSTAR = 5, 60, 10, 60
FLOOD_RATIO = 1.05

def evac(i): return int(round(30 + 150 * POP[i]))

succ = {n: [b for (a, b) in EDGES if a == n] for n in nodes}
pred = {n: [a for (a, b) in EDGES if b == n] for n in nodes}
edges = DIST

def tau_of(dkm, v): return math.ceil(round(1000.0 * dkm / (60.0 * v), 9))

def taus(v, dist=None, scale=1.0):
    dist = DIST if dist is None else dist
    return {e: tau_of(d * scale, v) for e, d in dist.items()}

tauN, tauF = taus(VN), taus(VF)

def ffr_generic(tau, bursts):
    dist = {n: math.inf for n in nodes}; pq = []
    for s, t in bursts.items(): dist[s] = t; heapq.heappush(pq, (t, s))
    while pq:
        t, u = heapq.heappop(pq)
        if t > dist[u]: continue
        for w in succ[u]:
            if t + tau[(u, w)] < dist[w]: dist[w] = t + tau[(u, w)]; heapq.heappush(pq, (dist[w], w))
    return dist

def norm(b):
    return {s: (v if isinstance(v, tuple) else (v, None)) for s, v in b.items()}

def t0s(b): return {s: v[0] for s, v in norm(b).items()}

def ffr(bursts): return ffr_generic(tauF, t0s(bursts))

def ffr_last(bursts, tau=None):
    tau = tauF if tau is None else tau
    last = {n: -math.inf for n in nodes}
    for s, t in t0s(bursts).items(): last[s] = t
    for u in order:
        for w in succ[u]:
            if last[u] > -math.inf: last[w] = max(last[w], last[u] + tau[(u, w)])
    return last

def splitting_flows(src):
    f = {}
    for n in order:
        f[n] = src[n] if n in SOURCES else sum(f[u] / len(succ[u]) for u in pred[n])
    return f

A_N, A_F = QN / 2, QF / 2
EXC = A_F - A_N
STEADY = splitting_flows({"v01": A_N, "v02": A_N})
FLOODSS = splitting_flows({"v01": A_F, "v02": A_F})

def default_T(bursts):
    nb = norm(bursts)
    longest = {n: 0 for n in nodes}
    for u in order:
        for w in succ[u]: longest[w] = max(longest[w], longest[u] + tauF[(u, w)])
    end = max(t0 + (d or 0) for t0, d in nb.values())
    return int(end + max(longest.values()) + 600)

def injection(n, c, bursts, t):
    if c == "N": return A_N
    nb = norm(bursts)
    if n in nb:
        t0, d = nb[n]
        if t >= t0 and (d is None or t < t0 + d): return EXC
    return 0.0

def simulate(bursts, T=None):
    T = default_T(bursts) if T is None else T
    mx = max(tauN.values()); H = mx
    arr = {(e, c): np.zeros(T + mx + 2) for e in EDGES for c in "NF"}
    ent = {(e, c): np.zeros(H + T) for e in EDGES for c in "NF"}
    dlv = {(e, c): np.zeros(T) for e in EDGES for c in "NF"}
    I0 = {}
    for e in EDGES:
        f = 60.0 * STEADY[e[0]] / len(succ[e[0]])
        ent[(e, "N")][:H] = f
        arr[(e, "N")][0:tauN[e]] += f
        I0[(e, "N")] = f * tauN[e]; I0[(e, "F")] = 0.0
    Qc = {(n, c): np.zeros(T) for n in nodes for c in "NF"}
    for t in range(T):
        for u in order:
            for c in "NF":
                if u in SOURCES:
                    q = injection(u, c, bursts, t)
                else:
                    vol = 0.0
                    for p in pred[u]:
                        v = arr[((p, u), c)][t]; dlv[((p, u), c)][t] = v; vol += v
                    q = vol / 60.0
                Qc[(u, c)][t] = q
                for w in succ[u]:
                    e = (u, w); tau = tauN[e] if c == "N" else tauF[e]
                    vol = 60.0 * q / len(succ[u])
                    arr[(e, c)][t + tau] += vol; ent[(e, c)][H + t] = vol
    q = {n: Qc[(n, "N")] + Qc[(n, "F")] for n in nodes}
    return dict(t=np.arange(T), H=0, HE=H, T=T, q=q, Qc=Qc, ent=ent, arr=arr, dlv=dlv, I0=I0, bursts=dict(bursts))

def after0(sim, n): return sim["q"][n]

def first_arrival(sim):
    out = {}
    for n in nodes:
        f = sim["Qc"][(n, "F")] > 1e-12
        out[n] = int(np.argmax(f)) if f.any() else math.inf
    return out

def peaks(sim):
    pk = {}
    for n in nodes:
        h = after0(sim, n); m = float(h.max()); k = int(np.argmax(h >= m - 1e-9)); pk[n] = (m, k)
    return pk

def settle_time(sim, n):
    h = after0(sim, n); d = np.where(np.abs(np.diff(h)) > 1e-9)[0]
    return int(d[-1] + 1) if len(d) else 0

def level_ratio(sim, n):
    return (after0(sim, n) / STEADY[n]) ** 0.6

def in_transit_total(sim, c, t0=None):
    HE = sim["HE"]; T = sim["T"]; tot = np.zeros(T)
    for e in EDGES:
        cs = np.concatenate([[0.0], np.cumsum(sim["ent"][(e, c)])])
        tau = tauN[e] if c == "N" else tauF[e]
        idx = HE + np.arange(T) + 1
        tot += cs[idx] - cs[np.maximum(idx - tau, 0)]
    return tot

def evacuation(sim, comm=DELTA_C, evac_scale=1.0):
    T = sim["T"]; t_alarm = min(t0s(sim["bursts"]).values()); out = {}
    for n in CITIES:
        r = 1.0 / (evac_scale * evac(n)); alert = t_alarm + comm
        e = np.zeros(T + 1)
        for t in range(T):
            e[t + 1] = min(1.0, e[t] + (r if t >= alert else 0.0))
        out[n] = dict(e=e, alert=alert, Prem=POP[n] * (1 - e))
    return out

def distributed_bellman_ford(bursts):
    b0 = t0s(bursts); cur = {n: (b0.get(n, math.inf)) for n in nodes}; rounds = 0
    while True:
        new = dict(cur)
        for i in nodes:
            for j in pred[i]: new[i] = min(new[i], cur[j] + tauF[(j, i)])
        if new == cur: return cur, rounds
        cur = new; rounds += 1

def _shift(a, k, default):
    out = np.full(len(a), default, dtype=float)
    if k < len(a): out[k:] = a[:len(a) - k]
    return out

def distributed_hydrographs(bursts, T=None):
    T = default_T(bursts) if T is None else T
    Q = {}
    for n in nodes:
        for c in "NF":
            if n in SOURCES: Q[(n, c)] = np.array([injection(n, c, bursts, t) for t in range(T)], dtype=float)
            else: Q[(n, c)] = np.full(T, STEADY[n] if c == "N" else 0.0)
    rounds = 0; changed_rounds = 0
    while True:
        new = {}
        for i in nodes:
            for c in "NF":
                if i in SOURCES: new[(i, c)] = Q[(i, c)]; continue
                a = np.zeros(T)
                for j in pred[i]:
                    tau = tauN[(j, i)] if c == "N" else tauF[(j, i)]
                    dflt = STEADY[j] / len(succ[j]) if c == "N" else 0.0
                    a += _shift(Q[(j, c)] / len(succ[j]), tau, dflt)
                new[(i, c)] = a
        rounds += 1
        if all(np.allclose(new[k], Q[k], atol=1e-9) for k in Q): break
        Q = new; changed_rounds = rounds
    tot = {n: Q[(n, "N")] + Q[(n, "F")] for n in nodes}
    pk = {}
    for n in nodes:
        m = float(tot[n].max()); pk[n] = (m, int(np.argmax(tot[n] >= m - 1e-9)))
    return tot, pk, changed_rounds, 0

def longest_path(tau):
    lg = {n: 0 for n in nodes}
    for u in order:
        for w in succ[u]: lg[w] = max(lg[w], lg[u] + tau[(u, w)])
    return max(lg.values())

def simulate_regime(bursts, T=None):
    nb = norm(bursts); end = max(t0 + (d or 0) for t0, d in nb.values())
    T = int(end + longest_path(tauN) + 600) if T is None else T
    mx = max(tauN.values()); H = mx
    arr = {e: np.zeros(T + mx + 2) for e in EDGES}
    ent = {e: np.zeros(H + T) for e in EDGES}
    dlv = {e: np.zeros(T) for e in EDGES}
    I0 = {}
    for e in EDGES:
        f = 60.0 * STEADY[e[0]] / len(succ[e[0]])
        ent[e][:H] = f; arr[e][0:tauN[e]] += f; I0[e] = f * tauN[e]
    Q = {n: np.zeros(T) for n in nodes}; flooded = {n: np.zeros(T, bool) for n in nodes}
    for t in range(T):
        for u in order:
            if u in SOURCES:
                q = A_N + injection(u, "F", bursts, t)
            else:
                vol = 0.0
                for p in pred[u]:
                    v = arr[(p, u)][t]; dlv[(p, u)][t] = v; vol += v
                q = vol / 60.0
            Q[u][t] = q
            fl = bool(q > FLOOD_RATIO * STEADY[u]); flooded[u][t] = fl
            for w in succ[u]:
                e = (u, w); tau = tauF[e] if fl else tauN[e]
                vol = 60.0 * q / len(succ[u])
                arr[e][t + tau] += vol; ent[e][H + t] = vol
    return dict(T=T, HE=H, q=Q, flooded=flooded, ent=ent, arr=arr, dlv=dlv, I0=I0, bursts=dict(bursts))

def regime_in_transit(sim):
    T = sim["T"]; HE = sim["HE"]; tot = np.zeros(T); per = {}
    for e in EDGES:
        v = sim["I0"][e] + np.cumsum(sim["ent"][e][HE:HE + T]) - np.cumsum(sim["arr"][e][:T])
        per[e] = v; tot += v
    return tot, per

def regime_first_arrival(sim):
    out = {}
    for n in nodes:
        f = sim["flooded"][n]
        out[n] = int(np.argmax(f)) if f.any() else math.inf
    return out

def regime_peaks(sim):
    pk = {}
    for n in nodes:
        h = sim["q"][n]; m = float(h.max()); pk[n] = (m, int(np.argmax(h >= m - 1e-9)))
    return pk

def passed_excess(sim, n):
    return np.cumsum(60.0 * sim["Qc"][(n, "F")])

LOG = []

def check(name, cond, detail=""):
    assert cond, "CHECK FAILED: " + name + " " + detail
    LOG.append((name, detail)); print("  [PASS] %s %s" % (name, detail))

SCEN = {"A": {"v01": 60}, "B": {"v02": 60}, "C": {"v01": 60, "v02": 60}, "D": {"v01": 60, "v02": 240},
        "E": {"v01": (60, 360), "v02": (240, 360)}}
PULSE_SINGLE = {"v01": {"v01": (60, 360)}, "v02": {"v02": (240, 360)}}
SCEN_TXT = {"A": r"burst at $v_{01}$ only ($t^\star=60$)", "B": r"burst at $v_{02}$ only ($t^\star=60$)",
            "C": r"both bursts at $t^\star=60$ (baseline)", "D": r"staggered persistent: $v_{01}$ at 60, $v_{02}$ at 240",
            "E": r"staggered 6-h pulses: $v_{01}$ at 60, $v_{02}$ at 240"}

def run_checks():
    print("== verification ==")
    check("all edge delays positive (normal and flood)", all(v > 0 for v in tauN.values()) and all(v > 0 for v in tauF.values()),
          "(min tauF = %d min)" % min(tauF.values()))
    check("delay formula tau = ceil(1000 L/(60 v)) min (L in km, v in m/s) on every edge, both classes",
          all(tauN[e] == math.ceil(round(1000.0 * DIST[e] / (60.0 * VN), 9)) and tauF[e] == math.ceil(round(1000.0 * DIST[e] / (60.0 * VF), 9)) for e in EDGES),
          "(e.g. v01->v4: %g km at 3 m/s = %d min, at 4 m/s = %d min)" % (DIST[("v01", "v4")], tauN[("v01", "v4")], tauF[("v01", "v4")]))
    check("flood delay <= normal delay on every edge", all(tauF[e] <= tauN[e] for e in EDGES))
    indeg = {n: len(pred[n]) for n in nodes}; stack = [n for n in nodes if indeg[n] == 0]; seen = 0
    while stack:
        u = stack.pop(); seen += 1
        for w in succ[u]:
            indeg[w] -= 1
            if indeg[w] == 0: stack.append(w)
    check("graph is acyclic (Kahn elimination removes all %d nodes)" % len(nodes), seen == len(nodes))
    pos = {n: i for i, n in enumerate(order)}
    check("topological order respects every directed edge", all(pos[a] < pos[b] for a, b in EDGES), "(%d edges)" % len(EDGES))
    check("exam edges (v4,v2) and (v5,v3) present, (v3,v5) absent", ("v4", "v2") in EDGES and ("v5", "v3") in EDGES and ("v3", "v5") not in EDGES)
    want = {"v4": 9.0, "v5": 3.0, "v3": 10.5, "v2": 13.5, "v1": 18.0, "v6": 6.0, "v7": 6.0, "v8": 6.0}
    check("splitting-rule steady flows: v4 9, v5 3, v3 10.5, v2 13.5, v1 18, v6=v7=v8 6 ML/s", all(abs(STEADY[n] - w) < 1e-9 for n, w in want.items()))
    check("flood steady flows: v1 = 70, v6=v7=v8 = 23.33 ML/s", abs(FLOODSS["v1"] - QF) < 1e-9 and abs(FLOODSS["v6"] - QF / 3) < 1e-9)
    check("glacier discharge: normal 2 x 9 = 18 ML/s, burst 2 x (9 + 26) = 70 ML/s", abs(2 * A_N - QN) < 1e-9 and abs(2 * (A_N + EXC) - QF) < 1e-9)
    sims = {}; worst_bal = 0.0
    for k, b in SCEN.items():
        s = simulate(b); sims[k] = s; T = s["T"]; HE = s["HE"]
        wj = 0.0; wi = 0.0
        for u in nodes:
            for c in "NF":
                if succ[u]:
                    out = sum(s["ent"][((u, w), c)][HE:HE + T] / 60.0 for w in succ[u])
                    wj = max(wj, float(np.max(np.abs(out - s["Qc"][(u, c)]))))
                if pred[u]:
                    inn = sum(s["dlv"][((p, u), c)] / 60.0 for p in pred[u])
                    wi = max(wi, float(np.max(np.abs(inn - s["Qc"][(u, c)]))))
        check("[%s] outflow sum_j q_{i->j}[t] = q_i[t], every node, class and minute" % k, wj < 1e-9, "(max error %.1e ML/s)" % wj)
        check("[%s] inflow over incoming edges = q_i[t] at every junction, class and minute" % k, wi < 1e-9, "(max error %.1e ML/s)" % wi)
        wj2 = 0.0
        for u in nodes:
            if pred[u] and succ[u]:
                for c in "NF":
                    i_n = sum(s["dlv"][((p, u), c)] for p in pred[u]); o_n = sum(s["ent"][((u, w), c)][HE:HE + T] for w in succ[u])
                    wj2 = max(wj2, float(np.max(np.abs(i_n - o_n))))
        check("[%s] junction balance: volume in = volume out (no storage at nodes), every minute" % k, wj2 < 1e-6, "(max error %.1e ML/min)" % wj2)
        we = 0.0
        for e in EDGES:
            for c in "NF":
                tau = tauN[e] if c == "N" else tauF[e]
                cs = np.concatenate([[0.0], np.cumsum(s["ent"][(e, c)])])
                idx = HE + np.arange(T) + 1
                window = cs[idx] - cs[np.maximum(idx - tau, 0)]
                dl = np.cumsum(s["arr"][(e, c)][:T])
                account = s["I0"][(e, c)] + np.cumsum(s["ent"][(e, c)][HE:HE + T]) - dl
                we = max(we, float(np.max(np.abs(window - account))))
                left = float(s["arr"][(e, c)][T:].sum())
                we = max(we, abs(left - window[-1]))
        check("[%s] per edge and class: entered = delivered + still in transit, every minute (window volume = ledger)" % k, we < 1e-6, "(max error %.1e ML)" % we)
        bal = 0.0
        for c in "NF":
            inj = sum(np.cumsum(60.0 * s["Qc"][(n, c)]) for n in SOURCES)
            out = sum(np.cumsum(60.0 * s["Qc"][(n, c)]) for n in nodes if not succ[n])
            tr = in_transit_total(s, c); i0 = sum(s["I0"][(e, c)] for e in EDGES)
            scale = max(1.0, float(inj[-1] + i0))
            bal = max(bal, float(np.max(np.abs(i0 + inj - out - tr))) / scale)
        worst_bal = max(worst_bal, bal)
        check("[%s] global mass balance per class: initial storage + injected = delivered to sinks + in transit, every minute" % k, bal < 1e-9, "(max relative error %.1e)" % bal)
        over = max(float(after0(s, n).max() - FLOODSS[n]) for n in nodes)
        under = min(float(after0(s, n).min() - STEADY[n]) for n in nodes)
        check("[%s] hydrographs stay within [steady, flood-steady] flow" % k, over < 1e-9 and under > -1e-9, "(max excess %.1e, min deficit %.1e ML/s)" % (over, under))
        check("[%s] baseline class N stays at its steady flow at every node" % k, all(float(np.max(np.abs(s["Qc"][(n, "N")] - STEADY[n]))) < 1e-9 for n in nodes))
        if any(d is not None for _, d in norm(b).values()):
            fin = max(abs(after0(s, n)[-1] - STEADY[n]) for n in nodes)
            check("[%s] after the pulses have passed every node returns to its steady flow" % k, fin < 1e-9, "(max dev %.1e)" % fin)
        elif set(b) == set(SOURCES):
            fin = max(abs(after0(s, n)[-1] - FLOODSS[n]) for n in nodes)
            check("[%s] final state equals flood splitting-rule flows (all nodes)" % k, fin < 1e-9, "(max dev %.1e)" % fin)
        fa = first_arrival(s); an = ffr(b)
        bad = [n for n in nodes if fa[n] != an[n]]
        check("[%s] simulated first arrival of flood water = FlashFloodReach at all %d nodes" % (k, len(nodes)), not bad, "mismatch: %s" % bad if bad else "")
        bf, rb = distributed_bellman_ford(b)
        check("[%s] distributed Bellman-Ford FFR = Dijkstra FFR" % k, bf == an, "(%d rounds, DAG depth bound 6)" % rb)
        assert rb <= 6
        Q, pk, rc, H2 = distributed_hydrographs(b, s["T"])
        err = max(float(np.max(np.abs(Q[n] - s["q"][n]))) for n in nodes)
        pks = peaks(s)
        same = all(abs(pk[n][0] - pks[n][0]) < 1e-9 and pk[n][1] == pks[n][1] for n in nodes)
        check("[%s] hydrograph exchange reproduces the simulation, peaks and argmax at every node" % k, err < 1e-9 and same, "(%d changing rounds, max err %.1e)" % (rc, err))
        assert rc <= 6
    sC0 = sims["C"]
    wp_end = sum(float(passed_excess(sC0, n)[-1]) for n in SOURCES)
    wp_sink = sum(float(passed_excess(sC0, n)[-1]) for n in nodes if not succ[n])
    wp_tr = float(in_transit_total(sC0, "F")[-1])
    mono = all(np.all(np.diff(passed_excess(sC0, n)) >= -1e-12) for n in nodes)
    check("[C] cumulative passed excess volume W_i^pass is non-decreasing and (passed at sinks) + (excess in transit) = (excess injected at sources)",
          mono and abs(wp_sink + wp_tr - wp_end) < 1e-6 * max(1.0, wp_end), "(%.0f ML injected)" % wp_end)
    for k in "CDE":
        r = simulate_regime(SCEN[k]); sims["R_" + k] = r; T = r["T"]; HE = r["HE"]
        wj = 0.0; wi = 0.0
        for u in nodes:
            if succ[u]:
                out = sum(r["ent"][(u, w)][HE:HE + T] / 60.0 for w in succ[u]); wj = max(wj, float(np.max(np.abs(out - r["q"][u]))))
            if pred[u]:
                inn = sum(r["dlv"][(p, u)] / 60.0 for p in pred[u]); wi = max(wi, float(np.max(np.abs(inn - r["q"][u]))))
        check("[%s, all-water-at-4 m/s] outflow = q_i and inflow = q_i at every node and minute" % k, wj < 1e-9 and wi < 1e-9, "(max error %.1e ML/s)" % max(wj, wi))
        tot, per = regime_in_transit(r)
        neg = min(float(v.min()) for v in per.values())
        lost = max(abs(float(r["arr"][e].sum()) - (r["I0"][e] + float(r["ent"][e][HE:HE + T].sum()))) for e in EDGES)
        check("[%s, all-water-at-4 m/s] per edge: entered = delivered + in transit (ledger non-negative, nothing lost)" % k, neg > -1e-6 and lost < 1e-6, "(min ledger %.1e, max loss %.1e ML)" % (neg, lost))
        inj = sum(np.cumsum(60.0 * r["q"][n]) for n in SOURCES)
        out = sum(np.cumsum(60.0 * r["q"][n]) for n in nodes if not succ[n])
        i0 = sum(r["I0"].values()); scale = max(1.0, float(inj[-1] + i0))
        bal = float(np.max(np.abs(i0 + inj - out - tot))) / scale
        check("[%s, all-water-at-4 m/s] global mass balance: initial storage + injected = delivered to sinks + in transit, every minute" % k, bal < 1e-9, "(max relative error %.1e)" % bal)
        fa = regime_first_arrival(r); an_k = ffr(SCEN[k])
        bad = [n for n in nodes if fa[n] != an_k[n]]
        check("[%s, all-water-at-4 m/s] first flooded minute = FlashFloodReach at all %d nodes (so RawSlack and SafetyMargin are identical)" % (k, len(nodes)), not bad, "mismatch: %s" % bad if bad else "")
    pkC = peaks(sims["C"]); lastC = ffr_last(SCEN["C"])
    check("[C] simulated peak flow equals flood splitting-rule flow at every node", all(abs(pkC[n][0] - FLOODSS[n]) < 1e-9 for n in nodes))
    check("[C] time of peak = latest arrival of the full flood flow (max,+) at every node", all(pkC[n][1] == lastC[n] for n in nodes if n not in SOURCES))
    sims["_bal"] = worst_bal
    sC = sims["C"]; ev = evacuation(sC); an = ffr(SCEN["C"])
    for n in CITIES:
        e = ev[n]["e"]; assert np.all(np.diff(e) >= -1e-12) and e[-1] >= 1 - 1e-9
        t_clear = int(np.argmax(e >= 1 - 1e-9)); margin = an[n] - (TSTAR + DELTA_C + evac(n) + DSAFE)
        ok_sim = t_clear + DSAFE <= an[n]
        assert t_clear == TSTAR + DELTA_C + evac(n), (n, t_clear)
        assert ok_sim == (margin >= 0), (n, t_clear, an[n], margin)
        if margin >= 0: assert ev[n]["Prem"][an[n] - DSAFE] < 1e-9
    check("evacuation: simulated e_i[t] reaches 1 at t*+delta_c+EvacuateTime_i; (cleared+buffer <= FFR) <=> margin>=0 for all cities", True)
    check("remaining population P_i[1-e_i] = 0 at FFR_i - Delta_safe whenever margin >= 0", True)
    return sims

MAC = []

def write_table(fname, colspec, header, rows):
    body = "\n".join(rows)
    open(os.path.join(REP, fname), "w").write("\\begin{tabular}{%s}\\toprule\n%s\\\\\\midrule\n%s\n\\bottomrule\\end{tabular}\n" % (colspec, header, body))

def mac(name, val): MAC.append(r"\newcommand{\%s}{%s}" % (name, val))

def fmt_h(m): return "%.1f" % (m / 60.0)

def relay_and_none(sim):
    an = ffr(sim["bursts"]); ts = min(t0s(sim["bursts"]).values()); res = {}
    for n in CITIES:
        c = an[n] - (ts + DELTA_C + evac(n))
        direct = [ts + T_RELAY] if any(p in SOURCES for p in pred[n]) else []
        via = [an[u] + T_RELAY for u in pred[n] if u not in SOURCES and an[u] < math.inf]
        al = min(direct + via + [an[n]])
        r = an[n] - (al + evac(n)); res[n] = (c, r, -evac(n))
    return res

def write_all(sims):
    sC = sims["C"]; an = ffr(SCEN["C"]); last = ffr_last(SCEN["C"]); pk = peaks(sC)

    rows = []
    src = {("v01", "v4"): r"Gaumukh--Gangotri trek 18 km (\href{https://www.stayvista.com/blog/gangotri-to-gaumukh-trek-2026/}{StayVista}); Gangotri--Uttarkashi 90--105 km (\href{https://www.pilgrimagetour.in/blog/how-to-reach-gangotri/}{Pilgrimage Tour} 100, \href{https://www.moxtain.com/trips/gomukh-tapovan-trek}{Moxtain} 99.5, \href{https://www.chardhamtour.in/how-to-reach-gangotri.html}{Chardham Tour} 105, \href{https://www.manchalamushafir.com/uttarkashi-distance-chart}{Manchala Musafir} 96, \href{https://discoverwithdheeraj.com/gaumukh-travel-guide-the-birthplace-bhagirathi-river/}{Discover with Dheeraj} 90)",
           ("v02", "v3"): r"Badrinath--Joshimath 42.9--46 km (\href{https://www.travanya.com/tourism/india/uttarakhand/temples/joshimath/distance/from-badrinath/}{Travanya}, \href{https://rishikeshdaytour.com/blog/badrinath-to-joshimath-distance/}{Rishikesh Day Tour}); Satopanth--Badrinath trek leg \emph{rough}, 9 km to Laxmivan + 5 km to the lake (\href{https://uttarakhandtourism.gov.in/trek/satopanth-lake-trek}{Uttarakhand Tourism}, \href{https://himalayandreamtreks.in/trek/satopanth-tal/}{Himalayan Dream Treks}); I allowed 10--25 km",
           ("v4", "v5"): r"\href{https://www.travanya.com/tourism/india/uttarakhand/hill-stations/uttarkashi/distance/from-tehri}{Travanya} 115.6 (``Tehri''); \href{https://www.chardhamtour.in/how-to-reach-gangotri.html}{Chardham Tour} 27+92=119 via Dharasu; \href{https://www.manchalamushafir.com/uttarkashi-distance-chart}{Manchala Musafir} 76 (``Tehri'')",
           ("v4", "v2"): r"\textbf{ARTIFICIAL edge.} Road distance used only as a proxy for the delay: \href{https://www.manchalamushafir.com/uttarkashi-distance-chart}{Manchala Musafir} 178; \href{https://distancebetween2.com/uttarkashi/rudraprayag}{distancebetween2} 177.3; \href{https://www.himalayanhigh.in/how-to-reach-uttarkashi}{Himalayan High} 175; one further listing found by web search gave 164 (URL not retained)",
           ("v4", "v1"): r"\href{https://cabbazar.com/taxi/cab/uttarkashi-to-devprayag}{Cabbazar} 156; \href{https://distancebetween2.com/devprayag/uttarkashi}{distancebetween2} 160; \href{https://www.manchalamushafir.com/uttarkashi-distance-chart}{Manchala Musafir} 173; \href{https://www.himalayanhigh.in/how-to-reach-uttarkashi}{Himalayan High} 170; one further listing found by web search gave 136 (URL not retained)",
           ("v5", "v3"): r"\textbf{ARTIFICIAL edge, derived} (no direct source): New Tehri--Devprayag 74 + Devprayag--Rudraprayag 67 + Rudraprayag--Joshimath 113 (legs below); range = sum of leg ranges",
           ("v3", "v2"): r"\href{https://www.travanya.com/tourism/india/uttarakhand/temples/joshimath/distance/from-rudraprayag/}{Travanya} 113; \href{https://distancebetween2.com/joshimath/rudraprayag}{distancebetween2} (listing with Auli start 122); further listing 111",
           ("v5", "v1"): r"\href{https://uttarakhandguide.com/?p=7683}{Uttarakhand Guide} 74 (New Tehri town); Tehri-Dam variants 61--79 from other pages",
           ("v2", "v1"): r"\href{https://distancebetween2.com/rudraprayag/devprayag}{distancebetween2} 67.3; \href{https://cabbazar.com/taxi/cab/rudraprayag-to-devprayag}{Cabbazar} 65; distance charts 66, 70",
           ("v1", "v6"): r"\href{https://en.wikipedia.org/wiki/Devprayag}{Wikipedia} 74; \href{https://www.travanya.com/tourism/india/uttarakhand/temples/rishikesh/distance/from-devprayag/}{Travanya} 72.2; other pages 70, 75",
           ("v1", "v7"): r"\href{https://www.rome2rio.com/s/Devprayag/Haridwar}{Rome2rio} 93.1; \href{https://devdhamyatra.com/?p=4148}{Dev Dham Yatra} 98; other pages 95--99",
           ("v1", "v8"): r"\textbf{Derived}: Devprayag--Haridwar 96 + Haridwar--Bijnor 79 (\href{https://www.rome2rio.com/s/Haridwar/Bijnor}{Rome2rio} 76.9, \href{https://www.clearcarrental.com/haridwar-to-bijnor-distance}{Clear Car Rental} 76, \href{https://www.travanya.com/tourism/india/uttarakhand/temples/haridwar/distance/from-bijnor}{Travanya} 79, \href{https://cabbazar.com/taxi/cab/haridwar-to-bijnor}{Cabbazar} 81, Uber 82)"}
    gm_url = {e: row["google_maps_url"] for e, row in GMAPS_ROWS}
    for (a, b), (d, lo, hi, art) in EDGES.items():
        spread = 100.0 * (hi - lo) / d
        if (a, b) in GMAPS:
            rows.append(r"%s$\to$%s %s & %g & -- & \scriptsize Google Maps (read by the student): \href{%s}{route}\\" % (LAB[a], LAB[b], "(artificial)" if art else "", d, gm_url[(a, b)]))
        else:
            rows.append(r"%s$\to$%s %s & %g & %g--%g (%.0f\%%) & \scriptsize %s\\" % (LAB[a], LAB[b], "(artificial)" if art else "", d, lo, hi, spread, src[(a, b)]))
    write_table("gen_p3_dist.tex", "@{}llcp{8.7cm}@{}", "edge & km used & source range (spread) & sources and leg values", rows)

    glr = []
    for e, row in GMAPS_ROWS:
        a, b = e
        glr.append(r"%s$\to$%s %s & %s to %s & \href{%s}{open route} & %s\\" % (LAB[a], LAB[b], "(artificial)" if EDGES[e][3] else "", row["from_place"].split(",")[0], row["to_place"].split(",")[0], row["google_maps_url"], ("%g" % GMAPS[e]) if e in GMAPS else "not read"))
    write_table("gen_p3_gmaps.tex", "llll", r"edge & places & Google Maps directions link & km read from Google Maps", glr)

    t1 = []; raw = {}; mar = {}
    for n in CITIES:
        f = an[n] - TSTAR; rs = an[n] - (TSTAR + DELTA_C + evac(n)); mg = rs - DSAFE; raw[n] = rs; mar[n] = mg
        t1.append(r"%s %s & %d & %s & %d & %d & %d\\" % (LAB[n], NAMES[n], f, fmt_h(f), evac(n), rs, mg))
    write_table("gen_p3_timing.tex", "lrrrrr", r"city & FFR$-t^\star$ (min) & FFR$-t^\star$ (h) & $T_{evac}$ (min) & raw slack (min) & safety margin (min)", t1)

    t2 = []
    for n in CITIES:
        qp, kp = pk[n]; lr = (qp / STEADY[n]) ** 0.6
        t_full = last[n] - TSTAR; t_set = settle_time(sC, n) - TSTAR
        t2.append(r"%s %s & %.1f & %.1f & %.1f & %s & %s & %s & %.2f\\" % (LAB[n], NAMES[n], STEADY[n], qp, qp / STEADY[n], fmt_h(an[n] - TSTAR), fmt_h(kp - TSTAR), fmt_h(t_full), lr))
    write_table("gen_p3_flow.tex", "lrrrrrrr", r"city & nominal $q$ (ML/s) & $q^{peak}$ (ML/s) & $q^{peak}/q^{nom}$ & FFR (h after $t^\star$) & $t^{peak}$ (h) & $t^{full}$ (h) & level ratio $h/h^{nom}$", t2)

    pA = peaks(simulate(PULSE_SINGLE["v01"])); pB = peaks(simulate(PULSE_SINGLE["v02"])); pE = peaks(sims["E"])
    t3 = []; gaps = []
    for n in CITIES:
        naive = pA[n][0] + pB[n][0] - STEADY[n]; simv = pE[n][0]
        t3.append(r"%s %s & %.1f & %.1f & %.1f & %.1f (at %s h)\\" % (LAB[n], NAMES[n], pA[n][0], pB[n][0], naive, simv, fmt_h(pE[n][1] - TSTAR)))
        gaps.append((naive - simv, n))
    write_table("gen_p3_stag.tex", "lrrrr", r"city & peak, $v_{01}$ pulse only & peak, $v_{02}$ pulse only & naive sum $q_A+q_B-q^{nom}$ & simulated peak (ML/s)", t3)
    gmax, gn = max(gaps)
    mac("PthreeStagGap", "%.1f" % gmax); mac("PthreeStagGapCity", NAMES[gn])

    t4 = []
    for k in "ABCDE":
        ak = ffr(SCEN[k]); ts = min(t0s(SCEN[k]).values())
        mgs = {n: ak[n] - (ts + DELTA_C + evac(n) + DSAFE) for n in CITIES if ak[n] < math.inf}
        wn = min(mgs, key=mgs.get)
        t4.append(r"%s & %s & %s %s (%d) & %s\\" % (k, SCEN_TXT[k], LAB[wn], NAMES[wn], mgs[wn], "all $\ge0$" if min(mgs.values()) >= 0 else "some $<0$"))
    write_table("gen_p3_scen.tex", "llll", r" & scenario & binding city (margin, min) & status", t4)

    rn = relay_and_none(sC); t5 = []
    for n in CITIES:
        c, r, z = rn[n]
        t5.append(r"%s %s & %d & %d & %d\\" % (LAB[n], NAMES[n], c, r, z))
    write_table("gen_p3_modes.tex", "lrrr", r"city & raw slack, central warning (min) & raw slack, relay only (min) & raw slack, no warning (min)", t5)

    rC = regime_peaks(sims["R_C"]); tR = []
    for n in CITIES:
        qa, ka = pk[n]; qb, kb = rC[n]
        tR.append(r"%s %s & %d & %.1f & %s & %.1f & %s & %.1f\\" % (LAB[n], NAMES[n], an[n] - TSTAR, qa, fmt_h(ka - TSTAR), qb, fmt_h(kb - TSTAR), qb - FLOODSS[n]))
    write_table("gen_p3_regime.tex", "lrrrrrr", r"city & FFR (min after $t^\star$) & peak $q$, two-class (ML/s) & at (h) & peak $q$, all water at 4 m/s (ML/s) & at (h) & overshoot above flood-steady (ML/s)", tR)
    ovC = max((rC[n][0] - FLOODSS[n], n) for n in CITIES)
    rD = regime_peaks(sims["R_D"]); ovD = max((rD[n][0] - FLOODSS[n], n) for n in CITIES)
    mac("PthreeRegOvC", "%.1f" % ovC[0]); mac("PthreeRegOvCCity", NAMES[ovC[1]])
    mac("PthreeRegOvCPct", "%.0f" % (100.0 * ovC[0] / FLOODSS[ovC[1]]))
    mac("PthreeRegOvD", "%.1f" % ovD[0]); mac("PthreeRegOvDCity", NAMES[ovD[1]])
    rE = sims["R_E"]; faE = regime_first_arrival(rE); gapmin = []
    for n in CITIES:
        qn = rE["q"][n][faE[n]:]; gapmin.append((float(qn.min()) / STEADY[n], n))
    gm = min(gapmin)
    mac("PthreeRegGapMin", "%.2f" % gm[0]); mac("PthreeRegGapCity", NAMES[gm[1]])
    mac("PthreeRegSameFFR", "yes")

    j = "v3"; sorted_raw = sorted((raw[n], n) for n in CITIES)
    mac("PthreeJoshFFR", str(an[j] - TSTAR)); mac("PthreeJoshRaw", str(raw[j])); mac("PthreeJoshMargin", str(mar[j]))
    mac("PthreeJoshEvac", str(evac(j))); mac("PthreeJoshTau", str(an[j] - TSTAR))
    mac("PthreeNextRaw", str(sorted_raw[1][0])); mac("PthreeNextCity", NAMES[sorted_raw[1][1]])
    mac("PthreeMinMarginOther", str(min(mar[n] for n in CITIES if n != j))); mac("PthreeMaxMargin", str(max(mar.values())))
    mac("PthreeRawJoshExtra", str(raw[j])); mac("PthreeMarginExtra", str(mar[j]))
    mac("PthreeMaxSpread", "%.0f" % max(100 * (hi - lo) / d for (d, lo, hi, art) in EDGES.values() if not art))
    ms = max(((100 * (hi - lo) / d), e) for e, (d, lo, hi, art) in EDGES.items() if not art)[1]
    mac("PthreeMaxSpreadEdge", "$%s\\to %s$" % (LAB[ms[0]][1:-1], LAB[ms[1]][1:-1]))
    mac("PthreeGmapsN", str(len(GMAPS)))
    mac("PthreeDepth", "6"); mac("PthreeNChecks", str(len(LOG)))
    mac("PthreeBalErr", "%.0e" % max(sims["_bal"], 1e-16))
    mac("PthreeTotalInj", "%.0f" % (60.0 * sum(float(np.sum(sims["C"]["Qc"][(n, "F")])) for n in SOURCES)))
    for n in CITIES:
        mac("PthreeFFR" + {"v4": "Four", "v5": "Five", "v3": "Three", "v2": "Two", "v1": "One", "v6": "Six", "v7": "Seven", "v8": "Eight"}[n], str(an[n] - TSTAR))
    mac("PthreeRelayHar", "%d" % rn["v7"][1]); mac("PthreeRelayRish", "%d" % rn["v6"][1])
    mac("PthreeRelayJosh", "%d" % rn["v3"][1])

    aA = ffr(SCEN["A"]); mac("PthreeAJoshFFR", str(aA["v3"] - 60)); mac("PthreeAOneFFR", str(aA["v1"] - 60)); mac("PthreeATwoFFR", str(aA["v2"] - 60))
    mac("PthreeAViaFive", "%d+%d+%d" % (tauF[("v01", "v4")], tauF[("v4", "v5")], tauF[("v5", "v3")]))

    mac("PthreeTclear", str(TSTAR + DELTA_C + evac(j) - TSTAR))
    shortest = [(tauF[e], e) for e in tauF]
    open(os.path.join(REP, "gen_p3_macros.tex"), "w").write("\n".join(MAC) + "\n")
    with open(os.path.join(REP, "gen_p3_checks.txt"), "w") as fh:
        for nme, det in LOG: fh.write("[PASS] %s %s\n" % (nme, det))

    sD = sims["D"]; fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for n in ["v4", "v3", "v1", "v6", "v8"]:
        ax[0].plot((sD["t"][sD["H"]:]) / 60.0, after0(sD, n), label="%s %s" % (n, NAMES[n]))
    ax[0].set_xlabel("time since start (h)"); ax[0].set_ylabel("inflow $q_i$ (ML/s)")
    ax[0].set_title("Staggered bursts: $v_{01}$ at 1 h, $v_{02}$ at 4 h", fontsize=9); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    x = np.arange(len(CITIES)); w = 0.27
    ax[1].bar(x - w, [rn[n][0] / 60 for n in CITIES], w, label="central warning ($\\delta_c=5$ min)")
    ax[1].bar(x, [rn[n][1] / 60 for n in CITIES], w, label="city-to-city relay (10 min/hop)")
    ax[1].bar(x + w, [rn[n][2] / 60 for n in CITIES], w, label="no warning (Nepal)")
    ax[1].axhline(0, color="k", lw=.8); ax[1].axhline(DSAFE / 60, color="r", ls="--", lw=.8, label="$\\Delta_{safe}$ = 1 h")
    ax[1].set_xticks(x); ax[1].set_xticklabels(CITIES, fontsize=8); ax[1].set_ylabel("raw slack (h)"); ax[1].set_title("Raw slack = FFR - (alert + EvacuateTime), both bursts at 1 h", fontsize=9)
    ax[1].legend(fontsize=6); ax[1].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "p3_sim.pdf")); plt.close()
    ev = evacuation(sC); fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, n in zip(ax, ["v3", "v4", "v7"]):
        tt = np.arange(0, an[n] + 200)
        a.plot((tt - TSTAR) / 60.0, ev[n]["Prem"][tt] * 1e3, label=r"$P^{rem}_i[t]$ (thousand people)")
        a.axvline((an[n] - TSTAR) / 60.0, color="b", label="FFR (flood arrives)")
        a.axvline((an[n] - DSAFE - TSTAR) / 60.0, color="r", ls="--", label=r"FFR $-\Delta_{safe}$")
        a.set_title("%s (%s)" % (NAMES[n], n), fontsize=9); a.set_xlabel("hours after $t^\\star$"); a.grid(alpha=.3)
    ax[0].set_ylabel("people still in zone (thousand)"); ax[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "p3_evac.pdf")); plt.close()
    return dict(an=an, raw=raw, mar=mar, pk=pk)

if __name__ == "__main__":
    print("edge delays (min): flood / normal")
    for e in EDGES: print(" ", e, tauF[e], tauN[e], "ARTIFICIAL" if EDGES[e][3] else "")
    sims = run_checks()
    res = write_all(sims)
    print("\nBASELINE (both bursts at t*=60): city  FFR-t*  evac  raw  margin  q_peak  t_peak(min after t*)")
    for n in CITIES:
        print("  %-4s %-12s %5d %4d %5d %5d   %5.1f %6d" % (n, NAMES[n], res["an"][n] - TSTAR, evac(n), res["raw"][n], res["mar"][n], res["pk"][n][0], res["pk"][n][1] - TSTAR))
