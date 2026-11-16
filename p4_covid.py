import numpy as np
from dataclasses import dataclass
from p4_epi import Epi, trace

S, E, I, R = 0, 1, 2, 3
N, T = 300, 120
bE, bI, sig, gam = 0.04, 0.10, 0.25, 0.12
deg_mean, N0 = 6, 5
Imax, Isafe, Twait, Tiso, Ttest = 45, 15, 7, 10, 2

@dataclass
class Policy:
    name: str
    control: bool = True
    screen: float = 0.02
    delay: str = "same"
    L: int = 5
    iso_exposed: bool = False
    sens: float = 1.0
    refresh: bool = True

def graphs_for(seed):
    rg = np.random.default_rng(seed)
    x0 = np.zeros(N, int); x0[rg.choice(N, N0, replace=False)] = I
    gs = []
    for _ in range(T):
        A = np.triu(rg.random((N, N)) < deg_mean / (N - 1), 1); gs.append(A | A.T)
    return x0, gs

def run(pol, seed, graphs=None):
    x0, gs = graphs if graphs is not None else graphs_for(seed)
    rd = np.random.default_rng(10_000 + seed); rt = np.random.default_rng(20_000 + seed)
    x = x0.copy(); epi = Epi(N); iso_left = np.zeros(N, int); sched = np.full(N, -1)
    hist = []
    iso_log = np.zeros((T, N), bool); KI_snap = np.zeros((T, N), bool); unc_log = np.zeros((T, N), bool); test_log = np.zeros((T, N), bool)
    ts = {k: np.zeros(T, int) for k in ("NE", "NI", "CKI", "CKNI", "CKE", "CUNK", "TEST", "TESTKE", "CKET", "ISO", "NEW_POS", "NOTIF", "INF")}
    n_fact = n_graph = n_pform = 0
    for k in range(T):
        A = gs[k]
        unc = epi.unc(); unc_log[k] = unc
        KE0 = epi.KE.copy(); KI_snap[k] = epi.KI
        infected = (x == E) | (x == I)
        ts["INF"][k] = infected.sum(); ts["NE"][k] = (x == E).sum(); ts["NI"][k] = (x == I).sum()
        ts["CKI"][k] = epi.KI.sum(); ts["CKNI"][k] = epi.KNI.sum(); ts["CKE"][k] = epi.KE.sum(); ts["CUNK"][k] = epi.unknown().sum()
        iso = iso_left > 0
        if pol.control and pol.iso_exposed: iso = iso | unc
        iso_log[k] = iso
        Au = A & ~iso[:, None] & ~iso[None, :]
        assert (Au == Au.T).all() and not (Au & ~A).any() and not Au[iso].any() and not Au[:, iso].any()
        n_graph += 1
        iso_left = np.maximum(iso_left - 1, 0)
        u_scr, u_del, u_sens = rt.random(N), rt.random(N), rt.random(N)
        test = np.zeros(N, bool)
        if pol.control:
            sched[~unc] = -1
            new = unc & (sched < 0)
            d = {"same": np.zeros(N, int), "two": np.full(N, Ttest), "unif": np.floor(u_del * 3).astype(int)}[pol.delay]
            sched[new] = k + d[new]
            test = (sched == k) | (u_scr < pol.screen); sched[test] = -1
        test_log[k] = test
        ts["CKET"][k] = (test & KE0).sum()
        pos = test & infected & (u_sens < pol.sens)
        neg = test & ~pos
        newKI = pos & ~epi.KI
        epi.record_test(pos, neg)
        if pol.control:
            notif = trace(epi, newKI, hist); ts["NOTIF"][k] = notif.sum()
            iso_left[newKI] = Tiso
            hist.append(Au); hist[:] = hist[-pol.L:]
            if pol.refresh: iso_left[epi.KI] = np.maximum(iso_left[epi.KI], Tiso)
        epi.check_factivity(infected, strict=(pol.sens >= 1.0)); n_fact += 1
        ts["TEST"][k] = test.sum(); ts["TESTKE"][k] = (test & unc).sum()
        ts["ISO"][k] = iso.sum(); ts["NEW_POS"][k] = newKI.sum()

        u_inf, u_e, u_r = rd.random(N), rd.random(N), rd.random(N)
        nE = (Au & (x == E)[None, :]).sum(1); nI = (Au & (x == I)[None, :]).sum(1)
        p = 1 - (1 - bE) ** nE * (1 - bI) ** nI
        for i in np.where(x == S)[0][:3]:
            q = 1.0
            for j in np.where(Au[i])[0]: q *= 1 - (bE if x[j] == E else bI if x[j] == I else 0.0)
            assert abs(p[i] - (1 - q)) < 1e-12; n_pform += 1
        new_inf = (x == S) & (u_inf < p); e2i = (x == E) & (u_e < sig); i2r = (x == I) & (u_r < gam)
        x[new_inf] = E; x[e2i] = I; x[i2r] = R
        epi.recover(i2r); epi.end_of_step()
    NI = ts["NI"]
    rho1 = (Imax - NI).min()
    rho2 = max((Isafe - NI[t:t + Twait]).min() for t in range(T - Twait + 1))
    ok3 = tot3 = 0
    for k_, i_ in zip(*np.where(KI_snap)):
        if k_ + Tiso < T: tot3 += 1; ok3 += iso_log[k_ + 1:k_ + Tiso + 1, i_].all()
    rising = unc_log & ~np.vstack([np.zeros((1, N), bool), unc_log[:-1]])
    ok4 = tot4 = 0
    for k_, i_ in zip(*np.where(rising)):
        if k_ + Ttest < T: tot4 += 1; ok4 += test_log[k_:k_ + Ttest + 1, i_].any()
    det = np.where(ts["NEW_POS"] > 0)[0]; fd = int(det[0]) if len(det) else -1
    return dict(ts=ts, peak=int(NI.max()), peak_day=int(NI.argmax()), attack=float((x != S).mean()),
                rho1=int(rho1), rho2=int(rho2), sat1=bool(rho1 > 0), sat2=bool(rho2 > 0),
                ok3=int(ok3), tot3=int(tot3), ok4=int(ok4), tot4=int(tot4),
                first_detect=fd, inf_at_detect=int(ts["NE"][fd] + ts["NI"][fd]) if fd >= 0 else -1,
                tests=int(ts["TEST"].sum()), iso_days=int(ts["ISO"].sum()), iso_max=int(ts["ISO"].max()), notif=int(ts["NOTIF"].sum()),
                n_fact=n_fact, n_graph=n_graph, n_pform=n_pform)

POLICIES = [
    Policy("no control", control=False),
    Policy("main: same-day test, screen 2%, L=5"),
    Policy("two-day test (exactly day 2)", delay="two"),
    Policy("two-day test (uniform 0,1,2)", delay="unif"),
    Policy("screen 0%", screen=0.0), Policy("screen 1%", screen=0.01),
    Policy("screen 5%", screen=0.05), Policy("screen 10%", screen=0.10),
    Policy("tracing L=3", L=3), Policy("tracing L=7", L=7),
    Policy("stronger A: two-day + quarantine pending test", delay="two", iso_exposed=True),
    Policy("stronger B: screen 10%, L=7", screen=0.10, L=7),
    Policy("stronger C: B + two-day + quarantine", screen=0.10, L=7, delay="two", iso_exposed=True),
    Policy("ablation: isolate 10 d after first positive only", refresh=False),
    Policy("extension: test sensitivity 0.9", sens=0.9),
]

def run_seed(args):
    seed, pols = args
    g = graphs_for(seed)
    return seed, {p.name: run(p, seed, g) for p in pols}

if __name__ == "__main__":
    import time; t0 = time.time()
    r = run(POLICIES[1], 0); print("main policy seed 0:", {k: r[k] for k in ("peak", "peak_day", "attack", "rho1", "rho2", "ok3", "tot3", "ok4", "tot4", "first_detect")}, "%.1fs" % (time.time() - t0))
