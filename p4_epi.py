import numpy as np

class Epi:
    def __init__(self, n, Tvalid=1):
        self.n = n; self.Tvalid = Tvalid
        self.KI = np.zeros(n, bool); self.KNI = np.zeros(n, bool)
        self.KE = np.zeros(n, bool); self.RE = np.zeros(n, bool)
        self.neg_age = np.zeros(n, int)

    def unknown(self):  return ~self.KI & ~self.KNI
    def unc(self):      return self.KE & ~self.KI & ~self.KNI

    def record_test(self, pos, neg):
        self.KI[pos] = True;  self.KNI[pos] = False; self.KE[pos] = False; self.RE[pos] = True
        self.KNI[neg] = True; self.KI[neg] = False;  self.KE[neg] = False; self.RE[neg] = True
        self.neg_age[neg] = 0
    def notify(self, mask):
        self.KE[mask] = True; self.KNI[mask] = False; self.RE[mask] = False
    def recover(self, mask):
        self.KI[mask] = False
    def end_of_step(self):
        self.neg_age[self.KNI] += 1
        self.KNI[self.neg_age >= self.Tvalid] = False

    def check_factivity(self, infected, strict=True):
        assert not (self.KI & ~infected).any(), "K Inf without Inf: factivity violated"
        if strict: assert not (self.KNI & infected).any(), "K not-Inf while infected: factivity violated"

def trace(epi, newKI, hist):
    if not newKI.any() or not hist: return np.zeros(epi.n, bool)
    notif = np.zeros(epi.n, bool)
    for Ah in hist: notif |= Ah[newKI].any(axis=0)
    notif &= ~newKI
    epi.notify(notif)
    return notif

def run_unit_tests(verbose=True):
    res = []
    def check(name, cond):
        res.append((name, bool(cond)))
        if verbose: print(("PASS  " if cond else "FAIL  ") + name)
    def fact(ep, infected, strict):
        try: ep.check_factivity(infected, strict); return True
        except AssertionError: return False
    x = np.array([0, 1, 2, 3])
    inf = np.isin(x, [1, 2])
    e = Epi(4)
    check("initially nobody knows anything: not KI and not KNI for all, no exposure, not 'uncertain-exposed'",
          (~e.KI).all() and (~e.KNI).all() and e.unknown().all() and not e.KE.any() and not e.unc().any())

    e.record_test(pos=np.array([0, 1, 1, 0], bool), neg=np.zeros(4, bool))
    check("positive test: K_i Inf true, K_i not-Inf false, exposure resolved (agents 1,2)",
          e.KI[[1, 2]].all() and not e.KNI[[1, 2]].any() and e.RE[[1, 2]].all() and not e.KE[[1, 2]].any())
    check("factivity K_i Inf -> Inf holds after positive tests", fact(e, inf, strict=False))

    e.record_test(pos=np.zeros(4, bool), neg=np.array([1, 0, 0, 1], bool))
    check("negative test: K_i not-Inf true and K_i Inf false (agents 0,3); agent 0: not KI AND KNI",
          e.KNI[[0, 3]].all() and not e.KI[[0, 3]].any())
    check("factivity K_i not-Inf -> not Inf holds after negative tests", fact(e, inf, strict=True))
    e2 = Epi(4)
    check("untested agent 1 (truly E): not KI true but KNI false -> status unknown (Inf and not K Inf, spec 2a)",
          (not e2.KI[1]) and (not e2.KNI[1]) and e2.unknown()[1] and inf[1])

    e2.notify(np.array([0, 1, 0, 0], bool))
    check("exposure notification without test: K Exp true and uncertain-exposed true; not KI, not KNI",
          e2.KE[1] and e2.unc()[1] and (not e2.KI[1]) and (not e2.KNI[1]))
    check("agent 1 is truly infected (E) yet does NOT know it: Inf and not K Inf (Kripke example of the answer)",
          inf[1] and not e2.KI[1])

    e2.record_test(pos=np.zeros(4, bool), neg=np.array([0, 1, 0, 0], bool))
    check("negative test after exposure: exposure resolved (KE false, RE true), KNI true, no longer uncertain-exposed",
          (not e2.KE[1]) and e2.RE[1] and e2.KNI[1] and not e2.unc()[1])
    e2.notify(np.array([0, 1, 0, 0], bool))
    check("re-exposure after negative test: old K not-Inf is cleared, KE true again, uncertain-exposed again",
          (not e2.KNI[1]) and e2.KE[1] and (not e2.RE[1]) and e2.unc()[1])

    e3 = Epi(4); e3.record_test(np.array([0, 0, 1, 0], bool), np.zeros(4, bool))
    x3 = x.copy(); x3[2] = 3; inf3 = np.isin(x3, [1, 2])
    e3.notify(np.array([0, 0, 1, 0], bool)); e3.recover(np.array([0, 0, 1, 0], bool))
    check("recovery clears K_i Inf (agent 2), factivity holds again, and an unresolved exposure notification is kept (agent does not know it recovered)",
          (not e3.KI[2]) and fact(e3, inf3, strict=True) and e3.KE[2])

    e4 = Epi(2, Tvalid=1); e4.record_test(np.zeros(2, bool), np.array([1, 0], bool))
    ok_today = e4.KNI[0]; e4.end_of_step()
    check("K not-Inf is valid on the day of the test and expires at the end of the step (agent may be infected later)",
          ok_today and not e4.KNI[0] and e4.unknown()[0])

    n = 4
    def path_edges(edges):
        A = np.zeros((n, n), bool)
        for a, b in edges: A[a, b] = A[b, a] = True
        return A
    hist = [path_edges([(0, 3)]), path_edges([(0, 1)]), path_edges([(1, 2)])]
    L = 2; hist = hist[-L:]
    today = path_edges([(0, 2)])
    e5 = Epi(n); newKI = np.array([1, 0, 0, 0], bool)
    notif = trace(e5, newKI, hist)
    check("tracing uses exactly the previous L=2 graphs: agent 1 notified (contact in day k-2), agent 3 not (contact older than L), agent 2 not (contact only with agent 1; today's graph unused)",
          list(notif) == [False, True, False, False] and list(e5.KE) == [False, True, False, False])

    A = np.ones((n, n), bool) & ~np.eye(n, dtype=bool)
    iso = np.array([0, 1, 0, 0], bool); Au = A & ~iso[:, None] & ~iso[None, :]
    D = np.diag((~iso).astype(float)); Au2 = (D @ A.astype(float) @ D) > 0
    check("controlled graph: A^u = D A D removes row and column of the isolating agent and keeps symmetry",
          (Au == Au2).all() and not Au[1].any() and not Au[:, 1].any() and (Au == Au.T).all())

    bE, bI = 0.04, 0.10
    xs = np.array([0, 1, 1, 2, 2, 2]); nbr = [1, 2, 3, 4, 5]
    p_formula = 1 - (1 - bE) ** 2 * (1 - bI) ** 3
    p_explicit = 1 - np.prod([1 - (bE if xs[j] == 1 else bI if xs[j] == 2 else 0.0) for j in nbr])
    check("p_infected: closed form 1-(1-bE)^nE (1-bI)^nI equals the explicit product over neighbours", abs(p_formula - p_explicit) < 1e-12)
    ok = all(c for _, c in res)
    if verbose: print("UNIT TESTS: %d/%d passed" % (sum(c for _, c in res), len(res)))
    return res

if __name__ == "__main__":
    r = run_unit_tests(); assert all(c for _, c in r)
