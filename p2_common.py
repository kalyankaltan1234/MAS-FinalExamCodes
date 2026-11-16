import numpy as np, networkx as nx, hashlib

N, D, M, SIGMA = 20, 20, 500, 0.1
N_CHORDS = 20

class Problem:
    pass

def make_problem(seed):
    r = np.random.default_rng(seed)
    P = Problem(); P.seed = seed
    P.theta_star = r.standard_normal(D)
    P.A = r.standard_normal((N, M, D))
    P.b = np.einsum('imd,d->im', P.A, P.theta_star) + SIGMA * r.standard_normal((N, M))
    P.H = np.einsum('imd,ime->ide', P.A, P.A) / M
    P.c = np.einsum('imd,im->id', P.A, P.b) / M
    P.bb = np.einsum('im,im->i', P.b, P.b) / (2 * M)
    P.Hbar, P.cbar, P.bbbar = P.H.mean(0), P.c.mean(0), P.bb.mean()
    P.theta_ls = np.linalg.solve(P.Hbar, P.cbar)

    G = nx.cycle_graph(N)
    while G.number_of_edges() < N + N_CHORDS:
        u, v = r.choice(N, 2, replace=False); G.add_edge(int(u), int(v))
    assert nx.is_connected(G) and G.number_of_edges() == 40
    P.G = G; P.adj = nx.to_numpy_array(G, nodelist=range(N)); P.deg = P.adj.sum(1)
    P.nb = [list(G[i]) for i in range(N)]

    W = np.zeros((N, N))
    for i, j in G.edges(): W[i, j] = W[j, i] = 1.0 / (1 + max(P.deg[i], P.deg[j]))
    W += np.diag(1 - W.sum(1))
    assert np.allclose(W, W.T), "W not symmetric"
    assert np.allclose(W.sum(1), 1) and np.allclose(W.sum(0), 1), "W not doubly stochastic"
    assert (W >= -1e-15).all(), "W has negative entries"
    pattern = (W > 0); assert np.array_equal(pattern, (P.adj + np.eye(N)) > 0), "exam condition (4) violated"
    P.W = W
    ev = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]; P.lam2 = ev[1]
    P.theta0 = np.zeros((N, D))
    P.L = max(np.linalg.eigvalsh(h).max() for h in P.H)
    P.mu = np.linalg.eigvalsh(P.Hbar).min()
    rt = np.random.default_rng(seed + 10 ** 6); th = rt.standard_normal(D)
    direct = np.mean([np.sum((P.A[i] @ th - P.b[i]) ** 2) / (2 * M) for i in range(N)])
    assert np.isclose(direct, 0.5 * th @ P.Hbar @ th - P.cbar @ th + P.bbbar), "closed-form F differs from (1/N) sum_i f_i"
    gdirect = np.mean([P.A[i].T @ (P.A[i] @ th - P.b[i]) / M for i in range(N)], axis=0)
    assert np.allclose(gdirect, P.Hbar @ th - P.cbar), "closed-form grad F differs from (1/N) sum_i grad f_i"
    P.hash = hashlib.sha256(P.A.tobytes() + P.b.tobytes() + P.W.tobytes() + P.theta0.tobytes()).hexdigest()
    return P

def F(P, th):
    return 0.5 * th @ P.Hbar @ th - P.cbar @ th + P.bbbar

def _metrics(P, X):
    tb = X.mean(0)
    return (np.sqrt(np.mean(np.sum((X - tb) ** 2, 1))), np.linalg.norm(tb - P.theta_star), F(P, tb), np.linalg.norm(tb - P.theta_ls))

def _finish(h, K):
    h = np.array(h);
    if len(h) < K + 1: h = np.vstack([h, np.full((K + 1 - len(h), 4), np.inf)])
    return h

def run_dgd(P, alpha, K, diminishing=False):
    X = P.theta0.copy(); h = [_metrics(P, X)]
    for k in range(K):
        g = np.einsum('ide,ie->id', P.H, X) - P.c
        step = alpha / np.sqrt(k + 1) if diminishing else alpha
        X = P.W @ X - step * g
        h.append(_metrics(P, X))
        if not np.isfinite(h[-1][3]) or h[-1][3] > 1e8: break
    return _finish(h, K)

def run_gt(P, alpha, K):
    X = P.theta0.copy(); g = np.einsum('ide,ie->id', P.H, X) - P.c; Y = g.copy()
    h = [_metrics(P, X)]
    for k in range(K):
        Xn = P.W @ X - alpha * Y
        gn = np.einsum('ide,ie->id', P.H, Xn) - P.c
        Y = P.W @ Y + gn - g
        X, g = Xn, gn
        h.append(_metrics(P, X))
        if not np.isfinite(h[-1][3]) or h[-1][3] > 1e8: break
    return _finish(h, K)

def admm_setup(P, rho):

    return np.linalg.inv(P.H + (2 * rho * P.deg)[:, None, None] * np.eye(D)[None])

def admm_theta_update(P, rho, X, Pdual, inv):
    rhs = P.c - Pdual + rho * (P.deg[:, None] * X + P.adj @ X)
    return np.einsum('ide,ie->id', inv, rhs)

def run_admm(P, rho, K, check=False):
    inv = admm_setup(P, rho); X = P.theta0.copy(); Pd = np.zeros((N, D)); h = [_metrics(P, X)]
    for k in range(K):
        Xn = admm_theta_update(P, rho, X, Pd, inv)
        if check and k < 5:
            grad = (np.einsum('ide,ie->id', P.H, Xn) - P.c) + Pd + 2 * rho * (P.deg[:, None] * Xn - 0.5 * (P.deg[:, None] * X + P.adj @ X))
            assert np.abs(grad).max() < 1e-9, "closed-form update is not the stationary point of the local subproblem"
        Pd = Pd + rho * (P.deg[:, None] * Xn - P.adj @ Xn)
        X = Xn; h.append(_metrics(P, X))
    return _finish(h, K)

def first_below(err, tol):
    i = np.where(err < tol)[0]; return int(i[0]) if len(i) else None
