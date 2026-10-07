"""Reproducción SIN scikit-learn del protocolo del Entregable 1 (solo numpy, pandas y scipy).

Reimplementa, con las mismas fórmulas que ``src/modelo.py``:
  - el preprocesador (winsorizar -> mediana + indicador -> log1p -> estandarizar; one-hot con
    categorías < 1 % agrupadas),
  - la regresión logística multinomial con penalización L2 (objetivo de scikit-learn),
  - las métricas y el bootstrap por bloques,
y usa tal cual las funciones de partición y folds del Entregable 1.
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import logsumexp


# --- En algunos equipos Windows una política de control de aplicaciones bloquea las librerías compiladas de
# scikit-learn. Si no se puede importar, se sustituye SOLO el KDTree que usa src/particion.py por uno
# equivalente de scipy (mismo resultado: distancia al vecino más cercano).
class _KDTree:
    def __init__(self, X):
        self._t = cKDTree(np.asarray(X, float))

    def query(self, X, k=1):
        X = np.asarray(X, float)
        d, i = self._t.query(X, k=k)
        return d.reshape(len(X), -1), i.reshape(len(X), -1)


try:
    import sklearn.neighbors  # noqa: F401
except Exception:
    for _k in [k for k in sys.modules if k == "sklearn" or k.startswith("sklearn.")]:
        del sys.modules[_k]
    _sk = types.ModuleType("sklearn")
    _nb = types.ModuleType("sklearn.neighbors")
    _nb.KDTree = _KDTree
    _sk.neighbors = _nb
    sys.modules["sklearn"] = _sk
    sys.modules["sklearn.neighbors"] = _nb

# Ruta al Entregable 1 (carpeta con src/ y datos/predios_residenciales_barranquilla.csv).
# Por defecto, una carpeta "entregable1_jbook" junto a este repositorio; se puede cambiar con la
# variable de entorno ENTREGABLE1.
E1 = Path(os.environ.get("ENTREGABLE1", Path(__file__).resolve().parents[2] / "entregable1_jbook"))
sys.path.insert(0, str(E1))
from src import config as C  # noqa: E402
from src import limpieza as L  # noqa: E402
from src import particion as P  # noqa: E402

CLASES = np.array(C.CLASES)


# ---------------------------------------------------------------------------
# Datos y partición (idénticos a los capítulos 1, 2 y 10 del Entregable 1)
# ---------------------------------------------------------------------------
def construir_datos():
    dec = C.leer_decisiones()
    df = L.cargar_datos_limpios()
    con = df[df["tiene_coordenadas"]].reset_index(drop=True)
    particion, bloques = P.particion_test_espacial(con, tam_km=C.TAM_BLOQUE_KM, seed=C.SEED)
    con["bloque"] = bloques
    con["particion"] = particion.to_numpy()
    comp = con.groupby("npn_edificio")["particion"].nunique()
    partidos = comp[comp > 1].index
    if len(partidos):
        con.loc[con["npn_edificio"].isin(partidos), "particion"] = "test"
    return dec, df, con


def separar(con, dec):
    buffer_km = dec["buffer_km"]["valor"]
    train_total = con[con["particion"] == "train"].reset_index(drop=True)
    test = con[con["particion"] == "test"].reset_index(drop=True)
    mantener = P.mascara_buffer(train_total[["x_km", "y_km"]].to_numpy(), test[["x_km", "y_km"]].to_numpy(), buffer_km)
    train = train_total[mantener].reset_index(drop=True)
    folds = P.folds_espaciales_con_buffer(train, buffer_km, verbose=False)
    return train_total, train, test, folds


# ---------------------------------------------------------------------------
# Preprocesador (réplica de construir_preprocesador)
# ---------------------------------------------------------------------------
class Prep:
    def __init__(self, num_log, num_lin, cats, q_inf=0.001, q_sup=0.999, min_frec=0.01):
        self.num_log, self.num_lin, self.cats = list(num_log), list(num_lin), list(cats)
        self.q_inf, self.q_sup, self.min_frec = q_inf, q_sup, min_frec

    def _num_fit(self, X, usa_log):
        lo = np.nanquantile(X, self.q_inf, axis=0)
        hi = np.nanquantile(X, self.q_sup, axis=0)
        Xc = np.clip(X, lo, hi)
        med = np.nanmedian(Xc, axis=0)
        tiene_na = np.isnan(Xc).any(axis=0)
        Z = self._num_aplicar(X, dict(lo=lo, hi=hi, med=med, tiene_na=tiene_na, usa_log=usa_log, mu=0.0, sd=1.0))
        mu = Z.mean(0)
        sd = Z.std(0)
        sd = np.where(sd < 10 * np.finfo(float).eps * np.maximum(np.abs(mu), 1.0), 1.0, sd)
        return dict(lo=lo, hi=hi, med=med, tiene_na=tiene_na, usa_log=usa_log, mu=mu, sd=sd)

    @staticmethod
    def _num_aplicar(X, p):
        Xc = np.clip(X, p["lo"], p["hi"])
        ind = np.isnan(Xc[:, p["tiene_na"]]).astype(float)
        Xi = np.where(np.isnan(Xc), p["med"], Xc)
        Z = np.hstack([Xi, ind])
        if p["usa_log"]:
            Z = np.log1p(Z)
        return (Z - p["mu"]) / p["sd"]

    @staticmethod
    def _obj(s):
        s = s.astype(object)
        return s.where(s.notna(), "faltante").astype(str)

    def fit(self, df):
        self.p_log = self._num_fit(df[self.num_log].to_numpy(float), True)
        self.p_lin = self._num_fit(df[self.num_lin].to_numpy(float), False)
        self.cat_info = {}
        n = len(df)
        for c in self.cats:
            v = self._obj(df[c])
            cuentas = v.value_counts()
            frecuentes = sorted(cuentas.index[cuentas >= self.min_frec * n])
            hay_infrec = (cuentas < self.min_frec * n).any()
            self.cat_info[c] = (frecuentes, hay_infrec)
        return self

    def transform(self, df):
        partes = [self._num_aplicar(df[self.num_log].to_numpy(float), self.p_log),
                  self._num_aplicar(df[self.num_lin].to_numpy(float), self.p_lin)]
        for c in self.cats:
            frecuentes, hay_infrec = self.cat_info[c]
            v = self._obj(df[c]).to_numpy()
            M = np.stack([(v == k) for k in frecuentes], axis=1).astype(float)
            if hay_infrec:
                M = np.hstack([M, (~np.isin(v, frecuentes))[:, None].astype(float)])
            partes.append(M)
        return np.hstack(partes)


# ---------------------------------------------------------------------------
# Regresión logística multinomial con L2 (mismo objetivo que scikit-learn)
#   min_{W,b}  C * sum_i w_i * CE_i  +  0.5 * ||W||^2     (el intercepto no se penaliza)
# ---------------------------------------------------------------------------
class LogisticaMultinomial:
    def __init__(self, C=1.0, class_weight=None, max_iter=1000, tol=1e-5):
        self.C, self.class_weight, self.max_iter, self.tol = C, class_weight, max_iter, tol

    def fit(self, X, y, theta0=None):
        n, p = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        idx = np.searchsorted(self.classes_, y)
        Y = np.zeros((n, K))
        Y[np.arange(n), idx] = 1.0
        if self.class_weight == "balanced":
            cnt = Y.sum(0)
            sw = (n / (K * cnt))[idx]
        else:
            sw = np.ones(n)
        C_ = self.C

        def f(theta):
            W = theta[:K * p].reshape(K, p)
            b = theta[K * p:]
            Z = X @ W.T + b
            lse = logsumexp(Z, axis=1)
            loss = C_ * np.sum(sw * (lse - Z[np.arange(n), idx])) + 0.5 * np.sum(W * W)
            R = (np.exp(Z - lse[:, None]) - Y) * (C_ * sw)[:, None]
            gW = R.T @ X + W
            gb = R.sum(0)
            return loss, np.concatenate([gW.ravel(), gb])

        theta0 = np.zeros(K * p + K) if theta0 is None else np.asarray(theta0, float).copy()
        # Parada equivalente a la de scikit-learn (tol = 1e-4 sobre el gradiente del objetivo normalizado
        # por C * sum(w)); aquí se usa 1e-5, diez veces más estricta.
        gtol = self.tol * C_ * sw.sum()
        res = minimize(f, theta0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=self.max_iter, maxfun=self.max_iter * 2, ftol=1e-13, gtol=gtol, maxcor=20))
        self.W_ = res.x[:K * p].reshape(K, p)
        self.b_ = res.x[K * p:]
        self.n_iter_ = res.nit
        self.theta_ = res.x
        return self

    def predict_proba(self, X):
        Z = X @ self.W_.T + self.b_
        return np.exp(Z - logsumexp(Z, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(X @ self.W_.T + self.b_, axis=1)]


# ---------------------------------------------------------------------------
# Métricas (mismas definiciones que src/modelo.metricas)
# ---------------------------------------------------------------------------
def f1_macro(y, p, clases=CLASES):
    f = []
    for k in clases:
        tp = np.sum((y == k) & (p == k))
        fp = np.sum((y != k) & (p == k))
        fn = np.sum((y == k) & (p != k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def kappa_cuadratico(y, p, clases=CLASES):
    K = len(clases)
    O = np.zeros((K, K))
    np.add.at(O, (np.searchsorted(clases, y), np.searchsorted(clases, p)), 1)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    i = np.arange(K)
    W = (i[:, None] - i[None, :]) ** 2 / (K - 1) ** 2
    return float(1 - (W * O).sum() / (W * E).sum())


def auc_ovr_macro(y, proba, clases=CLASES):
    from scipy.stats import rankdata
    aucs = []
    for j, k in enumerate(clases):
        pos = y == k
        n1, n0 = pos.sum(), (~pos).sum()
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(proba[:, j])
        aucs.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(aucs))


def metricas(y, pred, proba=None):
    out = {"accuracy": float(np.mean(y == pred)), "f1_macro": f1_macro(y, pred),
           "MAE_ordinal": float(np.mean(np.abs(y - pred))), "accuracy_±1": float(np.mean(np.abs(y - pred) <= 1)),
           "kappa_cuadrático": kappa_cuadratico(y, pred)}
    if proba is not None:
        out["AUC_OvR_macro"] = auc_ovr_macro(y, proba)
        out["log_loss"] = float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), np.searchsorted(CLASES, y)], 1e-12, 1))))
    return out


def bootstrap_diferencia(y, pa, pb, bloques, metrica, B=500, seed=C.SEED):
    """Igual que src/modelo.bootstrap_diferencia (pareado, por bloques, percentil)."""
    rng = np.random.default_rng(seed)
    y, pa, pb, bloques = map(np.asarray, (y, pa, pb, bloques))
    ub, inv = np.unique(bloques, return_inverse=True)
    idx_por_bloque = [np.where(inv == b)[0] for b in range(len(ub))]
    difs = []
    for _ in range(B):
        ii = np.concatenate([idx_por_bloque[b] for b in rng.integers(0, len(ub), len(ub))])
        difs.append(metrica(y[ii], pa[ii]) - metrica(y[ii], pb[ii]))
    difs = np.array(difs)
    obs = metrica(y, pa) - metrica(y, pb)
    p = 2 * min((difs <= 0).mean(), (difs >= 0).mean())
    return {"diferencia": float(obs), "IC95_inf": float(np.percentile(difs, 2.5)),
            "IC95_sup": float(np.percentile(difs, 97.5)), "p_bootstrap": float(max(p, 1 / B))}


# ---------------------------------------------------------------------------
# Búsqueda con CV espacial y regla 1-SE (réplica del capítulo 10)
# ---------------------------------------------------------------------------
GRID = [(c, cw) for cw in (None, "balanced") for c in (0.01, 0.1, 1, 10)]


def cv_grid(train, y_tr, folds, num_log_cols, num_lin_cols, cats, dec, grid=GRID, verbose=True, max_iter=1000):
    """F1 macro por fold de cada (C, class_weight). Para cada class_weight, los C se ajustan en orden creciente
    y cada ajuste parte de la solución del anterior (warm start): el problema es el mismo, solo llega antes."""
    q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
    min_frec = dec["onehot_min_frecuencia"]["valor"]
    cols = num_log_cols + num_lin_cols + cats
    res = {g: [] for g in grid}
    for i, (itr, iva) in enumerate(folds):
        prep = Prep(num_log_cols, num_lin_cols, cats, q_inf, q_sup, min_frec).fit(train.iloc[itr][cols])
        Xtr, Xva = prep.transform(train.iloc[itr][cols]), prep.transform(train.iloc[iva][cols])
        for cw in dict.fromkeys(g[1] for g in grid):
            theta = None
            for g in sorted((g for g in grid if g[1] == cw), key=lambda g: g[0]):
                m = LogisticaMultinomial(C=g[0], class_weight=g[1], max_iter=max_iter).fit(Xtr, y_tr[itr], theta0=theta)
                theta = m.theta_
                res[g].append(f1_macro(y_tr[iva], m.predict(Xva)))
        if verbose:
            print(f"   fold {i + 1}/{len(folds)} listo", flush=True)
    return res


def regla_1se(res):
    claves = list(res)
    media = np.array([np.mean(res[k]) for k in claves])
    se = np.array([np.std(res[k]) for k in claves]) / np.sqrt(len(next(iter(res.values()))))
    i_mejor = int(np.nanargmax(media))
    cand = np.flatnonzero(media >= media[i_mejor] - se[i_mejor])
    c_min = min(claves[i][0] for i in cand)
    fin = [i for i in cand if claves[i][0] == c_min]
    return claves[fin[int(np.argmax(media[fin]))]]


def ajustar_y_evaluar(train, y_tr, test, y_te, num_log_cols, num_lin_cols, cats, dec, C_, cw):
    q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
    cols = num_log_cols + num_lin_cols + cats
    prep = Prep(num_log_cols, num_lin_cols, cats, q_inf, q_sup, dec["onehot_min_frecuencia"]["valor"]).fit(train[cols])
    m = LogisticaMultinomial(C=C_, class_weight=cw).fit(prep.transform(train[cols]), y_tr)
    Xte = prep.transform(test[cols])
    proba = m.predict_proba(Xte)
    pred = m.classes_[np.argmax(proba, axis=1)]
    return pred, proba, m, prep


# ---------------------------------------------------------------------------
# Nombres de las columnas del preprocesador y Moran de los residuos
# ---------------------------------------------------------------------------
def nombres_columnas(prep):
    nombres = []
    for cols, p in ((prep.num_log, prep.p_log), (prep.num_lin, prep.p_lin)):
        nombres += list(cols) + [f"falta_{c}" for c, t in zip(cols, p["tiene_na"]) if t]
    for c in prep.cats:
        frecuentes, hay_infrec = prep.cat_info[c]
        nombres += [f"{c}={k}" for k in frecuentes] + ([f"{c}=infrecuente"] if hay_infrec else [])
    return nombres


def moran_residuos(test, proba, k=8):
    """I de Moran de los residuos ordinales y - E[y], un punto por edificio, k vecinos, W estandarizada por filas
    (mismo procedimiento que preparar_datos.py del Entregable 2)."""
    esperado = proba @ CLASES.astype(float)
    t = test[["npn_edificio", "x_km", "y_km"]].copy()
    t["residuo"] = test[C.OBJETIVO].to_numpy() - esperado
    e = t.groupby("npn_edificio", as_index=False).agg(x_km=("x_km", "mean"), y_km=("y_km", "mean"), residuo=("residuo", "mean"))
    xy = e[["x_km", "y_km"]].to_numpy()
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = idx[:, 1:]
    z = e["residuo"].to_numpy() - e["residuo"].mean()
    lag = z[idx].mean(axis=1)            # W fila-estandarizada: promedio de los k vecinos
    return float((z @ lag) / (z @ z)), float((test[C.OBJETIVO].to_numpy() - esperado).mean())
