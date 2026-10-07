"""Escenario 2: predecir DENTRO de zonas conocidas (interpolación).

Se usan solo las viviendas de los bloques de entrenamiento (el test del Entregable 1 no interviene).
Se ocultan EDIFICIOS completos al azar (5 folds, semilla 42, sin buffer): el modelo ve a los vecinos de la vivienda,
pero nunca a otras unidades de su mismo edificio (ICC = 0.991, ocultar filas sueltas sería fuga).
Hiperparámetros FIJOS en los que eligió la validación espacial (no se re-buscan para este escenario).
En este escenario el estrato de los vecinos (de edificios con estrato conocido) es información legítima;
en el escenario de zonas nuevas sería fuga.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree

from pathlib import Path as _Path
import os as _os
AQUI = _Path(__file__).resolve().parent
VEC = str(AQUI.parent / "datos" / "vecindad")   # resultados (JSON) e intermedios (pickles, no se suben)
_os.makedirs(VEC, exist_ok=True)
sys.path.insert(0, str(AQUI))
import replica_sin_sklearn as B  # noqa: E402
from variables_vecindad import CELDA_KM, _disco, _suma_en_disco, calcular_vecindad  # noqa: E402

C = B.C
CLASES = B.CLASES
NTHREAD = 3
t00 = time.time()
dec, df, con = B.construir_datos()
tt = con[con["particion"] == "train"].reset_index(drop=True)
print(f"viviendas de bloques de entrenamiento: {len(tt):,} (esperado 218 870)", flush=True)
V = calcular_vecindad(con)
V["id_fila"] = con["id_fila"].to_numpy()
VALL = [c for c in V.columns if c.startswith("vec")]
tt = tt.merge(V, on="id_fila", how="left")
y = tt[C.OBJETIVO].to_numpy()
vm = dec["variables_modelo"]["valor"]
num_log = dec["num_log"]["valor"]
cats = vm["categoricas"]
NB = vm["numericas"] + vm["espaciales"]
q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
mf = dec["onehot_min_frecuencia"]["valor"]

# --- folds por edificio (aleatorios, deterministas)
edif = tt["npn_edificio"].astype(str).to_numpy()
ued, inv = np.unique(edif, return_inverse=True)
fold_ed = np.random.default_rng(C.SEED).permutation(len(ued)) % 5
fold = fold_ed[inv]
print("viviendas por fold:", np.bincount(fold).tolist(), flush=True)


def estrato_vecinos(xy, yv, edif_, conocido, radio_km):
    """Estrato medio de las viviendas CONOCIDAS de otros edificios a <= radio_km (rejilla de 50 m, como vecindad.py)."""
    x, yy = xy[:, 0], xy[:, 1]
    pad = int(np.ceil(radio_km / CELDA_KM)) + 2
    cx = np.floor((x - x.min()) / CELDA_KM).astype(int) + pad
    cy = np.floor((yy - yy.min()) / CELDA_KM).astype(int) + pad
    forma = (cy.max() + pad + 1, cx.max() + pad + 1)
    n_, s_ = np.zeros(forma), np.zeros(forma)
    np.add.at(n_, (cy[conocido], cx[conocido]), 1.0)
    np.add.at(s_, (cy[conocido], cx[conocido]), yv[conocido].astype(float))
    S = _suma_en_disco([n_, s_], _disco(radio_km / CELDA_KM))
    tot = pd.DataFrame({"n": conocido.astype(float), "s": np.where(conocido, yv, 0.0)}).groupby(edif_).transform("sum")
    n = np.clip(S[0][cy, cx] - tot["n"].to_numpy(), 0, None)
    s = S[1][cy, cx] - tot["s"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0.5, s / n, np.nan)


# --- XGBoost con la API básica (scikit-learn está bloqueado en este equipo); mismos hiperparámetros que avanzados.py
def xgb_fit(X, yy, n_rounds, max_depth, w=None):
    cl = np.unique(yy)
    if len(cl) == 1:
        return ("constante", cl, None)
    mapa = np.searchsorted(cl, yy)
    p = {"max_depth": max_depth, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
         "tree_method": "hist", "nthread": NTHREAD, "seed": C.SEED, "verbosity": 0}
    if len(cl) == 2:
        p["objective"] = "binary:logistic"
    else:
        p.update(objective="multi:softprob", num_class=len(cl))
    return ("xgb", cl, xgb.train(p, xgb.DMatrix(X, label=mapa, weight=w), n_rounds))


def xgb_proba(m, X):
    tipo, cl, booster = m
    out = np.zeros((len(X), 6))
    if tipo == "constante":
        out[:, int(cl[0]) - 1] = 1.0
        return out
    p = booster.predict(xgb.DMatrix(X))
    if p.ndim == 1:
        p = np.c_[1 - p, p]
    for j, k in enumerate(cl):
        out[:, int(k) - 1] = p[:, j]
    return out


def geoxgb(Xtr, ytr, xytr, Xva, xyva, p_global_va, frac=0.15, alpha=0.75, tam_ancla_km=2.0, min_ancla=300):
    """Réplica de GeoXGBEstrato (avanzados.py) con los hiperparámetros elegidos (ventana 15 %, alpha 0.75)."""
    celda = np.floor(xytr / tam_ancla_km).astype(int)
    _, invc, cuenta = np.unique(celda, axis=0, return_inverse=True, return_counts=True)
    invc = invc.ravel()
    anclas = np.array([xytr[invc == g].mean(0) for g in np.where(cuenta >= min_ancla)[0]])
    arbol = cKDTree(xytr)
    k = min(int(frac * len(xytr)), len(xytr))
    h = arbol.query(anclas, k=[k])[0].ravel()
    num = np.zeros((len(Xva), 6))
    den = np.zeros(len(Xva))
    for a, hh in zip(anclas, h):
        idx = np.array(arbol.query_ball_point(a, r=hh))
        u = np.linalg.norm(xytr[idx] - a, axis=1) / hh
        w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        sel = w > 0
        m = xgb_fit(Xtr[idx][sel], ytr[idx][sel], 150, 4, w[sel])
        uv = np.linalg.norm(xyva - a, axis=1) / hh
        wv = np.where(uv < 1, (1 - uv ** 2) ** 2, 0.0)
        sv = wv > 0
        if sv.any():
            num[sv] += wv[sv, None] * xgb_proba(m, Xva[sv])
            den[sv] += wv[sv]
    cub = den > 0
    p_local = p_global_va.copy()
    p_local[cub] = num[cub] / den[cub, None]
    return alpha * p_global_va + (1 - alpha) * p_local, float(cub.mean()), len(anclas)


nombres = ["Moda por zona (2 km)", "Vecinos más cercanos (k = 15)", "Logística B", "Logística B + vecindad",
           "Logística B + vecindad + estrato de vecinos", "XGBoost", "XGBoost geográfico"]
oof = {n: np.zeros(len(tt), int) for n in nombres}
f1f = {n: [] for n in nombres}
info = {"cobertura_geo": [], "anclas": []}
xy_all = tt[["x_km", "y_km"]].to_numpy()
for f in range(5):
    t0 = time.time()
    itr, iva = np.where(fold != f)[0], np.where(fold == f)[0]
    Ttr, Tva = tt.iloc[itr], tt.iloc[iva]
    ytr, yva = y[itr], y[iva]
    pr = {}
    # 1) moda por zona de 2 km
    zt = pd.Series(list(zip(np.floor(Ttr.x_km / 2).astype(int), np.floor(Ttr.y_km / 2).astype(int))))
    moda = pd.Series(ytr).groupby(zt.values).agg(lambda s: s.value_counts().idxmax())
    zv = list(zip(np.floor(Tva.x_km / 2).astype(int), np.floor(Tva.y_km / 2).astype(int)))
    glob = pd.Series(ytr).value_counts().idxmax()
    pr[nombres[0]] = np.array([moda.get(z, glob) for z in zv])
    # 2) vecinos más cercanos (voto de las 15 viviendas conocidas más cercanas)
    _, nn_idx = cKDTree(xy_all[itr]).query(xy_all[iva], k=15)
    votos = np.zeros((len(iva), 6))
    for j in range(15):
        np.add.at(votos, (np.arange(len(iva)), ytr[nn_idx[:, j]] - 1), 1)
    pr[nombres[1]] = np.argmax(votos, 1) + 1
    # estrato de los vecinos conocidos (solo los edificios del entrenamiento del fold)
    conocido = fold != f
    tt["est300"] = estrato_vecinos(xy_all, y, edif, conocido, 0.3)
    tt["est1000"] = estrato_vecinos(xy_all, y, edif, conocido, 1.0)
    Ttr, Tva = tt.iloc[itr], tt.iloc[iva]
    # 3-5) logísticas (C = 0.01, sin pesos)
    nl = [c for c in NB if c in num_log]
    for nombre, extra in [(nombres[2], []), (nombres[3], VALL), (nombres[4], VALL + ["est300", "est1000"])]:
        nn = [c for c in NB if c not in num_log] + extra
        cols = nl + nn + cats
        prep = B.Prep(nl, nn, cats, q_inf, q_sup, mf).fit(Ttr[cols])
        m = B.LogisticaMultinomial(C=0.01).fit(prep.transform(Ttr[cols]), ytr)
        pr[nombre] = m.predict(prep.transform(Tva[cols]))
    # 6-7) XGBoost y XGBoost geográfico (variables B, mismo preprocesador)
    nn = [c for c in NB if c not in num_log]
    cols = nl + nn + cats
    prep = B.Prep(nl, nn, cats, q_inf, q_sup, mf).fit(Ttr[cols])
    Xtr, Xva = prep.transform(Ttr[cols]), prep.transform(Tva[cols])
    mg = xgb_fit(Xtr, ytr, 300, 4)
    pg = xgb_proba(mg, Xva)
    pr[nombres[5]] = np.argmax(pg, 1) + 1
    pgeo, cob, nanc = geoxgb(Xtr, ytr, xy_all[itr], Xva, xy_all[iva], pg)
    pr[nombres[6]] = np.argmax(pgeo, 1) + 1
    info["cobertura_geo"].append(cob)
    info["anclas"].append(nanc)
    for n in nombres:
        oof[n][iva] = pr[n]
        f1f[n].append(B.f1_macro(yva, pr[n]))
    print(f"fold {f + 1}/5 ({time.time() - t0:.0f}s) | " + " | ".join(f"{n[:22]}: {f1f[n][-1]:.3f}" for n in nombres)
          + f" | cobertura geo {cob:.2f}, anclas {nanc}", flush=True)

acc = lambda a, p: float(np.mean(a == p))  # noqa: E731
mae = lambda a, p: float(np.mean(np.abs(a - p)))  # noqa: E731
bl = tt["bloque"].to_numpy()
out = {"n": int(len(tt)), "info": info, "modelos": {}, "comparaciones": {}}
print("\n=== Zonas conocidas: resultados agregados (predicciones fuera de muestra de las 218 870 viviendas) ===")
for n in nombres:
    p = oof[n]
    out["modelos"][n] = {"F1_macro": B.f1_macro(y, p), "accuracy": acc(y, p), "MAE": mae(y, p), "kappa": B.kappa_cuadratico(y, p),
                         "F1_por_fold": f1f[n], "recall": {int(k): float(np.mean(p[y == k] == k)) for k in CLASES}}
    r = out["modelos"][n]
    print(f"{n:45s} F1 {r['F1_macro']:.4f} | acc {r['accuracy']:.4f} | MAE {r['MAE']:.4f} | kappa {r['kappa']:.4f} | F1 folds {np.round(f1f[n], 3).tolist()}")
for a, b in [("XGBoost geográfico", "XGBoost"), ("XGBoost", "Logística B"), ("XGBoost geográfico", "Logística B"),
             ("Logística B + vecindad", "Logística B"), ("Logística B + vecindad + estrato de vecinos", "Logística B"),
             ("Logística B", "Moda por zona (2 km)"), ("XGBoost geográfico", "Vecinos más cercanos (k = 15)"),
             ("Logística B + vecindad + estrato de vecinos", "Vecinos más cercanos (k = 15)")]:
    c = {"F1": B.bootstrap_diferencia(y, oof[a], oof[b], bl, B.f1_macro), "MAE": B.bootstrap_diferencia(y, oof[a], oof[b], bl, mae)}
    out["comparaciones"][f"{a} − {b}"] = c
    print(f"{a} − {b}: F1 Δ={c['F1']['diferencia']:+.4f} IC95 [{c['F1']['IC95_inf']:+.4f}, {c['F1']['IC95_sup']:+.4f}] | "
          f"MAE Δ={c['MAE']['diferencia']:+.4f} IC95 [{c['MAE']['IC95_inf']:+.4f}, {c['MAE']['IC95_sup']:+.4f}]")
json.dump(out, open(_os.path.join(VEC, "zonas_conocidas.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"guardado zonas_conocidas.json ({time.time() - t00:.0f}s en total)")
