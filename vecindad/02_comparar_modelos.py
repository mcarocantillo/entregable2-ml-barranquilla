"""Compara la logística B con la logística B + variables de vecindad física, con el protocolo del Entregable 1.

Especificación fijada ANTES de ver resultados:
  - Modelo principal del experimento: "B + vecindad (300 m y 1 km)" = B más las 12 variables de vecindad.
  - Mismos folds, misma búsqueda (C y pesos de clase), misma regla 1-SE, mismo test y mismo buffer que B.
  - Criterio de mejora (el del capítulo 4 del Entregable 2): reemplaza a B solo si su F1 de CV la supera en más de un
    error estándar de la diferencia pareada por fold.
Las demás variantes son análisis de sensibilidad (exploratorios) y se rotulan así.
"""
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from pathlib import Path as _Path
import os as _os
AQUI = _Path(__file__).resolve().parent
VEC = str(AQUI.parent / "datos" / "vecindad")   # resultados (JSON) e intermedios (pickles, no se suben)
_os.makedirs(VEC, exist_ok=True)
sys.path.insert(0, str(AQUI))
import replica_sin_sklearn as B  # noqa: E402
from variables_vecindad import calcular_vecindad  # noqa: E402

C = B.C
d = pickle.load(open(_os.path.join(VEC, "datos.pkl"), "rb"))
con, train, test, folds = d["con"], d["train"], d["test"], d["folds"]
dec = C.leer_decisiones()
vm = dec["variables_modelo"]["valor"]
num_log = dec["num_log"]["valor"]
cats = vm["categoricas"]
NB = vm["numericas"] + vm["espaciales"]
NA = vm["numericas"]
split = lambda cols: ([c for c in cols if c in num_log], [c for c in cols if c not in num_log])  # noqa: E731

# --- Variables de vecindad: se calculan UNA vez sobre todas las viviendas con coordenadas (solo atributos físicos).
t0 = time.time()
V, diag = calcular_vecindad(con, devolver_diagnostico=True)
print(f"vecindad calculada en {time.time() - t0:.0f}s | diagnóstico: {diag}", flush=True)
V300 = [c for c in V.columns if c.startswith("vec300_")]
V1000 = [c for c in V.columns if c.startswith("vec1000_")]
VALL = V300 + V1000
Vid = V.copy()
Vid["id_fila"] = con["id_fila"].to_numpy()
tr = train.merge(Vid, on="id_fila", how="left")
te = test.merge(Vid, on="id_fila", how="left")
assert (tr["id_fila"].to_numpy() == train["id_fila"].to_numpy()).all() and (te["id_fila"].to_numpy() == test["id_fila"].to_numpy()).all()
y_tr, y_te = tr[C.OBJETIVO].to_numpy(), te[C.OBJETIVO].to_numpy()
print("resumen de las variables de vecindad en train:\n", tr[VALL].describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string(), flush=True)

nl_b, nn_b = split(NB)
nl_a, nn_a = split(NA)
SPECS = {
    "B": (nl_b, nn_b),
    "B + vecindad (300 m y 1 km)": (nl_b, nn_b + VALL),          # PRINCIPAL
    "B + vecindad (solo 300 m)": (nl_b, nn_b + V300),             # exploratorio
    "B + vecindad (solo 1 km)": (nl_b, nn_b + V1000),             # exploratorio
    "A": (nl_a, nn_a),                                            # exploratorio (sin coordenadas)
    "A + vecindad (sin coordenadas)": (nl_a, nn_a + VALL),        # exploratorio
}
SALIDA = {"config": {"radios_km": [0.3, 1.0], "celda_km": 0.05, "variables_vecindad": VALL, "n_train": int(len(tr)),
                     "n_test": int(len(te)), "diagnostico_vecindad": diag, "principal": "B + vecindad (300 m y 1 km)"},
          "modelos": {}}
pred_test, proba_test = {}, {}
for nombre, (nl, nn) in SPECS.items():
    t0 = time.time()
    print(f"\n=== {nombre} ===", flush=True)
    res = B.cv_grid(tr, y_tr, folds, nl, nn, cats, dec, verbose=True)
    el = B.regla_1se(res)
    f1cv = res[el]
    pred, proba, m, prep = B.ajustar_y_evaluar(tr, y_tr, te, y_te, nl, nn, cats, dec, *el)
    met = B.metricas(y_te, pred, proba)
    mor, rmed = B.moran_residuos(te, proba)
    rec = {int(k): float(np.mean(pred[y_te == k] == k)) for k in B.CLASES}
    pred_test[nombre], proba_test[nombre] = pred, proba
    names = B.nombres_columnas(prep)
    coef = {n: [float(v) for v in m.W_[:, i]] for i, n in enumerate(names)}
    SALIDA["modelos"][nombre] = {
        "elegido": {"C": el[0], "class_weight": el[1]},
        "grid_cv": {f"C={g[0]}|cw={g[1]}": [float(x) for x in v] for g, v in res.items()},
        "cv_f1_por_fold": [float(x) for x in f1cv], "cv_f1_media": float(np.mean(f1cv)), "cv_f1_desv": float(np.std(f1cv)),
        "test": met, "recall_test": rec, "moran_residuos": mor, "residuo_medio": rmed, "coeficientes": coef,
        "n_columnas": len(names), "iteraciones": int(m.n_iter_)}
    print(f"  elegido {el} | CV F1 {np.mean(f1cv):.4f} ± {np.std(f1cv):.4f} | folds {np.round(f1cv, 3)}")
    print(f"  TEST {{{', '.join(f'{k}: {v:.4f}' for k, v in met.items())}}}")
    print(f"  recall por estrato {rec} | Moran residuos {mor:.3f} | residuo medio {rmed:+.3f} | {time.time() - t0:.0f}s", flush=True)

# --- Comparaciones frente a B
ref = "B"
cmp_ = {}
bloques = te["bloque"].to_numpy()
for nombre in SPECS:
    if nombre == ref:
        continue
    a, b = np.array(SALIDA["modelos"][nombre]["cv_f1_por_fold"]), np.array(SALIDA["modelos"]["B"]["cv_f1_por_fold"])
    dif = a - b
    se = dif.std(ddof=1) / np.sqrt(len(dif))
    cmp_[nombre] = {
        "cv_dif_por_fold": [float(x) for x in dif], "cv_dif_media": float(dif.mean()), "cv_se_pareado": float(se),
        "reemplaza_a_B": bool(dif.mean() > se),
        "test_f1": B.bootstrap_diferencia(y_te, pred_test[nombre], pred_test[ref], bloques, B.f1_macro),
        "test_mae": B.bootstrap_diferencia(y_te, pred_test[nombre], pred_test[ref], bloques, lambda y, p: float(np.mean(np.abs(y - p)))),
        "test_kappa": B.bootstrap_diferencia(y_te, pred_test[nombre], pred_test[ref], bloques, B.kappa_cuadratico),
    }
    c = cmp_[nombre]
    print(f"\n{nombre} − B: CV Δ={c['cv_dif_media']:+.4f} (SE pareado {c['cv_se_pareado']:.4f}, por fold {np.round(c['cv_dif_por_fold'], 3)}) "
          f"→ reemplaza: {c['reemplaza_a_B']}")
    for k in ("test_f1", "test_mae", "test_kappa"):
        r = c[k]
        print(f"   {k}: Δ={r['diferencia']:+.4f} IC95 [{r['IC95_inf']:+.4f}, {r['IC95_sup']:+.4f}] p={r['p_bootstrap']:.3f}")
SALIDA["comparaciones_vs_B"] = cmp_
json.dump(SALIDA, open(_os.path.join(VEC, "vecindad.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pickle.dump({"pred": pred_test, "proba": proba_test, "y": y_te, "bloque": bloques, "x_km": te["x_km"].to_numpy(),
             "y_km": te["y_km"].to_numpy(), "edificio": te["npn_edificio"].to_numpy()}, open(_os.path.join(VEC, "pred_test.pkl"), "wb"))
print("\nguardado vecindad.json")
