import sys
from pathlib import Path as _Path
import os as _os
AQUI = _Path(__file__).resolve().parent
VEC = str(AQUI.parent / "datos" / "vecindad")   # resultados (JSON) e intermedios (pickles, no se suben)
_os.makedirs(VEC, exist_ok=True)
sys.path.insert(0, str(AQUI))
"""Validación cruzada espacial más fina (15 folds de bloques con buffer de 1 km, dentro del entrenamiento; el test no interviene).
Modelos con hiperparámetros FIJOS en lo que eligió la búsqueda anterior (C=0.01, sin pesos), el mismo para todos.
Se acumulan las predicciones fuera de muestra de TODAS las viviendas de entrenamiento y se comparan por pares con bootstrap de bloques."""
import json, pickle, sys, time
import numpy as np
import replica_sin_sklearn as B
from variables_vecindad import calcular_vecindad
C = B.C; P = B.P
d = pickle.load(open(_os.path.join(VEC, "datos.pkl"), "rb")); con, train, test = d["con"], d["train"], d["test"]
dec = C.leer_decisiones(); vm = dec["variables_modelo"]["valor"]; num_log = dec["num_log"]["valor"]; cats = vm["categoricas"]
NB = vm["numericas"] + vm["espaciales"]; NA = vm["numericas"]
V = calcular_vecindad(con); V["id_fila"] = con["id_fila"].to_numpy(); VALL = [c for c in V.columns if c.startswith("vec")]
tr = train.merge(V, on="id_fila", how="left"); assert (tr["id_fila"].to_numpy() == train["id_fila"].to_numpy()).all()
y = tr[C.OBJETIVO].to_numpy(); bl = tr["bloque"].to_numpy()
K = 15
folds = P.folds_espaciales_con_buffer(tr, dec["buffer_km"]["valor"], n_splits=K, verbose=False)
print(f"{K} folds | tamaños validación: {[len(v) for _, v in folds]} | bloques de train: {len(np.unique(bl))}", flush=True)
sp = lambda cols: ([c for c in cols if c in num_log], [c for c in cols if c not in num_log])
SPECS = {"B": sp(NB), "B + vecindad (300 m y 1 km)": (sp(NB)[0], sp(NB)[1] + VALL), "A + vecindad (sin coordenadas)": (sp(NA)[0], sp(NA)[1] + VALL)}
q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]; mf = dec["onehot_min_frecuencia"]["valor"]
oof = {k: np.zeros(len(tr), int) for k in SPECS}; f1fold = {k: [] for k in SPECS}; cubierto = np.zeros(len(tr), bool)
for i, (itr, iva) in enumerate(folds):
    t0 = time.time(); cubierto[iva] = True
    for nombre, (nl, nn) in SPECS.items():
        cols = nl + nn + cats
        prep = B.Prep(nl, nn, cats, q_inf, q_sup, mf).fit(tr.iloc[itr][cols])
        m = B.LogisticaMultinomial(C=0.01).fit(prep.transform(tr.iloc[itr][cols]), y[itr])
        p = m.predict(prep.transform(tr.iloc[iva][cols])); oof[nombre][iva] = p
        f1fold[nombre].append(B.f1_macro(y[iva], p))
    print(f"fold {i + 1}/{K} listo ({time.time() - t0:.0f}s) | F1 por fold " + " | ".join(f"{k[:12]}: {f1fold[k][-1]:.3f}" for k in SPECS), flush=True)
yy, bb = y[cubierto], bl[cubierto]
acc = lambda a, p: float(np.mean(a == p)); mae = lambda a, p: float(np.mean(np.abs(a - p)))
out = {"K": K, "modelos": {}, "comparaciones_vs_B": {}}
for k in SPECS:
    p = oof[k][cubierto]
    out["modelos"][k] = {"F1_pooled": B.f1_macro(yy, p), "accuracy": acc(yy, p), "MAE": mae(yy, p), "kappa": B.kappa_cuadratico(yy, p),
                         "F1_por_fold": f1fold[k], "F1_fold_media": float(np.mean(f1fold[k])), "F1_fold_desv": float(np.std(f1fold[k]))}
    print(f"{k:34s} F1 agregado {B.f1_macro(yy, p):.4f} | F1 medio por fold {np.mean(f1fold[k]):.4f} ± {np.std(f1fold[k]):.4f} | acc {acc(yy, p):.4f} | MAE {mae(yy, p):.4f}")
for k in list(SPECS)[1:]:
    dif = np.array(f1fold[k]) - np.array(f1fold["B"]); se = dif.std(ddof=1) / np.sqrt(len(dif))
    c = {"dif_F1_por_fold": dif.tolist(), "dif_media": float(dif.mean()), "se": float(se), "folds_mejores": int((dif > 0).sum()), "reemplaza_a_B": bool(dif.mean() > se)}
    for nom, f in [("F1_pooled", B.f1_macro), ("MAE", mae), ("kappa", B.kappa_cuadratico)]:
        c[nom] = B.bootstrap_diferencia(yy, oof[k][cubierto], oof["B"][cubierto], bb, f)
    out["comparaciones_vs_B"][k] = c
    print(f"\n{k} − B: Δ F1 por fold media {c['dif_media']:+.4f} (SE {se:.4f}) | mejor en {c['folds_mejores']}/{K} folds | reemplaza (criterio 1-SE): {c['reemplaza_a_B']}")
    for nom in ("F1_pooled", "MAE", "kappa"):
        r = c[nom]; print(f"   {nom:9s} Δ={r['diferencia']:+.4f} IC95 [{r['IC95_inf']:+.4f}, {r['IC95_sup']:+.4f}] p={r['p_bootstrap']:.3f}")
    print("   Δ F1 por fold:", np.round(dif, 3).tolist())
json.dump(out, open(_os.path.join(VEC, "cv15.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("guardado cv15.json")
