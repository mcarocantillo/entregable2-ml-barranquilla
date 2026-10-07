"""Verificación rápida: ¿la réplica sin scikit-learn reproduce la logística B del Entregable 1 (C=0.01, sin pesos)?"""
import json
import pickle
import sys
import time

import numpy as np

from pathlib import Path as _Path
import os as _os
AQUI = _Path(__file__).resolve().parent
VEC = str(AQUI.parent / "datos" / "vecindad")   # resultados (JSON) e intermedios (pickles, no se suben)
_os.makedirs(VEC, exist_ok=True)
sys.path.insert(0, str(AQUI))
import replica_sin_sklearn as B  # noqa: E402

C = B.C
t0 = time.time()
dec, df, con = B.construir_datos()
train_total, train, test, folds = B.separar(con, dec)
print(f"train {len(train):,} | test {len(test):,} | {time.time() - t0:.0f}s", flush=True)
pickle.dump(dict(train=train, test=test, folds=folds, con=con), open(_os.path.join(VEC, "datos.pkl"), "wb"))

vm = dec["variables_modelo"]["valor"]
num_log = dec["num_log"]["valor"]
NUMS = vm["numericas"] + vm["espaciales"]
nl = [c for c in NUMS if c in num_log]
nn = [c for c in NUMS if c not in num_log]
cats = vm["categoricas"]
y_tr, y_te = train[C.OBJETIVO].to_numpy(), test[C.OBJETIVO].to_numpy()
q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
cols = nl + nn + cats
esperado = [0.4204006168146636, 0.19747681905953918, 0.3787957799013431, 0.2408349959056977, 0.2565400286512281]
obt = []
for i, (itr, iva) in enumerate(folds):
    t0 = time.time()
    prep = B.Prep(nl, nn, cats, q_inf, q_sup, dec["onehot_min_frecuencia"]["valor"]).fit(train.iloc[itr][cols])
    m = B.LogisticaMultinomial(C=0.01).fit(prep.transform(train.iloc[itr][cols]), y_tr[itr])
    f = B.f1_macro(y_tr[iva], m.predict(prep.transform(train.iloc[iva][cols])))
    obt.append(f)
    print(f"fold {i + 1}: réplica {f:.4f} | guardado {esperado[i]:.4f} | dif {f - esperado[i]:+.4f} | {m.n_iter_} iter | {time.time() - t0:.0f}s", flush=True)
print("media réplica", round(float(np.mean(obt)), 4), "| guardado 0.2988 | desv réplica", round(float(np.std(obt)), 4), "| guardado 0.0856", flush=True)

t0 = time.time()
pred, proba, m, prep = B.ajustar_y_evaluar(train, y_tr, test, y_te, nl, nn, cats, dec, 0.01, None)
met = B.metricas(y_te, pred, proba)
r = json.load(open(C.CARPETA_PROCESADOS / "resultados_modelo.json", encoding="utf-8"))["resultados_test"]
g = "Logística B_fisicas+ubicacion"
print(f"TEST ({time.time() - t0:.0f}s)")
for k in ["accuracy", "f1_macro", "MAE_ordinal", "kappa_cuadrático", "AUC_OvR_macro", "log_loss"]:
    print(f"  {k:18s} réplica {met[k]:.4f} | guardado {r[k][g]:.4f}")
mor, rm = B.moran_residuos(test, proba)
print(f"Moran de residuos: réplica {mor:.3f} | guardado ≈ 0.70 | residuo medio {rm:+.3f} (guardado -0.14)")
claves = ["accuracy", "f1_macro", "MAE_ordinal", "kappa_cuadrático", "AUC_OvR_macro", "log_loss"]
json.dump({"n_train": int(len(train)), "n_test": int(len(test)),
           "F1_por_fold_replica": [float(x) for x in obt], "F1_por_fold_guardado": esperado,
           "test_replica": {k: float(met[k]) for k in claves}, "test_guardado": {k: float(r[k][g]) for k in claves},
           "moran_residuos_replica": mor, "residuo_medio_replica": rm},
          open(_os.path.join(VEC, "verificacion_replica.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("guardado verificacion_replica.json")
