"""Verificación rápida: ¿la réplica sin scikit-learn reproduce la logística B del Entregable 1?

Los valores de referencia se leen de los resultados guardados: hiperparámetros y métricas de test del Entregable 1,
F1 por fold de ``entrenar_avanzados.py`` (scikit-learn) y Moran de residuos de ``preparar_datos.py``.
"""
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
res_e2 = json.load(open(AQUI.parent / "datos" / "resultados.json", encoding="utf-8"))
diag_e2 = json.load(open(AQUI.parent / "datos" / "diagnosticos.json", encoding="utf-8"))
hp_b = res_e2["mejores_hiperparametros"]["B_fisicas+ubicacion"]
C_B = float(hp_b["clf__C"])
CW_B = None if hp_b["clf__class_weight"] in (None, "None") else hp_b["clf__class_weight"]
mor_guardado = diag_e2["moran_residuos"]["B_fisicas+ubicacion"]
# F1 por fold de referencia: los de entrenar_avanzados.py si ya se ejecutó; si no, se calculan aquí con scikit-learn
# (mismo Pipeline del Entregable 1, mismos folds).
esperado = res_e2.get("avanzados", {}).get("logistica_cv_por_fold")
if esperado is None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline
    from src import modelo as M
    esperado = []
    for itr, iva in folds:
        pipe = Pipeline([("prep", M.construir_preprocesador(nl, nn, cats, q_inf=q_inf, q_sup=q_sup,
                                                            min_frecuencia=dec["onehot_min_frecuencia"]["valor"])),
                         ("clf", LogisticRegression(C=C_B, class_weight=CW_B, max_iter=1000))])
        pipe.fit(train.iloc[itr][cols], y_tr[itr])
        esperado.append(f1_score(y_tr[iva], pipe.predict(train.iloc[iva][cols]), average="macro",
                                 labels=C.CLASES, zero_division=0))
esperado = [float(x) for x in esperado]
obt = []
for i, (itr, iva) in enumerate(folds):
    t0 = time.time()
    prep = B.Prep(nl, nn, cats, q_inf, q_sup, dec["onehot_min_frecuencia"]["valor"]).fit(train.iloc[itr][cols])
    m = B.LogisticaMultinomial(C=C_B, class_weight=CW_B).fit(prep.transform(train.iloc[itr][cols]), y_tr[itr])
    f = B.f1_macro(y_tr[iva], m.predict(prep.transform(train.iloc[iva][cols])))
    obt.append(f)
    print(f"fold {i + 1}: réplica {f:.4f} | guardado {esperado[i]:.4f} | dif {f - esperado[i]:+.4f} | {m.n_iter_} iter | {time.time() - t0:.0f}s", flush=True)
print("media réplica", round(float(np.mean(obt)), 4), "| guardado", round(float(np.mean(esperado)), 4),
      "| desv réplica", round(float(np.std(obt)), 4), "| guardado", round(float(np.std(esperado)), 4), flush=True)

t0 = time.time()
pred, proba, m, prep = B.ajustar_y_evaluar(train, y_tr, test, y_te, nl, nn, cats, dec, C_B, CW_B)
met = B.metricas(y_te, pred, proba)
r = json.load(open(C.CARPETA_PROCESADOS / "resultados_modelo.json", encoding="utf-8"))["resultados_test"]
g = "Logística B_fisicas+ubicacion"
print(f"TEST ({time.time() - t0:.0f}s)")
for k in ["accuracy", "f1_macro", "MAE_ordinal", "kappa_cuadrático", "AUC_OvR_macro", "log_loss"]:
    print(f"  {k:18s} réplica {met[k]:.4f} | guardado {r[k][g]:.4f}")
mor, rm = B.moran_residuos(test, proba)
print(f"Moran de residuos: réplica {mor:.3f} | guardado {mor_guardado['I']:.3f} | residuo medio {rm:+.3f} "
      f"(guardado {mor_guardado['residuo_medio']:+.3f})")
claves = ["accuracy", "f1_macro", "MAE_ordinal", "kappa_cuadrático", "AUC_OvR_macro", "log_loss"]
json.dump({"n_train": int(len(train)), "n_test": int(len(test)),
           "F1_por_fold_replica": [float(x) for x in obt], "F1_por_fold_guardado": esperado,
           "test_replica": {k: float(met[k]) for k in claves}, "test_guardado": {k: float(r[k][g]) for k in claves},
           "moran_residuos_replica": mor, "residuo_medio_replica": rm},
          open(_os.path.join(VEC, "verificacion_replica.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("guardado verificacion_replica.json")
