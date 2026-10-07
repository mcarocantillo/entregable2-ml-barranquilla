import sys
from pathlib import Path as _Path
import os as _os
AQUI = _Path(__file__).resolve().parent
VEC = str(AQUI.parent / "datos" / "vecindad")   # resultados (JSON) e intermedios (pickles, no se suben)
_os.makedirs(VEC, exist_ok=True)
sys.path.insert(0, str(AQUI))
"""Guarda las predicciones fuera de muestra (15 folds, solo entrenamiento) de B y B + vecindad, para analizar por zona."""
import pickle, sys, time
import numpy as np
import replica_sin_sklearn as B
from variables_vecindad import calcular_vecindad
C = B.C; P = B.P
d = pickle.load(open(_os.path.join(VEC, "datos.pkl"), "rb")); con, train = d["con"], d["train"]
dec = C.leer_decisiones(); vm = dec["variables_modelo"]["valor"]; num_log = dec["num_log"]["valor"]; cats = vm["categoricas"]
NB = vm["numericas"] + vm["espaciales"]
V = calcular_vecindad(con); V["id_fila"] = con["id_fila"].to_numpy(); VALL = [c for c in V.columns if c.startswith("vec")]
tr = train.merge(V, on="id_fila", how="left")
y = tr[C.OBJETIVO].to_numpy()
folds = P.folds_espaciales_con_buffer(tr, dec["buffer_km"]["valor"], n_splits=15, verbose=False)
nl = [c for c in NB if c in num_log]; nn = [c for c in NB if c not in num_log]
SPECS = {"B": (nl, nn), "B+vec": (nl, nn + VALL)}
q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]; mf = dec["onehot_min_frecuencia"]["valor"]
oof = {k: np.zeros(len(tr), int) for k in SPECS}; fold_de = np.full(len(tr), -1)
for i, (itr, iva) in enumerate(folds):
    t0 = time.time(); fold_de[iva] = i
    for nombre, (a, b) in SPECS.items():
        cols = a + b + cats
        prep = B.Prep(a, b, cats, q_inf, q_sup, mf).fit(tr.iloc[itr][cols])
        m = B.LogisticaMultinomial(C=0.01).fit(prep.transform(tr.iloc[itr][cols]), y[itr])
        oof[nombre][iva] = m.predict(prep.transform(tr.iloc[iva][cols]))
    print(f"fold {i + 1}/15 listo ({time.time() - t0:.0f}s)", flush=True)
cols_guardar = ["id_fila", "bloque", "x_km", "y_km", "dist_centro_km", C.OBJETIVO, "condicion_predio", "uso", "area_construida", "total_banios", "antiguedad"] + VALL
pickle.dump({"oof": oof, "fold": fold_de, "datos": tr[cols_guardar]}, open(_os.path.join(VEC, "oof15.pkl"), "wb"))
print("guardado oof15.pkl", flush=True)
