import sys
from pathlib import Path as _Path
import os as _os
AQUI = _Path(__file__).resolve().parent
VEC = str(AQUI.parent / "datos" / "vecindad")   # resultados (JSON) e intermedios (pickles, no se suben)
_os.makedirs(VEC, exist_ok=True)
sys.path.insert(0, str(AQUI))
import pickle, sys, numpy as np, pandas as pd
import replica_sin_sklearn as B
from variables_vecindad import calcular_vecindad
C = B.C
d = pickle.load(open(_os.path.join(VEC, "datos.pkl"), "rb")); con, train, test = d["con"], d["train"], d["test"]
dec = C.leer_decisiones(); vm = dec["variables_modelo"]["valor"]; num_log = dec["num_log"]["valor"]; cats = vm["categoricas"]
NB = vm["numericas"] + vm["espaciales"]
nl = [c for c in NB if c in num_log]; nn = [c for c in NB if c not in num_log]
y_tr, y_te = train[C.OBJETIVO].to_numpy(), test[C.OBJETIVO].to_numpy()
bl = test["bloque"].to_numpy()
def con_vec(pool_tr, pool_te):
    """pool_tr/pool_te: DataFrames desde los que se calculan las variables de vecindad de train y de test."""
    Vtr = calcular_vecindad(pool_tr); Vtr["id_fila"] = pool_tr["id_fila"].to_numpy()
    Vte = calcular_vecindad(pool_te); Vte["id_fila"] = pool_te["id_fila"].to_numpy()
    return train.merge(Vtr, on="id_fila", how="left"), test.merge(Vte, on="id_fila", how="left"), [c for c in Vtr.columns if c.startswith("vec")]
# (a) base: vecindad con TODAS las viviendas con coordenadas (lo que ya se corrió)
V = calcular_vecindad(con); V["id_fila"] = con["id_fila"].to_numpy(); VALL = [c for c in V.columns if c.startswith("vec")]
trA, teA = train.merge(V, on="id_fila", how="left"), test.merge(V, on="id_fila", how="left")
# (b) ESTRICTA: train solo con viviendas de train_total (sin test) y test solo con viviendas de test
tt = con[con["particion"] == "train"]; ts = con[con["particion"] == "test"]
trB, teB, _ = con_vec(tt, ts)
pB, _, _, _ = B.ajustar_y_evaluar(train, y_tr, test, y_te, nl, nn, cats, dec, 0.01, None)
pA, _, _, _ = B.ajustar_y_evaluar(trA, y_tr, teA, y_te, nl, nn + VALL, cats, dec, 0.01, None)
pS, prS, mS, _ = B.ajustar_y_evaluar(trB, y_tr, teB, y_te, nl, nn + VALL, cats, dec, 0.01, None)
acc = lambda y, p: float(np.mean(y == p)); mae = lambda y, p: float(np.mean(np.abs(y - p)))
for nombre, p in [("B", pB), ("B+vec (vecindad con todas las viviendas)", pA), ("B+vec ESTRICTA (solo dentro de cada partición)", pS)]:
    print(f"{nombre:50s} F1 {B.f1_macro(y_te, p):.4f} | acc {acc(y_te, p):.4f} | MAE {mae(y_te, p):.4f} | kappa {B.kappa_cuadratico(y_te, p):.4f}")
for nombre, f in [("f1_macro", B.f1_macro), ("MAE", mae)]:
    r = B.bootstrap_diferencia(y_te, pS, pB, bl, f)
    print(f"ESTRICTA − B  {nombre:9s} Δ={r['diferencia']:+.4f} IC95 [{r['IC95_inf']:+.4f}, {r['IC95_sup']:+.4f}] p={r['p_bootstrap']:.3f}")
print("\nPor bloque (acc B -> completa -> estricta):")
for b in np.unique(bl):
    m = bl == b
    print(f"  {b:>5} n={m.sum():>6}: {acc(y_te[m], pB[m]):.2f} -> {acc(y_te[m], pA[m]):.2f} -> {acc(y_te[m], pS[m]):.2f}")
print("\nDejando fuera un bloque a la vez: Δ F1 macro (B+vec − B) [completa | estricta]")
for b in np.unique(bl):
    m = bl != b
    print(f"  sin bloque {b:>5}: {B.f1_macro(y_te[m], pA[m]) - B.f1_macro(y_te[m], pB[m]):+.4f} | {B.f1_macro(y_te[m], pS[m]) - B.f1_macro(y_te[m], pB[m]):+.4f}")

import json  # noqa: E402
salida = {"modelos": {}, "estricta_vs_B": {}, "por_bloque": [], "sin_un_bloque": []}
for nombre, p in [("B", pB), ("B + vecindad", pA), ("B + vecindad (estricta)", pS)]:
    salida["modelos"][nombre] = {"f1_macro": B.f1_macro(y_te, p), "accuracy": acc(y_te, p), "MAE": mae(y_te, p),
                                 "kappa": B.kappa_cuadratico(y_te, p)}
for nombre, f in [("f1_macro", B.f1_macro), ("MAE", mae)]:
    salida["estricta_vs_B"][nombre] = B.bootstrap_diferencia(y_te, pS, pB, bl, f)
for b in np.unique(bl):
    m = bl == b
    salida["por_bloque"].append({"bloque": int(b), "n": int(m.sum()), "acc_B": acc(y_te[m], pB[m]),
                                 "acc_B_vecindad": acc(y_te[m], pA[m]), "acc_estricta": acc(y_te[m], pS[m])})
    m = bl != b
    salida["sin_un_bloque"].append({"bloque_omitido": int(b),
                                    "dF1_completa": B.f1_macro(y_te[m], pA[m]) - B.f1_macro(y_te[m], pB[m]),
                                    "dF1_estricta": B.f1_macro(y_te[m], pS[m]) - B.f1_macro(y_te[m], pB[m])})
json.dump(salida, open(_os.path.join(VEC, "variante_estricta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("guardado variante_estricta.json")
