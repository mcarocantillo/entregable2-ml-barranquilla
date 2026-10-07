"""¿Dónde ayudan y dónde perjudican las variables de vecindad? (análisis exploratorio, posterior a los resultados)

Se usan las predicciones fuera de muestra de la CV de 15 folds (entrenamiento) y las de test.
El estrato de los vecinos se usa SOLO para diagnosticar (nunca entra a un modelo):
  atipicidad = estrato de la vivienda − estrato medio de las viviendas de OTROS edificios a ≤ 300 m.
Hipótesis: la vecindad "suaviza" hacia el estrato típico del entorno; ayuda donde la vivienda se parece a su
entorno y perjudica donde es distinta.
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pathlib import Path as _Path
import os as _os
AQUI = _Path(__file__).resolve().parent
VEC = str(AQUI.parent / "datos" / "vecindad")   # resultados (JSON) e intermedios (pickles, no se suben)
_os.makedirs(VEC, exist_ok=True)
sys.path.insert(0, str(AQUI))
import replica_sin_sklearn as B  # noqa: E402
from variables_vecindad import CELDA_KM, _disco, _suma_en_disco, calcular_vecindad  # noqa: E402

C = B.C
d = pickle.load(open(_os.path.join(VEC, "datos.pkl"), "rb"))
con, test = d["con"], d["test"]
o = pickle.load(open(_os.path.join(VEC, "oof15.pkl"), "rb"))
pt = pickle.load(open(_os.path.join(VEC, "pred_test.pkl"), "rb"))


def estrato_vecinos(df, radio_km=0.3, celda_km=CELDA_KM):
    """Estrato medio (y su desviación) de las viviendas de OTROS edificios a <= radio_km. Solo diagnóstico."""
    x, y = df["x_km"].to_numpy(float), df["y_km"].to_numpy(float)
    pad = int(np.ceil(radio_km / celda_km)) + 2
    cx = np.floor((x - x.min()) / celda_km).astype(int) + pad
    cy = np.floor((y - y.min()) / celda_km).astype(int) + pad
    forma = (cy.max() + pad + 1, cx.max() + pad + 1)
    e = df[C.OBJETIVO].to_numpy(float)
    capas = {"n": np.ones(len(df)), "s": e, "s2": e ** 2}
    grids = []
    for k in capas:
        g = np.zeros(forma)
        np.add.at(g, (cy, cx), capas[k])
        grids.append(g)
    S = _suma_en_disco(grids, _disco(radio_km / celda_km))
    tot = pd.DataFrame(capas).groupby(df["npn_edificio"].astype(str).to_numpy()).transform("sum")
    n = np.clip(S[0][cy, cx] - tot["n"].to_numpy(), 0, None)
    s = S[1][cy, cx] - tot["s"].to_numpy()
    s2 = S[2][cy, cx] - tot["s2"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        media = np.where(n > 0.5, s / n, np.nan)
        desv = np.sqrt(np.clip(np.where(n > 0.5, s2 / n - media ** 2, np.nan), 0, None))
    return pd.DataFrame({"vec_estrato_medio": media, "vec_estrato_desv": desv, "id_fila": df["id_fila"].to_numpy()})


diag = estrato_vecinos(con)
V = calcular_vecindad(con)
V["id_fila"] = con["id_fila"].to_numpy()

# --- tabla por vivienda: entrenamiento (OOF de 15 folds) y test
tr = o["datos"].merge(diag, on="id_fila", how="left")
tr["pB"], tr["pV"], tr["conj"] = o["oof"]["B"], o["oof"]["B+vec"], "train (CV 15 folds)"
te = test[["id_fila", "bloque", "x_km", "y_km", C.OBJETIVO, "condicion_predio", "uso"]].merge(diag, on="id_fila", how="left")
te = te.merge(V[["id_fila", "vec300_apto", "vec300_informal"]], on="id_fila", how="left")
te["pB"], te["pV"], te["conj"] = pt["pred"]["B"], pt["pred"]["B + vecindad (300 m y 1 km)"], "test"
assert (te[C.OBJETIVO].to_numpy() == pt["y"]).all()
T = pd.concat([tr, te], ignore_index=True)
T["y"] = T[C.OBJETIVO].astype(int)
T["aciertoB"], T["aciertoV"] = (T.pB == T.y), (T.pV == T.y)
T["errB"], T["errV"] = (T.pB - T.y).abs(), (T.pV - T.y).abs()
T["atip"] = T.y - T.vec_estrato_medio

R = {}


def resumen(g):
    return pd.Series({"n": len(g), "acc_B": g.aciertoB.mean(), "acc_Bvec": g.aciertoV.mean(),
                      "d_acc": g.aciertoV.mean() - g.aciertoB.mean(), "MAE_B": g.errB.mean(), "MAE_Bvec": g.errV.mean(),
                      "d_MAE": g.errV.mean() - g.errB.mean()})


# 1) Según qué tan distinta es la vivienda de su entorno (atipicidad)
cortes = [-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf]
etq = ["≤ −1 (más baja que su entorno)", "−1 a −0.5", "−0.5 a 0.5 (parecida a su entorno)", "0.5 a 1", "≥ 1 (más alta que su entorno)"]
T["grupo_atip"] = pd.cut(T.atip, cortes, labels=etq)
print("\n=== 1. Según la atipicidad de la vivienda frente a su entorno (300 m) ===")
for conj, g in T.groupby("conj"):
    t = g.groupby("grupo_atip", observed=True).apply(resumen, include_groups=False)
    t["% viviendas"] = 100 * t["n"] / t["n"].sum()
    print(f"\n[{conj}]\n", t[["n", "% viviendas", "acc_B", "acc_Bvec", "d_acc", "MAE_B", "MAE_Bvec", "d_MAE"]].round(3).to_string())
    R[f"atipicidad|{conj}"] = t.reset_index().to_dict(orient="list")

# 2) Según la heterogeneidad del entorno (desviación del estrato de los vecinos)
T["grupo_het"] = pd.cut(T.vec_estrato_desv, [-0.01, 0.3, 0.6, 1.0, np.inf],
                        labels=["muy homogéneo (< 0.3)", "0.3–0.6", "0.6–1.0", "muy mezclado (> 1.0)"])
print("\n=== 2. Según qué tan mezclado está el entorno (desv. del estrato a 300 m) ===")
for conj, g in T.groupby("conj"):
    t = g.groupby("grupo_het", observed=True).apply(resumen, include_groups=False)
    t["% viviendas"] = 100 * t["n"] / t["n"].sum()
    print(f"\n[{conj}]\n", t[["n", "% viviendas", "acc_B", "acc_Bvec", "d_acc", "MAE_B", "MAE_Bvec", "d_MAE"]].round(3).to_string())
    R[f"heterogeneidad|{conj}"] = t.reset_index().to_dict(orient="list")

# 3) Según el estrato real
print("\n=== 3. Recall por estrato real ===")
for conj, g in T.groupby("conj"):
    t = g.groupby("y").apply(lambda h: pd.Series({"n": len(h), "recall_B": h.aciertoB.mean(), "recall_Bvec": h.aciertoV.mean(),
                                                   "atip_media": h.atip.mean()}), include_groups=False)
    t["d_recall"] = t.recall_Bvec - t.recall_B
    print(f"\n[{conj}]\n", t.round(3).to_string())
    R[f"recall|{conj}"] = t.reset_index().to_dict(orient="list")

# 4) Por bloque (zonas de 2 km), bloques con >= 1000 viviendas
print("\n=== 4. Por bloque de 2 km (≥ 1000 viviendas) ===")
bl = T.groupby(["conj", "bloque"]).apply(lambda g: pd.concat([resumen(g), pd.Series({
    "estrato_medio": g.y.mean(), "estrato_desv": g.y.std(), "het_entorno": g.vec_estrato_desv.mean(),
    "atip_abs": g.atip.abs().mean(), "y_km": g.y_km.mean(), "pct_informal": (g.condicion_predio == "Informal").mean(),
    "pct_apto": g.uso.astype(str).str.contains("Apartamentos").mean()})]), include_groups=False).reset_index()
blg = bl[bl.n >= 1000].sort_values("d_MAE")
print(blg[["conj", "bloque", "n", "estrato_medio", "estrato_desv", "het_entorno", "atip_abs", "pct_informal", "pct_apto", "y_km",
           "acc_B", "acc_Bvec", "d_MAE"]].round(2).to_string(index=False))
print("\nCorrelación de Spearman entre d_MAE del bloque y sus características (bloques ≥ 1000 viviendas, n = %d):" % len(blg))
cors = {}
for c in ["estrato_medio", "estrato_desv", "het_entorno", "atip_abs", "pct_informal", "pct_apto", "y_km"]:
    m = blg[c].notna()   # un bloque de un solo edificio no tiene vecinos: se omite en esa correlación
    r, p = spearmanr(blg.loc[m, c], blg.loc[m, "d_MAE"])
    cors[c] = {"rho": float(r), "p": float(p)}
    print(f"  {c:14s} rho = {r:+.2f} (p = {p:.3f})")
# 5) Hipótesis descartada: ¿la vecindad falla en zonas de apartamentos de estrato bajo (vivienda social VIS/VIP)?
g = con.groupby("bloque").apply(lambda h: pd.Series({
    "pct_VIS_VIP": h.tipo_vivienda.isin(["tv_1", "tv_2"]).mean(),
    "pct_apto_bloque": h.uso.astype(str).str.contains("Apartamentos").mean(),
    "estrato_apto": h.loc[h.uso.astype(str).str.contains("Apartamentos"), C.OBJETIVO].mean()}), include_groups=False)
blg = blg.merge(g, left_on="bloque", right_index=True, how="left")
blg["aptos_bajos"] = (blg["pct_apto_bloque"] > 0.3) & (blg["estrato_apto"] <= 2)
print("\n=== 5. Bloques con muchos apartamentos de estrato bajo (> 30 % apartamentos y estrato medio de ellos <= 2) ===")
t5 = blg.groupby("aptos_bajos").apply(lambda h: pd.Series({"bloques": len(h), "viviendas": h.n.sum(),
      "d_MAE_ponderado": np.average(h.d_MAE, weights=h.n), "bloques_que_empeoran": int((h.d_MAE > 0.02).sum())}), include_groups=False)
print(t5.round(3).to_string())
r5, p5 = spearmanr(blg["pct_VIS_VIP"], blg["d_MAE"])
print(f"Spearman d_MAE vs % VIS/VIP del bloque: rho = {r5:+.2f} (p = {p5:.3f})")
R["hipotesis_vis_vip"] = {"tabla": t5.reset_index().to_dict(orient="list"), "rho_vis_vip": float(r5), "p_vis_vip": float(p5)}
R["bloques"] = blg.to_dict(orient="list")
R["correlaciones_bloques"] = cors
json.dump(R, open(_os.path.join(VEC, "zonas.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("\nguardado zonas.json")
