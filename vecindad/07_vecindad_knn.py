"""Versión 3: vecindad con grafo KNN y validación kNNDM (sin buffer fijo).

Idea del libro de Modelos Espacio-Temporales (Rubio, cap. 2): construir un grafo de k vecinos más cercanos y usar
el "lag espacial" (promedio de los vecinos, A_norm·x) como variable. Aquí se prueban dos lags:
  - vecindad física KNN: promedio de atributos físicos de los k edificios más cercanos (sin el propio edificio).
    El catastro tiene estos atributos para todas las viviendas, también las que se predicen.
  - estrato de vecinos conocidos: promedio del estrato de los k edificios de ENTRENAMIENTO más cercanos.
    Para que no haya fuga, en las filas de entrenamiento se excluye su propio grupo de kNNDM: así la variable se
    calcula con vecinos a la misma distancia que al predecir (el principio "solo hasta t-1" del libro, sec. 2.5.2).

Dos escenarios, cada uno con sus folds kNNDM (src/vecinos.py del Entregable 1, versión 3):
  - zonas nuevas: la geometría del test del Entregable 1 (zona de 2 km con su entorno conocido). Se evalúa también
    en el test.
  - zonas conocidas: ocultar manzanas completas dentro de train (distancia a la manzana conocida más cercana).
Hiperparámetros FIJOS en los del modelo B de la versión 3 (cap. 14 del Entregable 1).

Uso (desde e2_git):  ENTREGABLE1=../e1_rama ../venv_e2/Scripts/python -u vecindad/07_vecindad_knn.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline

AQUI = Path(__file__).resolve().parent
E1 = Path(os.environ.get("ENTREGABLE1", AQUI.parents[1] / "e1_rama"))
sys.path.insert(0, str(E1))
from src import config as C  # noqa: E402
from src import modelo as M  # noqa: E402
from src import particion as P  # noqa: E402
from src import vecinos as V  # noqa: E402

SALIDA = AQUI.parent / "datos" / "vecindad" / "vecindad_knn.json"
KS_X = [8, 30, 100]       # vecinos para la vecindad física
KS_Y = [8, 30, 100]       # vecinos para el estrato de vecinos conocidos
KS_MORAN = [2, 4, 8, 15, 30, 60, 100, 200]
t00 = time.time()

dec = C.leer_decisiones()
v3 = json.loads((E1 / "datos" / "procesados" / "resultados_v3.json").read_text(encoding="utf-8"))
HP = v3["hiperparametros"]["B v3"]
vm = dec["variables_modelo"]["valor"]
num_log = dec["num_log"]["valor"]
q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
min_frec = dec["onehot_min_frecuencia"]["valor"]
nums, cats = vm["numericas"] + vm["espaciales"], vm["categoricas"]

datos = P.cargar_particion()
train = datos[datos.particion == "train"].reset_index(drop=True)
test = datos[datos.particion == "test"].reset_index(drop=True)
y_tr, y_te = train[C.OBJETIVO].to_numpy(), test[C.OBJETIVO].to_numpy()
print(f"train {len(train):,} | test {len(test):,} | hiperparámetros B v3: {HP}", flush=True)


def xy(df):
    return df[["x_km", "y_km"]].to_numpy()


# ---------------------------------------------------------------------------
# 1) Vecindad física con grafo KNN (no usa el estrato: se calcula con todas las viviendas)
# ---------------------------------------------------------------------------
todo = pd.concat([train, test], ignore_index=True)
attrs = pd.DataFrame({
    "log_area": np.log1p(todo.area_construida),
    "banios": todo.total_banios,
    "habitaciones": todo.total_habitaciones,
    "antiguedad": todo.antiguedad,
    "piso": todo.planta_ubicacion,
    "prop_PH": (todo.condicion_predio == "PH_Unidad_Predial").astype(float),
})
attrs = attrs.fillna(attrs.median())
xy_g, n_g, cod = V.puntos_unicos(xy(todo), todo.npn_edificio.to_numpy())
A_g = attrs.groupby(cod).mean().to_numpy()            # atributos medios por edificio
g_ids = np.arange(len(xy_g))
lagX = {}
for k in KS_X:
    lag_g, dist_g = V.lag_knn(xy_g, xy_g, A_g, k, grupo_destino=g_ids, grupo_fuente=g_ids)
    cols = [f"knn{k}_{c}" for c in attrs.columns]
    df_k = pd.DataFrame(lag_g[cod], columns=cols)
    df_k[f"knn{k}_dist_km"] = dist_g[cod]
    lagX[k] = (df_k.iloc[:len(train)].reset_index(drop=True), df_k.iloc[len(train):].reset_index(drop=True))
# Control: la vecindad por radios fijos del capítulo 5 (300 m y 1 km), para separar el efecto del grafo KNN
# del efecto de la nueva validación.
sys.path.insert(0, str(AQUI))
from variables_vecindad import calcular_vecindad  # noqa: E402
VR = calcular_vecindad(todo).reset_index(drop=True)
lag_radios = (VR.iloc[:len(train)].reset_index(drop=True), VR.iloc[len(train):].reset_index(drop=True))
print(f"vecindad física KNN y por radios calculadas ({time.time() - t00:.0f} s)", flush=True)

# I de Moran del estrato por k (edificios de train): criterio del libro para elegir el grafo
xy_trg, n_trg, cod_tr = V.puntos_unicos(xy(train), train.npn_edificio.to_numpy())
y_trg = pd.Series(y_tr).groupby(cod_tr).mean().to_numpy()
moran_k = V.moran_por_k(xy_trg, y_trg, KS_MORAN)
print(moran_k.to_string(index=False), flush=True)

# ---------------------------------------------------------------------------
# 2) Folds kNNDM de los dos escenarios
# ---------------------------------------------------------------------------
xy_teg, n_teg, _ = V.puntos_unicos(xy(test), test.npn_edificio.to_numpy())
print("kNNDM, zonas nuevas:", flush=True)
kn_nuevas = V.knndm(xy(train), train.npn_edificio.to_numpy(), y_tr, xy_teg, n_teg)
# Zonas conocidas: distancia de cada edificio de train a la manzana conocida más cercana (otra manzana).
# La manzana comodín "0000" se trata edificio por edificio (como en 06_zonas_conocidas.py).
manz = train.npn_manzana.astype(str).to_numpy()
manz = np.where(pd.Series(manz).str.endswith("0000"), train.npn_edificio.astype(str).to_numpy(), manz)
manz_g = pd.Series(manz).groupby(cod_tr).first().to_numpy()
dd, ii = cKDTree(xy_trg).query(xy_trg, k=64)
otra = manz_g[ii] != manz_g[:, None]
d_zc = np.where(otra.any(1), dd[np.arange(len(dd)), otra.argmax(1)], dd[:, -1])
print("kNNDM, zonas conocidas (manzanas ocultas):", flush=True)
kn_conocidas = V.knndm(xy(train), train.npn_edificio.to_numpy(), y_tr, d_pred=d_zc, pesos_pred=n_trg)
ESCENARIOS = {"zonas nuevas": kn_nuevas, "zonas conocidas": kn_conocidas}
for n, kn in ESCENARIOS.items():
    print(f"  {n}: q = {kn['q']}, W = {kn['W']:.3f} km, mediana escenario "
          f"{V.cuantiles_ponderados(kn['d_pred'], kn['pesos_pred'], [0.5])[0]:.3f} km", flush=True)


# ---------------------------------------------------------------------------
# 3) Estrato de vecinos conocidos (depende de qué estratos se conocen: se calcula dentro de cada fold)
# ---------------------------------------------------------------------------
def lag_estrato(df_dest, grupo_dest, df_fuente, y_fuente, grupo_fuente, k):
    """Promedio del estrato de los k edificios conocidos más cercanos fuera del grupo propio."""
    xs, ns, cs = V.puntos_unicos(xy(df_fuente), df_fuente.npn_edificio.to_numpy())
    ys = pd.Series(y_fuente).groupby(cs).mean().to_numpy()
    gs = pd.Series(grupo_fuente).groupby(cs).first().to_numpy()
    lag, dist = V.lag_fuera_de_grupo(xy(df_dest), grupo_dest, xs, ys, gs, k)
    return pd.DataFrame({f"est{k}_estrato": lag, f"est{k}_dist_km": dist})


def pipeline(extra):
    prep = M.construir_preprocesador([c for c in nums if c in num_log], [c for c in nums if c not in num_log] + extra,
                                     cats, q_inf=q_inf, q_sup=q_sup, min_frecuencia=min_frec)
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000, **HP))])


def ajustar(df_tr, y_fit, df_ev, ex_tr, ex_ev):
    """Ajusta la logística B (+ columnas extra) y devuelve (predicción, probabilidades) en df_ev."""
    extra = list(ex_tr.columns) if ex_tr is not None else []
    Xtr = df_tr[nums + cats].reset_index(drop=True)
    Xev = df_ev[nums + cats].reset_index(drop=True)
    if extra:
        Xtr = pd.concat([Xtr, ex_tr.reset_index(drop=True)], axis=1)
        Xev = pd.concat([Xev, ex_ev.reset_index(drop=True)], axis=1)
    est = pipeline(extra).fit(Xtr, y_fit)
    return est.predict(Xev), est.predict_proba(Xev)


def un_fold(itr, iva, grupo, kx, ky):
    tr, va = train.iloc[itr], train.iloc[iva]
    partes_tr, partes_va = [], []
    if kx == "radios":
        partes_tr.append(lag_radios[0].iloc[itr])
        partes_va.append(lag_radios[0].iloc[iva])
    elif kx:
        partes_tr.append(lagX[kx][0].iloc[itr])
        partes_va.append(lagX[kx][0].iloc[iva])
    if ky:
        partes_tr.append(lag_estrato(tr, grupo[itr], tr, y_tr[itr], grupo[itr], ky))
        partes_va.append(lag_estrato(va, grupo[iva], tr, y_tr[itr], grupo[itr], ky))
    ex_tr = pd.concat([p.reset_index(drop=True) for p in partes_tr], axis=1) if partes_tr else None
    ex_va = pd.concat([p.reset_index(drop=True) for p in partes_va], axis=1) if partes_va else None
    pred, _ = ajustar(tr, y_tr[itr], va, ex_tr, ex_va)
    yv = y_tr[iva]
    return {"F1_macro": f1_score(yv, pred, average="macro", zero_division=0),
            "MAE": float(np.mean(np.abs(yv - pred))),
            "recall_E4": recall_score(yv, pred, labels=[4], average="macro", zero_division=0)}


def evaluar_cv(kn, kx=None, ky=None):
    out = Parallel(n_jobs=5)(delayed(un_fold)(itr, iva, kn["grupo_fila"], kx, ky) for itr, iva in kn["folds"])
    return pd.DataFrame(out)


def nombre(kx, ky):
    n = "Logística B"
    if kx == "radios":
        return n + " + vecindad por radios (300 m y 1 km, cap. 5)"
    if kx:
        n += f" + vecindad KNN (k = {kx})"
    if ky:
        n += f" + estrato de vecinos (k = {ky})"
    return n


resultados = {}
for esc, kn in ESCENARIOS.items():
    t0 = time.time()
    res = {nombre(None, None): evaluar_cv(kn), nombre("radios", None): evaluar_cv(kn, kx="radios")}
    for kx in KS_X:
        res[nombre(kx, None)] = evaluar_cv(kn, kx=kx)
    # k de la vecindad física: el de mayor F1 medio en CV (el criterio "minimizar el error" del libro)
    kx_mejor = max(KS_X, key=lambda k: res[nombre(k, None)].F1_macro.mean())
    for ky in KS_Y:
        res[nombre(None, ky)] = evaluar_cv(kn, ky=ky)
        res[nombre(kx_mejor, ky)] = evaluar_cv(kn, kx=kx_mejor, ky=ky)
    ky_mejor = max(KS_Y, key=lambda k: res[nombre(kx_mejor, k)].F1_macro.mean())
    base = res["Logística B"]
    tabla = {}
    for n, r in res.items():
        d = r.F1_macro.to_numpy() - base.F1_macro.to_numpy()
        tabla[n] = {"F1_macro": r.F1_macro.mean(), "MAE": r.MAE.mean(), "recall_E4": r.recall_E4.mean(),
                    "dif_F1_vs_B": d.mean(), "SE_dif": d.std(ddof=0) / np.sqrt(len(d)),
                    "folds_F1": r.F1_macro.round(4).tolist()}
    resultados[esc] = {"q": kn["q"], "W_km": kn["W"], "kx_elegido": kx_mejor, "ky_elegido": ky_mejor,
                       "modelos": tabla,
                       "distancia_escenario_km": V.cuantiles_ponderados(kn["d_pred"], kn["pesos_pred"]),
                       "distancia_cv_km": V.cuantiles_ponderados(kn["d_cv"], kn["pesos_g"])}
    print(f"\n== {esc} ({time.time() - t0:.0f} s): k vecindad = {kx_mejor}, k estrato = {ky_mejor}", flush=True)
    print(pd.DataFrame(tabla).T.drop(columns="folds_F1").round(4).to_string(), flush=True)

# ---------------------------------------------------------------------------
# 4) Test (zonas nuevas): mismo test del Entregable 1, entrenando con todo train (sin buffer)
# ---------------------------------------------------------------------------
kx, ky = resultados["zonas nuevas"]["kx_elegido"], resultados["zonas nuevas"]["ky_elegido"]
g_tr = kn_nuevas["grupo_fila"]
g_te = np.full(len(test), -1)
ey_tr = lag_estrato(train, g_tr, train, y_tr, g_tr, ky)
ey_te = lag_estrato(test, g_te, train, y_tr, g_tr, ky)
variantes = {
    "Logística B": (None, None),
    nombre("radios", None): (lag_radios[0], lag_radios[1]),
    nombre(kx, None): (lagX[kx][0], lagX[kx][1]),
    nombre(None, ky): (ey_tr, ey_te),
    nombre(kx, ky): (pd.concat([lagX[kx][0], ey_tr], axis=1), pd.concat([lagX[kx][1], ey_te], axis=1)),
}
bloques = test.bloque.to_numpy()
f1m = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
mae = lambda a, b: float(np.mean(np.abs(a - b)))
pred_te, test_res = {}, {}
for n, (etr, ete) in variantes.items():
    p, pr = ajustar(train, y_tr, test, etr, ete)
    pred_te[n] = p
    m = M.metricas(y_te, p, pr)
    ic = M.bootstrap_por_bloques(y_te, p, pr, bloques, B=500)
    test_res[n] = {"F1_macro": m["f1_macro"], "accuracy": m["accuracy"], "MAE": m["MAE_ordinal"],
                   "kappa": m["kappa_cuadrático"], "AUC": m["AUC_OvR_macro"],
                   "IC95_F1": [ic.loc["f1_macro", "IC95_inf"], ic.loc["f1_macro", "IC95_sup"]],
                   "recall": dict(zip([f"E{k}" for k in C.CLASES],
                                      recall_score(y_te, p, labels=C.CLASES, average=None, zero_division=0).tolist()))}
for n in list(variantes)[1:]:
    test_res[n]["dif_vs_B"] = {
        "F1": M.bootstrap_diferencia(y_te, pred_te[n], pred_te["Logística B"], bloques, f1m, B=500),
        "MAE": M.bootstrap_diferencia(y_te, pred_te[n], pred_te["Logística B"], bloques, mae, B=500)}
print("\n== test (zonas nuevas)", flush=True)
print(pd.DataFrame({n: {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                    for n, r in test_res.items()}).T.round(4).to_string(), flush=True)
for n in list(variantes)[1:]:
    print(n, {k: {kk: round(float(vv), 4) for kk, vv in v.items() if isinstance(vv, (int, float, np.floating))}
              for k, v in test_res[n]["dif_vs_B"].items()}, flush=True)


def _limpio(o):
    if isinstance(o, dict):
        return {str(k): _limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpio(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    return o


SALIDA.write_text(json.dumps(_limpio({
    "hiperparametros_B": HP, "ks_vecindad": KS_X, "ks_estrato": KS_Y,
    "moran_por_k": moran_k.to_dict(orient="list"),
    "cv": resultados, "test_zonas_nuevas": test_res,
    "atributos_vecindad": list(attrs.columns),
}), ensure_ascii=False, indent=1), encoding="utf-8")
# Copia de los resultados del cap. 14 del Entregable 1 para el dashboard y el libro (solo agregados).
(AQUI.parent / "datos" / "proyecto" / "v3_e1.json").write_text(json.dumps(v3, ensure_ascii=False, indent=1),
                                                              encoding="utf-8")
print(f"\nguardado {SALIDA.name} ({time.time() - t00:.0f} s en total)", flush=True)
