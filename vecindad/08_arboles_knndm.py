"""Versión 3, fase 2: árboles del Entregable 2 con validación kNNDM y vecindad física.

En el capítulo 5 los árboles (XGBoost y XGBoost geográfico) se evaluaron con la validación de la versión 2 (bloques
de 2 km + buffer de 1 km); la logística con vecindad se evaluó después con kNNDM (07_vecindad_knn.py). Aquí los
árboles pasan por las MISMAS condiciones que la logística de la versión 3:
  - mismos folds kNNDM (src/vecinos.py del Entregable 1 v3, semilla 42): zonas nuevas (escenario = test del
    Entregable 1, q = 30) y zonas conocidas (manzanas ocultas: distancia a la manzana conocida más cercana);
  - mismos conjuntos de variables: B; B + vecindad por radios (cap. 5); B + vecindad KNN; B + vecindad KNN +
    estrato de vecinos (calculado fuera del grupo kNNDM). Los k son los que eligió 07_vecindad_knn.py por CV;
  - mismo test (zonas nuevas), entrenando con todo train (sin buffer), IC por bootstrap de bloques.
Hiperparámetros FIJOS: logística B v3 (cap. 14 del Entregable 1) y XGBoost / XGBoost geográfico del cap. 4
(datos/resultados.json); no se re-buscan con kNNDM.
La logística se reajusta en los mismos folds para tener diferencias PAREADAS por fold; su F1 por fold se compara
con el de 07_vecindad_knn.py (si coinciden, los folds y las variables son los mismos).
El XGBoost es el modelo global del XGBoost geográfico (mismos hiperparámetros y semilla), así que se ajusta una vez.
Zonas conocidas no tiene test externo: además de la CV se dan las métricas de las predicciones fuera de muestra
(OOF) con IC por bootstrap de los bloques de 2 km de train.

Uso (desde e2_git):  ENTREGABLE1=../e1_rama ../venv_e2/Scripts/python -u vecindad/08_arboles_knndm.py
Opcional: PUNTOS_CONTROL=<carpeta> guarda folds y predicciones para reanudar; N_TRABAJOS (4) procesos en paralelo;
RAPIDO=1 es una prueba de humo (20 % de los edificios, 1 fold, 2 conjuntos; escribe en SALIDA o en la carpeta
temporal, nunca en datos/).
"""
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score

sys.stdout.reconfigure(encoding="utf-8")   # el registro lleva Δ, → y −, que no existen en cp1252
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
E1 = Path(os.environ.get("ENTREGABLE1", AQUI.parents[1] / "e1_rama"))
sys.path.insert(0, str(E1))
from src import config as C  # noqa: E402
from src import modelo as M  # noqa: E402
from src import particion as P  # noqa: E402
from src import vecinos as V  # noqa: E402

# append (no insert): e2_git también tiene una carpeta src/ que no debe tapar la del Entregable 1
sys.path.append(str(RAIZ))
sys.path.append(str(AQUI))
from avanzados import GeoXGBEstrato, _proba_completa  # noqa: E402
from variables_vecindad import calcular_vecindad  # noqa: E402

RAPIDO = os.environ.get("RAPIDO") == "1"
SALIDA = Path(os.environ.get("SALIDA") or (Path(tempfile.gettempdir()) / "arboles_knndm_rapido.json" if RAPIDO
                                          else RAIZ / "datos" / "vecindad" / "arboles_knndm.json"))
CHK = Path(os.environ["PUNTOS_CONTROL"]) if os.environ.get("PUNTOS_CONTROL") else None
SUF = "_rapido" if RAPIDO else ""
N_TRABAJOS = int(os.environ.get("N_TRABAJOS", 4))
HILOS_XGB = 2
B_BOOT = 50 if RAPIDO else 500
MODELOS = ["Logística B", "XGBoost", "XGBoost geográfico"]
ARBOLES = MODELOS[1:]
ETIQ = [f"E{k}" for k in C.CLASES]
t00 = time.time()

dec = C.leer_decisiones()
v3 = json.loads((E1 / "datos" / "procesados" / "resultados_v3.json").read_text(encoding="utf-8"))
HP_LOG = v3["hiperparametros"]["B v3"]
hp_avz = json.loads((RAIZ / "datos" / "resultados.json").read_text(encoding="utf-8"))["avanzados"]["hiperparametros"]
HP_XGB = dict(hp_avz["XGBoost"])
HP_GEO = dict(hp_avz["XGBoost geográfico"])
REF07 = json.loads((RAIZ / "datos" / "vecindad" / "vecindad_knn.json").read_text(encoding="utf-8"))
vm = dec["variables_modelo"]["valor"]
num_log = dec["num_log"]["valor"]
q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
min_frec = dec["onehot_min_frecuencia"]["valor"]
nums, cats = vm["numericas"] + vm["espaciales"], vm["categoricas"]

datos = P.cargar_particion()
train = datos[datos.particion == "train"].reset_index(drop=True)
test = datos[datos.particion == "test"].reset_index(drop=True)
if RAPIDO:
    rng = np.random.default_rng(C.SEED)

    def _submuestra(df):
        e = df.npn_edificio.unique()
        return df[df.npn_edificio.isin(rng.choice(e, int(0.2 * len(e)), replace=False))].reset_index(drop=True)

    train, test = _submuestra(train), _submuestra(test)
y_tr = train[C.OBJETIVO].to_numpy().astype(int)
y_te = test[C.OBJETIVO].to_numpy().astype(int)
print(f"train {len(train):,} | test {len(test):,} | logística {HP_LOG} | XGBoost {HP_XGB} | geográfico {HP_GEO}"
      f"{' | MODO RÁPIDO' if RAPIDO else ''}", flush=True)


def xy(df):
    return df[["x_km", "y_km"]].to_numpy()


XY_TR, XY_TE = xy(train), xy(test)


def guardar_chk(nombre, obj):
    if CHK is not None:
        CHK.mkdir(parents=True, exist_ok=True)
        with open(CHK / f"{nombre}{SUF}.pkl", "wb") as fh:
            pickle.dump(obj, fh)


def leer_chk(nombre):
    if CHK is not None and (CHK / f"{nombre}{SUF}.pkl").exists():
        with open(CHK / f"{nombre}{SUF}.pkl", "rb") as fh:
            print(f"  (punto de control {nombre}{SUF}.pkl)", flush=True)
            return pickle.load(fh)
    return None


# ---------------------------------------------------------------------------
# 1) Variables de vecindad física (igual que 07_vecindad_knn.py)
# ---------------------------------------------------------------------------
ELEGIDOS = {esc: (REF07["cv"][esc]["kx_elegido"], REF07["cv"][esc]["ky_elegido"]) for esc in REF07["cv"]}
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
A_g = attrs.groupby(cod).mean().to_numpy()
g_ids = np.arange(len(xy_g))
lagX = {}
for k in sorted({kx for kx, _ in ELEGIDOS.values()}):
    lag_g, dist_g = V.lag_knn(xy_g, xy_g, A_g, k, grupo_destino=g_ids, grupo_fuente=g_ids)
    df_k = pd.DataFrame(lag_g[cod], columns=[f"knn{k}_{c}" for c in attrs.columns])
    df_k[f"knn{k}_dist_km"] = dist_g[cod]
    lagX[k] = (df_k.iloc[:len(train)].reset_index(drop=True), df_k.iloc[len(train):].reset_index(drop=True))
VR = calcular_vecindad(todo).reset_index(drop=True)
lag_radios = (VR.iloc[:len(train)].reset_index(drop=True), VR.iloc[len(train):].reset_index(drop=True))
print(f"vecindad física KNN {sorted(lagX)} y por radios calculadas ({time.time() - t00:.0f} s)", flush=True)

# ---------------------------------------------------------------------------
# 2) Folds kNNDM de los dos escenarios (mismo código y semilla que 07_vecindad_knn.py)
# ---------------------------------------------------------------------------
xy_trg, n_trg, cod_tr = V.puntos_unicos(XY_TR, train.npn_edificio.to_numpy())
KN = leer_chk("knndm")
if KN is None:
    xy_teg, n_teg, _ = V.puntos_unicos(XY_TE, test.npn_edificio.to_numpy())
    print("kNNDM, zonas nuevas:", flush=True)
    kn_nuevas = V.knndm(XY_TR, train.npn_edificio.to_numpy(), y_tr, xy_teg, n_teg)
    manz = train.npn_manzana.astype(str).to_numpy()
    manz = np.where(pd.Series(manz).str.endswith("0000"), train.npn_edificio.astype(str).to_numpy(), manz)
    manz_g = pd.Series(manz).groupby(cod_tr).first().to_numpy()
    dd, ii = cKDTree(xy_trg).query(xy_trg, k=64)
    otra = manz_g[ii] != manz_g[:, None]
    d_zc = np.where(otra.any(1), dd[np.arange(len(dd)), otra.argmax(1)], dd[:, -1])
    print("kNNDM, zonas conocidas (manzanas ocultas):", flush=True)
    kn_conocidas = V.knndm(XY_TR, train.npn_edificio.to_numpy(), y_tr, d_pred=d_zc, pesos_pred=n_trg)
    KN = {"zonas nuevas": kn_nuevas, "zonas conocidas": kn_conocidas}
    guardar_chk("knndm", KN)
for esc, kn in KN.items():
    print(f"  {esc}: q = {kn['q']} (07: {REF07['cv'][esc]['q']}), W = {kn['W']:.3f} km, k vecindad = "
          f"{ELEGIDOS[esc][0]}, k estrato = {ELEGIDOS[esc][1]}", flush=True)


# ---------------------------------------------------------------------------
# 3) Conjuntos de variables y matrices
# ---------------------------------------------------------------------------
def conjuntos(esc):
    kx, ky = ELEGIDOS[esc]
    c = {"B": (None, None),
         "B + vecindad por radios (300 m y 1 km)": ("radios", None),
         f"B + vecindad KNN (k = {kx})": (kx, None),
         f"B + vecindad KNN (k = {kx}) + estrato de vecinos (k = {ky})": (kx, ky)}
    if RAPIDO:
        c = {n: v for i, (n, v) in enumerate(c.items()) if i in (0, 3)}
    return c


def nombre07(kx, ky):
    """Nombre de la misma logística en vecindad_knn.json (para verificar que los folds coinciden)."""
    n = "Logística B"
    if kx == "radios":
        return n + " + vecindad por radios (300 m y 1 km, cap. 5)"
    if kx:
        n += f" + vecindad KNN (k = {kx})"
    if ky:
        n += f" + estrato de vecinos (k = {ky})"
    return n


def lag_estrato(df_dest, grupo_dest, df_fuente, y_fuente, grupo_fuente, k):
    """Promedio del estrato de los k edificios conocidos más cercanos fuera del grupo propio (como en 07)."""
    xs, ns, cs = V.puntos_unicos(xy(df_fuente), df_fuente.npn_edificio.to_numpy())
    ys = pd.Series(y_fuente).groupby(cs).mean().to_numpy()
    gs = pd.Series(grupo_fuente).groupby(cs).first().to_numpy()
    lag, dist = V.lag_fuera_de_grupo(xy(df_dest), grupo_dest, xs, ys, gs, k)
    return pd.DataFrame({f"est{k}_estrato": lag, f"est{k}_dist_km": dist})


def extras_cv(kx, ky, itr, iva, grupo):
    tr, va = train.iloc[itr], train.iloc[iva]
    p_tr, p_va = [], []
    if kx == "radios":
        p_tr.append(lag_radios[0].iloc[itr])
        p_va.append(lag_radios[0].iloc[iva])
    elif kx:
        p_tr.append(lagX[kx][0].iloc[itr])
        p_va.append(lagX[kx][0].iloc[iva])
    if ky:
        p_tr.append(lag_estrato(tr, grupo[itr], tr, y_tr[itr], grupo[itr], ky))
        p_va.append(lag_estrato(va, grupo[iva], tr, y_tr[itr], grupo[itr], ky))
    une = lambda ps: pd.concat([p.reset_index(drop=True) for p in ps], axis=1) if ps else None  # noqa: E731
    return une(p_tr), une(p_va)


def extras_test(kx, ky, g_tr):
    p_tr, p_te = [], []
    if kx == "radios":
        p_tr.append(lag_radios[0])
        p_te.append(lag_radios[1])
    elif kx:
        p_tr.append(lagX[kx][0])
        p_te.append(lagX[kx][1])
    if ky:
        p_tr.append(lag_estrato(train, g_tr, train, y_tr, g_tr, ky))
        p_te.append(lag_estrato(test, np.full(len(test), -1), train, y_tr, g_tr, ky))
    une = lambda ps: pd.concat([p.reset_index(drop=True) for p in ps], axis=1) if ps else None  # noqa: E731
    return une(p_tr), une(p_te)


def matrices(df_tr, df_ev, ex_tr, ex_ev):
    """Preprocesador del Entregable 1 (ajustado solo con df_tr) con las columnas extra como numéricas lineales;
    es el mismo que usa la logística dentro de su Pipeline en 07_vecindad_knn.py."""
    extra = list(ex_tr.columns) if ex_tr is not None else []
    Xtr = df_tr[nums + cats].reset_index(drop=True)
    Xev = df_ev[nums + cats].reset_index(drop=True)
    if extra:
        Xtr = pd.concat([Xtr, ex_tr.reset_index(drop=True)], axis=1)
        Xev = pd.concat([Xev, ex_ev.reset_index(drop=True)], axis=1)
    prep = M.construir_preprocesador([c for c in nums if c in num_log], [c for c in nums if c not in num_log] + extra,
                                     cats, q_inf=q_inf, q_sup=q_sup, min_frecuencia=min_frec).fit(Xtr)
    return prep.transform(Xtr), prep.transform(Xev)


# ---------------------------------------------------------------------------
# 4) Ajuste de los tres modelos sobre las mismas matrices (se ejecuta en paralelo)
# ---------------------------------------------------------------------------
def ajustar_modelos(etiqueta, Xtr, ytr, xytr, Xev, xyev, hp_log, hp_xgb, hp_geo, hilos, con_proba):
    t0 = time.time()
    clases = np.arange(1, 7)
    log = LogisticRegression(max_iter=1000, **hp_log).fit(Xtr, ytr)
    p_log = _proba_completa(log, Xev, log.classes_)
    t1 = time.time()
    geo = GeoXGBEstrato(params_global={**hp_xgb, "n_jobs": hilos},
                        params_local={"balanceado": hp_xgb["balanceado"], "n_jobs": hilos}, **hp_geo).fit(Xtr, ytr, xytr)
    pg, pl, cub = geo.proba_partes(Xev, xyev)
    p_geo = geo.alpha * pg + (1 - geo.alpha) * pl
    # XGBoost entrega float32 (suma 1 ± 1e-7): se normaliza para el log loss; el argmax no cambia
    probas = {n: p / p.sum(1, keepdims=True) for n, p in
              {"Logística B": p_log, "XGBoost": pg.astype(float), "XGBoost geográfico": p_geo.astype(float)}.items()}
    print(f"    {etiqueta}: logística {t1 - t0:.0f} s, árboles {time.time() - t1:.0f} s, "
          f"{len(geo.anclas_)} anclas, cobertura {cub.mean():.2f}", flush=True)
    return {"pred": {n: clases[np.argmax(p, 1)].astype(np.int8) for n, p in probas.items()},
            "proba": probas if con_proba else None,
            "cobertura": float(cub.mean()), "anclas": int(len(geo.anclas_)), "k_vecinos": int(geo.k_usado_),
            "segundos_logistica": t1 - t0, "segundos_arboles": time.time() - t1}


def correr(lista, con_proba=False):
    """lista de (clave, función que arma las matrices) -> {clave: resultado}; las matrices se arman al despachar."""
    def gen():
        for clave, armar in lista:
            Xtr, ytr, xytr, Xev, xyev = armar()
            yield delayed(ajustar_modelos)(clave, Xtr, ytr, xytr, Xev, xyev, HP_LOG, HP_XGB, HP_GEO, HILOS_XGB, con_proba)
    out = Parallel(n_jobs=N_TRABAJOS, pre_dispatch="n_jobs")(gen())
    return dict(zip([c for c, _ in lista], out))


def f1_rapido(y, p):
    """F1 macro por tabla de confusión (igual a f1_score(average="macro", zero_division=0)); para el bootstrap."""
    cm = np.bincount(np.asarray(y, int) * 7 + np.asarray(p, int), minlength=49).reshape(7, 7)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    presentes = den > 0
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)[presentes]))


mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))  # noqa: E731


def metricas_basicas(y, p):
    return {"F1_macro": f1_score(y, p, average="macro", zero_division=0), "MAE": mae(y, p),
            "recall": dict(zip(ETIQ, recall_score(y, p, labels=C.CLASES, average=None, zero_division=0).tolist()))}


def pareada(a, b, mayor_es_mejor=True):
    """Diferencia pareada por fold a − b, su error estándar (ddof = 1, como la regla del cap. 4) y en cuántos
    folds gana a."""
    d = np.asarray(a, float) - np.asarray(b, float)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    gana = d > 0 if mayor_es_mejor else d < 0
    return {"dif_por_fold": d.tolist(), "media": float(d.mean()), "SE": se, "folds_a_favor": int(gana.sum()),
            "supera_por_1SE": bool((d.mean() > se) if mayor_es_mejor else (d.mean() < -se))}


# ---------------------------------------------------------------------------
# 5) Validación cruzada kNNDM en los dos escenarios
# ---------------------------------------------------------------------------
FOLDS_USAR = [0] if RAPIDO else list(range(C.N_FOLDS))
bloques_tr = train.bloque.to_numpy()
_p = np.roll(y_tr, 1)
assert abs(f1_rapido(y_tr, _p) - f1_score(y_tr, _p, average="macro", zero_division=0)) < 1e-12
cv_out, OOF_PRED = {}, {}
for esc, kn in KN.items():
    t0 = time.time()
    conj = conjuntos(esc)
    res = leer_chk(f"cv_{esc.replace(' ', '_')}")
    if res is None:
        print(f"\n== CV {esc}: {len(conj)} conjuntos × {len(FOLDS_USAR)} folds", flush=True)
        lista = []
        for nc, (kx, ky) in conj.items():
            for f in FOLDS_USAR:
                itr, iva = kn["folds"][f]

                def armar(kx=kx, ky=ky, itr=itr, iva=iva):
                    ex_tr, ex_va = extras_cv(kx, ky, itr, iva, kn["grupo_fila"])
                    Xtr, Xva = matrices(train.iloc[itr], train.iloc[iva], ex_tr, ex_va)
                    return Xtr, y_tr[itr], XY_TR[itr], Xva, XY_TR[iva]
                lista.append(((nc, f), armar))
        res = correr(lista)
        guardar_chk(f"cv_{esc.replace(' ', '_')}", res)
    ref_conj = list(conj)[-1]   # B + vecindad KNN + estrato de vecinos: la mejor logística por CV en 07
    filas, oof = {}, {}
    for nc in conj:
        for mod in MODELOS:
            r = {"modelo": mod, "variables": nc, "folds": {"F1_macro": [], "MAE": [], "recall": {e: [] for e in ETIQ}}}
            pred_oof = np.zeros(len(train), int)
            for f in FOLDS_USAR:
                iva = kn["folds"][f][1]
                p = res[(nc, f)]["pred"][mod].astype(int)
                pred_oof[iva] = p
                m = metricas_basicas(y_tr[iva], p)
                r["folds"]["F1_macro"].append(m["F1_macro"])
                r["folds"]["MAE"].append(m["MAE"])
                for e in ETIQ:
                    r["folds"]["recall"][e].append(m["recall"][e])
            r["F1_macro"] = float(np.mean(r["folds"]["F1_macro"]))
            r["MAE"] = float(np.mean(r["folds"]["MAE"]))
            r["recall"] = {e: float(np.mean(v)) for e, v in r["folds"]["recall"].items()}
            filas[f"{mod} | {nc}"] = r
            oof[(mod, nc)] = pred_oof
    # Verificación: la logística de este script frente a la de 07_vecindad_knn.py (mismos folds => mismo F1)
    verif = {}
    for nc, (kx, ky) in conj.items():
        ref = REF07["cv"][esc]["modelos"].get(nombre07(kx, ky))
        if ref is not None:
            mio = np.round(filas[f"Logística B | {nc}"]["folds"]["F1_macro"], 4)
            verif[nc] = float(np.max(np.abs(mio - np.asarray(ref["folds_F1"])[FOLDS_USAR])))
    # Diferencias pareadas por fold frente a tres logísticas
    refs = {"Logística B (mismas variables)": None, f"Logística B + vecindad (referencia: {ref_conj})": ref_conj,
            "Logística B (sin vecindad)": "B"}
    for clave, r in filas.items():
        if r["modelo"] == "Logística B":
            if r["variables"] != "B":
                b = filas["Logística B | B"]["folds"]
                r["vs Logística B (sin vecindad)"] = {"F1": pareada(r["folds"]["F1_macro"], b["F1_macro"]),
                                                      "MAE": pareada(r["folds"]["MAE"], b["MAE"], False)}
            continue
        for nref, cref in refs.items():
            b = filas[f"Logística B | {cref or r['variables']}"]["folds"]
            r[f"vs {nref}"] = {"F1": pareada(r["folds"]["F1_macro"], b["F1_macro"]),
                               "MAE": pareada(r["folds"]["MAE"], b["MAE"], False)}
        if r["modelo"] == "XGBoost geográfico":
            b = filas[f"XGBoost | {r['variables']}"]["folds"]
            r["vs XGBoost (mismas variables)"] = {"F1": pareada(r["folds"]["F1_macro"], b["F1_macro"]),
                                                  "MAE": pareada(r["folds"]["MAE"], b["MAE"], False)}
    # Predicciones fuera de muestra juntas (todas las filas de train) con IC por bootstrap de bloques de 2 km
    fun = {"f1_macro": lambda a, b, _: f1_rapido(a, b), "MAE_ordinal": lambda a, b, _: mae(a, b)}
    filas_oof = np.concatenate([kn["folds"][f][1] for f in FOLDS_USAR])
    yo, bo = y_tr[filas_oof], bloques_tr[filas_oof]
    OOF_PRED[esc] = (filas_oof, oof)
    oof_res = {}
    p_ref = oof[("Logística B", ref_conj)][filas_oof]
    for (mod, nc), p in oof.items():
        p = p[filas_oof]
        ic = M.bootstrap_por_bloques(yo, p, None, bo, B=B_BOOT, funciones=fun)
        o = metricas_basicas(yo, p)
        o["IC95_F1"] = [ic.loc["f1_macro", "IC95_inf"], ic.loc["f1_macro", "IC95_sup"]]
        o["IC95_MAE"] = [ic.loc["MAE_ordinal", "IC95_inf"], ic.loc["MAE_ordinal", "IC95_sup"]]
        if mod in ARBOLES:
            o["dif_vs_logistica_referencia"] = {
                "F1": M.bootstrap_diferencia(yo, p, p_ref, bo, f1_rapido, B=B_BOOT),
                "MAE": M.bootstrap_diferencia(yo, p, p_ref, bo, mae, B=B_BOOT)}
            pm = oof[("Logística B", nc)][filas_oof]
            o["dif_vs_logistica_mismas_variables"] = {
                "F1": M.bootstrap_diferencia(yo, p, pm, bo, f1_rapido, B=B_BOOT),
                "MAE": M.bootstrap_diferencia(yo, p, pm, bo, mae, B=B_BOOT)}
        oof_res[f"{mod} | {nc}"] = o
    eleccion = {mod: max(conj, key=lambda nc: filas[f"{mod} | {nc}"]["F1_macro"]) for mod in MODELOS}
    cv_out[esc] = {"q": kn["q"], "W_km": kn["W"], "k_vecindad": ELEGIDOS[esc][0], "k_estrato": ELEGIDOS[esc][1],
                   "conjuntos": list(conj), "referencia_logistica": ref_conj,
                   "verificacion_folds_logistica_vs_07_max_dif_F1": verif,
                   "variables_elegidas_por_cv": eleccion, "modelos": filas, "oof": oof_res,
                   "geo_info": {nc: {"anclas": [res[(nc, f)]["anclas"] for f in FOLDS_USAR],
                                     "k_vecinos": [res[(nc, f)]["k_vecinos"] for f in FOLDS_USAR],
                                     "cobertura": [res[(nc, f)]["cobertura"] for f in FOLDS_USAR]} for nc in conj},
                   "segundos_por_tarea": {"logistica": float(np.mean([v["segundos_logistica"] for v in res.values()])),
                                          "arboles": float(np.mean([v["segundos_arboles"] for v in res.values()]))}}
    print(f"\n== {esc} ({time.time() - t0:.0f} s) | verificación logística vs 07 (máx. |ΔF1| por fold): {verif}",
          flush=True)
    tabla = pd.DataFrame({k: {"F1": r["F1_macro"], "MAE": r["MAE"], **{e: r["recall"][e] for e in ETIQ},
                              "ΔF1 vs ref": r.get(f"vs Logística B + vecindad (referencia: {ref_conj})", {})
                              .get("F1", {}).get("media"),
                              "SE": r.get(f"vs Logística B + vecindad (referencia: {ref_conj})", {})
                              .get("F1", {}).get("SE")} for k, r in filas.items()}).T
    print(tabla.round(4).to_string(), flush=True)

# ---------------------------------------------------------------------------
# 6) Test (zonas nuevas): mismo test del Entregable 1, entrenando con todo train
# ---------------------------------------------------------------------------
conj = conjuntos("zonas nuevas")
ref_conj = list(conj)[-1]
g_tr = KN["zonas nuevas"]["grupo_fila"]
res_te = leer_chk("test")
if res_te is None:
    print("\n== test (zonas nuevas)", flush=True)
    lista = []
    for nc, (kx, ky) in conj.items():
        def armar(kx=kx, ky=ky):
            ex_tr, ex_te = extras_test(kx, ky, g_tr)
            Xtr, Xte = matrices(train, test, ex_tr, ex_te)
            return Xtr, y_tr, XY_TR, Xte, XY_TE
        lista.append((nc, armar))
    res_te = correr(lista, con_proba=True)
    guardar_chk("test", res_te)
bloques = test.bloque.to_numpy()
test_out = {}
for nc in conj:
    for mod in MODELOS:
        p = res_te[nc]["pred"][mod].astype(int)
        pr = res_te[nc]["proba"][mod]
        m = M.metricas(y_te, p, pr)
        ic = M.bootstrap_por_bloques(y_te, p, pr, bloques, B=B_BOOT)
        test_out[f"{mod} | {nc}"] = {
            "modelo": mod, "variables": nc, "F1_macro": m["f1_macro"], "accuracy": m["accuracy"],
            "MAE": m["MAE_ordinal"], "kappa": m["kappa_cuadrático"], "AUC": m["AUC_OvR_macro"],
            "IC95_F1": [ic.loc["f1_macro", "IC95_inf"], ic.loc["f1_macro", "IC95_sup"]],
            "IC95_MAE": [ic.loc["MAE_ordinal", "IC95_inf"], ic.loc["MAE_ordinal", "IC95_sup"]],
            "recall": dict(zip(ETIQ, recall_score(y_te, p, labels=C.CLASES, average=None, zero_division=0).tolist()))}
refs_te = {"Logística B (sin vecindad)": "B", "Logística B (mismas variables)": None,
           f"Logística B + vecindad (referencia: {ref_conj})": ref_conj}
if "B + vecindad por radios (300 m y 1 km)" in conj:
    refs_te["Logística B + vecindad por radios (la mejor logística en test)"] = "B + vecindad por radios (300 m y 1 km)"
for clave, r in test_out.items():
    if r["modelo"] == "Logística B":
        continue
    p = res_te[r["variables"]]["pred"][r["modelo"]].astype(int)
    for nref, cref in refs_te.items():
        q = res_te[cref or r["variables"]]["pred"]["Logística B"].astype(int)
        r[f"vs {nref}"] = {"F1": M.bootstrap_diferencia(y_te, p, q, bloques, f1_rapido, B=B_BOOT),
                           "MAE": M.bootstrap_diferencia(y_te, p, q, bloques, mae, B=B_BOOT)}
    if r["modelo"] == "XGBoost geográfico":
        q = res_te[r["variables"]]["pred"]["XGBoost"].astype(int)
        r["vs XGBoost (mismas variables)"] = {"F1": M.bootstrap_diferencia(y_te, p, q, bloques, f1_rapido, B=B_BOOT),
                                              "MAE": M.bootstrap_diferencia(y_te, p, q, bloques, mae, B=B_BOOT)}
verif_te = {}
for nc, (kx, ky) in conj.items():
    ref = REF07["test_zonas_nuevas"].get(nombre07(kx, ky))
    if ref is not None:
        verif_te[nc] = abs(test_out[f"Logística B | {nc}"]["F1_macro"] - ref["F1_macro"])
print(f"verificación logística de test vs 07 (|ΔF1|): {verif_te}", flush=True)
print(pd.DataFrame({k: {"F1": r["F1_macro"], "IC_inf": r["IC95_F1"][0], "IC_sup": r["IC95_F1"][1], "MAE": r["MAE"],
                        **r["recall"]} for k, r in test_out.items()}).T.round(4).to_string(), flush=True)

# ---------------------------------------------------------------------------
# 7) Resumen: ¿superan los árboles a la logística B + vecindad con la nueva validación?
# ---------------------------------------------------------------------------
resumen = {}
for esc, c in cv_out.items():
    ref = c["referencia_logistica"]
    nref = f"vs Logística B + vecindad (referencia: {ref})"
    log_ref = c["modelos"][f"Logística B | {ref}"]
    arb = {k: r for k, r in c["modelos"].items() if r["modelo"] in ARBOLES}
    mejor = max(arb, key=lambda k: arb[k]["F1_macro"])
    d = arb[mejor][nref]["F1"]
    dm = arb[mejor][nref]["MAE"]
    txt = (f"CV kNNDM ({esc}, q = {c['q']}): el mejor árbol es {mejor} con F1 {arb[mejor]['F1_macro']:.3f} frente a "
           f"{log_ref['F1_macro']:.3f} de la logística B + vecindad ({ref}); diferencia pareada {d['media']:+.3f} "
           f"(SE {d['SE']:.3f}, gana en {d['folds_a_favor']} de {len(d['dif_por_fold'])} folds) → "
           f"{'la SUPERA' if d['supera_por_1SE'] else 'NO la supera'} por más de 1 SE. "
           f"MAE {arb[mejor]['MAE']:.3f} frente a {log_ref['MAE']:.3f} (diferencia {dm['media']:+.3f}, SE {dm['SE']:.3f}).")
    e = {"cv": txt, "mejor_arbol_cv": mejor, "dif_F1_cv": d, "dif_MAE_cv": dm}
    if esc == "zonas nuevas":
        el = c["variables_elegidas_por_cv"]
        lg = test_out[f"Logística B | {el['Logística B']}"]
        partes = []
        for mod in ARBOLES:
            t = test_out[f"{mod} | {el[mod]}"]
            q = res_te[el["Logística B"]]["pred"]["Logística B"].astype(int)
            p = res_te[el[mod]]["pred"][mod].astype(int)
            dt = M.bootstrap_diferencia(y_te, p, q, bloques, f1_rapido, B=B_BOOT)
            e[f"test_{mod}_elegido_por_cv_vs_logistica_elegida_por_cv"] = dt
            sig = dt["IC95_inf"] > 0 or dt["IC95_sup"] < 0
            partes.append(f"{mod} ({el[mod]}) F1 {t['F1_macro']:.3f} [{t['IC95_F1'][0]:.3f}, {t['IC95_F1'][1]:.3f}], "
                          f"MAE {t['MAE']:.3f}: diferencia {dt['diferencia']:+.3f} "
                          f"[{dt['IC95_inf']:+.3f}, {dt['IC95_sup']:+.3f}]{' (significativa)' if sig else ' (IC incluye 0)'}")
        e["test"] = (f"Test (variables elegidas por CV para cada modelo): logística B ({el['Logística B']}) F1 "
                     f"{lg['F1_macro']:.3f} [{lg['IC95_F1'][0]:.3f}, {lg['IC95_F1'][1]:.3f}], MAE {lg['MAE']:.3f}; "
                     + "; ".join(partes) + ".")
    else:
        o = c["oof"][mejor]["dif_vs_logistica_referencia"]["F1"]
        e["oof"] = (f"OOF ({esc}, IC por bloques de train): {mejor} − logística de referencia "
                    f"{o['diferencia']:+.3f} [{o['IC95_inf']:+.3f}, {o['IC95_sup']:+.3f}].")
    resumen[esc] = e
    print("\n" + "\n".join(e[k] for k in ("cv", "test", "oof") if k in e), flush=True)

# ---------------------------------------------------------------------------
# 8) Foco del proyecto: ¿el XGBoost geográfico reemplaza a la logística B v3 o a la logística B + vecindad?
# ---------------------------------------------------------------------------
# Cada variante del XGBoost geográfico frente a cada logística (B v3 sin vecindad y B + cada vecindad), en los
# mismos folds. Criterio del cap. 4: reemplaza si la diferencia pareada de F1 por fold supera 1 SE. En test (zonas
# nuevas) y en las predicciones fuera de muestra (OOF) la diferencia lleva IC por bootstrap de bloques.
GEO = "XGBoost geográfico"
foco = {}
for esc, c in cv_out.items():
    filas_oof, oof = OOF_PRED[esc]
    yo, bo = y_tr[filas_oof], bloques_tr[filas_oof]
    refs = {("Logística B v3 (sin vecindad)" if nc == "B" else f"Logística {nc}"): nc for nc in c["conjuntos"]}
    tabla, textos = {}, []
    for nc in c["conjuntos"]:
        g = c["modelos"][f"{GEO} | {nc}"]
        p = oof[(GEO, nc)][filas_oof]
        fila = {"F1_cv": g["F1_macro"], "MAE_cv": g["MAE"], "recall_cv": g["recall"]}
        if esc == "zonas nuevas":
            t = test_out[f"{GEO} | {nc}"]
            fila.update(F1_test=t["F1_macro"], IC95_F1_test=t["IC95_F1"], MAE_test=t["MAE"], recall_test=t["recall"])
        for nref, cref in refs.items():
            lg = c["modelos"][f"Logística B | {cref}"]
            q = oof[("Logística B", cref)][filas_oof]
            d = {"F1_logistica_cv": lg["F1_macro"],
                 "cv_F1": pareada(g["folds"]["F1_macro"], lg["folds"]["F1_macro"]),
                 "cv_MAE": pareada(g["folds"]["MAE"], lg["folds"]["MAE"], False),
                 "oof_F1": M.bootstrap_diferencia(yo, p, q, bo, f1_rapido, B=B_BOOT),
                 "oof_MAE": M.bootstrap_diferencia(yo, p, q, bo, mae, B=B_BOOT)}
            d["reemplaza_en_cv"] = d["cv_F1"]["supera_por_1SE"]
            if esc == "zonas nuevas":
                pt = res_te[nc]["pred"][GEO].astype(int)
                qt = res_te[cref]["pred"]["Logística B"].astype(int)
                d["F1_logistica_test"] = test_out[f"Logística B | {cref}"]["F1_macro"]
                d["test_F1"] = M.bootstrap_diferencia(y_te, pt, qt, bloques, f1_rapido, B=B_BOOT)
                d["test_MAE"] = M.bootstrap_diferencia(y_te, pt, qt, bloques, mae, B=B_BOOT)
            fila[f"vs {nref}"] = d
            # texto: cada variante frente a la B v3 y frente a la logística con las mismas variables o la de referencia
            if cref == "B" or cref == nc or cref == c["referencia_logistica"]:
                cv_ = d["cv_F1"]
                s = (f"{GEO} ({nc}) − {nref}: CV {cv_['media']:+.3f} (SE {cv_['SE']:.3f}, {cv_['folds_a_favor']}/"
                     f"{len(cv_['dif_por_fold'])} folds) → {'REEMPLAZA' if d['reemplaza_en_cv'] else 'no reemplaza'}")
                iv = d["test_F1"] if esc == "zonas nuevas" else d["oof_F1"]
                s += (f"; {'test' if esc == 'zonas nuevas' else 'OOF'} {iv['diferencia']:+.3f} "
                      f"[{iv['IC95_inf']:+.3f}, {iv['IC95_sup']:+.3f}]")
                if s not in textos:
                    textos.append(s)
        tabla[f"{GEO} | {nc}"] = fila
    foco[esc] = {"criterio": "reemplaza si la diferencia pareada de F1 por fold (CV kNNDM) supera 1 SE (ddof = 1); "
                             + ("test con IC 95 % por bootstrap de los 7 bloques de test" if esc == "zonas nuevas"
                                else "sin test externo: IC 95 % de las predicciones OOF por bootstrap de bloques de train"),
                 "variantes": tabla, "lectura": textos}
    resumen[esc]["foco_xgb_geografico"] = textos
    print(f"\n== Foco {GEO}, {esc}:\n  " + "\n  ".join(textos), flush=True)


def _limpio(o):
    if isinstance(o, dict):
        return {str(k): _limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_limpio(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    return o


SALIDA.parent.mkdir(parents=True, exist_ok=True)
SALIDA.write_text(json.dumps(_limpio({
    "descripcion": "Árboles (XGBoost y XGBoost geográfico) con validación kNNDM y vecindad física (v3, fase 2). "
                   "Mismos folds, variables y test que vecindad_knn.json (07_vecindad_knn.py).",
    "modo_rapido": RAPIDO, "n_train": len(train), "n_test": len(test),
    "hiperparametros": {"Logística B": HP_LOG, "XGBoost": HP_XGB, "XGBoost geográfico": HP_GEO},
    "notas": ["Hiperparámetros fijos (logística B v3; árboles del cap. 4, elegidos con la CV de bloques + buffer).",
              "SE de la diferencia pareada por fold con ddof = 1 (07_vecindad_knn.py usaba ddof = 0).",
              "XGBoost = modelo global del XGBoost geográfico (mismos hiperparámetros y semilla).",
              "LightGBM no se evaluó: no está instalado en venv_e2."],
    "resumen": resumen, "xgb_geografico_vs_logistica": foco, "cv": cv_out, "test_zonas_nuevas": test_out,
    "verificacion_test_logistica_vs_07_dif_F1": verif_te,
}), ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nguardado {SALIDA} ({time.time() - t00:.0f} s en total)", flush=True)
