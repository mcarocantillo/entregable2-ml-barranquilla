"""Prepara los datos y los resultados que usa el dashboard del Entregable 2.

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)

Se ejecuta UNA vez, en local, antes de desplegar. Lee las salidas del
Entregable 1 (partición espacial, decisiones del EDA y resultados del modelo)
y escribe en ``datos/`` todo lo que el dashboard necesita, para que la app
desplegada no tenga que reentrenar nada:

- ``viviendas.pkl.gz``: las 332 718 viviendas limpias (con y sin
  coordenadas), sin el número predial (identificador) y con coordenadas
  redondeadas a 3 decimales (~110 m).
- ``test_predicciones.pkl.gz``: estrato real, predicción y probabilidades de
  cada modelo en el conjunto de prueba.
- ``coeficientes.pkl.gz``: coeficientes de las regresiones logísticas.
- ``resultados.json``: métricas de CV y test, intervalos y comparaciones.
- ``modelo_logistica_B.joblib``: el modelo principal ajustado (simulador).

Uso:  python preparar_datos.py  [ruta_al_entregable1]
"""
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

AQUI = Path(__file__).resolve().parent
E1 = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI.parent / "entregable1_jbook"
SALIDA = AQUI / "datos"
SALIDA.mkdir(exist_ok=True)

# Se usan los módulos del Entregable 1 apuntando a sus datos procesados.
sys.path.insert(0, str(E1))
from src import config as C  # noqa: E402
from src import modelo as M  # noqa: E402
from src import particion as P  # noqa: E402

dec = C.leer_decisiones()
res_e1 = json.loads((C.CARPETA_PROCESADOS / "resultados_modelo.json").read_text(encoding="utf-8"))
dominios = json.loads((C.CARPETA_DATOS / "dominios_arcgis.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# 1. Viviendas (dataset completo limpio)
# ---------------------------------------------------------------------------
con = pd.read_parquet(C.CARPETA_PROCESADOS / "dataset_modelado.parquet")
sin = pd.read_parquet(C.CARPETA_PROCESADOS / "sin_coordenadas.parquet")
sin["particion"] = "sin coordenadas"
todo = pd.concat([con, sin], ignore_index=True)

# Identificador anónimo de edificio (en lugar del prefijo del número predial).
todo["edificio_id"] = pd.factorize(todo["npn_edificio"])[0]
etq_tv = {f"tv_{k}": v for k, v in dominios["tipo_vivienda"]["dominio"].items()}
todo["tipo_vivienda"] = todo["tipo_vivienda"].map(lambda v: etq_tv.get(v, v))
todo["uso"] = todo["uso"].str.replace("_", " ")
flags = [c for c in todo.columns if c.startswith("flag_imposible") or c.startswith("flag_incoherente")]
todo["valor_corregido"] = todo[flags].any(axis=1)
todo["lat"] = todo["centroide_lat"].round(3)
todo["lon"] = todo["centroide_lon"].round(3)

cols = ["estrato_num", "tipo_vivienda", "uso", "condicion_predio", "area_construida",
        "area_catastral_terreno", "total_habitaciones", "total_banios", "total_plantas",
        "planta_ubicacion", "altura", "anio_construccion", "antiguedad", "lat", "lon",
        "dist_centro_km", "tiene_coordenadas", "particion", "edificio_id"]
viv = todo[cols].copy()
for c in ["tipo_vivienda", "uso", "condicion_predio", "particion"]:
    viv[c] = viv[c].astype("category")
# Tipos compactos: el plan gratuito de Render tiene 512 MB de RAM.
for c in viv.select_dtypes("float64"):
    viv[c] = viv[c].astype("float32")
viv["edificio_id"] = viv["edificio_id"].astype("int32")
viv["estrato_num"] = viv["estrato_num"].astype("int8")
# Pickle comprimido (no parquet) para no cargar pyarrow en el servidor.
viv.to_pickle(SALIDA / "viviendas.pkl.gz")
print(f"viviendas.pkl.gz: {len(viv):,} filas, {viv.edificio_id.nunique():,} edificios")

# ---------------------------------------------------------------------------
# 2. Modelos: reajuste con los hiperparámetros elegidos en el Entregable 1
# ---------------------------------------------------------------------------
datos = P.cargar_particion()
vm = dec["variables_modelo"]["valor"]
num_log = dec["num_log"]["valor"]
q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
min_frec = dec["onehot_min_frecuencia"]["valor"]
buffer_km = dec["buffer_km"]["valor"]

train_total = datos[datos.particion == "train"].reset_index(drop=True)
test = datos[datos.particion == "test"].reset_index(drop=True)
mantener = P.mascara_buffer(train_total[["x_km", "y_km"]].to_numpy(),
                            test[["x_km", "y_km"]].to_numpy(), buffer_km)
train = train_total[mantener].reset_index(drop=True)
y_tr, y_te = train[C.OBJETIVO].to_numpy(), test[C.OBJETIVO].to_numpy()

conjuntos = {
    "A_fisicas": {"num": vm["numericas"], "cat": vm["categoricas"]},
    "B_fisicas+ubicacion": {"num": vm["numericas"] + vm["espaciales"], "cat": vm["categoricas"]},
}


def pipeline_logistica(conj, C_reg, class_weight):
    """Mismo Pipeline del Entregable 1 (preprocesamiento + logística multinomial)."""
    nums, cats = conj["num"], conj["cat"]
    prep = M.construir_preprocesador([c for c in nums if c in num_log], [c for c in nums if c not in num_log],
                                     cats, q_inf=q_inf, q_sup=q_sup, min_frecuencia=min_frec)
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=C_reg, class_weight=class_weight,
                                                                max_iter=1000))])


hp = res_e1["mejores_hiperparametros"]


def leer_hp(nombre):
    cw = hp[nombre]["clf__class_weight"]
    return float(hp[nombre]["clf__C"]), (None if cw in (None, "None") else cw)


modelos = {
    "Dummy (mayoritaria)": (DummyClassifier(strategy="most_frequent"), ["x_km"]),
    "Dummy (estratificada)": (DummyClassifier(strategy="stratified", random_state=C.SEED), ["x_km"]),
    "Dummy (uniforme)": (DummyClassifier(strategy="uniform", random_state=C.SEED), ["x_km"]),
    "Moda por zona": (M.BaselineModaZona(tam_km=2.0), ["x_km", "y_km"]),
}
for nombre, conj in conjuntos.items():
    modelos[f"Logística {nombre}"] = (pipeline_logistica(conj, *leer_hp(nombre)), conj["num"] + conj["cat"])

pred = pd.DataFrame({"estrato": y_te, "bloque": test["bloque"].to_numpy(),
                     "lat": test["centroide_lat"].round(3).to_numpy(),
                     "lon": test["centroide_lon"].round(3).to_numpy(),
                     "edificio": pd.factorize(test["npn_edificio"])[0]})
ajustados = {}
for nombre, (est, cols_m) in modelos.items():
    est.fit(train[cols_m], y_tr)
    ajustados[nombre] = est
    proba = est.predict_proba(test[cols_m])
    pred[f"pred|{nombre}"] = est.predict(test[cols_m])
    for j, k in enumerate(est.classes_):
        pred[f"p{k}|{nombre}"] = proba[:, j].astype("float32")
    m = M.metricas(y_te, pred[f"pred|{nombre}"].to_numpy(), proba)
    ref = res_e1["resultados_test"]["f1_macro"][nombre]
    print(f"{nombre:32s} F1 test {m['f1_macro']:.4f} (Entregable 1: {ref:.4f})")
pred.to_pickle(SALIDA / "test_predicciones.pkl.gz")

# ---------------------------------------------------------------------------
# 3. Coeficientes de las logísticas (por desviación estándar del predictor)
# ---------------------------------------------------------------------------
filas = []
for nombre in conjuntos:
    est = ajustados[f"Logística {nombre}"]
    nombres = est.named_steps["prep"].get_feature_names_out()
    coef = est.named_steps["clf"].coef_
    for j, k in enumerate(est.named_steps["clf"].classes_):
        for f, b in zip(nombres, coef[j]):
            filas.append({"modelo": nombre, "estrato": int(k), "variable": f, "coef": float(b)})
pd.DataFrame(filas).to_pickle(SALIDA / "coeficientes.pkl.gz")

# ---------------------------------------------------------------------------
# 4. Resultados agregados y modelo para el simulador
# ---------------------------------------------------------------------------
res = dict(res_e1)
res["n_train"] = int(len(train))
res["n_train_antes_buffer"] = int(len(train_total))
res["n_test"] = int(len(test))
res["bloques_test"] = int(test.bloque.nunique())
res["variables_modelo"] = vm
res["decisiones"] = {k: v["valor"] for k, v in dec.items()}
(SALIDA / "resultados.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

principal = ajustados["Logística B_fisicas+ubicacion"]
joblib.dump(principal, SALIDA / "modelo_logistica_B.joblib", compress=3)
# Rangos de referencia para los controles del simulador (percentiles de train).
ref = {c: [float(train[c].quantile(q)) for q in (0.01, 0.5, 0.99)] for c in vm["numericas"]}
ref["categorias"] = {c: sorted(map(str, train[c].dropna().unique())) for c in vm["categoricas"]}
ref["moda_categorias"] = {c: str(train[c].mode().iloc[0]) for c in vm["categoricas"]}
(SALIDA / "simulador_rangos.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. Diagnósticos que el dashboard muestra como figuras (cálculo pesado offline)
# ---------------------------------------------------------------------------
from sklearn.model_selection import cross_validate, learning_curve  # noqa: E402

from src import espacial as S  # noqa: E402

diag = {}
cols_b = conjuntos["B_fisicas+ubicacion"]["num"] + conjuntos["B_fisicas+ubicacion"]["cat"]
folds = P.folds_espaciales_con_buffer(train, buffer_km, verbose=False)
tam, sc_tr, sc_va = learning_curve(principal, train[cols_b], y_tr, cv=folds, scoring="f1_macro",
                                   train_sizes=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0], n_jobs=-1,
                                   shuffle=True, random_state=C.SEED)
diag["curva_aprendizaje"] = {"n": tam.tolist(), "train_media": sc_tr.mean(1).tolist(),
                             "train_desv": sc_tr.std(1).tolist(), "val_media": sc_va.mean(1).tolist(),
                             "val_desv": sc_va.std(1).tolist()}
print("Curva de aprendizaje:", [round(v, 3) for v in sc_va.mean(1)])

sens = []
for b in [0.0, 0.5, 1.0, 2.875]:
    fb = P.folds_espaciales_con_buffer(train, b, verbose=False)
    r = cross_validate(principal, train[cols_b], y_tr, cv=fb, scoring="f1_macro", n_jobs=-1)
    sens.append({"buffer_km": b, "f1": float(r["test_score"].mean()), "desv": float(r["test_score"].std())})
diag["sensibilidad_buffer"] = sens
print("Sensibilidad al buffer:", sens)

# Correlograma y Moran global del estrato (una fila por edificio de train).
# Igual que el cap. 8 del Entregable 1: muestra estratificada de train y una fila por edificio.
muestra = S.muestra_estratificada(train_total, C.N_MUESTRA_ESPACIAL)
edif = S.deduplicar_por_edificio(muestra[["npn_edificio", "x_km", "y_km", C.OBJETIVO]])
xy = edif[["x_km", "y_km"]].to_numpy()
corr = S.correlograma(xy, edif[C.OBJETIVO].to_numpy(), np.arange(0, 6.25, 0.25), n_max=4000, seed=C.SEED)
diag["correlograma"] = corr.to_dict(orient="list")
W, _, _ = S.pesos_knn(xy, k=C.K_VECINOS)
mg = S.moran_global(edif[C.OBJETIVO].to_numpy(), W, n_perm=999, seed=C.SEED)
diag["moran_estrato"] = {"I": float(mg["I"]), "p": float(mg["p_perm"]), "n_edificios": int(mg["n"])}
print("Moran estrato:", diag["moran_estrato"])

# Moran de los residuos ordinales y = E[y] por edificio de test.
diag["moran_residuos"] = {}
residuos_mapa = None
for nombre in conjuntos:
    est = ajustados[f"Logística {nombre}"]
    cm = conjuntos[nombre]["num"] + conjuntos[nombre]["cat"]
    esperado = est.predict_proba(test[cm]) @ est.classes_.astype(float)
    t = test.assign(residuo=y_te - esperado)
    e = S.deduplicar_por_edificio(t[["npn_edificio", "x_km", "y_km", "residuo"]])
    Wt, _, _ = S.pesos_knn(e[["x_km", "y_km"]].to_numpy(), k=C.K_VECINOS)
    r = S.moran_global(e["residuo"].to_numpy(), Wt, n_perm=199, seed=C.SEED)
    diag["moran_residuos"][nombre] = {"I": float(r["I"]), "p": float(r["p_perm"]),
                                      "residuo_medio": float(t["residuo"].mean())}
    if nombre == "B_fisicas+ubicacion":
        g = t.assign(lat=t.centroide_lat.round(3), lon=t.centroide_lon.round(3))
        residuos_mapa = g.groupby(["lat", "lon"], as_index=False).agg(residuo=("residuo", "mean"),
                                                                       n=("residuo", "size"))
print("Moran residuos:", diag["moran_residuos"])
residuos_mapa.to_pickle(SALIDA / "residuos_mapa.pkl.gz")
(SALIDA / "diagnosticos.json").write_text(json.dumps(diag, ensure_ascii=False, indent=1), encoding="utf-8")
print("diagnosticos.json escrito")

# El dashboard debe poder cargar el modelo: necesita el paquete src del Entregable 1.
destino_src = AQUI / "src"
if destino_src.resolve() != (E1 / "src").resolve():
    shutil.copytree(E1 / "src", destino_src, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
print("Listo:", sorted(p.name for p in SALIDA.iterdir()))
