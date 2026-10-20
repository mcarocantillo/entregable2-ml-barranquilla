"""Recalcula el modelo base con el protocolo de la versión 3 para el dashboard (pestaña «Modelo base»).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Versión 3 (sugerencia del profesor): la validación cruzada usa folds **kNNDM** (que reproducen la distancia a la que
se predice una zona nueva) en lugar de bloques con buffer de 1 km, y los modelos finales se entrenan con **todo** el
entrenamiento (257 273 viviendas, sin buffer). El test es el mismo de siempre: 7 bloques de 2 km.

Repite, con ese protocolo, los cálculos del capítulo 10 del Entregable 1 y de ``preparar_datos.py`` (mismo código del
Entregable 1: preprocesamiento, búsqueda con la regla 1-SE, líneas base, bootstrap por bloques) y escribe en
``datos/base_v3/`` archivos con el MISMO formato que los de ``datos/`` (que siguen siendo los de la versión 2 y los usa
el libro del Entregable 2):

- ``resultados.json``, ``diagnosticos.json``, ``test_predicciones.pkl.gz``, ``coeficientes.pkl.gz``,
  ``residuos_mapa.pkl.gz``, ``modelo_logistica_B.joblib``, ``simulador_rangos.json``;
- ``grid_cv.json`` (búsqueda de hiperparámetros), ``odds.json`` y ``importancia.json``.

Uso:  python proyecto/preparar_base_v3.py RUTA_AL_ENTREGABLE1_V3
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline

E1 = Path(sys.argv[1])
AQUI = Path(__file__).resolve().parent.parent
SALIDA = AQUI / "datos" / "base_v3"
SALIDA.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(E1))
from src import config as C  # noqa: E402
from src import espacial as S  # noqa: E402
from src import modelo as M  # noqa: E402
from src import particion as P  # noqa: E402
from src import vecinos as V  # noqa: E402

t_inicio = time.time()
dec = C.leer_decisiones()
datos = P.cargar_particion()
vm = dec["variables_modelo"]["valor"]
num_log = dec["num_log"]["valor"]
q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
min_frec = dec["onehot_min_frecuencia"]["valor"]
train = datos[datos.particion == "train"].reset_index(drop=True)      # todo el entrenamiento: sin buffer
test = datos[datos.particion == "test"].reset_index(drop=True)
y_tr, y_te = train[C.OBJETIVO].to_numpy(), test[C.OBJETIVO].to_numpy()
conjuntos = {"A_fisicas": {"num": vm["numericas"], "cat": vm["categoricas"]},
             "B_fisicas+ubicacion": {"num": vm["numericas"] + vm["espaciales"], "cat": vm["categoricas"]}}
B = "B_fisicas+ubicacion"
cols = {k: v["num"] + v["cat"] for k, v in conjuntos.items()}
print(f"Entrenamiento: {len(train):,} | test: {len(test):,} en {test.bloque.nunique()} bloques", flush=True)


def pipeline_logistica(conj, **kw):
    nums, cats = conj["num"], conj["cat"]
    prep = M.construir_preprocesador([c for c in nums if c in num_log], [c for c in nums if c not in num_log],
                                     cats, q_inf=q_inf, q_sup=q_sup, min_frecuencia=min_frec)
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000, **kw))])


def xy(df):
    return df[["x_km", "y_km"]].to_numpy()


# ---------------------------------------------------------------------------
# 1. Folds kNNDM (igual que el capítulo 14 del Entregable 1)
# ---------------------------------------------------------------------------
xy_te_g, n_te, _ = V.puntos_unicos(xy(test), test.npn_edificio.to_numpy())
res_knndm = V.knndm(xy(train), train.npn_edificio.to_numpy(), y_tr, xy_te_g, n_te, verbose=False)
folds = res_knndm["folds"]
print(f"kNNDM: q = {res_knndm['q']}, Wasserstein = {res_knndm['W']:.3f} km ({time.time() - t_inicio:.0f} s)", flush=True)

# ---------------------------------------------------------------------------
# 2. Búsqueda de hiperparámetros con la regla 1-SE (capítulo 14.3)
# ---------------------------------------------------------------------------
grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}


def regla_1se(cv_results, k=len(folds)):
    media = np.asarray(cv_results["mean_test_score"], float)
    se = np.asarray(cv_results["std_test_score"], float) / np.sqrt(k)
    i_mejor = int(np.nanargmax(media))
    candidatos = np.flatnonzero(media >= media[i_mejor] - se[i_mejor])
    c_vals = np.asarray(cv_results["param_clf__C"], float)
    finalistas = [i for i in candidatos if c_vals[i] == c_vals[candidatos].min()]
    return int(finalistas[int(np.argmax(media[finalistas]))])


busquedas, filas_grid = {}, []
for nombre, conj in conjuntos.items():
    t0 = time.time()
    gs = GridSearchCV(pipeline_logistica(conj), grid, cv=folds, scoring="f1_macro", n_jobs=-1, refit=False,
                      return_train_score=True)
    gs.fit(train[cols[nombre]], y_tr)
    i = regla_1se(gs.cv_results_)
    busquedas[nombre] = {"params": gs.cv_results_["params"][i], "i": i, "cv": gs.cv_results_}
    for j, p in enumerate(gs.cv_results_["params"]):
        filas_grid.append({"conjunto": nombre, "C": float(p["clf__C"]),
                           "pesos": "sin pesos" if p["clf__class_weight"] is None else "balanced",
                           "f1_cv": [float(gs.cv_results_[f"split{f}_test_score"][j]) for f in range(len(folds))],
                           "f1_train": [float(gs.cv_results_[f"split{f}_train_score"][j]) for f in range(len(folds))]})
    print(f"{nombre}: {gs.cv_results_['params'][i]} | F1 CV {gs.cv_results_['mean_test_score'][i]:.4f} "
          f"({time.time() - t0:.0f} s)", flush=True)
(SALIDA / "grid_cv.json").write_text(json.dumps(filas_grid, ensure_ascii=False), encoding="utf-8")


def hp(nombre):
    p = busquedas[nombre]["params"]
    return {"C": p["clf__C"], "class_weight": p["clf__class_weight"]}


# ---------------------------------------------------------------------------
# 3. Líneas base y tabla de validación cruzada (capítulo 10.4, con folds kNNDM)
# ---------------------------------------------------------------------------
zonas = {}
for tam_z in [2.0, 4.0, 6.0, 8.0]:
    r = cross_validate(M.BaselineModaZona(tam_km=tam_z), train[["x_km", "y_km"]], y_tr, cv=folds, scoring="f1_macro")
    zonas[tam_z] = float(r["test_score"].mean())
tam_zona = max(zonas, key=zonas.get)
print("Moda por zona, F1 CV por tamaño:", {k: round(v, 4) for k, v in zonas.items()}, "->", tam_zona, flush=True)

modelos = {
    "Dummy (mayoritaria)": (DummyClassifier(strategy="most_frequent"), ["x_km"]),
    "Dummy (estratificada)": (DummyClassifier(strategy="stratified", random_state=C.SEED), ["x_km"]),
    "Dummy (uniforme)": (DummyClassifier(strategy="uniform", random_state=C.SEED), ["x_km"]),
    "Moda por zona": (M.BaselineModaZona(tam_km=tam_zona), ["x_km", "y_km"]),
}
for nombre, conj in conjuntos.items():
    modelos[f"Logística {nombre}"] = (pipeline_logistica(conj, **hp(nombre)), cols[nombre])
filas = []
for nombre, (est, c) in modelos.items():
    r = cross_validate(est, train[c], y_tr, cv=folds, n_jobs=-1,
                       scoring={"f1_macro": "f1_macro", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"})
    filas.append({"modelo": nombre, **{f"{m} (media)": r[f"test_{m}"].mean()
                                       for m in ["f1_macro", "accuracy", "balanced_accuracy"]},
                  "F1 macro (desv. entre folds)": r["test_f1_macro"].std()})
tabla_cv = pd.DataFrame(filas).set_index("modelo")
print(tabla_cv.round(4), flush=True)

# ---------------------------------------------------------------------------
# 4. Modelos finales con todo el entrenamiento y evaluación en test
# ---------------------------------------------------------------------------
pred = pd.DataFrame({"estrato": y_te, "bloque": test["bloque"].to_numpy(),
                     "lat": test["centroide_lat"].round(3).to_numpy(), "lon": test["centroide_lon"].round(3).to_numpy(),
                     "edificio": pd.factorize(test["npn_edificio"])[0]})
ajustados, p_pred, p_proba = {}, {}, {}
for nombre, (est, c) in modelos.items():
    est.fit(train[c], y_tr)
    ajustados[nombre] = est
    proba = est.predict_proba(test[c])
    p_pred[nombre], p_proba[nombre] = est.predict(test[c]), proba
    pred[f"pred|{nombre}"] = p_pred[nombre]
    for j, k in enumerate(est.classes_):
        pred[f"p{k}|{nombre}"] = proba[:, j].astype("float32")
pred.to_pickle(SALIDA / "test_predicciones.pkl.gz")
resultados = pd.DataFrame({n: M.metricas(y_te, p_pred[n], p_proba[n]) for n in modelos}).T
PRINCIPAL = f"Logística {B}"
print(resultados[["f1_macro", "MAE_ordinal", "accuracy_±1", "kappa_cuadrático"]].round(4), flush=True)

bloques = test["bloque"].to_numpy()
ic = M.bootstrap_por_bloques(y_te, p_pred[PRINCIPAL], p_proba[PRINCIPAL], bloques, B=500)
ic.insert(0, "valor en test", [resultados.loc[PRINCIPAL, k] for k in ic.index])
f1m = lambda a, b: f1_score(a, b, average="macro", zero_division=0)  # noqa: E731
comparaciones = {f"{PRINCIPAL} − {base}": M.bootstrap_diferencia(y_te, p_pred[PRINCIPAL], p_pred[base], bloques, f1m, B=500)
                 for base in ["Dummy (mayoritaria)", "Dummy (estratificada)", "Moda por zona", "Logística A_fisicas"]}
comp = pd.DataFrame(comparaciones).T
comp["¿IC excluye 0?"] = (comp["IC95_inf"] > 0) | (comp["IC95_sup"] < 0)

# Validación aleatoria frente a kNNDM: el optimismo de ignorar el espacio (capítulo 10.2, con el protocolo v3)
principal = ajustados[PRINCIPAL]
r_alea = cross_validate(clone(principal), train[cols[B]], y_tr, scoring="f1_macro", n_jobs=-1,
                        cv=StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED))
cvB = busquedas[B]["cv"]
f1_knndm, sd_knndm = cvB["mean_test_score"][busquedas[B]["i"]], cvB["std_test_score"][busquedas[B]["i"]]
optimismo = pd.DataFrame({"F1 macro CV": [r_alea["test_score"].mean(), f1_knndm],
                          "desv.": [r_alea["test_score"].std(), sd_knndm]},
                         index=["aleatoria (StratifiedKFold)", "espacial (kNNDM)"])
optimismo["diferencia vs espacial"] = optimismo["F1 macro CV"] - f1_knndm
print(optimismo.round(4), flush=True)

v_ant = json.loads((AQUI / "datos" / "resultados.json").read_text(encoding="utf-8"))
res = {
    "version": "v3: validación kNNDM, entrenamiento sin buffer",
    "tabla_cv": tabla_cv.round(4).to_dict(), "resultados_test": resultados.round(4).to_dict(),
    "ic_modelo_principal": ic.round(4).to_dict(), "comparaciones_f1": comp.round(4).astype(str).to_dict(),
    "optimismo_validacion_aleatoria": optimismo.round(4).to_dict(),
    "mejores_hiperparametros": {k: {kk: str(vv) for kk, vv in v["params"].items()} for k, v in busquedas.items()},
    "n_train_final": int(len(train)), "n_train": int(len(train)), "n_train_antes_buffer": int(len(train)),
    "n_test": int(len(test)), "bloques_test": int(test.bloque.nunique()), "buffer_km": 0.0,
    "tam_zona_moda_km": float(tam_zona), "f1_moda_por_tamano": zonas,
    "knndm": {"q": int(res_knndm["q"]), "W_km": float(res_knndm["W"])},
    "variables_modelo": vm, "decisiones": {k: v["valor"] for k, v in dec.items()},
    # Versión anterior (bloques con buffer de 1 km), para la diapositiva de cambios
    "version_anterior": {"tabla_cv": v_ant["tabla_cv"], "resultados_test": v_ant["resultados_test"],
                         "mejores_hiperparametros": v_ant["mejores_hiperparametros"], "n_train": v_ant["n_train"]},
}
(SALIDA / "resultados.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------------------------------------------------------------------------
# 5. Coeficientes, odds ratio, importancia por permutación y simulador
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

clf = principal.named_steps["clf"]
nombres = [n.split("__", 1)[1] for n in principal.named_steps["prep"].get_feature_names_out()]
beta = pd.Series(clf.coef_[-1] - clf.coef_[0], index=nombres)
top = beta.reindex(beta.abs().sort_values(ascending=False).index).head(15)
(SALIDA / "odds.json").write_text(json.dumps({"variable": top.index.tolist(), "beta_6_menos_1": top.round(4).tolist(),
                                              "odds": np.exp(top.clip(-50, 50)).round(4).tolist()}, ensure_ascii=False),
                                  encoding="utf-8")

itr, iva = folds[0]
modelo_fold = clone(principal).fit(train.iloc[itr][cols[B]], y_tr[itr])
sub = np.random.default_rng(C.SEED).choice(iva, min(20_000, len(iva)), replace=False)
pi = permutation_importance(modelo_fold, train.iloc[sub][cols[B]], y_tr[sub], scoring="f1_macro", n_repeats=3,
                            random_state=C.SEED, n_jobs=-1)
imp = pd.DataFrame({"caida": pi.importances_mean, "desv": pi.importances_std}, index=cols[B]).sort_values("caida")
(SALIDA / "importancia.json").write_text(json.dumps({"variable": imp.index.tolist(), "caida": imp.caida.round(5).tolist(),
                                                     "desv": imp.desv.round(5).tolist()}, ensure_ascii=False),
                                         encoding="utf-8")

joblib.dump(principal, SALIDA / "modelo_logistica_B.joblib", compress=3)
ref = {c: [float(train[c].quantile(q)) for q in (0.01, 0.5, 0.99)] for c in vm["numericas"]}
ref["categorias"] = {c: sorted(map(str, train[c].dropna().unique())) for c in vm["categoricas"]}
combo = train[vm["categoricas"]].astype(str).value_counts().idxmax()
ref["moda_categorias"] = dict(zip(vm["categoricas"], map(str, combo)))
(SALIDA / "simulador_rangos.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Coeficientes, importancia y simulador listos ({time.time() - t_inicio:.0f} s)", flush=True)

# ---------------------------------------------------------------------------
# 6. Diagnósticos: curva de aprendizaje, residuos espaciales
# ---------------------------------------------------------------------------
diag_ant = json.loads((AQUI / "datos" / "diagnosticos.json").read_text(encoding="utf-8"))
diag = {"correlograma": diag_ant["correlograma"], "moran_estrato": diag_ant["moran_estrato"],
        "sensibilidad_buffer": diag_ant["sensibilidad_buffer"]}    # del EDA y de la versión anterior (sin cambios)
tam, sc_tr, sc_va = learning_curve(clone(principal), train[cols[B]], y_tr, cv=folds, scoring="f1_macro",
                                   train_sizes=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0], n_jobs=-1, shuffle=True,
                                   random_state=C.SEED)
diag["curva_aprendizaje"] = {"n": tam.tolist(), "train_media": sc_tr.mean(1).tolist(), "train_desv": sc_tr.std(1).tolist(),
                             "val_media": sc_va.mean(1).tolist(), "val_desv": sc_va.std(1).tolist()}
print("Curva de aprendizaje:", [round(v, 3) for v in sc_va.mean(1)], flush=True)

diag["moran_residuos"] = {}
for nombre in conjuntos:
    est = ajustados[f"Logística {nombre}"]
    esperado = est.predict_proba(test[cols[nombre]]) @ est.classes_.astype(float)
    t = test.assign(residuo=y_te - esperado)
    e = S.deduplicar_por_edificio(t[["npn_edificio", "x_km", "y_km", "ubicado_por_manzana", "residuo"]])
    Wt, _, _ = S.pesos_knn(e[["x_km", "y_km"]].to_numpy(), k=C.K_VECINOS)
    r = S.moran_global(e["residuo"].to_numpy(), Wt, n_perm=199, seed=C.SEED)
    diag["moran_residuos"][nombre] = {"I": float(r["I"]), "p": float(r["p_perm"]), "residuo_medio": float(t["residuo"].mean())}
    if nombre == B:
        g = t.assign(lat=t.centroide_lat.round(3), lon=t.centroide_lon.round(3))
        g.groupby(["lat", "lon"], as_index=False).agg(residuo=("residuo", "mean"), n=("residuo", "size")).to_pickle(
            SALIDA / "residuos_mapa.pkl.gz")
print("Moran de residuos:", diag["moran_residuos"], flush=True)
(SALIDA / "diagnosticos.json").write_text(json.dumps(diag, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Listo en {time.time() - t_inicio:.0f} s:", sorted(p.name for p in SALIDA.iterdir()), flush=True)
