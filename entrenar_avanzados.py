"""Entrena y evalúa los modelos avanzados con el MISMO protocolo del Entregable 1.

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)

Protocolo (idéntico al de la logística):
- Misma partición train/test por bloques de 2 km y mismo buffer de 1 km.
- Selección de hiperparámetros SOLO con validación cruzada espacial (mismos
  5 folds con buffer), métrica F1 macro; el test no interviene.
- Cada modelo elegido se reentrena con todo el entrenamiento y se evalúa UNA
  vez en test; IC 95 % por bootstrap de bloques y diferencias pareadas.

Se ejecuta después de ``preparar_datos.py``:  python entrenar_avanzados.py
Guarda un punto de control por modelo en ``datos/avanzados_cv.json`` para poder
reanudar si se interrumpe.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

AQUI = Path(__file__).resolve().parent
E1 = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI.parent / "entregable1_jbook"
sys.path.insert(0, str(E1))
sys.path.insert(1, str(AQUI))
from src import config as C  # noqa: E402
from src import espacial as S  # noqa: E402
from src import modelo as M  # noqa: E402
from src import particion as P  # noqa: E402

from avanzados import GeoXGBEstrato, RotationForestEstrato, SVMNystroem, XGBEstrato  # noqa: E402

DATOS = AQUI / "datos"
CHK = DATOS / "avanzados_cv.json"
chk = json.loads(CHK.read_text()) if CHK.exists() else {}


def guardar_chk():
    CHK.write_text(json.dumps(chk, ensure_ascii=False, indent=1))


def f1m(y, p):
    return f1_score(y, p, average="macro", labels=C.CLASES, zero_division=0)


# ---------------------------------------------------------------------------
# Datos (igual que el capítulo 10 del Entregable 1)
# ---------------------------------------------------------------------------
dec = C.leer_decisiones()
vm = dec["variables_modelo"]["valor"]
num_log = dec["num_log"]["valor"]
q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
buffer_km = dec["buffer_km"]["valor"]
datos = P.cargar_particion()
train_total = datos[datos.particion == "train"].reset_index(drop=True)
test = datos[datos.particion == "test"].reset_index(drop=True)
mantener = P.mascara_buffer(train_total[["x_km", "y_km"]].to_numpy(), test[["x_km", "y_km"]].to_numpy(), buffer_km)
train = train_total[mantener].reset_index(drop=True)
y_tr, y_te = train[C.OBJETIVO].to_numpy(), test[C.OBJETIVO].to_numpy()
NUMS = vm["numericas"] + vm["espaciales"]   # conjunto B: físicas + ubicación
CATS = vm["categoricas"]
COLS = NUMS + CATS
folds = P.folds_espaciales_con_buffer(train, buffer_km, verbose=False)
print(f"Train {len(train):,} | test {len(test):,} | folds: {[len(v) for _, v in folds]}", flush=True)


def preprocesador():
    return M.construir_preprocesador([c for c in NUMS if c in num_log], [c for c in NUMS if c not in num_log], CATS,
                                     q_inf=q_inf, q_sup=q_sup,
                                     min_frecuencia=dec["onehot_min_frecuencia"]["valor"])


# Matrices preprocesadas por fold (el preprocesador se ajusta SOLO con el train del fold).
print("Preprocesando folds...", flush=True)
FOLDS = []
for tr_idx, va_idx in folds:
    pre = preprocesador().fit(train.iloc[tr_idx][COLS])
    FOLDS.append({
        "Xtr": pre.transform(train.iloc[tr_idx][COLS]), "ytr": y_tr[tr_idx],
        "Xva": pre.transform(train.iloc[va_idx][COLS]), "yva": y_tr[va_idx],
        "xytr": train.iloc[tr_idx][["x_km", "y_km"]].to_numpy(),
        "xyva": train.iloc[va_idx][["x_km", "y_km"]].to_numpy(),
    })


def cv(fabrica, usa_xy=False, n_max=None, seed=C.SEED):
    """F1 macro por fold. n_max: submuestra aleatoria (uniforme) del train de cada fold (solo para la búsqueda TPE)."""
    rng = np.random.default_rng(seed)
    out = []
    for f in FOLDS:
        Xtr, ytr, xytr = f["Xtr"], f["ytr"], f["xytr"]
        if n_max and len(ytr) > n_max:
            idx = rng.choice(len(ytr), n_max, replace=False)
            Xtr, ytr, xytr = Xtr[idx], ytr[idx], xytr[idx]
        m = fabrica()
        if usa_xy:
            m.fit(Xtr, ytr, xytr)
            out.append(f1m(f["yva"], m.predict(f["Xva"], f["xyva"])))
        else:
            m.fit(Xtr, ytr)
            out.append(f1m(f["yva"], m.predict(f["Xva"])))
    return out


def regla_1se(resultados, complejidad):
    """Regla de una desviación estándar: la combinación MENOS compleja cuyo F1 medio esté a < 1 SE de la mejor."""
    medias = {k: np.mean(v) for k, v in resultados.items()}
    mejor = max(medias, key=medias.get)
    se = np.std(resultados[mejor]) / np.sqrt(len(resultados[mejor]))
    cand = [k for k in medias if medias[k] >= medias[mejor] - se]
    return min(cand, key=lambda k: (complejidad(k), -medias[k]))


# ---------------------------------------------------------------------------
# 1. XGBoost (control)
# ---------------------------------------------------------------------------
if "xgb" not in chk:
    t0 = time.time()
    res = {}
    for prof in [4, 6, 8]:
        for bal in [False, True]:
            clave = f"max_depth={prof}|balanceado={bal}"
            res[clave] = cv(lambda: XGBEstrato(max_depth=prof, balanceado=bal))
            print(f"  XGB {clave}: {np.mean(res[clave]):.4f} ± {np.std(res[clave]):.4f}", flush=True)
    elegido = regla_1se(res, lambda k: int(k.split("|")[0].split("=")[1]))
    chk["xgb"] = {"grilla": res, "elegido": elegido, "segundos": time.time() - t0}
    guardar_chk()
p = chk["xgb"]["elegido"].split("|")
HP_XGB = {"max_depth": int(p[0].split("=")[1]), "balanceado": p[1].split("=")[1] == "True"}
print("XGBoost elegido:", HP_XGB, flush=True)

# ---------------------------------------------------------------------------
# 2. XGBoost geográfico: ancho de banda (k vecinos) y peso del global (alpha)
# ---------------------------------------------------------------------------
# alpha = 1 sería el XGBoost global puro, que ya se evalúa aparte como control.
ALPHAS = [0.0, 0.25, 0.5, 0.75]
if "geoxgb" not in chk:
    t0 = time.time()
    res = {}
    info = {}
    for k in [0.05, 0.15]:   # fracción del entrenamiento del fold (≈ 5 000 y 15 000 viviendas)
        por_alpha = {a: [] for a in ALPHAS}
        cobertura, n_anclas = [], []
        for f in FOLDS:
            m = GeoXGBEstrato(params_global=HP_XGB, params_local={"balanceado": HP_XGB["balanceado"]},
                              frac_vecinos=k).fit(f["Xtr"], f["ytr"], f["xytr"])
            pg, pl, cub = m.proba_partes(f["Xva"], f["xyva"])
            cobertura.append(float(cub.mean()))
            n_anclas.append(len(m.anclas_))
            for a in ALPHAS:
                pred = C.CLASES[0] + np.argmax(a * pg + (1 - a) * pl, axis=1)
                por_alpha[a].append(f1m(f["yva"], pred))
        for a in ALPHAS:
            res[f"frac={k}|alpha={a}"] = por_alpha[a]
            print(f"  GeoXGB frac={k} alpha={a}: {np.mean(por_alpha[a]):.4f} ± {np.std(por_alpha[a]):.4f}", flush=True)
        info[str(k)] = {"cobertura_validacion": cobertura, "anclas": n_anclas}
    # Complejidad: más peso local (alpha bajo) y ventana más pequeña = más complejo.
    elegido = regla_1se(res, lambda c: (float(c.split("|")[1].split("=")[1]) * -1,
                                        -float(c.split("|")[0].split("=")[1])))
    chk["geoxgb"] = {"grilla": res, "elegido": elegido, "info": info, "segundos": time.time() - t0}
    guardar_chk()
p = chk["geoxgb"]["elegido"].split("|")
HP_GEO = {"frac_vecinos": float(p[0].split("=")[1]), "alpha": float(p[1].split("=")[1])}
print("XGBoost geográfico elegido:", HP_GEO, flush=True)

# ---------------------------------------------------------------------------
# 3. Rotation Forest
# ---------------------------------------------------------------------------
if "rotf" not in chk:
    t0 = time.time()
    res = {}
    for prof in [8, 14]:
        for bal in [False, True]:
            clave = f"max_depth={prof}|balanceado={bal}"
            res[clave] = cv(lambda: RotationForestEstrato(n_estimators=30, max_depth=prof, balanceado=bal))
            print(f"  RotF {clave}: {np.mean(res[clave]):.4f} ± {np.std(res[clave]):.4f}", flush=True)
    elegido = regla_1se(res, lambda k: int(k.split("|")[0].split("=")[1]))
    chk["rotf"] = {"grilla": res, "elegido": elegido, "segundos": time.time() - t0}
    guardar_chk()
p = chk["rotf"]["elegido"].split("|")
HP_ROT = {"n_estimators": 30, "max_depth": int(p[0].split("=")[1]), "balanceado": p[1].split("=")[1] == "True"}
print("Rotation Forest elegido:", HP_ROT, flush=True)

# ---------------------------------------------------------------------------
# 4. SVM RBF (Nyström) con búsqueda TPE (Optuna)
# ---------------------------------------------------------------------------
if "svm" not in chk:
    t0 = time.time()

    def objetivo(trial):
        params = {"gamma": trial.suggest_float("gamma", 0.005, 0.5, log=True),
                  "n_componentes": trial.suggest_categorical("n_componentes", [200, 400]),
                  "alpha": trial.suggest_float("alpha", 1e-6, 1e-3, log=True),
                  "balanceado": trial.suggest_categorical("balanceado", [False, True])}
        # La búsqueda usa una submuestra de 30 000 viviendas del train de cada fold (por costo).
        return float(np.mean(cv(lambda: SVMNystroem(**params), n_max=30_000)))

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=4))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio.optimize(objetivo, n_trials=12)
    mejores = estudio.best_params
    # CV del elegido con el train COMPLETO de cada fold (comparable con los demás modelos).
    completo = cv(lambda: SVMNystroem(**mejores))
    chk["svm"] = {"ensayos": [{"params": t.params, "f1_submuestra": t.value} for t in estudio.trials],
                  "elegido": mejores, "cv_completo": completo, "segundos": time.time() - t0}
    guardar_chk()
    print(f"  SVM elegido {mejores}: CV completo {np.mean(completo):.4f}", flush=True)
HP_SVM = chk["svm"]["elegido"]

# ---------------------------------------------------------------------------
# 4b. Criterio de selección fijado ANTES del test: diferencias pareadas por fold frente a la logística B
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression  # noqa: E402

if "logistica_cv" not in chk:
    hpb = json.loads((DATOS / "resultados.json").read_text(encoding="utf-8"))["mejores_hiperparametros"][
        "B_fisicas+ubicacion"]
    cw = hpb["clf__class_weight"]
    chk["logistica_cv"] = cv(lambda: LogisticRegression(C=float(hpb["clf__C"]), max_iter=1000,
                                                        class_weight=None if cw in (None, "None") else cw))
    guardar_chk()
print("Logística B por fold:", np.round(chk["logistica_cv"], 4), flush=True)

# ---------------------------------------------------------------------------
# 5. Entrenamiento final y evaluación ÚNICA en test
# ---------------------------------------------------------------------------
print("Entrenamiento final y test...", flush=True)
pre = preprocesador().fit(train[COLS])
Xtr, Xte = pre.transform(train[COLS]), pre.transform(test[COLS])
xytr, xyte = train[["x_km", "y_km"]].to_numpy(), test[["x_km", "y_km"]].to_numpy()

finales = {
    "XGBoost": XGBEstrato(**HP_XGB).fit(Xtr, y_tr),
    "XGBoost geográfico": GeoXGBEstrato(params_global=HP_XGB, params_local={"balanceado": HP_XGB["balanceado"]},
                                        **HP_GEO).fit(Xtr, y_tr, xytr),
    "Rotation Forest": RotationForestEstrato(**HP_ROT).fit(Xtr, y_tr),
    "SVM RBF (Nyström) + TPE": SVMNystroem(**HP_SVM, calibrar=True).fit(Xtr, y_tr),
}
# La clase predicha de la SVM sale del MISMO modelo evaluado en la CV (sin calibrar); el modelo calibrado
# (Platt) solo aporta las probabilidades para AUC, log loss y calibración.
svm_sin_calibrar = SVMNystroem(**HP_SVM).fit(Xtr, y_tr)
cv_por_modelo = {
    "XGBoost": chk["xgb"]["grilla"][chk["xgb"]["elegido"]],
    "XGBoost geográfico": chk["geoxgb"]["grilla"][chk["geoxgb"]["elegido"]],
    "Rotation Forest": chk["rotf"]["grilla"][chk["rotf"]["elegido"]],
    "SVM RBF (Nyström) + TPE": chk["svm"]["cv_completo"],
}

pred_tab = pd.read_pickle(DATOS / "test_predicciones.pkl.gz")
assert (pred_tab["estrato"].to_numpy() == y_te).all(), "El orden del test no coincide con preparar_datos.py"
res = json.loads((DATOS / "resultados.json").read_text(encoding="utf-8"))
diag = json.loads((DATOS / "diagnosticos.json").read_text(encoding="utf-8"))
bloques = test["bloque"].to_numpy()
principal = pred_tab["pred|Logística B_fisicas+ubicacion"].to_numpy()

avz = {"hiperparametros": {"XGBoost": HP_XGB, "XGBoost geográfico": HP_GEO, "Rotation Forest": HP_ROT,
                           "SVM RBF (Nyström) + TPE": HP_SVM},
       "busqueda": {k: chk[k] for k in ["xgb", "geoxgb", "rotf", "svm"]},
       "ic": {}, "comparaciones_f1": {}, "comparaciones_mae": {}}
mae = lambda y, p: float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))  # noqa: E731
residuos_mapa = {}
for nombre, est in finales.items():
    if nombre == "XGBoost geográfico":
        proba = est.predict_proba(Xte, xyte)
        _, _, cub = est.proba_partes(Xte, xyte)
        avz["cobertura_test"] = float(cub.mean())
        avz["anclas_final"] = len(est.anclas_)
    else:
        proba = est.predict_proba(Xte)
    pred = np.asarray(C.CLASES)[np.argmax(proba, axis=1)]
    if nombre == "SVM RBF (Nyström) + TPE":
        pred = svm_sin_calibrar.predict(Xte)
    pred_tab[f"pred|{nombre}"] = pred
    for j, k in enumerate(C.CLASES):
        pred_tab[f"p{k}|{nombre}"] = proba[:, j].astype("float32")
    m = M.metricas(y_te, pred, proba)
    for met, val in m.items():
        res["resultados_test"].setdefault(met, {})[nombre] = float(val)
    res["tabla_cv"]["f1_macro (media)"][nombre] = float(np.mean(cv_por_modelo[nombre]))
    res["tabla_cv"]["F1 macro (desv. entre folds)"][nombre] = float(np.std(cv_por_modelo[nombre]))
    ic = M.bootstrap_por_bloques(y_te, pred, proba, bloques, B=500)
    ic.insert(0, "valor en test", [float(m[k]) for k in ic.index])
    avz["ic"][nombre] = ic.to_dict()
    avz["comparaciones_f1"][f"{nombre} − Logística B"] = M.bootstrap_diferencia(y_te, pred, principal, bloques,
                                                                                 f1m, B=500)
    avz["comparaciones_mae"][f"{nombre} − Logística B"] = M.bootstrap_diferencia(y_te, pred, principal, bloques,
                                                                                  mae, B=500)
    # Moran de los residuos ordinales por edificio de test.
    esperado = proba @ np.asarray(C.CLASES, float)
    t = test.assign(residuo=y_te - esperado)
    e = S.deduplicar_por_edificio(t[["npn_edificio", "x_km", "y_km", "residuo"]])
    Wt, _, _ = S.pesos_knn(e[["x_km", "y_km"]].to_numpy(), k=C.K_VECINOS)
    r = S.moran_global(e["residuo"].to_numpy(), Wt, n_perm=199, seed=C.SEED)
    diag["moran_residuos"][nombre] = {"I": float(r["I"]), "p": float(r["p_perm"]),
                                      "residuo_medio": float(t["residuo"].mean())}
    g = t.assign(lat=t.centroide_lat.round(3), lon=t.centroide_lon.round(3))
    residuos_mapa[nombre] = g.groupby(["lat", "lon"], as_index=False).agg(residuo=("residuo", "mean"),
                                                                          n=("residuo", "size"))
    print(f"{nombre:28s} CV {np.mean(cv_por_modelo[nombre]):.3f} ± {np.std(cv_por_modelo[nombre]):.3f} | "
          f"test F1 {m['f1_macro']:.3f} acc {m['accuracy']:.3f} MAE {m['MAE_ordinal']:.3f} | "
          f"Moran res {r['I']:.2f}", flush=True)

# G-XGBoost frente a su control (XGBoost): ¿aporta la parte geográfica?
avz["comparaciones_f1"]["XGBoost geográfico − XGBoost"] = M.bootstrap_diferencia(
    y_te, pred_tab["pred|XGBoost geográfico"].to_numpy(), pred_tab["pred|XGBoost"].to_numpy(), bloques, f1m, B=500)
avz["comparaciones_mae"]["XGBoost geográfico − XGBoost"] = M.bootstrap_diferencia(
    y_te, pred_tab["pred|XGBoost geográfico"].to_numpy(), pred_tab["pred|XGBoost"].to_numpy(), bloques, mae, B=500)

# Importancia por ganancia del XGBoost global (nombres del preprocesador).
imp = finales["XGBoost"].modelo_.feature_importances_
avz["importancia_xgb"] = dict(zip(pre.get_feature_names_out().tolist(), imp.astype(float).tolist()))

# Regla de selección (fijada antes de mirar el test): un modelo reemplaza a la logística B solo si su F1 de CV la
# supera en más de un error estándar de la DIFERENCIA pareada por fold (desviación con ddof=1 / raíz de 5).
log_cv = np.asarray(chk["logistica_cv"])
avz["cv_pareado"] = {}
for nombre in finales:
    d = np.asarray(cv_por_modelo[nombre]) - log_cv
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    avz["cv_pareado"][nombre] = {"diferencias_por_fold": d.tolist(), "media": float(d.mean()), "se": se,
                                 "reemplaza_a_la_logistica": bool(d.mean() > se)}
avz["logistica_cv_por_fold"] = log_cv.tolist()
avz["k_usado_final"] = int(finales["XGBoost geográfico"].k_usado_)

res["avanzados"] = json.loads(json.dumps(avz, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
res["modelos_avanzados"] = list(finales)
(DATOS / "resultados.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
(DATOS / "diagnosticos.json").write_text(json.dumps(diag, ensure_ascii=False, indent=1), encoding="utf-8")
pred_tab.to_pickle(DATOS / "test_predicciones.pkl.gz")
pd.concat({k: v for k, v in residuos_mapa.items()}, names=["modelo"]).reset_index(level=0).to_pickle(
    DATOS / "residuos_mapa_avanzados.pkl.gz")
print("Listo.", flush=True)
