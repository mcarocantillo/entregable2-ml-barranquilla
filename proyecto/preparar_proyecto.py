"""Precalcula los análisis del Entregable 1 que muestra el dashboard final del proyecto.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Reproduce, con el mismo código y las mismas semillas del Entregable 1 (versión 2), los análisis cuyo resultado
necesita datos fila a fila (PCA, outliers multivariados, K-Means, t-SNE, correlaciones, LISA, mapa de la
partición) y guarda solo lo que el dashboard dibuja, en ``datos/proyecto/``. Así la app
desplegada no recalcula nada pesado.

Uso:  python proyecto/preparar_proyecto.py RUTA_AL_ENTREGABLE1 [--grid]
      --grid  además repite la búsqueda de hiperparámetros del capítulo 10 (C x pesos de clase, A y B), ~30 min.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent.parent
E1 = Path(sys.argv[1]).resolve()
SALIDA = AQUI / "datos" / "proyecto"
SALIDA.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(E1))
from src import config as C  # noqa: E402
from src import espacial as S  # noqa: E402
from src import estadistica as E  # noqa: E402
from src import particion as P  # noqa: E402

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.covariance import MinCovDet  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
t00 = time.time()
dec = C.leer_decisiones()
datos = P.cargar_particion()
train = datos[datos.particion == "train"].reset_index(drop=True)
test = datos[datos.particion == "test"].reset_index(drop=True)
num_log = dec["num_log"]["valor"]
out = {}


def guardar_json(nombre, obj):
    (SALIDA / nombre).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Multivariado (capítulo 7): PCA, outliers, K-Means y t-SNE
# ---------------------------------------------------------------------------
vars_pca = [c for c in C.NUMERICAS if c in train]
Z = train[vars_pca].copy()
for c in num_log:
    if c in Z:
        Z[c] = np.log1p(Z[c].clip(lower=0))
Z = Z.fillna(Z.median())
Zs = StandardScaler().fit_transform(Z)
pca = PCA(random_state=C.SEED).fit(Zs)
ev = pca.explained_variance_
mv = {"variables": vars_pca, "varianza": pca.explained_variance_ratio_.tolist(), "autovalor": ev.tolist(),
      "cargas": pca.components_.tolist(),
      "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum())}
pc_all = pca.transform(Zs)[:, :4]
mv["spearman_pc_estrato"] = [float(stats.spearmanr(pc_all[:, i], train[C.OBJETIVO])[0]) for i in range(4)]

rng = np.random.default_rng(C.SEED)
sub = rng.choice(len(Zs), min(30_000, len(Zs)), replace=False)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Determinant has increased", category=RuntimeWarning)
    mcd = MinCovDet(random_state=C.SEED, support_fraction=0.9).fit(Zs[sub])
out_maha = mcd.mahalanobis(Zs) > stats.chi2.ppf(0.999, df=Zs.shape[1])
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=C.SEED, n_jobs=-1).fit(Zs[sub])
out_iso = iso.predict(Zs) == -1
ambos = out_maha & out_iso
mv["outliers"] = {"pct_mahalanobis": float(100 * out_maha.mean()), "pct_isolation": float(100 * out_iso.mean()),
                  "pct_ambos": float(100 * ambos.mean()),
                  "jaccard": float(ambos.sum() / max(1, (out_maha | out_iso).sum())),
                  "estrato_outliers": train.loc[ambos, C.OBJETIVO].value_counts(normalize=True).sort_index().mul(100).to_dict(),
                  "estrato_resto": train.loc[~ambos, C.OBJETIVO].value_counts(normalize=True).sort_index().mul(100).to_dict()}
print(f"multivariado: PCA y outliers listos ({time.time() - t00:.0f}s)", flush=True)

Xs = Zs[sub]
km_por_k, sil = {}, {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=5, random_state=C.SEED).fit(Xs)
    km_por_k[k] = km
    sil[k] = float(silhouette_score(Xs, km.labels_, sample_size=8000, random_state=C.SEED))
k_opt = max(sil, key=sil.get)
mv["silueta"] = sil
mv["k_opt"] = k_opt
km = KMeans(n_clusters=k_opt, n_init=10, random_state=C.SEED).fit(Xs)
mv["centroides"] = km.cluster_centers_.tolist()
t_sub = train.iloc[sub]
tab = pd.crosstab(km.labels_, t_sub[C.OBJETIVO].to_numpy(), normalize="index").mul(100)
mv["cluster_estrato"] = {"filas": tab.index.tolist(), "columnas": [int(c) for c in tab.columns], "valores": tab.to_numpy().tolist(),
                         "n": pd.Series(km.labels_).value_counts().sort_index().tolist(),
                         "V_cramer": float(E.cramers_v(pd.Series(km.labels_), t_sub[C.OBJETIVO].reset_index(drop=True))["V"])}
guardar_json("multivariado.json", mv)

idx = S.muestra_estratificada(train.assign(_i=np.arange(len(train))), 8000)["_i"].to_numpy()
m = pd.DataFrame(pc_all[idx], columns=["PC1", "PC2", "PC3", "PC4"]).astype("float32")
m["estrato"] = train[C.OBJETIVO].to_numpy()[idx].astype("int8")
m["condicion_predio"] = train["condicion_predio"].astype(str).to_numpy()[idx]
m["outlier"] = np.select([ambos[idx], out_maha[idx], out_iso[idx]],
                         ["ambos métodos", "solo Mahalanobis", "solo Isolation Forest"], "no")
for k, mk in km_por_k.items():
    m[f"k{k}"] = mk.predict(Zs[idx]).astype("int8")
m.to_pickle(SALIDA / "multivariado_muestra.pkl.gz")

s3 = rng.choice(len(Zs), min(3000, len(Zs)), replace=False)
emb = TSNE(n_components=2, perplexity=40, random_state=C.SEED, init="pca").fit_transform(Zs[s3])
ts = pd.DataFrame(emb, columns=["d1", "d2"]).astype("float32")
ts["estrato"] = train[C.OBJETIVO].to_numpy()[s3].astype("int8")
ts["condicion_predio"] = train["condicion_predio"].astype(str).to_numpy()[s3]
ts["cluster"] = km.predict(Zs[s3]).astype("int8")
ts.to_pickle(SALIDA / "tsne.pkl.gz")
print(f"multivariado: K-Means y t-SNE listos ({time.time() - t00:.0f}s)", flush=True)

# ---------------------------------------------------------------------------
# 2. Correlaciones entre numéricas (capítulo 6.1), en train
# ---------------------------------------------------------------------------
num = train[vars_pca]
guardar_json("correlaciones.json", {"variables": vars_pca,
                                    "pearson": num.corr(method="pearson").round(4).to_numpy().tolist(),
                                    "spearman": num.corr(method="spearman").round(4).to_numpy().tolist()})

# ---------------------------------------------------------------------------
# 3. LISA y Gi* (capítulo 8.4): una fila por edificio de una muestra de train
# ---------------------------------------------------------------------------
muestra = S.muestra_estratificada(train, C.N_MUESTRA_ESPACIAL)
edif = S.deduplicar_por_edificio(muestra)
xy = edif[["x_km", "y_km"]].to_numpy()
W, dist_vec, idx_vec = S.pesos_knn(xy, k=C.K_VECINOS)
lis = S.lisa(edif[C.OBJETIVO].to_numpy(), idx_vec, n_perm=199, seed=C.SEED)
gi = S.getis_ord_gi_star(edif[C.OBJETIVO].to_numpy(), idx_vec)
li = pd.DataFrame({"lat": edif["centroide_lat"].round(4), "lon": edif["centroide_lon"].round(4),
                   "estrato": edif[C.OBJETIVO].round(2), "cuadrante": lis["cuadrante"].to_numpy(),
                   "gi_z": np.asarray(gi).round(3)})
li.to_pickle(SALIDA / "lisa.pkl.gz")
print(f"LISA: {len(li):,} edificios ({time.time() - t00:.0f}s)", flush=True)

# ---------------------------------------------------------------------------
# 4. Mapa de la partición y de los folds de la validación cruzada
# ---------------------------------------------------------------------------
buffer_km = dec["buffer_km"]["valor"]
mantener = P.mascara_buffer(train[["x_km", "y_km"]].to_numpy(), test[["x_km", "y_km"]].to_numpy(), buffer_km)
etiqueta = np.where(mantener, "entrenamiento", "excluida por el buffer")
tr_buf = train[mantener].reset_index(drop=True)
folds = P.folds_espaciales_con_buffer(tr_buf, buffer_km, verbose=False)
fold = np.full(len(tr_buf), -1)
for i, (_, iva) in enumerate(folds):
    fold[iva] = i + 1
mapa = pd.concat([
    pd.DataFrame({"lat": train.centroide_lat, "lon": train.centroide_lon, "grupo": etiqueta, "fold": 0,
                  "bloque": train.bloque.astype(str)}),
    pd.DataFrame({"lat": test.centroide_lat, "lon": test.centroide_lon, "grupo": "prueba (test)", "fold": 0,
                  "bloque": test.bloque.astype(str)}),
], ignore_index=True)
mapa.loc[np.flatnonzero(mantener), "fold"] = fold
mapa["lat"], mapa["lon"] = mapa["lat"].round(3), mapa["lon"].round(3)
mapa = mapa.groupby(["lat", "lon", "grupo", "fold"], as_index=False).agg(n=("bloque", "size"), bloque=("bloque", "first"))
mapa.to_pickle(SALIDA / "particion_mapa.pkl.gz")
print(f"partición: {len(mapa):,} puntos de mapa ({time.time() - t00:.0f}s)", flush=True)

# ---------------------------------------------------------------------------
# 5. (opcional) Búsqueda de hiperparámetros del capítulo 10: F1 de CV por combinación
# ---------------------------------------------------------------------------
if "--grid" in sys.argv:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import Pipeline
    from src import modelo as M
    vm = dec["variables_modelo"]["valor"]
    q_inf, q_sup = dec["winsorizacion_cuantiles"]["valor"]
    mf = dec["onehot_min_frecuencia"]["valor"]
    y_tr = tr_buf[C.OBJETIVO].to_numpy()
    conjuntos = {"A_fisicas": vm["numericas"], "B_fisicas+ubicacion": vm["numericas"] + vm["espaciales"]}
    filas = []
    for nombre, nums in conjuntos.items():
        cols = nums + vm["categoricas"]
        for C_ in [0.01, 0.1, 1, 10]:
            for cw in [None, "balanced"]:
                prep = M.construir_preprocesador([c for c in nums if c in num_log], [c for c in nums if c not in num_log],
                                                 vm["categoricas"], q_inf=q_inf, q_sup=q_sup, min_frecuencia=mf)
                pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(C=C_, class_weight=cw, max_iter=1000))])
                r = cross_validate(pipe, tr_buf[cols], y_tr, cv=folds, scoring="f1_macro", n_jobs=-1,
                                   return_train_score=True)
                filas.append({"conjunto": nombre, "C": C_, "pesos": cw or "sin pesos",
                              "f1_cv": r["test_score"].tolist(), "f1_train": r["train_score"].tolist()})
                print(f"  {nombre} C={C_} {cw}: {r['test_score'].mean():.4f} ({time.time() - t00:.0f}s)", flush=True)
    guardar_json("grid_cv.json", filas)

print(f"Listo en {time.time() - t00:.0f}s:", sorted(p.name for p in SALIDA.iterdir()))
