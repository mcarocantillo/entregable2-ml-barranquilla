"""Modelos avanzados del Entregable 2 (compatibles con scikit-learn).

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)

Contiene cuatro clasificadores para el estrato (1-6), todos pensados para ir
DESPUÉS del preprocesador del Entregable 1 (que entrega una matriz numérica):

- ``XGBEstrato``: XGBoost multiclase (control: el mismo algoritmo sin la parte
  geográfica).
- ``GeoXGBEstrato``: adaptación a CLASIFICACIÓN del XGBoost geográfico
  (Grekousis, 2025). Combina un XGBoost global con XGBoost locales entrenados
  alrededor de puntos ancla, ponderando las viviendas con un kernel espacial
  adaptativo. El artículo original es de regresión; la versión multiclase con
  mezcla de probabilidades es la adaptación propuesta en este proyecto.
- ``RotationForestEstrato``: Rotation Forest (Rodríguez, Kuncheva y Alonso,
  2006): cada árbol se entrena sobre una rotación PCA de subconjuntos de
  variables.
- ``SVMNystroem``: SVM con kernel RBF aproximado (Nyström) y entrenamiento por
  descenso de gradiente estocástico (pérdida hinge); sus hiperparámetros se
  buscan con TPE (Optuna) en ``entrenar_avanzados.py``.
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

CLASES = np.arange(1, 7)


def _proba_completa(est, X, clases_est):
    """Probabilidades en las 6 columnas de CLASES aunque el modelo no haya visto alguna clase."""
    p = est.predict_proba(X)
    out = np.zeros((len(X), len(CLASES)))
    for j, k in enumerate(clases_est):
        out[:, int(k) - 1] = p[:, j]
    return out


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
class XGBEstrato(ClassifierMixin, BaseEstimator):
    """XGBoost multiclase con pesos de clase opcionales.

    Parámetros principales: ``max_depth``, ``n_estimators``, ``learning_rate``
    y ``balanceado`` (si True, cada vivienda pesa 1/frecuencia de su clase).
    """

    def __init__(self, max_depth=6, n_estimators=300, learning_rate=0.1, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=5, balanceado=False, n_jobs=2, random_state=42):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.balanceado = balanceado
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _nuevo(self, n_clases):
        return XGBClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators,
                             learning_rate=self.learning_rate, subsample=self.subsample,
                             colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
                             tree_method="hist", n_jobs=self.n_jobs, random_state=self.random_state,
                             objective="multi:softprob" if n_clases > 2 else "binary:logistic", verbosity=0)

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        mapa = {k: i for i, k in enumerate(self.classes_)}
        w = compute_sample_weight("balanced", y) if self.balanceado else np.ones(len(y))
        if sample_weight is not None:
            w = w * sample_weight
        self.modelo_ = self._nuevo(len(self.classes_)).fit(X, np.vectorize(mapa.get)(y), sample_weight=w)
        return self

    def predict_proba(self, X):
        p = self.modelo_.predict_proba(X)
        if p.ndim == 1 or p.shape[1] == 1:
            p = np.c_[1 - p.ravel(), p.ravel()]
        return p

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _Constante:
    """Modelo local trivial para zonas donde el entrenamiento tiene un solo estrato."""

    def __init__(self, clases):
        self.classes_ = np.asarray(clases)

    def predict_proba(self, X):
        return np.ones((len(X), 1))


# ---------------------------------------------------------------------------
# XGBoost geográfico para clasificación
# ---------------------------------------------------------------------------
class GeoXGBEstrato(ClassifierMixin, BaseEstimator):
    """XGBoost geográfico adaptado a clasificación multiclase.

    Idea (Grekousis, 2025, para regresión): además de un modelo global, se
    entrenan modelos LOCALES en los que cada observación pesa según su
    distancia al punto local mediante un kernel espacial; la predicción final
    combina ambos. Adaptación propuesta aquí:

    1. Modelo global: ``XGBEstrato`` sobre todo el entrenamiento.
    2. Anclas: centroides de las celdas de ``tam_ancla_km`` con al menos
       ``min_ancla`` viviendas de entrenamiento.
    3. Ancho de banda ADAPTATIVO: para cada ancla, h = distancia a su vecino
       de entrenamiento número k, con k = ``frac_vecinos`` × n (o ``k_vecinos`` fijo); en zonas densas la
       ventana es pequeña.
    4. Kernel bicuadrado: w = (1 - (d/h)^2)^2 si d < h, 0 si no. Cada ancla
       entrena un XGBoost local (más pequeño) con esos pesos.
    5. Predicción de una vivienda: promedio de las probabilidades de las anclas
       que la cubren, ponderado por el mismo kernel; si ninguna la cubre se usa
       solo el global. Mezcla final: p = alpha·p_global + (1 − alpha)·p_local.

    Requiere que las columnas ``col_x`` y ``col_y`` (km) de X sean las
    coordenadas SIN escalar; por eso recibe aparte ``xy`` en fit y predict.
    """

    def __init__(self, params_global=None, params_local=None, k_vecinos=8000, frac_vecinos=None,
                 tam_ancla_km=2.0, min_ancla=300, alpha=0.5, random_state=42):
        self.params_global = params_global
        self.params_local = params_local
        self.k_vecinos = k_vecinos
        self.frac_vecinos = frac_vecinos
        self.tam_ancla_km = tam_ancla_km
        self.min_ancla = min_ancla
        self.alpha = alpha
        self.random_state = random_state

    @staticmethod
    def _kernel(d, h):
        u = d / h
        return np.where(u < 1, (1 - u ** 2) ** 2, 0.0)

    def fit(self, X, y, xy):
        y = np.asarray(y)
        xy = np.asarray(xy, float)
        self.classes_ = CLASES
        self.global_ = XGBEstrato(**(self.params_global or {})).fit(X, y)
        # Anclas: centroide de las viviendas de cada celda con suficiente entrenamiento.
        celda = np.floor(xy / self.tam_ancla_km).astype(int)
        _, inv, cuenta = np.unique(celda, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        anclas = np.array([xy[inv == g].mean(0) for g in np.where(cuenta >= self.min_ancla)[0]])
        arbol = cKDTree(xy)
        # Con frac_vecinos la ventana es una FRACCIÓN del entrenamiento: así se traslada igual de la CV
        # (folds más pequeños) al modelo final (todo el entrenamiento).
        k = int(self.frac_vecinos * len(xy)) if self.frac_vecinos else self.k_vecinos
        k = min(k, len(xy))
        self.k_usado_ = k
        d_k, _ = arbol.query(anclas, k=[k])
        self.anclas_, self.h_ = anclas, d_k.ravel()
        self.locales_ = []
        pl = dict(max_depth=4, n_estimators=150, learning_rate=0.1)
        pl.update(self.params_local or {})
        for a, h in zip(self.anclas_, self.h_):
            idx = np.array(arbol.query_ball_point(a, r=h))
            w = self._kernel(np.linalg.norm(xy[idx] - a, axis=1), h)
            sel = w > 0
            if len(np.unique(y[idx][sel])) < 2:
                # Zona con un solo estrato: el modelo local es esa clase con probabilidad 1.
                self.locales_.append(_Constante(np.unique(y[idx][sel])))
                continue
            m = XGBEstrato(**pl, random_state=self.random_state).fit(X[idx][sel], y[idx][sel],
                                                                    sample_weight=w[sel])
            self.locales_.append(m)
        return self

    def proba_partes(self, X, xy):
        """Devuelve (p_global, p_local, cubierto): útil para afinar alpha sin reentrenar."""
        xy = np.asarray(xy, float)
        pg = _proba_completa(self.global_, X, self.global_.classes_)
        num = np.zeros_like(pg)
        den = np.zeros(len(X))
        for a, h, m in zip(self.anclas_, self.h_, self.locales_):
            w = self._kernel(np.linalg.norm(xy - a, axis=1), h)
            sel = w > 0
            if sel.any():
                num[sel] += w[sel, None] * _proba_completa(m, X[sel], m.classes_)
                den[sel] += w[sel]
        cubierto = den > 0
        pl = pg.copy()
        pl[cubierto] = num[cubierto] / den[cubierto, None]
        return pg, pl, cubierto

    def predict_proba(self, X, xy):
        pg, pl, _ = self.proba_partes(X, xy)
        return self.alpha * pg + (1 - self.alpha) * pl

    def predict(self, X, xy):
        return CLASES[np.argmax(self.predict_proba(X, xy), axis=1)]


# ---------------------------------------------------------------------------
# Rotation Forest
# ---------------------------------------------------------------------------
class RotationForestEstrato(ClassifierMixin, BaseEstimator):
    """Rotation Forest (Rodríguez, Kuncheva y Alonso, 2006).

    Para cada árbol: se parten las variables al azar en ``n_subconjuntos``;
    en cada subconjunto se ajusta un PCA sobre una muestra bootstrap (75 %) de
    un subconjunto aleatorio de clases; las cargas forman una matriz de
    rotación por bloques y el árbol se entrena sobre los datos rotados. La
    rotación hace que los cortes del árbol sean oblicuos respecto a las
    variables originales, útil con variables correlacionadas (área, baños,
    habitaciones).
    """

    def __init__(self, n_estimators=30, n_subconjuntos=3, max_depth=12, min_samples_leaf=20,
                 balanceado=False, random_state=42):
        self.n_estimators = n_estimators
        self.n_subconjuntos = n_subconjuntos
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.balanceado = balanceado
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        p = X.shape[1]
        self.rotaciones_, self.medias_, self.arboles_ = [], [], []
        for _ in range(self.n_estimators):
            grupos = np.array_split(rng.permutation(p), self.n_subconjuntos)
            R = np.zeros((p, p))
            for g in grupos:
                clases = rng.choice(self.classes_, size=max(2, rng.integers(2, len(self.classes_) + 1)),
                                    replace=False)
                filas = np.where(np.isin(y, clases))[0]
                filas = rng.choice(filas, size=min(len(filas), max(len(g) + 1, int(0.75 * len(filas)))),
                                   replace=True)
                filas = filas[: 20_000]  # el PCA solo necesita una muestra
                pca = PCA(random_state=int(rng.integers(1e9))).fit(X[np.ix_(filas, g)])
                comp = pca.components_
                R[np.ix_(g, g[: comp.shape[0]])] = comp.T
            media = X.mean(0)
            w = compute_sample_weight("balanced", y) if self.balanceado else None
            arbol = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                           random_state=int(rng.integers(1e9)))
            arbol.fit((X - media) @ R, y, sample_weight=w)
            self.rotaciones_.append(R)
            self.medias_.append(media)
            self.arboles_.append(arbol)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        p = np.zeros((len(X), len(self.classes_)))
        for R, media, arbol in zip(self.rotaciones_, self.medias_, self.arboles_):
            pa = arbol.predict_proba((X - media) @ R)
            for j, k in enumerate(arbol.classes_):
                p[:, np.searchsorted(self.classes_, k)] += pa[:, j]
        return p / len(self.arboles_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# SVM con kernel RBF aproximado (Nyström)
# ---------------------------------------------------------------------------
class SVMNystroem(ClassifierMixin, BaseEstimator):
    """SVM multiclase (uno contra el resto) con kernel RBF aproximado.

    Nyström proyecta los datos a ``n_componentes`` dimensiones donde el
    producto interno aproxima el kernel RBF de parámetro ``gamma``; sobre esa
    proyección se entrena una SVM lineal (pérdida hinge) con descenso de
    gradiente estocástico. ``alpha`` es la regularización (equivale a 1/(n·C)).
    Si ``calibrar`` es True, las probabilidades se obtienen con calibración
    sigmoide (Platt) por validación cruzada interna.
    """

    def __init__(self, gamma=0.05, n_componentes=400, alpha=1e-4, balanceado=False, calibrar=False,
                 random_state=42):
        self.gamma = gamma
        self.n_componentes = n_componentes
        self.alpha = alpha
        self.balanceado = balanceado
        self.calibrar = calibrar
        self.random_state = random_state

    def fit(self, X, y):
        svm = SGDClassifier(loss="hinge", alpha=self.alpha, max_iter=20, tol=None,
                            class_weight="balanced" if self.balanceado else None, random_state=self.random_state)
        base = make_pipeline(Nystroem(gamma=self.gamma, n_components=self.n_componentes,
                                      random_state=self.random_state), svm)
        self.modelo_ = (CalibratedClassifierCV(base, method="sigmoid", cv=3) if self.calibrar else base).fit(X, y)
        self.classes_ = self.modelo_.classes_
        return self

    def predict(self, X):
        return self.modelo_.predict(X)

    def predict_proba(self, X):
        if self.calibrar:
            return self.modelo_.predict_proba(X)
        # Sin calibrar: softmax de las distancias al hiperplano (solo para ordenar).
        d = self.modelo_.decision_function(X)
        e = np.exp(d - d.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
