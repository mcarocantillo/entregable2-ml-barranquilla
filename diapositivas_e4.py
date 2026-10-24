"""Diapositivas de la pestaña del Entregable 4 (proyecto final): el modelo original EPL y su validación.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Mismo patrón que ``diapositivas.py`` (``componentes.diapositiva``: visual + «Lo que se ve» + «En una frase» + fichas);
los datos y las figuras están en ``figuras_e4.py``. La predicción en vivo usa el EPL serializado del E4.
"""
import json
import re

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update
from scipy.spatial import cKDTree

import figuras_e4 as F4
import figuras_mejoras as FM
from componentes import control, controles, desplegable, diapositiva, fmt, grafico, kpi, radio, tabla

E4C = "Entregable 4"
LIBRO_E4 = "https://mcarocantillo.github.io/entregable4-ml-barranquilla/"
ORIGINAL = F4.ORIGINAL
CATEGORICAS = ["tipo_vivienda", "uso", "condicion_predio"]
FISICAS = ["area_construida", "area_catastral_terreno", "total_habitaciones", "total_banios", "total_plantas",
           "antiguedad", "planta_ubicacion", "altura"]
ENTRADAS = FISICAS + CATEGORICAS + ["x_km", "y_km"]

PSEUDOCODIGO = """Entrada: datos (X, s, y); M expertos; J anclas; ancho h; k anclas vecinas; λ; espacio de competencia
Salida:  expertos f_1..f_M; anclas a_1..a_J; pesos W = (w_1..w_J), cada w_j en el símplex Δ_M

1  Bloques espaciales de 1 km → B pliegues
2  para cada pliegue y experto m: ajustar f_m sin el pliegue; q_im ← P_m(y_i | x_i)   (fuera de muestra)
3  reajustar cada f_m con todos los datos
4  z_i ← coordenadas de competencia (geográficas | 2 CP de x | clase esperada y desacuerdo)
5  a_1..a_J ← K-medias(z);   K_ij ← núcleo gaussiano normalizado sobre las k anclas más cercanas
6  π ← argmin_{w∈Δ_M}  −Σ_i α_i log(wᵀq_i)                   (gradiente exponenciado, J = 1)
7  W ← π en cada ancla;  η ← η_0
8  repetir
9      g_j ← −Σ_i α_i K_ij q_i / r_i  +  (λ/J)·log(w_j / π)        con  r_i = Σ_j K_ij w_jᵀ q_i
10     w_j ← w_j ⊙ exp(−η g_j) / Σ (w_j ⊙ exp(−η g_j))
11     si L(W) no bajó: deshacer y η ← η/2;  si bajó: η ← 1.5 η
12 hasta que la mejora relativa de L(W) < tol
Predicción: w(z) = Σ_j K_j(z) w_j;   p(y | x) = Σ_m w_m(z) f_m(y | x)"""

DIFERENCIAS = pd.DataFrame([
    ("Promedio simple o voto", "Pesos iguales", "Pesos aprendidos y distintos según la región"),
    ("Pesos globales por gradiente exponenciado (Helmbold et al., 1997)", "Un solo vector de pesos",
     "Campo de pesos con núcleo y penalización KL hacia los globales"),
    ("Stacking lineal (Wolpert, 1992)", "Regresión sobre las salidas",
     "Pesos en el símplex (interpretables), problema convexo, dependen de la región"),
    ("Selección dinámica (KNORA, META-DES)", "Elige modelos por su acierto en vecinos",
     "Pesos suaves que minimizan una pérdida global convexa; ajuste cruzado por bloques"),
    ("Mezcla de expertos (Jacobs et al., 1991)", "Compuerta y expertos entrenados juntos (no convexo)",
     "Expertos fijos ya validados; óptimo único; compuerta no neuronal"),
    ("Ensamble con ponderación espacial (Liu, 2024)", "Ponderación geográfica local + explicabilidad",
     "Espacio de competencia elegido por validación; clasificación ordinal multiclase; regla derivada"),
    ("Combinación convexa por regiones (Zhou y Lin, 2026)", "Mezcla convexa distinta por región (regresión)",
     "Clasificación ordinal; regiones suaves con núcleo; encogimiento KL; espacio elegido por validación"),
    ("XGBoost geográfico (E2 y E3)", "Mezcla fija α entre global y locales",
     "Pesos aprendidos por ancla y por experto, con cualquier familia de expertos"),
], columns=["Método", "Qué hace", "Qué es nuevo en el EPL"])


def _m():
    return F4.modelos()


def _ver():
    return _m()["version_original"]


def _mejor():
    return _m()["estadistica"][_ver()]["veredicto_validacion"]["mejor_benchmark"]


def _media(modelo):
    return next(x["media"] for x in _m()["estadistica"][_ver()]["resumen"] if x["modelo"] == modelo)


# ---------------------------------------------------------------------------
# 1. Qué pide el E4 y correcciones del E3
# ---------------------------------------------------------------------------
def d_e4_resumen():
    b = F4.leer("base_vs_epl.json")
    x = F4.rival(_mejor())
    pasos = [("Corregir el E3", "18 correcciones; validación repetida 5 × 5 con las mismas particiones para todos"),
             ("Modelo original", "Ensamble con pesos locales (EPL): programado con numpy y scipy, 30 pruebas unitarias"),
             ("Novedad", "Scopus, Web of Science, Google Scholar y arXiv; 6 cadenas"),
             ("Superioridad", "Friedman-Nemenyi, Wilcoxon-Holm, t de Nadeau-Bengio, bayesiana, δ de Cliff y d pareada"),
             ("Confirmar", "Test de 65 145 viviendas, 3 conjuntos externos, robustez y costo"),
             ("Entregar", "Modelos serializados con metadata, libro y dashboard")]
    cajas = html.Div([html.Div([html.Div(f"{i + 1}", className="paso-num"), html.Div(t, className="paso-titulo"),
                                html.Div(d, className="paso-desc")], className="paso") for i, (t, d) in enumerate(pasos)],
                     className="flujo")
    cifras = dbc.Row([
        dbc.Col(kpi(f"{_media(ORIGINAL):.3f}", "F1 del EPL en la 5 × 5", "el mayor de los 11 modelos", "#eb6834"), md=3),
        dbc.Col(kpi(f"{x['dif_media']:+.3f}", f"frente a {_mejor()}", f"gana en {x['pliegues_a_favor']} de 25 pliegues"),
                md=3),
        dbc.Col(kpi(f"{b['epl_vs_base']['dif_media']:+.3f}", "frente al modelo base del E1",
                    f"{b['epl_vs_base']['pliegues_a_favor']} de 25 pliegues"), md=3),
        dbc.Col(kpi("R1 ✓ · R2 ✓ · R3 ✗", "condiciones de la guía", "tope de la nota: 4.0"), md=3)], className="g-2 mt-3")
    return diapositiva("8.1", "El Entregable 4: un modelo original y la prueba de que es mejor", html.Div([cajas, cifras]),
                       puntos=["La guía pide un modelo original con tres condiciones: novedad (R1), implementación propia "
                               "(R2) y superioridad estadística (R3) frente a los benchmarks.",
                               "La propuesta: combinar los 7 modelos de la guía con pesos que cambian según la región, "
                               "aprendidos con predicciones fuera de muestra.",
                               f"Tiene el mayor F1 medio ({_media(ORIGINAL):.3f}), pero frente a {_mejor()} solo una de "
                               "las tres pruebas rechaza: R3 no se cumple.",
                               "Frente al modelo base del E1 la mejora es grande y la confirman todas las pruebas."],
                       conclusion="R1 y R2 se cumplen; R3 no, así que el tope de la nota es 4.0. Se reporta tal cual.",
                       metricas=["f1_macro"], fuente=f"{E4C} · caps. 0 y 8")


def d_e4_correcciones():
    c = F4.leer("correcciones.json")
    df = pd.DataFrame(c["filas"], columns=c["columnas"])[["#", "Observación", "Acción realizada", "Ubicación en el libro"]]
    return diapositiva("8.2", "Correcciones del Entregable 3", tabla(df, alto=520),
                       puntos=["Cada fila: la debilidad del E3, qué se hizo, dónde está en el libro del E4 y con qué "
                               "evidencia.",
                               "La más importante: con 5 pliegues ninguna prueba pareada tiene potencia (el p mínimo de "
                               "Wilcoxon es 0.0625). Por eso la validación repetida 5 × 5.",
                               "Los IC del test pasan a BCa y fijan las 6 clases en cada remuestreo.",
                               "Varias correcciones bajan cifras que el E3 había publicado (filas 2, 5 y 17)."],
                       conclusion="Se corrigió lo que inflaba los resultados, aunque las cifras bajaran.",
                       metricas=["ic95"], fuente=f"{E4C} · cap. 1")


def d_e4_e3_vs_5x5():
    e3 = {r["modelo"]: r["f1_macro_mean"] for r in FM.e3("analisis_clasificacion")["mejor_por_modelo"]}
    lid = {r["modelo"]: r["f1_cv"] for r in _m()["lideres"]}
    caida = [lid[m] - v for m, v in e3.items() if m in lid]
    p = F4.leer("particiones_resumen.json")
    acuerdo = np.mean([np.mean(f[1:]) for f in p["acuerdo_pares_mismo_pliegue"][:1]])
    return diapositiva("8.3", "Validación repetida 5 × 5: las cifras del E3 bajan", grafico(F4.fig_e3_vs_5x5(e3)),
                       puntos=["Repetición 0 = los pliegues kNNDM del E3; repeticiones 1 a 4 = nuevas particiones con "
                               "semillas 43 a 46, las mismas para los 11 modelos.",
                               f"Dos viviendas que compartían pliegue en el E3 lo comparten en otra repetición solo el "
                               f"{100 * acuerdo:.0f} % de las veces: las particiones sí son distintas.",
                               f"Todos los modelos bajan entre {-max(caida):.2f} y {-min(caida):.2f} de F1: en el E3 la "
                               "mejor combinación se eligió y se midió con los mismos 5 pliegues.",
                               "El orden se conserva: los ensambles de árboles siguen arriba y Naive Bayes abajo."],
                       conclusion="Elegir y reportar con los mismos pliegues infla el desempeño; la 5 × 5 lo corrige.",
                       metricas=["knndm"], fuente=f"{E4C} · caps. 1 y 3")


# ---------------------------------------------------------------------------
# 2. El modelo original
# ---------------------------------------------------------------------------
def d_e4_idea():
    c = _m()["campo_epl"]
    visual = dbc.Row([
        dbc.Col(html.Pre(PSEUDOCODIGO, className="pseudocodigo"), lg=7),
        dbc.Col(grafico(F4.fig_convergencia()), lg=5)], className="g-2")
    return diapositiva("8.4", "El modelo original: ensamble con pesos locales (EPL)", visual,
                       puntos=["Combina las probabilidades de los 7 modelos de la guía con pesos que cambian según la "
                               "región del espacio de competencia (las «anclas»).",
                               "Los pesos minimizan una log-pérdida ponderada por clase más una penalización KL hacia los "
                               "pesos globales: problema convexo, estrictamente convexo con λ > 0 (óptimo único).",
                               "Se aprenden con predicciones fuera de muestra por bloques de 1 km (ajuste cruzado): así no "
                               "se premia al experto que memoriza.",
                               f"Modelo final: espacio {F4.NOMBRE_ESPACIO[c['espacio']]}, J = {c['params']['n_anclas']} "
                               f"anclas, λ = {c['params']['lam']:.2f}. La pérdida baja en cada iteración."],
                       conclusion="Una regla de combinación local, convexa y entrenada sin fuga de información.",
                       fuente=f"{E4C} · cap. 5", ancho_visual=9)


def _md(texto):
    """Markdown con MathJax; las ecuaciones de bloque necesitan los $$ en líneas propias (remark-math)."""
    texto = re.sub(r"^\$\$(.+)\$\$$", r"$$\n\1\n$$", texto, flags=re.M)
    return dcc.Markdown(texto, mathjax=True, className="formulas")


FORMULACION_EPL = r"""
**Expertos.** $M = 7$ modelos de la guía con probabilidades $P_m(y\mid x)$ sobre $K = 6$ estratos.

**Pesos que dependen de la región.** $z(x)$ = coordenadas en el *espacio de competencia*; $J$ anclas $a_j$ (K-medias)
con un vector de pesos $w_j \in \Delta_M$ cada una:

$$w(z) = \sum_{j \in N_k(z)} \kappa_j(z)\, w_j, \qquad \kappa_j(z) = \frac{e^{-\lVert z - a_j\rVert^2 / 2h^2}}{\sum_{l \in N_k(z)} e^{-\lVert z - a_l\rVert^2 / 2h^2}}$$

**Predicción** (una mezcla convexa: sigue siendo una distribución):

$$p(y \mid x) = \sum_{m=1}^{M} w_m\big(z(x)\big)\, P_m(y \mid x)$$

**Función de pérdida.** Con $q_{im} = \max\{P_m^{\text{oof}}(y_i \mid x_i), \varepsilon\}$ (probabilidad *fuera de muestra*
de la clase observada), $\alpha_i$ el peso de su clase y $r_i(W) = \sum_j K_{ij}\, w_j^\top q_i$:

$$L(W) = -\frac{1}{\sum_i \alpha_i} \sum_{i=1}^{n} \alpha_i \log r_i(W) \;+\; \frac{\lambda}{J} \sum_{j=1}^{J} \mathrm{KL}(w_j \,\Vert\, \pi)$$

$\pi$ = pesos globales (el mismo problema con $J = 1$, $\lambda = 0$).
"""

OPTIMIZACION_EPL = r"""
**Principio de aprendizaje: gradiente exponenciado** (descenso espejo con la entropía; mantiene cada $w_j$ en el
símplex sin proyecciones):

$$g_{jm} = \frac{\partial L}{\partial w_{jm}} = -\frac{1}{\sum_i \alpha_i} \sum_i \alpha_i \frac{K_{ij}\, q_{im}}{r_i} + \frac{\lambda}{J}\Big(\log\frac{w_{jm}}{\pi_m} + 1\Big)$$

$$w_{jm}^{(t+1)} = \frac{w_{jm}^{(t)}\, e^{-\eta g_{jm}}}{\sum_{m'} w_{jm'}^{(t)}\, e^{-\eta g_{jm'}}}$$

Paso adaptativo: si $L$ no baja, $\eta \leftarrow \eta/2$ y se repite; si baja, $\eta \leftarrow 1.5\,\eta$.

**Propiedades (demostradas en el libro, cap. 5.3):**

1. *Convexidad y unicidad.* $L$ es convexa en $\Delta_M^J$ ($-\log$ de una función lineal positiva, más KL); con
   $\lambda > 0$ es estrictamente convexa: el mínimo existe y es **único**.
2. *Descenso monótono.* $L(W^{(t)})$ no crece y converge; con paso fijo,
   $L(\bar W_T) - L^\star \le \dfrac{J \log M}{\eta T} + \dfrac{\eta G^2}{2}$.
3. *Nunca peor que el mejor experto* en la log-pérdida fuera de muestra: $L(W^\star) \le L_{\text{datos}}(\pi) \le \min_m L_{\text{datos}}(e_m)$.
4. *Complejidad.* Entrenar: $O\big((B+1)\sum_m C_m + nJdR + TnkM\big)$; predecir: $O\big(\sum_m I_m + \log J + kM\big)$.
"""

GENERALIZACION_EPL = r"""
**Regularización.** Tres frenos al sobreajuste del combinador: (i) $\lambda\,\mathrm{KL}(w_j\Vert\pi)$ encoge hacia los
pesos globales las anclas con pocos datos; (ii) el núcleo ($h$, $k$) suaviza los pesos entre anclas vecinas; (iii)
pocas regiones ($J$) frente a $n \approx 8\,000$ viviendas: $J \times M \le 560$ parámetros.

**Generalización.** Los pesos se aprenden con predicciones **fuera de muestra por bloques espaciales de 1 km**
(ajuste cruzado): sin él, el combinador premia al experto que memoriza y el F1 cae 0.054.

**Supuestos.** (a) Las probabilidades fuera de muestra imitan las que el experto dará en zonas nuevas; (b) las
viviendas de bloques distintos son aproximadamente independientes; (c) $q_{im} \ge \varepsilon$ (el piso acota el
gradiente); (d) la competencia de cada experto varía de forma suave en el espacio $z$.
"""


def d_e4_formulacion():
    return diapositiva("8.5", "Fundamentos (1): el modelo y su función de pérdida", _md(FORMULACION_EPL),
                       puntos=["Cada vivienda recibe su propio vector de pesos sobre los 7 expertos, mezcla suave de los "
                               "pesos de las anclas cercanas en el espacio de competencia.",
                               "La pérdida es la log-verosimilitud negativa ponderada por clase (para el F1 macro) de la "
                               "mezcla, evaluada con probabilidades fuera de muestra.",
                               "El término λ·KL es la regularización: acerca los pesos de cada ancla a los globales π.",
                               "Versión 2: z es geográfico (km), 2 componentes principales de las variables o la clase "
                               "esperada y el desacuerdo entre expertos, según elija la validación."],
                       conclusion="Un problema de máxima verosimilitud regularizada sobre el símplex.",
                       fuente=f"{E4C} · cap. 5.2", ancho_visual=9)


def d_e4_optimizacion():
    visual = dbc.Row([dbc.Col(_md(OPTIMIZACION_EPL), lg=8), dbc.Col(grafico(F4.fig_convergencia(), alto=340), lg=4)],
                     className="g-2")
    return diapositiva("8.6", "Fundamentos (2): optimización y propiedades", visual,
                       puntos=["El gradiente exponenciado es el descenso espejo natural del símplex: actualiza los pesos "
                               "multiplicativamente y los renormaliza.",
                               "Por convexidad, no hay mínimos locales: el resultado no depende del punto de partida.",
                               "La proposición 3 garantiza que el combinador no empeora la log-pérdida del mejor experto "
                               "en el entrenamiento; no garantiza un mejor F1 en datos nuevos (eso se prueba después).",
                               "El costo lo dominan los expertos (el término (B + 1)ΣC_m); el combinador es lineal en n."],
                       conclusion="Óptimo único, descenso garantizado y costo lineal en el número de viviendas.",
                       fuente=f"{E4C} · caps. 5.2 y 5.3", ancho_visual=9)


def d_e4_generalizacion():
    hp = pd.DataFrame([
        ("J (anclas)", "4 a 80 (log)", "Pocas: casi global; muchas: pesos ruidosos", "16"),
        ("h (ancho de banda)", "0.25 a 4 (log)", "Escala de la ciudad (diagonal ≈ 20)", "1.38"),
        ("k (anclas vecinas)", "1 a 10", "Cuántas regiones se mezclan", "1"),
        ("λ (penalización KL)", "10⁻³ a 10 (log)", "De pesos libres a pesos globales", "0.12"),
        ("Peso de clase", "balanceado o ninguno", "El F1 macro pesa igual los 6 estratos", "balanceado"),
        ("Espacio de competencia", "geográfico, variables, probabilidades", "Dónde varía la competencia", "variables"),
    ], columns=["Hiperparámetro", "Rango buscado", "Por qué", "Modelo final"])
    visual = dbc.Row([dbc.Col(_md(GENERALIZACION_EPL), lg=6),
                      dbc.Col([tabla(hp), html.P("Búsqueda bayesiana (TPE, Optuna) con 12 evaluaciones y 3 pliegues "
                                                 "internos kNNDM, el mismo presupuesto que los benchmarks; fijos: "
                                                 "ε = 10⁻⁴, η₀ = 1, tolerancia 10⁻⁷, bloques de 1 km.",
                                                 className="nota-visual")], lg=6)], className="g-2")
    return diapositiva("8.7", "Fundamentos (3): regularización, supuestos e hiperparámetros", visual,
                       puntos=["La capacidad del combinador es pequeña y está regularizada; el riesgo de sobreajuste "
                               "está en los expertos, que ya se validaron en el E3.",
                               "Los rangos van de un extremo con sentido al otro (casi global ↔ muy local) en escala "
                               "logarítmica.",
                               "La sensibilidad es plana: moviendo J, h, k o λ el F1 va de 0.615 a 0.624.",
                               "El supuesto (a) es el que se rompe en el test: zonas con estructura distinta."],
                       conclusion="Generaliza por diseño (ajuste cruzado + KL + núcleo), no por suerte en la búsqueda.",
                       fuente=f"{E4C} · caps. 5 y 6.7", ancho_visual=9)


def d_e4_sobreajuste():
    s = F4.leer("sobreajuste.json")["modelos"]
    e = s[ORIGINAL]
    visual = html.Div([controles(control("Vista", radio(
        "e4-sa-vista", [("brechas", "Interno → externo → test"), ("ajuste", "Sin ajuste cruzado")], "brechas"))),
        grafico(F4.fig_sobreajuste("brechas"), "e4-sa-fig")])
    return diapositiva("8.20", "Diagnóstico de sobreajuste", visual,
                       puntos=[f"EPL: {e['interno']:.3f} en el bucle interno (donde se eligen los hiperparámetros), "
                               f"{e['externo']:.3f} en los pliegues externos y {e['test']:.3f} en el test.",
                               "Interno → externo (0.01 a 0.04 en todos los modelos): optimismo de elegir entre 12 "
                               "configuraciones. Externo → test: zonas con otra composición (estrato 6 en un solo bloque).",
                               "La brecha del EPL es como la de CatBoost o Random Forest: el combinador no sobreajusta "
                               "más que sus expertos. SVM y XGBoost geográfico son los que más caen en el test.",
                               "Sin ajuste cruzado (pesos aprendidos con predicciones dentro de muestra) el F1 cae 0.054: "
                               "ese sí es sobreajuste."],
                       conclusion="La validación anidada absorbe el sobreajuste de la búsqueda; el ajuste cruzado evita el "
                                  "del combinador.",
                       metricas=["f1_macro"], fuente=f"{E4C} · caps. 3 y 6")


@callback(Output("e4-sa-fig", "figure"), Input("e4-sa-vista", "value"))
def _sa(v):
    return F4.fig_sobreajuste(v)


def d_e4_implementacion():
    filas = pd.DataFrame([
        ("K-medias de las anclas", "kmedias", "numpy"),
        ("Núcleo gaussiano sobre las k anclas más cercanas", "matriz_nucleo", "scipy cKDTree"),
        ("Log-pérdida ponderada + KL y su gradiente", "perdida, gradiente", "numpy"),
        ("Gradiente exponenciado con paso adaptativo", "gradiente_exponenciado", "numpy"),
        ("Bloques espaciales y predicciones fuera de muestra", "bloques_espaciales, probabilidades_fuera_de_muestra", "numpy"),
        ("Espacio de competencia (geográfico, 2 CP vía SVD, probabilidades)", "ajustar_transformacion", "numpy"),
        ("Clase compatible con scikit-learn", "EnsamblePesosLocales (fit, predict_proba, predict)", "BaseEstimator"),
        ("Valores de Shapley por permutaciones", "shapley_muestreo (src/explicacion.py)", "numpy, pandas"),
    ], columns=["Componente", "Función en src/original_model.py", "Solo usa"])
    pruebas = pd.DataFrame([
        ("Probabilidades válidas", "suman 1 y están en [0, 1]"), ("Pesos en el símplex", "no negativos y suman 1"),
        ("Pérdida monótona", "nunca sube entre iteraciones"), ("Convexidad", "óptimo único con λ > 0"),
        ("Caso J = 1", "coincide con los pesos globales"), ("Ida y vuelta", "joblib reproduce las probabilidades"),
        ("Compatibilidad", "clone, get_params y Pipeline de scikit-learn"),
    ], columns=["Prueba unitaria (30 en total)", "Qué verifica"])
    visual = dbc.Row([dbc.Col(tabla(filas), lg=7), dbc.Col(tabla(pruebas), lg=5)], className="g-2")
    return diapositiva("8.5", "Implementación propia (R2): sin librerías de ensambles", visual,
                       puntos=["Todo el núcleo del modelo está programado con numpy y scipy; no se importa ningún "
                               "ensamble, compuerta ni optimizador.",
                               "Cuatro proposiciones demostradas en el libro: convexidad, unicidad, descenso monótono y "
                               "reducción al caso global.",
                               "30 pruebas unitarias en tests/test_original_model.py; 45 pruebas en total pasan.",
                               "Los modelos se guardan con joblib y metadata.json; una prueba de ida y vuelta verifica "
                               "que el archivo reproduce las probabilidades."],
                       conclusion="R2 se cumple: la regla de aprendizaje es código propio y está probada.",
                       fuente=f"{E4C} · cap. 5")


def d_e4_literatura():
    lit = F4.leer("literatura.json")
    bases = sorted({f[0] for f in lit["matriz"]})
    visual = html.Div([controles(
        control("Vista", radio("e4-lit-vista", [("matriz", "Matriz de búsqueda"), ("diferencias", "Qué lo diferencia")],
                               "matriz")),
        control("Base", desplegable("e4-lit-base", [("todas", "Todas")] + [(b, b) for b in bases], "todas", ancho=200))),
        html.Div(_tabla_lit("matriz", "todas"), id="e4-lit-tabla")])
    return diapositiva("8.6", "Novedad (R1): revisión de literatura", visual,
                       puntos=["Cuatro bases (Scopus, Web of Science, Google Scholar y arXiv) con seis cadenas el "
                               "23-oct-2026: problema, familia del modelo, regla de combinación, ensambles geográficos, "
                               "selección dinámica y búsqueda en español.",
                               "Ningún registro predice el estrato de cada vivienda con el catastro; donde aparece, el "
                               "estrato es una variable de entrada o se estima con datos del hogar.",
                               "Lo más cercano en el método: combinación convexa por regiones (Zhou y Lin, 2026), "
                               "ensamble espaciotemporal con gradiente exponenciado (Zeng et al., 2026) y ponderación "
                               "espacial local (Liu, 2024).",
                               "Lo nuevo («Qué lo diferencia»): la regla de aprendizaje (log-pérdida convexa con KL y "
                               "gradiente exponenciado), el ajuste cruzado por bloques, el espacio de competencia elegido "
                               "por validación y la aplicación."],
                       conclusion="La aplicación es nueva y la regla de aprendizaje no aparece igual en la literatura.",
                       fuente=f"{E4C} · cap. 4")


def _tabla_lit(vista, base):
    if vista == "diferencias":
        return tabla(DIFERENCIAS, alto=430)
    lit = F4.leer("literatura.json")
    df = pd.DataFrame(lit["matriz"], columns=lit["columnas"]).drop(columns="Fecha")
    if base != "todas":
        df = df[df.Base == base]
    return tabla(df, alto=430)


@callback(Output("e4-lit-tabla", "children"), Input("e4-lit-vista", "value"), Input("e4-lit-base", "value"))
def _lit(vista, base):
    return _tabla_lit(vista, base)


# ---------------------------------------------------------------------------
# 3. Validación repetida y pruebas
# ---------------------------------------------------------------------------
def d_e4_lideres():
    lid = pd.DataFrame(_m()["lideres"]).sort_values("f1_cv", ascending=False)
    visual = html.Div([controles(control("Medida", radio("e4-lid-vista", [("cv", "Validación 5 × 5"), ("test", "Test")],
                                                         "cv"))),
                       grafico(F4.fig_lideres("cv"), "e4-lid-fig")])
    return diapositiva("8.8", "Los 11 modelos en la validación repetida 5 × 5", visual,
                       puntos=["Mejor combinación de balanceo y optimizador del E3 para cada uno de los 8 modelos, los "
                               "recientes LightGBM y CatBoost, y el EPL; mismas 25 particiones para todos.",
                               f"El EPL tiene la mayor media ({lid.f1_cv.iloc[0]:.3f}); le siguen "
                               f"{lid.modelo.iloc[1]} ({lid.f1_cv.iloc[1]:.3f}) y {lid.modelo.iloc[2]} "
                               f"({lid.f1_cv.iloc[2]:.3f}).",
                               "La desviación entre pliegues (≈ 0.10) es mucho mayor que las diferencias entre los "
                               "mejores: hay zonas fáciles y difíciles para todos.",
                               "En el test el orden cambia: CatBoost (0.585) y el EPL (0.583) quedan casi empatados."],
                       conclusion="El EPL encabeza la validación, pero por poco: el grupo de cabeza está muy junto.",
                       metricas=["f1_macro", "ic95"], fuente=f"{E4C} · cap. 3")


@callback(Output("e4-lid-fig", "figure"), Input("e4-lid-vista", "value"))
def _lid(v):
    return F4.fig_lideres(v)


def d_e4_cd():
    fr = _m()["estadistica"][_ver()]["friedman"]
    r = sorted(fr["rangos"].items(), key=lambda t: t[1])
    visual = html.Div([controles(control("Versión del EPL", radio(
        "e4-cd-ver", [("v2", "2: espacio elegido por validación"), ("v1", "1: espacio geográfico")], "v2"))),
        grafico(F4.fig_cd("v2"), "e4-cd-fig")])
    return diapositiva("8.9", "Prueba global: Friedman y diferencia crítica de Nemenyi", visual,
                       puntos=[f"Friedman rechaza que todos tengan el mismo rango (χ² = {fr['chi2']:.1f}, "
                               f"p = {FM.cientifico(fr['p'])}).",
                               f"El EPL tiene el mejor rango medio ({r[0][1]:.2f}); {r[1][0]} ({r[1][1]:.2f}) y "
                               f"{r[2][0]} ({r[2][1]:.2f}) le siguen.",
                               f"La diferencia crítica es {fr['cd']:.2f}: todo lo que cae dentro de la franja no se "
                               "distingue del EPL según Nemenyi.",
                               "Las versiones 1 y 2 dan casi el mismo resultado."],
                       conclusion="El EPL queda primero, pero Nemenyi no lo separa de los ensambles de árboles.",
                       metricas=["friedman"], fuente=f"{E4C} · cap. 6.2")


@callback(Output("e4-cd-fig", "figure"), Input("e4-cd-ver", "value"))
def _cd(v):
    return F4.fig_cd(v)


def _tabla_pruebas(version):
    e = _m()["estadistica"][version]
    mejor = e["veredicto_validacion"]["mejor_benchmark"]
    x = F4.rival(mejor, version)
    fr = e["friedman"]

    def concl(p):
        return "rechaza" if p < 0.05 else "no rechaza"
    filas = [
        ("Friedman → Nemenyi", "Mismo rango medio", f"χ² = {fr['chi2']:.1f}", f"{x['nemenyi_p']:.3f}",
         f"CD {fr['cd']:.2f}", concl(x["nemenyi_p"])),
        ("Wilcoxon de rangos con signo", "Mediana de la diferencia = 0", f"W = {x['wilcoxon_W']:.0f}",
         f"Holm {x['wilcoxon_p_holm']:.3f}", f"δ Cliff {x['delta_cliff']:+.3f} ({x['delta_magnitud']})",
         concl(x["wilcoxon_p_holm"])),
        ("t corregida de Nadeau-Bengio", "Media de la diferencia = 0", f"t = {x['t_NB']:.2f} (gl {x['t_NB_gl']})",
         f"Holm {x['t_NB_p_holm']:.3f}", f"d pareada {x['d_cohen_pareada']:.2f}", concl(x["t_NB_p_holm"])),
        ("Bayesiana de rangos con signo (ROPE 0.01)", "—", "—", "—",
         f"P(EPL mejor) {x['bayes']['P(EPL mejor)']:.2f}", "probable mejora" if x["bayes"]["P(EPL mejor)"] > 0.95
         else "incierta"),
    ]
    return tabla(pd.DataFrame(filas, columns=["Prueba", "H₀", "Estadístico", "p ajustado", "Efecto", "Conclusión"]))


def d_e4_pruebas():
    x = F4.rival(_mejor())
    visual = html.Div([controles(control("Versión del EPL", radio(
        "e4-pr-ver", [("v2", "Versión 2"), ("v1", "Versión 1")], "v2"))),
        html.Div(_tabla_pruebas("v2"), id="e4-pr-tabla"),
        dbc.Alert([html.B("Veredicto de R3: no se cumple. "),
                   f"Condición 1 (mayor media que todos): sí. Condición 2 (al menos 3 pruebas rechazan con Holm): no, "
                   f"solo Wilcoxon. Condición 3 (|δ| ≥ 0.147 o |d| ≥ 0.2): sí por la d pareada "
                   f"({x['d_cohen_pareada']:.2f}), aunque el δ de Cliff ({x['delta_cliff']:.3f}) es despreciable."],
                  color="warning", className="mt-3")])
    return diapositiva("8.11", f"Superioridad (R3): EPL frente a {_mejor()}", visual,
                       puntos=["Las pruebas se aplican frente al mejor benchmark de la validación; las comparaciones con "
                               "los 10 rivales se corrigen con Holm.",
                               "Wilcoxon rechaza (p de Holm 0.032): el EPL gana en la mayoría de los pliegues.",
                               "La t de Nadeau-Bengio corrige que los pliegues comparten entrenamiento y no rechaza "
                               "(p ≈ 0.99 tras Holm); Nemenyi tampoco.",
                               "La bayesiana da 95 % de probabilidad de que el EPL sea mejor por más de 0.01, pero no es "
                               "una de las pruebas que cuenta la guía."],
                       conclusion="Una de tres pruebas rechaza: la superioridad no queda demostrada y R3 no se cumple.",
                       metricas=["holm", "cliff", "p_valor"], fuente=f"{E4C} · caps. 6.3 y 6.9")


@callback(Output("e4-pr-tabla", "children"), Input("e4-pr-ver", "value"))
def _pr(v):
    return _tabla_pruebas(v)


def d_e4_base():
    b = F4.leer("base_vs_epl.json")
    e, t, fr = b["epl_vs_base"], b["test"], b["friedman_con_base"]
    cifras = dbc.Row([
        dbc.Col(kpi(f"{e['media_rival']:.3f} → {e['media_rival'] + e['dif_media']:.3f}", "F1 en la 5 × 5",
                    f"{e['pliegues_a_favor']} de 25 pliegues a favor"), md=4),
        dbc.Col(kpi(f"{t['f1'][F4.BASE_E1]:.3f} → {t['f1'][ORIGINAL]:.3f}", "F1 en el test",
                    f"IC 95 % [{t['ic95_bca_bloques'][0]:+.3f}, {t['ic95_bca_bloques'][1]:+.3f}]"), md=4),
        dbc.Col(kpi(f"{e['delta_cliff']:.2f}", "δ de Cliff (grande)", f"d pareada {e['d_cohen_pareada']:.1f}"), md=4)],
        className="g-2 mb-2")
    visual = html.Div([cifras, grafico(F4.fig_pliegues(F4.BASE_E1), alto=360)])
    return diapositiva("8.12", "Frente al modelo base del Entregable 1", visual,
                       puntos=["El modelo base (logística multinomial sin vecindad física) se reentrenó en las mismas 25 "
                               "particiones; en la repetición 0 reproduce el E3 exacto.",
                               f"Wilcoxon p = {FM.cientifico(e['wilcoxon_p_holm'])}, t de Nadeau-Bengio "
                               f"p = {e['t_NB_p_holm']:.4f}, Nemenyi p = {FM.cientifico(fr['nemenyi_p_epl_vs_base'])} y "
                               f"bayesiana P(mejor) = {e['bayes']['P(EPL mejor)']:.2f}: las cuatro rechazan.",
                               f"En el test gana en {t['bloques_a_favor']} de {t['n_bloques']} bloques y McNemar también "
                               "rechaza.",
                               "Buena parte de la mejora la explican la vecindad física y los modelos no lineales, que "
                               "comparten los benchmarks."],
                       conclusion="Frente al modelo base del proyecto la superioridad sí está demostrada.",
                       metricas=["f1_macro", "cliff"], fuente=f"{E4C} · cap. 6.3")


def d_e4_ablacion():
    sc = F4.variante("sin_ajuste_cruzado")
    pg = F4.variante("pesos_error_global")
    visual = grafico(F4.fig_ablacion())
    return diapositiva("8.13", "Reglas estándar y ablación: qué aporta cada pieza", visual,
                       puntos=["Mismos 7 expertos, mismas particiones; solo cambia la regla de combinación o se retira "
                               "un componente del EPL.",
                               f"Lo que más pesa es el ajuste cruzado: sin él el F1 cae {sc['dif_media']:.3f} (aprender "
                               "los pesos sobre datos ya vistos premia al experto que memoriza).",
                               f"Los pesos por error global quedan a {abs(pg['dif_media']):.3f} del EPL: el campo local "
                               "no aporta de forma medible en zonas nuevas.",
                               "Las versiones con un solo espacio de competencia empatan con la versión 2."],
                       conclusion="Lo que funciona es combinar expertos con predicciones fuera de muestra; el campo "
                                  "local, no tanto.",
                       metricas=["ic95"], fuente=f"{E4C} · caps. 5.8 y 6.4")


def d_e4_campo():
    c = _m()["campo_epl"]
    p = np.array(c["pesos"])
    exp_var = c["expertos"][int(np.argmax(np.ptp(p, axis=0)))]
    esp = _m()["espacios_elegidos_v2"]
    visual = html.Div([controles(
        control("Vista", radio("e4-campo-vista", [("campo", "Campo de un experto"), ("pesos", "Todos los pesos")],
                               "campo")),
        control("Experto", desplegable("e4-campo-exp", [(e, e) for e in c["expertos"]], exp_var, ancho=200))),
        grafico(F4.fig_campo(exp_var), "e4-campo-fig")])
    return diapositiva("8.14", "El espacio de competencia y el campo de pesos", visual,
                       puntos=["Versión 2: el bucle interno elige dónde se definen las regiones. En los 25 pliegues eligió "
                               + ", ".join(f"{F4.NOMBRE_ESPACIO[k]} {v}" for k, v in esp.items()) + " veces.",
                               "Espacio de las variables: 2 componentes principales de las variables de la vivienda; "
                               "anclas cercanas = viviendas parecidas, no vecinas.",
                               f"{exp_var} es el experto cuyo peso más varía entre anclas. En los pesos globales, "
                               "XGBoost (0.51) y Random Forest (0.20) suman el 70 %; SVM queda en 0.",
                               "El tamaño y el color del punto son el peso del experto en esa ancla."],
                       conclusion="El modelo aprende a confiar en expertos distintos según el tipo de vivienda.",
                       fuente=f"{E4C} · caps. 5 y 6.4")


@callback(Output("e4-campo-fig", "figure"), Input("e4-campo-vista", "value"), Input("e4-campo-exp", "value"))
def _campo(v, e):
    return F4.fig_pesos_globales() if v == "pesos" else F4.fig_campo(e)


# ---------------------------------------------------------------------------
# 4. Confirmación: test, diagnóstico, conjuntos externos, robustez y costo
# ---------------------------------------------------------------------------
def d_e4_test():
    t = F4.leer("test_confirmatorio.json")
    d = t["epl_menos_mejor"]
    mc = t["mcnemar"]["CatBoost"]
    visual = html.Div([controles(control("Vista", radio("e4-test-vista", [("f1", "F1 con IC"), ("bloques", "Por bloque")],
                                                        "f1"))),
                       grafico(F4.fig_lideres("test"), "e4-test-fig")])
    return diapositiva("8.15", "Test confirmatorio: 65 145 viviendas que nadie vio", visual,
                       puntos=[f"CatBoost {t['f1_macro']['CatBoost']:.3f} y EPL {t['f1_macro'][ORIGINAL]:.3f}: "
                               f"diferencia {d['dif_f1']:+.4f}, IC 95 % BCa por bloques [{d['ic95_bca_bloques'][0]:+.3f}, "
                               f"{d['ic95_bca_bloques'][1]:+.3f}], permutación por bloques p = "
                               f"{d['p_permutacion_signo_bloques']:.2f}.",
                               f"McNemar rechaza a favor de CatBoost ({fmt(mc['c'])} viviendas solo acierta CatBoost frente a "
                               f"{fmt(mc['b'])} solo el EPL), pero supone viviendas independientes y aquí el ICC es 0.99.",
                               "DeLong: el EPL tiene mayor AUC en los estratos 1 a 4 y menor en 5 y 6.",
                               "El test tiene 7 bloques y todo el estrato 6 cae en uno: los IC son anchos."],
                       conclusion="En el test el EPL y CatBoost no se distinguen: no confirma la ventaja de la validación.",
                       metricas=["f1_macro", "ic95", "icc"], fuente=f"{E4C} · cap. 6.5")


@callback(Output("e4-test-fig", "figure"), Input("e4-test-vista", "value"))
def _test(v):
    return F4.fig_test_bloques() if v == "bloques" else F4.fig_lideres("test")


def d_e4_diagnostico():
    nombres = [r["modelo"] for r in sorted(_m()["lideres"], key=lambda r: -r["f1_cv"])]
    cm = np.array(_m()["diagnosticos"][ORIGINAL]["confusion"])
    cerca = (np.trace(cm) + np.trace(cm, 1) + np.trace(cm, -1)) / cm.sum()
    visual = html.Div([controles(
        control("Modelo", desplegable("e4-dg-mod", [(n, n) for n in nombres], ORIGINAL, ancho=220)),
        control("Vista", radio("e4-dg-vista", [("confusion", "Confusión"), ("calibracion", "Calibración"),
                                               ("roc", "ROC"), ("pr", "Precisión-recall"),
                                               ("importancia", "Importancia")], "confusion"))),
        grafico(F4.fig_diagnostico(ORIGINAL, "confusion"), "e4-dg-fig")])
    return diapositiva("8.16", "Diagnóstico en el test de cada modelo", visual,
                       puntos=[f"El {100 * cerca:.0f} % de las predicciones del EPL cae en el estrato real o en uno "
                               "vecino; los errores grandes son raros.",
                               "El estrato 1 es el que mejor se reconoce (79 %). Los altos se corren hacia abajo: el 5 "
                               "va al 4 en el 42 % de los casos y el 6 al 5 en el 70 % (en el test hay solo 793 "
                               "viviendas de estrato 6, todas en un bloque).",
                               "Importancia: la misma estimación de Shapley por permutaciones para todos los modelos, "
                               "así que se pueden comparar.",
                               "En el EPL y en CatBoost encabezan la ubicación y el entorno (vecinos), no la vivienda."],
                       conclusion="El EPL se equivoca como los demás: entre estratos vecinos, sobre todo en los altos.",
                       metricas=["acc1", "ece", "auc"], fuente=f"{E4C} · caps. 6.5 y 6.8")


@callback(Output("e4-dg-fig", "figure"), Input("e4-dg-mod", "value"), Input("e4-dg-vista", "value"))
def _dg(m, v):
    return F4.fig_diagnostico(m, v)


def d_e4_externos():
    e = F4.leer("resumen_externos_v2.json")
    filas = []
    for k, v in e.items():
        mejor_rival = max((m for m in v["media"] if m != ORIGINAL and not m.startswith("[")), key=v["media"].get)
        filas.append((F4.NOMBRE_EXTERNO[k].split(" (")[0], v["media"][ORIGINAL], mejor_rival, v["media"][mejor_rival],
                      v["friedman"]["rangos"][ORIGINAL]))
    resumen = pd.DataFrame(filas, columns=["Conjunto", "F1 del EPL", "Mejor rival", "F1 del rival", "Rango del EPL"])
    visual = html.Div([controles(control("Conjunto", radio(
        "e4-ext", [(k, F4.NOMBRE_EXTERNO[k].split(" (")[0]) for k in e], "king_county"))),
        grafico(F4.fig_externos("king_county"), "e4-ext-fig", alto=380), tabla(resumen)])
    return diapositiva("8.17", "Tres conjuntos de datos adicionales (OpenML)", visual,
                       puntos=["King County y California (vivienda, con coordenadas) y vino tinto (sin ubicación), con "
                               "el mismo protocolo: 3 × 5 pliegues, 12 evaluaciones, semilla 42.",
                               "California: el EPL gana a todos y tiene el mejor rango.",
                               "King County: empata con LightGBM, Random Forest y CatBoost.",
                               "Vino tinto: pierde frente a Random Forest; sin ubicación, el campo local tiene menos que "
                               "aprovechar."],
                       conclusion="Fuera de Barranquilla el EPL gana una vez, empata una y pierde una.",
                       metricas=["f1_macro", "friedman"], fuente=f"{E4C} · cap. 6.6")


@callback(Output("e4-ext-fig", "figure"), Input("e4-ext", "value"))
def _ext(k):
    return F4.fig_externos(k)


def d_e4_robustez():
    r = F4.leer("robustez_resumen.json")
    s = list(r["semillas_EPL"].values())
    visual = html.Div([controles(control("Vista", radio(
        "e4-rob-vista", [("condiciones", "Condiciones adversas"), ("sensibilidad", "Sensibilidad")], "condiciones"))),
        grafico(F4.fig_robustez("condiciones"), "e4-rob-fig")])
    return diapositiva("8.18", "Robustez y sensibilidad", visual,
                       puntos=["Condiciones: 20 % de etiquetas cambiadas, 25 % del entrenamiento y los estratos 5 y 6 "
                               "reducidos; repetición 0 de la 5 × 5.",
                               "El EPL supera a Random Forest y CatBoost en las cuatro condiciones; con los estratos altos "
                               "reducidos, los pesos por error global quedan un poco arriba.",
                               f"Tres semillas: {', '.join(f'{v:.3f}' for v in s)} (desviación "
                               f"{r['sd_entre_semillas']:.4f}).",
                               "Sensibilidad plana: moviendo J, h, k o λ el F1 medio va de 0.615 a 0.624."],
                       conclusion="El resultado no depende de una semilla ni de un ajuste fino de hiperparámetros.",
                       metricas=["f1_macro"], fuente=f"{E4C} · cap. 6.7")


@callback(Output("e4-rob-fig", "figure"), Input("e4-rob-vista", "value"))
def _rob(v):
    return F4.fig_robustez(v)


def d_e4_complejidad():
    c = F4.leer("complejidad.json")
    visual = html.Div([controles(control("Serie", desplegable(
        "e4-cx-serie", list(F4.SERIES_COMPLEJIDAD.items()), "combinador vs n", ancho=320))),
        grafico(F4.fig_complejidad("combinador vs n"), "e4-cx-fig")])
    return diapositiva("8.19", "Costo computacional", visual,
                       puntos=["Teoría: el combinador cuesta O(n·M·k) por iteración, lineal en el número de viviendas.",
                               f"Medido: pendiente log-log {c['pendientes']['combinador vs n']:.2f} en n (teoría: 1), "
                               f"{c['pendientes']['combinador vs M']:.2f} en M y {c['pendientes']['combinador vs J']:.2f} "
                               "en J.",
                               "El EPL completo tarda 90 s con 7 943 viviendas; más del 99 % es entrenar los expertos 6 "
                               "veces (ajuste cruzado).",
                               "Predecir cuesta 0.3 ms por vivienda."],
                       conclusion="El combinador es barato; el costo está en los expertos.",
                       fuente=f"{E4C} · cap. 5.7")


@callback(Output("e4-cx-fig", "figure"), Input("e4-cx-serie", "value"))
def _cx(s):
    return F4.fig_complejidad(s)


# ---------------------------------------------------------------------------
# 5. Predicción en vivo con explicación
# ---------------------------------------------------------------------------
def _esquema():
    meta = json.loads((F4.E4 / "metadata.json").read_text(encoding="utf-8"))
    return {c["columna"]: c for c in meta["esquema_entrada"]}


def _unidades():
    d = F4.leer("diccionario.json")
    return {x["variable"]: x["unidad"] for x in d}


def _tipica(esq):
    v = F4.viviendas()
    return {c: (v[c].mode().iloc[0] if esq[c]["tipo"] == "categórica" else float(v[c].median())) for c in esq}


def _campo_entrada(c, esq, tip, uni):
    e = esq[c]
    u = uni.get(c, "—")
    etiqueta = html.Label(f"{c} ({u})" if u not in ("—", "") else c, className="control-etiqueta")
    if e["tipo"] == "categórica":
        comp = dcc.Dropdown(id=f"e4-in-{c}", options=e["categorias"], value=tip[c], clearable=False)
    else:
        comp = dbc.Input(id=f"e4-in-{c}", type="number", value=round(tip[c], 2), min=e["minimo"], max=e["maximo"],
                         step="any", size="sm")
    return dbc.Col([etiqueta, comp], xs=6, md=4, lg=3, className="mb-2")


def d_e4_prediccion():
    esq, uni = _esquema(), _unidades()
    tip = _tipica(esq)
    ej = F4.ejemplos()
    formulario = html.Div([
        dbc.Row([dbc.Col([html.Label("Cargar una vivienda del test (con su estrato real)", className="control-etiqueta"),
                          dcc.Dropdown(id="e4-ejemplo", placeholder="(opcional)", options=[
                              {"label": f"Ejemplo {i + 1}: estrato real {int(e)}", "value": i}
                              for i, e in enumerate(ej.estrato_num)])], md=6),
                 dbc.Col(dbc.Button("Predecir", id="e4-predecir", color="primary", className="mt-4"), md=2)],
                className="mb-2"),
        dbc.Row([_campo_entrada(c, esq, tip, uni) for c in ENTRADAS]),
        dcc.Store(id="e4-ejemplo-fila")])
    visual = html.Div([formulario, dbc.Row([
        dbc.Col(grafico(F4.fig_mapa_clic(), "e4-mapa-clic", alto=400), lg=5),
        dbc.Col([html.Div(id="e4-alerta"), dcc.Loading(html.Div(id="e4-salida",
                                                                children=html.P("Pulse «Predecir». La primera vez "
                                                                                "tarda unos segundos: se carga el modelo.",
                                                                                className="nota-visual")))], lg=7)])])
    return diapositiva("8.20", "Predicción en vivo con el EPL y su explicación", visual,
                       puntos=["Se cargan las variables de una vivienda (o un ejemplo del test) y el EPL serializado del "
                               "E4 devuelve la probabilidad de cada estrato.",
                               "Las variables de vecindad (8 edificios más cercanos) y la distancia al centro se calculan "
                               "solas a partir de la ubicación; también se puede hacer clic en el mapa.",
                               "Valores fuera del rango del entrenamiento o faltantes se rechazan con un aviso.",
                               "La explicación son valores de Shapley por permutaciones (implementación propia): cuánto "
                               "sube o baja cada variable la probabilidad del estrato predicho."],
                       conclusion="El modelo guardado funciona fuera del cuaderno y explica cada predicción.",
                       fuente=f"{E4C} · caps. 5 y 7", ancho_visual=9)


@callback([Output(f"e4-in-{c}", "value") for c in ENTRADAS] + [Output("e4-ejemplo-fila", "data")],
          Input("e4-ejemplo", "value"), prevent_initial_call=True)
def _cargar_ejemplo(i):
    if i is None:
        return [no_update] * (len(ENTRADAS) + 1)
    fila = F4.ejemplos().iloc[int(i)]
    esq = _esquema()
    valores = [fila[c] if esq[c]["tipo"] == "categórica" else (None if pd.isna(fila[c]) else float(fila[c]))
               for c in ENTRADAS]
    return valores + [{k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in fila.items()}]


@callback(Output("e4-in-x_km", "value", allow_duplicate=True), Output("e4-in-y_km", "value", allow_duplicate=True),
          Input("e4-mapa-clic", "clickData"), prevent_initial_call=True)
def _ubicar(click):
    if not click:
        return no_update, no_update
    p = click["points"][0]
    return round(p["x"], 2), round(p["y"], 2)


_ARBOL = {}


def _derivar(fila, ejemplo):
    """Distancia al centro y vecindad de los 8 vecinos más cercanos de la submuestra (o las del ejemplo)."""
    if ejemplo and abs(ejemplo["x_km"] - fila["x_km"]) < 1e-9 and abs(ejemplo["y_km"] - fila["y_km"]) < 1e-9:
        for c in ["dist_centro_km", "knn8_log_area", "knn8_banios", "knn8_habitaciones", "knn8_antiguedad", "knn8_piso",
                  "knn8_prop_PH", "knn8_dist_km"]:
            fila[c] = ejemplo[c]
        return fila, "las del ejemplo"
    v = F4.viviendas()
    if "arbol" not in _ARBOL:
        _ARBOL["arbol"] = cKDTree(v[["x_km", "y_km"]].to_numpy())
    dist, idx = _ARBOL["arbol"].query([fila["x_km"], fila["y_km"]], k=8)
    w = v.iloc[idx]
    c0 = F4.leer("centro.json")
    fila.update({"dist_centro_km": float(np.hypot(fila["x_km"] - c0["x0"], fila["y_km"] - c0["y0"])),
                 "knn8_log_area": float(np.log1p(w.area_construida).mean()),
                 "knn8_banios": float(w.total_banios.mean()), "knn8_habitaciones": float(w.total_habitaciones.mean()),
                 "knn8_antiguedad": float(w.antiguedad.mean()), "knn8_piso": float(w.planta_ubicacion.mean()),
                 "knn8_prop_PH": float((w.condicion_predio == "PH_Unidad_Predial").mean()),
                 "knn8_dist_km": float(np.mean(dist))})
    return fila, "aproximadas con la submuestra de 10 001 viviendas"


@callback(Output("e4-salida", "children"), Output("e4-alerta", "children"), Input("e4-predecir", "n_clicks"),
          State("e4-ejemplo-fila", "data"), [State(f"e4-in-{c}", "value") for c in ENTRADAS], prevent_initial_call=True)
def _predecir(_, ejemplo, *valores):
    esq = _esquema()
    fila = dict(zip(ENTRADAS, valores))
    errores = []
    for c in ENTRADAS:
        e, v = esq[c], fila[c]
        if e["tipo"] == "categórica":
            if v not in e["categorias"]:
                errores.append(f"{c}: elija una categoría de la lista.")
        elif v is None or not np.isfinite(float(v)):
            errores.append(f"{c}: falta el valor o no es un número.")
        elif not e["minimo"] <= float(v) <= e["maximo"]:
            errores.append(f"{c}: {v} está fuera del rango del entrenamiento [{e['minimo']:g}, {e['maximo']:g}].")
    if errores:
        return no_update, dbc.Alert([html.B("Revise los datos: "), html.Ul([html.Li(x) for x in errores])],
                                    color="danger")
    try:
        from estrato3 import evaluacion as EV
        from src.explicacion import shapley_muestreo
        fila, origen = _derivar(fila, ejemplo)
        X = pd.DataFrame([fila])[list(esq)]
        for c in esq:
            if esq[c]["tipo"] != "categórica":
                X[c] = X[c].astype(float)
        est = F4.modelo_epl()

        def f(df):
            return EV.proba_completa(est, df)
        proba = f(X)[0]
        k = int(np.argmax(proba))
        fondo = F4.viviendas()[list(esq)].sample(40, random_state=42)
        phi, base, fx = shapley_muestreo(f, X, fondo, k, n_permutaciones=25, semilla=0)
    except Exception as e:  # el dashboard no se cae: muestra el error
        return no_update, dbc.Alert(f"No se pudo predecir: {type(e).__name__}: {e}", color="danger")
    real = (f" · estrato real del ejemplo: {int(ejemplo['estrato_num'])}" if ejemplo and origen == "las del ejemplo"
            else "")
    return html.Div([
        kpi(f"Estrato {k + 1}", f"predicción del EPL (probabilidad {proba[k]:.0%}){real}", color="#eb6834"),
        grafico(F4.fig_probabilidad(proba), alto=300),
        grafico(F4.fig_shapley(phi, base, fx, k + 1), alto=380),
        html.P(f"Vecindad y distancia al centro: {origen}. Azul sube la probabilidad del estrato predicho y naranja la "
               f"baja; los aportes suman {fx - base:+.3f}.", className="nota-visual")]), None


# ---------------------------------------------------------------------------
# 6. Veredicto
# ---------------------------------------------------------------------------
def d_e4_veredicto():
    x = F4.rival(_mejor())
    filas = pd.DataFrame([
        ("R1. Novedad", "Sí", "4 bases y 6 cadenas: nadie predice el estrato de viviendas con el catastro; la regla de "
                              "aprendizaje, el ajuste cruzado y el espacio elegido por validación no aparecen juntos."),
        ("R2. Implementación propia", "Sí", "Núcleo con numpy y scipy, clase compatible con scikit-learn, 4 proposiciones "
                                            "demostradas, 30 pruebas unitarias y ablación."),
        ("R3. Superioridad", "No", f"Mayor media que los 10 benchmarks; frente a {_mejor()} solo Wilcoxon-Holm rechaza "
                                   f"(1 de 3). d pareada {x['d_cohen_pareada']:.2f}; δ de Cliff "
                                   f"{x['delta_cliff']:.3f}. El test y los conjuntos externos no confirman la ventaja."),
    ], columns=["Condición", "¿Se cumple?", "Evidencia"])
    visual = html.Div([tabla(filas),
                       html.Div([html.A(["Libro del Entregable 4 ", html.Span("↗", className="enlace-flecha")],
                                        href=LIBRO_E4, target="_blank", className="enlace-recurso"),
                                 html.A(["Repositorio del Entregable 4 ", html.Span("↗", className="enlace-flecha")],
                                        href="https://github.com/mcarocantillo/entregable4-ml-barranquilla",
                                        target="_blank", className="enlace-recurso")], className="recursos mt-3")])
    return diapositiva("8.21", "Veredicto del Entregable 4", visual,
                       puntos=["Por la regla de la guía (sección 12.1), si R3 falla el tope de la nota es 4.0.",
                               "Manipular particiones, semillas o pruebas para conseguir significancia no sería válido; "
                               "se reporta el resultado desfavorable.",
                               "Lo que sí queda demostrado: el EPL supera con claridad al modelo base del E1 y encabeza la "
                               "validación, empatado con los mejores ensambles.",
                               "Trabajo futuro: probarlo en zonas conocidas (interpolar dentro de barrios con datos), "
                               "donde la información local sí ayudó en el E2."],
                       conclusion="Modelo nuevo y propio, competitivo con los mejores, pero sin superioridad demostrada.",
                       fuente=f"{E4C} · caps. 6.9 y 8")


# Diapositivas del E4 que van en la pestaña Modelos (el resumen va en Contexto y el veredicto en Conclusiones).
DIAPOSITIVAS_MODELOS = [
    ("Correcciones del E3", d_e4_correcciones), ("Validación repetida 5 × 5", d_e4_e3_vs_5x5),
    ("El modelo original (EPL)", d_e4_idea), ("Fundamentos: modelo y pérdida", d_e4_formulacion),
    ("Fundamentos: optimización", d_e4_optimizacion), ("Fundamentos: regularización", d_e4_generalizacion),
    ("Implementación propia (R2)", d_e4_implementacion),
    ("Los 11 modelos", d_e4_lideres), ("Friedman y Nemenyi", d_e4_cd),
    ("Superioridad (R3)", d_e4_pruebas), ("Frente al modelo base", d_e4_base),
    ("Reglas estándar y ablación", d_e4_ablacion), ("Campo de pesos", d_e4_campo),
    ("Diagnóstico de sobreajuste", d_e4_sobreajuste), ("Test confirmatorio", d_e4_test),
    ("Diagnóstico en el test", d_e4_diagnostico), ("Conjuntos externos", d_e4_externos),
    ("Robustez y sensibilidad", d_e4_robustez), ("Costo computacional", d_e4_complejidad),
    ("Predicción en vivo", d_e4_prediccion),
]
