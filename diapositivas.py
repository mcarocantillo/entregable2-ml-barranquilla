"""Contenido del dashboard final: una lista de diapositivas por pestaña y sus controles interactivos.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Cada diapositiva es una función sin argumentos que devuelve el componente (se construye solo cuando se abre, para que
la app cargue rápido). Los textos son breves a propósito: el detalle está en los libros de los Entregables 1, 2 y 3.
"""
import json

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

import figuras as F
import diapositivas_e4 as DE4
import figuras_e4 as F4
import figuras_mejoras as FM
import figuras_proyecto as FP
from componentes import (callback, control, controles, desplegable, diapositiva, fmt, grafico, interruptor, kpi, radio, tabla)

# El dashboard muestra el modelo base de la versión 3 (validación kNNDM, entrenamiento sin buffer), que calcula
# proyecto/preparar_base_v3.py en datos/base_v3/. Los archivos de datos/ (versión 2) los sigue usando el libro del E2.
if (F.DATOS / "base_v3" / "resultados.json").exists():
    F.usar_base(F.DATOS / "base_v3")
F.usar_fuente(13)   # letra de las gráficas un poco menor que en el libro, para que quepa más en la pantalla
R = F.resultados()
D = F.diagnosticos()
E1 = F.entregable1()
GX = E1["grupo"]["con_excepcion"]
T = R["resultados_test"]
CV = R["tabla_cv"]
IC = R["ic_modelo_principal"]
B = "Logística B_fisicas+ubicacion"
MV = FP._json("multivariado.json")
MZ = FP._json("manzanas.json")
COB = pd.DataFrame(MZ["cobertura"]).T.astype(float)
COB.index = COB.index.astype(int)
LIBRO_E1 = "https://mcarocantillo.github.io/entregable1-ml-barranquilla/v3/"
LIBRO_E1_V2 = "https://mcarocantillo.github.io/entregable1-ml-barranquilla/v2/"
LIBRO_E1_V1 = "https://mcarocantillo.github.io/entregable1-ml-barranquilla/"
LIBRO_E2 = "https://mcarocantillo.github.io/entregable2-ml-barranquilla/v3/"
LIBRO_E2_V2 = "https://mcarocantillo.github.io/entregable2-ml-barranquilla/v2/"
LIBRO_E2_V1 = "https://mcarocantillo.github.io/entregable2-ml-barranquilla/"
LIBRO_E3 = "https://mcarocantillo.github.io/entregable3-ml-barranquilla/"
E1C, E2C, E3C = "Entregable 1", "Entregable 2", "Entregable 3"
OPC_NUM = [(k, v) for k, v in F.NUMERICAS.items()]


def cientifico(x, dec=1):
    """2.6e-05 → «2.6·10⁻⁵» (p-valores muy pequeños)."""
    m, e = f"{x:.{dec}e}".split("e")
    sup = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m}·10{sup}"


def piso(x, dec=2):
    """Redondea hacia abajo, para cotas «≥» que no exageren (0.548 → 0.54)."""
    return np.floor(x * 10 ** dec) / 10 ** dec


def pct(a, b=1, dec=1):
    return f"{100 * a / b:.{dec}f} %"


def flujo(pasos):
    return html.Div([html.Div([html.Div(n, className="paso-num"), html.Div(t, className="paso-titulo"),
                               html.Div(d, className="paso-desc")], className="paso") for n, t, d in pasos], className="flujo")


# ===========================================================================
# 1. CONTEXTO
# ===========================================================================
def d_pregunta():
    visual = html.Div([
        html.Img(src="/assets/escala_estratos.svg", className="escala-estratos",
                 alt="Escala de estratos 1 a 6 y su efecto en las tarifas"),
        dbc.Row([dbc.Col(kpi("6", "estratos", "de 1 (bajo-bajo) a 6 (alto)"), md=4),
                 dbc.Col(kpi("1 – 3", "reciben subsidio", "en servicios públicos"), md=4),
                 dbc.Col(kpi("5 – 6", "pagan contribución", "Ley 142 de 1994"), md=4)], className="g-3 mt-1"),
    ])
    return diapositiva("1.1", "¿Se puede predecir el estrato de una vivienda con sus datos catastrales?", visual,
                       puntos=["Cada vivienda tiene un estrato del 1 al 6 que fija cuánto paga de servicios públicos.",
                               "Estratos 1 a 3: reciben subsidio. Estratos 5 y 6: pagan contribución."],
                       conclusion="¿Cuánto del estrato se puede predecir con el catastro abierto de Barranquilla?",
                       fuente=f"{E2C} · cap. 1")


def d_dane():
    pasos = [("1", "Zonas", "Zonas homogéneas físicas y de valor del suelo"),
             ("2", "Estrato de la zona", "Puntaje de las edificaciones (0 a 100)"),
             ("3", "Viviendas atípicas", "Se mueven ±1 estrato")]
    insumos = pd.DataFrame([
        ["Zonas homogéneas", "No: se aproximan con coordenadas y vecindad"],
        ["Puntaje de las edificaciones", "No"],
        ["Área y cuartos de apartamentos", "Sí"],
        ["Casa o apartamento", "Sí"],
        ["Revisión del comité de estratificación", "No observable"],
    ], columns=["Insumo del DANE", "¿Está en el catastro abierto?"])
    visual = html.Div([flujo(pasos), html.Div(tabla(insumos), className="mt-3")])
    return diapositiva("1.2", "Cómo asigna el DANE el estrato", visual,
                       puntos=["El DANE asigna el estrato por zonas.",
                               "Solo mueve las viviendas atípicas: ±1 estrato.",
                               "Sus dos insumos clave no son públicos: los aproximamos con la ubicación y la vecindad."],
                       conclusion="¿Cuánto del estrato se recupera con la parte pública del catastro?",
                       fuente=f"{E2C} · cap. 1.2")


def d_datos():
    seg = [("Depto.", 2, "#86b6ef"), ("Municipio", 3, "#86b6ef"), ("Zona", 2, "#5598e7"), ("Sector", 2, "#5598e7"),
           ("Comuna", 2, "#5598e7"), ("Barrio", 2, "#5598e7"), ("Manzana", 4, "#eb6834"), ("Terreno", 4, "#2a78d6"),
           ("Cond.", 1, "#1c5cab"), ("Edificio", 2, "#104281"), ("Piso", 2, "#0d366b"), ("Unidad", 4, "#0d366b")]
    npn = html.Div([html.Div([html.Div("0" * n, className="npn-digitos"), html.Div(e, className="npn-etiqueta")],
                             className="npn-seg", style={"flex": n, "borderTopColor": c}) for e, n, c in seg],
                   className="npn")
    llaves = html.Div([html.Span("17 dígitos = manzana", className="npn-llave naranja"),
                       html.Span("22 dígitos = predio matriz (edificio, conjunto o casa)", className="npn-llave azul"),
                       html.Span("30 dígitos = la vivienda", className="npn-llave oscuro")], className="npn-llaves")
    visual = html.Div([
        dbc.Row([dbc.Col(kpi("382 597", "filas descargadas", "catastro abierto, 15-sep-2026"), md=3),
                 dbc.Col(kpi(fmt(E1["viviendas"]), "viviendas analizadas", "casas y apartamentos"), md=3),
                 dbc.Col(kpi(fmt(E1["predios_matriz"]), "predios matriz", "edificios, conjuntos o casas"), md=3),
                 dbc.Col(kpi("14", "variables", "físicas, régimen, uso y ubicación"), md=3)], className="g-3"),
        html.Div("El número predial nacional (NPN) dice dónde está cada vivienda", className="subtitulo-visual mt-4"),
        npn, llaves])
    return diapositiva("1.3", "Los datos: el catastro abierto de Barranquilla", visual,
                       puntos=["Cada fila es una unidad de vivienda.",
                               "Variables: áreas, habitaciones, baños, pisos, piso de ubicación, antigüedad, régimen, "
                               "uso, tipo de vivienda y coordenadas.",
                               "El NPN agrupa las viviendas en manzanas y predios matriz."],
                       conclusion="Un dato por vivienda, pero las viviendas vienen en grupos: edificios y manzanas.",
                       fuente=f"{E1C} · cap. 1.3")


CERCANOS_E4 = pd.DataFrame([
    ["Problema", "Cantillo-García, Guzmán y Arellana (2019), DYNA",
     "Estiman el estrato con un logit ordenado y variables del hogar (ingreso, tamaño, vehículos) en Bogotá, Medellín, "
     "Cali y Barranquilla", "Usa encuestas del hogar, no el catastro; un solo modelo, sin validación espacial"],
    ["Problema", "Chica-Olmo, Sánchez y Sepúlveda-Murillo (2020), Cities",
     "Evalúan si el estrato refleja la calidad de vida (logit multinivel, Medellín)",
     "El estrato es el criterio que se evalúa, no la salida de un clasificador"],
    ["Problema", "Vargas-Calderón y Camargo (2022), JORS; tesis colombianas de avalúo",
     "XGBoost para el precio de la vivienda en Bogotá", "El estrato es una variable de entrada, no el objetivo"],
    ["Problema", "Arias Cupitra y Moreno Moreno (2025), trabajo de grado",
     "Zonas homogéneas físicas con inteligencia artificial (Entrerríos, Antioquia)",
     "Primer paso de la metodología del DANE, no la asignación del estrato"],
    ["Método", "Zhou y Lin (2026), arXiv",
     "Combinación convexa distinta por región para regresión y diagnóstico de cuándo supera a la mezcla fija",
     "El EPL clasifica (6 clases ordenadas), suaviza las regiones con un núcleo, encoge hacia los pesos globales y "
     "elige el espacio por validación"],
    ["Método", "Zeng et al. (2026), IEEE Trans. Power Systems",
     "Pesos de un ensamble espaciotemporal por gradiente exponenciado, en línea (carga eléctrica)",
     "El EPL aprende fuera de línea, con un vector de pesos por ancla y ajuste cruzado por bloques"],
    ["Método", "Liu (2024), IJAEOG",
     "Ensamble con ponderación espacial local y explicabilidad (datos sintéticos)",
     "Regla derivada de una pérdida convexa con óptimo único; espacio de competencia elegido por validación"],
    ["Método", "Grekousis (2025), J. Geographical Systems",
     "XGBoost geográfico: promedio con peso fijo entre un modelo global y modelos locales (regresión)",
     "Base del modelo nuevo del E2/E3; el EPL aprende los pesos por ancla y por experto"],
], columns=["Cercano al", "Trabajo", "Qué hace", "Diferencia con este proyecto"])


def _bib_e12():
    usados = pd.DataFrame([
        ["Precio de la vivienda (regresión)", "XGBoost, Random Forest, SVR, árbol, Gradient Boosting, lineal"],
        ["Submercados y valorización de vivienda", "XGBoost + SHAP"],
        ["Pobreza por zona (México)", "XGBoost con datos censales"],
        ["Elección residencial con datos por sector censal (1998)", "Regresión logística"],
        ["Nivel socioeconómico como predictor (salud, PISA)", "Logística; RF, XGBoost, LASSO, stacking"],
    ], columns=["Qué predicen los trabajos cercanos (Tabla II)", "Modelos usados"])
    t = pd.DataFrame([
        ["1", "XGBoost geográfico", "Modelos locales ponderados por distancia", "Sí: modelo principal"],
        ["2", "GWO-XGBoost", "Optimizador metaheurístico de hiperparámetros", "Como idea: E3 compara 4 optimizadores"],
        ["3", "Bagged TAO Trees", "Árboles oblicuos optimizados", "No: sin implementación abierta"],
        ["4", "RF + SVM", "RF selecciona variables, SVM clasifica", "Sí"],
        ["5", "Rotation Forest", "Rotación PCA antes de cada árbol", "Sí"],
        ["6", "EFSHB", "Selección de variables por ensambles", "No: selector de variables"],
        ["7", "TPE-SVM", "SVM con optimización bayesiana", "Sí (sin vecindad)"],
        ["8", "Stacking", "Un metamodelo combina varios modelos", "Sí"],
        ["9", "NSGA-II", "Hiperparámetros y variables a la vez", "No: selector de variables"],
    ], columns=["#", "Candidato original (Tabla III)", "Idea", "¿Se probó?"])
    return html.Div([
        dbc.Row([dbc.Col(kpi("0", "estudios que clasifiquen el estrato con el catastro", "Scopus, 5-oct-2026"), md=4),
                 dbc.Col(kpi("1998", "antecedente más cercano", "logística con datos censales agregados"), md=4),
                 dbc.Col(kpi("5 de 9", "candidatos originales probados", "los otros: optimizadores o selectores"), md=4)],
                className="g-3 mb-3"),
        tabla(usados), html.Div(tabla(t), className="mt-3")])


def _bib_e4():
    lit = F4.leer("literatura.json")
    m = [f for f in lit["matriz"] if f[2] == "23-oct-2026"]
    revisados = sum(int(f[4]) for f in m)
    relevantes = sum(int(f[5]) for f in m)
    return html.Div([
        dbc.Row([dbc.Col(kpi("4 × 6", "bases × cadenas de búsqueda", "Scopus, Web of Science, Google Scholar y arXiv; "
                                                                      "23-oct-2026"), md=4),
                 dbc.Col(kpi(f"{revisados}", "registros revisados (título y resumen)",
                             f"{relevantes} relevantes: mismo método o mismo problema"), md=4),
                 dbc.Col(kpi("0", "estudios que predicen el estrato de viviendas", "con el catastro, con cualquier modelo"),
                         md=4)], className="g-3 mb-3"),
        grafico(F4.fig_literatura(), alto=330),
        html.P([html.B("Cadenas: "), "; ".join(f"{k}: {v.lower()}" for k, v in lit["cadenas"].items()), "."],
               className="nota-visual")])


def _bib_cercanos():
    return html.Div([
        html.P("Los trabajos más cercanos de la revisión del Entregable 4, al problema (el estrato) y al método (combinar "
               "modelos con pesos que cambian según la región):", className="nota-visual"),
        tabla(CERCANOS_E4, alto=470)])


VISTAS_BIB = {"e12": _bib_e12, "e4": _bib_e4, "cercanos": _bib_cercanos}


def d_bibliografia():
    visual = html.Div([controles(control("Revisión", radio(
        "bib-vista", [("e12", "Entregables 1 y 2: candidatos"), ("e4", "Entregable 4: búsqueda de novedad"),
                      ("cercanos", "Trabajos más cercanos")], "e12"))),
        html.Div(_bib_e12(), id="bib-contenido")])
    return diapositiva("1.4", "Revisión bibliográfica: la brecha, los candidatos y la novedad", visual,
                       puntos=["Los trabajos cercanos predicen el precio de la vivienda o la pobreza por zonas, no el "
                               "estrato.",
                               "E4: Scopus, Web of Science, Google Scholar y arXiv con 6 cadenas. Tampoco aparece.",
                               "Lo más parecido al método: combinación convexa por regiones para regresión (Zhou y Lin, "
                               "2026)."],
                       conclusion="Nadie ha predicho el estrato con el catastro: esa es la brecha.",
                       fuente=f"{E2C} · caps. 1.3 y 9.1 · {DE4.E4C} · cap. 4")


@callback(Output("bib-contenido", "children"), Input("bib-vista", "value"))
def _bib(v):
    return VISTAS_BIB[v]()


def d_vacio():
    diferencias = pd.DataFrame([
        ["Objetivo", "Precio de la vivienda, pobreza o un índice socioeconómico del hogar",
         "El estrato oficial, que solo existe en Colombia"],
        ["Unidad", "Zonas (celdas, sectores censales), salvo los estudios de precio", "Cada vivienda"],
        ["Tarea y métrica", "Regresión, evaluada con R²", "Clasificación en 6 clases ordenadas, con F1 macro"],
        ["Datos", "Censo, anuncios inmobiliarios o imágenes satelitales", "Catastro abierto y ubicación"],
    ], columns=["Diferencia", "Trabajos cercanos (Tabla II)", "Este proyecto"])
    consulta = ('TS = (("socioeconomic status" OR "socioeconomic stratum" OR "socioeconomic stratification") '
                'AND ("cadastral data" OR "housing characteristics"))')
    visual = html.Div([
        html.Div("Búsqueda complementaria en Web of Science", className="subtitulo-visual"),
        html.Img(src="/assets/busqueda_wos.png", alt=f"Web of Science Core Collection: 0 resultados para {consulta}",
                 style={"width": "100%", "border": "1px solid #dcdcd6", "borderRadius": "10px"}),
        html.P("Web of Science Core Collection, 19-oct-2026: 0 documentos.",
               style={"fontSize": ".85rem", "color": "#52514e", "marginTop": "6px"}),
        html.Div("Por qué no hay estudios con los que compararse", className="subtitulo-visual mt-3"),
        tabla(diferencias)])
    return diapositiva("1.5", "El vacío en la literatura: por qué no hay con quién compararse", visual,
                       puntos=["El estrato solo existe en Colombia.",
                               "No hay cifras externas con qué compararnos."],
                       conclusion="Todo se mide contra un modelo base propio, con el mismo protocolo.",
                       fuente=f"{E2C} · caps. 1.1 a 1.4")


def d_ruta():
    pasos = [("1", "Descarga", "Catastro abierto"), ("2", "Limpieza", "Reglas por fila"),
             ("3", "Partición", "Bloques de 2 km"), ("4", "EDA", "Solo entrenamiento"),
             ("5", "Pipeline", "Imputar, log, escalar"), ("6", "CV espacial", "kNNDM (v3; antes buffer 1 km)"),
             ("7", "Test", "Una sola vez")]
    hitos = html.Div([
        html.Div([html.Div("Entregable 1", className="hito-titulo"),
                  html.Div("Datos, limpieza, EDA, partición espacial y modelo base (regresión logística)")], className="hito"),
        html.Div([html.Div("Entregable 2", className="hito-titulo"),
                  html.Div("Dashboard, modelos de la revisión bibliográfica y vecindad física")], className="hito"),
        html.Div([html.Div("Versión 2", className="hito-titulo"),
                  html.Div("Ubicación por manzana (cobertura 96.9 %), pesos balanceados y dos escenarios de predicción")],
                 className="hito"),
        html.Div([html.Div("Versión 3", className="hito-titulo"),
                  html.Div("Validación kNNDM, vecindad con grafo KNN y el modelo nuevo con vecindad frente al base")],
                 className="hito"),
        html.Div([html.Div("Entregable 3", className="hito-titulo"),
                  html.Div("156 combinaciones de modelo, balanceo y optimizador con validación anidada, costo "
                           "computacional, pruebas estadísticas y SHAP")], className="hito"),
        html.Div([html.Div("Después del E3 (19-oct-2026)", className="hito-titulo"),
                  html.Div("Todos los modelos originales de la revisión con todos los datos, pruebas de significancia "
                           "y EDA mejorado")], className="hito"),
        html.Div([html.Div("Entregable 4 (proyecto final)", className="hito-titulo"),
                  html.Div("Modelo original (ensamble con pesos locales), validación repetida 5 × 5, pruebas de "
                           "superioridad, test confirmatorio, conjuntos externos y predicción en vivo")],
                 className="hito destacado"),
    ], className="hitos")
    return diapositiva("1.5", "Cómo se trabajó", html.Div([flujo(pasos), hitos]),
                       puntos=["Las reglas que miran una sola fila se aplican antes de partir.",
                               "Todo lo que se aprende de los datos se calcula solo con entrenamiento.",
                               "El test son zonas completas que el modelo nunca vio."],
                       conclusion="El orden de los pasos evita la fuga de datos.", fuente="Entregables 1 a 4")


# ===========================================================================
# 2. DATOS Y CALIDAD
# ===========================================================================
def d_embudo():
    return diapositiva("2.1", "Limpieza: de 382 597 filas a 332 718 viviendas", grafico(F.fig_embudo()),
                       puntos=["Se excluyen usos no habitacionales, parqueaderos y depósitos (< 10 m²).",
                               "También los registros que describen un edificio entero en una fila.",
                               "Valores imposibles o incoherentes pasan a faltante; la fila se conserva."],
                       conclusion="Queda el 87 % de lo descargado: 332 718 viviendas.", fuente=f"{E1C} · cap. 1.5")


def d_manzana():
    visual = html.Div([
        controles(control("Vista", radio("mz-vista", [("umbral", "Umbral de dispersión"), ("mapa", "Mapa")], "umbral")),
                  control("Umbral (m)", dcc.Slider(id="mz-umbral", min=50, max=1000, step=50, value=250,
                                                   marks={50: "50", 250: "250", 500: "500", 1000: "1000"},
                                                   tooltip={"placement": "bottom"}), ancho=340)),
        grafico(FP.fig_recuperacion_umbral(250), "mz-fig")])
    return diapositiva("2.2", "Viviendas sin coordenadas: ubicarlas con su manzana", visual,
                       puntos=["53 894 viviendas (16 %) no tenían coordenadas propias: casi todas informales, estratos 1 y 2.",
                               "Su número predial sí dice su manzana: se usa el centroide de los edificios ubicados.",
                               "Solo en manzanas compactas (≤ 250 m) y nunca la manzana comodín «0000».",
                               "Error de ubicación medido con edificios conocidos: 31 m en la mediana."],
                       conclusion="Se recuperan 43 594 viviendas; quedan sin ubicar 10 300 (3.1 %).",
                       fuente=f"{E1C} v2 · cap. 1.7")


@callback(Output("mz-fig", "figure"), Input("mz-vista", "value"), Input("mz-umbral", "value"))
def _mz(vista, umbral):
    return FP.fig_mapa_recuperadas() if vista == "mapa" else FP.fig_recuperacion_umbral(umbral)


def d_cobertura():
    c1 = COB.loc[1]
    return diapositiva("2.4", "Cobertura por estrato: la ciudad informal entra al modelo", grafico(FP.fig_cobertura_v1_v2()),
                       puntos=[f"Estrato 1: {pct(c1.v1, c1.total, 0)} de las viviendas con coordenadas propias; "
                               f"{pct(c1.v2, c1.total, 0)} con la ubicación por manzana.",
                               f"Estrato 2: {pct(COB.loc[2].v1, COB.loc[2].total, 0)} → {pct(COB.loc[2].v2, COB.loc[2].total, 0)}.",
                               "Los estratos 3 a 6 casi no cambian: ya tenían coordenadas."],
                       conclusion="Ubicar por manzana corrige un sesgo: el estrato 1 estaba subrepresentado.",
                       fuente=f"{E1C} v2 · cap. 1.7")


def d_calidad():
    imp = pd.DataFrame(E1["tabla_imposibles"])[["variable", "rango plausible", "n imposibles"]]
    imp.columns = ["Variable", "Rango plausible", "Filas fuera de rango"]
    visual = html.Div([controles(control("Vista", radio("cal-vista", [("falt", "Faltantes en train"),
                                                                      ("imp", "Reglas de valores imposibles")], "falt"))),
                       html.Div(grafico(FP.fig_faltantes_train()), id="cal-cont")])
    return diapositiva("2.5", "Calidad: faltantes, imposibles e incoherentes", visual,
                       puntos=["Los faltantes son menos del 0.2 % y vienen casi todos de las reglas de calidad.",
                               "Imposible: fuera de lo físicamente posible (piso 99, año 1512).",
                               "Incoherente: posible solo, imposible con el resto de la fila (más baños que habitaciones + 3).",
                               "No faltan al azar, pero dependen de variables observadas (MAR)."],
                       conclusion="Se imputan con la mediana de entrenamiento y un indicador de faltante, dentro del Pipeline.",
                       metricas=[("smd", 2.1, "|SMD| hasta 2.1")], fuente=f"{E1C} · caps. 1.5 y 3.1")


@callback(Output("cal-cont", "children"), Input("cal-vista", "value"))
def _cal(vista):
    if vista == "imp":
        imp = pd.DataFrame(E1["tabla_imposibles"])[["variable", "rango plausible", "n imposibles", "%"]]
        imp.columns = ["Variable", "Rango plausible", "Filas", "% de filas"]
        imp["Variable"] = imp["Variable"].map(FP.et)
        return tabla(imp, 3)
    return grafico(FP.fig_faltantes_train())


def d_outliers():
    return diapositiva("2.6", "Outliers univariados: se marcan, no se borran", grafico(FP.fig_outliers_tukey()),
                       puntos=["Piso de ubicación (10.7 %) y pisos de la vivienda (10 %) encabezan: son torres reales.",
                               "En variables muy asimétricas la regla de Tukey marca muchos valores legítimos.",
                               "Los extremos se recortan en los percentiles 0.1 y 99.9 (winsorización)."],
                       conclusion="Ninguna vivienda se elimina por ser atípica.",
                       metricas=[("iqr", 10.7, "10.7 % (piso de ubicación)")], fuente=f"{E1C} · cap. 3.2")


def d_dependencia():
    visual = html.Div([
        dbc.Row([dbc.Col(kpi(f"{GX['icc']:.3f}", "ICC(1) del estrato", "dentro del predio matriz"), md=4),
                 dbc.Col(kpi(f"{GX['deff']:.0f}", "efecto de diseño", "deff = 1 + (m̃ − 1)·ICC"), md=4),
                 dbc.Col(kpi(f"≈ {fmt(round(GX['n_ef'], -2))}", "tamaño efectivo", f"de {fmt(E1['viviendas'])} filas"), md=4)],
                className="g-3 mb-2"),
        grafico(F.fig_kish(round(GX["icc"], 3), GX["m_pond"]), "kish-fig"),
        control("Prueba otro ICC", dcc.Slider(id="kish-icc", min=0, max=1, step=0.001, value=round(GX["icc"], 3),
                                              marks={0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1"},
                                              tooltip={"placement": "bottom"}))])
    return diapositiva("2.7", "Las filas no son independientes", visual,
                       puntos=["Dentro de un edificio o conjunto el estrato casi nunca cambia.",
                               f"Visto desde una vivienda típica, su grupo tiene {GX['m_pond']:.0f} unidades.",
                               "Las 332 718 filas informan como unas 2 300 viviendas independientes."],
                       conclusion="Por eso se parte por zonas y se remuestrean bloques, no filas.",
                       metricas=[("icc", GX["icc"]), ("deff", GX["deff"], f"{GX['deff']:.0f}")], fuente=f"{E1C} · cap. 1.9")


@callback(Output("kish-fig", "figure"), Input("kish-icc", "value"))
def _kish(icc):
    return F.fig_kish(GX["icc"] if icc is None else icc, GX["m_pond"])


def d_particion():
    m = FP._pkl("particion_mapa.pkl.gz")
    tb = FM.eda()["test_por_bloque"]
    tam = [v["total"] for v in tb.values()]
    excl = int(m.loc[m.grupo == "excluida por el buffer", "n"].sum())
    visual = html.Div([controles(control("Vista", radio("part-vista", [("grupo", "Entrenamiento y test"),
                                                                       ("balance", "Mezcla de estratos"),
                                                                       ("anterior", "Buffer de la versión anterior")],
                                                        "grupo"))),
                       grafico(FP.fig_particion_mapa("grupo"), "part-fig")])
    return diapositiva("2.8", "Partición espacial: el test son zonas que el modelo nunca vio", visual,
                       puntos=[f"La ciudad se divide en bloques de 2 km ({m.bloque.nunique()} con viviendas): cada bloque "
                               "va completo a un solo lado, así que dos vecinas casi idénticas no quedan separadas.",
                               f"Test: {fmt(R['n_test'])} viviendas en {R['bloques_test']} bloques (20 %). Un algoritmo "
                               "reparte los bloques para equilibrar los estratos: por eso quedan dispersos por la ciudad.",
                               f"Los bloques son muy desiguales ({fmt(min(tam))} a {fmt(max(tam))} viviendas): el test tiene "
                               "más estrato 1 y todo su estrato 6 en un solo bloque (EDA: el test).",
                               f"El modelo entrena con las {fmt(R['n_train'])} viviendas restantes. El buffer de 1 km de la "
                               f"versión anterior quitaba {fmt(excl)} (vista «Buffer de la versión anterior»)."],
                       conclusion="Así se mide lo que importa: predecir en barrios sin datos.",
                       fuente=f"{E1C} · caps. 2 y 14")


@callback(Output("part-fig", "figure"), Input("part-vista", "value"))
def _part(vista):
    return FP.fig_balance_particion() if vista == "balance" else FP.fig_particion_mapa(vista)


# ===========================================================================
# 3. ANÁLISIS EXPLORATORIO
# ===========================================================================
def d_objetivo():
    visual = html.Div([controles(control("Vista", radio("obj-vista", [("dist", "Distribución"), ("ns", "Sur → norte"),
                                                                      ("mapa", "Mapa")], "dist"))),
                       grafico(F.fig_distribucion_estrato("train"), "obj-fig")])
    return diapositiva("3.1", "La variable objetivo: desbalanceada, ordinal y segregada", visual,
                       puntos=["El estrato 1 es el 30 % de train; el 6, menos del 5 % (razón 6.2:1).",
                               "Predecir siempre el estrato 1 acertaría 3 de cada 10 viviendas sin aprender nada.",
                               "La ciudad sube de estrato de sur a norte, pero no en línea recta."],
                       conclusion="Métrica principal: F1 macro, y validación espacial.",
                       metricas=[("f1_macro", None)], fuente=f"{E1C} · cap. 4")


@callback(Output("obj-fig", "figure"), Input("obj-vista", "value"))
def _obj(v):
    return {"dist": lambda: F.fig_distribucion_estrato("train"), "ns": lambda: F.fig_gradiente_norte_sur("train"),
            "mapa": lambda: F.fig_mapa("estrato_num", "train")}[v]()


def d_numericas():
    visual = html.Div([controles(control("Variable", desplegable("num-var", OPC_NUM, "area_construida")),
                                 control("Escala", interruptor("num-log", "log(1 + x)", True))),
                       grafico(F.fig_numerica("area_construida"), "num-fig")])
    return diapositiva("3.2", "Variables numéricas: colas largas a la derecha", visual,
                       puntos=["Casi todas son muy asimétricas: pocas viviendas enormes.",
                               "A mayor estrato, más área y más baños, con mucho traslape entre vecinos.",
                               "Altura casi siempre vale 3 m: no informa."],
                       conclusion="Se transforman con log(1 + x) y se resumen con mediana e IQR.",
                       metricas=[("asimetria", 4.05, "4.05 (área construida)"), ("curtosis", None)],
                       fuente=f"{E1C} · cap. 5")


@callback(Output("num-fig", "figure"), Input("num-var", "value"), Input("num-log", "value"))
def _num(var, log):
    return F.fig_numerica(var, bool(log))


def d_transformar():
    return diapositiva("3.3", "¿Hace falta transformar? Sí, con log(1 + x)", grafico(FP.fig_transformacion()),
                       puntos=["Área construida: asimetría de 4.05 a 0.35.",
                               "Área de terreno: de 56 a −1.05.",
                               "Altura no se transforma: es casi constante."],
                       conclusion="El log acerca las variables a la simetría y reduce el peso de los extremos.",
                       metricas=[("asimetria", 0.35, "0.35 tras el log (área)"), "normalidad"], fuente=f"{E1C} · cap. 5.5")


def d_correlaciones():
    visual = html.Div([controles(control("Vista", radio("corr-vista", [("spearman", "Spearman"), ("pearson", "Pearson"),
                                                                       ("pares", "Pares con más diferencia")], "spearman"))),
                       grafico(FP.fig_correlaciones("spearman"), "corr-fig")])
    return diapositiva("3.4", "Correlaciones: Spearman frente a Pearson", visual,
                       puntos=["Las variables de tamaño (área, baños, habitaciones) se mueven juntas.",
                               "El terreno baja con el piso: en un edificio cada apartamento tiene una cuota del lote.",
                               "Cuando Pearson y Spearman difieren, mandan las colas: se prefiere Spearman."],
                       conclusion="Hay redundancia, pero ninguna pareja obliga a quitar una variable.",
                       metricas=["spearman"], fuente=f"{E1C} · cap. 6.1")


@callback(Output("corr-fig", "figure"), Input("corr-vista", "value"))
def _corr(v):
    return FP.fig_pearson_vs_spearman() if v == "pares" else FP.fig_correlaciones(v)


def d_hexbin():
    visual = html.Div([controles(control("Eje x", desplegable("hex-x", OPC_NUM, "area_construida", 260)),
                                 control("Eje y", desplegable("hex-y", OPC_NUM, "total_banios", 260)),
                                 control("Escala", interruptor("hex-log", "log(1 + x)", True))),
                       grafico(FP.fig_hexbin(), "hex-fig")])
    return diapositiva("3.5", "Dos variables a la vez (densidad)", visual,
                       puntos=["Cada celda cuenta viviendas: el color muestra dónde se concentran.",
                               "Con cientos de miles de puntos, un diagrama de dispersión se vuelve una mancha.",
                               "Prueba área construida frente a baños, o terreno frente a piso."],
                       conclusion="Las relaciones son monótonas, no lineales, y con mucha dispersión.",
                       metricas=["spearman"], fuente=f"{E1C} · cap. 6.1")


@callback(Output("hex-fig", "figure"), Input("hex-x", "value"), Input("hex-y", "value"), Input("hex-log", "value"))
def _hex(x, y, log):
    return FP.fig_hexbin(x, y, bool(log))


def d_violin():
    ef = F.efecto_numerica("total_banios")
    visual = html.Div([controles(control("Variable", desplegable("vio-var", OPC_NUM, "total_banios")),
                                 control("Escala", interruptor("vio-log", "log(1 + x)", True))),
                       grafico(FP.fig_violin("total_banios"), "vio-fig")])
    return diapositiva("3.6", "Numéricas frente al estrato", visual,
                       puntos=["Posición norte-sur: el efecto más grande (η² 0.46).",
                               "Entre las físicas: baños (0.33), piso de ubicación (0.25) y área construida (0.17).",
                               "Pisos de la vivienda y altura casi no separan los estratos."],
                       conclusion="La ubicación separa más que cualquier característica física.",
                       metricas=[html.Div(id="vio-ficha", children=_ficha_eta(ef["eta2_H"], "Baños"))],
                       fuente=f"{E1C} · cap. 6.2")


def _ficha_eta(v, nombre):
    from componentes import ficha
    return ficha("eta2", v, f"{v:.3f} ({nombre})")


@callback(Output("vio-fig", "figure"), Output("vio-ficha", "children"), Input("vio-var", "value"), Input("vio-log", "value"))
def _vio(var, log):
    return FP.fig_violin(var, bool(log)), _ficha_eta(F.efecto_numerica(var)["eta2_H"], F.NUMERICAS[var])


def d_categoricas():
    visual = html.Div([controles(control("Variable", desplegable("cat-var", list(F.CATEGORICAS.items()), "condicion_predio"))),
                       grafico(F.fig_categorica("condicion_predio"), "cat-fig")])
    return diapositiva("3.7", "Categóricas frente al estrato", visual,
                       puntos=["Régimen: los informales son casi todos estratos 1 y 2; las unidades en PH, medios y altos.",
                               "Uso y tipo de vivienda aportan menos y repiten lo que dice el régimen.",
                               "Categorías con menos del 1 % se agrupan en «infrecuentes»."],
                       conclusion="El régimen de propiedad es la categórica más informativa (V = 0.33).",
                       metricas=[("cramer", 0.326, "0.33 (régimen)")], fuente=f"{E1C} · cap. 6.3")


@callback(Output("cat-fig", "figure"), Input("cat-var", "value"))
def _cat(var):
    return F.fig_categorica(var)


def d_ranking():
    return diapositiva("3.8", "¿Qué variable dice más del estrato?", grafico(FP.fig_ranking_asociacion()),
                       puntos=["La ubicación encabeza: norte-sur, este-oeste y distancia al centro.",
                               "El terreno informa mucho pero de forma no monótona (η² bajo, MI alta).",
                               "Destinación económica no informa nada: se excluye."],
                       conclusion="Ubicación primero; régimen, área y baños después.",
                       metricas=[("mi", 0.618, "0.62 (norte-sur)")], fuente=f"{E1C} · cap. 6.6–6.7")


def d_vif():
    return diapositiva("3.9", "Multicolinealidad", grafico(FP.fig_vif()),
                       puntos=["Las variables físicas tienen VIF menor que 2.",
                               "Posición este-oeste y distancia al centro rondan 6: la distancia se calcula con x e y.",
                               "Con regularización L2 la logística tolera esa redundancia."],
                       conclusion="No hace falta quitar variables por colinealidad.",
                       metricas=[("vif", 6.1, "6.1 (este-oeste)")], fuente=f"{E1C} · cap. 6.8")


def d_pca():
    v = np.array(MV["varianza"])
    visual = html.Div([controles(control("Vista", radio("pca-vista", [("scree", "Varianza"), ("cargas", "Cargas")], "scree"))),
                       grafico(FP.fig_scree(), "pca-fig")])
    return diapositiva("3.10", "PCA: ¿cuántas dimensiones tienen los datos?", visual,
                       puntos=[f"PC1 resume el {v[0]:.0%} (tamaño de la vivienda) y PC2 el {v[1]:.0%} (edificio frente a casa).",
                               "Se necesitan 5 componentes para el 80 %: las variables son poco redundantes.",
                               f"PC1 es el eje más asociado al estrato (Spearman {MV['spearman_pc_estrato'][0]:.2f})."],
                       conclusion="No hay un atajo de pocas dimensiones: se usan las variables originales.",
                       metricas=[("participation_ratio", None, f"{MV['participation_ratio']:.1f} de 8"),
                                 ("cargas", None)], fuente=f"{E1C} · cap. 7.1")


@callback(Output("pca-fig", "figure"), Input("pca-vista", "value"))
def _pca(v):
    return FP.fig_cargas() if v == "cargas" else FP.fig_scree()


def d_biplot():
    pcs = [(f"PC{i}", f"PC{i}") for i in range(1, 5)]
    visual = html.Div([controles(control("Eje x", desplegable("bi-x", pcs, "PC1", 120)),
                                 control("Eje y", desplegable("bi-y", pcs, "PC2", 120)),
                                 control("Color", radio("bi-color", [("estrato", "Estrato"), ("condicion", "Régimen")], "estrato"))),
                       grafico(FP.fig_biplot(), "bi-fig")])
    return diapositiva("3.11", "Biplot: viviendas y variables en el mismo plano", visual,
                       puntos=["Cada punto es una vivienda; cada flecha, una variable.",
                               "Flechas en la misma dirección: variables que se mueven juntas.",
                               "Los estratos se ordenan a lo largo de PC1, pero se superponen mucho."],
                       conclusion="Las características físicas ordenan el estrato, pero no lo separan.",
                       metricas=["cargas"], fuente=f"{E1C} · cap. 7.1")


@callback(Output("bi-fig", "figure"), Input("bi-x", "value"), Input("bi-y", "value"), Input("bi-color", "value"))
def _bi(x, y, c):
    return FP.fig_biplot(x, y, c)


def d_outliers_mv():
    o = MV["outliers"]
    return diapositiva("3.12", "Outliers multivariados", grafico(FP.fig_outliers_multivariados()),
                       puntos=[f"Mahalanobis robusta marca el {o['pct_mahalanobis']:.1f} %; Isolation Forest, el {o['pct_isolation']:.1f} %.",
                               f"Solo el {o['pct_ambos']:.2f} % lo marcan ambos.",
                               "Entre ellos hay el doble de estratos 5 y 6 que en el resto: casas grandes reales."],
                       conclusion="Se conservan: quitarlos sesgaría los estratos altos.",
                       metricas=["mahalanobis", ("jaccard", o["jaccard"])], fuente=f"{E1C} · cap. 7.2")


def d_clustering():
    visual = html.Div([controles(control("Número de grupos k", dcc.Slider(id="km-k", min=2, max=8, step=1, value=MV["k_opt"],
                                                                         marks={k: str(k) for k in range(2, 9)}), ancho=320),
                                 control("Vista", radio("km-vista", [("grupos", "Grupos"), ("silueta", "Silueta")], "grupos"))),
                       grafico(FP.fig_kmeans(), "km-fig")])
    s = max(MV["silueta"].values())
    return diapositiva("3.13", "Clustering: ¿hay tipos de vivienda?", visual,
                       puntos=[f"La mejor silueta es {s:.2f} (k = {MV['k_opt']}): grupos débiles.",
                               "Los grupos son tramos de un continuo (de casa pequeña a apartamento grande).",
                               "Cada grupo mezcla varios estratos."],
                       conclusion="No hay grupos naturales: el estrato no sale de la tipología sola.",
                       metricas=[("silueta", s)], fuente=f"{E1C} · cap. 7.3")


@callback(Output("km-fig", "figure"), Input("km-k", "value"), Input("km-vista", "value"))
def _km(k, v):
    return FP.fig_silueta() if v == "silueta" else FP.fig_kmeans(k)


def d_tsne():
    visual = html.Div([controles(control("Color", radio("ts-color", [("estrato", "Estrato"), ("cluster", "Grupo K-Means"),
                                                                     ("condicion", "Régimen")], "estrato"))),
                       grafico(FP.fig_tsne(), "ts-fig")])
    return diapositiva("3.14", "t-SNE: el mapa de parecidos físicos", visual,
                       puntos=["Viviendas parecidas quedan cerca; las distancias grandes no se interpretan.",
                               "Se forman islas por régimen y tipología, no por estrato.",
                               "Los colores de estrato se mezclan dentro de cada isla."],
                       conclusion="Lo físico agrupa tipologías; el estrato lo decide sobre todo la ubicación.",
                       fuente=f"{E1C} · cap. 7.3")


@callback(Output("ts-fig", "figure"), Input("ts-color", "value"))
def _ts(c):
    return FP.fig_tsne(c)


def d_mapa():
    opciones = [("estrato_num", "Estrato medio")] + [(k, v) for k, v in F.NUMERICAS.items() if k != "dist_centro_km"]
    visual = html.Div([controles(control("Variable", desplegable("mapa-var", opciones, "estrato_num"))),
                       grafico(F.fig_mapa("estrato_num", "train"), "mapa-fig")])
    return diapositiva("3.15", "El estrato en el mapa", visual,
                       puntos=["Norte y nororiente concentran los estratos 4 a 6.",
                               "Sur y suroccidente son casi todo estratos 1 y 2.",
                               "Los huecos rectangulares son los bloques de test: el EDA no los mira."],
                       conclusion="Barranquilla está segregada: la ubicación será la información dominante.",
                       fuente=f"{E1C} · cap. 8.2")


@callback(Output("mapa-fig", "figure"), Input("mapa-var", "value"))
def _mapa(var):
    return F.fig_mapa(var, "train")


def d_moran():
    m = D["moran_estrato"]
    return diapositiva("3.16", "Autocorrelación espacial: los vecinos se parecen", grafico(F.fig_correlograma()),
                       puntos=[f"I de Moran del estrato: {m['I']:.2f} (8 edificios vecinos, p = {m['p']:.3f}).",
                               "El parecido baja con la distancia y se vuelve débil hacia los 2.6 km.",
                               "Por eso se valida separando zonas, a la distancia a la que se predice (kNNDM, pestaña Modelos)."],
                       conclusion="Validar con filas al azar sería hacer trampa: hay que separar zonas.",
                       metricas=[("moran", m["I"]), "correlograma"], fuente=f"{E1C} · cap. 8.4")


def d_lisa():
    visual = html.Div([controles(control("Indicador", radio("lisa-modo", [("lisa", "LISA (Moran local)"), ("gi", "Gi*")], "lisa"))),
                       grafico(FP.fig_lisa("lisa"), "lisa-fig")])
    return diapositiva("3.17", "¿Dónde están los clusters?", visual,
                       puntos=["29 % de los edificios: bajo rodeado de bajo (sur y occidente).",
                               "17 %: alto rodeado de alto (norte).",
                               "Los atípicos espaciales (HL, LH) son 0.1 %: casi no hay islas."],
                       conclusion="Los barrios son homogéneos; los errores se concentrarán en sus bordes.",
                       metricas=["lisa", "gi"], fuente=f"{E1C} · cap. 8.4")


@callback(Output("lisa-fig", "figure"), Input("lisa-modo", "value"))
def _lisa(m):
    return FP.fig_lisa(m)


def d_maup():
    return diapositiva("3.18", "La escala importa (MAUP)", grafico(FP.fig_maup()),
                       puntos=["Con celdas de 250 m el I de Moran es 0.94; con celdas de 4 km, 0.12.",
                               "Al agregar mucho, las zonas mezclan barrios distintos.",
                               "Por eso el análisis espacial se hace por edificio, no por celdas grandes."],
                       conclusion="La autocorrelación es local: de cientos de metros, no de kilómetros.",
                       metricas=["moran"], fuente=f"{E1C} · cap. 8.6")


def d_fuga():
    return diapositiva("3.19", "Auditoría de fuga de datos", grafico(FP.fig_auc_univariado()),
                       puntos=["Ninguna variable sola llega a 0.85 de AUC: no hay proxies del estrato.",
                               "La más alta es la posición norte-sur (0.75), y es legítima.",
                               "Identificadores y metadatos del proceso quedan fuera del modelo."],
                       conclusion="Las variables del modelo están disponibles al predecir y no se derivan del estrato.",
                       metricas=[("auc_univariado", 0.746, "0.75 (norte-sur)")], fuente=f"{E1C} · cap. 9")


# --- EDA mejorado (Entregable 1, versión 3, cap. 15): hallazgos que el EDA original no había visto -------------
def d_eda_entorno():
    d = FM.eda()
    eta = d["eta2"]
    ban, area = eta["baños"], eta["área construida (log)"]
    return diapositiva("3.20", "EDA mejorado: el entorno dice más del estrato que la vivienda", grafico(FM.fig_eta2_entorno()),
                       puntos=["Para cada vivienda se promedian seis atributos en sus 8 edificios más cercanos, sin usar "
                               "el estrato de nadie: la vecindad física.",
                               f"En los seis, el promedio de los vecinos se asocia más con el estrato que el atributo "
                               f"propio: baños {ban['vecinos']:.2f} frente a {ban['propia']:.2f}; área "
                               f"{area['vecinos']:.2f} frente a {area['propia']:.2f}.",
                               f"Hasta la distancia media a los vecinos (una medida de densidad) tiene η² = "
                               f"{d['eta2_distancia_vecinos']:.2f}.",
                               "Con estas variables, el F1 de test de la logística sube de 0.49 a 0.57 (pestaña Modelos)."],
                       conclusion="El estrato es una propiedad de la zona: describir el entorno informa más que describir "
                                  "la vivienda.",
                       metricas=[("eta2", ban["vecinos"], f"{ban['vecinos']:.2f} (baños de los vecinos)")],
                       fuente=f"{E1C} · cap. 15.2 (v3)")


def d_eda_fronteras():
    d = FM.eda()
    fr = d["fronteras"]
    visual = html.Div([controles(control("Medida", radio("fr-vista", [("pct", "Viviendas en frontera"),
                                                                       ("vec", "Vecinos con otro estrato")], "pct"))),
                       grafico(FM.fig_fronteras("pct"), "fr-fig")])
    return diapositiva("3.21", "EDA mejorado: las fronteras entre estratos", visual,
                       puntos=[f"El {fr['todas']['pct_frontera']:.0f} % de las viviendas está en una frontera: la mitad "
                               "o más de sus 8 edificios vecinos tiene otro estrato.",
                               f"El estrato 5 es el más fronterizo ({fr['5']['pct_frontera']:.0f} %); el 1 y el 2, los "
                               f"más homogéneos ({fr['2']['pct_frontera']:.0f}–{fr['1']['pct_frontera']:.0f} %).",
                               f"El {d['pct_un_solo_estrato']:.0f} % de esas diferencias es de un solo estrato.",
                               "Explica por qué los errores son casi siempre de un estrato, el 5 es el más difícil y "
                               "ADASYN es el peor balanceo (Entregable 3, pestaña Modelos)."],
                       conclusion="La dificultad está donde las zonas se tocan, no repartida por la ciudad.",
                       fuente=f"{E1C} · cap. 15.3 (v3)")


@callback(Output("fr-fig", "figure"), Input("fr-vista", "value"))
def _fr(v):
    return FM.fig_fronteras(v)


def d_eda_test():
    d = FM.eda()
    c = d["composicion"]
    tr, te = c["entrenamiento (%)"], c["prueba (%)"]
    e6 = d["test_por_bloque"]["4004"]["E6"]
    visual = html.Div([controles(control("Vista", radio("ct-vista", [("estratos", "Por estrato"),
                                                                     ("bloques", "Por bloque de test")], "estratos"))),
                       grafico(FM.fig_composicion_test("estratos"), "ct-fig")])
    return diapositiva("3.22", "EDA mejorado: el test no tiene la composición del entrenamiento", visual,
                       puntos=[f"El test tiene más estrato 1 ({te['1']:.0f} % frente a {tr['1']:.0f} %) y mucho menos "
                               f"estrato 6 ({te['6']:.1f} % frente a {tr['6']:.1f} %).",
                               f"Sus {e6} viviendas de estrato 6 están en un solo bloque (4004); otro bloque tiene 3 "
                               "viviendas.",
                               "El F1 del estrato 6 pesa un sexto del F1 macro y depende de un bloque: por eso los "
                               "intervalos del test son anchos.",
                               "Con 7 bloques el p-valor mínimo posible es 0.016: ninguna diferencia pequeña resiste la "
                               "corrección por comparaciones múltiples."],
                       conclusion="Diagnóstico hecho después de evaluar, sin cambiar decisiones: las conclusiones firmes "
                                  "son las que aparecen en validación y en test.",
                       fuente=f"{E1C} · cap. 15.4 (v3)")


@callback(Output("ct-fig", "figure"), Input("ct-vista", "value"))
def _ct(v):
    return FM.fig_composicion_test(v)


# ===========================================================================
# 4. MODELO BASE
# ===========================================================================
def d_protocolo():
    o = R["optimismo_validacion_aleatoria"]
    a, e = o["F1 macro CV"]["aleatoria (StratifiedKFold)"], o["F1 macro CV"]["espacial (kNNDM)"]
    return diapositiva("5.4", "Por qué separar zonas al validar", grafico(FM.fig_optimismo_v3()),
                       puntos=[f"Mezclando filas al azar, el modelo base parece lograr un F1 de {a:.2f}; validando con "
                               f"zonas nuevas (kNNDM), {e:.2f}; en el test, {T['f1_macro'][B]:.2f}.",
                               "La diferencia es fuga: con filas al azar, cada vivienda de validación tiene vecinas casi "
                               "idénticas (las de su edificio) en el entrenamiento.",
                               "kNNDM no exagera ni castiga: valida a la misma distancia a la que se predice."],
                       conclusion="Validar como se va a usar el modelo: en barrios sin datos.",
                       metricas=[("optimismo", a - e), "knndm"], fuente=f"{E1C} · caps. 10.2 y 14")


def d_cambios():
    d = FP.v3_e1()
    q = d["distancias"]["cuantiles"]
    cvt = d["cv_vs_test"]
    v2, v3 = cvt["v2: bloques + buffer de 1 km"], cvt["v3: kNNDM sin buffer"]
    versiones = pd.DataFrame([
        ["1 (entregada)", "Bloques de 2 km y buffer de 1 km", "Primera entrega"],
        ["2", "Ubicación por manzana (cobertura 83.8 % → 96.9 %), pesos de clase, vecindad física",
         "Las viviendas sin coordenadas eran casi todas de estratos 1 y 2"],
        ["3 (la de este dashboard)", "Validación kNNDM en lugar del buffer; entrenamiento sin buffer; vecindad con grafo KNN",
         "Sugerencia del profesor: el buffer evaluaba un escenario más difícil que el real"],
        ["Después del E3", "Todos los modelos originales con todos los datos; pruebas de significancia; EDA mejorado",
         "El objetivo es probar modelos no usados antes, no solo el XGBoost geográfico"],
    ], columns=["Versión", "Qué cambió", "Por qué"])
    visual = html.Div([controles(control("Vista", radio("camb-vista", [("cvtest", "Validación y test"),
                                                                       ("buffer", "¿Por qué era de 1 km?")], "cvtest"))),
                       grafico(FM.fig_cambios("cvtest"), "camb-fig"), html.Div(tabla(versiones), className="mt-2")])
    return diapositiva("1.7", "Qué cambió en la versión 3 (y por qué)", visual,
                       puntos=["El buffer de 1 km salía del correlograma: el parecido entre viviendas se pierde hacia "
                               "2.6 km, y se recortó a 1 km porque más buffer quitaba demasiado entrenamiento.",
                               f"Pero con él se validaba a {q['CV de la versión 2 (bloques + buffer de 1 km)']['mediana (km)']:.1f} km "
                               f"del dato más cercano, cuando una zona nueva se predice a "
                               f"{q['Escenario: zona nueva de 2 km (test → todo train)']['mediana (km)']:.2f} km.",
                               "Sugerencia del profesor: armar los pliegues con kNNDM, que reproduce esa distancia, y "
                               "entrenar con todo el entrenamiento.",
                               f"Efecto en el modelo base: validación {v2['F1 macro CV']:.2f} → {v3['F1 macro CV']:.2f} con el "
                               f"mismo test ({v3['F1 macro test']:.2f}); la brecha test − validación pasa de "
                               f"{v2['test − CV']:+.2f} a {v3['test − CV']:+.2f}."],
                       conclusion="Todo este dashboard usa la versión 3; las anteriores siguen publicadas en los libros.",
                       metricas=["buffer", "knndm", ("dist_vecino", q["Escenario: zona nueva de 2 km (test → todo train)"]["mediana (km)"],
                                                     "0.45 km (escenario)")],
                       fuente=f"{E1C} · caps. 8, 10 y 14 · {E2C} · apéndice A")


@callback(Output("camb-fig", "figure"), Input("camb-vista", "value"))
def _camb(v):
    return FM.fig_cambios(v)


def hp_texto(nombre):
    h = R["mejores_hiperparametros"][nombre.replace("Logística ", "")]
    pesos = "pesos balanceados" if h["clf__class_weight"] == "balanced" else "sin pesos de clase"
    return f"C = {float(h['clf__C']):g}, {pesos}"


def d_grid():
    fig = FP.fig_grid()
    visual = grafico(fig) if fig is not None else html.P("La búsqueda de hiperparámetros aún se está calculando.")
    g = pd.DataFrame(json.loads((F.BASE / "grid_cv.json").read_text(encoding="utf-8")))
    gb = g[g.conjunto == B.replace("Logística ", "")].f1_cv.map(np.mean)
    return diapositiva("4.1", "Hiperparámetros: el más simple que no pierde", visual,
                       puntos=["Se probaron C = 0.01, 0.1, 1 y 10, con y sin pesos de clase, con los 5 pliegues kNNDM.",
                               f"La superficie es plana: en B, el F1 de validación va de {gb.min():.3f} a {gb.max():.3f}.",
                               f"La regla 1-SE elige el modelo más regularizado: B, {hp_texto(B)}; A (solo físicas), "
                               f"{hp_texto('A_fisicas')}.",
                               f"F1 de validación del modelo base: {CV['f1_macro (media)'][B]:.3f}."],
                       conclusion="Ante un empate, se prefiere el modelo más simple.",
                       metricas=["C", "class_weight", "se_pareado"], fuente=f"{E1C} · cap. 14.3")


def comparacion(base):
    c = R["comparaciones_f1"]
    k = f"{B} − {base}"
    return float(c["diferencia"][k]), float(c["IC95_inf"][k]), float(c["IC95_sup"][k])


def d_comparacion():
    visual = html.Div([controles(control("Métrica", desplegable("comp-met", list(F.METRICAS_COMPARABLES.items()), "f1_macro")),
                                 control("Vista", radio("comp-vista", [("fig", "Gráfica"), ("tabla", "Tabla")], "fig"))),
                       html.Div(grafico(F.fig_comparacion("f1_macro")), id="comp-cont")])
    dm, lm, hm = comparacion("Moda por zona")
    return diapositiva("4.2", "Modelo base frente a las líneas base", visual,
                       puntos=[f"Logística B: F1 {CV['f1_macro (media)'][B]:.3f} en validación kNNDM y "
                               f"{T['f1_macro'][B]:.3f} en test.",
                               "Supera con claridad a las Dummy en todas las métricas.",
                               f"La moda por zona ({R.get('tam_zona_moda_km', 4):g} km) usa solo la ubicación: B la supera "
                               f"por {dm:+.3f} de F1 en test, " + ("con el intervalo lejos del 0." if lm > 0 else
                                                                  "pero el intervalo incluye el 0.")],
                       conclusion="La logística con variables físicas y ubicación es la referencia del proyecto.",
                       metricas=[("f1_macro", T["f1_macro"][B]), ("mae_ordinal", T["MAE_ordinal"][B]),
                                 ("kappa", T["kappa_cuadrático"][B])], fuente=f"{E1C} · caps. 10.4 y 14")


@callback(Output("comp-cont", "children"), Input("comp-met", "value"), Input("comp-vista", "value"))
def _comp(met, vista):
    return tabla(F.tabla_metricas()) if vista == "tabla" else grafico(F.fig_comparacion(met))


def fig_significancia():
    c = R["comparaciones_f1"]
    nombres = list(c["diferencia"])
    dif = np.array([float(c["diferencia"][n]) for n in nombres])
    lo = np.array([float(c["IC95_inf"][n]) for n in nombres])
    hi = np.array([float(c["IC95_sup"][n]) for n in nombres])
    et = [n.replace("Logística B_fisicas+ubicacion", "B").replace("Logística A_fisicas", "A (físicas)")
          .replace("Moda por zona", f"moda por zona") for n in nombres]
    fig = go.Figure(go.Scatter(x=dif, y=et, mode="markers", marker=dict(size=15, color=["#2a78d6" if l > 0 else "#a3a29d" for l in lo]),
                               error_x=dict(type="data", symmetric=False, array=hi - dif, arrayminus=dif - lo, color="#52514e", thickness=2),
                               customdata=np.c_[lo, hi],
                               hovertemplate="%{y}: %{x:+.3f} [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]<extra></extra>"))
    fig.add_vline(x=0, line_dash="dot", line_color="#52514e")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="diferencia de F1 macro en test (IC 95 % por bloques)")
    fig.update_layout(title="¿Las diferencias son reales? Azul: el intervalo excluye el 0")
    return F._estilo(fig, alto=400, leyenda=False)


def d_significancia():
    c = R["comparaciones_f1"]
    gana = [n.split(" − ", 1)[1] for n in c["diferencia"] if float(c["IC95_inf"][n]) > 0]
    no = [n.split(" − ", 1)[1] for n in c["diferencia"] if float(c["IC95_inf"][n]) <= 0]
    nombre = lambda n: F.NOMBRE_CORTO[n] if n in F.NOMBRE_CORTO else n  # noqa: E731
    return diapositiva("4.3", "¿Las ventajas son reales o ruido?", grafico(fig_significancia()),
                       puntos=["Bootstrap por bloques: se remuestrean zonas, no viviendas.",
                               "B gana con el intervalo lejos del 0 a: " + ", ".join(nombre(n) for n in gana) + ".",
                               ("El intervalo incluye el 0 frente a: " + ", ".join(nombre(n) for n in no) + "."
                                if no else "Todas las diferencias excluyen el 0.")],
                       conclusion="La ubicación aporta de forma significativa frente a las variables físicas solas.",
                       metricas=["ic95"], fuente=f"{E1C} · caps. 10.4 y 14")


OPC_BASE = [(m, F.NOMBRE_CORTO[m]) for m in F.MODELOS]


def d_confusion():
    visual = html.Div([controles(control("Modelo", desplegable("conf-mod", OPC_BASE, B, 300)),
                                 control("Vista", radio("conf-vista", [("matriz", "Matriz"), ("clase", "Por estrato")], "matriz")),
                                 control("Valores", interruptor("conf-norm", "% de la fila", True))),
                       grafico(F.fig_confusion(B), "conf-fig")])
    rc = F.tabla_por_clase(B)
    rec = rc.Recall.to_numpy()
    orden = np.argsort(rec)
    y, pr, _ = F._pred(B)
    peor = F.CLASES[orden[0]]
    otros = pd.Series(pr[(y == peor) & (pr != peor)]).value_counts()
    return diapositiva("4.4", "¿En qué se equivoca?", visual,
                       puntos=[f"Mejor reconocidos: estratos {F.CLASES[orden[-1]]} y {F.CLASES[orden[-2]]} (recall "
                               f"{rec[orden[-1]]:.2f} y {rec[orden[-2]]:.2f}).",
                               f"El más difícil: el estrato {peor} (recall {rec[orden[0]]:.2f}), que se confunde sobre todo "
                               f"con el {int(otros.index[0])}.",
                               "Sin pesos de clase, el modelo favorece los estratos grandes; casi todos los errores son con "
                               "un estrato vecino."],
                       conclusion=f"El {T['accuracy_±1'][B]:.0%} de las predicciones cae en el estrato real o en uno vecino.",
                       metricas=[("recall", rec[orden[0]], f"{rec[orden[0]]:.2f} (estrato {peor})"), ("acc1", T["accuracy_±1"][B])],
                       fuente=f"{E1C} · cap. 14.4")


@callback(Output("conf-fig", "figure"), Input("conf-mod", "value"), Input("conf-vista", "value"), Input("conf-norm", "value"))
def _conf(m, v, n):
    return F.fig_por_clase(m) if v == "clase" else F.fig_confusion(m, bool(n))


def d_roc():
    from sklearn.metrics import roc_auc_score
    visual = html.Div([controles(control("Modelo", desplegable("roc-mod", OPC_BASE, B, 300)),
                                 control("Vista", radio("roc-vista", [("roc", "Curvas ROC"), ("cal", "Calibración")], "roc"))),
                       grafico(F.fig_roc(B), "roc-fig")])
    y, _, proba = F._pred(B)
    auc_k = {k: roc_auc_score(y == k, proba[:, i]) for i, k in enumerate(F.CLASES)}
    mejor, peor = max(auc_k, key=auc_k.get), min(auc_k, key=auc_k.get)
    e = FM.ece(y, proba)
    return diapositiva("4.5", "Probabilidades: ¿ordenan bien y son creíbles?", visual,
                       puntos=[f"AUC macro {T['AUC_OvR_macro'][B]:.2f}: el modelo ordena bien las viviendas.",
                               f"El estrato que mejor separa es el {mejor} (AUC {auc_k[mejor]:.2f}); el que peor, el "
                               f"{peor} ({auc_k[peor]:.2f}).",
                               f"Error de calibración (ECE) de {e:.2f}: en promedio, la probabilidad que da el modelo se "
                               f"aparta {100 * e:.0f} puntos de su tasa real de acierto."],
                       conclusion="Las probabilidades sirven para medir incertidumbre, no solo la clase final.",
                       metricas=[("auc", T["AUC_OvR_macro"][B]), ("ece", e, f"{e:.2f}")], fuente=f"{E1C} · cap. 14.4")


@callback(Output("roc-fig", "figure"), Input("roc-mod", "value"), Input("roc-vista", "value"))
def _roc(m, v):
    return F.fig_calibracion(m) if v == "cal" else F.fig_roc(m)


def d_coeficientes():
    visual = html.Div([controles(control("Vista", radio("coef-vista", [("odds", "Odds 6 frente a 1"), ("B", "Coeficientes B"),
                                                                       ("A", "Coeficientes A")], "odds"))),
                       grafico(FP.fig_odds(), "coef-fig")])
    od = json.loads((F.BASE / "odds.json").read_text(encoding="utf-8"))
    t = pd.DataFrame(od).sort_values("beta_6_menos_1")
    minus = lambda x: x[0].lower() + x[1:]  # noqa: E731
    sube = [minus(F._nombre_variable(v)) for v in t.variable[::-1][:3]]
    baja = [minus(F._nombre_variable(v)) for v in t.variable[:2]]
    mx = t.iloc[-1]
    return diapositiva("4.6", "Qué aprendió el modelo", visual,
                       puntos=["Cada barra: por cuánto se multiplican las odds del estrato 6 frente al 1 al subir la "
                               "variable una desviación estándar.",
                               "Hacia el estrato 6 empujan, sobre todo: " + ", ".join(sube) + ".",
                               "Hacia el estrato 1: " + " y ".join(baja) + ".",
                               "Cada coeficiente es un efecto parcial (las demás variables fijas): las variables "
                               "correlacionadas se reparten el efecto."],
                       conclusion="Los coeficientes cuentan la misma historia que el EDA: ubicación y tamaño.",
                       metricas=[("odds_ratio", float(mx.odds), f"×{float(mx.odds):.3g} ({minus(F._nombre_variable(mx.variable))})")],
                       fuente=f"{E1C} · cap. 10.6 (con el modelo v3)")


@callback(Output("coef-fig", "figure"), Input("coef-vista", "value"))
def _coef(v):
    return FP.fig_odds() if v == "odds" else F.fig_coeficientes("B_fisicas+ubicacion" if v == "B" else "A_fisicas")


def d_permutacion():
    im = json.loads((F.BASE / "importancia.json").read_text(encoding="utf-8"))
    t = pd.DataFrame(im).sort_values("caida", ascending=False)
    minus = lambda x: x[0].lower() + x[1:]  # noqa: E731
    top = [minus(FP.et(v)) for v in t.variable[:3]]
    neg = [minus(FP.et(v)) for v in t.variable[t.caida < 0]]
    return diapositiva("4.7", "Importancia por permutación", grafico(FP.fig_importancia_permutacion()),
                       puntos=["Se calcula en una zona de validación (pliegue 1 de kNNDM), no en el test.",
                               "Barajar estas variables es lo que más daño hace: " + ", ".join(top) + ".",
                               ("Barajar " + " o ".join(neg) + " no empeora el F1 en esa zona: su información la cubren "
                                "otras variables." if neg else "Todas las variables aportan algo en esa zona.")],
                       conclusion="La ubicación y el tamaño de la vivienda son lo que el modelo más usa.",
                       metricas=["importancia_perm"], fuente=f"{E1C} · cap. 10.6 (con el modelo v3)")


def d_aprendizaje():
    ca = D["curva_aprendizaje"]
    return diapositiva("4.8", "¿Más datos ayudarían?", grafico(F.fig_curva_aprendizaje()),
                       puntos=[f"La validación se queda entre {min(ca['val_media']):.2f} y {max(ca['val_media']):.2f} "
                               f"desde {fmt(ca['n'][0])} hasta {fmt(ca['n'][-1])} viviendas.",
                               "El entrenamiento no sube mucho: no es sobreajuste clásico.",
                               "La brecha es la distancia entre barrios conocidos y nuevos."],
                       conclusion="Más filas del mismo tipo no ayudan: falta otra información.",
                       fuente=f"{E1C} · cap. 10.3 (con el modelo v3)")


def d_residuos():
    m = D["moran_residuos"]["B_fisicas+ubicacion"]
    return diapositiva("4.10", "¿Queda estructura espacial en los errores?", grafico(F.fig_mapa_residuos()),
                       puntos=["Rojo: el modelo predice un estrato más alto que el real; azul, más bajo.",
                               "Los errores vienen en manchas: barrios enteros mal estimados.",
                               "Al modelo le falta información de barrio."],
                       conclusion="La siguiente mejora tiene que venir del entorno, no de la vivienda.",
                       metricas=[("moran", m["I"], f"{m['I']:.2f} (residuos; ideal 0)")],
                       fuente=f"{E1C} · cap. 10.5 (con el modelo v3)")


RANGOS = F.rangos_simulador()
ETQ_TV = {"tv_1": "VIS", "tv_2": "VIP", "tv_3": "No VIS", "tv_4": "No aplica"}
NUM_SIM = ["area_construida", "area_catastral_terreno", "total_habitaciones", "total_banios", "total_plantas",
           "planta_ubicacion", "antiguedad"]
CAT_SIM = ["condicion_predio", "uso", "tipo_vivienda"]


def d_simulador():
    nums = [dbc.Col([html.Label(F.NUMERICAS[v], className="control-etiqueta"),
                     dbc.Input(id=f"sim-{v}", type="number", value=round(RANGOS[v][1], 1), min=0,
                               step=1 if RANGOS[v][2] < 50 else 5, size="sm")], xs=6, md=3) for v in NUM_SIM]
    cats = [dbc.Col([html.Label(F.CATEGORICAS[c], className="control-etiqueta"),
                     dcc.Dropdown(id=f"sim-{c}", value=RANGOS["moda_categorias"][c], clearable=False,
                                  options=[{"label": ETQ_TV.get(o, o.replace("_", " ")), "value": o}
                                           for o in RANGOS["categorias"][c]])], md=4) for c in CAT_SIM]
    visual = html.Div([
        dbc.Row(nums, className="g-2"), dbc.Row(cats, className="g-2 mt-1"),
        html.Div("Haz clic en el mapa para mover la vivienda", className="control-etiqueta mt-3"),
        dbc.Row([dbc.Col(dcc.Graph(id="sim-mapa", config={"displaylogo": False}), md=6),
                 dbc.Col(dcc.Graph(id="sim-fig", config={"displaylogo": False}), md=6)], className="g-2"),
        dcc.Store(id="sim-punto", data={"lat": 10.97, "lon": -74.81})])
    return diapositiva("4.11", "Simulador: ¿qué estrato predice el modelo?", visual,
                       puntos=["Cambia área, baños o régimen y mira cómo se mueven las probabilidades.",
                               "Mueve la vivienda de sur a norte: el cambio es mucho mayor.",
                               "Valores iniciales: la vivienda típica de entrenamiento."],
                       conclusion="Es una herramienta para entender el modelo, no para asignar estratos.",
                       fuente=f"{E2C} · cap. 3")


@callback(Output("sim-punto", "data"), Input("sim-mapa", "clickData"), prevent_initial_call=True)
def _sim_click(click):
    p = click["points"][0]
    return {"lat": p["lat"], "lon": p["lon"]}


@callback(Output("sim-fig", "figure"), Output("sim-mapa", "figure"),
          [Input(f"sim-{v}", "value") for v in NUM_SIM + CAT_SIM] + [Input("sim-punto", "data")])
def _simular(*valores):
    entrada = dict(zip(NUM_SIM + CAT_SIM, valores[:-1]))
    for v in NUM_SIM:
        entrada[v] = np.nan if entrada[v] is None else float(entrada[v])
    entrada["altura"] = 3.0
    p = valores[-1]
    probas = F.predecir(dict(entrada, lat=p["lat"], lon=p["lon"]))
    mapa = F.fig_mapa("estrato_num", "todas")
    mapa.add_scattermap(lat=[p["lat"]], lon=[p["lon"]], mode="markers", marker=dict(size=20, color="#e34948"),
                        hovertemplate="vivienda simulada<extra></extra>", showlegend=False)
    mapa.update_layout(height=420, title="Ubicación (punto rojo)", showlegend=False, map=dict(zoom=10.8))
    return F.fig_simulador(probas), mapa


# ===========================================================================
# 5. MODELOS PROPUESTOS Y PROTOCOLO
# ===========================================================================
def d_candidatos():
    t = pd.DataFrame([
        ["XGBoost geográfico", "Un XGBoost global más XGBoost locales por zona, con pesos por distancia",
         "El estrato cambia de un barrio a otro", "Con y sin vecindad; zonas nuevas y conocidas; búsqueda con todos los datos"],
        ["Rotation Forest", "Cada árbol ve las variables rotadas con un PCA por grupos",
         "Variables de tamaño correlacionadas", "Con vecindad y todos los datos (implementación propia)"],
        ["Stacking", "Una logística combina las probabilidades de varios modelos",
         "Modelos que aciertan estratos distintos", "S1: logística, XGBoost y RF; S2: S1 + XGBoost geográfico"],
        ["RF + SVM", "Un Random Forest elige las variables y una SVM clasifica",
         "Variables redundantes", "SVM (Nyström) con las 10 o 20 variables más importantes"],
        ["TPE-SVM", "SVM de núcleo RBF con búsqueda bayesiana (TPE)", "Fronteras no lineales",
         "Solo sin vecindad, con el protocolo anterior (Entregable 2, cap. 6)"],
    ], columns=["Modelo original", "Idea", "Problema que ataca", "Cómo se probó"])
    ref = pd.DataFrame([["Logística con vecindad", "El modelo base con la misma información que reciben los originales"],
                        ["XGBoost y Random Forest", "Modelos conocidos con la misma información: ¿la mejora es del "
                                                    "algoritmo o de la información?"]],
                       columns=["Referencias (no originales)", "Para qué"])
    visual = html.Div([tabla(t), html.Div(tabla(ref), className="mt-3")])
    return diapositiva("5.1", "Los modelos propuestos: los originales de la revisión", visual,
                       puntos=["«Original»: la revisión no halló ninguna aplicación del modelo al estrato con datos "
                               "catastrales (Tabla III, 1.3).",
                               "Se probaron 5 de los 9 candidatos; los otros 4 son optimizadores o selectores de variables "
                               "(GWO, EFSHB, NSGA-II) o no tienen implementación abierta (Bagged TAO).",
                               "Todos reciben la misma información: las variables del modelo base más la vecindad física "
                               "(los 8 edificios más cercanos, sin usar su estrato).",
                               "Mismo protocolo para todos: 5 pliegues kNNDM y el test de 7 bloques, mirado una sola vez."],
                       conclusion="La pregunta: ¿algún modelo no usado antes supera al modelo base?",
                       fuente=f"{E2C} · caps. 4 y 9.1")


def d_geoxgb_como():
    pasos = [("1", "Global", "XGBoost con toda la ciudad"), ("2", "Anclas", "Centros de celdas de 2 km"),
             ("3", "Ventana", "Una fracción de las viviendas más cercanas"), ("4", "Locales", "Un XGBoost por ancla, "
                                                                               "con pesos por distancia"),
             ("5", "Mezcla", "α·global + (1 − α)·local")]
    visual = html.Div([flujo(pasos), html.Div(
        "p = α · p_global + (1 − α) · p_local", className="respuesta-grande mt-3", style={"fontSize": "1.6rem"}),
        html.P("Con α = 1 es un XGBoost normal: por eso el XGBoost con las mismas variables es su control natural (la "
               "diferencia entre los dos mide cuánto aporta la parte geográfica).", className="mt-2")])
    return diapositiva("5.2", "Cómo funciona el XGBoost geográfico (adaptación propia)", visual,
                       puntos=["El original (Grekousis, 2025) es de regresión; la adaptación a clasificación multiclase "
                               "es nuestra.",
                               "Kernel bicuadrado adaptativo: en zonas densas la ventana es pequeña; en zonas dispersas, "
                               "grande.",
                               "Hiperparámetros propios: la fracción de la ventana y el peso del global α."],
                       conclusion="Hipótesis: los modelos locales aprenden lo que cambia de un barrio a otro.",
                       metricas=["alpha_geo"], fuente=f"{E2C} · cap. 4")


def d_otros_originales():
    tarjetas = [
        ("Rotation Forest", ["Las variables se reparten al azar en grupos", "Cada grupo se rota con un PCA",
                             "Un árbol por rotación: cortes oblicuos"]),
        ("Stacking", ["Logística con vecindad, XGBoost y Random Forest", "Sus probabilidades fuera de muestra",
                      "Una logística las combina (metamodelo)"]),
        ("RF + SVM en dos etapas", ["Un Random Forest ordena las variables", "Se quedan las 10 o 20 más importantes",
                                    "Una SVM (Nyström) clasifica con ellas"]),
        ("TPE-SVM", ["SVM con núcleo RBF aproximado (Nyström)", "Búsqueda bayesiana de C y γ con TPE",
                     "Probada solo sin vecindad"]),
    ]
    visual = dbc.Row([dbc.Col(html.Div([html.Div(t, className="paso-titulo")] +
                                       [html.Div(f"{i + 1}. {x}", className="paso-desc") for i, x in enumerate(pasos)],
                                       className="paso", style={"height": "100%"}), md=6, className="mb-3")
                      for t, pasos in tarjetas], className="g-3")
    return diapositiva("5.3", "Cómo funcionan los demás modelos originales", visual,
                       puntos=["Rotation Forest ataca la colinealidad de las variables de tamaño (área, baños, "
                               "habitaciones).",
                               "El stacking aprovecha que los modelos aciertan estratos distintos (en el test del E3, la "
                               "logística con vecindad reconoce mejor los estratos 1 y 6 que los árboles).",
                               "RF + SVM separa la selección de variables de la clasificación."],
                       conclusion="Cuatro formas distintas de mejorar a un clasificador convencional.",
                       fuente=f"{E2C} · caps. 4.3 y 9.3")


OPC_TODOS = [(m, F.NOMBRE_CORTO[m]) for m in [B] + F.AVANZADOS]


# ===========================================================================
# VERSIÓN 3: validación por vecinos más cercanos (kNNDM) y vecindad con grafo KNN
# ===========================================================================
def _sin_v3(numero, titulo):
    return diapositiva(numero, titulo, html.P("Los resultados de la versión 3 aún no se han calculado."))


def d_v3_validacion():
    d = FP.v3_e1()
    if not d:
        return _sin_v3("4.12", "Validar a la distancia a la que se predice")
    q = d["distancias"]["cuantiles"]
    esc = q["Escenario: zona nueva de 2 km (test → todo train)"]["mediana (km)"]
    tv2 = q["Test de la versión 2 (test → train tras buffer)"]["mediana (km)"]
    cv2 = q["CV de la versión 2 (bloques + buffer de 1 km)"]
    kn = d["knndm"]
    visual = html.Div([controles(control("Vista", radio("v3v-vista", [("dist", "Distancias"), ("q", "Elegir q")], "dist"))),
                       grafico(FP.fig_v3_distancias(), "v3v-fig")])
    return diapositiva("4.12", "Validar a la distancia a la que se predice", visual,
                       puntos=[f"Una zona nueva de 2 km tiene su vecino conocido a {esc:.2f} km (mediana).",
                               f"Con el buffer de 1 km, la versión anterior validaba a {cv2['mediana (km)']:.1f} km y probaba a {tv2:.1f} km: "
                               "un escenario más difícil que el declarado.",
                               f"kNNDM arma los folds para imitar el escenario: {kn['q']} grupos de K-Means, "
                               f"Wasserstein {kn['W_km']:.2f} km (la validación anterior: {cv2['Wasserstein al escenario (km)']:.2f} km).",
                               "Solo usa las coordenadas del test, nunca su estrato."],
                       conclusion="El tamaño de la zona excluida ya no se fija a mano: sale de la geometría del problema.",
                       metricas=[("dist_vecino", esc, f"{esc:.2f} km (escenario)"),
                                 ("wasserstein", kn["W_km"], f"{kn['W_km']:.2f} km (kNNDM)"), "knndm"],
                       fuente=f"{E1C} v3 · cap. 14.1–14.2")


@callback(Output("v3v-fig", "figure"), Input("v3v-vista", "value"))
def _v3v(v):
    return FP.fig_v3_knndm_q() if v == "q" else FP.fig_v3_distancias()


def d_v3_mejoras():
    d = FP.v3_e1()
    if not d:
        return _sin_v3("4.10", "Variantes del modelo base")
    r = d["resumen"]
    b3, o3, bu = r["B v3 (kNNDM, sin buffer)"], r["Ordinal v3 (kNNDM, sin buffer)"], r["B v3 + umbrales F1"]

    def veredicto(m):
        return "supera" if m["Δ CV vs B v3"] > m["SE Δ"] else ("empata con" if m["Δ CV vs B v3"] > -m["SE Δ"] else "queda por debajo de")

    visual = html.Div([controles(control("Métrica", radio("v3m-met", [("f1", "F1 (CV y test)"), ("mae", "MAE ordinal"),
                                                                      ("kappa", "Kappa"), ("e4", "Recall E4")], "f1"))),
                       grafico(FP.fig_v3_modelos(incluir_v2=False), "v3m-fig")])
    return diapositiva("4.10", "Variantes del modelo base: logística ordinal y umbrales", visual,
                       puntos=["El estrato tiene orden: una logística ordinal (un solo vector de coeficientes y cinco "
                               "umbrales) lo respeta por construcción.",
                               f"La ordinal {veredicto(o3)} la multinomial en validación y en test se equivoca por más "
                               f"(MAE {o3['MAE test']:.3f} frente a {b3['MAE test']:.3f}).",
                               "Umbrales para el F1 macro: un sesgo por estrato, aprendido en la validación, antes de "
                               "elegir la clase.",
                               f"Con umbrales, el modelo {veredicto(bu)} la multinomial en validación; en test, el recall "
                               f"del estrato 4 pasa de {b3['recall E4']:.2f} a {bu['recall E4']:.2f}."],
                       conclusion="Ninguna variante supera a la multinomial en validación: el modelo base se queda como está.",
                       metricas=["ordinal", "sesgos_f1"], fuente=f"{E1C} · cap. 14.5–14.7")


@callback(Output("v3m-fig", "figure"), Input("v3m-met", "value"))
def _v3m(m):
    return FP.fig_v3_modelos(m, incluir_v2=False)


def d_vecindad_knn():
    vk = FP.vecindad_knn()
    if not vk:
        return _sin_v3("5.12", "Vecindad con grafo KNN")
    zn, zc = vk["cv"]["zonas nuevas"], vk["cv"]["zonas conocidas"]
    kx, ky = zn["kx_elegido"], zn["ky_elegido"]
    n_x = f"Logística B + vecindad KNN (k = {kx})"
    n_xy = f"Logística B + vecindad KNN (k = {kx}) + estrato de vecinos (k = {ky})"
    t = vk["test_zonas_nuevas"]
    kxc, kyc = zc["kx_elegido"], zc["ky_elegido"]
    n_xyc = f"Logística B + vecindad KNN (k = {kxc}) + estrato de vecinos (k = {kyc})"

    def dif(m):
        return f"{m['dif_F1_vs_B']:+.3f} (SE {m['SE_dif']:.3f})"

    radios = zn["modelos"]["Logística B + vecindad por radios (300 m y 1 km, cap. 5)"]

    visual = html.Div([controles(control("Escenario", radio("vk-esc", [("zonas nuevas", "Zonas nuevas"),
                                                                      ("zonas conocidas", "Zonas conocidas"),
                                                                      ("moran", "Elegir k")], "zonas nuevas"))),
                       grafico(FP.fig_vecindad_knn("zonas nuevas"), "vk-fig")])
    return diapositiva("5.12", "Vecindad con grafo KNN (lag espacial)", visual,
                       puntos=[f"Zonas nuevas, CV kNNDM: vecindad física KNN {dif(zn['modelos'][n_x])}; "
                               f"con estrato de vecinos {dif(zn['modelos'][n_xy])}.",
                               f"En test: F1 {t['Logística B']['F1_macro']:.3f} → {t[n_xy]['F1_macro']:.3f} "
                               f"con vecindad y estrato de vecinos.",
                               f"Zonas conocidas (manzanas ocultas): {dif(zc['modelos'][n_xyc])}.",
                               f"Control: la vecindad por radios del cap. 5 da lo mismo ({dif(radios)}). La ganancia viene de "
                               "validar en el escenario correcto, no del grafo.",
                               "El estrato de los vecinos se calcula sin el propio grupo: misma distancia que al predecir."],
                       conclusion="Con la validación correcta, describir el entorno sí ayuda: en zonas nuevas y, mucho más, "
                                  "en zonas conocidas.",
                       metricas=["lag_espacial", "se_pareado"], fuente=f"{E2C} · caps. 7.3–7.4")


@callback(Output("vk-fig", "figure"), Input("vk-esc", "value"))
def _vk(e):
    return FP.fig_moran_k() if e == "moran" else FP.fig_vecindad_knn(e)

# ===========================================================================
# Historia de la investigación: hipótesis y respuesta
# ===========================================================================
def d_hipotesis():
    visual = html.Div([
        html.Div("Objetivo", className="pregunta-grande"),
        html.P("Predecir el estrato (1 a 6) de cada vivienda con el catastro abierto, en zonas que el modelo no vio.",
               style={"fontSize": "1.15rem"}),
        dbc.Row([
            dbc.Col(html.Div([html.H5("H1 · Entregables 2 y 3"),
                              html.P("Los modelos de la revisión superan al modelo base."),
                              html.P("Resultado: sí, con la vecindad física.", className="nota-visual")],
                             className="caja-si"), md=6),
            dbc.Col(html.Div([html.H5("H2 · Entregable 4"),
                              html.P("El EPL supera al modelo base y a los mejores benchmarks."),
                              html.P("Resultado: al base sí (4 de 4 pruebas); a CatBoost, solo 1 de 3.",
                                     className="nota-visual")], className="caja-no"), md=6)], className="g-3 mt-2")])
    return diapositiva("1.4", "Objetivo e hipótesis", visual,
                       puntos=["H1: los modelos de la revisión frente al modelo base (E2 y E3).",
                               "H2: nuestro modelo original, el EPL, frente al base y a los mejores (E4)."],
                       conclusion="H1 se cumplió; H2 solo frente al modelo base.",
                       fuente=f"{E2C} · cap. 1.4 · {DE4.E4C} · cap. 0")


FORMULACION_PROBLEMA = r"""
**Datos.** $\mathcal{D} = \{(x_i, s_i, y_i)\}_{i=1}^{n}$: $x_i$ variables del catastro y de vecindad,
$s_i \in \mathbb{R}^2$ ubicación en km y $y_i \in \{1, \dots, 6\}$ el estrato (**ordinal**).

**Tarea.** Clasificación supervisada multiclase: aprender $f: (x, s) \mapsto \hat p(\cdot \mid x, s) \in \Delta_6$ y
predecir $\hat y = \arg\max_k \hat p(k \mid x, s)$.

**Objetivo.** El menor error esperado en **zonas nuevas** ($\ell$: log-pérdida ponderada por clase):

$$f^\star = \arg\min_{f \in \mathcal{F}} \; \mathbb{E}_{(x, s, y) \sim \mathcal{P}_{\text{zona nueva}}} \big[\ell\big(y, f(x, s)\big)\big]$$

**Métrica.** $\displaystyle F_1^{\text{macro}} = \frac{1}{6} \sum_{k=1}^{6} \frac{2\, P_k R_k}{P_k + R_k}$: los 6 estratos pesan igual.

**Validación.** Por zonas: test de 7 bloques de 2 km y kNNDM en el entrenamiento, porque las viviendas vecinas se
parecen.
"""


def d_formulacion():
    return diapositiva("1.5", "Formulación matemática del problema", DE4._md(FORMULACION_PROBLEMA),
                       puntos=["Clasificación supervisada multiclase y ordinal: una probabilidad por estrato.",
                               "Objetivo: el menor error en zonas nuevas.",
                               "Métrica: F1 macro, los 6 estratos pesan igual.",
                               "Validación por zonas: las viviendas vecinas se parecen (I de Moran 0.92)."],
                       conclusion="Clasificación ordinal de 6 clases, evaluada en zonas que el modelo no vio.",
                       fuente=f"{E1C} · caps. 1 y 6 · {DE4.E4C} · cap. 2", ancho_visual=9)


def d_nuevo_vs_base():
    a = FP.arboles_knndm()
    if not a:
        return diapositiva("6.12", "¿Supera el modelo nuevo al modelo base?", html.P("Resultados pendientes."))
    zn = a["xgb_geografico_vs_logistica"]["zonas nuevas"]["variantes"]
    zc = a["xgb_geografico_vs_logistica"]["zonas conocidas"]["variantes"]
    knn = zn["XGBoost geográfico | B + vecindad KNN (k = 8)"]
    b = knn["vs Logística B v3 (sin vecindad)"]
    t = b["test_F1"]
    zc_b = [v["vs Logística B v3 (sin vecindad)"]["cv_F1"]["media"] for v in zc.values()]
    visual = html.Div([controles(
        control("Escenario", radio("nvb-esc", [("zonas nuevas", "Zonas nuevas"), ("zonas conocidas", "Zonas conocidas")],
                                   "zonas nuevas")),
        control("Comparar con", radio("nvb-ref", [("base", "Modelo base"), ("misma", "Logística con la misma información")],
                                      "base"))),
        grafico(FP.fig_nuevo_vs_base("zonas nuevas", "base"), "nvb-fig")])
    return diapositiva("6.4", "XGBoost geográfico: zonas nuevas y zonas conocidas", visual,
                       puntos=[f"Zonas nuevas con vecindad: F1 de test {knn['F1_test']:.3f} frente a "
                               f"{knn['F1_test'] - t['diferencia']:.3f} del base (+{t['diferencia']:.3f}, "
                               f"IC 95 % [{t['IC95_inf']:+.3f}, {t['IC95_sup']:+.3f}]).",
                               f"En validación lo supera en {b['cv_F1']['folds_a_favor']} de 5 pliegues "
                               f"(+{b['cv_F1']['media']:.3f}, SE {b['cv_F1']['SE']:.3f}).",
                               f"Zonas conocidas: +{min(zc_b):.2f} a +{max(zc_b):.2f} de F1 en validación.",
                               "Con solo las variables del base, el test no lo confirma (−0.009).",
                               "Ojo: con el intervalo BCa del E3 el mismo +0.09 da [−0.024, +0.172]; con 7 bloques la "
                               "confirmación depende del método. La prueba con 30 bloques sí es significativa."],
                       conclusion="Sí: con información del entorno, el modelo nuevo supera al base en los dos escenarios.",
                       metricas=[("f1_macro", knn["F1_test"], f"{knn['F1_test']:.3f} (modelo nuevo, test)"), "se_pareado", "ic95"],
                       fuente=f"{E2C} · cap. 8")


@callback(Output("nvb-fig", "figure"), Input("nvb-esc", "value"), Input("nvb-ref", "value"))
def _nvb(esc, ref):
    return FP.fig_nuevo_vs_base(esc, ref)


# --- Todos los modelos originales de la revisión (Entregable 2, cap. 9; código en mejoras/ del Entregable 3) -----
def d_originales():
    m = FM.mejoras()
    te, cv = m["test"], m["cv"]
    rof, s1 = te["Rotation Forest"], te["Stacking S1 (logística, XGBoost, RF)"]
    ic = lambda v: f"[{v[0]:+.3f}, {v[1]:+.3f}]"  # noqa: E731
    visual = html.Div([controles(
        control("Mirada", radio("orig-mir", [("test", "Test"), ("cv", "Validación"), ("tabla", "Tabla")], "test")),
        control("Comparar con", radio("orig-ref", [("base", "Modelo base"), ("lv", "Logística con la misma vecindad")],
                                      "base"))),
        html.Div(grafico(FM.fig_originales("test", "base"), minimo=400), id="orig-cont")])
    return diapositiva("6.13", "Todos los modelos originales de la revisión frente al modelo base", visual,
                       puntos=["Con la vecindad física y todos los datos (257 273 viviendas, mismos pliegues kNNDM y "
                               "mismo test) se probaron el XGBoost geográfico, Rotation Forest, stacking y RF + SVM.",
                               f"Rotation Forest: {rof['vs_base']['dif']:+.3f} en test, IC {ic(rof['vs_base']['ic95_bca'])}, "
                               f"{rof['vs_base']['bloques_a_favor']} de 7 bloques: el único original cuyo intervalo "
                               "excluye el 0.",
                               f"Stacking S1: la misma validación ({cv['Stacking S1 (logística, XGBoost, RF)']['f1']:.3f}), "
                               f"pero su intervalo de test toca el 0 {ic(s1['vs_base']['ic95_bca'])}.",
                               "Frente a la logística con la misma vecindad, ningún original se confirma en el test."],
                       conclusion="Varios modelos originales superan al base; ninguno a los convencionales que reciben "
                                  "la misma información.",
                       metricas=["ic95", "se_pareado"], fuente=f"{E2C} · cap. 9.3 · {E3C} · cap. 11")


@callback(Output("orig-cont", "children"), Input("orig-mir", "value"), Input("orig-ref", "value"))
def _orig(mir, ref):
    if mir == "tabla":
        return tabla(FM.tabla_originales())
    return grafico(FM.fig_originales(mir, ref), minimo=400)


def d_signif_originales():
    e = FM.estadistica()
    fr = e["friedman"]
    vb = e["comparaciones"]["vs modelo base"]
    vl = e["comparaciones"]["vs logística con vecindad"]
    con_vec = [r for r in vb if not r["a"].startswith(("RF + SVM", "SVM sola"))]
    rof = next(r for r in vb if r["a"] == "Rotation Forest")
    visual = html.Div([controles(control("Comparación", radio(
        "sig-fam", [("base", "Frente al modelo base"), ("lv", "Frente a la logística con vecindad"),
                    ("rangos", "Rangos (Friedman)")], "base"))),
        grafico(FM.fig_significancia("base"), "sig-fig", minimo=400)])
    return diapositiva("6.14", "¿Son estadísticamente significativas las diferencias?", visual,
                       puntos=[f"Unidad con más réplicas: {e['bloques']['n_usados']} bloques de 2 km del entrenamiento "
                               f"(predicciones fuera de muestra). Friedman: χ² = {fr['chi2']:.1f}, p = {cientifico(fr['p'])}.",
                               f"Frente al base: Wilcoxon con Holm p < 0.01 para todos los modelos con vecindad, con δ de "
                               f"Cliff de {min(r['delta_cliff'] for r in con_vec):.2f} a "
                               f"{max(r['delta_cliff'] for r in con_vec):.2f} (mediano a grande); RF + SVM no (p = 0.10).",
                               f"Frente a la logística con vecindad: nada es significativo (p de Holm ≥ "
                               f"{piso(min(r['wilcoxon_p_holm'] for r in vl)):.2f}).",
                               "Con 5 pliegues (t de Nadeau-Bengio) o 7 bloques de test (permutación exacta) nada resiste "
                               "la corrección de Holm: son muy pocas réplicas."],
                       conclusion="Lo significativo es la vecindad física: los modelos originales la aprovechan igual de "
                                  "bien, no mejor.",
                       metricas=[("cliff", rof["delta_cliff"], f"{rof['delta_cliff']:.2f} (Rotation Forest frente al base)"),
                                 "holm", "friedman"],
                       fuente=f"{E2C} · cap. 9.4 · {E3C} · cap. 11")


@callback(Output("sig-fig", "figure"), Input("sig-fam", "value"))
def _sig(f):
    return FM.fig_significancia(f)


def d_busqueda_geo():
    g = FM.geo_busqueda()
    m = FM.mejoras()
    gl, lo = g["globales"][g["mejor"]["global"]], g["locales"][g["mejor"]["local"]]
    dif_e2 = m["cv"]["XGBoost geográfico afinado"]["vs_geo_E2"]
    vs_x = m["cv"]["XGBoost geográfico afinado"]["vs_xgboost_afinado"]
    te = m["test"]
    visual = html.Div([controles(control("Vista", radio("bg-vista", [("alpha", "Curva de α"),
                                                                     ("comparar", "Validación y test")], "alpha"))),
                       grafico(FM.fig_busqueda_geo("alpha"), "bg-fig")])
    return diapositiva("6.15", "XGBoost geográfico: búsqueda de hiperparámetros con todos los datos", visual,
                       puntos=[f"{len(g['globales'])} configuraciones del modelo global × {len(g['locales'])} de los "
                               f"locales × {len(g['alfas'])} valores de α = {g['n_combinaciones']:,} combinaciones, con los "
                               "5 pliegues kNNDM.".replace(",", " "),
                               f"Eligió profundidad {gl['max_depth']} {'con' if gl['balanceado'] else 'sin'} pesos de clase, "
                               f"rejilla de {lo['tam']:.0f} km, {100 * lo['frac']:.0f} % de vecinos por ancla y "
                               f"α = {g['mejor']['alpha']}.",
                               f"Sube la validación ({g['referencia_E2']['f1']:.3f} → {g['mejor']['f1']:.3f}, "
                               f"{dif_e2['pliegues_a_favor']} de 5 pliegues) pero no el test "
                               f"({te['XGBoost geográfico (configuración del E2)']['f1']:.3f} → "
                               f"{te['XGBoost geográfico afinado']['f1']:.3f}).",
                               f"Frente al XGBoost con el mismo global (α = 1), la parte geográfica aporta "
                               f"{vs_x['media']:+.3f} en validación y "
                               f"{te['XGBoost geográfico afinado']['vs_xgboost_afinado']['dif']:+.3f} en test."],
                       conclusion="Afinar no cambia la conclusión: en zonas nuevas, la parte geográfica no se distingue "
                                  "del XGBoost.",
                       metricas=["alpha_geo", "knndm"], fuente=f"{E2C} · cap. 9.3 · {E3C} · cap. 11")


@callback(Output("bg-fig", "figure"), Input("bg-vista", "value"))
def _bg(v):
    return FM.fig_busqueda_geo(v)


# ===========================================================================
# 7. OPTIMIZACIÓN (ENTREGABLE 3)
# ===========================================================================
def d_e3_diseno():
    a = FM.e3("analisis_clasificacion")
    mp = {r["modelo"]: r["f1_macro_mean"] for r in a["mejor_por_modelo"]}
    pasos = [("8 modelos", "KNN, Naive Bayes, logística, árbol, RF, XGBoost, SVM y XGBoost geográfico"),
             ("× 4 balanceos", "ninguno, SMOTE, ADASYN y class_weight"),
             ("× 4 optimizadores", "Grid, Random, bayesiana (TPE) y genética"),
             ("= 156 combinaciones", f"128 + 28 de regresión; {a['n_tareas']} tareas con {fmt(a['submuestra'])} viviendas")]
    cajas = html.Div([html.Div([html.Div(t, className="paso-titulo"), html.Div(d, className="paso-desc")], className="paso")
                      for t, d in pasos], className="flujo")
    visual = html.Div([cajas, grafico(FM.fig_e3_modelos())])
    return diapositiva("7.1", "El pipeline combinatorio del Entregable 3", visual,
                       puntos=["Validación anidada 5 × 3 con kNNDM: el bucle interno elige los hiperparámetros y el "
                               "externo, que nunca los vio, mide.",
                               "Mismo presupuesto para todos: 12 evaluaciones por búsqueda, semilla 42; 38 h de cómputo "
                               "en 4.8 h de reloj.",
                               f"Entre modelos el F1 va de {mp['Naive Bayes']:.2f} (Naive Bayes) a "
                               f"{mp['Random Forest']:.2f} (Random Forest); dentro de un modelo, sus 16 combinaciones se "
                               "separan apenas 0.02 a 0.05.",
                               "Los tres ensambles de árboles forman el grupo de cabeza, sin diferencias demostrables."],
                       conclusion="El modelo pesa mucho más que el balanceo o el optimizador.",
                       metricas=["knndm", "f1_macro"], fuente=f"{E3C} · caps. 2 y 5")


def d_e3_optimizadores():
    a = FM.e3("analisis_clasificacion")
    fo, dg = a["friedman_optimizadores"], a["diversidad_genetica"]
    visual = html.Div([grafico(FM.fig_e3_optimizadores()), html.Div(tabla(FM.tabla_e3_optimizadores()), className="mt-2")])
    return diapositiva("7.2", "Optimizadores: el mismo resultado, distinta velocidad", visual,
                       puntos=["Grid, Random, bayesiana (Optuna, TPE) y genética (DEAP), con 12 evaluaciones cada uno y "
                               "el pipeline completo dentro de cada evaluación.",
                               f"F1 externo de 0.586 a 0.590: Friedman p = {fo['p']:.2f}, empatan.",
                               "La bayesiana y la genética llegan al 99 % en 5 evaluaciones; Grid necesita 10.",
                               f"Genético con población 4: la diversidad cae de {dg['0']['mean']:.2f} a "
                               f"{dg['3']['mean']:.2f} en 3 generaciones (convergencia prematura).",
                               "Successive Halving: el mismo F1 en XGBoost con ≈ ¼ del cómputo; en Random Forest pierde "
                               "0.019."],
                       conclusion="Con espacios bien acotados todos llegan a la meseta; la bayesiana es la más barata.",
                       metricas=["p_valor"], fuente=f"{E3C} · cap. 3")


def d_e3_balanceo():
    a = FM.e3("analisis_clasificacion")
    b, rk = a["balanceo"], a["friedman_balanceo"]["rangos"]
    visual = html.Div([controles(control("Medida", radio(
        "bal-met", [("f1", "F1 macro"), ("rango", "Rango medio"), ("ece", "Calibración (ECE)"),
                    ("tiempo", "Tiempo")], "f1"))), grafico(FM.fig_e3_balanceo("f1"), "bal-fig")])
    return diapositiva("7.3", "Balanceo de clases: class_weight gana, ADASYN pierde", visual,
                       puntos=[f"class_weight: el mejor rango medio ({rk['class_weight']:.2f} de 4) y el más barato "
                               f"({b['class_weight']['seg_busqueda_mean']:.0f} s por búsqueda frente a "
                               f"{b['smote']['seg_busqueda_mean']:.0f} s de SMOTE).",
                               f"SMOTE y no balancear quedan casi empatados; ADASYN es el peor (Friedman p = "
                               f"{cientifico(a['friedman_balanceo']['p'], 0)}).",
                               "ADASYN crea viviendas sintéticas junto a las «difíciles», que aquí son las de frontera "
                               "(EDA: el entorno): refuerza el ruido.",
                               "Todas las técnicas empeoran algo la calibración (ECE)."],
                       conclusion="Basta con pesar las clases; generar viviendas sintéticas no ayuda.",
                       metricas=["class_weight", "ece"], fuente=f"{E3C} · cap. 5")


@callback(Output("bal-fig", "figure"), Input("bal-met", "value"))
def _bal(m):
    return FM.fig_e3_balanceo(m)


def d_e3_computo():
    visual = html.Div([controles(control("Familia", desplegable(
        "cmp-fam", [(k, k) for k in FM.FAMILIAS_COMPLEJIDAD], "SVM", ancho=220))),
        grafico(FM.fig_e3_complejidad("SVM"), "cmp-fig")])
    return diapositiva("7.4", "Costo computacional: la teoría predice el tiempo", visual,
                       puntos=["Se midió cada versión con 1 000 a 32 000 viviendas y se ajustó log t = a + b·log n: "
                               "b ≈ 1 es lineal y b ≈ 2, cuadrático.",
                               "SVM de núcleo completo: b = 1.88; Nyström: 1.27, predice 80 veces más rápido y pierde "
                               "0.036 de F1.",
                               "XGBoost hist: 9 veces más rápido que exact con el mismo F1. KNN con FAISS HNSW: 28 veces "
                               "más rápido con 256 000 viviendas.",
                               "El 91 % del tiempo es construir árboles; paralelizar las 640 tareas: 38.4 h → 4.8 h."],
                       conclusion="No toda «optimización» conviene: con 35 variables densas, SAGA es 7 veces más lenta "
                                  "que lbfgs y KD-Tree, más lento que la búsqueda exhaustiva.",
                       fuente=f"{E3C} · cap. 4")


@callback(Output("cmp-fig", "figure"), Input("cmp-fam", "value"))
def _cmp(f):
    return FM.fig_e3_complejidad(f)


def d_e3_test():
    f = FM.e3("finalistas")
    s = f["semillas"]
    lv = f["finalistas"]["Logística L1/L2"]["test"]["f1_macro"]
    visual = html.Div([controles(control("Entrenados con", radio(
        "e3t-datos", [("submuestra", "Submuestra (10 001)"), ("todos", "Todos los datos (257 273)")], "submuestra"))),
        grafico(FM.fig_e3_test("submuestra"), "e3t-fig")])
    return diapositiva("7.5", "Test de los finalistas: todo baja y los mejores empatan", visual,
                       puntos=["Finalistas: la mejor combinación de cada modelo, evaluada una sola vez en los 7 bloques "
                               "de test.",
                               "Todo baja: los ensambles pasan de ≈ 0.66 en validación a 0.52–0.57 en test (el test "
                               "tiene 1.2 % de estrato 6, ver «EDA: el test»).",
                               f"Con 5 semillas: logística con vecindad {lv:.3f}, XGBoost {s['XGBoost']['media']:.3f} y "
                               f"Random Forest {s['Random Forest']['media']:.3f}: empatan.",
                               f"Todos menos Naive Bayes superan al modelo base ({f['modelo_base']['test']['f1_macro']:.3f} "
                               "con la submuestra).",
                               "Con todos los datos suben 0.006 a 0.025 los modelos cuyos hiperparámetros no dependen "
                               "del tamaño de la muestra."],
                       conclusion="Las conclusiones firmes son las que aparecen en validación y en test.",
                       metricas=["f1_macro", "ic95"], fuente=f"{E3C} · caps. 5 y 9")


@callback(Output("e3t-fig", "figure"), Input("e3t-datos", "value"))
def _e3t(d):
    return FM.fig_e3_test(d)


def d_e3_shap():
    visual = html.Div([controles(control("Modelo", radio(
        "shap-mod", [("XGBoost", "XGBoost"), ("XGBoost geográfico", "XGBoost geográfico"),
                     ("Logística L1/L2", "Logística con vecindad")], "XGBoost"))),
        grafico(FM.fig_e3_shap("XGBoost"), "shap-fig")])
    return diapositiva("7.6", "¿Qué decide el estrato? Interpretabilidad con SHAP", visual,
                       puntos=["XGBoost (TreeSHAP exacto): encabezan la posición norte-sur, el área y los baños de los "
                               "vecinos y la distancia al centro.",
                               "La primera variable propia de la vivienda (sus baños) aparece en el décimo lugar.",
                               "Las escalas no se comparan entre modelos (log-odds en XGBoost, probabilidad en los "
                               "demás): se muestra el % de la importancia total.",
                               "LIME coincide solo en parte con SHAP (3 a 5 de las 8 variables principales)."],
                       conclusion="Lo que decide el estrato es la ubicación y el entorno, no la vivienda misma.",
                       fuente=f"{E3C} · cap. 8")


@callback(Output("shap-fig", "figure"), Input("shap-mod", "value"))
def _shap(m):
    return FM.fig_e3_shap(m)


def d_e3_calibracion():
    f = FM.e3("finalistas")
    cal, fin = f["calibracion"], f["finalistas"]
    geo, lv = "XGBoost geográfico", "Logística L1/L2"
    visual = html.Div([controles(control("Medida", radio(
        "cal-med", [("ece", "Calibración (ECE)"), ("brier", "Brier"), ("f1", "F1 macro")], "ece"))),
        grafico(FM.fig_e3_calibracion("ece"), "cal-fig")])
    return diapositiva("7.6", "¿Se puede confiar en sus probabilidades? Calibración", visual,
                       puntos=[f"Bien calibrados de entrada: la logística con vecindad (ECE {cal[lv]['original']['ece']:.3f}), "
                               f"KNN ({cal['KNN']['original']['ece']:.3f}) y XGBoost ({cal['XGBoost']['original']['ece']:.3f}).",
                               f"Los peores: Naive Bayes ({cal['Naive Bayes']['original']['ece']:.2f}), el árbol "
                               f"({cal['Árbol de decisión']['original']['ece']:.2f}) y SVM ({cal['SVM']['original']['ece']:.2f}); "
                               "Platt o la regresión isotónica los corrigen.",
                               f"Donde ya estaba bien no ayuda: la logística se descalibra (ECE "
                               f"{cal[lv]['isotonica']['ece']:.2f}–{cal[lv]['platt']['ece']:.2f}).",
                               f"El recalibrado promedia 5 modelos y cambia decisiones: el XGBoost geográfico sube de "
                               f"{fin[geo]['test']['f1_macro']:.3f} a {cal[geo]['isotonica']['f1']:.3f} de F1, aún por debajo de "
                               f"la logística ({fin[lv]['test']['f1_macro']:.3f})."],
                       conclusion="Un AUC alto no garantiza probabilidades creíbles: se recalibra solo cuando hace falta.",
                       metricas=["ece", "brier"], fuente=f"{E3C} · cap. 5")


@callback(Output("cal-fig", "figure"), Input("cal-med", "value"))
def _cal(m):
    return FM.fig_e3_calibracion(m)


def d_e3_regresion():
    r = pd.DataFrame(FM.e3("resumen_regresion"))
    ar = FM.e3("analisis_regresion")
    mejor = r.sort_values("rmse_mean").groupby("modelo").head(1).set_index("modelo")
    rf, xg, sv = (mejor.loc[m] for m in ["Random Forest (regresión)", "XGBoost (regresión)", "SVR"])
    sup = ar["supuestos_mejor"]
    en_mcs = sum(1 for m in ar["mcs"] if m["en_el_conjunto"])
    visual = html.Div([controles(control("Medida", radio(
        "reg-met", [("rmse", "RMSE"), ("mae", "MAE"), ("f1", "F1 al redondear")], "rmse"))),
        grafico(FM.fig_e3_regresion("rmse"), "reg-fig")])
    return diapositiva("7.7", "Regresión: el estrato como número no ayuda a clasificar", visual,
                       puntos=["Mismos datos y pliegues: 7 modelos × 4 optimizadores = 28 combinaciones, sin balanceo.",
                               f"Gana Random Forest (RMSE {rf['rmse_mean']:.3f} estratos, R² {rf['r2_mean']:.2f}), seguido de "
                               f"XGBoost ({xg['rmse_mean']:.3f}) y SVR ({sv['rmse_mean']:.3f}); el MCS conserva {en_mcs} de las "
                               "28 combinaciones.",
                               "Al redondear, su F1 macro queda por debajo del mejor clasificador: la regresión comprime los "
                               "extremos (solo el 19 % del estrato 6 llega a 6).",
                               f"Residuos heterocedásticos (White), dependientes (BDS y Ljung-Box) y correlacionados en el "
                               f"espacio (I de Moran {sup['moran_residuos_k8']:.2f}): queda estructura espacial."],
                       conclusion="Sirve como lectura ordinal del error, no como clasificador.",
                       metricas=["mae_ordinal", "moran"], fuente=f"{E3C} · caps. 6 y 7")


@callback(Output("reg-fig", "figure"), Input("reg-met", "value"))
def _reg(m):
    return FM.fig_e3_regresion(m)


def d_e3_estadistica():
    a = FM.e3("analisis_clasificacion")
    f = a["friedman_mejor_por_modelo"]
    cliff = next(c for c in a["cliff_finalistas"] if c["A"].startswith("XGBoost geográfico") and c["B"].startswith("Logística"))
    visual = html.Div([controles(control("Vista", radio(
        "est-vista", [("rangos", "Rangos (Nemenyi)"), ("Logística L1/L2", "DeLong vs logística"),
                      ("XGBoost", "DeLong vs XGBoost"), ("Random Forest", "DeLong vs Random Forest")], "rangos"))),
        grafico(FM.fig_e3_rangos(), "est-fig")])
    return diapositiva("7.8", "¿Diferencias reales? Friedman, Nemenyi y DeLong", visual,
                       puntos=[f"Friedman con la mejor combinación de cada modelo en 32 bloques espaciales: difieren "
                               f"(p = {cientifico(f['p'])}).",
                               f"El XGBoost geográfico tiene el mejor rango medio ({f['rangos']['XGBoost geográfico']:.2f}), "
                               f"pero con la diferencia crítica de Nemenyi (CD = {f['cd']:.2f}) solo Naive Bayes queda "
                               "claramente atrás.",
                               "DeLong por estrato con Holm: frente a la logística con vecindad, el geográfico ordena mejor "
                               "los estratos 1 a 4 y peor el 5 y el 6.",
                               f"Tamaño del efecto: δ de Cliff = {cliff['delta_cliff_bloques']:.2f} frente a la logística "
                               f"({cliff['magnitud']}); despreciable entre ensambles."],
                       conclusion="Entre los mejores, las diferencias son pequeñas y cambian de signo según el estrato.",
                       metricas=["friedman", "holm", "cliff"], fuente=f"{E3C} · cap. 7")


@callback(Output("est-fig", "figure"), Input("est-vista", "value"))
def _est(v):
    return FM.fig_e3_rangos() if v == "rangos" else FM.fig_e3_delong(v)


def d_e3_nuevo():
    f, t = FM.e3("finalistas"), FM.e3("todos_los_datos")
    g = {c["rival"]: c for c in f["geo_vs_rivales"]}
    gt = {c["rival"]: c for c in t["geo_vs_rivales"]}
    base = "Logística B (modelo base del E1)"
    b, bt = g[base], gt[base]
    alpha = f["finalistas"]["XGBoost geográfico"]["params"]["alpha"]
    visual = html.Div([controles(control("Frente a", radio(
        "nuevo-rival", [("base", "Modelo base"), ("logistica", "Logística con vecindad"), ("xgboost", "XGBoost"),
                        ("rf", "Random Forest")], "base"))),
        grafico(FM.fig_e3_nuevo("base"), "nuevo-fig")])
    return diapositiva("7.10", "El modelo nuevo frente al modelo base en el Entregable 3", visual,
                       puntos=["La propuesta es el XGBoost geográfico adaptado a clasificación; los 7 modelos de la guía "
                               "son referencias. XGBoost es su parte global sin modelos locales: compararlos es una ablación.",
                               f"Frente al modelo base: {b['cv_dif_media']:+.3f} en validación ({b['cv_pliegues_a_favor']}/5) y "
                               f"{b['test_dif']:+.3f} en test (IC [{b['test_ic95_bca'][0]:+.3f}, {b['test_ic95_bca'][1]:+.3f}], "
                               f"{b['test_bloques_a_favor']}/{b['test_n_bloques']} bloques).",
                               f"Con todos los datos y los hiperparámetros de la submuestra (α = {alpha:.2f}) ya no se "
                               f"distingue del base ({bt['test_dif']:+.3f}).",
                               f"Con la misma vecindad no gana en zonas nuevas: con todos los datos pierde frente a XGBoost "
                               f"({gt['XGBoost']['test_dif']:+.3f}) y Random Forest ({gt['Random Forest']['test_dif']:+.3f})."],
                       conclusion="Supera al modelo base, pero la mejora la explica la vecindad física; su ventaja propia "
                                  "está en las zonas con datos cerca (Entregable 2).",
                       metricas=["f1_macro", "ic95", "alpha_geo"], fuente=f"{E3C} · cap. 9")


@callback(Output("nuevo-fig", "figure"), Input("nuevo-rival", "value"))
def _nuevo(r):
    return FM.fig_e3_nuevo(r)

# ===========================================================================
# 6. CONCLUSIONES
# ===========================================================================
def hallazgo(valor, titulo, texto, color="#2a78d6"):
    return html.Div([html.Div(valor, className="hallazgo-valor", style={"color": color}), html.Div(titulo, className="hallazgo-titulo"),
                     html.P(texto, className="hallazgo-texto")], className="hallazgo", style={"borderTopColor": color})


def d_hallazgos():
    ar = FP.arboles_knndm()
    nuevo = []
    if ar:
        k = ar["xgb_geografico_vs_logistica"]["zonas nuevas"]["variantes"]["XGBoost geográfico | B + vecindad KNN (k = 8)"]
        tt = k["vs Logística B v3 (sin vecindad)"]["test_F1"]
        nuevo = [hallazgo(f"{k['F1_test'] - tt['diferencia']:.2f} → {k['F1_test']:.2f}",
                          "F1 de test: modelo base → XGBoost geográfico con vecindad",
                          "El modelo nuevo supera al base en zonas nuevas; con 30 bloques, la mejora es significativa "
                          "(Holm p < 0.01).", "#008300")]
    if F4.disponible():
        b4 = F4.leer("base_vs_epl.json")["epl_vs_base"]
        x4 = F4.rival("CatBoost")
        nuevo = [hallazgo(f"{b4['media_rival']:.2f} → {b4['media_rival'] + b4['dif_media']:.2f}",
                          "F1 en la validación 5 × 5: modelo base → modelo original (EPL)",
                          f"Gana en {b4['pliegues_a_favor']} de 25 pliegues y en el test (0.43 → 0.58); las cuatro "
                          "pruebas lo confirman.", "#eb6834"),
                 hallazgo("1 de 3", f"pruebas que rechazan frente a CatBoost (+{x4['dif_media']:.3f} de F1)",
                          "El EPL tiene la mayor media, pero su superioridad sobre los mejores no queda demostrada (R3).",
                          "#52514e")] + nuevo
    es, eda = FM.estadistica(), FM.eda()
    if es and eda:
        vl = es["comparaciones"]["vs logística con vecindad"]
        ban = eda["eta2"]["baños"]
        nuevo += [
            hallazgo("3 de 4", "modelos originales con vecindad que superan al base con significancia",
                     "XGBoost geográfico, Rotation Forest y stacking (efecto mediano a grande); RF + SVM no.", "#eb6834"),
            hallazgo("0", "originales mejores que los convencionales con la misma información",
                     f"Frente a la logística con vecindad, p de Holm ≥ {piso(min(r['wilcoxon_p_holm'] for r in vl)):.2f}: "
                     "lo que mejora es describir el entorno.", "#52514e"),
            hallazgo(f"{ban['vecinos']:.2f} vs {ban['propia']:.2f}", "η² de los baños: los 8 vecinos frente a la vivienda",
                     "El entorno dice más del estrato que la vivienda misma (EDA mejorado).", "#0d366b"),
            hallazgo("p = 0.74", "Cuatro optimizadores, el mismo F1 (E3)",
                     "Grid, Random, bayesiana y genética empatan; la bayesiana llega antes. El modelo pesa más.", "#4a3aa7")]
    mj = FM.mejoras()
    if mj:
        rof = mj["test"]["Rotation Forest"]["vs_base"]
        nuevo.insert(1, hallazgo(f"{rof['dif']:+.2f}", "F1 de test: Rotation Forest frente al modelo base",
                                 f"IC 95 % [{rof['ic95_bca'][0]:+.3f}, {rof['ic95_bca'][1]:+.3f}], "
                                 f"{rof['bloques_a_favor']} de 7 bloques: el modelo original más sólido.", "#eb6834"))
    o = R["optimismo_validacion_aleatoria"]["F1 macro CV"]
    tarjetas = nuevo + [
        hallazgo(f"{CV['f1_macro (media)'][B]:.2f} → {T['f1_macro'][B]:.2f}", "Modelo base: validación kNNDM y test",
                 f"La validación anticipa el test; {T['accuracy_±1'][B]:.0%} de aciertos a ±1 estrato."),
        hallazgo(f"{o['aleatoria (StratifiedKFold)']:.2f} → {o['espacial (kNNDM)']:.2f}",
                 "F1 con validación aleatoria frente a kNNDM",
                 "Mezclar filas al azar promete un F1 que en zonas nuevas no se cumple.", "#e34948"),
        hallazgo(f"{D['moran_estrato']['I']:.2f}", "I de Moran del estrato", "La ciudad está segregada: la ubicación domina.", "#0d366b"),
        hallazgo("96.9 %", "Cobertura con la ubicación por manzana", "La ciudad informal entra al modelo sin costo en F1.", "#eb6834"),
    ]
    if ar:
        zc = [v["vs Logística B v3 (sin vecindad)"]["cv_F1"]["media"]
              for v in ar["xgb_geografico_vs_logistica"]["zonas conocidas"]["variantes"].values()]
        tarjetas.append(hallazgo(f"+{min(zc):.2f} a +{max(zc):.2f}", "XGBoost geográfico frente al base en zonas conocidas",
                                 "Con datos cerca, la parte geográfica sí aporta: el escenario manda.", "#4a3aa7"))
    v3 = FP.v3_e1()
    if v3:
        q = v3["distancias"]["cuantiles"]
        tarjetas.append(hallazgo(
            f"{q['CV de la versión 2 (bloques + buffer de 1 km)']['mediana (km)']:.1f} → "
            f"{q['Escenario: zona nueva de 2 km (test → todo train)']['mediana (km)']:.2f} km",
            "Distancia al dato más cercano: validación anterior frente al escenario",
            "Por eso la versión 3 valida con kNNDM, a la distancia a la que se predice.", "#2a78d6"))
    visual = dbc.Row([dbc.Col(t, md=4, className="mb-3") for t in tarjetas], className="g-3")
    return diapositiva("6.1", "Hallazgos principales", visual)


def d_respuesta():
    visual = html.Div([
        html.Div("¿Se puede predecir el estrato con el catastro abierto?", className="pregunta-grande"),
        html.Div("En parte.", className="respuesta-grande"),
        dbc.Row([
            dbc.Col(html.Div([html.H5("Lo que sí revela"), html.Ul([
                html.Li("La zona: dónde está la vivienda y cómo son sus vecinas."),
                html.Li("El tamaño y la tipología: área, baños, régimen.")])], className="caja-si"), md=6),
            dbc.Col(html.Div([html.H5("Lo que no está publicado"), html.Ul([
                html.Li("El puntaje de las edificaciones."),
                html.Li("Las zonas homogéneas del DANE.")])], className="caja-no"), md=6)], className="g-3 mt-2")])
    return diapositiva("6.2", "Respuesta a la pregunta del proyecto", visual,
                       conclusion="La parte pública del catastro revela sobre todo la zona; lo que distingue a la vivienda "
                                  "dentro de su zona no está publicado.")


def d_limitaciones():
    items = [("Test pequeño", "7 bloques, con todo el estrato 6 en uno: intervalos anchos."),
             ("Submuestra", "Los modelos se compararon con 10 001 viviendas."),
             ("Etiqueta administrativa", "El estrato puede estar desactualizado o ajustado a mano."),
             ("Uso ético", "No usar el modelo para decidir sobre personas (crédito, seguros): el uso es analítico.")]
    visual = dbc.Row([dbc.Col(html.Div([html.Div(t, className="lim-titulo"), html.P(d, className="lim-texto")],
                                       className="limitacion"), md=6, className="mb-3") for t, d in items], className="g-3")
    return diapositiva("6.3", "Limitaciones", visual)


def d_pasos():
    pasos = [("1", "Zonas conocidas", "Probar el EPL donde hay datos cerca: ahí la información local sí ayudó."),
             ("2", "Más zonas de evaluación", "Con 7 bloques no se confirman diferencias pequeñas."),
             ("3", "Modelo en dos etapas", "Primero la zona y luego la vivienda atípica, como el DANE.")]
    return diapositiva("8.4", "Próximos pasos", flujo(pasos))


def d_recursos():
    enlaces = [("Libro del Entregable 4 (proyecto final)", DE4.LIBRO_E4),
               ("Libro del Entregable 1 (versión 3)", LIBRO_E1),
               ("E1 · cap. 15: EDA mejorado", LIBRO_E1 + "notebooks/15_eda_mejorado.html"),
               ("Libro del Entregable 2 (versión 3)", LIBRO_E2),
               ("E2 · cap. 9: todos los modelos originales de la revisión", LIBRO_E2 + "notebooks/09_modelos_revision.html"),
               ("Libro del Entregable 3", LIBRO_E3),
               ("E3 · cap. 11: todos los datos y otros modelos de la revisión", LIBRO_E3 + "notebooks/11_mejoras.html"),
               ("Libro del Entregable 1 (versión 2)", LIBRO_E1_V2), ("Libro del Entregable 1 (versión 1, entregada)", LIBRO_E1_V1),
               ("Libro del Entregable 2 (versión 2)", LIBRO_E2_V2), ("Libro del Entregable 2 (versión 1, entregada)", LIBRO_E2_V1),
               ("Repositorio del Entregable 1", "https://github.com/mcarocantillo/entregable1-ml-barranquilla"),
               ("Repositorio del Entregable 2", "https://github.com/mcarocantillo/entregable2-ml-barranquilla"),
               ("Repositorio del Entregable 3", "https://github.com/mcarocantillo/entregable3-ml-barranquilla"),
               ("Repositorio del Entregable 4", "https://github.com/mcarocantillo/entregable4-ml-barranquilla")]
    visual = html.Div([html.A([html.Span(t), html.Span("↗", className="enlace-flecha")], href=u, target="_blank", className="enlace-recurso")
                       for t, u in enlaces], className="recursos")
    return diapositiva("6.5", "Para profundizar", visual,
                       puntos=["Cada diapositiva indica el capítulo del libro de donde sale.",
                               "El código reproduce todas las cifras con la semilla 42.",
                               "Glosario de métricas: botón «Glosario» arriba a la derecha."])


# ===========================================================================
# Índice de pestañas
# ===========================================================================
PESTANAS = [
    {"id": "contexto", "titulo": "Contexto: problema, revisión y datos", "corto": "Contexto", "diapositivas": [
        ("La pregunta", d_pregunta), ("Cómo asigna el DANE", d_dane), ("Revisión bibliográfica", d_bibliografia),
        ("El vacío en la literatura", d_vacio), ("Objetivo e hipótesis", d_hipotesis),
        ("Formulación matemática", d_formulacion), ("Los datos", d_datos), ("Limpieza", d_embudo),
        ("Partición espacial", d_particion)]},
    {"id": "eda", "titulo": "Análisis exploratorio", "corto": "EDA", "diapositivas": [
        ("Variable objetivo", d_objetivo), ("Faltantes e imposibles", d_calidad), ("Outliers", d_outliers),
        ("Numéricas", d_numericas), ("Transformación", d_transformar), ("Categóricas vs estrato", d_categoricas),
        ("Correlaciones", d_correlaciones), ("Ranking de asociación", d_ranking), ("Mapa", d_mapa),
        ("Autocorrelación", d_moran), ("Dependencia y tamaño efectivo", d_dependencia), ("Fuga de datos", d_fuga),
        ("El entorno", d_eda_entorno), ("El test", d_eda_test)]},
    # En el orden del guion de la sustentación: fundamentos del EPL, implementación y diseño experimental, resultados.
    {"id": "modelos", "titulo": "Modelos: fundamentos, diseño experimental y resultados", "corto": "Modelos",
     "diapositivas": [
        ("E4 · El modelo original (EPL)", DE4.d_e4_idea), ("E4 · Fundamentos: modelo y pérdida", DE4.d_e4_formulacion),
        ("E4 · Fundamentos: optimización", DE4.d_e4_optimizacion),
        ("E4 · Fundamentos: regularización", DE4.d_e4_generalizacion), ("E4 · Campo de pesos", DE4.d_e4_campo),
        ("E4 · Costo computacional", DE4.d_e4_complejidad),
        ("E2 · Validar a la distancia correcta", d_v3_validacion), ("E2 · La vecindad física", d_vecindad_knn),
        ("E2 · Modelos de la revisión", d_candidatos), ("E3 · Pipeline combinatorio", d_e3_diseno),
        ("E3 · Optimizadores", d_e3_optimizadores), ("E3 · Balanceo", d_e3_balanceo),
        ("E4 · Correcciones del E3", DE4.d_e4_correcciones), ("E4 · Validación repetida 5 × 5", DE4.d_e4_e3_vs_5x5),
        ("E4 · Implementación propia (R2)", DE4.d_e4_implementacion), ("E4 · Predicción en vivo", DE4.d_e4_prediccion),
        ("Base · Frente a líneas base", d_comparacion), ("Base · ¿Diferencias reales?", d_significancia),
        ("E2 · Todos los originales", d_originales), ("E4 · Los 11 modelos", DE4.d_e4_lideres),
        ("E4 · Friedman y Nemenyi", DE4.d_e4_cd), ("E4 · Superioridad (R3)", DE4.d_e4_pruebas),
        ("E4 · Frente al modelo base", DE4.d_e4_base), ("E4 · Reglas estándar y ablación", DE4.d_e4_ablacion),
        ("E4 · Diagnóstico de sobreajuste", DE4.d_e4_sobreajuste), ("E4 · Test confirmatorio", DE4.d_e4_test),
        ("E4 · Diagnóstico en el test", DE4.d_e4_diagnostico), ("E4 · Robustez y sensibilidad", DE4.d_e4_robustez)]},
    {"id": "conclusiones", "titulo": "Conclusiones", "diapositivas": [
        ("Veredicto del E4", DE4.d_e4_veredicto), ("Conjuntos externos", DE4.d_e4_externos), ("Respuesta", d_respuesta),
        ("Limitaciones", d_limitaciones), ("Próximos pasos", d_pasos)]},
]

# Quién presenta cada diapositiva en la sustentación (M = María Carolina, J = Juan Camilo), en el orden de cada pestaña.
PRESENTA = {"contexto": "MMMMMMJJJ", "eda": "J" * 14, "modelos": "M" * 6 + "J" * 13 + "M" * 9, "conclusiones": "MMMMM"}
NOMBRE_PRESENTA = {"M": "María Carolina", "J": "Juan Camilo"}
assert all(len(PRESENTA[p["id"]]) == len(p["diapositivas"]) for p in PESTANAS)
