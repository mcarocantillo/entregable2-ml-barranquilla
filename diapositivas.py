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
from dash import Input, Output, callback, dcc, html

import figuras as F
import figuras_mejoras as FM
import figuras_proyecto as FP
from componentes import (control, controles, desplegable, diapositiva, fmt, grafico, interruptor, kpi, radio, tabla)

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
                       puntos=["El estrato fija la tarifa de los servicios públicos de cada vivienda.",
                               "Lo asigna cada alcaldía con la metodología del DANE.",
                               "Problema: clasificación multiclase ordinal, con datos espaciales."],
                       conclusion="Predecir el estrato (1 a 6) de cada vivienda de Barranquilla con su ficha catastral y su "
                                  "ubicación.", fuente=f"{E2C} · cap. 1")


def d_dane():
    pasos = [("1", "Zonas", "Se cruzan zonas homogéneas físicas y de valor del suelo"),
             ("2", "Estrato de la zona", "Puntaje de las edificaciones (0 a 100) y cortes por municipio"),
             ("3", "Viviendas atípicas", "Fuera de Q1 − 1.5·RIC o Q3 + 1.5·RIC: ±1 estrato")]
    insumos = pd.DataFrame([
        ["Zonas homogéneas", "No: se aproximan con coordenadas y vecindad"],
        ["Puntaje de las edificaciones", "No"],
        ["Área y cuartos de apartamentos", "Sí"],
        ["Casa o apartamento", "Sí"],
        ["Revisión del comité de estratificación", "No observable"],
    ], columns=["Insumo del DANE", "¿Está en el catastro abierto?"])
    visual = html.Div([flujo(pasos), html.Div(tabla(insumos), className="mt-3")])
    return diapositiva("1.2", "Cómo asigna el DANE el estrato", visual,
                       puntos=["El estrato es, ante todo, una propiedad de la zona.",
                               "Una vivienda solo se aparta de su zona si es claramente atípica.",
                               "No usa los avalúos catastrales ni datos del hogar (ingreso, ocupación, educación).",
                               "Los dos insumos clave del DANE no son públicos."],
                       conclusion="La pregunta real: ¿cuánto del estrato se recupera con la parte pública del catastro?",
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


def d_bibliografia():
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
        ["7", "TPE-SVM", "SVM con optimización bayesiana", "Sí (ronda 1)"],
        ["8", "Stacking", "Un metamodelo combina varios modelos", "Sí"],
        ["9", "NSGA-II", "Hiperparámetros y variables a la vez", "No: selector de variables"],
    ], columns=["#", "Candidato original (Tabla III)", "Idea", "¿Se probó?"])
    visual = html.Div([
        dbc.Row([dbc.Col(kpi("0", "estudios que clasifiquen el estrato con el catastro", "Scopus, 5-oct-2026"), md=4),
                 dbc.Col(kpi("1998", "antecedente más cercano", "logística con datos censales agregados"), md=4),
                 dbc.Col(kpi("5 de 9", "candidatos originales probados", "los otros: optimizadores o selectores"), md=4)],
                className="g-3 mb-3"),
        tabla(t), html.Div(tabla(usados), className="mt-3")])
    return diapositiva("1.4", "Revisión bibliográfica: la brecha y los candidatos", visual,
                       puntos=["Los trabajos cercanos usan XGBoost, Random Forest, SVR o la logística, pero para el "
                               "precio de la vivienda o con datos agregados por zona, no para el estrato.",
                               "La Tabla III reúne nueve modelos que reducen el error frente a los clasificadores "
                               "convencionales y nunca se aplicaron a este problema: los candidatos originales.",
                               "El mejor respaldado, el XGBoost geográfico, solo existía para regresión."],
                       conclusion="Nadie ha clasificado el estrato con variables catastrales: esa es la brecha.",
                       fuente=f"{E2C} · caps. 1.3 y 9.1")


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
        html.P(["Captura de Web of Science Core Collection (búsqueda de Juan Camilo Oñoro): la consulta ",
                html.Code(consulta), " devolvió 0 documentos. Repetida directamente desde Advanced Search, sin refinar "
                "resultados anteriores (19-oct-2026), tampoco devolvió ninguno."],
               style={"fontSize": ".85rem", "color": "#52514e", "marginTop": "6px"}),
        html.Div("Por qué no hay estudios con los que compararse", className="subtitulo-visual mt-3"),
        tabla(diferencias)])
    return diapositiva("1.5", "El vacío en la literatura: por qué no hay con quién compararse", visual,
                       puntos=["El DANE: el cobro por estratos «se institucionalizó sin que existiera un modelo similar en "
                               "otros países». La variable objetivo no existe fuera de Colombia.",
                               "Los 9 trabajos cercanos en Scopus predicen el precio de la vivienda (5), la pobreza por "
                               "zonas (1) o la elección residencial (1998), o usan el nivel socioeconómico como entrada.",
                               "Brecha de aplicación: nadie clasifica el estrato de cada vivienda con el catastro. Brecha "
                               "de método: los 9 modelos originales nunca se aplicaron a este problema.",
                               "El DANE usa insumos que el catastro abierto no publica: la pregunta es cuánto del estrato "
                               "se recupera con la parte pública.",
                               "Web of Science lo confirma: la consulta, sin refinar, no devuelve ningún documento. "
                               "Falta revisar SciELO, Redalyc y tesis colombianas."],
                       conclusion="Sin cifras externas con las que compararse, todo se mide contra un modelo base propio y "
                                  "con el mismo protocolo espacial.",
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
                           "y EDA mejorado")], className="hito destacado"),
    ], className="hitos")
    return diapositiva("1.5", "Cómo se trabajó", html.Div([flujo(pasos), hitos]),
                       puntos=["Las reglas que miran una sola fila se aplican antes de partir.",
                               "Todo lo que se aprende de los datos se calcula solo con entrenamiento.",
                               "El test son zonas completas que el modelo nunca vio."],
                       conclusion="El orden de los pasos evita la fuga de datos.", fuente="Entregables 1, 2 y 3")


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
    return diapositiva("2.3", "Cobertura v1 → v2: el modelo ahora ve la ciudad informal", grafico(FP.fig_cobertura_v1_v2()),
                       puntos=[f"Estrato 1: {pct(c1.v1, c1.total, 0)} → {pct(c1.v2, c1.total, 0)} de las viviendas.",
                               f"Estrato 2: {pct(COB.loc[2].v1, COB.loc[2].total, 0)} → {pct(COB.loc[2].v2, COB.loc[2].total, 0)}.",
                               "Los estratos 3 a 6 casi no cambian: ya tenían coordenadas."],
                       conclusion="La versión 2 corrige un sesgo: el estrato 1 estaba subrepresentado.",
                       fuente=f"{E1C} v2 · cap. 1.7")


def d_v1v2():
    c = FP._json("comparacion_v1_v2.json")["test"]
    rm = FP._json("comparacion_v1_v2.json")["recall_por_manzana"]
    return diapositiva("2.4", "¿Cambia el modelo? v1 frente a v2 con la misma vara", grafico(FP.fig_v1_v2_modelo()),
                       puntos=[f"Mismo test y mismos folds: v2 {c['v2 (pesos balanceados)']['F1 test']:.3f} frente a "
                               f"v1 con pesos {c['v1 con pesos balanceados']['F1 test']:.3f}: empate.",
                               "La mejora frente a la v1 entregada viene de los pesos de clase balanceados.",
                               f"En las {fmt(rm['n_test_por_manzana'])} viviendas de test ubicadas por manzana: recall del estrato 1 "
                               f"{rm['v1 (sin pesos)']['1']:.2f} → {rm['v2']['1']:.2f}, pero el del 2 baja "
                               f"{rm['v1 (sin pesos)']['2']:.2f} → {rm['v2']['2']:.2f}."],
                       conclusion="La v2 amplía la cobertura sin costo en F1; en lo informal cambia qué estrato acierta.",
                       metricas=[("f1_macro", T["f1_macro"][B]), "ic95"], fuente="Comparación v1–v2")


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
        grafico(F.fig_kish(GX["icc"], GX["m_pond"]), "kish-fig"),
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
    visual = html.Div([controles(control("Vista", radio("part-vista", [("grupo", "Train / test / buffer"),
                                                                       ("fold", "Folds de la CV"),
                                                                       ("balance", "Mezcla de estratos")], "grupo"))),
                       grafico(FP.fig_particion_mapa("grupo"), "part-fig")])
    return diapositiva("2.8", "Partición espacial: el test son zonas que el modelo nunca vio", visual,
                       puntos=[f"Test: {fmt(R['n_test'])} viviendas en {R['bloques_test']} bloques de 2 km.",
                               f"Entrenamiento: {fmt(R['n_train_antes_buffer'])}; quedan {fmt(R['n_train'])} tras el buffer de 1 km.",
                               "La validación cruzada repite la idea dentro de train: 5 folds de bloques completos.",
                               "El test tiene más estrato 1 y menos estrato 6 que train: el azar de las zonas."],
                       conclusion="Así se mide lo que importa: predecir en barrios sin datos.",
                       metricas=[("buffer", 1.0, "1 km")], fuente=f"{E1C} · cap. 2")


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
                               "Un buffer de 1 km reduce la fuga sin vaciar el entrenamiento."],
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
                               "Con estas variables, el F1 de test de la logística sube de 0.49 a 0.57 (pestaña 6)."],
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
                               "ADASYN es el peor balanceo (pestaña 7)."],
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
    a, e = o["F1 macro CV"]["aleatoria (StratifiedKFold)"], o["F1 macro CV"]["espacial (bloques + buffer)"]
    return diapositiva("4.1", "Protocolo: validar como se va a usar", grafico(F.fig_optimismo()),
                       puntos=[f"Mezclando filas al azar, el F1 de CV sería {a:.2f}; separando zonas, {e:.2f}.",
                               "La diferencia es fuga: vecinos casi idénticos en train y validación.",
                               "Más buffer quita más fuga pero también más datos."],
                       conclusion="Hay que separar zonas; la v3 mide cuánto con la distancia al vecino conocido.",
                       metricas=[("optimismo", a - e), ("buffer", 1.0, "1 km")], fuente=f"{E1C} · cap. 10.2")


def d_grid():
    fig = FP.fig_grid()
    visual = grafico(fig) if fig is not None else html.P("La búsqueda de hiperparámetros aún se está calculando.")
    return diapositiva("4.2", "Hiperparámetros: el más simple que no pierde", visual,
                       puntos=["Se probaron C = 0.01, 0.1, 1 y 10, con y sin pesos de clase.",
                               "La superficie es plana: las diferencias caben en un error estándar.",
                               "La regla 1-SE elige el modelo más regularizado: C = 0.01 en A y en B.",
                               "B usa pesos balanceados; A, sin pesos."],
                       conclusion="Ante un empate, se prefiere el modelo más simple.",
                       metricas=["C", "class_weight", "se_pareado"], fuente=f"{E1C} · cap. 10.2")


def d_comparacion():
    visual = html.Div([controles(control("Métrica", desplegable("comp-met", list(F.METRICAS_COMPARABLES.items()), "f1_macro")),
                                 control("Vista", radio("comp-vista", [("fig", "Gráfica"), ("tabla", "Tabla")], "fig"))),
                       html.Div(grafico(F.fig_comparacion("f1_macro")), id="comp-cont")])
    return diapositiva("4.3", "Modelo base frente a las líneas base", visual,
                       puntos=[f"Logística B: F1 {CV['f1_macro (media)'][B]:.3f} en CV y {T['f1_macro'][B]:.3f} en test.",
                               "Supera con claridad a las Dummy en todas las métricas.",
                               f"La moda por zona ({R.get('tam_zona_moda_km', 4):g} km) es exigente: B la supera en todo, "
                               "pero en F1 sin significancia."],
                       conclusion="La logística con variables físicas y ubicación es la referencia.",
                       metricas=[("f1_macro", T["f1_macro"][B]), ("mae_ordinal", T["MAE_ordinal"][B]),
                                 ("kappa", T["kappa_cuadrático"][B])], fuente=f"{E1C} · cap. 10.4")


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
    return diapositiva("4.4", "¿Las ventajas son reales o ruido?", grafico(fig_significancia()),
                       puntos=["Bootstrap por bloques: se remuestrean zonas, no viviendas.",
                               "B gana a las Dummy y a A (solo físicas) con intervalos lejos del 0.",
                               "Frente a la moda por zona, el intervalo incluye el 0."],
                       conclusion="La ubicación aporta de forma significativa; ganarle a la geografía sola no está demostrado.",
                       metricas=["ic95"], fuente=f"{E1C} · cap. 10.4")


OPC_BASE = [(m, F.NOMBRE_CORTO[m]) for m in F.MODELOS]


def d_confusion():
    visual = html.Div([controles(control("Modelo", desplegable("conf-mod", OPC_BASE, B, 300)),
                                 control("Vista", radio("conf-vista", [("matriz", "Matriz"), ("clase", "Por estrato")], "matriz")),
                                 control("Valores", interruptor("conf-norm", "% de la fila", True))),
                       grafico(F.fig_confusion(B), "conf-fig")])
    rc = F.tabla_por_clase(B)
    return diapositiva("4.5", "¿En qué se equivoca?", visual,
                       puntos=[f"Estratos 1, 2 y 3 se reconocen bien (recall {rc.Recall[0]:.2f}, {rc.Recall[1]:.2f} y {rc.Recall[2]:.2f}).",
                               f"El estrato 4 es el más difícil (recall {rc.Recall[3]:.2f}): se confunde con 3 y 5.",
                               "Casi todos los errores son con un estrato vecino."],
                       conclusion=f"El {T['accuracy_±1'][B]:.0%} de las predicciones cae en el estrato real o en uno vecino.",
                       metricas=[("recall", rc.Recall[3], f"{rc.Recall[3]:.2f} (estrato 4)"), ("acc1", T["accuracy_±1"][B])],
                       fuente=f"{E1C} · cap. 10.4")


@callback(Output("conf-fig", "figure"), Input("conf-mod", "value"), Input("conf-vista", "value"), Input("conf-norm", "value"))
def _conf(m, v, n):
    return F.fig_por_clase(m) if v == "clase" else F.fig_confusion(m, bool(n))


def d_roc():
    visual = html.Div([controles(control("Modelo", desplegable("roc-mod", OPC_BASE, B, 300)),
                                 control("Vista", radio("roc-vista", [("roc", "Curvas ROC"), ("cal", "Calibración")], "roc"))),
                       grafico(F.fig_roc(B), "roc-fig")])
    return diapositiva("4.6", "Probabilidades: ¿ordenan bien y son creíbles?", visual,
                       puntos=[f"AUC macro {T['AUC_OvR_macro'][B]:.2f}: el modelo ordena bien las viviendas.",
                               "Los extremos (1 y 6) se distinguen mejor que los medios.",
                               "Las probabilidades siguen la diagonal: si dice 70 %, acierta cerca del 70 %."],
                       conclusion="Las probabilidades sirven para medir incertidumbre, no solo la clase final.",
                       metricas=[("auc", T["AUC_OvR_macro"][B]), ("ece", 0.03, "0.03")], fuente=f"{E1C} · cap. 10.4")


@callback(Output("roc-fig", "figure"), Input("roc-mod", "value"), Input("roc-vista", "value"))
def _roc(m, v):
    return F.fig_calibracion(m) if v == "cal" else F.fig_roc(m)


def d_coeficientes():
    visual = html.Div([controles(control("Vista", radio("coef-vista", [("odds", "Odds 6 frente a 1"), ("B", "Coeficientes B"),
                                                                       ("A", "Coeficientes A")], "odds"))),
                       grafico(FP.fig_odds(), "coef-fig")])
    return diapositiva("4.7", "Qué aprendió el modelo", visual,
                       puntos=["Más al norte, más área y más baños: más probable el estrato 6 que el 1.",
                               "Ser informal o casa de hasta 3 pisos empuja hacia el estrato 1.",
                               "Con las demás variables fijas, más habitaciones restan: el área ya dice el tamaño."],
                       conclusion="Los coeficientes cuentan la misma historia que el EDA.",
                       metricas=[("odds_ratio", 18.0, "×18 por +1 desv. de área")], fuente=f"{E1C} · cap. 10.6")


@callback(Output("coef-fig", "figure"), Input("coef-vista", "value"))
def _coef(v):
    return FP.fig_odds() if v == "odds" else F.fig_coeficientes("B_fisicas+ubicacion" if v == "B" else "A_fisicas")


def d_permutacion():
    return diapositiva("4.8", "Importancia por permutación", grafico(FP.fig_importancia_permutacion()),
                       puntos=["Se calcula en una zona de validación (fold 1), no en test.",
                               "Barajar la distancia al centro es lo que más daño hace.",
                               "En esa zona, barajar x o y mejora el F1: la posición absoluta no siempre se traslada a "
                               "un barrio nuevo."],
                       conclusion="La ubicación relativa generaliza mejor que las coordenadas absolutas.",
                       metricas=["importancia_perm"], fuente=f"{E1C} · cap. 10.6")


def d_aprendizaje():
    ca = D["curva_aprendizaje"]
    return diapositiva("4.9", "¿Más datos ayudarían?", grafico(F.fig_curva_aprendizaje()),
                       puntos=[f"La validación se queda entre {min(ca['val_media']):.2f} y {max(ca['val_media']):.2f} "
                               f"desde {fmt(ca['n'][0])} hasta {fmt(ca['n'][-1])} viviendas.",
                               "El entrenamiento no sube mucho: no es sobreajuste clásico.",
                               "La brecha es la distancia entre barrios conocidos y nuevos."],
                       conclusion="Más filas del mismo tipo no ayudan: falta otra información.",
                       fuente=f"{E1C} · cap. 10.3")


def d_residuos():
    m = D["moran_residuos"]["B_fisicas+ubicacion"]
    return diapositiva("4.10", "¿Queda estructura espacial en los errores?", grafico(F.fig_mapa_residuos()),
                       puntos=["Rojo: el modelo predice un estrato más alto que el real; azul, más bajo.",
                               "Los errores vienen en manchas: barrios enteros mal estimados.",
                               "Al modelo le falta información de barrio."],
                       conclusion="La siguiente mejora tiene que venir del entorno, no de la vivienda.",
                       metricas=[("moran", m["I"], f"{m['I']:.2f} (residuos; ideal 0)")], fuente=f"{E1C} · cap. 10.5")


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
# 5. MÁS ALLÁ DEL MODELO BASE
# ===========================================================================
AV = R["avanzados"]


def d_candidatos():
    t = pd.DataFrame([
        ["XGBoost geográfico", "Sí", "Modelo principal, adaptado a clasificación", "Efectos que cambian por barrio"],
        ["XGBoost", "No", "Control del anterior (sin parte geográfica)", "No linealidades e interacciones"],
        ["Rotation Forest", "Sí", "Candidato de la revisión (implementación propia)", "Variables de tamaño correlacionadas"],
        ["SVM RBF (Nyström) + TPE", "Sí", "Candidato de la revisión (ronda 1)", "Fronteras no lineales"],
        ["RF + SVM en dos etapas", "Sí", "Candidato de la revisión (v3)", "Variables redundantes"],
        ["Stacking", "Sí", "Candidato de la revisión (v3)", "Modelos que aciertan estratos distintos"],
    ], columns=["Modelo", "¿Original?", "Papel", "Problema que ataca"])
    pasos = [("1", "Global", "XGBoost con toda la ciudad"), ("2", "Anclas", "Centros de celdas de 2 km"),
             ("3", "Ventana", "15 % del entrenamiento"), ("4", "Locales", "Un XGBoost por ancla"),
             ("5", "Mezcla", "0.75·global + 0.25·local")]
    visual = html.Div([tabla(t), html.Div("Cómo funciona el XGBoost geográfico", className="subtitulo-visual mt-3"), flujo(pasos)])
    return diapositiva("5.1", "El modelo propuesto: XGBoost geográfico para clasificación", visual,
                       puntos=["Adaptación propia: el original (Grekousis, 2025) es de regresión.",
                               "Un XGBoost global más XGBoost locales alrededor de anclas, con un kernel espacial.",
                               "Mezcla p = α·global + (1 − α)·local; con α = 1 es su control, el XGBoost global.",
                               "Compite con los demás candidatos originales de la revisión (Rotation Forest, SVM con "
                               "TPE, RF + SVM y stacking) con el mismo protocolo.",
                               "«Original»: la revisión no halló ninguna aplicación del modelo al estrato con datos "
                               "catastrales."],
                       conclusion="Hipótesis: los modelos locales aprenden lo que cambia de un barrio a otro.",
                       fuente=f"{E2C} · caps. 4 y 9.1")


def d_avanzados():
    visual = html.Div([controles(control("Métrica", desplegable("avz-met", list(F.METRICAS_COMPARABLES.items()), "f1_macro")),
                                 control("Vista", radio("avz-vista", [("fig", "Gráfica"), ("tabla", "Tabla")], "fig"))),
                       html.Div(grafico(F.fig_comparacion_avanzados("f1_macro")), id="avz-cont")])
    return diapositiva("5.2", "Resultados: nadie le gana a la logística en test", visual,
                       puntos=[f"En CV, XGBoost y la SVM llegan a {CV['f1_macro (media)']['XGBoost']:.3f} frente a "
                               f"{CV['f1_macro (media)'][B]:.3f} de la logística.",
                               f"En test, la logística es la mejor ({T['f1_macro'][B]:.3f}); la SVM y Rotation Forest "
                               "quedan cerca (0.45).",
                               f"Rotation Forest tiene el menor error ordinal (MAE {T['MAE_ordinal']['Rotation Forest']:.2f})."],
                       conclusion="Más flexibilidad no se traduce en mejores predicciones en barrios nuevos.",
                       metricas=[("f1_macro", T["f1_macro"]["SVM RBF (Nyström) + TPE"], "0.45 (SVM)"),
                                 ("mae_ordinal", T["MAE_ordinal"]["Rotation Forest"], "0.39 (Rotation Forest)")],
                       fuente=f"{E2C} · cap. 6.3")


@callback(Output("avz-cont", "children"), Input("avz-met", "value"), Input("avz-vista", "value"))
def _avz(met, vista):
    return tabla(F.tabla_metricas_avanzados()) if vista == "tabla" else grafico(F.fig_comparacion_avanzados(met))


def d_criterio():
    s = AV["cv_pareado"]["SVM RBF (Nyström) + TPE"]
    visual = html.Div([controles(control("Vista", radio("crit-vista", [("mapa", "CV frente a test"), ("folds", "Por fold")], "mapa"))),
                       grafico(FP.fig_criterio_cv_test(), "crit-fig")])
    return diapositiva("5.3", "La SVM pasa el criterio de la CV, pero el test no lo confirma", visual,
                       puntos=[f"SVM: +{s['media']:.3f} de F1 en CV con error estándar {s['se']:.3f}: cumple el criterio.",
                               "Pero sus hiperparámetros se buscaron en esos mismos folds: su CV es la más optimista.",
                               "En test queda por debajo (−0.04, no significativo) y con errores más grandes."],
                       conclusion="Se mantiene la logística, como desviación declarada del criterio.",
                       metricas=[("se_pareado", None), ("ic95", None)], fuente=f"{E2C} · cap. 6.3")


@callback(Output("crit-fig", "figure"), Input("crit-vista", "value"))
def _crit(v):
    return FP.fig_cv_pareado() if v == "folds" else FP.fig_criterio_cv_test()


OPC_TODOS = [(m, F.NOMBRE_CORTO[m]) for m in [B] + F.AVANZADOS]


def d_errores():
    visual = html.Div([controles(control("Modelo", desplegable("err-mod", OPC_TODOS, "XGBoost", 300)),
                                 control("Vista", radio("err-vista", [("matriz", "Matriz"), ("clase", "Por estrato")], "matriz"))),
                       grafico(F.fig_confusion("XGBoost"), "err-fig")])
    return diapositiva("5.4", "Cada modelo se equivoca distinto", visual,
                       puntos=["XGBoost colapsa la escala: casi no predice el 3 y manda al 4 casi todo el 5 y el 6.",
                               "La SVM reconoce mejor los extremos (1, 5 y 6) y falla en el 2.",
                               "La logística es la más equilibrada."],
                       conclusion="Los árboles no extrapolan: en un barrio nuevo copian la zona más parecida.",
                       metricas=["recall"], fuente=f"{E2C} · caps. 6.4–6.5")


@callback(Output("err-fig", "figure"), Input("err-mod", "value"), Input("err-vista", "value"))
def _err(m, v):
    return F.fig_por_clase(m) if v == "clase" else F.fig_confusion(m)


def d_geoxgb():
    return diapositiva("5.5", "XGBoost geográfico: los modelos locales no ayudan en zonas nuevas", grafico(F.fig_geoxgb_busqueda()),
                       puntos=["Dar más peso a los modelos locales (α bajo) empeora el F1 de CV.",
                               "Los modelos locales aprenden de los barrios vecinos, no del barrio de la vivienda.",
                               "En test sube el F1 de su control (0.375 → 0.418), pero en CV pasa lo contrario."],
                       conclusion="Los efectos locales necesitan datos de la misma zona.",
                       metricas=[("alpha_geo", None, "α = 0.75 (elegido)")], fuente=f"{E2C} · cap. 6.1")


def d_residuos_av():
    visual = html.Div([controles(control("Modelo", desplegable("res-mod", OPC_TODOS, "XGBoost geográfico", 300))),
                       grafico(F.fig_mapa_residuos_modelo("XGBoost geográfico"), "res-fig")])
    mr = {m: D["moran_residuos"][m]["I"] for m in F.AVANZADOS}
    return diapositiva("5.6", "Residuos: ningún modelo borra el patrón de barrio", visual,
                       puntos=[f"Logística: Moran de residuos {D['moran_residuos']['B_fisicas+ubicacion']['I']:.2f}.",
                               f"Modelos nuevos: entre {min(mr.values()):.2f} y {max(mr.values()):.2f}, peor.",
                               "La información que falta no está en la ficha de la vivienda."],
                       conclusion="El cuello de botella es la información, no el algoritmo.",
                       metricas=[("moran", max(mr.values()), f"{max(mr.values()):.2f} (SVM, residuos)")],
                       fuente=f"{E2C} · cap. 6.6")


@callback(Output("res-fig", "figure"), Input("res-mod", "value"))
def _res(m):
    return F.fig_mapa_residuos_modelo(m)


def d_vecindad():
    c = FP._json("comparacion_zonas_nuevas.json", "vecindad")
    p = c["comparaciones_vs_B"]["B + vecindad (300 m y 1 km)"]
    visual = html.Div([grafico(FP.fig_vecindad_forest()),
                       html.Div("12 variables: 6 atributos físicos de las viviendas vecinas, a 300 m y a 1 km. Nunca su estrato.",
                                className="nota-visual")])
    return diapositiva("5.7", "Vecindad física: ¿ayuda describir el entorno?", visual,
                       puntos=[f"Modelo principal (fijado antes): +{p['cv_dif_media']:.3f} en CV, sin superar un error estándar.",
                               f"En test sube el F1 de {T['f1_macro'][B]:.2f} a {c['modelos']['B + vecindad (300 m y 1 km)']['test']['f1_macro']:.2f}, "
                               "con intervalo que incluye el 0.",
                               "El error ordinal sí baja de forma significativa (MAE −0.11).",
                               "Las variantes de una sola escala pasan la CV, pero son exploratorias."],
                       conclusion="Resultado mixto: prometedor, no demostrado.",
                       metricas=[("se_pareado", None), ("ic95", None)], fuente=f"{E2C} · cap. 7.2")


def d_vecindad15():
    return diapositiva("5.8", "Con 15 folds: la mejora no se confirma", grafico(FP.fig_vecindad_15folds()),
                       puntos=["Más folds = más comparaciones por pares, solo con entrenamiento.",
                               "La vecindad gana en 10 zonas y pierde en 5, a veces por mucho.",
                               "Las dos versiones con vecindad ganan y pierden en las mismas zonas."],
                       conclusion="El efecto depende de la zona, no del azar del modelo.",
                       metricas=[("se_pareado", None)], fuente=f"{E2C} · cap. 7.2")


def d_donde():
    visual = html.Div([controles(control("Vista", radio("don-vista", [("bloques", "Por bloque"), ("heterogeneidad", "Entorno mezclado"),
                                                                      ("atipicidad", "Vivienda atípica")], "bloques")),
                                 control("Datos", radio("don-conj", [("test", "Test"), ("cv", "CV 15 folds")], "test"))),
                       grafico(FP.fig_vecindad_bloques(), "don-fig")])
    z = FP._json("zonas.json", "vecindad")["correlaciones_bloques"]["y_km"]
    return diapositiva("5.9", "¿Dónde ayuda la vecindad?", visual,
                       puntos=[f"Ayuda más en los bloques del norte (Spearman ρ = {z['rho']:.2f}, p < 0.001).",
                               "En test ayuda a las viviendas parecidas a su entorno y daña a las atípicas.",
                               "En la CV ese patrón no se repite: no es una regla general."],
                       conclusion="La vecindad suma en el norte; el modelo no sabe cuándo no confiar en ella.",
                       metricas=["spearman"], fuente=f"{E2C} · cap. 7.5")


@callback(Output("don-fig", "figure"), Input("don-vista", "value"), Input("don-conj", "value"))
def _don(v, c):
    return FP.fig_vecindad_bloques() if v == "bloques" else FP.fig_vecindad_donde(c, v)


def d_conocidas():
    ze, zm = FP.zonas_conocidas("edificio"), FP.zonas_conocidas("manzana")
    me = ze["modelos"]
    puntos = [f"Ocultando edificios: XGBoost {me['XGBoost']['F1_macro']:.2f} frente a la logística {me['Logística B']['F1_macro']:.2f}.",
              f"Votar con las 15 viviendas conocidas más cercanas ya da {me['Vecinos más cercanos (k = 15)']['F1_macro']:.2f}."]
    if zm:
        mm = zm["modelos"]
        puntos.append(f"Ocultando manzanas completas: vecinos {mm['Vecinos más cercanos (k = 15)']['F1_macro']:.2f}, "
                      f"XGBoost geográfico {mm['XGBoost geográfico']['F1_macro']:.2f}, logística {mm['Logística B']['F1_macro']:.2f}.")
        c = zm.get("comparaciones", {}).get("XGBoost geográfico − Vecinos más cercanos (k = 15)", {}).get("F1")
        if c:
            puntos.append(f"Con manzanas ocultas, el XGBoost geográfico le gana al voto de vecinos: {c['diferencia']:+.2f} "
                          f"(IC 95 % [{c['IC95_inf']:+.3f}, {c['IC95_sup']:+.3f}]).")
        concl = "Parte de la ventaja era copiar la manzana, pero los árboles siguen ganando: aprenden el mapa."
    else:
        puntos.append("La versión que oculta manzanas completas se está calculando.")
        concl = "Donde ya se conocen los estratos del barrio, basta con mirar a los vecinos."
    visual = html.Div([controles(control("Métrica", radio("zc-met", [("F1_macro", "F1 macro"), ("MAE", "MAE ordinal"),
                                                                     ("accuracy", "Accuracy")], "F1_macro"))),
                       grafico(FP.fig_zonas_conocidas(), "zc-fig")])
    return diapositiva("5.10", "Zonas conocidas: el orden se invierte", visual, puntos=puntos, conclusion=concl,
                       metricas=[("f1_macro", me["XGBoost"]["F1_macro"], f"{me['XGBoost']['F1_macro']:.2f} (XGBoost)")],
                       fuente=f"{E2C} · caps. 7.4 y 8")


@callback(Output("zc-fig", "figure"), Input("zc-met", "value"))
def _zc(m):
    return FP.fig_zonas_conocidas(m)


def d_escenarios():
    return diapositiva("5.11", "El mejor modelo depende del escenario", grafico(FP.fig_escenarios()),
                       puntos=["Zonas nuevas (sin datos cerca): la logística es la mejor; los árboles no extrapolan.",
                               "Zonas conocidas: los árboles aprenden el mapa y superan a la logística por más de 0.2.",
                               "Ocultar manzanas completas es la prueba más honesta de ese segundo escenario."],
                       conclusion="Hay que decir para qué escenario se recomienda cada modelo.",
                       fuente=f"{E2C} · caps. 7.4 y 8")


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
                               f"Con el buffer de 1 km, la v2 validaba a {cv2['mediana (km)']:.1f} km y probaba a {tv2:.1f} km: "
                               "un escenario más difícil que el declarado.",
                               f"kNNDM arma los folds para imitar el escenario: {kn['q']} grupos de K-Means, "
                               f"Wasserstein {kn['W_km']:.2f} km (la CV de la v2: {cv2['Wasserstein al escenario (km)']:.2f} km).",
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
        return _sin_v3("4.13", "Versión 3 del modelo base")
    r = d["resumen"]
    b2, b3 = r["B v2 (con buffer)"], r["B v3 (kNNDM, sin buffer)"]
    o3, bu = r["Ordinal v3 (kNNDM, sin buffer)"], r["B v3 + umbrales F1"]
    cvt = d["cv_vs_test"]
    gap2 = cvt["v2: bloques + buffer de 1 km"]["test − CV"]
    gap3 = cvt["v3: kNNDM sin buffer"]["test − CV"]
    hp = d["hiperparametros"]["B v3"]

    def veredicto(m):
        return "supera" if m["Δ CV vs B v3"] > m["SE Δ"] else ("empata con" if m["Δ CV vs B v3"] > -m["SE Δ"] else "queda por debajo de")

    visual = html.Div([controles(control("Métrica", radio("v3m-met", [("f1", "F1 (CV y test)"), ("mae", "MAE ordinal"),
                                                                      ("kappa", "Kappa"), ("e4", "Recall E4")], "f1"))),
                       grafico(FP.fig_v3_modelos(), "v3m-fig")])
    return diapositiva("4.13", "Versión 3 del modelo base", visual,
                       puntos=[f"Mismo test. Sin buffer, C = {hp['C']} y {'sin pesos' if hp['class_weight'] is None else 'pesos balanceados'}: "
                               f"F1 {b3['F1 macro test']:.3f} (v2 {b2['F1 macro test']:.3f}), MAE {b3['MAE test']:.3f} "
                               f"(v2 {b2['MAE test']:.3f}).",
                               f"La CV ahora anticipa mejor el test: brecha {gap3:+.2f} frente a {gap2:+.2f} en la v2.",
                               f"Logística ordinal: {veredicto(o3)} la multinomial en CV; MAE test {o3['MAE test']:.3f} "
                               f"(multinomial {b3['MAE test']:.3f}).",
                               f"Umbrales para el F1 macro: {veredicto(bu)} la multinomial en CV; recall del estrato 4 "
                               f"{b3['recall E4']:.2f} → {bu['recall E4']:.2f}."],
                       conclusion="La validación ahora mide el escenario declarado; el modelo se equivoca por menos, "
                                  "pero el F1 no cambia.",
                       metricas=[("f1_macro", b3["F1 macro test"], f"{b3['F1 macro test']:.3f} (B v3)"), "ordinal", "sesgos_f1"],
                       fuente=f"{E1C} v3 · cap. 14.3–14.7")


@callback(Output("v3m-fig", "figure"), Input("v3m-met", "value"))
def _v3m(m):
    return FP.fig_v3_modelos(m)


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
        dbc.Row([
            dbc.Col(html.Div([html.H5("La brecha"), html.Ul([
                html.Li("Ningún estudio clasifica el estrato de una vivienda con datos catastrales."),
                html.Li("El XGBoost geográfico, el mejor respaldado para datos espaciales, solo existe para regresión.")])],
                className="caja-no"), md=6),
            dbc.Col(html.Div([html.H5("Los modelos propuestos"), html.Ul([
                html.Li("Principal: XGBoost geográfico adaptado por nosotros a clasificación multiclase (un global más "
                        "modelos locales por zona, como la estratificación del DANE)."),
                html.Li("Los demás candidatos originales de la revisión: Rotation Forest, SVM con TPE, RF + SVM y "
                        "stacking.")])],
                className="caja-si"), md=6)], className="g-3"),
        html.Div("Hipótesis", className="pregunta-grande mt-3"),
        html.Div("Los modelos originales de la revisión, encabezados por el XGBoost geográfico, superan al modelo base "
                 "(regresión logística) en F1 macro.", className="respuesta-grande", style={"fontSize": "1.4rem"}),
        html.P("Criterio fijado antes de ver resultados: superarlo por más de un error estándar de la diferencia pareada "
               "en validación cruzada, y que el test (que se mira una sola vez) lo confirme. Después del E3 se agregaron "
               "pruebas de significancia con corrección por comparaciones múltiples.", className="mt-2")])
    return diapositiva("1.4", "Objetivo e hipótesis", visual,
                       puntos=["Objetivo: usar modelos no aplicados antes a este problema (los candidatos de la "
                               "revisión), encabezados por el XGBoost geográfico.",
                               "Y comprobar si alguno es mejor que el modelo base.",
                               "Modelo base: logística multinomial con las variables físicas y la ubicación (pestaña 4).",
                               "Respuesta en las pestañas 6 (Resultados) y 8 (Conclusiones)."],
                       conclusion="La revisión bibliográfica deja una brecha; los modelos originales la atacan.",
                       fuente=f"{E2C} · cap. 1.4")


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
    return diapositiva("6.12", "¿Supera el modelo nuevo al modelo base?", visual,
                       puntos=[f"Zonas nuevas con vecindad: F1 de test {knn['F1_test']:.3f} frente a "
                               f"{knn['F1_test'] - t['diferencia']:.3f} del base (+{t['diferencia']:.3f}, "
                               f"IC 95 % [{t['IC95_inf']:+.3f}, {t['IC95_sup']:+.3f}]).",
                               f"En validación lo supera en {b['cv_F1']['folds_a_favor']} de 5 pliegues "
                               f"(+{b['cv_F1']['media']:.3f}, SE {b['cv_F1']['SE']:.3f}).",
                               f"Zonas conocidas: +{min(zc_b):.2f} a +{max(zc_b):.2f} de F1 en validación.",
                               "Con solo las variables del base, el test no lo confirma (−0.009).",
                               "Ojo: con el intervalo BCa del E3 el mismo +0.09 da [−0.024, +0.172]; con 7 bloques la "
                               "confirmación depende del método. La prueba con 30 bloques sí es significativa (6.14)."],
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
        html.Div(grafico(FM.fig_originales("test", "base")), id="orig-cont")])
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
    return grafico(FM.fig_originales(mir, ref))


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
        grafico(FM.fig_significancia("base"), "sig-fig")])
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
    cajas = html.Div([
        html.Div([html.Div("8 modelos", className="paso-titulo"),
                  html.Div("KNN, Naive Bayes, logística L1/L2, árbol, Random Forest, XGBoost, SVM (Nyström) y XGBoost "
                           "geográfico", className="paso-desc")], className="paso"),
        html.Div([html.Div("× 4 balanceos", className="paso-titulo"),
                  html.Div("ninguno, SMOTE, ADASYN y class_weight", className="paso-desc")], className="paso"),
        html.Div([html.Div("× 4 optimizadores", className="paso-titulo"),
                  html.Div("Grid, Random, bayesiana (TPE) y genética", className="paso-desc")], className="paso"),
        html.Div([html.Div("= 128 + 28", className="paso-titulo"),
                  html.Div("de clasificación más 28 de regresión: 156 combinaciones", className="paso-desc")],
                 className="paso")], className="flujo")
    visual = html.Div([
        dbc.Row([dbc.Col(kpi("156", "combinaciones", "con el mismo protocolo"), md=4),
                 dbc.Col(kpi(f"{a['n_tareas']}", "tareas", "38 h de cómputo en 4.8 h de reloj"), md=4),
                 dbc.Col(kpi(fmt(a["submuestra"]), "viviendas", "submuestra estratificada"), md=4)], className="g-3"),
        html.Div(cajas, className="mt-3"), grafico(FM.fig_e3_modelos())])
    return diapositiva("7.1", "El pipeline combinatorio del Entregable 3", visual,
                       puntos=["Validación anidada 5 × 3 con kNNDM: el bucle interno elige los hiperparámetros y el "
                               "externo, que nunca los vio, mide.",
                               "Mismo presupuesto para todos: 12 evaluaciones por búsqueda, semilla 42.",
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
                               "(EDA mejorado, 3.21): refuerza el ruido.",
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
                               "tiene 1.2 % de estrato 6, ver 3.22).",
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
    tarjetas = nuevo + [
        hallazgo(f"{T['f1_macro'][B]:.2f}", "F1 macro en test (logística B)",
                 f"{CV['f1_macro (media)'][B]:.2f} en CV espacial; {T['accuracy_±1'][B]:.0%} de aciertos a ±1 estrato."),
        hallazgo(f"{D['moran_estrato']['I']:.2f}", "I de Moran del estrato", "La ciudad está segregada: la ubicación domina.", "#0d366b"),
        hallazgo("0.62 → 0.32", "F1 aleatorio frente a espacial", "Validar mezclando filas habría duplicado el F1.", "#e34948"),
        hallazgo("96.9 %", "Cobertura con la ubicación por manzana", "La ciudad informal entra al modelo sin costo en F1.", "#eb6834"),
        hallazgo("0", "Candidatos que superan a la logística en la ronda 1",
                 "Solo con las variables del base, ninguno lo logra en test.", "#52514e"),
        hallazgo(f"{FP.zonas_conocidas('edificio')['modelos']['XGBoost']['F1_macro']:.2f}", "F1 de XGBoost en zonas conocidas",
                 "Con datos cerca, los árboles sí ganan: el escenario manda.", "#4a3aa7"),
    ]
    v3 = FP.v3_e1()
    if v3:
        q = v3["distancias"]["cuantiles"]
        cvt = v3["cv_vs_test"]["v3: kNNDM sin buffer"]
        tarjetas.append(hallazgo(
            f"{q['CV de la versión 2 (bloques + buffer de 1 km)']['mediana (km)']:.1f} → "
            f"{q['Escenario: zona nueva de 2 km (test → todo train)']['mediana (km)']:.2f} km",
            "Distancia al vecino conocido: validación v2 frente al escenario",
            f"Con kNNDM (v3) la CV anticipa el test: {cvt['F1 macro CV']:.2f} frente a {cvt['F1 macro test']:.2f}.", "#2a78d6"))
    visual = dbc.Row([dbc.Col(t, md=4, className="mb-3") for t in tarjetas], className="g-3")
    return diapositiva("6.1", "Hallazgos principales", visual)


def d_respuesta():
    visual = html.Div([
        html.Div("¿Se puede predecir el estrato con el catastro abierto?", className="pregunta-grande"),
        html.Div("En parte.", className="respuesta-grande"),
        dbc.Row([
            dbc.Col(html.Div([html.H5("Lo que sí revela"), html.Ul([
                html.Li("La zona: dónde está la vivienda y cómo son sus vecinas."),
                html.Li("El tamaño y la tipología: área, baños, régimen de propiedad."),
                html.Li("Errores casi siempre de un estrato, como las viviendas atípicas del DANE.")])], className="caja-si"), md=6),
            dbc.Col(html.Div([html.H5("Lo que no está publicado"), html.Ul([
                html.Li("Las zonas homogéneas con que el DANE estratifica."),
                html.Li("El puntaje de calificación de las edificaciones."),
                html.Li("Los ajustes del comité de estratificación.")])], className="caja-no"), md=6)], className="g-3 mt-2"),
        html.Div([html.H5("¿Y los modelos originales de la revisión?"),
                  html.P("Sí superan al modelo base cuando reciben la vecindad física: el XGBoost geográfico, el "
                         "Rotation Forest y el stacking lo superan con significancia estadística (30 bloques, Wilcoxon "
                         "con Holm p < 0.01, efecto mediano a grande). El Rotation Forest es el más sólido: también en el "
                         "test (+0.099, IC 95 % [+0.024, +0.170]). Pero ninguno supera de forma significativa a los "
                         "modelos convencionales que reciben la misma información: la mejora viene de describir el "
                         "entorno. La ventaja propia del XGBoost geográfico está en las zonas con datos cerca.")],
                 className="caja-si mt-3")])
    return diapositiva("6.2", "Respuesta a la pregunta del proyecto", visual,
                       conclusion="La parte pública del catastro revela sobre todo la zona; lo que distingue a la vivienda "
                                  "dentro de su zona no está publicado.")


def d_limitaciones():
    items = [("Cobertura", "Quedan 10 300 viviendas (3.1 %) sin ubicar, casi todas de estrato 1."),
             ("Pocos bloques de test", "7 bloques, con todo el estrato 6 en uno: intervalos anchos y p mínimo 0.016, así "
                                       "que ninguna diferencia resiste la corrección de Holm."),
             ("Submuestra en el E3", "Las 156 combinaciones usaron 10 001 viviendas; sus hiperparámetros no se trasladan "
                                     "a los modelos que dependen del tamaño de la muestra."),
             ("Candidatos sin probar", "Bagged TAO Trees (sin implementación abierta) y los selectores EFSHB y NSGA-II."),
             ("Presupuesto de búsqueda", "12 evaluaciones: basta para llegar a la meseta, no para que el genético o la "
                                         "multi-fidelidad muestren todo su potencial."),
             ("Etiqueta administrativa", "El estrato puede estar desactualizado o ajustado a mano."),
             ("Calidad del catastro", "Pisos del edificio registrados como de la vivienda; altura casi constante."),
             ("Uso ético", "Predecir el estrato puede discriminar (crédito, seguros): el uso aquí es analítico.")]
    visual = dbc.Row([dbc.Col(html.Div([html.Div(t, className="lim-titulo"), html.P(d, className="lim-texto")],
                                       className="limitacion"), md=6, className="mb-3") for t, d in items], className="g-3")
    return diapositiva("6.3", "Limitaciones", visual)


def d_pasos():
    hechos = html.Div([html.Div("Ya hecho", className="hito-titulo"), html.Ul([
        html.Li("Validación anidada y búsqueda de hiperparámetros con kNNDM para todos los modelos (Entregable 3)."),
        html.Li("Búsqueda del XGBoost geográfico con todos los datos: sube la validación, no el test (6.15)."),
        html.Li("Stacking, Rotation Forest y RF + SVM con vecindad, con pruebas de significancia (6.13 y 6.14)."),
        html.Li("EDA mejorado con los hallazgos de los modelos (3.20 a 3.22).")], className="mb-0")],
        className="hito mb-3")
    pasos = [("1", "Más zonas de evaluación", "Otra ciudad u otra partición: con 7 bloques no se confirman diferencias "
                                              "pequeñas."),
             ("2", "Variedad del entorno", "Medir qué tan mezclado es el barrio para saber cuándo confiar en la vecindad."),
             ("3", "Modelo en dos etapas", "Estrato de la zona y luego vivienda atípica, como el DANE."),
             ("4", "Recalibrar en el pipeline", "Calibración con validación cruzada como último paso: en el test mejoró "
                                                "la calibración y varios F1."),
             ("5", "Zonas conocidas", "Usar el XGBoost geográfico para completar el mapa donde hay datos cerca."),
             ("6", "Bagged TAO Trees", "El único modelo de la revisión que falta probar.")]
    return diapositiva("8.4", "Próximos pasos", html.Div([hechos, flujo(pasos)]))


def d_recursos():
    enlaces = [("Libro del Entregable 1 (versión 3)", LIBRO_E1),
               ("E1 · cap. 15: EDA mejorado", LIBRO_E1 + "notebooks/15_eda_mejorado.html"),
               ("Libro del Entregable 2 (versión 3)", LIBRO_E2),
               ("E2 · cap. 9: todos los modelos originales de la revisión", LIBRO_E2 + "notebooks/09_modelos_revision.html"),
               ("Libro del Entregable 3", LIBRO_E3),
               ("E3 · cap. 11: todos los datos y otros modelos de la revisión", LIBRO_E3 + "notebooks/11_mejoras.html"),
               ("Libro del Entregable 1 (versión 2)", LIBRO_E1_V2), ("Libro del Entregable 1 (versión 1, entregada)", LIBRO_E1_V1),
               ("Libro del Entregable 2 (versión 2)", LIBRO_E2_V2), ("Libro del Entregable 2 (versión 1, entregada)", LIBRO_E2_V1),
               ("Repositorio del Entregable 1", "https://github.com/mcarocantillo/entregable1-ml-barranquilla"),
               ("Repositorio del Entregable 2", "https://github.com/mcarocantillo/entregable2-ml-barranquilla"),
               ("Repositorio del Entregable 3", "https://github.com/mcarocantillo/entregable3-ml-barranquilla")]
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
    {"id": "contexto", "titulo": "Problema y revisión", "diapositivas": [
        ("La pregunta", d_pregunta), ("Cómo asigna el DANE", d_dane), ("Revisión bibliográfica", d_bibliografia),
        ("El vacío en la literatura", d_vacio), ("Objetivo e hipótesis", d_hipotesis), ("Cómo se trabajó", d_ruta)]},
    {"id": "datos", "titulo": "Datos y calidad", "diapositivas": [
        ("Los datos", d_datos), ("Limpieza", d_embudo), ("Ubicación por manzana", d_manzana),
        ("Cobertura v1 → v2", d_cobertura), ("v1 frente a v2", d_v1v2), ("Faltantes e imposibles", d_calidad),
        ("Outliers univariados", d_outliers), ("Dependencia y tamaño efectivo", d_dependencia),
        ("Partición espacial", d_particion)]},
    {"id": "eda", "titulo": "Análisis exploratorio", "diapositivas": [
        ("Variable objetivo", d_objetivo), ("Numéricas", d_numericas), ("Transformación", d_transformar),
        ("Correlaciones", d_correlaciones), ("Pares (densidad)", d_hexbin), ("Numéricas vs estrato", d_violin),
        ("Categóricas vs estrato", d_categoricas), ("Ranking de asociación", d_ranking), ("Multicolinealidad", d_vif),
        ("PCA", d_pca), ("Biplot", d_biplot), ("Outliers multivariados", d_outliers_mv), ("Clustering", d_clustering),
        ("t-SNE", d_tsne), ("Mapa", d_mapa), ("Autocorrelación", d_moran), ("LISA y Gi*", d_lisa),
        ("Escala (MAUP)", d_maup), ("Fuga de datos", d_fuga), ("EDA mejorado: el entorno", d_eda_entorno),
        ("EDA mejorado: fronteras", d_eda_fronteras), ("EDA mejorado: el test", d_eda_test)]},
    {"id": "modelo", "titulo": "Modelo base", "diapositivas": [
        ("Hiperparámetros", d_grid), ("Frente a líneas base", d_comparacion), ("¿Diferencias reales?", d_significancia),
        ("Errores por estrato", d_confusion), ("Probabilidades", d_roc), ("Coeficientes", d_coeficientes),
        ("Importancia", d_permutacion), ("Curva de aprendizaje", d_aprendizaje), ("Residuos espaciales", d_residuos),
        ("Simulador", d_simulador)]},
    {"id": "propuesto", "titulo": "Modelo propuesto y protocolo", "diapositivas": [
        ("El modelo propuesto", d_candidatos), ("Por qué separar zonas", d_protocolo),
        ("Validar a la distancia correcta", d_v3_validacion), ("El modelo base con kNNDM", d_v3_mejoras)]},
    {"id": "mas", "titulo": "Resultados", "diapositivas": [
        ("Ronda 1: resultados", d_avanzados), ("Ronda 1: criterio", d_criterio), ("Ronda 1: errores", d_errores),
        ("Ronda 1: XGBoost geográfico", d_geoxgb), ("Ronda 1: residuos", d_residuos_av),
        ("Ronda 2: vecindad (buffer)", d_vecindad), ("Ronda 2: 15 pliegues", d_vecindad15),
        ("Ronda 2: ¿dónde ayuda?", d_donde), ("Ronda 2: vecindad con kNNDM", d_vecindad_knn),
        ("Zonas conocidas", d_conocidas), ("Dos escenarios", d_escenarios),
        ("¿Supera el modelo nuevo al base?", d_nuevo_vs_base),
        ("Todos los modelos originales", d_originales), ("¿Son significativas?", d_signif_originales),
        ("XGBoost geográfico afinado", d_busqueda_geo)]},
    {"id": "e3", "titulo": "Optimización (E3)", "diapositivas": [
        ("Pipeline combinatorio", d_e3_diseno), ("Optimizadores", d_e3_optimizadores), ("Balanceo", d_e3_balanceo),
        ("Costo computacional", d_e3_computo), ("Test de los finalistas", d_e3_test), ("SHAP", d_e3_shap)]},
    {"id": "conclusiones", "titulo": "Conclusiones", "diapositivas": [
        ("Hallazgos", d_hallazgos), ("Respuesta", d_respuesta), ("Limitaciones", d_limitaciones),
        ("Próximos pasos", d_pasos), ("Para profundizar", d_recursos)]},
]
