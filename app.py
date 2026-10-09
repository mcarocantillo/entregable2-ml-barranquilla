"""Dashboard del proyecto — Estrato socioeconómico de las viviendas de Barranquilla.

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)
Machine Learning — Maestría, Universidad del Norte. Profesor Lihki Rubio.

Tres pestañas: (1) Contexto, (2) EDA y (3) ML models (modelo base, diagnóstico, modelos de la revisión
bibliográfica, variables de vecindad y los dos escenarios de predicción, simulador y conclusiones).
Todas las figuras salen de ``figuras.py``; los datos y resultados, de ``datos/`` (ver ``preparar_datos.py``,
``entrenar_avanzados.py`` y la carpeta ``vecindad/``). Las imágenes de ``assets/`` las genera
``generar_portada.py`` a partir de los datos. Para Render: ``gunicorn app:server``.
"""
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dcc, html

import figuras as F

# El tema Bootstrap (Flatly) va en assets/bootstrap_flatly.css: no depende de un CDN externo.
app = Dash(__name__, title="Estrato socioeconómico — Barranquilla", suppress_callback_exceptions=True)
server = app.server

AUTORES = "María Carolina Cantillo Orozco (200179105) · Juan Camilo Oñoro Araujo (200177329)"
GRAF = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


# ===========================================================================
# Componentes reutilizables
# ===========================================================================
def fmt(n):
    return f"{n:,}".replace(",", " ")


def interpretacion(*parrafos, titulo="Interpretación"):
    """Recuadro de lectura que acompaña a cada figura."""
    return dbc.Alert([html.H6(titulo, className="fw-bold mb-2")] + [html.P(p, className="mb-2") for p in parrafos],
                     color="light", className="interpretacion")


def tarjeta_kpi(valor, etiqueta, nota=None):
    return dbc.Card(dbc.CardBody([html.Div(valor, className="kpi-valor"), html.Div(etiqueta, className="kpi-etiqueta"),
                                  html.Div(nota, className="kpi-nota") if nota else None]), className="h-100 kpi")


def hallazgo(valor, titulo, texto, color="#2a78d6"):
    """Tarjeta de hallazgo: cifra grande, título y una frase."""
    return dbc.Card(dbc.CardBody([html.Div(valor, className="hallazgo-valor", style={"color": color}),
                                  html.Div(titulo, className="hallazgo-titulo"), html.P(texto, className="hallazgo-texto")]),
                    className="h-100 hallazgo", style={"borderTopColor": color})


def grafico(fig, id_=None):
    props = {"figure": fig, "config": GRAF}
    return dcc.Graph(id=id_, **props) if id_ else dcc.Graph(**props)


def tarjeta_grafico(fig, id_=None):
    return dbc.Card(dbc.CardBody(grafico(fig, id_), className="p-2"), className="tarjeta-grafico h-100")


def seccion(titulo, *hijos, sub=None, id_=None):
    extra = {"id": id_} if id_ else {}
    return html.Section([html.H3(titulo, className="mt-4 mb-1"), html.P(sub, className="text-muted") if sub else None,
                         *hijos], className="mb-4", **extra)


def tabla(df, decimales=3):
    df = df.copy()
    for c in df.select_dtypes("number"):
        df[c] = df[c].map(lambda x: f"{x:,.{decimales}f}".replace(",", " "))
    return dbc.Table.from_dataframe(df, striped=True, bordered=False, hover=True, size="sm", className="tabla")


def pasos(lista, clase="flujo"):
    return html.Div([html.Div([html.Div(n, className="paso-num"), html.Div(t, className="paso-titulo"),
                               html.Div(d, className="paso-desc")], className="paso") for n, t, d in lista],
                    className=clase)


def subpestanas(id_, items):
    return dbc.Tabs([dbc.Tab(contenido, label=etq, tab_id=f"{id_}-{i}") for i, (etq, contenido) in enumerate(items)],
                    id=id_, active_tab=f"{id_}-0", className="subpestanas mt-3")


K = F.kpis()
R = F.resultados()
D = F.diagnosticos()
IC = R["ic_modelo_principal"]
VEC = F.hay_vecindad()

# ===========================================================================
# Cabecera (la imagen de fondo es assets/portada.svg, generada con los datos)
# ===========================================================================
cabecera = html.Header(dbc.Container([
    html.Div("Proyecto final · Machine Learning · Maestría, Universidad del Norte · Profesor Lihki Rubio",
             className="hero-kicker"),
    html.H1("¿Qué revela el catastro sobre el estrato de una vivienda?", className="hero-titulo"),
    html.P(["Clasificación supervisada del estrato socioeconómico (1 a 6) de ", html.B(fmt(K["viviendas"])),
            " viviendas de Barranquilla, con sus características físicas y su ubicación."], className="hero-sub"),
    html.Div(AUTORES, className="hero-autores"),
], fluid=True), className="hero")

# ===========================================================================
# Pestaña 1 — Contexto
# ===========================================================================
flujo = pasos([("1", "Descarga", "Catastro abierto (ArcGIS REST), 15-sep-2026"),
               ("2", "Limpieza", "Reglas por fila, antes de partir"),
               ("3", "Partición espacial", "Bloques de 2 km; test reservado"),
               ("4", "EDA en train", "Uni, bi, multivariado y espacial"),
               ("5", "Pipeline", "Imputar, log, escalar, one-hot"),
               ("6", "CV espacial", "Bloques + buffer de 1 km"),
               ("7", "Test", "Una sola evaluación")])

ruta = pasos([("E1", "Entregable 1", "Base de datos, EDA riguroso y modelo base (regresión logística)"),
              ("E2", "Entregable 2", "Dashboard y modelos de la revisión bibliográfica (XGBoost geográfico, "
                                     "Rotation Forest, SVM RBF + TPE)"),
              ("+", "Análisis adicional", "Variables de vecindad física y comparación de dos escenarios de predicción")],
             clase="flujo ruta")

candidatos = pd.DataFrame([
    ["Geographical XGBoost", "Pesos espaciales locales sobre XGBoost", "El estrato cambia por barrios", "Sí, adaptado a clasificación"],
    ["TPE-SVM", "Optimización bayesiana de hiperparámetros", "Espacio de hiperparámetros amplio", "Sí (SVM RBF + TPE)"],
    ["Rotation Forest", "Rotación PCA antes de cada árbol", "Variables de tamaño correlacionadas", "Sí"],
    ["Ensemble RF + SVM", "RF selecciona variables y SVM clasifica", "Variables catastrales correlacionadas", "No"],
    ["Bagged TAO Trees", "Árboles oblicuos optimizados", "Datos tabulares heterogéneos", "No (sin implementación mantenida)"],
], columns=["Modelo de la revisión", "Idea", "Por qué encaja", "¿Se evaluó?"])

pasos_dane = pasos([("1", "Zonas", "Se cruzan las zonas homogéneas físicas (vías, servicios, uso) y "
                                   "geoeconómicas (valor del suelo) del catastro"),
                    ("2", "Estrato de la zona", "Se califica con el puntaje de las edificaciones (estructura, acabados, "
                                                "baño, cocina) y se agrupa con cortes de mínima varianza"),
                    ("3", "Revisión", "La alcaldía y el Comité de Estratificación ajustan zonas"),
                    ("4", "Atípicas", "Una vivienda muy distinta de su estrato sube o baja 1 o 2 estratos (regla del RIC)"),
                    ("5", "Adopción", "Decreto del alcalde; el avalúo catastral no se usa")])

insumos_dane = pd.DataFrame([
    ["Zonas homogéneas físicas y geoeconómicas", "Delimitar la zona", "No (se aproximan con coordenadas y vecindad)"],
    ["Puntaje de calificación de la edificación", "Calificar zona y atípicas", "No"],
    ["Área y cuartos (apartamentos)", "Calificar apartamentos", "Sí"],
    ["Tipo: casa o apartamento", "Se califican por separado", "Sí"],
    ["Revisión de alcaldía y comité", "Ajustes manuales", "No observable"],
    ["Avalúo catastral", "No se usa (Ley 14 de 1983)", "Tampoco se usa"],
], columns=["Insumo del DANE", "Para qué", "¿En los datos del proyecto?"])

tab_contexto = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card([
            html.Img(src=app.get_asset_url("portada.svg"), className="portada",
                     alt="Mapa de Barranquilla de noche: cada luz es una zona con viviendas y el brillo indica el estrato"),
            dbc.CardBody(html.P(["Barranquilla de noche, vista desde el catastro. ", html.B("Cada luz es una zona de "
                                 "unos 165 m"), " con viviendas; su brillo es el estrato medio. Las luces doradas (estratos "
                                 "5 y 6) se concentran en el norte y el sur es casi todo azul tenue (estratos 1 y 2). La "
                                 "imagen se generó con los datos del proyecto."], className="mb-0 text-muted pie-imagen")),
        ], className="tarjeta-portada"), lg=7),
        dbc.Col([
            html.Div("La pregunta", className="etiqueta-seccion"),
            html.H2("¿Se puede predecir el estrato de una vivienda solo con sus datos catastrales?", className="pregunta"),
            html.P(["El estrato define cuánto paga cada hogar por agua, luz y gas, y quién recibe subsidios. Este proyecto "
                    "usa ", html.B("aprendizaje supervisado"), " para estimarlo a partir del área, los baños, las "
                    "habitaciones, los pisos, la antigüedad, el régimen de propiedad y la ubicación de cada vivienda."]),
            dbc.Row([
                dbc.Col(tarjeta_kpi(fmt(K["viviendas"]), "viviendas analizadas"), xs=6, className="mb-3"),
                dbc.Col(tarjeta_kpi(fmt(K["edificios"]), "predios matriz", "edificios, conjuntos o lotes"), xs=6, className="mb-3"),
                dbc.Col(tarjeta_kpi("6", "clases ordinales", "estratos 1 a 6"), xs=6),
                dbc.Col(tarjeta_kpi(f"{K['moran']:.2f}", "I de Moran del estrato", "la ciudad está segregada"), xs=6),
            ], className="g-3"),
        ], lg=5),
    ], className="g-4 mt-1"),

    seccion("¿Qué es el estrato y por qué importa?",
            html.Img(src=app.get_asset_url("escala_estratos.svg"), className="escala-estratos",
                     alt="Escala de estratos: 1 a 3 con subsidio, 4 con tarifa plena, 5 y 6 con contribución"),
            dbc.Row([
                dbc.Col(hallazgo("Ley 142", "Tarifas de servicios públicos",
                                 "Los estratos 1 a 3 reciben subsidios y los estratos 5 y 6 pagan una contribución "
                                 "adicional. Un estrato mal asignado es un subsidio mal dirigido.", "#1baf7a"), md=4),
                dbc.Col(hallazgo("DANE", "Lo asigna cada alcaldía por zonas",
                                 "Con la metodología del DANE: primero se estratifica la zona y luego se corrigen las "
                                 "viviendas atípicas. Se actualiza con poca frecuencia.", "#2a78d6"), md=4),
                dbc.Col(hallazgo("Focalización", "Más allá de las tarifas",
                                 "El estrato se usa para focalizar programas sociales y para planear la ciudad. Un "
                                 "modelo podría señalar estratos incoherentes con las características de la vivienda.",
                                 "#eb6834"), md=4),
            ], className="g-3")),

    seccion("Cómo asigna el DANE el estrato", pasos_dane,
            sub="Metodología de estratificación urbana vigente (DANE, 2015), basada en la información del catastro.",
            *[dbc.Row([
                dbc.Col(html.Div(tabla(insumos_dane), className="tabla-scroll"), lg=7),
                dbc.Col(interpretacion(
                    "El estrato es, ante todo, una propiedad de la zona: todas las viviendas de una zona reciben su "
                    "estrato y solo cambian las que son claramente distintas, uno o dos estratos. Por eso la ubicación "
                    "domina los modelos, el estrato casi no varía dentro de un edificio y los errores son casi siempre "
                    "de un estrato.",
                    "El catastro abierto no publica las zonas homogéneas ni el puntaje de calificación de las "
                    "edificaciones, los dos insumos con los que el DANE calcula el estrato. La pregunta del proyecto es, "
                    "en el fondo, cuánto del estrato se recupera con la parte pública del catastro.",
                    titulo="Qué implica para el proyecto"), lg=5),
            ], className="g-4 mt-1")]),

    seccion("Qué propone el proyecto",
            dbc.Row([
                dbc.Col([
                    html.P(["Es un problema de ", html.B("clasificación multiclase ordinal"), " con dos rasgos que "
                            "condicionan todo el análisis: las clases tienen ", html.B("orden"), " (confundir 1 con 2 es "
                            "menos grave que 1 con 6) y los datos tienen ", html.B("coordenadas"), " (viviendas cercanas "
                            "se parecen). Por eso, además de las métricas habituales, se reportan métricas ordinales (MAE "
                            "ordinal, accuracy ±1, kappa cuadrático) y toda la validación es por bloques espaciales."]),
                    html.P(["La revisión bibliográfica en Scopus (octubre de 2026) no encontró estudios que clasifiquen "
                            "el estrato con variables catastrales: los trabajos cercanos predicen el ", html.B("precio"),
                            " de la vivienda o usan el nivel socioeconómico como predictor. Esa es la ",
                            html.B("brecha"), " del proyecto. La revisión identificó modelos con evidencia de mejora "
                            "frente a clasificadores convencionales, que aquí se ponen a prueba con el mismo protocolo:"]),
                    html.Div(tabla(candidatos), className="tabla-scroll"),
                ], lg=7),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Ficha técnica", className="fw-bold"),
                    html.Ul([
                        html.Li([html.B("Variable objetivo: "), "estrato (1–6), ordinal y desbalanceado"]),
                        html.Li([html.B("Unidad de observación: "), "unidad de vivienda del catastro"]),
                        html.Li([html.B("Tipo de datos: "), "transversal con componente espacial"]),
                        html.Li([html.B("Fuente: "), "catastro abierto, Alcaldía de Barranquilla (15-sep-2026)"]),
                        html.Li([html.B("Predictoras: "), "8 físicas, 3 categóricas y 3 de ubicación"]),
                        html.Li([html.B("Métrica principal: "), "F1 macro (todas las clases pesan igual)"]),
                        html.Li([html.B("Validación: "), "bloques espaciales de 2 km con buffer de 1 km"]),
                        html.Li([html.B("Sin coordenadas: "), f"{K['sin_coord']:.0%} de las viviendas (casi todas "
                                                              "informales); quedan fuera del modelo"]),
                    ], className="mb-0"),
                ]), className="ficha"), lg=5),
            ], className="g-4")),

    seccion("Cómo se trabajó", flujo,
            interpretacion("El orden importa: las reglas que miran una sola fila (por ejemplo, descartar áreas "
                           "imposibles) se aplican antes de partir los datos; todo lo que se aprende de los datos "
                           "(medianas, cuantiles, escalas) se calcula solo con entrenamiento y dentro de un Pipeline. "
                           "El conjunto de prueba son zonas completas de la ciudad que el modelo nunca vio.",
                           titulo="Por qué este orden")),
    seccion("Hoja de ruta del proyecto", ruta),
], fluid=True)

# ===========================================================================
# Pestaña 2 — EDA
# ===========================================================================
selector_particion = dbc.RadioItems(
    id="eda-particion", value="train", inline=True,
    options=[{"label": "Solo entrenamiento (como en el EDA)", "value": "train"},
             {"label": "Todas las viviendas", "value": "todas"}])

eda_objetivo = html.Div([
    dbc.Row([dbc.Col(tarjeta_grafico(F.fig_distribucion_estrato("train"), "eda-dist"), lg=5),
             dbc.Col(tarjeta_grafico(F.fig_gradiente_norte_sur("train"), "eda-ns"), lg=7)], className="g-3 mt-1"),
    interpretacion(
        "En entrenamiento los estratos 1 y 3 son los más frecuentes (alrededor de una cuarta parte cada uno) y el 6 "
        "apenas pasa del 5 %: la razón entre la clase mayor y la menor es 4.6:1. Con este desbalance, un modelo que "
        "siempre predijera la clase más frecuente acertaría cerca de una de cada cuatro viviendas sin aprender nada. Por "
        "eso la métrica principal es el F1 macro: las notas de clase recomiendan la media macro cuando importa cada clase "
        "por igual (sección 9.5).",
        "La ciudad está segregada de sur a norte: las franjas del sur son casi todas de estratos 1 y 2, y el estrato medio "
        "sube hasta cerca de 5 en la penúltima franja. En la franja más al norte vuelve a bajar, porque allí también hay "
        "barrios de estratos bajos: el gradiente no es una línea recta. La ubicación será la información más fuerte del "
        "modelo, y también la razón por la que la validación debe ser espacial."),
    seccion("De los datos descargados a las viviendas analizadas", tarjeta_grafico(F.fig_embudo()),
            interpretacion("La limpieza excluye usos no habitacionales, unidades de menos de 10 m² (parqueaderos o "
                           "depósitos), registros que describen un edificio entero en una fila y filas sin estrato "
                           "residencial. Quedan 332 718 viviendas (87 % de lo descargado).")),
])

eda_numericas = html.Div([
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="eda-num", options=[{"label": v, "value": k} for k, v in F.NUMERICAS.items()],
                             value="total_banios", clearable=False), md=5),
        dbc.Col(dbc.Checklist(id="eda-log", options=[{"label": "escala log(1 + x)", "value": "log"}],
                              value=["log"], switch=True), md=3),
    ], className="filtros my-3"),
    tarjeta_grafico(F.fig_numerica("total_banios"), "eda-num-fig"),
    html.Div(id="eda-num-texto", className="mt-2"),
    html.H5("Resumen estadístico", className="mt-3"),
    html.Div(tabla(F.tabla_resumen_numericas(), 2), id="eda-resumen", className="tabla-scroll"),
    interpretacion(
        "Casi todas las variables físicas son muy asimétricas a la derecha (asimetría > 1) y tienen colas largas: por eso "
        "se transforman con log(1 + x) dentro del Pipeline. Los valores faltantes son muy pocos (< 0.2 %) y se imputan con "
        "la mediana de entrenamiento.",
        "El número de baños es la variable física más asociada al estrato (η² ≈ 0.31), seguida del piso de ubicación y del "
        "área construida. La relación es monótona: a mayor estrato, más baños y más área, pero con mucho traslape entre "
        "estratos vecinos, que es justo donde el modelo se equivocará."),
])

eda_categoricas = html.Div([
    dcc.Dropdown(id="eda-cat", options=[{"label": v, "value": k} for k, v in F.CATEGORICAS.items()],
                 value="condicion_predio", clearable=False, className="filtros my-3", style={"maxWidth": 480}),
    tarjeta_grafico(F.fig_categorica("condicion_predio"), "eda-cat-fig"),
    interpretacion(
        "El régimen de propiedad es la categórica más informativa: los predios informales son casi todos de estratos 1 y 2, "
        "y las unidades en propiedad horizontal (apartamentos) se concentran en estratos medios y altos. El uso y el tipo de "
        "vivienda aportan menos y son redundantes con el régimen.",
        "Las categorías con menos del 1 % de las viviendas se agrupan en una sola dentro del Pipeline (one-hot con "
        "categorías infrecuentes), para no crear columnas casi vacías."),
])

eda_relaciones = html.Div([
    dbc.Row([dbc.Col(tarjeta_grafico(F.fig_correlacion(), "eda-corr"), lg=7),
             dbc.Col(tarjeta_grafico(F.fig_faltantes(), "eda-falt"), lg=5)], className="g-3 mt-1"),
    interpretacion(
        "Se usa Spearman porque las relaciones son monótonas pero no lineales y hay colas largas. Las variables de tamaño "
        "(área, baños, habitaciones) están correlacionadas entre sí, pero ninguna pareja es tan alta como para excluir una "
        "variable (los VIF del Entregable 1 quedaron por debajo del umbral).",
        "El terreno se correlaciona negativamente con el piso de ubicación: en los edificios, cada apartamento registra solo "
        "su cuota del lote, que es más pequeña cuanto más alto es el edificio."),
])

eda_espacio = html.Div([
    dbc.Row([dbc.Col(dcc.Dropdown(id="eda-mapa-var",
                                  options=[{"label": "Estrato medio", "value": "estrato_num"}] +
                                          [{"label": v, "value": k} for k, v in F.NUMERICAS.items()
                                           if k != "dist_centro_km"],
                                  value="estrato_num", clearable=False), md=5)], className="filtros my-3"),
    tarjeta_grafico(F.fig_mapa("estrato_num", "train"), "eda-mapa"),
    dbc.Row([dbc.Col(tarjeta_grafico(F.fig_correlograma()), lg=6),
             dbc.Col(tarjeta_grafico(F.fig_tamano_edificios()), lg=6)], className="g-3 mt-1"),
    interpretacion(
        f"El estrato tiene una autocorrelación espacial extremadamente fuerte (I de Moran = {K['moran']:.2f} entre "
        "edificios vecinos): viviendas cercanas casi siempre comparten estrato. El correlograma muestra que ese parecido cae "
        "por debajo de 0.3 a unos 2.9 km; con eso se fijó un buffer de 1 km entre entrenamiento y validación.",
        "Además, el estrato casi no varía dentro de un edificio o conjunto (ICC = 0.992) y unos pocos conjuntos concentran "
        "muchas viviendas. Las filas no son datos independientes: las 332 718 viviendas equivalen, en precisión, a unas "
        "2 300 observaciones independientes. Por eso la partición mantiene juntos los edificios y las zonas, y la incertidumbre se "
        "mide remuestreando bloques, no filas.",
        "Con «solo entrenamiento» el mapa muestra huecos rectangulares: son los bloques de 2 km reservados para test, que el "
        "EDA no mira. Cerca del 16 % de las viviendas (predios informales, casi todos de estratos 1–2) no tienen coordenadas "
        "y quedan fuera del modelo: las conclusiones aplican a la ciudad formal."),
])

tab_eda = dbc.Container([
    html.Div("Lo esencial del EDA", className="etiqueta-seccion mt-4"),
    dbc.Row([
        dbc.Col(hallazgo("4.6 : 1", "Desbalance moderado", "Entre la clase más y la menos frecuente en entrenamiento. "
                         "La accuracy engaña: se usa F1 macro.", "#2a78d6"), md=6, lg=3),
        dbc.Col(hallazgo("η² = 0.50", "La ubicación domina", "La posición norte-sur es la variable más asociada al "
                         "estrato; entre las físicas, los baños (0.31).", "#0f8a7a"), md=6, lg=3),
        dbc.Col(hallazgo(f"{K['moran']:.2f}", "Ciudad segregada", "I de Moran del estrato: los vecinos casi siempre "
                         "comparten estrato. Validar al azar sería engañoso.", "#4a3aa7"), md=6, lg=3),
        dbc.Col(hallazgo("0.992", "Edificios homogéneos", "ICC del estrato dentro de un edificio o conjunto: el tamaño "
                         "efectivo es de unas 2 300 observaciones.", "#eb6834"), md=6, lg=3),
    ], className="g-3"),
    dbc.Alert(["El EDA se hace ", html.B("solo con el conjunto de entrenamiento"), ", para que ninguna decisión del modelo "
               "se tome mirando el test. Puedes cambiar a todas las viviendas para describir la ciudad completa."],
              color="info", className="mt-4 mb-2"),
    html.Div(selector_particion, className="filtros"),
    subpestanas("eda-sub", [("Variable objetivo", eda_objetivo), ("Variables numéricas", eda_numericas),
                            ("Variables categóricas", eda_categoricas), ("Correlaciones y faltantes", eda_relaciones),
                            ("Componente espacial", eda_espacio)]),
], fluid=True)

# ===========================================================================
# Pestaña 3 — ML models
# ===========================================================================
opciones_modelo = [{"label": F.NOMBRE_CORTO[m], "value": m} for m in F.modelos_disponibles()]
rangos = F.rangos_simulador()
etq_tv = {"tv_1": "Vivienda de Interés Social", "tv_2": "Vivienda de Interés Prioritario",
          "tv_3": "Vivienda que no es de interés social", "tv_4": "No aplica"}


def control_numero(var, etiqueta):
    p1, p50, p99 = rangos[var]
    return dbc.Col([dbc.Label(etiqueta), dbc.Input(id=f"sim-{var}", type="number", value=round(p50, 1),
                                                   min=0, step=1 if p99 < 50 else 5)], md=3)


simulador = dbc.Card(dbc.CardBody([
    html.H5("Simulador: ¿qué estrato predice el modelo?", className="fw-bold"),
    html.P("Cambia las características de una vivienda y su ubicación. Los valores iniciales son las medianas de "
           "entrenamiento. Usa la logística B, el modelo principal en zonas nuevas. Es una herramienta para entender el "
           "modelo, no para asignar estratos reales.", className="text-muted"),
    dbc.Row([control_numero(v, F.NUMERICAS[v]) for v in ["area_construida", "area_catastral_terreno",
                                                          "total_habitaciones", "total_banios"]], className="g-2"),
    dbc.Row([control_numero(v, F.NUMERICAS[v]) for v in ["total_plantas", "planta_ubicacion", "altura",
                                                          "antiguedad"]], className="g-2 mt-1"),
    dbc.Row([dbc.Col([dbc.Label(F.CATEGORICAS[c]), dcc.Dropdown(
        id=f"sim-{c}", value=rangos["moda_categorias"][c], clearable=False,
        options=[{"label": etq_tv.get(o, o.replace("_", " ")), "value": o} for o in rangos["categorias"][c]])], md=4)
        for c in ["condicion_predio", "uso", "tipo_vivienda"]], className="g-2 mt-1"),
    dbc.Row([dbc.Col([dbc.Label("Latitud (sur ↔ norte)"),
                      dcc.Slider(id="sim-lat", min=10.90, max=11.04, step=0.002, value=10.97,
                                 marks={10.9: "10.90", 10.97: "10.97", 11.04: "11.04"})], md=6),
             dbc.Col([dbc.Label("Longitud (oeste ↔ este)"),
                      dcc.Slider(id="sim-lon", min=-74.90, max=-74.76, step=0.002, value=-74.81,
                                 marks={-74.9: "−74.90", -74.83: "−74.83", -74.76: "−74.76"})], md=6)],
            className="g-2 mt-2"),
    dbc.Row([dbc.Col(dcc.Graph(id="sim-fig", config=GRAF), md=6),
             dbc.Col(dcc.Graph(id="sim-mapa", config=GRAF), md=6)]),
]), className="mt-3")

ml_protocolo = html.Div([
    seccion("Un protocolo que respeta el espacio",
            pasos([("1", "Partición por bloques", "Ciudad en celdas de 2 km; el 20 % de los bloques es test"),
                   ("2", "Buffer de 1 km", "Se quita del entrenamiento lo que está pegado a la validación"),
                   ("3", "CV espacial (5 folds)", "Hiperparámetros elegidos solo con la CV"),
                   ("4", "Regla 1-SE", "Entre combinaciones empatadas, la más regularizada"),
                   ("5", "Test una vez", "Zonas que el modelo nunca vio"),
                   ("6", "Bootstrap por bloques", "Intervalos de confianza honestos")]),
            tarjeta_grafico(F.fig_optimismo()),
            interpretacion(
                "Con validación aleatoria, viviendas del mismo edificio y de la misma cuadra caen a la vez en "
                "entrenamiento y validación, y el modelo «reconoce» la zona: el F1 sube a 0.60, el doble que con bloques. "
                "Ese número sería engañoso para predecir en zonas nuevas.",
                "El buffer también importa: sin él el F1 de CV es 0.40, con 1 km baja a 0.30 y con el alcance completo del "
                "correlograma (2.9 km) cae a 0.18, porque se pierde mucho entrenamiento. Se eligió 1 km como compromiso.",
                "Nota crítica de las instrucciones del proyecto: si la accuracy supera 0.80–0.90 hay que sospechar fuga. "
                "Aquí es 0.60 en test y 0.46 en la CV espacial: el problema no es trivial y no hay señales de fuga.")),
])

ml_base = html.Div([
    seccion("Modelo base y líneas base",
            html.P(["Modelo base: ", html.B("regresión logística multinomial"), " con penalización L2, dentro de un "
                    "Pipeline (winsorización → imputación por mediana → log(1 + x) → estandarización; categóricas a "
                    "one-hot). Se comparan dos conjuntos de variables: ", html.B("A"), " solo físicas y ", html.B("B"),
                    " físicas + ubicación (x, y y distancia al centro). Los hiperparámetros (C y pesos de clase) se "
                    "eligieron con validación cruzada espacial y la regla de una desviación estándar: A con C = 0.01 y "
                    "pesos balanceados; B con C = 0.01 sin pesos. Líneas base: tres DummyClassifier y la «moda por zona» "
                    "(la clase más frecuente en la celda de 2 km), que es la versión para clasificación de la media por "
                    "zona."]),
            dcc.Dropdown(id="mod-metrica", value="f1_macro", clearable=False, style={"maxWidth": 420},
                         options=[{"label": v, "value": k} for k, v in F.METRICAS_COMPARABLES.items()],
                         className="filtros mb-2"),
            tarjeta_grafico(F.fig_comparacion("f1_macro"), "mod-comp"),
            html.Div(tabla(F.tabla_metricas()), className="tabla-scroll mt-3"),
            interpretacion(
                "La logística B es el modelo principal: tiene el mayor F1 en la validación cruzada espacial (0.30) y en test "
                "(0.45). Supera con claridad a las Dummy: la diferencia de F1 frente a la mayoritaria es +0.38 (IC 95 % "
                "[0.11, 0.39]) y frente a la estratificada +0.28 (IC [0.03, 0.31]).",
                "Frente a la moda por zona también gana en todas las métricas de test (F1 0.45 frente a 0.33; MAE 0.46 "
                "frente a 0.94), pero la diferencia de F1 no es estadísticamente significativa (+0.12, IC [−0.01, 0.19]): "
                "con solo 7 bloques de test la incertidumbre es grande. Lo mismo pasa entre B y A.",
                "El F1 de test (0.45) es mayor que el de la CV (0.30) por dos razones probables: los 7 bloques de test "
                "resultaron favorables (con muy poco estrato 6) y el modelo final se entrena con 138 239 viviendas, entre 1.9 "
                "y 2.6 veces las de cada fold. La estimación más representativa del desempeño en zonas nuevas es la de la CV "
                "espacial.")),
    seccion("Coeficientes",
            dbc.RadioItems(id="mod-coef", value="B_fisicas+ubicacion", inline=True, className="filtros mb-2",
                           options=[{"label": "Logística B", "value": "B_fisicas+ubicacion"},
                                    {"label": "Logística A", "value": "A_fisicas"}]),
            tarjeta_grafico(F.fig_coeficientes(), "mod-coef-fig"),
            interpretacion(
                "Cada celda es el coeficiente de la variable (estandarizada) en la ecuación de ese estrato: azul empuja hacia "
                "el estrato, rojo lo aleja. Se muestran las variables con mayor diferencia entre el estrato 6 y el 1.",
                "La posición norte-sur domina: moverse al norte aumenta la probabilidad de estratos altos y reduce la de "
                "estratos bajos. Entre las físicas, más baños y más área empujan hacia estratos altos, y los predios "
                "informales hacia el estrato 1. Cautela: las variables están correlacionadas, así que un coeficiente aislado "
                "no es un efecto causal, y la multinomial no usa el orden de los estratos.")),
    seccion("Curva de aprendizaje y residuos espaciales",
            dbc.Row([dbc.Col(tarjeta_grafico(F.fig_curva_aprendizaje()), lg=5),
                     dbc.Col(tarjeta_grafico(F.fig_mapa_residuos()), lg=7)], className="g-3"),
            interpretacion(
                "La curva de validación casi no sube al multiplicar los datos por 20, de 2 700 a 54 000 viviendas (F1 de 0.27 "
                "a 0.30): es poco probable que más filas cambien mucho este modelo. La brecha entre entrenamiento y "
                "validación refleja sobre todo el cambio de zona, no sobreajuste a filas.",
                f"Los residuos siguen fuertemente autocorrelacionados en el espacio (Moran = "
                f"{D['moran_residuos']['B_fisicas+ubicacion']['I']:.2f}): el modelo capta el gradiente norte-sur, pero no "
                "los barrios. Esa fue la motivación para probar modelos con efectos locales y variables de vecindad.")),
])

ml_diagnostico = html.Div([
    dbc.Row([dbc.Col(dcc.Dropdown(id="mod-modelo", options=opciones_modelo,
                                  value="Logística B_fisicas+ubicacion", clearable=False), md=5),
             dbc.Col(dbc.Checklist(id="mod-normalizar", value=["si"], switch=True,
                                   options=[{"label": "matriz en % por fila", "value": "si"}]), md=3)],
            className="filtros my-3"),
    dbc.Row([dbc.Col(tarjeta_grafico(F.fig_confusion(), "mod-conf"), lg=6),
             dbc.Col(tarjeta_grafico(F.fig_por_clase(), "mod-clase"), lg=6)], className="g-3"),
    dbc.Row([dbc.Col(tarjeta_grafico(F.fig_roc(), "mod-roc"), lg=6),
             dbc.Col(tarjeta_grafico(F.fig_calibracion(), "mod-cal"), lg=6)], className="g-3 mt-1"),
    interpretacion(
        "En la logística B los errores son casi siempre entre estratos vecinos (accuracy ±1 = 0.94). Hay dos "
        "desplazamientos hacia el centro de la escala: el estrato 2 casi nunca se predice (recall 0.06; la mayoría se "
        "clasifica como 3) y más de la mitad de los estratos 5 y 6 se predicen como 4.",
        "Las curvas ROC muestran que el modelo sí ordena bien el estrato 2 (AUC alto), pero casi nunca gana el argmax frente "
        "al 3: el problema es de calibración y de umbral, no de falta de señal. La calibración confirma que el estrato 3 "
        "está sobreestimado y el 2 subestimado.",
        "Selecciona otro modelo en la lista para comparar: la Dummy mayoritaria solo predice una clase, la moda por zona "
        "acierta las zonas grandes pero falla en los bordes entre estratos, y XGBoost casi nunca predice los estratos 5 y 6 "
        "en zonas nuevas."),
])


def ml_avanzados():
    if not F.hay_avanzados():
        return html.Div()
    A = R["avanzados"]
    cmp = A["comparaciones_f1"]
    hp = A["hiperparametros"]["XGBoost geográfico"]
    pasos_geo = pasos([("1", "Modelo global", "XGBoost con todas las viviendas de entrenamiento"),
                       ("2", "Anclas", f"{A['anclas_final']} centros de celdas de 2 km con ≥ 300 viviendas"),
                       ("3", "Ventana adaptativa", f"{hp['frac_vecinos']:.0%} del entrenamiento "
                                                   f"({fmt(A['k_usado_final'])} viviendas)"),
                       ("4", "Modelos locales", "un XGBoost por ancla, kernel bicuadrado"),
                       ("5", "Mezcla", f"p = {hp['alpha']}·global + {1 - hp['alpha']:.2f}·local")])
    f = lambda k: float(cmp[k]["diferencia"])  # noqa: E731
    lo = lambda k: float(cmp[k]["IC95_inf"])  # noqa: E731
    hi = lambda k: float(cmp[k]["IC95_sup"])  # noqa: E731
    cv = R["tabla_cv"]["f1_macro (media)"]
    t = R["resultados_test"]
    return html.Div([
        seccion("Modelos de la revisión bibliográfica en zonas nuevas",
                html.P(["Se evaluaron cuatro modelos con el ", html.B("mismo protocolo"), " de la logística: misma "
                        "partición, mismos folds espaciales con buffer, selección de hiperparámetros solo con la CV y una "
                        "única evaluación en test. ", html.B("XGBoost geográfico"), " es la propuesta novedosa: el artículo "
                        "original (Grekousis, 2025) es de regresión y aquí se adaptó a clasificación de estratos mezclando "
                        "probabilidades de un modelo global y de modelos locales. ", html.B("XGBoost"), " es su control. "
                        "Además se probaron ", html.B("Rotation Forest"), " y una ", html.B("SVM con kernel RBF"),
                        " (aproximación de Nyström) con hiperparámetros buscados por optimización bayesiana TPE."]),
                html.H6("Cómo funciona la adaptación del XGBoost geográfico", className="fw-bold mt-2"),
                pasos_geo,
                dcc.Dropdown(id="avz-metrica", value="f1_macro", clearable=False, style={"maxWidth": 420},
                             options=[{"label": v, "value": k} for k, v in F.METRICAS_COMPARABLES.items()],
                             className="filtros mb-2"),
                tarjeta_grafico(F.fig_comparacion_avanzados("f1_macro"), "avz-comp"),
                html.Div(tabla(F.tabla_metricas_avanzados()), className="tabla-scroll mt-3"),
                dbc.Row([dbc.Col(tarjeta_grafico(F.fig_diferencias()), lg=6),
                         dbc.Col(tarjeta_grafico(F.fig_geoxgb_busqueda()), lg=6)], className="g-3 mt-1"),
                interpretacion(
                    "El criterio para cambiar de modelo se fijó antes de mirar el test: un modelo reemplaza a la logística "
                    "B solo si su F1 de CV la supera en más de un error estándar de la diferencia pareada por fold. Ninguno "
                    f"lo cumple. La SVM RBF es la única con ventaja media en CV "
                    f"(+{A['cv_pareado']['SVM RBF (Nyström) + TPE']['media']:.3f}), pero su error estándar es "
                    f"{A['cv_pareado']['SVM RBF (Nyström) + TPE']['se']:.3f} y la ventaja depende de un solo fold.",
                    f"El test lo confirma: la SVM queda en {t['f1_macro']['SVM RBF (Nyström) + TPE']:.3f} (diferencia "
                    f"{f('SVM RBF (Nyström) + TPE − Logística B'):+.3f}, IC [{lo('SVM RBF (Nyström) + TPE − Logística B'):+.3f}, "
                    f"{hi('SVM RBF (Nyström) + TPE − Logística B'):+.3f}]). XGBoost, que en la CV empataba con la logística "
                    f"({cv['XGBoost']:.3f}), en test queda en {t['f1_macro']['XGBoost']:.3f}: los árboles parten el mapa "
                    "en rectángulos y no extrapolan, mientras que la logística prolonga el gradiente sur-norte.",
                    f"El XGBoost geográfico apenas se distingue de su control en zonas nuevas (diferencia de F1 "
                    f"{f('XGBoost geográfico − XGBoost'):+.3f}). Con el buffer de 1 km, sus modelos locales se entrenan con "
                    "los barrios vecinos y no con el de la vivienda. Por eso se midió también en zonas conocidas: ver la "
                    "sub-pestaña «Vecindad y escenarios».")),
        seccion("Residuos espaciales por modelo",
                dcc.Dropdown(id="avz-residuo", value="XGBoost geográfico", clearable=False, style={"maxWidth": 420},
                             options=[{"label": F.NOMBRE_CORTO[m], "value": m}
                                      for m in ["Logística B_fisicas+ubicacion"] + F.AVANZADOS], className="filtros mb-2"),
                tarjeta_grafico(F.fig_mapa_residuos_modelo("XGBoost geográfico"), "avz-residuo-fig"),
                interpretacion("Todos los modelos nuevos dejan los residuos más agrupados que la logística (I de Moran entre "
                               "0.77 y 0.87, frente a 0.70): la información de barrio que falta no está en las variables "
                               "del catastro de la propia vivienda.")),
    ])


def ml_vecindad():
    if not VEC:
        return html.Div("Los resultados de vecindad/ no están disponibles.", className="mt-3")
    return html.Div([
        html.P(["Los residuos agrupados por barrios sugerían que al modelo le falta información del ", html.B("entorno"),
                ". Se hicieron dos experimentos con el mismo protocolo (código en la carpeta ", html.Code("vecindad/"),
                " y capítulo 5 del libro):"], className="mt-3"),
        dbc.Row([
            dbc.Col(hallazgo("Experimento 1", "Variables de vecindad física",
                             "12 variables que describen a las viviendas vecinas a 300 m y 1 km (tamaño, baños, "
                             "antigüedad, % de apartamentos, % informal, densidad). Nunca usan el estrato de los vecinos.",
                             "#0f8a7a"), md=6),
            dbc.Col(hallazgo("Experimento 2", "Dos escenarios de predicción",
                             "Zonas nuevas (bloques no vistos, como en todo el proyecto) frente a zonas conocidas "
                             "(edificios completos ocultos dentro de zonas con datos).", "#4a3aa7"), md=6),
        ], className="g-3"),
        seccion("El resultado central: el mejor modelo depende del escenario",
                tarjeta_grafico(F.fig_escenarios()),
                interpretacion(
                    "En zonas nuevas, la logística B supera a los árboles: XGBoost y su versión geográfica no saben "
                    "extrapolar a zonas sin datos. En zonas conocidas el orden se invierte: los árboles superan a la "
                    "logística por +0.20 de F1.",
                    "La estrella es la regla más simple: votar con el estrato de las 15 viviendas conocidas más cercanas. "
                    "En zonas conocidas empata con el mejor modelo (0.788 frente a 0.784), pero no se puede usar en zonas "
                    "nuevas, donde no hay estratos conocidos alrededor.",
                    "Los dos escenarios se evalúan con conjuntos distintos (test de 7 bloques frente a predicciones fuera de "
                    "muestra en 5 folds por edificio): lo que se compara es el orden de los modelos dentro de cada "
                    "escenario, no los valores entre escenarios.", titulo="Cómo leer esta figura")),
        seccion("Experimento 1 · Zonas nuevas: ¿ayuda ver el barrio?",
                tarjeta_grafico(F.fig_vecindad_diferencias()),
                html.Div(tabla(F.tabla_vecindad()), className="tabla-scroll mt-3"),
                interpretacion(
                    "El modelo principal (fijado antes de ver resultados) es la logística B más las 12 variables. En test "
                    "mejora todas las métricas: F1 de 0.450 a 0.590, MAE ordinal de 0.463 a 0.281, y el recall del estrato 2 "
                    "pasa de 0.06 a 0.68. Pero el estrato 5 empeora (de 0.19 a 0.04).",
                    "En la validación cruzada la mejora es pequeña (+0.025) y no supera un error estándar: por el criterio "
                    "fijado de antemano, no reemplaza a B. Con una variante estricta (vecindad calculada solo dentro de cada "
                    "partición) la mejora en test se mantiene (F1 0.568), y al quitar un bloque a la vez sigue siendo "
                    "positiva en los siete casos.")),
        dbc.Row([dbc.Col(tarjeta_grafico(F.fig_vecindad_folds15()), lg=6),
                 dbc.Col(tarjeta_grafico(F.fig_vecindad_entorno()), lg=6)], className="g-3"),
        interpretacion(
            "Con 15 folds la vecindad mejora en solo 7: la diferencia media es +0.006 (error estándar 0.021). Los dos modelos "
            "con vecindad ganan y pierden en los mismos folds, lo que apunta a una propiedad de las zonas.",
            "El análisis por entorno (exploratorio) explica parte: la vecindad «suaviza» la predicción hacia lo típico del "
            "barrio. Ayuda a casi todas las viviendas, que se parecen a su entorno, y perjudica mucho a las pocas que viven en "
            "entornos muy mezclados. Es el único patrón que se repite igual en la validación cruzada y en test.",
            "Veredicto: resultado mixto. La vecindad es una línea prometedora, pero no una mejora demostrada en zonas nuevas.",
            titulo="Por qué el resultado es mixto"),
        seccion("Experimento 2 · Zonas conocidas: el escenario natural del XGBoost geográfico",
                dbc.Row([dbc.Col(tarjeta_grafico(F.fig_zonas_conocidas()), lg=5),
                         dbc.Col(html.Div(tabla(F.tabla_comparaciones_conocidas()), className="tabla-scroll"), lg=7)],
                        className="g-3"),
                interpretacion(
                    "En zonas conocidas, el XGBoost geográfico supera a su control en los cinco folds (+0.010 de F1, IC "
                    "[+0.007, +0.024]): en su escenario natural, la adaptación propuesta sí aporta, aunque poco. En zonas "
                    "nuevas su ganancia era de +0.003.",
                    "La vecindad física también ayuda a la logística aquí (+0.111), pero incluso con el estrato de los "
                    "vecinos la logística (0.718) queda muy por debajo del voto de los vecinos (0.788): un modelo lineal no "
                    "aprovecha bien la información local.",
                    "Implicación práctica: para estimar el estrato en una zona ya estratificada basta una regla de vecinos; "
                    "los modelos con variables catastrales son necesarios donde no hay estratos conocidos alrededor.")),
        dbc.Alert(["Estos experimentos se corrieron con una réplica del protocolo hecha con numpy y scipy (en el equipo "
                   "donde se corrieron, Windows bloquea scikit-learn). La réplica reproduce el modelo B guardado con "
                   "diferencias de una milésima como máximo (F1 de test 0.4500)."], color="secondary", className="mt-2"),
    ])


ml_conclusiones = html.Div([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Lo que aprendimos", className="fw-bold"),
            html.Ul([
                html.Li([html.B("El estrato se puede predecir con el catastro, pero solo en parte: "),
                         "F1 macro 0.30 en CV espacial y 0.45 en test, con errores casi siempre de un estrato."]),
                html.Li([html.B("La ubicación es la información dominante: "),
                         "Barranquilla está segregada de sur a norte (I de Moran = 0.91)."]),
                html.Li([html.B("La estructura manda sobre la cantidad de filas: "),
                         "validar al azar habría duplicado el F1 (0.60 frente a 0.30)."]),
                html.Li([html.B("El mejor modelo depende del escenario: "),
                         "en zonas nuevas gana la logística; en zonas conocidas, los árboles."]),
                html.Li([html.B("La propuesta novedosa aporta en su escenario: "),
                         "el XGBoost geográfico supera a su control en zonas conocidas, aunque empata con votar con los "
                         "vecinos."]),
            ], className="mb-0"),
        ]), className="h-100 conclusion"), lg=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Limitaciones", className="fw-bold"),
            html.Ul([
                html.Li("El 16 % de las viviendas (predios informales) no tiene coordenadas: las conclusiones aplican a "
                        "la ciudad formal."),
                html.Li("Solo 7 bloques de test: intervalos muy anchos; la CV espacial es la cifra más representativa."),
                html.Li("Buffer parcial (1 km frente a un alcance de 2.9 km): la validación sigue algo optimista."),
                html.Li("El estrato es una etiqueta administrativa con información del entorno que el catastro no tiene."),
                html.Li("Riesgo ético: un modelo de estrato podría usarse para discriminar; su propósito aquí es "
                        "analítico. Las coordenadas se publican redondeadas y sin número predial."),
            ], className="mb-0"),
        ]), className="h-100 conclusion"), lg=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Próximos pasos", className="fw-bold"),
            html.Ul([
                html.Li("Medir la variedad física del entorno, para que el modelo sepa cuándo confiar en la vecindad."),
                html.Li("Combinar la logística y la SVM RBF (stacking): aciertan estratos distintos en la parte alta."),
                html.Li("Regresión logística ordinal y recalibración de umbrales (el estrato 2 ordena bien pero pierde "
                        "el argmax)."),
                html.Li("Más bloques de evaluación para distinguir mejoras pequeñas."),
            ], className="mb-0"),
        ]), className="h-100 conclusion"), lg=4),
    ], className="g-3 mt-3"),
])

tab_modelos = dbc.Container([
    dbc.Row([
        dbc.Col(tarjeta_kpi(f"{IC['valor en test']['f1_macro']:.2f}", "F1 macro en test (logística B)",
                            f"IC 95 % por bloques: [{IC['IC95_inf']['f1_macro']:.2f}, {IC['IC95_sup']['f1_macro']:.2f}]"),
                md=6, lg=3),
        dbc.Col(tarjeta_kpi(f"{R['resultados_test']['accuracy_±1']['Logística B_fisicas+ubicacion']:.2f}",
                            "accuracy ±1 estrato", "casi todos los errores son entre vecinos"), md=6, lg=3),
        dbc.Col(tarjeta_kpi(f"{IC['valor en test']['kappa_cuadrático']:.2f}", "kappa cuadrático", "acuerdo ordinal alto"),
                md=6, lg=3),
        dbc.Col(tarjeta_kpi(f"{F.vecindad('zonas_conocidas')['modelos']['XGBoost geográfico']['F1_macro']:.2f}"
                            if VEC else "—", "F1 del XGBoost geográfico", "en zonas conocidas"), md=6, lg=3),
    ], className="g-3 mt-3"),
    html.Div("La ruta de modelado", className="etiqueta-seccion mt-4"),
    pasos([("1", "Líneas base", "Dummy y moda por zona"),
           ("2", "Logística A y B", "Modelo base con CV espacial"),
           ("3", "Revisión bibliográfica", "XGBoost geográfico, XGBoost, Rotation Forest, SVM"),
           ("4", "Vecindad física", "El modelo ve el barrio"),
           ("5", "Dos escenarios", "Zonas nuevas frente a zonas conocidas")]),
    subpestanas("ml-sub", [("Protocolo", ml_protocolo), ("Modelo base", ml_base), ("Diagnóstico", ml_diagnostico),
                           ("Modelos de la revisión", ml_avanzados()), ("Vecindad y escenarios", ml_vecindad()),
                           ("Simulador", simulador), ("Conclusiones", ml_conclusiones)]),
], fluid=True)

# ===========================================================================
# Layout
# ===========================================================================
app.layout = html.Div([
    cabecera,
    dbc.Container(dbc.Tabs([
        dbc.Tab(tab_contexto, label="Contexto", tab_id="t1"),
        dbc.Tab(tab_eda, label="EDA", tab_id="t2"),
        dbc.Tab(tab_modelos, label="ML models", tab_id="t3"),
    ], active_tab="t1", className="pestanas mt-3"), fluid=True),
    html.Footer(dbc.Container(html.Small(
        "Datos: catastro abierto de la Alcaldía de Barranquilla (descarga del 15-sep-2026). Coordenadas redondeadas a "
        "~110 m y sin número predial. Semilla 42. Informe completo en el Jupyter Book del proyecto. " + AUTORES),
        fluid=True), className="pie"),
])


# ===========================================================================
# Callbacks
# ===========================================================================
@app.callback(Output("eda-dist", "figure"), Output("eda-ns", "figure"), Output("eda-corr", "figure"),
              Output("eda-falt", "figure"), Output("eda-resumen", "children"), Input("eda-particion", "value"))
def actualizar_eda_general(part):
    return (F.fig_distribucion_estrato(part), F.fig_gradiente_norte_sur(part), F.fig_correlacion(part),
            F.fig_faltantes(part), tabla(F.tabla_resumen_numericas(part), 2))


@app.callback(Output("eda-num-fig", "figure"), Output("eda-num-texto", "children"),
              Input("eda-num", "value"), Input("eda-log", "value"), Input("eda-particion", "value"))
def actualizar_numerica(var, log, part):
    ef = F.efecto_numerica(var, part)
    fuerza = "fuerte" if ef["eta2_H"] >= 0.14 else "moderada" if ef["eta2_H"] >= 0.06 else "débil"
    texto = html.P([f"Kruskal-Wallis: H = {ef['H']:,.0f}, p < 0.001 (n = {ef['n']:,}). ".replace(",", " "),
                    "Con tantas viviendas cualquier diferencia es significativa; lo que importa es el ",
                    html.B("tamaño del efecto"), f": η² = {ef['eta2_H']:.3f}, asociación {fuerza} con el estrato."],
                   className="text-muted")
    return F.fig_numerica(var, "log" in (log or []), part), texto


@app.callback(Output("eda-cat-fig", "figure"), Input("eda-cat", "value"), Input("eda-particion", "value"))
def actualizar_categorica(var, part):
    return F.fig_categorica(var, part)


@app.callback(Output("eda-mapa", "figure"), Input("eda-mapa-var", "value"), Input("eda-particion", "value"))
def actualizar_mapa(var, part):
    return F.fig_mapa(var, part)


@app.callback(Output("mod-comp", "figure"), Input("mod-metrica", "value"))
def actualizar_comparacion(metrica):
    return F.fig_comparacion(metrica)


@app.callback(Output("mod-conf", "figure"), Output("mod-clase", "figure"), Output("mod-roc", "figure"),
              Output("mod-cal", "figure"), Input("mod-modelo", "value"), Input("mod-normalizar", "value"))
def actualizar_diagnostico(modelo, norm):
    return (F.fig_confusion(modelo, "si" in (norm or [])), F.fig_por_clase(modelo), F.fig_roc(modelo),
            F.fig_calibracion(modelo))


@app.callback(Output("mod-coef-fig", "figure"), Input("mod-coef", "value"))
def actualizar_coeficientes(conj):
    return F.fig_coeficientes(conj)


NUM_SIM = ["area_construida", "area_catastral_terreno", "total_habitaciones", "total_banios", "total_plantas",
           "planta_ubicacion", "altura", "antiguedad"]
CAT_SIM = ["condicion_predio", "uso", "tipo_vivienda"]


@app.callback(Output("sim-fig", "figure"), Output("sim-mapa", "figure"),
              [Input(f"sim-{v}", "value") for v in NUM_SIM + CAT_SIM] + [Input("sim-lat", "value"),
                                                                         Input("sim-lon", "value")])
def simular(*valores):
    entrada = dict(zip(NUM_SIM + CAT_SIM, valores[:-2]))
    for v in NUM_SIM:
        entrada[v] = np.nan if entrada[v] is None else float(entrada[v])
    lat, lon = valores[-2:]
    probas = F.predecir(dict(entrada, lat=lat, lon=lon))
    mapa = F.fig_mapa("estrato_num", "todas")
    mapa.add_scattermap(lat=[lat], lon=[lon], mode="markers", marker=dict(size=18, color="#e34948"),
                        hovertemplate="vivienda simulada<extra></extra>", showlegend=False)
    mapa.update_layout(height=380, title="Ubicación simulada (punto rojo)", showlegend=False,
                       map=dict(center=dict(lat=lat, lon=lon), zoom=11))
    return F.fig_simulador(probas), mapa


@app.callback(Output("avz-comp", "figure"), Input("avz-metrica", "value"))
def actualizar_comparacion_avanzados(metrica):
    return F.fig_comparacion_avanzados(metrica)


@app.callback(Output("avz-residuo-fig", "figure"), Input("avz-residuo", "value"))
def actualizar_residuos_avanzados(modelo):
    return F.fig_mapa_residuos_modelo(modelo)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
