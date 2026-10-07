"""Dashboard del Entregable 2 — Estrato socioeconómico de las viviendas de Barranquilla.

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)
Machine Learning — Maestría, Universidad del Norte. Profesor Lihki Rubio.

Tres pestañas: (1) contexto del problema, (2) EDA y (3) modelos base
(regresión logística frente a líneas base triviales). Todas las figuras salen
de ``figuras.py``; los datos y resultados, de ``datos/`` (ver
``preparar_datos.py``). Para Render: ``gunicorn app:server``.
"""
import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash, Input, Output, State, dcc, html

import figuras as F

# El tema Bootstrap (Flatly) va en assets/bootstrap_flatly.css: no depende de un CDN externo.
app = Dash(__name__,
           title="Estrato socioeconómico — Barranquilla", suppress_callback_exceptions=True)
server = app.server

AUTORES = "María Carolina Cantillo Orozco (200179105) · Juan Camilo Oñoro Araujo (200177329)"
GRAF = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def fmt(n):
    return f"{n:,}".replace(",", " ")


def interpretacion(*parrafos, titulo="Interpretación"):
    """Recuadro de lectura que acompaña a cada figura."""
    return dbc.Alert([html.H6(titulo, className="fw-bold mb-2")] + [html.P(p, className="mb-2") for p in parrafos],
                     color="light", className="interpretacion")


def tarjeta_kpi(valor, etiqueta, nota=None):
    return dbc.Card(dbc.CardBody([html.Div(valor, className="kpi-valor"), html.Div(etiqueta, className="kpi-etiqueta"),
                                  html.Div(nota, className="kpi-nota") if nota else None]), className="h-100 kpi")


def grafico(fig, id_=None):
    props = {"figure": fig, "config": GRAF}
    return dcc.Graph(id=id_, **props) if id_ else dcc.Graph(**props)


def seccion(titulo, *hijos, sub=None):
    return html.Section([html.H3(titulo, className="mt-4 mb-1"), html.P(sub, className="text-muted") if sub else None,
                         *hijos], className="mb-4")


def tabla(df, decimales=3):
    df = df.copy()
    for c in df.select_dtypes("number"):
        df[c] = df[c].map(lambda x: f"{x:,.{decimales}f}".replace(",", " "))
    return dbc.Table.from_dataframe(df, striped=True, bordered=False, hover=True, size="sm", className="tabla")


K = F.kpis()
R = F.resultados()
D = F.diagnosticos()
IC = R["ic_modelo_principal"]

# ===========================================================================
# Pestaña 1 — Contexto del problema
# ===========================================================================
pasos = [("1", "Descarga", "Catastro abierto (ArcGIS REST), 15-sep-2026"),
         ("2", "Limpieza", "Reglas por fila, antes de partir"),
         ("3", "Partición espacial", "Bloques de 2 km; test reservado"),
         ("4", "EDA en train", "Uni, bi, multivariado y espacial"),
         ("5", "Pipeline", "Imputar, log, escalar, one-hot"),
         ("6", "CV espacial", "Bloques + buffer de 1 km"),
         ("7", "Test", "Una sola evaluación")]
flujo = html.Div([html.Div([html.Div(n, className="paso-num"), html.Div(t, className="paso-titulo"),
                            html.Div(d, className="paso-desc")], className="paso") for n, t, d in pasos],
                 className="flujo")

tab_contexto = dbc.Container([
    seccion("¿Se puede predecir el estrato de una vivienda con sus datos catastrales?",
            dbc.Row([
                dbc.Col([
                    html.P(["En Colombia, el ", html.B("estrato socioeconómico"), " (1 a 6) clasifica los inmuebles "
                            "residenciales para cobrar tarifas diferenciadas de servicios públicos: los estratos 1 a 3 "
                            "reciben subsidios y los estratos 5 y 6 pagan una contribución (Ley 142 de 1994). Lo asigna "
                            "cada alcaldía por manzana, con una metodología que mira la vivienda y su entorno."]),
                    html.P(["Este proyecto plantea un problema de ", html.B("clasificación multiclase ordinal"), ": "
                            "predecir el estrato de cada unidad de vivienda de Barranquilla a partir de sus "
                            "características físicas en el catastro (área, baños, habitaciones, pisos, antigüedad, "
                            "régimen de propiedad) y de su ubicación. Un modelo así serviría para detectar viviendas "
                            "con un estrato inconsistente con sus características, apoyar la actualización de la "
                            "estratificación y estimarlo donde falta."]),
                    html.P(["La revisión bibliográfica en Scopus (octubre de 2026) no encontró estudios que clasifiquen "
                            "el estrato directamente con variables catastrales: los trabajos cercanos predicen precios "
                            "de vivienda o usan el nivel socioeconómico como predictor. Esa es la ", html.B("brecha"),
                            " que motiva el proyecto. Este entregable fija la referencia: un modelo base lineal "
                            "comparado con líneas base triviales. Los modelos novedosos identificados en la revisión "
                            "(por ejemplo, XGBoost geográfico o árboles TAO) se evaluarán contra esta referencia."]),
                ], md=7),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Ficha del problema", className="fw-bold"),
                    html.Ul([
                        html.Li([html.B("Objetivo: "), "estrato (1–6), ordinal y desbalanceado"]),
                        html.Li([html.B("Unidad de observación: "), "unidad de vivienda del catastro"]),
                        html.Li([html.B("Tipo de datos: "), "transversal con componente espacial (foto del catastro)"]),
                        html.Li([html.B("Fuente: "), "datos abiertos de catastro, Alcaldía de Barranquilla"]),
                        html.Li([html.B("Ruta del enunciado: "), "A (clasificación) + componente espacial"]),
                        html.Li([html.B("Métrica principal: "), "F1 macro (todas las clases pesan igual)"]),
                    ], className="mb-0"),
                ])), md=5),
            ])),
    dbc.Row([
        dbc.Col(tarjeta_kpi(fmt(K["viviendas"]), "viviendas analizadas"), md=3),
        dbc.Col(tarjeta_kpi(fmt(K["edificios"]), "edificios distintos"), md=3),
        dbc.Col(tarjeta_kpi(f"{K['moran']:.2f}", "I de Moran del estrato", "autocorrelación espacial muy fuerte"), md=3),
        dbc.Col(tarjeta_kpi(f"{K['f1_test']:.2f}", "F1 macro del modelo en test",
                            f"IC 95 %: [{IC['IC95_inf']['f1_macro']:.2f}, {IC['IC95_sup']['f1_macro']:.2f}]"), md=3),
    ], className="g-3 my-2"),
    seccion("Cómo se trabajó", flujo,
            interpretacion("El orden importa: las reglas que miran una sola fila (por ejemplo, descartar áreas "
                           "imposibles) se aplican antes de partir los datos; todo lo que se aprende de los datos "
                           "(medianas, cuantiles, escalas) se calcula solo con entrenamiento y dentro de un Pipeline. "
                           "El conjunto de prueba son zonas completas de la ciudad que el modelo nunca vio.",
                           titulo="Por qué este orden")),
    dbc.Row([
        dbc.Col(grafico(F.fig_embudo()), md=7),
        dbc.Col(grafico(F.fig_distribucion_estrato()), md=5),
    ]),
    interpretacion(
        "La limpieza excluye usos no habitacionales, unidades de menos de 10 m² (parqueaderos o depósitos), registros "
        "que describen un edificio entero en una fila y filas sin estrato residencial. Quedan 332 718 viviendas "
        "(87 % de lo descargado).",
        "El estrato está desbalanceado: el estrato 1 es la clase más frecuente y el 6 la más escasa. Por eso la métrica "
        "principal es el F1 macro y no la accuracy, que premiaría predecir solo las clases grandes."),
], fluid=True)

# ===========================================================================
# Pestaña 2 — EDA
# ===========================================================================
selector_particion = dbc.RadioItems(
    id="eda-particion", value="train", inline=True,
    options=[{"label": "Solo entrenamiento (como en el EDA)", "value": "train"},
             {"label": "Todas las viviendas", "value": "todas"}])

tab_eda = dbc.Container([
    dbc.Alert(["El EDA se hace ", html.B("solo con el conjunto de entrenamiento"), ", para que ninguna decisión del "
               "modelo se tome mirando el test. Puedes cambiar a todas las viviendas para describir la ciudad completa."],
              color="info", className="mt-3"),
    html.Div(selector_particion, className="mb-3 filtros"),

    seccion("1. Variable objetivo",
            dbc.Row([dbc.Col(grafico(F.fig_distribucion_estrato("train"), "eda-dist"), md=5),
                     dbc.Col(grafico(F.fig_gradiente_norte_sur("train"), "eda-ns"), md=7)]),
            interpretacion(
                "En entrenamiento los estratos 1 y 3 son los más frecuentes (alrededor de una cuarta parte cada uno) y el 6 "
                "apenas pasa del 5 %: la razón entre la clase mayor y la menor es 4.6:1. Con este desbalance, un modelo que "
                "siempre predijera la clase más frecuente acertaría cerca de una de cada cuatro viviendas sin aprender nada.",
                "La ciudad está segregada de sur a norte: las franjas del sur son casi todas de estratos 1 y 2, y el "
                "estrato medio sube hasta cerca de 5 en la penúltima franja (F9), donde dominan los estratos 4 a 6. En la "
                "franja más al norte vuelve a bajar (3.3), porque allí también hay barrios de estratos bajos: el gradiente "
                "no es una línea recta. La ubicación será la información más fuerte del modelo, y también la razón por la "
                "que la validación debe ser espacial.")),

    seccion("2. Variables numéricas",
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="eda-num", options=[{"label": v, "value": k} for k, v in F.NUMERICAS.items()],
                                     value="total_banios", clearable=False), md=5),
                dbc.Col(dbc.Checklist(id="eda-log", options=[{"label": "escala log(1 + x)", "value": "log"}],
                                      value=["log"], switch=True), md=3),
            ], className="filtros mb-2"),
            grafico(F.fig_numerica("total_banios"), "eda-num-fig"),
            html.Div(id="eda-num-texto"),
            html.H5("Resumen estadístico (train)", className="mt-3"),
            html.Div(tabla(F.tabla_resumen_numericas(), 2), id="eda-resumen"),
            interpretacion(
                "Casi todas las variables físicas son muy asimétricas a la derecha (asimetría > 1) y tienen colas largas: "
                "por eso se transforman con log(1 + x) dentro del Pipeline. Los valores faltantes son muy pocos (< 0.2 %) "
                "y se imputan con la mediana de entrenamiento.",
                "El número de baños es la variable física más asociada al estrato (η² ≈ 0.31), seguida del piso de "
                "ubicación y del área construida. La relación es monótona: a mayor estrato, más baños y más área, pero "
                "con mucho traslape entre estratos vecinos, que es justo donde el modelo se equivocará.")),

    seccion("3. Variables categóricas",
            dcc.Dropdown(id="eda-cat", options=[{"label": v, "value": k} for k, v in F.CATEGORICAS.items()],
                         value="condicion_predio", clearable=False, className="filtros mb-2", style={"maxWidth": 480}),
            grafico(F.fig_categorica("condicion_predio"), "eda-cat-fig"),
            interpretacion(
                "El régimen de propiedad es la categórica más informativa: los predios informales son casi todos de "
                "estratos 1 y 2, y las unidades en propiedad horizontal (apartamentos) se concentran en estratos medios y "
                "altos. El uso y el tipo de vivienda aportan menos y son redundantes con el régimen.",
                "Las categorías con menos del 1 % de las viviendas se agrupan en una sola dentro del Pipeline "
                "(one-hot con categorías infrecuentes), para no crear columnas casi vacías.")),

    seccion("4. Correlaciones y valores faltantes",
            dbc.Row([dbc.Col(grafico(F.fig_correlacion(), "eda-corr"), md=7),
                     dbc.Col(grafico(F.fig_faltantes(), "eda-falt"), md=5)]),
            interpretacion(
                "Se usa Spearman porque las relaciones son monótonas pero no lineales y hay colas largas. Las variables "
                "de tamaño (área, baños, habitaciones) están correlacionadas entre sí, pero ninguna pareja es tan alta "
                "como para excluir una variable (los VIF del Entregable 1 quedaron por debajo del umbral).",
                "El terreno se correlaciona negativamente con el piso de ubicación: en los edificios, cada apartamento "
                "registra solo su cuota del lote, que es más pequeña cuanto más alto es el edificio.")),

    seccion("5. Componente espacial",
            dbc.Row([dbc.Col(dcc.Dropdown(id="eda-mapa-var",
                                          options=[{"label": "Estrato medio", "value": "estrato_num"}] +
                                                  [{"label": v, "value": k} for k, v in F.NUMERICAS.items()
                                                   if k != "dist_centro_km"],
                                          value="estrato_num", clearable=False), md=5)], className="filtros mb-2"),
            grafico(F.fig_mapa("estrato_num", "train"), "eda-mapa"),
            dbc.Row([dbc.Col(grafico(F.fig_correlograma()), md=6), dbc.Col(grafico(F.fig_tamano_edificios()), md=6)]),
            interpretacion(
                f"El estrato tiene una autocorrelación espacial extremadamente fuerte (I de Moran = {K['moran']:.2f} entre "
                "edificios vecinos): viviendas cercanas casi siempre comparten estrato. El correlograma muestra que ese "
                "parecido cae por debajo de 0.3 a unos 2.9 km; con eso se fijó un buffer de 1 km entre entrenamiento y "
                "validación.",
                "Además, el estrato casi no varía dentro de un edificio (ICC = 0.991) y unas pocas torres concentran "
                "muchas viviendas. Las filas no son datos independientes: por eso la partición mantiene juntos los "
                "edificios y las zonas, y la incertidumbre se mide remuestreando bloques, no filas.",
                "Con «solo entrenamiento» el mapa muestra huecos rectangulares: son los bloques de 2 km reservados para "
                "test, que el EDA no mira.",
                "Cerca del 16 % de las viviendas (predios informales, casi todos de estratos 1–2) no tienen coordenadas y "
                "quedan fuera del modelo: las conclusiones aplican a la ciudad formal.")),
], fluid=True)

# ===========================================================================
# Pestaña 3 — Modelos base
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
           "entrenamiento. Es una herramienta para entender el modelo, no para asignar estratos reales.",
           className="text-muted"),
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
]), className="mt-2")


# ---------------------------------------------------------------------------
# Sección 7 — Modelos de la revisión bibliográfica (solo si se ejecutó entrenar_avanzados.py)
# ---------------------------------------------------------------------------
def seccion_avanzados():
    if not F.hay_avanzados():
        return html.Div()
    A = R["avanzados"]
    cmp = A["comparaciones_f1"]
    pasos_geo = [("1", "Modelo global", "XGBoost con todas las viviendas de entrenamiento"),
                 ("2", "Anclas", f"{A['anclas_final']} centros de celdas de 2 km con ≥ 300 viviendas"),
                 ("3", "Ventana adaptativa", f"{A['hiperparametros']['XGBoost geográfico']['frac_vecinos']:.0%} del "
                                             f"entrenamiento ({A['k_usado_final']:,} viviendas)".replace(",", " ")),
                 ("4", "Modelos locales", "un XGBoost por ancla, kernel bicuadrado"),
                 ("5", "Mezcla", f"p = {A['hiperparametros']['XGBoost geográfico']['alpha']}·global + "
                                 f"{1 - A['hiperparametros']['XGBoost geográfico']['alpha']:.2f}·local")]
    flujo_geo = html.Div([html.Div([html.Div(n, className="paso-num"), html.Div(t, className="paso-titulo"),
                                    html.Div(d, className="paso-desc")], className="paso") for n, t, d in pasos_geo],
                         className="flujo")
    f = lambda k: float(cmp[k]["diferencia"])  # noqa: E731
    lo = lambda k: float(cmp[k]["IC95_inf"])  # noqa: E731
    hi = lambda k: float(cmp[k]["IC95_sup"])  # noqa: E731
    cv = R["tabla_cv"]["f1_macro (media)"]
    sd = R["tabla_cv"]["F1 macro (desv. entre folds)"]
    t = R["resultados_test"]
    return seccion(
        "7. Más allá del modelo base: modelos de la revisión bibliográfica",
        html.P(["Se evaluaron cuatro modelos con el ", html.B("mismo protocolo"), " de la logística: misma partición, "
                "mismos folds espaciales con buffer, selección de hiperparámetros solo con la CV y una única "
                "evaluación en test. ", html.B("XGBoost geográfico"), " es la propuesta novedosa: el artículo original "
                "(Grekousis, 2025) es de regresión y aquí se adaptó a clasificación de estratos mezclando "
                "probabilidades de un modelo global y de modelos locales. ", html.B("XGBoost"), " es su control (el "
                "mismo algoritmo sin la parte geográfica). Además se probaron ", html.B("Rotation Forest"),
                " y una ", html.B("SVM con kernel RBF"), " (aproximación de Nyström) con hiperparámetros buscados "
                "por optimización bayesiana TPE."]),
        flujo_geo,
        dcc.Dropdown(id="avz-metrica", value="f1_macro", clearable=False, style={"maxWidth": 420},
                     options=[{"label": v, "value": k} for k, v in F.METRICAS_COMPARABLES.items()],
                     className="filtros mb-2"),
        grafico(F.fig_comparacion_avanzados("f1_macro"), "avz-comp"),
        html.Div(tabla(F.tabla_metricas_avanzados()), className="tabla-scroll"),
        dbc.Row([dbc.Col(grafico(F.fig_diferencias()), md=6), dbc.Col(grafico(F.fig_geoxgb_busqueda()), md=6)]),
        interpretacion(
            "El criterio para cambiar de modelo se fijó antes de mirar el test: un modelo reemplaza a la logística B solo "
            "si su F1 de CV la supera en más de un error estándar de la diferencia pareada por fold. Ninguno lo cumple. "
            f"La SVM RBF es la única con ventaja media en CV (+{A['cv_pareado']['SVM RBF (Nyström) + TPE']['media']:.3f}), "
            f"pero su error estándar es {A['cv_pareado']['SVM RBF (Nyström) + TPE']['se']:.3f} y la ventaja depende de "
            "un solo fold. La logística B sigue siendo el modelo principal.",
            f"El test lo confirma: la SVM queda en {t['f1_macro']['SVM RBF (Nyström) + TPE']:.3f} (diferencia "
            f"{f('SVM RBF (Nyström) + TPE − Logística B'):+.3f}, IC [{lo('SVM RBF (Nyström) + TPE − Logística B'):+.3f}, "
            f"{hi('SVM RBF (Nyström) + TPE − Logística B'):+.3f}]) y reconoce mejor el estrato 5, pero a costa del 4. "
            f"XGBoost, que en la CV empataba con la logística ({cv['XGBoost']:.3f}), en test queda en "
            f"{t['f1_macro']['XGBoost']:.3f} y con un MAE ordinal claramente peor ({t['MAE_ordinal']['XGBoost']:.2f} "
            f"frente a {t['MAE_ordinal']['Logística B_fisicas+ubicacion']:.2f}). Los árboles parten el mapa en "
            "rectángulos y no extrapolan: en una zona nueva repiten el estrato de la zona de entrenamiento más parecida, "
            "mientras que la logística prolonga el gradiente sur-norte.",
            f"El XGBoost geográfico apenas se distingue de su control (diferencia de F1 "
            f"{f('XGBoost geográfico − XGBoost'):+.3f}). En la CV, dar más peso a los modelos locales empeoró el F1 "
            "(figura de la derecha), aun cuando cubrían la mayoría de las viviendas de validación: con el buffer de 1 km, "
            "los modelos locales se entrenan con los barrios vecinos y no con el de la vivienda, y el estrato cambia de "
            f"forma brusca entre barrios. En test, solo el {A['cobertura_test']:.0%} de las viviendas cae dentro de "
            "alguna ventana local. Los efectos locales que este modelo busca necesitan datos de la misma zona, que es "
            "justo lo que la validación espacial excluye: serviría para completar viviendas dentro de barrios ya "
            "conocidos, no para zonas nuevas.",
            f"Rotation Forest queda cerca de la logística en las métricas ordinales (kappa "
            f"{t['kappa_cuadrático']['Rotation Forest']:.2f}) pero por debajo en F1 (diferencia "
            f"{f('Rotation Forest − Logística B'):+.3f}, IC [{lo('Rotation Forest − Logística B'):+.3f}, "
            f"{hi('Rotation Forest − Logística B'):+.3f}]).",
            titulo="Interpretación"),
        dcc.Dropdown(id="avz-residuo", value="XGBoost geográfico", clearable=False, style={"maxWidth": 420},
                     options=[{"label": F.NOMBRE_CORTO[m], "value": m}
                              for m in ["Logística B_fisicas+ubicacion"] + F.AVANZADOS], className="filtros mb-2"),
        grafico(F.fig_mapa_residuos_modelo("XGBoost geográfico"), "avz-residuo-fig"),
        interpretacion(
            "Compara los mapas de residuos: todos los modelos nuevos dejan los residuos más agrupados que la logística "
            "(I de Moran entre 0.77 y 0.87, frente a 0.70). Ninguno de los modelos elimina la autocorrelación de los residuos: la información "
            "de barrio que falta no está en las variables del catastro de la propia vivienda.",
            titulo="Residuos espaciales por modelo"),
    )


tab_modelos = dbc.Container([
    dbc.Row([
        dbc.Col(tarjeta_kpi(f"{IC['valor en test']['f1_macro']:.2f}", "F1 macro en test",
                            f"IC 95 % por bloques: [{IC['IC95_inf']['f1_macro']:.2f}, {IC['IC95_sup']['f1_macro']:.2f}]"), md=3),
        dbc.Col(tarjeta_kpi(f"{IC['valor en test']['accuracy']:.2f}", "accuracy en test", "por debajo de la alerta de 0.80"), md=3),
        dbc.Col(tarjeta_kpi(f"{R['resultados_test']['accuracy_±1']['Logística B_fisicas+ubicacion']:.2f}",
                            "accuracy ±1 estrato", "casi todos los errores son entre vecinos"), md=3),
        dbc.Col(tarjeta_kpi(f"{IC['valor en test']['kappa_cuadrático']:.2f}", "kappa cuadrático",
                            "acuerdo ordinal alto"), md=3),
    ], className="g-3 mt-2"),

    seccion("1. Modelo base y líneas base",
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
            grafico(F.fig_comparacion("f1_macro"), "mod-comp"),
            html.Div(tabla(F.tabla_metricas()), className="tabla-scroll"),
            interpretacion(
                "La logística B es el modelo principal: tiene el mayor F1 en la validación cruzada espacial (0.30) y en "
                "test (0.45). Supera con claridad a las Dummy: la diferencia de F1 frente a la mayoritaria es +0.38 "
                "(IC 95 % [0.11, 0.39]) y frente a la estratificada +0.28 (IC [0.03, 0.31]).",
                "Frente a la moda por zona también gana en todas las métricas de test (F1 0.45 frente a 0.33; MAE 0.46 "
                "frente a 0.94), pero la diferencia de F1 no es estadísticamente significativa (+0.12, IC [−0.01, 0.19]): "
                "con solo 7 bloques de test la incertidumbre es grande. Lo mismo pasa entre B y A.",
                "El F1 de test (0.45) es mayor que el de la CV (0.30) por dos razones probables: los 7 bloques de test "
                "resultaron favorables (con muy poco estrato 6) y el modelo final se entrena con 138 239 viviendas, entre "
                "1.9 y 2.6 veces las de cada fold. La estimación más representativa del desempeño en zonas nuevas es la "
                "de la CV espacial. La accuracy (0.60) no activa la alerta de posible fuga (≥ 0.80).")),

    seccion("2. ¿Por qué validar por bloques espaciales?",
            grafico(F.fig_optimismo()),
            interpretacion(
                "Con validación aleatoria, viviendas del mismo edificio y de la misma cuadra caen a la vez en "
                "entrenamiento y validación, y el modelo «reconoce» la zona: el F1 sube a 0.60, el doble que con "
                "bloques. Ese número sería engañoso para predecir en zonas nuevas.",
                "El buffer también importa: sin él el F1 de CV es 0.40, con 1 km baja a 0.30 y con el alcance completo "
                "del correlograma (2.9 km) cae a 0.18, porque se pierde mucho entrenamiento. Se eligió 1 km como "
                "compromiso.")),

    seccion("3. Diagnóstico por modelo",
            dbc.Row([dbc.Col(dcc.Dropdown(id="mod-modelo", options=opciones_modelo,
                                          value="Logística B_fisicas+ubicacion", clearable=False), md=5),
                     dbc.Col(dbc.Checklist(id="mod-normalizar", value=["si"], switch=True,
                                           options=[{"label": "matriz en % por fila", "value": "si"}]), md=3)],
                    className="filtros mb-2"),
            dbc.Row([dbc.Col(grafico(F.fig_confusion(), "mod-conf"), md=6),
                     dbc.Col(grafico(F.fig_por_clase(), "mod-clase"), md=6)]),
            dbc.Row([dbc.Col(grafico(F.fig_roc(), "mod-roc"), md=6),
                     dbc.Col(grafico(F.fig_calibracion(), "mod-cal"), md=6)]),
            interpretacion(
                "En la logística B los errores son casi siempre entre estratos vecinos (accuracy ±1 = 0.94). Hay dos "
                "desplazamientos hacia el centro de la escala: el estrato 2 casi nunca se predice (recall 0.06; la "
                "mayoría se clasifica como 3) y más de la mitad de los estratos 5 y 6 se predicen como 4.",
                "Las curvas ROC muestran que el modelo sí ordena bien el estrato 2 (AUC alto), pero casi nunca gana el "
                "argmax frente al 3: el problema es de calibración y de umbral, no de falta de señal. La calibración "
                "confirma que el estrato 3 está sobreestimado y el 2 subestimado.",
                "Selecciona otro modelo en la lista para comparar: la Dummy mayoritaria solo predice estrato 1 (la clase más frecuente en el entrenamiento que queda tras el buffer) y la moda "
                "por zona acierta las zonas grandes pero falla en los bordes entre estratos.")),

    seccion("4. Coeficientes",
            dbc.RadioItems(id="mod-coef", value="B_fisicas+ubicacion", inline=True, className="filtros mb-2",
                           options=[{"label": "Logística B", "value": "B_fisicas+ubicacion"},
                                    {"label": "Logística A", "value": "A_fisicas"}]),
            grafico(F.fig_coeficientes(), "mod-coef-fig"),
            interpretacion(
                "Cada celda es el coeficiente de la variable (estandarizada) en la ecuación de ese estrato: azul empuja "
                "hacia el estrato, rojo lo aleja. Se muestran las variables con mayor diferencia entre el estrato 6 y el 1.",
                "La posición norte-sur domina: moverse al norte aumenta la probabilidad de estratos altos y reduce la de "
                "estratos bajos. Entre las físicas, más baños y más área empujan hacia estratos altos, y los predios "
                "informales hacia el estrato 1.",
                "Cautela: las variables están correlacionadas entre sí, así que un coeficiente aislado no es un efecto "
                "causal, y la multinomial no usa el orden de los estratos.")),

    seccion("5. Curva de aprendizaje y residuos espaciales",
            dbc.Row([dbc.Col(grafico(F.fig_curva_aprendizaje()), md=5), dbc.Col(grafico(F.fig_mapa_residuos()), md=7)]),
            interpretacion(
                "La curva de validación casi no sube al multiplicar los datos por 20, de 2 700 a 54 000 viviendas (F1 de "
                "0.27 a 0.30): es poco probable que más filas cambien mucho este modelo. Lo que falta es flexibilidad (no linealidades, interacciones) y variables del "
                "entorno. La brecha entre entrenamiento y validación refleja sobre todo el cambio de zona, no "
                "sobreajuste a filas.",
                "En el mapa, cada punto es el residuo medio (estrato real − estrato esperado) de las viviendas de test en "
                "una celda de ~110 m: azul significa que el modelo subestima el estrato, rojo que lo sobreestima.",
                f"Los residuos siguen fuertemente autocorrelacionados en el espacio (Moran = "
                f"{D['moran_residuos']['B_fisicas+ubicacion']['I']:.2f}): el modelo capta el gradiente norte-sur, pero no "
                "los barrios. Hay zonas enteras donde subestima (azul) o sobreestima (rojo). Esa fue la motivación para "
                "probar modelos con efectos locales, como el XGBoost geográfico de la revisión bibliográfica (sección 7).")),

    seccion("6. Simulador", simulador),
    seccion_avanzados(),
], fluid=True)

# ===========================================================================
# Layout
# ===========================================================================
app.layout = html.Div([
    html.Header(dbc.Container([
        html.H1("Estrato socioeconómico de las viviendas de Barranquilla", className="titulo"),
        html.Div("Clasificación con datos catastrales — Entregable 2, Machine Learning (Maestría, Universidad del "
                 "Norte). Profesor Lihki Rubio.", className="subtitulo"),
        html.Div(AUTORES, className="autores"),
    ], fluid=True), className="cabecera"),
    dbc.Container(dbc.Tabs([
        dbc.Tab(tab_contexto, label="1. Contexto del problema", tab_id="t1"),
        dbc.Tab(tab_eda, label="2. Análisis exploratorio", tab_id="t2"),
        dbc.Tab(tab_modelos, label="3. Modelos base", tab_id="t3"),
    ], active_tab="t1", className="mt-3"), fluid=True),
    html.Footer(dbc.Container(html.Small(
        "Datos: catastro abierto de la Alcaldía de Barranquilla (descarga del 15-sep-2026). Coordenadas redondeadas a "
        "~110 m y sin número predial. Semilla 42. " + AUTORES), fluid=True), className="pie"),
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
