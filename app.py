"""Dashboard final del proyecto — Estrato socioeconómico de las viviendas de Barranquilla.

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)
Machine Learning — Maestría en Ingeniería Mecánica, Universidad del Norte. Profesor Lihki Rubio.

Reúne los Entregables 1, 2 y 3 y los desarrollos posteriores en ocho pestañas, ordenadas como la investigación
(problema y revisión → datos → EDA → modelo base → modelo propuesto y protocolo → resultados → optimización del
Entregable 3 → conclusiones). Cada pestaña es una secuencia de
diapositivas (gráfica o tabla interactiva + lectura breve + fichas de cómo leer cada métrica); se navega con el menú
lateral, con los botones o con las flechas del teclado. El contenido está en ``diapositivas.py``; las figuras, en
``figuras.py``, ``figuras_proyecto.py`` y ``figuras_mejoras.py``; el glosario de métricas, en ``glosario.py``.
Para Render: ``gunicorn app:server``.
"""
import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

import componentes as CMP
import diapositivas as DP
from componentes import fmt, glosario_completo, radio

app = Dash(__name__, title="Estrato socioeconómico en Barranquilla — Proyecto final de Machine Learning",
           suppress_callback_exceptions=True,
           external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"])
server = app.server

PESTANAS = {p["id"]: p for p in DP.PESTANAS}
# Número de cada pestaña (la presentación de 20 min no lleva número: reúne diapositivas de las demás).
ORDEN = {p["id"]: n + 1 for n, p in enumerate(p for p in DP.PESTANAS if not p.get("sin_numero"))}
AUTORES = "María Carolina Cantillo Orozco · Juan Camilo Oñoro Araujo"
T, CV, B = DP.T, DP.CV, DP.B


def cifra(valor, etiqueta):
    return html.Div([html.Div(valor, className="hero-cifra-valor"), html.Div(etiqueta, className="hero-cifra-etiqueta")],
                    className="hero-cifra")


def cifra_modelo_nuevo():
    """F1 de test en zonas nuevas: modelo base → modelo nuevo (XGBoost geográfico con vecindad), si ya está calculado."""
    ar = DP.FP.arboles_knndm()
    if not ar:
        return cifra(f"{T['f1_macro'][B]:.2f}", "F1 macro en zonas nuevas")
    k = ar["xgb_geografico_vs_logistica"]["zonas nuevas"]["variantes"]["XGBoost geográfico | B + vecindad KNN (k = 8)"]
    base = k["F1_test"] - k["vs Logística B v3 (sin vecindad)"]["test_F1"]["diferencia"]
    return cifra(f"{base:.2f} → {k['F1_test']:.2f}", "F1 en zonas nuevas: modelo base → modelo nuevo")


portada = html.Header(html.Div([
    html.Div("Machine Learning · Maestría en Ingeniería Mecánica · Universidad del Norte · Proyecto final",
             className="hero-kicker"),
    html.H1(["¿Se puede predecir el ", html.Span("estrato", className="resalte"), " de una vivienda con sus datos catastrales?"],
            className="hero-titulo"),
    html.P("Clasificación del estrato socioeconómico de las viviendas de Barranquilla con el catastro abierto, "
           "validación espacial, modelos originales de la revisión bibliográfica y optimización de hiperparámetros.",
           className="hero-sub"),
    html.Div([cifra(fmt(DP.E1["viviendas"]), "viviendas"), cifra("96.9 %", "ubicadas en el mapa"),
              cifra(f"{DP.D['moran_estrato']['I']:.2f}", "I de Moran del estrato"),
              cifra_modelo_nuevo()], className="hero-cifras"),
    html.Div([html.Div(AUTORES, className="hero-autores"), html.Div("Profesor Lihki Rubio · Octubre de 2026", className="hero-profesor")]),
    html.A("Comenzar  ↓", href="#navegacion", className="hero-boton"),
    html.Div("Imagen: desembocadura del río Magdalena y Barranquilla. Contains modified Copernicus Sentinel data (2021), "
             "processed by ESA.", className="hero-credito"),
], className="hero-contenido"), className="hero")

def etiqueta_pestana(p):
    if p.get("sin_numero"):
        total = sum(d[2]["minutos"] for d in p["diapositivas"])
        return f"▶ {total:g} min"
    return f"{ORDEN[p['id']]}. {p.get('corto', p['titulo'])}"


navegacion = html.Div(dbc.Container([
    html.Div([html.Span("Estrato · Barranquilla", className="nav-marca"),
              dbc.Tabs(id="pestana", active_tab="contexto", className="pestanas",
                       children=[dbc.Tab(label=etiqueta_pestana(p), tab_id=p["id"], label_class_name=(
                                     "pestana-presentacion" if p.get("sin_numero") else None)) for p in DP.PESTANAS]),
              html.Div([radio("vista", [("continua", "Continua"), ("diapositivas", "Por diapositiva")], "continua"),
                        dbc.Button("Glosario", id="btn-glosario", className="btn-nav", n_clicks=0),
                        dbc.Button("Presentar", id="btn-presentar", className="btn-nav", n_clicks=0)],
                       className="nav-acciones")], className="nav-fila")], fluid=True), id="navegacion", className="barra-nav")

glosario = dbc.Offcanvas([
    dbc.Input(id="glosario-buscar", placeholder="Buscar una métrica (p. ej., Moran, F1, VIF)…", type="search", className="mb-3"),
    html.Div(glosario_completo(), id="glosario-lista")],
    id="glosario", title="Glosario: qué mide cada métrica y cómo leer su valor", placement="end", is_open=False,
    className="glosario", scrollable=True)

app.layout = html.Div([
    portada, navegacion,
    dbc.Container(html.Div(id="cuerpo"), fluid=True, className="cuerpo"),
    glosario,
    dcc.Store(id="indices", data={p["id"]: 0 for p in DP.PESTANAS}),
    html.Footer(dbc.Container(
        "Datos: catastro abierto de la Alcaldía de Barranquilla (descarga del 15-sep-2026), con coordenadas redondeadas a "
        "~110 m y sin número predial. Semilla 42. " + AUTORES + ".", fluid=True), className="pie"),
], id="raiz")


def num_menu(p, i):
    """Número de la diapositiva en el menú lateral; en la presentación de 20 min, los minutos sugeridos."""
    extra = p["diapositivas"][i][2:]
    return f"{extra[0]['minutos']:g}′" if extra else f"{i + 1:02d}"


def menu(pid, activo):
    p = PESTANAS[pid]
    return html.Div([html.Button([html.Span(num_menu(p, i), className="menu-num"), html.Span(t, className="menu-texto")],
                                 id={"type": "menu", "i": i}, n_clicks=0,
                                 className="menu-item" + (" activo" if i == activo else ""))
                     for i, (t, *_) in enumerate(p["diapositivas"])], className="menu")


def pie_navegacion(pid, i):
    n = len(PESTANAS[pid]["diapositivas"])
    return html.Div([
        html.Button("← Anterior", id="btn-anterior", n_clicks=0, className="btn-paso", disabled=i == 0),
        html.Div([html.Div(className="progreso-barra", style={"width": f"{100 * (i + 1) / n:.0f}%"})], className="progreso"),
        html.Span(f"{i + 1} / {n}", className="contador"),
        html.Button("Siguiente →", id="btn-siguiente", n_clicks=0, className="btn-paso", disabled=i == n - 1)],
        className="pie-navegacion")




def construir(pid, i):
    titulo, funcion, *extra = PESTANAS[pid]["diapositivas"][i]
    # Numeración automática según la posición; en la presentación de 20 min, la de la pestaña de origen.
    pid_n, i_n = extra[0]["origen"] if extra else (pid, i)
    CMP.NUMERO_ACTUAL = f"{ORDEN[pid_n]}.{i_n + 1}"
    CMP.COMPACTA, CMP.PUNTOS = bool(extra), (extra[0]["puntos"] if extra else None)
    try:
        return funcion()
    except Exception as e:  # una diapositiva con un error no debe tumbar la presentación
        return dbc.Alert(f"No se pudo construir «{titulo}»: {e}", color="danger")
    finally:
        CMP.NUMERO_ACTUAL, CMP.COMPACTA, CMP.PUNTOS = None, False, None


def cuerpo_diapositiva(pid, i):
    return html.Div([construir(pid, i), pie_navegacion(pid, i)], className="diapo-marco")


# Vista continua: todas las diapositivas de la pestaña una debajo de otra; el menú salta a cada una.
def menu_continuo(pid):
    p = PESTANAS[pid]
    return html.Div([html.A([html.Span(num_menu(p, i), className="menu-num"), html.Span(t, className="menu-texto")],
                            href=f"#d-{pid}-{i}", className="menu-item menu-enlace")
                     for i, (t, *_) in enumerate(p["diapositivas"])], className="menu")


def cuerpo_continuo(pid):
    return html.Div([html.Section(construir(pid, i), id=f"d-{pid}-{i}", className="diapo-marco diapo-continua")
                     for i in range(len(PESTANAS[pid]["diapositivas"]))])


@app.callback(Output("cuerpo", "children"), Input("pestana", "active_tab"), Input("vista", "value"),
              State("indices", "data"))
def mostrar_pestana(pid, vista, indices):
    if vista == "continua":
        return dbc.Row([dbc.Col(menu_continuo(pid), lg=2, className="col-menu"),
                        dbc.Col(cuerpo_continuo(pid), lg=10, className="col-area")], className="g-4")
    i = indices.get(pid, 0)
    return dbc.Row([dbc.Col(menu(pid, i), lg=2, className="col-menu"),
                    dbc.Col(html.Div(cuerpo_diapositiva(pid, i), id="area"), lg=10, className="col-area")], className="g-4")


@app.callback(Output("area", "children"), Output("indices", "data"), Output({"type": "menu", "i": ALL}, "className"),
              Input({"type": "menu", "i": ALL}, "n_clicks"), Input("btn-anterior", "n_clicks"),
              Input("btn-siguiente", "n_clicks"), State("pestana", "active_tab"), State("indices", "data"),
              prevent_initial_call=True)
def navegar(_menu, _ant, _sig, pid, indices):
    disparo = ctx.triggered_id
    if disparo is None or not any(t["value"] for t in ctx.triggered):
        raise PreventUpdate  # un menú recién creado dispara el callback sin clic
    n = len(PESTANAS[pid]["diapositivas"])
    i = indices.get(pid, 0)
    if disparo == "btn-anterior":
        i = max(i - 1, 0)
    elif disparo == "btn-siguiente":
        i = min(i + 1, n - 1)
    else:
        i = disparo["i"]
    indices = dict(indices, **{pid: i})
    return cuerpo_diapositiva(pid, i), indices, ["menu-item" + (" activo" if j == i else "") for j in range(n)]


@app.callback(Output("glosario", "is_open"), Input("btn-glosario", "n_clicks"), State("glosario", "is_open"),
              prevent_initial_call=True)
def abrir_glosario(_, abierto):
    return not abierto


@app.callback(Output("glosario-lista", "children"), Input("glosario-buscar", "value"), prevent_initial_call=True)
def buscar_glosario(texto):
    return glosario_completo(texto)


# Modo presentación: oculta la portada y el menú lateral y agranda la diapositiva (clase en la raíz).
app.clientside_callback(
    """function(n) {
        const r = document.getElementById('raiz');
        if (!r) { return window.dash_clientside.no_update; }
        const activo = r.classList.toggle('modo-presentacion', n % 2 === 1);
        if (activo) { document.getElementById('navegacion').scrollIntoView(); }
        return activo ? 'Salir' : 'Presentar';
    }""",
    Output("btn-presentar", "children"), Input("btn-presentar", "n_clicks"), prevent_initial_call=True)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
