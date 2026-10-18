"""Dashboard final del proyecto — Estrato socioeconómico de las viviendas de Barranquilla.

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)
Machine Learning — Maestría, Universidad del Norte. Profesor Lihki Rubio.

Reúne los Entregables 1 y 2 y los desarrollos posteriores en seis pestañas. Cada pestaña es una secuencia de
diapositivas (gráfica o tabla interactiva + lectura breve + fichas de cómo leer cada métrica); se navega con el menú
lateral, con los botones o con las flechas del teclado. El contenido está en ``diapositivas.py``; las figuras, en
``figuras.py`` y ``figuras_proyecto.py``; el glosario de métricas, en ``glosario.py``.
Para Render: ``gunicorn app:server``.
"""
import dash_bootstrap_components as dbc
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

import diapositivas as DP
from componentes import fmt, glosario_completo

app = Dash(__name__, title="Estrato socioeconómico en Barranquilla — Proyecto final de Machine Learning",
           suppress_callback_exceptions=True,
           external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"])
server = app.server

PESTANAS = {p["id"]: p for p in DP.PESTANAS}
AUTORES = "María Carolina Cantillo Orozco · Juan Camilo Oñoro Araujo"
T, CV, B = DP.T, DP.CV, DP.B


def cifra(valor, etiqueta):
    return html.Div([html.Div(valor, className="hero-cifra-valor"), html.Div(etiqueta, className="hero-cifra-etiqueta")],
                    className="hero-cifra")


portada = html.Header(html.Div([
    html.Div("Machine Learning · Maestría · Universidad del Norte · Proyecto final", className="hero-kicker"),
    html.H1(["¿Se puede predecir el ", html.Span("estrato", className="resalte"), " de una vivienda con sus datos catastrales?"],
            className="hero-titulo"),
    html.P("Clasificación del estrato socioeconómico de las viviendas de Barranquilla con el catastro abierto, "
           "validación espacial y modelos de la revisión bibliográfica.", className="hero-sub"),
    html.Div([cifra(fmt(DP.E1["viviendas"]), "viviendas"), cifra("96.9 %", "ubicadas en el mapa"),
              cifra(f"{DP.D['moran_estrato']['I']:.2f}", "I de Moran del estrato"),
              cifra(f"{T['f1_macro'][B]:.2f}", "F1 macro en zonas nuevas")], className="hero-cifras"),
    html.Div([html.Div(AUTORES, className="hero-autores"), html.Div("Profesor Lihki Rubio · Octubre de 2026", className="hero-profesor")]),
    html.A("Comenzar  ↓", href="#navegacion", className="hero-boton"),
    html.Div("Imagen: desembocadura del río Magdalena y Barranquilla. Contains modified Copernicus Sentinel data (2021), "
             "processed by ESA.", className="hero-credito"),
], className="hero-contenido"), className="hero")

navegacion = html.Div(dbc.Container([
    html.Div([html.Span("Estrato · Barranquilla", className="nav-marca"),
              dbc.Tabs(id="pestana", active_tab="contexto", className="pestanas",
                       children=[dbc.Tab(label=f"{i + 1}. {p['titulo']}", tab_id=p["id"]) for i, p in enumerate(DP.PESTANAS)]),
              html.Div([dbc.Button("Glosario", id="btn-glosario", className="btn-nav", n_clicks=0),
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


def menu(pid, activo):
    p = PESTANAS[pid]
    return html.Div([html.Button([html.Span(f"{i + 1:02d}", className="menu-num"), html.Span(t, className="menu-texto")],
                                 id={"type": "menu", "i": i}, n_clicks=0,
                                 className="menu-item" + (" activo" if i == activo else ""))
                     for i, (t, _) in enumerate(p["diapositivas"])], className="menu")


def pie_navegacion(pid, i):
    n = len(PESTANAS[pid]["diapositivas"])
    return html.Div([
        html.Button("← Anterior", id="btn-anterior", n_clicks=0, className="btn-paso", disabled=i == 0),
        html.Div([html.Div(className="progreso-barra", style={"width": f"{100 * (i + 1) / n:.0f}%"})], className="progreso"),
        html.Span(f"{i + 1} / {n}", className="contador"),
        html.Button("Siguiente →", id="btn-siguiente", n_clicks=0, className="btn-paso", disabled=i == n - 1)],
        className="pie-navegacion")


def cuerpo_diapositiva(pid, i):
    titulo, funcion = PESTANAS[pid]["diapositivas"][i]
    try:
        contenido = funcion()
    except Exception as e:  # una diapositiva con un error no debe tumbar la presentación
        contenido = dbc.Alert(f"No se pudo construir «{titulo}»: {e}", color="danger")
    return html.Div([contenido, pie_navegacion(pid, i)], className="diapo-marco")


@app.callback(Output("cuerpo", "children"), Input("pestana", "active_tab"), State("indices", "data"))
def mostrar_pestana(pid, indices):
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
