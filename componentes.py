"""Componentes visuales del dashboard final: diapositiva, fichas de lectura de métricas, tarjetas y tablas.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)
"""
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from glosario import GLOSARIO

GRAF = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2},
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def fmt(n, dec=0):
    return f"{n:,.{dec}f}".replace(",", " ")


def grafico(fig, id_=None, alto=None, minimo=None):
    """Gráfica que se adapta a la pantalla: usa la altura de la figura (o ``alto``) como máximo y, en pantallas bajas,
    lo que queda descontando la barra de navegación y la cabecera de la diapositiva, para que quepa sin desplazarse.
    ``responsive=True`` hace que Plotly ignore la altura fija de la figura (también en las que llegan por callback).
    ``minimo``: alto mínimo para gráficas con muchas filas, que con menos alto ocultarían etiquetas."""
    h = round(0.9 * (alto or (fig.layout.height if fig is not None and fig.layout.height else 450)))
    m = minimo or min(h, 320)
    props = {"figure": fig, "config": GRAF, "className": "grafico", "responsive": True,
             "style": {"height": f"clamp({m}px, calc(100vh - var(--alto-nav, 52px) - 210px), {max(h, m)}px)"}}
    return dcc.Graph(id=id_, **props) if id_ else dcc.Graph(**props)


def control(etiqueta, componente, ancho=None):
    return html.Div([html.Label(etiqueta, className="control-etiqueta"), componente], className="control",
                    style={"minWidth": f"{ancho}px"} if ancho else None)


def controles(*hijos):
    return html.Div(list(hijos), className="controles")


def radio(id_, opciones, valor):
    """Botones segmentados (pills) para alternar vistas."""
    return dbc.RadioItems(id=id_, value=valor, inline=True, className="segmentado", inputClassName="btn-check",
                          labelClassName="btn btn-segmento", labelCheckedClassName="activo",
                          options=[{"label": e, "value": v} for v, e in opciones])


def desplegable(id_, opciones, valor, ancho=320):
    return dcc.Dropdown(id=id_, value=valor, clearable=False, options=[{"label": e, "value": v} for v, e in opciones],
                        style={"minWidth": f"{ancho}px"}, className="desplegable")


def interruptor(id_, etiqueta, valor=True):
    return dbc.Switch(id=id_, label=etiqueta, value=valor, className="interruptor")


def kpi(valor, etiqueta, nota=None, color=None):
    return html.Div([html.Div(valor, className="kpi-valor", style={"color": color} if color else None),
                     html.Div(etiqueta, className="kpi-etiqueta"),
                     html.Div(nota, className="kpi-nota") if nota else None], className="kpi")


def tabla(df, decimales=3, alto=None):
    df = df.copy()
    for c in df.select_dtypes("number"):
        df[c] = df[c].map(lambda x: "—" if pd.isna(x) else f"{x:,.{decimales}f}".replace(",", " "))
    t = dbc.Table.from_dataframe(df, striped=False, bordered=False, hover=True, size="sm", className="tabla")
    return html.Div(t, className="tabla-scroll", style={"maxHeight": f"{alto}px"} if alto else None)


# ---------------------------------------------------------------------------
# Fichas de lectura (qué mide la métrica y cómo saber si el valor es bueno)
# ---------------------------------------------------------------------------
def _posicion(tramos, valor):
    """Posición (0 a 100 %) del valor en una escala de tramos de igual ancho."""
    n = len(tramos)
    for j, (a, b, _, _) in enumerate(tramos):
        if valor <= b or j == n - 1:
            f = 0 if b == a else min(max((valor - a) / (b - a), 0), 1)
            return 100 * (j + f) / n, j
    return 100.0, n - 1


def escala(clave, valor=None, etiquetas=False, absoluto=False):
    g = GLOSARIO[clave]
    tramos = g["tramos"]
    if not tramos:
        return None
    segs = [html.Div(html.Span(lectura, className="seg-texto") if etiquetas else None, className="seg",
                     style={"background": color}, title=f"{lectura}: {a:g} a {b:g}") for a, b, lectura, color in tramos]
    hijos = [html.Div(segs, className="escala-barra")]
    if valor is not None:
        v = abs(valor) if absoluto else valor
        pos, j = _posicion(tramos, v)
        hijos.append(html.Div(className="escala-marca", style={"left": f"{pos}%"}))
    cortes = [tramos[0][0]] + [b for _, b, _, _ in tramos]
    marcas = [html.Span(f"{c:g}" if abs(c) < 100 else "", className="escala-corte",
                        style={"left": f"{100 * i / len(tramos)}%"}) for i, c in enumerate(cortes)]
    hijos.append(html.Div(marcas, className="escala-cortes"))
    return html.Div(hijos, className="escala")


def lectura_valor(clave, valor, absoluto=False):
    tramos = GLOSARIO[clave]["tramos"]
    if not tramos or valor is None:
        return None
    _, j = _posicion(tramos, abs(valor) if absoluto else valor)
    return tramos[j][2]


def ficha(clave, valor=None, texto_valor=None, absoluto=False, compacta=True):
    """Ficha de una métrica: qué mide, rango, escala de colores y (si se da) el valor de este proyecto."""
    g = GLOSARIO[clave]
    cab = html.Div([html.Span(g["nombre"], className="ficha-nombre"),
                    html.Span(g["grupo"], className="ficha-grupo")], className="ficha-cab")
    cuerpo = [cab, html.P(g["mide"], className="ficha-mide")]
    if valor is not None or texto_valor:
        lec = lectura_valor(clave, valor, absoluto) if valor is not None else None
        cuerpo.append(html.Div([html.Span("En este proyecto: ", className="ficha-aqui-et"),
                                html.B(texto_valor or f"{valor:.3g}"),
                                html.Span(f" → {lec}", className="ficha-lectura") if lec else None], className="ficha-aqui"))
    e = escala(clave, valor, etiquetas=not compacta, absoluto=absoluto)
    if e:
        cuerpo.append(e)
    if g["rango"] not in ("—", ""):
        cuerpo.append(html.Div([html.Span("Rango: ", className="ficha-aqui-et"), g["rango"]], className="ficha-rango"))
    if g.get("nota") and compacta and not g["tramos"]:
        cuerpo.append(html.P(g["nota"], className="ficha-nota"))
    if g.get("nota") and not compacta:
        cuerpo.append(html.P(g["nota"], className="ficha-nota"))
    if g.get("fuente") and not compacta:
        cuerpo.append(html.Div(f"Fuente de los cortes: {g['fuente']}", className="ficha-fuente"))
    return html.Div(cuerpo, className="ficha" + (" compacta" if compacta else ""))


def glosario_completo(filtro=""):
    from glosario import GRUPOS
    filtro = (filtro or "").strip().lower()
    secciones = []
    for gr in GRUPOS:
        claves = [k for k, g in GLOSARIO.items() if g["grupo"] == gr and
                  (not filtro or filtro in (g["nombre"] + " " + g["mide"]).lower())]
        if claves:
            secciones.append(html.Div([html.H5(gr, className="glosario-grupo")] +
                                      [ficha(k, compacta=False) for k in claves]))
    return secciones or [html.P("Ninguna métrica coincide con la búsqueda.", className="text-muted")]


# ---------------------------------------------------------------------------
# Diapositiva
# ---------------------------------------------------------------------------
NUMERO_ACTUAL = None   # lo fija app.py según la posición de la diapositiva (numeración automática)
COMPACTA = False       # presentación de 20 min: solo los puntos elegidos y la frase final, sin fichas de lectura
PUNTOS = None          # índices de los puntos que se muestran en modo compacto (por defecto, los dos primeros)


def diapositiva(numero, titulo, visual, puntos=None, conclusion=None, metricas=None, fuente=None, ancho_visual=8):
    """Una pantalla de la presentación: título, visual interactivo y panel lateral breve.

    El número que se muestra es el de su posición (pestaña.diapositiva), que fija app.py en ``NUMERO_ACTUAL``;
    el argumento ``numero`` solo se usa si la diapositiva se construye fuera de la app.

    puntos: 2 a 4 frases cortas ("Lo que se ve"); conclusion: una frase; metricas: lista de fichas
    (tuplas (clave, valor[, texto, absoluto]) o claves); fuente: capítulo del libro de donde sale.
    """
    if COMPACTA:
        puntos = [puntos[i] for i in PUNTOS] if (puntos and PUNTOS) else (puntos or [])[:2]
        metricas = None
    panel = []
    if puntos:
        panel.append(html.Div([html.Div("Lo que se ve", className="panel-titulo"),
                               html.Ul([html.Li(p) for p in puntos], className="panel-lista")], className="panel-bloque"))
    if conclusion:
        panel.append(html.Div([html.Div("En una frase", className="panel-titulo claro"), html.P(conclusion)],
                              className="conclusion"))
    if metricas:
        fichas = []
        for m in metricas:
            if isinstance(m, str):
                fichas.append(ficha(m))
            elif not isinstance(m, tuple):
                fichas.append(m)  # componente ya construido (ficha que se actualiza con un callback)
            else:
                clave, valor, *resto = m
                texto = resto[0] if resto else None
                absoluto = resto[1] if len(resto) > 1 else False
                fichas.append(ficha(clave, valor, texto, absoluto))
        panel.append(html.Div([html.Div("¿Cómo se lee?", className="panel-titulo")] + fichas, className="panel-bloque"))
    numero = NUMERO_ACTUAL or numero
    cab = html.Div([html.Div([html.Span(numero, className="diapo-num"),
                              html.Span(fuente, className="diapo-fuente") if fuente else None], className="diapo-kicker"),
                    html.H2(titulo, className="diapo-titulo")], className="diapo-cab")
    if not panel:
        return html.Div([cab, html.Div(visual, className="diapo-visual")], className="diapositiva")
    return html.Div([cab, dbc.Row([dbc.Col(html.Div(visual, className="diapo-visual"), lg=ancho_visual),
                                   dbc.Col(html.Div(panel, className="diapo-panel"), lg=12 - ancho_visual)],
                                  className="g-4")], className="diapositiva")
