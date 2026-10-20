"""Figuras del dashboard final del proyecto (Entregables 1 y 2 y desarrollos posteriores).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Complementa a ``figuras.py`` (que sigue siendo la fuente de las figuras del Entregable 2) con las figuras de los
capítulos del Entregable 1 y de los análisis de vecindad y escenarios. Todo se lee de ``datos/``:
``datos/proyecto/`` lo generan ``proyecto/extraer_tablas_e1.py`` y ``proyecto/preparar_proyecto.py``, y
``datos/vecindad/`` los scripts de la carpeta ``vecindad/``. Nada se reentrena aquí.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import figuras as F
from figuras import CLASES, COLOR_ESTRATO, DATOS, DIVERGENTE, SECUENCIAL, _estilo

PROY = DATOS / "proyecto"
VEC = DATOS / "vecindad"
AZUL, AZUL_CLARO, NARANJA, ROJO, VERDE, GRIS = "#2a78d6", "#86b6ef", "#eb6834", "#e34948", "#1baf7a", "#a3a29d"
ETIQUETA = {**F.NUMERICAS, "anio_construccion": "Año de construcción", "x_km": "Posición este-oeste",
            "y_km": "Posición norte-sur", "estrato_num": "Estrato", **F.CATEGORICAS,
            "destinacion_economica": "Destinación económica", "tipo_planta": "Tipo de planta"}


def et(v):
    return ETIQUETA.get(v, str(v).replace("_", " "))


def _num(x):
    """Convierte celdas de tablas extraídas ("1 234", "12.3 %", "0.45") en número."""
    if isinstance(x, (int, float)) or x is None:
        return x
    s = str(x).replace(" ", "").replace(" ", "").replace("%", "").replace("−", "-").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _tablas():
    return json.loads((PROY / "tablas_e1.json").read_text(encoding="utf-8"))


def tabla_e1(clave, indice=True):
    t = _tablas()[clave]
    df = pd.DataFrame(t["filas"], columns=t["columnas"])
    if indice and df.columns[0].startswith("Unnamed"):
        df = df.set_index(df.columns[0])
        df.index.name = None
    return df


@lru_cache(maxsize=None)
def _json(nombre, carpeta="proyecto"):
    base = PROY if carpeta == "proyecto" else VEC
    p = base / nombre
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@lru_cache(maxsize=None)
def _pkl(nombre):
    return pd.read_pickle(PROY / nombre)


def _layout_mapa(fig, titulo, alto=560, zoom=11.3, lat=10.97, lon=-74.81):
    fig.update_layout(map=dict(style=F.ESTILO_MAPA, center=dict(lat=lat, lon=lon), zoom=zoom), title=titulo,
                      height=alto, margin=dict(l=0, r=0, t=50, b=0), paper_bgcolor="white",
                      font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15),
                      legend=dict(orientation="v", y=0.98, x=0.01, yanchor="top", xanchor="left",
                                  bgcolor="rgba(255,255,255,0.88)", bordercolor="#e3e6eb", borderwidth=1, font=dict(size=13)))
    return fig


def _estilo_sub(fig, alto):
    """Estilo para figuras con dos paneles: la leyenda va abajo para no tapar los subtítulos."""
    _estilo(fig, alto=alto)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1), margin=dict(t=110, b=60))
    fig.update_annotations(font_size=14)
    return fig


# ===========================================================================
# DATOS Y CALIDAD (Entregable 1, capítulos 1 a 3)
# ===========================================================================
def fig_manzana_umbral(umbral=250):
    """Dispersión máxima de cada manzana (distancia del punto más lejano a su centroide) y el umbral elegido."""
    m = _json("manzanas.json")
    d = np.array(m["dist_max_m"])
    validas = (d <= umbral).mean()
    fig = go.Figure(go.Histogram(x=np.clip(d, 0, 1500), xbins=dict(start=0, end=1500, size=25), marker_color=AZUL,
                                 marker_line_color="white", marker_line_width=1,
                                 hovertemplate="%{x} m: %{y:,} manzanas<extra></extra>"))
    fig.add_vline(x=umbral, line_color=NARANJA, line_width=3, line_dash="dash",
                  annotation_text=f"umbral {umbral} m: {validas:.0%} de las manzanas son compactas",
                  annotation_position="top right", annotation_font_color=NARANJA)
    fig.update_xaxes(title="distancia del edificio más lejano al centroide de su manzana (m)")
    fig.update_yaxes(title="manzanas")
    fig.update_layout(title="¿Qué tan compacta es cada manzana?", bargap=0.02)
    return _estilo(fig, alto=430, leyenda=False)


def fig_cobertura_v1_v2():
    c = pd.DataFrame(_json("manzanas.json")["cobertura"]).T.astype(float)
    c.index = c.index.astype(int)
    x = [f"Estrato {k}" for k in c.index]
    fig = go.Figure([
        go.Bar(name="v1: solo coordenadas propias", x=x, y=c.v1 / c.total, marker_color="#c3c2b7",
               text=[f"{v:.0%}" for v in c.v1 / c.total], textposition="outside",
               hovertemplate="%{x}: %{y:.1%}<extra>v1</extra>"),
        go.Bar(name="v2: + centroide de la manzana", x=x, y=c.v2 / c.total, marker_color=[COLOR_ESTRATO[k] for k in c.index],
               text=[f"{v:.0%}" for v in c.v2 / c.total], textposition="outside",
               hovertemplate="%{x}: %{y:.1%}<extra>v2</extra>")])
    tot = c.sum()
    fig.update_yaxes(tickformat=".0%", range=[0, 1.15], title="viviendas que entran al modelo")
    fig.update_layout(barmode="group", title=f"Cobertura por estrato: {tot.v1 / tot.total:.1%} (v1) → {tot.v2 / tot.total:.1%} (v2)")
    return _estilo(fig, alto=430)


def fig_mapa_recuperadas():
    v = F.viviendas()
    v = v[v.tiene_coordenadas]
    g = v.assign(origen=np.where(v.ubicado_por_manzana, "ubicada por su manzana (v2)", "coordenadas propias"))
    g = g.groupby(["lat", "lon", "origen"], observed=True).size().reset_index(name="n")
    fig = go.Figure()
    for origen, color, op in [("coordenadas propias", "#9ec5f4", 0.45), ("ubicada por su manzana (v2)", NARANJA, 0.9)]:
        a = g[g.origen == origen]
        fig.add_trace(go.Scattermap(lat=a.lat, lon=a.lon, mode="markers", name=f"{origen} ({a.n.sum():,})".replace(",", " "),
                                    marker=dict(size=np.clip(np.sqrt(a.n) * 1.5, 4, 12), color=color, opacity=op),
                                    customdata=a.n, hovertemplate="%{customdata:,} viviendas<extra>" + origen + "</extra>"))
    return _layout_mapa(fig, "Dónde están las viviendas recuperadas con el centroide de su manzana")


def fig_v1_v2_modelo():
    c = _json("comparacion_v1_v2.json")
    nombres = {"v1 (sus hiperparámetros: sin pesos)": "v1 (como se entregó)", "v1 con pesos balanceados": "v1 + pesos balanceados",
               "v2 (pesos balanceados)": "v2 (versión final)"}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("F1 macro en el MISMO test", "F1 macro en los MISMOS 5 folds de CV"),
                        horizontal_spacing=0.1)
    colores = ["#c3c2b7", AZUL_CLARO, AZUL]
    for (k, nom), col in zip(nombres.items(), colores):
        t = c["test"][k]
        fig.add_trace(go.Bar(x=["todas las viviendas", "solo coordenadas propias"],
                             y=[t["F1 test"], t["F1 test (coordenadas propias)"]], name=nom, marker_color=col,
                             text=[f"{t['F1 test']:.3f}", f"{t['F1 test (coordenadas propias)']:.3f}"], textposition="outside",
                             legendgroup=nom, hovertemplate="%{x}: %{y:.3f}<extra>" + nom + "</extra>"), 1, 1)
        cv = np.array(c["cv"][k]["todo"])
        fig.add_trace(go.Scatter(x=[f"fold {i + 1}" for i in range(5)], y=cv, mode="lines+markers", name=nom,
                                 marker=dict(size=10, color=col), line=dict(color=col, width=2), legendgroup=nom,
                                 showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra>" + nom + f" (media {cv.mean():.3f})</extra>"), 1, 2)
    fig.update_yaxes(range=[0, 0.6], title_text="F1 macro", row=1, col=1)
    fig.update_layout(title="v1 frente a v2 con la misma vara: el F1 no cambia, la cobertura sí", barmode="group")
    return _estilo_sub(fig, 460)


def fig_faltantes_train():
    t = tabla_e1("03_3.1_1")
    t["%"] = t["% nulos"].map(_num)
    t = t[t["%"] > 0].sort_values("%")
    colores = {o: c for o, c in zip(sorted(t["origen principal"].astype(str).unique()), [AZUL, NARANJA, VERDE, "#9085e9", GRIS])}
    fig = go.Figure()
    for o, c in colores.items():
        a = t[t["origen principal"].astype(str) == o]
        fig.add_trace(go.Bar(x=a["%"], y=[et(i) for i in a.index], orientation="h", name=o, marker_color=c,
                             text=[f"{v:.2f} %" for v in a["%"]], textposition="outside",
                             hovertemplate="%{y}: %{x:.3f} %<extra>" + o + "</extra>"))
    fig.update_xaxes(title="% de filas de entrenamiento sin dato", range=[0, t["%"].max() * 1.35])
    fig.update_yaxes(showgrid=False, categoryorder="total ascending")
    fig.update_layout(title="Valores faltantes en entrenamiento y su origen", barmode="stack")
    return _estilo(fig, alto=440)


def fig_outliers_tukey():
    t = tabla_e1("03_3.2_1", indice=False)
    col = [c for c in t.columns if c.startswith("% fuera 1.5")][0]
    col3 = [c for c in t.columns if "3·IQR" in c and c.startswith("%")]
    t["p15"] = t[col].map(_num)
    t["p3"] = t[col3[0]].map(_num) if col3 else np.nan
    t = t.sort_values("p15")
    fig = go.Figure([
        go.Bar(y=[et(v) for v in t.variable], x=t.p15, orientation="h", name="fuera de 1.5·IQR (moderados)",
               marker_color=AZUL_CLARO, hovertemplate="%{y}: %{x:.2f} %<extra>1.5·IQR</extra>"),
        go.Bar(y=[et(v) for v in t.variable], x=t.p3, orientation="h", name="fuera de 3·IQR (extremos)",
               marker_color=AZUL, hovertemplate="%{y}: %{x:.2f} %<extra>3·IQR</extra>")])
    fig.update_xaxes(title="% de filas de entrenamiento")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="Outliers univariados (regla de Tukey): se marcan, no se borran", barmode="overlay")
    fig.update_traces(opacity=0.95)
    return _estilo(fig, alto=430)


def fig_particion_mapa(modo="grupo"):
    m = _pkl("particion_mapa.pkl.gz")
    fig = go.Figure()
    if modo == "grupo" and F.es_v3():
        # Versión 3: el modelo entrena con todo el entrenamiento (sin buffer); el test son los mismos 7 bloques.
        m = m.assign(grupo=m.grupo.replace({"excluida por el buffer": "entrenamiento"}))
        for g, color, op in [("entrenamiento", AZUL, 0.55), ("prueba (test)", NARANJA, 0.8)]:
            a = m[m.grupo == g].groupby(["lat", "lon"], as_index=False).n.sum()
            fig.add_trace(go.Scattermap(lat=a.lat, lon=a.lon, mode="markers", name=f"{g} ({a.n.sum():,})".replace(",", " "),
                                        marker=dict(size=5, color=color, opacity=op), customdata=a.n,
                                        hovertemplate="%{customdata:,} viviendas<extra>" + g + "</extra>"))
        titulo = "7 bloques de prueba; el modelo entrena con todo el resto"
    elif modo in ("grupo", "anterior"):
        for g, color, op in [("entrenamiento", AZUL, 0.55), ("excluida por el buffer", "#c3c2b7", 0.6),
                             ("prueba (test)", NARANJA, 0.75)]:
            a = m[m.grupo == g].groupby(["lat", "lon"], as_index=False).n.sum()
            fig.add_trace(go.Scattermap(lat=a.lat, lon=a.lon, mode="markers", name=f"{g} ({a.n.sum():,})".replace(",", " "),
                                        marker=dict(size=5, color=color, opacity=op), customdata=a.n,
                                        hovertemplate="%{customdata:,} viviendas<extra>" + g + "</extra>"))
        titulo = ("Versión anterior: 7 bloques de prueba y un buffer de 1 km a su alrededor (ya no se usa)" if modo == "anterior"
                  else "Partición espacial: 7 bloques de prueba y un buffer de 1 km a su alrededor")
    else:
        colores = ["#2a78d6", "#eb6834", "#1baf7a", "#9085e9", "#eda100"]
        for f in range(1, 6):
            a = m[m.fold == f].groupby(["lat", "lon"], as_index=False).n.sum()
            fig.add_trace(go.Scattermap(lat=a.lat, lon=a.lon, mode="markers", name=f"fold {f} ({a.n.sum():,})".replace(",", " "),
                                        marker=dict(size=5, color=colores[f - 1], opacity=0.7), customdata=a.n,
                                        hovertemplate="%{customdata:,} viviendas<extra>fold " + str(f) + "</extra>"))
        a = m[m.grupo == "prueba (test)"].groupby(["lat", "lon"], as_index=False).n.sum()
        fig.add_trace(go.Scattermap(lat=a.lat, lon=a.lon, mode="markers", name="test (no se toca)",
                                    marker=dict(size=4, color="#c3c2b7", opacity=0.5), hoverinfo="skip"))
        titulo = "Validación cruzada espacial: cada fold son bloques completos de 2 km"
    return _layout_mapa(fig, titulo, alto=580)


def fig_balance_particion():
    t = tabla_e1("02_Verificaciones de la partición_2", indice=False)
    t["e"] = t.iloc[:, 0].astype(str)
    fig = go.Figure([go.Bar(x=t.e, y=t.train.map(_num), name="entrenamiento", marker_color=AZUL),
                     go.Bar(x=t.e, y=t.test.map(_num), name="prueba", marker_color=NARANJA)])
    fig.update_yaxes(title="% de las viviendas de la partición")
    fig.update_layout(barmode="group", title="¿Train y test tienen la misma mezcla de estratos?")
    return _estilo(fig, alto=400)


# ===========================================================================
# EDA (Entregable 1, capítulos 4 a 9)
# ===========================================================================
def fig_transformacion():
    t = tabla_e1("05_5.5_1", indice=False)
    t["o"], t["l"] = t["asimetría original"].map(_num), t["asimetría log1p"].map(_num)
    t = t.sort_values("o")
    fig = go.Figure()
    for _, r in t.iterrows():
        fig.add_trace(go.Scatter(x=[r.o, r.l], y=[et(r.variable)] * 2, mode="lines", line=dict(color="#c3c2b7", width=3),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t.o, y=[et(v) for v in t.variable], mode="markers", name="original",
                             marker=dict(size=14, color=ROJO), hovertemplate="%{y}: asimetría %{x:.2f}<extra>original</extra>"))
    fig.add_trace(go.Scatter(x=t.l, y=[et(v) for v in t.variable], mode="markers", name="con log(1 + x)",
                             marker=dict(size=14, color=AZUL), hovertemplate="%{y}: asimetría %{x:.2f}<extra>log1p</extra>"))
    fig.add_vrect(x0=-1, x1=1, fillcolor=VERDE, opacity=0.08, line_width=0, annotation_text="|asimetría| < 1",
                  annotation_position="top left")
    fig.update_xaxes(title="asimetría (0 = simétrica)", type="linear")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="¿Hace falta transformar? Asimetría antes y después de log(1 + x)")
    return _estilo(fig, alto=430)


def fig_correlaciones(metodo="spearman"):
    c = _json("correlaciones.json")
    m = np.array(c[metodo])
    nom = [et(v) for v in c["variables"]]
    fig = go.Figure(go.Heatmap(z=m, x=nom, y=nom, zmin=-1, zmax=1, colorscale=DIVERGENTE, text=np.round(m, 2),
                               texttemplate="%{text}", textfont_size=12,
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>", colorbar=dict(title="r")))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title=f"Correlación de {'Spearman (rangos)' if metodo == 'spearman' else 'Pearson (lineal)'} entre numéricas (train)")
    return _estilo(fig, alto=560, leyenda=False)


def fig_pearson_vs_spearman():
    t = tabla_e1("06_6.1_1")
    t["p"], t["s"] = t.pearson.map(_num), t.spearman.map(_num)
    t["par"] = [f"{et(a)} · {et(b)}" for a, b in zip(t.var1, t.var2)]
    t = t.sort_values("s")
    fig = go.Figure()
    for _, r in t.iterrows():
        fig.add_trace(go.Scatter(x=[r.p, r.s], y=[r.par] * 2, mode="lines", line=dict(color="#c3c2b7", width=3),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t.p, y=t.par, mode="markers", name="Pearson", marker=dict(size=12, color=NARANJA)))
    fig.add_trace(go.Scatter(x=t.s, y=t.par, mode="markers", name="Spearman", marker=dict(size=12, color=AZUL)))
    fig.add_vline(x=0, line_color="#8a8984", line_dash="dot")
    fig.update_xaxes(title="coeficiente de correlación", range=[-1, 1])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="Pares con mayor diferencia: Pearson se deja llevar por las colas")
    return _estilo(fig, alto=500)


def fig_hexbin(x="area_construida", y="total_banios", log=True):
    v = F.base_eda("train")[[x, y]].dropna()
    v = v.sample(min(80_000, len(v)), random_state=42)
    a, b = (np.log1p(v[x].clip(lower=0)), np.log1p(v[y].clip(lower=0))) if log else (v[x], v[y])
    from scipy import stats
    rs, rp = stats.spearmanr(v[x], v[y])[0], np.corrcoef(a, b)[0, 1]
    H, xe, ye = np.histogram2d(a, b, bins=60)
    z = np.where(H > 0, np.log10(np.maximum(H, 1)), np.nan).T
    marcas = [1, 10, 100, 1000, 10000]
    fig = go.Figure(go.Heatmap(x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, z=z, colorscale=SECUENCIAL[1:],
                               customdata=H.T, hovertemplate="x %{x:.2f}<br>y %{y:.2f}<br>%{customdata:,.0f} viviendas<extra></extra>",
                               colorbar=dict(title="viviendas", tickvals=np.log10(marcas), ticktext=[f"{m:,}".replace(",", " ") for m in marcas])))
    pre = "log(1 + " if log else ""
    suf = ")" if log else ""
    fig.update_xaxes(title=f"{pre}{et(x)}{suf}")
    fig.update_yaxes(title=f"{pre}{et(y)}{suf}")
    fig.update_layout(title=f"{et(x)} vs {et(y)}: Spearman {rs:.2f} · Pearson ({'log' if log else 'original'}) {rp:.2f}")
    return _estilo(fig, alto=500, leyenda=False)


def fig_violin(var="total_banios", log=True):
    v = F.base_eda("train")[[var, "estrato_num"]].dropna()
    v = v.sample(frac=1, random_state=42).groupby("estrato_num").head(4000)
    fig = go.Figure()
    for k in CLASES:
        x = v.loc[v.estrato_num == k, var]
        fig.add_trace(go.Violin(y=np.log1p(x) if log else x, name=f"E{k}", line_color=COLOR_ESTRATO[k],
                                fillcolor=COLOR_ESTRATO[k], opacity=0.75, box_visible=True, meanline_visible=False,
                                points=False, spanmode="hard", hoverinfo="skip"))
    ef = F.efecto_numerica(var)
    fig.update_yaxes(title=f"log(1 + {et(var)})" if log else et(var))
    fig.update_layout(title=f"{et(var)} por estrato — η² de Kruskal-Wallis = {ef['eta2_H']:.3f}")
    return _estilo(fig, alto=480, leyenda=False)


def fig_ranking_asociacion():
    """Ranking consolidado del capítulo 6.7: información mutua normalizada, con la medida propia de cada tipo."""
    t = tabla_e1("06_6.7_1")
    t["mi"] = t["MI / H(Y)"].map(_num)
    t = t.sort_values("mi")
    eta = t["η²_H (num.)"].map(_num)
    v = t["V Cramér (cat.)"].map(_num)
    tipo = np.where(eta.notna(), "numérica", "categórica")
    fig = go.Figure(go.Bar(x=t.mi, y=[et(i) for i in t.index], orientation="h",
                           marker_color=[AZUL if a == "numérica" else NARANJA for a in tipo],
                           customdata=np.c_[eta.fillna(-1), v.fillna(-1), tipo],
                           hovertemplate="%{y}<br>MI / H(Y) = %{x:.3f}<br>%{customdata[2]}<extra></extra>"))
    fig.update_xaxes(title="información mutua normalizada con el estrato (MI / H(Y))")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="¿Qué variable dice más del estrato? (azul: numéricas · naranja: categóricas)")
    return _estilo(fig, alto=520, leyenda=False)


def fig_vif():
    t = pd.concat([tabla_e1("06_6.8_1", indice=False).assign(grupo="numéricas"),
                   tabla_e1("06_6.8_2", indice=False).assign(grupo="espaciales")])
    t["vif"] = t.VIF.map(_num)
    t = t.drop_duplicates("variable").sort_values("vif")
    fig = go.Figure(go.Bar(x=t.vif, y=[et(v) for v in t.variable], orientation="h",
                           marker_color=[ROJO if x >= 10 else (NARANJA if x >= 5 else AZUL) for x in t.vif],
                           text=[f"{x:.1f}" for x in t.vif], textposition="outside",
                           hovertemplate="%{y}: VIF %{x:.2f}<extra></extra>"))
    fig.add_vline(x=5, line_dash="dot", line_color=NARANJA, annotation_text="5")
    fig.add_vline(x=10, line_dash="dash", line_color=ROJO, annotation_text="10")
    fig.update_xaxes(title="VIF = 1 / (1 − R² con las demás)", range=[0, max(11.5, t.vif.max() * 1.15)])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="Multicolinealidad: factor de inflación de la varianza")
    return _estilo(fig, alto=460, leyenda=False)


def fig_scree():
    m = _json("multivariado.json")
    var = np.array(m["varianza"])
    pcs = [f"PC{i + 1}" for i in range(len(var))]
    fig = go.Figure([go.Bar(x=pcs, y=var, name="varianza de cada componente", marker_color=AZUL,
                            text=[f"{v:.0%}" for v in var], textposition="outside",
                            customdata=m["autovalor"], hovertemplate="%{x}: %{y:.1%} (λ = %{customdata:.2f})<extra></extra>"),
                     go.Scatter(x=pcs, y=np.cumsum(var), name="acumulada", mode="lines+markers",
                                line=dict(color=NARANJA, width=2), marker=dict(size=9),
                                hovertemplate="hasta %{x}: %{y:.1%}<extra></extra>")])
    fig.add_hline(y=0.8, line_dash="dot", line_color="#8a8984", annotation_text="80 %")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.08], title="varianza explicada")
    fig.update_layout(title=f"PCA de las variables físicas: dimensionalidad efectiva ≈ {m['participation_ratio']:.1f} de {len(var)}")
    return _estilo(fig, alto=430)


def fig_cargas(n=4):
    m = _json("multivariado.json")
    c = np.array(m["cargas"])[:n].T
    var = np.array(m["varianza"])
    fig = go.Figure(go.Heatmap(z=c, x=[f"PC{i + 1} ({var[i]:.0%})" for i in range(n)], y=[et(v) for v in m["variables"]],
                               zmin=-1, zmax=1, colorscale=DIVERGENTE, text=np.round(c, 2), texttemplate="%{text}",
                               textfont_size=13, hovertemplate="%{y} en %{x}: %{z:.2f}<extra></extra>",
                               colorbar=dict(title="carga")))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title="Cargas: cuánto pesa cada variable en cada componente")
    return _estilo(fig, alto=430, leyenda=False)


def fig_biplot(px="PC1", py="PC2", color="estrato"):
    m = _json("multivariado.json")
    s = _pkl("multivariado_muestra.pkl.gz")
    i, j = int(px[2:]) - 1, int(py[2:]) - 1
    var = np.array(m["varianza"])
    fig = go.Figure()
    if color == "estrato":
        for k in CLASES:
            a = s[s.estrato == k]
            fig.add_trace(go.Scattergl(x=a[px], y=a[py], mode="markers", name=f"E{k}",
                                       marker=dict(size=4, color=COLOR_ESTRATO[k], opacity=0.55), hoverinfo="skip"))
    else:
        cats = s.condicion_predio.value_counts().index
        pal = [AZUL, NARANJA, VERDE, "#9085e9", "#eda100", GRIS]
        for c, col in zip(cats, pal):
            a = s[s.condicion_predio == c]
            fig.add_trace(go.Scattergl(x=a[px], y=a[py], mode="markers", name=c,
                                       marker=dict(size=4, color=col, opacity=0.55), hoverinfo="skip"))
    cargas = np.array(m["cargas"])
    largo = np.hypot(cargas[i], cargas[j])
    # La flecha más larga llega al 85 % de la nube; solo se rotulan las flechas que pesan en este plano.
    esc = float(np.abs(s[[px, py]]).quantile(0.99).max()) * 0.85 / largo.max()
    for v, a, b, L in zip(m["variables"], cargas[i], cargas[j], largo):
        fig.add_annotation(x=a * esc, y=b * esc, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowwidth=2.5, arrowcolor=ROJO if L >= 0.35 else "#e9a3a3", text="")
        if L >= 0.35:
            # El rótulo sale hacia afuera desde la punta: a la derecha/izquierda si la flecha es más horizontal,
            # arriba/abajo si es más vertical.
            if abs(a) >= abs(b):
                xa, ya, dx, dy = ("left" if a > 0 else "right"), "middle", (6 if a > 0 else -6), 0
            else:
                xa, ya, dx, dy = "center", ("bottom" if b > 0 else "top"), 0, (6 if b > 0 else -6)
            fig.add_annotation(x=a * esc, y=b * esc, text=f"<b>{et(v)}</b>", showarrow=False, xanchor=xa, yanchor=ya,
                               xshift=dx, yshift=dy, font=dict(color="#8b1e2d", size=13), bgcolor="rgba(255,255,255,0.8)")
    fig.update_xaxes(title=f"{px} ({var[i]:.0%})", zeroline=True, zerolinecolor="#c3c2b7")
    fig.update_yaxes(title=f"{py} ({var[j]:.0%})", zeroline=True, zerolinecolor="#c3c2b7")
    fig.update_layout(title=f"Biplot {px}–{py} (muestra de 8 000 viviendas de train; flechas claras: pesan poco aquí)",
                      legend_itemsizing="constant")
    return _estilo(fig, alto=560)


def fig_outliers_multivariados():
    m = _json("multivariado.json")["outliers"]
    s = _pkl("multivariado_muestra.pkl.gz")
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.1,
                        subplot_titles=("Dónde caen (espacio PC1–PC2)", "Estrato: outliers frente al resto"))
    for g, col, sz in [("no", "#dcdbd6", 3), ("solo Isolation Forest", "#9085e9", 6), ("solo Mahalanobis", NARANJA, 6),
                       ("ambos métodos", ROJO, 7)]:
        a = s[s.outlier == g]
        fig.add_trace(go.Scattergl(x=a.PC1, y=a.PC2, mode="markers", name=g, marker=dict(size=sz, color=col, opacity=0.7),
                                   hoverinfo="skip"), 1, 1)
    eo = pd.Series(m["estrato_outliers"]).reindex([str(k) for k in CLASES]).fillna(0)
    er = pd.Series(m["estrato_resto"]).reindex([str(k) for k in CLASES]).fillna(0)
    fig.add_trace(go.Bar(x=[f"E{k}" for k in CLASES], y=eo.values, name="outliers (ambos)", marker_color=ROJO, showlegend=False,
                         hovertemplate="%{x}: %{y:.1f} %<extra>outliers</extra>"), 1, 2)
    fig.add_trace(go.Bar(x=[f"E{k}" for k in CLASES], y=er.values, name="resto", marker_color="#c3c2b7", showlegend=False,
                         hovertemplate="%{x}: %{y:.1f} %<extra>resto</extra>"), 1, 2)
    fig.update_yaxes(title_text="% de cada grupo", row=1, col=2)
    fig.update_layout(barmode="group", title=f"Outliers multivariados: {m['pct_ambos']:.2f} % los marcan ambos métodos "
                                             f"(Jaccard {m['jaccard']:.2f})")
    fig.update_layout(legend_itemsizing="constant")
    return _estilo_sub(fig, 480)


def fig_silueta():
    m = _json("multivariado.json")
    s = pd.Series(m["silueta"]).rename(index=int).sort_index()
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines+markers", line=dict(color=AZUL, width=3),
                               marker=dict(size=12, color=[NARANJA if k == m["k_opt"] else AZUL for k in s.index]),
                               hovertemplate="k = %{x}: silueta %{y:.3f}<extra></extra>"))
    fig.add_hrect(y0=0.5, y1=1, fillcolor=VERDE, opacity=0.07, line_width=0, annotation_text="estructura clara (> 0.5)")
    fig.update_xaxes(title="número de grupos k", dtick=1)
    fig.update_yaxes(title="silueta media", range=[0, 0.7])
    fig.update_layout(title=f"K-Means: la mejor silueta es {s.max():.2f} (k = {m['k_opt']}): grupos débiles")
    return _estilo(fig, alto=420, leyenda=False)


def fig_kmeans(k=None):
    m = _json("multivariado.json")
    k = k or m["k_opt"]
    s = _pkl("multivariado_muestra.pkl.gz")
    pal = [AZUL, NARANJA, VERDE, "#9085e9", "#eda100", ROJO, "#e87ba4", GRIS]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.12,
                        subplot_titles=(f"Grupos en el espacio PC1–PC2 (k = {k})", "Estrato dentro de cada grupo"))
    for g in sorted(s[f"k{k}"].unique()):
        a = s[s[f"k{k}"] == g]
        fig.add_trace(go.Scattergl(x=a.PC1, y=a.PC2, mode="markers", name=f"grupo {g + 1}",
                                   marker=dict(size=4, color=pal[g % len(pal)], opacity=0.55), hoverinfo="skip"), 1, 1)
    t = pd.crosstab(s[f"k{k}"], s.estrato, normalize="index")
    for e in CLASES:
        fig.add_trace(go.Bar(x=[f"grupo {g + 1}" for g in t.index], y=t.get(e, 0), name=f"E{e}", marker_color=COLOR_ESTRATO[e],
                             showlegend=False, hovertemplate="%{x}: %{y:.0%}<extra>E" + str(e) + "</extra>"), 1, 2)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_layout(barmode="stack", title=f"K-Means con k = {k}: los grupos mezclan estratos")
    fig.update_layout(legend_itemsizing="constant")
    return _estilo_sub(fig, 470)


def fig_tsne(color="estrato"):
    t = _pkl("tsne.pkl.gz")
    fig = go.Figure()
    if color == "estrato":
        for k in CLASES:
            a = t[t.estrato == k]
            fig.add_trace(go.Scattergl(x=a.d1, y=a.d2, mode="markers", name=f"E{k}",
                                       marker=dict(size=6, color=COLOR_ESTRATO[k], opacity=0.75), hoverinfo="skip"))
    else:
        col = "cluster" if color == "cluster" else "condicion_predio"
        pal = [AZUL, NARANJA, VERDE, "#9085e9", "#eda100", ROJO, GRIS]
        for c, p in zip(sorted(t[col].unique(), key=str), pal):
            a = t[t[col] == c]
            fig.add_trace(go.Scattergl(x=a.d1, y=a.d2, mode="markers",
                                       name=f"grupo {c + 1}" if col == "cluster" else str(c),
                                       marker=dict(size=6, color=p, opacity=0.75), hoverinfo="skip"))
    fig.update_xaxes(showticklabels=False, title="")
    fig.update_yaxes(showticklabels=False, title="", showgrid=False)
    fig.update_layout(title="t-SNE de las variables físicas (3 000 viviendas): mapa de vecindad en 2D")
    fig.update_layout(legend_itemsizing="constant")
    return _estilo(fig, alto=540)


def fig_lisa(modo="lisa"):
    li = _pkl("lisa.pkl.gz")
    fig = go.Figure()
    if modo == "lisa":
        for c, col, op in [("no significativo", "#d9d8d3", 0.4), ("LL", "#2c7bb6", 0.8), ("HH", "#d7191c", 0.8),
                           ("LH", "#abd9e9", 0.95), ("HL", "#fdae61", 0.95)]:
            a = li[li.cuadrante == c]
            nombre = {"HH": "alto rodeado de alto (HH)", "LL": "bajo rodeado de bajo (LL)", "HL": "alto entre bajos (HL)",
                      "LH": "bajo entre altos (LH)"}.get(c, c)
            fig.add_trace(go.Scattermap(lat=a.lat, lon=a.lon, mode="markers", name=f"{nombre} ({len(a) / len(li):.0%})",
                                        marker=dict(size=5 if c in ("HL", "LH") else 4, color=col, opacity=op),
                                        customdata=a.estrato, hovertemplate="estrato %{customdata}<extra>" + nombre + "</extra>"))
        titulo = "LISA (Moran local): clusters y atípicos espaciales del estrato"
    else:
        fig.add_trace(go.Scattermap(lat=li.lat, lon=li.lon, mode="markers",
                                    marker=dict(size=4, color=li.gi_z.clip(-6, 6), colorscale=[[0, "#2c7bb6"], [0.5, "#f0efec"], [1, "#d7191c"]],
                                                cmin=-6, cmax=6, colorbar=dict(title="z Gi*")),
                                    customdata=np.c_[li.gi_z, li.estrato],
                                    hovertemplate="Gi* z = %{customdata[0]:.1f}<br>estrato %{customdata[1]}<extra></extra>"))
        titulo = "Getis-Ord Gi*: puntos calientes (rojo) y fríos (azul) del estrato"
    return _layout_mapa(fig, titulo, alto=580)


def fig_maup():
    t = tabla_e1("08_8.6_1")
    t["c"], t["I"] = t.celda_km.map(_num), t.moran_I.map(_num)
    fig = go.Figure(go.Scatter(x=t.c, y=t.I, mode="lines+markers+text", text=[f"{v:.2f}" for v in t.I],
                               textposition="top center", line=dict(color=AZUL, width=3), marker=dict(size=11),
                               customdata=t.n_celdas.map(_num),
                               hovertemplate="celda %{x} km: I = %{y:.3f} (%{customdata:,} celdas)<extra></extra>"))
    fig.update_xaxes(title="tamaño de la celda de agregación (km)", type="log")
    fig.update_yaxes(title="I de Moran del estrato medio", range=[0, 1.05])
    fig.update_layout(title="Efecto de la escala (MAUP): la autocorrelación depende de cómo se agregue")
    return _estilo(fig, alto=420, leyenda=False)


def fig_auc_univariado():
    t = tabla_e1("09_9.2_1", indice=False)
    ce = [c for c in t.columns if c.startswith("AUC espacial")][0]
    ca = [c for c in t.columns if c.startswith("AUC aleatorio")][0]
    t["e"], t["a"] = t[ce].map(_num), t[ca].map(_num)
    t = t.sort_values("e")
    fig = go.Figure([go.Bar(y=[et(v) for v in t.variable], x=t.e, orientation="h", name="validación espacial",
                            marker_color=AZUL, hovertemplate="%{y}: %{x:.3f}<extra>espacial</extra>"),
                     go.Scatter(y=[et(v) for v in t.variable], x=t.a, mode="markers", name="validación aleatoria",
                                marker=dict(size=11, color=ROJO, symbol="diamond"),
                                hovertemplate="%{y}: %{x:.3f}<extra>aleatoria</extra>")])
    fig.add_vline(x=0.5, line_dash="dot", line_color="#8a8984", annotation_text="azar")
    fig.update_xaxes(title="AUC uno-contra-resto (macro) de cada variable sola", range=[0.45, 1])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="Auditoría de fuga: ninguna variable predice sola el estrato de forma sospechosa")
    return _estilo(fig, alto=560)


# ===========================================================================
# MODELO BASE (Entregable 1, capítulo 10; Entregable 2, capítulo 3)
# ===========================================================================
def fig_grid():
    v3 = F.BASE / "grid_cv.json"
    g = json.loads(v3.read_text(encoding="utf-8")) if v3.exists() else _json("grid_cv.json")
    if not g:
        return None
    d = pd.DataFrame(g)
    d["m"] = d.f1_cv.map(np.mean)
    d["se"] = d.f1_cv.map(lambda v: np.std(v) / np.sqrt(len(v)))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("A: solo variables físicas", "B: físicas + ubicación"),
                        shared_yaxes=True, horizontal_spacing=0.06)
    for i, conj in enumerate(["A_fisicas", "B_fisicas+ubicacion"], start=1):
        for pesos, col in [("sin pesos", "#8a8984"), ("balanced", AZUL)]:
            a = d[(d.conjunto == conj) & (d.pesos == pesos)].sort_values("C")
            fig.add_trace(go.Scatter(x=a.C, y=a.m, mode="lines+markers", name="pesos balanceados" if pesos == "balanced" else "sin pesos",
                                     line=dict(color=col, width=2), marker=dict(size=10),
                                     error_y=dict(array=a.se, color=col, thickness=1.2), showlegend=i == 1,
                                     hovertemplate="C = %{x}: F1 %{y:.3f}<extra>" + pesos + "</extra>"), 1, i)
    fig.update_xaxes(type="log", title_text="C (inverso de la regularización)", tickvals=[0.01, 0.1, 1, 10])
    fig.update_yaxes(title_text="F1 macro (CV kNNDM)" if v3.exists() else "F1 macro (CV espacial)", row=1, col=1)
    fig.update_layout(title="Búsqueda de hiperparámetros: superficie plana → regla 1-SE elige C = 0.01")
    return _estilo_sub(fig, 440)


def fig_importancia_permutacion():
    v3 = F.BASE / "importancia.json"
    if v3.exists():
        d = json.loads(v3.read_text(encoding="utf-8"))
        t = pd.DataFrame({"c": d["caida"], "d": d["desv"]}, index=d["variable"]).sort_values("c")
    else:
        t = tabla_e1("10_10.6_2")
        t["c"], t["d"] = t["caída F1 macro"].map(_num), t["desv."].map(_num)
        t = t.sort_values("c")
    fig = go.Figure(go.Bar(x=t.c, y=[et(i) for i in t.index], orientation="h", marker_color=AZUL,
                           error_x=dict(array=t.d, color="#52514e"),
                           hovertemplate="%{y}: el F1 cae %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="caída del F1 macro al barajar la variable (test)")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="Importancia por permutación (fold 1 de la CV " + ("kNNDM)" if v3.exists() else "espacial)"))
    return _estilo(fig, alto=480, leyenda=False)


def fig_odds():
    v3 = F.BASE / "odds.json"
    if v3.exists():
        d = json.loads(v3.read_text(encoding="utf-8"))
        t = pd.DataFrame({"o": d["odds"]}, index=d["variable"]).sort_values("o")
    else:
        t = tabla_e1("10_10.6_1")
        col = [c for c in t.columns if c.startswith("odds")][0]
        t["o"] = t[col].map(_num)
        t = t.sort_values("o")
    # Barras que salen de 1 (sin efecto): con base=1, el largo es odds − 1, así que las odds < 1 van hacia la izquierda.
    fig = go.Figure(go.Bar(x=t.o - 1, y=[F._nombre_variable(str(i)) for i in t.index], orientation="h", base=1,
                           marker_color=[AZUL if o >= 1 else ROJO for o in t.o], customdata=t.o,
                           hovertemplate="%{y}: odds ×%{customdata:.3g}<extra></extra>"))
    fig.add_vline(x=0, line_color="#52514e")  # en un eje log, las líneas van en log10: log10(1) = 0
    lo, hi = float(np.log10(t.o.min())), float(np.log10(t.o.max()))
    fig.update_xaxes(type="log", title="odds 6 frente a 1 por +1 desv. estándar (escala log)",
                     range=[min(lo, 0) - 0.15, max(hi, 0) + 0.15])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="Cómo leer los coeficientes: odds ratio del estrato 6 frente al 1")
    return _estilo(fig, alto=520, leyenda=False)


# ===========================================================================
# MÁS ALLÁ DEL MODELO BASE (Entregable 2, capítulos 4 y 5)
# ===========================================================================
def fig_cv_pareado():
    a = F.resultados()["avanzados"]["cv_pareado"]
    fig = go.Figure()
    for m, d in a.items():
        dif = d["diferencias_por_fold"]
        fig.add_trace(go.Bar(x=[f"fold {i + 1}" for i in range(len(dif))], y=dif, name=F.NOMBRE_CORTO[m],
                             marker_color=F.COLOR_MODELO[m],
                             hovertemplate="%{x}: %{y:+.3f}<extra>" + F.NOMBRE_CORTO[m] +
                                           f" (media {d['media']:+.3f}, SE {d['se']:.3f})</extra>"))
    fig.add_hline(y=0, line_color="#52514e")
    fig.update_yaxes(title="F1 del modelo − F1 de la logística B")
    fig.update_layout(barmode="group", title="Criterio de selección: diferencia con la logística en cada fold de CV")
    return _estilo(fig, alto=440)


def fig_criterio_cv_test():
    """Ventaja en CV (con su error estándar) frente a diferencia en test (con su IC 95 %)."""
    r = F.resultados()["avanzados"]
    fig = go.Figure()
    # Cada etiqueta va en una diagonal distinta para no cruzarse con las barras de error ni con las otras etiquetas.
    lado = {"XGBoost": ("left", "top"), "SVM RBF + TPE": ("left", "bottom"),
            "Rotation Forest": ("right", "bottom"), "XGBoost geográfico": ("right", "top")}
    for m in F.AVANZADOS:
        cv = r["cv_pareado"][m]
        t = r["comparaciones_f1"][f"{m} − Logística B"]
        d, lo, hi = float(t["diferencia"]), float(t["IC95_inf"]), float(t["IC95_sup"])
        xa, ya = lado.get(F.NOMBRE_CORTO[m], ("left", "bottom"))
        fig.add_annotation(x=cv["media"], y=d, text=f"<b>{F.NOMBRE_CORTO[m]}</b>", showarrow=False, xanchor=xa, yanchor=ya,
                           xshift=10 if xa == "left" else -10, yshift=6 if ya == "bottom" else -6,
                           font=dict(color=F.COLOR_MODELO[m], size=14), bgcolor="rgba(255,255,255,0.85)")
        fig.add_trace(go.Scatter(x=[cv["media"]], y=[d], mode="markers",
                                 marker=dict(size=16, color=F.COLOR_MODELO[m], line=dict(color="white", width=2)),
                                 error_x=dict(array=[cv["se"]], color=F.COLOR_MODELO[m], thickness=2),
                                 error_y=dict(type="data", symmetric=False, array=[hi - d], arrayminus=[d - lo],
                                              color=F.COLOR_MODELO[m], thickness=2),
                                 name=F.NOMBRE_CORTO[m],
                                 hovertemplate=f"{F.NOMBRE_CORTO[m]}<br>CV: {cv['media']:+.3f} (SE {cv['se']:.3f})"
                                               f"<br>test: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]<extra></extra>"))
    fig.add_hline(y=0, line_color="#52514e", line_dash="dot")
    fig.add_vline(x=0, line_color="#52514e", line_dash="dot")
    fig.add_annotation(x=0.06, y=0.07, text="mejor en CV y en test", showarrow=False, font=dict(color=VERDE))
    fig.add_annotation(x=0.06, y=-0.185, text="mejor en CV, peor en test", showarrow=False, font=dict(color=ROJO))
    fig.update_xaxes(title="ventaja de F1 en CV frente a la logística (± 1 error estándar)", range=[-0.06, 0.09])
    fig.update_yaxes(title="diferencia de F1 en test (IC 95 %)", range=[-0.2, 0.1])
    fig.update_layout(title="Lo que promete la CV frente a lo que entrega el test")
    return _estilo(fig, alto=500, leyenda=False)


NOMBRE_VEC = {"B": "Logística B", "B + vecindad (300 m y 1 km)": "B + vecindad (300 m y 1 km)",
              "B + vecindad (solo 300 m)": "B + vecindad (300 m)", "B + vecindad (solo 1 km)": "B + vecindad (1 km)",
              "A": "A: solo físicas", "A + vecindad (sin coordenadas)": "A + vecindad (sin coordenadas)"}


def fig_vecindad_forest():
    c = _json("comparacion_zonas_nuevas.json", "vecindad")
    comp = c["comparaciones_vs_B"]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=("Validación cruzada (± 1 error estándar)", "Test (IC 95 % por bloques)"))
    nombres = list(comp)
    y = [NOMBRE_VEC.get(n, n) for n in nombres]
    for col, (clave_m, clave_e) in enumerate([("cv_dif_media", "cv_se_pareado"), ("test_f1", None)], start=1):
        if col == 1:
            x = [comp[n]["cv_dif_media"] for n in nombres]
            err = dict(array=[comp[n]["cv_se_pareado"] for n in nombres], color="#52514e", thickness=2)
            colores = [VERDE if comp[n]["reemplaza_a_B"] else GRIS for n in nombres]
        else:
            x = [comp[n]["test_f1"]["diferencia"] for n in nombres]
            hi = [comp[n]["test_f1"]["IC95_sup"] for n in nombres]
            lo = [comp[n]["test_f1"]["IC95_inf"] for n in nombres]
            err = dict(type="data", symmetric=False, array=np.array(hi) - np.array(x), arrayminus=np.array(x) - np.array(lo),
                       color="#52514e", thickness=2)
            colores = [AZUL if l > 0 else GRIS for l in lo]
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=15, color=colores), error_x=err,
                                 showlegend=False, hovertemplate="%{y}: %{x:+.3f}<extra></extra>"), 1, col)
        fig.add_vline(x=0, line_color="#52514e", line_dash="dot", row=1, col=col)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="Δ F1 frente a la logística B")
    fig.update_layout(title="Variables de vecindad física en zonas nuevas: ¿mejoran a la logística B?")
    return _estilo(fig, alto=430, leyenda=False)


def fig_vecindad_15folds():
    c = _json("cv_15_folds.json", "vecindad")
    d = c["comparaciones_vs_B"]["B + vecindad (300 m y 1 km)"]["dif_F1_por_fold"]
    orden = np.argsort(d)
    fig = go.Figure(go.Bar(x=[f"fold {i + 1}" for i in orden], y=np.array(d)[orden],
                           marker_color=[VERDE if v > 0 else ROJO for v in np.array(d)[orden]],
                           hovertemplate="%{x}: %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=np.mean(d), line_dash="dash", line_color=AZUL, annotation_text=f"media {np.mean(d):+.3f}")
    fig.update_yaxes(title="Δ F1 (B + vecindad − B)")
    fig.update_layout(title=f"CV de 15 folds: la vecindad mejora en {int((np.array(d) > 0).sum())} de 15 zonas, sin margen suficiente")
    return _estilo(fig, alto=420, leyenda=False)


def fig_vecindad_donde(conj="test", tipo="heterogeneidad"):
    z = _json("zonas.json", "vecindad")
    clave = f"{tipo}|{'test' if conj == 'test' else 'train (CV 15 folds)'}"
    d = pd.DataFrame(z[clave])
    g = d.columns[0]
    fig = go.Figure([go.Bar(x=d[g], y=d.acc_B, name="Logística B", marker_color="#c3c2b7",
                            hovertemplate="%{x}: %{y:.2f}<extra>B</extra>"),
                     go.Bar(x=d[g], y=d.acc_Bvec, name="B + vecindad", marker_color=AZUL,
                            customdata=d.n, hovertemplate="%{x}: %{y:.2f} (%{customdata:,.0f} viviendas)<extra>B + vecindad</extra>")])
    fig.update_yaxes(title="accuracy", range=[0, 1.05])
    lugar = "test" if conj == "test" else "CV de 15 folds"
    tit = "según qué tan mezclado es el entorno (desv. del estrato a 300 m)" if tipo == "heterogeneidad" else \
        "según qué tan atípica es la vivienda frente a su entorno"
    fig.update_layout(barmode="group", title=f"¿Dónde ayuda la vecindad? {tit} — {lugar}")
    return _estilo(fig, alto=440)


def fig_vecindad_bloques():
    z = _json("zonas.json", "vecindad")
    b = pd.DataFrame(z["bloques"])
    r = z["correlaciones_bloques"]["y_km"] if "y_km" in z["correlaciones_bloques"] else None
    fig = go.Figure()
    for conj, col, sym in [("train (CV 15 folds)", AZUL, "circle"), ("test", NARANJA, "diamond")]:
        a = b[b.conj == conj]
        fig.add_trace(go.Scatter(x=a.y_km, y=-a.d_MAE, mode="markers", name="bloque de " + ("CV" if conj != "test" else "test"),
                                 marker=dict(size=np.clip(np.sqrt(a.n) / 4, 8, 32), color=col, symbol=sym, opacity=0.8,
                                             line=dict(color="white", width=1)),
                                 customdata=np.c_[a.n, a.estrato_medio],
                                 hovertemplate="y = %{x:.1f} km<br>mejora de MAE %{y:+.2f}<br>%{customdata[0]:,} viviendas,"
                                               " estrato medio %{customdata[1]:.1f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#52514e", line_dash="dot")
    fig.update_xaxes(title="posición norte-sur del bloque (km; más alto = más al norte)")
    fig.update_yaxes(title="mejora del MAE con la vecindad (positivo = ayuda)")
    t = f"ρ = {r['rho']:.2f}, p = {r['p']:.3f}" if r else ""
    fig.update_layout(title=f"La vecindad ayuda más en los bloques del norte ({t})")
    return _estilo(fig, alto=460)


def zonas_conocidas(agrupar="edificio"):
    return _json("zonas_conocidas.json" if agrupar == "edificio" else "zonas_conocidas_manzana.json", "vecindad")


ORDEN_ZC = ["Moda por zona (2 km)", "Logística B", "Logística B + vecindad", "Logística B + vecindad + estrato de vecinos",
            "XGBoost", "XGBoost geográfico", "Vecinos más cercanos (k = 15)"]
COLOR_ZC = {"Moda por zona (2 km)": NARANJA, "Logística B": AZUL, "Logística B + vecindad": "#6da7ec",
            "Logística B + vecindad + estrato de vecinos": "#9ec5f4", "XGBoost": "#e87ba4", "XGBoost geográfico": "#4a3aa7",
            "Vecinos más cercanos (k = 15)": "#52514e"}


def fig_zonas_conocidas(metrica="F1_macro"):
    fig = go.Figure()
    for agrupar, patron, nombre in [("edificio", "", "ocultando edificios"), ("manzana", "/", "ocultando manzanas completas")]:
        z = zonas_conocidas(agrupar)
        if not z:
            continue
        mods = [m for m in ORDEN_ZC if m in z["modelos"]]
        vals = [z["modelos"][m][metrica] for m in mods]
        fig.add_trace(go.Bar(x=[m.replace(" + ", "<br>+ ") for m in mods], y=vals, name=nombre,
                             marker_color=[COLOR_ZC[m] for m in mods], marker_pattern_shape=patron,
                             marker_line_color="white", text=[f"{v:.2f}" for v in vals], textposition="outside",
                             textfont_size=12,
                             hovertemplate="%{x}: %{y:.3f}<extra>" + nombre + "</extra>"))
    fig.update_yaxes(title={"F1_macro": "F1 macro", "MAE": "MAE ordinal", "accuracy": "accuracy"}[metrica], rangemode="tozero")
    fig.update_layout(barmode="group", title="Zonas conocidas: predecir viviendas ocultas dentro de barrios con datos")
    return _estilo(fig, alto=480)


def fig_escenarios():
    """F1 de cada modelo en zonas nuevas (test espacial) frente a zonas conocidas."""
    r = F.resultados()["resultados_test"]["f1_macro"]
    z = zonas_conocidas("edificio")
    zm = zonas_conocidas("manzana")
    pares = [("Logística B", "Logística B_fisicas+ubicacion"), ("XGBoost", "XGBoost"), ("XGBoost geográfico", "XGBoost geográfico"),
             ("Moda por zona (2 km)", "Moda por zona")]
    # Valores cercanos: una etiqueta arriba y la otra abajo para que no se monten.
    posicion = {"Logística B": "top right", "Moda por zona (2 km)": "bottom right", "XGBoost": "bottom right",
                "XGBoost geográfico": "top right"}
    fig = go.Figure()
    for nombre_zc, nombre_r in pares:
        ys = [r[nombre_r], z["modelos"][nombre_zc]["F1_macro"]]
        xs = ["zonas nuevas<br>(test espacial)", "zonas conocidas<br>(edificios ocultos)"]
        if zm:
            ys.insert(1, zm["modelos"][nombre_zc]["F1_macro"])
            xs.insert(1, "zonas conocidas<br>(manzanas ocultas)")
        pos = [posicion[nombre_zc]] * len(ys)
        if nombre_zc in ("XGBoost", "XGBoost geográfico"):
            pos[0] = "middle right"  # en zonas nuevas los cuatro valores están a menos de 0.15
        moda = nombre_zc.startswith("Moda")
        # La moda del test espacial usa zonas de 4 km (como el E1); la de zonas conocidas, de 2 km.
        nombre = "Moda por zona (4 km en test; 2 km en zonas conocidas)" if moda else nombre_zc
        zona = ["zonas de 4 km"] + ["zonas de 2 km"] * (len(ys) - 1) if moda else [""] * len(ys)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers+text", name=nombre, text=[f"{v:.2f}" for v in ys],
                                 textposition=pos, customdata=zona, cliponaxis=False,
                                 line=dict(color=COLOR_ZC[nombre_zc], width=3, dash="dot" if moda else "solid"),
                                 marker=dict(size=13), hovertemplate="%{x}: %{y:.3f} %{customdata}<extra>" + nombre_zc.split(" (")[0] + "</extra>"))
    fig.update_yaxes(title="F1 macro", range=[0.2, 0.9])
    fig.update_layout(title="El mejor modelo depende del escenario")
    _estilo(fig, alto=520)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0), margin=dict(t=60, r=60, b=130))
    return fig


def recuperacion_por_umbral(umbral=250):
    """Viviendas sin coordenadas propias que se recuperarían con un umbral de dispersión dado, por estrato."""
    m = _json("manzanas.json")
    d = np.array(m["candidatas"]["dist_max_m"])
    ok = d <= umbral
    rec = {int(k): int(np.array(v)[ok].sum()) for k, v in m["candidatas"]["sin_coord_por_estrato"].items()}
    return {"recuperadas": rec, "total": sum(rec.values()), "manzanas": int(ok.sum()), "candidatas": len(d)}


def fig_recuperacion_umbral(umbral=250):
    m = _json("manzanas.json")
    r = recuperacion_por_umbral(umbral)
    c = pd.DataFrame(m["cobertura"]).T.astype(float)
    c.index = c.index.astype(int)
    v1 = c.v1 / c.total
    nuevo = (c.v1 + pd.Series(r["recuperadas"]).reindex(c.index).fillna(0)) / c.total
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.1,
                        subplot_titles=("Dispersión de cada manzana", "Cobertura por estrato con este umbral"))
    d = np.array(m["dist_max_m"])
    fig.add_trace(go.Histogram(x=np.clip(d, 0, 1000), xbins=dict(start=0, end=1000, size=20), marker_color=AZUL_CLARO, showlegend=False,
                               hovertemplate="%{x} m: %{y:,} manzanas<extra></extra>"), 1, 1)
    fig.add_vline(x=umbral, line_color=NARANJA, line_width=3, line_dash="dash", row=1, col=1)
    x = [f"E{k}" for k in c.index]
    fig.add_trace(go.Bar(x=x, y=v1, name="solo coordenadas propias", marker_color="#c3c2b7",
                         hovertemplate="%{x}: %{y:.1%}<extra>sin recuperar</extra>"), 1, 2)
    fig.add_trace(go.Bar(x=x, y=nuevo, name=f"+ manzanas ≤ {umbral} m", marker_color=[COLOR_ESTRATO[k] for k in c.index],
                         text=[f"{v:.0%}" for v in nuevo], textposition="outside",
                         hovertemplate="%{x}: %{y:.1%}<extra>con recuperación</extra>"), 1, 2)
    fig.update_xaxes(title_text="distancia del edificio más lejano al centroide (m)", row=1, col=1)
    fig.update_yaxes(title_text="manzanas", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.15], row=1, col=2)
    tot = (c.v1.sum() + r["total"]) / c.total.sum()
    fig.update_layout(barmode="group", title=f"Umbral {umbral} m: se ubican {r['total']:,} viviendas más → cobertura {tot:.1%}".replace(",", " "))
    return _estilo_sub(fig, 450)


# ---------------------------------------------------------------------------
# Versión 3: validación por vecinos más cercanos (kNNDM) y vecindad con grafo KNN
# (datos/proyecto/v3_e1.json: cap. 14 del Entregable 1; datos/vecindad/vecindad_knn.json: vecindad/07)
# ---------------------------------------------------------------------------
def v3_e1():
    return _json("v3_e1.json")


def vecindad_knn():
    return _json("vecindad_knn.json", "vecindad")


ESTILO_DIST = {"Escenario: zona nueva de 2 km (test → todo train)": ("#111111", "solid", 4, "Escenario: zona nueva de 2 km"),
               "Test de la versión 2 (test → train tras buffer)": (ROJO, "dot", 3, "Test v2 (con buffer de 1 km)"),
               "CV de la versión 2 (bloques + buffer de 1 km)": (ROJO, "dash", 3, "CV v2 (bloques + buffer)"),
               "CV aleatoria por edificios": (GRIS, "dash", 3, "CV aleatoria por edificios"),
               "CV kNNDM (versión 3)": (AZUL, "solid", 3, "CV kNNDM (v3)")}


def fig_v3_distancias():
    d = v3_e1()
    if not d:
        return None
    x = d["distancias"]["ecdf_rejilla_km"]
    q = d["distancias"]["cuantiles"]
    fig = go.Figure()
    for k, F_ in d["distancias"]["ecdf"].items():
        col, dash, ancho, nombre = ESTILO_DIST.get(k, (GRIS, "solid", 2, k))
        med = q[k]["mediana (km)"]
        fig.add_trace(go.Scatter(x=x, y=F_, mode="lines", name=f"{nombre} (mediana {med:.2f} km)",
                                 line=dict(color=col, dash=dash, width=ancho),
                                 hovertemplate="%{x:.2f} km: %{y:.0%}<extra>" + nombre + "</extra>"))
    fig.update_xaxes(title="distancia a la vivienda de estrato conocido más cercana (km)", range=[0, 2.5])
    fig.update_yaxes(title="viviendas (acumulado)", tickformat=".0%", range=[0, 1.02])
    fig.update_layout(title="¿A qué distancia se evalúa cada diseño?")
    _estilo(fig, alto=540)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="left", x=0),
                      margin=dict(t=60, b=190))
    return fig


def fig_v3_knndm_q():
    d = v3_e1()
    if not d:
        return None
    t = pd.DataFrame(d["knndm"]["tabla_q"])
    q = d["knndm"]["q"]
    fig = go.Figure(go.Scatter(x=t.q, y=t.W_km, mode="lines+markers", line=dict(color=AZUL, width=3), marker=dict(size=9),
                               customdata=t.mediana_cv_km, showlegend=False,
                               hovertemplate="q = %{x}: Wasserstein %{y:.3f} km<br>mediana de la CV %{customdata:.2f} km"
                                             "<extra></extra>"))
    fig.add_vline(x=q, line_color=ROJO, line_dash="dash")
    fig.add_annotation(x=np.log10(q), xref="x", y=float(t.W_km.max()), text=f"q elegido = {q}", showarrow=False,
                       xanchor="left", xshift=6, font=dict(color=ROJO))
    fig.add_hline(y=d["knndm"]["W_cv_aleatoria_km"], line_color=GRIS, line_dash="dot")
    fig.add_annotation(x=1, xref="paper", y=d["knndm"]["W_cv_aleatoria_km"], text="CV aleatoria", showarrow=False,
                       xanchor="right", yshift=10, font=dict(color=GRIS))
    fig.update_xaxes(title="número de grupos de K-Means (q): pocos = bloques grandes, muchos = casi aleatorio", type="log")
    fig.update_yaxes(title="Wasserstein al escenario (km)", rangemode="tozero")
    fig.update_layout(title="Elegir q: la validación más parecida al escenario")
    return _estilo(fig, alto=430, leyenda=False)


ORDEN_V3 = ["B v2 (con buffer)", "B v3 (kNNDM, sin buffer)", "B v3 + umbrales F1", "Ordinal v3 (kNNDM, sin buffer)",
            "Ordinal v3 + umbrales F1"]
NOMBRE_V3 = {"B v2 (con buffer)": "Logística B v2<br>(bloques + buffer)",
             "B v3 (kNNDM, sin buffer)": "Logística B v3<br>(kNNDM, sin buffer)",
             "B v3 + umbrales F1": "B v3<br>+ umbrales F1", "Ordinal v3 (kNNDM, sin buffer)": "Ordinal v3",
             "Ordinal v3 + umbrales F1": "Ordinal v3<br>+ umbrales F1"}


def fig_v3_modelos(metrica="f1", incluir_v2=True):
    d = v3_e1()
    if not d:
        return None
    r = d["resumen"]
    mods = [m for m in ORDEN_V3 if m in r and (incluir_v2 or not m.startswith("B v2"))]
    xs = [NOMBRE_V3[m] for m in mods]
    fig = go.Figure()
    if metrica == "f1":
        test = [r[m]["F1 macro test"] for m in mods]
        lo = [r[m]["F1 macro test"] - r[m]["IC95 inf"] for m in mods]
        hi = [r[m]["IC95 sup"] - r[m]["F1 macro test"] for m in mods]
        fig.add_trace(go.Bar(x=xs, y=test, name="test (IC 95 % por bloques)", marker_color=AZUL,
                             error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color="#52514e"),
                             text=[f"{v:.3f}" for v in test], textposition="inside", insidetextanchor="start",
                             hovertemplate="%{x}: F1 test %{y:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=xs, y=[r[m]["F1 macro CV"] for m in mods], mode="markers", name="validación cruzada",
                                 marker=dict(size=16, color=NARANJA, symbol="diamond", line=dict(color="white", width=2)),
                                 hovertemplate="%{x}: F1 CV %{y:.3f}<extra></extra>"))
        fig.update_yaxes(title="F1 macro", range=[0, 0.7])
        titulo = "F1 macro: validación cruzada frente a test (mismo test de siempre)"
    else:
        col = {"mae": ("MAE test", "MAE ordinal en test (menos es mejor)"), "e4": ("recall E4", "recall del estrato 4 en test"),
               "kappa": ("kappa test", "kappa cuadrático en test")}[metrica]
        vals = [r[m][col[0]] for m in mods]
        fig.add_trace(go.Bar(x=xs, y=vals, marker_color=AZUL, text=[f"{v:.3f}" for v in vals], textposition="outside",
                             showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
        fig.update_yaxes(title=col[1], rangemode="tozero")
        titulo = col[1][0].upper() + col[1][1:]
    fig.update_layout(title=titulo)
    return _estilo(fig, alto=470)


def fig_vecindad_knn(escenario="zonas nuevas"):
    vk = vecindad_knn()
    if not vk:
        return None
    c = vk["cv"][escenario]
    mods = c["modelos"]
    kx, ky = c["kx_elegido"], c["ky_elegido"]
    filas = [(n, m["dif_F1_vs_B"], m["SE_dif"], m["F1_macro"]) for n, m in mods.items() if n != "Logística B"]
    filas.sort(key=lambda f: f[1])
    elegidos = {f"Logística B + vecindad KNN (k = {kx})", f"Logística B + estrato de vecinos (k = {ky})",
                f"Logística B + vecindad KNN (k = {kx}) + estrato de vecinos (k = {ky})"}
    col = [NARANJA if "radios" in f[0] else (AZUL if f[0] in elegidos else AZUL_CLARO) for f in filas]
    fig = go.Figure(go.Scatter(x=[f[1] for f in filas], y=[f[0].replace("Logística B + ", "B + ") for f in filas],
                               mode="markers", marker=dict(size=13, color=col),
                               error_x=dict(type="data", array=[f[2] for f in filas], color="#52514e", thickness=2),
                               customdata=[f[3] for f in filas], showlegend=False,
                               hovertemplate="%{y}<br>Δ F1 vs B: %{x:+.3f} (± 1 SE)<br>F1 CV: %{customdata:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#52514e", line_dash="dot")
    base = mods["Logística B"]["F1_macro"]
    fig.update_xaxes(title="Δ F1 frente a B (CV kNNDM, ± 1 SE)")
    fig.update_layout(title=f"{escenario.capitalize()}: F1 de la logística B = {base:.3f} (naranja: radios del cap. 5)")
    return _estilo(fig, alto=470, leyenda=False)


def fig_moran_k():
    vk = vecindad_knn()
    if not vk:
        return None
    m = pd.DataFrame(vk["moran_por_k"])
    fig = go.Figure(go.Scatter(x=m.k, y=m.moran_I, mode="lines+markers+text", text=[f"{v:.2f}" for v in m.moran_I],
                               textposition="top center", line=dict(color=AZUL, width=3), marker=dict(size=10),
                               showlegend=False, hovertemplate="k = %{x}: I = %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="vecinos del grafo (k)", type="log")
    fig.update_yaxes(title="I de Moran del estrato", range=[0, 1])
    fig.update_layout(title="¿Cuántos vecinos? El parecido cae despacio con k")
    return _estilo(fig, alto=400, leyenda=False)


# ---------------------------------------------------------------------------
# ¿Supera el modelo nuevo (XGBoost geográfico) al modelo base? (datos/vecindad/arboles_knndm.json)
# ---------------------------------------------------------------------------
def arboles_knndm():
    return _json("arboles_knndm.json", "vecindad")


def _variante_corta(v):
    v = v.replace("XGBoost geográfico | ", "")
    return {"B": "solo variables del modelo base"}.get(v, v.replace("B + ", "+ "))


def tabla_nuevo_vs_base(escenario="zonas nuevas"):
    """Una fila por variante del XGBoost geográfico: F1 propio, diferencia con el modelo base (logística B v3, sin
    vecindad) y con la logística que recibe las mismas variables."""
    a = arboles_knndm()
    if not a:
        return None
    filas = []
    for var, v in a["xgb_geografico_vs_logistica"][escenario]["variantes"].items():
        base = v["vs Logística B v3 (sin vecindad)"]
        corto = var.replace("XGBoost geográfico | ", "")
        misma_clave = "vs Logística B v3 (sin vecindad)" if corto == "B" else next(
            (k for k in v if k.startswith("vs ") and corto in k and "sin vecindad" not in k),
            "vs Logística B v3 (sin vecindad)")
        misma = v[misma_clave]
        f = {"Variables del XGBoost geográfico": _variante_corta(var), "F1 CV": v["F1_cv"],
             "Δ CV vs modelo base": base["cv_F1"]["media"], "SE": base["cv_F1"]["SE"],
             "folds a favor": f"{base['cv_F1']['folds_a_favor']}/5",
             "¿Supera al base (> 1 SE)?": "Sí" if base["cv_F1"]["supera_por_1SE"] else "No"}
        if v.get("F1_test") is not None:
            t = base["test_F1"]
            f.update({"F1 test": v["F1_test"], "Δ test vs base": t["diferencia"],
                      "IC 95 % test": f"[{t['IC95_inf']:+.3f}, {t['IC95_sup']:+.3f}]"})
        f.update({"Δ CV vs logística con las mismas variables": misma["cv_F1"]["media"],
                  "SE (misma)": misma["cv_F1"]["SE"]})
        filas.append(f)
    return pd.DataFrame(filas)


def fig_nuevo_vs_base(escenario="zonas nuevas", referencia="base"):
    """Diferencias de F1 del XGBoost geográfico frente al modelo base (o frente a la logística con las mismas
    variables): CV kNNDM ± 1 SE y, en zonas nuevas, test con IC 95 % por bloques."""
    a = arboles_knndm()
    if not a:
        return None
    vs = a["xgb_geografico_vs_logistica"][escenario]["variantes"]
    nombres, cv, se, te, lo, hi = [], [], [], [], [], []
    for var, v in vs.items():
        if referencia == "base":
            c = v["vs Logística B v3 (sin vecindad)"]
        else:
            corto = var.replace("XGBoost geográfico | ", "")
            c = v["vs Logística B v3 (sin vecindad)"] if corto == "B" else next(
                (v[k] for k in v if k.startswith("vs ") and corto in k and "sin vecindad" not in k),
                v["vs Logística B v3 (sin vecindad)"])
        nombres.append(_variante_corta(var))
        cv.append(c["cv_F1"]["media"])
        se.append(c["cv_F1"]["SE"])
        t = c.get("test_F1")
        te.append(t["diferencia"] if t else np.nan)
        lo.append(t["diferencia"] - t["IC95_inf"] if t else np.nan)
        hi.append(t["IC95_sup"] - t["diferencia"] if t else np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cv, y=nombres, mode="markers", name="validación cruzada kNNDM (± 1 SE)",
                             marker=dict(size=14, color=AZUL), error_x=dict(type="data", array=se, color=AZUL, thickness=2),
                             hovertemplate="%{y}<br>Δ F1 CV: %{x:+.3f}<extra></extra>"))
    if not np.all(np.isnan(te)):
        fig.add_trace(go.Scatter(x=te, y=nombres, mode="markers", name="test (IC 95 % por bloques)",
                                 marker=dict(size=13, color=NARANJA, symbol="diamond"),
                                 error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color=NARANJA,
                                              thickness=2),
                                 hovertemplate="%{y}<br>Δ F1 test: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#52514e", line_dash="dot")
    ref = "el modelo base (logística B, sin vecindad)" if referencia == "base" else "la logística con las mismas variables"
    fig.update_xaxes(title=f"diferencia de F1 macro: XGBoost geográfico − {ref}")
    fig.update_layout(title=f"¿Supera el modelo nuevo? {escenario.capitalize()}")
    _estilo(fig, alto=430)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0), margin=dict(t=60, b=130))
    return fig
