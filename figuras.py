"""Datos y figuras del dashboard (Entregable 2).

Autores: María Carolina Cantillo Orozco (200179105) y
         Juan Camilo Oñoro Araujo (200177329)

Este módulo es la ÚNICA fuente de las figuras: lo usan la app Dash
(``app.py``) y los cuadernos del Jupyter Book, así que lo que se ve en el
dashboard y lo que se interpreta en el informe es exactamente lo mismo.
Todas las funciones leen los archivos de ``datos/`` generados por
``preparar_datos.py`` (nada se reentrena aquí).
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, roc_curve, auc)

DATOS = Path(__file__).resolve().parent / "datos"
CLASES = [1, 2, 3, 4, 5, 6]

# ---------------------------------------------------------------------------
# Paleta (validada): rampa azul ordinal para el estrato, colores fijos por
# modelo, divergente azul-rojo con gris en el centro.
# ---------------------------------------------------------------------------
COLOR_ESTRATO = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#1c5cab", 5: "#104281", 6: "#0d366b"}
COLOR_MODELO = {
    "Dummy (mayoritaria)": "#bdbcb7", "Dummy (estratificada)": "#a3a29d", "Dummy (uniforme)": "#8a8984",
    "Moda por zona": "#eb6834", "Logística A_fisicas": "#1baf7a", "Logística B_fisicas+ubicacion": "#2a78d6",
}
NOMBRE_CORTO = {
    "Dummy (mayoritaria)": "Dummy mayoritaria", "Dummy (estratificada)": "Dummy estratificada",
    "Dummy (uniforme)": "Dummy uniforme", "Moda por zona": "Moda por zona (2 km)",
    "Logística A_fisicas": "Logística A (físicas)", "Logística B_fisicas+ubicacion": "Logística B (físicas + ubicación)",
}
SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGENTE = [[0, "#c22f2e"], [0.25, "#e98a89"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1, "#1c5cab"]]
# Estilo del mapa base (MapLibre). "white-bg" sirve sin conexión a internet.
ESTILO_MAPA = os.environ.get("ESTILO_MAPA", "carto-positron")
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e6e5e1"

NUMERICAS = {
    "area_construida": "Área construida (m²)", "area_catastral_terreno": "Área de terreno (m²)",
    "total_habitaciones": "Habitaciones", "total_banios": "Baños", "total_plantas": "Pisos del edificio",
    "planta_ubicacion": "Piso de ubicación", "altura": "Altura", "antiguedad": "Antigüedad (años)",
    "dist_centro_km": "Distancia al centro (km)",
}
CATEGORICAS = {"condicion_predio": "Condición del predio (régimen)", "uso": "Uso", "tipo_vivienda": "Tipo de vivienda"}

# Embudo de limpieza del Entregable 1 (cap. 1).
EMBUDO = [("Filas descargadas", 382_597), ("Usos no habitacionales", -36_497), ("Área construida = 0", -11),
          ("Área < 10 m² (parqueaderos, depósitos)", -1_044), ("Registros agregados (edificio en una fila)", -234),
          ("Sin estrato residencial válido", -12_093), ("Viviendas analizadas", 332_718)]


def _estilo(fig, alto=420, leyenda=True):
    """Estilo común: fondo claro, rejilla tenue, tipografía grande."""
    fig.update_layout(template="simple_white", height=alto, margin=dict(l=60, r=20, t=95 if leyenda else 60, b=50),
                      font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15, color=TINTA),
                      title=dict(y=0.985, yanchor="top", x=0.01, xanchor="left", font=dict(size=18)),
                      hoverlabel=dict(font_size=14), showlegend=leyenda,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text="",
                                  traceorder="normal"),
                      paper_bgcolor="white", plot_bgcolor="white")
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", tickfont_color=TINTA_2)
    fig.update_yaxes(showgrid=True, gridcolor=REJILLA, linecolor="#c3c2b7", tickfont_color=TINTA_2)
    return fig


# ---------------------------------------------------------------------------
# Carga (con caché: la app lee los archivos una sola vez)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def viviendas():
    return pd.read_pickle(DATOS / "viviendas.pkl.gz")


@lru_cache(maxsize=None)
def predicciones():
    return pd.read_pickle(DATOS / "test_predicciones.pkl.gz")


@lru_cache(maxsize=None)
def resultados():
    return json.loads((DATOS / "resultados.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def diagnosticos():
    return json.loads((DATOS / "diagnosticos.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def coeficientes():
    return pd.read_pickle(DATOS / "coeficientes.pkl.gz")


def base_eda(particion="train"):
    """Datos para el EDA: por defecto SOLO train, como en el Entregable 1."""
    v = viviendas()
    return v if particion == "todas" else v[v.particion == particion]


MODELOS = list(COLOR_MODELO)

# Modelos avanzados (entrenar_avanzados.py). Colores fijos por modelo, distintos de los del modelo base.
COLOR_MODELO.update({"XGBoost": "#e87ba4", "XGBoost geográfico": "#4a3aa7", "Rotation Forest": "#eda100",
                     "SVM RBF (Nyström) + TPE": "#008300"})
NOMBRE_CORTO.update({"XGBoost": "XGBoost", "XGBoost geográfico": "XGBoost geográfico",
                     "Rotation Forest": "Rotation Forest", "SVM RBF (Nyström) + TPE": "SVM RBF + TPE"})
AVANZADOS = ["XGBoost", "XGBoost geográfico", "Rotation Forest", "SVM RBF (Nyström) + TPE"]


def hay_avanzados():
    return "avanzados" in resultados()


def modelos_disponibles():
    return MODELOS + (AVANZADOS if hay_avanzados() else [])


# ---------------------------------------------------------------------------
# Pestaña 1: contexto
# ---------------------------------------------------------------------------
def kpis():
    v = viviendas()
    r = resultados()
    return {"viviendas": len(v), "edificios": int(v.edificio_id.nunique()),
            "sin_coord": float((~v.tiene_coordenadas).mean()), "moran": diagnosticos()["moran_estrato"]["I"],
            "f1_test": r["resultados_test"]["f1_macro"]["Logística B_fisicas+ubicacion"]}


def fig_embudo():
    etiquetas = [e for e, _ in EMBUDO]
    valores = [v for _, v in EMBUDO]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=etiquetas, x=valores,
        measure=["absolute"] + ["relative"] * 5 + ["total"],
        text=[f"{v:+,}".replace(",", " ") if 0 < i < 6 else f"{v:,}".replace(",", " ") for i, v in enumerate(valores)],
        textposition="outside", connector=dict(line=dict(color="#c3c2b7", width=1)),
        decreasing=dict(marker_color="#e98a89"), totals=dict(marker_color="#2a78d6"),
        increasing=dict(marker_color="#2a78d6"),
        hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, 470_000], title="filas")
    fig.update_layout(title="Embudo de limpieza: de 382 597 filas a 332 718 viviendas")
    return _estilo(fig, alto=430, leyenda=False)


def fig_distribucion_estrato(particion="todas"):
    v = base_eda(particion)
    c = v.estrato_num.value_counts().reindex(CLASES, fill_value=0)
    pct = c / c.sum()
    fig = go.Figure(go.Bar(x=[f"Estrato {k}" for k in CLASES], y=pct.values,
                           marker_color=[COLOR_ESTRATO[k] for k in CLASES],
                           text=[f"{p:.1%}" for p in pct.values], textposition="outside",
                           customdata=c.values, hovertemplate="%{x}: %{y:.1%} (%{customdata:,} viviendas)<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", title="proporción de viviendas", range=[0, pct.max() * 1.18])
    fig.update_layout(title=f"Distribución del estrato (razón mayor/menor = {c.max() / c.min():.1f}:1)")
    return _estilo(fig, alto=400, leyenda=False)


# ---------------------------------------------------------------------------
# Pestaña 2: EDA
# ---------------------------------------------------------------------------
def tabla_resumen_numericas(particion="train"):
    v = base_eda(particion)
    filas = []
    for c, et in NUMERICAS.items():
        x = v[c]
        filas.append({"Variable": et, "Media": x.mean(), "Mediana": x.median(), "Desv. est.": x.std(),
                      "P1": x.quantile(0.01), "P99": x.quantile(0.99), "Asimetría": x.skew(),
                      "Curtosis": x.kurt(), "% faltante": 100 * x.isna().mean()})
    return pd.DataFrame(filas)


def efecto_numerica(var, particion="train"):
    """η² de Kruskal-Wallis: fracción de la variación (en rangos) explicada por el estrato."""
    v = base_eda(particion)[[var, "estrato_num"]].dropna()
    grupos = [g[var].to_numpy() for _, g in v.groupby("estrato_num")]
    h, p = stats.kruskal(*grupos)
    k, n = len(grupos), len(v)
    return {"H": float(h), "p": float(p), "eta2_H": float(max(0.0, (h - k + 1) / (n - k))), "n": n}


def fig_numerica(var, log=True, particion="train"):
    v = base_eda(particion)[[var, "estrato_num"]].dropna()
    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.09,
                        subplot_titles=("Distribución", "Por estrato"))
    x = np.log1p(v[var]) if log else v[var]
    fig.add_trace(go.Histogram(x=x, nbinsx=60, marker_color="#3987e5", marker_line_color="white",
                               marker_line_width=1, hovertemplate="%{x}: %{y:,}<extra></extra>"), 1, 1)
    for k in CLASES:
        xs = v.loc[v.estrato_num == k, var]
        fig.add_trace(go.Box(y=np.log1p(xs) if log else xs, name=f"E{k}", marker_color=COLOR_ESTRATO[k],
                             boxpoints=False, line_width=1.5), 1, 2)
    eje = f"log(1 + {NUMERICAS[var]})" if log else NUMERICAS[var]
    fig.update_xaxes(title_text=eje, row=1, col=1)
    fig.update_yaxes(title_text="viviendas", row=1, col=1)
    fig.update_yaxes(title_text=eje, row=1, col=2)
    ef = efecto_numerica(var, particion)
    fig.update_layout(title=f"{NUMERICAS[var]} — η² (Kruskal-Wallis) = {ef['eta2_H']:.3f}")
    return _estilo(fig, alto=430, leyenda=False)


def cramer_v(tabla):
    chi2 = stats.chi2_contingency(tabla, correction=False)[0]
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))), float(chi2)


def fig_categorica(var, particion="train"):
    v = base_eda(particion)
    t = pd.crosstab(v[var].astype(str), v.estrato_num)
    t = t.loc[t.sum(1).sort_values(ascending=False).index]
    prop = t.div(t.sum(1), axis=0)
    V, _ = cramer_v(t)
    fig = go.Figure()
    for k in CLASES:
        fig.add_trace(go.Bar(y=prop.index, x=prop[k], name=f"Estrato {k}", orientation="h",
                             marker_color=COLOR_ESTRATO[k], marker_line_color="white", marker_line_width=2,
                             customdata=t[k], hovertemplate="%{y}<br>Estrato " + str(k) +
                             ": %{x:.1%} (%{customdata:,})<extra></extra>"))
    etiquetas = [f"{i}  (n={n:,})".replace(",", " ") for i, n in zip(prop.index, t.sum(1))]
    fig.update_yaxes(autorange="reversed", tickvals=list(prop.index), ticktext=etiquetas, showgrid=False)
    fig.update_xaxes(tickformat=".0%", title="proporción dentro de la categoría")
    fig.update_layout(barmode="stack", title=f"{CATEGORICAS[var]} vs estrato — V de Cramér = {V:.2f}")
    return _estilo(fig, alto=120 + 55 * len(prop))


def fig_correlacion(particion="train"):
    v = base_eda(particion)
    cols = list(NUMERICAS) + ["estrato_num"]
    m = v[cols].sample(min(60_000, len(v)), random_state=42).corr(method="spearman")
    et = [NUMERICAS.get(c, "Estrato") for c in cols]
    fig = go.Figure(go.Heatmap(z=m.values, x=et, y=et, zmin=-1, zmax=1, colorscale=DIVERGENTE,
                               text=np.round(m.values, 2), texttemplate="%{text}", textfont_size=12,
                               hovertemplate="%{y} vs %{x}: ρ = %{z:.2f}<extra></extra>",
                               colorbar=dict(title="ρ")))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title="Correlación de Spearman (muestra de 60 000 viviendas de train)")
    return _estilo(fig, alto=560, leyenda=False)


def fig_faltantes(particion="train"):
    v = base_eda(particion)
    cols = list(NUMERICAS) + list(CATEGORICAS)
    pct = (v[cols].isna().mean() * 100).sort_values()
    fig = go.Figure(go.Bar(x=pct.values, y=[NUMERICAS.get(c, CATEGORICAS.get(c)) for c in pct.index],
                           orientation="h", marker_color="#3987e5", text=[f"{p:.2f} %" for p in pct.values],
                           textposition="outside", hovertemplate="%{y}: %{x:.3f} %<extra></extra>"))
    fig.update_xaxes(title="% de viviendas sin dato", range=[0, max(pct.max() * 1.3, 0.5)])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="Valores faltantes por variable")
    return _estilo(fig, alto=430, leyenda=False)


def _grilla(df, col, celda=0.005, min_n=5):
    g = df.dropna(subset=["lat", "lon"]).assign(glat=lambda d: (d.lat / celda).round() * celda,
                                                 glon=lambda d: (d.lon / celda).round() * celda)
    a = g.groupby(["glat", "glon"]).agg(valor=(col, "mean"), n=(col, "size")).reset_index()
    return a[a.n >= min_n]


def fig_mapa(var="estrato_num", particion="todas"):
    v = base_eda(particion)
    a = _grilla(v, var)
    titulo = "Estrato medio" if var == "estrato_num" else f"{NUMERICAS[var]} (media)"
    if var == "estrato_num":
        rango, escala = (1, 6), SECUENCIAL
    else:
        rango, escala = tuple(a.valor.quantile([0.02, 0.98])), SECUENCIAL
    fig = go.Figure(go.Scattermap(
        lat=a.glat, lon=a.glon, mode="markers",
        marker=dict(size=np.clip(np.sqrt(a.n) * 1.1, 5, 16), color=a.valor, colorscale=escala,
                    cmin=rango[0], cmax=rango[1], opacity=0.85, colorbar=dict(title=titulo.split(" (")[0])),
        customdata=np.c_[a.valor, a.n],
        hovertemplate=titulo + ": %{customdata[0]:.2f}<br>viviendas en la celda: %{customdata[1]:,}<extra></extra>"))
    fig.update_layout(map=dict(style=ESTILO_MAPA, center=dict(lat=10.97, lon=-74.81), zoom=11.3),
                      title=f"{titulo} por celda de ~550 m", height=600, margin=dict(l=0, r=0, t=50, b=0),
                      font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15))
    return fig


def fig_gradiente_norte_sur(particion="todas"):
    v = base_eda(particion).dropna(subset=["lat"])
    v = v.assign(franja=pd.qcut(v.lat, 10, labels=[f"F{i}" for i in range(1, 11)]))
    t = pd.crosstab(v.franja, v.estrato_num, normalize="index")
    fig = go.Figure()
    for k in CLASES:
        fig.add_trace(go.Bar(x=t.index.astype(str), y=t[k], name=f"Estrato {k}", marker_color=COLOR_ESTRATO[k],
                             marker_line_color="white", marker_line_width=2,
                             hovertemplate="%{x}: %{y:.1%}<extra>Estrato " + str(k) + "</extra>"))
    media = v.groupby("franja", observed=True).estrato_num.mean()
    fig.update_layout(barmode="stack",
                      title=f"Composición por franjas de latitud (sur → norte)<br><sup>estrato medio de {media.iloc[0]:.1f} "
                            f"a {media.max():.1f} (franja {media.idxmax()}), {media.iloc[-1]:.1f} en la más al norte</sup>")
    fig.update_yaxes(tickformat=".0%", title="proporción")
    fig.update_xaxes(title="franja (F1 = más al sur, F10 = más al norte; 10 % de viviendas cada una)")
    return _estilo(fig, alto=430)


def fig_correlograma():
    d = diagnosticos()["correlograma"]
    fig = go.Figure(go.Scatter(x=d["centro_km"], y=d["moran_I"], mode="lines+markers",
                               line=dict(color="#2a78d6", width=2), marker=dict(size=8),
                               hovertemplate="%{x:.2f} km: I = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.3, line_dash="dot", line_color="#8a8984",
                  annotation_text="I = 0.3 (umbral de dependencia débil)", annotation_position="top right")
    fig.add_vline(x=1.0, line_dash="dash", line_color="#eb6834",
                  annotation_text="buffer elegido: 1 km", annotation_position="bottom right")
    fig.update_xaxes(title="distancia entre edificios (km)")
    fig.update_yaxes(title="I de Moran del estrato")
    fig.update_layout(title=f"Correlograma: ¿hasta dónde se parecen los vecinos? (Moran global = {diagnosticos()['moran_estrato']['I']:.3f})")
    return _estilo(fig, alto=420, leyenda=False)


def fig_tamano_edificios():
    v = viviendas()
    t = v.groupby("edificio_id").size()
    bins = [0, 1, 5, 20, 100, 500, 4000]
    et = ["1 (casa)", "2–5", "6–20", "21–100", "101–500", "> 500"]
    c = pd.cut(t, bins, labels=et)
    edif = c.value_counts().reindex(et) / len(t)
    unid = t.groupby(c, observed=False).sum().reindex(et) / t.sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=et, y=edif.values, name="% de edificios", marker_color="#86b6ef",
                         hovertemplate="%{x}: %{y:.1%} de los edificios<extra></extra>"))
    fig.add_trace(go.Bar(x=et, y=unid.values, name="% de viviendas", marker_color="#1c5cab",
                         hovertemplate="%{x}: %{y:.1%} de las viviendas<extra></extra>"))
    fig.update_layout(barmode="group", title=f"Unidades por edificio: mediana {int(t.median())}, máximo {t.max():,}".replace(",", " ").replace("  ", ", "))
    fig.update_yaxes(tickformat=".0%", title="proporción")
    fig.update_xaxes(title="viviendas en el edificio")
    return _estilo(fig, alto=420)


# ---------------------------------------------------------------------------
# Pestaña 3: modelos base
# ---------------------------------------------------------------------------
def tabla_metricas():
    r = resultados()
    t = pd.DataFrame(r["resultados_test"]).loc[MODELOS]
    cv = pd.DataFrame(r["tabla_cv"]).loc[MODELOS]
    out = pd.DataFrame({
        "Modelo": [NOMBRE_CORTO[m] for m in MODELOS],
        "F1 macro CV (± desv.)": [f"{a:.3f} ± {b:.3f}" for a, b in zip(cv["f1_macro (media)"], cv["F1 macro (desv. entre folds)"])],
        "F1 macro test": t["f1_macro"].values, "Accuracy": t["accuracy"].values,
        "Acc. balanceada": t["balanced_accuracy"].values, "MAE ordinal": t["MAE_ordinal"].values,
        "Acc. ±1": t["accuracy_±1"].values, "Kappa cuadr.": t["kappa_cuadrático"].values,
        "AUC OvR": t["AUC_OvR_macro"].values, "Log loss": t["log_loss"].values})
    return out


METRICAS_COMPARABLES = {"f1_macro": "F1 macro", "accuracy": "Accuracy", "balanced_accuracy": "Accuracy balanceada",
                        "MAE_ordinal": "MAE ordinal (menor es mejor)", "accuracy_±1": "Accuracy ±1 estrato",
                        "kappa_cuadrático": "Kappa cuadrático", "AUC_OvR_macro": "AUC OvR macro"}


def fig_comparacion(metrica="f1_macro"):
    r = resultados()
    t = r["resultados_test"][metrica]
    fig = go.Figure()
    vals = [t[m] for m in MODELOS]
    error = None
    if metrica == "f1_macro":
        cv = r["tabla_cv"]
        fig.add_trace(go.Bar(x=[NOMBRE_CORTO[m] for m in MODELOS], y=[cv["f1_macro (media)"][m] for m in MODELOS],
                             name="CV espacial (media ± desv.)", marker_color="#e6e5e1",
                             marker_pattern=dict(shape="/", fgcolor="#8a8984", size=7), marker_line_color="#8a8984",
                             marker_line_width=1,
                             error_y=dict(type="data", array=[cv["F1 macro (desv. entre folds)"][m] for m in MODELOS],
                                          color="#52514e", thickness=1.5),
                             hovertemplate="%{x}<br>CV: %{y:.3f}<extra></extra>"))
        ic = r["ic_modelo_principal"]
        principal = MODELOS.index("Logística B_fisicas+ubicacion")
        error = dict(type="data", symmetric=False,
                     array=[0] * principal + [ic["IC95_sup"]["f1_macro"] - vals[principal]] + [0],
                     arrayminus=[0] * principal + [vals[principal] - ic["IC95_inf"]["f1_macro"]] + [0],
                     color="#0b0b0b", thickness=1.5)
    fig.add_trace(go.Bar(x=[NOMBRE_CORTO[m] for m in MODELOS], y=vals, name="Test (7 bloques no vistos)",
                         marker_color=[COLOR_MODELO[m] for m in MODELOS], error_y=error,
                         text=[f"{x:.3f}" for x in vals], textposition="outside",
                         hovertemplate="%{x}<br>test: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group", title=f"{METRICAS_COMPARABLES[metrica]}: modelos frente a líneas base")
    fig.update_yaxes(title=METRICAS_COMPARABLES[metrica], rangemode="tozero")
    return _estilo(fig, alto=460)


def fig_optimismo():
    r = resultados()
    o = r["optimismo_validacion_aleatoria"]
    sens = diagnosticos()["sensibilidad_buffer"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Validación aleatoria vs espacial", "Efecto del buffer"),
                        horizontal_spacing=0.12)
    nombres = ["Aleatoria (filas mezcladas)", "Espacial (bloques + buffer)"]
    claves = ["aleatoria (StratifiedKFold)", "espacial (bloques + buffer)"]
    fig.add_trace(go.Bar(x=nombres, y=[o["F1 macro CV"][k] for k in claves], marker_color=["#e34948", "#2a78d6"],
                         error_y=dict(array=[o["desv."][k] for k in claves], color="#52514e"),
                         text=[f"{o['F1 macro CV'][k]:.2f}" for k in claves], textposition="outside",
                         hovertemplate="%{x}: %{y:.3f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=[s["buffer_km"] for s in sens], y=[s["f1"] for s in sens], mode="lines+markers",
                             line=dict(color="#2a78d6", width=2), marker=dict(size=9),
                             error_y=dict(array=[s["desv"] for s in sens], color="#86b6ef"),
                             hovertemplate="buffer %{x} km: F1 %{y:.3f}<extra></extra>"), 1, 2)
    fig.update_yaxes(title_text="F1 macro (CV)", row=1, col=1, range=[0, 0.75])
    fig.update_xaxes(title_text="buffer (km)", row=1, col=2)
    fig.update_layout(title="La validación aleatoria duplica el F1: la estructura espacial no se puede ignorar")
    return _estilo(fig, alto=430, leyenda=False)


def _pred(modelo):
    p = predicciones()
    proba = p[[f"p{k}|{modelo}" for k in CLASES]].to_numpy()
    return p["estrato"].to_numpy(), p[f"pred|{modelo}"].to_numpy(), proba


def fig_confusion(modelo="Logística B_fisicas+ubicacion", normalizar=True):
    y, pr, _ = _pred(modelo)
    m = confusion_matrix(y, pr, labels=CLASES)
    z = m / m.sum(1, keepdims=True) if normalizar else m
    texto = np.vectorize(lambda a: f"{a:.0%}")(z) if normalizar else m
    fig = go.Figure(go.Heatmap(z=z, x=[f"E{k}" for k in CLASES], y=[f"E{k}" for k in CLASES],
                               colorscale=SECUENCIAL, text=texto, texttemplate="%{text}", textfont_size=14,
                               customdata=m, hovertemplate="real %{y} → predicho %{x}: %{customdata:,} viviendas<extra></extra>",
                               colorbar=dict(title="% de la fila" if normalizar else "viviendas")))
    fig.update_yaxes(autorange="reversed", title="estrato real", showgrid=False)
    fig.update_xaxes(title="estrato predicho", side="bottom")
    fig.update_layout(title=f"Matriz de confusión — {NOMBRE_CORTO[modelo]}")
    return _estilo(fig, alto=480, leyenda=False)


def tabla_por_clase(modelo="Logística B_fisicas+ubicacion"):
    y, pr, _ = _pred(modelo)
    p, r, f, s = precision_recall_fscore_support(y, pr, labels=CLASES, zero_division=0)
    return pd.DataFrame({"Estrato": CLASES, "Precisión": p, "Recall": r, "F1": f, "Viviendas en test": s})


def fig_por_clase(modelo="Logística B_fisicas+ubicacion"):
    t = tabla_por_clase(modelo)
    fig = go.Figure()
    for col, color in [("Precisión", "#86b6ef"), ("Recall", "#2a78d6"), ("F1", "#104281")]:
        fig.add_trace(go.Bar(x=[f"E{k}" for k in t.Estrato], y=t[col], name=col, marker_color=color,
                             hovertemplate="%{x}: " + col + " %{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group", title=f"Desempeño por estrato — {NOMBRE_CORTO[modelo]}")
    fig.update_yaxes(range=[0, 1], title="valor")
    return _estilo(fig, alto=420)


def fig_roc(modelo="Logística B_fisicas+ubicacion"):
    y, _, proba = _pred(modelo)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c3c2b7", dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    for j, k in enumerate(CLASES):
        if (y == k).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y == k, proba[:, j])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=COLOR_ESTRATO[k], width=2),
                                 name=f"E{k} (AUC {auc(fpr, tpr):.2f})",
                                 hovertemplate=f"E{k}: FPR %{{x:.2f}}, TPR %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title="tasa de falsos positivos", range=[0, 1])
    fig.update_yaxes(title="tasa de verdaderos positivos", range=[0, 1.02])
    fig.update_layout(title=f"Curvas ROC uno-contra-resto — {NOMBRE_CORTO[modelo]}")
    return _estilo(fig, alto=460)


def fig_calibracion(modelo="Logística B_fisicas+ubicacion"):
    y, _, proba = _pred(modelo)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#c3c2b7", dash="dot"),
                             name="calibración perfecta", hoverinfo="skip"))
    for j, k in enumerate(CLASES):
        if (y == k).sum() < 50:
            continue
        fr, mp = calibration_curve(y == k, proba[:, j], n_bins=10, strategy="quantile")
        fig.add_trace(go.Scatter(x=mp, y=fr, mode="lines+markers", name=f"E{k}", marker=dict(size=8),
                                 line=dict(color=COLOR_ESTRATO[k], width=2),
                                 hovertemplate=f"E{k}: predicho %{{x:.2f}}, observado %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title="probabilidad predicha", range=[0, 1])
    fig.update_yaxes(title="frecuencia observada", range=[0, 1])
    fig.update_layout(title=f"Calibración por estrato — {NOMBRE_CORTO[modelo]}")
    return _estilo(fig, alto=460)


TIPO_VIVIENDA = {"tv_1": "VIS", "tv_2": "VIP", "tv_3": "no VIS", "tv_4": "no aplica"}


def _nombre_variable(f):
    """Nombre legible de una columna del Pipeline (numérica, indicador de faltante u one-hot)."""
    f = f.split("__", 1)[-1]
    nums = {**NUMERICAS, "x_km": "Posición este-oeste (x)", "y_km": "Posición norte-sur (y)"}
    if f in nums:
        return nums[f]
    if f.startswith("missingindicator_"):
        return f"{nums.get(f[17:], f[17:])}: dato faltante"
    for c, et in [("condicion_predio_", "Régimen: "), ("uso_", "Uso: "), ("tipo_vivienda_", "Tipo de vivienda: ")]:
        if f.startswith(c):
            v = f[len(c):]
            v = TIPO_VIVIENDA.get(v, v)
            return et + ("otras (infrecuentes)" if v == "infrequent_sklearn" else v.replace("_", " "))
    return f.replace("_", " ")


def fig_coeficientes(conjunto="B_fisicas+ubicacion", top=15):
    c = coeficientes()
    c = c[c.modelo == conjunto].pivot(index="variable", columns="estrato", values="coef")
    orden = (c[6] - c[1]).abs().sort_values(ascending=False).index[:top]
    c = c.loc[orden]
    lim = float(np.abs(c.values).max())
    fig = go.Figure(go.Heatmap(z=c.values, x=[f"E{k}" for k in c.columns], y=[_nombre_variable(v) for v in c.index],
                               colorscale=DIVERGENTE, zmin=-lim, zmax=lim, text=np.round(c.values, 2),
                               texttemplate="%{text}", textfont_size=12,
                               hovertemplate="%{y} → %{x}: β = %{z:.3f}<extra></extra>", colorbar=dict(title="β")))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    nombre = "Logística B (físicas + ubicación)" if conjunto.startswith("B") else "Logística A (físicas)"
    fig.update_layout(title=f"Coeficientes por estrato — {nombre} ({top} variables con mayor |β6 − β1|)")
    return _estilo(fig, alto=170 + 32 * top, leyenda=False)


def fig_curva_aprendizaje():
    d = diagnosticos()["curva_aprendizaje"]
    n = np.array(d["n"])
    fig = go.Figure()
    for clave, nombre, color in [("train", "entrenamiento", "#e34948"), ("val", "validación espacial", "#2a78d6")]:
        m, s = np.array(d[f"{clave}_media"]), np.array(d[f"{clave}_desv"])
        fig.add_trace(go.Scatter(x=np.r_[n, n[::-1]], y=np.r_[m + s, (m - s)[::-1]], fill="toself",
                                 fillcolor=color, opacity=0.12, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=n, y=m, mode="lines+markers", name=nombre, line=dict(color=color, width=2),
                                 marker=dict(size=8), hovertemplate="n = %{x:,}: F1 %{y:.3f}<extra>" + nombre + "</extra>"))
    fig.update_xaxes(type="log", title="viviendas de entrenamiento (escala log)",
                     tickvals=[5_000, 10_000, 20_000, 50_000, 100_000], ticktext=["5 mil", "10 mil", "20 mil", "50 mil", "100 mil"])
    fig.update_yaxes(title="F1 macro", range=[0, 0.75])
    fig.update_layout(title="Curva de aprendizaje — logística B")
    return _estilo(fig, alto=430)


def fig_mapa_residuos():
    a = pd.read_pickle(DATOS / "residuos_mapa.pkl.gz")
    lim = float(a.residuo.abs().quantile(0.98))
    m = diagnosticos()["moran_residuos"]["B_fisicas+ubicacion"]
    fig = go.Figure(go.Scattermap(
        lat=a.lat, lon=a.lon, mode="markers",
        marker=dict(size=np.clip(np.sqrt(a.n) * 2, 5, 14), color=a.residuo, colorscale=DIVERGENTE,
                    cmin=-lim, cmax=lim, colorbar=dict(title="residuo")),
        customdata=np.c_[a.residuo, a.n],
        hovertemplate="residuo medio %{customdata[0]:+.2f}<br>viviendas: %{customdata[1]:,}<extra></extra>"))
    fig.update_layout(map=dict(style=ESTILO_MAPA, center=dict(lat=a.lat.mean(), lon=a.lon.mean()), zoom=11.8),
                      title=f"Residuo ordinal en test (I de Moran = {m['I']:.2f})",
                      height=560, margin=dict(l=0, r=0, t=50, b=0), font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15))
    return fig


# ---------------------------------------------------------------------------
# Simulador
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def modelo_principal():
    import joblib
    return joblib.load(DATOS / "modelo_logistica_B.joblib")


@lru_cache(maxsize=None)
def rangos_simulador():
    return json.loads((DATOS / "simulador_rangos.json").read_text(encoding="utf-8"))


def predecir(entrada):
    """entrada: dict con variables físicas, categóricas y lat/lon -> probabilidades por estrato.
    Las coordenadas se proyectan con la MISMA función del Entregable 1 (src/limpieza.py)."""
    from src import limpieza
    lat, lon = entrada.pop("lat"), entrada.pop("lon")
    x, y = limpieza.proyectar_km(np.array([lat]), np.array([lon]))
    fila = dict(entrada, x_km=float(x[0]), y_km=float(y[0]), dist_centro_km=float(np.hypot(x[0], y[0])))
    X = pd.DataFrame([fila])
    mdl = modelo_principal()
    return dict(zip(mdl.classes_.tolist(), mdl.predict_proba(X)[0].tolist()))


def fig_simulador(probas):
    fig = go.Figure(go.Bar(x=[f"E{k}" for k in CLASES], y=[probas[k] for k in CLASES],
                           marker_color=[COLOR_ESTRATO[k] for k in CLASES],
                           text=[f"{probas[k]:.0%}" for k in CLASES], textposition="outside",
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%", title="probabilidad")
    k = max(probas, key=probas.get)
    esperado = sum(c * p for c, p in probas.items())
    fig.update_layout(title=f"Predicción: estrato {k} (valor esperado {esperado:.1f})")
    return _estilo(fig, alto=360, leyenda=False)


# ---------------------------------------------------------------------------
# Modelos avanzados (más allá del modelo base)
# ---------------------------------------------------------------------------
def tabla_metricas_avanzados():
    r = resultados()
    mods = ["Logística B_fisicas+ubicacion", "Moda por zona"] + AVANZADOS
    t = pd.DataFrame(r["resultados_test"]).loc[mods]
    cv = pd.DataFrame(r["tabla_cv"]).loc[mods]
    return pd.DataFrame({
        "Modelo": [NOMBRE_CORTO[m] for m in mods],
        "F1 macro CV (± desv.)": [f"{a:.3f} ± {b:.3f}" for a, b in zip(cv["f1_macro (media)"], cv["F1 macro (desv. entre folds)"])],
        "F1 macro test": t["f1_macro"].values, "Accuracy": t["accuracy"].values,
        "Acc. balanceada": t["balanced_accuracy"].values, "MAE ordinal": t["MAE_ordinal"].values,
        "Acc. ±1": t["accuracy_±1"].values, "Kappa cuadr.": t["kappa_cuadrático"].values,
        "AUC OvR": t["AUC_OvR_macro"].values, "Log loss": t["log_loss"].values})


def fig_comparacion_avanzados(metrica="f1_macro"):
    r = resultados()
    mods = ["Moda por zona", "Logística B_fisicas+ubicacion"] + AVANZADOS
    t = r["resultados_test"][metrica]
    cv = r["tabla_cv"]
    fig = go.Figure()
    if metrica == "f1_macro":
        fig.add_trace(go.Bar(x=[NOMBRE_CORTO[m] for m in mods], y=[cv["f1_macro (media)"][m] for m in mods],
                             name="CV espacial (media ± desv.)", marker_color="#e6e5e1",
                             marker_pattern=dict(shape="/", fgcolor="#8a8984", size=7), marker_line_color="#8a8984",
                             marker_line_width=1,
                             error_y=dict(type="data", array=[cv["F1 macro (desv. entre folds)"][m] for m in mods],
                                          color="#52514e", thickness=1.5),
                             hovertemplate="%{x}<br>CV: %{y:.3f}<extra></extra>"))
    vals = [t[m] for m in mods]
    fig.add_trace(go.Bar(x=[NOMBRE_CORTO[m] for m in mods], y=vals, name="Test (7 bloques no vistos)",
                         marker_color=[COLOR_MODELO[m] for m in mods], text=[f"{x:.3f}" for x in vals],
                         textposition="outside", hovertemplate="%{x}<br>test: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group", title=f"{METRICAS_COMPARABLES[metrica]}: modelos avanzados frente a la referencia")
    fig.update_yaxes(title=METRICAS_COMPARABLES[metrica], rangemode="tozero")
    return _estilo(fig, alto=460)


def fig_diferencias():
    """Diferencias pareadas (bootstrap por bloques) de F1 macro frente a la logística B y G-XGBoost frente a XGBoost."""
    c = resultados()["avanzados"]["comparaciones_f1"]
    nombres = list(c)
    dif = [float(c[n]["diferencia"]) for n in nombres]
    lo = [float(c[n]["IC95_inf"]) for n in nombres]
    hi = [float(c[n]["IC95_sup"]) for n in nombres]
    etiquetas = [n.replace("Logística B", "logística B").replace("SVM RBF (Nyström) + TPE", "SVM RBF + TPE") for n in nombres]
    colores = [COLOR_MODELO[n.split(" − ")[0]] for n in nombres]
    fig = go.Figure()
    fig.add_vline(x=0, line_color="#8a8984", line_dash="dot")
    fig.add_trace(go.Scatter(x=dif, y=etiquetas, mode="markers", marker=dict(size=13, color=colores),
                             error_x=dict(type="data", symmetric=False, array=np.array(hi) - np.array(dif),
                                          arrayminus=np.array(dif) - np.array(lo), color="#52514e", thickness=2),
                             customdata=np.c_[lo, hi],
                             hovertemplate="%{y}<br>diferencia %{x:+.3f}<br>IC 95 % [%{customdata[0]:+.3f}, "
                                           "%{customdata[1]:+.3f}]<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="diferencia de F1 macro en test (IC 95 %)", zeroline=False)
    fig.update_layout(title="¿Mejoran a la referencia? Diferencias pareadas de F1 macro")
    return _estilo(fig, alto=380, leyenda=False)


def fig_geoxgb_busqueda():
    """F1 de CV del XGBoost geográfico según el peso del modelo global (alpha) y el ancho de banda."""
    a = resultados()["avanzados"]["busqueda"]
    grilla = a["geoxgb"]["grilla"]
    ref = np.mean(a["xgb"]["grilla"][a["xgb"]["elegido"]])
    fig = go.Figure()
    for k, color in [("0.05", "#9085e9"), ("0.15", "#4a3aa7")]:
        claves = sorted([c for c in grilla if c.startswith(f"frac={k}|")], key=lambda c: float(c.split("=")[-1]))
        al = [float(c.split("=")[-1]) for c in claves]
        m = [np.mean(grilla[c]) for c in claves]
        sd = [np.std(grilla[c]) for c in claves]
        fig.add_trace(go.Scatter(x=al, y=m, mode="lines+markers", name=f"ventana: {float(k):.0%} del entrenamiento",
                                 line=dict(color=color, width=2), marker=dict(size=9),
                                 error_y=dict(array=sd, color=color, thickness=1),
                                 hovertemplate="alpha %{x}: F1 CV %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[1.0], y=[ref], mode="markers", name="XGBoost global (alpha = 1)",
                             marker=dict(size=12, color="#e87ba4", symbol="diamond"),
                             hovertemplate="XGBoost global: %{y:.3f}<extra></extra>"))
    el = a["geoxgb"]["elegido"]
    fig.update_xaxes(title="alpha = peso del modelo global (0 = solo modelos locales, 1 = solo global)",
                     tickvals=[0, 0.25, 0.5, 0.75, 1])
    fig.update_yaxes(title="F1 macro (CV espacial)")
    fig.update_layout(title="XGBoost geográfico: F1 de CV según alpha y la ventana")
    return _estilo(fig, alto=430)


def fig_importancia_xgb(top=12):
    imp = pd.Series(resultados()["avanzados"]["importancia_xgb"]).sort_values(ascending=False).head(top)
    fig = go.Figure(go.Bar(x=imp.values[::-1], y=[_nombre_variable(v) for v in imp.index[::-1]], orientation="h",
                           marker_color="#e87ba4", hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="importancia (ganancia relativa)")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="XGBoost: variables más importantes")
    return _estilo(fig, alto=440, leyenda=False)


def fig_mapa_residuos_modelo(modelo="XGBoost geográfico"):
    if modelo == "Logística B_fisicas+ubicacion":
        return fig_mapa_residuos()
    a = pd.read_pickle(DATOS / "residuos_mapa_avanzados.pkl.gz")
    a = a[a.modelo == modelo]
    lim = float(a.residuo.abs().quantile(0.98))
    m = diagnosticos()["moran_residuos"][modelo]
    fig = go.Figure(go.Scattermap(
        lat=a.lat, lon=a.lon, mode="markers",
        marker=dict(size=np.clip(np.sqrt(a.n) * 2, 5, 14), color=a.residuo, colorscale=DIVERGENTE,
                    cmin=-lim, cmax=lim, colorbar=dict(title="residuo")),
        customdata=np.c_[a.residuo, a.n],
        hovertemplate="residuo medio %{customdata[0]:+.2f}<br>viviendas: %{customdata[1]:,}<extra></extra>"))
    fig.update_layout(map=dict(style=ESTILO_MAPA, center=dict(lat=a.lat.mean(), lon=a.lon.mean()), zoom=11.8),
                      title=f"Residuo ordinal en test — {NOMBRE_CORTO[modelo]} (I de Moran = {m['I']:.2f})",
                      height=560, margin=dict(l=0, r=0, t=50, b=0),
                      font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=15))
    return fig


# ---------------------------------------------------------------------------
# Capítulo 5: variables de vecindad física y los dos escenarios de predicción
# (resultados de vecindad/, guardados en datos/vecindad/; aquí solo se grafican)
# ---------------------------------------------------------------------------
DATOS_VEC = DATOS / "vecindad"
PRINCIPAL_VEC = "B + vecindad (300 m y 1 km)"
COLOR_VEC = {"B": "#2a78d6", PRINCIPAL_VEC: "#0f8a7a", "B + vecindad (solo 300 m)": "#5fb3a8",
             "B + vecindad (solo 1 km)": "#5fb3a8", "A": "#1baf7a", "A + vecindad (sin coordenadas)": "#8fcfc6"}
NOMBRE_VEC = {"B": "Logística B", PRINCIPAL_VEC: "B + vecindad (principal)",
              "B + vecindad (solo 300 m)": "B + vecindad, solo 300 m", "B + vecindad (solo 1 km)": "B + vecindad, solo 1 km",
              "A": "A: solo variables físicas", "A + vecindad (sin coordenadas)": "A + vecindad (sin coordenadas)"}
COLOR_CONOCIDAS = {"Vecinos más cercanos (k = 15)": "#eb6834", "XGBoost geográfico": "#4a3aa7", "XGBoost": "#e87ba4",
                   "Logística B + vecindad + estrato de vecinos": "#0b6e61", "Logística B + vecindad": "#0f8a7a",
                   "Logística B": "#2a78d6", "Moda por zona (2 km)": "#f3a37a"}


@lru_cache(maxsize=None)
def vecindad(nombre):
    """Resultados del capítulo 5: comparacion_zonas_nuevas, cv_15_folds, variante_estricta, zonas, zonas_conocidas."""
    return json.loads((DATOS_VEC / f"{nombre}.json").read_text(encoding="utf-8"))


def hay_vecindad():
    return all((DATOS_VEC / f"{n}.json").exists()
               for n in ["comparacion_zonas_nuevas", "cv_15_folds", "zonas", "zonas_conocidas"])


def tabla_vecindad():
    c = vecindad("comparacion_zonas_nuevas")
    filas = []
    for k, m in c["modelos"].items():
        cmp_ = c["comparaciones_vs_B"].get(k)
        filas.append({
            "Modelo": NOMBRE_VEC[k], "F1 CV": m["cv_f1_media"],
            "Δ CV (error est.)": "—" if cmp_ is None else f"{cmp_['cv_dif_media']:+.3f} ({cmp_['cv_se_pareado']:.3f})",
            "¿Reemplaza a B?": "—" if cmp_ is None else ("Sí" if cmp_["reemplaza_a_B"] else "No"),
            "F1 test": m["test"]["f1_macro"],
            "Δ F1 test [IC 95 %]": "—" if cmp_ is None else
            f"{cmp_['test_f1']['diferencia']:+.3f} [{cmp_['test_f1']['IC95_inf']:+.3f}, {cmp_['test_f1']['IC95_sup']:+.3f}]",
            "MAE test": m["test"]["MAE_ordinal"], "Moran residuos": m["moran_residuos"]})
    return pd.DataFrame(filas)


def fig_vecindad_diferencias():
    """Diferencia frente a la logística B: en la CV espacial (media ± error estándar) y en test (IC 95 % por bloques)."""
    c = vecindad("comparacion_zonas_nuevas")["comparaciones_vs_B"]
    nombres = list(c)
    etq = [NOMBRE_VEC[n] for n in nombres]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("Validación cruzada (5 folds): media ± error estándar",
                                        "Test (7 bloques): IC 95 % por bloques"))
    paneles = [([c[n]["cv_dif_media"] for n in nombres],
                [c[n]["cv_dif_media"] - c[n]["cv_se_pareado"] for n in nombres],
                [c[n]["cv_dif_media"] + c[n]["cv_se_pareado"] for n in nombres]),
               ([c[n]["test_f1"]["diferencia"] for n in nombres], [c[n]["test_f1"]["IC95_inf"] for n in nombres],
                [c[n]["test_f1"]["IC95_sup"] for n in nombres])]
    for col, (val, lo, hi) in enumerate(paneles, start=1):
        fig.add_vline(x=0, line_color="#8a8984", line_dash="dot", row=1, col=col)
        fig.add_trace(go.Scatter(
            x=val, y=etq, mode="markers",
            marker=dict(size=[17 if n == PRINCIPAL_VEC else 12 for n in nombres], color=[COLOR_VEC[n] for n in nombres],
                        line=dict(color="white", width=1.5)),
            error_x=dict(type="data", symmetric=False, array=np.array(hi) - np.array(val),
                         arrayminus=np.array(val) - np.array(lo), color="#52514e", thickness=2),
            customdata=np.c_[lo, hi],
            hovertemplate="%{y}<br>diferencia %{x:+.3f}<br>[%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]<extra></extra>"),
            1, col)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="diferencia de F1 macro frente a la logística B", zeroline=False)
    fig.update_layout(title="Zonas nuevas: ¿mejora la logística al ver el barrio?")
    return _estilo(fig, alto=430, leyenda=False)


def fig_vecindad_folds15():
    """Diferencia de F1 por fold en la CV de 15 folds: la vecindad gana en unos y pierde en otros."""
    q = vecindad("cv_15_folds")["comparaciones_vs_B"]
    fig = go.Figure()
    fig.add_hline(y=0, line_color="#8a8984")
    for k, nombre, color in [(PRINCIPAL_VEC, "B + vecindad (principal)", "#0f8a7a"),
                             ("A + vecindad (sin coordenadas)", "A + vecindad (sin coordenadas)", "#8fcfc6")]:
        d = q[k]["dif_F1_por_fold"]
        fig.add_trace(go.Bar(x=[f"F{i + 1}" for i in range(len(d))], y=d,
                             name=f"{nombre}: mejora en {q[k]['folds_mejores']} de {len(d)} folds",
                             marker_color=color, hovertemplate="%{x}: %{y:+.3f}<extra></extra>"))
    fig.update_layout(barmode="group", title="15 folds espaciales: diferencia de F1 frente a la logística B")
    fig.update_yaxes(title="Δ F1 macro", zeroline=False)
    return _estilo(fig, alto=400)


def fig_vecindad_entorno():
    """Cambio de accuracy al agregar la vecindad, según qué tan mezclado está el estrato del entorno."""
    z = vecindad("zonas")
    fig = go.Figure()
    fig.add_hline(y=0, line_color="#8a8984")
    for clave, nombre, color in [("heterogeneidad|train (CV 15 folds)", "Validación cruzada (15 folds)", "#86b6ef"),
                                 ("heterogeneidad|test", "Test (7 bloques)", "#0f8a7a")]:
        t = z[clave]
        fig.add_trace(go.Bar(x=t["grupo_het"], y=t["d_acc"], name=nombre, marker_color=color,
                             customdata=np.c_[t["% viviendas"], t["acc_B"], t["acc_Bvec"]],
                             text=[f"{v:+.2f}" for v in t["d_acc"]], textposition="outside",
                             hovertemplate="%{x}<br>%{customdata[0]:.1f} % de las viviendas<br>accuracy "
                                           "%{customdata[1]:.2f} → %{customdata[2]:.2f}<extra></extra>"))
    fig.update_layout(barmode="group", title="¿Dónde ayuda la vecindad? Según qué tan mezclado es el entorno (300 m)")
    fig.update_xaxes(title="desviación del estrato a 300 m (solo diagnóstico)")
    fig.update_yaxes(title="cambio en accuracy", zeroline=False, range=[-0.25, 0.27])
    return _estilo(fig, alto=430)


def fig_escenarios():
    """Pendientes: F1 macro de cada modelo en zonas nuevas y en zonas conocidas. Lo que importa es el ORDEN
    dentro de cada escenario (los conjuntos de evaluación son distintos)."""
    r = resultados()["resultados_test"]["f1_macro"]
    vn = vecindad("comparacion_zonas_nuevas")["modelos"][PRINCIPAL_VEC]["test"]["f1_macro"]
    kc = vecindad("zonas_conocidas")["modelos"]
    lineas = [("Logística B", r["Logística B_fisicas+ubicacion"], kc["Logística B"]["F1_macro"]),
              ("Logística B + vecindad", vn, kc["Logística B + vecindad"]["F1_macro"]),
              ("XGBoost", r["XGBoost"], kc["XGBoost"]["F1_macro"]),
              ("XGBoost geográfico", r["XGBoost geográfico"], kc["XGBoost geográfico"]["F1_macro"]),
              ("Moda por zona (2 km)", r["Moda por zona"], kc["Moda por zona (2 km)"]["F1_macro"])]
    xs = ["Zonas nuevas<br>(bloques no vistos)", "Zonas conocidas<br>(edificios ocultos)"]
    k = kc["Vecinos más cercanos (k = 15)"]["F1_macro"]
    fig = go.Figure()
    for nombre, a, b in lineas:
        color = COLOR_CONOCIDAS[nombre]
        grueso = 4.5 if nombre in ("XGBoost geográfico", "Logística B") else 2.5
        fig.add_trace(go.Scatter(x=xs, y=[a, b], mode="lines+markers", name=nombre,
                                 line=dict(color=color, width=grueso), marker=dict(size=12, color=color),
                                 hovertemplate=nombre + "<br>%{x}: F1 %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[xs[1]], y=[k], mode="markers", name="Votar con los 15 vecinos",
                             marker=dict(size=20, color=COLOR_CONOCIDAS["Vecinos más cercanos (k = 15)"], symbol="star",
                                         line=dict(color="white", width=1)),
                             hovertemplate="Votar con el estrato de las 15 viviendas conocidas más cercanas: "
                                           "%{y:.3f}<extra></extra>"))

    def repartir(valores, gap=0.032):
        """Posiciones de etiqueta en el mismo orden que los valores, separadas al menos `gap`."""
        orden = np.argsort(valores)
        pos = np.array(valores, float)
        for i in range(1, len(orden)):
            pos[orden[i]] = max(pos[orden[i]], pos[orden[i - 1]] + gap)
        return pos

    izq = [(n, a) for n, a, _ in lineas]
    der = [(n, b) for n, _, b in lineas] + [("Votar con los 15 vecinos", k)]
    for lado, datos, x, anclaje, desplaz in [("izq", izq, xs[0], "right", -14), ("der", der, xs[1], "left", 14)]:
        ys = repartir([v for _, v in datos])
        for (nombre, v), y in zip(datos, ys):
            color = COLOR_CONOCIDAS["Vecinos más cercanos (k = 15)" if nombre.startswith("Votar") else nombre]
            texto = f"<b>{v:.2f}</b>" if lado == "izq" else f"<b>{v:.2f}</b>  {nombre}"
            fig.add_annotation(x=x, y=y, text=texto, showarrow=False, xanchor=anclaje, xshift=desplaz,
                               font=dict(color=color, size=14))
    fig.update_yaxes(title="F1 macro", range=[0.25, 0.86])
    fig.update_xaxes(range=[-0.45, 1.05])
    fig.update_layout(title="El mejor modelo depende del escenario: el orden se invierte")
    fig = _estilo(fig, alto=540, leyenda=False)
    fig.update_layout(margin=dict(l=60, r=300, t=60, b=50))
    return fig


def fig_zonas_conocidas():
    kc = vecindad("zonas_conocidas")["modelos"]
    orden = sorted(kc, key=lambda m: kc[m]["F1_macro"])
    f1 = [kc[m]["F1_macro"] for m in orden]
    lo = [min(kc[m]["F1_por_fold"]) for m in orden]
    hi = [max(kc[m]["F1_por_fold"]) for m in orden]
    fig = go.Figure(go.Bar(
        x=f1, y=orden, orientation="h", marker_color=[COLOR_CONOCIDAS[m] for m in orden],
        error_x=dict(type="data", symmetric=False, array=np.array(hi) - np.array(f1),
                     arrayminus=np.array(f1) - np.array(lo), color="#52514e", thickness=1.5),
        text=[f"{v:.3f}" for v in f1], textposition="inside", insidetextanchor="end", textfont=dict(color="white"),
        hovertemplate="%{y}<br>F1 macro %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="F1 macro (error: rango en 5 folds)",
                     range=[0, 0.9])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="Zonas conocidas: F1 macro por modelo")
    return _estilo(fig, alto=440, leyenda=False)


def tabla_comparaciones_conocidas():
    c = vecindad("zonas_conocidas")["comparaciones"]
    filas = []
    for k, v in c.items():
        f, m = v["F1"], v["MAE"]
        filas.append({"Comparación": k,
                      "Δ F1 macro [IC 95 %]": f"{f['diferencia']:+.3f} [{f['IC95_inf']:+.3f}, {f['IC95_sup']:+.3f}]",
                      "Δ MAE ordinal [IC 95 %]": f"{m['diferencia']:+.3f} [{m['IC95_inf']:+.3f}, {m['IC95_sup']:+.3f}]",
                      "¿Distinguible de cero?": "Sí" if f["IC95_inf"] > 0 or f["IC95_sup"] < 0 else "No"})
    return pd.DataFrame(filas)
