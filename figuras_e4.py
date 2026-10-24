"""Figuras del dashboard para el Entregable 4 (proyecto final): modelo original EPL, validación repetida 5 × 5,
pruebas estadísticas, test confirmatorio, conjuntos externos, robustez, costo y predicción en vivo.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Todo sale de ``datos/e4/``, copiado de https://github.com/mcarocantillo/entregable4-ml-barranquilla (``data/dashboard``,
``resultados/`` y ``artifacts/``). Aquí no se entrena nada: la predicción en vivo carga el EPL serializado del E4.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from figuras import DATOS, _estilo
from figuras_mejoras import AZUL, AZUL_CLARO, GRIS, NARANJA, ROJO, TINTA, VERDE

E4 = DATOS / "e4"
ORIGINAL = "EPL (original)"
BASE_E1 = "Modelo base (E1)"
ESTRATOS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
NOMBRE_ESPACIO = {"geografico": "geográfico", "variables": "de las variables", "probabilidades": "de las probabilidades"}
NOMBRE_EXTERNO = {"king_county": "King County (precio de vivienda en 4 clases)",
                  "california": "California (valor de vivienda en 5 clases)",
                  "vino_tinto": "Vino tinto (calidad en 6 clases)"}
NOMBRE_COND = {"base": "Datos originales", "ruido_20": "20 % de etiquetas con ruido",
               "tamano_25": "25 % del entrenamiento", "desbalance_56": "Estratos 5 y 6 reducidos"}
NOMBRE_REGLA = {"EPL": "EPL (original)", "promedio_simple": "Promedio simple", "pesos_error_global": "Pesos por error global",
                "Random Forest": "Random Forest", "CatBoost": "CatBoost"}
NOMBRE_VARIANTE = {"solo geografico": "Solo espacio geográfico (versión 1)", "solo variables": "Solo espacio de las variables",
                   "solo probabilidades": "Solo espacio de las probabilidades", "promedio_simple": "Regla: promedio simple",
                   "pesos_error_global": "Regla: pesos por error global", "stacking_lineal": "Regla: stacking lineal",
                   "sin_campo_local": "Sin campo local (solo pesos globales)", "sin_kl": "Sin penalización KL (λ = 0)",
                   "sin_suavizado": "Sin suavizado del núcleo", "pesos_clase_contrarios": "Pesos de clase contrarios",
                   "sin_ajuste_cruzado": "Sin ajuste cruzado"}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def leer(nombre):
    p = E4 / nombre
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def disponible():
    return (E4 / "modelos.json").exists()


def modelos():
    return leer("modelos.json")


@lru_cache(maxsize=1)
def pliegues():
    """F1 macro de los 11 modelos en los 25 pliegues externos (+ el modelo base del E1, mismas particiones)."""
    df = pd.read_csv(E4 / "f1_por_pliegue_v2.csv")
    b = leer("base_vs_epl.json")["f1_por_pliegue"]
    df[BASE_E1] = [b[f"r{r}_f{f}"] for r, f in zip(df.rep, df.fold)]
    return df


@lru_cache(maxsize=1)
def viviendas():
    return pd.read_parquet(E4 / "viviendas.parquet")


@lru_cache(maxsize=1)
def ejemplos():
    return pd.read_parquet(E4 / "ejemplos.parquet")


@lru_cache(maxsize=1)
def modelo_epl():
    """EPL serializado del E4 (≈ 200 MB en memoria con XGBoost); se carga la primera vez que se predice."""
    import joblib
    return joblib.load(E4 / "epl__original.joblib")


def variante(nombre):
    v = modelos()["estadistica"][modelos()["version_original"]]
    return next(x for x in v["epl_vs_variantes"] if x["rival"] == nombre)


def rival(nombre, version=None):
    v = modelos()["estadistica"][version or modelos()["version_original"]]
    return next(x for x in v["epl_vs_benchmarks"] if x["rival"] == nombre)


def _color(nombre):
    return NARANJA if nombre == ORIGINAL else (TINTA if nombre == BASE_E1 else AZUL)


# ---------------------------------------------------------------------------
# Revisión de literatura
# ---------------------------------------------------------------------------
def fig_literatura():
    """Registros relevantes por base y cadena (23-oct-2026); el texto da relevantes / revisados y los resultados."""
    lit = leer("literatura.json")
    m = [f for f in lit["matriz"] if f[2] == "23-oct-2026"]
    bases = list(dict.fromkeys(f[0] for f in m))
    cadenas = sorted({f[1] for f in m})
    celda = {(f[0], f[1]): f for f in m}
    z, texto, notas = [], [], []
    for b in bases:
        fila_z, fila_t, fila_n = [], [], []
        for c in cadenas:
            f = celda.get((b, c))
            fila_z.append(int(f[5]) if f else None)
            fila_t.append(f"{f[5]} / {f[4]}<br>({f[3]})" if f else "—")
            fila_n.append(f[6] if f else "")
        z.append(fila_z), texto.append(fila_t), notas.append(fila_n)
    fig = go.Figure(go.Heatmap(
        z=z, x=cadenas, y=bases,
        text=texto, texttemplate="%{text}", customdata=notas, colorscale=[[0, "#f0efec"], [1, NARANJA]], zmin=0,
        colorbar=dict(title="Relevantes", thickness=12),
        hovertemplate="%{y}, %{x}<br>%{customdata}<extra></extra>"))
    fig.update_layout(title="Relevantes / revisados (resultados) por base y cadena")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig = _estilo(fig, alto=330, leyenda=False)
    fig.update_layout(margin=dict(l=120, t=60))
    return fig


# ---------------------------------------------------------------------------
# Corrección del E3 y validación repetida
# ---------------------------------------------------------------------------
def fig_e3_vs_5x5(e3_mejores):
    """F1 que publicó el E3 (5 pliegues, elegidos y medidos con los mismos) frente a la media en la 5 × 5."""
    lid = {r["modelo"]: r for r in modelos()["lideres"]}
    filas = sorted([(m, v, lid[m]["f1_cv"]) for m, v in e3_mejores.items() if m in lid], key=lambda t: t[2])
    fig = go.Figure()
    for m, a, b in filas:
        fig.add_trace(go.Scatter(x=[a, b], y=[m, m], mode="lines", line=dict(color=GRIS, width=2), showlegend=False,
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[a for _, a, _ in filas], y=[m for m, _, _ in filas], mode="markers", name="E3 publicado",
                             marker=dict(size=13, color=AZUL_CLARO, line=dict(color=AZUL, width=1)),
                             hovertemplate="%{y}: %{x:.3f} en el E3<extra></extra>"))
    fig.add_trace(go.Scatter(x=[b for *_, b in filas], y=[m for m, _, _ in filas], mode="markers+text",
                             name="Validación repetida 5 × 5 (E4)", marker=dict(size=13, color=AZUL),
                             text=[f"{b - a:+.3f}" for _, a, b in filas], textposition="middle left",
                             textfont=dict(color=ROJO),
                             hovertemplate="%{y}: %{x:.3f} en la 5 × 5<extra></extra>"))
    fig.update_layout(title="F1 macro: lo que publicó el E3 frente a la validación repetida 5 × 5")
    fig.update_xaxes(title="F1 macro (media de los pliegues externos)")
    return _estilo(fig, alto=430)


def fig_lideres(vista="cv"):
    """F1 de los 11 modelos (y el modelo base del E1): media ± desviación en la 5 × 5, o test con su IC BCa."""
    lid = pd.DataFrame(modelos()["lideres"])
    if vista == "cv":
        b = leer("base_vs_epl.json")
        df = pd.concat([lid[["modelo", "f1_cv", "f1_cv_sd"]],
                        pd.DataFrame([{"modelo": BASE_E1, "f1_cv": b["f1_medio_25"],
                                       "f1_cv_sd": float(np.std(list(b["f1_por_pliegue"].values()), ddof=1))}])])
        df = df.sort_values("f1_cv")
        x, err, err_m = df.f1_cv, df.f1_cv_sd, df.f1_cv_sd
        titulo, eje = "F1 macro en la validación repetida 5 × 5 (25 pliegues)", "F1 macro (media ± desviación)"
    else:
        t = leer("test_confirmatorio.json")
        bt = leer("base_vs_epl.json")["test"]
        df = pd.DataFrame([{"modelo": m, "f1": v, "lo": t["ic95_bca_bloques"][m][0], "hi": t["ic95_bca_bloques"][m][1]}
                           for m, v in t["f1_macro"].items()])
        df = pd.concat([df, pd.DataFrame([{"modelo": BASE_E1, "f1": bt["f1"][BASE_E1], "lo": np.nan, "hi": np.nan}])])
        df = df.sort_values("f1")
        x, err, err_m = df.f1, (df.hi - df.f1).fillna(0), (df.f1 - df.lo).fillna(0)
        titulo, eje = "F1 macro en el test (65 145 viviendas, 7 bloques)", "F1 macro (IC 95 % BCa por bloques)"
    fig = go.Figure(go.Bar(x=x, y=df.modelo, orientation="h", marker_color=[_color(m) for m in df.modelo],
                           error_x=dict(type="data", symmetric=False, array=err, arrayminus=err_m, color=GRIS,
                                        thickness=1.2),
                           text=[f"{v:.3f}" for v in x], textposition="inside", insidetextanchor="end",
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title=titulo)
    fig.update_xaxes(title=eje, range=[0, 0.8])
    fig.update_yaxes(showgrid=False)
    fig = _estilo(fig, alto=470, leyenda=False)
    fig.update_layout(margin=dict(l=160))
    return fig


def fig_cd(version="v2"):
    """Diagrama de diferencia crítica de Nemenyi: rango medio en los 25 pliegues; la franja mide CD desde el mejor."""
    fr = modelos()["estadistica"][version]["friedman"]
    r = sorted(fr["rangos"].items(), key=lambda t: t[1])
    nombres, valores = [n for n, _ in r], [v for _, v in r]
    fig = go.Figure(go.Scatter(x=valores, y=nombres, mode="markers+text", marker=dict(size=13, color=[_color(n) for n in nombres]),
                               text=[f"{v:.2f}" for v in valores], textposition="middle right",
                               hovertemplate="%{y}: rango medio %{x:.2f}<extra></extra>"))
    fig.add_shape(type="rect", x0=valores[0], x1=valores[0] + fr["cd"], y0=-0.5, y1=len(nombres) - 0.5,
                  fillcolor="rgba(237,161,0,.15)", line=dict(width=0), layer="below")
    fig.add_annotation(x=valores[0] + fr["cd"], y=-0.5, text=f"CD = {fr['cd']:.2f}", showarrow=False, yanchor="bottom",
                       xanchor="left", font=dict(color="#8a6510"))
    fig.update_layout(title=f"Rango medio en los 25 pliegues, versión {version[1]} (Friedman χ² = {fr['chi2']:.0f})")
    fig.update_xaxes(title="Rango medio (1 = mejor en el pliegue)")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig = _estilo(fig, alto=470, leyenda=False)
    fig.update_layout(margin=dict(l=160))
    return fig


def fig_pliegues(rival_=None):
    """EPL − rival en cada uno de los 25 pliegues externos, ordenados; verde = gana el EPL."""
    df = pliegues()
    rival_ = rival_ or modelos()["estadistica"][modelos()["version_original"]]["veredicto_validacion"]["mejor_benchmark"]
    d = (df[ORIGINAL] - df[rival_]).to_numpy()
    etiquetas = [f"rep. {r}, pliegue {f}" for r, f in zip(df.rep, df.fold)]
    orden = np.argsort(d)
    d, etiquetas = d[orden], [etiquetas[i] for i in orden]
    fig = go.Figure(go.Bar(x=list(range(1, len(d) + 1)), y=d, marker_color=[VERDE if v > 0 else ROJO for v in d],
                           customdata=etiquetas, hovertemplate="%{customdata}: %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=float(np.mean(d)), line_dash="dot", line_color=TINTA,
                  annotation_text=f"media {np.mean(d):+.3f}", annotation_position="top left")
    fig.update_layout(title=f"EPL − {rival_} en cada pliegue: gana en {int((d > 0).sum())} de {len(d)}")
    fig.update_xaxes(title="Pliegues ordenados por la diferencia", showticklabels=False)
    fig.update_yaxes(title="Diferencia de F1 macro", zeroline=True, zerolinecolor=TINTA)
    return _estilo(fig, alto=420, leyenda=False)


def fig_ablacion():
    """EPL − variante (IC 95 % BCa en los 25 pliegues): reglas estándar con los mismos expertos y componentes retirados."""
    v = modelos()["estadistica"][modelos()["version_original"]]["epl_vs_variantes"]
    v = sorted(v, key=lambda x: x["dif_media"])
    y = [NOMBRE_VARIANTE.get(x["rival"], x["rival"]) for x in v]
    m = np.array([x["dif_media"] for x in v])
    lo, hi = np.array([x["IC95_BCa"][0] for x in v]), np.array([x["IC95_BCa"][1] for x in v])
    col = [VERDE if a > 0 else (ROJO if b < 0 else GRIS) for a, b in zip(lo, hi)]
    fig = go.Figure(go.Scatter(x=m, y=y, mode="markers", marker=dict(size=13, color=col),
                               error_x=dict(type="data", symmetric=False, array=hi - m, arrayminus=m - lo, color=TINTA),
                               customdata=[x["pliegues_a_favor"] for x in v],
                               hovertemplate="%{y}: EPL %{x:+.4f} (gana en %{customdata}/25)<extra></extra>"))
    fig.add_vline(x=0, line_dash="dot", line_color=TINTA)
    fig.update_layout(title="F1 macro: EPL − cada variante (IC 95 % BCa, 25 pliegues)")
    fig.update_xaxes(title="Diferencia de F1 macro (a la derecha, el EPL completo es mejor)")
    fig.update_yaxes(showgrid=False)
    fig = _estilo(fig, alto=440, leyenda=False)
    fig.update_layout(margin=dict(l=230))
    return fig


# ---------------------------------------------------------------------------
# El modelo por dentro
# ---------------------------------------------------------------------------
def fig_convergencia():
    h = modelos()["campo_epl"]["historia"]
    fig = go.Figure(go.Scatter(x=list(range(len(h))), y=h, mode="lines+markers", line=dict(color=NARANJA),
                               marker=dict(size=6), hovertemplate="Iteración %{x}: pérdida %{y:.5f}<extra></extra>"))
    fig.update_layout(title="Convergencia del gradiente exponenciado (modelo final)")
    fig.update_xaxes(title="Iteración")
    fig.update_yaxes(title="Log-pérdida ponderada + KL")
    return _estilo(fig, alto=380, leyenda=False)


def fig_campo(experto=None):
    """Peso de un experto en cada ancla del espacio de competencia del modelo final."""
    c = modelos()["campo_epl"]
    anclas, pesos = np.array(c["anclas"]), np.array(c["pesos"])
    experto = experto or c["expertos"][int(np.argmax(np.ptp(pesos, axis=0)))]
    j = c["expertos"].index(experto)
    fig = go.Figure(go.Scatter(
        x=anclas[:, 0], y=anclas[:, 1], mode="markers+text", text=[f"{w:.2f}" for w in pesos[:, j]],
        textposition="top center",
        marker=dict(size=10 + 30 * pesos[:, j], color=pesos[:, j], cmin=0, cmax=1,
                    colorscale=[[0, "#fdf1ea"], [1, NARANJA]], line=dict(width=1, color=GRIS),
                    colorbar=dict(title="Peso", thickness=12)),
        hovertemplate="ancla (%{x:.1f}, %{y:.1f}): peso %{marker.color:.2f}<extra></extra>"))
    g = c["pesos_globales"][j]
    ejes = {"geografico": ("Este-oeste (km)", "Sur-norte (km)"),
            "variables": ("Componente principal 1 (escalada)", "Componente principal 2 (escalada)"),
            "probabilidades": ("Clase esperada por los expertos (escalada)", "Desacuerdo entre expertos")}[c["espacio"]]
    fig.update_layout(title=f"Peso de {experto} en las {len(anclas)} anclas (peso global {g:.2f})")
    fig.update_xaxes(title=ejes[0])
    fig.update_yaxes(title=ejes[1])
    return _estilo(fig, alto=440, leyenda=False)


def fig_pesos_globales():
    c = modelos()["campo_epl"]
    p = np.array(c["pesos"])
    orden = np.argsort(c["pesos_globales"])
    nombres = [c["expertos"][i] for i in orden]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[c["pesos_globales"][i] for i in orden], y=nombres, orientation="h", name="Peso global π",
                         marker_color=NARANJA, hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.add_trace(go.Box(x=p[:, orden].T.ravel(), y=np.repeat(nombres, p.shape[0]), orientation="h",
                         name="Pesos en las anclas", marker_color=AZUL, boxpoints="all", jitter=0.3, pointpos=0,
                         line=dict(width=1), fillcolor="rgba(0,0,0,0)"))
    fig.update_layout(title="Pesos de cada experto: global y en cada ancla")
    fig.update_xaxes(title="Peso (cada fila de anclas suma 1)", range=[0, 1])
    fig = _estilo(fig, alto=420)
    fig.update_layout(margin=dict(l=140))
    return fig


# ---------------------------------------------------------------------------
# Test, diagnóstico, conjuntos externos, robustez y costo
# ---------------------------------------------------------------------------
def fig_test_bloques():
    """F1 macro de cada modelo en cada bloque de test (el bloque 3002, con 3 viviendas, queda fuera)."""
    t = leer("test_confirmatorio.json")
    orden = sorted(t["f1_macro"], key=t["f1_macro"].get)
    bloques = list(t["f1_por_bloque"][orden[0]])
    z = np.array([[t["f1_por_bloque"][m][b] for b in bloques] for m in orden])
    fig = go.Figure(go.Heatmap(z=z, x=[f"{b} ({t['bloques'][b]:,} viv.)".replace(",", " ") for b in bloques], y=orden,
                               zmin=0, zmax=0.8, colorscale=[[0, "#f0efec"], [1, "#0d366b"]], text=np.round(z, 2),
                               texttemplate="%{text}", colorbar=dict(title="F1", thickness=12),
                               hovertemplate="%{y}, bloque %{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(title="F1 macro por bloque de test")
    fig = _estilo(fig, alto=470, leyenda=False)
    fig.update_layout(margin=dict(l=160))
    return fig


def fig_diagnostico(modelo=ORIGINAL, vista="confusion"):
    d = modelos()["diagnosticos"][modelo]
    if vista == "confusion":
        cm = np.array(d["confusion"], float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        et = [f"E{k}" for k in range(1, 7)]
        fig = go.Figure(go.Heatmap(z=norm, x=et, y=et, zmin=0, zmax=1, colorscale=[[0, "#ffffff"], [1, "#0d366b"]],
                                   text=cm.astype(int), texttemplate="%{z:.2f}<br>(%{text})",
                                   colorbar=dict(title="Proporción", thickness=12),
                                   hovertemplate="Real %{y}, predicho %{x}: %{z:.1%}<extra></extra>"))
        fig.update_layout(title=f"Matriz de confusión de {modelo} (por fila)")
        fig.update_xaxes(title="Estrato predicho")
        fig.update_yaxes(title="Estrato real", autorange="reversed")
        return _estilo(fig, alto=460, leyenda=False)
    if vista == "calibracion":
        c = d["calibracion"]
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                                    line=dict(color=GRIS, dash="dot")),
                         go.Scatter(x=c["confianza"], y=c["acierto"], mode="lines+markers", name=modelo,
                                    marker=dict(size=[max(5, min(18, n / 400)) for n in c["n"]],
                                                color=_color(modelo)), line=dict(color=_color(modelo)),
                                    customdata=c["n"],
                                    hovertemplate="Confianza %{x:.2f}: acierto %{y:.2f} (%{customdata:,} viv.)<extra></extra>")])
        fig.update_layout(title=f"Calibración de {modelo}")
        fig.update_xaxes(title="Probabilidad de la clase predicha", range=[0, 1])
        fig.update_yaxes(title="Proporción de aciertos", range=[0, 1])
        return _estilo(fig, alto=460)
    if vista == "importancia":
        imp = list(modelos()["importancia"][modelo].items())[:12][::-1]
        fig = go.Figure(go.Bar(x=[v for _, v in imp], y=[k for k, _ in imp], orientation="h",
                               marker_color=_color(modelo), hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig.update_layout(title=f"Importancia global de {modelo}: media de |valor de Shapley|")
        fig.update_xaxes(title="Aporte medio a la probabilidad de la clase predicha")
        fig = _estilo(fig, alto=460, leyenda=False)
        fig.update_layout(margin=dict(l=170))
        return fig
    curvas = d["roc"] if vista == "roc" else d["pr"]
    fig = go.Figure()
    for k, c in curvas.items():
        fig.add_trace(go.Scatter(x=c["x"], y=c["y"], mode="lines", name=f"E{k} ({c['area']:.2f})",
                                 line=dict(color=ESTRATOS[int(k) - 1], width=2)))
    if vista == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color=GRIS, dash="dot")))
    fig.update_layout(title=(f"Curvas ROC de {modelo} (uno contra el resto; área entre paréntesis)" if vista == "roc"
                             else f"Precisión-recall de {modelo} (área entre paréntesis)"))
    fig.update_xaxes(title="Tasa de falsos positivos" if vista == "roc" else "Recall", range=[0, 1])
    fig.update_yaxes(title="Tasa de verdaderos positivos" if vista == "roc" else "Precisión", range=[0, 1.02])
    return _estilo(fig, alto=460)


def fig_externos(conjunto="king_county"):
    e = leer("resumen_externos_v2.json")[conjunto]
    orden = sorted(e["media"], key=e["media"].get)
    fig = go.Figure(go.Bar(x=[e["media"][m] for m in orden], y=orden, orientation="h",
                           marker_color=[_color(m) for m in orden],
                           error_x=dict(type="data", array=[e["sd"][m] for m in orden], color=GRIS, thickness=1.2),
                           text=[f"{e['media'][m]:.3f}" for m in orden], textposition="inside", insidetextanchor="end",
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title=f"{NOMBRE_EXTERNO[conjunto]}: F1 macro en 15 pliegues (3 × 5)")
    fig.update_xaxes(title="F1 macro (media ± desviación)")
    fig.update_yaxes(showgrid=False)
    fig = _estilo(fig, alto=450, leyenda=False)
    fig.update_layout(margin=dict(l=150))
    return fig


def fig_robustez(vista="condiciones"):
    if vista == "condiciones":
        r = leer("robustez_resumen.json")["condiciones"]
        fig = go.Figure()
        colores = {"EPL": NARANJA, "promedio_simple": AZUL_CLARO, "pesos_error_global": "#5598e7",
                   "Random Forest": AZUL, "CatBoost": "#0d366b"}
        conds = list(r)
        for k, nombre in NOMBRE_REGLA.items():
            fig.add_trace(go.Bar(x=[NOMBRE_COND[c] for c in conds], y=[r[c].get(k) for c in conds], name=nombre,
                                 marker_color=colores[k], hovertemplate=f"{nombre}: %{{y:.3f}}<extra></extra>"))
        fig.update_layout(barmode="group", title="F1 macro bajo condiciones adversas (repetición 0, 5 pliegues)")
        fig.update_yaxes(title="F1 macro (media)", range=[0.55, 0.7])
        return _estilo(fig, alto=440)
    s = leer("robustez_sensibilidad.json")
    grupos = {"n_anclas": "Anclas J", "ancho_banda": "Ancho de banda h", "k_anclas": "Anclas vecinas k", "lam": "λ (penalización KL)"}
    fig = go.Figure()
    for (g, nombre), color in zip(grupos.items(), [AZUL, VERDE, TINTA, AZUL_CLARO]):
        claves = [k for k in s if k.startswith(g + "=")]
        fig.add_trace(go.Scatter(x=[f"{nombre} = {k.split('=')[1]}" for k in claves], y=[s[k]["media"] for k in claves],
                                 mode="lines+markers", name=nombre, line=dict(color=color),
                                 hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    esp = [k for k in s if k.startswith("espacio=")]
    fig.add_trace(go.Scatter(x=[f"espacio {NOMBRE_ESPACIO[k.split('=')[1]]}" for k in esp], y=[s[k]["media"] for k in esp],
                             mode="markers", name="Espacio de competencia", marker=dict(size=10, color=NARANJA)))
    fig.update_layout(title="Sensibilidad: F1 macro medio en los 25 pliegues al mover un hiperparámetro")
    fig.update_yaxes(title="F1 macro (media)", range=[0.60, 0.635])
    fig.update_xaxes(tickangle=-40)
    fig = _estilo(fig, alto=470)
    fig.update_layout(margin=dict(b=130))
    return fig


SERIES_COMPLEJIDAD = {"combinador vs n": "Combinador frente a n (viviendas)", "combinador vs M": "Combinador frente a M (expertos)",
                      "combinador vs J": "Combinador frente a J (anclas)", "EPL completo vs n": "EPL completo frente a n",
                      "EPL completo vs p": "EPL completo frente a p (variables)",
                      "inferencia por vivienda (ms) vs n de entrenamiento": "Predicción por vivienda frente a n"}


def fig_complejidad(serie="combinador vs n"):
    c = leer("complejidad.json")
    t = [r for r in c["tabla"] if r["serie"] == serie]
    x, y = np.array([r["tamano"] for r in t], float), np.array([r["segundos"] for r in t], float)
    b = c["pendientes"][serie]
    a = np.mean(np.log(y) - b * np.log(x))
    xx = np.geomspace(x.min(), x.max(), 20)
    unidad = "ms por vivienda" if "ms" in serie else "segundos"
    fig = go.Figure([go.Scatter(x=x, y=y, mode="markers", name="Medido", marker=dict(size=11, color=NARANJA),
                                hovertemplate=f"%{{x:,}}: %{{y:.3f}} {unidad}<extra></extra>"),
                     go.Scatter(x=xx, y=np.exp(a) * xx ** b, mode="lines", name=f"Ajuste log-log: pendiente {b:.2f}",
                                line=dict(color=TINTA, dash="dot"))])
    fig.update_layout(title=SERIES_COMPLEJIDAD[serie])
    fig.update_xaxes(type="log", title="Tamaño (escala log)")
    fig.update_yaxes(type="log", title=f"Tiempo ({unidad}, escala log)")
    return _estilo(fig, alto=420)


# ---------------------------------------------------------------------------
# Predicción en vivo
# ---------------------------------------------------------------------------
def fig_mapa_clic(n=2500):
    v = viviendas().sample(n, random_state=1)
    fig = go.Figure()
    for k in range(1, 7):
        s = v[v.estrato_num == k]
        fig.add_trace(go.Scattergl(x=s.x_km, y=s.y_km, mode="markers", name=f"E{k}",
                                   marker=dict(size=4, color=ESTRATOS[k - 1]),
                                   hovertemplate=f"Estrato {k}<br>x %{{x}} km, y %{{y}} km<extra></extra>"))
    fig.update_layout(title="Clic en el mapa para ubicar la vivienda")
    fig.update_xaxes(title="Este-oeste (km)")
    fig.update_yaxes(title="Sur-norte (km)", scaleanchor="x", scaleratio=1)
    return _estilo(fig, alto=420)


def fig_probabilidad(proba):
    fig = go.Figure(go.Bar(x=[f"Estrato {k}" for k in range(1, 7)], y=proba, marker_color=ESTRATOS,
                           text=[f"{100 * p:.1f} %" for p in proba], textposition="outside",
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_layout(title="Probabilidad que asigna el EPL a cada estrato")
    fig.update_yaxes(title="Probabilidad", tickformat=".0%", range=[0, 1.1])
    return _estilo(fig, alto=300, leyenda=False)


def fig_shapley(phi, base, fx, k):
    orden = sorted(phi.items(), key=lambda t: abs(t[1]))[-10:]
    fig = go.Figure(go.Bar(x=[v for _, v in orden], y=[n for n, _ in orden], orientation="h",
                           marker_color=[AZUL if v >= 0 else NARANJA for _, v in orden],
                           text=[f"{v:+.3f}" for _, v in orden], textposition="outside",
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_layout(title=f"Por qué: aporte de cada variable a la probabilidad del estrato {k} (valores de Shapley)")
    fig.update_xaxes(title=f"Aporte a la probabilidad (de {base:.2f} en promedio a {fx:.2f} en esta vivienda)")
    fig = _estilo(fig, alto=380, leyenda=False)
    fig.update_layout(margin=dict(l=170))
    return fig
