"""Figuras del dashboard para lo agregado al proyecto después del Entregable 2 (19-oct-2026).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Tres fuentes, todas copiadas en ``datos/`` (nada se reentrena aquí):

- ``datos/proyecto/eda_mejorado.json``: EDA mejorado, capítulo 15 del libro del Entregable 1 (versión 3). Lo genera
  ``proyecto/extraer_eda_mejorado.py`` a partir del cuaderno ejecutado.
- ``datos/revision_v3/``: todos los modelos originales de la revisión con vecindad física y todos los datos, y sus
  pruebas de significancia (capítulo 9 del libro del Entregable 2; código en la carpeta ``mejoras/`` del Entregable 3).
- ``datos/e3/``: resultados del Entregable 3 (pipeline combinatorio, optimizadores, balanceo, cómputo, test,
  calibración, regresión, comparación estadística, interpretabilidad y el modelo nuevo frente al base), copiados de la
  carpeta ``resultados/`` de su repositorio.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from figuras import COLOR_ESTRATO, DATOS, _estilo

AZUL, AZUL_CLARO, NARANJA, ROJO, VERDE, GRIS, TINTA = (
    "#2a78d6", "#86b6ef", "#eb6834", "#e34948", "#1baf7a", "#a3a29d", "#52514e")
BASE = "Modelo base (logística B, sin vecindad)"
LV = "Logística con vecindad"
ORIGINALES = ["Rotation Forest", "Stacking S1 (logística, XGBoost, RF)", "Stacking S2 (S1 + XGBoost geográfico)",
              "XGBoost geográfico (configuración del E2)", "XGBoost geográfico afinado", "RF + SVM (dos etapas)"]
CORTO = {BASE: "Modelo base (logística B)", LV: "Logística con vecindad",
         "Stacking S1 (logística, XGBoost, RF)": "Stacking S1 (logística, XGBoost, RF)",
         "Stacking S2 (S1 + XGBoost geográfico)": "Stacking S2 (S1 + XGB geográfico)",
         "XGBoost geográfico (configuración del E2)": "XGBoost geográfico (config. E2)",
         "XGBoost geográfico afinado": "XGBoost geográfico afinado",
         "SVM sola (referencia del RF + SVM)": "SVM sola (referencia)"}


def cientifico(x, dec=1):
    m, e = f"{x:.{dec}e}".split("e")
    return f"{m}·10" + str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def corto(n):
    return CORTO.get(n, n)


def tipo(n):
    return "original" if n in ORIGINALES else ("base" if n == BASE else "referencia")


COLOR_TIPO = {"original": NARANJA, "referencia": AZUL, "base": TINTA}
NOMBRE_TIPO = {"original": "original de la revisión", "referencia": "referencia (no original)", "base": "modelo base"}


@lru_cache(maxsize=None)
def _leer(ruta):
    p = DATOS / ruta
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def eda():
    return _leer("proyecto/eda_mejorado.json")


def mejoras():
    return _leer("revision_v3/mejoras.json")


def estadistica():
    return _leer("revision_v3/estadistica.json")


def geo_busqueda():
    return _leer("revision_v3/geo_busqueda.json")


def e3(nombre):
    return _leer(f"e3/{nombre}.json")


# ---------------------------------------------------------------------------
# EDA mejorado (Entregable 1, cap. 15)
# ---------------------------------------------------------------------------
def fig_eta2_entorno():
    """η² de Kruskal-Wallis de cada atributo de la vivienda frente al mismo atributo promediado en sus 8 vecinos."""
    d = eda()
    if not d:
        return None
    filas = sorted(d["eta2"].items(), key=lambda kv: kv[1]["vecinos"])
    nombres = [k for k, _ in filas]
    propia = [v["propia"] for _, v in filas]
    vec = [v["vecinos"] for _, v in filas]
    fig = go.Figure()
    for n, a, b in zip(nombres, propia, vec):
        fig.add_shape(type="line", x0=a, x1=b, y0=n, y1=n, line=dict(color="#c3c2b7", width=3), layer="below")
    fig.add_trace(go.Scatter(x=propia, y=nombres, mode="markers", name="la propia vivienda",
                             marker=dict(size=15, color=AZUL_CLARO, line=dict(color="white", width=2)),
                             hovertemplate="%{y}, propia: η² = %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=vec, y=nombres, mode="markers+text", name="sus 8 vecinos (promedio)",
                             marker=dict(size=15, color="#0d366b", line=dict(color="white", width=2)),
                             text=[f"×{b / a:.1f}" for a, b in zip(propia, vec)], textposition="middle right",
                             textfont=dict(color=TINTA, size=13),
                             hovertemplate="%{y}, 8 vecinos: η² = %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0.14, line_color=GRIS, line_dash="dot",
                  annotation_text="efecto grande (0.14)", annotation_position="top right",
                  annotation_font=dict(size=12, color=TINTA))
    fig.update_xaxes(title="η² de Kruskal-Wallis con el estrato (más es más asociación)", range=[0, 0.7])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="¿Qué dice más del estrato: la vivienda o su entorno?")
    return _estilo(fig, alto=430)


def fig_fronteras(vista="pct"):
    d = eda()
    if not d:
        return None
    fr = d["fronteras"]
    est = [k for k in fr if k != "todas"]
    if vista == "pct":
        y = [fr[k]["pct_frontera"] for k in est]
        total, fmt, titulo_y = fr["todas"]["pct_frontera"], "{:.0f} %", "% de viviendas en frontera"
        titulo = "Viviendas en frontera: la mitad o más de sus 8 edificios vecinos tiene otro estrato"
    else:
        y = [100 * fr[k]["vecinos_otro_estrato"] for k in est]
        total, fmt, titulo_y = 100 * fr["todas"]["vecinos_otro_estrato"], "{:.0f} %", "% medio de vecinos con otro estrato"
        titulo = "De los 8 edificios vecinos, ¿cuántos tienen otro estrato? (promedio)"
    fig = go.Figure(go.Bar(x=[f"Estrato {k}" for k in est], y=y, marker_color=[COLOR_ESTRATO[int(k)] for k in est],
                           text=[fmt.format(v) for v in y], textposition="inside", insidetextanchor="end",
                           textfont=dict(color="white"), showlegend=False,
                           hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[f"Estrato {k}" for k in est], y=[total] * len(est), mode="lines",
                             name=f"todas las viviendas: {total:.0f} %", line=dict(color=NARANJA, dash="dash", width=2),
                             hoverinfo="skip"))
    fig.update_yaxes(title=titulo_y, range=[0, max(y) * 1.2])
    fig.update_layout(title=titulo)
    return _estilo(fig, alto=430)


def fig_composicion_test(vista="estratos"):
    d = eda()
    if not d:
        return None
    if vista == "estratos":
        c = d["composicion"]
        fig = go.Figure()
        for nombre, color in (("entrenamiento (%)", AZUL_CLARO), ("prueba (%)", NARANJA)):
            v = c[nombre]
            fig.add_trace(go.Bar(x=[f"Estrato {k}" for k in v], y=list(v.values()), name=nombre.replace(" (%)", ""),
                                 marker_color=color, text=[f"{x:.1f} %" for x in v.values()], textposition="outside",
                                 hovertemplate="%{x}: %{y:.1f} %<extra>" + nombre.replace(" (%)", "") + "</extra>"))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06,
                          title="Composición por estrato: entrenamiento frente a test")
        fig.update_yaxes(title="% de las viviendas", range=[0, 42])
        return _estilo(fig, alto=430)
    b = d["test_por_bloque"]
    bloques = sorted(b, key=lambda k: -b[k]["total"])
    fig = go.Figure()
    for k in range(1, 7):
        fig.add_trace(go.Bar(x=[f"bloque {x}" for x in bloques], y=[b[x][f"E{k}"] for x in bloques], name=f"E{k}",
                             marker_color=COLOR_ESTRATO[k],
                             hovertemplate="%{x}: %{y:,} viviendas de estrato " + str(k) + "<extra></extra>"))
    fig.update_layout(barmode="stack", title="Los 7 bloques de test: viviendas por estrato")
    fig.update_yaxes(title="viviendas", tickformat=",")
    fig.add_annotation(x="bloque 4004", y=b["4004"]["total"], text="todas las de estrato 6", showarrow=True,
                       arrowhead=2, ay=-40, font=dict(size=13, color=TINTA))
    return _estilo(fig, alto=430)


# ---------------------------------------------------------------------------
# Todos los modelos originales de la revisión (Entregable 2, cap. 9)
# ---------------------------------------------------------------------------
def fig_originales(mirada="test", referencia="base"):
    """Diferencia de F1 macro de cada modelo frente al modelo base (o a la logística con vecindad): validación
    (± 1 SE pareado) o test (IC 95 % BCa por bloques)."""
    m = mejoras()
    if not m:
        return None
    clave = "vs_base" if referencia == "base" else "vs_logistica_vecindad"
    excluir = BASE if referencia == "base" else LV
    filas = []
    for n in m["test"]:
        if n == excluir:
            continue
        if mirada == "test":
            t = m["test"][n][clave]
            if not t:
                continue
            filas.append((n, t["dif"], t["ic95_bca"][0], t["ic95_bca"][1], f"{t['bloques_a_favor']}/{t['n_bloques']} bloques"))
        else:
            c = m["cv"][n][clave]
            filas.append((n, c["media"], c["media"] - c["SE"], c["media"] + c["SE"], f"{c['pliegues_a_favor']}/5 pliegues"))
    filas.sort(key=lambda f: f[1])
    fig = go.Figure()
    for t in ("original", "referencia", "base"):
        sel = [f for f in filas if tipo(f[0]) == t]
        if not sel:
            continue
        x = np.array([f[1] for f in sel])
        fig.add_trace(go.Scatter(
            x=x, y=[corto(f[0]) for f in sel], mode="markers", name=NOMBRE_TIPO[t],
            marker=dict(size=14, color=COLOR_TIPO[t], symbol="diamond" if t == "original" else "circle",
                        line=dict(color="white", width=1.5)),
            error_x=dict(type="data", symmetric=False, array=np.array([f[3] for f in sel]) - x,
                         arrayminus=x - np.array([f[2] for f in sel]), color=COLOR_TIPO[t], thickness=2.5),
            customdata=[[f[2], f[3], f[4]] for f in sel],
            hovertemplate="%{y}<br>Δ F1 = %{x:+.3f} [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]"
                          "<br>gana en %{customdata[2]}<extra></extra>"))
    fig.add_vline(x=0, line_color=TINTA, line_dash="dot")
    ref = "al modelo base" if referencia == "base" else "a la logística con vecindad"
    banda = "IC 95 % BCa" if mirada == "test" else "± 1 SE pareado"
    fig.update_xaxes(title=f"Δ F1 macro frente {ref} · {banda}", zeroline=False)
    fig.update_yaxes(categoryorder="array", categoryarray=[corto(f[0]) for f in filas], showgrid=False)
    fig.update_layout(title=("Test (7 bloques, mirado una vez)" if mirada == "test" else "Validación kNNDM (5 pliegues)")
                      + ": ¿cuánto mejora cada modelo?")
    return _estilo(fig, alto=500)


def tabla_originales():
    m = mejoras()
    if not m:
        return None
    cv, te = m["cv"], m["test"]
    ic = lambda v: f"[{v[0]:+.3f}, {v[1]:+.3f}]"  # noqa: E731
    filas = []
    for n in sorted(te, key=lambda n: -te[n]["f1"]):
        f = {"Modelo": corto(n), "Tipo": NOMBRE_TIPO[tipo(n)], "F1 validación": round(cv[n]["f1"], 3),
             "F1 test": round(te[n]["f1"], 3)}
        if n != BASE:
            f["Δ test vs base [IC 95 %]"] = f"{te[n]['vs_base']['dif']:+.3f} {ic(te[n]['vs_base']['ic95_bca'])}"
            f["Bloques a favor"] = f"{te[n]['vs_base']['bloques_a_favor']}/7"
        filas.append(f)
    return pd.DataFrame(filas).fillna("—")


def fig_significancia(familia="base"):
    """30 bloques espaciales de 2 km: δ de Cliff y p de Wilcoxon con Holm (familia = base o lv), o rangos medios de
    Friedman con la diferencia crítica de Nemenyi (familia = rangos)."""
    e = estadistica()
    if not e:
        return None
    if familia == "rangos":
        r = e["friedman"]["rangos"]
        orden = sorted(r, key=lambda n: -r[n])
        fig = go.Figure(go.Scatter(
            x=[r[n] for n in orden], y=[corto(n) for n in orden], mode="markers",
            marker=dict(size=15, color=[COLOR_TIPO[tipo(n)] for n in orden],
                        symbol=["diamond" if tipo(n) == "original" else "circle" for n in orden]),
            showlegend=False, hovertemplate="%{y}: rango medio %{x:.2f}<extra></extra>"))
        cd, mejor = e["friedman"]["cd"], min(r.values())
        fig.add_shape(type="rect", x0=mejor, x1=mejor + cd, y0=-0.5, y1=len(orden) - 0.5, fillcolor=AZUL,
                      opacity=0.08, line_width=0, layer="below")
        fig.add_annotation(x=mejor + cd, y=len(orden) - 0.5, text=f"diferencia crítica de Nemenyi = {cd:.2f}",
                           showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=12, color=TINTA))
        fig.update_xaxes(title="rango medio en los 30 bloques (1 = el mejor en todos)", autorange="reversed")
        fig.update_yaxes(showgrid=False)
        fig.update_layout(title=f"Friedman: χ² = {e['friedman']['chi2']:.1f}, p = {cientifico(e['friedman']['p'])}")
        return _estilo(fig, alto=500, leyenda=False)
    filas = e["comparaciones"]["vs modelo base" if familia == "base" else "vs logística con vecindad"]
    filas = sorted(filas, key=lambda r: r["delta_cliff"])
    sig = [r["wilcoxon_p_holm"] < 0.05 for r in filas]
    fig = go.Figure(go.Bar(
        x=[r["delta_cliff"] for r in filas], y=[corto(r["a"]) for r in filas], orientation="h",
        marker_color=[(COLOR_TIPO[tipo(r["a"])] if s else "#d9d8d2") for r, s in zip(filas, sig)],
        marker_line=dict(color=[COLOR_TIPO[tipo(r["a"])] for r in filas], width=1.5),
        text=[f"p Holm {r['wilcoxon_p_holm']:.3f}" if r["wilcoxon_p_holm"] >= 0.001 else "p Holm < 0.001"
              for r in filas], textposition="outside", showlegend=False,
        customdata=[[r["magnitud"], r["bloques_a_favor"], r["n_bloques"]] for r in filas],
        hovertemplate="%{y}<br>δ de Cliff = %{x:+.2f} (%{customdata[0]})<br>gana en %{customdata[1]} de "
                      "%{customdata[2]} bloques<extra></extra>"))
    for corte, txt in ((0.147, "pequeño"), (0.33, "mediano"), (0.474, "grande")):
        fig.add_vline(x=corte, line_color=GRIS, line_dash="dot", annotation_text=txt, annotation_position="top",
                      annotation_font=dict(size=11, color=TINTA))
    fig.add_vline(x=0, line_color=TINTA)
    ref = "al modelo base" if familia == "base" else "a la logística con vecindad"
    fig.update_xaxes(title="δ de Cliff · barra rellena = significativo (Holm, p < 0.05)", range=[-0.3, 0.75])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title=f"¿Supera cada modelo {ref}? 30 bloques de 2 km, Wilcoxon con Holm", margin=dict(t=110))
    return _estilo(fig, alto=500, leyenda=False)


def fig_busqueda_geo(vista="alpha"):
    g = geo_busqueda()
    if not g:
        return None
    if vista == "alpha":
        c = sorted((float(a), f) for a, f in g["curva_alpha_mejor"].items())
        fig = go.Figure(go.Scatter(x=[a for a, _ in c], y=[f for _, f in c], mode="lines+markers", name="mejor configuración (G5 + L5)",
                                   line=dict(color=NARANJA, width=2.5), marker=dict(size=8),
                                   hovertemplate="α = %{x:.2f}: F1 %{y:.4f}<extra></extra>"))
        e2 = g["referencia_E2"]["f1"]
        fig.add_hline(y=e2, line_color=AZUL, line_dash="dash", annotation_text=f"configuración del E2: {e2:.3f}",
                      annotation_position="bottom right", annotation_font=dict(color=AZUL, size=13))
        fig.add_annotation(x=1, y=g["xgboost_afinado_alpha1"]["f1"], text="α = 1: XGBoost sin parte geográfica",
                           showarrow=True, arrowhead=2, ax=-150, ay=-45, font=dict(size=12, color=TINTA))
        fig.update_xaxes(title="α (peso del modelo global; 1 − α es el peso de los modelos locales)")
        fig.update_yaxes(title="F1 macro en validación kNNDM", tickformat=".3f")
        fig.update_layout(title=f"Búsqueda con todos los datos: {g['n_combinaciones']:,} combinaciones".replace(",", " "))
        return _estilo(fig, alto=430)
    m = mejoras()["test"]
    nombres = ["XGBoost geográfico (configuración del E2)", "XGBoost geográfico afinado", "XGBoost afinado"]
    cv = [g["referencia_E2"]["f1"], g["mejor"]["f1"], g["xgboost_afinado_alpha1"]["f1"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[corto(n) for n in nombres], y=cv, name="validación kNNDM", marker_color=AZUL_CLARO,
                         text=[f"{v:.3f}" for v in cv], textposition="outside"))
    te = [m[n]["f1"] for n in nombres]
    fig.add_trace(go.Bar(x=[corto(n) for n in nombres], y=te, name="test", marker_color=NARANJA,
                         text=[f"{v:.3f}" for v in te], textposition="outside"))
    fig.update_layout(barmode="group", title="La búsqueda sube la validación, no el test")
    fig.update_yaxes(title="F1 macro", range=[0.5, 0.62])
    return _estilo(fig, alto=430)


# ---------------------------------------------------------------------------
# Entregable 3
# ---------------------------------------------------------------------------
def fig_e3_modelos():
    a = e3("analisis_clasificacion")
    if not a:
        return None
    filas = sorted(a["mejor_por_modelo"], key=lambda r: r["f1_macro_mean"])
    fig = go.Figure(go.Bar(
        x=[r["f1_macro_mean"] for r in filas], y=[r["modelo"] for r in filas], orientation="h",
        marker_color=[NARANJA if r["modelo"] == "XGBoost geográfico" else AZUL for r in filas],
        error_x=dict(type="data", array=[r["f1_macro_std"] for r in filas], color=GRIS, thickness=1.5),
        text=[f"{r['f1_macro_mean']:.3f}" for r in filas], textposition="inside", insidetextanchor="start",
        textfont=dict(color="white"), showlegend=False,
        customdata=[[r["balanceo"], r["metodo"]] for r in filas],
        hovertemplate="%{y}: F1 %{x:.3f}<br>mejor combinación: %{customdata[0]} + %{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title="F1 macro en el bucle externo (media ± DE de 5 pliegues kNNDM)", range=[0, 0.8])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title="La mejor de las 16 combinaciones de cada modelo")
    return _estilo(fig, alto=430, leyenda=False)


ORDEN_OPT = ["Grid", "Random", "Bayesiana", "Genética"]
COLOR_OPT = {"Grid": GRIS, "Random": AZUL_CLARO, "Bayesiana": AZUL, "Genética": NARANJA}


def fig_e3_optimizadores():
    a = e3("analisis_clasificacion")
    if not a:
        return None
    fig = go.Figure()
    for m in ORDEN_OPT:
        y = a["anytime"][m]["media"]
        fig.add_trace(go.Scatter(x=list(range(1, len(y) + 1)), y=y, mode="lines+markers", name=m,
                                 line=dict(color=COLOR_OPT[m], width=2.5), marker=dict(size=7),
                                 hovertemplate=m + ", evaluación %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.99, line_color=TINTA, line_dash="dot", annotation_text="99 %", annotation_position="bottom right")
    fig.update_xaxes(title="número de evaluaciones", dtick=1)
    fig.update_yaxes(title="fracción del mejor F1 final", tickformat=".2f")
    fig.update_layout(title="Desempeño en cualquier momento de la búsqueda (promedio de 160 búsquedas por método)")
    return _estilo(fig, alto=430)


def tabla_e3_optimizadores():
    a = e3("analisis_clasificacion")
    o, at = a["optimizadores"], a["anytime"]
    return pd.DataFrame([{"Método": m, "F1 externo": f"{o[m]['f1_macro_mean']:.3f}",
                          "Evaluaciones para el 99 %": str(at[m]["evals_para_99pct"]),
                          "Segundos por búsqueda": f"{o[m]['seg_busqueda_mean']:.0f}",
                          "Rango medio (Friedman)": f"{a['friedman_optimizadores']['rangos'][m]:.2f}"} for m in ORDEN_OPT])


NOMBRE_BAL = {"ninguno": "Ninguno", "class_weight": "class_weight", "smote": "SMOTE", "adasyn": "ADASYN"}


def fig_e3_balanceo(metrica="f1"):
    a = e3("analisis_clasificacion")
    if not a:
        return None
    b, rk = a["balanceo"], a["friedman_balanceo"]["rangos"]
    orden = sorted(rk, key=lambda k: rk[k])
    col, titulo_y, fmt = {"f1": ("f1_macro_mean", "F1 macro externo (media)", "{:.3f}"),
                          "rango": (None, "rango medio de Friedman (1 = mejor)", "{:.2f}"),
                          "ece": ("ece_mean", "ECE: error de calibración (menos es mejor)", "{:.3f}"),
                          "tiempo": ("seg_busqueda_mean", "segundos por búsqueda", "{:.0f}")}[metrica]
    y = [rk[k] if col is None else b[k][col] for k in orden]
    colores = [VERDE if k == "class_weight" else (ROJO if k == "adasyn" else AZUL_CLARO) for k in orden]
    fig = go.Figure(go.Bar(x=[NOMBRE_BAL[k] for k in orden], y=y, marker_color=colores,
                           text=[fmt.format(v) for v in y], textposition="outside", showlegend=False,
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    rango = {"f1": [0.55, 0.6], "rango": [2, 3.1]}.get(metrica)
    fig.update_yaxes(title=titulo_y, range=rango, rangemode=None if rango else "tozero")
    fig.update_layout(title=f"Balanceo de clases (Friedman sobre 160 bloques: p = {cientifico(a['friedman_balanceo']['p'])})")
    return _estilo(fig, alto=430, leyenda=False)


FAMILIAS_COMPLEJIDAD = {
    "SVM": ["SVM | núcleo RBF completo (SVC)", "SVM | Nyström + LinearSVC", "SVM | SGD (hinge)"],
    "XGBoost": ["XGBoost | exact", "XGBoost | hist", "XGBoost | hist + early stopping"],
    "KNN": ["KNN | fuerza bruta (p≈30)", "KNN | KD-Tree (p≈30)", "KNN | Ball Tree (p≈30)", "KNN | fuerza bruta (p=2)",
            "KNN | KD-Tree (p=2)"],
    "Logística": ["Logística | lbfgs", "Logística | SAGA"],
    "Naive Bayes": ["Naive Bayes | fit", "Naive Bayes | partial_fit (lotes de 1000)"],
}


def fig_e3_complejidad(familia="SVM"):
    c = e3("complejidad")
    if not c:
        return None
    med = pd.DataFrame(c["mediciones"])
    fig = go.Figure()
    paleta = [ROJO, AZUL, VERDE, NARANJA, GRIS]
    for i, v in enumerate(FAMILIAS_COMPLEJIDAD[familia]):
        d = med[med.variante == v].sort_values("n")
        b = c["exponentes"].get(v)
        fig.add_trace(go.Scatter(x=d.n, y=d.segundos, mode="lines+markers",
                                 name=f"{v.split(' | ', 1)[1]} (b = {b:.2f})", line=dict(color=paleta[i], width=2.5),
                                 marker=dict(size=8), hovertemplate="n = %{x:,}: %{y:.3f} s<extra></extra>"))
    ns = sorted(med.n.unique())
    fig.update_xaxes(type="log", title="viviendas de entrenamiento (n, escala log)", tickvals=ns,
                     ticktext=[f"{n:,}".replace(",", " ") for n in ns])
    fig.update_yaxes(type="log", title="segundos (escala log)", dtick=1, tickformat="~g")
    fig.update_layout(title=f"{familia}: tiempo frente a n. La pendiente b es el exponente empírico (t ∝ nᵇ)")
    return _estilo(fig, alto=430)


def fig_e3_test(datos="submuestra"):
    f = e3("finalistas")
    if not f:
        return None
    fig = go.Figure()
    if datos == "submuestra":
        filas = [(n, v["test"]["f1_macro"], v["ic95_bca_f1_test"], v["f1_busqueda_final"]) for n, v in f["finalistas"].items()]
        mb = f["modelo_base"]
        filas.append(("Modelo base del E1", mb["test"]["f1_macro"], mb["ic95_bca_f1_test"], float(np.mean(mb["f1_pliegues_externos"]))))
        segunda, nombre2 = [r[3] for r in sorted(filas, key=lambda r: r[1])], "validación (bucle externo)"
        titulo = "Finalistas entrenados con la submuestra de 10 001 viviendas"
    else:
        t = e3("todos_los_datos")["modelos"]
        filas = [("Modelo base del E1" if n.startswith("Logística B") else n, v["test"]["f1_macro"], v["ic95_bca_f1_test"],
                  v.get("f1_test_submuestra")) for n, v in t.items()]
        segunda, nombre2 = [r[3] for r in sorted(filas, key=lambda r: r[1])], "test con la submuestra"
        titulo = "Finalistas reentrenados con las 257 273 viviendas (mismos hiperparámetros)"
    filas.sort(key=lambda r: r[1])
    nombres = [r[0] for r in filas]
    x = np.array([r[1] for r in filas])
    colores = [NARANJA if n == "XGBoost geográfico" else (TINTA if n == "Modelo base del E1" else AZUL) for n in nombres]
    fig.add_trace(go.Scatter(x=x, y=nombres, mode="markers", name="test (IC 95 % BCa por bloques)",
                             marker=dict(size=14, color=colores),
                             error_x=dict(type="data", symmetric=False, array=np.array([r[2][1] for r in filas]) - x,
                                          arrayminus=x - np.array([r[2][0] for r in filas]), color=GRIS, thickness=2),
                             hovertemplate="%{y}: F1 test %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=segunda, y=nombres, mode="markers", name=nombre2,
                             marker=dict(size=13, color="white", line=dict(color=TINTA, width=2)),
                             hovertemplate="%{y}: %{x:.3f}<extra>" + nombre2 + "</extra>"))
    fig.update_xaxes(title="F1 macro", range=[0.25, 0.85])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title=titulo)
    return _estilo(fig, alto=470)


ETQ_SHAP = {"y_km": "Posición norte-sur", "x_km": "Posición este-oeste", "dist_centro_km": "Distancia al centro",
            "knn8_log_area": "Área de los 8 vecinos", "knn8_banios": "Baños de los vecinos",
            "knn8_dist_km": "Distancia media a los vecinos", "knn8_prop_PH": "Propiedad horizontal de los vecinos",
            "knn8_piso": "Piso de los vecinos", "knn8_habitaciones": "Habitaciones de los vecinos",
            "knn8_antiguedad": "Antigüedad de los vecinos", "total_banios": "Baños", "area_construida": "Área construida",
            "antiguedad": "Antigüedad", "planta_ubicacion": "Piso de la unidad", "total_habitaciones": "Habitaciones",
            "condicion_predio_NPH": "Régimen: no PH", "condicion_predio_PH_Unidad_Predial": "Régimen: PH",
            "condicion_predio_Informal": "Régimen: informal"}


def fig_e3_shap(modelo="XGBoost", n=12):
    i = e3("interpretabilidad")
    if not i:
        return None
    imp = i["clasificacion"][modelo]["importancia_global"]
    total = sum(imp.values())
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:n][::-1]
    entorno = lambda k: k.startswith("knn8_") or k in ("x_km", "y_km", "dist_centro_km")  # noqa: E731
    fig = go.Figure(go.Bar(
        x=[100 * v / total for _, v in top], y=[ETQ_SHAP.get(k, k.replace("_", " ")) for k, _ in top], orientation="h",
        marker_color=[AZUL if entorno(k) else NARANJA for k, _ in top], showlegend=False,
        text=[f"{100 * v / total:.0f} %" for _, v in top], textposition="outside",
        hovertemplate="%{y}: %{x:.1f} % de la importancia total<extra></extra>"))
    fig.update_xaxes(title="% de la importancia SHAP total (media de |SHAP|)", range=[0, 100 * top[-1][1] / total * 1.15])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title=f"{modelo}: azul = ubicación y entorno; naranja = la propia vivienda")
    return _estilo(fig, alto=470, leyenda=False)


# ---------------------------------------------------------------------------
# Entregable 3: calibración, regresión, comparación estadística y el modelo nuevo
# ---------------------------------------------------------------------------
NOMBRE_CAL = {"original": "Sin recalibrar", "platt": "Platt (sigmoide)", "isotonica": "Regresión isotónica"}
COLOR_CAL = {"original": TINTA, "platt": AZUL, "isotonica": AZUL_CLARO}


def fig_e3_calibracion(medida="ece"):
    """ECE, Brier o F1 de test de cada finalista antes y después de recalibrar (Platt e isotónica)."""
    f = e3("finalistas")
    if not f:
        return None
    cal = f["calibracion"]
    orden = sorted(cal, key=lambda n: cal[n]["original"]["ece"] if medida != "f1" else -f["finalistas"][n]["test"]["f1_macro"])
    fig = go.Figure()
    for clave, nombre in NOMBRE_CAL.items():
        if medida == "f1":
            x = [f["finalistas"][n]["test"]["f1_macro"] if clave == "original" else cal[n][clave]["f1"] for n in orden]
        else:
            x = [cal[n][clave][medida] for n in orden]
        fig.add_trace(go.Bar(x=x, y=orden, orientation="h", name=nombre, marker_color=COLOR_CAL[clave],
                             hovertemplate="%{y}: %{x:.3f}<extra>" + nombre + "</extra>"))
    titulo = {"ece": "Error de calibración esperado (ECE) en test: más bajo es mejor",
              "brier": "Brier en test: más bajo es mejor",
              "f1": "F1 macro en test: el recalibrado también cambia las decisiones"}[medida]
    fig.update_layout(barmode="group", title=titulo, bargap=0.25)
    fig.update_xaxes(title={"ece": "ECE", "brier": "Brier", "f1": "F1 macro"}[medida])
    fig.update_yaxes(showgrid=False)
    return _estilo(fig, alto=500)


def fig_e3_regresion(metrica="rmse"):
    """Mejor optimizador de cada modelo de regresión: RMSE, MAE o F1 al redondear, media ± desviación entre pliegues."""
    r = e3("resumen_regresion")
    if not r:
        return None
    df = pd.DataFrame(r)
    col = {"rmse": "rmse", "mae": "mae", "f1": "f1_macro_redondeado"}[metrica]
    mejor = df.sort_values("rmse_mean").groupby("modelo").head(1)
    mejor = mejor.sort_values(f"{col}_mean", ascending=metrica != "f1")
    nombres = [m.replace(" (regresión)", "") + f" · {met}" for m, met in zip(mejor.modelo, mejor.metodo)]
    fig = go.Figure(go.Bar(
        x=mejor[f"{col}_mean"], y=nombres, orientation="h", showlegend=False,
        marker_color=[AZUL if m.startswith(("Random Forest", "XGBoost")) else AZUL_CLARO for m in mejor.modelo],
        error_x=dict(type="data", array=mejor[f"{col}_std"], color=GRIS, thickness=1.5),
        text=[f"{v:.3f}" for v in mejor[f"{col}_mean"]], textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    if metrica == "f1":
        a = e3("analisis_clasificacion")
        f1c = max(rr["f1_macro_mean"] for rr in a["mejor_por_modelo"])
        fig.add_vline(x=f1c, line_dash="dot", line_color=NARANJA,
                      annotation_text=f"mejor clasificador: {f1c:.3f}", annotation_position="top")
    titulo = {"rmse": "RMSE en el bucle externo (estratos): más bajo es mejor",
              "mae": "MAE en el bucle externo (estratos): más bajo es mejor",
              "f1": "F1 macro al redondear la predicción al estrato más cercano"}[metrica]
    fig.update_layout(title=titulo)
    fig.update_xaxes(title={"rmse": "RMSE", "mae": "MAE", "f1": "F1 macro"}[metrica])
    fig.update_yaxes(showgrid=False)
    return _estilo(fig, alto=450, leyenda=False)


def fig_e3_rangos():
    """Diagrama de diferencia crítica (Friedman → Nemenyi) de la mejor combinación de cada modelo."""
    a = e3("analisis_clasificacion")
    if not a:
        return None
    f = a["friedman_mejor_por_modelo"]
    r = sorted(f["rangos"].items(), key=lambda kv: kv[1])
    cd, mejor = f["cd"], r[0][1]
    nombres = [n for n, _ in r][::-1]
    vals = [v for _, v in r][::-1]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=mejor, x1=mejor + cd, y0=-0.5, y1=len(r) - 0.5, fillcolor="rgba(42,120,214,0.10)",
                  line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=vals, y=nombres, mode="markers+text", text=[f"{v:.2f}" for v in vals], textposition="middle right",
        marker=dict(size=15, color=[NARANJA if n == "XGBoost geográfico" else (AZUL if v - mejor < cd else GRIS)
                                    for n, v in zip(nombres, vals)], line=dict(color="white", width=2)),
        showlegend=False, hovertemplate="%{y}: rango medio %{x:.2f}<extra></extra>"))
    fig.add_annotation(x=mejor + cd / 2, y=len(r) - 0.3, showarrow=False, text=f"CD = {cd:.2f}: indistinguibles del mejor",
                       font=dict(color=AZUL))
    fig.update_xaxes(title="rango medio en 32 bloques espaciales (1 = siempre el mejor)", range=[2, 7.3])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title=f"Friedman χ² = {f['chi2']:.1f}, p = {cientifico(f['p'])} · franja azul: CD de Nemenyi")
    return _estilo(fig, alto=450, leyenda=False)


def fig_e3_delong(rival="Logística L1/L2"):
    """Diferencia de AUC por estrato (uno contra el resto) del XGBoost geográfico frente a un rival, con Holm."""
    a = e3("analisis_clasificacion")
    if not a:
        return None
    filas = []
    for d in a["delong_finalistas"]:
        A, B = d["A"].split(" | ")[0], d["B"].split(" | ")[0]
        if {A, B} == {"XGBoost geográfico", rival}:
            signo = 1 if A == "XGBoost geográfico" else -1
            filas.append((d["estrato"], signo * d["diferencia"], signo * 1.96 * d["se"], d["p_holm"]))
    filas.sort()
    x = [f"Estrato {e}" for e, *_ in filas]
    y = [v for _, v, _, _ in filas]
    colores = [(VERDE if v > 0 else ROJO) if p < 0.05 else GRIS for _, v, _, p in filas]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colores, showlegend=False,
                           error_y=dict(type="data", array=[abs(s) for *_, s, _ in filas], color=TINTA, thickness=1.5),
                           customdata=[[p] for *_, p in filas],
                           hovertemplate="%{x}: ΔAUC = %{y:+.3f} (p de Holm = %{customdata[0]:.3g})<extra></extra>"))
    fig.add_hline(y=0, line_color=TINTA)
    nombre = "logística con vecindad" if rival == "Logística L1/L2" else rival
    fig.update_layout(title=f"ΔAUC por estrato: XGBoost geográfico − {nombre}")
    fig.update_yaxes(title="diferencia de AUC (IC 95 %)")
    fig.add_annotation(x=0, y=-0.17, xref="paper", yref="paper", xanchor="left", showarrow=False,
                       text="verde o rojo: diferencia significativa con Holm (p < 0.05) · gris: no significativa",
                       font=dict(color=TINTA, size=13))
    fig = _estilo(fig, alto=470, leyenda=False)
    fig.update_layout(margin=dict(b=85))
    return fig


RIVALES_NUEVO = {"base": "Logística B (modelo base del E1)", "logistica": "Logística L1/L2", "xgboost": "XGBoost",
                 "rf": "Random Forest"}


def fig_e3_nuevo(rival="base"):
    """XGBoost geográfico − rival en F1 macro: validación anidada, test con la submuestra y test con todos los datos."""
    f, t = e3("finalistas"), e3("todos_los_datos")
    if not f or not t:
        return None
    nombre = RIVALES_NUEVO[rival]
    g = next(c for c in f["geo_vs_rivales"] if c["rival"] == nombre)
    gt = next(c for c in t["geo_vs_rivales"] if c["rival"] == nombre)
    filas = [("Validación anidada (5 pliegues, ±1 SE)", g["cv_dif_media"], g["cv_dif_media"] - g["cv_se"],
              g["cv_dif_media"] + g["cv_se"], f"{g['cv_pliegues_a_favor']}/5 pliegues"),
             ("Test, submuestra de 10 001 (IC 95 %)", g["test_dif"], *g["test_ic95_bca"],
              f"{g['test_bloques_a_favor']}/{g['test_n_bloques']} bloques"),
             ("Test, todos los datos (IC 95 %)", gt["test_dif"], *gt["test_ic95_bca"],
              f"{gt['test_bloques_a_favor']}/{gt['test_n_bloques']} bloques")][::-1]
    y = [r[0] for r in filas]
    x = np.array([r[1] for r in filas])
    lo, hi = np.array([r[2] for r in filas]), np.array([r[3] for r in filas])
    colores = [VERDE if l > 0 else (ROJO if h < 0 else GRIS) for l, h in zip(lo, hi)]
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers+text", text=[f"{v:+.3f} · {r[4]}" for v, r in zip(x, filas)], textposition="top center",
        marker=dict(size=16, color=colores), showlegend=False,
        error_x=dict(type="data", symmetric=False, array=hi - x, arrayminus=x - lo, color=TINTA, thickness=2),
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=TINTA, line_dash="dot")
    etiqueta = {"base": "modelo base del E1", "logistica": "logística con la misma vecindad",
                "xgboost": "XGBoost (sin modelos locales)", "rf": "Random Forest"}[rival]
    fig.update_layout(title=f"F1 macro: XGBoost geográfico − {etiqueta}")
    fig.update_xaxes(title="diferencia de F1 macro", range=[-0.16, 0.26])
    fig.update_yaxes(showgrid=False)
    fig.add_annotation(x=0, y=-0.3, xref="paper", yref="paper", xanchor="left", showarrow=False,
                       text="verde: gana · rojo: pierde · gris: el intervalo incluye el 0 (no se distingue)",
                       font=dict(color=TINTA, size=13))
    fig = _estilo(fig, alto=420, leyenda=False)
    fig.update_layout(margin=dict(b=95))
    return fig


# ---------------------------------------------------------------------------
# Versión 3 (sugerencia del profesor): qué cambió y por qué
# ---------------------------------------------------------------------------
def fig_cambios(vista="cvtest"):
    """«cvtest»: F1 de validación y de test del modelo base con el protocolo anterior (bloques con buffer de 1 km) y
    con el de la versión 3 (kNNDM). «buffer»: por qué el buffer era de 1 km (F1 de validación y entrenamiento
    excluido según su tamaño; Entregable 1, cap. 10)."""
    if vista == "buffer":
        import figuras_proyecto as FP
        t = FP.tabla_e1("10_Sensibilidad al buffer_1", indice=False)
        b = t["buffer_km"].astype(float)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"{v:g} km" for v in b], y=t["% train excluido"].astype(float), name="% del entrenamiento excluido",
                             marker_color=AZUL_CLARO, yaxis="y2", text=[f"{v:.0f} %" for v in t["% train excluido"].astype(float)],
                             textposition="outside", hovertemplate="buffer %{x}: se excluye el %{y:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=[f"{v:g} km" for v in b], y=t["F1 macro CV"].astype(float), mode="lines+markers+text",
                                 name="F1 macro de validación", line=dict(color=NARANJA, width=3), marker=dict(size=11),
                                 text=[f"{v:.2f}" for v in t["F1 macro CV"].astype(float)], textposition="top center",
                                 hovertemplate="buffer %{x}: F1 %{y:.3f}<extra></extra>"))
        fig.update_layout(title="Protocolo anterior: más buffer quita más fuga, pero también más datos",
                          yaxis=dict(title="F1 macro (validación)", range=[0, 0.6]),
                          yaxis2=dict(title="% del entrenamiento excluido", overlaying="y", side="right", range=[0, 110],
                                      showgrid=False))
        fig.update_xaxes(title="tamaño del buffer (2.6 km = alcance del correlograma)")
        return _estilo(fig, alto=430)
    import figuras_proyecto as FP
    d = FP.v3_e1()
    if not d:
        return None
    c = d["cv_vs_test"]
    filas = [("Versión anterior\n(bloques + buffer de 1 km)", c["v2: bloques + buffer de 1 km"]),
             ("Versión 3\n(kNNDM, sin buffer)", c["v3: kNNDM sin buffer"])]
    x = [f[0].replace("\n", "<br>") for f in filas]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[f[1]["F1 macro CV"] for f in filas], name="validación cruzada", marker_color=AZUL_CLARO,
                         text=[f"{f[1]['F1 macro CV']:.2f}" for f in filas], textposition="outside"))
    fig.add_trace(go.Bar(x=x, y=[f[1]["F1 macro test"] for f in filas], name="test (7 bloques, el mismo)", marker_color=AZUL,
                         text=[f"{f[1]['F1 macro test']:.2f}" for f in filas], textposition="outside"))
    for i, f in enumerate(filas):
        fig.add_annotation(x=x[i], y=max(f[1]["F1 macro CV"], f[1]["F1 macro test"]) + 0.08,
                           text=f"brecha test − validación: {f[1]['test − CV']:+.2f}", showarrow=False,
                           font=dict(size=13, color=TINTA))
    fig.update_layout(barmode="group", title="Modelo base: la validación ahora anticipa el test")
    fig.update_yaxes(title="F1 macro", range=[0, 0.7])
    return _estilo(fig, alto=430)


def fig_optimismo_v3():
    """F1 del modelo base con validación aleatoria, con kNNDM y en el test (protocolo de la versión 3)."""
    import figuras as F
    r = F.resultados()
    o = r["optimismo_validacion_aleatoria"]
    B = "Logística B_fisicas+ubicacion"
    t = r["resultados_test"]["f1_macro"][B]
    nombres = ["Validación aleatoria<br>(filas mezcladas)", "Validación kNNDM<br>(zonas nuevas)", "Test<br>(7 bloques de 2 km)"]
    vals = [o["F1 macro CV"]["aleatoria (StratifiedKFold)"], o["F1 macro CV"]["espacial (kNNDM)"], t]
    err = [o["desv."]["aleatoria (StratifiedKFold)"], o["desv."]["espacial (kNNDM)"], 0]
    fig = go.Figure(go.Bar(x=nombres, y=vals, marker_color=[ROJO, AZUL, TINTA], error_y=dict(array=err, color=GRIS),
                           text=[f"{v:.2f}" for v in vals], textposition="outside", showlegend=False,
                           hovertemplate="%{x}: F1 %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title="F1 macro del modelo base", range=[0, 0.85])
    fig.update_layout(title="Mezclar filas al azar promete un F1 que en zonas nuevas no se cumple")
    return _estilo(fig, alto=430, leyenda=False)


def ece(y, proba, clases=(1, 2, 3, 4, 5, 6), bins=10):
    """Error de calibración esperado de la clase predicha (10 tramos de confianza)."""
    proba = np.asarray(proba)
    conf = proba.max(1)
    acierto = (np.asarray(clases)[proba.argmax(1)] == np.asarray(y)).astype(float)
    cortes = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(cortes[:-1], cortes[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs(acierto[m].mean() - conf[m].mean())
    return float(total)
