"""Resumen del capítulo 1 del Entregable 1 (base de datos y calidad) para el dashboard.

Usa el CSV crudo y las reglas de limpieza del Entregable 1 (``src/limpieza.py``) y guarda
solo agregados en ``datos/entregable1.json``: no se publica ningún número predial ni
coordenada. Uso: ``python preparar_entregable1.py RUTA_AL_LIBRO_E1``.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_E1 = Path(sys.argv[1] if len(sys.argv) > 1 else "../entregable1_jbook").resolve()
sys.path.insert(0, str(RAIZ_E1))
from src import config as C      # noqa: E402
from src import limpieza as L    # noqa: E402

crudo = L.cargar_crudo()
df, embudo, tabla_imposibles = L.limpiar(crudo, verbose=False)
y = df[C.OBJETIVO]
N = len(df)
out = {"filas_descargadas": int(len(crudo)), "viviendas": int(N)}

# Unidades de conteo
npn = df["numero_predial_nacional"].astype("string")
out["predios"] = int(npn.nunique())
out["predios_matriz"] = int(df["npn_edificio"].nunique())

# % por estrato: todas vs con coordenadas
con = df["tiene_coordenadas"].astype(bool)
todas = y.value_counts().sort_index()
cc = y[con].value_counts().reindex(todas.index, fill_value=0)
out["estratos"] = [{"estrato": int(e), "todas": int(todas[e]), "con_coord": int(cc[e])} for e in todas.index]
out["sin_coordenadas"] = int((~con).sum())

# Faltantes por variable (y cuántos vienen de reglas de limpieza)
vars_num = [v for v in ["area_construida", "area_catastral_terreno", "total_habitaciones", "total_banios",
                        "total_plantas", "planta_ubicacion", "altura", "anio_construccion"] if v in df]
out["faltantes"] = [{"variable": v, "pct": float(100 * df[v].isna().mean()),
                     "imposible": float(100 * df[f"flag_imposible_{v}"].mean()) if f"flag_imposible_{v}" in df else 0.0,
                     "incoherente": float(100 * df[f"flag_incoherente_{v}"].mean()) if f"flag_incoherente_{v}" in df else 0.0}
                    for v in vars_num]
out["filas_con_imposible"] = int(df.filter(like="flag_imposible_").any(axis=1).sum())
out["filas_con_incoherente"] = int(df.filter(like="flag_incoherente_").any(axis=1).sum())
out["tabla_imposibles"] = json.loads(tabla_imposibles.reset_index().to_json(orient="records", force_ascii=False))


# Grupos, ICC(1) y efecto de diseño de Kish (con y sin la excepción de la condición 5)
def icc_kish(grupo):
    g = y.groupby(grupo)
    m = g.size()
    k = len(m)
    mu = y.mean()
    msb = (m * (g.mean() - mu) ** 2).sum() / (k - 1)
    msw = ((y - g.transform("mean")) ** 2).sum() / (N - k)
    m0 = (N - (m ** 2).sum() / N) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw)
    mp = (m ** 2).sum() / N
    deff = 1 + (mp - 1) * icc
    return m, g, {"grupos": int(k), "max": int(m.max()), "media": float(m.mean()), "m_pond": float(mp),
                  "icc": float(icc), "deff": float(deff), "n_ef": float(N / deff)}


es_mejora = npn.str[21] == "5"
grupo = df["npn_edificio"].where(~es_mejora, npn)
m, g, con_exc = icc_kish(grupo)
_, _, sin_exc = icc_kish(df["npn_edificio"])
mayor = m.idxmax()
multi = m[m > 1].index
con_exc.update({
    "torres_mayor": int(npn[grupo == mayor].str[22:24].nunique()),
    "grupos_1": int((m == 1).sum()), "multi": int(len(multi)), "filas_multi": int(m[multi].sum()),
    "multi_no_uniformes": int((g.nunique().loc[multi] > 1).sum()),
    "grupos_100": int((m >= 100).sum()), "filas_100": int(m[m >= 100].sum())})
out["mejoras"] = {"viviendas": int(es_mejora.sum()), "prefijos": int(df.loc[es_mejora, "npn_edificio"].nunique()),
                  "grupos": int(npn[es_mejora].nunique())}
out["grupo"] = {"con_excepcion": con_exc, "sin_excepcion": sin_exc}
# Viviendas según el tamaño del grupo al que pertenecen
cortes = [0, 1, 4, 19, 49, 99, 499, 10_000]
etq = ["1", "2–4", "5–19", "20–49", "50–99", "100–499", "500 o más"]
tam = grupo.map(m)
out["viviendas_por_tamano"] = [{"tamano": e, "viviendas": int(v)} for e, v in
                               pd.cut(tam, cortes, labels=etq).value_counts().reindex(etq).items()]

dest = Path(__file__).with_name("datos") / "entregable1.json"
dest.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k not in ("tabla_imposibles",)}, ensure_ascii=False, indent=1,
                 default=float))
