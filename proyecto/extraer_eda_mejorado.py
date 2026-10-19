"""Extrae las cifras del EDA mejorado (capítulo 15 del libro del Entregable 1, versión 3) para el dashboard final.

Lee las salidas del cuaderno ejecutado ``notebooks/15_eda_mejorado.ipynb`` (no recalcula nada) y las guarda en
``datos/proyecto/eda_mejorado.json``: el η² de cada atributo de la vivienda frente al de sus 8 vecinos, las
fronteras entre estratos, la composición del test y las distancias al dato más cercano.

Uso:  python proyecto/extraer_eda_mejorado.py RUTA_AL_ENTREGABLE1_V3
"""
import io
import json
import re
import sys
from pathlib import Path

import nbformat
import pandas as pd

E1 = Path(sys.argv[1])
SALIDA = Path(__file__).resolve().parent.parent / "datos" / "proyecto" / "eda_mejorado.json"

nb = nbformat.read(E1 / "notebooks" / "15_eda_mejorado.ipynb", as_version=4)
textos, tablas = [], []
for cell in nb.cells:
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            textos.append(o["text"])
        html = o.get("data", {}).get("text/html")
        if html:
            tablas.extend(pd.read_html(io.StringIO(html)))
texto = "\n".join(textos)


def plano(df):
    """Aplana encabezados de dos niveles y usa la primera columna como índice."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [next((str(x) for x in c if not str(x).startswith("Unnamed")), str(c[-1])) for c in df.columns]
    return df.set_index(df.columns[0])


eta, fr, comp, bloques, dist = (plano(t) for t in tablas[:5])
datos = {
    "fuente": "Entregable 1 (versión 3), capítulo 15: notebooks/15_eda_mejorado.ipynb",
    "eta2": {k: {"propia": float(r.iloc[0]), "vecinos": float(r.iloc[1])} for k, r in eta.iterrows()},
    "eta2_distancia_vecinos": float(re.search(r"η² = ([\d.]+)", texto).group(1)),
    "fronteras": {str(k): {"vecinos_otro_estrato": float(r.iloc[0]), "pct_frontera": float(r.iloc[1])}
                  for k, r in fr.iterrows()},
    "pct_un_solo_estrato": float(re.search(r"el ([\d.]+) % difiere en un solo estrato", texto).group(1)),
    "composicion": {str(k): {str(c): float(v) for c, v in r.items()} for k, r in comp.iterrows()},
    "test_por_bloque": {str(k): {str(c): int(v) for c, v in r.items()} for k, r in bloques.iterrows()},
    "distancias": {str(k): {str(c): float(v) for c, v in r.items()} for k, r in dist.iterrows()},
}
SALIDA.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(datos, ensure_ascii=False, indent=1)[:3000])
