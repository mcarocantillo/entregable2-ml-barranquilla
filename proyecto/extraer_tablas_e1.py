"""Extrae las tablas de los cuadernos ejecutados del Entregable 1 (versión 2) para el dashboard final.

Recorre cada cuaderno, guarda cada salida HTML (DataFrame) como tabla con un identificador estable
(cuaderno + sección + orden), el título de la sección y el código que la produjo.

Uso:  python proyecto/extraer_tablas_e1.py RUTA_AL_ENTREGABLE1
"""
import io
import json
import re
import sys
from pathlib import Path

import nbformat
import pandas as pd

E1 = Path(sys.argv[1])
SALIDA = Path(__file__).resolve().parent.parent / "datos" / "proyecto"
SALIDA.mkdir(parents=True, exist_ok=True)

tablas = {}
for nb_path in sorted((E1 / "notebooks").glob("[0-9][0-9]_*.ipynb")):
    nb = nbformat.read(nb_path, as_version=4)
    cap = nb_path.stem[:2]
    seccion, n = "intro", 0
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            m = re.search(r"^#{2,3} +(\d+(?:\.\d+)*)?\.? *(.*)$", cell.source, re.M)
            if m:
                seccion, n = (m.group(1) or m.group(2)[:30]).strip(), 0
            continue
        for o in cell.get("outputs", []):
            html = o.get("data", {}).get("text/html")
            if not html:
                continue
            try:
                dfs = pd.read_html(io.StringIO(html))
            except ValueError:
                continue
            for df in dfs:
                n += 1
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [" | ".join(str(x) for x in c if not str(x).startswith("Unnamed")) for c in df.columns]
                df.columns = [str(c) for c in df.columns]
                clave = f"{cap}_{seccion}_{n}"
                tablas[clave] = {"cuaderno": nb_path.stem, "seccion": seccion, "codigo": cell.source[:400],
                                 "columnas": list(df.columns),
                                 "filas": json.loads(df.to_json(orient="values", force_ascii=False))}

(SALIDA / "tablas_e1.json").write_text(json.dumps(tablas, ensure_ascii=False), encoding="utf-8")
print(len(tablas), "tablas")
for k, v in tablas.items():
    print(f"{k:28s} {len(v['filas']):4d} x {len(v['columnas']):2d} | {', '.join(v['columnas'][:6])[:90]}")
