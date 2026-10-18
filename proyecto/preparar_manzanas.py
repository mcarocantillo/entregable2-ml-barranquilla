"""Datos de la recuperación por manzana (Entregable 1, capítulo 1.7) para el dashboard final.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Para cada manzana con edificios ubicados: la distancia del edificio más lejano a su centroide y cuántas viviendas
sin coordenadas propias tiene. Con eso el dashboard recalcula, para cualquier umbral, cuántas viviendas se
recuperarían (la regla del Entregable 1 usa 250 m y excluye la manzana comodín "0000").

Uso:  python proyecto/preparar_manzanas.py RUTA_AL_ENTREGABLE1
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent.parent
E1 = Path(sys.argv[1]).resolve()
SALIDA = AQUI / "datos" / "proyecto"
SALIDA.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(E1))
from src import config as C  # noqa: E402
from src.limpieza import proyectar_km  # noqa: E402

con = pd.read_parquet(C.CARPETA_PROCESADOS / "dataset_modelado.parquet")
sin = pd.read_parquet(C.CARPETA_PROCESADOS / "sin_coordenadas.parquet")
todo = pd.concat([con, sin], ignore_index=True)
por_manzana = todo["ubicado_por_manzana"].fillna(False).astype(bool)
propias = todo[todo["tiene_coordenadas"] & ~por_manzana]

# Distancia de cada edificio ubicado (punto distinto) al centroide de su manzana
pts = propias[["npn_manzana", "centroide_lat", "centroide_lon"]].drop_duplicates()
px, py = proyectar_km(pts["centroide_lat"], pts["centroide_lon"])
pts = pts.assign(x=np.asarray(px), y=np.asarray(py))
cen = pts.groupby("npn_manzana")[["x", "y"]].transform("mean")
pts["d_m"] = 1000 * np.hypot(pts.x - cen.x, pts.y - cen.y)
manz = pts.groupby("npn_manzana").agg(n_puntos=("d_m", "size"), dist_max_m=("d_m", "max"))
manz["generica"] = manz.index.str[13:17] == C.CODIGO_MANZANA_GENERICA

# Viviendas sin coordenadas propias (recuperadas o no) por manzana y por estrato
sin_propias = todo[~todo["tiene_coordenadas"] | por_manzana]
n_sin = sin_propias.groupby(["npn_manzana", C.OBJETIVO]).size().unstack(fill_value=0)
manz = manz.join(n_sin, how="left").fillna(0)
cand = manz[~manz.generica]

cob = todo.groupby(C.OBJETIVO).agg(total=("tiene_coordenadas", "size"), v2=("tiene_coordenadas", "sum"))
cob["por_manzana"] = por_manzana.groupby(todo[C.OBJETIVO]).sum()
cob["v1"] = cob["v2"] - cob["por_manzana"]
recup = todo[por_manzana]

salida = {
    "umbral_m": C.DISPERSION_MAX_MANZANA_M,
    "dist_max_m": manz.loc[~manz.generica & (manz.n_puntos > 1), "dist_max_m"].round(1).tolist(),
    "n_manzanas": int(len(manz)), "n_genericas": int(manz.generica.sum()),
    "n_validas": int((~manz.generica & (manz.dist_max_m <= C.DISPERSION_MAX_MANZANA_M)).sum()),
    "cobertura": {int(k): {c: int(v) for c, v in r.items()} for k, r in cob.iterrows()},
    "recuperadas_condicion": recup["condicion_predio"].astype(str).value_counts().to_dict(),
    # Para el slider: por manzana candidata, su dispersión y las viviendas sin coordenadas propias por estrato
    "candidatas": {"dist_max_m": cand["dist_max_m"].round(1).tolist(),
                   "sin_coord_por_estrato": {int(k): cand[k].astype(int).tolist() for k in C.CLASES if k in cand}},
    "sin_coord_total_por_estrato": sin_propias[C.OBJETIVO].value_counts().sort_index().astype(int).to_dict(),
}
(SALIDA / "manzanas.json").write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
umbral = C.DISPERSION_MAX_MANZANA_M
rec = sum(np.array(v)[np.array(salida["candidatas"]["dist_max_m"]) <= umbral].sum()
          for v in salida["candidatas"]["sin_coord_por_estrato"].values())
print(f"manzanas: {len(manz):,} | candidatas: {len(cand):,} | recuperables con {umbral} m: {rec:,} "
      f"(guardadas como recuperadas: {int(por_manzana.sum()):,})")
