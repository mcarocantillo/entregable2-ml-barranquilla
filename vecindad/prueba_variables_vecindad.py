import sys
from pathlib import Path as _Path
import os as _os
AQUI = _Path(__file__).resolve().parent
VEC = str(AQUI.parent / "datos" / "vecindad")   # resultados (JSON) e intermedios (pickles, no se suben)
_os.makedirs(VEC, exist_ok=True)
sys.path.insert(0, str(AQUI))
import numpy as np, pandas as pd
from variables_vecindad import calcular_vecindad
rng = np.random.default_rng(1)
n = 3000
# edificios con 1-6 unidades en el mismo punto; puntos en celdas centradas para que la rejilla no cambie el resultado
nb = 900
pos = (rng.integers(1, 80, size=(nb, 2)) + 0.5) * 0.05
ed = rng.integers(0, nb, n)
df = pd.DataFrame({"x_km": pos[ed, 0], "y_km": pos[ed, 1], "npn_edificio": ed.astype(str),
                   "area_construida": rng.uniform(20, 300, n), "total_banios": rng.integers(0, 5, n).astype(float),
                   "antiguedad": rng.uniform(0, 60, n),
                   "uso": rng.choice(["Residencial Apartamentos 4 y mas pisos en PH", "Residencial Vivienda Hasta 3 Pisos"], n),
                   "condicion_predio": rng.choice(["Informal", "NPH", "PH_Unidad_Predial"], n)})
df = pd.concat([pd.DataFrame({'x_km':[0.0],'y_km':[0.0],'npn_edificio':['ancla'],'area_construida':[50.0],'total_banios':[1.0],'antiguedad':[5.0],'uso':['Residencial Vivienda Hasta 3 Pisos'],'condicion_predio':['NPH']}), df], ignore_index=True)
n = len(df)
df.loc[rng.choice(n, 200, replace=False), 'total_banios'] = np.nan
V = calcular_vecindad(df, radios_km=(0.3,))
# fuerza bruta: vecinos a distancia <= r (en celdas) y de OTRO edificio
xy = df[["x_km", "y_km"]].to_numpy(); area = np.log1p(df.area_construida.to_numpy())
apto = df.uso.str.contains("Apartamentos").to_numpy(float); ban = df.total_banios.to_numpy(float)
ok = []
for i in range(0, n, 37):
    d = np.hypot(*(xy - xy[i]).T)
    m = (d <= 0.3 + 1e-9) & (df.npn_edificio.to_numpy() != df.npn_edificio.iloc[i])
    ok.append((abs(V["vec300_area_log"].iloc[i] - (area[m].mean() if m.any() else np.nan)) < 1e-6 or (np.isnan(V["vec300_area_log"].iloc[i]) and not m.any()),
               abs(V["vec300_apto"].iloc[i] - apto[m].mean()) < 1e-6 if m.any() else np.isnan(V["vec300_apto"].iloc[i]),
               abs(V["vec300_dens"].iloc[i] - np.log1p(m.sum())) < 1e-6,
               (abs(V["vec300_banios"].iloc[i] - np.nanmean(ban[m])) < 1e-6) if m.any() and (~np.isnan(ban[m])).any() else True))
ok = np.array(ok); print("coinciden con fuerza bruta (area, apto, dens, baños):", ok.mean(0), "de", len(ok), "viviendas")

# --- depuración
xy = df[["x_km","y_km"]].to_numpy(); ed = df.npn_edificio.to_numpy()
bad = 0
for i in range(0, n, 37):
    d = np.hypot(*(xy - xy[i]).T)
    m = (d <= 0.3 + 1e-9) & (ed != ed[i])
    cnt = np.expm1(V["vec300_dens"].iloc[i])
    if abs(cnt - m.sum()) > 0.5 and bad < 6:
        bad += 1
        m_all = (d <= 0.3 + 1e-9)
        print(f"i={i} fuerza bruta (otro edificio)={m.sum()} | función={cnt:.2f} | fuerza bruta incluyendo propio edificio={m_all.sum()} | unidades propio edificio={(ed==ed[i]).sum()}")
