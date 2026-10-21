import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)

# Tabla de flujo de Ecobici (estacion x dia x franja de 15 min, agosto 2026,
# 100 estaciones). Se genera con preparar_datos.py porque el warehouse de
# Snowflake ya no esta disponible. estacion_codigo se lee como texto porque
# hay codigos de estacion doble como "271-272".
datos = pd.read_csv(
    "../../../data/processed/flujo_estaciones_2026_08.csv",
    dtype={"estacion_codigo": str},
)

# Salidas de la estacion en los proximos 30 minutos (2 franjas).
OBJETIVO = "y_salidas_prox_30m"

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))

# Paso 2: distribucion de la variable objetivo.
plt.figure()
plt.hist(datos[OBJETIVO].dropna(), bins=40)
plt.title(OBJETIVO)
plt.savefig("figuras/distribucion_objetivo.png")
plt.close()

# Paso 3: atipicos de la variable objetivo con la regla del IQR.
q1 = datos[OBJETIVO].quantile(0.25)
q3 = datos[OBJETIVO].quantile(0.75)
iqr = q3 - q1
fuera = datos[(datos[OBJETIVO] < q1 - 1.5 * iqr) | (datos[OBJETIVO] > q3 + 1.5 * iqr)]
print(f"Atipicos de {OBJETIVO} segun IQR: {len(fuera)}")
print(fuera.head(10))

# Paso 4: salidas_ultima_hora es la variable que mas espero que influya:
# una estacion con mucha demanda en la ultima hora deberia seguir con
# demanda en la proxima media hora.
plt.figure()
plt.scatter(datos["salidas_ultima_hora"], datos[OBJETIVO], s=4, alpha=0.1)
plt.xlabel("salidas_ultima_hora")
plt.ylabel(OBJETIVO)
plt.title("Salidas de la ultima hora contra salidas en los proximos 30 min")
plt.savefig("figuras/ultima_hora_vs_objetivo.png")
plt.close()

# Paso 5: consultas para confirmar los hallazgos.

# Hallazgo 1 (filtro): casi la mitad de las franjas no tienen ninguna salida
# en los proximos 30 minutos.
print("Proporcion de franjas con objetivo 0:", (datos[OBJETIVO] == 0).mean().round(3))

# Hallazgo 2 (groupby): la demanda depende de la hora y del tipo de dia.
print(datos.groupby("hora")[OBJETIVO].mean().round(2))
print(datos.groupby("es_fin_de_semana")[OBJETIVO].mean().round(2))

# Hallazgo 3 (correlacion): la historia reciente de la estacion explica el
# objetivo mejor que sus atributos fijos.
columnas = [
    "salidas",
    "salidas_lag_15m",
    "salidas_ultima_hora",
    "salidas_mismo_momento_ayer",
    "capacidad",
]
print(datos[columnas].corrwith(datos[OBJETIVO]).round(3))
