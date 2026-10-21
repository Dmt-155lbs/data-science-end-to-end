import os

import numpy as np
import pandas as pd

# La misma ruta que usaste en la exploracion de la Semana 8.
# estacion_codigo se lee como texto por los codigos dobles como "271-272".
datos = pd.read_csv(
    "../../semana-08-sesion-02/data/flujo_estaciones_2026_08.csv",
    dtype={"estacion_codigo": str},
)

# La columna que el modelo va a predecir: salidas en los proximos 30 min.
OBJETIVO = "y_salidas_prox_30m"

# Feature 1: periodo de demanda segun la hora. Sale del hallazgo de la
# Semana 8: no hay salidas de 1 a 4 am y la demanda sube 8-9h y 14-18h.
# Solo usa la hora actual, que se conoce al predecir.
hora = datos["hora"]
datos["periodo_demanda"] = np.select(
    [hora < 5, (hora >= 7) & (hora <= 9), (hora >= 14) & (hora <= 18)],
    ["madrugada", "pico_manana", "pico_tarde"],
    default="resto",
)

# Feature 2: salidas de la ultima hora relativas a la capacidad de la
# estacion (demanda reciente por anclaje). Usa solo el pasado y el atributo
# fijo de la estacion. La capacidad minima es 11, asi que no divide entre 0.
datos["salidas_hora_por_capacidad"] = (
    datos["salidas_ultima_hora"] / datos["capacidad"].clip(lower=1)
)

# Evidencia: compara el objetivo segun cada feature nueva.
print(datos.groupby("periodo_demanda")[OBJETIVO].mean().round(2))
print(
    datos[["salidas_hora_por_capacidad", "salidas_ultima_hora", OBJETIVO]]
    .corr()[OBJETIVO]
    .round(3)
)

print(datos.head())

# Guardar el dataset con las 2 features para la siguiente sesion.
os.makedirs("../data", exist_ok=True)
datos.to_csv("../data/flujo_estaciones_features_2026_08.csv", index=False)
print("Guardado:", datos.shape)
