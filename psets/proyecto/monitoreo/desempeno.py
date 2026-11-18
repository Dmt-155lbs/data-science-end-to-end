"""Une el registro de predicciones del servicio con las salidas reales y
calcula el desempeno del modelo por dia y por version. Desde psets/proyecto/:

    python -m monitoreo.desempeno

La verdad de cada prediccion son las salidas de esa estacion en los 30
minutos siguientes (y_salidas_prox_30m del archivo limpio): llega 30 minutos
despues. Escribe monitoreo/desempeno_por_dia.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import OBJETIVO, RUTA_DATOS
from servicio.api import RUTA_REGISTRO

MAE_REFERENCIA = 1.0278          # MAE en prueba de la version 1 (docs/decision_modelo.md)
MAE_VALIDACION_DESVIACION = 0.0369  # variacion entre pliegues de la validacion temporal
COSTO_SUBESTIMAR, COSTO_SOBREESTIMAR = 3, 1
UMBRAL_MAE_DIA = 1.25            # por encima del peor dia normal de la semana de prueba (1.12)
UMBRAL_COSTO_SEMANA = 2.30       # 15% sobre el 2.00 de la validacion temporal
UMBRAL_COSTO_DIA = 2.60          # por encima del peor dia normal (2.35)
SALIDA = Path(__file__).resolve().parent / "desempeno_por_dia.csv"


def costo_asimetrico(real, estimada):
    diferencia = real - estimada
    return np.where(diferencia > 0, COSTO_SUBESTIMAR * diferencia, -COSTO_SOBREESTIMAR * diferencia).mean()


def unir_con_la_verdad(registro, verdad):
    """Cada prediccion con las salidas reales de su estacion, fecha y franja."""
    verdad = verdad[["estacion_codigo", "fecha", "franja", OBJETIVO]].rename(columns={OBJETIVO: "salidas_reales"})
    return registro.merge(verdad, on=["estacion_codigo", "fecha", "franja"], how="inner")


def resumir(grupo):
    real, estimada = grupo["salidas_reales"], grupo["salidas_prox_30m"]
    return pd.Series({
        "predicciones": len(grupo),
        "mae": round(float((real - estimada).abs().mean()), 4),
        "costo_asimetrico": round(float(costo_asimetrico(real, estimada)), 4),
        "prediccion_media": round(float(estimada.mean()), 3),
        "salidas_reales_media": round(float(real.mean()), 3),
    })


if __name__ == "__main__":
    registro = pd.read_json(RUTA_REGISTRO, lines=True, dtype={"estacion_codigo": str, "version_modelo": str})
    registro["fecha"] = pd.to_datetime(registro["fecha"]).dt.strftime("%Y-%m-%d")
    verdad = pd.read_csv(RUTA_DATOS, dtype={"estacion_codigo": str}).dropna(subset=[OBJETIVO])
    unidas = unir_con_la_verdad(registro, verdad)
    print(f"{len(registro)} predicciones en el registro, {len(unidas)} unidas con su verdad")

    por_dia = unidas.groupby(["fecha", "version_modelo"]).apply(resumir, include_groups=False)
    por_dia["veces_la_referencia"] = (por_dia["mae"] / MAE_REFERENCIA).round(2)
    por_dia.to_csv(SALIDA)
    print("\nPor dia:")
    print(por_dia.to_string())

    print("\nPor version:")
    print(unidas.groupby("version_modelo").apply(resumir, include_groups=False).to_string())

    # Los umbrales de docs/monitoreo.md. El de la semana sale de la validacion
    # temporal (bloques de varios dias); un dia suelto varia mas (entre semana
    # y fin de semana), asi que su umbral queda por encima del peor dia normal.
    alertas = []
    semanal = resumir(unidas)
    if semanal["mae"] > MAE_REFERENCIA + 2 * MAE_VALIDACION_DESVIACION:
        alertas.append(f"MAE de la semana {semanal['mae']} > {MAE_REFERENCIA + 2 * MAE_VALIDACION_DESVIACION:.3f}")
    if semanal["costo_asimetrico"] > UMBRAL_COSTO_SEMANA:
        alertas.append(f"costo asimetrico de la semana {semanal['costo_asimetrico']} > {UMBRAL_COSTO_SEMANA}")
    for (fecha, version), fila in por_dia.iterrows():
        if fila["mae"] > UMBRAL_MAE_DIA:
            alertas.append(f"{fecha}, version {version}: MAE {fila['mae']} > {UMBRAL_MAE_DIA}")
        if fila["costo_asimetrico"] > UMBRAL_COSTO_DIA:
            alertas.append(f"{fecha}, version {version}: costo asimetrico {fila['costo_asimetrico']} > {UMBRAL_COSTO_DIA}")
    print("\nAlertas del modelo:", "ninguna" if not alertas else "")
    for alerta in alertas:
        print("-", alerta)
