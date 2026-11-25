"""Simula deriva de datos y deriva conceptual en el proyecto de Ecobici y
elige el umbral de la alerta de deriva. Desde psets/proyecto/, despues de
python servicio/exportar_modelo.py:

    python -m monitoreo.simular_deriva

La ventana es lo que recibe el servicio en un dia: 100 estaciones x 96
franjas = 9600 predicciones. La referencia son los dias de entrenamiento (1 al
24 de agosto). Escribe:

- servicio/referencia_deriva.json: cortes y proporciones de las features
  vigiladas, lo que necesita el monitor del servicio (servicio/deriva.py).
- monitoreo/deriva_ventanas_normales.csv: el PSI de cada dia de agosto, sin
  ninguna deriva.
- monitoreo/deriva_simulacion.csv: PSI, MAE y costo asimetrico de los dias de
  prueba (25 al 31) con cada deriva simulada, en promedio.
- monitoreo/deriva_umbrales.csv: falsas alarmas y deteccion de cada umbral.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import (CATEGORICAS, NUMERICAS, OBJETIVO, RUTA_DATOS, ULTIMO_DIA_ENTRENAMIENTO,
                                        YA_LISTAS, agregar_calendario, agregar_features)
from monitoreo.desempeno import costo_asimetrico
from servicio.deriva import FEATURES_VIGILADAS, construir_referencia, proporciones_numericas, psi

AQUI = Path(__file__).resolve().parent
RUTA_MODELO = AQUI.parent / "servicio" / "modelo.joblib"
RUTA_REFERENCIA = AQUI.parent / "servicio" / "referencia_deriva.json"
CONTEOS = ["salidas", "llegadas", "salidas_lag_15m", "salidas_ultima_hora", "salidas_mismo_momento_ayer", OBJETIVO]


def predecir(modelo, datos):
    """Igual que el servicio: features de pipeline_proyecto y recorte a 0."""
    X = agregar_features(agregar_calendario(datos.copy()))[NUMERICAS + CATEGORICAS + YA_LISTAS].copy()
    X["es_fin_de_semana"] = X["es_fin_de_semana"].astype(int)
    return np.clip(modelo.predict(X), 0, None)


def medir(modelo, referencia, dia):
    medidas = {f"psi_{f}": round(psi(referencia[f]["proporciones"],
                                     proporciones_numericas(dia[f], referencia[f]["cortes"])), 4)
               for f in FEATURES_VIGILADAS}
    estimada = predecir(modelo, dia)
    real = dia[OBJETIVO].to_numpy()
    medidas["mae"] = round(float(np.abs(real - estimada).mean()), 4)
    medidas["costo_asimetrico"] = round(float(costo_asimetrico(real, estimada)), 4)
    return medidas


def dia_con_otra_demanda(dia, factor, semilla):
    """Deriva de datos: la demanda de todo el dia es `factor` veces la normal,
    en las entradas y en el objetivo por igual. Con factor < 1, un dia de
    lluvia o de vacaciones (cada viaje se conserva con probabilidad `factor`);
    con factor > 1, un evento o una ciclovia nueva (a cada conteo se le suman
    viajes con probabilidad `factor - 1` por viaje). Cambia cuanto llega, no
    la relacion entre lo reciente y lo que viene."""
    dia = dia.copy()
    azar = np.random.default_rng(semilla)
    for columna in CONTEOS:
        presentes = dia[columna].notna()
        conteos = dia.loc[presentes, columna].astype(int).to_numpy()
        if factor <= 1:
            dia.loc[presentes, columna] = azar.binomial(conteos, factor)
        else:
            dia.loc[presentes, columna] = conteos + azar.binomial(conteos, factor - 1)
    return dia


def dia_con_mas_salidas(dia, factor):
    """Deriva conceptual: las salidas de los proximos 30 minutos son `factor`
    veces las de siempre con la misma historia reciente (por ejemplo, una
    promocion o una estacion vecina cerrada que manda usuarios a estas). Las
    entradas no cambian; cambia lo que hay que predecir."""
    dia = dia.copy()
    dia[OBJETIVO] = dia[OBJETIVO] * factor
    return dia


def alerta(medidas, umbral):
    return (medidas[[f"psi_{f}" for f in FEATURES_VIGILADAS]] > umbral).any(axis=1)


if __name__ == "__main__":
    modelo = joblib.load(RUTA_MODELO)
    datos = pd.read_csv(RUTA_DATOS, dtype={"estacion_codigo": str}).dropna(subset=[OBJETIVO])
    datos["dia"] = pd.to_datetime(datos["fecha"]).dt.day
    entrenamiento = datos[datos["dia"] <= ULTIMO_DIA_ENTRENAMIENTO]
    referencia = construir_referencia(entrenamiento)
    RUTA_REFERENCIA.write_text(json.dumps(referencia, indent=2), encoding="utf-8")
    print("Referencia:", {f: len(r["proporciones"]) for f, r in referencia.items()}, "casillas")

    # Paso 1: cuanto se mueve un dia normal (los 31 de agosto).
    dias = {fecha: dia for fecha, dia in datos.groupby("fecha")}
    normales = pd.DataFrame([{"fecha": f, "es_prueba": int(f[-2:]) > ULTIMO_DIA_ENTRENAMIENTO,
                              **medir(modelo, referencia, d)} for f, d in dias.items()])
    normales.to_csv(AQUI / "deriva_ventanas_normales.csv", index=False)
    print("\nDias normales:\n", normales.describe().loc[["mean", "max"]].round(4).to_string())

    # Paso 2: las derivas, en los 7 dias de prueba, por intensidad.
    prueba = [d for f, d in dias.items() if int(f[-2:]) > ULTIMO_DIA_ENTRENAMIENTO]
    filas = []
    detalle = {}
    for factor in [1.0, 0.75, 0.5, 1.25, 1.5]:
        medidas = pd.DataFrame([medir(modelo, referencia, dia_con_otra_demanda(d, factor, s))
                                for s, d in enumerate(prueba)])
        detalle[factor] = medidas
        filas.append({"escenario": f"datos: demanda al {factor:.0%}", **medidas.mean().round(4).to_dict()})
    for factor in [1.0, 1.1, 1.25, 1.5]:
        medidas = pd.DataFrame([medir(modelo, referencia, dia_con_mas_salidas(d, factor)) for d in prueba])
        filas.append({"escenario": f"conceptual: salidas x{factor}", **medidas.mean().round(4).to_dict()})
    simulacion = pd.DataFrame(filas)
    simulacion.to_csv(AQUI / "deriva_simulacion.csv", index=False)
    print("\nDerivas simuladas (promedio de los 7 dias de prueba):\n", simulacion.to_string(index=False))

    # Paso 3: el umbral. Falsas alarmas en los 31 dias normales; deteccion en
    # los 7 dias de prueba con cada deriva de datos.
    umbrales = pd.DataFrame([{"umbral": u, "falsas_alarmas": round(alerta(normales, u).mean(), 3),
                              **{f"deteccion_demanda_{f:.0%}": round(alerta(detalle[f], u).mean(), 3)
                                 for f in [0.75, 0.5, 1.25, 1.5]}}
                             for u in [0.05, 0.10, 0.12, 0.15, 0.20]])
    umbrales.to_csv(AQUI / "deriva_umbrales.csv", index=False)
    print("\nUmbrales:\n", umbrales.to_string(index=False))
