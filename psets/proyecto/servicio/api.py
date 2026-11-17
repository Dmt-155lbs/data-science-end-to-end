"""Servicio de prediccion del proyecto de Ecobici.

Recibe una franja de 15 minutos de una estacion, como la conoce quien llama
(el lote de cada 15 minutos o un tablero), y devuelve cuantas bicicletas se
estima que salen en los proximos 30 minutos. Desde psets/proyecto/, despues
de python servicio/exportar_modelo.py:

    uvicorn servicio.api:app --port 8000
"""

import os
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import CATEGORICAS, NUMERICAS, YA_LISTAS, agregar_calendario, agregar_features

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", CARPETA / "modelo.joblib"))
RUTA_VERSION = CARPETA / "modelo_version.txt"

# El modelo y su version del registro (ecobici-rebalanceo@produccion, los dos
# exportados por exportar_modelo.py) se cargan una sola vez, al arrancar.
modelo = joblib.load(RUTA_MODELO) if RUTA_MODELO.exists() else None
version_servida = RUTA_VERSION.read_text(encoding="utf-8").strip() if RUTA_VERSION.exists() else None

app = FastAPI(title="Salidas de Ecobici en los proximos 30 minutos")


class Franja(BaseModel):
    """Lo que se sabe de una estacion al cierre de una franja de 15 minutos.
    Los rezagos pueden llegar vacios (sin historia): la preparacion los imputa."""
    estacion_codigo: str
    fecha: date
    hora: int = Field(ge=0, le=23)
    capacidad: int = Field(gt=0)
    salidas: int = Field(ge=0)
    llegadas: int = Field(ge=0)
    salidas_lag_15m: float | None = None
    salidas_ultima_hora: float | None = None
    salidas_mismo_momento_ayer: float | None = None


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "version_modelo": version_servida}


@app.post("/predecir")
def predecir(franja: Franja):
    # Las mismas features del entrenamiento, con las funciones de pipeline_proyecto.
    datos = agregar_features(agregar_calendario(pd.DataFrame([franja.model_dump()])))
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS].copy()
    X["es_fin_de_semana"] = X["es_fin_de_semana"].astype(int)
    salidas = float(modelo.predict(X)[0])
    return {
        "estacion_codigo": franja.estacion_codigo,
        "salidas_prox_30m": round(max(salidas, 0.0), 2),
        "version_modelo": version_servida,
    }
