"""Servicio de prediccion del proyecto de Ecobici, instrumentado.

Recibe una franja de 15 minutos de una estacion, como la conoce quien llama
(el lote de cada 15 minutos o un tablero), y devuelve cuantas bicicletas se
estima que salen en los proximos 30 minutos. Ademas mide la latencia y el
codigo de cada prediccion (/metricas) y guarda cada prediccion en
monitoreo/predicciones.jsonl, para unirla despues con las salidas reales
(monitoreo/desempeno.py), y vigila la deriva de datos de lo que recibe
(/deriva, servicio/deriva.py). Desde psets/proyecto/, despues de
python servicio/exportar_modelo.py:

    uvicorn servicio.api:app --port 8000
"""

import json
import os
import time
import uuid
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import CATEGORICAS, NUMERICAS, YA_LISTAS, agregar_calendario, agregar_features
from servicio.deriva import MonitorDeDeriva

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", CARPETA / "modelo.joblib"))
RUTA_VERSION = CARPETA / "modelo_version.txt"
RUTA_REGISTRO = Path(os.environ.get("RUTA_REGISTRO", CARPETA.parent / "monitoreo" / "predicciones.jsonl"))
RUTA_REFERENCIA = CARPETA / "referencia_deriva.json"

# El modelo y su version del registro (ecobici-rebalanceo@produccion, los dos
# exportados por exportar_modelo.py) se cargan una sola vez, al arrancar.
modelo = joblib.load(RUTA_MODELO) if RUTA_MODELO.exists() else None
version_servida = RUTA_VERSION.read_text(encoding="utf-8").strip() if RUTA_VERSION.exists() else None

# La distribucion de entrenamiento de las features vigiladas (la escribe
# monitoreo/simular_deriva.py). Sin ella, /deriva lo dice y /predecir sigue.
monitor = (MonitorDeDeriva(json.loads(RUTA_REFERENCIA.read_text(encoding="utf-8")))
           if RUTA_REFERENCIA.exists() else None)


class Metricas:
    """Las metricas del sistema, en memoria: una latencia y un codigo por
    cada peticion a /predecir."""

    def __init__(self):
        self.latencias_ms = []
        self.codigos = []

    def registrar(self, latencia_ms, codigo):
        self.latencias_ms.append(latencia_ms)
        self.codigos.append(codigo)

    def resumen(self):
        hay = len(self.latencias_ms) > 0
        return {
            "peticiones": len(self.codigos),
            "errores_validacion": sum(codigo == 422 for codigo in self.codigos),
            "errores_servidor": sum(codigo >= 500 for codigo in self.codigos),
            "latencia_p50_ms": round(float(np.percentile(self.latencias_ms, 50)), 2) if hay else None,
            "latencia_p95_ms": round(float(np.percentile(self.latencias_ms, 95)), 2) if hay else None,
        }


metricas = Metricas()
app = FastAPI(title="Salidas de Ecobici en los proximos 30 minutos")


@app.middleware("http")
async def medir(request: Request, call_next):
    """Mide la latencia y el codigo de cada peticion a /predecir. Si la ruta
    falla, la cuenta como error de servidor (500) en vez de perderla."""
    if request.url.path != "/predecir":
        return await call_next(request)
    inicio = time.perf_counter()
    try:
        respuesta = await call_next(request)
    except Exception:
        respuesta = JSONResponse({"detail": "error interno del servicio"}, status_code=500)
    metricas.registrar((time.perf_counter() - inicio) * 1000, respuesta.status_code)
    return respuesta


class Franja(BaseModel):
    """Lo que se sabe de una estacion al cierre de una franja de 15 minutos.
    Los rezagos pueden llegar vacios (sin historia): la preparacion los imputa.
    franja (0 a 95, la franja de 15 minutos del dia) no la usa el modelo: sirve
    para unir la prediccion con las salidas reales en el monitoreo."""
    estacion_codigo: str
    fecha: date
    hora: int = Field(ge=0, le=23)
    franja: int | None = Field(default=None, ge=0, le=95)
    capacidad: int = Field(gt=0)
    salidas: int = Field(ge=0)
    llegadas: int = Field(ge=0)
    salidas_lag_15m: float | None = None
    salidas_ultima_hora: float | None = None
    salidas_mismo_momento_ayer: float | None = None


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "version_modelo": version_servida}


@app.get("/metricas")
def ver_metricas():
    return metricas.resumen()


@app.get("/deriva")
def ver_deriva():
    """El PSI de la ventana actual (el ultimo dia de franjas) contra
    entrenamiento, y si hay alerta de deriva de datos."""
    if monitor is None:
        return {"franjas_en_ventana": 0, "umbral": None, "psi": {}, "alerta": False,
                "detalle": "sin referencia de deriva"}
    return monitor.estado()


@app.post("/predecir")
def predecir(franja: Franja):
    # Las mismas features del entrenamiento, con las funciones de pipeline_proyecto.
    datos = agregar_features(agregar_calendario(pd.DataFrame([franja.model_dump()])))
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS].copy()
    X["es_fin_de_semana"] = X["es_fin_de_semana"].astype(int)
    salidas = round(max(float(modelo.predict(X)[0]), 0.0), 2)
    id_prediccion = uuid.uuid4().hex

    # El registro guarda todo lo que hace falta para unir la prediccion con la
    # verdad (estacion, fecha y franja) y para saber que modelo la produjo.
    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_REGISTRO, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps({
            "id_prediccion": id_prediccion,
            "momento": datetime.now().isoformat(),
            "version_modelo": version_servida,
            **franja.model_dump(mode="json"),
            "salidas_prox_30m": salidas,
        }) + "\n")
    if monitor is not None:
        monitor.registrar(franja.model_dump())

    return {
        "id_prediccion": id_prediccion,
        "estacion_codigo": franja.estacion_codigo,
        "salidas_prox_30m": salidas,
        "version_modelo": version_servida,
    }
