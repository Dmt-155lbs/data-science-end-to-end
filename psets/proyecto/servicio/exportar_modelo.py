"""Exporta el modelo final del registro de MLflow a servicio/modelo.joblib.

El archivo no se sube a git: se regenera con este script, desde psets/proyecto/
y con el servidor de MLflow levantado (variable MLFLOW_TRACKING_URI):

    python servicio/exportar_modelo.py
"""

import logging
from pathlib import Path

import joblib
import mlflow.sklearn

from version import RUN_ID

RUTA_MODELO = Path(__file__).resolve().parent / "modelo.joblib"

logging.getLogger("mlflow").setLevel(logging.ERROR)

modelo = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/modelo")
joblib.dump(modelo, RUTA_MODELO)
print(f"Modelo de la corrida {RUN_ID} exportado a {RUTA_MODELO}")
