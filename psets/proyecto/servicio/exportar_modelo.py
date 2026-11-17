"""Exporta la version de "ecobici-rebalanceo" con el alias "produccion" a
servicio/modelo.joblib, y guarda su numero en servicio/modelo_version.txt. La
imagen lleva los dos archivos: el contenedor no necesita llegar a MLflow.

Ninguno de los dos se sube a git: se regeneran con este script, desde
psets/proyecto/ y con el servidor de MLflow levantado:

    python servicio/exportar_modelo.py
"""

import logging
from pathlib import Path

import joblib
import mlflow.sklearn
from mlflow import MlflowClient

NOMBRE = "ecobici-rebalanceo"
ALIAS = "produccion"
CARPETA = Path(__file__).resolve().parent

logging.getLogger("mlflow").setLevel(logging.ERROR)

version = MlflowClient().get_model_version_by_alias(NOMBRE, ALIAS)
modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{version.version}")
joblib.dump(modelo, CARPETA / "modelo.joblib")
(CARPETA / "modelo_version.txt").write_text(version.version, encoding="utf-8")
print(f"Version {version.version} de {NOMBRE} (alias {ALIAS}, corrida {version.run_id[:12]}) exportada a servicio/modelo.joblib")
