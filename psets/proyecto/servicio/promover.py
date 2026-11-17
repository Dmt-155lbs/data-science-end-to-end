"""Pone el alias "produccion" en una version de "ecobici-rebalanceo". Desde
psets/proyecto/:

    python -m servicio.promover 1

Despues hay que volver a exportar el modelo (python servicio/exportar_modelo.py)
y reconstruir la imagen del servicio: el contenedor no llega a MLflow.
"""

import logging
import sys

from mlflow import MlflowClient

from servicio.registrar_modelo import NOMBRE

logging.getLogger("mlflow").setLevel(logging.ERROR)

ALIAS = "produccion"

if __name__ == "__main__":
    version = sys.argv[1]
    cliente = MlflowClient()
    cliente.set_registered_model_alias(NOMBRE, ALIAS, version)
    actual = cliente.get_model_version_by_alias(NOMBRE, ALIAS)
    mae = cliente.get_run(actual.run_id).data.metrics.get("mae_prueba")
    print(f"{NOMBRE}@{ALIAS} es ahora la version {actual.version} (corrida {actual.run_id[:12]}, MAE {mae:.4f})")
