"""Entrena el modelo final del proyecto (gradient boosting), mide su MAE en
prueba y lo registra como una version nueva de "ecobici-rebalanceo" en el
registro de modelos de MLflow. Desde psets/proyecto/:

    python -m servicio.registrar_modelo 3

El numero es la profundidad maxima de los arboles (3 es la del modelo final
de la Semana 11). Cada vez que lo corres, el registro guarda una version
nueva; ninguna reemplaza a otra.
"""

import hashlib
import logging
import sys
import time

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import RUTA_DATOS, ULTIMO_DIA_ENTRENAMIENTO, cargar_y_dividir, construir_preparacion

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "ecobici-rebalanceo"


def huella(ruta):
    with open(ruta, "rb") as archivo:
        return hashlib.md5(archivo.read()).hexdigest()[:12]


if __name__ == "__main__":
    profundidad = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()

    mlflow.set_experiment("ecobici-registro")
    with mlflow.start_run(run_name=f"gradient-boosting-profundidad-{profundidad}"):
        estimador = GradientBoostingRegressor(max_depth=profundidad, random_state=42)
        modelo = Pipeline([("preparar", construir_preparacion()), ("modelo", estimador)])
        inicio = time.perf_counter()
        modelo.fit(X_ent, y_ent)
        segundos = time.perf_counter() - inicio
        mae = mean_absolute_error(y_prueba, modelo.predict(X_prueba))

        mlflow.log_params({**estimador.get_params(), "fecha_corte": f"2026-08-{ULTIMO_DIA_ENTRENAMIENTO}",
                           "huella_datos": huella(RUTA_DATOS)})
        mlflow.log_metrics({"mae_prueba": mae, "segundos": segundos})
        info = mlflow.sklearn.log_model(modelo, "modelo", registered_model_name=NOMBRE)

    print(f"Version {info.registered_model_version} de {NOMBRE}: profundidad {profundidad}, MAE {mae:.4f}, {segundos:.1f} s")
