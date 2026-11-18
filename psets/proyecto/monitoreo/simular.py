"""Simula una semana de produccion: manda al servicio, en orden, cada franja
de 15 minutos del periodo de prueba (25 al 31 de agosto de 2026) de las 100
estaciones, como lo haria el lote de cada 15 minutos. Cada prediccion queda
en monitoreo/predicciones.jsonl. Desde psets/proyecto/, despues de
python servicio/exportar_modelo.py:

    python -m monitoreo.simular

Usa TestClient: corre el servicio de verdad (middleware, registro y modelo
exportado) sin levantar uvicorn.
"""

import pandas as pd
from fastapi.testclient import TestClient

from modelado.pipeline_proyecto import RUTA_DATOS, ULTIMO_DIA_ENTRENAMIENTO
from servicio import api

CAMPOS = ["estacion_codigo", "fecha", "hora", "franja", "capacidad", "salidas", "llegadas",
          "salidas_lag_15m", "salidas_ultima_hora", "salidas_mismo_momento_ayer"]

if __name__ == "__main__":
    datos = pd.read_csv(RUTA_DATOS, dtype={"estacion_codigo": str})
    prueba = datos[pd.to_datetime(datos["fecha"]).dt.day > ULTIMO_DIA_ENTRENAMIENTO]
    prueba = prueba.sort_values(["fecha", "franja", "estacion_codigo"])[CAMPOS]
    # Los rezagos faltantes viajan como null en el JSON, igual que los mandaria el lote.
    franjas = prueba.astype(object).where(prueba.notna(), None).to_dict(orient="records")

    if api.RUTA_REGISTRO.exists():
        api.RUTA_REGISTRO.unlink()
    cliente = TestClient(api.app)
    for franja in franjas:
        cliente.post("/predecir", json=franja)

    print(f"{len(franjas)} franjas enviadas ({prueba['fecha'].min()} a {prueba['fecha'].max()}), "
          f"version del modelo {api.version_servida}")
    print("Metricas del sistema:", cliente.get("/metricas").json())
