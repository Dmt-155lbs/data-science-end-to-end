"""Pruebas del servicio. Se corren desde la raiz del fork con:

    pytest psets/proyecto -v

No usan servicio/modelo.joblib, que no existe en GitHub Actions: reemplazan el
modelo por uno pequeno que siempre estima 2 salidas, entrenado con la
preparacion real. Prueban el contrato de la API, no la calidad del modelo.
"""

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import CATEGORICAS, NUMERICAS, YA_LISTAS, agregar_calendario, agregar_features, construir_preparacion
from servicio import api

FRANJA = {
    "estacion_codigo": "271-272",
    "fecha": "2026-08-25",
    "hora": 8,
    "capacidad": 20,
    "salidas": 4,
    "llegadas": 2,
    "salidas_lag_15m": 3,
    "salidas_ultima_hora": 9,
    "salidas_mismo_momento_ayer": 5,
}


def modelo_de_prueba():
    datos = agregar_features(agregar_calendario(pd.DataFrame([FRANJA, {**FRANJA, "hora": 15}])))
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS].copy()
    X["es_fin_de_semana"] = X["es_fin_de_semana"].astype(int)
    modelo = Pipeline([("preparar", construir_preparacion()), ("modelo", DummyRegressor(strategy="mean"))])
    return modelo.fit(X, [2.0, 2.0])


api.modelo = modelo_de_prueba()
api.version_servida = "7"
cliente = TestClient(api.app)


def test_salud():
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "version_modelo": "7"}


def test_prediccion_valida():
    respuesta = cliente.post("/predecir", json=FRANJA)
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estacion_codigo": "271-272", "salidas_prox_30m": 2.0, "version_modelo": "7"}


def test_cada_prediccion_dice_que_version_la_calculo():
    # Version inventada "7" y modelo de prueba: sin servidor de MLflow.
    for hora in (3, 8, 18):
        respuesta = cliente.post("/predecir", json={**FRANJA, "hora": hora})
        assert respuesta.json()["version_modelo"] == "7"


def test_franja_sin_historia_tambien_se_predice():
    franja = {**FRANJA, "salidas_lag_15m": None, "salidas_ultima_hora": None, "salidas_mismo_momento_ayer": None}
    respuesta = cliente.post("/predecir", json=franja)
    assert respuesta.status_code == 200


def test_hora_invalida_se_rechaza():
    respuesta = cliente.post("/predecir", json={**FRANJA, "hora": 25})
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == ["body", "hora"]


def test_calendario_igual_que_en_entrenamiento():
    # En el archivo de entrenamiento, el 2026-08-01 (sabado) tiene
    # dia_semana 6 y es fin de semana; el 2026-08-03 (lunes), 1 y no.
    datos = agregar_calendario(pd.DataFrame({"fecha": ["2026-08-01", "2026-08-03"]}))
    assert list(datos["dia_semana"]) == [6, 1]
    assert list(datos["es_fin_de_semana"]) == [True, False]
