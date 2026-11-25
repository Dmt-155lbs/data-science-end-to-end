"""Pruebas del servicio. Se corren desde la raiz del fork con:

    pytest psets/proyecto -v

No usan servicio/modelo.joblib, que no existe en GitHub Actions: reemplazan el
modelo por uno pequeno que siempre estima 2 salidas, entrenado con la
preparacion real, y una version inventada ("7"). Cada prueba escribe su
registro de predicciones en una carpeta temporal y empieza con metricas
nuevas. Prueban el contrato de la API, no la calidad del modelo.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import CATEGORICAS, NUMERICAS, YA_LISTAS, agregar_calendario, agregar_features, construir_preparacion
from servicio import api
from servicio.deriva import MonitorDeDeriva

FRANJA = {
    "estacion_codigo": "271-272",
    "fecha": "2026-08-25",
    "hora": 8,
    "franja": 33,
    "capacidad": 20,
    "salidas": 4,
    "llegadas": 2,
    "salidas_lag_15m": 3,
    "salidas_ultima_hora": 9,
    "salidas_mismo_momento_ayer": 5,
}


# Una referencia de deriva inventada: la mitad de las franjas con 0 salidas en
# la ultima hora y la otra mitad con mas; lo mismo con las llegadas.
REFERENCIA = {
    "salidas_ultima_hora": {"cortes": [0.0], "proporciones": [0.5, 0.5]},
    "llegadas": {"cortes": [0.0], "proporciones": [0.5, 0.5]},
}


def modelo_de_prueba():
    datos = agregar_features(agregar_calendario(pd.DataFrame([FRANJA, {**FRANJA, "hora": 15}])))
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS].copy()
    X["es_fin_de_semana"] = X["es_fin_de_semana"].astype(int)
    modelo = Pipeline([("preparar", construir_preparacion()), ("modelo", DummyRegressor(strategy="mean"))])
    return modelo.fit(X, [2.0, 2.0])


@pytest.fixture
def cliente(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "modelo", modelo_de_prueba())
    monkeypatch.setattr(api, "version_servida", "7")
    monkeypatch.setattr(api, "RUTA_REGISTRO", tmp_path / "predicciones.jsonl")
    monkeypatch.setattr(api, "metricas", api.Metricas())
    monkeypatch.setattr(api, "monitor", MonitorDeDeriva(REFERENCIA, ventana=4, umbral=0.10))
    return TestClient(api.app)


def leer_registro():
    with open(api.RUTA_REGISTRO, encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo]


def test_salud(cliente):
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "version_modelo": "7"}


def test_prediccion_valida(cliente):
    respuesta = cliente.post("/predecir", json=FRANJA)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["salidas_prox_30m"] == 2.0
    assert cuerpo["estacion_codigo"] == "271-272"
    assert len(cuerpo["id_prediccion"]) == 32


def test_cada_prediccion_dice_que_version_la_calculo(cliente):
    for hora in (3, 8, 18):
        respuesta = cliente.post("/predecir", json={**FRANJA, "hora": hora})
        assert respuesta.json()["version_modelo"] == "7"


def test_cada_prediccion_queda_registrada(cliente):
    respuesta = cliente.post("/predecir", json=FRANJA).json()
    lineas = leer_registro()
    assert len(lineas) == 1
    # Lo que hace falta para unirla con la verdad y saber que modelo la produjo.
    assert lineas[0]["id_prediccion"] == respuesta["id_prediccion"]
    assert lineas[0]["estacion_codigo"] == "271-272"
    assert lineas[0]["fecha"] == "2026-08-25" and lineas[0]["franja"] == 33
    assert lineas[0]["version_modelo"] == "7"
    assert lineas[0]["salidas_prox_30m"] == 2.0


def test_metricas_separan_errores_de_validacion_y_de_servidor(cliente, monkeypatch):
    cliente.post("/predecir", json=FRANJA)                       # 200
    cliente.post("/predecir", json={**FRANJA, "hora": 25})        # 422: no llega al modelo
    monkeypatch.setattr(api, "modelo", None)                     # el modelo falla: 500
    TestClient(api.app, raise_server_exceptions=False).post("/predecir", json=FRANJA)

    resumen = cliente.get("/metricas").json()
    assert resumen["peticiones"] == 3
    assert resumen["errores_validacion"] == 1
    assert resumen["errores_servidor"] == 1
    assert resumen["latencia_p50_ms"] > 0 and resumen["latencia_p95_ms"] >= resumen["latencia_p50_ms"]
    # Las peticiones con error no dejan prediccion en el registro.
    assert len(leer_registro()) == 1


def test_franja_sin_historia_tambien_se_predice(cliente):
    franja = {**FRANJA, "salidas_lag_15m": None, "salidas_ultima_hora": None, "salidas_mismo_momento_ayer": None}
    assert cliente.post("/predecir", json=franja).status_code == 200


def test_hora_invalida_se_rechaza(cliente):
    respuesta = cliente.post("/predecir", json={**FRANJA, "hora": 25})
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == ["body", "hora"]


def test_calendario_igual_que_en_entrenamiento():
    # En el archivo de entrenamiento, el 2026-08-01 (sabado) tiene
    # dia_semana 6 y es fin de semana; el 2026-08-03 (lunes), 1 y no.
    datos = agregar_calendario(pd.DataFrame({"fecha": ["2026-08-01", "2026-08-03"]}))
    assert list(datos["dia_semana"]) == [6, 1]
    assert list(datos["es_fin_de_semana"]) == [True, False]


def test_deriva_sin_alerta_con_ventana_parecida_a_la_referencia(cliente):
    for salidas_ultima_hora, llegadas in [(0, 0), (9, 2), (0, 0), (5, 3)]:
        cliente.post("/predecir", json={**FRANJA, "salidas_ultima_hora": salidas_ultima_hora, "llegadas": llegadas})
    estado = cliente.get("/deriva").json()
    assert estado["franjas_en_ventana"] == 4
    assert estado["psi"] == {"salidas_ultima_hora": 0.0, "llegadas": 0.0}
    assert estado["alerta"] is False


def test_deriva_alerta_con_ventana_alejada_de_la_referencia(cliente):
    # Un dia de lluvia: casi nadie sale y todas las franjas llegan con 0
    # salidas en la ultima hora. Con la ventana a medias no alerta; llena, si.
    for _ in range(3):
        cliente.post("/predecir", json={**FRANJA, "salidas_ultima_hora": 0})
    assert cliente.get("/deriva").json()["alerta"] is False
    cliente.post("/predecir", json={**FRANJA, "salidas_ultima_hora": 0})
    estado = cliente.get("/deriva").json()
    assert estado["franjas_en_ventana"] == 4
    assert estado["psi"]["salidas_ultima_hora"] > 0.10
    assert estado["alerta"] is True
