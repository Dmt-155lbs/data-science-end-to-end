"""Pruebas de pipeline_proyecto.py. Se corren desde la raiz del fork con:

    pytest psets/proyecto -v

No leen data/processed/: en GitHub Actions ese archivo no existe. Cada prueba
arma sus propias franjas de pocas filas, escritas a mano.
"""

import numpy as np
import pandas as pd
import pytest

from modelado.pipeline_proyecto import (
    CATEGORICAS,
    NUMERICAS,
    OBJETIVO,
    YA_LISTAS,
    agregar_features,
    cargar_y_dividir,
    construir_preparacion,
)


def franjas_de_prueba():
    """Seis franjas de una estacion de 20 anclajes: tres del 24 de agosto
    (ultimo dia de entrenamiento) y tres del 25 (primer dia de prueba). La
    ultima no tiene objetivo, como la ultima franja de cada estacion."""
    return pd.DataFrame({
        "estacion_codigo": ["007"] * 6,
        "fecha": ["2026-08-24"] * 3 + ["2026-08-25"] * 3,
        "hora": [3, 8, 15, 3, 8, 20],
        "franja": [12, 32, 60, 12, 32, 80],
        "dia_semana": [1, 1, 1, 2, 2, 2],
        "es_fin_de_semana": [False] * 6,
        "capacidad": [20] * 6,
        "salidas": [0, 4, 3, 0, 5, 1],
        "llegadas": [0, 2, 3, 1, 4, 2],
        "salidas_lag_15m": [np.nan, 3, 2, 0, 4, 1],
        "salidas_ultima_hora": [np.nan, 6, 9, 0, 12, 4],
        "salidas_mismo_momento_ayer": [np.nan, 4, 2, 0, 5, 2],
        "y_salidas_prox_30m": [0, 7, 5, 1, 8, np.nan],
    })


def test_objetivo_no_entra_al_modelo():
    entradas = NUMERICAS + CATEGORICAS + YA_LISTAS
    assert OBJETIVO not in entradas
    # Ninguna columna del futuro (y_*) puede ser entrada del modelo.
    assert not [columna for columna in entradas if columna.startswith("y_")]


def test_features_calculadas_a_mano():
    datos = agregar_features(franjas_de_prueba())
    assert list(datos["periodo_demanda"]) == ["madrugada", "pico_manana", "pico_tarde", "madrugada", "pico_manana", "resto"]
    # 6 salidas en la ultima hora en una estacion de 20 anclajes: 6 / 20 = 0.3.
    assert datos.loc[1, "salidas_hora_por_capacidad"] == pytest.approx(0.4)
    # Sin historia (rezago faltante), la feature tambien queda faltante: la
    # imputa la preparacion, no se inventa aqui.
    assert np.isnan(datos.loc[0, "salidas_hora_por_capacidad"])


def test_division_por_fecha_sin_futuro_en_entrenamiento(tmp_path):
    ruta = tmp_path / "franjas.csv"
    franjas_de_prueba().to_csv(ruta, index=False)

    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir(ruta)

    # El 24 va a entrenamiento y el 25 a prueba; la franja sin objetivo se quita.
    assert len(X_ent) == 3 and len(X_prueba) == 2
    assert y_ent.notna().all() and y_prueba.notna().all()
    # Entrenamiento no tiene ninguna franja posterior a las de prueba.
    fechas = pd.to_datetime(franjas_de_prueba()["fecha"])
    assert fechas.loc[X_ent.index].max() < fechas.loc[X_prueba.index].min()
    assert OBJETIVO not in X_ent.columns


def test_preparacion_acepta_categoria_nueva_y_faltantes(tmp_path):
    ruta = tmp_path / "franjas.csv"
    franjas_de_prueba().to_csv(ruta, index=False)
    X_ent, X_prueba, _, _ = cargar_y_dividir(ruta)

    preparacion = construir_preparacion().fit(X_ent)
    # Un periodo que no aparecio al entrenar y rezagos faltantes.
    nueva = X_prueba.head(1).copy()
    nueva["periodo_demanda"] = "nocturno"
    nueva[["salidas_lag_15m", "salidas_ultima_hora"]] = np.nan

    lista = preparacion.transform(nueva)
    assert lista.shape == (1, len(preparacion.get_feature_names_out()))
    assert not np.isnan(lista).any()
