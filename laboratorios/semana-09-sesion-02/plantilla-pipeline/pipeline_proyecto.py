"""Preparacion de los datos de mi proyecto, lista para produccion.

Las features que decidiste en tu notebook de la sesion pasada viven aqui,
dentro de funciones. Desde la Semana 10, tus notebooks de experimentos
importan este archivo en vez de copiar el codigo:

    from pipeline_proyecto import cargar_y_dividir, construir_preparacion

Para comprobar que corre de principio a fin, sin notebook:

    python3 pipeline_proyecto.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# El mismo archivo del notebook de features: el flujo de estaciones de
# Ecobici de la exploracion de la Semana 8. La ruta sale de la ubicacion de
# este archivo, asi funciona igual desde los notebooks de otras carpetas.
RUTA_DATOS = Path(__file__).resolve().parent / "../../semana-08-sesion-02/data/flujo_estaciones_2026_08.csv"

OBJETIVO = "y_salidas_prox_30m"

# Solo columnas que existen al momento de predecir (nada de y_*, ni
# identificadores como nombre o fecha). Los NaN de los rezagos al inicio de
# cada serie los imputa el pipeline.
NUMERICAS = [
    "salidas", "llegadas", "salidas_lag_15m", "salidas_ultima_hora",
    "salidas_mismo_momento_ayer", "salidas_hora_por_capacidad",
    "capacidad", "hora", "dia_semana",
]
# estacion_codigo no entra: son 100 estaciones y un one-hot agregaria 100
# columnas; la capacidad y los rezagos ya describen a cada estacion.
# (Si entrara, se dejaria handle_unknown="ignore" por estaciones nuevas.)
CATEGORICAS = ["periodo_demanda"]
# es_fin_de_semana es booleana: se pasa como 0/1, ya esta lista.
YA_LISTAS = ["es_fin_de_semana"]

# Entrenamiento dias 1-24 de agosto, prueba dias 25-31.
ULTIMO_DIA_ENTRENAMIENTO = 24


def agregar_features(datos):
    # Feature 1: periodo de demanda segun la hora (hallazgo de la Semana 8).
    hora = datos["hora"]
    datos["periodo_demanda"] = np.select(
        [hora < 5, (hora >= 7) & (hora <= 9), (hora >= 14) & (hora <= 18)],
        ["madrugada", "pico_manana", "pico_tarde"],
        default="resto",
    )
    # Feature 2: salidas de la ultima hora relativas a la capacidad de la
    # estacion. La capacidad minima es 11, asi que no divide entre 0.
    datos["salidas_hora_por_capacidad"] = (
        datos["salidas_ultima_hora"] / datos["capacidad"].clip(lower=1)
    )
    return datos


def cargar_y_dividir(ruta=RUTA_DATOS):
    # estacion_codigo se lee como texto por los codigos dobles como "271-272".
    datos = agregar_features(pd.read_csv(ruta, dtype={"estacion_codigo": str}))
    # Sin objetivo no se puede entrenar: la ultima franja de cada estacion no
    # tiene "proximos 30 min". Se quitan esas filas antes de dividir.
    datos = datos.dropna(subset=[OBJETIVO])
    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS].copy()
    X["es_fin_de_semana"] = X["es_fin_de_semana"].astype(int)
    y = datos[OBJETIVO]
    # Las filas tienen fecha y el modelo predice el futuro, asi que se divide
    # por fecha. Un corte al azar mezclaria franjas vecinas (los rezagos y el
    # objetivo se traslapan entre franjas contiguas) y dejaria pasar
    # informacion del futuro al entrenamiento.
    es_ent = pd.to_datetime(datos["fecha"]).dt.day <= ULTIMO_DIA_ENTRENAMIENTO
    return X[es_ent], X[~es_ent], y[es_ent], y[~es_ent]


def construir_preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
    ])


# Esto corre con "python3 pipeline_proyecto.py", pero no cuando un
# notebook hace "from pipeline_proyecto import ...".
if __name__ == "__main__":
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    print("Filas entrenamiento:", len(X_ent), "Filas prueba:", len(X_prueba))
    preparacion = construir_preparacion()
    X_ent_listo = preparacion.fit_transform(X_ent)
    X_prueba_listo = preparacion.transform(X_prueba)
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print(list(preparacion.get_feature_names_out()))

    # Verificacion: las medianas aprendidas son las del entrenamiento.
    medianas = preparacion.named_transformers_["numericas"].named_steps["imputar"].statistics_
    print("Medianas iguales a las de entrenamiento:",
          bool((medianas == X_ent[NUMERICAS].median().values).all()))
    print("NaN restantes:", int(pd.isna(X_ent_listo).sum() + pd.isna(X_prueba_listo).sum()))
