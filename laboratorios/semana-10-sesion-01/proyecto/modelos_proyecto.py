"""Primeros modelos del proyecto de Ecobici: una linea base, un modelo
lineal y un ensamblaje, con el pipeline y la division por fecha de la
Semana 9, Sesion 2.

    python3 modelos_proyecto.py
"""

import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

datos = pd.read_csv(
    "../../semana-09-sesion-01/data/flujo_estaciones_features_2026_08.csv",
    dtype={"estacion_codigo": str},
)

OBJETIVO = "y_salidas_prox_30m"
datos = datos.dropna(subset=[OBJETIVO])

NUMERICAS = [
    "salidas", "llegadas", "salidas_lag_15m", "salidas_ultima_hora",
    "salidas_mismo_momento_ayer", "salidas_hora_por_capacidad",
    "capacidad", "hora", "dia_semana",
]
CATEGORICAS = ["periodo_demanda"]
YA_LISTAS = ["es_fin_de_semana"]

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS].copy()
X["es_fin_de_semana"] = X["es_fin_de_semana"].astype(int)
y = datos[OBJETIVO]

dia = pd.to_datetime(datos["fecha"]).dt.day
es_ent = dia <= 24
X_ent, X_prueba = X[es_ent], X[~es_ent]
y_ent, y_prueba = y[es_ent], y[~es_ent]


def preparacion():
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


resultados = []


def registrar(nombre, mae_ent, mae_prueba, segundos):
    resultados.append({
        "modelo": nombre,
        "mae_entrenamiento": round(mae_ent, 4),
        "mae_prueba": round(mae_prueba, 4),
        "brecha": round(mae_prueba - mae_ent, 4),
        "segundos": round(segundos, 3),
    })
    print(f"{nombre:24s} entrenamiento {mae_ent:7.4f}   prueba {mae_prueba:7.4f}   {segundos:6.2f} s")


def evaluar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    mae_ent = mean_absolute_error(y_ent, pipeline.predict(X_ent))
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
    registrar(nombre, mae_ent, mae_prueba, segundos)
    return pipeline


# Linea base: regla del dominio, los proximos 30 minutos iguales a los
# ultimos 30 (la franja actual mas la anterior). No aprende nada.
def persistencia(tabla):
    return tabla["salidas"] + tabla["salidas_lag_15m"].fillna(0)


registrar(
    "persistencia 30 min",
    mean_absolute_error(y_ent, persistencia(X_ent)),
    mean_absolute_error(y_prueba, persistencia(X_prueba)),
    0.0,
)
evaluar("ridge", Ridge(alpha=1.0))
evaluar("gradient boosting", GradientBoostingRegressor(random_state=42))

tabla = pd.DataFrame(resultados)
tabla.to_csv("resultados_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))
