"""Compara el mejor ensamblaje del proyecto de Ecobici con una red
neuronal simple, y prepara la prueba del LLM sobre 10 filas de prueba.

    python3 comparar_proyecto.py

Escribe prompt_llm_proyecto.txt. Guarda los 10 numeros que responda el
LLM, uno por linea, en respuestas_llm_proyecto.txt y vuelve a correrlo.
"""

import os
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
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
LEGIBLES = ["capacidad", "hora", "dia_semana", "es_fin_de_semana", "salidas",
            "llegadas", "salidas_lag_15m", "salidas_ultima_hora", "salidas_mismo_momento_ayer"]

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS].copy()
X["es_fin_de_semana"] = X["es_fin_de_semana"].astype(int)
y = datos[OBJETIVO]

dia = pd.to_datetime(datos["fecha"]).dt.day
es_ent = dia <= 24
X_ent, X_prueba = X[es_ent], X[~es_ent]
y_ent, y_prueba = y[es_ent], y[~es_ent]

muestra = X_prueba.sample(10, random_state=7)
y_muestra = y_prueba.loc[muestra.index]
ejemplos = X_ent.sample(30, random_state=7)
y_ejemplos = y_ent.loc[ejemplos.index]


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


def evaluar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
    mae_muestra = mean_absolute_error(y_muestra, pipeline.predict(muestra))
    print(f"{nombre:20s} prueba {mae_prueba:7.4f}   10 filas {mae_muestra:7.4f}   {segundos:7.2f} s")
    return pipeline, {
        "modelo": nombre,
        "mae_prueba": round(mae_prueba, 4),
        "mae_10_filas": round(mae_muestra, 4),
        "segundos": round(segundos, 3),
    }


ensamblaje, fila_ensamblaje = evaluar("gradient boosting", GradientBoostingRegressor(random_state=42))
red, fila_red = evaluar("red neuronal", MLPRegressor(hidden_layer_sizes=(64,), max_iter=200, early_stopping=True, random_state=42))

modelo_red = red.named_steps["modelo"]
fila_red["parametros"] = sum(c.size for c in modelo_red.coefs_) + sum(i.size for i in modelo_red.intercepts_)
fila_red["iteraciones"] = modelo_red.n_iter_

tabla = pd.DataFrame([fila_ensamblaje, fila_red])
tabla.to_csv("comparacion_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))

filas_ejemplo = ejemplos[LEGIBLES].copy()
filas_ejemplo["salidas_prox_30m"] = y_ejemplos.values
texto = (
    "Estimas cuantas bicicletas saldran de una estacion de Ecobici en los proximos "
    "30 minutos. Cada fila es una estacion en una franja de 15 minutos: capacidad en "
    "anclajes, hora, dia_semana (1 es lunes, 7 es domingo), salidas y llegadas de la franja actual, "
    "salidas de la franja anterior, de la ultima hora y del mismo momento de ayer. "
    "Abajo hay 30 filas con su valor real y 10 sin el. Responde solo con 10 numeros, "
    "uno por linea, en el mismo orden, sin texto adicional. Si un valor falta, esta vacio.\n\n"
    "Filas con valor real:\n"
    + filas_ejemplo.to_csv(index=False)
    + "\nFilas sin valor:\n"
    + muestra[LEGIBLES].to_csv(index=False)
)
with open("prompt_llm_proyecto.txt", "w") as archivo:
    archivo.write(texto)
print()
print("prompt_llm_proyecto.txt escrito:", len(texto), "caracteres")

if os.path.exists("respuestas_llm_proyecto.txt"):
    with open("respuestas_llm_proyecto.txt") as archivo:
        predicciones_llm = [float(linea) for linea in archivo if linea.strip()]
    print("MAE del LLM en las 10 filas:", round(mean_absolute_error(y_muestra, predicciones_llm), 4))
else:
    print("Falta respuestas_llm_proyecto.txt")
