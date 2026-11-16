# Decisión del modelo final

> **Versión preliminar.** El modelado de esta versión usa solo los viajes de agosto de 2026. Se va a rehacer cuando el PSet 3 tenga los datos completos: el historial de más meses y el estado de las estaciones capturado del feed GBFS cada 2 minutos desde el 21 de octubre de 2026.

**Para quién:** alguien del equipo que tiene que decidir si este modelo pasa a producción.
**Qué predice:** cuántas bicicletas van a salir de cada estación de Ecobici en los próximos 30 minutos (`y_salidas_prox_30m`), con datos de cada franja de 15 minutos, para anticipar qué estaciones se van a vaciar y mandar el camión de rebalanceo a tiempo.
**Datos:** 100 estaciones, agosto de 2026, 297,600 franjas (`data/processed/flujo_estaciones_2026_08.csv`, huella MD5 `968012844003`). Entrenamiento del 1 al 24 de agosto (230,400 filas) y prueba del 25 al 31 (67,100 filas).

## 1. El modelo elegido

**Gradient boosting** (`GradientBoostingRegressor(random_state=42)`, con la preparación de [`pipeline_proyecto.py`](../modelado/pipeline_proyecto.py)), `run_id` **`7749f7c3a8a74957baf3eb06d655ba7b`** en el experimento `ecobici-rebalanceo` de MLflow, con la etiqueta `candidato` = `final`. Cargado desde `runs:/7749f7c3a8a74957baf3eb06d655ba7b/modelo`, reproduce exactamente su error registrado: 1.027782.

## 2. Contra qué lo comparé

Los cuatro candidatos usan la misma preparación, la misma división por fecha, la misma huella de datos y la misma métrica: el error absoluto medio (MAE) en salidas de bicicleta. Fuente: [`corridas_proyecto.csv`](../modelado/corridas_proyecto.csv).

| Modelo | MAE entrenamiento | MAE prueba | Brecha | Segundos de entrenamiento |
|---|---|---|---|---|
| Persistencia (línea base: los próximos 30 min iguales a los últimos 30) | 1.1527 | 1.2278 | 0.075 | 0 |
| Ridge (`alpha=1.0`) | 1.0117 | 1.0632 | 0.052 | 0.3 |
| **Gradient boosting** | 0.9739 | **1.0278** | 0.054 | 20.0 |
| Red neuronal (64 neuronas, `early_stopping`) | 0.9649 | 1.0209 | 0.056 | 71.0 |

El gradient boosting se equivoca 16.3% menos que la línea base. Ningún modelo se sobreajusta: la brecha entre entrenamiento y prueba es de 0.05 a 0.06 en todos. También probé un LLM con 10 filas ([`comparar_proyecto.ipynb`](../modelado/comparar_proyecto.ipynb)): acertó mejor en esas 10 (MAE 0.40), pero es una muestra mínima, no da siempre la misma respuesta y costaría unos 28 dólares al mes frente a casi 0.

## 3. ¿La diferencia con el segundo es real?

No. La red neuronal le gana al gradient boosting por 0.007 salidas, menos de 1%. Con validación cruzada **temporal** de 5 pliegues (cada pliegue entrena con los días anteriores y valida con los siguientes, como en producción), en [`validacion_proyecto.ipynb`](../modelado/validacion_proyecto.ipynb):

| Esquema | Gradient boosting | Red neuronal |
|---|---|---|
| Temporal (el que corresponde) | 1.0000 ± 0.0369 | 0.9930 ± 0.0322 |
| Al azar (solo para comparar) | 0.9904 ± 0.0022 | 0.9790 ± 0.0053 |

La diferencia entre los dos (0.007) es cinco veces menor que la variación entre pliegues (0.03 a 0.04): está dentro del ruido. El esquema al azar da errores más bajos y casi sin dispersión porque entrena con días posteriores a los que valida, y por eso no lo uso para decidir.

## 4. El error caro y la métrica del negocio

**Subestimar es más caro que sobreestimar.** Si el modelo dice que saldrán pocas bicis y salen muchas, la estación se vacía antes de que llegue el camión, y cada bici que faltó es un viaje perdido. Sobreestimar cuesta un viaje del camión que no hacía falta, repartido entre las varias bicis que mueve. **Supuesto aproximado: una bici subestimada cuesta 3 veces lo que una sobreestimada.** Medido sobre los pliegues temporales ([`metricas_negocio_proyecto.csv`](../modelado/metricas_negocio_proyecto.csv)):

| Modelo | Costo asimétrico por franja | Franjas subestimadas | Demanda alta (3+) corta por 2 o más |
|---|---|---|---|
| Persistencia | 2.35 | 27.4% | 51.9% |
| **Gradient boosting** | **2.00** | 39.0% | 39.1% |
| Red neuronal | 1.99 | 38.7% | 39.2% |

El modelo elegido baja el costo 15% respecto de la línea base y falla menos justo donde importa: se queda corto en 39% de las franjas de demanda alta, contra 52% de la persistencia. **Riesgo a vigilar:** se queda corto más seguido que la persistencia (39% de las franjas contra 27%), porque al minimizar el MAE apunta al centro. Para rebalanceo convendría corregirlo, por ejemplo entrenando con pérdida por cuantiles.

## 5. Costo y explicabilidad

El gradient boosting entrena en unos 20 segundos, contra 71 de la red neuronal, y predecir las 100 estaciones cada 15 minutos cuesta prácticamente nada. Se puede explicar: según sus importancias, más de la mitad de lo que decide sale de las salidas de la última hora (0.54), seguidas por las salidas y las llegadas de la franja actual (0.15 cada una) y la hora (0.07). Es decir, predice sobre todo con la historia reciente de la estación, lo mismo que encontró la exploración, y cada predicción sale de árboles que se pueden recorrer. La red neuronal no tiene una lectura directa. Con rendimiento equivalente, elijo el modelo más simple, barato y explicable.

## 6. Cómo obtener exactamente el mismo número

Hace falta:

- **El código:** este repositorio en el commit de esta entrega, con `pipeline_proyecto.py` y `experimentos_proyecto.ipynb`.
- **Los datos:** `data/processed/flujo_estaciones_2026_08.csv` con huella `968012844003`. No está en git: se regenera con `calidad/preparar_datos.py` a partir de los viajes de agosto de Ecobici y del `station_information.json` del feed GBFS (ver la sección "Datos" del [README](../README.md)).
- **La configuración:** `random_state=42` y la fecha de corte 2026-08-24.
- **El entorno:** el devcontainer del curso (Python 3.11, scikit-learn 1.5.2, pandas 2.2.2, numpy 1.26.4, MLflow 2.17.2).

**Quedó registrado en MLflow:** los hiperparámetros con la semilla, el tipo de división y la fecha de corte, la huella de los datos, las filas de entrenamiento, las métricas, el modelo entrenado (con el `requirements.txt` de su entorno) y la etiqueta de la decisión. **No quedó:** el archivo de datos en sí (la huella dice qué versión es, pero no la guarda) ni el commit del código, que se anota en la entrega con el tag `pset-03`.
