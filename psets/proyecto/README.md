# Rebalanceo predictivo de cicloestaciones de Ecobici

> Este es el README de tu proyecto: reemplaza cada sección entre corchetes a medida que avanzas. Escríbelo para alguien que clona tu fork sin haber estado en clase y necesita entender qué hace el proyecto y correrlo, no como una nota para ti.

> **Versión preliminar.** El modelado de esta versión usa solo los viajes de agosto de 2026. Se va a rehacer cuando el PSet 3 tenga los datos completos: el historial de más meses y el estado de las estaciones capturado del feed GBFS cada 2 minutos desde el 21 de octubre de 2026.

Las estaciones de Ecobici (Ciudad de México) se vacían o se llenan en las horas pico, y el equipo de rebalanceo mueve bicicletas en camión sin saber con anticipación cuáles van a quedar en estado crítico. Este proyecto predice, al cierre de cada franja de 15 minutos, cuántas bicicletas van a salir de cada estación en los próximos 30 minutos (`y_salidas_prox_30m`), para que el despachador priorice a qué estaciones mandar el camión antes de que se vacíen.

**Métrica de éxito:** menos estación-horas en estado crítico y menos demanda no atendida con el mismo presupuesto de rebalanceo (misma flota y mismo personal). Umbral de decisión: una estación está en riesgo cuando su nivel proyectado queda en 10% de su capacidad o menos (desabastecimiento), o en 90% o más (saturación). Métrica técnica del modelo: MAE de las salidas por estación y franja, contra la persistencia como línea base.

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| Viajes de Ecobici, datos abiertos | Un viaje por fila (1,473,684 en agosto de 2026): estación y fecha/hora de retiro y de arribo, género, edad y bicicleta | CSV mensual de [ecobici.cdmx.gob.mx/datos-abiertos](https://ecobici.cdmx.gob.mx/datos-abiertos/) | `data/raw/public_data_web_2026-08_2.csv` |
| Feed GBFS de Ecobici, `station_information` | Una fila por estación: código, nombre, ubicación y capacidad (estado actual, no histórico) | `https://gbfs.mex.lyftbikes.com/gbfs/es/station_information.json` | `data/raw/station_information.json` |

Los datos no están en el repositorio. Para regenerarlos desde cero: descarga los dos archivos de arriba a `data/raw/` y, desde `calidad/`, corre `python preparar_datos.py`. El script limpia los viajes (duplicados, fechas inválidas, llegadas antes de la salida), toma 100 estaciones con semilla 42, arma la grilla estación × día × franja de 15 minutos con sus rezagos y el objetivo, y deja `data/processed/flujo_estaciones_2026_08.csv` (297,600 filas, huella MD5 `968012844003`), el archivo que lee todo el modelado.

## Cómo correr el pipeline de datos

[Los comandos, en orden y desde qué carpeta, para ir de la fuente al warehouse: levantar la infraestructura, construir la imagen, subir el flow a Kestra, limpiar, cargar y correr dbt. Cada paso dice qué archivo deja y dónde. Se completa con el PSet 2.]

## Cómo entrenar y evaluar el modelo

Todo está en `modelado/`. Los notebooks importan la preparación y la división de [`pipeline_proyecto.py`](modelado/pipeline_proyecto.py) en vez de copiarlas: el script lee `../data/processed/flujo_estaciones_2026_08.csv`, agrega las dos features (`periodo_demanda` y `salidas_hora_por_capacidad`), quita la última franja de cada estación (no tiene objetivo) y divide **por fecha**: entrenamiento del 1 al 24 de agosto (230,400 filas) y prueba del 25 al 31 (67,100). Para comprobar que corre solo: `cd modelado && python pipeline_proyecto.py`.

Los notebooks, en orden (cada uno con Restart y Run All):

| Orden | Notebook | Qué hace | Deja |
|---|---|---|---|
| 1 | [`explorar_proyecto.ipynb`](modelado/explorar_proyecto.ipynb) | Resumen, objetivo y tres hallazgos que afectan el modelado | |
| 2 | [`features_proyecto.ipynb`](modelado/features_proyecto.ipynb) | Las dos features, con su evidencia | |
| 3 | [`modelos_proyecto.ipynb`](modelado/modelos_proyecto.ipynb) | Persistencia, Ridge y gradient boosting | `resultados_proyecto.csv` |
| 4 | [`comparar_proyecto.ipynb`](modelado/comparar_proyecto.ipynb) | Gradient boosting contra red neuronal, y la prueba del LLM | `comparacion_proyecto.csv`, `prompt_llm_proyecto.txt` |
| 5 | [`experimentos_proyecto.ipynb`](modelado/experimentos_proyecto.ipynb) | Registra los cuatro modelos en MLflow, marca el final y lo reproduce desde el registro | `corridas_proyecto.csv` |
| 6 | [`validacion_proyecto.ipynb`](modelado/validacion_proyecto.ipynb) | Validación cruzada temporal contra al azar, y la métrica del negocio | `validacion_proyecto.csv`, `metricas_negocio_proyecto.csv` |

El notebook 5 necesita el servidor de MLflow levantado: el devcontainer lo arranca en el puerto 5050 y los notebooks lo encuentran por `MLFLOW_TRACKING_URI` (si no responde: `bash .devcontainer/mlflow-server.sh` desde la raíz del fork).

## Modelo final

**Gradient boosting**, `run_id` `7749f7c3a8a74957baf3eb06d655ba7b` en el experimento `ecobici-rebalanceo`, con la etiqueta `candidato` = `final`. MAE en prueba **1.0278** salidas por franja, 16.3% menos que la persistencia (1.2278). En validación cruzada temporal: 1.0000 ± 0.0369. La red neuronal queda 0.007 por debajo, dentro de la variación entre pliegues, así que elegí el modelo más rápido y explicable. El razonamiento completo está en [`docs/decision_modelo.md`](docs/decision_modelo.md).

## Cómo correr el servicio

[Cómo se construye y se levanta el contenedor de `servicio/`, qué versión del modelo carga desde el registro, y un ejemplo de petición con su respuesta. Se completa con el PSet 4.]
