# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

El código, los datos y el modelo. El autor versiona el código con **Git** (una rama por funcionalidad, merge a `master` y tags como `v1.0`), los datos con **DVC** (Data Version Control, que trabaja encima de Git) y el modelo con **MLflow** (un servidor de seguimiento que guarda cada corrida y el modelo con su entorno).

**¿Por qué no alcanza con Git para versionar los datos?**

Porque Git no está hecho para datasets grandes: cada versión de los datos se guardaría completa dentro del repositorio. DVC guarda los datos fuera de Git y deja en el repositorio solo un archivo `.dvc` pequeño con los metadatos (el hash) de cada versión; la carpeta de datos queda en `.gitignore`. Así Git versiona *qué* versión de los datos se usó, y para volver a una anterior basta `git checkout` del `.dvc` y `dvc checkout`.

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Qué versión exacta de los datos usó cada uno, cómo dividió entrenamiento y prueba (la semilla), los hiperparámetros y la semilla del modelo, las versiones de Python y de scikit-learn, y el commit del código que corrieron. Con eso se ve cuál de esas cosas es distinta.

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

Ninguno de los dos está equivocado: es el mismo gradient boosting sobre el mismo CSV, y cada error es correcto para su propia división. Uno da 1.2754 y el otro 1.1867 porque los 595 viajes de prueba no son los mismos. La salida deja ver la semilla de la división (la línea imprime `division 42` y `division 7`) y el error, pero no los hiperparámetros ni con qué versión de los datos se entrenó, y no queda guardada: al reiniciar el kernel se pierde.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La división. Mismo modelo y mismos datos, así que espero que en la tabla del paso 5 la `huella_datos` y `n_estimators` sean iguales en las dos corridas, y que lo único distinto sea `semilla_division` (42 contra 7).

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Para poder usar exactamente ese modelo sin volver a entrenarlo: comprobar que reproduce la métrica registrada, compararlo con otro sobre datos nuevos o ponerlo en producción. Los parámetros dicen cómo se entrenó, pero volver a entrenar no garantiza el mismo modelo (otra versión de la librería, otros datos); el artefacto sí es el modelo que produjo ese número.

**Lo que mostró la tabla del paso 5:**

| Corrida | Semilla de la división | Huella de los datos | Error en prueba |
|---|---|---|---|
| gradient boosting | 7 | `cc636c7813cf` | 1.1867 |
| gradient boosting | 42 | `cc636c7813cf` | 1.2754 |
| bosque aleatorio | 42 | `cc636c7813cf` | 1.3016 |

La huella es la misma en las tres: los datos no cambiaron. Entre los dos gradient boosting solo cambia la semilla de la división, y eso solo mueve el error 0.09.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que compara las corridas con evidencia del registro y no por el número suelto: dos errores solo se pueden comparar si salen de los mismos datos (la misma huella), la misma división y la misma métrica, y cada corrida queda definida por su código, sus datos, su configuración y su entorno. Las métricas y el modelo son lo que la corrida produce, no lo que la define. El caso de los dos gradient boosting lo muestra: 1.2754 contra 1.1867 sin que nadie se equivocara, solo por la semilla de la división.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión confirmó que un error sin su contexto (datos, división, hiperparámetros) no es evidencia, y que la diferencia entre dos modelos tiene que ser mayor que la que produce cambiar la división antes de declarar un ganador.

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

La red neuronal, con 1.1695 en prueba. Le gana al gradient boosting (1.2754) por 0.1059, alrededor de 8% menos error. Después vienen el bosque (1.3016), Ridge (2.5565) y la línea base (12.5308).

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

Que el artefacto guardado es exactamente el modelo que produjo ese número: cargado desde el registro y evaluado sobre los mismos 595 viajes, da 1.169505229635501, igual que lo registrado. No garantiza que ese modelo sea el mejor en general: es un solo error sobre una sola división, y con otra división el orden podría cambiar (en el Ejercicio 1 la semilla movió el error del gradient boosting en 0.09). Tampoco dice cómo se va a comportar con viajes de otro mes.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

Por la huella de los datos: cualquier cambio en un byte del CSV cambia el MD5, así que las corridas nuevas tendrían una `params.huella_datos` distinta de `cc636c7813cf`. Filtrando por huella se ve qué corridas usaron qué versión de los datos, y que no se pueden comparar directamente.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

Todo lo que hace falta para repetirla: sus hiperparámetros completos (activación `relu`, solver `adam`, `learning_rate_init`, `alpha`, `batch_size`, `max_iter` 1000, `random_state` 42), la semilla de la división, la huella de los datos, las filas de entrenamiento, el tiempo (12.6 s contra 0.2 s del gradient boosting) y el modelo entrenado como artefacto. Una tabla escrita a mano solo tendría el nombre y el error.

## Ejercicio 3: el registro de tu proyecto

Usa tu `pipeline_proyecto.py` y los modelos que entrenaste en las dos sesiones de la Semana 10. Esta vez no hay plantilla.

### Entregable 1: `experimentos_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división y registre en MLflow, en un experimento con el nombre de tu proyecto, una corrida por cada modelo de la Semana 10 (línea base, modelo lineal, ensamblaje, red neuronal). Cada corrida registra:

- Los hiperparámetros del modelo.
- Cómo se dividieron los datos (la semilla, o la fecha de corte si divides por fecha) y la huella del archivo de datos.
- La métrica de tu proyecto en entrenamiento y en prueba, y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, el notebook marca la corrida que eliges como final (`candidato` = `final`), carga su modelo desde el registro y comprueba que reproduce la métrica registrada.

### Entregable 2: `corridas_proyecto.csv`

La tabla de corridas de tu experimento, exportada con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)` desde el mismo notebook. Antes de hacer commit, borra las corridas de prueba del experimento de tu proyecto (en la interfaz de MLflow: selecciónalas y **Delete**), reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

El gradient boosting, `run_id` **`b3cdcd9d46894965b2f2b9fbc5fde396`**, marcado con `candidato` = `final` en el experimento `ecobici-rebalanceo` ([`experimentos_proyecto.ipynb`](./proyecto/experimentos_proyecto.ipynb), tabla en [`corridas_proyecto.csv`](./proyecto/corridas_proyecto.csv)). Las cuatro corridas usan la misma huella de datos (`968012844003`) y la misma división (entrenamiento hasta el 2026-08-24):

| Corrida | MAE entrenamiento | MAE prueba | Segundos |
|---|---|---|---|
| red neuronal | 0.9649 | 1.0209 | 88.7 |
| **gradient boosting** | 0.9739 | **1.0278** | 18.7 |
| ridge | 1.0117 | 1.0632 | 0.3 |
| persistencia 30 min (línea base) | 1.1527 | 1.2278 | 0 |

La red neuronal le gana por 0.0069 salidas, menos de 1%. A cambio tarda casi 5 veces más en entrenar y no deja leer qué columnas mandan, así que con un rendimiento prácticamente igual elijo el modelo más simple y explicable. El gradient boosting mejora 16.3% a la línea base. Cargado desde `runs:/b3cdcd9d46894965b2f2b9fbc5fde396/modelo`, reproduce exactamente su MAE registrado (1.027782).

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

Necesita cuatro cosas: el **código** (el commit de este repositorio, con `pipeline_proyecto.py` y el notebook), los **datos** (`flujo_estaciones_2026_08.csv` con la huella `968012844003`), la **configuración** (hiperparámetros, `random_state=42` y la fecha de corte) y el **entorno** (Python 3.11 con scikit-learn 1.5.2, pandas 2.2.2 y numpy 1.26.4, que fija el `requirements.txt` y el devcontainer).

- **Quedó en MLflow:** los hiperparámetros con la semilla, la fecha de corte y el tipo de división, la huella de los datos, las filas de entrenamiento, las métricas, el modelo entrenado (con el `requirements.txt` del entorno en que se guardó) y la etiqueta de la decisión.
- **No quedó:** el archivo de datos en sí (la huella dice qué versión es, pero no la guarda, y el CSV está en `.gitignore`: hay que regenerarlo desde los viajes de agosto con el `preparar_datos.py` de la Semana 8 o tenerlo aparte, como haría DVC), ni el commit del código, que hay que anotar a mano porque el notebook no lo registra.

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

No lo sé con un solo número: el registro tiene un error por corrida, calculado sobre una sola división, y ese número no trae su margen. Sí se ve una pista en el Ejercicio 1: el mismo gradient boosting dio 1.2754 con la semilla 42 y 1.1867 con la 7, una diferencia de 0.09 solo por cambiar qué viajes quedan en prueba. Eso es comparable a la distancia entre algunos modelos, así que haría falta validación cruzada para saber si la diferencia entre el primero y el segundo es real.
