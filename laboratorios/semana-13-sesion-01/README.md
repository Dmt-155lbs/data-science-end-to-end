# Semana 13, Sesión 1: Despliegue en producción: versionado y registro de modelos

> Objetivo de la sesión: versionar y registrar el modelo desplegado de tu proyecto, explicando qué problema resuelve un registro de modelos frente a sobrescribir el mismo archivo cada vez que se reentrena.

**Lectura previa de esta sesión:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 10: Infrastructure and Tooling for MLOps.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El registro de modelos es parte del mismo servidor de MLflow que usas desde la Semana 11: se levanta solo al iniciar el Codespace, en el puerto 5050 (si no responde, `bash .devcontainer/mlflow-server.sh`). Los scripts lo encuentran por la variable `MLFLOW_TRACKING_URI`. Los laboratorios de hoy son scripts.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la gobernanza de modelos, y qué componentes la forman?**

Según IBM, la gobernanza de modelos es el proceso de principio a fin con que una organización establece, aplica y mantiene controles sobre el uso de sus modelos, para que funcionen como se pretende, cumplan la normativa y den resultados confiables a lo largo del tiempo. Nació en el sector financiero para manejar el riesgo de modelos complejos. Sus componentes: **desarrollo y documentación** (objetivos, datos de entrenamiento, metodología, supuestos y salidas esperadas, para auditar después), un **inventario de modelos** centralizado, **validación y back-testing** independientes (contra datos históricos, sesgo, robustez y sobreajuste), **despliegue y monitoreo** continuo (degradación, drift de las entradas, calidad de los datos, con controles dentro del CI/CD) y **roles y flujos de trabajo** definidos, desde el desarrollo hasta la retirada del modelo.

**El artículo pide un inventario central de modelos. ¿Qué datos tendría que tener ese inventario sobre cada modelo, y para qué los necesita alguien que no lo entrenó?**

Por cada modelo: qué es y para qué se usa (su propósito), quién es su responsable, la metodología con que se construyó y en qué estado del ciclo de vida está (en desarrollo, en producción, retirado). Llevado a este curso, también su versión, con qué datos y qué división se entrenó y qué métrica obtuvo. Alguien que no lo entrenó lo necesita para saber si puede confiar en él y a quién preguntar, para auditar una decisión que tomó el modelo, para validarlo con la documentación en vez de adivinar los supuestos y para saber qué modelos dependen de qué datos cuando algo cambia. IBM insiste en que el inventario incluya todo, hasta los modelos en hojas de cálculo.

## Pregunta de apertura

La sesión pasada, tu servicio cargaba `modelo.joblib`. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Con un archivo que se reemplaza no lo sabría: el `modelo.joblib` no dice de qué corrida salió, y el de hoy desaparece cuando exporto el de la semana que viene. Haría falta un registro de modelos: cada versión guardada con su número, su corrida, sus datos y su métrica, sin que ninguna reemplace a otra, y que cada predicción diga qué versión la produjo. Con eso, para el reclamo de mañana busco la versión que dice la predicción de hoy; y para volver al modelo anterior apunto el servicio a la versión anterior del registro, que sigue guardada, sin reentrenar nada.

## Ejercicio 1: la tarifa, con registro

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa de la sesión pasada, ahora con el modelo en el registro de MLflow en vez de un archivo. [`registrar_modelo.py`](./starter/ejercicio1/registrar_modelo.py) entrena la red neuronal con el número de neuronas que le pases y la registra como una versión nueva del modelo `tarifa-taxis`. [`promover.py`](./starter/ejercicio1/promover.py) pone el alias `produccion` en la versión que le pases. [`versiones.py`](./starter/ejercicio1/versiones.py) lista las versiones con su error y sus alias (no lo modifiques). [`servicio.py`](./starter/ejercicio1/servicio.py) carga la versión que tiene el alias `produccion`. Los 4 `TODO`: registrar, promover, cargar desde el registro y responder qué versión calculó cada tarifa.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python registrar_modelo.py 64
python versiones.py
python promover.py 1
pytest -v
```

**Respuesta 1 (antes de los TODO):** `registrar_modelo.py` dice `Version None`, `versiones.py` no imprime nada y `promover.py` termina con un error. ¿Qué dice el error, y qué te dice sobre lo que hay en el registro?

Como `registrar_modelo.py` no guardaba ni registraba el modelo, imprime `Version None` y no crea ninguna versión de `tarifa-taxis`; por eso `versiones.py` no tiene nada que listar. `promover.py` intenta poner el alias en la versión 1 de un modelo que no existe y MLflow responde `RESOURCE_DOES_NOT_EXIST: Model Version (name=..., version=1) not found`. El registro está vacío: entrenar y medir el error no deja nada en él; solo queda lo que se registra explícitamente.

**Respuesta 2 (TODO 1):** registra dos versiones, una con 64 neuronas y otra con 8. Antes de correr `versiones.py`, ¿cuántas versiones esperas ver, y qué pasó con la primera cuando registraste la segunda?

Esperaba dos versiones más, y que la primera siguiera intacta: registrar una versión nueva no reemplaza a la anterior, le suma un número. En mi registro ya había 4 versiones de pruebas anteriores (tres de 100 neuronas y una de 200), así que la de 64 neuronas quedó como **versión 5** (MAE 1.1695) y la de 8 como **versión 6** (MAE 1.1877). Las seis siguen ahí, cada una con su corrida y su error.

**Respuesta 3 (TODO 2 y 3):** pon `produccion` en la versión 2 y levanta el servicio (`uvicorn servicio:app --port 8000`). ¿Qué versión dice `/salud`? Ahora mueve el alias a la versión 1 sin reiniciar uvicorn: ¿qué versión responde el servicio? ¿Y después de reiniciarlo?

Con `produccion` en la versión de 8 neuronas (la 6), `/salud` dijo `"version_modelo": "6"`. Al mover el alias a la de 64 (la 5) **sin reiniciar** uvicorn, el servicio siguió respondiendo la versión 6: carga el modelo una sola vez, al arrancar, y no vuelve a mirar el registro. **Después de reiniciarlo**, respondió la 5. Mover el alias no cambia lo que sirve un servicio que ya está corriendo: hay que reiniciarlo (y en un contenedor, reconstruirlo o relanzarlo).

**Respuesta 4 (TODO 4):** las pruebas usan un modelo de prueba y una versión inventada, `"7"`, sin servidor de MLflow. ¿Qué prueba falla antes del `TODO 4`, y qué comprueba que el `/salud` no comprobaba?

Fallaba `test_cada_prediccion_dice_que_version_la_calculo`: la predicción devolvía solo la tarifa, sin `version_modelo`. `/salud` comprobaba que el servicio dice qué versión cargó; esta prueba comprueba que **cada tarifa** lleva la versión que la calculó, que es lo que permite responder un reclamo después, cuando el servicio ya pudo haber cambiado de versión. Con el `TODO 4`, 2 passed.

**Lo que respondió el servicio:**

El viaje de Midtown a JFK Airport (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta), con mi numeración (la versión de 8 neuronas es la 6 y la de 64 es la 5):

- Con la red de **8 neuronas** en producción: `{"tarifa_estimada": 69.98, "version_modelo": "6"}`.
- Con la de **64 neuronas**, después de reiniciar: `{"tarifa_estimada": 70.07, "version_modelo": "5"}`.
- Con el alias ya movido a la 5 pero sin reiniciar: seguía respondiendo `69.98` y `"version_modelo": "6"`.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que distingue un registro de modelos de sobrescribir un archivo: cada entrenamiento queda como una versión nueva con su número, su corrida y su métrica, ninguna borra a otra, y "cuál está en producción" es solo un alias que se mueve. Así se puede saber qué versión produjo cada predicción (si el servicio la devuelve) y volver a la anterior moviendo el alias, sin reentrenar. Con un archivo que se reemplaza, se pierde la versión anterior y no hay forma de saber cuál calculó una predicción de ayer.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión confirmó que el alias decide qué se sirve, pero que el servicio solo lo lee al arrancar: mover el alias sin reiniciar no cambia nada, como mostró el ejercicio.

## Ejercicio 2: el ciclo de vida del modelo de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El equipo de fraude entrena modelos candidatos cada tanto, y quiere una regla: una versión llega a `produccion` solo si detecta más fraudes (mayor recall al umbral de 0.10) que la que está ahí. [`ciclo_fraude.py`](./starter/ejercicio2/ciclo_fraude.py) entrena tres candidatos, registra cada uno y decide si lo promueve. Los pasos 1 a 3 tienen `TODO`: registrar una versión con sus métricas, leer la métrica de una versión y averiguar qué versión está en producción. Los pasos 4 (decidir y mover el alias) y 5 (etiquetar la versión con el resultado) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python ciclo_fraude.py`.

**Respuesta 1:** ¿qué versión quedó en producción, y por qué no la última que se registró?

La **versión 2**, el gradient boosting (recall 0.661, precisión 0.429). La 1 (logística, recall 0.644) entró primero porque no había ninguna en producción; la 2 la reemplazó porque detecta más fraudes; y la 3, la última registrada, no llegó porque la regla no mira cuál es la más reciente sino cuál detecta más fraudes, y la 3 tiene un recall de solo 0.102.

**Respuesta 2:** la regresión logística muy regularizada tiene la mayor precisión de las tres. ¿Por qué la regla la rechazó? ¿Qué tendría que cambiar en la regla si la empresa decide que las falsas alarmas cuestan más que los fraudes no detectados?

Porque la regla mira solo el recall: con `C=0.001` la logística casi nunca marca un pago, así que cuando lo marca suele acertar (precisión 0.667), pero deja pasar casi 9 de cada 10 fraudes (recall 0.102), justo lo que la empresa quiere evitar. Si las falsas alarmas costaran más que los fraudes no detectados, la regla tendría que comparar el costo total con los dos costos (fraudes que pasan × su costo + falsas alarmas × el suyo, como el umbral de la Semana 11) o priorizar la precisión con un recall mínimo, y probablemente también subir el umbral de 0.10.

**Respuesta 3:** abre el modelo `fraude-pagos` en la interfaz de MLflow (puerto 5050, pestaña **Models**). Si dentro de un mes alguien pregunta por qué la versión 3 nunca llegó a producción, ¿dónde está la respuesta?

En las etiquetas de la versión 3, en la pestaña **Models** de MLflow: `validacion: rechazada` y `comparada_con: 2`. Dicen que se rechazó al compararla con la versión 2, y la corrida de la versión 3 (enlazada desde la versión) guarda su recall (0.102) y su precisión, así que cualquiera puede ver que no superó el recall de la 2 (0.661). La decisión y su evidencia quedan en el registro, no en la memoria de quien la tomó.

**Si terminas antes:** vuelve a correr `python ciclo_fraude.py`. ¿Cuántas versiones hay ahora, y cuál está en producción? ¿Es lo que esperabas?

Seis versiones: el ciclo registró otras tres (4, 5 y 6) y las tres quedaron `rechazada`, comparadas con la 2, que sigue en producción. Es lo que esperaba: los candidatos son los mismos, con la misma semilla y los mismos datos, así que dan el mismo recall, y la regla pide superar a producción *estrictamente*; un empate no promueve. El registro igual guarda cada intento, que es lo que debe hacer.

## Ejercicio 3: el registro de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio que construiste la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: el modelo, en el registro

- `servicio/registrar_modelo.py`: entrena tu modelo final, guarda su métrica principal en la corrida y lo registra como una versión nueva de un modelo con el nombre de tu proyecto. Cada vez que lo corres crea una versión; ninguna reemplaza a otra.
- `servicio/promover.py`: pone el alias `produccion` en la versión que le pases.
- Al menos dos versiones registradas (por ejemplo, con dos valores de un hiperparámetro), y una de ellas con el alias `produccion`.

Corre los scripts desde `psets/proyecto/` como módulos (`python -m servicio.registrar_modelo`), para que encuentren `modelado/`.

### Entregable 2: el servicio sabe qué versión sirve

- El script que exportaba el modelo (Semana 12) ahora exporta la versión con el alias `produccion`, y guarda su número junto al modelo (por ejemplo, en `servicio/modelo_version.txt`). La imagen lleva los dos archivos adentro: el contenedor no necesita llegar al servidor de MLflow.
- `/salud` y cada predicción responden `version_modelo`.
- Una prueba en `servicio/test_api.py` que compruebe que la predicción incluye la versión, con un modelo de prueba y una versión inventada.

### Entregable 3: `docs/despliegue.md`

Agrega la sección **Versionado y registro**: cómo se llama tu modelo en el registro, qué versión está en producción y con qué métrica, con qué criterio se mueve el alias, y los pasos para volver a la versión anterior.

**¿Con qué criterio decides que una versión nueva de tu modelo pasa a producción? ¿Quién lo decide en tu proyecto, y dónde queda registrada la decisión?**

Una versión nueva pasa a producción solo si, con la misma división y los mismos datos (misma huella), mejora el MAE de la que está en producción **por más que la variación entre pliegues de la validación temporal** (±0.037) y no empeora el costo asimétrico de subestimar. La versión 2 (`max_depth=5`) mejora 0.016, dentro del ruido, y tarda un 50% más; por eso `produccion` se queda en la versión 1 (`max_depth=3`, MAE 1.0278). Lo decide quien mantiene el modelo (en este proyecto, yo) con `python -m servicio.promover`. La decisión queda en el registro (el alias `produccion` en `ecobici-rebalanceo`, con la métrica en la corrida de cada versión) y en la sección "Versionado y registro" de [`docs/despliegue.md`](../../psets/proyecto/docs/despliegue.md).

**Si tu servicio corre en un contenedor sin acceso al servidor de MLflow, ¿cómo sabe qué versión está sirviendo?**

Porque la versión viaja con el modelo: `exportar_modelo.py` busca la versión con el alias `produccion`, guarda el modelo en `servicio/modelo.joblib` y su número en `servicio/modelo_version.txt`, y la imagen lleva los dos archivos. Al arrancar, el servicio lee ese número y lo devuelve en `/salud` y en cada predicción. En el contenedor: `{"estado":"ok","modelo_cargado":true,"version_modelo":"1"}` y, para la estación 495 a las 8:00 del 26 de agosto, `{"estacion_codigo":"495","salidas_prox_30m":10.5,"version_modelo":"1"}`, sin conexión a MLflow. Por eso, cambiar de versión exige exportar de nuevo y reconstruir la imagen. La prueba `test_cada_prediccion_dice_que_version_la_calculo` lo comprueba con una versión inventada (`"7"`) y un modelo de prueba (10 passed).

## Antes de la próxima sesión

**Lectura:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Se degrada en silencio: sigue respondiendo en milisegundos, `/salud` dice `ok` y cada respuesta trae su versión, pero sus predicciones son cada vez peores, porque el mundo cambió y el modelo no. La lectura de Datadog nombra las causas: **data drift** (las entradas cambian de distribución, por ejemplo otra estación del año u otras estaciones de Ecobici), **concept drift** (cambia la relación entre las entradas y lo que se predice, por ejemplo una línea de metro nueva), **training-serving skew** (el servicio calcula una feature distinto que el entrenamiento) y fallas del pipeline de datos (una columna que cambia o datos faltantes). Monitorear la salud y el volumen del servicio no alcanza. Me enteraría midiendo el modelo, no el servicio: guardando cada predicción y comparándola con lo que de verdad pasó cuando se conoce (en mi caso, las salidas reales 30 minutos después), con el MAE por día y por versión; y, mientras llegan las etiquetas, vigilando la distribución de las entradas y de las predicciones contra la del entrenamiento, con una alerta cuando se aleje.
