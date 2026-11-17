# Semana 12, Sesión 1: Patrones de despliegue: ¿batch u online?

> Objetivo de la sesión: elegir entre servir el modelo de tu proyecto en modalidad batch o en modalidad online, justificando la decisión según la latencia que exige el caso de uso propuesto, y proteger con pruebas automáticas, que corren en cada cambio, el código que lo sirve.

**Lectura previa de esta sesión:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 7 (despliegue de modelos y servicio de predicción).

## Requisitos antes de esta sesión

Hoy se agregan tres herramientas: joblib (guardar un modelo entrenado en un archivo), pytest (pruebas automáticas) y GitHub Actions (integración continua). Las librerías nuevas ya están en `requirements.txt`, junto con FastAPI para la sesión siguiente. Sincroniza tu fork y reconstruye el contenedor ("Rebuild Container"), como en la Semana 11. Comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Debe mostrar `pytest 8.3.3` y `fastapi 0.115.4`.

Los laboratorios de hoy son scripts, no notebooks: lo que producen corre sin nadie mirando. Cada ejercicio entrena primero su modelo con un script que ya viene completo (`entrenar.py`), que lo guarda en un archivo `.joblib` dentro de la carpeta del ejercicio. Ese archivo no se sube al repositorio (`*.joblib` está en `.gitignore`).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué distingue un sistema de predicción batch de uno online? Da un ejemplo de cada uno.**

El artículo distingue entre entrenar una vez y aprender todo el tiempo. En el sistema **batch** el modelo se entrena offline con todos los datos históricos, se despliega y ya no aprende: se reentrena cada cierto tiempo (cada semana, cada mes) con datos nuevos. Ejemplo: un detector de fraude, cuyos patrones cambian lento. En el **online** el modelo sigue aprendiendo en producción, con mini-lotes de los datos que van llegando, y se adapta a los patrones nuevos casi en tiempo real. Ejemplo: un sistema de recomendaciones o la predicción de precios de acciones. Llevado a las predicciones de este laboratorio: batch es calcular de antemano las predicciones de todos los casos conocidos y dejarlas en una tabla; online es responder cada caso en el momento en que llega.

**¿Qué limitación tiene el sistema batch que el online resuelve, y qué cuesta resolverla?**

El batch queda congelado en el momento en que se entrenó: cuando cambia el comportamiento real, el modelo se desactualiza y no reacciona hasta el siguiente reentrenamiento (y no puede responder casos que no estaban en el lote). El online lo resuelve adaptándose de inmediato, pero cuesta: puede "olvidar" patrones de largo plazo por darle demasiado peso a lo reciente, un dato malo lo contamina al instante porque no hay tiempo de limpiarlo, no todos los algoritmos aprenden de a poco (los ensamblajes de árboles, por ejemplo) y, como el modelo cambia todo el tiempo, reproducir y depurar un error es mucho más difícil. Por eso muchos sistemas combinan los dos: base batch y ajuste online.

## Pregunta de apertura

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La tarifa sí se podría calcular de noche, al menos en parte: depende de origen, destino, hora y distancia, que se pueden tabular por pares de zonas y franja horaria (8 × 8 zonas × 24 horas), y para los viajes reservados se conocen todos los casos de antemano. El fraude no: depende del pago concreto (monto, tarjeta nueva, intentos fallidos), que no existe hasta que el pasajero paga, y la respuesta tiene que llegar antes de aprobar el cobro, en menos de un segundo (del orden de cien milisegundos). La tarifa de un viaje pedido en el momento también tiene que aparecer al abrir la app, en menos de un segundo.

## Ejercicio 1: la tarifa de los viajes programados, por lotes

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). La app permite reservar un viaje para el día siguiente. Cada noche, [`predecir_lote.py`](./starter/ejercicio1/predecir_lote.py) estima la tarifa de los 60 viajes reservados en [`viajes_programados.csv`](./starter/ejercicio1/viajes_programados.csv) con la red neuronal que elegiste en la Semana 11, y deja las estimaciones en una tabla que la app lee por la mañana. [`test_predecir_lote.py`](./starter/ejercicio1/test_predecir_lote.py) tiene sus pruebas. Los 5 `TODO`: cargar el modelo, estimar, armar la tabla de salida y dos pruebas.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python predecir_lote.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** sin resolver, `predecir_lote.py` escribe una tabla con una sola columna, y `pytest` dice `1 failed, 2 passed`. ¿Cómo pueden pasar dos pruebas si el script no estima ninguna tarifa?

Porque las pruebas de los `TODO 4` y `5` todavía no tenían ningún `assert`: pytest da por buena cualquier prueba que termina sin error, aunque no compruebe nada. La única que fallaba era `test_columnas_de_salida`, porque la tabla tenía solo `id_viaje`. Que una prueba pase no dice nada si no tiene `assert`.

**Respuesta 2 (TODO 1 y 2):** 4 de los 60 viajes no tienen pasajeros y 2 no tienen método de pago. ¿Esperas que el script falle en esos viajes, que los salte, o que les estime una tarifa? ¿Por qué?

Que les estime una tarifa. El modelo es el pipeline completo, con la preparación adentro: el `SimpleImputer` rellena `pasajeros` con la mediana de entrenamiento y `metodo_pago` con el valor más frecuente, así que el script no falla ni se salta esos viajes. Saltarlos sería peor: por la mañana la app no encontraría la tarifa de un viaje reservado. Así fue: los 60 viajes salieron con tarifa.

**Respuesta 3 (TODO 3):** todas las filas de la tabla llevan la misma `version_modelo`. ¿Para qué guardarla en cada fila?

Para saber qué modelo produjo cada tarifa. La tabla se lee al día siguiente y el modelo se reentrena cada cierto tiempo: si un pasajero reclama, o si un día las tarifas salen raras, la `version_modelo` (la huella MD5 del archivo `.joblib`) dice exactamente qué archivo las calculó, aunque la tabla se mezcle con lotes de otros días o el modelo ya haya cambiado. Es la misma idea de la huella de los datos de la Semana 11, aplicada al modelo.

**Respuesta 4 (TODO 4 y 5):** antes de escribir las dos pruebas, ¿qué cambio en `predecir_lote.py` haría fallar la del `TODO 4`? ¿Y la del `TODO 5`?

El `TODO 4` fallaría si el script se salta o reordena filas: por ejemplo, un `viajes.dropna()` antes de estimar (quedarían 54 filas, no 60) o un `sort_values("tarifa_estimada")` en la salida (los `id_viaje` dejan de estar en el orden de entrada). El `TODO 5` fallaría si un viaje incompleto se queda sin tarifa (por ejemplo, un modelo sin la imputación de la preparación, o devolver `NaN` en vez de estimar) o si la versión no es la que se pasó (escribir un texto fijo en vez de usar `version_modelo`).

**Lo que mostró el lote:**

- **Versión del modelo:** `bc17c84eead9` (la misma en las 60 filas).
- **Tarifa más baja:** 4.67, viaje 90014, Upper East Side → Upper East Side, 0.40 km.
- **Tarifa más alta:** 70.26, viaje 90051, Upper East Side → JFK Airport, 24.04 km: la tarifa fija de 70 del aeropuerto.
- **Viajes sin pasajeros:** 90023 (Queens → Upper West Side, 10.58 km) 33.94; 90024 (Brooklyn → Harlem, 10.17 km) 29.88; 90025 (Midtown → Harlem, 8.40 km) 25.80; 90048 (Harlem → Upper East Side, 4.75 km) 17.58. Tarifas normales para su distancia: la mediana imputada no las distorsiona. Los dos sin método de pago (90044 y 90050) también se estimaron: 60.79 y 29.59.

`pytest -v`: 3 passed.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que decide el patrón por la latencia que exige el caso y por si los casos se conocen de antemano, no por cuál tecnología es más moderna. Si la decisión puede esperar horas y los casos están en una lista (los viajes reservados para mañana), batch: se calcula todo de noche y la app solo lee una tabla. Si la respuesta tiene que llegar en milisegundos y el caso no existe hasta ese momento (un pago que hay que aprobar), online. Y en los dos casos el código que sirve el modelo necesita pruebas que corran solas en cada cambio.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión confirmó que batch y online no compiten por ser mejores en general: cada uno responde a una latencia distinta, y lo que se pierde con batch es poder responder casos nuevos en el momento.

## Ejercicio 2: la cola de revisión de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). Cada madrugada, [`puntuar_pagos.py`](./starter/ejercicio2/puntuar_pagos.py) calcula la probabilidad de fraude de los 300 pagos de ayer ([`pagos_ayer.csv`](./starter/ejercicio2/pagos_ayer.csv)) con el gradient boosting de la Semana 11, y arma la cola de revisión manual con los pagos que alcanzan el umbral de 0.10 que elegiste ese día. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO`; el paso 4 (la cola) y el paso 5 (las pruebas de [`test_puntuar_pagos.py`](./starter/ejercicio2/test_puntuar_pagos.py)) solo tienen un comentario que dice qué hacer. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** ¿cuántos pagos quedaron en la cola de revisión? ¿Cuántos fraudes esperarías encontrar en los 300 pagos, si la proporción es la de la Semana 11?

26 pagos quedaron en la cola de revisión con el umbral 0.10. Con la proporción de la Semana 11 (295 fraudes en 6000 pagos, 4.9%), en 300 pagos esperaría unos 15 fraudes. La suma de las probabilidades del modelo da 19.6. La cola es más grande que los fraudes esperados a propósito: el umbral bajo acepta falsas alarmas, que cuestan 3 dólares, para dejar pasar menos fraudes, que cuestan 40.

**Respuesta 2:** este modelo corre por lotes, una vez al día. ¿Qué pierde la empresa por no evaluar cada pago en el momento en que ocurre? ¿Qué tendría que pasar con la revisión para que eso importe?

Pierde la posibilidad de frenar el fraude: cuando la cola se arma de madrugada, el pago ya se aprobó y el viaje ya ocurrió, así que lo único que queda es investigar y absorber el contracargo. Además, el estafador puede seguir usando la misma tarjeta todo el día antes de que alguien lo vea. Eso importa solo si la revisión puede actuar antes de aprobar el cobro: bloquear el pago o pedir una verificación en el momento, en una fracción de segundo. Mientras la revisión sea manual y se haga al día siguiente, puntuar en línea no cambia nada.

**Respuesta 3:** la prueba de la cola de revisión arma su propia tabla de cuatro pagos con probabilidades escritas a mano, en vez de usar el modelo. ¿Qué gana la prueba con eso?

Saber de antemano el resultado correcto: con probabilidades escritas a mano, la cola tiene que ser exactamente 4, 2 y 3, en ese orden, y la prueba comprueba la lógica del umbral (que el 0.10 entra porque es `>=`, que el 0.05 no, y el orden) sin depender de lo que estime el modelo. Así, si falla, el error está en `a_revisar` y no en el modelo. Además es rápida, siempre da lo mismo y no necesita el archivo `.joblib`, que no existe en GitHub Actions.

**Si terminas antes:** sube `UMBRAL` a 0.50 y vuelve a correr `puntuar_pagos.py`. ¿Cuántos pagos quedan en la cola? ¿Siguen pasando todas las pruebas? ¿Por qué?

Quedan 13 pagos en la cola, contra 26 con 0.10. Las pruebas siguen pasando todas, porque ninguna usa la constante `UMBRAL`: `test_cola_de_revision` llama a `a_revisar` con su propio umbral (0.10) sobre sus cuatro pagos escritos a mano. Las pruebas comprueban que el código hace bien su trabajo, no qué umbral eligió el negocio; esa es una decisión de negocio que se justifica con los costos de la Semana 11, no con una prueba.

## Ejercicio 3: el despliegue y las pruebas de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/despliegue.md`

Una página que diga si tu modelo se sirve por lotes o en línea, y por qué:

- Qué decisión habilita la predicción, quién la usa y cuánto tiempo puede esperar por ella (un número: milisegundos, horas, un día).
- Cuántos casos distintos hay que predecir, y si se conocen de antemano.
- Qué datos necesitan tus features en el momento de predecir, y si están disponibles en ese momento.
- Qué se pierde con el patrón que descartas.

### Entregable 2: las primeras pruebas de tu proyecto

Un archivo `psets/proyecto/modelado/test_pipeline_proyecto.py` con al menos tres pruebas sobre tu `pipeline_proyecto.py`, cada una con al menos un `assert`. Las pruebas arman sus propios datos (un `DataFrame` de pocas filas escrito a mano), no leen tu archivo de `data/`: en GitHub Actions ese archivo no existe. Ideas de qué probar: que la columna objetivo no está entre las que recibe el modelo; que una feature calculada da el valor que sabes calcular a mano; que una feature que mira hacia atrás en el tiempo no usa una fila del futuro; que la preparación acepta una categoría que no vio al entrenar.

Comprueba desde la raíz de tu fork que pasan, con el mismo comando que va a usar GitHub:

```
pytest psets/proyecto -v
```

### Entregable 3: la integración continua de tu fork

El archivo [`.github/workflows/pruebas.yml`](../../.github/workflows/pruebas.yml) le pide a GitHub que, en cada push, cree una máquina limpia, instale `requirements.txt` y corra `pytest psets/proyecto`. En tu fork viene apagado: en la página de tu fork, pestaña **Actions**, presiona "I understand my workflows, go ahead and enable them". Haz push de tus pruebas y espera la marca verde junto a tu commit.

**Pega el enlace a la ejecución de GitHub Actions donde pasan tus pruebas:**

[https://github.com/Dmt-155lbs/data-science-end-to-end/actions/runs/43988950285](https://github.com/Dmt-155lbs/data-science-end-to-end/actions/runs/43988950285): las cuatro pruebas de [`test_pipeline_proyecto.py`](../../psets/proyecto/modelado/test_pipeline_proyecto.py) pasan (`4 passed`) en una máquina limpia de GitHub. Prueban que el objetivo no entra al modelo, que las dos features dan lo que se calcula a mano, que la división por fecha no deja franjas del futuro en entrenamiento y que la preparación acepta un periodo que no vio y rezagos faltantes.

**¿Qué patrón elegiste para tu proyecto, y qué número de latencia lo decide?**

**Por lotes, cada 15 minutos**, al cierre de cada franja ([`docs/despliegue.md`](../../psets/proyecto/docs/despliegue.md)). El número que lo decide: la predicción tiene que estar lista **entre 1 y 2 minutos** después del cierre de la franja, para un horizonte de 30 minutos. El despachador de rebalanceo la revisa cada 15 minutos y nadie espera una respuesta en milisegundos. Las 100 estaciones se conocen de antemano y se predicen juntas en una fracción de segundo, así que una API en línea no gana nada. Tampoco sirve un lote nocturno: las features son las salidas de la franja que acaba de cerrar, que no existen la noche anterior.

**Rompe a propósito una de tus pruebas (cambia un valor esperado), haz push y mira qué pasa en GitHub. Después arréglala. ¿Qué te dijo GitHub, y dónde lo viste?**

Cambié el valor esperado de `salidas_hora_por_capacidad` de 0.3 a 0.4 en `test_features_calculadas_a_mano` y hice push. GitHub marcó el commit con una **X roja** junto al mensaje, en la lista de commits y en la pestaña **Actions** ([ejecución fallida](https://github.com/Dmt-155lbs/data-science-end-to-end/actions/runs/43989150022)). En el log del paso "Correr las pruebas de psets/proyecto" dijo `1 failed, 3 passed`, con la prueba exacta y la línea: `assert 0.3 == 0.4 ± 4.0e-07`, `test_pipeline_proyecto.py:56: AssertionError`. Es decir, el código calcula 0.3 (6 salidas entre 20 anclajes) y la prueba esperaba 0.4. Volví a poner 0.3, hice push y la marca volvió a verde ([ejecución corregida](https://github.com/Dmt-155lbs/data-science-end-to-end/actions/runs/43989366392)).

## Antes de la próxima sesión

**Lectura:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

Hoy la app lee las tarifas de una tabla calculada de noche. Pero cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Un servicio de predicción en línea: un programa que corre todo el tiempo con el modelo ya cargado en memoria, detrás de una API (por ejemplo, un endpoint `POST /predecir` con FastAPI), y que responde en menos de un segundo. La app le mandaría, en un JSON, las features del viaje que existen en ese momento: zona de origen y de destino, la distancia estimada por la ruta, los pasajeros, el método de pago y la hora del pedido (o ya convertida en `hora_sin`, `hora_cos`, `es_fin_de_semana`, `es_hora_pico` y `es_tarifa_fija`). El servicio respondería la tarifa estimada y la versión del modelo, igual que la tabla de la noche, pero calculadas en el momento.
