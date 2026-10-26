# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

scikit-learn ya viene en el Codespace. En tu laptop, instálalo con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. El `requirements.txt` del repositorio ya fija una versión posterior.

Ten a mano el script de features de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

La fuga de información ocurre cuando el modelo, durante el entrenamiento, recibe información que no tendría disponible al momento de predecir en producción: datos del conjunto de prueba, del futuro o columnas que se calculan con el propio objetivo. El modelo aprende a aprovechar esa información "prestada", así que en la evaluación se ve excelente (la evaluación también la contiene), pero en producción esa información no existe y el desempeño cae, a veces de forma drástica y sin que nadie lo haya anticipado.

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

La fuga por usar como feature una variable que solo se conoce después del evento que se quiere predecir (fuga del objetivo o "target leakage"): la duración del viaje se conoce cuando el viaje termina, igual que la tarifa, así que usarla para predecir la tarifa le da al modelo una pista que no existe al pedir el taxi.

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. La media y la desviación estándar son parámetros que el preprocesamiento "aprende", igual que los coeficientes del modelo: si se calculan con todos los viajes, los de prueba influyen en cómo se transforman los datos de entrenamiento y la evaluación deja de ser sobre datos nunca vistos. Numéricamente puede cambiar poco, pero el procedimiento deja de reproducir lo que pasará en producción, donde los viajes nuevos no existían cuando se calcularon esas estadísticas.

## Ejercicio 1: encontrar y corregir la fuga

Archivo: [`starter/ejercicio1_fuga.py`](./starter/ejercicio1_fuga.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv): los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. La primera parte del script es el código de un compañero: imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. La segunda parte es la versión correcta, con 3 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-02/starter
python3 ejercicio1_fuga.py
```

**Respuesta 1, antes de ver la solución:** en la versión del compañero, ¿en qué línea entra el conjunto de prueba al entrenamiento? ¿Qué número de la salida lo delata?

En la línea 26, `X_preparado = escalador.fit_transform(imputador.fit_transform(X))`: el imputador y el escalador se ajustan con `X` completo, antes de `train_test_split`, así que las medianas, la media y la desviación estándar ya incluyen las filas de prueba. Lo delata la salida "Filas que vio el escalador: 2975.0", que es el dataset entero; si solo hubiera visto entrenamiento, serían 2380 (el 80%).

**Respuesta 2 (TODO 1 y 3):** sin resolver, la versión correcta dice que tiene 2975 filas de prueba. ¿Qué está midiendo ese error, si prueba y entrenamiento son las mismas filas?

Mide el error de entrenamiento, no el de prueba: el modelo se evalúa sobre las mismas filas que usó para aprender, así que el número dice qué tan bien memoriza o se ajusta a datos que ya vio, no qué tan bien generaliza a viajes nuevos. Ese error (2.5590) no sirve para estimar cómo se comportará el modelo en producción.

**Respuesta 3:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

Esperaba un cambio pequeño, y así fue: el error pasó de 2.5375 (con fuga) a 2.5373 (corregido), prácticamente igual. Eso no significa que no había fuga: la fuga es un error de procedimiento y existía igual (el escalador vio las 2975 filas). Aquí casi no se nota porque una media y una desviación estándar calculadas con 2975 o con 2380 filas son casi iguales (8.5034 contra 8.5576) y una regresión lineal es poco sensible a eso. Con transformaciones que aprenden más del dato (codificar con el objetivo, seleccionar features, imputar con modelos) o con datos que cambian en el tiempo, la misma fuga puede inflar mucho la evaluación.

**Lo que confirmó la salida, TODO por TODO:**

- Versión del compañero: el escalador vio 2975 filas (todo el dataset), media de distancia 8.5034, error absoluto medio en prueba 2.5375.
- TODO 1: con la división antes de cualquier estadística quedan 2380 filas de entrenamiento y 595 de prueba (antes de resolverlo decía 2975 filas de prueba y error 2.5590, porque evaluaba sobre entrenamiento).
- TODO 2: el pipeline imputa con la mediana, escala y entrena la regresión lineal (antes decía "El pipeline todavía no escala").
- TODO 3: con `fit` solo sobre entrenamiento, el escalador vio 2380 filas y usó una media de distancia de 8.5576. Error absoluto medio en prueba: 2.5373.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

C

**Tu razonamiento:**

Elegí la opción que pone la división en entrenamiento y prueba antes de calcular cualquier estadística: todo lo que se "aprende" de los datos (medianas para imputar, media y desviación para escalar, categorías para codificar) tiene que salir solo de entrenamiento, y el pipeline garantiza que se aplique lo mismo a prueba sin volver a aprenderlo. Las otras opciones o ajustaban el preprocesamiento con todo el dataset, o confundían una métrica parecida con la ausencia de fuga.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión reforzó que la fuga se juzga por el procedimiento y no por cuánto cambia el error.

## Ejercicio 2: el pipeline completo

Solo o en pareja. Archivo: [`starter/ejercicio2_pipeline.py`](./starter/ejercicio2_pipeline.py), sobre el mismo `viajes_features.csv`. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_pipeline.py
```

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

Quedan 25 columnas (entrenamiento 2380 x 25, prueba 595 x 25). Salen de los tres grupos:
- Numéricas: 2 (`distancia_km` y `pasajeros`), imputadas y escaladas, siguen siendo una columna cada una.
- Categóricas: 18, porque one-hot crea una columna por cada categoría vista en entrenamiento: 8 zonas de origen + 8 zonas de destino + 2 métodos de pago.
- Ya listas: 5 (`hora_sin`, `hora_cos`, `es_fin_de_semana`, `es_hora_pico`, `es_tarifa_fija`), pasan sin cambios.

2 + 18 + 5 = 25.

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Coinciden con las de entrenamiento: el imputador aprendió `distancia_km` = 6.94 y `pasajeros` = 1.0, iguales a las medianas de entrenamiento, mientras que la mediana de la distancia en todo el dataset es 6.93. Eso demuestra que el `ColumnTransformer` aprendió sus estadísticas solo con entrenamiento: las filas de prueba no influyeron en cómo se rellenan los faltantes, que es justo lo que evita la fuga del Ejercicio 1.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

El Bronx no apareció en entrenamiento, así que el `OneHotEncoder` con `handle_unknown="ignore"` lo transforma en 0 en las 8 columnas de `zona_origen` (ninguna queda en 1), sin error y conservando las mismas 25 columnas; el modelo lo trata como "ninguna de las zonas conocidas". Con `pd.get_dummies` aplicado por separado a ese viaje se habría creado una sola columna nueva, `zona_origen_Bronx`, y habrían faltado todas las demás: el viaje tendría un número y un orden de columnas distinto al del entrenamiento, y el modelo fallaría o, peor, interpretaría columnas equivocadas.

## Ejercicio 3: el pipeline de tu proyecto

Usa tu archivo con las 2 features de la sesión pasada y la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py).

### Paso 1: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

_(tu clasificación)_

### Paso 2: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha; si no, al azar. Justifica.

_(tu decisión)_

### Paso 3: construye y corre el pipeline (15 minutos)

Completa la plantilla y córrela. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

_(lo que imprimió tu pipeline)_

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está sobreajustando: aprendió detalles y ruido propios del conjunto de entrenamiento (casi lo memorizó) en lugar del patrón general, por eso se equivoca poco en los datos que ya vio y mucho más en los nuevos. La brecha entre el error de entrenamiento y el de prueba es la señal; para corregirlo se simplifica el modelo (limitar la profundidad, regularizar) o se combinan muchos modelos, por ejemplo con un ensamblaje.
