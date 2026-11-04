# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el script `pipeline_proyecto.py` de tu proyecto, de la Semana 9, Sesión 2.

Los laboratorios de hoy son notebooks: entrenar y comparar modelos es experimentar, y en un notebook vuelves a correr solo el modelo que cambiaste, sin reentrenar los demás. La preparación no se copia en el notebook: se importa del script, para que todos los modelos usen exactamente la misma.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un ensamblaje combina varios modelos para que sus errores se compensen. Bagging entrena muchos modelos en paralelo, cada uno con una muestra distinta de los datos, y promedia sus predicciones. Boosting los entrena en secuencia: cada modelo nuevo corrige los errores que dejaron los anteriores.

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Porque cada árbol se equivoca de forma distinta. Al promediar, los errores en direcciones opuestas se cancelan y baja la varianza, mientras que lo que los árboles aprendieron en común se conserva.

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está sobreajustando: aprendió detalles y ruido del conjunto de entrenamiento en lugar del patrón general, por eso falla en filas que no vio.

## Ejercicio 1: línea base, un árbol y un bosque

Notebook: [`starter/ejercicio1_linea_base_y_ensamblaje.ipynb`](./starter/ejercicio1_linea_base_y_ensamblaje.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. La preparación de la Semana 9 está en [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py) y el notebook la importa. Ya trae una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos, cada uno en su celda. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Espero un error grande, de unos 12 dólares, porque responde lo mismo para un viaje corto y para uno a JFK. Salió 12.53 en prueba. Sirve como piso: un modelo que no le gana a esa regla no está aprendiendo nada útil.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

En entrenamiento un error casi cero, porque memoriza cada viaje, y en prueba bastante peor. Salió 0.0029 en entrenamiento y 1.7017 en prueba.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad la brecha baja mucho: de 1.70 a 0.26, y además mejora en prueba (1.27). Con 200 árboles esperaba el mejor resultado en prueba, pero no fue así: el bosque quedó en 1.30, con brecha 0.83, y el árbol limitado le ganó por poco.

**Lo que confirmó la tabla, TODO por TODO:**

| Modelo | Entrenamiento | Prueba | Brecha |
|---|---|---|---|
| Línea base | 13.1998 | 12.5308 | -0.6690 |
| Árbol sin límite | 0.0029 | 1.7017 | 1.6988 |
| Árbol `max_depth=8` | 1.0098 | 1.2728 | 0.2630 |
| Bosque aleatorio | 0.4739 | 1.3016 | 0.8276 |

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

C

**Tu razonamiento:**

Elegí la opción que juzga a un modelo por su error en prueba y por la brecha con entrenamiento, no por el error en entrenamiento. El árbol sin límite lo muestra: 0.0029 en entrenamiento y 1.7017 en prueba, porque memorizó los viajes. Limitar la profundidad o promediar árboles sacrifica un poco de entrenamiento a cambio de generalizar, y la línea base da el piso que cualquier modelo tiene que superar.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión confirmó el criterio, con un matiz que mostró la tabla: más complejidad no garantiza ganar. El bosque de 200 árboles (1.3016) quedó detrás del árbol con `max_depth=8` (1.2728), así que hay que medirlo en prueba y no suponerlo.

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Notebook: [`starter/ejercicio2_regularizacion.ipynb`](./starter/ejercicio2_regularizacion.ipynb), sobre el mismo `viajes_features.csv` y con la misma preparación importada. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

Recibe 3275 columnas y hay 2380 filas de entrenamiento. Con más columnas que filas, la regresión puede ajustar el entrenamiento casi a la perfección de muchas formas, y fuera de esas filas sus predicciones se disparan: el error en prueba fue de 307 871 307.81.

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

En entrenamiento pasó de 0.9138 a 1.0272, y en prueba de 307 871 307.81 a 1.3209. Ridge empeora un poco en entrenamiento porque penaliza los coeficientes grandes: ya no puede memorizar las filas, y a cambio generaliza.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

Gradient boosting, con 1.2754 en prueba. Son 11.26 dólares menos que la línea base (12.5308), cerca de 90% menos error.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

La línea base no aprendió nada: da siempre la misma mediana. La brecha negativa solo refleja que, por azar de la división, las tarifas de prueba quedaron algo más cerca de esa mediana que las de entrenamiento.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el notebook tú.

### Entregable 1: `modelos_proyecto.ipynb`

Un notebook, en la misma carpeta que tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división (`from pipeline_proyecto import cargar_y_dividir, construir_preparacion`) en vez de copiarlas, y que entrene:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento, escrita por el notebook. Antes de hacer commit, reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa, no de celdas corridas en momentos distintos.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

Persistencia: los próximos 30 minutos iguales a los últimos 30 (la franja actual más la anterior). La elegí porque predigo una serie en el tiempo y esta regla aprovecha lo mismo que usa una persona que mira la estación. Una mediana constante sería demasiado fácil de superar, porque de madrugada casi no hay salidas.

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

No. Las brechas son pequeñas: persistencia 0.0751, Ridge 0.0516 y gradient boosting 0.0539 (errores en prueba de 1.2278, 1.0632 y 1.0278). Hasta el modelo más flexible queda casi igual en entrenamiento (0.9739) que en prueba.

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

Ridge con `alpha=1.0` y gradient boosting con sus valores por defecto (`max_depth=3`). No probé otros valores: con brechas de 0.05 no hay sobreajuste que corregir, así que más regularización no tenía de dónde mejorar.

**¿Cuánto mejora tu mejor modelo a la línea base?**

Gradient boosting baja el error absoluto medio de 1.2278 a 1.0278: 0.20 salidas menos por predicción, 16.3% mejor que la línea base.

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error: cuánto cuesta cada predicción, cuánto tarda en responder, si el mismo viaje da siempre la misma tarifa, si se puede explicar una tarifa al conductor y cuánto cuesta entrenarlo y mantenerlo.
