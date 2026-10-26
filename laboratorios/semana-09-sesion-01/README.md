# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tus 3 hallazgos de la exploración de la Semana 8, Sesión 2: las features de hoy salen de ahí.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

Una feature con sentido de negocio traduce un mecanismo real del dominio (una regla de precios, un comportamiento de los clientes, una restricción operativa) a una columna que se puede explicar y que sigue siendo válida cuando el modelo está en producción. Una feature que solo mejora la métrica puede estar capturando ruido, una coincidencia del conjunto de datos o información que no existe al momento de predecir: sube el número en la evaluación, pero nadie puede explicar por qué funciona y se rompe cuando cambian los datos.

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

El patrón de convertir dos columnas crudas en una razón que el negocio ya usa para decidir: por ejemplo, en vez de darle al modelo el monto total gastado y el número de compras por separado, se crea el gasto promedio por compra (monto total / número de compras). El conocimiento de negocio es que lo que distingue a un cliente no es cuánto gastó en total, sino cuánto gasta cada vez; el modelo no tiene que "descubrir" esa división, se la damos hecha como una columna nueva.

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

Construyendo columnas nuevas que hagan explícito cada patrón. Para la tarifa fija, una columna `es_tarifa_fija` que vale 1 cuando el viaje va de JFK a una zona de Manhattan o al revés; así el modelo puede asignarle a esos viajes un valor propio en lugar de aplicarles la regla de precio por distancia. Para el costo por kilómetro según la hora, columnas derivadas de `fecha_hora_inicio`: la hora, el día de la semana y un indicador `es_hora_pico`, que sí existen al pedir el taxi.

## Ejercicio 1: features de fecha y hora

Archivo: [`starter/ejercicio1_features_tiempo.py`](./starter/ejercicio1_features_tiempo.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada. El script ya quita los errores confirmados, convierte `fecha_hora_inicio` y compara la tarifa por kilómetro según la hora pico; faltan 4 `TODO`: hora, día de la semana, fin de semana, hora pico y la hora como ciclo. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-01/starter
python3 ejercicio1_features_tiempo.py
```

**Respuesta 1 (TODO 4), antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como números, 23 y 0 quedan a 23 unidades de distancia, la mayor distancia posible entre dos horas, aunque en el reloj solo las separa 1 hora. Un modelo que mide distancias entre números creería que las 23:00 y las 00:00 son los momentos más distintos del día. Representando la hora como un punto en un círculo (`hora_sin`, `hora_cos`), 23 y 0 quedan tan cerca como 11 y 12: la salida da 0.26 en los dos casos.

**Respuesta 2 (TODO 3):** sin el TODO 3, el Paso 4 muestra un solo grupo. Con la hora pico construida, ¿esperas que la tarifa por kilómetro sea más alta o más baja en hora pico? ¿Por qué, si la tarifa depende de la distancia y de la duración?

Más alta. La tarifa no depende solo de la distancia sino también del tiempo: en hora pico hay tráfico, el taxi tarda más minutos por cada kilómetro y el taxímetro cobra ese tiempo extra. La salida lo confirma: la mediana de la tarifa por kilómetro sube de 3.22 fuera de hora pico a 3.52 en hora pico (alrededor de 9% más).

**Respuesta 3:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

La duración le diría al modelo cuánto tráfico hubo en el viaje, es decir, cuántos minutos se pagaron por cada kilómetro. `es_hora_pico` no sabe la duración real, pero sí sabe si el viaje empieza en un horario donde normalmente hay congestión, y eso existe al momento de pedir el taxi. Trae la parte "esperada" del efecto del tráfico sin filtrar información del futuro.

**Lo que confirmó la salida, TODO por TODO:**

- TODO 1 y 2: `hora`, `dia_semana` y `es_fin_de_semana` salen de `fecha_hora_inicio` (por ejemplo, el primer viaje, 2026-09-06 13:46, es hora 13, día 6 = domingo, fin de semana).
- TODO 3: tarifa por km (mediana) = 3.22 fuera de hora pico (0) y 3.52 en hora pico (1). Antes de resolverlo había un solo grupo con 3.35.
- TODO 4: Paso 5, 23 y 0: como número 23, como ciclo 0.26. 11 y 12: como número 1, como ciclo 0.26.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que respeta el momento de la predicción: una feature solo sirve si existe cuando el modelo tiene que predecir, y su codificación depende del modelo que la va a consumir (por ejemplo, la hora como ciclo o en one-hot para un modelo lineal, mientras que un árbol puede trabajar con el número directamente). Las otras opciones o usaban información que solo se conoce al terminar el viaje, o asumían que una sola codificación sirve igual para cualquier modelo.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión confirmó el criterio: primero se descartan las columnas que no existen al predecir, y después se elige la codificación según el modelo.

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Archivo: [`starter/ejercicio2_features_zona.py`](./starter/ejercicio2_features_zona.py), sobre el mismo `viajes_taxi.csv`. Completa los 4 `TODO`. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_features_zona.py
```

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

300 viajes tienen `es_tarifa_fija` en 1, y todos cuestan exactamente 70.0: la tarifa mínima y la máxima son iguales. Para esos viajes la tarifa no depende de la distancia ni de la duración. Un modelo lineal puede aprender con esta columna un ajuste propio para esos viajes (un coeficiente que los lleva hacia 70), en lugar de forzarlos a seguir la recta de precio por kilómetro; sin la columna, esos 300 viajes distorsionan la pendiente de la distancia para todos los demás. Aun así, como el efecto de la distancia sigue sumando, un modelo lineal no puede anular del todo la distancia en esos viajes; un árbol lo resolvería con una sola partición.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea 8 columnas (una por cada zona de origen: Brooklyn, Harlem, JFK Airport, Lower Manhattan, Midtown, Queens, Upper East Side y Upper West Side); la codificación ordinal crea 1 sola columna, `origen_ordinal`, con valores del 0 al 7. Para una regresión lineal usaría one-hot. La ordinal le da información falsa porque el modelo multiplica el número por un coeficiente: asume que las zonas tienen un orden y que están igual de separadas (que Upper West Side, 7, "vale" siete veces más que Harlem, 1, o que Midtown, 4, está a medio camino entre Brooklyn y Upper West Side), cuando el número solo sale del orden alfabético.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

`duracion_min`, `metodo_pago`, `propina` y la propia `tarifa`, que es el objetivo. Las cuatro se conocen recién cuando el viaje termina (la distancia sí existe, porque la app la estima con la ruta). Si las usara para entrenar habría fuga de información: el modelo aprendería, por ejemplo, que la duración explica casi toda la tarifa y se vería excelente en la evaluación, pero en producción, al momento de pedir el taxi, esas columnas no existen y el modelo no podría usarse o sus predicciones se derrumbarían.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de la exploración de la Semana 8 y la plantilla [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py).

### Paso 1: el momento de la predicción (5 minutos)

Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

Proyecto: flujo de estaciones de Ecobici (CDMX), agosto de 2026, 100 estaciones en franjas de 15 minutos (297,600 filas). El modelo predice `y_salidas_prox_30m`: cuántas bicicletas van a salir de una estación en los próximos 30 minutos, y la predicción se hace al cierre de cada franja de 15 minutos, para anticipar si una estación se va a vaciar.

- **Existe en ese instante:** la estación y sus atributos fijos (código, ubicación, `capacidad`), la fecha, la hora, la franja, el día de la semana, si es fin de semana u hora pico, y todo el flujo ya observado: `salidas` y `llegadas` de la franja que acaba de cerrar, `salidas_lag_15m`, `salidas_ultima_hora` y `salidas_mismo_momento_ayer`.
- **Todavía no existe:** las salidas y llegadas de los próximos 30 minutos (el objetivo) y cualquier columna calculada con ellas.

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

Las dos están en [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py):

```python
# Feature 1: periodo de demanda segun la hora.
hora = datos["hora"]
datos["periodo_demanda"] = np.select(
    [hora < 5, (hora >= 7) & (hora <= 9), (hora >= 14) & (hora <= 18)],
    ["madrugada", "pico_manana", "pico_tarde"],
    default="resto",
)

# Feature 2: salidas de la ultima hora relativas a la capacidad de la estacion.
datos["salidas_hora_por_capacidad"] = (
    datos["salidas_ultima_hora"] / datos["capacidad"].clip(lower=1)
)
```

### Paso 3: justifica cada feature (5 minutos)

Para cada una: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta (una correlación, una diferencia de promedios), y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar.

**Feature 1:**

`periodo_demanda` sale directamente del hallazgo de la exploración de la Semana 8: de 1 a 4 de la mañana prácticamente no hay salidas, y la demanda sube en la mañana (7 a 9) y en la tarde (14 a 18). La evidencia es la diferencia de promedios del objetivo por periodo: madrugada 0.02 salidas en los próximos 30 min, resto 1.39, pico de la mañana 2.27 y pico de la tarde 2.62 (el promedio general es 1.47). Existe al predecir porque solo usa la hora actual. Es categórica sin un orden real: para un modelo lineal (Ridge) usaría one-hot; para un ensamblaje de árboles también funciona con one-hot, y como son solo 4 categorías no agrega muchas columnas.

**Feature 2:**

`salidas_hora_por_capacidad` sale del conocimiento del dominio: 5 salidas en una hora no significan lo mismo en una estación de 11 anclajes que en una de 40; dividir entre la capacidad mide qué tan "presionada" está la estación y hace comparables estaciones de distinto tamaño. La evidencia: su correlación con el objetivo es 0.541. Es algo menor que la de `salidas_ultima_hora` sola (0.600), porque el objetivo está en salidas absolutas y no en proporción; su valor es complementar a la variable cruda, no reemplazarla, y sirve sobre todo para el objetivo de alertas de estaciones vacías (que dependen de la capacidad). Existe al predecir porque usa solo la última hora ya observada y un atributo fijo de la estación; la capacidad mínima es 11, así que nunca divide entre 0. Es numérica, no necesita codificación (solo escalado para el modelo lineal).

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. Si la media y la desviación estándar se calculan con todos los viajes, el conjunto de prueba "le cuenta" al preprocesamiento cómo son sus datos antes de evaluar, y el error en prueba sale más optimista de lo que será en producción, donde los viajes nuevos no estaban cuando se calcularon esas estadísticas. Con la distancia el cambio numérico puede ser pequeño, pero conceptualmente es una fuga de información: la evaluación deja de medir lo que el modelo haría con datos nunca vistos.
