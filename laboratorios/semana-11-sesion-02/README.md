# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

Que una exactitud alta no significa que el modelo sirva. El ejemplo del artículo: una enfermedad rara que tiene 1 de cada 100 pacientes; un modelo que dice "sano" a todos acierta 99% y no detecta al único enfermo, que es justo para lo que existe. Pasa en datos desbalanceados. En su lugar recomienda métricas que miran la clase minoritaria: precision, recall y F1 (y PR-AUC en los LLM), o tratar el problema como detección de anomalías y balancear las clases.

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

Una agencia de noticias pone como meta subir 20% la duración de las sesiones; el equipo alarga las introducciones y agrega relleno: la duración sube, pero el contenido vale menos para el usuario. También da el de una app que esconde el botón de cancelar para bajar la tasa de abandono, y el de un LLM que copia frases enteras para subir su puntaje ROUGE. Con los modelos pasa lo mismo: si elijo solo por el menor error en una división, puedo terminar premiando al que mejor se ajustó a esos 595 viajes y no al que mejor sirve para el objetivo real, que es otra cosa (por ejemplo, no subestimar la tarifa).

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

No lo sé con un solo número: el registro tiene un error por corrida, sobre una sola división. Lo único que lo insinúa es la corrida del compañero: el mismo gradient boosting dio 1.2754 con la semilla 42 y 1.1867 con la 7. Esa variación de 0.09 no aparece como tal en ninguna columna; hay que provocarla cambiando la división, que es lo que hace la validación cruzada.

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Parecidos, pero no iguales: cada pliegue valida con otros 595 viajes. Esperaba que variaran unos ±0.05 alrededor de 1.25. Salieron entre 1.16 y 1.28, con promedio 1.2288 y desviación 0.0361.

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

Mayor. Con pliegues al azar el modelo entrena con viajes de todo el periodo, incluidos algunos posteriores a los que valida; en la temporal solo ve el pasado, y los primeros pliegues entrenan con pocos viajes. Así salió: promedio 1.3099 contra 1.2288 al azar, y con el doble de desviación (0.0693).

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

El pliegue que deja afuera a JFK Airport: ahí están los viajes de tarifa fija de 70 dólares y los más largos, y si ninguna fila de JFK está en entrenamiento, el modelo no tiene cómo aprenderlos. Así fue: el pliegue 5 (Brooklyn y JFK Airport, tarifa media 43.9 contra 26.5 del total) dio 6.43, mientras los demás quedaron entre 1.19 y 1.31.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El porcentaje de viajes subestimados en más de 20%: cuenta directamente los viajes en que el pasajero reclamaría. El MAE y el RMSE tratan igual pasarse que quedarse corto y no tienen el umbral del 20%; el RMSE solo castiga más los errores grandes. En prueba: MAE 1.2754, RMSE 1.9213 y 4.37% de viajes subestimados en más de 20%, es decir, unos 4 reclamos por cada 100 viajes.

**Lo que confirmó la tabla, esquema por esquema:**

| Esquema | Promedio | Desviación | Peor pliegue |
|---|---|---|---|
| Al azar, gradient boosting | 1.2288 | 0.0361 | 1.28 |
| Al azar, bosque | 1.2880 | 0.0291 | 1.34 |
| Temporal, gradient boosting | 1.3099 | 0.0693 | 1.40 |
| Por zona, gradient boosting | 2.2996 | 2.0648 | 6.43 (Brooklyn y JFK Airport) |

Con una sola división, gradient boosting 1.2754 y bosque 1.3016. Al azar el gradient boosting sigue ganándole al bosque (1.2288 contra 1.2880), y la diferencia (0.06) es mayor que la desviación de cada uno.

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción del esquema que imita cómo se va a usar el modelo: para estimar tarifas de días que todavía no ocurrieron, la validación temporal, porque con pliegues al azar el modelo entrena con viajes posteriores a los que valida y el error sale optimista (1.2288 al azar contra 1.3099 temporal). Y si el modelo se va a usar en zonas que nunca vio, la validación por grupo, que es la que revela el pliegue de JFK con 6.43.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión reforzó que el promedio de los pliegues no basta: hay que mirar la dispersión, porque una desviación alta dice que el error depende mucho de qué datos quedaron para validar.

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que mide el error que le cuesta al negocio y no la que da el número más bonito: con 5% de fraudes, una accuracy de 95% no detecta ninguno (la paradoja de la exactitud), y lo que importa es el recall y el costo de cada tipo de error. Por eso el umbral no es 0.5 por defecto, sino el que minimiza el costo total: con 40 dólares por fraude no detectado y 3 por falsa alarma, el umbral 0.1 cuesta 5546 contra 6402 del 0.5.

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

Accuracy 0.9508 y recall 0: acierta 95% porque 95% de los pagos son legítimos, y no detecta ni un fraude.

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

La accuracy sube apenas de 0.9508 a 0.9712 (+0.02); el recall sube de 0 a 0.4610: detecta casi la mitad de los fraudes. El recall describe mejor lo que gana la empresa, porque cada fraude detectado es un contracargo de 40 dólares que se evita; la accuracy casi no se mueve porque los fraudes son solo 5% de los pagos.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con el umbral 0.5 deja pasar 159 de los 295 fraudes (detecta 136) y da 14 falsas alarmas, con un costo de 6402 dólares. El paso 5 eligió el umbral 0.1: deja pasar 122 fraudes (detecta 173) a cambio de 222 falsas alarmas, con un costo de 5546 dólares, el menor de la lista.

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

Sube. Con 20 dólares por falsa alarma, el umbral de menor costo pasa a ser 0.5: 159 fraudes que pasan y 14 falsas alarmas, 6640 dólares, contra 9320 del umbral 0.1. Cuando la falsa alarma se encarece, conviene marcar menos pagos como fraude.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

La temporal. El modelo predice las salidas de los próximos 30 minutos a partir de lo que ya pasó, así que en producción siempre se entrena con el pasado y se predice el futuro. `TimeSeriesSplit` de 5 pliegues sobre las 297,500 franjas de agosto ordenadas por fecha y franja: cada pliegue entrena con los días anteriores y valida con los siguientes. Al azar, franjas vecinas de la misma estación (cuyos rezagos y objetivo se traslapan) caerían una en entrenamiento y otra en validación.

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

El orden no cambió, pero sí cuánto confiar en él. Tabla en [`validacion_proyecto.csv`](./proyecto/validacion_proyecto.csv):

| Esquema | Gradient boosting | Red neuronal |
|---|---|---|
| Temporal | 1.0000 ± 0.0369 (pliegues 0.95 a 1.05) | 0.9930 ± 0.0322 (0.96 a 1.04) |
| Al azar | 0.9904 ± 0.0022 (0.99 a 0.99) | 0.9790 ± 0.0053 (0.97 a 0.98) |

Al azar, el error es menor y la desviación es entre 6 y 17 veces más chica: parece que la diferencia entre los modelos fuera clara. Es optimista, porque cada pliegue entrena con días posteriores a los que valida. La temporal muestra que el error cambia de un tramo del mes a otro (de 0.95 a 1.05), algo que al azar no se ve.

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

No. En la validación temporal la red neuronal le gana al gradient boosting por 0.007 (0.9930 contra 1.0000), y la desviación entre pliegues es de 0.03 a 0.04, cinco veces más. En la división de la sesión pasada también fue 0.007. La diferencia está dentro del ruido de la partición, así que no hay evidencia de que la red sea mejor. Sigo eligiendo el gradient boosting: con rendimiento equivalente, entrena en 18.7 s contra 88.7 s y se puede explicar con sus importancias.

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

**El error caro es subestimar.** Si el modelo dice que saldrán pocas bicis y salen muchas, la estación se vacía sin que llegue el camión de rebalanceo, y cada bici que faltó es un viaje perdido. Sobreestimar cuesta un viaje del camión que no hacía falta, repartido entre las varias bicis que mueve. Supuesto aproximado: **una bici subestimada cuesta 3 veces una sobreestimada**. La métrica es el **costo asimétrico**: 3 por bici que faltó y 1 por bici de más, promediado por franja. Además mido qué proporción de las franjas de demanda alta (3 salidas o más) se quedó corta por 2 bicis o más. Sobre los pliegues temporales ([`metricas_negocio_proyecto.csv`](./proyecto/metricas_negocio_proyecto.csv)):

| Modelo | MAE | Costo asimétrico | Franjas subestimadas | Cortas en demanda alta |
|---|---|---|---|---|
| persistencia | 1.1767 | 2.3534 | 27.4% | 51.9% |
| gradient boosting | 1.0000 | 1.9993 | 39.0% | 39.1% |
| red neuronal | 0.9930 | 1.9889 | 38.7% | 39.2% |

El MAE trata igual quedarse corto que pasarse. La métrica nueva dice dos cosas que el MAE no decía. Primero, los modelos bajan el costo 15% respecto de la persistencia y fallan menos justo donde importa: se quedan cortos en 39% de las franjas de demanda alta, contra 52%. Segundo, se quedan cortos más seguido que la persistencia (39% de las franjas contra 27%), porque al minimizar el error absoluto apuntan al centro y no a cubrirse de la demanda alta. Para rebalanceo convendría corregir eso, por ejemplo entrenando con una pérdida asimétrica o por cuantiles. Entre los dos modelos, la diferencia de costo (0.01) tampoco cambia la elección.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La tarifa sí se podría calcular de noche, al menos en parte: depende de origen, destino, hora y distancia, que se pueden tabular por pares de zonas y franja horaria; aun así la app la tiene que mostrar al abrirla, en menos de un segundo, así que en la práctica es más simple predecirla en el momento con el modelo ya entrenado. El fraude no se puede precalcular: depende del pago concreto (monto, tarjeta nueva, intentos fallidos), que no existe hasta que el pasajero paga, y la respuesta tiene que llegar antes de aprobar el cobro, en una fracción de segundo. Ojo con la lectura: habla de *aprender* en batch u online (cuándo se entrena), que es otra decisión; los dos modelos se pueden entrenar en batch, reentrenando cada cierto tiempo, y el propio artículo pone la detección de fraude como ejemplo de patrones que cambian lento.
