# Semana 13, Sesión 2: Monitoreo en producción I: métricas de rendimiento

> Objetivo de la sesión: definir las métricas de monitoreo del modelo de tu proyecto en producción, distinguiendo métricas de sistema (latencia, disponibilidad) de métricas de modelo (desempeño).

**Lectura previa de esta sesión:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 8: Data Distribution Shifts and Monitoring.

## Requisitos antes de esta sesión

Nada nuevo que instalar. Los laboratorios de hoy son scripts. El Ejercicio 1 no necesita levantar uvicorn: `simular_noviembre.py` le manda las peticiones al servicio con `TestClient`, como las pruebas.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿por qué no alcanza con vigilar la salud y el volumen de peticiones del servicio? ¿Qué hay que medir además?**

Porque la calidad de un modelo se degrada con el tiempo aunque el servicio siga sano, y que el servicio responda rápido y sin errores no dice nada de si sus predicciones siguen siendo buenas. El artículo dice que no alcanza con vigilar la salud y el volumen del servicio: hay que evaluar el desempeño del modelo de forma continua. Las causas que nombra son el **data drift** (las entradas cambian de distribución), el **concept drift** (cambia la relación entre las entradas y lo que se predice, y lo aprendido deja de valer), el **training-serving skew** (producción calcula las features distinto que el entrenamiento) y los problemas del pipeline de datos (un esquema que cambia, datos faltantes o inválidos).

**El artículo dice que la verdad (ground truth) a veces llega con demora, o no llega. ¿Qué propone medir mientras tanto?**

Cuando se puede, compararlas con la verdad que llega después: unir los registros de predicción con los valores reales y calcular la métrica (un **backtest**), por día y por versión. Eso solo funciona si la verdad llega rápido y es confiable. Mientras llega, recomienda vigilar **proxies**: el drift de las predicciones (cambia su distribución) y el drift de los datos de entrada respecto del entrenamiento, que avisan antes aunque no prueban que el modelo esté peor; y alertar con pocos umbrales claros, para no saturar al equipo con alertas.

## Pregunta de apertura

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Se degrada en silencio: sigue respondiendo en milisegundos, `/salud` dice `ok` y cada tarifa trae su versión, pero se equivoca cada vez más porque el mundo cambió (otras tarifas, otro tráfico) y el modelo no. Ninguna de esas señales lo muestra, porque todas miden el servicio y no el modelo. Me enteraría guardando cada predicción y comparándola con la tarifa que de verdad se cobró, para medir el error por semana y por versión; y, mientras esa verdad llega, vigilando la distribución de los viajes que entran y de las tarifas que salen, contra la del entrenamiento.

## Ejercicio 1: la tarifa, instrumentada

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa con la versión 1 del modelo (la red de 64 neuronas), ahora con dos agregados. Un middleware (una función que corre alrededor de cada petición) mide cuánto tarda cada predicción y con qué código responde; `/metricas` resume esas mediciones. Y cada predicción queda guardada en `predicciones.jsonl`, con un identificador que la app guarda junto al viaje. [`simular_noviembre.py`](./starter/ejercicio1/simular_noviembre.py) manda al servicio los 1156 viajes de noviembre y escribe `cobros.csv`, lo que el sistema de cobros anotó al terminar cada viaje: el identificador de la predicción y la tarifa cobrada (no lo modifiques). [`desempeno.py`](./starter/ejercicio1/desempeno.py) une las dos cosas y calcula el error por semana. Los 4 `TODO`: medir la latencia, resumir las métricas, guardar cada predicción y calcular el error por semana.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python simular_noviembre.py
python desempeno.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** los 1156 viajes recibieron su tarifa. ¿Qué dicen `/metricas` y `desempeno.py`? ¿Qué podrías afirmar sobre el servicio en noviembre?

Nada: `/metricas` devolvía `{}` y `desempeno.py` terminaba con "No hay predicciones.jsonl: el servicio no está guardando sus predicciones". Lo único que podría afirmar sobre noviembre es que el servicio respondió a los 1156 viajes, pero no cuánto tardó, cuántas peticiones rechazó ni si las tarifas fueron buenas o malas. Que todos los viajes tengan tarifa no dice nada de la calidad de esas tarifas.

**Respuesta 2 (TODO 1 y 2):** antes de correr la simulación, ¿qué latencia p95 esperas, y cuántos errores de validación? ¿Esperas que la latencia cambie entre la primera y la última semana del mes?

Esperaba una latencia p95 de unos pocos milisegundos, como la de la Semana 12, y casi ningún error de validación, porque la app manda viajes bien formados (los pocos que hubiera serían viajes con un dato mal escrito). Esperaba que la latencia no cambiara entre semanas: el modelo hace el mismo cálculo para cualquier viaje, sea bueno o malo. Salió una p95 de **15.77 ms** (p50 de 10.91 ms) y **3 errores de validación** de 1159 peticiones, sin errores de servidor.

**Respuesta 3 (TODO 3):** el servicio devuelve `id_prediccion` y lo guarda en el registro. ¿Para qué lo necesita el sistema de cobros? ¿Qué pasaría si solo se guardaran la fecha y la tarifa estimada?

Para unir cada tarifa estimada con la tarifa cobrada del mismo viaje: el sistema de cobros guarda el `id_prediccion` junto al viaje, y así, al terminar, se sabe exactamente qué predicción le corresponde a qué cobro (es la columna que usa `desempeno.py` para unirlas). Si solo se guardara la tarifa estimada, no habría cómo emparejarla con el cobro (dos viajes pueden tener la misma tarifa, y uno puede reintentar), ni saber qué versión ni qué entrada la produjeron, así que no se podría medir el error real ni investigar un caso.

**Respuesta 4 (TODO 4):** el MAE en prueba de la versión 1 fue 1.17. Antes de correr `desempeno.py`, ¿qué esperas para cada semana de noviembre?

Esperaba algo parecido al 1.17 de prueba en todas las semanas, si los viajes de noviembre se parecen a los de octubre, o un aumento lento si algo cambiaba poco a poco. Las dos primeras semanas fueron así (1.18 y 1.34). Lo que no esperaba es el salto: desde la semana del 16 de noviembre el error casi se cuadruplica (4.29 y 4.30) de golpe, sin que la latencia ni los errores del servicio se movieran.

**Lo que mostró el monitoreo:**

| Medida | Valor |
|---|---|
| Latencia p95 | 15.77 ms (p50: 10.91 ms) |
| Errores de validación | 3 de 1159 peticiones |
| Errores de servidor | 0 |

| Semana | Viajes | MAE | Veces la referencia (1.17) |
|---|---|---|---|
| 2 de noviembre | 291 | 1.18 | 1.01 |
| 9 de noviembre | 318 | 1.34 | 1.15 |
| 16 de noviembre | 308 | 4.29 | 3.67 |
| 23 de noviembre | 239 | 4.30 | 3.68 |

`pytest -v`: 2 passed. El sistema estuvo sano todo el mes; el modelo, no.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

B

**Tu razonamiento:**

Elegí la opción que separa las métricas de sistema de las de modelo: la latencia, la disponibilidad y los errores dicen si el servicio responde, pero no si responde bien. La calidad del modelo solo se ve al comparar sus predicciones con la verdad, y para eso cada predicción tiene que quedar registrada con un identificador que permita unirla después. El ejercicio lo mostró: un servicio con p95 de 16 ms y cero errores de servidor estaba cuadruplicando su error desde el 16 de noviembre.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No. La discusión confirmó que un `/salud` en verde y una latencia baja no son evidencia de que el modelo funcione, y que el monitoreo del modelo depende de cuándo llega la verdad.

## Ejercicio 2: el fraude, cuando la verdad tarda tres semanas

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de fraude guardó los 5874 pagos que evaluó en noviembre en [`predicciones_fraude.csv`](./starter/ejercicio2/predicciones_fraude.csv): la probabilidad, si mandó el pago a revisión y cuánto tardó. Un fraude se confirma cuando el titular de la tarjeta reclama el cobro, entre 2 y 21 días después; los confirmados están en [`contracargos.csv`](./starter/ejercicio2/contracargos.csv). Hoy es 14 de diciembre. [`monitoreo_fraude.py`](./starter/ejercicio2/monitoreo_fraude.py) arma el reporte del mes. Los pasos 1 a 3 tienen `TODO`: la latencia p95 por día, la tasa de alertas por semana y los fraudes confirmados hasta hoy. Los pasos 4 (el recall de las semanas cuyos fraudes ya se confirmaron) y 5 (las alertas) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python monitoreo_fraude.py`.

**Respuesta 1:** el reporte da tres tipos de alerta. ¿Cuáles son de sistema y cuáles de modelo? ¿El día de la latencia alta coincide con alguna semana de recall bajo?

**De sistema:** la de latencia ("el 2026-11-10 la latencia p95 fue 78.2 ms"). **De modelo:** la tasa de alertas (indirecta: las semanas del 16 y del 23 de noviembre mandaron a revisión 0.050 y 0.046 de los pagos, contra 0.076 de referencia) y el recall (la semana del 16 de noviembre detectó solo 0.276 de los fraudes, contra 0.661 de referencia). El día de la latencia alta, el 10 de noviembre, no coincide con la semana del recall bajo, que empieza el 16: esa semana el recall fue 0.736, normal. Un problema del servicio y un problema del modelo son independientes.

**Respuesta 2:** la tasa de alertas se puede calcular el mismo día; el recall, tres semanas después. ¿Qué semana mostró primero el problema la tasa de alertas, y cuándo lo confirmó el recall? ¿Por qué la tasa de alertas no alcanza para decir que el modelo empeoró?

La tasa de alertas mostró el problema en la misma **semana del 16 de noviembre**, que se puede calcular ese mismo día; el recall de esa semana recién se puede calcular el 13 de diciembre (su domingo, el 22, más 21 días de demora). La tasa de alertas no se puede usar sola para decidir porque no mide si el modelo detecta los fraudes: cambia también si cambia la cantidad de fraudes o el tipo de pagos (menos alertas podría ser simplemente menos fraude). Es una señal temprana para investigar, no una prueba de que el modelo empeoró; la confirmación es el recall.

**Respuesta 3:** ¿por qué el reporte no muestra el recall de la semana del 23 de noviembre, si ya hay fraudes confirmados de esa semana?

Porque la semana del 23 de noviembre todavía no está madura: su domingo es el 29, y un fraude puede tardar hasta 21 días en confirmarse, así que sus fraudes recién están todos confirmados el 20 de diciembre. Al 14 de diciembre ya hay contracargos de esa semana, pero faltan los que todavía no llegan; un recall calculado con ellos saldría con un denominador incompleto y no se podría comparar con las semanas maduras. Por eso el reporte solo calcula el recall de las semanas cuyo domingo, más 21 días, ya pasó.

**Si terminas antes:** cambia `HOY` a `"2026-11-30"` y vuelve a correr. ¿Qué semanas tienen recall? ¿Desde qué día el reporte puede mostrar la caída del recall de la semana del 16?

Con `HOY = "2026-11-30"` solo tiene recall la semana del 2 de noviembre (0.767): su domingo, el 8, más 21 días es el 29 de noviembre. La del 9 necesita llegar al 6 de diciembre y la del 16, al 13 de diciembre. Lo comprobé con la función a distintas fechas: al 12 de diciembre aparecen las semanas del 2 y del 9; **desde el 13 de diciembre** aparece la del 16, con su recall de 0.276. Es decir, la caída que la tasa de alertas insinuó el mismo 16 de noviembre recién se confirma casi un mes después.

## Ejercicio 3: el monitoreo de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio de las dos sesiones anteriores. Esta vez no hay plantilla.

### Entregable 1: el servicio, instrumentado

- Un middleware que mida la latencia y el código de cada predicción, y una ruta `/metricas` con, al menos, la cantidad de peticiones, los errores de validación, los errores de servidor y la latencia p50 y p95.
- Cada predicción queda en un registro (un archivo JSONL en `monitoreo/`, por ejemplo) con su identificador, el momento, la versión del modelo, la entrada y la salida. Lo que haga falta para unirla después con la verdad tiene que estar en ese registro.
- Pruebas en `servicio/test_api.py` que comprueben que una predicción queda registrada y que `/metricas` separa los errores de validación de los de servidor, con un registro temporal (`tmp_path`).

### Entregable 2: `monitoreo/desempeno.py`

Une el registro con la verdad de tu proyecto y calcula la métrica de la Semana 11 por período (día o semana) y por versión. Si en tu proyecto la verdad no llega, o llega muy tarde, define una métrica que se pueda calcular antes y explica qué te dice y qué no.

### Entregable 3: `docs/monitoreo.md`

Una tabla con cada métrica que vigilas: si es de sistema o de modelo, de dónde sale, cuándo se conoce y con qué umbral alerta.

**¿Cuánto tarda en llegar la verdad en tu proyecto, y qué métrica te avisa mientras tanto?**

Llega **30 minutos después** de cada predicción: la verdad son las salidas reales de esa estación en la media hora siguiente, que se conocen con los viajes que terminan. Por eso, en mi proyecto, las métricas de modelo (el MAE y el costo asimétrico por día, en [`monitoreo/desempeno.py`](../../psets/proyecto/monitoreo/desempeno.py)) se pueden calcular el mismo día, a diferencia del fraude, que tarda tres semanas. Mientras tanto, lo que avisa en el momento es la **predicción media por franja**, comparada con la del mismo día de la semana anterior: si se mueve mucho, cambió lo que entra al modelo (otra demanda, una estación cerrada, rezagos que llegan vacíos), antes de saber si el error subió. En la semana simulada (25 al 31 de agosto) el MAE fue 1.0254, sin alertas, con la predicción media (1.55) pegada a las salidas reales medias (1.59). La tabla completa de métricas y umbrales está en [`docs/monitoreo.md`](../../psets/proyecto/docs/monitoreo.md).

**¿Qué falla de tu modelo no vería ninguna de tus métricas de sistema?**

Cualquier cambio en la demanda que el modelo no aprendió: que llegue diciembre con lluvias y vacaciones, que se abra una ciclovía o una estación nueva cerca, un evento, o que el lote empiece a mandar los rezagos mal calculados (por ejemplo, salidas de la última hora contadas en otra ventana). En todos esos casos el servicio sigue respondiendo en unos 50 ms, sin errores de validación ni de servidor, con 100 predicciones por franja y la versión correcta en `/salud`: todas mis métricas de sistema quedan en verde mientras las predicciones empeoran. Solo lo muestran el MAE y el costo asimétrico, al unir el registro con las salidas reales, y antes, como aviso, el cambio en la predicción media.

## Antes de la próxima sesión

**Lectura:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe el servicio son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

Cambió la relación entre los viajes y la tarifa, no los viajes: es **concept drift**. Según la lectura, el data drift es un cambio en los datos de entrada con las mismas relaciones entre variables, y el concept drift es un cambio en la relación entre las entradas y el objetivo "aunque los datos de entrada sigan iguales", como su ejemplo de una campaña de marketing que cambia cómo se comportan los clientes. Aquí los viajes tienen las mismas zonas, distancias y horas que en octubre, pero desde el 16 de noviembre el mismo viaje se cobra distinto (por ejemplo, una tarifa o un recargo nuevo), así que lo que el modelo aprendió dejó de valer y el error casi se cuadruplicó (de 1.18 a 4.29). **No lo hubiera detectado mirando solo los viajes que llegan**: las pruebas estadísticas sobre las entradas (Kolmogorov-Smirnov, Jensen-Shannon) no verían ningún cambio, porque las entradas no cambiaron. La lectura dice que el concept drift se detecta con las métricas de desempeño a lo largo del tiempo, comparando con modelos anteriores y vigilando la confianza de las predicciones. Aquí eso es justo `desempeno.py`: unir cada predicción con la tarifa cobrada por su `id_prediccion` y mirar el error por semana. La solución es reentrenar con viajes posteriores al 16 de noviembre, cuando se cobra la tarifa nueva.
