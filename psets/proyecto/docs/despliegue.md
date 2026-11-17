# Despliegue: por lotes, cada 15 minutos

**Decisión:** el modelo se sirve **por lotes**, en un lote cada 15 minutos (al cierre de cada franja), no en línea. No es el lote de una vez por noche, porque las features necesitan la franja que acaba de cerrar.

## Qué decisión habilita la predicción, quién la usa y cuánto puede esperar

El modelo estima cuántas bicicletas van a salir de cada estación en los próximos 30 minutos. Con eso, el **despachador de rebalanceo** decide a qué estaciones mandar el camión antes de que se vacíen. La predicción solo sirve mientras quedan esos 30 minutos: si llega tarde, el camión ya no alcanza a llegar. El despachador revisa la lista al cierre de cada franja, así que la predicción puede esperar **del orden de 1 a 2 minutos** después del cierre, no segundos ni horas. Nadie está frente a una pantalla esperando una respuesta en milisegundos.

## Cuántos casos hay que predecir y si se conocen de antemano

**100 estaciones por franja**, unas 9,600 predicciones al día (96 franjas). Las estaciones se conocen de antemano y no cambian de una franja a otra: no aparece un caso nuevo que alguien pida en el momento. El lote completo se predice en una fracción de segundo con el gradient boosting ya entrenado.

## Qué datos necesitan las features y si están disponibles

Las features centrales son la historia reciente de cada estación: las salidas y llegadas de la franja que acaba de cerrar, las de la franja anterior, las de la última hora y las del mismo momento de ayer, más la hora, el día y la capacidad (ver [`pipeline_proyecto.py`](../modelado/pipeline_proyecto.py)). Todas existen **al cierre de la franja**: se calculan con los viajes que ya terminaron y con el catálogo de estaciones. Ninguna existe antes, y por eso **no se puede predecir todo de noche**: a las 3 de la mañana no se conocen las salidas de las 8:00.

## Qué se pierde con el patrón descartado

- **Con el servicio en línea (una API que responde cada consulta), que descarto:** no se pierde nada importante. Ningún usuario pregunta por una estación en un momento arbitrario; las 100 estaciones se predicen juntas y en el mismo instante, y una API que corra todo el tiempo agregaría costo y una pieza más que puede caerse, sin ganar nada en la decisión.
- **Con el lote cada 15 minutos, que elijo:** se pierde la posibilidad de reaccionar a mitad de una franja (por ejemplo, si una estación se vacía 5 minutos después del cierre) y de responder por una estación nueva que no está en el lote. Para el rebalanceo, que trabaja en ventanas de 15 a 30 minutos, eso no cambia la decisión.

**El número que decide:** la predicción tiene que estar lista entre 1 y 2 minutos después de cada cierre de franja, para cubrir un horizonte de 30 minutos. Eso es un lote frecuente, no una respuesta en milisegundos.
