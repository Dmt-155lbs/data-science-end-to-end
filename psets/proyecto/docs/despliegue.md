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

## Versionado y registro

**El modelo en el registro:** `ecobici-rebalanceo`, en el registro de modelos de MLflow. Cada vez que se corre `python -m servicio.registrar_modelo <profundidad>` (desde `psets/proyecto/`) se entrena el gradient boosting con la misma preparación, la misma división por fecha y la misma huella de datos, se guarda su MAE de prueba en la corrida (experimento `ecobici-registro`) y se registra una **versión nueva**. Ninguna reemplaza a otra.

| Versión | Modelo | MAE en prueba | Entrenamiento | Alias |
|---|---|---|---|---|
| 1 | Gradient boosting, `max_depth=3` (el modelo final de [`decision_modelo.md`](decision_modelo.md)) | 1.0278 | 19.8 s | **`produccion`** |
| 2 | Gradient boosting, `max_depth=5` | 1.0122 | 29.4 s | |

**Criterio para mover el alias `produccion`:** una versión nueva pasa a producción solo si, con la misma división y los mismos datos, **mejora el MAE de la que está en producción por más que la variación entre pliegues de la validación temporal** (±0.037, ver [`validacion_proyecto.csv`](../modelado/validacion_proyecto.csv)), y sin empeorar el costo asimétrico de subestimar (la métrica del negocio). La versión 2 mejora 0.016: está dentro del ruido y tarda un 50% más en entrenar, así que la versión 1 se queda. La decisión la toma quien mantiene el modelo (en este proyecto, yo) y queda registrada en el propio registro: la versión con el alias y esta tabla.

**Cómo llega al servicio:** `python servicio/exportar_modelo.py` busca la versión con el alias `produccion`, la guarda en `servicio/modelo.joblib` y guarda su número en `servicio/modelo_version.txt`. La imagen lleva los dos archivos, así que el contenedor no necesita llegar a MLflow; `/salud` y cada predicción responden `version_modelo`.

**Para volver a la versión anterior** (si la nueva resulta peor):

1. Mover el alias: `python -m servicio.promover <versión anterior>` (desde `psets/proyecto/`, con el servidor de MLflow levantado). La versión anterior sigue guardada en el registro: no hay que reentrenar nada.
2. Volver a exportar: `python servicio/exportar_modelo.py`.
3. Reconstruir y relanzar la imagen: `docker build -f servicio/Dockerfile -t servicio-ecobici .` y `docker run -p 8000:8000 servicio-ecobici`.
4. Comprobar en `/salud` que `version_modelo` es la versión anterior.

